import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import pdb
from fast_transformers.causal_product import CausalDotProduct

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
from contextlib import contextmanager

@contextmanager
def null_context():
    yield
class MultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))

    def forward(self, query=None, key=None, value=None, original=True, input_shape=None):
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                concat = self.head[N](query, key, value, original, input_shape)
                isFirst = False
            else:
                concat = torch.cat((concat, self.head[N](query, key, value)), -1)
        # output = self.out_conv(concat)
        output = concat
        return output
    

class RelationUnit(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.temp = 30
        self.WK = nn.Linear(feature_dim, key_feature_dim)  
        self.WV = nn.Linear(feature_dim, feature_dim)

        # ----------------------------
        # self.original = True


        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        
        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))

            if m.bias is not None:
                m.bias.data.zero_()
        

    def forward(self, query=None, key=None, value=None, original=True, input_shape=None):

        if original is True:
            # --------- orignal ---------------------------
            w_k = self.WK(key)
            w_k = F.normalize(w_k, p=2, dim=-1)
            w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1

            w_q = self.WK(query)
            w_q = F.normalize(w_q, p=2, dim=-1)
            w_q = w_q.permute(1,0,2) # Batch, Len_2, Dim

            dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1
            affinity = F.softmax(dot_prod*self.temp, dim=-1)

            w_v = value.permute(1,0,2) # Batch, Len_1, Dim
            output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
            output = output.permute(1,0,2)
            return output
        else:
            num_ = causal_numerator(query, key, value)
            num_ = F.softmax(num_,dim=1)# label102 'dim=1' best performance
            return num_

## inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v, chunk_size = 512, eps = 1e-6):
    last_k_cumsum = 0
    last_context_cumsum = 0
    outs = []
    try:
        for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim = -2), (q, k, v))):
            k_cumsum = last_k_cumsum + k.cumsum(dim=-2)

            D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q) + eps)
            context = torch.einsum('...nd,...ne->...nde', k, v)
            context_cumsum = context
            out = torch.einsum('...nde,...nd,...n->...ne', context_cumsum, q, D_inv)

            last_k_cumsum = k_cumsum[:, :, -1:]
            # last_context_cumsum = context_cumsum[:, :, -1:]
            outs.append(out)
    except Exception as e:
        print(e)
        return

    return torch.cat(outs, dim = -2)

# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, eps = 1e-6):
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'

    try:
        cuda_context = null_context #if not autocast_enabled else partial(autocast, enabled = False)

        causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

        k_cumsum = k.cumsum(dim=-2) + eps
        D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

        with cuda_context():
            if autocast_enabled:
                q, k, v = map(lambda t: t.float(), (q, k, v))

            out = causal_dot_product_fn(q, k, v)

        q, k, v = map(lambda t: t.float(), (q, k, v))
        out = causal_dot_product_fn(q, k, v)
        out = torch.einsum('...nd,...n->...nd', out, D_inv)
        return out
    except Exception as e:
        print(e)
        return

def causal_numerator(qs, ks, vs):

    """Computes not-normalized FAVOR causal attention A_{masked}V.

  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].

  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
    """

    result = []
    sums = torch.zeros_like(torch.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    for index in range(qs.shape[0]):
        k_index = F.normalize(ks[index])
        v_index = F.normalize(vs[index])
        sums = sums + torch.einsum("ijk,ijl->ijkl", k_index, v_index)
        # sums = sums + torch.einsum("ijk,ijl->ijkl", ks[index], vs[index])
        result.append(torch.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

    result = torch.cat(result, 0)

    return result#, grad

from functools import partial
class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention)
            except ImportError:
                print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn,
                                    projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix,
                                    device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out

# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out
