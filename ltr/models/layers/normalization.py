import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:            
            aa = torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True)
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (aa + self.eps)).sqrt())  # view
        else:
            return input * (self.scale / (torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())

class GroupNorm(nn.Module):
    # def __init__(self, num_features, num_groups=32, eps=1e-5):

    def __init__(self, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        # self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        # self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size() # 15, 512, 22, 22
        G = self.num_groups
        assert C % G == 0
        x = x.reshape(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.reshape(N,C,H,W)
        return x
