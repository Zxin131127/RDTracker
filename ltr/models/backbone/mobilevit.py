from torch import nn, Tensor

import argparse
from typing import Dict, Tuple, Optional

# from utils import logger
from .init_utils import initialize_weights
import torch

wt_loc = "/data/zxin/ml-cvnets-main/results_coco_mobilevit_s/checkpoint_ema.pt"
wts = torch.load(wt_loc, map_location="cpu")


class BaseEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseEncoder, self).__init__()
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        self.round_nearest = 8

        self.model_conf_dict = dict()

    @classmethod
    def check_model(self):
        assert self.model_conf_dict, "Model configuration dictionary should not be empty"
        assert self.conv_1 is not None, 'Please implement self.conv_1'
        assert self.layer_1 is not None, 'Please implement self.layer_1'
        assert self.layer_2 is not None, 'Please implement self.layer_2'
        assert self.layer_3 is not None, 'Please implement self.layer_3'
        assert self.layer_4 is not None, 'Please implement self.layer_4'
        assert self.layer_5 is not None, 'Please implement self.layer_5'
        assert self.conv_1x1_exp is not None, 'Please implement self.conv_1x1_exp'
        assert self.classifier is not None, 'Please implement self.classifier'

    def reset_parameters(self, opts):
        initialize_weights(opts=opts, modules=self.modules())

    def extract_end_points_all(self, x: Tensor, use_l5: Optional[bool] = True, use_l5_exp: Optional[bool] = False) -> Dict:
        out_dict = {} # Use dictionary over NamedTuple so that JIT is happy
        x = self.conv_1(x)  # 112 x112
        x = self.layer_1(x)  # 112 x112
        out_dict["out_l1"] = x

        x = self.layer_2(x)  # 56 x 56
        out_dict["out_l2"] = x

        x = self.layer_3(x)  # 28 x 28
        out_dict["out_l3"] = x

        x = self.layer_4(x)  # 14 x 14
        out_dict["out_l4"] = x

        if use_l5:
            x = self.layer_5(x)  # 7 x 7
            out_dict["out_l5"] = x

            if use_l5_exp:
                x = self.conv_1x1_exp(x)
                out_dict["out_l5_exp"] = x
        return out_dict

    def extract_end_points_l4(self, x: Tensor) -> Dict:
        return self.extract_end_points_all(x, use_l5=False)

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.extract_features(x)
        x = self.classifier(x)
        return x


def get_configuration(opts) -> Dict:
    mode = getattr(opts, "model.classification.mit.mode", "small")

    head_dim = getattr(opts, "model.classification.mit.head_dim", None)
    # num_heads = getattr(opts, "model.classification.mit.number_heads", None)
    num_heads = getattr(opts, "model.classification.mit.number_heads", 4)

    mode = mode.lower()
    if mode == "small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2"
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2"
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit"
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit"
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "head_dim": head_dim,
                "num_heads": num_heads,
                "block_type": "mobilevit"
            },
            "last_layer_exp_factor": 4
        }
    else:
        raise NotImplementedError
    return config

class MobileViT(BaseEncoder):
    """
        MobileViT: https://arxiv.org/abs/2110.02178?context=cs.LG
    """
    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration(opts=opts)

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the classification backbones
        # We allow that using `output_stride` arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        super(MobileViT, self).__init__()
        self.dilation = 1

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
                opts=opts, in_channels=image_channels, out_channels=out_channels,
                kernel_size=3, stride=2, use_norm=True, use_act=True
            )

        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict['layer1'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict['layer2'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict['layer3'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer4"], dilate=dilate_l4
        )
        self.model_conf_dict['layer4'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer5"], dilate=dilate_l5
        )
        self.model_conf_dict['layer5'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
                opts=opts, in_channels=in_channels, out_channels=exp_channels,
                kernel_size=1, stride=1, use_act=True, use_norm=True
            )

        self.model_conf_dict['exp_before_cls'] = {'in': in_channels, 'out': exp_channels}

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False))
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="dropout", module=Dropout(p=classifier_dropout, inplace=True))
        self.classifier.add_module(
            name="fc",
            module=LinearLayer(in_features=exp_channels, out_features=num_classes, bias=True)
        )

        # # check model
        # self.check_model()
        #
        # # weight initialization
        # self.reset_parameters(opts=opts)

    @classmethod


    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads


        block.append(
            MobileViTBlock(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.classification.mit.attn_dropout", 0.1),
                head_dim=head_dim,
                no_fusion=getattr(opts, "model.classification.mit.no_fuse_local_global_features", False),
                conv_ksize=getattr(opts, "model.classification.mit.conv_kernel_size", 3)
            )
        )

        return nn.Sequential(*block), input_channel

class BaseLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseLayer, self).__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(self, *args, **kwargs) ->  Tensor or Tuple[Tensor]:
        pass

    def profile_module(self, *args, **kwargs) -> (Tensor, float, float):
        raise NotImplementedError

class LinearLayer(BaseLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: Optional[bool] = True,
                 *args, **kwargs) -> None:
        """
            Applies a linear transformation to the input data

            :param in_features: size of each input sample
            :param out_features:  size of each output sample
            :param bias: Add bias (learnable) or not
        """
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--model.layer.linear-init', type=str, default='xavier_uniform',
                            help='Init type for linear layers')
        parser.add_argument('--model.layer.linear-init-std-dev', type=float, default=0.01,
                            help='Std deviation for Linear layers')
        return parser

    def reset_params(self):
        if self.weight is not None:
            torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.bias is not None and x.dim() == 2:
            x = torch.addmm(self.bias, x, self.weight.t())
        else:
            x = x.matmul(self.weight.t())
            if self.bias is not None:
                x += self.bias
        return x

    def __repr__(self):
        repr_str = "{}(in_features={}, out_features={}, bias={})".format(
            self.__class__.__name__, self.in_features,
            self.out_features,
            True if self.bias is not None else False
        )
        return repr_str

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        out_size = list(input.shape)
        out_size[-1] = self.out_features
        params = sum([p.numel() for p in self.parameters()])
        macs = params
        output = torch.zeros(size=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs


class ConvLayer(BaseLayer):
    def __init__(self, opts, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 stride: Optional[int or tuple] = 1,
                 dilation: Optional[int or tuple] = 1, groups: Optional[int] = 1,
                 bias: Optional[bool] = False, padding_mode: Optional[str] = 'zeros',
                 use_norm: Optional[bool] = True, use_act: Optional[bool] = True,
                 *args, **kwargs
                 ) -> None:
        """
            Applies a 2D convolution over an input signal composed of several input planes.
            :param opts: arguments
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param kernel_size: kernel size
            :param stride: move the kernel by this amount during convolution operation
            :param dilation: Add zeros between kernel elements to increase the effective receptive field of the kernel.
            :param groups: Number of groups. If groups=in_channels=out_channels, then it is a depth-wise convolution
            :param bias: Add bias or not
            :param padding_mode: Padding mode. Default is zeros
            :param use_norm: Use normalization layer after convolution layer or not. Default is True.
            :param use_act: Use activation layer after convolution layer/convolution layer followed by batch
            normalization or not. Default is True.
        """
        super(ConvLayer, self).__init__()

        if use_norm:
            assert not bias, 'Do not use bias when using normalization layers.'

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])

        if in_channels % groups != 0:
            logger.error('Input channels are not divisible by groups. {}%{} != 0 '.format(in_channels, groups))
        if out_channels % groups != 0:
            logger.error('Output channels are not divisible by groups. {}%{} != 0 '.format(out_channels, groups))

        block = nn.Sequential()

        conv_layer = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                            padding_mode=padding_mode)

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "prelu")

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(act_type=act_type,
                                          inplace=inplace,
                                          negative_slope=neg_slope,
                                          num_parameters=out_channels)
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

