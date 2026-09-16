# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torch.utils.checkpoint import checkpoint
from typing import Dict, List

from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class FiLMLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lang_proj = nn.Linear(768, 2 * channels)

    def forward(self, x, lang_emb):
        beta, gamma = self.lang_proj(lang_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return F.relu((1 + gamma) * x + beta)


class FiLMIntermediateLayerGetter(IntermediateLayerGetter):
    def __init__(self, backbone, return_layers):
        super().__init__(backbone, return_layers)
        self.film_layers = nn.ModuleDict({name: nn.ModuleList([
            FiLMLayer(block.conv3.out_channels if hasattr(block, 'conv3') else block.conv2.out_channels)
            for block in layer]) for name, layer in self.items() if name.startswith('layer')})

    @staticmethod
    def conditioned_block(block, film, x, lang_emb):
        return film(block(x), lang_emb)

    def forward(self, x, lang_emb):
        if lang_emb is None or lang_emb.shape != (x.shape[0], 768):
            raise ValueError('CLIP FiLM requires language embeddings with shape (B, 768)')
        out = OrderedDict()
        for name, layer in self.items():
            if name == 'film_layers':
                continue
            if name in self.film_layers:
                for block, film in zip(layer, self.film_layers[name]):
                    if self.training and torch.is_grad_enabled():
                        x = checkpoint(self.conditioned_block, block, film, x, lang_emb, use_reentrant=False)
                    else:
                        x = self.conditioned_block(block, film, x, lang_emb)
            else:
                x = layer(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int,
                 return_interm_layers: bool, language_conditioning: str = 'none'):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        if language_conditioning not in ('none', 'clip_film'):
            raise ValueError(f'Unsupported language conditioning {language_conditioning}')
        self.language_conditioning = language_conditioning
        getter = FiLMIntermediateLayerGetter if language_conditioning == 'clip_film' else IntermediateLayerGetter
        self.body = getter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor, lang_emb=None):
        if self.language_conditioning == 'clip_film':
            return self.body(tensor, lang_emb)
        if lang_emb is not None:
            raise ValueError('Language embeddings require clip_film conditioning')
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, pretrained: bool = True, language_conditioning: str = 'none'):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained and is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, language_conditioning)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, lang_emb=None):
        xs = self[0](tensor_list, lang_emb=lang_emb)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation,
                        getattr(args, 'pretrained_backbone', True), getattr(args, 'language_conditioning', 'none'))
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
