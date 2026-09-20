# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from functools import partial

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
        # Keep the affine in the activation dtype (no-op in fp32) so autocast bf16 activations are not
        # promoted back to fp32 here; the frozen statistics stay in fp32.
        return x * scale.to(x.dtype) + bias.to(x.dtype)


FILM_INITS = ('random', 'identity')
BACKBONE_NORMS = ('frozen', 'batch', 'batch_per_camera')


class PairedConv2d(nn.Conv2d):
    """Stem convolution accepting a camera image channel-stacked with its goal image (goal-image early fusion).

    For a 6-channel input `[current; goal]` it computes `conv(current) + conv_goal(goal)`, which is exactly one
    6-channel convolution with weights `[W_obs, W_goal]` (BridgeData V2's channel stacking for ACT). `W_goal`
    (`goal_weight`) starts at zero and `W_obs` keeps the stem's (ImageNet) weights, so at initialization the
    paired stem computes what the unpaired stem computes and a zero goal image is an exact "absent goal". A
    3-channel input (other cameras, goal-free passes) runs the plain convolution. State-dict keys stay
    `weight`/`bias`; `goal_weight` is the only addition. The zero initialization is our controlled engineering
    choice, not a claim about BridgeData's unpublished details.
    """
    def __init__(self, conv):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                         conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None:
                self.bias.copy_(conv.bias)
        self.goal_weight = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x):
        if x.shape[1] == self.in_channels:
            return super().forward(x)
        if x.shape[1] != 2 * self.in_channels:
            raise ValueError(f'PairedConv2d expects {self.in_channels} or {2 * self.in_channels} input channels, '
                             f'got {x.shape[1]}')
        weight = torch.cat([self.weight, self.goal_weight.to(self.weight.dtype)], dim=1)
        return self._conv_forward(x, weight, self.bias)


def pair_stem(body):
    """Replace a ResNet body's `conv1` with a PairedConv2d (idempotent); returns the paired stem."""
    if not isinstance(body.conv1, PairedConv2d):
        body.conv1 = PairedConv2d(body.conv1)
    return body.conv1


class PerCameraBatchNorm2d(nn.Module):
    """BatchNorm2d with one set of running statistics per camera (domain-specific BatchNorm).

    A shared backbone that sees one camera per pass normalizes every camera by its own batch statistics in
    training; a single running mean/variance can only describe the mixture of the cameras, so eval mode
    normalizes differently from anything training saw. This layer keeps that training computation exactly
    (batch statistics of the current pass, shared affine weight/bias) but tracks running statistics per
    camera: `camera` -- set through `BackboneBase.select_camera` before each pass -- names the set that the
    pass updates in training and normalizes with in eval. State dict: `running_mean_<c>`, `running_var_<c>`,
    `num_batches_tracked_<c>` per camera; a state dict with a single `running_mean`/`running_var`/
    `num_batches_tracked` (nn.BatchNorm2d: ImageNet weights or a `--backbone-norm batch` checkpoint) loads by
    copying those statistics to every camera (see scripts/b1k/recalibrate_camera_batchnorm.py to re-estimate them).
    """
    def __init__(self, num_features, num_cameras, eps=1e-5, momentum=0.1):
        super().__init__()
        if num_cameras < 1:
            raise ValueError('PerCameraBatchNorm2d needs at least one camera')
        self.num_features = num_features
        self.num_cameras = num_cameras
        self.eps = eps
        self.momentum = momentum  # None: cumulative average, as nn.BatchNorm2d
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        for camera in range(num_cameras):
            self.register_buffer(f'running_mean_{camera}', torch.zeros(num_features))
            self.register_buffer(f'running_var_{camera}', torch.ones(num_features))
            self.register_buffer(f'num_batches_tracked_{camera}', torch.tensor(0, dtype=torch.long))
        self.camera = 0

    def statistics(self, camera=None):
        camera = self.camera if camera is None else camera
        if not 0 <= camera < self.num_cameras:
            raise ValueError(f'Camera {camera} outside the {self.num_cameras} tracked cameras')
        return (getattr(self, f'running_mean_{camera}'), getattr(self, f'running_var_{camera}'),
                getattr(self, f'num_batches_tracked_{camera}'))

    def reset_running_stats(self, camera=None):
        for index in range(self.num_cameras) if camera is None else (camera,):
            mean, var, tracked = self.statistics(index)
            mean.zero_()
            var.fill_(1)
            tracked.zero_()

    def forward(self, x):
        mean, var, tracked = self.statistics()
        if self.training:
            tracked.add_(1)
            factor = 1.0 / float(tracked) if self.momentum is None else self.momentum
        else:
            factor = 0.0
        return F.batch_norm(x, mean, var, self.weight, self.bias, self.training, factor, self.eps)

    def extra_repr(self):
        return f'{self.num_features}, num_cameras={self.num_cameras}, eps={self.eps}, momentum={self.momentum}'

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if prefix + 'running_mean' in state_dict and prefix + 'running_mean_0' not in state_dict:
            for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                value = state_dict.pop(prefix + name, None)
                if value is None:
                    continue
                for camera in range(self.num_cameras):
                    state_dict[f'{prefix}{name}_{camera}'] = value.clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class FiLMLayer(nn.Module):
    """Per-block FiLM: relu((1 + gamma) * x + beta) with (beta, gamma) projected from the language embedding.

    `init='identity'` zeroes the projection so beta = gamma = 0 and the layer starts as the identity on the
    (already non-negative) residual-block output; `init='random'` keeps nn.Linear's default initialization.
    """
    def __init__(self, channels, init='random', lang_dim=768):
        super().__init__()
        if init not in FILM_INITS:
            raise ValueError(f'Unsupported FiLM initialization {init!r}; expected one of {FILM_INITS}')
        self.lang_proj = nn.Linear(lang_dim, 2 * channels)
        if init == 'identity':
            nn.init.zeros_(self.lang_proj.weight)
            nn.init.zeros_(self.lang_proj.bias)

    def forward(self, x, lang_emb):
        beta, gamma = self.lang_proj(lang_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return F.relu((1 + gamma) * x + beta)


class FiLMIntermediateLayerGetter(IntermediateLayerGetter):
    def __init__(self, backbone, return_layers, film_init='random', lang_dim=768):
        super().__init__(backbone, return_layers)
        self.lang_dim = lang_dim
        self.film_layers = nn.ModuleDict({name: nn.ModuleList([
            FiLMLayer(block.conv3.out_channels if hasattr(block, 'conv3') else block.conv2.out_channels, film_init,
                      lang_dim)
            for block in layer]) for name, layer in self.items() if name.startswith('layer')})
        # Runtime option (not architecture/state): recompute each conditioned residual block's activations
        # during backward instead of storing them (same math, less memory, extra forward work).
        self.recompute = True

    @staticmethod
    def conditioned_block(block, film, x, lang_emb):
        return film(block(x), lang_emb)

    def forward(self, x, lang_emb):
        # lang_emb=None is the explicit identity pass (gamma = beta = 0, i.e. relu(x) = x on the block outputs):
        # used for goal images encoded without language modulation (language_on_goal_encoder=False).
        if lang_emb is not None and lang_emb.shape != (x.shape[0], self.lang_dim):
            raise ValueError(f'CLIP FiLM requires language embeddings with shape (B, {self.lang_dim})')
        recompute = self.recompute and self.training and torch.is_grad_enabled()
        out = OrderedDict()
        for name, layer in self.items():
            if name == 'film_layers':
                continue
            if name in self.film_layers and lang_emb is not None:
                for block, film in zip(layer, self.film_layers[name]):
                    if recompute:
                        x = checkpoint(self.conditioned_block, block, film, x, lang_emb, use_reentrant=False)
                    else:
                        x = self.conditioned_block(block, film, x, lang_emb)
            else:
                x = layer(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out


MT_ACT_FILM_STAGES = ('layer2', 'layer3', 'layer4')  # RoboAgent film_config['use_in_layers'] = [1, 2, 3]


class ResidualFiLMBody(nn.Module):
    """RoboAgent MT-ACT visual encoder: FiLM inside the residual branch of the selected ResNet stages.

    Follows robopen/roboagent `detr/models/resnet_film.py`: one `Linear(cond_dim, num_blocks * 2 * planes)`
    per FiLM stage produces (gamma, beta) for each BasicBlock, applied after `bn2` and before the skip
    connection is added and the block's ReLU runs: `relu(identity + (1 + gamma) * bn2(conv2(...)) + beta)`.
    Stage 1 and the stem are unmodulated. `cond` is the *projected* task embedding (the transformer width),
    shared with the language token in `DETRVAE`. Keeps the torchvision module names (`conv1`, `bn1`,
    `layer1`...) so ImageNet weights and `--fast-maxpool` apply unchanged; only returns the last stage.
    """
    def __init__(self, backbone, cond_dim, stages=MT_ACT_FILM_STAGES, film_init='random'):
        super().__init__()
        if film_init not in FILM_INITS:
            raise ValueError(f'Unsupported FiLM initialization {film_init!r}; expected one of {FILM_INITS}')
        for name in ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'):
            setattr(self, name, getattr(backbone, name))
        self.cond_dim = cond_dim
        self.film_generators = nn.ModuleDict()
        for name in stages:
            layer = getattr(self, name)
            if any(hasattr(block, 'conv3') for block in layer):
                raise ValueError('MT-ACT FiLM is implemented for BasicBlock ResNets (resnet18/34) only')
            planes = layer[0].conv2.out_channels
            generator = nn.Linear(cond_dim, len(layer) * 2 * planes)
            if film_init == 'identity':
                nn.init.zeros_(generator.weight)
                nn.init.zeros_(generator.bias)
            self.film_generators[name] = generator

    @staticmethod
    def film_block(block, x, gamma, beta):
        identity = x
        out = block.relu(block.bn1(block.conv1(x)))
        out = block.bn2(block.conv2(out))
        out = (1 + gamma[:, :, None, None]) * out + beta[:, :, None, None]
        if block.downsample is not None:
            identity = block.downsample(x)
        return block.relu(out + identity)

    def forward(self, x, cond):
        # cond=None is the explicit identity pass (gamma = beta = 0 in every modulated block), used for goal
        # images encoded without language modulation (language_on_goal_encoder=False).
        if cond is not None and cond.shape != (x.shape[0], self.cond_dim):
            raise ValueError(f'MT-ACT FiLM requires projected task embeddings with shape (B, {self.cond_dim})')
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for name in ('layer1', 'layer2', 'layer3', 'layer4'):
            layer = getattr(self, name)
            if name not in self.film_generators or cond is None:
                x = layer(x)
                continue
            # RoboAgent: film_feat.view(-1, 2, num_blocks, planes) -> gamma first, beta second
            film = self.film_generators[name](cond).view(x.shape[0], 2, len(layer), -1).to(x.dtype)
            for index, block in enumerate(layer):
                x = self.film_block(block, x, film[:, 0, index], film[:, 1, index])
        return OrderedDict([('0', x)])


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int,
                 return_interm_layers: bool, language_conditioning: str = 'none', film_init: str = 'random',
                 lang_dim: int = 768, film_cond_dim: int = 512):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        if language_conditioning not in ('none', 'clip_film', 'mt_act'):
            raise ValueError(f'Unsupported language conditioning {language_conditioning}')
        self.language_conditioning = language_conditioning
        if language_conditioning == 'clip_film':
            self.body = FiLMIntermediateLayerGetter(backbone, return_layers=return_layers, film_init=film_init,
                                                    lang_dim=lang_dim)
        elif language_conditioning == 'mt_act':
            if return_interm_layers:
                raise ValueError('MT-ACT FiLM returns the last stage only')
            self.body = ResidualFiLMBody(backbone, film_cond_dim, film_init=film_init)
        else:
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        # Plain list (not registered): the per-camera BatchNorm layers whose statistics set select_camera switches.
        self.per_camera_norms = [module for module in self.body.modules() if isinstance(module, PerCameraBatchNorm2d)]

    def select_camera(self, camera):
        """Route the next pass to `camera`'s BatchNorm statistics (no-op without per-camera normalization)."""
        for module in self.per_camera_norms:
            module.camera = camera

    def pair_stem(self):
        """Goal-image early fusion: make the stem accept `[current; goal]` 6-channel inputs (see PairedConv2d)."""
        return pair_stem(self.body)

    def forward(self, tensor, lang_emb=None, film_identity=False):
        if self.language_conditioning != 'none':
            if lang_emb is None and not film_identity:
                raise ValueError(f'{self.language_conditioning} requires language embeddings; a deliberate identity '
                                 'pass (gamma = beta = 0, e.g. goal images without language) must set film_identity')
            # film_identity: the FiLM body runs as the identity (goal images encoded without language).
            return self.body(tensor, lang_emb)
        if lang_emb is not None:
            raise ValueError('Language embeddings require clip_film or mt_act conditioning')
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
    """ResNet backbone with frozen BatchNorm (upstream ACT), regular BatchNorm (`norm='batch'`, as MT-ACT) or
    BatchNorm with per-camera running statistics (`norm='batch_per_camera'`, see PerCameraBatchNorm2d)."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool, pretrained: bool = True, language_conditioning: str = 'none',
                 film_init: str = 'random', lang_dim: int = 768, film_cond_dim: int = 512, norm: str = 'frozen',
                 num_cameras: int = 1):
        if norm not in BACKBONE_NORMS:
            raise ValueError(f'Unsupported backbone normalization {norm!r}; expected one of {BACKBONE_NORMS}')
        if norm == 'frozen':
            norm_layer = FrozenBatchNorm2d
        elif norm == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = partial(PerCameraBatchNorm2d, num_cameras=num_cameras)
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained and is_main_process(),
            norm_layer=norm_layer) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, language_conditioning, film_init,
                         lang_dim, film_cond_dim)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        # Runtime option (not architecture/state): run only the convolutional body under autocast with
        # this dtype and hand fp32 features to the rest of the network. None keeps the caller's precision.
        self.body_autocast_dtype = None

    def forward(self, tensor_list: NestedTensor, lang_emb=None, camera=None, film_identity=False):
        if camera is not None:
            self[0].select_camera(camera)
        elif self[0].per_camera_norms:
            raise ValueError('Per-camera BatchNorm statistics need the camera index of this pass')
        if self.body_autocast_dtype is None:
            xs = self[0](tensor_list, lang_emb=lang_emb, film_identity=film_identity)
        else:
            with torch.autocast(device_type=tensor_list.device.type, dtype=self.body_autocast_dtype):
                xs = self[0](tensor_list, lang_emb=lang_emb, film_identity=film_identity)
            xs = {name: x.float() for name, x in xs.items()}
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
                        getattr(args, 'pretrained_backbone', True), getattr(args, 'language_conditioning', 'none'),
                        getattr(args, 'film_init', 'random'), getattr(args, 'language_dim', 768),
                        args.hidden_dim, getattr(args, 'backbone_norm', 'frozen'), len(args.camera_names))
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
