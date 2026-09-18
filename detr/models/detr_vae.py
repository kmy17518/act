# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, action_dim=None,
                 mt_act_language_dim=None, camera_batch=False):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            mt_act_language_dim: text-embedding width for the RoboAgent MT-ACT variant (None: upstream ACT).
                MT-ACT projects the frozen text embedding with one learned `proj_text_emb` linear layer to the
                transformer width; that vector conditions the ResNet through FiLM (see backbone.py) and enters
                the transformer encoder as a third extra token next to the latent and proprioception tokens.
                Its CVAE style encoder sees [CLS, actions] only (no proprioception token), and `qpos` beyond the
                first `state_dim` entries (the adapter's one-hot task category) is ignored.
            camera_batch: run the shared backbone once over the images of all cameras (camera-major batch) instead
                of once per camera. Per-sample layers give the same result either way; trainable BatchNorm then
                computes its batch statistics over all cameras jointly, which is what its running statistics
                describe at evaluation time (per-camera passes normalize each camera by its own statistics, which
                eval mode cannot reproduce).
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.camera_batch = camera_batch
        self.transformer = transformer
        self.encoder = encoder
        # forward() consumes only the first stacked decoder output (`hs[0]`), so the remaining decoder
        # layers never influence predictions or gradients. `decoder_layers_used = 1` skips computing
        # them; None runs every layer as upstream does. Plain attribute: not saved, not architecture.
        self.decoder_layers_used = None
        self.mt_act = mt_act_language_dim is not None
        self.state_dim = state_dim
        hidden_dim = transformer.d_model
        action_dim = state_dim if action_dim is None else action_dim
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        if self.mt_act:
            # RoboAgent: encoder_proj over actions only, pos_table(num_queries + 1)
            self.register_buffer('pos_table', get_sinusoid_encoding_table(1+num_queries, hidden_dim)) # [CLS], a_seq
        else:
            self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
            self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        if self.mt_act:
            self.proj_text_emb = nn.Linear(mt_act_language_dim, hidden_dim) # project text embedding to hidden_dim
            self.additional_pos_embed = nn.Embedding(3, hidden_dim) # learned position embedding for proprio, latent and text
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, lang_emb=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        task_emb = None
        if self.mt_act:
            if lang_emb is None:
                raise ValueError('MT-ACT requires language embeddings')
            if qpos.shape[1] < self.state_dim:
                raise ValueError(f'MT-ACT expects at least {self.state_dim} proprioception values')
            qpos = qpos[:, :self.state_dim]  # the adapter's trailing one-hot task category is not an input
            task_emb = self.proj_text_emb(lang_emb)
            lang_emb = task_emb  # FiLM is generated from the projected embedding
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            if self.mt_act:
                encoder_input = torch.cat([cls_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                cls_joint_is_pad = torch.full((bs, 1), False, device=qpos.device) # False: not a padding
            else:
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+2, hidden_dim)
                cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device) # False: not a padding
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            with torch.autocast(device_type=encoder_output.device.type, enabled=False):
                # Latent distribution, KL inputs and the reparametrized sample stay fp32 under autocast.
                latent_info = self.latent_proj(encoder_output.float())
                mu = latent_info[:, :self.latent_dim]
                logvar = latent_info[:, self.latent_dim:]
                latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            ncam = len(self.camera_names)
            if self.camera_batch:
                # One backbone pass over every camera: (bs, ncam, 3, H, W) -> (ncam * bs, 3, H, W), camera-major so
                # each camera's slice stays the dense block the loader produced.
                flat = image[:, :ncam].transpose(0, 1).reshape(ncam * bs, *image.shape[2:])
                features, pos = self.backbones[0](flat, lang_emb=None if lang_emb is None else lang_emb.repeat(ncam, 1))
                features = self.input_proj(features[0]) # (ncam * bs, hidden, h, w)
                # fold camera dimension into width dimension, in camera order (same layout as the per-camera cat)
                src = features.view(ncam, bs, *features.shape[1:]).permute(1, 2, 3, 0, 4).reshape(
                    bs, features.shape[1], features.shape[2], ncam * features.shape[3])
                pos = pos[0]
                pos = torch.cat([pos if pos.shape[0] == 1 else pos[:bs]] * ncam, axis=3)
            else:
                all_cam_features = []
                all_cam_pos = []
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, pos = self.backbones[0](image[:, cam_id], lang_emb=lang_emb) # HARDCODED
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
                # fold camera dimension into width dimension
                src = torch.cat(all_cam_features, axis=3)
                pos = torch.cat(all_cam_pos, axis=3)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight,
                                  decoder_layers=self.decoder_layers_used, task_emb=task_emb)[0]
        else:
            if self.mt_act:
                raise ValueError('MT-ACT requires image backbones')
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight,
                                  decoder_layers=self.decoder_layers_used)[0]
        with torch.autocast(device_type=hs.device.type, enabled=False):
            # Output heads (and therefore the L1 loss inputs) stay fp32 under autocast.
            hs = hs.float()
            a_hat = self.action_head(hs)
            is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

    def unused_parameters(self):
        """Parameters of decoder layers that forward() never consumes (their autograd gradient is exactly zero)."""
        if self.decoder_layers_used is None:
            return []
        return [p for layer in self.transformer.decoder.layers[self.decoder_layers_used:] for p in layer.parameters()]



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names, action_dim=None,
                 image_size=(480, 640), backbone_stride=32):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        action_dim = state_dim if action_dim is None else action_dim
        self.image_size = tuple(image_size)
        feature_height, feature_width = [(size + backbone_stride - 1) // backbone_stride - 12
                                         for size in self.image_size]
        if min(feature_height, feature_width) < 1:
            raise ValueError('CNNMLP images must yield at least 13x13 backbone features '
                             '(minimum 385x385 with stride 32); use --image-size 480 640')
        # Retained for strict loading of upstream CNNMLP checkpoints; forward uses self.mlp.
        self.action_head = nn.Linear(1000, action_dim)
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 32 * feature_height * feature_width * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        if tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(f'CNNMLP expects image size {self.image_size}, got {tuple(image.shape[-2:])}')
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)
        features = torch.cat([flattened_features, qpos], axis=1)
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = getattr(args, 'state_dim', 14)

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    mt_act = getattr(args, 'language_conditioning', 'none') == 'mt_act'
    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        action_dim=getattr(args, 'action_dim', 14),
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        mt_act_language_dim=getattr(args, 'language_dim', 768) if mt_act else None,
        camera_batch=bool(getattr(args, 'camera_batch', False)),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    if getattr(args, 'language_conditioning', 'none') != 'none':
        raise ValueError('CLIP FiLM language conditioning is only supported for ACT, not CNNMLP')
    state_dim = getattr(args, 'state_dim', 14)

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
        action_dim=getattr(args, 'action_dim', 14),
        image_size=getattr(args, 'image_size', (480, 640)),
        backbone_stride=16 if args.dilation else 32,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

