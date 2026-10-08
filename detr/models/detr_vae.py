# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
from contextlib import nullcontext
import copy

import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed

# Goal-image conditioning (see b1k.md "Goal-image conditioning"):
#   none  -- the unchanged policy;
#   early -- the goal image is channel-stacked with its camera's current image before the ResNet stem
#            (BridgeData V2's ACT conditioning; PairedConv2d in backbone.py, goal half zero-initialised);
#   late  -- the goal image runs through the RGB backbone and its projected spatial feature grid is appended to
#            the transformer encoder memory after the current-view tokens, with the spatial position encoding kept
#            and a learned per-view goal identity added to it (zero-initialised).
GOAL_FUSIONS = ('none', 'early', 'late')
# Late fusion's goal encoder: the current RGB backbone, a trainable copy of it, or a copy frozen at its
# initialization (the ImageNet weights) whose forward pass runs without gradients.
GOAL_ENCODERS = ('shared_base', 'separate_base', 'frozen')
# Late fusion's goal tokens: the full projected feature grid, or its spatial mean as one token per view.
GOAL_TOKENS = ('grid', 'pooled')
# Late fusion: the goal tokens' position codes (the camera grid's sine code, or none: the goal tag only), their
# content (the projected goal features, or their difference to the projected current features of the same camera),
# and where they enter (the encoder memory, the decoder's cross-attention memory only, or the action queries).
GOAL_POSITIONS = ('sine', 'none')
GOAL_CONTENTS = ('goal', 'diff')
GOAL_ENTRIES = ('encoder', 'decoder', 'queries')


def parse_goal_tag_init(spec):
    """Canonical `--goal-tag-init` value: 'zero' or 'normal:STD' (STD a positive finite float)."""
    spec = str(spec).strip()
    if spec == 'zero':
        return spec
    if spec.startswith('normal:'):
        try:
            std = float(spec[7:])
        except ValueError:
            std = float('nan')
        if 0 < std < float('inf'):
            return f'normal:{std:g}'
    raise ValueError(f'goal tag init must be zero or normal:STD with STD > 0, got {spec!r}')


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
                 mt_act_language_dim=None, camera_batch=False, goal_fusion='none', goal_views=(),
                 goal_role_embedding=True, goal_encoder='shared_base', language_on_goal_encoder=False,
                 goal_tag_init='zero', goal_stem_init='zero', goal_tokens='grid', goal_pos='sine', goal_content='goal',
                 goal_entry='encoder', goal_stem_gain=None, goal_stem_diff=False, goal_fusion_depth=0,
                 goal_detach_views=()):
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
            goal_fusion / goal_views / goal_role_embedding / goal_encoder / language_on_goal_encoder: goal-image
                conditioning (GOAL_FUSIONS). `goal_views` are camera indices into `camera_names` whose goal image is
                supplied, in the order of the `goal` tensor passed to forward ((B, len(goal_views), 3, H, W), same
                normalization as `image`). `goal_encoder` (late fusion): `shared_base` runs the goal through the
                current RGB backbone, `separate_base` through an architecturally identical copy. With language
                conditioning, `language_on_goal_encoder=False` (default) encodes the goal with the FiLM layers held
                at the identity (gamma = beta = 0); True modulates the goal pass with the same language embedding.
                `goal_encoder='frozen'` encodes the goal with a copy of the backbone frozen at its initialization
                (the ImageNet weights; no gradients, eval mode); only the shared `input_proj` and the goal tag
                learn on the goal path.
                All goal modules are created after every base module, so with a fixed seed the base parameters are
                initialised identically with and without goal conditioning.
            goal_tag_init: late fusion's goal identity (`goal_role_embed`) start, 'zero' or 'normal:STD'.
            goal_stem_init: early fusion's goal half of the paired stem, 'zero', 'random' or 'copy:SCALE'
                (see backbone.PairedConv2d).
            goal_tokens: late fusion's goal tokens, 'grid' (the projected feature grid with the spatial positions)
                or 'pooled' (its spatial mean, one token per view with no spatial position, appended after every
                camera token).
            goal_pos: late fusion's goal-token position codes, 'sine' (the camera grid's) or 'none' (goal tag only).
            goal_content: late fusion's goal-token content, 'goal' (projected goal features) or 'diff' (projected goal
                features minus the projected current features of the goal view's camera, location by location).
            goal_entry: where late fusion's goal tokens enter: 'encoder' (encoder memory), 'decoder' (appended to the
                encoder output only: the decoder's cross-attention reads them, the encoder never processes them) or
                'queries' (the pooled goal token of every view, summed, is added to every action query's position
                embedding; requires goal_tokens='pooled' and no goal tag).
            goal_stem_gain / goal_stem_diff: early fusion's learned per-filter goal gain (its start) and explicit
                difference channel (see backbone.PairedConv2d).
            goal_fusion_depth: early fusion's merge point: 0 pairs the goal with its camera in the stem (PairedConv2d);
                K = 1-3 merges the goal's stage-K features into the current ones through a goal-only 1x1 convolution
                whose start follows `goal_stem_init` (see backbone.BackboneBase.enable_goal_merge).
            goal_detach_views: goal views (camera indices) whose paired-stem passes do not train the goal half of the
                stem (backbone.PairedConv2d.detach_goal); the forward pass is unchanged. Early fusion through the
                paired stem with one backbone pass per camera only.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.camera_batch = camera_batch
        if goal_fusion not in GOAL_FUSIONS:
            raise ValueError(f'goal_fusion must be one of {GOAL_FUSIONS}, got {goal_fusion!r}')
        if goal_encoder not in GOAL_ENCODERS:
            raise ValueError(f'goal_encoder must be one of {GOAL_ENCODERS}, got {goal_encoder!r}')
        if goal_tokens not in GOAL_TOKENS:
            raise ValueError(f'goal_tokens must be one of {GOAL_TOKENS}, got {goal_tokens!r}')
        goal_tag_init = parse_goal_tag_init(goal_tag_init)
        for value, choices, label in [(goal_pos, GOAL_POSITIONS, 'goal_pos'), (goal_content, GOAL_CONTENTS, 'goal_content'),
                                      (goal_entry, GOAL_ENTRIES, 'goal_entry')]:
            if value not in choices:
                raise ValueError(f'{label} must be one of {choices}, got {value!r}')
        late_options = (goal_pos, goal_content, goal_entry) != ('sine', 'goal', 'encoder')
        early_options = goal_stem_gain is not None or bool(goal_stem_diff)
        goal_fusion_depth = int(goal_fusion_depth)
        if goal_fusion_depth not in (0, 1, 2, 3):
            raise ValueError(f'goal_fusion_depth must be 0, 1, 2 or 3, got {goal_fusion_depth}')
        if goal_fusion_depth and goal_fusion != 'early':
            raise ValueError('goal_fusion_depth applies to early fusion only')
        if goal_fusion_depth and early_options:
            raise ValueError('goal_stem_gain / goal_stem_diff modify the paired stem (goal_fusion_depth 0)')
        if goal_fusion_depth and camera_batch:
            raise ValueError('goal_fusion_depth >= 1 needs one backbone pass per camera (no camera_batch)')
        self.goal_fusion_depth = goal_fusion_depth
        if late_options and goal_fusion != 'late':
            raise ValueError('goal_pos / goal_content / goal_entry apply to late fusion only')
        if early_options and goal_fusion != 'early':
            raise ValueError('goal_stem_gain / goal_stem_diff apply to early fusion only')
        if goal_entry == 'queries' and (goal_tokens != 'pooled' or goal_role_embedding):
            raise ValueError("goal_entry='queries' adds pooled goal tokens to the action queries: it requires "
                             "goal_tokens='pooled' and no goal role embedding")
        self.goal_fusion = goal_fusion
        self.goal_views = tuple(int(view) for view in goal_views) if goal_fusion != 'none' else ()
        self.goal_detach_views = tuple(int(view) for view in goal_detach_views)
        if self.goal_detach_views:
            if goal_fusion != 'early' or goal_fusion_depth or camera_batch:
                raise ValueError('goal_detach_views need early fusion through the paired stem (goal_fusion_depth 0) '
                                 'with one backbone pass per camera (no camera_batch)')
            if len(set(self.goal_detach_views)) != len(self.goal_detach_views) or \
                    not set(self.goal_detach_views) <= set(self.goal_views):
                raise ValueError(f'goal_detach_views {goal_detach_views} must be distinct goal views {self.goal_views}')
        self.goal_encoder = goal_encoder
        self.goal_tokens = goal_tokens
        self.goal_pos, self.goal_content, self.goal_entry = goal_pos, goal_content, goal_entry
        self.language_on_goal_encoder = bool(language_on_goal_encoder)
        self.goal_backbone = None
        self.goal_role_embed = None
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

        # goal-image conditioning (created last: see the docstring)
        if goal_fusion != 'none':
            if backbones is None:
                raise ValueError('Goal-image conditioning requires image backbones')
            if not self.goal_views or len(set(self.goal_views)) != len(self.goal_views) or \
                    any(not 0 <= view < len(camera_names) for view in self.goal_views):
                raise ValueError(f'goal_views must be distinct camera indices below {len(camera_names)}, got {goal_views}')
        if goal_fusion == 'early':
            if goal_encoder != 'shared_base':
                raise ValueError('Early fusion pairs the goal with its camera inside the shared stem; use goal_encoder=shared_base')
            if goal_tokens != 'grid':
                raise ValueError('goal_tokens applies to late fusion only')
            if goal_fusion_depth:
                self.backbones[0][0].enable_goal_merge(goal_fusion_depth, goal_stem_init)
            else:
                self.backbones[0][0].pair_stem(goal_stem_init, goal_stem_gain, bool(goal_stem_diff))
        elif goal_fusion == 'late':
            if goal_stem_init != 'zero':
                raise ValueError('goal_stem_init applies to early fusion only')
            if goal_encoder == 'frozen' and self.language_on_goal_encoder:
                raise ValueError('A frozen goal encoder cannot be language-modulated (language_on_goal_encoder)')
            if goal_encoder in ('separate_base', 'frozen'):
                self.goal_backbone = copy.deepcopy(self.backbones[0])
            if goal_encoder == 'frozen':
                self.goal_backbone.requires_grad_(False)
                self.goal_backbone.eval()
            if goal_role_embedding:
                self.goal_role_embed = nn.Embedding(len(self.goal_views), hidden_dim)
                if goal_tag_init == 'zero':
                    nn.init.zeros_(self.goal_role_embed.weight)
                else:
                    nn.init.normal_(self.goal_role_embed.weight, std=float(goal_tag_init[7:]))
            elif goal_tag_init != 'zero':
                raise ValueError('goal_tag_init requires the goal role embedding')
        elif goal_tag_init != 'zero' or goal_stem_init != 'zero' or goal_tokens != 'grid':
            raise ValueError('goal_tag_init / goal_stem_init / goal_tokens require goal fusion')

    def train(self, mode=True):
        """nn.Module.train, except that a frozen goal encoder always stays in eval mode."""
        super().train(mode)
        if self.goal_encoder == 'frozen' and self.goal_backbone is not None:
            self.goal_backbone.eval()
        return self

    @staticmethod
    def goal_difference(current, goal, goal_present, view):
        """Early fusion's explicit difference channel `goal - current`, zero for an absent goal."""
        difference = goal - current
        return difference if goal_present is None else difference * goal_present[:, view, None, None, None]

    def goal_token_layout(self, feature_width):
        """Memory layout of late fusion: (extra tokens, camera token columns, goal token columns) per feature row."""
        extra = 3 if self.mt_act else 2
        cameras = len(self.camera_names) * feature_width
        return extra, cameras, len(self.goal_views) * feature_width

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, lang_emb=None, goal=None, goal_valid=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        goal: batch, len(goal_views), channel, height, width (goal-image conditioning only; same normalization as image)
        goal_valid: optional bool (batch, len(goal_views)); False marks an absent goal: early fusion zeroes its
            channels (the paired stem then equals the base stem exactly), late fusion masks its tokens out of
            every attention. None means every goal is present (no mask is built).
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        if self.goal_fusion != 'none':
            if goal is None or goal.dim() != 5 or goal.shape[:2] != (bs, len(self.goal_views)):
                raise ValueError(f'goal_fusion={self.goal_fusion} requires goal images of shape '
                                 f'(B, {len(self.goal_views)}, 3, H, W)')
            if goal_valid is not None:
                goal_valid = goal_valid.to(dtype=torch.bool)
                if goal_valid.shape != goal.shape[:2]:
                    raise ValueError('goal_valid must have shape (B, len(goal_views))')
                if self.goal_fusion == 'early':
                    goal = goal * goal_valid.to(goal.dtype)[:, :, None, None, None]
        elif goal is not None and goal.shape[1:2] != (0,):
            raise ValueError('This policy has no goal-image path; do not pass goal images')
        goal_present = None if goal_valid is None else goal_valid.to(image.dtype)
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
                stacked = image[:, :ncam]
                if self.goal_fusion == 'early':
                    # Every camera gets a goal half so one pass covers all of them; cameras without a goal view get
                    # zeros, which the bias-free goal stem maps to exactly zero (the base stem).
                    goal_half = torch.zeros_like(stacked)
                    for view, cam_id in enumerate(self.goal_views):
                        goal_half[:, cam_id] = goal[:, view]
                    parts = [stacked, goal_half]
                    if getattr(self.backbones[0][0].body.conv1, 'goal_diff_weight', None) is not None:
                        diff = torch.zeros_like(stacked)
                        for view, cam_id in enumerate(self.goal_views):
                            diff[:, cam_id] = self.goal_difference(stacked[:, cam_id], goal[:, view], goal_present, view)
                        parts.append(diff)
                    stacked = torch.cat(parts, dim=2)
                flat = stacked.transpose(0, 1).reshape(ncam * bs, *stacked.shape[2:])
                features, pos = self.backbones[0](flat, lang_emb=None if lang_emb is None else lang_emb.repeat(ncam, 1))
                features = self.input_proj(features[0]) # (ncam * bs, hidden, h, w)
                projected = [features[c * bs:(c + 1) * bs] for c in range(ncam)]
                # fold camera dimension into width dimension, in camera order (same layout as the per-camera cat)
                src = features.view(ncam, bs, *features.shape[1:]).permute(1, 2, 3, 0, 4).reshape(
                    bs, features.shape[1], features.shape[2], ncam * features.shape[3])
                pos = pos[0]
                pos = torch.cat([pos if pos.shape[0] == 1 else pos[:bs]] * ncam, axis=3)
            else:
                all_cam_features = []
                all_cam_pos = []
                for cam_id, cam_name in enumerate(self.camera_names):
                    cam_image = image[:, cam_id]
                    goal_kwargs = {}
                    if self.goal_fusion == 'early' and cam_id in self.goal_views:
                        view = self.goal_views.index(cam_id)
                        if self.goal_fusion_depth:
                            # merged after stage K inside the backbone; an absent goal scales the merge to zero
                            goal_kwargs = {'goal': goal[:, view],
                                           'goal_scale': None if goal_present is None else goal_present[:, view]}
                        else:
                            # [current; goal] channel stacking; both halves carry the same normalization
                            parts = [cam_image, goal[:, view]]
                            if getattr(self.backbones[0][0].body.conv1, 'goal_diff_weight', None) is not None:
                                parts.append(self.goal_difference(cam_image, goal[:, view], goal_present, view))
                            cam_image = torch.cat(parts, dim=1)
                            if self.goal_detach_views:
                                self.backbones[0][0].body.conv1.detach_goal = cam_id in self.goal_detach_views
                    features, pos = self.backbones[0](cam_image, lang_emb=lang_emb, camera=cam_id, **goal_kwargs) # HARDCODED
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
                if self.goal_detach_views:
                    self.backbones[0][0].body.conv1.detach_goal = False
                projected = all_cam_features
                # fold camera dimension into width dimension
                src = torch.cat(all_cam_features, axis=3)
                pos = torch.cat(all_cam_pos, axis=3)
            mask = None
            memory_tokens = memory_pos = decoder_tokens = decoder_pos = decoder_mask = query_add = None
            if self.goal_fusion == 'late':
                # Goal-view tokens: same backbone (or its copy), projection and spatial positions as a camera,
                # plus the learned goal identity on the positional stream; appended after the current-view tokens.
                goal_backbone = self.goal_backbone if self.goal_backbone is not None else self.backbones[0]
                goal_lang = lang_emb if self.language_on_goal_encoder else None
                goal_features, goal_pos = [], []
                for view, cam_id in enumerate(self.goal_views):
                    with torch.no_grad() if self.goal_encoder == 'frozen' else nullcontext():
                        features, gpos = goal_backbone(goal[:, view], lang_emb=goal_lang, camera=cam_id,
                                                       film_identity=goal_lang is None)
                    features = self.input_proj(features[0])
                    if self.goal_content == 'diff':
                        features = features - projected[cam_id]  # the projection's bias cancels
                    gpos = gpos[0]
                    if self.goal_pos == 'none':
                        gpos = torch.zeros_like(gpos[:1])
                    if self.goal_tokens == 'pooled':
                        features = features.mean(dim=(2, 3))  # (bs, hidden): one token, no spatial position
                        gpos = torch.zeros_like(features[:1])
                    if self.goal_role_embed is not None:
                        role = self.goal_role_embed.weight[view].to(gpos.dtype)
                        gpos = gpos + (role.view(1, -1) if self.goal_tokens == 'pooled' else role.view(1, -1, 1, 1))
                    goal_features.append(features)
                    goal_pos.append(gpos if gpos.shape[0] in (1, bs) else gpos[:bs])
                if self.goal_entry == 'queries':
                    shown = goal_valid.to(goal_features[0].dtype) if goal_valid is not None else None
                    query_add = sum(f if shown is None else f * shown[:, view, None]
                                    for view, f in enumerate(goal_features))
                elif self.goal_entry == 'decoder':
                    # flattened like the encoder memory: (tokens, batch, hidden), view after view
                    tokens = [f.unsqueeze(0) if self.goal_tokens == 'pooled' else f.flatten(2).permute(2, 0, 1)
                              for f in goal_features]
                    codes = [p.unsqueeze(0).expand(-1, bs, -1) if self.goal_tokens == 'pooled'
                             else p.expand(bs, -1, -1, -1).flatten(2).permute(2, 0, 1) for p in goal_pos]
                    decoder_tokens, decoder_pos = torch.cat(tokens, dim=0), torch.cat(codes, dim=0)
                    if goal_valid is not None:
                        decoder_mask = torch.cat([(~goal_valid[:, view, None]).expand(-1, t.shape[0])
                                                  for view, t in enumerate(tokens)], dim=1)
                elif self.goal_tokens == 'pooled':
                    # Encoder memory [extra tokens, camera tokens, one token per goal view]
                    memory_tokens = torch.stack(goal_features, dim=0)
                    memory_pos = torch.stack(goal_pos, dim=0)
                    if goal_valid is not None:
                        extra = 3 if self.mt_act else 2
                        shown = torch.zeros(bs, extra + src.shape[2] * src.shape[3], dtype=torch.bool, device=src.device)
                        mask = torch.cat([shown, ~goal_valid], dim=1)
                else:
                    feature_width = goal_features[0].shape[3]
                    src = torch.cat([src, *goal_features], axis=3)
                    pos = torch.cat([pos, *goal_pos], axis=3)
                    if goal_valid is not None:
                        # Key padding mask over the encoder memory: True hides a token. Layout per feature row is
                        # [camera columns | goal-view columns]; extra tokens (latent, proprio, task) come first.
                        extra, camera_columns, _ = self.goal_token_layout(feature_width)
                        grid = torch.zeros(bs, src.shape[2], src.shape[3], dtype=torch.bool, device=src.device)
                        for view in range(len(self.goal_views)):
                            start = camera_columns + view * feature_width
                            grid[:, :, start:start + feature_width] = ~goal_valid[:, view, None, None]
                        mask = torch.cat([torch.zeros(bs, extra, dtype=torch.bool, device=src.device), grid.flatten(1)], dim=1)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            hs = self.transformer(src, mask, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight,
                                  decoder_layers=self.decoder_layers_used, task_emb=task_emb,
                                  memory_tokens=memory_tokens, memory_pos=memory_pos, decoder_tokens=decoder_tokens,
                                  decoder_pos=decoder_pos, decoder_mask=decoder_mask, query_add=query_add)[0]
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
    if getattr(args, 'camera_batch', False) and getattr(args, 'backbone_norm', 'frozen') == 'batch_per_camera':
        raise ValueError('Per-camera BatchNorm statistics need one backbone pass per camera; '
                         'camera batching and backbone_norm batch_per_camera are exclusive')

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
        goal_fusion=getattr(args, 'goal_fusion', 'none'),
        goal_views=getattr(args, 'goal_camera_indices', ()),
        goal_role_embedding=bool(getattr(args, 'goal_role_embedding', True)),
        goal_encoder=getattr(args, 'goal_encoder', 'shared_base'),
        language_on_goal_encoder=bool(getattr(args, 'language_on_goal_encoder', False)),
        goal_tag_init=getattr(args, 'goal_tag_init', 'zero'),
        goal_stem_init=getattr(args, 'goal_stem_init', 'zero'),
        goal_tokens=getattr(args, 'goal_tokens', 'grid'),
        goal_pos=getattr(args, 'goal_pos', 'sine'),
        goal_content=getattr(args, 'goal_content', 'goal'),
        goal_entry=getattr(args, 'goal_entry', 'encoder'),
        goal_stem_gain=getattr(args, 'goal_stem_gain', None),
        goal_stem_diff=bool(getattr(args, 'goal_stem_diff', False)),
        goal_fusion_depth=int(getattr(args, 'goal_fusion_depth', 0)),
        goal_detach_views=getattr(args, 'goal_detach_camera_indices', ()),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    if getattr(args, 'language_conditioning', 'none') != 'none':
        raise ValueError('CLIP FiLM language conditioning is only supported for ACT, not CNNMLP')
    if getattr(args, 'goal_fusion', 'none') != 'none':
        raise ValueError('Goal-image conditioning is only supported for ACT, not CNNMLP')
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

