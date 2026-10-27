"""Stop study, Method B: a policy-agnostic task-completion verifier on frozen DINOv2 features of the head camera.

A frame is embedded by a frozen `facebook/dinov2-*` backbone (CLS token and the mean patch token, `FrameEncoder`);
the verifier (`CompletionVerifier`, a small MLP) scores the current frame together with `history` earlier frames
`gap` frames apart (1 s at 30 Hz) and the goal image (the frame the policy is conditioned on): input
[f(t), f(t - gap), ..., f(t - history * gap), f(goal), |f(t) - f(goal)|]. Trained with BCE on demonstration frames,
positives at and after T_end (settle frames included), negatives every earlier frame (leading idle frames and
mid-episode pauses are negatives by construction). scripts/b1k/stop_verifier.py extracts features, trains, tunes the
threshold on held-out episodes and evaluates; a server-side use loads `CompletionVerifier.load`."""

import numpy as np
import torch
import torch.nn as nn

DEFAULT_BACKBONE = 'facebook/dinov2-base'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameEncoder:
    """Frozen DINOv2 features of uint8 RGB frames: [CLS, mean patch token] (2 x hidden) in float16. The backbone
    `tiny` (no download, no transformers) is a 4 x 4 average-pooled RGB thumbnail (48-d) for smoke runs and tests."""
    def __init__(self, backbone=DEFAULT_BACKBONE, device='cuda', dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == 'cuda' else torch.float32
        self.backbone = backbone
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        if backbone == 'tiny':
            self.model, self.dim = None, 48
            return
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(backbone, torch_dtype=self.dtype).to(self.device).eval()
        self.dim = 2 * self.model.config.hidden_size

    @torch.inference_mode()
    def __call__(self, frames):
        """`frames` uint8 (B, H, W, 3) at the training size (224 x 224 for the ACT caches) -> float16 (B, dim)."""
        images = torch.as_tensor(np.ascontiguousarray(frames), device=self.device).permute(0, 3, 1, 2).float() / 255
        images = (images - self.mean) / self.std
        if self.model is None:
            return torch.nn.functional.adaptive_avg_pool2d(images, 4).flatten(1).to(torch.float16)
        hidden = self.model(pixel_values=images.to(self.dtype)).last_hidden_state
        return torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1).to(torch.float16)


class CompletionVerifier(nn.Module):
    def __init__(self, feature_dim, history=3, gap=30, hidden=512, dropout=0.1, backbone=DEFAULT_BACKBONE):
        super().__init__()
        self.feature_dim, self.history, self.gap, self.backbone = int(feature_dim), int(history), int(gap), backbone
        self.net = nn.Sequential(nn.Linear((history + 3) * feature_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 4), nn.GELU(), nn.Linear(hidden // 4, 1))

    def inputs(self, current, past, goal):
        """`current` (B, D), `past` (B, history, D) oldest last, `goal` (B, D) -> the MLP input (B, (history + 3) D)."""
        return torch.cat([current, past.flatten(1), goal, (current - goal).abs()], dim=-1)

    def forward(self, current, past, goal):
        return self.net(self.inputs(current.float(), past.float(), goal.float())).squeeze(-1)

    def p_done(self, current, past, goal):
        return torch.sigmoid(self.forward(current, past, goal))

    def config(self):
        return {'feature_dim': self.feature_dim, 'history': self.history, 'gap': self.gap, 'backbone': self.backbone,
                'hidden': self.net[0].out_features, 'dropout': float(self.net[2].p)}

    def save(self, path, extra=None):
        torch.save({'format': 'b1k-stop-verifier/v1', 'config': self.config(), 'state_dict': self.state_dict(),
                    **(extra or {})}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        saved = torch.load(path, map_location=device, weights_only=False)
        model = cls(**saved['config']).to(device)
        model.load_state_dict(saved['state_dict'])
        model.eval()
        return model, saved


def history_rows(frame, row_of_frame, history, gap, sampled_frames):
    """Rows of the `history` frames gap, 2 gap, ... before `frame` (clamped to the episode start, snapped down to the
    nearest sampled frame). `row_of_frame` maps sampled frame -> feature row; `sampled_frames` is their sorted array."""
    rows = []
    for k in range(1, history + 1):
        target = max(0, frame - k * gap)
        snapped = sampled_frames[np.searchsorted(sampled_frames, target, side='right') - 1]
        rows.append(row_of_frame[int(snapped)])
    return rows
