import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        # 'all' (upstream ACT; checkpoints without the key) averages the masked L1 over every chunk step, so padding
        # scales a short episode's L1 down relative to KL; 'valid' averages over the unpadded steps only.
        self.l1_normalization = args_override.get('l1_normalization', 'all')
        if self.l1_normalization not in ('all', 'valid'):
            raise ValueError(f"l1_normalization must be 'all' or 'valid', got {self.l1_normalization!r}")
        # Stop study, Method A: the last two action dims are the stop and progress targets (b1k_dataset.stop_targets).
        # Their L1 covers every chunk step, padded ones included (that is where stop = 1 lives); the motion dims keep
        # the is_pad masking above. `stop_pos_weight` upweights the positive (stop = 1) steps of the stop term.
        self.stop_dims = bool(args_override.get('stop_dims', False))
        self.stop_pos_weight = float(args_override.get('stop_pos_weight', 1.0))
        self.stop_loss_weight = float(args_override.get('stop_loss_weight', 1.0))
        self.progress_loss_weight = float(args_override.get('progress_loss_weight', 1.0))
        if self.stop_dims and min(self.stop_pos_weight, self.stop_loss_weight, self.progress_loss_weight) <= 0:
            raise ValueError('stop_pos_weight, stop_loss_weight and progress_loss_weight must be positive')
        print(f'KL Weight {self.kl_weight}')

    def forward(self, qpos, image, actions=None, is_pad=None, lang_emb=None, goal=None, goal_valid=None,
                l1_terms=False):
        """Training (`actions` given): the loss dict; `l1_terms` adds per-sample masked L1 sums and valid element
        counts (`l1_sum`, `l1_count`, shape (B,)) for per-task validation metrics. Inference: the action chunk."""
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        goal_kwargs = {}
        if goal is not None and goal.shape[1] > 0:
            # Goal images (B, views, 3, H, W) in [0, 1]: the same ImageNet normalization as the camera images, applied
            # before any channel stacking or goal encoding.
            goal_kwargs = {'goal': normalize(goal), 'goal_valid': goal_valid}
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad, lang_emb=lang_emb,
                                                         **goal_kwargs)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            extra = None
            if self.stop_dims:
                all_l1, extra = all_l1[..., :-2], all_l1[..., -2:]
            valid = ~is_pad.unsqueeze(-1)
            if self.l1_normalization == 'valid':
                l1 = (all_l1 * valid).sum() / (valid.sum() * all_l1.shape[-1]).clamp_min(1)
            else:
                l1 = (all_l1 * valid).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            if extra is not None:
                positive = (actions[..., -2] > 0).to(extra.dtype)
                weight = 1 + (self.stop_pos_weight - 1) * positive
                loss_dict['l1_stop'] = (extra[..., 0] * weight).sum() / weight.sum().clamp_min(1)
                loss_dict['l1_progress'] = extra[..., 1].mean()
                loss_dict['loss'] = (loss_dict['loss'] + self.stop_loss_weight * loss_dict['l1_stop']
                                     + self.progress_loss_weight * loss_dict['l1_progress'])
            if l1_terms:
                return loss_dict, {'l1_sum': (all_l1 * valid).sum((1, 2)), 'l1_count': valid.sum((1, 2)) * all_l1.shape[-1]}
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, lang_emb=lang_emb, **goal_kwargs) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
