"""Channels-last 3x3 / stride 2 / padding 1 max pooling with a compact index tensor.

ATen's NHWC max-pool kernels record int64 flat indices and gather them per input element in the
backward pass, which costs ~8x the memory roofline on the ResNet stem output (batch x 64 x 120 x 120).
These Triton kernels store the winning tap (0..8) as one int8 per output and read channels
contiguously, reproducing `F.max_pool2d(x, 3, 2, 1)` bit for bit: strict `>` comparisons in
row-major window order (first maximum wins ties), NaN propagation, and gradients accumulated over the
covering windows in increasing output order like PyTorch's NHWC backward kernel.

`MaxPool3x3NHWC` is a stateless drop-in for torchvision's stem `nn.MaxPool2d(3, 2, 1)`; it uses the
kernels for CUDA tensors and falls back to `F.max_pool2d` elsewhere. The op is registered through
`torch.library.custom_op` so torch.compile treats it as an opaque, differentiable operator.
"""

import torch
from torch import nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments
    triton = tl = None

BLOCK_C = 64
FORWARD_BLOCK_P, FORWARD_WARPS = 16, 4
BACKWARD_BLOCK_P, BACKWARD_WARPS = 16, 2

if triton is not None:
    @triton.jit
    def _forward_kernel(x_ptr, out_ptr, idx_ptr, total, C, H, W, OH, OW, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
        # int64 indexing: the stem output exceeds 2**31 elements at a few thousand images.
        p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)  # flattened (n, oh, ow)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        pmask = p < total  # total = N * OH * OW (int64 when large)
        cmask = c < C
        ow = p % OW
        rest = p // OW
        oh = rest % OH
        n = rest // OH
        best = tl.full((BLOCK_P, BLOCK_C), float('-inf'), tl.float32)
        best_idx = tl.zeros((BLOCK_P, BLOCK_C), tl.int8)
        for kh in tl.static_range(3):
            ih = oh * 2 - 1 + kh
            hvalid = (ih >= 0) & (ih < H)
            for kw in tl.static_range(3):
                iw = ow * 2 - 1 + kw
                valid = pmask & hvalid & (iw >= 0) & (iw < W)
                offset = ((n * H + ih) * W + iw) * C
                value = tl.load(x_ptr + offset[:, None] + c[None, :], mask=valid[:, None] & cmask[None, :],
                                other=float('-inf')).to(tl.float32)
                take = (value > best) | (value != value)  # strict: first maximum wins; NaN propagates
                best = tl.where(take, value, best)
                best_idx = tl.where(take, tl.full((BLOCK_P, BLOCK_C), kh * 3 + kw, tl.int8), best_idx)
        out_offset = p[:, None] * C + c[None, :]
        omask = pmask[:, None] & cmask[None, :]
        tl.store(out_ptr + out_offset, best.to(out_ptr.dtype.element_ty), mask=omask)
        tl.store(idx_ptr + out_offset, best_idx, mask=omask)

    @triton.jit
    def _backward_kernel(gout_ptr, idx_ptr, gin_ptr, total, C, H, W, OH, OW, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
        p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)  # flattened (n, ih, iw)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        pmask = p < total  # total = N * H * W (int64 when large)
        cmask = c < C
        iw = p % W
        rest = p // W
        ih = rest % H
        n = rest // H
        # Output rows covering input row ih: oh_hi = (ih + 1) // 2 through tap kh_hi = ih + 1 - 2 * oh_hi (0 or 1)
        # and, when kh_hi == 0 (odd ih), oh_hi - 1 through tap 2; same along the width. Windows are visited in
        # increasing (oh, ow) order, which is PyTorch's accumulation order.
        oh_hi = (ih + 1) // 2
        kh_hi = ih + 1 - 2 * oh_hi
        ow_hi = (iw + 1) // 2
        kw_hi = iw + 1 - 2 * ow_hi
        acc = tl.zeros((BLOCK_P, BLOCK_C), tl.float32)
        for dh in tl.static_range(2):
            oh = oh_hi - 1 + dh
            kh = kh_hi + 2 - 2 * dh
            hvalid = (oh >= 0) & (oh < OH) & (kh <= 2)
            for dw in tl.static_range(2):
                ow = ow_hi - 1 + dw
                kw = kw_hi + 2 - 2 * dw
                valid = pmask & hvalid & (ow >= 0) & (ow < OW) & (kw <= 2)
                offset = ((n * OH + oh) * OW + ow) * C
                mask = valid[:, None] & cmask[None, :]
                idx = tl.load(idx_ptr + offset[:, None] + c[None, :], mask=mask, other=-1)
                grad = tl.load(gout_ptr + offset[:, None] + c[None, :], mask=mask, other=0.0).to(tl.float32)
                acc += tl.where(idx == (kh * 3 + kw)[:, None], grad, 0.0)
        tl.store(gin_ptr + p[:, None] * C + c[None, :], acc.to(gin_ptr.dtype.element_ty),
                 mask=pmask[:, None] & cmask[None, :])


def _output_size(size):
    return (size + 2 - 3) // 2 + 1


@torch.library.custom_op('act_b1k::max_pool3x3_nhwc', mutates_args=())
def max_pool3x3_nhwc(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(pooled values in channels-last layout, int8 winning tap per output element)."""
    x = x.contiguous(memory_format=torch.channels_last)
    N, C, H, W = x.shape
    OH, OW = _output_size(H), _output_size(W)
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    idx = torch.empty((N, OH, OW, C), device=x.device, dtype=torch.int8)
    grid = (triton.cdiv(N * OH * OW, FORWARD_BLOCK_P), triton.cdiv(C, BLOCK_C))
    _forward_kernel[grid](x, out, idx, N * OH * OW, C, H, W, OH, OW, BLOCK_P=FORWARD_BLOCK_P, BLOCK_C=BLOCK_C,
                          num_warps=FORWARD_WARPS)
    return out, idx


@max_pool3x3_nhwc.register_fake
def _(x):
    N, C, H, W = x.shape
    OH, OW = _output_size(H), _output_size(W)
    return (torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype, memory_format=torch.channels_last),
            torch.empty((N, OH, OW, C), device=x.device, dtype=torch.int8))


@torch.library.custom_op('act_b1k::max_pool3x3_nhwc_backward', mutates_args=())
def max_pool3x3_nhwc_backward(grad_out: torch.Tensor, idx: torch.Tensor, H: int, W: int) -> torch.Tensor:
    grad_out = grad_out.contiguous(memory_format=torch.channels_last)
    N, C, OH, OW = grad_out.shape
    grad_in = torch.empty((N, C, H, W), device=grad_out.device, dtype=grad_out.dtype, memory_format=torch.channels_last)
    grid = (triton.cdiv(N * H * W, BACKWARD_BLOCK_P), triton.cdiv(C, BLOCK_C))
    _backward_kernel[grid](grad_out, idx, grad_in, N * H * W, C, H, W, OH, OW, BLOCK_P=BACKWARD_BLOCK_P,
                           BLOCK_C=BLOCK_C, num_warps=BACKWARD_WARPS)
    return grad_in


@max_pool3x3_nhwc_backward.register_fake
def _(grad_out, idx, H, W):
    N, C = grad_out.shape[:2]
    return torch.empty((N, C, H, W), device=grad_out.device, dtype=grad_out.dtype, memory_format=torch.channels_last)


def _setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(output[1])
    ctx.height, ctx.width = x.shape[-2], x.shape[-1]


def _backward(ctx, grad_out, grad_idx):
    (idx,) = ctx.saved_tensors
    return max_pool3x3_nhwc_backward(grad_out, idx, ctx.height, ctx.width)


max_pool3x3_nhwc.register_autograd(_backward, setup_context=_setup_context)


class MaxPool3x3NHWC(nn.Module):
    """Stateless replacement for `nn.MaxPool2d(3, 2, 1)` using the channels-last kernels on CUDA."""

    def forward(self, x):
        if triton is not None and x.is_cuda and x.dim() == 4 and x.dtype in (torch.float32, torch.bfloat16, torch.float16):
            return max_pool3x3_nhwc(x)[0]
        return F.max_pool2d(x, 3, 2, 1)

    def extra_repr(self):
        return 'kernel_size=3, stride=2, padding=1, layout=channels_last'


def replaces(module):
    """True for the torchvision stem pooling this module reproduces exactly."""
    return (isinstance(module, nn.MaxPool2d) and module.kernel_size in (3, (3, 3)) and module.stride in (2, (2, 2))
            and module.padding in (1, (1, 1)) and module.dilation in (1, (1, 1)) and not module.ceil_mode)
