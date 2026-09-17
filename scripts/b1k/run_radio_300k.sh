#!/usr/bin/env bash
# Workspace launch recipe for the 300k-step turning_on_radio ACT run: one GPU, 30 CPU cores.
#
# Throughput settings (measured on this Blackwell host; details in b1k.md "Throughput options" and
# b1k_runs.md "Throughput"):
#   - uint8 240x240 resized-frame cache, built/verified below (about 8 minutes once, a no-op afterwards),
#     so 8 loader workers keep the GPU fed (<10 ms loader wait) instead of 24 cores decoding HEVC,
#   - fp32 weights/optimizer with TF32 matmuls (--matmul-precision high), explicit upstream attention,
#     channels-last convolutions, fused AdamW, bit-identical Triton stem pooling, discarded decoder
#     layers skipped, batches assembled on the GPU one step ahead, and --compile regions-autotune
#     (Inductor-fused elementwise work plus autotuned GEMM/convolution kernels; a few minutes of
#     compilation on the first step for a new batch size, cached under /tmp/.cache/torchinductor).
#   Steady state: 0.30 s/step at batch 1024, 0.45 s/step at batch 1560 (baseline 1.77 / 3.40 s/step).
# Overrides (environment variables, all prefixed ACT_ so that a tmux server or shell shared with the
# Diffusion Policy recipe -- which uses BATCH_SIZE, FRAME_CACHE, GPU_UUID, ... -- can never redirect this run;
# launch from a dedicated tmux server, e.g. `tmux -L b1k-act new-session -d -s act-radio 'bash .../run_radio_300k.sh'`):
#   ACT_BATCH_SIZE  physical batch (default 1560; 1024 is the other measured size).
#   ACT_RUN_TAG     default 20260916 = the original run directory outputs/turning-on-radio-act-bs1560-300k-20260916,
#                   resumed from latest.pt when present with the original W&B identity. Any other tag names a fresh
#                   run (outputs/turning-on-radio-act-bs${ACT_BATCH_SIZE}-300k-${ACT_RUN_TAG}) with its own log/exit
#                   file and W&B run (ACT_WANDB_ID defaults to actradio-${ACT_RUN_TAG}); it resumes its own latest.pt.
#   ACT_AUTOCAST    none (default: TF32 recipe, resumed losses match the original run within dropout noise),
#                   bf16-backbone (bf16 inside the ResNet bodies only: 0.28 / 0.42 s/step, mean L1 +0.04 % over a
#                   300-step resume) or bf16 (0.29 / 0.44 s/step before compile; L1 +0.6-1.0 %, not neutral).
#   ACT_COMPILE_MODE regions-autotune (default), regions, backbone or none.
#   ACT_GPU_UUID    default: GPU index 2 of the host provisioned on 2026-09-17 (the earlier host's GPU 2 was
#                   GPU-aa99f910-8e39-d04c-a717-a6f7a06f52e8). One GPU per run.
#   ACT_CORES       taskset range for the loader workers and trainer (default 60-89, 30 cores).
#   ACT_FRAME_CACHE, ACT_DATASET, ACT_WANDB_ID  as named. The cache builder refuses a directory that holds
#                   another tool's or another image size's cache entries.
set -euo pipefail
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
export CUDA_VISIBLE_DEVICES=${ACT_GPU_UUID:-GPU-10567c56-9603-b2aa-1ce1-63234ee50192}
if [[ -n "$(nvidia-smi --id "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf 'Assigned ACT GPU is occupied; refusing to start.\n' >&2
    exit 1
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Triton compiles its kernel launcher with gcc and needs Python.h. This host has no python3-dev, so
# scripts/b1k/setup_venv.sh stages the headers under /tmp/dev/sysroots (same convention as the GR00T
# launch scripts); harmless when the system headers exist.
export CPATH=/tmp/dev/sysroots/libpython3.10-dev/usr/include/python3.10:/tmp/dev/sysroots/libpython3.10-dev/usr/include${CPATH:+:$CPATH}
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
BATCH_SIZE=${ACT_BATCH_SIZE:-1560}
RUN_TAG=${ACT_RUN_TAG:-20260916}
AUTOCAST=${ACT_AUTOCAST:-none}
COMPILE_MODE=${ACT_COMPILE_MODE:-regions-autotune}
CORES=${ACT_CORES:-60-89}
DATASET=${ACT_DATASET:-/tmp/dev/datasets/2026-challenge-demos}
CACHE=${ACT_FRAME_CACHE:-/tmp/dev/datasets/2026-challenge-demos-act-frame-cache-240x240}
if [[ "$RUN_TAG" == 20260916 ]]; then
    # The original run: directory, log and W&B identity are fixed regardless of the other overrides.
    RUN=outputs/turning-on-radio-act-bs1560-300k-20260916
    LOG=/tmp/dev/logs/act-radio-300k-20260916.log
    STATUS=/tmp/dev/logs/act-radio-300k-20260916.exit
    WANDB_NAME=turning-on-radio-act-bs1560-300k
    WANDB_ID=${ACT_WANDB_ID:-actradio16}
else
    RUN=outputs/turning-on-radio-act-bs${BATCH_SIZE}-300k-${RUN_TAG}
    LOG=/tmp/dev/logs/act-radio-300k-${RUN_TAG}.log
    STATUS=/tmp/dev/logs/act-radio-300k-${RUN_TAG}.exit
    WANDB_NAME=turning-on-radio-act-bs${BATCH_SIZE}-300k-${RUN_TAG}
    WANDB_ID=${ACT_WANDB_ID:-actradio-${RUN_TAG}}
fi
# Build any missing/stale cache entries (about 8 minutes from scratch on 30 cores; a no-op when
# complete) and spot-check 64 random samples against native decoding before every launch.
taskset -c "$CORES" env CUDA_VISIBLE_DEVICES='' .venv/bin/python -u scripts/b1k/build_frame_cache.py \
    --dataset-path "$DATASET" --cache-dir "$CACHE" --task-names turning_on_radio --image-size 240 240 \
    --workers 10 --cpu-budget 30 --verify 64 >>"$LOG" 2>&1
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
set +e
taskset -c "$CORES" .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path "$DATASET" --task-names turning_on_radio --frame-cache "$CACHE" \
    --output-dir "$RUN" --policy-class ACT --max-steps 300000 \
    --hidden-dim 512 --dim-feedforward 3200 --enc-layers 4 --dec-layers 7 --nheads 8 \
    --chunk-size 100 --image-size 240 240 --position-embedding sine --no-pre-norm \
    --kl-weight 10 --lr 1e-5 --lr-backbone 1e-5 --weight-decay 1e-4 \
    --batch-size "$BATCH_SIZE" --loader-batch-size 128 --num-workers 8 --prefetch-factor 2 \
    --torch-threads 2 --worker-threads 1 --arrow-threads 1 --opencv-threads 1 --device cuda \
    --matmul-precision high --autocast "$AUTOCAST" --compile "$COMPILE_MODE" \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-act \
    --wandb-name "$WANDB_NAME" --wandb-id "$WANDB_ID" \
    "${args[@]}" >>"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
