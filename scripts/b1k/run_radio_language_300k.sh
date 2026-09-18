#!/usr/bin/env bash
# Task-name CLIP/FiLM 300k-step turning_on_radio ACT run: one GPU, 30 CPU cores. Same recipe as
# run_radio_300k.sh (frame cache, TF32 matmuls, channels-last, fused AdamW, Triton stem pooling, skipped
# unused decoder layers, GPU batch assembly, --compile regions-autotune) plus
# `--language-conditioning clip_film --prompt-source task_name`, so the runs are directly comparable with
# the unconditioned outputs/turning-on-radio-act-bs1560-300k-opt20260917 run.
#
# The script runs the checkout it lives in (a git worktree such as /tmp/dev/baselines/act-lang works), with
# that checkout's own .venv (scripts/b1k/setup_venv.sh plus transformers, see requirements-b1k.lock.txt).
# Overrides (environment variables, all prefixed ACT_ for the same reason as in run_radio_300k.sh; launch
# every run from a dedicated tmux server, e.g.
#   tmux -L b1k-act-lang new-session -d -s act-radio-lang-random 'bash .../run_radio_language_300k.sh'):
#   ACT_FILM_INIT      random (default; nn.Linear default FiLM projection) or identity (zero projection, the
#                      conditioned network starts as the unconditioned one). Part of the run name and W&B id.
#   ACT_FILM_RECOMPUTE 0 (default) stores the FiLM-conditioned ResNet block activations: 0.45 s/step at batch
#                      1560 with 165 GiB peak, the same step time as the unconditioned recipe. 1 recomputes them
#                      in backward (0.48 s/step, 135 GiB); the trainer's own default is to recompute.
#   ACT_RUN_TAG        default opt20260917; names outputs/turning-on-radio-act-clipfilm-taskname-${ACT_FILM_INIT}-bs${ACT_BATCH_SIZE}-300k-${ACT_RUN_TAG}
#                      and the matching log/exit files and W&B run (ACT_WANDB_ID defaults to
#                      actradioclip-${ACT_FILM_INIT}-${ACT_RUN_TAG}). A run resumes its own latest.pt.
#   ACT_GPU_UUID       default by initialization on the host provisioned 2026-09-17: random -> GPU 1
#                      (GPU-82d44829-...), identity -> GPU 3 (GPU-3b6edd76-...). One GPU per run.
#   ACT_CORES          taskset range for loader workers and trainer; default random -> 0-29, identity -> 30-59
#                      (the unconditioned ACT run uses 60-89, Diffusion Policy 90-119, uploaders 120-123).
#   ACT_BATCH_SIZE (1560), ACT_AUTOCAST (none), ACT_COMPILE_MODE (regions-autotune), ACT_FRAME_CACHE, ACT_DATASET,
#   ACT_WANDB_ID       as in run_radio_300k.sh.
set -euo pipefail
source /tmp/dev/env.sh
cd "$(dirname "$0")/../.."
FILM_INIT=${ACT_FILM_INIT:-random}
case "$FILM_INIT" in
    random) DEFAULT_GPU=GPU-82d44829-7cec-7d3c-9918-6dc1321320d4; DEFAULT_CORES=0-29 ;;
    identity) DEFAULT_GPU=GPU-3b6edd76-2c6a-3087-f7f1-964db3c27635; DEFAULT_CORES=30-59 ;;
    *) printf 'ACT_FILM_INIT must be random or identity, got %s\n' "$FILM_INIT" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=${ACT_GPU_UUID:-$DEFAULT_GPU}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Triton (fast max-pool, torch.compile) builds its launcher with gcc and needs Python.h; setup_venv.sh
# stages the headers under /tmp/dev/sysroots when the host has no python3-dev. Harmless otherwise.
export CPATH=/tmp/dev/sysroots/libpython3.10-dev/usr/include/python3.10:/tmp/dev/sysroots/libpython3.10-dev/usr/include${CPATH:+:$CPATH}
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
BATCH_SIZE=${ACT_BATCH_SIZE:-1560}
RUN_TAG=${ACT_RUN_TAG:-opt20260917}
AUTOCAST=${ACT_AUTOCAST:-none}
COMPILE_MODE=${ACT_COMPILE_MODE:-regions-autotune}
CORES=${ACT_CORES:-$DEFAULT_CORES}
DATASET=${ACT_DATASET:-/tmp/dev/datasets/2026-challenge-demos}
CACHE=${ACT_FRAME_CACHE:-/tmp/dev/datasets/2026-challenge-demos-act-frame-cache-240x240}
RECOMPUTE_FLAG=$([[ "${ACT_FILM_RECOMPUTE:-0}" == 1 ]] && echo --film-recompute || echo --no-film-recompute)
STEM=turning-on-radio-act-clipfilm-taskname-${FILM_INIT}-bs${BATCH_SIZE}-300k-${RUN_TAG}
RUN=outputs/$STEM
LOG=/tmp/dev/logs/act-radio-clipfilm-${FILM_INIT}-300k-${RUN_TAG}.log
STATUS=/tmp/dev/logs/act-radio-clipfilm-${FILM_INIT}-300k-${RUN_TAG}.exit
WANDB_NAME=$STEM
WANDB_ID=${ACT_WANDB_ID:-actradioclip-${FILM_INIT}-${RUN_TAG}}
mkdir -p "$(dirname "$LOG")"
if [[ -e "$STATUS" ]]; then
    printf 'Archive the previous exit status before restarting: %s\n' "$STATUS" >&2
    exit 1
fi
trap 'rc=$?; printf "%s\n" "$rc" >"$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"' EXIT
exec >>"$LOG" 2>&1
if [[ -n "$(nvidia-smi --id "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf 'Assigned ACT GPU is occupied; refusing to start.\n' >&2
    exit 1
fi
mkdir -p "$RUN"
if [[ ! -e "$RUN/trainer_commit.txt" ]]; then
    git rev-parse HEAD >"$RUN/trainer_commit.txt"
fi
# Build any missing/stale cache entries (a no-op when complete) and spot-check 64 random samples against
# native decoding before every launch.
taskset -c "$CORES" env CUDA_VISIBLE_DEVICES='' .venv/bin/python -u scripts/b1k/build_frame_cache.py \
    --dataset-path "$DATASET" --cache-dir "$CACHE" --task-names turning_on_radio --image-size 240 240 \
    --workers 10 --cpu-budget 30 --verify 64
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
taskset -c "$CORES" .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path "$DATASET" --task-names turning_on_radio --frame-cache "$CACHE" \
    --output-dir "$RUN" --policy-class ACT --max-steps 300000 \
    --language-conditioning clip_film --prompt-source task_name --film-init "$FILM_INIT" "$RECOMPUTE_FLAG" \
    --hidden-dim 512 --dim-feedforward 3200 --enc-layers 4 --dec-layers 7 --nheads 8 \
    --chunk-size 100 --image-size 240 240 --position-embedding sine --no-pre-norm \
    --kl-weight 10 --lr 1e-5 --lr-backbone 1e-5 --weight-decay 1e-4 \
    --batch-size "$BATCH_SIZE" --loader-batch-size 128 --num-workers 8 --prefetch-factor 2 \
    --torch-threads 2 --worker-threads 1 --arrow-threads 1 --opencv-threads 1 --device cuda \
    --matmul-precision high --autocast "$AUTOCAST" --compile "$COMPILE_MODE" \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-act \
    --wandb-name "$WANDB_NAME" --wandb-id "$WANDB_ID" \
    "${args[@]}" "$@"
