#!/usr/bin/env bash
# MT-ACT reproduction (RoboAgent, Bharadhwaj et al. 2023) on the 300k-step turning_on_radio recipe: one GPU,
# 30 CPU cores. Same optimizer, transformer, batch (1560), sampler seed and checkpoint schedule as
# run_radio_300k.sh / run_radio_language_300k.sh so the runs are comparable; the MT-ACT-specific parts are
#   --language-conditioning mt_act      FiLM inside the residual branch of ResNet stages 2-4, generated from a
#                                       learned text projection that also enters the transformer encoder as a
#                                       token; CVAE style encoder over actions only; the one-hot task input is
#                                       not fed to the network (detr/models/{backbone,detr_vae}.py),
#   --language-encoder minilm           frozen all-MiniLM-L6-v2 sentence embeddings (384-d, unit norm) of the
#   --prompt-source task_description    natural-language task instruction (meta/tasks.jsonl),
#   --no-pretrained-backbone --backbone-norm batch   ResNet-18 trained from scratch with BatchNorm2d, as
#                                       RoboAgent's build_film_backbone does.
# Deliberate deviations: 96x96 images (the Diffusion Policy baseline's resolution; MT-ACT used RoboSet's
# native frames), our three cameras, 23-D actions and action chunk 100 (MT-ACT: four cameras, 8-D, H=20),
# so the L1 is comparable with the other 300k runs, and ACT_CAMERA_BATCH=1 (default): the ResNet runs once over
# the images of all cameras so its BatchNorm statistics are shared across cameras, which is what the running
# statistics used when serving describe (RoboAgent's per-camera passes leave a large train/eval gap, see
# b1k_runs.md). Throughput options are the ones measured in b1k.md.
#
# Runs the checkout it lives in (git worktree /tmp/dev/baselines/mt-act) with that checkout's .venv.
# Overrides (environment variables, ACT_-prefixed as in run_radio_300k.sh; launch from a dedicated tmux
# server, e.g. `tmux -L b1k-mt-act new-session -d -s mt-act-radio 'bash .../run_radio_mt_act_300k.sh'`):
#   ACT_BACKBONE_NORM batch (default): BatchNorm2d as RoboAgent; batch_per_camera: the same training computation
#                    with one set of running statistics per camera (PerCameraBatchNorm2d, "-percam" in the run name;
#                    eval mode then normalizes each camera pass as training did). Implies ACT_CAMERA_BATCH=0.
#   ACT_CAMERA_BATCH 1 (default with batch): --backbone-camera-batch (shared BatchNorm statistics, "-cambatch" in
#                    the run name); 0: RoboAgent's per-camera backbone passes.
#   ACT_RUN_TAG      default opt20260917; names outputs/turning-on-radio-mt-act[-cambatch|-percam]-${ACT_IMAGE_SIZE}px-bs${ACT_BATCH_SIZE}-300k-${ACT_RUN_TAG},
#                    the log/exit files /tmp/dev/logs/act-radio-mt-act[-cambatch|-percam]-${ACT_IMAGE_SIZE}px-300k-${ACT_RUN_TAG}.{log,exit} and the
#                    W&B run (ACT_WANDB_ID defaults to actradio-mtact[-cambatch|-percam]-${ACT_IMAGE_SIZE}px-${ACT_RUN_TAG}; the first 96 px
#                    per-camera run keeps its size-less names act-radio-mt-act-300k-<tag> / actradio-mtact-<tag>). A run resumes
#                    its own latest.pt.
#   ACT_GPU_UUID     default GPU 1 of the host provisioned 2026-09-17 (GPU-82d44829-...). One GPU per run.
#   ACT_CORES        taskset range for loader workers and trainer (default 0-29).
#   ACT_IMAGE_SIZE   default 96 (square); 240 is this adapter's ACT default and the other radio runs' size.
#                    The frame cache directory follows it (…-act-frame-cache-<size>x<size>).
#   ACT_BATCH_SIZE (1560), ACT_AUTOCAST (none), ACT_COMPILE_MODE (regions-autotune), ACT_FRAME_CACHE, ACT_DATASET,
#   ACT_WANDB_ID     as in run_radio_300k.sh.
set -euo pipefail
source /tmp/dev/env.sh
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${ACT_GPU_UUID:-GPU-82d44829-7cec-7d3c-9918-6dc1321320d4}
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
CORES=${ACT_CORES:-0-29}
IMAGE_SIZE=${ACT_IMAGE_SIZE:-96}
DATASET=${ACT_DATASET:-/tmp/dev/datasets/2026-challenge-demos}
CACHE=${ACT_FRAME_CACHE:-/tmp/dev/datasets/2026-challenge-demos-act-frame-cache-${IMAGE_SIZE}x${IMAGE_SIZE}}
BACKBONE_NORM=${ACT_BACKBONE_NORM:-batch}
case "$BACKBONE_NORM" in
    batch_per_camera)
        if [[ "${ACT_CAMERA_BATCH:-0}" == 1 ]]; then
            printf 'ACT_BACKBONE_NORM=batch_per_camera needs one backbone pass per camera; unset ACT_CAMERA_BATCH\n' >&2
            exit 2
        fi
        CAMERA_FLAG=--no-backbone-camera-batch; VARIANT=-percam ;;
    batch)
        if [[ "${ACT_CAMERA_BATCH:-1}" == 1 ]]; then
            CAMERA_FLAG=--backbone-camera-batch; VARIANT=-cambatch
        else
            CAMERA_FLAG=--no-backbone-camera-batch; VARIANT=
        fi ;;
    *) printf 'ACT_BACKBONE_NORM must be batch or batch_per_camera, got %s\n' "$BACKBONE_NORM" >&2; exit 2 ;;
esac
STEM=turning-on-radio-mt-act${VARIANT}-${IMAGE_SIZE}px-bs${BATCH_SIZE}-300k-${RUN_TAG}
RUN=outputs/$STEM
WANDB_NAME=$STEM
if [[ "$IMAGE_SIZE" == 96 && -z "$VARIANT" ]]; then
    # The first (documented) 96 px per-camera run was launched with size-less log and W&B names; keep them resumable.
    LOG=/tmp/dev/logs/act-radio-mt-act-300k-${RUN_TAG}.log
    STATUS=/tmp/dev/logs/act-radio-mt-act-300k-${RUN_TAG}.exit
    WANDB_ID=${ACT_WANDB_ID:-actradio-mtact-${RUN_TAG}}
else
    LOG=/tmp/dev/logs/act-radio-mt-act${VARIANT}-${IMAGE_SIZE}px-300k-${RUN_TAG}.log
    STATUS=/tmp/dev/logs/act-radio-mt-act${VARIANT}-${IMAGE_SIZE}px-300k-${RUN_TAG}.exit
    WANDB_ID=${ACT_WANDB_ID:-actradio-mtact${VARIANT}-${IMAGE_SIZE}px-${RUN_TAG}}
fi
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
# Build any missing/stale cache entries (a few minutes from scratch on 30 cores; a no-op when complete) and
# spot-check 64 random samples against native decoding before every launch.
taskset -c "$CORES" env CUDA_VISIBLE_DEVICES='' .venv/bin/python -u scripts/b1k/build_frame_cache.py \
    --dataset-path "$DATASET" --cache-dir "$CACHE" --task-names turning_on_radio \
    --image-size "$IMAGE_SIZE" "$IMAGE_SIZE" --workers 10 --cpu-budget 30 --verify 64
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
taskset -c "$CORES" .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path "$DATASET" --task-names turning_on_radio --frame-cache "$CACHE" \
    --output-dir "$RUN" --policy-class ACT --max-steps 300000 \
    --language-conditioning mt_act --language-encoder minilm --prompt-source task_description \
    --no-pretrained-backbone --backbone-norm "$BACKBONE_NORM" "$CAMERA_FLAG" \
    --hidden-dim 512 --dim-feedforward 3200 --enc-layers 4 --dec-layers 7 --nheads 8 \
    --chunk-size 100 --image-size "$IMAGE_SIZE" "$IMAGE_SIZE" --position-embedding sine --no-pre-norm \
    --kl-weight 10 --lr 1e-5 --lr-backbone 1e-5 --weight-decay 1e-4 \
    --batch-size "$BATCH_SIZE" --loader-batch-size 128 --num-workers 8 --prefetch-factor 2 \
    --torch-threads 2 --worker-threads 1 --arrow-threads 1 --opencv-threads 1 --device cuda \
    --matmul-precision high --autocast "$AUTOCAST" --compile "$COMPILE_MODE" \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-act \
    --wandb-name "$WANDB_NAME" --wandb-id "$WANDB_ID" \
    "${args[@]}" "$@"
