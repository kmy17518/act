#!/usr/bin/env bash
# Task-name CLIP/FiLM run; defaults match the documented unconditioned radio run.
set -euo pipefail
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-GPU-fe99daa5-20f4-4f6f-9792-14a1fd4091a1}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
RUN=${B1K_RUN_DIR:-outputs/turning-on-radio-act-clipfilm-taskname-bs1560-300k-20260916}
LOG=${B1K_LOG:-/tmp/dev/logs/act-radio-clipfilm-taskname-300k-20260916.log}
STATUS=${B1K_STATUS:-/tmp/dev/logs/act-radio-clipfilm-taskname-300k-20260916.exit}
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
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
taskset -c 60-89 .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path /tmp/dev/datasets/2026-challenge-demos --task-names turning_on_radio \
    --output-dir "$RUN" --policy-class ACT --max-steps 300000 \
    --language-conditioning clip_film --prompt-source task_name \
    --hidden-dim 512 --dim-feedforward 3200 --enc-layers 4 --dec-layers 7 --nheads 8 \
    --chunk-size 100 --image-size 240 240 --position-embedding sine --no-pre-norm \
    --kl-weight 10 --lr 1e-5 --lr-backbone 1e-5 --weight-decay 1e-4 \
    --batch-size 1560 --loader-batch-size 128 --num-workers 24 --prefetch-factor 1 \
    --torch-threads 2 --worker-threads 1 --arrow-threads 1 --opencv-threads 1 --device cuda \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-act \
    --wandb-name turning-on-radio-act-clipfilm-taskname-bs1560-300k \
    --wandb-id actradioclipname16 \
    "${args[@]}" "$@"
