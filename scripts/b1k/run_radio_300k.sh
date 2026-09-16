#!/usr/bin/env bash
# Workspace launch recipe: one GPU, 30 CPU cores, FP32 ACT, and resumable logging.
set -euo pipefail
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
export CUDA_VISIBLE_DEVICES=GPU-aa99f910-8e39-d04c-a717-a6f7a06f52e8
if [[ -n "$(nvidia-smi --id "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf 'Assigned ACT GPU is occupied; refusing to start.\n' >&2
    exit 1
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
RUN=outputs/turning-on-radio-act-bs1560-300k-20260916
LOG=/tmp/dev/logs/act-radio-300k-20260916.log
STATUS=/tmp/dev/logs/act-radio-300k-20260916.exit
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
set +e
taskset -c 60-89 .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path /tmp/dev/datasets/2026-challenge-demos --task-names turning_on_radio \
    --output-dir "$RUN" --policy-class ACT --max-steps 300000 \
    --hidden-dim 512 --dim-feedforward 3200 --enc-layers 4 --dec-layers 7 --nheads 8 \
    --chunk-size 100 --image-size 240 240 --position-embedding sine --no-pre-norm \
    --kl-weight 10 --lr 1e-5 --lr-backbone 1e-5 --weight-decay 1e-4 \
    --batch-size 1560 --loader-batch-size 128 --num-workers 24 --prefetch-factor 1 \
    --torch-threads 2 --worker-threads 1 --arrow-threads 1 --opencv-threads 1 --device cuda \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 10000 \
    --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-act \
    --wandb-name turning-on-radio-act-bs1560-300k --wandb-id actradio16 \
    "${args[@]}" >>"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
