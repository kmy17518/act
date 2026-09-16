#!/usr/bin/env bash
# Single writer for this run's eval archive and quota-cleaned latest full checkpoint.
set -euo pipefail
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
LOG=/tmp/dev/logs/act-radio-upload-20260916.log
STATUS=/tmp/dev/logs/act-radio-upload-20260916.exit
set +e
taskset -c 120-121 .venv/bin/python -u scripts/b1k/upload_checkpoints.py \
    --run-dir outputs/turning-on-radio-act-bs1560-300k-20260916 \
    --staging-dir /tmp/dev/hf-staging/act-radio-300k-20260916 \
    --repo-id kmy17518/b1k-act-turning-on-radio-20260916 --run-id act-radio-300k-20260916 \
    --max-steps 300000 --eval-every 10000 --poll-seconds 30 \
    --metadata policy=ACT --metadata task=turning_on_radio --metadata batch_size=1560 \
    --metadata trainer_commit=1c83843771887e5cac2214e8eba18a6e47a6b385 \
    --metadata wandb_url=https://wandb.ai/kmy17518/b1k-challenge-2026-act/runs/actradio16 \
    >>"$LOG" 2>&1
rc=$?
printf '%s\n' "$rc" >"$STATUS.tmp"
mv "$STATUS.tmp" "$STATUS"
exit "$rc"
