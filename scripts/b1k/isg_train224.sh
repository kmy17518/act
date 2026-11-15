#!/usr/bin/env bash
# isg-train224 ACT run on the batched replay's native 224 x 224 LeRobot v3 datasets (b1k.md "isg-train224").
#
#   scripts/b1k/isg_train224.sh cache     build the uint8 frame cache (native size: decoded frames stored as is)
#   scripts/b1k/isg_train224.sh prepare   exact statistics + the frame-cache tables / worker sidecars (CPU, one process)
#   scripts/b1k/isg_train224.sh train     DDP training (--require-prepared) + the Hugging Face upload monitor beside it;
#                                         resumes from the newest step_*.pt of the run directory
#   scripts/b1k/isg_train224.sh upload    the upload monitor alone (e.g. restarted after a crash; resumes its journal)
#   scripts/b1k/isg_train224.sh all       cache, prepare, train
#
# Recipe (relq-late-piaug, the earlier ACT side runs' recipe) with the isg-train224 changes: absolute grippers
# (--gripper-action absolute; gripper state = q1 + q2 in metres, --gripper-state sum), exact padding-free quantile
# statistics computed before the run, Pi-style frame epochs (--frame-selection pi: a fresh permutation of every
# training frame per epoch, without replacement), --val-fraction 0.05 (the same episodes as openpi / DP for the seed),
# pi05's warmup + cosine LR schedule over the whole run, 3 local full checkpoints, online W&B.
#
# Environment (defaults in brackets):
#   DATASET        LeRobot v3 root (required)                 FRAME_CACHE  [$DATASET-act-frame-cache-224x224]
#   RUN            [act-isg224-relq-late-piaug-ddp4-bs1024-s$SEED]   OUT_ROOT [/tmp/dev/runs/train224]
#   PY             [python]   GPUS [0,1,2,3]   NUM_WORKERS per rank [8]   CACHE_WORKERS [16]
#   MAX_STEPS      [200000]   SAVE [10000] (save = export = validate)   SEED [0]   VAL_FRACTION [0.05]
#   SETTLE_STEPS   [all]: the dataset's own settle window (batched replay: max(ceil(0.2 A), 50) frames)
#   TASK_NAMES     space-separated tasks / task groups [every task of the dataset]
#   WANDB_ENTITY   [kmy17518]   WANDB_PROJECT [b1k-isg-train224]   WANDB_MODE [online] (needs WANDB_API_KEY)
#   HF_REPO        [kmy17518/$RUN]   HF_PUBLIC [0]   HF_XET [1]   (needs HF_TOKEN with write access)
#   B1K_DATASET_REVISION  the dataset's immutable revision (recommended; recorded in the statistics)
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${DATASET:?set DATASET to the LeRobot v3 root}"
SEED=${SEED:-0}
FRAME_CACHE=${FRAME_CACHE:-${DATASET%/}-act-frame-cache-224x224}
RUN=${RUN:-act-isg224-relq-late-piaug-ddp4-bs1024-s$SEED}
OUT_ROOT=${OUT_ROOT:-/tmp/dev/runs/train224}
OUT=$OUT_ROOT/$RUN
STATS=$OUT_ROOT/stats/$RUN.json
PY=${PY:-python}
GPUS=${GPUS:-0,1,2,3}
NPROC=$(( $(tr -cd ',' <<< "$GPUS" | wc -c) + 1 ))
MAX_STEPS=${MAX_STEPS:-200000}
SAVE=${SAVE:-10000}
HF_REPO=${HF_REPO:-kmy17518/$RUN}
WANDB_PROJECT=${WANDB_PROJECT:-b1k-isg-train224}

recipe=(--dataset-path "$DATASET" --frame-cache "$FRAME_CACHE" --episode-split auto --settle-steps "${SETTLE_STEPS:-all}"
    --gripper-state sum --gripper-action absolute --image-size 224 224 --regime image --goal-views zed_link
    --goal-source episode_last --goal-fusion late --goal-tokens grid --goal-encoder shared_base --goal-tag-init zero
    --lr-goal 1e-4 --lr 1e-5 --lr-backbone 1e-5 --weight-decay 1e-4
    --lr-schedule pi_cosine --lr-warmup-steps 1000 --lr-decay-steps "$MAX_STEPS" --lr-decay-ratio 0.1
    --policy-class ACT --hidden-dim 512 --dim-feedforward 3200 --enc-layers 4 --dec-layers 7 --nheads 8
    --chunk-size 100 --kl-weight 10 --dropout 0.1 --position-embedding sine --no-pre-norm
    --action-representation relative --normalization quantile --stats-estimator exact --image-augment pi05
    --frame-selection pi --batch-size 1024 --loader-batch-size 128 --val-fraction "${VAL_FRACTION:-0.05}"
    --seed "$SEED" --max-steps "$MAX_STEPS" --stats-file "$STATS")
[[ -z ${TASK_NAMES:-} ]] || recipe+=(--task-names $TASK_NAMES)
runtime=(--prefetch-factor 2 --torch-threads 2 --worker-threads 1 --arrow-threads 1 --opencv-threads 1 --device cuda
    --matmul-precision high --autocast bf16-backbone --compile regions-autotune --num-workers "${NUM_WORKERS:-8}"
    --save-every "$SAVE" --export-every "$SAVE" --val-every "$SAVE" --save-total-limit 3 --output-dir "$OUT"
    --stop-file "$OUT_ROOT/$RUN.stop" --wandb-mode "${WANDB_MODE:-online}" --wandb-entity "${WANDB_ENTITY:-kmy17518}"
    --wandb-project "$WANDB_PROJECT" --wandb-name "$RUN" --wandb-id "$RUN")

cache() {
    (cd "$HERE" && "$PY" -u b1k_frame_cache.py --dataset-path "$DATASET" --cache-dir "$FRAME_CACHE" --format v1 \
        --image-size 224 224 --workers "${CACHE_WORKERS:-16}" --cpu-budget "${CACHE_WORKERS:-16}" \
        ${TASK_NAMES:+--task-names $TASK_NAMES})
}

prepare() {
    mkdir -p "$OUT_ROOT/stats" "$OUT_ROOT/prepare"
    (cd "$HERE" && "$PY" -u scripts/b1k/train_b1k.py "${recipe[@]}" --device cpu --num-workers 0 \
        --output-dir "$OUT_ROOT/prepare/$RUN" --prepare-only)
}

upload() {
    mkdir -p "$OUT/export_queue/eval" "$OUT/export_queue/full" "$OUT_ROOT/upload/$RUN"
    local flags=(--run-dir "$OUT" --staging-dir "$OUT_ROOT/upload/$RUN" --repo-id "$HF_REPO" --run-id "$RUN"
                 --max-steps "$MAX_STEPS" --eval-every "$SAVE" --poll-seconds 60)
    [[ ${HF_PUBLIC:-0} == 1 ]] && flags+=(--public)
    [[ ${HF_XET:-1} == 1 ]] && flags+=(--xet)
    (cd "$HERE" && "$PY" -u scripts/b1k/upload_checkpoints.py "${flags[@]}")
}

train() {
    [[ -f $STATS ]] || { echo "no prepared statistics $STATS: run '$0 prepare' first" >&2; exit 1; }
    mkdir -p "$OUT"
    local resume=() newest
    newest=$(find "$OUT" -maxdepth 1 -name 'step_*.pt' | sort | tail -n 1)
    [[ -z $newest ]] || resume=(--resume "$newest")
    rm -f "$OUT_ROOT/$RUN.stop"
    upload >> "$OUT_ROOT/upload/$RUN.log" 2>&1 &
    local uploader=$!
    echo "[$(date -u +%FT%TZ)] $RUN: training on GPUs $GPUS (${resume[*]:-fresh start}); uploader pid $uploader -> $HF_REPO"
    local rc=0
    (cd "$HERE" && CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        ARROW_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_CONSOLE=off \
        "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
        scripts/b1k/train_b1k.py "${recipe[@]}" "${runtime[@]}" --require-prepared "${resume[@]}") || rc=$?
    if ((rc == 0)); then
        echo "[$(date -u +%FT%TZ)] $RUN: trainer finished; waiting for the uploader to publish the last checkpoints"
        wait "$uploader" || echo "uploader exited with $?; restart it with '$0 upload'" >&2
    else
        echo "[$(date -u +%FT%TZ)] $RUN: trainer exited with $rc; the uploader (pid $uploader) keeps publishing what is" \
             "queued; '$0 train' resumes both" >&2
    fi
    return "$rc"
}

case ${1:-} in
    cache) cache ;;
    prepare) prepare ;;
    train) train ;;
    upload) upload ;;
    all) cache && prepare && train ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
esac
