#!/usr/bin/env bash
# Conditioning-regime smoke runs on the two-task radio mixture (5,000 steps each): one GPU, 30 CPU cores.
#
# Dataset: /tmp/dev/datasets/2026-challenge-demos-radio-navpickup-goal, the merge (scripts/b1k/merge_lerobot_roots.py)
# of the nav-goal and pickup-goal skill-segment datasets -- 400 episodes, 257,818 frames, tasks
# turning_on_radio-navigate_to_radio (0) and turning_on_radio-pick_up_radio (1). Language = the task name
# (--prompt-source task_name); goal image = the last frame of the episode's head camera (--goal-source episode_last).
# The 240 px frame cache of the challenge demos is reused unchanged (the merged root hard-links the same video files).
#
# ACT_CONDITION selects the run and, by default, the checkout whose code trains it:
#   vanilla                the `my` branch recipe (checkout /tmp/dev/baselines/act): no language, no goal; NOTE the `my`
#                          adapter always appends the one-hot task id to the state, so on two tasks this is the
#                          task-ID-conditioned baseline, not the plan's strict N (use `none` for that)
#   none                   strict N from this checkout: --regime none (no task id, language or goal)
#   language               MT-ACT-style language from the lang_optimized_mt_act checkout (/tmp/dev/baselines/mt-act):
#                          --language-conditioning mt_act --language-encoder minilm --prompt-source task_name on the
#                          base backbone (ImageNet init, frozen BatchNorm, 240 px), so only the language path differs
#                          from vanilla; the branch's from-scratch BatchNorm reproduction recipe is not used here
#   image-early | image-late                     --regime image, goal fusion early | late (this checkout)
#   image_language-early | image_language-late   --regime image_language with mt_act language (this checkout)
# Overrides: ACT_CHECKOUT (code + .venv + outputs/ to use), ACT_GPU_UUID (default GPU 0), ACT_CORES (default 0-29),
# ACT_BATCH_SIZE (1560, the base recipe's validated maximum at 240 px), ACT_MAX_STEPS (5000), ACT_RUN_TAG (20260920),
# ACT_COMPILE_MODE (regions-autotune), ACT_AUTOCAST (none), ACT_DATASET, ACT_FRAME_CACHE, ACT_WANDB_ID, ACT_WANDB_MODE
# (online). Extra arguments are passed to train_b1k.py. Launch from a dedicated tmux server, e.g.
#   tmux -L b1k-smoke new-session -d -s act-vanilla 'ACT_CONDITION=vanilla ACT_GPU_UUID=... bash .../run_navpickup_conditioning_smoke.sh'
main() {
set -euo pipefail
source /tmp/dev/env.sh
CONDITION=${ACT_CONDITION:?set ACT_CONDITION (vanilla|none|language|image-early|image-late|image_language-early|image_language-late)}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
case "$CONDITION" in
    vanilla)  CHECKOUT=${ACT_CHECKOUT:-/tmp/dev/baselines/act};    cond_args=() ;;
    none)     CHECKOUT=${ACT_CHECKOUT:-$HERE};                       cond_args=(--regime none) ;;
    language) CHECKOUT=${ACT_CHECKOUT:-/tmp/dev/baselines/mt-act}
              cond_args=(--language-conditioning mt_act --language-encoder minilm --prompt-source task_name) ;;
    image-early|image-late)
              CHECKOUT=${ACT_CHECKOUT:-$HERE}
              cond_args=(--regime image --goal-fusion "${CONDITION#image-}" --goal-views zed_link --goal-source episode_last) ;;
    image_language-early|image_language-late)
              CHECKOUT=${ACT_CHECKOUT:-$HERE}
              cond_args=(--regime image_language --language-conditioning mt_act --language-encoder minilm
                         --prompt-source task_name --goal-fusion "${CONDITION#image_language-}" --goal-views zed_link
                         --goal-source episode_last) ;;
    *) printf 'Unknown ACT_CONDITION %s\n' "$CONDITION" >&2; exit 2 ;;
esac
cd "$CHECKOUT"
export CUDA_VISIBLE_DEVICES=${ACT_GPU_UUID:-GPU-8a92797c-0df1-b962-f810-3ef2ca82ab80}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CPATH=/tmp/dev/sysroots/libpython3.10-dev/usr/include/python3.10:/tmp/dev/sysroots/libpython3.10-dev/usr/include${CPATH:+:$CPATH}
export WANDB_BASE_URL=https://api.wandb.ai
BATCH_SIZE=${ACT_BATCH_SIZE:-1560}
MAX_STEPS=${ACT_MAX_STEPS:-5000}
RUN_TAG=${ACT_RUN_TAG:-20260920}
AUTOCAST=${ACT_AUTOCAST:-none}
COMPILE_MODE=${ACT_COMPILE_MODE:-regions-autotune}
CORES=${ACT_CORES:-0-29}
DATASET=${ACT_DATASET:-/tmp/dev/datasets/2026-challenge-demos-radio-navpickup-goal}
CACHE=${ACT_FRAME_CACHE:-/tmp/dev/datasets/2026-challenge-demos-act-frame-cache-240x240}
WANDB_MODE_ARG=${ACT_WANDB_MODE:-online}
TASKS=(turning_on_radio-navigate_to_radio turning_on_radio-pick_up_radio)
STEM=navpickup-act-${CONDITION}-bs${BATCH_SIZE}-$((MAX_STEPS / 1000))k-${RUN_TAG}
RUN=outputs/$STEM
LOG=/tmp/dev/logs/${STEM}.log
STATUS=/tmp/dev/logs/${STEM}.exit
WANDB_ID=${ACT_WANDB_ID:-actnp-${CONDITION}-${RUN_TAG}}
mkdir -p "$(dirname "$LOG")"
if [[ -e "$STATUS" ]]; then
    printf 'Archive the previous exit status before restarting: %s\n' "$STATUS" >&2
    exit 1
fi
trap 'rc=$?; printf "%s\n" "$rc" >"$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"' EXIT
exec >>"$LOG" 2>&1
if [[ -n "$(nvidia-smi --id "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    printf 'Assigned GPU %s is occupied; refusing to start.\n' "$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
mkdir -p "$RUN"
if [[ ! -e "$RUN/trainer_commit.txt" ]]; then
    { git rev-parse HEAD; git rev-parse --abbrev-ref HEAD; } >"$RUN/trainer_commit.txt"
fi
printf '[%s] condition=%s checkout=%s commit=%s gpu=%s cores=%s\n' "$(date -u +%FT%TZ)" "$CONDITION" "$CHECKOUT" \
    "$(git rev-parse --short HEAD)" "$CUDA_VISIBLE_DEVICES" "$CORES"
# Cache entries for the merged root are the challenge-demo entries (same files); this validates them and is a no-op.
taskset -c "$CORES" env CUDA_VISIBLE_DEVICES='' .venv/bin/python -u scripts/b1k/build_frame_cache.py \
    --dataset-path "$DATASET" --cache-dir "$CACHE" --task-names "${TASKS[@]}" --image-size 240 240 \
    --workers 10 --cpu-budget 30 --verify 16
args=()
if [[ -e "$RUN/latest.pt" ]]; then
    args+=(--resume "$RUN/latest.pt")
fi
taskset -c "$CORES" .venv/bin/python -u scripts/b1k/train_b1k.py \
    --dataset-path "$DATASET" --task-names "${TASKS[@]}" --frame-cache "$CACHE" \
    --output-dir "$RUN" --policy-class ACT --max-steps "$MAX_STEPS" \
    "${cond_args[@]}" \
    --hidden-dim 512 --dim-feedforward 3200 --enc-layers 4 --dec-layers 7 --nheads 8 \
    --chunk-size 100 --image-size 240 240 --position-embedding sine --no-pre-norm \
    --kl-weight 10 --lr 1e-5 --lr-backbone 1e-5 --weight-decay 1e-4 \
    --batch-size "$BATCH_SIZE" --loader-batch-size 128 --num-workers 8 --prefetch-factor 2 \
    --torch-threads 2 --worker-threads 1 --arrow-threads 1 --opencv-threads 1 --device cuda \
    --matmul-precision high --autocast "$AUTOCAST" --compile "$COMPILE_MODE" \
    --save-every 2500 --save-first-step --save-total-limit 3 --export-every 5000 \
    --wandb-mode "$WANDB_MODE_ARG" --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-act \
    --wandb-name "$STEM" --wandb-id "$WANDB_ID" \
    "${args[@]}" "$@"
}
main "$@"
