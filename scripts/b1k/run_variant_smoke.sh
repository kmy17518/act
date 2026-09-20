#!/usr/bin/env bash
# Variant branch goal-image-early: the `image-early` condition of the two-task conditioning smoke matrix, trained by this
# checkout's code. Thin wrapper over scripts/b1k/run_navpickup_conditioning_smoke.sh (same overrides, e.g.
# ACT_GPU_UUID, ACT_CORES, ACT_BATCH_SIZE, ACT_MAX_STEPS; extra arguments go to train_b1k.py).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export ACT_CONDITION=image-early ACT_CHECKOUT="$HERE"
exec bash "$HERE/scripts/b1k/run_navpickup_conditioning_smoke.sh" "$@"
