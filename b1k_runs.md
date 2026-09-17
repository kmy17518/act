# ACT radio training run — 2026-09-16

## Configuration

- Dataset: `/tmp/dev/datasets/2026-challenge-demos`, **turning_on_radio only**, 200 episodes / 429,928 frames; exact full-task statistics.
- Target: **300,000 optimizer steps**.
- Policy: upstream ACT, ImageNet-initialized ResNet18, sine positions, post-norm, hidden 512, feedforward 3200, 4 encoder / 7 decoder layers, 8 heads, action chunk 100, KL weight 10.
- Three RGB cameras at 240x240; R1Pro state plus one task category; 23-D actions.
- FP32 weights and AdamW state (fused CUDA AdamW), learning rate and backbone learning rate `1e-5`, weight decay `1e-4`; TF32 tensor-core matmuls (`--matmul-precision high`), no autocast.
- Physical batch **1,560**, no gradient accumulation. Loader slices of 128 are reassembled in original order on the GPU before one optimizer update.
- GPU 2, CPU affinity **60-89**. Since 2026-09-17: uint8 resized-frame cache (`--frame-cache`), 8 data workers, prefetch factor 2, `--compile regions-autotune`, bit-identical channels-last stem pooling; through step 10,777 the run used native per-sample video decoding with 24 workers and prefetch factor 1. The host was re-provisioned later on 2026-09-17 (new GPU UUIDs; the recipe defaults to the new GPU 2, `GPU-10567c56-9603-b2aa-1ce1-63234ee50192`, and accepts `GPU_UUID=`). The local run directory, `.venv`, frame cache and staged headers were lost in that move; the step-10000 full checkpoint was restored from the Hugging Face `resume/` path into `outputs/turning-on-radio-act-bs1560-300k-20260916/` (`latest.pt` -> `step_00010000.pt`), the dataset was re-synced (same bytes, new mtimes: the trainer now verifies exact statistics instead of refusing the changed fingerprint), and the cache was rebuilt.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` avoids allocator fragmentation at this near-capacity batch.

## Throughput (2026-09-17)

The run was paused at step 10,777 (`latest.pt` = step 10,000) to speed up the trainer without changing the model, loss, optimizer, sampling or which parameters train. Steady-state medians on the assigned GPU with the same 30 cores:

| Configuration | batch 1024 | batch 1560 | peak GPU memory (1560) |
| --- | --- | --- | --- |
| Original recipe (native decode, fp32 matmuls, 24 workers) | 1.77 s/step | 3.40 s/step (2.37 s compute + ~1 s loader wait) | 269 GiB |
| + frame cache, GPU batch assembly, channels-last, fused AdamW (fp32) | 1.47 s | — | 236 GiB |
| + TF32 matmuls | 0.60 s | 0.93 s | 236 GiB |
| + unused decoder layers skipped (bitwise identical) | 0.41 s | 0.65 s | 187 GiB |
| + `--compile regions` | 0.32 s | 0.48 s | 162 GiB |
| + bit-identical channels-last max-pool kernels | 0.31 s | 0.47 s | 150 GiB |
| + `--compile regions-autotune` (GEMM/convolution kernel autotuning; **current recipe**) | **0.30 s** | **0.45 s** (3,500 samples/s) | 150 GiB |
| opt-in `--autocast bf16-backbone` on top (bf16 only inside the ResNet bodies) | 0.28 s | 0.42 s | 139 GiB |
| opt-in `--autocast bf16` instead of TF32 (not used: shifts L1 by +0.6–1.0 %) | 0.29 s | 0.44 s | 104 GiB |

Loader wait is below 10 ms per step in every cached configuration. Validation: cached frames are byte-identical to native decoding (1,536 of 1,536 sampled frames) up to the documented half-LSB rounding of the resize; skipping the discarded decoder layers gives bitwise-identical predictions, gradients and optimizer trajectories; the step-10000 checkpoint resumed through the new pipeline reproduces the original run's per-step L1 within ±0.3 % (the dropout-RNG noise floor, identical to what the untouched native path shows) under fp32 and TF32; a 150-step batch-1560 run with saves/exports and a resume from its step-100 checkpoint reproduced the uninterrupted losses exactly. Over 300 resumed steps from step 10000, the mean per-step L1 deviation from the original run is +0.040 % (TF32 eager), +0.046 % (TF32 + `--compile regions`), +0.082 % (`--autocast bf16-backbone`) and +0.77 % (full `--autocast bf16`), with KL and gradient norms indistinguishable except under full bf16 (+25 % gradient norm). Details: `/tmp/dev/audits/act-speed-20260917/` (benchmarks, `numerics-b256.json`, resume comparisons).

Measured but not adopted: PyTorch's multi-tensor AdamW and the fused kernel take the same 2.55 ms per step for ACT's 263 parameter tensors (either is fine; the recipe keeps fused), and CUDA graphs for the compiled regions (`reduce-overhead`) gave 0.293 vs 0.297 s/step at batch 1024, within noise, for extra allocator complexity. On the re-provisioned host the recipe measured 0.297 s/step (batch 1024) and 0.440 s/step (batch 1560), with the restored step-10000 checkpoint resuming to the same losses as before.

At 0.45 s/step the remaining 289,223 steps take about 36 hours instead of roughly 12 days. Compiling for a new shape set costs about three minutes once (Inductor caches under `/tmp/.cache/torchinductor`).

The node has a 130-CPU quota. Two other runs were budgeted 30 cores each; ACT and DP each get 30, while uploaders use cores 120-123. These are process-affinity limits, not an exclusive system reservation of CPUs.

## Batch qualification

Repeated-real-sample model probes passed batches 256, 1408, 1536, 1552 and 1560; 1568 and 1600 ran out of memory. Fresh-data validation at batch 1560 completed six steps with W&B online, exact statistics, checkpoint saves and eval exports, followed by resume through step 22. Steady compute was approximately 2.4 seconds/step; measured data waits added roughly 0.3-2.3 seconds with the shared CPU budget.

The first detached run exhausted fragmented CUDA memory at step 7. Its step-1 checkpoint had already been uploaded. The launch recipe was promptly committed/pushed with expandable CUDA segments and resumed that checkpoint with the same W&B identity. This changes memory allocation, not the model, batch, or optimizer math. Batch 1560 is the largest tested stable aligned batch for this recipe, not a guarantee that every one-sample increment or future runtime condition was exhaustively tested.

Detailed probe and live verification artifacts are outside git at `/tmp/dev/audits/act-dp-radio-300k-20260916/`.

## Detached processes

Dedicated tmux server socket name: **`b1k-act-dp`**.

```bash
source /tmp/dev/env.sh
tmux -L b1k-act-dp list-sessions
tmux -L b1k-act-dp attach -t act-radio-train
tmux -L b1k-act-dp attach -t act-radio-upload
```

Launch recipes (run in separate tmux sessions, trainer first):

```bash
tmux -L b1k-act-dp new-session -d -s act-radio-train \
  'bash /tmp/dev/baselines/act/scripts/b1k/run_radio_300k.sh'
tmux -L b1k-act-dp new-session -d -s act-radio-upload \
  'bash /tmp/dev/baselines/act/scripts/b1k/upload_radio_300k.sh'
```

The training recipe automatically resumes `latest.pt` if present. It first (re)builds and spot-checks the frame cache (a no-op once complete, ~8 minutes from scratch) and the first step after a restart includes a few minutes of `torch.compile` time (cached on disk under `/tmp/.cache/torchinductor` afterwards). Its header documents the overrides: `BATCH_SIZE`, `RUN_TAG` (any tag other than `20260916` starts a fresh run directory/log/W&B run instead of resuming this one), `AUTOCAST` (`none` = the TF32 recipe; `bf16-backbone` and `bf16` are the faster, measured, non-default variants), `COMPILE_MODE`, `GPU_UUID`, `CORES`, `FRAME_CACHE`, `WANDB_ID`. Environment: `scripts/b1k/setup_venv.sh` recreates `.venv` from `requirements-b1k.lock.txt` and stages Triton's Python headers. Before deliberately restarting an exited job, inspect and archive its `.exit` file; never start a second trainer/uploader for the same run. GPU occupancy checks and file locks reject overlapping jobs. Tmux survives the Grok session ending, but not a machine/container termination.

## Checkpoints and cloud destinations

- Run directory: `outputs/turning-on-radio-act-bs1560-300k-20260916/`.
- Full saves: step 1, then every **2,500 steps**, retaining the newest **three local full checkpoints**.
- Eval-only exports: every **10,000 steps**, retaining all 30 scheduled snapshots on the Hub. They contain model/config/normalization but no optimizer or RNG state.
- Private Hugging Face repository: https://huggingface.co/kmy17518/b1k-act-turning-on-radio-20260916
- Online W&B: https://wandb.ai/kmy17518/b1k-challenge-2026-act/runs/actradio16
- Uploader staging/journal: `/tmp/dev/hf-staging/act-radio-300k-20260916/`.
- Logs: `/tmp/dev/logs/act-radio-300k-20260916.log` and `/tmp/dev/logs/act-radio-upload-20260916.log`.
- Exit statuses: corresponding `.exit` files appear only after those processes exit.

The single uploader manages both `eval/step-XXXXXXXX.pt` and `resume/step-XXXXXXXX.pt`. Each replacement atomically publishes the new full checkpoint and removes the old path. It then permanently removes only the journal-recorded old-full LFS object after validating the dedicated repo's ownership marker, main-only refs, complete live tree, and candidate hashes. Eval/current objects are excluded. A live scratch-repository test verified actual quota-object removal and retained-eval download integrity. Initial actual step-1 upload was verified by local/remote SHA-256.

**Keep the staging journal.** Adding branches, tags, pull-request refs, or foreign files to this dedicated checkpoint repository makes the uploader stop safely rather than risk deleting data. Network errors retry; integrity/ownership failures exit 2 and require inspection. The uploader exits successfully after step 300000 and all scheduled evals are verified.

Status command:

```bash
source /tmp/dev/env.sh
CUDA_VISIBLE_DEVICES='' /tmp/dev/baselines/act/.venv/bin/python /tmp/dev/scripts/act-dp-radio-status.py
```

Training is running, not completed. No simulator success measurement is claimed; BEHAVIOR evaluation still requires an RTX-capable host.
