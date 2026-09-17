# ACT radio training run — 2026-09-16

## Task-name CLIP/FiLM run

**Stopped by user on 2026-09-17 at 05:15 UTC.** The trainer stopped after recorded step **10,656**; the latest resumable/full and eval checkpoint is **step 10,000**. Trainer, uploader, scheduled health reviews and failure watcher are stopped. Checkpoints and upload journals remain intact. The historical launch/monitoring statements below describe the earlier running state; do not restart this run without a new request.

The language-conditioned reproduction uses the same radio subset, **300,000 steps**, physical batch **1,560**, FP32 optimizer, architecture, images, CPU affinity and checkpoint schedule below, with `--language-conditioning clip_film --prompt-source task_name`. The original one-hot task input remains. Frozen CLIP ViT-L/14 projected text embeddings condition every ResNet residual block through FiLM. Conditioned blocks use activation recomputation to accommodate the original physical batch; this is not gradient accumulation or mixed precision.

The new run has separate local, Hugging Face and W&B identities; it never resumes or replaces the original run:

- Run directory: `outputs/turning-on-radio-act-clipfilm-taskname-bs1560-300k-20260916/`.
- Private Hugging Face destination: https://huggingface.co/kmy17518/b1k-act-turning-on-radio-clipfilm-taskname-20260916
- W&B project is unchanged; new run: https://wandb.ai/kmy17518/b1k-challenge-2026-act/runs/actradioclipname16
- W&B experiment: `turning-on-radio-act-clipfilm-taskname-bs1560-300k`.
- GPU 2 on the current node: `GPU-fe99daa5-20f4-4f6f-9792-14a1fd4091a1` (the historical GPU UUID below belongs to the prior node).
- Trainer log/exit: `/tmp/dev/logs/act-radio-clipfilm-taskname-300k-20260916.{log,exit}`.
- Uploader log/exit: `/tmp/dev/logs/act-radio-clipfilm-taskname-upload-20260916.{log,exit}`.
- Durable uploader journal: `/tmp/dev/hf-staging/act-radio-clipfilm-taskname-300k-20260916/`.

Launch the trainer first, then wait for its first `latest.pt` checkpoint before starting the uploader. The new scripts reject occupied GPUs and existing exit files; preserve the journal and archive an old exit file before a deliberate restart. Shell exit traps also record startup failures.

```bash
source /tmp/dev/env.sh
tmux -L b1k-act-dp-language new-session -d -s act-radio-language-train \
  'bash /tmp/dev/baselines/act/scripts/b1k/run_radio_language_300k.sh'
tmux -L b1k-act-dp-language new-session -d -s act-radio-language-upload \
  'bash /tmp/dev/baselines/act/scripts/b1k/upload_radio_language_300k.sh'
CUDA_VISIBLE_DEVICES='' /tmp/dev/baselines/act/.venv/bin/python \
  /tmp/dev/scripts/act-dp-language-status.py
```

Qualification passed **16 fresh-data optimizer steps at batch 1,560**, then a full-checkpoint resume through **step 20 with Hugging Face/transformers offline**. Exact statistics covered all **429,928 frames / 200 episodes**. Peak allocated memory was **253.39 GiB**; median steady training compute was **2.573 s/step**, plus data wait. Three local full checkpoints and two eval exports were verified. Real step-16 eval serving passed eight websocket action requests covering reset, replanning, batching and separate clients with dataset/CLIP access denied. All 100 task-name and description embeddings matched the DP implementation exactly; the five long descriptions were chunked without truncation.

The 300,000-step trainer was launched on **2026-09-16 at 18:11 UTC** from commit `8deea34` using the new identities above. It is a fresh run, not the qualification checkpoint. The uploader published `resume/step-00000001.pt` to the new private repo; local SHA-256 and remote LFS SHA-256 matched (`2086b50e28638ed67349d88ba0348ea2202d94b5b89df45737820c035c4bb985`). The original checkpoint repo was untouched. A session monitor checks both trainers/uploaders every 30 seconds, and a durable 10-minute health review checks loss, timings and publication status. Training is **in progress**, not complete; no simulator success rate is claimed. Local qualification and monitoring artifacts use `/tmp/dev/audits/act-dp-language-20260916/`. Tmux survives client disconnects, not machine/container termination.

## Controlled short initialization comparison — 2026-09-17

A separate diagnostic uses the full ACT architecture above with **1,000 steps per arm**, physical batch **128**, and seed **0**. Three arms share byte-identical initial parameters/buffers: unconditioned baseline, randomly initialized FiLM, and identity-initialized FiLM (`beta=gamma=0`). Each receives the same minibatch and paired dropout/posterior random draws. Every tenth sorted episode is held out: **180 train / 20 held-out episodes**; exact normalization uses training episodes only. Evaluation uses 128 fixed held-out examples with the inference-time zero latent.

This is intentionally a smaller-batch, single-seed optimization experiment, not directly comparable at equal steps to the stopped batch-1,560 run. Initial full-GPU baseline/identity outputs and losses matched exactly. The harness, CPU tests and full-architecture GPU smoke passed before launch (harness commit `56e6239`).

```bash
source /tmp/dev/env.sh
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True WANDB_BASE_URL=https://api.wandb.ai \
  taskset -c 60-89 .venv/bin/python scripts/b1k/compare_language_init.py \
  --output-dir /tmp/dev/audits/act-dp-identity-init-20260917/act-1000 \
  --device cuda:0 --max-steps 1000 --batch-size 128 --num-workers 8 \
  --eval-samples 128 --eval-batch-size 16 --eval-every 250 \
  --wandb-mode online --wandb-entity kmy17518 --wandb-project b1k-challenge-2026-act \
  --wandb-name act-init-20260917 --wandb-group act-init-20260917
```

Use a new output directory for another experiment; the harness rejects overwrite/resume. W&B runs: baseline [`a10bae591455`](https://wandb.ai/kmy17518/b1k-challenge-2026-act/runs/a10bae591455), random FiLM [`30123523e55f`](https://wandb.ai/kmy17518/b1k-challenge-2026-act/runs/30123523e55f), identity FiLM [`42308356b2b5`](https://wandb.ai/kmy17518/b1k-challenge-2026-act/runs/42308356b2b5). These are finite local diagnostics; there is no HF uploader or recurring monitor.

**Completed all 1,000 steps in about 19.6 minutes; all three final checkpoints and finished W&B runs verified.** Steps 901–1,000 mean training L1: baseline **0.229200**, random FiLM **0.227419** (-0.78%), identity FiLM **0.229427** (+0.10%). Step-1,000 held-out zero-latent inference L1: baseline **0.190817**, random FiLM **0.176195** (-7.66%), identity FiLM **0.189424** (-0.73%). Lower is better. Identity initialization did not improve final training L1 in this diagnostic; random FiLM's early penalty reversed. This weakens, but does not conclusively reject, random FiLM initialization as the explanation for the original large-batch gap. The diagnostic changed batch and train/held-out split and has only one seed; no simulator success or long-run convergence conclusion is established.

## Original unconditioned run

## Configuration

- Dataset: `/tmp/dev/datasets/2026-challenge-demos`, **turning_on_radio only**, 200 episodes / 429,928 frames; exact full-task statistics.
- Target: **300,000 optimizer steps**.
- Policy: upstream ACT, ImageNet-initialized ResNet18, sine positions, post-norm, hidden 512, feedforward 3200, 4 encoder / 7 decoder layers, 8 heads, action chunk 100, KL weight 10.
- Three RGB cameras at 240x240; R1Pro state plus one task category; 23-D actions.
- FP32 AdamW, learning rate and backbone learning rate `1e-5`, weight decay `1e-4`.
- Physical batch **1,560**, no gradient accumulation. Loader slices of 128 are reassembled in original order before one optimizer update.
- GPU 2 (`GPU-aa99f910-8e39-d04c-a717-a6f7a06f52e8`), CPU affinity **60-89**, 24 data workers, one thread per worker, two main-process torch threads, prefetch factor 1.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` avoids allocator fragmentation at this near-capacity batch.

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

The training recipe automatically resumes `latest.pt` if present. Before deliberately restarting an exited job, inspect and archive its `.exit` file; never start a second trainer/uploader for the same run. GPU occupancy checks and file locks reject overlapping jobs. Tmux survives the Grok session ending, but not a machine/container termination.

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
