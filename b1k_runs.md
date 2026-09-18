# ACT radio training run — 2026-09-16

## MT-ACT reproduction at 96 px — 2026-09-18

Branch `mt-act` (worktree `/tmp/dev/baselines/mt-act`, own `.venv`, branched from `lang`) adds `--language-conditioning mt_act` (see b1k.md "Optional MT-ACT reproduction"). One run, launched from this commit with `scripts/b1k/run_radio_mt_act_300k.sh`, on the same batch (**1,560**), optimizer, transformer, sampler seed 0, 300,000-step schedule and checkpoint cadence as the `opt20260917` runs, to be **stopped at 5,000 steps** like them:

| Run | Model | Images | GPU | cores | W&B run | log/exit |
| --- | --- | --- | --- | --- | --- | --- |
| `outputs/turning-on-radio-mt-act-96px-bs1560-300k-opt20260917` | MT-ACT: MiniLM task description → `proj_text_emb` → residual-branch FiLM (stages 2–4) + encoder token; style encoder over actions; no one-hot; ResNet-18 from scratch with BatchNorm | **96×96** (Diffusion Policy's resolution; new cache `2026-challenge-demos-act-frame-cache-96x96`, verified against native decoding) | 1 (`GPU-82d44829-…`) | 0-29 | `actradio-mtact-opt20260917` | `/tmp/dev/logs/act-radio-mt-act-300k-opt20260917.{log,exit}` |

Kept from our recipe rather than RoboAgent's: three cameras, 25-D proprioception, 23-D actions, chunk 100 (MT-ACT used four cameras, 8-D actions and H=20 on 42-step trajectories), no random crop (Diffusion Policy crops 96→86; ACT does not crop). The prompt is the dataset instruction *"Turn on the radio receiver that's on the table in the living room."* With one task the language vector is constant, so — as for the CLIP runs — this measures the MT-ACT architecture (residual FiLM affine, extra token, from-scratch BatchNorm ResNet at 96 px), not language grounding.

Probes (GPU 1, batch 1,560, 25 steps, steady-state median): eager **0.20 s/step**, 74 GiB peak; `--compile regions-autotune` **0.16 s/step**, 61 GiB peak, 134 s first-step compile; loader wait ~1.4 ms. Step-1–3 L1 compiled vs eager 0.8352/0.7532/0.7011 vs 0.8352/0.7537/0.7020. CPU suite: **121 passed, 5 skipped**; the cached real-MiniLM check reproduces RoboAgent's shipped `TEXT_EMBEDDINGS` to 1.3e-7.

```bash
source /tmp/dev/env.sh
tmux -L b1k-mt-act new-session -d -s mt-act-radio 'bash /tmp/dev/baselines/mt-act/scripts/b1k/run_radio_mt_act_300k.sh'
tmux -L b1k-mt-act new-session -d -s mt-act-stop5k \
  'ACT_STOP_RUN_DIR=/tmp/dev/baselines/mt-act/outputs/turning-on-radio-mt-act-96px-bs1560-300k-opt20260917 ACT_STOP_EXIT_FILE=/tmp/dev/logs/act-radio-mt-act-300k-opt20260917.exit /tmp/dev/scripts/act-stop-run-at-step.sh'
```

**Launched 2026-09-18 08:50 UTC from commit `a208ea5`, stopped at step 5,028 at 09:04 UTC** by the watcher (SIGINT after `step_00005000.pt` was saved; exit status 130, `latest.pt -> step_00005000.pt`, full checkpoints at steps 1 / 2,500 / 5,000, W&B run finished). Steady state **0.160 s/step**, 61 GiB peak, 0.23 h to 5,000 steps (vs 0.45 s/step and 0.63 h for the 240 px runs). Training-loss comparison with the three other `opt20260917` runs — same batch sequence (seed 0), mean over steps 4,901–5,000; lower is better:

| Run | L1 | KL | loss | L1 (2,401–2,500) | L1 (901–1,000) | s/step | peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MT-ACT reproduction, 96 px, from scratch | **0.1135** | 0.0054 | **0.1674** | 0.1509 | 0.1934 | 0.160 | 61 |
| unconditioned `opt20260917`, 240 px, ImageNet init | 0.1165 | 0.0057 | 0.1738 | 0.1549 | 0.2006 | 0.447 | 150 |
| clip_film identity init, 240 px | 0.1235 | 0.0047 | 0.1700 | 0.1606 | 0.2028 | 0.455 | 165 |
| clip_film random init, 240 px | 0.1366 | 0.0047 | 0.1831 | 0.1689 | 0.2061 | 0.455 | 165 |

The MT-ACT variant fits the training actions slightly faster than the pretrained frozen-BN baseline despite the 6.25× smaller images and the from-scratch ResNet (three confounded differences at once: architecture, trainable BatchNorm, resolution). These are training losses at 1.7 % of the schedule with different image resolutions, not held-out or simulator results; no convergence or success claim. Resume with `bash scripts/b1k/run_radio_mt_act_300k.sh` after archiving the `.exit` file.

### 240 px (the ACT default) and the precision options — 2026-09-18

Three more MT-ACT runs at **240×240** (`ACT_IMAGE_SIZE=240`; the launch script now gives non-96 px sizes size-qualified log/exit/W&B names), same batch 1,560, seed, schedule and 5,000-step stop, differing only in `ACT_AUTOCAST`. All three were stopped by `/tmp/dev/scripts/act-stop-run-at-step.sh` after `step_00005000.pt` was saved (exit 130, `latest.pt -> step_00005000.pt`, W&B runs finished):

| Run (`outputs/turning-on-radio-mt-act-240px-bs1560-300k-<tag>`) | `ACT_AUTOCAST` | GPU / cores | W&B id | stopped after |
| --- | --- | --- | --- | --- |
| `opt20260917` | `none` (TF32) | 1 / 0-29 | `actradio-mtact-240px-opt20260917` | step 5,003 |
| `opt20260917-bf16bb` | `bf16-backbone` | 2 / 30-59 | `actradio-mtact-240px-opt20260917-bf16bb` | step 5,020 |
| `opt20260917-bf16` | `bf16` | 3 / 60-89 | `actradio-mtact-240px-opt20260917-bf16` | step 5,000 |

Logs/exit files: `/tmp/dev/logs/act-radio-mt-act-240px-300k-<tag>.{log,exit}`. Step-5,000 comparison (identical batch sequence; mean over steps 4,901–5,000; lower is better):

| Run | L1 | KL | loss | grad norm | s/step | peak GiB | h to 5k |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MT-ACT 240 px, TF32 | 0.1234 | 0.0054 | 0.1773 | 2.81 | 0.499 | 196 | 0.70 |
| MT-ACT 240 px, `bf16-backbone` | 0.1235 (+0.08 %) | 0.0054 | 0.1774 | 2.86 | **0.427** | 156 | 0.60 |
| MT-ACT 240 px, `bf16` | 0.1240 (+0.5 %) | 0.0055 | 0.1789 | 2.89 | **0.337** | 103 | 0.47 |
| MT-ACT 96 px, TF32 (above) | 0.1135 | 0.0054 | 0.1674 | 1.98 | 0.160 | 61 | 0.23 |
| unconditioned `opt20260917`, 240 px, TF32 | 0.1165 | 0.0057 | 0.1738 | 2.36 | 0.447 | 150 | 0.63 |
| clip_film identity init, 240 px, TF32 | 0.1235 | 0.0047 | 0.1700 | 2.44 | 0.455 | 165 | 0.64 |

Where the time goes (b1k.md "Throughput of the MT-ACT architecture"): the MT-ACT modules themselves are free; trainable BatchNorm costs +0.05 s/step and +40 GiB over frozen BatchNorm (the same network with `--backbone-norm frozen` runs at 0.452 s/step — a speed-only diagnostic). Both precision options recover that and more: `bf16-backbone` is 14 % faster at an L1 deviation inside the ±0.3 % dropout noise floor measured for the CLIP runs, `bf16` is 33 % faster at +0.5 % L1. At 240 px the from-scratch MT-ACT trails the ImageNet-initialized baseline at this early point (0.1234 vs 0.1165) and its own 96 px run (0.1135); the smaller images also let it run 3× faster. Training losses only; no convergence or simulator claim.

## Optimized CLIP/FiLM runs, random vs identity initialization — 2026-09-17

The `lang` branch (this checkout, worktree `/tmp/dev/baselines/act-lang` with its own `.venv`) merged the `my` branch throughput work (frame cache, TF32, channels-last, fused AdamW, Triton stem pooling, skipped unused decoder layers, GPU batch assembly, `--compile regions-autotune`; see "Throughput (2026-09-17)" below). The merge added `--film-init random|identity` (saved as `model_config['film_init']`) and the runtime `--film-recompute/--no-film-recompute` switch; the FiLM layers are part of the compiled backbone region. CPU suite after the merge: **118 passed, 4 skipped** (`tests/`).

Two 300,000-step runs launched on 2026-09-18 from this commit, identical to the unconditioned `outputs/turning-on-radio-act-bs1560-300k-opt20260917` recipe (batch **1,560**, TF32, fp32 weights/optimizer, same architecture, images, sampler seed 0 and checkpoint schedule) plus `--language-conditioning clip_film --prompt-source task_name`, which is the comparison they are meant for:

| Run | FiLM init | GPU | cores | W&B run | log/exit |
| --- | --- | --- | --- | --- | --- |
| `outputs/turning-on-radio-act-clipfilm-taskname-random-bs1560-300k-opt20260917` | `random` (nn.Linear default) | 1 (`GPU-82d44829-7cec-7d3c-9918-6dc1321320d4`) | 0-29 | `actradioclip-random-opt20260917` | `/tmp/dev/logs/act-radio-clipfilm-random-300k-opt20260917.{log,exit}` |
| `outputs/turning-on-radio-act-clipfilm-taskname-identity-bs1560-300k-opt20260917` | `identity` (zero projection) | 3 (`GPU-3b6edd76-2c6a-3087-f7f1-964db3c27635`) | 30-59 | `actradioclip-identity-opt20260917` | `/tmp/dev/logs/act-radio-clipfilm-identity-300k-opt20260917.{log,exit}` |

Both use W&B project `b1k-challenge-2026-act` (entity `kmy17518`), the run directory name as the W&B name, and no uploader (checkpoints stay local: three newest full saves plus eval exports every 10,000 steps). With the shared seed 0 the two conditioned runs start from byte-identical non-FiLM weights and draw the same batch sequence; only the FiLM projections differ (default vs zero). Their weights are not byte-identical to the unconditioned run (constructing the FiLM projections consumes RNG before the transformer is initialized); the byte-paired three-arm comparison is the controlled diagnostic below.

```bash
source /tmp/dev/env.sh
tmux -L b1k-act-lang new-session -d -s act-radio-lang-random \
  'ACT_FILM_INIT=random bash /tmp/dev/baselines/act-lang/scripts/b1k/run_radio_language_300k.sh'
tmux -L b1k-act-lang new-session -d -s act-radio-lang-identity \
  'ACT_FILM_INIT=identity bash /tmp/dev/baselines/act-lang/scripts/b1k/run_radio_language_300k.sh'
```

Restart rule as before: a run resumes its own `latest.pt`; archive its `.exit` file first, and never start a second trainer for the same directory (the GPU occupancy check and `run.lock` reject overlaps).

**Stopped at 5,000 steps on the user's request (2026-09-18 01:00–01:02 UTC).** `/tmp/dev/scripts/act-lang-stop-at-step.sh` (tmux `act-radio-lang-stop5k`, log `/tmp/dev/logs/act-lang-stop-at-5000.log`) sent SIGINT once each run had logged step ≥ 5,000 with `step_00005000.pt` saved: random stopped after step 5,019, identity after 5,003, both exit status 130, `latest.pt -> step_00005000.pt`, local full checkpoints at steps 1 / 2,500 / 5,000, W&B runs finished. The unconditioned `opt20260917` run was stopped at step 5,000 as well (00:29 UTC, exit 130), so all three have step-5,000 checkpoints. Training-loss comparison on the identical batch sequence (seed 0; dropout differs), mean over steps 4,901–5,000:

| Run | L1 | KL | loss | L1 (2,401–2,500) | median s/step |
| --- | --- | --- | --- | --- | --- |
| unconditioned `opt20260917` | **0.1165** | 0.0057 | 0.1738 | 0.1549 | 0.447 |
| clip_film, identity FiLM init | 0.1235 | 0.0047 | 0.1700 | 0.1606 | 0.455 |
| clip_film, random FiLM init | 0.1366 | 0.0047 | 0.1831 | 0.1689 | 0.455 |

Each run took 0.64 h for 5,000 steps. These are training losses at 1.7 % of the planned schedule, not held-out or simulator results; no convergence claim. Resume with `ACT_FILM_INIT=<init> bash scripts/b1k/run_radio_language_300k.sh` after archiving the `.exit` file.

**Throughput of the conditioned trainer** (batch 1,560, 30 cores, 25-step probes on GPU 1, steady-state median `timing/step_s`, peak allocated memory):

| Configuration | s/step | peak GPU memory |
| --- | --- | --- |
| unconditioned recipe (`opt20260917` run, for reference) | 0.45 s | 150 GiB |
| clip_film, eager, `--film-recompute` (the original language recipe's memory behaviour) | 0.78 s | 165 GiB |
| clip_film, eager, `--no-film-recompute` | 0.66 s | 196 GiB |
| clip_film, `--compile regions-autotune`, `--film-recompute` | 0.48 s | 135 GiB |
| clip_film, `--compile regions-autotune`, `--no-film-recompute` (**launched configuration**) | **0.45 s** | 165 GiB |

The FiLM layers themselves are nearly free once compiled (the eight `Linear(768, 2C)` projections and one fused affine+ReLU per block); the activation recomputation that the original language recipe needed for memory costs ~7 % per step and is no longer necessary at this batch (165 of 284 GiB). Loader wait stays below 10 ms. At 0.45 s/step, 300,000 steps take about 37.5 hours per run. Probe logs: `/tmp/act-lang-probe/*.log` (outside git).

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

For the user's subsequent **one-hour limit**, the primary paired ACT/DP comparison uses the common completed **step500** endpoint. ACT steps401–500 training L1: baseline **0.265343**, random FiLM **0.254325** (-4.15%), identity FiLM **0.264750** (-0.22%). Held-out zero-latent inference L1 at step500: **0.216195 / 0.190886 / 0.211755**, respectively. Random FiLM improves held-out L1 by11.71% and identity by2.05% versus baseline. The completed ACT1000 results above are supplemental; all training/monitoring is stopped. Saved analysis: `/tmp/dev/audits/act-dp-identity-init-20260917/comparison-results.{json,md}` and `comparison-curves.png`.

## Original unconditioned run

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

Dedicated tmux server socket name: **`b1k-act`** (the original 2026-09-16 launches used `b1k-act-dp`, now shared with a DP run whose global environment must not leak into ACT launches).

```bash
source /tmp/dev/env.sh
tmux -L b1k-act list-sessions
tmux -L b1k-act attach -t act-radio-train
tmux -L b1k-act attach -t act-radio-upload
```

Launch recipes (run in separate tmux sessions, trainer first):

```bash
tmux -L b1k-act new-session -d -s act-radio-train \
  'bash /tmp/dev/baselines/act/scripts/b1k/run_radio_300k.sh'
tmux -L b1k-act new-session -d -s act-radio-upload \
  'bash /tmp/dev/baselines/act/scripts/b1k/upload_radio_300k.sh'
```

The training recipe automatically resumes `latest.pt` if present. It first (re)builds and spot-checks the frame cache (a no-op once complete, ~8 minutes from scratch) and the first step after a restart includes a few minutes of `torch.compile` time (cached on disk under `/tmp/.cache/torchinductor` afterwards). Its header documents the overrides, all prefixed `ACT_` so a tmux server shared with the Diffusion Policy recipe cannot redirect the run: `ACT_BATCH_SIZE`, `ACT_RUN_TAG` (any tag other than `20260916` starts a fresh run directory/log/W&B run instead of resuming this one), `ACT_AUTOCAST` (`none` = the TF32 recipe; `bf16-backbone` and `bf16` are the faster, measured, non-default variants), `ACT_COMPILE_MODE`, `ACT_GPU_UUID`, `ACT_CORES`, `ACT_FRAME_CACHE`, `ACT_WANDB_ID`. Use the dedicated tmux server `b1k-act` (not the `b1k-act-dp` server the DP recipe launched with its own environment). Environment: `scripts/b1k/setup_venv.sh` recreates `.venv` from `requirements-b1k.lock.txt` and stages Triton's Python headers. Before deliberately restarting an exited job, inspect and archive its `.exit` file; never start a second trainer/uploader for the same run. GPU occupancy checks and file locks reject overlapping jobs. Tmux survives the Grok session ending, but not a machine/container termination.

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
