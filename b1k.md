# ACT and CNNMLP on BEHAVIOR-1K (native LeRobot v3)

This adapter trains both upstream policies in `policy.py` and `detr/models/detr_vae.py`:

- **ACT** (default): shared ResNet18 camera backbone, CVAE action encoder, latent dimension 32, transformer action decoder, masked L1 plus KL.
- **CNNMLP**: one ResNet18 per camera, three unpadded 5×5 convolution projections (128→64→32 channels), flattened camera features concatenated with state, two 1024-wide ReLU MLP layers, and single-action MSE. This is the actual upstream `CNNMLPPolicy`/`CNNMLP`, not ACT behind a renamed wrapper.

Legacy HDF5 entrypoints, 14-dimensional defaults, CNNMLP parameter names (including its unused `action_head`), and original 480×640 CNNMLP topology remain compatible. B1K version-1 checkpoints without `model_config.policy_class` still dispatch to ACT. New checkpoints record `ACT` or `CNNMLP`, and restore/serving use that saved selection.

## Install

Python 3.10 is verified. Keep environments and caches local; no simulator, LeRobot conversion, or Hugging Face download is required.

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
uv venv --python /usr/bin/python3.10 .venv
uv pip install --python .venv/bin/python torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements-b1k.txt
```

The CUDA 13 wheels above work on this aarch64 GB300 host. On other machines choose a matching torch/torchvision CPU or CUDA wheel pair. This does not modify the GR00T/OpenPI environments. ACT's default ImageNet ResNet18 initialization downloads a ~45 MB weight file to `$TORCH_HOME` once. `--no-pretrained-backbone` disables that initialization; checkpoint restore/serving never downloads backbone weights.

On this ARM host, `uv pip check` reports a platform warning for `nvidia-cusparselt-cu13==0.8.0`: its wheel declares the nonstandard `manylinux2014_sbsa` tag. The installed shared library is an aarch64 binary and loads successfully; the GPU training and serving checks below passed. The dependency check therefore has this known warning rather than a clean result.

All shell examples in this workspace start by sourcing `/tmp/dev/env.sh`, which redirects package/model caches under `/tmp`. The isolated environment is `.venv`; outputs are ignored by git. No editable `detr` installation is needed for the B1K scripts.

## Dataset and sample semantics

Point directly at `/tmp/dev/datasets/2026-challenge-demos`, or a partial local root containing:

- `meta/info.json`, `meta/tasks.parquet`, and `meta/episodes/chunk-*/file-*.parquet`;
- selected `data/chunk-*/file-*.parquet` files;
- the three selected RGB streams under `videos/observation.rgb.<camera>_camera_0/`.

Depth videos are not required or decoded. The loader never writes to the dataset, converts episodes, downloads missing files, loads a full-frame index, or scans all videos. Startup reads projected episode metadata and checks referenced file existence. It indexes actual `episode_index` values, including nonzero/noncontiguous IDs. Episodes missing any required Parquet/RGB file are excluded with a count in the log. An unknown task or any explicitly requested task with no complete local episodes fails clearly. Omitting `--task-names` means **all complete local tasks**, not all tasks listed in global metadata. Metadata may provide `task_index` directly or a single known task name in `tasks`.

For each sample, only intersecting Parquet row groups are read into a bounded LRU (default two per worker). It seeks each camera to **row timestamp + that camera's episode `from_timestamp`** and decodes from the preceding keyframe. Three video containers are cached per worker; videos are never decoded in full. Frame matching tolerance is 8 ms to accommodate long packed-file float timestamps without accepting a whole-frame error at 30 Hz. Metadata/data episode, frame, task, and absolute-index consistency is checked. An interrupted/corrupt local file fails rather than triggering a download.

### R1Pro mapping and preprocessing

- Input proprio: `observation.state` / serving `robot_r1::proprio`, 61 dimensions.
- Selected 25 dimensions, in order: `[0:3, 53:57, 3:10, 24:26, 28:35, 49:51]`.
- A one-hot vector is appended **after** continuous-state normalization, ordered by sorted checkpoint task IDs (including one category for single-task training). Model state dimension is `25 + number_of_tasks`.
- Action: all 23 dataset dimensions, retaining original base velocity / absolute joint / gripper command semantics. **No additional delta transformation**, clipping, or gripper threshold is applied.
- Camera order: head (`zed_link`), left wrist, right wrist. RGB uint8 becomes float CHW `/255`; RGBA drops alpha. Bilinear antialiased resize defaults to **240×240 for ACT** and **480×640 for CNNMLP**, then the selected upstream policy applies ImageNet mean/std normalization. Training and serving call the same preprocessing functions.
- CNNMLP preserves the original unpadded convolutions: a 480×640 image produces 15×20 ResNet18 features, then 3×8×32 = 768 flattened values per camera. A configured image size changes the MLP input width using the true spatial result, not adaptive pooling or a replacement CNN. Both dimensions must be at least 385 pixels for the stride-32 backbone; 240×240 is rejected. The original two 1024-wide hidden layers are unchanged.

Normalization streams selected state/action columns once per packed file, with fixed-size batches and no video decoding. Mean and **sample** standard deviation (`ddof=1`, as `torch.std` in `utils.get_norm_stats`) use all real selected frames; std is floored at **0.01**. Statistics are cached under the output directory with a selected-dataset fingerprint and embedded in every checkpoint. Frame-weighted statistics support variable episode lengths. Full-dataset statistics may take time on 210M frames but memory remains bounded. `--stats-max-frames N` is an explicitly labeled first-N approximation for smoke tests, not recommended for training.

ACT samples an episode uniformly, then a starting frame uniformly. A deterministic step-indexed sampler avoids a full-frame permutation and supports resume with any worker count. Unlike the upstream one-pass-without-replacement episode epoch, this step-based sampler draws episodes with replacement. Actions start at the same row as the observation, stop at that episode's boundary, and are zero-padded **before normalization** to `--chunk-size`; `is_pad` masks padding. Only this chunk is allocated rather than upstream HDF5's full episode-sized target (the upstream policy truncates it to the same chunk).

**HDF5 alignment difference:** upstream `EpisodicDataset` uses `actions[max(0,t-1):]` for `sim=False` to compensate its real recording latency. LeRobot rows are already aligned: this adapter deliberately uses `actions[t:]`, matching the upstream `sim=True` sample convention. Applying the HDF5 real-data shift here would misalign actions. Boundary tests explicitly compare both conventions. The loss retains upstream masked mean L1 (mean over all batch/chunk/action elements, not just valid elements) plus KL; no new loss normalization or default gradient clipping is introduced.

## Train and resume

**GPU safety on this host:** all GPUs are currently occupied by external GR00T jobs. Use `CUDA_VISIBLE_DEVICES='' --device cpu` for current checks. CUDA examples below are for a later explicitly reserved idle GPU only: first confirm `nvidia-smi` shows no compute processes, negligible memory use, and zero utilization for that GPU. Do not stop or share existing jobs. The variant runner performs its own idle check before importing torch for `--device cuda`.

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos \
  --task-names turning_on_radio picking_up_trash \
  --output-dir outputs/b1k-two-tasks \
  --max-steps 50000 --batch-size 8 --num-workers 4 --device cuda
```

`--dataset-root` aliases `--dataset-path`. Drop `--task-names` for all complete local tasks; multiple names are space-separated. `--policy-class ACT|CNNMLP` selects the upstream policy (default ACT). Default B1K ACT architecture matches the upstream README training recipe: ResNet18, hidden 512, feedforward 3200, 4 encoder / 7 decoder layers, 8 heads, chunk 100, KL weight 10, LR 1e-5, backbone LR 1e-5, AdamW weight decay 1e-4, dropout 0.1. The lower-level upstream parser defaults remain intact.

For CNNMLP, add `--policy-class CNNMLP`; omit `--image-size` to use the upstream 480×640 spatial path. CNNMLP always records `num_queries=1`, takes the first action target, and uses upstream MSE; `--chunk-size` and transformer/KL hyperparameters do not change it into a chunked policy. State input is 25 + task count, output is 23. ACT accepts `--position-embedding sine|learned` and `--pre-norm`/`--no-pre-norm`; defaults remain sine and post-norm. Nondefault positional/norm options are rejected for CNNMLP, which does not use transformer positions or normalization.

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-root /tmp/dev/datasets/2026-challenge-demos \
  --output-dir outputs/b1k-two-tasks \
  --resume outputs/b1k-two-tasks/step_00010000.pt \
  --max-steps 50000 --batch-size 8 --num-workers 4 --device cuda
```

`--max-steps` is the total optimizer count, not additional steps. Resume restores model, optimizer, torch/CUDA RNG, normalization, task map, preprocessing, architecture and sampler seed. Saved architecture/preprocessing are authoritative; dataset root/subset must have the same fingerprint. Worker count/device/batch size can change; exact training continuation requires the same batch size. Fresh training refuses an existing run, and snapshot paths are never silently overwritten. Resume into another output directory is supported. `latest.pt` is a relative symlink to the newest immutable `step_XXXXXXXX.pt`; copy the actual `.pt` file for deployment.

Checkpoints contain all serving and training configuration, selected task IDs/names, continuous normalization, image/camera/state mapping, action semantics, model/optimizer state, RNG and completed step. Serving does **not** need the dataset, stats JSON, `run.json`, or pretrained backbone cache. Checkpoints are loaded with `torch.load(weights_only=True)`.

Other useful knobs: `--policy-class`, `--position-embedding`, `--pre-norm`, `--chunk-size`, `--image-size HEIGHT WIDTH`, `--hidden-dim`, `--dim-feedforward`, `--enc-layers`, `--dec-layers`, `--nheads`, `--kl-weight`, `--lr`, `--save-every`, `--cache-row-groups`. See `--help`. Metrics are JSON lines in `metrics.jsonl`; every checkpoint is suitable for inference, but no validation-driven best-checkpoint selection or simulator success evaluation is performed by this trainer.

This ACT/CNNMLP integration is **single-process, single-GPU** (or CPU) only. Unlike GR00T's distributed training path, it does not implement `torchrun`/DDP; do not launch multiple training ranks against one output directory. `--num-workers` controls CPU data-loading workers, not GPU training ranks.

## Serve BEHAVIOR

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/serve_b1k.py \
  --model-path outputs/b1k-two-tasks/step_00050000.pt \
  --host 0.0.0.0 --port 8000 --device cuda --action-horizon 16
```

Serving dispatches from the checkpoint; there is no serving policy override that can accidentally load CNNMLP as ACT.

- **ACT chunked:** `--action-horizon` is the number of cached actions executed before replanning, **not** the model's predicted chunk length. It must be 1…chunk size; default is `min(16, chunk size)`, preserving the prior default for normal ACT checkpoints. Set it equal to the checkpoint chunk size for upstream full-chunk execution.
- **ACT temporal aggregation:** add `--temporal-agg` and omit `--action-horizon` (or set it to 1). Query every observation and combine all unexpired overlapping predictions, ordered **oldest to newest**, with normalized `exp(-0.01 * arange(N))`, exactly the weight orientation in `imitate_episodes.py`. This counterintuitively gives older predictions slightly more weight. Validity is explicit, not inferred from nonzero coordinates: partial-zero and all-zero actions participate. Buffers are bounded by chunk length, per connection and batch slot; task changes clear only affected slots. Aggregating denormalized actions is equivalent to upstream aggregation before affine denormalization because weights sum to one.
- **CNNMLP single action:** each observation produces one `(B,23)` action; horizon and chunk size metadata are both 1, and `execution_mode=single_action`. Aggregation and horizons above 1 are rejected. No action-chunk capability is claimed.

`--task-name NAME` provides a known default when the request omits `task_id`; an explicit request task ID is always validated and takes precedence. A single-task checkpoint infers its task when absent; a multi-task checkpoint otherwise requires `task_id`. Unseen IDs/names are rejected, never silently mapped to a trained task.

Wire protocol matches BEHAVIOR's OpenPI/GR00T NumPy MessagePack encoding (`__ndarray__`, `data`, `dtype`, `shape` byte keys):

1. Connect over WebSocket. Server immediately sends binary metadata with action dimension, chunk/replan sizes, observation keys, and task map (string keys for strict MessagePack compatibility).
2. Send a binary MessagePack observation map:
   - `robot_r1::proprio`: `(61,)` or `(B,61)` finite state.
   - `robot_r1::robot_r1:zed_link:Camera:0::rgb`
   - `robot_r1::robot_r1:left_realsense_link:Camera:0::rgb`
   - `robot_r1::robot_r1:right_realsense_link:Camera:0::rgb`
   - Camera arrays: uint8 `(H,W,3|4)` or `(B,H,W,3|4)` RGB/RGBA, batch matching proprio.
   - `task_id`: integer scalar or one integer per batch element, using original challenge IDs.
3. Receive `{"action": ndarray}` with **float32 `(B,23)`**, including `(1,23)` for unbatched inputs.
4. Send `{"reset": true}` to clear this connection's buffers. **No acknowledgement is sent.**

Model weights are shared, but queues/positions/task IDs are per connection. A batch-size change resets the batch; a task-ID change invalidates only changed slots. Invalid inputs do not commit queue updates; protocol validation errors close that connection with code 1008. Reset is connection-wide; send it between episodes even when task IDs stay unchanged. HTTP `GET /healthz` returns `200 OK` and `OK\n`. Inference is serialized in the event loop; this is a simple evaluation server, not an authenticated public service. Bind localhost or use a trusted network.

## Original ACT-only smoke (September 15, 2026)

The following is historical evidence from the earlier ACT-only integration, **not** GPU verification of the newly added CNNMLP, architecture switches, or temporal aggregation. For the current all-variant CPU matrix and pending GPU status, see the final section. Do not rerun any CUDA command while a GPU is occupied.

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_b1k.py
```

Result: **12 passed** (`outputs/pytest.log`); only upstream torchvision pretrained-argument deprecation warnings.

Tests cover lossless packed videos with differing camera offsets and nonuniform timestamps; noncontiguous episode IDs; partial roots and missing tasks; first/last frame HDF5 sample and `ddof=1` stats equivalence; real-HDF5 shift difference; no cross-episode targets; bounded caches/read-only access; upstream numerical masked loss/KL and finite gradients; bit-exact CPU uninterrupted-vs-resumed optimization; serving after hiding the dataset; strict-codec handshake, HTTP health, concurrent connections, no reset acknowledgement, replan/batch/task changes, and rejection of invalid arrays/tasks.

Executed on reserved GPU 0 (GB300), torch 2.10.0+cu130 / torchvision 0.25.0+cu130, isolated `.venv`:

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos \
  --task-names turning_on_radio picking_up_trash --output-dir outputs/smoke-default \
  --max-steps 2 --batch-size 2 --num-workers 2 --device cuda --stats-max-frames 4096 --save-every 1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-root /tmp/dev/datasets/2026-challenge-demos --output-dir outputs/smoke-default \
  --max-steps 3 --batch-size 2 --num-workers 0 --device cuda \
  --resume outputs/smoke-default/step_00000002.pt
```

Default 83.95M-parameter ACT, 240×240 images, 2 tasks/400 episodes/1,483,478 available frames; losses **80.5304 → 60.3782 → 73.8589**. The 4096-frame statistics approximation and three updates are smoke-only: **not converged and not a usable robot policy**. Logs: `outputs/smoke-default-train.log`, `outputs/smoke-default-resume.log`; checkpoint: `outputs/smoke-default/step_00000003.pt`.

A complementary small genuine ACT run uses exact full-task statistics rather than the approximation:

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/train_b1k.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos --task-names turning_on_radio \
  --output-dir outputs/smoke-small-fullstats --max-steps 2 --batch-size 2 --num-workers 0 --device cuda \
  --chunk-size 4 --image-size 64 64 --hidden-dim 64 --dim-feedforward 128 \
  --enc-layers 1 --dec-layers 1 --nheads 4 --save-every 1
```

This 11.33M-parameter run completed both steps with losses **83.9877, 85.7992** and exact statistics from all **429,928** task frames (streamed in ~0.6 seconds). Log: `outputs/smoke-small-fullstats-train.log`; checkpoint: `outputs/smoke-small-fullstats/step_00000002.pt`. It is also smoke-only and not converged.

Real network smoke (start server in a separate shell, then stop it after testing):

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/serve_b1k.py \
  --model-path outputs/smoke-default/step_00000003.pt --host 127.0.0.1 --port 18765 \
  --action-horizon 2 --device cuda
```

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/smoke_client.py \
  --dataset-path /tmp/dev/datasets/2026-challenge-demos --port 18765
```

The client passed health/metadata, real RGB request, finite float32 actions, reset without reply, deterministic replan, batch resize, per-slot task changes, two-client isolation, and unseen-task rejection. Logs: `outputs/smoke-default-serve-18765.log`, `outputs/smoke-default-network-18765.log`. The owned test server was stopped. Port 8765 was occupied; the initial bind attempt failed and was retried on 18765.

Independent review additionally matched raw samples at both ends of episodes 199, 8401 and 19692 (including long packed-video timestamps), partial chunk-042/chunk-098 roots, full-task 429,928-frame torch statistics, and the full 20,000-episode/210,916,774-frame/100-task metadata catalog. It exercised the real checkpoint with the existing OpenPI codec. Reports are outside this repository at `/tmp/dev/audits/act-diffusion-20260915/act-independent-data.json` and `act-websocket.json`. Independent legacy 14-dimensional ACT forward, backward, and inference checks also passed. No convergence or BEHAVIOR simulator success rate has been measured. Simulator evaluation is unavailable on this GB300 host: GB300 is not an RTX GPU and does not provide the RTX rendering support required by OmniGibson/BEHAVIOR. Run simulator evaluation on a supported RTX host connected to this policy server; the network smoke does not establish simulator success.

## All-variant CPU coverage (September 15, 2026)

### Inventory and scope

`imitate_episodes.py` exposes exactly two policy families: ACT and CNNMLP. ACT has full-chunk execution and temporal aggregation; the B1K server also supports shortened-horizon chunk replanning. `detr/main.py` exposes sine/learned image positions and post-/pre-norm transformer branches; these are exercised in all four ACT combinations because learned positions previously failed on raw tensors and batch sizes above one. CNNMLP does not consume these transformer options. Learning rates, widths/depths, heads, chunk lengths, and camera counts are hyperparameters, **not** additional discrete policy algorithms; this is not a Cartesian sweep over those values. Commented-out state-only construction, DETR segmentation flags, and alternative torchvision backbone choices are not additional policy families exposed by `imitate_episodes.py` and are not claimed as B1K variants here. Legacy simulation/real HDF5 alignment is covered by existing regressions, not changed into another B1K action convention.

### Reproducible runner

The checked-in `scripts/b1k/test_variants.py` runs the **same matrix** with `--device cpu` or `--device cuda`, emits per-variant `evidence.json` plus a root `summary.json`, and refuses to overwrite a nonempty output directory. It trains using the real B1K trainer and local Parquet/video rows, saves step 2, restores optimizer/RNG/configuration, and trains step 3. Every step must have finite loss and gradients; resumed head weights must change and AdamW step counters must reach 3. `--variants NAME ...` optionally selects cases for review, but omitting it runs the full matrix.

CPU command (executed; no GPU initialization):

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/test_variants.py \
  --device cpu \
  --output-dir /tmp/dev/audits/act-diffusion-variants-20260915/act/cpu-complete
```

Regression command:

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_b1k.py \
  --junitxml=/tmp/dev/audits/act-diffusion-variants-20260915/act/pytest-final.xml
```

The matrix uses two real tasks (`turning_on_radio`, `picking_up_trash`), 27 state inputs and 23 action outputs, all three cameras, no pretrained download, and deliberately approximate 64-frame statistics. ACT is a **small genuine model**: hidden 32, feedforward 64, 1 encoder/1 decoder layer, 4 heads, chunk 4, 64×64 images, batch 2 (about 11.22M trainable parameters). CNNMLP uses the **unchanged upstream CNN/1024-wide MLP**, three independent ResNet18 backbones, batch 1 (42.67M parameters at 480×640). Transformer smoke dimensions passed alongside CNNMLP are unused by its MLP and do not shrink it.

| Configuration / evidence directory | Policy and architecture | Training image size | Steps 1, 2, resumed 3 loss | Full chunk (H=4) | Replan (H=3) | Aggregation (H=1) | One action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `act-sine-post` | ACT, sine, post-norm | 64×64 | 145.186935, 120.966133, 143.659454 | PASS | PASS | PASS | n/a |
| `act-learned-post` | ACT, learned, post-norm | 64×64 | 143.561386, 110.661423, 104.807724 | PASS | PASS | PASS | n/a |
| `act-sine-pre` | ACT, sine, pre-norm | 64×64 | 111.598618, 99.390610, 91.814896 | PASS | PASS | PASS | n/a |
| `act-learned-pre` | ACT, learned, pre-norm | 64×64 | 132.316360, 118.076920, 117.898773 | PASS | PASS | PASS | n/a |
| `cnnmlp-native` | CNNMLP, default 768 features/camera | 480×640 | 3575.869629, 3492.153809, 1418.792603 | n/a | n/a | rejected | PASS |
| `cnnmlp-spatial` | CNNMLP, derived 32 features/camera | 416×416 | 3513.170410, 3500.891602, 1372.536499 | n/a | n/a | rejected | PASS |

Each serving case runs in a **fresh process** that loads only the standalone checkpoint and an exported real observation NPZ. A Python audit hook rejects dataset opens/listing, and a deliberate dataset-open probe confirms the guard is active. No dataset rename, modification, or conversion is needed. Requests travel over actual localhost WebSockets, not direct `Session.act` alone. Observations are original episode 0 and 200, frames 0 and 1: head RGB 720×720, wrist RGB 480×480, 61-dimensional R1Pro state. The worker compares returned values against independently calculated chunk indices or upstream exponential-weight formulas, and checks health, metadata dispatch, finite float32 `(B,23)`, two-client isolation, reset without acknowledgement, batch resizing, task-slot changes, both tasks' real observations, and unknown-task rejection. No simulator success is implied.

Evidence root: `/tmp/dev/audits/act-diffusion-variants-20260915/act/`:

- `cpu-complete/summary.json`: all configuration results, exact train/resume/worker command argument lists, dimensions, metrics, checkpoint SHA256, observation sources, and serving checks.
- `cpu-complete/<configuration>/evidence.json`, `training/metrics.jsonl`, `training/step_0000000{1,2,3}.pt`, and per-mode JSON/logs.
- `cpu-complete-matrix.log`: actual background-run training output.
- `pytest-final.log` / `pytest-final.xml`: complete regression evidence. The original 12 tests remain, extended to **23 passing cases**: architecture branches, native CNNMLP MSE/backprop/spatial shape, original-source strict-key and bit-exact forward compatibility, bit-exact CNNMLP and ACT uninterrupted-vs-resumed optimization, old ACT checkpoint fallback, explicit-zero aggregation, weight orientation, history expiry, transactional failure, per-slot resets, and unsupported modes.
- `cpu/` and `cpu-matrix.log`: earlier successful six-configuration / ten-serving-case run before the final runner added a separate full-chunk case and both-task observations.
- `cpu-final/`: an interrupted attempt at the expanded matrix; the tool wrapper killed it at 120 seconds after 13 serving cases, so its summary is explicitly `CANCELLED`, not counted as a complete pass. `cpu-complete/` is the rerun without that artificial timeout.
- Independent reviewer artifacts in the parent audit directory: `act-legacy.xml`, `act-independent.xml`, `cnnmlp-cli-websocket.json`, and `act-aggregation-cli-websocket.json`. The reviewer separately verified original CNNMLP checkpoint keys/bit-exact forward, old B1K ACT restore, and actual CLI servers with the existing OpenPI MessagePack codec, real RGB/RGBA observations, 11 steps, reset, batching, task changes, invalid input/reconnect, and two-client isolation. Reviewer-owned servers were stopped.

Final verified result: **6/6 training configurations, 18 finite optimizer steps (including six resumed step-3 updates), 14/14 real-WebSocket serving cases, and 23/23 regressions passed**. The complete matrix finished in 120.95 seconds; the final regression run finished in 38.17 seconds. All owned matrix processes and monitors completed; no serving jobs were left running.

### GPU status and limitations

**GPU validation of this expanded matrix is pending.** All GPUs are occupied by external GR00T jobs (approximately 195 GB each per coordinator); this work ran CPU-only and did not launch CUDA jobs or disturb those jobs. The earlier ACT-only GPU evidence above is not evidence for new variants. When the coordinator confirms and reserves a genuinely idle GPU, rerun the identical matrix with a new output directory:

```bash
source /tmp/dev/env.sh
cd /tmp/dev/baselines/act
# Set IDLE_GPU only after the coordinator verifies no processes and free memory.
CUDA_VISIBLE_DEVICES="$IDLE_GPU" OMP_NUM_THREADS=1 .venv/bin/python scripts/b1k/test_variants.py \
  --device cuda \
  --output-dir /tmp/dev/audits/act-diffusion-variants-20260915/act/cuda
```

The runner requires one explicit physical GPU index and checks `nvidia-smi` for no compute processes, at most 100 MiB used, and zero utilization **before importing torch**. It fails closed otherwise. This finite CPU smoke is not full-scale ACT training, converged CNNMLP training, a hyperparameter search, or robot/simulator evaluation. No new algorithm or loss replaces either upstream policy.
