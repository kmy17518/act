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
scripts/b1k/setup_venv.sh          # exact environment from requirements-b1k.lock.txt (about 25 s with a warm uv cache)
```

`setup_venv.sh` creates `.venv` from `requirements-b1k.lock.txt`, the `uv pip freeze` of the verified environment (CPython 3.10, aarch64, torch 2.10.0+cu130, torchvision 0.25.0+cu130, triton 3.6.0, every transitive pin), prints an import/CUDA check, and then makes sure Triton can build its gcc launcher: if `sysconfig` reports no `Python.h` (this host has no `python3-dev`, and `/usr` is read-only for us) it downloads the matching Ubuntu `libpython3.X-dev` package and extracts it under `/tmp/dev/sysroots`, printing the `CPATH` export that `run_radio_300k.sh` already applies. It is idempotent (`--allow-existing`), writes nothing outside `/tmp`, and honours `PYTHON`, `TORCH_INDEX`, `SYSROOT`. Re-running it after the 2026-09-17 host move reproduced the environment and the GPU tests from a clean checkout. On another platform or CUDA version use `LOCK=0 scripts/b1k/setup_venv.sh`, which installs the pinned torch/torchvision pair from `TORCH_INDEX` and the looser `requirements-b1k.txt`; the equivalent manual steps are:

```bash
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

### Resized-frame cache (`--frame-cache`)

Native random access costs ~33 ms of single-core work per sample (HEVC GOP 8 decodes ~4.5 frames per requested frame, three cameras, plus the antialiased resize), which caps a 24-worker loader near 900 samples/s. `scripts/b1k/build_frame_cache.py --dataset-path ROOT --cache-dir DIR --task-names ... --image-size 240 240` decodes every selected video **once, sequentially**, and stores `preprocess_image()` output rounded to the nearest 1/255 as memory-mapped uint8 `[N, H, W, 3]` arrays with presentation timestamps and a manifest (source size/mtime, image size, decoder). Entries are written atomically and rebuilt when stale; `--verify N` compares random samples byte for byte against native seek-and-decode (the radio cache: 1,536 frames from 512 samples identical). With no B-frames, sequential and seek-then-decode pictures are the same.

`train_b1k.py --frame-cache DIR` then serves samples from the cache: frame rows are resolved once per episode with the native reader's nearest-frame/tolerance semantics, and timestamp/state/action columns for every selected row are held in memory per process (the packed Parquet files contain one 229k-row group each, so per-sample filtering cost ~1.3 ms) after the same per-row episode/frame/task/index checks. Images travel to the GPU as uint8 and become `uint8 / 255` there (a true division by a 0-dim tensor, bit-identical to the CPU path), so the only difference from the native path is the half-LSB rounding of the resized frame (`max |Δ| = 0.5/255`, below the source video's own 8-bit quantization); states, actions and padding are bit-identical. Per-sample loader cost drops from ~33 ms to ~0.2 ms, and the radio cache (429,928 frames × 3 cameras at 240×240) occupies 223 GB of page cache/disk under `/tmp/dev/datasets/2026-challenge-demos-act-frame-cache-240x240`. Serving is unaffected: it still runs `preprocess_image` on the incoming uint8 observation.

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

**GPU safety on this host:** use only a GPU explicitly reserved by the launch coordinator, and check its current processes, memory, and utilization before starting. Never stop or share external GR00T/OpenPI jobs. CPU checks use `CUDA_VISIBLE_DEVICES='' --device cpu`. Historical GPU assignments below are not current reservations. The variant runner performs its own idle check before importing torch for `--device cuda`.

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

`--max-steps` is the total optimizer count, not additional steps. Resume restores model, optimizer, torch/CUDA RNG, normalization, task map, preprocessing, architecture and sampler seed. Saved architecture/preprocessing are authoritative; dataset root/subset must have the same fingerprint, or — when files were merely re-synced (the fingerprint includes Parquet mtimes) — recomputed exact full-dataset statistics must equal the checkpoint's, in which case the resume is accepted with a warning and later checkpoints carry the new fingerprint; approximate smoke statistics cannot be verified this way and are refused. Worker count/device/batch size can change; exact training continuation requires the same batch size. Fresh training refuses an existing run, and snapshot paths are never silently overwritten. Resume into another output directory is supported. `latest.pt` is a relative symlink to the newest immutable `step_XXXXXXXX.pt`; copy the actual `.pt` file for deployment.

Checkpoints contain all serving and training configuration, selected task IDs/names, continuous normalization, image/camera/state mapping, action semantics, model/optimizer state, RNG and completed step. Serving does **not** need the dataset, stats JSON, `run.json`, or pretrained backbone cache. Checkpoints are loaded with `torch.load(weights_only=True)`.

Other useful knobs: `--policy-class`, `--position-embedding`, `--pre-norm`, `--chunk-size`, `--image-size HEIGHT WIDTH`, `--hidden-dim`, `--dim-feedforward`, `--enc-layers`, `--dec-layers`, `--nheads`, `--kl-weight`, `--lr`, `--save-every`, `--cache-row-groups`. See `--help`. Metrics are JSON lines in `metrics.jsonl`; every checkpoint is suitable for inference, but no validation-driven best-checkpoint selection or simulator success evaluation is performed by this trainer.

This ACT/CNNMLP integration is **single-process, single-GPU** (or CPU) only. Unlike GR00T's distributed training path, it does not implement `torchrun`/DDP; do not launch multiple training ranks against one output directory. `--num-workers` controls CPU data-loading workers, not GPU training ranks. An exclusive nonblocking `flock` on `output/run.lock` rejects a concurrent trainer before metadata, W&B, or dataset setup. The file remains after exit; the kernel releases the lock automatically, so do not delete it while training.

### Throughput options (Blackwell)

All of the following keep fp32 weights, fp32 optimizer state, the upstream loss, the same trained/frozen parameters and the same sampling; they change kernels, layouts and precision of intermediate arithmetic only. Measured on one Blackwell GPU with 30 cores (`turning_on_radio`, 14–300 step probes, steady-state median): batch 1560 **3.40 → 0.45 s/step** and batch 1024 **1.77 → 0.30 s/step** with the radio recipe (`--frame-cache`, `--matmul-precision high`, `--compile regions-autotune`, defaults otherwise).

| Option | Default | Effect | Numerics (same batch/weights, dropout off) |
| --- | --- | --- | --- |
| `--frame-cache DIR` | off | uint8 resized frames, in-memory tabular columns, loader wait <10 ms | images within 0.5/255; everything else bit-identical |
| GPU batch assembly | always on CUDA | each pinned worker slice is copied into a device-resident batch on a side stream while the previous step computes; no 3.2 GB host merge | exact |
| unused decoder layers skipped | on (`--compute-unused-decoder-layers` restores upstream work) | ACT consumes only the first stacked decoder output, so decoder layers 2..N get exactly-zero gradients; they are no longer executed and receive the same zero gradients (AdamW weight decay unchanged) | predictions, gradients and optimizer trajectories bitwise identical |
| `--attention auto\|fused\|explicit` | `auto` | upstream explicit matmul+softmax attention in fp32/TF32 (tensor-core GEMMs beat the fp32-only fused kernel on Blackwell), fused `scaled_dot_product_attention` under autocast | same attention math; fused vs explicit rel Δloss 1e-7, gradient cosine 1.0 |
| `--channels-last` | on | NHWC layout for the ResNet convolutions and images (frame-cache batches are laid out camera-major so each camera slice is a dense NHWC block); restored optimizer moments are re-laid out to match | layout only; rel Δl1 8e-7 |
| `--fused-optimizer` | on (CUDA) | single-kernel fused AdamW | identical update rule; max Δparam 1.5e-8 after 3 identical-gradient steps; restored state bit-identical |
| `--cudnn-benchmark` | on | cuDNN autotunes convolution algorithms for the fixed shapes | algorithm choice only |
| `--fast-maxpool` | on (CUDA) | Triton channels-last kernels for the ResNet stem `MaxPool2d(3, 2, 1)` (`detr/models/maxpool_nhwc.py`): int8 winning-tap indices instead of int64 gathers, ~4x less time than ATen's NHWC kernels eager, ~2 % per step and −12 GiB under `--compile regions`; falls back to `F.max_pool2d` on CPU. Triton compiles its launcher with gcc, so the launch recipe exports `CPATH` to the staged Python headers under `/tmp/dev/sysroots` | bitwise identical to `F.max_pool2d` (values, argmax ties, NaN propagation, gradient accumulation order), verified up to 4,680 images (int64 indexing above 2^31 elements) |
| `--matmul-precision high` | `highest` | TF32 tensor cores for fp32 matmuls (convolutions already default to TF32 in PyTorch) | rel Δloss 7e-6, gradient cosine 0.9999997; resumed step-10000 losses within the ±0.3 % dropout noise of the original run |
| `--compile backbone\|regions\|regions-autotune` | `none` | Inductor fuses the frozen-BN affine/ReLU/residual elementwise passes around cuDNN convolutions (`backbone`), plus the style encoder and transformer as separate graphs with the attention pattern matcher off (`regions`); `regions-autotune` also benchmarks Triton/cuBLAS GEMM and convolution choices per shape (GEMM time 79 → 58 ms per step at batch 1024). Compile takes a few minutes once per shape set (on-disk cache under `/tmp/.cache/torchinductor` afterwards) | backbone rel Δl1 3e-7; regions rel Δloss 2.5e-6; regions-autotune rel Δl1 4.5e-5 (TF32-class kernel differences), gradient cosine 1.0 for all; dropout masks come from Inductor's RNG (same distribution); resume reproduces the uninterrupted losses exactly |
| `--autocast bf16-backbone` | off | bf16 autocast inside the ResNet bodies only; features, position encodings, transformer, heads and losses stay fp32/TF32 (batch 1024 0.28 s/step, batch 1560 0.42 s/step with `--compile regions`, 12 % faster than TF32) | small: over a 300-step resume from step 10000, mean per-step L1 sits +0.04 % above the exact TF32 path (TF32 itself sits +0.04 % above the original fp32 run), gradient norm and KL indistinguishable; opt-in |
| `--autocast bf16` | off | bf16 forward for the backbone/transformer with fp32 output heads, latent distribution and losses; fastest (batch 1024 0.29 s/step, batch 1560 0.44 s/step before compile) | **not numerically neutral**: rel Δloss 1e-4 on one batch, and resumed step-10000 L1 is systematically +0.6–1.0 % (+0.77 % over 300 steps) with a 20–50 % larger gradient norm; kept opt-in, not used by the radio recipe |

`timing/data_wait_s` is the host time blocked fetching a step's batch (overlapped with the previous step's compute after step 1); `timing/train_s` covers launch, compute and that overlap. In cache mode `data/video_decode_s` is 0 and `data/frame_cache_s` reports the memcpy time per sample.

**Launching.** `scripts/b1k/run_radio_300k.sh` is the complete recipe (cache build/verify, GPU occupancy check, resume, W&B, exit-status file) with the fastest numerically neutral settings as defaults; its header documents every override. Examples:

```bash
bash scripts/b1k/run_radio_300k.sh                                            # resume the original run (batch 1560, TF32)
ACT_BATCH_SIZE=1024 ACT_RUN_TAG=bs1024 bash scripts/b1k/run_radio_300k.sh     # fresh run at batch 1024, own directory/log/W&B run
ACT_AUTOCAST=bf16-backbone ACT_RUN_TAG=bf16bb bash scripts/b1k/run_radio_300k.sh   # fastest option with a small measured deviation
ACT_AUTOCAST=bf16 ACT_RUN_TAG=bf16 bash scripts/b1k/run_radio_300k.sh         # fastest, not numerically neutral (see table)
ACT_COMPILE_MODE=none bash scripts/b1k/run_radio_300k.sh                      # eager kernels (0.41 / 0.65 s/step), no compile time
```

The overrides are prefixed `ACT_` because the Diffusion Policy recipe uses `BATCH_SIZE`, `FRAME_CACHE`, `GPU_UUID`, ... and a tmux server or shell shared with it would otherwise redirect this run (its `FRAME_CACHE` once pointed the builder at the DP 96 px cache directory; the builder now refuses directories holding another format or size). Launch from a dedicated tmux server (`tmux -L b1k-act ...`). A fresh `ACT_RUN_TAG` never touches the original directory or W&B run (`ACT_WANDB_ID` defaults to `actradio-<tag>`); a fresh run also needs its own uploader invocation if its checkpoints should be published. Direct `train_b1k.py` invocations take the same flags (`--frame-cache`, `--matmul-precision high`, `--compile regions-autotune`, `--autocast ...`).

### Long-run controls

For the 300,000-step single-task run, keep the default ACT architecture, fp32 weights/optimizer, and optimizer recipe above. Set `--task-names turning_on_radio --max-steps 300000`; the launch coordinator must supply the largest **validated physical batch** for its reserved GPU. This integration does not enable gradient accumulation or change learning rates automatically; bf16 autocast is opt-in only (see the numerics above).

- `--save-every 2500 --save-total-limit 3` keeps the three newest committed local full checkpoints plus the `latest.pt` symlink. `--save-total-limit 0` is the backward-compatible default (keep all). `--save-first-step` additionally saves full step 1 for immediate upload/resume verification; it does not create an unscheduled eval export. The final step is always saved.
- `--export-every 10000` writes `export_queue/eval/step_XXXXXXXX.pt` on interval boundaries. Default `0` disables export queues. Eval-only checkpoints retain model weights, architecture, adapter/camera/state mapping, normalization, task map, training provenance, W&B identity and step, but omit optimizer and RNG. Tensors are stored on CPU; both `load_checkpoint` and the existing serving command accept them without a dataset. Training resume rejects them and requests a full checkpoint.
- Full saves also hardlink the newest committed snapshot into `export_queue/full/step_XXXXXXXX.pt` **before** pruning local snapshots. Only the newest full queue entry is retained. A separate uploader stages its own hardlink before hashing/uploading, so local/full-queue pruning cannot remove an in-flight upload. Eval queue entries are never removed by trainer retention: the uploader acknowledges them only after a verified remote commit. Pending evals can therefore consume disk during a prolonged upload outage.
- Checkpoint data is flushed and fsynced to a hidden temporary file, then atomically published without overwriting an existing snapshot. Directory entries are fsynced, `latest.pt` is atomically switched, and only committed numeric step files are pruned. Temporary files, unrelated files and eval exports are excluded. An identical pending eval export from an interrupted save can be reused on deterministic resume; conflicting model/configuration content is rejected.
- Full normalization uses **every selected frame**, with no video decode or approximation. Omit `--stats-max-frames` for the real run. A metadata/statistics frame-count mismatch fails; resuming an approximate smoke checkpoint without an explicit smoke flag also fails. Statistics remain embedded in checkpoints and cached atomically.

W&B options are `--wandb-project`, `--wandb-entity`, `--wandb-name`, `--wandb-id`, and `--wandb-mode online|offline|disabled` (default `disabled`). Online mode requires `WANDB_API_KEY` in the environment, verifies credentials before loading data/model, and fails rather than silently switching offline. Set `WANDB_BASE_URL=https://api.wandb.ai` explicitly for the public W&B service if this host has a different inherited endpoint. Do not pass secrets on the command line or run `wandb login` to create a home-directory credentials file. Identity is persisted in `wandb.json`, run metadata and checkpoints; resume reuses the saved ID/project/entity/name, and conflicting overrides fail. Online-to-online checkpoint resume requires the existing remote run. `offline` is explicit, not an authentication fallback.

Each optimizer step writes the same loss components, gradient norm, LR, elapsed time, throughput, data wait/train/checkpoint/step durations, and GPU allocated/reserved/peak bytes to `metrics.jsonl` and (when enabled) W&B at the completed optimizer step. `data/parquet_s`, `data/video_decode_s`, and `data/sample_s` are per-sample means across the batch; they are worker wall times, not additive pipeline latency. GPU timings synchronize the completed update; GPU memory fields are zero on CPU. No simulator evaluation or validation-best selection is implied by exporting a checkpoint.

`--prefetch-factor` defaults to 1 and applies only with workers. `--worker-threads`, `--arrow-threads`, and `--opencv-threads` default to 1; PyArrow CPU/I/O and installed OpenCV pools are capped in the main process and each spawned worker. PyAV already uses one decoder thread per video. `--torch-threads` optionally caps main-process PyTorch threads without changing its default setting. Keep `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` bounded at launch as well.

`--loader-batch-size 128` is optional (default: one complete optimizer batch per worker request). It dispatches slices of the **same ordered sampled batch** to independent workers, then assembles them into a single physical batch before the forward pass. A nondivisible final slice is supported. It does not accumulate gradients, change sampling order, or change optimizer step counts. On CUDA each pinned slice is copied straight into a device-resident batch on a side stream while the previous step's kernels run (the next batch is fetched right after `backward()` is queued); on CPU the batch is assembled in host memory as before. Choose worker count, slice size and prefetch together to bound host and shared-memory use; validate the complete data pipeline on the reserved GPU rather than relying only on synthetic batch probes. With the frame cache, 8 workers and prefetch 2 keep the loader wait below 10 ms at 3,200 samples/s.

The uploader is a separate process (`scripts/b1k/upload_checkpoints.py`); the trainer never blocks on network transfers. Its queue contract preserves scheduled eval files and the latest resumable full checkpoint independently of three-file local retention. Pass a stable uploader `--run-id` and stable metadata for restarts; checkpoint/run metadata includes `model_config`, `adapter_config`, `train_config`, `task_map`, `normalization`, and `wandb`. The dependency set includes `huggingface_hub==0.36.2` for the uploader's verified LFS garbage-collection API.

Long-run trainer regression evidence (September 16, 2026): **29 CPU tests passed**, including original ACT/CNNMLP numerics, bit-exact resumed training with worker-sliced batches, eval-only serving without the dataset, three-file retention, interrupted-save isolation, exclusive locks, per-step W&B mock logging/identity/auth failures, spawned-worker prefetch/caps, and exact-statistics guards. Command: `taskset -c 60-89 env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_b1k.py -q` after sourcing `/tmp/dev/env.sh`. Evidence is in `outputs/pytest-longrun-verified.log` and `.xml`. The installed W&B 0.30.0 SDK also passed an explicit offline metadata/logging smoke. These checks do not validate a maximum GPU batch, a live online training run, or model convergence; those are separate launch/evaluation responsibilities.

Throughput work (September 17, 2026) added six tests (**34 CPU tests pass, one CUDA-only test skipped on CPU and passing on the GPU**): frame cache vs native decoding and stale-entry detection, bitwise resume through the cache path, fused vs explicit attention, exactness of skipping unused decoder layers (predictions, gradients, two AdamW steps and optimizer state), image layouts/optimizer-state relayout/host batch assembly, and side-stream device batch assembly. GPU evidence (same weights and batch, fp32 explicit-attention reference; fused-vs-foreach and compiled-vs-eager comparisons; step-10000 resume comparisons against the original run) lives outside git under `/tmp/dev/audits/act-speed-20260917/`.

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
