#!/usr/bin/env bash
# Create (or recreate) the project-local .venv used by every scripts/b1k command, reproducibly.
#
#   scripts/b1k/setup_venv.sh            # exact lock: requirements-b1k.lock.txt (CPython 3.10, CUDA 13 wheels)
#   LOCK=0 scripts/b1k/setup_venv.sh     # other platforms/CUDA: pinned torch/torchvision + requirements-b1k.txt
#
# Environment: PYTHON (interpreter for the venv, default /usr/bin/python3.10), TORCH_INDEX (default
# https://download.pytorch.org/whl/cu130), SYSROOT (where Python headers are staged when the system lacks
# python3-dev, default /tmp/dev/sysroots). Idempotent: re-running reinstalls into the existing .venv.
#
# The Triton kernels (--fast-maxpool, torch.compile) build their launcher with gcc and need Python.h. If
# `sysconfig` finds no headers, the matching Ubuntu libpython3.X-dev package is downloaded and extracted
# under $SYSROOT (no root, nothing written outside /tmp); export the printed CPATH before training, as
# run_radio_300k.sh does. Nothing here touches $HOME or system paths.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -f /tmp/dev/env.sh ]]; then
    # shellcheck disable=SC1091
    source /tmp/dev/env.sh   # caches under /tmp (UV_CACHE_DIR, PIP_CACHE_DIR, TORCH_HOME, ...)
fi
PYTHON=${PYTHON:-/usr/bin/python3.10}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
SYSROOT=${SYSROOT:-/tmp/dev/sysroots}
LOCK=${LOCK:-1}

uv venv --allow-existing --python "$PYTHON" .venv
if [[ "$LOCK" == 1 ]]; then
    uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match -r requirements-b1k.lock.txt
else
    uv pip install --python .venv/bin/python --index-url "$TORCH_INDEX" torch==2.10.0 torchvision==0.25.0
    uv pip install --python .venv/bin/python -r requirements-b1k.txt
fi
.venv/bin/python - <<'EOF'
import torch, torchvision, triton, av, pyarrow, numpy
print(f'torch {torch.__version__} torchvision {torchvision.__version__} triton {triton.__version__} '
      f'av {av.__version__} pyarrow {pyarrow.__version__} numpy {numpy.__version__} cuda={torch.cuda.is_available()}')
EOF

# Python headers for Triton's gcc-built launcher.
include=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
if [[ -f "$include/Python.h" ]]; then
    echo "Python headers present at $include; no CPATH needed."
    exit 0
fi
version=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
staged="$SYSROOT/libpython${version}-dev"
if [[ ! -f "$staged/usr/include/python${version}/Python.h" ]]; then
    arch=$(dpkg --print-architecture 2>/dev/null || echo arm64)
    pool=$([[ "$arch" == amd64 ]] && echo http://archive.ubuntu.com/ubuntu || echo http://ports.ubuntu.com)
    pool="$pool/pool/main/p/python${version}/"
    deb=$(curl -sf "$pool" | grep -oE "libpython${version}-dev_[^\"<>]+_${arch}\.deb" | sort -V | tail -1)
    if [[ -z "$deb" ]]; then
        printf 'No Python.h and could not find libpython%s-dev in %s; install python3-dev or set CPATH yourself.\n' "$version" "$pool" >&2
        exit 1
    fi
    mkdir -p "$SYSROOT/debs" "$staged"
    curl -sf -o "$SYSROOT/debs/$deb" "$pool$deb"
    dpkg-deb -x "$SYSROOT/debs/$deb" "$staged"
    echo "Staged $deb under $staged"
fi
echo "Python.h is not installed system-wide; before training export:"
echo "  export CPATH=$staged/usr/include/python${version}:$staged/usr/include\${CPATH:+:\$CPATH}"
