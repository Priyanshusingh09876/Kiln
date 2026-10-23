#!/usr/bin/env bash
# Type-check csrc/*.cu with the host C++ compiler on machines without nvcc.
# Rewrites `kernel<<<g, b, s, st>>>(args)` into a plain call so the device code
# is parsed, template-instantiated and type-checked by g++/clang++ against a
# CUDA stub header + the real libtorch headers. Catches most compile errors
# (bad template args, missing includes, signature mismatches) on CPU CI; the
# GPU runner still does the real nvcc build.
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_INC=$(python -c "import torch, os; p=os.path.dirname(torch.__file__); print(f'-I{p}/include -I{p}/include/torch/csrc/api/include')")
PY_INC=$(python -c "import sysconfig; print('-I'+sysconfig.get_paths()['include'])")
# CUDA headers: a real toolkit if present, else the pip wheel (nvidia-cuda-runtime-cu12 ships the headers)
CUDA_INC=$(python - <<'PY'
import os, glob, importlib.util
# a real toolkit wins; otherwise the pip wheels nvidia-cuda-runtime-cu12 (runtime/fp16/bf16 headers)
# + nvidia-cuda-nvcc-cu12 (crt/ headers) together provide everything needed for host type-checking.
inc = []
home = os.environ.get("CUDA_HOME") or ("/usr/local/cuda" if os.path.exists("/usr/local/cuda/include/cuda_runtime.h") else None)
if home:
    inc.append(os.path.join(home, "include"))
else:
    spec = importlib.util.find_spec("nvidia")
    roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    for sub in ("cuda_runtime", "cuda_nvcc", "cuda_cccl"):
        for r in roots:
            d = os.path.join(r, sub, "include")
            if os.path.isdir(d): inc.append(d)
    ok = any(os.path.exists(os.path.join(d, "cuda_bf16.h")) for d in inc) and any(os.path.exists(os.path.join(d, "crt", "host_config.h")) for d in inc)
    if not ok:
        raise SystemExit("CUDA headers not found: pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12")
print(" ".join("-I" + d for d in inc))
PY
)
CXX=${CXX:-g++}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
status=0
for src in csrc/*.cu; do
  out="$tmp/$(basename "$src" .cu).cpp"
  # <<<grid, block, smem, stream>>>(args)  ->  kernel_launch(kernel, grid, block, smem, stream, args)
  perl -0pe 's/(\w+(?:<[^<>;]*(?:<[^<>;]*>[^<>;]*)*>)?)\s*<<<\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^>]+)>>>\s*\(/kernel_launch($1, $2, $3, $4, $5, /g' "$src" > "$out"
  if $CXX -std=c++20 -fsyntax-only -w -include csrc/host_stub/cuda_stub.h \
       -DTORCH_EXTENSION_NAME=kiln_cuda \
       -Icsrc $CUDA_INC $TORCH_INC $PY_INC "$out" 2> "$tmp/err.txt"; then
    echo "OK   $src"
  else
    echo "FAIL $src"; head -40 "$tmp/err.txt"; status=1
  fi
done
exit $status
