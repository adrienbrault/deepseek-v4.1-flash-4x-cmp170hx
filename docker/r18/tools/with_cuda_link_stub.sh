#!/usr/bin/env bash
# Resolve explicit -lcuda and libcuda.so.1 for link and import checks in a
# driverless build. This process may not initialize CUDA or run a kernel.
set -euo pipefail
cuda_home=${CUDA_HOME:-/usr/local/cuda}
stub=
for candidate in "$cuda_home/lib64/stubs/libcuda.so" \
    "$cuda_home/targets/x86_64-linux/lib/stubs/libcuda.so"; do
    if [[ -f "$candidate" ]]; then
        stub=$candidate
        break
    fi
done
[[ -n "$stub" ]] || { echo 'CUDA toolkit libcuda link stub missing' >&2; exit 1; }
temp_dir=$(mktemp -d)
trap 'rm -rf -- "$temp_dir"' EXIT
ln -s "$stub" "$temp_dir/libcuda.so"
ln -s "$stub" "$temp_dir/libcuda.so.1"
export LIBRARY_PATH="$temp_dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$temp_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$@"
