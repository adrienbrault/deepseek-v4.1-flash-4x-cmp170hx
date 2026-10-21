#!/usr/bin/env bash
# Copy only compiled kernels out of a running container; do not restart it.
set -euo pipefail
name=${1:?container name required}
cache=${DSV41_CACHE_DIR:?persistent cache directory required}
image=$(docker inspect "$name" --format '{{.Image}}')
[[ $image =~ ^sha256:[0-9a-f]{64}$ ]]
dest=$cache/compiler/${image#sha256:}
mkdir -p "$dest/triton" "$dest/tilelang"
docker cp "$name:/root/.triton/." "$dest/triton/"
docker cp "$name:/root/.tilelang/." "$dest/tilelang/"
echo "Preserved compiler caches for $image"
du -sh "$dest/triton" "$dest/tilelang"
