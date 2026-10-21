# Startup caches

On 2026-10-21, the fixed TC prefill+v2 TP4 DSpark K5 configuration took 591 seconds from launch to readiness: 280 seconds loading the model and 182 seconds initializing the engine; about 129 seconds remained in launcher checks, process startup, imports and polling. The workload, configuration and timing receipt are in `bench/results/2026-10-21T104329Z-dsv41-fixed-tc/startup.json`. This is startup timing rather than token throughput.

The persistent mount at `/root/.cache` did not cover Triton at `/root/.triton` or TileLang at `/root/.tilelang`. The launcher change preserves those original container paths with separate bind mounts below the persistent cache, partitioned by the full Docker image digest. `scripts/preserve-compiler-cache.sh` copies the two directories from a running container without restarting it. The saved cache contained 42 MB of Triton files and 4 MB of TileLang files. The script requires a container name and `DSV41_CACHE_DIR`.

The future launcher mounts are `$DSV41_CACHE_DIR/compiler/<image digest>/triton:/root/.triton` and `$DSV41_CACHE_DIR/compiler/<image digest>/tilelang:/root/.tilelang`. The cache directories must be created first. The saving has not been measured by a subsequent boot. CUDA graph capture and model weight loading still run. The served configuration has not changed.
