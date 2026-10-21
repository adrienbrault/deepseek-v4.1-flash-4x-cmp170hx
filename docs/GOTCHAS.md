# Startup caches

On 2026-10-21, the fixed TC prefill+v2 TP4 DSpark K5 configuration took 591 seconds from launch to readiness: 280 seconds loading the model and 182 seconds initializing the engine; about 129 seconds remained in launcher checks, process startup, imports and polling. The workload, configuration and timing receipt are in `bench/results/2026-10-21T104329Z-dsv41-fixed-tc/startup.json`. This is startup timing rather than token throughput.

The persistent mount at `/root/.cache` did not cover Triton at `/root/.triton` or TileLang at `/root/.tilelang`. The launcher change preserves those original container paths with separate bind mounts below the persistent cache, partitioned by the full Docker image digest. `scripts/preserve-compiler-cache.sh` copies the two directories from a running container without restarting it. The saved cache contained 42 MB of Triton files and 4 MB of TileLang files. The script requires a container name and `DSV41_CACHE_DIR`.

The future launcher mounts are `$DSV41_CACHE_DIR/compiler/<image digest>/triton:/root/.triton` and `$DSV41_CACHE_DIR/compiler/<image digest>/tilelang:/root/.tilelang`. The cache directories must be created first. The saving has not been measured by a subsequent boot. CUDA graph capture and model weight loading still run. The served configuration has not changed.


## Reusing startup checks

`scripts/startup-preflight.py` caches the successful checkpoint audit and the runtime, SSD, kernel and Marlin checks. Its key includes the exact image ID, launcher and helper hashes, GPU UUIDs/serials/PCI addresses/model/driver/memory, host boot and kernel, hashes of checkpoint JSON metadata, and inode/size/mtime/ctime of every indexed checkpoint shard. This is change detection over the previously audited local checkpoint; it does not rehash weight payloads. A different checkpoint, image, GPU set, driver, reboot or launcher/helper edit runs the checks again. New feature-specific gates remain separate. Corrupt or incomplete receipts rerun the checks, and a failed forced recheck removes an older success. Use `--force` to rerun unchanged inputs.

The live validation first populates the cache during launch and then verifies reuse while leaving the same model loaded. Nine launcher tests pass on Linux, including checkpoint invalidation, corrupt-receipt detection, forced recheck and failed-check invalidation. The source launcher no longer restores the other model automatically; experiment ownership must request restoration explicitly. A validated candidate can remain loaded across compatible tests. These changes affect startup rather than the model's numerical configuration.

The generic `sharded_state` loader is deferred: the EXL3 plugin constructs packed trellis arenas, expert views and native handles during loading. A compatible rank-local checkpoint path needs an explicit reconstruction contract and parity checks. No checkpoint conversion or unvalidated loader switch is part of this startup change.


Startup validation on 2026-10-21 completed with the unchanged TP4 DSpark K5 TC prefill+v2 numerical configuration: launch-to-readiness fell from 591 to 446 seconds, with engine initialization falling from 181.92 to 36.86 seconds and model loading remaining near 279 seconds. A real preflight receipt hit took 0.289 seconds versus its 74.052-second miss. The next unchanged full boot is estimated at 372 seconds, not yet measured. Results, method and correctness checks: `bench/results/2026-10-21T112351Z-dsv41-startup/NOTES.md`. DeepSeek was left loaded after the validation.
