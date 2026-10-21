# Loading-only candidates, 2026-10-21

These two patches are experimental and default off. They target the measured host-side model loading cost described in `bench/results/2026-10-21T113641Z-dsv41-indexer/NOTES.md`. A thin child image replaces only the two reviewed Python files in the measured parent; compiler libraries, native kernels and kernel source remain unchanged. Independent review found no P1 blocker for a bounded startup experiment. No startup gain or daily promotion is established yet.

| Patch | Purpose | Enable flag | License |
|---|---|---|---|
| `patches/vllm-exl3/0008-load-reclaim-fast.patch` | Skip reclaim scans on newly allocated CPU buffers; preserve fresh file-view validation | `VLLM_EXL3_LOAD_RECLAIM_FAST=1` | AGPL-3.0-only |
| `patches/vllm/0026-filter-dspark-unused-shards.patch` | Reuse the existing unused-file filter before DSpark tensor construction | `VLLM_DSV4_DSPARK_FILTER_SHARDS=1` | Apache-2.0 |

The plugin patch applies after the existing seven plugin patches on vcruz305/vllm-exl3 commit `3c60e6b45b5ec9e845b07aa70b4f974819010ffe`. Its installed base file hash is `5f8b22b99948c36f6e1f1a8c406ad21a1225da8c38854a8b59922f314d2aa35d`, candidate `0dd89079d21181a8c35f668c4de51b95e0e2463e6a0ca3130f289e99c43e1a87`. Thirty-six CPU tests passed on Linux with each flag state; three hardware/checkpoint cases were skipped. Real MAP_PRIVATE dirty-view and same-address-remapping tests preserve unconsumed data. The allocation-aware change has a CPU microbenchmark saving of 16.69 microseconds per 64 KiB materialized fill on the test CPU, not a measured model gain. Multiplying by 188,241 eligible calls would be about 3.14 CPU seconds; the actual eligible count and host speed differ.

The DSpark patch applies to the installed vLLM file from base `df3bd094aeea829f27baa0a5282830464f601d9f` plus the existing local series. Its base file hash is `c12b0b9151d3bc56232afc1dbd8c3a8a6f20db18273cadba2d6d0bdb3a2c78f9`, candidate `95b30c63add4b8f86f664dec692955b558ecd02cbf54cc8a6a13e7d2899dc0cb`. Six CPU tests passed on Linux. The predicate uses the original remapper and preserves private-embedding aliases and mixed shards; flag-off returns None, so no filter or header work is added. The measured checkpoint puts MTP weights in 3 of 88 shards, containing 2,401 of 188,241 tensor keys. Header inspection still visits all files. Only an instrumented boot can establish savings in the preceding 28–32-second draft iteration; the iterator's own timer excludes the new header-filter work, so total model-load timing is required.

The candidate image ID is `sha256:e160ba8946628acd473e887f8d8c662cd25ad6c9d15e7cd39eadab7be62d1a31`, derived from `sha256:cae174260cb6d6c9ab113ecb9db179bf20180662e4e16860252f9b8000bdb916`. The planned screen uses TP4 DSpark K5, TC prefill+v2, OMP_NUM_THREADS=2, sparse and indexer candidates off, both loading flags on. It captures boot disk/CPU/PCIe counters, checks answers/vision/cache reporting and retained shard count, then runs the existing short-chat c1/c4/c8 qualification on the same loaded model. No intermediate model restoration is performed.


`scripts/build-loader-candidate.py` reproduces the child from the pinned parent image and the numbered patches. Set `DSV41_CACHE_DIR` to the persistent parent cache directory before running it. It checks both source-file hashes before applying either patch and records the child image ID and file hashes in its output manifest. Compiler-cache reuse is limited to this child, whose only filesystem changes are the two loading-only Python modules.


## 2026-10-21 hardware screen

The combined allocation-aware reclaim and DSpark file filter reduced total model loading from280.956 to246.656seconds (12.2%), including header selection overhead. Six answers, vision and prefix-cache reporting passed. Fixed decode and the growing-agent replay establish no serving-speed improvement. Workloads, samplers, concurrency and caveats are recorded in `bench/results/2026-10-21T120933Z-dsv41-loader/NOTES.md`; no daily promotion is implied.
