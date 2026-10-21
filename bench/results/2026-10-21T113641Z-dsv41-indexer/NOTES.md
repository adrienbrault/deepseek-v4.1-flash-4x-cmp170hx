# Weight-load attribution and pending indexer screen

Measured on 2026-10-21 in `2026-10-21T113641Z-dsv41-indexer`. Image `dsv41:int-r17`, TP4 DSpark K5, TC prefill+v2, sparse off, indexer predecode and fused postprocessing enabled, OMP_NUM_THREADS=2. This report measures loading, without a generation workload or sampler. The indexer agent replay is still running; no candidate throughput verdict is included here. The numerical checkpoint is unchanged.

Launch at 11:36:43 UTC reached readiness at 11:42:56 UTC: 373 seconds (6m13s), validating the preceding estimate of about 372 seconds. The common preflight receipt hit took 0.291 seconds. Model loading itself took 280.96 seconds. One-second iostat, pidstat and per-card PCIe counters were collected only during boot and stopped before quality or performance probes. Disk sample times are inferred from collector start, with approximately one-second alignment. A 12-second native CPU sample and separate nonblocking Python snapshots were collected during loading.

| Metric | Whole model load | Main weight iterator |
|---|---:|---:|
| Duration | 281 s | 202.36 s |
| Physical NVMe reads | 227.9 GB | 208.5 GB |
| Mean NVMe read bandwidth | 0.811 GB/s | 1.027 GB/s |
| Peak sampled NVMe read bandwidth | 2.118 GB/s | 2.118 GB/s |
| Mean disk read latency | 0.272 ms | 0.154 ms |
| Mean disk queue depth | 2.13 | 2.45 |
| Mean loader CPU per worker | 98.0% | 98.9% |
| Mean per-card PCIe receive | 206–213 MB/s | 268–278 MB/s |

The main pass contains roughly 66.6% user and 32.4% system CPU per worker, with about 15,326 minor and 1,089 major faults per second per worker. The native sample attributed about 30.3% of sampled cycles to Python, 30.3% to kernel code, 16.3% to OpenMP and 10.3% to PyTorch CPU routines. Safetensors Rust accounted for 0.55% of sampled cycles. These are CPU-sample proportions, not a wall-time decomposition. Sampled disk utilization was 39.1% in the main pass; NVMe utilization alone is not a saturation test. The disk reached its previously measured 2.2 GB/s sequential neighborhood only in bursts, while the four loader workers stayed near one CPU core each. PCIe receive also stayed far below each card's link ceiling. This supports the host loader, tensor/memory handling and copy scheduling as the current primary limit, rather than sustained raw NVMe or PCIe bandwidth.

The physical checkpoint contains 214,804,147,240 bytes, 88 files and 188,241 indexed tensors. The loader walks tensor metadata per rank, takes aligned TP slices, materializes contiguous CPU buffers and fills the EXL3 arenas. This provides several CPU and small-copy costs to optimize. Additional drives could increase available aggregate read bandwidth when the loader issues enough parallel I/O, but no multi-drive gain has been measured. Three drives at the measured 2.2 GB/s each imply an ideal 6.6 GB/s aggregate ceiling; four cards at 2 GB/s each would require 8 GB/s. Those are arithmetic ceilings rather than a prediction. No disks were added or storage layout changed.

The next queued load uses one OpenMP thread instead of two to test the observed barrier overhead. It waits for the current agent screen, captures the same load counters, runs short correctness/cache/vision checks and leaves DeepSeek loaded. It has not run yet. Ten launcher tests pass on Linux; static serving validation remains 13 pass, zero fail, two unresolved-expansion warnings.


The candidate agent replay completed at 11:51:10 UTC: 64 requests, zero failures, 132.3 aggregate tok/s versus 139.3 for TC prefill+v2 control (−5.0%), mixed decode26.4 tok/s, TTFT median/p90 2.880/14.072 seconds. It used the frozen ten-session replay described above, temperature0.6, workload-selected output lengths. Indexer predecode and fused postprocessing remain off after this single screen; no throughput gain was established.
