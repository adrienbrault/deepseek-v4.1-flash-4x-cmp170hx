# TC prefill plus v2: fixed concurrency and CPU sensitivity

Measured on 2026-10-21, image `dsv41:int-r17`, TP4, DSpark K5, TC prefill and medium-batch v2 enabled. Results directory: `2026-10-21T104329Z-dsv41-fixed-tc`. The fixed chat workload uses 68 prompt tokens, 400 completion tokens, ignored EOS, temperature zero and seed 15. Each concurrency has one excluded warmup and three measured batches. The aggregate is the measured sum of concurrent stream decode rates; TTFT is separate.

| Concurrency | Aggregate tok/s | Per-stream mean tok/s | Median TTFT s |
|---|---:|---:|---:|
| 1 | 72.7 | 72.7 | 0.245 |
| 4 | 135.7 | 33.9 | 0.520 |
| 8 | 182.2 | 22.8 | 0.754 |

This sweep has no matched flag-off fixed-concurrency control, so it establishes rates rather than a v2 speedup. Three short batches show variation: c1 aggregate 66.4–81.1, c4 127.0–146.0 and c8 179.0–185.9 tok/s. All 39 measured requests produced 400 tokens. Six known answers, vision and cache reporting passed before the sweep.

The same loaded model then ran six interleaved CPU clock rounds, with three normal and three capped rounds at each of c1/c4. This separate raw-completion code workload has 37 prompt tokens and 400 output tokens, ignored EOS, temperature zero and seed 15. Normal-clock aggregate means were 93.05 at c1 and 183.61 at c4; the 1.6 GHz cap gave 96.04 and 174.32 respectively (+3.2% and −5.1%). Sampled median clocks were about 2.78 GHz normal and 1.60 GHz capped. This screen found no c1 slowdown and modest c4 sensitivity; it does not predict a newer CPU's gain. The immediate restoration receipt is false, but a live readback at 11:05 UTC found all 24 policy limits matched the saved original values. Do not treat the immediate receipt as verified restoration.

The next sparse-attention model screen started at 10:58:11 UTC. No sparse model throughput result is included here.
