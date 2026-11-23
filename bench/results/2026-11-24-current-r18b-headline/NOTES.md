# current-r18b headline, 2026-11-24 UTC

One boot on the 4x CMP 170HX box ran the base name `deepseek-v4.1-flash-1017` from 00:33 to 00:45 UTC on 2026-11-24. All four GPUs ran at PCIe Gen2 x16 with 175 W caps. The source record gives run `headline-r18b-20261123T002211Z-mUo7Jh`, image `sha256:f3027769f90387db0c6f6d0cbd2f05348a8e74023094f9c0d4ec4946d1ea206c` and argv matching seal `seal-nORCnYLm` subject to the benchmark listener and reset settings. The timestamp in the run label differs from the source's recorded measurement window.

[Summary](summary.json) transcribes the 2026-11-24 table in the retained headline ledger at source commit `9c9b8910911af97a22794566906a9f79ba630d24`. The local detailed report is for an earlier x8 boot. The x16 report and its raw counters, request rows and input receipt were unavailable in the local source checkout. Code TTFT p50 is stored in milliseconds from the published seconds rounded to two decimals. A single stream's measured sum is that stream's rate.

## Protocol

- **Spec-Bench** uses the frozen stratified 120-prompt subset at 1, 4 and 8 concurrent streams, temperature 0 and 256 forced output tokens per request. The client submits requests together up to the concurrency cap. Output tok/s divides generated tokens by benchmark wall time. Per-stream tok/s is 1,000 divided by median time per output token in milliseconds. Time to first token (TTFT) includes request and scheduling costs. Acceptance length is one plus accepted draft tokens per draft.
- **Cached code** uses one batch per cell at 1, 4 and 8 streams, temperature 0, seed 15 and 400 forced output tokens per request. Each request reuses a verified shared 1K, 32K or 64K code prefix and adds a fresh 2K or 8K append as indicated. The aggregate sums measured rates inside the common decode window. Per-stream is their mean. Append TTFT starts at the request with the fresh append.

## Comparison with the previous recipe

The 2026-11-16 [current-r7f x16 headline](../2026-11-16-current-r7f-headline/NOTES.md) used the same two workloads, output budgets and samplers. Spec-Bench wall output is equal within noise at 1, 4 and 8 streams. Cached-code decode sums at eight streams rose 8.5%, 4.7% and 12.9% for 1K + 2K, 32K + 2K and 64K + 8K. The median first token for the 64K + 8K append took 7.92 s at four streams and 14.00 s at eight, against 6.62 and 11.29 s on r7f. These are cross-boot comparisons of the full serving configurations with one code batch per cell.

The [cold-prefill suite](../2026-11-16-current-r7e-suite/NOTES.md) was measured on r7e, not re-measured on r18b. The scheduler changed between the two configurations.
