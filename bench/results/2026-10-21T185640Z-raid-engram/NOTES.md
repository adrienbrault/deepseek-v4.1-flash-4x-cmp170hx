# RAID Engram placement screen,2026-10-21

Results directory `2026-10-21T185640Z-raid-engram`. Four CMP170HX cards, TP4, DSparkK5, TC-MoE prefill+V2, OMP2, batch2048, loader-r2 with strided bounce. Same immutable ring-capable image as `../2026-10-21T182229Z-ring-control/`, ring disabled. Hybrid weights remain on the two-drive970EVOPlus RAID0; only Engram moves from a separate970EVOPlus to the RAID. Independent byte verification completed before scoring.

Agent147.5tok/s,72requests/0failures, mean6.44streams, versus149.0control. Fixed c1/c4/c8 aggregate70.783/126.576/173.067tok/s,400output tokens, ignored EOS, temperature0, seed15, one warmup and three measured batches. No material agent/decode gain in this sequential screen.

Fresh2k c1/c4/c8 input1095.6/1174.9/1203.3tok/s; fresh8k1215.8/1232.7/1243.2. Thinking off, one output token, random-leading input and every cached_tokens0. Rate divides total batch input tokens by submit-to-last-first-output, including HTTP, tokenization, scheduling, queueing and first decode. The8k sequential rates are13.5%/13.8%/14.3% above control. One batch per point, differing random inputs and cache history; no confidence interval or daily promotion. Exact rates and differences are in `comparison.json`.
