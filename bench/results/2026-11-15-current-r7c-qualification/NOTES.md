# Current serving qualification, 2026-11-15 UTC

Results directory: `rounds/indexer-bounded-confirm3/out/20261115T173926Z-1666654-indexer-confirm3`, 17:39–18:30 UTC. The OFF/ON/OFF chain compares current-r7b with current-r7c, adding exact bounded indexer top-k (`DSV41_INDEXER_BOUNDED=1`, k1b native library mounted read-only). The ON boot uses TP4, four Gen2 x16 CMP 170HX cards at 175 W each, fixed640 cooperative MoE, DSpark K5, adaptive verification disabled, cap-1 calibrated row masking, FULL_DECODE_ONLY graphs, eight sequences and 2,048 batched tokens. Correctable PCIe errors and replays were recorded; the report records x16 links and no uncorrectable errors.

[summary.json](summary.json) transcribes the retained headline table and the committed confirmation and promotion RESULTS and LEDGER entries. The sealed report and raw per-request records are absent locally. Values are rounded as supplied, with no recomputation from raw traffic and no local performance run. The suite v3 seal covers the loaded suite v2 components. Decode throughput is burst-v2; the step times are ordered-v3 measurements and are not reciprocals of the aggregate rates.

| concurrency | ON decode aggregate, tok/s | OFF boot rates, tok/s | ON ordered step, ms |
| ---: | ---: | --- | ---: |
| 1 | 158.6 | 153.8 / 153.9 | 19.80 |
| 4 | 360.5 | 345.0 / 344.5 | 33.02 |
| 8 | 431.5 | 406.3 / 410.9 | 48.80 |

Decode uses `decode-prompts-v2` chat completions, mostly reasoning prose before code, temperature zero, seed 15, EOS suppression and 400 output tokens per request. Each request rate is `(completion_tokens - 1) / (stream_end - first_content_event)`; each concurrent batch sums those measured rates, and each boot averages scored batches. Concurrency one has eight scored batches and concurrency four/eight have three each, after one unscored warm batch. The current result is one ON boot, not the two-boot mean used by the [previous qualification](../2026-11-14-current-r7b-qualification/). Individual stream statistics and shared-overlap-window rates are unavailable.

| nominal cold input tokens per request | concurrency | TTFT, s | aggregate input rate, tok/s |
| --- | ---: | ---: | ---: |
| 2,048 | 1 | 0.74 | 2,779 |
| 2,048 | 4 | 2.09 | 3,254 |
| 2,048 | 8 | 3.18 | 3,474 |
| 8,192 | 1 | 2.30 | 3,561 |
| 8,192 | 4 | 5.86 | 3,780 |
| 8,192 | 8 | 10.17 | 3,801 |

Prefill uses temperature zero, thinking off, EOS suppression and one output token, with one unscored warm batch followed by three scored batches per point. Each cell is the ON boot's median over scored batches b0–b2. Every batch must have zero prompt-cache hits; host page caches remain warm-steady without a cache drop. The aggregate input rate divides total reported prompt tokens by the span from earliest submit to latest first output. TTFT and input rate are transcribed independently. Nominal lengths are calibrated before the chat template; exact request token counts and template overhead are absent from this summary. TTFT includes the request path, not just GPU execution.

Fixed work (`agent_fixed_work.py`, same date, chain and ON configuration) records 134.2 tok/s against OFF boots 134.1 / 133.8. Full workload, output lengths and sampler metadata are absent from the retained headline table, so this row remains contextual and is omitted from the README table and card.

The committed confirmation records WIN and PROMOTE_ELIGIBLE under the unchanged evaluator after a report-level c8 declaration was restored from the sealed boots. All boots passed identity, with raw c8 in incumbent mode A throughout; the prospective rule also permitted an exact match to archived mode B. Every other identity group remained byte-identical. These are recorded results, not checks repeated here.

Actual-launcher acceptance passed on 2026-11-15 19:29–19:45 UTC, boot `20261115T193118Z-2547641`: inherited ready and sampled gates passed and the identity replay passed. The daily definition became current-r7c at 19:45 UTC (20:45 Paris). Serving was absent after teardown. Acceptance adds no speed measurement and loopback checks do not establish client-network reachability.

No current memory-category breakdown, measured KV pool capacity, decode TTFT, per-stream decode rates, sampled agent throughput or measured power draw is provided by these records. The [charts](../../plot.py) draw only published summary cells. Earlier promotion entries without numeric summaries are omitted from the history plot; October configurations are not joined to the November qualification series.
