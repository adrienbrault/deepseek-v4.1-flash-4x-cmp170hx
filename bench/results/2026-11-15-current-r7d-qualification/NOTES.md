# Current-r7d qualification, 2026-11-15 UTC

Results directory: `rounds/moe-stack-confirm-r1/out/20261115T214428Z-3587331-moe-stack-confirm`, measured 21:44–22:02 UTC on 2026-11-15. The only source for the new absolute numbers is the current-r7d section of `NUMBERS.md`. [summary.json](summary.json) transcribes that table. No raw traffic, sealed report or new benchmark was used.

Current-r7d adds the rows-family kernels for 24 row counts and K2 fixed16 to current-r7c. Configuration: TP4, four Gen2 x16 CMP 170HX cards at 175 W each, DSpark K5, adaptive verification off, FULL_DECODE_ONLY, capture sizes 1–48, eight sequences and 2,048 batched tokens. Decode uses frozen `decode-prompts-v2` chat prompts, mostly reasoning prose before code, temperature zero, seed 15, EOS suppression and 400 output tokens per request. Each aggregate sums measured stream decode rates; per-stream statistics and shared-overlap-window rates are unavailable.

| concurrency | decode aggregate, tok/s | protocol | step, ms | same-day r7c reference, tok/s |
| ---: | ---: | --- | ---: | ---: |
| 1 | 168.2 | ordered | 18.67 | 158.6 |
| 4 | 369.9 | ordered | 32.18 | 360.5 |
| 8 | 457.5 | burst | 44.70 | 431.5 |

Ordered c8 was 419.7 tok/s against 401.5 tok/s in the same-day reference, on the same date, workload, output length, sampler and configuration. Ordered c8 was not the deciding metric. Burst throughput and step timing are separate measurements. The reference is the earlier current-r7c ON boot in `20261115T173926Z-1666654-indexer-confirm3`; this is a one-ON comparison against a stored same-day reference, with cross-boot drift limits.

No clean same-config absolute prefill result is published for current-r7d. The full ON boot and the older reference differed in host state. The 2026-11-16 same-hour control reported prefill +2 to +6 % against current-r7c, using nominal 2K/8K prompts at c1/c4/c8, one output token, temperature zero, thinking off, zero prompt-cache hits and warm-steady host cache; source `rounds/moe-stack-confirm-r1/RESULTS.md`, paired diagnostic runs `20261115T222815Z-3927983-moe-stack-confirm` and `20261115T230222Z-1823-moe-stack-confirm`. Two single boots establish limited evidence; a new full suite is needed for absolute prefill reporting.

The retained confirmation records PROMOTE_ELIGIBLE and BITWISE_PASS for both names. The final plugin scopes STACK entrypoints to eligible target graph capture forwards and restores exact incumbent functions on exit; eager prefill retains those functions. It preserves the incumbent descriptor set and dispatch, with one target graph bank. K2 fixed16 is restricted to four-request, 16-row verify calls in layers 18–22; the rows-family selector covers 24 winning row counts. No second graph bank or rung-2 memory reserve is added.

Actual-launcher acceptance passed on 2026-11-15 at 23:43–23:56 UTC, run `20261115T234323Z-320896-actual-acceptance`. The daily definition became current-r7d at 2026-11-16 00:00 UTC (01:00 Paris); serving remained stopped after teardown. Acceptance adds no speed measurement or client-network reachability evidence. This export was checked locally; no image build, native compile or GPU execution was repeated.

Fixed work (`agent_fixed_work.py`, same suite date, run and r7d configuration) records 133.8 tok/s against 134.2 tok/s in the stored r7c reference, from the same `NUMBERS.md` section. Complete workload, output length and sampler metadata are absent, so these contextual values are omitted from the headline table and card.
