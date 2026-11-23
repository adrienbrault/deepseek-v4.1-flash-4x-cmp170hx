# Cold prefill on current-r18b, 2026-11-23 UTC

Measured on 2026-11-23 from 07:56 UTC on the served `current-r18b` boot of 07:44 UTC, four CMP 170HX cards at PCIe Gen2 x16 and 175 W per card. Instrument: [`bench/prefill_curve.py`](../../prefill_curve.py). Raw rows: [`curve.json`](curve.json); medians: [`summary.json`](summary.json).

Each request is a fresh prompt of random words with its own cache salt, sent alone to the base model name `deepseek-v4.1-flash-1017` on the completions endpoint, with one forced output token at temperature 0. Three requests per target length. The rate is the server's prompt token count divided by the client's time to the first streamed token, so it includes HTTP, scheduling and one decode step. All 21 requests reported zero cached prompt tokens and started and finished with no other request running on the server.

| prompt tokens (median request) | prefill tok/s, median | range of 3 | time to first token, median |
| ---: | ---: | ---: | ---: |
| 2,019 | 3,159 | 2,190–3,370 | 0.64 s |
| 8,048 | 3,894 | 3,815–3,925 | 2.07 s |
| 29,781 | 3,892 | 3,884–3,895 | 7.65 s |
| 60,127 | 3,738 | 3,727–3,745 | 16.08 s |
| 120,080 | 3,532 | 3,513–3,556 | 34.00 s |
| 200,420 | 3,305 | 3,288–3,320 | 60.64 s |
| 239,854 | 3,226 | 3,223–3,247 | 74.36 s |

The first 2K request of the run was slower (2,190 tok/s, 0.90 s) than the next two. At 2K the fixed per-request cost is a large share of the time to first token. Peak rate is about 3,900 tok/s between 8K and 30K; it falls to 3,226 tok/s at 240K as attention cost grows with context. One run per length set; no boot-to-boot repeat.
