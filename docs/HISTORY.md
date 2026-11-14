# Serving history

| UTC date | definition | change and evidence |
| --- | --- | --- |
| 2026-10-21 | initial preparation | Historical measurements retained under `bench/results/2026-10-21*`; the original README had no serving definition. Those rates do not describe the November stack. |
| 2026-11-10 | current-r6 | Attention, draft argmax, row statistics, replicated wkv, fused dispatch and extended true-ring rows; the hardware links returned to four Gen2 x16 links. |
| 2026-11-13 | current-r7 | Expanded true-ring row list; source `rounds/stack-r2/out/20261112T192726Z-3009517-r7-equiv-speed`. Fair prefill was removed from this daily composition. |
| 2026-11-13 | current-r7a | Per-request rank-one abliterated adapter, fused dense update and GPU-local worker CPU affinity. |
| 2026-11-14 09:40 | current-r7b | Cumulative fixed640 cooperative native MoE module; source `rounds/current-r7b-r1/out/20261114T052150Z-2097332-qualify` and actual-launcher acceptance `20261114T092653Z-4010770-actual-acceptance`. |

The former PCIe Gen2 x4 README statement belongs to older hardware state. The current serving record states four x16 links at 5 GT/s and 175 W per card. Correctable PCIe replays remain a recorded hardware caveat. Earlier x8 or x4 results are not relabelled as current x16 measurements.

This export was prepared locally without contacting a serving host. The measurement summary is transcribed from the owner's qualification note; the full source report is absent from the local checkout. Historical ledger decisions were read for context, and no queued or experimental candidate was added to the current recipe.
