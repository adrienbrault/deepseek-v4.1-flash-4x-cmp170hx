# Startup cache validation

Measured on 2026-10-21, image `dsv41:int-r17`, TP4 DSpark K5 with TC prefill and medium-batch v2, vision enabled and sparse disabled. Results directory: `2026-10-21T112351Z-dsv41-startup`; baseline: `2026-10-21T104329Z-dsv41-fixed-tc`. This compares startup with unchanged numerical configuration rather than token throughput. Warm startup began at 11:23:51 UTC and readiness was observed at 11:31:17 UTC. It includes a full initial preflight cache miss.

| Stage | Baseline | Persistent compiler caches |
|---|---:|---:|
| Launch to observed readiness | 591 s | 446 s |
| Model loading | 280.04 s | 278.92 s |
| Engine initialization | 181.92 s | 36.86 s |

The observed total saving is 145 seconds (24.5%). Engine initialization accounts for that saving: 145.06 seconds less. Triton and TileLang caches are bind-mounted at their original container paths, separated by image digest. Weight loading is unchanged within 1.2 seconds. This is one sequential startup comparison.

The first common preflight ran all checks in 74.052 seconds. After startup the identical inputs reused the successful receipt in 0.289 seconds. The next unchanged boot is estimated at about 372 seconds (6m12s) by replacing the measured miss duration with the measured hit duration in this boot. That complete boot has not been run. No extra reload was made to measure the sum.

Six known answers, vision and prefix-cache usage reporting passed (12,544 of 12,618 prompt tokens cached). The model remains loaded after the successful screen; no automatic restoration of the other model is enabled. Nine launcher tests passed on Linux, including cache invalidation and failure cases. Static serving validation had 13 passes, zero failures and two unresolved-expansion warnings; this is not a daily promotion or a full new serving qualification.

Checkpoint pre-sharding was deferred because the EXL3 arena planner and native handles need a compatible reconstruction path and parity checks. No generic loader switch or checkpoint conversion was made.
