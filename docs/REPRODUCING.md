# Reproducing the recorded serving definition

The package mirrors the current-r7f configuration and the source files that can be exported from the local source tree. It does not yet contain every input needed for a standalone build. Source gaps below are blocking, not optional fallbacks. No generated-output directory, captured tensor, model weight, token record or private log was copied.

## Inputs

Use an x86_64 Linux host with four memory-unlocked CMP 170HX cards, each at PCIe Gen2 x16 and a 175 W cap, with the recorded EPYC 7401P and the dated host conditions when comparing measurements. Address GPUs by their locally discovered UUID or serial; this repository contains no machine identifiers. Engram tables remain on NVMe. Store the hybrid checkpoint, official Engram files, derived rank-one adapter and compiler cache outside this repository.

The engine is pinned by installed image ID in `config/served.json`. This is a local image content ID, not a pullable registry digest. The original nightly base is `vllm/vllm-openai:nightly-df3bd094aeea829f27baa0a5282830464f601d9f@sha256:17845138a26f2b67ae78985ad4804518060dc6aa253ef76c8bb52e4fcc898d44`; applying the r17 patches to it alone does not recreate f760. No script pulls an image or selects a newer revision.

`scripts/host.env.example` names the external model and artifact directories. Copy it to ignored `scripts/host.env` and set absolute local paths there. The plan renderer parameterizes every host bind and writes no private paths into tracked files. The full original host argv and its authenticated final mount map were not available as an exportable source artifact; the plan records the documented engine settings and explicit source requirements, with execution disabled.

```bash
bash scripts/serve.sh --print
bash scripts/serve.sh --check
python3 scripts/check-source-package.py
uv run bench/plot.py
bash scripts/check-public-hygiene.sh --tree
bash scripts/check-prose.sh
```

`--print` reads only this repository and emits the parameterized plan. `--check` checks supplied local inputs and reports unresolved prerequisites; it never invokes Docker or a network client. `--run` refuses while `reproduction_complete` is false. This intentional block prevents a partial export from silently serving a different configuration. The original serving lifecycle owns the GPU lock, confirms health and activation, pins each worker to local CPUs, preserves API/engine masks, verifies both model names and removes its own container on exit. That authenticated lifecycle has not been reconstructed or exercised here.

## Source gaps

| required input | retained source reference | state |
| --- | --- | --- |
| exact installed f760 image build chain | r17 kit plus subsequent installed layers | r17 patches are included; the later complete Dockerfile/input chain is absent outside excluded outputs |
| r18 hook overlay | `rounds/moe-gap-r2/out/stage-r18/overlay` | utilities and depth policy source are included; final model, scheduler, graph manager, worker, speculator and rejection hook files are not copied |
| tuned BF16x2 sparse configuration | `rounds/splitk-numerics-r1/out/tables/p2.json` | kernel source is reconstructed; tuned shape table remains unavailable for export |
| r7 true-ring runtime and exact geometry | `rounds/depth-by-load-r1/out/ring-k3/model` and `rounds/ring-rows-r2/rung2/out/assets/extra` | row list is recorded; generated runtime, geometry and native extras are excluded |
| fixed640 native build | `rounds/coop-native-eaq-r1/out/payload` plus the eight cumulative patches | patch series is included; qualified baseline source payload, builder and native library are excluded |
| model-view and adapter reproducibility | EAQ v1.1, official model and Heretic direction receipts | EAQ and direction revisions are known; full official revision, direction licence and derived adapter receipt remain unresolved |
| bounded indexer plugin and k1b native library | exact bounded indexer top-k, read-only native mount | selector and plugin name recorded; plugin source, native build and authenticated mount mapping are not exported |
| measured MoE STACK native bundle and seal | rows-family and K2 fixed16 source generators and selectors, capture-only plugin | source included; exact measured DSOs, authenticated seal digest and shared native baseline builder remain external; no replacement build is authenticated |
| complete benchmark derivation | qualification `report.json`, per-boot prefill logs and acceptance seal | supplied summary is included; full qualification and acceptance files are absent locally |

Resolve these from source-controlled inputs or a separately reviewed export before changing `reproduction_complete`. Then verify patch application on the stated bases, build offline on Linux, inspect the final image and full argv, compile native code for SM80, verify original source/hash receipts and run the serving lifecycle and both-name acceptance gates. Preserve `/health`, model-list and plugin activation checks, GPU-local worker affinity and confirmed container teardown. No run, promotion, network reachability or new performance validation has been performed for this export.

## Current-r7d additive plan

`config/served.json` preserves the r7c engine command and bounded indexer, appends `dsv41_moe_stack` to `VLLM_PLUGINS`, sets `DSV41_MOE_STACK=1` and parameterizes `DSV41_STACK_NATIVE_SEAL`. Its seven read-only `runtime_bind_inputs` entries mirror the final r7d transform: plugin and metadata under site-packages, the exact native bundle at `/opt/stack-native`, the binding at `/opt/stack-code/stack_runtime.py`, and three semantic selector helpers under `/stack-rounds`. `RECIPE_ROOT` denotes this checkout; `RUNTIME_ASSETS_ROOT` denotes externally supplied assets. The native seal is SHA256 of that bundle's `seal.json`, not an arbitrary generated replacement. The launcher prints these settings without resolving private host paths or starting a service. No authenticated seal value is fabricated, and execution remains blocked by the existing source gaps.

## Current-r7e/current-r7f additive plan

The r7f plan retains all r7d engine arguments and STACK binds, appends `dsv41_prefill_7b` to plugin discovery and fixes `DSV41_PREFILL_7B_R7D=1`. Thirteen added read-only binds expose the serving plugin/metadata, six unchanged worker helpers under `/opt/qv`, four helpers under `/opt/prefill-bitwise`, and the measured `_dsv41_mm_try.abi3.so`. Sources are pinned in the inventory. The native library must match the original qualified build receipt; neither that library nor its private receipt is fabricated here. Its generated source baseline remains external, in addition to the older source gaps above.

The container creation options record exactly `--memory 55834574848 --memory-swap 55834574848`. No kernel, model, sequence count or graph descriptor changes at r7f. Host capacity is unverified locally; the old 46 GB measurement label and the r7f packet's 60 GiB budget are not treated as a new inventory. The export still has no authenticated lifecycle executor. `scripts/serve.py` renders the complete documented plan and refuses execution as before.
