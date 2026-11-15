# DeepSeek-V4.1-Flash on 4x CMP 170HX

![Recipe card: four CMP 170HX cards, measured decode aggregates and configured limits, 2026-11-15][card]

Serving configuration, [patch series][patches], [runtime overlays][overlays] and [measurements][results] for [DeepSeek-V4.1-Flash-1017][model] on four NVIDIA CMP 170HX cards (GA100, `sm_80`, 64 GB each after the memory unlock). The single-socket host has an EPYC 7401P, 46 GB of RAM and NVMe-backed Engram tables. All four cards use PCIe Gen2 x16 links and a 175 W cap per card. Tensor parallelism spans all four GPUs.

The current serving definition is **current-r7c**, promoted on 2026-11-15 at 19:45 UTC (20:45 Paris): current-r7b plus exact bounded indexer top-k. It combines [vLLM][vllm] nightly [df3bd094][engine], [EXL3 EAQ v1.1 2.9 bpw routed experts][eaq], the official FP8/MXFP4 components, the model's [DSpark drafter][dspark] at K5, the [cumulative fixed640 cooperative MoE kernel][native] and the bounded indexer (`DSV41_INDEXER_BOUNDED=1`, k1b native library mounted read-only). The raw model is `deepseek-v4.1-flash-1017`; the [per-request rank-one adapter][ablit] exposes `deepseek-v4.1-flash-1017-abliterated`. Weights and adapters are external.

This describes the current serving definition; the retained acceptance record ends with serving stopped. [Reproduction][reproduce] identifies the remaining source and model-pin gaps. The exported [launcher][serve] prints a plan without contacting Docker by default and refuses execution while those gaps remain unresolved.

## Numbers

Measured on **2026-11-15, 17:39–18:30 UTC**, in the single current-r7c ON boot of the OFF/ON/OFF confirmation chain `20261115T173926Z-1666654-indexer-confirm3`, using loaded suite v2 components under the v3 seal. The configuration is TP4 with four Gen2 x16 links and 175 W per card. [Results and conditions][qualification], [machine-readable summary][summary], [results directory][qualification-dir].

| workload | concurrency | output per request | decode aggregate, tok/s | ordered step, ms |
| --- | ---: | ---: | ---: | ---: |
| frozen chat prompts, mostly reasoning prose before code | 4 | 400 | **360.5** | 33.02 |
| same workload | 8 | 400 | **431.5** | 48.80 |
| same workload | 1 | 400 | **158.6** | 19.80 |

Decode uses `decode-prompts-v2`, temperature zero, seed 15 and EOS suppression. Each batch sums the measured request decode rates, `(completion_tokens - 1) / (stream_end - first_content_event)`; each boot averages its scored batches. These are directly measured aggregates. Individual stream statistics are unavailable. Burst-v2 throughput and ordered-v3 step times are separate measurements. These rows do not quantify sampled agent traffic at temperature one or the abliterated variant. [conditions][qualification].

![Decode aggregate at c1, c4 and c8: current-r7c and the previous daily's qualification; dated sources and same-chain controls are shown][decode-chart]

The previous daily's measurements in this figure belong to their dated configurations. The same-chain controls are the r7b OFF boots; the earlier two-boot r7b qualification is separate evidence. [Promotion history][history] retains those results and their conditions.

| nominal cold input tokens per request | concurrency | TTFT, s | aggregate input rate, tok/s |
| --- | ---: | ---: | ---: |
| 2,048 | 1 | 0.74 | 2,779 |
| 2,048 | 4 | 2.09 | 3,254 |
| 2,048 | 8 | 3.18 | 3,474 |
| 8,192 | 1 | 2.30 | 3,561 |
| 8,192 | 4 | 5.86 | 3,780 |
| 8,192 | 8 | 10.17 | 3,801 |

Prefill comes from the same 2026-11-15 ON boot and configuration: median scored batches b0–b2, after one unscored warm batch per point. It uses temperature zero, thinking off, EOS suppression and one output token. Every batch must have zero prompt-cache hits; the host page cache remains warm-steady without a cache drop. Input rates and TTFT are transcribed independently. Nominal lengths are calibrated before the chat template; actual per-request token counts and template overhead are absent from the supplied summary. TTFT includes the request path. [Prefill conditions and rate definition][qualification].

![Current-r7c cold prefill: TTFT and aggregate input rate at 2K and 8K tokens, c1, c4 and c8, measured 2026-11-15][prefill-chart]

## Served configuration

| item | current-r7c |
| --- | --- |
| runtime | [vLLM][vllm] [df3bd094aeea829f27baa0a5282830464f601d9f][engine], installed image ID `sha256:f760136967d6c4cac9f4de317e2b9a19a170d27519ddeb0ba9aad3e420ec3c27` |
| weights | [EAQ v1.1 EXL3 2.9 bpw routed experts][eaq]; [official attention, shared experts, head, draft and Engram components][model] |
| parallelism | TP4, PP1; four CMP 170HX cards |
| power and bus | 175 W per GPU; PCIe Gen2 x16 on each GPU |
| Engram | NVMe row reads, [fused dequantization, early prefetch and bounded host row cache][engram] |
| request window | 262,144 tokens configured |
| scheduling | eight sequences; 2,048 batched tokens; async scheduling and prefix caching |
| graphs | FULL_DECODE_ONLY; capture sizes every integer from 1 through 48; LoRA specialization disabled |
| speculation | [DSpark][dspark] maximum five draft tokens; adaptive verification disabled; [calibrated row-mask features][depth] only through batch one |
| KV | `fp8_ds_mla`; FP8 indexer |
| attention | [tuned BF16x2 split-K table][splitk], [fused SWA metadata and top-k glue][attention]; exact bounded indexer top-k enabled |
| communication | true ring at the [r7 row list][served-plan]; existing INT8 prefill reduction; [replicated Engram wkv][wkv] |
| MoE | [fused dispatch preparation][dispatch] and [cumulative fixed640 native cooperative kernel][native] |
| API | DeepSeek tokenizer, tool and reasoning parsers; raw default plus the [per-request abliterated name][ablit] |
| chat | earlier reasoning preserved with `drop_thinking=false`; image count limit 999 |
| host | GPU workers pinned to their card's local CPUs; engine/API affinity retained |
| other policies | fair prefill and prompt lookup disabled; client sampling defaults unchanged |

## What the stack is

The base r17 image kit supplies SM80 sparse attention, SSD Engram storage, [ExLlamaV3][exl3] integration through [vllm-exl3][vllm-exl3] and tensor-core prefill support on the pinned nightly. The installed f760 image contains later layers; rebuilding the r17 kit alone does not reproduce it. The [ordered patch documentation][patches] gives each exported patch's purpose, target and base, with links into [`patches/vllm/`][vllm-patches], [`patches/vllm-exl3/`][exl3-patches], [`patches/native/`][native], [`patches/engram/`][engram] and [`patches/splitk/`][splitk]. **vllm-exl3 is AGPL-3.0-only**; its [licence][agpl], [NOTICE][notice] and [third-party notices][third-party] apply to the modified integration.

The runtime adds the [split-K precision table][splitk], [cap-1 depth bridge][depth], [target][target-argmax] and [draft sharded argmax][draft-argmax], [attention metadata fusions][attention], [lean row statistics][rowstats], [replicated wkv][wkv], [fused dispatch][dispatch] and expanded true-ring rows. The [r7a plugin][ablit] binds the dense rank-one update per request. R7b supplies the [fixed640 native series][native]; r7c adds exact bounded indexer top-k with the read-only k1b native library. The bounded plugin and native build remain an explicit [source gap][reproduce]. The [source inventory][inventory] records exported hashes; [provenance][provenance] and [third-party inputs][third-party] record their origins and unresolved pins.

The model's DSpark, sparse attention and Engram components are described in the [DeepSeek-V4.1 technical report][report] and [official inference source][dspark]. The [sibling Qwen3.8-Flash-Next recipe for two CMP 170HX cards][qwen-cmp] records the same card family on a different model and serving stack.

## Reproducing

Start with [the reproduction guide][reproduce], [host environment example][host-env], [serving plan][served-plan] and [launcher][serve]. `bash scripts/serve.sh --print` renders the local plan. A complete Linux image build, native compile and GPU acceptance have not been run for this export. The source package still needs the listed runtime assets and full checkpoint/adapter provenance before it can reproduce a serving boot.

## Measurement limits

The tables above cover the current definition's frozen greedy decode and six prefill cells. Current sampled agent throughput, per-stream TPOT, decode sums over a shared overlap window, public benchmark scores, mixed prefill/decode, long-context speed, maximum KV capacity, memory-category use, wall power and abliterated-model performance are not established by these records. [Older measurements][results] describe their own configurations. The fixed-work result is retained with its metadata gaps in the [qualification notes][qualification].

The 2026-11-15 actual-launcher acceptance passed at 19:29–19:45 UTC, boot `20261115T193118Z-2547641`; the retained record reports inherited ready and sampled gates and identity replay passing. The confirmation chain matched raw c8 incumbent mode A throughout; the rule permitted an exact match to either archived incumbent mode A or B. Serving was absent after teardown. Loopback acceptance does not establish client-network reachability. Correctable PCIe errors and replays were recorded. [Acceptance and confirmation evidence][qualification].

![Daily promotion history: dated c1, c4 and c8 aggregates from the r7b and r7c qualifications][history-chart]

[History][history] retains the r7b qualification, acceptance and separate diagnostic profile. [Instrument notes][gotchas] describe the measurement limits. Regenerate the figures with `uv run bench/plot.py` using the [plot script][plot]. Original recipe work uses [MIT][license]; third-party components retain the notices in [NOTICE][notice] and [THIRD_PARTY.md][third-party].

[model]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
[vllm]: https://github.com/vllm-project/vllm
[engine]: https://github.com/vllm-project/vllm/commit/df3bd094aeea829f27baa0a5282830464f601d9f
[exl3]: https://github.com/turboderp-org/exllamav3
[vllm-exl3]: https://github.com/vcruz305/vllm-exl3
[eaq]: https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw
[dspark]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py#L1022
[report]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/DeepSeek_V41_Tech_Report.pdf
[qwen-cmp]: https://github.com/adrienbrault/qwen3.8-flash-next-2x-cmp170hx
[card]: docs/img/recipe-card-d6667df1.png
[decode-chart]: docs/img/decode-concurrency.svg
[prefill-chart]: docs/img/prefill.svg
[history-chart]: docs/img/history.svg
[qualification]: bench/results/2026-11-15-current-r7c-qualification/NOTES.md
[summary]: bench/results/2026-11-15-current-r7c-qualification/summary.json
[qualification-dir]: bench/results/2026-11-15-current-r7c-qualification/
[results]: bench/results/
[history]: docs/HISTORY.md
[gotchas]: docs/GOTCHAS.md
[reproduce]: docs/REPRODUCING.md
[third-party]: THIRD_PARTY.md
[notice]: NOTICE
[license]: LICENSE
[agpl]: licenses/vllm-exl3-AGPL-3.0-only.txt
[provenance]: docs/PROVENANCE.md
[inventory]: docs/SOURCE-INVENTORY.json
[patches]: docs/PATCHES.md
[overlays]: docker/overlays/
[vllm-patches]: patches/vllm/
[exl3-patches]: patches/vllm-exl3/
[native]: patches/native/
[engram]: patches/engram/
[splitk]: patches/splitk/
[depth]: docker/overlays/dsv41_depth/
[target-argmax]: docker/overlays/dsv41_lora_argmax/
[draft-argmax]: docker/overlays/dsv41_draft_argmax/
[attention]: docker/overlays/dsv41_attn_port/
[rowstats]: docker/overlays/dsv41_rowstats/
[wkv]: docker/overlays/dsv41_wkv_replica/
[dispatch]: docker/overlays/dsv41_dispatch_fuse/
[ablit]: docker/overlays/dsv41_ablit_r7/
[serve]: scripts/serve.sh
[host-env]: scripts/host.env.example
[served-plan]: config/served.json
[plot]: bench/plot.py
