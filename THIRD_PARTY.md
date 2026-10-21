# Third-party attribution

Original documentation, scripts and benchmark instruments in this repository use [MIT](LICENSE). The runtime images, their dependencies and the checkpoints keep their own licenses. This repository does not redistribute weights.

| project | source and license | role |
| --- | --- | --- |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | [Model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT | Official weights: attention, shared experts, head, DSpark draft, Engram tables. |
| LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw | [Model card](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), MIT | EXL3 2.9 bpw routed experts used by the first hybrid. |
| Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw | [Model card](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw), revision `45ee5d0f`, MIT | Planned source of the routed experts. |
| CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX | [Project](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX) | Hybrid checkpoint layout and TP4 launch for 4x CMP 170HX. |
| CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE | [Project](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE), Apache-2.0 | Tensor-core EXL3 MoE kernels and hook. |
| vcruz305/vllm-exl3 | [Project](https://github.com/vcruz305/vllm-exl3), commit `3c60e6b`, AGPL-3.0-only (historical Apache-2.0 text retained in its repository) | EXL3 quantization plugin for vLLM. Patches to it and images that contain it carry AGPL-3.0 obligations. |
| vcruz305/exllamav3 | [Fork](https://github.com/vcruz305/exllamav3), commit `d968d96`, MIT | ExLlamaV3 kernels with the coop MoE extension. |
| turboderp-org/exllamav3 | [Project](https://github.com/turboderp-org/exllamav3), MIT | EXL3 format and kernels. |
| vLLM | [Project](https://github.com/vllm-project/vllm), Apache-2.0 | Inference engine. |
| Morrowmake/vllm-cmp170hx | [Project](https://github.com/Morrowmake/vllm-cmp170hx), Apache-2.0 | CMP 170HX runtime, host all-reduce. |
| zebgop-ops/dsv41reap-pp | [Project](https://github.com/zebgop-ops/dsv41reap-pp), Apache-2.0 | sm_80 overlay for the DeepSeek-V4.1 vLLM image, under evaluation as the runtime base. |
| liumorrisclaw/dsv41-flash-cmp170hx | [Project](https://github.com/liumorrisclaw/dsv41-flash-cmp170hx), MIT | Prior art for serving the Engram tables from disk on 4x CMP 170HX. |
| amoghmunikote/cmpunlocker | [Project](https://github.com/amoghmunikote/cmpunlocker) | Host memory unlock and driver tooling used in the setup. |

Licenses not stated above were not verified when this file was written; check each project before redistributing its code.


The optional loading-only patches documented in [LOADER-CANDIDATES](docs/LOADER-CANDIDATES.md) retain the target projects' licenses: plugin patch 0008 is AGPL-3.0-only and vLLM patch 0026 is Apache-2.0. They add no third-party dependency. The document pins the source revisions, installed base hashes and candidate hashes.
