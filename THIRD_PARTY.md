# Third-party attribution and pins

Original recipe documentation and scripts use [MIT](LICENSE). Modified runtime files retain their own licences. This repository does not redistribute model weights, adapters or native binaries. Local source notices establish the pins below; no upstream site was contacted during this preparation. Missing model and overlay provenance is explicit and blocks a standalone rebuild.

| project | repository and exact pin | licence | role |
| --- | --- | --- | --- |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm), `df3bd094aeea829f27baa0a5282830464f601d9f`; installed image ID `sha256:f760136967d6c4cac9f4de317e2b9a19a170d27519ddeb0ba9aad3e420ec3c27` | Apache-2.0 | Engine base and source overlays; f760 includes subsequent local modifications. |
| vllm-exl3 | [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3), `3c60e6b45b5ec9e845b07aa70b4f974819010ffe` plus the exported patches and installed source snapshot | **AGPL-3.0-only** | EXL3 integration and cooperative-kernel loader. Modified sources and images retain applicable AGPL obligations. |
| ExLlamaV3 fork | [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3), `d968d96dbe56072317329d573bd06e1403eb949b` | MIT | Kernels, mixed-K and cooperative MoE extensions; upstream Turboderp and contributors retain credit. |
| GNmac cooperative source | [GNmac-lab/deepseek-v41-flash-1017-cmp170hx](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx), baseline `bbaf3672914eb1bbbca11de685be5f0c150f577a`, loop-trim source `2ad5741ab1a278d64f33b842706a41717dfbf181` | MIT | Native source lineage and the first four cooperative trim ideas. Native layers five through eight are original local work. |
| Curtis tensor-core port | [CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE), `88eb6e13de0ea5491579fddb195116a4176f4bc7` | Apache-2.0; EXL3 codec MIT | DeepSeek shape port and original hook by Curtis White. |
| Morrowmake kernels | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx), `18305948acba3f4c4d912cdd56f2202d2ebd94e8` | Apache-2.0 | Tensor-core kernel/reference ancestry used by the Curtis port. |
| Morrowmake calibrated policy | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx), v1.0.2 plugin `e11b375ab98ce84a3ee5792364e21bc25dcc78fa` | Apache-2.0 | Calibrated depth policy and Engram methodology; pin recorded by the local release-provenance note. |
| EAQ routed experts | [Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1.1-2.9-bpw), `aee640e2afba93be683bb5b9b420115fe7fbac20` | model terms require verification at this revision; earlier EAQ card recorded MIT | Current routed experts; revision from the retained download manifest. No weights exported. |
| Official checkpoint | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), full repository revision unresolved | MIT recorded by the original recipe; revision-specific notice unresolved | Attention, shared experts, head, DSpark draft and Engram tables. |
| Heretic direction | [pqhaz/DeepSeek-V4.1-Flash-1017-heretic](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic), `12d58ad0878b5008e80d720251adb969a0069412` | unresolved in retained non-output records | Direction used by the external rank-one adapter; no tensors exported. |
| SM80 overlay | [zebgop-ops/dsv41reap-pp](https://github.com/zebgop-ops/dsv41reap-pp), original upstream commit unresolved | Apache-2.0, supplied overlay licence retained | Ancestry of the r17 SM80, pipeline source-shadow and SSD row-store patches; 0xSero row-store attribution retained. |

## Notices retained with source

`licenses/` contains vLLM/SM80 Apache-2.0, ExLlamaV3 MIT, the vllm-exl3 AGPL text, its historical Apache text, NOTICE and third-party notices. The historical Apache file does not relicense vllm-exl3. The cumulative native series applies to the qualified MIT EXL3/GNmac source header whose hash is recorded in [the patch documentation](docs/PATCHES.md). The modified installed AGPL loader has additional local code; the exported source inventory records its exact bytes.

The local attention, row-statistics, wkv, dispatch, target/draft-argmax and r7a plugins are MIT orchestration and kernels over the pinned runtime APIs. Their file-level origins are in [SOURCE-INVENTORY.json](docs/SOURCE-INVENTORY.json). No claim is made that every file in f760 comes directly from the unmodified upstream commit.

## Historical attribution

The retained October documents and instruments also reference [LibertAIDAI's EXL3 pack](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw), the earlier Mia-AiLab EAQ v1 revision `45ee5d0f`, [CurtisAccelerate's checkpoint-layout recipe](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX), [liumorrisclaw's NVMe recipe](https://github.com/liumorrisclaw/dsv41-flash-cmp170hx) and [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker). Their full historical revision/licence receipts are not established by this export; those references are not pinned current build inputs. The upstream [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) project supplies the EXL3 format and MIT kernel ancestry.

The existing optional decode instrument fetches `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` at `6278ecb`, file `tests/bench_decode.py`, blob `d9b24cf29a56c6e215144b267bc874e428b9dcb4`, AGPL-3.0. It is not vendored or executed for the current qualification. See the historical instrument documentation before use.
