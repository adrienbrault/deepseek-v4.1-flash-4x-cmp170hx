# DeepSeek-V4.1-Flash on 4x CMP 170HX

Serving recipe for [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on four NVIDIA CMP 170HX cards (GA100, sm_80, 64 GB each after the memory unlock, PCIe Gen2 x4) in a single-socket EPYC 7401P host with 46 GB of RAM and three NVMe drives.

No configuration is served yet. This repository is in preparation.

## Approach

Routed experts use an EXL3 2.9 bpw quantization. Attention, shared experts, the output head, the DSpark draft and the Engram tables keep the official FP8 and MXFP4 weights. The two Engram tables do not fit in host RAM, so their rows are read from NVMe on demand.

## Status

| item | state |
| --- | --- |
| Runtime base | under evaluation |
| Checkpoint | hybrid of official 1017 weights and EXL3 2.9 bpw routed experts |
| Engram | served from NVMe |
| Numbers | none yet |

Experimental measurements and profiling instruments are described in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

See [THIRD_PARTY.md](THIRD_PARTY.md) for the projects this recipe builds on.
