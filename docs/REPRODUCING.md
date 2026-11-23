# Reproducing the recorded serving definition

The [launch plan](../config/served.json) records current-r18b. It contains the arguments reconstructed from the redacted incumbent, the vLLM 0.34.0 derivation and the sealed r18b policy. The recorded installed image ID and the pinned release manifest digest are distinct. The release image is the first layer of the [Dockerfile](../docker/r18/Dockerfile). The rest of the image depends on the inputs below.

## Inspect the plan

Use a Linux x86-64 machine with four memory-unlocked CMP 170HX GPUs and the NVIDIA container runtime for serving. Use four Gen2 x16 links and a 175 W cap per card to compare to the headline boot. Keep the checkpoint, adapter, two Engram tables and writable compiler caches outside the repository. Copy [`scripts/host.env.example`](../scripts/host.env.example) to ignored `scripts/host.env` for local input paths.

```sh
bash scripts/serve.sh --print
bash scripts/serve.sh --check
python3 scripts/check-source-package.py
python3 bench/plot.py --headline bench/results/2026-11-24-current-r18b-headline/summary.json
```

`--print` emits the parameterized settings. `--check` verifies the policy hash and checks local directories, then reports missing export inputs. `--run` refuses while the authenticated lifecycle and required inputs are pending. The original lifecycle acquires the exclusive GPU lock, pins workers to their local CPUs, warms the server, checks ring activation and both adapter states, and owns cleanup. The portable launcher has no executor for that lifecycle.

## External build and launch inputs

| Required input | Available in this export | Remaining evidence or asset |
| --- | --- | --- |
| vLLM 0.34.0 for Linux amd64 | vLLM commit and registry digest in `config/served.json`, core source patches and checked source hashes | Base image and exact final build-context receipt for the installed image |
| EXL3 and build dependencies | vllm-exl3 and ExLlamaV3 revisions and licences, inherited and r18 integration patches, verified source-file hashes | Staged `vendor/vllm-exl3`, `vendor/exllamav3` and offline wheel set with original pins and notices |
| Native row store, cooperative MoE and bounded indexer | Integration and source selectors, native build scripts, digest checks and fixed640 delta | Qualified source snapshots, source receipts and rebuilt native seals |
| True-ring implementation | FlashInfer 0.7.3 base file hashes, CUDA delta, loader delta and gate harness | Pinned loader base, policy and extra geometry, native input bundles and original build receipt |
| Measured decode and eager prefill MoE | Runtime packages, graph guards and selectors | Measured stack libraries and authenticated seal, eager-prefill build source and original native receipt |
| Engine data and policy | SHA-verified adaptive verification and calibrated row setting | Complete launch configuration mount and tuned sparse-attention shape table |
| Model and adapter | EAQ revision, reference code and direction revision in [third-party pins](../THIRD_PARTY.md) | Full official weight revision, adapter build receipt and revision-specific terms for the unresolved model inputs |
| Host lifecycle | Sanitized bind targets, reconstructed environment and server arguments | Authenticated final argv and private lifecycle receipt for the seal |

The Dockerfile is for an assembled offline build context. It expects the [`docker/r18/`](../docker/r18/) source files, inherited [vllm-exl3 patches 0001–0009](../patches/vllm-exl3/) staged before 0010–0011, and the external `vendor/`, `offline/`, `native/` and `adaptive-verify/` inputs listed above. The pinned image default and true-ring build default reflect the recorded r18b recipe. An image built from this checkout alone would lack those external inputs. [Provenance and source inventory](PROVENANCE.md)

The local checks cover source bytes, Python syntax, the launch plan and archived measurements. Image build, native compilation, hardware acceptance and server execution require the external environment.
