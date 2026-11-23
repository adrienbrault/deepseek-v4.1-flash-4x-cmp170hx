# Export provenance

Prepared from local source records on 2026-11-14. No upstream fetch, serving-host access, GPU run, image build, commit or publication occurred. The structure follows the existing CMP 170HX Flash-Next and two RTX 5090 recipe repositories: current serving state first, dated result directories, `scripts/` launchers, `docs/HISTORY.md`, `docs/GOTCHAS.md` and generated `docs/img/` charts.

`SOURCE-INVENTORY.json` records the original relative source location, licence and SHA256 for directly exported source files. `patches/manifest.json` records ordered patches and hashes after removal of assistant trailers from mail metadata; no diff body or copyright statement was changed. Engram overlay files and their patches are regenerated with the retained source transformations over hash-checked non-output inputs. Split-K overlay files are regenerated from the retained reference sources and transformations. The source verifier parses those modules without importing torch or running kernels. Both regenerated Engram files and both regenerated split-K files also match the current r7a served-source pin table exactly.

The AGPL loader is copied from the retained installed-f760 source snapshot, with repository/commit/licence attribution in `THIRD_PARTY.md` and original licence texts under `licenses/`. This is source code, not a capture of model inputs or outputs. No files whose source location has an `out` path component are exported. Native patches are copied from `rounds/moe-trim-r1/patches`, not from generated payloads. Generated model and adapter artifacts remain external.

The qualification summary transcribes the owner's current-r7b speed note; its referenced full report is absent locally. The profile summary derives only aggregate rank-zero operation shares from the retained partial report, excluding addresses and trace payloads. These are dated historical measurements, not newly executed tests. Unknown pins and missing assets are enumerated in `REPRODUCING.md` and are not replaced by guessed revisions.

## Current-r7c documentation update, 2026-11-15

The [r7c qualification summary](../bench/results/2026-11-15-current-r7c-qualification/summary.json) transcribes the retained headline table and committed confirmation and promotion RESULTS and LEDGER entries. No raw traffic, sealed report or generated source payload was copied. The [notes](../bench/results/2026-11-15-current-r7c-qualification/NOTES.md) distinguish burst throughput, ordered step timing and actual-launcher acceptance. The [serving plan](../config/served.json) records the bounded selector and plugin while retaining the execution block and explicitly listing the missing plugin/native assets.

README upstream links use the repository URLs and pins already recorded in [THIRD_PARTY.md](../THIRD_PARTY.md), plus the retained official-source survey's report filename and reference-code paths. Commit and file links are formed from those exact repositories, revisions and paths; they were not checked over the network. Charts use only the published summaries and the [promotion index](HISTORY.md), with no inferred stream statistics or memory breakdown.

## Current-r7d mirror, 2026-11-16

The [r7d summary](../bench/results/2026-11-15-current-r7d-qualification/summary.json) takes every new absolute number solely from the current-r7d section of `NUMBERS.md`, dated 2026-11-15 and naming `20261115T214428Z-3587331-moe-stack-confirm`. No r7c prefill cell is relabelled as served r7d prefill. The same-hour relative prefill observation is dated 2026-11-16. Confirmation and acceptance context comes from the retained source results and operating record; no host was contacted.

Layer 60 exports exact non-output source bytes for the single-bank plugin, shared binding, three selector helpers and three native source generators. Their source-relative locations, commits, MIT licences and SHA256 values are in [SOURCE-INVENTORY.json](SOURCE-INVENTORY.json). The serving plan preserves r7c engine arguments and adds precisely the plugin, two environment values and seven read-only binds from the r7d transform. Host bind roots and the unavailable authenticated native-seal digest are parameters. The inherited full argv and lifecycle gaps remain; this is a static mirror, not a newly qualified serving executor.


## Current-r7e/current-r7f mirror, 2026-11-16

The retained promotion/acceptance records establish r7e selected at 09:08 UTC and r7f at 18:25 UTC on 2026-11-16; neither selection restored serving. The local mirror copies only non-output runtime source, records its last source commit and exact content hash, and preserves file-level MIT/Apache notices. Layers 61/62 derive the configuration in order. The exact 7b native library/build receipt remains an explicit source gap. No network access, image build, native compile or GPU validation was repeated.

The r7e suite summary transcribes the 2026-11-16 retained per-boot table from `suite-r7e-r1/RESULTS.md`, including its failed report-window check. New headline results will be allowlist exports from headline-report/2; prompts, token IDs, source paths and report prose are excluded. Synthetic tests write only temporary output and do not become measured results.

## Current-r18b export

The [r18 build source](../docker/r18/) comes from the retained image source revision `698723db6b76b5983fee38aa19c577e40c5c8186`. The root Dockerfile pins the recorded Linux amd64 release digest and defaults to the served true-ring build; the container ID remains the independently recorded installed image. Patched integration code, vendor identity manifests and runtime source remain versioned separately from the older r7 files. [SOURCE-INVENTORY.json](SOURCE-INVENTORY.json) pins the exported bytes and records the version of their source.

The launch plan composes the redacted r7h parent capture, the v0.34 argument derivation and the current-r18b packet. The packet pins the policy hash, the image ID, the serving profile and the four API names. The retained headline window references seal `seal-nORCnYLm` in source commit `ac8b0c0a333f683a3b96948ab57bf09a4c0b9917`. The bundled policy bytes from the earlier r18b acceptance workload match the packet SHA256. The original sealed final argv and lifecycle receipt for the named seal are absent locally. The export preserves known engine settings, replaces host paths and device IDs with named inputs, and leaves the original image, native build chain and authentication material external.

The [2026-11-24 results](../bench/results/2026-11-24-current-r18b-headline/) are a transcription of the x16 headline ledger entry. The earlier x8 report documents the protocol and supplies the hash-matched policy. It supplies zero numeric benchmark cells for the x16 summary. The [reproduction guide](REPRODUCING.md) lists the remaining build and serving gaps.
