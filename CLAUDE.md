# Working agreement — DeepSeek-V4.1-Flash on four CMP 170HX cards

Read this agreement before changing the repository. `AGENTS.md` is a symlink to this file.

## Scope

This repository will hold the serving recipe for DeepSeek-V4.1-Flash-1017 on the 4x CMP 170HX box: image build, launcher, patch series, benchmark instruments and measurements. The box has 46 GB of host RAM, so the two Engram tables (about 203 GB) are served from NVMe. The README describes the served state; until a configuration serves, it states that no configuration is served yet.

## Visibility

The owner will publish this repository later. Treat its contents and every future commit as public now. Commit identity: `Adrien Brault <adrien.brault@gmail.com>`; do not substitute a noreply address. Do not push until the owner asks.

- Run `bash scripts/check-public-hygiene.sh` before every commit and `bash scripts/check-public-hygiene.sh --tree` before publication.
- Stage explicit paths only. Never use `git add .` or `git add -A`.
- Keep host paths, private infrastructure references, nonpublic addresses, credentials, machine identifiers, session links and assistant attribution out of published files and commit messages. Refer to the host as "the 4x CMP 170HX box".
- `scripts/host.env` is local and ignored. Never publish boot logs or generated benchmark text without reviewing and removing private data.
- Keep `LICENSE` (MIT for original work) and `THIRD_PARTY.md`. Credit every patch, kernel, overlay and idea with its origin, commit and license in the same change that introduces it.

## README

The opening sections describe the served state: introduction, `## Numbers`, `## Served configuration`, `## What the stack is`. A row not yet re-measured on the served configuration is omitted, not labelled. Promotion history belongs in `docs/HISTORY.md`, measurement analysis in `bench/results/<dir>/NOTES.md`, instrument traps in `docs/GOTCHAS.md`.

Per-stream decode rate and a directly measured decode aggregate (sum of concurrent stream decode rates) are the preferred decode metrics. Never multiply a per-stream statistic by concurrency. Report time to first token separately.

## Prose

- Run `bash scripts/check-prose.sh <file>` on every Markdown file written.
- Declarative, factual sentences. No promotional language, rhetorical questions, exclamation marks or jokes.
- Every performance number carries its UTC date, results directory, workload, concurrency, output length, sampler and configuration.
- A paragraph occupies one source line.

## Patches

Patches form one ordered, numbered series. Each has a one-line purpose, the upstream file it targets and the base it applies to. Pin every third-party input (repository, commit or image digest, license). A local static check does not establish that an image builds or a GPU workload runs; say which checks ran where.
