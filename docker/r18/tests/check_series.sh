#!/usr/bin/env bash
# Replay the source series and compare every file to the ported checkout.
set -euo pipefail

kit=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
round=$(dirname -- "$kit")
base=${1:-"$round/work/v034"}
expected=${2:-"$round/work/ported"}
series="$kit/patches/vllm"
python=${PYTHON:-python3}

if [[ $# -gt 2 || ! -d "$base/vllm" || ! -d "$expected/vllm" ]]; then
    echo "Usage: $0 [pristine-v0.34.0-tree ported-tree]" >&2
    exit 2
fi

scratch=$(mktemp -d "${DSV41_SERIES_TMPDIR:-$round/work}/check-series.XXXXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
mkdir -- "$scratch/tree"
cp -R -- "$base/." "$scratch/tree/"

"$python" - "$series" > "$scratch/series" <<'PY'
import json
import sys
from pathlib import Path

series = Path(sys.argv[1])
metadata = json.loads((series / "series.json").read_text())
patches = [entry["patch"] for entry in metadata["commits"]]
if not patches or patches != sorted(p.name for p in series.glob("*.patch")):
    raise SystemExit("Patch order or inventory differs from series.json")
print(*patches, sep="\n")
PY

while IFS= read -r patch; do
    # Do not let an enclosing git worktree interpret paths relative to the repo.
    GIT_CEILING_DIRECTORIES="$scratch" git -C "$scratch/tree" apply -p1 "$series/$patch"
    printf 'Applied %s\n' "$patch"
done < "$scratch/series"

diff -qr -- "$scratch/tree" "$expected"
printf 'Patch series reproduces the complete ported tree\n'
PYTHONPYCACHEPREFIX="$scratch/pycache" "$python" - "$scratch/tree" <<'PY'
import py_compile
import sys
from pathlib import Path

files = sorted(Path(sys.argv[1]).rglob("*.py"))
for path in files:
    py_compile.compile(str(path), doraise=True)
print(f"py_compile passed for {len(files)} Python files")
PY
"$python" "$kit/tests/check_symbols.py" "$scratch/tree"
