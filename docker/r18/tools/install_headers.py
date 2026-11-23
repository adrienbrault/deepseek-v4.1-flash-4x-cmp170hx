"""Retain ExLlama headers next to its installed package for plugin builds."""
from pathlib import Path
import shutil
import exllamav3

source = Path(__file__).resolve().parents[1] / "vendor/exllamav3/exllamav3/exllamav3_ext"
dest = Path(exllamav3.__file__).resolve().parent / "exllamav3_ext"
for header in source.rglob("*"):
    if header.suffix in (".h", ".cuh", ".hpp"):
        target = dest / header.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(header, target)
