import importlib.util
from pathlib import Path
import subprocess

package = Path(importlib.util.find_spec("vllm").origin).parent
source = package / "models/deepseek_v41/ssd/row_store.cpp"
subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-pthread",
                str(source), "-o", str(source.with_name("librow_store.so"))], check=True)
