"""Load the real extension artifacts directly, using the exact runtime Torch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig

import torch

native, torch_site = (Path(value).resolve() for value in sys.argv[1:3])
assert not torch.cuda.is_available(), "native build check must be CPU-only"
assert Path(torch.__file__).resolve().is_relative_to(torch_site)
suffix = sysconfig.get_config_var("EXT_SUFFIX")
site = native / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
modules = [
    "sglang.srt.mem_cache.rust_tree_core.mem_cache",
    "sglang.srt.rust_extensions._multimodal",
]
artifacts = {}
for name in modules:
    path = site / (name.replace(".", "/") + suffix)
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path.resolve()
    artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(dict(event="complete", native=str(native), torch=str(torch_site),
                     torch_version=torch.__version__, modules=artifacts, gpu_used=False)))
