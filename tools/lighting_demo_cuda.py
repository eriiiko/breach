"""Launch lighting_demo with the CUDA build + temperature on the GPU (S1 milestone).

The game normally imports breach_physics from cpp/build/Release (CPU-only). This
launcher pre-loads the CUDA build (cpp/build_cuda) so `import breach_physics`
resolves to it everywhere, registers the CUDA runtime DLLs, and flips the
temperature pass onto the GPU. Output is bit-identical to CPU (the S1 gate
proves it) — this just runs the GPU path in the real engine loop.

Run with the anaconda 3.11 interpreter:
    C:/Users/steen/anaconda3/python.exe tools/lighting_demo_cuda.py [demo flags]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _cuda_dll_dir() -> str:
    cp = os.environ.get("CUDA_PATH")
    cands = []
    if cp:
        cands.append(Path(cp) / "bin")
    cands.append(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"))
    for c in cands:
        if c.is_dir() and list(c.glob("cudart64_*.dll")):
            return str(c)
    raise RuntimeError("CUDA runtime bin dir not found (set CUDA_PATH)")


# Register CUDA DLLs (py3.8+ does not search PATH for ext-module deps), then put
# the CUDA build FIRST so breach_physics resolves to it before lighting_demo's
# own sys.path setup runs.
os.add_dll_directory(_cuda_dll_dir())
sys.path.insert(0, str(ROOT / "cpp" / "build_cuda"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import breach_physics as bp  # noqa: E402

assert getattr(bp, "HAS_CUDA", False), "expected the CUDA build (cpp/build_cuda)"
print("CUDA build loaded:", bp.cuda_device_info(), flush=True)
bp.set_temperature_backend(True)
print("temperature backend -> GPU", flush=True)

import lighting_demo  # noqa: E402

lighting_demo.main()
