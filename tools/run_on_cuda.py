"""Run the Breach game with physics on the GPU (CUDA), all backends ON.

This is the GPU launch path. It is ADDITIVE — the normal CPU launch
(``python main.py``) is completely untouched and still imports the CPU
build from ``cpp/build/Release``. This wrapper instead points
``import breach_physics`` at the CUDA build in ``cpp/build_cuda`` (mirroring
``tests/cuda_harness.py``), asserts a usable GPU, flips ALL solver backends
to CUDA, then hands off to ``main.main()`` — it does NOT duplicate the game
loop.

Speed is not the goal: each kernel does its own malloc/H2D/D2H per call (the
"per-call path"). It RUNS on the GPU and is bit-identical to the CPU solvers;
that is the point — Erik can SEE the sim running with physics on the card.

Usage (the CUDA build must exist first — run ``cpp/build_cuda.bat``):

    C:/Users/steen/anaconda3/python.exe tools/run_on_cuda.py
    C:/Users/steen/anaconda3/python.exe tools/run_on_cuda.py --res 2   # 2x denser grid
    C:/Users/steen/anaconda3/python.exe tools/run_on_cuda.py --windowed

Equivalent shortcut: ``python main.py --cuda`` (main.py routes --cuda here).

All 7 solvers run on the GPU: the 6 field solvers (temperature, water, smoke,
wave, fire, atmosphere) dispatch inside PhysicsEngine::step, and the 7th — the
raycaster (the fire->heat ray cast in PhysicsRunner.cast_fire_heat) — is now
live-wired too (set_raycaster_backend, CUDA-S2 live). The cast's synced `heat`
output is bit-identical CPU<->GPU (the S2 gate); the light channels it also
produces are render-only / deterministic-exempt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUDA_BUILD_DIR = ROOT / "cpp" / "build_cuda"


def _cuda_dll_dir() -> Path | None:
    """The CUDA runtime ``bin`` dir holding cudart64_*.dll (CUDA_PATH or the
    pinned v12.4 install). Mirrors tests/cuda_harness.cuda_dll_dir."""
    cands = []
    cp = os.environ.get("CUDA_PATH")
    if cp:
        cands.append(Path(cp) / "bin")
    cands.append(
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"))
    for c in cands:
        if c.is_dir() and list(c.glob("cudart64_*.dll")):
            return c
    return None


def _cuda_pyd() -> Path | None:
    hits = list(CUDA_BUILD_DIR.glob("breach_physics*.pyd"))
    return hits[0] if hits else None


def setup_cuda_import() -> None:
    """Put the CUDA build on sys.path BEFORE the CPU build and register the
    cudart DLL dir, so ``import breach_physics`` resolves to the GPU build.

    Must run BEFORE main.py inserts ``cpp/build/Release`` — so this wrapper
    inserts the CUDA dir at index 0 and imports breach_physics here, locking
    the module into sys.modules before main.py's own (later, lower-priority)
    path insert can matter.
    """
    pyd = _cuda_pyd()
    if pyd is None:
        raise SystemExit(
            "CUDA build not found at cpp/build_cuda/breach_physics*.pyd.\n"
            "Build it first:  cpp/build_cuda.bat\n"
            "(The CPU game still runs with:  python main.py)")
    dll = _cuda_dll_dir()
    if dll is None:
        raise SystemExit(
            "CUDA runtime DLLs (cudart64_*.dll) not found. Set CUDA_PATH or "
            "install the CUDA Toolkit v12.4.")
    os.add_dll_directory(str(dll))
    # CUDA build first; then repo root + src so the game's own modules import.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(CUDA_BUILD_DIR))


# The seven live-dispatched GPU backends. The first six are the field solvers
# (PhysicsEngine::step); the seventh — the raycaster (the fire->heat ray cast in
# PhysicsRunner.cast_fire_heat) — is now live-wired (CUDA-S2 live) so --cuda is a
# full 7/7. cast_fire_heat reads the raycaster flag per tick (heat bit-identical).
_BACKEND_SETTERS = (
    "set_temperature_backend",
    "set_water_backend",
    "set_smoke_backend",
    "set_wave_backend",
    "set_fire_backend",
    "set_atmos_backend",
    "set_raycaster_backend",
)


def enable_all_backends(bp) -> None:
    """Assert a usable GPU and flip every live solver backend to CUDA."""
    if not getattr(bp, "HAS_CUDA", False):
        raise SystemExit(
            "Imported breach_physics has HAS_CUDA=False — this is the CPU "
            "build. Run cpp/build_cuda.bat and retry.")
    if not bp.cuda_available():
        raise SystemExit(
            "breach_physics.cuda_available() is False — no usable CUDA device. "
            "Check the GPU / driver / CUDA Toolkit install.")
    try:
        print("[run_on_cuda] GPU:", bp.cuda_device_info())
    except Exception as e:  # device_info is best-effort, never fatal
        print(f"[run_on_cuda] (cuda_device_info unavailable: {e})")
    for name in _BACKEND_SETTERS:
        getattr(bp, name)(True)
    print("[run_on_cuda] backends ON (7/7):",
          ", ".join(n.replace("set_", "").replace("_backend", "")
                    for n in _BACKEND_SETTERS))


def main() -> None:
    setup_cuda_import()
    import breach_physics as bp
    enable_all_backends(bp)
    # Hand off to the real game entry (no duplication of the loop). main.main()
    # imports the already-loaded CUDA breach_physics from sys.modules and reads
    # --res itself.
    import main as game
    game.main()


if __name__ == "__main__":
    main()
