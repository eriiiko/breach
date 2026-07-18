"""X-ARCH digest runner (CUDA-S0 seed) — the only real cross-GPU proof.

Captures the canonical A/B scenario's trajectory and folds it into ONE
``trajectory_digest`` (field_digest spec v1). Run it on each machine/backend and
diff the recorded numbers: the per-tick integer fields are bit-identical across
machines iff the digests match.

  - **CPU baseline (today):** the CPU-int golden. Every merged kernel's CUDA
    digest must equal this on the same machine (the P1 gate, in-process) AND the
    Ada/Turing replay must equal it cross-machine (the X-ARCH gate).
  - **GPU (S1+):** run with PHYSICS_BACKEND[_<solver>]=cuda once kernels exist;
    this same runner then records ``digest_<host>_cuda_<arch>.txt``.

Usage:
    C:/Users/steen/anaconda3/python.exe tests/xarch_digest.py [--steps N] [--write]

Records ``<host>  <backend>  <pyd-arch>  <digest>`` so a human (Erik owns the
hardware) can eyeball-diff Ampere vs Ada when the Lenovo lands. The committed
``XARCH_PENDING.md`` lists kernels merged on one arch awaiting the other's beat.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--write", action="store_true",
                    help="also write digest_<host>_<backend>_<arch>.txt")
    args = ap.parse_args()

    import breach_physics as bp
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest, DIGEST_SPEC_VERSION
    from simulation.entities.serialize import ENTITY_DIGEST_KEY

    backend = os.environ.get("PHYSICS_BACKEND", "cpu")
    arch = "gpu" if getattr(bp, "HAS_CUDA", False) else "cpu"
    host = platform.node()

    traj = capture_trajectory(n_steps=args.steps)
    digest = trajectory_digest(traj)

    line = f"{host}\t{backend}\t{arch}\tspec_v{DIGEST_SPEC_VERSION}\tsteps={args.steps}\t{digest}"
    # A4: entity-present artifacts record section presence + the registry
    # content-hash, so a cross-machine mismatch is attributable in one diff
    # (impl note critique 3+4). Entity-free runs leave the line unchanged.
    carrier = traj[-1].get(ENTITY_DIGEST_KEY) if traj else None
    if carrier is not None and carrier["n_entities"] > 0:
        line += (f"\tents={carrier['n_entities']},esect_v1,"
                 f"reg={carrier['registry_hash'][:12]}")
    print(line)
    if args.write:
        out = ROOT / "tests" / f"digest_{host}_{backend}_{arch}.txt"
        out.write_text(line + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
