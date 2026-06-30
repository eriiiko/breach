"""X-ARCH per-field, per-tick digest dumper  (THROWAWAY diagnostic — underscore).

Purpose
-------
The aggregate 30-tick trajectory digest (``xarch_digest.py`` -> one hash) tells us
two machines DISAGREE, but not WHERE. This script breaks that one hash open: it
emits a hash for EVERY (field, tick) pair over the canonical A/B scenario, plus
the synced unit-state hash per tick. Run the IDENTICAL script on two machines and
``diff`` the two output files — the FIRST line that differs names the exact
(field, tick) where the trajectories first diverge, which names the responsible
solver (e.g. ``water_depth`` -> WaterSolver, ``atmosphere`` -> AtmosphereSolver).

This is the localization tool the X-ARCH Ada findings (docs/xarch_ada_beatB_findings)
asked for: per-field per-tick dumping to find the first diverging (field, tick).

How to run (NO ARGS — dead simple, same on every machine)
---------------------------------------------------------
    C:/Users/steen/anaconda3/python.exe tests/_xarch_perfield_digest.py

It loads the clean CPU build (``cpp/build/Release`` on sys.path), runs 30 ticks of
the default A/B scenario, and writes:

    tests/_xarch_perfield_<host>.txt     (host = platform.node())

Each line is::

    <tick>\t<field>\t<per-field-hash>

in a FIXED (tick, field) order, so two files line up for a plain ``diff``. The
script also prints, to stdout:
  - the host + the aggregate 30-tick trajectory digest (compare to 60bd331f… on
    Ampere as a sanity check that the build is the clean golden), and
  - if a SECOND file from another host is present in tests/, the first diverging
    (field, tick) between this run and that file (a built-in cross-machine diff).

Cross-machine workflow
----------------------
  1. Run here (Ampere)  -> tests/_xarch_perfield_DESKTOP-0E98HUV.txt
  2. Copy that file to the Lenovo, run there -> tests/_xarch_perfield_<lenovo>.txt
  3. ``diff`` the two files (or let step 2's run auto-report the first divergence).

Leaves only diagnostics in the tree; commits nothing.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

from field_ab_harness import capture_trajectory, UNIT_DIGEST_KEY
from field_digest import (
    DIGEST_FIELDS, DIGEST_SPEC_VERSION, _field_bytes, trajectory_digest,
)

N_STEPS = 30
# A stable, machine-independent label for the synced unit-state hash so it sorts
# and diffs alongside the gmap fields (it is not in DIGEST_FIELDS).
UNIT_FIELD_LABEL = "__unit_state__"
GOLDEN_AGGREGATE = "60bd331faccc0b08c11e1ccad3ca75fa6f2aa26232b0b04c1a070b6c65c86ba1"


def _perfield_hash(name: str, arr: np.ndarray) -> str:
    """blake2b-256 over ONE field's spec bytes (name|dtype|shape + raw bytes).
    Same byte recipe as field_digest so a (field,tick) hash here is meaningful."""
    h = hashlib.blake2b(digest_size=32)
    h.update(f"PERFIELD_V{DIGEST_SPEC_VERSION}\n".encode("ascii"))
    h.update(_field_bytes(name, arr))
    return h.hexdigest()


def build_perfield_lines(traj) -> list[str]:
    """One '<tick>\\t<field>\\t<hash>' line per (tick, field), in FIXED order
    (tick ascending; fields in DIGEST_FIELDS order, then the unit-state hash)."""
    lines: list[str] = []
    for t, snap in enumerate(traj):
        for name, _dtype in DIGEST_FIELDS:
            if name not in snap:
                lines.append(f"{t}\t{name}\tMISSING")
                continue
            lines.append(f"{t}\t{name}\t{_perfield_hash(name, snap[name])}")
        # Synced unit state (HP/life/event stream) rides alongside the fields so a
        # combat/kill desync that leaves every gmap cell identical still localizes.
        uhash = snap.get(UNIT_DIGEST_KEY, {}).get("hash", "NO_UNIT_STATE")
        lines.append(f"{t}\t{UNIT_FIELD_LABEL}\t{uhash}")
    return lines


def first_divergence(lines_a: list[str], lines_b: list[str]):
    """Return the first (line_index, a_line, b_line) where two perfield dumps
    differ, or None if identical. Files share a fixed order so index lines up."""
    if len(lines_a) != len(lines_b):
        return (-1, f"<{len(lines_a)} lines>", f"<{len(lines_b)} lines>")
    for i, (la, lb) in enumerate(zip(lines_a, lines_b)):
        if la != lb:
            return (i, la, lb)
    return None


def main() -> int:
    import breach_physics as bp
    pyd = getattr(bp, "__file__", "?")
    host = platform.node()

    traj = capture_trajectory(n_steps=N_STEPS)
    lines = build_perfield_lines(traj)
    agg = trajectory_digest(traj)

    out = ROOT / "tests" / f"_xarch_perfield_{host}.txt"
    header = [
        f"# X-ARCH per-field per-tick digest",
        f"# host={host}",
        f"# pyd={pyd}",
        f"# spec_v={DIGEST_SPEC_VERSION}  n_steps={N_STEPS}",
        f"# aggregate_trajectory_digest={agg}",
        f"# columns: tick<TAB>field<TAB>blake2b256",
    ]
    out.write_text("\n".join(header + lines) + "\n", encoding="utf-8")

    print(f"host                = {host}")
    print(f"pyd                 = {pyd}")
    print(f"wrote               = {out}")
    print(f"per-field lines     = {len(lines)}  ({N_STEPS} ticks x "
          f"{len(DIGEST_FIELDS)+1} fields)")
    print(f"aggregate digest    = {agg}")
    print(f"matches 60bd331f    = {agg == GOLDEN_AGGREGATE}  "
          f"(Ampere clean-build sanity)")

    # Built-in cross-machine diff: if any OTHER host's perfield file is present,
    # report the first diverging (field, tick) against it.
    others = sorted(
        p for p in (ROOT / "tests").glob("_xarch_perfield_*.txt")
        if p.name != out.name
    )
    if others:
        print("\n--- cross-machine first-divergence vs other host files ---")
    for other in others:
        other_lines = [
            ln for ln in other.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")
        ]
        this_lines = [ln for ln in lines]  # already header-free
        d = first_divergence(this_lines, other_lines)
        if d is None:
            print(f"  {other.name}: IDENTICAL (no divergence over {N_STEPS} ticks)")
        elif d[0] == -1:
            print(f"  {other.name}: length mismatch {d[1]} vs {d[2]}")
        else:
            i, la, lb = d
            print(f"  {other.name}: FIRST DIVERGENCE at line {i}")
            print(f"      this ({host}) : {la}")
            print(f"      other         : {lb}")
            tick = la.split('\t', 1)[0]
            field = la.split('\t')[1] if '\t' in la else '?'
            print(f"      => first diverging (field, tick) = ({field}, tick {tick})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
