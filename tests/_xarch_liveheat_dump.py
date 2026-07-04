"""X-ARCH live-heat + peak_raw dumper  (THROWAWAY diagnostic — underscore).

Purpose
-------
The 2026-07-04 Lenovo confirm run (docs/lenovo_dev_setup.md §8) narrowed the
cross-machine divergence to __unit_hp__ at tick 0 with every captured field
identical. The captured `heat` is POST-clear (~zeros), so the digest cannot see
the quantity HP actually depends on: the LIVE pre-clear `heat` sampled by
combat.apply_environmental_damage. This tool dumps exactly that, per tick, at
the damage call's entry:

  - blake2b hash of the live `gmap.heat` + EVERY nonzero cell (ty, tx, raw)
  - hashes of the occlusion inputs the ray march read (heat_atten,
    dyn_light_atten, solid) — discriminates "march inputs differ" from
    "march output differs"
  - per living unit: footprint tiles, peak_raw (recomputed identically to
    combat), hp BEFORE and AFTER the damage apply (float.hex() — exact bits)

Diff the two machines' files: the FIRST differing line answers the open
question — live-heat cells differ => the C++ deposit side; live heat identical
but hp differs => the Python damage chain. Either way the exact tile/value
delta is in hand.

How to run (NO ARGS — same on every machine)
--------------------------------------------
    <python> tests/_xarch_liveheat_dump.py

Writes tests/_xarch_liveheat_<host>.txt and auto-diffs vs any other host's
file already present in tests/ (same workflow as _xarch_perfield_digest.py).

Env overrides (for same-box experiments, e.g. the MSVC 14.44 build):
    BREACH_PYD_DIR   directory holding breach_physics*.pyd
                     (default: cpp/build/Release)
    BREACH_RUN_TAG   extra label appended to the output filename

Leaves only diagnostics in the tree; commits nothing by itself.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYD_DIR = Path(os.environ.get("BREACH_PYD_DIR", ROOT / "cpp" / "build" / "Release"))
for _p in (ROOT, ROOT / "src", ROOT / "tests", PYD_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

N_STEPS = 30


def _arr_hash(arr: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def main() -> int:
    host = platform.node() or "unknown"
    tag = os.environ.get("BREACH_RUN_TAG", "")
    label = f"{host}__{tag}" if tag else host

    # Import order matters: harness first (sets up the scenario), then the
    # simulation module whose bound name we intercept.
    from field_ab_harness import default_scenario_sim
    import simulation.simulation as sim_mod

    lines: list[str] = []
    tick_box = {"t": -1}

    original = sim_mod.apply_environmental_damage

    def instrumented(units, gmap, ticks_per_second, events=None):
        t = tick_box["t"]
        heat = gmap.heat
        lines.append(
            f"tick={t}\tlive_heat_hash={_arr_hash(heat)}"
            f"\tnonzero={int((heat != 0).sum())}"
            f"\theat_atten={_arr_hash(gmap.heat_atten)}"
            f"\tdyn_light_atten={_arr_hash(gmap.dyn_light_atten)}"
            f"\tsolid={_arr_hash(gmap.solid)}"
        )
        for (ty, tx) in np.argwhere(heat != 0):  # C order == row-major sorted
            lines.append(f"tick={t}\tcell\t{int(ty)}\t{int(tx)}\t{int(heat[ty, tx])}")
        pre = []
        for u in units:
            if not getattr(u, "alive", False):
                continue
            peak_raw = 0
            tiles = []
            h, w = heat.shape
            for (fx, fy) in u.occupied_tiles():
                tiles.append((int(fx), int(fy)))
                if 0 <= fy < h and 0 <= fx < w:
                    v = int(heat[fy, fx])
                    if v > peak_raw:
                        peak_raw = v
            pre.append((u, peak_raw, sorted(tiles), float(u.current_hp)))
        result = original(units, gmap, ticks_per_second, events=events)
        for (u, peak_raw, tiles, hp_before) in pre:
            uid = getattr(u, "id", -1)
            lines.append(
                f"tick={t}\tunit\tid={uid}\tfootprint={tiles}"
                f"\tpeak_raw={peak_raw}"
                f"\thp_before={hp_before.hex()}"
                f"\thp_after={float(u.current_hp).hex()}"
                f"\talive={bool(u.alive)}"
            )
        return result

    sim_mod.apply_environmental_damage = instrumented
    try:
        sim = default_scenario_sim()
        for t in range(N_STEPS):
            tick_box["t"] = t
            sim.set_paused(False)
            sim.step()
    finally:
        sim_mod.apply_environmental_damage = original

    out = ROOT / "tests" / f"_xarch_liveheat_{label}.txt"
    with open(out, "w", newline="\n") as f:
        f.write(f"# liveheat dump v1  host={host} tag={tag} pyd={PYD_DIR}\n")
        f.write("\n".join(lines) + "\n")
    print(f"[liveheat] wrote {out}  ({len(lines)} lines)")

    # Auto-diff vs any other liveheat file present (skip our own).
    others = sorted(p for p in (ROOT / "tests").glob("_xarch_liveheat_*.txt")
                    if p != out)
    for other in others:
        a = out.read_text().splitlines()
        b = other.read_text().splitlines()
        a = [l for l in a if not l.startswith("#")]
        b = [l for l in b if not l.startswith("#")]
        first = None
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                first = (i, a[i], b[i])
                break
        if first is None and len(a) != len(b):
            first = (min(len(a), len(b)), "<EOF>", "<EOF>")
        if first is None:
            print(f"[liveheat] IDENTICAL to {other.name} — live heat, peak_raw "
                  f"and hp all match this run bit-for-bit.")
        else:
            i, la, lb = first
            print(f"[liveheat] FIRST DIVERGENCE vs {other.name} at line {i}:")
            print(f"  this : {la}")
            print(f"  other: {lb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
