"""Gate (a) capture — WHOLE-ENGINE byte-identity across the FUEL-FRACTION axis.

The fire logistic's fuel term used to divide by ONE GLOBAL:

    F = clamp01(wall_hp[i] / fuel_ref)          fuel_ref = 60.0

``fuel_ref`` is WOOD's hp, so every material with a different hp read a
permanently wrong "fraction of my fuel remaining": a full-health furniture crate
(hp 30) reported F = 0.5. The axis makes the divisor the tile's OWN material hp,
baked per material into ``GameMap.fuel_recip`` (a ``make_recip`` reciprocal, so
the sim path keeps its no-divide contract).

The claim THIS script gates is the PLUMBING claim, not the behaviour claim:

    with every tile's ``fuel_recip`` pinned back to the global ``fuel_ref``, the
    whole engine must be byte-identical to the pre-patch build on EVERY
    scenario, tolerance ZERO.

The behaviour change at the real per-material divisor is gate (b), and it is
EXPECTED (furniture only: F 0.5 -> 1.0).

This is a BEFORE/AFTER artifact, not a self-checking gate (deliberately no
``test_`` prefix — pytest does not collect it, and pinning digests here would
manufacture a red the moment Erik's fire re-tune moves a dial). It is
VERSION-AGNOSTIC on purpose: the pin is a ``hasattr`` guard, so the identical
file runs against a pristine PRE-patch tree (where it is a no-op, because the
law already divides by the global) and against the patched tree.

WHY THE PIN GOES ON THE MATERIAL TABLE, not just the grid: ``on_tile_changed``
re-derives a tile's ``fuel_recip`` from the table whenever its material changes
— and a crate burning through is exactly that — so a grid-only pin would silently
un-pin itself mid-run. Pinning the table column pins the grid AND every future
patch of it.

Scenarios: the five the cool-shift axis gated on (default 16x16 space room,
planetside ambient sky, space breach, FURNITURE BURN, sealed room) PLUS the O2
split's ``burning_fuel_22x22`` — fire seeded directly ON wood and ON a crate, so
the fuel term is stepped from tick 0. NOTE, and this is why that last one is
mandatory: ``field_ab_harness.default_scenario_sim`` has ``flammable.sum() == 0``
and seeds fire on AIR, so it is VACUOUS for any fuel-law change on its own.

Usage (from a repo root that has cpp/build/Release built):

    python tests/fuel_fraction_axis_gate_a_capture.py <outdir>

writes ``<outdir>/<scenario>.npz`` holding every captured field for every tick
(so the comparison is genuinely per-tick per-CELL, not just a hash), and prints
one canonical trajectory digest per scenario. Run it in the pristine tree and in
the patched tree with different outdirs, then diff the printed digests and
compare the npz files cell by cell.

A second mode, ``--unpinned``, captures the SAME scenarios with NO pin at all.
Every scenario whose fuel is wood (hp 60 == fuel_ref) must STILL be byte-
identical to the pristine tree in that mode — the sharper statement that the
axis moves only what it was meant to move — while the furniture scenarios are
expected to diverge (that divergence IS the fix).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp   # noqa: E402

N_TICKS = 30


def _pin_fuel_recip_to_global(sim):
    """Back-compat pin: every material's fuel reciprocal := 1/``fuel_ref``.

    On a pristine (pre-patch) tree the material table has no ``fuel_recip``
    column and this is a no-op — which is exactly right, because there the
    divisor IS the global. On the patched tree it restores the old normaliser,
    so the two trees must then agree bit-for-bit.
    """
    tbl = getattr(sim.gmap, "materials", None)
    if tbl is None or not hasattr(tbl, "fuel_recip"):
        return None
    from simulation.materials import fuel_recip_from_hp
    pr = getattr(sim, "physics_runner", None)
    fuel_ref = float(pr.fire.params.fuel_ref) if pr is not None else 60.0
    r = fuel_recip_from_hp(fuel_ref)
    tbl.fuel_recip[:] = r          # pins on_tile_changed + any cache rebuild
    sim.gmap.fuel_recip[:] = r     # pins the already-built grid
    return (fuel_ref, int(r))


def main(outdir: str, pinned: bool = True) -> int:
    from cool_shift_axis_gate_a_capture import _SCENARIOS
    from o2_full_reference_gate_a_capture import _sim_burning_fuel
    from field_ab_harness import SIM_FIELDS, capture_trajectory
    from field_digest import trajectory_digest

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}  outdir={out}  "
          f"pinned={pinned}")

    ok = True
    scenarios = tuple(_SCENARIOS) + (("burning_fuel_22x22", _sim_burning_fuel),)
    for name, make_sim in scenarios:

        def _built(_make=make_sim):
            sim = _make()
            if pinned:
                _pin_fuel_recip_to_global(sim)
            return sim

        # Diagnostic probe, built WITHOUT the pin whatever the mode — otherwise
        # the "off-reference fuel tiles" count below would read the pinned grid
        # and report 0 for every scenario, i.e. it would silently stop being a
        # non-vacuousness measure.
        probe = make_sim()
        n_fire = int(np.count_nonzero(probe.gmap.fire))
        n_fuel = int(np.count_nonzero(probe.gmap.flammable))
        # How much of the map is fuel whose hp differs from the global — the
        # measure of whether this scenario can SEE the axis at all.
        n_offref = 0
        if hasattr(probe.gmap, "fuel_recip"):
            from simulation.materials import fuel_recip_from_hp
            ref = fuel_recip_from_hp(float(probe.physics_runner.fire.params.fuel_ref))
            n_offref = int(np.count_nonzero(
                probe.gmap.flammable & (probe.gmap.fuel_recip != ref)))
        pin = _pin_fuel_recip_to_global(_built()) if pinned else None
        if n_fire == 0:
            ok = False
            print(f"  {name}: no fire seeded — vacuous for the fuel law")
            continue

        traj = capture_trajectory(make_sim=_built, n_steps=N_TICKS)
        peak_fire = max(int(np.abs(s["fire"]).max()) for s in traj)
        dig = trajectory_digest(traj)

        payload = {}
        for t, snap in enumerate(traj):
            for f in SIM_FIELDS:
                if f in snap:
                    payload[f"t{t:03d}__{f}"] = snap[f]
        np.savez_compressed(out / f"{name}.npz", **payload)

        print(f"  GATE_A_DIGEST {name} = {dig}   "
              f"(pin={pin}, fire seeds {n_fire}, fuel tiles {n_fuel}, "
              f"off-reference fuel tiles {n_offref}, "
              f"peak fire {peak_fire} counts, arrays {len(payload)})")
        if peak_fire == 0:
            ok = False
            print(f"  {name}: fire never moved — vacuous")

    print("GATE_A_CAPTURE: " + ("OK" if ok else "VACUOUS"))
    return 0 if ok else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--unpinned"]
    sys.exit(main(args[0] if args else "_gate_a_out",
                  pinned=("--unpinned" not in sys.argv)))
