"""Gate (a) capture — WHOLE-ENGINE byte-identity for the O2 full-response split.

The O2 availability law used to normalize by AMBIENT:

    o2f = clamp01((X - o2_frac_ext) / (o2_frac_amb - o2_frac_ext))

so ambient air always produced o2f == 1 and the clamp made ambient the ceiling.
The split introduces a separate **full-response reference** ``o2_frac_full``
(default 1.0, pure O2, NOT map-overridden) as the denominator's upper end:

    o2f = clamp01((X - o2_frac_ext) / (o2_frac_full - o2_frac_ext))

``o2_frac_amb`` keeps its value (0.21), its per-map override and its meaning.

The claim THIS script gates is the plumbing claim, not the behaviour claim:

    with ``o2_frac_full`` pinned back to ``o2_frac_amb``, the whole engine must
    be byte-identical to the pre-patch build on every scenario, tolerance ZERO.

That is the proof the new dial is threaded correctly and that nothing else moved
(the behaviour change at the NEW default is gate (b), and it is EXPECTED).

This is a BEFORE/AFTER artifact, not a self-checking gate (deliberately no
``test_`` prefix — pytest does not collect it, and pinning digests here would
manufacture a red the moment Erik's joint fire re-tune moves a dial). It is
VERSION-AGNOSTIC on purpose: the pin is a ``hasattr`` guard, so the identical
file runs against a pristine PRE-patch tree (where the pin is a no-op, because
the law already normalizes by ambient) and against the patched tree.

Scenarios are imported verbatim from ``cool_shift_axis_gate_a_capture`` — the
same five the cool_shift axis gated on (default 16x16 space room, planetside
ambient sky, space breach, furniture burn, sealed room), all fire-seeded, so the
O2 law is live in every one.

Usage (from a repo root that has cpp/build/Release built):

    python tests/o2_full_reference_gate_a_capture.py <outdir>

writes ``<outdir>/<scenario>.npz`` holding every captured field for every tick
(so the comparison is genuinely per-tick per-CELL, not just a hash), and prints
one canonical trajectory digest per scenario. Run it in the pristine tree and in
the patched tree with different outdirs, then diff the printed digests and
compare the npz files cell by cell.
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


def _pin_full_to_ambient(sim):
    """Back-compat pin: ``o2_frac_full := o2_frac_amb`` on BOTH O2 laws.

    On a pristine (pre-patch) tree neither solver has the attribute and this is
    a no-op — which is exactly right, because there the span IS the ambient
    span. On the patched tree it restores the old denominator, so the two trees
    must then agree bit-for-bit.

    The per-map refresh (``PhysicsRunner._ambient_args``) rewrites
    ``o2_frac_amb`` every tick from the level's authored ``[ambient] o2_frac``;
    ``o2_frac_full`` is NOT map-overridden, so we pin it to the level's ambient
    fraction where one exists, and to the config-bound value otherwise.
    """
    pr = getattr(sim, "physics_runner", None)
    if pr is None:
        return None
    amb = getattr(sim.gmap, "_ambient", None)
    x_amb = float(amb.o2_frac) if amb is not None else float(pr.fire.params.o2_frac_amb)
    pinned = False
    if hasattr(pr.fire.params, "o2_frac_full"):
        pr.fire.params.o2_frac_full = x_amb
        pinned = True
    if hasattr(pr.combustion, "o2_frac_full"):
        pr.combustion.o2_frac_full = x_amb
        pinned = True
    return (x_amb, pinned)


def _sim_burning_fuel() -> "object":
    """THE O2-LAW scenario: fire seeded directly ON flammable WOOD tiles, in a
    sealed hull room with an O2-ENRICHED pocket (X well above ambient).

    The five imported scenarios seed fire on AIR tiles and rely on radiative
    ignition to reach fuel, so two of them never actually step the O2 law inside
    30 ticks. This one drives both O2 laws from tick 0: the fire logistic (a lit
    flammable tile with open, gas-holding neighbours) AND the combustion draw
    (flammable claimants beside O2-bearing air), at a mole fraction ABOVE
    ambient — exactly the region the old normalize-by-ambient law clamped away.
    """
    from level_loader import LevelData
    from simulation import Simulation, atmosphere_fixed, fire_fixed
    from simulation.gases import O2
    from simulation.unit import Unit

    H = W = 22
    tm = np.full((H, W), 1, dtype=np.int32)            # hull shell
    tm[2:H - 2, 2:W - 2] = 0                           # interior air
    tm[8, 5:17] = 2                                    # wood partition (fuel)
    tm[12:16, 6:10] = 6                                # furniture crates (fuel)
    level = LevelData(name="o2_full_gate_a_fuel", version="2", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=20260730, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    q = atmosphere_fixed.quantize_scalar
    # Fire ON the fuel itself (flammable tiles) -> the logistic steps at once.
    for x in (7, 8, 9, 10, 11):
        g.fire[8, x] = fire_fixed.quantize_scalar(0.8)
        g.temperature[8, x] += q(900.0)                # hot = 1, well above T_ext
    g.fire[12, 7] = fire_fixed.quantize_scalar(0.6)    # a burning crate
    g.temperature[12, 7] += q(700.0)
    # O2 ENRICHMENT around the partition: X climbs well above ambient 0.21, the
    # band the pre-split law clamped to o2f == 1 and this patch opens up.
    g.gas[O2][6:11, 5:17] += q(1.2)
    sim.set_paused(False)
    sim.add_unit(Unit("M1", x=17, y=17, team=0))
    assert int(np.count_nonzero(g.fire & g.flammable)) > 0, \
        "fire must be seeded ON flammable tiles"
    return sim


def main(outdir: str) -> int:
    from cool_shift_axis_gate_a_capture import _SCENARIOS
    from field_ab_harness import SIM_FIELDS, capture_trajectory
    from field_digest import trajectory_digest

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}  outdir={out}")

    ok = True
    scenarios = tuple(_SCENARIOS) + (("burning_fuel_22x22", _sim_burning_fuel),)
    for name, make_sim in scenarios:

        def _pinned_sim(_make=make_sim):
            sim = _make()
            _pin_full_to_ambient(sim)
            return sim

        probe = _pinned_sim()
        pin = _pin_full_to_ambient(probe)
        n_fire = int(np.count_nonzero(probe.gmap.fire))
        if n_fire == 0:
            ok = False
            print(f"  {name}: no fire seeded — vacuous for the O2 law")
            continue

        traj = capture_trajectory(make_sim=_pinned_sim, n_steps=N_TICKS)
        peak_fire = max(int(np.abs(s["fire"]).max()) for s in traj)
        dig = trajectory_digest(traj)

        # Save EVERY field of EVERY tick so the A/B is per-cell, not per-hash.
        payload = {}
        for t, snap in enumerate(traj):
            for f in SIM_FIELDS:
                if f in snap:
                    payload[f"t{t:03d}__{f}"] = snap[f]
        np.savez_compressed(out / f"{name}.npz", **payload)

        print(f"  GATE_A_DIGEST {name} = {dig}   "
              f"(pin x_amb={pin[0]!r} applied={pin[1]}, fire seeds {n_fire}, "
              f"peak fire {peak_fire} counts, arrays {len(payload)})")
        if peak_fire == 0:
            ok = False
            print(f"  {name}: fire never moved — vacuous")

    print("GATE_A_CAPTURE: " + ("OK" if ok else "VACUOUS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "_gate_a_out"))
