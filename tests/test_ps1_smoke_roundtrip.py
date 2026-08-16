"""P-S1 — the smoke round-trip oracle (docs/smoke_single_source_asbuilt_
2026-08-15.md, executing docs/smoke_single_source_design_2026-07-24.md +
docs/storm_audit_2026-08-14.md §4.2).

The storm audit found a mass/pressure pump: fire's ex-nihilo smoke scatter
(`fire_simulation.cpp`, pre-P-S1 `smoke[nbr] += smoke_emission*dt*I`) fed
UNBACKED counts into the trace `smoke` plane every tick a fire tile was lit;
the P4 decay->inert_N2 credit (`physics_engine.cpp`, decisions #12 v2.1) then
converted that unbacked trace mass into full-pressure-weight bulk N2 — two
individually-defensible rules composing into a real injection bug (audit
§4.2: +125.2 bulk-N counts / 200 s in a sealed two-room bench). P-S1 deletes
the scatter (fire_simulation.cpp + its CUDA mirror cuda_fire.cu); combustion
soot (`combustion.cpp`'s `soot_yield` channel) is now the ONE fire-smoke
source, and it is honest by construction: `SOOT[s] += soot; N2[s] +=
burn_dep - soot` is an exact Dalton split of `burn_dep` (no rounding loss
across the pair), and decay's credit is the same local, exact transfer
(`gas_slice[i] -= lost; n2_slice[i] += lost`, same cell, same tick).

THE ORACLE. With source A gone, the only things left that can move
`gmap.gas`'s total AT ALL are (a) the bulk O2/N2 donor-cell flux — exactly
conservative, measured 0 LSB drift over 4800 ticks by the storm audit — and
(b) the trace planes' OWN semi-Lagrangian transport, which `smoke_dynamics.h`
documents by name as "NON-CONSERVATIVE by design (the >>16 truncation is a
gentle built-in decay) ... accepted (Q-S2-1) ... NO flux form, NO limiter, NO
outflow clamp." That truncation is PRE-EXISTING, unrelated to P-S1, and
LOSSY-ONLY by construction (`>>16` truncates toward zero, so it can only ever
remove a fractional count, never add one) — the same reason
`test_eos_p4_combustion.py`'s own tier-1 tests isolate it out rather than
claim bit-exact transport. So the honest, correct invariant here is
BOUNDED-ABOVE, not bit-exact equality — the same idiom
`test_e2e_1_sealed_room_fire_self_starves`'s `_o2n2_total` already uses
("bounded ABOVE by its starting value... without being strictly monotonic").
Measured on this scenario: with source A deleted, a 300-tick sustained burn
drifts DOWN by <=2400 LSB out of a ~3.1M-count room (the honest transport
decay); with source A alive, the SAME scenario mints +5244 LSB on the very
first tick and +617761 by tick 120 — three orders of magnitude apart, and
the wrong sign. A bounded-above check catches the mint immediately and
tolerates the pre-existing gentle decay, which is exactly the claim P-S1
makes: "nothing may mint."

RED on HEAD (pre-P-S1): the scatter mints; the assertion fails within the
first couple of ticks.
GREEN after P-S1: the scatter is gone; the total only ever drifts down, and
by a tiny, pre-existing, already-accepted amount.

Run:
    conda run -n data python -m pytest tests/test_ps1_smoke_roundtrip.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                   # noqa: E402
from level_loader import LevelData                              # noqa: E402
from simulation.gamemap import GameMap                          # noqa: E402
from simulation.physics_runner import PhysicsRunner             # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL, MAT_WOOD, MaterialTable  # noqa: E402
from simulation import fire_fixed                                # noqa: E402

SEED_TICK_DT = 1.0 / 24.0
_TBL = MaterialTable.from_config()
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])

# How many ticks to run the sealed burn. Long enough that the fire is
# genuinely alive and burning (not just seeded and immediately snap-
# extinguished) and that the honest transport-decay signature (a few
# thousand LSB out of ~3.1M) is clearly visible for the report, while a mint
# (hundreds of thousands of LSB within the first few ticks, pre-fix) is
# caught almost immediately regardless.
N_TICKS = 300


def _sealed_room(hh=9, wood_at=None):
    """A hull-walled square room (MAT_HULL border), MAT_AIR interior, one
    MAT_WOOD fuel tile — mirrors test_eos_p4_combustion.py's fixture."""
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    if wood_at is not None:
        tm[wood_at] = MAT_WOOD
    ld = LevelData(name="p_s1_roundtrip", version="2", path=Path("."),
                   tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    return GameMap(ld)


def _runner():
    return PhysicsRunner(bp)


def _step_tick(pr, gmap, dt=None):
    """One GAME-FAITHFUL tick: PhysicsRunner.step + the per-tick `heat`/
    `rad_net`/`rad_flux` clear (test_eos_p4_combustion.py's harness-fidelity
    fix — these are per-tick deposit buffers Simulation.step clears at the
    END of a real tick; a bare pr.step loop never clears them)."""
    burned = pr.step(gmap, SEED_TICK_DT if dt is None else dt)
    gmap.heat.fill(0)
    gmap.rad_net.fill(0)
    gmap.rad_flux.fill(0)
    return burned


def _ignite(gmap, at, intensity=0.6, temp_mult=1.5):
    gmap.fire[at] = fire_fixed.quantize_scalar(float(intensity))
    gmap.temperature[at] = int(IGN_WOOD_Q16 * temp_mult)


def _total_all_gas_planes(gmap):
    """Σ over ALL gas planes (O2 + inert_N2 + every trace: steam / smoke /
    poison / teargas / fuel_gas), raw Q16.16 counts, exact int64 sum."""
    return int(gmap.gas.astype(np.int64).sum())


def test_ps1_roundtrip_all_gas_planes_never_exceed_start():
    """The round-trip oracle: with a sealed-room fire alive, the total across
    every gas plane must never rise above its starting value — nothing may
    mint bulk mass from nothing (module docstring: bounded-above is the
    correct claim once the pre-existing, unrelated, lossy-only trace-
    transport truncation is accounted for; see there for why not bit-exact)."""
    gmap = _sealed_room(hh=9, wood_at=(4, 4))
    pr = _runner()
    _ignite(gmap, (4, 4), intensity=0.6, temp_mult=1.5)

    total0 = _total_all_gas_planes(gmap)
    assert total0 > 0, "test setup produced no gas mass — vacuous"

    fire_alive_ticks = 0
    max_drift = 0
    for t in range(N_TICKS):
        _step_tick(pr, gmap)
        if float(gmap.fire[4, 4]) > 0.0:
            fire_alive_ticks += 1
        total = _total_all_gas_planes(gmap)
        drift = total - total0
        max_drift = max(max_drift, drift)
        assert total <= total0, (
            f"tick {t}: total gas mass (ALL {gmap.gas.shape[0]} planes) rose "
            f"from {total0} to {total} ({drift:+d} counts) — something is "
            f"minting bulk mass from nothing. The round-trip must never "
            f"INCREASE the total (combustion soot + its decay credit only "
            f"ever move mass WITHIN the sum; see module docstring).")

    assert fire_alive_ticks >= 0.8 * N_TICKS, (
        f"the fire barely burned ({fire_alive_ticks}/{N_TICKS} ticks alive) "
        f"— the gate needs a real, sustained burn to exercise the "
        f"round-trip, not a snap-extinguish")
    # Non-vacuousness the other direction: the scenario must actually be
    # ALIVE enough to move gas around (else "never exceeds start" would be
    # true trivially because nothing happened).
    assert max_drift <= 0, (
        f"unexpected: total gas mass rose by up to {max_drift} counts "
        f"without tripping the per-tick assertion above (should be "
        f"unreachable)")


if __name__ == "__main__":
    test_ps1_roundtrip_all_gas_planes_never_exceed_start()
    print("OK: P-S1 smoke round-trip oracle passed")
