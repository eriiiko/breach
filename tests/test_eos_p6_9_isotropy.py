"""EOS P6.9a — combustion isotropy (docs/eos_p6_9_combustion_design.md §6).

The whole point of the two-gather reformulation is that combustion is now
DIRECTION-FREE: contested oxygen is split proportionally/uniformly and the
ignition gate reads a pass-entry snapshot, so no scan direction is favoured.
This test proves it objectively (design §1: "validation by principle + objective
proof, not feel-check").

Scenario — a 4-fold-symmetric PLUS of flammable sources around one shared air
cell:

        . . o . .
        . . S . .
        o S C S o        C = central AIR cell, contested by the 4 sources
        . . S . .        S = flammable wood source (hot, fuelled), one per arm
        . . o . .        o = each source's PRIVATE outer air burn-site

By the 90-degrees rotational symmetry of the construction, an order-free
combustion pass MUST leave the four arms bit-identical: equal source wall_hp,
and equal O2/temperature/soot/N2 in the four outer cells. The OLD row-major
scatter does NOT: processing the sources N, W, E, S in index order, the first
sources drain the shared centre cell first-come-first-served (N and W would each
take a full burn_cap, E whatever is left, S nothing), so the four arms diverge
by tens of thousands of counts. This test would FAIL on the pre-P6.9 scatter and
PASS on the gather — that is the regression it guards.

Two variants (design §6 offers either; we do BOTH):
  * BIT-EXACT — the centre O2 is divisible by the claimant count (4), so the
    exact-integer split has ZERO remainder and every arm draws exactly O2/4 from
    the centre. Asserted bit-for-bit (the strongest possible isotropy claim).
  * BOUNDED-BIAS — the centre O2 is NOT divisible by 4, so the split leaves a
    remainder of R in [0,4) LSBs that the fixed tiebreak (largest key, ties ->
    lowest source index; here all keys tie -> the lowest-index faces) hands to R
    arms. Asserted within a <=3-LSB tolerance — the deliberate sub-LSB bias
    design §1 accepts, bounded by construction (R <= claimants-1 = 3).

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_eos_p6_9_isotropy.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                       # noqa: E402
from simulation.gases import O2, INERT_N2, SMOKE            # noqa: E402
from simulation.materials import MaterialTable, MAT_WOOD          # noqa: E402
from simulation import gas_fixed                                  # noqa: E402

FP_ONE = 65536
IGN_WOOD_Q16 = int(MaterialTable.from_config().ignition_temp_q16[MAT_WOOD])
# Arm order N, S, W, E — the four sources around the centre (matches D4).
ARMS = (("N", (-1, 0), (-2, 0)),
        ("S", (1, 0), (2, 0)),
        ("W", (0, -1), (0, -2)),
        ("E", (0, 1), (0, 2)))


def _plus_scene(center_o2_raw, outer_o2=0.21, hp=60.0, h=9, w=9):
    """A symmetric flammable PLUS. Returns the mutable field arrays + centre.

    center_o2_raw is set as a RAW Q16.16 count (so we control divisibility by
    the claimant count exactly). Every open cell carries ambient N2; only the
    centre and the four outer cells carry O2 (everything else is inert, so no
    other cell can burn and break the symmetry)."""
    gas = np.zeros((7, h, w), dtype=np.int32)
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    flammable = np.zeros((h, w), dtype=bool)
    wall_hp = np.zeros((h, w), dtype=np.int32)
    fire = np.zeros((h, w), dtype=np.int32)
    ign = np.zeros((h, w), dtype=np.int32)
    temperature = np.zeros((h, w), dtype=np.int32)

    cy, cx = h // 2, w // 2
    gas[INERT_N2][~solid] = gas_fixed.quantize_scalar(0.79)   # symmetric ambient

    # The four flammable sources (the arms of the plus), all identical + hot.
    for _name, (dy, dx), _outer in ARMS:
        sy, sx = cy + dy, cx + dx
        solid[sy, sx] = True
        flammable[sy, sx] = True
        wall_hp[sy, sx] = int(round(hp * FP_ONE))
        ign[sy, sx] = IGN_WOOD_Q16
        temperature[sy, sx] = IGN_WOOD_Q16 * 2       # well above ignition
        # each arm's private outer burn-site
        oy, ox = cy + _outer[0], cx + _outer[1]
        gas[O2][oy, ox] = gas_fixed.quantize_scalar(outer_o2)

    gas[O2][cy, cx] = int(center_o2_raw)             # the shared contested cell
    return gas, solid, is_vacuum, flammable, wall_hp, fire, ign, temperature, (cy, cx)


def _step(comb, scene, dt=0.25):
    gas, solid, is_vacuum, flammable, wall_hp, fire, ign, temperature, _ = scene
    comb.step(gas, O2, INERT_N2, SMOKE, temperature, wall_hp, fire,
              flammable, solid, is_vacuum, ign, dt, 1.0, 0.05)


def _arm_readouts(scene):
    """Per-arm (wall_hp, outer O2, outer temperature, outer soot, outer N2)."""
    gas, _, _, _, wall_hp, _, _, temperature, (cy, cx) = scene
    out = {}
    for name, (dy, dx), (oy_d, ox_d) in ARMS:
        sy, sx = cy + dy, cx + dx
        oy, ox = cy + oy_d, cx + ox_d
        out[name] = (int(wall_hp[sy, sx]), int(gas[O2][oy, ox]),
                     int(temperature[oy, ox]), int(gas[SMOKE][oy, ox]),
                     int(gas[INERT_N2][oy, ox]))
    return out


def _comb():
    c = bp.CombustionSolver()
    c.burn_rate = 1.0        # burn_cap = quantize(1.0*0.25) = 16384 counts
    c.o2_thresh_burn = 0.03
    c.H_fuel = 4.0
    c.soot_yield = 0.3
    c.fuel_per_o2 = 0.7
    return c


def test_isotropy_bit_exact_zero_remainder():
    """Centre O2 = 40000 counts is divisible by the 4 claimants (10000 each,
    zero remainder) and below D = 4*burn_cap = 65536 (so the cell is genuinely
    CONTESTED, exercising the split — not the trivial full-demand path). The
    four arms must come out BIT-IDENTICAL over a sustained burn."""
    comb = _comb()
    scene = _plus_scene(center_o2_raw=40000)
    gas = scene[0]
    cy, cx = scene[8]

    assert 40000 % 4 == 0, "fixture invariant: centre O2 divisible by claimants"
    burned_any = False
    for _ in range(10):
        gas[O2][cy, cx] = 40000                       # refill the contested cell
        for _name, _s, (oy_d, ox_d) in ARMS:          # refill each private site
            gas[O2][cy + oy_d, cx + ox_d] = gas_fixed.quantize_scalar(0.21)
        _step(comb, scene)
        r = _arm_readouts(scene)
        base = r["N"]
        for name in ("S", "W", "E"):
            assert r[name] == base, (
                f"isotropy broken: arm {name} {r[name]} != arm N {base} "
                "(a scan-direction bias survived the reformulation)")
        if base[0] < int(round(60.0 * FP_ONE)):
            burned_any = True
    assert burned_any, "vacuous: the plus never actually burned"


def test_isotropy_bounded_bias_nonzero_remainder():
    """Centre O2 = 40002 counts is NOT divisible by 4 (10000 each + a remainder
    of 2 LSBs). The fixed tiebreak hands those 2 LSBs to the two lowest-index
    arms, so the arms differ — but by AT MOST the remainder, R <= claimants-1 =
    3 LSBs PER TICK (design §1's accepted sub-LSB bias; the tiebreak is fixed,
    not rotated — Erik "not a purist"). Guards that the PER-TICK bias stays
    bounded and never blooms into a directional artifact.

    NOTE the bound is per-tick: the fixed tiebreak deposits its <=3-LSB bias
    into the SAME (lowest-index) arms every tick, so wall_hp would drift
    unboundedly over a long burn (the accepted, deterministic cost of not
    rotating). We reset the sources to a common wall_hp each tick so the
    measurement isolates one tick's remainder — exactly the design §6 quantity."""
    comb = _comb()
    scene = _plus_scene(center_o2_raw=40002)
    gas = scene[0]
    wall_hp = scene[4]
    cy, cx = scene[8]
    hp_uniform = int(round(60.0 * FP_ONE))

    max_spread = 0
    for _ in range(10):
        gas[O2][cy, cx] = 40002
        for _name, (dy, dx), (oy_d, ox_d) in ARMS:
            gas[O2][cy + oy_d, cx + ox_d] = gas_fixed.quantize_scalar(0.21)
            wall_hp[cy + dy, cx + dx] = hp_uniform   # re-level so we read 1 tick
        _step(comb, scene)
        r = _arm_readouts(scene)
        # Spread across the four arms, per readout channel.
        for chan in range(len(r["N"])):
            vals = [r[name][chan] for name in ("N", "S", "W", "E")]
            max_spread = max(max_spread, max(vals) - min(vals))
    assert max_spread <= 3, (
        f"the per-tick sub-LSB remainder bias exceeded 3 LSB (spread="
        f"{max_spread}) — the tiebreak leaked more than the bounded remainder")
    assert max_spread >= 1, (
        "vacuous: a non-divisible centre produced NO arm spread — the "
        "remainder path was not exercised")


if __name__ == "__main__":
    test_isotropy_bit_exact_zero_remainder()
    test_isotropy_bounded_bias_nonzero_remainder()
    print("OK: EOS P6.9 isotropy tests passed")
