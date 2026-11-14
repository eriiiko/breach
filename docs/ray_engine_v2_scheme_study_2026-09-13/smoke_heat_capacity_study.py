"""Does smoke's heat capacity decide how it behaves when it absorbs heat? (2026-09-14)

Erik: "the heating smoke would automatically radiate once it got hot - and it would be
pretty cool i think.  i wonder if it needs it heat capacitance to determine how fast it
drops in temp from that radiation tho... right?"

Right, and the tree makes it literal.  `cpp/src/gas_energy.h`: gas energy is the exact
product `N_raw * T_abs_raw`, and temperature is read back as `floordiv(E, N)`.  So a gas
cell's heat capacity IS its particle count.  Twice the gas, half the temperature rise for
the same absorbed energy.

THE PART THAT IS NOT OBVIOUS, and is what this measures.  Two different counts are
involved and they are not the same count:

    absorption  scales with the SOOT in the cell   (only soot absorbs)
    capacity    scales with ALL the gas in the cell (air included)

so the warming rate is governed by the soot FRACTION, not by how much smoke there is in
absolute terms.  A wisp of soot in thin air and a dense plume can behave very
differently even at the same opacity.

WHAT THIS MEANS FOR THE STABILITY QUESTION.  The claim under test is that heat capacity
changes WHERE smoke ends up, i.e. that thin smoke would run away.  It does not: with the
Fleck factor and the maximum-principle clamp from design v2 section 2.8, the equilibrium
is set by the radiation field alone -- the clamp bounds temperature by E_deg_inverse(Phi)
and knows nothing about heat capacity.  What capacity changes is HOW FAST the cell gets
there.  That is the useful answer: thin smoke flashes, thick smoke lags, both settle at
the same place.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/smoke_heat_capacity_study.py
"""
from __future__ import annotations
import math

RAD_SCALE = 5.1427e-5
K_AMB = 293.0
ONE = 65536.0

# A solid's heat capacity is 2**his in the fold; a gas cell's is its particle count.
# Expressed as "how many ambient-air-cells' worth of gas is in this cell", so that
# capacity = 1.0 is a normal air tile and 0.05 is a nearly-evacuated one.
SOLID_WOOD_HIS = 3


def E_deg(T):
    return RAD_SCALE * (K_AMB + max(T, 0.0)) ** 4


def E_inv(E):
    lo, hi = 0.0, 1e7
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if E_deg(mid) < E:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def step(T, phi, a, capacity, alpha_floor=0.5):
    """One tick of a gas cell absorbing `phi` and radiating, with Fleck + clamp.

    capacity is in units where 1.0 is an ambient air tile; it divides both the gain
    and the loss, exactly as N divides the gas energy to give temperature."""
    gain = a * phi / capacity / ONE
    raw_loss = a * E_deg(T) / capacity / ONE
    g = 4.0 * raw_loss / (max(T, 0.0) + K_AMB)
    alpha = max(alpha_floor, 1.0 - 1.0 / g) if g > 0 else alpha_floor
    f = 1.0 / (1.0 + alpha * g)
    return max(T + gain - f * raw_loss, 0.0), g


def run(phi, a, capacity, ticks=4000):
    cap_T = E_inv(phi)                       # the maximum-principle clamp
    T, gmax, t90 = 0.0, 0.0, None
    for k in range(ticks):
        T, g = step(T, phi, a, capacity)
        gmax = max(gmax, g)
        if T > cap_T:
            T = cap_T
        if t90 is None and T >= 0.9 * cap_T:
            t90 = (k + 1) / 24.0
    return T, cap_T, gmax, t90


SRC = 1263.0          # the measured crate plateau
PHI = 0.25 * E_deg(SRC)

print(__doc__.split("Run:")[0])
print("=" * 78)
print("1. EQUILIBRIUM vs RESPONSE TIME, for a smoke cell one tile from a 1263-game fire")
print("=" * 78)
print("   'capacity' 1.00 = a normal air tile, 0.05 = nearly evacuated.")
print("   'a' is the cell's absorptivity, which tracks its SOOT, not its total gas.\n")
print(f"{'capacity':>9}{'a (soot)':>10}{'T_eq':>10}{'clamp':>10}{'time to 90%':>13}"
      f"{'peak g':>10}")
for cap in (2.0, 1.0, 0.3, 0.05):
    for a in (0.6, 0.2):
        T, capT, gmax, t90 = run(PHI, a, cap)
        t = f"{t90:.2f} s" if t90 is not None else "  never"
        print(f"{cap:9.2f}{a:10.2f}{T:10.1f}{capT:10.1f}{t:>13}{gmax:10.1f}")

print()
print("   -> the equilibrium column is IDENTICAL down every row: it is set by the")
print("      radiation field, not by the gas.  Heat capacity only moves the clock.")
print("      Thin smoke flashes to temperature; thick smoke lags behind it.")

print()
print("=" * 78)
print("2. DOES LOW CAPACITY DESTABILISE ANYTHING?")
print("=" * 78)
print(f"{'capacity':>9}{'peak g = 4*loss/T_abs':>24}{'verdict without Fleck':>24}"
      f"{'with Fleck+clamp':>19}")
for cap in (2.0, 1.0, 0.3, 0.05, 0.01, 0.002):
    T, capT, gmax, _t = run(PHI, 0.6, cap)
    naive = "ok" if gmax < 1 else ("marginal" if gmax < 2 else "WOULD OVERSHOOT")
    ok = "bounded" if abs(T - capT) < 1.0 else "DRIFTED"
    print(f"{cap:9.3f}{gmax:24.1f}{naive:>24}{ok:>19}")
print()
print("   -> a nearly-evacuated cell with soot in it IS stiff: the group runs away as")
print("      capacity falls, which is exactly the regime a hull breach creates.  The")
print("      Fleck factor and the clamp hold it at the right answer regardless, which")
print("      is the whole reason design v2 section 5.3 says both must cover gas cells.")

print()
print("=" * 78)
print("3. HOW A SMOKE CELL COMPARES WITH A WOOD TILE")
print("=" * 78)
solid_cap = (2.0 ** SOLID_WOOD_HIS)
print(f"   wood tile   : capacity {solid_cap:.0f} in fold units")
print(f"   air tile    : capacity 1.0 -- {solid_cap:.0f}x less than wood")
print()
print("   So smoke responds to radiation about an order of magnitude faster than the")
print("   solid it came off.  Physically that is right -- soot in a flame reaches flame")
print("   temperature almost at once, which is why flames are yellow -- and visually it")
print("   is what you want: smoke that lights up and dims with the fire rather than")
print("   lagging a second behind it.")


# ---------------------------------------------------------------------------
# 4. CORRECTION (2026-09-15), prompted by Erik asking whether cutting absorption
#    on low-N gas cells would help the stiffness.
#
# Section 1 above varies `capacity` while holding `a` FIXED. That is the slice
# "the soot stays, the air leaves", and it is not what happens: a decompressing
# room loses every species together. Redo it physically -- a and capacity fall
# together -- and the density CANCELS:
#
#       g  ~  (n_soot / N_total) * E_deg(T) / T_abs  =  phi * E_deg / T_abs
#
# so the stiffness depends only on the soot FRACTION, not on how much gas is in
# the cell. Measured at the equilibrium a smoke cell reaches one tile from a
# 1263-game fire:
#
#       density 1.000, a 0.600  ->  g = 2.5
#       density 0.300, a 0.180  ->  g = 2.5
#       density 0.050, a 0.030  ->  g = 2.5
#       density 0.002, a 0.001  ->  g = 2.5
#
# Three consequences, and the first one is a correction to what section 2 claims:
#
#   1. A decompressing room does NOT get stiffer as it empties. Section 2's
#      "peak g 50.2 / 250.8 / 1254.2" rows are the unphysical slice and should
#      not be quoted as a hull-breach result.
#   2. Gas stiffness is BOUNDED and modest. Even at a soot fraction of 1.0 --
#      a cell of pure soot, which cannot happen -- g is 4.2, against 848 for a
#      solid at T_MAX_PHYS. Gas is the milder half, not the harder one.
#      It is still above 1, so Fleck + clamp are still required on gas; they are
#      just not holding back a runaway.
#   3. Erik's low-N cutoff is therefore NOT a stability lever -- it would remove
#      cells that were never the problem.
#
# But the suggestion lands somewhere better. The engine ALREADY has this floor:
# `gas_energy.h` sets `N_EPS_RAW = 1`, "the one bulk floor, shared verbatim with
# eos_solver.cpp's recovery and bulk_transport.cpp's divide policy (design 2.6:
# ONE value, every file)", below which the mirror reads ambient rather than
# dividing and the recovery wipes the cell. So a sub-N_EPS cell is DEFINED to be
# at ambient -- and a cell defined to be at ambient must not be emitting at some
# other temperature. Gas absorption and emission should ride that existing floor
# for consistency, not a new threshold of their own. Not an approximation: the
# removal of an inconsistency, using the canonical value.
# ---------------------------------------------------------------------------
