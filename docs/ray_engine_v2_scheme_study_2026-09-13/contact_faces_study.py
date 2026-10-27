"""Do two touching solids radiate to each other, and does that replace conduction?

Erik: "two solids touching, so how can they radiate to each other? via the touching
side? ... if it's from the touching side - then we dont even need conduction anymore,
because conduction would be radiation at distance 0.  but u all ready told me that we
DO need conduction, so im a little confused here."

The confusion is my fault: I said radiation cannot move heat inside a solid, and that
is WRONG for a sweep.  It can, and this measures exactly how.

WHAT THE SWEEP ACTUALLY DOES AT A CONTACT FACE.  An opaque cell absorbs everything
arriving on its upwind faces and emits a*E_deg[T] of its own.  Put two opaque cells
side by side and A's emission lands entirely in B -- so yes, via the touching side, and
it is strictly NEAREST-NEIGHBOUR, because B is opaque and nothing gets past it.  Chain
that along a wall and heat does propagate through it, one cell per tick, which is a
diffusion process.  So Erik's instinct is right about the MECHANISM.

WHY IT STILL IS NOT CONDUCTION.  The two laws have different shapes:

    radiative contact exchange  ~  E_deg[T_A] - E_deg[T_B]   ~  T_A^4 - T_B^4
    conduction face flux        ~  (T_A - T_B) >> face_shift ~  T_A - T_B

A fourth-power difference and a linear difference cross over somewhere, and where they
cross depends on the material's conductivity.  That is what this measures: for each
material, the ratio of the two channels at several temperature gaps.  If radiation
dominates, adding it to a material that already conducts is a large double count; if
conduction dominates, the radiative term is a rounding error.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/contact_faces_study.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sweep_ref import sweep_shear                     # noqa: E402

RAD_SCALE = 5.1427e-5
K_AMB = 293.0
ONE = 65536.0
N = 16

# (name, heat_atten, conductivity, thermal_mass shift `his`, self face_shift)
# face_shift = round(-log2(harmonic_mean_kappa / KAPPA_REF)), KAPPA_REF = 50,
# as materials.py._build_conduction_tables bakes it.
MATERIALS = [
    ("hull",      1.0, 50.0, 5, 2),
    ("steel",     1.0, 45.0, 5, 2),
    ("glass",     0.3,  1.0, 4, 6),
    ("wood",      1.0,  0.15, 3, 8),
    ("furniture", 0.5,  0.0,  3, None),   # NO_FACE: no conduction at all
]


def E_deg(T):
    return RAD_SCALE * (K_AMB + max(T, 0.0)) ** 4


def radiative_face(a, his, T_hot, T_cold, n=41):
    """Measured, not derived: an all-opaque slab with a hot face, swept.

    Returns the per-tick temperature change of the first cold cell, in game units,
    and the profile going into the slab.
    """
    A = np.full((n, n), a)
    T = np.full((n, n), float(T_cold))
    T[:, :3] = T_hot                      # a hot three-cell face on the left
    E = np.vectorize(E_deg)(T)
    dE, _fl, _so, _si, _ce = sweep_shear(A, E, kleak=0.0, E_amb=E_deg(0.0), n=n_ord())
    mid = n // 2
    prof = dE[mid, :] / (2.0 ** his) / ONE
    return prof


def n_ord():
    return N


def conduction_face(shift, T_hot, T_cold):
    """The engine's conduction face flux, per tick, in game units: dT >> face_shift."""
    if shift is None:
        return 0.0
    return (T_hot - T_cold) / (2.0 ** shift)


def multi_tick_wall(depth=10, ticks=400, his=3):
    """Run the sweep AND the temperature fold for many ticks on an all-opaque wall,
    hot face held at 1263 game, and see how far in the heat actually gets."""
    n = 21
    A = np.ones((n, n))
    T = np.zeros((n, n))
    hist = []
    for t in range(ticks):
        T[:, :2] = 1263.0                      # the hot face is held
        E = RAD_SCALE * (K_AMB + np.maximum(T, 0.0)) ** 4
        dE = sweep_shear(A, E, kleak=0.0, E_amb=E_deg(0.0), n=N)[0]
        T = np.maximum(T + dE / (2.0 ** his) / ONE, 0.0)
        if t + 1 in (24, 120, 240, 400):
            hist.append((t + 1, T[n // 2, 2:2 + depth].copy()))
    print("")
    print("   Held hot face, temperature (game) at each depth into the wall,")
    print("   radiation ONLY, no conduction:")
    print("        " + "".join(f"{'d='+str(d):>9}" for d in range(depth)))
    for (t, row) in hist:
        print(f"     {t/24:5.1f}s " + "".join(f"{v:9.0f}" for v in row))


def main():
    print("1. DOES HEAT PROPAGATE THROUGH A SOLID BY RADIATION ALONE?")
    print("   An all-opaque slab, three hot cells on the left at 1263 game, rest at")
    print("   ambient.  Per-tick temperature change going into the slab (wood, his=3):\n")
    prof = radiative_face(1.0, 3, 1263.0, 0.0)
    for x in range(2, 12):
        print(f"     cell {x-2:2d} deep:  {prof[x]:+12.2f} game/tick")
    print("")
    print("   -> in ONE tick it is strictly NEAREST-NEIGHBOUR: an opaque cell absorbs")
    print("      everything, so nothing reaches cell 2 at all.  But cell 1 is hotter")
    print("      next tick, so over TIME it chains -- a diffusion process with a")
    print("      T^4-difference law.  So it DOES move heat through a wall, slowly:")
    multi_tick_wall()

    print("\n2. RADIATIVE CONTACT EXCHANGE vs THE CONDUCTION FACE, same temperature gap")
    print("   (both in game units per tick, into the colder cell)\n")
    gaps = [(300.0, 280.0), (500.0, 300.0), (900.0, 300.0), (1263.0, 0.0), (3000.0, 300.0)]
    for (name, a, kappa, his, shift) in MATERIALS:
        print(f"   {name}  (conductivity {kappa}, his {his}, "
              f"face_shift {shift if shift is not None else 'NO_FACE'})")
        print(f"     {'T_hot':>7}{'T_cold':>8}{'radiation':>13}{'conduction':>13}"
              f"{'ratio':>10}")
        for (th, tc) in gaps:
            prof = radiative_face(a, his, th, tc)
            rad = prof[3]                       # the first cell past the hot face
            con = conduction_face(shift, th, tc)
            ratio = ("  n/a" if con == 0 else f"{rad/con:9.1f}x")
            print(f"     {th:7.0f}{tc:8.0f}{rad:13.2f}{con:13.2f}{ratio:>10}")
        print()

    print("   READ IT AS: ratio > 1 means radiation would dominate the contact face and")
    print("   adding it on top of conduction is a large double count.  ratio < 1 means")
    print("   it is a rounding error on a channel conduction already owns.")


if __name__ == "__main__":
    main()
