"""Aperture follow-up — correcting the first pass (2026-09-13).

The first bake-off reported "shadow edge width 0.00 tiles" for long
characteristics and for shear, and I wrote that up as a perfect shadow. Looking
at `scheme_profiles.png` shows that was the METRIC failing, not the schemes
winning: those profiles are not sharp-edged plateaus, they are PENCIL BEAMS —
two spikes where an ordinate happens to pass through the aperture, and nothing
in between. A 90%->10% edge-width measure crosses both thresholds inside one
tile of a spike and dutifully reports 0.00.

So the real question is not "how sharp is the edge" but "does the scheme
reproduce the lit region at all". This script measures that: normalised RMS
error against the analytic solution along the cut, plus a fill/spill split —

    fill  : mean fluence inside the true lit region, relative to exact.
            1.0 is correct, below 1 means the aperture is under-lit, and the
            pencil-beam failure shows up as a low fill with a huge peak.
    spill : mean fluence outside the true lit region, relative to the exact
            in-region mean. 0.0 is correct; diffusion shows up here.

and it does all of it WITH and WITHOUT per-tick quadrature rotation, because
the first pass only tested rotation on the isotropy scene and rotation is the
one lever that plausibly fixes an aperture ray effect.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/aperture_study.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheme_study import (N_ORD, TWO_PI, scene_shadow, solve_exact, solve_long,
                          solve_shear, solve_step)      # noqa: E402

OUT = Path(__file__).resolve().parent
PROBE = 30


def rotated(fn, kappa, source, nrot=N_ORD, **kw):
    """Average a solver over nrot quadrature phases spanning one ordinate gap."""
    acc = None
    for k in range(nrot):
        ph = k * (TWO_PI / (N_ORD * nrot))
        r = fn(kappa, source, phase=ph, **kw)
        f = r[0] if isinstance(r, tuple) else r
        acc = f.copy() if acc is None else acc + f
    return acc / nrot


def fill_spill(prof, ref, lit):
    """fill = mean inside the true lit band / exact's mean there.
       spill = mean outside it / exact's mean inside."""
    ref_in = ref[lit].mean()
    if ref_in <= 0:
        return float("nan"), float("nan")
    return prof[lit].mean() / ref_in, prof[~lit].mean() / ref_in


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k2, s2, o2, (ey, ex), wx = scene_shadow()
    g_exact = solve_exact(s2, o2)
    ref = g_exact[:, PROBE]
    # the geometric lit band at the probe column: the aperture is rows ey-3..ey+3
    # at x = wx, seen from the emitter at (ey, ex)
    ys = np.arange(k2.shape[0])
    half = 3.5 * (PROBE - ex) / (wx - ex)
    lit = np.abs(ys - ey) <= half

    variants = [
        ("step S16", solve_step(k2, s2)[0][:, PROBE]),
        ("step S16 + rot", rotated(solve_step, k2, s2)[:, PROBE]),
        ("shear S16", solve_shear(k2, s2)[0][:, PROBE]),
        ("shear S16 + rot", rotated(solve_shear, k2, s2)[:, PROBE]),
        ("long char S16", solve_long(k2, s2)[:, PROBE]),
        ("long char S16 + rot", rotated(solve_long, k2, s2)[:, PROBE]),
    ]

    print("=" * 78)
    print("APERTURE FIDELITY  (7-tile gap, cut 12 tiles downstream, S16)")
    print("=" * 78)
    print(f"  true lit band: rows {ey-half:.1f}..{ey+half:.1f} "
          f"({2*half:.1f} tiles wide)")
    print(f"\n  {'scheme':<22}{'fill':>8}{'spill':>8}{'RMS err':>10}")
    print(f"  {'(ideal)':<22}{1.0:>8.2f}{0.0:>8.2f}{0.0:>10.3f}")
    for name, prof in variants:
        f, s = fill_spill(prof, ref, lit)
        rms = math.sqrt(np.mean(((prof - ref) / max(ref.max(), 1e-12)) ** 2))
        print(f"  {name:<22}{f:>8.2f}{s:>8.2f}{rms:>10.3f}")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(ys - ey, ref / ref.max(), "k-", lw=2.4, label="exact", zorder=5)
    for name, prof in variants:
        ax.plot(ys - ey, prof / max(ref.max(), 1e-12), lw=1.4, label=name)
    ax.axvspan(-half, half, color="0.85", zorder=0, label="true lit band")
    ax.set_xlim(-16, 16)
    ax.set_xlabel("tiles from the gap centre")
    ax.set_ylabel("fluence / exact peak")
    ax.set_title("Aperture fidelity: 7-tile gap, cut 12 tiles downstream (S16)\n"
                 "does the scheme reproduce the LIT REGION, not just an edge?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "aperture_fidelity.png", dpi=130)
    print(f"\nwrote {OUT / 'aperture_fidelity.png'}")


if __name__ == "__main__":
    main()
