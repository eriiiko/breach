"""Smoke tau/alpha saturation diagnostic — issue #6's histogram half.

Measures, on REAL recorder dumps (debug_manual_*.npz / debug_blowup_*.npz),
how the shipped gas-medium opacity pipeline maps smoke density to screen
alpha (renderer/gas_medium.py, design docs/fire_b2_smoke_honesty_design_2026-07-21.md):

    tau   = smoke_absorb_scale * plume_k_scale * k_soot * rho
    tau'  = tau_curve_a * tau**tau_curve_b
    alpha = 1 - exp(-tau')

and answers #6's SPLIT question with numbers: how much of the smoky area is
flat black (alpha above the saturation threshold), where the visible ramp
sits in density space, and whether the sim's 0..1 density scale or the render
constants own the saturation.

The tau math is IMPORTED from renderer.gas_medium (gas_optical_depth) so this
instrument measures the shipped pipeline, never a reimplementation; the dials
are read live from config.toml through CFG. Dumps carry the SMOKE plane only,
so the tau here is the soot term of the full trace-gas sum — the dominant
term for fire smoke (steam absorbs ~9x less; poison/teargas only if deployed).

Usage:
    conda run -n data python tools/smoke_tau_histogram.py <dump.npz> [more ...]
        [--alpha-sat 0.98] [--alpha-vis 0.05] [--out PNG] [--stride 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from renderer.gas_medium import gas_optical_depth
from config import CFG

# Fixed log-spaced accumulation bins: exact enough for percentiles, and memory
# stays flat no matter how many 1200-snapshot dumps are pooled.
TAU_BINS = np.logspace(-4, 4, 161)
RHO_BINS = np.logspace(-7, 0.5, 151)
ALPHA_BINS = np.linspace(0.0, 1.0, 101)


def read_dials():
    """The live opacity dials, getattr-guarded to the design defaults."""
    smoke = getattr(CFG, "smoke", None)
    gm = getattr(getattr(CFG, "render", None), "gas_medium", None)
    soot = getattr(getattr(CFG, "gases", None), "smoke", None)
    base = float(getattr(smoke, "smoke_absorb_scale", 1.4))
    k_scale = float(getattr(gm, "plume_k_scale", 1.0))
    curve_a = float(getattr(gm, "tau_curve_a", 1.0))
    curve_b = float(getattr(gm, "tau_curve_b", 1.0))
    k_soot = float(np.mean(getattr(soot, "absorption", [0.88, 0.90, 0.93])))
    return base, k_scale, curve_a, curve_b, k_soot


def alpha_from_tau(tau, curve_a, curve_b):
    return 1.0 - np.exp(-curve_a * np.power(tau, curve_b))


def rho_at_alpha(alpha, base, k_scale, curve_a, curve_b, k_soot):
    """Invert the pipeline: the soot density that renders at a given alpha."""
    tau_p = -np.log(1.0 - alpha)
    tau = (tau_p / curve_a) ** (1.0 / curve_b)
    return tau / (base * k_scale * k_soot)


def percentile_from_hist(counts, edges, q):
    """Approximate percentile (q in [0,100]) from accumulated bin counts."""
    total = counts.sum()
    if total == 0:
        return float("nan")
    target = total * (q / 100.0)
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, target))
    i = min(i, len(counts) - 1)
    return float(np.sqrt(edges[i] * edges[i + 1]))  # geometric bin centre


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dumps", nargs="+", help="recorder .npz dump(s)")
    ap.add_argument("--alpha-sat", type=float, default=0.98,
                    help="alpha above this = 'flat black' (default 0.98)")
    ap.add_argument("--alpha-vis", type=float, default=0.05,
                    help="alpha above this = 'visible smoke' (default 0.05)")
    ap.add_argument("--stride", type=int, default=1,
                    help="analyze every Nth snapshot")
    ap.add_argument("--out", type=str, default=None,
                    help="output PNG path (default: tools/results/...)")
    args = ap.parse_args()

    base, k_scale, curve_a, curve_b, k_soot = read_dials()
    coef = base * k_scale * k_soot
    k_s = np.array([k_soot], dtype=np.float32)

    print(f"dials: smoke_absorb_scale={base}  plume_k_scale={k_scale}  "
          f"tau_curve=(a={curve_a}, b={curve_b})  k_soot={k_soot:.4f}")
    print(f"=> tau = {coef:.1f} * rho   (soot term)")
    for a in (args.alpha_vis, 0.5, 0.9, args.alpha_sat):
        r = rho_at_alpha(a, base, k_scale, curve_a, curve_b, k_soot)
        print(f"   alpha {a:4.2f}  at rho = {r:.3e}")
    rho_vis = rho_at_alpha(args.alpha_vis, base, k_scale, curve_a, curve_b, k_soot)
    rho_sat = rho_at_alpha(args.alpha_sat, base, k_scale, curve_a, curve_b, k_soot)
    print(f"   visible ramp spans rho in [{rho_vis:.2e}, {rho_sat:.2e}] — "
          f"{100.0 * (rho_sat - rho_vis):.2f}% of the 0..1 density scale")

    tau_h = np.zeros(len(TAU_BINS) - 1, dtype=np.int64)
    rho_h = np.zeros(len(RHO_BINS) - 1, dtype=np.int64)
    alpha_h = np.zeros(len(ALPHA_BINS) - 1, dtype=np.int64)
    series = []      # (label, per-snapshot % saturated of visible)
    pooled_vis = 0
    pooled_sat = {0.5: 0, 0.9: 0, 0.95: 0, args.alpha_sat: 0}
    rho_ge_half = 0
    rho_ge_99 = 0
    rho_max_all = 0.0

    for path in args.dumps:
        d = np.load(path)
        if "smoke" not in d:
            print(f"\n{path}: NO 'smoke' plane (fields: {sorted(d.keys())}) — skipped")
            continue
        smoke = d["smoke"]  # (S, H, W) float32, REAL density (recorder dequantized)
        s_n, h, w = smoke.shape
        label = Path(path).stem.replace("debug_manual_", "").replace("debug_blowup_", "blowup_")
        sat_pct = np.full(s_n, np.nan)
        for t in range(0, s_n, args.stride):
            rho = np.asarray(smoke[t], dtype=np.float64)
            tau = gas_optical_depth(rho[None], k_s, base_absorb_scale=base,
                                    plume_k_scale=k_scale)
            alpha = alpha_from_tau(tau, curve_a, curve_b)
            vis = alpha > args.alpha_vis
            n_vis = int(vis.sum())
            if n_vis == 0:
                sat_pct[t] = 0.0
                continue
            a_v, t_v, r_v = alpha[vis], tau[vis], rho[vis]
            tau_h += np.histogram(t_v, bins=TAU_BINS)[0]
            rho_h += np.histogram(r_v, bins=RHO_BINS)[0]
            alpha_h += np.histogram(a_v, bins=ALPHA_BINS)[0]
            pooled_vis += n_vis
            for thr in pooled_sat:
                pooled_sat[thr] += int((a_v > thr).sum())
            rho_ge_half += int((r_v >= 0.5).sum())
            rho_ge_99 += int((r_v >= 0.99).sum())
            rho_max_all = max(rho_max_all, float(r_v.max()))
            sat_pct[t] = 100.0 * (a_v > args.alpha_sat).sum() / n_vis
        series.append((label, sat_pct))
        worst = np.nanmax(sat_pct) if np.isfinite(sat_pct).any() else 0.0
        print(f"\n{path}\n  {s_n} snapshots ({h}x{w}); worst snapshot: "
              f"{worst:.1f}% of visible smoke flat-black")

    if pooled_vis == 0:
        print("\nNo visible smoke in any dump — nothing to histogram.")
        return

    print(f"\n{'=' * 66}\nPOOLED over {pooled_vis:,} visible-smoke tile-samples "
          f"(alpha > {args.alpha_vis}):")
    for thr in sorted(pooled_sat):
        print(f"  alpha > {thr:4.2f}: {100.0 * pooled_sat[thr] / pooled_vis:5.1f}%")
    for q in (50, 90, 99):
        tq = percentile_from_hist(tau_h, TAU_BINS, q)
        rq = percentile_from_hist(rho_h, RHO_BINS, q)
        print(f"  p{q:02d}  tau = {tq:9.3g}   rho = {rq:9.3g}")
    print(f"  rho >= 0.5: {100.0 * rho_ge_half / pooled_vis:.3f}%   "
          f"rho >= 0.99: {100.0 * rho_ge_99 / pooled_vis:.3f}%   "
          f"max rho seen: {rho_max_all:.4f}")

    # ---- figure: the money plots ------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    centres = np.sqrt(TAU_BINS[:-1] * TAU_BINS[1:])
    ax.stairs(tau_h, TAU_BINS, fill=True, color="#4878a8", alpha=0.85)
    tau_sat = -np.log(1.0 - args.alpha_sat)
    ax.axvspan(tau_sat, TAU_BINS[-1], color="black", alpha=0.18,
               label=f"flat black (alpha>{args.alpha_sat})")
    ax.axvline(-np.log(0.5), color="#a85448", ls="--", lw=1, label="alpha=0.5")
    ax.set_xscale("log")
    ax.set_xlabel("tau (soot optical depth)")
    ax.set_ylabel("tile-samples")
    ax.set_title("tau over visible smoke")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.stairs(rho_h, RHO_BINS, fill=True, color="#6a8f5f", alpha=0.85)
    ax.axvspan(rho_vis, rho_sat, color="#e0c060", alpha=0.35,
               label="the entire visible ramp")
    ax.axvspan(rho_sat, RHO_BINS[-1], color="black", alpha=0.18,
               label="beyond black")
    ax.set_xscale("log")
    ax.set_xlabel("rho (soot density, sim units; scale caps at 1.0)")
    ax.set_title("density over visible smoke")
    ax.legend(fontsize=8)

    ax = axes[2]
    for label, sp in series:
        ax.plot(np.arange(len(sp)), sp, lw=1.0, label=label[:28])
    ax.set_xlabel("snapshot")
    ax.set_ylabel(f"% of visible smoke with alpha>{args.alpha_sat}")
    ax.set_title("saturation over time")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)

    fig.suptitle(f"#6 smoke saturation diagnostic — tau = {coef:.0f}*rho "
                 f"(dials: {base} x {k_scale} x {k_soot:.2f}), curve a={curve_a} b={curve_b}",
                 fontsize=10)
    fig.tight_layout()
    out = args.out or str(ROOT / "tools" / "results" / "smoke_tau_histogram.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
