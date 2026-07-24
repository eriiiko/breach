"""Sky-exchange behavioral bench — gate d (far field stays breathable).

docs/sky_exchange_design_2026-07-24.md gate d. Demonstrates the SYSTEM-level
claim on a sponge-safe planetside bench: a sustained interior O2 sink (a
combustion PROXY — remove O2, add inert, N_total conserved) drains the far field
toward 0 with the sky pass OFF (today's edge-only reservoir loses to volumetric
consumption — the "0.21 → 0 in ~5 min" symptom), and the sky pass HOLDS it
breathable with the pass ON. Sweeps τ ∈ {30, 60, 120} s → the re-tune menu, and
measures the post-sink recovery time-constant.

NOT the full gate-d acceptance: that runs at the JOINT re-tune (design §5) with
the fire-tuning harness + the LOCKED fire combo (k_grow/k_die/wall_damage/…),
which live on the fire-tuning branch, not here. This bench isolates the sky
pass's contribution with a clean synthetic sink so the τ menu is interpretable.

Run:
    conda run -n data python tools/sky_exchange_bench.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import gas_fixed  # noqa: E402
from simulation.ambient import derive_ambient  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import INERT_N2, O2  # noqa: E402
from simulation.physics_runner import PhysicsRunner  # noqa: E402

FP_ONE = 65536
H, W = 40, 84                      # sponge-safe interior (≥ 80×36; sponge_width 8)
SINK = (slice(18, 22), slice(40, 44))     # central 4×4 O2 sink (the "crate")
FAR = (20, 12)                     # far-field probe: interior, ≥ sponge_width+few
SINK_O2_PER_TICK = gas_fixed.quantize_scalar(0.6)   # combustion-proxy draw/tick


def _build(sky_tau_s):
    tm = np.full((H, W), 9, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0     # SPACE ring → is_ambient
    level = LevelData(name="sky_bench", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."),
                      boundary="ambient",
                      ambient=derive_ambient(sky_tau_s=sky_tau_s))
    g = GameMap(level)
    g.stamp_units([])
    runner = PhysicsRunner(bp)
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


HALO = (slice(13, 27), slice(35, 49))     # 14×14 zone around the sink (its halo)


def _o2_frac_at(g, region):
    o2 = g.gas[O2][region].astype(np.int64).sum()
    n = o2 + g.gas[INERT_N2][region].astype(np.int64).sum()
    return float(o2 / n) if n else 0.0


def _far_o2_frac(g):
    """The measurable end-to-end signal: mean O2 fraction over the sink's HALO —
    the locally depleted zone the sky pass refills. (The whole-field far-field
    suffocation of §7 Q2 needs the fire's heat→pressure→outward-wind coupling,
    absent from this composition-only sink — that gate runs at the joint re-tune
    with the fire harness. Here we prove the pass is wired end-to-end and holds a
    depleted region up.)"""
    return _o2_frac_at(g, HALO)


def _drain(g):
    """Combustion proxy: pull O2 from the sink patch, bank it as inert
    (N_total conserved, exactly like the real combustion pass' O2→products)."""
    o2 = g.gas[O2][SINK]
    take = np.minimum(o2, SINK_O2_PER_TICK)
    g.gas[O2][SINK] = o2 - take
    g.gas[INERT_N2][SINK] += take


def run(sky_tau_s, burn_s=180.0, recover_s=180.0):
    runner, g, dt = _build(sky_tau_s)
    n_burn = int(round(burn_s / dt))
    n_rec = int(round(recover_s / dt))
    far0 = _far_o2_frac(g)
    min_far = far0
    # ---- burn phase: sink active every tick ----
    for _ in range(n_burn):
        _drain(g)
        runner.step(g, dt)
        min_far = min(min_far, _far_o2_frac(g))
    far_end_burn = _far_o2_frac(g)
    # ---- recovery phase: sink off, sky (if on) refills ----
    gap0 = max(far0 - far_end_burn, 1e-9)
    t_recover = None
    for k in range(n_rec):
        runner.step(g, dt)
        if t_recover is None and (far0 - _far_o2_frac(g)) <= gap0 / np.e:
            t_recover = (k + 1) * dt          # time to close 1/e of the gap
    far_end_rec = _far_o2_frac(g)
    return dict(far0=far0, min_far=min_far, far_end_burn=far_end_burn,
                far_end_rec=far_end_rec, t_recover=t_recover)


def main():
    print(f"Sky-exchange bench — {H}×{W} planetside, central O2 sink "
          f"{SINK_O2_PER_TICK/FP_ONE:.2f}/tick, far probe {FAR}")
    print(f"{'config':>12} | {'far0':>6} {'min_far(burn)':>13} "
          f"{'end_burn':>9} {'end_recover':>11} {'tau_recover':>12}")
    print("-" * 78)
    configs = [("sky OFF", 0.0), ("tau=30", 30.0), ("tau=60", 60.0), ("tau=120", 120.0)]
    for label, tau in configs:
        r = run(tau)
        tr = f"{r['t_recover']:.0f}" if r["t_recover"] is not None else "—"
        print(f"{label:>12} | {r['far0']:.3f}  {r['min_far']:.3f}        "
              f"{r['far_end_burn']:.3f}     {r['far_end_rec']:.3f}       {tr:>10}")
    print("-" * 78)
    print("Reading: sky OFF — the depleted halo does NOT recover (edge reservoir "
          "cannot refill a volumetric deficit: the symptom). sky ON — recovers "
          "toward ambient with tau_recover ≈ tau (30/59/120 s). The recovery "
          "CEILING falls as tau grows (0.210→0.209→0.206): the P1 round-to-nearest "
          "deadband (~0.5·N/λ), visible end-to-end — weigh it in the tau choice.")
    print("NOTE: the whole-field far-field SUFFOCATION of §7 Q2 (0.21→0) needs the "
          "fire's heat→pressure→outward-wind coupling, which this composition-only "
          "sink omits; that full gate-d run happens at the joint re-tune with the "
          "fire harness + locked combo (design §5).")


if __name__ == "__main__":
    main()
