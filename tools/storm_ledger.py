"""storm_ledger.py — per-pass momentum/energy ledger for the storming audit
(2026-08-14). ANALYSIS ONLY: harness-level accounting over the shipped engine.

Breach's synced state is Q16.16 integer, so budgets can be accounted EXACTLY:
this tool snapshots the mirrored planes around each solver pass inside one
tick and attributes the change in total momentum, kinetic energy, and thermal
energy to the pass that made it. Nothing in cpp/ or src/ changes — the passes
are seamed at the Python call sites in PhysicsRunner.step():

    fire_cast   PhysicsRunner.cast_fire_heat      (radiation -> heat/rad_net)
    water       PhysicsRunner._step_water         (dormant here)
    eos         PhysicsEngine.run_substeps        (SL advect + solve + kick +
                                                   wave_absorb + compression)
    combustion  PhysicsRunner._run_combustion     (O2 burn + gas heat deposit)
    sky         PhysicsRunner._run_sky_exchange   (dormant on space maps)
    tail        PhysicsEngine.step_tail           (fire feedback + T pass)
    other       everything else in Simulation.step (ignition, damage, ...)

The C++ engine calls are seamed by swapping ``runner.engine`` for a forwarding
proxy — a harness-side wrap, not an engine change.

Ledger quantities per pass per tick (int64-exact sums over open gas cells):
    sum_ux, sum_uy      net velocity vector (raw Q16.16 counts)
    mom_x, mom_y        N-weighted momentum sum(N_bulk * u)         [dequant]
    ke                  0.5 * sum(N_bulk * |u|^2)                   [dequant]
    ke_probe            sum(|u|^2) — the storm_probe metric          [dequant]
    eth_gas             sum(N_bulk * T_abs), T_abs = T + 290, c_v=1 [dequant]
    t_obj               sum(T) over thermal_solid tiles             [dequant]
    n_bulk, n_o2        gas inventory (dequant)
    t_min_gas           min gas T (floor telemetry)

Combustion-amplifier telemetry (the suspected 1/N gain, combustion.cpp:798-803
``dT = deposit / (max(O2+N2, n_floor)*c_v)``): for every cell the combustion
pass heated, the divisor N is read back EXACTLY from the post-pass planes
(Pass C updates O2/N2 before it divides, and nothing later in the call touches
them), so ``amp = N_amb_bulk / max(N, n_floor)`` is measured, not modeled.

Usage:
  conda run -n data python tools/storm_ledger.py --ticks 4800 --damp 0.005 \
      --pf1b --set k_wind_strip=0.5 --out ledger_unstable.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                      # noqa: E402
from config import CFG                           # noqa: E402
from simulation import Simulation                # noqa: E402
from simulation import fire_fixed                # noqa: E402

import storm_probe as sp                         # noqa: E402
import bench_two_room as bench                   # noqa: E402
from fire_timing_harness import (                # noqa: E402
    FP_ONE, apply_overrides, restore_overrides,
)

PASSES = ("fire_cast", "water", "eos", "combustion", "sky", "tail", "other")

# Ambient bulk N (Q16.16 == FP_ONE): the o2+n2 split sums to 1.0 by design
# (ambient pin C*N_amb*T_amb == 1.0). Verified against the level's own planes
# at t=0 in run_ledger().
N_AMB_BULK = 1.0


class _EngineProxy:
    """Forwarding proxy over the C++ PhysicsEngine that lets the ledger
    snapshot around run_substeps / step_tail. Harness-side only."""

    def __init__(self, engine, before, after):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_before", before)
        object.__setattr__(self, "_after", after)

    def __getattr__(self, name):
        real = getattr(self._engine, name)
        if name == "run_substeps":
            def wrapped(*a, **kw):
                self._before("eos")
                r = real(*a, **kw)
                self._after("eos")
                return r
            return wrapped
        if name == "step_tail":
            def wrapped(*a, **kw):
                self._before("tail")
                r = real(*a, **kw)
                self._after("tail")
                return r
            return wrapped
        return real


def measure_state(gmap, masks):
    """Exact conserved-quantity sums from the mirrored int32 planes.

    Velocity/N sums run over ALL flow-open cells (the crate is flow-open at
    permeability 0.5 and carries gas + wind), so donor-cell conservation and
    the momentum budget close exactly on a sealed map. T-based sums are split:
    gas cells vs thermal_solid objects (the crate's T is an OBJECT temperature
    owned by the TemperatureSolver, not gas state).
    """
    flow = masks.flow_open
    gas_open, obj = masks.gas_open, masks.obj
    ux = gmap.wind_x[flow].astype(np.int64)
    uy = gmap.wind_y[flow].astype(np.int64)
    o2 = gmap.gas[masks.o2_idx][flow].astype(np.int64)
    n2 = gmap.gas[masks.n2_idx][flow].astype(np.int64)
    nb = o2 + n2
    t = gmap.temperature[gas_open].astype(np.int64)
    o2g = gmap.gas[masks.o2_idx][gas_open].astype(np.int64)
    nbg = o2g + gmap.gas[masks.n2_idx][gas_open].astype(np.int64)
    f = FP_ONE
    uxf, uyf, nbf, tf = ux / f, uy / f, nb / f, t / f
    t_abs = tf + 290.0
    return {
        "sum_ux": int(ux.sum()), "sum_uy": int(uy.sum()),
        "mom_x": float((nbf * uxf).sum()), "mom_y": float((nbf * uyf).sum()),
        "mom_abs": float((nbf * np.hypot(uxf, uyf)).sum()),
        "ke": float(0.5 * (nbf * (uxf * uxf + uyf * uyf)).sum()),
        "ke_probe": float((uxf * uxf + uyf * uyf).sum()),
        "eth_gas": float(((nbg / f) * t_abs).sum()),
        "t_obj": float(gmap.temperature[obj].astype(np.int64).sum() / f),
        "t_solid": float(gmap.temperature[gmap.solid].astype(np.int64).sum() / f),
        "n_bulk": float(nbf.sum()), "n_o2": float(o2.sum() / f),
        "n_smoke": float(gmap.smoke[flow].astype(np.int64).sum() / f),
        "p_sum": float(gmap.atmosphere[flow].astype(np.int64).sum() / f),
        "t_min_gas": float(t.min() / f) if t.size else 0.0,
        "t_max_gas": float(t.max() / f) if t.size else 0.0,
        "umax": float(np.hypot(uxf, uyf).max()) if ux.size else 0.0,
    }


class _Masks:
    def __init__(self, gmap):
        open_mask = ~gmap.solid
        for attr in ("is_vacuum", "is_ambient"):
            m = getattr(gmap, attr, None)
            if m is not None:
                open_mask &= ~m
        ts = getattr(gmap, "thermal_solid", None)
        obj = (ts.astype(bool) & open_mask) if ts is not None \
            else np.zeros_like(open_mask)
        self.flow_open = open_mask          # crate included (flow-open)
        self.gas_open = open_mask & ~obj    # pure-gas cells (T is gas T here)
        self.obj = obj
        self.o2_idx = int(gmap.gases.name_to_id["o2"])
        self.n2_idx = int(gmap.gases.name_to_id["inert_n2"])


# P-E0 (energy-books §2.5): these two are PER-TICK deltas (reset at every
# EOSSolver.step entry), unlike the cumulative hit counters — the series
# code below must read them raw, never as a diff of consecutive reads.
# P-E1 (energy-books §2.1.5/§2.5) adds the transport law's one-way guard terms
# and the active-flux telemetry §7's truncation bound is scaled by. All five
# are PER-TICK too (same reset-at-step()-entry idiom), so they belong in this
# tuple: the run totals below accumulate them instead of diffing them.
PER_TICK_COUNTERS = ("eos.eth_transport_delta", "eos.eth_compression_delta",
                     "eos.e_ts_residual", "eos.e_wipe_sum", "eos.e_floor_sum",
                     "eos.n_active_flux", "eos.n_bulk_active_sum")


def counters(runner):
    out = {}
    # (holder, prefix, names). P-E2a added the third holder: the temperature
    # solver's own energy books — conduction's two counted residuals plus the
    # three SIGNED boundary channels (design §2.3, round-1 finding L3-6). These
    # are CUMULATIVE (the `t_max_phys_hits` idiom of that class), so they are
    # deliberately NOT in PER_TICK_COUNTERS: the series code diffs them.
    for holder, prefix, names in (
            (runner.eos, "eos.",
             ("u_clamp_hits", "u_max_hits", "work_clamp_hits",
              "energy_floor_hits", "t_max_phys_hits",
              "eth_transport_delta", "eth_compression_delta",
              # P-E1: rule (d) destruction, the N_EPS wipe, the
              # T_MIN creator, and the active-flux pair.
              "e_ts_residual", "e_wipe_sum", "e_floor_sum",
              "n_active_flux", "n_bulk_active_sum")),
            (runner.combustion, "comb.", ("heat_floor_hits",)),
            (runner.temperature, "temp.",
             # P-E2a: conduction's endpoint-truncation and capacity-floor
             # residuals + the limiter's engagement count; then Pass 3 /
             # sky, the breach wipe and the ambient-ring pin — all three
             # SIGNED, all three named creators as well as sinks.
             ("e_cond_trunc_sum", "e_cond_cap_sum", "cond_limit_hits",
              "e_cool_sum", "e_vac_wipe_sum", "e_ring_pin_sum",
              "t_max_phys_hits", "t_low_rail_hits")),
    ):
        for nm in names:
            try:
                out[prefix + nm] = int(getattr(holder, nm))
            except Exception:
                pass
    return out


def run_ledger(ticks=4800, damp=0.0, dials=None, keep_series=True):
    overrides = dict(dials or {})
    restore = apply_overrides(overrides) if overrides else []
    try:
        level = bench.load_fixture()
        sim = Simulation(level, seed=12345, breach_physics=bp,
                         enable_recorder=False)
        gmap = sim.gmap
        if damp > 0.0:
            sp.stamp_air_damping(gmap, damp)
        masks = _Masks(gmap)

        # Sanity: the ambient bulk N really is 1.0 per cell.
        nb0 = (gmap.gas[masks.o2_idx].astype(np.int64)
               + gmap.gas[masks.n2_idx].astype(np.int64))
        amb_pin = float(np.median(nb0[masks.gas_open]) / FP_ONE)
        assert abs(amb_pin - N_AMB_BULK) < 0.01, f"ambient N_bulk = {amb_pin}"

        fx, fy = bench.CRATE_XY
        seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
        gmap.fire[fy, fx] = fire_fixed.quantize_scalar(seed_i)
        gmap.temperature[fy, fx] = fire_fixed.quantize_scalar(280.0)

        runner = sim.physics_runner
        n_floor = float(getattr(CFG.physics.thermal, "n_floor_heat", 0.05))

        # ---- seam the passes -------------------------------------------
        state = {"pre": None, "tick_deltas": None, "comb_pre_T": None}
        keys = None

        def snap():
            return measure_state(gmap, masks)

        def before(pass_name):
            state["pre"] = snap()
            if pass_name == "combustion":
                state["comb_pre_T"] = gmap.temperature.copy()

        amp_series = []

        def after(pass_name):
            post = snap()
            pre = state["pre"]
            d = {k: post[k] - pre[k] for k in pre
                 if k not in ("t_min_gas", "t_max_gas", "umax")}
            state["tick_deltas"][pass_name] = d
            if pass_name == "combustion":
                # Amplifier telemetry — exact divisor read-back.
                dT = (gmap.temperature.astype(np.int64)
                      - state["comb_pre_T"].astype(np.int64))
                heated = (dT > 0) & masks.gas_open
                if heated.any():
                    o2 = gmap.gas[masks.o2_idx][heated].astype(np.int64) / FP_ONE
                    n2 = gmap.gas[masks.n2_idx][heated].astype(np.int64) / FP_ONE
                    div = np.maximum(o2 + n2, n_floor)
                    amp = N_AMB_BULK / div
                    dTe = dT[heated] / FP_ONE
                    amp_series.append((state["tick"], float(amp.max()),
                                       float((amp * dTe).sum() / dTe.sum()),
                                       float(dTe.sum()), int(heated.sum())))
                state["comb_pre_T"] = None

        runner.engine = _EngineProxy(runner.engine, before, after)
        for nm, pass_name in (("cast_fire_heat", "fire_cast"),
                              ("_step_water", "water"),
                              ("_run_combustion", "combustion"),
                              ("_run_sky_exchange", "sky")):
            real = getattr(runner, nm)

            def mk(real, pass_name):
                def wrapped(*a, **kw):
                    before(pass_name)
                    r = real(*a, **kw)
                    after(pass_name)
                    return r
                return wrapped
            setattr(runner, nm, mk(real, pass_name))

        # ---- run ---------------------------------------------------------
        per_pass_totals = {p: None for p in PASSES}
        series = {"tick": [], "ke": [], "ke_probe": [], "eth_gas": [],
                  "t_obj": [], "n_o2": [], "n_bulk": [], "n_smoke": [],
                  "p_sum": [], "t_min_gas": [], "umax": [],
                  "mom_abs": [], "fire_I": [],
                  # P-E0 pocket telemetry (design §2.3): the flat index of the
                  # gas-T argmin cell and the bulk N sitting there — the
                  # window-pocket N the trust-band decision reads.
                  "t_min_cell": [], "n_at_tmin": []}
        eth_totals = {c: 0 for c in PER_TICK_COUNTERS}
        pass_series = {p: {"ke": [], "eth_gas": [], "mom_abs": [],
                           "sum_ux": [], "sum_uy": []} for p in PASSES}
        counter_series = []
        prev_counters = counters(runner)

        tick_state_prev = snap()
        for k in range(1, ticks + 1):
            state["tick_deltas"] = {}
            state["tick"] = k
            sim.set_paused(False)
            sim.step()
            tick_state = snap()

            # "other" = whole-tick delta minus the seamed passes.
            dk = {q: tick_state[q] - tick_state_prev[q]
                  for q in tick_state
                  if q not in ("t_min_gas", "t_max_gas", "umax")}
            seamed = state["tick_deltas"]
            other = dict(dk)
            for p, d in seamed.items():
                for q in d:
                    other[q] -= d[q]
            seamed["other"] = other
            tick_state_prev = tick_state

            if keys is None:
                keys = sorted(other.keys())
            for p in PASSES:
                d = seamed.get(p)
                if d is None:
                    d = {q: 0 for q in keys}
                if per_pass_totals[p] is None:
                    per_pass_totals[p] = {q: 0.0 for q in keys}
                for q in keys:
                    per_pass_totals[p][q] += d[q]
                if keep_series:
                    for q in pass_series[p]:
                        pass_series[p][q].append(d.get(q, 0))

            cnow = counters(runner)
            for c in PER_TICK_COUNTERS:      # per-tick values: accumulate
                if c in cnow:
                    eth_totals[c] += cnow[c]
            if keep_series:
                series["tick"].append(k)
                for q in ("ke", "ke_probe", "eth_gas", "t_obj", "n_o2",
                          "n_bulk", "n_smoke", "p_sum",
                          "t_min_gas", "umax", "mom_abs"):
                    series[q].append(tick_state[q])
                series["fire_I"].append(int(gmap.fire[fy, fx]) / FP_ONE)
                # P-E0 pocket telemetry: argmin-T gas cell + its bulk N.
                t_masked = np.where(masks.gas_open, gmap.temperature,
                                    np.int32(np.iinfo(np.int32).max))
                j = int(np.argmin(t_masked))
                nb_j = (int(gmap.gas[masks.o2_idx].flat[j])
                        + int(gmap.gas[masks.n2_idx].flat[j]))
                series["t_min_cell"].append(j)
                series["n_at_tmin"].append(nb_j / FP_ONE)
                counter_series.append(
                    {c: (cnow[c] if c in PER_TICK_COUNTERS
                         else cnow[c] - prev_counters.get(c, 0))
                     for c in cnow})
                prev_counters = cnow

        return {
            "per_pass_totals": per_pass_totals,
            "series": series, "pass_series": pass_series,
            "counter_series": counter_series,
            "amp_series": amp_series,
            "final_counters": counters(runner),
            "eth_totals": eth_totals,   # P-E0: run totals of the per-tick deltas
            "n_open": int(masks.gas_open.sum()),
        }
    finally:
        restore_overrides(restore)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=4800)
    ap.add_argument("--damp", type=float, default=0.0)
    ap.add_argument("--pf1b", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--out", default=None, help="npz path for the series")
    a = ap.parse_args(argv)

    dials = dict(sp.PF1B) if a.pf1b else {}
    for item in a.set:
        k, _, v = item.partition("=")
        dials[k] = v

    out = run_ledger(ticks=a.ticks, damp=a.damp, dials=dials)

    print(f"storm_ledger ticks={a.ticks} damp={a.damp} "
          f"overrides={sorted(dials)} open_cells={out['n_open']}")
    print(f"\n== per-pass totals over the run (dequantized units) ==")
    qs = ("ke", "ke_probe", "mom_abs", "sum_ux", "sum_uy", "eth_gas",
          "t_obj", "t_solid", "n_bulk", "n_o2", "n_smoke", "p_sum")
    print(f"  {'pass':10s} " + "  ".join(f"{q:>12s}" for q in qs))
    for p in PASSES:
        t = out["per_pass_totals"][p]
        if t is None:
            continue
        print(f"  {p:10s} " + "  ".join(f"{t.get(q, 0):>12.4g}" for q in qs))
    print(f"\n  final counters: {out['final_counters']}")
    print(f"  eth bracket totals (raw Q16.16^2): {out['eth_totals']}")
    amp = out["amp_series"]
    if amp:
        mx = max(r[1] for r in amp)
        print(f"  amplifier: max gain {mx:.1f}x  "
              f"(n_ticks with combustion heating: {len(amp)})")
    if a.out:
        np.savez_compressed(
            a.out,
            **{f"s_{q}": np.asarray(v) for q, v in out["series"].items()},
            **{f"p_{p}_{q}": np.asarray(v)
               for p in PASSES for q, v in out["pass_series"][p].items()},
            amp=np.asarray(out["amp_series"], dtype=np.float64),
            counters=json.dumps(out["counter_series"]),
            per_pass_totals=json.dumps(out["per_pass_totals"]),
            final_counters=json.dumps(out["final_counters"]),
            eth_totals=json.dumps(out["eth_totals"]))
        print(f"  series -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
