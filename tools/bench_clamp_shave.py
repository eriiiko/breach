"""THE CLAMP'S SHAVE -- how much radiative gain the Pass-1 maximum-principle clamp
withholds, per medium, on a burning scene (ray-engine-v2 P5c, issue #12; design
v3 §2.8 / §8.4).

The clamp  T_new = min(T_after, max(T_before, E°⁻¹(Φ)))  bounds the radiative
sub-step by the black body in equilibrium with the fluence a cell absorbs. E°⁻¹
returns a bucket's LOW edge (so it is idempotent), while the explicit fixed point
of a cell approached from below lies INSIDE its bucket -- so such a cell is
clipped on every tick it sits at its cap (p5a_gas_stiffness_study.py §5). Since
P5c the engine counts what that costs: `TemperatureSolver.e_rad_clamp_drop_sum`,
the withheld step priced at each cell's real capacity. This bench splits it by
MEDIUM and sets it against the radiation each medium absorbs, which is the
number Erik's decision reads: is e_inv_q's ceiling worth making exact?

HOW. The engine's own counter mixes both media, so the bench REPLAYS the fold's
radiative sub-step, per cell, and must reproduce that counter's per-tick delta
EXACTLY (asserted every tick -- the replay is then the counter, split). It wraps
PhysicsEngine.step_tail (the proxy idiom of tests/_sealedbox_bisect_bench.py) to
snapshot the fold's inputs at entry -- `temperature`, `gas_energy`, the bulk
count -- and after the tick reads the sweep's own planes, which the sweep leaves
in place until its next run (design row 38): `rad_net_sweep`, `rad_fluence`, and
the effective extinction the CPU sweep read (`RadiationSweep.a_eff_plane`). The
arithmetic is the integer reference's (sweep_ref_q: shr_round0_signed, the gas
chain gas_rad_dT_q, e_inv_q, gas_mirror_q, cap_real_q) -- never a second copy.

Per medium it reports, per second of sim time (24 ticks) and in watts
(J_per_count, config.toml [physics.radiation]):
  absorbed   -- the gross radiation the medium took from the stream, sum of
                (Phi * a) >> 16 over its cells (heat counts)
  net gain   -- sum of max(rad_net, 0): what radiation tried to add to heating cells
  drop       -- what the clamp withheld (e_rad_clamp_drop_sum's share)
  hits       -- clamped cell-ticks
  held share -- the part of the drop taken from cells whose OWN bucket already
                lies above their cap's, clamped at T_before (a net absorber above
                its equilibrium: the Fleck damping or the per-ordinate emission
                floor). An exact e_inv_q ceiling would NOT release that part; the
                rest is the bucket-edge pinning at the cap itself, which it would.
and, for GAS, P5b's finding checked on the live table: cell-ticks where a HOT gas
cell -- in a bucket ABOVE its own cap E°⁻¹(Φ)'s -- was nevertheless a net ABSORBER
(rad_net > 0): its Fleck-damped emission fell below what it absorbs, and the clamp
then holds it at T_before, i.e. keeps it from cooling that tick. The same test
UNDAMPED (f = 1) is counted apart as the per-ordinate emission floor, and a net
absorber INSIDE its cap's own bucket as the bucket-edge pinning (E°[bucket] <= Phi
there by construction) -- neither is the damping.

SCENES
  bench  -- the canonical fire bench (tools/fire_timing_harness.build_level at its
            own defaults: an 84 x 40 open-field planetside arena, ONE kindling
            tile seeded at its ignition_temp, the sky exchange live), 90 s.
  room   -- a sealed hull room (18 x 12 interior) with a 3 x 3 furniture block
            ignited at its ignition_temp: smoke accumulates, 60 s.
  playground -- the shipped playground level with a wood tile at the 1263-game
            plateau and lit (the CUDA gate's live scene), 30 s.

Run:
    C:/Users/steen/anaconda3/python.exe tools/bench_clamp_shave.py [scene ...] [--seconds S]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "tests",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "docs" / "ray_engine_v2_scheme_study_2026-09-13"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import sweep_ref_q as R  # noqa: E402
from config import CFG  # noqa: E402

TPS = 24


def _j_per_count():
    """One heat count in joules, the currency pin (config.toml documents it as
    rho_c * V_tile / (8 * 65536)); read from the material table's own derivation
    so the bench cannot hold a second copy of the number."""
    from simulation.materials import RHO_C_PIN, THERMAL_MASS_PIN
    v_tile = float(CFG.physics.thermal.tile_size_ref_m) ** 2 * float(CFG.physics.water.ceiling_h)
    return RHO_C_PIN * v_tile / (THERMAL_MASS_PIN * 65536.0)


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
def scene_bench():
    from fire_timing_harness import build_level, KIND
    from simulation import Simulation, fire_fixed
    iw, ih = 84, 40
    cx, cy = 12, 1 + ih // 2
    sim = Simulation(build_level(iw, ih, (cx, cy), 0.333), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    g.fire[cy, cx] = fire_fixed.quantize_scalar(float(CFG.physics.fire.ignition_seed))
    g.temperature[cy, cx] = fire_fixed.quantize_scalar(float(g.materials.ignition_temp[KIND]))
    return sim, 90.0


def scene_room():
    from level_loader import LevelData
    from simulation import Simulation, fire_fixed
    from fire_timing_harness import AIR, HULL, FURN
    iw, ih = 18, 12
    h, w = ih + 2, iw + 2
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = HULL
    tm[:, 0] = tm[:, -1] = HULL
    y0, x0 = h // 2 - 1, 4
    tm[y0:y0 + 3, x0:x0 + 3] = FURN
    lvl = LevelData(name="p5c_clamp_room", version="2", path=Path("."), tilemap=tm,
                    tile_size_m=0.333, diffuse_path=Path("."), boundary="space")
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    blk = (slice(y0, y0 + 3), slice(x0, x0 + 3))
    g.fire[blk] = fire_fixed.quantize_scalar(float(CFG.physics.fire.ignition_seed))
    g.temperature[blk] = fire_fixed.quantize_scalar(float(g.materials.ignition_temp[FURN]))
    return sim, 60.0


def scene_playground():
    from level_loader import load as load_level
    from simulation import Simulation, fire_fixed
    from simulation.materials import MAT_WOOD
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    for y, x in zip(ys, xs):
        if 0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1:
            if not g.thermal_solid[y, x + 1] or not g.thermal_solid[y + 1, x]:
                g.temperature[y, x] = 1263 << 16
                g.fire[y, x] = fire_fixed.quantize_scalar(0.8)
                break
    return sim, 30.0


SCENES = {"bench": scene_bench, "room": scene_room, "playground": scene_playground}


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------
class ShaveMeter:
    """Wraps the engine's step_tail; after every tick, replays the fold's
    radiative sub-step per cell and accumulates the per-medium books."""

    def __init__(self, sim):
        self.sim = sim
        self.g = sim.gmap
        self.runner = sim.physics_runner
        self.engine = self.runner.engine
        self.ts = self.engine.temperature
        self.pre = None
        eng = self.engine
        meter = self

        class _Proxy:
            def __getattr__(self, name):
                return getattr(eng, name)

            def step_tail(self, *args, **kwargs):
                g = meter.g
                meter.pre = dict(T=g.temperature.copy(), E=g.gas_energy.copy(),
                                 nb=sum(g.gas[gi].astype(np.int64) for gi in
                                        np.flatnonzero(g.gases.conservative)),
                                 drop=int(eng.temperature.e_rad_clamp_drop_sum),
                                 hits=int(eng.temperature.rad_clamp_hits))
                return eng.step_tail(*args, **kwargs)

        self.runner.engine = _Proxy()
        n_floor_q, c_v_q, _rcv = self.engine.gas_capacity_q()
        self.c_v_q, self.n_floor_q = int(c_v_q), int(n_floor_q)
        self.t_amb_q = int(self.runner._eos_t_amb_raw())
        self.table = [int(v) for v in np.asarray(self.engine.emissive.table())]
        self.tbl_np = np.asarray(self.table, dtype=np.int64)
        self.acc = {m: dict(absorbed=0, gain=0, drop=0, drop_held=0, hits=0)
                    for m in ("solid", "gas")}
        self.held_hot = 0
        self.held_hot_max_T = None
        self.held_floor = 0
        self.held_floor_max_T = None
        self.held_edge = 0
        self.ticks = 0

    def close(self):
        self.runner.engine = self.engine

    def _e_inv(self, phi):
        """e_inv_q vectorized: the largest bucket b with E°[b] <= Phi, as its low
        edge in Q16.16; 0 below E°[0]. Held to R.e_inv_q on a sample every tick."""
        b = np.searchsorted(self.tbl_np, phi, side="right") - 1
        return np.where(b < 0, 0, (4 * np.maximum(b, 0)) << 16)

    def after_tick(self):
        g, pre = self.g, self.pre
        rn = g.rad_net_sweep.astype(np.int64)
        phi = g.rad_fluence.astype(np.int64)
        a_eff = np.asarray(self.engine.radiation.a_eff_plane()).astype(np.int64)
        fl = np.asarray(self.engine.radiation.fleck_plane())
        ts = g.thermal_solid
        acct = g._gas_energy_accountable()
        T0 = pre["T"].astype(np.int64)
        drop_total = 0
        # ---- thermal solids (vectorized; the kit's own int64 arithmetic) ----
        sm = ts & (rn != 0)
        his = g.heat_inv_shift.astype(np.int64)
        if np.any(sm):
            r, s, t0, f = rn[sm], his[sm], T0[sm], phi[sm]
            pos = s >= 0
            mag = np.abs(r)
            d = np.where(pos, mag >> np.where(pos, s, 0), mag << np.where(pos, 0, -s))
            dTr = np.where(r < 0, -d, d)
            t_after = np.clip(t0 + dTr, -(1 << 31), (1 << 31) - 1)
            t_cap = self._e_inv(f)
            ceiling = np.maximum(t_cap, t0)
            cut = np.maximum(t_after - ceiling, 0)
            s_cap = np.clip(s, R.CAP_SHIFT_MIN, 30)
            cap = np.left_shift(np.int64(1), s_cap + 16)
            e_cut = (cut * cap).astype(object)
            drop = int(e_cut.sum())
            # WHERE it bound: at the cell's own cap (T_before at or below the
            # cap's bucket -- the bucket-edge pinning an exact ceiling would
            # move) or HELD at T_before (its bucket above the cap's)
            def bucket(t):                      # R.e_bucket_of, vectorized
                return np.clip(np.where(t <= 0, 0, t >> R.E_INDEX_SHIFT),
                               0, R.E_TABLE_SIZE - 1)
            held = bucket(t0) > bucket(t_cap)
            self.acc["solid"]["drop_held"] += int(e_cut[held].sum())
            self.acc["solid"]["drop"] += drop
            self.acc["solid"]["hits"] += int(np.count_nonzero(cut > 0))
            drop_total += drop
        self.acc["solid"]["absorbed"] += int(((phi[ts] * a_eff[ts]) >> 16).sum())
        self.acc["solid"]["gain"] += int(np.maximum(rn[ts], 0).sum())
        # ---- accountable gas cells (Python ints: the chain needs 128 bits) --
        gm = acct & (rn != 0)
        for (y, x) in zip(*np.nonzero(gm)):
            r = int(rn[y, x])
            n_raw = int(pre["nb"][y, x])
            nb = max(0, n_raw)
            dT = R.gas_rad_dT_q(r, n_raw, c_v_q=self.c_v_q, n_floor_q=self.n_floor_q)
            t_before = R.gas_mirror_q(int(pre["E"][y, x]), nb, self.t_amb_q)
            t_after = R.sat_add_q16(t_before, dT)
            t_cap = R.e_inv_q(int(phi[y, x]), self.table)
            ceiling = max(t_cap, t_before)
            if t_after > ceiling:
                drop = (t_after - ceiling) * R.cap_real_q(False, 0, nb, self.c_v_q)
                self.acc["gas"]["drop"] += drop
                self.acc["gas"]["hits"] += 1
                if R.e_bucket_of(t_before) > R.e_bucket_of(t_cap):
                    self.acc["gas"]["drop_held"] += drop
                drop_total += drop
            if r > 0 and t_before > t_cap:
                # a net absorber the clamp holds at T_before. THREE causes, told
                # apart: inside its cap's own bucket it is the bucket-edge
                # pinning (E°[bucket] <= Phi by construction); in a HIGHER bucket
                # with a damped Fleck factor it is P5b's finding -- a hot cell
                # whose emission the damping cut below what it absorbs; in a
                # higher bucket UNDAMPED it is the sweep's per-ordinate floor on
                # the emission (src_m = amb_m + ex_m, each (x * w_m) >> 16), which
                # on the live table's small counts (E°[0] = 125) under-reads a
                # near-ambient cell's emission against its Phi
                tg = t_before / 65536.0
                if R.e_bucket_of(t_before) > R.e_bucket_of(t_cap):
                    if int(fl[y, x]) < R.F_ONE:
                        self.held_hot += 1
                        self.held_hot_max_T = tg if self.held_hot_max_T is None else max(
                            self.held_hot_max_T, tg)
                    else:
                        self.held_floor += 1
                        self.held_floor_max_T = tg if self.held_floor_max_T is None \
                            else max(self.held_floor_max_T, tg)
                else:
                    self.held_edge += 1
        self.acc["gas"]["absorbed"] += int(((phi[acct] * a_eff[acct]) >> 16).sum())
        self.acc["gas"]["gain"] += int(np.maximum(rn[acct], 0).sum())
        # the replay IS the engine's counter, split -- or the bench is wrong
        eng_drop = int(self.ts.e_rad_clamp_drop_sum) - pre["drop"]
        eng_hits = int(self.ts.rad_clamp_hits) - pre["hits"]
        rep_hits = self.acc["solid"]["hits"] + self.acc["gas"]["hits"]
        assert drop_total == eng_drop, (self.ticks, drop_total, eng_drop)
        self._hits_check = getattr(self, "_hits_check", 0) + eng_hits
        assert rep_hits == self._hits_check, (self.ticks, rep_hits, self._hits_check)
        # spot-check the vectorized E°⁻¹ against the reference's own
        k = int(phi.flat[(self.ticks * 7919) % phi.size])
        assert int(self._e_inv(np.asarray([k]))[0]) == R.e_inv_q(k, self.table)
        self.ticks += 1


def run(scene_name, seconds=None):
    sim, default_s = SCENES[scene_name]()
    seconds = default_s if seconds is None else seconds
    meter = ShaveMeter(sim)
    from simulation.gases import SMOKE
    n = int(round(seconds * TPS))
    smoke_peak = 0.0
    try:
        for _ in range(n):
            sim.set_paused(False)
            sim.step()
            meter.after_tick()
            smoke_peak = max(smoke_peak, float(sim.gmap.gas[SMOKE].max()) / 65536.0)
    finally:
        meter.close()
    j = _j_per_count()
    print(f"\n=== {scene_name}: {seconds:.0f} s ({n} ticks), smoke peak density "
          f"{smoke_peak:.3f}; the replay reproduced e_rad_clamp_drop_sum and "
          f"rad_clamp_hits exactly on every tick ===")
    print(f"{'medium':>7} {'absorbed W':>12} {'net gain W':>12} {'drop W':>10} "
          f"{'drop/absorbed':>14} {'drop/net gain':>14} {'hits':>7} {'held share':>11}")
    out = {}
    for m in ("solid", "gas"):
        a = meter.acc[m]
        # heat counts (Q16) per tick -> W; the drop is Q32 (cap * dT), so >> 16
        absorbed_w = a["absorbed"] * j * TPS / n
        gain_w = a["gain"] * j * TPS / n
        drop_w = (a["drop"] / 65536.0) * j * TPS / n
        pa = 100.0 * drop_w / absorbed_w if absorbed_w else float("nan")
        pg = 100.0 * drop_w / gain_w if gain_w else float("nan")
        held = 100.0 * a["drop_held"] / a["drop"] if a["drop"] else float("nan")
        print(f"{m:>7} {absorbed_w:12.1f} {gain_w:12.1f} {drop_w:10.3f} "
              f"{pa:13.3f}% {pg:13.3f}% {a['hits']:7d} {held:10.1f}%")
        out[m] = dict(absorbed_w=absorbed_w, gain_w=gain_w, drop_w=drop_w,
                      pct_absorbed=pa, pct_gain=pg, hits=a["hits"], pct_held=held)
    print("  'held share': the part of the drop taken from cells whose OWN bucket is "
          "above their cap's (clamped at T_before: an exact e_inv_q ceiling would not "
          "release it); the rest is the bucket-edge pinning at the cap itself")
    held = (f"{meter.held_hot} cell-ticks, hottest {meter.held_hot_max_T:.1f} game"
            if meter.held_hot else "none")
    floor = (f"{meter.held_floor} cell-ticks, hottest {meter.held_floor_max_T:.1f} game"
             if meter.held_floor else "none")
    print(f"  gas cells the clamp held from cooling (a net absorber ABOVE its own cap's "
          f"bucket): DAMPED, P5b's finding: {held}; UNDAMPED, the per-ordinate emission "
          f"floor: {floor}; inside its cap's bucket (the bucket-edge pinning): "
          f"{meter.held_edge} cell-ticks")
    out["held_hot"] = meter.held_hot
    out["held_floor"] = meter.held_floor
    out["held_edge"] = meter.held_edge
    return out


def main(argv):
    seconds = None
    if "--seconds" in argv:
        i = argv.index("--seconds")
        seconds = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    names = [a for a in argv[1:] if not a.startswith("--")] or list(SCENES)
    for name in names:
        run(name, seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
