"""THE CLAMP'S SHAVE -- how much radiative gain the Pass-1 maximum-principle clamp
withholds, per medium, on a burning scene (ray-engine-v2 P5c, issue #12; design
v3 §2.8 / §8.4) -- and, since P5d, where the clamp's headroom ceiling lets the
sweep's per-ordinate emission floor land.

The clamp  T_new = min(T_after, max(T_before, e_ceiling_q(Φ)))  bounds the
radiative sub-step by the TOP of the first E° bucket that out-emits the fluence
a cell absorbs (P5d, Erik's ruling of 2026-09-24; docs/ray_engine_v2_p5d_clamp_
headroom_brief_2026-09-24.md). Until P5d the ceiling was E°⁻¹(Φ), a bucket's
LOW edge, while an undamped cell balances by flickering across the NEXT edge --
so such a cell was clipped on every tick it sat at its cap: P5c measured that
bucket-edge pinning here (solids ~0.2 % of their gross absorption in the
playground) and it is what P5d removes. Since P5c the engine counts what the
clamp costs: `TemperatureSolver.e_rad_clamp_drop_sum`, the withheld step priced
at each cell's real capacity. This bench splits it by MEDIUM and sets it against
the radiation each medium absorbs.

HOW. The engine's own counter mixes both media, so the bench REPLAYS the fold's
radiative sub-step, per cell, and must reproduce that counter's per-tick delta
EXACTLY (asserted every tick -- the replay is then the counter, split). It wraps
PhysicsEngine.step_tail (the proxy idiom of tests/_sealedbox_bisect_bench.py) to
snapshot the fold's inputs at entry -- `temperature`, `gas_energy`, the bulk
count -- and after the tick reads the sweep's own planes, which the sweep leaves
in place until its next run (design row 38): `rad_net_sweep`, `rad_fluence`, the
effective extinction the CPU sweep read (`RadiationSweep.a_eff_plane`) and its
Fleck plane. The arithmetic is the integer reference's (sweep_ref_q:
shr_round0_signed, the gas chain gas_rad_dT_q, e_ceiling_q, e_inv_q,
gas_mirror_q, cap_real_q) -- never a second copy. The ceiling lookups go through
ShaveMeter._ceiling / _ceiling_1, the ONE place a comparison run may substitute
another ceiling (with an engine built to clamp there, or the replay assertion
fails).

Per medium it reports, per second of sim time (24 ticks) and in watts
(J_per_count, config.toml [physics.radiation]):
  absorbed   -- the gross radiation the medium took from the stream, sum of
                (Phi * a) >> 16 over its cells (heat counts)
  net gain   -- sum of max(rad_net, 0): what radiation tried to add to heating cells
  drop       -- what the clamp withheld (e_rad_clamp_drop_sum's share)
  hits       -- clamped cell-ticks
and the drop split three ways, by WHERE the clamp bound:
  shadow     -- cells whose Phi is below E°[0], where the ceiling is 0 on purpose
                (design row 31, UNCHANGED by P5d): a shadowed cell's rad_net is
                <= 0 except through rounding -- the per-ordinate floors -- and the
                clamp keeps that rounding from warming it
  at ceiling -- Phi >= E°[0], the cell at or below its ceiling: clamped AT it
  held       -- Phi >= E°[0], the cell already ABOVE its ceiling (a bucket past
                the first one that out-emits Phi), held at T_before: a net
                absorber above its equilibrium -- the Fleck damping or the
                per-ordinate emission floor.
Per medium, the net ABSORBERS (rad_net > 0, and large enough that the radiative
step raises T) with Phi >= E°[0] that sit IN OR ABOVE the ceiling's bucket --
where E°[bucket] > Phi, so the continuous balance says they should be cooling --
are split by their Fleck factor: DAMPED (f < 2^24; P5b's finding: the
damped emission fell below what they absorb) or UNDAMPED (f == 2^24: the sweep's
per-ordinate floor on emission, src_m = amb_m + ex_m with each (x * w_m) >> 16,
which on the live table's small counts, E°[0] = 125, under-reads a slightly warm
cell's emission by more than a bucket's difference).

P5d's measurement (brief §4 (c)), THE HEADROOM ARTIFACT'S SIGNATURE: cell-ticks
where an UNDAMPED cell is clamped AT the headroom ceiling itself (the landing is
e_ceiling_q(Phi)) -- with Phi >= E°[0], i.e. where the headroom exists (below
E°[0] the ceiling is row 31's 0, identical before and after P5d, and it holds a
cell AT ambient: that is the shadow class above, not warming). The ceiling's
bucket out-emits Phi, so in exact arithmetic an undamped cell never climbs there
far enough to be clamped (gate G16 (b)); only an under-read emission can pin it.
Reported: the count per medium, the subset
already sitting in the ceiling's bucket, the hottest such cell and where, and the
longest run of CONSECUTIVE such ticks on one cell whose radiation temperature
e_inv_q(Phi) stays below 20 game -- the brief's STOP test (more than 24 would mean
the artifact is warming the unlit environment).

SCENES
  bench  -- the canonical fire bench (tools/fire_timing_harness.build_level at its
            own defaults: an 84 x 40 open-field planetside arena, ONE kindling
            tile seeded at its ignition_temp, the sky exchange live), 90 s.
  room   -- a sealed hull room (18 x 12 interior) with a 3 x 3 furniture block
            ignited at its ignition_temp: smoke accumulates, 60 s.
  playground -- the shipped playground level with a wood tile at the 1263-game
            plateau and lit (the CUDA gate's live scene), 30 s.

Run:
    C:/Users/steen/anaconda3/python.exe tools/bench_clamp_shave.py [scene ...]
        [--seconds S] [--dump PATH]
`--dump PATH` writes the run-end temperature plane, the mask of cells the clamp
ever touched and the (c) signature counts per cell to PATH (.npz), one file per
scene (PATH is suffixed with the scene name) -- for a paired comparison.
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
STOP_E_INV_GAME = 20          # brief §4: the unlit environment's radiation temperature
STOP_RUN_TICKS = 24           # ... pinned for MORE than this many consecutive ticks


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


def _bucket(t):
    """R.e_bucket_of, vectorized."""
    return np.clip(np.where(t <= 0, 0, t >> R.E_INDEX_SHIFT), 0, R.E_TABLE_SIZE - 1)


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
        self.acc = {m: dict(absorbed=0, gain=0, drop=0, drop_shadow=0, drop_at=0,
                            drop_held=0, hits=0) for m in ("solid", "gas")}
        # shadowed (Phi < E°[0]) net absorbers: rounding the row-31 ceiling holds
        self.shadow = {m: dict(n=0, max_T=None) for m in ("solid", "gas")}
        # net absorbers in or above the ceiling's bucket, by Fleck factor
        self.held = {m: dict(damped=0, damped_max_T=None, undamped=0, undamped_max_T=None)
                     for m in ("solid", "gas")}
        # P5d (c): undamped cells clamped AT the headroom ceiling
        shape = self.g.temperature.shape
        self.sig = {m: dict(n=0, in_bucket=0, hottest=None) for m in ("solid", "gas")}
        self.sig_count = np.zeros(shape, dtype=np.int64)
        self.sig_dark_run = np.zeros(shape, dtype=np.int64)     # consecutive, e_inv < 20
        self.sig_dark_max_run = 0
        self.sig_dark_max_at = None
        self.sig_dark_n = 0
        self.sig_run = np.zeros(shape, dtype=np.int64)          # consecutive, any Phi
        self.sig_max_run = 0
        self.sig_max_at = None
        self.touched = np.zeros(shape, dtype=bool)
        self.ticks = 0

    def close(self):
        self.runner.engine = self.engine

    # -- the ceiling: the ONE place a comparison run may substitute another --
    def _ceiling(self, phi):
        """e_ceiling_q vectorized: the top of the first bucket whose E° exceeds
        Phi; 0 below E°[0]. Held to R.e_ceiling_q on a sample every tick."""
        b = np.searchsorted(self.tbl_np, phi, side="right") - 1
        top = np.minimum(4 * (np.maximum(b, 0) + 2), 4 * R.E_TABLE_SIZE)
        return np.where(b < 0, 0, (top << 16) - 1)

    def _ceiling_1(self, phi):
        return R.e_ceiling_q(int(phi), self.table)

    def _e_inv(self, phi):
        """e_inv_q vectorized -- the RADIATION temperature, for the STOP test:
        the largest bucket b with E°[b] <= Phi, as its low edge; 0 below E°[0]."""
        b = np.searchsorted(self.tbl_np, phi, side="right") - 1
        return np.where(b < 0, 0, (4 * np.maximum(b, 0)) << 16)

    def _signature(self, medium, t_new, e_inv, t0, t_cap, ys, xs):
        """Book (c) on the given cells -- undamped, clamped AT the headroom
        ceiling -- as parallel 1-D arrays with their grid positions."""
        n = len(t_new)
        if not n:
            return
        s = self.sig[medium]
        s["n"] += n
        s["in_bucket"] += int(np.count_nonzero(_bucket(t0) == _bucket(t_cap)))
        k = int(np.argmax(t_new))
        y, x = int(ys[k]), int(xs[k])
        hot = (float(t_new[k]) / 65536.0, float(e_inv[k]) / 65536.0, (y, x),
               int(self.g.material[y, x]), self.ticks)
        if s["hottest"] is None or hot[0] > s["hottest"][0]:
            s["hottest"] = hot

    def after_tick(self):
        g, pre = self.g, self.pre
        rn = g.rad_net_sweep.astype(np.int64)
        phi = g.rad_fluence.astype(np.int64)
        a_eff = np.asarray(self.engine.radiation.a_eff_plane()).astype(np.int64)
        fl = np.asarray(self.engine.radiation.fleck_plane())
        undamped = fl.astype(np.int64) == R.F_ONE
        ts = g.thermal_solid
        acct = g._gas_energy_accountable()
        T0 = pre["T"].astype(np.int64)
        e_inv_all = self._e_inv(phi)
        dark = e_inv_all < (STOP_E_INV_GAME << 16)
        sig_plane = np.zeros(phi.shape, dtype=bool)
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
            t_cap = self._ceiling(f)
            ceiling = np.maximum(t_cap, t0)
            cut = np.maximum(t_after - ceiling, 0)
            s_cap = np.clip(s, R.CAP_SHIFT_MIN, 30)
            cap = np.left_shift(np.int64(1), s_cap + 16)
            e_cut = (cut * cap).astype(object)
            drop = int(e_cut.sum())
            # WHERE it bound: under a SHADOW ceiling (Phi < E°[0], row 31's 0),
            # AT a real ceiling (T_before at or below it), or HELD at T_before
            # (the cell already above it: a bucket past the ceiling's)
            shadow = f < self.table[0]
            held = ~shadow & (t0 > t_cap)
            at = ~shadow & ~held
            self.acc["solid"]["drop_shadow"] += int(e_cut[shadow].sum())
            self.acc["solid"]["drop_at"] += int(e_cut[at].sum())
            self.acc["solid"]["drop_held"] += int(e_cut[held].sum())
            self.acc["solid"]["drop"] += drop
            self.acc["solid"]["hits"] += int(np.count_nonzero(cut > 0))
            drop_total += drop
            ud = undamped[sm]
            warms = t_after > t0
            absorber = warms & ~shadow & (_bucket(t0) >= _bucket(t_cap))
            self._held("solid", absorber & ~ud, absorber & ud, t0)
            self._shadow("solid", warms & shadow, t0)
            at_ceiling = (cut > 0) & at & ud
            t_new = np.minimum(t_after, ceiling)
            sy, sx = np.nonzero(sm)
            self._signature("solid", t_new[at_ceiling], e_inv_all[sm][at_ceiling],
                            t0[at_ceiling], t_cap[at_ceiling], sy[at_ceiling],
                            sx[at_ceiling])
            sig_plane[sm] = at_ceiling
            self.touched[sm] |= cut > 0
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
            t_cap = self._ceiling_1(phi[y, x])
            ceiling = max(t_cap, t_before)
            ud = bool(undamped[y, x])
            shadow = int(phi[y, x]) < self.table[0]
            if t_after > ceiling:
                drop = (t_after - ceiling) * R.cap_real_q(False, 0, nb, self.c_v_q)
                self.acc["gas"]["drop"] += drop
                self.acc["gas"]["hits"] += 1
                if shadow:
                    self.acc["gas"]["drop_shadow"] += drop
                elif t_before > t_cap:
                    self.acc["gas"]["drop_held"] += drop
                else:
                    self.acc["gas"]["drop_at"] += drop
                drop_total += drop
                self.touched[y, x] = True
                if ud and not shadow and t_cap >= t_before:
                    self._signature("gas", np.asarray([t_cap]), np.asarray([e_inv_all[y, x]]),
                                    np.asarray([t_before]), np.asarray([t_cap]),
                                    np.asarray([y]), np.asarray([x]))
                    sig_plane[y, x] = True
            if t_after > t_before and shadow:
                self._shadow("gas", np.asarray([True]), np.asarray([t_before]))
            elif t_after > t_before and R.e_bucket_of(t_before) >= R.e_bucket_of(t_cap):
                # a net absorber in or above the first bucket that out-emits Phi
                tg = np.asarray([t_before])
                self._held("gas", np.asarray([not ud]), np.asarray([ud]), tg)
        self.acc["gas"]["absorbed"] += int(((phi[acct] * a_eff[acct]) >> 16).sum())
        self.acc["gas"]["gain"] += int(np.maximum(rn[acct], 0).sum())
        # ---- P5d (c): the STOP test's runs -- consecutive signature ticks on one
        # cell whose radiation temperature stays below 20 game
        dark_sig = sig_plane & dark
        self.sig_dark_n += int(np.count_nonzero(dark_sig))
        self.sig_count += sig_plane
        self.sig_run = np.where(sig_plane, self.sig_run + 1, 0)
        m_any = int(self.sig_run.max())
        if m_any > self.sig_max_run:
            self.sig_max_run = m_any
            yy, xx = np.unravel_index(int(np.argmax(self.sig_run)), self.sig_run.shape)
            self.sig_max_at = (int(yy), int(xx), self.ticks,
                               float(g.temperature[yy, xx]) / 65536.0,
                               float(e_inv_all[yy, xx]) / 65536.0)
        self.sig_dark_run = np.where(dark_sig, self.sig_dark_run + 1, 0)
        m = int(self.sig_dark_run.max())
        if m > self.sig_dark_max_run:
            self.sig_dark_max_run = m
            yy, xx = np.unravel_index(int(np.argmax(self.sig_dark_run)), self.sig_dark_run.shape)
            self.sig_dark_max_at = (int(yy), int(xx), self.ticks,
                                    float(g.temperature[yy, xx]) / 65536.0,
                                    float(e_inv_all[yy, xx]) / 65536.0)
        # the replay IS the engine's counter, split -- or the bench is wrong
        eng_drop = int(self.ts.e_rad_clamp_drop_sum) - pre["drop"]
        eng_hits = int(self.ts.rad_clamp_hits) - pre["hits"]
        rep_hits = self.acc["solid"]["hits"] + self.acc["gas"]["hits"]
        assert drop_total == eng_drop, (self.ticks, drop_total, eng_drop)
        self._hits_check = getattr(self, "_hits_check", 0) + eng_hits
        assert rep_hits == self._hits_check, (self.ticks, rep_hits, self._hits_check)
        # spot-check the vectorized lookups against the reference's own
        k = int(phi.flat[(self.ticks * 7919) % phi.size])
        assert int(self._ceiling(np.asarray([k]))[0]) == self._ceiling_1(k)
        assert int(self._e_inv(np.asarray([k]))[0]) == R.e_inv_q(k, self.table)
        self.ticks += 1

    def _shadow(self, medium, mask, t0):
        n = int(np.count_nonzero(mask))
        if n:
            sh = self.shadow[medium]
            sh["n"] += n
            tmax = float(np.max(t0[mask])) / 65536.0
            sh["max_T"] = tmax if sh["max_T"] is None else max(sh["max_T"], tmax)

    def _held(self, medium, damped_mask, undamped_mask, t0):
        h = self.held[medium]
        for key, mask in (("damped", damped_mask), ("undamped", undamped_mask)):
            n = int(np.count_nonzero(mask))
            if n:
                h[key] += n
                tmax = float(np.max(t0[mask])) / 65536.0
                prev = h[key + "_max_T"]
                h[key + "_max_T"] = tmax if prev is None else max(prev, tmax)


def run(scene_name, seconds=None, dump=None, meter_cls=ShaveMeter):
    sim, default_s = SCENES[scene_name]()
    seconds = default_s if seconds is None else seconds
    meter = meter_cls(sim)
    from simulation.gases import SMOKE
    from simulation.materials import MATERIAL_NAMES
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
          f"{'drop/absorbed':>14} {'drop/net gain':>14} {'hits':>7}   drop split: "
          f"{'shadow':>7} {'at ceil':>8} {'held':>6}")
    out = {}
    for m in ("solid", "gas"):
        a = meter.acc[m]
        # heat counts (Q16) per tick -> W; the drop is Q32 (cap * dT), so >> 16
        absorbed_w = a["absorbed"] * j * TPS / n
        gain_w = a["gain"] * j * TPS / n
        drop_w = (a["drop"] / 65536.0) * j * TPS / n
        pa = 100.0 * drop_w / absorbed_w if absorbed_w else float("nan")
        pg = 100.0 * drop_w / gain_w if gain_w else float("nan")
        split = {k: (100.0 * a["drop_" + k] / a["drop"] if a["drop"] else float("nan"))
                 for k in ("shadow", "at", "held")}
        print(f"{m:>7} {absorbed_w:12.1f} {gain_w:12.1f} {drop_w:10.3f} "
              f"{pa:13.3f}% {pg:13.3f}% {a['hits']:7d}               "
              f"{split['shadow']:6.1f}% {split['at']:7.1f}% {split['held']:5.1f}%")
        out[m] = dict(absorbed_w=absorbed_w, gain_w=gain_w, drop_w=drop_w,
                      pct_absorbed=pa, pct_gain=pg, hits=a["hits"],
                      pct_shadow=split["shadow"], pct_at=split["at"],
                      pct_held=split["held"])
    print("  drop split: 'shadow' = Phi < E°[0] (row 31's 0 ceiling, unchanged by P5d); "
          "'at ceil' = clamped AT the ceiling; 'held' = above its ceiling, held at T_before")
    for m in ("solid", "gas"):
        sh = meter.shadow[m]
        print(f"  {m} shadowed net absorbers (Phi < E°[0], warmed only by rounding, held "
              f"by row 31): " + (f"{sh['n']} cell-ticks, hottest {sh['max_T']:.2f} game"
                                 if sh["n"] else "none"))
        out[m + "_shadow"] = dict(sh)
    for m in ("solid", "gas"):
        h = meter.held[m]
        dmp = (f"{h['damped']} cell-ticks, hottest {h['damped_max_T']:.1f} game"
               if h["damped"] else "none")
        und = (f"{h['undamped']} cell-ticks, hottest {h['undamped_max_T']:.1f} game"
               if h["undamped"] else "none")
        print(f"  {m} net absorbers (Phi >= E°[0]) in or above the ceiling's bucket "
              f"(E°[bucket] > Phi): "
              f"DAMPED (Fleck, P5b's finding): {dmp}; UNDAMPED (the per-ordinate "
              f"emission floor): {und}")
        out[m + "_held"] = dict(h)
    for m in ("solid", "gas"):
        s = meter.sig[m]
        hot = s["hottest"]
        where = (f"hottest {hot[0]:.2f} game (E_inv(Phi) {hot[1]:.1f}) at (y, x) = "
                 f"{hot[2]}, {MATERIAL_NAMES.get(hot[3], hot[3]) if m == 'solid' else 'gas'}"
                 f", tick {hot[4]}") if hot else "none"
        print(f"  (c) {m}: UNDAMPED cells clamped AT the headroom ceiling (Phi >= E°[0]): {s['n']} "
              f"cell-ticks ({s['in_bucket']} already sitting in the ceiling's bucket); {where}")
        out[m + "_sig"] = dict(n=s["n"], in_bucket=s["in_bucket"], hottest=hot)
    ys, xs = np.nonzero(meter.sig_count)
    top = sorted(zip(meter.sig_count[ys, xs].tolist(), ys.tolist(), xs.tolist()),
                 reverse=True)[:6]
    g = sim.gmap
    print(f"  (c) cells carrying the signature: {len(ys)}; the most frequent "
          + ", ".join(f"({y}, {x}) {'solid ' + str(MATERIAL_NAMES.get(int(g.material[y, x]), '?')) if g.thermal_solid[y, x] else 'gas'} x{c}"
                      for c, y, x in top))
    at_any = meter.sig_max_at
    print(f"  (c) longest consecutive pin at the headroom ceiling on one cell, any Phi: "
          f"{meter.sig_max_run} ticks"
          + (f" at (y, x) = ({at_any[0]}, {at_any[1]}), ending tick {at_any[2]}, T "
             f"{at_any[3]:.2f} game, E_inv(Phi) {at_any[4]:.1f} game" if at_any else ""))
    out.update(sig_max_run=meter.sig_max_run, sig_max_at=at_any)
    stop = meter.sig_dark_max_run > STOP_RUN_TICKS
    at = meter.sig_dark_max_at
    print(f"  (c) STOP TEST: signature cell-ticks with E_inv(Phi) < {STOP_E_INV_GAME} game: "
          f"{meter.sig_dark_n}; longest consecutive run on one cell: "
          f"{meter.sig_dark_max_run} ticks"
          + (f" at (y, x) = ({at[0]}, {at[1]}), ending tick {at[2]}, T {at[3]:.2f} game, "
             f"E_inv(Phi) {at[4]:.1f} game" if at else "")
          + f" -> {'STOP (more than ' + str(STOP_RUN_TICKS) + ')' if stop else 'carry on'}")
    out.update(sig_dark_n=meter.sig_dark_n, sig_dark_max_run=meter.sig_dark_max_run,
               sig_dark_max_at=at, stop=stop)
    if dump is not None:
        path = Path(f"{dump}_{scene_name}.npz")
        np.savez(path, T=g.temperature.copy(), touched=meter.touched,
                 sig_count=meter.sig_count, thermal_solid=g.thermal_solid.copy(),
                 material=g.material.copy(), ticks=np.int64(n))
        print(f"  wrote {path}")
    return out


def main(argv):
    seconds = None
    dump = None
    if "--seconds" in argv:
        i = argv.index("--seconds")
        seconds = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if "--dump" in argv:
        i = argv.index("--dump")
        dump = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    names = [a for a in argv[1:] if not a.startswith("--")] or list(SCENES)
    for name in names:
        run(name, seconds, dump)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
