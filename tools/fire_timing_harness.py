"""Headless single-crate FIRE-TIMING harness (Fire & Heat tuning, §5.2 bench).

A tuning INSTRUMENT: it measures, in SECONDS, how long one furniture crate takes
to ramp up and burn out. It is DIAGNOSTIC ONLY -- it does NOT change any game
mechanic. It instruments and DRIVES the EXISTING fire model (config.toml
[physics.fire]/[physics.combustion]/[physics.thermal]) via allowed debug field
writes, and can PATCH config dials at runtime (--k-grow/--k-die/--set) so we tune
without editing config.toml.

Companion to docs/fire_tuning_plan_2026-07-22.md §2 (the model) + §5.2 (the
first-principles single-crate numbers this bench confirms against).

SCENARIO (isolates fire from O2 starvation, per Erik):
  * PLANETSIDE boundary (boundary="ambient"): a SPACE(9) ring border becomes the
    open-sky reservoir (GameMap.is_ambient), so the interior air stays oxygenated.
    Normal air: O2 = 0.21 (the Earth-normal ambient default; NOT enriched).
  * SKY-EXCHANGE AWARE (docs/sky_exchange_design_2026-07-24.md): the bench's
    LevelData carries an explicit [ambient] AmbientConfig with `sky_tau_s` (the
    vertical-mixing timescale, default 60 s here) and `sponge_width` (8). With
    the sky exchange ON, every sky-connected air tile relaxes its O2/N2
    COMPOSITION toward ambient at fixed local N_total each tick -- the vertical
    refill a top-down slice cannot resolve. This is what lets a crate deep in the
    room stay O2-fed. NOTE: a hand-built LevelData with ambient=None makes GameMap
    synthesize derive_ambient() with ALL DEFAULTS (sky_tau_s=0 == DORMANT); we
    must pass an explicit AmbientConfig to turn the pass on.
  * ONE furniture crate (material `furniture`: hp 30, ignition_temp 280,
    flammable). With sky exchange ON the crate is placed DEEP in the room
    (default x=12, y=centre), SPONGE-SAFE (>= sponge_width + a few tiles from
    EVERY ring). The old ring-adjacent placement (x=1) was a workaround for the
    pre-sky-exchange O2-starvation of a deep crate; the sky refill now keeps the
    deep crate O2-fed, so the fire ramps/burns on fuel + thermal + k_grow/k_die
    alone AND the crate is no longer sitting inside the boundary sponge band.
  * Ignite at LOW intensity: gmap.fire[crate] = ignition_seed (0.1).

WIND (two modes):
  * DEFAULT / still air = NATURAL wind: the full runner tick computes the wind
    itself (the fire's own plume overpressure + convective cooling). NO forcing --
    a fire always interacts with still air + its own plume. This is the honest
    baseline for k_grow/k_die.
  * --wind W (W != 0) = FORCED constant wind: a steady, uniform +x wind of
    magnitude W (dequantized units) is injected each tick (WindForcer, pure
    harness-side -- no sim-code change) so a strong breeze dominates the plume.

DIAL OVERRIDES (patch config.toml values before the sim is built; nothing is
written to disk):
    --k-grow 0.08 --k-die 0.04            # shortcuts for [physics.fire]
    --set physics.fire.fuel_ref=120       # any dotted CFG path (repeatable)
    --set wall_damage=0.05                # bare key => [physics.fire]

RUN MODES:
    python tools/fire_timing_harness.py                 # NATURAL still air
    python tools/fire_timing_harness.py --wind 3.0      # forced-wind run
    python tools/fire_timing_harness.py --wind-sweep    # W=0(natural)+forced -> W->m/s map
    python tools/fire_timing_harness.py --k-sweep       # (k_grow,k_die) pair sweep table

Deterministic: fixed seed, no RNG in the driven path (same args -> same numbers).
Headless: builds a synthetic LevelData in memory; never opens a display.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# --- import path (mirror tests/field_ab_harness.py) ------------------------
ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                       # noqa: E402
from config import CFG                            # noqa: E402
from level_loader import LevelData                # noqa: E402
from simulation import Simulation                 # noqa: E402
from simulation import fire_fixed                 # noqa: E402
from simulation.gases import SMOKE, O2, INERT_N2  # noqa: E402

FP_ONE = 1 << 16

# Canon material ids == CSV codes (tools/gen_fire_studio.py, simulation.materials).
AIR, HULL, WOOD, DOOR, STEEL, GLASS, FURN, SPACE = 0, 1, 2, 3, 4, 5, 6, 9


def _smoothstep(a, b, x):
    """The fire's O2 gate curve: smoothstep(P_min, P_full, O2)."""
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def _hot_gate(T, fire_T_ext, fire_T_span):
    """The REAL temperature gate fire_simulation.cpp's growth logistic reads:
    hot = clamp01((T - fire_T_ext) / fire_T_span) -- LINEAR, not a smoothstep
    (mirrors the C++ clamp01_q(recip_mul(T - fire_T_ext_q, recip_T_span)) exactly).
    hot < 1 means the flame temperature itself has dipped toward extinction --
    gate-limited death, as distinct from O2-limited (see x_local below)."""
    if fire_T_span <= 0:
        return 1.0 if T >= fire_T_ext else 0.0
    return min(1.0, max(0.0, (T - fire_T_ext) / fire_T_span))


# ---------------------------------------------------------------------------
# Config-dial overrides (patch CFG in place; NOTHING is written to config.toml).
# CFG is a plain mutable Namespace tree -- PhysicsRunner reads [physics.fire] at
# construction, so patching before Simulation(...) is picked up. We snapshot +
# restore so runs don't leak into each other.
# ---------------------------------------------------------------------------
def _resolve_key(dotted):
    """'k_grow' -> (CFG.physics.fire, 'k_grow'); 'physics.combustion.burn_rate' ->
    navigate the CFG tree. A bare key defaults to the [physics.fire] section."""
    parts = dotted.split(".")
    if len(parts) == 1:
        parts = ["physics", "fire", parts[0]]
    obj = CFG
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"--set: unknown config key '{dotted}'")
    return obj, leaf


def _coerce(old, v):
    if not isinstance(v, str):
        return v
    if isinstance(old, bool):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(old, int) and not isinstance(old, bool):
        return int(v)
    if isinstance(old, float):
        return float(v)
    return v


def apply_overrides(overrides):
    """Patch CFG per an ordered dict {dotted_key: value}. Returns a restore list."""
    restore = []
    for k, v in overrides.items():
        obj, leaf = _resolve_key(k)
        old = getattr(obj, leaf)
        restore.append((obj, leaf, old))
        setattr(obj, leaf, _coerce(old, v))
    return restore


def restore_overrides(restore):
    for obj, leaf, old in restore:
        setattr(obj, leaf, old)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------
def build_level(interior_w, interior_h, crate_xy, tile_size_m,
                sky_tau_s=60.0, sponge_width=8):
    """A synthetic planetside bench: a 1-tile SPACE ring (-> is_ambient reservoir)
    around an open AIR interior with ONE furniture crate. No hull, so the interior
    air is directly bounded by the ambient sky (open field).

    SKY-EXCHANGE AWARE: the LevelData carries an EXPLICIT AmbientConfig (via
    simulation.ambient.derive_ambient) so the per-tick sky-exchange pass is ACTIVE
    with the given ``sky_tau_s`` (0 == dormant) and the given ``sponge_width``.
    If we left ambient=None, GameMap would synthesize derive_ambient() with ALL
    defaults for a hand-built LevelData -- and DEFAULT_SKY_TAU_S == 0.0, so the
    sky refill would be DORMANT (the very thing under test). See gamemap.py
    ("hand-built LevelData: Earth defaults") + sky_exchange_design_2026-07-24."""
    from simulation.ambient import derive_ambient
    h, w = interior_h + 2, interior_w + 2
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = SPACE
    tm[h - 1, :] = SPACE
    tm[:, 0] = SPACE
    tm[:, w - 1] = SPACE
    cx, cy = crate_xy
    tm[cy, cx] = FURN
    ambient = derive_ambient(sky_tau_s=float(sky_tau_s),
                             sponge_width=int(sponge_width))
    return LevelData(
        name="fire_bench", version="2", path=Path("."),
        tilemap=tm, tile_size_m=float(tile_size_m), diffuse_path=Path("."),
        boundary="ambient",   # planetside: SPACE ring -> ambient O2=0.21 reservoir
        ambient=ambient,      # EXPLICIT: turns the sky-exchange pass ON (tau>0)
    )


# ---------------------------------------------------------------------------
# Forced-wind injection -- PURE harness-side, no sim-code change. ONLY used for
# the FORCED windy scenario (--wind W, W != 0); the still-air baseline runs the
# runner tick untouched so the wind is the fire's own natural plume/convection.
# ---------------------------------------------------------------------------
class WindForcer:
    """Monkeypatch a PhysicsRunner instance so the fire (and smoke transport) sees
    a STEADY UNIFORM +x wind of magnitude ``wind_q`` (Q16.16) each tick.

    Tick order (physics_runner.PhysicsRunner.step): cast_fire_heat -> _step_water
    -> eos.run_substeps (WRITES wind_x/wind_y, advects smoke on its output wind)
    -> _run_combustion -> step_tail (the FIRE logistic READS |wind| here). Two
    seams, both pure ``gmap.wind_*`` writes: top-of-step() seeds the transport
    wind so smoke rides ~W; _run_combustion re-forces so the fire reads exactly W.
    """

    def __init__(self, runner, wind_q: int):
        self.wind_q = int(wind_q)
        self._orig_step = runner.step
        self._orig_comb = runner._run_combustion
        runner.step = self._step
        runner._run_combustion = self._comb

    def _force(self, gmap):
        m = ~gmap.is_ambient
        gmap.wind_x[m] = self.wind_q
        gmap.wind_y[m] = 0

    def _step(self, gmap, sim_time):
        self._force(gmap)                 # seam 1: smoke rides ~W in run_substeps
        return self._orig_step(gmap, sim_time)

    def _comb(self, gmap, sim_time):
        self._force(gmap)                 # seam 2: fire reads exactly W in step_tail
        return self._orig_comb(gmap, sim_time)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def _open_neighbors(gmap, cy, cx):
    h, w = gmap.fire.shape
    out = []
    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
        if 0 <= ny < h and 0 <= nx < w:
            if not bool(gmap.solid[ny, nx]) and not bool(gmap.is_vacuum[ny, nx]):
                out.append((ny, nx))
    return out


def _far_probes(interior_w, interior_h, sponge_width):
    """Genuinely-far O2 sample points (FULL-map coords) for the sky gate-d
    acceptance -- room centre + two opposite far quadrants. All are kept clear of
    the boundary SPONGE band (>= sponge_width+2 tiles from every ring) so we read
    true far-field air, not the sponge/ring, and (with a deep crate) they sit far
    from the flame's local vitiated zone. The far-field O2 reported per tick is
    the MIN over these (the worst-case far point -> the strongest 'field holds'
    statement)."""
    m = int(sponge_width) + 2
    lo_r, hi_r = 1 + m, interior_h - m
    lo_c, hi_c = 1 + m, interior_w - m
    clamp_r = lambda r: max(lo_r, min(hi_r, r))
    clamp_c = lambda c: max(lo_c, min(hi_c, c))
    pts = [(1 + interior_h // 2, 1 + interior_w // 2),            # room centre
           (1 + interior_h // 4, 1 + (3 * interior_w) // 4),      # upper-far
           (1 + (3 * interior_h) // 4, 1 + (3 * interior_w) // 4)]  # lower-far
    seen, out = set(), []
    for r, c in pts:
        p = (clamp_r(r), clamp_c(c))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def run_one(wind_dq, *, interior_w, interior_h, crate_xy, tile_size_m,
            max_seconds, tail_seconds, overrides=None, seed=12345, verbose=True,
            snapshot_times_s=None, sky_tau_s=60.0, sponge_width=8):
    """Run one single-crate burn. ``wind_dq == 0`` -> NATURAL wind (no forcing);
    ``wind_dq != 0`` -> FORCED constant +x wind. ``overrides`` patches CFG dials
    (restored afterwards). ``sky_tau_s`` / ``sponge_width`` -> the [ambient]
    sky-exchange timescale (0 == dormant) + sponge band width baked into the
    bench LevelData. ``snapshot_times_s`` (increasing list) -> capture a copy of
    the FULL 2D ``gmap.gas[O2]`` field (raw Q16.16) at each requested sim-time,
    returned under ``metrics['o2_snapshots']`` as (tick, t_s, (h,w) int32). Returns
    a metrics dict."""
    restore = apply_overrides(overrides or {})
    try:
        return _run_one_inner(
            wind_dq, interior_w=interior_w, interior_h=interior_h, crate_xy=crate_xy,
            tile_size_m=tile_size_m, max_seconds=max_seconds, tail_seconds=tail_seconds,
            overrides=overrides or {}, seed=seed, verbose=verbose,
            snapshot_times_s=snapshot_times_s,
            sky_tau_s=sky_tau_s, sponge_width=sponge_width)
    finally:
        restore_overrides(restore)


def _run_one_inner(wind_dq, *, interior_w, interior_h, crate_xy, tile_size_m,
                   max_seconds, tail_seconds, overrides, seed, verbose,
                   snapshot_times_s=None, sky_tau_s=60.0, sponge_width=8):
    forced = (float(wind_dq) != 0.0)
    level = build_level(interior_w, interior_h, crate_xy, tile_size_m,
                        sky_tau_s=sky_tau_s, sponge_width=sponge_width)
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
    gmap = sim.gmap
    cx, cy = crate_xy

    wind_q = int(round(float(wind_dq) * FP_ONE))
    if forced:
        WindForcer(sim.physics_runner, wind_q)

    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
    gmap.fire[cy, cx] = fire_fixed.quantize_scalar(seed_i)
    # Game-faithful seed (Fable 2026-07-25): in-engine a tile only ignites
    # BECAUSE its T crossed ignition_temp — a cold-started seed (T=ambient)
    # is an unphysical bootstrap race the game never runs (it made the
    # k_fire_heat sweeps look chaotic). Seed the crate tile at furniture's
    # ignition_temp (280) so the bench starts where real ignition starts.
    gmap.temperature[cy, cx] = fire_fixed.quantize_scalar(280.0)

    p_min = float(getattr(CFG.physics.fire, "P_min", 0.01))
    p_full = float(getattr(CFG.physics.fire, "P_full", 0.03))
    # PER-MATERIAL fire_T_ext (P-R3, 2026-07-31 — ruling A3 ride-along). The
    # `hot` gate's FOOT is no longer the [physics.fire] global: it is derived per
    # material as `ignition_temp[mat] - ignition_to_ext_delta` and baked into
    # `GameMap.fire_T_ext_plane`. Read the CRATE TILE'S OWN value straight out
    # of that plane — the exact integer the solver subtracts — so the `hot`
    # column below stays a faithful mirror of the C++ gate instead of a stale
    # global. (For the shipped crate that is furniture 280-100 = 180, i.e. the
    # blessed bench value, so a derived run reproduces the old
    # `--set fire_T_ext=180` run.) `fire_T_span` is still global.
    fire_t_ext = int(gmap.fire_T_ext_plane[cy, cx]) / FP_ONE
    fire_t_span = float(getattr(CFG.physics.fire, "fire_T_span", 150.0))
    k_grow = float(getattr(CFG.physics.fire, "k_grow", 4.0))
    k_die = float(getattr(CFG.physics.fire, "k_die", 2.0))

    tps = float(CFG.clock.ticks_per_second)
    dt = 1.0 / tps
    nbrs = _open_neighbors(gmap, cy, cx)
    h, w = gmap.fire.shape
    xgrid = np.arange(w, dtype=np.float64)[None, :]
    # Far-field O2 probes (sky gate d): min over room-centre + far quadrants,
    # all clear of the sponge band and far from the deep crate's flame zone.
    far_pts = _far_probes(interior_w, interior_h, sponge_width)
    # sky-connected interior air mask (fixed in this bench: no structural edits) —
    # room-mean far-field bulk. Used to separate a PLANE collapse (thermal
    # decompression at ring-pinned P) from true composition vitiation (X drops).
    sky_mask = gmap.ensure_sky_mask()
    if not sky_mask.any():
        sky_mask = (~gmap.solid) & (~gmap.is_vacuum) & (~gmap.is_ambient)

    crate_hp0 = int(gmap.wall_hp[cy, cx]) / FP_ONE
    o2_seed = float(np.mean([int(gmap.gas[O2, ny, nx]) for (ny, nx) in nbrs])) / FP_ONE

    n_max = int(round(max_seconds * tps))
    # 2D O2-field snapshot targets (ticks), captured at the first loop-tick at or
    # past each requested time (assumes increasing snapshot_times_s).
    snap_targets = [max(1, min(int(round(ts * tps)), n_max))
                    for ts in (snapshot_times_s or [])]
    o2_snaps = []
    _sptr = 0
    rec = {k: [] for k in ("t", "I", "T", "hp", "o2", "gate", "o2far", "cx", "mass",
                           "o2far_x", "o2room", "o2room_x", "ntot_room", "tfar",
                           "x_local", "hot")}
    had_fire = False
    fuel_out_tick = None       # wall_hp first <= 0  (the meaningful "burnout")
    snap_tick = None           # I first snaps to 0 after burning
    for k in range(1, n_max + 1):
        sim.set_paused(False)
        sim.step()
        t = k * dt
        fire_q = int(gmap.fire[cy, cx])
        I = fire_q / FP_ONE
        T = int(gmap.temperature[cy, cx]) / FP_ONE
        hp = int(gmap.wall_hp[cy, cx]) / FP_ONE
        o2 = (float(np.mean([int(gmap.gas[O2, ny, nx]) for (ny, nx) in nbrs]))
              / FP_ONE) if nbrs else float("nan")
        gate = _smoothstep(p_min, p_full, o2)
        # LOCAL flame O2 mole fraction X (step-4 diagnosis addition, 2026-07-25):
        # Sigma n_o2 / Sigma n_total over the SAME open 4-neighbours `o2` reads
        # above -- mirrors fire_simulation.cpp's own X read EXACTLY (a fraction
        # of SUMS over the neighbours, not a mean of per-tile fractions), so this
        # is the TRUE local value the continuous-O2 law's growth-logistic factor
        # o2f = clamp01((X-X_ext)/(X_amb-X_ext)) gates on -- not a proxy.
        if nbrs:
            _o2_loc = float(sum(int(gmap.gas[O2, ny, nx]) for (ny, nx) in nbrs))
            _tot_loc = float(sum(int(gmap.gas[O2, ny, nx]) + int(gmap.gas[INERT_N2, ny, nx])
                                 for (ny, nx) in nbrs))
            x_local = _o2_loc / max(1.0, _tot_loc)
        else:
            x_local = float("nan")
        # The OTHER real gate fire_simulation.cpp's growth logistic reads:
        # hot = clamp01((T - fire_T_ext) / fire_T_span). hot < 1 means the flame
        # temperature itself has dipped toward extinction -- gate-limited, as
        # distinct from O2-limited (x_local falling instead; see above).
        hot = _hot_gate(T, fire_t_ext, fire_t_span)
        o2far = min(int(gmap.gas[O2, py, px]) for (py, px) in far_pts) / FP_ONE
        # Far-field TEMPERATURE (step-3 heat-balance addition, 2026-07-25): same
        # far_pts tile set as o2far/o2far_x (room centre + two far quadrants,
        # already >10 tiles from a DEEP crate and clear of the sponge band) --
        # mirrors that probe's LOCATIONS. Unlike O2 (where the worst case is the
        # MIN), the worst case for a "does the room overheat" reading is the MAX
        # over the probes -- the room-T rise Erik wants bounded (target <= ~20
        # game units, far from the flame's own local hot zone).
        tfar = max(int(gmap.temperature[py, px]) for (py, px) in far_pts) / FP_ONE
        # O2 MOLE FRACTION X = O2/(O2+inert_N2) — the density-invariant quantity
        # the continuous-O2 law gates on. Far-field X (min over probes) + room-mean
        # X disambiguate a PLANE drop (decompression: N_total falls, X holds) from
        # true composition vitiation (products displace O2: X falls).
        xfar = min((int(gmap.gas[O2, py, px]) /
                    max(1, int(gmap.gas[O2, py, px]) + int(gmap.gas[INERT_N2, py, px])))
                   for (py, px) in far_pts)
        _o2m = gmap.gas[O2][sky_mask].astype(np.float64)
        _ntm = _o2m + gmap.gas[INERT_N2][sky_mask].astype(np.float64)
        o2room = float(_o2m.mean()) / FP_ONE
        ntot_room = float(_ntm.mean()) / FP_ONE
        o2room_x = float(np.divide(_o2m, _ntm, out=np.zeros_like(_o2m),
                                   where=_ntm > 0).mean())
        smoke = gmap.gas[SMOKE].astype(np.float64)
        mass = float(smoke.sum())
        cxc = float((smoke * xgrid).sum() / mass) if mass > 1.0 else float("nan")
        for key, val in (("t", t), ("I", I), ("T", T), ("hp", hp), ("o2", o2),
                         ("gate", gate), ("o2far", o2far), ("cx", cxc), ("mass", mass),
                         ("o2far_x", xfar), ("o2room", o2room), ("o2room_x", o2room_x),
                         ("ntot_room", ntot_room), ("tfar", tfar),
                         ("x_local", x_local), ("hot", hot)):
            rec[key].append(val)
        if I > 0.05:
            had_fire = True
        if had_fire and hp <= 0.0 and fuel_out_tick is None:
            fuel_out_tick = k
        if had_fire and fire_q == 0 and snap_tick is None:
            snap_tick = k
        while _sptr < len(snap_targets) and k >= snap_targets[_sptr]:
            o2_snaps.append((k, k * dt, gmap.gas[O2].copy()))
            _sptr += 1
        done_tick = snap_tick if snap_tick is not None else fuel_out_tick
        if done_tick is not None and t > done_tick * dt + tail_seconds:
            break

    # Any snapshot targets past the (early-stopped) run end -> the final field.
    while _sptr < len(snap_targets):
        o2_snaps.append((k, k * dt, gmap.gas[O2].copy()))
        _sptr += 1

    for key in rec:
        rec[key] = np.asarray(rec[key], dtype=np.float64)

    # --- derive metrics -----------------------------------------------------
    I_arr, t_arr, T_arr, hp_arr = rec["I"], rec["t"], rec["T"], rec["hp"]
    peak_I = float(I_arr.max()) if I_arr.size else 0.0
    peak_idx = int(np.argmax(I_arr)) if I_arr.size else 0
    peak_time = float(t_arr[peak_idx]) if I_arr.size else float("nan")  # argmax time (I at MAX)
    time_to_peak = float("nan")     # ignition -> I first reaches 0.9 * its own peak
    if peak_I > 0:
        hit = np.nonzero(I_arr >= 0.9 * peak_I)[0]
        if hit.size:
            time_to_peak = float(t_arr[hit[0]])
    burnout_time = float(fuel_out_tick * dt) if fuel_out_tick is not None else float("nan")
    snap_time = float(snap_tick * dt) if snap_tick is not None else float("nan")
    # STALL = never grew meaningfully above the ignition seed (pinned ~0.1).
    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
    stalled = peak_I < 1.3 * seed_i

    # steady/sustained T: median T over the burning window [time_to_peak, fuel-out]
    end_i = fuel_out_tick if fuel_out_tick is not None else len(t_arr)
    start_i = int(np.searchsorted(t_arr, time_to_peak)) if not np.isnan(time_to_peak) else 0
    win = T_arr[start_i:end_i] if end_i > start_i else T_arr
    steady_T = float(np.median(win)) if win.size else float("nan")
    # Plateau readings of the two REAL gates (step-4 diagnosis addition,
    # 2026-07-25), over the EXACT SAME sustained-burning window as steady_T --
    # disambiguates O2-limited (x_local_plateau falls) from gate-limited
    # (hot_plateau falls below 1) fire behaviour.
    x_local_win = rec["x_local"][start_i:end_i] if end_i > start_i else rec["x_local"]
    hot_win = rec["hot"][start_i:end_i] if end_i > start_i else rec["hot"]
    x_local_plateau = float(np.nanmedian(x_local_win)) if x_local_win.size else float("nan")
    hot_plateau = float(np.nanmedian(hot_win)) if hot_win.size else float("nan")

    # self-collapse: after the peak, does T fall BELOW fire_T_ext while a lot of
    # fuel remains (crate goes cold with fuel left = a plume-cooling collapse)?
    collapsed = False
    if peak_I >= 0.2 and len(T_arr) > peak_idx + 1:
        post = slice(peak_idx + 1, len(T_arr))
        cold = (T_arr[post] < fire_t_ext) & (hp_arr[post] > 0.5 * crate_hp0)
        collapsed = bool(cold.any())

    o2_arr = rec["o2"][~np.isnan(rec["o2"])]
    o2_min = float(o2_arr.min()) if o2_arr.size else float("nan")
    gate_min = float(rec["gate"].min()) if rec["gate"].size else float("nan")
    o2far_min = float(rec["o2far"].min()) if rec["o2far"].size else float("nan")
    # Mole-fraction (density-invariant) far-field acceptance: the plane can fall
    # from thermal DECOMPRESSION at pinned P while X (composition) holds — X is the
    # quantity the continuous-O2 law actually gates on, so it is the correct gate-d
    # metric under the new law.
    o2far_x_min = float(rec["o2far_x"].min()) if rec["o2far_x"].size else float("nan")
    o2room_x_min = float(rec["o2room_x"].min()) if rec["o2room_x"].size else float("nan")
    o2room_min = float(rec["o2room"].min()) if rec["o2room"].size else float("nan")
    ntot_room_min = float(rec["ntot_room"].min()) if rec["ntot_room"].size else float("nan")
    tfar_max = float(rec["tfar"].max()) if rec["tfar"].size else float("nan")

    # smoke drift (only meaningful under forced wind): centroid slope, clean window
    drift_tiles = drift_ms = float("nan")
    valid = ~np.isnan(rec["cx"])
    idx = np.nonzero(valid)[0]
    idx = idx[(rec["cx"][idx] < (interior_w * 0.70)) & (rec["mass"][idx] > 50.0)
              & (t_arr[idx] <= 6.0)]
    if idx.size >= 6:
        idx = idx[2:]
        slope = np.polyfit(t_arr[idx], rec["cx"][idx], 1)[0]
        drift_tiles = float(slope)
        drift_ms = float(slope * tile_size_m)

    metrics = dict(
        wind_dq=float(wind_dq), wind_q=wind_q, forced=forced, tps=tps, dt=dt,
        k_grow=k_grow, k_die=k_die, overrides=dict(overrides),
        crate_hp0=crate_hp0, o2_seed=o2_seed, p_full=p_full,
        peak_I=peak_I, peak_time=peak_time, time_to_peak=time_to_peak,
        burnout_time=burnout_time,
        snap_time=snap_time, stalled=stalled, collapsed=collapsed, steady_T=steady_T,
        o2_min=o2_min, gate_min=gate_min, o2far_min=o2far_min,
        o2far_x_min=o2far_x_min, o2room_x_min=o2room_x_min,
        o2room_min=o2room_min, ntot_room_min=ntot_room_min, tfar_max=tfar_max,
        x_local_plateau=x_local_plateau, hot_plateau=hot_plateau,
        # P-R3: the crate's OWN derived extinction floor (ignition_temp - Delta),
        # recorded so the scorecard reports the number the solver actually used
        # rather than the retired [physics.fire] global.
        fire_T_ext=float(fire_t_ext), fire_T_span=float(fire_t_span),
        hp_end=float(hp_arr[-1]) if hp_arr.size else float("nan"),
        drift_tiles=drift_tiles, drift_ms=drift_ms,
        n_ticks=len(t_arr), rec=rec, nbrs=len(nbrs),
        interior_w=interior_w, interior_h=interior_h, crate_xy=tuple(crate_xy),
        tile_size_m=tile_size_m, o2_snapshots=o2_snaps,
        sky_tau_s=float(sky_tau_s), sponge_width=int(sponge_width),
        far_pts=[tuple(int(v) for v in p) for p in far_pts],
    )
    if verbose:
        _print_run(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Time-series CSV dump (the full per-tick record; feeds external plotting)
# ---------------------------------------------------------------------------
def write_timeseries_csv(m, path):
    """Dump the harness's own per-tick time-series to CSV (t in s AND minutes,
    plus I / T / wall_hp / O2-at-flame / O2-gate / O2-far / soot / far-field
    room temperature). A commented header carries the dials + the derived
    peak/burnout metrics."""
    import csv
    rec = m["rec"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# fire_timing_harness single-crate time-series"])
        w.writerow([f"# k_grow={m['k_grow']} k_die={m['k_die']} overrides={m['overrides']}"])
        w.writerow([f"# tps={m['tps']:.0f} dt={m['dt']:.6f}s wind={'forced ' + str(m['wind_dq']) if m['forced'] else 'natural'}"])
        w.writerow([f"# peak_I={m['peak_I']:.4f} peak_time_s={m['peak_time']:.2f} "
                    f"time_to_90pct_s={m['time_to_peak']:.2f} burnout_s={m['burnout_time']:.2f} "
                    f"steady_T={m['steady_T']:.0f}"])
        w.writerow(["t_s", "t_min", "I", "T_game", "wall_hp",
                    "O2_flame_nbr", "O2_gate", "O2far_plane", "soot_deq",
                    "O2far_X", "O2room_plane", "O2room_X", "Ntot_room", "Tfar_game",
                    "X_local", "hot"])
        for i in range(len(rec["t"])):
            w.writerow([f"{rec['t'][i]:.4f}", f"{rec['t'][i] / 60.0:.6f}",
                        f"{rec['I'][i]:.6f}", f"{rec['T'][i]:.3f}", f"{rec['hp'][i]:.6f}",
                        f"{rec['o2'][i]:.6f}", f"{rec['gate'][i]:.4f}",
                        f"{rec['o2far'][i]:.6f}", f"{rec['mass'][i] / FP_ONE:.4f}",
                        f"{rec['o2far_x'][i]:.6f}", f"{rec['o2room'][i]:.6f}",
                        f"{rec['o2room_x'][i]:.6f}", f"{rec['ntot_room'][i]:.6f}",
                        f"{rec['tfar'][i]:.4f}",
                        f"{rec['x_local'][i]:.6f}", f"{rec['hot'][i]:.4f}"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _mmss(sec):
    if np.isnan(sec):
        return "  n/a "
    return f"{sec:6.1f}s ({sec/60:4.2f}min)"


def _print_run(m):
    mode = (f"FORCED wind {m['wind_dq']:.3f} dq (wind_x={m['wind_q']} Q16.16)"
            if m["forced"] else "NATURAL wind (own plume + convection)")
    print("=" * 76)
    print(f"{mode}   dt = {m['dt']:.6f} s  ({m['tps']:.0f} tps)")
    print(f"  dials: k_grow={m['k_grow']:g}  k_die={m['k_die']:g}"
          + (f"   overrides: {m['overrides']}" if m["overrides"] else ""))
    _sky = (f"sky_tau_s={m.get('sky_tau_s', 0.0):g}s"
            + (" DORMANT" if m.get('sky_tau_s', 0.0) <= 0 else ""))
    print(f"  scenario: interior {m['interior_w']}x{m['interior_h']}, crate at "
          f"{m['crate_xy']} (deep), tile {m['tile_size_m']} m; PLANETSIDE O2=0.21; "
          f"{_sky}, sponge_width={m.get('sponge_width', 0)}")
    print(f"  far-field O2 probes (min reported): {m.get('far_pts', [])}")
    print("-" * 76)
    if m["stalled"]:
        print(f"  ** STALLED: fire never ramped (peak I={m['peak_I']:.3f}) -- "
              f"O2-starved smolder or death-term dominates the seed **")
    if m["collapsed"]:
        print(f"  ** SELF-COLLAPSE: after peak, T fell below fire_T_ext with >50% fuel "
              f"left (plume cooling snuffed it) **")
    print(f"  peak I:                     {m['peak_I']:.3f}   at t = {_mmss(m['peak_time'])}  (I at MAX)")
    print(f"    (time-to-90%-of-peak:     {_mmss(m['time_to_peak'])} -- crossing on the way up)")
    print(f"  steady T (game units):      {m['steady_T']:.0f}   "
          f"(>fire_T_ext={m.get('fire_T_ext', float('nan')):.0f} -> sustains; "
          f"PER-MATERIAL since P-R3 — ignition_temp - ignition_to_ext_delta)")
    print(f"  burnout (wall_hp -> 0):     {_mmss(m['burnout_time'])}   [fuel exhausted]")
    if np.isnan(m["snap_time"]):
        print(f"  fire snap-out (I -> 0):     none (zombie smolder: furniture never destroyed;")
        print(f"                                    re-ignition re-seeds I=0.1 while T>ignition_temp)")
    else:
        print(f"  fire snap-out (I -> 0):     {_mmss(m['snap_time'])}")
    sat = "SATURATED (O2-unlimited)" if m["gate_min"] > 0.999 else f"DIPPED to {m['gate_min']:.2f}"
    print(f"  O2 (diag smoothstep):       {sat}  (flame-nbr O2 min {m['o2_min']:.4f})")
    print(f"  --- sky gate d (far-field O2, want >= 0.19) ---")
    print(f"    O2 PLANE   far-min: {m['o2far_min']:.4f}   room-mean-min: {m.get('o2room_min', float('nan')):.4f}"
          f"   (plane falls with DECOMPRESSION: N_total drops at pinned P)")
    print(f"    O2 X=frac  far-min: {m.get('o2far_x_min', float('nan')):.4f}   room-mean-min: "
          f"{m.get('o2room_x_min', float('nan')):.4f}   <- the density-invariant gate the new law reads")
    print(f"    room N_total min: {m.get('ntot_room_min', float('nan')):.4f}  (1.0 = ambient; "
          f"<1 = the room decompressed as it heated)")
    print(f"  far-field ROOM TEMPERATURE (>10 tiles, max over probes): "
          f"{m.get('tfar_max', float('nan')):7.2f} game  (target rise <= ~20)")
    print(f"  --- plateau REAL gates (sustained-burn window, same as steady_T) ---")
    print(f"    x_local (flame-nbr O2 mole frac X): {m.get('x_local_plateau', float('nan')):.4f}"
          f"   (X_ext={float(getattr(CFG.physics.fire, 'o2_frac_ext', 0.13)):.2f} extinguishes)")
    print(f"    hot     (clamp01((T-fire_T_ext)/fire_T_span)): {m.get('hot_plateau', float('nan')):.4f}"
          f"   (<1 => T-gate limited)")
    if m["forced"]:
        print(f"  smoke drift:                {m['drift_tiles']:.3f} tiles/s = {m['drift_ms']:.3f} m/s")
    print("-" * 76)
    rec = m["rec"]
    step = max(1, int(round(5.0 * m["tps"])))    # ~every 5 s
    print("   t[s]     I      T(game)   wall_hp   O2nbr  gate   O2far  soot(deq)")
    idxs = list(range(0, len(rec["t"]), step))
    if (len(rec["t"]) - 1) not in idxs:
        idxs.append(len(rec["t"]) - 1)
    for i in idxs:
        tag = "  <- final" if i == len(rec["t"]) - 1 else ""
        print(f"  {rec['t'][i]:6.1f}  {rec['I'][i]:5.3f}  {rec['T'][i]:8.1f}  "
              f"{rec['hp'][i]:7.3f}  {rec['o2'][i]:6.4f}  {rec['gate'][i]:4.2f}  "
              f"{rec['o2far'][i]:6.4f}  {rec['mass'][i]/FP_ONE:8.1f}{tag}")
    print()


def _print_wind_sweep(rows):
    print("=" * 76)
    print("WIND SWEEP  (W=0 is NATURAL; W>0 is FORCED constant wind)  -> W->m/s->burnout")
    print("  drift(m/s) is the calibration; subtract the W=0 (natural) baseline.")
    print("-" * 76)
    print("  W(dq) forced  t-to-peak  peak_I  burnout_s  drift_t/s  drift_m/s  outcome")
    for m in rows:
        bt = f"{m['burnout_time']:8.1f}" if not np.isnan(m['burnout_time']) else "   n/a  "
        tp = f"{m['time_to_peak']:6.1f}" if not np.isnan(m['time_to_peak']) else "  n/a "
        outc = ("BLOWN OUT" if not np.isnan(m["snap_time"]) else
                ("collapse" if m["collapsed"] else "sustains"))
        fr = "yes" if m["forced"] else "NAT"
        print(f"  {m['wind_dq']:5.2f}  {fr:>4s}  {tp}    {m['peak_I']:5.3f}  {bt}   "
              f"{m['drift_tiles']:8.3f}  {m['drift_ms']:8.3f}   {outc}")
    print("=" * 76)


def _print_k_sweep(rows, target_min=2.5):
    print("=" * 76)
    print("(k_grow, k_die) SWEEP -- NATURAL still air, 2:1 ratio preserved")
    print("TARGET: time-to-peak ~2-3 min (peak intensity reached). Burnout is the")
    print("fuel param (tuned next) -- reported as-is.")
    print("-" * 76)
    print("  k_grow  k_die   peak@ (I at MAX)     peak_I  90%peak       burnout             outcome")
    best = None
    for m in rows:
        tp = m["peak_time"]        # time of the ACTUAL peak (argmax) == "peak intensity reached at"
        outc = ("STALL" if m["stalled"] else
                ("collapse" if m["collapsed"] else "sustains"))
        print(f"  {m['k_grow']:6.3f}  {m['k_die']:6.3f}  {_mmss(tp):>18s}  "
              f"{m['peak_I']:5.3f}  {_mmss(m['time_to_peak']):>16s}  "
              f"{_mmss(m['burnout_time']):>18s}  {outc}")
        if not np.isnan(tp) and not m["stalled"]:
            d = abs(tp / 60.0 - target_min)
            if best is None or d < best[0]:
                best = (d, m)
    print("-" * 76)
    if best is not None:
        m = best[1]
        print(f"CLOSEST to ~{target_min:g} min PEAK time: "
              f"k_grow={m['k_grow']:g}, k_die={m['k_die']:g}  "
              f"(peak at {m['peak_time']:.1f}s = {m['peak_time']/60:.2f} min, "
              f"peak I={m['peak_I']:.3f}, burnout {m['burnout_time']/60:.2f} min)")
    else:
        print("No pair produced a non-stalled ramp with a measurable time-to-peak.")
    print("=" * 76)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wind", type=float, default=0.0,
                    help="0 = NATURAL still air (default); !=0 = FORCED constant +x wind (dequantized units)")
    ap.add_argument("--wind-sweep", action="store_true",
                    help="sweep W = 0(natural) + forced values -> W->m/s->burnout map")
    ap.add_argument("--sweep-values", type=float, nargs="+",
                    default=[0.0, 1.0, 3.0, 5.0, 10.0], help="W values for --wind-sweep")
    ap.add_argument("--k-sweep", action="store_true",
                    help="sweep the (k_grow,k_die) pair (natural wind) -> time-to-peak table")
    ap.add_argument("--k-grow", type=float, default=None, help="override [physics.fire].k_grow")
    ap.add_argument("--k-die", type=float, default=None, help="override [physics.fire].k_die")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="override any CFG dial (repeatable). Bare KEY => [physics.fire]; "
                         "or a dotted path like physics.combustion.burn_rate=0.5")
    ap.add_argument("--interior-w", type=int, default=84,
                    help="interior width; default 84 (SPONGE-SAFE, >= 80)")
    ap.add_argument("--interior-h", type=int, default=40,
                    help="interior height; default 40 (SPONGE-SAFE, >= 36)")
    ap.add_argument("--crate-x", type=int, default=12,
                    help="crate column; default 12 = DEEP in the room, >= sponge_width"
                         "+few from every ring (sky exchange keeps it O2-fed)")
    ap.add_argument("--crate-y", type=int, default=-1, help="default = interior vertical centre")
    ap.add_argument("--tile-size-m", type=float, default=0.333)
    ap.add_argument("--max-seconds", type=float, default=90.0)
    ap.add_argument("--tail-seconds", type=float, default=3.0)
    ap.add_argument("--sky-tau-s", type=float, default=60.0,
                    help="[ambient] sky_tau_s: sky-exchange vertical-mixing timescale "
                         "in seconds (0 == DORMANT/no refill). Default 60.")
    ap.add_argument("--sponge-width", type=int, default=8,
                    help="[ambient] sponge_width: boundary absorber band width in tiles. "
                         "Default 8 (DEFAULT_SPONGE_WIDTH).")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="dump the full per-tick time-series to CSV (single-run modes)")
    args = ap.parse_args(argv)

    # assemble base CLI overrides (--k-grow/--k-die/--set)
    overrides = {}
    if args.k_grow is not None:
        overrides["physics.fire.k_grow"] = args.k_grow
    if args.k_die is not None:
        overrides["physics.fire.k_die"] = args.k_die
    for s in args.sets:
        if "=" not in s:
            ap.error(f"--set expects KEY=VALUE, got {s!r}")
        key, val = s.split("=", 1)
        overrides[key.strip()] = val.strip()

    crate_y = args.crate_y if args.crate_y >= 0 else (args.interior_h // 2 + 1)
    crate_xy = (args.crate_x, crate_y)
    common = dict(interior_w=args.interior_w, interior_h=args.interior_h,
                  crate_xy=crate_xy, tile_size_m=args.tile_size_m,
                  max_seconds=args.max_seconds, tail_seconds=args.tail_seconds,
                  sky_tau_s=args.sky_tau_s, sponge_width=args.sponge_width)

    if args.k_sweep:
        # Erik's 2:1 pairs (§3). Natural wind. Longer window: slow ramps + ~tens-of-s burnout.
        pairs = [(4.0, 2.0), (0.10, 0.05), (0.08, 0.04), (0.05, 0.025), (0.03, 0.015)]
        ks_common = dict(common)
        ks_common["max_seconds"] = max(common["max_seconds"], 360.0)
        rows = []
        for kg, kd in pairs:
            ov = dict(overrides)
            ov["physics.fire.k_grow"] = kg
            ov["physics.fire.k_die"] = kd
            rows.append(run_one(0.0, overrides=ov, verbose=True, **ks_common))
        _print_k_sweep(rows)
    elif args.wind_sweep:
        rows = [run_one(wdq, overrides=overrides, verbose=True, **common)
                for wdq in args.sweep_values]
        _print_wind_sweep(rows)
    else:
        m = run_one(args.wind, overrides=overrides, verbose=True, **common)
        if args.csv:
            write_timeseries_csv(m, args.csv)
            print(f"[csv] wrote {args.csv}  ({m['n_ticks']} ticks)")


if __name__ == "__main__":
    main()
