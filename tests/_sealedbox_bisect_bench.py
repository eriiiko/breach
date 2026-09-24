"""Issue #54 bisection bench — the sealed-box probe, one solver term per run.

The all-systems scenario (2026-08-27) found the cleanest #54 repro yet: a
glass box born sealed at t=0 via ``seal_tiles`` (no doors, no history, no
interior heat source) heats +124 game-deg and self-pressurizes 1.0->1.52 atm
in 18 s from a crate fire ~20 tiles away, while the arena around it COOLS.
P/T ratio ~ constant-N heating: energy, not mass, crosses the sealed wall.

This bench reduces that to the minimal deterministic probe (fire + sealed
dry box, no water/blasts/breach) and re-runs it with ONE energy-chain term
disabled per pass — the bisection the #54 session plan prescribes:

    baseline        as configured
    drag_heat       k_drag_heat_frac = 0
    drag            k_drag = 0 (whole staged momentum drag off)
    comp_work       adiabatic_index = 1.0 (compression work off)
    flat_gs         use_multigrid = False (flat RB-GS — MG wall suspect)
    no_vrail        U_MAX = 1e9 (v2.4 store-clamp rail effectively off)

All fields are live ``def_readwrite`` members of the C++ EOSSolver
(bindings.cpp), set on ``sim.physics_runner.eos`` post-construction, fresh
Simulation per variant. FIXED behavior = box dT ~ 0 while only the crate's
neighbourhood warms. The toggle that kills the box heating names the
mechanism (or flat_gs indicts the MG wall handling specifically).

HARNESS, not a pytest gate (``_`` prefix): prints the table, exits 0.

Run:
    conda run -n data python tests/_sealedbox_bisect_bench.py
    conda run -n data python tests/_sealedbox_bisect_bench.py --cuda [variant...]

P-G2b: ``--cuda`` (anywhere in argv, stripped before variant-name parsing)
selects the CUDA build + the resident-backend GPU path through the SAME
plumbing ``tools/run_on_cuda.py`` (== ``python main.py --cuda``) uses — the
project's one GPU-launch path (CLAUDE.md) — never a second launch path here.

ISSUE #70 (2026-09-24) — THE HEAT-PATH MEASUREMENT. A fireless sealed box
warms ~6 K in 18 s with conduction on while every books identity closes. This
bench is the instrument for finding out WHY (a measurement, not a fix). Two
flags, stripped from argv like ``--cuda``:

    --trace DIR    record a per-tick trace of every variant run and write
                   DIR/<variant>.npz: energy-weighted region temperatures,
                   the per-PHASE energy brackets of every region (Python pre,
                   EOS, combustion+sky, tail, Python post), an exact replay
                   of the temperature solver's Pass 2 that splits conduction
                   by FACE TYPE (checked cell by cell against the region
                   deltas and against the engine's own counters every tick),
                   the engine's counters themselves, a SECOND-LAW WATCH on
                   the box air (its lowest specific entropy against the t=0
                   minimum; an adiabatic box can only raise it) and whole-grid
                   snapshots at fixed ticks. Prints the t=0 check and a
                   cumulative attribution table. tools/plot_sealedbox_70.py
                   draws the figures from these files.
    --seconds S    run S seconds instead of 18.

The #70 variants run ONLY when named, so the default run is unchanged:

    nofire_nodrag         k_drag = k_drag2 = 0
    nofire_noleak         k_leak = 0 (the sweep's out-of-plane leak)
    nofire_ambuniform     per-cell ambient forced uniform (vac_level < 0)
    nofire_norad          both extinction planes zeroed (the sweep is inert)
    nofire_nopush         the seal's AIR PUSH undone: every open cell's gas
                          planes restored to their pre-seal values and its
                          energy reseeded (reseed_gas_energy, the sanctioned
                          scenario-builder seam) -- the box is born at uniform
                          N and T, and the ring's own air leaves the map
    nofire_push_box       the push kept on the BOX side only
    nofire_push_arena     the push kept on the ARENA side only
    nofire_nopush_nocond  the no-push control with conduction off
    wall2_nofire / wall3_nofire   the thick-wall probe without the fire
    nofire_mg6            mg_cycles = 6 (does the MG sealed-pocket lag matter?)

Run (from the repo root):
    python tests/_sealedbox_bisect_bench.py --trace C:/tmp/breach_70 nofire
    python tests/_sealedbox_bisect_bench.py --seconds 180 --trace DIR nofire
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def _pop_flag_value(flag):
    """#70: strip ``flag VALUE`` from argv (before variant-name parsing, like
    --cuda) and return VALUE, or None when the flag is absent."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        raise SystemExit(f"{flag} needs a value")
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


_TRACE_DIR = _pop_flag_value("--trace")        # #70
_SECONDS = _pop_flag_value("--seconds")        # #70

# P-G2b: --cuda must be resolved BEFORE `import breach_physics` — whichever
# build lands in sys.modules first wins for the rest of this process, exactly
# the constraint tools/run_on_cuda.py's own docstring states.
_USE_CUDA = "--cuda" in sys.argv
if _USE_CUDA:
    sys.argv.remove("--cuda")
    sys.path.insert(0, str(ROOT / "tools"))
    from run_on_cuda import enable_all_backends, setup_cuda_import  # noqa: E402
    setup_cuda_import()
    if str(ROOT / "tests") not in sys.path:
        sys.path.insert(0, str(ROOT / "tests"))
else:
    for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import materials  # noqa: E402
from simulation.payloads import ignite_ring  # noqa: E402
from simulation.gas_fixed import FP_ONE_F as _Q, FP_SHIFT as _SH  # noqa: E402  (#70)

if _USE_CUDA:
    enable_all_backends(bp)

TPS = 24
END_TICK = 18 * TPS
IGNITE_TICK = 2 * TPS
CRATE = (26, 41)                 # mid-arena crate stack — the only heat source
AQ_BOX = (50, 58, 24, 32)        # sealed box, built on open arena floor
AQ_IN = np.s_[51:58, 25:32]
BUNKER = np.s_[27:42, 83:96]     # #54 bench R6 (steel, doored)
PEN = np.s_[49:66, 83:96]        # #54 bench R8 (glass, sealed)
ARENA = np.s_[3:67, 3:58]

VARIANTS = [
    ("baseline",  {}),
    ("drag",      {"k_drag": 0.0}),
    ("flat_gs",   {"use_multigrid": False}),
    ("no_vrail",  {"U_MAX": 1e9}),
    # MG thin-wall probe (2026-08-29): same box, 2- and 3-tile glass walls.
    ("wall2",       {}, 2),
    ("wall3",       {}, 3),
    # 2026-08-29: does the sealed box drift with NO forcing at all? If P
    # still climbs, the driver is the box's own equation, not the fire.
    ("nofire",        {}, 1, False),
    # arc #54 P-G1a CONTROL: the same run with the TemperatureSolver's
    # conduction pass switched off (every face -> NO_FACE). P-G1a moves the
    # EOS's energy chain onto the conservative flux form but leaves the
    # thermal solver on the T side (design §2.7 row 3 = P-G1b), so this
    # variant isolates THIS PATCH's contribution to the sealed box. It is the
    # gate's honest denominator, not a setting anyone plays with.
    ("nofire_nocond", {}, 1, False, True),
]
# RETIRED VARIANTS (gas-energy conservation arc #54, P-G1a). Every variant
# below keyed a dial this arc deleted, so they are gone rather than silently
# no-oping (a bisection bench whose knob does nothing is worse than no bench):
#   drag_heat        k_drag_heat_frac = 0   -> D5: the deposit constant is
#                                             DERIVED (k_ke) now, not dialled
#   comp_clamp0      T_WORK_CLAMP = 0       -> D11: there is no step-4c work
#   stiff_K          + adiabatic_index      -> term left to clamp; the whole
#   flat_S128_clamp0                           point of `_clamp0` was "run
#   flat_S512_clamp0                           with #54's driver off", which
#   wall2_clamp0                               is now the ONLY way it runs
#   nofire_clamp0
# `comp_work` (adiabatic_index = 1.0) survived P-G1a with a changed meaning (it
# zeroed k_work = (gamma-1)*T_AMB_K, the flux constant) but is RETIRED too
# (2026-09-23, T7): the later bind-time range guard on `k_flux_q`
# (gas_energy_conservation_design_2026-08-29.md §2.4) rejects gamma = 1, so the
# variant raised before its first tick.

# ISSUE #70 variants — one heat path knocked out per run, all fireless. Run
# ONLY when named on the command line (main() below), so the default bisection
# run above is unchanged. A trailing dict carries run_variant's keyword knobs.
VARIANTS_70 = [
    ("nofire_nodrag",        {"k_drag": 0.0, "k_drag2": 0.0}, 1, False),
    ("nofire_noleak",        {}, 1, False, False,
     {"runner_overrides": {"k_leak_q": 0}}),
    # vac_level < 0 -> derive_ambient returns e_table[0] on EVERY cell (the
    # uniform plane). The shipped 295 K already bakes to exactly E°[0], so this
    # is expected to be the identity; the run proves it rather than assumes it.
    ("nofire_ambuniform",    {}, 1, False, False,
     {"runner_overrides": {"rad_amb_vacuum_q": -1}}),
    ("nofire_norad",         {}, 1, False, False, {"radiation_off": True}),
    ("nofire_nopush",        {}, 1, False, False, {"seal_push": "none"}),
    ("nofire_push_box",      {}, 1, False, False, {"seal_push": "box"}),
    ("nofire_push_arena",    {}, 1, False, False, {"seal_push": "arena"}),
    ("nofire_nopush_nocond", {}, 1, False, True, {"seal_push": "none"}),
    ("wall2_nofire",         {}, 2, False),
    ("wall3_nofire",         {}, 3, False),
    ("nofire_mg6",           {"mg_cycles": 6}, 1, False),
]


def _spec_args(spec):
    """(name, overrides, positional tail, keyword dict) of a variant spec —
    the positional tail keeps the historical tuple convention, a trailing dict
    (issue #70) carries keyword knobs."""
    rest = list(spec[2:])
    kwargs = rest.pop() if rest and isinstance(rest[-1], dict) else {}
    return spec[0], spec[1], rest, kwargs


# ===========================================================================
# ISSUE #70 — the heat-path trace (see the module docstring).
#
# Everything below is bench-side instrumentation. Nothing in the engine is
# changed: the phase brackets come from a forwarding proxy the bench installs
# on `sim.physics_runner.engine` (the runner calls exactly `run_substeps` and
# `step_tail` on it per tick, physics_runner.py), and the face-type split comes
# from an integer REPLAY of TemperatureSolver Pass 2 (temperature_solver.h's
# conduction kit, transcribed), which is only trusted because every tick it is
# checked CELL BY CELL against the region deltas the engine actually produced
# and against the engine's own counters (e_gas_cond_sum, e_solid_cond_sum,
# e_cond_trunc_sum). A mismatch is counted and printed, never absorbed. The
# engine has no per-face-type counter, which is the only reason a replay
# exists at all.
# ===========================================================================
_R70_LABELS = ("other", "box_air", "box_wall", "halo_air", "arena_air",
               "arena_wall", "pen_air")
_R70_GAS = ("box_air", "box_layer", "box_core", "halo_air", "arena_air",
            "pen_air", "all_gas")
_R70_SOLID = ("box_wall", "arena_wall", "all_solid")
_R70_PHASES = ("py_pre", "eos", "comb_sky", "tail", "py_post")
NECK = (54, 3)   # the SW room's one-tile doorway (a Helmholtz neck, #70 item 6)


def _nb4(mask):
    """Cells 4-adjacent to any cell of `mask`."""
    out = np.zeros_like(mask)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _t0_lines(g, tag, masks=None):
    """#70 item 2: is every cell at exactly the temperature and energy it
    should be? Gas: the mirror must be floor(E/N) - T_amb, i.e. the remainder
    E - N*(T + T_amb) lies in [0, N). Solids: T is their own truth."""
    acct = g._gas_energy_accountable()
    ts = g.thermal_solid
    t_amb = int(g._gas_energy_t_amb_raw())
    n = g._gas_bulk_n_raw()
    T = g.temperature.astype(np.int64)
    E = g.gas_energy.astype(np.int64)
    out = [f"  [t0 check: {tag}]"]
    for label, mk in (("thermal solids", ts), ("accountable gas", acct)):
        v = T[mk]
        out.append(f"    {label:15s} {int(mk.sum()):5d} cells  T raw "
                   f"[{int(v.min()):+d}, {int(v.max()):+d}]  nonzero "
                   f"{int(np.count_nonzero(v)):5d}  "
                   f"([{v.min() / _Q:+.5f}, {v.max() / _Q:+.5f}] K)")
    r = (E - n * (T + t_amb))[acct]
    bad = int(np.count_nonzero((r < 0) | (r >= n[acct])))
    out.append(f"    gas E - N*(T+T_amb): [{int(r.min())}, {int(r.max())}]  "
               f"cells outside [0, N): {bad}   E off the accountable set: "
               f"{int(np.count_nonzero(E[~acct]))} nonzero")
    if masks is not None:
        for k in ("box_layer", "box_core", "halo_air", "arena_air", "pen_air"):
            mk = masks[k]
            nk = n[mk] / _Q
            tk = T[mk] / _Q
            out.append(f"    {k:10s} {int(mk.sum()):4d} cells  N [{nk.min():.3f}, "
                       f"{nk.max():.3f}] atm  T [{tk.min():+8.3f}, "
                       f"{tk.max():+8.3f}] K")
        for k in ("box_wall", "arena_wall"):
            tk = T[masks[k]] / _Q
            out.append(f"    {k:10s} {int(masks[k].sum()):4d} cells  T "
                       f"[{tk.min():+.6f}, {tk.max():+.6f}] K")
    return out


class _EnginePhaseProxy:
    """#70: forwards every attribute to the real PhysicsEngine and brackets
    the per-tick phase calls with the trace's hook. Installed on the RUNNER
    only (`gmap` keeps its own reference to the real engine); the engine
    itself is untouched."""

    def __init__(self, engine, hook):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_hook", hook)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_engine"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_engine"), name, value)

    def _bracket(self, pre, post, fn, *a, **k):
        self._hook(pre)
        r = fn(*a, **k)
        self._hook(post)
        return r

    def run_substeps(self, *a, **k):
        return self._bracket("eos_pre", "eos_post",
                             self._engine.run_substeps, *a, **k)

    def run_substeps_resident(self, *a, **k):
        return self._bracket("eos_pre", "eos_post",
                             self._engine.run_substeps_resident, *a, **k)

    def step_tail(self, *a, **k):
        return self._bracket("tail_pre", "tail_post",
                             self._engine.step_tail, *a, **k)


class _Trace70:
    """Issue #70: one variant's per-tick heat-path trace (module docstring).

    Currencies. Gas books count `N_raw*T_raw` (the arc #54 field). Solid books
    and every conduction face quantum count HEAT, `cap_raw*T_raw` with cap in
    Q16.16 thermal-mass units. One heat count is `THERMAL_MASS_UNIT*V_tile /
    2^32` J; one gas book count is `c_v` of that (T5a, T2 §1.2 -- the two
    ledgers share a unit only at c_v == 1). Everything this trace reports in
    joules goes through exactly those two factors.
    """

    def __init__(self, sim, box_in_mask, box_wall_mask, box_footprint):
        from config import CFG
        from simulation import gas_fixed
        from simulation.materials import THERMAL_MASS_UNIT
        g = sim.gmap
        self.g = g
        self.eos = sim.physics_runner.eos
        self.tsol = sim.physics_runner.engine.temperature
        self.t_amb = int(g._gas_energy_t_amb_raw())
        self.no_face = int(self.tsol.no_face)
        # The SAME quantizations TemperatureSolver::step makes of its float
        # dials (fixed_point.h quantize == gas_fixed.quantize_scalar: round
        # half away from zero; the pybind float reads back the float32 value).
        self.c_v = float(self.tsol.c_v)
        self.c_v_q = gas_fixed.quantize_scalar(self.c_v if self.c_v > 0 else 1.0)
        self.n_floor_q = gas_fixed.quantize_scalar(float(self.tsol.n_floor_heat))
        self.t_max_q = gas_fixed.quantize_scalar(float(self.tsol.T_MAX_PHYS))
        v_tile = float(g.tile_size_m) ** 2 * float(CFG.physics.water.ceiling_h)
        self.J_heat = THERMAL_MASS_UNIT * v_tile / _Q ** 2
        self.J_books = self.c_v * self.J_heat
        gam = float(self.eos.adiabatic_index)
        self.k_ke = gam * (gam - 1.0) * float(self.eos.T_AMB_K) / (
            2.0 * float(self.eos.c_max) ** 2)          # K per (m/s)^2

        acct = g._gas_energy_accountable()
        ts = g.thermal_solid.copy()
        wall_nb = _nb4(box_wall_mask)
        arena = np.zeros_like(acct)
        arena[ARENA] = True
        pen = np.zeros_like(acct)
        pen[PEN] = True
        m = {"box_air": acct & box_in_mask}
        m["box_layer"] = m["box_air"] & wall_nb          # air touching a box wall
        m["box_core"] = m["box_air"] & ~wall_nb
        m["halo_air"] = acct & ~box_footprint & wall_nb  # outside air on the walls
        m["arena_air"] = acct & arena & ~box_footprint & ~wall_nb
        m["pen_air"] = acct & pen                        # load-time sealed control
        m["all_gas"] = acct.copy()
        m["box_wall"] = box_wall_mask & ts
        m["arena_wall"] = ts & arena & ~box_footprint
        m["all_solid"] = ts.copy()
        self.masks = m
        self.acct0, self.ts0 = acct.copy(), ts
        lab = np.zeros(acct.shape, np.int64)
        for k, name in enumerate(_R70_LABELS):
            if name != "other":
                lab[m[name]] = k
        self.lab = lab
        his = g.heat_inv_shift.astype(np.int64)
        s = np.maximum(his, -16)                         # conduction::CAP_SHIFT_MIN
        self.his = his
        self.cap_used_ts = np.left_shift(np.int64(1), np.minimum(s, 12) + _SH)
        self.cap_real_ts = np.left_shift(np.int64(1), np.minimum(s, 30) + _SH)

        self.rows = []
        self.bad = {"ticks": 0, "gas_cells": 0, "solid_cells": 0,
                    "counter_ticks": 0, "heat_ticks": 0, "clamp_ticks": 0,
                    "no_bracket_ticks": 0}
        self._ph = {}
        self._attr = {}
        self._start = self._sums()
        self.first = self._start
        self.t0_lines = _t0_lines(g, "t = 0 (after construction)", m)
        self.t1_lines = []
        self.snaps = {}              # tick -> whole-grid fields, for the maps
        self._snap(0)
        # SECOND-LAW WATCH (#70): the box air's specific entropy per cell,
        # s = ln(T_abs) - (gamma-1) ln(N) (ideal gas, per unit c_v). With the
        # box adiabatic, mixing and dissipation only RAISE it, and conduction
        # from warmer walls raises it too -- so no cell may fall below the
        # smallest value present at t = 0. `gam` is the EOS's own index.
        self.gam = gam
        self.s_min0 = float(self._box_entropy()[0].min())

    def _box_entropy(self):
        g = self.g
        mb = self.masks["box_air"]
        n = g._gas_bulk_n_raw()[mb].astype(np.float64)
        t_abs = g.gas_energy[mb].astype(np.float64) / n / _Q
        n_atm = n / _Q
        return np.log(t_abs) - (self.gam - 1.0) * np.log(n_atm), t_abs, n_atm

    SNAP_TICKS = (1, 2, 3, 6, 12, 24, 48, 96, 240, 432, 1440, 4320)

    def _snap(self, t):
        g = self.g
        self.snaps[t] = {"T": g.temperature.copy(), "N": g._gas_bulk_n_raw(),
                         "wx": g.wind_x.copy(), "wy": g.wind_y.copy(),
                         "P": g.atmosphere.copy()}

    # ---- region sums -----------------------------------------------------
    def _sums(self):
        g = self.g
        n = g._gas_bulk_n_raw()
        e_rel = g.gas_energy.astype(np.int64) - n * self.t_amb
        H = self.cap_real_ts * g.temperature.astype(np.int64)
        out = {}
        for k in _R70_GAS:
            mk = self.masks[k]
            out[k] = (int(e_rel[mk].sum()), int(n[mk].sum()))
        for k in _R70_SOLID:
            mk = self.masks[k]
            out[k] = (int(H[mk].sum()), int(self.cap_real_ts[mk].sum()))
        return out

    def _counters(self):
        e, t = self.eos, self.tsol
        return {
            # EOS: RESET at step() entry, so a post-tick read IS that tick's
            "eos_transport": int(e.e_transport_net_sum),
            "eos_kick_ke": int(e.e_kick_ke_sum),
            "eos_drag_heat": int(e.e_drag_heat_sum),
            "eos_work_export": int(e.e_work_export_sum),
            "eos_rail": int(e.e_rail_sum),
            "eos_wipe": int(e.e_wipe_sum),
            "eos_absorb_export": int(e.e_absorb_export_sum),
            "eos_sponge_export": int(e.e_sponge_export_sum),
            "eos_clamp_destroyed": int(e.e_clamp_destroyed_sum),
            "eos_energy_floor": int(e.e_energy_floor_sum),
            # thermal solver: ACCUMULATING (differenced per tick at save)
            "ts_gas_cond": int(t.e_gas_cond_sum),
            "ts_solid_cond": int(t.e_solid_cond_sum),
            "ts_cond_trunc": int(t.e_cond_trunc_sum),
            "ts_cond_cap": int(t.e_cond_cap_sum),
            "ts_gas_deposit": int(t.e_gas_deposit_sum),
            "ts_gas_rail": int(t.e_gas_rail_sum),
            "ts_solid_deposit": int(t.e_solid_deposit_sum),
            "ts_low_rail_hits": int(t.t_low_rail_hits),
            "ts_rad_clamp_hits": int(t.rad_clamp_hits),
            "ts_t_max_hits": int(t.t_max_phys_hits),
        }

    def _caps(self, n):
        """conduction::cell_capacity_q, vectorized (cap_used, cap_real)."""
        nr = np.maximum(n, 0)
        nu = np.maximum(nr, self.n_floor_q)
        ceiling = np.int64(1) << (12 + _SH)              # CAP_SHIFT_MAX
        cu = np.clip((nu * self.c_v_q) >> _SH, 1, ceiling)
        cr = np.clip((nr * self.c_v_q) >> _SH, 0, ceiling)
        return (np.where(self.ts0, self.cap_used_ts, cu),
                np.where(self.ts0, self.cap_real_ts, cr))

    # ---- the phase hook ----------------------------------------------------
    def hook(self, tag):
        if tag == "tail_pre":
            g = self.g
            self._T_pre = g.temperature.astype(np.int64)
            self._N_pre = g._gas_bulk_n_raw()
            self._E_pre = g.gas_energy.astype(np.int64)
            self._c_pre = self._counters()
        self._ph[tag] = self._sums()
        if tag == "tail_post":
            self._replay_tail()

    def _replay_tail(self):
        """TemperatureSolver::step Passes 0-2, replayed in int64 from the
        tail's own inputs, then checked cell by cell against what the engine
        wrote. Returns the per-face-type split in `self._attr`."""
        g = self.g
        ts, acct = self.ts0, self.acct0
        T0, n = self._T_pre, self._N_pre
        c_pre, c_post = self._c_pre, self._counters()
        # Pass 0: non-thermal-solid vacuum / ring cells pinned to 0.
        T1 = T0.copy()
        T1[(g.is_vacuum | g.is_ambient) & ~ts] = 0
        # Pass 1: the radiation fold on thermal solids. `rad_net_sweep` is the
        # plane step 2b wrote in THIS tail (run() overwrites it every tick).
        rn = g.rad_net_sweep.astype(np.int64)
        fm = ts & (rn != 0)
        fold_H = np.zeros_like(T0)
        if fm.any():
            his, x = self.his[fm], rn[fm]
            sh = np.maximum(his, 0)
            dtr = np.where(his >= 0,
                           np.where(x < 0, -((-x) >> sh), x >> sh),
                           x * (np.int64(1) << np.minimum(-his, 62)))
            t = np.clip(T1[fm] + dtr, -2 ** 31, 2 ** 31 - 1)
            t = np.maximum(np.minimum(t, self.t_max_q), 0)   # both rails
            fold_H[fm] = (t - T0[fm]) * self.cap_real_ts[fm]
            T1[fm] = t
        # The maximum-principle clamp needs e_inv_q, which is not replayed:
        # a tick where it engaged is flagged instead. Likewise a heat deposit.
        if c_post["ts_rad_clamp_hits"] != c_pre["ts_rad_clamp_hits"]:
            self.bad["clamp_ticks"] += 1
        if np.any(g.heat != 0):
            self.bad["heat_ticks"] += 1
        # Pass 2: the conduction kit (temperature_solver.h), face by face.
        cap_used, cap_real = self._caps(n)
        no_cond = ~ts & (g.is_vacuum | (cap_real == 0))
        de = np.zeros_like(T0)
        L = len(_R70_LABELS)
        flow = np.zeros((L, L), dtype=np.int64)   # flow[a, b]: heat INTO a FROM b
        fs = g.face_shift.astype(np.int64)
        for si, sj, a_, b_ in (
                (fs[:, :-1, 2], fs[:, 1:, 3], np.s_[:, :-1], np.s_[:, 1:]),
                (fs[:-1, :, 1], fs[1:, :, 0], np.s_[:-1, :], np.s_[1:, :])):
            valid = ((si != self.no_face) & (sj != self.no_face)
                     & ~no_cond[a_] & ~no_cond[b_])
            s = np.where(valid, np.maximum(si, sj), 0)
            d = T1[b_] - T1[a_]
            full = np.abs(d) * np.minimum(cap_used[a_], cap_used[b_])
            q = np.minimum(full >> s, full >> 1)      # constraint 4's limiter
            q = np.where(valid, np.where(d < 0, -q, q), 0)   # heat INTO cell a
            de[a_] += q
            de[b_] -= q
            la, lb = self.lab[a_].ravel(), self.lab[b_].ravel()
            np.add.at(flow, (la, lb), q.ravel())
            np.add.at(flow, (lb, la), -q.ravel())
        nz = de != 0
        # accountable gas: no endpoint divide; the face sum is converted into
        # the books through the SAME cap it was priced at (muldiv_floor_q).
        ga = acct & nz
        de_books = np.zeros_like(de)
        if ga.any():
            a, b, dd = de[ga], n[ga], cap_used[ga]
            q_ = np.floor_divide(a, dd)
            de_books[ga] = q_ * b + np.floor_divide((a - q_ * dd) * b, dd)
        # everything else that conducts (thermal solids here): T-form divide.
        oth = ~acct & nz & ~no_cond
        dT = np.zeros_like(de)
        dT[oth] = np.floor_divide(de[oth], cap_used[oth])
        trunc = np.zeros_like(de)
        trunc[oth] = dT[oth] * cap_used[oth] - de[oth]
        cond_H = np.zeros_like(de)
        cond_H[oth & ts] = dT[oth & ts] * cap_real[oth & ts]
        # ---- verification: cell by cell, then the engine's own counters ----
        dE = g.gas_energy.astype(np.int64) - self._E_pre
        dH = (g.temperature.astype(np.int64) - T0) * self.cap_real_ts
        self.bad["gas_cells"] += int(np.count_nonzero(acct & (dE != de_books)))
        self.bad["solid_cells"] += int(np.count_nonzero(
            ts & (dH != fold_H + cond_H)))
        if ((c_post["ts_gas_cond"] - c_pre["ts_gas_cond"]) != int(de_books[acct].sum())
                or (c_post["ts_solid_cond"] - c_pre["ts_solid_cond"]) != int(cond_H[ts].sum())
                or (c_post["ts_cond_trunc"] - c_pre["ts_cond_trunc"]) != int(trunc.sum())):
            self.bad["counter_ticks"] += 1
        self.bad["ticks"] += 1
        ix = {name: k for k, name in enumerate(_R70_LABELS)}
        mb, mw, mh = (self.masks["box_air"], self.masks["box_wall"],
                      self.masks["halo_air"])
        self._attr = {
            # HEAT counts crossing each face family this tick (signed, INTO the
            # first-named region). Exact: the replay reproduces the engine.
            "q_wall_to_box": int(flow[ix["box_air"], ix["box_wall"]]),
            "q_wall_to_halo": int(flow[ix["halo_air"], ix["box_wall"]]),
            "q_box_internal": int(flow[ix["box_air"], ix["box_air"]]),
            "q_wall_internal": int(flow[ix["box_wall"], ix["box_wall"]]),
            "q_wall_from_rest": int(flow[ix["box_wall"]].sum()
                                    - flow[ix["box_wall"], ix["box_air"]]
                                    - flow[ix["box_wall"], ix["halo_air"]]
                                    - flow[ix["box_wall"], ix["box_wall"]]),
            "q_halo_from_arena": int(flow[ix["halo_air"], ix["arena_air"]]),
            # what the engine booked from them (books for gas, heat for walls)
            "box_books_cond": int(de_books[mb].sum()),
            "box_heat_cond": int(de[mb].sum()),
            "halo_books_cond": int(de_books[mh].sum()),
            "halo_heat_cond": int(de[mh].sum()),
            "wall_trunc": int(trunc[mw].sum()),       # destroyed by floordiv
            "wall_fold": int(fold_H[mw].sum()),       # the radiation fold
            "wall_cond_H": int(cond_H[mw].sum()),
            "all_trunc": int(trunc.sum()),
            "all_fold": int(fold_H.sum()),
            "fold_cells": int(np.count_nonzero(fm)),
        }

    # ---- per tick ----------------------------------------------------------
    def end_tick(self, t):
        g = self.g
        now = self._sums()
        ph = self._ph
        if not all(k in ph for k in ("eos_pre", "eos_post", "tail_pre", "tail_post")):
            self.bad["no_bracket_ticks"] += 1
            ph = {k: now for k in ("eos_pre", "eos_post", "tail_pre", "tail_post")}
        pts = (self._start, ph["eos_pre"], ph["eos_post"], ph["tail_pre"],
               ph["tail_post"], now)
        row = {"t": t}
        for k in _R70_GAS + _R70_SOLID:
            row[k + "_E"], row[k + "_C"] = now[k]
            for p, a, b in zip(_R70_PHASES, pts[:-1], pts[1:]):
                row[f"{k}_d_{p}"] = b[k][0] - a[k][0]
        row.update(self._attr)
        row.update({"c_" + k: v for k, v in self._counters().items()})
        # diagnostics: motion, pressure vs the EOS's own p* = N*T_abs/T_amb
        u = np.hypot(g.wind_x.astype(np.float64), g.wind_y.astype(np.float64)) / _Q
        n = g._gas_bulk_n_raw().astype(np.float64)
        T = g.temperature.astype(np.float64)
        P = g.atmosphere.astype(np.float64) / _Q
        for k in ("box_air", "box_layer", "box_core", "halo_air", "pen_air"):
            mk = self.masks[k]
            row[k + "_umax"] = float(u[mk].max())
            row[k + "_P"] = float(P[mk].mean())
            row[k + "_pstar"] = float((n[mk] / _Q * (T[mk] + self.t_amb)
                                       / self.t_amb).mean())
            row[k + "_Tmin"] = float(T[mk].min() / _Q)
            row[k + "_Tmax"] = float(T[mk].max() / _Q)
        mb = self.masks["box_air"]
        row["box_ke_books"] = float((n[mb] * self.k_ke * u[mb] ** 2 * _Q).sum())
        k = int(np.argmax(u))
        row["map_umax"], row["map_umax_idx"] = float(u.flat[k]), k
        row["neck_u"] = float(u[NECK])
        row["rad_net_absmax"] = int(np.abs(g.rad_net_sweep).max())
        s_box, t_abs, n_atm = self._box_entropy()
        row["box_s_min_excess"] = float(s_box.min() - self.s_min0)
        # the energy by which box air sits BELOW the most permissive physical
        # floor (the t=0 minimum-entropy isentrope at each cell's own N), box-K
        t_floor = np.exp(self.s_min0 + (self.gam - 1.0) * np.log(n_atm))
        row["box_subisentrope_K"] = float(
            (n_atm * np.maximum(t_floor - t_abs, 0.0)).sum() / n_atm.sum())
        self.rows.append(row)
        if t in self.SNAP_TICKS:
            self._snap(t)
        if t == 1:
            self.t1_lines = _t0_lines(g, "after tick 1", self.masks)
        self._start = now
        self._ph = {}
        self._attr = {}

    # ---- output ------------------------------------------------------------
    def series(self):
        keys = self.rows[0].keys()
        out = {k: np.array([r.get(k, 0) for r in self.rows]) for k in keys}
        # the accumulating thermal-solver counters -> per-tick deltas
        for k in list(out):
            if k.startswith("c_ts_"):
                v = out[k].astype(np.int64)
                out[k] = np.diff(np.concatenate(([v[0]], v)))
        return out

    def report(self, name, tps):
        s = self.series()
        J = {"heat": self.J_heat, "books": self.J_books}
        n_box = self.first["box_air"][1]
        box_JK = n_box * _Q * self.J_books        # J per K of the box air

        def cum(key, cur):
            return float(np.sum(s[key].astype(np.float64))) * J[cur]

        dE_box = (self.rows[-1]["box_air_E"] - self.first["box_air"][0]) * self.J_books
        dH_wall = (self.rows[-1]["box_wall_E"] - self.first["box_wall"][0]) * self.J_heat
        dE_halo = (self.rows[-1]["halo_air_E"] - self.first["halo_air"][0]) * self.J_books
        lines = [f"  [#70 attribution: {name}, {len(self.rows) / tps:.0f} s, "
                 f"box air {n_box / _Q:.2f} atm-tiles = {box_JK / 1e3:.2f} kJ/K]"]

        def row(label, joules):
            lines.append(f"    {label:52s} {joules / 1e3:+10.3f} kJ  "
                         f"({joules / box_JK:+8.3f} box-K)")
        row("BOX AIR  measured dE (Sum E_rel, books x c_v)", dE_box)
        for p in _R70_PHASES:
            row(f"  phase {p}", cum(f"box_air_d_{p}", "books"))
        row("  conduction heat in through wall faces (replay)", cum("q_wall_to_box", "heat"))
        row("  air<->air faces inside the box (cancel by antisymmetry)",
            cum("q_box_internal", "heat"))
        row("  books booked from that heat, x c_v", cum("box_books_cond", "books"))
        row("BOX WALLS measured dH (Sum cap*T)", dH_wall)
        row("  heat out to box air (inner faces)", -cum("q_wall_to_box", "heat"))
        row("  heat out to halo air (outer faces)", -cum("q_wall_to_halo", "heat"))
        row("  wall<->wall faces inside the ring", cum("q_wall_internal", "heat"))
        row("  faces to anything else", cum("q_wall_from_rest", "heat"))
        row("  destroyed by the endpoint floordiv (trunc)", cum("wall_trunc", "heat"))
        row("  radiation fold", cum("wall_fold", "heat"))
        for p in _R70_PHASES:
            row(f"  phase {p}", cum(f"box_wall_d_{p}", "heat"))
        row("HALO AIR measured dE", dE_halo)
        row("  conduction heat in from the box walls", cum("q_wall_to_halo", "heat"))
        for p in _R70_PHASES:
            row(f"  phase {p}", cum(f"halo_air_d_{p}", "books"))
        lines.append("    ENGINE (whole map, cumulative): "
                     f"trunc {cum('c_ts_cond_trunc', 'heat') / 1e3:+.3f} kJ, "
                     f"cap {cum('c_ts_cond_cap', 'books') / 1e3:+.3f} kJ, "
                     f"gas rail {cum('c_ts_gas_rail', 'books') / 1e3:+.3f} kJ, "
                     f"EOS rail {cum('c_eos_rail', 'books') / 1e3:+.3f} kJ, "
                     f"EOS drag heat {cum('c_eos_drag_heat', 'books') / 1e3:+.3f} kJ, "
                     f"EOS kick KE {cum('c_eos_kick_ke', 'books') / 1e3:+.3f} kJ, "
                     f"low-rail hits {int(s['c_ts_low_rail_hits'].sum())}, "
                     f"clamp hits {int(s['c_ts_rad_clamp_hits'].sum())}, "
                     f"max |rad_net| {int(s['rad_net_absmax'].max())}, "
                     f"fold cells {int(s['fold_cells'].sum())}")
        b = self.bad
        lines.append(f"    REPLAY CHECK over {b['ticks']} tails: gas cells off "
                     f"{b['gas_cells']}, solid cells off {b['solid_cells']}, "
                     f"counter ticks off {b['counter_ticks']}, heat-deposit "
                     f"ticks {b['heat_ticks']}, clamp ticks {b['clamp_ticks']}, "
                     f"unbracketed ticks {b['no_bracket_ticks']}"
                     + ("   -> EXACT" if not any(b[k] for k in b if k != "ticks")
                        else "   -> NOT EXACT: attribution unverified"))
        lines.append(f"    box air u_max: tick 1 {s['box_air_umax'][0]:.2f}, max "
                     f"{s['box_air_umax'].max():.2f}, end {s['box_air_umax'][-1]:.2f}"
                     f" m/s;  map u_max end {s['map_umax'][-1]:.2f} m/s at "
                     f"{np.unravel_index(int(s['map_umax_idx'][-1]), self.g.temperature.shape)}"
                     f";  neck {NECK} end {s['neck_u'][-1]:.2f} m/s")
        lines.append(f"    box P/p*: tick 1 {s['box_air_P'][0]:.4f}/"
                     f"{s['box_air_pstar'][0]:.4f}, end {s['box_air_P'][-1]:.4f}/"
                     f"{s['box_air_pstar'][-1]:.4f} atm;  pen u_max max "
                     f"{s['pen_air_umax'].max():.3f} m/s")
        k = int(np.argmin(s["box_s_min_excess"]))
        lines.append(f"    SECOND LAW (box air): min s - min s(t=0) worst "
                     f"{s['box_s_min_excess'][k]:+.4f} c_v at t = {(k + 1) / tps:.2f} s"
                     f"   energy below the t=0 minimum-entropy isentrope: max "
                     f"{s['box_subisentrope_K'].max():.3f} box-K"
                     + ("   -> VIOLATED" if s["box_s_min_excess"][k] < -1e-6 else "   -> ok"))
        return lines

    def save(self, path, meta):
        import json
        s = self.series()
        m = dict(meta)
        m.update({"J_heat": self.J_heat, "J_books": self.J_books,
                  "t_amb_raw": self.t_amb, "c_v": self.c_v, "c_v_q": self.c_v_q,
                  "k_ke": self.k_ke, "gamma": self.gam, "s_min0": self.s_min0,
                  "first": self.first, "bad": self.bad,
                  "t0_lines": self.t0_lines, "t1_lines": self.t1_lines,
                  "masks": {k: int(v.sum()) for k, v in self.masks.items()}})
        self._snap(len(self.rows))          # the final state, whatever its tick
        snaps = {f"snap{t}_{k}": v for t, d in self.snaps.items()
                 for k, v in d.items()}
        m["snap_ticks"] = sorted(self.snaps)
        m["material"] = self.g.material.tolist()
        np.savez_compressed(path, meta=json.dumps(m), **s, **snaps,
                            **{"mask_" + k: v for k, v in self.masks.items()})


def run_variant(name, overrides, wall_thick=1, ignite=True, no_conduction=False,
                *, seal_push="both", runner_overrides=None, radiation_off=False,
                end_tick=None, trace_dir=None):
    """One fresh Simulation; ``wall_thick`` = glass ring thickness in tiles
    (2026-08-29: the MG thin-wall probe — if a coarse cell straddling a
    1-tile wall is the leak, thicker walls should shrink it).

    Issue #70 keyword knobs (module docstring): ``seal_push`` ("both" = what
    seal_tiles does; "none" / "box" / "arena" = undo the evacuated-air push
    everywhere / keep it on the box side only / on the arena side only),
    ``runner_overrides`` (PhysicsRunner attributes, e.g. k_leak_q),
    ``radiation_off`` (zero both extinction planes), ``end_tick`` and
    ``trace_dir`` (write a per-tick trace there)."""
    end_tick = END_TICK if end_tick is None else int(end_tick)
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    lvl = replace(lvl, entities=[e for e in lvl.entities
                                 if e.class_name not in ("vent", "duct")])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    for field, value in overrides.items():
        setattr(sim.physics_runner.eos, field, value)
    for field, value in (runner_overrides or {}).items():   # #70
        setattr(sim.physics_runner, field, value)
    t0_load = _t0_lines(g, "level load, before the seal") if trace_dir else []

    r0, r1, c0, c1 = AQ_BOX
    box_in = np.s_[r0 + wall_thick:r1 + 1 - wall_thick,
                   c0 + wall_thick:c1 + 1 - wall_thick]
    gas_pre_seal = g.gas.copy()          # #70: for the seal_push knob only
    # Seal one layer per call, INNERMOST first: seal_tiles evacuates each
    # tile's gas to an OPEN non-span neighbour and refuses a tile with none
    # (its sealed-pocket guard) — an inner layer's corners only have open
    # neighbours while the layer outside them is still open.
    for k in reversed(range(wall_thick)):
        layer = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if min(r - r0, r1 - r, c - c0, c1 - c) == k]
        g.seal_tiles(layer, materials.MAT_GLASS)
    # #70: seal_tiles moves each sealed tile's air to its open neighbours --
    # half INTO the box, half into the arena, at the tile's own (ambient)
    # temperature -- so the wall-adjacent cells on both sides start at 1.5 atm
    # (2.0 in the box's inner corners). `seal_push` undoes that push on the
    # chosen side(s): each open cell's gas planes go back to their pre-seal
    # values and its stored energy is re-derived from its (unchanged, ambient)
    # mirror through `reseed_gas_energy`, the sanctioned scenario-builder
    # seam. The ring's own air then simply leaves the map.
    if seal_push != "both":
        fp = np.zeros_like(g.solid)
        fp[r0:r1 + 1, c0:c1 + 1] = True
        inside = np.zeros_like(g.solid)
        inside[box_in] = True
        restore = {"none": ~g.solid, "box": ~g.solid & ~fp,
                   "arena": ~g.solid & inside}[seal_push]
        g.gas[:, restore] = gas_pre_seal[:, restore]
        g.reseed_gas_energy(restore)
    if radiation_off:                    # #70: an inert sweep (a = d = 0)
        g.heat_atten_q[:] = 0
        g.dyn_heat_atten_q[:] = 0
    open0 = ~g.solid.copy()
    T0 = g.temperature.astype(np.int64)
    # P-G5: the box's OWN sealed glass ring (footprint minus its accountable
    # interior), for a BOX-SCOPED reading of the thermostat's contribution —
    # the GLOBAL `e_thermostat_sum` (below) is dominated by the crate fire's
    # own immediate walls, nothing to do with this box 20+ tiles away, so it
    # cannot be the (ii) subtraction term; this ring mask is.
    box_footprint = np.zeros_like(g.solid)
    box_footprint[r0:r1 + 1, c0:c1 + 1] = True
    box_in_mask = np.zeros_like(g.solid)
    box_in_mask[box_in] = True
    box_wall_mask = box_footprint & (~box_in_mask) & g.thermal_solid

    # ======================================================================
    # arc #54 SB gate instrumentation (design §6 "SB").
    #
    # (i) THE CLOSURE IDENTITY, exact in int64, measured ACROSS WHOLE TICKS.
    #     P-G1a could only bracket `run_substeps`, because the writers outside
    #     the EOS still wrote `temperature` and the solver's entry re-sync
    #     absorbed them. P-G1b lands every one of those writers on the seam and
    #     DELETES the re-sync (D1 live), so the honest bracket is now the whole
    #     `Simulation.step`: the field's per-tick drift has to equal the sum of
    #     the four counter groups that are allowed to move it —
    #         EOS       (design §2.8's seven terms, reset per step)
    #         tail      the thermal solver's gas side (accumulating)
    #         combustion the two-hop energy ledger (accumulating)
    #         seam      every Python writer's net (GameMap.gas_energy_books)
    #     Restricting the identity to the box would need per-region counters;
    #     the GLOBAL identity being exact is strictly stronger for the flux
    #     term, since the flux contributes exactly 0 to it only by per-face
    #     cancellation -- and the box's own sealed guarantee then follows from
    #     telescoping. The box's Delta E is reported alongside.
    #
    # (ii) DeltaT_box = Delta(Sum E / Sum N) -- N-WEIGHTED (an unweighted mirror
    #     mean is not conserved by mixing). THIS is the arc's headline.
    #
    # (iii) the D4 wall probe, the ts-wall probe, and the Sum N|u|^2 drift (D7).
    # ======================================================================
    eos = sim.physics_runner.eos
    engine = sim.physics_runner.engine
    tsolver = engine.temperature
    comb = sim.physics_runner.combustion
    t_amb_raw = g._gas_energy_t_amb_raw()
    ident = {"worst": 0, "ticks": 0, "bad": 0}

    def _n_plane():
        n = np.zeros(g.temperature.shape, dtype=np.int64)
        for gi in np.flatnonzero(g.gases.conservative):
            n += g.gas[gi].astype(np.int64)
        return n

    def _e_sum(mask):
        """Sum gas_energy over `mask` as a PYTHON int -- design §2.2 forbids
        absolute int64 sums (a 160^2 map at ambient sits near 2^55, but the
        bench must not be the thing that wraps at a blast core)."""
        return int(g.gas_energy[mask].astype(object).sum())

    def _ident_terms():
        """The four counter groups §2.8's identity is allowed to move the field
        with, PLUS (P-G5) the solid side's own three. The EOS and water-evac
        groups RESET every step (so they are read absolutely); the rest
        ACCUMULATE (so they are differenced tick to tick)."""
        return (
            # EOS (design §2.8). `e_entry_resync_sum` is RETIRED at P-G1b and
            # structurally 0; it is kept in the sum so this transcription still
            # names all seven terms.
            int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
            - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
            + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
            + int(eos.e_rail_sum),
            # thermal solver, gas side (temperature_solver.h)
            int(tsolver.e_gas_deposit_sum) + int(tsolver.e_gas_cond_sum)
            + int(tsolver.e_gas_rail_sum),
            # combustion's two-hop ledger, identity (A) (combustion.h)
            -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
            + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
            # every Python seam (GameMap.gas_energy_books, diagnostics excluded)
            int(g.gas_energy_seam_net()),
            # the water-displacement evacuation's export (design 2.7 row 2,
            # R3-#10): `step_water_tail` runs on the host BEFORE the EOS and
            # moves gas_energy with the bulk shares it pushes out of a flooding
            # cell. The move is conservative INSIDE the accountable set, so the
            # only term is what left it. Reset per call, like the EOS group.
            -int(engine.e_water_evac_export_sum),
            # P-G5 (design gas_energy_thermostat_ledger_2026-08-30.md): the
            # SOLID side's own channels — Pass 1's landing on thermal solids,
            # Pass 2's landing on thermal solids, AND combustion's own
            # `e_comb_solid_heat_sum` (the object-site fuel deposit that
            # bypasses TemperatureSolver's Pass 1 entirely — combustion.cpp
            # writes `temperature[s]` directly). Accumulating, like the
            # tail/combustion/seam groups above. (The thermostat term went with
            # Pass 3: T5b step 7 deleted it, and the identity closes with the
            # term REMOVED, not zeroed — report_t5b.md §7.3.)
            int(tsolver.e_solid_deposit_sum) + int(tsolver.e_solid_cond_sum)
            + int(comb.e_comb_solid_heat_sum),
        )

    def _solid_books():
        """(P-G5) Σ thermal_mass_raw·T_raw over thermal_solid cells — the
        SOLID side's own books, a SNAPSHOT refreshed by every step()."""
        return int(tsolver.solid_energy_books_sum)

    def _box_wall_solid_sum():
        """(P-G5) Σ thermal_mass_raw·T_raw over the box's OWN sealed glass
        ring only — the same currency `solid_energy_books_sum` uses
        (cap_real = thermal_mass << FP_SHIFT for a thermal solid, T already
        ambient-relative), computed directly here since there is no per-
        region C++ counter (and none is needed for a diagnostic print)."""
        shift = np.minimum(
            g.heat_inv_shift[box_wall_mask].astype(np.int64), 30)
        cap_real = np.int64(1) << (shift + 16)
        t = g.temperature[box_wall_mask].astype(np.int64)
        return int((cap_real * t).astype(object).sum())

    def _box_ET():
        """(Sum_box E, Sum_box N) over the box's ACCOUNTABLE cells."""
        acct = g._gas_energy_accountable()
        m = np.zeros_like(acct)
        m[box_in] = acct[box_in]
        n = _n_plane()
        return _e_sum(m), int(n[m].astype(object).sum())

    def _ke_sum():
        n = _n_plane()
        u2 = (g.wind_x.astype(np.int64) ** 2 + g.wind_y.astype(np.int64) ** 2)
        return int((n * (u2 >> 32)).astype(object).sum())

    # P-G1b: NO re-derive here any more. `seal_tiles` is an energy writer now
    # (design §2.7: the evacuated mass is MOVED, carrying the sealed tile's own
    # T_abs to each receiver, and the sub-count remainder retires), so the
    # field is already correct -- and re-deriving it from the mirror would
    # DESTROY exactly the remainders the seam just booked.
    if no_conduction:
        # Every conduction face becomes the NO_FACE sentinel, so the thermal
        # solver's §2 conduction pass is a structural no-op. The dial is read
        # off the live solver so this can never drift from config.
        g.face_shift[:] = sim.physics_runner.engine.temperature.no_face
    E0_box, N0_box = _box_ET()
    KE0 = _ke_sum()
    box_wall_sum0 = _box_wall_solid_sum()   # P-G5: box-scoped solid books, t=0
    P0_aq = float(g.atmosphere[box_in][open0[box_in]].mean()) / 65536.0
    # MASS vs PRESSURE-FIELD (2026-08-29, Erik's question): N inside the box
    # from the two conservative bulk planes — if P rises while N holds, the
    # pressure SOLVE is contaminated (no mass moved); if N rises, mass
    # actually crossed the sealed faces.
    o2 = int(g.gases.name_to_id["o2"])
    n2 = int(g.gases.name_to_id["inert_n2"])

    def n_box():
        return int(g.gas[o2][box_in].sum(dtype=np.int64) +
                   g.gas[n2][box_in].sum(dtype=np.int64))
    N0 = n_box()

    def dT(sl):
        d = (g.temperature.astype(np.int64) - T0)[sl]
        return float(d[open0[sl]].mean()) / 65536.0

    # (i) the ACROSS-TICK closure identity: bracket the WHOLE tick. P-G5 adds
    # a SECOND identity alongside it, over gas books + solid books together
    # (`ident_total`) — the same bracket, the same per-tick loop, extended
    # with the solid side's own three channels (design gas_energy_thermostat_
    # ledger_2026-08-30.md).
    acct0 = g._gas_energy_accountable()
    prev_e = _e_sum(acct0)
    prev_solid = _solid_books()
    prev_terms = _ident_terms()
    ident_total = {"worst": 0, "ticks": 0, "bad": 0}
    # #70: the trace (built AFTER every construction step, so its t=0 is the
    # state the first tick sees) and its phase-bracketing proxy on the runner.
    trace = None
    if trace_dir:
        trace = _Trace70(sim, box_in_mask, box_wall_mask, box_footprint)
        sim.physics_runner.engine = _EnginePhaseProxy(
            sim.physics_runner.engine, trace.hook)
    for t in range(1, end_tick + 1):
        if t == IGNITE_TICK and ignite:
            ignite_ring(g, sim.edit_queue, *CRATE, 2.5, 1.0)
        sim.set_paused(False)
        sim.step()
        if trace is not None:
            trace.end_tick(t)
        acct = g._gas_energy_accountable()
        e_now = _e_sum(acct)
        solid_now = _solid_books()
        terms = _ident_terms()
        expected = (terms[0]                      # EOS: absolute (reset/step)
                    + (terms[1] - prev_terms[1])  # tail: accumulating
                    + (terms[2] - prev_terms[2])  # combustion: accumulating
                    + (terms[3] - prev_terms[3])  # seams: accumulating
                    + terms[4])                   # water tail: reset per call
        resid = (e_now - prev_e) - expected
        ident["ticks"] += 1
        if resid:
            ident["bad"] += 1
            ident["worst"] = max(ident["worst"], abs(resid))
        # P-G5: the TOTAL ledger — gas books + solid books — against the same
        # `expected` PLUS the solid side's own accumulating group (terms[5]).
        expected_total = expected + (terms[5] - prev_terms[5])
        resid_total = ((e_now + solid_now) - (prev_e + prev_solid)) - expected_total
        ident_total["ticks"] += 1
        if resid_total:
            ident_total["bad"] += 1
            ident_total["worst"] = max(ident_total["worst"], abs(resid_total))
        prev_e, prev_solid, prev_terms = e_now, solid_now, terms

    P_aq = float(g.atmosphere[box_in][open0[box_in]].mean()) / 65536.0
    # 2026-08-29 LESSON: seal_tiles pushes the ring's gas INTO the box, so the
    # box starts at N ~ 1.29 atm-equivalent while P still reads 1.000 (the
    # solve catches up over ~2 s). The honest sealed-pocket invariant is
    # P == N x T_abs/T_amb, so report P/N — 1.000 means the solve is right.
    n_mean = float((g.gas[o2] + g.gas[n2])[box_in].mean()) / 65536.0
    u = np.sqrt((g.wind_x / 65536.0) ** 2 + (g.wind_y / 65536.0) ** 2)
    # #70: `u_max` below is the MAP-WIDE maximum (it has been since it was
    # written); the box's own maximum is printed beside it, because the
    # map-wide one sits in an unrelated doorway and was read as the box's.
    print(f"{name:>11}: box dT={dT(box_in):+7.1f}  box P {P0_aq:.3f}->{P_aq:.3f}"
          f"  box N x{n_box()/N0:5.3f} P/N={P_aq/n_mean:5.3f}"
          f"  bunker dT={dT(BUNKER):+7.1f}  pen dT={dT(PEN):+7.1f}"
          f"  arena dT={dT(ARENA):+6.1f}  map u_max={float(u.max()):5.1f}"
          f"  box u_max={float(u[box_in_mask].max()):5.2f}"
          f"  wall={wall_thick}"
          + (f"  [{overrides}]" if overrides else ""))

    # ---- arc #54 SB gate report (design §6 "SB") --------------------------
    E1_box, N1_box = _box_ET()
    # (ii) the HEADLINE: N-weighted mean absolute T over the box, in game-deg.
    q = 65536.0
    t0 = (E0_box / N0_box / q - t_amb_raw / q) if N0_box else 0.0
    t1 = (E1_box / N1_box / q - t_amb_raw / q) if N1_box else 0.0
    ok_i = (ident["bad"] == 0)
    ok_i_total = (ident_total["bad"] == 0)
    # P-G5 (Erik's ruling 2026-08-30): the box's OWN sealed glass ring warming
    # over the run, in box-deg equivalent — the sealed box's remaining game-deg
    # above the old +/-2 tolerance is the seal event's cold boundary shell
    # being warmed back up by ambient-held walls, i.e. the thermostat doing
    # its job (directly, and via the arena's own walls staying pinned near
    # ambient and feeding this box's ring through conduction), to be BOOKED
    # here rather than chased as a leak. BOX-SCOPED, not the GLOBAL
    # `e_thermostat_sum` printed below — that global sum is dominated by the
    # crate fire's own immediate walls (a different, much bigger effect with
    # nothing to do with this box 20+ tiles away), so it is not the right
    # subtraction term for a box-local headline.
    box_wall_sum1 = _box_wall_solid_sum()
    thermostat_box_deg = ((box_wall_sum1 - box_wall_sum0) / N1_box / q
                          if N1_box else 0.0)
    dT_box_raw = t1 - t0
    dT_box_adj = dT_box_raw - thermostat_box_deg
    ok_ii = abs(dT_box_adj) <= 2.0
    ok_iii = float(u.max()) < 3.0
    detail = "" if ok_i else (f" ({ident['bad']} bad, worst |resid|="
                              f"{ident['worst']})")
    detail_total = "" if ok_i_total else (
        f" ({ident_total['bad']} bad, worst |resid|={ident_total['worst']})")
    print(f"{'':>11}  arc#54  (i) closure identity: "
          f"{'EXACT' if ok_i else 'BROKEN'} across {ident['ticks']} TICKS"
          f"{detail}"
          f"   (ii) dT_box-thermostat={dT_box_adj:+7.2f} "
          f"{'PASS' if ok_ii else 'FAIL'} (+/-2)"
          f"   (iii) u_max={float(u.max()):5.1f} "
          f"{'PASS' if ok_iii else 'FAIL'} (<3)")
    # (The GLOBAL `e_thermostat_sum` print that stood here went with Pass 3:
    # T5b step 7 deleted the thermostat. `thermostat_box_deg` above is kept as
    # the box walls' own energy change; its name is historical.)
    print(f"{'':>11}  P-G5    (ii) raw dT_box=D(SumE/SumN)={dT_box_raw:+7.2f}  "
          f"box-wall energy change={thermostat_box_deg:+7.2f} box-deg"
          f"   TOTAL ledger (gas+solid): "
          f"{'EXACT' if ok_i_total else 'BROKEN'} across "
          f"{ident_total['ticks']} TICKS{detail_total}")
    print(f"{'':>11}  probes  D4 wall Sum|p.u|={int(eos.e_wall_work_probe_sum)}"
          f"  ts-wall={int(eos.e_ts_work_sum)}"
          f"  Sum N|u|^2 drift={_ke_sum() - KE0}"
          f"  rail_shortfall={int(eos.e_energy_floor_sum)}"
          f"  hits rad_clip={int(eos.rad_clip_hits)}"
          f" p_floor={int(eos.p_face_floor_hits)}"
          f" flux_sat={int(eos.flux_sat_hits)}"
          f" t_max={int(eos.t_max_phys_hits)}")
    # arc #54 P-G1b: THE THERMAL SOLVER'S GAS-SIDE CHANNEL, named and counted.
    # `e_gas_cond_sum` is the whole of gate (ii)'s residual and the bench's
    # `nofire_nocond` control is its zero: conduction across gas<->THERMAL_SOLID
    # faces carries energy INTO the gas books, because the solids are held at
    # ambient by Pass 3's two-way relaxation while conduction diffuses the
    # UNWEIGHTED T and the books are N-WEIGHTED. In an acoustically ringing
    # cell T and N are positively correlated, so the unweighted mean sits below
    # the N-weighted one and the walls top the gas up forever. P-G1b BOOKS this
    # channel (that is what makes (i) exact across ticks); REMOVING it is a
    # physics decision outside this patch's scope -- see the P-G1b report.
    print(f"{'':>11}  gas-side  cond={int(tsolver.e_gas_cond_sum)}"
          f"  deposit={int(tsolver.e_gas_deposit_sum)}"
          f"  rail={int(tsolver.e_gas_rail_sum)}"
          f"  (cond in box-deg = "
          f"{int(tsolver.e_gas_cond_sum) / N1_box / q if N1_box else 0.0:+.2f})")
    # #70: `thermostat_box_deg` divides a change in SOLID books (heat counts,
    # cap*T) by the box's N, i.e. it prices a heat count as one gas book count.
    # They are the same unit only at c_v == 1 (T5a, T2 §1.2); at the shipped
    # c_v a heat count is 1/c_v book counts. The gate (ii) arithmetic above is
    # left exactly as it was -- this line only states the same change in the
    # box air's own currency.
    c_v = float(tsolver.c_v)
    if N1_box and c_v != 1.0:
        print(f"{'':>11}  #70     box-wall energy change in the box air's books "
              f"currency (heat/c_v) = {thermostat_box_deg / c_v:+9.2f} box-deg "
              f"(the P-G5 line prints it x c_v = {c_v})")
    if trace is not None:
        for line in t0_load + trace.t0_lines + trace.t1_lines:
            print(line)
        for line in trace.report(name, TPS):
            print(line)
        out = Path(trace_dir)
        out.mkdir(parents=True, exist_ok=True)
        trace.save(out / f"{name}.npz", {
            "variant": name, "overrides": overrides, "wall_thick": wall_thick,
            "ignite": ignite, "no_conduction": no_conduction,
            "seal_push": seal_push, "radiation_off": radiation_off,
            "runner_overrides": runner_overrides or {},
            "end_tick": end_tick, "tps": TPS, "cuda": _USE_CUDA,
            "k_leak_q": int(sim.physics_runner.k_leak_q),
            "rad_amb_vacuum_q": int(sim.physics_runner.rad_amb_vacuum_q),
            "e_table_0": int(sim.physics_runner.engine.emissive.table()[0]),
            "t0_load_lines": t0_load})
        print(f"{'':>11}  #70     trace -> {out / (name + '.npz')}")


def main() -> None:
    end_tick = END_TICK if _SECONDS is None else int(round(float(_SECONDS) * TPS))
    print(f"sealed-box bisection — crate fire only, {end_tick/TPS:.0f} s, "
          f"FIXED = box dT ~ 0")
    wanted = set(sys.argv[1:])          # optional: run only the named variants
    # #70: the issue-#70 variants run only when NAMED, so a bare run is the
    # historical bisection set, unchanged.
    for spec in VARIANTS + [s for s in VARIANTS_70 if s[0] in wanted]:
        name, overrides, pos, kwargs = _spec_args(spec)
        if wanted and name not in wanted:
            continue
        run_variant(name, overrides, *pos, end_tick=end_tick,
                    trace_dir=_TRACE_DIR, **kwargs)


if __name__ == "__main__":
    main()
