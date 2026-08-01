"""Headless ROOM/VENT fire bench (P-F4a task order item 1, Fire & Heat
tuning arc — docs/fire_realism_design_2026-08-01.md v5.2's execution order,
first bullet: "P-F4a (bench tooling + CAMPFIRE REFERENCE OBJECT ... )").

A sibling of ``tools/fire_timing_harness.py``, importing its machinery
(config-dial overrides, the O2/gate diagnostics, material ids) rather than
duplicating it. Where that module's bench is a PLANETSIDE open field (a
SPACE ring reservoir, sky-exchange refill — the STILL-AIR REFERENCE ARENA,
see its own recharter docstring), THIS module is the SHIP-STYLE bench the
F4 supply pre-measurements need (design doc §0: "SPACE SHIPS: the
sky-exchange refill is planetside-only — most levels have NO sky; supply
design + T2's bench must lead with sealed/vented ship rooms"):

  * a hull-enclosed rectangular room of parametric interior size
    (``build_room_level`` / ``interior_w`` x ``interior_h``);
  * boundary = "space" (the engine default when a level omits ``boundary``)
    — NO planetside SPACE ring, no ambient reservoir, no sky-exchange refill.
    SEALED MEANS SEALED: air lost to combustion or to a breach is gone for
    good, exactly the "ships cannot afford" smother requirement the v4/v5
    rulings hinge on (Option 2b, plain edition §10);
  * an OPTIONAL VENT: a parametric-width gap in the hull ring, one of three
    states —
      "closed" — no gap; the hull ring is intact (a fully sealed room);
      "open"   — the gap is present FROM TICK 0 (pre-Simulation tilemap
                 surgery: the vent tiles carry the v2 SPACE(9) code directly,
                 the exact idiom fire_timing_harness.build_level uses for its
                 sky ring — GameMap reads them as real vacuum at load since
                 boundary != "ambient");
      "breach" — the gap is ABSENT at build time (hull intact) and opened
                 MID-RUN, at a parametric tick, via the EXISTING destroy/edit
                 path: ``GameMap.destroy_wall(fy, fx)`` — the exact call
                 tests/field_ab_harness.py's scenario already uses
                 (``g.destroy_wall(8, 0)``) to open a hull breach. No new
                 mechanic: destroy_wall already converts hull -> air and,
                 for an edge-hull tile (the vent ring IS the map edge here),
                 auto-joins it to the vacuum boundary (breach_mask) and reads
                 the neighbour-mean atmosphere seed — see its docstring in
                 src/simulation/gamemap.py.
  * crate layouts as a list of ``(x, y, material)`` tuples in
    INTERIOR-RELATIVE coordinates (0,0 = the interior's own top-left cell,
    one tile in from the hull ring on every side) — independent of hull
    thickness;
  * the SAME CSV / scorecard conventions as fire_timing_harness (recording
    loop shape, ``t``/``I``/``T``/``hp`` columns, the harness's own
    ``_hot_gate``/``_open_neighbors`` diagnostics reused verbatim, not
    reimplemented).

RUN MODES:
    python tools/fire_room_bench.py --demo     # sealed + vent-open + breach
    python tools/fire_room_bench.py --mode sealed
    python tools/fire_room_bench.py --mode vent-open --vent-width 2
    python tools/fire_room_bench.py --mode breach --breach-tick-s 2.0

Deterministic: fixed seed, no RNG in the driven path. Headless: builds a
synthetic LevelData in memory; never opens a display.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# --- import path (mirror tools/fire_timing_harness.py) ---------------------
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
from simulation.gases import O2, INERT_N2         # noqa: E402

# tools/ is sys.path[0] when this file is run as a script (fire_tune_loop.py
# precedent) — a bare module import finds the sibling harness.
from fire_timing_harness import (                 # noqa: E402
    FP_ONE, AIR, HULL, WOOD, DOOR, STEEL, GLASS, FURN, SPACE, KIND,
    apply_overrides, restore_overrides, _hot_gate, _open_neighbors,
)

# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------
def vent_tile_coords(interior_w, interior_h, vent_width, vent_side="east"):
    """Full-grid ``(fy, fx)`` coordinates of a parametric-width gap centered
    in the hull ring on the given side. ``vent_width <= 0`` -> ``[]`` (no
    vent — the room stays fully sealed regardless of ``vent_state``).

    The gap sits IN the one-tile hull ring, which is also the map's outer
    edge for this bench's tilemap — so every vent tile is an edge-hull tile,
    and ``GameMap.destroy_wall``'s ``on_edge_hull`` rule fires on it exactly
    (auto-joins the boundary vacuum reservoir on breach)."""
    if vent_width <= 0:
        return []
    h, w = interior_h + 2, interior_w + 2
    if vent_side in ("east", "west"):
        start = 1 + max(0, (interior_h - vent_width) // 2)
        rows = list(range(start, min(start + vent_width, h - 1)))
        col = (w - 1) if vent_side == "east" else 0
        return [(r, col) for r in rows]
    if vent_side in ("north", "south"):
        start = 1 + max(0, (interior_w - vent_width) // 2)
        cols = list(range(start, min(start + vent_width, w - 1)))
        row = (h - 1) if vent_side == "south" else 0
        return [(row, c) for c in cols]
    raise ValueError(
        f"vent_side must be one of north/south/east/west, got {vent_side!r}")


def _resolve_material_id(mat):
    """Accept either a material id (int) or a ``[materials.<name>]`` name."""
    if isinstance(mat, str):
        from simulation.materials import MATERIAL_NAMES
        name_to_id = {v: k for k, v in MATERIAL_NAMES.items()}
        if mat not in name_to_id:
            raise KeyError(f"unknown material name {mat!r}")
        return name_to_id[mat]
    return int(mat)


def build_room_level(interior_w, interior_h, tile_size_m, *,
                     vent_width=0, vent_side="east", vent_open_at_build=False,
                     crates=()):
    """A hull-enclosed rectangular room, SHIP-STYLE (``boundary="space"`` —
    the engine default when omitted: NO planetside SPACE ring, no ambient
    reservoir, no sky-exchange refill; "sealed means sealed").

    ``crates``: list of ``(x, y, material)`` tuples, interior-relative
    (0,0 == the interior's own top-left cell — one tile in from the hull
    ring on every side). ``material`` is a material id int or a
    ``[materials.<name>]`` name string.

    ``vent_open_at_build``: True -> the vent tiles (see
    :func:`vent_tile_coords`) carry the v2 SPACE(9) code directly in the
    returned tilemap (pre-Simulation tilemap surgery — the vent is open from
    tick 0). False -> the hull ring is left fully intact at build time; a
    caller wanting a MID-RUN breach converts those same tiles later via
    ``GameMap.destroy_wall`` (see :func:`run_room`'s ``vent_state="breach"``
    handling) — the one existing destroy/edit path, not a new mechanic.
    """
    h, w = interior_h + 2, interior_w + 2
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = HULL
    tm[h - 1, :] = HULL
    tm[:, 0] = HULL
    tm[:, w - 1] = HULL
    for (cx, cy, mat) in crates:
        fy, fx = cy + 1, cx + 1
        if not (1 <= fy < h - 1 and 1 <= fx < w - 1):
            raise ValueError(
                f"crate at interior ({cx},{cy}) falls outside the "
                f"{interior_w}x{interior_h} interior")
        tm[fy, fx] = _resolve_material_id(mat)
    if vent_open_at_build:
        for (fy, fx) in vent_tile_coords(interior_w, interior_h,
                                         vent_width, vent_side):
            tm[fy, fx] = SPACE
    return LevelData(
        name="fire_room_bench", version="2", path=Path("."),
        tilemap=tm, tile_size_m=float(tile_size_m), diffuse_path=Path("."),
        boundary="space",   # SHIP-style: no planetside sky ring (the default)
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def run_room(*, interior_w, interior_h, tile_size_m, crates, ignite_xy,
            vent_width=0, vent_side="east", vent_state="closed",
            breach_tick_s=None, max_seconds=60.0, tail_seconds=3.0,
            overrides=None, seed=12345, verbose=True):
    """Run one room/vent burn. ``vent_state`` is one of:
      "closed" — no gap, ever;
      "open"   — the gap exists from tick 0;
      "breach" — the gap opens mid-run at ``breach_tick_s`` (requires
                 ``vent_width > 0`` and ``breach_tick_s`` set).
    ``ignite_xy`` — interior-relative (x, y) of the tile to ignite; it must
    be one of the placed ``crates`` (a flammable material). ``overrides``
    patches CFG dials exactly like fire_timing_harness.run_one (restored
    afterwards). Returns a metrics dict (same shape/spirit as
    fire_timing_harness.run_one's, room-scoped fields added)."""
    if vent_state not in ("closed", "open", "breach"):
        raise ValueError(
            f"vent_state must be closed/open/breach, got {vent_state!r}")
    if vent_state == "breach" and (breach_tick_s is None or vent_width <= 0):
        raise ValueError(
            "vent_state='breach' needs vent_width > 0 and breach_tick_s set")
    restore = apply_overrides(overrides or {})
    try:
        return _run_room_inner(
            interior_w=interior_w, interior_h=interior_h,
            tile_size_m=tile_size_m, crates=crates, ignite_xy=ignite_xy,
            vent_width=vent_width, vent_side=vent_side, vent_state=vent_state,
            breach_tick_s=breach_tick_s, max_seconds=max_seconds,
            tail_seconds=tail_seconds, overrides=overrides or {}, seed=seed,
            verbose=verbose)
    finally:
        restore_overrides(restore)


def _diagnose(rec, snap_tick, fuel_out_tick, had_fire, x_ext, crate_hp0,
             margin=0.01):
    """Classify WHY the fire stopped — the gate (c) "record which" call.

    One of: never-ignited, sustained-to-timeout, burned-out (fuel floor
    reached while I was still > 0), fuel-floor-at-snap (hp ~ 0 exactly when
    I -> 0 — indistinguishable from ordinary burnout), O2 (the flame
    neighbourhood's mole fraction X was at/near X_ext when I snapped to 0),
    or knee (I snapped to 0 with fuel AND O2 still available — the
    intensity-logistic collapse gate (vii) calls "dies BY THE KNEE, not the
    fuel floor")."""
    if not had_fire:
        return "never ignited (seed did not bootstrap past its own hot/O2 gates)"
    if snap_tick is None:
        if fuel_out_tick is not None:
            return "burned out (fuel floor reached while still burning)"
        return "sustained to timeout"
    idx = min(max(0, snap_tick - 2), len(rec["hp"]) - 1)
    hp_at = rec["hp"][idx]
    x_at = rec["x_local"][idx]
    hot_at = rec["hot"][idx]
    if hp_at <= 0.02 * max(crate_hp0, 1e-9):
        return f"fuel floor (hp={hp_at:.3f} ~ 0 at snap-out)"
    if not np.isnan(x_at) and x_at <= x_ext + margin:
        return f"O2 (X_local={x_at:.4f} at/near X_ext={x_ext:.2f}, fuel remained)"
    if hot_at < 0.999:
        return (f"knee (T-gate limited: hot={hot_at:.3f}, "
               f"X_local={x_at:.4f} well above X_ext={x_ext:.2f} -- O2 was available)")
    return (f"knee (intensity collapse: hot=1.0, X_local={x_at:.4f} "
           f"well above X_ext={x_ext:.2f} -- neither gate was binding)")


def _run_room_inner(*, interior_w, interior_h, tile_size_m, crates, ignite_xy,
                    vent_width, vent_side, vent_state, breach_tick_s,
                    max_seconds, tail_seconds, overrides, seed, verbose):
    level = build_room_level(
        interior_w, interior_h, tile_size_m,
        vent_width=vent_width, vent_side=vent_side,
        vent_open_at_build=(vent_state == "open"), crates=crates)
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
    gmap = sim.gmap
    vent_tiles = (vent_tile_coords(interior_w, interior_h, vent_width, vent_side)
                 if vent_width > 0 else [])

    ix, iy = ignite_xy
    fy, fx = iy + 1, ix + 1
    mat_id = int(gmap.material[fy, fx])
    if mat_id == AIR or not bool(gmap.materials.flammable[mat_id]):
        raise ValueError(
            f"ignite_xy {ignite_xy} (full-grid {(fy, fx)}) is not a "
            f"flammable placed crate (material id {mat_id})")

    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
    gmap.fire[fy, fx] = fire_fixed.quantize_scalar(seed_i)
    # Game-faithful seed (mirrors fire_timing_harness.run_one): a tile only
    # ignites in-engine BECAUSE its T crossed ignition_temp, so seed T at the
    # tile's OWN material ignition_temp (per-material, not a hardcoded value).
    ign_temp = float(gmap.materials.ignition_temp[mat_id])
    gmap.temperature[fy, fx] = fire_fixed.quantize_scalar(ign_temp)

    fire_t_ext = int(gmap.fire_T_ext_plane[fy, fx]) / FP_ONE
    fire_t_span = float(getattr(CFG.physics.fire, "fire_T_span", 150.0))
    x_ext = float(getattr(CFG.physics.fire, "o2_frac_ext", 0.13))
    crate_hp0 = float(gmap.materials.hp[mat_id])

    tps = float(CFG.clock.ticks_per_second)
    dt = 1.0 / tps
    n_max = int(round(max_seconds * tps))
    breach_tick = (max(1, int(round(breach_tick_s * tps)))
                  if vent_state == "breach" else None)

    nbrs = _open_neighbors(gmap, fy, fx)
    room_mask = (~gmap.solid) & (~gmap.is_vacuum)

    rec = {k: [] for k in ("t", "I", "T", "hp", "o2", "x_local", "hot",
                           "o2room_x", "vent_open")}
    had_fire = False
    fuel_out_tick = None
    snap_tick = None
    breach_done = False
    k = 0
    for k in range(1, n_max + 1):
        if vent_state == "breach" and not breach_done and k == breach_tick:
            # THE MID-RUN BREACH: the existing destroy/edit path
            # (GameMap.destroy_wall — see tests/field_ab_harness.py's
            # `g.destroy_wall(8, 0)` for the identical call shape). Every
            # vent tile is an edge-hull tile (the vent sits in the map's
            # outer hull ring), so destroy_wall's on_edge_hull rule fires:
            # material -> air, tile auto-joins the vacuum boundary, the
            # atmosphere is seeded from the neighbour mean (no artificial
            # vacuum pulse).
            for (vy, vx) in vent_tiles:
                gmap.destroy_wall(vy, vx)
            breach_done = True
        sim.set_paused(False)
        sim.step()
        t = k * dt
        fire_q = int(gmap.fire[fy, fx])
        I = fire_q / FP_ONE
        T = int(gmap.temperature[fy, fx]) / FP_ONE
        hp = int(gmap.wall_hp[fy, fx]) / FP_ONE
        o2 = (float(np.mean([int(gmap.gas[O2, ny, nx]) for (ny, nx) in nbrs]))
             / FP_ONE) if nbrs else float("nan")
        if nbrs:
            _o2_loc = float(sum(int(gmap.gas[O2, ny, nx]) for (ny, nx) in nbrs))
            _tot_loc = float(sum(int(gmap.gas[O2, ny, nx]) + int(gmap.gas[INERT_N2, ny, nx])
                                 for (ny, nx) in nbrs))
            x_local = _o2_loc / max(1.0, _tot_loc)
        else:
            x_local = float("nan")
        hot = _hot_gate(T, fire_t_ext, fire_t_span)
        _o2m = gmap.gas[O2][room_mask].astype(np.float64)
        _ntm = _o2m + gmap.gas[INERT_N2][room_mask].astype(np.float64)
        o2room_x = float(np.divide(_o2m, _ntm, out=np.zeros_like(_o2m),
                                   where=_ntm > 0).mean())
        vent_open_now = bool(vent_tiles) and bool(gmap.is_vacuum[vent_tiles[0]])
        for key, val in (("t", t), ("I", I), ("T", T), ("hp", hp), ("o2", o2),
                        ("x_local", x_local), ("hot", hot),
                        ("o2room_x", o2room_x), ("vent_open", float(vent_open_now))):
            rec[key].append(val)
        if I > 0.05:
            had_fire = True
        if had_fire and hp <= 0.0 and fuel_out_tick is None:
            fuel_out_tick = k
        if had_fire and fire_q == 0 and snap_tick is None:
            snap_tick = k
        done_tick = snap_tick if snap_tick is not None else fuel_out_tick
        if done_tick is not None and t > done_tick * dt + tail_seconds:
            break

    for key in rec:
        rec[key] = np.asarray(rec[key], dtype=np.float64)

    cause = _diagnose(rec, snap_tick, fuel_out_tick, had_fire, x_ext, crate_hp0)

    metrics = dict(
        vent_state=vent_state, vent_width=vent_width, vent_side=vent_side,
        breach_tick_s=breach_tick_s, breach_tick=breach_tick,
        breach_done=breach_done,
        interior_w=interior_w, interior_h=interior_h, tile_size_m=tile_size_m,
        ignite_xy=tuple(ignite_xy), material_id=mat_id,
        material_name=gmap.materials.names[mat_id], crate_hp0=crate_hp0,
        fire_T_ext=fire_t_ext, fire_T_span=fire_t_span, x_ext=x_ext,
        had_fire=had_fire, fuel_out_tick=fuel_out_tick, snap_tick=snap_tick,
        cause=cause, n_ticks=len(rec["t"]), rec=rec, tps=tps, dt=dt,
        vent_tiles=vent_tiles, overrides=dict(overrides), seed=seed,
    )
    if verbose:
        _print_room(metrics)
    return metrics


# ---------------------------------------------------------------------------
# CSV + reporting (same conventions as fire_timing_harness.write_timeseries_csv)
# ---------------------------------------------------------------------------
def write_room_csv(m, path):
    import csv
    rec = m["rec"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# fire_room_bench room/vent time-series"])
        w.writerow([f"# vent_state={m['vent_state']} vent_width={m['vent_width']} "
                    f"vent_side={m['vent_side']} breach_tick_s={m['breach_tick_s']}"])
        w.writerow([f"# interior={m['interior_w']}x{m['interior_h']} "
                    f"material={m['material_name']} ignite_xy={m['ignite_xy']}"])
        w.writerow([f"# cause={m['cause']}"])
        w.writerow(["t_s", "I", "T_game", "wall_hp", "O2_flame_nbr", "X_local",
                    "hot", "O2room_X", "vent_open"])
        for i in range(len(rec["t"])):
            w.writerow([f"{rec['t'][i]:.4f}", f"{rec['I'][i]:.6f}",
                        f"{rec['T'][i]:.3f}", f"{rec['hp'][i]:.6f}",
                        f"{rec['o2'][i]:.6f}", f"{rec['x_local'][i]:.6f}",
                        f"{rec['hot'][i]:.4f}", f"{rec['o2room_x'][i]:.6f}",
                        f"{rec['vent_open'][i]:.0f}"])


def _mmss(sec):
    if sec is None or np.isnan(sec):
        return "  n/a "
    return f"{sec:6.1f}s"


def _print_room(m):
    print("=" * 76)
    print(f"ROOM BENCH  vent_state={m['vent_state']!r}  interior "
         f"{m['interior_w']}x{m['interior_h']}  tile {m['tile_size_m']} m  "
         f"boundary=space (SHIP-style, no sky)")
    if m["vent_width"] > 0:
        print(f"  vent: width={m['vent_width']} side={m['vent_side']!r} "
             f"tiles={m['vent_tiles']}"
             + (f"  breach_tick_s={m['breach_tick_s']} (tick {m['breach_tick']})"
                if m["vent_state"] == "breach" else ""))
    else:
        print("  vent: NONE (sealed hull ring)")
    print(f"  ignite: {m['material_name']} at interior {m['ignite_xy']} "
         f"(hp0={m['crate_hp0']:.1f}, fire_T_ext={m['fire_T_ext']:.0f})")
    print("-" * 76)
    rec = m["rec"]
    peak_I = float(rec["I"].max()) if rec["I"].size else 0.0
    print(f"  had_fire={m['had_fire']}  peak_I={peak_I:.3f}  "
         f"burnout={_mmss(m['fuel_out_tick'] * m['dt'] if m['fuel_out_tick'] else None)}  "
         f"snap-out={_mmss(m['snap_tick'] * m['dt'] if m['snap_tick'] else None)}")
    print(f"  CAUSE: {m['cause']}")
    if m["vent_state"] == "breach":
        print(f"  breach fired: {m['breach_done']}  "
             f"vent is_vacuum final: {bool(rec['vent_open'][-1]) if rec['vent_open'].size else 'n/a'}")
    print(f"  room O2 X (mean over open interior air, final): {rec['o2room_x'][-1] if rec['o2room_x'].size else float('nan'):.4f}"
         f"  (min over run: {rec['o2room_x'].min() if rec['o2room_x'].size else float('nan'):.4f})")
    print("=" * 76)


# ---------------------------------------------------------------------------
# Demo — the gate (c) canonical three-mode run
# ---------------------------------------------------------------------------
def _demo_common(interior_w=12, interior_h=10, tile_size_m=0.5,
                 vent_width=2, vent_side="east", crate_material=FURN,
                 crate_xy=(6, 5)):
    return dict(interior_w=interior_w, interior_h=interior_h,
               tile_size_m=tile_size_m, vent_width=vent_width,
               vent_side=vent_side,
               crates=[(crate_xy[0], crate_xy[1], crate_material)],
               ignite_xy=crate_xy)


def run_demo(*, max_seconds=60.0, tail_seconds=3.0, breach_tick_s=0.3,
            overrides=None, verbose=True, csv_dir=None):
    """The three canonical room/vent runs gate (c) asks for: sealed,
    vent-open, breach-at-tick. Returns {mode: metrics}."""
    common = _demo_common()
    results = {}
    for mode, kw in (
        ("sealed", dict(vent_state="closed")),
        ("vent-open", dict(vent_state="open")),
        ("breach", dict(vent_state="breach", breach_tick_s=breach_tick_s)),
    ):
        m = run_room(max_seconds=max_seconds, tail_seconds=tail_seconds,
                    overrides=overrides, verbose=verbose, **common, **kw)
        results[mode] = m
        if csv_dir is not None:
            write_room_csv(m, Path(csv_dir) / f"room_bench_{mode}.csv")
    print("=" * 76)
    print("DEMO SUMMARY — sealed / vent-open / breach-at-tick")
    print("-" * 76)
    for mode, m in results.items():
        print(f"  {mode:10s}  cause: {m['cause']}")
    print("=" * 76)
    return results


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true",
                    help="run the three canonical modes (sealed/vent-open/breach)")
    ap.add_argument("--mode", choices=("sealed", "vent-open", "breach"),
                    default="sealed")
    ap.add_argument("--interior-w", type=int, default=12)
    ap.add_argument("--interior-h", type=int, default=10)
    ap.add_argument("--tile-size-m", type=float, default=0.5)
    ap.add_argument("--vent-width", type=int, default=2)
    ap.add_argument("--vent-side", choices=("north", "south", "east", "west"),
                    default="east")
    ap.add_argument("--breach-tick-s", type=float, default=0.3)
    ap.add_argument("--crate-x", type=int, default=6)
    ap.add_argument("--crate-y", type=int, default=5)
    ap.add_argument("--material", default="furniture",
                    help="[materials.<name>] to place/ignite (default furniture)")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--tail-seconds", type=float, default=3.0)
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="KEY=VALUE", help="override any CFG dial (repeatable)")
    ap.add_argument("--csv", default=None, metavar="PATH")
    ap.add_argument("--csv-dir", default=None, metavar="DIR",
                    help="--demo: write room_bench_<mode>.csv into this dir")
    args = ap.parse_args(argv)

    overrides = {}
    for s in args.sets:
        if "=" not in s:
            ap.error(f"--set expects KEY=VALUE, got {s!r}")
        key, val = s.split("=", 1)
        overrides[key.strip()] = val.strip()

    if args.demo:
        run_demo(max_seconds=args.max_seconds, tail_seconds=args.tail_seconds,
                 breach_tick_s=args.breach_tick_s, overrides=overrides,
                 verbose=True, csv_dir=args.csv_dir)
        return

    crate_xy = (args.crate_x, args.crate_y)
    vent_state = {"sealed": "closed", "vent-open": "open",
                 "breach": "breach"}[args.mode]
    m = run_room(
        interior_w=args.interior_w, interior_h=args.interior_h,
        tile_size_m=args.tile_size_m, vent_width=args.vent_width,
        vent_side=args.vent_side, vent_state=vent_state,
        breach_tick_s=(args.breach_tick_s if vent_state == "breach" else None),
        crates=[(crate_xy[0], crate_xy[1], args.material)],
        ignite_xy=crate_xy, max_seconds=args.max_seconds,
        tail_seconds=args.tail_seconds, overrides=overrides, verbose=True)
    if args.csv:
        write_room_csv(m, args.csv)
        print(f"[csv] wrote {args.csv}  ({m['n_ticks']} ticks)")


if __name__ == "__main__":
    main()
