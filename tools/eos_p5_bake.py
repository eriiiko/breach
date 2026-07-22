"""EOS refactor — P5 combined-system bake-off harness (docs/eos_refactor_design.md
§8 P5, decisions log #15/#16 resume plan).

Drives the REAL C++ engine (via the `breach_physics` pybind11 module + the
`simulation.gamemap.GameMap` / `simulation.physics_runner.PhysicsRunner`
Python layer — NOT the numpy prototype) through a fixed set of synthetic
scenarios (B1-B7 below), on a 160x160 map built from the same wall-ring /
room / door / corridor building blocks throughout. Renders side-by-side
old-engine-vs-new-engine GIFs (adapting prototypes/eos/render.py's
compositing approach — walls flat color, smoke bright veil, water blue,
fire red->orange->yellow driven by `gmap.fire` intensity since `temperature`
is not a comparable quantity across the two engines, P optionally modulating
brightness), collects per-tick cost, and writes an index.html.

NO TUNING, NO PHYSICS CHANGES — this script only observes. Any scenario that
looks ugly is reported as-is (see the P5 task instructions).

This script has three modes, because the OLD (pre-refactor, commit b17e150)
and NEW (main, P1-P4 merged) engines live in different worktrees/checkouts
with different compiled `breach_physics` pybind11 extensions and different
`src/simulation/*.py` trees — they cannot be imported into the same Python
process. Each mode is invoked as a *separate* interpreter run:

    python tools/eos_p5_bake.py run --engine-root <root> --tag old|new \\
        --raw-dir tools/eos_p5_out/_raw [--scenarios b1,b2,...] [--ticks-scale 1.0]

        Builds the scenarios applicable to `--tag`, runs them against the
        engine checked out at `--engine-root` (which must already be built:
        `cpp/build/Release/breach_physics*.pyd` must exist there), and writes
        one GIF + one trace json + one timing json per scenario into
        `<raw-dir>/<tag>/`.

    python tools/eos_p5_bake.py compose --raw-dir tools/eos_p5_out/_raw \\
        --out tools/eos_p5_out

        Reads whatever `run` produced under `<raw-dir>/{old,new}/`, builds
        the final side-by-side (or single, for new-only scenarios) GIFs,
        the B4/B6 trace plots, the cost table, and `index.html`. Pure
        post-processing — imports no simulation code.

    python tools/eos_p5_bake.py determinism --engine-root <root> \\
        --raw-dir tools/eos_p5_out/_raw

        Runs one scenario (B3 by default) TWICE on the given (new) engine,
        hashing the same field set tests/test_eos_p4_combustion.py uses per
        tick, and records a pass/fail digest-match json.

Typical invocation (see docs/... for the exact commands used for the P5
report):

    <new_python> tools/eos_p5_bake.py run --engine-root <new_root> --tag new --raw-dir tools/eos_p5_out/_raw
    <old_python> tools/eos_p5_bake.py run --engine-root <old_root> --tag old --raw-dir tools/eos_p5_out/_raw
    <new_python> tools/eos_p5_bake.py determinism --engine-root <new_root> --raw-dir tools/eos_p5_out/_raw
    <new_python> tools/eos_p5_bake.py compose --raw-dir tools/eos_p5_out/_raw --out tools/eos_p5_out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

Q16 = 65536


def q16(x: float) -> int:
    return int(round(float(x) * Q16))


# ---------------------------------------------------------------------------
# Map-building helpers (engine-agnostic — plain numpy tilemap arrays).
# Materials (simulation/materials.py, unchanged across the refactor):
#   0 air, 1 hull, 2 wood, 3 door, 4 steel, 5 glass, 6 furniture, 9 = SPACE
# ---------------------------------------------------------------------------
GRID = 160
MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR = 0, 1, 2, 3
SPACE_CODE = 9


def blank_map() -> np.ndarray:
    """Everything defaults to SPACE (open vacuum) — rooms are carved out of it,
    so any wall that's never explicitly walled off from SPACE is naturally an
    'edge hull' a breach can vent into, and we never need to hand-place an
    outer vacuum ring."""
    return np.full((GRID, GRID), SPACE_CODE, dtype=np.int32)


def hull_box(tm, y0, y1, x0, x1, interior=MAT_AIR, wall=MAT_HULL):
    tm[y0, x0:x1 + 1] = wall
    tm[y1, x0:x1 + 1] = wall
    tm[y0:y1 + 1, x0] = wall
    tm[y0:y1 + 1, x1] = wall
    if interior is not None:
        tm[y0 + 1:y1, x0 + 1:x1] = interior


def punch_door_h(tm, y, x0, x1):
    tm[y, x0:x1 + 1] = MAT_DOOR


def punch_door_v(tm, y0, y1, x):
    tm[y0:y1 + 1, x] = MAT_DOOR


def content_bbox(tilemap, margin=6):
    """Bounding box of everything that isn't the default SPACE fill, padded by
    `margin` tiles and clamped to the grid — used so a scenario's actual room(s)
    fill the rendered frame instead of drowning in a 160x160 canvas of vacuum
    (the first-cut GIFs were unwatchable: a 9-row corridor or a 40x40 room on a
    160x160 canvas renders as a barely-visible sliver)."""
    ys, xs = np.where(tilemap != SPACE_CODE)
    y0, y1 = int(ys.min()) - margin, int(ys.max()) + margin
    x0, x1 = int(xs.min()) - margin, int(xs.max()) + margin
    h, w = tilemap.shape
    return max(0, y0), min(h - 1, y1), max(0, x0), min(w - 1, x1)


# ---------------------------------------------------------------------------
# Engine loading — inserts the given engine root's paths BEFORE importing any
# `simulation.*` / `breach_physics` module, so the same script file drives
# either checkout depending on --engine-root.
# ---------------------------------------------------------------------------
def load_engine(engine_root: Path):
    root = str(engine_root.resolve())
    for p in (root, str(engine_root / "src"), str(engine_root / "cpp" / "build" / "Release")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import breach_physics as bp                                    # noqa
    from level_loader import LevelData                              # noqa
    from simulation.gamemap import GameMap                          # noqa
    from simulation.physics_runner import PhysicsRunner             # noqa
    from simulation.field_edit import EditQueue                     # noqa
    from simulation.physics import apply_explosion, add_explosion_smoke  # noqa
    from simulation import gases as gases_mod                       # noqa
    from simulation.materials import MaterialTable, MAT_WOOD as _MAT_WOOD  # noqa

    has_o2 = hasattr(gases_mod, "O2")
    tbl = MaterialTable.from_config()
    ign_wood_q16 = int(tbl.ignition_temp_q16[_MAT_WOOD])
    return SimpleNamespace(
        bp=bp, LevelData=LevelData, GameMap=GameMap, PhysicsRunner=PhysicsRunner,
        EditQueue=EditQueue, apply_explosion=apply_explosion,
        add_explosion_smoke=add_explosion_smoke, gases=gases_mod,
        has_o2=has_o2, ign_wood_q16=ign_wood_q16,
    )


def new_gamemap(eng, tilemap, name="p5"):
    ld = eng.LevelData(name=name, version="2", path=Path("."), tilemap=tilemap,
                        tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    return eng.GameMap(ld)


def do_blast(eng, gmap, queue, rng, fy, fx, radius, pressure, wall_damage, smoke_noise=0.85):
    eng.apply_explosion(gmap, queue, fy, fx, radius, pressure, wall_damage)
    eng.add_explosion_smoke(gmap, queue, fy, fx, radius, noise=smoke_noise)
    queue.flush(gmap, rng)


def ignite(eng, gmap, at, intensity=0.6, temp_mult=1.5):
    gmap.fire[at] = q16(intensity)
    gmap.temperature[at] = int(eng.ign_wood_q16 * temp_mult)


# ---------------------------------------------------------------------------
# Scenario specs. Each returns a dict consumed by run_scenario():
#   name, ticks, engines (subset of {'old','new'}), frame_stride,
#   build(eng) -> tilemap, setup(eng, gmap) [optional],
#   events: {tick_index: callback(eng, gmap, queue, rng)},
#   trace: {field_name: callback(gmap) -> float}  (sampled every tick)
#   caption: one-line "what to look for" for index.html
# ---------------------------------------------------------------------------
def spec_b1_corridor_blast():
    def build(eng):
        tm = blank_map()
        hull_box(tm, 50, 78, 10, 150)
        return tm

    def ev(eng, gmap, queue, rng):
        do_blast(eng, gmap, queue, rng, fy=64, fx=25, radius=10, pressure=8.0, wall_damage=40.0)

    return dict(
        name="B1_corridor_blast", ticks=120, engines=("old", "new"), frame_stride=1,
        build=build, events={3: ev},
        trace={"P_mid": lambda g: float(g.atmosphere[64, 80]) / Q16,
               "P_far": lambda g: float(g.atmosphere[64, 130]) / Q16},
        caption=("A blast in a long sealed corridor: watch the compression front travel "
                  "and reflect off the far end. New engine = real acoustic wave off one P "
                  "field; old engine = wave_p buffet over the atmosphere dome."),
    )


def spec_b2_room_door_jet():
    def build(eng):
        tm = blank_map()
        hull_box(tm, 20, 70, 20, 85)
        hull_box(tm, 20, 70, 85, 150)
        punch_door_v(tm, 44, 46, 85)
        return tm

    def ev(eng, gmap, queue, rng):
        do_blast(eng, gmap, queue, rng, fy=45, fx=40, radius=12, pressure=9.0, wall_damage=40.0)

    return dict(
        name="B2_room_door_jet", ticks=120, engines=("old", "new"), frame_stride=1,
        build=build, events={3: ev},
        trace={"P_src_room": lambda g: float(g.atmosphere[45, 40]) / Q16,
               "P_dst_room": lambda g: float(g.atmosphere[45, 120]) / Q16},
        caption=("Blast in room A jets through a single-tile door into room B — look for "
                  "a focused jet vs. the old engine's smeared/omnidirectional leak."),
    )


def spec_b3_breach_vent():
    def build(eng):
        tm = blank_map()
        hull_box(tm, 30, 90, 30, 90)
        return tm

    def setup(eng, gmap):
        interior = (~gmap.solid) & (~gmap.is_vacuum)
        gmap.gas[eng.gases.SMOKE][interior] = q16(0.85)

    def ev(eng, gmap, queue, rng):
        for y in range(55, 60):
            gmap.destroy_wall(y, 90)

    return dict(
        name="B3_breach_vacuum_vent", ticks=150, engines=("old", "new"), frame_stride=1,
        build=build, setup=setup, events={15: ev},
        trace={"P_room": lambda g: float(g.atmosphere[60, 45]) / Q16,
               "smoke_room": lambda g: float(g.gas[1, 60, 45]) / Q16},
        caption=("THE sink_hop-vs-native comparison: a smoky sealed room breached to true "
                  "vacuum. Old engine vents via the geometric sink_hop BFS hack; new engine "
                  "vents natively off -grad(P) toward N=0. Watch how the outflow shapes."),
    )


def spec_b4_sealed_fire():
    def build(eng):
        tm = blank_map()
        hull_box(tm, 60, 100, 60, 100)
        tm[80, 80] = MAT_WOOD
        return tm

    def setup(eng, gmap):
        ignite(eng, gmap, (80, 80), intensity=0.6, temp_mult=1.5)

    return dict(
        name="B4_sealed_room_fire", ticks=260, engines=("old", "new"), frame_stride=2,
        build=build, setup=setup, events={},
        trace={"P_room": lambda g: float(g.atmosphere[70, 80]) / Q16,
               "fire_I": lambda g: float(g.fire[80, 80]) / Q16,
               "T_fire": lambda g: float(g.temperature[80, 80]) / Q16},
        caption=("Fire in a sealed room, no vent. Intended story: new engine's real O2 "
                  "depletion self-starves the fire; old engine's atmosphere-as-O2-proxy "
                  "never runs out, fire burns on. UPDATED FINDING (new engine, post "
                  "v2.4 thermal-ceiling fix + this harness's heat-clear fidelity fix, "
                  "design doc §4 v2.4): the old ceiling-pin was a harness artifact, not "
                  "a physics one -- with heat cleared per tick like the real game loop, "
                  "T_fire rises smoothly to ~1.2-1.6 kK (nowhere near the 16000 K "
                  "T_MAX_PHYS rail; 0 rail hits) and fire_I self-starves to near-zero by "
                  "t~=39 as O2 depletes, exactly as intended. It then flickers/partially "
                  "reignites through the rest of the run -- this is the SEPARATE, already-"
                  "flagged 'fuel-free smolder': hot gas conducts the wood back above "
                  "ignition_temp and CombustionSolver burns O2 without consuming wall_hp "
                  "fuel (design doc §4 v2.4, 'Remaining P5 flags' #1). Erik should see "
                  "this smolder-flicker before judging B4's feel."),
    )


def spec_b5_water_pushes_smoke():
    def build(eng):
        tm = blank_map()
        hull_box(tm, 40, 74, 40, 74)
        return tm

    def setup(eng, gmap):
        interior = (~gmap.solid) & (~gmap.is_vacuum)
        gmap.gas[eng.gases.SMOKE][interior] = q16(0.5)
        gmap.water_sources.append((43, 43, 1.8))

    return dict(
        name="B5_water_displaces_smoke", ticks=220, engines=("old", "new"), frame_stride=1,
        build=build, setup=setup, events={},
        trace={"water_far": lambda g: float(g.water_depth[60, 60]) / Q16,
               "smoke_far": lambda g: float(g.gas[1, 60, 60]) / Q16},
        caption=("A water source held in one corner floods the room; rising water "
                  "evacuates its cell's air into neighbours (new engine: the v2.1 "
                  "occupancy-transition rule, no field multiply) pushing smoke ahead "
                  "of the waterline. Old engine: the legacy atmosphere*=ratio W3 term."),
    )


def spec_b6_o2_pocket_flare():
    def build(eng):
        tm = blank_map()
        hull_box(tm, 60, 90, 30, 60)
        hull_box(tm, 60, 90, 100, 130)
        tm[75, 45] = MAT_WOOD
        tm[75, 115] = MAT_WOOD
        return tm

    def setup(eng, gmap):
        O2 = eng.gases.O2
        for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            gmap.gas[O2][75 + dy, 115 + dx] = q16(3.0)
        ignite(eng, gmap, (75, 45), intensity=0.3, temp_mult=1.05)
        ignite(eng, gmap, (75, 115), intensity=0.3, temp_mult=1.05)

    return dict(
        name="B6_o2_pocket_flare", ticks=90, engines=("new",), frame_stride=1,
        build=build, setup=setup, events={},
        trace={"fire_control": lambda g: float(g.fire[75, 45]) / Q16,
               "fire_boosted": lambda g: float(g.fire[75, 115]) / Q16,
               # sampled one tile above the wood/ignition cell (open-air, not
               # the solid wood tile itself, which never carries gas) -- the
               # boosted room's O2 spike was seeded exactly at this neighbour.
               "o2_control": lambda g: float(g.gas[5, 74, 45]) / Q16,
               "o2_boosted": lambda g: float(g.gas[5, 74, 115]) / Q16},
        caption=("New engine only (needs real N_O2): left room = ambient ignition "
                  "(control), right room = an O2-tank-rupture spike at the same "
                  "ignition — the O2-rich pocket should flare visibly hotter/brighter."),
    )


def spec_b7_thermal_spike():
    def build(eng):
        tm = blank_map()
        hull_box(tm, 70, 90, 70, 90)
        tm[80, 80] = MAT_WOOD
        return tm

    def setup(eng, gmap):
        ignite(eng, gmap, (80, 80), intensity=0.8, temp_mult=3.0)

    return dict(
        name="B7_thermal_spike_repro", ticks=100, engines=("new",), frame_stride=1,
        build=build, setup=setup, events={},
        trace={"T_peak_tile": lambda g: float(g.temperature[80, 80]) / Q16,
               "P_room": lambda g: float(g.atmosphere[75, 80]) / Q16},
        caption=("New engine only, FLAGGED (decisions.md #16): a strongly-seeded fire "
                  "in a small sealed room drives `temperature` near the Q16.16 ceiling "
                  "— pre-existing P3 plume->T-shim x compression-work coupling, "
                  "reproduced here with the exact test_eos_p4_combustion.py recipe. "
                  "Erik needs to SEE what this looks like, not just the number."),
    )


ALL_SPECS = [
    spec_b1_corridor_blast, spec_b2_room_door_jet, spec_b3_breach_vent,
    spec_b4_sealed_fire, spec_b5_water_pushes_smoke, spec_b6_o2_pocket_flare,
    spec_b7_thermal_spike,
]


# ---------------------------------------------------------------------------
# Rendering — adapted from prototypes/eos/render.py's compositing approach.
# Walls flat color, water blue tint, smoke bright veil, fire red->orange->
# yellow (driven by `gmap.fire` intensity, NOT `temperature` — `temperature`
# is not a comparable quantity between the two engines: old engine's field is
# solids-only DeltaT-above-ambient, new engine's is a unified gas+solid
# Kelvin field: see docs/eos_refactor_interaction_map.md §C), P optionally
# modulating brightness on open-air tiles.
# ---------------------------------------------------------------------------
COLOR_BG = np.array([12, 12, 16], dtype=np.float32)
COLOR_SOLID = np.array([95, 95, 105], dtype=np.float32)
COLOR_VACUUM = np.array([2, 2, 8], dtype=np.float32)
COLOR_DOOR = np.array([150, 110, 40], dtype=np.float32)
COLOR_SMOKE = np.array([235, 235, 245], dtype=np.float32)
COLOR_WATER = np.array([30, 90, 200], dtype=np.float32)
_FIRE_RED = np.array([200, 0, 0], dtype=np.float32)
_FIRE_ORANGE = np.array([255, 120, 0], dtype=np.float32)
_FIRE_YELLOW = np.array([255, 230, 60], dtype=np.float32)


def _fire_color(intensity: np.ndarray) -> np.ndarray:
    i = np.clip(intensity, 0.0, 1.0)
    t_lo = np.clip(i * 2.0, 0.0, 1.0)
    t_hi = np.clip(i * 2.0 - 1.0, 0.0, 1.0)
    color = _FIRE_RED[None, None, :] * (1 - t_lo)[..., None] + _FIRE_ORANGE[None, None, :] * t_lo[..., None]
    color = color * (1 - t_hi)[..., None] + _FIRE_YELLOW[None, None, :] * t_hi[..., None]
    return color * i[..., None]


def render_frame(gmap, crop, px, water_ref=0.5, smoke_ref=1.0) -> np.ndarray:
    y0, y1, x0, x1 = crop
    sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    solid = gmap.solid[sl]
    is_vacuum = gmap.is_vacuum[sl]
    material = gmap.material[sl]
    atmosphere = gmap.atmosphere[sl]
    water_depth = gmap.water_depth[sl]
    smoke = gmap.smoke[sl]
    fire = gmap.fire[sl]

    h, w = solid.shape
    rgb = np.empty((h, w, 3), dtype=np.float32)
    rgb[:] = COLOR_BG
    rgb[is_vacuum] = COLOR_VACUUM
    rgb[solid] = COLOR_SOLID
    door = material == MAT_DOOR
    rgb[door] = COLOR_DOOR

    open_air = (~solid) & (~is_vacuum)

    # P-as-brightness: modulate open-air tiles by pressure deviation from 1 atm.
    P = atmosphere.astype(np.float64) / Q16
    bright = np.clip(1.0 + (P - 1.0) * 1.2, 0.55, 1.7)
    idx = open_air
    rgb[idx] = rgb[idx] * bright[idx][:, None].astype(np.float32)

    water_frac = np.clip(water_depth.astype(np.float64) / Q16 / water_ref, 0.0, 1.0)
    rgb += (COLOR_WATER * water_frac[..., None].astype(np.float32)) * open_air[..., None]

    smoke_frac = np.clip(smoke.astype(np.float64) / Q16 / smoke_ref, 0.0, 1.0)
    rgb += (COLOR_SMOKE * smoke_frac[..., None].astype(np.float32)) * open_air[..., None]

    fire_i = fire.astype(np.float64) / Q16
    rgb += _fire_color(fire_i).astype(np.float32) * open_air[..., None]

    frame = np.clip(rgb, 0, 255).astype(np.uint8)
    frame = np.repeat(np.repeat(frame, px, axis=0), px, axis=1)
    return frame


def adaptive_px(crop, target=460, lo=3, hi=10):
    y0, y1, x0, x1 = crop
    span = max(y1 - y0 + 1, x1 - x0 + 1)
    return int(min(hi, max(lo, round(target / max(1, span)))))


# ---------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------
def run_scenario(eng, spec, seed_dt=1.0 / 24):
    tm = spec["build"](eng)
    gmap = new_gamemap(eng, tm, name=spec["name"])
    pr = eng.PhysicsRunner(eng.bp)
    rng = np.random.default_rng(20260710)
    queue = eng.EditQueue()

    crop = content_bbox(tm)
    px = adaptive_px(crop)

    if "setup" in spec:
        spec["setup"](eng, gmap)

    frames = []
    trace = {k: [] for k in spec.get("trace", {})}
    tick_ms = []
    events = spec.get("events", {})

    frames.append(render_frame(gmap, crop, px))
    for t in range(spec["ticks"]):
        if t in events:
            events[t](eng, gmap, queue, rng)
        t0 = time.perf_counter()
        burned = pr.step(gmap, seed_dt)
        for (y, x) in burned:
            gmap.destroy_wall(y, x)
        gmap.heat.fill(0)   # game-loop fidelity: Simulation.step clears the per-tick
                            # heat deposit after its readers (v2.4 fix-branch finding —
                            # without this, stale heat re-radiates every tick and T
                            # pins at the rail, a harness artifact not a physics one)
        tick_ms.append((time.perf_counter() - t0) * 1000.0)
        if t % spec["frame_stride"] == 0:
            frames.append(render_frame(gmap, crop, px))
        for k, fn in spec.get("trace", {}).items():
            trace[k].append(float(fn(gmap)))

    return frames, trace, tick_ms


def cmd_run(args):
    eng = load_engine(Path(args.engine_root))
    raw_dir = Path(args.raw_dir) / args.tag
    raw_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.scenarios.split(",")) if args.scenarios else None
    ran = []
    for spec_fn in ALL_SPECS:
        spec = spec_fn()
        short = spec["name"].split("_")[0].lower()
        if wanted and short not in wanted and spec["name"] not in wanted:
            continue
        if args.tag not in spec["engines"]:
            print(f"[skip] {spec['name']} not applicable to engine tag {args.tag!r}")
            continue
        print(f"[run ] {spec['name']} on tag={args.tag} ({spec['ticks']} ticks)...")
        t0 = time.perf_counter()
        frames, trace, tick_ms = run_scenario(eng, spec)
        wall_s = time.perf_counter() - t0
        print(f"       done in {wall_s:.2f}s wall, {len(frames)} frames, "
              f"tick p50/p97/max = {np.percentile(tick_ms, 50):.3f}/"
              f"{np.percentile(tick_ms, 97):.3f}/{max(tick_ms):.3f} ms")

        import imageio.v2 as imageio
        imageio.mimsave(str(raw_dir / f"{spec['name']}.gif"), frames, fps=20)
        (raw_dir / f"{spec['name']}_trace.json").write_text(json.dumps(trace))
        (raw_dir / f"{spec['name']}_timing.json").write_text(json.dumps({
            "tick_ms": tick_ms,
            "p50": float(np.percentile(tick_ms, 50)),
            "p97": float(np.percentile(tick_ms, 97)),
            "max": float(max(tick_ms)),
            "n_ticks": len(tick_ms),
            "caption": spec["caption"],
        }))
        ran.append(spec["name"])
    print(f"[done] tag={args.tag}: {len(ran)} scenarios -> {raw_dir}")


def cmd_determinism(args):
    eng = load_engine(Path(args.engine_root))
    spec = spec_b3_breach_vent()

    def digest_of(n_ticks):
        tm = spec["build"](eng)
        gmap = new_gamemap(eng, tm, name="determinism")
        pr = eng.PhysicsRunner(eng.bp)
        rng = np.random.default_rng(20260710)
        queue = eng.EditQueue()
        spec["setup"](eng, gmap)
        h = hashlib.sha256()
        for t in range(n_ticks):
            if t in spec["events"]:
                spec["events"][t](eng, gmap, queue, rng)
            burned = pr.step(gmap, 1.0 / 24)
            for (y, x) in burned:
                gmap.destroy_wall(y, x)
            gmap.heat.fill(0)   # game-loop fidelity (see main loop note)
            for arr in (gmap.gas, gmap.temperature, gmap.fire, gmap.wall_hp,
                        gmap.atmosphere, gmap.wind_x, gmap.wind_y):
                h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    n = args.ticks or spec["ticks"]
    d1 = digest_of(n)
    d2 = digest_of(n)
    ok = d1 == d2
    out = Path(args.raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"scenario": spec["name"], "ticks": n, "digest1": d1, "digest2": d2, "match": ok}
    (out / "determinism_check.json").write_text(json.dumps(result, indent=2))
    print(f"[determinism] {spec['name']} x{n} ticks: {'MATCH' if ok else 'MISMATCH'}")
    print(json.dumps(result, indent=2))
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Compose — pure post-processing, no simulation imports.
# ---------------------------------------------------------------------------
def cmd_compose(args):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = Path(args.raw_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    specs = {s["name"]: s for s in (fn() for fn in ALL_SPECS)}
    rows = []  # for index.html

    all_timing = {"old": {}, "new": {}}
    for tag in ("old", "new"):
        d = raw / tag
        if not d.exists():
            continue
        for f in d.glob("*_timing.json"):
            name = f.name[: -len("_timing.json")]
            all_timing[tag][name] = json.loads(f.read_text())

    for name, spec in specs.items():
        gif_paths = {}
        for tag in ("old", "new"):
            p = raw / tag / f"{name}.gif"
            if p.exists():
                gif_paths[tag] = p
        if not gif_paths:
            continue

        if "old" in gif_paths and "new" in gif_paths:
            f_old = list(imageio.mimread(str(gif_paths["old"])))
            f_new = list(imageio.mimread(str(gif_paths["new"])))
            n = min(len(f_old), len(f_new))
            combined = []
            gap = np.zeros((f_old[0].shape[0], 4, 3), dtype=np.uint8) + 40
            for i in range(n):
                a = f_old[i][..., :3]
                b = f_new[i][..., :3]
                combined.append(np.concatenate([a, gap, b], axis=1))
            out_gif = out / f"{name}.gif"
            imageio.mimsave(str(out_gif), combined, fps=20)
            layout = "old | new"
        else:
            tag = next(iter(gif_paths))
            frames = list(imageio.mimread(str(gif_paths[tag])))
            out_gif = out / f"{name}.gif"
            imageio.mimsave(str(out_gif), [f[..., :3] for f in frames], fps=20)
            layout = f"{tag} only"

        # Trace plot (B4 pressure-vs-time, B6 O2/fire traces, others skipped
        # unless they carry a trace dict with >1 series worth plotting).
        plot_path = None
        trace_data = {}
        for tag in ("old", "new"):
            tp = raw / tag / f"{name}_trace.json"
            if tp.exists():
                trace_data[tag] = json.loads(tp.read_text())
        if trace_data:
            # One subplot per trace KEY (not one shared axis) -- series with
            # wildly different scales (e.g. B4's T_fire in Kelvin next to
            # fire_I in [0,1]) would otherwise squash the small ones flat.
            keys = []
            for series in trace_data.values():
                for k in series:
                    if k not in keys:
                        keys.append(k)
            ncols = len(keys)
            fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 2.6), dpi=130, squeeze=False)
            axes = axes[0]
            for ax, key in zip(axes, keys):
                for tag, series in trace_data.items():
                    if key in series:
                        ax.plot(series[key], label=tag, linewidth=1.2)
                ax.set_title(key, fontsize=8)
                ax.set_xlabel("tick", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.legend(fontsize=6)
            fig.suptitle(name, fontsize=9)
            fig.tight_layout()
            plot_path = out / f"{name}_trace.png"
            fig.savefig(plot_path)
            plt.close(fig)

        rows.append(dict(
            name=name, gif=out_gif.name, layout=layout, caption=spec["caption"],
            plot=plot_path.name if plot_path else None,
        ))

    # --- cost table -----------------------------------------------------
    cost_rows = []
    for tag in ("old", "new"):
        for name, t in all_timing[tag].items():
            cost_rows.append((tag, name, t["p50"], t["p97"], t["max"], t["n_ticks"]))
    agg = {}
    for tag in ("old", "new"):
        all_ticks = []
        for t in all_timing[tag].values():
            all_ticks.extend(t["tick_ms"])
        if all_ticks:
            agg[tag] = (float(np.percentile(all_ticks, 50)),
                        float(np.percentile(all_ticks, 97)),
                        float(max(all_ticks)), len(all_ticks))

    determinism = None
    det_path = raw / "determinism_check.json"
    if det_path.exists():
        determinism = json.loads(det_path.read_text())

    # --- §9 tuning dial list (LISTED only, per task instructions — not tuned) ---
    dials = [
        ("gamma / K / c_amb", "[physics.eos]", "adiabatic index (1.4, compile-time), the one unit-bridge constant K = c_amb^2/gamma, ambient sound speed (SET 300 m/s)"),
        ("k_push + knockdown thresholds", "[exchange] k_push=400.0", "unit shockwave impulse, vs the new transient grad(P) scale"),
        ("k_p (water head)", "[physics.water] k_p=0.5", "water pressure-head coefficient, recalibrated for integer P"),
        ("air conductivity", "[materials.air] conductivity", "small-but-nonzero: big enough the solid<->gas interface sink fires, small enough air doesn't become the heat-advecting field"),
        ("combustion constants", "[physics.combustion]", "burn_rate=1.0, H_fuel=4.0, soot_yield=0.3, o2_thresh_burn=0.03, o2_thresh_breathe=0.08 (unwired)"),
        ("cool_shift_vacuum", "[physics.thermal] COOL_SHIFT_VACUUM=3", "hull radiate-to-space rate under the real energy path"),
        ("CFL_ADV / N_SUB_MAX", "[physics.eos]", "CFL_ADV=0.5 (pinned constraint), N_SUB_MAX=8 (re-pinned from 16 at the P3 gate)"),
        ("dyn_wave_absorb / absorb_strength", "[physics.eos] absorb_strength=8.0", "unit/material shockwave absorption; now also locally damps smoke-carrying wind near units (named P5 feel item)"),
        ("trace_mass_scale / trace advection", "engine-owned, [physics] advection_rate now dead", "trace_mass_scale=0.02 (opacity not molar density); wind_diffusion_scale disabled pending this P5 feel pass"),
    ]

    html = _render_index_html(rows, cost_rows, agg, determinism, dials)
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"[compose] wrote {out / 'index.html'} ({len(rows)} scenarios)")


def _render_index_html(rows, cost_rows, agg, determinism, dials):
    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    scenario_html = []
    for r in rows:
        plot_html = f'<img src="{esc(r["plot"])}" class="trace-plot">' if r["plot"] else ""
        scenario_html.append(f"""
        <section class="scenario">
          <h2>{esc(r['name'])} <span class="layout">({esc(r['layout'])})</span></h2>
          <p class="caption">{esc(r['caption'])}</p>
          <img src="{esc(r['gif'])}" class="gif">
          {plot_html}
        </section>""")

    cost_table_rows = "\n".join(
        f"<tr><td>{esc(tag)}</td><td>{esc(name)}</td><td>{p50:.3f}</td>"
        f"<td>{p97:.3f}</td><td>{mx:.3f}</td><td>{n}</td></tr>"
        for (tag, name, p50, p97, mx, n) in sorted(cost_rows)
    )
    agg_rows = "\n".join(
        f"<tr><td colspan=2><b>{esc(tag)} — ALL SCENARIOS COMBINED</b></td>"
        f"<td>{v[0]:.3f}</td><td>{v[1]:.3f}</td><td>{v[2]:.3f}</td><td>{v[3]}</td></tr>"
        for tag, v in agg.items()
    )

    det_html = "no determinism check found"
    if determinism is not None:
        status = "MATCH (deterministic)" if determinism["match"] else "MISMATCH -- INVESTIGATE"
        det_html = (f"scenario={esc(determinism['scenario'])}, ticks={determinism['ticks']}: "
                    f"<b>{status}</b>")

    dial_rows = "\n".join(
        f"<tr><td>{esc(n)}</td><td><code>{esc(loc)}</code></td><td>{esc(desc)}</td></tr>"
        for (n, loc, desc) in dials
    )

    # --- provisional-items banner (v2.4 as-built amendment, design doc §4 —
    # everything landed on this rebuild that Erik has not yet blessed) -------
    provisional = [
        ("T_MAX_PHYS / U_MAX counted rails",
         "cpp/src/eos_solver.h / eos_solver.cpp (config.toml [physics.thermal] T_MAX_PHYS=16000, [physics.eos] U_MAX=1000)",
         "Defense-in-depth backstops on temperature (step 4c, thermal Pass 1, combustion deposit) and velocity "
         "(step-4 store clamp). t_max_phys_hits and u_max_hits (the rail-specific counters, checked "
         "post-hoc across all 7 scenarios on this bake) read 0 everywhere -- pure backstops, never engaged. "
         "(The general |u|<=c_LOCAL clamp counter, a separate normal-physics stat, does fire in B3 as expected.)"),
        ("Absorption-proportional radiant deposit",
         "cpp/src/temperature_solver.cpp (~L239-274, v2.4 ABSORPTION-PROPORTIONAL comment block)",
         "Gas absorbs the fire's radiant heat deposit in proportion to its own density (optically-thin form, "
         "this project's own engine/05 optics model applied to the heat channel) instead of the full ray energy "
         "regardless of how thin the gas is — bit-identical to the old path at/above ambient density."),
        ("O2-gate hot-zone rescale",
         "config.toml [physics.combustion] P_min/P_full (0.126/0.21 -> 0.01/0.03), ignition o2_threshold (0.12 -> 0.01)",
         "Second half of the P4 O2-proxy-to-real-N_O2 rescale, needed because a fire's own thermal expansion "
         "evacuates local O2 density even at the flame edge. Restores strong O2 differentiation "
         "(sealed 172 / vented 49 / flooded 39 ticks, e2e trio), perturbation-stable."),
        ("T_FLAME_MAX shim limiter",
         "cpp/src/fire_simulation.h/.cpp (T_FLAME_MAX=2000.0f default)",
         "Fire's own plume-heating self-limiter, now correctly gated on T (was structurally dead, gating on "
         "atmosphere at the fire's own solid tile, which the solver force-zeroes) — tapers the deposit to "
         "nothing as the plume approaches T_FLAME_MAX."),
    ]
    provisional_rows = "\n".join(
        f"<tr><td>{esc(n)}</td><td><code>{esc(loc)}</code></td><td>{esc(desc)}</td></tr>"
        for (n, loc, desc) in provisional
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>EOS P5 bake-off — old vs new engine</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem;
        background: #111; color: #eee; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.05rem; margin-bottom: 0.2rem; }}
.layout {{ font-weight: normal; color: #999; font-size: 0.8rem; }}
.caption {{ color: #bbb; font-size: 0.9rem; max-width: 800px; }}
.gif {{ max-width: 100%; border: 1px solid #444; image-rendering: pixelated; }}
.trace-plot {{ display: block; margin-top: 0.3rem; max-width: 100%; background: #fff; }}
section.scenario {{ margin-bottom: 2.5rem; border-bottom: 1px solid #333; padding-bottom: 1.5rem; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; font-size: 0.85rem; }}
td, th {{ border: 1px solid #444; padding: 0.3rem 0.6rem; text-align: right; }}
td:first-child, td:nth-child(2), th:first-child, th:nth-child(2) {{ text-align: left; }}
code {{ color: #9cf; }}
.provisional {{ border: 1px solid #a86; background: #2a2117; padding: 0.9rem 1.1rem; border-radius: 6px;
                 margin-bottom: 1.5rem; }}
.provisional h2 {{ margin-top: 0; color: #e8b96a; }}
.provisional table {{ margin-bottom: 0; }}
</style></head>
<body>
<h1>EOS refactor — P5 combined-system bake-off</h1>

<div class="provisional">
<h2>Provisional items for Erik's review</h2>
<p class="caption">Landed on this rebuild (v2.4 as-built amendment, design doc §4), not yet Erik-blessed.
None of these are physics changes to this bake harness itself — they are engine-side commits this bake
observes. See design doc §4 v2.4 for the full derivation.</p>
<table>
<tr><th>item</th><th>where it lives</th><th>what it does</th></tr>
{provisional_rows}
</table>
</div>

<p>Pre-refactor engine (commit b17e150) vs NEW engine (main, P1-P4 merged), same synthetic
160&sup2; scenarios, real C++ engine both sides (no numpy prototype). Erik's eyes are the
gate — nothing here has been tuned.</p>

<h2>Determinism spot-check (new engine, two-run digest)</h2>
<p>{det_html}</p>

<h2>Cost table (ms/tick)</h2>
<table>
<tr><th>engine</th><th>scenario</th><th>p50</th><th>p97</th><th>max</th><th>n_ticks</th></tr>
{cost_table_rows}
{agg_rows}
</table>

<h2>&sect;9 tuning dials (LISTED, not touched here — Erik's feel pass)</h2>
<table>
<tr><th>dial</th><th>where</th><th>what it does</th></tr>
{dial_rows}
</table>

<h2>Scenarios</h2>
{"".join(scenario_html)}
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--engine-root", required=True)
    p_run.add_argument("--tag", required=True, choices=["old", "new"])
    p_run.add_argument("--raw-dir", required=True)
    p_run.add_argument("--scenarios", default=None, help="comma list, e.g. b1,b3,b4")
    p_run.set_defaults(func=cmd_run)

    p_det = sub.add_parser("determinism")
    p_det.add_argument("--engine-root", required=True)
    p_det.add_argument("--raw-dir", required=True)
    p_det.add_argument("--ticks", type=int, default=None)
    p_det.set_defaults(func=cmd_determinism)

    p_comp = sub.add_parser("compose")
    p_comp.add_argument("--raw-dir", required=True)
    p_comp.add_argument("--out", required=True)
    p_comp.set_defaults(func=cmd_compose)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
