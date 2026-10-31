"""Creature scenarios (#106 P6d2) -- the P2-P5 runs of the octopus prototypes
(``prototypes/octopus_motion/scenarios3d.py``) re-run THROUGH the shared
``Simulation`` on a flooded room built with ``level_lib`` (water everywhere:
the arm chain's v1 assumption), driven only by the unit's intents (MOVE_DIR,
AIM, USE) -- the way the ``creature`` match controller drives it.

A harness (``_*``), not a test: ``tests/test_creature_primitives.py`` gates
on what these return, ``tools/octopus_clips.py`` draws them. Every run
returns a :class:`Run` -- the per-tick decoded ``__creature__`` record of the
one octopus (``creature_system.decode_record``: the bytes the digest hashes)
and the real-unit series the gates read (crown, nodes, states, the prop),
dequantized ONCE here (tests and tools only).

Scenario set-ups that write a body's state (the swim's start in mid water,
``lift``) do it before the first tick, as P6d1's limp gate does: a scenario
is a fixture, never a primitive.
"""
from __future__ import annotations

import contextlib
import copy
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TILE_M = 1.0 / 3.0
TPS = 24
DT = 1.0 / TPS


# ---------------------------------------------------------------------------
# The room
# ---------------------------------------------------------------------------
def write_room(root: Path, name: str, *, w: int, h: int, creatures=(), flood_m: float = 2.0,
               rooms=(), match=None, walls=()):
    """A hull box of (w, h) tiles at the game's tile, flooded everywhere
    inside, with extra wall tiles, ``[[creature]]`` rows, spawn rooms and a
    match -- every file through level_lib."""
    import level_lib
    import level_loader
    from PIL import Image
    from simulation import water_fixed
    from simulation.materials import MAT_AIR, MAT_HULL
    lvl = Path(root) / name
    lvl.mkdir(parents=True)
    level_lib.write_level_header(lvl, name=name, tile_size_m=TILE_M)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    for (x, y) in walls:
        tm[y, x] = MAT_HULL
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    depth = np.where(tm == MAT_AIR, water_fixed.quantize_scalar(flood_m), 0).astype(np.int32)
    level_lib.write_water(lvl, depth)
    if creatures:
        level_lib.write_creatures(lvl / "level.toml", list(creatures))
    if rooms:
        level_lib.write_spawn_rooms(lvl / "level.toml", list(rooms))
    if match is not None:
        level_lib.write_match(lvl, match)
    return level_loader.load(name, str(root))


@contextlib.contextmanager
def row_override(**groups):
    """Run with a modified ``[creatures.octopus]`` row (a CONTROL, e.g. the
    clocked crawl): ``groups`` maps a group name to ``{key: value}``. The
    shared creature table is rebuilt through the door from the modified raw
    row and restored afterwards."""
    from config import CFG
    from simulation import creatures as C
    raw = C._as_dict(CFG.creatures, "creatures")
    row = copy.deepcopy(C._canonical(raw["octopus"]))
    for g, kv in groups.items():
        row[g].update(kv)
    old = C._TABLE
    t = C.CreatureTable({"octopus": row}, CFG.clock.ticks_per_second)
    C._TABLE = t
    try:
        yield t.get("octopus")
    finally:
        C._TABLE = old


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------
@dataclass
class Run:
    name: str
    records: list = field(default_factory=list)      # decoded per tick
    views: list = field(default_factory=list)        # creature_system.read_chain per tick (P7)
    raw: list = field(default_factory=list)          # the record bytes per tick
    t: np.ndarray = None                             # s, after each tick
    X: np.ndarray = None                             # (T, 3) crown, m
    TH: np.ndarray = None                            # (T,) the pose heading, rad (screen CCW)
    P: np.ndarray = None                             # (T, A, n, 3) nodes, m
    S: np.ndarray = None                             # (T, A) arm states
    PR: np.ndarray = None                            # (T, 3) the prop, m (or None)
    NG: np.ndarray = None                            # (T,) seals held
    pops: np.ndarray = None                          # (T,) anchor pops this tick
    meta: dict = field(default_factory=dict)

    def finish(self) -> "Run":
        from simulation import creature_fixed as CF
        one = float(CF.CHAIN_ONE)
        R = self.records
        self.t = (np.arange(len(R)) + 1) * DT
        self.X = np.array([r["body"][:3] for r in R], dtype=float) / one
        self.TH = np.array([r["body"][3] for r in R], dtype=float) / one
        self.P = np.array([r["p"] for r in R], dtype=float) / one
        self.S = np.array([r["arm_state"] for r in R], dtype=np.int64)
        if R and R[0]["bodies"]:
            self.PR = np.array([r["bodies"][0][:3] for r in R], dtype=float) / one
        self.NG = np.array([len(r["grabs"]) for r in R], dtype=np.int64)
        return self


class Scenario:
    """One flooded room (``size`` square, or ``w`` x ``h`` tiles) with one
    octopus (its footprint's top-left at ``at``, default the centre; heading 0
    = East), stepped tick by tick under ContinuousRealtime."""

    def __init__(self, name: str, *, size: int = 90, w: int | None = None, h: int | None = None,
                 at=None, seed: int = 7, heading: float = 0.0, tmp: Path | None = None,
                 engine=None):
        import breach_physics as bp
        from simulation import Simulation
        from simulation.match import CreatureSpawn
        from simulation.ruleset import ContinuousRealtime
        tmp = Path(tempfile.mkdtemp(prefix="creature_")) if tmp is None else Path(tmp)
        w = size if w is None else w
        h = size if h is None else h
        x, y = (w / 2.0 - 1.5, h / 2.0 - 1.5) if at is None else at
        self.level = write_room(tmp, name, w=w, h=h,
                                creatures=(CreatureSpawn("octopus", x, y, heading=heading),))
        self.sim = Simulation(self.level, seed=seed, breach_physics=engine or bp,
                              enable_recorder=False, ruleset=ContinuousRealtime())
        (self.unit,) = self.sim.spawn_creatures(self.level.creatures)
        self.body = self.sim._creature_system.bodies[self.unit.id]
        self.run = Run(name)

    # -- fixtures ---------------------------------------------------------------
    def lift(self, z_m: float) -> None:
        """Raise the crown and every node by ``z_m`` (a scenario's start in
        mid water, before the first tick; the seated anchors go too)."""
        from simulation import arm_chain as AC
        from simulation import creature_fixed as CF
        b = self.body
        b.bufs.anchored[...] = 0
        dz = CF.chain_from_m(z_m)
        b.bufs.p[0, ..., 2] += dz
        b.bufs.body[0, 0, AC.BS_Z] += dz

    # -- intents ----------------------------------------------------------------
    def speed_of(self, kind: str) -> int:
        """The analog speed (Q16) that asks for a P6d2 gait EXACTLY (#106 P6f:
        each band's top is that gait at tempo ONE): ``crawl`` = the row's
        ``crawl_until``, ``cruise`` = ``swim_until`` (the umbrella), ``escape``
        = ONE (the jet train)."""
        return gait_speed_q16(self.body, kind)

    def move(self, deg: float, speed=None) -> None:
        """MOVE_DIR toward ``deg`` (screen CCW from East, the facing
        convention: North = -y) at ``speed``: a gait name (``crawl`` /
        ``cruise`` / ``escape``, :meth:`speed_of`) or a Q16 analog speed;
        None = the crawl (the P2 gait at its full tempo)."""
        from control_source import quantize_stick_direction
        r = np.radians(deg)
        dx, dy, _ = quantize_stick_direction(float(np.cos(r)), float(-np.sin(r)), deadzone=0.0)
        if speed is None:
            speed = "crawl"
        sq = self.speed_of(speed) if isinstance(speed, str) else int(speed)
        self.sim.set_move_dir(self.unit.id, dx, dy, 0, speed_q=sq)

    def stop(self) -> None:
        self.sim.clear_move_dir(self.unit.id)

    def aim(self, deg: float | None) -> None:
        from control_source import quantize_stick_direction
        if deg is None:
            self.unit.live_aim = None
            return
        r = np.radians(deg)
        dx, dy, _ = quantize_stick_direction(float(np.cos(r)), float(-np.sin(r)), deadzone=0.0)
        self.sim.set_aim(self.unit.id, dx, dy)

    def face(self, deg: float | None) -> None:
        """The FACING channel (#106 P6g): an AIM toward ``deg`` (screen CCW
        from East) WITHOUT a reach -- the keyboard cursor's / the right
        stick's aim; the body turns toward it while it crawls or idles.
        None drops the aim."""
        from control_source import quantize_stick_direction
        if deg is None:
            self.unit.live_aim = None
            return
        r = np.radians(deg)
        dx, dy, _ = quantize_stick_direction(float(np.cos(r)), float(-np.sin(r)), deadzone=0.0)
        self.sim.set_aim(self.unit.id, dx, dy, reach=False)

    def use(self) -> None:
        self.sim.use_intent(self.unit.id)

    # -- stepping ---------------------------------------------------------------
    def step(self, n: int = 1, each=None) -> None:
        from simulation import creature_system as CS
        for _ in range(int(n)):
            if each is not None:
                each(self)
            self.sim.set_paused(False)
            self.sim.step()
            # the record the digest hashes (the carrier's serializer)
            rec = CS.serialize_record(self.body)
            self.run.raw.append(rec)
            d = CS.decode_record(rec)
            d["popped"] = int(self.body.bufs.popped.sum())
            self.run.records.append(d)
            # THE chain accessor's view of the same tick (#106 P7)
            self.run.views.append(CS.read_chain(self.body, self.unit))

    def seconds(self, s: float, each=None) -> None:
        self.step(int(round(s * TPS)), each)

    def finish(self) -> Run:
        r = self.run.finish()
        r.pops = np.array([d["popped"] for d in r.records], dtype=np.int64)
        return r


def gait_speed_q16(body, kind: str) -> int:
    """The Q16 analog speed at the TOP of a gait's band (#106 P6f), from the
    body's quantized dials: what the P6d2 drivers' speed modes became."""
    d = body.dials
    return {"crawl": d.crawl_until, "cruise": d.swim_until, "escape": 1 << 16}[kind]


# ---------------------------------------------------------------------------
# The scenarios (scenarios3d's, through the Simulation)
# ---------------------------------------------------------------------------
def crawl(direction_deg: float, seconds: float, *, seed: int = 7, size: int = 110,
          w: int | None = None, h: int | None = None, at=None,
          name: str | None = None) -> Run:
    """P2: crawl toward ``direction_deg`` (MOVE_DIR every tick) from the
    seated start (the room centre, or ``at``). The prototypes crawled on an
    endless deck: size the room so the run never reaches a wall."""
    sc = Scenario(name or f"crawl_{direction_deg:+.0f}_{seed}", size=size, w=w, h=h, at=at,
                  seed=seed)
    sc.seconds(seconds, each=lambda s: s.move(direction_deg))
    r = sc.finish()
    r.meta.update(direction_deg=direction_deg, seed=seed)
    return r


def swim(kind: str, seconds: float, *, z0: float = 8.0, w: int = 150, h: int = 40,
         seed: int = 7, linear_v_ref: float | None = None, name: str | None = None,
         before=()) -> Run:
    """P3: the umbrella (``cruise``: MOVE_DIR at the speed ``swim_until``) or
    the jet train (``escape``: speed ONE) along the heading (East), from mid water
    (P5z's z0 = 8 m; the swim does not hold depth -- reported), starting near
    the west wall of a long room. ``linear_v_ref`` (m/s): P3's CONTROL -- the
    linear isotropic drag on every arm and on the crown, under which the
    out-and-back stroke nets nothing. ``before``: (kind, seconds) phases run
    first (``meta["t_main"]`` is when the main phase starts)."""
    from simulation import arm_chain as AC
    from simulation import creature_fixed as CF
    sc = Scenario(name or f"swim_{kind}", w=w, h=h, at=(12.0, h / 2.0 - 1.5), seed=seed)
    if linear_v_ref is not None:
        b = sc.body
        b.bufs.arm_cfg[0, :, AC.AC_DRAG_MODE] = AC.DRAG_LINEAR
        b.bufs.arm_cfg[0, :, AC.AC_VREF_Q24] = CF.chain_from_m(linear_v_ref)
        for cfg in (b.body_cfg_swim, b.body_cfg_crawl):
            cfg[AC.BC_DRAG_MODE] = AC.DRAG_LINEAR
            cfg[AC.BC_VREF_Q24] = CF.chain_from_m(linear_v_ref)
    sc.lift(z0)
    for k, sec in before:
        sc.seconds(sec, each=lambda s, k=k: s.move(0.0, k))
    t_main = len(sc.run.records) * DT
    sc.seconds(seconds, each=lambda s: s.move(0.0, kind))
    r = sc.finish()
    r.meta.update(kind=kind, z0=z0, t_main=t_main)
    return r


def grab(*, crawl_s: float = 15.0, escape_s: float = 6.0, seed: int = 3, size: int = 110) -> Run:
    """P4 + the P5z lift: crawl East 0.5 s, USE (the grab: reach, wrap, hold,
    lift), keep crawling until the arm carries, carry ``crawl_s`` s, then
    the escape (the carry jets) for ``escape_s`` s -- scenarios3d.p4_grab."""
    from simulation import creature_primitives as P
    sc = Scenario("grab", size=size, seed=seed)
    sc.body.add_prop()
    sc.seconds(0.5, each=lambda s: s.move(0.0))
    sc.use()
    a = sc.body.dials.grab_arm
    n = 0
    while int(sc.body.arm_state[a]) != P.ARM_CARRY and n < 8 * TPS:
        sc.step(1, each=lambda s: s.move(0.0))
        n += 1
    t_carry = len(sc.run.records) * DT
    sc.seconds(crawl_s, each=lambda s: s.move(0.0))
    t_jets = len(sc.run.records) * DT
    sc.seconds(escape_s, each=lambda s: s.move(0.0, "escape"))
    r = sc.finish()
    r.meta.update(t_carry=t_carry, t_jets=t_jets, arm=a, grab_pops=sc.body.grab_pops)
    return r


def idle(seconds: float, *, seed: int = 5, size: int = 60) -> Run:
    """P5: no intent at all -- the seated animal explores."""
    sc = Scenario("idle", size=size, seed=seed)
    sc.seconds(seconds)
    return sc.finish()


def reach(aim_deg: float, seconds: float, *, seed: int = 7, size: int = 60) -> Run:
    """P1 through the hub: AIM toward ``aim_deg`` with no travel goal."""
    sc = Scenario("reach", size=size, seed=seed)
    sc.seconds(seconds, each=lambda s: s.aim(aim_deg))
    r = sc.finish()
    r.meta.update(aim_deg=aim_deg)
    return r


__all__ = ["TILE_M", "TPS", "DT", "write_room", "row_override", "Run", "Scenario",
           "gait_speed_q16", "crawl", "swim", "grab", "idle", "reach"]
