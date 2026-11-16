"""#29 P6b.3 -- THE SIGNALS GATE: the signal bus, the sensors, the logic
nodes, the doors' bus inputs and the pumps on the device (residency design
4.4c, the cut table's P6b.3 row; cpp/src/cuda_unit_signals.cuh, the applier's
EMIT / LOGIC / DOOR_INPUTS / PUMPS / SWAP stages).

PART TABLES -- the eight signal tables (the bus, the sensors and their disc
tiles, the nodes and their inputs, the pumps and their inputs, the door
drives) ARE the sim's: unpack(records) is the runtime row for row, the bus's
digest rows are SignalBus.digest_rows, a node's / pump's entity rows are the
serializer's, the landing writes the runtimes back, on the signal scene and
on airlock_demo. (P6b.3 adds no Device tables row: the every-row gate has
nothing new to read.)

PART SAMPLE -- the accessor's RAW sample on the device (the EMIT stage over
synthetic sensor tables) equals EntityFieldAccessor.sample / .area for every
channel on four maps whose planes are randomized over each channel's whole
range (both sides of every trace floor, negatives, every solid mask of a
disc, a disc with no open tile); the clock over random periods and ticks;
motion plates over random unit placements (dead, factions, footprints, sight
through walls and grilles) equal _MotionSensorRuntime.evaluate.

PART LOGIC -- every predicate code around its threshold, the gates' empty and
mixed inputs, require_alive both ways, the filter over long random inputs
(negatives: the floor) at every k, equal to logic_nodes' evaluators step by
step, the node tables carried on the device.

PART AIRLOCK -- eight airlock controllers (hull / interior, far outer /
inner, equalize +1 / -1 / 0) driven by random inputs from random states:
every transition of the state machine taken, the dwell, the outputs equal.

PART REFUSAL -- a filter at its proved bound equal to the host; one past it
(ema, input, k, each sign) refused by name with nothing of the env written.

PART SCENE -- the SHADOW LOCKSTEP of 1b, 9e(a) signals, 9e(b) logic, 9e(c)
door inputs, 9e(d) pumps, 9e(d) doors and 9e(e) swap on the signal scene
(two airlocks cycling fully both ways, hull and interior, every sensor
channel, motion plates seen / behind a wall / behind a grille, every node
class, a chain five deep, two wires into one input, a pump against a hull
port and one whose source runs dry, a door driven by the bus and used by a
unit on the same tick, a sensor tile destroyed and one sealed mid-run) and on
airlock_demo (cycled by a fixture): the bus, every table, the planes and
the books equal every tick, the tables' IDENTITY every tick; a census of the
Python law's branches, each required.

PART BATCH -- N = 2 / 8 scene variants (node count, wiring, layout, seed,
boundary, an env with NO bus) in one launch, env i == env i alone.
PART PADDING / OVERFLOW -- garbage past every count neither read nor
written; every table past its capacity refused by name.
PART CHAINED -- 9e(a) (b) (c), pumps, vents, doors and (e) replaced by the
device's in a running sim (the scene, airlock_demo): unit hash, field digest,
planes, books, the tables, the entity section and the signal rows equal an
untouched reference every tick, the host laws run 0 times in the twin;
ship_demo's bots match for 500 ticks (no bus: the signal chains idle).

BREAKS IF any of those differs from the law. Prints per-part lines and
``UNIT_SIGNALS_RESULT: PASS`` / ``FAIL``; ``bench()`` prints the TIMING table.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_lockstep import (ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep)
from simulation import (applier, object_tables as ot, unit_kernels as uk,  # noqa: E402
                        unit_schema as us, unit_store, unit_tables)
from simulation.gases import N_GASES, O2, SMOKE  # noqa: E402
from simulation.materials import MAT_AIR, MAT_GRILLE, MAT_HULL, MAT_WOOD  # noqa: E402

FAILS: list = []
ONE = 65536
SPACE = level_loader.SPACE_CODE
SCENE_TICKS = int(os.environ.get("UNIT_SIGNALS_SCENE_TICKS", "320"))
BATCH_TICKS = int(os.environ.get("UNIT_SIGNALS_BATCH_TICKS", "120"))
LEVEL_TICKS = int(os.environ.get("UNIT_SIGNALS_LEVEL_TICKS", "260"))
SHIP_TICKS = int(os.environ.get("UNIT_SIGNALS_SHIP_TICKS", "500"))
CHAIN_TICKS = int(os.environ.get("UNIT_SIGNALS_CHAIN_TICKS", "320"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_SIGNALS_BATCH_NS", "2,8").split(",")]


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", str(msg).encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
PLANE_INPUTS = tuple("field:" + n for n in applier.PLANE_NAMES)
PUMP_BOOK_NAMES = ("ge_pump_inject", "ge_pump_extract", "ge_pump_transfer",
                   "ge_diag_pump_transfer_from", "ge_diag_pump_transfer_to") + tuple(
    f"trace_clip_{g}" for g in range(N_GASES))
PUMP_BOOKS = tuple("book:" + b for b in PUMP_BOOK_NAMES)
HOST_ONLY = (
    ("_gas_t_amb_raw_cache", "a lazily filled cache of a config constant (the "
     "device reads the table row topo_t_amb_q)"),
    ("smoke", "GameMap.smoke IS a view of gas[SMOKE] (the same memory): compared "
     "through field:gas"),
)
_APPLIERS: dict = {}


def _applier_for(store):
    n = store.shape[0]
    if n not in _APPLIERS:
        _APPLIERS[n] = applier.DeviceApplier(n, 4, bp=bp)
    return _APPLIERS[n]


def _run_a(st, inp, lists):
    return uk.signals_emit(st, _applier_for(st), inp.fields, lists, inp.tick)


def _run_b(st, inp, lists):
    return uk.signals_logic(st, _applier_for(st), lists)


def _run_c(st, inp, lists):
    return uk.signals_door_inputs(st, _applier_for(st), lists)


def _run_p(st, inp, lists):
    return uk.signals_pumps(st, _applier_for(st), inp.fields, lists)


def _run_e(st, inp, lists):
    return uk.signals_swap(st, _applier_for(st), lists)


# Each sub-step writes no unit row (columns=("id",)); the conductor runs it
# only when a bus exists (skip_ok: an env with no bus is compared as a run on
# the state the tick ended with, which must change nothing).
SLOT_A = DeviceSlot(label="9e(a) signals", run=_run_a, columns=("id",), skip_ok=True,
                    inputs=("tick",) + PLANE_INPUTS + (
                        "list:sensor_tiles", "list:sensors", "list:door_tiles",
                        "list:doors", "list:signals"),
                    outputs=("list:signals",))
SLOT_B = DeviceSlot(label="9e(b) logic", run=_run_b, columns=("id",), skip_ok=True,
                    inputs=("list:signals", "list:node_ins", "list:nodes"),
                    outputs=("list:signals", "list:node_ins", "list:nodes"))
SLOT_C = DeviceSlot(label="9e(c) door inputs", run=_run_c, columns=("id",), skip_ok=True,
                    inputs=("list:signals", "list:door_drives", "list:door_tiles",
                            "list:doors"),
                    outputs=("list:door_tiles", "list:doors"))
SLOT_P = DeviceSlot(label="9e(d) pumps", run=_run_p, columns=("id",), skip_ok=True,
                    inputs=PLANE_INPUTS + ("list:signals", "list:pump_ins", "list:pumps"),
                    outputs=tuple("field:" + n for n in applier.PLANE_NAMES) + PUMP_BOOKS
                    + ("list:signals", "list:pump_ins", "list:pumps"),
                    host_only=HOST_ONLY)
SLOT_E = DeviceSlot(label="9e(e) swap", run=_run_e, columns=("id",), skip_ok=True,
                    inputs=("list:signals",), outputs=("list:signals",))
SIGNAL_SLOTS = [SLOT_A, SLOT_B, SLOT_C, SLOT_P, SLOT_E]


def _slots():
    from cuda_unit_doors_check import SLOT_1B, SLOT_9ED
    from cuda_unit_vents_check import SLOT_VENTS
    return [SLOT_1B, SLOT_A, SLOT_B, SLOT_C, SLOT_P, SLOT_VENTS, SLOT_9ED, SLOT_E]


# ---------------------------------------------------------------------------
# The signal scene
# ---------------------------------------------------------------------------
def _entity(eid, cls, ordinal, **fields):
    from level_loader import EntityInstance
    from simulation.entities import REGISTRY
    f = {x.name: x.default for x in REGISTRY[cls].FIELDS}
    f.update(fields)
    return EntityInstance(id=eid, class_name=cls, ordinal=ordinal, fields=f)


def _signal_level(variant):
    """1 m tiles, 20 x 40. A_top (x 1..12, y 1..6) | the door DE (h, y 7,
    x 5..7) | A_bot (y 8..18); the hull airlock DA (x 13, y 2..6, open) |
    C1 (x 14..17) | DB (x 18, closed) | space (x 19..39); the interior
    airlock DC (x 13, y 11..15) | C2 | DD (x 18) | room B (x 19..38, y 9..18,
    thin air: the scene lowers it). A one-tile pocket (17, 15); a hull wall
    (13, 2..4) and a grille (13, 8) in A_bot; a hull tile (16, 5) and a wood
    tile (16, 2). Ordinals are not row indices (offset and shuffled).
    ``variant`` -1: the same map with its doors only (NO bus); odd: planetside;
    2..3: fewer nodes and other thresholds; >= 4: DE closed by another node."""
    from level_loader import LevelData, Wire
    from simulation.entities import REGISTRY
    h, w = 20, 40
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:7, 1:13] = MAT_AIR
    tm[7, 5:8] = MAT_AIR                        # DE's span
    tm[8:19, 1:13] = MAT_AIR
    tm[2:7, 13] = MAT_AIR                       # DA
    tm[2:7, 14:18] = MAT_AIR                    # C1
    tm[2:7, 18] = MAT_AIR                       # DB
    tm[1:7, 19:40] = SPACE
    tm[11:16, 13] = MAT_AIR                     # DC
    tm[11:16, 14:18] = MAT_AIR                  # C2
    tm[11:16, 18] = MAT_AIR                     # DD
    tm[9:19, 19:39] = MAT_AIR                   # B
    tm[17, 15] = MAT_AIR                        # the pocket
    tm[13, 2:5] = MAT_HULL
    tm[13, 8] = MAT_GRILLE
    tm[16, 5] = MAT_HULL
    tm[16, 2] = MAT_WOOD
    v = variant
    doors = [
        _entity("de", "door", 0, x=5, y=7, orientation="h", length_m=3.0,
                initial_state="closed"),
        _entity("da", "door", 0, x=13, y=2, orientation="v", length_m=5.0,
                initial_state="open"),
        _entity("db", "door", 0, x=18, y=2, orientation="v", length_m=5.0,
                initial_state="closed"),
        _entity("dc", "door", 0, x=13, y=11, orientation="v", length_m=5.0,
                initial_state="open"),
        _entity("dd", "door", 0, x=18, y=11, orientation="v", length_m=5.0,
                initial_state="closed"),
    ]
    rest = [] if v < 0 else [
        _entity("p1", "pump", 0, x=15, y=1, port_dy=1, rate=8.0, target_atm=0,
                hysteresis_band=13107),
        _entity("p2", "pump", 0, x=16, y=10, port_dy=1, feed="transfer", source_dx=3,
                source_dy=1, rate=4.0, target_atm=ONE // 2, hysteresis_band=13107),
        _entity("p3", "pump", 0, x=15, y=16, port_dy=-1, feed="transfer", source_dx=-3,
                source_dy=-1, rate=4.0, target_atm=ONE, hysteresis_band=13107),
        _entity("p4", "pump", 0, x=14, y=17, port_dx=-2, feed="transfer", source_dx=1,
                rate=2.0, target_atm=ONE, hysteresis_band=8000),
        _entity("p5", "pump", 0, x=0, y=10, rate=1.0, target_atm=ONE),
        _entity("plate1", "sensor_motion", 0, x=15, y=4, radius=2.0),
        _entity("plate2", "sensor_motion", 0, x=15, y=13, radius=2.0),
        _entity("mseen", "sensor_motion", 0, x=7, y=17, radius=4.0, needs_los=True),
        _entity("mwall", "sensor_motion", 0, x=3, y=11, radius=4.0, needs_los=True),
        _entity("mgrille", "sensor_motion", 0, x=8, y=11, radius=4.0, needs_los=True),
        _entity("mnolos", "sensor_motion", 0, x=8, y=11, radius=4.0),
        _entity("mfac", "sensor_motion", 0, x=8, y=16, radius=5.0, faction_filter=2,
                min_footprint=2),
        _entity("c1p", "pressure", 0, x=16, y=1, sample_dy=1),
        _entity("c2p", "pressure", 0, x=16, y=16, sample_dy=-1),
        _entity("ap", "pressure", 0, x=10, y=19, sample_dy=-1),
        _entity("bp", "pressure", 0, x=30, y=19, sample_dy=-1),
        _entity("smk", "smoke", 0, x=6, y=0, sample_dy=3, area_m=2.0),
        _entity("smkd", "smoke", 0, x=6, y=8, sample_dy=-1),
        _entity("wat", "water_depth", 0, x=3, y=0, sample_dy=1),
        _entity("tmp", "temperature", 0, x=3, y=13),
        _entity("fir", "fire", 0, x=2, y=16),
        _entity("o2s", "o2", 0, x=5, y=19, sample_dy=-2, area_m=2.0),
        _entity("clk", "clock", 0, period=5),
        _entity("dfar", "decider", 0, comparator="le", threshold=6554),
        _entity("dnear", "decider", 0, comparator="ge", threshold=58982),
        _entity("ctrl1", "airlock_controller", 0, far_door="outer", target_far_atm=0,
                target_near_atm=ONE),
        _entity("ctrl2", "airlock_controller", 0, far_door="outer", mode="interior",
                equalize_band=3277),
        _entity("kgt", "decider", 0, comparator="gt", threshold=3 + v % 3),
        _entity("kge", "decider", 0, comparator="ge", threshold=4),
        _entity("klt", "decider", 0, comparator="lt", threshold=9),
        _entity("kle", "decider", 0, comparator="le", threshold=5 + v % 2),
        _entity("keq", "decider", 0, comparator="eq", threshold=7),
        _entity("kne", "decider", 0, comparator="ne", threshold=2),
        _entity("and2", "gate_and", 0),
        _entity("or2", "gate_or", 0),
        _entity("not1", "gate_not", 0),
        _entity("filt2", "filter", 0, tau_s=0.1),
        _entity("dchain", "decider", 0, comparator="gt", threshold=ONE // 2),
        _entity("ra1", "decider", 0, comparator="ge", threshold=1, require_alive=True),
        _entity("and0", "gate_and", 0),
        _entity("not0", "gate_not", 0),
        _entity("filt0", "filter", 0, tau_s=1.0),
    ]
    if 0 <= v and v not in (2, 3):
        rest += [_entity("filt1", "filter", 0, tau_s=0.5),
                 _entity("ra2", "decider", 0, comparator="gt", threshold=-1,
                         require_alive=True)]
    ents = doors + rest
    rng = np.random.default_rng(4400 + variant)
    ords = rng.permutation(len(ents)) + 100        # never a row index
    for e, o in zip(ents, ords):
        e.ordinal = int(o)
    by = {e.id: e for e in ents}
    wires = []

    def wire(src, sig, dst, inp):
        if src not in by or dst not in by:
            return
        mode = {i.name: i.mode for i in REGISTRY[by[dst].class_name].INPUTS}[inp]
        wires.append(Wire(source_ordinal=int(by[src].ordinal), signal=sig,
                          target_ordinal=int(by[dst].ordinal), input=inp,
                          aggregation_mode=mode))
    if v >= 0:
        for c, plate, inner, outer in (("ctrl1", "plate1", "da", "db"),
                                       ("ctrl2", "plate2", "dc", "dd")):
            wire(plate, "value", c, "presence")
            wire(inner, "is_open", c, "inner_open")
            wire(outer, "is_open", c, "outer_open")
            wire(inner, "alive", c, "inner_alive")
            wire(outer, "alive", c, "outer_alive")
            wire(c, "inner_close", inner, "close")
            wire(c, "inner_open_cmd", inner, "open")
            wire(c, "outer_close", outer, "close")
            wire(c, "outer_open_cmd", outer, "open")
        wire("c1p", "value", "dfar", "in")
        wire("c1p", "value", "dnear", "in")
        wire("dfar", "out", "ctrl1", "at_far")
        wire("dnear", "out", "ctrl1", "at_near")
        wire("ctrl1", "pump_inject", "p1", "inject")
        wire("ctrl1", "pump_extract", "p1", "extract")
        wire("c2p", "value", "ctrl2", "chamber_p")
        wire("ap", "value", "ctrl2", "inner_side_p")
        wire("bp", "value", "ctrl2", "outer_side_p")
        wire("ctrl2", "pump_inject", "p2", "inject")
        wire("ctrl2", "pump_extract", "p2", "extract")
        wire("ctrl2", "near_pump_inject", "p3", "inject")
        wire("ctrl2", "near_pump_extract", "p3", "extract")
        for d in ("kgt", "kge", "klt", "kle", "keq", "kne"):
            wire("clk", "value", d, "in")
        wire("kgt", "out", "and2", "in")           # two wires into one input
        wire("klt", "out", "and2", "in")
        wire("keq", "out", "or2", "in")
        wire("kne", "out", "or2", "in")
        wire("p4", "at_target", "or2", "in")
        wire("smk", "alive", "and2", "in")          # a constant alive
        wire("and2", "out", "not1", "in")           # the chain: clk -> kgt -> and2
        wire("not1", "out", "filt2", "in")          #   -> not1 -> filt2 -> dchain
        wire("filt2", "out", "dchain", "in")
        wire("de", "is_open", "ra1", "in")
        wire("c1p", "value", "filt1", "in")
        wire("or2", "out", "de", "open")            # two wires into DE's open
        wire("dchain", "out", "de", "open")
        wire("kge" if v >= 4 else "keq", "out", "de", "close")
        wire("kge", "out", "p4", "inject")
        wire("not1", "out", "p5", "inject")
        wire("keq", "out", "p5", "extract")
    order = list(rng.permutation(len(ents)))
    ents = [ents[i] for i in order]                # the list order is not the ordinal order
    return LevelData(name=f"p6b3_signals{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."), entities=ents,
                     wires=wires, boundary="ambient" if variant % 2 else "vacuum")


def _reseed(g, ys, xs):
    g.reseed_gas_energy((np.asarray(ys), np.asarray(xs)))


class SignalScene:
    """The signal scene, driven by fixtures between ticks (teleports into and
    out of both chambers as each cycle reaches its far door, smoke, a flood,
    a wall destroyed under a sensor and inside a disc, a death, a dead pump
    and sensor, a door destroyed while closed, USE on DE on the ticks the
    bus drives it). ``variant`` shifts the schedule, seed, layout, boundary."""

    def __init__(self, variant=0):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.v, self.k = variant, 0
        sim = Simulation(_signal_level(variant), seed=63300 + max(variant, 0) * 7,
                         breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        self.byid = {e.id: e for e in sim.entities}
        self.u0 = self._add("U0", 3, 3)           # the hull airlock's user
        self.u1 = self._add("U1", 3, 12)          # the interior airlock's user
        self.u2 = self._add("U2", 3, 15)          # behind the wall from mwall
        self.u3 = self._add("U3", 8, 15, team=2, fp=2)   # behind the grille, team 2
        self.u4 = self._add("U4", 6, 8)           # below DE (its USE)
        self.u5 = self._add("U5", 10, 17)         # team 0 near mfac
        self.u6 = self._add("U6", 11, 18, team=2)  # team 2, too small for mfac
        g = sim.gmap
        # room B at about half pressure (the interior legs equalise to it)
        ys, xs = np.mgrid[9:19, 19:39]
        for s in range(N_GASES):
            g.gas[s][9:19, 19:39] //= 2
        _reseed(g, ys.ravel(), xs.ravel())
        self.out0 = self.out1 = 0

    def _add(self, name, x, y, team=0, fp=1):
        from simulation.unit import Unit
        u = Unit(name, x=float(x), y=float(y), team=team, footprint=fp)
        self.sim.add_unit(u)
        u.current_hp = 1.0e6
        return u

    def _ctrl(self, cid):
        return self.byid.get(cid)

    def tick(self):
        sim, k = self.sim, self.k - max(self.v, 0)
        g = sim.gmap
        E = {e.id: e for e in sim.entities}
        rng = np.random.default_rng(1000 + self.k)
        if k == 2:                                # smoke in A_top (raw trace counts:
            sm = rng.integers(0, 1 << 26, (6, 12)).astype(np.int32)   # scale-free, #88)
            g.gas[SMOKE][1:7, 1:13] = sm
            ys, xs = np.mgrid[1:7, 1:13]
            _reseed(g, ys.ravel(), xs.ravel())
        if k == 5:
            self.u0.set_pos(15.0, 3.0)            # into C1
        if k == 7:
            self.u1.set_pos(15.0, 13.0)           # into C2
        if k in (8, 70, 140):
            sim.use_intent(self.u4.id)            # DE used (the bus drives it too)
        if k == 12:
            g.water_depth[1, 3] = 1 << 14
        if k == 20:
            g.water_depth[1, 3] = 0
        if k == 30:
            g.destroy_wall(13, 3)                 # tmp's tile; mwall sees U2
        if k == 35:
            g.destroy_wall(16, 5)                 # inside o2s's disc
        if k == 60:
            self.u2.alive = False
        if k == 100 and "p4" in E:
            E["p4"].alive = False                 # a dead pump: its latch frozen
        if k == 110 and "fir" in self.byid:
            self.byid["fir"].alive = False        # a dead sensor publishes 0
            self.byid["smk"].alive = False        # ... and its wired alive reads 0
        if k >= 36 and "de" in E and E["de"].alive and E["de"].state == 0:
            g.destroy_wall(7, 6)                  # DE destroyed while closed
        for c, u, attr, fy, fx in (("ctrl1", self.u0, "out0", 3.0, 22.0),
                                   ("ctrl2", self.u1, "out1", 13.0, 22.0)):
            ct = E.get(c)
            if ct is None:
                continue
            st = getattr(ct, "state", 0)
            if st == 3 and getattr(self, attr) == 0:      # OPEN_FAR: step out
                u.set_pos(fx, fy)
                setattr(self, attr, 1)
            elif st == 0 and getattr(self, attr) == 1:    # IDLE again: come back in
                u.set_pos(15.0, fy)
                setattr(self, attr, 2)
        sim.step()
        sim.set_paused(False)
        self.k += 1


class AirlockDriver:
    """airlock_demo, its marine teleported into the chamber at tick 5 and out
    through the far door once it opens (the level has no driver of its own)."""

    def __init__(self):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        level = level_loader.load("airlock_demo", str(_ROOT / "levels"))
        self.sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                              ruleset=ContinuousRealtime())
        self.sim.set_paused(False)
        from simulation.unit import Unit
        u = Unit("Alpha", x=3.0, y=4.0, team=0, footprint=3)   # the level's [[spawn]]
        self.sim.add_unit(u)
        u.current_hp = 1.0e6
        self.k = 0
        self.out = 0

    def tick(self):
        u = self.sim.units[0]
        ctrl = [e for e in self.sim.entities if e.class_name == "airlock_controller"][0]
        if self.k == 5:
            u.set_pos(9.0, 3.0)
        if ctrl.state == 3 and not self.out:
            u.set_pos(15.0, 3.0)
            self.out = 1
        self.sim.step()
        self.sim.set_paused(False)
        self.k += 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _planes(gmap) -> dict:
    return {n: np.array(getattr(gmap, n), copy=True) for n in applier.PLANE_NAMES}


def _dev_planes(per_env: list) -> dict:
    import cupy as cp
    return {n: cp.asarray(np.ascontiguousarray(np.stack([p[n] for p in per_env])))
            for n in applier.PLANE_NAMES}


def _store(sims, U=None):
    U = U or max([1] + [len(s.units) for s in sims])
    st = unit_store.DeviceStore(len(sims), U, bp=bp, tables=unit_tables.values_batch(sims))
    st.upload(np.stack([unit_store.pack(s.units, U) for s in sims]))   # the rows: motion plates
    return st


def _records(table, rows):
    """Row objects as their table's records (and the side table's, if any)."""
    recs = uk._records_of(table, rows)
    if not table.spans:
        return recs, None
    return recs, uk._side_of(table, rows, table.spans[0].name)


def _stage(tabs, name, per_env):
    tabs[name].stage_lists([np.asarray(p, dtype=np.int64) for p in per_env])


def _identity(sim, where):
    """The eight tables ARE the sim's: unpack(records) row for row, the
    digest rows, the entity rows, the landing a no-op on itself."""
    from simulation.entities.serialize import entity_records  # noqa: F401
    from simulation.entities import REGISTRY
    ok = True
    for rows_fn, tab in ((uk.signal_rows, us.SIGNAL_TABLE), (uk.sensor_rows, us.SENSOR_TABLE),
                         (uk.node_rows, us.NODE_TABLE), (uk.pump_rows, us.PUMP_TABLE),
                         (uk.drive_rows, us.DOOR_DRIVE_TABLE)):
        rows = rows_fn(sim)
        recs, side = _records(tab, rows)
        sides = None
        if tab.spans:
            sides = {tab.spans[0].name: (side, side.shape[0])}
        back = ot.unpack(tab, recs, recs.shape[0], sides=sides)
        if back != rows:
            fail(f"identity {where}: table {tab.name}: unpack(records) differs")
            ok = False
    bus = getattr(sim, "_signal_bus", None)
    if bus is not None and uk.signal_digest_rows(uk.signals_of(sim)) != bus.digest_rows():
        fail(f"identity {where}: the bus table's digest rows are not SignalBus.digest_rows")
        ok = False
    by_ord = {int(e.ordinal): e for e in sim.entities}
    for rec in uk.nodes_of(sim):
        e = by_ord[int(rec[0])]
        want = (("alive", 1 if getattr(e, "alive", True) else 0),) + tuple(
            REGISTRY[e.class_name].runtime_digest_rows(e))
        if uk.node_entity_rows(rec) != want:
            fail(f"identity {where}: node {e.id}: the table's entity rows are not the "
                 f"serializer's")
            ok = False
    for rec in uk.pumps_of(sim):
        e = by_ord[int(rec[0])]
        want = (("alive", 1 if e.alive else 0),) + tuple(REGISTRY["pump"].runtime_digest_rows(e))
        if uk.pump_entity_rows(rec) != want:
            fail(f"identity {where}: pump {e.id}: the table's entity rows are not the "
                 f"serializer's")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def part_tables():
    sc = SignalScene(0)
    for _ in range(40):
        sc.tick()
    ad = AirlockDriver()
    for _ in range(40):
        ad.tick()
    for nm, sim in (("scene", sc.sim), ("airlock_demo", ad.sim)):
        if not _identity(sim, nm):
            continue
        # the landing writes the runtimes back: mutate the device-side rows
        recs = uk.nodes_of(sim)
        lay = ot.table_layout(us.NODE_TABLE)
        recs2 = recs.copy()
        recs2[:, lay.column("ema").offset] += 12345
        recs2[:, lay.column("dwell").offset] += 7
        uk.nodes_to_host(sim, recs2)
        if uk.nodes_of(sim).tolist() != [
                [int(x) for x in r] for r in _relabel_stateless(recs, recs2)]:
            fail(f"tables {nm}: nodes_to_host did not land ema / dwell")
        uk.nodes_to_host(sim, recs)
        if uk.nodes_of(sim).tolist() != recs.tolist():
            fail(f"tables {nm}: nodes_to_host is not its own inverse")
        brec = uk.signals_of(sim)
        b2 = brec.copy()
        b2[:, 2] += 3
        uk.signals_to_host(sim, b2)
        if uk.signals_of(sim).tolist() != b2.tolist():
            fail(f"tables {nm}: signals_to_host did not land pub")
        uk.signals_to_host(sim, brec)
        for fn, bad in ((uk.sensors_to_host, uk.sensors_of(sim)),
                        (uk.drives_to_host, uk.door_drives_of(sim))):
            bad = bad.copy()
            if bad.size:
                bad[0, 0] += 1
                try:
                    fn(sim, bad)
                    fail(f"tables {nm}: {fn.__name__} took rows that are not the sim's")
                except uk.SignalTableMismatch:
                    pass
        print(f"tables {nm}: the eight signal tables are the sim's (counts "
              + ", ".join(f"{k} {len(f(sim))}" for k, f in uk.SIGNAL_RECORDS.items())
              + "); digest rows, entity rows and landings equal")
    try:
        sim = SignalScene(0).sim
        heat = SimpleNamespace(class_name="heat_source", id="hs", alive=True, ordinal=0)
        bus = sim._signal_bus
        o, n = bus.slots[0]
        keep = sim._entity_by_ordinal.get(o)
        sim._entity_by_ordinal[o] = heat
        try:
            uk.signal_rows(sim)
            if n in ("alive", "is_open"):
                fail("tables: a heat source's alive emit was not refused by name")
        except uk.SignalTableRefused:
            pass
        finally:
            sim._entity_by_ordinal[o] = keep
    except Exception as ex:                      # pragma: no cover
        fail(f"tables: the refusal check raised {ex!r}")


def _relabel_stateless(recs, recs2):
    """recs2's ema / dwell land only where the node keeps state (a filter's
    ema, an airlock's dwell); the stateless rows keep theirs."""
    lay = ot.table_layout(us.NODE_TABLE)
    k, e, d = (lay.column(c).offset for c in ("kind", "ema", "dwell"))
    out = []
    for a, b in zip(recs, recs2):
        r = list(int(x) for x in a)
        kind = us.NODE_KINDS[r[k]]
        if kind in ("filter", "airlock_controller"):
            r[e], r[d] = int(b[e]), int(b[d])
            if kind == "filter":
                r[d] = int(a[d])
            else:
                r[e] = int(a[e])
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# PART SAMPLE
# ---------------------------------------------------------------------------
def _random_planes(rng, base):
    """Copies of a map's planes with every channel over its whole range: the
    trace slice on both sides of every 256 boundary and negative (the floor),
    signed temperatures, fire, water, a random solid mask."""
    p = {k: v.copy() for k, v in base.items()}
    shp = p["atmosphere"].shape
    p["atmosphere"][...] = rng.integers(-(1 << 31), 1 << 31, shp, dtype=np.int64)
    sm = rng.integers(-(1 << 20), 1 << 31, shp, dtype=np.int64)
    edge = rng.integers(-64, 64, shp) * 256 + rng.integers(-1, 2, shp)
    sm = np.where(rng.random(shp) < 0.5, edge, sm)
    p["gas"][SMOKE][...] = np.clip(sm, -(1 << 31), (1 << 31) - 1)
    p["gas"][O2][...] = rng.integers(0, 1 << 31, shp, dtype=np.int64)
    p["water_depth"][...] = rng.integers(-(1 << 20), 1 << 20, shp, dtype=np.int64)
    p["temperature"][...] = rng.integers(-(1 << 31), 1 << 31, shp, dtype=np.int64)
    p["fire"][...] = rng.integers(0, 1 << 31, shp, dtype=np.int64)
    p["solid"][...] = rng.random(shp) < rng.choice([0.0, 0.3, 0.9, 1.0])
    return p


def _sample_maps():
    sims = [SignalScene(0).sim, AirlockDriver().sim]
    for name in ("ship_demo", "playground"):
        from simulation import Simulation
        sims.append(Simulation(level_loader.load(name, str(_ROOT / "levels")), seed=1,
                               breach_physics=bp, enable_recorder=False))
    return sims


def part_sample():
    from simulation import sensor_system as ss
    from simulation.sensor_accessor import CHANNEL_ORDER, EntityFieldAccessor
    rng = np.random.default_rng(6301)
    census = Counter()
    total = 0
    maps = _sample_maps()
    by_shape = {}
    for s in maps:
        by_shape.setdefault(s.gmap.solid.shape, []).append(s)
    for shape, sims in by_shape.items():
        for rep in range(24):
            K = us.SENSOR_CAPACITY
            per_planes, per_sensors, per_tiles, per_bus, want, ticks = [], [], [], [], [], []
            for e, sim in enumerate(sims):
                h, w = shape
                planes = _random_planes(rng, _planes(sim.gmap))
                acc = EntityFieldAccessor(SimpleNamespace(
                    atmosphere=planes["atmosphere"], gas=planes["gas"],
                    water_depth=planes["water_depth"], temperature=planes["temperature"],
                    fire=planes["fire"], solid=planes["solid"]))
                tick = int(rng.integers(0, 1 << 40))
                rows, exp = [], []
                for i in range(K):
                    ch = CHANNEL_ORDER[i % len(CHANNEL_ORDER)]
                    fy, fx = int(rng.integers(0, h)), int(rng.integers(0, w))
                    kind = "clock" if i % 9 == 8 else "field"
                    alive = bool(rng.random() > 0.05)
                    kw = dict(ordinal=i, alive=alive, kind=kind, slot=i, channel=-1, gas=-1,
                              sy=0, sx=0, has_disc=False, disc=[], period=0, r2=0,
                              needs_los=False, faction=0, min_footprint=0)
                    if kind == "clock":
                        per = int(rng.choice([1, 2, 3, 7, 24, 1000, 1 << 20]))
                        kw["period"] = per
                        val = tick // per
                        census["clock"] += 1
                    else:
                        disc = None
                        if rng.random() < 0.5:
                            disc = ss._disc_tiles(fy, fx, int(rng.integers(1, 4)), h, w)
                        kw.update(channel=us.sensor_channels().index(ch.value),
                                  gas=SMOKE if ch.value == "smoke" else (
                                      O2 if ch.value == "o2" else -1),
                                  sy=fy, sx=fx, has_disc=disc is not None,
                                  disc=list(disc or []))
                        if disc is None:
                            val = acc.sample(ch, fy, fx)
                        else:
                            tot, n = acc.area(ch, disc)
                            val = tot // n if n > 0 else 0
                            census["disc empty" if n == 0 else "disc"] += 1
                        census[f"channel {ch.value}"] += 1
                        if ch.value == "smoke":
                            raw = int(planes["gas"][SMOKE][fy, fx])
                            census["smoke negative" if raw < 0 else "smoke >= 0"] += 1
                    if not alive:
                        val = 0
                        census["dead"] += 1
                    rows.append(us.SensorRow(**kw))
                    exp.append(int(val))
                srec, trec = _records(us.SENSOR_TABLE, rows)
                bus = [us.SignalRow(ordinal=i, name="value", pub=-1, stg=0, node=False,
                                    emit="none", emit_ref=0) for i in range(K)]
                per_planes.append(planes)
                per_sensors.append(srec)
                per_tiles.append(trec)
                per_bus.append(_records(us.SIGNAL_TABLE, bus)[0])
                want.append(exp)
                ticks.append(tick)
            n = len(sims)
            st = _store(sims)
            tabs = uk.signal_tables(n, bp=bp)
            from simulation import object_tables
            doors = object_tables.DeviceTable(us.DOOR_TABLE, n, bp=bp)
            doors.begin()
            _stage(tabs, "sensors", per_sensors)
            _stage(tabs, "sensor_tiles", per_tiles)
            _stage(tabs, "signals", per_bus)
            for t in ("nodes", "node_ins", "pumps", "pump_ins", "door_drives"):
                tabs[t].begin()
            lists = dict(tabs, doors=doors)
            fields = _dev_planes(per_planes)
            uk.signals_emit(st, _applier_for(st), fields, lists, np.array(ticks))
            got = tabs["signals"].records()
            pc = ot.table_layout(us.SIGNAL_TABLE).column("pub").offset
            for e in range(n):
                g = [r[pc] for r in got[e]]
                for i, (a, b) in enumerate(zip(g, want[e])):
                    if a != b:
                        fail(f"sample {shape} env {e} sensor {i} (channel "
                             f"{CHANNEL_ORDER[i % 7].value}): host {b} device {a}")
                        return
                total += len(g)
            for k2, v in fields.items():
                if not np.array_equal(v.get(), np.stack([p[k2] for p in per_planes])):
                    fail(f"sample {shape}: the EMIT stage wrote plane {k2}")
                    return
    for need in [f"channel {c.value}" for c in CHANNEL_ORDER] + [
            "disc", "disc empty", "clock", "dead", "smoke negative", "smoke >= 0"]:
        if not census[need]:
            fail(f"sample: the census branch {need!r} was never taken")
    print(f"sample: {total} field / clock samples on {len(maps)} maps equal the accessor "
          f"(census {dict(census)})")
    _motion_probe(maps, rng)


def _motion_probe(maps, rng):
    """Motion plates over random unit placements on each real map (its own
    solid and sight screen): dead units, factions, footprints, LOS."""
    from simulation import sensor_system as ss
    census = Counter()
    total = 0
    for sim in maps:
        h, w = sim.gmap.solid.shape
        units = list(sim.units)
        if not units:
            from simulation.unit import Unit
            u = Unit("P", x=1.0, y=1.0, team=0, footprint=1)
            sim.add_unit(u)
            units = [u]
        for rep in range(20):
            for u in units:
                u.set_pos(float(rng.integers(0, w)), float(rng.integers(0, h)))
                u.alive = bool(rng.random() > 0.2)
            K = us.SENSOR_CAPACITY
            rows, exp = [], []
            for i in range(K):
                sy, sx = int(rng.integers(0, h)), int(rng.integers(0, w))
                r = int(rng.integers(0, 12))
                los = bool(rng.random() < 0.7)
                fac = int(rng.choice([-1, 0, 1, 2, units[0].team]))
                fpm = int(rng.integers(0, 4))
                rt = ss._MotionSensorRuntime(None, i, sy, sx, r * r, los, fac, fpm)
                exp.append(rt.evaluate(sim))
                rows.append(us.SensorRow(ordinal=i, alive=True, kind="motion", slot=i,
                                         channel=-1, gas=-1, sy=sy, sx=sx, has_disc=False,
                                         disc=[], period=0, r2=r * r, needs_los=los,
                                         faction=fac, min_footprint=fpm))
                census["seen" if exp[-1] else "none"] += 1
                if los:
                    for u in units:
                        if u.alive and not sim.gmap.has_los(sy, sx, u.tile_y, u.tile_x):
                            census["los blocked"] += 1
                            break
            srec, trec = _records(us.SENSOR_TABLE, rows)
            bus = [us.SignalRow(ordinal=i, name="value", pub=-1, stg=0, node=False,
                                emit="none", emit_ref=0) for i in range(K)]
            st = _store([sim])
            tabs = uk.signal_tables(1, bp=bp)
            from simulation import object_tables
            doors = object_tables.DeviceTable(us.DOOR_TABLE, 1, bp=bp)
            doors.begin()
            _stage(tabs, "sensors", [srec])
            _stage(tabs, "sensor_tiles", [trec])
            _stage(tabs, "signals", [_records(us.SIGNAL_TABLE, bus)[0]])
            for t in ("nodes", "node_ins", "pumps", "pump_ins", "door_drives"):
                tabs[t].begin()
            uk.signals_emit(st, _applier_for(st), _dev_planes([_planes(sim.gmap)]),
                            dict(tabs, doors=doors), np.array([0]))
            pc = ot.table_layout(us.SIGNAL_TABLE).column("pub").offset
            got = [r[pc] for r in tabs["signals"].records()[0]]
            if got != exp:
                i = next(j for j in range(K) if got[j] != exp[j])
                fail(f"motion probe {sim.level.name}: plate {i}: host {exp[i]} device {got[i]}")
                return
            total += K
    for need in ("seen", "none", "los blocked"):
        if not census[need]:
            fail(f"motion probe: {need!r} never happened")
    print(f"motion probe: {total} plates equal _MotionSensorRuntime.evaluate "
          f"(census {dict(census)})")


# ---------------------------------------------------------------------------
# PART LOGIC / AIRLOCK / REFUSAL: synthetic node tables against the evaluators
# ---------------------------------------------------------------------------
def _fake_inst(cls, ordinal):
    return SimpleNamespace(class_name=cls, ordinal=ordinal, id=f"n{ordinal}", fields={})


class NodeRig:
    """Host evaluators over a SignalBus beside the device's node and bus
    tables (the node table carried on the device between steps)."""

    def __init__(self, n_slots, builders):
        from simulation.signal_bus import SignalBus
        self.bus = SignalBus([(i, "value") for i in range(n_slots)])
        self.nodes, self.insts = [], []
        for i, (cls, fn) in enumerate(builders):
            self.insts.append(_fake_inst(cls, 1000 + i))
            self.nodes.append(fn(self.insts[-1]))
        self.fake = SimpleNamespace(entities=self.insts, _logic_nodes=self.nodes)
        self.st = _store([SignalScene(-1).sim])
        self.tabs = uk.signal_tables(1, bp=bp)
        nrec, irec = _records(us.NODE_TABLE, uk.node_rows(self.fake))
        _stage(self.tabs, "nodes", [nrec])
        _stage(self.tabs, "node_ins", [irec])

    def bus_rows(self):
        return [us.SignalRow(ordinal=o, name=n, pub=int(self.bus.pub[i]),
                             stg=int(self.bus.stg[i]), node=False, emit="none", emit_ref=0)
                for i, (o, n) in enumerate(self.bus.slots)]

    def step(self, raise_ok=False):
        _stage(self.tabs, "signals", [_records(us.SIGNAL_TABLE, self.bus_rows())[0]])
        err = None
        try:
            uk.signals_logic(self.st, _applier_for(self.st), self.tabs)
        except applier.SignalRangeRefused as ex:
            if not raise_ok:
                raise
            err = ex
        if err is None:
            for n in self.nodes:
                n.evaluate(self.bus)
        return err

    def compare(self, where):
        sc = ot.table_layout(us.SIGNAL_TABLE).column("stg").offset
        got = [r[sc] for r in self.tabs["signals"].records()[0]]
        want = [int(x) for x in self.bus.stg]
        if got != want:
            i = next(j for j in range(len(got)) if got[j] != want[j])
            fail(f"{where}: stg slot {i}: host {want[i]} device {got[i]}")
            return False
        dev = [list(r) for r in self.tabs["nodes"].records()[0]]
        host = uk._records_of(us.NODE_TABLE, uk.node_rows(self.fake)).tolist()
        if dev != host:
            fail(f"{where}: the node table differs from the evaluators (ema / state / dwell)")
            return False
        return True


def part_logic():
    from simulation import logic_nodes as ln
    from simulation.entities.schema import INPUT_AND, INPUT_HELD
    rng = np.random.default_rng(6302)
    census = Counter()
    cmps = us.comparators()
    # slots: 0 the compared value, 1..5 gate / alive inputs, 6 the filters'
    # input; outputs from 8 on, each node its own
    builders = [("decider", (lambda inst, c=c, o=8 + i: ln._DeciderEval(
        o, [0], c, 1000, False, None))) for i, c in enumerate(cmps)]
    builders += [
        ("decider", lambda inst: ln._DeciderEval(14, [1], "gt", 0, True, 2)),   # alive slot 2
        ("decider", lambda inst: ln._DeciderEval(15, [1], "gt", 0, True, None)),  # unwired src
        ("decider", lambda inst: ln._DeciderEval(16, [], "lt", 5, False, None)),  # unwired in
        ("gate_and", lambda inst: ln._GateEval(17, [1, 3, 4], INPUT_AND)),
        ("gate_and", lambda inst: ln._GateEval(18, [], INPUT_AND)),
        ("gate_or", lambda inst: ln._GateEval(19, [1, 3, 4], INPUT_HELD)),
        ("gate_or", lambda inst: ln._GateEval(20, [], INPUT_HELD)),
        ("gate_not", lambda inst: ln._NotEval(21, [5])),
        ("gate_not", lambda inst: ln._NotEval(22, [])),
    ]
    for i, kk in enumerate((0, 1, 3, 7, 20, 40)):
        builders.append(("filter", (lambda inst, kk=kk, s=24 + i: ln.FilterRuntime(
            inst, s, 6 if kk != 40 else None, kk))))
    S = 32
    rig = NodeRig(S, builders)
    steps = 3000
    for t in range(steps):
        b = rig.bus.pub
        b[0] = 1000 + int(rng.integers(-2, 3))     # around the threshold
        b[1] = int(rng.choice([0, 0, 1, -3, 7]))
        b[2] = int(rng.choice([0, 1]))
        b[3] = int(rng.choice([0, 1, 2]))
        b[4] = int(rng.choice([0, 1]))
        b[5] = int(rng.choice([0, 0, 9, -9]))
        b[6] = int(rng.integers(-(1 << 30), 1 << 30)) if t % 500 < 250 else \
            int(rng.integers(-5, 6))
        for c, i in zip(cmps, range(len(cmps))):
            census[f"{c} {'true' if ln._COMPARATORS[c](int(b[0]), 1000) else 'false'}"] += 1
        census["require_alive zeroed" if int(b[1]) > 0 and not int(b[2]) else "ra"] += 1
        census["and mixed" if len({int(b[1]) != 0, int(b[3]) != 0, int(b[4]) != 0}) > 1
               else "and uniform"] += 1
        census["filter in < 0" if int(b[6]) < 0 else "filter in >= 0"] += 1
        rig.step()
        if not rig.compare(f"logic step {t}"):
            return
    emas = [n.ema for n in rig.nodes if isinstance(n, ln.FilterRuntime)]
    if not any(e < 0 for e in emas) and not census["filter in < 0"]:
        fail("logic: no filter ever held a negative ema (the floor untested)")
    for c in cmps:
        for tv in ("true", "false"):
            if not census[f"{c} {tv}"]:
                fail(f"logic: predicate {c} was never {tv}")
    for need in ("require_alive zeroed", "and mixed", "filter in < 0"):
        if not census[need]:
            fail(f"logic: {need!r} never happened")
    print(f"logic: {steps} steps of {len(rig.nodes)} nodes (six predicates around their "
          f"threshold, require_alive, the gates empty and mixed, filters k 0..40) equal the "
          f"evaluators; final emas {emas}")


AIR_STATES = ("IDLE", "CLOSING", "EQUALIZE", "OPEN_FAR", "RESEAL", "REPRESSURIZE", "FAULT")


def part_airlock():
    from simulation import logic_nodes as ln
    rng = np.random.default_rng(6303)
    in_names = us.AIRLOCK_INPUTS
    out_names = us.AIRLOCK_OUTPUTS
    builders = []
    seen = Counter()
    n_in = len(in_names)
    variants = [(io, fo, eq) for io in (False, True) for fo in (True, False)
                for eq in ((1, -1, 0) if not io else (0,))]
    base = 0
    for io, fo, eq in variants:
        ins = {n: base + j for j, n in enumerate(in_names)}
        outs = {n: base + n_in + j for j, n in enumerate(out_names)}
        base += n_in + len(out_names)
        builders.append(("airlock_controller", (lambda inst, io=io, fo=fo, eq=eq, ins=ins,
                                                outs=outs: ln.AirlockControllerRuntime(
                                                    inst, fo, eq, ins, outs, interior=io,
                                                    band=3277))))
    rig = NodeRig(base, builders)
    steps = 6000
    for t in range(steps):
        if t % 40 == 0:                             # a random state to start from
            for n in rig.nodes:
                n.state = int(rng.integers(0, 7))
                n.dwell = int(rng.integers(0, 5))
            _stage(rig.tabs, "nodes", [uk._records_of(us.NODE_TABLE, uk.node_rows(rig.fake))])
        pub = rig.bus.pub
        for n in rig.nodes:
            for j, nm in enumerate(in_names):
                s = getattr(n, us._AIRLOCK_IN_ATTR[j])
                if nm in ("chamber_p", "inner_side_p", "outer_side_p"):
                    pub[s] = int(rng.integers(0, 2 * ONE)) if rng.random() < 0.6 else \
                        int(rng.integers(30000, 36000))
                elif nm == "presence":
                    pub[s] = int(rng.choice([0, 0, 1, 2]))
                elif nm.endswith("alive"):
                    pub[s] = int(rng.random() > 0.05)
                else:
                    pub[s] = int(rng.random() < 0.5)
        prev = [n.state for n in rig.nodes]
        rig.step()
        for p, n in zip(prev, rig.nodes):
            seen[(AIR_STATES[p], AIR_STATES[n.state])] += 1
        if not rig.compare(f"airlock step {t}"):
            return
    need = [("IDLE", "IDLE"), ("IDLE", "CLOSING"), ("CLOSING", "FAULT"),
            ("CLOSING", "EQUALIZE"), ("CLOSING", "CLOSING"), ("EQUALIZE", "OPEN_FAR"),
            ("EQUALIZE", "EQUALIZE"), ("OPEN_FAR", "RESEAL"), ("OPEN_FAR", "OPEN_FAR"),
            ("RESEAL", "FAULT"), ("RESEAL", "REPRESSURIZE"), ("RESEAL", "RESEAL"),
            ("REPRESSURIZE", "IDLE"), ("REPRESSURIZE", "REPRESSURIZE"), ("FAULT", "FAULT")]
    for tr in need:
        if not seen[tr]:
            fail(f"airlock: the transition {tr} was never taken")
    print(f"airlock: {steps} steps of {len(rig.nodes)} controllers (hull x far outer / inner "
          f"x equalize +1 / -1 / 0; interior x far outer / inner) equal the evaluators; "
          f"every transition taken ({len([t for t in need if seen[t]])} / {len(need)})")


def part_refusal():
    from simulation import logic_nodes as ln
    L = uk.FILTER_LIM
    seen = Counter()
    cases = [("at bound", L, L, 62, True), ("at -bound", -L, -L, 0, True),
             ("ema +1", L + 1, 0, 3, False), ("ema -1", -L - 1, 0, 3, False),
             ("in +1", 0, L + 1, 3, False), ("in -1", 0, -L - 1, 3, False),
             ("k 63", 0, 0, 63, False), ("k -1", 0, 0, -1, False)]
    for name, ema, x, kk, ok in cases:
        rig = NodeRig(4, [("filter", lambda inst, kk=kk: ln.FilterRuntime(inst, 1, 0, kk)),
                          ("gate_not", lambda inst: ln._NotEval(2, [3]))])
        rig.nodes[0].ema = ema
        _stage(rig.tabs, "nodes", [uk._records_of(us.NODE_TABLE, uk.node_rows(rig.fake))])
        rig.bus.pub[0] = x
        before = [list(r) for r in rig.tabs["nodes"].records()[0]]
        err = rig.step(raise_ok=True)
        if ok:
            if err is not None:
                fail(f"refusal {name}: refused inside the proved range: {err}")
            elif rig.compare(f"refusal {name}"):
                seen["equal at the bound"] += 1
            continue
        if err is None:
            fail(f"refusal {name}: not refused")
            continue
        if "ordinal 1000" not in str(err) or "outside" not in str(err):
            fail(f"refusal {name}: the refusal does not name the node and operand: {err}")
        after = [list(r) for r in rig.tabs["nodes"].records()[0]]
        sc = ot.table_layout(us.SIGNAL_TABLE).column("stg").offset
        if after != before or any(r[sc] for r in rig.tabs["signals"].records()[0]):
            fail(f"refusal {name}: something of the env was written")
        seen["refused"] += 1
    print(f"refusal: {dict(seen)} (a filter one past each bound refused by name, nothing "
          f"written; at the bound equal to the host)")


# ---------------------------------------------------------------------------
# The census of the Python law's branches (reads only)
# ---------------------------------------------------------------------------
class Census:
    def __init__(self):
        self.c = Counter()
        self._orig = []

    def _wrap(self, cont, attr, fn):
        orig = getattr(cont, attr)
        self._orig.append((cont, attr, orig))

        def w(*a, **k):
            return fn(orig, *a, **k)
        setattr(cont, attr, w)

    def install(self):
        from simulation import logic_nodes as ln, pump_system as ps, sensor_system as ss
        from simulation import simulation as simmod
        from simulation.signal_bus import SignalBus
        c = self.c

        def emit(orig, sim):
            bus = sim._signal_bus
            for o, n in bus.slots:
                ent = sim._entity_by_ordinal.get(o)
                if ent is None:
                    c["emit: dangling"] += 1
                elif n == "is_open":
                    c["emit: is_open"] += 1
                elif n == "alive":
                    c["emit: alive " + ("door" if hasattr(ent, "span") else "constant")] += 1
            return orig(sim)
        self._wrap(simmod.Simulation, "_signal_emit", emit)

        def sample(orig, sim):
            for s in sim._sensors:
                if not getattr(s.inst, "alive", True):
                    c["sensor: dead"] += 1
                elif isinstance(s, ss._FieldSensorRuntime):
                    c[f"sensor: {s.channel.value}{' disc' if s.disc is not None else ''}"] += 1
                    if s.disc is not None:
                        n = sum(1 for (y, x) in s.disc if not sim.gmap.solid[y, x])
                        c["disc: a solid tile excluded" if n < len(s.disc) else "disc: all open"] += 1
                    fy, fx = s.sample_tile
                    if sim.gmap.solid[fy, fx]:
                        c["sensor: sample tile solid"] += 1
                elif isinstance(s, ss._ClockSensorRuntime):
                    c["sensor: clock"] += 1
                else:
                    for u in sim.units:
                        if not u.alive:
                            c["motion: dead unit"] += 1
                            continue
                        d2 = (u.tile_y - s.sy) ** 2 + (u.tile_x - s.sx) ** 2
                        if d2 > s.r2:
                            c["motion: out of range"] += 1
                        elif s.faction != -1 and u.team != s.faction:
                            c["motion: other faction"] += 1
                        elif u.footprint < s.min_footprint:
                            c["motion: footprint too small"] += 1
                        elif s.needs_los and not sim.gmap.has_los(s.sy, s.sx, u.tile_y,
                                                                  u.tile_x):
                            phys = sim.gmap.has_los(s.sy, s.sx, u.tile_y, u.tile_x,
                                                    physical=True)
                            c["motion: behind a grille" if phys else
                              "motion: behind a wall"] += 1
                        else:
                            c["motion: counted"] += 1
            return orig(sim)
        self._wrap(simmod, "sample_sensors", sample)

        def dec(orig, self_, bus):
            orig(self_, bus)
            c[f"decider {self_.comparator} -> {int(bus.stg[self_.out_slot])}"] += 1
            if self_.require_alive:
                c["decider: require_alive " + ("unwired" if self_.alive_slot is None else
                                               ("dead" if not int(bus.pub[self_.alive_slot])
                                                else "alive"))] += 1
            if not self_.in_slots:
                c["decider: unwired in"] += 1
        self._wrap(ln._DeciderEval, "evaluate", dec)

        def gate(orig, self_, bus):
            orig(self_, bus)
            from simulation.entities.schema import INPUT_AND
            k = "and" if self_.mode == INPUT_AND else "or"
            c[f"gate {k} -> {int(bus.stg[self_.out_slot])}{' (empty)' if not self_.in_slots else ''}"] += 1
        self._wrap(ln._GateEval, "evaluate", gate)

        def notg(orig, self_, bus):
            orig(self_, bus)
            c[f"not -> {int(bus.stg[self_.out_slot])}{' (empty)' if not self_.in_slots else ''}"] += 1
        self._wrap(ln._NotEval, "evaluate", notg)

        def filt(orig, self_, bus):
            orig(self_, bus)
            c["filter: ema < 0" if self_.ema < 0 else "filter: ema >= 0"] += 1
            if self_.in_slot is None:
                c["filter: unwired"] += 1
        self._wrap(ln.FilterRuntime, "evaluate", filt)

        def air(orig, self_, bus):
            p = self_.state
            orig(self_, bus)
            c[f"airlock{' interior' if self_.interior else ''} {AIR_STATES[p]} -> "
              f"{AIR_STATES[self_.state]}"] += 1
        self._wrap(ln.AirlockControllerRuntime, "evaluate", air)

        def din(orig, sim):
            bus = sim._signal_bus
            from simulation.logic_nodes import aggregate_input
            from simulation.entities.schema import INPUT_HELD
            for d in sim._doors:
                drv = sim._door_drives.get(int(d.ordinal))
                if drv is None:
                    c["9e(c): unwired door"] += 1
                    continue
                cl = aggregate_input(bus, drv["close"], INPUT_HELD) != 0
                op = aggregate_input(bus, drv["open"], INPUT_HELD) != 0
                c["9e(c): close beats open" if cl and op else "9e(c): close" if cl else
                  "9e(c): open" if op else "9e(c): neither (latch kept)"] += 1
                if len(drv["open"]) > 1:
                    c["9e(c): two wires into open"] += 1
            return orig(sim)
        self._wrap(simmod.Simulation, "_resolve_door_inputs", din)

        def pump(orig, self_, sim):
            from simulation.logic_nodes import aggregate_input
            from simulation.entities.schema import INPUT_HELD
            if not self_.alive:
                c["pump: dead"] += 1
                return orig(self_, sim)
            bus = sim._signal_bus
            inj = aggregate_input(bus, self_.inject_slots, INPUT_HELD) != 0
            ext = aggregate_input(bus, self_.extract_slots, INPUT_HELD) != 0
            kind = "transfer" if self_.source is not None else "external"
            g = sim.gmap
            before = int(g.gas[O2][self_.port_y, self_.port_x])
            src0 = None if self_.source is None else int(
                g.gas[O2][self_.source[0], self_.source[1]] + g.gas[1][self_.source[0],
                                                                       self_.source[1]])
            a0 = self_.at_target
            orig(self_, sim)
            act = "extract" if ext else ("inject" if inj else "idle")
            c[f"pump {kind}: {act}{' (both: extract beats inject)' if inj and ext else ''}"] += 1
            if g.solid[self_.port_y, self_.port_x] and act != "idle":
                c["pump: port on a solid tile"] += 1
            if kind == "transfer" and act == "inject" and src0 is not None and src0 == 0:
                c["pump: source dry"] += 1
            c[f"pump: at_target {a0} -> {self_.at_target}"] += 1
            if self_.at_target_slot is None:
                c["pump: no at_target slot"] += 1
            _ = before
        self._wrap(ps.PumpRuntime, "sweep", pump)

        def swap(orig, self_):
            ch = sum(1 for i in self_._node_slots if self_.pub[i] != self_.stg[i])
            c["swap: a node slot changed" if ch else "swap: nothing changed"] += 1
            return orig(self_)
        self._wrap(SignalBus, "swap_node_signals", swap)
        return self

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()


NEED = (
    "emit: is_open", "emit: alive door", "emit: alive constant",
    "sensor: pressure", "sensor: smoke disc", "sensor: smoke", "sensor: water_depth",
    "sensor: temperature", "sensor: fire", "sensor: o2 disc", "sensor: clock", "sensor: dead",
    "sensor: sample tile solid", "disc: a solid tile excluded", "disc: all open",
    "motion: dead unit", "motion: out of range", "motion: other faction",
    "motion: footprint too small", "motion: behind a grille", "motion: behind a wall",
    "motion: counted",
    "decider gt -> 1", "decider gt -> 0", "decider ge -> 1", "decider ge -> 0",
    "decider lt -> 1", "decider lt -> 0", "decider le -> 1", "decider le -> 0",
    "decider eq -> 1", "decider eq -> 0", "decider ne -> 1", "decider ne -> 0",
    "decider: require_alive dead", "decider: require_alive alive",
    "gate and -> 1", "gate and -> 0", "gate and -> 0 (empty)", "gate or -> 1", "gate or -> 0",
    "not -> 1", "not -> 0", "not -> 1 (empty)", "filter: unwired",
    "airlock IDLE -> CLOSING", "airlock CLOSING -> EQUALIZE", "airlock EQUALIZE -> OPEN_FAR",
    "airlock OPEN_FAR -> RESEAL", "airlock RESEAL -> REPRESSURIZE",
    "airlock REPRESSURIZE -> IDLE",
    "airlock interior IDLE -> CLOSING", "airlock interior CLOSING -> EQUALIZE",
    "airlock interior EQUALIZE -> OPEN_FAR", "airlock interior OPEN_FAR -> RESEAL",
    "airlock interior RESEAL -> REPRESSURIZE", "airlock interior REPRESSURIZE -> IDLE",
    "9e(c): close", "9e(c): open", "9e(c): neither (latch kept)",
    "9e(c): close beats open", "9e(c): two wires into open",
    "pump external: extract", "pump external: inject", "pump external: idle",
    "pump transfer: extract", "pump transfer: inject", "pump: dead",
    "pump: port on a solid tile", "pump: source dry", "pump: at_target 0 -> 1",
    "pump: at_target 1 -> 1", "pump: no at_target slot",
    "swap: a node slot changed", "swap: nothing changed",
)
NEED_V0 = ("decider: require_alive unwired",)
UNREACHABLE = {
    "emit: dangling": "the loader drops a wire whose source id no entity declares (a "
                      "dangling emitter never reaches the bus); the device's EMIT_NONE row "
                      "is its form",
    "pump: (both: extract beats inject) from one controller": "an airlock's Moore outputs "
                      "never raise inject and extract together; p5 takes it from two nodes",
    "airlock CLOSING -> FAULT in the scene": "a door dying mid-CLOSING needs a blast at "
                      "that tick; PART AIRLOCK takes every FAULT transition",
}


def run_shadow(name, drivers, ticks, slots=None, n_units=None, twins=None):
    slots = slots or _slots()
    acts = Counter()
    n = 0
    t0 = time.perf_counter()
    with ShadowLockstep(drivers, list(slots), n_units=n_units) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            reps = ls.tick()
            n += 1
            for lab, rep in reps.items():
                if not rep.equal:
                    fail(f"{name}: {ls.divergences[-1]}")
                    return acts, n
                if rep.books is not None and "signal_writes" in rep.books:
                    acts[lab] += int(np.sum(rep.books["signal_writes"]) +
                                     np.sum(rep.books["applied"]) > 0)
            for e, d in enumerate(drivers):
                if not _identity(d.sim, f"{name} tick {d.sim.tick} env {e}"):
                    return acts, n
            if twins is not None:
                from cuda_unit_verbs_check import _same
                for d, tw in zip(drivers, twins):
                    tw.tick()
                    if not _same(d.sim, tw.sim):
                        fail(f"{name}: the hooked run diverged from the unhooked one at tick "
                             f"{d.sim.tick}")
                        return acts, n
    print(f"{name}: N={len(drivers)}, {n} ticks, the slots equal every tick (bus, tables, "
          f"planes, books, write set){' (env i == env i alone)' if len(drivers) > 1 else ''}"
          f"{' (the unhooked twin equal)' if twins is not None else ''}; ticks a sub-step "
          f"wrote: {dict(acts)} ({time.perf_counter() - t0:.0f} s)")
    return acts, n


def part_scene():
    census = Census().install()
    try:
        sc = SignalScene(0)
        acts, _ = run_shadow("scene", [sc], SCENE_TICKS, twins=[SignalScene(0)])
        run_shadow("scene planetside", [SignalScene(1)], SCENE_TICKS)
        run_shadow("level airlock_demo", [AirlockDriver()], LEVEL_TICKS)
    finally:
        census.uninstall()
    c = census.c
    for b in NEED + NEED_V0:
        if not c[b]:
            fail(f"scene: the census branch {b!r} was never taken")
    print("scene census: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    print("scene unreachable: " + "; ".join(f"{k} -- {v}" for k, v in UNREACHABLE.items()))


def part_batch():
    for n in BATCH_NS:
        # env 0 holds FEWER nodes than the next and one env has NO bus
        vs = ([2, 0] if n == 2 else [2, -1, 0, 1, 4, 3, 5, 6][:n])
        drivers = [SignalScene(v) for v in vs]
        acts, _ = run_shadow(f"batch N={n} (variants {vs})", drivers, BATCH_TICKS)
        counts = {len(uk.nodes_of(d.sim)) for d in drivers}
        if len(counts) < 2:
            fail(f"batch N={n}: every env holds the same node count (vacuous)")
        if n == 8 and not any(d.sim._signal_bus is None for d in drivers):
            fail("batch N=8: no env without a bus")


def part_padding():
    scs = [SignalScene(2), SignalScene(0)]
    for sc in scs:
        for _ in range(12):
            sc.tick()
    sims = [s.sim for s in scs]
    st = _store(sims)

    def run(dirty):
        tabs = uk.signal_tables(2, bp=bp)
        from simulation import object_tables
        doors = object_tables.DeviceTable(us.DOOR_TABLE, 2, bp=bp)
        dtiles = object_tables.DeviceTable(us.DOOR_TILE_TABLE, 2, bp=bp)
        staged = {}
        for ln_ in uk.SIGNAL_LISTS + ("doors",):
            t = tabs[ln_] if ln_ != "doors" else doors
            rows = np.zeros(t.shape, dtype=np.int64)
            cnt = []
            for e, sim in enumerate(sims):
                a = (uk.SIGNAL_RECORDS[ln_](sim) if ln_ != "doors" else uk.doors_of(sim))
                rows[e, :len(a)] = a
                cnt.append(len(a))
                if dirty and len(a):
                    rows[e, len(a):] = a[0]
            t.upload(rows, np.array(cnt))
            staged[ln_] = (rows.copy(), cnt)
        dtiles.stage_lists([uk.door_tiles_of(s) for s in sims])
        fields = _dev_planes([_planes(s.gmap) for s in sims])
        lists = dict(tabs, doors=doors)
        uk.signals_emit(st, _applier_for(st), fields, lists, np.array([s.tick for s in sims]))
        uk.signals_logic(st, _applier_for(st), lists)
        uk.signals_door_inputs(st, _applier_for(st), lists)
        uk.signals_pumps(st, _applier_for(st), fields, lists)
        uk.signals_swap(st, _applier_for(st), lists)
        out = {ln_: (tabs[ln_] if ln_ != "doors" else doors).download()[0]
               for ln_ in staged}
        return out, staged, {k: v.get() for k, v in fields.items()}
    clean, cs, cf = run(False)
    dirty, ds, df = run(True)
    for ln_ in cs:
        for e in range(2):
            n = cs[ln_][1][e]
            if not np.array_equal(clean[ln_][e, :n], dirty[ln_][e, :n]):
                fail(f"padding env {e}: garbage past {ln_}'s count changed its valid rows")
            if not np.array_equal(dirty[ln_][e, n:], ds[ln_][0][e, n:]):
                fail(f"padding env {e}: a padding row of {ln_} was written")
    for k in cf:
        if not np.array_equal(cf[k], df[k]):
            fail(f"padding: garbage rows changed plane {k}")
    if cs["nodes"][1][0] == cs["nodes"][1][1]:
        fail("padding: the two envs hold the same node count (vacuous)")
    print(f"padding: garbage rows past each of the nine tables' counts (node counts "
          f"{cs['nodes'][1]}) neither read nor written, through all five stages")


def part_overflow():
    sim = SignalScene(0).sim
    seen = Counter()
    for ln_, tn in zip(uk.SIGNAL_LISTS, uk._SIGNAL_TABLE_NAMES):
        rows = uk.SIGNAL_RECORDS[ln_](sim)
        if not len(rows):
            fail(f"overflow: the scene holds no {tn} rows (vacuous)")
            continue
        t = {x.name: x for x in us.OBJECT_TABLES}[tn]
        small = ot.DeviceTable(t, 2, cap=len(rows) - 1 if len(rows) > 1 else 1, bp=bp)
        try:
            small.stage_lists([rows[:1], np.concatenate([rows, rows])[:max(2, len(rows))]])
            fail(f"overflow: a full {tn} table took more rows than its capacity")
        except ot.ObjectTableOverflow as ex:
            if f"'{tn}'" not in str(ex) or "env 1" not in str(ex):
                fail(f"overflow: the {tn} refusal does not name the table and the env: {ex}")
            seen[tn] += 1
    for tab, rows in ((us.SIGNAL_TABLE, uk.signal_rows(sim)),
                      (us.NODE_TABLE, uk.node_rows(sim)),
                      (us.SENSOR_TABLE, uk.sensor_rows(sim))):
        many = (rows * (tab.capacity // len(rows) + 1))[:tab.capacity + 1]
        try:
            ot.pack(tab, many)
            fail(f"overflow: a {tab.name} list past the capacity was packed")
        except (ot.ObjectTableOverflow, ValueError) as ex:
            if f"'{tab.name}'" not in str(ex):
                fail(f"overflow: the {tab.name} list's refusal does not name it: {ex}")
            seen[f"{tab.name} list"] += 1
    print(f"overflow: every signal table past its capacity refused by name {dict(seen)}")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
_PUMP_GE = {"ge_pump_inject": "pump_inject", "ge_pump_extract": "pump_extract",
            "ge_pump_transfer": "pump_transfer",
            "ge_diag_pump_transfer_from": "diag_pump_transfer_from_nonacct",
            "ge_diag_pump_transfer_to": "diag_pump_transfer_to_nonacct"}


def _host_side_pumps(ch):
    """Slot 9e(d) pumps' host-only side after the device ran it: the books
    landed (a pump changes no tile)."""
    def hs(sim, _pre, _post):
        g = sim.gmap
        for name, key in _PUMP_GE.items():
            g._gas_energy_book(key, ch.books.get(name, 0))
        for s in range(N_GASES):
            g.trace_clip_books[s] += ch.books.get(f"trace_clip_{s}", 0)
    return hs


_HOST_LAWS = (("simulation.simulation", "Simulation._signal_emit"),
              ("simulation.simulation", "sample_sensors"),
              ("simulation.simulation", "sweep_logic_nodes"),
              ("simulation.simulation", "Simulation._resolve_door_inputs"),
              ("simulation.simulation", "sweep_pumps"),
              ("simulation.simulation", "sweep_vents"),
              ("simulation.simulation", "sweep_doors"),
              ("simulation.signal_bus", "SignalBus.swap_node_signals"))


def _chain(name, make, ticks):
    import importlib
    from cuda_unit_doors_check import SLOT_9ED, _host_side_9ed
    from cuda_unit_topology_check import _same_twin
    from cuda_unit_vents_check import SLOT_VENTS, _host_side_vents
    from simulation.entities.serialize import entity_records
    twin, ref = make(), make()
    calls = Counter()
    saved = []
    # count the host laws' calls per sim (the innermost wrappers)
    for mod, qual in _HOST_LAWS:
        m = importlib.import_module(mod)
        cont, attr = (m, qual) if "." not in qual else (getattr(m, qual.split(".")[0]),
                                                        qual.split(".")[1])
        orig = getattr(cont, attr)
        saved.append((cont, attr, orig))

        def counted(*a, __o=orig, __q=qual, **k):
            who = "twin" if any(x is twin.sim or x is getattr(twin.sim, "_signal_bus", 0)
                                for x in a) else "reference"
            calls[f"{who} {__q}"] += 1
            return __o(*a, **k)
        setattr(cont, attr, counted)
    holder = {}
    plan = [(SLOT_A, None), (SLOT_B, None), (SLOT_C, None),
            (SLOT_P, "p"), (SLOT_VENTS, "v"), (SLOT_9ED, "d"), (SLOT_E, None)]
    chs = []
    for slot, hk in plan:
        if hk is None:
            chs.append(ChainedSlots(twin, [slot], bp=bp))
            continue
        c = ChainedSlots(twin, [slot], bp=bp,
                         host_side=lambda s, a, b, __k=hk: holder[__k](s, a, b))
        holder[hk] = {"p": _host_side_pumps, "v": _host_side_vents,
                      "d": _host_side_9ed}[hk](c)
        chs.append(c)
    t0 = time.perf_counter()
    n = 0
    try:
        for c in chs:
            c.install()
        for _ in range(ticks):
            if getattr(twin, "is_terminal", lambda: False)():
                break
            twin.tick()
            ref.tick()
            n += 1
            why = _same_twin(twin.sim, ref.sim)
            for nm, f in list(uk.SIGNAL_RECORDS.items()) + [("doors", uk.doors_of),
                                                           ("ducts", uk.ducts_of),
                                                           ("vents", uk.vents_of)]:
                if why is None and f(twin.sim).tolist() != f(ref.sim).tolist():
                    why = f"the {nm} table"
            if why is None and entity_records(twin.sim.entities) != \
                    entity_records(ref.sim.entities):
                why = "the entity section"
            if why is None and twin.sim._digest_signals() != ref.sim._digest_signals():
                why = "the signal section"
            if why is not None:
                fail(f"chained {name}: the twin left the reference at tick {twin.sim.tick} "
                     f"({why})")
                break
    finally:
        for c in reversed(chs):
            c.uninstall()
        for cont, attr, orig in reversed(saved):
            setattr(cont, attr, orig)
    twin_calls = {k: v for k, v in calls.items() if k.startswith("twin")}
    if twin_calls:
        fail(f"chained {name}: the host laws ran in the twin {twin_calls}")
    runs = [c.runs for c in chs]
    print(f"chained {name}: {n} ticks, the twin's unit hash, field digest, planes, books, "
          f"signal / sensor / node / pump / drive / door / duct / vent tables, entity and "
          f"signal sections equal the reference every tick (chain runs per sub-step "
          f"{runs}; host law calls {dict(calls)}; {time.perf_counter() - t0:.0f} s)")
    return chs, n, calls


def part_chained():
    from cuda_unit_objects_check import ShipDriver
    for v in (0, 1):
        chs, n, calls = _chain(f"signal scene v{v}", lambda: SignalScene(v), CHAIN_TICKS)
        if any(c.runs != n for i, c in enumerate(chs) if i != 4) or chs[4].runs != 0:
            fail(f"chained signal scene v{v}: runs {[c.runs for c in chs]} in {n} ticks "
                 f"(vacuous or skipped; the scene has no duct)")
    chs, n, calls = _chain("airlock_demo", AirlockDriver, LEVEL_TICKS)
    if any(c.runs != n for i, c in enumerate(chs) if i != 4):
        fail(f"chained airlock_demo: runs {[c.runs for c in chs]} in {n} ticks")
    chs, n, calls = _chain("ship_demo bots match (no bus: the signal chains idle)",
                           ShipDriver, SHIP_TICKS)
    if any(c.runs for i, c in enumerate(chs) if i not in (4, 5)) or chs[4].runs != n:
        fail(f"chained ship: runs {[c.runs for c in chs]} in {n} ticks (the signal chains "
             f"must idle, the vents run every tick)")


# ---------------------------------------------------------------------------
PARTS = ("tables", "sample", "logic", "airlock", "refusal", "scene", "batch", "padding",
         "overflow", "chained")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_SIGNALS_RESULT: FAIL (not the CUDA build)")
        return 1
    from cuda_unit_vents_check import gate_config
    gate_config()
    parts = parts or os.environ.get("UNIT_SIGNALS_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "sample": part_sample, "logic": part_logic,
             "airlock": part_airlock, "refusal": part_refusal, "scene": part_scene,
             "batch": part_batch, "padding": part_padding, "overflow": part_overflow,
             "chained": part_chained}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
        sys.stdout.flush()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_SIGNALS_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table (median ms over ``reps``): each signal sub-step for N =
    1 / 8 / 64 copies of airlock_demo and of the signal scene -- the launch,
    its books' D2H and the Python face (each rep re-stages the tables and the
    planes, not timed)."""
    from cuda_unit_vents_check import gate_config
    from simulation import object_tables
    gate_config()
    ad = AirlockDriver()
    for _ in range(30):
        ad.tick()
    sc = SignalScene(0)
    for _ in range(30):
        sc.tick()
    print("TIMING (median ms over %d reps)" % reps)
    print("| sub-step | map | N=1 | N=8 | N=64 |")
    for label, sim in (("airlock_demo", ad.sim), ("the signal scene", sc.sim)):
        recs = {ln_: f(sim) for ln_, f in uk.SIGNAL_RECORDS.items()}
        planes = _planes(sim.gmap)
        res = {}
        for n in (1, 8, 64):
            sims = [sim] * n
            st = _store(sims)
            tabs = uk.signal_tables(n, bp=bp)
            doors = object_tables.DeviceTable(us.DOOR_TABLE, n, bp=bp)
            fields = _dev_planes([planes] * n)
            ticks = np.full(n, sim.tick)
            ops = (("9e(a) emit + sensors", lambda: uk.signals_emit(
                        st, _applier_for(st), fields, dict(tabs, doors=doors), ticks)),
                   ("9e(b) logic", lambda: uk.signals_logic(st, _applier_for(st), tabs)),
                   ("9e(c) door inputs", lambda: uk.signals_door_inputs(
                        st, _applier_for(st), dict(tabs, doors=doors))),
                   ("9e(d) pumps", lambda: uk.signals_pumps(st, _applier_for(st), fields,
                                                            tabs)),
                   ("9e(e) swap", lambda: uk.signals_swap(st, _applier_for(st), tabs)))
            for op, fn in ops:
                ts = []
                for _ in range(reps):
                    for ln_ in uk.SIGNAL_LISTS:
                        tabs[ln_].stage_lists([recs[ln_]] * n)
                    doors.stage_lists([uk.doors_of(sim)] * n)
                    s = time.perf_counter()
                    fn()
                    ts.append(time.perf_counter() - s)
                res.setdefault(op, []).append(float(np.median(ts)))
        for op, row in res.items():
            print(f"| {op} | {label} | " + " | ".join(f"{x * 1e3:.3f}" for x in row) + " |")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        sys.exit(bench())
    sys.exit(main())
