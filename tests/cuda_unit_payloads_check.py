"""#29 P7c.1 -- THE PAYLOADS GATE (the disc path): ``payloads.execute_payload``
for a row with ``strength = 0`` (no bulk gas, no explosive) as the PAYLOAD
effect function of the per-env serial applier (``cpp/src/cuda_unit_payload.cuh``)
against the Python law at tol 0 -- ``physics.apply_explosion`` (the disc, its
wall damage and destroy_wall at once, the wave stencil on the live planes, the
atmosphere and smoke-clear DISC edits), ``exchange.apply_blast_damage`` (the
IMMEDIATE packet), ``flash_blind.apply_flash_blind``, ``emit_gas``,
``ignite_ring``, ``deposit_heat`` (THE DEVICE EDIT DOOR's DISC form,
``cpp/src/cuda_edit_door.cuh``), the claims, ``noise.emit`` and the blast log
-- at every call site (slot 2's grenades and slot 9e(d')'s charges in the
DETONATE stage; a payload round's stop in slot 4c and in slot 2''s landing,
the RESOLVE stage). Design 4.4c, the cut table's P7c.1 row.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_payloads.py).

PART TABLES -- every P7c.1 device-table row equals its reference value (two
tile sizes); the every-row gate (cuda_unit_kernels_check) reads
``table_reference``.
PART PROBE -- synthetic PAYLOAD lists on random maps (walls of every
material, low hp, vacuum pockets; units of random hp, facing and BLINDED
state; centres near and past the edge; owners and the environment), each
env's list walked by the device in the DETONATE or the RESOLVE stage and
applied by the host's ``execute_payload`` (in a resolve: through
``SimultaneousEffects`` with HITs before): rows, synced events, every plane,
the books, the door's rows (``edit_door.door_rows`` of the host's edits),
the claims, the noise list and the blast log equal; a census of the law's
branches, each required.
PART DISC -- every DISC edit's door rows against ``door_rows`` of the host's
same FieldEdit (fields, radii, falloffs, clamps, centres off the grid).
PART SCENE -- the SHADOW LOCKSTEP of slot 2 (grenades + DETONATE), slot 2'
(the march) and its landing, slot 4c and slot 9e(d') (charges + DETONATE) on
the payload scene, in row sets that put every shipped strength = 0 row
through every call site, and a variant set (rows raised to hurt and blind)
that kills with a blast, breaks walls, orders two detonations on one tick, a
blast-killed unit with an exposure raise pending in its resolve (the
landing's corpse skip) and a claimed kill; a census.
PART BATCH -- N = 2 and 8 of differing envs (layout, seed, unit count, the
fields): env i == env i alone every tick.
PART PADDING -- garbage past every list's count and in padding rows: the
valid results equal a clean run.
PART OVERFLOW -- each list the function writes (the door's edits and tiles,
the claims, the noise list, the blast log, the ring) one short: refused by
name.
PART REFUSE -- a strength row, a bulk-gas row, an explosive row: refused by
name, nothing written; a PAYLOAD without its lists faults.
PART CHAINED -- slot 2 (grenades + DETONATE), 2' + landing, 4 + 4a + 4b + 4c
and 9e(d') on the device in a twin match: unit hash, field digest, the edit
queue, the noise and blast logs equal an untouched reference every tick; the
host execute_payload run 0 times in the twin.
PART SHIP -- ship_demo's bots match, 500 ticks: slot 2's grenades and 4c
shadowed (a refused row named), and chained with the refused rows replayed.

PROPERTY: a PAYLOAD record of a disc-path row is ``execute_payload``
integer for integer in every stage and at every call site; a refused row
writes nothing and is refused by name. BREAKS IF a payload row, the disc's
order or material filter, the wall damage, the stencil's live read, a DISC
edit's field / radius / amount / mode / falloff / key / source, the blast
damage's falloff / threshold / row order / immediate form, the flash's
sight / facing / duration / never-shorten rule, a claim, the noise, the
blast log, the event order, the landing's corpse skip, the call order within
a tick or the batch indexing differs from the law.

Prints per-part lines and ``UNIT_PAYLOADS_RESULT: PASS`` / ``FAIL``; exits
0 / 1. ``bench()`` prints the TIMING table.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
import cuda_unit_lockstep as LS  # noqa: E402
from cuda_unit_lockstep import (ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep)
from simulation import (applier, attack_lists, edit_door, object_tables as ot,  # noqa: E402
                        unit_events, unit_kernels, unit_schema, unit_store,
                        unit_tables)

FAILS: list = []
ONE = 65536
PI_Q16 = 205887
SPACE = level_loader.SPACE_CODE
EAST, WEST, SOUTH, NORTH = 0, PI_Q16, -PI_Q16 // 2, PI_Q16 // 2
PROBE_TRIALS = int(os.environ.get("UNIT_PAYLOADS_PROBE_TRIALS", "60"))
SCENE_TICKS = int(os.environ.get("UNIT_PAYLOADS_SCENE_TICKS", "90"))
BATCH_TICKS = int(os.environ.get("UNIT_PAYLOADS_BATCH_TICKS", "60"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_PAYLOADS_BATCH_NS", "2,8").split(",")]
CHAIN_TICKS = int(os.environ.get("UNIT_PAYLOADS_CHAIN_TICKS", "90"))
SHIP_TICKS = int(os.environ.get("UNIT_PAYLOADS_SHIP_TICKS", "500"))


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


#: The shipped rows the function applies (payload_refuse 0), by name.
def applied_rows():
    from simulation.weapons import get_tables
    return [n for n, p in get_tables().payloads.by_name.items()
            if applier.payload_refusal(p) == 0]


def refused_rows():
    from simulation.weapons import get_tables
    return {n: applier.payload_refusal(p)
            for n, p in get_tables().payloads.by_name.items()
            if applier.payload_refusal(p) != 0}


# ---------------------------------------------------------------------------
# Shared tables and the row variants
# ---------------------------------------------------------------------------
def share_tables(sims):
    """Every sim on the process's weapon tables (one tile size): a payload
    round carries the PROCESS table's PayloadDef (combat.weapon_tables()), and
    ``applier.payload_index`` names a row of the sim's own table by identity,
    so a batch of sims built one after another shares the last one's tables
    (equal in content at one tile size; a gate choice, the reference's
    finding stands)."""
    from simulation.weapons import get_tables
    t = get_tables()
    for s in sims:
        if abs(float(s.gmap.tile_size_m) - float(sims[0].gmap.tile_size_m)) > 0:
            raise ValueError("share_tables: one tile size per batch")
        s.weapons_tables = t
    return t


class Rows:
    """A context that sets attributes of the shared table's payload rows and
    ammo rows (``{"payloads.<name>": {attr: value}, "ammo.<name>": {"payload":
    <row>}}``) and restores them on exit: the device rows read the same
    objects (``unit_tables`` reads ``sim.weapons_tables``)."""

    def __init__(self, changes):
        self.changes = changes
        self.saved = []

    def __enter__(self):
        from simulation.weapons import get_tables
        t = get_tables()
        for key, attrs in self.changes.items():
            kind, name = key.split(".", 1)
            obj = (t.payloads if kind == "payloads" else t.ammo).by_name[name]
            for a, v in attrs.items():
                self.saved.append((obj, a, getattr(obj, a)))
                setattr(obj, a, v)
        return self

    def __exit__(self, *exc):
        for obj, a, v in reversed(self.saved):
            setattr(obj, a, v)
        self.saved.clear()
        return False


#: The variant rows (shipped rows raised so the disc path hurts, blinds and
#: breaks walls; strength 0, no explosive: still the disc path).
VARIANTS = {
    "payloads.plasma_splash_small": dict(unit_damage=200),
    "payloads.plasma_splash_large": dict(unit_damage=40, blind_radius=8.0,
                                         blind_radius_q=8 * ONE, blind_ticks=60,
                                         blind_away_q=ONE // 4),
    "payloads.incendiary_splash": dict(radius=4, pressure=3.0, wall_damage=900,
                                       unit_damage=60),
    "payloads.studio_smoke": dict(radius=3, pressure=1.0, wall_damage=500,
                                  unit_damage=30, blind_radius=6.0,
                                  blind_radius_q=6 * ONE + ONE // 2, blind_ticks=40,
                                  blind_away_q=0),
    "payloads.hazmat_release": dict(radius=2, pressure=0.5, wall_damage=40),
}


# ---------------------------------------------------------------------------
# Planes, books, the host's lists
# ---------------------------------------------------------------------------
PLANE_OUTPUTS = tuple("field:" + n for n in applier.PLANE_NAMES)
BOOKS = ("n_destruction_seed", "ge_retire", "ge_mint", "ge_destroy_seed",
         "solid_destroy_wall", "tiles_changed")
BOOK_OUTPUTS = tuple("book:" + b for b in BOOKS)
LIST_OUTPUTS = ("list:edits", "list:claims", "list:noise", "list:blast_log")
HOST_ONLY = (
    ("render_dirty", "the wall seam's render mark (a blast's destroy; no device form)"),
    ("_sky_mask_dirty", "the sky exchange's lazy-rebuild flag (ambient maps)"),
    ("_gas_t_amb_raw_cache", "a lazily filled cache of a config constant"),
    ("damage_credit", "the TickCredit dict: its device form is the claims list "
                      "(list:claims, every claim in order)"),
    ("noise_log", "the noise list: compared as list:noise"),
    ("blast_log", "the blast log: compared as list:blast_log"),
)


def _planes(gmap) -> dict:
    return {n: np.array(getattr(gmap, n), copy=True) for n in applier.PLANE_NAMES}


def _dev_planes(per_env: list) -> dict:
    import cupy as cp
    return {n: cp.asarray(np.ascontiguousarray(np.stack([p[n] for p in per_env])))
            for n in applier.PLANE_NAMES}


def _books(gmap) -> dict:
    return {b: applier.host_book(gmap, LS.book_ref(b)) for b in BOOKS}


def _walk_lists(lists, cells=False):
    return unit_kernels.WalkLists(
        cells=lists.get("spray_cells") if cells else None,
        door=lists["edits"].door, claims=lists["claims"], noise=lists["noise"],
        blast_log=lists["blast_log"])


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
def _run_2g(st, inp, ring, lists):
    unit_kernels.grenades(st, lists["grenades"], lists["payloads"], inp.tick)
    lists["edits"].set_seq(inp.scalars["edit_seq"])
    return unit_kernels.detonate(st, lists["payloads"], inp.fields, ring, _walk_lists(lists))


def _run_9e(st, inp, ring, lists):
    unit_kernels.charges_tick(st, lists["charges"], lists["payloads"], inp.tick)
    lists["edits"].set_seq(inp.scalars["edit_seq"])
    return unit_kernels.detonate(st, lists["payloads"], inp.fields, ring, _walk_lists(lists))


def _run_resolve(st, inp, ring, lists):
    lists["edits"].set_seq(inp.scalars["edit_seq"])
    return unit_kernels.resolve(st, lists["effects"], inp.fields, ring,
                                _walk_lists(lists, cells="spray_cells" in lists))


def _conductor(sim, args, kw, caller):
    return caller == "_step_pre_physics"


def _landing(sim, args, kw, caller):
    return caller == "_update_rounds"


DET_IN = ("tick", "edit_seq", "list:noise", "list:blast_log")
SLOT_2G = DeviceSlot(label="2 grenades", run=_run_2g, events=True,
                     inputs=DET_IN + ("list:grenades",),
                     outputs=PLANE_OUTPUTS + BOOK_OUTPUTS + LIST_OUTPUTS
                     + ("list:grenades", "list:payloads"), host_only=HOST_ONLY)
SLOT_9E = DeviceSlot(label="9e(d') planted charges", run=_run_9e, events=True,
                     skip_ok=True, inputs=DET_IN + ("list:charges",),
                     outputs=PLANE_OUTPUTS + BOOK_OUTPUTS + LIST_OUTPUTS
                     + ("list:charges", "list:payloads"), host_only=HOST_ONLY)
RES_IN = ("list:effects", "edit_seq", "list:noise", "list:blast_log")
SLOT_2L = DeviceSlot(label="4c resolve", name="2' landing", run=_run_resolve, events=True,
                     inputs=RES_IN, outputs=PLANE_OUTPUTS + BOOK_OUTPUTS + LIST_OUTPUTS,
                     host_only=HOST_ONLY, select=_landing)
SLOT_4C = DeviceSlot(label="4c resolve", run=_run_resolve, events=True,
                     inputs=RES_IN + ("list:spray_cells",),
                     outputs=PLANE_OUTPUTS + BOOK_OUTPUTS + LIST_OUTPUTS,
                     host_only=HOST_ONLY, select=_conductor)
SCENE_SLOTS = [SLOT_2G, SLOT_2L, SLOT_4C, SLOT_9E]


# ---------------------------------------------------------------------------
# Twin compare
# ---------------------------------------------------------------------------
def _unit_hash(sim):
    from field_ab_harness import _capture_unit_state
    return _capture_unit_state(sim)["hash"]


def _field_digest(sim):
    from field_ab_harness import SIM_FIELDS, _snapshot
    from field_digest import field_digest
    return field_digest(_snapshot(sim.gmap, SIM_FIELDS))


def _queue_state(sim):
    return [(e.field, int(e.region), tuple(e.coords) if e.region != 4 else None,
             int(e.mode), int(e.falloff), e.amount_q, int(e.source_id), e.channel, s)
            for e, s in sim.edit_queue._edits] + [("seq", sim.edit_queue._seq)]


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    """The P7c.1 device-table rows read from the reference objects, each by
    the law's own quantization (the every-row gate reads this too)."""
    import math
    from config import CFG
    from simulation import field_edit as fe, flash_blind, payloads, physics
    from simulation import unit_fixed as ufx, wall_fixed
    from simulation.gases import SMOKE
    from simulation.materials import MATERIAL_NAMES
    P = list(sim.weapons_tables.payloads.by_name.values())
    gases = sim.gmap.gases

    def cur(v, field):
        return fe._quantize_in(float(v), fe._CURRENCY[fe.FIELD_POLICY[field].dtype])
    inner = []
    for p in P:
        q = ufx.div_round(int(p.radius) * 2 * ufx.FP_ONE, 5)
        inner.append(ufx.quantize_scalar(ufx.dequantize_scalar(q)))
    clamp = fe._clamp_q(payloads.IGNITE_CLAMP, fe._CURRENCY["fire"])
    return {
        "payload_refuse": tuple(applier.payload_refusal(p) for p in P),
        "payload_radius": tuple(int(p.radius) for p in P),
        "payload_radius_q": tuple(ufx.quantize_scalar(float(p.radius)) for p in P),
        "payload_atm_q": tuple(cur(p.pressure, "atmosphere") for p in P),
        "payload_press_q": tuple(ufx.quantize_scalar(float(p.pressure)) for p in P),
        "payload_wall_q": tuple(wall_fixed.quantize_scalar(float(p.wall_damage)) for p in P),
        "payload_inner_q": tuple(inner),
        "payload_blast_live": tuple(int(p.unit_damage > 0) for p in P),
        "payload_unit_damage": tuple(int(p.unit_damage) for p in P),
        "payload_loudness_tiles": tuple(int(p.loudness_tiles) for p in P),
        "payload_blind_radius_q": tuple(int(p.blind_radius_q) for p in P),
        "payload_blind_ticks": tuple(int(p.blind_ticks) for p in P),
        "payload_blind_away_q": tuple(int(p.blind_away_q) for p in P),
        "payload_gas_id": tuple(int(gases.name_to_id[p.gas_species]) if p.gas_species
                                else -1 for p in P),
        "payload_gas_q": tuple(cur(p.gas_amount, "gas") for p in P),
        "payload_gas_radius_q": tuple(ufx.quantize_scalar(float(p.gas_radius)) for p in P),
        "payload_ignite_live": tuple(int(p.ignite_radius > 0) for p in P),
        "payload_ignite_radius_q": tuple(ufx.quantize_scalar(float(p.ignite_radius))
                                         for p in P),
        "payload_ignite_q": tuple(cur(p.ignite_intensity, "fire") for p in P),
        "payload_heat_live": tuple(int(p.heat_amount > 0) for p in P),
        "payload_heat_q": tuple(ufx.quantize_scalar(float(p.heat_amount)) for p in P),
        "payload_heat_radius_q": tuple(ufx.quantize_scalar(float(p.heat_radius)) for p in P),
        "payload_src_explosion": physics._SRC_EXPLOSION,
        "payload_src_gas": payloads._SRC_PAYLOAD_GAS,
        "payload_src_ignite": payloads._SRC_PAYLOAD_IGNITE,
        "payload_src_heat": payloads._SRC_PAYLOAD_HEAT,
        "payload_smoke_slice": int(SMOKE),
        "payload_smoke_clear_q": cur(physics._SMOKE_CLEAR_AMOUNT, "smoke"),
        "payload_ignite_heat_q": cur(payloads.IGNITE_HEAT, "heat"),
        "payload_ignite_clamp": (3, int(clamp[0]), int(clamp[1])),
        "blast_wall_mat": tuple(int(i in physics._BLAST_WALL_MATERIALS)
                                for i in range(len(MATERIAL_NAMES))),
        "blast_damage_threshold": int(math.ceil(float(CFG.combat.blast_damage_threshold))),
        "flash_front_q16": int(flash_blind.FRONT_Q16),
    }


def part_tables():
    sims = [PayloadScene(0).sim, PayloadScene(1, tile_m=1.0 / 3.0).sim]
    got = unit_store.DeviceStore(len(sims), 4,
                                 tables=unit_tables.values_batch(sims)).tables.download()
    n = 0
    as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
    for e, sim in enumerate(sims):
        ref = table_reference(sim)
        miss = [r.name for r in unit_tables._PAYLOAD_ROWS if r.name not in ref]
        if miss:
            fail(f"tables: no reference for {miss}")
        for k, v in ref.items():
            dv = unit_tables.read(got[e], k)
            n += 1
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    l0 = unit_tables.read(got[0], "payload_loudness_tiles")
    l1 = unit_tables.read(got[1], "payload_loudness_tiles")
    if tuple(l0) == tuple(l1):
        fail("tables: two tile sizes gave one loudness row")
    print(f"tables: {n} P7c.1 values over 2 envs (1 m and 1/3 m tiles) equal the "
          f"reference objects; applied rows {applied_rows()}; refused "
          f"{ {k: v for k, v in refused_rows().items()} }")


# ---------------------------------------------------------------------------
# The worlds
# ---------------------------------------------------------------------------
def _probe_level(rng, h, w, name):
    from level_loader import LevelData
    from simulation.materials import (MAT_AIR, MAT_DOOR, MAT_DOOR_CLOSED, MAT_FURNITURE,
                                      MAT_GLASS, MAT_GRILLE, MAT_HULL, MAT_KINDLING,
                                      MAT_STEEL, MAT_WOOD)
    tm = np.full((h, w), SPACE, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_HULL
    tm[2:h - 2, 2:w - 2] = MAT_AIR
    mats = [MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_DOOR_CLOSED, MAT_GRILLE, MAT_GLASS,
            MAT_STEEL, MAT_FURNITURE, MAT_KINDLING]
    roll = rng.random((h, w))
    inner = np.zeros((h, w), dtype=bool)
    inner[2:h - 2, 2:w - 2] = True
    pick = rng.integers(0, len(mats), size=(h, w))
    sel = inner & (roll < 0.22)
    tm[sel] = np.asarray(mats)[pick[sel]]
    vac = inner & (roll > 0.985)                 # vacuum pockets inside
    tm[vac] = SPACE
    return LevelData(name=name, version="3", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."), boundary="space")


def _probe_sim(rng, k, hw=None):
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    from simulation.status import BLINDED, apply_status
    from simulation.turning import place
    from simulation.unit import Unit
    h, w = hw if hw is not None else (int(rng.integers(20, 34)), int(rng.integers(24, 44)))
    sim = Simulation(_probe_level(rng, h, w, f"p7c1_probe{k}"), seed=900 + k,
                     breach_physics=bp, enable_recorder=False, ruleset=ContinuousRealtime())
    sim._next_unit_id += 5 + k % 7               # ids never the row index
    g = sim.gmap
    from simulation.physics import _BLAST_WALL_MATERIALS
    for y, x in np.argwhere(np.isin(g.material, _BLAST_WALL_MATERIALS)):
        if rng.random() < 0.7:                   # a topology value: low hp
            g.wall_hp[y, x] = int(rng.integers(1, 60 * ONE))
    open_t = np.argwhere(~g.solid)
    for i in range(int(rng.integers(4, 12))):
        y, x = (int(v) for v in open_t[int(rng.integers(len(open_t)))])
        u = Unit(f"u{i}", x=float(x), y=float(y), team=int(rng.integers(0, 2)))
        sim.add_unit(u)
        place(u, int(rng.integers(-PI_Q16, PI_Q16)))
        u.current_hp = float(rng.choice([3.0, 20.0, 45.0, 90.0, 400.0]))
        if rng.random() < 0.35:
            apply_status(u, BLINDED, 0, int(rng.integers(1, 90)))
    return sim


# ---------------------------------------------------------------------------
# PART PROBE: a list walked by the device and applied by the host
# ---------------------------------------------------------------------------
class ProbeCensus:
    """The Python law's branches, counted on the host's own decisions
    (reads only): wraps exchange.apply_packet, flash_blind.apply_status,
    GameMap.has_los, GameMap.destroy_wall, noise.emit, the edit enqueue and
    the claim."""

    def __init__(self):
        self.c = Counter()
        self._saved = []

    def __enter__(self):
        from simulation import exchange, flash_blind, gamemap, noise
        from simulation.field_edit import EditQueue
        c = self.c
        o_ap = exchange.apply_packet

        def ap(u, pk, events=None, source=None, **kw):
            was = u.alive
            r = o_ap(u, pk, events, source=source, **kw)
            c["blast: a packet lands"] += 1
            if was and not u.alive:
                c["blast: a kill"] += 1
            return r
        exchange.apply_packet = ap
        self._saved.append((exchange, "apply_packet", o_ap))
        o_st = flash_blind.apply_status

        def st(u, kind, **kw):
            c["flash: BLINDED applied"] += 1
            if any(s.kind == kind for s in (getattr(u, "statuses", None) or ())):
                c["flash: a shorter BLINDED refreshed longer"] += 1
            return o_st(u, kind, **kw)
        flash_blind.apply_status = st
        self._saved.append((flash_blind, "apply_status", o_st))
        o_dw = gamemap.GameMap.destroy_wall

        def dw(g, y, x, *a, **k):
            c["disc: a wall destroyed"] += 1
            return o_dw(g, y, x, *a, **k)
        gamemap.GameMap.destroy_wall = dw
        self._saved.append((gamemap.GameMap, "destroy_wall", o_dw))
        o_em = noise.emit

        def em(gmap, fy, fx, radius):
            c["noise: a noise logged" if radius > 0 else "noise: radius 0 (silent)"] += 1
            return o_em(gmap, fy, fx, radius)
        noise.emit = em
        self._saved.append((noise, "emit", o_em))
        o_en = EditQueue.enqueue

        def en(q, e):
            c[f"edit: {e.field} {e.region.name}"] += 1
            return o_en(q, e)
        EditQueue.enqueue = en
        self._saved.append((EditQueue, "enqueue", o_en))
        return self

    def __exit__(self, *exc):
        for cont, attr, o in reversed(self._saved):
            setattr(cont, attr, o)
        self._saved.clear()
        return False


def _flash_census(c, sim, fy, fx, p):
    """The flash's per-unit decisions, from the law's own functions."""
    from simulation import flash_blind, unit_fixed
    from simulation.status import BLINDED
    if p.blind_radius_q <= 0:
        return
    r_tiles = (p.blind_radius_q >> 16) + 1
    for u in sim.units:
        if not u.alive:
            c["flash: a dead unit skipped"] += 1
            continue
        dx, dy = fx - u.center_tile_x(), fy - u.center_tile_y()
        d2 = dx * dx + dy * dy
        if d2 > r_tiles * r_tiles:
            c["flash: out of the bound"] += 1
            continue
        if not sim.gmap.has_los(u.center_tile_y(), u.center_tile_x(), fy, fx):
            c["flash: no line of sight"] += 1
            continue
        c["flash: sees the flash"] += 1
        fq = unit_fixed.quantize_scalar(u.facing)
        f = flash_blind.facing_factor_q16(dx, dy, fq, p.blind_away_q)
        c["flash: facing it (f = 1)" if f == ONE else "flash: turned away (f < 1)"] += 1
        t = flash_blind.blind_ticks_for(p, d2, dx, dy, fq)
        if flash_blind.distance_factor_q16(d2, p.blind_radius_q) == 0:
            c["flash: at or past the radius (g = 0)"] += 1
        elif t < 1:
            c["flash: under one tick"] += 1
        elif any(s.kind == BLINDED and s.remaining_ticks >= t for s in u.statuses):
            c["flash: already blinded longer (kept)"] += 1


def _blast_census(c, sim, fy, fx, p):
    from config import CFG
    from simulation import unit_fixed as ufx
    if not p.unit_damage > 0:
        return
    r_q = ufx.quantize_scalar(float(p.radius))
    for u in sim.units:
        if not u.alive:
            c["blast: a dead unit skipped"] += 1
            continue
        dx, dy = u.center_tile_x() - fx, u.center_tile_y() - fy
        d2 = (dx * dx + dy * dy) << 32
        if d2 > r_q * r_q:
            c["blast: outside the radius"] += 1
            continue
        dmg = ufx.falloff_floor(int(p.unit_damage), d2, r_q * r_q)
        c["blast: below the threshold" if dmg < CFG.combat.blast_damage_threshold
          else "blast: at or above the threshold"] += 1


def _disc_census(c, sim, fy, fx, p):
    """apply_explosion's disc decisions on the planes as they stand, run
    beside the host's own call: a blast wall damaged / not destroyed, a
    non-blast solid left, and a wave tile whose stencil reads a tile the same
    disc destroyed earlier (the live read)."""
    from simulation.physics import _BLAST_WALL_MATERIALS
    if p.radius <= 0:
        return
    g = sim.gmap
    h, w = g.material.shape
    r = int(p.radius)
    order = [(fy + dy, fx + dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
             if 0 <= fy + dy < h and 0 <= fx + dx < w and dy * dy + dx * dx <= r * r]
    return order, {t: (int(g.material[t]), bool(g.solid[t])) for t in order}


def _disc_after(c, sim, info):
    from simulation.materials import MAT_AIR
    from simulation.physics import _BLAST_WALL_MATERIALS
    if info is None:
        return
    order, before = info
    g = sim.gmap
    gone = []
    for i, t in enumerate(order):
        m0, s0 = before[t]
        if m0 in _BLAST_WALL_MATERIALS:
            if int(g.material[t]) == MAT_AIR and m0 != MAT_AIR:
                gone.append(i)
            else:
                c["disc: a blast wall damaged, standing"] += 1
        elif s0:
            c["disc: a non-blast solid untouched"] += 1
    for i in gone:
        ty, tx = order[i]
        for j in range(i + 1, len(order)):
            uy, ux = order[j]
            if abs(uy - ty) <= 1 and abs(ux - tx) <= 1 and not g.solid[order[j]]:
                c["disc: a stencil reads a tile destroyed earlier in the disc"] += 1
                break


def _record(sim, call):
    from simulation import payloads, simultaneous
    return applier.record_of(sim, (simultaneous._WORLD, payloads.execute_payload,
                                   call[0], call[1]))


def compare_walk(sims, lists_of, stage, census=None, what="probe"):
    """Every env's list (``lists_of[e]``: ("payload", args, kw) / ("hit",
    unit, packet, source) items) walked by the device in ``stage`` and applied
    by the host (a ``SimultaneousEffects`` resolved: the landing for the
    HITs, the payload calls at once), then everything compared. Returns the
    counts compared."""
    from simulation import payloads
    from simulation.damage import tick_credit
    from simulation.simultaneous import SimultaneousEffects
    N = len(sims)
    U = max(len(s.units) for s in sims)
    tot = Counter()
    pre = []
    recs = []
    fxs = []
    for e, sim in enumerate(sims):
        fx = SimultaneousEffects()
        for it in lists_of[e]:
            if it[0] == "payload":
                fx.world(payloads.execute_payload, *it[1], **it[2])
            else:
                fx.hit(it[1], it[2], sim.tick_events, source=it[3])
        fxs.append(fx)
        recs.append(applier.records_of(sim, fx._pending))
        pre.append(dict(
            planes=_planes(sim.gmap), rows=unit_store.pack(sim.units, U),
            noise=unit_kernels.noise_of(sim), blast=unit_kernels.blast_log_of(sim),
            seq=int(sim.edit_queue._seq), nq=len(sim.edit_queue._edits),
            nev=len(sim.tick_events), books=_books(sim.gmap)))
        LS._CLAIM_LOG[id(tick_credit(sim.gmap))] = []
    st = unit_store.DeviceStore(N, U, tables=unit_tables.values_batch(sims))
    st.upload(np.stack([p["rows"] for p in pre]))
    ring = unit_events.EventRing(N, U)
    ring.begin()
    dl = applier.DeviceApplier(N, applier.E_CAP, n_units=U)
    dl.stage_lists(recs)
    door = edit_door.DeviceEditDoor(N)
    door.begin()
    door.set_seq(np.array([p["seq"] for p in pre], dtype=np.int64))
    claims = attack_lists.claims_list(N)
    claims.begin()
    noise = ot.DeviceTable(unit_schema.NOISE_TABLE, N)
    noise.stage_lists([p["noise"] for p in pre])
    blast = ot.DeviceTable(unit_schema.BLAST_LOG_TABLE, N)
    blast.stage_lists([p["blast"] for p in pre])
    planes = _dev_planes([p["planes"] for p in pre])
    wl = unit_kernels.WalkLists(door=door, claims=claims, noise=noise, blast_log=blast)
    _cov, books = unit_kernels.walk_effects(st, dl, planes, stage, ring, wl)
    # the host
    for e, sim in enumerate(sims):
        if census is None:
            fxs[e].resolve()
        else:
            _resolve_counting(sim, fxs[e], census)
    got_rows = st.download()
    dev_rings = ring.records()
    dev_door = door.rows()
    dev_claims = claims.records()
    dev_noise = noise.records()
    dev_blast = blast.records()
    for e, sim in enumerate(sims):
        p = pre[e]
        want = unit_store.pack(sim.units, U)
        if not np.array_equal(got_rows[e], want):
            r, c = (int(x) for x in np.argwhere(got_rows[e] != want)[0])
            fail(f"{what} env {e}: row {r} column {_col_name(c)}: host "
                 f"{int(want[r, c])} device {int(got_rows[e][r, c])}")
            return tot
        hev = [unit_events.record_of(ev) for ev in sim.tick_events[p["nev"]:]
               if unit_events.is_synced(ev)]
        if [tuple(x) for x in hev] != [tuple(x) for x in dev_rings[e]]:
            fail(f"{what} env {e}: the events differ ({len(hev)} host / "
                 f"{len(dev_rings[e])} device)")
            return tot
        for n in applier.PLANE_NAMES:
            dv = planes[n].get()[e]
            hv = np.asarray(getattr(sim.gmap, n))
            if not np.array_equal(dv, hv):
                fail(f"{what} env {e}: plane {n} differs at "
                     f"{np.argwhere(dv != hv)[0].tolist()}")
                return tot
        hb = _books(sim.gmap)
        for b in BOOKS:
            if int(books[b][e]) != hb[b] - p["books"][b]:
                fail(f"{what} env {e}: book {b} device {int(books[b][e])} host "
                     f"{hb[b] - p['books'][b]}")
                return tot
        he, ht = edit_door.door_rows(sim.gmap, sim.edit_queue._edits[p["nq"]:],
                                     LS._door_sources())
        de, dt = dev_door[e]
        if not (np.array_equal(he, de) and np.array_equal(ht, dt)):
            fail(f"{what} env {e}: the door rows differ (edits {len(he)} / {len(de)}, "
                 f"tiles {len(ht)} / {len(dt)}); first edit diff "
                 f"{_first_diff(he, de)}; first tile diff {_first_diff(ht, dt)}")
            return tot
        hc = [tuple(x) for x in LS._CLAIM_LOG[id(tick_credit(sim.gmap))]]
        if hc != [tuple(x) for x in dev_claims[e]]:
            fail(f"{what} env {e}: the claims differ ({len(hc)} host / "
                 f"{len(dev_claims[e])} device)")
            return tot
        if [tuple(x) for x in unit_kernels.noise_of(sim)] != list(dev_noise[e]):
            fail(f"{what} env {e}: the noise list differs")
            return tot
        if [tuple(x) for x in unit_kernels.blast_log_of(sim)] != list(dev_blast[e]):
            fail(f"{what} env {e}: the blast log differs")
            return tot
        tot["records"] += len(recs[e])
        tot["PAYLOAD"] += sum(1 for r in recs[e] if r[0] == applier.KIND_PAYLOAD)
        tot["events"] += len(hev)
        tot["edits"] += len(he)
        tot["tiles"] += len(ht)
        tot["claims"] += len(hc)
        tot["noises"] += len(dev_noise[e]) - len(p["noise"])
        tot["blasts"] += len(dev_blast[e]) - len(p["blast"])
        tot["rows changed"] += int(np.any(got_rows[e] != p["rows"], axis=1).sum())
    return tot


def _col_name(c):
    for col in unit_store.layout().columns:
        if col.offset <= c < col.offset + col.width:
            return f"{col.name}[{c - col.offset}]"
    return str(c)


def _first_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        n = min(len(a), len(b))
        for i in range(n):
            if not np.array_equal(a[i], b[i]):
                return (i, a[i].tolist(), b[i].tolist())
        return ("len", a.shape, b.shape)
    d = np.argwhere(a != b)
    return None if not len(d) else (d[0].tolist(), a[tuple(d[0])], b[tuple(d[0])])


def _resolve_counting(sim, fx, census):
    """``fx.resolve()`` with the disc, blast and flash census taken around
    each payload call at its turn (the planes and units as they stand)."""
    from simulation import payloads, simultaneous
    c = census.c
    o_ep = payloads.execute_payload

    def ep(gmap, queue, units, fy, fx_, payload, rng, events=None, kind="explosion",
           owner_id=-1, approach=None):
        _flash_census(c, sim, fy, fx_, payload)
        _blast_census(c, sim, fy, fx_, payload)
        info = _disc_census(c, sim, fy, fx_, payload)
        if payload.gas_species:
            c["emit_gas"] += 1
        if payload.ignite_radius > 0:
            c["ignite_ring"] += 1
        if payload.heat_amount > 0:
            c["deposit_heat"] += 1
        if owner_id < 0 and (payload.gas_species or payload.ignite_radius > 0
                             or payload.heat_amount > 0):
            c["claims: the environment's (none)"] += 1
        r = o_ep(gmap, queue, units, fy, fx_, payload, rng, events=events, kind=kind,
                 owner_id=owner_id, approach=approach)
        _disc_after(c, sim, info)
        return r
    ep.__wrapped__ = o_ep
    pend = []
    for e in fx._pending:
        if e[0] == simultaneous._WORLD:
            pend.append((e[0], ep, e[2], e[3]))
        else:
            pend.append(e)
    fx._pending[:] = pend
    fx.resolve()


def _payload_call(sim, rng, names, k):
    from simulation.weapons import get_tables
    h, w = sim.gmap.material.shape
    name = names[k % len(names)]
    pl = get_tables().payloads.by_name[name]
    if rng.random() < 0.12:                                   # near / past the edge
        fy, fx = int(rng.integers(-3, h + 3)), int(rng.integers(-3, w + 3))
    else:
        fy, fx = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
    ids = [u.id for u in sim.units]
    owner = -1 if rng.random() < 0.25 else int(rng.choice(ids))
    kind = applier.PAYLOAD_KINDS[int(rng.integers(len(applier.PAYLOAD_KINDS)))]
    appr = None if rng.random() < 0.3 else (fy - 1, fx)
    args = (sim.gmap, sim.edit_queue, sim.units, fy, fx, pl, sim.draws)
    return ("payload", args, dict(events=sim.tick_events, kind=kind, owner_id=owner,
                                  approach=appr))


def _hit_item(sim, rng):
    from simulation.damage import DamagePacket
    live = [u for u in sim.units if u.alive]
    if not live:
        return None
    u = live[int(rng.integers(len(live)))]
    pk = DamagePacket(amount=int(rng.integers(1, 30)), dtype=0,
                      attacker_id=int(rng.choice([-1] + [v.id for v in sim.units])),
                      expose=int(rng.integers(1, 3)) * ONE // 2)
    return ("hit", u, pk, "bullet")


#: The law's branches the probe must take (a 0 fails as vacuous).
PROBE_BRANCHES = (
    "disc: a wall destroyed", "disc: a blast wall damaged, standing",
    "disc: a non-blast solid untouched",
    "disc: a stencil reads a tile destroyed earlier in the disc",
    "blast: a packet lands", "blast: a kill", "blast: outside the radius",
    "blast: below the threshold", "blast: at or above the threshold",
    "blast: a dead unit skipped",
    "flash: sees the flash", "flash: no line of sight", "flash: out of the bound",
    "flash: facing it (f = 1)", "flash: turned away (f < 1)", "flash: BLINDED applied",
    "flash: already blinded longer (kept)", "flash: a shorter BLINDED refreshed longer",
    "flash: at or past the radius (g = 0)", "flash: a dead unit skipped",
    "noise: a noise logged", "noise: radius 0 (silent)",
    "emit_gas", "ignite_ring", "deposit_heat", "claims: the environment's (none)",
    "edit: atmosphere DISC", "edit: smoke DISC", "edit: wave_source TILE",
    "edit: gas DISC", "edit: fire DISC", "edit: heat DISC",
)
PROBE_UNREACHABLE = {
    "flash: under one tick":
        "blind_ticks * f * g >> 32 is >= 1 wherever g > 0 for every shipped / variant "
        "row (blind_ticks >= 40, f >= away 0 only at half a turn); a 0-tick result "
        "needs g within 1/40 of the radius AND facing away: the probe rolls it rarely, "
        "counted when it happens, not required",
}


def part_probe(trials=PROBE_TRIALS):
    rng = np.random.default_rng(7101)
    LS_taps = LS.install_claims_tap()
    census = ProbeCensus()
    tot = Counter()
    t0 = time.perf_counter()
    try:
        for pas, changes in (("shipped", {}), ("variants", VARIANTS)):
            names = applied_rows()
            for t in range(trials):
                N = int(rng.integers(1, 4))
                hw = (int(rng.integers(20, 34)), int(rng.integers(24, 44)))
                sims = [_probe_sim(rng, 100 * t + e, hw) for e in range(N)]
                share_tables(sims)        # the rows set AFTER the sims are built
                with census, Rows(changes):
                    stage = applier.STAGE_RESOLVE if t % 3 == 2 else applier.STAGE_DETONATE
                    lists = []
                    for e, sim in enumerate(sims):
                        items = []
                        for k in range(int(rng.integers(1, 5))):
                            if stage == applier.STAGE_RESOLVE and rng.random() < 0.5:
                                hi = _hit_item(sim, rng)
                                if hi is not None:
                                    items.append(hi)
                            items.append(_payload_call(sim, rng, names, int(rng.integers(99))))
                        lists.append(items)
                    got = compare_walk(sims, lists, stage, census, what=f"probe {pas} {t}")
                    if FAILS:
                        return
                    tot.update(got)
                    tot[f"stage {applier.STAGE_NAMES[stage]}"] += 1
    finally:
        for cont, attr, o in LS_taps:
            setattr(cont, attr, o)
    c = census.c
    miss = [b for b in PROBE_BRANCHES if not c[b]]
    if miss:
        fail(f"probe: branches never taken {miss}")
    print(f"probe: {2 * trials} trials (shipped rows, then the variants) compared "
          f"{dict(tot)} ({time.perf_counter() - t0:.0f} s)")
    print("probe census: " + "; ".join(f"{k} = {v}" for k, v in sorted(c.items())))
    for k, why in PROBE_UNREACHABLE.items():
        print(f"  not required: {k} ({c[k]} seen): {why}")


# ---------------------------------------------------------------------------
# PART DISC: each DISC edit form's door rows against the host's same edit
# ---------------------------------------------------------------------------
def part_disc():
    """Every DISC edit the function writes (the atmosphere, the clear, the
    gas, the flame, the two heats) for many centres and radii, through the
    PAYLOAD function on one-step rows, is gated by PART PROBE; here the door's
    DISC form alone: a payload of only that edit (a mutated row) at centres
    in, on the edge of and off the grid, radii from 0.4 to 7.5 tiles."""
    rng = np.random.default_rng(3)
    LS_taps = LS.install_claims_tap()
    n = 0
    try:
        sims = [_probe_sim(rng, 5000 + e, (26, 36)) for e in range(2)]
        share_tables(sims)
        for radius in (0.4, 0.5, 1.0, 1.5, 2.5, 3.0, 4.75, 7.5):
            for row, attrs in (
                    ("studio_smoke", dict(gas_radius=radius, gas_amount=1.25)),
                    ("studio_ignite", dict(heat_radius=radius, heat_amount=123.4)),
                    ("incendiary_splash", dict(ignite_radius=radius,
                                               ignite_intensity=0.37))):
                with Rows({f"payloads.{row}": attrs}):
                    lists = []
                    for sim in sims:
                        h, w = sim.gmap.material.shape
                        items = []
                        for fy, fx in ((h // 2, w // 2), (0, 0), (-2, w + 1), (h - 1, 3)):
                            from simulation.weapons import get_tables
                            pl = get_tables().payloads.by_name[row]
                            items.append(("payload", (sim.gmap, sim.edit_queue, sim.units,
                                                      fy, fx, pl, sim.draws),
                                          dict(events=sim.tick_events, kind="explosion",
                                               owner_id=sim.units[0].id, approach=None)))
                        lists.append(items)
                    got = compare_walk(sims, lists, applier.STAGE_DETONATE,
                                       what=f"disc {row} r {radius}")
                    if FAILS:
                        return
                    n += got["edits"]
    finally:
        for cont, attr, o in LS_taps:
            setattr(cont, attr, o)
    print(f"disc: {n} DISC edits (gas LINEAR, heat FLAT / LINEAR, fire MAX with its "
          f"clamp) at 8 radii and 4 centres (in, corner, off the grid, the edge) equal "
          f"door_rows of the host's edits")


# ---------------------------------------------------------------------------
# The payload scene
# ---------------------------------------------------------------------------
SH, SW = 44, 100
#: The row sets: which payload row each delivering ammo row carries (the
#: shipped rows through every call site: grenades by kind, payload rounds by
#: the three firing weapons, charges by any ammo row's payload).
ROW_SETS = {
    "A": {"grenade_frag": "studio_smoke", "40mm_frag": "hazmat_release"},
    "B": {"grenade_frag": "plasma_splash_small", "grenade_smoke": "plasma_splash_large",
          "grenade_tear": "studio_ignite", "grenade_poison": "hazmat_release",
          "grenade_flash": "studio_smoke", "grenade_incendiary": "smoke_screen",
          "plasma_standard": "smoke_screen", "plasma_heavy": "tear_burst",
          "40mm_frag": "flashbang"},
    "C": {"grenade_frag": "flashbang", "plasma_standard": "poison_cloud",
          "plasma_heavy": "incendiary_splash", "40mm_frag": "studio_ignite"},
    "D": {"grenade_frag": "poison_cloud", "plasma_standard": "studio_smoke",
          "plasma_heavy": "smoke_screen", "40mm_frag": "tear_burst"},
    "V": {"grenade_frag": "incendiary_splash", "40mm_frag": "studio_ignite",
          "demo_c4": "studio_smoke", "demo_breach": "plasma_splash_small"},
}
#: The noise-only row of set V (radius 0, nothing but its loudness).
NOISE_ONLY = {"payloads.studio_ignite": dict(heat_amount=0.0, heat_radius=0.0)}


def row_set(name):
    ch = {f"ammo.{a}": {"payload": p} for a, p in ROW_SETS[name].items()}
    if name == "V":
        ch.update(VARIANTS)
        ch.update(NOISE_ONLY)
    return Rows(ch)


def _scene_level(variant, tile_m=1.0):
    from level_loader import LevelData
    from simulation.materials import (MAT_AIR, MAT_DOOR_CLOSED, MAT_GLASS, MAT_GRILLE,
                                      MAT_HULL, MAT_STEEL, MAT_WOOD)
    h, w = SH + 2 * (variant % 2), SW + 4 * (variant % 3)
    tm = np.full((h, w), SPACE, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_HULL
    tm[2:h - 2, 2:w - 2] = MAT_AIR
    dy = variant % 3
    tm[25 + dy, 3:54] = MAT_WOOD                 # the wall the grenades' discs reach
    tm[25 + dy, 20:24] = MAT_GRILLE
    tm[25 + dy, 30:33] = MAT_DOOR_CLOSED
    tm[31, 3:54:7] = MAT_GLASS                   # non-blast solids in the discs
    tm[31, 6:54:9] = MAT_STEEL
    tm[16:19, 76] = MAT_WOOD                     # beside the corpse-skip group
    return LevelData(name=f"p7c1_scene{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=tile_m, diffuse_path=Path("."),
                     boundary="space")


class PayloadScene:
    """The payload scene, scripted (the module docstring): throwers of every
    hand-grenade kind (Z1), plasma / 40 mm shooters at an adjacent target
    (Z2: the round stops in its firing tick, slot 4c) and at a far one (Z3:
    it lands in slot 2''s landing), charges laid by fixture of every row
    (Z4: slot 9e(d')), weak units in the heat and gas discs (their deaths
    credited through the claims), units facing every way around the flashes
    and behind the wall, and (set V) the corpse-skip group: a carbine and a
    plasma caster on one victim in one resolve. ``variant`` > 0: another
    seed, layout, bystanders, the schedule shifted."""

    def __init__(self, variant=0, tile_m=1.0, row_set_name="A"):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.v, self.k, self.set = variant, 0, row_set_name
        sim = Simulation(_scene_level(variant, tile_m), seed=29071 + 13 * variant,
                         breach_physics=bp, enable_recorder=False,
                         ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        sim._next_unit_id += 4 + variant             # ids never the row index
        self.who = {}
        self.fire_at = {}
        self.n_kinds = len(sim.weapons_tables.hand_grenade_rows)
        for i in range(variant % 3):
            self._add(f"by{i}", 40 + 3 * i, 12, facing=SOUTH, hp=1.0e6)
        self._setup()

    def _add(self, name, x, y, team=0, facing=EAST, weapon="k5_carbine", hp=1.0e6):
        from simulation.turning import place
        from simulation.unit import Unit
        dy = self.v % 2
        u = Unit(name, x=float(x), y=float(y + (dy if y > 20 else 0)), team=team)
        self.sim.add_unit(u)
        place(u, facing)
        u.current_hp = float(hp)
        u.weapon_id = weapon
        self.who[name] = u
        return u

    def _setup(self):
        a = self._add
        dy = self.v % 3
        # Z1: the throwers (kind i at x = 8 + 8 i), victims at the landings
        for i in range(self.n_kinds):
            t = a(f"T{i}", 8 + 8 * i, 40, facing=NORTH, weapon="hand_grenade")
            t.grenade_counts = [6] * self.n_kinds
            a(f"V{i}", 8 + 8 * i, 28 + dy, team=1, facing=(EAST, WEST, NORTH, SOUTH)[i % 4],
              hp=(55.0 if i % 2 else 400.0))
            a(f"W{i}", 9 + 8 * i, 29 + dy, team=1, facing=(SOUTH, NORTH)[i % 2], hp=3.0)
            a(f"B{i}", 8 + 8 * i, 23 + dy, team=1, facing=SOUTH, hp=400.0)   # behind the wall
        # Z2: a round stopping in its firing tick (slot 4c)
        for j, wpn in enumerate(("sunspot", "helios", "gl6_revolver")):
            a(f"S{j}", 62, 4 + 4 * j, facing=EAST, weapon=wpn)
            a(f"ST{j}", 64, 4 + 4 * j, team=1, facing=WEST, hp=1.0e6)
            a(f"SW{j}", 65, 5 + 4 * j, team=1, facing=NORTH, hp=2.0)        # heat-credit victim
        # Z3: a round landing in slot 2''s landing (25 tiles at 1.5 t/t)
        for j, wpn in enumerate(("sunspot", "helios", "gl6_revolver")):
            a(f"F{j}", 58, 30 + 4 * j, facing=EAST, weapon=wpn)
            a(f"FT{j}", 83, 30 + 4 * j, team=1, facing=WEST, hp=1.0e6)
            a(f"FW{j}", 84, 31 + 4 * j, team=1, facing=SOUTH, hp=2.0)
        # Z5: the corpse skip -- K's carbine (a raise pending) and P's plasma
        # (its blast kills the victim at once) in ONE resolve; K's id first
        a("K", 70, 17, facing=EAST)
        a("KV", 72, 17, team=1, facing=WEST, hp=95.0)
        a("P", 74, 17, facing=WEST, weapon="sunspot")

    def _throw(self, u, kind, fuse, reach_t):
        self.sim.throw_grenade_intent(u.id, 0, -ONE, fuse, reach_q=reach_t * ONE, kind=kind)

    def _lay(self, ammo, x, y, det, owner):
        from simulation import charges
        charges.place(self.sim, charges.PlantedCharge(
            x, y, ammo, owner, det_tick=det, approach=(y + 1, x) if x % 2 else None))

    def charge_ammo(self):
        """An ammo row per applied payload under the set (charges carry any
        ammo row's payload)."""
        from simulation.weapons import get_tables
        t = get_tables()
        seen, out = set(), []
        for n, am in t.ammo.by_name.items():
            if am.payload and applier.payload_refusal(t.payloads.by_name[am.payload]) == 0 \
                    and am.payload not in seen:
                seen.add(am.payload)
                out.append(n)
        return out

    def tick(self):
        sim, k, w, s = self.sim, self.k, self.who, 2 + self.v % 4
        if k == s:                                     # every kind; two pairs on one tick
            for i in range(self.n_kinds):
                self._throw(w[f"T{i}"], i, 18, 12)
        if k == s + 1:                                 # a second wave of the same kinds
            for i in range(0, self.n_kinds, 2):        # lands ONE tick after: pairs differ
                self._throw(w[f"T{i}"], i, 17, 12)
        if k == s + 30:                                # the other kinds at the same spots
            for i in range(self.n_kinds):
                self._throw(w[f"T{i}"], (i + 3) % self.n_kinds, 14, 12)
        if k == s + 2 and self.set == "V":             # two on ONE tick at one spot
            self._throw(w["T0"], 0, 19, 12)
            self._throw(w["T1"], 0, 12, 12)            # (T1 stands 8 east: reach 12 + 8?)
        for name in ("S0", "S1", "S2", "F0", "F1", "F2", "K", "P"):
            sim.set_trigger(w[name].id, k in (s, s + 1, s + 40) and w[name].alive)
        if k == s + 5:                                 # the charges: every applied row
            for j, am in enumerate(self.charge_ammo()):
                x = 60 + 3 * (j % 12)
                y = 38 + (j // 12) * 2
                self._lay(am, x, y, sim.tick + 3 + (j % 3), w[f"T{j % self.n_kinds}"].id
                          if j % 4 else -1)
            self._add_charge_victims = True
        sim.step()
        sim.set_paused(False)
        self.k += 1

    def is_terminal(self):
        return False


class ShipDriver:
    """ship_demo's bots match (MatchSession)."""

    def __init__(self, seed=1):
        from simulation.match_session import MatchSession
        level = level_loader.load("ship_demo", str(_ROOT / "levels"))
        self.session = MatchSession(level, seed=seed, breach_physics=bp)
        self.sim = self.session.sim

    def tick(self):
        self.session.tick()

    def is_terminal(self):
        return self.session.is_terminal()


# ---------------------------------------------------------------------------
# The scene census: the call sites and rows, the order, the corpse skip
# ---------------------------------------------------------------------------
class SceneCensus:
    """Wraps (reads only) execute_payload at its three call sites, the
    rounds' landing and ExposureLanding.land, and counts: (call site, row)
    pairs; two detonations on one tick where the second's disc or blast or
    flash holds a tile / unit the first changed; blast kills; flash blinds and
    spares; a raise skipped for a unit a BLAST killed in the same resolve;
    kills credited to a payload's owner through the claims."""

    def __init__(self):
        self.c = Counter()
        self.pairs = set()
        self._saved = []
        self.in_landing = False
        self.tick_calls = {}
        self.owners = set()

    def __enter__(self):
        import importlib
        from simulation import exposure
        from simulation.simulation import Simulation
        census = self
        for modname, site in (("simulation.simulation", "grenade"),
                              ("simulation.charges", "charge"),
                              ("simulation.combat", "shell")):
            mod = importlib.import_module(modname)
            o = mod.execute_payload

            def ep(gmap, queue, units, fy, fx, payload, rng, __o=o, __site=site, **kw):
                return census._call(__o, __site, gmap, queue, units, fy, fx, payload, rng, kw)
            ep.__wrapped__ = o
            mod.execute_payload = ep
            self._saved.append((mod, "execute_payload", o))
        o_ur = Simulation._update_rounds

        def ur(sim, __o=o_ur):
            census.in_landing = True
            try:
                return __o(sim)
            finally:
                census.in_landing = False
        Simulation._update_rounds = ur
        self._saved.append((Simulation, "_update_rounds", o_ur))
        o_land = exposure.ExposureLanding.land

        def land(lg, __o=o_land):
            dead = {id(u) for u in lg._dead}
            for u, _dt, _add in lg._raises:
                if not getattr(u, "alive", True) and id(u) not in dead:
                    census.c["landing: a raise skipped for a unit a BLAST killed"] += 1
            return __o(lg)
        exposure.ExposureLanding.land = land
        self._saved.append((exposure.ExposureLanding, "land", o_land))
        return self

    def __exit__(self, *exc):
        for cont, attr, o in reversed(self._saved):
            setattr(cont, attr, o)
        self._saved.clear()
        return False

    def _call(self, orig, site, gmap, queue, units, fy, fx, payload, rng, kw):
        c = self.c
        if site == "shell":
            site = "shell (2' landing)" if self.in_landing else "shell (4c)"
        self.pairs.add((site, payload.name))
        mat0 = gmap.material.copy()
        alive0 = {id(u): u.alive for u in units}
        bl0 = {id(u): sum(1 for s in u.statuses if s.kind == 8) for u in units}
        owner = int(kw.get("owner_id", -1))
        if owner >= 0:
            self.owners.add(owner)
        prev = self.tick_calls.setdefault(id(gmap), [])
        changed_tiles = set()
        killed = set()
        for (pt, pk) in prev:
            changed_tiles |= pt
            killed |= pk
        # the second detonation of a tick reading what an earlier one changed
        r = max(int(payload.radius), int(getattr(payload, "blind_radius_q", 0) >> 16) + 1)
        if prev and any(abs(y - fy) <= r + 1 and abs(x - fx) <= r + 1
                        for y, x in changed_tiles):
            c["order: a later detonation's disc holds a tile an earlier one destroyed"] += 1
        if prev and any(id(u) in killed and abs(u.center_tile_x() - fx) <= r
                        and abs(u.center_tile_y() - fy) <= r for u in units):
            c["order: a later detonation meets a unit an earlier one killed"] += 1
        res = orig(gmap, queue, units, fy, fx, payload, rng, **kw)
        gone = {(int(y), int(x)) for y, x in np.argwhere((mat0 != 0) & (gmap.material == 0))}
        dead = {id(u) for u in units if alive0.get(id(u)) and not u.alive}
        prev.append((gone, dead))
        if gone:
            c["disc: walls destroyed by a blast"] += 1
        if dead:
            c["blast: a unit killed"] += 1
        if payload.blind_radius_q > 0:
            n_bl = sum(1 for u in units
                       if sum(1 for s in u.statuses if s.kind == 8) > bl0[id(u)]
                       or any(s.kind == 8 for s in u.statuses))
            c["flash: some units blinded"] += int(n_bl > 0)
            c["flash: some units spared"] += int(n_bl < sum(1 for u in units if u.alive))
        if payload.radius <= 0 and not payload.gas_species and payload.ignite_radius <= 0 \
                and payload.heat_amount <= 0 and payload.blind_radius_q <= 0 \
                and payload.unit_damage <= 0:
            c["a radius-0 payload: noise only"] += 1
        if payload.radius > 0:
            c["blast log: a disc logged"] += 1
        return res

    def after_tick(self, sim):
        from simulation.events import UnitKilledEvent
        self.tick_calls.pop(id(sim.gmap), None)
        for ev in sim.tick_events:
            if isinstance(ev, UnitKilledEvent) and ev.killed_by in ("heat", "poison_gas",
                                                                     "poisoned", "burning") \
                    and int(ev.attacker_id) in self.owners:
                self.c[f"claims: a {ev.killed_by} kill credited to a payload's owner"] += 1


SCENE_BRANCHES = (
    "order: a later detonation's disc holds a tile an earlier one destroyed",
    "order: a later detonation meets a unit an earlier one killed",
    "disc: walls destroyed by a blast", "blast: a unit killed",
    "flash: some units blinded", "flash: some units spared",
    "a radius-0 payload: noise only", "blast log: a disc logged",
    "landing: a raise skipped for a unit a BLAST killed",
)
SITES = ("grenade", "charge", "shell (4c)", "shell (2' landing)")


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def after_tick(self, sim):
        pass


def run_lockstep(name, drivers, slots, ticks, census=None):
    """Shadow ``slots`` on ``drivers`` for ``ticks`` ticks: every tick each slot
    equal, its coverage == the rows the host changed. Returns the counts."""
    t0 = time.perf_counter()
    tot = Counter()
    n = 0
    ok = True
    ctx = census if census is not None else _Null()
    taps = LS.install_payload_tap() + LS.install_claims_tap()
    try:
        with ctx, ShadowLockstep(drivers, list(slots)) as ls:
            for _ in range(ticks):
                if any(d.is_terminal() for d in drivers):
                    break
                reps = ls.tick()
                for d in drivers:
                    ctx.after_tick(d.sim)
                n += 1
                for lab, rep in reps.items():
                    if not rep.equal:
                        fail(f"{name}: {ls.divergences[-1]}")
                        ok = False
                        break
                    if not np.array_equal(rep.coverage, rep.host_changed):
                        fail(f"{name} [{lab}] tick {rep.tick}: coverage "
                             f"{rep.coverage.tolist()} != host-changed "
                             f"{rep.host_changed.tolist()}")
                        ok = False
                        break
                    if rep.lists is None:
                        continue
                    for e in range(len(drivers)):
                        pl = rep.lists.get("payloads", rep.lists.get("effects", [[]] * 99))[e]
                        np_ = sum(1 for r in pl if r[0] == applier.KIND_PAYLOAD)
                        tot[f"PAYLOAD [{lab}]"] += np_
                        tot["door rows"] += len(rep.lists["edits"][e])
                        tot["claims"] += len(rep.lists["claims"][e])
                        tot["blast log"] += len(rep.lists["blast_log"][e])
                    tot[f"events [{lab}]"] += int(rep.events.sum())
                if not ok:
                    break
    finally:
        for cont, attr, o in taps:
            setattr(cont, attr, o)
    print(f"{name}: N={len(drivers)} units per env {[len(d.sim.units) for d in drivers]}, "
          f"{n} ticks{', env i == env i alone' if len(drivers) > 1 and ok else ''}; "
          f"compared {dict(tot)} ({time.perf_counter() - t0:.0f} s)")
    return tot


def part_scene():
    census = SceneCensus()
    allt = Counter()
    for set_name in ROW_SETS:
        d = PayloadScene(0, row_set_name=set_name)
        share_tables([d.sim])                      # the rows set AFTER the build
        with row_set(set_name):
            allt.update(run_lockstep(f"scene set {set_name}", [d], SCENE_SLOTS,
                                     SCENE_TICKS, census))
        if FAILS:
            return
    c = census.c
    miss = [b for b in SCENE_BRANCHES if not c[b]]
    if not any(k.startswith("claims: ") for k in c):
        miss.append("claims: a kill credited to a payload's owner")
    want = {(s, r) for s in SITES for r in applied_rows()}
    gaps = sorted(want - census.pairs)
    if gaps:
        fail(f"scene: (call site, row) pairs never detonated {gaps}")
    if miss:
        fail(f"scene: branches never taken {miss}")
    print(f"scene: {len(census.pairs & want)} of {len(want)} (call site, shipped row) "
          f"pairs detonated; census: " + "; ".join(f"{k} = {v}" for k, v in sorted(c.items())))


def part_batch():
    for n in BATCH_NS:
        ds = [PayloadScene(v, row_set_name="V") for v in range(n)]
        share_tables([d.sim for d in ds])
        with row_set("V"):
            # the fields differ too: a warm room in the odd envs
            for i, d in enumerate(ds):
                if i % 2:
                    d.sim.gmap.wall_hp[d.sim.gmap.wall_hp > 0] //= 2
            run_lockstep(f"batch N={n}", ds, SCENE_SLOTS, BATCH_TICKS)
        if FAILS:
            return


# ---------------------------------------------------------------------------
# PADDING / OVERFLOW / REFUSE (synthetic lists through compare_walk's pieces)
# ---------------------------------------------------------------------------
def _device_walk(sims, recs, stage, caps=None, garbage=False):
    """The device side of a walk alone: returns (store, planes, books, the
    lists) or raises what the walk raises."""
    caps = caps or {}
    N = len(sims)
    U = max(len(s.units) for s in sims) + (2 if garbage else 0)
    st = unit_store.DeviceStore(N, U, tables=unit_tables.values_batch(sims))
    rows = np.stack([unit_store.pack(s.units, U) for s in sims])
    if garbage:                          # padding rows full of live-looking junk
        lay = unit_store.layout()
        for e, s in enumerate(sims):
            rows[e, len(s.units):, :] = 7
            rows[e, len(s.units):, lay.column("valid").offset] = 0
    st.upload(rows)
    ring = unit_events.EventRing(N, U, caps.get("ring", unit_events.E_MAX))
    ring.begin()
    dl = applier.DeviceApplier(N, applier.E_CAP, n_units=U)
    dl.stage_lists(recs)
    door = edit_door.DeviceEditDoor(N, edit_cap=caps.get("edits", edit_door.EDIT_CAP),
                                    tile_cap=caps.get("tiles", edit_door.TILE_CAP))
    door.begin()
    claims = attack_lists.claims_list(N, cap=caps.get("claims", attack_lists.CLAIM_CAP))
    claims.begin()
    noise = ot.DeviceTable(unit_schema.NOISE_TABLE, N, cap=caps.get("noise"))
    noise.begin()
    blast = ot.DeviceTable(unit_schema.BLAST_LOG_TABLE, N, cap=caps.get("blast"))
    blast.begin()
    planes = _dev_planes([_planes(s.gmap) for s in sims])
    wl = unit_kernels.WalkLists(door=None if caps.get("no_door") else door, claims=claims,
                                noise=noise, blast_log=blast)
    out = dict(store=st, planes=planes, ring=ring, door=door, claims=claims, noise=noise,
               blast=blast, dl=dl)
    _cov, books = unit_kernels.walk_effects(st, dl, planes, stage, ring, wl)
    out["books"] = books
    return out


def _recs_for(sim, names, rng, k=3):
    from simulation.weapons import get_tables
    h, w = sim.gmap.material.shape
    out = []
    for j in range(k):
        pl = get_tables().payloads.by_name[names[j % len(names)]]
        out.append(_record(sim, ((sim.gmap, sim.edit_queue, sim.units, h // 2, w // 2 + j,
                                  pl, sim.draws),
                                 dict(events=sim.tick_events, kind="grenade",
                                      owner_id=sim.units[0].id, approach=None))))
    return np.array(out, dtype=np.int64).reshape(-1, applier.F)


def part_padding():
    rng = np.random.default_rng(55)
    sims = [_probe_sim(rng, 7000 + e, (26, 36)) for e in range(3)]
    share_tables(sims)
    with Rows(VARIANTS):
        names = ["plasma_splash_small", "plasma_splash_large", "studio_smoke"]
        recs = [_recs_for(s, names, rng, 2 + e) for e, s in enumerate(sims)]
        clean = _device_walk(sims, recs, applier.STAGE_DETONATE)
        dirty = _device_walk(sims, recs, applier.STAGE_DETONATE, garbage=True)
    U = max(len(s.units) for s in sims)
    a, b = clean["store"].download(), dirty["store"].download()
    lay = unit_store.layout()
    for e, s in enumerate(sims):
        n = len(s.units)
        if not np.array_equal(a[e, :n], b[e, :n]):
            fail(f"padding env {e}: a valid row differs beside garbage padding rows")
        if not (b[e, n:, :lay.column("valid").offset] == 7).all():
            fail(f"padding env {e}: a padding row was written")
    for nm in ("ring", "door", "claims", "noise", "blast"):
        x, y = clean[nm], dirty[nm]
        xr = x.records() if nm != "door" else [(r[0].tolist(), r[1].tolist()) for r in x.rows()]
        yr = y.records() if nm != "door" else [(r[0].tolist(), r[1].tolist()) for r in y.rows()]
        if nm == "ring":
            xr, yr = x.records(), y.records()
        if xr != yr:
            fail(f"padding: the {nm} differs beside padding rows")
    for n_ in applier.PLANE_NAMES:
        if not np.array_equal(clean["planes"][n_].get(), dirty["planes"][n_].get()):
            fail(f"padding: plane {n_} differs")
    print(f"padding: 3 envs (U {U} + 2 garbage rows each): rows, ring, door, claims, "
          f"noise, blast log and planes equal a clean run; the padding untouched")


def part_overflow():
    rng = np.random.default_rng(66)
    n_ok = 0
    sims = [_probe_sim(rng, 8000)]
    share_tables(sims)
    with Rows(VARIANTS):
        names = ["plasma_splash_large", "studio_smoke", "incendiary_splash"]
        recs = [_recs_for(sims[0], names, rng, 3)]
        full = _device_walk(sims, recs, applier.STAGE_DETONATE)
        sizes = dict(edits=len(full["door"].rows()[0][0]), tiles=len(full["door"].rows()[0][1]),
                     claims=len(full["claims"].records()[0]),
                     noise=len(full["noise"].records()[0]),
                     blast=len(full["blast"].records()[0]),
                     ring=len(full["ring"].records()[0]))
        if min(sizes.values()) < 1:
            fail(f"overflow: a list the scene never writes {sizes}")
            return
        for nm, exc in (("edits", edit_door.DoorOverflow), ("tiles", edit_door.DoorOverflow),
                        ("claims", attack_lists.ListOverflow),
                        ("noise", ot.ObjectTableOverflow),
                        ("blast", ot.ObjectTableOverflow),
                        ("ring", unit_kernels.UnitEventOverflow)):
            try:
                _device_walk(sims, recs, applier.STAGE_DETONATE, caps={nm: sizes[nm] - 1})
                fail(f"overflow: the {nm} one short was not refused")
            except exc as ex:
                n_ok += 1
                if "env 0" not in str(ex) and "[0]" not in str(ex):
                    fail(f"overflow: the {nm} refusal does not name the env: {ex}")
    print(f"overflow: {n_ok} lists one short of the walk's need {sizes}: each refused "
          f"by name (DoorOverflow / ListOverflow / ObjectTableOverflow / UnitEventOverflow)")


def part_refuse():
    rng = np.random.default_rng(77)
    sims = [_probe_sim(rng, 9000)]
    share_tables(sims)
    n = 0
    for name, code in sorted(refused_rows().items()):
        recs = [_recs_for(sims[0], [name], rng, 1)]
        before = _planes(sims[0].gmap)
        try:
            out = _device_walk(sims, recs, applier.STAGE_DETONATE)
            fail(f"refuse: {name} was applied")
            continue
        except applier.EffectRefused as ex:
            reason = applier.PAYLOAD_REFUSALS[code - 1][1]
            if name not in str(ex) or reason not in str(ex):
                fail(f"refuse: the refusal of {name} names neither it nor {reason}: {ex}")
            n += 1
        _ = before
    # nothing written by a refused row: the walk with refused_ok
    recs = [_recs_for(sims[0], ["frag_standard", "o2_store_rupture", "frag_heat"], rng, 3)]
    N, U = 1, len(sims[0].units)
    st = unit_store.DeviceStore(N, U, tables=unit_tables.values_batch(sims))
    rows = unit_store.pack(sims[0].units, U)[None]
    st.upload(rows)
    ring = unit_events.EventRing(N, U)
    ring.begin()
    dl = applier.DeviceApplier(N, n_units=U)
    dl.stage_lists(recs)
    door = edit_door.DeviceEditDoor(N)
    door.begin()
    claims = attack_lists.claims_list(N)
    claims.begin()
    noise = ot.DeviceTable(unit_schema.NOISE_TABLE, N)
    noise.begin()
    blast = ot.DeviceTable(unit_schema.BLAST_LOG_TABLE, N)
    blast.begin()
    planes = _dev_planes([_planes(sims[0].gmap)])
    p0 = {k: v.get().copy() for k, v in planes.items()}
    _c, books = unit_kernels.walk_effects(st, dl, planes, applier.STAGE_DETONATE, ring,
                                          unit_kernels.WalkLists(door=door, claims=claims,
                                                                 noise=noise, blast_log=blast),
                                          refused_ok=True)
    wrote = [k for k, v in planes.items() if not np.array_equal(v.get(), p0[k])]
    if wrote or not np.array_equal(st.download(), rows) or door.rows()[0][0].size \
            or claims.records()[0] or noise.records()[0] or blast.records()[0] \
            or ring.records()[0] or int(books["refused"][0]) != 3:
        fail(f"refuse: a refused row wrote something (planes {wrote}, books {books})")
    # a PAYLOAD without its lists faults, nothing written
    with Rows(VARIANTS):
        recs = [_recs_for(sims[0], ["plasma_splash_small"], rng, 1)]
        try:
            _device_walk(sims, recs, applier.STAGE_DETONATE, caps={"no_door": True})
            fail("refuse: a PAYLOAD without the edit door was applied")
        except applier.ApplierFault:
            n += 1
    print(f"refuse: {n - 1} refused rows each refused by name with its reason "
          f"({sorted(set(refused_rows().values()))} codes); three refused records wrote "
          f"nothing (refused = 3); a PAYLOAD without its door faulted")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
def _host_side(ch, books_only=False):
    def hs(sim, pre, post):
        g = sim.gmap
        b = ch.books
        if not b:
            return
        g.n_destruction_seed_sum += b["n_destruction_seed"]
        for name, key in (("ge_retire", "retire"), ("ge_mint", "mint"),
                          ("ge_destroy_seed", "destroy_seed")):
            g._gas_energy_book(key, b[name])
        g._solid_energy_book("destroy_wall", b["solid_destroy_wall"])
        g._tile_serial += b["tiles_changed"]
        mat0 = ch.inputs._host_fields["material"][0]
        changed = np.argwhere(mat0 != g.material)
        for y, x in changed:
            g.render_dirty.add((int(y), int(x)))
        if len(changed):
            g._sky_mask_dirty = True
    return hs


def _det_slot(base, outputs_payloads="list:detonations"):
    """A chained form of a detonation slot: its PAYLOAD list is the
    ``detonations`` list (consumed, never replayed)."""
    outs = tuple(outputs_payloads if o == "list:payloads" else o for o in base.outputs)

    def run(st, inp, ring, lists, __b=base):
        lists = dict(lists)
        lists["payloads"] = lists["detonations"]
        return __b.run(st, inp, ring, lists)
    return DeviceSlot(label=base.label, run=run, events=True, inputs=base.inputs,
                      outputs=outs, host_only=base.host_only, skip_ok=base.skip_ok)


def _count_payload_calls(twin):
    import importlib
    calls = Counter()
    saved = []
    for modname in ("simulation.simulation", "simulation.charges", "simulation.combat"):
        mod = importlib.import_module(modname)
        o = mod.execute_payload

        def ep(gmap, *a, __o=o, __m=modname, **k):
            if gmap is twin.sim.gmap:
                calls[__m] += 1
            return __o(gmap, *a, **k)
        ep.__wrapped__ = o
        mod.execute_payload = ep
        saved.append((mod, "execute_payload", o))
    return calls, saved


def chained_run(name, slots, set_name, ticks=CHAIN_TICKS, driver=PayloadScene,
                replay_refused=False):
    twin, ref = driver(), driver()
    share_tables([twin.sim, ref.sim])
    with row_set(set_name) if set_name else Rows({}):
        calls, saved = _count_payload_calls(twin)
        LS.REPLAY_REFUSED["on"] = replay_refused
        holder = {}
        n = 0
        t0 = time.perf_counter()
        taps = LS.install_payload_tap() + LS.install_claims_tap()
        try:
            ch = ChainedSlots(twin, slots, host_side=lambda s, p, q: holder["hs"](s, p, q))
            holder["hs"] = _host_side(ch)
            with ch:
                for _ in range(ticks):
                    if twin.is_terminal() or ref.is_terminal():
                        break
                    twin.tick()
                    ref.tick()
                    n += 1
                    if _unit_hash(twin.sim) != _unit_hash(ref.sim):
                        fail(f"chained {name}: the unit hash left the reference at tick "
                             f"{twin.sim.tick}")
                        break
                    if _field_digest(twin.sim) != _field_digest(ref.sim):
                        fail(f"chained {name}: the field digest left the reference at "
                             f"tick {twin.sim.tick}")
                        break
                    if _queue_state(twin.sim) != _queue_state(ref.sim):
                        fail(f"chained {name}: the edit queue left the reference at tick "
                             f"{twin.sim.tick}")
                        break
                    if list(twin.sim.gmap.noise_log) != list(ref.sim.gmap.noise_log) or \
                            list(twin.sim.gmap.blast_log) != list(ref.sim.gmap.blast_log):
                        fail(f"chained {name}: the noise / blast log left the reference "
                             f"at tick {twin.sim.tick}")
                        break
        finally:
            for cont, attr, o in taps + saved:
                setattr(cont, attr, o)
            LS.REPLAY_REFUSED["on"] = False
    recs = Counter()
    for x in ch.lists:
        for nm in ("effects", "detonations"):
            for r in x.get(nm, ()):
                recs[applier.KINDS[r[0] - 1]] += 1
    print(f"chained {name}: {n} ticks, the twin equal to the reference every tick ({ch.runs} "
          f"chain runs; records walked {dict(recs)}); host execute_payload calls in the "
          f"twin {dict(calls) or 0} ({time.perf_counter() - t0:.0f} s)")
    return n, ch, recs, calls


def part_chained():
    from cuda_unit_shooting_check import SLOT_2R, SLOT_4, SLOT_4A
    from cuda_unit_sprays_check import SLOT_4B
    for name, slots, set_name in (
            ("2 grenades", [_det_slot(SLOT_2G)], "V"),
            ("2' + landing", [SLOT_2R, SLOT_2L], "V"),
            ("4 + 4a + 4b + 4c", [SLOT_4, SLOT_4A, SLOT_4B, SLOT_4C], "V"),
            ("9e(d')", [_det_slot(SLOT_9E)], "V")):
        n, ch, recs, calls = chained_run(name, slots, set_name)
        if FAILS:
            return
        if calls:
            fail(f"chained {name}: the host execute_payload ran in the twin {dict(calls)}")
        if not recs.get("PAYLOAD"):
            fail(f"chained {name}: vacuous (no PAYLOAD walked)")


def part_ship():
    """ship_demo's bots: slots 2 and 4c shadowed for SHIP_TICKS (a refused row
    -- the frag -- would stop the shadow by name: the census says whether it
    came), then the same match chained with the refused rows replayed."""
    d = ShipDriver()
    share_tables([d.sim])
    census = SceneCensus()
    try:
        run_lockstep("ship (2 + 4c)", [d], [SLOT_2G, SLOT_4C], SHIP_TICKS, census)
    except applier.EffectRefused as ex:
        print(f"ship: the shadow met a refused row at tick {d.sim.tick}: {ex}")
        FAILS[:] = [f for f in FAILS if "EffectRefused" not in f]
    print(f"ship: detonations seen {sorted(census.pairs)}")
    n, ch, recs, calls = chained_run("ship 2 grenades (refused rows replayed)",
                                     [_det_slot(SLOT_2G)], None, SHIP_TICKS,
                                     driver=ShipDriver, replay_refused=True)
    refused_calls = {k: v for k, v in calls.items()}
    print(f"ship: chained 500 ticks; the host execute_payload ran {refused_calls or 0} "
          f"time(s) (each a refused row's replay)")


# ---------------------------------------------------------------------------
# Main, bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "probe", "disc", "scene", "batch", "padding", "overflow", "refuse",
         "chained", "ship")


def main(parts=None) -> int:
    t0 = time.perf_counter()
    want = parts or os.environ.get("UNIT_PAYLOADS_PARTS", ",".join(PARTS)).split(",")
    for p in want:
        globals()[f"part_{p}"]()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print(f"UNIT_PAYLOADS_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def bench(reps: int = 50) -> int:
    """The TIMING table: median ms of ONE detonation's walk (the DETONATE
    stage over one PAYLOAD record per env) for each applied row at N = 1 / 8
    / 64, a small and a large radius (the row's own and its disc raised)."""
    import cupy as cp
    rng = np.random.default_rng(1)
    base = _probe_sim(rng, 1)
    lines = []
    for row in applied_rows():
        for size, attrs in (("own", {}), ("r 6", dict(radius=6, pressure=1.0,
                                                       wall_damage=10, gas_radius=6,
                                                       heat_radius=6.0, ignite_radius=6.0))):
            with Rows({f"payloads.{row}": attrs} if attrs else {}):
                for n in (1, 8, 64):
                    sims = [base] * n
                    share_tables([base])
                    recs = [_recs_for(base, [row], rng, 1)] * n
                    ts = []
                    U = len(base.units)
                    st = unit_store.DeviceStore(n, U, tables=unit_tables.values_batch(sims))
                    rows = np.stack([unit_store.pack(base.units, U)] * n)
                    ring = unit_events.EventRing(n, U)
                    dl = applier.DeviceApplier(n, n_units=U)
                    door = edit_door.DeviceEditDoor(n)
                    claims = attack_lists.claims_list(n)
                    noise = ot.DeviceTable(unit_schema.NOISE_TABLE, n)
                    blast = ot.DeviceTable(unit_schema.BLAST_LOG_TABLE, n)
                    planes = _dev_planes([_planes(base.gmap)] * n)
                    wl = unit_kernels.WalkLists(door=door, claims=claims, noise=noise,
                                                blast_log=blast)
                    for _ in range(reps):
                        st.upload(rows)
                        ring.begin()
                        dl.stage_lists(recs)
                        door.begin()
                        claims.begin()
                        noise.begin()
                        blast.begin()
                        cp.cuda.Stream.null.synchronize()
                        t = time.perf_counter()
                        unit_kernels.walk_effects(st, dl, planes, applier.STAGE_DETONATE,
                                                  ring, wl)
                        ts.append((time.perf_counter() - t) * 1e3)
                    lines.append((row, size, n, float(np.median(ts))))
    print("TIMING (median ms, the walk incl. its books / count downloads):")
    for row, size, n, ms in lines:
        print(f"  {row:20s} {size:4s} N={n:3d}: {ms:.3f}")
    return 0
