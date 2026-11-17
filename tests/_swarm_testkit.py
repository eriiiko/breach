"""The swarm tests' shared kit (arc #63 P3, docs/swarm_P3_impl.md §11; F7,
R10, R18). A harness module (``_*``), not collected.

* Species rows: each species' REPO row is captured at import
  (``dict(vars(CFG.swarm.<sp>))``), so a test's rows grow with
  ``SPECIES_FIELDS`` and never pin the field set. ``species_row`` /
  ``swarm_namespace`` / ``swarm_cfg`` return FRESH objects per call;
  ``swarm_cfg`` is a full config wrapper carrying ``swarm`` AND
  ``physics.temperature_scale`` (and ``clock``) copied from ``CFG``, so a
  rejection can only come from the validator.
* Law worlds: ``larva_world`` builds the direct-binding arrays (the 9 ROSTER
  columns ``(n_env, max_units)``, ``high_water``, ``T``, ``solid``, ``env``,
  the table rows, the records) for ``bp.swarm_species_step("larva", ...)``;
  ``larva_tables`` builds the rows in ``swarm_tables.ROWS`` order from named
  values (explicit test params, never the config unless a test says so).
  Synthetic ``T`` / ``solid`` arrays are test inputs, never
  ``gmap.temperature`` / ``gmap.solid``.
* ``as_store_namespace(world, e)`` presents env e with the GameMap attribute
  names as per-map views, so tests call the REAL ``swarm.check_invariants``.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG, Namespace  # noqa: E402
from simulation import swarm, swarm_fixed, swarm_tables  # noqa: E402
from simulation.rng_salts import Salt  # noqa: E402

FP_ONE = swarm_fixed.FP_ONE
PI_Q16 = swarm_fixed.PI_Q16

# Each species' repo row, captured once at import.
_REPO_ROWS = {sp: dict(vars(getattr(CFG.swarm, sp))) for sp in swarm.SWARM_SPECIES}


def species_row(sp: str, **overrides) -> dict:
    """A fresh copy of ``sp``'s repo row with ``overrides`` applied."""
    row = dict(_REPO_ROWS[sp])
    row.update(overrides)
    return row


def swarm_namespace(**overrides) -> Namespace:
    """A fresh ``[swarm]`` section (every species, full rows) -- the shape
    ``CFG.swarm`` has; for ``monkeypatch.setattr(CFG, "swarm", ...)``."""
    return Namespace({sp: species_row(sp, **overrides) for sp in swarm.SWARM_SPECIES})


def cfg_with(swarm_section) -> Namespace:
    """A full config wrapper: the given ``swarm`` section plus copies of
    ``CFG.physics.temperature_scale`` and ``CFG.clock``."""
    cfg = Namespace({})
    if swarm_section is not None:
        cfg.swarm = swarm_section
    cfg.physics = Namespace({})
    cfg.physics.temperature_scale = copy.deepcopy(CFG.physics.temperature_scale)
    cfg.clock = copy.deepcopy(CFG.clock)
    return cfg


def swarm_cfg(**overrides) -> Namespace:
    """``cfg_with(swarm_namespace(**overrides))``."""
    return cfg_with(swarm_namespace(**overrides))


# ---------------------------------------------------------------------------
# Law worlds (the direct binding's arrays)
# ---------------------------------------------------------------------------

# Explicit test params (Q16 ints): a comfortable band around 0, coma below
# -20, waking above -10, death above +50 (game counts; never the config).
DEFAULT_LARVA = {
    "T_prefer_q": 0,
    "T_lethal_hot_q": 50 * FP_ONE,
    "T_ctmin_q": -20 * FP_ONE,
    "T_coma_exit_q": -10 * FP_ONE,
    "turn_bias_q": swarm_fixed.quantize_scalar(0.8),
    "turn_min_q": swarm_fixed.quantize_scalar(0.5236),
    "turn_max_q": swarm_fixed.quantize_scalar(2.0944),
    "sweep_angle_q": swarm_fixed.quantize_scalar(0.7854),
    "step_q": 1640,
    "p_base_q": 273,
    "p_alarm_q": 2731,
    "salt_tumble": int(Salt.SWARM_TUMBLE),
    "salt_wall_turn": int(Salt.SWARM_WALL_TURN),
    "bite_q": 218,          # arc #63 P5a: 0.08 bites/s x 1 unit at 24 tps
}


def larva_tables(n_env: int = 1, **named) -> np.ndarray:
    """``(n_env, width)`` int64 rows in ``swarm_tables.ROWS['larva']`` order;
    a named value may be a scalar (every env) or a length-n_env sequence."""
    rows = swarm_tables.ROWS["larva"]
    vals = dict(DEFAULT_LARVA)
    vals.update(named)
    out = np.zeros((n_env, len(rows)), dtype=np.int64)
    for i, r in enumerate(rows):
        v = vals[r.name]
        out[:, i] = np.asarray(v, dtype=np.int64)
    return out


class LarvaWorld(SimpleNamespace):
    """The direct-binding arrays of one law call (mutated in place)."""

    def columns(self):
        return tuple(getattr(self, c.name) for c in swarm.ROSTER)

    def copy(self):
        w = LarvaWorld(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                          for k, v in vars(self).items()})
        return w

    def args(self):
        return (self.columns(), self.high_water, self.T, self.solid, self.env,
                self.tables, self.death)


def larva_world(h: int, w: int, max_units: int, n_env: int = 1, *, T=None,
                solid=None, tick=0, seed=(12345, 678), **table_values) -> LarvaWorld:
    """An empty world: every slot the all-zero form, ``hw`` 0, a flat field
    of 0 (comfortable), no solid tile, records pre-filled with garbage."""
    wd = LarvaWorld()
    for c in swarm.ROSTER:
        setattr(wd, c.name, np.zeros((n_env, max_units), dtype=c.dtype))
    wd.high_water = np.zeros((n_env,), dtype=np.int32)
    wd.T = (np.zeros((n_env, h, w), dtype=np.int32) if T is None
            else np.ascontiguousarray(np.broadcast_to(np.asarray(T, dtype=np.int32),
                                                      (n_env, h, w))).copy())
    wd.solid = (np.zeros((n_env, h, w), dtype=np.bool_) if solid is None
                else np.ascontiguousarray(np.broadcast_to(np.asarray(solid, dtype=np.bool_),
                                                          (n_env, h, w))).copy())
    wd.env = np.zeros((n_env, swarm.ENV_WORDS), dtype=np.uint32)
    wd.env[:, swarm.ENV_TICK] = np.asarray(tick, dtype=np.uint64) & 0xFFFFFFFF
    wd.env[:, swarm.ENV_SEED_LO] = seed[0]
    wd.env[:, swarm.ENV_SEED_HI] = seed[1]
    wd.tables = larva_tables(n_env, **table_values)
    wd.death = np.full((n_env, max_units, swarm.DEATH_WORDS), -7, dtype=np.int32)
    wd.next_unit_id = np.full((n_env,), swarm.FIRST_UNIT_ID, dtype=np.int64)
    wd.h, wd.w, wd.max_units, wd.n_env = h, w, max_units, n_env
    return wd


def place(world: LarvaWorld, e: int, slot: int, x_q: int, y_q: int, heading_q: int,
          state: int = swarm.AI_RUN, uid=None, dist: int = 0, hp: int = FP_ONE,
          faction: int = 0) -> int:
    """Write a live larva into env e's slot (a test fixture's own store, not
    a GameMap: the facade rule is the sim's). Returns its unit id."""
    if uid is None:
        uid = int(world.next_unit_id[e])
        world.next_unit_id[e] = uid + 1
    else:
        world.next_unit_id[e] = max(int(world.next_unit_id[e]), int(uid) + 1)
    world.pos_x[e, slot] = x_q
    world.pos_y[e, slot] = y_q
    world.heading[e, slot] = swarm_fixed.wrap_heading_q16(int(heading_q))
    world.hp[e, slot] = hp
    world.ai_state[e, slot] = state
    world.dist_walked[e, slot] = dist
    world.unit_id[e, slot] = uid
    world.faction[e, slot] = faction
    world.valid[e, slot] = 1
    world.high_water[e] = max(int(world.high_water[e]), slot + 1)
    return int(uid)


def step_cpu(world: LarvaWorld) -> int:
    import breach_physics as bp
    return int(bp.swarm_species_step("larva", *world.args()))


def as_store_namespace(world: LarvaWorld, e: int) -> SimpleNamespace:
    """Env e of a law world with the GameMap attribute names (per-map views),
    for the REAL ``swarm.check_invariants``."""
    ns = SimpleNamespace()
    for sp in swarm.SWARM_SPECIES:
        for c in swarm.ROSTER:
            setattr(ns, swarm.store_attr(sp, c.name), getattr(world, c.name)[e])
        setattr(ns, swarm.high_water_attr(sp), np.array(world.high_water[e], dtype=np.int32))
    ns.swarm_next_unit_id = np.array(int(world.next_unit_id[e]), dtype=np.int32)
    ns.swarm_species = swarm.SwarmSpeciesTable.from_rows([
        swarm.SwarmSpeciesRow(**{**{f.name: 0 for f in _row_fields()},
                                 "species": sp, "max_units": world.max_units})
        for sp in swarm.SWARM_SPECIES])
    ns._h, ns._w = world.h, world.w
    return ns


def _row_fields():
    from dataclasses import fields
    return fields(swarm.SwarmSpeciesRow)


def philox_words(tick, salt, uid, key):
    """The Philox block a draw (tick, 0, salt, uid) takes under key."""
    import breach_physics as bp
    return bp.philox32_4x32_10(int(tick) & 0xFFFFFFFF, 0, int(salt), int(uid),
                               int(key[0]), int(key[1]))


def turn_heading_ref(heading, pref, w_side, w_mag, bias_q, tmin, tmax):
    """The kit's turn draw, transcribed for a test oracle (philox maps
    through the module's own functions; the wrap through swarm_fixed)."""
    import breach_physics as bp
    u = bp.philox32_uniform_q16(int(w_side))
    if pref == 0:
        side = 1 if u < 32768 else -1
    else:
        side = pref if u < bias_q else -pref
    m = tmin + bp.philox32_below(int(w_mag), (tmax - tmin) + 1)
    return swarm_fixed.wrap_heading_q16(int(heading) + side * m)
