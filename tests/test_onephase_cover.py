"""P5 — physical cover (onephase_wego design §7).

§7 is blunt about what this replaces: "There is no statistical to-hit model and
no XCOM-style modifier stack… Cover is physical. Bullets ray-march in
continuous space; a cover object physically eats the rays that clip it. A
marine hugging a crate is protected exactly as much as geometry says — no
cover bonus stat anywhere."

These pin that literally: rectangles stop rounds, rectangles break when shot,
the statistical roll is never consulted under this ruleset, cover does NOT
block vision unless authored to, and the shipped statistical path is untouched
for TwoPhaseWEGO.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import orders as O  # noqa: E402
from simulation import vision as V  # noqa: E402
from simulation.cover_system import (  # noqa: E402
    CoverRuntime, build_cover, cover_at,
)
from simulation.entities import REGISTRY  # noqa: E402
from simulation.ruleset import OnePhaseWEGO, TwoPhaseWEGO  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402


class _Inst:
    """Minimal stand-in for a parsed [[entity]] instance (the loader's own
    EntityInstance shape: ordinal / id / class_name / fields)."""

    def __init__(self, ordinal, id_, **fields):
        self.ordinal = ordinal
        self.id = id_
        self.class_name = "cover"
        self.fields = {"width_m": 1.0, "height_m": 1.0, "hp": 60,
                       "blocks_los": False, **fields}


def _level(h=48, w=48, entities=()):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    lvl = LevelData(name="onephase_cover", version="2", path=Path("."),
                    tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))
    lvl.entities = list(entities)
    return lvl


def _sim(entities=(), ruleset=None):
    sim = Simulation(_level(entities=entities), seed=5, breach_physics=None,
                     enable_recorder=False,
                     ruleset=ruleset if ruleset is not None else OnePhaseWEGO())
    sim.set_paused(False)
    return sim


def _marine(sim, x, y, name="m"):
    u = Unit(name, x=x, y=y, team=0)
    sim.add_unit(u)
    return u


def _zombie(sim, x, y, name="z", hp=10_000):
    u = Unit(name, x=x, y=y, team=1)
    sim.add_unit(u)
    u.current_hp = hp
    return u


def _run(sim, n):
    for _ in range(n):
        sim.step()
        sim.set_paused(False)


def _shoot(sim, shooter, target, ticks=24):
    sim.apply_action(shooter.id, O.Order(O.ORDER_SHOOT, target.tile_x,
                                         target.tile_y, 0,
                                         target_unit_id=target.id))
    _run(sim, ticks)


# ---------------------------------------------------------------------------
# The entity + its runtime
# ---------------------------------------------------------------------------
def test_cover_is_a_registered_entity_class():
    assert "cover" in REGISTRY
    fields = {f.name for f in REGISTRY["cover"].FIELDS}
    assert {"x", "y", "width_m", "height_m", "hp", "blocks_los"} <= fields


def test_cover_never_occupies_the_material_grid():
    """A crate must not become architecture: if it stamped into `material` it
    would block pathfinding and airflow and stop being cover."""
    assert REGISTRY["cover"].INTANGIBLE is True
    sim = _sim([_Inst(0, "c1", x=10, y=10, width_m=3.0, height_m=1.0)])
    assert sim.gmap.material[10, 10] == 0
    assert not sim.gmap.solid[10, 10]
    assert sim.gmap.is_passable_block(10, 10, 3)


def test_extents_quantize_from_meters():
    sim = _sim([_Inst(0, "c1", x=10, y=12, width_m=3.0, height_m=2.0)])
    c = sim.cover[0]
    assert (c.x0, c.y0, c.x1, c.y1) == (10.0, 12.0, 13.0, 14.0)


def test_the_rectangle_is_half_open():
    """Two abutting crates must not both claim the seam between them."""
    sim = _sim([_Inst(0, "a", x=10, y=10, width_m=2.0, height_m=2.0),
                _Inst(1, "b", x=12, y=10, width_m=2.0, height_m=2.0)])
    assert cover_at(sim.cover, 11.9, 10.5).id == "a"
    assert cover_at(sim.cover, 12.0, 10.5).id == "b"


def test_cover_is_built_in_ordinal_order():
    sim = _sim([_Inst(2, "c", x=20, y=10), _Inst(0, "a", x=10, y=10),
                _Inst(1, "b", x=15, y=10)])
    assert [c.id for c in sim.cover] == ["a", "b", "c"]


def test_a_cover_free_level_builds_an_empty_list():
    assert _sim().cover == []


# ---------------------------------------------------------------------------
# Cover rows must BE their runtime objects in sim.entities
# ---------------------------------------------------------------------------
# ★ The crash Erik hit on the first play: sim.cover was built as a PARALLEL
# list while the bare EntityInstance stayed in sim.entities — and the entity
# list is what the serializer walks, asking each row's class for
# runtime_digest_rows. A parsed instance has no `alive`/`hp_now`, so the
# recorder raised on the first tick of a cover level. Every cover test here ran
# enable_recorder=False, which is exactly why nothing caught it.
def test_cover_rows_are_runtime_objects_in_the_entity_list():
    sim = _sim([_Inst(0, "crate", x=10, y=10)])
    row = next(e for e in sim.entities if e.class_name == "cover")
    assert row is sim.cover[0], "sim.entities holds a different object"
    assert hasattr(row, "alive") and hasattr(row, "hp_now")


def test_a_cover_level_serializes():
    """The direct regression: the call the recorder makes every tick."""
    from simulation.entities.serialize import serialize_entity_state
    sim = _sim([_Inst(0, "crate", x=10, y=10)])
    assert serialize_entity_state(sim.entities)


def test_a_cover_level_ticks_with_the_recorder_on():
    """The end-to-end shape of Erik's crash — a real recorder, a real tick."""
    lvl = _level(entities=[_Inst(0, "crate", x=10, y=10)])
    sim = Simulation(lvl, seed=5, breach_physics=None, enable_recorder=True,
                     ruleset=OnePhaseWEGO())
    sim.set_paused(False)
    _marine(sim, 4.0, 20.0)
    _run(sim, 5)
    assert sim.tick == 5


def test_destroying_cover_moves_its_digest_rows():
    sim = _sim([_Inst(0, "crate", x=10, y=10, hp=5)])
    crate = sim.cover[0]
    before = REGISTRY["cover"].runtime_digest_rows(crate)
    crate.chew(5)
    after = REGISTRY["cover"].runtime_digest_rows(crate)
    assert dict(before)["alive"] == 1 and dict(after)["alive"] == 0
    assert dict(after)["hp_now"] == 0


# ---------------------------------------------------------------------------
# Cover eats rounds (§7)
# ---------------------------------------------------------------------------
def test_a_crate_between_shooter_and_target_stops_the_rounds():
    sim = _sim([_Inst(0, "crate", x=12, y=20, width_m=2.0, height_m=4.0)])
    m = _marine(sim, 4.0, 20.0)
    z = _zombie(sim, 20.0, 20.0)
    hp0 = z.current_hp
    _shoot(sim, m, z)
    assert z.current_hp == hp0, "rounds went through the crate"


def test_the_same_shot_connects_once_the_crate_is_gone():
    """The control for the test above — same geometry, no crate."""
    sim = _sim()
    m = _marine(sim, 4.0, 20.0)
    z = _zombie(sim, 20.0, 20.0)
    hp0 = z.current_hp
    _shoot(sim, m, z)
    assert z.current_hp < hp0


def test_rounds_chew_the_crate_and_it_breaks():
    """§7: cover is static-but-DESTRUCTIBLE."""
    sim = _sim([_Inst(0, "crate", x=12, y=20, width_m=2.0, height_m=4.0,
                      hp=4)])
    m = _marine(sim, 4.0, 20.0)
    z = _zombie(sim, 20.0, 20.0)
    crate = sim.cover[0]
    _shoot(sim, m, z, ticks=60)
    assert crate.hp_now < crate.hp_max
    assert crate.alive is False


def test_a_broken_crate_stops_protecting():
    sim = _sim([_Inst(0, "crate", x=12, y=20, width_m=2.0, height_m=4.0,
                      hp=2)])
    m = _marine(sim, 4.0, 20.0)
    z = _zombie(sim, 20.0, 20.0)
    hp0 = z.current_hp
    _shoot(sim, m, z, ticks=120)
    assert sim.cover[0].alive is False
    assert z.current_hp < hp0, "rounds never got through the wreckage"


def test_flanking_the_crate_needs_no_directional_bookkeeping():
    """Directional cover is geometric by construction (§7): an approach that
    does not have the crate in the way simply hits."""
    sim = _sim([_Inst(0, "crate", x=12, y=20, width_m=2.0, height_m=4.0)])
    blocked = _marine(sim, 4.0, 20.0, name="blocked")
    flanker = _marine(sim, 20.0, 8.0, name="flanker")
    z = _zombie(sim, 20.0, 20.0)
    hp0 = z.current_hp
    _shoot(sim, flanker, z)
    assert z.current_hp < hp0
    assert blocked is not None


def test_cover_position_matters_not_a_stat():
    """"A marine hugging a crate is protected exactly as much as geometry
    says" — step the target out from behind it and the same shot connects."""
    sim = _sim([_Inst(0, "crate", x=12, y=20, width_m=2.0, height_m=2.0)])
    m = _marine(sim, 4.0, 20.0)
    z = _zombie(sim, 20.0, 20.0)
    hp0 = z.current_hp
    _shoot(sim, m, z, ticks=12)
    assert z.current_hp == hp0
    z.x, z.y = 20.0, 28.0                    # step clear of the crate's row
    m.facing = 0.0
    sim.apply_action(m.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 24)
    assert z.current_hp < hp0


# ---------------------------------------------------------------------------
# No statistical to-hit model under this ruleset (§7)
# ---------------------------------------------------------------------------
def test_the_exposure_roll_is_never_consulted(monkeypatch):
    from simulation import attack_resolver

    def _boom(*a, **k):
        raise AssertionError("the statistical exposure roll ran under "
                             "OnePhaseWEGO (design §7 forbids it)")

    monkeypatch.setattr(attack_resolver, "roll_exposure", _boom)
    sim = _sim()
    m = _marine(sim, 4.0, 20.0)
    z = _zombie(sim, 12.0, 20.0)
    # Paint a furniture tile in the line of fire — the material the shipped
    # statistical path treats as concealment.
    from simulation.materials import MAT_FURNITURE
    sim.gmap.material[20, 8] = MAT_FURNITURE
    _shoot(sim, m, z)


def test_the_shipped_statistical_path_is_untouched():
    """TwoPhaseWEGO still runs the exposure roll — the model is a property of
    the RULESET, and the old one is unchanged."""
    from simulation import attack_resolver
    from simulation.combat import BulletInFlight
    import inspect
    src = inspect.getsource(BulletInFlight.advance)
    assert "cover_exposure_at" in src
    assert attack_resolver.roll_exposure is not None
    sim = _sim(ruleset=TwoPhaseWEGO())
    assert sim.ruleset.drives_units is False


# ---------------------------------------------------------------------------
# Cover vs vision (§7/§8): you can see over a crate
# ---------------------------------------------------------------------------
def test_an_ordinary_crate_does_not_block_line_of_sight():
    sim = _sim([_Inst(0, "crate", x=12, y=20, width_m=2.0, height_m=4.0)])
    m = _marine(sim, 4.0, 20.0)
    m.facing = 0.0                            # look east, down the crate line
    z = _zombie(sim, 20.0, 20.0)
    assert V.can_see(sim, m, z) is True
    assert z.id in sim.visible_enemy_ids(0)


def test_a_full_height_barricade_does_block_it():
    sim = _sim([_Inst(0, "barricade", x=12, y=16, width_m=2.0, height_m=12.0,
                      blocks_los=True)])
    m = _marine(sim, 4.0, 20.0)
    m.facing = 0.0
    z = _zombie(sim, 20.0, 20.0)
    assert V.can_see(sim, m, z) is False


def test_a_destroyed_barricade_stops_occluding():
    sim = _sim([_Inst(0, "barricade", x=12, y=16, width_m=2.0, height_m=12.0,
                      blocks_los=True)])
    m = _marine(sim, 4.0, 20.0)
    m.facing = 0.0
    z = _zombie(sim, 20.0, 20.0)
    assert V.can_see(sim, m, z) is False
    sim.cover[0].alive = False
    sim._vision_cache = None
    assert V.can_see(sim, m, z) is True


def test_cover_lowers_the_exposed_profile():
    """§7/§9: physical cover is what makes "largest exposed profile"
    meaningful — it is measured against these rectangles."""
    open_sim = _sim()
    m = _marine(open_sim, 4.0, 20.0)
    z = _zombie(open_sim, 20.0, 20.0)
    assert V.exposed_profile(open_sim, m, z) == 1.0

    covered = _sim([_Inst(0, "barricade", x=18, y=16, width_m=1.0,
                          height_m=5.0, blocks_los=True)])
    m2 = _marine(covered, 4.0, 20.0)
    z2 = _zombie(covered, 20.0, 20.0)
    assert V.exposed_profile(covered, m2, z2) < 1.0


# ---------------------------------------------------------------------------
# The slab test
# ---------------------------------------------------------------------------
def test_blocks_segment_geometry():
    c = CoverRuntime(_Inst(0, "c", x=10, y=10), 10, 10, 2, 2)
    assert c.blocks_segment(5, 11, 15, 11) is True     # straight through
    assert c.blocks_segment(5, 5, 15, 5) is False      # passes above
    assert c.blocks_segment(11, 5, 11, 15) is True     # vertical, through
    assert c.blocks_segment(5, 11, 9, 11) is False     # stops short
    assert c.blocks_segment(5, 5, 15, 15) is True      # diagonal, clips it


def test_a_dead_shape_neither_blocks_nor_contains():
    c = CoverRuntime(_Inst(0, "c", x=10, y=10), 10, 10, 2, 2)
    c.alive = False
    assert c.contains(11, 11) is False
    assert c.blocks_segment(5, 11, 15, 11) is False


def test_chew_reports_the_break_once():
    c = CoverRuntime(_Inst(0, "c", x=10, y=10, hp=5), 10, 10, 2, 2)
    assert c.chew(2) is False
    assert c.hp_now == 3
    assert c.chew(3) is True
    assert c.hp_now == 0 and c.alive is False
    assert c.chew(9) is False, "an already-broken shape cannot break again"


# ---------------------------------------------------------------------------
# Facing determines defense (§9), wired into the march
# ---------------------------------------------------------------------------
def test_a_shot_from_behind_hits_harder():
    front = _sim()
    fm = _marine(front, 20.0, 4.0, name="shooter")
    fz = _zombie(front, 20.0, 20.0)
    fz.facing = -3.14159 / 2          # facing SOUTH, away from the shooter...
    back = _sim()
    bm = _marine(back, 20.0, 4.0, name="shooter")
    bz = _zombie(back, 20.0, 20.0)
    bz.facing = 3.14159 / 2           # ...vs facing NORTH, into the shooter

    _shoot(front, fm, fz, ticks=12)
    _shoot(back, bm, bz, ticks=12)
    took_from_behind = 10_000 - front.units[1].current_hp
    took_head_on = 10_000 - back.units[1].current_hp
    assert took_from_behind > took_head_on > 0
