"""P10 — ``levels/wego_test`` loads and delivers what the HUMAN-TEST needs.

Erik's kickoff ruling 3: a dedicated level, with playground and
planetside_demo left exactly as they are. These pin that the level actually
exercises the arc — cover in both flavours, a closed door to breach, and two
sides far enough apart that round one is a planning round — so a green suite
means the thing Erik is about to play is really there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import load as load_level  # noqa: E402
from simulation import orders as O  # noqa: E402
from simulation.ruleset import OnePhaseWEGO  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402
import ui  # noqa: E402


@pytest.fixture(scope="module")
def level():
    return load_level("wego_test")


def _sim(level):
    sim = Simulation(level, seed=42, breach_physics=None,
                     enable_recorder=False, ruleset=OnePhaseWEGO())
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    sim.set_paused(False)
    return sim


def test_the_level_loads(level):
    assert level.name == "wego_test"
    assert level.tile_size_m == 0.333


def test_it_fields_both_sides(level):
    marines = [s for s in level.spawns if s.team == 0]
    zombies = [s for s in level.spawns if s.team == 1]
    assert len(marines) == 4
    assert len(zombies) == 5


def test_round_one_is_a_planning_round_not_a_brawl(level):
    """The two sides start far enough apart (and behind a closed door) that
    the player gets to plan before anything happens."""
    sim = _sim(level)
    marines = [u for u in sim.units if u.team == 0]
    zombies = [u for u in sim.units if u.team == 1]
    closest = min(abs(m.x - z.x) + abs(m.y - z.y)
                  for m in marines for z in zombies)
    assert closest > 40


def test_cover_is_present_in_both_flavours(level):
    sim = _sim(level)
    assert len(sim.cover) >= 8
    crates = [c for c in sim.cover if not c.blocks_los]
    barricades = [c for c in sim.cover if c.blocks_los]
    assert crates, "no see-over crates — §7's main case is untestable"
    assert barricades, "no full-height barricades — blocks_los is unexercised"


def test_a_one_metre_crate_is_exactly_three_tiles(level):
    """0.333 m/tile gives tiles_per_m == 3 exactly (a6 §3), which is why the
    level authors round metre extents."""
    sim = _sim(level)
    crate = next(c for c in sim.cover if c.id == "crate_entry_a")
    assert (crate.x1 - crate.x0) == 3.0
    assert (crate.y1 - crate.y0) == 3.0


def test_cover_never_became_architecture(level):
    """Cover is INTANGIBLE: it must not stamp into the material grid, or it
    would block pathfinding and airflow and stop being cover."""
    sim = _sim(level)
    for c in sim.cover:
        assert sim.gmap.is_passable_block(int(c.y0), int(c.x0), 3), \
            f"cover {c.id} made its tiles impassable"


def test_there_is_a_closed_door_to_breach(level):
    sim = _sim(level)
    assert sim._doors, "no door — §12's breach problem is missing"
    assert any(int(getattr(d, "state", 0)) == 0 for d in sim._doors)


def test_the_bulkhead_actually_separates_the_rooms(level):
    """A marine must not simply be able to walk around the door."""
    sim = _sim(level)
    marine = next(u for u in sim.units if u.team == 0)
    zombie = next(u for u in sim.units if u.team == 1)
    assert not sim.gmap.has_los(marine.center_tile_y(), marine.center_tile_x(),
                                zombie.center_tile_y(), zombie.center_tile_x())


def test_fog_hides_the_far_room_at_the_start(level):
    """§8: with the bulkhead closed, the player cannot see what is waiting."""
    sim = _sim(level)
    assert ui.drawable_enemies(sim, 0) == []


def test_a_cross_round_plan_arrives_exactly_when_promised(level):
    """The real integration check, and a nice demonstration of §3 + §13 on the
    actual play level: a walk across this room takes LONGER THAN ONE ROUND, so
    the plan spans the seam — and still arrives on the tick the teal label
    promised, because absolute ticks make the boundary a non-event."""
    from config import CFG
    sim = _sim(level)
    marine = next(u for u in sim.units if u.team == 0)
    assert sim.apply_action(marine.id, O.Order(O.ORDER_MOVE, 30, 31, 0))
    overlay = ui.plan_overlay(sim, marine)
    assert overlay.paths, "the move compiled no path"
    promised = overlay.paths[0].arrival_seconds
    assert promised > CFG.clock.round_duration_seconds, \
        "this walk was meant to overrun a single round"

    for _ in range(round(promised * CFG.clock.ticks_per_second)):
        sim.step()
        sim.set_paused(False)
    assert sim.round_index >= 1, "the plan should have crossed a seam"
    assert (marine.tile_x, marine.tile_y) == (30, 31)


def test_a_full_round_of_simulation_runs_clean(level):
    """Every system on at once — timeline, vision, cover, engagement, charges —
    for a whole round, with no exception."""
    sim = _sim(level)
    for u in sim.units:
        if u.team == 0:
            sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 30, int(u.y) + 1, 0))
    for _ in range(sim.ticks_per_round + 5):
        sim.step()
        sim.set_paused(False)
    assert sim.round_index >= 1
    assert any(u.alive for u in sim.units if u.team == 0)


def test_playground_and_planetside_are_untouched():
    """Erik's ruling 3: the new level exists precisely so these do not change."""
    for name in ("playground", "planetside_demo"):
        lvl = load_level(name)
        assert not [e for e in (lvl.entities or [])
                    if e.class_name == "cover"], \
            f"{name} grew cover entities; wego_test exists to avoid that"
