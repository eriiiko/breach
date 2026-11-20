"""The blast flash (graphics G4 P5, #75; docs/explosion_look_design_2026-11-07.md
§3.5 as amended by §6 B1, S1, S9, N2, N5, N7).

A detonation's flash is a ROW property (``[payloads.*] flash``, ``flash_decay_s``,
``flash_color``) carried on the render-only ``ExplosionEvent`` and lit by
``renderer/frame_lights.blast_flash_lights`` as omni transients at the blast
walk's seed cells, sharing ONE cap with the muzzle flashes
(``cap_transient_group``). These tests hold its properties: the envelope, the
source cells, the shared cap, the payload door, that a flash moves no sim field,
and that the sweep's own door holds with everything at its cap.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_blast_flash.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import frame_lights as fl  # noqa: E402
from simulation.weapons import PayloadTable  # noqa: E402

TPS = 24.0


def _blast(pos=(10, 10), flash=100.0, decay=0.5, color=(1.0, 1.0, 1.0),
           tick0=0, approach=None, strength=0.0):
    return {"kind": "blast", "pos": pos, "flash": flash, "flash_decay_s": decay,
            "flash_color": color, "tick0": tick0, "approach": approach,
            "strength": strength, "t": 0.0, "life": 3.0}


def _open(h=21, w=21):
    """An open room with a one-tile wall all round."""
    m = np.zeros((h, w), bool)
    m[1:-1, 1:-1] = True
    return m


def _at(age_s):
    return lambda fx: age_s


def _total(lights):
    return sum(d["intensity"] * max(d["color"]) for d in lights)


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------

def test_the_flash_is_full_on_its_first_tick_and_decays_exponentially():
    """PROPERTY: at age 0 the flash carries exactly the row's ``flash`` light
    units (summed over its cells); at age t it carries ``flash * exp(-t /
    flash_decay_s)``, so the ratio over equal steps is constant. BREAKS IF the
    envelope starts below full (a frame-delayed or ramped flash), is linear, or
    reads the decay as a duration rather than an e-fold."""
    fx = _blast(flash=100.0, decay=0.5)
    assert _total(fl.blast_flash_lights([fx], _at(0.0), _open())) \
        == pytest.approx(100.0)
    for t in (0.1, 0.5, 1.0, 2.0):
        got = _total(fl.blast_flash_lights([fx], _at(t), _open()))
        assert got == pytest.approx(100.0 * math.exp(-t / 0.5), rel=1e-12)
    a = _total(fl.blast_flash_lights([fx], _at(0.2), _open()))
    b = _total(fl.blast_flash_lights([fx], _at(0.4), _open()))
    c = _total(fl.blast_flash_lights([fx], _at(0.6), _open()))
    assert b / a == pytest.approx(c / b, rel=1e-12)


def test_the_flash_is_gone_below_one_percent_of_its_peak():
    """PROPERTY: once ``exp(-age / decay)`` falls under 1 % the flash emits
    nothing (no light, not a vanishing one), and just before it still emits.
    BREAKS IF the cut-off moves (a flash that never ends keeps a cone row
    alive for ever) or the comparison is inverted."""
    fx = _blast(flash=100.0, decay=0.5)
    t_cut = math.log(100.0) * 0.5
    assert fl.blast_flash_lights([fx], _at(t_cut * 0.999), _open())
    assert fl.blast_flash_lights([fx], _at(t_cut * 1.001), _open()) == []


def test_only_blast_effects_with_a_flash_light_anything():
    """PROPERTY: a ``"blast"`` without a flash, or any other effect kind,
    gives no light, and the open mask is not even asked for. BREAKS IF a
    row's default (flash 0) starts lighting, or other effect kinds leak in."""
    asked = []

    def mask():
        asked.append(1)
        return _open()
    effects = [_blast(flash=0.0), {"kind": "tracer", "t": 0.0},
               {"kind": "jet", "t": 0.0}]
    assert fl.blast_flash_lights(effects, _at(0.0), mask) == []
    assert asked == []


def test_the_flash_ages_on_the_callers_sim_clock():
    """PROPERTY: the age is whatever the caller's sim clock says for each
    effect -- two blasts of different ``tick0`` at the same tick have
    different strengths, and the same tick asked twice (a paused relight)
    gives the same light. BREAKS IF the flash keeps its own wall clock (S1:
    a paused game would see it fade, a look shot would not reproduce)."""
    a, b = _blast(pos=(5, 5), tick0=10), _blast(pos=(15, 15), tick0=4)
    clock = lambda fx: (12 - fx["tick0"]) / TPS   # noqa: E731
    one = fl.blast_flash_lights([a, b], clock, _open())
    two = fl.blast_flash_lights([a, b], clock, _open())
    assert one == two
    near_a = _total([d for d in one if d["x"] < 10])
    near_b = _total([d for d in one if d["x"] > 10])
    assert near_a == pytest.approx(100.0 * math.exp(-(2 / TPS) / 0.5))
    assert near_b == pytest.approx(100.0 * math.exp(-(8 / TPS) / 0.5))


# ---------------------------------------------------------------------------
# The source cells
# ---------------------------------------------------------------------------

def test_an_open_centre_flashes_from_itself_and_its_open_neighbours():
    """PROPERTY: a blast on open floor lights its centre and the open
    4-neighbours of it, each at its tile centre (+0.5), the intensity split
    equally; a neighbour that is a wall is skipped. BREAKS IF the flash is a
    single cell (the 16-ray star), lands on tile corners (N2), or a wall
    cell emits."""
    m = _open()
    cells = {(d["y"], d["x"]) for d in fl.blast_flash_lights(
        [_blast(pos=(10, 10))], _at(0.0), m)}
    assert cells == {(10.5, 10.5), (9.5, 10.5), (11.5, 10.5), (10.5, 9.5),
                     (10.5, 11.5)}
    m[10, 11] = False                                 # a wall east of it
    lights = fl.blast_flash_lights([_blast(pos=(10, 10))], _at(0.0), m)
    assert (10.5, 11.5) not in {(d["y"], d["x"]) for d in lights}
    assert len(lights) == 4
    assert all(d["intensity"] == pytest.approx(25.0) for d in lights)


def test_a_charge_on_a_wall_flashes_from_the_walks_seeds_only():
    """PROPERTY: a centre still solid takes the blast walk's own seed rule
    (``blast_expansion.walk_seeds``): the approach side when known, else
    every open 4-neighbour -- the cells the sparks and debris are born at --
    and never a cell inside the wall. BREAKS IF the flash builds its own seed
    rule (it would light through the wall from the wrong side) or lights the
    solid centre."""
    from simulation.blast_expansion import walk_seeds
    m = _open()
    m[:, 10] = False                                  # a wall down column 10
    m[10, 10] = False
    with_side = fl.blast_flash_lights(
        [_blast(pos=(10, 10), approach=(10, 9))], _at(0.0), m)
    assert [(d["y"] - 0.5, d["x"] - 0.5) for d in with_side] \
        == [tuple(t) for t in walk_seeds(m, 10, 10, (10, 9))] == [(10, 9)]
    no_side = fl.blast_flash_lights([_blast(pos=(10, 10))], _at(0.0), m)
    got = sorted((int(d["y"]), int(d["x"])) for d in no_side)
    assert got == sorted(walk_seeds(m, 10, 10, None))
    assert all(m[y, x] for (y, x) in got)


# ---------------------------------------------------------------------------
# The ONE transient cap
# ---------------------------------------------------------------------------

def _muzzles(n, each, color=(1.0, 0.8, 0.6)):
    return [{"x": 5.5 + i % 40, "y": 5.5 + i // 40, "max_range": 0,
             "intensity": each, "color": color} for i in range(n)]


def test_the_group_cap_dims_the_muzzle_flashes_first():
    """PROPERTY: muzzle + blast flashes together carry at most
    ``TRANSIENT_MAX_TOTAL``; when they would pass it the blast flash keeps its
    full light and the muzzle flashes give way (scaled as a group, every
    shooter still present); under the cap nothing moves. BREAKS IF the cap
    is applied per kind (the sum could reach twice the cap and trip the
    sweep's door), or the blast dims before the muzzles (B1's order)."""
    blast = fl.blast_flash_lights([_blast(flash=800.0)], _at(0.0), _open())
    muzzle = _muzzles(200, fl.TRANSIENT_MAX_TOTAL / 200 / 0.999999)
    out = fl.cap_transient_group(muzzle, blast)
    assert len(out) == len(muzzle) + len(blast)
    assert _total(out) <= fl.TRANSIENT_MAX_TOTAL * (1 + 1e-9)
    assert _total(out[len(muzzle):]) == pytest.approx(800.0)
    assert all(d["intensity"] > 0.0 for d in out[:len(muzzle)])
    small = _muzzles(4, 12.0)
    assert fl.cap_transient_group(small, blast) == small + blast


def test_a_blast_alone_over_the_cap_is_dimmed_to_it_and_takes_it_all():
    """PROPERTY: blast flashes that alone pass the cap are scaled to exactly
    it, and the muzzle flashes beside them go to zero (kept in the list).
    BREAKS IF a row's flash can exceed the sweep's share (a mis-tuned
    flashbang would trip the door mid-game) or muzzles still add on top."""
    blast = fl.blast_flash_lights([_blast(flash=3000.0, pos=(5, 5)),
                                   _blast(flash=1000.0, pos=(15, 15))],
                                  _at(0.0), _open())
    muzzle = _muzzles(3, 12.0)
    out = fl.cap_transient_group(muzzle, blast)
    assert _total(out[len(muzzle):]) == pytest.approx(fl.TRANSIENT_MAX_TOTAL)
    assert all(d["intensity"] == 0.0 for d in out[:len(muzzle)])
    near5 = _total([d for d in out[len(muzzle):] if d["x"] < 10])
    assert near5 / _total(out[len(muzzle):]) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# The payload door
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cols, words", [
    (dict(flash=-1.0), "flash"),
    (dict(flash=10.0, flash_decay_s=0.0), "flash_decay_s"),
    (dict(flash=10.0, flash_decay_s=-0.2), "flash_decay_s"),
    (dict(flash=10.0, flash_color=[1.0, 1.2, 1.0]), "flash_color"),
    (dict(flash=10.0, flash_color=[1.0, 1.0]), "flash_color"),
    (dict(flash=10.0, flash_color=[-0.1, 1.0, 1.0]), "flash_color"),
    (dict(flash=10.0, flash_decay=0.5), "flash_decay"),
    (dict(flash_colour=[1.0, 1.0, 1.0]), "flash_colour"),
])
def test_the_payload_door_refuses_a_bad_flash(cols, words):
    """PROPERTY: the payload door refuses a negative flash, a non-positive
    decay on a row with a flash, a colour outside [0, 1]^3 or not three
    numbers, and ANY ``flash*`` key that is not one of the three columns
    (S9), naming the row and the column. BREAKS IF a misspelt column silently
    falls back to its default (a flash Erik tunes that never changes)."""
    with pytest.raises(ValueError) as ei:
        PayloadTable({"bad_row": dict(cols)})
    assert "bad_row" in str(ei.value) and words in str(ei.value)


def test_the_flash_columns_default_to_no_flash_and_pass_the_strength_door():
    """PROPERTY: a row without flash columns has ``flash`` 0 (no light), a
    decay of 0 is accepted when there is no flash, and a strength row carries
    its flash through the strength door. BREAKS IF the default lights every
    explosion, or the strength door starts refusing the flash columns."""
    t = PayloadTable({
        "plain": dict(radius=2, wall_damage=10),
        "dark": dict(flash=0.0, flash_decay_s=0.0),
        "lit": dict(strength=1.0, radius=2, explosive="comp_b",
                    emit_blast_smoke=True, flash=60.0, flash_decay_s=0.4,
                    flash_color=[1.0, 0.9, 0.8])})
    assert t.by_name["plain"].flash == 0.0
    assert t.by_name["dark"].flash == 0.0
    lit = t.by_name["lit"]
    assert (lit.strength, lit.flash, lit.flash_decay_s, lit.flash_color) \
        == (1.0, 60.0, 0.4, (1.0, 0.9, 0.8))


# ---------------------------------------------------------------------------
# A flash moves no sim field
# ---------------------------------------------------------------------------

def _flash_level():
    from level_loader import LevelData
    tm = np.zeros((20, 20), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    return LevelData(name="blast_flash", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def test_a_flash_only_row_emits_one_event_and_no_field_edit():
    """PROPERTY: a row with a flash and nothing else (the shipped
    ``flashbang``) executes as exactly one ``ExplosionEvent`` carrying its
    flash columns, enqueues no FieldEdit, logs no blast, and a sim that ran it
    has the SAME field digest as its twin that did not, tick after tick.
    BREAKS IF the flash ever reaches a sim field (it would move the goldens
    and make light game state before #81), or a flash-only row loses its
    event (the renderer would never see it)."""
    bp = pytest.importorskip("breach_physics")
    from field_ab_harness import capture_snapshot
    from field_digest import field_digest
    from simulation import Simulation
    from simulation.events import ExplosionEvent
    from simulation.payloads import execute_payload
    from simulation.weapons import get_tables

    row = get_tables().payloads.by_name["flashbang"]
    assert row.flash > 0.0 and row.strength == 0.0 and row.radius == 0
    sims = [Simulation(_flash_level(), seed=5, breach_physics=bp,
                       enable_recorder=False) for _ in range(2)]
    for s in sims:
        s.set_paused(False)
        s.step()
    lit = sims[0]
    events = []
    n_queue = len(lit.edit_queue)
    log_before = list(getattr(lit.gmap, "blast_log", []))
    execute_payload(lit.gmap, lit.edit_queue, lit.units, 10, 10, row, lit.draws,
                    events=events, kind="flashbang")
    assert len(lit.edit_queue) == n_queue
    assert list(getattr(lit.gmap, "blast_log", [])) == log_before
    assert len(events) == 1 and isinstance(events[0], ExplosionEvent)
    ev = events[0]
    assert (ev.flash, ev.flash_decay_s, tuple(ev.flash_color)) \
        == (row.flash, row.flash_decay_s, tuple(row.flash_color))
    for _ in range(3):
        for s in sims:
            s.step()
        a, b = (field_digest(capture_snapshot(s)) for s in sims)
        assert a == b


def test_a_strength_rows_event_carries_its_flash():
    """PROPERTY: the shipped frag's detonation event carries the row's flash
    columns as authored (the renderer reads nothing else). BREAKS IF
    ``execute_payload`` drops the columns from the event."""
    from simulation.events import ExplosionEvent
    from simulation.field_edit import EditQueue
    from simulation.gamemap import GameMap
    from simulation.payloads import execute_payload
    from simulation.weapons import get_tables

    row = get_tables().payloads.by_name["frag_standard"]
    gmap = GameMap(_flash_level())
    events = []
    execute_payload(gmap, EditQueue(), [], 10, 10, row,
                    np.random.default_rng(1), events=events, kind="grenade")
    ev = [e for e in events if isinstance(e, ExplosionEvent)]
    assert len(ev) == 1
    assert (ev[0].flash, ev[0].flash_decay_s, tuple(ev[0].flash_color)) \
        == (row.flash, row.flash_decay_s, tuple(row.flash_color))


# ---------------------------------------------------------------------------
# The sweep's own door holds with everything at its cap (N7)
# ---------------------------------------------------------------------------

def test_a_recipe_ships_lamps_muzzles_and_a_flashbang_pass_the_sweeps_door():
    """PROPERTY: a recipe ship's lamps, a full volley of muzzle flashes, a
    flashbang at its full flash and the cursor lamp, assembled through the
    one transient cap and ``cone_rows``, put at most ``LIGHT_CONE_BUDGET``
    into every ordinate per channel -- measured with the sweep's own
    projection ``RadiationSweep.cone_emission`` (N7: ``cone_rows`` enforces
    nothing). BREAKS IF the blast flashes get a cap of their own on top of
    the muzzles' (B1), or gen_ship's lamp budget stops leaving the transient
    share free."""
    bp = pytest.importorskip("breach_physics")
    import gen_ship as G
    plan = G.generate(2, "large", recipe="freighter", profile="standard",
                      zombies=4)
    lamps = plan.embedding.lights
    h, w = plan.tilemap.shape
    mask = np.zeros((h, w), bool)
    mask[1:-1, 1:-1] = True
    blast = fl.blast_flash_lights(
        [_blast(pos=(w // 2, h // 2), flash=fl.TRANSIENT_MAX_TOTAL * 3)],
        _at(0.0), mask)
    muzzle = _muzzles(200, 12.0, color=(1.0, 1.0, 1.0))
    extra = [fl.cursor_lamp_spec((w / 2.0, h / 2.0))]
    extra += fl.transient_specs(fl.cap_transient_group(muzzle, blast))
    specs = fl.frame_light_specs(lamps, total_tick=0,
                                 sim_time_per_tick=1.0 / 60, extra=extra)
    rows = fl.cone_rows(specs.specs, w, h, int(bp.L_FINE_BITS))
    per_ordinate = sum(np.asarray(bp.RadiationSweep.cone_emission(
        r[2:5].tolist(), int(r[5]), int(r[6]))) for r in rows)
    assert per_ordinate.max() <= fl.LIGHT_CONE_BUDGET
