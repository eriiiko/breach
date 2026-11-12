"""#29 P3c: a unit's position is the Q16.16 ints ``x_q`` / ``y_q``, written
ONLY through ``Unit.set_pos_q`` (the float door ``Unit.set_pos`` quantizes
once, half away from zero); ``x`` / ``y`` are read-only float views
(``docs/residency_batching_design_2026-11-07.md`` §4.4, §4.4a, §7 P3c,
§12.1).
"""
from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import unit_schema as us  # noqa: E402
from simulation.unit import Unit  # noqa: E402

ONE = 1 << 16


def test_a_position_cannot_be_assigned_only_set_through_the_seam():
    """Property: ``x``, ``y``, ``x_q`` and ``y_q`` have no setter, so a write
    that bypasses ``set_pos_q`` raises at its own line; ``set_pos_q`` refuses
    a value that is not an integer. Breaks if a setter is added to any of
    the four (validated: an ``x`` setter makes the first ``raises`` fail) or
    ``set_pos_q`` accepts a float with a fraction."""
    u = Unit("u", 3.0, 4.0)
    for name in ("x", "y", "x_q", "y_q"):
        with pytest.raises(AttributeError):
            setattr(u, name, 5)
    with pytest.raises(ValueError):
        u.set_pos_q(3.5, 0)
    with pytest.raises(ValueError):
        u.set_pos_q(True, 0)
    u.set_pos_q(7 * ONE + 3, -2)
    assert (u.x_q, u.y_q) == (7 * ONE + 3, -2)
    assert u.x == (7 * ONE + 3) / 65536 and u.y == -2 / 65536


def test_the_float_door_rounds_half_away_from_zero_once():
    """Property: ``set_pos`` quantizes each coordinate once with the kit's
    rule (round half away from zero, ``unit_fixed.quantize_scalar``), keeps
    a value already on the grid exactly, and ``tile_x`` / ``tile_y`` are
    ``int(x)`` / ``int(y)`` (truncation) on negative positions too. Breaks
    if the door rounds half to even (validated: ``round(v * 65536)`` fails
    the 2.5 / 65536 case) or the tile index floors instead of truncating
    (validated: ``x_q >> 16`` fails the -0.5 case)."""
    u = Unit("u", 0.0, 0.0)
    u.set_pos(2.5 / ONE, -2.5 / ONE)
    assert (u.x_q, u.y_q) == (3, -3)
    u.set_pos(0.5 / ONE, -0.5 / ONE)
    assert (u.x_q, u.y_q) == (1, -1)
    u.set_pos(12.375, 4.0)
    assert (u.x, u.y) == (12.375, 4.0)
    for v in (-1.0, -0.9, -0.5, -1 / ONE, 0.0, 0.5, 1.0, 2.99, 7.0):
        u.set_pos(v, -v)
        assert u.tile_x == int(u.x) and u.tile_y == int(u.y), v


# The places allowed to name the private stores; every other write of a
# unit position in src/ goes through Unit.set_pos_q / set_pos.
_STORE_OK = {"simulation/unit.py", "simulation/unit_schema.py"}
# A non-unit object with its own x / y (a planted charge's tile).
_ATTR_OK = {("simulation/charges.py", "self.x = int(x)"),
            ("simulation/charges.py", "self.y = int(y)")}
_WRITE = re.compile(r"\.(x|y|x_q|y_q)\s*([-+*/]?=)(?!=)"
                    r"|\.(x|y)\s*,\s*[\w\[\]\.]+\.(x|y)\s*=(?!=)"
                    r"|setattr\([^,]+,\s*['\"](x|y|x_q|y_q)['\"]")
_STORE = re.compile(r"(?<!\w)_[xy]_q\b")


def _src_files():
    for p in sorted((ROOT / "src").rglob("*.py")):
        rel = p.relative_to(ROOT / "src").as_posix()
        if rel.startswith("renderer/"):
            continue                      # raylib vectors; draws, never units
        yield rel, p.read_text(encoding="utf-8")


def test_no_position_write_in_src_bypasses_the_seam():
    """Property: no file in ``src/`` (bar the renderer's raylib vectors)
    assigns ``.x`` / ``.y`` / ``.x_q`` / ``.y_q`` or names the private
    stores ``_x_q`` / ``_y_q`` outside ``unit.py`` / ``unit_schema.py``
    (the ``test_footprint_samples_builder.py`` scan precedent). Breaks if a
    writer comes back (validated: ``u.x = px`` in the WEGO path of
    ``simulation.py`` is named)."""
    bad = []
    for rel, text in _src_files():
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            if _WRITE.search(code) and (rel, code.strip()) not in _ATTR_OK:
                bad.append(f"{rel}:{n}: {line.strip()}")
            if _STORE.search(code) and rel not in _STORE_OK:
                bad.append(f"{rel}:{n}: {line.strip()}")
    assert not bad, "position writes outside Unit.set_pos_q:\n" + \
        "\n".join(bad)


def test_the_schema_hashes_the_position_as_its_ints():
    """Property: the record carries ``x_q`` / ``y_q`` as plain ints, always,
    and no float position key; the encoding round-trips. Breaks if the
    position columns are hashed in another encoding (validated: encoding
    ``x_q`` as ``x_q * 15258`` -- ~ the old ``round(x * 1e9)`` -- fails) or
    lose *always*."""
    u = Unit("u", 3.25, 4.0)
    rec = us.digest_record(u)
    assert rec["x_q"] == 3 * ONE + ONE // 4 and rec["y_q"] == 4 * ONE
    assert "x" not in rec and "y" not in rec
    row = us.encode_unit(u)
    v = Unit("v", 0.0, 0.0)
    us.decode_into(v, row)
    assert (v.x_q, v.y_q, v.x, v.y) == (u.x_q, u.y_q, 3.25, 4.0)


def test_every_position_of_a_match_is_its_q16_view():
    """Property, every tick of the ship_demo bots_v1 match (264 ticks, five
    conversions, pushes): every unit's ``x_q`` / ``y_q`` are ints, ``x`` /
    ``y`` equal ``x_q / 65536`` exactly, and ``tile_x`` / ``tile_y`` equal
    ``int(x)`` / ``int(y)``. Breaks if the store holds anything but an int
    (validated: ``set_pos_q`` storing ``x_q / 1`` fails the ``type`` check)
    or the view drifts from the int."""
    import level_loader
    from run_match import run_match
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = replace(level.match, teams=tuple(
        replace(t, controller="bots_v1") for t in level.match.teams))
    seen = {"ticks": 0, "moved": 0}
    first = {}

    def tick(sim):
        seen["ticks"] += 1
        for u in sim.units:
            assert type(u.x_q) is int and type(u.y_q) is int, u.name
            assert u.x == u.x_q / 65536 and u.y == u.y_q / 65536
            assert u.tile_x == int(u.x) and u.tile_y == int(u.y)
            p = first.setdefault(u.id, (u.x_q, u.y_q))
            if p != (u.x_q, u.y_q):
                seen["moved"] += 1
    run_match(level, spec, on_tick=tick)
    assert seen["ticks"] >= 200 and seen["moved"] > 100
