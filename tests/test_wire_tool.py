"""tools/wire_tool.py — the two-click wire tool + LOGIC overlay + tag badges
(Arc C6, editor doc §8).

Pins:
  - positionless_layout: a zone instance anchors at its painted tiles'
    centroid; an unpainted zone (or a true logic node) falls back to a
    deterministic id-sorted strip past the grid's right edge, stable
    regardless of `entities` list order;
  - anchor_tile / hit_test_with_layout: every selectable instance (incl. a
    positionless one, via the layout) resolves to ONE clickable tile;
  - signal/input vocabulary: primary_signal defaults to the first non-
    `alive` signal, else `alive` itself (button/terminal/light);
  - validate_wire REUSES level_loader._parse_wires (not re-derived): a bad
    signal/input name hard-errors, a unit reference hard-errors (even when
    the id also happens to resolve to a live entity), a dangling id is
    refused directly (liveness pre-check — the interactive tool can only
    ever click a live id);
  - commit_add_wire / commit_remove_wire: ONE `CollectionOp("wires")`
    transaction each, undo restores; an invalid add opens NO transaction;
  - the pending-wire state machine (begin/cycle-signal/set-target-id/
    set-target-tag/cycle-input): defaults + cycling + tag-target input
    intersection;
  - wires_touching_selection / wire_endpoints_for_draw / hit_test_wire: the
    LOGIC overlay's selection filter, line-vs-badge split (a tag wire is
    ONE badge, never fanned), and wire hit-testing;
  - a create -> save -> reload wire round-trip through level_lib.

Run:
    python -m pytest tests/test_wire_tool.py -q
"""
from __future__ import annotations

import struct
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import wire_tool as wt  # noqa: E402
import door_entity_port as dep  # noqa: E402
import light_entity_port as lep  # noqa: E402
import entity_editor_ui  # noqa: E402
import undo_log  # noqa: E402
from level_loader import EntityInstance, SpawnEntry, WireSpec  # noqa: E402

TILE_SIZE_M = 0.333          # the shipped level default -> tiles_per_m == 3
GRID_SHAPE = (10, 20)        # (h, w)


def _door(id_, x, y, orientation="h", n_tiles=3, initial_state="closed",
         tags=()):
    length_m = dep.length_m_for_tiles(n_tiles, TILE_SIZE_M)
    inst = dep.build_door_instance(x, y, orientation, length_m,
                                   initial_state, id_)
    return replace(inst, tags=tuple(tags)) if tags else inst


def _light(id_, x, y, tags=()):
    return lep.EditableLight(x=x, y=y, color=(1.0, 0.0, 0.0), id=id_,
                             tags=tuple(tags))


def _generic(id_, cls_name, x, y, tags=()):
    return EntityInstance(id=id_, class_name=cls_name, ordinal=0,
                          tags=tuple(tags), fields={"x": x, "y": y},
                          authored_keys=("x", "y"))


def _positionless(id_, cls_name="decider", tags=()):
    return EntityInstance(id=id_, class_name=cls_name, ordinal=0,
                          tags=tuple(tags), fields={}, authored_keys=())


def _zone_entity(id_, zone_id, cls_name="breach_site"):
    fields = {"zone_id": zone_id, "faction": 0}
    if cls_name == "breach_site":
        fields["roster"] = []
    return EntityInstance(id=id_, class_name=cls_name, ordinal=0, tags=(),
                          fields=fields, authored_keys=("zone_id",))


@pytest.fixture(scope="module")
def registry_payload():
    result = entity_editor_ui.load_registry()
    assert result.ok, result.error
    return result.payload


# ---------------------------------------------------------------------------
# positionless_layout
# ---------------------------------------------------------------------------

def test_positionless_layout_zone_anchors_at_centroid():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    zones[1:4, 4:7] = 3          # x in {4,5,6}, y in {1,2,3} -> centroid (5,2)
    e = _zone_entity("site_1", 3)
    layout = wt.positionless_layout([e], zones=zones, grid_shape=GRID_SHAPE)
    assert layout["site_1"] == (5, 2)


def test_positionless_layout_unpainted_zone_falls_back_to_strip():
    zones = np.zeros(GRID_SHAPE, dtype=np.uint8)
    e = _zone_entity("site_1", 3)
    layout = wt.positionless_layout([e], zones=zones, grid_shape=GRID_SHAPE)
    assert layout["site_1"] == (GRID_SHAPE[1] + wt.LAYOUT_MARGIN, 0)


def test_positionless_layout_no_zones_grid_falls_back_to_strip():
    e = _zone_entity("site_1", 3)
    layout = wt.positionless_layout([e], zones=None, grid_shape=GRID_SHAPE)
    assert layout["site_1"] == (GRID_SHAPE[1] + wt.LAYOUT_MARGIN, 0)


def test_positionless_layout_logic_nodes_sorted_by_id_deterministic():
    a = _positionless("dec_b", "decider")
    b = _positionless("dec_a", "decider")
    layout_ab = wt.positionless_layout([a, b], grid_shape=GRID_SHAPE)
    layout_ba = wt.positionless_layout([b, a], grid_shape=GRID_SHAPE)
    assert layout_ab == layout_ba
    base_x = GRID_SHAPE[1] + wt.LAYOUT_MARGIN
    assert layout_ab["dec_a"] == (base_x, 0)
    assert layout_ab["dec_b"] == (base_x + wt.LAYOUT_SPACING, 0)


def test_positionless_layout_wraps_columns():
    ids = [_positionless(f"n{i}", "decider") for i in range(wt.LAYOUT_COLS + 1)]
    layout = wt.positionless_layout(ids, grid_shape=GRID_SHAPE)
    # LAYOUT_COLS ids fill row 0 (one per column); the (COLS+1)-th wraps to
    # row 1 — regardless of WHICH id (sorted-by-id) lands where.
    rows = sorted(pos[1] for pos in layout.values())
    assert rows == [0] * wt.LAYOUT_COLS + [wt.LAYOUT_SPACING]


def test_positionless_layout_excludes_positioned_entities():
    e = _generic("sensor_1", "pressure", 5, 6)
    assert wt.positionless_layout([e], grid_shape=GRID_SHAPE) == {}


def test_positionless_layout_no_grid_shape_bases_at_zero():
    e = _positionless("dec_1")
    layout = wt.positionless_layout([e])
    assert layout["dec_1"] == (0, 0)


# ---------------------------------------------------------------------------
# anchor_tile / class_name_of / class_payload_of / hit_test_with_layout
# ---------------------------------------------------------------------------

def test_anchor_tile_door_is_its_span_center():
    d = _door("door_1", 2, 3, n_tiles=3)
    pos = wt.anchor_tile("door_1", [], [d], TILE_SIZE_M, {})
    assert pos == (3, 3)


def test_anchor_tile_light_is_its_tile():
    l = _light("light_1", 2.9, 3.1)
    pos = wt.anchor_tile("light_1", [l], [], TILE_SIZE_M, {})
    assert pos == (2, 3)


def test_anchor_tile_positionless_uses_layout():
    e = _positionless("dec_1")
    pos = wt.anchor_tile("dec_1", [], [e], TILE_SIZE_M, {"dec_1": (9, 4)})
    assert pos == (9, 4)


def test_anchor_tile_unknown_id_is_none():
    assert wt.anchor_tile("nope", [], [], TILE_SIZE_M, {}) is None


def test_anchor_tile_positionless_absent_from_layout_is_none():
    e = _positionless("dec_1")
    assert wt.anchor_tile("dec_1", [], [e], TILE_SIZE_M, {}) is None


def test_hit_test_with_layout_hits_positioned_entity():
    e = _generic("sensor_1", "pressure", 5, 6)
    assert wt.hit_test_with_layout(5, 6, [], [e], TILE_SIZE_M, {}) == "sensor_1"


def test_hit_test_with_layout_hits_positionless_via_layout():
    e = _positionless("dec_1")
    layout = {"dec_1": (9, 4)}
    assert wt.hit_test_with_layout(9, 4, [], [e], TILE_SIZE_M, layout) == "dec_1"


def test_hit_test_with_layout_miss_is_none():
    assert wt.hit_test_with_layout(0, 0, [], [], TILE_SIZE_M, {}) is None


def test_class_payload_of_resolves_class(registry_payload):
    e = _generic("sensor_1", "pressure", 5, 6)
    cls_name, payload = wt.class_payload_of(
        "sensor_1", [], [e], registry_payload)
    assert cls_name == "pressure"
    assert payload is registry_payload["classes"]["pressure"]


def test_class_payload_of_unknown_id_is_none_none(registry_payload):
    assert wt.class_payload_of("nope", [], [], registry_payload) == (None, None)


# ---------------------------------------------------------------------------
# Signal / input vocabulary
# ---------------------------------------------------------------------------

def test_primary_signal_door_is_is_open(registry_payload):
    assert wt.primary_signal(registry_payload["classes"]["door"]) == "is_open"


def test_primary_signal_button_is_alive_only(registry_payload):
    assert wt.primary_signal(registry_payload["classes"]["button"]) == "alive"


def test_signal_names_includes_alive_first(registry_payload):
    assert wt.signal_names(registry_payload["classes"]["door"])[0] == "alive"


def test_input_names_door(registry_payload):
    assert wt.input_names(registry_payload["classes"]["door"]) == (
        "open", "close")


def test_tag_input_names_intersection(registry_payload):
    d1 = _door("door_1", 2, 3, tags=("airlock",))
    d2 = _door("door_2", 8, 3, tags=("airlock",))
    inputs = wt.tag_input_names("airlock", [], [d1, d2], registry_payload)
    assert inputs == ("close", "open")


def test_tag_input_names_no_members_is_empty(registry_payload):
    assert wt.tag_input_names("nope", [], [], registry_payload) == ()


def test_tag_input_names_no_common_input_is_empty(registry_payload):
    door = _door("door_1", 2, 3, tags=("grp",))
    sensor = _generic("sensor_1", "pressure", 5, 6, tags=("grp",))
    inputs = wt.tag_input_names("grp", [], [door, sensor], registry_payload)
    assert inputs == ()          # pressure declares no inputs at all


def test_all_tags_sorted_unique():
    d1 = _door("door_1", 2, 3, tags=("b", "a"))
    d2 = _door("door_2", 8, 3, tags=("a",))
    assert wt.all_tags([], [d1, d2]) == ("a", "b")


# ---------------------------------------------------------------------------
# validate_wire — reused loader validation
# ---------------------------------------------------------------------------

def test_validate_wire_accepts_valid_wire():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    door = _door("door_1", 2, 3)
    ok, reason = wt.validate_wire("sensor_1", "value", "door_1", "close",
                                  [], [sensor, door], [])
    assert ok and reason is None


def test_validate_wire_refuses_unknown_signal_name():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    door = _door("door_1", 2, 3)
    ok, reason = wt.validate_wire("sensor_1", "bogus_signal", "door_1",
                                  "close", [], [sensor, door], [])
    assert not ok
    assert "signal" in reason


def test_validate_wire_refuses_unknown_input_name():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    door = _door("door_1", 2, 3)
    ok, reason = wt.validate_wire("sensor_1", "value", "door_1",
                                  "bogus_input", [], [sensor, door], [])
    assert not ok
    assert "input" in reason


def test_validate_wire_refuses_dangling_from_id():
    door = _door("door_1", 2, 3)
    ok, reason = wt.validate_wire("nope", "alive", "door_1", "close",
                                  [], [door], [])
    assert not ok
    assert "live entity id" in reason


def test_validate_wire_refuses_dangling_to_id():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    ok, reason = wt.validate_wire("sensor_1", "value", "nope", "close",
                                  [], [sensor], [])
    assert not ok
    assert "live entity id" in reason


def test_validate_wire_refuses_unit_reference():
    """A [[spawn]] unit's name coincides with a live entity's id — the
    loader's unit-rejection rule fires REGARDLESS (canon §3e), proving the
    check is truly reused, not shadowed by the liveness pre-check above."""
    node = _positionless("unit_1", "decider")
    door = _door("door_1", 2, 3)
    spawn = SpawnEntry(name="unit_1", team=0, x=0.0, y=0.0)
    ok, reason = wt.validate_wire("unit_1", "alive", "door_1", "close",
                                  [], [node, door], [spawn])
    assert not ok
    assert "not entities" in reason.lower() or "unit" in reason.lower()


def test_validate_wire_accepts_tag_target():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    d1 = _door("door_1", 2, 3, tags=("airlock",))
    d2 = _door("door_2", 8, 3, tags=("airlock",))
    ok, reason = wt.validate_wire("sensor_1", "value", "tag:airlock",
                                  "close", [], [sensor, d1, d2], [])
    assert ok and reason is None


def test_validate_wire_accepts_light_as_source():
    light = _light("light_1", 2.0, 3.0)
    door = _door("door_1", 2, 3)
    ok, reason = wt.validate_wire("light_1", "alive", "door_1", "close",
                                  [light], [door], [])
    assert ok and reason is None


# ---------------------------------------------------------------------------
# commit_add_wire / commit_remove_wire
# ---------------------------------------------------------------------------

def _log_for(wires):
    ctx = undo_log.UndoContext(collections={"wires": wires})
    return undo_log.TransactionLog(ctx)


def test_commit_add_wire_commits_one_transaction_and_undoes():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    door = _door("door_1", 2, 3)
    wires = []
    log = _log_for(wires)
    ok, result = wt.commit_add_wire(log, wires, [], [sensor, door], [],
                                    "sensor_1", "value", "door_1", "close")
    assert ok and result is not None
    assert [(w.from_, w.to) for w in wires] == [
        ("sensor_1.value", "door_1.close")]
    assert log.undo_count == 1
    log.undo()
    assert wires == []


def test_commit_add_wire_refuses_invalid_opens_no_transaction():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    wires = []
    log = _log_for(wires)
    ok, reason = wt.commit_add_wire(log, wires, [], [sensor], [],
                                    "sensor_1", "bogus", "sensor_1", "x")
    assert not ok and isinstance(reason, str)
    assert wires == [] and log.undo_count == 0


def test_commit_remove_wire_one_transaction_and_undoes():
    w = WireSpec(from_="sensor_1.value", to="door_1.close")
    wires = [w]
    log = _log_for(wires)
    txn = wt.commit_remove_wire(log, wires, w)
    assert txn is not None and wires == []
    log.undo()
    assert wires == [w]


def test_commit_remove_wire_missing_is_noop():
    wires = []
    log = _log_for(wires)
    w = WireSpec(from_="a.b", to="c.d")
    assert wt.commit_remove_wire(log, wires, w) is None
    assert log.undo_count == 0


# ---------------------------------------------------------------------------
# Pending-wire state machine
# ---------------------------------------------------------------------------

def test_begin_pending_wire_defaults_to_primary_signal(registry_payload):
    p = wt.begin_pending_wire("door_1", registry_payload["classes"]["door"])
    assert p["signals"][p["signal_idx"]] == "is_open"
    assert p["to_spec"] is None


def test_begin_pending_wire_alive_only_class_defaults_alive(registry_payload):
    p = wt.begin_pending_wire("btn_1", registry_payload["classes"]["button"])
    assert p["signals"][p["signal_idx"]] == "alive"


def test_cycle_pending_signal_wraps(registry_payload):
    p = wt.begin_pending_wire("door_1", registry_payload["classes"]["door"])
    p2 = wt.cycle_pending_signal(p, 1)
    assert p2["signals"][p2["signal_idx"]] == "alive"
    p3 = wt.cycle_pending_signal(p2, 1)
    assert p3["signals"][p3["signal_idx"]] == "is_open"


def test_cycle_pending_signal_noop_once_target_set(registry_payload):
    p = wt.begin_pending_wire("door_1", registry_payload["classes"]["door"])
    p, _reason = wt.set_pending_target_id(
        p, "door_2", registry_payload["classes"]["door"])
    before = p["signal_idx"]
    p2 = wt.cycle_pending_signal(p, 1)
    assert p2["signal_idx"] == before


def test_set_pending_target_id_locks_inputs(registry_payload):
    p = wt.begin_pending_wire("sensor_1", registry_payload["classes"]["pressure"])
    p2, reason = wt.set_pending_target_id(
        p, "door_1", registry_payload["classes"]["door"])
    assert reason is None
    assert p2["inputs"] == ("open", "close")
    assert p2["to_spec"] == "door_1"
    assert p2["to_is_tag"] is False


def test_set_pending_target_id_refuses_no_input_class(registry_payload):
    p = wt.begin_pending_wire("door_1", registry_payload["classes"]["door"])
    p2, reason = wt.set_pending_target_id(
        p, "sensor_1", registry_payload["classes"]["pressure"])
    assert p2 is None
    assert "no inputs" in reason


def test_cycle_pending_input_wraps(registry_payload):
    p = wt.begin_pending_wire("sensor_1", registry_payload["classes"]["pressure"])
    p, _r = wt.set_pending_target_id(
        p, "door_1", registry_payload["classes"]["door"])
    p2 = wt.cycle_pending_input(p, 1)
    assert p2["inputs"][p2["input_idx"]] == "close"
    p3 = wt.cycle_pending_input(p2, 1)
    assert p3["inputs"][p3["input_idx"]] == "open"


def test_pending_ready_to_commit_and_strings(registry_payload):
    p = wt.begin_pending_wire("sensor_1", registry_payload["classes"]["pressure"])
    assert not wt.pending_ready_to_commit(p)
    p, _r = wt.set_pending_target_id(
        p, "door_1", registry_payload["classes"]["door"])
    assert wt.pending_ready_to_commit(p)
    from_str, to_str = wt.pending_wire_strings(p)
    assert from_str == "sensor_1.value"
    assert to_str == "door_1.open"


def test_set_pending_target_tag_locks_common_inputs(registry_payload):
    p = wt.begin_pending_wire("sensor_1", registry_payload["classes"]["pressure"])
    d1 = _door("door_1", 2, 3, tags=("airlock",))
    d2 = _door("door_2", 8, 3, tags=("airlock",))
    p2, reason = wt.set_pending_target_tag(
        p, "airlock", [], [d1, d2], registry_payload)
    assert reason is None
    assert p2["to_spec"] == "tag:airlock"
    assert p2["to_is_tag"] is True
    assert p2["inputs"] == ("close", "open")


def test_set_pending_target_tag_refuses_no_members(registry_payload):
    p = wt.begin_pending_wire("sensor_1", registry_payload["classes"]["pressure"])
    p2, reason = wt.set_pending_target_tag(p, "ghost", [], [], registry_payload)
    assert p2 is None
    assert "no members" in reason


# ---------------------------------------------------------------------------
# LOGIC overlay: selection filter / line-badge split / wire hit-test
# ---------------------------------------------------------------------------

def test_wires_touching_selection_from_and_to():
    w1 = WireSpec("a.alive", "b.open")
    w2 = WireSpec("c.alive", "d.open")
    w3 = WireSpec("e.alive", "tag:grp.open")
    assert wt.wires_touching_selection([w1, w2, w3], {"b"}) == [w1]


def test_wires_touching_selection_tag_target_never_touched():
    w = WireSpec("a.alive", "tag:grp.open")
    assert wt.wires_touching_selection([w], {"grp"}) == []


def test_wire_endpoints_for_draw_splits_lines_and_badges():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    door = _door("door_1", 2, 3)              # span center (3, 3)
    node = _positionless("dec_1")
    layout = {"dec_1": (9, 4)}
    w_line = WireSpec("sensor_1.value", "door_1.close")
    w_badge = WireSpec("dec_1.out", "tag:airlock.open")
    lines, badges = wt.wire_endpoints_for_draw(
        [w_line, w_badge], [], [sensor, door, node], TILE_SIZE_M, layout)
    assert lines == [(w_line, (5, 6), (3, 3))]
    assert badges == [(w_badge, (9, 4), "airlock", "open")]


def test_wire_endpoints_for_draw_skips_dangling_endpoint():
    sensor = _generic("sensor_1", "pressure", 5, 6)
    w = WireSpec("sensor_1.value", "ghost.close")
    lines, badges = wt.wire_endpoints_for_draw(
        [w], [], [sensor], TILE_SIZE_M, {})
    assert lines == [] and badges == []


def test_hit_test_wire_hits_line_near_midpoint():
    w = WireSpec("a.alive", "b.open")
    lines = [(w, (2, 2), (6, 2))]
    assert wt.hit_test_wire(4, 2, lines, []) is w


def test_hit_test_wire_hits_badge():
    w = WireSpec("a.alive", "tag:grp.open")
    badges = [(w, (5, 5), "grp", "open")]
    assert wt.hit_test_wire(5, 5, [], badges) is w


def test_hit_test_wire_miss_returns_none():
    w = WireSpec("a.alive", "b.open")
    lines = [(w, (2, 2), (6, 2))]
    assert wt.hit_test_wire(20, 20, lines, []) is None


def test_hit_test_wire_picks_nearest():
    w_far = WireSpec("a.alive", "b.open")
    w_near = WireSpec("c.alive", "d.open")
    lines = [(w_far, (0, 0), (0, 0)), (w_near, (4, 4), (4, 4))]
    assert wt.hit_test_wire(4, 4, lines, []) is w_near


# ---------------------------------------------------------------------------
# Integration: level_lib round-trip — the gate's own explicit requirement.
# ---------------------------------------------------------------------------

def _write_png(path: Path, w: int = 8, h: int = 6) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * w for _ in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                     + chunk(b"IDAT", zlib.compress(raw))
                     + chunk(b"IEND", b""))


PREFIX = ('version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n'
          'tile_size_m = 0.333\ndiffuse = "diffuse.png"\n\n')
SUFFIX = ('[art.bare]\ndiffuse = "diffuse.png"\n\n'
          '[bake]\ntileset = "x"\npx_per_tile = 8\nseed = 0\n')
DOOR_A = ('[[entity]]\nid = "door_1"\nclass = "door"\n'
         'x = 0\ny = 3\norientation = "h"\nlength_m = 1.0\n'
         'initial_state = "closed"\n\n')
SENSOR_A = ('[[entity]]\nid = "sensor_1"\nclass = "pressure"\n'
           'x = 6\ny = 3\n\n')


def _mini_level(tmp_path: Path, body: str = "", name: str = "mini") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join("0" for _ in range(10)) for _ in range(8)) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(PREFIX + body + SUFFIX,
                                  encoding="utf-8", newline="\n")
    return d


def test_create_save_reload_wire_round_trip(tmp_path):
    import level_lib
    d = _mini_level(tmp_path, DOOR_A + SENSOR_A)
    handle = level_lib.open_level(str(d))
    lvl = handle.data
    entities = [e for e in lvl.entities if e.class_name != lep.LIGHT_CLASS]
    lights = lep.initial_editable_lights(lvl)
    wires = list(lvl.wire_specs)
    assert wires == []

    log = _log_for(wires)
    ok, txn = wt.commit_add_wire(log, wires, lights, entities, lvl.spawns,
                                 "sensor_1", "value", "door_1", "close")
    assert ok and txn is not None
    assert [(w.from_, w.to) for w in wires] == [
        ("sensor_1.value", "door_1.close")]

    handle.save({"wire": lambda nl: level_lib.format_wire_lines(wires, nl)})
    reloaded = level_lib.open_level(str(d))
    assert [(w.from_, w.to) for w in reloaded.data.wire_specs] == [
        ("sensor_1.value", "door_1.close")]

    # undo after save still round-trips the IN-MEMORY collection correctly
    # (a save doesn't freeze the log — later gates rely on this staying true).
    log.undo()
    assert wires == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
