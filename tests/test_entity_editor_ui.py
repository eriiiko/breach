"""tools/entity_editor_ui.py — registry load/fallback + palette + inspector
(Arc C1: registry-driven palette + inspector, canon engine/16 §1).

Pins:
  - registry import success rewrites the last-good fallback file and
    returns ok=True (canon §1: "rewritten on every successful launch");
  - a simulated import failure falls back to reading the last-good file
    and reports ok=False + the error (editor design §3b failure mode);
  - both failing (no fallback readable either) is a loud RuntimeError —
    the one case the editor truly cannot start from;
  - palette generation: one entry per registered class, sorted, a
    deterministic chip colour + class-initial (the permanent no-icon
    fallback, editor doc §8);
  - the exact length_m -> tiles snap (editor doc §4), agreeing with
    simulation.entities.door's own quantizer for positive lengths;
  - kind-aware field formatting + inspector row generation, including
    which kinds are editable in C1 (str/str_list/roster/entity_ref stay
    display-only — no in-UI text input in v1, tags are C4's job).

Run:
    python -m pytest tests/test_entity_editor_ui.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import entity_editor_ui as eui  # noqa: E402


# ---------------------------------------------------------------------------
# Registry load + fallback (canon §1 / entity design §3b)
# ---------------------------------------------------------------------------

def test_load_registry_success_rewrites_last_good(tmp_path):
    export_path = tmp_path / "entity_registry.json"
    result = eui.load_registry(export_path=export_path,
                               fallback_path=export_path)
    assert result.ok and result.error is None
    assert "light" in result.payload["classes"]
    assert "door" in result.payload["classes"]
    assert export_path.is_file()
    on_disk = json.loads(export_path.read_text(encoding="utf-8"))
    # The exported file IS the payload the caller just got (content_hash
    # aside — registry_payload() itself doesn't carry it, export adds it).
    assert on_disk["classes"]["light"] == result.payload["classes"]["light"]


def test_load_registry_falls_back_on_import_failure(tmp_path):
    fallback = tmp_path / "last_good.json"
    fallback.write_text(json.dumps(
        {"schema_version": 1, "classes": {"light": {"fields": []}}}))

    def boom():
        raise ImportError("simulated half-written door.py")

    result = eui.load_registry(importer=boom, fallback_path=fallback)
    assert not result.ok
    assert "simulated half-written door.py" in result.error
    assert result.payload["classes"]["light"]["fields"] == []


def test_load_registry_raises_when_no_fallback_available(tmp_path):
    def boom():
        raise ImportError("broken")

    with pytest.raises(RuntimeError, match="broken"):
        eui.load_registry(importer=boom, fallback_path=tmp_path / "missing.json")


def test_load_registry_success_path_never_touches_fallback_reader(tmp_path):
    """A successful import must not even attempt to read the fallback —
    only the export path is written."""
    export_path = tmp_path / "reg.json"
    fallback_path = tmp_path / "does_not_exist.json"
    result = eui.load_registry(export_path=export_path,
                               fallback_path=fallback_path)
    assert result.ok
    assert not fallback_path.exists()


# ---------------------------------------------------------------------------
# Palette — one entry per registered class (editor design §3 pillar 1)
# ---------------------------------------------------------------------------

def test_palette_entries_sorted_with_chip_and_initial():
    payload = {"classes": {"door": {}, "light": {}, "breach_site": {}}}
    entries = eui.palette_entries(payload)
    assert [e.class_name for e in entries] == ["breach_site", "door", "light"]
    for e in entries:
        assert e.initial == e.class_name[0].upper()
        assert len(e.chip_rgb) == 3
        assert all(0 <= c <= 255 for c in e.chip_rgb)
    # Deterministic: rebuilding from the same payload gives the same chips.
    entries2 = eui.palette_entries(payload)
    assert [e.chip_rgb for e in entries] == [e.chip_rgb for e in entries2]


def test_palette_entries_from_real_registry_includes_canon_classes(tmp_path):
    result = eui.load_registry(export_path=tmp_path / "r.json",
                               fallback_path=tmp_path / "r.json")
    names = {e.class_name for e in eui.palette_entries(result.payload)}
    # The four classes the kickoff doc names explicitly (Arc A); the real
    # registry also carries Arc B's logic/sensor/actuator classes — the
    # schema has no "placeable" flag to filter on (INTANGIBLE means "no
    # tile", not "not an instance"), so every registered class appears.
    assert {"door", "light", "breach_site", "extraction_zone"} <= names


def test_palette_entries_empty_registry():
    assert eui.palette_entries({"classes": {}}) == []
    assert eui.palette_entries({}) == []


# ---------------------------------------------------------------------------
# quantize_length_m — the exact editor doc §4 snap rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length_m,tiles_per_m,want", [
    (1.0, 3, 3),      # default door: 1 m -> 3 tiles
    (0.5, 3, 2),      # 1.5 rounds HALF-UP -> 2 (never banker's round)
    (0.34, 3, 1),     # ~1.02 -> 1
    (0.333, 3, 1),    # the 1-tile door spelling
    (0.0, 3, 0),
    (2.0, 1, 2),
    (0.25, 4, 1),
])
def test_quantize_length_m_exact_ties(length_m, tiles_per_m, want):
    assert eui.quantize_length_m(length_m, tiles_per_m) == want


def test_quantize_length_m_rejects_non_positive_resolution():
    with pytest.raises(ValueError):
        eui.quantize_length_m(1.0, 0)
    with pytest.raises(ValueError):
        eui.quantize_length_m(1.0, -3)


def test_quantize_length_m_agrees_with_door_module():
    """The same exact rule, cross-checked against
    simulation.entities.door.quantize_span_tiles (base tile_size_m=0.333 ->
    tiles_per_m=3) for every positive length that module accepts."""
    from simulation.entities import door as door_mod
    for length_m in (1.0, 0.5, 0.34, 0.333, 2.5, 0.667):
        assert (eui.quantize_length_m(length_m, 3)
                == door_mod.quantize_span_tiles(length_m, 0.333))


# ---------------------------------------------------------------------------
# format_field_value — kind-aware display text
# ---------------------------------------------------------------------------

def test_format_field_value_every_kind():
    fmt = eui.format_field_value
    assert fmt(eui.KIND_LENGTH_M, 1.0) == "1 m -> 3 tiles"
    assert fmt(eui.KIND_LENGTH_M, 0.5) == "0.5 m -> 2 tiles"
    assert fmt(eui.KIND_COLOR_RGB, (255, 0, 0)) == "(255, 0, 0)"
    assert fmt(eui.KIND_STR_LIST, []) == "(none)"
    assert fmt(eui.KIND_STR_LIST, ["deck", "aft"]) == "deck, aft"
    assert fmt(eui.KIND_ROSTER, []) == "(empty)"
    assert (fmt(eui.KIND_ROSTER, [("marine", 3), ("zombie", 2)])
            == "marine x3; zombie x2")
    assert fmt(eui.KIND_BOOL, True) == "yes"
    assert fmt(eui.KIND_BOOL, False) == "no"
    assert fmt(eui.KIND_FLOAT_RENDER, 2.5) == "2.5"
    assert fmt(eui.KIND_ENTITY_REF, "") == "(unwired)"
    assert fmt(eui.KIND_ENTITY_REF, "door_1") == "door_1"
    assert fmt(eui.KIND_INT, 3) == "3"
    assert fmt(eui.KIND_ENUM, "closed") == "closed"


# ---------------------------------------------------------------------------
# inspector_rows — per-field rendering + editability
# ---------------------------------------------------------------------------

def _field(name, kind, default=None, minimum=None, maximum=None,
          choices=None):
    return {"name": name, "kind": kind, "default": default,
            "minimum": minimum, "maximum": maximum, "choices": choices}


def test_inspector_rows_authored_value_overrides_default():
    cls_payload = {"fields": [_field("x", eui.KIND_INT, default=0),
                              _field("orientation", eui.KIND_ENUM,
                                     default="h", choices=["h", "v"])]}
    rows = eui.inspector_rows(cls_payload, {"x": 5})
    assert rows[0].name == "x" and rows[0].value == 5 and rows[0].display == "5"
    assert rows[0].editable
    assert rows[1].value == "h"          # falls back to the default
    assert rows[1].choices == ("h", "v")
    assert rows[1].editable


def test_inspector_rows_length_m_uses_real_door_schema(tmp_path):
    result = eui.load_registry(export_path=tmp_path / "r.json",
                               fallback_path=tmp_path / "r.json")
    door_payload = result.payload["classes"]["door"]
    rows = eui.inspector_rows(door_payload, {})
    length_row = next(r for r in rows if r.name == "length_m")
    assert length_row.display == "1 m -> 3 tiles"     # the class default


def test_inspector_rows_display_only_kinds_are_not_editable():
    cls_payload = {"fields": [
        _field("id", eui.KIND_STR, default=None),
        _field("tags", eui.KIND_STR_LIST, default=[]),
        _field("roster", eui.KIND_ROSTER, default=[]),
        _field("watches", eui.KIND_ENTITY_REF, default=""),
    ]}
    rows = eui.inspector_rows(cls_payload, {})
    assert not any(r.editable for r in rows)


def test_inspector_rows_numeric_and_enum_kinds_are_editable():
    cls_payload = {"fields": [
        _field("hp", eui.KIND_Q16, default=0),
        _field("count", eui.KIND_INT, default=1),
        _field("kind", eui.KIND_ENUM, default="static",
              choices=["static", "beacon"]),
        _field("intensity", eui.KIND_FLOAT_RENDER, default=1.0),
        _field("color", eui.KIND_COLOR_RGB, default=(255, 255, 255)),
    ]}
    rows = eui.inspector_rows(cls_payload, {})
    assert all(r.editable for r in rows)


# ---------------------------------------------------------------------------
# Generic place-one (Arc C3): required_field_names / default_instance_fields
# ---------------------------------------------------------------------------

def test_required_field_names_is_the_default_none_fields():
    cls_payload = {"fields": [
        _field("x", eui.KIND_INT, default=None),
        _field("y", eui.KIND_INT, default=None),
        _field("period", eui.KIND_INT, default=1),
    ]}
    assert eui.required_field_names(cls_payload) == ("x", "y")


def test_required_field_names_empty_for_a_pure_logic_node():
    """decider/gate_*/filter/airlock_controller: no x/y, nothing required —
    the generic place-one authors nothing but id/class (registry defaults
    cover every field)."""
    cls_payload = {"fields": [
        _field("comparator", eui.KIND_ENUM, default="gt",
              choices=["gt", "lt"]),
        _field("threshold", eui.KIND_Q16, default=0),
    ]}
    assert eui.required_field_names(cls_payload) == ()


def test_default_instance_fields_overrides_xy_only_when_declared():
    cls_payload = {"fields": [
        _field("x", eui.KIND_INT, default=None),
        _field("y", eui.KIND_INT, default=None),
        _field("period", eui.KIND_INT, default=1),
    ]}
    fields = eui.default_instance_fields(cls_payload, x=4, y=7)
    assert fields == {"x": 4, "y": 7, "period": 1}


def test_default_instance_fields_ignores_xy_when_class_has_none():
    """A pure logic node (no x/y field) ignores the placement tile —
    positionless classes still get a template, per the C3 kickoff note."""
    cls_payload = {"fields": [_field("threshold", eui.KIND_Q16, default=0)]}
    fields = eui.default_instance_fields(cls_payload, x=4, y=7)
    assert fields == {"threshold": 0}


def test_default_instance_fields_leaves_unfillable_required_at_none():
    """A zone's zone_id (required, no x/y-style placement fills it) stays
    None in the template — callers must refuse these classes rather than
    author an invalid instance (see required_field_names -> the "unfillable"
    guard in map_editor.py's ENTITY mode)."""
    cls_payload = {"fields": [_field("zone_id", eui.KIND_INT, default=None),
                              _field("faction", eui.KIND_INT, default=0)]}
    fields = eui.default_instance_fields(cls_payload)
    assert fields == {"zone_id": None, "faction": 0}
    assert eui.required_field_names(cls_payload) == ("zone_id",)


# ---------------------------------------------------------------------------
# icon_png_path (Arc C8) — the palette's PNG-vs-chip decision
# ---------------------------------------------------------------------------

def test_icon_png_path_hits_when_file_exists(tmp_path):
    (tmp_path / "door.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    found = eui.icon_png_path("door", icons_dir=tmp_path)
    assert found == tmp_path / "door.png"


def test_icon_png_path_none_when_missing(tmp_path):
    assert eui.icon_png_path("nope", icons_dir=tmp_path) is None


def test_icon_png_path_default_dir_matches_committed_icons():
    """Sanity: at least one real committed icon (door) resolves through the
    default ICONS_DIR without an explicit override."""
    found = eui.icon_png_path("door")
    assert found is not None
    assert found.is_file()
    assert found.name == "door.png"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
