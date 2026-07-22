"""tools/light_entity_port.py — the LIGHT port (Arc C1, Amendment A1).

LIGHT is a registry entity, so it ports onto `[[entity]]` and the bespoke
level_lib.write_lights path is DELETED — with PARITY (escalation trigger 5:
same authored result, via entities). Pins:

  - which family a level's lights live in is decided ONCE at load and never
    forced to migrate (editor doc §6 / Erik ruling 2 / canon §2);
  - id assignment on load: entity-form levels keep their authored ids,
    legacy/light-free levels mint fresh ``light_N`` ids;
  - the editor's NEW authoring path (EditableLight -> EntityInstance ->
    level_lib.format_entity_lines -> reload) reproduces field-for-field the
    SAME LightEntry a legacy [[light]] authoring of the identical values
    would — the parity proof this port is on the hook for;
  - merge_light_entities: edits/deletes replace in place, new lights
    append, unrelated (non-light) entities are untouched and keep their
    file position.

Run:
    python -m pytest tests/test_light_entity_port.py -q
"""
from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_lib import format_entity_lines, format_light_lines  # noqa: E402
from level_loader import EntityInstance, LightEntry  # noqa: E402

import light_entity_port as lep  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic level folders (never the repo's levels/)
# ---------------------------------------------------------------------------

def _write_png(path: Path, w: int = 8, h: int = 6) -> None:
    """Smallest valid RGB PNG (pure stdlib) — the loader reads its IHDR."""
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


def _mini_level(tmp_path: Path, body: str = "", name: str = "mini") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join("0" for _ in range(8)) for _ in range(6)) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(PREFIX + body + SUFFIX,
                                  encoding="utf-8", newline="\n")
    return d


def _load(d: Path):
    return level_loader.load(str(d))


LEGACY_LIGHTS = ('[[light]]\npos = [2.5, 3.5]\ncolor = [255, 0, 0]\n'
                 'intensity = 2.0\nrange = 8.0\nkind = "beacon"\n'
                 'period_s = 1.5\nbeam_deg = 45.0\nphase = 0.5\n\n'
                 '[[light]]\npos = [1.0, 4.0]\ncolor = [10, 20, 30]\n\n')
ENTITY_LIGHTS = ('[[entity]]\nid = "lamp_1"\nclass = "light"\n'
                 'x = 2.5\ny = 3.5\ncolor = [255, 0, 0]\n'
                 'intensity = 2.0\nrange = 8.0\nkind = "beacon"\n'
                 'period_s = 1.5\nbeam_deg = 45.0\nphase = 0.5\n\n'
                 '[[entity]]\nid = "lamp_2"\nclass = "light"\n'
                 'x = 1.0\ny = 4.0\ncolor = [10, 20, 30]\n\n')
DOOR_ENTITY = ('[[entity]]\nid = "door_1"\nclass = "door"\n'
              'x = 0\ny = 3\norientation = "h"\n\n')


# ---------------------------------------------------------------------------
# light_form — which family a level's lights live in, decided at load
# ---------------------------------------------------------------------------

def test_light_form_legacy_when_raw_light_blocks_present(tmp_path):
    lvl = _load(_mini_level(tmp_path, LEGACY_LIGHTS))
    assert lep.light_form(lvl.raw_toml) == "legacy"


def test_light_form_entity_when_entity_lights_present(tmp_path):
    lvl = _load(_mini_level(tmp_path, ENTITY_LIGHTS))
    assert lep.light_form(lvl.raw_toml) == "entity"


def test_light_form_entity_when_level_has_no_lights_at_all(tmp_path):
    """The bespoke [[light]] WRITER is deleted (Amendment A1): a light-free
    level authors its first light the modern way — not a migration, since
    nothing legacy exists to convert."""
    lvl = _load(_mini_level(tmp_path))
    assert lep.light_form(lvl.raw_toml) == "entity"


def test_light_form_entity_with_coexisting_nonlight_entity(tmp_path):
    """The mixed-form rule is alias-scoped (level_loader): legacy [[light]]
    can coexist with a non-light [[entity]] (e.g. a door). light_form only
    cares about raw [[light]] block presence."""
    lvl = _load(_mini_level(tmp_path, LEGACY_LIGHTS + DOOR_ENTITY))
    assert lep.light_form(lvl.raw_toml) == "legacy"


# ---------------------------------------------------------------------------
# initial_editable_lights — id assignment at load
# ---------------------------------------------------------------------------

def test_initial_editable_lights_preserves_entity_ids(tmp_path):
    lvl = _load(_mini_level(tmp_path, ENTITY_LIGHTS))
    lights = lep.initial_editable_lights(lvl)
    assert [l.id for l in lights] == ["lamp_1", "lamp_2"]
    assert (lights[0].x, lights[0].y) == (2.5, 3.5)
    assert lights[0].kind == "beacon"


def test_initial_editable_lights_mints_fresh_ids_for_legacy(tmp_path):
    lvl = _load(_mini_level(tmp_path, LEGACY_LIGHTS))
    lights = lep.initial_editable_lights(lvl)
    assert [l.id for l in lights] == ["light_1", "light_2"]


def test_initial_editable_lights_empty_for_light_free_level(tmp_path):
    lvl = _load(_mini_level(tmp_path))
    assert lep.initial_editable_lights(lvl) == []


def test_unique_light_id_skips_existing():
    assert lep.unique_light_id(set()) == "light_1"
    assert lep.unique_light_id({"light_1"}) == "light_2"
    assert lep.unique_light_id({"light_1", "light_2", "light_4"}) == "light_3"


# ---------------------------------------------------------------------------
# THE parity proof: the editor's new authoring path reproduces, field for
# field, what legacy [[light]] authoring of the identical values would have
# (escalation trigger 5 — a bespoke-path DELETE must reach parity first).
# ---------------------------------------------------------------------------

def _round_trip_via_legacy(tmp_path, l: LightEntry, name: str):
    d = _mini_level(tmp_path, name=name)
    format_light_lines([l])   # sanity: formats without error
    from level_lib import write_managed_blocks
    write_managed_blocks(d / "level.toml",
                        {"light": lambda nl: format_light_lines([l], nl)})
    return _load(d).lights[0]


def _round_trip_via_entity_port(tmp_path, l: LightEntry, id_: str, name: str):
    d = _mini_level(tmp_path, name=name)
    editable = lep.to_editable(l, id_)
    entities = lep.merge_light_entities([], [editable])
    from level_lib import write_managed_blocks
    write_managed_blocks(d / "level.toml",
                        {"entity": lambda nl: format_entity_lines(entities, nl)})
    lvl = _load(d)
    assert [e.id for e in lvl.entities] == [id_]
    return lvl.lights[0]


@pytest.mark.parametrize("kind,extra", [
    ("static", {}),
    ("beacon", dict(period_s=1.5, beam_deg=45.0, phase=0.5)),
])
def test_editor_authored_light_matches_legacy_authoring(tmp_path, kind, extra):
    l = LightEntry(x=2.5, y=3.5, color=(1.0, 0.0, 0.0), intensity=2.0,
                   range=8.0, kind=kind, **extra)
    via_legacy = _round_trip_via_legacy(tmp_path, l, "leg")
    via_entity = _round_trip_via_entity_port(tmp_path, l, "lamp_1", "ent")
    assert via_entity == via_legacy


def test_light_authored_keys_omits_beacon_params_for_static():
    assert lep.light_authored_keys("static") == (
        "x", "y", "color", "intensity", "range", "kind")


def test_light_authored_keys_includes_beacon_params_for_beacon():
    assert lep.light_authored_keys("beacon") == (
        "x", "y", "color", "intensity", "range", "kind",
        "period_s", "beam_deg", "phase")


def test_light_to_entity_instance_fields_effective_values():
    l = lep.EditableLight(x=1.0, y=2.0, color=(1.0, 0.0, 0.0), id="lamp_1")
    inst = lep.light_to_entity_instance(l, 0)
    assert inst.id == "lamp_1" and inst.class_name == "light"
    assert inst.fields["x"] == 1.0 and inst.fields["color"] == [255, 0, 0]
    assert inst.authored_keys == ("x", "y", "color", "intensity", "range",
                                  "kind")


# ---------------------------------------------------------------------------
# merge_light_entities — edit/delete in place, new lights append, other
# entities keep their file position untouched
# ---------------------------------------------------------------------------

def test_merge_light_entities_edits_in_place_keeps_order():
    door = EntityInstance(id="door_1", class_name="door", ordinal=0,
                          fields={"x": 0, "y": 0, "orientation": "h",
                                  "length_m": 1.0, "initial_state": "closed"},
                          authored_keys=("x", "y"))
    original = [
        EntityInstance(id="lamp_1", class_name="light", ordinal=1,
                      fields=lep.light_entry_to_fields(
                          LightEntry(x=1.0, y=1.0, color=(1.0, 0.0, 0.0))),
                      authored_keys=("x", "y", "color")),
        door,
        EntityInstance(id="lamp_2", class_name="light", ordinal=2,
                      fields=lep.light_entry_to_fields(
                          LightEntry(x=2.0, y=2.0, color=(0.0, 1.0, 0.0))),
                      authored_keys=("x", "y", "color")),
    ]
    edited_lamp1 = lep.EditableLight(x=9.0, y=9.0, color=(1.0, 0.0, 0.0),
                                     id="lamp_1")
    lights = [edited_lamp1]     # lamp_2 deleted, lamp_1 edited
    out = lep.merge_light_entities(original, lights)
    assert [e.id for e in out] == ["lamp_1", "door_1"]
    assert out[1] is door                       # untouched, same object
    assert out[0].fields["x"] == 9.0             # edit applied


def test_merge_light_entities_appends_new_lights_at_end():
    original = []
    lights = [lep.EditableLight(x=1.0, y=1.0, color=(1.0, 1.0, 1.0),
                                id="light_1")]
    out = lep.merge_light_entities(original, lights)
    assert [e.id for e in out] == ["light_1"]
    assert out[0].class_name == "light"


def test_merge_light_entities_all_deleted_is_empty():
    original = [EntityInstance(
        id="lamp_1", class_name="light", ordinal=0,
        fields=lep.light_entry_to_fields(LightEntry(x=1.0, y=1.0,
                                                    color=(1.0, 0.0, 0.0))),
        authored_keys=("x", "y", "color"))]
    assert lep.merge_light_entities(original, []) == []


# ---------------------------------------------------------------------------
# format_lights_for_save — chooses the family, never both, never migrates
# ---------------------------------------------------------------------------

def test_format_lights_for_save_legacy_never_touches_entity_family():
    lights = [lep.EditableLight(x=1.0, y=1.0, color=(1.0, 0.0, 0.0),
                                id="light_1")]
    repl = lep.format_lights_for_save("legacy", lights, other_entities=[])
    assert set(repl) == {"light"}


def test_format_lights_for_save_entity_never_touches_light_family():
    lights = [lep.EditableLight(x=1.0, y=1.0, color=(1.0, 0.0, 0.0),
                                id="light_1")]
    repl = lep.format_lights_for_save("entity", lights, other_entities=[])
    assert set(repl) == {"entity"}


def test_format_lights_for_save_legacy_writes_light_lines(tmp_path):
    lights = [lep.EditableLight(x=1.0, y=1.0, color=(1.0, 0.0, 0.0),
                                id="light_1")]
    repl = lep.format_lights_for_save("legacy", lights, other_entities=[])
    text = "".join(repl["light"]("\n"))
    assert "[[light]]" in text and "pos = [1.0, 1.0]" in text


def test_format_lights_for_save_entity_writes_entity_lines():
    lights = [lep.EditableLight(x=1.0, y=1.0, color=(1.0, 0.0, 0.0),
                                id="light_1")]
    repl = lep.format_lights_for_save("entity", lights, other_entities=[])
    text = "".join(repl["entity"]("\n"))
    assert '[[entity]]' in text and 'class = "light"' in text


# ---------------------------------------------------------------------------
# level_lib byte-stable round trip stays green with the new writer choice
# (the gate's own explicit requirement)
# ---------------------------------------------------------------------------

def test_legacy_level_save_stays_legacy_form_no_entity_family(tmp_path):
    d = _mini_level(tmp_path, LEGACY_LIGHTS)
    lvl = _load(d)
    lights = lep.initial_editable_lights(lvl)
    form = lep.light_form(lvl.raw_toml)
    other = [e for e in lvl.entities if e.class_name != lep.LIGHT_CLASS]
    from level_lib import open_level
    handle = open_level(str(d))
    handle.save(lep.format_lights_for_save(form, lights, other))
    text = (d / "level.toml").read_text(encoding="utf-8")
    assert "[[light]]" in text
    assert "[[entity]]" not in text
    assert len(_load(d).lights) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
