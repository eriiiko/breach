"""``[[entity]]`` format (Arc A3 — entity design §3a/§3b/§3e, editor §6).

Pins the load semantics: mandatory unique ids assigned runtime ordinals in
FILE order, registry-validated fields with effective defaults (entities.toml
overlay included), tags, the ref machinery (dangling ref warns, ref-to-unit
hard-errors — §3e), the ``[[light]]`` legacy alias (downstream equivalence +
mixed-form hard error), ``[[spawn]]`` permanence, and level_lib's 'entity'
managed family with a byte-stable round-trip. Dormancy: entities are parsed
data only — nothing steps them, digests are A4's.

Run:
    conda run -n data python -m pytest tests/test_entity_format.py -q
"""
from __future__ import annotations

import struct
import sys
import warnings
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_lib import (MANAGED_FAMILIES, format_entity_lines,  # noqa: E402
                       format_light_lines, open_level, write_managed_blocks)
from level_lights import light_source_params  # noqa: E402
from simulation.entities import (  # noqa: E402
    Entity, Field, KIND_ENTITY_REF, REGISTRY, apply_tuning_overlay,
    clear_tuning_overlay, register,
)


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


PREFIX = ("# hand comment stays\n"
          'version = "2"\nname = "T"\ntilemap = "tilemap.csv"\n'
          'tile_size_m = 0.333\ndiffuse = "diffuse.png"\n\n')
SUFFIX = ('[art.bare]\n# baked by the P2 baker\n'
          'diffuse = "diffuse.png"\n\n'
          '[bake]\ntileset = "x"\npx_per_tile = 8\nseed = 0\n')

SPAWN = '[[spawn]]\nname = "marine_1"\nteam = 0\nx = 2.0\ny = 3.0\n\n'
LAMP = ('[[entity]]\nid = "lamp_1"\nclass = "light"\n'
        'x = 2.5\ny = 3.5\ncolor = [255, 0, 0]\n\n')


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


@pytest.fixture
def watcher_cls():
    """A test-only registered class carrying a KIND_ENTITY_REF field — no
    SHIPPED class gains a ref field in Arc A; registered into the real
    REGISTRY for one test, then removed."""
    class twatcher(Entity):
        INTANGIBLE = True
        FIELDS = (Field("watches", KIND_ENTITY_REF, default=""),)
    register(twatcher)
    yield twatcher
    del REGISTRY["twatcher"]


# ---------------------------------------------------------------------------
# (a) Ids — mandatory, unique, slugs, file-order ordinals (design §3a)
# ---------------------------------------------------------------------------

def test_entity_free_level_has_empty_entities(tmp_path):
    lvl = _load(_mini_level(tmp_path, SPAWN))
    assert lvl.entities == []           # dormancy: nothing appears unasked


def test_duplicate_id_hard_error(tmp_path):
    body = ('[[entity]]\nid = "lamp_1"\nclass = "light"\n\n'
            '[[entity]]\nid = "lamp_1"\nclass = "light"\n\n')
    with pytest.raises(ValueError, match="duplicate id 'lamp_1'"):
        _load(_mini_level(tmp_path, body))


def test_missing_or_malformed_id_hard_error(tmp_path):
    with pytest.raises(ValueError, match="'id' must be a slug"):
        _load(_mini_level(tmp_path, '[[entity]]\nclass = "light"\n\n'))
    body = '[[entity]]\nid = "bad id!"\nclass = "light"\n\n'
    with pytest.raises(ValueError, match="'id' must be a slug"):
        _load(_mini_level(tmp_path, body, name="mini2"))


def test_file_order_id_assignment(tmp_path):
    body = ('[[entity]]\nid = "zeta"\nclass = "light"\n\n'
            '[[entity]]\nid = "alpha"\nclass = "light"\n\n')
    lvl = _load(_mini_level(tmp_path, body))
    assert [e.id for e in lvl.entities] == ["zeta", "alpha"]  # never sorted
    assert [e.ordinal for e in lvl.entities] == [0, 1]        # file order


# ---------------------------------------------------------------------------
# (b) Registry validation (design §3b) + tags (§3c)
# ---------------------------------------------------------------------------

def test_unknown_class_hard_error(tmp_path):
    body = '[[entity]]\nid = "g_1"\nclass = "ghost"\n\n'
    with pytest.raises(ValueError, match="unknown entity class 'ghost'"):
        _load(_mini_level(tmp_path, body))


def test_unknown_field_hard_error(tmp_path):
    # `heat` doubles as the P4 forbidden-knob guard: it is simply not in
    # the light schema, so the registry validation rejects it.
    body = '[[entity]]\nid = "lamp_1"\nclass = "light"\nheat = 1.0\n\n'
    with pytest.raises(ValueError, match="unknown field 'heat'"):
        _load(_mini_level(tmp_path, body))


def test_type_mismatch_hard_error(tmp_path):
    body = ('[[entity]]\nid = "lamp_1"\nclass = "light"\n'
            'intensity = "bright"\n\n')
    with pytest.raises(ValueError, match="must be a number"):
        _load(_mini_level(tmp_path, body))
    body = '[[entity]]\nid = "lamp_1"\nclass = "light"\nkind = "strobe"\n\n'
    with pytest.raises(ValueError, match="must be one of"):
        _load(_mini_level(tmp_path, body, name="mini2"))
    body = '[[entity]]\nid = "lamp_1"\nclass = "light"\nbeam_deg = 400.0\n\n'
    with pytest.raises(ValueError, match="above maximum"):
        _load(_mini_level(tmp_path, body, name="mini3"))


def test_missing_required_field_hard_error(tmp_path):
    class treq(Entity):
        INTANGIBLE = True
        FIELDS = (Field("target", KIND_ENTITY_REF, default=None),)
    register(treq)
    try:
        body = '[[entity]]\nid = "r_1"\nclass = "treq"\n\n'
        with pytest.raises(ValueError, match="missing required"):
            _load(_mini_level(tmp_path, body))
    finally:
        del REGISTRY["treq"]


def test_single_table_spelling_hard_error(tmp_path):
    body = '[entity]\nid = "lamp_1"\nclass = "light"\n\n'
    with pytest.raises(ValueError, match=r"spell it \[\[entity\]\]"):
        _load(_mini_level(tmp_path, body))


def test_tags_parsed_and_validated(tmp_path):
    body = ('[[entity]]\nid = "lamp_1"\nclass = "light"\n'
            'tags = ["deck", "aft"]\n\n')
    lvl = _load(_mini_level(tmp_path, body))
    assert lvl.entities[0].tags == ("deck", "aft")
    bad = '[[entity]]\nid = "lamp_1"\nclass = "light"\ntags = [1, 2]\n\n'
    with pytest.raises(ValueError, match="'tags' must be an array"):
        _load(_mini_level(tmp_path, bad, name="mini2"))


def test_effective_defaults_and_overlay_applied(tmp_path):
    d = _mini_level(tmp_path, LAMP)
    lvl = _load(d)
    inst = lvl.entities[0]
    assert inst.class_name == "light"
    assert inst.authored_keys == ("x", "y", "color")   # only what was spelled
    assert inst.fields["x"] == 2.5                     # authored, as authored
    assert inst.fields["intensity"] == 1.0             # registry default
    # entities.toml overlay numbers ARE the effective defaults (design §2).
    overlay = tmp_path / "entities.toml"
    overlay.write_text("[light]\nintensity = 3.0\n", encoding="utf-8")
    apply_tuning_overlay(overlay)
    try:
        assert _load(d).entities[0].fields["intensity"] == 3.0
    finally:
        clear_tuning_overlay()


# ---------------------------------------------------------------------------
# (c) Legacy coexistence — [[spawn]] permanent, [[light]] an exclusive alias
# ---------------------------------------------------------------------------

def test_spawn_and_entity_coexist(tmp_path):
    # [[spawn]] is PERMANENT syntax (design §3e), never a migration target.
    lvl = _load(_mini_level(tmp_path, SPAWN + LAMP))
    assert [s.name for s in lvl.spawns] == ["marine_1"]
    assert [e.id for e in lvl.entities] == ["lamp_1"]


def test_mixed_light_forms_hard_error(tmp_path):
    body = '[[light]]\npos = [1.0, 1.0]\ncolor = [255, 255, 255]\n\n' + LAMP
    with pytest.raises(ValueError, match="migration"):
        _load(_mini_level(tmp_path, body))


def test_legacy_light_with_nonlight_entity_ok(tmp_path, watcher_cls):
    # The mixed-form rule is alias-scoped: [[light]] excludes [[entity]]
    # LIGHT instances specifically; other entity classes coexist fine.
    body = ('[[light]]\npos = [1.0, 1.0]\ncolor = [255, 255, 255]\n\n'
            '[[entity]]\nid = "w_1"\nclass = "twatcher"\n\n')
    lvl = _load(_mini_level(tmp_path, body))
    assert len(lvl.lights) == 1 and len(lvl.entities) == 1


# ---------------------------------------------------------------------------
# (d) Refs — dangling warns, unit hard-errors (design §3a / §3e)
# ---------------------------------------------------------------------------

def test_dangling_ref_warns_not_fatal(tmp_path, watcher_cls):
    body = ('[[entity]]\nid = "w_1"\nclass = "twatcher"\n'
            'watches = "ghost_9"\n\n')
    with pytest.warns(UserWarning, match="dangling ref"):
        lvl = _load(_mini_level(tmp_path, body))
    assert lvl.entities[0].fields["watches"] == "ghost_9"   # load succeeded


def test_valid_and_unwired_refs_do_not_warn(tmp_path, watcher_cls):
    body = (LAMP
            + '[[entity]]\nid = "w_1"\nclass = "twatcher"\n'
              'watches = "lamp_1"\n\n'
            + '[[entity]]\nid = "w_2"\nclass = "twatcher"\n\n')
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lvl = _load(_mini_level(tmp_path, body))
    assert [e.id for e in lvl.entities] == ["lamp_1", "w_1", "w_2"]


def test_ref_to_unit_hard_error(tmp_path, watcher_cls):
    body = (SPAWN
            + '[[entity]]\nid = "w_1"\nclass = "twatcher"\n'
              'watches = "marine_1"\n\n')
    with pytest.raises(ValueError, match="units are NOT entities"):
        _load(_mini_level(tmp_path, body))


# ---------------------------------------------------------------------------
# (e) The [[light]] alias — identical downstream render inputs
# ---------------------------------------------------------------------------

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


def test_light_alias_equivalence(tmp_path):
    lvl_leg = _load(_mini_level(tmp_path, LEGACY_LIGHTS, name="leg"))
    lvl_ent = _load(_mini_level(tmp_path, ENTITY_LIGHTS, name="ent"))
    # Same LightEntry list, field for field (authored + defaulted values).
    assert lvl_ent.lights == lvl_leg.lights
    assert len(lvl_ent.lights) == 2
    # And the same raycaster LightSource parameters downstream.
    for a, b in zip(lvl_ent.lights, lvl_leg.lights):
        assert (light_source_params(a, total_tick=7, tick_dt_s=1.0 / 24.0)
                == light_source_params(b, total_tick=7, tick_dt_s=1.0 / 24.0))
    # The entity spelling also lands in `entities` (parsed data, dormant).
    assert [e.id for e in lvl_ent.entities] == ["lamp_1", "lamp_2"]
    assert lvl_leg.entities == []


# ---------------------------------------------------------------------------
# (f) level_lib — the 'entity' managed family + byte-stable round-trips
# ---------------------------------------------------------------------------

ROUND_TRIP_BODY = ('[[entity]]\n'
                   'id = "lamp_1"\n'
                   'class = "light"\n'
                   'tags = ["deck", "aft"]\n'
                   'x = 2.5\n'
                   'y = 3.5\n'
                   'color = [255, 0, 0]\n'
                   'kind = "beacon"\n'
                   'period_s = 1.5\n'
                   '\n'
                   '[[entity]]\n'
                   'id = "lamp_2"\n'
                   'class = "light"\n'
                   'x = 1.0\n'
                   'y = 1.0\n'
                   'color = [10, 20, 30]\n'
                   '\n')


def test_entity_family_registered():
    assert "entity" in MANAGED_FAMILIES
    assert MANAGED_FAMILIES["entity"].array is True


def test_entity_round_trip_byte_stable(tmp_path):
    d = _mini_level(tmp_path, ROUND_TRIP_BODY)
    toml = d / "level.toml"
    before = toml.read_bytes()
    handle = open_level(str(d))
    handle.save()                       # no replacements — pure identity
    assert toml.read_bytes() == before
    # Load -> format -> write lands byte-identically: authored fields only,
    # authored order, canonical value formatting.
    handle.save({"entity":
                 lambda nl: format_entity_lines(handle.data.entities, nl)})
    assert toml.read_bytes() == before
    reloaded = _load(d)
    assert reloaded.entities == handle.data.entities


def test_legacy_level_round_trip_untouched(tmp_path):
    """A legacy ([[spawn]]/[[light]]) level is NEVER silently converted:
    saving keeps the legacy form and no [[entity]] family appears."""
    d = _mini_level(tmp_path, SPAWN + LEGACY_LIGHTS)
    toml = d / "level.toml"
    before = toml.read_bytes()
    handle = open_level(str(d))
    handle.save()                       # no replacements — byte-identical
    assert toml.read_bytes() == before
    # The Ctrl+S shape on an unmigrated level rewrites the LEGACY family.
    handle.save({"light":
                 lambda nl: format_light_lines(handle.data.lights, nl)})
    text = toml.read_text(encoding="utf-8")
    assert "[[light]]" in text
    assert "[[entity]]" not in text
    assert len(_load(d).lights) == 2


def test_write_entities_into_entity_free_level_only_on_request(tmp_path):
    """The writer emits [[entity]] only when a caller ASKS for the family —
    and then appends it as a well-formed block an A3 load accepts."""
    d = _mini_level(tmp_path, SPAWN)
    toml = d / "level.toml"
    ents = _load(_mini_level(tmp_path, ROUND_TRIP_BODY, name="src")).entities
    write_managed_blocks(toml, {"entity":
                                lambda nl: format_entity_lines(ents, nl)})
    lvl = _load(d)
    assert [e.id for e in lvl.entities] == ["lamp_1", "lamp_2"]
    assert lvl.entities == ents
    assert [s.name for s in lvl.spawns] == ["marine_1"]   # untouched
