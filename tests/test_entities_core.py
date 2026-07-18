"""Entity core unit tests (Arc A patch 1 — entity design §2/§3b/§4/§5).

Covers the schema machinery itself: registration + validation, the
duplicate-class-name hard error, the entities.toml numeric overlay (applied,
"No schema in TOML" hard errors), registry content-hash stability, the
entity_registry.json export, and the `light` exemplar's schema shape
(mirroring level_loader.LightEntry). Runtime behavior is Arc B — nothing
here steps a sim.

Run:
    conda run -n data python -m pytest tests/test_entities_core.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation.entities import (  # noqa: E402
    ALIVE_SIGNAL, Entity, EntitySchemaError, EntityTomlError, Field,
    INPUT_EDGE, INPUT_HELD, InputDecl, KIND_COLOR_RGB, KIND_ENUM,
    KIND_FLOAT_RENDER, KIND_INT, KIND_LENGTH_M, KIND_Q16, KIND_STR_LIST,
    REGISTRY, Signal, all_signals, apply_tuning_overlay, clear_tuning_overlay,
    effective_defaults, export_registry_json, register, registry_content_hash,
    registry_payload,
)
from simulation.entities.schema import INSTANCE_FIELDS  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_overlay():
    """Every test starts and ends with no tuning overrides applied."""
    clear_tuning_overlay()
    yield
    clear_tuning_overlay()


def _door_like(name="tdoor"):
    """A door-shaped test class exercising the full vocabulary (design §4):
    Q16/meters fields, a signal, held+edge inputs, close-beats-open priority,
    a format-reserved interaction. Fresh class per call, renamed to `name`
    (the registry keys by ``__name__``)."""
    class tdoor(Entity):
        FIELDS = (
            Field("state", KIND_ENUM, default="closed",
                  choices=("open", "closed")),
            Field("close_force_q16", KIND_Q16, default=65536),
            Field("sensor_radius", KIND_LENGTH_M, default=2.5),
            Field("cycles", KIND_INT, default=0, minimum=0),
        )
        SIGNALS = (Signal("is_open"),)
        INPUTS = (
            InputDecl("open", INPUT_HELD),
            InputDecl("close", INPUT_HELD),
            InputDecl("toggle", INPUT_EDGE),
        )
        INPUT_PRIORITY = (("close", "open"),)   # close beats open (§4)
        INTERACTIONS = ("use",)                 # format-reserved (§3d)
    tdoor.__name__ = tdoor.__qualname__ = name
    return tdoor


# ---------------------------------------------------------------------------
# Registration + validation
# ---------------------------------------------------------------------------
def test_registration_into_private_registry():
    reg = {}
    cls = register(_door_like(), registry=reg)
    assert reg == {"tdoor": cls}
    # the free `alive` signal is machinery-provided, first in the tuple
    names = [s.name for s in all_signals(cls)]
    assert names == ["alive", "is_open"]


def test_duplicate_class_name_hard_errors():
    reg = {}
    register(_door_like(), registry=reg)
    with pytest.raises(EntitySchemaError, match="duplicate entity class"):
        register(_door_like(), registry=reg)


def test_q16_field_rejects_float_default():
    class tbad(Entity):
        FIELDS = (Field("force", KIND_Q16, default=1.5),)
    with pytest.raises(EntitySchemaError, match="plain int"):
        register(tbad, registry={})


def test_declaring_alive_is_rejected():
    class tbad2(Entity):
        SIGNALS = (Signal("alive"),)
    with pytest.raises(EntitySchemaError, match="free machinery"):
        register(tbad2, registry={})


def test_field_shadowing_instance_facts_is_rejected():
    class tbad3(Entity):
        FIELDS = (Field("id", KIND_INT, default=0),)
    with pytest.raises(EntitySchemaError, match="instance-level"):
        register(tbad3, registry={})


def test_input_priority_must_name_declared_inputs():
    class tbad4(Entity):
        INPUTS = (InputDecl("open", INPUT_HELD),)
        INPUT_PRIORITY = (("close", "open"),)
    with pytest.raises(EntitySchemaError, match="unknown input"):
        register(tbad4, registry={})


def test_instance_facts_are_schema_level():
    facts = {f.name: f for f in INSTANCE_FIELDS}
    assert facts["id"].default is None          # mandatory (design §3a)
    assert facts["tags"].kind == KIND_STR_LIST  # design §3c
    assert facts["tags"].default == ()


# ---------------------------------------------------------------------------
# entities.toml overlay
# ---------------------------------------------------------------------------
def test_overlay_applied_onto_defaults(tmp_path):
    reg = {}
    cls = register(_door_like(), registry=reg)
    toml = tmp_path / "entities.toml"
    toml.write_text("[tdoor]\nclose_force_q16 = 131072\ncycles = 3\n",
                    encoding="utf-8")
    applied = apply_tuning_overlay(toml, registry=reg)
    assert applied == {"tdoor": {"close_force_q16": 131072, "cycles": 3}}
    eff = effective_defaults("tdoor", registry=reg)
    assert eff["close_force_q16"] == 131072
    assert eff["cycles"] == 3
    assert eff["sensor_radius"] == 2.5          # untouched field keeps default
    # declared Field defaults are never mutated
    assert dict((f.name, f.default) for f in cls.FIELDS)["close_force_q16"] \
        == 65536


def test_overlay_unknown_class_and_field_hard_error(tmp_path):
    reg = {}
    register(_door_like(), registry=reg)
    bad_cls = tmp_path / "a.toml"
    bad_cls.write_text("[ghost]\nx = 1\n", encoding="utf-8")
    with pytest.raises(EntityTomlError, match="unknown entity class 'ghost'"):
        apply_tuning_overlay(bad_cls, registry=reg)
    bad_field = tmp_path / "b.toml"
    bad_field.write_text("[tdoor]\nghost = 1\n", encoding="utf-8")
    with pytest.raises(EntityTomlError, match="unknown field 'ghost'"):
        apply_tuning_overlay(bad_field, registry=reg)


def test_overlay_numbers_only(tmp_path):
    reg = {}
    register(_door_like(), registry=reg)
    non_number = tmp_path / "a.toml"
    non_number.write_text('[tdoor]\ncycles = "many"\n', encoding="utf-8")
    with pytest.raises(EntityTomlError, match="NUMBERS only"):
        apply_tuning_overlay(non_number, registry=reg)
    non_numeric_kind = tmp_path / "b.toml"
    non_numeric_kind.write_text('[tdoor]\nstate = 1\n', encoding="utf-8")
    with pytest.raises(EntityTomlError, match="NUMBERS only"):
        apply_tuning_overlay(non_numeric_kind, registry=reg)
    float_on_q16 = tmp_path / "c.toml"
    float_on_q16.write_text("[tdoor]\nclose_force_q16 = 1.5\n",
                            encoding="utf-8")
    with pytest.raises(EntityTomlError, match="integer-domain"):
        apply_tuning_overlay(float_on_q16, registry=reg)


def test_overlay_missing_file_is_noop(tmp_path):
    reg = {}
    register(_door_like(), registry=reg)
    assert apply_tuning_overlay(tmp_path / "absent.toml", registry=reg) == {}


def test_shipped_entities_toml_applies_cleanly():
    # The repo-root overlay must always load against the real registry.
    apply_tuning_overlay()


# ---------------------------------------------------------------------------
# Content hash + export
# ---------------------------------------------------------------------------
def test_content_hash_stable_and_order_independent():
    reg_a, reg_b = {}, {}
    door, lamp = _door_like(), _door_like("tlamp")
    register(door, registry=reg_a)
    register(lamp, registry=reg_a)
    register(lamp, registry=reg_b)     # same classes, reversed insertion
    register(door, registry=reg_b)
    h = registry_content_hash(reg_a)
    assert h == registry_content_hash(reg_a)   # same registry -> same hash
    assert h == registry_content_hash(reg_b)   # insertion order irrelevant
    assert len(h) == 64 and int(h, 16) >= 0    # sha256 hex


def test_content_hash_tracks_overlay(tmp_path):
    reg = {}
    register(_door_like(), registry=reg)
    base = registry_content_hash(reg)
    toml = tmp_path / "entities.toml"
    toml.write_text("[tdoor]\ncycles = 7\n", encoding="utf-8")
    apply_tuning_overlay(toml, registry=reg)
    assert registry_content_hash(reg) != base  # tuning is match-setup (§2)
    clear_tuning_overlay()
    assert registry_content_hash(reg) == base


def test_export_registry_json(tmp_path):
    out = tmp_path / "entity_registry.json"
    export_registry_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["content_hash"] == registry_content_hash()
    assert "light" in data["classes"]
    assert [f["name"] for f in data["instance_fields"]] == ["id", "tags"]
    lf = {f["name"]: f for f in data["classes"]["light"]["fields"]}
    assert lf["range"]["default"] == 12.0


# ---------------------------------------------------------------------------
# The light exemplar (mirrors level_loader.LightEntry)
# ---------------------------------------------------------------------------
def test_light_exemplar_schema_shape():
    cls = REGISTRY["light"]
    assert cls.INTANGIBLE is True
    fields = {f.name: f for f in cls.FIELDS}
    assert list(fields) == ["x", "y", "color", "intensity", "range", "kind",
                            "period_s", "beam_deg", "phase"]
    assert fields["color"].kind == KIND_COLOR_RGB
    assert fields["color"].default == (255, 255, 255)
    assert fields["kind"].kind == KIND_ENUM
    assert fields["kind"].choices == ("static", "beacon")
    # defaults mirror LightEntry (level_loader.py)
    assert fields["intensity"].default == 1.0
    assert fields["range"].default == 12.0
    assert fields["kind"].default == "static"
    assert fields["period_s"].default == 2.0
    assert fields["beam_deg"].default == 30.0
    assert fields["phase"].default == 0.0
    # render-only floats, never synced (design §5 / LightEntry contract)
    for name in ("x", "y", "intensity", "range", "period_s", "beam_deg",
                 "phase"):
        assert fields[name].kind == KIND_FLOAT_RENDER
    # the P4 forbidden knobs never entered the schema
    assert "heat" not in fields and "jitter" not in fields
    # vocabulary: only the free alive signal, no inputs yet
    assert [s.name for s in all_signals(cls)] == [ALIVE_SIGNAL.name]
    assert cls.INPUTS == () and cls.INTERACTIONS == ()
    # payload round-trips through the serializer
    payload = registry_payload()["classes"]["light"]
    assert payload["intangible"] is True
    assert payload["signals"][0]["name"] == "alive"
