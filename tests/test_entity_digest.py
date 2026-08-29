"""A4 — entity digest scaffolding (docs/a4_digest_impl_note_2026-07-18.md v2).

The gate for the ``__entity__``/``__signals__`` tick-digest sections:

1. DORMANCY — an entity-free digest through the new path is bit-identical to
   a pre-A4 reference implementation frozen inline here (the full suite's
   zero-golden-edit run is the other half of this gate).
2. An entity-present level's digest differs from its entity-stripped twin.
3. STRICTNESS — a sim with entities + a snapshot lacking ``__entity__``
   raises (an entity-present run can never hash entity-free silently);
   pre-A4 snapshots (no key) stay entity-free by construction.
4. STABILITY — two loads serialize identically; iteration order is
   ordinal + schema-declaration only (dict-insertion independence).
5. Ref encoding — unwired ""/dangling -> -1, wired -> target ordinal.
6. The [[light]]-alias digest consequence is PINNED (documents the A7
   re-baseline scope: the alias migration IS digest-changing).
7. Registration name-charset guard + loud int64 overflow.
8. Recorder round-trip — sections present iff entities present; digest and
   recorder consume THE one serializer's bytes.

Fixture levels are synthetic tmp folders ONLY — adding an entity to an
existing digest-suite level would flip its golden (impl note critique 11).

Run:
    conda run -n data python -m pytest tests/test_entity_digest.py -q
"""
from __future__ import annotations

import abc
import hashlib
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tests", ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_loader import EntityInstance  # noqa: E402
from field_digest import DIGEST_FIELDS, field_digest, tick_digest  # noqa: E402
from simulation.entities import (  # noqa: E402
    ENTITY_DIGEST_KEY, ENTITY_SECT_PREAMBLE, Entity, EntitySchemaError, Field,
    KIND_BOOL, KIND_ENTITY_REF, KIND_ENUM, KIND_FLOAT_RENDER, KIND_INT,
    KIND_LENGTH_M, KIND_Q16, KIND_STR, SIGNAL_SECT_PREAMBLE, Signal,
    entity_carrier, entity_section_bytes, register, registry_content_hash,
    require_entity_carrier, serialize_entity_state, serialize_signal_state,
    signal_section_bytes,
)
from simulation.gases import O2  # noqa: E402
from simulation.recorder import PhysicsRecorder  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — synthetic level folders + snapshots (never the repo's levels/)
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

LEGACY_LIGHTS = ('[[light]]\npos = [2.5, 3.5]\ncolor = [255, 0, 0]\n\n'
                 '[[light]]\npos = [1.0, 4.0]\ncolor = [10, 20, 30]\n\n')
ENTITY_LIGHTS = ('[[entity]]\nid = "lamp_1"\nclass = "light"\n'
                 'x = 2.5\ny = 3.5\ncolor = [255, 0, 0]\n\n'
                 '[[entity]]\nid = "lamp_2"\nclass = "light"\n'
                 'x = 1.0\ny = 4.0\ncolor = [10, 20, 30]\n\n')


def _mini_level(tmp_path: Path, body: str = "", name: str = "mini") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join("0" for _ in range(8)) for _ in range(6)) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(PREFIX + body, encoding="utf-8",
                                  newline="\n")
    return d


def _load(d: Path):
    return level_loader.load(str(d))


def _mini_snapshot(h: int = 4, w: int = 4) -> dict:
    """A deterministic synthetic snapshot carrying every DIGEST_FIELDS array
    (field_digest checks names + dtypes, not shapes) plus a fixed unit hash,
    so tick_digest exercises its full pre-A4 fold."""
    snap = {}
    for j, (name, dtype) in enumerate(DIGEST_FIELDS):
        shape = (2, h, w) if name == "gas" else (h, w)
        n = int(np.prod(shape))
        arr = (np.arange(n, dtype=np.int64) * 7 + j).reshape(shape)
        snap[name] = (arr % 2).astype(bool) if dtype == "bool" \
            else arr.astype(dtype)
    snap["__unit_state__"] = {"units": [], "events": [], "hash": "f" * 64}
    return snap


def _pre_a4_tick_digest(snapshot: dict) -> str:
    """The pre-A4 tick_digest, frozen VERBATIM as the dormancy reference —
    if the new path ever perturbs an entity-free hash, this catches it
    without any committed golden moving."""
    fd = field_digest(snapshot)
    unit_hash = ""
    if "__unit_state__" in snapshot:
        unit_hash = snapshot["__unit_state__"]["hash"]
    h = hashlib.blake2b(digest_size=32)
    h.update(fd.encode("ascii"))
    h.update(b"|")
    h.update(unit_hash.encode("ascii"))
    return h.hexdigest()


# A private-registry test class covering every synced kind AND every excluded
# kind — no SHIPPED class gains these fields in Arc A (critique 11 spirit:
# fixture-only, never a repo level/class).
_REG: dict = {}


@register(registry=_REG)
class tdig(Entity):
    INTANGIBLE = True
    FIELDS = (
        Field("hp", KIND_INT, default=5),
        Field("rate_q16", KIND_Q16, default=65536),
        Field("armed", KIND_BOOL, default=False),
        Field("mode", KIND_ENUM, default="idle", choices=("idle", "hot")),
        Field("watches", KIND_ENTITY_REF, default=""),
        Field("label", KIND_STR, default="x"),           # excluded: str
        Field("gain", KIND_FLOAT_RENDER, default=1.0),   # excluded: render
        Field("span_m", KIND_LENGTH_M, default=2.5),     # excluded: blocker 1
    )


def _inst(eid: str, ordinal: int, **over) -> EntityInstance:
    fields = {f.name: f.default for f in tdig.FIELDS}
    fields.update(over)
    return EntityInstance(id=eid, class_name="tdig", ordinal=ordinal,
                          fields=fields)


def _record_rows(record: bytes) -> dict:
    """Parse one ENTITY_SECT_V1 record's ``name|int64`` rows (the header is
    the record's first line; row names are charset-guarded, so the first
    '|' from a row start always delimits the name)."""
    body = record[record.index(b"\n") + 1:]
    rows, pos = {}, 0
    while pos < len(body) and body[pos:pos + 1] != b"\n":
        bar = body.index(b"|", pos)
        rows[body[pos:bar].decode("ascii")] = \
            struct.unpack("<q", body[bar + 1:bar + 9])[0]
        assert body[bar + 9:bar + 10] == b"\n"
        pos = bar + 10
    return rows


# ---------------------------------------------------------------------------
# 1. Dormancy — entity-free through the new path == pre-A4 reference
# ---------------------------------------------------------------------------

def test_dormancy_entity_free_digest_equals_pre_a4_reference():
    snap = _mini_snapshot()
    ref = _pre_a4_tick_digest(snap)
    # Pre-A4 snapshot (no carrier key at all) — entity-free by construction.
    assert tick_digest(snap) == ref
    # New-path snapshot: carrier present with n_entities == 0 — the marker
    # gates the fold and is never hashed, so the bytes are identical.
    gated = dict(snap)
    gated[ENTITY_DIGEST_KEY] = entity_carrier([])
    assert tick_digest(gated) == ref


# ---------------------------------------------------------------------------
# 2. Entity-present digest differs from the entity-stripped twin
# ---------------------------------------------------------------------------

def test_entity_present_digest_differs_from_stripped_twin(tmp_path):
    ents = _load(_mini_level(tmp_path, ENTITY_LIGHTS)).entities
    snap = _mini_snapshot()
    stripped = dict(snap)
    stripped[ENTITY_DIGEST_KEY] = entity_carrier([])
    present = dict(snap)
    present[ENTITY_DIGEST_KEY] = entity_carrier(ents)
    assert present[ENTITY_DIGEST_KEY]["n_entities"] == 2
    assert tick_digest(present) != tick_digest(stripped)
    # The FIELD digest is untouched either way — only the tick fold moves.
    assert field_digest(present) == field_digest(stripped)


# ---------------------------------------------------------------------------
# 3. Strictness — entities loaded + missing carrier raises
# ---------------------------------------------------------------------------

def test_missing_carrier_with_entities_loaded_raises(tmp_path):
    ents = _load(_mini_level(tmp_path, ENTITY_LIGHTS)).entities
    snap = _mini_snapshot()                  # no __entity__ key
    with pytest.raises(KeyError, match="__entity__"):
        require_entity_carrier(ents, snap)
    # Entity-free sims accept pre-A4 snapshots (no key) — no raise.
    require_entity_carrier([], snap)


def test_capture_and_get_state_always_write_the_carrier(tmp_path):
    import breach_physics as bp
    from simulation import Simulation
    from field_ab_harness import _scenario_level, capture_trajectory
    # Capture path, entity-free scenario: carrier present, N == 0.
    traj = capture_trajectory(n_steps=2)
    for snap in traj:
        assert snap[ENTITY_DIGEST_KEY]["n_entities"] == 0
    # get_state, entity-present level: carrier carries the serialized
    # payload (not just a hash) from THE one serializer.
    ents = _load(_mini_level(tmp_path, ENTITY_LIGHTS)).entities
    level = _scenario_level()
    level.entities = ents
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False)
    st = sim.get_state()
    assert st.entity_state["n_entities"] == 2
    assert entity_section_bytes(st.entity_state) == serialize_entity_state(ents)
    assert st.entity_state["registry_hash"] == registry_content_hash()
    # get_state, entity-free level: carrier still written, N == 0.
    sim0 = Simulation(_scenario_level(), seed=1, breach_physics=bp,
                      enable_recorder=False)
    assert sim0.get_state().entity_state["n_entities"] == 0


# ---------------------------------------------------------------------------
# 4. Stability — two loads identical; ordinal + declaration order only
# ---------------------------------------------------------------------------

def test_two_loads_serialize_identically(tmp_path):
    d = _mini_level(tmp_path, ENTITY_LIGHTS)
    assert serialize_entity_state(_load(d).entities) \
        == serialize_entity_state(_load(d).entities)


def test_iteration_order_is_ordinal_plus_declaration_only():
    fields = {f.name: f.default for f in tdig.FIELDS}
    fwd = EntityInstance(id="e_0", class_name="tdig", ordinal=0,
                         fields=dict(fields))
    rev = EntityInstance(id="e_0", class_name="tdig", ordinal=0,
                         fields=dict(reversed(list(fields.items()))))
    assert serialize_entity_state([fwd], registry=_REG) \
        == serialize_entity_state([rev], registry=_REG)     # dict order moot
    a, b = _inst("a_0", 0), _inst("b_1", 1)
    assert serialize_entity_state([a, b], registry=_REG) \
        == serialize_entity_state([b, a], registry=_REG)    # list order moot


def test_synced_kind_partition_and_row_encodings():
    e = _inst("e_0", 0, hp=7, rate_q16=131072, armed=True, mode="hot")
    (record,) = entity_carrier([e], registry=_REG)["records"]
    assert record.split(b"\n", 1)[0] == b"0|e_0|tdig"
    rows = _record_rows(record)
    # Synced kinds, declaration order, + the free alive row (always 1, Arc A).
    assert list(rows) == ["hp", "rate_q16", "armed", "mode", "watches", "alive"]
    assert rows == {"hp": 7, "rate_q16": 131072, "armed": 1, "mode": 1,
                    "watches": -1, "alive": 1}
    # KIND_LENGTH_M / render / str NEVER enter the bytes (critique blocker
    # 1): mutating them leaves the serialization bit-identical — length_m's
    # synced consequence is quantized tile state, hashed via the fields.
    base = serialize_entity_state([_inst("e_0", 0)], registry=_REG)
    moved = serialize_entity_state(
        [_inst("e_0", 0, span_m=99.0, gain=5.0, label="zz")], registry=_REG)
    assert moved == base
    assert base.startswith(ENTITY_SECT_PREAMBLE)


# ---------------------------------------------------------------------------
# 5. Ref encoding — unwired/dangling -> -1, wired -> ordinal
# ---------------------------------------------------------------------------

def test_ref_encoding_unwired_dangling_wired():
    unwired = _inst("w_0", 0)                       # "" = unwired
    dangling = _inst("w_1", 1, watches="ghost_9")   # no such id
    wired = _inst("w_2", 2, watches="w_0")          # -> ordinal 0
    recs = entity_carrier([unwired, dangling, wired],
                          registry=_REG)["records"]
    assert _record_rows(recs[0])["watches"] == -1
    assert _record_rows(recs[1])["watches"] == -1   # dangling == unwired: -1
    assert _record_rows(recs[2])["watches"] == 0
    # Wiring a ref changes the digest bytes; unwired vs dangling does NOT
    # (both resolve to nothing at runtime, so they must hash alike).
    assert serialize_entity_state([unwired], registry=_REG) \
        != serialize_entity_state([_inst("w_0", 0, watches="w_0")],
                                  registry=_REG)
    assert serialize_entity_state([_inst("s", 0)], registry=_REG) \
        == serialize_entity_state([_inst("s", 0, watches="ghost_9")],
                                  registry=_REG)


# ---------------------------------------------------------------------------
# 6. The [[light]]-alias digest consequence, pinned (A7 re-baseline scope)
# ---------------------------------------------------------------------------

def test_light_alias_digest_consequence_pinned(tmp_path):
    leg = _load(_mini_level(tmp_path, LEGACY_LIGHTS, name="leg"))
    ent = _load(_mini_level(tmp_path, ENTITY_LIGHTS, name="ent"))
    assert leg.entities == [] and len(ent.entities) == 2
    snap = _mini_snapshot()
    s_leg = dict(snap)
    s_leg[ENTITY_DIGEST_KEY] = entity_carrier(leg.entities)
    s_ent = dict(snap)
    s_ent[ENTITY_DIGEST_KEY] = entity_carrier(ent.entities)
    # Legacy [[light]] hashes entity-free (== pre-A4); the entity twin does
    # not — the alias migration IS digest-changing, hence A7-scoped inside
    # the single sanctioned re-baseline (impl note critique 8).
    assert tick_digest(s_leg) == _pre_a4_tick_digest(snap)
    assert tick_digest(s_ent) != tick_digest(s_leg)


# ---------------------------------------------------------------------------
# 7. Name-charset registration guard + loud int64 overflow
# ---------------------------------------------------------------------------

def test_registration_name_charset_guard():
    # abc.ABCMeta (Entity's metaclass) builds classes whose names a `class`
    # statement could never spell — exactly what the guard must refuse.
    with pytest.raises(EntitySchemaError, match=r"A-Za-z0-9_"):
        register(registry={})(abc.ABCMeta("t-bad", (Entity,), {}))
    bad_field = abc.ABCMeta("tbadf", (Entity,), {
        "FIELDS": (Field("bad-name", KIND_INT, default=0),)})
    with pytest.raises(EntitySchemaError, match=r"A-Za-z0-9_"):
        register(registry={})(bad_field)
    bad_signal = abc.ABCMeta("tbads", (Entity,),
                             {"SIGNALS": (Signal("bad name"),)})
    with pytest.raises(EntitySchemaError, match=r"A-Za-z0-9_"):
        register(registry={})(bad_signal)


def test_int64_overflow_raises_loudly():
    with pytest.raises(OverflowError, match="int64"):
        serialize_entity_state([_inst("big", 0, hp=1 << 63)], registry=_REG)
    with pytest.raises(OverflowError, match="int64"):
        serialize_entity_state([_inst("neg", 0, hp=-(1 << 63) - 1)],
                               registry=_REG)


# ---------------------------------------------------------------------------
# __signals__ — empty-but-defined; alive excluded (critique 7)
# ---------------------------------------------------------------------------

def test_signal_section_empty_defined_sorted_alive_excluded():
    assert serialize_signal_state(()) == SIGNAL_SECT_PREAMBLE  # stable empty
    blob = serialize_signal_state([(1, "opened", 3), (0, "opened", 2)])
    assert blob.startswith(SIGNAL_SECT_PREAMBLE)
    assert blob.index(b"0|opened|") < blob.index(b"1|opened|")  # (ord, name)
    with pytest.raises(ValueError, match="alive"):
        serialize_signal_state([(0, "alive", 1)])
    # The carrier's signal half: empty until Arc B, hashed as bare preamble.
    carrier = entity_carrier([_inst("e", 0)], registry=_REG)
    assert signal_section_bytes(carrier) == SIGNAL_SECT_PREAMBLE


# ---------------------------------------------------------------------------
# 8. Recorder round-trip — sections iff entities; ONE serializer
# ---------------------------------------------------------------------------

class _FakeGmap:
    """Minimal stand-in exposing only what recorder.record() reads."""
    def __init__(self, fh, fw, n_gases=16):
        atm = np.full((fh, fw), 65536, dtype=np.int32)
        self.atmosphere = atm
        self.wave_p = atm.copy()          # P_prev == P: no blowup transient
        self.temperature = np.zeros((fh, fw), dtype=np.int32)
        self.smoke = np.zeros((fh, fw), dtype=np.int32)
        self.fire = np.zeros((fh, fw), dtype=np.int32)
        self.obstacles = np.zeros((fh, fw), dtype=bool)
        # P-E5: wind_x/wind_y joined Recorder.DEFAULT_FIELDS (momentum evidence
        # for the pressure-transient investigation) — mirror the real GameMap,
        # which has carried these Q16.16 planes since the EOS refactor. Same
        # fix as the twin fake in tests/test_recorder_dump.py.
        self.wind_x = np.zeros((fh, fw), dtype=np.int32)
        self.wind_y = np.zeros((fh, fw), dtype=np.int32)
        self.gas = np.zeros((n_gases, fh, fw), dtype=np.int32)
        # gas-energy conservation arc #54, P-G0: `gas_energy` joined
        # Recorder.DEFAULT_FIELDS (design §5). Same fix as the twin fake in
        # tests/test_recorder_dump.py.
        self.gas_energy = np.zeros((fh, fw), dtype=np.int64)
        assert O2 < n_gases


class _FakeUnit:
    def __init__(self, name, x, y):
        self.name = name
        self.team = 0
        self.x = x
        self.y = y
        self.current_hp = 100
        self.alive = True


def test_recorder_sections_present_iff_entities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)              # dump() writes to CWD
    ents = _load(_mini_level(tmp_path, ENTITY_LIGHTS)).entities
    fh, fw = 4, 4
    # Entity-free: the frozen .npz key set, no entity keys at all.
    rec = PhysicsRecorder(fh, fw, capacity=2)
    rec.record(_FakeGmap(fh, fw), tick=0, real_time=0.0,
               units=[_FakeUnit("a", 1, 2)])
    with np.load(tmp_path / rec.dump("efree")) as d0:
        assert "entity_state" not in d0.files
        assert "entity_registry_hash" not in d0.files
    # Entity-present: additive keys carry THE one serializer's bytes — the
    # exact bytes the tick digest hashes (entity_section_bytes on the same
    # carrier), so digest and recorder can never drift apart.
    rec2 = PhysicsRecorder(fh, fw, capacity=2)
    rec2.record(_FakeGmap(fh, fw), tick=0, real_time=0.0,
                units=[_FakeUnit("a", 1, 2)], entities=ents)
    with np.load(tmp_path / rec2.dump("epresent")) as d1:
        blob = bytes(d1["entity_state"][0])
        assert blob == serialize_entity_state(ents)
        assert blob == entity_section_bytes(entity_carrier(ents))
        assert d1["entity_registry_hash"].item() == registry_content_hash()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
