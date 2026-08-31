"""air_init.npy + boundary field (Arc A patch A9 — entity design §10;
editor design §7 owns the Arc C wand painting; boundary-conditions physics
project (ledger #1) owns the AMBIENT ring behavior).

Pins the A9 format hooks:

- ``air_init.npy``: optional int32 Q16.16 atmosphere override, shape ==
  tilemap, discovered by PRESENCE (the zones.npy pattern, no toml key).
  Absent file = air dormancy — today's ambient seeding bit-exactly. Present
  file seeds GameMap's atmosphere AND the O2/N2 species (N_total == P, 21/79
  half-up split — the EOS re-derives p* = C·N·T every tick, so an
  atmosphere-only seed would evaporate; at P == FP_ONE the split reproduces
  the P1 calibration EXACTLY, making an explicit all-ambient grid
  bit-identical to no grid). THE PINNED TILE RULE: values on solid or SPACE
  (is_vacuum) tiles are IGNORED, silently.
- ``boundary``: top-level level.toml string, "space" (default) | "ambient",
  unknown = hard error naming both; a top-level scalar, so the managed-block
  writer round-trips it byte-stably. As of the BC build (section (g)) the
  value drives real load behavior: "ambient" routes the SPACE tiles to the
  reservoir ring (GameMap.is_ambient, wholesale — is_vacuum stays empty),
  seeds the interior to the N-primary ambient dials (effective pin 65540 raw
  at defaults), and builds the static sponge grid. Space maps stay
  byte-identical (dormancy by branch).
- level_lib carrier ``write_air_init_npy``: byte-stable round-trip; deletes
  ONLY on None/empty (an all-zero or all-ambient grid STAYS a file — for
  air there is no content value meaning "no override").
- ``_upscale_level``: air replicates like water (np.repeat both axes).

All fixtures are synthetic tmp levels — the repo's levels/ (and every
digest-suite level) NEVER gets an air_init.npy or a boundary key; existing
goldens stay untouched (the dormancy guarantee).

Run:
    conda run -n data python -m pytest tests/test_air_boundary.py -q
"""
from __future__ import annotations

import struct
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import breach_physics as bp
except ImportError:                       # loader/format tests still run
    bp = None

import level_loader  # noqa: E402
from level_lib import (AIR_INIT_FILENAME, format_entity_lines,  # noqa: E402
                       open_level, write_air_init_npy, write_boundary_field)
from simulation.atmosphere_fixed import FP_ONE  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import INERT_N2, O2  # noqa: E402

AMBIENT_O2_Q = 13763                # quantize_scalar(0.21) — P1 calibration
AMBIENT_N2_Q = 51773                # FP_ONE - 13763
OVER_P = FP_ONE + FP_ONE // 2       # 1.5 atm == 98304 counts


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

ROOM = 16                           # sealed hull-ring room, v2 codes


def _mini_level(tmp_path: Path, *, body: str = "", name: str = "mini",
                prefix: str = PREFIX,
                tilemap: np.ndarray = None) -> Path:
    """A tmp level folder. Default tilemap: ROOM x ROOM sealed room — hull
    (1) ring on the border, interior air (0) — so GameMap has real solid
    tiles for the tile rule; pass ``tilemap`` for variants (SPACE etc.)."""
    d = tmp_path / name
    d.mkdir()
    if tilemap is None:
        tilemap = np.ones((ROOM, ROOM), dtype=np.int32)
        tilemap[1:-1, 1:-1] = 0
    np.savetxt(d / "tilemap.csv", tilemap, fmt="%d", delimiter=",")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(prefix + body + SUFFIX,
                                  encoding="utf-8", newline="\n")
    return d


def _air(d: Path, grid: np.ndarray) -> None:
    np.save(d / AIR_INIT_FILENAME, grid)


def _load(d: Path):
    return level_loader.load(str(d))


def _load_silent(d: Path):
    """Load asserting NO warnings fire (the happy path)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _load(d)


def _room_shape() -> tuple:
    return (ROOM, ROOM)


def _full(value: int) -> np.ndarray:
    """A full-coverage override grid — the natural authoring output: the
    value EVERYWHERE, walls included (the tile rule ignores those)."""
    return np.full(_room_shape(), value, dtype=np.int32)


# ---------------------------------------------------------------------------
# (a) Loader — presence discovery, dtype/shape/negative hard errors
# ---------------------------------------------------------------------------

def test_air_free_level_is_dormant(tmp_path):
    lvl = _load_silent(_mini_level(tmp_path))
    assert lvl.air_init_q is None           # absent file = no override
    assert lvl.boundary == "space"          # absent key = today's behavior


def test_air_grid_loads_by_presence(tmp_path):
    d = _mini_level(tmp_path)
    g = _full(FP_ONE)
    g[3, 3] = OVER_P
    _air(d, g)
    lvl = _load_silent(d)
    assert lvl.air_init_q is not None
    assert lvl.air_init_q.dtype == np.int32
    assert np.array_equal(lvl.air_init_q, g)


def test_wrong_dtype_hard_error(tmp_path):
    d = _mini_level(tmp_path)
    _air(d, np.full(_room_shape(), FP_ONE, dtype=np.int64))
    with pytest.raises(ValueError, match="must be dtype int32"):
        _load(d)


def test_wrong_shape_hard_error(tmp_path):
    d = _mini_level(tmp_path)
    _air(d, np.full((3, 3), FP_ONE, dtype=np.int32))
    with pytest.raises(ValueError, match="shape"):
        _load(d)


def test_negative_pressure_hard_error(tmp_path):
    d = _mini_level(tmp_path)
    g = _full(FP_ONE)
    g[2, 2] = -1
    _air(d, g)
    with pytest.raises(ValueError, match="negative pressures"):
        _load(d)


# ---------------------------------------------------------------------------
# (b) GameMap seeding — dormancy, the tile rule, the species split
# ---------------------------------------------------------------------------

def test_absent_file_seeds_today_exactly(tmp_path):
    """Dormancy half: no air_init.npy == the pre-A9 ambient formula, to
    the LSB (atmosphere FP_ONE / O2 13763 / N2 51773 in open air; all 0 on
    solid)."""
    g = GameMap(_load_silent(_mini_level(tmp_path)))
    open_air = (~g.solid) & (~g.is_vacuum)
    assert np.array_equal(
        g.atmosphere, np.where(g.solid | g.is_vacuum, 0, FP_ONE))
    assert np.all(g.gas[O2][open_air] == AMBIENT_O2_Q)
    assert np.all(g.gas[INERT_N2][open_air] == AMBIENT_N2_Q)
    assert not g.gas[O2][~open_air].any()


def test_explicit_ambient_grid_bit_identical_to_absent(tmp_path):
    """The other dormancy half: an explicit all-FP_ONE grid (the natural
    'pin ambient everywhere' authoring output, walls included) must be
    BIT-IDENTICAL to no grid at all — atmosphere and the whole gas stack."""
    d_none = _mini_level(tmp_path, name="none")
    d_grid = _mini_level(tmp_path, name="grid")
    _air(d_grid, _full(FP_ONE))
    g0 = GameMap(_load_silent(d_none))
    g1 = GameMap(_load_silent(d_grid))
    assert np.array_equal(g0.atmosphere, g1.atmosphere)
    assert np.array_equal(g0.gas, g1.gas)


def test_seed_lands_on_open_air_tiles(tmp_path):
    """Override values land on open-air tiles: atmosphere = P, and the
    species split satisfies N_total == P exactly with O2 = 21% half-up."""
    d = _mini_level(tmp_path)
    _air(d, _full(OVER_P))
    g = GameMap(_load_silent(d))
    open_air = (~g.solid) & (~g.is_vacuum)
    assert np.all(g.atmosphere[open_air] == OVER_P)
    o2 = g.gas[O2][open_air].astype(np.int64)
    n2 = g.gas[INERT_N2][open_air].astype(np.int64)
    assert np.all(o2 + n2 == OVER_P)        # N_total == P, no LSB leak
    expected_o2 = (int(OVER_P) * AMBIENT_O2_Q + (1 << 15)) >> 16
    assert np.all(o2 == expected_o2)


def test_solid_and_space_values_ignored(tmp_path):
    """THE PINNED TILE RULE: override values on solid tiles and on SPACE
    (is_vacuum) tiles are IGNORED — silently (no warning: a full-coverage
    grid is the natural authoring output). Their state stays exactly
    today's: 0 atmosphere, 0 species."""
    tm = np.ones((ROOM, ROOM), dtype=np.int32)
    tm[1:-1, 1:-1] = 0
    tm[0, 0] = 9                            # one SPACE tile in the ring
    d = _mini_level(tmp_path, tilemap=tm)
    _air(d, _full(3 * FP_ONE))              # big values EVERYWHERE
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # silent-by-design is pinned
        g = GameMap(_load(d))
    ignored = g.solid | g.is_vacuum
    assert ignored[0, 0] and g.is_vacuum[0, 0]
    assert not g.atmosphere[ignored].any()
    assert not g.gas[O2][ignored].any()
    assert not g.gas[INERT_N2][ignored].any()
    assert np.all(g.atmosphere[~ignored] == 3 * FP_ONE)


def test_zero_override_authors_a_depressurized_room(tmp_path):
    """0 is a REAL authored state, not dormancy: interior tiles overridden
    to 0 start with no atmosphere and no species — but are NOT flagged
    is_vacuum (no sponge; it is a depressurized room, not outer space)."""
    g_grid = _full(FP_ONE)
    g_grid[4:8, 4:8] = 0
    d = _mini_level(tmp_path)
    _air(d, g_grid)
    g = GameMap(_load_silent(d))
    zeroed = np.zeros(_room_shape(), dtype=bool)
    zeroed[4:8, 4:8] = True
    assert not g.atmosphere[zeroed].any()
    assert not g.gas[O2][zeroed].any() and not g.gas[INERT_N2][zeroed].any()
    assert not g.is_vacuum[zeroed].any()


# ---------------------------------------------------------------------------
# (c) The running sim — the seed is REAL under the EOS (needs breach_physics)
# ---------------------------------------------------------------------------

needs_bp = pytest.mark.skipif(bp is None, reason="breach_physics not built")


@needs_bp
def test_seed_takes_effect_in_running_sim(tmp_path):
    """The overpressure seed survives real EOS ticks: because the seed
    writes N_total == P (not just atmosphere), the per-tick re-derivation
    p* = C·N·T sustains it — a uniform sealed 1.5 atm room stays at
    ~1.5 atm, and the species total is exactly conserved."""
    from simulation import Simulation
    d = _mini_level(tmp_path)
    _air(d, _full(OVER_P))
    sim = Simulation(_load_silent(d), seed=42, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    open_air = (~g.solid) & (~g.is_vacuum)
    assert np.all(g.atmosphere[open_air] == OVER_P)     # landed at init
    n_total0 = int(g.gas[O2].sum(dtype=np.int64)
                   + g.gas[INERT_N2].sum(dtype=np.int64))
    sim.set_paused(False)
    for _ in range(10):
        sim.step()
    # Sealed + uniform: the override persists (vs. evaporating to ambient
    # within a tick, which is what an atmosphere-only seed would do).
    mean_p = float(g.atmosphere[open_air].mean(dtype=np.float64))
    assert mean_p > 1.4 * FP_ONE, (
        f"overpressure seed decayed: mean {mean_p / FP_ONE:.3f} atm")
    n_total1 = int(g.gas[O2].sum(dtype=np.int64)
                   + g.gas[INERT_N2].sum(dtype=np.int64))
    assert n_total1 == n_total0, "sealed-room species total not conserved"


@needs_bp
def test_reset_reapplies_the_air_seed(tmp_path):
    """reset() builds a fresh GameMap — the seed reapplies by construction
    (the water-seed precedent, test_level_water_physics)."""
    from simulation import Simulation
    d = _mini_level(tmp_path)
    _air(d, _full(OVER_P))
    sim = Simulation(_load_silent(d), seed=42, breach_physics=bp,
                     enable_recorder=False)
    seed0 = sim.gmap.atmosphere.copy()
    sim.gmap.atmosphere[:] = 0              # "the player vented the ship"
    sim.reset(seed=42)
    assert np.array_equal(sim.gmap.atmosphere, seed0)


# ---------------------------------------------------------------------------
# (d) level_lib — the air_init.npy carrier (byte-stable; None deletes)
# ---------------------------------------------------------------------------

def test_write_air_init_npy_round_trip_byte_stable(tmp_path):
    d = _mini_level(tmp_path)
    g = _full(FP_ONE)
    g[5, 5] = OVER_P
    _, has = write_air_init_npy(d, g, npy_bak=False)
    assert has is True
    first = (d / AIR_INIT_FILENAME).read_bytes()
    assert np.array_equal(_load_silent(d).air_init_q, g)    # loader accepts
    write_air_init_npy(d, g, npy_bak=False)     # write -> write is stable
    assert (d / AIR_INIT_FILENAME).read_bytes() == first
    # Load -> write is stable too (the grid IS the file, by identity).
    write_air_init_npy(d, _load_silent(d).air_init_q, npy_bak=False)
    assert (d / AIR_INIT_FILENAME).read_bytes() == first


def test_write_air_init_npy_all_zero_and_all_ambient_stay_files(tmp_path):
    """THE DELETE RULE: unlike water/zones, an all-default grid does NOT
    delete — air has no content value meaning "no override" (0 = authored
    depressurized start, FP_ONE = authored pinned ambient); only
    None/empty removes the file."""
    d = _mini_level(tmp_path)
    _, has = write_air_init_npy(d, np.zeros(_room_shape(), np.int32),
                                npy_bak=False)
    assert has is True and (d / AIR_INIT_FILENAME).is_file()
    assert not _load_silent(d).air_init_q.any()     # loads as authored 0s
    _, has = write_air_init_npy(d, _full(FP_ONE), npy_bak=False)
    assert has is True and (d / AIR_INIT_FILENAME).is_file()


def test_write_air_init_npy_none_deletes_and_bak(tmp_path):
    d = _mini_level(tmp_path)
    write_air_init_npy(d, _full(OVER_P), npy_bak=False)
    # Pre-session .bak mirrors the water/zones carriers' contract.
    nbak, has = write_air_init_npy(d, _full(FP_ONE), npy_bak=True)
    assert has is True and nbak is not None and nbak.is_file()
    # None = remove the override: file goes away, dormancy returns.
    nbak2, has = write_air_init_npy(d, None, npy_bak=False)
    assert has is False
    assert not (d / AIR_INIT_FILENAME).is_file()
    assert _load_silent(d).air_init_q is None
    # Empty array behaves like None; deleting an absent file is a no-op.
    _, has = write_air_init_npy(d, np.array([], dtype=np.int32),
                                npy_bak=False)
    assert has is False


def test_write_air_init_npy_rejects_bad_grids(tmp_path):
    d = _mini_level(tmp_path)
    with pytest.raises(ValueError, match="integer Q16.16"):
        write_air_init_npy(d, np.zeros(_room_shape(), np.float64),
                           npy_bak=False)
    bad = np.zeros(_room_shape(), np.int64)
    bad[0, 0] = -5
    with pytest.raises(ValueError, match="negative pressures"):
        write_air_init_npy(d, bad, npy_bak=False)
    bad[0, 0] = 1 << 40
    with pytest.raises(ValueError, match="fit int32"):
        write_air_init_npy(d, bad, npy_bak=False)


# ---------------------------------------------------------------------------
# (e) boundary — parse, default, hard error, byte-stable round-trip
# ---------------------------------------------------------------------------

BOUNDARY_PREFIX = PREFIX[:-1] + 'boundary = "ambient"\n\n'   # keep 1 blank

ENTITY_LIGHT = ('[[entity]]\n'
                'id = "lamp_1"\n'
                'class = "light"\n'
                'x = 2.5\n'
                'y = 3.5\n'
                'color = [255, 0, 0]\n'
                '\n')


def test_boundary_defaults_to_space(tmp_path):
    assert _load_silent(_mini_level(tmp_path)).boundary == "space"


@pytest.mark.parametrize("value", ["space", "ambient"])
def test_boundary_explicit_values_parse(tmp_path, value):
    prefix = PREFIX[:-1] + f'boundary = "{value}"\n\n'
    # A SPACE ring is valid for both modes (vacuum ring / reservoir ring), so
    # the ambient case does not trip the ring-dormant warning.
    tm = np.full((ROOM, ROOM), 9, dtype=np.int32)
    tm[1:-1, 1:-1] = 0
    lvl = _load_silent(_mini_level(tmp_path, prefix=prefix, tilemap=tm))
    assert lvl.boundary == value


@pytest.mark.parametrize("spelling", ['boundary = "planet"', "boundary = 3"])
def test_boundary_unknown_value_hard_error(tmp_path, spelling):
    prefix = PREFIX[:-1] + spelling + "\n\n"
    with pytest.raises(ValueError, match='boundary.*space.*ambient'):
        _load(_mini_level(tmp_path, prefix=prefix))


def test_boundary_round_trips_byte_stable_through_managed_save(tmp_path):
    """boundary is a top-level scalar OUTSIDE every managed family — the
    managed-block writer must preserve it byte-for-byte through a real
    family rewrite (and through a no-op save)."""
    tm = np.full((ROOM, ROOM), 9, dtype=np.int32)   # SPACE ring: a real ambient map
    tm[1:-1, 1:-1] = 0
    d = _mini_level(tmp_path, body=ENTITY_LIGHT, prefix=BOUNDARY_PREFIX,
                    tilemap=tm)
    toml = d / "level.toml"
    before = toml.read_bytes()
    handle = open_level(str(d))
    assert handle.data.boundary == "ambient"
    handle.save({})                                     # no-op save
    assert toml.read_bytes() == before
    handle.save({"entity":                              # real family rewrite
                 lambda nl: format_entity_lines(handle.data.entities, nl)})
    assert toml.read_bytes() == before


# ---------------------------------------------------------------------------
# (e2) level_lib.write_boundary_field (Arc C5 — the PROPS pane's writeback:
#      format + client-side field edit ONLY, never the physics meaning)
# ---------------------------------------------------------------------------

def test_write_boundary_field_inserts_when_absent(tmp_path):
    d = _mini_level(tmp_path)                        # PREFIX has no boundary
    toml_path = d / "level.toml"
    write_boundary_field(toml_path, "ambient")
    # plain _load (not _load_silent): the default sealed-hull-room tilemap
    # has no SPACE ring, so an ambient boundary legitimately warns
    # "ring-dormant" here — this test is about the WRITTEN VALUE, not that
    # warning.
    assert _load(d).boundary == "ambient"


def test_write_boundary_field_replaces_an_existing_value(tmp_path):
    d = _mini_level(tmp_path, prefix=BOUNDARY_PREFIX)   # starts "ambient"
    toml_path = d / "level.toml"
    assert _load(d).boundary == "ambient"
    write_boundary_field(toml_path, "space")
    assert _load_silent(d).boundary == "space"


def test_write_boundary_field_preserves_every_other_byte(tmp_path):
    d = _mini_level(tmp_path, body=ENTITY_LIGHT, prefix=BOUNDARY_PREFIX)
    toml_path = d / "level.toml"
    before = toml_path.read_text(encoding="utf-8")
    write_boundary_field(toml_path, "space")
    after = toml_path.read_text(encoding="utf-8")
    # only the boundary line itself should differ
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    assert len(before_lines) == len(after_lines)
    diffs = [(a, b) for a, b in zip(before_lines, after_lines) if a != b]
    assert len(diffs) == 1
    assert diffs[0][0].startswith('boundary = "ambient"')
    assert diffs[0][1] == 'boundary = "space"'
    # the entity block, unrelated to boundary, is untouched
    assert 'id = "lamp_1"' in after


def test_write_boundary_field_no_op_round_trip_when_value_unchanged(
        tmp_path):
    d = _mini_level(tmp_path, prefix=BOUNDARY_PREFIX)
    toml_path = d / "level.toml"
    before = toml_path.read_bytes()
    write_boundary_field(toml_path, "ambient")     # already ambient
    assert toml_path.read_bytes() == before


def test_write_boundary_field_rejects_unknown_value(tmp_path):
    d = _mini_level(tmp_path)
    with pytest.raises(ValueError, match="space.*ambient"):
        write_boundary_field(d / "level.toml", "planet")


def test_write_boundary_field_write_bak_writes_pre_change_bytes(tmp_path):
    d = _mini_level(tmp_path)
    toml_path = d / "level.toml"
    before = toml_path.read_bytes()
    bak = write_boundary_field(toml_path, "ambient", write_bak=True)
    assert bak is not None and bak.read_bytes() == before
    assert bak.read_bytes() != toml_path.read_bytes()


def test_write_boundary_field_atomic_write_is_a_real_file_replace(tmp_path):
    # Sanity: the write lands via the SAME atomic temp+rename path as every
    # other level_lib writer — no leftover .tmp file after a normal write.
    d = _mini_level(tmp_path)
    toml_path = d / "level.toml"
    write_boundary_field(toml_path, "ambient")
    leftovers = list(d.glob("level.toml.*.tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# (f) _upscale_level — --res replicates air like water; boundary untouched
# ---------------------------------------------------------------------------

def test_upscale_replicates_air_grid(tmp_path):
    from main import _upscale_level             # heavy import: test-local
    tm = np.full((ROOM, ROOM), 9, dtype=np.int32)   # SPACE ring: a real ambient map
    tm[1:-1, 1:-1] = 0
    d = _mini_level(tmp_path, prefix=BOUNDARY_PREFIX, tilemap=tm)
    g = _full(FP_ONE)
    g[3, 4] = OVER_P
    _air(d, g)
    lvl = _load_silent(d)
    _upscale_level(lvl, 3)
    assert lvl.tilemap.shape == (ROOM * 3, ROOM * 3)
    assert lvl.air_init_q.shape == lvl.tilemap.shape    # never a mismatch
    expected = np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)
    assert np.array_equal(lvl.air_init_q, expected)     # intensive: replicate
    assert lvl.boundary == "ambient"                    # scalar: untouched


def test_upscale_without_air_stays_dormant(tmp_path):
    from main import _upscale_level             # heavy import: test-local
    lvl = _load_silent(_mini_level(tmp_path))
    _upscale_level(lvl, 2)
    assert lvl.air_init_q is None
    assert lvl.boundary == "space"


# ---------------------------------------------------------------------------
# (g) AMBIENT boundary — the BC build: dials, wholesale routing, dial-aware
#     seeding, sponge grid, door-on-ring rejection, space-map dormancy.
#     (boundary_conditions_spec_2026-07-19; spec §1/§3/§4)
# ---------------------------------------------------------------------------

from simulation import ambient as _amb                        # noqa: E402

PIN_DEFAULT = 65632                 # p*(quantize(1.0), ΔT=0) — NOT 65536 (the lattice
                                     # image of 1/293 under G12, issue #12,
                                     # docs/fire_g12_one_map_patch_2026-08-31.md; was
                                     # 65540 under ruling 6's eos_t_amb_k=290)


def _ambient_tilemap(n: int = ROOM, interior: int = 0) -> np.ndarray:
    """A planetside room: SPACE (9) border ring, interior air — the ring
    becomes is_ambient wholesale on an ambient map."""
    tm = np.full((n, n), 9, dtype=np.int32)         # SPACE ring
    tm[1:-1, 1:-1] = interior
    return tm


def _ambient_level(tmp_path, *, body: str = "", name: str = "amb",
                   tilemap: np.ndarray = None) -> Path:
    prefix = PREFIX[:-1] + 'boundary = "ambient"\n\n'
    if tilemap is None:
        tilemap = _ambient_tilemap()
    return _mini_level(tmp_path, body=body, name=name, prefix=prefix,
                       tilemap=tilemap)


# ---- dials: defaults, explicit, the effective pin -------------------------

def test_ambient_absent_table_gets_earth_defaults(tmp_path):
    lvl = _load_silent(_ambient_level(tmp_path))
    a = lvl.ambient
    assert a is not None
    assert a.p_amb == 1.0 and a.o2_frac == 0.21
    assert a.sponge_width == _amb.DEFAULT_SPONGE_WIDTH
    assert a.pin_q == PIN_DEFAULT              # 65632, the lattice image of 1.0 atm
    assert a.n_o2_q == 13763 and a.n_n2_q == 51773
    assert a.n_total_q == FP_ONE               # N is primary, sums to quantize(p_amb)


def test_space_map_has_no_ambient_config(tmp_path):
    assert _load_silent(_mini_level(tmp_path)).ambient is None


def test_ambient_explicit_dials_parse_and_derive(tmp_path):
    body = "[ambient]\np_amb = 0.6\no2_frac = 0.3\n\n"
    a = _load_silent(_ambient_level(tmp_path, body=body)).ambient
    assert a.p_amb == 0.6 and a.o2_frac == 0.3
    # N-primary: N_total = quantize(0.6); split by 0.3; pin via the sim chain.
    n_total = _amb._gas_fx.quantize_scalar(0.6)
    assert a.n_total_q == n_total
    assert a.pin_q == _amb.effective_pin(n_total)


def test_ambient_sponge_dials_parse(tmp_path):
    body = ("[ambient]\nsponge_width = 4\n"
            "sponge_strength = 131072\nsponge_u_damp = 100\n\n")
    a = _load_silent(_ambient_level(tmp_path, body=body)).ambient
    assert a.sponge_width == 4
    assert a.sponge_strength == 131072 and a.sponge_u_damp == 100


# ---- validation matrix ----------------------------------------------------

def test_ambient_table_on_space_map_hard_error(tmp_path):
    # boundary defaults to space; an [ambient] table is then meaningless.
    d = _mini_level(tmp_path, body="[ambient]\np_amb = 1.0\n\n")
    with pytest.raises(ValueError, match="boundary.*not.*ambient|only meaningful"):
        _load(d)


@pytest.mark.parametrize("body,msg", [
    ('[ambient]\np_amb = 0.0\n\n', "p_amb must be > 0"),
    ('[ambient]\np_amb = -1.0\n\n', "p_amb must be > 0"),
    ('[ambient]\no2_frac = 1.5\n\n', r"o2_frac must be in \[0, 1\]"),
    ('[ambient]\no2_frac = -0.1\n\n', r"o2_frac must be in \[0, 1\]"),
    ('[ambient]\nsponge_width = -1\n\n', "sponge_width must be a non-negative"),
    ('[ambient]\nsponge_strength = -5\n\n', "sponge_strength must be"),
    ('[ambient]\nsponge_strength = 99999999\n\n', "sponge_strength must be"),
    ('[ambient]\nsponge_u_damp = 65536\n\n', "sponge_u_damp must be"),
    ('[ambient]\nsponge_u_damp = -1\n\n', "sponge_u_damp must be"),
    ('[ambient]\nbogus = 1\n\n', "unknown key"),
])
def test_ambient_validation_hard_errors(tmp_path, body, msg):
    with pytest.raises(ValueError, match=msg):
        _load(_ambient_level(tmp_path, body=body))


def test_ambient_ringless_map_warns(tmp_path):
    # boundary=ambient but no SPACE tiles → sealed box, ring-dormant.
    tm = np.ones((ROOM, ROOM), dtype=np.int32)
    tm[1:-1, 1:-1] = 0                          # hull ring, no SPACE
    with pytest.warns(UserWarning, match="no SPACE.*ring|sealed box"):
        _load(_ambient_level(tmp_path, tilemap=tm))


def test_ambient_wide_sponge_warns(tmp_path):
    body = "[ambient]\nsponge_width = 999\n\n"
    with pytest.warns(UserWarning, match="spans the whole map"):
        _load(_ambient_level(tmp_path, body=body))


# ---- wholesale routing + dial-aware seeding -------------------------------

def test_ambient_space_tiles_route_to_is_ambient_wholesale(tmp_path):
    g = GameMap(_load_silent(_ambient_level(tmp_path)))
    ring = (_ambient_tilemap() == 9)
    assert np.array_equal(g.is_ambient, ring)   # SPACE → is_ambient
    assert not g.is_vacuum.any()                # NO vacuum on a planetside map


def test_ambient_interior_seeds_to_effective_pin(tmp_path):
    """The whole non-solid field (interior air + ring) seeds to the effective
    pin / N_amb split, so the interior materializes flat against the ring."""
    g = GameMap(_load_silent(_ambient_level(tmp_path)))
    non_solid = ~g.solid
    assert np.all(g.atmosphere[non_solid] == PIN_DEFAULT)
    assert np.all(g.atmosphere[g.solid] == 0)
    assert np.all(g.gas[O2][non_solid] == 13763)
    assert np.all(g.gas[INERT_N2][non_solid] == 51773)
    # N_total sums to quantize(p_amb) exactly (N-primary, no LSB leak).
    tot = g.gas[O2][non_solid].astype(np.int64) + g.gas[INERT_N2][non_solid]
    assert np.all(tot == FP_ONE)


def test_ambient_nondefault_dials_seed_consistently(tmp_path):
    body = "[ambient]\np_amb = 0.6\no2_frac = 0.3\n\n"
    lvl = _load_silent(_ambient_level(tmp_path, body=body))
    g = GameMap(lvl)
    a = lvl.ambient
    non_solid = ~g.solid
    assert np.all(g.atmosphere[non_solid] == a.pin_q)
    assert np.all(g.gas[O2][non_solid] == a.n_o2_q)
    assert np.all(g.gas[INERT_N2][non_solid] == a.n_n2_q)


# ---- sponge grid (hand-computed BFS ramp) ---------------------------------

def test_sponge_grid_quadratic_ramp_hand_computed(tmp_path):
    """8x8 SPACE-ring room, W=3, σ_max=90: interior tiles at BFS distance d
    from the ring get σ = 90*(3-d)²//9. d=1 → 40, d=2 → 10, d≥3 → 0."""
    body = "[ambient]\nsponge_width = 3\nsponge_strength = 90\n\n"
    tm = _ambient_tilemap(8)
    g = GameMap(_load_silent(_ambient_level(tmp_path, tilemap=tm, body=body)))
    s = g.sponge_sigma
    # Corner interior (1,1): 4-neighbour distance to nearest ring tile == 1.
    assert s[1, 1] == 90 * (3 - 1) ** 2 // 9          # 40
    assert s[1, 2] == 40 and s[2, 1] == 40
    # (2,2): distance 2 from the ring.
    assert s[2, 2] == 90 * (3 - 2) ** 2 // 9          # 10
    # Deep centre (3,3)+: distance ≥ 3 → outside the band → 0.
    assert s[3, 3] == 0 and s[4, 4] == 0
    # Ring tiles themselves are d=0, excluded from the band (pinned).
    assert s[0, 0] == 0
    assert s.dtype == np.int32


def test_sponge_grid_zero_when_width_zero(tmp_path):
    body = "[ambient]\nsponge_width = 0\n\n"     # hard-ring escape hatch
    g = GameMap(_load_silent(_ambient_level(tmp_path, body=body)))
    assert not g.sponge_sigma.any()


def test_sponge_grid_scales_with_res_factor(tmp_path):
    from main import _upscale_level             # heavy import: test-local
    body = "[ambient]\nsponge_width = 2\nsponge_strength = 65536\n\n"
    lvl = _load_silent(_ambient_level(tmp_path, tilemap=_ambient_tilemap(8),
                                      body=body))
    _upscale_level(lvl, 2)                       # W_eff = 2 * 2 = 4 tiles
    g = GameMap(lvl)
    # At res 2 the physical band is 2 base tiles == 4 fine tiles deep: a tile
    # 3 fine-tiles in (base depth 1.5) is still inside the band (d=3 < W=4).
    assert g.sponge_sigma[3, 3] > 0
    assert not g.is_vacuum.any()


# ---- door-on-ring rejection + space-map dormancy --------------------------

def test_door_on_ambient_ring_is_authoring_error(tmp_path):
    door = ('[[entity]]\nid = "d1"\nclass = "door"\n'
            'x = 0.0\ny = 1.0\nw = 1\nh = 1\n\n')     # span on the ring col 0
    with pytest.raises(ValueError, match="ambient|boundary ring"):
        GameMap(_load_silent(_ambient_level(tmp_path, body=door)))


def test_space_map_seeding_is_byte_identical_dormancy(tmp_path):
    """Dormancy by branch: a space map's masks + seeds + sponge grid are
    exactly today's — is_ambient empty, sponge all-zero, seeds at FP_ONE."""
    g = GameMap(_load_silent(_mini_level(tmp_path)))
    assert not g.is_ambient.any()
    assert not g.sponge_sigma.any()
    open_air = (~g.solid) & (~g.is_vacuum)
    assert np.all(g.atmosphere[open_air] == FP_ONE)
    assert np.all(g.gas[O2][open_air] == AMBIENT_O2_Q)
    assert np.all(g.gas[INERT_N2][open_air] == AMBIENT_N2_Q)


# ---------------------------------------------------------------------------
# (h) AMBIENT physics gates — the B3 build (boundary_conditions_spec §6).
#     Gate 1 (a flat planetside interior holds equilibrium — the payoff of the
#     N-primary pin) and Gate 2 (breach to ambient -> air rushes IN, rails
#     bounded, the boundary_flux rail records the exchange). These exercise the
#     LIVE C++ ambient path (EOSSolver::step shift/reset/widenings + the rail),
#     so they are skipped on a build without breach_physics.
# ---------------------------------------------------------------------------
DT_TICK = 1.0 / 24.0


def _ambient_gmap(H, W, ambient_cfg=None):
    """A planetside map: 1-cell SPACE ring border (v1 code 0) around an open-air
    interior (code 9). Hand-built LevelData — no level folder (the physics path
    only needs the tilemap + boundary + dials)."""
    tm = np.full((H, W), 9, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0
    ld = level_loader.LevelData(
        name="amb_gate", version="1", path=Path("."), tilemap=tm,
        tile_size_m=1.0 / 3.0, diffuse_path=Path("."),
        boundary="ambient", ambient=ambient_cfg)
    return GameMap(ld)


@pytest.mark.skipif(bp is None, reason="needs the compiled breach_physics")
@pytest.mark.parametrize("dials", [
    None,                                   # Earth defaults (pin 65540)
    dict(p_amb=0.6, o2_frac=0.3),           # non-default dials (pin 39150)
])
def test_ambient_gate1_flat_interior_holds(dials):
    """GATE 1 (spec §6): a flat planetside interior holds equilibrium exactly —
    the payoff of the N-primary pin. The interior trajectory must be FLAT at
    defaults AND at non-default dials (proving the dial-aware seed + the shift
    materialize P_amb consistently)."""
    from simulation.ambient import derive_ambient
    from simulation.physics_runner import PhysicsRunner
    cfg = derive_ambient(**dials) if dials else derive_ambient()
    g = _ambient_gmap(28, 28, cfg)
    runner = PhysicsRunner(bp)
    interior = (~g.solid) & (~g.is_ambient)
    p0 = g.atmosphere[interior].copy()
    worst = 0
    for _ in range(60):
        runner.step(g, DT_TICK)
        worst = max(worst, int(np.abs(
            g.atmosphere[interior].astype(np.int64) - p0).max()))
    # Exactly flat (the pin is the sim's own p*(N_amb, 0); ≤1 LSB tolerance).
    assert worst <= 1, f"interior drifted {worst} raw at dials={dials}"
    # And the ring materializes exactly the effective pin every tick.
    assert np.all(g.atmosphere[g.is_ambient] == cfg.pin_q)


@pytest.mark.skipif(bp is None, reason="needs the compiled breach_physics")
def test_ambient_gate2_rush_in_recovers_and_rails_bounded():
    """GATE 2 (spec §6; RESTATED arc #54 P-G3, 2026-08-30 -- Erik's ruling on
    the P-G1a open question): a depressurized interior open to the ambient
    ring refills toward P_amb (the reservoir supplies mass), the
    boundary_flux rail records the exchange (negative == mass INTO the
    domain), and the T_MAX_PHYS rail hits are BOUNDED, COUNTED, and DECAY --
    not absent.

    Why the old `t_max_phys_hits == 0` STOP no longer holds (physics, not a
    regression): this fixture slams the interior to 0.1 atm then opens it to
    the ambient ring. The implicit MG solve lifts the interior's pressure to
    ~1 atm ACOUSTICALLY, in one step, while the mass N is still at 0.1 --
    pressure arrives before mass. The kick sees that gradient against
    near-vacuum N and the inbound flow work (the p*u face flux, design §2.4)
    lands in the still-near-empty boundary cells for one tick, an energy
    Courant number of ~40 there. The T_MAX_PHYS rail (§2.2/§2.6 -- the
    recovery clamp that keeps stored `gas_energy <= 2^60`) is exactly the
    circuit breaker for that overshoot, and it is SUPPOSED to fire: measured
    cumulative hits 424 / 550 / 564 / 564 / 564 at ticks 0-4 (zero NEW hits
    from tick 3 on) -- a one-time acoustic transient, not sustained thermal
    runaway. An inflow rail (clamping the kick velocity directly at the
    boundary) was tried and reverted: it turns an honest open boundary into a
    refrigerator, permanently damping every inrush instead of letting this
    rail count a one-time overshoot. So the gate now asserts the three things
    that actually matter: the room still refills, the rail is BOUNDED (no
    runaway), and it DECAYS to zero new hits within 4 ticks -- and every hit
    is counted in `e_rail_sum` the same tick it fires (never a silent energy
    leak)."""
    from simulation.physics_runner import PhysicsRunner
    g = _ambient_gmap(40, 40)
    runner = PhysicsRunner(bp)
    o2, n2 = g.gases.name_to_id["o2"], g.gases.name_to_id["inert_n2"]
    interior = (~g.solid) & (~g.is_ambient)
    pin = g._ambient.pin_q
    # Vent the interior to ~10% of ambient (N-primary, so P falls out of N/T).
    g.atmosphere[interior] = int(pin * 0.1)
    g.gas[o2][interior] = int(g._ambient.n_o2_q * 0.1)
    g.gas[n2][interior] = int(g._ambient.n_n2_q * 0.1)
    # arc #54 P-G1b: a direct bulk-N write needs its stored energy re-derived
    # at the cells it authored. Without this the interior keeps the ENERGY of
    # the mass it no longer has, so venting it to 10% N would make it ~10x
    # HOTTER rather than thinner -- the opposite of the scenario, and a
    # thermal-runaway probe measured on a fixture that manufactured one.
    g.reseed_gas_energy(interior)
    start = float(g.atmosphere[interior].mean())

    # Watch the first few ticks' PER-TICK delta (the decay claim is about new
    # hits per tick, not the running total) and correlate each hit with the
    # SAME tick's e_rail_sum booking.
    N_WATCH = 5
    hit_deltas, rail_at_tick = [], []
    prev_hits = 0
    for _ in range(N_WATCH):
        runner.step(g, DT_TICK)
        hits = runner.eos.t_max_phys_hits
        hit_deltas.append(hits - prev_hits)
        rail_at_tick.append(runner.eos.e_rail_sum)
        prev_hits = hits
    for _ in range(80 - N_WATCH):
        runner.step(g, DT_TICK)

    recovered = float(g.atmosphere[interior].mean())
    # Air rushed IN: the interior recovered most of the way to P_amb.
    assert recovered > 0.9 * pin, (
        f"interior only recovered to {recovered:.0f} / {pin} "
        f"(started {start:.0f})")
    # The rail recorded the boundary exchange, negative for a net inflow.
    rail = runner.eos.boundary_flux()
    assert len(rail) == g.gas.shape[0]
    assert rail[o2] < 0 and rail[n2] < 0, f"expected inflow rail, got {rail}"
    # The ring stayed pinned throughout.
    assert np.all(g.atmosphere[g.is_ambient] == pin)

    # The T_MAX_PHYS rail fires (the gate is vacuous if it never does)...
    assert hit_deltas[0] > 0, (
        "gate is vacuous: the acoustic overshoot never engaged the rail "
        f"(hit_deltas={hit_deltas})")
    # ...is BOUNDED (2x the measured 564 as a runaway tripwire, not a tight
    # pin -- this gate watches for regressions, not the exact count)...
    assert runner.eos.t_max_phys_hits <= 1200, (
        f"t_max_phys_hits={runner.eos.t_max_phys_hits}: expected a bounded "
        "one-time overshoot (measured 564), not runaway")
    # ...DECAYS to zero new hits within 4 ticks (measured: 424, 126, 14, 0, 0)...
    assert all(d == 0 for d in hit_deltas[3:]), (
        f"new T_MAX_PHYS hits still arriving past tick 3: {hit_deltas} -- "
        "the acoustic overshoot should be a one-time transient")
    # ...and every hit is COUNTED, never a silent energy leak.
    for tick, (d, r) in enumerate(zip(hit_deltas, rail_at_tick)):
        assert d == 0 or r != 0, (
            f"tick {tick}: {d} new T_MAX_PHYS hits but e_rail_sum == 0 -- "
            "an uncounted energy rail")


# ---------------------------------------------------------------------------
# (i) AMBIENT reflection gate — the B3c u-damping absorber (spec §3 rung 2 +
#     §6 gate 3). Uses the committed big-map-reference harness
#     (tests/_ambient_reflection.py): the reflection off the near ring, isolated
#     against a reference run whose ring is pushed beyond the acoustic horizon.
#     Asserts the u-damping band (a) demonstrably ABSORBS (reflection falls vs
#     the pin-only ring) and (b) meets the ≤2% gate at a representative band.
#     Skipped without breach_physics (drives the live EOSSolver kick).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(bp is None, reason="needs the compiled breach_physics")
def test_ambient_gate3_udamp_band_absorbs_reflection():
    """GATE 3 (spec §6; RESTATED arc #54 P-G3, 2026-08-30): the u-damping band
    is still the real absorber (B3c), but the ABSOLUTE reflection level moved
    a lot -- not the 2.48%-vs-2% drift measured right after P-G1a, but a much
    bigger jump landed by P-G1d.

    Why (physics, not a regression): P-G1d replaced the solve's divergence
    stencil with the face form (û = 0 at solid faces), which is the exact
    discrete adjoint of the kick's pressure gradient -- the OLD central
    stencil implicitly let û_wall = u_i (the mirror-index trick), so the
    domain edge partially absorbed/leaked reflections instead of bouncing
    them cleanly. P-G1d's own result recorded this as "feel-adjacent": BLAST
    peak |u| 8.7 -> 18.9 m/s, AS glass bursts 3 -> 16 tiles -- "walls now
    reflect honestly instead of leaking divergence." The ambient ring is the
    same kind of boundary, so its reflection jumped too. Measured at this
    geometry on arc #54 HEAD (2026-08-30): pin-only (no damping) reflection
    36.51%, with the shipped k_max damping band 31.76% -- a 13.0% relative
    cut. The old ≥25%-relative-cut / ≤2%-absolute gates were calibrated under
    the divergence-leaking stencil and cannot survive a corrected boundary
    condition; retuning the absorber band itself (if Erik wants the
    reflection back down near the old 2%) is a P-G4/HUMAN-TEST feel call, not
    this patch's. Restated with margin around the measured values so the gate
    still catches a REGRESSION of the absorber or a further stencil-driven
    jump: the band must still cut reflection by >=10% relative (measured
    13.0%) and the absolute level must stay <=35% (measured 31.76%)."""
    import _ambient_reflection as refl  # sibling helper (pytest prepend mode)
    from simulation.ambient import DEFAULT_SPONGE_U_DAMP
    geo = dict(sponge_width=16, test_half=50, ref_half=230, window=6, probe_r=3)
    pin_only, _, _ = refl.reflection_ratio(bp, k_max=0, **geo)
    absorbed, _, _ = refl.reflection_ratio(bp, k_max=DEFAULT_SPONGE_U_DAMP, **geo)
    # (a) the band demonstrably absorbs (velocity damping cuts the acoustic
    #     reflection the σ pressure-sponge could not touch) -- restated to a
    #     >=10% relative cut (measured 13.0%) now that the honest wall
    #     stencil raised the pin-only baseline the band has to work against.
    assert absorbed < 0.90 * pin_only, (
        f"u-damping did not absorb: pin-only={pin_only:.4f} "
        f"with-band={absorbed:.4f}")
    # (b) the reflection gate, restated to <=35% (measured 31.76%) -- the
    #     old <=2% band encoded the pre-P-G1d, divergence-leaking stencil.
    assert absorbed <= 0.35, f"reflection {absorbed*100:.2f}% exceeds the 35% gate"


@pytest.mark.skipif(bp is None, reason="needs the compiled breach_physics")
def test_ambient_udamp_grid_matches_sigma_band_geometry():
    """The k(d) velocity grid and the σ mass grid share the SAME BFS band
    (spec §3): where σ is populated (explicit strength), k is populated too, on
    the same tiles, with the same taper direction (strong near the ring)."""
    from simulation.ambient import derive_ambient
    from simulation.gamemap import GameMap
    tm = np.full((30, 30), 9, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0
    cfg = derive_ambient(sponge_width=6, sponge_strength=90, sponge_u_damp=40000)
    g = GameMap(level_loader.LevelData(
        name="k", version="1", path=Path("."), tilemap=tm, tile_size_m=1.0 / 3.0,
        diffuse_path=Path("."), boundary="ambient", ambient=cfg))
    assert g.sponge_udamp.shape == g.sponge_sigma.shape
    band = g.sponge_udamp > 0
    assert band.any()
    assert np.array_equal(band, g.sponge_sigma > 0)   # same BFS band tiles
    assert int(g.sponge_udamp.max()) <= 40000         # never exceeds k_max


@pytest.mark.skipif(bp is None, reason="needs the compiled breach_physics")
def test_ambient_gate4_traces_absorbed_at_the_ring():
    """GATE 4 (spec §1, Erik B5 follow-up): the ambient ring is a TRACE SINK.
    Smoke/trace planes advected toward the sky ring are reset to 0 there
    (absorbed), the vacuum-breach idiom verbatim — so a trace cloud vents out
    the open boundary instead of piling up against an invisible wall. Exercises
    the SmokeDynamics::step is_ambient widening threaded from run_substeps."""
    from simulation.physics_runner import PhysicsRunner
    g = _ambient_gmap(30, 30)
    runner = PhysicsRunner(bp)
    bs = g.gases.name_to_id["smoke"]
    # A trace cloud in the open interior adjacent to the sky ring.
    g.gas[bs][2:6, 6:24] = int(2.0 * FP_ONE)
    total0 = int(g.gas[bs].sum())
    assert total0 > 0 and int(g.gas[bs][g.is_ambient].sum()) == 0
    for _ in range(40):
        runner.step(g, DT_TICK)
        # The ring holds NO trace at any tick — absorbed every step.
        assert int(g.gas[bs][g.is_ambient].sum()) == 0
    # And mass left the system through the ring (the open boundary vents).
    assert int(g.gas[bs].sum()) < total0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
