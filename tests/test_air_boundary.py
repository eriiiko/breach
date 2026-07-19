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
  unknown = hard error naming both; NO behavior change for either value in
  Arc A; a top-level scalar, so the managed-block writer round-trips it
  byte-stably.
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
                       open_level, write_air_init_npy)
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
    lvl = _load_silent(_mini_level(tmp_path, prefix=prefix))
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
    d = _mini_level(tmp_path, body=ENTITY_LIGHT, prefix=BOUNDARY_PREFIX)
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
# (f) _upscale_level — --res replicates air like water; boundary untouched
# ---------------------------------------------------------------------------

def test_upscale_replicates_air_grid(tmp_path):
    from main import _upscale_level             # heavy import: test-local
    d = _mini_level(tmp_path, prefix=BOUNDARY_PREFIX)
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
