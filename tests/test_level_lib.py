"""level_lib — THE single read+write data layer for level folders (Arc A2,
entity doc §3c).

Pins the writer contract the map editor (and later the baker, the migration
tool and ML variant generation) build on:

  - byte-stable round-trip: for EVERY levels/*/level.toml, open + save with
    no changes writes the file back byte-identically (run against tmp
    copies — the real levels are never mutated).
  - managed-block rewrite: a multi-family replace ([[spawn]] + [[light]] +
    [water] in ONE call) lands the edits, preserves every byte outside the
    managed tables, keeps CRLF, appends missing families at EOF, and an
    empty block removes a family; unknown families are refused.
  - atomicity: the write goes through a same-directory temp file +
    os.replace; a failure mid-save (formatting or replace) leaves the
    original bytes and no temp litter.
  - staleness: LevelHandle records level.toml mtime+hash at open and after
    save; check_stale() flips on an external write, stays False on a
    touch-without-change, and clears after save (the Arc C
    reload-or-overwrite seam).

Run:
    python -m pytest tests/test_level_lib.py -q
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
import tomllib
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import level_loader  # noqa: E402
from level_lib import (LevelHandle, MANAGED_FAMILIES,  # noqa: E402
                       WATER_FILENAME, format_light_lines,
                       format_spawn_lines, open_level, water_block_format,
                       write_managed_blocks, write_water_npy)
from level_loader import LightEntry, SpawnEntry  # noqa: E402

ALL_LEVELS = sorted(p.parent.name
                    for p in (ROOT / "levels").glob("*/level.toml"))


# ---------------------------------------------------------------------------
# Fixtures — tmp copies of real levels + a minimal synthetic folder
# ---------------------------------------------------------------------------

def _copy_level(name: str, tmp_path: Path) -> Path:
    """Copy one real level folder into tmp — round-trip tests run against
    the copy, never the repo's levels/ content."""
    dst = tmp_path / name
    shutil.copytree(ROOT / "levels" / name, dst)
    return dst


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


def _mini_level(tmp_path: Path, body: str = "", name: str = "mini") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "tilemap.csv").write_text(
        "\n".join(",".join("0" for _ in range(8)) for _ in range(6)) + "\n")
    _write_png(d / "diffuse.png")
    (d / "level.toml").write_text(PREFIX + body + SUFFIX,
                                  encoding="utf-8", newline="\n")
    return d


def _no_tmp_litter(d: Path) -> bool:
    return not list(d.glob("*.tmp"))


# ---------------------------------------------------------------------------
# (a) Byte-stable round-trip over every shipped level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_LEVELS)
def test_round_trip_byte_identical(name, tmp_path):
    """open + save-with-no-changes == identity, for every real level.toml
    (hand-authored formatting, comments, [art]/[bake] blocks, newline style
    — everything survives the full open/save pipeline)."""
    real_bytes = (ROOT / "levels" / name / "level.toml").read_bytes()
    d = _copy_level(name, tmp_path)
    original = (d / "level.toml").read_bytes()
    handle = open_level(str(d))
    handle.save()
    assert (d / "level.toml").read_bytes() == original
    assert _no_tmp_litter(d)
    # The reloaded copy still parses identically to the pristine original.
    assert level_loader.load(str(d)).raw_toml == handle.data.raw_toml
    # Belt and braces: the REAL level was never touched (copies only).
    assert (ROOT / "levels" / name / "level.toml").read_bytes() == real_bytes


# ---------------------------------------------------------------------------
# (b) Managed-block rewrite — multi-family, byte-preservation, families
# ---------------------------------------------------------------------------

def test_multi_family_replace_in_one_call(tmp_path):
    body = ('[[spawn]]\nname = "Old"\nteam = 0\nx = 3\ny = 4\n\n'
            '[[light]]\n# doomed comment inside a managed table\n'
            'pos = [1.0, 2.0]\ncolor = [255, 0, 0]\n\n')
    d = _mini_level(tmp_path, body)
    toml = d / "level.toml"

    spawns = [SpawnEntry("Alpha", 0, 3.0, 4.0),
              SpawnEntry("Bravo", 1, 10.5, 2.0, footprint=4)]
    lights = [LightEntry(x=2.5, y=3.5, color=(1.0, 0.0, 0.0))]
    bak = write_managed_blocks(toml, {
        "spawn": lambda nl: format_spawn_lines(spawns, nl),
        "light": lambda nl: format_light_lines(lights, nl),
        "water": water_block_format(True),
    }, write_bak=True)
    assert bak.read_bytes() == (PREFIX + body + SUFFIX).encode()

    text = toml.read_text(encoding="utf-8")
    assert text.startswith(PREFIX)          # bytes before the managed blocks
    assert SUFFIX in text                   # bytes after them
    # [water] had no existing table, so its block appends at EOF.
    assert text.endswith(SUFFIX + f'\n[water]\ndepth_map = "{WATER_FILENAME}"\n')
    raw = tomllib.loads(text)
    assert [s["name"] for s in raw["spawn"]] == ["Alpha", "Bravo"]
    assert raw["spawn"][1]["footprint"] == 4
    assert raw["light"] == [{"pos": [2.5, 3.5], "color": [255, 0, 0],
                             "intensity": 1.0, "range": 12.0,
                             "kind": "static"}]
    assert raw["water"] == {"depth_map": WATER_FILENAME}   # appended at EOF
    assert raw["bake"]["px_per_tile"] == 8                 # untouched
    assert _no_tmp_litter(d)


def test_empty_block_removes_family_and_absent_family_untouched(tmp_path):
    body = ('[[spawn]]\nname = "A"\nteam = 0\nx = 1.0\ny = 2.0\n\n'
            '[water]\ndepth_map = "water_init.npy"\n\n')
    d = _mini_level(tmp_path, body)
    toml = d / "level.toml"
    before = toml.read_bytes()
    write_managed_blocks(toml, {"water": water_block_format(False)})
    raw = tomllib.loads(toml.read_text(encoding="utf-8"))
    assert "water" not in raw
    assert raw["spawn"][0]["name"] == "A"   # untouched family stays put
    # Removing an already-absent family is a no-op write.
    write_managed_blocks(toml, {"water": water_block_format(False)})
    assert "water" not in tomllib.loads(toml.read_text(encoding="utf-8"))
    assert toml.read_bytes() != before      # the removal itself did land


def test_crlf_preserved(tmp_path):
    d = _mini_level(tmp_path)
    toml = d / "level.toml"
    toml.write_bytes(b'version = "2"\r\nname = "T"\r\n')
    write_managed_blocks(
        toml, {"spawn": lambda nl: format_spawn_lines(
            [SpawnEntry("A", 0, 1.0, 2.0)], nl)})
    data = toml.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n")
    assert tomllib.loads(data.decode())["spawn"][0]["name"] == "A"


def test_unknown_family_refused_file_untouched(tmp_path):
    d = _mini_level(tmp_path)
    toml = d / "level.toml"
    before = toml.read_bytes()
    with pytest.raises(ValueError, match="ghost"):
        write_managed_blocks(toml, {"ghost": lambda nl: []})
    assert toml.read_bytes() == before
    assert "entity" in MANAGED_FAMILIES   # A3 added it, as one registry entry


def test_editor_save_shape_round_trips_through_loader(tmp_path):
    """The Ctrl+S shape: open a level, replace all three families through
    the handle, reload through level_loader — entries land, unmanaged
    raw_toml keys are untouched, and the .bak carries pre-save bytes."""
    d = _mini_level(tmp_path)
    pre = (d / "level.toml").read_bytes()
    handle = open_level(str(d))
    assert handle.data.spawns == [] and handle.data.lights == []

    spawns = [SpawnEntry("marine_1", 0, 2.0, 3.0),
              SpawnEntry("zombie_1", 1, 6.0, 4.0, footprint=5)]
    lights = [LightEntry(x=3.5, y=2.5, color=(1.0, 0.0, 0.0),
                         kind="beacon", period_s=1.5, phase=0.5)]
    depth = np.zeros((6, 8), np.int32)
    depth[2, 2] = 65536
    _, has_water = write_water_npy(d, depth, npy_bak=False)
    handle.save({
        "spawn": lambda nl: format_spawn_lines(spawns, nl),
        "light": lambda nl: format_light_lines(lights, nl),
        "water": water_block_format(has_water),
    }, write_bak=True)

    lvl = level_loader.load(str(d))
    assert lvl.spawns == spawns
    assert lvl.lights == lights
    assert np.array_equal(lvl.water_depth_q, depth)
    assert lvl.raw_toml["bake"] == handle.data.raw_toml["bake"]
    assert lvl.raw_toml["name"] == handle.data.raw_toml["name"]
    assert (d / "level.toml.bak").read_bytes() == pre


# ---------------------------------------------------------------------------
# (c) Atomicity — temp + os.replace, no partial file on failure
# ---------------------------------------------------------------------------

def test_write_goes_through_same_dir_temp_and_replace(tmp_path, monkeypatch):
    d = _mini_level(tmp_path)
    toml = d / "level.toml"
    calls = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    write_managed_blocks(
        toml, {"spawn": lambda nl: format_spawn_lines(
            [SpawnEntry("A", 0, 1.0, 2.0)], nl)})
    assert len(calls) == 1
    src, dst = calls[0]
    assert dst == toml and src != dst
    assert src.parent == toml.parent        # same-dir temp: replace is atomic
    assert _no_tmp_litter(d)


def test_failed_replace_leaves_original_and_no_litter(tmp_path, monkeypatch):
    d = _mini_level(tmp_path)
    toml = d / "level.toml"
    before = toml.read_bytes()

    def boom(src, dst):
        raise OSError("simulated crash at the rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated"):
        write_managed_blocks(
            toml, {"spawn": lambda nl: format_spawn_lines(
                [SpawnEntry("A", 0, 1.0, 2.0)], nl)})
    assert toml.read_bytes() == before
    assert _no_tmp_litter(d)


def test_formatter_failure_mid_multi_family_touches_nothing(tmp_path):
    """A crash while building the SECOND family's block happens before any
    write: the multi-family replace is all-or-nothing."""
    d = _mini_level(tmp_path, '[[spawn]]\nname = "A"\nteam = 0\n'
                              'x = 1.0\ny = 2.0\n\n')
    toml = d / "level.toml"
    before = toml.read_bytes()

    def boom(nl):
        raise RuntimeError("simulated formatter crash")

    with pytest.raises(RuntimeError, match="formatter"):
        write_managed_blocks(toml, {
            "spawn": lambda nl: format_spawn_lines([], nl),
            "light": boom,
        })
    assert toml.read_bytes() == before
    assert _no_tmp_litter(d)


# ---------------------------------------------------------------------------
# (d) Staleness — the Arc C reload-or-overwrite seam
# ---------------------------------------------------------------------------

def test_handle_records_state_and_detects_external_write(tmp_path):
    d = _mini_level(tmp_path)
    toml = d / "level.toml"
    handle = open_level(str(d))
    assert isinstance(handle, LevelHandle)
    assert handle.toml_path == toml
    assert handle.toml_sha256 and handle.toml_mtime_ns > 0
    assert handle.check_stale() is False

    # External writer (a second session, a git checkout, a hand edit).
    toml.write_bytes(toml.read_bytes() + b"# someone else was here\n")
    os.utime(toml, ns=(handle.toml_mtime_ns + 10_000_000,
                       handle.toml_mtime_ns + 10_000_000))
    assert handle.check_stale() is True

    # Saving re-records: the handle owns the file again.
    handle.save()
    assert handle.check_stale() is False


def test_touch_without_change_is_not_stale(tmp_path):
    d = _mini_level(tmp_path)
    toml = d / "level.toml"
    handle = open_level(str(d))
    os.utime(toml, ns=(handle.toml_mtime_ns + 10_000_000,
                       handle.toml_mtime_ns + 10_000_000))
    assert handle.check_stale() is False    # mtime moved, hash says same file


def test_missing_file_is_stale(tmp_path):
    d = _mini_level(tmp_path)
    handle = open_level(str(d))
    (d / "level.toml").unlink()
    assert handle.check_stale() is True


def test_save_updates_recorded_state(tmp_path):
    d = _mini_level(tmp_path)
    handle = open_level(str(d))
    before = (handle.toml_mtime_ns, handle.toml_sha256)
    handle.save({"spawn": lambda nl: format_spawn_lines(
        [SpawnEntry("A", 0, 1.0, 2.0)], nl)})
    assert handle.toml_sha256 != before[1]
    assert handle.check_stale() is False
