"""Migrate a level from format v2 to v3: SPACE moves out of the material ids.

Level format v3 (2026-10-29, Erik's ruling "rows suffice; design for 127
materials"): the tilemap CSV's codes ARE canon material ids, unbounded rows up
to the int8 material plane's 127, plus ONE reserved non-material code,
``level_loader.SPACE_CODE`` (-1) = outer space (air + vacuum), OUTSIDE the id
range. v2 used 9 for SPACE, which is also ``MAT_FOLIAGE``'s id. This tool
rewrites, in place and through ``level_lib`` (the one writer):

    v2 code                -> v3 code
    9  SPACE               -> -1 SPACE   (level_loader.SPACE_CODE)
    every material id      -> itself

bumps ``version`` "2" -> "3" (``level_lib.write_version_field``) and updates
a header code legend ("9=SPACE" -> "-1=SPACE", "v2 codes" -> "v3 codes";
``level_lib.rewrite_header_comments``); every other byte of level.toml is
preserved. ``.bak`` copies of both files are
written first. All validation happens before anything is written; it refuses
anything that is not a clean v2 level (a v3 level is "nothing to migrate" —
running twice can never turn foliage into space).

The material and vacuum planes the loader produces are unchanged by the
migration, bit for bit (``tests/test_level_format_v3.py``).

Usage:
    python tools/migrate_tilemap_v3.py <level_name|level_dir> [...]
    python tools/migrate_tilemap_v3.py --all      # every v2 level under levels/
"""
from __future__ import annotations

import re
import sys
import tomllib
from collections import Counter
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import level_lib
from level_loader import (LEVEL_FORMAT_VERSION, SPACE_CODE, V2_SPACE_CODE,
                          materials_from_tilemap)

FROM_VERSION = "2"

_LEGEND_SPACE_RE = re.compile(r"(?<![0-9-])9=SPACE")
_LEGEND_VERSION_RE = re.compile(r"\bv2 codes\b")


def fix_legend(comment: str) -> str:
    """A level.toml header comment with its v2 code legend made v3
    ("9=SPACE" -> "-1=SPACE", "v2 codes" -> "v3 codes"); others unchanged."""
    comment = _LEGEND_SPACE_RE.sub(f"{SPACE_CODE}=SPACE", comment)
    return _LEGEND_VERSION_RE.sub(f"v{LEVEL_FORMAT_VERSION} codes", comment)


def histogram(grid: np.ndarray) -> dict:
    """Sorted {code: count} over a tile grid."""
    return dict(sorted(Counter(np.asarray(grid).ravel().tolist()).items()))


def migrate_grid(grid: np.ndarray) -> np.ndarray:
    """Pure v2 -> v3 code translation (9 -> -1, ids unchanged). Validates the
    grid as v2 first (raises ValueError on a code v2 does not know)."""
    grid = np.asarray(grid)
    materials_from_tilemap(grid, FROM_VERSION)      # raises on non-v2 codes
    out = grid.astype(np.int32, copy=True)
    out[grid == V2_SPACE_CODE] = SPACE_CODE
    return out


def migrate_level(level_dir: Path) -> tuple[dict, dict]:
    """Migrate one level folder in place. Returns (before, after) histograms.

    Refuses (ValueError, nothing written) unless level.toml says version "2";
    then checks the rewrite keeps the loader's material and vacuum planes
    bit-identical before it writes anything.
    """
    level_dir = Path(level_dir)
    toml_path = level_dir / "level.toml"
    if not toml_path.is_file():
        raise ValueError(f"level.toml not found in {level_dir}")
    raw = tomllib.loads(toml_path.read_bytes().decode("utf-8"))
    version = str(raw.get("version", "")).strip()
    if version != FROM_VERSION:
        raise ValueError(
            f"{toml_path}: version is {version!r}, expected "
            f"{FROM_VERSION!r} — nothing to migrate")
    tilemap_rel = raw.get("tilemap")
    if not tilemap_rel:
        raise ValueError(f"{toml_path}: missing required 'tilemap' field")
    csv_path = level_dir / tilemap_rel
    if not csv_path.is_file():
        raise ValueError(f"Tilemap CSV not found: {csv_path}")
    grid = np.loadtxt(BytesIO(csv_path.read_bytes()), delimiter=",",
                      dtype=np.int32)
    if grid.ndim != 2:
        raise ValueError(f"{csv_path}: tilemap must be 2D, got shape "
                         f"{grid.shape}")

    new_grid = migrate_grid(grid)
    mat2, vac2 = materials_from_tilemap(grid, FROM_VERSION)
    mat3, vac3 = materials_from_tilemap(new_grid, LEVEL_FORMAT_VERSION)
    if not (np.array_equal(mat2, mat3) and np.array_equal(vac2, vac3)):
        raise AssertionError(f"{csv_path}: v3 decode differs from v2 decode")

    level_lib.write_tilemap_csv(level_dir, new_grid,
                                tilemap_rel=str(tilemap_rel), csv_bak=True)
    level_lib.write_version_field(toml_path, LEVEL_FORMAT_VERSION,
                                  write_bak=True)
    level_lib.rewrite_header_comments(toml_path, fix_legend)
    return histogram(grid), histogram(new_grid)


def main(argv: list) -> int:
    if not argv:
        print("Usage: python tools/migrate_tilemap_v3.py <level_name> ... | --all")
        return 2
    if argv == ["--all"]:
        names = []
        for d in sorted((ROOT / "levels").iterdir()):
            toml = d / "level.toml"
            if toml.is_file() and str(tomllib.loads(
                    toml.read_text(encoding="utf-8")).get("version", "")) \
                    == FROM_VERSION:
                names.append(d.name)
    else:
        names = argv
    for name in names:
        level_dir = Path(name) if Path(name).is_dir() else ROOT / "levels" / name
        if not level_dir.is_dir():
            print(f"Level folder does not exist: {level_dir}")
            return 2
        before, after = migrate_level(level_dir)
        print(f"Migrated '{name}' to level format v{LEVEL_FORMAT_VERSION} "
              f"(.bak written for the tilemap + level.toml):")
        print(f"  codes before (v{FROM_VERSION}): {before}")
        print(f"  codes after  (v{LEVEL_FORMAT_VERSION}): {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
