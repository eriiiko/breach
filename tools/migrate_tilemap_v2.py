"""Migrate a level's tilemap.csv from v1 generator codes to v2 canon codes.

Level format v2 (level_editor_and_format_v2_proposal §1.1): CSV codes ARE the
canon material ids from ``src/simulation/materials.py``, plus the one reserved
non-material code SPACE_CODE (9) = outer space (air + vacuum). This script
retires the generator's room-type vocabulary in place:

    v1 code                          -> v2 code
    0  outer space                   -> 9  SPACE
    1  hull wall                     -> 1  MAT_HULL
    2  generator "floor" (landmine)  -> 0  MAT_AIR
    3  door                          -> 3  MAT_DOOR
    4..8 interior-floor variants     -> 0  MAT_AIR

It rewrites ``tilemap.csv`` (preserving the file's line endings), bumps
``version = "2"`` in ``level.toml`` with a line-targeted replace (all other
content and comments preserved byte-for-byte), and writes a ``.bak`` of both
files before touching either. Refuses to run on anything that is not a clean
v1 level (already-migrated levels contain code 9 and version "2" — running
twice would turn interior air into outer space).

Usage:
    python tools/migrate_tilemap_v2.py <level_name>      # e.g. unhcr_vessel
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

from level_loader import SPACE_CODE
from simulation.materials import MAT_AIR, MAT_DOOR, MAT_HULL

# v1 generator vocabulary -> v2 canon codes. Code 2 was the generator's
# *floor*, not MAT_WOOD — the "code-2 landmine" the proposal kills — so it
# maps to air, NOT to material id 2.
V1_TO_V2 = {
    0: SPACE_CODE,   # outer space            -> SPACE (air + vacuum)
    1: MAT_HULL,     # hull wall              -> hull
    2: MAT_AIR,      # generator "floor"      -> air
    3: MAT_DOOR,     # door                   -> door
    4: MAT_AIR,      # interior-floor variants -> air
    5: MAT_AIR,
    6: MAT_AIR,
    7: MAT_AIR,
    8: MAT_AIR,
}

_VERSION_RE = re.compile(r'(?m)^(version\s*=\s*)"1"')


def histogram(grid: np.ndarray) -> dict:
    """Sorted {code: count} over a tile grid."""
    return dict(sorted(Counter(np.asarray(grid).ravel().tolist()).items()))


def migrate_grid(grid: np.ndarray) -> np.ndarray:
    """Pure v1 -> v2 code translation. Raises ValueError on non-v1 codes
    (an already-migrated grid contains 9 and is rejected here)."""
    unknown = sorted(int(c) for c in np.unique(grid) if int(c) not in V1_TO_V2)
    if unknown:
        raise ValueError(
            f"tilemap contains non-v1 codes {unknown} "
            f"(valid v1 codes: {sorted(V1_TO_V2)}) — already migrated?"
        )
    out = grid.copy()
    for v1, v2 in V1_TO_V2.items():
        out[grid == v1] = v2
    return out


def migrate_level(level_dir: Path) -> tuple[dict, dict]:
    """Migrate one level folder in place. Returns (before, after) histograms.

    All validation happens before anything is written; then ``.bak`` copies of
    both files are written, then the originals are rewritten.
    """
    level_dir = Path(level_dir)
    toml_path = level_dir / "level.toml"
    if not toml_path.is_file():
        raise ValueError(f"level.toml not found in {level_dir}")

    toml_bytes = toml_path.read_bytes()
    toml_text = toml_bytes.decode("utf-8")
    raw = tomllib.loads(toml_text)

    version = str(raw.get("version", "")).strip()
    if version != "1":
        raise ValueError(
            f"{toml_path}: version is {version!r}, expected '1' — nothing to migrate"
        )
    tilemap_rel = raw.get("tilemap")
    if not tilemap_rel:
        raise ValueError(f"{toml_path}: missing required 'tilemap' field")
    csv_path = level_dir / tilemap_rel
    if not csv_path.is_file():
        raise ValueError(f"Tilemap CSV not found: {csv_path}")

    csv_bytes = csv_path.read_bytes()
    grid = np.loadtxt(BytesIO(csv_bytes), delimiter=",", dtype=np.int32)
    if grid.ndim != 2:
        raise ValueError(f"{csv_path}: tilemap must be 2D, got shape {grid.shape}")

    before = histogram(grid)
    new_grid = migrate_grid(grid)          # raises on non-v1 codes
    after = histogram(new_grid)

    # The version bump is line-targeted so every other byte (comments, spawns,
    # art keys) survives untouched. Exactly one version line, or we refuse.
    n_version_lines = len(_VERSION_RE.findall(toml_text))
    if n_version_lines != 1:
        raise ValueError(
            f"{toml_path}: expected exactly one 'version = \"1\"' line, "
            f"found {n_version_lines}"
        )
    new_toml_text = _VERSION_RE.sub(r'\g<1>"2"', toml_text, count=1)

    # Backups FIRST, then rewrite both originals.
    csv_path.with_name(csv_path.name + ".bak").write_bytes(csv_bytes)
    toml_path.with_name(toml_path.name + ".bak").write_bytes(toml_bytes)

    newline = "\r\n" if b"\r\n" in csv_bytes else "\n"
    csv_text = newline.join(
        ",".join(str(int(v)) for v in row) for row in new_grid.tolist()
    ) + newline
    csv_path.write_bytes(csv_text.encode("ascii"))
    toml_path.write_bytes(new_toml_text.encode("utf-8"))

    return before, after


def main(argv: list) -> int:
    if len(argv) != 1:
        print("Usage: python tools/migrate_tilemap_v2.py <level_name>")
        return 2
    level_dir = ROOT / "levels" / argv[0]
    if not level_dir.is_dir():
        print(f"Level folder does not exist: {level_dir}")
        return 2
    before, after = migrate_level(level_dir)
    print(f"Migrated '{argv[0]}' to level format v2 "
          f"(.bak written for tilemap.csv + level.toml):")
    print(f"  codes before (v1): {before}")
    print(f"  codes after  (v2): {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
