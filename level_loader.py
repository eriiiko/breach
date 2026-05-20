"""Level loader: reads a level folder, validates assets, returns a LevelData object.

A "level" lives at levels/<name>/ and contains:
  - level.toml      : metadata (required)
  - tilemap.csv     : physics tile grid (required)
  - diffuse.png     : base color art (required)
  - normal.png      : Laigter normal map (optional)
  - emissive_mask.png  : pixels that glow (optional)
  - emissive_bloom.png : blurred halos (optional)
  - wall_mask.png   : overrides CSV-derived walls (optional)

The CSV is the source of truth for physics. Art assets are for rendering only.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


SUPPORTED_VERSIONS = {"1"}


@dataclass
class SpawnEntry:
    """One unit spawn declared in level.toml. Fed to Simulation.add_unit."""
    name: str
    team: int           # 0 = marine, 1 = zombie
    x: float            # physics-tile x (top-left of footprint)
    y: float            # physics-tile y
    footprint: int = 3  # side length of unit's square footprint in tiles


@dataclass
class LevelData:
    name: str
    version: str
    path: Path                      # levels/<name>/
    tilemap: np.ndarray             # (rows, cols), int
    tile_size_m: float
    diffuse_path: Path              # required, exists
    normal_path: Optional[Path] = None
    emissive_mask_path: Optional[Path] = None
    emissive_bloom_path: Optional[Path] = None
    wall_mask_path: Optional[Path] = None
    background_path: Optional[Path] = None     # screen-fixed backdrop
    floor_id: int = 0
    spawns: list = field(default_factory=list)  # list[SpawnEntry]
    raw_toml: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.tilemap.shape[0])

    @property
    def width(self) -> int:
        return int(self.tilemap.shape[1])


def load(level_name: str, levels_dir: str = "levels") -> LevelData:
    """Load a level by folder name. Raises ValueError on validation failure."""
    here = Path(__file__).resolve().parent
    base = (here / levels_dir / level_name).resolve()
    if not base.is_dir():
        raise ValueError(f"Level folder does not exist: {base}")

    toml_path = base / "level.toml"
    if not toml_path.is_file():
        raise ValueError(f"level.toml not found in {base}")

    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    version = str(raw.get("version", "")).strip()
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported level.toml version '{version}'. "
            f"Supported: {sorted(SUPPORTED_VERSIONS)}"
        )

    name = raw.get("name", level_name)

    tilemap_rel = raw.get("tilemap")
    if not tilemap_rel:
        raise ValueError("level.toml missing required 'tilemap' field")
    tilemap_path = base / tilemap_rel
    if not tilemap_path.is_file():
        raise ValueError(f"Tilemap CSV not found: {tilemap_path}")
    tilemap = np.loadtxt(tilemap_path, delimiter=",", dtype=np.int32)
    if tilemap.ndim != 2:
        raise ValueError(f"Tilemap must be 2D, got shape {tilemap.shape}")

    tile_size_m = float(raw.get("tile_size_m", 0.333))

    diffuse_rel = raw.get("diffuse")
    if not diffuse_rel:
        raise ValueError("level.toml missing required 'diffuse' field")
    diffuse_path = base / diffuse_rel
    if not diffuse_path.is_file():
        raise ValueError(f"Diffuse texture not found: {diffuse_path}")

    def opt(key: str) -> Optional[Path]:
        rel = raw.get(key)
        if not rel:
            return None
        p = base / rel
        if not p.is_file():
            raise ValueError(f"{key} declared but file missing: {p}")
        return p

    spawns = []
    for i, entry in enumerate(raw.get("spawn", [])):
        try:
            spawns.append(SpawnEntry(
                name=str(entry["name"]),
                team=int(entry["team"]),
                x=float(entry["x"]),
                y=float(entry["y"]),
                footprint=int(entry.get("footprint", 3)),
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid [[spawn]] entry #{i} in {toml_path}: {e}. "
                f"Required fields: name (str), team (int), x (float), y (float)."
            )

    return LevelData(
        name=name,
        version=version,
        path=base,
        tilemap=tilemap,
        tile_size_m=tile_size_m,
        diffuse_path=diffuse_path,
        normal_path=opt("normal"),
        emissive_mask_path=opt("emissive_mask"),
        emissive_bloom_path=opt("emissive_bloom"),
        wall_mask_path=opt("wall_mask"),
        background_path=opt("background"),
        floor_id=int(raw.get("floor_id", 0)),
        spawns=spawns,
        raw_toml=raw,
    )


def materials_from_tilemap(tilemap: np.ndarray):
    """Translate CSV tile values into material grid + vacuum mask.

    Returns (material, is_vacuum) — both (H, W) numpy arrays.
    Convention:
      CSV 0     -> outer space (vacuum, MAT_AIR)
      CSV 1     -> hull wall  (MAT_HULL)
      CSV 3     -> door       (MAT_DOOR)
      else      -> interior air (MAT_AIR) with no vacuum
    """
    MAT_AIR, MAT_HULL, MAT_DOOR = 0, 1, 3
    material = np.full(tilemap.shape, MAT_AIR, dtype=np.int8)
    material[tilemap == 1] = MAT_HULL
    material[tilemap == 3] = MAT_DOOR
    is_vacuum = (tilemap == 0)
    return material, is_vacuum


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "unhcr_vessel"
    lvl = load(name)
    print(f"Loaded: {lvl.name} (v{lvl.version})")
    print(f"  Grid: {lvl.width} x {lvl.height} tiles @ {lvl.tile_size_m} m each")
    print(f"  Diffuse: {lvl.diffuse_path.name}")
    print(f"  Normal:  {lvl.normal_path.name if lvl.normal_path else '(none)'}")
    print(f"  Emissive: {lvl.emissive_mask_path.name if lvl.emissive_mask_path else '(none)'}")
    print(f"  Bloom:   {lvl.emissive_bloom_path.name if lvl.emissive_bloom_path else '(none)'}")
    print(f"  Wall mask: {lvl.wall_mask_path.name if lvl.wall_mask_path else '(derived from CSV)'}")
    print(f"  Floor:   {lvl.floor_id}")
    print(f"  Tile values: {sorted(np.unique(lvl.tilemap).tolist())}")
    mat, vac = materials_from_tilemap(lvl.tilemap)
    print(f"  Materials: hull={int((mat==1).sum())} door={int((mat==3).sum())} air={int((mat==0).sum())} vacuum={int(vac.sum())}")
