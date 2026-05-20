"""Unit sprite loading and lookup for the pyray renderer.

Loads all unit sprites once at GameRenderer init. No atlas — at 17 sprites
total, individual texture loads are simpler and cheap. Holds loaded textures
in typed containers; unloads them in shutdown().

Marine sprites are 8-directional individual PNGs.
Zombie sprites are a pool of 9 type variants; assignment is stable across
reloads via unit.id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pyray as rl

from . import core

# Art root, relative to this file's location (renderer/ is inside the repo).
_REPO_ROOT = Path(__file__).parent.parent
_MARINE_DIR = _REPO_ROOT / "art" / "sprites" / "marine"
_ZOMBIE_DIR = _REPO_ROOT / "art" / "sprites" / "zombies"

MARINE_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# Pool of 8 zombie variants (excludes soldier — kept visually distinctive).
_ZOMBIE_POOL_NAMES = (
    "zombie-civilian-1",
    "zombie-civilian-2",
    "zombie-civilian-3",
    "zombie-civilian-4",
    "zombie-mechanic",
    "zombie-officer",
    "zombie-researcher",
    "zombie-scientist",
)
_ZOMBIE_SOLDIER_NAME = "zombie-soldier"


class UnitSprites:
    """Holds all unit textures, loaded once and unloaded on shutdown."""

    def __init__(self) -> None:
        self.marine: Dict[str, Optional[rl.Texture]] = {}
        self.zombies: Dict[str, Optional[rl.Texture]] = {}
        self._zombie_pool: list = []   # 8 textures (or None) for pool variants
        self._zombie_soldier: Optional[rl.Texture] = None

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all marine directions and zombie variants from disk.

        Failures are logged and stored as None — callers fall back to circle.
        Must be called after rl.init_window() (OpenGL context required).
        """
        # Marines — 8 directional sprites.
        for direction in MARINE_DIRECTIONS:
            path = _MARINE_DIR / f"marine_{direction}.png"
            self.marine[direction] = self._try_load(path)

        # Zombie pool (8 non-soldier variants).
        for name in _ZOMBIE_POOL_NAMES:
            path = _ZOMBIE_DIR / f"{name}_32.png"
            tex = self._try_load(path)
            self.zombies[name] = tex
            self._zombie_pool.append(tex)

        # Zombie soldier (the "scary one", special-cased for Zomb1).
        path = _ZOMBIE_DIR / f"{_ZOMBIE_SOLDIER_NAME}_32.png"
        self._zombie_soldier = self._try_load(path)
        self.zombies[_ZOMBIE_SOLDIER_NAME] = self._zombie_soldier

    def unload(self) -> None:
        """Unload every texture loaded by load()."""
        for tex in self.marine.values():
            if tex is not None:
                rl.unload_texture(tex)
        for tex in self.zombies.values():
            if tex is not None:
                rl.unload_texture(tex)
        self.marine.clear()
        self.zombies.clear()
        self._zombie_pool = []
        self._zombie_soldier = None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_marine(self, facing: str) -> Optional[rl.Texture]:
        """Return the marine sprite for *facing* (e.g. "N"), or None."""
        return self.marine.get(facing)

    def get_zombie(self, unit) -> Optional[rl.Texture]:
        """Return the zombie sprite for *unit*.

        - ``unit.name == "Zomb1"`` → soldier variant (visually distinctive).
        - All others → stable assignment from the 8-variant pool via unit.id.
        """
        if unit.name == "Zomb1":
            return self._zombie_soldier
        if not self._zombie_pool:
            return None
        idx = unit.id % len(self._zombie_pool)
        return self._zombie_pool[idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_load(path: Path) -> Optional[rl.Texture]:
        """Attempt to load *path* as a texture. Return None on failure."""
        try:
            tex = core.load_texture_from_path(path)
            return tex
        except Exception as exc:
            print(f"[sprites] WARN: could not load {path}: {exc}")
            return None


__all__ = ["UnitSprites", "MARINE_DIRECTIONS"]
