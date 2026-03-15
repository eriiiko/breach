"""
Game configuration loader with hot-reload support.

Usage:
    from config import CFG
    print(CFG.clock.ticks_per_second)

    # In game loop, on F5 keypress:
    CFG.reload()
"""

import tomllib
from pathlib import Path


CONFIG_PATH = Path(__file__).parent / "config.toml"


class Namespace:
    """Simple attribute-access wrapper around a dict."""

    def __init__(self, d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Namespace(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Namespace({items})"


class GameConfig:
    """Loads config.toml and exposes all values as attributes.

    Derived values are computed automatically after each load.
    Call reload() to re-read the file (bound to F5 in-game).
    """

    def __init__(self, path: Path = CONFIG_PATH):
        self._path = path
        self._load()

    def _load(self):
        with open(self._path, "rb") as f:
            raw = tomllib.load(f)

        # Expose each top-level section as a Namespace attribute
        for key, value in raw.items():
            if isinstance(value, dict):
                setattr(self, key, Namespace(value))
            else:
                setattr(self, key, value)

        # Derived values
        self.display.fine_w = self.display.map_w * self.display.coarse
        self.display.fine_h = self.display.map_h * self.display.coarse
        self.display.coarse_px = self.display.fine_tile_px * self.display.coarse

        self.clock.ticks_per_phase = int(
            self.clock.ticks_per_second * self.clock.phase_duration_seconds
        )
        self.clock.ticks_per_round = (
            self.clock.ticks_per_phase * self.clock.phases_per_round
        )

    def reload(self):
        """Re-read config.toml from disk. Call on F5."""
        self._load()
        print(f"[config] Reloaded {self._path.name}")


# Global config instance — import this everywhere
CFG = GameConfig()
