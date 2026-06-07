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


def ticks_from_seconds(seconds: float, ticks_per_second: float) -> int:
    """Convert a per-second duration to an integer tick count.

    Tick-rate-independent tunables are authored in seconds; the engine
    consumes integer tick counts. ``max(1, ...)`` guarantees at least one
    tick so a non-zero duration never collapses to an instantaneous action
    at low tick rates. Kept as a pure function so the derivation can be
    unit-tested without touching the global ``CFG`` singleton.
    """
    return max(1, round(seconds * ticks_per_second))


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
        self.clock.ticks_per_phase = int(
            self.clock.ticks_per_second * self.clock.phase_duration_seconds
        )
        self.clock.ticks_per_round = (
            self.clock.ticks_per_phase * self.clock.phases_per_round
        )

        # Tick-rate-independent tunables: authored per-second in config.toml,
        # derived here into the integer tick counts the engine consumes. These
        # are exposed under the SAME attribute names the downstream code reads
        # (simulation.py / combat.py / ai_zombie.py / unit.py), so no consumer
        # changes are needed when ticks_per_second is later bumped to 24/60.
        tps = self.clock.ticks_per_second

        self.movement.marine_attack_ticks_per_tile = ticks_from_seconds(
            self.movement.marine_attack_seconds_per_tile, tps
        )
        self.movement.marine_cover_ticks_per_tile = ticks_from_seconds(
            self.movement.marine_cover_seconds_per_tile, tps
        )
        self.movement.marine_sprint_ticks_per_tile = ticks_from_seconds(
            self.movement.marine_sprint_seconds_per_tile, tps
        )

        self.zombie.ticks_per_tile = ticks_from_seconds(
            self.zombie.seconds_per_tile, tps
        )
        self.zombie.attack_cooldown_ticks = ticks_from_seconds(
            self.zombie.attack_cooldown_seconds, tps
        )

        self.weapons.rifle.burst_interval_ticks = ticks_from_seconds(
            self.weapons.rifle.burst_interval_seconds, tps
        )

        self.recorder.capacity = round(self.recorder.replay_seconds * tps)

    def reload(self):
        """Re-read config.toml from disk. Call on F5."""
        self._load()
        print(f"[config] Reloaded {self._path.name}")


# Global config instance — import this everywhere
CFG = GameConfig()
