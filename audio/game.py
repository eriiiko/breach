"""The sound layer as main.py holds it (#202 S1): the model, the device and
the read-only view of the sim between them. main.py calls

    layer = SoundLayer.create(CFG, sim, muted=...)      # None when disabled
    layer.consume_events(sim.tick_events, listener)     # after EVERY sim step
    layer.update()                                      # once per frame
    layer.close()

and `listener_tile` says where the player hears from. Nothing here writes
to the sim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pyray as rl

from audio.device import SoundDevice
from audio.model import SoundConfigError, SoundModel, sound_settings

SOUNDS_ROOT = Path(__file__).resolve().parent.parent / "assets" / "sounds"


class SimWorld:
    """The model's `World` over a live Simulation: reads only."""

    def __init__(self, sim):
        self._sim = sim
        self._tables = sim.weapons_tables
        self.tile_size_m = float(sim.gmap.tile_size_m)

    def unit(self, unit_id: int) -> Optional[Tuple[float, float, str]]:
        u = self._sim.get_unit(unit_id)
        if u is None:
            return None
        half = 0.5 * float(getattr(u, "footprint", 1))
        return (float(u.x) + half, float(u.y) + half, str(u.weapon_id))

    def weapon_loudness_m(self, weapon_id: str) -> float:
        w = self._tables.weapons.by_name.get(weapon_id)
        return 0.0 if w is None else float(getattr(w, "loudness", 0.0))

    def payload_loudness_m(self, payload: str) -> float:
        p = self._tables.payloads.by_name.get(payload)
        return 0.0 if p is None else float(getattr(p, "loudness", 0.0))


def listener_tile(sim, control_source, camera) -> Tuple[float, float]:
    """Where the player hears from, in tiles: the possessed unit under
    direct control, else the centre of the view."""
    if getattr(control_source, "camera_mode", None) is not None:
        uid = getattr(control_source, "selected_unit_id", None)
        u = sim.get_unit(uid) if uid is not None else None
        if u is not None:
            half = 0.5 * float(getattr(u, "footprint", 1))
            return (float(u.x) + half, float(u.y) + half)
    w, h = camera.visible_tiles()
    return (camera.pos_tile_x + 0.5 * w, camera.pos_tile_y + 0.5 * h)


class SoundLayer:
    def __init__(self, settings, sim, *, muted: bool):
        self.device = SoundDevice(settings, SOUNDS_ROOT, muted=muted)
        self.model = SoundModel(settings, self.device.durations)
        self.world = SimWorld(sim)
        self.started = 0

    @classmethod
    def create(cls, cfg, sim, *, muted: bool = False) -> Optional["SoundLayer"]:
        """The layer, or None when `[sound] enabled = false`. A refused
        setting ends the launch, naming the key."""
        tables = sim.weapons_tables
        try:
            settings = sound_settings(
                cfg, root=SOUNDS_ROOT,
                weapon_ids=tables.weapons.by_name,
                payload_ids=tables.payloads.by_name)
        except SoundConfigError as e:
            raise SystemExit(f"config.toml: {e}")
        return cls(settings, sim, muted=muted) if settings.enabled else None

    def rebind(self, sim) -> None:
        """A rebuilt sim (a replay's restart) replaces the world view."""
        self.world = SimWorld(sim)

    def consume_events(self, events: Sequence, listener: Tuple[float, float]) -> None:
        now = rl.get_time()
        commands = self.model.on_events(events, self.world, listener, now)
        self.started += sum(1 for c in commands if type(c).__name__ == "Start")
        self.device.execute(commands, now)

    def update(self) -> None:
        self.device.update(rl.get_time())

    def close(self) -> None:
        self.device.close()
