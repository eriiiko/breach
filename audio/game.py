"""The sound layer as main.py holds it (#202 S1): the model, the device and
the read-only view of the sim between them. main.py calls

    layer = SoundLayer.create(CFG, sim, muted=...)      # None when disabled
    layer.consume_events(sim.tick_events, listener)     # after EVERY sim step
    layer.update(listener, paused)                      # once per frame: loops, fades
    layer.close()

and `listener_tile` says where the player hears from. Nothing here writes
to the sim.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
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

    def vents(self) -> List[Tuple[object, float, float]]:
        """Every RUNNING vent (alive, wired to a duct, a circulation rate
        above 0), keyed by its entity id, placed at its aperture tile's
        centre (where its air comes and goes)."""
        out = []
        for v in getattr(self._sim, "_vents", ()) or ():
            if (getattr(v, "alive", True) and v.duct is not None
                    and int(v.rate_raw_per_s) > 0):
                out.append((("vent", v.id), v.aperture_x + 0.5, v.aperture_y + 0.5))
        return out

    def fire_near(self, x: float, y: float, r_tiles: float):
        """The burning tiles within the square of half-side `r_tiles`
        around (x, y) (the model's distance law trims it to a disc): tile
        centres and the fire level, dequantized at this read (0..1)."""
        from simulation.fire_fixed import dequantize
        fire = self._sim.gmap.fire
        h, w = fire.shape
        r = int(np.ceil(r_tiles))
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return (), (), ()
        win = fire[y0:y1, x0:x1]
        ys, xs = np.nonzero(win > 0)
        level = np.minimum(dequantize(win[ys, xs]), 1.0)
        return xs + (x0 + 0.5), ys + (y0 + 0.5), level


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
        self.loops_started = 0
        self.loops_peak = 0          # most loops sounding at once (the E2E reads it)

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

    def update(self, listener: Tuple[float, float], paused: bool) -> None:
        """Once per frame: the loops from the world's state (S2), then the
        device's fades and stream feeding."""
        now = rl.get_time()
        commands = self.model.update_loops(self.world, listener, paused)
        self.loops_started += sum(1 for c in commands if type(c).__name__ == "LoopStart")
        self.device.execute(commands, now)
        self.device.update(now)
        self.loops_peak = max(self.loops_peak, len(self.model.loops))

    def close(self) -> None:
        self.device.close()
