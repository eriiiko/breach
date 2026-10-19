"""tools/look_shots.py — the look-shot set: load, validate, name (pyray-free).

Graphics arc G0 P3 (docs/graphics_direction_2026-10-18.md §7, §8.1). Every
graphics patch renders the SAME named studio shots before and after, and the
implementer looks at the PNGs before claiming the patch works; Erik judges
the same shots live. The shots are data in ``tools/look_shots.toml``; the
renderer is ``tools/lighting_demo.py --shots`` (one fresh demo process per
shot, so a shot never inherits another's sim or GL state). This module holds
only the part with no window: parsing the table, checking every row against
the levels on disk, and naming the output file. It never imports pyray or
breach_physics, so the table is testable headless.

A shot is fixed by construction: one sim tick per rendered frame (the
demo's ``--auto`` rule), a fixed camera, a fixed seed (the demo's 42), the
code DEFAULTS of the studio panel (never a saved preset, which the demo
rewrites on Save) and scripted tool actions at known frames. Two runs of the
same shot on the same commit therefore give the same picture, up to the
GPU's own float behaviour.
"""
from __future__ import annotations

import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SHOTS_TOML = Path(__file__).resolve().parent / "look_shots.toml"
DEFAULT_OUT_DIR = ROOT / "scratchpad" / "look"
LEVELS_DIR = ROOT / "levels"

_NAME_RE = re.compile(r"^[a-z0-9_]+$")
# Layers a shot may switch off to isolate the rest (the renderer's own
# toggles: show_temperature / show_smoke / show_water).
HIDEABLE_LAYERS = frozenset(("heat", "smoke", "water"))


@dataclass(frozen=True)
class Flashlight:
    """The studio's cursor lamp, pinned for a shot: tile position, power and
    full cone width in radians (2*pi = omni, the studio's default)."""
    pos: tuple
    intensity: float
    spread: float


@dataclass(frozen=True)
class LookShot:
    name: str
    level: str
    frames: int                       # rendered frames == sim ticks
    center: Optional[tuple] = None    # camera centre, tiles; None = map centre
    zoom: Optional[float] = None      # screen px per tile; None = fit the map
    lamps: bool = True                # the level's static/spot lamps
    beacon: bool = True               # the level's rotating beacons
    flashlight: Optional[Flashlight] = None
    ignite: tuple = ()                # tiles ignited before the first tick
    smoke: tuple = ()                 # tiles puffed with soot before tick 1
    steam: tuple = ()                 # tiles puffed with steam before tick 1
    detonate: Optional[tuple] = None  # (x, y, frame): frag grenade
    light_gain: object = None         # None = studio default, float, or
                                      # "game" = [render.lighting] master_gain
    hide: tuple = ()                  # layers switched off (HIDEABLE_LAYERS)
    note: str = ""


@dataclass(frozen=True)
class ShotTable:
    window_w: int
    window_h: int
    shots: tuple = field(default_factory=tuple)

    def by_name(self, name: str) -> LookShot:
        for s in self.shots:
            if s.name == name:
                return s
        raise KeyError(f"no look shot named {name!r} in {SHOTS_TOML.name} "
                       f"(have: {', '.join(s.name for s in self.shots)})")


def _tile(v, what: str) -> tuple:
    if (not isinstance(v, (list, tuple)) or len(v) != 2
            or not all(isinstance(c, (int, float)) for c in v)):
        raise ValueError(f"{what}: expected [x, y] tiles, got {v!r}")
    return (float(v[0]), float(v[1]))


def _tiles(v, what: str) -> tuple:
    if v is None:
        return ()
    if not isinstance(v, list):
        raise ValueError(f"{what}: expected a list of [x, y], got {v!r}")
    return tuple(tuple(int(c) for c in _tile(t, what)) for t in v)


def parse_shot(d: dict) -> LookShot:
    """One ``[[shot]]`` row -> LookShot, or ValueError naming the bad key."""
    name = str(d.get("name", ""))
    if not _NAME_RE.match(name):
        raise ValueError(f"shot name {name!r} must match [a-z0-9_]+")
    level = str(d.get("level", ""))
    if not level:
        raise ValueError(f"shot {name}: missing level")
    frames = int(d.get("frames", 0))
    if frames < 1:
        raise ValueError(f"shot {name}: frames must be >= 1, got {frames}")
    zoom = d.get("zoom")
    if zoom is not None:
        zoom = float(zoom)
        if zoom <= 0:
            raise ValueError(f"shot {name}: zoom must be > 0")
    fl = d.get("flashlight")
    flashlight = None
    if fl is not None:
        flashlight = Flashlight(
            pos=_tile(fl.get("pos"), f"shot {name} flashlight.pos"),
            intensity=float(fl.get("intensity", 2.5)),
            spread=float(fl.get("spread", 6.283)))
    det = d.get("detonate")
    detonate = None
    if det is not None:
        if not isinstance(det, list) or len(det) != 3:
            raise ValueError(f"shot {name}: detonate = [x, y, frame]")
        detonate = (int(det[0]), int(det[1]), int(det[2]))
        if not 0 <= detonate[2] < frames:
            raise ValueError(f"shot {name}: detonate frame {detonate[2]} "
                             f"must fall inside the shot's {frames} frames")
    hide = tuple(str(x) for x in d.get("hide", []))
    bad = set(hide) - HIDEABLE_LAYERS
    if bad:
        raise ValueError(f"shot {name}: cannot hide {sorted(bad)} "
                         f"(hideable: {sorted(HIDEABLE_LAYERS)})")
    gain = d.get("light_gain")
    if gain is not None and gain != "game":
        gain = float(gain)
    return LookShot(
        name=name, level=level, frames=frames,
        center=_tile(d["center"], f"shot {name} center") if "center" in d else None,
        zoom=zoom,
        lamps=bool(d.get("lamps", True)), beacon=bool(d.get("beacon", True)),
        flashlight=flashlight,
        ignite=_tiles(d.get("ignite"), f"shot {name} ignite"),
        smoke=_tiles(d.get("smoke"), f"shot {name} smoke"),
        steam=_tiles(d.get("steam"), f"shot {name} steam"),
        detonate=detonate,
        light_gain=gain,
        hide=hide,
        note=str(d.get("note", "")))


def load_shots(path: Path = SHOTS_TOML) -> ShotTable:
    """Parse the whole table; names must be unique."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    win = raw.get("window", {})
    shots = tuple(parse_shot(d) for d in raw.get("shot", []))
    names = [s.name for s in shots]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"duplicate look-shot names: {sorted(dup)}")
    return ShotTable(window_w=int(win.get("width", 1600)),
                     window_h=int(win.get("height", 900)), shots=shots)


def level_exists(level: str, levels_dir: Path = LEVELS_DIR) -> bool:
    return (levels_dir / level / "level.toml").is_file()


def git_tag(root: Path = ROOT) -> str:
    """Short HEAD sha, plus ``+dirty`` when tracked files differ from it — so
    a before/after pair taken around uncommitted work never shares a name
    with the committed picture."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, check=True
                             ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return f"{sha}+dirty" if dirty else sha


def shot_path(out_dir: Path, name: str, tag: str) -> Path:
    """``<out_dir>/<name>_<tag>.png`` — the §8.1 naming."""
    return Path(out_dir) / f"{name}_{tag}.png"


__all__ = ["Flashlight", "LookShot", "ShotTable", "SHOTS_TOML",
           "DEFAULT_OUT_DIR", "parse_shot", "load_shots", "level_exists",
           "git_tag", "shot_path"]
