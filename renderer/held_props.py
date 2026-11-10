"""Held props -- the pure, raylib-free half of a prop a 3D unit holds (#184).

A character model carries no props (CLAUDE.md, Character kit): a weapon is
its own asset, drawn on a HAND SOCKET of the unit's composed pose. What a
unit holds is data:

* ``[render.held_props] weapons`` maps a ``[weapons.*]`` id (the unit's
  loadout weapon, ``Unit.weapon_id``, read-only) to a prop row;
* ``[render.held_props.props.<name>]``: the prop's model (``assets/models/
  <model>/<model>.glb``), its length in metres (the file's longest side is
  scaled to it), the socket it is held on by default, the point of the file
  the hand closes on (``grip``, file units), its gloss (the shader reads
  gloss from the albedo's alpha; a prop's albedo has none) and one grip row
  per socket it may be held on (``grips.<socket>``: where that point sits in
  the hand bone's frame, metres, and how the prop is turned there, XYZ Euler
  degrees);
* a skeleton table's ``sockets`` (``[render.pose.skeletons.<s>.sockets]``)
  names the bone of each socket; a look whose skeleton lacks the socket
  draws no prop.

The drawn transform is ``body @ hand @ grip``: the body's model transform
(the one ``draw_model_ex`` builds), the hand bone's composed global pose
(the pose layer's ``global_trs`` row) and the prop's grip in that hand.
WHICH socket a unit holds a prop on is ``PoseLayer.socket_bone`` -- the row
default, or the unit's override a future reaction row sets (a throw moving
the carbine to the left hand); nothing sets one yet.

Render-only: nothing is written to a unit or read from one but its weapon id.
The pyray half (loading, drawing through the units' one lit shader, the
projected shadow) is in ``renderer/unit_model_renderer.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_TOP_KEYS = ("weapons", "props")
_PROP_KEYS = ("model", "length_m", "socket", "grip", "gloss", "grips")
_GRIP_KEYS = ("offset_m", "rotation_deg")


class HeldPropConfigError(ValueError):
    """A ``[render.held_props]`` setting refused (names the key)."""


@dataclass(frozen=True)
class Grip:
    """A prop held on one socket: the grip point's place in the hand bone's
    frame (metres) and the prop's rotation there (XYZ Euler, degrees)."""
    offset_m: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]


@dataclass(frozen=True)
class HeldProp:
    name: str
    model: str
    length_m: float
    socket: str
    grip: Tuple[float, float, float]      # the point the hand closes on, file units
    gloss: float
    grips: Dict[str, Grip]


@dataclass(frozen=True)
class HeldPropSettings:
    weapons: Dict[str, str]               # [weapons.*] id -> prop name
    props: Dict[str, HeldProp]

    def prop_for(self, weapon_id: str) -> Optional[HeldProp]:
        """The prop a unit with ``weapon_id`` holds (None: none)."""
        name = self.weapons.get(weapon_id or "")
        return None if name is None else self.props[name]

    def sockets(self) -> Tuple[str, ...]:
        return tuple(sorted({s for p in self.props.values() for s in p.grips}))


def _as_dict(obj):
    if isinstance(obj, Mapping):
        return {k: _as_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, int, float, list)):
        return {k: _as_dict(v) for k, v in vars(obj).items()}
    return obj


def _exact(d, keys: Sequence[str], where: str) -> dict:
    if not isinstance(d, dict):
        raise HeldPropConfigError(f"{where} must be a table")
    for k in keys:
        if k not in d:
            raise HeldPropConfigError(f"{where} is missing {k!r}")
    bad = sorted(set(d) - set(keys))
    if bad:
        raise HeldPropConfigError(f"{where} has unknown keys {bad}")
    return d


def _vec3(v, where: str) -> Tuple[float, float, float]:
    if not isinstance(v, list) or len(v) != 3 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        raise HeldPropConfigError(f"{where} must be three numbers, got {v!r}")
    return tuple(float(x) for x in v)


def _text(v, where: str) -> str:
    if not isinstance(v, str) or not v:
        raise HeldPropConfigError(f"{where} must be a non-empty string, got {v!r}")
    return v


def held_prop_settings(cfg, weapon_ids: Optional[Sequence[str]] = None) -> HeldPropSettings:
    """``[render.held_props]`` from a loaded config (``CFG``) or a plain dict
    holding ``render.held_props``. Every key is required; a missing, unknown
    or mistyped one is refused, naming it -- and a ``weapons`` key that is no
    ``[weapons.*]`` id (``weapon_ids``; None = the config's own table), a
    prop name that is no row, a default socket without a grip row."""
    if isinstance(cfg, Mapping):
        section = cfg.get("render", {}).get("held_props")
        weapons_cfg = cfg.get("weapons")
    else:
        section = getattr(getattr(cfg, "render", None), "held_props", None)
        weapons_cfg = getattr(cfg, "weapons", None)
    if section is None:
        raise HeldPropConfigError("config.toml has no [render.held_props] section")
    d = _exact(_as_dict(section), _TOP_KEYS, "[render.held_props]")
    if weapon_ids is None:
        weapon_ids = sorted(_as_dict(weapons_cfg) or {}) if weapons_cfg is not None else []
    praw = d["props"]
    if not isinstance(praw, dict):
        raise HeldPropConfigError("[render.held_props.props] must be a table of prop rows")
    props: Dict[str, HeldProp] = {}
    for name, row in praw.items():
        w = f"[render.held_props.props.{name}]"
        r = _exact(row, _PROP_KEYS, w)
        length = r["length_m"]
        if not isinstance(length, (int, float)) or isinstance(length, bool) or not length > 0:
            raise HeldPropConfigError(f"{w} length_m must be a number > 0, got {length!r}")
        gloss = r["gloss"]
        if not isinstance(gloss, (int, float)) or isinstance(gloss, bool) or \
                not 0.0 <= gloss <= 1.0:
            raise HeldPropConfigError(f"{w} gloss must be a number in [0, 1], got {gloss!r}")
        graw = r["grips"]
        if not isinstance(graw, dict) or not graw:
            raise HeldPropConfigError(f"{w} grips must hold one row per socket")
        grips = {}
        for sock, g in graw.items():
            gw = f"{w} grips.{sock}"
            g = _exact(g, _GRIP_KEYS, gw)
            grips[sock] = Grip(offset_m=_vec3(g["offset_m"], f"{gw} offset_m"),
                               rotation_deg=_vec3(g["rotation_deg"], f"{gw} rotation_deg"))
        socket = _text(r["socket"], f"{w} socket")
        if socket not in grips:
            raise HeldPropConfigError(f"{w} socket {socket!r} has no grips row "
                                      f"(rows: {sorted(grips)})")
        props[name] = HeldProp(name=name, model=_text(r["model"], f"{w} model"),
                               length_m=float(length), socket=socket,
                               grip=_vec3(r["grip"], f"{w} grip"), gloss=float(gloss),
                               grips=grips)
    wraw = d["weapons"]
    if not isinstance(wraw, dict):
        raise HeldPropConfigError("[render.held_props] weapons must be a table of "
                                  "weapon id = prop name")
    for wid, pname in wraw.items():
        if wid not in weapon_ids:
            raise HeldPropConfigError(f"[render.held_props] weapons.{wid}: not a "
                                      f"[weapons.*] id")
        if pname not in props:
            raise HeldPropConfigError(f"[render.held_props] weapons.{wid} = {pname!r}: "
                                      f"not a prop row {sorted(props)}")
    return HeldPropSettings(weapons=dict(wraw), props=props)


# ---------------------------------------------------------------------------
# Transforms (4x4, column vectors: p_world = M @ p)
# ---------------------------------------------------------------------------

def euler_xyz(deg: Sequence[float]) -> np.ndarray:
    """The rotation of XYZ Euler angles in degrees: about x, then y, then z
    (fixed axes): ``Rz @ Ry @ Rx``."""
    x, y, z = (math.radians(float(a)) for a in deg)
    cx, sx, cy, sy, cz, sz = (math.cos(x), math.sin(x), math.cos(y), math.sin(y),
                              math.cos(z), math.sin(z))
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def trs_matrix(row: np.ndarray) -> np.ndarray:
    """A pose layer ``global_trs`` row (T xyz, R xyzw, S xyz) as a 4x4."""
    t = np.asarray(row[0:3], dtype=np.float64)
    x, y, z, w = (float(v) for v in row[3:7])
    s = np.asarray(row[7:10], dtype=np.float64)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4)
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = t
    return m


def grip_matrix(prop: HeldProp, socket: str, file_extent: float) -> np.ndarray:
    """The prop's file space -> the hand bone's frame, held on ``socket``:
    ``T(offset) @ R(rotation) @ S(length / file_extent) @ T(-grip)``."""
    g = prop.grips[socket]
    s = prop.length_m / float(file_extent)
    m = np.eye(4)
    m[:3, :3] = euler_xyz(g.rotation_deg) * s
    m[:3, 3] = np.asarray(g.offset_m) - m[:3, :3] @ np.asarray(prop.grip)
    return m


def body_matrix(x: float, z: float, yaw_deg: float, scale: float,
                model_transform: Optional[np.ndarray] = None) -> np.ndarray:
    """The transform ``draw_model_ex(model, (x, 0, z), (0, 1, 0), yaw_deg,
    (scale,) * 3)`` draws a model with: ``T @ Ry @ S @ model.transform``."""
    a = math.radians(yaw_deg)
    m = np.eye(4)
    m[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                          [-math.sin(a), 0, math.cos(a)]]) * scale
    m[0, 3], m[2, 3] = x, z
    if model_transform is not None:
        m = m @ np.asarray(model_transform, dtype=np.float64)
    return m


def prop_world(body: np.ndarray, hand_row: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """The prop's world transform: ``body @ hand @ grip``."""
    return body @ trs_matrix(hand_row) @ grip


def model_extent(path) -> float:
    """The longest side of a .glb's mesh bounds (its POSITION accessors'
    min / max, every primitive), in file units."""
    from .pose_clips import read_glb
    doc, _ = read_glb(Path(path))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            lo, hi = np.minimum(lo, acc["min"]), np.maximum(hi, acc["max"])
    ext = float((hi - lo).max())
    if not ext > 0:
        raise HeldPropConfigError(f"{Path(path).name}: no mesh bounds")
    return ext


__all__ = ["HeldPropConfigError", "Grip", "HeldProp", "HeldPropSettings",
           "held_prop_settings", "euler_xyz", "trs_matrix", "grip_matrix",
           "body_matrix", "prop_world", "model_extent"]
