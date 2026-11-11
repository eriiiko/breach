"""Test helper (not collected): write a small synthetic skinned glTF-binary
skeleton with animations, for the pose layer's skeleton-agnostic and refusal
tests. Pure json + numpy; no mesh is needed by the pose reader."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def write_glb(path: Path, bones: Sequence[dict], clips: Sequence[dict],
              *, extra_skin: bool = False, armature: Optional[dict] = None) -> Path:
    """``bones``: dicts ``name, parent (index or None), t, r[, s][, matrix]``.
    ``clips``: dicts ``name, channels: [(bone, path, interp, times, values)]``.
    The skin's joints are the bones in order, hung under an armature node
    (``armature`` = its TRS dict, default identity)."""
    blob = bytearray()
    views: List[dict] = []
    accessors: List[dict] = []

    def add(arr: np.ndarray, kind: str) -> int:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes})
        blob.extend(arr.tobytes())
        acc = {"bufferView": len(views) - 1, "componentType": 5126,
               "count": int(arr.shape[0]), "type": kind}
        if kind == "SCALAR":
            acc["min"] = [float(arr.min())]
            acc["max"] = [float(arr.max())]
        accessors.append(acc)
        return len(accessors) - 1

    nodes = []
    for b in bones:
        node = {"name": b["name"]}
        if "matrix" in b:
            node["matrix"] = list(map(float, b["matrix"]))
        else:
            node["translation"] = list(map(float, b["t"]))
            node["rotation"] = list(map(float, b["r"]))
            if "s" in b:
                node["scale"] = list(map(float, b["s"]))
        kids = [i for i, c in enumerate(bones) if c["parent"] == len(nodes)]
        if kids:
            node["children"] = kids
        nodes.append(node)
    arm = {"name": "Armature", "children": [i for i, b in enumerate(bones) if b["parent"] is None]}
    arm.update(armature or {})
    nodes.append(arm)
    # rest globals -> inverse bind matrices (column-major in the file)
    J = len(bones)
    glob = [None] * J
    for i, b in enumerate(bones):
        m = np.eye(4)
        if "matrix" in b:
            m = np.array(b["matrix"], dtype=float).reshape(4, 4).T
        else:
            m[:3, :3] = _quat_to_mat(b["r"]) * np.array(b.get("s", [1, 1, 1]))[None, :]
            m[:3, 3] = b["t"]
        glob[i] = m if b["parent"] is None else glob[b["parent"]] @ m
    ibm = np.stack([np.linalg.inv(g).T for g in glob]).reshape(J, 16)
    skins = [{"joints": list(range(J)), "inverseBindMatrices": add(ibm, "MAT4")}]
    if extra_skin:
        skins.append(dict(skins[0]))
    anims = []
    for clip in clips:
        samplers, channels = [], []
        for bone, pth, interp, times, values in clip["channels"]:
            values = np.asarray(values, dtype=np.float32)
            kind = {3: "VEC3", 4: "VEC4"}[values.shape[1]]
            samplers.append({"input": add(np.asarray(times, np.float32)[:, None], "SCALAR"),
                             "output": add(values, kind), "interpolation": interp})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": bone, "path": pth}})
        anims.append({"name": clip["name"], "samplers": samplers, "channels": channels})
    while len(blob) % 4:
        blob.append(0)
    doc = {"asset": {"version": "2.0"}, "scene": 0,
           "scenes": [{"nodes": [len(nodes) - 1]}], "nodes": nodes, "skins": skins,
           "animations": anims, "accessors": accessors, "bufferViews": views,
           "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(out)
    return path


def axis_quat(axis, angle) -> List[float]:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)]


# A five-bone creature with its OWN bone names (nothing like the shipped rig).
CRITTER_BONES = [
    {"name": "pelvis_x", "parent": None, "t": [0, 1, 0], "r": axis_quat([1, 0, 0], -0.3)},
    {"name": "back_x", "parent": 0, "t": [0, 0.5, 0], "r": [0, 0, 0, 1]},
    {"name": "skull_x", "parent": 1, "t": [0, 0.4, 0], "r": [0, 0, 0, 1]},
    {"name": "hindleg_l", "parent": 0, "t": [0.2, -0.3, 0], "r": [0, 0, 0, 1]},
    {"name": "hindleg_r", "parent": 0, "t": [-0.2, -0.3, 0], "r": [0, 0, 0, 1]},
]


def critter_clips() -> List[dict]:
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    sway = [axis_quat([0, 0, 1], a) for a in (0.0, 0.4, 0.0, -0.4, 0.0)]
    kick = [axis_quat([1, 0, 0], a) for a in (0.0, 0.8, 0.0, 0.8, 0.0)]
    nod = [axis_quat([1, 0, 0], a) for a in (0.2, 0.5, 0.2, 0.5, 0.2)]
    walk_t = [[0, 1, 0], [0, 1.1, 0.5], [0, 1, 1.0], [0, 1.1, 1.5], [0, 1, 2.0]]
    return [
        {"name": "trot", "channels": [(1, "rotation", "LINEAR", times, sway),
                                      (3, "rotation", "LINEAR", times, kick),
                                      (0, "translation", "LINEAR", times, walk_t)]},
        {"name": "look", "channels": [(2, "rotation", "LINEAR", times, nod)]},
        {"name": "rest", "channels": [(1, "rotation", "STEP", [0.0, 1.0],
                                       [[0, 0, 0, 1], [0, 0, 0, 1]])]},
    ]


def critter_settings(**over) -> dict:
    """A ``{"render": {"pose": ...}}`` dict for the critter, valid as is."""
    pose = {
        "enabled": True, "rate": 1.0, "crossfade_s": 0.2,
        "layers": ["hind", "front", "top"], "fallback_state": "idle",
        "event_queue": 4,
        "skeletons": {"critter": {
            "masks": {
                "hind": {"subtree": ["pelvis_x"], "minus": ["back_x"]},
                "front": {"subtree": ["back_x"], "minus": ["skull_x"]},
                "top": {"subtree": ["skull_x"]},
            },
            "twist": {"lower": "pelvis_x", "counter": ["back_x"], "shares": [1.0]},
            "posture": critter_posture_rule(),
        }},
        "states": {"idle": {"hind": "rest", "front": "rest", "top": "rest"},
                   "walk": {"hind": "trot", "front": "trot", "top": "trot"}},
        "clips": {"trot": {"root_motion": "keep"}, "look": {"loop": False}},
        "overlays": {"aim": {"front": "rest", "top": "look"}},
        "fire": {"overlay": "aim", "aim_hold_s": 0.5, "shot_clip": "look",
                 "shot_layer": "front"},
        "hit": {"clip": "look", "masks": ["front", "top"], "causes": ["bullet"],
                "ramp_in_s": 0.05, "ramp_out_s": 0.1, "weight_min": 0.25,
                "weight_max": 0.75, "damage_full": 10},
        "aim_travel": {"theta_deg": 90.0, "hysteresis_deg": 10.0, "smooth_s": 0.1,
                       "twist_rate_deg_s": 360.0},
        "posture": {
            "profile": "prowl", "seed": 7, "tempo_hz": [0.3, 0.5], "twitch_hz": [2.0, 3.0],
            "hz_spread": 0.2, "desync_start": True, "variation": {"spine_pitch": 0.1},
            "profiles": {"prowl": {"spine_pitch": 40.0, "knee_bend": 0.1,
                                   "shoulder_protraction": 0.5, "arm_forward": 20.0,
                                   "arm_open": 0.5, "head_drop": 0.5,
                                   "tempo_jitter": 0.3, "twitch": 0.5}},
        },
        "melee": {"clips": ["look"], "layers": ["front"], "causes": ["melee"],
                  "suppress": ["arm_forward"]},
        "roles": {"player": {"reactions": ["fire", "hit"], "posture": 0.0},
                  "zombie": {"reactions": ["melee", "hit"], "posture": 1.0}},
        # #184: the reload / throw one-shots, off ("" = no clip)
        "reload": {"clip": "", "layers": ["front"]},
        "throw": {"clip": "", "layers": ["front"], "release_s": 0.0},
    }
    pose.update(over)
    return {"render": {"pose": pose}}


def critter_posture_rule() -> dict:
    """The critter's posture mapping (its own bone names): its back pitches,
    its skull drops and twitches, its pelvis drops (no leg chains)."""
    def row(bones, axis, deg, space="model"):
        return {"bones": bones, "axis": axis, "deg": deg, "space": space}
    return {
        "up": [0.0, 1.0, 0.0], "forward": [0.0, 0.0, 1.0],
        "spine_pitch": [row(["back_x"], [1.0, 0.0, 0.0], 1.0)],
        "shoulder_protraction": [],
        "arm_forward": [row(["hindleg_l", "hindleg_r"], [1.0, 0.0, 0.0], -0.5)],
        "arm_open": [],
        "head_drop": [row(["skull_x"], [1.0, 0.0, 0.0], 30.0)],
        "twitch": [row(["skull_x"], [0.0, 1.0, 0.0], 10.0, "local")],
        "knee_bend": {"hips": "pelvis_x", "legs": []},
    }
