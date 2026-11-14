"""Pose skinner -- takes a composed pose to the skinned mesh (M13, #162).

The pose layer (``renderer/pose_layer.py``) outputs, per unit, a global-TRS
row ``(J, 10)`` and skin matrices ``(J, 4, 4)``. A :class:`Skinner` puts one
unit's pose on its look's shared mesh just before that unit's draw; which
skinner is used changes nothing in the pose layer or the draw.

:class:`CpuInjectSkinner` is option (a) of the design's §4.5
(``docs/animation_milestones_design_2026-11-02.md``): one scratch
``ModelAnimation`` per look with ONE frame whose poses are ours, then raylib's
own CPU skinner, ``update_model_animation(model, scratch, 0)``. Exact (raylib
fed its own frame pose reproduces its vertices bit for bit) and costs what
today's player costs. The frame's poses are MODEL-SPACE (global) TRS, the
layout raylib's loader stores after ``BuildPoseFromParentJoints``.

Binding split (#80): raylib 5.5 (pyray 5.5.0.4, the home desktop) names the
struct's fields ``frameCount`` / ``framePoses``; the 6.x bindings
``keyframeCount`` / ``keyframePoses``. Both are read through
:func:`anim_pose_fields`, beside ``unit_model_renderer.anim_frame_count``;
a struct with neither raises naming both. ``tests/gl_pose_inject_check.py``
checks the injection on each machine's binding.

The GPU skinner (W13.2) is a second implementation of :class:`Skinner` that
writes ``mesh.boneMatrices`` from ``PoseBatch.skin``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from .pose_layer import GLOBAL_TRS_WIDTH

# (frame count, frame poses) field names, raylib 5.5 first, then 6.x.
_POSE_FIELD_PAIRS = (("frameCount", "framePoses"),
                     ("keyframeCount", "keyframePoses"))
_FLOAT_BYTES = 4


def anim_pose_fields(ffi) -> Tuple[str, str]:
    """``(count_field, poses_field)`` of the binding's ``ModelAnimation``.
    Raises AttributeError naming every pair when it has none of them."""
    fields = {f[0] for f in ffi.typeof("ModelAnimation").fields}
    for count, poses in _POSE_FIELD_PAIRS:
        if count in fields and poses in fields:
            return count, poses
    raise AttributeError(f"ModelAnimation carries none of {_POSE_FIELD_PAIRS} "
                         f"(fields: {sorted(fields)})")


class Skinner(Protocol):
    """Puts a composed pose on a look's mesh. ``prepare`` once per look (GL
    context live); ``apply`` per unit, immediately before its draw."""

    def prepare(self, look: str, model, template_anim, bone_names) -> None: ...

    def apply(self, look: str, model, pose_row: np.ndarray) -> None: ...


@dataclass
class _Scratch:
    anim: object               # cffi ModelAnimation * (owns nothing raylib frees)
    poses: object              # cffi Transform[J]  (kept alive here)
    pose_ptrs: object          # cffi Transform *[1] (kept alive here)
    view: np.ndarray           # (J, 10) float32, writable, aliases ``poses``


class CpuInjectSkinner:
    """Option (a): our global pose written into a one-frame scratch
    ``ModelAnimation`` that shares the look's bone table, then raylib's CPU
    skinner. Never frees the scratch through raylib (``bones`` belongs to the
    look's loaded clips; the pose buffers belong to cffi)."""

    def __init__(self, rl) -> None:
        self._rl = rl
        self._ffi = rl.ffi
        self._count_field, self._poses_field = anim_pose_fields(rl.ffi)
        if self._ffi.sizeof("Transform") != GLOBAL_TRS_WIDTH * _FLOAT_BYTES:
            raise RuntimeError("this binding's Transform is not 10 floats; "
                               "the pose layer's row layout does not fit it")
        self._scratch: Dict[str, _Scratch] = {}

    def prepare(self, look: str, model, template_anim, bone_names=None) -> None:
        """Build ``look``'s scratch clip from one of its loaded clips (its
        ``bones`` table). With ``bone_names`` given, refuses a look whose
        raylib bone table differs from the pose layer's skeleton (count,
        order or names) -- a pose row would land on the wrong bones."""
        ffi = self._ffi
        J = int(template_anim.boneCount)
        if bone_names is not None:
            raylib_names = [ffi.string(template_anim.bones[k].name).decode("utf-8", "replace")
                            for k in range(J)]
            if list(bone_names) != raylib_names:
                raise ValueError(f"{look}: raylib's bone table differs from the pose "
                                 f"layer's skeleton ({J} vs {len(bone_names)} bones)")
        if int(model.boneCount) != J:
            raise ValueError(f"{look}: model has {model.boneCount} bones, its clips {J}")
        poses = ffi.new("Transform[]", J)
        ptrs = ffi.new("Transform *[1]", [poses])
        anim = ffi.new("ModelAnimation *")
        anim.boneCount = J
        setattr(anim, self._count_field, 1)
        anim.bones = template_anim.bones
        setattr(anim, self._poses_field, ptrs)
        view = np.frombuffer(ffi.buffer(poses, J * GLOBAL_TRS_WIDTH * _FLOAT_BYTES),
                             dtype=np.float32).reshape(J, GLOBAL_TRS_WIDTH)
        self._scratch[look] = _Scratch(anim=anim, poses=poses, pose_ptrs=ptrs, view=view)

    def view(self, look: str) -> np.ndarray:
        """The writable ``(J, 10)`` pose buffer of ``look``'s scratch frame."""
        return self._scratch[look].view

    def apply(self, look: str, model, pose_row: np.ndarray) -> None:
        sc = self._scratch[look]
        sc.view[...] = pose_row
        self._rl.update_model_animation(model, sc.anim[0], 0)

    def release(self) -> None:
        self._scratch.clear()


__all__ = ["Skinner", "CpuInjectSkinner", "anim_pose_fields"]
