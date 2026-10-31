"""creature_model -- the raylib half of drawing a creature (#106 P7): a loaded
creature look's skeleton fed ONE frame pose per draw from
``renderer/creature_pose.py``, then raylib's native CPU skin.

``CreatureSkin`` is the game's form of the art prototype's ``LiveOctopus``
(``art/characters/octopus/scripts/runtime_octopus.py``, which measured its
parity against the NumPy reference skin at 11.4 um): a one-frame
``ModelAnimation`` over the loaded model's own bones, matched to
``octopus_math.bones("game")`` BY NAME, each frame pose = the rig's
deformation rotation composed with the bone's bind rotation, and raylib's
model-axis scale pre-compensated (``p / s + (s - 1) R b``, the prototype's
measured convention). The write is vectorised (one numpy view over the cffi
``Transform`` array) instead of a per-bone attribute loop.

The door :func:`check_bind_pose`: the loaded file's bind heads must be the
rig's bind pose at the size the creature row named (``creature_pose.
match_size``) within 1 mm, or the look is refused -- a GLB built at another
size would skin a different animal than the sim moves.

Render-only: nothing here reads or writes the sim.
"""
from __future__ import annotations

import numpy as np

#: The bind door's tolerance, metres.
BIND_TOLERANCE_M = 1e-3


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternions (..., 4) in raylib's xyzw order from rotations
    (..., 3, 3) -- ``arm_math.quaternion``'s branches, vectorised (the sign
    may differ: q and -q are one rotation)."""
    R = np.asarray(R, float)
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    case = np.argmax(np.stack([tr, m00, m11, m22], -1), -1)
    q = np.empty(R.shape[:-2] + (4,))
    with np.errstate(invalid="ignore", divide="ignore"):
        s0 = np.sqrt(np.maximum(tr + 1.0, 0.0)) * 2.0
        c0 = np.stack([(m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0, 0.25 * s0], -1)
        s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2.0
        c1 = np.stack([0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1, (m21 - m12) / s1], -1)
        s2 = np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0)) * 2.0
        c2 = np.stack([(m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2, (m02 - m20) / s2], -1)
        s3 = np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0)) * 2.0
        c3 = np.stack([(m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3, (m10 - m01) / s3], -1)
    for k, c in enumerate((c0, c1, c2, c3)):
        sel = case == k
        q[sel] = c[sel]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_mul(q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton products q x b (xyzw), ``LiveOctopus.write_pose``'s order."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w * bx + x * bw + y * bz - z * by,
                     w * by - x * bz + y * bw + z * bx,
                     w * bz + x * by - y * bx + z * bw,
                     w * bw - x * bx - y * by - z * bz], -1)


def frame_transforms(points, rotations, scales, bind_head, bind_quat) -> np.ndarray:
    """The (bones, 10) float32 rows of raylib ``Transform``s (translation,
    rotation xyzw, scale) for one pose, in the RIG's bone order:
    rotation = deformation x bind rotation; the translation pre-compensated
    for raylib's model-axis scale where a scale is not 1 (class doc)."""
    p = np.asarray(points, float)
    s = np.asarray(scales, float)
    if (s != 1.0).any():
        rb = np.einsum("nij,nj->ni", rotations, bind_head)
        p = p / s[:, None] + (s - 1.0)[:, None] * rb
    out = np.empty((p.shape[0], 10), np.float32)
    out[:, 0:3] = p
    out[:, 3:7] = quat_mul(quat_from_matrix(rotations), bind_quat)
    out[:, 7:10] = s[:, None]
    return out


def check_bind_pose(bind_head: np.ndarray, rig_bind: np.ndarray, where: str) -> float:
    """The bind door: the model's bind heads (rig order, metres) against the
    rig's bind pose. Returns the largest gap; raises ValueError over
    :data:`BIND_TOLERANCE_M`."""
    gap = float(np.abs(np.asarray(bind_head) - np.asarray(rig_bind)).max())
    if gap > BIND_TOLERANCE_M:
        raise ValueError(f"{where}: the model's bind pose is {gap * 1000:.1f} mm from "
                         f"the rig's at this size -- it was built at another size")
    return gap


class CreatureSkin:
    """One loaded creature model's one-frame animation (shared by every unit
    drawn with that look: each draw re-skins it, the unit renderer's
    shared-model seam)."""

    def __init__(self, rl, model, bone_names) -> None:
        self.rl, self.model = rl, model
        n = int(model.boneCount)
        if n != len(bone_names):
            raise ValueError(f"the model has {n} bones, the rig {len(bone_names)}")
        self.pose_array = rl.ffi.new("Transform[]", n)
        self.frames = rl.ffi.new("Transform*[]", [self.pose_array])
        self.animation = rl.ffi.new("ModelAnimation *")
        self.animation.boneCount = n
        self.animation.bones = model.bones
        self.animation.framePoses = self.frames
        for f in ("frameCount", "keyframeCount"):
            if hasattr(self.animation, f):
                setattr(self.animation, f, 1)
                break
        order = {nm: i for i, nm in enumerate(bone_names)}
        to_rig = np.empty(n, int)
        bind_head = np.zeros((n, 3))
        bind_quat = np.zeros((n, 4))
        for i in range(n):
            name = rl.ffi.string(model.bones[i].name).decode()
            if name not in order:
                raise ValueError(f"the model's bone {name!r} is not in the rig")
            j = order[name]
            to_rig[i] = j
            self.pose_array[i] = model.bindPose[i]
            t, q = model.bindPose[i].translation, model.bindPose[i].rotation
            bind_head[j] = (t.x, t.y, t.z)
            bind_quat[j] = (q.x, q.y, q.z, q.w)
        if sorted(to_rig.tolist()) != list(range(n)):
            raise ValueError("the model's skeleton does not match the rig's bones")
        self.to_rig = to_rig
        self.bind_head = bind_head          # rig order
        self.bind_quat = bind_quat          # rig order
        size = rl.ffi.sizeof("Transform")
        if size != 40:                      # pragma: no cover - a raylib change
            raise ValueError(f"raylib Transform is {size} bytes, not 40")
        self._rows = np.frombuffer(rl.ffi.buffer(self.pose_array), np.float32).reshape(n, 10)

    def write(self, pose) -> None:
        """Write one ``creature_pose.CreaturePose`` and skin the model (raylib's
        CPU skin)."""
        rows = frame_transforms(pose.points, pose.rotations, pose.scales,
                                self.bind_head, self.bind_quat)
        self._rows[:] = rows[self.to_rig]
        self.rl.update_model_animation(self.model, self.animation[0], 0)


__all__ = ["BIND_TOLERANCE_M", "quat_from_matrix", "quat_mul", "frame_transforms",
           "check_bind_pose", "CreatureSkin"]
