"""Pose clips -- the pose layer's own glTF-binary reader (M13, #162).

Pure: json + numpy, no pyray, no GL. It reads ONE skinned look's ``.glb`` into

* a :class:`Skeleton` -- bone names, parents, depth levels (for FK by level),
  rest locals and the skin's inverse bind matrices, in the SKIN's joint order
  (the order raylib's ``ModelAnimation.bones`` uses);
* a :class:`ClipBank` -- every animation of the file baked ONCE into dense
  ``(K, J, 3/4)`` arrays on the clip's own key grid, quaternions made
  hemisphere-continuous along time, with per-clip data (duration, key step,
  loop, playback rate, root motion, additive reference frame).

Design: ``docs/animation_milestones_design_2026-11-02.md`` §4.5 (correction 4:
our own reader, never a reverse-engineering of raylib's stepped
``framePoses``). Skeleton-agnostic by rule (§2 rule 1): nothing here names a
bone; which bones form a region is data (``[render.pose]`` masks).

The reader REFUSES, by name, what it does not implement, so a different
skeleton (a SOMA / Kimodo rig, a Mixamo armature) fails loudly instead of
animating wrongly: CUBICSPLINE samplers, ``matrix`` nodes in the joint chain,
more than one skin, a non-joint node between two joints, a non-identity node
above the skin's root joints, non-float animation data.

Conventions: glTF is Y-up, quaternions are ``(x, y, z, w)``, translations are
in the parent bone's space. Matrices here are row-major acting on column
vectors (``M @ [x, y, z, 1]``), so a glTF (column-major) matrix is transposed
on read.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_N_COMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_FLOAT = 5126

# Interpolations the bake implements. CUBICSPLINE is refused by name.
INTERP_LINEAR = "LINEAR"
INTERP_STEP = "STEP"
_SUPPORTED_INTERP = (INTERP_LINEAR, INTERP_STEP)
_PATHS = ("translation", "rotation", "scale")

# Root-motion policies (a per-clip row, ``[render.pose.clips]``).
ROOT_MOTION_KEEP = "keep"
ROOT_MOTION_STRIP = "strip"
ROOT_MOTION_MODES = (ROOT_MOTION_KEEP, ROOT_MOTION_STRIP)
# glTF is Y-up (spec §3.4): a root bone's HORIZONTAL translation is its x, z.
_HORIZONTAL_AXES = (0, 2)

_IDENT_EPS = 1e-6


class PoseDataError(ValueError):
    """A glb the pose layer refuses (names the file and the reason)."""


# ---------------------------------------------------------------------------
# glb container + accessors
# ---------------------------------------------------------------------------

def read_glb(path) -> Tuple[dict, bytes]:
    """The JSON document and the BIN chunk of a binary glTF file."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] != _GLB_MAGIC:
        raise PoseDataError(f"{path.name}: not a binary glTF (.glb) file")
    off = 12
    doc, binary = None, b""
    while off + 8 <= len(raw):
        length, kind = struct.unpack_from("<II", raw, off)
        off += 8
        chunk = raw[off:off + length]
        off += length
        if kind == _CHUNK_JSON:
            doc = json.loads(chunk)
        elif kind == _CHUNK_BIN:
            binary = bytes(chunk)
    if doc is None:
        raise PoseDataError(f"{path.name}: glb has no JSON chunk")
    for buf in doc.get("buffers", []):
        if "uri" in buf:
            raise PoseDataError(f"{path.name}: external buffer {buf['uri']!r} "
                                "(the pose reader reads self-contained .glb only)")
    return doc, binary


def _accessor(doc: dict, binary: bytes, index: int, *, what: str,
              src: str) -> np.ndarray:
    """Accessor ``index`` as a float64 ``(count, n)`` array. Refuses non-float
    and sparse accessors (normalized quantized animation data is not
    implemented)."""
    acc = doc["accessors"][index]
    if "sparse" in acc:
        raise PoseDataError(f"{src}: {what} uses a sparse accessor (refused)")
    if acc["componentType"] != _FLOAT:
        raise PoseDataError(f"{src}: {what} is not float32 data "
                            f"(componentType {acc['componentType']}; refused)")
    view = doc["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_COMPONENT[acc["componentType"]])
    n = _N_COMP[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", 0)
    count = acc["count"]
    if stride and stride != dtype.itemsize * n:
        arr = np.ndarray((count, n), dtype, binary, start,
                         (stride, dtype.itemsize))
    else:
        arr = np.frombuffer(binary, dtype, count * n, start).reshape(count, n)
    return np.array(arr, dtype=np.float64)


# ---------------------------------------------------------------------------
# Quaternion helpers (x, y, z, w), vectorised over leading axes
# ---------------------------------------------------------------------------

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (apply b, then a)."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.array(q, copy=True)
    out[..., :3] *= -1.0
    return out


def quat_nlerp(a: np.ndarray, b: np.ndarray, w) -> np.ndarray:
    """Normalised lerp from ``a`` (w = 0) to ``b`` (w = 1) along the short
    arc. ``w`` broadcasts against ``a[..., :1]``."""
    sign = np.where((a * b).sum(-1, keepdims=True) < 0.0, -1.0, 1.0)
    q = a + (b * sign - a) * w
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_slerp(a: np.ndarray, b: np.ndarray, u) -> np.ndarray:
    """Spherical lerp (the glTF LINEAR rotation interpolant), short arc."""
    u = np.asarray(u, dtype=np.float64)[..., None]
    d = (a * b).sum(-1, keepdims=True)
    b = np.where(d < 0.0, -b, b)
    d = np.abs(d)
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-6
    s_safe = np.where(small, 1.0, s)
    wa = np.where(small, 1.0 - u, np.sin((1.0 - u) * theta) / s_safe)
    wb = np.where(small, u, np.sin(u * theta) / s_safe)
    q = wa * a + wb * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_mat3(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def mat3_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` -> unit quaternions, branch-free
    over the batch (Shepperd's choice of the largest diagonal term)."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    cand = np.stack([tr, m00, m11, m22], axis=-1)
    pick = np.argmax(cand, axis=-1)
    out = np.empty(m.shape[:-2] + (4,), dtype=m.dtype)
    # w largest
    s = np.sqrt(np.maximum(1.0 + tr, 1e-12)) * 2.0
    qw = np.stack([(m[..., 2, 1] - m[..., 1, 2]) / s, (m[..., 0, 2] - m[..., 2, 0]) / s,
                   (m[..., 1, 0] - m[..., 0, 1]) / s, 0.25 * s], -1)
    s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2.0
    qx = np.stack([0.25 * s, (m[..., 0, 1] + m[..., 1, 0]) / s,
                   (m[..., 0, 2] + m[..., 2, 0]) / s, (m[..., 2, 1] - m[..., 1, 2]) / s], -1)
    s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2.0
    qy = np.stack([(m[..., 0, 1] + m[..., 1, 0]) / s, 0.25 * s,
                   (m[..., 1, 2] + m[..., 2, 1]) / s, (m[..., 0, 2] - m[..., 2, 0]) / s], -1)
    s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2.0
    qz = np.stack([(m[..., 0, 2] + m[..., 2, 0]) / s, (m[..., 1, 2] + m[..., 2, 1]) / s,
                   0.25 * s, (m[..., 1, 0] - m[..., 0, 1]) / s], -1)
    p = pick[..., None]
    out[...] = np.where(p == 0, qw, np.where(p == 1, qx, np.where(p == 2, qy, qz)))
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def trs_matrices(t: np.ndarray, q: np.ndarray, s: Optional[np.ndarray]) -> np.ndarray:
    """Local TRS -> ``(..., 4, 4)`` matrices ``T @ R @ S``."""
    shape = t.shape[:-1]
    m = np.zeros(shape + (4, 4), dtype=t.dtype)
    rot = quat_to_mat3(q)
    if s is not None:
        rot = rot * s[..., None, :]
    m[..., :3, :3] = rot
    m[..., :3, 3] = t
    m[..., 3, 3] = 1.0
    return m


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Skeleton:
    """One skin's joint hierarchy, in the skin's joint order.

    ``levels[d]`` holds the bones at depth ``d`` (roots at 0): FK runs one
    batched matrix product per level, never a per-bone loop."""
    names: Tuple[str, ...]
    parents: np.ndarray            # (J,) int, -1 for a root
    levels: Tuple[np.ndarray, ...]
    rest_t: np.ndarray             # (J, 3)
    rest_r: np.ndarray             # (J, 4) xyzw
    rest_s: np.ndarray             # (J, 3)
    bind_inverse: np.ndarray       # (J, 4, 4) row-major
    source: str = ""

    @property
    def n_bones(self) -> int:
        return len(self.names)

    @property
    def roots(self) -> np.ndarray:
        return np.flatnonzero(self.parents < 0)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"{self.source}: skeleton has no bone {name!r}") from None

    def subtree(self, bone: int) -> np.ndarray:
        """Boolean ``(J,)``: ``bone`` and every descendant."""
        inside = np.zeros(self.n_bones, dtype=bool)
        inside[bone] = True
        for lvl in self.levels:                 # parents precede children
            inside[lvl] |= inside[np.maximum(self.parents[lvl], 0)] & (self.parents[lvl] >= 0)
        return inside


def _node_trs(node: dict, src: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "matrix" in node:
        raise PoseDataError(f"{src}: node {node.get('name', '?')!r} carries a "
                            "'matrix' transform (refused: TRS nodes only)")
    return (np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
            np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64),
            np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64))


def _is_identity(node: dict, src: str) -> bool:
    t, r, s = _node_trs(node, src)
    return (np.abs(t).max() < _IDENT_EPS and np.abs(np.abs(r[3]) - 1.0) < _IDENT_EPS
            and np.abs(s - 1.0).max() < _IDENT_EPS)


def read_skeleton(doc: dict, binary: bytes, src: str) -> Skeleton:
    skins = doc.get("skins", [])
    if len(skins) != 1:
        raise PoseDataError(f"{src}: {len(skins)} skins (the pose layer reads "
                            "exactly one skin per look)")
    skin = skins[0]
    joints: List[int] = list(skin["joints"])
    nodes = doc["nodes"]
    jpos = {node: i for i, node in enumerate(joints)}
    node_parent: Dict[int, int] = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            node_parent[child] = i
    parents = np.full(len(joints), -1, dtype=np.int64)
    for i, node in enumerate(joints):
        p = node_parent.get(node)
        if p in jpos:
            parents[i] = jpos[p]
            continue
        # The chain above a root joint: must hold no joint (no non-joint node
        # between two joints) and no transform (raylib ignores it).
        while p is not None:
            if p in jpos:
                raise PoseDataError(
                    f"{src}: joint {nodes[node].get('name')!r} hangs under the "
                    f"non-joint node {nodes[node_parent[node]].get('name')!r} "
                    "inside the skeleton (refused)")
            if not _is_identity(nodes[p], src):
                raise PoseDataError(
                    f"{src}: node {nodes[p].get('name')!r} above the skeleton "
                    "carries a transform (refused: it would move every bone)")
            p = node_parent.get(p)
    names = tuple(str(nodes[n].get("name", f"joint{i}")) for i, n in enumerate(joints))
    if len(set(names)) != len(names):
        raise PoseDataError(f"{src}: duplicate bone names in the skin")
    depth = np.zeros(len(joints), dtype=np.int64)
    for i in range(len(joints)):          # load time, once per look
        p = parents[i]
        while p >= 0:
            depth[i] += 1
            p = parents[p]
    levels = tuple(np.flatnonzero(depth == d) for d in range(int(depth.max()) + 1))
    trs = [_node_trs(nodes[n], src) for n in joints]
    rest_t = np.array([x[0] for x in trs])
    rest_r = np.array([x[1] for x in trs])
    rest_r /= np.linalg.norm(rest_r, axis=-1, keepdims=True)
    rest_s = np.array([x[2] for x in trs])
    if "inverseBindMatrices" in skin:
        ibm = _accessor(doc, binary, skin["inverseBindMatrices"],
                        what="inverseBindMatrices", src=src)
        bind_inv = ibm.reshape(-1, 4, 4).transpose(0, 2, 1).copy()
    else:
        bind_inv = np.tile(np.eye(4), (len(joints), 1, 1))
    return Skeleton(names=names, parents=parents, levels=levels,
                    rest_t=rest_t, rest_r=rest_r, rest_s=rest_s,
                    bind_inverse=bind_inv, source=src)


# ---------------------------------------------------------------------------
# Clip bank
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClipRow:
    """Per-clip playback data (``[render.pose.clips]``; defaults = every clip
    loops at the global rate, root motion kept, additive reference frame 0)."""
    loop: bool = True
    rate: float = 1.0
    root_motion: str = ROOT_MOTION_KEEP
    additive_ref: int = 0


@dataclass
class ClipBank:
    """Every clip of one look, baked to dense arrays (concatenated along the
    key axis; clip ``c`` owns rows ``offset[c] : offset[c] + n_keys[c]``)."""
    skeleton: Skeleton
    names: Tuple[str, ...]
    t: np.ndarray                  # (K_total, J, 3) float32
    r: np.ndarray                  # (K_total, J, 4) float32, hemisphere-continuous
    s: Optional[np.ndarray]        # (K_total, J, 3) float32, None = all ones
    step_t: np.ndarray             # (C, J) bool: STEP translation channel
    step_r: np.ndarray             # (C, J) bool
    step_s: np.ndarray             # (C, J) bool
    offset: np.ndarray             # (C,) int
    n_keys: np.ndarray             # (C,) int
    key_dt: np.ndarray             # (C,) float, seconds between grid keys
    duration: np.ndarray           # (C,) float, seconds
    loop: np.ndarray               # (C,) bool
    rate: np.ndarray               # (C,) float, per-clip rate multiplier
    additive_ref: np.ndarray       # (C,) int, frame index of the additive reference
    root_motion: Tuple[str, ...] = ()
    _index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {n: i for i, n in enumerate(self.names)}

    @property
    def n_clips(self) -> int:
        return len(self.names)

    def has(self, name: str) -> bool:
        return name in self._index

    def index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"{self.skeleton.source}: no clip {name!r}") from None


def _clip_grid(times: Sequence[np.ndarray]) -> Tuple[float, int]:
    """The clip's key step (the smallest key interval over its channels) and
    its grid key count. Keys that lie on the grid bake exactly; any others
    are resampled by their own interpolant."""
    allt = np.unique(np.concatenate([np.asarray(t).ravel() for t in times]))
    dur = float(allt[-1]) if len(allt) else 0.0
    gaps = np.diff(allt)
    gaps = gaps[gaps > 1e-7]
    if dur <= 0.0 or len(gaps) == 0:
        return 1.0, 1
    step = float(gaps.min())
    n_keys = int(round(dur / step)) + 1
    return dur / (n_keys - 1), n_keys


def _sample_channel(inp: np.ndarray, out: np.ndarray, grid: np.ndarray,
                    interp: str, is_rotation: bool) -> np.ndarray:
    inp = inp.ravel()
    k = np.clip(np.searchsorted(inp, grid + 1e-7, side="right") - 1, 0, len(inp) - 1)
    k1 = np.minimum(k + 1, len(inp) - 1)
    span = inp[k1] - inp[k]
    u = np.where(span > 0, (grid - inp[k]) / np.where(span > 0, span, 1.0), 0.0)
    u = np.clip(u, 0.0, 1.0)
    if interp == INTERP_STEP:
        return out[k]
    if is_rotation:
        a = out[k] / np.linalg.norm(out[k], axis=-1, keepdims=True)
        b = out[k1] / np.linalg.norm(out[k1], axis=-1, keepdims=True)
        return quat_slerp(a, b, u)
    return out[k] + (out[k1] - out[k]) * u[:, None]


def build_clip_bank(path, clip_rows: Optional[Mapping[str, ClipRow]] = None,
                    ) -> ClipBank:
    """Read one look's ``.glb`` and bake every animation it holds.

    ``clip_rows`` maps a clip name to its :class:`ClipRow`; a clip without a
    row takes the defaults. A row naming a clip the file lacks is not an
    error here (looks may differ); the caller checks the names it needs."""
    path = Path(path)
    src = path.name
    doc, binary = read_glb(path)
    sk = read_skeleton(doc, binary, src)
    rows = dict(clip_rows or {})
    jpos = {node: i for i, node in enumerate(doc["skins"][0]["joints"])}
    J = sk.n_bones
    names: List[str] = []
    t_parts, r_parts, s_parts = [], [], []
    step_t, step_r, step_s = [], [], []
    offset, n_keys_l, key_dt, duration = [], [], [], []
    loop, rate, add_ref, root_mode = [], [], [], []
    any_scale = bool(np.abs(sk.rest_s - 1.0).max() > _IDENT_EPS)
    total = 0
    for a_i, anim in enumerate(doc.get("animations", [])):
        name = str(anim.get("name", f"clip{a_i}"))
        if name in names:
            raise PoseDataError(f"{src}: duplicate clip name {name!r}")
        what = f"clip {name!r}"
        channels = []
        for ch in anim["channels"]:
            tgt = ch["target"]
            node = tgt.get("node")
            if node not in jpos:
                continue                      # a non-joint target (raylib ignores it too)
            if tgt["path"] not in _PATHS:
                raise PoseDataError(f"{src}: {what} animates {tgt['path']!r} (refused)")
            smp = anim["samplers"][ch["sampler"]]
            interp = smp.get("interpolation", INTERP_LINEAR)
            if interp not in _SUPPORTED_INTERP:
                raise PoseDataError(f"{src}: {what} uses {interp} interpolation "
                                    f"(refused: only {_SUPPORTED_INTERP})")
            inp = _accessor(doc, binary, smp["input"], what=what, src=src)
            out = _accessor(doc, binary, smp["output"], what=what, src=src)
            channels.append((jpos[node], tgt["path"], interp, inp, out))
        step, K = _clip_grid([c[3] for c in channels] or [np.zeros(1)])
        grid = np.arange(K) * step
        ct = np.repeat(sk.rest_t[None], K, 0)
        cr = np.repeat(sk.rest_r[None], K, 0)
        cs = np.repeat(sk.rest_s[None], K, 0)
        st_t = np.zeros(J, bool)
        st_r = np.zeros(J, bool)
        st_s = np.zeros(J, bool)
        for bone, pth, interp, inp, out in channels:
            v = _sample_channel(inp, out, grid, interp, pth == "rotation")
            if pth == "translation":
                ct[:, bone] = v
                st_t[bone] = interp == INTERP_STEP
            elif pth == "rotation":
                cr[:, bone] = v / np.linalg.norm(v, axis=-1, keepdims=True)
                st_r[bone] = interp == INTERP_STEP
            else:
                cs[:, bone] = v
                st_s[bone] = interp == INTERP_STEP
        # hemisphere continuity along time (so a sub-frame nlerp never takes
        # the long way round between two neighbouring keys)
        if K > 1:
            dots = (cr[1:] * cr[:-1]).sum(-1)
            sign = np.cumprod(np.where(dots < 0.0, -1.0, 1.0), axis=0)
            cr[1:] *= sign[..., None]
        row = rows.get(name, ClipRow())
        if row.root_motion not in ROOT_MOTION_MODES:
            raise PoseDataError(f"{src}: clip {name!r} root_motion must be one of "
                                f"{ROOT_MOTION_MODES}, got {row.root_motion!r}")
        if row.root_motion == ROOT_MOTION_STRIP:
            for rb in sk.roots:
                for ax in _HORIZONTAL_AXES:
                    ct[:, rb, ax] = ct[0, rb, ax]
        if not 0 <= row.additive_ref < K:
            raise PoseDataError(f"{src}: clip {name!r} additive_ref {row.additive_ref} "
                                f"is outside its {K} keys")
        any_scale = any_scale or bool(np.abs(cs - 1.0).max() > _IDENT_EPS)
        names.append(name)
        t_parts.append(ct)
        r_parts.append(cr)
        s_parts.append(cs)
        step_t.append(st_t)
        step_r.append(st_r)
        step_s.append(st_s)
        offset.append(total)
        n_keys_l.append(K)
        key_dt.append(step)
        duration.append(step * (K - 1))
        loop.append(bool(row.loop))
        rate.append(float(row.rate))
        add_ref.append(int(row.additive_ref))
        root_mode.append(row.root_motion)
        total += K
    if not names:
        raise PoseDataError(f"{src}: no animations")
    f32 = np.float32
    return ClipBank(
        skeleton=sk, names=tuple(names),
        t=np.concatenate(t_parts).astype(f32),
        r=np.concatenate(r_parts).astype(f32),
        s=np.concatenate(s_parts).astype(f32) if any_scale else None,
        step_t=np.array(step_t), step_r=np.array(step_r), step_s=np.array(step_s),
        offset=np.array(offset, dtype=np.int64),
        n_keys=np.array(n_keys_l, dtype=np.int64),
        key_dt=np.array(key_dt), duration=np.array(duration),
        loop=np.array(loop), rate=np.array(rate),
        additive_ref=np.array(add_ref, dtype=np.int64),
        root_motion=tuple(root_mode))


__all__ = [
    "PoseDataError", "Skeleton", "ClipBank", "ClipRow", "read_glb",
    "read_skeleton", "build_clip_bank", "quat_mul", "quat_conj", "quat_nlerp",
    "quat_slerp", "quat_to_mat3", "mat3_to_quat", "trs_matrices",
    "ROOT_MOTION_KEEP", "ROOT_MOTION_STRIP", "ROOT_MOTION_MODES",
]
