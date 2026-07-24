"""Canonical per-tick FIELD DIGEST — the cross-GPU bit-identity gate (CUDA-S0).

The CUDA port's contract is that every SYNCED INTEGER field is byte-for-byte
identical whether the math ran on the CPU or a GPU, and across GPU architectures
(Ampere sm_86 today, Ada sm_89 on the Lenovo, Turing sm_75 opportunistically).
The same-machine A/B harness (``field_ab_harness.diff_trajectories``) LOCATES a
divergence per-cell; this module is the fast, position-stable EQUALITY check and
the artifact recorded per machine for the X-ARCH gate (``xarch_digest.py``).

Why a separate, FROZEN spec: a digest is only a cross-machine proof if both
machines serialize the SAME fields, in the SAME order, with the SAME dtype and
byte order. We freeze that contract here (``DIGEST_SPEC_VERSION`` +
``DIGEST_FIELDS``) and in the committed ``field_digest_spec.toml``. A field whose
dtype drifts from the spec raises (loudly) rather than silently changing the hash
— if a field legitimately changes dtype/shape, BUMP the version and regenerate
every golden in the same commit (the §7.6 gate-7 rule).

SCOPE — int fields only. The five remaining float sim fields (``ripple``,
``ripple_v``, ``dyn_permeability``, ``dyn_wave_absorb``, ``dyn_light_atten``) are
EXCLUDED: they are render/lighting-bound or a known residual float boundary, NOT
part of the cross-GPU integer bit-identity contract. The same-machine A/B harness
still covers them. The synced UNIT state (HP/life/events) rides in via
``tick_digest`` (folding ``field_ab_harness``'s unit hash) so a fire->heat->kill
desync on the Q2-fenced float-HP path is caught alongside the fields. A4 adds
the ``__entity__``/``__signals__`` sections the same way — absence-transparent
(folded ONLY when entities are present, via the snapshot's ``__entity__``
presence carrier), section-local versioned (ENTITY_SECT_V1 / SIGNAL_SECT_V1),
serialized by the ONE canonical ``simulation.entities.serialize`` module.

Endianness: x86 and CUDA are both little-endian, so the raw int bytes compare
directly. A big-endian dtype is REFUSED so a silent byteorder mismatch can never
forge a match.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Bump this (and regenerate every golden) whenever DIGEST_FIELDS changes shape,
# order, dtype, or membership. The version is hashed into the digest so a stale
# golden compared against a new spec fails loudly instead of mismatching opaquely.
# v2 (2026-07-24): +ignition_armed — the edge-triggered ignition arm bool
# (combat.apply_temperature_ignition; Fable zombie-smolder ruling) is synced state.
DIGEST_SPEC_VERSION = 2

# The frozen (name, dtype-string) contract — the integer/bool SYNCED fields, in a
# fixed order. `gas` is the 3D (5,h,w) multi-gas stack (covers `smoke`, a view).
# Topology fields (wall_hp/material/obstacles/is_vacuum) are included: a wall
# destruction mutates them and that IS synced state. `ignition_armed` (v2) is the
# per-tile edge-trigger arm — synced so an arm/disarm desync (which forks ignition
# timing, hence the fire/HP stream) is caught directly.
DIGEST_FIELDS = (
    ("atmosphere",     "int32"),
    ("wave_p",         "int32"),
    ("wave_v",         "int32"),
    ("wave_source",    "int32"),
    ("wind_x",         "int32"),
    ("wind_y",         "int32"),
    ("gas",            "int32"),
    ("fire",           "int32"),
    ("water_depth",    "int32"),
    ("flow_vx",        "int32"),
    ("flow_vy",        "int32"),
    ("heat",           "int32"),
    ("temperature",    "int32"),
    ("wall_hp",        "int32"),
    ("material",       "int8"),
    ("obstacles",      "bool"),
    ("is_vacuum",      "bool"),
    ("ignition_armed", "bool"),
)

# Float sim fields deliberately NOT in the cross-GPU integer digest (documented).
EXCLUDED_FLOAT_FIELDS = (
    "ripple", "ripple_v", "dyn_permeability", "dyn_wave_absorb", "dyn_light_atten",
)


def _field_bytes(name: str, arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == ">":
        raise ValueError(f"digest field '{name}': big-endian dtype unsupported")
    # Header pins name + exact dtype (incl. byteorder via .str) + shape, so a
    # reshaped/retyped field cannot collide with a different one.
    header = f"{name}|{a.dtype.str}|{a.shape}".encode("ascii")
    return header + a.tobytes(order="C")


def field_digest(snapshot: dict, *, strict: bool = True) -> str:
    """blake2b-256 hex over the frozen integer fields of one tick snapshot."""
    h = hashlib.blake2b(digest_size=32)
    h.update(f"FIELD_DIGEST_V{DIGEST_SPEC_VERSION}\n".encode("ascii"))
    for name, expect_dtype in DIGEST_FIELDS:
        if name not in snapshot:
            if strict:
                raise KeyError(f"digest field '{name}' missing from snapshot")
            continue
        arr = snapshot[name]
        if str(arr.dtype) != expect_dtype:
            raise TypeError(
                f"digest field '{name}' dtype {arr.dtype} != spec '{expect_dtype}' "
                f"(spec v{DIGEST_SPEC_VERSION} stale — bump DIGEST_SPEC_VERSION + "
                f"regenerate goldens)")
        h.update(_field_bytes(name, arr))
    return h.hexdigest()


def tick_digest(snapshot: dict) -> str:
    """field_digest folded with the synced unit-state hash (HP/life/event stream),
    so a kill/HP desync that leaves every gmap cell identical still changes it.

    A4 (docs/a4_digest_impl_note_2026-07-18.md): the ``__entity__`` /
    ``__signals__`` sections fold in ABSENCE-TRANSPARENTLY — appended only
    when the snapshot's presence carrier says ``n_entities > 0``, so an
    entity-free snapshot's hashed byte stream is bit-identical to pre-A4
    (the entity-free tail is hex-only; the ``|__entity__|`` marker can never
    appear in it) and every existing golden stays valid unchanged. Pre-A4
    snapshots (no carrier key) are entity-free by construction; the strict
    presence rule (a sim WITH entities must always write the carrier) is
    enforced at capture (``simulation.entities.serialize.
    require_entity_carrier``). The ENTITY/SIGNAL sections do NOT drive
    DIGEST_SPEC_VERSION (they carry their own version in the hashed
    ENTITY_SECT_V1 preamble); the field-plane spec bumps it independently
    (v2 added ``ignition_armed``).
    """
    from field_ab_harness import UNIT_DIGEST_KEY  # local: avoid import cycle
    from simulation.entities.serialize import (   # local: path set by ^
        ENTITY_DIGEST_KEY, entity_section_bytes, signal_section_bytes)
    fd = field_digest(snapshot)
    unit_hash = ""
    if UNIT_DIGEST_KEY in snapshot:
        unit_hash = snapshot[UNIT_DIGEST_KEY]["hash"]
    h = hashlib.blake2b(digest_size=32)
    h.update(fd.encode("ascii"))
    h.update(b"|")
    h.update(unit_hash.encode("ascii"))
    carrier = snapshot.get(ENTITY_DIGEST_KEY)
    if carrier is not None and carrier["n_entities"] > 0:
        eh = hashlib.blake2b(entity_section_bytes(carrier),
                             digest_size=32).hexdigest()
        sh = hashlib.blake2b(signal_section_bytes(carrier),
                             digest_size=32).hexdigest()
        h.update(b"|__entity__|")
        h.update(eh.encode("ascii"))
        h.update(b"|__signals__|")
        h.update(sh.encode("ascii"))
    return h.hexdigest()


def trajectory_digest(traj: list) -> str:
    """A single hash over a whole trajectory (per-tick tick_digests folded in
    order) — the one number two machines compare for the X-ARCH gate."""
    h = hashlib.blake2b(digest_size=32)
    h.update(f"TRAJ_DIGEST_V{DIGEST_SPEC_VERSION}\n".encode("ascii"))
    for snap in traj:
        h.update(tick_digest(snap).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()
