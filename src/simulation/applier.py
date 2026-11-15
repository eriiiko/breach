"""THE PER-ENV SERIAL APPLIER's one table (#29 P6.4b; residency design §4.3,
"The per-env serial applier", and §4.4b's P6.4 row).

The applier is THE device form of every order-dependent law that writes the
map: ONE BLOCK per env (``blockIdx.y`` the env, the ``bgrid`` convention),
thread 0 walking that env's recorded EFFECTS in the reference's order, each
effect a DEVICE FUNCTION (``cpp/src/cuda_topology.cuh``: the topology seam --
``on_tile_changed``, ``destroy_wall``) whose book writes land in per-env BOOK
channels. This module is its ONE Python table, with generated offsets
(``cpp/src/applier_layout.h``, written by ``tools/gen_unit_layout.py --write``
beside the unit store's, the tables' and the event ring's; fingerprint-checked
by :class:`DeviceApplier`):

- :data:`FIELDS` / :data:`KINDS` -- an effect RECORD (int64 ``kind, y, x,
  arg``) and its kinds. Today ``DESTROY_WALL`` (slot 9's burn-through; P7's
  chew and blasts and P6b's door deaths append the same kind). A per-env list
  ``(N, cap, F)`` with a count per env: an effect past ``cap`` is never
  applied, the env's ``dropped`` book rises and the Python face refuses the
  run naming the kernel and the env (:class:`EffectOverflow`); the first
  ``cap`` are applied.
- :data:`BOOKS` -- the per-env book channels the effects write, each with the
  HOST quantity it twins (``gmap.<attr>`` or ``gmap.<dict>:<key>``), so the
  shadow harness compares a book with the host's delta over the slot and the
  chained harness lands it on the host.
- :data:`PLANES` -- the planes the topology seam reads and writes, by
  ``GameMap`` field name, each in its own dtype and per-cell width: what the
  applier's launch takes as device pointers to ``(N, ...)`` arrays. Most are
  resident fields (``GameMap._RESIDENT_FIELD_NAMES``: the arrays
  ``ResidentBatch.buffers`` holds); :func:`non_resident_planes` names the rest
  (P8b must hold them on the device).
- :data:`CLEAR_PLANES` -- the per-tick planes the tick-end clears zero.

A NEW EFFECT KIND, BOOK OR PLANE is a row here, ``tools/gen_unit_layout.py
--write``, commit, rebuild ``cpp/build_cuda``.

The fan-out of an effect's tile set over the block's threads with a barrier
between effects (§4.3: a blast's deposits, a reach walk's frontier) arrives
with its first user (P7c's deposits); the launch already has the block shape
so it can be added without moving the kernel.

NOT PART OF THE PLAYED TICK: Class T (training only), like the unit store.
Import-light: numpy; ``breach_physics`` only inside :class:`DeviceApplier`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

#: The generated header (C++'s copy of this table).
HEADER_REL = "cpp/src/applier_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"

#: An effect record's fields, in order.
FIELDS = ("kind", "y", "x", "arg")
F = len(FIELDS)
#: The effect kinds (code = position + 1; 0 = no record).
KINDS = ("DESTROY_WALL",)
KIND_DESTROY_WALL = 1
#: The default list capacity per env per launch.
E_CAP = 256


@dataclass(frozen=True)
class BookRow:
    """One per-env book channel: ``host`` names the host quantity it twins
    (``attr`` or ``attr:key`` on the GameMap; None = the applier's own
    count, with no host attribute)."""
    name: str
    host: str | None
    note: str = ""


BOOKS: tuple = (
    BookRow("n_destruction_seed", "n_destruction_seed_sum",
            "destroy_wall's measured bulk delta (design §5.10: the destroyed "
            "tile's own bulk delta)"),
    BookRow("ge_retire", "gas_energy_books:retire",
            "the gas energy seam's 'retire' channel (on_tile_changed's "
            "membership retire; a breach's evacuation)"),
    BookRow("ge_mint", "gas_energy_books:mint",
            "the 'mint' channel (on_tile_changed's born-at-ambient)"),
    BookRow("ge_destroy_seed", "gas_energy_books:destroy_seed",
            "destroy_wall's seed: the retire, then the one true mint"),
    BookRow("solid_destroy_wall", "solid_energy_books:destroy_wall",
            "the solid books' destroy_wall channel (its _solid_books_sink "
            "is the host's: the chained harness books it there)"),
    BookRow("tiles_changed", "_tile_serial",
            "on_tile_changed calls (the host's _tile_serial bump)"),
    BookRow("applied", None, "effects that acted (a destroy of a non-air "
            "tile in bounds)"),
    BookRow("dropped", None, "effects past the list's capacity (refused)"),
    BookRow("faults", None, "effects whose operands left the proved range "
            "(nothing written; refused)"),
)
BOOK_NAMES = tuple(b.name for b in BOOKS)
N_BOOKS = len(BOOKS)

#: The topology seam's planes: (GameMap field, numpy dtype, per-cell width;
#: "G" = the gas planes' count, the leading axis). Float planes are read and
#: written as their exact 32-bit pattern (never a float on the device).
PLANES: tuple = (
    ("material", "int8", 1),
    ("wall_hp", "int32", 1),
    ("gas", "int32", "G"),
    ("gas_energy", "int64", 1),
    ("temperature", "int32", 1),
    ("fire", "int32", 1),
    ("atmosphere", "int32", 1),
    ("solid", "bool", 1),
    ("is_vacuum", "bool", 1),
    ("is_ambient", "bool", 1),
    ("thermal_solid", "bool", 1),
    ("flammable", "bool", 1),
    ("sight_screen", "bool", 1),
    ("heat_inv_shift", "int32", 1),
    ("face_shift", "int32", 4),
    ("conductivity", "float32", 1),
    ("permeability", "float32", 1),
    ("wave_absorb", "float32", 1),
    ("heat_atten", "float32", 1),
    ("heat_atten_q", "int32", 1),
    ("dyn_heat_atten_q", "int32", 1),
    ("light_atten", "float32", 3),
    ("light_atten_q", "int32", 3),
    ("dyn_light_atten_q", "int32", 3),
    ("fuel_recip", "int64", 1),
    ("fire_T_ext_plane", "int32", 1),
    ("fuel_per_o2_plane", "int32", 1),
)
PLANE_NAMES = tuple(p[0] for p in PLANES)
N_PLANES = len(PLANES)

#: The tick-end clears' planes (Simulation._clear_tick_buffers, with a
#: physics runner): (GameMap field, dtype). The credit claims' clear is P7's
#: (its device form is the claims table); the shot tracers' expiry is render.
CLEAR_PLANES: tuple = (
    ("heat", "int32"),
    ("rad_net", "int64"),
    ("rad_amb", "int64"),
    ("rad_flux", "int64"),
)
CLEAR_NAMES = tuple(p[0] for p in CLEAR_PLANES)


class EffectOverflow(RuntimeError):
    """An env's effects did not fit the applier's list (its capacity)."""


class ApplierFault(RuntimeError):
    """An effect's operands left the range the applier was proved for."""


def non_resident_planes() -> tuple:
    """The planes the topology seam writes that are NOT resident fields today
    (P8b must hold them on the device)."""
    from simulation.gamemap import GameMap
    res = set(GameMap._RESIDENT_FIELD_NAMES)
    return tuple(n for n in PLANE_NAMES + CLEAR_NAMES if n not in res)


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    fields: tuple
    kinds: tuple
    books: tuple
    planes: tuple
    clears: tuple
    fingerprint: str

    @property
    def width(self) -> int:
        return len(self.fields)


_LAYOUT_CACHE: dict = {}


def layout() -> Layout:
    key = (FIELDS, KINDS, BOOK_NAMES, PLANES, CLEAR_PLANES)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    text = "".join(f"field:{i}:{n}\n" for i, n in enumerate(FIELDS))
    text += "".join(f"kind:{i + 1}:{n}\n" for i, n in enumerate(KINDS))
    text += "".join(f"book:{i}:{n}\n" for i, n in enumerate(BOOK_NAMES))
    text += "".join(f"plane:{i}:{n}:{d}:{c}\n"
                    for i, (n, d, c) in enumerate(PLANES))
    text += "".join(f"clear:{i}:{n}:{d}\n"
                    for i, (n, d) in enumerate(CLEAR_PLANES))
    lay = Layout(fields=FIELDS, kinds=KINDS, books=BOOK_NAMES, planes=PLANES,
                 clears=CLEAR_PLANES,
                 fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest())
    _LAYOUT_CACHE[key] = lay
    return lay


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/applier_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/applier.py",
        "// (#29 P6.4b, residency design 4.3 / 4.4b) -- NEVER EDIT BY HAND.",
        "// The per-env serial applier: an effect record of APPLIER_F int64,",
        "// its kinds; the per-env book channels; the topology seam's planes",
        "// (the order of the launch's pointer table) and the tick-end clears'",
        "// planes. A new kind / book / plane: a row of simulation/applier.py,",
        f"// then `{GENERATOR}` and commit; tests/test_applier_layout.py",
        "// refuses a stale copy.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace applier {",
        "",
        f'constexpr const char* APPLIER_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int APPLIER_F = {lay.width};",
        "",
        "// An effect record's fields.",
    ]
    for i, n in enumerate(lay.fields):
        lines.append(f"constexpr int APPLIER_FX_{n.upper()} = {i};")
    lines += ["", "// The effect kinds (0 = no record)."]
    for i, n in enumerate(lay.kinds):
        lines.append(f"constexpr int64_t APPLIER_KIND_{n.upper()} = {i + 1};")
    lines += ["", "// The per-env book channels."]
    for i, n in enumerate(lay.books):
        lines.append(f"constexpr int APPLIER_BOOK_{n.upper()} = {i};")
    lines.append(f"constexpr int APPLIER_N_BOOKS = {len(lay.books)};")
    lines += ["", "// The topology seam's planes: the launch's pointer table order."]
    for i, (n, _d, _c) in enumerate(lay.planes):
        lines.append(f"constexpr int APPLIER_PLANE_{n.upper()} = {i};")
    lines.append(f"constexpr int APPLIER_N_PLANES = {len(lay.planes)};")
    lines += ["", "// The tick-end clears' planes (their element size in bytes)."]
    for i, (n, d) in enumerate(lay.clears):
        lines.append(f"constexpr int APPLIER_CLEAR_{n.upper()} = {i};")
        lines.append(f"constexpr int APPLIER_CLEAR_{n.upper()}_BYTES = "
                     f"{np.dtype(d).itemsize};")
    lines.append(f"constexpr int APPLIER_N_CLEARS = {len(lay.clears)};")
    lines += ["", "}  // namespace applier", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Effect lists
# ---------------------------------------------------------------------------
def destroy_records(tiles) -> np.ndarray:
    """A list of ``(y, x)`` tiles as DESTROY_WALL records ``(R, F)``."""
    t = np.asarray(list(tiles), dtype=np.int64).reshape(-1, 2)
    out = np.zeros((t.shape[0], F), dtype=np.int64)
    out[:, 0] = KIND_DESTROY_WALL
    out[:, 1:3] = t
    return out


def pack(per_env, cap: int = E_CAP):
    """N envs' ``(R_e, F)`` records as the list the device walks: ``recs``
    ``(N, cap, F)`` (the first ``cap`` of each env) and ``counts`` ``(N,)``
    -- each env's REQUESTED count, which may exceed ``cap`` (the device
    applies ``cap`` and counts the rest in its ``dropped`` book)."""
    per_env = [np.asarray(r, dtype=np.int64).reshape(-1, F) for r in per_env]
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"applier: cap {cap} must be >= 1")
    recs = np.zeros((len(per_env), cap, F), dtype=np.int64)
    counts = np.zeros(len(per_env), dtype=np.int64)
    for e, r in enumerate(per_env):
        k = min(cap, r.shape[0])
        recs[e, :k] = r[:k]
        counts[e] = r.shape[0]
    return recs, counts


# ---------------------------------------------------------------------------
# The device applier
# ---------------------------------------------------------------------------
class DeviceApplier:
    """THE per-env effect list ``(N, cap, F)`` and book channels ``(N,
    N_BOOKS)`` on the tick stream (``breach_physics.FieldApplierBuffer``, the
    CUDA build only). :meth:`stage` uploads a packed list; the launch
    (``simulation.unit_kernels.apply_effects``) zeroes the books, walks every
    env's list and downloads the books. Refuses a compiled module whose
    applier layout is not this table's."""

    def __init__(self, n_env: int, cap: int = E_CAP, bp=None):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError("applier: the device applier needs the CUDA "
                               "build (breach_physics.HAS_CUDA)")
        lay = layout()
        compiled = bp.applier_layout()
        if compiled["fingerprint"] != lay.fingerprint \
                or int(compiled["f"]) != lay.width \
                or int(compiled["n_books"]) != N_BOOKS \
                or int(compiled["n_planes"]) != N_PLANES:
            raise ValueError(
                f"applier: the compiled layout ({compiled['fingerprint'][:12]}) "
                f"is not simulation.applier's ({lay.fingerprint[:12]}): run "
                f"`{GENERATOR}` and rebuild cpp/build_cuda")
        self.bp = bp
        self.n_env = int(n_env)
        self.cap = int(cap)
        self._buf = bp.FieldApplierBuffer(self.n_env, self.cap)

    @property
    def buffer(self):
        return self._buf

    def stage(self, recs, counts) -> None:
        recs = np.ascontiguousarray(recs, dtype=np.int64)
        counts = np.ascontiguousarray(counts, dtype=np.int64).reshape(-1)
        if recs.shape != (self.n_env, self.cap, F) or counts.shape != (self.n_env,):
            raise ValueError(f"applier: a list must be ({self.n_env}, {self.cap}, "
                             f"{F}) with ({self.n_env},) counts, not "
                             f"{recs.shape} / {counts.shape}")
        self._buf.stage(recs, counts)


def books_dict(books) -> dict:
    """``(N, N_BOOKS)`` as ``{name: (N,) int64}``."""
    b = np.asarray(books, dtype=np.int64)
    return {n: b[:, i].copy() for i, n in enumerate(BOOK_NAMES)}


def host_book(gmap, ref: str) -> int:
    """The host quantity a book twins (``attr`` or ``attr:key``)."""
    if ":" in ref:
        attr, key = ref.split(":", 1)
        return int(getattr(gmap, attr).get(key, 0))
    return int(getattr(gmap, ref))


__all__ = ["HEADER_REL", "GENERATOR", "FIELDS", "F", "KINDS",
           "KIND_DESTROY_WALL", "E_CAP", "BookRow", "BOOKS", "BOOK_NAMES",
           "N_BOOKS", "PLANES", "PLANE_NAMES", "N_PLANES", "CLEAR_PLANES",
           "CLEAR_NAMES", "EffectOverflow", "ApplierFault",
           "non_resident_planes", "Layout", "layout", "header_text",
           "destroy_records", "pack", "DeviceApplier", "books_dict",
           "host_book"]
