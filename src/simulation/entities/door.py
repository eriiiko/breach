"""The `door` entity class — Arc A patch A6, doors v0.

Design: docs/a6_doors_v0_impl_2026-07-19.md (v2, critique folded). This
module carries the SCHEMA (fields / signals / inputs, §2) and THE canonical
span quantization (§3) — the one rule the loader, the slot-9e sweep, the
editor (Arc C) and the migration tool (A7) all consume. Runtime state and
the sweep live sim-side in :mod:`simulation.door_system`; this package
stays import-light (stdlib only — design §3b, CI-tested).

Field semantics (§2a):

- ``x``/``y`` are the anchor TILE at BASE resolution (col, row of the
  span's first tile — leftmost for "h", topmost for "v"), integers by
  construction (the editor's DOOR tool wall-run-snaps placement).
- ``length_m`` + ``orientation`` parameterize a straight run; tiles are
  ALWAYS derived (§3), never stored. length_m is authoring-bound
  (KIND_LENGTH_M — never hashed, A4 critique blocker 1); its synced
  consequence is the material grid.
- ``initial_state`` picks the load-time stamp (§4): "closed" →
  MAT_DOOR_CLOSED, "open" → plain MAT_AIR.

Runtime digest rows (§2c): ``state`` (0 CLOSED / 1 OPEN / 2 DESTROYED),
``want_open`` (the synced desired-state latch — Erik's ruling 5), and one
``hp_i`` Q16.16 row per runtime span tile in row-major span order (the S4
pin: the runtime span list IS the replicated tile set sorted row-major and
``hp_i`` indexes that list everywhere). Row count is span-length-dependent
but constant for an instance's lifetime (rows stay present, zeroed, after
destruction). Called on a bare ``EntityInstance`` (no runtime attrs) this
raises ``AttributeError`` — loud, a bug, never a fallback: digests are only
captured from constructed sims (§6.1).
"""
from __future__ import annotations

from fractions import Fraction

from simulation.entities.schema import (
    Entity, Field, INPUT_HELD, InputDecl, KIND_ENUM, KIND_INT,
    KIND_LENGTH_M, Signal, register,
)

# Runtime `state` row values (§2c).
DOOR_CLOSED = 0
DOOR_OPEN = 1
DOOR_DESTROYED = 2


@register
class door(Entity):
    """One entity door: a straight wall-run span that seals/opens (§2)."""

    INTANGIBLE = False   # physical by default (entity doc §5)

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="anchor tile COL at base resolution (span's leftmost/"
                  "topmost tile) — REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="anchor tile ROW at base resolution — REQUIRED"),
        Field("orientation", KIND_ENUM, default="h", choices=("h", "v"),
              doc="span direction: h = along +x (cols), v = along +y (rows)"),
        Field("length_m", KIND_LENGTH_M, default=1.0, minimum=0.0,
              doc="span length in meters (quantized once at load, §3); "
                  "must be > 0 — the quantizer enforces strictness (N10: "
                  "schema minimums are inclusive)"),
        Field("initial_state", KIND_ENUM, default="closed",
              choices=("closed", "open"),
              doc="authored state; the load-time stamp (§4)"),
    )

    # Format-reserved, inert in v1 (§2b): Arc B's SignalBus adds drivers
    # without touching levels. Nothing emits/consumes these yet.
    SIGNALS = (Signal("is_open", "1 while the door is OPEN (Arc B emits)"),)
    INPUTS = (InputDecl("open", INPUT_HELD),
              InputDecl("close", INPUT_HELD))
    INPUT_PRIORITY = (("close", "open"),)   # close beats open (safe state)
    INTERACTIONS = ()

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """§2c: state / want_open / per-tile hp rows off the runtime object.

        Reads the runtime attrs plainly — a bare EntityInstance raises
        AttributeError (loud; digests only come from constructed sims).
        """
        rows = [("state", int(entity.state)),
                ("want_open", 1 if entity.want_open else 0)]
        rows.extend((f"hp_{i}", int(v)) for i, v in enumerate(entity.hp))
        return tuple(rows)


# ---------------------------------------------------------------------------
# THE canonical span quantization (§3) — exact Fraction arithmetic, one rule.
# ---------------------------------------------------------------------------

def tiles_per_m(tile_size_m) -> int:
    """Exact integer tiles-per-meter for a level's BASE ``tile_size_m``.

    The shipped ``0.333`` maps to EXACTLY 3 (the editor-doc §4 migration
    rule); any other value converts through ``1 / Fraction(str(...))`` and
    must be integral or the level has no defined door quantization (hard
    ``ValueError``). Callers at ``--res`` MUST pass the BASE tile size
    (``LevelData.tile_size_m_base`` when set — the S1 recovery), never the
    scaled one.
    """
    ts = Fraction(str(float(tile_size_m)))
    if ts <= 0:
        raise ValueError(f"tile_size_m must be > 0, got {tile_size_m!r}")
    if ts == Fraction(333, 1000):        # the shipped 0.333 → exactly 3
        return 3
    tpm = Fraction(1, 1) / ts
    if tpm.denominator != 1:
        raise ValueError(
            f"tile_size_m {tile_size_m!r} gives non-integral tiles-per-meter "
            f"{tpm} — door quantization is undefined for this level "
            f"(a6 doors design §3; the integer-tiles_per_m format "
            f"migration is Arc C's)")
    return int(tpm)


def quantize_span_tiles(length_m, tile_size_m, *, context: str = "door") -> int:
    """§3: ``n_base = floor(length_m * tiles_per_m + 1/2)`` — round-half-up
    in exact Fraction arithmetic (never banker's ``round``), clamped >= 1.

    ``length_m`` ingresses as ``Fraction(str(length_m))`` (N10 pin — the
    decimal the author typed, not the binary float's expansion) and must be
    strictly positive (explicit check: schema minimums are inclusive).
    """
    lm = Fraction(str(float(length_m)))
    if lm <= 0:
        raise ValueError(
            f"{context}: length_m must be > 0, got {length_m!r}")
    n = (lm * tiles_per_m(tile_size_m) + Fraction(1, 2)).__floor__()
    return max(1, int(n))


def base_span(fields: dict, tile_size_m, *, context: str = "door") -> list:
    """The BASE-resolution span: ``n_base`` tiles from the anchor along
    ``orientation``, as gamemap ``(fy, fx)`` (row, col) tuples in span
    order — "h" walks +x, "v" walks +y (§3)."""
    n = quantize_span_tiles(fields["length_m"], tile_size_m, context=context)
    x, y = int(fields["x"]), int(fields["y"])
    if fields["orientation"] == "h":
        return [(y, x + i) for i in range(n)]
    return [(y + i, x) for i in range(n)]


def runtime_span(fields: dict, tile_size_m, res_factor: int = 1,
                 *, context: str = "door") -> list:
    """The RUNTIME tile set: the base span replicated ``res_factor`` x
    ``res_factor`` per base tile (each base tile (fy, fx) becomes the NxN
    block at (N*fy .., N*fx ..) — exactly what ``np.repeat`` does to the
    painted grid), then ROW-MAJOR SORTED. This sorted list IS the pinned
    hp_i <-> tile order (S4): every consumer indexes it, nobody re-derives.
    Never re-quantized from meters at the scaled resolution (§3).
    """
    n = int(res_factor)
    if n < 1:
        raise ValueError(f"{context}: res_factor must be >= 1, got {res_factor!r}")
    tiles = []
    for (fy, fx) in base_span(fields, tile_size_m, context=context):
        for dy in range(n):
            for dx in range(n):
                tiles.append((n * fy + dy, n * fx + dx))
    tiles.sort()                         # row-major — the ONE span order
    return tiles
