"""Physics events that read/write GameMap fields.

These are not per-tick solver passes (those live in :mod:`simulation.physics_runner`).
They are discrete physical events triggered by gameplay — explosions are the
canonical example: a grenade detonates, an explosive charge fires, a wall is
shattered, all going through the same entry point that updates pressure +
fire + walls + smoke in one go.

Field WRITES now go through the canonical write primitive
(:mod:`simulation.field_edit`): the explosion / smoke deposits are **enqueued**
as :class:`~simulation.field_edit.FieldEdit`s on an :class:`EditQueue`, and the
Simulation flushes the whole queue at one fixed tick point (before the physics
solvers run), in a deterministic stable-sorted order. The disc/falloff/skip
logic that used to be re-derived inline is now expressed declaratively as edits;
only **structural** wall damage (``wall_hp -= ...`` + ``destroy_wall``) stays
immediate (it changes topology, not a continuous field — see engine/13).

The plan keeps explosion side effects bundled here on purpose (see
``docs/patch_game_logic_migration.md`` and the matching reasoning in the
C++ code where fire damages walls and atmosphere drains through breaches as
part of a single physical step). Combat / explosives / future weapons call
into this module — they never reach into individual physics fields directly.
"""
from __future__ import annotations

import math

from config import CFG
from simulation.gamemap import MAT_HULL, MAT_WOOD, MAT_DOOR
from simulation.field_edit import (
    FieldEdit, EditMode, Region, Falloff,
)

# source_id namespace for explosion-issued edits (engine/13 stable-sort key).
# Distinct ids keep an explosion's field families grouped + ordered in the flush
# independently of any other emitter.
_SRC_EXPLOSION = 1
_SRC_EXPLOSION_SMOKE = 2

# A "set this tile to 0" REMOVE needs an amount large enough to drive any
# in-range value below the clamp floor; smoke lives in [0, 1] so 1.0 suffices,
# but we use a generous margin to be unambiguous (clamp catches the rest).
_SMOKE_CLEAR_AMOUNT = 1e6


def apply_explosion(gmap, queue, fy, fx, radius, pressure, wall_damage):
    """Lay down an explosion at (fy, fx) with the given radius / pressure / wall HP damage.

    Behaviour-preserving migration of the legacy inline field mutations onto the
    :class:`~simulation.field_edit.EditQueue` (engine/13). Side effects:

    - Damages every solid tile within ``radius`` (HP -= ``wall_damage * falloff``);
      tiles reaching 0 HP are destroyed via :meth:`GameMap.destroy_wall`. This is
      **structural** and stays immediate — it is NOT a FieldEdit (engine/13's
      one carve-out: edits that change topology are not field edits).
    - Enqueues a smoothed 3x3 kernel of pressure into ``wave_source``
      (the propagating shockwave) and a direct ``atmosphere`` boost (the
      sustained wind that drives smoke) — both ADD edits.
    - Enqueues a smoke-clear (REMOVE-to-0) over the inner 40 percent of the radius.
    - Enqueues an ``fire = max(...)`` ignite (MAX edit) over flammable tiles
      inside 70 percent of the radius.

    The deposits are not applied here — they land when the Simulation flushes the
    queue (before the solvers), in deterministic stable-sorted order. ``queue``
    is the sim's :class:`EditQueue`.

    No unit damage here — call :func:`simulation.combat.apply_blast_damage`
    separately.
    """
    h, w = gmap.material.shape

    # Atmosphere boost: a clean DISC ADD with LINEAR falloff (== the old
    # ``atmosphere += pressure * (1 - dist/radius)`` over non-solid/non-vacuum
    # tiles). The field policy supplies the skip-mask (solid) and we add vacuum
    # explicitly to mirror the legacy ``not is_vacuum`` guard.
    queue.enqueue(FieldEdit(
        field="atmosphere", region=Region.DISC, coords=(fy, fx, float(radius)),
        amount=float(pressure), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        source_id=_SRC_EXPLOSION,
    ))

    # Smoke clear over the inner 40 percent — a REMOVE-to-0 (large amount, the
    # smoke policy's [0, 1] clamp drives it to exactly 0). FLAT so every inner
    # tile is fully cleared, matching the old ``smoke[...] = 0.0``.
    inner = float(radius) * 0.4
    if inner > 0.0:
        queue.enqueue(FieldEdit(
            field="smoke", region=Region.DISC, coords=(fy, fx, inner),
            amount=_SMOKE_CLEAR_AMOUNT, mode=EditMode.REMOVE, falloff=Falloff.FLAT,
            clamp=(0.0, 1.0), source_id=_SRC_EXPLOSION,
        ))

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = fy + dy, fx + dx
            if 0 <= ny < h and 0 <= nx < w:
                dist = math.sqrt(dy * dy + dx * dx)
                if dist <= radius:
                    falloff = 1.0 - (dist / radius)
                    # --- Structural wall damage (immediate, NOT a FieldEdit) ---
                    if gmap.material[ny, nx] in (MAT_HULL, MAT_WOOD, MAT_DOOR):
                        gmap.wall_hp[ny, nx] -= wall_damage * falloff
                        if gmap.wall_hp[ny, nx] <= 0:
                            gmap.destroy_wall(ny, nx)
                    if not gmap.solid[ny, nx] and not gmap.is_vacuum[ny, nx]:
                        # Smoothed 3x3 deposit into wave_source (shockwave). The
                        # per-tile stencil spread is not a single region, so we
                        # enqueue the 9 weighted TILE ADDs here — their `seq`
                        # preserves the exact original summation order
                        # (bit-identical float accumulation through the flush).
                        amount = pressure * falloff
                        for ky, kx, kw in [
                                (0, 0, 4),
                                (-1, 0, 2), (1, 0, 2), (0, -1, 2), (0, 1, 2),
                                (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]:
                            sy, sx = ny + ky, nx + kx
                            if (0 <= sy < h and 0 <= sx < w
                                    and not gmap.solid[sy, sx]
                                    and not gmap.is_vacuum[sy, sx]):
                                queue.enqueue(FieldEdit(
                                    field="wave_source", region=Region.TILE,
                                    coords=(sy, sx),
                                    amount=amount * kw / 16.0,
                                    mode=EditMode.ADD,
                                    source_id=_SRC_EXPLOSION,
                                ))
                    # Ignite flammable tiles within 70 percent of the radius. The
                    # MEMBERSHIP radius (0.7 r) differs from the FALLOFF radius
                    # (full r), so this is a per-tile MAX edit with the
                    # pre-computed ``0.5 * falloff`` amount rather than one DISC
                    # edit (which would couple the two radii). The fire policy's
                    # non-flammable skip-mask + [0, 1] clamp are applied at flush.
                    if gmap.flammable[ny, nx] and dist <= radius * 0.7:
                        queue.enqueue(FieldEdit(
                            field="fire", region=Region.TILE, coords=(ny, nx),
                            amount=0.5 * falloff, mode=EditMode.MAX,
                            clamp=(0.0, 1.0), source_id=_SRC_EXPLOSION,
                        ))


def add_explosion_smoke(gmap, queue, fy, fx, radius, noise=None):
    """Enqueue a noisy smoke disc into ``gmap.smoke`` (engine/13 ADD edit).

    Behaviour-preserving migration of the legacy inline deposit onto the
    :class:`EditQueue`. Each tile gets ``base * mult`` where
    ``base = 0.8 * (1 - dist/radius)`` (a DISC + LINEAR falloff with
    ``amount = 0.8``) and ``mult`` is the per-tile noise multiplier drawn uniform
    in ``[1 - noise, 1.0]`` — but the draw now happens **at flush time**, from
    the sim's seeded ``rng``, in the queue's deterministic per-tile order, so the
    seeded-rollout guarantee is structural (one RNG consumer) rather than a
    per-caller convention.

    ``noise`` is the per-tile contrast knob (ch.05 §4 "explosion smoke noise too
    subtle"):

    - ``noise = 0``    -> ``mult == 1`` everywhere: a flat blob, no texture.
    - ``noise = 0.6``  -> ``[0.4, 1.0]``: the old shipped look (too subtle).
    - ``noise = 0.85`` (default) -> ``[0.15, 1.0]``: ragged holes / missing
      patches — visible initial structure for advection to grab and carry.
    - ``noise = 1.0``  -> ``[0.0, 1.0]``: maximal contrast / holes.

    When ``noise`` is ``None`` it is read from ``CFG.physics.explosion_smoke_noise``
    so the look is config-tunable; callers (e.g. the demo dial) may override it.

    The smoke policy supplies the skip-mask (``solid``) and the [0, 1] clamp, so
    a deposited tile reproduces the old ``min(1, smoke + base*mult)`` and a solid
    tile is skipped (and draws no RNG — the per-tile draw order matches the
    legacy nested-loop order, keeping the deposit bit-identical for a fixed seed).
    """
    if noise is None:
        noise = float(getattr(CFG.physics, "explosion_smoke_noise", 0.85))
    # Clamp so the multiplier range [low, 1] stays well-formed: noise in [0, 1].
    noise = min(1.0, max(0.0, noise))
    queue.enqueue(FieldEdit(
        field="smoke", region=Region.DISC, coords=(fy, fx, float(radius)),
        amount=0.8, mode=EditMode.ADD, falloff=Falloff.LINEAR,
        clamp=(0.0, 1.0), noise=noise, source_id=_SRC_EXPLOSION_SMOKE,
    ))
