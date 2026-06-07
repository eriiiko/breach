"""Physics events that read/write GameMap fields.

These are not per-tick solver passes (those live in :mod:`simulation.physics_runner`).
They are discrete physical events triggered by gameplay — explosions are the
canonical example: a grenade detonates, an explosive charge fires, a wall is
shattered, all going through the same entry point that updates pressure +
fire + walls + smoke in one go.

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


def apply_explosion(gmap, fy, fx, radius, pressure, wall_damage):
    """Lay down an explosion at (fy, fx) with the given radius / pressure / wall HP damage.

    Lifted verbatim from ``game.py:Physics.apply_explosion`` (lines 704-741).
    Side effects on ``gmap``:

    - Damages every solid tile within ``radius`` (HP -= ``wall_damage * falloff``);
      tiles reaching 0 HP are destroyed via :meth:`GameMap.destroy_wall`.
    - Deposits a smoothed 3x3 kernel of pressure into ``wave_source``
      (creates the propagating shockwave) and a direct boost into
      ``atmosphere`` (the sustained wind that drives smoke).
    - Clears smoke in the inner 40 percent of the radius.
    - Ignites flammable tiles inside 70 percent of the radius.

    No unit damage here — call :func:`simulation.combat.apply_blast_damage`
    separately.
    """
    h, w = gmap.material.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = fy + dy, fx + dx
            if 0 <= ny < h and 0 <= nx < w:
                dist = math.sqrt(dy * dy + dx * dx)
                if dist <= radius:
                    falloff = 1.0 - (dist / radius)
                    # Damage every wall type (hull, wood, door)
                    if gmap.material[ny, nx] in (MAT_HULL, MAT_WOOD, MAT_DOOR):
                        gmap.wall_hp[ny, nx] -= wall_damage * falloff
                        if gmap.wall_hp[ny, nx] <= 0:
                            gmap.destroy_wall(ny, nx)
                    if not gmap.solid[ny, nx] and not gmap.is_vacuum[ny, nx]:
                        # Smoothed 3x3 deposit into wave_source (shockwave).
                        amount = pressure * falloff
                        for ky, kx, kw in [
                                (0, 0, 4),
                                (-1, 0, 2), (1, 0, 2), (0, -1, 2), (0, 1, 2),
                                (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]:
                            sy, sx = ny + ky, nx + kx
                            if (0 <= sy < h and 0 <= sx < w
                                    and not gmap.solid[sy, sx]
                                    and not gmap.is_vacuum[sy, sx]):
                                gmap.wave_source[sy, sx] += amount * kw / 16.0
                        # Direct atmosphere boost (sustained pressure gradient
                        # — drives smoke transport; IMEX absorbs the spike).
                        gmap.atmosphere[ny, nx] += pressure * falloff
                    if dist <= radius * 0.4:
                        gmap.smoke[ny, nx] = 0.0
                    # Ignite flammable tiles within 70 percent of the radius.
                    if gmap.flammable[ny, nx] and dist <= radius * 0.7:
                        gmap.fire[ny, nx] = max(gmap.fire[ny, nx],
                                                 0.5 * falloff)


def add_explosion_smoke(gmap, fy, fx, radius, rng):
    """Deposit noisy smoke into ``gmap.smoke`` over a disc.

    Lifted from ``game.py:_add_explosion_smoke`` (lines 1998-2012). Random
    multiplier in [0.4, 1.0] per tile gives texture; in practice the
    inner tiles still saturate at 1.0 — flagged as a known issue in
    docs/architecture.md §6.4.

    ``rng`` is a :class:`numpy.random.Generator`, owned by the Simulation
    facade. Sampling through it (instead of process-global ``random``)
    keeps AI rollouts deterministic from the seed.
    """
    h, w = gmap.material.shape
    for ddy in range(-radius, radius + 1):
        for ddx in range(-radius, radius + 1):
            ny, nx = fy + ddy, fx + ddx
            if (0 <= ny < h and 0 <= nx < w
                    and not gmap.solid[ny, nx]):
                dist = math.sqrt(ddy ** 2 + ddx ** 2)
                if dist < radius:
                    base = 0.8 * (1 - dist / radius)
                    noise = float(rng.uniform(0.4, 1.0))
                    gmap.smoke[ny, nx] = min(
                        1.0, gmap.smoke[ny, nx] + base * noise)
