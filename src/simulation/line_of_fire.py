"""simulation/line_of_fire.py -- what stands in a round's way (#195 1g).

Two read-only questions the observation's type-1 entries ask
(``docs/rl_agent_obs_spec_v1_2026-11-12.md`` 1.7, 2.2 cols 25-26), judged
AS A ROUND FLIES, never as sight:

- :func:`solid_in_way`: is there something solid between two centre tiles --
  the sight routine's walk under ``physical=True`` (``vision.rays_clear``:
  ``has_los(physical=True)``'s question, a stop on ``solid`` only; a sight
  screen such as a grille does not stop a round) OR a live cover entity's
  ``blocks_segment`` (EVERY live cover: the march stops on any cover, a
  crate you can see over included, ``combat.py``'s cover stop);
- :func:`mate_in_way`: is a team-mate's body near the segment shooter ->
  target -- the layered bots' line geometry verbatim
  (``control_bots.BotTeam._teammate_on_line``), as a module function over
  centres: the projection ahead of the shooter and not past the target, the
  perpendicular distance within half a body plus ``clear_tiles``
  (``[bots] line_clear_tiles``). Integer geometry. The bots are NOT moved
  onto it in 1g (a one-line finding); ``tests/test_obs_spec_v1.py`` holds
  the two equal.

Both take ``(x, y)`` centre tiles (the bots' ``_centre`` convention) and
read the map only through the sight routine (``sim.gmap``, ``sim.cover``):
the observation hands them its ``ObsWorld`` facade as ``sim``.
"""
from __future__ import annotations

from simulation import vision


def solid_in_way(sim, a, b) -> bool:
    """1 = something solid between the centre tiles ``a`` and ``b`` (each
    ``(x, y)``) as a round flies: the physical walk is blocked, or a live
    cover entity crosses the segment."""
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    clear = vision.rays_clear(sim, [(ay, ax, by, bx)], physical=True)
    if not bool(clear[0]):
        return True
    for cov in (getattr(sim, "cover", ()) or ()):
        if cov.alive and cov.blocks_segment(ax, ay, bx, by):
            return True
    return False


def mate_in_way(a, b, mates, clear_tiles: int, fp: int) -> bool:
    """Is a team-mate's body near the segment ``a`` -> ``b`` (centre tiles,
    ``(x, y)``), anywhere along it? ``mates`` are the OTHER mates' centre
    tiles ``(x, y)`` (the shooter excluded by the caller), each a body of
    footprint ``fp``; ``clear_tiles`` the margin (``[bots]
    line_clear_tiles``). The bots' ``_teammate_on_line`` verbatim."""
    sx, sy = int(a[0]), int(a[1])
    tx, ty = int(b[0]), int(b[1])
    dx, dy = tx - sx, ty - sy
    dd = dx * dx + dy * dy
    if dd == 0:
        return False
    cr = int(fp) // 2 + int(clear_tiles)
    for mx, my in mates:
        vx, vy = int(mx) - sx, int(my) - sy
        along = vx * dx + vy * dy
        if along <= 0 or along > dd:
            continue
        cross = vx * dy - vy * dx
        if cross * cross <= cr * cr * dd:
            return True
    return False


__all__ = ["solid_in_way", "mate_in_way"]
