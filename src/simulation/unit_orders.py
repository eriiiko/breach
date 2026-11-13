"""The ORDER (#195 patch 1f; ``docs/rl_agent_rung1_note_1f_2026-11-12.md``,
design ``docs/rl_agent_design_2026-11-11.md`` 5.1, 5.4, 5.7, 9).

An order is game state on a unit, set through ONE verb by any seat (the bots'
team layer as the commander today; a scripted, network or human seat later):
WHAT (none / go to / hold / clear / fall back to), WHERE (a place on the
level's room graph: a room id, or ``ORDER_OPENING_BASE + opening id`` --
never a tile), UNTIL (an absolute tick, or until told otherwise) and an
optional priority TARGET (an enemy's unit id). It lives in four schema
columns (``unit_schema``'s ``order`` block: ``order_what`` / ``order_where`` /
``order_until`` / ``order_target``), hashed only while held.

THE ONE WRITER. ``Simulation.order_intent`` runs :func:`check` (refusing by
name, before the intent tap, so a recording never holds a refused call),
records itself and LATCHES the request on the sim (``Simulation.
_order_requests``, the ``drive_zombies`` precedent: no unit record moves
outside the tick); :func:`apply_requests` writes the columns at slot 1b';
:func:`expire` clears an order whose tick has come right after the tick
advance (slot 10); :func:`clear` is also called at death
(``damage.apply_packet``) and at conversion (``ai_zombie.
convert_marines_to_zombies``). Nothing else writes the columns.

INFORMATION ONLY until rung 4 (design 5.4): nothing in the game reads an
order for a decision. A reader is a design line first.

(The module is ``unit_orders``, not ``orders``: ``simulation.orders`` is
WEGO's ``Order`` vocabulary, a different thing.)
"""
from __future__ import annotations

from numbers import Integral

#: The kinds of order; the index is the code held in ``order_what``.
ORDER_WHAT = ("none", "go_to", "hold", "clear", "fall_back_to")
ORDER_NONE = 0
ORDER_GO_TO = 1
ORDER_HOLD = 2
ORDER_CLEAR = 3
ORDER_FALL_BACK_TO = 4

#: ``order_where`` codes: a room id ``r`` (0 <= r < ORDER_OPENING_BASE) or an
#: opening ``ORDER_OPENING_BASE + o`` -- one int tells a room from an opening
#: by range, independent of the level's room count.
ORDER_OPENING_BASE = 1 << 16
NO_PLACE = -1
#: ``order_until``: held until told otherwise.
NO_LIMIT = -1
#: ``order_target``: no priority target.
NO_TARGET = -1

#: The four columns, in the order a request carries them.
ORDER_FIELDS = ("order_what", "order_where", "order_until", "order_target")
#: The four defaults (a unit holding no order).
NO_ORDER = (ORDER_NONE, NO_PLACE, NO_LIMIT, NO_TARGET)


def graph_of(sim):
    """The sim's room graph when an order can name a place on it: the
    level's ``RoomGraph`` at the map's shape, else None (no graph, or a
    ``--res``-scaled map, whose graph is at base resolution)."""
    graph = getattr(getattr(sim, "level", None), "rooms", None)
    if graph is None:
        return None
    if tuple(graph.grid.shape) != tuple(sim.gmap.material.shape):
        return None
    return graph


def _int(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise TypeError(f"order_intent: {name} must be an integer, "
                        f"got {v!r}")
    return int(v)


def place_name(graph, where: int) -> str:
    """A readable name of an ``order_where`` code: "room 3", "opening 7" or
    "no place" (messages and tools)."""
    where = int(where)
    if where == NO_PLACE:
        return "no place"
    if where >= ORDER_OPENING_BASE:
        return f"opening {where - ORDER_OPENING_BASE}"
    return f"room {where}"


def check(sim, unit_id, what, where, ticks, target_id) -> tuple:
    """THE order door: the verb's five arguments -> ``(unit_id, what,
    where, ticks, target_id)`` as ints, or TypeError (a bool or a
    non-integer) / ValueError naming what is refused: a level without a room
    graph (or a map not at the graph's shape); an unknown, dead or zombie
    unit; ``what`` outside :data:`ORDER_WHAT`; a ``where`` naming no room
    and no opening; ``ticks < 0``; a target that is not a living unit of
    ANOTHER team; any field set with ``what = 0`` (none = clear). Pure: it
    does not ask who calls."""
    unit_id = _int("unit_id", unit_id)
    what = _int("what", what)
    where = _int("where", where)
    ticks = _int("ticks", ticks)
    target_id = _int("target_id", target_id)
    level = getattr(getattr(sim, "level", None), "name", "?")
    graph = getattr(getattr(sim, "level", None), "rooms", None)
    if graph is None:
        raise ValueError(f"order_intent: the order names a place on the "
                         f"room graph; level {level!r} has none")
    if graph_of(sim) is None:
        raise ValueError(f"order_intent: the order names a place on the "
                         f"room graph; level {level!r}'s graph "
                         f"{tuple(graph.grid.shape)} is not at the map's "
                         f"shape {tuple(sim.gmap.material.shape)} (a "
                         f"--res-scaled map has none)")
    u = sim.get_unit(unit_id)
    if u is None:
        raise ValueError(f"order_intent: unit {unit_id} does not exist")
    if not u.alive:
        raise ValueError(f"order_intent: unit {unit_id} is dead")
    if u.is_zombie:
        raise ValueError(f"order_intent: unit {unit_id} is a zombie (a "
                         f"zombie takes no order)")
    if not 0 <= what < len(ORDER_WHAT):
        raise ValueError(f"order_intent: what = {what} is not an order "
                         f"code (0..{len(ORDER_WHAT) - 1}: {ORDER_WHAT})")
    if ticks < 0:
        raise ValueError(f"order_intent: ticks = {ticks} (0 = until told "
                         f"otherwise, else the ticks the order holds)")
    if what == ORDER_NONE:
        set_ = [n for n, v, d in (("where", where, NO_PLACE),
                                  ("ticks", ticks, 0),
                                  ("target_id", target_id, NO_TARGET))
                if v != d]
        if set_:
            raise ValueError(f"order_intent: what = 0 (none) clears the "
                             f"order and takes no other field; {set_} "
                             f"set")
        return (unit_id, what, where, ticks, target_id)
    n_rooms, n_open = len(graph.rooms), len(graph.openings)
    if n_rooms >= ORDER_OPENING_BASE:
        raise ValueError(f"order_intent: level {level!r} has {n_rooms} "
                         f"rooms, at or past ORDER_OPENING_BASE "
                         f"({ORDER_OPENING_BASE})")
    is_room = 0 <= where < n_rooms
    is_open = 0 <= where - ORDER_OPENING_BASE < n_open
    if not (is_room or is_open):
        raise ValueError(f"order_intent: where = {where} names no place "
                         f"(a room id 0..{n_rooms - 1}, or "
                         f"ORDER_OPENING_BASE + an opening id 0.."
                         f"{n_open - 1}; level {level!r} has {n_rooms} "
                         f"rooms and {n_open} openings)")
    if target_id != NO_TARGET:
        t = sim.get_unit(target_id)
        if t is None:
            raise ValueError(f"order_intent: target {target_id} does not "
                             f"exist")
        if not t.alive:
            raise ValueError(f"order_intent: target {target_id} is dead")
        if int(t.team) == int(u.team):
            raise ValueError(f"order_intent: target {target_id} is a "
                             f"teammate of unit {unit_id} (a priority "
                             f"target is an enemy)")
    return (unit_id, what, where, ticks, target_id)


def clear(u) -> None:
    """The unit holds no order (the four defaults)."""
    u.order_what, u.order_where, u.order_until, u.order_target = NO_ORDER


def held(u) -> tuple:
    """The unit's order as ``(what, where, until, target)``."""
    return (u.order_what, u.order_where, u.order_until, u.order_target)


def apply_requests(sim) -> None:
    """Slot 1b': write this tick's latched orders (``sim._order_requests``,
    one per unit, the last call of the tick) onto their units, in unit-id
    order -- the four columns, or the four defaults for ``what = 0`` -- for
    a unit still alive and not a zombie, and drop every request (a dead
    unit's too: nothing stays latched). Dormant on an empty dict."""
    req = sim._order_requests
    if not req:
        return
    for uid in sorted(req):
        u = sim.get_unit(uid)
        if u is None or not u.alive or u.is_zombie:
            continue
        what, where, until, target = req[uid]
        if what == ORDER_NONE:
            clear(u)
        else:
            u.order_what = what
            u.order_where = where
            u.order_until = until
            u.order_target = target
    req.clear()


def expire(sim) -> None:
    """After the tick advance (slot 10): clear every order whose tick has
    come (``order_until <= sim.tick``), so between ticks a held order always
    has time left. An order of ``ticks = d`` issued at tick T holds through
    the d steps T .. T + d - 1. One integer compare per unit per tick."""
    tick = int(sim.tick)
    for u in sim.units:
        if (u.order_what != ORDER_NONE and u.order_until != NO_LIMIT
                and u.order_until <= tick):
            clear(u)
