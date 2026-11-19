"""BROOD PLACEMENT on a recipe ship (vocabulary Part G, arc #63 P5d): where
the larvae nest is DATA -- each department's ``[brood]`` row
(``levelgen_recipes.BroodRule``) and the archetype's ``[infestation]``
weights (``ship_grammar.Archetype.infestation``) -- drawn on the level seed's
OWN stream and applied to the laid-out ``gen_ship.ShipGraph`` before stage 2.
No move, no generator, no placement pass after ``embed``: this module only
writes the ``brood`` room tag (stage 2's ``gen_ship._build_broods`` turns it
into a ``swarm_brood`` zone entity) and, on an ``infested`` ship, rewrites the
hosting departments' corridor doors to the existing ``entity_door`` row (an
open door entity).

THE STREAM: ``SeedSequence([level seed, BROOD_STREAM])`` -- the grammar draws
from ``[seed, 0, g]`` and the layout from ``[seed, 1, g, a]``, so this draw
moves none of theirs, and a ``clean`` ship (or an archetype without the
table) is tile for tile the ship it was. Draw 0 picks the state; draw 1 + k
is department k's roll, k walking the archetype's departments that carry a
``[brood]`` row in archetype order WHATEVER the ship holds, so one
department's presence never shifts another's roll.

THE STATES (``ship_grammar.INFESTATION_STATES``): ``clean`` -- no broods;
``nest`` -- each department on the ship with a row hosts when its roll is
under its effective chance (``chance`` x the ``with`` multipliers of the
other departments present, capped at 1), and when every roll fails the one
with the highest effective chance hosts (ties: archetype order);
``infested`` -- every department on the ship with a row hosts, and every
``door`` edge between a hosting department's room and the PASSAGEWAY is
emitted open. A hosting department hosts in every instance; a wing room's
brood is tagged on the room AND its mirror twin (fairness).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import gen_ship as G
import levelgen_recipes as R
from ship_grammar import INFESTATION_STATES

#: The level seed's stream index for the brood draw (the grammar is 0, the
#: layout 1): ``SeedSequence([seed, BROOD_STREAM])``.
BROOD_STREAM = 2
#: The species every generated brood spawns (the one species today).
BROOD_SPECIES = "larva"
#: The skeleton role of the corridor an infested department's doors open onto.
CORRIDOR_ROLE = "passageway"
#: The edge kind an infested corridor door is rewritten FROM, and TO (the
#: existing open entity-door row: same opening, width, compartment rule).
CLOSED_DOOR_KIND, OPEN_DOOR_KIND = "door", "entity_door"


@dataclass(frozen=True)
class Infestation:
    """One ship's brood draw: the ``state`` and each department's roll
    (``rolls``: ``((department, u), ...)``, archetype order, u in [0, 1))."""
    state: str
    rolls: tuple = ()


def _brood_departments(arch) -> tuple:
    """The archetype's departments that carry a ``[brood]`` row, archetype
    order (the roll order)."""
    lib = arch.library
    return tuple(n for n in arch.departments()
                 if lib.departments[n].brood is not None)


def draw(arch, seed: int) -> Infestation:
    """The ship's state and rolls from ``SeedSequence([seed,
    BROOD_STREAM])``. An archetype without ``[infestation]`` is ``clean``
    and draws nothing."""
    if not arch.infestation:
        return Infestation(state=INFESTATION_STATES[0])
    rng = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([int(seed), BROOD_STREAM])))
    u0 = float(rng.random())
    names = _brood_departments(arch)
    us = rng.random(len(names)) if names else ()
    acc, state = 0.0, None
    for s, w in arch.infestation:
        acc += w
        if w > 0 and u0 < acc:
            state = s
            break
    if state is None:                     # u0 at the float top: the last state
        state = next(s for s, w in reversed(arch.infestation) if w > 0)
    return Infestation(state=state,
                       rolls=tuple((n, float(u)) for n, u in zip(names, us)))


def effective_chance(arch, name: str, present) -> float:
    """Department ``name``'s chance on a ship holding the departments
    ``present``: its ``chance`` times the ``with`` multiplier of every OTHER
    department present, capped at 1 (0 without a row)."""
    rule = arch.library.departments[name].brood
    if rule is None:
        return 0.0
    p = rule.chance
    there = set(present)
    for other, m in rule.with_:
        if other in there and other != name:
            p *= m
    return min(1.0, p)


def hosts(arch, inf: Infestation, present) -> tuple:
    """The departments that host a brood, archetype order, on a ship holding
    the departments ``present`` (quotient department names) in state
    ``inf.state``."""
    there = set(present)
    cands = [n for n in _brood_departments(arch) if n in there]
    if inf.state == "clean" or not cands:
        return ()
    if inf.state == "infested":
        return tuple(cands)
    rolls = dict(inf.rolls)
    chance = {n: effective_chance(arch, n, there) for n in cands}
    won = tuple(n for n in cands if rolls[n] < chance[n])
    if won:
        return won
    best = max(chance.values())
    if best <= 0.0:
        return ()
    return (next(n for n in cands if chance[n] == best),)


def _brood_room(q, room_ids: dict, graph, dept, instance: int) -> int:
    """The ShipGraph room id that hosts department instance ``(dept,
    instance)``'s brood: the largest eligible recipe room (floor area, ties
    to the lowest id), never a secret. ValueError when none is eligible."""
    eligible = set(R.brood_eligible(dept))
    best = None
    for r in q.rooms:
        if (r.dept != dept.name or r.instance != instance
                or r.recipe_id not in eligible or r.tags.get("secret")):
            continue
        rid = room_ids[r.id][0]
        rm = graph.rooms[rid]
        key = (-(rm.w * rm.h), rid)
        if best is None or key < best:
            best = key
    if best is None:
        raise ValueError(f"department {dept.name!r} instance {instance}: none "
                         f"of its rooms {sorted(eligible)} is on this ship")
    return best[1]


def apply(graph, q, room_ids: dict, arch, state: str, hosting: tuple):
    """The laid-out ``graph`` with its broods: a copy whose hosting rooms
    (and their mirror twins) carry the ``brood`` tag and, on an ``infested``
    ship, whose hosting departments' corridor doors are ``entity_door``
    rows. ``graph`` itself is returned untouched when nothing hosts."""
    if not hosting:
        return graph
    lib = arch.library
    tagged = {}
    for d in q.departments:
        if d.name not in hosting:
            continue
        dept = lib.departments[d.name]
        rid = _brood_room(q, room_ids, graph, dept, d.instance)
        tag = {"species": BROOD_SPECIES, "density": dept.brood.density,
               "max": dept.brood.max}
        tagged[rid] = tag
        twin = graph.rooms[rid].mirror
        if twin is not None:
            tagged[twin] = tag
    rooms = [replace(rm, tags={**rm.tags, "brood": dict(tagged[rm.id])})
             if rm.id in tagged else rm for rm in graph.rooms]
    edges = list(graph.edges)
    if state == "infested":
        host_rooms = {rm.id for rm in graph.rooms
                      if rm.tags.get("dept") in hosting}
        corridor = {rm.id for rm in graph.rooms
                    if rm.tags.get("role") == CORRIDOR_ROLE
                    and not rm.tags.get("dept")}
        for i, e in enumerate(edges):
            if e.kind != CLOSED_DOOR_KIND:
                continue
            ends = {e.a, e.b}
            if ends & host_rooms and ends & corridor:
                edges[i] = replace(e, kind=OPEN_DOOR_KIND)
    return replace(graph, rooms=rooms, edges=edges)


def opened_doors(before, after) -> list:
    """The edge indices whose kind :func:`apply` changed (the infested
    ship's opened corridor doors)."""
    return [i for i, (a, b) in enumerate(zip(before.edges, after.edges))
            if a.kind != b.kind]


__all__ = ["BROOD_STREAM", "BROOD_SPECIES", "CORRIDOR_ROLE", "Infestation",
           "draw", "effective_chance", "hosts", "apply", "opened_doors"]
