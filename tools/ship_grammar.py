r"""tools/ship_grammar.py -- STAGE 1 of family B: the GRAMMAR (#136;
docs/levelgen_archetype_recipe_2026-10-31.md rev 2 with its §7.1
amendments, docs/levelgen_layout_stage_2026-10-31.md §1, §2, §4, §7 and §13
A1-A7, docs/levelgen_cycle_types_and_recipes_2026-10-31.md §0.1, §1.5, §5.3).

An ARCHETYPE RECIPE (``levels/recipes/archetypes/<name>.toml``) is the build
order for a whole ship on an existing skeleton: the axiom departments, an
ordered list of (phase, department | cycle type, ``on``, count per size)
entries, the budgets (spawn exits, secrets), the promoted cells, difficulty
profiles with named operators and a validation profile. This module is

- THE ARCHETYPE DOOR (:func:`load_archetype`): the file into frozen
  dataclasses, refusing by file and field (archetype doc §1.1, §7.1), with
  the transitive-closure walk that refuses a department whose resolved rooms
  still wait for an unbuilt mechanism, and the closed-form feasibility per
  (size, profile);
- THE EXECUTOR (:func:`plan_quotient`, :func:`grammar_attempt`; archetype
  doc §2): a grammar attempt loop of :data:`G_MAX` attempts, attempt ``g``
  drawing every draw from ``SeedSequence([seed, 0, g])``; the axiom, then
  phases 1-4 (recipe doc §5.3), entries in file order, counts uniform,
  candidates sorted by a stable global id and drawn without replacement,
  each application checked against the phase's invariants, a rejected
  candidate logged and not counted, ``placed < lo`` a :class:`GrammarReject`;
- THE SEVEN MOVES (recipe doc §0.1) as small pure functions on a
  :class:`Draft`: :func:`subdivide`, :func:`edge_to_cycle`, :func:`leaf`,
  :func:`ring`, :func:`parallel`, :func:`duct`, :func:`second_leg` -- each
  with its precondition (:class:`PreconditionFailed`, naming the move) and
  provenance (rule id + application index);
- THE INVARIANTS (recipe doc §1.5, :data:`INVARIANTS`), each refusing by
  name (:class:`InvariantViolation`);
- THE QUOTIENT GRAPH (:class:`QuotientGraph`): the grammar's product and the
  layout stage's input -- one wing plus the axis band, no geometry;
- :func:`recipe_hash` (archetype doc §5.1).

The grammar never places a rectangle. Layout feasibility (layout doc §13 A5,
archetype doc §6 items 1-2) is ONE call site, ``ship_layout.
layout_feasibility``: asked with the whole quotient graph at the end of every
attempt (exact: a graph it accepts packs; a refusal is a ``GrammarReject``),
and by the door per (size, profile) with the archetype at its minima (the
demand test; a refusal names the size, the profile and what overflows), with
``ship_layout.maxima_warning`` kept in ``Archetype.warnings``.

Home: ``tools/`` beside ``gen_ship.py`` and ``levelgen_recipes.py`` (a
stage-1 input of the offline generator); the skeleton's graph side is
imported from ``tools/ship_layout.py``.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import levelgen_recipes as R  # noqa: E402
import ship_layout as L  # noqa: E402

ARCHETYPES_DIR = R.RECIPES_DIR / "archetypes"
G_MAX = 16                     # grammar attempts per seed (layout doc §13 A5)
AXIS, WING = "axis", "wing"    # a room's mirror class: self-twin / has a twin
PHASES = (1, 2, 3, 4)
ROWS = ("A", "B")
EXITS_RANGE = (1, 3)           # archetype doc §1.1: `spawn.exits` within 1-3
ZOMBIE_ROOMS = ("axis_section",)
PROFILE_OPS = ("cycles_shift", "secrets", "zombies_scale", "promote")
# [validation]: the closed registry of metric gates (archetype doc §1, L8).
# The gates themselves run after layout + embed (the layout patch); here the
# keys and their shapes. "sized" = {size = [lo, hi]} per allowed size.
VALIDATION_GATES = {"bot_walkable": "bool", "spawn_distance": "sized",
                    "cells": "range", "rooms": "sized"}
# The walk graph (recipe doc §1.1): kinds a unit passes; an airlock counts as
# walkable (in sequence). A `secret` room is reached through a secret-path
# kind (gen_ship.EDGE_KINDS[k].secret_path) instead. A `vented_door` is a
# door (a grille in each jamb, #134); a `grille` is NEVER a walk kind -- it
# passes air only, so a room reached only through grilles is unreachable.
WALK_KINDS = ("door", "open", "entity_door", "airlock", "vented_door")
EXT_DEPT_CYCLES = ("a department's own [[cycle]] rows (recipe doc §3.2): the "
                   "grammar does not apply them in v1 (no shipped department "
                   "has one)")


class ArchetypeError(R.RecipeError):
    """An archetype the door refuses; the message names the file and the
    field (and the size / profile where it is per size or profile)."""


class Rejected(Exception):
    """A candidate application refused: a move's precondition or an
    invariant. Logged in provenance; it does not consume the count.
    ``rule`` names the move or the invariant."""

    def __init__(self, rule: str, why: str):
        super().__init__(f"{rule}: {why}")
        self.rule = rule


class PreconditionFailed(Rejected):
    """A move's (or a cycle realisation's) precondition does not hold."""


class InvariantViolation(Rejected):
    """An application would break a recipe-doc §1.5 invariant;
    ``rule`` is the invariant's name (a key of :data:`INVARIANTS`)."""


class GrammarReject(Exception):
    """The grammar attempt is over (an entry placed fewer than its minimum,
    a mandatory step broke an invariant, the layout's feasibility refused);
    attempt ``g + 1`` draws afresh."""

    def __init__(self, rule: str, why: str):
        super().__init__(f"{rule}: {why}")
        self.rule = rule


class GrammarExhausted(G.ShipGenError):
    """No grammar attempt of the seed returned a graph within G_MAX."""


# ===========================================================================
# The quotient graph (the grammar's product; the layout stage's input)
# ===========================================================================

@dataclass(frozen=True)
class QSize:
    """A room's size ranges: ``axes`` ``("w", "h")`` (x, y extents) or
    ``("across", "along")`` (a corridor; the layout rotates), each an
    inclusive ``(lo, hi)`` -- or None where the LAYOUT derives it (the
    slab's, the lobby's height, a band section's, the passageway's and the
    hull strip's length, a ring room's long side)."""
    axes: tuple
    a: tuple | None
    b: tuple | None


@dataclass(frozen=True)
class QRoom:
    """One quotient room. ``id``: stable string id. ``role``: the skeleton
    role (``ship_layout.ROLES``) or a made room's role
    (``ship_layout.MADE_ROLES``), None for a recipe room. A department room
    names (``dept``, ``instance``, ``recipe_id``, ``index``): the department,
    its draw index, the recipe room id and the count index. ``side``:
    ``axis`` (a self-twin) or ``wing`` (reflected into a twin by the layout;
    a ring side room is a wing-class room INSIDE the band). ``kind``:
    gen_ship's ``Room.kind``. ``attach`` / ``parent``: what it is laid out
    against -- recipe ``side`` / ``end`` / ``behind`` (parent: the stub or
    the behind parent) / ``fore`` / ``aft`` (parent: the hub), or a made
    room's ``stub`` (a second stub; parent: the first), ``hull_strip``,
    ``ring_fore`` / ``ring_aft`` / ``ring_side`` (parent: the hub); None for
    an entry or a skeleton room. ``exit``: may carry a spawn exit. ``cell``:
    the pressure cell (main = 0). ``tags``: resolved gen_ship ``ROOM_TAGS``
    values, ready for a ``gen_ship.Room``. ``prov``: (rule id, application
    index)."""
    id: str
    side: str = WING
    kind: str = "room"
    shape: str = "room"
    width: str = "wide"
    size: QSize | None = None
    role: str | None = None
    dept: str | None = None
    instance: int | None = None
    recipe_id: str | None = None
    index: int = 0
    attach: str | None = None
    parent: str | None = None
    exit: bool = False
    cell: int = R.MAIN_CELL
    tags: dict = field(default_factory=dict)
    prov: tuple = ("", 0)

    @property
    def gid(self) -> tuple:
        """The stable global id (archetype doc §2): ``(side, department,
        instance, room, index)``, assigned when the room is made."""
        return (self.side, self.dept or "",
                -1 if self.instance is None else self.instance,
                self.recipe_id or self.role or self.id, self.index)


@dataclass(frozen=True)
class QEdge:
    """A DECLARED edge (a contact the layout realises): ``kind`` a row of
    ``gen_ship.EDGE_KINDS``; ``a`` / ``b`` ORDERED where it matters (an
    airlock's vestibule is carved on room ``a``). ``role``: ``skeleton``,
    ``entry`` (passageway to a stub), ``department``, ``exit``, ``cycle``,
    ``secret``. ``prov``: (rule id, application index)."""
    a: str
    b: str
    kind: str
    role: str = "cycle"
    prov: tuple = ("", 0)

    @property
    def pair(self) -> frozenset:
        return frozenset((self.a, self.b))


@dataclass(frozen=True)
class QDepartment:
    """One department INSTANCE. ``row`` / ``position``: a wing instance's
    row (``A`` / ``B``) and its index in the row from the slab end (layout
    doc §13 A7); None for the axis department. ``stubs``: its entry and, with
    a retreat loop, the second stub. ``order``: its gradient rank."""
    name: str
    instance: int
    placement: str
    cell: int
    toward: str
    entry: str
    stubs: tuple
    rooms: tuple = ()
    row: str | None = None
    position: int | None = None
    promoted: bool = False
    order: int = 0
    prov: tuple = ("", 0)


@dataclass(frozen=True)
class CycleApp:
    """One cycle-type application: the cycle ``type`` (its file), its
    ``move``, the anchor class ``on`` and the ``anchor`` rooms, the
    ``rooms`` and ``edges`` (``(a, b, kind)``) it made, the edges it
    ``removed`` or rewrote, the edge ``kinds`` it drew, ``params`` (e.g.
    the ring's full room count), its [[apply]] ``entry`` index and its
    application ``index``."""
    type: str
    move: str
    on: str
    anchor: tuple
    rooms: tuple
    edges: tuple
    removed: tuple
    kinds: tuple
    params: tuple
    entry: int
    index: int


@dataclass(frozen=True)
class QuotientGraph:
    """THE GRAMMAR'S PRODUCT AND THE LAYOUT STAGE'S INPUT (layout doc §1):
    one wing plus the axis band of a ship, as a graph -- no geometry.

    - ``rooms``: :class:`QRoom` (skeleton rooms by ``role``; department rooms
      with their department / instance / recipe id / index; made rooms by
      ``role``), each with ``side``, ``kind``, ``shape``, ``size`` ranges,
      ``width`` class, ``attach`` + ``parent``, ``exit``, ``cell`` and its
      resolved ``tags`` (gen_ship ``ROOM_TAGS`` values: ``dept``, ``role``,
      ``cell``, ``light``, ``fuel``, ``vent`` with the duct names already
      instance-qualified, ``heat_source``, ``hazard``, ``temp``,
      ``hull_door``, ``secret``, ``spawn_team`` on the slab, ``zombie`` on
      the zombie room);
    - ``edges``: every DECLARED edge (:class:`QEdge`), ``a`` before ``b``
      where the kind cares (airlock: the vestibule is on ``a`` -- the lobby
      for the hub's, the stub for a promoted department's); a room pair
      carries ONE edge, bar the airlock bypass's airlock + breakable;
    - ``departments``: every department instance (:class:`QDepartment`),
      wing ones with ``(row, position)`` -- departments in gradient order
      dealt to rows A, B alternately (layout doc §13 A7); there is no
      first-fit: a row that cannot pack is the layout's reject;
    - ``band``: the axis sections' ids fore to aft -- the main-cell sections,
      then the hub department's block (with a ring: ``ring_fore``, hub,
      ``ring_aft`` inside it);
    - ``cycles``: the cycle applications (:class:`CycleApp`);
    - ``exits``: the spawn slab's exit count; ``exit_edges``: its exits as
      edges, slab-passageway first, then the slab to the first department
      of row A's / row B's ``exit`` room (a flank strip's door is extra);
    - ``ducts``: ``((name, filter), ...)``: ``ShipGraph.ducts``'s content;
      a wing duct is the quotient's, the layout gives each wing its copy;
    - ``zombie_room``: the aft-most main section; ``zombies``: the profile's
      scaled count of ``zombies_base`` (archetype doc §4);
    - ``secrets``: the annex rooms; ``placed``: ``((entry index, count),
      ...)`` per [[apply]] entry; ``log``: provenance, every candidate tried
      ``(phase, rule, candidate, outcome)``;
    - the inputs (``archetype``, ``size``, ``profile``, ``seed``,
      ``max_per_team``, ``zombies_base``), ``recipe_hash`` and the attempt
      ``g``.

    The mirror holds by construction: the grammar only ever builds this
    half. :meth:`digest` is a sha256 over a canonical serialisation: the
    same inputs give the same digest."""
    archetype: str
    skeleton: str
    size: str
    profile: str
    seed: int
    g: int
    max_per_team: int
    zombies_base: int
    zombies: int
    recipe_hash: str
    rooms: tuple
    edges: tuple
    departments: tuple
    band: tuple
    cycles: tuple
    exits: int
    exit_edges: tuple
    ducts: tuple
    zombie_room: str
    secrets: tuple
    placed: tuple
    log: tuple

    def room(self, rid: str) -> QRoom:
        for r in self.rooms:
            if r.id == rid:
                return r
        raise KeyError(rid)

    def cells(self) -> tuple:
        return tuple(sorted({r.cell for r in self.rooms}))

    def digest(self) -> str:
        def js(o):
            if isinstance(o, (QRoom, QEdge, QDepartment, CycleApp, QSize)):
                d = {k: getattr(o, k) for k in o.__dataclass_fields__}
                return {k: js(v) for k, v in d.items()}
            if isinstance(o, dict):
                return {str(k): js(v) for k, v in sorted(o.items(),
                                                         key=lambda kv:
                                                         str(kv[0]))}
            if isinstance(o, (list, tuple)):
                return [js(x) for x in o]
            return o
        body = {k: js(getattr(self, k)) for k in self.__dataclass_fields__}
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


# ===========================================================================
# The working graph and the seven moves
# ===========================================================================

class Draft:
    """The grammar's working graph. A move never mutates its input: it
    returns a new Draft (:meth:`copy` + its change), so a rejected candidate
    leaves the graph as it was."""

    def __init__(self):
        self.rooms: dict = {}          # id -> QRoom, in creation order
        self.edges: list = []          # QEdge
        self.ducts: dict = {}          # name -> filter
        self.depts: list = []          # QDepartment
        self.cycles: list = []         # CycleApp
        self.band: tuple = ()
        self.n_app = 0

    @classmethod
    def of(cls, q: "QuotientGraph") -> "Draft":
        """A returned graph as a Draft again (to re-check its invariants)."""
        d = cls()
        d.rooms = {r.id: r for r in q.rooms}
        d.edges = list(q.edges)
        d.ducts = dict(q.ducts)
        d.depts = list(q.departments)
        d.cycles = list(q.cycles)
        d.band = q.band
        d.n_app = max(x.prov[1] for x in q.rooms + q.edges)
        return d

    def copy(self) -> "Draft":
        out = Draft()
        out.rooms = dict(self.rooms)
        out.edges = list(self.edges)
        out.ducts = dict(self.ducts)
        out.depts = list(self.depts)
        out.cycles = list(self.cycles)
        out.band = self.band
        out.n_app = self.n_app
        return out

    def next_app(self) -> int:
        self.n_app += 1
        return self.n_app

    def between(self, a: str, b: str) -> list:
        return [e for e in self.edges if e.pair == frozenset((a, b))]

    def edges_of(self, rid: str) -> list:
        return [e for e in self.edges if rid in (e.a, e.b)]

    def children(self, rid: str) -> list:
        return [r for r in self.rooms.values() if r.parent == rid]


def _need(move: str, d: Draft, *ids) -> None:
    for rid in ids:
        if rid not in d.rooms:
            raise PreconditionFailed(move, f"room {rid!r} is not in the graph")


def _new(move: str, d: Draft, *rooms) -> None:
    for r in rooms:
        if r.id in d.rooms:
            raise PreconditionFailed(move, f"room {r.id!r} already exists")


def _kind(move: str, kind: str) -> None:
    if kind not in G.EDGE_KINDS:
        raise PreconditionFailed(move, f"edge kind {kind!r} is not a row of "
                                       f"gen_ship.EDGE_KINDS")


def _one_edge(move: str, d: Draft, a: str, b: str) -> QEdge:
    es = d.between(a, b)
    if len(es) != 1:
        raise PreconditionFailed(move, f"rooms {a!r} and {b!r} carry "
                                       f"{len(es)} edges, not one")
    return es[0]


def join(d: Draft, a: str, b: str, kind: str, prov: tuple,
         role: str) -> Draft:
    """Declare ONE edge ``a``-``b`` between existing rooms -- not a move:
    the primitive the skeleton, the departments, the exits, the secrets and
    every move declare their edges with."""
    _need("join", d, a, b)
    _kind("join", kind)
    if a == b:
        raise PreconditionFailed("join", f"joins room {a!r} to itself")
    out = d.copy()
    out.edges.append(QEdge(a, b, kind, role, prov))
    return out


def _add(d: Draft, *rooms) -> Draft:
    out = d.copy()
    for r in rooms:
        out.rooms[r.id] = r
    return out


def subdivide(d: Draft, a: str, b: str, c: QRoom, kinds: tuple,
              prov: tuple) -> Draft:
    """``subdivide`` (L4 move 1): A-B becomes A-C-B, C new. Precondition:
    exactly one edge A-B. ``kinds``: (A-C, C-B)."""
    _need("subdivide", d, a, b)
    _new("subdivide", d, c)
    old = _one_edge("subdivide", d, a, b)
    out = _add(d, c)
    out.edges.remove(old)
    out = join(out, a, c.id, kinds[0], prov, "cycle")
    return join(out, c.id, b, kinds[1], prov, "cycle")


def edge_to_cycle(d: Draft, a: str, b: str, c: QRoom, dd: QRoom,
                  kinds: tuple, prov: tuple) -> Draft:
    """``edge_to_cycle`` (L4 move 2): A-B becomes A-C-B plus A-D-B, C and D
    new; A-B is REMOVED (so no triangle). Precondition: exactly one edge
    A-B. ``kinds``: (A-C, C-B, A-D, D-B)."""
    _need("edge_to_cycle", d, a, b)
    _new("edge_to_cycle", d, c, dd)
    old = _one_edge("edge_to_cycle", d, a, b)
    out = _add(d, c, dd)
    out.edges.remove(old)
    for (x, y), k in zip(((a, c.id), (c.id, b), (a, dd.id), (dd.id, b)),
                         kinds):
        out = join(out, x, y, k, prov, "cycle")
    return out


def leaf(d: Draft, a: str, n: QRoom, kind: str, prov: tuple,
         role: str = "cycle") -> Draft:
    """``leaf`` (L4 move 3): adds A-N, N new. Precondition: A exists."""
    _need("leaf", d, a)
    _new("leaf", d, n)
    return join(_add(d, n), a, n.id, kind, prov, role)


def ring(d: Draft, hub: str, fore: QRoom, aft: QRoom, sides: tuple,
         k_ring: str, k_hub: str, prov: tuple) -> Draft:
    """``ring`` on the QUOTIENT (recipe doc §0.1, layout doc §4.4): an EVEN
    ring of ``n = 2 + 2·len(sides)`` rooms around ``hub``. Clockwise from
    fore: r_1 = ``fore`` (axis), r_2..r_(n/2) the right side (the twins of
    ``sides``, made by the layout's reflection), r_(n/2+1) = ``aft`` (axis),
    r_(n/2+2)..r_n = ``sides`` (the left side, given AFT TO FORE). Ring
    neighbours are joined by ``k_ring`` (one kind per application), the hub
    by ``k_hub`` to the ODD-numbered rooms only, so H-r_i-r_(i+1) is never a
    triangle. Only the quotient's rooms and edges are made."""
    move = "ring"
    _need(move, d, hub)
    if not sides:
        raise PreconditionFailed(move, "a ring has at least one side room "
                                       "(n >= 4)")
    _new(move, d, fore, aft, *sides)
    n = 2 + 2 * len(sides)
    out = _add(d, fore, aft, *sides)
    chain = [aft.id] + [s.id for s in sides] + [fore.id]
    for x, y in zip(chain, chain[1:]):
        out = join(out, x, y, k_ring, prov, "cycle")
    numbered = [(1, fore.id), (n // 2 + 1, aft.id)] + \
        [(n // 2 + 2 + i, s.id) for i, s in enumerate(sides)]
    for j, rid in numbered:
        if j % 2 == 1:
            out = join(out, hub, rid, k_hub, prov, "cycle")
    return out


def parallel(d: Draft, a: str, b: str, kind: str, prov: tuple) -> Draft:
    """``parallel``: a second opening of ANOTHER kind on the shared wall of
    A and B (a multi-edge, no new room). Precondition: A-B carries exactly
    one edge, of a different kind."""
    _need("parallel", d, a, b)
    old = _one_edge("parallel", d, a, b)
    if old.kind == kind:
        raise PreconditionFailed("parallel", f"rooms {a!r} and {b!r} already "
                                             f"carry a {kind!r}")
    return join(d, a, b, kind, prov, "cycle")


def duct(d: Draft, a: str, b: str, name: str, filt: str, roles: tuple,
         prov: tuple) -> Draft:
    """``duct`` (the L4 vent exception): rooms A and B, of ONE cell, each
    gain a vent on the new duct ``name`` with ``roles`` (no wall edge).
    Preconditions: two distinct rooms, one cell, a new duct name."""
    move = "duct"
    _need(move, d, a, b)
    if a == b:
        raise PreconditionFailed(move, f"joins room {a!r} to itself")
    ra, rb = d.rooms[a], d.rooms[b]
    if ra.cell != rb.cell:
        raise PreconditionFailed(move, f"rooms {a!r} (cell {ra.cell}) and "
                                       f"{b!r} (cell {rb.cell}): a duct never "
                                       f"crosses a cell")
    if name in d.ducts:
        raise PreconditionFailed(move, f"duct {name!r} already exists")
    out = d.copy()
    out.ducts[name] = filt
    for r in (ra, rb):
        out.rooms[r.id] = replace(r, tags=_with_vent(r.tags, name, roles))
    return out


def second_leg(d: Draft, a: str, c: str, b: str, dd: QRoom, kinds: tuple,
               prov: tuple) -> Draft:
    """``second_leg`` (the seventh move, archetype doc §3): precondition, a
    path A-C-B exists; effect, add A-D-B with D ONE new room; the existing
    leg is not redrawn. ``kinds``: (A-D, D-B)."""
    move = "second_leg"
    _need(move, d, a, c, b)
    _new(move, d, dd)
    if not d.between(a, c) or not d.between(c, b):
        raise PreconditionFailed(move, f"no path {a!r}-{c!r}-{b!r}")
    out = _add(d, dd)
    out = join(out, a, dd.id, kinds[0], prov, "cycle")
    return join(out, dd.id, b, kinds[1], prov, "cycle")


MOVES = {"subdivide": subdivide, "edge_to_cycle": edge_to_cycle,
         "leaf": leaf, "ring": ring, "parallel": parallel, "duct": duct,
         "second_leg": second_leg}


def _with_vent(tags: dict, name: str, roles) -> dict:
    """``tags`` with a vent on duct ``name`` added (the list form when the
    room is then on several ducts)."""
    out = dict(tags)
    new = {"duct": name, "roles": list(roles)}
    if "vent" not in out:
        out["vent"] = new
    else:
        old = out["vent"]
        out["vent"] = (list(old) if isinstance(old, list) else [old]) + [new]
    return out


def _vents(r: QRoom) -> tuple:
    return G.vent_specs(r.tags["vent"]) if "vent" in r.tags else ()


# ===========================================================================
# The invariants (recipe doc §1.5), each by name
# ===========================================================================

def inv_one_edge_per_pair(d: Draft) -> None:
    """A room pair carries ONE declared edge, bar the airlock bypass (one
    airlock + one breakable on the shared wall; gen_ship.BYPASS_KINDS)."""
    pairs: dict = {}
    for e in d.edges:
        pairs.setdefault(e.pair, []).append(e.kind)
    for p, ks in pairs.items():
        if len(ks) > 1 and sorted(ks) != sorted(G.BYPASS_KINDS):
            raise InvariantViolation(
                "one_edge_per_pair", f"rooms {sorted(p)} carry {ks}: a pair "
                                     f"carries one edge bar the airlock "
                                     f"bypass")


def inv_no_triangle(d: Draft) -> None:
    """No 3-cycle of DECLARED edges, any kinds (recipe doc §1.5) -- bar a
    T-junction the SKELETON draws (every pair a skeleton contact;
    ``ship_layout``'s docstring)."""
    adj: dict = {}
    skel = set()
    for e in d.edges:
        adj.setdefault(e.a, set()).add(e.b)
        adj.setdefault(e.b, set()).add(e.a)
        if e.role == "skeleton":
            skel.add(e.pair)
    for a in sorted(adj):
        for b in sorted(x for x in adj[a] if x > a):
            for c in sorted(x for x in adj[a] & adj[b] if x > b):
                pairs = (frozenset((a, b)), frozenset((b, c)),
                         frozenset((a, c)))
                if all(p in skel for p in pairs):
                    continue
                raise InvariantViolation(
                    "no_triangle", f"rooms {[a, b, c]} form a 3-cycle of "
                                   f"declared edges (a rectangular layout "
                                   f"needs the contact graph triangle-free)")


def inv_cell_crossing(d: Draft) -> None:
    """An edge whose rooms lie in different cells is an airlock, a bulkhead
    or a breakable wall (L5 as amended; gen_ship.CELL_CROSSING_KINDS)."""
    for e in d.edges:
        ca, cb = d.rooms[e.a].cell, d.rooms[e.b].cell
        if ca != cb and e.kind not in G.CELL_CROSSING_KINDS:
            raise InvariantViolation(
                "cell_crossing", f"edge {e.a}-{e.b} ({e.kind!r}) crosses "
                                 f"cells {ca} -> {cb}: only "
                                 f"{list(G.CELL_CROSSING_KINDS)} may (L5)")


def inv_duct_in_cell(d: Draft) -> None:
    """A duct's rooms lie in ONE cell (vocabulary §0.5)."""
    cells: dict = {}
    for r in d.rooms.values():
        for duct_name, _roles, _sh in _vents(r):
            cells.setdefault(duct_name, {}).setdefault(r.cell, []).append(r.id)
    for name, by_cell in sorted(cells.items()):
        if len(by_cell) > 1:
            raise InvariantViolation(
                "duct_in_cell", f"duct {name!r} spans cells "
                                f"{sorted(by_cell)} ({by_cell}): a duct never "
                                f"crosses a cell")


def inv_air_volume(d: Draft) -> None:
    """THE AIR-VOLUME VENT RULE (vocabulary §0.6; it replaced the sealed-room
    rule): every air volume a duct has a vent in carries a supply and a
    return on that duct -- ``gen_ship.vent_air_rule``, the one statement."""
    bad = G.vent_air_rule({rid: _vents(r) for rid, r in d.rooms.items()},
                          [(e.a, e.b, e.kind) for e in d.edges])
    if bad:
        raise InvariantViolation("air_volume", str(bad[0]))


def inv_reachable(d: Draft) -> None:
    """Every room is reachable in the walk graph from the slab (an airlock
    walkable; a ``secret`` room through a secret-path edge from a reached
    room). The mirror makes it "from both spawns"."""
    start = [r.id for r in d.rooms.values() if r.role == L.SLAB]
    if not start:
        raise InvariantViolation("reachable", "the graph has no slab")
    adj: dict = {}
    for e in d.edges:
        if e.kind in WALK_KINDS:
            adj.setdefault(e.a, []).append(e.b)
            adj.setdefault(e.b, []).append(e.a)
    seen, todo = set(start), list(start)
    while todo:
        for nb in adj.get(todo.pop(), ()):
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    lost = []
    for r in d.rooms.values():
        if r.id in seen:
            continue
        if r.tags.get("secret") and any(
                G.EDGE_KINDS[e.kind].secret_path
                and (e.b if e.a == r.id else e.a) in seen
                for e in d.edges_of(r.id)):
            continue
        lost.append(r.id)
    if lost:
        raise InvariantViolation(
            "reachable", f"rooms {lost} are not reachable from the slab in "
                         f"the walk graph")


# name -> (check, the first phase it runs in; recipe doc §5.3 as the
# archetype doc §2 reads it: the contact invariants on every application,
# reachability from phase 3 on, the air invariants in phase 4)
INVARIANTS = {
    "one_edge_per_pair": (inv_one_edge_per_pair, 1),
    "no_triangle": (inv_no_triangle, 1),
    "cell_crossing": (inv_cell_crossing, 1),
    "reachable": (inv_reachable, 3),
    "duct_in_cell": (inv_duct_in_cell, 4),
    "air_volume": (inv_air_volume, 1),
}


def check_invariants(d: Draft, phase: int = 4) -> None:
    """Every invariant that runs in ``phase``; the first violated one
    raises :class:`InvariantViolation` naming it."""
    for _name, (fn, first) in INVARIANTS.items():
        if phase >= first:
            fn(d)


# ===========================================================================
# The archetype door
# ===========================================================================

@dataclass(frozen=True)
class AxiomEntry:
    department: str
    placement: str


@dataclass(frozen=True)
class ApplyEntry:
    """One [[apply]] row: its file ``index``, ``phase``, exactly one of
    ``department`` / ``cycle``, ``on`` for a cycle, the file's ``count`` per
    size (``{size: (lo, hi)}``, a QUOTIENT count)."""
    index: int
    phase: int
    department: str | None
    cycle: str | None
    on: str | None
    count: dict

    @property
    def rule(self) -> str:
        return f"apply[{self.index}]:{self.department or self.cycle}"


@dataclass(frozen=True)
class Profile:
    """A difficulty profile; each key names its operator (archetype doc
    §1.1): ``cycles_shift`` added to every phase-2 / phase-4 cycle range,
    ``secrets`` the ABSOLUTE budget, ``zombies_scale`` on the generator's
    zombies argument, ``promote`` the ABSOLUTE promoted set. None = the
    file's value."""
    name: str
    cycles_shift: int = 0
    secrets: int | None = None
    zombies_scale: float = 1.0
    promote: tuple | None = None


@dataclass(frozen=True)
class Archetype:
    """An archetype recipe through its door. ``ranges``: ``{(size,
    profile): {entry index: (lo, hi)}}``, every [[apply]] entry's EFFECTIVE
    count (profile-shifted, capped at the anchor class's realisable maximum);
    ``sections``: ``{size: (lo, hi)}`` the band's section count (the file's
    range intersected with the preset's, at least the hub department's block
    + 1); ``library``: the recipe library it was checked against."""
    name: str
    skeleton: str
    sizes: tuple
    passageway: tuple
    lobby_x: tuple
    band_sections: tuple
    exits: tuple
    zombies: str
    secret_budget: int
    secret_size: tuple
    allow_cross_cell_leak: bool
    promote: tuple
    axiom: tuple
    applies: tuple
    validation: dict
    profiles: dict
    ranges: dict
    sections: dict
    path: str
    library: R.RecipeLibrary = field(compare=False, repr=False, default=None)
    # {(size, profile): why}: the layout's arithmetic at the recipe's MAXIMA
    # does not fit that size -- the grammar will reroll draws near the top
    # of its ranges there (archetype doc §6 item 2: a warning, not a refusal)
    warnings: dict = field(compare=False, repr=False, default_factory=dict)

    def promoted(self, profile: str) -> tuple:
        p = self.profiles[profile].promote
        return self.promote if p is None else p

    def secrets(self, profile: str) -> int:
        p = self.profiles[profile].secrets
        return self.secret_budget if p is None else p

    def scaled_zombies(self, profile: str, base: int) -> int:
        """Archetype doc §4: round half up, at least 1 when ``base`` > 0."""
        if base <= 0:
            return 0
        return max(1, math.floor(base * self.profiles[profile].zombies_scale
                                 + 0.5))

    def departments(self) -> tuple:
        """Every department the file names, in file order (axiom, then the
        [[apply]] entries)."""
        out = [a.department for a in self.axiom]
        out += [e.department for e in self.applies if e.department]
        return tuple(dict.fromkeys(out))

    def cells(self, profile: str) -> dict:
        """Department name -> cell (recipe doc §3.4): ``main`` = 0, each
        ``own`` department (by its file) the next int in file order, then
        each promoted one in file order. Fixed per profile, whatever the
        draw."""
        lib = self.library
        promoted = set(self.promoted(profile))
        out, nxt = {}, R.MAIN_CELL + 1
        names = self.departments()
        for n in names:
            if lib.departments[n].cell == "own":
                out[n], nxt = nxt, nxt + 1
        for n in names:
            if n in promoted:
                out[n], nxt = nxt, nxt + 1
        for n in names:
            out.setdefault(n, R.MAIN_CELL)
        return out

    @property
    def hub_department(self) -> str:
        return L.SKELETONS[self.skeleton].hub_department


def _err(path, where: str, why: str) -> ArchetypeError:
    return ArchetypeError(f"{path}: {where}: {why}")


_FILE_KEYS = ("archetype", "cells", "axiom", "apply", "validation",
              "profiles")
_ARCH_KEYS = ("name", "skeleton", "sizes", "passageway", "lobby", "band",
              "spawn", "zombies", "secrets", "allow_cross_cell_leak")
_APPLY_KEYS = ("phase", "department", "cycle", "on", "count")


def _sub_range(path, where: str, table, key: str, lo_min: int,
               hi_max: int | None = None) -> tuple:
    R._keys(path, where, table, (key,), (key,))
    return R._range(path, f"{where}.{key}", table[key], lo_min, hi_max)


def _int(path, where: str, v, lo: int | None = None) -> int:
    if not R._is_int(v) or (lo is not None and v < lo):
        raise _err(path, where, f"{v!r} is not an integer"
                                + (f" >= {lo}" if lo is not None else ""))
    return v


def load_archetype(name_or_path, *, library: R.RecipeLibrary | None = None
                   ) -> Archetype:
    """An archetype recipe (a name under ``levels/recipes/archetypes`` or a
    path) as an :class:`Archetype`, or :class:`ArchetypeError` /
    ``RecipeError`` naming the file and the field. ``library``: the recipe
    library (default: the one under the file's recipe root)."""
    p = Path(name_or_path)
    path = p if p.suffix == ".toml" else ARCHETYPES_DIR / f"{p}.toml"
    if library is None:
        library = R.load_recipes(path.resolve().parent.parent)
    raw = R._read(path)
    R._keys(path, "file", raw, _FILE_KEYS, ("archetype", "axiom", "profiles"))
    a = raw["archetype"]
    R._keys(path, "[archetype]", a, _ARCH_KEYS, _ARCH_KEYS)
    name = R._ident(path, "archetype.name", a["name"])
    if name != path.stem:
        raise _err(path, "archetype.name", f"{name!r} differs from the file "
                                           f"name {path.stem!r}")
    skel_name = R._one_of(path, "archetype.skeleton", a["skeleton"],
                          tuple(L.SKELETONS))
    skel = L.SKELETONS[skel_name]
    sizes = R._str_list(path, "archetype.sizes", a["sizes"])
    if not sizes or len(set(sizes)) != len(sizes):
        raise _err(path, "archetype.sizes", f"{list(sizes)} is not a "
                                            f"non-empty list of distinct "
                                            f"sizes")
    for i, s in enumerate(sizes):
        R._one_of(path, f"archetype.sizes[{i}]", s, tuple(G.SIZES))
    passageway = _sub_range(path, "archetype.passageway", a["passageway"],
                            "across", R.MIN_WIDE)
    lobby_x = _sub_range(path, "archetype.lobby", a["lobby"], "x",
                         L.LOBBY_X_MIN)
    band = _sub_range(path, "archetype.band", a["band"], "sections", 1)
    exits = _sub_range(path, "archetype.spawn", a["spawn"], "exits",
                       EXITS_RANGE[0], EXITS_RANGE[1])
    zombies = R._one_of(path, "archetype.zombies", a["zombies"], ZOMBIE_ROOMS)
    sec = a["secrets"]
    R._keys(path, "archetype.secrets", sec, ("budget", "size"),
            ("budget", "size"))
    budget = _int(path, "archetype.secrets.budget", sec["budget"], 0)
    secret_size = R._range(path, "archetype.secrets.size", sec["size"],
                           R.MIN_WIDE)
    leak = R._bool(path, "archetype.allow_cross_cell_leak",
                   a["allow_cross_cell_leak"])
    if leak:
        raise _err(path, "archetype.allow_cross_cell_leak",
                   "true is refused until the damper entity (#49) exists "
                   "(L5's leak-path clause)")
    depts = library.departments

    # ---- the axiom ----
    axiom = []
    for i, t in enumerate(raw["axiom"]):
        w = f"axiom[{i}]"
        R._keys(path, w, t, ("department", "placement"),
                ("department", "placement"))
        dn = R._str(path, f"{w}.department", t["department"])
        if dn not in depts:
            raise _err(path, f"{w}.department", f"{dn!r} is no department "
                                                f"in the library "
                                                f"({sorted(depts)})")
        pl = R._one_of(path, f"{w}.placement", t["placement"], R.PLACEMENTS)
        if pl != depts[dn].placement:
            raise _err(path, f"{w}.placement", f"{pl!r} but department "
                                               f"{dn!r} is placement "
                                               f"{depts[dn].placement!r}")
        if any(x.department == dn for x in axiom):
            raise _err(path, f"{w}.department", f"{dn!r} is in the axiom "
                                                f"twice")
        axiom.append(AxiomEntry(dn, pl))
    hub = skel.hub_department
    if not any(x.department == hub and x.placement == "axis" for x in axiom):
        raise _err(path, "axiom", f"skeleton {skel_name!r} requires the axis "
                                  f"department {hub!r} in the axiom (the "
                                  f"lobbies' airlocks land on its hub)")
    hub_dept = depts[hub]
    for x in axiom:
        if x.placement == "axis" and x.department != hub:
            raise _err(path, "axiom", f"axis department {x.department!r}: "
                                      f"skeleton {skel_name!r} realises one "
                                      f"axis department, {hub!r}")

    # ---- [[apply]] ----
    applies = []
    for i, t in enumerate(raw.get("apply", [])):
        applies.append(_apply_entry(path, i, t, sizes, library, skel,
                                    hub_dept))

    # ---- [cells] and profiles ----
    named = {x.department for x in axiom} | \
        {e.department for e in applies if e.department}
    cells = raw.get("cells", {})
    R._keys(path, "[cells]", cells, ("promote",))
    promote = _promote(path, "cells.promote", cells.get("promote", []),
                       library, named)
    validation = _validation(path, raw.get("validation", {}), sizes)
    profiles = {}
    if not isinstance(raw["profiles"], dict) or not raw["profiles"]:
        raise _err(path, "[profiles]", "declare at least one profile (an "
                                       "empty [profiles.standard] is the "
                                       "file as written)")
    for pn, t in raw["profiles"].items():
        w = f"profiles.{pn}"
        R._ident(path, w, pn)
        R._keys(path, w, t, PROFILE_OPS)
        zs = t.get("zombies_scale", 1.0)
        if not (R._is_num(zs) and zs > 0):
            raise _err(path, f"{w}.zombies_scale", f"{zs!r} is not a number "
                                                   f"> 0")
        profiles[pn] = Profile(
            name=pn,
            cycles_shift=_int(path, f"{w}.cycles_shift",
                              t.get("cycles_shift", 0)),
            secrets=(_int(path, f"{w}.secrets", t["secrets"], 0)
                     if "secrets" in t else None),
            zombies_scale=float(zs),
            promote=(_promote(path, f"{w}.promote", t["promote"], library,
                              named) if "promote" in t else None))

    arch = Archetype(
        name=name, skeleton=skel_name, sizes=tuple(sizes),
        passageway=passageway, lobby_x=lobby_x, band_sections=band,
        exits=exits, zombies=zombies, secret_budget=budget,
        secret_size=secret_size, allow_cross_cell_leak=leak, promote=promote,
        axiom=tuple(axiom), applies=tuple(applies), validation=validation,
        profiles=profiles, ranges={}, sections={}, path=str(path),
        library=library)
    _closure(arch)
    for size in sizes:
        arch.sections[size] = _sections(arch, size)
        wing_min = sum(1 for x in axiom if x.placement == "wing") + sum(
            e.count[size][0] for e in applies
            if e.department and depts[e.department].placement == "wing")
        if wing_min < 2:
            raise _err(path, f"size {size!r}",
                       f"{wing_min} wing department(s) at minimum count: the "
                       f"skeleton's two rows each need one (layout doc §6)")
        for pn in profiles:
            arch.ranges[(size, pn)] = _ranges(arch, size, pn)
            why = L.layout_feasibility(size, archetype=arch, profile=pn)
            if why:
                raise _err(path, f"size {size!r}, profile {pn!r}",
                           f"the layout cannot pack it: {why}")
            warn = L.maxima_warning(size, arch, pn)
            if warn:
                arch.warnings[(size, pn)] = warn
    return arch


def _apply_entry(path, i: int, t, sizes, library, skel, hub_dept
                 ) -> ApplyEntry:
    w = f"apply[{i}]"
    R._keys(path, w, t, _APPLY_KEYS, ("phase", "count"))
    phase = t["phase"]
    if not (R._is_int(phase) and phase in PHASES):
        raise _err(path, f"{w}.phase", f"{phase!r} is not an L6 phase in "
                                       f"{list(PHASES)}")
    which = [k for k in ("department", "cycle") if k in t]
    if len(which) != 1:
        raise _err(path, w, f"sets {which or 'none'} of department / cycle: "
                            f"exactly one")
    count = t["count"]
    R._keys(path, f"{w}.count", count, sizes, sizes)
    count = {s: R._range(path, f"{w}.count.{s}", count[s], 0)
             for s in sizes}
    if phase == 3:
        raise _err(path, f"{w}.phase", "phase 3 (terminalise) takes no "
                                       "entries: fixtures are department "
                                       "data and the secrets are the "
                                       "archetype's budget")
    if "department" in t:
        dn = R._str(path, f"{w}.department", t["department"])
        if "on" in t:
            raise _err(path, f"{w}.on", "a department entry has no anchor")
        if phase != 1:
            raise _err(path, f"{w}.phase", f"a department grows in phase 1, "
                                           f"not {phase}")
        if dn not in library.departments:
            raise _err(path, f"{w}.department", f"{dn!r} is no department "
                                                f"in the library "
                                                f"({sorted(library.departments)})")
        if library.departments[dn].placement != "wing":
            raise _err(path, f"{w}.department",
                       f"{dn!r} is an axis department: the skeleton's one "
                       f"axis department is its hub, {skel.hub_department!r}, "
                       f"in the axiom")
        return ApplyEntry(i, phase, dn, None, None, count)
    cn = R._str(path, f"{w}.cycle", t["cycle"])
    if cn not in library.cycles:
        raise _err(path, f"{w}.cycle", f"{cn!r} is no cycle type in the "
                                       f"library ({sorted(library.cycles)})")
    cyc = library.cycles[cn]
    if phase == 1:
        raise _err(path, f"{w}.phase", "phase 1 takes departments; a cycle "
                                       "inside a department is its own "
                                       "[[cycle]] row")
    if phase not in cyc.phase:
        raise _err(path, f"{w}.phase", f"cycle type {cn!r} may be applied in "
                                       f"phases {list(cyc.phase)}")
    if (phase == 4) != (cyc.move == "duct"):
        raise _err(path, f"{w}.cycle", f"phase {phase} applies "
                                       + ("duct links" if phase == 4 else
                                          "the walk cycle types")
                                       + f", not a {cyc.move!r} type")
    if "on" not in t:
        raise _err(path, f"{w}.on", "a cycle entry names its anchor class")
    on = R._str(path, f"{w}.on", t["on"])
    cls = skel.anchor(on, hub_dept.entry)
    if cls is None:
        known = [k for k in L.ANCHORS if k not in (L.HUB, L.LOBBY_HUB)] + [
            f"{skel.hub_department}.{hub_dept.entry}",
            f"{L.LOBBY}-{hub_dept.entry}"]
        raise _err(path, f"{w}.on", f"{on!r} is no anchor class of skeleton "
                                    f"{skel.name!r} ({known})")
    if cls.move != cyc.move:
        raise _err(path, f"{w}.on", f"{on!r} realises the {cls.move!r} move; "
                                    f"cycle type {cn!r} is a {cyc.move!r}")
    for s in sizes:
        if count[s][1] == 0:
            continue
        if cls.key == L.HUB:
            if G.SIZES[s].band_w[1] < L.RING_BAND_W_MIN:
                raise _err(path, f"{w}.count.{s}",
                           f"a ring needs a band >= {L.RING_BAND_W_MIN} wide "
                           f"and size {s!r}'s band is at most "
                           f"{G.SIZES[s].band_w[1]} (layout doc §4.4)")
            if not _ring_sizes(cyc):
                raise _err(path, f"{w}.cycle",
                           f"cycle type {cn!r}'s rooms {list(cyc.rooms)} "
                           f"admit no even ring of 4..{L.RING_ROOMS_MAX} "
                           f"(the skeleton's maximum)")
        if cls.key == L.LOBBY_HUB and \
                hub_dept.room(hub_dept.entry).size.b[1] < L.BYPASS_HUB_H_MIN:
            raise _err(path, f"{w}.count.{s}",
                       f"a bypass needs the hub section >= "
                       f"{L.BYPASS_HUB_H_MIN} tall; {hub_dept.name!r}'s hub "
                       f"is at most {hub_dept.room(hub_dept.entry).size.b[1]}"
                       f" (layout doc §4.3)")
    return ApplyEntry(i, phase, None, cn, on, count)


def _promote(path, where: str, v, library, named: set) -> tuple:
    names = R._str_list(path, where, v)
    for j, n in enumerate(names):
        w = f"{where}[{j}]"
        if n not in library.departments:
            raise _err(path, w, f"{n!r} is no department in the library")
        dept = library.departments[n]
        if n not in named:
            raise _err(path, w, f"{n!r} is not a department this archetype "
                                f"names ({sorted(named)})")
        if dept.cell == "own":
            raise _err(path, w, f"{n!r} is already cell = \"own\" in its "
                                f"file: promoting it is redundant")
        if dept.placement != "wing":
            raise _err(path, w, f"{n!r} is an axis department; promote "
                                f"makes a WING department its own cell")
        stub = dept.room(dept.entry)
        if stub.size.a[1] < L.OWN_STUB_ACROSS_MIN:
            raise _err(path, w, f"{n!r}'s stub across {list(stub.size.a)} "
                                f"cannot take the airlock entry (>= "
                                f"{L.OWN_STUB_ACROSS_MIN}; layout doc §4)")
    if len(set(names)) != len(names):
        raise _err(path, where, f"{list(names)} names a department twice")
    return names


def _validation(path, t, sizes) -> dict:
    R._keys(path, "[validation]", t, tuple(VALIDATION_GATES))
    out = {}
    for k, v in t.items():
        w = f"validation.{k}"
        shape = VALIDATION_GATES[k]
        if shape == "bool":
            out[k] = R._bool(path, w, v)
        elif shape == "range":
            out[k] = R._range(path, w, v, 0)
        else:
            R._keys(path, w, v, sizes, sizes)
            out[k] = {s: R._range(path, f"{w}.{s}", v[s], 0) for s in sizes}
    return out


def _closure(arch: Archetype) -> None:
    """The transitive-closure walk (archetype doc §1.1): every department the
    file names resolves with NOTHING pending (a pending key names the
    unbuilt extension it waits for) and has no [[cycle]] rows the grammar
    does not apply; no department or cycle uses an edge kind gen_ship cannot
    stamp yet."""
    lib = arch.library
    rng = np.random.default_rng(0)
    for n in arch.departments():
        dept = lib.departments[n]
        if dept.cycles:
            raise _err(arch.path, f"department {n!r}",
                       f"its file has [[cycle]] rows: refused, naming the "
                       f"extension -- {EXT_DEPT_CYCLES}")
        for e in dept.edges:
            if G.EDGE_KINDS[e.kind].pending:
                raise _err(arch.path, f"department {n!r}",
                           f"edge {e.a}-{e.b} is a {e.kind!r}, which waits "
                           f"for {G.EDGE_KINDS[e.kind].pending}")
        for r in dept.rooms:
            rr = R.resolve_room(dept, r.id, lib.fixtures, rng,
                                cell=R.MAIN_CELL + 1)
            if rr.pending:
                k = sorted(rr.pending)[0]
                raise _err(arch.path, f"department {n!r}",
                           f"room {r.id!r}'s {k!r} waits for "
                           f"{rr.pending[k][1]}: the department cannot be "
                           f"stamped yet")
    for e in arch.applies:
        if e.cycle:
            for lg in lib.cycles[e.cycle].legs:
                for k in lg.edges:
                    if G.EDGE_KINDS[k].pending:
                        raise _err(arch.path, f"apply[{e.index}].cycle",
                                   f"cycle type {e.cycle!r} may draw "
                                   f"{k!r}, which waits for "
                                   f"{G.EDGE_KINDS[k].pending}")


def _block(dept: R.Department) -> int:
    """The hub department's BLOCK at its minimum: the hub plus its required
    fore / aft rooms (layout doc §13 A2)."""
    return 1 + sum(1 for r in dept.rooms
                   if r.attach in R.ATTACH_AXIS and r.required)


def _sections(arch: Archetype, size: str) -> tuple:
    plo, phi = G.SIZES[size].sections
    block = _block(arch.library.departments[arch.hub_department])
    lo = max(arch.band_sections[0], plo, block + 1)
    hi = min(arch.band_sections[1], phi)
    if lo > hi:
        raise _err(arch.path, "archetype.band.sections",
                   f"{list(arch.band_sections)} intersected with size "
                   f"{size!r}'s {[plo, phi]} cannot hold the hub "
                   f"department's {block}-section block plus one main "
                   f"section (layout doc §13 A2)")
    return (lo, hi)


def _ring_sizes(cyc: R.Cycle) -> tuple:
    lo, hi = cyc.rooms
    return tuple(n for n in range(max(4, lo), min(L.RING_ROOMS_MAX, hi) + 1)
                 if n % 2 == 0)


def _cap(arch: Archetype, cls: L.AnchorClass, size: str) -> int | None:
    """The anchor class's closed-form realisable maximum on ``size`` (None:
    bounded only by the layout's slack)."""
    if cls.cap == "one":
        return 1
    if cls.cap == "departments_with_end":
        lib = arch.library
        n = 0
        for x in arch.axiom:
            d = lib.departments[x.department]
            n += d.placement == "wing" and any(r.attach == "end"
                                               for r in d.rooms)
        for e in arch.applies:
            if e.department:
                d = lib.departments[e.department]
                if any(r.attach == "end" for r in d.rooms):
                    n += e.count[size][1]
        return n
    return None


def _ranges(arch: Archetype, size: str, profile: str) -> dict:
    """Every [[apply]] entry's EFFECTIVE count on (``size``, ``profile``):
    a department's is the file's; a cycle's is shifted by ``cycles_shift``
    (both ends, floored at 0) and capped at its class's realisable maximum,
    where a file range of [0, 0] stays [0, 0] (the file's own statement
    that the type is not realised on that size), and the minima of an
    EXCLUSIVE group (the ring and the bypass) are capped in file order so
    the group's minima never exceed what it can realise together."""
    skel = L.SKELETONS[arch.skeleton]
    hub = arch.library.departments[arch.hub_department]
    shift = arch.profiles[profile].cycles_shift
    out, left = {}, dict(L.EXCLUSIVE_MAX)
    for e in arch.applies:
        lo, hi = e.count[size]
        if e.cycle and hi > 0:
            cls = skel.anchor(e.on, hub.entry)
            lo, hi = max(0, lo + shift), max(0, hi + shift)
            cap = _cap(arch, cls, size)
            if cap is not None:
                lo, hi = min(lo, cap), min(hi, cap)
            if cls.exclusive:
                lo = min(lo, left[cls.exclusive])
                left[cls.exclusive] -= lo
        out[e.index] = (lo, hi)
    return out


# ===========================================================================
# recipe_hash (archetype doc §5.1)
# ===========================================================================

def _plain(v):
    """A config namespace / TOML value as plain JSON-able data."""
    if hasattr(v, "__dict__") and not isinstance(v, type):
        return {k: _plain(x) for k, x in sorted(vars(v).items())}
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def recipe_hash(archetype: Archetype, profile: str,
                library: R.RecipeLibrary | None = None) -> str:
    """SHA-256 over the parsed, key-sorted archetype file WITHOUT its
    [profiles] table, plus ``profile``'s table, plus the parsed files of
    every department and cycle it names and the fixtures those departments
    use, plus every preset value they resolve (lights, fuel classes, temps,
    pressures, heat sources, hazards, filters) and, when any duct exists,
    ``[levelgen.vents] ach`` (the vents' design flow). A comment edit does
    not move it; retuning another profile does not move this one's."""
    lib = library or archetype.library
    raw = R._read(Path(archetype.path))
    prof = raw.get("profiles", {})
    if profile not in prof:
        raise _err(archetype.path, "profiles", f"no profile {profile!r} "
                                               f"({sorted(prof)})")
    body = {"archetype": {k: v for k, v in raw.items() if k != "profiles"},
            "profile": {profile: prof[profile]}}
    depts, cycles, fixtures, presets = {}, {}, {}, {}

    def preset(kind: str, name: str, fn):
        presets[f"{kind}.{name}"] = _plain(fn(name))

    for n in archetype.departments():
        d = lib.departments[n]
        depts[n] = R._read(Path(d.path))
        for r in d.rooms:
            if r.fixture:
                fixtures[r.fixture] = _plain(vars(lib.fixtures[r.fixture]))
            eff = R._effective(r, lib.fixtures)
            if r.light:
                preset("lights", r.light, R.light_preset)
            if "fuel" in eff:
                preset("fuel", eff["fuel"], R.fuel_preset)
            if "heat_source" in eff:
                preset("heat_sources", eff["heat_source"],
                       R.heat_source_preset)
            if "hazard" in eff:
                preset("hazards", eff["hazard"], R.hazard_preset)
        for x in d.ducts:
            presets[f"filters.{x.filter}"] = _plain(
                R._table(f"filters.{x.filter}"))
        if d.physics.temp != R.AMBIENT:
            preset("temps", d.physics.temp, R.temp_preset)
        if d.physics.pressure != R.AMBIENT:
            preset("pressures", d.physics.pressure, R.pressure_preset)
    for e in archetype.applies:
        if e.cycle:
            c = lib.cycles[e.cycle]
            cycles[e.cycle] = R._read(Path(c.path))
            for lg in c.legs:
                if lg.light:
                    preset("lights", lg.light, R.light_preset)
            for x in c.ducts:
                presets[f"filters.{x.filter}"] = _plain(
                    R._table(f"filters.{x.filter}"))
    if any(k.startswith("filters.") for k in presets):
        # the vent design flow's dial (vocabulary §0.6 item 3): a ship's
        # vent rates are a function of it, so a retune makes the record stale
        presets["levelgen.vents.ach"] = G.vent_ach()
    # the roof lights' shapes (#177): every recipe ship draws from them
    presets["levelgen.lamp_shapes"] = _plain(R._table("levelgen.lamp_shapes"))
    body.update(departments=depts, cycles=cycles, fixtures=fixtures,
                presets=presets)
    blob = json.dumps(_plain(body), sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ===========================================================================
# The executor (archetype doc §2)
# ===========================================================================

@dataclass
class _Ctx:
    arch: Archetype
    size: str
    profile: str
    rng: np.random.Generator
    cells: dict
    promoted: tuple
    zombies: int
    placed: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    instances: dict = field(default_factory=dict)

    @property
    def lib(self) -> R.RecipeLibrary:
        return self.arch.library

    @property
    def skel(self) -> L.Skeleton:
        return L.SKELETONS[self.arch.skeleton]

    def hub(self, d: Draft) -> str:
        """The hub room's id (the hub department's one instance)."""
        return next(q.entry for q in d.depts
                    if q.name == self.arch.hub_department)

    def draw(self, n: int) -> int:
        return int(self.rng.integers(n))


def _check(d: Draft, phase: int, rule: str) -> None:
    """The phase's invariants on a MANDATORY step: a violation ends the
    attempt."""
    try:
        check_invariants(d, phase)
    except InvariantViolation as e:
        raise GrammarReject(rule, str(e)) from None


def _dept_tags(dept: R.Department, role: str, cell: int) -> dict:
    """A made room's tags inside a department: dept, role, cell, and the
    department's temperature (every room of it starts so)."""
    tags = {"dept": dept.name, "role": role, "cell": cell}
    if dept.physics.temp != R.AMBIENT:
        tags["temp"] = R.temp_preset(dept.physics.temp, path=dept.path,
                                     where="physics.temp")
    return tags


def _qsize(s: R.Size) -> QSize:
    return QSize(s.axes, s.a, s.b)


def _expand(d: Draft, ctx: _Ctx, name: str, rule: str, order: int) -> Draft:
    """Phase 1: one INSTANCE of department ``name`` -- its rooms (``count``
    drawn, ``p`` rolled; a behind room whose parent was not rolled is
    absent), each resolved through ``levelgen_recipes.resolve_room`` with the
    grammar's cell and rng (duct names qualified per instance), its edges
    (per instance pair), its ducts, and its entry edge to the passageway (an
    airlock with ``a`` = the stub when the department is promoted)."""
    dept = ctx.lib.departments[name]
    inst = ctx.instances.get(name, 0)
    ctx.instances[name] = inst + 1
    cell = ctx.cells[name]
    promoted = name in ctx.promoted
    side = AXIS if dept.placement == "axis" else WING
    out = d.copy()
    prov = (rule, out.next_app())
    qual = f"{name}_{inst}_"
    ids: dict = {}
    for r in dept.rooms:
        if r.required:
            n = 1
        elif r.count:
            n = int(ctx.rng.integers(r.count[0], r.count[1] + 1))
        else:
            n = 1 if ctx.rng.random() < r.p else 0
        if r.parent and not ids.get(r.parent):
            n = 0
        ids[r.id] = [f"{name}.{inst}.{r.id}.{k}" for k in range(n)]
    entry = ids[dept.entry][0]
    for r in dept.rooms:
        for k, rid in enumerate(ids[r.id]):
            rr = R.resolve_room(dept, r.id, ctx.lib.fixtures, ctx.rng,
                                cell=cell if dept.cell == "own" else None)
            tags = rr.stamp_tags()
            tags["cell"] = cell
            if "vent" in tags:
                vt = [{"duct": qual + dn, "roles": list(roles)}
                      for dn, roles, _sh in G.vent_specs(tags["vent"])]
                tags["vent"] = vt[0] if len(vt) == 1 else vt
            attach, parent = r.attach, None
            if r.parent:
                attach, parent = "behind", ids[r.parent][0]
            elif r.attach is not None:
                parent = entry
            out.rooms[rid] = QRoom(
                id=rid, side=side,
                kind="section" if side == AXIS else "room",
                shape=r.shape, width=r.width or "wide", size=_qsize(r.size),
                dept=name, instance=inst, recipe_id=r.id, index=k,
                attach=attach, parent=parent, exit=r.exit, cell=cell,
                tags=tags, prov=prov)
    for e in dept.edges:
        for ia in ids[e.a]:
            for ib in ids[e.b]:
                out = join(out, ia, ib, e.kind, prov, "department")
    used = {dn for r in out.rooms.values() if r.dept == name
            and r.instance == inst for dn, _ro, _sh in _vents(r)}
    for x in dept.ducts:
        if qual + x.name in used:
            out.ducts[qual + x.name] = x.filter
    if side == WING:
        if promoted:
            out = join(out, entry, L.PASSAGEWAY,
                       ctx.skel.edges["entry-own"], prov, "entry")
        else:
            out = join(out, L.PASSAGEWAY, entry, ctx.skel.edges["entry"],
                       prov, "entry")
    out.depts.append(QDepartment(
        name=name, instance=inst, placement=dept.placement, cell=cell,
        toward=dept.toward, entry=entry, stubs=(entry,), promoted=promoted,
        order=order, prov=prov))
    ctx.log.append((1, rule, entry, "ok" + (" (promoted: entry is an "
                                            "airlock)" if promoted else "")))
    return out


def _axiom(ctx: _Ctx) -> Draft:
    """The skeleton's rooms and the axiom departments, then the band (its
    section count drawn, the hub department's block aft) and the axiom
    edges (layout doc §2, §13 A1 / A2; decisions of 2026-11-06)."""
    arch, skel = ctx.arch, ctx.skel
    d = Draft()
    prov = ("skeleton", 0)
    d.rooms[L.SLAB] = QRoom(
        id=L.SLAB, kind="spawn", role=L.SLAB,
        size=QSize(("w", "h"), None, None),
        tags={"spawn_team": 0, "role": L.SLAB, "cell": R.MAIN_CELL},
        prov=prov)
    d.rooms[L.PASSAGEWAY] = QRoom(
        id=L.PASSAGEWAY, shape="corridor", role=L.PASSAGEWAY,
        size=QSize(("across", "along"), arch.passageway, None),
        tags={"role": L.PASSAGEWAY, "cell": R.MAIN_CELL}, prov=prov)
    d.rooms[L.LOBBY] = QRoom(
        id=L.LOBBY, role=L.LOBBY, size=QSize(("w", "h"), arch.lobby_x, None),
        tags={"role": L.LOBBY, "cell": R.MAIN_CELL}, prov=prov)
    d = join(d, L.SLAB, L.PASSAGEWAY, skel.edges["slab-passageway"], prov,
             "skeleton")
    d = join(d, L.PASSAGEWAY, L.LOBBY, skel.edges["passageway-lobby"], prov,
             "skeleton")
    for i, x in enumerate(arch.axiom):
        d = _expand(d, ctx, x.department, f"axiom:{x.department}", i)
    hub_q = next(q for q in d.depts if q.name == arch.hub_department)
    hub = hub_q.entry
    rooms = [r for r in d.rooms.values() if r.dept == hub_q.name]
    block = [r.id for r in rooms if r.attach == "fore"] + [hub] + \
        [r.id for r in rooms if r.attach == "aft"]
    lo, hi = arch.sections[ctx.size]
    lo = max(lo, len(block) + 1)
    if lo > hi:
        raise GrammarReject("skeleton", f"a {len(block)}-section block "
                                        f"leaves no main section in "
                                        f"{hi} sections")
    k = int(ctx.rng.integers(lo, hi + 1))
    mains = [f"{L.SECTION}_{i}" for i in range(k - len(block))]
    for i, sid in enumerate(mains):
        d.rooms[sid] = QRoom(
            id=sid, side=AXIS, kind="section", role=L.SECTION, index=i,
            size=QSize(("w", "h"), None, None),
            tags={"role": L.SECTION, "cell": R.MAIN_CELL}, prov=prov)
    for x, y in zip(mains, mains[1:]):
        d = join(d, x, y, skel.edges["section-section"], prov, "skeleton")
    last = mains[-1]
    d = join(d, last, block[0], skel.edges["main-block"], prov, "skeleton")
    d = join(d, L.LOBBY, last, skel.edges["lobby-section"], prov, "skeleton")
    d = join(d, L.LOBBY, hub, skel.edges["lobby-hub"], prov, "skeleton")
    if ctx.zombies:
        r = d.rooms[last]
        d.rooms[last] = replace(r, tags={**r.tags, "zombie": True})
    d.band = tuple(mains + block)
    _check(d, 1, "skeleton")
    return d


def _rows(d: Draft) -> Draft:
    """Layout doc §13 A7: wing departments in gradient order (``toward =
    "spawn"`` first, ``"axis"`` last; ties in file order) dealt to rows A, B
    alternately starting with A; ``position`` counts from the slab end."""
    out = d.copy()
    wing = [i for i, q in enumerate(out.depts) if q.placement == "wing"]
    ranked = sorted(wing, key=lambda i: (0 if out.depts[i].toward == "spawn"
                                         else 1, i))
    for j, i in enumerate(ranked):
        out.depts[i] = replace(out.depts[i], row=ROWS[j % 2],
                               position=j // 2, order=j)
    return out


# ---- candidates and realisations per anchor class -------------------------

def _can_take_behind(d: Draft, r: QRoom) -> bool:
    """A room the layout can split a behind room off: a wing side / end
    room with no behind child, never the exit room (its far side is the
    slab), a hull-door room (it must stay against the hull) or a secret."""
    return (r.side == WING and r.dept is not None
            and r.attach in ("side", "end") and not r.exit
            and "hull_door" not in r.tags and not r.tags.get("secret")
            and not any(c.attach == "behind" for c in d.children(r.id)))


def _is_stub(r: QRoom) -> bool:
    return (r.side == WING and r.dept is not None and r.shape == "corridor"
            and (r.attach is None or r.role == L.STUB2))


def _candidates(d: Draft, ctx: _Ctx, cls: L.AnchorClass) -> list:
    """``(global id, payload)`` per candidate of the class."""
    if cls.key == L.DEPARTMENT_ENTRY:
        return [(d.rooms[q.entry].gid, q.entry) for q in d.depts
                if q.placement == "wing"]
    if cls.key == L.SLAB_LOBBY:
        return [(d.rooms[L.SLAB].gid, (L.SLAB, L.LOBBY))]
    if cls.key in (L.HUB, L.LOBBY_HUB):
        hub = ctx.hub(d)
        return [(d.rooms[hub].gid, hub)]
    if cls.key == L.DEPARTMENT_ROOM:
        return [(r.gid, r.id) for r in d.rooms.values()
                if r.role not in (L.RING_FORE, L.RING_AFT, L.RING_SIDE)
                and (_is_stub(r) or _can_take_behind(d, r))]
    if cls.key == L.SAME_CELL_ROOMS:
        rooms = sorted((r for r in d.rooms.values() if r.dept is not None
                        and not r.tags.get("secret")), key=lambda r: r.gid)
        out = []
        for i, a in enumerate(rooms):
            for b in rooms[i + 1:]:
                if ((a.dept, a.instance) != (b.dept, b.instance)
                        and a.cell == b.cell and a.side == b.side):
                    out.append(((a.gid, b.gid), (a.id, b.id)))
        return out
    raise AssertionError(cls.key)


def _kinds(ctx: _Ctx, leg: R.Leg, n: int) -> tuple:
    """``n`` edge kinds drawn uniformly from the leg's ``edges``."""
    return tuple(leg.edges[ctx.draw(len(leg.edges))] for _ in range(n))


def _walk_leg(cyc: R.Cycle) -> R.Leg:
    return next(lg for lg in cyc.legs if lg.walk)


def _retreat_loop(d, ctx, cyc, entry, stub):
    """``second_leg`` on a department's ENTRY (layout doc §4.1): a second
    stub from the passageway into the END room; the side rooms are split
    between the two stubs alternately (the exit room stays on the first)."""
    q = next(x for x in d.depts if x.entry == stub)
    if len(q.stubs) > 1:
        raise PreconditionFailed("retreat_loop", f"{q.name}#{q.instance} "
                                                 f"already has two stubs")
    ends = [c for c in d.children(stub) if c.attach == "end"]
    if not ends:
        raise PreconditionFailed("retreat_loop", f"{q.name}#{q.instance} has "
                                                 f"no attach = \"end\" room")
    end = ends[0].id
    leg = _walk_leg(cyc)
    kinds = _kinds(ctx, leg, 2)
    dept = ctx.lib.departments[q.name]
    s1 = d.rooms[stub]
    out = d.copy()
    prov = (entry.rule, out.next_app())
    s2 = QRoom(id=f"{q.name}.{q.instance}.{L.STUB2}.{prov[1]}", side=WING,
               shape=s1.shape, width=leg.width or s1.width, size=s1.size,
               role=L.STUB2, dept=q.name, instance=q.instance, index=prov[1],
               attach="stub", parent=stub, cell=q.cell,
               tags=_dept_tags(dept, L.STUB2, q.cell), prov=prov)
    out = second_leg(out, L.PASSAGEWAY, stub, end, s2, kinds, prov)
    sides = [c for c in out.children(stub) if c.attach == "side"
             and not c.exit]
    for i, c in enumerate(sides):
        if i % 2 == 1:
            out = _rewrite(out, stub, c.id, s2.id, c.id, prov)
            out.rooms[c.id] = replace(c, parent=s2.id)
    j = out.depts.index(q)
    out.depts[j] = replace(q, stubs=(stub, s2.id))
    return out, (stub, end), kinds, ()


def _flank_pair(d, ctx, cyc, entry, pair):
    """``second_leg`` on slab-passageway-lobby (layout doc §4.2, §13 A3):
    one ``narrow``, dim hull strip with a door down into the slab and one
    into the lobby."""
    if any(r.role == L.FLANK_STRIP for r in d.rooms.values()):
        raise PreconditionFailed("flank_pair", "the wing already has its "
                                               "hull strip")
    leg = _walk_leg(cyc)
    kinds = _kinds(ctx, leg, 2)
    tags = {"role": L.FLANK_STRIP, "cell": R.MAIN_CELL}
    if leg.light:
        tags["light"] = R.light_preset(leg.light, path=cyc.path,
                                       where="leg.light")
    out = d.copy()
    prov = (entry.rule, out.next_app())
    strip = QRoom(id=f"{cyc.name}.{prov[1]}.strip", shape="corridor",
                  width=leg.width or "narrow",
                  size=QSize(("across", "along"),
                             (L.FLANK_STRIP_ACROSS, L.FLANK_STRIP_ACROSS),
                             None),
                  role=L.FLANK_STRIP, index=prov[1], attach="hull_strip",
                  tags=tags, prov=prov)
    out = second_leg(out, L.SLAB, L.PASSAGEWAY, L.LOBBY, strip, kinds, prov)
    return out, pair, kinds, ()


def _rewrite(d: Draft, a: str, b: str, na: str, nb: str,
             prov: tuple) -> Draft:
    """An application's edge rewrite: the one edge a-b becomes na-nb, its
    kind, order and role kept, its provenance the application's."""
    old = _one_edge("rewrite", d, a, b)
    out = d.copy()
    i = out.edges.index(old)
    if old.a == a:
        out.edges[i] = replace(old, a=na, b=nb, prov=prov)
    else:
        out.edges[i] = replace(old, a=nb, b=na, prov=prov)
    return out


def _maintenance_ring(d, ctx, cyc, entry, hub):
    """``ring`` on the hub (layout doc §4.4, §13 A2): 4 or 6 rooms, the hub
    joined to the odd-numbered ones; then the application's rewrites:
    lobby-hub -> lobby-<the aft-most side room>, hub-<aft room> ->
    <ring aft>-<aft room> (fore likewise), the block's bulkhead onto the
    ring's fore room when the hub led the block."""
    move = "maintenance_ring"
    if G.SIZES[ctx.size].band_w[1] < L.RING_BAND_W_MIN:
        raise PreconditionFailed(move, f"size {ctx.size!r}'s band is under "
                                       f"{L.RING_BAND_W_MIN}")
    if any(e.kind == "breakable" for e in d.between(L.LOBBY, hub)):
        raise PreconditionFailed(move, "mutually exclusive with the airlock "
                                       "bypass (archetype doc §3)")
    if any(c.role == L.RING_FORE for c in d.children(hub)):
        raise PreconditionFailed(move, f"{hub!r} already has its ring")
    sizes = _ring_sizes(cyc)
    n = sizes[ctx.draw(len(sizes))]
    k_ring = _kinds(ctx, cyc.legs[0], 1)[0]
    k_hub = _kinds(ctx, cyc.legs[1], 1)[0]
    hq = next(q for q in d.depts if q.entry == hub)
    dept = ctx.lib.departments[hq.name]
    out = d.copy()
    prov = (entry.rule, out.next_app())
    base = f"{cyc.name}.{prov[1]}"

    def mk(rid, role, side, size, idx):
        return QRoom(id=f"{base}.{rid}", side=side,
                     kind="section" if side == AXIS else "room", size=size,
                     role=role, dept=hq.name, instance=hq.instance,
                     index=idx, attach=role, parent=hub, cell=hq.cell,
                     tags=_dept_tags(dept, "ring", hq.cell), prov=prov)
    depth = (L.RING_DEPTH, L.RING_DEPTH)
    fore = mk("r_fore", L.RING_FORE, AXIS, QSize(("w", "h"), None, depth), 1)
    aft = mk("r_aft", L.RING_AFT, AXIS, QSize(("w", "h"), None, depth),
             n // 2 + 1)
    sides = tuple(mk(f"r_side{i}", L.RING_SIDE, WING,
                     QSize(("w", "h"), depth, None), n // 2 + 2 + i)
                  for i in range((n - 2) // 2))
    out = ring(out, hub, fore, aft, sides, k_ring, k_hub, prov)
    out = _rewrite(out, L.LOBBY, hub, L.LOBBY, sides[0].id, prov)
    for c in out.children(hub):
        if c.attach in ("fore", "aft"):
            via = fore if c.attach == "fore" else aft
            out = _rewrite(out, hub, c.id, via.id, c.id, prov)
            out.rooms[c.id] = replace(c, parent=via.id)
    band = list(out.band)
    block_first = next(x for x in band if out.rooms[x].role != L.SECTION)
    if block_first == hub:
        last = band[band.index(hub) - 1]
        out = _rewrite(out, last, hub, last, fore.id, prov)
    i = band.index(hub)
    out.band = tuple(band[:i] + [fore.id, hub, aft.id] + band[i + 1:])
    return out, (hub,), (k_ring, k_hub), (("n", n),)


def _airlock_bypass(d, ctx, cyc, entry, hub):
    """``parallel`` on the lobby's airlock into the hub (layout doc §4.3): a
    ``breakable`` beside it; mutually exclusive with the ring."""
    move = "airlock_bypass"
    if any(c.role == L.RING_FORE for c in d.children(hub)):
        raise PreconditionFailed(move, "mutually exclusive with the "
                                       "maintenance ring (archetype doc §3)")
    if not any(e.kind == "airlock" for e in d.between(L.LOBBY, hub)):
        raise PreconditionFailed(move, f"no airlock lobby-{hub}")
    hs = d.rooms[hub].size
    if hs is None or hs.b is None or hs.b[1] < L.BYPASS_HUB_H_MIN:
        raise PreconditionFailed(move, f"the hub {hub!r} cannot be "
                                       f"{L.BYPASS_HUB_H_MIN} tall")
    leg = next(lg for lg in cyc.legs if not lg.walk)
    kinds = _kinds(ctx, leg, 1)
    out = d.copy()
    prov = (entry.rule, out.next_app())
    out = parallel(out, L.LOBBY, hub, kinds[0], prov)
    return out, (L.LOBBY, hub), kinds, ()


def _pocket(d, ctx, cyc, entry, anchor):
    """``leaf`` on a department room (layout doc §4.5): a new side room on
    a stub, or a new behind room off a side / end room with none."""
    r = d.rooms[anchor]
    if not (_is_stub(r) or _can_take_behind(d, r)):
        raise PreconditionFailed("pocket", f"{anchor!r} can no longer take a "
                                           f"room")
    leg = _walk_leg(cyc)
    kinds = _kinds(ctx, leg, 1)
    dept = ctx.lib.departments[r.dept]
    out = d.copy()
    prov = (entry.rule, out.next_app())
    w, h = L.POCKET_SIZE
    room = QRoom(id=f"{r.dept}.{r.instance}.{L.POCKET}.{prov[1]}", side=WING,
                 width=leg.width or "wide", size=QSize(("w", "h"), w, h),
                 role=L.POCKET, dept=r.dept, instance=r.instance,
                 index=prov[1],
                 attach="side" if _is_stub(r) else "behind", parent=anchor,
                 cell=r.cell, tags=_dept_tags(dept, L.POCKET, r.cell),
                 prov=prov)
    out = leaf(out, anchor, room, kinds[0], prov)
    return out, (anchor,), kinds, ()


def _duct_link(d, ctx, cyc, entry, pair):
    """``duct`` between two rooms of one cell in different department
    instances (layout doc §4.6): a new duct, both rooms carrying the cycle
    file's roles; a room joins at most one link (one vent site per role)."""
    a, b = pair
    for x in pair:
        if any(dn.startswith("link_") for dn, _r, _s in _vents(d.rooms[x])):
            raise PreconditionFailed("duct_link", f"{x!r} is already on a "
                                                  f"duct link")
    spec = cyc.ducts[0]
    out = d.copy()
    prov = (entry.rule, out.next_app())
    out = duct(out, a, b, f"link_{prov[1]}", spec.filter, spec.roles, prov)
    return out, pair, (), (("duct", f"link_{prov[1]}"),)


REALISE = {L.DEPARTMENT_ENTRY: _retreat_loop, L.SLAB_LOBBY: _flank_pair,
           L.HUB: _maintenance_ring, L.LOBBY_HUB: _airlock_bypass,
           L.DEPARTMENT_ROOM: _pocket, L.SAME_CELL_ROOMS: _duct_link}


def _multiset(edges) -> dict:
    out: dict = {}
    for e in edges:
        k = (e.a, e.b, e.kind)
        out[k] = out.get(k, 0) + 1
    return out


def _diff(before: dict, after: dict) -> tuple:
    return tuple(sorted(k for k, n in after.items()
                        for _ in range(n - before.get(k, 0))))


def _run_entry(d: Draft, ctx: _Ctx, e: ApplyEntry, phase: int) -> Draft:
    """One [[apply]] cycle entry (archetype doc §2): ``k`` uniform in the
    effective range, candidates sorted by global id and drawn uniformly
    without replacement; each application must keep the phase's
    invariants; a rejected candidate is logged and does not count."""
    lo, hi = ctx.arch.ranges[(ctx.size, ctx.profile)][e.index]
    k = int(ctx.rng.integers(lo, hi + 1))
    cyc = ctx.lib.cycles[e.cycle]
    hub = ctx.lib.departments[ctx.arch.hub_department]
    cls = ctx.skel.anchor(e.on, hub.entry)
    cand = sorted(_candidates(d, ctx, cls), key=lambda c: c[0])
    placed = 0
    while placed < k and cand:
        gid, payload = cand.pop(ctx.draw(len(cand)))
        try:
            nd, anchor, kinds, params = REALISE[cls.key](d, ctx, cyc, e,
                                                         payload)
            check_invariants(nd, phase)
        except Rejected as err:
            ctx.log.append((phase, e.rule, repr(gid), f"rejected: {err}"))
            continue
        before, after = _multiset(d.edges), _multiset(nd.edges)
        nd.cycles.append(CycleApp(
            type=e.cycle, move=cyc.move, on=e.on, anchor=tuple(anchor),
            rooms=tuple(r for r in nd.rooms if r not in d.rooms),
            edges=_diff(before, after), removed=_diff(after, before),
            kinds=tuple(kinds), params=tuple(params), entry=e.index,
            index=nd.n_app))
        d = nd
        placed += 1
        ctx.log.append((phase, e.rule, repr(gid), "ok"))
    ctx.placed[e.index] = placed
    if placed < lo:
        raise GrammarReject(e.rule, f"placed {placed} of at least {lo} "
                                    f"(drew {k}, {len(cand)} candidates left)")
    return d


def _exits(d: Draft, ctx: _Ctx) -> tuple:
    """Layout doc §7 with §13 A7: 1 = slab-passageway; 2 = + the slab to the
    ``exit`` room of row A's first department; 3 = + row B's -- an exit is
    available only when that department has an exit room in the slab's cell;
    the draw is capped at what the ship realises (a flank strip's door is
    extra, archetype doc §9)."""
    lo, hi = ctx.arch.exits
    want = int(ctx.rng.integers(lo, hi + 1))
    avail = []
    for row in ROWS:
        q = next((x for x in d.depts if x.row == row and x.position == 0),
                 None)
        if q is None:
            continue
        ex = [r for r in d.rooms.values() if r.dept == q.name
              and r.instance == q.instance and r.exit
              and r.cell == d.rooms[L.SLAB].cell]
        if ex:
            avail.append(ex[0].id)
    n = min(want, 1 + len(avail))
    out = d.copy()
    prov = ("exits", out.next_app())
    edges = [e for e in out.between(L.SLAB, L.PASSAGEWAY)]
    for rid in avail[:n - 1]:
        out = join(out, L.SLAB, rid, ctx.skel.edges["exit"], prov, "exit")
        edges.append(out.edges[-1])
    _check(out, 2, "exits")
    ctx.log.append((2, "exits", f"drew {want}", f"ok: {n} (available "
                                                f"{1 + len(avail)})"))
    return out, n, tuple(edges)


def _secrets(d: Draft, ctx: _Ctx) -> tuple:
    """Phase 3: the archetype's secret budget (the profile's if set). Each
    unit draws a department instance by its ``[secret].weight`` among those
    whose host room has an instance that can take a behind room, then one
    such instance uniformly; the annex (``secrets.size`` both sides) goes
    BEHIND it, joined by the department's secret kind, tagged ``secret``;
    a ``vent_only`` annex rides the host's duct with both roles."""
    out, annexes = d, []
    for unit in range(ctx.arch.secrets(ctx.profile)):
        elig = []
        for q in out.depts:
            sec = ctx.lib.departments[q.name].secret
            if sec is None:
                continue
            hosts = [r.id for r in out.rooms.values() if r.dept == q.name
                     and r.instance == q.instance and r.recipe_id == sec.host
                     and _can_take_behind(out, r)]
            if hosts:
                elig.append((q, sec, hosts))
        if not elig:
            raise GrammarReject("secrets", f"unit {unit}: no department can "
                                           f"host an annex")
        total = sum(sec.weight for _q, sec, _h in elig)
        x, acc = ctx.rng.random() * total, 0.0
        q, sec, hosts = elig[-1]
        for cand in elig:
            acc += cand[1].weight
            if x < acc:
                q, sec, hosts = cand
                break
        host = out.rooms[hosts[ctx.draw(len(hosts))]]
        dept = ctx.lib.departments[q.name]
        out = out.copy()
        prov = ("secrets", out.next_app())
        tags = {**_dept_tags(dept, L.SECRET, q.cell), "secret": True}
        if sec.kind == "vent_only":
            tags = _with_vent(tags, _vents(host)[0][0], G.VENT_ROLES)
        lo, hi = ctx.arch.secret_size
        annex = QRoom(id=f"{q.name}.{q.instance}.{L.SECRET}.{prov[1]}",
                      side=host.side, size=QSize(("w", "h"), (lo, hi),
                                                 (lo, hi)),
                      role=L.SECRET, dept=q.name, instance=q.instance,
                      index=prov[1], attach="behind", parent=host.id,
                      cell=q.cell, tags=tags, prov=prov)
        out = leaf(out, host.id, annex, sec.kind, prov, role="secret")
        _check(out, 3, "secrets")
        annexes.append(annex.id)
        ctx.log.append((3, "secrets", host.id, "ok"))
    return out, tuple(annexes)


def grammar_attempt(arch: Archetype, size: str, profile: str, seed: int,
                    g: int, *, max_per_team: int = 6, zombies: int = 0,
                    rhash: str | None = None) -> QuotientGraph:
    """ONE grammar attempt ``g`` of ``seed`` (every draw from
    ``SeedSequence([seed, 0, g])``): the axiom, phases 1-4, the final
    invariants and the layout's feasibility -- a :class:`QuotientGraph`, or
    :class:`GrammarReject`."""
    if size not in arch.sizes:
        raise ArchetypeError(f"{arch.path}: archetype.sizes: size {size!r} "
                             f"is not one of {list(arch.sizes)}")
    if profile not in arch.profiles:
        raise ArchetypeError(f"{arch.path}: profiles: no profile "
                             f"{profile!r} ({sorted(arch.profiles)})")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, g]))
    ctx = _Ctx(arch=arch, size=size, profile=profile, rng=rng,
               cells=arch.cells(profile), promoted=arch.promoted(profile),
               zombies=arch.scaled_zombies(profile, zombies))
    # phase 1: the axiom, then the department entries
    d = _axiom(ctx)
    for e in arch.applies:
        if e.phase != 1:
            continue
        lo, hi = arch.ranges[(size, profile)][e.index]
        n = int(rng.integers(lo, hi + 1))
        for _ in range(n):
            d = _expand(d, ctx, e.department, e.rule, len(d.depts))
            _check(d, 1, e.rule)
        ctx.placed[e.index] = n
    d = _rows(d)
    # phase 2: the cycle types, then the spawn exits
    for e in arch.applies:
        if e.phase == 2:
            d = _run_entry(d, ctx, e, 2)
    d, n_exits, exit_edges = _exits(d, ctx)
    # phase 3: the secret budget
    _check(d, 3, "phase 3")
    d, annexes = _secrets(d, ctx)
    # phase 4: the duct links
    _check(d, 4, "phase 4")
    for e in arch.applies:
        if e.phase == 4:
            d = _run_entry(d, ctx, e, 4)
    _check(d, 4, "final")
    for r in d.rooms.values():
        for k, v in r.tags.items():
            what, ok = G.ROOM_TAGS[k]
            if not ok(v):
                raise AssertionError(f"room {r.id!r}: tag {k!r} = {v!r} is "
                                     f"not {what}")
    depts = tuple(replace(q, rooms=tuple(r.id for r in d.rooms.values()
                                         if r.dept == q.name
                                         and r.instance == q.instance))
                  for q in d.depts)
    q = QuotientGraph(
        archetype=arch.name, skeleton=arch.skeleton, size=size,
        profile=profile, seed=seed, g=g, max_per_team=max_per_team,
        zombies_base=zombies, zombies=ctx.zombies,
        recipe_hash=rhash or recipe_hash(arch, profile),
        rooms=tuple(d.rooms.values()), edges=tuple(d.edges),
        departments=depts, band=d.band, cycles=tuple(d.cycles),
        exits=n_exits, exit_edges=exit_edges,
        ducts=tuple(sorted(d.ducts.items())),
        zombie_room=[x for x in d.band if d.rooms[x].role == L.SECTION][-1],
        secrets=annexes, placed=tuple(sorted(ctx.placed.items())),
        log=tuple(ctx.log))
    why = L.layout_feasibility(size, graph=q)
    if why:
        raise GrammarReject("layout_feasibility", why)
    return q


def plan_quotient(arch, size: str, profile: str, seed: int, *,
                  max_per_team: int = 6, zombies: int = 0, g0: int = 0
                  ) -> QuotientGraph:
    """THE GRAMMAR LOOP (archetype doc §2): attempts ``g = g0 ..
    G_MAX - 1`` of ``seed``; the first that returns a graph wins (its ``g``
    is on it). ``arch``: an :class:`Archetype` or an archetype name. The
    layout's attempt loop resumes here with ``g0 = g + 1`` when it cannot
    realise a graph. :class:`GrammarExhausted` if none does."""
    if not isinstance(arch, Archetype):
        arch = load_archetype(arch)
    rhash = recipe_hash(arch, profile)
    why = []
    for g in range(g0, G_MAX):
        try:
            return grammar_attempt(arch, size, profile, seed, g,
                                   max_per_team=max_per_team,
                                   zombies=zombies, rhash=rhash)
        except GrammarReject as e:
            why.append(f"g={g}: {e}")
    raise GrammarExhausted(f"archetype {arch.name!r} size {size!r} profile "
                           f"{profile!r} seed {seed}: no grammar attempt in "
                           f"{g0}..{G_MAX - 1} returned a graph ({why[-3:]})")
