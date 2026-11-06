r"""tools/ship_layout.py -- the LAYOUT stage (stage 1.5) of family B: today
only the freighter SKELETON's GRAPH side (#136;
docs/levelgen_layout_stage_2026-10-31.md §2 and §13 A1 / A2 / A7,
docs/levelgen_archetype_recipe_2026-10-31.md §3 and §7.1).

ONE MODULE IS THE SKELETON'S HOME (archetype doc §7.1): the grammar
(``tools/ship_grammar.py``) imports from here what a skeleton IS as a graph --
its quotient rooms by role, its axiom edge kinds, the band's section chain,
what each archetype ``on`` anchor class may act on and how many it can
realise -- and the layout patch adds the GEOMETRY (the seven guillotine
primitives, the row / department / band packers, the wall-slide repair, the
attempt loop) to this same module. Nothing here places a rectangle.

THE FREIGHTER SKELETON (layout doc §2, §13 A1 / A2), as a graph on the
QUOTIENT half (one wing + the axis band):

- ``slab`` (``kind = "spawn"``, ``spawn_team`` 0) -- ``passageway`` (ONE
  wide corridor) -- ``lobby`` (ONE wide room at the band end of the wing),
  joined by doors;
- the BAND: ``section`` rooms on the axis, fore to aft the main-cell
  sections then the hub department's BLOCK (its ``fore`` room, its hub, its
  ``aft`` room); consecutive main sections joined by a ``door``, the last
  main section and the block by a ``bulkhead``; the lobby's ``door`` goes to
  the aft-most main section (which carries the horde);
- the hub department's entry: the AIRLOCK ``(a = lobby, b = hub)`` -- the
  vestibule carved on the LOBBY's side (A1).

A 3-cycle all of whose pairs are skeleton contacts (the lobby, the aft-most
main section and the block's first section) is a T-junction the skeleton
DRAWS (the lobby's band wall runs along both stacked sections): the
grammar's no-triangle invariant exempts exactly that, never a triangle a
move or a department makes.
"""
from __future__ import annotations

from dataclasses import dataclass

from gen_ship import (AIRLOCK_W, DOOR_W, FORMATION_FOOTPRINT, MIN_ROOM,
                      VESTIBULE_D)

# ---- the skeleton's quotient rooms, by role (layout doc §2) ---------------
SLAB = "slab"
PASSAGEWAY = "passageway"
LOBBY = "lobby"
SECTION = "section"
ROLES = (SLAB, PASSAGEWAY, LOBBY, SECTION)

# ---- rooms the grammar's cycle realisations and budgets make, by role ----
# (what the layout maps to a primitive, layout doc §4; it fails loudly on an
# unknown role)
STUB2 = "stub2"                # retreat loop: the second stub (two-stub, §4.1)
FLANK_STRIP = "flank_strip"    # flank pair: the narrow hull strip (§4.2, A3)
RING_FORE = "ring_fore"        # maintenance ring: the axis room fore of the hub
RING_AFT = "ring_aft"          # ... aft of the hub
RING_SIDE = "ring_side"        # ... a side room (a twin pair in the band)
POCKET = "pocket"              # pocket: a side room on a stub or a behind room
SECRET = "secret"              # the secret annex, behind its host
MADE_ROLES = (STUB2, FLANK_STRIP, RING_FORE, RING_AFT, RING_SIDE, POCKET,
              SECRET)

# ---- the skeleton's closed-form numbers (graph side) ----------------------
# A1: the engineering airlock's vestibule stands in the lobby: VESTIBULE_D + 1
# deep plus a footprint in front of the inner door.
LOBBY_X_MIN = VESTIBULE_D + 1 + FORMATION_FOOTPRINT          # 9
RING_BAND_W_MIN = 20           # §4.4: ring 6 + 1 + hub + 1 + ring 6
RING_ROOMS_MAX = 6             # §4.4: rooms > 6 are refused in v1
RING_DEPTH = 6                 # §4.4: a ring room's depth
# §4.3: the bypass's wall demand on the hub section's height.
BYPASS_HUB_H_MIN = 1 + AIRLOCK_W + 1 + 1 + DOOR_W + 1         # 13
OWN_STUB_ACROSS_MIN = AIRLOCK_W + 2                           # 7 (§4)
FLANK_STRIP_ACROSS = 3         # §4.2: the narrow leg's floor
POCKET_SIZE = ((MIN_ROOM, 8), (MIN_ROOM, 8))   # (w, h): a one-door room

# ---- anchor classes (archetype doc §3) ------------------------------------
# What a ship-level [[apply]] entry's `on` may name. `cap`: the closed-form
# realisable maximum the profiles' shifts are capped at (archetype doc §1.1);
# "departments_with_end" = one per wing department instance that has an
# `end` room, "one" = one per (quotient) ship, "slack" = no closed form here
# (bounded by the layout's packing slack -- the layout patch's feasibility).
# `exclusive`: a group of classes a ship realises at most `EXCLUSIVE_MAX`
# applications of, together (the ring and the bypass, archetype doc §3).
DEPARTMENT_ENTRY = "department-entry"
SLAB_LOBBY = "slab-lobby"
HUB = "<hub>"                  # spelled "<hub department>.<hub room id>"
LOBBY_HUB = "lobby-<hub>"      # spelled "lobby-<hub room id>"
DEPARTMENT_ROOM = "department-room"
SAME_CELL_ROOMS = "same-cell-rooms"


@dataclass(frozen=True)
class AnchorClass:
    key: str
    move: str                  # the move the class realises (recipe doc §0.1)
    candidates: str            # what it may act on, in graph terms
    realised_as: str           # the layout primitive (layout doc §4)
    cap: str                   # "departments_with_end" | "one" | "slack"
    exclusive: str | None = None


ANCHORS = {
    DEPARTMENT_ENTRY: AnchorClass(
        DEPARTMENT_ENTRY, "second_leg",
        "a wing department instance with an attach = \"end\" room",
        "two-stub (§4.1)", "departments_with_end"),
    SLAB_LOBBY: AnchorClass(
        SLAB_LOBBY, "second_leg", "the wing's slab and lobby",
        "hull strip (§4.2, §13 A3)", "one"),
    HUB: AnchorClass(
        HUB, "ring", "the hub department's hub section",
        "ring in the band (§4.4)", "one", exclusive="hub"),
    LOBBY_HUB: AnchorClass(
        LOBBY_HUB, "parallel", "the lobby's airlock edge into the hub",
        "breakable on the airlock wall (§4.3)", "one", exclusive="hub"),
    DEPARTMENT_ROOM: AnchorClass(
        DEPARTMENT_ROOM, "leaf",
        "a wing department's stub (a new side room) or a side / end room "
        "with no behind child (a new behind room); never a skeleton, ring, "
        "secret, exit or hull-door room",
        "packed room or behind room (§4.5)", "slack"),
    SAME_CELL_ROOMS: AnchorClass(
        SAME_CELL_ROOMS, "duct",
        "two rooms of one cell in different department instances, both in "
        "the quotient wing or both on the axis",
        "none (§4.6)", "slack"),
}
EXCLUSIVE_MAX = {"hub": 1}     # the ring OR the bypass, never both (v1)


@dataclass(frozen=True)
class Skeleton:
    """A skeleton's graph side. ``hub_department``: the REQUIRED axiom
    department, ``placement = "axis"``, whose entry (the hub) the lobbies'
    airlocks land on; it is the only axis department the skeleton realises.
    ``edges``: the axiom edge kinds by what they join."""
    name: str
    hub_department: str
    edges: dict

    def anchor(self, on: str, hub_entry: str) -> AnchorClass | None:
        """The anchor class an archetype's ``on`` names (None: no such
        class on this skeleton). The hub classes are spelled with the hub
        department's names: ``engineering.engine_room`` and
        ``lobby-engine_room`` on the freighter."""
        if on == f"{self.hub_department}.{hub_entry}":
            return ANCHORS[HUB]
        if on == f"{LOBBY}-{hub_entry}":
            return ANCHORS[LOBBY_HUB]
        if on in (HUB, LOBBY_HUB):
            return None
        return ANCHORS.get(on)


FREIGHTER = Skeleton(
    name="freighter",
    hub_department="engineering",
    edges={
        "slab-passageway": "door",
        "passageway-lobby": "door",
        "lobby-section": "door",       # to the aft-most MAIN section
        "section-section": "door",     # consecutive main sections
        "main-block": "bulkhead",      # the last main section to the block
        "lobby-hub": "airlock",        # a = lobby: the vestibule in the lobby
        "entry": "door",               # passageway to a wing department's stub
        "entry-own": "airlock",        # a promoted department's: a = its stub
        "exit": "door",                # slab to a first department's exit room
    })
SKELETONS = {FREIGHTER.name: FREIGHTER}


def layout_feasibility(size: str, *, graph=None, archetype=None,
                       profile: str | None = None) -> str | None:
    """THE LAYOUT FEASIBILITY CALL SITE -- None when the layout's minima
    arithmetic can pack it, else the reason. Called (1) by the grammar at the
    end of every attempt with the whole quotient ``graph`` (layout doc §13
    A5: an infeasible graph costs one grammar attempt, not ``MAX_ATTEMPTS``
    layout attempts), and (2) by the archetype loader for every allowed
    (``size``, ``profile``) with the ``archetype`` (archetype doc §6 items
    1-2: the row packing at the recipe's minima refuses, at its maxima
    warns).

    NOT BUILT YET: the packing twin (two-bin side packing, the end room
    against the row depth, the behind depths, the hull strip's row, the
    two-stub x-demand) ARRIVES WITH THE LAYOUT PATCH, which fills this body
    in. Until then it accepts everything; the grammar and its loader check
    only what is graph-side and closed-form (the ring's band width, a
    promoted stub's ``across``, two wing departments at minimum count, the
    band's section count against the block)."""
    return None
