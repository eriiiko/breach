"""Door runtime + slot-9e structural sweep — Arc A patch A6, doors v0.

Design: docs/a6_doors_v0_impl_2026-07-19.md (v2). The entities package
stays import-light, so everything that touches the sim lives HERE:

- :func:`door_spans` — per-door runtime spans (ordinal order) via THE
  canonical quantization in :mod:`simulation.entities.door` (§3), with the
  S1 base-resolution recovery (``tile_size_m_base`` / ``res_factor``).
- :func:`stamp_door_tiles` — the load-order stamp (§4): called by
  ``GameMap.__init__`` between the tilemap fill and ``_update_caches`` so
  field seeding sees post-stamp solidity (authored-open ≡ authored-air is
  FIELD-identity by construction; no gas exists yet, so no seal call).
- :class:`DoorRuntime` — the sim-side runtime object (§6.1): wraps the
  parsed ``EntityInstance``, exposes the serializer duck-type
  (ordinal/id/class_name/fields + alive) and the runtime attrs
  (state / want_open / hp / span).
- :func:`sweep_doors` — the slot-9e door sweep (§5): external-destruction
  reconciliation (whole-door rule, §8), then the want_open latch
  (retry-until-clear close, ruling 5) in ordinal order.

Determinism (§12): integer arithmetic only, pinned iteration orders
(ordinal doors; row-major spans — the S4 hp_i↔tile pin), the latch is
synced state read at one slot, no RNG, no dict-order dependence.
"""
from __future__ import annotations

from simulation.entities import door as door_schema
from simulation.events import DoorDestroyedEvent
from simulation.materials import MAT_AIR, MAT_DOOR, MAT_DOOR_CLOSED
from simulation import wall_fixed

DOOR_CLOSED = door_schema.DOOR_CLOSED
DOOR_OPEN = door_schema.DOOR_OPEN
DOOR_DESTROYED = door_schema.DOOR_DESTROYED

# CSV materials a door span may sit on (§4.2): the editor bakes a door
# stamp into the grid; hand-authored files may leave air. A door through
# hull/steel/glass is an authoring bug.
_SPAN_OK_MATERIALS = (MAT_AIR, MAT_DOOR, MAT_DOOR_CLOSED)


def _base_tile_size(level_data) -> float:
    """The S1 recovery (design §3): ``tile_size_m_base`` when set (a --res
    run divided the live ``tile_size_m`` before GameMap ever saw it),
    else the unscaled ``tile_size_m``."""
    base = getattr(level_data, "tile_size_m_base", None)
    return float(base) if base is not None else float(level_data.tile_size_m)


def door_instances(level_data) -> list:
    """The level's door ``EntityInstance``s in ordinal order (§3a rule)."""
    ents = getattr(level_data, "entities", None) or []
    doors = [e for e in ents if e.class_name == "door"]
    doors.sort(key=lambda e: int(e.ordinal))
    return doors


def door_spans(level_data) -> list:
    """``[(instance, runtime_span), ...]`` in ordinal order — THE span rule
    (design §3): quantize at BASE resolution, replicate by ``res_factor``,
    row-major sort (the S4 pin)."""
    ts = _base_tile_size(level_data)
    rf = int(getattr(level_data, "res_factor", 1) or 1)
    out = []
    for inst in door_instances(level_data):
        span = door_schema.runtime_span(
            inst.fields, ts, rf, context=f"door entity '{inst.id}'")
        out.append((inst, span))
    return out


def stamp_door_tiles(material, is_vacuum, level_data) -> None:
    """Load-order stamp (design §4.1) — mutates ``material`` IN PLACE.

    Called by ``GameMap.__init__`` between the tilemap fill and
    ``_update_caches()``: per door in ordinal order, validate (§4.2 — hard
    path-named ValueErrors) then stamp MAT_DOOR_CLOSED (closed) or MAT_AIR
    (open). Field seeding then runs against the POST-stamp solidity, so
    conservation at t=0 is trivially exact (no seal call — no gas exists).
    """
    h, w = material.shape
    seen: dict = {}                       # tile -> door id (overlap check)
    lvl = getattr(level_data, "name", "?")
    for inst, span in door_spans(level_data):
        ctx = f"door entity '{inst.id}' (level '{lvl}')"
        for t in span:
            fy, fx = t
            if not (0 <= fy < h and 0 <= fx < w):
                raise ValueError(
                    f"{ctx}: span tile ({fy}, {fx}) out of bounds for the "
                    f"{h}x{w} grid")
            if t in seen:
                raise ValueError(
                    f"{ctx}: span tile ({fy}, {fx}) overlaps door entity "
                    f"'{seen[t]}' — door spans must be disjoint (a6 design "
                    f"§4.2)")
            if is_vacuum[fy, fx]:
                raise ValueError(
                    f"{ctx}: span tile ({fy}, {fx}) is vacuum — a door on "
                    f"the hull ring is an authoring error (a6 design §4.2)")
            if int(material[fy, fx]) not in _SPAN_OK_MATERIALS:
                raise ValueError(
                    f"{ctx}: span tile ({fy}, {fx}) has CSV material "
                    f"{int(material[fy, fx])} — a door span may only cover "
                    f"air or a door stamp "
                    f"{tuple(int(m) for m in _SPAN_OK_MATERIALS)} (a6 "
                    f"design §4.2)")
            seen[t] = inst.id
        stamp = (MAT_DOOR_CLOSED
                 if inst.fields["initial_state"] == "closed" else MAT_AIR)
        for fy, fx in span:
            material[fy, fx] = stamp


class DoorRuntime:
    """Sim-side runtime object for one door (design §6.1).

    Wraps the parsed ``EntityInstance`` (never mutated — the shared
    ``LevelData`` carries no runtime state) and exposes the serializer
    duck-type: ``ordinal`` / ``id`` / ``class_name`` / ``fields``
    delegated, plus ``alive`` and the runtime attrs ``state`` /
    ``want_open`` / ``hp`` (list, §7) / ``span`` (runtime tile list, §3 —
    row-major; ``hp[i]`` belongs to ``span[i]``, the S4 pin).
    """

    def __init__(self, inst, span, hp_full_q16: int):
        self.inst = inst
        self.span = list(span)
        self.alive = True
        if inst.fields["initial_state"] == "open":
            self.state = DOOR_OPEN
            self.want_open = True     # latch agrees with state (§6.1)
        else:
            self.state = DOOR_CLOSED
            self.want_open = False
        # §6.1: hp_i = the quantized table HP per runtime span tile — for
        # closed doors this equals the freshly stamped wall_hp; for open
        # doors it is the full panel HP the first close will stamp.
        self.hp = [int(hp_full_q16)] * len(self.span)

    # --- serializer duck-type (serialize.py:121-132) -------------------
    @property
    def ordinal(self):
        return self.inst.ordinal

    @property
    def id(self):
        return self.inst.id

    @property
    def class_name(self):
        return self.inst.class_name

    @property
    def fields(self):
        return self.inst.fields

    def contains(self, fy: int, fx: int) -> bool:
        """Span membership — geometry, not material (an OPEN door's span
        still matches; the §10 hotkey hit test)."""
        return (fy, fx) in self._span_set

    @property
    def _span_set(self):
        s = getattr(self, "_span_set_cache", None)
        if s is None:
            s = self._span_set_cache = set(self.span)
        return s


def build_runtime_entities(level_data, gmap):
    """§6.1: the sim's runtime entity list + doors sublist.

    Returns ``(entities, doors)``: the level's instances with door entries
    replaced by their :class:`DoorRuntime` wrappers (ordinal order
    preserved — the list is the level list, positionally patched), and the
    doors sublist in ordinal order.
    """
    spans = {id(inst): span for inst, span in door_spans(level_data)}
    hp_full = int(wall_fixed.quantize_scalar(
        float(gmap.materials.hp[MAT_DOOR_CLOSED])))
    entities = []
    doors = []
    for inst in (getattr(level_data, "entities", None) or []):
        if inst.class_name == "door":
            rt = DoorRuntime(inst, spans[id(inst)], hp_full)
            entities.append(rt)
            doors.append(rt)
        else:
            entities.append(inst)
    doors.sort(key=lambda d: int(d.ordinal))
    return entities, doors


def _occupancy_clear(units, span_set) -> bool:
    """§5.2: no LIVING unit footprint tile on the span. Note the pinned
    (col,row)→(row,col) flip: ``occupied_tiles()`` yields (tx, ty)."""
    for u in units:
        if not u.alive:
            continue                     # corpses never block (§15.2)
        for (tx, ty) in u.occupied_tiles():
            if (ty, tx) in span_set:
                return False
    return True


def sweep_doors(sim) -> None:
    """The slot-9e door sweep (design §5) — per door in ordinal order:
    reconcile external destruction (§8, whole-door rule), then apply the
    ``want_open`` latch (§6.2, retry-until-clear)."""
    gmap = sim.gmap
    material = gmap.material
    for d in sim._doors:
        if not d.alive:
            continue                     # destroyed: latch is dead (§6.2)

        # 1. External destruction (§8): a CLOSED door observes its grid —
        # any span tile no longer MAT_DOOR_CLOSED was destroyed by slots
        # 8/9/9b (or blast/chew). Whole-door rule: the assembly dies; every
        # REMAINING intact tile is destroyed via the minting destroy_wall
        # (row-major span order), each emitting DoorDestroyedEvent — the
        # S3 event contract's 9e half. NOT subject to the 9b per-tick cap
        # (assembly-completion of an already-dying door, §8).
        if d.state == DOOR_CLOSED:
            destroyed_any = any(
                int(material[fy, fx]) != MAT_DOOR_CLOSED for fy, fx in d.span)
            if destroyed_any:
                d.alive = False
                d.state = DOOR_DESTROYED
                d.hp = [0] * len(d.hp)
                for fy, fx in d.span:    # row-major (span is sorted)
                    if int(material[fy, fx]) == MAT_DOOR_CLOSED:
                        gmap.destroy_wall(fy, fx)
                        sim.tick_events.append(
                            DoorDestroyedEvent(pos=(fy, fx)))
                continue

        # 2. The latch (§6.2). Blocked closes do NOT consume the latch —
        # the sweep retries next tick (ruling 5's while-held mirror).
        if d.state == DOOR_CLOSED and d.want_open:
            # Open: fold per-tile HP (§7 — the panel remembers damage),
            # then unseal. Unconditional: a primitive raise here is a bug
            # (rider 3), never "door stays shut".
            d.hp = [int(gmap.wall_hp[fy, fx]) for fy, fx in d.span]
            gmap.unseal_tiles(d.span)
            d.state = DOOR_OPEN
        elif d.state == DOOR_OPEN and not d.want_open:
            # Close attempt: rider 3's exact composition.
            if _occupancy_clear(sim.units, d._span_set) \
                    and gmap.can_seal_tiles(d.span):
                gmap.seal_tiles(d.span, MAT_DOOR_CLOSED)
                # §7 restamp: undo on_tile_changed's table re-quantize
                # inside the seal — the door's OWN hp, tile by tile (S4).
                for i, (fy, fx) in enumerate(d.span):
                    gmap.wall_hp[fy, fx] = d.hp[i]
                d.state = DOOR_CLOSED
            # else: do nothing — latch retained, retry next tick.
