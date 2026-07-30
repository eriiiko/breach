"""Planted charges and scheduled detonations (onephase_wego design §12).

§12 replaces the two-phase round's three detonation SLOTS (start / between /
end) with a **moment**: "Door/wall charges detonate at a player-chosen time
anywhere in the round (0–4 s)… A charge thrown/planted during planning may
detonate at t=0 of the next round — preserving the cool breach opening: door
blows at 0.0, grenades and fire follow."

That is the whole feature, and it is only expressible because the ruleset's
clock is monotonic: a detonation time is an ABSOLUTE tick, so "1.8 s into the
round" and "t=0 of the next round" are the same kind of number, and neither
needs a special case at the seam.

Two verbs, because planting and firing are genuinely different acts (§5):

- **plant_charge** is CHANNELED (``interruptible = False``) — the charge comes
  into existence when the planting FINISHES, not when it starts, so a marine
  interrupted mid-plant has planted nothing;
- **detonate** fires the planter's live charges at its scheduled tick.

A charge planted with its own ``det_tick`` is on a timer and needs no detonate
order at all; one planted without is a remote charge waiting for the button.

Determinism: integer tick comparisons, ordinal (plant-order) iteration, and the
detonation itself goes through the shipped payload EXECUTOR — the same call
``process_door_explosives`` makes — so a charge's blast is the blast it has
always been. No RNG is drawn here; the executor draws its own, in queue order.
"""
from __future__ import annotations

from simulation.payloads import execute_payload


class PlantedCharge:
    """One charge sitting on a tile, waiting for its moment."""

    __slots__ = ("x", "y", "det_tick", "ammo_name", "owner_id", "live")

    def __init__(self, x, y, ammo_name, owner_id, det_tick=None):
        self.x = int(x)
        self.y = int(y)
        self.ammo_name = str(ammo_name)
        self.owner_id = int(owner_id)
        #: Absolute tick to fire at, or ``None`` for a remote charge that waits
        #: for a Detonate order.
        self.det_tick = None if det_tick is None else int(det_tick)
        self.live = True

    def due(self, tick: int) -> bool:
        return self.live and self.det_tick is not None and tick >= self.det_tick

    def __repr__(self):
        when = "remote" if self.det_tick is None else f"t={self.det_tick}"
        return (f"PlantedCharge({self.ammo_name!r} @ ({self.x},{self.y}), "
                f"{when}, live={self.live})")


def plant(sim, unit, order) -> PlantedCharge:
    """Bring a charge into existence at the completion of a planting step.

    Called when the channeled ``plant_charge`` step RETIRES — see the module
    docstring: an interrupted plant leaves nothing behind, which is the point
    of the action being channeled in the first place.
    """
    charge = PlantedCharge(
        order.target_fx, order.target_fy,
        getattr(order, "ammo_name", None) or "demo_breach",
        getattr(unit, "id", -1),
        det_tick=getattr(order, "det_tick", None))
    sim.planted_charges.append(charge)
    return charge


def fire(sim, charge) -> None:
    """Detonate one charge through the shipped payload executor."""
    if not charge.live:
        return
    charge.live = False
    payload = sim.weapons_tables.payload_for_ammo(charge.ammo_name)
    execute_payload(sim.gmap, sim.edit_queue, sim.units, charge.y, charge.x,
                    payload, sim.rng, events=sim.tick_events,
                    kind="door_explosive")


def detonate_owned(sim, unit) -> int:
    """Fire every live charge this unit planted — the Detonate order's effect.

    Returns how many went off. Plant order, so a stack of charges blows in the
    order it was laid.
    """
    n = 0
    for charge in sim.planted_charges:
        if charge.live and charge.owner_id == int(getattr(unit, "id", -1)):
            fire(sim, charge)
            n += 1
    return n


def tick(sim) -> None:
    """Fire every charge whose scheduled moment has arrived, and forget spent
    ones.

    Runs at the head of the ruleset's unit simulation, so a charge that blows
    on tick T has already reshaped the world — blown the door, vented the room
    — before anybody moves through it that tick. That ordering is what makes
    the §12 breach opening read the way it is supposed to.
    """
    for charge in list(sim.planted_charges):
        if charge.due(sim.tick):
            fire(sim, charge)
    if sim.planted_charges:
        sim.planted_charges = [c for c in sim.planted_charges if c.live]


__all__ = ["PlantedCharge", "detonate_owned", "fire", "plant", "tick"]
