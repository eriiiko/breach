"""The payload EXECUTOR — mechanics/03 §4 ``[payloads.*]``, wired by W3.

A payload row is *what happens at the destination* of a delivery (grenade
fuse-out, placed-charge det slot, a 40 mm round's stop tile). This module is
the ONE place a payload row becomes world effects; every delivery archetype
calls :func:`execute_payload` instead of hand-sequencing physics calls.

The executor generalizes the shipped explosion triple **behaviour-
preservingly**: for the explosion part it sequences EXACTLY the calls the
grenade fuse-out and door-charge sites shipped —

    apply_explosion(...)      when radius > 0        (walls + wave + clear + ignite)
    apply_blast_damage(...)   when unit_damage > 0   (the wave_p blast coupling row)
    add_explosion_smoke(...)  when emit_blast_smoke  (the noisy render cloud)
    ExplosionEvent(...)       always                 (the renderer's cue)

— same argument values, same call order, same event, so ``frag_standard`` and
``breach_focus`` detonations are byte-identical to the pre-W3 inline sites
(proven by the replica gate in tests/test_payloads.py: fields + events + RNG
end-state). The smoke boolean SPLIT is the W1 finding of record
(mechanics/03 §8): ``clear_smoke`` documents the inner-radius smoke clearing
that v1 keeps INSIDE ``apply_explosion`` (data-of-record — it becomes the
live gate when FieldEdit takes over the explosion internals), while
``emit_blast_smoke`` is live TODAY and gates the textured cloud. Both must be
true on ``frag_standard`` AND ``breach_focus`` or the door charge silently
loses its smoke.

Then the NEW W3 effects:

    gas emission     when gas_species is nonempty   (:func:`emit_gas`)
    bulk gas         when bulk_gas is nonempty      (:func:`inject_bulk_gas`, #133)
    ignition ring    when ignite_radius > 0         (:func:`ignite_ring`)

Determinism (engine/14): the executor DRAWS NO RANDOMNESS. The explosion
smoke's per-tile noise is drawn at the EditQueue flush (the single RNG
consumer, unchanged from pre-W3); the gas deposit is deliberately noise-free
(a flat deterministic radial falloff — texture can come later as a dial);
the ignite ring is a pure MAX deposit. ``rng`` stays in the signature for
symmetry with the detonation sites (process_door_explosives already carries
it) and for future payload effects that may legitimately draw — today it is
untouched.
"""
from __future__ import annotations

from simulation.damage import CREDIT_HEAT, tick_credit
from simulation.events import NO_ATTACKER, ExplosionEvent
from simulation.exchange import apply_blast_damage
from simulation.field_edit import (FieldEdit, EditMode, Region, Falloff,
                                   _iter_region)
from simulation.physics import apply_explosion, add_explosion_smoke

# source_id namespace for payload-issued edits (engine/13 stable-sort key).
# physics.py owns 1 (_SRC_EXPLOSION) and 2 (_SRC_EXPLOSION_SMOKE); the W3
# payload effects continue the sequence so each emitter's edits stay grouped
# and ordered in the flush independently of any other emitter. combat.py's
# spray owns 5 (heat) and 6 (gas); the W6 plasma heat splash continues at 7.
_SRC_PAYLOAD_GAS = 3
_SRC_PAYLOAD_IGNITE = 4
_SRC_PAYLOAD_HEAT = 7


def _claim_disc(gmap, channel, fy, fx, radius, owner_id):
    """#123: claim a deposit disc's tiles for ``owner_id`` this tick
    (damage.TickCredit) — the same DISC membership the edit's flush walks.
    No-op for the environment (-1)."""
    if owner_id is None or int(owner_id) < 0:
        return
    tick_credit(gmap).claim(
        channel,
        ((r, c) for r, c, _ in _iter_region(
            Region.DISC, (fy, fx, float(radius)), Falloff.FLAT,
            gmap.material.shape)),
        owner_id)


def emit_gas(gmap, queue, fy, fx, gas_species, gas_amount, gas_radius,
             owner_id=NO_ATTACKER):
    """Enqueue a gas-cloud deposit into the ``gmap.gas`` slice for
    ``gas_species`` (mechanics/03 §4 gas payload columns; engine/05 §6.2).

    ONE deterministic DISC ADD FieldEdit: per tile,
    ``density += gas_amount × (1 − dist/gas_radius)`` — radial linear
    falloff, **NO RNG** (deliberately unlike ``add_explosion_smoke``'s noisy
    deposit: a flat deterministic cloud; per-tile texture can come later as a
    dial). The slice is int32 Q16.16 (S2b): the edit is authored in real
    density and the FieldEdit "gas" combine quantizes ONCE at the write
    boundary (round-half-away — door 2), ADDITIVE with no ceiling (the trace
    policy's floor-0 TRACE_CLAMP — smoke transport v2 D1, Erik 2026-10-04) and
    the solid skip-mask (gas does not enter walls).
    Traversal is the flush's fixed row-major region order. A ``gas_amount``
    above 1.0 (e.g. smoke_screen's 1.5) now puts exactly that density in the
    cloud's core (D1: deposits add, nothing cuts a tile back to 1).

    ``gas_species`` resolves BY NAME through the map's gas table
    (``gmap.gases.name_to_id`` — gases.py is the single source of truth;
    never hardcode a slice index). Unknown names fail LOUDLY here at
    detonation time; :class:`simulation.weapons.PayloadTable` already
    validates rows against the canonical name set at load, so this raise is
    the belt-and-suspenders for hand-built defs.
    """
    gas_id = int(gmap.gases.name_to_id[gas_species])
    _claim_disc(gmap, gas_id, fy, fx, gas_radius, owner_id)
    queue.enqueue(FieldEdit(
        field="gas", region=Region.DISC, coords=(fy, fx, float(gas_radius)),
        amount=float(gas_amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        channel=gas_id, source_id=_SRC_PAYLOAD_GAS,   # policy TRACE_CLAMP (D1)
    ))


# THE HEAT AN IGNITER DELIVERS (fire session #12, 2026-09-12, issue #65).
#
# Sized so a flammable tile reaches its own ignition point FROM AMBIENT. Every
# shipped flammable (wood, furniture, kindling, foliage, doors) carries
# `thermal_mass = 8`, so `heat_inv_shift` is 3 and the temperature solver
# converts `ΔT = heat >> 3`, i.e. heat/8. Wood has the highest ignition point at
# 300, and the dev key's decisive-igniter margin is 55 (src/debug_keys.py:
# DEBUG_IGNITE_MARGIN — measured: survival is marginal within ~+25 of ignition
# and reliable above it). So:
#
#     (300 + 55) * 2**3 = 2840
#
# Independent cross-check: the flamethrower's own tuned `heat_deposit` is 2400
# == 300 * 2**3, i.e. exactly "enough to light wood from ambient", arrived at
# from the other direction entirely (config.toml [ammo.fuel_standard]).
IGNITE_HEAT = 2840.0


def ignite_ring(gmap, queue, fy, fx, ignite_radius, ignite_intensity,
                ignite_heat=IGNITE_HEAT, owner_id=NO_ATTACKER):
    """Enqueue an incendiary ignition disc: ``fire = max(fire, seed)`` over
    the ring, **plus the heat that lighting implies** (mechanics/03 §4 ignite
    columns; heat added 2026-09-12, issue #65).

    ONE DISC **MAX** FieldEdit with LINEAR falloff: per flammable tile,
    ``fire = max(fire, ignite_intensity × (1 − dist/ignite_radius))`` — the
    established never-lowers integer max pattern (the same write form as
    ``apply_temperature_ignition``'s ``fire = max(fire, ignition_seed)`` and
    ``apply_explosion``'s per-tile MAX ignite edits; a MAX of two exactly-
    dequantized Q16.16 values re-quantizes to the exact larger int, so the
    FieldEdit combine IS an integer max). The fire policy supplies the
    non-flammable skip-mask and the [0, 1] clamp. No RNG.

    **THE HEAT IS NOT OPTIONAL, and that is the whole point of this function.**
    Ruling R3 (2026-09-01) tied the O2 demand and the fuel destruction to
    ``hotf = clamp((T - fire_T_ext)/fire_T_span, 0, hotf_cap)``, so a fire
    seeded on an AMBIENT tile draws no oxygen, burns no fuel and deposits no
    heat. Seeding ``fire`` alone therefore stopped meaning "light a fire here"
    the day R3 landed: it produced a flame that consumed nothing, spread
    nothing, and — because ``k_die``'s e-fold is 125 s — did not even go out.
    Measured on a kindling tile over 400 ticks: 0.0% of fuel consumed, versus
    110.8% once the tile is heated. Both incendiary ammo rows carry
    ``damage = 0``, so that flame WAS the entire weapon.

    Physically this is just the truth the rest of the engine already respects:
    combustion needs the fuel bed at pyrolysis temperature, and the in-engine
    ignition path only ever fires on a tile whose temperature crossed its own
    ``ignition_temp``. The flamethrower was never affected precisely because it
    delivers ``heat_deposit`` and lets ignition happen; the dev key ``I`` had to
    learn the same lesson (``debug_keys.debug_ignite``).

    The heat rides the **`heat` plane**, not a direct `temperature` write: that
    is the canonical deposit channel, the temperature solver owns the
    solid-vs-gas conversion, and it keeps the gas-mirror rule intact (nothing
    writes a gas cell's ``temperature`` directly). It uses **FLAT** falloff
    while the flame uses LINEAR — deliberately: a linearly-faded heat edge would
    light the rim of the disc with too little heat to sustain it, recreating the
    exact inert-flame bug at the ring's edge.

    ``ignite_heat`` defaults to :data:`IGNITE_HEAT`; pass 0.0 only for a
    deliberately heatless flame (nothing shipped wants one).
    """
    queue.enqueue(FieldEdit(
        field="fire", region=Region.DISC, coords=(fy, fx, float(ignite_radius)),
        amount=float(ignite_intensity), mode=EditMode.MAX,
        falloff=Falloff.LINEAR, clamp=(0.0, 1.0),
        source_id=_SRC_PAYLOAD_IGNITE,
    ))
    if ignite_heat > 0.0:
        _claim_disc(gmap, CREDIT_HEAT, fy, fx, ignite_radius, owner_id)
        queue.enqueue(FieldEdit(
            field="heat", region=Region.DISC,
            coords=(fy, fx, float(ignite_radius)),
            amount=float(ignite_heat), mode=EditMode.ADD,
            falloff=Falloff.FLAT,
            source_id=_SRC_PAYLOAD_IGNITE,
        ))


def deposit_heat(gmap, queue, fy, fx, heat_amount, heat_radius,
                 owner_id=NO_ATTACKER):
    """Enqueue a one-shot heat splash into the engine/06 ``heat`` ingress
    buffer (mechanics/03 §4 heat payload columns — W6, the plasma splash).

    ONE deterministic DISC ADD FieldEdit: per tile,
    ``heat += heat_amount × (1 − dist/heat_radius)`` — the emit_gas shape on
    the ``heat`` field. NO RNG. The heat policy has no skip-mask (heat lands
    on solids — that is how a plasma bolt chars the wall it hit) and the
    combine quantizes ONCE at the write boundary (Q16.16 saturating add,
    door 2). The C++ TemperatureSolver converts the splash to temperature
    the SAME tick (the flush runs before physics), so ignition and the
    heat|max unit-damage row both come free — the SPRAY two-terminals
    discipline (zero new damage code) applied to a detonation."""
    _claim_disc(gmap, CREDIT_HEAT, fy, fx, heat_radius, owner_id)
    queue.enqueue(FieldEdit(
        field="heat", region=Region.DISC, coords=(fy, fx, float(heat_radius)),
        amount=float(heat_amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        source_id=_SRC_PAYLOAD_HEAT,
    ))


# The bulk-gas effect's books channels (#133 E.2): ordinary named seam
# channels in GameMap.gas_energy_books (counted by gas_energy_seam_net -- the
# "seams" group of the closure identity, never a fifth group), named apart
# from the vent sweep's pump_inject_vec / pump_rail so a books reader can tell
# a payload's gas from a duct's.
BULK_GAS_BOOKS = "payload_bulk_gas"
BULK_GAS_RAIL_BOOKS = "payload_bulk_gas_rail"


def bulk_gas_tiles(gmap, fy, fx, radius):
    """The tiles a bulk-gas deposit lands on: the FieldEdit DISC membership
    (``field_edit._iter_region``, strict ``dist < radius``, row-major -- the
    same disc ``emit_gas`` walks), minus every tile the vent sweep's runtime
    aperture guard refuses (solid / thermal_solid / vacuum / ambient ring /
    flooded: ``vent_system._aperture_blocked``, the one rule for where the
    gas pump primitives may deposit)."""
    from simulation.vent_system import _aperture_blocked
    return [(r, c) for r, c, _w in _iter_region(
                Region.DISC, (fy, fx, float(radius)), Falloff.FLAT,
                gmap.material.shape)
            if not _aperture_blocked(gmap, r, c)]


def inject_bulk_gas(gmap, fy, fx, payload):
    """Inject the payload's BULK gas -- pure O2 or pure inert N2 -- around
    (fy, fx), NOW (#133 E.2, levelgen vocabulary Part E; the medbay O2 store,
    the halon dump).

    Per open tile of the ``bulk_radius`` disc (:func:`bulk_gas_tiles`), FLAT:
    one call of THE gas pump primitive :meth:`GameMap.inject_gas_n_vec` with a
    vector that is zero except the named bulk slice (``bulk_amount_q`` bulk
    counts), deposited at ``bulk_temp_q`` (relative Q16, 0 = ambient) and
    railed to ``[T_MIN, T_MAX_PHYS]`` -- the SAME rails the vent sweep passes
    (``vent_system._t_rails_q``, one source). The primitive credits the
    cell's ``gas_energy`` with ``ΔN·(t_dep + T_AMB)`` through the gas energy
    seam and books it in :data:`BULK_GAS_BOOKS` (a rail clamp in
    :data:`BULK_GAS_RAIL_BOOKS`), so the across-tick closure identity sees it
    as one more seam. No FieldEdit (the ``gas`` policy refuses a conservative
    slice), no direct ``gas[...]`` or ``temperature[...]`` write.

    Like ``apply_explosion``'s structural wall damage this write is IMMEDIATE
    rather than queued -- the pump primitives are immediate by contract (the
    vent / pump sweeps call them in slot 9e); the deposit carries its own
    rails because nothing after slot 9e re-rails it this tick (the vent
    sweep's own reasoning). Returns the number of tiles that received gas.
    No RNG.
    """
    from simulation.gases import N_GASES
    from simulation.vent_system import _t_rails_q
    g = int(gmap.gases.name_to_id[payload.bulk_gas])
    if not bool(gmap.gases.conservative[g]):        # belt-and-suspenders:
        raise ValueError(                          # the door refuses this
            f"inject_bulk_gas: {payload.bulk_gas!r} is not a bulk gas")
    add_vec = [0] * N_GASES
    add_vec[g] = int(payload.bulk_amount_q)
    if add_vec[g] <= 0:
        return 0
    t_min_q, t_max_phys_q = _t_rails_q()
    tiles = bulk_gas_tiles(gmap, fy, fx, payload.bulk_radius)
    for r, c in tiles:
        gmap.inject_gas_n_vec(r, c, add_vec, int(payload.bulk_temp_q),
                              t_min_q, t_max_phys_q,
                              channel=BULK_GAS_BOOKS,
                              rail_channel=BULK_GAS_RAIL_BOOKS)
    return len(tiles)


def blast_smoke_peak(gmap, fy, fx, radius, soot_g, noise=None):
    """The peak density of the blast-smoke disc that deposits ``soot_g`` grams
    of soot (#12 handle 3, 2026-10-04): the disc keeps its shape -- LINEAR
    falloff to ``radius``, per-tile noise uniform in [1 - noise, 1] drawn at
    the flush -- and its peak is scaled so the EXPECTED total equals the
    charge's soot in smoke units (gases.smoke_units_of_soot):

        peak = units / (sum_{disc, non-solid} (1 - d/r) * (1 - noise/2))

    The sum walks the SAME disc the FieldEdit flush walks
    (field_edit._iter_region) and skips the smoke policy's solid tiles, so
    smoke is not budgeted onto walls. 0 for no soot (an RDX/C4 charge) or a
    disc with no open tile. Nothing clamps the deposit from above (smoke
    transport v2 D1): it adds exactly its soot onto whatever the tile holds."""
    if radius <= 0 or soot_g <= 0.0:
        return 0.0
    from simulation.field_edit import _iter_region, Region, Falloff
    from simulation.gases import smoke_units_of_soot
    from config import CFG
    if noise is None:
        noise = float(getattr(CFG.physics, "explosion_smoke_noise", 0.85))
    noise = min(1.0, max(0.0, noise))
    solid = gmap.solid
    wsum = 0.0
    for r, c, wgt in _iter_region(Region.DISC, (fy, fx, float(radius)),
                                  Falloff.LINEAR, solid.shape):
        if not solid[r, c]:
            wsum += wgt
    if wsum <= 0.0:
        return 0.0
    return smoke_units_of_soot(soot_g) / (wsum * (1.0 - 0.5 * noise))


def execute_payload(gmap, queue, units, fy, fx, payload, rng, events=None,
                    kind="explosion", owner_id=NO_ATTACKER):
    """Execute one ``[payloads.*]`` row at tile (fy, fx) — the single owner
    of the payload → world-effects sequence (mechanics/03 §4, W3).

    Effect order (fixed — the first three are the shipped detonation triple,
    verbatim call order and argument values; see the module docstring):

    1. ``apply_explosion``     — when ``radius > 0`` (structural wall damage
       immediate; wave/atmosphere/smoke-clear/ignite edits enqueued;
       ``clear_smoke`` is data-of-record: the inner-radius clear lives inside
       ``apply_explosion`` in v1).
    2. ``apply_blast_damage``  — when ``unit_damage > 0`` (the wave_p blast
       coupling row at its detonation-site position; emits UnitHit/UnitKilled
       into ``events``). A row authoring ``unit_damage > 0`` with
       ``radius == 0`` is a config bug (the geometric falloff needs a
       radius); no shipped row does.
    3. ``add_explosion_smoke`` — when ``emit_blast_smoke`` (the noisy cloud;
       its per-tile noise is drawn at the queue flush, not here).
    4. ``emit_gas``            — when ``gas_species`` is nonempty (W3).
    4b. ``inject_bulk_gas``    — when ``bulk_gas`` is nonempty (#133 E.2:
        pure O2 / inert N2 through the gas pump primitive, IMMEDIATE, booked
        through the gas energy seam in ``payload_bulk_gas``).
    5. ``ignite_ring``         — when ``ignite_radius > 0`` (W3).
    5b. ``deposit_heat``       — when ``heat_amount > 0`` (W6, the plasma
        splash: a one-shot DISC heat deposit; converts to temperature the
        same tick).
    6. ``ExplosionEvent(pos=(fx, fy), radius=radius, kind=kind)`` — always
       (the detonation happened whatever the payload mix; the renderer
       ignores unknown kinds by design).

    ``rng`` is the sim generator — carried for signature symmetry with the
    detonation sites; the executor itself never draws (module docstring).

    ``owner_id`` (#123) is the unit the detonation is credited to — the
    grenade's thrower, the payload round's firer, the charge's planter; -1
    for nobody (a timed charge, a tool). It credits the blast directly and
    claims the payload's heat (``ignite_ring``'s, ``deposit_heat``'s) and gas
    discs for THIS tick (damage.TickCredit): a unit hurt by that deposit
    while it is fresh is credited to the owner; the fire it lit and the
    cloud it leaves are the environment's (the accepted gap).
    Returns None; all field writes ride ``queue`` (engine/13) except
    ``apply_explosion``'s structural wall damage (the documented carve-out).
    """
    if payload.radius > 0:
        apply_explosion(gmap, queue, fy, fx, payload.radius,
                        payload.pressure, payload.wall_damage)
    if payload.unit_damage > 0:
        apply_blast_damage(units, fx, fy, payload.radius,
                           payload.unit_damage, events=events,
                           attacker_id=owner_id)
    if payload.emit_blast_smoke:
        peak = blast_smoke_peak(gmap, fy, fx, payload.radius,
                                getattr(payload, "blast_soot_g", 0.0))
        if peak > 0.0:
            add_explosion_smoke(gmap, queue, fy, fx, payload.radius,
                                amount=peak)
    if payload.gas_species:
        emit_gas(gmap, queue, fy, fx, payload.gas_species,
                 payload.gas_amount, payload.gas_radius, owner_id=owner_id)
    if getattr(payload, "bulk_gas", ""):
        inject_bulk_gas(gmap, fy, fx, payload)
    if payload.ignite_radius > 0:
        ignite_ring(gmap, queue, fy, fx, payload.ignite_radius,
                    payload.ignite_intensity, owner_id=owner_id)
    if getattr(payload, "heat_amount", 0.0) > 0:
        deposit_heat(gmap, queue, fy, fx, payload.heat_amount,
                     payload.heat_radius, owner_id=owner_id)
    if events is not None:
        events.append(ExplosionEvent(pos=(fx, fy), radius=payload.radius,
                                     kind=kind))


__all__ = ["execute_payload", "emit_gas", "ignite_ring", "deposit_heat",
           "inject_bulk_gas", "bulk_gas_tiles", "blast_smoke_peak",
           "BULK_GAS_BOOKS", "BULK_GAS_RAIL_BOOKS"]
