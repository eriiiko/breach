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

from simulation.events import ExplosionEvent
from simulation.exchange import apply_blast_damage
from simulation.field_edit import FieldEdit, EditMode, Region, Falloff
from simulation.physics import apply_explosion, add_explosion_smoke

# source_id namespace for payload-issued edits (engine/13 stable-sort key).
# physics.py owns 1 (_SRC_EXPLOSION) and 2 (_SRC_EXPLOSION_SMOKE); the W3
# payload effects continue the sequence so each emitter's edits stay grouped
# and ordered in the flush independently of any other emitter. combat.py's
# spray owns 5 (heat) and 6 (gas); the W6 plasma heat splash continues at 7.
_SRC_PAYLOAD_GAS = 3
_SRC_PAYLOAD_IGNITE = 4
_SRC_PAYLOAD_HEAT = 7


def emit_gas(gmap, queue, fy, fx, gas_species, gas_amount, gas_radius):
    """Enqueue a gas-cloud deposit into the ``gmap.gas`` slice for
    ``gas_species`` (mechanics/03 §4 gas payload columns; engine/05 §6.2).

    ONE deterministic DISC ADD FieldEdit: per tile,
    ``density += gas_amount × (1 − dist/gas_radius)`` — radial linear
    falloff, **NO RNG** (deliberately unlike ``add_explosion_smoke``'s noisy
    deposit: a flat deterministic cloud; per-tile texture can come later as a
    dial). The slice is int32 Q16.16 (S2b): the edit is authored in real
    density and the FieldEdit "gas" combine quantizes ONCE at the write
    boundary (round-half-away — door 2), with the [0, 1] policy clamp as the
    saturation guard and the solid skip-mask (gas does not enter walls).
    Traversal is the flush's fixed row-major region order. A ``gas_amount``
    above 1.0 (e.g. smoke_screen's 1.5) saturates the cloud's core to full
    density and feathers the edge — the clamp makes that authoring shape
    safe.

    ``gas_species`` resolves BY NAME through the map's gas table
    (``gmap.gases.name_to_id`` — gases.py is the single source of truth;
    never hardcode a slice index). Unknown names fail LOUDLY here at
    detonation time; :class:`simulation.weapons.PayloadTable` already
    validates rows against the canonical name set at load, so this raise is
    the belt-and-suspenders for hand-built defs.
    """
    gas_id = int(gmap.gases.name_to_id[gas_species])
    queue.enqueue(FieldEdit(
        field="gas", region=Region.DISC, coords=(fy, fx, float(gas_radius)),
        amount=float(gas_amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        clamp=(0.0, 1.0), channel=gas_id, source_id=_SRC_PAYLOAD_GAS,
    ))


def ignite_ring(gmap, queue, fy, fx, ignite_radius, ignite_intensity):
    """Enqueue an incendiary ignition disc: ``fire = max(fire, seed)`` over
    the ring (mechanics/03 §4 ignite columns).

    ONE DISC **MAX** FieldEdit with LINEAR falloff: per flammable tile,
    ``fire = max(fire, ignite_intensity × (1 − dist/ignite_radius))`` — the
    established never-lowers integer max pattern (the same write form as
    ``apply_temperature_ignition``'s ``fire = max(fire, ignition_seed)`` and
    ``apply_explosion``'s per-tile MAX ignite edits; a MAX of two exactly-
    dequantized Q16.16 values re-quantizes to the exact larger int, so the
    FieldEdit combine IS an integer max). The fire policy supplies the
    non-flammable skip-mask and the [0, 1] clamp. No RNG.
    """
    queue.enqueue(FieldEdit(
        field="fire", region=Region.DISC, coords=(fy, fx, float(ignite_radius)),
        amount=float(ignite_intensity), mode=EditMode.MAX,
        falloff=Falloff.LINEAR, clamp=(0.0, 1.0),
        source_id=_SRC_PAYLOAD_IGNITE,
    ))


def deposit_heat(gmap, queue, fy, fx, heat_amount, heat_radius):
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
    queue.enqueue(FieldEdit(
        field="heat", region=Region.DISC, coords=(fy, fx, float(heat_radius)),
        amount=float(heat_amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        source_id=_SRC_PAYLOAD_HEAT,
    ))


def execute_payload(gmap, queue, units, fy, fx, payload, rng, events=None,
                    kind="explosion"):
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
    5. ``ignite_ring``         — when ``ignite_radius > 0`` (W3).
    5b. ``deposit_heat``       — when ``heat_amount > 0`` (W6, the plasma
        splash: a one-shot DISC heat deposit; converts to temperature the
        same tick).
    6. ``ExplosionEvent(pos=(fx, fy), radius=radius, kind=kind)`` — always
       (the detonation happened whatever the payload mix; the renderer
       ignores unknown kinds by design).

    ``rng`` is the sim generator — carried for signature symmetry with the
    detonation sites; the executor itself never draws (module docstring).
    Returns None; all field writes ride ``queue`` (engine/13) except
    ``apply_explosion``'s structural wall damage (the documented carve-out).
    """
    if payload.radius > 0:
        apply_explosion(gmap, queue, fy, fx, payload.radius,
                        payload.pressure, payload.wall_damage)
    if payload.unit_damage > 0:
        apply_blast_damage(units, fx, fy, payload.radius,
                           payload.unit_damage, events=events)
    if payload.emit_blast_smoke:
        add_explosion_smoke(gmap, queue, fy, fx, payload.radius)
    if payload.gas_species:
        emit_gas(gmap, queue, fy, fx, payload.gas_species,
                 payload.gas_amount, payload.gas_radius)
    if payload.ignite_radius > 0:
        ignite_ring(gmap, queue, fy, fx, payload.ignite_radius,
                    payload.ignite_intensity)
    if getattr(payload, "heat_amount", 0.0) > 0:
        deposit_heat(gmap, queue, fy, fx, payload.heat_amount,
                     payload.heat_radius)
    if events is not None:
        events.append(ExplosionEvent(pos=(fx, fy), radius=payload.radius,
                                     kind=kind))


__all__ = ["execute_payload", "emit_gas", "ignite_ring", "deposit_heat"]
