"""The DamagePacket pipeline (mechanics/06 §2–§3) — P2 wiring gate.

P2's hard contract is BEHAVIOUR PRESERVATION: with the neutral default
tables, routing the four shipped damage sites (heat, blast, bullet, melee)
through ``damage.apply_packet`` must be BIT-IDENTICAL to the inline chains it
replaced — the lockstep digest hashes the applied deltas, the emitted event
stream (raw floats via repr), and hp. These tests prove the identities
bitwise instead of assuming them:

  - mitigation order: flat armor FIRST, then multiplier (§3 — DECIDED), with
    the damage floor at 0 and the AP-vs-armor floor at 0;
  - neutral default = IEEE-exact no-op (``mitigate(x) == x`` to the BIT for
    representative amounts, int and float);
  - the zombie ×4: ``resist_mult[HEAT] = 4.0`` reproduces the dissolved
    ``if u.is_zombie: dmg *= CFG.zombie.fire_damage_multiplier`` branch
    bit-for-bit on a value sweep (×4.0 is an exact binary scale, applied at
    the same pre-quantize position);
  - apply_packet's event shape == the pre-P2 inline emission (same source
    strings, same applied damage values, same order, same kill semantics);
  - all v1 + reserved damage types exist and the tables cover them.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_damage_pipeline.py -q
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from config import CFG  # noqa: E402
from simulation import damage, unit_fixed  # noqa: E402
from simulation.damage import (  # noqa: E402
    KINETIC, BLAST, HEAT, ENERGY, POISON, ASPHYX, HEAL, ELECTRIC, PSY,
    N_DAMAGE_TYPES, DAMAGE_TYPE_NAMES,
    DamagePacket, MitigationProfile, NEUTRAL_MITIGATION,
    apply_packet, build_mitigation, mitigate, mitigation_for,
)
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402
from simulation.unit import Unit  # noqa: E402


def _bits(x) -> bytes:
    """The exact float64 bit pattern — bitwise comparison, not tolerance."""
    return struct.pack("<d", float(x))


# Representative pre-mitigation amounts: exact dyadics, snap-scale tiny,
# "ugly" decimals, thirds (full-mantissa), int damages (blast/bullet/melee
# shapes), and the actual heat-model output scale.
AMOUNT_SWEEP = [
    0.0, 1.0 / 131072.0, 7.6e-6, 0.1, 1.0 / 3.0, 0.30000000000000004,
    0.5, 1.0, 2.5, 3.14159265358979, 10, 25, 150, 123.456, 1234.5678,
    41.666666666666664, 1e6,
]


# ---------------------------------------------------------------------------
# Type vocabulary
# ---------------------------------------------------------------------------
def test_damage_types_exist_distinct_and_named():
    """All v1 types + the reserved ELECTRIC/PSY exist, are distinct ints in
    table order, and every one has a name + a slot in the tables."""
    types = (KINETIC, BLAST, HEAT, ENERGY, POISON, ASPHYX, HEAL,
             ELECTRIC, PSY)
    assert len(set(types)) == len(types) == N_DAMAGE_TYPES
    assert sorted(types) == list(range(N_DAMAGE_TYPES))
    assert set(DAMAGE_TYPE_NAMES) == set(types)
    assert len(NEUTRAL_MITIGATION.armor) == N_DAMAGE_TYPES
    assert len(NEUTRAL_MITIGATION.resist_mult) == N_DAMAGE_TYPES


def test_reserved_types_are_neutral_by_default():
    """ELECTRIC / PSY are defined-but-unused: the neutral tables carry them,
    and mitigation through them is the same no-op as every other type."""
    for dtype in (ELECTRIC, PSY):
        assert NEUTRAL_MITIGATION.armor[dtype] == 0.0
        assert NEUTRAL_MITIGATION.resist_mult[dtype] == 1.0
        assert _bits(mitigate(17.25, dtype, NEUTRAL_MITIGATION)) == _bits(17.25)


# ---------------------------------------------------------------------------
# Mitigation math (§3): flat-then-mult, floors, AP
# ---------------------------------------------------------------------------
def test_flat_armor_subtracts_before_multiplier():
    """(20 − 5) × 0.5 = 7.5 — NOT 20 × 0.5 − 5 = 5. Order is DECIDED."""
    p = build_mitigation(armor={KINETIC: 5.0}, resist_mult={KINETIC: 0.5})
    assert mitigate(20.0, KINETIC, p) == 7.5


def test_damage_floor_zero_when_armor_exceeds_amount():
    """Small arms chip harmlessly off heavy plate: floor 0 BEFORE the
    multiplier, and a vulnerability multiplier cannot resurrect it."""
    p = build_mitigation(armor={KINETIC: 5.0})
    assert mitigate(3.0, KINETIC, p) == 0.0
    p_vuln = build_mitigation(armor={KINETIC: 5.0}, resist_mult={KINETIC: 4.0})
    assert mitigate(3.0, KINETIC, p_vuln) == 0.0


def test_ap_reduces_flat_armor_but_never_below_zero():
    p = build_mitigation(armor={KINETIC: 5.0}, resist_mult={KINETIC: 0.5})
    # ap 3: effective armor 2 -> (20 - 2) * 0.5
    assert mitigate(20.0, KINETIC, p, ap=3) == 9.0
    # ap == armor: armor fully cancelled
    assert mitigate(20.0, KINETIC, p, ap=5) == 10.0
    # ap > armor: floored at 0 — AP can cancel armor, never grant bonus damage
    assert mitigate(20.0, KINETIC, p, ap=50) == 10.0


def test_armor_applies_per_type_only():
    """armor[KINETIC] must not touch a HEAT packet (per-dtype tables)."""
    p = build_mitigation(armor={KINETIC: 5.0})
    assert _bits(mitigate(20.0, HEAT, p)) == _bits(20.0)


def test_heal_is_unresisted_identity():
    """HEAL (negative-direction) bypasses mitigation in v1: the damage floor
    must not zero a heal, and armor never blocks one."""
    p = build_mitigation(armor={HEAL: 50.0}, resist_mult={HEAL: 0.25})
    assert mitigate(-5.0, HEAL, p) == -5.0
    assert mitigate(-5.0, HEAL, NEUTRAL_MITIGATION) == -5.0


def test_authored_values_snap_to_q16(monkeypatch):
    """build_mitigation Q16.16-snaps authored values (door 2): every table
    entry is an exact dyadic n/65536."""
    p = build_mitigation(armor={KINETIC: 1.23}, resist_mult={HEAT: 0.1})
    for table in (p.armor, p.resist_mult):
        for v in table:
            assert v == round(v * 65536.0) / 65536.0
    # And the snap is round-half-away-from-zero on the Q16.16 grid.
    assert p.resist_mult[HEAT] == unit_fixed.dequantize_scalar(
        unit_fixed.quantize_scalar(0.1))


# ---------------------------------------------------------------------------
# Neutral default = bitwise no-op (the P2 behaviour-preservation crux)
# ---------------------------------------------------------------------------
def test_neutral_default_is_bitwise_noop_for_a_marine():
    """For a real marine Unit, mitigate(x) == x to the BIT across the sweep
    and across every non-HEAL type: x − 0 and x × 1.0 are IEEE-exact
    identity ops, so the pipeline cannot move any shipped damage value."""
    m = Unit("M1", x=5, y=5, team=0)
    prof = mitigation_for(m)
    for dtype in range(N_DAMAGE_TYPES):
        if dtype == HEAL:
            continue
        for x in AMOUNT_SWEEP:
            out = mitigate(x, dtype, prof)
            assert _bits(out) == _bits(x), (
                f"neutral mitigation moved {x!r} (dtype {dtype}) -> {out!r}")


def test_neutral_noop_survives_the_quantize_boundary():
    """The full P2 chain shape (mitigate -> quantize_hp_delta) equals the old
    chain (quantize_hp_delta alone) bit-for-bit — int AND float amounts."""
    m = Unit("M1", x=5, y=5, team=0)
    prof = mitigation_for(m)
    for x in AMOUNT_SWEEP:
        new = unit_fixed.quantize_hp_delta(mitigate(x, KINETIC, prof))
        old = unit_fixed.quantize_hp_delta(x)
        assert _bits(new) == _bits(old)


def test_mitigation_for_fallbacks():
    """Bare objects (test stubs) resolve to the neutral profile — the same
    getattr tolerance the shipped damage responses use."""
    class Stub:
        pass
    assert mitigation_for(Stub()) is NEUTRAL_MITIGATION
    m = Unit("M1", x=5, y=5, team=0)
    prof = mitigation_for(m)
    assert tuple(prof.armor) == tuple(NEUTRAL_MITIGATION.armor)
    assert tuple(prof.resist_mult) == tuple(NEUTRAL_MITIGATION.resist_mult)


# ---------------------------------------------------------------------------
# apply_packet: event shape == the pre-P2 inline emission
# ---------------------------------------------------------------------------
def test_apply_packet_hit_event_matches_pre_p2_shape():
    """Non-lethal HEAT packet on a marine: hp drops by EXACTLY
    quantize_hp_delta(amount) (the old chain), one UnitHitEvent carrying the
    APPLIED value with source='heat', no kill event, life untouched."""
    m = Unit("M1", x=5, y=5, team=0)
    m.current_hp = 100.0
    events = []
    amount = 0.37   # heat-model-ish float
    applied = apply_packet(m, DamagePacket(amount, HEAT, None),
                           events, source="heat")

    expected = unit_fixed.quantize_hp_delta(amount)   # the pre-P2 chain
    assert _bits(applied) == _bits(expected)
    assert _bits(100.0 - m.current_hp) == _bits(expected)
    assert len(events) == 1
    ev = events[0]
    assert type(ev) is UnitHitEvent
    assert (ev.unit_id, ev.source) == (-1, "heat")
    assert _bits(ev.damage) == _bits(expected)
    assert m.alive


def test_apply_packet_int_amount_passes_through_exactly():
    """Blast/bullet/melee-shaped INT damages pass the pipeline exactly
    unchanged (the quantize twin is exact on integers), and the event carries
    the same float the old inline quantize produced."""
    m = Unit("M1", x=5, y=5, team=0)
    m.current_hp = 500.0
    events = []
    apply_packet(m, DamagePacket(150, BLAST, None), events, source="explosion")
    assert _bits(500.0 - m.current_hp) == _bits(unit_fixed.quantize_hp_delta(150))
    assert _bits(events[0].damage) == _bits(150.0)
    assert events[0].source == "explosion"


def test_apply_packet_kill_semantics_and_emission_order():
    """A lethal packet: hit THEN kill in emission order, alive=False,
    killed_by == source, and killed_by_zombie NOT set (bullet/heat/blast
    deaths never convert)."""
    m = Unit("M1", x=5, y=5, team=0)
    m.current_hp = 0.5
    events = []
    apply_packet(m, DamagePacket(10, KINETIC, 3), events, source="bullet")

    assert not m.alive
    assert m.killed_by_zombie is False
    assert [type(e) for e in events] == [UnitHitEvent, UnitKilledEvent]
    hit, kill = events
    assert (hit.unit_id, hit.source) == (-1, "bullet")
    assert (kill.unit_id, kill.killed_by) == (-1, "bullet")


def test_apply_packet_melee_shape_no_events_and_conversion_flag():
    """The melee site's shipped shape: NO events list -> nothing emitted;
    a kill sets killed_by_zombie=True (the only converting death)."""
    m = Unit("M1", x=5, y=5, team=0)
    m.current_hp = 1.0
    applied = apply_packet(
        m, DamagePacket(CFG.zombie.melee_damage, KINETIC, 7),
        events=None, source="melee", mark_killed_by_zombie=True)

    assert _bits(applied) == _bits(
        unit_fixed.quantize_hp_delta(CFG.zombie.melee_damage))
    assert not m.alive
    assert m.killed_by_zombie is True


def test_apply_packet_non_lethal_never_touches_conversion_flag():
    m = Unit("M1", x=5, y=5, team=0)
    m.current_hp = 1000.0
    apply_packet(m, DamagePacket(1, KINETIC, 7), events=None,
                 source="melee", mark_killed_by_zombie=True)
    assert m.alive
    assert m.killed_by_zombie is False


def test_apply_packet_uses_unit_id_when_present():
    m = Unit("M1", x=5, y=5, team=0)
    m.id = 42
    m.current_hp = 100.0
    events = []
    apply_packet(m, DamagePacket(1.0, HEAT, None), events, source="heat")
    assert events[0].unit_id == 42
