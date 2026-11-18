"""The body temperature (#215; ``docs/body_temperature_design_2026-11-17.md``
section 2): an energy balance with a heat capacity, written by the heat row
alone (``src/simulation/body_temp.py`` through ``exchange.apply_environmental_
damage``), the band read on the body. Each test names the property it protects
and the change that must break it. The steady state over the per-tick law's
fixtures is also held by tests/test_unit_heat_damage.py.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from simulation import body_temp, unit_fixed, unit_schema  # noqa: E402
from simulation.exchange import (HEAT_SCALE, apply_environmental_damage,  # noqa: E402
                                 felt_temperature_q, heat_dials, heat_dials_q)
from simulation.unit import Unit  # noqa: E402

ONE = unit_fixed.FP_ONE
TPS = int(CFG.clock.ticks_per_second)
AMB_Q = body_temp.ambient_q()
#: irradiance per injected phi (tests/test_unit_heat_damage.py's derivation)
KW_M2_PER_PHI = 224.775


class _Stub:
    """A map with one-atmosphere ambient air and the 0.333 m tile."""
    def __init__(self, h=30, w=30, p_atm=1.0, t_game=0.0):
        self.heat = np.zeros((h, w), dtype=np.int32)
        self.atmosphere = np.full((h, w), unit_fixed.quantize_scalar(p_atm),
                                  dtype=np.int32)
        self.temperature = np.full((h, w), unit_fixed.quantize_scalar(t_game),
                                   dtype=np.int32)
        self.solid = np.zeros((h, w), dtype=bool)
        self.tile_size_m = 0.333


def _marine(x=10, y=10):
    u = Unit("M", x=x, y=y, team=0)
    u.set_pos(float(x), float(y))
    u.current_hp = 1e9
    return u


def _flux(stub, u, phi):
    raw = int(round(phi * HEAT_SCALE))
    for (tx, ty) in u.occupied_tiles():
        stub.heat[ty, tx] = raw


def _old_dps(phi):
    """The pre-#215 instantaneous law's HP per second at a held phi (the
    float model of tests/test_unit_heat_damage.py)."""
    cmb = CFG.combat
    t = float(cmb.heat_ambient_ref) + float(cmb.heat_flux_to_temp) * phi \
        * float(cmb.unit_absorption) * (1.0 - float(cmb.unit_reflectivity))
    over = t - 60.0
    return 0.0 if over <= 0 else 1.0 + float(cmb.heat_overtemp_scale) * over


def _run(units, stub, ticks):
    for _ in range(ticks):
        apply_environmental_damage(units, stub, ticks_per_second=TPS)


def _rates(stub):
    return body_temp.rates_q(stub, 320.0)


#: The shipped protection (every profile's heat_absorption).
SHIPPED_H_Q = body_temp.absorption_q(None)


@pytest.fixture
def unprotected(monkeypatch):
    """The UNSCALED law: every body takes up all the heat reaching it
    (heat_absorption 1.0) -- the literature table's anchor and the energy
    properties below are the law's at absorption 1; the shipped protection
    scales them (test_the_protection_scales_both_inflows_alike)."""
    monkeypatch.setattr(body_temp, "absorption_q", lambda env: ONE)


def _with_h(monkeypatch, h):
    q = unit_fixed.quantize_scalar(h)
    monkeypatch.setattr(body_temp, "absorption_q", lambda env, _q=q: _q)


# ---------------------------------------------------------------------------
# The steady state is today's law
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw_m2", (1.0, 2.5, 5.0, 10.0, 20.0, 50.0))
def test_the_steady_state_is_the_literature_table(kw_m2, unprotected):
    """Property: under a constant absorbed flux in air at one atmosphere and
    20 C the body settles at today's felt temperature and the damage per
    second equals today's, for every row of [combat]'s literature table (the
    anchor: 2.5 kW/m2 sits on the band's edge) -- the UNSCALED law, at
    heat_absorption 1. Breaks if the loss stops being
    derived from heat_flux_to_temp, the air and radiated shares stop summing to
    it, or the band reads anything but the body."""
    stub = _Stub()
    u = _marine()
    phi = kw_m2 / KW_M2_PER_PHI
    _flux(stub, u, phi)
    _run([u], stub, 40 * TPS)
    felt_q, _, _ = felt_temperature_q(stub, u, heat_dials_q(heat_dials(CFG)))
    assert abs(u.body_temp_q - felt_q) <= 32, (u.body_temp_q, felt_q)
    hp = u.current_hp
    _run([u], stub, TPS)
    dps = hp - u.current_hp
    assert dps == pytest.approx(_old_dps(round(phi * HEAT_SCALE) / HEAT_SCALE),
                                rel=1e-4, abs=5e-4)


def test_the_loss_is_derived_from_the_anchor():
    """Property: the derived loss is 2.5 kW/m2 on the 3.33 m2 shell, absorbed
    at 0.765, over the 40 C band (159 W/K at 0.333 m tiles), and 320 J/K over
    it is the page's 2 s time constant. Breaks if J_per_heat_unit, the tile
    volume or heat_flux_to_temp drift apart."""
    stub = _Stub()
    loss = body_temp.loss_w_per_k(stub, TPS)
    assert loss == pytest.approx(2500 * 4 * 0.333 * 2.5 * 0.765 / 40, rel=2e-3)
    a_q, a_rad_q = _rates(stub)
    tau_s = ONE / (a_q * TPS)
    assert tau_s == pytest.approx(320.0 / loss, rel=2e-3)
    assert a_rad_q == unit_fixed.mul_q16_round(
        a_q, unit_fixed.quantize_scalar(float(CFG.combat.body_radiated_share)))


# ---------------------------------------------------------------------------
# A burst: energy, no ceiling
# ---------------------------------------------------------------------------
def _burst(joules):
    """One tick of a deposit that the body absorbs as ``joules``, then 30 s of
    nothing: (the jump on the burst's tick in degrees, the total damage, the
    unit, the damage on the ticks AFTER the burst, with no flux at all)."""
    stub = _Stub()
    u = _marine()
    j_hu = float(body_temp.j_per_heat_unit(stub))
    absorbed_per_phi = j_hu * float(CFG.combat.unit_absorption) \
        * (1.0 - float(CFG.combat.unit_reflectivity))
    _flux(stub, u, joules / absorbed_per_phi)
    hp0 = u.current_hp
    _run([u], stub, 1)
    jump = (u.body_temp_q - AMB_Q) / ONE
    hp1 = u.current_hp
    stub.heat[...] = 0
    _run([u], stub, 30 * TPS)
    return jump, hp0 - u.current_hp, u, hp1 - u.current_hp


def test_a_burst_moves_the_body_by_joules_over_capacity(unprotected):
    """Property: a one-tick burst moves the body by its absorbed joules over
    the 320 J/K capacity, and the harm comes after it, over the cooling, the
    body back at ambient. Breaks if the body is billed the burst's felt
    temperature (the pre-#215 law: 576 HP beside a frag) or the capacity is
    not the profile's."""
    jump, dmg, u, after = _burst(30_000.0)
    assert jump == pytest.approx(30_000.0 / 320.0, rel=2e-3)
    assert abs(u.body_temp_q - AMB_Q) <= ONE // 100
    assert dmg > 0
    assert after > 0.5 * dmg, "the harm did not come over the cooling"


def test_more_joules_more_harm_with_no_ceiling(unprotected):
    """Property: 3 MJ moves the body ten times as far as 300 kJ (in
    proportion: the law is linear) and harms more -- nothing clips what a
    body absorbs (Erik: plasma guns must pay off). Breaks if a ceiling on the
    felt or the body temperature comes back."""
    j10, d10, _, _ = _burst(300_000.0)
    j100, d100, _, _ = _burst(3_000_000.0)
    assert j100 == pytest.approx(10.0 * j10, rel=1e-3)
    assert d100 > 5.0 * d10


# ---------------------------------------------------------------------------
# The air and the vacuum
# ---------------------------------------------------------------------------
def test_at_one_atmosphere_and_ambient_air_the_split_is_the_unsplit_loss():
    """Property: at one atmosphere in 20 C air the air part and the radiated
    part sum to the whole loss: every step equals the unsplit ``body + round(a
    x (felt - body))``. Breaks if the shares do not sum to ``a`` or the air is
    read against anything but its own temperature."""
    a_q, a_rad_q = _rates(_Stub())
    for body in (AMB_Q, AMB_Q + 7 * ONE, AMB_Q + 913 * ONE + 5, AMB_Q - 3 * ONE):
        for felt in (AMB_Q, AMB_Q + 500 * ONE + 11, AMB_Q + 40 * ONE):
            got = body_temp.step_q(body, felt, ONE, AMB_Q, a_q, a_rad_q, AMB_Q)
            assert got == body + unit_fixed.div_round(a_q * (felt - body), ONE)


def _cool_time(p_atm):  # cooling: no inflow, the absorption plays no part
    """Seconds a body at 220 C takes to fall to 120 C with no flux, in air of
    ``p_atm`` at ambient."""
    stub = _Stub(p_atm=p_atm)
    u = _marine()
    u.body_temp_q = AMB_Q + 200 * ONE
    t = 0
    while u.body_temp_q > AMB_Q + 100 * ONE:
        _run([u], stub, 1)
        t += 1
        assert t < 600 * TPS
    return t / TPS


def test_vacuum_cools_by_the_radiated_share_only():
    """Property: in vacuum only the radiated part cools: a hot suit takes
    1/share as long (about seven times) to cool as in air. Breaks if the air
    part ignores the pressure or the radiated part is dropped (a vacuum body
    would never cool)."""
    share = float(CFG.combat.body_radiated_share)
    ratio = _cool_time(0.0) / _cool_time(1.0)
    assert ratio == pytest.approx(1.0 / share, rel=0.03)


def test_thin_hot_gas_barely_warms_a_body(unprotected):
    """Property: hot air warms a body in proportion to its pressure: in 2000 C
    gas at a hundredth of an atmosphere the body settles less than a tenth as
    far above ambient as in the same gas at one atmosphere. Breaks if the air
    term loses its pressure scale (thin hot gas would cook)."""
    def settled(p):
        stub = _Stub(p_atm=p, t_game=1980.0)
        u = _marine()
        _run([u], stub, 120 * TPS)
        return (u.body_temp_q - AMB_Q) / ONE
    thin, dense = settled(0.01), settled(1.0)
    assert dense > 0.8 * 1980.0          # 0.85 of the gap: the radiated share pulls to ambient
    assert 0.0 < thin < 0.1 * dense


def test_the_air_is_read_over_the_footprints_gas_tiles_by_pressure():
    """Property: the air temperature is the pressure-weighted mean of the
    footprint's GAS tiles: a solid tile and a vacuum tile add nothing. Breaks
    if a wall's temperature or a vacuum tile's leaks into the air read."""
    stub = _Stub(t_game=0.0)
    u = _marine()
    tiles = u.occupied_tiles()
    (tx0, ty0), (tx1, ty1) = tiles[0], tiles[1]
    stub.solid[ty0, tx0] = True
    stub.temperature[ty0, tx0] = unit_fixed.quantize_scalar(5000.0)
    stub.atmosphere[ty1, tx1] = 0
    stub.temperature[ty1, tx1] = unit_fixed.quantize_scalar(-292.0)
    p_q, t_q = body_temp.air_read(stub, u, AMB_Q, ONE)
    assert t_q == AMB_Q
    assert p_q == unit_fixed.div_round(7 * ONE, 8)


# ---------------------------------------------------------------------------
# The frag
# ---------------------------------------------------------------------------
def _frag_room():
    import breach_physics as bp
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.materials import MAT_AIR, MAT_HULL
    tm = np.full((40, 40), MAT_HULL, dtype=np.int32)
    tm[1:39, 1:39] = MAT_AIR
    lv = LevelData(name="frag", version="2", path=Path("."), tilemap=tm,
                   tile_size_m=0.333, diffuse_path=Path("."), entities=[],
                   wires=[])
    return Simulation(lv, seed=3, breach_physics=bp, enable_recorder=False)


@pytest.mark.parametrize("which", ("flux", "air"))
def test_the_protection_scales_both_inflows_alike(monkeypatch, which):
    """Property (Erik, 2026-11-18: "protect only the body"): heat_absorption
    is the share of the heat reaching the body that it takes up, for the
    radiation and deposits the felt temperature reads AND for hot air alike:
    at 0.5 the settled excess over ambient is half the unprotected one, at
    the shipped value it is that share, and the losses are untouched (the
    damage law reads the body as before). Breaks if the absorption scales
    only one inflow, scales the losses, or is read anywhere but the body."""
    def settled(h):
        _with_h(monkeypatch, h)
        stub = _Stub(t_game=600.0 if which == "air" else 0.0)
        u = _marine()
        if which == "flux":
            _flux(stub, u, 10.0 / KW_M2_PER_PHI)
        _run([u], stub, 60 * TPS)
        return (u.body_temp_q - AMB_Q) / ONE
    full, half = settled(1.0), settled(0.5)
    assert full > 100.0
    assert half == pytest.approx(0.5 * full, rel=1e-3, abs=0.01)
    shipped = SHIPPED_H_Q / ONE
    assert settled(shipped) == pytest.approx(shipped * full, rel=2e-3, abs=0.01)


def test_at_absorption_one_the_step_is_the_unsplit_law():
    """Property: at heat_absorption 1 the step (the air term split at
    ambient into its inflow and its loss) is the unscaled law bit for bit, at
    any pressure and air temperature. Breaks if the split rounds anywhere."""
    a_q, a_rad_q = _rates(_Stub())
    for p in (0, ONE // 3, ONE, 3 * ONE):
        for t_air in (AMB_Q, AMB_Q + 500 * ONE + 7, AMB_Q - 200 * ONE):
            for body in (AMB_Q, AMB_Q + 913 * ONE + 5):
                felt = AMB_Q + 40 * ONE + 3
                k_air = min(unit_fixed.mul_q16_round(a_q - a_rad_q, p), ONE - a_rad_q)
                old = body + unit_fixed.div_round(
                    a_q * (felt - AMB_Q) - k_air * (body - t_air)
                    - a_rad_q * (body - AMB_Q), ONE)
                assert body_temp.step_q(body, felt, p, t_air, a_q, a_rad_q,
                                        AMB_Q, ONE) == old


def test_a_frag_beside_a_marine_no_longer_kills_by_heat():
    """Property (the calibration's target (a), Erik 2026-11-18): a frag whose
    centre tile is one tile from a marine's (the rung 2c finding: 30 blast +
    576 heat in one tick) leaves a 100 HP marine alive, its heat bill 15 to 40
    HP on top of the blast's 30, at the shipped protection and the shipped
    10 MJ. Breaks if the body law or the protection is undone, or the
    protection is set so high a frag no longer burns at all."""
    from simulation.events import UnitHitEvent
    from simulation.payloads import execute_payload
    from simulation.weapons import get_tables
    s = _frag_room()
    m = Unit("M", x=20.5, y=19.5, team=0)
    s.add_unit(m)
    s.set_paused(False)
    s.step()
    by = {}
    events = []
    execute_payload(s.gmap, s.edit_queue, s.units, 20, 20,
                    get_tables().payloads.by_name["frag_standard"], s.draws,
                    events=events)
    for _ in range(6 * TPS):
        s.set_paused(False)
        s.step()
        events += list(s.tick_events)
    for ev in events:
        if isinstance(ev, UnitHitEvent) and ev.unit_id == m.id:
            by[str(ev.source)] = by.get(str(ev.source), 0.0) + float(ev.damage)
    assert m.alive, by
    assert by.get("explosion", 0.0) == 30.0
    assert 15.0 <= by.get("heat", 0.0) <= 40.0, by


# ---------------------------------------------------------------------------
# One writer; the clears; determinism
# ---------------------------------------------------------------------------
def test_only_the_body_temp_module_writes_the_body():
    """Property: nothing in ``src/`` but ``simulation/body_temp.py`` assigns
    ``body_temp_q`` (the schema's default aside): the heat row writes it
    through ``body_temp.advance``, the clears through ``body_temp.clear``.
    Breaks if a second writer appears (anything that warms a body must change
    the felt temperature or the air, never the body)."""
    bad = []
    for p in sorted((ROOT / "src").rglob("*.py")):
        if p.name in ("body_temp.py", "unit_schema.py"):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute) and sub.attr == "body_temp_q":
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    assert not bad, bad


def test_death_resets_the_body_and_an_ambient_body_hashes_as_untouched():
    """Property: a unit that dies goes back to ambient, and a body at ambient
    carries no digest row. Breaks if the death clear forgets the body or the
    column's default is not the ambient."""
    from simulation.damage import KINETIC, DamagePacket, apply_packet
    u = _marine()
    assert "body_temp_q" not in unit_schema.digest_record(u)
    u.body_temp_q = AMB_Q + 300 * ONE
    assert "body_temp_q" in unit_schema.digest_record(u)
    u.current_hp = 1.0
    apply_packet(u, DamagePacket(amount=5.0, dtype=KINETIC), None, source="bullet")
    assert not u.alive and u.body_temp_q == AMB_Q


def test_a_cold_body_in_still_ambient_air_never_moves(unprotected):
    """Property: with no flux, ambient air and an ambient body the law moves
    nothing, so a match without heat leaves every body untouched and out of
    the digest. Breaks if the law drifts at rest (a rounding bias)."""
    stub = _Stub()
    u = _marine()
    _run([u], stub, 10 * TPS)
    assert u.body_temp_q == AMB_Q


def test_the_same_scene_twice_is_the_same_body_bit_for_bit():
    """Property: the law is integer: the same flux history gives the same
    body and hit points on every run. Breaks if a float enters the step."""
    def run():
        stub = _Stub(p_atm=0.73, t_game=113.0)
        us = [_marine(5, 5), _marine(15, 15)]
        for k in range(3 * TPS):
            stub.heat[...] = 0
            _flux(stub, us[k % 2], 0.01 * (k % 7))
            _run(us, stub, 1)
        return [(u.body_temp_q, u.hp_q) for u in us]
    assert run() == run()


# ---------------------------------------------------------------------------
# The readouts
# ---------------------------------------------------------------------------
def test_the_panel_and_the_inspector_show_body_stress_and_heart_rate():
    """Property (UI split; the tile inspector seam): the side panel's data
    carries the selected unit's body temperature, stress and heart rate, and
    the tile inspector's body line (added for a living unit standing on the
    hovered tile) carries the same, the heart rate read off the one
    Hill curve. Breaks if a readout is dropped or computes its own curve."""
    from renderer.hover_readout import body_line
    from simulation import stress
    from ui.model import body_text
    u = _marine()
    u.body_temp_q = AMB_Q + 15 * ONE
    u.stress_q = 50 * stress.LEVEL_ONE
    hr = stress.heart_rate_q(u) / ONE
    assert body_text(u) == f"Body {(AMB_Q / ONE) + 15:.1f} C  stress 50.0  heart {hr:.0f} bpm"
    assert "stress  50.0" in body_line(u) and f"heart {hr:5.1f} bpm" in body_line(u)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
