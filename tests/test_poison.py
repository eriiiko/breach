"""The poison law's gate (#29 PX; ``docs/poison_law_design_2026-11-14.md`` §6).

``src/simulation/poison.py``: two per-unit levels, depot and blood, moved by
ONE integer law that conserves what it moves and charges ``hp_per_unit`` for
every count leaving the blood, through ``damage.apply_packet``; every
poisoning enters through ``poison.dose`` (today from the ``gas[poison]``
coupling row). Each test names the property it protects and the change that
must break it.

Run:
    <breach py> -m pytest tests/test_poison.py -q
"""
from __future__ import annotations

import ast
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation, gas_fixed, poison, unit_fixed  # noqa: E402
from simulation import unit_schema  # noqa: E402
from simulation.damage import KINETIC, DamagePacket, apply_packet, tick_credit  # noqa: E402
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402
from simulation.exchange import apply_poison_dose  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import POISON as GAS_POISON  # noqa: E402
from simulation.unit import Unit  # noqa: E402

ONE = unit_fixed.FP_ONE
TPS = int(CFG.clock.ticks_per_second)
S = poison.get_settings()


def _level(h=24, w=24):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = tm[-1, :] = 1
    tm[:, 0] = tm[:, -1] = 1
    return LevelData(name="px_poison", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _marine(hp=100000.0, uid=1, team=0, x=8, y=8):
    u = Unit("M", x=x, y=y, team=team)
    u.id = uid
    u.current_hp = hp
    return u


def _fill(gmap, u, density):
    q = gas_fixed.quantize_trace_scalar(density)
    for (tx, ty) in u.occupied_tiles():
        gmap.gas[GAS_POISON][ty, tx] = q
    return q


def _run_dose(dose_q, max_ticks=24 * 900):
    """One dose at tick 0, the law run to dormancy: (per-tick damage in Q16
    hp, per-tick blood level, the unit, ticks to dormancy)."""
    u = _marine()
    poison.dose(u, dose_q, -1)
    dmg, blood = [], []
    for t in range(max_ticks):
        h0 = u.hp_q
        poison.tick([u], t, None)
        dmg.append(h0 - u.hp_q)
        blood.append(u.poison_x1)
        if not poison.holds(u):
            return dmg, blood, u, t + 1
    return dmg, blood, u, None


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------
def _cfg(**over):
    sec = dict(onset_seconds=4.0, decay_seconds=20.0, hp_per_unit=10.0,
               gas_units_per_second=0.6)
    sec.update(over)
    sec = {k: v for k, v in sec.items() if v is not None}
    return SimpleNamespace(damage=SimpleNamespace(poison=sec),
                           clock=SimpleNamespace(ticks_per_second=24))


def test_the_door_quantizes_once_and_refuses_by_name():
    """Property: ``[damage.poison]`` is read through a refusing door: the
    first values give the design's fractions (683, 137 at 24 tps) and H = 10
    points exactly; a missing or unknown key, a non-number, a time below one
    tick, ``hp_per_unit`` outside (0, 8192) and ``gas_units_per_second``
    outside [0, 65536) are refused naming the key. Breaks if a dial is read
    raw, the fraction is taken through a float division or exponential, or
    a bound of the int64 argument is dropped."""
    s = poison.settings_from_config(_cfg())
    assert (s.onset_q, s.decay_q, s.harm_q) == (683, 137, 10 * ONE)
    assert s.gas_q == unit_fixed.quantize_scalar(0.6)
    for over, key in ((dict(onset_seconds=None), "onset_seconds"),
                      (dict(bogus=1.0), "bogus"),
                      (dict(decay_seconds="20"), "decay_seconds"),
                      (dict(onset_seconds=0.03), "onset_seconds"),
                      (dict(hp_per_unit=0.0), "hp_per_unit"),
                      (dict(hp_per_unit=8192.0), "hp_per_unit"),
                      (dict(gas_units_per_second=-0.1),
                       "gas_units_per_second"),
                      (dict(gas_units_per_second=65536.0),
                       "gas_units_per_second")):
        with pytest.raises(ValueError, match=key):
            poison.settings_from_config(_cfg(**over))
    # The shipped config passes and IS the first values.
    live = poison.settings_from_config(CFG)
    assert (live.onset_q, live.decay_q, live.harm_q) == (683, 137, 10 * ONE)


# ---------------------------------------------------------------------------
# Conservation and the total harm
# ---------------------------------------------------------------------------
def test_poison_is_conserved_over_any_dosing_schedule():
    """Property (Erik's rule): over any dosing schedule, every tick,
    ``sum(doses taken) == x0 + x1 + sum(m1)`` in counts -- the law creates
    and loses nothing, because m0 and m1 are each computed ONCE and applied
    to both sides. Breaks if m0 (or m1) is computed twice with different
    rounding (validated: ``x1 += (x0_before * a + 32768) >> 16`` beside
    ``x0 -= m0`` turns it red)."""
    rng = random.Random(29)
    for trial in range(6):
        u = _marine()
        taken = 0
        left = 0
        for t in range(24 * 120):
            if t < 24 * 40 and rng.random() < 0.3:
                taken += poison.dose(u, rng.choice(
                    (1, 7, 300, poison.FLUSH_Q, 2000, ONE, 5 * ONE)), -1)
            _m0, m1 = poison.advance(u)
            left += m1
            assert taken == u.poison_x0 + u.poison_x1 + left, (trial, t)
        assert taken > 0


def test_the_total_harm_of_a_dose_is_h_times_the_dose():
    """Property: a dose's total damage is ``H x dose`` (Q16 points) to within
    half a count per damaging tick -- whatever the two times are; and the
    blood takes exactly the dose (``sum(m1) == dose``). Breaks if the damage
    is charged on the blood LEVEL instead of what leaves it, if a count
    leaves uncharged, or if the per-tick rounding is not half-away once."""
    for dose_q in (1, 513, poison.FLUSH_Q, ONE, 7 * ONE + 12345):
        dmg, _blood, _u, n = _run_dose(dose_q)
        assert n is not None
        hits = sum(1 for d in dmg if d)
        want = dose_q * S.harm_q / ONE
        assert abs(sum(dmg) - want) <= 0.5 * hits, dose_q
    # Other times, the same total (the times set WHEN, not HOW MUCH).
    fast = poison.settings_from_config(_cfg(onset_seconds=1.0,
                                            decay_seconds=3.0))
    u = _marine()
    poison.dose(u, 3 * ONE, -1)
    total = m1s = 0
    while poison.holds(u):
        _m0, m1 = poison.advance(u, fast)
        m1s += m1
        total += unit_fixed.mul_q16_round(m1, fast.harm_q)
    assert m1s == 3 * ONE
    assert abs(total - 3 * ONE * fast.harm_q / ONE) <= 0.5 * 24 * 60


# ---------------------------------------------------------------------------
# The shape of one dose
# ---------------------------------------------------------------------------
def _closed_form_peak_tick(a0, a1):
    """The discrete law's blood peak for one dose (no rounding): the blood
    seen after tick n is ``a0 D r1 (r1^n - r0^n) / (r1 - r0)``, ``r = 1 -
    a/ONE``; its maximum over integer n."""
    r0, r1 = 1 - a0 / ONE, 1 - a1 / ONE
    best, best_n = -1.0, 0
    for n in range(1, 24 * 120):
        v = r1 ** n - r0 ** n
        if v > best:
            best, best_n = v, n
    return best_n


def test_one_dose_rises_peaks_once_and_decays():
    """Property: one dose's blood level rises to ONE peak and decays, the
    peak where the discrete law puts it (the closed form of the two
    fractions, about 8 s at the first values), and the per-tick damage rises
    and falls with it (weakly unimodal: it is the floor of a fraction of the
    blood) bar the final flush tick, which charges at most the flush's
    ``FLUSH_Q x H``. Breaks if the two compartments are collapsed (damage
    from the depot directly), the fractions are swapped, or the flush fires
    while the depot still feeds the blood."""
    dmg, blood, _u, n = _run_dose(ONE)
    peak = max(range(len(blood)), key=lambda i: blood[i]) + 1
    want = _closed_form_peak_tick(S.onset_q, S.decay_q)
    assert abs(peak - want) <= 2, (peak, want)
    assert 7.5 <= want / TPS <= 8.5
    i = blood.index(max(blood))
    assert all(blood[k] <= blood[k + 1] for k in range(i))
    assert all(blood[k] >= blood[k + 1] for k in range(i, len(blood) - 1))
    body = dmg[:-1]
    j = body.index(max(body))
    assert all(body[k] <= body[k + 1] for k in range(j))
    assert all(body[k] >= body[k + 1] for k in range(j, len(body) - 1))
    assert dmg[-1] <= poison.FLUSH_Q * S.harm_q // ONE + 1
    # The peak rate is the design's ~0.33 HP a second for one bite.
    assert 0.30 <= max(body) * TPS / ONE <= 0.36


def test_a_fed_level_is_never_flushed():
    """Property (as built): the flush empties the BLOOD only once the depot
    is empty, so a dose smaller than ``FLUSH_Q`` still goes depot -> blood
    -> out over time (the decay stage exists for a fringe-gas breath). With
    the design's literal flush (every level at or below 1/64 unit, every
    tick) the blood of a 1/40-unit-a-tick inflow never exceeds the flush and
    is charged the tick it arrives. Breaks if the flush is applied to a
    level that is still being fed."""
    u = _marine()
    first = []
    for t in range(24 * 10):
        poison.dose(u, poison.FLUSH_Q // 2, -1)
        h0 = u.hp_q
        poison.tick([u], t, None)
        first.append(h0 - u.hp_q)
    assert u.poison_x1 > poison.FLUSH_Q          # the blood builds up
    assert first[0] < poison.FLUSH_Q // 2 * S.harm_q // ONE // 10


def test_a_level_returns_to_exactly_zero_and_hashes_as_never_poisoned():
    """Property: every dose drains to EXACTLY 0 in both levels, ``poison_src``
    returns to -1 and the unit's digest record carries no poison row -- the
    same record as a twin never poisoned bar the hit points it lost; and the
    tail is the flush's: once the depot is empty and the blood is at or
    below ``FLUSH_Q``, the next tick ends it. Breaks if the flush is removed
    (validated: the tail then drips one count a tick for hundreds of
    ticks), if a level is left at a residue, or if the credit outlives it."""
    for dose_q in (1, 96, poison.FLUSH_Q, ONE, 100 * ONE):
        u = _marine()
        poison.dose(u, dose_q, 4)
        assert u.poison_src == 4
        ticks = 0
        while poison.holds(u):
            if u.poison_x0 == 0 and u.poison_x1 <= poison.FLUSH_Q:
                poison.tick([u], ticks, None)
                ticks += 1
                assert not poison.holds(u), (dose_q, "the flush did not end it")
                break
            poison.tick([u], ticks, None)
            ticks += 1
            assert ticks < 24 * 900, dose_q
        assert (u.poison_x0, u.poison_x1, u.poison_src) == (0, 0, -1)
        rec = unit_schema.digest_record(u)
        assert not [k for k in rec if k.startswith("poison")]
        twin = _marine()
        twin.hp_q = u.hp_q
        assert rec == unit_schema.digest_record(twin)


def test_a_poisoned_unit_hashes_its_levels():
    """Property: while a unit holds poison its two levels and its credit are
    in the digest record (hashed only while non-default). Breaks if a column
    is left out of the schema or hashed always."""
    u = _marine()
    assert not [k for k in unit_schema.digest_record(u)
                if k.startswith("poison")]
    poison.dose(u, ONE, 9)
    rec = unit_schema.digest_record(u)
    assert rec["poison_x0"] == ONE and rec["poison_src"] == 9
    assert "poison_x1" not in rec
    poison.tick([u], 0, None)
    rec = unit_schema.digest_record(u)
    assert rec["poison_x1"] > 0


def test_the_body_is_bounded_and_the_overflow_counted():
    """Property: the body holds at most ``X_MAX`` counts in all (the int64
    argument's bound); a dose past it is clipped and counted, never
    wrapped. Breaks if the door stops clipping or stops counting."""
    u = _marine()
    before = dict(poison.SATURATION)
    took = poison.dose(u, poison.X_MAX - 5, -1)
    assert took == poison.X_MAX - 5
    assert poison.dose(u, 100, -1) == 5
    assert poison.dose(u, 100, -1) == 0
    assert u.poison_x0 + u.poison_x1 == poison.X_MAX
    assert poison.SATURATION["hits"] == before["hits"] + 2
    assert poison.SATURATION["dropped_q"] == before["dropped_q"] + 195
    poison.clear(u)


# ---------------------------------------------------------------------------
# Who holds poison
# ---------------------------------------------------------------------------
def test_a_zombie_holds_no_poison_and_emits_nothing():
    """Property: a zombie in full poison gas takes no dose, holds no poison
    and emits nothing, through the real conductor (slot 9c3), while a marine
    beside it is dosed and hurt -- and the dose door itself refuses a zombie
    and a corpse. Breaks if the zombie gate is removed from the dose door
    (validated: the door's ``is_zombie`` check deleted turns it red; the
    row's resist gate alone does not cover a direct dose)."""
    sim = Simulation(_level(), seed=5, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=6, y=6, team=0)
    z = Unit("Z", x=14, y=14, team=ZOMBIE_TEAM)
    sim.add_unit(m)
    sim.add_unit(z)
    hp_m, hp_z = m.current_hp, z.current_hp
    for _ in range(48):
        _fill(sim.gmap, m, 1.0)
        _fill(sim.gmap, z, 1.0)
        sim.set_paused(False)
        sim.step()
        assert z.poison_x0 == z.poison_x1 == 0
        assert not [e for e in sim.tick_events
                    if isinstance(e, UnitHitEvent) and e.unit_id == z.id]
    assert z.current_hp == hp_z
    assert m.current_hp < hp_m and poison.holds(m)
    # The door alone.
    z2 = _marine(team=ZOMBIE_TEAM)
    z2.is_zombie = True
    assert poison.dose(z2, ONE, -1) == 0 and not poison.holds(z2)
    dead = _marine()
    dead.alive = False
    assert poison.dose(dead, ONE, -1) == 0 and not poison.holds(dead)


def test_a_unit_that_dies_or_turns_is_cleared():
    """Property: a death (by the poison itself or by any other packet)
    zeroes both levels and the credit; so does a conversion to a zombie.
    Breaks if ``apply_packet``'s death path or the conversion stops calling
    ``poison.clear``."""
    from simulation.ai_zombie import convert_marines_to_zombies
    # Killed by the poison itself.
    u = _marine(hp=1.0)
    poison.dose(u, 5 * ONE, 3)
    events = []
    for t in range(24 * 120):
        poison.tick([u], t, events)
        if not u.alive:
            break
    assert not u.alive
    assert (u.poison_x0, u.poison_x1, u.poison_src) == (0, 0, -1)
    kills = [e for e in events if isinstance(e, UnitKilledEvent)]
    assert len(kills) == 1 and kills[0].killed_by == "poison_gas"
    assert kills[0].attacker_id == 3
    # Killed by a bullet while holding poison.
    v = _marine(hp=5.0, uid=2)
    poison.dose(v, ONE, 3)
    apply_packet(v, DamagePacket(10, KINETIC, 7), [], source="bullet")
    assert not v.alive and not poison.holds(v) and v.poison_src == -1
    # Turned: a corpse marked by a bite (the levels set by hand: the death
    # path cleared them already, the conversion is its own rule).
    w = _marine(hp=5.0, uid=3)
    apply_packet(w, DamagePacket(10, KINETIC, 7), [], source="melee",
                 mark_killed_by_zombie=True)
    w.poison_x0, w.poison_x1, w.poison_src = ONE, ONE, 7
    convert_marines_to_zombies([w])
    assert w.is_zombie and w.alive
    assert (w.poison_x0, w.poison_x1, w.poison_src) == (0, 0, -1)


# ---------------------------------------------------------------------------
# The gas input
# ---------------------------------------------------------------------------
def test_the_gas_dose_is_linear_in_density_and_gated():
    """Property: the gas row's dose is ``gas_units_per_second x density /
    tps`` rounded once -- linear in density to within one count -- and
    nothing at all below ``poison_min_density``. Breaks if the dose stops
    scaling with density, is rounded twice, or the gate moves."""
    def dose_at(d):
        g = GameMap(_level())
        u = _marine()
        _fill(g, u, d)
        apply_poison_dose([u], g, TPS, events=None)
        return u.poison_x0
    d1 = dose_at(0.25)
    for k in (2, 3, 4):
        assert abs(dose_at(0.25 * k) - k * d1) <= k
    thresh = float(CFG.exchange.poison_min_density)
    g = GameMap(_level())
    u = _marine()
    q = gas_fixed.quantize_trace_scalar(thresh) - 1
    for (tx, ty) in u.occupied_tiles():
        g.gas[GAS_POISON][ty, tx] = q
    apply_poison_dose([u], g, TPS, events=None)
    assert not poison.holds(u) and u.poison_src == -1


def test_steady_gas_hurts_at_h_times_the_breathing_rate():
    """Property: at a steady exposure the damage per second tends to ``H x
    gas_units_per_second x density`` (the deleted ``poison_dps x density``:
    the patch moved the timing and not the total), and the harm starts late
    (under 1 HP after 4 s in full gas). Breaks if the gas gain or ``H`` is
    applied twice or not at all, or the law loses poison at steady state."""
    g = GameMap(_level())
    u = _marine()
    density = 0.5
    _fill(g, u, density)
    curve = []
    for t in range(24 * 400):
        apply_poison_dose([u], g, TPS, events=None)
        h0 = u.hp_q
        poison.tick([u], t, None)
        curve.append(h0 - u.hp_q)
    want = (S.harm_q / ONE) * (S.gas_q / ONE) * density     # HP per second
    got = sum(curve[-24 * 10:]) / ONE / 10
    assert abs(got - want) <= 0.01 * want, (got, want)
    g2 = GameMap(_level())
    v = _marine()
    _fill(g2, v, 1.0)
    lost = 0
    for t in range(24 * 4):
        apply_poison_dose([v], g2, TPS, events=None)
        h0 = v.hp_q
        poison.tick([v], t, None)
        lost += h0 - v.hp_q
    assert lost < ONE


def test_damage_after_leaving_the_cloud_is_credited_to_the_dose():
    """Property: the poison a unit breathed while a sprayer's claim held its
    tile keeps hurting after it left the cloud, every such packet credited
    to that sprayer, INDIRECTLY (on the events, never in
    ``recent_attackers``), with the cause ``poison_gas``. Breaks if the
    packets lose ``poison_src`` (credited to the environment) or are marked
    direct (validated: ``indirect=True`` removed fills recent_attackers)."""
    g = GameMap(_level())
    u = _marine()
    u.recent_attackers = {}
    for t in range(24 * 3):                   # three seconds in the jet
        _fill(g, u, 1.0)
        tick_credit(g).claim(GAS_POISON,
                             [(ty, tx) for (tx, ty) in u.occupied_tiles()],
                             42)
        apply_poison_dose([u], g, TPS, events=None)
        poison.tick([u], t, None)
        tick_credit(g).clear()
    assert u.poison_src == 42
    g.gas[GAS_POISON][...] = 0                # left the cloud
    events = []
    for t in range(24 * 30):
        poison.tick([u], t, events)
    hits = [e for e in events if isinstance(e, UnitHitEvent)]
    assert len(hits) > 24 * 20
    assert all(e.attacker_id == 42 and e.source == "poison_gas"
               for e in hits)
    assert not u.recent_attackers


def test_the_conductor_keeps_hurting_after_the_cloud_is_gone():
    """Property (wiring, slot 9c3'): through ``Simulation.step`` a marine
    dosed in a cloud keeps losing hit points after the cloud is cleared, the
    law running outside the gas row's lazy early-out. Breaks if
    ``poison.tick`` is called only while the poison plane is non-zero."""
    sim = Simulation(_level(), seed=5, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=8, y=8, team=0)
    sim.add_unit(m)
    for _ in range(24 * 2):
        _fill(sim.gmap, m, 1.0)
        sim.set_paused(False)
        sim.step()
    sim.gmap.gas[GAS_POISON][...] = 0
    hp = m.current_hp
    for _ in range(24 * 5):
        sim.set_paused(False)
        sim.step()
        assert not sim.gmap.gas[GAS_POISON].any()
    assert m.current_hp < hp


# ---------------------------------------------------------------------------
# The scan: one writer, one emitter
# ---------------------------------------------------------------------------
_COLUMNS = ("poison_x0", "poison_x1", "poison_src")
_SRC = ROOT / "src"


def _py_files():
    return sorted(p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_only_the_poison_module_writes_the_columns():
    """Property: nothing in ``src/`` but ``simulation/poison.py`` assigns a
    poison column (an attribute store or a ``setattr`` naming it) -- the
    schema's default aside. Breaks if a second writer appears (e.g. a row
    adding to ``poison_x0`` itself instead of calling ``poison.dose``)."""
    bad = []
    for p in _py_files():
        if p.name in ("poison.py", "unit_schema.py"):
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
                    if isinstance(sub, ast.Attribute) and sub.attr in _COLUMNS:
                        bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "setattr" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in _COLUMNS):
                bad.append(f"{p.relative_to(ROOT)}:{node.lineno}")
    assert not bad, f"poison columns written outside poison.py: {bad}"


def test_only_the_law_emits_a_poison_packet():
    """Property: the only ``DamagePacket`` of type POISON built in ``src/``
    is the one in ``poison.tick`` (a status kind naming POISON as its
    dtype is not a packet). Breaks if a coupling row or a weapon builds a
    POISON packet of its own (validated: the pre-PX gas row's packet put
    back turns it red)."""
    sites = []
    for p in _py_files():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        funcs = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(fn):
                    funcs.setdefault(id(sub), fn.name)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id",
                                getattr(node.func, "attr", "")) == "DamagePacket"):
                continue
            dt = [k.value for k in node.keywords if k.arg == "dtype"]
            if len(node.args) >= 2:
                dt.append(node.args[1])
            for v in dt:
                name = getattr(v, "id", getattr(v, "attr", None))
                if name == "POISON":
                    sites.append((p.name, funcs.get(id(node))))
    assert sites == [("poison.py", "tick")], sites


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
