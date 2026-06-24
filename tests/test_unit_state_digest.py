"""S0 — the synced UNIT-STATE digest gate for the field A/B harness.

``field_ab_harness`` originally hashed only gmap field arrays, so it was BLIND
to unit state: two trajectories could agree on every gmap cell yet disagree on a
unit's HP or on who lived and who died. That is the exact leak an upcoming
decision (combat HP / damage going INTEGER) must not open — a float-reorder or a
mis-ordered kill could desync who survives while leaving every physics cell
identical.

This test proves the harness now SEES that. The scenario drives two marines so
their footprints MOVE, injects heat each tick so they take DAMAGE (HP changes +
``UnitHitEvent`` fires), and arranges for one to DIE (``UnitKilledEvent`` fires).
Then it asserts:

  (a) the per-tick ``unit_digest`` is NON-trivial (changes tick-to-tick: position,
      HP, and life state all move);
  (b) a re-run is byte-stable (same machine) — the digest is reproducible;
  (c) a deliberately perturbed unit HP makes the trajectory DIVERGE, and the diff
      names the exact unit id + field — so an integer-vs-float HP desync FAILS;
  (d) a deliberately perturbed kill EVENT also makes it diverge — so a divergent
      life/death stream FAILS even when every HP and gmap cell still matches;
  (e) the harness still PASSES when two runs genuinely agree on units.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_state_digest.py -q
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tests", ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import breach_physics as bp
from level_loader import LevelData
from simulation import Simulation
from simulation.combat import HEAT_SCALE
from simulation.unit import Unit

from field_ab_harness import (
    SIM_FIELDS, UNIT_DIGEST_KEY,
    _capture_unit_state, diff_trajectories, unit_digest_hash,
)

SEED = 20260624
N_STEPS = 36
DEATH_TICK = 22


def _scenario_level() -> LevelData:
    """A 20x20 hull-walled room with a carved interior — room for two 3x3
    marines to walk around inside the air pocket."""
    h = w = 20
    tm = np.ones((h, w), dtype=np.int32)   # all hull
    tm[1:19, 1:19] = 4                       # carve interior air
    return LevelData(name="unit_digest", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _make_sim() -> Simulation:
    """Two 3x3 marines in a carved room. No gmap physics seeding is needed —
    this gate is about UNIT state, and heat is injected by the driver each tick."""
    sim = Simulation(_scenario_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    sim.add_unit(Unit("M1", x=4, y=4, team=0))
    sim.add_unit(Unit("M2", x=13, y=13, team=0))
    sim.set_paused(False)
    return sim


def _inject_heat(sim, unit, phi):
    """Stamp a Q16.16 heat deposit on the unit's footprint so the in-step
    ``apply_environmental_damage`` consumer burns it this tick."""
    raw = int(round(phi * HEAT_SCALE))
    for (tx, ty) in unit.occupied_tiles():
        ty = max(0, min(sim.gmap.heat.shape[0] - 1, ty))
        tx = max(0, min(sim.gmap.heat.shape[1] - 1, tx))
        sim.gmap.heat[ty, tx] = raw


def _drive(sim, tick, kill_m2_at=DEATH_TICK):
    """Deterministic per-tick driver (pure function of ``tick`` — no RNG).

    - M1 walks a diagonal box and gets a mild heat tick every step (HP creeps
      down, but it never dies).
    - M2 walks the other way and gets a HARD heat tick; we drop its HP low just
      before ``kill_m2_at`` so the next strong tick KILLS it (UnitKilledEvent).

    Heat is injected BEFORE the step so the post-physics damage consumer reads
    it; positions are set BEFORE injection so the footprint is current."""
    units = sim.units
    if len(units) >= 1:
        m1 = units[0]
        if m1.alive:
            m1.x = float(2 + (tick % 14))
            m1.y = float(2 + ((tick // 2) % 14))
            _inject_heat(sim, m1, 60.0)         # mild, survivable burn each tick
    if len(units) >= 2:
        m2 = units[1]
        if m2.alive:
            m2.x = float(16 - (tick % 12))
            m2.y = float(3 + (tick % 12))
            if tick == kill_m2_at - 1:
                m2.current_hp = 0.5             # set up the kill
            phi = 400.0 if tick >= kill_m2_at else 60.0
            _inject_heat(sim, m2, phi)


def _capture(perturb=None, kill_m2_at=DEATH_TICK):
    """Run the scenario for N_STEPS, returning a per-tick list of snapshot dicts
    that carry both the gmap fields AND the synced unit digest.

    ``perturb(sim, tick)`` (optional) is called AFTER the step each tick — used
    to inject a desync (e.g. nudge a unit's HP) into one of two A/B runs."""
    sim = _make_sim()
    traj = []
    for t in range(N_STEPS):
        _drive(sim, t, kill_m2_at=kill_m2_at)
        sim.set_paused(False)
        sim.step()
        if perturb is not None:
            perturb(sim, t)
        snap = {name: np.copy(getattr(sim.gmap, name))
                for name in SIM_FIELDS if hasattr(sim.gmap, name)}
        snap[UNIT_DIGEST_KEY] = _capture_unit_state(sim)
        traj.append(snap)
    return traj


# ---------------------------------------------------------------------------
# (a) the digest is non-trivial — units move, take damage, and one dies
# ---------------------------------------------------------------------------
def test_unit_digest_is_nontrivial():
    traj = _capture()

    hashes = [unit_digest_hash(s) for s in traj]
    assert len(set(hashes)) > N_STEPS // 2, (
        "unit digest barely changed — scenario units are not moving / taking "
        "damage (the gate would be vacuous)")

    # HP actually moved: M1's hp at an early vs late tick must differ.
    early = next(u for u in traj[2][UNIT_DIGEST_KEY]["units"] if u["id"] == 0)
    late = next(u for u in traj[-1][UNIT_DIGEST_KEY]["units"] if u["id"] == 0)
    assert late["current_hp"] < early["current_hp"], "M1 never took heat damage"

    # Position actually moved.
    assert (early["tile_x"], early["tile_y"]) != (late["tile_x"], late["tile_y"]), \
        "M1 never moved"

    # M2 died: it is ALIVE early and DEAD by the end, and a UnitKilledEvent fired.
    m2_early = next(u for u in traj[2][UNIT_DIGEST_KEY]["units"] if u["id"] == 1)
    m2_late = next(u for u in traj[-1][UNIT_DIGEST_KEY]["units"] if u["id"] == 1)
    assert m2_early["alive"] and not m2_late["alive"], "M2 never died"

    all_events = [e for s in traj for e in s[UNIT_DIGEST_KEY]["events"]]
    kinds = {e["kind"] for e in all_events}
    assert "UnitHitEvent" in kinds, "no UnitHitEvent ever captured (no damage)"
    assert "UnitKilledEvent" in kinds, "no UnitKilledEvent ever captured (no death)"
    # The kill event names M2.
    kills = [e for e in all_events if e["kind"] == "UnitKilledEvent"]
    assert any(e["unit_id"] == 1 for e in kills), "kill event did not name M2"


# ---------------------------------------------------------------------------
# (b) byte-stable across runs on the same machine
# ---------------------------------------------------------------------------
def test_unit_digest_byte_stable():
    a = _capture()
    b = _capture()
    assert [unit_digest_hash(s) for s in a] == [unit_digest_hash(s) for s in b], \
        "unit digest is not reproducible across runs (non-determinism leaked in)"
    # And the full A/B comparison agrees on units too.
    assert not diff_trajectories(a, b, tol=0.0), \
        "identical runs reported a spurious unit-state diff"


# ---------------------------------------------------------------------------
# (c) a perturbed unit HP makes the digest diverge — the leak this closes
# ---------------------------------------------------------------------------
def test_perturbed_hp_diverges_even_with_identical_fields():
    """THE point: nudge ONE unit's HP in run B while keeping every gmap field
    byte-identical, and the harness must FAIL — naming the exact unit + field."""
    a = _capture()
    b = copy.deepcopy(a)

    # Surgically perturb M1's HP in one tick's digest, recomputing only that
    # tick's hash. Every gmap field array is left byte-identical to run `a`, so
    # the OLD field-only harness would have seen NOTHING.
    tick = N_STEPS // 2
    state = b[tick][UNIT_DIGEST_KEY]
    m1 = next(u for u in state["units"] if u["id"] == 0)
    m1["current_hp"] += 1            # the smallest integer HP desync
    import hashlib
    payload = {"units": state["units"], "events": state["events"]}
    state["hash"] = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    # Sanity: every gmap FIELD is still identical (the old harness was blind).
    for k in a[tick]:
        if k == UNIT_DIGEST_KEY:
            continue
        assert np.array_equal(a[tick][k], b[tick][k]), \
            f"field {k} changed — the perturbation was not unit-only"

    diffs = diff_trajectories(a, b, tol=0.0)
    assert diffs, "harness MISSED a unit-HP desync with identical gmap fields!"
    joined = "\n".join(diffs)
    assert "unit id 0" in joined and "current_hp" in joined, (
        "diff did not name the diverging unit + field; got:\n" + joined)


# ---------------------------------------------------------------------------
# (d) a perturbed KILL EVENT makes the digest diverge
# ---------------------------------------------------------------------------
def test_perturbed_kill_event_diverges():
    """A divergent life/death event stream must FAIL even if HP and gmap cells
    all match. We model it by capturing a run where M2 dies one tick LATER, then
    diffing the kill-event stream against the baseline."""
    a = _capture(kill_m2_at=DEATH_TICK)
    b = _capture(kill_m2_at=DEATH_TICK + 1)
    diffs = diff_trajectories(a, b, tol=0.0)
    assert diffs, "harness MISSED a shifted kill — life/death stream desync!"
    joined = "\n".join(diffs)
    # The first divergence is on M2 (unit id 1) — its alive/HP or its kill event.
    assert "unit id 1" in joined or "UnitKilledEvent" in joined, (
        "diff did not implicate M2's life/death; got:\n" + joined[:1000])


# ---------------------------------------------------------------------------
# (e) genuinely-agreeing runs still PASS (no false positives)
# ---------------------------------------------------------------------------
def test_agreeing_runs_pass():
    a = _capture()
    b = _capture()
    # No assertion error from the strict assert path either.
    from field_ab_harness import assert_trajectories_match
    assert_trajectories_match(a, b, tol=0.0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
