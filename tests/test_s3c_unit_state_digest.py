"""S3c — the SYNCED UNIT-STATE determinism digest, gated through an
ignite -> fire -> unit-damage -> KILL scenario (plan §S3c gate, Q5).

S3 closes the fire -> heat -> damage loop: a fire deterministically heats a unit's
footprint, the C++ heat -> temperature pass + the (Q2-fenced, still float-Python)
``apply_environmental_damage`` drain the unit's HP, and the unit dies — emitting a
``UnitHitEvent`` stream and a ``UnitKilledEvent``. The Q2 decision keeps the HP
math float-Python for now (scalar float +/-/*/÷ is cross-machine reproducible — no
FMA, no transcendental jitter), but S3 is where that float-HP step first becomes
EXERCISABLE by deterministic integer fire. So the fire determinism must be WATCHED
end-to-end THROUGH the kill event — not silently leaked at the HP step.

The field A/B harness (``field_ab_harness.py``) hashes only gmap arrays; it is
blind to unit HP/life and the hit/kill event stream. This module extends the gate:
it captures the SYNCED UNIT-STATE digest each tick (per-unit HP/life/faction/
position/footprint + the pending hit/kill EVENT stream, defined ONCE in
``field_ab_harness.SYNCED_UNIT_FIELDS`` / ``_capture_unit_state``, master plan
§6.1.3) and asserts it is BIT-IDENTICAL across two runs of the fire->kill scenario.

A nondeterminism leaking through the float-HP step (e.g. a non-reproducible damage
sum, an order-dependent event emission, a kill landing on a different tick) leaves
the gmap fields identical yet FAILS here — which is exactly the leak this digest
exists to catch.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_s3c_unit_state_digest.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "tests", ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                       # noqa: E402
from level_loader import LevelData                # noqa: E402
from simulation import Simulation                 # noqa: E402
from simulation import fire_fixed                 # noqa: E402  gmap.fire is int32 Q16.16
from simulation.materials import MAT_WOOD         # noqa: E402
from simulation.unit import Unit                  # noqa: E402

from field_ab_harness import (                    # noqa: E402
    capture_trajectory, assert_trajectories_match, diff_trajectories,
    unit_digest_hash, UNIT_DIGEST_KEY,
)

SEED = 0xF12E            # "F12E" — fire
TICKS = 40
# A wood burner column near the unit; the unit stands one tile away so its
# footprint reads the fire's incident radiant flux every tick.
FY, FX = 8, 8
UNIT_X, UNIT_Y = 6, 8   # one+ tile from the burner column -> sustained heat flux


def _wood_room() -> LevelData:
    """A 16x16 hull-walled room with a wood interior (so the burner sustains)."""
    h = w = 16
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:15, 1:15] = 4                   # carve interior air
    return LevelData(name="s3c_kill", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _fire_kill_sim() -> Simulation:
    """An ignite -> fire -> unit-damage -> KILL scenario: a wood burner held lit
    next to a LOW-HP marine, so the radiant heat drains the marine to death within
    the trajectory (exercising the float-HP step end-to-end to a kill event)."""
    sim = Simulation(_wood_room(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    # A patch of wood fuel around the burner so the fire sustains + spreads heat.
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            y, x = FY + dy, FX + dx
            if 1 <= y < 15 and 1 <= x < 15:
                g.material[y, x] = MAT_WOOD
    g._update_caches()
    sim.add_unit(Unit("M1", x=UNIT_X, y=UNIT_Y, team=0))
    sim.set_paused(False)
    return sim


def _drive(sim: Simulation):
    """Hold the burner lit each tick (re-seed fire before the step) so a sustained
    radiant flux pushes the marine past its tolerance band and kills it. Returns
    the captured trajectory (gmap fields + the per-tick unit-state digest)."""
    g = sim.gmap
    seed_q = fire_fixed.quantize_scalar(0.7)
    # P-R4: the emitters' flame temperature (see the hold loop below).
    flame_q = fire_fixed.quantize_scalar(443.0)
    # Set a fixed HP so the steady heat DPS drains the unit over ~15 ticks and
    # KILLS it MID-trajectory — the kill lands well inside TICKS, with HP visibly
    # decreasing each tick BEFORE death (so the float-HP step is genuinely
    # exercised, not a one-tick instakill). The number is about exercising the
    # digest, not the balance.
    # P-R4 re-anchor (ruling amendment 5 D2): the DPS moved. The retired painter
    # delivered `k_fire_heat * I` = 1600-scale energy into the AIR tiles a unit
    # stands on (~14 HP/tick here); the radiant flux a unit now feels is D3's
    # SENSOR reading `tau * w * a_s * E[T_s]` — the honest incident flux from an
    # emitter at its flame temperature — which at this geometry is ~1.4 HP/tick.
    # HP is re-scaled to keep the kill in the same PLACE in the trajectory. The
    # absolute damage number is P-R5 feel-tuning territory (the [combat]
    # heat_flux_to_temp / environmental_damage_rate dials), not this test's job.
    for u in sim.units:
        u.current_hp = 22.0

    traj = []
    from field_ab_harness import _snapshot, _capture_unit_state, SIM_FIELDS
    for _ in range(TICKS):
        # Hold the burner lit: re-light the wood patch each tick (the fire would
        # otherwise starve / blow out) so the radiant flux is sustained.
        # P-R4 re-anchor (ruling amendment 5 D2): hold the patch HOT as well as
        # LIT. The retired painter deposited `k_fire_heat * I` regardless of the
        # tile's temperature, so re-seeding `fire` alone produced radiant flux.
        # The net-T^4 exchange radiates against the emitter's OWN temperature —
        # a synthetically-lit ice-cold tile emits E[0] and correctly damages
        # nobody (measured: unit HP never dropped). In play a burning tile sits
        # at the ~440 game plateau that combustion's H_bed holds (P-R4 gate f),
        # so holding both fields is what reproduces the scenario as described.
        # The flux itself reaches the unit through D3's radiant-flux SENSOR
        # plane, which is what `apply_environmental_damage` now samples on air.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = FY + dy, FX + dx
                if 1 <= y < 15 and 1 <= x < 15:
                    g.fire[y, x] = max(int(g.fire[y, x]), seed_q)
                    g.temperature[y, x] = max(int(g.temperature[y, x]),
                                              flame_q)
        sim.set_paused(False)
        sim.step()
        snap = _snapshot(sim.gmap, SIM_FIELDS)
        snap[UNIT_DIGEST_KEY] = _capture_unit_state(sim)
        traj.append(snap)
    return traj


def test_fire_kill_unit_state_digest_bit_identical_run_twice():
    """The synced unit-state digest (HP + hit/kill events + position/footprint) is
    bit-identical across two runs of the ignite->fire->unit-damage->kill scenario.

    This catches nondeterminism leaking through the float-HP step: two runs agree
    on every gmap cell yet could disagree on who dies / how much HP / on what tick —
    THAT is what this digest gates."""
    a = _drive(_fire_kill_sim())
    b = _drive(_fire_kill_sim())

    # 1. The whole trajectory (fields AND the unit digest) is bit-identical.
    assert_trajectories_match(a, b, tol=0.0)

    # 2. The unit-state digest specifically is bit-identical EVERY tick (a sharper,
    #    named assertion than the bundled trajectory match above).
    for t in range(TICKS):
        assert unit_digest_hash(a[t]) == unit_digest_hash(b[t]), (
            f"unit-state digest diverged at tick {t}:\n  "
            + "\n  ".join(diff_trajectories([a[t]], [b[t]], tol=0.0)))


def test_scenario_actually_kills_the_unit():
    """Guard against a vacuous gate: the scenario MUST exercise a real
    fire->heat->HP-drain->kill, with HP strictly decreasing and a UnitKilledEvent
    emitted (a fire that never reaches the unit would make the digest trivially
    match without testing the float-HP path)."""
    traj = _drive(_fire_kill_sim())

    # HP draws down over the run (the float-HP step is genuinely exercised).
    def hp_of(snap):
        recs = snap[UNIT_DIGEST_KEY]["units"]
        assert recs, "no units captured — scenario is empty"
        return recs[0]["current_hp"]          # quantized int (1e-9 quantum)

    hp_start = hp_of(traj[0])
    hp_end = hp_of(traj[-1])
    assert hp_end < hp_start, (
        f"unit HP never dropped (start {hp_start} end {hp_end}) — the fire never "
        f"reached the unit; the float-HP step is NOT being exercised")

    # A heat hit stream AND a kill event are emitted somewhere in the run.
    any_hit = False
    killed = False
    for snap in traj:
        for ev in snap[UNIT_DIGEST_KEY]["events"]:
            if ev["kind"] == "UnitHitEvent" and ev["source"] == "heat":
                any_hit = True
            if ev["kind"] == "UnitKilledEvent":
                killed = True
    assert any_hit, "no heat UnitHitEvent emitted — fire never damaged the unit"
    assert killed, (
        "the unit never died (no UnitKilledEvent) — the fire->kill loop did not "
        "close; raise the burner intensity / lower the unit HP")

    # The unit ends DEAD (life/alive flips are part of the synced digest).
    final = traj[-1][UNIT_DIGEST_KEY]["units"][0]
    assert final["alive"] is False, "unit should be dead at the end of the scenario"


def test_kill_event_lands_on_the_same_tick_run_twice():
    """The DISCRETE kill event must land on the SAME tick on both runs (a 1-LSB HP
    slip would flip the death tick -> a desync the per-tick digest catches)."""
    def kill_tick(traj):
        for t, snap in enumerate(traj):
            for ev in snap[UNIT_DIGEST_KEY]["events"]:
                if ev["kind"] == "UnitKilledEvent":
                    return t
        return None

    ta = kill_tick(_drive(_fire_kill_sim()))
    tb = kill_tick(_drive(_fire_kill_sim()))
    assert ta is not None and ta == tb, (
        f"kill landed on different ticks run-to-run (a={ta} b={tb}) — a "
        f"nondeterministic float-HP drain")


if __name__ == "__main__":
    test_fire_kill_unit_state_digest_bit_identical_run_twice()
    test_scenario_actually_kills_the_unit()
    test_kill_event_lands_on_the_same_tick_run_twice()
    print("OK: S3c unit-state digest — fire->kill scenario bit-identical run-to-run "
          "(HP + hit/kill event stream watched end-to-end through the float-HP step)")
