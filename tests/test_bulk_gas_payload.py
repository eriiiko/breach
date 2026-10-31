"""The BULK-GAS PAYLOAD EFFECT (levelgen vocabulary E.2 row 2,
docs/levelgen_zone_vocabulary_2026-10-29.md; #133).

A ``[payloads.*]`` row with ``bulk_gas = "o2" | "inert_n2"`` injects that ONE
conservative bulk gas, ``bulk_amount`` bulk N per tile (1.0 = a whole tile of
ambient air), FLAT over the open tiles of a ``bulk_radius`` disc, through THE
gas pump primitive ``GameMap.inject_gas_n_vec`` -- booked through the gas
energy seam in ``payload_bulk_gas`` (``payloads.BULK_GAS_BOOKS``). It is the
only honest path to O2 / N2: the FieldEdit ``gas`` policy refuses a
conservative slice, which is why ``gas_species`` cannot do it.

The shipped rows: ``o2_store_rupture`` (B.3, the medbay O2 store: pure O2, a
fire next door roars) and ``halon_dump`` (A.6, the server farm: inert N2, the
fire dies). Their values are PLACEHOLDERS; these tests assert the properties
that must survive any retune within the rows' stated intent.

Each test names the property it protects and the change that must break it.
Fires are started by DELIVERING HEAT (``payloads.deposit_heat`` with
``IGNITE_HEAT`` on a wood tile, then ``apply_temperature_ignition`` lights
it) -- never by writing ``fire`` alone (CLAUDE.md "Starting a fire").

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_bulk_gas_payload.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.events import ExplosionEvent  # noqa: E402
from simulation.gases import INERT_N2, O2  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL, MAT_WOOD  # noqa: E402
from simulation.payloads import (BULK_GAS_BOOKS,  # noqa: E402
                                 BULK_GAS_RAIL_BOOKS, IGNITE_HEAT,
                                 bulk_gas_tiles, deposit_heat,
                                 execute_payload)
from simulation.timed_charge_system import EVENT_KIND  # noqa: E402
from simulation.weapons import PayloadTable  # noqa: E402

TPS = int(CFG.clock.ticks_per_second)
HH = 11                     # 9x9 open interior: 81 tiles, under halon_dump's
CENTRE = (HH // 2, HH // 2)  # ~110-tile mixed-room budget (config comment)
FIRE_AT = (HH // 2, HH // 2 + 1)
IGNITE_STEP = 2             # the step whose start carries the igniting heat
PAYLOAD_STEP = 40           # the fire is established by then


def _room(hh=HH, wood=True, entities=()):
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    if wood:
        tm[FIRE_AT] = MAT_WOOD
    return LevelData(name="bulk_gas", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[])


def _sim(wood=True, entities=(), hh=HH, seed=3):
    return Simulation(_room(hh=hh, wood=wood, entities=entities), seed=seed,
                      breach_physics=bp, enable_recorder=False)


def _step(sim):
    sim.set_paused(False)            # the ruleset auto-pauses at seams
    sim.step()


def _payload(sim, name):
    return sim.weapons_tables.payloads.by_name[name]


def _fire(sim, payload_name, at, n_steps, on_step=None):
    """Light the wood tile by delivering heat at IGNITE_STEP, execute
    ``payload_name`` (None = control) at ``at`` before PAYLOAD_STEP, and step
    ``n_steps``. ``on_step(i, sim)`` is called after every step."""
    g = sim.gmap
    for i in range(n_steps):
        if i == IGNITE_STEP:
            deposit_heat(g, sim.edit_queue, FIRE_AT[0], FIRE_AT[1],
                         IGNITE_HEAT, 1.0)
        if i == PAYLOAD_STEP and payload_name is not None:
            execute_payload(g, sim.edit_queue, sim.units, at[0], at[1],
                            _payload(sim, payload_name), sim.rng,
                            events=sim.tick_events)
        _step(sim)
        if on_step is not None:
            on_step(i, sim)


def _x_o2(g, y, x):
    o2, n2 = int(g.gas[O2][y, x]), int(g.gas[INERT_N2][y, x])
    return o2 / (o2 + n2)


def _neighbour_x_o2(g, y, x):
    """The sustain law's X: Σ O2 / Σ bulk over the open 4-neighbours."""
    o2 = n = 0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = y + dy, x + dx
        if g.solid[yy, xx] or g.is_vacuum[yy, xx]:
            continue
        o2 += int(g.gas[O2][yy, xx])
        n += int(g.gas[O2][yy, xx]) + int(g.gas[INERT_N2][yy, xx])
    return o2 / n


# ---------------------------------------------------------------------------
# The effect
# ---------------------------------------------------------------------------

def test_o2_store_rupture_enriches_the_disc_and_nothing_else():
    """Property: ``o2_store_rupture`` adds exactly ``bulk_amount_q`` O2
    counts to every open disc tile and nothing to its N2 -- so the O2 mole
    fraction RISES, the inert fraction FALLS and total bulk N RISES there --
    while hull tiles inside the disc radius and every tile outside the disc
    are untouched, and gas born at ambient leaves an ambient room's
    temperature mirror at exactly ambient. Breaks if the effect goes back to
    ``inject_gas_n``'s fixed 21/79 split (the N2 slice would move and the
    fraction would not), if it scales by a falloff, if it deposits into walls,
    or if it credits energy at anything but the deposit's T."""
    sim = _sim(wood=False)
    g = sim.gmap
    _step(sim)
    pl = _payload(sim, "o2_store_rupture")
    at = (2, 2)                              # the disc overlaps the hull
    tiles = set(bulk_gas_tiles(g, at[0], at[1], pl.bulk_radius))
    assert tiles, "vacuous: the disc covers no open tile"
    o2_0 = g.gas[O2].astype(np.int64).copy()
    n2_0 = g.gas[INERT_N2].astype(np.int64).copy()
    t_0 = g.temperature.astype(np.int64).copy()
    execute_payload(g, sim.edit_queue, sim.units, at[0], at[1], pl, sim.rng)
    o2_1 = g.gas[O2].astype(np.int64)
    n2_1 = g.gas[INERT_N2].astype(np.int64)
    hit = np.zeros(g.solid.shape, dtype=bool)
    for r, c in tiles:
        hit[r, c] = True
        x0 = o2_0[r, c] / (o2_0[r, c] + n2_0[r, c])
        x1 = o2_1[r, c] / (o2_1[r, c] + n2_1[r, c])
        assert x1 > x0 and (1 - x1) < (1 - x0), (r, c, x0, x1)
        assert o2_1[r, c] + n2_1[r, c] > o2_0[r, c] + n2_0[r, c]
    assert np.array_equal(o2_1[hit] - o2_0[hit],
                          np.full(int(hit.sum()), pl.bulk_amount_q))
    assert np.array_equal(n2_1, n2_0), "the N2 slice moved: not pure O2"
    assert np.array_equal(o2_1[~hit], o2_0[~hit]), "O2 landed off the disc"
    assert not (hit & g.solid).any(), "the disc deposited into a wall"
    assert np.array_equal(g.temperature.astype(np.int64), t_0), (
        "gas born at ambient moved an ambient room's temperature mirror")
    assert g.gas_energy_books.get(BULK_GAS_BOOKS, 0) > 0


def test_halon_dump_puts_the_fire_out_and_the_control_keeps_burning():
    """Property: ``halon_dump`` fired beside an established wood fire drives
    the O2 mole fraction the fire reads (its open 4-neighbours) below the
    sustain law's ``o2_frac_ext`` and KEEPS it there, and the fire goes out
    (``fire == 0``) within 1800 ticks (measured ~1450 at the placeholder row),
    while the identical room without the dump is still burning at 1800 and
    burned more of its fuel. Breaks if the dump injects air instead of pure
    N2, if its amount is retuned below what the room needs (the config
    comment's derivation), or if the injected N2 is lost (the fraction would
    recover above the wall)."""
    n = 1800
    x_ext = float(CFG.physics.fire.o2_frac_ext)
    xs, out_at = [], []

    def watch(i, sim):
        if i >= PAYLOAD_STEP:
            xs.append(_neighbour_x_o2(sim.gmap, *FIRE_AT))
            if not out_at and int(sim.gmap.fire[FIRE_AT]) == 0:
                out_at.append(i)

    halon = _sim()
    _fire(halon, "halon_dump", CENTRE, n, watch)
    control = _sim()
    burning = []
    _fire(control, None, CENTRE, n,
          lambda i, s: burning.append(int(s.gmap.fire[FIRE_AT])))

    assert burning[PAYLOAD_STEP - 1] > 0, "vacuous: the fire never lit"
    assert max(xs) < x_ext, (
        f"the dump left the fire's O2 fraction at {max(xs):.4f}, not below "
        f"o2_frac_ext {x_ext}")
    assert out_at, f"the halon fire is still burning after {n} ticks"
    assert burning[-1] > 0, "the control fire went out too -- no contrast"
    assert (int(halon.gmap.wall_hp[FIRE_AT])
            > int(control.gmap.wall_hp[FIRE_AT])), (
        "the smothered fire burned as much fuel as the control")


def test_o2_store_rupture_intensifies_a_nearby_burn():
    """Property (the payload-driven twin of test_eos_p4_combustion's
    test_e2e_3_o2_rich_pocket_intensifies_burn): an ``o2_store_rupture``
    beside an established fire makes it burn VISIBLY harder than the same
    fire in ambient air -- a higher peak intensity and more fuel consumed
    over the same ticks. Breaks if the effect stops enriching (an air
    inject, an N2 inject, a deposit the EOS loses), or if the O2 sustain law
    stops reading enrichment."""
    n = 300
    peaks = {}
    hp = {}
    for name in ("o2_store_rupture", None):
        sim = _sim()
        trace = []
        _fire(sim, name, CENTRE, n,
              lambda i, s, t=trace: t.append(float(s.gmap.fire[FIRE_AT])))
        peaks[name] = max(trace[PAYLOAD_STEP:]) / 65536.0
        hp[name] = int(sim.gmap.wall_hp[FIRE_AT])
    assert peaks["o2_store_rupture"] > 1.3 * peaks[None], peaks
    assert hp["o2_store_rupture"] < hp[None], hp


# ---------------------------------------------------------------------------
# The books
# ---------------------------------------------------------------------------

def _gas_books(g):
    return int(g.gas_energy[g._gas_energy_accountable()].astype(object).sum())


def _terms(g, eos, tsolver, comb, engine):
    """The full ledger's right-hand side, test_thermostat_books' shape: EOS
    (absolute per step), thermal solver gas side, combustion, the Python
    seams (where this effect books), water evac (absolute), the solid side."""
    return (
        int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
        - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
        + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
        + int(eos.e_rail_sum),
        int(tsolver.e_gas_deposit_sum) + int(tsolver.e_gas_cond_sum)
        + int(tsolver.e_gas_rail_sum),
        -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
        + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
        int(g.gas_energy_seam_net()),
        -int(engine.e_water_evac_export_sum),
        int(tsolver.e_solid_deposit_sum) + int(tsolver.e_solid_cond_sum)
        + int(comb.e_comb_solid_heat_sum)
        + int(tsolver.e_solid_low_rail_sum) + int(tsolver.e_solid_seam_sum),
    )


def _closure_run(sim, n, before_step=None):
    """Step ``n`` times; return the ticks on which the total ledger did not
    close exactly (int, Python ints throughout)."""
    g = sim.gmap
    rp = sim.physics_runner
    eos, tsolver, comb, engine = (rp.eos, rp.engine.temperature,
                                  rp.combustion, rp.engine)
    prev_total = _gas_books(g) + int(tsolver.solid_energy_books_sum)
    prev = _terms(g, eos, tsolver, comb, engine)
    bad = []
    for i in range(n):
        if before_step is not None:
            before_step(i, sim)
        _step(sim)
        total = _gas_books(g) + int(tsolver.solid_energy_books_sum)
        t = _terms(g, eos, tsolver, comb, engine)
        expected = (t[0] + (t[1] - prev[1]) + (t[2] - prev[2])
                    + (t[3] - prev[3]) + t[4] + (t[5] - prev[5]))
        resid = (total - prev_total) - expected
        if resid:
            bad.append((i, resid))
        prev_total, prev = total, t
    return bad


def _timer_charge(payload, first_at_s, eid="store"):
    fields = {f.name: f.default for f in REGISTRY["timed_charge"].FIELDS}
    fields.update(x=CENTRE[1], y=CENTRE[0], payload=payload,
                  first_at_s=first_at_s, period_s=0.0, enabled=True)
    return EntityInstance(id=eid, class_name="timed_charge", ordinal=0,
                          tags=(), fields=fields)


def test_closure_identity_holds_across_ticks_with_the_payload_firing_mid_run():
    """Property: with a fire burning in a sealed room and a timed charge
    executing ``o2_store_rupture`` in slot 9e mid-run, the across-tick
    closure identity ``Δ(Σ gas_energy + solid books) == EOS + thermal +
    combustion + seams + water`` is EXACT in int64 on every tick, the
    payload's energy is booked in its own seam channel and in no pump
    channel. Breaks if the effect writes ``gas`` or ``gas_energy`` outside
    the seam, books into a channel ``gas_energy_seam_net`` filters out, or
    re-uses the vent's ``pump_inject_vec`` name."""
    fire_s = 2.0
    sim = _sim(entities=[_timer_charge("o2_store_rupture", fire_s)])
    fired = []

    def ignite(i, s):
        if i == IGNITE_STEP:
            deposit_heat(s.gmap, s.edit_queue, FIRE_AT[0], FIRE_AT[1],
                         IGNITE_HEAT, 1.0)

    g = sim.gmap
    n = 4 * TPS
    bad = []
    books = []
    for chunk in range(n):
        bad += [(chunk, r) for _i, r in _closure_run(
            sim, 1, before_step=(lambda i, s, c=chunk: ignite(c, s)))]
        if any(isinstance(e, ExplosionEvent) and e.kind == EVENT_KIND
               for e in sim.tick_events):
            fired.append(chunk)
        books.append(g.gas_energy_books.get(BULK_GAS_BOOKS, 0))
    assert len(fired) == 1, f"the charge fired on steps {fired}"
    assert books[fired[0] - 1] == 0 and books[fired[0]] > 0, (
        "the payload's energy is not booked on the step it fired")
    assert int(g.fire[FIRE_AT]) > 0, "vacuous: no fire burned"
    assert not [k for k in g.gas_energy_books if k.startswith("pump")], (
        "the payload booked into a pump channel")
    assert not bad, f"closure broken on (step, residual) {bad[:5]}"


def test_closure_identity_counts_a_railed_hot_deposit():
    """Property: a bulk deposit hot enough to bind the high rail is a COUNTED
    destruction (``payload_bulk_gas_rail``), and the identity still closes
    exactly across the ticks around it. Breaks if the rail clamp writes
    ``gas_energy`` without booking it, or books it to the vent's
    ``pump_rail``."""
    t_max = float(CFG.physics.thermal.T_MAX_PHYS)
    table = PayloadTable({"hot_o2": {"bulk_gas": "o2", "bulk_amount": 2.0,
                                     "bulk_radius": 2.0,
                                     "bulk_temp": 4.0 * t_max}})
    pl = table.by_name["hot_o2"]
    sim = _sim(wood=False)

    def fire(i, s):
        if i == 5:
            execute_payload(s.gmap, s.edit_queue, s.units, CENTRE[0],
                            CENTRE[1], pl, s.rng)

    bad = _closure_run(sim, 20, before_step=fire)
    books = sim.gmap.gas_energy_books
    assert books.get(BULK_GAS_RAIL_BOOKS, 0) < 0, books
    assert "pump_rail" not in books
    assert not bad, f"closure broken on (step, residual) {bad[:5]}"


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row, needle", [
    ({"bulk_gas": "smoke", "bulk_amount": 0.2, "bulk_radius": 2.0},
     "TRACE gas"),
    ({"bulk_gas": "argon", "bulk_amount": 0.2, "bulk_radius": 2.0},
     "not a bulk gas"),
    ({"bulk_gas": "o2", "bulk_amount": -0.2, "bulk_radius": 2.0},
     "bulk_amount"),
    ({"bulk_gas": "o2", "bulk_amount": 0.0, "bulk_radius": 2.0},
     "bulk_amount"),
    ({"bulk_gas": "o2", "bulk_amount": 1e-9, "bulk_radius": 2.0},
     "quantizes to 0"),
    ({"bulk_gas": "o2", "bulk_amount": 0.2, "bulk_radius": 0.0},
     "bulk_radius"),
    ({"bulk_gas": "o2", "bulk_amount": float("nan"), "bulk_radius": 2.0},
     "bulk_amount"),
    ({"bulk_amount": 0.2, "bulk_radius": 2.0}, "without bulk_gas"),
    ({"gas_species": "o2", "gas_amount": 1.0, "gas_radius": 2},
     "conservative"),
    ({"gas_species": "inert_n2", "gas_amount": 1.0, "gas_radius": 2,
      "bulk_gas": "inert_n2", "bulk_amount": 0.2, "bulk_radius": 2.0},
     "conservative"),
])
def test_the_door_refuses_a_malformed_bulk_row(row, needle):
    """Property: the ``[payloads.*]`` door refuses, at load and by name, a
    trace gas in ``bulk_gas`` (that is ``emit_gas``'s), an unknown gas, a
    non-positive / non-finite / sub-LSB amount, a non-positive radius,
    orphan bulk columns, and a bulk gas in ``gas_species`` (one path per gas:
    FieldEdit would refuse it at the flush, mid-match). Breaks if any of
    those checks is dropped."""
    with pytest.raises(ValueError, match=needle):
        PayloadTable({"bad": row})


def test_the_door_quantizes_once_and_a_plain_row_carries_no_bulk_gas():
    """Property: a valid bulk row is quantized ONCE at the door (the amount
    in the bulk currency, the temperature in Q16 game degrees), and a row
    without the columns has no bulk effect. Breaks if the door stops
    quantizing (the executor would read a float) or defaults a bulk gas."""
    from simulation import gas_fixed
    t = PayloadTable({"o2": {"bulk_gas": "o2", "bulk_amount": 0.21,
                             "bulk_radius": 3.0, "bulk_temp": 25.0},
                      "plain": {"radius": 2, "pressure": 1.0}})
    o2 = t.by_name["o2"]
    assert o2.bulk_amount_q == gas_fixed.quantize_bulk_scalar(0.21)
    assert o2.bulk_temp_q == gas_fixed.quantize_q16_scalar(25.0)
    assert t.by_name["plain"].bulk_gas == ""
    shipped = PayloadTable(CFG.payloads).by_name
    assert shipped["o2_store_rupture"].bulk_gas == "o2"
    assert shipped["halon_dump"].bulk_gas == "inert_n2"


# ---------------------------------------------------------------------------
# B.3 end to end: the medbay O2 store
# ---------------------------------------------------------------------------

def test_a_heated_o2_store_ruptures_once_and_enriches_the_room():
    """Property: a ``trigger = "heat"`` timed charge carrying
    ``o2_store_rupture`` stays inert in an ambient room, fires exactly once
    after heat is delivered to its tile, and the O2 mole fraction on its
    tile rises above ambient -- the B.3 medbay O2 store, headless. Breaks if
    the heat trigger and the bulk effect stop composing (a charge whose
    payload has no explosion part must still run the bulk column), or if the
    store re-fires."""
    fields = {f.name: f.default for f in REGISTRY["timed_charge"].FIELDS}
    fields.update(x=CENTRE[1], y=CENTRE[0], payload="o2_store_rupture",
                  first_at_s=0.0, period_s=0.0, enabled=True,
                  trigger="heat", trigger_temp=50.0)
    sim = _sim(wood=False, entities=[EntityInstance(
        id="o2_store", class_name="timed_charge", ordinal=0, tags=(),
        fields=fields)])
    g = sim.gmap
    fired = []
    x_before = None
    for i in range(3 * TPS):
        if i == 10:
            x_before = _x_o2(g, *CENTRE)
            deposit_heat(g, sim.edit_queue, CENTRE[0], CENTRE[1], 3000.0, 1.5)
        _step(sim)
        if any(isinstance(e, ExplosionEvent) and e.kind == EVENT_KIND
               for e in sim.tick_events):
            fired.append(i)
    assert len(fired) == 1 and fired[0] >= 10, fired
    assert x_before == pytest.approx(0.21, abs=1e-3)
    assert _x_o2(g, *CENTRE) > x_before + 0.05, (
        f"the store ruptured but the O2 fraction is {_x_o2(g, *CENTRE):.4f}")
