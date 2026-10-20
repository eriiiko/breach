"""#88 V3: every trace READER gives the pre-#88 answer on the rescaled field --
driven through its PUBLIC entry point, with a registry that every reading
module must join.

Design: docs/smoke_unit_rescale_design_2026-10-20.md §10 V3, §2.4; critiques
A1 (the gates move by under one old count), A7 (the sensor converts where both
sample and area see it), C2 (call sites, not conversion functions), C9 (the
renderer's one trace dequantize).

How it works. Each registered reader runs on a REAL GameMap (or the engine's
sweep through the binding) whose trace planes are seeded with ``S_old << k``
(``k = TRACE_FINE_BITS``), over a sweep of S_old that crosses each gameplay
threshold, the sweep's ONE cap and a negative cell. Then:

* an integer reader must equal the pre-#88 integer law evaluated on S_old,
  written below as the documented expression (``>> 16``, the Q16 currency);
* a float reader must be float-identical to S_old / 2^16;
* the two GATES (teargas BLINDED, the poison dose threshold) are asserted by
  their real-unit law instead (A1): ``gate(S) <=> S >= quantize_trace_scalar(th)``,
  monotone in S, since quantizing a threshold at a finer currency moves it by
  less than one old count, toward the real value.

At TRACE_FINE_BITS = 0 every assertion is an identity on today's code; the dry
flip (P1) and the flip (P3) run it at 8, where a skipped, doubled or mis-named
conversion at any registered call site fails.
"""
from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import _trace_currency_lint as lint  # noqa: E402
from _radiation_sweep_harness import ONE, R, cpp_sweep, live_table  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData, load as load_level  # noqa: E402
from simulation import gas_fixed, unit_fixed  # noqa: E402
from simulation.gases import (  # noqa: E402
    GAS_NAMES, INERT_N2, N_GASES, N_TRACE_GASES, O2, POISON, SMOKE, STEAM,
    TEARGAS, GasTable,
)

OLD_ONE = 1 << 16                      # the pre-#88 currency: Q16 per real unit


TRACE_LIFT = gas_fixed.TRACE_ONE // OLD_ONE   # 2^TRACE_FINE_BITS


def trace_lift(s_old):
    """S_old (pre-#88 Q16 counts) -> today's trace currency, exactly (a
    multiply, so a negative cell keeps its sign)."""
    return int(s_old) * TRACE_LIFT


# The S_old sweep: zero, one count, the two gameplay thresholds and their
# neighbours, thin to thick smoke, the sweep's ONE cap for the shipped smoke
# (heat_absorb ~25 -> a_gas saturates near 2600 counts), 3 units, a negative.
_TG = gas_fixed.quantize_bulk_scalar(float(CFG.exchange.teargas_blind_density))
_PO = gas_fixed.quantize_bulk_scalar(float(CFG.exchange.poison_min_density))
S_OLD = sorted({0, 1, 7, 131, 2584, 2585, 2600, 3932, 13107, 17695, 39322,
                OLD_ONE, 3 * OLD_ONE, _TG - 1, _TG, _TG + 1, _PO - 1, _PO, _PO + 1})
S_NEG = -500


def _box_gmap(h=12, w=12):
    from simulation.gamemap import GameMap
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    return GameMap(LevelData(name="v3", version="2", path=Path("."), tilemap=tm,
                             tile_size_m=1.0, diffuse_path=Path(".")))


# ---------------------------------------------------------------------------
# the registered readers
# ---------------------------------------------------------------------------
def reader_hover_readout():
    """renderer/hover_readout.py::pack_hover_readout -- the tile inspector: the
    five trace densities (floats) and, on a live engine, the effective
    extinction the sweep reads on a gas cell (the engine's FP_HD finish)."""
    from renderer.hover_readout import pack_hover_readout
    from simulation import Simulation
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    air = g._gas_energy_accountable() & ~g.thermal_solid
    ys, xs = np.nonzero(air)
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    hq = [int(v) for v in g.gases.heat_absorb_q16]
    n_bulk = int(g.gas[O2][y, x]) + int(g.gas[INERT_N2][y, x])
    a_mat = int(g.heat_atten_q[y, x])
    checked = 0
    for s_old in S_OLD + [S_NEG]:
        olds = [s_old if t == SMOKE else (s_old // 3 if t == STEAM else 0)
                for t in range(N_TRACE_GASES)]
        for t in range(N_TRACE_GASES):
            g.gas[t][y, x] = trace_lift(olds[t])
        r = pack_hover_readout(g, x, y, kelvin_fn=lambda tg: tg + 293.0)
        for t in range(N_TRACE_GASES):
            name = GAS_NAMES[t]
            if name in r.gases:
                assert r.gases[name] == olds[t] / float(OLD_ONE), (name, s_old)
        # the pre-#88 sweep law on S_old: min(ONE, sum hq * max(0, S) >> 16)
        if n_bulk >= R.N_EPS_RAW:
            a_old = min(ONE, sum(h * max(0, s) for h, s in zip(hq, olds)) >> 16)
            assert r.atten_a == pytest.approx(max(a_mat, a_old) / 65536.0, abs=0.0), s_old
        checked += 1
    return checked


def reader_sensor_accessor():
    """src/simulation/sensor_accessor.py::EntityFieldAccessor.sample / .area on
    Channel.SMOKE -- the sensor signal stays Q16 (KIND_Q16 thresholds)."""
    from simulation.sensor_accessor import Channel, EntityFieldAccessor
    g = _box_gmap()
    acc = EntityFieldAccessor(g)
    disc = [(5, 5), (5, 6), (6, 5), (6, 6)]
    checked = 0
    for s_old in S_OLD + [S_NEG]:
        olds = [s_old, s_old + 3, max(0, s_old - 7), 2 * s_old]
        for (fy, fx), so in zip(disc, olds):
            g.gas[SMOKE][fy, fx] = trace_lift(so)
        assert acc.sample(Channel.SMOKE, 5, 5) == s_old
        assert acc.area(Channel.SMOKE, disc) == (sum(olds), len(disc))
        checked += 1
    return checked


def _unit_on(g, x=5, y=5):
    from simulation.unit import Unit
    u = Unit("M", x=x, y=y, team=0)
    u.id = 1
    u.current_hp = 1000.0
    return u


def reader_exchange_gates():
    """src/simulation/exchange.py::apply_teargas_blind / apply_poison_dose --
    gated by their REAL-UNIT law (A1); the poison amount is float-identical."""
    from simulation.exchange import apply_poison_dose, apply_teargas_blind
    from simulation.status import BLINDED
    tps = int(CFG.clock.ticks_per_second)
    checked = 0
    for gas_id, th_real in ((TEARGAS, float(CFG.exchange.teargas_blind_density)),
                            (POISON, float(CFG.exchange.poison_min_density))):
        th = gas_fixed.quantize_trace_scalar(th_real)
        assert abs(th / gas_fixed.TRACE_ONE - th_real) <= 0.5 / gas_fixed.TRACE_ONE
        prev = False
        for s in sorted({0, th - 2, th - 1, th, th + 1, th + 2, 2 * th, trace_lift(_TG), trace_lift(_PO)}):
            g = _box_gmap(16, 16)
            u = _unit_on(g)
            for (tx, ty) in u.occupied_tiles():
                g.gas[gas_id][ty, tx] = s
            if gas_id == TEARGAS:
                apply_teargas_blind([u], g)
                fired = any(st.kind == BLINDED for st in getattr(u, "statuses", []))
            else:
                events = []
                apply_poison_dose([u], g, tps, events=events)
                fired = u.current_hp < 1000.0
                if fired:
                    want = unit_fixed.quantize_hp_delta(
                        float(CFG.exchange.poison_dps) * (s / gas_fixed.TRACE_ONE) / tps)
                    assert 1000.0 - u.current_hp == want, s
            assert fired == (s >= th), (GAS_NAMES[gas_id], s, th)
            assert fired >= prev, "the gate is not monotone in S"
            prev = fired
            checked += 1
    # the poison amount on S_old << k is float-identical to the pre-#88 law
    for s_old in (_PO, _PO + 1, OLD_ONE, 3 * OLD_ONE):
        g = _box_gmap(16, 16)
        u = _unit_on(g)
        for (tx, ty) in u.occupied_tiles():
            g.gas[POISON][ty, tx] = trace_lift(s_old)
        apply_poison_dose([u], g, tps, events=[])
        want = unit_fixed.quantize_hp_delta(
            float(CFG.exchange.poison_dps) * (s_old / 65536.0) * (1.0 / tps))
        assert 1000.0 - u.current_hp == want, s_old
        checked += 1
    return checked


def reader_fire_beam():
    """src/simulation/combat.py::fire_beam -- the laser's integer Beer-Lambert
    over a tile of smoke and steam: the hit equals the pre-#88 per-term law."""
    from simulation.combat import BEAM_MIN_ENERGY_Q16, fire_beam
    from simulation.events import UnitHitEvent
    from simulation.unit import Unit
    from simulation.weapons import get_tables
    t = get_tables()
    weapon, ammo = t.weapons.by_name["lance_3"], t.ammo.by_name["cell_laser_standard"]
    checked = 0
    for s_old in S_OLD + [S_NEG]:
        g = _box_gmap(20, 30)
        absorb = g.gases.beam_absorb_q16
        olds = {SMOKE: s_old, STEAM: max(0, s_old // 2)}
        for gid, so in olds.items():
            g.gas[gid, 9, 6] = trace_lift(so)
        shooter = Unit("S", x=2, y=8, team=0)
        z = Unit("Z", x=10, y=8, team=1)
        z.current_hp = 1e9
        shooter.id, z.id = 1, 2
        events = []
        fire_beam(g, [shooter, z], shooter, 3, 9, 25, 9, tick=0, shots=[],
                  real_time=0.0, rng=np.random.default_rng(1), events=events,
                  weapon=weapon, ammo=ammo, spread_deg=0.0)
        # the pre-#88 law on S_old (combat.fire_beam before #88, per term >> 16)
        energy = int(ammo.damage) << 16
        total = sum((int(absorb[gid]) * so) >> 16 for gid, so in olds.items() if so > 0)
        if total > 0:
            trans = OLD_ONE - total
            energy = 0 if trans <= 0 else (energy * trans) >> 16
        hits = [e.damage for e in events if isinstance(e, UnitHitEvent)]
        if energy < BEAM_MIN_ENERGY_Q16:
            assert hits == [], s_old
        else:
            assert hits == [float((energy + (OLD_ONE >> 1)) >> 16)], (s_old, hits)
        checked += 1
    return checked


def reader_recorder():
    """src/simulation/recorder.py::PhysicsRecorder.record -- the frozen .npz
    contract: smoke in REAL units, float-identical."""
    from simulation.recorder import PhysicsRecorder
    g = _box_gmap()
    h, w = g.material.shape
    rec = PhysicsRecorder(h, w, capacity=4, fields=("smoke",))
    olds = np.array(S_OLD[:h * w] + [0] * max(0, h * w - len(S_OLD)), dtype=np.int64)[:h * w]
    olds = olds.reshape(h, w)
    olds[1, 1] = S_NEG
    g.gas[SMOKE][:] = olds * TRACE_LIFT
    rec.record(g, 0, 0.0, [])
    got = np.asarray(rec.buffers["smoke"][0])
    want = (olds.astype(np.float64) / 65536.0).astype(got.dtype)
    assert np.array_equal(got, want)
    return int(olds.size)


def reader_renderer_trace_floats():
    """renderer/game_renderer.py::_trace_floats -- THE renderer's one trace
    dequantize: float32 real densities, float-identical."""
    from renderer.game_renderer import _trace_floats
    g = _box_gmap()
    h, w = g.material.shape
    rng = np.random.default_rng(88)
    olds = rng.choice(np.asarray(S_OLD + [S_NEG], dtype=np.int64), size=(N_TRACE_GASES, h, w))
    for t in range(N_TRACE_GASES):
        g.gas[t][:] = olds[t] * TRACE_LIFT
    got = _trace_floats(g)
    want = (olds.astype(np.float64) / 65536.0).astype(np.float32)
    assert got.dtype == np.float32 and np.array_equal(got, want)
    return int(olds.size)


def reader_engine_sweep():
    """The engine's radiation sweep through the binding: the heat term a_eff
    (min(ONE, Σ hq·S >> 16)), the light term per channel (ONE − exp(−Σ lab·S
    >> 16)) and the glow coefficient (min(ONE, Σ lgl·S >> 16)) on gas cells,
    with the SHIPPED gas columns, equal to the pre-#88 law on S_old."""
    tbl = GasTable.from_config()
    hq = [int(v) for v in tbl.heat_absorb_q16]
    lab = tbl.light_absorb_q16.tolist()
    lgl = tbl.light_glow_q16.tolist()
    vals = S_OLD + [S_NEG]
    h, w = 4, len(vals)
    old = np.zeros((N_GASES, h, w), dtype=np.int64)
    for x, s in enumerate(vals):
        old[SMOKE, :, x] = s
        old[STEAM, 1, x] = max(0, s // 2)
        old[POISON, 2, x] = max(0, s // 3)
        old[TEARGAS, 3, x] = s
    old[O2] = 13763
    old[INERT_N2] = 51773
    old[O2, 0, 0] = old[INERT_N2, 0, 0] = 0               # one cell below N_EPS
    planes = [[[int(v) * TRACE_LIFT if g < N_TRACE_GASES else int(v)
                for v in row] for row in old[g]] for g in range(N_GASES)]
    n_bulk = (old[O2] + old[INERT_N2]).tolist()
    zero = [[0] * w for _ in range(h)]
    light = dict(la=[zero] * 3, ld=[zero] * 3, light_absorb_q=lab, light_glow_q=lgl)
    got = cpp_sweep(zero, zero, 0, [[300 << 16] * w for _ in range(h)], 3, zero,
                    table=live_table(), gas=planes, hq=hq, n_bulk=n_bulk, light=light)
    sweep = got[5]
    a_eff = np.asarray(sweep.a_eff_plane())
    la_eff = np.moveaxis(np.asarray(sweep.light_a_eff_plane()), -1, 0)
    gco = np.moveaxis(np.asarray(sweep.light_gcoef_plane()), -1, 0)
    checked = 0
    for y in range(h):
        for x in range(w):
            dens = [int(old[g, y, x]) for g in range(N_GASES)]
            live = n_bulk[y][x] >= R.N_EPS_RAW
            s_heat = sum(c * max(0, d) for c, d in zip(hq, dens))
            want = min(ONE, s_heat >> 16) if live else 0
            assert int(a_eff[y, x]) == want, (y, x)
            for c in range(3):
                s_l = sum(lab[g][c] * max(0, dens[g]) for g in range(N_GASES))
                s_g = sum(lgl[g][c] * max(0, dens[g]) for g in range(N_GASES))
                want_l = (ONE - R.exp_neg_q16(s_l >> 16)) if live else 0
                want_g = min(ONE, s_g >> 16) if live else 0
                assert int(la_eff[c, y, x]) == want_l, (c, y, x)
                assert int(gco[c, y, x]) == want_g, (c, y, x)
            checked += 1
    assert int(a_eff.max()) == ONE and int(a_eff[a_eff > 0].min()) < ONE, \
        "the sweep never reached the ONE cap or never stayed below it"
    return checked


# The REGISTRY: every src/ and renderer/ module that reads a trace slice, with
# the public entry point(s) this file drives -- or, for a module that only
# MOVES trace (scale-free by construction, §2.5) or hands it on unread, the
# reason it is not a reader. The subset lint below fails until a new module is
# classified here; the table may grow freely.
REGISTRY = {
    "renderer/hover_readout.py": reader_hover_readout,
    "renderer/game_renderer.py": reader_renderer_trace_floats,
    "src/simulation/sensor_accessor.py": reader_sensor_accessor,
    "src/simulation/exchange.py": reader_exchange_gates,
    "src/simulation/combat.py": reader_fire_beam,
    "src/simulation/recorder.py": reader_recorder,
    "src/simulation/physics_runner.py": reader_engine_sweep,   # hands gas to the sweep
}
NOT_READERS = {
    "src/simulation/gamemap.py": "movers only: seal/unseal divmod, inject, the "
                                 "priced-by-air extract -- scale-free (§2.5)",
    "src/simulation/field_edit.py": "the trace WRITE door (V4: test_trace_currency)",
    "src/debug_keys.py": "a writer (one unit in the plane's own currency)",
}


@pytest.mark.parametrize("module", sorted(REGISTRY))
def test_a_registered_reader_gives_the_pre_88_answer_on_the_rescaled_field(module):
    """PROPERTY (V3): the registered reader, driven through its public entry
    point on S_old << TRACE_FINE_BITS, returns what the pre-#88 law returns on
    S_old (integer readers bit-identical, float readers float-identical, gates
    by their real-unit law), over thresholds, the ONE cap and a negative cell.

    BREAKS IF: the call site skips its conversion (reads trace counts as Q16),
    doubles it, or names the wrong currency (a bulk name on a trace slice).
    """
    n = REGISTRY[module]()
    assert n > 0, f"{module}: the reader checked nothing"


_READ_RE = lint.TRACE_SLICE


def _code_only(path):
    text = path.read_text(encoding="utf-8")
    code, _raw = lint._code_lines(text)
    return "\n".join(code)


def test_every_module_that_reads_a_trace_slice_is_registered():
    """PROPERTY (V3 subset lint, C2): every src/ and renderer/ module whose
    CODE (comments and strings stripped) indexes a trace slice is in the
    registry -- as a reader driven above, or classified with a reason as a
    scale-free mover / writer. A classification check, not a count.

    BREAKS IF: a new trace reader lands without being registered (it would be
    invisible to every gate at 0 bits and wrong after the flip).
    """
    found = set()
    for d in ("src", "renderer"):
        for p in sorted((ROOT / d).rglob("*.py")):
            if "__pycache__" in str(p):
                continue
            if _READ_RE.search(_code_only(p)):
                found.add(str(p.relative_to(ROOT)).replace("\\", "/"))
    known = set(REGISTRY) | set(NOT_READERS)
    missing = sorted(found - known)
    assert not missing, ("modules reading a trace slice but not in the V3 "
                         f"registry: {missing}")
