"""The swarm kernel tables (arc #63 P3, docs/swarm_P3_impl.md v4.3): every
value a swarm kernel reads is one row of ``swarm_tables.ROWS``, read from the
species row, the launch context or ``Salt``, with its proved range; C++ reads
offsets only from the generated header.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_swarm_tables.py -q
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import _swarm_testkit as tk  # noqa: E402
from config import CFG  # noqa: E402
from simulation import swarm, swarm_fixed, swarm_tables  # noqa: E402
from simulation.rng_salts import Salt  # noqa: E402

I32 = (-(1 << 31), (1 << 31) - 1)


def test_the_generated_header_is_current():
    """The checked-in cpp/src/swarm_tables_layout.h is exactly
    swarm_tables.header_text(), and the compiled module carries its
    fingerprint.

    Must break when a row is added, removed, renamed, moved or re-ranged
    without `tools/gen_swarm_layout.py --write` (and a rebuild)."""
    path = ROOT / swarm_tables.HEADER_REL
    assert path.read_text(encoding="utf-8") == swarm_tables.header_text(), (
        f"stale {swarm_tables.HEADER_REL}: run `{swarm_tables.GENERATOR}` and commit")
    assert bp.SWARM_TABLES_FINGERPRINT == swarm_tables.fingerprint()


def test_a_stale_module_is_refused_by_name():
    """swarm_tables.check_compiled refuses a module whose fingerprint differs
    (PhysicsRunner.bind_draws runs it before any gather).

    Must break if the check is dropped or compares something else."""
    class Fake:
        SWARM_TABLES_FINGERPRINT = "0" * 64
    with pytest.raises(RuntimeError, match="SWARM_TABLES_FINGERPRINT"):
        swarm_tables.check_compiled(Fake)
    swarm_tables.check_compiled(bp)


def test_every_row_reads_the_reference_objects():
    """Every row's value comes from the species row, the launch context or
    Salt: a changed species row moves exactly the rows that read it, a
    changed tick length moves the derived rows, and the salt rows are the
    Salt members.

    Must break if a row re-derives its value from config (it would not
    follow the table in force) or reads the wrong attribute."""
    import field_ab_harness as fab
    sim = fab.swarm_scenario_sim()
    g = sim.gmap
    dt = 1.0 / 24
    base = swarm_tables.values("larva", g, dt)
    row = g.swarm_species.row("larva")
    for f in dataclasses.fields(row):
        if f.name in ("species", "max_units"):
            continue
        bumped = dataclasses.replace(row, **{f.name: getattr(row, f.name) + 1})
        g.swarm_species = swarm.SwarmSpeciesTable.from_rows([bumped])
        moved = swarm_tables.values("larva", g, dt) != base
        g.swarm_species = swarm.SwarmSpeciesTable.from_rows([row])
        names = [r.name for r, m in zip(swarm_tables.ROWS["larva"], moved) if m]
        direct = f.name in [r.name for r in swarm_tables.ROWS["larva"]]
        if direct:
            assert names == [f.name], (f.name, names)
    # the derived rows follow the tick actually stepped
    other = swarm_tables.values("larva", g, 2 * dt)
    for name in ("step_q", "p_base_q", "p_alarm_q"):
        assert swarm_tables.read("larva", other, name) != swarm_tables.read("larva", base, name)
    assert swarm_tables.read("larva", base, "salt_tumble") == int(Salt.SWARM_TUMBLE)
    assert swarm_tables.read("larva", base, "salt_wall_turn") == int(Salt.SWARM_WALL_TURN)


def test_a_config_change_reaches_the_rows_only_through_the_reload_seam(monkeypatch):
    """Editing CFG moves no row until Simulation.on_config_reload re-binds the
    table; after it, the next gather's rows carry the new value.

    Must break if a row reads CFG directly (a lazy reload)."""
    import field_ab_harness as fab
    sim = fab.swarm_scenario_sim()
    g = sim.gmap
    before = swarm_tables.values("larva", g, 1 / 24)
    monkeypatch.setattr(CFG, "swarm", tk.swarm_namespace(turn_bias=0.6))
    assert np.array_equal(swarm_tables.values("larva", g, 1 / 24), before)
    sim.on_config_reload()
    after = swarm_tables.values("larva", g, 1 / 24)
    assert swarm_tables.read("larva", after, "turn_bias_q") == swarm_fixed.quantize_scalar(0.6)


def test_every_range_is_inside_its_integer_type_and_holds_the_field_bounds():
    """Every row's range lies inside int32 (the salts: uint32); every row
    that stores a species field holds that field's quantized bounds (Kelvin
    rows: their int32 conversion), and the R1-R4 boundary cases lie inside
    the rows' ranges.

    Must break if a range is widened past what the law narrows to, or a
    field bound escapes its row (the kernel would see an unproved value)."""
    for sp, rows in swarm_tables.ROWS.items():
        for r in rows:
            if r.name.startswith("salt_"):
                assert 0 <= r.lo <= r.hi <= (1 << 32) - 1, r.name
            else:
                assert I32[0] <= r.lo <= r.hi <= I32[1], r.name
        fields = {f.name: f for f in swarm.SPECIES_FIELDS}
        for r in rows:
            fname = r.name[:-2] if r.name.endswith("_q") else None
            f = fields.get(fname)
            if f is None or fname in swarm.SPECIES_UNITS:
                continue
            for bound in (f.minimum, f.maximum):
                q = swarm_fixed.quantize_scalar(bound)
                assert r.lo <= q <= r.hi, (r.name, bound, q)
    by = {r.name: r for r in swarm_tables.ROWS["larva"]}
    one = swarm_fixed.FP_ONE
    # R1 boundary: step_q in [1, FP_ONE - 1]; R2: 0 <= p_base <= p_alarm <= FP_ONE;
    # R3: T_coma_exit = T_ctmin + 1; R4: turn_min == turn_max.
    for name, v in (("step_q", 1), ("step_q", one - 1), ("p_base_q", 0), ("p_alarm_q", one),
                    ("T_coma_exit_q", I32[0] + 1), ("turn_min_q", 0), ("turn_max_q", 0)):
        assert by[name].lo <= v <= by[name].hi, (name, v)


def test_a_value_outside_its_range_is_refused_by_name_and_nothing_launches(monkeypatch):
    """A row value outside its proved range raises ValueError naming species,
    row and value at the gather, and the swarm step launches nothing (part
    of T-P3-36).

    Must break if the range check is dropped or runs after the launch."""
    import field_ab_harness as fab
    sim = fab.swarm_scenario_sim()
    g = sim.gmap
    row = g.swarm_species.row("larva")
    bad = dataclasses.replace(row, turn_bias_q=swarm_fixed.FP_ONE + 1)
    g.swarm_species = swarm.SwarmSpeciesTable.from_rows([bad])
    c0 = bp.swarm_cpu_calls()
    snap = {c.name: swarm.swarm_column(g, "larva", c.name).copy() for c in swarm.ROSTER}
    with pytest.raises(ValueError, match=r"\[swarm.larva\] row 'turn_bias_q' = 65537"):
        sim.set_paused(False)
        sim.step()
    assert bp.swarm_cpu_calls() == c0
    for c in swarm.ROSTER:
        assert np.array_equal(swarm.swarm_column(g, "larva", c.name), snap[c.name])
