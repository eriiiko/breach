"""The swarm step's C++ twins (arc #63 P3, docs/swarm_P3_impl.md §11
T-P3-18, 19, 21, 31; v4.3).

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_swarm_constants.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from simulation import swarm, swarm_fixed, swarm_larva, swarm_tables  # noqa: E402
from simulation.rng_salts import Salt  # noqa: E402


def test_the_cpp_twins_are_identical():
    """T-P3-18: every bp.SWARM_CONSTANTS entry with a Python source equals it;
    the roster's C type sizes equal the ROSTER itemsizes; the compiled
    SWARM_TABLES_FINGERPRINT equals swarm_tables.fingerprint();
    bp.SWARM_SPECIES_LAWS names equal SWARM_SPECIES in order, and each entry's
    table width and field / field_reduction are the species' (the larva:
    swarm_larva.FIELD / FIELD_REDUCTION).

    Must break on a drifted constant, a row added on one side only, or a
    registry reorder."""
    C = bp.SWARM_CONSTANTS
    for name in ("AI_DEAD", "AI_RUN", "AI_EAT", "AI_COMA", "FIRST_UNIT_ID", "ENV_TICK",
                 "ENV_SEED_LO", "ENV_SEED_HI", "ENV_WORDS", "DEATH_UNIT_ID", "DEATH_POS_X",
                 "DEATH_POS_Y", "DEATH_HEADING", "DEATH_WORDS", "ROW_SPECIES", "ROW_ENV",
                 "ROW_SLOT", "ROW_UNIT_ID", "ROW_POS_X", "ROW_POS_Y", "ROW_HEADING",
                 "ROW_WORDS"):
        assert C[name] == getattr(swarm, name), name
    assert C["PI_Q16"] == swarm_fixed.PI_Q16
    assert C["HEADING_PERIOD_Q16"] == swarm_fixed.HEADING_PERIOD_Q16
    assert C["N_COLUMNS"] == len(swarm.ROSTER)
    for col in swarm.ROSTER:
        assert C["ROSTER_SIZES"][col.name] == np.dtype(col.dtype).itemsize, col.name
    assert bp.SWARM_TABLES_FINGERPRINT == swarm_tables.fingerprint()
    assert [d["name"] for d in bp.SWARM_SPECIES_LAWS] == list(swarm.SWARM_SPECIES)
    for d in bp.SWARM_SPECIES_LAWS:
        assert d["table_width"] == swarm_tables.width(d["name"])
    larva = bp.SWARM_SPECIES_LAWS[swarm.SWARM_SPECIES.index("larva")]
    assert (larva["field"], larva["field_reduction"]) == (swarm_larva.FIELD,
                                                         swarm_larva.FIELD_REDUCTION)
    assert tuple(larva["branches"]) == tuple(bp.SWARM_LARVA_BRANCHES)


def test_the_heading_wrap_twin():
    """T-P3-19: bp.swarm_wrap_heading_q16(h) == swarm_fixed.wrap_heading_q16(h)
    on P2's T17 inputs plus 10^5 seeded int64 values in [-2^33, 2^33].

    Must break on a truncating `%` or an int32 intermediate."""
    P = swarm_fixed.PI_Q16
    fixed = [P, -P, 3 * P, -3 * P, 411774, 411775, 0, 1, -1, -(1 << 31), (1 << 31) - 1,
             0, 411775 - 1, 2 ** 32 - 1, -(2 ** 33), 2 ** 33]
    rng = np.random.default_rng(19)
    vals = fixed + [int(v) for v in rng.integers(-(2 ** 33), 2 ** 33 + 1, size=100_000)]
    for v in vals:
        assert bp.swarm_wrap_heading_q16(v) == swarm_fixed.wrap_heading_q16(v), v


def test_the_swarm_salts_are_salt_members(monkeypatch):
    """T-P3-21: Salt.SWARM_TUMBLE and Salt.SWARM_WALL_TURN exist, are distinct
    from every other member, are the values the table rows carry, and
    engine_args packs exactly those (spy on the gather).

    Must break on a literal salt, a salt shared with another site, or a
    renumbered member."""
    names = [s.name for s in Salt]
    values = [int(s) for s in Salt]
    assert len(set(values)) == len(values)
    assert "SWARM_TUMBLE" in names and "SWARM_WALL_TURN" in names
    rows = swarm_tables.ROWS["larva"]
    by = {r.name: r for r in rows}
    src = None
    assert by["salt_tumble"].read(src) == int(Salt.SWARM_TUMBLE)
    assert by["salt_wall_turn"].read(src) == int(Salt.SWARM_WALL_TURN)

    import field_ab_harness as fab
    sim = fab.swarm_scenario_sim()
    seen = []
    real = swarm.engine_args

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append(out)
        return out
    monkeypatch.setattr(swarm, "engine_args", spy)
    sim.set_paused(False)
    sim.step()
    assert seen, "the swarm step gathered nothing"
    tab = seen[-1]["packs"][0].tables[0]
    assert int(tab[swarm_tables.offset("larva", "salt_tumble")]) == int(Salt.SWARM_TUMBLE)
    assert int(tab[swarm_tables.offset("larva", "salt_wall_turn")]) == int(Salt.SWARM_WALL_TURN)


def test_the_raw_reader():
    """T-P3-31: bp.swarm_larva_felt_T(T, x, y) == T[y >> 16, x >> 16] over
    seeded random fields and in-grid points incl. tile edges; out of the
    grid raises ValueError.

    Must break on a reader other than the own tile (ruling 11) or an
    off-by-one tile lookup."""
    rng = np.random.default_rng(31)
    for _ in range(20):
        h, w = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        T = rng.integers(-(2 ** 31), 2 ** 31 - 1, size=(h, w)).astype(np.int32)
        xs = list(rng.integers(0, w << 16, size=50)) + [0, (w << 16) - 1, (w - 1) << 16]
        ys = list(rng.integers(0, h << 16, size=50)) + [0, (h << 16) - 1, (h - 1) << 16]
        for x, y in zip(xs, ys):
            assert bp.swarm_larva_felt_T(T, int(x), int(y)) == int(T[int(y) >> 16, int(x) >> 16])
        for x, y in ((-1, 0), (0, -1), (w << 16, 0), (0, h << 16)):
            with pytest.raises(ValueError):
                bp.swarm_larva_felt_T(T, x, y)
