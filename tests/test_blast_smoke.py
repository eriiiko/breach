"""Physical blast smoke (#12 smoke x light, handle 3, 2026-10-04): a blast's
smoke is the soot its explosive leaves -- explosive_kg x 1000 x
[explosives.<name>] soot_yield grams, deposited as smoke units through the same
soot mass per unit (gases.soot_g_per_smoke_unit) the heat and light extinctions
are derived from. These tests hold the explosives table to its sources and the
executor to the mass; no test reads a payload row's mass (Erik retunes those).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation.field_edit import EditQueue  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import SMOKE, smoke_units_of_soot  # noqa: E402
from simulation.payloads import blast_smoke_peak, execute_payload  # noqa: E402
from simulation.weapons import PayloadDef, PayloadTable  # noqa: E402

# Ornellas' TNT detonation products (Kuhl et al., UCRL-K-131748): 3.65 mol C(s)
# per mol TNT (C7H5N3O6).
C_S_PER_TNT, M_C, M_TNT = 3.65, 12.011, 227.13


def _room(h=24, w=24):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    return GameMap(LevelData(name="blast_smoke", version="2", path=Path("."),
                             tilemap=tm, tile_size_m=0.333, diffuse_path=Path(".")))


def test_explosives_table_is_its_sources():
    """PROPERTY: TNT's soot yield is Ornellas' measured carbon (3.65 mol C(s)
    per mol TNT, 0.193 g/g), RDX's is zero (no C-C bonds: no soot precursors),
    Composition B is its 60 RDX / 40 TNT mixture, and C4 (91 % RDX) is zero.

    BREAKS IF: a yield is hand-tuned away from its source, or Composition B
    stops being its own mixture.
    """
    ex = CFG.explosives
    tnt = C_S_PER_TNT * M_C / M_TNT
    assert abs(float(ex.tnt.soot_yield) - tnt) <= 0.005 * tnt
    assert float(ex.rdx.soot_yield) == 0.0 and float(ex.c4.soot_yield) == 0.0
    mix = 0.4 * float(ex.tnt.soot_yield) + 0.6 * float(ex.rdx.soot_yield)
    assert abs(float(ex.comp_b.soot_yield) - mix) <= 0.001


def test_a_blast_deposits_exactly_its_charges_soot():
    """PROPERTY: with the per-tile noise off, a blast on open floor puts the
    charge's soot mass into the smoke field -- the deposited smoke units equal
    smoke_units_of_soot(explosive_kg x 1000 x soot_yield) to within the field's
    quantisation -- and a charge with no soot (C4) puts none.

    BREAKS IF: the executor goes back to a fixed disc amount, budgets smoke onto
    walls, or a soot-free charge makes smoke.
    """
    soot_g = 0.184 * 1000.0 * float(CFG.explosives.comp_b.soot_yield)
    frag = PayloadDef("t_frag", radius=4, emit_blast_smoke=True,
                      explosive="comp_b", explosive_kg=0.184, blast_soot_g=soot_g)
    c4 = PayloadDef("t_c4", radius=4, emit_blast_smoke=True,
                    explosive="c4", explosive_kg=0.567, blast_soot_g=0.0)
    old_noise = getattr(CFG.physics, "explosion_smoke_noise", 0.85)
    CFG.physics.explosion_smoke_noise = 0.0
    try:
        for p, want in ((frag, smoke_units_of_soot(soot_g)), (c4, 0.0)):
            g, q = _room(), EditQueue()
            before = g.gas[SMOKE].astype(np.float64).sum()
            execute_payload(g, q, [], 12, 12, p, np.random.default_rng(1))
            q.flush(g, np.random.default_rng(1))
            got = (g.gas[SMOKE].astype(np.float64).sum() - before) / 65536.0
            assert abs(got - want) <= 1e-3 * max(want, 1.0), (p.name, got, want)
    finally:
        CFG.physics.explosion_smoke_noise = old_noise


def test_blast_smoke_needs_an_explosive_and_a_mass():
    """PROPERTY: a payload row that emits blast smoke must name an
    [explosives.*] row and a positive explosive_kg -- refused at load, by name.

    BREAKS IF: a row can make blast smoke without saying what it is made of.
    """
    base = {"radius": 2, "emit_blast_smoke": True}
    with pytest.raises(ValueError, match="t_bad"):
        PayloadTable({"t_bad": dict(base, explosive="unobtainium", explosive_kg=1.0)})
    with pytest.raises(ValueError, match="t_bad"):
        PayloadTable({"t_bad": dict(base, explosive="tnt", explosive_kg=0.0)})
    assert PayloadTable(CFG.payloads).by_name          # the shipped table loads


def test_blast_smoke_peak_is_zero_without_soot_or_open_floor():
    """PROPERTY: no soot, or a disc with no open tile, gives a zero peak.

    BREAKS IF: blast_smoke_peak divides by an empty disc or invents smoke.
    """
    g = _room()
    assert blast_smoke_peak(g, 12, 12, 4, 0.0) == 0.0
    assert blast_smoke_peak(g, 12, 12, 0, 10.0) == 0.0
    g.solid[...] = True
    assert blast_smoke_peak(g, 12, 12, 4, 10.0) == 0.0
