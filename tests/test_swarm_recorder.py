"""Swarm recorder ring family — arc #63 P2 (docs/swarm_P2_impl.md §11, T27).

An additive, presence-gated, env-0 per-unit ring family: a swarm-free
session's dump has no ``swarm_*`` key at all; a swarm-present session
records every ROSTER column verbatim (no cast) and the digest/recorder
agree byte-for-byte (re-encoding a recorded row through the ONE encoder
equals the section bytes the digest hashed); truncation is visible and
logged once; the species-hash ring changes exactly at a reload tick.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_swarm_recorder.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402

from config import CFG, Namespace  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import swarm  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.recorder import PhysicsRecorder  # noqa: E402


def _mini_level() -> LevelData:
    h = w = 8
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:7, 1:7] = 4
    return LevelData(name="swarm_recorder_test", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _swarm_cfg_namespace(**overrides) -> Namespace:
    base = {"max_units": overrides.pop("max_units", 8192),
            "hp_max": overrides.pop("hp_max", 1.0)}
    return Namespace({sp: dict(base) for sp in swarm.SWARM_SPECIES})


def test_swarm_free_dump_has_no_swarm_key(tmp_path, monkeypatch):
    """A swarm-free session's dump has no swarm_* key at all — the key set
    is identical to pre-P2. Must break if the recorder allocates the ring
    family unconditionally."""
    monkeypatch.chdir(tmp_path)
    rec = PhysicsRecorder(8, 8, capacity=4)
    g = GameMap(_mini_level())   # never spawned -> present == False
    for t in range(3):
        rec.record(g, tick=t, real_time=float(t), units=[],
                  swarm=swarm.swarm_carrier(g))
    fn = rec.dump("manual")
    data = np.load(tmp_path / fn)
    assert not any(k.startswith("swarm") for k in data.files)
    assert rec._swarm_rings is None


def test_swarm_present_dump_matches_the_digest_bytes_verbatim(tmp_path, monkeypatch):
    """Every ROSTER column is recorded at exactly its roster dtype, with
    width min(max recorded hw, cap). Rebuilding a carrier from a recorded
    row and re-encoding it through swarm_section_bytes equals the section
    bytes the digest hashed at that tick (the one-encoder rule, R4/T27).

    Must break under a float32 ring, a cast, or recorder/digest drift."""
    monkeypatch.chdir(tmp_path)
    sp = swarm.SWARM_SPECIES[0]
    rec = PhysicsRecorder(8, 8, capacity=10, swarm_units_cap=1024)
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)

    carriers = []
    for t in range(5):
        if t == 2:
            su.spawn(sp, [0, 1 << 16, 2 << 16], [0, 0, 0], [0, 100, -100])
        if t == 4:
            su.reclaim(sp, np.array([1]))
        carrier = swarm.swarm_carrier(g)
        carriers.append(carrier)
        rec.record(g, tick=t, real_time=float(t), units=[], swarm=carrier)

    fn = rec.dump("manual")
    data = np.load(tmp_path / fn)

    for col in swarm.ROSTER:
        key = swarm.store_attr(sp, col.name)
        assert data[key].dtype == col.dtype, (
            f"column '{col.name}' cast away from its roster dtype")

    max_hw = max((c["blocks"].get(sp, {}).get("high_water", 0)
                 for c in carriers), default=0)
    wmax = min(rec.swarm_units_cap, max_hw)
    for col in swarm.ROSTER:
        key = swarm.store_attr(sp, col.name)
        assert data[key].shape[1] == wmax

    for t, carrier in enumerate(carriers):
        if not carrier["present"]:
            continue
        hw = int(data[swarm.high_water_attr(sp)][t])
        rebuilt_columns = {
            col.name: data[swarm.store_attr(sp, col.name)][t][:hw]
            for col in swarm.ROSTER
        }
        rebuilt_carrier = {
            "present": True,
            "next_unit_id": int(data[swarm.NEXT_UNIT_ID_ATTR][t]),
            "blocks": {sp: {"high_water": hw, "columns": rebuilt_columns}},
            "species_hash": carrier["species_hash"],
        }
        assert (swarm.swarm_section_bytes(rebuilt_carrier)
               == swarm.swarm_section_bytes(carrier)), (
            f"tick {t}: recorded row does not re-encode to the digest bytes")


def test_truncation_caps_width_but_reports_the_true_high_water(
        tmp_path, monkeypatch, capsys):
    """With cap < hw: the recorded width is cap, swarm_<sp>_high_water
    still shows the TRUE hw, and exactly one truncation line is printed.

    Must break under silent truncation or a spamming warning."""
    monkeypatch.chdir(tmp_path)
    sp = swarm.SWARM_SPECIES[0]
    cap = 3
    monkeypatch.setattr(CFG, "swarm", _swarm_cfg_namespace(max_units=10))
    rec = PhysicsRecorder(8, 8, capacity=5, swarm_units_cap=cap)
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    xs = [i << 16 for i in range(6)]
    su.spawn(sp, xs, [0] * 6, [0] * 6)   # hw == 6 > cap == 3

    capsys.readouterr()
    for t in range(2):
        rec.record(g, tick=t, real_time=float(t), units=[],
                  swarm=swarm.swarm_carrier(g))
    out = capsys.readouterr().out
    trunc_lines = [l for l in out.splitlines() if "recording truncated" in l]
    assert len(trunc_lines) == 1, f"expected exactly one line, got {trunc_lines}"

    fn = rec.dump("manual")
    data = np.load(tmp_path / fn)
    assert data[swarm.store_attr(sp, "pos_x")].shape[1] == cap
    assert list(data[swarm.high_water_attr(sp)]) == [6, 6]


def test_species_hash_ring_changes_exactly_at_the_reload_tick(monkeypatch):
    """swarm_species_hash is a PER-TICK ring, and it changes exactly at the
    tick a seam reload changed .hash (never a single 0-d value). Must break
    under a single 0-d hash or a ring that lags/leads the reload tick."""
    sp = swarm.SWARM_SPECIES[0]
    monkeypatch.setattr(CFG, "swarm", _swarm_cfg_namespace(hp_max=1.0))
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    su.spawn(sp, [0], [0], [0])   # present, so the rings will allocate

    rec = PhysicsRecorder(8, 8, capacity=6)
    hashes = []
    for t in range(4):
        if t == 2:
            monkeypatch.setattr(CFG, "swarm", _swarm_cfg_namespace(hp_max=5.0))
            swarm.reload_species(g, CFG)
        carrier = swarm.swarm_carrier(g)
        hashes.append(carrier["species_hash"])
        rec.record(g, tick=t, real_time=float(t), units=[], swarm=carrier)

    ring = list(rec._swarm_shash_ring[:4])
    expected = [PhysicsRecorder._species_hash_int64(h) for h in hashes]
    assert ring == expected
    changed_at = [i for i in range(1, len(expected)) if expected[i] != expected[i - 1]]
    assert changed_at == [2], (
        "the species-hash ring must change exactly at the reload tick")
