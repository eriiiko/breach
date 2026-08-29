"""Regression: PhysicsRecorder.dump() must not crash when units are present.

The bug: record() stored unit position under keys 'x'/'y', but dump() read
'fx'/'fy' — keys that never existed — so EVERY blowup/F8 dump with units raised
`KeyError: 'fx'` (reproduced end-to-end on test_level, where air tiles adjacent
to SPACE vent violently and trip the blowup auto-dump at tick ~70).

This test drives the exact broken path (record a tick with units, then dump)
against a minimal fake GameMap, and asserts the .npz carries the unit_fx/unit_fy
positions the offline analysis tools expect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from simulation.recorder import PhysicsRecorder
from simulation.gases import O2


class _FakeGmap:
    """Minimal stand-in exposing only what recorder.record() reads."""
    def __init__(self, fh, fw, n_gases=16):
        atm = np.full((fh, fw), 65536, dtype=np.int32)   # 1.0 atm, Q16.16
        self.atmosphere = atm
        self.wave_p = atm.copy()                          # P_prev == P: no transient
        self.temperature = np.zeros((fh, fw), dtype=np.int32)
        self.smoke = np.zeros((fh, fw), dtype=np.int32)
        self.fire = np.zeros((fh, fw), dtype=np.int32)
        self.obstacles = np.zeros((fh, fw), dtype=bool)
        # P-E5: wind_x/wind_y joined Recorder.DEFAULT_FIELDS (momentum evidence
        # for the pressure-transient investigation). The real GameMap has
        # carried these Q16.16 planes since the EOS refactor; the fake was
        # simply behind, so this mirrors it rather than relaxing the recorder.
        self.wind_x = np.zeros((fh, fw), dtype=np.int32)
        self.wind_y = np.zeros((fh, fw), dtype=np.int32)
        self.gas = np.zeros((n_gases, fh, fw), dtype=np.int32)
        # gas-energy conservation arc #54, P-G0: `gas_energy` joined
        # Recorder.DEFAULT_FIELDS (design §5); mirrors the real GameMap.
        self.gas_energy = np.zeros((fh, fw), dtype=np.int64)
        assert O2 < n_gases


class _FakeUnit:
    def __init__(self, name, x, y):
        self.name = name
        self.team = 0
        self.x = x
        self.y = y
        self.current_hp = 100
        self.alive = True


def test_dump_with_units_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)              # dump() writes to CWD
    fh, fw = 4, 4
    rec = PhysicsRecorder(fh, fw, capacity=4)
    gmap = _FakeGmap(fh, fw)
    units = [_FakeUnit("alice", 10, 20), _FakeUnit("bob", 30, 40)]

    rec.record(gmap, tick=0, real_time=0.0, units=units)
    fn = rec.dump("manual")                  # <- used to raise KeyError: 'fx'

    assert fn is not None
    data = np.load(tmp_path / fn)
    assert data["unit_fx"].shape == (1, 2)   # (n_ticks, n_units)
    assert list(data["unit_fx"][0]) == [10, 30]   # 'x' stored, not the phantom 'fx'
    assert list(data["unit_fy"][0]) == [20, 40]
    assert list(data["unit_names"]) == ["alice", "bob"]
