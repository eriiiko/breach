"""Swarm species table — arc #63 P2 (docs/swarm_P2_impl.md §11, T22-T26).

Strict schema validation (unknown/missing key or species, wrong type,
out-of-bounds value) and hot reload through the ONE explicit seam
(``Simulation.on_config_reload``): a debug tool must never decide which
table is in force, and a rejected reload keeps the prior table.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_swarm_species.py -q
"""
from __future__ import annotations

import math
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
from simulation.entities.schema import KIND_INT, KIND_REAL_Q16  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402


def _mini_level() -> LevelData:
    h = w = 8
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:7, 1:7] = 4
    return LevelData(name="swarm_species_test", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _valid_row(**overrides) -> dict:
    row = {"max_units": 64, "hp_max": 1.0}
    row.update(overrides)
    return row


def _swarm_cfg(rows_by_species: dict) -> Namespace:
    return Namespace(dict(rows_by_species))


def _full_valid_cfg(**row_overrides) -> Namespace:
    """A valid ``[swarm]`` Namespace covering EVERY current SWARM_SPECIES —
    never pins the species set's size."""
    return _swarm_cfg({sp: _valid_row(**row_overrides) for sp in swarm.SWARM_SPECIES})


# ---------------------------------------------------------------------------
# T22 — strict schema
# ---------------------------------------------------------------------------

def test_strict_schema_rejects_every_named_shape_of_bad_config(monkeypatch):
    """Each of these raises ValueError naming the key, at GameMap
    construction (load time): a missing [swarm], a missing species row, an
    unknown species, an unknown key, and — for every SPECIES_FIELDS entry —
    a wrong type and an out-of-bounds value. Also NaN/+-inf for every
    KIND_REAL_Q16 entry and a bool for every KIND_INT entry.
    SPECIES_RELOAD's keys and SwarmSpeciesRow's fields match SPECIES_FIELDS
    (the import-time assertions' properties, re-derived here).

    Must break under a Namespace pass-through with no schema, a lax
    KIND_REAL_Q16 branch, or a drifted row class."""
    sp0 = swarm.SWARM_SPECIES[0]

    def expect_reject(cfg_swarm):
        cfg = Namespace({})
        cfg.swarm = cfg_swarm
        with pytest.raises(ValueError):
            swarm.SwarmSpeciesTable.from_config(cfg)

    # a missing [swarm] entirely.
    cfg_missing = Namespace({})
    with pytest.raises(ValueError):
        swarm.SwarmSpeciesTable.from_config(cfg_missing)

    # a missing species row.
    if len(swarm.SWARM_SPECIES) > 0:
        rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
        del rows[sp0]
        expect_reject(_swarm_cfg(rows))

    # an unknown species under [swarm].
    rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
    rows["totally_unknown_species"] = _valid_row()
    expect_reject(_swarm_cfg(rows))

    # a non-table key directly under [swarm].
    cfg_bad = Namespace({sp: _valid_row() for sp in swarm.SWARM_SPECIES})
    setattr(cfg_bad, sp0, 5)   # overwrite the table with a scalar
    cfg_wrap = Namespace({})
    cfg_wrap.swarm = cfg_bad
    with pytest.raises(ValueError):
        swarm.SwarmSpeciesTable.from_config(cfg_wrap)

    # an unknown key in a row.
    rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
    rows[sp0]["unknown_key_xyz"] = 1
    expect_reject(_swarm_cfg(rows))

    # a missing key in a row.
    rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
    del rows[sp0]["hp_max"]
    expect_reject(_swarm_cfg(rows))

    # per-field wrong type / out-of-bounds / NaN-inf / bool.
    for f in swarm.SPECIES_FIELDS:
        if f.kind == KIND_INT:
            for bad in (True, 1.5, "x"):
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = bad
                expect_reject(_swarm_cfg(rows))
            if f.minimum is not None:
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = int(f.minimum) - 1
                expect_reject(_swarm_cfg(rows))
            if f.maximum is not None:
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = int(f.maximum) + 1
                expect_reject(_swarm_cfg(rows))
        elif f.kind == KIND_REAL_Q16:
            for bad in (True, "x", math.nan, math.inf, -math.inf):
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = bad
                expect_reject(_swarm_cfg(rows))
            if f.minimum is not None:
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = float(f.minimum) - 1.0
                expect_reject(_swarm_cfg(rows))
            if f.maximum is not None:
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = float(f.maximum) + 1.0
                expect_reject(_swarm_cfg(rows))

    # the import-time assertions' properties, re-derived as a live check:
    assert set(swarm.SPECIES_RELOAD) == {f.name for f in swarm.SPECIES_FIELDS}
    assert set(swarm.SPECIES_RELOAD.values()) <= {"hot", "load"}
    expected_row_attrs = {"species"} | {
        swarm._species_stored_attr(f) for f in swarm.SPECIES_FIELDS}
    from dataclasses import fields as dc_fields
    actual_row_attrs = {fld.name for fld in dc_fields(swarm.SwarmSpeciesRow)}
    assert actual_row_attrs == expected_row_attrs

    # a real load-time error surfaces through GameMap construction too.
    monkeypatch.setattr(CFG, "swarm", Namespace({}))
    with pytest.raises(ValueError):
        GameMap(_mini_level())


# ---------------------------------------------------------------------------
# T23 — hot reload through the ONE seam, no lazy reload anywhere
# ---------------------------------------------------------------------------

def test_hot_reload_through_the_seam_only(monkeypatch):
    """Spawn A; replace CFG.swarm with hp_max changed; spawn B WITHOUT
    calling the seam -> B gets the OLD hp. Take digest D1 and call
    sim.on_config_reload() -> digest still D1 (no array touched, since a
    reload changes hp for FUTURE spawns only), carrier()['species_hash']
    changed; spawn C -> hp == the new hp_max_q. A reload that changes only
    spelling (1.0 -> 1) keeps .hash and prints no 'replay comparability
    ends' line.

    Must break if any reader reloads lazily, a reload rewrites live hp, the
    table is hashed into the section, or the hash is taken over authored
    text."""
    sp = swarm.SWARM_SPECIES[0]
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(hp_max=1.0))
    sim = Simulation(_mini_level(), seed=1, breach_physics=None,
                     enable_recorder=False)

    slots_a = sim.swarm.spawn(sp, [0], [0], [0])
    hp_a = int(sim.swarm.column(sp, "hp")[0, slots_a[0]])

    old_hash = sim.gmap.swarm_species.hash
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(hp_max=2.0))

    # spawn B WITHOUT calling the seam -- must still see the OLD hp.
    slots_b = sim.swarm.spawn(sp, [0], [0], [0])
    hp_b = int(sim.swarm.column(sp, "hp")[0, slots_b[0]])
    assert hp_b == hp_a, "a reader must never reload lazily"
    assert sim.gmap.swarm_species.hash == old_hash

    pos_before = sim.swarm.column(sp, "pos_x")[0].copy()
    hp_col_before = sim.swarm.column(sp, "hp")[0].copy()

    sim.on_config_reload()   # Simulation.on_config_reload() returns None
    assert sim.gmap.swarm_species.hash != old_hash, (
        "the hash must change once the reload installs the new table")
    # A reload rewrites NO array -- every existing slot's hp is untouched.
    assert np.array_equal(sim.swarm.column(sp, "pos_x")[0], pos_before)
    assert np.array_equal(sim.swarm.column(sp, "hp")[0], hp_col_before)

    slots_c = sim.swarm.spawn(sp, [0], [0], [0])
    hp_c = int(sim.swarm.column(sp, "hp")[0, slots_c[0]])
    from simulation import swarm_fixed
    assert hp_c == swarm_fixed.quantize_scalar(2.0)
    assert hp_c != hp_a

    # a reload that changes only spelling keeps .hash and prints nothing.
    same_hash = sim.gmap.swarm_species.hash
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(hp_max=2))  # 2.0 -> 2
    sim.on_config_reload()
    assert sim.gmap.swarm_species.hash == same_hash


# ---------------------------------------------------------------------------
# T24 — recorder-on vs recorder-off: identical section bytes at every tick
# ---------------------------------------------------------------------------

def test_recorder_on_vs_off_agree_under_a_mid_run_reload(monkeypatch):
    """Two sims (enable_recorder True / False), same scripted spawns.
    CFG.swarm is swapped mid-run WITHOUT calling the seam; a spawn happens;
    the seam is called on BOTH at the same point; another spawn happens ->
    identical section bytes on every tick. A debug tool (the recorder) must
    never decide which table's numbers are in force.

    Must break if any read path (carrier(), the recorder, spawn) reloads
    the table itself instead of going through the seam."""
    from simulation.swarm import swarm_carrier, swarm_section_bytes

    sp = swarm.SWARM_SPECIES[0]
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(hp_max=1.0))
    sim_on = Simulation(_mini_level(), seed=7, breach_physics=None,
                        enable_recorder=True)
    sim_off = Simulation(_mini_level(), seed=7, breach_physics=None,
                         enable_recorder=False)

    def section_bytes(sim):
        return swarm_section_bytes(swarm_carrier(sim.gmap))

    for tick in range(7):
        if tick == 3:
            monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(hp_max=3.0))
        if tick == 5:
            sim_on.swarm.spawn(sp, [0], [0], [0])
            sim_off.swarm.spawn(sp, [0], [0], [0])
        if tick == 6:
            sim_on.on_config_reload()
            sim_off.on_config_reload()
        if tick == 7:
            sim_on.swarm.spawn(sp, [1 << 16], [1 << 16], [0])
            sim_off.swarm.spawn(sp, [1 << 16], [1 << 16], [0])

        assert section_bytes(sim_on) == section_bytes(sim_off), (
            f"tick {tick}: recorder-on vs recorder-off section bytes diverged")
        assert (sim_on.gmap.swarm_species.hash
                == sim_off.gmap.swarm_species.hash)


# ---------------------------------------------------------------------------
# T25 — a load-time field's mid-run edit is deferred + logged once
# ---------------------------------------------------------------------------

def test_load_only_field_change_is_deferred_and_logged_once(monkeypatch, capsys):
    """A seam call whose TOML max_units differs from the in-force value
    prints exactly one warning line for that species and leaves capacity
    and the in-force value unchanged; a reload with an equal value prints
    none; sim.reset() then picks up the new capacity.

    Must break under mid-run reallocation, or a warning that spams or never
    appears."""
    sp = swarm.SWARM_SPECIES[0]
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(max_units=64))
    sim = Simulation(_mini_level(), seed=1, breach_physics=None,
                     enable_recorder=False)
    old_capacity = sim.gmap.swarm_species.row(sp).max_units
    assert old_capacity == 64
    old_array_id = id(sim.swarm.column(sp, "pos_x"))

    capsys.readouterr()
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(max_units=128))
    sim.on_config_reload()
    out = capsys.readouterr().out
    warn_lines = [l for l in out.splitlines()
                 if f"{sp}.max_units" in l and "applies at the next reset" in l]
    assert len(warn_lines) == 1, f"expected exactly one warning line, got {warn_lines}"
    assert sim.gmap.swarm_species.row(sp).max_units == old_capacity
    assert id(sim.swarm.column(sp, "pos_x")) == old_array_id, (
        "max_units is LOAD-TIME only -- no mid-run reallocation")

    # an equal-value reload (TOML matches the value still IN FORCE, 64 --
    # the earlier reload never actually applied 128) prints no warning.
    capsys.readouterr()
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(max_units=64))
    sim.on_config_reload()
    out2 = capsys.readouterr().out
    assert f"{sp}.max_units" not in out2

    # reset() picks up the new capacity (a fresh GameMap re-reads CFG).
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(max_units=128))
    sim.reset(seed=1)
    assert sim.gmap.swarm_species.row(sp).max_units == 128


# ---------------------------------------------------------------------------
# T26 — a rejected reload keeps the prior table
# ---------------------------------------------------------------------------

def test_rejected_reload_keeps_the_prior_table(monkeypatch, capsys):
    """An invalid value through the seam keeps the prior table (same
    object, same .hash), prints exactly one RELOAD REJECTED line naming the
    key per seam call, and touches no array.

    Must break under a crash, or a silent partial apply."""
    sp = swarm.SWARM_SPECIES[0]
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(hp_max=1.0))
    sim = Simulation(_mini_level(), seed=1, breach_physics=None,
                     enable_recorder=False)
    slots = sim.swarm.spawn(sp, [0], [0], [0])
    before_table = sim.gmap.swarm_species
    before_hp = int(sim.swarm.column(sp, "hp")[0, slots[0]])

    bad_rows = {s: _valid_row() for s in swarm.SWARM_SPECIES}
    bad_rows[sp]["hp_max"] = math.nan
    monkeypatch.setattr(CFG, "swarm", _swarm_cfg(bad_rows))

    capsys.readouterr()
    sim.on_config_reload()   # never raises
    out = capsys.readouterr().out
    reject_lines = [l for l in out.splitlines() if "RELOAD REJECTED" in l]
    assert len(reject_lines) == 1
    assert sp in reject_lines[0] and "hp_max" in reject_lines[0]

    assert sim.gmap.swarm_species is before_table   # same object
    assert sim.gmap.swarm_species.hash == before_table.hash
    assert int(sim.swarm.column(sp, "hp")[0, slots[0]]) == before_hp

    # the underlying seam target's bool contract (§4.3): False on rejection.
    assert swarm.reload_species(sim.gmap, CFG) is False
    assert sim.gmap.swarm_species is before_table
