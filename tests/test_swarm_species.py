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
    """arc #63 P3 (F7): the species' FULL repo row (testkit), P2's explicit
    max_units / hp_max kept as overrides -- rows grow with SPECIES_FIELDS."""
    import _swarm_testkit as testkit
    row = {"max_units": 64, "hp_max": 1.0}
    row.update(overrides)
    return testkit.species_row(swarm.SWARM_SPECIES[0], **row)


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
    import re
    import _swarm_testkit as testkit
    sp0 = swarm.SWARM_SPECIES[0]

    # arc #63 P3 (R10, L2-M4): every case runs on a FULL config wrapper
    # (swarm + physics.temperature_scale, testkit.cfg_with) and asserts the
    # rejection names ITS OWN reason, so a lax branch can never hide behind
    # an unrelated error (the Kelvin conversion's missing-scale error).
    def expect_reject(cfg_swarm, token):
        with pytest.raises(ValueError, match=re.escape(token)):
            swarm.SwarmSpeciesTable.from_config(testkit.cfg_with(cfg_swarm))

    # the positive case: the full valid config passes.
    swarm.SwarmSpeciesTable.from_config(testkit.cfg_with(_full_valid_cfg()))

    # a missing [swarm] entirely.
    with pytest.raises(ValueError, match=re.escape("[swarm] missing")):
        swarm.SwarmSpeciesTable.from_config(testkit.cfg_with(None))

    # a missing species row.
    if len(swarm.SWARM_SPECIES) > 0:
        rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
        del rows[sp0]
        expect_reject(_swarm_cfg(rows), f"[swarm.{sp0}] missing")

    # an unknown species under [swarm].
    rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
    rows["totally_unknown_species"] = _valid_row()
    expect_reject(_swarm_cfg(rows), "[swarm].totally_unknown_species")

    # a non-table key directly under [swarm].
    cfg_bad = Namespace({sp: _valid_row() for sp in swarm.SWARM_SPECIES})
    setattr(cfg_bad, sp0, 5)   # overwrite the table with a scalar
    expect_reject(cfg_bad, f"[swarm].{sp0} must be a table")

    # an unknown key in a row.
    rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
    rows[sp0]["unknown_key_xyz"] = 1
    expect_reject(_swarm_cfg(rows), "unknown_key_xyz")

    # a missing key in a row.
    rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
    del rows[sp0]["hp_max"]
    expect_reject(_swarm_cfg(rows), f"[swarm.{sp0}].hp_max missing")

    # per-field wrong type / out-of-bounds / NaN-inf / bool.
    for f in swarm.SPECIES_FIELDS:
        if f.kind == KIND_INT:
            for bad in (True, 1.5, "x"):
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = bad
                expect_reject(_swarm_cfg(rows), f"[swarm.{sp0}].{f.name}")
            if f.minimum is not None:
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = int(f.minimum) - 1
                expect_reject(_swarm_cfg(rows), f"[swarm.{sp0}].{f.name}")
            if f.maximum is not None:
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = int(f.maximum) + 1
                expect_reject(_swarm_cfg(rows), f"[swarm.{sp0}].{f.name}")
        elif f.kind == KIND_REAL_Q16:
            for bad in (True, "x", math.nan, math.inf, -math.inf):
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = bad
                expect_reject(_swarm_cfg(rows), f"[swarm.{sp0}].{f.name}")
            if f.minimum is not None:
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = float(f.minimum) - 1.0
                expect_reject(_swarm_cfg(rows), f"[swarm.{sp0}].{f.name}")
            if f.maximum is not None:
                rows = {sp: _valid_row() for sp in swarm.SWARM_SPECIES}
                rows[sp0][f.name] = float(f.maximum) + 1.0
                expect_reject(_swarm_cfg(rows), f"[swarm.{sp0}].{f.name}")

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
    hp_a = int(sim.swarm.column(sp, "hp")[slots_a[0]])

    old_hash = sim.gmap.swarm_species.hash
    monkeypatch.setattr(CFG, "swarm", _full_valid_cfg(hp_max=2.0))

    # spawn B WITHOUT calling the seam -- must still see the OLD hp.
    slots_b = sim.swarm.spawn(sp, [0], [0], [0])
    hp_b = int(sim.swarm.column(sp, "hp")[slots_b[0]])
    assert hp_b == hp_a, "a reader must never reload lazily"
    assert sim.gmap.swarm_species.hash == old_hash

    pos_before = sim.swarm.column(sp, "pos_x").copy()
    hp_col_before = sim.swarm.column(sp, "hp").copy()

    sim.on_config_reload()   # Simulation.on_config_reload() returns None
    assert sim.gmap.swarm_species.hash != old_hash, (
        "the hash must change once the reload installs the new table")
    # A reload rewrites NO array -- every existing slot's hp is untouched.
    assert np.array_equal(sim.swarm.column(sp, "pos_x"), pos_before)
    assert np.array_equal(sim.swarm.column(sp, "hp"), hp_col_before)

    slots_c = sim.swarm.spawn(sp, [0], [0], [0])
    hp_c = int(sim.swarm.column(sp, "hp")[slots_c[0]])
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
    before_hp = int(sim.swarm.column(sp, "hp")[slots[0]])

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
    assert int(sim.swarm.column(sp, "hp")[slots[0]]) == before_hp

    # the underlying seam target's bool contract (§4.3): False on rejection.
    assert swarm.reload_species(sim.gmap, CFG) is False
    assert sim.gmap.swarm_species is before_table


# ---------------------------------------------------------------------------
# arc #63 P3 (docs/swarm_P3_impl.md §11 T-P3-22..25): the larva rows, the
# coupling rows, the Kelvin conversion and the species rules.
# ---------------------------------------------------------------------------

def test_swarm_coupling_rows_cannot_drift_from_the_kernel():
    """T-P3-22: every COUPLING_TABLE row whose consumer is not "unit" names
    "swarm:<sp>" for a SWARM_SPECIES member, has no Python response, a
    reduction in the vocabulary, reads only species-table fields whose
    stored ints are rows of that species' kernel table, and senses through
    the registry's (field, field_reduction); every species has a row. Every
    "unit" row keeps a callable response.

    Must break on a renamed field or species, a row reading a number the
    kernel does not take, or a reduction label that is not the kernel's
    reader."""
    from simulation import exchange, swarm_tables
    by_name = {d["name"]: d for d in bp.SWARM_SPECIES_LAWS}
    field_names = {f.name for f in swarm.SPECIES_FIELDS}
    covered = set()
    for row in exchange.COUPLING_TABLE:
        if row.consumer == "unit":
            assert callable(row.response)
            continue
        assert row.consumer.startswith("swarm:")
        sp = row.consumer.split(":", 1)[1]
        assert sp in swarm.SWARM_SPECIES
        assert row.response is None
        assert row.reduction in exchange.REDUCTIONS
        assert set(row.reads) <= field_names
        kernel_rows = {r.name for r in swarm_tables.ROWS[sp]}
        assert {f"{r}_q" for r in row.reads} <= kernel_rows, row.reads
        assert (row.field, row.reduction) == (by_name[sp]["field"],
                                              by_name[sp]["field_reduction"])
        covered.add(sp)
    assert covered == set(swarm.SWARM_SPECIES)


def test_kelvin_rows_convert_through_the_scale_in_force():
    """T-P3-23: for every SPECIES_UNITS kelvin row, to_kelvin(dequantize(the
    stored int)) equals the authored Kelvin within k / 2^17 K.

    Must break on a missing from_kelvin, a double conversion, or the unit
    applied to the wrong field."""
    import temperature_scale
    from simulation import swarm_fixed
    ts = temperature_scale.load()
    table = swarm.SwarmSpeciesTable.from_config(CFG)
    for sp in swarm.SWARM_SPECIES:
        row = table.row(sp)
        authored = vars(getattr(CFG.swarm, sp))
        for name, unit in swarm.SPECIES_UNITS.items():
            assert unit == "kelvin"
            stored = getattr(row, f"{name}_q")
            back = ts.to_kelvin(swarm_fixed.dequantize(stored))
            assert abs(back - float(authored[name])) <= ts.k_temp_to_kelvin / 2 ** 17, name


def test_scale_immunity_and_the_reload_seam(monkeypatch, capsys):
    """T-P3-24: under a monkeypatched k_temp_to_kelvin (phi_exp kept so
    phi_exp * k == 1) a fresh GameMap stores different game ints for the
    kelvin rows but the same Kelvin; on a running sim the seam changes
    .hash, prints one "replay comparability ends here" line, and the next
    tick's launch uses the new ints (a larva on a tile lethal under one scale
    and not the other dies only after the reload); a phi_exp breaking the
    product makes the reload print one RELOAD REJECTED line and keeps the
    table.

    Must break on a cached conversion, a conversion outside the seam, a hash
    over authored text, or an AssertionError escaping the seam."""
    import temperature_scale
    from simulation import swarm_fixed
    from simulation.events import SwarmUnitDiedEvent
    tsc = CFG.physics.temperature_scale
    k0 = float(tsc.k_temp_to_kelvin)
    lv = _mini_level()
    g_a = GameMap(lv)
    sim = Simulation(lv, seed=4, breach_physics=bp, enable_recorder=False)
    ts_a = temperature_scale.load()
    # a patch at 20 game counts: 313 K under k = 1 (safe), 353 K under k = 3 (lethal)
    sim.gmap.seed_gas_temperature((slice(2, 6), slice(2, 6)), 20 * swarm_fixed.FP_ONE)
    sim.swarm.spawn("larva", [(4 << 16) + 9], [(4 << 16) + 9], [0])
    sim.set_paused(False)
    sim.step()
    assert sim.swarm.live_count("larva") == 1
    k1 = 3.0 * k0
    monkeypatch.setattr(tsc, "k_temp_to_kelvin", k1)
    monkeypatch.setattr(tsc, "phi_exp", float(tsc.phi_exp) / 3.0)
    g_b = GameMap(lv)
    ts_b = temperature_scale.load()
    for name in swarm.SPECIES_UNITS:
        qa = getattr(g_a.swarm_species.row("larva"), f"{name}_q")
        qb = getattr(g_b.swarm_species.row("larva"), f"{name}_q")
        assert qa != qb
        assert abs(ts_a.to_kelvin(swarm_fixed.dequantize(qa))
                   - ts_b.to_kelvin(swarm_fixed.dequantize(qb))) <= k1 / 2 ** 16
    old = sim.gmap.swarm_species.hash
    capsys.readouterr()
    sim.on_config_reload()
    out = capsys.readouterr().out
    assert sim.gmap.swarm_species.hash != old
    assert sum("replay comparability ends here" in ln for ln in out.splitlines()) == 1
    sim.set_paused(False)
    sim.step()
    assert any(isinstance(e, SwarmUnitDiedEvent) for e in sim.tick_events)
    table = sim.gmap.swarm_species
    monkeypatch.setattr(tsc, "phi_exp", 1.0)       # phi_exp * k = 3: refused
    capsys.readouterr()
    assert swarm.reload_species(sim.gmap, CFG) is False
    out = capsys.readouterr().out
    assert sum("RELOAD REJECTED" in ln for ln in out.splitlines()) == 1
    assert sim.gmap.swarm_species is table


# One config per species rule that violates ONLY that rule (a new rule needs
# its case here, or the coverage assertion below names it).
_RULE_BREAKERS = {
    "no_tunnelling": {"speed_mps": 100.0},
    "tumble_probability": {"tumble_rate_base": 2.0, "tumble_rate_alarm": 1.0},
    "coma_hysteresis": {"T_ctmin": 283.15, "T_coma_exit": 283.15},
    "turn_range": {"turn_min": 1.0, "turn_max": 0.5},
}


def test_species_rules_hold_at_load_reload_and_launch(monkeypatch, capsys):
    """T-P3-25: for every species and every SPECIES_RULES entry, a config
    violating only that rule raises ValueError naming the rule at
    GameMap(...), and through sim.on_config_reload() prints exactly one
    RELOAD REJECTED line and keeps the table object; R1 is also broken by a
    small tile_size_m alone. At launch: with a swarm present, a tick length
    that breaks R1 makes the next sim.step() raise RuntimeError naming
    no_tunnelling, while the same change on a swarm-free sim steps cleanly.

    Must break if a rule is not wired to an installer or to the launch, or
    the launch checks a swarm-free map."""
    import _swarm_testkit as testkit
    for sp in swarm.SWARM_SPECIES:
        names = [r.name for r in swarm.SPECIES_RULES[sp]]
        missing = set(names) - set(_RULE_BREAKERS)
        assert not missing, f"rule(s) without a breaking case: {missing}"
        for name in names:
            monkeypatch.setattr(CFG, "swarm", testkit.swarm_namespace(**_RULE_BREAKERS[name]))
            with pytest.raises(ValueError, match=name):
                GameMap(_mini_level())
            monkeypatch.setattr(CFG, "swarm", testkit.swarm_namespace())
            sim = Simulation(_mini_level(), seed=1, breach_physics=None, enable_recorder=False)
            table = sim.gmap.swarm_species
            monkeypatch.setattr(CFG, "swarm", testkit.swarm_namespace(**_RULE_BREAKERS[name]))
            capsys.readouterr()
            sim.on_config_reload()
            lines = [ln for ln in capsys.readouterr().out.splitlines() if "RELOAD REJECTED" in ln]
            assert len(lines) == 1 and name in lines[0]
            assert sim.gmap.swarm_species is table
    monkeypatch.setattr(CFG, "swarm", testkit.swarm_namespace())
    tiny = _mini_level()
    tiny.tile_size_m = 0.002
    with pytest.raises(ValueError, match="no_tunnelling"):
        GameMap(tiny)

    # launch: 23.8 m/s on 1 m tiles is just inside R1 at 24 tps, outside at 20.
    monkeypatch.setattr(CFG, "swarm", testkit.swarm_namespace(speed_mps=23.8))
    sim = Simulation(_mini_level(), seed=1, breach_physics=bp, enable_recorder=False)
    sim.swarm.spawn("larva", [(3 << 16)], [(3 << 16)], [0])
    sim.set_paused(False)
    sim.step()
    sim._tps = 20
    with pytest.raises(RuntimeError, match="no_tunnelling"):
        sim.set_paused(False)
        sim.step()
    free = Simulation(_mini_level(), seed=1, breach_physics=bp, enable_recorder=False)
    free._tps = 20
    free.set_paused(False)
    free.step()
