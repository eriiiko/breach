"""src/temperature_scale.py — the canonical game-T -> Kelvin map accessor (P-K1).

Pure-arithmetic + config-plumbing unit tests
(docs/temperature_scale_unification_design_2026-08-13.md §2/§3d/§3e):
  - the CFG-based entry point (:func:`temperature_scale.load`) reads
    ``[physics.temperature_scale]``,
  - the standalone entry point (:func:`temperature_scale.from_toml`) parses
    the SAME real config.toml and agrees with the CFG path,
  - ``tools/fire_tune_plot.kelvin_map()`` (the tool-facing wrapper) returns
    the same (293.0, 1.0) via the real config (G12: was (293.0, 3.0)),
  - the phi_exp value-freeze assert lands on exactly 65536 counts,
  - the migration guard hard-errors on a stale ``[render.blackbody]`` key,
    or a stale ``[physics.eos]`` t_amb_k/C key (P-K3).

Run:
    conda run -n data python -m pytest tests/test_temperature_scale.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import gas_fixed  # noqa: E402
import temperature_scale  # noqa: E402


# ---- CFG-based entry point ----------------------------------------------

def test_load_reads_temperature_scale_section_from_cfg():
    # G12 (issue #12, docs/fire_g12_one_map_patch_2026-08-31.md): the
    # collapse to slope 1, ambient 293 everywhere.
    ts = temperature_scale.load()
    assert ts.kelvin_ambient == 293.0
    assert ts.k_temp_to_kelvin == 1.0
    assert ts.phi_exp == 1.0
    assert ts.eos_t_amb_k == 293.0
    assert ts.to_kelvin(300.0) == pytest.approx(293.0 + 1.0 * 300.0)
    assert ts.from_kelvin(ts.to_kelvin(300.0)) == pytest.approx(300.0)


def test_load_defaults_match_config_toml():
    """The single defaults dict IS the §2 canonical table — so loading the
    real (unmigrated-away) config.toml reproduces the defaults exactly."""
    ts = temperature_scale.load()
    for key, value in temperature_scale.DEFAULTS.items():
        assert getattr(ts, key) == value


# ---- from_toml parity with the CFG path ----------------------------------

def test_from_toml_matches_cfg_on_the_real_config():
    from_cfg = temperature_scale.load()
    from_file = temperature_scale.from_toml()
    assert from_file == from_cfg


# ---- tools/fire_tune_plot.kelvin_map() ------------------------------------

def test_fire_tune_plot_kelvin_map_matches_real_config():
    import fire_tune_plot
    assert fire_tune_plot.kelvin_map() == (293.0, 1.0)  # G12: 3.0 -> 1.0


# ---- phi_exp value-freeze: quantized EOS slope == 65536 counts -----------

def test_eos_slope_quantizes_to_65536_while_phi_exp_is_value_frozen():
    ts = temperature_scale.load()
    assert ts.eos_slope == 1.0
    assert gas_fixed.quantize_scalar(ts.eos_slope) == 65536


def test_C_is_reciprocal_of_eos_t_amb_k():
    ts = temperature_scale.load()
    assert ts.C == pytest.approx(1.0 / ts.eos_t_amb_k)


# ---- migration guard: stale [render.blackbody] key hard-errors -----------

def _fake_cfg(*, with_section=True, stale_blackbody_key=False,
              stale_eos_key=False):
    physics = SimpleNamespace()
    if with_section:
        # phi_exp set to 1/k_temp_to_kelvin so the eos_slope invariant
        # (quantize_scalar(phi_exp * k_temp_to_kelvin) == 65536) holds for
        # THIS pinned, non-default k_temp_to_kelvin too — the invariant is
        # tied to the pair, not to phi_exp alone (see temperature_scale.py's
        # _assert_invariants).
        physics.temperature_scale = SimpleNamespace(
            kelvin_ambient=111.0, k_temp_to_kelvin=5.0,
            phi_exp=0.2,
            eos_t_amb_k=290.0,
        )
    eos = SimpleNamespace()
    if stale_eos_key:
        eos.t_amb_k = 290.0
        eos.C = 1.0 / 290.0
    physics.eos = eos
    blackbody = SimpleNamespace()
    if stale_blackbody_key:
        blackbody.kelvin_ambient = 293.0
        blackbody.k_temp_to_kelvin = 2.0
    render = SimpleNamespace(blackbody=blackbody)
    return SimpleNamespace(physics=physics, render=render)


def test_load_raises_on_stale_blackbody_key():
    cfg = _fake_cfg(stale_blackbody_key=True)
    with pytest.raises(RuntimeError, match=r"\[physics\.temperature_scale\]"):
        temperature_scale.load(cfg)


def test_load_raises_on_stale_eos_key():
    """[physics.eos] t_amb_k/C moved to [physics.temperature_scale]
    (eos_t_amb_k / the derived C property) — P-K3, design §2/§3c. A stale
    key left behind on the ORIGINAL section must hard-error, the same way a
    stale [render.blackbody] key does above."""
    cfg = _fake_cfg(stale_eos_key=True)
    with pytest.raises(RuntimeError, match=r"\[physics\.temperature_scale\]"):
        temperature_scale.load(cfg)


def test_load_raises_on_missing_section():
    cfg = _fake_cfg(with_section=False)
    with pytest.raises(RuntimeError, match=r"\[physics\.temperature_scale\]"):
        temperature_scale.load(cfg)


def test_load_with_a_clean_fake_cfg_pins_the_non_default_values():
    """Sanity check for the guard tests above: a fake cfg WITHOUT the stale
    key loads fine and returns exactly what was pinned (non-default, so this
    cannot pass vacuously against the real config's defaults)."""
    cfg = _fake_cfg(stale_blackbody_key=False)
    ts = temperature_scale.load(cfg)
    assert ts.kelvin_ambient == 111.0
    assert ts.k_temp_to_kelvin == 5.0


def test_from_toml_raises_on_stale_blackbody_key(tmp_path):
    stale_toml = tmp_path / "stale_config.toml"
    stale_toml.write_text(
        "[physics.temperature_scale]\n"
        "kelvin_ambient = 293.0\n"
        "k_temp_to_kelvin = 3.0\n"
        "phi_exp = 0.3333333333333333\n"
        "eos_t_amb_k = 290.0\n"
        "\n"
        "[render.blackbody]\n"
        "kelvin_ambient = 293.0\n"
        "k_temp_to_kelvin = 2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=r"\[physics\.temperature_scale\]"):
        temperature_scale.from_toml(stale_toml)


def test_from_toml_raises_on_stale_eos_key(tmp_path):
    stale_toml = tmp_path / "stale_eos_config.toml"
    stale_toml.write_text(
        "[physics.temperature_scale]\n"
        "kelvin_ambient = 293.0\n"
        "k_temp_to_kelvin = 3.0\n"
        "phi_exp = 0.3333333333333333\n"
        "eos_t_amb_k = 290.0\n"
        "\n"
        "[physics.eos]\n"
        "t_amb_k = 290.0\n"
        "C = 0.0034482758620689655\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=r"\[physics\.temperature_scale\]"):
        temperature_scale.from_toml(stale_toml)


def test_from_toml_raises_on_missing_section(tmp_path):
    empty_toml = tmp_path / "empty_config.toml"
    empty_toml.write_text("[render.blackbody]\nkelvin_floor = 800.0\n",
                          encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"\[physics\.temperature_scale\]"):
        temperature_scale.from_toml(empty_toml)
