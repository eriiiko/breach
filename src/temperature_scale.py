"""Canonical game-T -> Kelvin map + EOS ambient-pressure calibration (P-K1).

Single source of truth for ``[physics.temperature_scale]``
(docs/temperature_scale_unification_design_2026-08-13.md §2/§3d) — radiation
bake, render blackbody, hover readout, and the fire-tuning tools all read
this ONE table so "what does T_game=0 mean in Kelvin" cannot drift between
consumers again (the problem this arc fixes, §1).

Two entry points share ONE default set (critique F7 — no Namespace-vs-dict
drift between them):

- :func:`load` — reads the live ``config.CFG`` singleton (the game/render
  path). No module-level caching: every call re-derives from ``cfg`` fresh,
  so ``CFG.reload()`` (F5 in-game) is picked up automatically — there is no
  separate cache to invalidate.
- :func:`from_toml` — a standalone loader for tools that don't want the full
  game-config machinery (reads a config.toml path directly with
  ``tomllib``).

Both apply the SAME migration guards (design §2: "guards apply on the REAL
config path") and the SAME load-time invariant assert. A caller that wants a
soft fallback instead of a hard error (e.g. a tool pointed at a foreign or
partially-migrated config) catches narrowly around the call itself — see
``tools/fire_tune_plot.py::kelvin_map`` — this module never swallows
anything on its own.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import NamedTuple, Optional

from simulation.gas_fixed import quantize_scalar

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"

SECTION_NAME = "temperature_scale"
_DESIGN_DOC = "docs/temperature_scale_unification_design_2026-08-13.md"

# Per-key defaults — the §2 canonical values. Exactly one place both entry
# points read from (critique F7: Namespace-vs-dict drift).
DEFAULTS = {
    "kelvin_ambient": 293.0,
    "k_temp_to_kelvin": 1.0,
    "phi_exp": 1.0,
    "eos_t_amb_k": 293.0,
}

# The stale keys [render.blackbody] must no longer carry (design §2).
_STALE_BLACKBODY_KEYS = ("kelvin_ambient", "k_temp_to_kelvin")

# The stale keys [physics.eos] must no longer carry (design §2/§3c, P-K3):
# t_amb_k/C moved to this table's eos_t_amb_k / the derived C property.
_STALE_EOS_KEYS = ("t_amb_k", "C")


class TemperatureScale(NamedTuple):
    """The loaded table plus its derived quantities.

    ``eos_slope`` and ``C`` are computed, not authored — see design §2 for
    the single formula each is derived from.
    """
    kelvin_ambient: float
    k_temp_to_kelvin: float
    phi_exp: float
    eos_t_amb_k: float

    def to_kelvin(self, t_game: float) -> float:
        """Game ΔT -> absolute pseudo-Kelvin: ``kelvin_ambient +
        k_temp_to_kelvin * t_game`` (design §2's canonical map)."""
        return self.kelvin_ambient + self.k_temp_to_kelvin * t_game

    def from_kelvin(self, kelvin: float) -> float:
        """Inverse of :meth:`to_kelvin`: absolute Kelvin -> game ΔT."""
        return (kelvin - self.kelvin_ambient) / self.k_temp_to_kelvin

    @property
    def eos_slope(self) -> float:
        """``phi_exp * k_temp_to_kelvin`` — 1.0 exactly while phi_exp is
        value-frozen (design §2); the EOS identity mechanism P-K3 folds this
        into ``s_eos_q`` on the host."""
        return self.phi_exp * self.k_temp_to_kelvin

    @property
    def C(self) -> float:
        """``1 / eos_t_amb_k`` — the EOS pressure-calibration constant. G12
        (docs/fire_g12_one_map_patch_2026-08-31.md) dissolved ruling 6's
        exception: eos_t_amb_k now EQUALS kelvin_ambient (293), so
        C = 1/293 (was 1/290). Still authored as its own field, not derived
        from kelvin_ambient, so a future frame split could reintroduce the
        exception without touching this property's shape."""
        return 1.0 / self.eos_t_amb_k


def _build(d: dict) -> TemperatureScale:
    ts = TemperatureScale(
        kelvin_ambient=float(d.get("kelvin_ambient", DEFAULTS["kelvin_ambient"])),
        k_temp_to_kelvin=float(d.get("k_temp_to_kelvin", DEFAULTS["k_temp_to_kelvin"])),
        phi_exp=float(d.get("phi_exp", DEFAULTS["phi_exp"])),
        eos_t_amb_k=float(d.get("eos_t_amb_k", DEFAULTS["eos_t_amb_k"])),
    )
    _assert_invariants(ts)
    return ts


def _assert_invariants(ts: TemperatureScale) -> None:
    """Load-time asserts, via the EXISTING ``gas_fixed.quantize_scalar`` —
    no third hand-rolled rounding convention (design §2, the 139b/A8
    hazard). phi_exp is value-frozen this arc: the quantized EOS slope must
    land on exactly 65536 (== 1.0 in Q16.16)."""
    q = quantize_scalar(ts.eos_slope)
    assert q == 65536, (
        f"[physics.temperature_scale] eos_slope (phi_exp * k_temp_to_kelvin) "
        f"quantizes to {q}, not 65536 (== 1.0 Q16.16). phi_exp is "
        f"value-frozen this arc ({_DESIGN_DOC} §2) — retuning it is the "
        f"storm session's ruling, not a config edit here."
    )


def _migration_error(message: str) -> RuntimeError:
    return RuntimeError(
        f"{message} See [physics.temperature_scale] in {_DESIGN_DOC} §2."
    )


def load(cfg=None) -> TemperatureScale:
    """Load ``[physics.temperature_scale]`` from a live config object.

    ``cfg`` defaults to the global ``config.CFG`` singleton (attribute-
    Namespace sections, see config.py); a caller may pass any object with
    the same ``.physics.temperature_scale`` / ``.render.blackbody`` shape
    (tests use this to pin a non-default value without touching the real
    config.toml).
    """
    if cfg is None:
        from config import CFG as cfg

    physics = getattr(cfg, "physics", None)
    section = getattr(physics, SECTION_NAME, None)
    if section is None:
        raise _migration_error(
            "config is missing [physics.temperature_scale]."
        )

    blackbody = getattr(getattr(cfg, "render", None), "blackbody", None)
    stale = [k for k in _STALE_BLACKBODY_KEYS
             if getattr(blackbody, k, None) is not None]
    if stale:
        raise _migration_error(
            f"[render.blackbody] still carries {stale} — these moved to "
            f"[physics.temperature_scale]; remove them from "
            f"[render.blackbody]."
        )

    eos = getattr(physics, "eos", None)
    stale_eos = [k for k in _STALE_EOS_KEYS
                 if getattr(eos, k, None) is not None]
    if stale_eos:
        raise _migration_error(
            f"[physics.eos] still carries {stale_eos} — these moved to "
            f"[physics.temperature_scale] (eos_t_amb_k / the derived C "
            f"property); remove them from [physics.eos]."
        )

    d = {k: getattr(section, k) for k in DEFAULTS if hasattr(section, k)}
    return _build(d)


def from_toml(path: Optional[Path] = None) -> TemperatureScale:
    """Standalone loader for tools: read ``[physics.temperature_scale]``
    straight out of a config.toml file (default: the repo's own), bypassing
    ``config.CFG`` entirely.

    Applies the SAME migration guards as :func:`load` (design §2: guards
    apply on the real config path) — this function raises, it does not fall
    back. Tools that want a soft fallback catch narrowly around the call
    (see ``tools/fire_tune_plot.py::kelvin_map``).
    """
    p = Path(path) if path is not None else CONFIG_PATH
    with open(p, "rb") as f:
        raw = tomllib.load(f)

    physics = raw.get("physics", {})
    if SECTION_NAME not in physics:
        raise _migration_error(
            f"{p} is missing [physics.temperature_scale]."
        )

    blackbody = raw.get("render", {}).get("blackbody", {})
    stale = [k for k in _STALE_BLACKBODY_KEYS if k in blackbody]
    if stale:
        raise _migration_error(
            f"{p}: [render.blackbody] still carries {stale} — these moved "
            f"to [physics.temperature_scale]; remove them from "
            f"[render.blackbody]."
        )

    eos = physics.get("eos", {})
    stale_eos = [k for k in _STALE_EOS_KEYS if k in eos]
    if stale_eos:
        raise _migration_error(
            f"{p}: [physics.eos] still carries {stale_eos} — these moved to "
            f"[physics.temperature_scale] (eos_t_amb_k / the derived C "
            f"property); remove them from [physics.eos]."
        )

    return _build(physics[SECTION_NAME])
