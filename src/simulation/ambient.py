"""Ambient boundary derived constants (BC build — boundary_conditions_spec_2026-07-19 §1).

Planetside maps (`boundary = "ambient"`) hold an infinite ambient reservoir at the
border ring. The physics needs three integer quantities derived from the authored
dials, and BOTH the loader (which logs them) and `GameMap` (which seeds t=0 and, in
B3, pins the ring) must agree to the LSB — so the derivation lives here, once.

**N-primary (critique 2026-07-19).** The EOS materializes pressure every tick as
``p* = C · N_total · (T + T_AMB_K)`` through TRUNCATING q16 multiplies
(`eos_solver.cpp:463-476`). Every reachable ``p*`` is therefore a multiple of
``round(T_AMB_K)`` raw counts (~290), so ``quantize(1.0 atm) == 65536`` has *no*
integer ``N_total`` preimage: "derive N from a target P" is ill-posed for ~289/290
of dial values. We go the well-posed direction instead:

  * ``N_total := quantize(p_amb)`` — the N planes are primary;
  * split into O2 / inert-N2 by ``o2_frac`` (round-half-up + exact complement, the
    same idiom as the air_init seed, `gamemap.py:442-446`);
  * the **effective pin** ``P_amb := p*(N_total, ΔT=0)`` — the sim's own chain applied
    to the ambient N, so the ring pin and the interior materialized pressure agree.

At the Earth-normal defaults (p_amb=1.0, o2_frac=0.21) this yields N_total=65536,
O2=13763, N2=51773, and an effective pin of **65540 raw (1.000061 atm)** — NOT 65536.
That 4-count offset is the quantization lattice, not an error; it is what keeps a
sealed planetside room's interior trajectory flat (spec §6 gate 1).

Pure integer, no libm, deterministic cross-machine.
"""
from __future__ import annotations

from dataclasses import dataclass

import temperature_scale
from simulation import gas_fixed as _gas_fx

FP_ONE = _gas_fx.FP_ONE                  # 65536

# Pinned EOS constants (docs/eos_refactor_decisions.md 2026-07-10). Source of
# truth is [physics.temperature_scale] (eos_t_amb_k / the derived C property)
# via the canonical accessor — P-K3, temperature_scale_unification_design_
# 2026-08-13.md §2/§3c; [physics.eos] no longer carries t_amb_k/C at all.
# Derived once at import (no module-level caching in the accessor itself, but
# these two names are load-bearing defaults for callers below and for
# pump_system.py's DEFAULT_C/DEFAULT_T_AMB_K import, so they snapshot here the
# same way `from config import CFG` snapshots the singleton). Callers that
# have the live config pass its values so the pin matches runtime under any
# override.
_ts = temperature_scale.load()
DEFAULT_C = _ts.C
DEFAULT_T_AMB_K = _ts.eos_t_amb_k

# Dial defaults + validation bounds (spec §4).
DEFAULT_P_AMB = 1.0
DEFAULT_O2_FRAC = 0.21
# sky_tau_s — the vertical-mixing timescale for the SKY EXCHANGE pass
# (docs/sky_exchange_design_2026-07-24.md §1.2). Seconds; the per-tick relaxation
# rate is λ = dt_tick / sky_tau_s. This PARSE DEFAULT is 0.0 == DORMANT: an
# existing planetside level with no sky_tau_s key keeps today's edge-only refill,
# BYTE-IDENTICAL (gate a/b). The design's RECOMMENDED authored value is 60.0 s
# (bench-calibrated across {30,60,120} at P3); it is written into a level's
# [ambient] table when the feature is blessed — not baked in as the default, so
# no unblessed level silently changes behaviour.
DEFAULT_SKY_TAU_S = 0.0
DEFAULT_SPONGE_WIDTH = 8
# σ_max — a pressure-sponge mass on the level-0 Helmholtz diagonal (spec §3
# rung 1). B3 CALIBRATION OUTCOME (2026-07-19): the σ-pressure-sponge does NOT
# absorb — it pins P′ toward ambient (a soft Dirichlet), which REFLECTS acoustic
# fronts rather than absorbing them (a pressure-release BC is a perfect
# reflector; measurement across ring distances showed reflection monotonically
# equal-or-WORSE as σ_max rises). The momentum carrier `u` sails through the
# band to the hard ring and bounces. So the calibrated default is 0 (the dial
# is wired + live for experimentation, but ships OFF — pin-only is the better-
# measured behavior). The promising absorber is rung 2 (velocity damping in the
# band, the `sponge_u_damp`/k_max dial), which measurement showed DOES cut
# reflection — but wiring + a robust reflection gate is an open item for Erik's
# absorber design call (spec §0.5: "imperceptibility, not perfection"; the B5
# feel gate). See the B3b commit message + build report for the measurements.
DEFAULT_SPONGE_STRENGTH = 0
# k_max — the u-damping band coefficient (spec §3 rung 2, the REAL absorber,
# B3c). B3c CALIBRATION OUTCOME (2026-07-19): the u-damping band demonstrably
# absorbs the acoustic reflection (velocity is the momentum carrier the σ-sponge
# never touched) — the reflection-vs-k curve falls monotonically and knees at
# ~0.9·FP_ONE (saturating past it). Pinned to the knee. Effectiveness scales
# with the BAND WIDTH (sponge_width): at width 8 the front only gets ~1 damping
# bite (~2.5% residual, imperceptible at the ~0.02 atm transient amplitudes);
# width 16 reaches the ≤2% gate, width ≥24 the ≥2× margin. Width is the
# author's absorption-vs-interior dial (spec §4). NOTE: the near-range residual
# is dominated by the ELLIPTIC pressure solve's instantaneous image response to
# domain size (the MG solve is global), which is NOT an acoustic echo and is not
# absorbable by any Q16-friendly velocity/pressure treatment the spec allows —
# see the B3c report. σ (DEFAULT_SPONGE_STRENGTH) stays 0 (rung-1 reflects).
DEFAULT_SPONGE_U_DAMP = 58982            # 0.9 * FP_ONE — the damping knee
# σ_max may exceed FP_ONE: the ambient row mass is ~1/1409 real and face
# conductances ~1 real, so a useful sponge extends the Dirichlet ring inward and
# needs σ ≫ FP_ONE (the int64 row mass, M_CAP 2³⁸, keeps this overflow-safe). An
# FP_ONE cap would reject the gate-calibrated value (spec §4, v2.1 fix).
SPONGE_STRENGTH_MAX = 256 * FP_ONE
SPONGE_U_DAMP_MAX = FP_ONE                # k_max < 1.0 (a ≥1 multiply flips u sign)


def _mul_q16(a: int, b: int) -> int:
    """Replicate ``fixedpoint::mul_q16``: (int64) a*b >> 16, arithmetic shift
    toward -inf. All ambient quantities are non-negative, so Python ``>>`` on the
    exact product matches the C++ arithmetic shift bit-for-bit."""
    return (int(a) * int(b)) >> 16


def effective_pin(n_total_q: int,
                  c: float = DEFAULT_C,
                  t_amb_k: float = DEFAULT_T_AMB_K) -> int:
    """``p*(N_total, ΔT=0)`` through the sim's own truncating chain
    (`eos_solver.cpp:469-475`). Returns the raw Q16.16 pin the physics uses."""
    c_q = _gas_fx.quantize_scalar(c)
    t_amb_q = _gas_fx.quantize_scalar(t_amb_k)      # t_abs at ΔT (temperature) == 0
    cn = _mul_q16(c_q, int(n_total_q))
    p = _mul_q16(cn, t_amb_q)
    return 0 if p < 0 else p                          # the EOS floor (pstar < 0 -> 0)


@dataclass(frozen=True)
class AmbientConfig:
    """The parsed + derived planetside boundary constants. ``None`` on space maps."""
    p_amb: float                 # authored target pressure (atm)
    o2_frac: float               # authored O2 mole fraction
    sponge_width: int            # absorber band depth, BASE tiles (0 == hard ring)
    sponge_strength: int         # σ_max, raw Q16 (pinned by B3 calibration)
    sponge_u_damp: int           # k_max, raw Q16 (rung-2 mop-up; 0 == dormant)
    sky_tau_s: float             # sky-exchange vertical-mixing timescale, s (0 == dormant)
    n_o2_q: int                  # ambient O2 plane value, raw Q16
    n_n2_q: int                  # ambient inert-N2 plane value, raw Q16
    o2_frac_q: int               # ambient O2 MOLE FRACTION, raw Q16 (13763 at 0.21) —
                                 # the sky-exchange composition target uses this same
                                 # quantized fraction the ring N-split derives from (one
                                 # source of truth with the ring clamp; design §1.2)
    pin_q: int                   # effective P_amb the physics pins/materializes, raw Q16

    @property
    def n_total_q(self) -> int:
        return self.n_o2_q + self.n_n2_q


def derive_ambient(p_amb: float = DEFAULT_P_AMB,
                   o2_frac: float = DEFAULT_O2_FRAC,
                   sponge_width: int = DEFAULT_SPONGE_WIDTH,
                   sponge_strength: int = DEFAULT_SPONGE_STRENGTH,
                   sponge_u_damp: int = DEFAULT_SPONGE_U_DAMP,
                   sky_tau_s: float = DEFAULT_SKY_TAU_S,
                   c: float = DEFAULT_C,
                   t_amb_k: float = DEFAULT_T_AMB_K) -> AmbientConfig:
    """Build an :class:`AmbientConfig` from the authored dials (N-primary)."""
    n_total = int(_gas_fx.quantize_scalar(p_amb))
    o2_frac_q = int(_gas_fx.quantize_scalar(o2_frac))
    o2 = (n_total * o2_frac_q + (1 << 15)) >> 16        # round-half-up (seed idiom)
    n2 = n_total - o2                                     # exact complement, no LSB leak
    pin = effective_pin(n_total, c, t_amb_k)
    return AmbientConfig(
        p_amb=float(p_amb), o2_frac=float(o2_frac),
        sponge_width=int(sponge_width), sponge_strength=int(sponge_strength),
        sponge_u_damp=int(sponge_u_damp), sky_tau_s=float(sky_tau_s),
        n_o2_q=int(o2), n_n2_q=int(n2), o2_frac_q=int(o2_frac_q), pin_q=int(pin),
    )
