"""The larva law's Python side (arc #63 P3, docs/swarm_P3_impl.md §5.4, v4.3).

The law itself is C++ (``cpp/src/swarm_larva.h``, ONE FP_HD per-slot step run
by the CPU twin, the per-call CUDA kernel and the resident CUDA stage); there
is no Python copy of it (D1: the C++ CPU entry is the reference). What lives
here is what the HOST derives and checks before a launch:

* the reader the law senses through (``FIELD`` / ``FIELD_REDUCTION``: the RAW
  own-tile ``temperature``, ruling 11), which the coupling rows of
  ``exchange.COUPLING_TABLE`` and the C++ registry both name;
* the per-tick tumble probabilities from the per-second rates (D10);
* the larva's own cross-field rules (R2-R4), which ``swarm.SPECIES_RULES``
  runs after the shared no-tunnelling rule, on the exact ints the kernel's
  table rows carry (``swarm_tables``).

Must not import :mod:`simulation.swarm` (the import direction is
``swarm -> swarm_larva``).

Credit (the head-sweep law and its numbers):

* Luo L, Gershow M, Rosenzweig M, Kang K, Fang-Yen C, Garrity PA, Samuel ADT.
  "Navigational decision making in Drosophila thermotaxis." J Neurosci
  2010;30(12):4261-72. doi:10.1523/JNEUROSCI.4090-09.2010 -- run length and
  turn direction are regulated by the temperature a larva meets; the law's
  two-point comparison is a discrete stand-in for one head sweep (Luo
  describes the input as temporal variation during the sweep), not a claim
  that larvae compute a spatial gradient.
* Klein M, Afonso B, Vonner AJ, et al., Garrity PA, Samuel ADT. "Sensory
  determinants of behavioral dynamics in Drosophila thermotaxis." PNAS
  2015;112(2):E220-9. doi:10.1073/pnas.1416212112 -- the graded response the
  law omits (ACCEPTED GAP: sign-only).
* MacMillan HA, Sinclair BJ. "Mechanisms underlying insect chill-coma."
  J Insect Physiol 2011;57(1):12-20. doi:10.1016/j.jinsphys.2010.10.004 --
  CTmin and reversible chill coma (the T_ctmin / T_coma_exit rows).
* Monzon MA, Weidner LM, Rusch TW, Nehrozoglu S, Hamilton G. "High
  Temperature Limits of Survival and Oviposition of Phormia regina (Meigen)
  and Lucilia sericata (Meigen)." Insects 2022;13(11):991.
  doi:10.3390/insects13110991 -- adult blowflies over 24 h exposures; the
  instant 50 C kill is Erik's game compression of an exposure-time law,
  above the 24 h total-mortality point.
"""
from __future__ import annotations

from simulation import swarm_fixed

# The reader the larva senses through: the RAW own-tile temperature (ruling
# 11), labelled as exchange.reduce_center over a one-tile footprint returns
# exactly that tile. The C++ registry exports the same pair
# (bp.SWARM_SPECIES_LAWS), so the coupling rows cannot drift from the kernel.
FIELD, FIELD_REDUCTION = "temperature", "center"


def tumble_probabilities(row, ctx) -> tuple:
    """The per-tick tumble probabilities ``(p_base_q, p_alarm_q)`` (Q16) from
    the per-second rates: ``(rate_q * dt_q) >> 16`` -- first order, exact
    enough for ``rate * dt << 1`` (D10; no transcendental at the door)."""
    return ((int(row.tumble_rate_base_q) * int(ctx.dt_q)) >> swarm_fixed.FP_SHIFT,
            (int(row.tumble_rate_alarm_q) * int(ctx.dt_q)) >> swarm_fixed.FP_SHIFT)


def _check_tumble_probability(row, ctx):
    p_base, p_alarm = tumble_probabilities(row, ctx)
    if not (0 <= p_base <= p_alarm <= swarm_fixed.FP_ONE):
        return (f"the per-tick tumble probabilities must satisfy 0 <= p_base "
                f"<= p_alarm <= {swarm_fixed.FP_ONE} (got p_base={p_base}, "
                f"p_alarm={p_alarm} at dt_q={ctx.dt_q})")
    return None


def _check_coma_hysteresis(row, ctx):
    if not int(row.T_coma_exit_q) >= int(row.T_ctmin_q) + 1:
        return (f"T_coma_exit must lie at least one count above T_ctmin "
                f"(got T_ctmin_q={row.T_ctmin_q}, "
                f"T_coma_exit_q={row.T_coma_exit_q})")
    return None


def _check_turn_range(row, ctx):
    if not int(row.turn_min_q) <= int(row.turn_max_q):
        return (f"turn_min must not exceed turn_max (got "
                f"turn_min_q={row.turn_min_q}, turn_max_q={row.turn_max_q})")
    return None


# The larva's own invariants (R2-R4, §5.3): each protects something the
# kernel's integer arithmetic relies on, never tuning taste.
RULES = (
    ("tumble_probability", _check_tumble_probability),
    ("coma_hysteresis", _check_coma_hysteresis),
    ("turn_range", _check_turn_range),
)

__all__ = ["FIELD", "FIELD_REDUCTION", "tumble_probabilities", "RULES"]
