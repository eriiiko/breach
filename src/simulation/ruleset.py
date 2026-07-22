"""The ``Ruleset`` strategy seam — the turn structure + cost policy.

Design: ``docs/control_modularity_design_2026-07-22.md`` §3a. A game is
``Ruleset`` + ``ControlSource`` (+ ``AgentPolicy``) + config/content; this
module is the sim-side half of that split. ``Simulation`` owns one
``Ruleset`` instance, chosen at construction, and routes every
phase/AP-shaped decision through it instead of hard-coding WEGO.

P1 (2026-07-22, pure refactor, byte-identical digests/goldens): this patch
ONLY extracts the CURRENT behavior verbatim into :class:`TwoPhaseWEGO` — the
coupling-inventory items in the design doc's §2 (the ``Simulation.step()``
tail, the ``apply_action``/``undo_last_order`` AP chokepoint, and
``is_terminal``'s round-complete check). No logic changes, no reordering of
any load-bearing tick slot; ``Simulation.step()`` keeps slots 1-9e exactly
where they were and only the round-clock head/tail now calls out to
``self.ruleset`` instead of inlining the WEGO policy. ``_end_round`` stays a
``Simulation`` method (tests call it directly) — the ruleset's
``on_tick_end`` invokes it at the same tick position DET_END_PHASE2 always
fired at.

``ContinuousRealtime`` (no phases, no AP, no auto-pause) is P3 — not built
here.
"""
from __future__ import annotations

from simulation.combat import process_door_explosives
from simulation.orders import DET_START_PHASE1, DET_BETWEEN_PHASES, DET_END_PHASE2


class Ruleset:
    """Strategy interface a :class:`~simulation.simulation.Simulation` owns.

    Every method takes the owning ``sim`` explicitly (rather than closing
    over it) so a single stateless ``Ruleset`` instance could in principle
    serve multiple simulations — none of the shipped implementations need
    that, but it costs nothing and keeps the seam honest: all mutable
    round-clock state (``tick``, ``phase``, the ``_fired_*`` flags, AP)
    lives on ``sim`` / ``Unit``, never on the ruleset.
    """

    def on_round_start(self, sim) -> None:
        """Called every tick, early in :meth:`Simulation.step`, before any
        unit simulation. Tick-0-only work (DET_START_PHASE1, path-offset
        stamps) is the ruleset's own responsibility to gate."""
        raise NotImplementedError

    def on_tick_end(self, sim) -> None:
        """Called every tick, after the tick counter has advanced, at the
        very end of :meth:`Simulation.step`. Phase-boundary / round-end /
        auto-pause policy lives here."""
        raise NotImplementedError

    def validate_and_cost(self, sim, unit, order) -> bool:
        """Return ``True`` iff ``unit`` can afford ``order`` under this
        ruleset's cost policy, applying the cost as a side effect when it
        does. ``order.ap_cost`` must already be set by the caller
        (:meth:`Simulation.apply_action` sets it from the weapon/order
        table before calling in). Returning ``False`` must have NO side
        effect (nothing spent, nothing appended by the caller)."""
        raise NotImplementedError

    def refund(self, sim, unit, order) -> None:
        """Inverse of :meth:`validate_and_cost` for an order being popped
        (``Simulation.undo_last_order``)."""
        raise NotImplementedError

    def is_terminal(self, sim) -> bool:
        """Episode-boundary check for the AI training / Gymnasium contract
        (``Simulation.is_terminal``)."""
        raise NotImplementedError


class TwoPhaseWEGO(Ruleset):
    """The shipped WEGO round: two AP-gated planning phases, executed in
    one continuous run, auto-pausing at each phase/round boundary.

    Extracted VERBATIM from the pre-P1 ``Simulation`` — same tick numbers
    (tick 0 / ``ticks_per_phase`` / ``ticks_per_round``), same DET_* firing,
    same AP arithmetic, same order of operations. Every method below is a
    straight lift of code that used to be inlined at its call site; see the
    docstring of the corresponding ``Simulation`` method (pre-P1 git
    history) for the original context comments.
    """

    # ------------------------------------------------------------------
    # Round clock (Simulation.step() head/tail)
    # ------------------------------------------------------------------
    def on_round_start(self, sim) -> None:
        # Lifted verbatim from Simulation.step()'s tick-0 block (pre-P1
        # ~1092-1101): fires DET_START_PHASE1 once, resets path offsets for
        # team-0 units, and stamps initial unit positions.
        if sim.tick == 0 and not sim._fired_start_p1:
            process_door_explosives(
                sim.gmap, sim.edit_queue, sim.units, DET_START_PHASE1,
                sim.rng, events=sim.tick_events,
            )
            sim._fired_start_p1 = True
            # Initialise per-unit path offsets for this round.
            for u in sim.units:
                if u.team == 0:
                    u.path_tick_offset = 0
            # Stamp initial unit positions (legacy: done in _start_execution).
            sim.gmap.stamp_units(sim.units)

    def on_tick_end(self, sim) -> None:
        # Lifted verbatim from Simulation.step()'s tail (pre-P1
        # ~1096-1118), called right after `sim.tick` has been incremented
        # and `sim.real_time` advanced (those two lines stay in step() —
        # they are the tick clock itself, not WEGO policy).

        # Phase 1 -> Phase 2 boundary: fire between-phase explosives and
        # advance the phase counter. NO auto-pause — the round plays
        # through both phases smoothly in one execution. The split is a
        # mental planning aid for the player, not a sim interruption.
        if sim.tick == sim._ticks_per_phase and not sim._fired_between:
            process_door_explosives(
                sim.gmap, sim.edit_queue, sim.units, DET_BETWEEN_PHASES,
                sim.rng, events=sim.tick_events,
            )
            sim._fired_between = True
            sim.phase = 1

        # End of round: fire end-of-phase-2 explosives, convert zombies, reset.
        if sim.tick >= sim._ticks_per_round:
            if not sim._fired_end_p2:
                process_door_explosives(
                    sim.gmap, sim.edit_queue, sim.units, DET_END_PHASE2,
                    sim.rng, events=sim.tick_events,
                )
                sim._fired_end_p2 = True
            sim._end_round()
            sim.paused = True

    # ------------------------------------------------------------------
    # Cost policy (Simulation.apply_action / undo_last_order)
    # ------------------------------------------------------------------
    def validate_and_cost(self, sim, unit, order) -> bool:
        # Lifted verbatim from the per-phase AP gate inlined at every
        # apply_action order-type branch (pre-P1): `u.get_ap(phase) <
        # ap_cost` -> reject; else `u.spend_ap(phase, ap_cost)`.
        phase = order.phase
        if unit.get_ap(phase) < order.ap_cost:
            return False
        unit.spend_ap(phase, order.ap_cost)
        return True

    def refund(self, sim, unit, order) -> None:
        # Lifted verbatim from undo_last_order's AP refund line.
        if order.ap_cost > 0:
            unit.ap[order.phase] += order.ap_cost

    # ------------------------------------------------------------------
    # Episode boundary (Simulation.is_terminal)
    # ------------------------------------------------------------------
    def is_terminal(self, sim) -> bool:
        # Lifted verbatim from Simulation.is_terminal (pre-P1).
        if sim.tick >= sim._ticks_per_round:
            return True
        any_marine = any(u.team == 0 and u.alive for u in sim.units)
        any_zombie = any(u.team == 1 and u.alive for u in sim.units)
        # If we have neither, the simulation is degenerate; treat as terminal.
        if not any_marine and not any_zombie:
            return True
        # If the player had marines and they're all dead -> terminal.
        if not any_marine and sim.units:
            return True
        return False


__all__ = ["Ruleset", "TwoPhaseWEGO"]
