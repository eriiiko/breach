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

from simulation.ai_zombie import convert_marines_to_zombies
from simulation.combat import process_door_explosives
from simulation.orders import DET_START_PHASE1, DET_BETWEEN_PHASES, DET_END_PHASE2


class Ruleset:
    """Strategy interface a :class:`~simulation.simulation.Simulation` owns.

    ``drives_units`` (class attribute): does this ruleset own the per-tick
    unit-simulation slots itself? ``False`` — the shipped default — means
    ``Simulation.step`` runs its historical slot 3 (``_update_player_movement``)
    and slot 4 (``process_shooting``) bodies verbatim. ``True`` means the
    ruleset replaces both with :meth:`drive_units`, which is how OnePhaseWEGO
    substitutes the compiled timeline for the phase-indexed order scan without
    disturbing a single line of the legacy path.

    Every method takes the owning ``sim`` explicitly (rather than closing
    over it) so a single stateless ``Ruleset`` instance could in principle
    serve multiple simulations — none of the shipped implementations need
    that, but it costs nothing and keeps the seam honest: all mutable
    round-clock state (``tick``, ``phase``, the ``_fired_*`` flags, AP)
    lives on ``sim`` / ``Unit``, never on the ruleset.
    """

    #: See the class docstring. Overridden to True by OnePhaseWEGO.
    drives_units = False

    def drive_units(self, sim) -> None:
        """One tick of unit simulation, when ``drives_units`` is True.
        Replaces ``Simulation.step``'s slots 3 and 4 entirely."""
        raise NotImplementedError

    def on_orders_changed(self, sim, unit) -> None:
        """Called after ``unit``'s order queue is mutated (an order placed or
        undone), so a ruleset that precompiles can rebuild. No-op by default —
        the shipped rulesets recompute paths from ``Simulation`` instead."""
        pass

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

    # ------------------------------------------------------------------
    # Round-clock geometry (OnePhaseWEGO, onephase_wego design §2/§13)
    # ------------------------------------------------------------------
    # Concrete on the base — NOT abstract — so the two shipped rulesets keep
    # working untouched. The defaults describe the pre-existing world: a
    # round is CFG.clock.ticks_per_round long and ``sim.tick`` already counts
    # within it, because TwoPhaseWEGO rewinds the tick at every boundary.
    # OnePhaseWEGO overrides all three: its tick is FREE-RUNNING (§2.1 of the
    # kickoff doc — a rewound clock cannot carry cooldowns across a seam,
    # which §13 requires), so within-round position becomes a modulo.

    def ticks_per_round(self, sim) -> int:
        """Length of one round, in ticks."""
        return sim._ticks_per_round

    def round_tick(self, sim) -> int:
        """Ticks elapsed since the CURRENT round began (0 .. len-1)."""
        return sim.tick

    def round_index(self, sim) -> int:
        """How many complete rounds have been executed before this one."""
        return sim.turn_number - 1


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


class ContinuousRealtime(Ruleset):
    """Direct-action realtime: no rounds, no phases, no AP, no auto-pause.

    P3 (control-modularity §3a). The sibling of :class:`TwoPhaseWEGO` chosen
    by ``--control gamepad``. The load-bearing tick body (``Simulation.step``
    slots 1-9e) is identical; only the round-clock head/tail differ, and here
    they very nearly vanish:

    - ``on_round_start`` / ``on_tick_end``: NO phase advance, NO DET
      phase-boundary explosive slots, NO auto-pause, NO ``_end_round``
      teardown or tick rewind. ``Simulation.step`` still increments
      ``sim.tick`` (that line is the clock itself, not WEGO policy) — the tick
      is a free-running monotonic counter, never wrapped.

    - Replacing ``_end_round``'s housekeeping (§3a):

      * **Zombie conversion is death-triggered, not an end-of-round batch.**
        ``on_tick_end`` runs :func:`convert_marines_to_zombies` every tick, so
        a marine a zombie killed this tick is a walking zombie by the next
        tick's AI (the function is idempotent — it clears each unit's
        ``killed_by_zombie`` flag as it converts, so re-running it every tick
        costs one bool read per unit and never double-converts). No round has
        to end for the dead to rise.

      * **Corpses blocking physics is per-tick stamp semantics — and needs no
        code here.** ``Simulation.step`` slot 6 (``gmap.stamp_units``) rebuilds
        ``obstacles`` (= walls only) and the soft ``dyn_*`` occluder fields
        from the *living* units every tick, ruleset-independently. Dead units
        are already filtered out there (they are soft occluders while alive,
        nothing once dead), so continuous play never regresses corpse physics:
        the WEGO ``_end_round`` obstacle reset was only undoing that round's
        accumulated state, which per-tick stamping already keeps clean.

    - ``validate_and_cost`` / ``refund``: alive + physical preconditions ONLY.
      ``Simulation.apply_action`` already rejects dead / zombie / out-of-
      inventory actors before calling in, so the cost policy is a pure pass
      (return ``True``, spend nothing) — this ruleset NEVER touches the
      ``Unit`` AP fields, and ``refund`` is a no-op.

    - ``is_terminal``: one team eliminated. Continuous play has no round to
      complete, so the episode boundary (the AI/Gymnasium contract) is "the
      fight is over": no living marine, or no living zombie. A single-team or
      unit-free sandbox therefore reads terminal immediately — correct (there
      is no fight), and irrelevant to the human gamepad loop, which never
      calls ``is_terminal``.
    """

    def on_round_start(self, sim) -> None:
        # No tick-0 round setup: no DET_START_PHASE1, no path-offset reset, no
        # initial unit stamp (slot 6 stamps every tick before physics anyway).
        pass

    def on_tick_end(self, sim) -> None:
        # No phase boundary, no auto-pause, no round teardown. Death-triggered
        # zombie conversion stands in for the WEGO end-of-round batch (see the
        # class docstring): idempotent, so running it each tick is immediate
        # conversion, not a per-tick storm.
        convert_marines_to_zombies(sim.units)

    def validate_and_cost(self, sim, unit, order) -> bool:
        # No AP, no phase. The caller (apply_action) has already gated alive /
        # not-zombie / inventory; there is nothing left to charge.
        return True

    def refund(self, sim, unit, order) -> None:
        # No AP was spent — nothing to give back.
        pass

    def is_terminal(self, sim) -> bool:
        any_marine = any(u.team == 0 and u.alive for u in sim.units)
        any_zombie = any(u.team == 1 and u.alive for u in sim.units)
        # Terminal iff one side is gone (or the world is empty).
        return not (any_marine and any_zombie)


class OnePhaseWEGO(Ruleset):
    """The turn-formula redesign: ONE phase per round, time as the only
    currency (``docs/onephase_wego_design_2026-07-28.md``).

    Built BESIDE :class:`TwoPhaseWEGO`, which stays shipped and byte-identical
    until Erik blesses this one (design §18). Nothing in this class is reachable
    from another ruleset, so no existing golden/digest can move while it grows.

    What it is (design §2/§3/§13):

    - **No phases.** No phase tags on orders, no per-phase AP pools, no
      Tab toggle, no DET_BETWEEN_PHASES slot semantics.
    - **One ~4 s round** (``CFG.clock.round_duration_seconds``), flow
      PLAN (paused) -> EXECUTE (96 ticks @ 24 Hz) -> PLAN.
    - **AP is dead.** :meth:`validate_and_cost` charges nothing — the round's
      seconds ARE the budget, and actions cost ticks (durations, cooldowns,
      the GCD). That economy lives on the timeline (P3), not here.

    THE MONOTONIC CLOCK (the load-bearing structural difference)
    ------------------------------------------------------------
    ``TwoPhaseWEGO`` rewinds ``sim.tick`` to 0 at every boundary, which is
    exactly why its ``_end_round`` must scrub ``last_fire_tick``,
    ``reload_done_tick`` and the spray burst — a carried deadline compared
    against a rewound clock is nonsense (the hazard its own comments call
    out). Design §13 demands the OPPOSITE: cooldowns, GCD, overwatch state,
    ambush groups, in-flight projectiles and fires ALL persist across the
    boundary, because "round boundaries are invisible seams".

    So under this ruleset ``sim.tick`` is **free-running and monotonic**, and
    within-round position is ``tick % ticks_per_round``. Every timer in the
    ruleset is an ABSOLUTE tick deadline, which makes seam-crossing
    arithmetically invisible rather than something to special-case — there is
    no teardown to write, and no "cram an action at t=3.9 to reset it"
    exploit (§3) because nothing resets.

    What the boundary does, therefore, is exactly two things: **pause** (the
    player may issue orders now) and hand off to :meth:`on_round_boundary`,
    whose entire body is housekeeping that changes no observable world state.
    Notably ABSENT versus ``TwoPhaseWEGO._end_round``: the end-of-round
    integer-tile position snap (design §4 removes it outright — at 4 s rounds
    it would fire 2.5x as often and be visible), the order clear, the AP
    refill, the obstacle reset, and the tick rewind.
    """

    #: This ruleset runs the compiled timeline instead of the phase-indexed
    #: order scan — see :mod:`simulation.timeline`.
    drives_units = True

    def drive_units(self, sim) -> None:
        # Deferred import: timeline imports combat, which imports plenty;
        # keeping it lazy holds ruleset.py as import-light as it has always
        # been (tests import it bare).
        from simulation.timeline import drive_units as _drive
        _drive(sim)

    def on_orders_changed(self, sim, unit) -> None:
        """Recompile this unit's timeline (design §3).

        A plan is never patched in place — it is rebuilt from the pending
        order queue — so what the planning UI shows and what the executor runs
        cannot drift apart. Completed steps have already retired their orders,
        so a recompile can never re-run an action the unit already took.
        """
        from simulation.timeline import compile_plan
        unit.plan = compile_plan(sim, unit)

    # ------------------------------------------------------------------
    # Round-clock geometry
    # ------------------------------------------------------------------
    def ticks_per_round(self, sim) -> int:
        return sim._onephase_ticks_per_round

    def round_tick(self, sim) -> int:
        # Free-running tick -> within-round position. The one modulo that
        # replaces TwoPhaseWEGO's whole rewind-and-scrub teardown.
        return sim.tick % self.ticks_per_round(sim)

    def round_index(self, sim) -> int:
        return sim.tick // self.ticks_per_round(sim)

    # ------------------------------------------------------------------
    # Round clock (Simulation.step() head/tail)
    # ------------------------------------------------------------------
    def on_round_start(self, sim) -> None:
        # Tick-0-only world setup. No DET_START_PHASE1: scheduled detonations
        # (§12) are absolute ticks chosen by the player anywhere in the round,
        # resolved by the timeline executor, not by phase-boundary slots. No
        # path-offset reset either — a path offset is a within-round rewind
        # artifact, and this clock never rewinds.
        if sim.tick == 0:
            sim.gmap.stamp_units(sim.units)

    def on_tick_end(self, sim) -> None:
        # Death-triggered conversion, every tick (the ContinuousRealtime rule,
        # not TwoPhaseWEGO's end-of-round batch): a marine killed by a zombie
        # rises on the next tick's AI rather than waiting for the seam. The
        # function is idempotent — it clears each unit's killed_by_zombie flag
        # as it converts — so per-tick invocation costs one bool read per unit
        # and never double-converts. Batching it at the boundary would make
        # the seam VISIBLE, which is precisely what §13 forbids.
        convert_marines_to_zombies(sim.units)

        # Boundary test on the MONOTONIC tick. `sim.tick` has already been
        # incremented by step(), so this fires on the tick that COMPLETES a
        # round (96, 192, ...) and never at tick 0.
        if sim.tick > 0 and sim.tick % self.ticks_per_round(sim) == 0:
            self.on_round_boundary(sim)

    def on_round_boundary(self, sim) -> None:
        """The invisible seam (§13): pause for planning, and nothing else that
        the world can see.

        The only work here is housekeeping that is unobservable by
        construction: dropping already-detonated projectiles (they are inert —
        the projectile loop skips them every tick — so pruning them changes no
        trajectory, it just stops an all-day session from growing the list
        without bound), and bumping the human-facing round counter.
        """
        sim.projectiles = [p for p in sim.projectiles if not p.detonated]
        sim.turn_number += 1
        sim.paused = True

    # ------------------------------------------------------------------
    # Cost policy — time is the only currency (§3)
    # ------------------------------------------------------------------
    def validate_and_cost(self, sim, unit, order) -> bool:
        """AP is dead: there is nothing to charge at order-placement time.

        The real economy is the TIMELINE — an action costs the ticks its
        registry row says it costs (duration), plus what its cooldown and the
        GCD deny afterwards. That is enforced when the plan compiles and
        executes (P3), not by a gate here; a plan that overruns the round
        simply does not finish inside it, which is the intended feedback.
        ``Simulation.apply_action`` has already rejected dead / zombie /
        out-of-inventory actors before calling in.
        """
        return True

    def refund(self, sim, unit, order) -> None:
        # Nothing was spent at placement — nothing to give back. The ticks an
        # order would have consumed were never taken from a pool; undoing it
        # just shortens the compiled timeline.
        pass

    # ------------------------------------------------------------------
    # Episode boundary
    # ------------------------------------------------------------------
    def is_terminal(self, sim) -> bool:
        """One side eliminated.

        Deliberately NOT "the round is complete" (the ``TwoPhaseWEGO`` rule):
        with a free-running clock there is no round-completion tick to compare
        against, and a 4 s round is a planning cadence rather than an episode.
        The WEGO cadence is a natural RL action interface (design §1), so the
        episode boundary that matters to training is the fight ending.
        """
        any_marine = any(u.team == 0 and u.alive for u in sim.units)
        any_zombie = any(u.team == 1 and u.alive for u in sim.units)
        return not (any_marine and any_zombie)


__all__ = ["Ruleset", "TwoPhaseWEGO", "ContinuousRealtime", "OnePhaseWEGO"]
