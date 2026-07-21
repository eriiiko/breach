"""The automatic airlock_controller — Arc B patch B5 (impl doc §7, HUMAN-TEST).

Design: docs/arc_b_impl_2026-07-21.md §7 (v2, D12 folded). This module carries
the SCHEMA (fields / signals / inputs) + the integer STATE ENUM for the L0
``airlock_controller`` state machine. The sim-side RUNTIME (the transition
function over the SignalBus) lives in :mod:`simulation.logic_nodes` beside the
other node evaluators, exactly as the pump / sensor / door runtimes live
sim-side. This package stays IMPORT-LIGHT (stdlib only — design §3b, CI-tested
in tests/test_entities_import_light.py): a controller class is pure declaration.

The controller is an L0 **logic node** (``LOGIC_NODE = True``): its command
signals are prev-read/next-write NODE outputs the SignalBus stages at 9e(b) and
SWAPS into ``pub`` at 9e(e) — so a command reaches its door / pump ONE tick
later (the node-hop latency, §2c), like every other node output. It reads the
world ONLY through wired inputs (SignalBus, never a direct entity reference — so
its runtime list position is never observable, §9) and drives its doors / pump
ONLY through wired command signals. State + dwell are synced runtime rows.

The single automatic cycle (§7), the acceptance showcase:

    IDLE          presence==0 → stay; >=1 in chamber → CLOSING (busy=1)
    CLOSING       drive both doors close; both (alive AND is_open==0) → EQUALIZE
                  (a door reading alive==0 is a BREACH, not sealed — abort to
                   FAULT, never pump into a hole — D12)
    EQUALIZE      drive pump toward the far target; at_target → OPEN_FAR
    OPEN_FAR      drive the far door open; presence==0 (cleared) → RESEAL
    RESEAL        drive the far door close; far (alive AND is_open==0) →
                  REPRESSURIZE  (far reads alive==0 → BREACH → FAULT, D12)
    REPRESSURIZE  drive pump toward the near target; at_target → IDLE
                  (IDLE reopens the near door — the folded OPEN_NEAR step)
    FAULT         breach detected: doors released to manual, busy=0, latched

**Accepted v1 gaps (documented so they are chosen, §7):** one chamber / two
doors / one pump; level (not edge) triggers; the only anti-cycle is
occupancy-blocks-close (a living unit on a door span holds CLOSING with no
timeout — the permanent stall, §15.2). The presence plate MUST be authored OFF
the door spans (N5) or the unit that triggered the cycle blocks its own airlock.

**Single-pump limitation (v1, honest):** the built pump (§6) carries ONE fixed
``target_atm`` and latches ``at_target`` against it. A faithful TWO-pressure
airlock (evacuate to the far side, repressurize to the near side) would need the
controller to RETARGET the pump per phase — a command the v1 pump does not
expose. So in v1 the pump physically settles at its authored ``target_atm`` and
BOTH the EQUALIZE and REPRESSURIZE transitions gate on that single ``at_target``.
The controller's ``target_far_atm`` / ``target_near_atm`` select only the pump
DRIVE DIRECTION per phase (inject toward the higher pressure, extract toward the
lower). Full two-pressure cycling rides a retargetable pump (a follow-up / stack
2). The STATE GRAPH, the breach-abort, the occupancy stall and ``busy`` are all
complete and deterministic here; the pressure feel is Erik's B5 HUMAN-TEST.
"""
from __future__ import annotations

from simulation.entities.schema import (
    Entity, Field, INPUT_SINGLE, InputDecl, KIND_ENUM, KIND_Q16, Signal,
    register,
)

# ---------------------------------------------------------------------------
# THE state enum (§7) — integer, synced. Shared by the schema's
# runtime_digest_rows and the sim-side runtime so both name one set of ints.
# ---------------------------------------------------------------------------
AIRLOCK_IDLE = 0
AIRLOCK_CLOSING = 1
AIRLOCK_EQUALIZE = 2
AIRLOCK_OPEN_FAR = 3
AIRLOCK_RESEAL = 4
AIRLOCK_REPRESSURIZE = 5
AIRLOCK_FAULT = 6

# Human-readable names (tests / debugging / the editor HUD) — order == value.
AIRLOCK_STATE_NAMES = (
    "IDLE", "CLOSING", "EQUALIZE", "OPEN_FAR", "RESEAL", "REPRESSURIZE",
    "FAULT",
)


@register
class airlock_controller(Entity):
    """The automatic airlock state machine (§7, B5 — HUMAN-TEST).

    A pure logic node (no tile — ``INTANGIBLE``). Wired to a chamber presence
    plate, an inner + outer door (drives ``close``/``open``, reads each door's
    ``is_open`` AND ``alive`` — D12), and a pump (drives ``inject``/``extract``,
    reads ``at_target``). Emits the door / pump command signals + ``busy`` (1
    while cycling) + the free ``alive``. All I/O is SignalBus-only.

    ``far_door`` names which physical door is the FAR side (the one the cycle
    opens after equalizing); the other is the NEAR side (reopened at IDLE).
    ``target_far_atm`` / ``target_near_atm`` pick the pump drive direction per
    phase (see the module docstring's single-pump note).
    """

    INTANGIBLE = True    # pure logic, no tile (like a decider — §5/§7)
    LOGIC_NODE = True     # its command signals are swapped node outputs (§2c)

    FIELDS = (
        Field("far_door", KIND_ENUM, default="outer",
              choices=("inner", "outer"),
              doc="which wired door is the FAR side (opened after EQUALIZE); "
                  "the other is NEAR (reopened at IDLE)"),
        Field("target_far_atm", KIND_Q16, default=0,
              doc="far-side pressure target (Q16.16 atm; 65536 == 1.0). "
                  "EQUALIZE drives the pump toward it — selects inject vs "
                  "extract direction (see the single-pump note)."),
        Field("target_near_atm", KIND_Q16, default=65536,
              doc="near-side (cabin) pressure target (Q16.16 atm). "
                  "REPRESSURIZE drives the pump toward it."),
    )
    # The command signals (prev-read/next-write node outputs, §2c): one pair
    # per door + one pair for the pump + busy. Wired to the actuators' inputs.
    SIGNALS = (
        Signal("inner_close", "hold the INNER door closed (→ inner_door.close)"),
        Signal("inner_open_cmd", "hold the INNER door open (→ inner_door.open)"),
        Signal("outer_close", "hold the OUTER door closed (→ outer_door.close)"),
        Signal("outer_open_cmd", "hold the OUTER door open (→ outer_door.open)"),
        Signal("pump_inject", "hold the pump injecting (→ pump.inject)"),
        Signal("pump_extract", "hold the pump extracting (→ pump.extract)"),
        Signal("busy", "1 while a cycle is in progress (not IDLE / FAULT)"),
    )
    # All inputs are SINGLE — one wire each (the §1b arity check enforces it).
    INPUTS = (
        InputDecl("presence", INPUT_SINGLE,
                  "chamber occupancy count (← presence plate .value)"),
        InputDecl("inner_open", INPUT_SINGLE,
                  "the INNER door's is_open (← inner_door.is_open)"),
        InputDecl("outer_open", INPUT_SINGLE,
                  "the OUTER door's is_open (← outer_door.is_open)"),
        InputDecl("inner_alive", INPUT_SINGLE,
                  "the INNER door's alive (← inner_door.alive) — D12 breach"),
        InputDecl("outer_alive", INPUT_SINGLE,
                  "the OUTER door's alive (← outer_door.alive) — D12 breach"),
        InputDecl("at_target", INPUT_SINGLE,
                  "the pump's at_target (← pump.at_target)"),
    )

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """§7/§8: the state enum + dwell counter are synced runtime rows (they
        cannot be re-derived from the signals alone — the machine is path
        dependent). Read off the
        :class:`simulation.logic_nodes.AirlockControllerRuntime` wrapper. A bare
        EntityInstance has no ``state`` and raises AttributeError — loud, like
        the door / filter (digests only come from constructed sims). ABSENT
        (zero bytes) when no controller is instantiated (dormancy, §8)."""
        return (("state", int(entity.state)), ("dwell", int(entity.dwell)))
