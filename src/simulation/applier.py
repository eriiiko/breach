"""THE PER-ENV SERIAL APPLIER's one table (#29 P6.4b; residency design §4.3,
"The per-env serial applier", and §4.4b's P6.4 row).

The applier is THE device form of every order-dependent law that writes the
map: ONE BLOCK per env (``blockIdx.y`` the env, the ``bgrid`` convention),
thread 0 walking that env's recorded EFFECTS in the reference's order, each
effect a DEVICE FUNCTION (``cpp/src/cuda_topology.cuh``: the topology seam --
``on_tile_changed``, ``destroy_wall``) whose book writes land in per-env BOOK
channels. This module is its ONE Python table, with generated offsets
(``cpp/src/applier_layout.h``, written by ``tools/gen_unit_layout.py --write``
beside the unit store's, the tables' and the event ring's; fingerprint-checked
by :class:`DeviceApplier`):

- :data:`FIELDS` / :data:`KINDS` -- an effect RECORD (int64 ``kind, y, x,
  arg``) and its kinds. Today ``DESTROY_WALL`` (slot 9's burn-through; P7's
  chew and blasts and P6b's door deaths append the same kind). A per-env list
  ``(N, cap, F)`` with a count per env: an effect past ``cap`` is never
  applied, the env's ``dropped`` book rises and the Python face refuses the
  run naming the kernel and the env (:class:`EffectOverflow`); the first
  ``cap`` are applied.
- :data:`BOOKS` -- the per-env book channels the effects write, each with the
  HOST quantity it twins (``gmap.<attr>`` or ``gmap.<dict>:<key>``), so the
  shadow harness compares a book with the host's delta over the slot and the
  chained harness lands it on the host.
- :data:`PLANES` -- the planes the topology seam reads and writes, by
  ``GameMap`` field name, each in its own dtype and per-cell width: what the
  applier's launch takes as device pointers to ``(N, ...)`` arrays. Most are
  resident fields (``GameMap._RESIDENT_FIELD_NAMES``: the arrays
  ``ResidentBatch.buffers`` holds); :func:`non_resident_planes` names the rest
  (P8b must hold them on the device).
- :data:`CLEAR_PLANES` -- the per-tick planes the tick-end clears zero.

A NEW EFFECT KIND, BOOK OR PLANE is a row here, ``tools/gen_unit_layout.py
--write``, commit, rebuild ``cpp/build_cuda``.

The fan-out of an effect's tile set over the block's threads with a barrier
between effects (§4.3: a blast's deposits, a reach walk's frontier) arrives
with its first user (P7c's deposits); the launch already has the block shape
so it can be added without moving the kernel.

NOT PART OF THE PLAYED TICK: Class T (training only), like the unit store.
Import-light: numpy; ``breach_physics`` only inside :class:`DeviceApplier`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

#: The generated header (C++'s copy of this table).
HEADER_REL = "cpp/src/applier_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"

#: #29 P7a.1 -- THE EFFECT LIST (residency design 4.4c, decision 1): the
#: record is the kind, then ARG_SLOTS int64 whose meaning is the kind's own
#: (:data:`KIND_FIELDS`); C++ reads a kind's field as ``APPLIER_<KIND>_<FIELD>``
#: (its column), never a bare slot number.
ARG_SLOTS = 10
#: An effect record's fields, in order.
FIELDS = ("kind",) + tuple(f"a{i}" for i in range(ARG_SLOTS))
F = len(FIELDS)
#: The effect kinds (code = position + 1; 0 = no record) and each kind's
#: fields, in the order they occupy the arg slots. Every value is what the
#: reference's call receives at that point, already in its integer currency:
#:
#: - DESTROY_WALL ``GameMap.destroy_wall(y, x)`` (slot 9's burn-through; P6b's
#:   door deaths);
#: - HIT ``damage.apply_packet(units[row], packet, sim.tick_events,
#:   source=cause)``: the packet at its door (``DamagePacket.q()``, the AP
#:   quantized as ``mitigate_q`` quantizes it, the attacker normalized, the
#:   cause's index in ``events.CAUSE_NAMES``); in a stage that LANDS
#:   (:data:`STAGES`) with the slot's exposure landing;
#: - STATUS ``status.apply_status`` if the unit is alive at its turn (the
#:   magnitude quantized as ``apply_status`` quantizes it; source -1 = None);
#: - PUSH ``unit_impulse.add(sim.impulse_table, units[row], ...)`` if alive;
#: - CHEW ``combat.chew_wall(gmap, y, x, wall_damage)``: ``live`` 0 when
#:   ``wall_damage <= 0`` (a no-op), ``damage_q`` the per-call
#:   ``wall_fixed.quantize_scalar(float(wall_damage))``;
#: - PAYLOAD ``payloads.execute_payload(gmap, queue, units, y, x, payload,
#:   rng, events=sim.tick_events, kind=..., owner_id=owner, approach=...)``
#:   (#29 P7c.1: applied by ``cpp/src/cuda_unit_payload.cuh`` for a row of
#:   the disc path; a refused row, :data:`PAYLOAD_REFUSALS`, by name;
#:   ``payload`` the row's index in
#:   ``sim.weapons_tables.payloads``, ``kind`` its index in
#:   :data:`PAYLOAD_KINDS`, ``has_approach`` 0 for ``approach=None``).
#:
#: - SPRAY_CELLS (#29 P7a.3) ``combat._enqueue_spray_cells(gmap, queue,
#:   owner, heat_q, gas_id, gas_q, cells)``: the call's scalars as given
#:   (``gas_id`` -1 for ``None``) and the cells' SPAN in the env's CELL BUFFER
#:   (``simulation.attack_lists``: rows ``(y, x, div)``, filled by slot 4b's
#:   phase A against the walls standing at the slot's start; the walk never
#:   recomputes a cone).
#:
#: There is no FACE kind: its only producers are slot 4's WEGO branches
#: (design 4.4c scope); a host FACE effect is refused by name.
KIND_FIELDS: tuple = (
    ("DESTROY_WALL", ("y", "x")),
    ("HIT", ("row", "amount_q", "dtype", "attacker", "indirect", "expose",
             "dir_x_q", "dir_y_q", "ap_q", "cause")),
    ("STATUS", ("row", "status", "magnitude_q", "duration", "source")),
    ("PUSH", ("row", "jx_q", "jy_q", "lever_q", "twist_q16")),
    ("CHEW", ("y", "x", "damage_q", "live")),
    ("PAYLOAD", ("y", "x", "payload", "owner", "approach_y", "approach_x",
                 "has_approach", "kind")),
    ("SPRAY_CELLS", ("owner", "heat_q", "gas_id", "gas_q", "start", "count")),
)
KINDS = tuple(k for k, _f in KIND_FIELDS)
KIND_DESTROY_WALL, KIND_HIT, KIND_STATUS, KIND_PUSH, KIND_CHEW, \
    KIND_PAYLOAD, KIND_SPRAY_CELLS = range(1, 8)
#: The kinds the applier refuses to APPLY (counted in the ``refused`` book,
#: the run refused by name): a record with no device function yet. #29
#: P7c.1: none -- PAYLOAD has its function (``cpp/src/cuda_unit_payload.cuh``);
#: what it still refuses is a payload ROW (:data:`PAYLOAD_REFUSALS`).
UNAPPLIED_KINDS: tuple = ()
#: #29 P7c.1: the payload rows the PAYLOAD function refuses, by code (the
#: Device tables row ``payload_refuse``, ``unit_tables``): each counted in
#: the ``refused`` book, nothing written, the run refused by name. Code 0 is
#: a row the function applies. #29 P7c.2: EMPTY -- the strength path (the
#: reach walk, the air read, start_source, deposit_over_tiles), bulk gas, the
#: explosive rows' air read and smoke, and the multi-tick source
#: (start_expansion, the blast-source table) all have their device form, so
#: every row is applied; the mechanism stays for a future row the function
#: cannot serve (a code here, its test in :func:`payload_refusal`).
PAYLOAD_REFUSALS: tuple = ()


def payload_refusal(payload) -> int:
    """The refusal code of a ``[payloads.*]`` row (:data:`PAYLOAD_REFUSALS`;
    0 = the PAYLOAD function applies it). #29 P7c.2: every row is applied."""
    return 0


def payload_refusal_reason(code: int) -> str:
    for c, name, why in PAYLOAD_REFUSALS:
        if c == int(code):
            return f"{name}: {why}"
    return f"code {int(code)}"
#: ``execute_payload``'s ``kind`` values (its render-only ExplosionEvent
#: label), as a PAYLOAD record's ``kind`` index; another is refused by name.
PAYLOAD_KINDS = ("explosion", "shell", "grenade", "door_explosive",
                 "timed_charge", "heat_source")
#: The STAGES of the one applier kernel (decision 1: one walker, several
#: stages): (name, lands) -- ``lands`` True when a HIT in that stage holds
#: its exposure raises and zeroings for the walk's end (decision 2: the mode
#: is per (stage, kind); every other packet lands at once).
#:   TOPOLOGY  slot 9's burn-through list (DESTROY_WALL only today);
#:   RESOLVE   slot 4c, ``SimultaneousEffects.resolve`` of the attack slot;
#:             #29 P7a.2: also slot 2''s landing (``Simulation._update_rounds``'
#:             own resolve: the same law, the same landing, so the same stage).
#:   VENTS     #29 P6b.2: slot 9e(d)'s vent circulation
#:             (``vent_system.sweep_vents``) -- the stage walks the duct and
#:             vent TABLES (``unit_schema.DUCT_TABLE`` / ``VENT_TABLE``), not
#:             the effect list: ducts in row (= ordinal) order, within a duct
#:             its returns then its supplies, each applied at once (two vents
#:             may share a tile). It runs BEFORE the doors inside 9e(d) (pumps
#:             -> vents -> doors), so its number is below DOORS'.
#:   DOORS     #29 P6b.1: slot 9e(d)'s door sweep -- the stage walks the
#:             door TABLE (``unit_schema.DOOR_TABLE`` and its tile side
#:             table), not the effect list, in row (= ordinal) order, each door
#:             applied at once (a door's seal changes the gas the next door
#:             reads); the effect list is not read.
#:   DETONATE  #29 P7c.1: the PAYLOAD records slot 2's grenade fuse-outs and
#:             slot 9e(d')'s charges append (``unit_kernels.grenades`` /
#:             ``charges_tick``), walked right after their kernel, where the
#:             reference calls ``execute_payload`` at once; no HIT is ever on
#:             its list, and a blast's packets land at once in every stage.
#:   SOURCES   #29 P7c.2: slot 7's head, ``blast_expansion.take_source_plane``
#:             -- the pending ``div_source`` plane handed out (the stage's
#:             SOURCE OUT plane, ``source_taken`` 1 when it held a source) and
#:             zeroed, then the blast-source TABLE (``unit_schema.SOURCE_TABLE``)
#:             walked in row order, each source added at once (saturating,
#:             counted), its ticks counted down, the dead dropped by the span
#:             compaction; the effect list is not read.
#:   EMIT / LOGIC / DOOR_INPUTS / PUMPS / SWAP  #29 P6b.3: slot 9e's signal
#:             sub-steps (a), (b), (c), (d) pumps and (e), each walking the
#:             signal tables (``unit_schema.SIGNAL_TABLE``, ``SENSOR_TABLE``,
#:             ``NODE_TABLE``, ``PUMP_TABLE``, ``DOOR_DRIVE_TABLE`` and their
#:             side tables), not the effect list: (a) the bus's emits in slot
#:             order, then the sensors in ordinal order; (b) the nodes in
#:             ordinal order, pub -> stg; (c) the wired doors' want_open in
#:             door order; (d) the pumps in ordinal order, BEFORE the vents,
#:             each applied at once over the seam; (e) pub <- stg on the node
#:             slots, after every 9e(d) actuator. Numbered in the conductor's
#:             9e order (P6b.3 moved VENTS 2 -> 6, DOORS 3 -> 7, DETONATE
#:             4 -> 8, SOURCES 5 -> 10).
#:   TIMED_CHARGES / HEAT_SOURCES  #29 P6b.4: slot 9e(d') and 9e(d''), each
#:             walking its entity TABLE (``unit_schema.TIMED_CHARGE_TABLE`` /
#:             ``HEAT_SOURCE_TABLE``) in row (= ordinal) order and applying
#:             every detonation AT ONCE through the one PAYLOAD function, as
#:             the reference's sweeps call ``execute_payload`` at once (a
#:             timed charge's heat trigger reads the tile an earlier charge's
#:             blast may have destroyed; a heat source's death payload appends
#:             the disc the chain then reads). Each applied PAYLOAD is also
#:             appended to the stage's own effect list, a record of what it
#:             applied (never walked again). The conductor's order: timed
#:             charges (8) after the doors, the planted charges' DETONATE (9),
#:             then the heat sources (10), which read the whole tick's blast
#:             log and empty it; P6b.4 moved DETONATE 8 -> 9, SWAP 9 -> 11,
#:             SOURCES 10 -> 12.
STAGES: tuple = (("TOPOLOGY", False), ("RESOLVE", True), ("EMIT", False),
                 ("LOGIC", False), ("DOOR_INPUTS", False), ("PUMPS", False),
                 ("VENTS", False), ("DOORS", False), ("TIMED_CHARGES", False),
                 ("DETONATE", False), ("HEAT_SOURCES", False),
                 ("SWAP", False), ("SOURCES", False))
STAGE_NAMES = tuple(s for s, _l in STAGES)
(STAGE_TOPOLOGY, STAGE_RESOLVE, STAGE_EMIT, STAGE_LOGIC, STAGE_DOOR_INPUTS, STAGE_PUMPS,
 STAGE_VENTS, STAGE_DOORS, STAGE_TIMED_CHARGES, STAGE_DETONATE, STAGE_HEAT_SOURCES,
 STAGE_SWAP, STAGE_SOURCES) = range(13)
assert STAGE_NAMES[STAGE_SWAP] == "SWAP" and STAGE_NAMES[STAGE_SOURCES] == "SOURCES"
assert STAGE_NAMES[STAGE_TIMED_CHARGES] == "TIMED_CHARGES" \
    and STAGE_NAMES[STAGE_HEAT_SOURCES] == "HEAT_SOURCES"
#: The most effects ONE ROW of a per-row producer stages in one launch: slot
#: 4a's blow is a HIT, a STATUS and a PUSH; #29 P7a.2: slot 4's trigger is up
#: to four records per round (its recoil PUSH, then the march's HIT, PUSH and
#: a payload round's PAYLOAD, or a CHEW and a PAYLOAD) times the burst's
#: rounds (a shotgun's eight). A later producer that stages more raises it (a
#: row here, regenerate); past it the run is refused (EffectOverflow).
ROW_CAP = 32
#: The default list capacity per env per launch.
E_CAP = 256


assert all(len(fs) <= ARG_SLOTS for _k, fs in KIND_FIELDS),     "applier: a kind has more fields than the record's arg slots"


def kind_field(kind: str, name: str) -> int:
    """The record column of ``kind``'s field ``name``."""
    for k, fs in KIND_FIELDS:
        if k == kind:
            return 1 + fs.index(name)
    raise KeyError(f"applier: no kind {kind!r}")


def make_record(kind: str, /, **values) -> tuple:
    """One record of ``kind`` from its named fields (every field required)."""
    fs = dict(KIND_FIELDS)[kind]
    if set(values) != set(fs):
        raise ValueError(f"applier: a {kind} record has the fields {fs}, not "
                         f"{tuple(sorted(values))}")
    rec = [0] * F
    rec[0] = KINDS.index(kind) + 1
    for i, n in enumerate(fs):
        rec[1 + i] = int(values[n])
    return tuple(rec)


@dataclass(frozen=True)
class BookRow:
    """One per-env book channel: ``host`` names the host quantity it twins
    (``attr`` or ``attr:key`` on the GameMap; None = the applier's own
    count, with no host attribute)."""
    name: str
    host: str | None
    note: str = ""


BOOKS: tuple = (
    BookRow("n_destruction_seed", "n_destruction_seed_sum",
            "destroy_wall's measured bulk delta (design §5.10: the destroyed "
            "tile's own bulk delta)"),
    BookRow("ge_retire", "gas_energy_books:retire",
            "the gas energy seam's 'retire' channel (on_tile_changed's "
            "membership retire; a breach's evacuation)"),
    BookRow("ge_mint", "gas_energy_books:mint",
            "the 'mint' channel (on_tile_changed's born-at-ambient)"),
    BookRow("ge_destroy_seed", "gas_energy_books:destroy_seed",
            "destroy_wall's seed: the retire, then the one true mint"),
    BookRow("solid_destroy_wall", "solid_energy_books:destroy_wall",
            "the solid books' destroy_wall channel (its _solid_books_sink "
            "is the host's: the chained harness books it there)"),
    BookRow("tiles_changed", "_tile_serial",
            "on_tile_changed calls (the host's _tile_serial bump)"),
    BookRow("applied", None, "effects that acted (a destroy of a non-air "
            "tile in bounds)"),
    BookRow("dropped", None, "effects past the list's capacity (refused)"),
    BookRow("faults", None, "effects whose operands left the proved range "
            "(nothing written; refused)"),
    BookRow("refused", None, "#29 P7a.1: records of a kind with no device "
            "function yet (UNAPPLIED_KINDS); #29 P7c.1: a PAYLOAD record of a "
            "refused payload row (PAYLOAD_REFUSALS), never applied; the run "
            "is refused by name"),
)


def _n_gases() -> int:
    from simulation.gases import N_GASES
    return int(N_GASES)


#: #29 P6b.1: the books of seal / unseal (the door sweep) and of the gas-N
#: primitives (the pumps' and vents' from P6b.2 / P6b.3): each the seam
#: channel it twins, a diagnostic channel the same (``diag_*``: energy that
#: crossed the books' boundary, outside ``gas_energy_seam_net``), and the
#: counted trace saturation per gas slice (``trace_clip_books[g]``).
BOOKS = BOOKS + (
    BookRow("ge_seal", "gas_energy_books:seal",
            "seal_tiles' moves (the evacuated bulk carrying the tile's T_abs)"),
    BookRow("ge_diag_seal_from", "gas_energy_books:diag_seal_from_nonacct",
            "a seal move from a non-accountable tile (a sealed breach)"),
    BookRow("ge_diag_seal_to", "gas_energy_books:diag_seal_to_nonacct",
            "a seal move to a non-accountable receiver (vacuum, ambient)"),
    BookRow("ge_unseal", "gas_energy_books:unseal",
            "unseal_tiles' withdrawals (each donor's share at its own T_abs)"),
    BookRow("ge_diag_unseal_from", "gas_energy_books:diag_unseal_from_nonacct",
            "an unseal withdrawal from a non-accountable donor (an ambient ring)"),
    BookRow("ge_diag_unseal_to", "gas_energy_books:diag_unseal_to_nonacct",
            "an unseal withdrawal into a non-accountable tile"),
    BookRow("solid_seal", "solid_energy_books:seal",
            "the solid books' seal channel (a sealed thermal solid born)"),
    BookRow("solid_unseal", "solid_energy_books:unseal",
            "the solid books' unseal channel (an opened thermal solid retired)"),
    BookRow("ge_pump_inject", "gas_energy_books:pump_inject",
            "inject_gas_n's mint (born at ambient)"),
    BookRow("ge_pump_extract", "gas_energy_books:pump_extract",
            "extract_gas_n(_vec)'s withdrawal at the cell's T_abs"),
    BookRow("ge_pump_transfer", "gas_energy_books:pump_transfer",
            "transfer_gas_n's move at the donor's T_abs"),
    BookRow("ge_diag_pump_transfer_from",
            "gas_energy_books:diag_pump_transfer_from_nonacct",
            "a transfer from a non-accountable tile"),
    BookRow("ge_diag_pump_transfer_to",
            "gas_energy_books:diag_pump_transfer_to_nonacct",
            "a transfer to a non-accountable tile"),
    # #29 P6b.2: the vents' deposit (inject_gas_n_vec) and the sweep's
    # named ledger refusal.
    BookRow("ge_pump_inject_vec", "gas_energy_books:pump_inject_vec",
            "inject_gas_n_vec's deposit at the plenum's t_dep (+ T_AMB)"),
    BookRow("ge_pump_rail", "gas_energy_books:pump_rail",
            "inject_gas_n_vec's rail clamp (a counted destruction)"),
    BookRow("ledger_refused", None,
            "ducts refused at their sweep's head: a ledger outside the proved "
            "range (nothing of the duct written; LedgerRangeRefused)"),
    # #29 P7c.2: the strength path's divergence source (add_source's counted
    # saturation), bulk gas (inject_bulk_gas' seam channels and its named
    # overflow refusal) and the SOURCES stage's hand-out.
    BookRow("div_source_sat_hits", "div_source_sat_hits",
            "add_source's saturations at S_MAX_Q (start_source, start_expansion, "
            "take_source_plane)"),
    BookRow("div_source_cut_q", "div_source_cut_q",
            "add_source's cut Q16 above S_MAX_Q"),
    BookRow("ge_payload_bulk_gas", "gas_energy_books:payload_bulk_gas",
            "inject_bulk_gas' deposits (inject_gas_n_vec at bulk_temp + T_AMB)"),
    BookRow("ge_payload_bulk_gas_rail", "gas_energy_books:payload_bulk_gas_rail",
            "inject_bulk_gas' rail clamps"),
    BookRow("bulk_refused", None,
            "bulk-gas deposits refused: a tile whose bulk slice would pass int32 "
            "(the reference raises OverflowError; the device writes none of the "
            "deposit and the run is refused by name, BulkGasOverflow)"),
    BookRow("source_taken", None,
            "the SOURCES stage: 1 when the pending plane held a source "
            "(take_source_plane returned a plane, not None)"),
    # #29 P6b.3: the signal stages' named refusal and what they wrote
    BookRow("signal_refused", None,
            "LOGIC stages refused at their head: a filter node's ema, input or k "
            "outside the proved range (nothing of the env's nodes written; "
            "SignalRangeRefused)"),
    BookRow("signal_writes", None,
            "the signal stages' writes that changed a value (a pub / stg slot, a "
            "want_open, a node's ema / state / dwell, a pump's at_target)"),
    # #29 P6b.4: the timed charges' and heat sources' stages
    BookRow("timed_fired", None,
            "the TIMED_CHARGES stage: charges that detonated (a PAYLOAD applied)"),
    BookRow("heat_deaths", None,
            "the HEAT_SOURCES stage: sources a blast took to hp 0 this tick"),
    BookRow("heat_deposits", None,
            "the HEAT_SOURCES stage: per-tick deposits enqueued (deposit_heat)"),
    BookRow("heat_refused", None,
            "HEAT_SOURCES stages refused at their head: the chain's proved bound "
            "(the blast log's first discs plus the sources whose death payload "
            "appends a disc) past the blast log's capacity (nothing written; "
            "HeatChainRefused)"),
) + tuple(BookRow(f"trace_clip_{g}", f"trace_clip_books:{g}",
                  f"the counted trace saturation's clip on gas slice {g} "
                  f"(trace_store_sat / trace_add_sat)")
          for g in range(_n_gases()))
BOOK_NAMES = tuple(b.name for b in BOOKS)
N_BOOKS = len(BOOKS)

#: The topology seam's planes: (GameMap field, numpy dtype, per-cell width;
#: "G" = the gas planes' count, the leading axis). Float planes are read and
#: written as their exact 32-bit pattern (never a float on the device).
PLANES: tuple = (
    ("material", "int8", 1),
    ("wall_hp", "int32", 1),
    ("gas", "int32", "G"),
    ("gas_energy", "int64", 1),
    ("temperature", "int32", 1),
    ("fire", "int32", 1),
    ("atmosphere", "int32", 1),
    ("solid", "bool", 1),
    ("is_vacuum", "bool", 1),
    ("is_ambient", "bool", 1),
    ("thermal_solid", "bool", 1),
    ("flammable", "bool", 1),
    ("sight_screen", "bool", 1),
    ("heat_inv_shift", "int32", 1),
    ("face_shift", "int32", 4),
    ("conductivity", "float32", 1),
    ("permeability", "float32", 1),
    ("wave_absorb", "float32", 1),
    ("heat_atten", "float32", 1),
    ("heat_atten_q", "int32", 1),
    ("dyn_heat_atten_q", "int32", 1),
    ("light_atten", "float32", 3),
    ("light_atten_q", "int32", 3),
    ("dyn_light_atten_q", "int32", 3),
    ("fuel_recip", "int64", 1),
    ("fire_T_ext_plane", "int32", 1),
    ("fuel_per_o2_plane", "int32", 1),
    # #29 P6b.1: seal_tiles' solver-owned planes (set to their solid steady
    # state on a sealed tile; unseal writes wave_p), the float ripple pair
    # (written as the bit pattern of 0.0), and the water depth (the seal's
    # water rule and _aperture_blocked read it; nothing here writes it).
    ("wave_p", "int32", 1),
    ("wind_x", "int32", 1),
    ("wind_y", "int32", 1),
    ("flow_vx", "int32", 1),
    ("flow_vy", "int32", 1),
    ("ripple", "float32", 1),
    ("ripple_v", "float32", 1),
    ("water_depth", "int32", 1),
    # #29 P7c.2: the blast's pending divergence source (add_source's plane,
    # Q16 1/s), written by the strength path and the SOURCES stage.
    ("div_source", "int32", 1),
)
PLANE_NAMES = tuple(p[0] for p in PLANES)
#: #29 P6b.1: the topology seam's REFUSALS (seal_tiles / unseal_tiles /
#: can_seal_tiles and the gas-N primitives): every exception the Python
#: raises before it mutates, as a code (0 = done), in ``_seal_blockers``'
#: order; the device function writes NOTHING when it refuses. ``fault`` is an
#: operand past the range the device was proved for (Python's ints would hold
#: it); ``span_too_long`` a span past the device's bound (no Python twin).
SEAM_REFUSALS: tuple = (
    ("out_of_bounds", "ValueError: a tile outside the grid"),
    ("already_solid", "ValueError: seal_tiles over a solid tile"),
    ("unknown_material", "ValueError: a material id outside the table"),
    ("material_not_solid", "ValueError: a material whose permeability > 0.0"),
    ("standing_water", "SealBlocked: a span tile holds standing water"),
    ("sealed_pocket", "SealBlocked: a gas-holding tile with no receiver"),
    ("overflow", "OverflowError: a bulk store past int32"),
    ("duplicate_tile", "ValueError: a span names a tile twice"),
    ("not_solid", "ValueError: unseal_tiles over an open tile"),
    ("span_too_long", "a span longer than the device's bound (DOOR_TILE_CAPACITY)"),
    ("fault", "an operand past the device's proved range"),
)
SEAM_REFUSAL_NAMES = tuple(n for n, _w in SEAM_REFUSALS)
N_PLANES = len(PLANES)

#: The tick-end clears' planes (Simulation._clear_tick_buffers, with a
#: physics runner): (GameMap field, dtype). The credit claims' clear is P7's
#: (its device form is the claims table); the shot tracers' expiry is render.
CLEAR_PLANES: tuple = (
    ("heat", "int32"),
    ("rad_net", "int64"),
    ("rad_amb", "int64"),
    ("rad_flux", "int64"),
)
CLEAR_NAMES = tuple(p[0] for p in CLEAR_PLANES)


class EffectOverflow(RuntimeError):
    """An env's effects did not fit the applier's list (its capacity)."""


class ApplierFault(RuntimeError):
    """An effect's operands left the range the applier was proved for."""


class LedgerRangeRefused(RuntimeError):
    """#29 P6b.2: a duct ledger (the plenum's bulk pair, trace, sinks, energy,
    wipe, rail hits, a vent's guard count) outside the range the vent sweep
    was proved for (``cpp/src/cuda_unit_vents.cuh``, design 4.4c "As built
    (P6b.2)"): the duct is refused BY NAME at the head of its sweep and
    nothing of it is written (its vents, its ledgers, the map)."""


class BulkGasOverflow(RuntimeError):
    """#29 P7c.2: a bulk-gas deposit (``payloads.inject_bulk_gas``) whose slice
    would pass int32 on a tile: the reference raises OverflowError mid-loop;
    the device writes none of the deposit and the run is refused by name."""


class SignalRangeRefused(RuntimeError):
    """#29 P6b.3: a filter node's ``ema`` (an unbounded Python int), its input
    or its ``k`` outside the range the LOGIC stage was proved for
    (``cpp/src/cuda_unit_signals.cuh``: |ema|, |in| <= 2^60, 0 <= k <= 62):
    the env's LOGIC stage is refused BY NAME at its head and nothing of it is
    written."""


class HeatChainRefused(RuntimeError):
    """#29 P6b.4: a heat-source sweep whose chain could outgrow the blast log:
    its proved bound -- the discs the log holds at the sweep's head plus the
    living sources whose death payload appends a disc (each source dies at
    most once, and a death appends at most one disc) -- past the log's
    capacity. The env's HEAT_SOURCES stage is refused BY NAME at its head and
    nothing of it is written."""


class EffectRefused(RuntimeError):
    """#29 P7a.1: an effect of a kind the applier cannot apply yet (a PAYLOAD
    record until P7c.1), or a host effect with no device kind (FACE, a cover
    chew, a spray's cells, any other world call)."""


def _signal_codes() -> tuple:
    """#29 P6b.3: the code groups the signal stages read, as (group, names):
    each name's index is its code (``unit_schema``'s encoders index the same
    tuples)."""
    from simulation import unit_schema as us
    from simulation.entities.actuators import AIRLOCK_STATE_NAMES
    return (("EMIT", tuple(us.SIGNAL_EMITS)), ("SENSOR", tuple(us.SENSOR_KINDS)),
            ("CHANNEL", us.sensor_channels()), ("CMP", us.comparators()),
            ("NODE", tuple(us.NODE_KINDS)),
            ("AIRLOCK", tuple(str(n) for n in AIRLOCK_STATE_NAMES)))


def non_resident_planes() -> tuple:
    """The planes the topology seam writes that are NOT resident fields today
    (P8b must hold them on the device)."""
    from simulation.gamemap import GameMap
    res = set(GameMap._RESIDENT_FIELD_NAMES)
    return tuple(n for n in PLANE_NAMES + CLEAR_NAMES if n not in res)


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    fields: tuple
    kinds: tuple
    books: tuple
    planes: tuple
    clears: tuple
    fingerprint: str
    kind_fields: tuple = ()
    stages: tuple = ()
    row_cap: int = ROW_CAP

    @property
    def width(self) -> int:
        return len(self.fields)


_LAYOUT_CACHE: dict = {}


def layout() -> Layout:
    key = (FIELDS, KINDS, BOOK_NAMES, PLANES, CLEAR_PLANES, STAGES,
           SEAM_REFUSAL_NAMES, _signal_codes())
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    text = "".join(f"field:{i}:{n}\n" for i, n in enumerate(FIELDS))
    text += "".join(f"kind:{i + 1}:{n}\n" for i, n in enumerate(KINDS))
    text += "".join(f"book:{i}:{n}\n" for i, n in enumerate(BOOK_NAMES))
    text += "".join(f"plane:{i}:{n}:{d}:{c}\n"
                    for i, (n, d, c) in enumerate(PLANES))
    text += "".join(f"clear:{i}:{n}:{d}\n"
                    for i, (n, d) in enumerate(CLEAR_PLANES))
    text += "".join(f"stage:{i}:{n}:{int(lands)}\n"
                    for i, (n, lands) in enumerate(STAGES))
    text += "".join(f"refusal:{i + 1}:{n}\n"
                    for i, n in enumerate(SEAM_REFUSAL_NAMES))
    text += "".join(f"signal:{grp}:{i}:{n}\n" for grp, names in _signal_codes()
                    for i, n in enumerate(names))
    lay = Layout(fields=FIELDS, kinds=KINDS, books=BOOK_NAMES, planes=PLANES,
                 clears=CLEAR_PLANES,
                 fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest(),
                 kind_fields=KIND_FIELDS, stages=STAGES, row_cap=ROW_CAP)
    _LAYOUT_CACHE[key] = lay
    return lay


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/applier_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/applier.py",
        "// (#29 P6.4b / P7a.1, residency design 4.3 / 4.4b / 4.4c) -- NEVER EDIT",
        "// BY HAND. The per-env serial applier: an effect record of APPLIER_F",
        "// int64 (the kind, then the kind's own fields: APPLIER_<KIND>_<FIELD> is",
        "// a field's column), its kinds, its stages; the per-env book channels;",
        "// the topology seam's planes",
        "// (the order of the launch's pointer table) and the tick-end clears'",
        "// planes. A new kind / book / plane: a row of simulation/applier.py,",
        f"// then `{GENERATOR}` and commit; tests/test_applier_layout.py",
        "// refuses a stale copy.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace applier {",
        "",
        f'constexpr const char* APPLIER_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int APPLIER_F = {lay.width};",
        f"constexpr int APPLIER_ROW_CAP = {lay.row_cap};",
        "",
        "// An effect record's fields (the kind, then the arg slots).",
    ]
    for i, n in enumerate(lay.fields):
        lines.append(f"constexpr int APPLIER_FX_{n.upper()} = {i};")
    lines += ["", "// The effect kinds (0 = no record)."]
    for i, n in enumerate(lay.kinds):
        lines.append(f"constexpr int64_t APPLIER_KIND_{n.upper()} = {i + 1};")
    lines.append(f"constexpr int APPLIER_N_KINDS = {len(lay.kinds)};")
    lines += ["", "// Each kind's fields: their record column."]
    for n, fs in lay.kind_fields:
        for i, f in enumerate(fs):
            lines.append(f"constexpr int APPLIER_{n.upper()}_{f.upper()} = {1 + i};")
    lines += ["", "// The stages of the one walker; _LANDS: a HIT in the stage holds its",
              "// exposure raises and zeroings for the walk's end (the landing)."]
    for i, (n, l) in enumerate(lay.stages):
        lines.append(f"constexpr int APPLIER_STAGE_{n.upper()} = {i};")
        lines.append(f"constexpr bool APPLIER_STAGE_{n.upper()}_LANDS = "
                     f"{'true' if l else 'false'};")
    lines.append(f"constexpr int APPLIER_N_STAGES = {len(lay.stages)};")
    lines += ["", "// execute_payload's kind values (a PAYLOAD record's kind index)."]
    for i, n in enumerate(PAYLOAD_KINDS):
        lines.append(f"constexpr int64_t APPLIER_PAYLOAD_KIND_{n.upper()} = {i};")
    lines.append(f"constexpr int APPLIER_N_PAYLOAD_KINDS = {len(PAYLOAD_KINDS)};")
    lines += ["", "// The per-env book channels."]
    for i, n in enumerate(lay.books):
        lines.append(f"constexpr int APPLIER_BOOK_{n.upper()} = {i};")
    lines.append(f"constexpr int APPLIER_N_BOOKS = {len(lay.books)};")
    lines += ["", "// The topology seam's planes: the launch's pointer table order."]
    for i, (n, _d, _c) in enumerate(lay.planes):
        lines.append(f"constexpr int APPLIER_PLANE_{n.upper()} = {i};")
    lines.append(f"constexpr int APPLIER_N_PLANES = {len(lay.planes)};")
    lines += ["", "// The tick-end clears' planes (their element size in bytes)."]
    for i, (n, d) in enumerate(lay.clears):
        lines.append(f"constexpr int APPLIER_CLEAR_{n.upper()} = {i};")
        lines.append(f"constexpr int APPLIER_CLEAR_{n.upper()}_BYTES = "
                     f"{np.dtype(d).itemsize};")
    lines.append(f"constexpr int APPLIER_N_CLEARS = {len(lay.clears)};")
    lines += ["", "// #29 P6b.1: the topology seam's refusal codes (0 = done)."]
    for i, n in enumerate(SEAM_REFUSAL_NAMES):
        lines.append(f"constexpr int64_t APPLIER_SEAM_{n.upper()} = {i + 1};")
    lines.append(f"constexpr int APPLIER_N_SEAM_REFUSALS = {len(SEAM_REFUSAL_NAMES)};")
    lines += ["", "// #29 P6b.3: the signal tables' codes (unit_schema's SIGNAL_EMITS,",
              "// SENSOR_KINDS, sensor_channels(), comparators(), NODE_KINDS; the",
              "// airlock controller's states)."]
    for grp, names in _signal_codes():
        for i, n in enumerate(names):
            lines.append(f"constexpr int64_t SIGNAL_{grp}_{n.upper()} = {i};")
        lines.append(f"constexpr int SIGNAL_N_{grp} = {len(names)};")
    lines += ["", "}  // namespace applier", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Effect lists
# ---------------------------------------------------------------------------
def destroy_records(tiles) -> np.ndarray:
    """A list of ``(y, x)`` tiles as DESTROY_WALL records ``(R, F)``."""
    t = np.asarray(list(tiles), dtype=np.int64).reshape(-1, 2)
    out = np.zeros((t.shape[0], F), dtype=np.int64)
    out[:, 0] = KIND_DESTROY_WALL
    out[:, 1:3] = t
    return out


def pack(per_env, cap: int = E_CAP):
    """N envs' ``(R_e, F)`` records as the list the device walks: ``recs``
    ``(N, cap, F)`` (the first ``cap`` of each env) and ``counts`` ``(N,)``
    -- each env's REQUESTED count, which may exceed ``cap`` (the device
    applies ``cap`` and counts the rest in its ``dropped`` book)."""
    per_env = [np.asarray(r, dtype=np.int64).reshape(-1, F) for r in per_env]
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"applier: cap {cap} must be >= 1")
    recs = np.zeros((len(per_env), cap, F), dtype=np.int64)
    counts = np.zeros(len(per_env), dtype=np.int64)
    for e, r in enumerate(per_env):
        k = min(cap, r.shape[0])
        recs[e, :k] = r[:k]
        counts[e] = r.shape[0]
    return recs, counts


# ---------------------------------------------------------------------------
# #29 P7a.1: the host's effects as records (the effect list's ONE record
# function, decision 8: what the shadow harness compares and feeds)
# ---------------------------------------------------------------------------
def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _row_of(sim, unit) -> int:
    for r, u in enumerate(sim.units):
        if u is unit:
            return r
    raise EffectRefused(f"applier: an effect names a unit that is not in "
                        f"sim.units ({getattr(unit, 'id', None)!r})")


def payload_index(sim, payload) -> int:
    """A ``[payloads.*]`` row's index in ``sim.weapons_tables.payloads``."""
    names = list(sim.weapons_tables.payloads.by_name)
    name = getattr(payload, "name", None)
    if name not in names or sim.weapons_tables.payloads.by_name[name] is not payload:
        raise EffectRefused(f"applier: payload {name!r} is not a row of the "
                            f"sim's payload table")
    return names.index(name)


def record_of(sim, eff, cells=None) -> tuple:
    """ONE host effect of ``simulation.simultaneous.SimultaneousEffects``'s
    pending list as its device record (:data:`KIND_FIELDS`). A ``_WORLD``
    call is recognised by its function: ``combat.chew_wall`` is CHEW,
    ``payloads.execute_payload`` is PAYLOAD, ``combat._enqueue_spray_cells``
    is SPRAY_CELLS (#29 P7a.3: its cells appended to ``cells``, the CELL
    BUFFER's host list, the record holding their span; refused without it);
    FACE (WEGO-only), a cover chew and any other call are refused by name
    (:class:`EffectRefused`), as is an effect whose objects are not ``sim``'s
    own (another map, another impulse table, another event list)."""
    from simulation import combat, payloads, simultaneous, unit_fixed, wall_fixed
    from simulation.events import (CAUSE_NAMES, NO_ATTACKER, attacker_value,
                                   cause_value)
    kind = eff[0]
    if kind == simultaneous._HIT:
        _, unit, packet, events, source = eff
        if events is not sim.tick_events:
            raise EffectRefused("applier: a HIT whose events are not the sim's "
                                "tick_events has no device form")
        return make_record(
            "HIT", row=_row_of(sim, unit), amount_q=int(packet.q()),
            dtype=int(packet.dtype),
            attacker=attacker_value(getattr(packet, "attacker_id", NO_ATTACKER)),
            indirect=int(bool(getattr(packet, "indirect", False))),
            expose=int(getattr(packet, "expose", 0) or 0),
            dir_x_q=int(getattr(packet, "dir_x_q", 0)),
            dir_y_q=int(getattr(packet, "dir_y_q", 0)),
            ap_q=unit_fixed.quantize_scalar(float(packet.ap)),   # p3e-door: mitigate_q's
            cause=CAUSE_NAMES.index(cause_value(source)))
    if kind == simultaneous._STATUS:
        _, unit, skind, magnitude, duration, source_id = eff
        return make_record(
            "STATUS", row=_row_of(sim, unit), status=int(skind),
            magnitude_q=unit_fixed.quantize_scalar(float(magnitude)),  # p3e-door: apply_status's
            duration=int(duration),
            source=-1 if source_id is None else int(source_id))
    if kind == simultaneous._PUSH:
        _, table, unit, jx, jy, lever, twist = eff
        if table is not sim.impulse_table:
            raise EffectRefused("applier: a PUSH through another impulse table "
                                "than the sim's has no device form")
        return make_record("PUSH", row=_row_of(sim, unit), jx_q=jx, jy_q=jy,
                           lever_q=lever, twist_q16=twist)
    if kind == simultaneous._FACE:
        raise EffectRefused("applier: FACE has no device kind (its producers "
                            "are slot 4's WEGO branches, design 4.4c scope)")
    if kind != simultaneous._WORLD:
        raise EffectRefused(f"applier: no effect kind {kind!r}")
    _, fn, args, kwargs = eff
    f = _unwrap(fn)
    if f is _unwrap(combat.chew_wall):
        gmap, iy, ix, wall_damage = args
        if kwargs or gmap is not sim.gmap:
            raise EffectRefused("applier: a chew_wall on another map / with "
                                "keywords has no device form")
        live = wall_damage > 0
        return make_record(
            "CHEW", y=int(iy), x=int(ix),
            damage_q=(wall_fixed.quantize_scalar(float(wall_damage))  # p3e-door: chew_wall's
                      if live else 0),
            live=int(live))
    if f is _unwrap(payloads.execute_payload):
        gmap, queue, units, fy, fx, payload = args[:6]
        if gmap is not sim.gmap or queue is not sim.edit_queue or units is not sim.units \
                or kwargs.get("events", None) is not sim.tick_events:
            raise EffectRefused("applier: an execute_payload on another sim's "
                                "map / queue / units / events has no device form")
        pk = kwargs.get("kind", "explosion")
        if pk not in PAYLOAD_KINDS:
            raise EffectRefused(f"applier: payload kind {pk!r} is not one of "
                                f"{PAYLOAD_KINDS}")
        appr = kwargs.get("approach", None)
        return make_record(
            "PAYLOAD", y=int(fy), x=int(fx), payload=payload_index(sim, payload),
            owner=int(kwargs.get("owner_id", NO_ATTACKER)),
            approach_y=-1 if appr is None else int(appr[0]),
            approach_x=-1 if appr is None else int(appr[1]),
            has_approach=int(appr is not None), kind=PAYLOAD_KINDS.index(pk))
    if f is _unwrap(combat._enqueue_spray_cells):
        gmap, queue, owner, heat_q, gas_id, gas_q, cl = args
        if kwargs or gmap is not sim.gmap or queue is not sim.edit_queue:
            raise EffectRefused("applier: a spray's cells on another sim's map / "
                                "queue (or with keywords) have no device form")
        if cells is None:
            raise EffectRefused("applier: a spray's cells need the cell buffer "
                                "(records_and_cells_of)")
        start = len(cells)
        cells.extend((int(y), int(x), int(d)) for (y, x, d) in cl)
        return make_record(
            "SPRAY_CELLS", owner=int(owner), heat_q=int(heat_q),
            gas_id=-1 if gas_id is None else int(gas_id), gas_q=int(gas_q),
            start=start, count=len(cells) - start)
    raise EffectRefused(f"applier: the world call {getattr(f, '__qualname__', f)!r} "
                        f"has no device kind (CHEW is combat.chew_wall, PAYLOAD "
                        f"payloads.execute_payload, SPRAY_CELLS "
                        f"combat._enqueue_spray_cells)")


def records_and_cells_of(sim, pending, cell_base: int = 0):
    """#29 P7a.3: a host pending list as ``(R, F)`` int64 records in its
    order AND the cells its SPRAY_CELLS records point into (``(C, 3)`` int64
    rows ``(y, x, div)``), the spans counted from ``cell_base`` (the cells
    the env's buffer already holds)."""
    cells: list = [None] * int(cell_base)
    recs = [record_of(sim, e, cells) for e in pending]
    out = np.array(cells[int(cell_base):], dtype=np.int64).reshape(-1, 3)
    return np.array(recs, dtype=np.int64).reshape(len(recs), F), out


def records_of(sim, pending) -> np.ndarray:
    """A host pending list as ``(R, F)`` int64 records, in its order (a
    SPRAY_CELLS record's span counts from the list's first spray:
    :func:`records_and_cells_of` gives the cells)."""
    return records_and_cells_of(sim, pending)[0]


def effects_of(args, kwargs):
    """The ``SimultaneousEffects`` a slot callable was handed: ``self`` of
    ``resolve`` or the ``effects=`` of an attack slot's walk."""
    from simulation.simultaneous import SimultaneousEffects
    for a in tuple(args) + tuple(kwargs.values()):
        if isinstance(a, SimultaneousEffects):
            return a
    raise EffectRefused("applier: the slot callable was handed no "
                        "SimultaneousEffects")


# ---------------------------------------------------------------------------
# The device applier
# ---------------------------------------------------------------------------
class DeviceApplier:
    """THE per-env EFFECT LIST ``(N, cap, F)`` -- an instance of the record
    list (``UnitRecordList``: per-row staging and ONE gather per env, so a
    per-row producer's records land in row order; #29 P7a.1) -- and the book
    channels ``(N, N_BOOKS)``, on the tick stream
    (``breach_physics.FieldApplierBuffer``, the CUDA build only).

    Producers APPEND: :meth:`stage` sets every env's list from host records
    (the first ``cap`` kept, the rest counted as dropped: the host staging the
    gates use, and a chained run's host-made records); a device producer
    (slot 4a, ``simulation.unit_kernels.melee``) stages per row and gathers
    after whatever the list holds, so several producers continue one list
    before it is walked. :meth:`begin` empties it. The walk
    (``unit_kernels.apply_effects`` / ``resolve``, one kernel, a STAGE
    argument) zeroes the books, walks every env's list and downloads them.
    ``n_units`` is the store's row count (the staging's), ``row_cap`` a row's
    staging. Refuses a compiled module whose applier layout is not this
    table's."""

    def __init__(self, n_env: int, cap: int = E_CAP, bp=None, n_units: int = 1,
                 row_cap: int = ROW_CAP):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError("applier: the device applier needs the CUDA "
                               "build (breach_physics.HAS_CUDA)")
        lay = layout()
        compiled = bp.applier_layout()
        if compiled["fingerprint"] != lay.fingerprint                 or int(compiled["f"]) != lay.width                 or int(compiled["n_books"]) != N_BOOKS                 or int(compiled["n_planes"]) != N_PLANES:
            raise ValueError(
                f"applier: the compiled layout ({compiled['fingerprint'][:12]}) "
                f"is not simulation.applier's ({lay.fingerprint[:12]}): run "
                f"`{GENERATOR}` and rebuild cpp/build_cuda")
        self.bp = bp
        self.n_env = int(n_env)
        self.cap = int(cap)
        self.n_units = int(n_units)
        self.row_cap = int(row_cap)
        self._buf = bp.FieldApplierBuffer(self.n_env, self.cap, self.n_units,
                                          self.row_cap)

    @property
    def buffer(self):
        return self._buf

    @property
    def shape(self):
        return (self.n_env, self.cap, F)

    def begin(self) -> None:
        """Empty every env's list (counts, dropped, the staging)."""
        self._buf.begin()

    def stage(self, recs, counts) -> None:
        """Set every env's list from a packed host list (:func:`pack`):
        ``counts`` the REQUESTED counts (past ``cap`` counted as dropped)."""
        recs = np.ascontiguousarray(recs, dtype=np.int64)
        counts = np.ascontiguousarray(counts, dtype=np.int64).reshape(-1)
        if recs.shape != (self.n_env, self.cap, F) or counts.shape != (self.n_env,):
            raise ValueError(f"applier: a list must be ({self.n_env}, {self.cap}, "
                             f"{F}) with ({self.n_env},) counts, not "
                             f"{recs.shape} / {counts.shape}")
        self._buf.stage(recs, counts)

    def stage_lists(self, per_env) -> None:
        """:meth:`stage` of N envs' ``(R_e, F)`` record arrays."""
        self.stage(*pack(per_env, self.cap))

    def download(self):
        """``(records (N, cap, F), counts (N,), dropped (N,))`` int64."""
        return self._buf.download()

    def records(self) -> list:
        """Per env, its records as a list of int tuples (list order)."""
        recs, counts, _dropped = self.download()
        return [[tuple(int(x) for x in recs[e, i]) for i in range(int(counts[e]))]
                for e in range(recs.shape[0])]


def books_dict(books) -> dict:
    """``(N, N_BOOKS)`` as ``{name: (N,) int64}``."""
    b = np.asarray(books, dtype=np.int64)
    return {n: b[:, i].copy() for i, n in enumerate(BOOK_NAMES)}


def host_book(gmap, ref: str) -> int:
    """The host quantity a book twins (``attr`` or ``attr:key``; #29 P6b.1: a
    LIST attribute's ``key`` is an index, ``trace_clip_books:g``)."""
    if ":" in ref:
        attr, key = ref.split(":", 1)
        obj = getattr(gmap, attr)
        if isinstance(obj, (list, tuple)):
            return int(obj[int(key)])
        return int(obj.get(key, 0))
    return int(getattr(gmap, ref))


__all__ = ["HEADER_REL", "GENERATOR", "FIELDS", "F", "KINDS",
           "KIND_DESTROY_WALL", "E_CAP", "BookRow", "BOOKS", "BOOK_NAMES",
           "N_BOOKS", "PLANES", "PLANE_NAMES", "N_PLANES", "CLEAR_PLANES",
           "CLEAR_NAMES", "EffectOverflow", "ApplierFault", "EffectRefused",
           "ARG_SLOTS", "KIND_FIELDS", "KIND_HIT", "KIND_STATUS", "KIND_PUSH",
           "KIND_CHEW", "KIND_PAYLOAD", "UNAPPLIED_KINDS", "PAYLOAD_KINDS",
           "STAGES", "STAGE_NAMES", "STAGE_TOPOLOGY", "STAGE_RESOLVE", "ROW_CAP",
           "STAGE_DETONATE", "PAYLOAD_REFUSALS", "payload_refusal",
           "payload_refusal_reason", "STAGE_SOURCES", "BulkGasOverflow",
           "kind_field", "make_record", "record_of", "records_of", "effects_of",
           "payload_index", "KIND_SPRAY_CELLS", "records_and_cells_of",
           "non_resident_planes", "Layout", "layout", "header_text",
           "destroy_records", "pack", "DeviceApplier", "books_dict",
           "host_book", "STAGE_DOORS", "SEAM_REFUSALS", "SEAM_REFUSAL_NAMES",
           "STAGE_VENTS", "LedgerRangeRefused"]
__all__ += ["STAGE_EMIT", "STAGE_LOGIC", "STAGE_DOOR_INPUTS", "STAGE_PUMPS", "STAGE_SWAP",
            "STAGE_VENTS", "STAGE_DOORS", "SignalRangeRefused"]
__all__ += ["STAGE_TIMED_CHARGES", "STAGE_HEAT_SOURCES", "HeatChainRefused"]
