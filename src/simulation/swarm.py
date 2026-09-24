"""The swarm store scaffold (arc #63 P2, docs/swarm_P2_impl.md v2).

Executes ``docs/architecture/engine/17_swarm_units.md`` (BLESSED) §2, §6, §7
and §9b's "shared foundation" row. Per-species SoA arrays on GameMap, born
``(N_ENV, max_units)`` through the residency path, with a lifecycle facade
(:class:`SwarmUnits`), a strict species table re-bound through one explicit
reload seam (:func:`reload_species`), and a presence-gated ``SWARM_SECT_V1``
section the digest hashes and the recorder mirrors raw.

Scope (P2): the store, spawn/reclaim, the invariant checker, the facade, the
species table, the digest section + carrier, and picklability. NO behaviour,
no tick-conductor line, no C++, no save format, no rendering, no food — all
later patches (see the impl doc §9 for the exact hand-off each one inherits).

Must not import :mod:`simulation.gamemap` or :mod:`simulation.simulation`
(the import direction is gamemap -> swarm).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields as _dc_fields, replace as _dc_replace
from typing import NamedTuple

import numpy as np

from config import CFG, Namespace
from simulation.entities.schema import (
    Field, KIND_INT, KIND_REAL_Q16, field_value_error,
)
from simulation.factions import FACTION_NONE
from simulation import swarm_fixed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The RL-batch §A rule-1 batch axis, and the codebase's FIRST batch-axis
# constant (no other N exists in the code). The one N until the RL arc's B1
# turns it into a construction parameter -- never a second N.
N_ENV = 1

# The species set is CLOSED IN CODE and APPEND-ONLY (a species cannot exist
# without a kernel, which is code, so the SET lives in code and the NUMBERS
# live in TOML, §4). Tuple order == section block order, so it only ever
# grows at the end. Names match ``^[a-z][a-z0-9]*$`` (no underscore, R11).
SWARM_SPECIES = ("larva",)


class Column(NamedTuple):
    """One roster column: attribute-name suffix + numpy dtype."""
    name: str
    dtype: type


# The v1 roster (§1.2), declaration order == section column order. `T_prev`
# is deliberately ABSENT: the Review log (2026-09-10) ruled sensing to be a
# spatial head-sweep, so the chapter's `T_prev` mentions are stale.
#
# The empty-slot form is the ALL-ZERO slot: AI_DEAD == 0 and ids start at 1
# make every column of an unoccupied slot zero, whether the slot was never
# spawned or was reclaimed (§1.2). A live unit is never AI_DEAD.
ROSTER = (
    Column("pos_x", np.int32),          # Q16.16 tile-x, tile units of the runtime grid
    Column("pos_y", np.int32),          # Q16.16 tile-y
    Column("heading", np.int32),        # Q16.16 radians, canonical (-PI_Q16, PI_Q16]
    Column("hp", np.int32),             # Q16.16 hit points
    Column("ai_state", np.int32),       # AI_DEAD / AI_RUN / AI_EAT / AI_COMA
    Column("dist_walked", np.uint32),   # Q16.16 path length, wraps mod 2**32
    Column("unit_id", np.int32),        # persistent identity (0 == "no unit")
    Column("faction", np.int32),        # present from v1, unread in v1
    Column("valid", np.uint8),          # slot occupied
)

AI_DEAD, AI_RUN, AI_EAT, AI_COMA = 0, 1, 2, 3

FIRST_UNIT_ID = 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


# ---------------------------------------------------------------------------
# Names — formed in ONE place (§1.1), so a later rename is digest-neutral
# (the wire contract is the section, keyed by species + short column name).
# ---------------------------------------------------------------------------

_SPECIES_NAME_RE = re.compile(r"^[a-z][a-z0-9]*\Z")
_COLUMN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\Z")

# A column named "high_water" would alias `swarm_<species>_high_water` (the
# per-env scalar) -- reserved so `check_store_names` can catch it.
RESERVED_COLUMNS = ("high_water",)

NEXT_UNIT_ID_ATTR = "swarm_next_unit_id"
SPECIES_ATTR = "swarm_species"
HOST_DIRTY_ATTR = "swarm_host_dirty"
SPECIES_HASH_NPZ_KEY = "swarm_species_hash"


def store_attr(species: str, col: str) -> str:
    """``gmap.swarm_<species>_<col>`` -- the ONE place this name is formed.

    Doubles as the recorder's ``.npz`` key for the same column (§7)."""
    return f"swarm_{species}_{col}"


def high_water_attr(species: str) -> str:
    return f"swarm_{species}_high_water"


def check_store_names(species_tuple, roster) -> None:
    """Name hygiene (R11): raises ``ValueError`` on the first violation.

    - every species name matches ``^[a-z][a-z0-9]*$``;
    - every column name matches ``^[a-z][a-z0-9_]*$`` and is not reserved
      (``RESERVED_COLUMNS``, which would alias ``swarm_<sp>_high_water``);
    - every generated attribute name (== every ``.npz`` key) is pairwise
      distinct from every other one AND from the fixed names.

    Catches, e.g., a future species ``next``, whose ``store_attr("next",
    "unit_id")`` would equal ``swarm_next_unit_id``.
    """
    for sp in species_tuple:
        if not _SPECIES_NAME_RE.match(sp):
            raise ValueError(
                f"swarm species name {sp!r} must match "
                f"{_SPECIES_NAME_RE.pattern!r} (no underscore, R11)")
    for col in roster:
        if not _COLUMN_NAME_RE.match(col.name):
            raise ValueError(
                f"swarm column name {col.name!r} must match "
                f"{_COLUMN_NAME_RE.pattern!r}")
        if col.name in RESERVED_COLUMNS:
            raise ValueError(
                f"swarm column name {col.name!r} is reserved -- it would "
                f"alias swarm_<species>_{col.name}")

    names = [NEXT_UNIT_ID_ATTR, SPECIES_ATTR, HOST_DIRTY_ATTR,
             SPECIES_HASH_NPZ_KEY]
    for sp in species_tuple:
        names.append(high_water_attr(sp))
        for col in roster:
            names.append(store_attr(sp, col.name))
    seen: set = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"swarm store name collision: {name!r} is generated more "
                f"than once (name hygiene, R11)")
        seen.add(name)


check_store_names(SWARM_SPECIES, ROSTER)

# Every store array GameMap allocates (§1.3): the columns (species x ROSTER,
# in that nested order), then each species' high_water, then the shared
# next_unit_id. Born through the residency path from day one (§A rule 3).
SWARM_RESIDENT_NAMES = tuple(
    store_attr(sp, col.name) for sp in SWARM_SPECIES for col in ROSTER
) + tuple(high_water_attr(sp) for sp in SWARM_SPECIES) + (NEXT_UNIT_ID_ATTR,)


# ---------------------------------------------------------------------------
# The species table (§4) -- schema in code (entities/schema.py::Field),
# numbers in TOML.
# ---------------------------------------------------------------------------

# P2's fields (only what P2 consumes; §4.1). `max_units` sits in the species
# row because the store is per species -- different species will want
# different capacities. Deferred to P3/P6: speed/radius/thresholds/food dials.
SPECIES_FIELDS = (
    Field("max_units", KIND_INT, minimum=1, maximum=1 << 20,
          doc="slots per env (fork 2b); LOAD-TIME only"),
    Field("hp_max", KIND_REAL_Q16, minimum=2 ** -16, maximum=32767.0,
          doc="hit points at spawn; HOT (applies from the next tick)"),
)

# The reload class per field -- a separate mapping (R3): "hot" applies from
# the next tick (through the seam), "load" only at the next reset/level load.
SPECIES_RELOAD: dict = {
    "max_units": "load",
    "hp_max": "hot",
}

if set(SPECIES_RELOAD) != {f.name for f in SPECIES_FIELDS}:
    raise ValueError(
        "swarm.SPECIES_RELOAD keys must equal SPECIES_FIELDS names "
        f"(got {sorted(SPECIES_RELOAD)}, expected "
        f"{sorted(f.name for f in SPECIES_FIELDS)})")
if not set(SPECIES_RELOAD.values()) <= {"hot", "load"}:
    raise ValueError(
        f"swarm.SPECIES_RELOAD values must be 'hot' or 'load', got "
        f"{sorted(set(SPECIES_RELOAD.values()))}")


def _species_stored_attr(f: Field) -> str:
    """The stored-attribute rule: ``<name>`` for KIND_INT, ``<name>_q`` for
    KIND_REAL_Q16 (quantized once at load, door 2)."""
    if f.kind == KIND_INT:
        return f.name
    if f.kind == KIND_REAL_Q16:
        return f"{f.name}_q"
    raise ValueError(
        f"swarm species field {f.name!r}: unsupported kind {f.kind!r} "
        f"(P2 only uses KIND_INT / KIND_REAL_Q16)")


@dataclass(frozen=True)
class SwarmSpeciesRow:
    """One species' in-force tuning numbers. Module-level + statically
    declared, hence picklable (R4, T19) -- never a runtime-generated class."""
    species: str
    max_units: int
    hp_max_q: int


_expected_row_attrs = frozenset(
    {"species"} | {_species_stored_attr(f) for f in SPECIES_FIELDS})
_actual_row_attrs = frozenset(f.name for f in _dc_fields(SwarmSpeciesRow))
if _actual_row_attrs != _expected_row_attrs:
    raise ValueError(
        "swarm.SwarmSpeciesRow fields must equal {'species'} union the "
        f"stored attributes of SPECIES_FIELDS (got {sorted(_actual_row_attrs)}"
        f", expected {sorted(_expected_row_attrs)})")


def _species_table_hash(rows) -> str:
    """SHA-256 hex over the numbers IN FORCE (post-quantization), the
    ``registry_content_hash`` recipe (registry.py:171-181) over a swarm
    payload: sorted keys, compact separators, ascii-only, so two TOML
    spellings of the same numbers hash equal and the hash is stable across
    machines / dict insertion order."""
    species_payload = {}
    for r in rows:
        row_values = {}
        for f in SPECIES_FIELDS:
            attr = _species_stored_attr(f)
            row_values[attr] = int(getattr(r, attr))
        species_payload[r.species] = row_values
    canonical = json.dumps(
        {"format": "SWARM_SPECIES_V1", "species": species_payload},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SwarmSpeciesTable:
    """The strict, validated ``[swarm.<species>]`` table in force.

    No ``.source`` attribute: v1's lazy identity check (removed, R3) was its
    only use. Installed only at GameMap construction and by the reload seam
    (:func:`reload_species`); every other reader is pure (§4.3).
    """
    rows: tuple
    hash: str

    @classmethod
    def from_rows(cls, rows) -> "SwarmSpeciesTable":
        rows = tuple(rows)
        return cls(rows=rows, hash=_species_table_hash(rows))

    @classmethod
    def from_config(cls, cfg=CFG) -> "SwarmSpeciesTable":
        """Validate + quantize ``[swarm.<species>]`` (§4.2). Raises
        ``ValueError`` naming ``[swarm.<sp>].<key>`` on the first violation:
        a missing ``[swarm]``, a missing/unknown species row, a non-table
        key directly under ``[swarm]``, an unknown/missing key in a row, or
        a ``field_value_error`` (wrong type / out of bounds)."""
        swarm_cfg = getattr(cfg, "swarm", None)
        if swarm_cfg is None:
            raise ValueError(
                "[swarm] missing from config.toml -- every swarm species "
                "needs a [swarm.<species>] row")
        cfg_dict = vars(swarm_cfg)
        for key, value in cfg_dict.items():
            if key not in SWARM_SPECIES:
                raise ValueError(
                    f"[swarm].{key}: unknown swarm species {key!r} (known: "
                    f"{SWARM_SPECIES}) -- typo guard")
            if not isinstance(value, Namespace):
                raise ValueError(
                    f"[swarm].{key} must be a table ([swarm.{key}]) -- P2 "
                    f"has no global [swarm] keys")

        known_keys = {f.name for f in SPECIES_FIELDS}
        rows = []
        for sp in SWARM_SPECIES:
            if sp not in cfg_dict:
                raise ValueError(f"[swarm.{sp}] missing from config.toml")
            row_dict = vars(cfg_dict[sp])
            for key in row_dict:
                if key not in known_keys:
                    raise ValueError(
                        f"[swarm.{sp}].{key}: unknown key (known: "
                        f"{sorted(known_keys)})")
            stored = {"species": sp}
            for f in SPECIES_FIELDS:
                if f.name not in row_dict:
                    raise ValueError(f"[swarm.{sp}].{f.name} missing")
                value = row_dict[f.name]
                err = field_value_error(f, value)
                if err:
                    raise ValueError(
                        f"[swarm.{sp}].{f.name} = {value!r}: {err}")
                attr = _species_stored_attr(f)
                if f.kind == KIND_INT:
                    stored[attr] = int(value)
                else:  # KIND_REAL_Q16
                    stored[attr] = swarm_fixed.quantize_scalar(value)
            rows.append(SwarmSpeciesRow(**stored))
        return cls.from_rows(rows)

    def row(self, name: str) -> SwarmSpeciesRow:
        for r in self.rows:
            if r.species == name:
                return r
        raise KeyError(f"unknown swarm species {name!r}")


# ---------------------------------------------------------------------------
# Allocation (§1.3) -- the one GameMap hook.
# ---------------------------------------------------------------------------

def allocate_swarm_store(gmap, cfg=CFG) -> None:
    """Called once, as the LAST statement of ``GameMap.__init__``.

    1. ``gmap.swarm_species = SwarmSpeciesTable.from_config(cfg)`` -- the
       load-time validation (raises on error), the same role as
       ``self.materials`` / ``self.gases``.
    2. ``gmap.swarm_next_unit_id`` -- shape (N_ENV,) int32, FIRST_UNIT_ID.
    3. Per species: ``swarm_<sp>_high_water`` (N_ENV,) int32, and every
       ROSTER column at (N_ENV, max_units), the roster dtype.
    4. ``gmap.swarm_host_dirty`` -- a plain dict, NOT resident/digested.

    Allocation draws no RNG. Runs on every level and every reset."""
    gmap.swarm_species = SwarmSpeciesTable.from_config(cfg)
    gmap.swarm_next_unit_id = np.full((N_ENV,), FIRST_UNIT_ID, dtype=np.int32)
    for sp in SWARM_SPECIES:
        row = gmap.swarm_species.row(sp)
        setattr(gmap, high_water_attr(sp),
                np.zeros((N_ENV,), dtype=np.int32))
        for col in ROSTER:
            setattr(gmap, store_attr(sp, col.name),
                    np.zeros((N_ENV, row.max_units), dtype=col.dtype))
    gmap.swarm_host_dirty = {sp: False for sp in SWARM_SPECIES}


class SwarmFull(ValueError):
    """Spawn requested more units than there are free slots (§2.2).

    Raised BEFORE any write, so the store is left byte-identical (the
    ``SealBlocked`` precedent, ``gamemap.py::SealBlocked``). An authoring
    error, never a capacity clamp (R10) -- see :meth:`SwarmUnits.spawn`."""


def swarm_column(gmap, species: str, col: str):
    """The read accessor for one species' live column array (§3, R12).

    A free function (not a method) so P3's PhysicsRunner, P4 and P5 -- which
    hold ``gmap``, not ``sim.swarm`` -- never format a
    ``swarm_<species>_<col>`` attribute name themselves."""
    return getattr(gmap, store_attr(species, col))


# ---------------------------------------------------------------------------
# The invariant checker (§2.6, R4) -- a pure reader.
# ---------------------------------------------------------------------------

def check_invariants(gmap) -> list:
    """Human-readable violations of the lifecycle invariants (empty ==
    every invariant holds). Per env and species:

    1. ``0 <= high_water <= max_units``.
    2. ``FIRST_UNIT_ID <= next_unit_id <= INT32_MAX``, and
       ``next_unit_id == FIRST_UNIT_ID`` iff every species' ``high_water
       == 0`` in that env.
    3. ``valid in {0, 1}`` everywhere.
    4. Every slot with ``valid == 0`` (freed or never spawned), at ANY
       index, is all-zero in every column.
    5. Every slot with ``valid == 1`` lies in ``[0, high_water)``, has
       ``FIRST_UNIT_ID <= unit_id < next_unit_id``, has ``ai_state`` in
       ``{AI_RUN, AI_EAT, AI_COMA}``, and has heading in
       ``(-PI_Q16, PI_Q16]``.
    6. Live ``unit_id``\\ s are unique across ALL species in the env (the
       shared id space).
    """
    violations = []
    species_table = getattr(gmap, "swarm_species", None)
    if species_table is None:
        return violations

    next_unit_id_arr = gmap.swarm_next_unit_id
    for e in range(N_ENV):
        nid = int(next_unit_id_arr[e])
        if not (FIRST_UNIT_ID <= nid <= INT32_MAX):
            violations.append(
                f"env {e}: next_unit_id {nid} outside "
                f"[{FIRST_UNIT_ID}, {INT32_MAX}]")

        hw_by_species = {}
        for sp in SWARM_SPECIES:
            hw_by_species[sp] = int(getattr(gmap, high_water_attr(sp))[e])
        any_spawned = any(hw > 0 for hw in hw_by_species.values())
        if (nid == FIRST_UNIT_ID) == any_spawned:
            violations.append(
                f"env {e}: next_unit_id == FIRST_UNIT_ID must hold iff "
                f"every species' high_water == 0 (nid={nid}, "
                f"high_water={hw_by_species})")

        seen_ids: dict = {}
        for sp in SWARM_SPECIES:
            row = species_table.row(sp)
            hw = hw_by_species[sp]
            if not (0 <= hw <= row.max_units):
                violations.append(
                    f"env {e} species {sp}: high_water {hw} outside "
                    f"[0, {row.max_units}]")

            valid_arr = swarm_column(gmap, sp, "valid")[e]
            bad_valid = ~np.isin(valid_arr, (0, 1))
            if bad_valid.any():
                violations.append(
                    f"env {e} species {sp}: 'valid' has "
                    f"{int(bad_valid.sum())} value(s) outside {{0,1}}")

            is_valid = valid_arr == 1
            empty_mask = ~is_valid
            if empty_mask.any():
                for c in ROSTER:
                    col_arr = swarm_column(gmap, sp, c.name)[e]
                    nz = empty_mask & (col_arr != 0)
                    if nz.any():
                        bad = int(np.flatnonzero(nz)[0])
                        violations.append(
                            f"env {e} species {sp}: column '{c.name}' "
                            f"nonzero at empty slot {bad} "
                            f"({int(nz.sum())} such slot(s))")

            live_idx = np.flatnonzero(is_valid)
            if live_idx.size:
                out_of_hw = live_idx[live_idx >= hw]
                if out_of_hw.size:
                    violations.append(
                        f"env {e} species {sp}: {out_of_hw.size} live "
                        f"slot(s) >= high_water {hw} (e.g. "
                        f"{int(out_of_hw[0])})")

                uid_arr = swarm_column(gmap, sp, "unit_id")[e]
                live_uid = uid_arr[live_idx].astype(np.int64)
                bad_uid = (live_uid < FIRST_UNIT_ID) | (live_uid >= nid)
                if bad_uid.any():
                    violations.append(
                        f"env {e} species {sp}: {int(bad_uid.sum())} live "
                        f"unit_id(s) outside [{FIRST_UNIT_ID}, {nid})")

                ai_arr = swarm_column(gmap, sp, "ai_state")[e]
                bad_state = ~np.isin(ai_arr[live_idx], (AI_RUN, AI_EAT, AI_COMA))
                if bad_state.any():
                    violations.append(
                        f"env {e} species {sp}: {int(bad_state.sum())} live "
                        f"slot(s) with ai_state outside "
                        f"{{RUN, EAT, COMA}}")

                heading_arr = swarm_column(gmap, sp, "heading")[e]
                live_heading = heading_arr[live_idx].astype(np.int64)
                bad_heading = ((live_heading <= -swarm_fixed.PI_Q16)
                               | (live_heading > swarm_fixed.PI_Q16))
                if bad_heading.any():
                    violations.append(
                        f"env {e} species {sp}: {int(bad_heading.sum())} "
                        f"live heading(s) outside "
                        f"(-{swarm_fixed.PI_Q16}, {swarm_fixed.PI_Q16}]")

                for uid in live_uid.tolist():
                    if uid in seen_ids:
                        violations.append(
                            f"env {e}: duplicate live unit_id {uid} "
                            f"(species {seen_ids[uid]!r} and {sp!r})")
                    else:
                        seen_ids[uid] = sp
    return violations


# ---------------------------------------------------------------------------
# SWARM_SECT_V1 -- the carrier + the one byte encoder (§5).
# ---------------------------------------------------------------------------

SWARM_DIGEST_KEY = "__swarm__"
SWARM_SECT_PREAMBLE = b"SWARM_SECT_V1\n"


def swarm_present(gmap) -> bool:
    """"Present" (§5.1): at least one spawn ever succeeded this run (env 0).

    Monotone within a run (``high_water`` never decreases): a run whose
    larvae have all died still hashes its section."""
    species_table = getattr(gmap, "swarm_species", None)
    if species_table is None:
        return False
    for sp in SWARM_SPECIES:
        hw_arr = getattr(gmap, high_water_attr(sp), None)
        if hw_arr is not None and int(hw_arr[0]) > 0:
            return True
    return False


def swarm_carrier(gmap) -> dict:
    """The ``__swarm__`` snapshot value -- copies, never views, of env 0
    only (R6: multi-env is the RL arc's B1 decision, kept out of the bytes
    on purpose so "env k with seed s == an N=1 run with seed s" stays
    reachable)."""
    if N_ENV != 1:
        raise NotImplementedError(
            "multi-env swarm section is the RL arc's B1 decision")

    species_table = getattr(gmap, "swarm_species", None)
    if species_table is None:
        # A gmap stand-in with no store allocated (a test double).
        return {"present": False, "next_unit_id": FIRST_UNIT_ID,
                "blocks": {}, "species_hash": ""}

    next_unit_id = int(gmap.swarm_next_unit_id[0])
    blocks = {}
    for sp in SWARM_SPECIES:
        hw = int(getattr(gmap, high_water_attr(sp))[0])
        if hw <= 0:
            continue   # omitted: keeps "adding a species never bumps
                       # another species' digest" true when it never spawns
        columns = {}
        for c in ROSTER:
            columns[c.name] = swarm_column(gmap, sp, c.name)[0, :hw].copy()
        blocks[sp] = {"high_water": hw, "columns": columns}

    return {
        "present": bool(blocks),
        "next_unit_id": next_unit_id,
        "blocks": blocks,
        "species_hash": species_table.hash,   # provenance, NEVER hashed
    }


def swarm_section_bytes(carrier: dict) -> bytes:
    """The ONE encoder: the digest hashes exactly these bytes, the recorder
    stores the same columns raw (T27 gates them equal, R4).

    ``SWARM_SECT_V1 / next_unit_id|<nid>\\n / (species|<sp>|high_water|<hw>\\n
    (<col>|<dtype.str>|(<hw>,)\\n<raw bytes>)*)*`` -- blocks in SWARM_SPECIES
    order, a species with hw == 0 omitted; columns in ROSTER order.

    R2: every integer token is ``str(int(x))`` -- even though the carrier
    already holds Python ints -- so a hand-built carrier with numpy scalars
    encodes identically (``str((np.int32(5),))`` is NumPy-version-dependent;
    the header never goes through tuple formatting)."""
    parts = [SWARM_SECT_PREAMBLE,
             f"next_unit_id|{int(carrier['next_unit_id'])}\n".encode("ascii")]
    for sp in SWARM_SPECIES:
        block = carrier["blocks"].get(sp)
        if block is None:
            continue
        hw = int(block["high_water"])
        parts.append(f"species|{sp}|high_water|{hw}\n".encode("ascii"))
        for c in ROSTER:
            arr = np.ascontiguousarray(block["columns"][c.name])
            if arr.dtype.byteorder == ">":
                raise ValueError(
                    f"swarm section: column '{c.name}' has a big-endian "
                    f"dtype -- refused (cross-machine hash safety)")
            parts.append(
                f"{c.name}|{arr.dtype.str}|({int(hw)},)\n".encode("ascii"))
            parts.append(arr.tobytes(order="C"))
    return b"".join(parts)


def require_swarm_carrier(gmap, snapshot: dict) -> None:
    """Strictness (the A4 strict-presence idiom): a swarm-present gmap may
    never meet a snapshot without the carrier -- a capture path could
    silently hash swarm-free for a swarm-present run. Swarm-free gmaps
    accept a keyless snapshot (pre-P2 snapshots are swarm-free by
    construction)."""
    if swarm_present(gmap) and SWARM_DIGEST_KEY not in snapshot:
        raise KeyError(
            f"snapshot is missing the '{SWARM_DIGEST_KEY}' presence "
            f"carrier but the gmap has a present swarm -- a swarm-present "
            f"run must never hash swarm-free")


# ---------------------------------------------------------------------------
# The one config-reload seam's target (§4.3, R3).
# ---------------------------------------------------------------------------

def reload_species(gmap, cfg=CFG) -> bool:
    """Re-bind ``gmap.swarm_species`` from the live ``cfg`` (Ctrl+R ->
    ``Simulation.on_config_reload()`` -> here). THE only writer of
    ``gmap.swarm_species`` after allocation -- every other reader
    (spawn, ``carrier()``, the recorder, P3's H2D) is pure.

    1. Build the candidate with full validation; on ``ValueError`` print ONE
       line and keep the in-force table, returning False (a rejection is
       loud, never a silent fallback or a crash).
    2. ``load``-class fields keep their in-force value (arrays are never
       reallocated mid-run): print exactly one line per changed field, then
       rebuild the candidate with the in-force value so its hash describes
       what is actually in force.
    3. Install ``gmap.swarm_species = candidate``.
    4. If ``.hash`` changed, print a "replay comparability ends here" line.

    Applies from the next tick (the seam runs between ticks, in input
    handling)."""
    try:
        candidate = SwarmSpeciesTable.from_config(cfg)
    except ValueError as e:
        print(f"[swarm] RELOAD REJECTED: {e}")
        return False

    current = gmap.swarm_species
    current_by_name = {r.species: r for r in current.rows}
    kept_rows = []
    for row in candidate.rows:
        cur_row = current_by_name[row.species]
        overrides = {}
        for f in SPECIES_FIELDS:
            if SPECIES_RELOAD[f.name] != "load":
                continue
            attr = _species_stored_attr(f)
            new_val = getattr(row, attr)
            old_val = getattr(cur_row, attr)
            if new_val != old_val:
                print(f"[swarm] {row.species}.{f.name} {old_val} -> "
                      f"{new_val} applies at the next reset/level load")
                overrides[attr] = old_val
        kept_rows.append(_dc_replace(row, **overrides) if overrides else row)

    candidate = SwarmSpeciesTable.from_rows(kept_rows)
    gmap.swarm_species = candidate
    if candidate.hash != current.hash:
        print(f"[swarm] species table reloaded: {current.hash[:12]} -> "
              f"{candidate.hash[:12]} (replay comparability ends here)")
    return True


# ---------------------------------------------------------------------------
# Ingress validation helpers for the facade (door 2 of the ingress rule).
# ---------------------------------------------------------------------------

def _ingress_int64(name: str, value):
    """``value`` -> an int64 ndarray (scalar 0-D or 1-D), or TypeError /
    ValueError (§2.2 validation step 2). Raw Q16 only: bool, float, uint64,
    object and str inputs raise TypeError; an over-large Python int (which
    ``np.asarray`` would otherwise silently widen to an object array) raises
    TypeError too."""
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        raise TypeError(f"spawn: {name} must not be bool")
    if arr.dtype.kind == "f":
        raise TypeError(
            f"spawn: {name} must be an integer dtype, not float "
            f"({arr.dtype}) -- raw Q16 only, convert through swarm_fixed")
    if arr.dtype.kind in ("O", "U", "S"):
        raise TypeError(
            f"spawn: {name} must be a signed int (any width) or unsigned "
            f"up to uint32, got {arr.dtype}")
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(
            f"spawn: {name} must be a signed int (any width) or unsigned "
            f"up to uint32, got {arr.dtype}")
    if arr.dtype.kind == "u" and arr.dtype.itemsize > 4:
        raise TypeError(
            f"spawn: {name} unsigned dtype {arr.dtype} wider than uint32 "
            f"is not accepted")
    if arr.ndim > 1:
        raise ValueError(
            f"spawn: {name} must be scalar or 1-D, got {arr.ndim}-D")
    try:
        return arr.astype(np.int64)
    except OverflowError as e:
        raise TypeError(
            f"spawn: {name} value does not fit int64 -- {e}") from e


def _spawn_batch_len(named_arrays: dict) -> int:
    """The common 1-D broadcast length k across the spawn inputs (§2.2): all
    1-D inputs must share one length; an all-scalar call is k == 1."""
    lengths = {a.shape[0] for a in named_arrays.values() if a.ndim == 1}
    if len(lengths) > 1:
        raise ValueError(
            f"spawn: inconsistent batch lengths among "
            f"{sorted(named_arrays)}: {sorted(lengths)}")
    return lengths.pop() if lengths else 1


# ---------------------------------------------------------------------------
# The facade (§3) -- chapter §9b's "the only class".
# ---------------------------------------------------------------------------

class SwarmUnits:
    """The swarm-unit lifecycle facade. Holds no arrays -- GameMap stores
    them (fork 2a); this facade reads and writes ``gmap`` in place. THE only
    Python-side writer of the store (P3's kernel and its CPU twin are the
    sim-side writers). Constructed fresh on every ``Simulation`` reset."""

    def __init__(self, gmap):
        self.gmap = gmap

    def spawn(self, species: str, pos_x_q, pos_y_q, heading_q, *,
              faction=FACTION_NONE, env: int = 0):
        """The k-th request takes the k-th lowest free slot (§2.2).

        Draws NO randomness -- callers own randomness and pass explicit Q16
        values. Returns the slots written, an int64 array in request order.
        Validation runs BEFORE any write (atomic refusal): overflow raises
        :class:`SwarmFull`, never a partial fill / capacity clamp (R10).
        """
        gmap = self.gmap

        # 1. species / env.
        if species not in SWARM_SPECIES:
            raise ValueError(
                f"unknown swarm species {species!r} (known: {SWARM_SPECIES})")
        if isinstance(env, bool) or not isinstance(env, (int, np.integer)):
            raise ValueError(f"spawn: env must be a plain int, got {env!r}")
        env = int(env)
        if not (0 <= env < N_ENV):
            raise ValueError(f"spawn: env {env} outside [0, {N_ENV})")

        # 2. dtype/shape ingress -> int64 arrays, then k = the common length.
        pos_x64 = _ingress_int64("pos_x_q", pos_x_q)
        pos_y64 = _ingress_int64("pos_y_q", pos_y_q)
        heading64 = _ingress_int64("heading_q", heading_q)
        faction64 = _ingress_int64("faction", faction)
        named = {"pos_x_q": pos_x64, "pos_y_q": pos_y64,
                 "heading_q": heading64, "faction": faction64}
        k = _spawn_batch_len(named)
        if k == 0:
            return np.zeros((0,), dtype=np.int64)

        pos_x64 = np.broadcast_to(pos_x64, (k,))
        pos_y64 = np.broadcast_to(pos_y64, (k,))
        heading64 = np.broadcast_to(heading64, (k,))
        faction64 = np.broadcast_to(faction64, (k,))

        # 3. range checks, in int64 -- ALL before any write.
        w_bound = int(gmap._w) << swarm_fixed.FP_SHIFT
        h_bound = int(gmap._h) << swarm_fixed.FP_SHIFT
        if not (np.all(pos_x64 >= 0) and np.all(pos_x64 < w_bound)):
            raise ValueError(
                f"spawn: pos_x_q outside grid bounds [0, {w_bound})")
        if not (np.all(pos_y64 >= 0) and np.all(pos_y64 < h_bound)):
            raise ValueError(
                f"spawn: pos_y_q outside grid bounds [0, {h_bound})")
        if not (np.all(heading64 >= INT32_MIN) and np.all(heading64 <= INT32_MAX)):
            raise ValueError("spawn: heading_q outside the int32 range")
        if not (np.all(faction64 >= INT32_MIN) and np.all(faction64 <= INT32_MAX)):
            raise ValueError("spawn: faction outside the int32 range")

        next_unit_id_arr = gmap.swarm_next_unit_id
        nid0 = int(next_unit_id_arr[env])
        if nid0 + k > INT32_MAX:
            raise ValueError(
                f"spawn: next_unit_id {nid0} + {k} would exceed INT32_MAX "
                f"({INT32_MAX})")

        # 4. capacity -- atomic refusal, never a partial fill.
        valid_col = swarm_column(gmap, species, "valid")[env]
        free_slots = np.flatnonzero(valid_col == 0)
        if k > free_slots.size:
            raise SwarmFull(
                f"spawn: {k} requested but only {free_slots.size} free "
                f"slot(s) in species {species!r} env {env}")
        slots = free_slots[:k]   # already ascending (flatnonzero order)

        # --- writes: every check above passed, so nothing below may fail.
        ids = nid0 + np.arange(k, dtype=np.int64)
        row = gmap.swarm_species.row(species)
        wrapped_heading = swarm_fixed.wrap_heading_q16(heading64)

        swarm_column(gmap, species, "pos_x")[env, slots] = pos_x64
        swarm_column(gmap, species, "pos_y")[env, slots] = pos_y64
        swarm_column(gmap, species, "heading")[env, slots] = wrapped_heading
        swarm_column(gmap, species, "hp")[env, slots] = row.hp_max_q
        swarm_column(gmap, species, "ai_state")[env, slots] = AI_RUN
        swarm_column(gmap, species, "dist_walked")[env, slots] = 0
        swarm_column(gmap, species, "unit_id")[env, slots] = ids
        swarm_column(gmap, species, "faction")[env, slots] = faction64
        swarm_column(gmap, species, "valid")[env, slots] = 1

        next_unit_id_arr[env] = nid0 + k
        hw_arr = getattr(gmap, high_water_attr(species))
        hw_arr[env] = max(int(hw_arr[env]), int(slots.max()) + 1)
        gmap.swarm_host_dirty[species] = True

        return slots.astype(np.int64)

    def reclaim(self, species: str, slots, *, env: int = 0):
        """Write the empty-slot form (every column zero) at each slot
        (§2.5). ``high_water``/``next_unit_id`` are unchanged. Validation
        (atomic, before any write): ``slots`` must be an integer array-like
        (bool/float raise ``TypeError``); every slot must lie in
        ``[0, high_water)``, be unique, and have ``valid == 1`` (reclaiming
        an empty slot is a caller bug, ``ValueError``)."""
        gmap = self.gmap
        if species not in SWARM_SPECIES:
            raise ValueError(
                f"unknown swarm species {species!r} (known: {SWARM_SPECIES})")
        if isinstance(env, bool) or not isinstance(env, (int, np.integer)):
            raise ValueError(f"reclaim: env must be a plain int, got {env!r}")
        env = int(env)
        if not (0 <= env < N_ENV):
            raise ValueError(f"reclaim: env {env} outside [0, {N_ENV})")

        arr = np.asarray(slots)
        if arr.dtype == np.bool_ or arr.dtype.kind not in ("i", "u"):
            raise TypeError(
                f"reclaim: slots must be an integer array-like, got dtype "
                f"{arr.dtype}")
        slots64 = arr.astype(np.int64).reshape(-1)
        if slots64.size == 0:
            return

        hw = int(getattr(gmap, high_water_attr(species))[env])
        if np.unique(slots64).size != slots64.size:
            raise ValueError(
                f"reclaim: duplicate slot indices {slots64.tolist()}")
        if np.any((slots64 < 0) | (slots64 >= hw)):
            raise ValueError(
                f"reclaim: slot index outside [0, high_water={hw})")
        valid_col = swarm_column(gmap, species, "valid")[env]
        empty = valid_col[slots64] == 0
        if empty.any():
            raise ValueError(
                f"reclaim: slot(s) {slots64[empty].tolist()} are already "
                f"empty (valid == 0) -- reclaiming an empty slot is a "
                f"caller bug")

        for c in ROSTER:
            swarm_column(gmap, species, c.name)[env, slots64] = 0
        gmap.swarm_host_dirty[species] = True

    def live_count(self, species: str, env: int = 0) -> int:
        valid = swarm_column(self.gmap, species, "valid")[env]
        return int(valid.sum())

    def high_water(self, species: str, env: int = 0) -> int:
        return int(getattr(self.gmap, high_water_attr(species))[env])

    def present(self) -> bool:
        return swarm_present(self.gmap)

    def column(self, species: str, col: str):
        return swarm_column(self.gmap, species, col)

    @property
    def species(self) -> SwarmSpeciesTable:
        """Read-only -- NEVER reloads. The reload seam
        (:func:`reload_species`) is the one writer of ``gmap.swarm_species``
        after allocation."""
        return self.gmap.swarm_species

    def carrier(self) -> dict:
        return swarm_carrier(self.gmap)
