---
paths:
  - "src/simulation/swarm.py"
  - "src/simulation/swarm_fixed.py"
  - "tests/*swarm*.py"
---

# Swarm: the store of swarm units, its digest, its facade, its table, its heading

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.
Erik's P6 ruling (2026-11-14, `docs/unit_design_2026-11-14.md` §7): TWO
stores (units and swarm units), ONE batch axis, ONE source of draws, and the
two sides meet only through FIELDS.

## Swarm store
- A swarm unit is NOT a `Unit`: it never touches `src/simulation/unit_schema.py`, `Unit.__slots__` or the device unit store (`src/simulation/unit_store.py`), and the unit side never gains swarm columns. Two stores (Erik, 2026-11-14).
- THE BOUNDARY (Erik, 2026-11-14): a swarm law reads map fields, never a unit column; a unit law sees the swarm only through a per-tile plane the swarm writes (the bite-dose plane, the count field), read by a coupling row. Never a record or queue of swarm events read by a unit law (a record channel is a design line first); a swarm kernel sees units only through the fields the unit side already writes (the slot-6 occupancy stamp).
- The batch axis is `ResidentBatch`'s: the store is per map, born in `GameMap._RESIDENT_SYNCED`; never a second N (Erik, 2026-11-14).
- A species is a `SWARM_SPECIES` entry + a species-table row + a kernel: never a class, never a per-species if-chain. A new column is a `ROSTER` row + a `SWARM_SECT` version bump, never a stray gmap array or a `DIGEST_FIELDS` row.
- Stable slots: never compaction, never atomic-ticket spawns (thread order would decide a slot). `unit_id` is identity; the slot is not, and a freed slot is the all-zero form.
- Every random number a swarm law draws is a keyed `sim.draws` block with its own `rng_salts.Salt` member (the Match draws row): never a second salt enum, a second key derivation from the seed or a second tick clock (Erik, 2026-11-14: one source of draws).
- A swarm law whose result depends on which unit came first (eating a corpse or a wall) is an effect KIND of the per-env serial applier (`src/simulation/applier.py`), never a second walker or a queue of its own; anything that changes a tile's material calls the device topology seam (`cpp/src/cuda_topology.cuh`).

## Swarm digest section
- Per-unit swarm state enters the digest and every snapshot ONLY through the presence-gated `__swarm__` carrier: absent on a swarm-free run, so no golden moves without a swarm. Every capture path writes it (`field_ab_harness.capture_snapshot`); a snapshot without the key reads as absent.
- The digest hashes the ONE encoder's bytes (`swarm_section_bytes`); the recorder stores the same columns raw, held equal by `tests/test_swarm_recorder.py`. Capacity and the species table never enter hashed bytes.

## Swarm facade
- The only Python-side writer of the store; the sim-side writers are the swarm kernels and their CPU twins. Never index `gmap.swarm_*` to write from gameplay, tools or tests.
- Spawn takes explicit Q16 values and draws nothing: callers own randomness. Overflow is `SwarmFull`, never a capacity clamp or a partial fill.
- Every host write sets the species' host-dirty flag in the same call, so the device upload sees it: a new writer that skips it leaves the device stale.

## Swarm species table
- A swarm tunable is a `Field` row + a `SPECIES_RELOAD` class + a TOML value under `[swarm.<species>]`, validated strictly at load and on every reload (an unknown key or species is an error) and quantized once at ingress through `swarm_fixed`.
- Installed only at GameMap construction and by the Config reload seam; every consumer reads `gmap.swarm_species` and never reloads. An invalid reload is rejected loudly and the prior table stays in force. The table hash is provenance, never digested.
- Every value a swarm kernel reads is a row of this table, read from the validated table object (the Device tables discipline): never a law constant passed as a kernel argument.

## Swarm heading law
- Every heading write, kit-derived or drawn, passes `wrap_heading_q16`. Its period is `2·PI_Q16`, deliberately NOT `philox32.TWO_PI_Q16` (the draw range): draw, then wrap; never unify the two constants.
- A device twin floor-mods in int64 and is identity-tested against the Python wrap over the whole input range.
