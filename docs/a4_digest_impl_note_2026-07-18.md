# A4 impl note — `__entity__`/`__signals__` digest sections (2026-07-18)

Arc A patch A4 (plan: `arc_a_patch_plan_2026-07-18.md`). Gate question: how do
entity digest sections join the bit-identity gate while the dormancy guarantee
holds — entity-free levels byte-identical, existing goldens NEVER regenerated
(entity doc §7)?

## The rule: absence-transparent fold, spec version stays 1

`tick_digest` today hashes `fd | unit_hash`. A4 extends it:

```
h = H(fd); h |= "|" + unit_hash            # exactly today's bytes
if entity sections present in snapshot:
    h |= "|__entity__|"  + entity_hash     # appended ONLY when present
    h |= "|__signals__|" + signals_hash
```

- **Entity-free snapshot → the hashed byte stream is bit-identical to today**
  → every existing digest, golden, and x-arch artifact stays valid unchanged.
- `DIGEST_SPEC_VERSION` stays **1**: the version names the byte contract, and
  the v1 contract for entity-free snapshots is untouched — this is a strict,
  absence-transparent *extension*, not an alteration. (A bump exists to force
  golden regen; the dormancy guarantee explicitly forbids regen — bumping
  here would violate the design it implements.) `field_digest_spec.toml`
  documents the extension in an additive section.

## Presence rule (deterministic, level-derived)

Sections are present **iff the loaded level has ≥ 1 `[[entity]]` instance**
(`LevelData.entities` non-empty). Not class-conditional, not state-conditional
— one boolean derived from level content, identical on every machine. All
existing levels have zero entities, so the guarantee follows structurally.
No collision risk: the appended marker makes the message strictly longer and
distinct from any entity-free stream.

## `__entity__` serialization (canonical, cross-machine)

Per entity in **ordinal (file/id) order** — the §3a single ordering rule:

- header: `"{ordinal}|{id}|{class_name}"` ASCII
- then each **synced-kind field** in declared (schema) order:
  `"{field_name}|"` + value as **little-endian int64**. Synced kinds =
  KIND_INT, KIND_Q16, KIND_BOOL (0/1), enums (declared-order index),
  KIND_ENTITY_REF (target ordinal, −1 unwired/dangling). KIND_FLOAT_RENDER /
  KIND_STR / KIND_COLOR_RGB / KIND_STR_LIST are render/authoring-bound and
  EXCLUDED (mirror of `EXCLUDED_FLOAT_FIELDS`).
- then `"alive|"` + int64 0/1 (the free signal every entity carries; always 1
  in Arc A — no entity destruction path exists yet; A6/A-B make it real).

In Arc A this state is load-constant; the point is that A6 door state and
Arc B controller state land inside an ALREADY-GATED section with zero further
digest surgery.

## `__signals__` serialization

Defined now, empty until Arc B's SignalBus: tuples
`(emitter_ordinal, signal_name, int64 value)` sorted by (ordinal, name),
same header+LE-int64 encoding. An empty bus hashes as the fixed empty-section
constant — present (when entities exist) but stable, so Arc B's first real
signal changes the digest exactly when behavior begins, not before.

## get_state / recorder (the f601455 lesson)

`get_state` and recorder dumps grow `__entity__`/`__signals__` entries under
the same presence rule, carrying the serialized state (not just the hash) so
the A/B harness can LOCATE an entity-state divergence per-instance the way it
locates a field divergence per-cell.

## Tests (the gate)

1. Dormancy: full existing suite green with ZERO golden edits (structural
   proof); plus an explicit test — digest of an entity-free snapshot computed
   through the new path equals one computed by the pre-A4 algorithm (inline
   reference copy).
2. Entity-present level → digest differs from its entity-stripped twin.
3. Stability: same entity level, two loads → identical digest; field-order /
   dict-order independence (iteration is ordinal+declared order only).
4. Ref encoding: unwired (−1) vs wired (ordinal) digests differ; dangling ref
   encodes −1 (matches its load-warning semantics).
5. Recorder round-trip includes the sections iff entities present.
