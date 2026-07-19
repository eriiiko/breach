# A4 impl note — `__entity__`/`__signals__` digest sections (v2, critique folded, 2026-07-18)

Arc A patch A4 (plan: `arc_a_patch_plan_2026-07-18.md`). Gate question: how do
entity digest sections join the bit-identity gate while the dormancy guarantee
holds — entity-free levels byte-identical, existing goldens NEVER regenerated
(entity doc §7)? v1 survived adversarial critique WITH FIXES (2 blockers, 6
should-fix — all folded below).

## The rule: absence-transparent fold, spec version stays 1

`tick_digest` today hashes `fd | unit_hash`. A4 extends it:

```
h = H(fd); h |= "|" + unit_hash            # exactly today's bytes
if n_entities > 0:
    h |= "|__entity__|"  + entity_hash     # appended ONLY when present
    h |= "|__signals__|" + signals_hash
```

- **Entity-free snapshot → the hashed byte stream is bit-identical to today**
  → every existing digest, golden, and x-arch artifact stays valid unchanged.
  (Sound: the entity-free tail is hex-only, so the `|__entity__|` marker can
  never appear in it; appended sections make the message strictly longer.)
- `DIGEST_SPEC_VERSION` stays **1** globally (a bump forces golden regen; the
  dormancy guarantee forbids it). **Section-local version instead (critique
  3):** the entity section's hashed preamble is `ENTITY_SECT_V1\n` — a future
  format change bumps it loudly without touching entity-free bytes. The xarch
  artifact line appends `ents=N,esect_v1,reg=<registry_content_hash[:12]>`
  when sections are present, so a cross-machine mismatch is attributable in
  one diff (critique 3+4).

## Presence: the carrier is explicit and strict (critique blocker 2)

- `get_state` / trajectory capture **always** write snapshot key
  `__entity__ = {"n_entities": N, ...}` — `N = 0` for an entity-free level.
  The marker **gates** the fold; it is never hashed when `N == 0`, so
  dormancy is untouched.
- `tick_digest` appends sections iff `n_entities > 0`.
- **Strictness (loud, like a missing field):** when the running sim has
  entities loaded, a snapshot missing the `__entity__` key raises — a capture
  path can never silently compute the entity-free digest for an
  entity-present run. Snapshots from pre-A4 recordings (no key at all) are
  entity-free by construction and hash as today.
- Presence stays **class-blind** (one rule, level-derived): consequences are
  accepted and stated — a render-only `light` entity trips presence, so the
  `[[light]]`→entity alias migration IS digest-changing and is therefore
  **A7-scoped, all at once, inside the single sanctioned re-baseline**; the
  A7 rationale must list levels that flipped for lights-only reasons
  (critique 8). Corollary for A8/any pre-A7 patch: adding one entity to a
  digest-suite level flips that golden — don't; new-entity tests use new
  fixture levels (critique 11).

## `__entity__` serialization (canonical, cross-machine)

The serializer reads **the runtime entity object**; Arc A's `EntityInstance`
is its degenerate load-constant form (critique 6). One module-level
`serialize_entity_state()` is consumed by BOTH the digest and the recorder —
never two serializers (critique 9).

Preamble `ENTITY_SECT_V1\n`, then per entity in **ordinal (file/id) order**
(§3a single ordering rule):

- header `"{ordinal}|{id}|{class_name}\n"` ASCII
- each **synced-kind** declared field in schema declaration order:
  `"{field_name}|"` + value as **signed little-endian int64**
  (`struct.pack('<q', v)` — out-of-range raises loudly) + `"\n"`.
  Synced kinds: KIND_INT, KIND_Q16, KIND_BOOL (0/1), KIND_ENUM
  (declared-choice index), KIND_ENTITY_REF (target ordinal; −1 for unwired
  "" AND dangling — both resolve to nothing at runtime).
  **EXCLUDED:** KIND_FLOAT_RENDER, KIND_STR, KIND_COLOR_RGB, KIND_STR_LIST,
  **and KIND_LENGTH_M (critique blocker 1)** — length_m is authoring-bound,
  stored unquantized; its synced consequence is quantized tile state already
  hashed via `material`/`obstacles`/`wall_hp`. No digest-time quantization,
  ever (one quantization site: load).
- `"alive|"` + int64 0/1 + `"\n"` (always 1 in Arc A — no destruction path).
- a per-class **runtime-state row block** (same `name|int64\n` encoding),
  empty in Arc A: A6 door state, Arc B EMA accumulators / controller phase /
  edge-detector prevs land here as *runtime rows defined per class* — NOT as
  schema FIELDS (they are not authorable) — under the section version, with
  zero mechanism surgery (critique 6).
- record terminator `"\n"` (headers and records are newline-delimited —
  injectivity does not rest on the registry being closed; additionally the
  registry rejects field/class/signal names outside `[A-Za-z0-9_]+` at
  registration) (critique 10).

**Registry provenance (critique 4+5):** entity-present digests are only
comparable at equal `registry_content_hash()` (which folds entities.toml
effective defaults via `registry_payload`) — the overlay is match-setup
material, like the seed. The hash is recorded with every entity-present
artifact (xarch line, recorder metadata); `field_digest_spec.toml` gains an
additive section stating the entity field list/order is defined by the
registry at the recorded content-hash, section format `ENTITY_SECT_V1`.

## `__signals__` serialization

Defined now, empty until Arc B's SignalBus: tuples
`(emitter_ordinal, signal_name, int64 value)` sorted by (ordinal, name),
newline-delimited, same int64 encoding, preamble `SIGNAL_SECT_V1\n`.
**Excludes the free `alive` signal — hashed ONLY as the `__entity__` row**
(critique 7): the bus's introduction must not flip digests before behavior
changes. `__signals__` carries declared class signals only; empty bus hashes
as the bare preamble (stable).

## get_state / recorder (the f601455 lesson)

`get_state` and recorder dumps carry the `serialize_entity_state()` payload
(not just the hash) under the same presence rule, so the A/B harness can
LOCATE an entity-state divergence per-instance the way it locates a field
divergence per-cell. The recorder's frozen .npz schema is unchanged for
entity-free levels; entity keys are additive and presence-gated (compatible
with the documented freeze).

## Tests (the gate)

1. Dormancy: full existing suite green with ZERO golden edits; plus explicit
   test — entity-free digest through the new path equals a pre-A4 reference
   implementation inline in the test.
2. Entity-present level → digest differs from its entity-stripped twin.
3. Strictness: sim-with-entities + snapshot lacking `__entity__` → raises.
4. Stability: two loads → identical digest; iteration order is
   ordinal+declaration only (dict-order independence).
5. Ref encoding: unwired/dangling → −1; wired → ordinal; digests differ.
6. Light-alias digest consequence pinned: legacy-[[light]] level hashes
   entity-free; its entity twin hashes entity-present (documents the A7
   re-baseline scope).
7. Name-charset registration guard; int64 overflow raises.
8. Recorder round-trip: sections present iff entities present; digest and
   recorder bytes come from the one serializer.
