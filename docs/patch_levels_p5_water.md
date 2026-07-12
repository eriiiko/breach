# P5 impl — `[water]` initial state: level-seeded water + aquariums (levels-w1)

> **P5 patch record** — design-gated 2026-07-07 (1 blocker + 7 majors resolved across two independent critiques: format B1 → .npy carrier; physics M1/M2 → at-rest scoping + drain asymmetry); built as specified 2026-07-08, annotations binding.

**Status:** design-gate **v2** (2026-07-07) — v1 amended per adversarial critique (format/scope
lens: B1, M1–M5, m1–m5 all resolved below; physics lens report pending — §2.6 test may amend).
**Spec:** `docs/architecture/engine/15_level_authoring.md` §2.3 — **the P5 patch amends the
chapter**: the carrier changes from 8-bit PNG + max_depth_m to the .npy below (critique B1/M5).
**Erik veto point:** water seed is editor-authored `.npy`, not a hand-paintable PNG (rationale §2.1).
**HUMAN-TEST at end:** Erik's aquarium demo (glass tank; shoot it; it drains).

## 1. Scout facts (binding, all re-verified by critics)

- `gmap.water_depth`: int32 **Q16.16 metres** (`gamemap.py:296`), in-place writes only; solver
  zeroes depth on solid every step (mass sink) → never seed solid tiles.
- Glass = solid-for-water under current config (no permeability key + occludes light → 0);
  **furniture = permeability 0.5 → NOT solid — water stands on furniture legitimately.**
- Pre-existing-water path designed for us: `_water_depth_before` lazy copy
  (`physics_runner.py:585-595`, "level-painted = pre-existing", no tick-1 spike).
- `water_fixed.quantize` is vectorized (array-in/array-out) and round-half-away-from-zero,
  matching C++ and FieldEdit's water path.
- Atmosphere t=0 precedent lives in `_update_caches` (`gamemap.py:424`); `_update_caches`
  never touches `water_depth`, so seeding right after it in `GameMap.__init__` is safe.
- `reset()` → `_reset_internal` → fresh `GameMap(self.level)` AND fresh runner
  (`_water_depth_before` re-arms) → seed reapplies, no spike after reset (physics critic m4).
- WaterPass renders any nonzero depth by default (`show_water=True`).
- **At-rest: PROVEN bit-exact for seeds whose wetted boundary is entirely solid** (physics
  critic: uniform surface incl. uniform head; mirror BC at solid faces → zero gradient → zero
  velocity → zero flux; clamps don't fire; W3 ratio ≡ 1.0 bit-exactly; inductive over ticks).
  **Open-edge pools spread by design** (wet cell's surface exceeds dry neighbor's by D) — never
  assert UNCHANGED for those; only glass/hull-bounded seeds get the byte-identity test.

## 2. Design (v2)

### 2.1 Format — the file IS the field (resolves B1 + M5 + m2)
```toml
[water]
depth_map = "water_init.npy"   # int32 Q16.16 metres, shape == tilemap (H, W); 0 = dry
```
- **No `max_depth_m`, no PNG.** v1's auto-scaling 8-bit PNG made edits non-local (deepening one
  pool re-quantized every other pool's golden-pinned ints — critique B1 arithmetic) and PNG
  decoding would add the runtime imaging dependency `level_loader.py:53-59` deliberately avoids
  (hand-parsed headers; critique M5 — an Erik-rule "silent architecture change").
- `.npy` via `np.load`: zero new deps (`levels/ship1_materials.npy` precedent), exact
  round-trip by identity, rounding pinned where the editor quantizes (`water_fixed.quantize` —
  tools can import it, it's pure numpy). Trade-off accepted on record: not hand-paintable in an
  image editor — the map editor is the author; FieldEdit remains the runtime write path.

### 2.2 Loader (door 2 satisfied trivially — values are ALREADY Q16.16 ints)
`level_loader.py`: `np.load` (`allow_pickle=False`), validate: shape == tilemap shape, dtype
int32, min >= 0 — `ValueError` with path-bearing f-strings in the existing style (m5); any
other type/shape hard-errors. → `LevelData.water_depth_q: np.ndarray | None = None` —
**explicit None default in the dataclass's defaulted tail** (synthetic `LevelData(...)` in
existing tests must keep constructing, physics critic m3). None when no `[water]` key — dormancy.

### 2.3 Seeding (placement verified — with one anti-pattern pinned)
`GameMap.__init__` right after the `_update_caches()` call: `mask = (~solid) & (~is_vacuum)`;
`water_depth[mask] = depth_q[mask]`, in-place. Depth on solid/vacuum cells in the file:
**count + warn once** (belt-and-braces — the editor masks at save, §2.4, so this fires only on
hand-authored files).
**Anti-pattern (physics critic m1): the seed lives in `__init__` ONLY — never inside
`_update_caches`**, despite the atmosphere precedent living there: `_update_caches` re-runs on
config hot-reload (with a snapshot/restore dance for atmosphere only), and a literal mirror
would re-flood a drained aquarium on Ctrl+R.

### 2.4 Editor WATER mode (rewritten per M1–M4, m3, m4)
- **Solidity predicate (M1, THE seam):** sim-exact, not manifest groups —
  `MaterialTable.from_config().permeability <= 0.0` (pure numpy + config.toml, no
  `breach_physics`; map_editor already imports `simulation.materials`). **SPACE_CODE=9 is not a
  material id — handle explicitly before fancy-indexing** (IndexError otherwise). Manifest
  `wall_family_codes` is art-connectivity data — equal today by coincidence, semantically wrong
  (a future opaque-but-permeable grill would silently diverge the fill boundary from the
  solver's mass-sink boundary).
- **Fill region (M2):** connected component over tiles that are `~solid_for_water AND
  ~SPACE` — vacuum bounds the fill exactly like glass does. Fill *started* on SPACE refused
  with a status message. (So a breached room floods up to the breach, never into space.)
- **Depth state:** parallel int32 Q16.16 grid (not float — quantize at fill time via
  `water_fixed.quantize(depth_m)`; UI displays metres via dequantize). Default 1.0 m, `-`/`=`
  steps of 0.1 m; RMB fill-to-dry.
- **Wall-over-pool (M3):** SAVE masks the water grid against the *current* material grid
  (zero on solid/SPACE) before writing, status line reports "N water tiles cleared under
  walls/space". Loader warn (§2.3) stays as the hand-authoring backstop.
- **Undo (M4):** a third mode-scoped ring (`SpawnRing` copy-generic precedent,
  `map_editor.py:489-506`) — Ctrl+Z pops the water ring only in WATER mode, per P3's recorded
  separate-rings design call (which stands; v1's "join the editor's ring" is overturned).
- **Ctrl+S integration (m3):** `[water]` block writeback shares the `toml_bak_written`
  session flag (one level.toml `.bak` = pre-session bytes, same contract); `water_init.npy`
  gets its own once-per-session pre-session `.bak`. Writeback slots after `write_spawns`,
  before `bake_level(..., write_bak=False)` — the ordering comment in the Ctrl+S block extends.
  (m4: only `[[spawn]]` is a managed block today; `[water]` is the second — lights land in P4.)

### 2.5 Tests
- Loader: shape/dtype/negative validation errors; no `[water]` key → None + all-zero field +
  **dormancy pinned** (existing dormancy trio already covers dry ticks — add the loader-level
  assert). Identity round-trip: editor-written file → loader → `gmap.water_depth` equality.
- Seeding: glass-box fixture — water strictly inside; warn-count on a hand-broken file.
- Conservation + at-rest: aquarium fixture level through the REAL loader path, 100 runner
  ticks, Σ depth conserved (template `test_runner_conserves_painted_water`); flat glass-bounded
  seed → depth field byte-unchanged (**pending the physics critic's verdict on edge-tile
  gradients — may become "conserved + settled" instead of "unchanged"**).
- Editor: fill bounded by glass AND by space; furniture does NOT bound (water flows past
  crates); refusal on SPACE start; wall-over-pool masking; water undo ring mode-scoping.
- **Skip convention (m1):** physics-bound tests use the module-level
  `try: import breach_physics / except ImportError: pytest.skip(...)` pattern of
  `tests/test_bedrock_cliff_counts.py:77-83` (the named templates hard-fail at collection —
  do NOT imitate them); orchestrator additionally runs them on the main checkout before merge.

### 2.6 Demo (the HUMAN-TEST) — unchanged
Glass aquarium (~4×3 interior, 1.2 m) in a testbed room: look (WaterPass renders it), shoot
the glass, watch it drain through the breach. Zero new mechanics.

## 3. Out of scope (accepted gaps, on record)
`floor_height` painting; oil/ice; editor water-in-vacuum (FieldEdit is the deliberate path);
sub-tile shorelines on tiled levels (needs height map). Goldens: levels WITH `[water]` change
theirs by design (door-2 note); levels without are bit-identical (dormancy test pins it).
Hand-paintable water carrier (PNG) — deliberately dropped, on record (§2.1).

## 4. Build order inside P5
1. Loader `[water]` + validation + tests → 2. GameMap seed + mask/warn + dormancy tests →
3. Conservation/at-rest tests (skip-pattern §2.5; orchestrator re-runs on main) →
4. Editor WATER mode (predicate, fill, ring, save masking, writeback) + tests →
5. Aquarium demo level + **chapter §2.3 amendment** + patch record (`docs/patch_levels_p5_water.md`).
