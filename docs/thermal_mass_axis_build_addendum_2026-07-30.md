# Build addendum — thermal-mass axis (Opus, 2026-07-30)

Companion to `docs/thermal_mass_axis_design_2026-07-25.md` (Fable, blessed by
Erik 2026-07-25). **The design's diagnosis and its fix are unchanged and
confirmed in code.** This addendum exists because the design's §1 "current
state (verified 2026-07-25)" and §2.1 "the material column" describe a
codebase that predates the July 8 `thermal_mass` work. Following §2.1
literally would break the design's own gate (a). This document is the stable
spec for the build; where it and §2.1 disagree, this document wins. §2.2–§2.5,
§3 (gates), and §4 (escalation triggers) stand as written.

Nothing here is a design change: every correction below is forced either by
what the code already is, or by gate (a) (furniture-free byte-identity).

---

## 1. Corrected current state (verified 2026-07-30 against `fire-o2-integration`)

**`thermal_mass` already exists.** It landed with the July 8 levels-P5-era
work (`97b3de8` is an ancestor of `main`), so it is on every live branch.
The design's §2.1 reads as if the column were new; it is not.

| Design §2.1 claim | Actual code (2026-07-30) |
|---|---|
| "add a per-material `thermal_mass`" | **Already exists**: `config.toml` per-material rows; `_SCALAR_COLUMNS` in `src/simulation/materials.py:70` |
| "the loader validates (power-of-two)" | **Already exists**: `materials.py:138-147`, raises unless power-of-two **and `>= 1`** |
| "the value IS the convert divisor (today's global 8)" | **Already per-tile, not global**: `temperature_solver.cpp:231-234` does `int shift = heat_inv_shift[i]; deposit >> shift` |
| "Derived grid ... rebuilt where `solid` is rebuilt" | `heat_inv_shift` per-tile grid **already exists** on that seam: built `gamemap.py:656`, patched `gamemap.py:806` |
| "Initial values: every currently-solid material = 8" | **FALSE — and load-bearing.** Actual: hull **32**, steel **32**, glass **16**, wood 8, door 8, door_closed 8 |
| "furniture = 8 (the change)" | furniture is **already 8** (`config.toml:812`). The value was never the bug |
| "Air = 0" | air is **1**, with the row comment *"1 avoids a 0-shift guard"* (`config.toml:715`) |

Confirmed as the design states:
- Medium test is `solid[i]` throughout the thermal pass — this **is** the bug.
- `solid = permeability <= 0.0` (`gamemap.py:672`; design said :758-762, drifted).
- furniture `permeability = 0.5` (`config.toml:808`) and **`conductivity = 0.0`**
  (`config.toml:811`) — so §2.2's "furniture κ stays 0, COOL_SHIFT is its only
  loss channel" holds exactly as designed.

### The real defect, restated

`thermal_mass` was already the per-material thermal identity, but **nothing
read it as a medium selector**. The medium branch kept asking `solid`
(= a *flow* property), so furniture's correct `thermal_mass = 8` was simply
never consulted — permeability 0.5 routed the crate into the gas regime.
The design's framing is right: flow was standing in for thermal identity, one
axis over. The column existed; the *routing* is what's missing.

---

## 2. Resolved decisions (forced, not chosen)

**D1 — Existing solid materials keep their CURRENT `thermal_mass`.**
hull/steel 32, glass 16, wood/door/door_closed 8. Do **not** set them to 8.
*Forced by gate (a):* those values are live tuned physics; flattening them to
8 would move every heat→T convert on metal and glass and blow byte-identity.
§2.1's "= 8" was written believing a single global 8 was in force.

**D2 — Air goes `thermal_mass = 1` → `0`; the loader learns that 0 is legal
and means "gas thermal regime".**
The blessed predicate is `thermal_mass > 0` (Erik's words: *"thermal_mass > 0
I like this"*), which is unsatisfiable while air is 1 — air would become a
thermal solid and the whole grid would take the solid regime. The current
validator (`materials.py:141`) rejects 0 outright, so it must change:
- `thermal_mass == 0` → legal; `heat_inv_shift` entry stores 0 (never read,
  because the mask routes those tiles away from the shift path).
- `thermal_mass >= 1` → power-of-two, exactly as today.
- The air row comment ("1 avoids a 0-shift guard") is now obsolete: the mask
  is the guard. Rewrite the comment to say so.

**D3 — `thermal_solid` is a derived grid, built in ONE function.**
`thermal_solid = (thermal_mass[material] > 0)`, per-tile bool, h×w, on the
same structural-rebuild seam as `solid` / `heat_inv_shift` (`gamemap.py:656`
build, `:806` patch). §2.4 requires a single seam so the future movable-object
version has one place to become dynamic. **Do not** derive it ad hoc at call
sites.

**D4 — Identity check that makes gate (a) well-posed.** furniture is the
*only* material with `permeability > 0` **and** `thermal_mass > 0`. Therefore
on any map with no furniture, `thermal_solid == solid` elementwise:
- air: `permeability` omitted → derives 1 → not solid; `thermal_mass` 0 → not thermal_solid ✓
- hull/wood/door/steel/glass/door_closed: permeability omitted → 0 → solid; `thermal_mass > 0` → thermal_solid ✓
- furniture: not solid, **is** thermal_solid ← the intended and only divergence

**D5 — Trigger 3 sweep result: no escalation, two spots folded into P1.**
The design says "grep first; the agent found none, but verify". Verified: no
consumer *outside* the temperature solver derives thermal behaviour from
`solid` in a way that changes. Two adjacent spots, both handled in P1 rather
than escalated (neither can break gate (a), since they are no-ops where
`thermal_solid == solid`):
- `gamemap.py:1521-1543` — the structural-change/evacuation path seeds a
  destroyed tile's temperature from "the integer mean of its PRE-call **solid**
  4-neighbours". Route to `thermal_solid` for consistency (a burning crate
  should be able to seed its neighbour's T); document the change in the patch.
- `src/simulation/entities/sensors.py:154` — comment asserts "Temperature
  lives on solids only". Now also furniture. Comment-only unless it masks.

---

## 3. Revised patch scope

The design's P1 shrinks substantially (the column, the validator, the per-tile
shift grid and the per-tile convert all already exist). What remains:

**P1 (CPU).**
1. `materials.py`: allow `thermal_mass == 0` (D2); keep power-of-two for `>= 1`.
2. `config.toml`: air `1` → `0` + comment rewrite. **All other rows untouched** (D1).
3. `gamemap.py`: `thermal_solid` derived grid on the existing seam (D3); the
   destroy-path seed re-route (D5).
4. `temperature_solver.cpp`: swap the **medium** test `solid` → `thermal_solid`
   at these six sites — and nowhere else:
   - `:165` Pass-0a vacuum/ambient gas-T zero (the `!solid[i]` guard)
   - `:187` Pass-0 advection open-air skip
   - `:42` `gas_wall_at` (advection ray-walk occluder — §2.3: "no longer
     advects T ACROSS the crate tile")
   - `:113` sealed-corner check in the advection gather
   - `:231` heat→T convert branch
   - `:368` COOL_SHIFT ambient decay
   Conduction face bake stays on the material κ tables — **unchanged** (§2.2).
   `solid` keeps its flow/LoS/N==0 meaning at every other site in the file.
5. `bindings.cpp`: plumb `thermal_solid` through the temperature entry points.
6. Unit tests: mask derivation; furniture-free identity; per-tile shift
   correctness; air-thermal_mass-0 loader acceptance; a non-power-of-two `>= 2`
   still raises.
7. Gates (a) and (c).

**P2 (CUDA).** Mirror the same six medium tests in `cuda_temperature.cu`
(`:65`, `:78` `gas_wall_at`, `:147`, `:173`, `:203`, `:272`) + the resident
path (one static mask upload, sponge-grid precedent). Gate (d), tol 0, step
**and** resident, on a furniture-burn scenario.
*Trigger 2 pre-checked 2026-07-30: the CUDA kernels DO mirror the CPU
medium-branch structure line-for-line, so this is a mask swap. Trigger 2 does
not fire.*

**P3 (close).** Bench report (gate c numbers + the §2.5 analytic check);
`tools/fire_tune_loop.py` TUNE defaults; rewrite `fire_tuning_plan` §9.5 from
"open question" → "regression, fixed"; hand back to Erik's manual loop.

## 4. Build-side notes

- Machine: Lenovo (`ERIK_LENOVO`), RTX 1000 Ada. Build via
  `cpp/build_cuda_lenovo.bat`. Python is the conda env `data`; tests are
  `pytest tests -q`.
- Base: branch `thermal-mass-axis` off `fire-o2-integration` @ `423cd38`,
  in its own worktree. Erik's `fire-o2-integration` worktree holds live
  uncommitted tuning work and is not to be touched.
- **HUMAN-TEST gate.** This changes fire behaviour → feel-adjacent → per
  project CLAUDE.md it never auto-merges. Build, gate, push; Erik plays and
  runs his manual tuning loop before any merge. Gate (b) stands: furniture
  goldens may move, they are enumerated in the build report, and **no rebase**
  happens here — it rides the joint re-tune's one deliberate rebase.
