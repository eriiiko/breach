# The thermal-mass axis — furniture burns like an object (design, 2026-07-25, Fable)

**Blessed by Erik 2026-07-25** ("thermal_mass > 0 I like this … let's do this").
Context: fire-tuning plan §9.5 — the 2026-07-25 investigation showed a burning
crate's temperature is WINDY GAS (advected away by the fire's own plume), so
the hot-gate value lost physical meaning. This doc reclassifies that finding
and fixes it.

**This is a REGRESSION FIX, not a new mechanism.** The canon's own words:
- `docs/architecture/engine/06_temperature_and_fire.md:77-91` (original
  design): "**Temperature lives on solids only** … Air temperature would
  ignite nothing and **advect everything — the wrong** [behavior]."
- `06:31-49` (EOS-era amendment): ONE unified T field over gas + solids, run
  as **masked per-medium passes** (gas: advection + compression-work, no
  ambient decay; solid: convert / conduct / cool).
- The migration keyed the medium on `solid` = `permeability <= 0`
  (`02_state_and_ownership.md:147`). Furniture (permeability 0.5, the
  deliberate "shield but not seal" soft body) silently fell into the GAS
  regime. Nobody chose that; a mask definition did.
- The repo already names this error class: per-axis representation
  (`02:107` — movement and flow are separate axes) and the retired `is_wall`
  "light-occlusion accident" (`01:234-236`). Flow was standing in for thermal
  identity, one axis over.

**The fix: add the missing axis.** A per-material `thermal_mass`; the
per-medium thermal masks route on `thermal_mass > 0` instead of on `solid`.
Furniture joins walls in the solid thermal regime — object temperature,
COOL_SHIFT, no advection — while `permeability = 0.5` is UNTOUCHED (gas and
water still seep past; shield-but-not-seal stands).

---

## 1. Current state (file:line, verified 2026-07-25 by the read-only agent)

- Medium branch: ambient-cooling pass is solid-only
  (`temperature_solver.cpp:368` `if (!solid[i]) continue;` + `:384-390`);
  gas cells get Pass-0 wind advection (`:172-194`) and the gas heat→T
  conversion (`:265-275`); solid cells get the `heat >> 3` convert
  (`:231-234`, the global thermal_mass 8) and conduction (`:313`).
- `solid = (permeability == 0)` (`gamemap.py:758-762`).
- `[materials.furniture] permeability = 0.5` (config.toml:808), κ = 0
  (config.toml:811 → all faces NO_FACE, no conduction in or out).
- Combustion burns in the OPEN 4-NEIGHBOURS of the fueled tile
  (`combustion.h:19-24`); H_fuel heats those air cells — that (plus heat
  rays) is what births the plume, independent of the crate's own T.
- Measured consequence: seeded crate T 280 → ~90–110 "gas shelf" in seconds
  (~21 game/tick advection/decompression loss); no cooling dial governs it.

## 2. Design

### 2.1 The material column

`thermal_mass` (per material row, config.toml `[materials.*]`):
- **0** → gas thermal regime (unchanged): air, and any future gas-like row.
- **> 0** → solid thermal regime; the value IS the convert divisor (today's
  global 8). **Power-of-two only** (the convert is a free bit-shift; the
  loader validates, like other Q16-friendly dials).
- Initial values: every currently-solid material (hull, wood, door, steel,
  glass, door_closed) = **8** (today's behavior, bit-identical); **furniture
  = 8** (the change). Air = 0.
- Derived grid `thermal_solid` (bool, h×w) from the material map; rebuilt on
  structural change exactly where `solid` is rebuilt. NOTE: `is_ambient` /
  structural caches already have this rebuild seam — reuse it.

### 2.2 Solver re-route (the whole fix)

In every per-medium branch of the unified thermal pass, replace the medium
test `solid[i]` with `thermal_solid[i]`:
- Pass-0 advection + compression-work: SKIP thermal_solid tiles.
- Heat→T convert: thermal_solid tiles take the shift path with the PER-TILE
  divisor (`heat >> log2(thermal_mass[mat])` — same instruction, per-tile
  shift count); gas tiles unchanged.
- COOL_SHIFT ambient decay: applies to thermal_solid tiles.
- Conduction: face bake unchanged (κ still per-material; furniture κ stays
  0 this build → crate exchanges nothing by conduction; its ONLY loss is
  COOL_SHIFT — one clean channel for Erik's tuning). Raising furniture κ
  later is a realism dial, not this patch.
- Everything that is NOT the medium test is untouched: `solid` keeps its
  flow/LoS/N==0 meanings everywhere else.

### 2.3 EOS / pressure coupling (decided, with a tripwire)

Furniture tiles hold gas (N > 0) and the EOS reads P = C·N·T[i] with the
unified T — which on a burning crate is now OBJECT temperature. Accepted:
gas in the pores of a 1300 K object is hot; the resulting overpressure/plume
from the crate tile is desirable fire behavior. The gas-medium pass no
longer advects T ACROSS the crate tile — heat crosses via the surrounding
air cells (which combustion + rays heat directly, §1). **Escalation trigger
if the bench shows P pathology** (e.g. runaway pressure spikes or wind
artifacts pinned to the crate): fall back to `P = C·N·t_amb`-style neutral
pore-gas on thermal_solid tiles — a one-line branch — and bring the choice
back to Erik/Fable.

### 2.4 What deliberately does NOT change

- `permeability` / `dyn_permeability` / `solid` / mobility / LoS: untouched.
- Units: NOT in the thermal_solid mask (their soft-body dyn_permeability
  never touches material rows). **Forward note (Erik 2026-07-25): units
  should eventually be IGNITABLE with armor-dependent ignition temps — that
  belongs in the unit environment system (EnvironmentProfile gains
  ignition/flame state, reads local heat flux), a mechanics design of its
  own. Queued in the tuning plan; NOT this build.**
- Movable-objects future: when furniture becomes movable entities, the
  thermal_solid mask goes dynamic (the dyn_permeability precedent) and the
  object's T is carried conservatively on move (the gas-evacuation
  precedent, `gamemap.py:1373`) — eventually entity-carried T. Nothing in
  this patch blocks that path; the mask build must live in ONE function so
  the dynamic version has a single seam.

### 2.5 Expected new operating point (for the tuning loop, §9.3)

With convert ÷8 and COOL_SHIFT=5 active on the crate: steady
`T* ≈ (k_fire_heat·I/8)·2^5 = 4·k_fire_heat·I`. For T* = 450 game @ I=0.5 →
**k_fire_heat ≈ 225** (vs 12 on the artifact substrate). COOL_SHIFT is now a
real dial for the crate (integer; 5 → 1.3 s e-fold is FAST for a wood
object — Erik may prefer 6–7). `fire_T_ext` returns to PHYSICAL values
(~250–350 game; the §9.2 below-ignition rule still applies). The warm-seed
harness fix stays valid — and the t≈0 temperature DIP must be GONE (gate c).

## 3. Gates

a. **Furniture-free byte-identity (the strong one):** on any map with no
   thermal_mass>0-but-permeable tiles, `thermal_solid == solid` and every
   path must be BYTE-IDENTICAL — space maps, sealed rooms, the whole
   existing golden suite except furniture-bearing scenarios. Zero tolerance.
b. **Digest movement enumerated:** only furniture-bearing goldens move;
   list them in the build report; NO rebase (rides the joint re-tune's ONE
   deliberate rebase).
c. **Bench physics:** warm seed (T=280) on the crate — T rises MONOTONICALLY
   from 280 while I grows (no dip); fire sustains at physical
   `fire_T_ext ≈ 250–350`; T* tracks the §2.5 analytic within ~20%.
d. **CPU↔CUDA lockstep tol-0**, step AND resident paths (the mask is one
   static upload, like the sponge grids), on a furniture-burn scenario.
e. **Conservation/sealed-room + sky-exchange gates stay green** (the mask
   touches no gas plane).

## 4. Patch plan (Opus; on the `fire-o2-integration` line, stacked)

- **P1 (CPU):** material column + loader validation (power-of-two) +
  `thermal_solid` grid + solver re-route + unit tests (mask derivation;
  furniture-free identity; per-tile shift correctness) + gates a/c.
- **P2 (CUDA):** mirror the medium test in the CUDA thermal kernels + the
  resident path (one static mask upload) + gate d.
- **P3 (close):** bench verification report (gate c numbers + the §2.5
  analytic check); update `tools/fire_tune_loop.py` TUNE defaults
  (fire_T_ext≈300, span≈100, k_fire_heat≈225, COOL_SHIFT dial exposed);
  rewrite fire_tuning_plan §9.5 from "open question" → "regression, fixed
  (this doc)"; hand back to Erik's manual loop.

**Escalation triggers (stop, back to Fable/Erik):**
1. The §2.3 pressure tripwire.
2. The CUDA thermal kernels turn out NOT to mirror the CPU medium-branch
   structure (i.e., the re-route isn't a mask swap there).
3. Any consumer found deriving thermal behavior from `solid` OUTSIDE the
   temperature solver (grep first; the agent found none, but verify).
4. Water/steam interactions with furniture tiles behaving differently under
   the solid regime (boil/quench paths).
5. Anything tempting a change to `permeability`, `solid`, or unit stamping —
   out of scope, full stop.
