# Fire session #12 — the radiation session opener: every assumption that pertains to radiation (2026-09-11)

> **Status:** design-session input, not a ruling. Erik asked (2026-09-11) that the
> radiation discussion start by listing every assumption in the simulation that
> pertains to radiation, then "look hard at what we have", THEN decide the model
> and whether the ray engine needs a full or partial redesign. This is that list.
> Every claim carries a file:line citation into the tree at fire-12 `d7cbcb1`
> (main merged in). Line numbers were verified on 2026-09-11.
>
> Supersedes §6 of `docs/fire_3c_r4_tuning_and_radiation_2026-09-06.md` where
> they disagree — in particular the "~14× overestimate" framing there is only
> part of the story (§2 below).

## 1. The short version — six load-bearing assumptions and the chain they form

1. **Heat travels along 8 rays per emitter, evenly spaced, and a ray loses
   nothing to distance and nothing to air.** Only material occlusion reduces it
   (`heat_survival *= 1 − heat_atten[cell]`), air and vacuum have
   `heat_atten = 0`, and the ray runs to `RADIATION_RANGE = 320` tiles, which is
   deliberately ≥ every map's diagonal, so in clear air a heat ray always
   reaches a wall or the world edge. The 1/r falloff exists only as ray DENSITY.
   (`cpp/src/raycaster.cpp:674`, `:545`, `:691`; `raycaster.h:444-466`;
   `config.toml:569-570`.)
2. **Each hit deposits a full, distance-independent quantum.** The pair term is
   `x = a_s · a_r · τ · w · (E°[T_s] − E°[T_r])` with `w = 1/8`; nothing in it
   knows how far the ray has travelled. A tile 14 tiles away receives the same
   energy per hit as a tile 1 tile away — it is just hit ~14× less often.
   (`raycaster.cpp:596-614`; `raycaster.h:101-114`.)
3. **Emission goes as T⁴ with no ceiling that matters.** `E° = rad_scale·K⁴`
   from an int64 table, `K = 293 + T_game`. Radiated power therefore rises
   27× from the ×2 plateau (683 K) to the ×8 transient (1556 K).
   (`raycaster.cpp:62-97`; `config.toml:526`, `:813-814`.)
4. **A receiver below the emitter gate has NO radiative loss channel.** Only
   `burning ∪ (thermal_solid && T ≥ T_emit_gate = 930 game)` casts rays, and
   casting is the only way a tile loses heat radiatively. A crate at 500 K
   gains from the fire and radiates nothing back. The config comment says this
   out loud: "RECEIVERS ARE FREE … This gate only decides who can radiatively
   LOSE heat." (`raycaster.cpp:917-924`; `config.toml:527-542`.)
5. **Furniture-class solids have no other loss either.** `conductivity = 0`
   (no conduction face at all, `NO_FACE`), so the ONLY loss is the ambient
   relaxation `T -= T >> cool_shift` with `cool_shift = 13`: e-fold 341 s.
   (`config.toml:1549-1553`; `temperature_solver.cpp:643-644`.)
6. **Spread IS radiation.** There is no cellular spread term, no flame-contact
   channel, no convection to solids: a neighbour ignites only when its own
   temperature crosses `ignition_temp` (edge-triggered, plus an O2 and a fuel
   gate). Contact faces between solids are radiation-inert by rule 3
   ("conduction owns contact") — and for furniture conduction is off.
   (`fire_simulation.cpp:177`; `combat.py:463-477`; `raycaster.cpp:586`.)

**The chain:** (1)+(2) make the far field fall only as 1/r; (3) makes the far
field 27× stronger at the ×8 transient than at ×2; (4)+(5) mean a receiver keeps
essentially everything it gets (its only loss is a 341 s e-fold); and (6) means
that radiation is doing the job flame contact does in reality, so it MUST be
strong at 1 tile — which, under 1/r with free receivers, makes it far too
strong at 14. The 1/r-vs-1/r² question is real but it is one link, not the
chain.

## 2. Reconstruction: the 14-tile flashover from the constants alone

The night-of-09-06 measurement (§4 of the handoff doc): a crate at 1262.7 game
(1556 K) ignited station 1, 14 tiles away through 287–296 K air, in 7.5 s; the
receiver read 261 game (554 K) at t = 7 s.

From the model constants (estimate; each step cites the rule it uses):

| Step | Value | Basis |
|---|---|---|
| E°[1556 K] = 5.1427e-5 · 1556⁴ | ≈ 3.0e8 counts | int64 bake, `raycaster.cpp:62-97` |
| per-hit x = a_s·a_r·w·ΔE° = 0.5·0.5·(1/8)·3.0e8 | ≈ 9.4e6 counts | rule 1, furniture a = 0.5 |
| ΔT per hit = x >> heat_inv_shift (3) | ≈ 18 game units | fold, `temperature_solver.cpp:255` |
| flux limiter cap (≈ half the pair gap in counts) | ≈ 3.3e7 ≫ 9.4e6 | `RAD_LIM_SHIFT = 4`, not binding |
| hits per tick at r = 14: 8 / (2π·14) | ≈ 0.09 (one per ~11 ticks) | 8 rays, rotating fan, `raycaster.cpp:943-950` |
| heating rate = 0.09 · 18 · 24 ticks/s | ≈ 39 game/s | |
| loss at 261 game: 261 / 8192 per tick · 24 | ≈ 0.8 game/s | `cool_shift = 13` |
| **time to 280 game** | **≈ 7 s** | **measured: 7.5 s** |

The same arithmetic gives the receiver's equilibrium under a sustained 1556 K
emitter: gain = loss when `T_eq / 8192 = 1.6` game/tick → **T_eq ≈ 13,000
game**, fifty times the ignition temperature. And the ignition RADIUS: the
near-field rate is ~550 game/s at r = 1, falling as 1/r, while the receiver can
only shed ~0.8 game/s at ignition, so gain exceeds loss out to r ≈ 670 tiles —
beyond `RADIATION_RANGE`. **Under the current assumptions every furniture tile
in line of sight of a 1556 K crate ignites eventually; only the time differs,
roughly 0.5 s per tile of distance** (7 s at 14, ~50 s at 100). With a per-ray
1/r added (1/r² total) the radius drops to ~26 tiles and the 14-tile time to
~100 s — better, still a flashover in gameplay terms. The geometric law alone
cannot produce a horizon while receivers are loss-free.

## 3. The inventory (verified citations)

### A. Geometry and sampling
- A1. Radiation is its own march (`march_ray_radiation`), separate from the
  visible-light march since P-F1a (2026-08-01); both walk the same DDA tiles
  from the same origin/angles. In the live sim the visible-light cast is
  skipped entirely (`physics_runner.py:1624-1626`); fire light is the
  renderer's blackbody selector. (`raycaster.cpp:1042-1079`, `:786-797`.)
- A2. `fire_ray_count = 8`, fixed, no jitter (`config.toml:570`;
  `raycaster.cpp:956`; `raycaster.h:658-660`).
- A3. Full circle, evenly spaced; the fan's phase is a spatial hash of the
  emitter's (col,row) PLUS a per-tick rotation of 2π/N² — so over 8 ticks the 8
  rays cover 64 directions and "every pair is connected within N ticks"
  (`raycaster.cpp:943-950`, `:1055-1057`; rationale `raycaster.h:892-904`).
  Consequence: far-field heating arrives as discrete ~18-game quanta every few
  ticks, not as a smooth flux (engine/08 admits this as "sampling noise").
- A4. No per-distance term, no air/gas absorption, no `exp` on the path;
  survival only by `(1 − heat_atten)` per tile, skipped on the source tile
  (`raycaster.cpp:670-674`, `:528-533`).
- A5. Vacuum is not special for radiation: no `is_vacuum` test in the march;
  a vacuum tile is its material's `heat_atten` (air → 0). (Absence verified
  over `raycaster.cpp:505-699`.)
- A6. Termination: survival ≤ `heat_cull` (0.01, a C++ default, NOT a config
  key — `raycaster.h:527`), world edge (charges the sky term), contact face
  (rule 3), or `distance > RADIATION_RANGE` (unreachable on shipping levels).
- A7. `range_base`/`range_per_intensity` no longer bound a heat ray; they only
  gate the `rad_flux` unit-damage plane (`raycaster.cpp:941`, `:984`;
  `config.toml:571-577`). A warm non-burning emitter has I = 0 there.

### B. Emitters, receivers, Kirchhoff
- B1. Emitters = `burning ∪ (thermal_solid && T ≥ quantize(T_emit_gate))`,
  materialised once per tick as an integer mask plane (`raycaster.cpp:897-924`;
  `raycaster.h:92-95`). `T_emit_gate = 930` game = 1223 K, the G12 rescale of
  the P-F1b value; the history in `config.toml:527-541` records that at 180
  game "a receiver became an emitter too early and its ceiling collapsed …
  stalling spread" — i.e. the gate was RAISED precisely so receivers stay
  loss-free. That is the same switch as assumption 4.
- B2. Receivers are not gated; a cold tile absorbs whatever reaches it.
- B3. Kirchhoff: emissivity == absorptivity == `heat_atten`. Air `a = 0`
  neither emits nor absorbs (`raycaster.h:405-406`). An emitter with `a_s = 0`
  is dropped before any ray is cast (`raycaster.cpp:931-932`).
- B4. **Foliage has `heat_atten = 0.0` and `conductivity = 0.0`**
  (`config.toml:1640-1641`, from the #60 P3 row, comment "no interaction beyond
  burning"). By B3 a burning tree radiates nothing, a cold tree can be heated
  by nothing, and with κ = 0 it conducts nothing: it is thermally isolated. It
  can be lit only by a direct temperature write (dev key, payload) and can
  never spread fire. Needs a ruling (§7).
- B5. Material table, the radiation-relevant columns (`config.toml`):

| material | heat_atten (= ε) | conductivity | thermal_mass (shift) | cool_shift | ignition_temp |
|---|---|---|---|---|---|
| air | 0.0 | 0.024 | gas | — | — |
| hull | 1.0 | 50.0 | 32 (5) | 5 | — |
| wood | 1.0 | 0.15 | 8 (3) | 13 | 300 |
| door / door_closed | 1.0 | 0.3 | 8 (3) | 5 | 280 (not flammable) |
| steel | 1.0 | 45.0 | 32 (5) | 5 | — |
| glass | 0.3 | 1.0 | 16 (4) | 5 | — |
| furniture | 0.5 | 0.0 | 8 (3) | 13 | 280 |
| kindling | 0.5 | 0.0 | 8 (3) | 13 | 280 |
| foliage | **0.0** | 0.0 | 8 (3) | 13 | 280 |

### C. The energy law and its books
- C1. Net antisymmetric exchange, quantized once, applied ± as the same
  integer: `rad_net[r] += x; rad_net[s] −= x`. Two equal-T tiles exchange
  exactly 0 (same 4-game bucket ⇒ diff == 0). The emitter loses exactly what
  the receiver gains; there is no separate Stefan–Boltzmann loss term anywhere.
  (`raycaster.cpp:639-640`; `raycaster.h:140-148`.)
- C2. Rule 2: if the receiver is itself an emitter, the same term at half
  weight, so a mutual pair sums to exactly 1× (`raycaster.cpp:601`, `:612`).
- C3. Rule 3: a ray stepping solid→face-adjacent-solid terminates with no
  deposit and no charge (`raycaster.cpp:586`). The residual is charged to
  nobody (a named, accepted leak).
- C4. Rule 4: leaving the grid is the only escape; the emitter is charged
  `sky = a_s·τ_end·w·(E°[T_s] − E°[0])`, booked into `rad_amb`
  (`raycaster.cpp:552-568`). The ledger identity `Σ rad_net + Σ rad_amb == 0`
  is asserted in tests only, not in the shipped step
  (`tests/test_pf1a_radiation_books.py`).
- C5. Every pair term is flux-limited per ray per tick (`RAD_LIM_SHIFT = 4`,
  compile-time, `raycaster.h:226`) — ≈ 1/16 of the pair gap.
- C6. The E° table: int64, 4000 buckets × 4 game units, `K⁴` by repeated
  integer multiplication (never `pow`), no interpolation, saturates on the top
  bucket (`raycaster.h:189-215`; `raycaster.cpp:62-97`).
- C7. Ray heat lands in `rad_net`, folded into SOLIDS ONLY in temperature Pass
  1 as `ΔT = shr_round0(rad_net, heat_inv_shift)` (`temperature_solver.cpp:
  242-262`). Gas never receives ray radiation. There is no `e_rad` counter;
  radiation is inside `e_solid_deposit_sum` together with the `heat[]` deposit
  (`temperature_solver.cpp:297-298`).
- C8. `rad_flux` (unit damage) is a separate positive-only plane written at
  air cells within `damage_range`, explicitly outside the books
  (`raycaster.cpp:641-666`; consumer `exchange.py:299-340`).
- C9. Ordering: cast first, temperature solver consumes `rad_net` in the same
  tick; same on the resident path (`physics_runner.py:819`, `:977`, `:1207`,
  `:1372`). CUDA twin is a line-for-line copy of the march
  (`cuda_raycaster.cu:185-290`).

### D. Loss channels of a hot solid (what a receiver can shed)
- D1. Radiative: only as an emitter (B1), via C1 pairs and the C4 sky term.
- D2. Conduction (Pass 2): energy-form face flux with `face_shift` baked from
  the harmonic-mean conductivity; `NO_FACE` when κ = 0 on either side
  (`temperature_solver.cpp:405-462`; `config.toml:1405-1416`).
- D3. Ambient relaxation (Pass 3): `T -= T >> cool_shift`, thermal solids
  only, two-way (a sub-ambient tile is warmed), vacuum-exposed tiles use a
  faster shift (`temperature_solver.cpp:603-651`).
- D4. The fuel-bed deposit `H_bed` (`combustion.cpp:740-745`) is a GAIN on the
  burning tile itself, proportional to the O2 it actually got; `H_BED_M =
  18125`, `H_BED_SHIFT = 7` ⇒ 2.32e6 (the ×8 under review).

### E. Dials and constants that touch radiation
`rad_scale = 5.1427e-5` (`config.toml:526`, the T = 300-game flux anchor);
`T_emit_gate = 930` (`:542`); `RADIATION_RANGE = 320`, floor 287 (`:569`;
`raycaster.h:465`); `fire_ray_count = 8` (`:570`); `heat_cull = 0.01`
(C++ only); `RAD_LIM_SHIFT = 4` (C++ only); `kelvin_ambient = 293`,
`k_temp_to_kelvin = 1` (`:813-814`); per-material `heat_atten`, `conductivity`,
`thermal_mass`, `cool_shift` (B5). Tombstones: `k_fire_heat` (P-R4, the
painter), `dist_atten` (§4), `coarse_cluster`, `p_expand_ref`.

### F. Ignition and spread
- F1. Ignition: `armed && T ≥ ignition_temp[mat] && X > o2_frac_ext && wall_hp
  > 0 && fire == 0 → fire = ignition_seed (0.12)`, edge-triggered, re-armed only
  after cooling below threshold (`combat.py:463-488`, `:557-566`).
- F2. Sustain is a separate signed logistic on an already-lit tile, gated by
  `fire_T_ext = ignition_temp − Δ(200)`, O2 fraction, fuel and wind
  (`fire_simulation.cpp:181-260`).
- F3. R3 (2026-09-06): O2 DEMAND and fuel DESTRUCTION scale with `hotf =
  clamp((T − T_ext)/span, 0, 10)`, so **a fire on a cold tile draws no O2 and
  burns no fuel** — "a fire cannot bootstrap from cold; heat must come from
  outside". This is why the dev key needed heat, and why the props arc's
  `test_foliage_tile_burns` (seeds `fire` on an ambient tile, expects `wall_hp`
  to drop) fails on the merged tree (§7).

## 4. History: how we got to "1/r, no distance term"

Erik's recollection (2026-09-11): "the radiation was following the 1/r² rule
to start with, and I changed it because I thought in the plane it will decrease
with 1/r." The record (git archaeology, 2026-09-11):

- The per-ray factor was born in the design doc of 2026-03-16
  (`docs/archive/implementation_plan_radiation_temperature.md:319-321`) as
  `dist_atten = 1/(1 + 0.01·d²)` — a SOFTENED inverse square: ≈ 1 out to a few
  tiles, 1/2 at 10 tiles, 1/3 at 14, ~100/d² far out. It went into `game.py`
  the same day (`1b10a2e`), into the C++ port on 03-17 (`033497d`), and rode
  every version until 2026-06-28, when `e3d2283` ("CUDA-S2 step 1: re-derive
  the CPU raycaster march to the pure-density model") dropped it. Never
  retuned; one constant for 3.5 months.
- The rationale is in `docs/architecture/engine/08_ray_engine.md:205-212`:
  the per-ray factor "stacked a second falloff on top of the density falloff
  (~1/r³ total — far too steep) and was a band-aid for the source-cell
  pile-up"; the N cancels; brightness falls as 1/r, "the faithful 2D law".
  The admitted cost: sampling noise where rays separate past ~1 tile.
- So the choice was deliberate, argued, and CORRECT for light in a 2D world.
  Two things it did not consider: heat was then still the one-way painter
  (`k_fire_heat·I`), so the T⁴ exchange that now rides these rays did not
  exist; and receivers had losses back then only through the same weak
  `cool_shift`. Nobody measured spread until 2026-09-06.
- Honest note on the "14×": the OLD softened factor would have cut the 14-tile
  flux by only ~3× (1/(1+1.96)), not 14× — under today's T⁴ law the flashover
  would have happened with it too, ~20 s instead of 7.5. The old formula is
  not the thing to restore.
- The legacy scalar `march_ray` (flashlights' `cast_source`) STILL carries
  `dist_atten` at HEAD (`raycaster.cpp:134-138`) — not on the heat path.

## 5. What each candidate lever actually does

"Reach" below = the distance at which a receiver's equilibrium temperature
under a sustained emitter equals its ignition temperature; "t₁₄" = time for a
furniture tile 14 tiles from a 1556 K crate to reach ignition (7.5 s today).

| Lever | Far field | Near field / plateau | t₁₄ | Horizon? | Cost / determinism |
|---|---|---|---|---|---|
| **rad_scale ÷ k** (Erik's "10× weaker") | ÷ k | the burning tile's own loss is 99.6 % radiative (R2), so its plateau RISES ~k^(1/4) unless H_bed is cut too; at a held plateau the far field is ÷ k | × k | no — reach ∝ 1/k in 2D, shape unchanged; k = 10 → 75 s, still a flashover | free (re-bake) |
| **per-ray 1/r** (→ 1/r² total, "3D slice") | ÷ r | unchanged at r ≈ 1 | × 14 (~100 s) | reach ~26 tiles at 1556 K (÷√ of the 2D reach) | one float divide per step (exact, pinned) |
| **air transmittance s < 1 per tile** (Beer–Lambert without `exp`) | × s^r | unchanged nearby | any | YES, an e-fold length L = −1/ln s; s = 0.8 → L = 4.5 tiles | trivial in the march; BUT where does the absorbed energy go? (gas seam deposit, physical and booked; or a counted export) — and by Kirchhoff absorbing air also emits at its own T |
| **receiver losses** (lower `T_emit_gate`, or all solids grey-body) | unchanged flux, but the receiver now radiates back | reach at 1556 K ≈ 4 tiles by my estimate; at the 917 K plateau adjacent spread STALLS — the effect the gate was raised to avoid | — | YES, physical | emitter count ↑ (cost lever); forces the question in §6 (a contact channel) |
| **hard range cutoff** | zero beyond R | unchanged | ∞ if R < 14 | crude yes | must book the cut residual (sky-like counter) or it becomes the corridor leak again |
| **ray count N** | same mean flux (N cancels) | same | same | no | only granularity: 18-game quanta at N = 8, 4.5 at N = 32; ×N marches |
| **plateau T via H_bed** | ∝ T⁴ | the thing we tune for G1 | ∝ 1/T⁴ | no | none — but reach ∝ T⁴ in 2D, ∝ T² in 3D: G1's 1300 K target radiates 13× a 683 K fire |

Two facts follow from the table and should be on the table before any pick:

- **Reach is set by emitter power and the geometric law together.** In a 2D
  world with free receivers the ignition radius scales with the fourth power of
  the emitter temperature. Any model that keeps free receivers and 1/r will
  hand us a new flashover every time the plateau moves.
- **The near field and the far field are coupled through one channel.**
  Radiation has to be strong at 1 tile because it is the ONLY spread
  mechanism (assumption 6). The far field is then whatever the geometric law
  makes it. Decoupling them — a short-range flame-contact/adjacency channel
  for spread, and a radiation law with real reach for everything else — is the
  one option that lets both be right. It is also the most work.

## 6. Constraints any change must respect (so the session designs inside them)

- Integer/pinned determinism: no `exp`/`pow`/libm on the path (`E°` is a baked
  int64 table; per-step multiplies are fine; a float divide is exact if the
  operands are pinned; `--fmad=false` on CUDA). The CUDA twin
  (`cuda_raycaster.cu`) must stay line-for-line.
- The books: every energy that leaves an emitter must land somewhere counted
  (a receiver, `rad_amb`, or a new named counter). "Air absorbs" is either a
  gas-seam deposit (then the gas energy ledger, arc #54, gains a writer AND a
  term in one of its four groups) or a counted export. Not a silent decay.
- `RADIATION_RANGE ≥ diagonal` exists to make "escape" mean "left the world";
  any horizon shorter than that must be an explicit counted term, not a
  range.
- The golden: the canonical scenario's fire sits on non-flammable tiles and,
  since R4, the golden is independent of the fire dials; a radiation-law
  change WILL move it (hot solids radiate to the walls in that scenario) — one
  re-baseline, with rationale, when the model is ruled.
- Feel gate: reach is feel. HUMAN-TEST before merge.

## 7. Side findings from today (need rulings, none blocks the discussion)

- **Foliage is thermally isolated** (B4). Recommendation: `heat_atten = 0.5`
  like furniture/kindling (same class, same cool_shift), so a tree can be lit
  by a fire and burn its neighbours. One config line + the #60 design doc §4.1
  + `tests/test_prop_entity.py` if it pins the row. Erik's call — the P3 spec
  said "no interaction beyond burning" and being lit by a fire IS burning.
- **`test_foliage_tile_burns` (from #60) fails on the merged tree** because
  it seeds `fire` on an ambient tile and R3 makes such a fire consume nothing
  (F3). Its premise predates R3; the fix is in the test: seed the tile's
  temperature at ignition + margin (thermal_solid-only write, the gas-mirror
  rule), exactly as the dev key now does. The property it protects ("a burning
  foliage tile consumes its fuel") is still the right one.
- **Twelve stale comments** found by the inventory, to be fixed in the
  radiation patch, not before: `physics_runner.cast_fire_heat` docstring still
  describes the painter; `raycaster.h:55-56` "nothing reads heat yet";
  `raycaster.h:172-175` and `:163-165` assume `k_temp_to_kelvin = 3`;
  `raycaster.h:427`/`:443` quote pre-G12 dial values; `raycaster.h:391-393`
  claims a host reduction of `rad_amb` that does not exist; `config.toml:
  437-466` (the K2 block) describes the dead painter and dead cellular spread;
  `combat.py:522-525` calls the only ignition path "dormant"; `docs/
  architecture/engine/08_ray_engine.md` documents the painter as current and
  says main ships the old model; `combustion.h:426` and `materials.py:169-170`
  carry stale H_bed defaults; `temperature_solver.h:341` calls `c_v` the
  "radiation-deposit divide" (it divides the combustion deposit);
  `raycaster.h:595-605` documents an unreachable fire heat channel.
- The `fire_tuning` level's premise (stations ≥ 6 tiles apart are
  independent) is false under the current law; every multi-station
  measurement there is contaminated until reach is fixed.

## 8. The decisions the session has to make (Erik)

1. **Which universe for heat?** 2D-faithful (1/r, "live in the plane") or a 2D
   slice of 3D (1/r²)? Light and heat need not share the answer; they can share
   rays and carry different per-ray laws.
2. **Does clear air absorb heat?** If yes: gas-seam deposit (air warms near a
   fire) or counted export? And should smoke/soot absorb more (a radiant
   shield — the physical truth and a gameplay hook)?
3. **Do receivers radiate?** Keep the emitter gate as the switch (cheap,
   unphysical, free receivers), lower it toward Erik's original 653 K feel
   value, or make every warm solid a grey body (physical; needs §5's contact
   channel or spread stalls)?
4. **What reach do we WANT?** A design number, not a measurement: "a burning
   crate lights an adjacent crate in ~X s, one 3 tiles away in ~Y s, one 10
   tiles away never" — the reach curve is the object we fit to.
5. **Is a hard cutoff acceptable** (Erik: "not that bothered by the 1/r
   cutoff") as the interim, if it is booked?
6. **Spread channel**: is radiation allowed to keep doing flame contact's job,
   or do we add a short-range adjacency channel and let radiation be physical?
7. **Foliage `heat_atten`** (§7).

## 9. Proposed first instrument (after the assumptions are agreed)

The **reach-curve bench**: one emitter held at a fixed temperature (or a real
burning crate at ×2 and ×8), furniture probes at r = 1, 2, 3, 5, 8, 14, 30,
each logging its temperature and ignition time; one CSV per lever setting
(`rad_scale` ×1/×0.1, per-ray 1/r on/off, air transmittance 1.0/0.9/0.8,
receiver gate 930/360/all). The lab's `DIALS` seam already overrides config;
`tools/fire_tuning_lab.py` is the base to extend (single-tile today). The curve
is what §8 item 4 gets fitted against — and it is also the regression gate the
eventual patch ships with.

## Systems

- Existing canonical systems this session must use: the ray engine
  (`cpp/src/raycaster.*` + `cuda_raycaster.cu`, the only radiation path); the
  temperature solver's Pass-1 fold (the only place `rad_net` becomes
  temperature); the gas energy seam (`gamemap.py::gas_energy_*`) if air ever
  absorbs; the energy closure identity (a new channel = a counter AND a term in
  one of the four groups); the material table (any new per-material column is a
  table row); the fire tuning lab (`tools/fire_tuning_lab.py`) as the
  instrument; GOLDEN_AGGREGATE re-baseline discipline.
- New systems this session may create (draft rules, to be written into
  CLAUDE.md only when they exist): a per-channel distance/transmittance law in
  the march (rule: light and heat may carry different per-ray laws on shared
  rays, each named and cited); a counted "air absorption" export or seam
  deposit; a flame-contact spread channel (rule: short-range spread lives in
  ONE place, never as a second radiation).
