# Boundary conditions — planetside AMBIENT ring (2026-07-19)

**Status:** v2.2 — physics close-out, priority ledger #1. B2 (format/load) + B3a/B3b (CPU
physics) LANDED green on `bc-ambient-ring`; the shift + per-substep reset + pin work
(gate 1 flat interior at defaults AND non-default dials; gate 2 rush-in — both pass). **B3
inverted the §3 absorber ladder** (σ-pressure sponge reflects, ships OFF; u-damping is the
real absorber, not wired — OPEN feel decision, §3). Remaining: absorber decision → B4 CUDA
lockstep → B5 human feel-test.
**Sequencing:** lands **BEFORE** the S8a residency build (`cuda_s8a_residency_spec_2026-07-19.md` §5c).
**Sources:** Topic 4 survey (`notes_2026-07-17_topics_backlog.md`), A9 hook (`level_loader.py:468-533`),
Erik's decisions (§0), STEP-A audit (`bc_step_a_audit_2026-07-19.md`), three-lens adversarial
critique (determinism/lockstep, physics/EOS, scope/regression — same session).

> **v1 → v2 changelog (critique outcomes, all resolved on paper):**
> 1. Value-carrying pin REPLACED by the **shift trick** (solve P′ = P − P_amb) — the coarse
>    Galerkin anchor is only exact for pin value 0; the shift keeps the entire MG byte-identical.
> 2. Ring reset moved from "per tick" to **per-substep at the exact vacuum-idiom sites** —
>    the vacuum twin acts per substep; a per-tick reset lets ≤8 substeps drain the reservoir
>    and permits CPU/CUDA landing it in different places.
> 3. **AMBIENT ≡ vacuum for u and T** (ΔT form; ring T ≡ global T_AMB_K, NO T dial) — the
>    u/T side is literally the existing vacuum code; only N and P carry new values.
> 4. EOS identity direction pinned: **N planes are primary**; the effective pin is the sim's
>    own p\* chain applied to (N_amb, ΔT=0). At defaults that is 65540 raw (1.000061 atm),
>    NOT 65536 — no integer N hits 1.0 atm exactly (p\* lattice is multiples of 290 raw).
> 5. u-damping sponge DEMOTED: acoustics travel ~37.5 tiles/tick through the implicit solve —
>    an 8-tile u-band gets ≤1 bite at a front. Absorber ladder (§3): measure pin-only echo
>    first → **σ(d) pressure-sponge mass in the level-0 solve rows** → u-mop-up last.
> 6. Reflection gate re-protocol'd: big-map reference run, region metric (single probe is
>    aliased/sign-blind/corner-confounded).
> 7. Wholesale SPACE→AMBIENT reinterpretation (no interior-tile error; "sky shafts" legal).
> 8. Interior t=0 seeding becomes dial-aware; sponge grid computed post-upscale; joins-ambient
>    twin rules for destroy_wall/unseal; [ambient] validation table; rail = int64 per-substep
>    per-plane; assorted audit-driven mask widenings (§2).

**Goal:** planetside levels as well as space. Waves leaving the play area are absorbed to
imperceptibility (≤2% reflected, §6 gate 3); air exchanges freely at the boundary (infinite
ambient reservoir). Pragmatic simplest; determinism iron; no solve-structure change.

## 0. Decisions locked (Erik, 2026-07-19 + delegation)

1. **Whole-map single boundary type** via the A9 `boundary` field. Exotic cases = level cheats.
2. **Ambient dials per level**; defaults Earth-like. NO ambient-T dial (v2 §1 — ring T ≡
   T_AMB_K is what lets every `T:=0` vacuum idiom generalize verbatim; a cold-planet dial
   would break them all at once — future project if ever wanted).
3. **No water BC.** Oceans = indestructible authored reservoir. Ring = space-ring for water
   (one seed-mask widening; the water solver never reads is_vacuum at runtime — verified).
4. **Wind-in ≠ boundary mode** — source term, separate feature.
5. **Absorption target is imperceptibility, not perfection** (Erik). Mechanism was delegated
   to Claude — but B3 measurement (v2.2, §3) inverted the designed ladder: the σ-pressure
   sponge reflects; velocity damping is the real absorber, and is NOT yet wired. Whether to
   wire it at all is an OPEN feel-adjacent decision for Erik (the ~0.02 atm residual may be
   imperceptible → a B5 call). Characteristic/radiation BCs remain rejected.

## 1. The mechanism (v2 — the structural principle)

**AMBIENT ≡ vacuum for u and T; N: sink→clamp; P: pin via shift.** All local per-tile edits.

- **Ring representation:** `boundary == "ambient"` routes the tilemap's SPACE-code mask
  WHOLESALE to a new `is_ambient` bool mask (`gamemap.py:357` branch); `is_vacuum` is
  all-false on ambient maps. Interior SPACE tiles are legal = interior ambient columns
  ("sky shafts" — vents to the open air; kin of the Arc B pump, §8). Order: the ambient
  branch runs BEFORE the A6 door stamp; door-span validation widens to
  `is_vacuum | is_ambient` (a door on the ring stays an authoring error).
- **P — the shift trick:** solve the Helmholtz system in **P′ = P − P_amb**. Subtract
  P_amb inside the shared host-side `mg_build_levels` (rhs `eos_solver.cpp:736-738` + warm
  start `:744`; used by BOTH CPU and CUDA paths — verified `cuda_eos_step.cu:411-413`),
  add P_amb back at the step-5 store on both paths (`eos_solver.cpp:641` /
  `cuda_eos_step.cu:469`) — **masked to `!solid`** (solids leave the solve at P′=0 and
  must stay 0 absolute; ring excl cells WANT the add — v2.1 fix). The zero-Dirichlet MG
  (smoother, residual, coarse Galerkin anchor, V-cycle kernels) stays **byte-identical** —
  legal precisely because ambient maps have no P=0 pins. The shift is **branch-gated on
  ambient mode** (no unconditional "−0" on space maps — §5 dormancy-by-branch). Verified
  shift-safe: the kick reads only P-differences (shift-invariant, exact in integers);
  p_prev stores unshifted P and is re-shifted fresh each tick; P′ &lt; 0 is fine (all solve
  arithmetic signed; headroom ~2⁴⁶ vs 2⁶¹ budget); the near-vacuum degeneracy pins
  P′≈−P_amb ≡ P≈0, today's answer.
- **N — sink becomes clamp, per substep:** in the existing bulk-transport clamp pass
  (`bulk_transport.cpp:182-188` / `cuda_bulk_transport.cu:163-173`):
  `else if (is_ambient[i]) N[i] = N_amb[plane]` (conservative planes) — the reservoir,
  refilled every substep (a per-tick reset lets ≤8 substeps drain it — v2 change 2).
  Traces: ambient joins the existing zero at the smoke-SL vacuum sites (absorbed).
- **T — the vacuum code verbatim:** widen `is_vacuum → is_vacuum | is_ambient` at the SL
  write ternary (`eos_solver.cpp:402` + CUDA twin), the step-4c compression-work skip, the
  temperature-solver pre-pass wipe (`temperature_solver.cpp:159-161`), and the step-4
  velocity zero (`eos_solver.cpp:511,1183`, `cuda_kick_compression.cu:119,205`) + SL cmask
  barrier. ΔT=0 IS ambient; ring u ≡ 0 (still boundary; the pin drives interior flow).
- **Dial derivation (N-primary, v2 change 4):** loader computes
  `N_total_amb := quantize(p_amb)`; O2 split by the existing round-half-up + complement
  pattern (`gamemap.py:442-446`); **effective pin `P_amb := p*(N_amb, ΔT=0)` through the
  sim's own truncating mul chain**, logged at load (65540 raw at defaults). The `p_amb`
  dial is the author's target; the effective pin is what the physics uses everywhere
  (shift, burst differential, init).
- **t=0 seeding (dial-aware — v1 blocker fix):** on ambient maps the interior open-air
  default seed is (P_amb, N_amb split), NOT the hardwired FP_ONE/21% (`gamemap.py:518,533`);
  `air_init` O2-split uses `o2_frac`; ring tiles init at P_amb/N_amb (warm start + tick-0
  CFL scan read them). air_init override mask excludes the ring (`is_vacuum|is_ambient`
  widening at `gamemap.py:418,440` — ring rules win).
- **Structural edits (joins-ambient twins):** on ambient maps `destroy_wall`'s
  edge-hull/exposes rule and `unseal_tiles`' join rule set/join `is_ambient` (never
  `is_vacuum`) — a breached edge hull opens to sky, not to space.
- **Consequential naturals (accepted as correct planetside physics):** fire/combustion O2
  reads already include ring tiles (not vacuum → they hold N_amb — fires near the boundary
  breathe); ring-adjacent solids lose the vacuum 0 K fast-cool path (they now see a T_amb
  bath); `find_burst_walls` reads an ambient side as P_amb, not 0 (a wall with ambient on
  both sides does not burst); destroy_wall neighbor-mean refill counts ring neighbors
  (they hold real values). Space maps: all unchanged.

## 2. The audit (STEP A — DONE, `bc_step_a_audit_2026-07-19.md`)

Repo-wide (loader + GameMap init + sim + C++ + CUDA — not just kernels). Verdict: with §1's
structure, the edit surface is ~12–16 mechanical mask-widenings + the new writers
(clamp/reset, shift, σ-sponge, rail) + loader work. No solve-structure change. The audit
table is the build's checklist; every touched TU is listed there. The retired pre-EOS
`atmosphere_solver.cpp` contains a working BFS+ramp sponge (`:480-542`) — use as algorithm
reference, do NOT edit the dead file.

## 3. The absorber ladder — INVERTED by B3 measurement (v2.2, 2026-07-19)

> **B3 finding (empirical, `9d7d244`): the ladder below was backwards.** The σ-pressure
> sponge (old rung 1) **reflects, and reflects MORE as σ_max rises** — physically a soft
> Dirichlet pin is a pressure-release boundary, and adding diagonal mass just hardens it.
> Rung-1 was WIRED (consumption in `mg_build_levels`, ambient-gated, space-dormant) then
> **calibrated to σ_max = 0** (ships OFF; `DEFAULT_SPONGE_STRENGTH = 0`). The **velocity
> damping** (old rung 2, demoted to "mop-up") is the actual absorber — a scratch probe cut
> the residual substantially (e.g. 66%→21%) where σ made it worse. This matches the PML
> truth: you absorb by removing momentum energy (damp u), not by pinning P harder. So the
> real ladder is **rung 0 (pin only) → u-damping band**; the σ-pressure sponge is a dead
> dial kept only for the record. **DECISION (Erik, 2026-07-19): build the u-damping
> absorber FULLY now** — robust reflection harness + wire the velocity-damping band +
> calibrate k_max to a numeric target — BEFORE B4, so B4 locksteps the complete physics in
> one pass (quality over dev-cost; the correct path, not the lean one). This is B3c.

Acoustic fronts cross ~37.5 tiles/tick via the implicit solve (c·dt/dx; matches the ambient
Helmholtz k≈1409). The as-built ladder:

- **Rung 0 — pin + ring alone (SHIPPED).** The P_amb Dirichlet ring + mass-swallowing
  reservoir. Measured rung-0 residual is clearly non-zero but the B3 harness was
  timing/geometry-aliased (2–24% across geometries — NOT a trustworthy ≤2% verdict). The
  transient amplitudes are small (~0.02 atm), so whether the residual is even perceptible
  is likely a **B5 feel question** (§0.5 imperceptibility, not perfection), not a hard gate.
- **Rung 1 — σ(d) pressure sponge: DEAD (measured to reflect).** Wired + dormant (σ_max=0).
  Kept in-code (ambient-gated, space byte-identical) so the finding is reproducible; do NOT
  spend more on it.
- **The real absorber — u-damping band (B3c, BUILDING per Erik's decision):** magnitude-
  first Q16 multiply on the band velocity (the truncating-mul sign convention — a naive
  signed multiply leaves a stuck −1-count floor), placed after the existing absorb chain in
  step 4 (`eos_solver.cpp:531-541`) + its CUDA twin (B4). Reuses the B2 BFS distance grid
  for the band; `sponge_u_damp` (k_max) is the live dial. Requires FIRST a **robust
  reflection harness** (big-map reference run, ring pushed ≥ c·dt·T_window away; metric =
  max over an interior probe REGION and the window of |P_test − P_ref| / max|P_ref − P_amb|
  — immune to the phase/sign/corner aliasing that made B3's scratch metric read 2–24%).
  Then calibrate k_max until gate 3 passes at the best achievable margin; pin the default.
- **Grid facts (still valid):** the B2 `sponge_sigma` grid is computed in `GameMap.__init__`
  from the FINAL post-upscale grid, W scaled by res_factor, int32 Q16, quantized once at
  load. A u-damping band would reuse the same BFS distance infrastructure. Staleness
  accepted: destroy_wall near the ring can open un-sponged air — documented gap.
- **Calibration:** deferred with the u-damping decision; requires the robust harness first.

## 4. Level format (backward compatible)

- `boundary = "ambient"` (A9 field, semantics now live).
- Optional `[ambient]` table — validation per loader style (unknown keys rejected,
  path-bearing errors): `p_amb` float atm, **> 0** (default 1.0; effective pin logged);
  `o2_frac` **∈ [0,1]** (default 0.21); `sponge_width` **int ≥ 0** tiles (default 8;
  0 == hard ring; warn when ≥ min map dimension); `sponge_strength` (σ_max, Q16 raw, bound
  **[0, 256·FP_ONE]** — v2.1 fix: an FP_ONE bound caps the per-row pull at ~20% and would
  reject the gate-calibrated value; default = calibrated constant); `sponge_u_damp`
  (k_max, **[0, FP_ONE)**, default 0 = rung-2 dormant). `[ambient]` with `boundary="space"` → hard error.
  Ambient map with zero SPACE-code tiles → WARN (legal, ring-dormant sealed box).
- All values quantized once at ingress. NO ambient-T dial (§0.2).
- Same-patch prose updates: `level_loader.py:468-473,485-493,529-532`,
  `gamemap.py:415-424`, `tests/test_air_boundary.py:16` (all currently assert
  "changes NO behavior" / SPACE≡vacuum).

## 5. Conservation bookkeeping (open system, by design)

- Principle (Erik + Claude, 2026-07-19): **conservation proofs attach to WRITERS, not
  settings.** The ring reset is the one new non-conservative writer; the rail audits it.
- **`boundary_flux` rail:** int64, **per-plane**, accumulated at the reset sites **per
  substep** as Σ(N_pre-reset − N_amb); device path uses atomicAdd (integer sums are
  order-exact — unlike the FNV digests, reduction order is a non-issue). NOT folded into
  tick_digest (the absence-transparent precedent, `tests/field_digest_spec.toml:47-50` —
  zero re-baseline).
- **Gate-5 identity (restated):** on a fire-free, trace-free fixture,
  `Δ(total map N, conservative planes) == −Σ boundary_flux` at LSB level. (Combustion is
  a second N-writer; trace decay credits inert_N2 — both excluded by fixture, not by
  hand-waving. Pre-existing negative-N mint at `bulk_transport.cpp:185-187` is out of
  scope; the fixture avoids it.)
- Existing maps: no AMBIENT tiles → branches dormant → existing conservation guarantees
  and goldens untouched. **Dormancy is BY BRANCH, not by arithmetic identity** (no
  unconditional k=0 multiplies on the space path), asserted by running the standard digest
  suite in CI on the BC branch.

## 6. CUDA lockstep + gates

BC lands on BOTH paths in one physics patch (CPU + the CUDA twins named in the audit
table), same-tick lockstep, before S8a freezes kernels.

- **E2E gates (new AMBIENT fixture level, built fresh — no .bak strays):**
  1. Sealed planetside room holds equilibrium — **run at defaults AND at non-default
     dials** (p_amb=0.6, o2_frac=0.3) to prove dial-aware seeding.
  2. Breach to ambient → air rushes IN, room recovers toward P_amb; **record and bound
     per-tick rail-hit deltas** (u_clamp / work_clamp / t_max_phys) — the convergent-fill
     front is the documented compression-pocket regime; rails must bound it, and the
     breach-mouth T behavior is a named bake watch-item.
  3. **Reflection gate (v2 protocol):** same detonation, two runs — test map vs a big-map
     reference (ring pushed ≥ c·dt·T_window away). Metric: max over an interior probe
     REGION and window of |P_test − P_ref|, normalized by max|P_ref − P_amb|; **≤ 2%**
     with the shipped rung (≥2× margin at calibration). Single-probe |P−P_prev| is
     aliased/sign-blind — reference-run comparison only.
  4. O2 replenishment: sustained fire near the ring keeps burning (fire fixture, separate
     from gate 5's).
  5. Rail identity per §5 (fire/trace-free fixture).
- **Lockstep gate:** CPU vs CUDA A/B on the AMBIENT fixture, all synced fields
  byte-identical (tol 0), `cuda_*_check` pattern; new fixture golden committed. Existing
  space goldens byte-untouched (standard digest suite on the branch).
- Full suite green: `pytest tests -q` (conda env `data`).

## 7. Build plan (autonomous-patch-workflow; branch `bc-ambient-ring`)

- **B1 — STEP-A audit: DONE** (this session; `bc_step_a_audit_2026-07-19.md`).
- **B2 — format + load** (subagent; digest-gated, auto-merge on green): §4 parsing +
  validation, `is_ambient` mask + wholesale reinterpretation + door-stamp ordering,
  dial-aware seeding, sponge grid (post-upscale), placeholder config constants, prose
  updates, loader tests. Zero sim-behavior change (masks exist, nothing reads them yet).
- **B3 — CPU physics** (subagent; digest-gated, auto-merge on green): shift trick, clamp
  reset, u/T widenings, joins-ambient twins, burst-differential read, rail, rung-0 echo
  measurement → σ-sponge if needed → calibration; E2E gates 1–5; existing digest suite
  byte-untouched. **Build-riders (v2.1 verification):** (i) shift + add-back masked
  `!solid`, branch-gated on ambient mode; (ii) the CPU all-zero-plane skip
  (`bulk_transport.cpp:91-94`) skips the clamp — equivalence with the reset rests on the
  loader seeding ring N_amb; ASSERT it (or hoist the reset above the skip; CUDA has no
  skip); (iii) rail signature ripple crosses three bulk-transport entries (both
  `bulk_flux_transport_cached` definitions + the legacy wrapper) — device counter is
  int64 atomicAdd (two's-complement signed, precedent `cuda_combustion.cu:157`);
  (iv) documented inherited wrinkle, no fix: ring-adjacent fire deposits one tick of
  trace residue on the ring (fire runs post-smoke-zero), folded into next tick's Dalton
  sums — the same idiom exists at vacuum breaches on space maps today.
- **B4 — CUDA lockstep** (subagent; bit-identity-gated, auto-merge on green): the CUDA
  twins per the audit table; lockstep gate; fixture golden.
- **B5 — HUMAN-TEST** (Erik): planetside fixture + minimal ambient render tint
  (render-layer, determinism-exempt — the ring must not render as starfield for a feel
  test); Erik plays: breach rush-in feel, echo check by ear/eye, breach-mouth T
  watch-item. No planetside-shipping level merges before this.
- Patch boundaries checkpoint to memory; surprises surface rather than plough ahead.

## 8. Relation to the Arc B pump (concept unity — Erik, 2026-07-19)

The ring reset and the Arc B pump N-feed are the SAME concept at two extremes: the ring is
a degenerate pump (unconditional, per-substep clamp to a setpoint, static, in-kernel); the
pump is the dynamic sibling (rate-limited, signal-driven, entity-owned). A life-support
vent is a wireable, destroyable piece of ambient reservoir — and an interior ambient "sky
shaft" (§1) is the static version, now legal by construction. Arc B builds the pump
knowing this kinship — same setpoint/composition vocabulary, same counted-rail pattern.

## 9. Scope discipline

- No solve-structure change (the shift + σ-diagonal are row-data changes, not new solves).
  If any step balloons past the audit's estimate, STOP and report.
- Feel-adjacent: B5 is the feel gate; numbers come from gate 3, not eyes. Mechanical parts
  are digest-gated.
- Out of scope: per-edge modes, ambient-T dial, boundary trace species, water BC, wind-in,
  rain/weather, sponge-grid staleness repair, full render treatment beyond the B5 tint.
