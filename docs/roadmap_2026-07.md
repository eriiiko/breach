# Breach roadmap — July 2026 (vacation window)

Agreed Erik + Claude 2026-07-04. The working copy of "what we do in the coming days
and weeks," in order. Each item names its owner and its done-gate. Project memory
holds the detail; this is the shared map.

---

## Phase 0 — NOW (this week): close the determinism arc → tag `cuda-breached`

**0.1 The Q2-lift** *(Claude, green-lit 2026-07-04; autonomous patch workflow;
no feel-check — quantization deltas are ~1/65536, imperceptible)*
The last cross-machine divergence is the deliberately-deferred Python-float unit
state (the "Q2 fence") — prime suspect `facing = math.atan2(...)`, the only
transcendental in the synced unit state. The lift, ~3 small patches on one branch:
1. **Deterministic trig kit** in `cpp/src/fixed_point.h`: `atan2_q16`, `sin_q16`,
   `cos_q16` — pure-integer polynomials (range-reduced, minimax), FP_HD (host+device),
   pybind-exposed. Gate: accuracy sweep vs libm (pinned error bound), quadrant/edge
   cases, pure-integer determinism.
2. **Wire it**: unit `facing` via `atan2_q16`; combat damage deltas quantized to
   Q16.16 before HP applies (belt-and-suspenders — plain float mul/add is already
   IEEE-stable, this future-proofs it). Refine the per-field x-arch tool to hash unit
   sub-fields (hp / facing / pos) separately.
3. **Raycaster trig swap** (Erik pre-approved, no feel-check): `build_ray_list` +
   the CPU cast use `sin_q16`/`cos_q16` instead of `std::cos/sin` — retires the last
   latent cross-machine transcendental (matters for fire-on-solids heat).
Then ONE golden re-baseline (the trajectory moves by quantization), all CUDA gates
re-run green, auto-merge on green.

**0.2 Lenovo cross-machine confirm** *(Erik, today)*
On the Lenovo: `git pull`, then `<lenovo-py> tests/_xarch_perfield_digest.py`.
- If pulled AFTER the Q2-lift merge: expect **all green** (fields + unit sub-hashes
  match the new Ampere baseline) → that IS the cross-machine proof.
- If run BEFORE the merge: expect exactly ONE diverger — the unit-state hash (the
  facing sub-field) — naming the culprit. Also useful; send the line.

**0.3 Tag `cuda-breached`** *(Claude, when 0.2 reports green)*
The arc closes: whole engine GPU-ported (7/7, bit-identical) + the full synced
trajectory cross-machine deterministic (compiler + arch + runtime). Then we leave
determinism alone.

## Phase 1 — NEXT WEEK (fresh tokens): the EOS decision

**1.1 Literature research pass** *(Claude, deep-research workflow)*
Targeted survey: compressible/EOS schemes for a 2D game grid (Kwatra stable
compressible, Feldman–O'Brien suspended-particle explosions, thermal LBM, Euler +
artificial viscosity, divergence-control for fire), stability/CFL at game tick rates,
and integer/fixed-point portability of each. Output: a compared recommendation.
→ **Brief authored 2026-07-05: `docs/eos_research_brief.md`** — constraints, weighted
criteria, the worked hot-core CFL example, prototype scenario spec, and the black-body
session's decisions folded in (heat one-way/absorb-only; two-tier light; determinism
scope per §0.9).
→ **DONE 2026-07-08: `docs/eos_research_report.md`** — deep-research harness run (2 claims
adversarially confirmed, 0 refuted; verification cut short by rate limits, synthesis
completed in-thread by Opus with confidence labels). Bottom line: the acoustic-CFL fear is
retired (three precedented escapes). **Rung A → Feldman–O'Brien prescribed-divergence
incompressible (Bifrost ships it; keep wave_p separate); Rung B → Kwatra semi-implicit
(reuses the RB-GS Poisson kernel; may unify atmosphere+wave_p).** The whole A-vs-B call
reduces to ONE unmeasured thing — does rung-B baroclinic curl visibly beat rung-A
expansion at ≤256²? — which is precisely what 1.2 must settle by eye.

**1.2 In-engine-shaped Python prototype** *(Claude builds, Erik judges by eye)*
Rung A ("Darcy-EOS refit": conservative N-flux + derived P=C·N·T + advected gas T,
quasi-static wind — moderate cost, no real curl) vs rung B (full compressible with
momentum — the big rewrite, real mushroom clouds) on actual ship scenarios
(corridors, breach, explosion). Rung A ≈ Erik's "everything downstream stays the
same" intuition; rung B is what the gorgeous explosion-spike gif ran.

**1.3 DECISION (Erik):** adopt rung A / rung B / defer. If adopted: it is a proper
arc (spec → patches → gates → CUDA re-port → new goldens) and **must land before
serious NN training** (train once on final physics).
→ **Immediately AFTER the 1.3 decision — doc consolidation pass (Erik, 2026-07-05):**
fold the chosen path into real living architecture chapters (engine/04 + 05 + 08
updates, per the design-docs-are-canon rule) and archive/prune the brainstorms
(`blackbody_smoke_and_rendering_brainstorm.md`, `eos_research_brief.md` + its report,
`determinism_without_fixedpoint_research.md` → `docs/reference/`). Deliberately NOT
before: brainstorms stay live until the fork is chosen. Determinism note: the EOS needs
~zero new transcendentals — the toolkit already covers it (conservative flux, integer
SL advection, mul, sqrt-CFL). Cross-machine determinism is a HARD requirement
(multiplayer + distributed training + portfolio), so an authoritative EOS gets the
full integer treatment.

## Phase 2 — the game (Erik's roadmap, order confirmed)

**Weapons → units → game rules → self-play NN training.**
- S8 GPU-residency + CUDA graphs (spec'd, parked at `docs/cuda_s8a_residency_spec.md`)
  lands as the optimize-hard pass **before** big training runs.
- The explosion engine slice (mini-EOS heat→P tap + the expansion→pressure tap) and
  the black-body raycaster emitter (emit-light yes / absorb no; glow_T decoupled from
  received heat → no feedback) are parallel beauty tracks — post/alongside, per taste.
  If Phase 1 adopts the EOS, the explosion slice merges into it.

## Standing / loose ends
- Lenovo is the vacation dev machine: `docs/lenovo_dev_setup.md` (memory copy, builds,
  tests). Optional later: symlink the Claude memory dir through Google Drive.
- Scratch build dirs `cpp/build_perturb/`, `cpp/build_msvc1444/` — Erik deletes
  manually (deny-list blocks recursive delete).
- Python 3.11 vs 3.12 parity: NOT needed once the Q2-lift lands (the engine is
  runtime-independent). Ship-time story: frozen bundled runtime + the integer core.
