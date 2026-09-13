# Ray engine v2 — handoff to the next session (2026-09-13)

> Erik: *"It's getting hard for me to initiate the new chat, can u help me?
> create a prompt with all the docs the new claude should read, all the
> questions we want to solve (which method to use being the number one),
> conservation etc … and the flashover study case might be of interest."*
>
> **This file IS the prompt.** Point the new session at it and it has
> everything: what to read, what is decided, what is open, and what to do
> first.

---

## 0. Orient in three minutes

The branch is `fire-12`, green against its known baseline (3 parked reds:
2× `test_cool_shift_axis`, 1× `test_bench_two_room`). Run the suite with
`C:/Users/steen/anaconda3/python.exe -m pytest tests -q` — note the **full
interpreter path**, a bare `python` here is the Windows Store stub.

The arc: fire session #12 hit a 14-tile radiative flashover, which traced to a
ray engine whose heat law has no distance falloff and whose receivers cannot
radiate. Rather than patch it, Erik chose to rebuild radiation from first
principles. That rebuild is designed, critiqued, and prototyped — **but not
built, and one ruling is blocking.**

---

## 1. Read these, in this order

**Do not skim these. The last session's biggest errors were all caught by
reading carefully or looking at a picture, and its biggest wins came from the
papers.**

| # | Document | Why |
|---|---|---|
| 1 | `docs/ray_engine_v2_survey_2026-09-12.md` | The blessed shape and Erik's rulings R-A..R-S. **Rulings are not up for critique**; fidelity to them is. |
| 2 | `docs/ray_engine_v2_design_2026-09-12.md` | The design under review. Section 3 maps every current ray interaction to v2. |
| 3 | `docs/ray_engine_v2_critique_1_physics_2026-09-12.md` | The adversarial critique. **12 required changes, every number measured.** The design is "yes-with-fixes for P1, no as a whole" until these land. |
| 4 | `docs/ray_engine_v2_scheme_study_2026-09-13/report.md` | The bake-off + renders. Contains a **withdrawn conclusion** — read the correction at the top, it is a lesson about metrics. |
| 5 | `docs/fire_radiation_assumptions_2026-09-11.md` | What the CURRENT engine does, with citations. The baseline being replaced. |
| 6 | `docs/papers/README_ray_engine_v2_2026-09-13.md` | The paper archive, with a reading order and the finding that justified the architecture. |
| 7 | `docs/fire_3c_r4_tuning_and_radiation_2026-09-06.md` | Where the flashover was found. |

**Papers** are in `docs/papers/`, all eight verified complete. `pdftoppm` is NOT
installed so the Read tool cannot render pages; `pdftotext` IS at
`/mingw64/bin/pdftotext` and every one extracts cleanly.

**Erik's standing instruction: read them in full.** Only
`osborne_sannikov2024_...pdf` has been read, and only partly. Priority: its
merge and bilinear-fix sections, then Camminady on quadrature rotation against
our own contradictory measurement, then Davis on short-characteristics
interpolation and stability.

---

## 2. What is decided (do not relitigate)

- Radiation becomes a **field solve**; emitters stop casting rays.
- **The temperature map IS the emission map.** No emitter list, no gate.
- **Every cell radiates by its own temperature.** This, not the falloff
  exponent, is what makes reach finite.
- **Clear air does not absorb; smoke absorbs strongly.**
- **No short-range flame-contact channel** — radiation is the only spread path.
- **Solids heat adjacent gas** (direction set, mechanism deferred).
- **Two solvers**: heat as an integer conservative sweep in C++ with a CUDA
  twin; light as radiance cascades in GLSL, render-only. **Justified by the
  papers**: Osborne & Sannikov state cascades cause *"non-conservation of
  energy at locations where cascades overlap a light source"*, so heat cannot
  use them.
- **Diffuse bounce yes, specular no** (top-down, mirrors invisible).
- **A unit is an opacity stamp on the grid** — the solvers never learn units
  exist. Erik's own framing, and the cleanest part of the migration.
- **S16 with a half-offset quadrature** (no ordinate on an axis).

---

## 3. THE BLOCKING QUESTION — Erik must rule

**What sets the reach?** Measured by critique 1:

- The calibration constant **cancels** from the equilibrium that sets reach
  (verified: identical equilibrium at a 100× smaller `rad_scale`).
- So reach is set by geometry and source temperature alone. Under the design's
  1/r, at the measured 1556 K crate: **≈17 tiles along the axes, ≈10 diagonal.
  Target is 3.**
- And `r_ign` scales as the **fourth power** of source temperature, so the law
  stays fragile every time the plateau moves.
- **1/r² measures 3.68 tiles — on target — and halves that sensitivity.**

Three options, all colliding with something:

1. **Adopt 1/r² as structural.** Contradicts survey §9.5, which demoted it to a
   feel dial. Cleanest physics, on-target number.
2. **Make smoke/medium absorption load-bearing.** Needs a stated e-fold length
   and collides with R-L (clear air transparent).
3. **Lower the plateau to ~989 K.** Contradicts R-E (Erik kept `H_BED_SHIFT 7`).

Nothing downstream can be calibrated until this is settled.

---

## 4. The other open questions

### 4a. Which spatial scheme? (Erik: "number one thing")

Nothing measured is simultaneously sharp, isotropic **and** exactly conservative.

| scheme | conserves | near-field isotropy | aperture RMS | notes |
|---|---|---|---|---|
| step (upwind FV) | **exactly 0.0000** | 86% ripple, 2× axis/diag | 0.244 | loses **half** the energy down a corridor |
| shear | 26% leak *in our hands* | 51% ripple | 0.313 (0.201 rotated) | critique 1 measured an integer variant at **zero** — reconcile |
| long characteristics | no | 129% → **8.3%** rotated | 1.627 → **0.102** rotated | best accuracy, expensive |

**Rotation is the biggest single lever** and it is a *trade*: it fixes apertures
for every scheme and **degrades** point-source isotropy for grid schemes.

**More ordinates make shear worse** (S16→S64: 51%→69%) because a shear step is
exact at 0° and 45° and interpolates most between. S16 is a sweet spot.

Unexplored: the **lumped Linear Characteristic** family, which satisfies corner
balance and so is conservative *and* low-diffusion, at the cost of moments per
cell. This is the most promising untried option.

### 4b. Conservation — how exact does it need to be?

Erik: *"what if we drop the conservation requirement? … I don't think we should
treat conservation of energy as an end all be all in a computer game."*

He is largely right, **with one real danger he himself guessed**: in a
compartment near equilibrium every surface emits a lot and absorbs nearly as
much, so the net is a small difference between large numbers. A transport loss
of fraction *f* shrinks absorbed but not emitted, producing a systematic cooling
**proportional to emission, i.e. to T⁴** — the same shape as real radiative
loss, therefore invisible and untunable-around. It would impose a hidden
temperature ceiling and could silently cap flashover.

**The reframing that resolves it: the requirement is ACCOUNTING, not exactness.**
The arc-#54 ledger needs to know where energy went, not that none was lost. A
measured, booked leak is fine; the project already has named accepted leaks. So
the real choice is **accounted vs unaccounted**, not exact vs lossy. 26% is
fatal at any standard; a small booked residual is comfortable.

**Task: quantify it.** Build the flashover study case (§5) and measure what
leak fraction actually kills flashover. That converts a philosophical argument
into a number.

### 4c. The cascade leak

`cascade_diagnose.py` found it: **coarse cascade probes land inside walls.**
Cascade 4's probe columns are x = 8, 24, 40; the corridor spans 27–39 between
walls at 26 and 40. So corridor cells interpolate between a probe in the wrong
room and a dead probe inside a wall. 25% of cascade 4's probes are in walls.

Measured at the shadowed far end of the corridor: `N_CASCADES=3` gives 0.0000
(matching analytic), 4/5/6 give 0.0010 (leaked). **At our map sizes the coarse
cascades add leak without adding reach.**

This is the parallax/bilinear leak; **Osborne & Sannikov 2024 carries the
published fix** and it has not been implemented in the prototype.

### 4d. Erik's questions about the images

- **"Do these have reflections?"** **No. Neither render has any bounce at all.**
  Every emitter is primary; walls absorb and do not re-emit. What reads as
  reflection or diffusion is *error*: in the step render it is numerical
  diffusion of the transport scheme, in the cascade render it is the bilinear
  leak. **But real diffuse bounce IS in the design** (R-K makes every cell emit
  by its temperature, and cascades do multi-bounce natively), so the quality he
  likes is achievable honestly rather than as an artifact.
- **"Is log more beautiful, do our eyes see light logarithmically?"**
  Essentially yes — brightness perception follows roughly a power law with
  exponent ≈0.4 (Stevens), which is why display gamma exists at all. `GAMMA` in
  the render script is exactly that knob. This connects directly to **issue #64**
  (the two-state lit/dark prototype look), which is a transfer-curve question.
- **"The dials didn't seem to do anything."** He changed `BASE_DIRS` 16→8
  (ripple 15%→19%, nearly invisible) and `MARCH` 0.25→0.125 (no visible change,
  2× slower). **The knobs that visibly transform the image are `N_CASCADES`
  (3 vs 5) and `GAMMA` (0.30 vs 0.60).** Say this up front next time.
- **"I could accept the artifacts, but I'd like to see them with more light
  sources first."** A direct request — see §5.

---

## 5. First tasks for the new session

**Task 1 — the real-heatmap study (Erik's explicit ask).**
> *"I'd like to look at these solvers one more time, but using an actual
> heatmap from a game sim as input… we would need to start a few fires, or even
> fake such a heatmap. Also have some flashlights perhaps."*

Run the real sim (`levels/fire_tuning` or `unhcr_vessel`), ignite several
fires, dump `gmap.temperature`, and feed that field to the solvers in
`docs/ray_engine_v2_scheme_study_2026-09-13/scheme_study.py` instead of the
synthetic point source. Add flashlight cone emitters. This answers "do the
artifacts survive a realistic multi-source field, and do I still like them".
Multiple sources should average the striping down substantially — that is the
prediction to test.

**Task 2 — the flashover study case.**
A sealed compartment, all surfaces emitting, driven to flashover. Measure
whether it reaches flashover under each candidate scheme and each leak
fraction. This is the test that makes §4b quantitative, and it is the scenario
the whole arc exists to make work.

**Task 3 — critique rounds 2 and 3.**
Only the physics/maths critique has run. Engine integration (does it fit the
canonical systems, the #54 ledger, the migration of `rad_flux`/`rad_amb`
consumers) and determinism/CUDA have never been reviewed. Run them one at a
time, verdict to disk before the next spawns.

**Task 4 — apply critique 1's 11 non-blocking fixes** to the design doc: the
unpaired face-split truncation, the emission-term overflow and dimension error,
the missing ambient inflow boundary, the false "rule 3 is subsumed" claim, the
flux-limiter category error, `rad_amb` losing attribution, int64 widening, the
quadrature definition, the `a ≤ 1` invariant, and moving the isotropy gate into
P1.

---

## 6. Everything else still open in fire session #12

The radiation arc grew out of the fire session and these were parked, not
finished:

- **#61 wood cannot sustain fire** — conductivity 0.15 drains ignition heat.
- **The ~165 s relight** — a crate fades, re-ignites off its own residual heat,
  dies. G6/G7 ember territory, never ruled.
- **The O2-above-ambient ghost** — local X climbs to 0.2226 (> 0.21) from ~18
  min after death, inventory conserved, cause unknown, ~1% level.
- **Spread and wind** — `k_wind_fan` still flagged untuned, `k_wind_strip`
  parked; spread had never been measured before 2026-09-06.
- **Foliage `heat_atten = 0.0`** — trees are thermally isolated; recommend 0.5
  (furniture's). Needs Erik's ruling.
- **`dyn_heat_atten`** — should a body block radiant heat? Gameplay ruling.
- **The tick budget** — config says 24 Hz (41.67 ms) but every perf doc still
  budgets 12 Hz (83 ms), and atmosphere alone measured 18.97 ms.
- **#64** — the two-state lit/dark flashlight look.
- **Phase 4 fire tuning** — the whole point of the fire session, blocked behind
  this arc.
- **`fire-12` is not merged**, deliberately. Erik: *"we don't have to merge at
  all — we could even include the ray in this patch."*

---

## 7. How Erik wants to work

From `CLAUDE.md` and this session:

- **Language first, code second.** Sync understanding before implementing.
- **Never implement before the approach is agreed.**
- Ask questions as **plain text in the chat**, never via a popup.
- He is a PhD student: **precision over speed, explain why, be direct, ask
  rather than assume.**
- **Tests assert properties, never snapshots.** A negative assertion must be
  paired with a positive one or it cannot fail — this session found five
  fixtures that were silently measuring nothing.
- **Look at the picture before trusting a scalar summary of it.** This session
  published a wrong headline because a metric misfired on a spiky profile.
- Big changes run as arcs: design → adversarial critique → patches with gates →
  CUDA lockstep → close.
- Critics **one at a time**, verdict written to disk before the next spawns.
- He is happy for autonomous work and internet research; re-enter the loop only
  for rulings that are genuinely his.
