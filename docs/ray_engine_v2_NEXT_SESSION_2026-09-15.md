# Ray engine v2 — handoff (2026-09-15)

> **STATUS UPDATE, 2026-09-15 evening — §3 and §4 below are DONE.**
> Design v3 is `docs/ray_engine_v2_design_v3_2026-09-15.md` (committed
> `968f62a`, then amended in place). Critique 3 (determinism + CUDA, Fable) is
> `docs/ray_engine_v2_critique_3_determinism_cuda_2026-09-15.md`: 3 blocking,
> 12 required, 10 notes, and its integer probe confirmed the sweep's core. All
> of it is folded into v3 (its §0 rows 25–31, Appendix C), and Erik ruled the
> three open questions the same evening (a body radiates at ambient;
> `dyn_heat_atten_q` enters the digest at P3; the clamp's GPU twin lands inside
> P3). Erik's five answers to §5 below are in v3 §0 rows 1–6.
> **The next task is P0**, the integer reference sweep (v3 §11), on Opus, in
> its own worktree; then P1 on Fable. The agent plan per patch is v3 §11.5.
> Start at v3 §0, then §11.
>
> **STATUS UPDATE, 2026-09-16 evening — P0, P0b and P1 are MERGED; the
> orchestrator moves to an Opus session.** `fire-12` is green on both builds
> (2552 passed / 0 failed). The sweep runs every tick in shadow and holds the
> integer reference bit for bit (gate 0). Erik ruled the Fleck floor 0 (v3
> row 39) and cut every remaining patch into small Opus steps (v3 §11.5, row
> 40) after two Fable kills by the session rate limit (issue #66 — the token
> analysis has its own day; do not re-litigate it). **If you are the Opus
> session: you are the orchestrator now.** Read v3 §0 rows 32–40, §11 and
> §11.5, then the memory file `project_ray_engine_v2_arc`. Rules of the road:
> one implementer at a time in its own worktree off `fire-12`, branch
> `12-<step>-<slug>`, briefs SMALL (one system, a handful of files, one gate),
> every implementer writes its `report_pX.md` incrementally, never resume a
> killed agent except once for its report, merge `--no-ff` on green, rebuild
> BOTH `cpp/build` and `cpp/build_cuda` in main after any binding change, run
> the full suite in main, checkpoint memory at every boundary, HUMAN-TEST
> gates (P3b, P6b) are built + gated + pushed and NOT merged. **P2a is MERGED
> (`69a1f2d`): floor 0 is live in the reference and the engine, the sweep
> zeroes its own outputs, the tile inspector reads Φ.** The next step is
> **P2b** — calibration by derivation (v3 §9, §11's P2b row): the currency
> pinned on the furniture row's real heat capacity with stated assumptions;
> the derived scale as `[physics.radiation] rad_scale_derived`, fed to the
> SWEEP's emissive table only (the old cast keeps its fitted `rad_scale` until
> P3c, so the live game and the goldens do not move); per-row emissivities
> PROPOSED in the report, not applied (the rows are Erik's); the reach bench
> on the shadow planes (`E°⁻¹(Φ)` crossing `ignition_temp`, per source size,
> fitted vs derived, plotted and looked at) extending `tools/fire_tuning_lab.py`;
> `report_p2.md` written incrementally. Then P3a. Every step small, Opus.

> **This file IS the prompt.** Point a new session at it and it has everything:
> what to read, what is settled, what is open, and what to do first.
>
> Supersedes `ray_engine_v2_NEXT_SESSION_2026-09-13.md`, which is now history —
> its blocking question (reach / the falloff law) was set aside by Erik and is
> **not** what this arc is about any more.

---

## 0. Orient in two minutes

Branch `fire-12`, and as of `82b11a5` it is **GREEN — 2453 passed, 0 failed.**
That is survey §10.1's gate 1, met for the first time in the arc.

Python is **always** `C:/Users/steen/anaconda3/python.exe` (a bare `python` is a
Windows Store stub). Tests: `C:/Users/steen/anaconda3/python.exe -m pytest tests -q`.

The arc in one line: fire session #12 hit a 14-tile radiative flashover, which
traced to a ray engine whose heat law has no distance falloff and whose receivers
cannot radiate. Erik chose to rebuild radiation from first principles. **It is now
designed, critiqued twice, and measured — but not built.**

---

## 1. Read these, in this order

| # | Document | Why |
|---|---|---|
| 1 | `docs/ray_engine_v2_design_v2_2026-09-13.md` | **The design.** §0 is a table of what changed from v1 and why — read that first, it is the fastest way in |
| 2 | `docs/ray_engine_v2_critique_2_engine_2026-09-13.md` | **Your work order.** 5 BLOCKING, 31 REQUIRED, 9 NOTE, consolidated into 30 prioritised entries at the end |
| 3 | `docs/ray_engine_v2_survey_2026-09-12.md` §1 | Erik's rulings R-A..R-S. **Not under critique** — fidelity to them is |
| 4 | `docs/ray_engine_v2_reach_and_papers_2026-09-13.md` | The measurement session: the papers, the reach lever, the flashover case |
| 5 | `docs/ray_engine_v2_critique_1_physics_2026-09-12.md` | Round 1, for scope — its 12 changes are resolved in v2, do not re-litigate |
| 6 | `docs/papers/README_ray_engine_v2_2026-09-13.md` | 16 papers, with what each one changed. **Erik's standing instruction: read in full before implementing something from one** |

**The instruments** are in `docs/ray_engine_v2_scheme_study_2026-09-13/`. Every
number in the design is reproducible from them: `sweep_ref.py` (the reference
sweep), `reach_and_leak_study.py`, `flashover_study.py`, `contact_faces_study.py`,
`stability_study.py`, `smoke_heat_capacity_study.py`, `cascade_fix_study.py`,
`reflections_study.py`, `smoke_study.py`, plus the real-scene renders.

---

## 2. Settled — do not reopen

- **One sweep, not two solvers.** Step and shear are one parameter apart. **Heat
  takes shear** (rounder ignition footprint), **light takes step** (Erik's call on
  the real-scene renders). Light is a payload on machinery heat needs anyway.
- **Per-tick rotation is dropped.** Camminady's rSN interpolates a *carried*
  angular flux; we carry none, and the paper's §7 rules it out for sweeps.
- **S16, half-offset.** More ordinates make shear worse and do nothing for step's
  near field.
- **Stability: the Fleck factor + a maximum-principle clamp.** α adaptive,
  `max(0.5, 1 − 1/g)`, per Fleck's own condition. **The Fleck factor IS the local
  implicit solution** — that is what "implicitly and locally" means in the
  literature. What is ruled out is the *other* local route (Newton on the full
  material balance), which Fleck measured in 1971 as losing up to 20% of energy.
- **Radiation and conduction both act between touching solids, and are summed.**
  The textbook combined conduction–radiation model. Named honestly in the design
  as teleportation error, not dressed up as physics.
- **Units block heat and absorb into `rad_flux`.** No unit emission yet, but the
  interface must be per-unit absorbed flux so adding it later is additive.
- **Smoke absorbs heat in its own patch**, and glowing smoke then falls out free —
  the temperature map is the emission map.
- **Gas stiffness is bounded and modest, and density cancels.** `g ~ soot fraction
  × E°/T_abs`, so a decompressing room does not get stiffer as it empties; even a
  cell of pure soot reaches only g = 4.2, against a solid's 848 at `T_MAX_PHYS`.
  Fleck + clamp are still required on gas, but they are not holding back a runaway.
  **Gas absorption and emission must ride the existing `N_EPS_RAW` floor**
  (`gas_energy.h` — "ONE value, every file"), because a sub-`N_EPS` cell is
  *defined* to read ambient and must not emit at some other temperature.
  See `smoke_heat_capacity_study.py` §4, which corrects §§1–2 of that same file.
- **Ambient light becomes a directional boundary condition**, not a flat shader
  constant.
- **The rules-side light field is computed but not digested** until a rule reads it.
- **Cascades are not built**, but the bilinear fix is prototyped and measured, and
  §6.3 states the seam that keeps the option open.

---

## 3. First task — design v3

**Rewrite `docs/ray_engine_v2_design_v2_2026-09-13.md` into a v3 that resolves
critique 2's 30 entries.** Work the BLOCKING six first; they are what stands
between here and cutting P1.

The six, in Erik-readable form:

1. **There is no integer extinction plane.** §2.3's `(stream * a) >> 16` needs `a`
   as Q16. `heat_atten` is `np.float32` and the C++ march eats it as a float.
   Declare the plane, its `*_fixed.py` boundary module, its load-time quantization
   site, and what happens to the float one.
2. **The clamp cannot live where v2 puts it.** `T ← E°⁻¹(Φ)` is a bare
   `temperature[i] =` write. On gas that is forbidden by the "gas temperature is a
   mirror" rule — and that rule records the failure as **silent**. On solids it
   must sit inside the temperature solver's Pass-1 fold or the solid ledger stops
   closing.
3. **Φ is undeclared.** The clamp needs the per-cell fluence carried from the sweep
   to the material update. Declare the plane, its allocation, its clear line, and
   its resident-path treatment.
4. **P1's gate list omits the maximum-principle gate** — the gate for the thing the
   stability work exists to fix.
5. **§5.3 named the wrong writer** (already corrected in v2, verify it reads right):
   the sweep writes `rad_net`; what opens is the Pass-1 `ts[i]` mask; the counter is
   the existing `e_gas_deposit_sum`.
6. **Unit absorption needs an exit term in §2.3's conservation identity**, at P1,
   so gate 1 is built once rather than twice.

Then the 24 REQUIRED. Several need code archaeology rather than judgement — item
26 alone lists seven consumers whose disposition the design never states.

**Two things in the critique are wrong; do not apply them:**
- It reports the `config.toml` furniture comment as absent. It is present,
  committed at `678d9b6`.
- Its first suite figure (1 failed / 351 passed) was a truncated `-x` run; it
  corrected itself, and the branch is green now anyway.

---

## 4. Then — critique round 3

**Determinism and CUDA**, the one pass never run. Erik has Fable tokens and this is
the right place to spend them: a fresh critic, and the hardest remaining remit.
Run it against **v3**, not v2 — critiquing arithmetic that is about to change
wastes the pass. One critic at a time, verdict to disk before anything else spawns.

Its remit should include, at minimum: the Fleck divide against the number-ingress
doors and `/fp:strict`; the CPU↔GPU bit-identity story for a wavefront sweep
(shear's dependency is a whole column, step's is an anti-diagonal); the A/B
lockstep harness; and the digest/golden consequences of every new plane.

---

## 5. Open questions — Erik's, not yours

1. **Where the sweep lives.** A new `radiation_sweep.cpp` versus growing
   `raycaster.cpp`. Erik's instinct (recorded) is drop-in replacement, and the old
   raycaster survives for weapon beams and the legacy scalar march either way. The
   commitment that matters: the old heat march is **deleted** at P3, never left
   live alongside the new one.
2. **`rad_flux` recalibration is a feel ruling** — moving it from incident flux to
   absorbed energy shifts the burn threshold. P5, HUMAN-TEST.
3. **The gas branch's density double-count** (critique §2c): does radiation into
   gas ride the density-proportional absorption law or bypass it?
4. **Is the 16-light cap's removal a look decision or a cost one?**
5. **Foliage `heat_atten = 0.0`** — trees are thermally isolated. Unruled.

Two that were open on 09-13 are now **closed**: the tick budget (24 Hz / 41.67 ms
is settled; only a stale per-system gate in old perf docs needed restating, and
there is ~32 ms of headroom at shipped scale), and the `cool_shift` reds (the tests
were snapshots of a tuning value and have been rewritten to assert the property).

---

## 6. How Erik wants to work

- **Language first, code second.** Never implement before the approach is agreed.
- Ask questions as **plain text in the chat**, never a popup.
- He is a PhD student: **precision over speed, explain why, be direct.** And
  **explain the jargon** — he has said clearly that unexplained terms lose him.
- **Nothing is considered tuned until the physics works**, because it is all
  interconnected. Do not treat any current dial value as intentional.
- **Tests assert properties, never snapshots.** A negative assertion must be paired
  with a positive one or it cannot fail.
- **Look at the picture before trusting a scalar summary of it.** This arc has
  published two wrong headlines that a plotted profile caught.
- **Read the papers in full.** Doing so changed the design three times, including
  catching a failure in a scheme that had already been locked.
- Big changes run as arcs: design → adversarial critique → patches with gates →
  CUDA lockstep → close. Critics one at a time.
- He is happy with autonomous work; re-enter the loop only for rulings that are
  genuinely his.
