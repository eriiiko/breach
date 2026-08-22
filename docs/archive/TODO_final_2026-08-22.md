# Breach — TODO

## ★ PICK UP HERE (2026-08-20) — the post-clamp queue

> *As of the velocity-clamp arc close (2026-08-20, HUMAN-TEST PASS).*

The physics lid keeps tightening. Next, in order:

1. ~~**N_SUB_MAX ruling**~~ — **RULED (Erik, 2026-08-20): stays at 8.**
   *"I think we can leave it at 8 to be honest. It seems to work quite
   well."* The residual over-Courant transport during breach venting
   (required `n_sub` median 251 vs the rail, 99.5% of blast ticks — P-V2)
   is an ACCEPTED cost at current feel. Revisit only if pile-up flashes
   become a felt problem again; the measurements live in
   `docs/velocity_clamp_audit_2026-08-19.md` and
   `docs/velocity_clamp_pv2_measurement_2026-08-19.md`.
2. ~~**T_abs compression-work patch**~~ — **CLOSED 2026-08-21, HUMAN-TEST
   PASS, merged.** Step 4c runs on absolute T (`T_new = (T+290)·(1±w) − 290`,
   RULING R1 executed): compression finally heats ambient/cold gas, breach
   rarefaction is genuinely cold (cold RING around vents — the trust gate
   correctly fades the near-empty core), rooms stay honestly warm/pressurized
   after violent transients. Record:
   `docs/human_test_2026-08-21_tabs_compression_work.md`; design + critiques +
   measurements in `docs/archive/`. One sanctioned golden re-baseline
   (`docs/tabs_compression_work_rebaseline_2026-08-21.md`). Decisions at the
   gate: cap ambient floor KEPT (D-1), T_MIN −289 KEPT (D-6). New instruments:
   hover readout on F6 (game-deg + K/K_eos), cold overlay tier on T.
   **Handed forward:** grenades read COLD (payload is pressure-only, no heat
   term — correct physics exposing a payload bug; → grenade energy-budget
   retune), fires burn in vacuum (→ bug list below, fire retune session),
   quiet-room acoustic drift ≈ +10 game-deg/7 min non-saturating (monitored
   accepted gap; the KE↔eth kick-side debit remains the open half).
3. **Drag law design session** — item 3 below, now UNBLOCKED (flow is
   subsonic post-clamp). **Feel-probe COMPLETE (Erik, 2026-08-20), and it
   produced the session's key input.** k_drag 0.5 → 1.0 → 2.0: no felt
   difference. k = 10: *"totally changed everything — blowup in pressure…
   a few grenades pressurized the whole main room."* Dump analysis
   (`debug_manual_20260820_044708_kdrag10_molasses.npz`, kept at repo
   root): NOT thermal (T_max 179), NOT a mass mint — **venting death**.
   N fell on 0 of 719 snaps (vs 762/774 at k=0.5); final-window mean wind
   0.017 m/s (vs ~60 at k=0.5); P frozen at 1.357 atm indefinitely.
   Mechanism: linear drag is one rate for ALL flow, so at kd_q ≈ 0.42/tick
   it annihilates the pressure-equalization/venting wind the same as the
   aftermath swirl — the room integrates every grenade deposit and can
   never exhale. Lessons for the design session: (a) the linear dial
   cannot separate "kill the aftermath" from "kill the venting" — the
   discrimination-by-speed argument for quadratic, felt in-game; (b) felt
   differences only appear between k=2 and k=10, i.e. the linear law's
   usable range is narrow and its feel gradient is flat where it's safe;
   (c) any drag law must leave venting/equalization flow alive — add a
   VENTING gate (N falls on quiet snaps) to the drag arc's test set.
   Also magnified here: the known grenade payload item (one snap deposited
   779.8 cell-eq — with venting dead, the room keeps all of it).
4. **Post-pressure retune pass** — item 4 below, unblocked by this arc.

Also queued 2026-08-20: the **skills backlog** and the **bug list** (both
under "Pending — small").

## ✅ VELOCITY-CLAMP arc CLOSED 2026-08-20 — HUMAN-TEST PASS, merged

Erik: *"feel test is perfect."* Record:
`docs/human_test_2026-08-20_velocity_clamp.md`. Both audited defects fixed
(the global-scalar cap → per-cell cap² plane folded at tick entry; the
Chebyshev diagonal leak → exact squared magnitude test): own-cell cap
violations 52,923 → **0**, P_min −1.324 → −0.310 atm, worst cell 433× →
299×, peak single-tick pile-up 328 → 197 cell-eq, lockstep tol 0,
`u_max_hits` structurally 0. Bonus find: `cuda_kick_check` PART 2 had run
its CPU reference drag-dormant since k_drag shipped (consts never passed
the drag dials) — fixed, wind replay restored 120/120. The golden
re-baseline debt (six standing digest reds + the 11 GOLDEN_AGGREGATE flips)
was settled ONCE at this close, with rationale, per Erik's standing ruling.
What it deliberately did NOT fix: the substep ceiling (★ item 1 above).

<details of the original arc brief kept below for lineage>

The mass-books arc closed and handed this one a seed.

**The symptom Erik saw:** *"it is stable but i still get individual pressure
spikes… some individual tiles that flashes yellow or white."* Correct, and it is
NOT mass creation — the mass-books fix landed and destruction now books exactly
1.00 cell of ambient per destroyed wall. Mass is being **piled into** those tiles
by transport far faster than it should, then draining back out. Snap 616 deposits
1.99 cell-equivalents in total, yet a single cell in that event gains **278.34**.

**The measured state:**

```
peak |u|      773.0 m/s      local sound speed ~565   <-- SUPERSONIC
P_min         -1.324 atm                              <-- NEGATIVE, unphysical
worst cell    433.5x ambient                          <-- transient, 1-2 ticks
T_max         737.7           (ceiling 16000 — NOT thermal)
U_MAX = 1000                                          <-- never binds
```

Three symptoms, one cause: **`|u|` exceeding the local sound speed means
advection is running outside what the substep count can resolve.**

**A LEAD, found in a 2-minute read on 2026-08-19 — CONFIRM IT, do not trust it.**
Erik's framing was *"we had clamps but they didn't really work as intended"* and
that looks right. The clamp is not missing; it binds against the wrong number:

```c
// eos_solver.cpp:405-426
t_max_abs_raw = MAX over the ENTIRE FIELD of t_abs   // one global max
ratio_q       = t_max_abs_raw / t_amb
c_local_q     = c_amb * sqrt(ratio_q)                 // ONE scalar per tick
...
u_cap_q       = min(c_local_q, u_max_q)               // applied to EVERY cell
```

`c_local_q` is a **single per-tick scalar derived from the hottest cell on the
map**, then used as the velocity ceiling everywhere — the "local" in the name is
a lie. A fire or explosion anywhere raises the ceiling for every cell, including
cool ones whose true sound speed is far lower. That is exactly how a cool cell
legally carries 773 m/s while its own `c` is 565, and why `U_MAX = 1000` never
binds either.

**If it holds, it unifies two separately-reported defects.** The mass-books arc
also reported, still unruled (see "Also reported by the arc" below), that the EOS
CFL sound-speed max-reduction takes an **unweighted MAX of gas `t_abs`** and that
maximum steers `n_sub`, the substep count for the whole tick. Same mechanism: one
unweighted global max steering **both** the substep count and the velocity
ceiling. Fixing the reduction may fix both.

**What this arc starts with — a better position than mass-books had:**
- a **recorded seed session** with the defect isolated and no mass mint
  confounding it: `debug_manual_20260818_194038_velocity_clamp_seed.npz`
  (775 snapshots ≈ 32 s, kept at repo root — **do not delete**)
- a **discriminator that already works**: deposited-total vs peak-cell-delta
  separates "created" from "piled", and `analyze_blowup_dump.py --mass-books`
  prints both
- **three symptoms to gate against**: supersonic `|u|`, negative `P_min`,
  transient ≫100× cells

Read: `docs/human_test_2026-08-18_destroy_wall_seed.md` §3 (the measurement) and
§5 (what it hands this arc).

**CLOSE-OUT DEBT THIS ARC INHERITS — golden re-baseline.** Six digest tests are
red (`test_b1_signal_bus`, `test_b2_nodes`, `test_b5_airlock`,
`test_b6_logic_golden` ×3). The b6 golden was committed at Arc B (`add8969`,
July) and **three approved behavioral changes have landed since** — energy-books,
the pressure arc's `mg_cycles` 2→8, and mass-books' `destroy_wall` seed. This is
accumulated re-baseline debt, not a new break, and it means the digest gates are
currently providing **zero** regression protection. Erik's ruling 2026-08-19:
**do NOT re-baseline until this arc lands** — today's behaviour still contains the
supersonic defect, and stamping it as golden would enshrine the bug. Re-baseline
once, with written rationale, at this arc's close.

**Carries a known pre-existing defect:** `test_cuda_p64_kick_compression` PART 2
(blast + venting) diverges CPU↔GPU — verified at both `mg_cycles` 2 and 8, so
unrelated to the pressure fix. `docs/archive/e1_p_e2a_asbuilt_2026-08-17.md`
records P-E2a finding it and P-E4's as-built claims it repaired; it has not.
May be the same bug from the GPU side. **Plausibly related to this arc** — it is
the kick/compression kernel, which is where the clamp lives.

**Blocked on this arc:** the post-pressure retune pass (item 4 below) and the
drag-law change (item 3 below). Both would be sized against a flow that is
running outside its resolvable regime.

---

## ✅ Mass-books arc CLOSED 2026-08-19 — merged to main (`cd83d1a`)

**Root cause:** `destroy_wall` seeded a newly-opened cell from the **neighbour
mean**, so a high-pressure room paid out proportionally to its own pressure — and
`find_burst_walls` fires *on* a pressure differential. The valve and the seed
formed a feedback loop. **Fix:** a CONSTANT ambient cell, booked.

| | before | after |
|---|---|---|
| per destroyed wall | 40–130 cell-eq, 52 distinct payloads in 58 events | **1.00, every time** |
| deposits riding wall breaks | 87.7% | **19.0%** |
| scaling with local pressure | linear | **none** |

HUMAN-TEST **PASS** — Erik: *"it feels like the engine is finally starting to
behave now!"* Record: `docs/human_test_2026-08-18_destroy_wall_seed.md`.

**Two things this arc did NOT do, on purpose or by drift:**
- **The grenade payload is still open** — 260 cell-eq per throw, **81% of all
  deposits** in the session (snap 263 = 260.53, snap 503 = 521.06 for two). Erik
  scoped it out; it remains the largest single mass source in the game.
- **P-M1's per-pass mass ledger and P-M2's CUDA twin were never built.** P-M0's
  audit isolated the root cause directly, so the arc reached its fix without the
  instrument it specified as its first patch. The destruction seam IS gated
  (`test_destroy_wall_conserves_mass.py`, `test_destroy_order_pins.py` — property
  gates, so they survive the retune); the **general** per-pass ledger is not.
  Erik's ruling 2026-08-19: acceptable — prefer property gates over goldens while
  systems are still landing, and do not build the full ledger now.

### Water leaves a sealed aquarium under a shockwave (Erik, 2026-08-18 — UNVERIFIED)

Reported alongside the tuning register below; unconfirmed against current HEAD.
A shockwave (air pressure) passing a sealed glass box — a rectangular room with
glass walls, **no tiles destroyed** — puts water OUTSIDE the box. If it
reproduces, this is a **conservation bug, not a tuning item**, and it is the
same FAILURE CLASS as the mass-books arc closed above (mass where there should
be none) in a different field.

**Do these in order; the first two are cheap and either can invalidate the rest:**
1. **Verify the box is actually sealed** — `level_airtight.py` (flood-fill
   hull-seal checker) exists for exactly this. A level-authoring leak looks
   identical from the outside.
2. **Is total water mass CONSERVED when it happens?** This one measurement
   splits the hypothesis space cleanly:
   - *Conserved but relocated* → a transfer path is crossing a solid tile.
     Suspects, all the same shape (a missing solid mask): the pressure-head
     transfer (`docs/water_cuda_head_determinism_fix.md` — that path has had
     trouble before), an advection backtrace that doesn't clamp at solids, or
     a diffusion/smoothing pass that ignores `solid`.
   - *Not conserved* → minting or destruction at the boundary; belongs inside
     the mass-books arc, not beside it.
3. Only then chase the mechanism.

**Characterize it at a FROZEN, RECORDED config, and fix the mechanism, not the
threshold.** `k_drag` sets how much shock amplitude survives to reach the box,
so any drag change (item 3 below) can make this stop reproducing *without
fixing anything*. A threshold fix un-fixes itself at the next dial change.

---

> What needs to be done. Not what's done — git has that.
> Planning window: `roadmap_2026-07-30_rl_push.md` maps these items into tracks
> (the RL push); this file stays the item-level ledger. Staleness-swept 2026-07-30.

---

## ✅ Energy-books arc CLOSED 2026-08-17 (Erik-blessed) — and the three items it queued

The storm line's second half landed. The EOS moves **energy**, not copied
temperature: the mint the storm audit measured (**+7,805 eth per 200 s bench
run**) is closed, transport is one-way non-positive every tick, the hot rail
is gone (`t_max_phys_hits` 2130 → 0, peak T 15984 → 3702), conduction's free
cold-rail leg flipped sign (`t_min_gas` −0.1908 → 0.0000), and traces left the
physics books entirely. Shipped: `k_drag = 0.5`, `k_drag_heat_frac = 0.0014`.
Read `docs/energy_books_arc_close_2026-08-17.md` — canon is folded into engine
chapters 04/05/06 and the arc's working docs are in `docs/archive/`.

Blessed suite state: **48 failed / 2186 passed / 5 skipped**.

**Four items now queued, in order** (item 3 added 2026-08-18, and it *gates*
the retune that follows it):

1. ~~**Pressure / momentum arc**~~ — **CLOSED 2026-08-18, HUMAN-TESTED.** It was
   not physics: the pressure solve ran under-converged at `mg_cycles = 2`.
   Shipped `mg_cycles = 8` → on playground **P_max 103.2 → 1.4 atm**, negative
   `P_min` gone, all rail counters → 0, `n_sub` 8 → 1, **~18% faster**.
   `docs/pressure_arc_root_cause_2026-08-17.md`, canon in engine ch. 04.
   **It replaced itself with the MASS-BOOKS ARC, now CLOSED — see the closed-arc
   section above; the live front is the VELOCITY-CLAMP arc at ★.**
2. **T_abs compression-work patch** (design §2.9, RULING R1) — its own short
   design + critique round + HUMAN-TEST. Step 4c multiplies ambient-*relative*
   T, so below ambient it doesn't merely omit physics, it **inverts** it
   (compression freezes cold gas — the cold-rail window's engine). The honest
   form is `T_new = (T + 290)·(1±w) − 290`, which also restores the missing
   acoustic thermalization the §8 bound names. Feel-adjacent: breach
   rarefaction becomes genuinely cold (~97 game-deg at the clamp vs 0 today).
3. **Drag law: linear → quadratic `k_drag`** (Erik, 2026-08-18) — a MODEL
   change, and it must land BEFORE the retune below: it changes the velocity
   field every other dial is tuned against, and it retires the `k_drag = 0.5`
   sizing outright (a quadratic `k` has different units).

   *Why.* Today's fold is Stokes-linear — `u *= (1 - kd_q)`, one rate for every
   speed (`cpp/src/cuda_kick_compression.cu:219-252`, CPU twin in
   `eos_solver.cpp`'s kick loop). At `k_drag = 0.5` that is a ~2 s e-fold
   applied equally to a 40 m/s blast front and a 0.5 m/s convective curl, so
   the dial that kills the storm also flattens the slow structure. Real air
   drag is quadratic (~ρu²): it barely touches slow flow and bites hard on
   transients — exactly the discrimination we want, and plausibly half the
   answer to the smoke item under 4 below.

   *Cost — Erik asked; the honest read is CHEAP, for one structural reason.*
   The energy bookkeeping downstream (`du2_raw`, the heat deposit, counters
   5–8) is computed from `ux_old` vs `ux` and is therefore **law-agnostic** —
   it does not care how the shrink was derived, so the whole energy-books
   machinery survives untouched. The change is the two lines that build
   `kk_drag`, in the two places that already carry the block verbatim (CPU +
   CUDA). Concretely:
   - Needs `|u|` unconditionally. `sqrt_q16_dev(rad)` is already computed ~15
     lines above in the |u|-cap branch, but only *inside* that branch — so
     this costs one extra Q16 sqrt per gas cell per tick. Measure it; almost
     certainly fine, but it is the one real perf question.
   - The `(kd_q < FP_ONE) ? FP_ONE - kd_q : 0` guard already has the right
     shape for the new fold going negative at high `|u|` — reuse it; never let
     the fold reverse the velocity.
   - **Quantization is the subtle part, and it is load-bearing.** At small
     `|u|`, `k·|u|·dt` quantizes to 0 and drag vanishes for slow flow. That is
     the DESIRED behaviour — but it must be a *named, measured* threshold
     ("drag is identically zero below X m/s"), not an accident discovered
     later. Derive and record the cutoff speed.
   - CPU↔GPU lockstep on the sqrt is already exercised by the |u| cap, so the
     determinism risk is low. Still one deliberate golden re-baseline.

   *What it does NOT buy:* the new `k` needs sizing from scratch, and
   `k_drag_heat_frac = 0.0014` was measured at `k_drag = 0.02` under the OLD
   law — that measurement dies with the law. Both belong to item 4.

   *Scope ruling (Erik, 2026-08-18):* "pretty neat if it's not too much work…
   if it's really hard to implement and cost way too much we could also skip
   it." So if the sqrt cost or the lockstep turns ugly, this is **droppable,
   not load-bearing** — fall back to keeping the linear law and sizing it at
   item 4.

   **THREE SEQUENCING RULINGS (Erik, 2026-08-19) — read before starting:**
   - **A design session comes first**, house style: design doc → adversarial
     critique → patches with gates. Do NOT open this as a straight
     implementation task. Erik: *"i would like to have a nice design session
     before that implementation."*
   - **The EXPONENT is an open design question, not a settled 2.** Erik asked
     whether real drag is quadratic or quartic. Physically it is `u²` in the
     inertial/turbulent regime (`u¹` is Stokes; there is no `u⁴` fluid-drag
     regime) — but two things make "just use 2" premature. (a) Dissipated
     *power* for quadratic drag is `F·u ∝ u³`, and since our drag carries a
     heat counterparty we are already reasoning in energy terms, where the
     exponent is one higher. (b) Transonic drag genuinely steepens past `u²`
     as `C_d` climbs near Mach 1. **Q16 cost is the constraint on making the
     exponent a free dial:** `u²` needs `|u|` (one sqrt) × `u`; `u⁴` needs
     `|u|³`, and Q16.16 overflow headroom shrinks fast. Derive the headroom in
     the design session before promising a general `u^p`.
   - **DO NOT size this against today's supersonic flow.** The P-M3 human test
     measured peak `|u|` = 773 m/s against `c_local` ≈ 565
     (`docs/human_test_2026-08-18_destroy_wall_seed.md` §3) — the engine is
     running outside the regime the substep count can resolve. That is the
     **velocity-clamp arc's** defect, not a drag-law question. Tuning drag to
     tame supersonic flow would use a dial to paper over a bug — the same
     error pattern as the aquarium and smoke items. **The velocity-clamp arc
     must land first**, then the flow is subsonic and `u²` is defensible.
   - **Visual tuning of this dial waits on the smoke fix** (item 4's smoke
     bullet). Erik: *"i would also like to tune quadratic drag visually — and
     possibly after we fix the saturation of the smoke so we can actually see
     anything."* The smoke cloud IS the instrument for reading the velocity
     field by eye; with it saturated to black there is nothing to tune against.
     So: smoke diagnostic+fix → drag law → visual sizing.

4. **Post-pressure retune pass** — one sweep after the pressure arc lands:
   **fire anchors** (`peak time` 2.29 → 2.00 min fell out of its 2–5 min band;
   `peak I`, plateau T and `fire death` were already MISSing for pre-existing
   reasons — `k_grow`/`k_die`/`wall_damage` own them), **`k_drag`** (0.5 is a
   *starting* value Erik picked at the HUMAN-TEST, explicitly not a tuned one),
   and **the arc's one declared red**,
   `tests/test_p3_direct_e2e.py::test_directional_spray_cone_follows_facing`
   (with damping live a spray cone rides shortened wind; feel-adjacent, left
   honestly red rather than xfail-papered).

   **Added 2026-08-18 (mass-books arc, measured — fire is mistuned at BOTH
   ends).** Surfaced by repairing the fire tests, which had been masked by a
   signature drift since 547fb12 (2026-07-24) and were therefore not reporting:
   - **Ignition is far too fast.** From a 0.5 seed the centre cell goes to full
     intensity in **under 5 ticks** (~0.2 s) in every config; from a 0.1 seed it
     is capped by tick 15. Erik, on being shown it: *"maximum intensity in 5
     ticks is NOT what i want, but we are not at tuning yet, we need to make
     sure all the systems WORK before tuning."* Owners: `k_grow`, and the
     intensity cap.
   - **Extinction is far too slow — the "death wall".** `k_die` 2.0 → 0.008
     leaves fires taking **~2,334–2,400 ticks** to die against test horizons of
     200. This is the single largest cause of the remaining red fire tests, and
     it is why `test_s3b_fire_determinism.py:113`
     (`assert la[-1] == 0, "the fire never extinguished"`) is red — a
     non-vacuousness guard correctly reporting that no extinguish is exercised.
   - **Dead levers to re-decide, not just re-dial:** `k_wind_strip == 0.0` since
     2026-07-23, so wind can only fan a fire, never blow it out — the blow-out
     term is identically zero and its test asserts a mechanism that no longer
     exists. `wall_damage` 0.4 → 0.03 leaves burnout at hp 2.26 after 500 ticks.
   - **Test horizons must be re-derived from the retuned dials**, not nudged:
     several fire tests encode a 200–500 tick expectation from the pre-retune
     regime. `test_s3b_fire_determinism::test_cross_config_self_match` already
     had its measurement window re-derived this way (0.1 seed @ 10 ticks) and
     carries the measured saturation table in-file.

   **Added 2026-08-18 (Erik) — smoke saturates to black FAR too early.** The
   whole cloud reads as a flat black mass with only a thin transparent rim;
   the patterns that should be among the prettiest things in the game are
   invisible inside it. Erik: *"this is really sad because the patterns have
   potentiality to be so beautiful."* Render path is `tau = base_absorb_scale ·
   plume_k_scale · Σ(k_s·ρ_s)` → artistic remap `tau_p = a·tau^b` → `alpha =
   1 − exp(−tau_p)` (`renderer/gas_medium.py:150-159`; the tau-space curve
   replaced the retired `smoke_render_gamma`). Because of the exponential,
   `tau > ~4` means `alpha > 0.98` and ALL interior structure is gone — only
   the rim, where tau sweeps 0→3, still carries pattern.
   - **Split this BEFORE dialling anything:** is tau too large because the
     optical constants are too large (render), or because the sim is producing
     too much smoke MASS (sim)? With the mass-books arc having measured a
     2.15–2.20× mint, the second is live — and pulling `tau_curve_a/b` down to
     compensate for
     over-produced density would paper a sim bug with a render dial and
     un-tune itself the moment mass-books lands. **Diagnostic first (a tau
     histogram across a real cloud), taste second.**
   - The taste half — where the transition band sits and how wide it is — is
     Erik-at-the-screen work, not a number an agent picks.
   - **Coupled to item 3:** smoke advects on the velocity field `k_drag` damps,
     so linear drag's flattening of slow convection may be suppressing the very
     structure the opacity curve is then failing to show. Expect the quadratic
     law to move this problem before any optical dial does.

**Also reported by the arc, not fixed, awaiting a ruling** — three consumers
that read raw, N-unguarded gas temperature. The serious one is **sim-affecting**:
the EOS CFL sound-speed max-reduction (`eos_solver.cpp:347-351` + its CUDA
twin) takes an unweighted MAX of gas `t_abs` and that maximum steers `n_sub`,
the substep count for the whole tick — so one thin-N cell with a
rounding-dominated T can change the substep count everywhere. (Contrast `p*`
in the same function, which is N-weighted and benign by construction.) The
other two: the `temperature` sensor's area-mean (`sensor_accessor.py:154-175`,
wire-able by a level author, covered by no test) and render fire-light
selection (cosmetic). A threshold change is feel-adjacent — Erik's call.

---

## ⏸ Superseded pointer — energy-books merge, 2026-08-17 (kept for lineage)

> NOT the live front. The live front is the ★ VELOCITY-CLAMP arc at the top of
> this file. Pressure arc: CLOSED. Mass-books arc: CLOSED.

(Superseded two stale pointers at this merge: main's "storm audit running" — the
audit finished and the arc it spawned has now closed — and this branch's own
"Breach paused 2026-08-04". The live thread is the three queued items above,
in order; the pressure arc leads and starts with an audit, not a dial.)

### Map of docs/
- **Canon (live-edited, source of truth):** `docs/architecture/**`.
- **Open items:** this file (`docs/TODO.md`) — item-level; `docs/priority_ledger.md`
  — the standing stack, coarser, in order.
- **Capture (append-only, dated):** everything else in `docs/` — design docs,
  seeds, critiques, audits, session records. Archived to `docs/archive/` at
  arc close, unchanged (`git mv`); see `docs/archive/ARCHIVE_INDEX.md` for
  what moved and why.

### Where things stand
The fire arc and the temperature-scale unification arc (P-K0–P-K5) are both
**complete and merged to main**, Erik-played and blessed (P-K5). The 2026-08-03/04
audit's Patch A (9 bounded items, decision-free) **landed** — commits A1–A9
are on main. The docs those investigations produced are archived; see
`docs/archive/ARCHIVE_INDEX.md`'s "2026-08-14 — fire arc + temperature-scale
unification arc close" entry for the full list and what superseded what.

**The canonical map**, as shipped (`[physics.temperature_scale]`):
`K = 293 + 3·T_game`; EOS keeps a named, deliberate exception (`eos_t_amb_k =
290`); `phi_exp` exists as a named (still frozen at 1/3) dial. Full as-built
record, including what each old doc's formula/claim is now superseded by:
`docs/temperature_scale_unification_design_2026-08-13.md` §10.

**Two of the four "DECISIONS WAITING ON ERIK" from the 2026-08-04 audit are
now TAKEN:** the two Kelvin maps are unified (above), and `phi_exp` naming is
done (value still frozen). This did not change storming dynamics — the EOS
stayed byte-identical through the arc by construction.

**Storm-damping session — STAGED, audit running.** Erik's ruling
(2026-08-14): the session runs the **momentum/energy ledger audit BEFORE
choosing any damping dial** (reproduce → two-room bench → budget audit incl.
the 2026-08-14 blowup dump → explain the `wave_absorb` 0.002–0.01 instability
window and the density-division amplifier → then dissipation). A parallel
agent is running that audit now — **do not touch `docs/storm_audit_*.md` if
one appears while this is in flight.** `docs/fire_atmosphere_oscillation_analysis_2026-08-03.md`
(root-caused the storming: connected geometry with zero momentum dissipation,
not fire temperature or flicker) is the **live input** to that session — kept
at `docs/`, not archived.

**Still open, not yet decided:**
- **Interior air damping** — the storm audit above is the precondition for
  this; see its ruling.
- **`cool_shift` 9 vs shipped 5** — the fire calibration runs off its own
  anchor (`config.toml:317,602`); flagged "RE-TUNE AT P-R5", not yet resolved.
- **P-F1b branch cleanup** — its calibration values are already promoted into
  the shipped TUNE set (P-K0); the branch (`origin/pf1b-recalibration`) itself
  is now closable.
- **Un-xfail the ignition handoff test** — `test_fire_heat_source.py::test_full_chain_heat_ignites_air_separated_wood`
  is `XPASS(strict)` now that the handoff it signals has landed; Erik's call
  whether to un-xfail it (`temperature_scale_unification_design_2026-08-13.md`
  §10).

**Deferred (post-tuning, Erik's initiative, §7 of the design doc):**
- **Golden-suite co-design with Erik** — one deterministic canonical scenario
  exercising the full sim surface (standing water, a living fire, rooms at
  different pressures, an explosion), replacing re-baseline-from-whatever-state
  with a suite designed to catch regressions everywhere, not just the quiet
  systems. Own design session. **Re-confirmed by Erik 2026-08-20** (at the
  velocity-clamp close's re-baseline): the current goldens still carry the
  old coverage gaps (no water, no doors, no burnables) — re-stamping them at
  arc closes is fine for now, but the co-design happens once the physics
  engine is fully fixed, WITH Erik, and replaces this set.
- **Wind assessment** and **fire-vs-wind + O2-suffocation tuning** — queued
  behind the storm-damping session (wind/damping share the momentum surface).

**Two findings worth carrying to other projects** (from the 2026-08-04 audit,
still true): gate coverage — not language — predicted code quality here; and
*in an agent-built repo, a technique that is not in a file agents must read
does not exist* (which is why the dormancy discipline propagated across 15
arcs and `_ep`/`RC_HD` did not).

---

## Planning burst 2026-08-08→10 — three new capture docs (post-fire-retune queue)

Erik's vacation planning pass. All three are **capture/design docs on main**;
none starts before the fire/atmosphere lid closes (Erik's sequencing: stable
fire + radiation first, then basic fires working, then explosions). Scope creep
deluxe, by choice.

- **Level generation v0.1** — `docs/breach_levelgen_design_v0.1.md`
  (2026-08-10, claude.ai sessions). Graph-grammar-first levelgen; nine LOCKED
  decisions L1–L9 (planarity by construction, pressure cells as grammar
  concept, LLM-authored offline recipes, (ruleset hash, seed) reproducibility).
  Design-only until fully specified. **NEXT work package = the vocabulary page
  + consolidation pass vs the existing level generator (§4, a Claude Code
  task); then the embedding session (§5).** Own roadmap in its §7.
- **Tsetlin-machine / engine architecture** — `docs/breach_tm_architecture.md`
  (2026-08-09, airplane session). TM primer + hazard-prognosis self-labeling,
  hot/cold GPU split + command-buffer/status-mirror/event-queue membrane,
  three-tier mind hierarchy, behavioral dials (clause truncation = intelligence,
  vote bias = temperament), room graph as first-class citizen. **NEXT (its §16
  step 1) = communication-contract doc: walk the repo, annotate each item
  existing vs aspirational — the gap list becomes the implementation plan.**
  The room graph is the shared spine with levelgen (five consumers).
- **Brainstorm 2026-08-08** — `docs/brainstorm_2026-08-08.md`. (1) enemies
  changing behaviour on damage (the enrage-trigger dial → focus-fire vs
  spread-damage tactics); (2) reward-vector-as-personality (curriculum/
  annealing, potential-based shaping, OpenAI-Five/AlphaStar precedents) —
  serves the RL push directly; (3) water-pass-2 seeds (trapped-air-pocket
  experiment, §5.5 unit↔water couplings, swim/float/drown) — feeds the
  roadmap's open "aquarium/water arc scope" question.
- **Generic explosion design (Erik, 2026-08-10 chat — capture, needs a design
  pass).** Instead of per-weapon explosion tuning: ONE parameterized explosion
  archetype (yield/radius/heat/pressure profile), and weapons — grenades,
  bazooka shots, future ordnance — are *scaled instances* of it. Goal:
  variation without balancing every weapon by hand; later hook for automatic
  balancing (self-play / headless sweeps). Slot: with the explosions item in
  Erik's sequencing above, after basic fires work.
- Mission/beastiary notes updated same burst: `docs/missions/missions.md`
  (mission-1 comments: ship floats in space not sea, org rethink, stealth-lean
  phase 1–2) and `docs/beastiary/beastiary.md` (zombie-as-a-STATE inheriting
  the victim's attributes + equipment; robots immune; extract the claude.ai
  bestiary entries via egregore someday).


## Waiting on Erik (human-gated)

- **Armory tuning session — AFTER mission 1 exists (re-scoped 2026-07-21)** —
  W6 merged with standard values (PR #3, `f482131`); Erik's call: balance is
  meaningless against an empty playground, so the grand tuning session waits
  until weapons → units → enemies → a first mission are all in place, then
  tune against real encounters. Tool: **N** cycles the selected unit's weapon
  (walkthrough `docs/playground_guide.md` §9). Standing dial list: the
  chain-stun pair (`rof_interval_seconds` vs `status_seconds` on
  `[weapons.arc_baton]`), the plasma-vs-zombie resist wash (bullet ×0.25 then
  HEAT ×4 ≈ no-op), flamethrower feel at the 10 m / 20 m meter-based ranges —
  plus whatever mission 1 teaches. Quick residual: 30 s look at the W6 jet
  fans + 3D marines rendered together (never exercised in one window).

- **Temperature-scale unification — P-K5 PASSED, arc merged (2026-08-14,
  built on `thermal-mass-axis`)** — one canonical game-T→Kelvin map
  (`[physics.temperature_scale]`, `K = 293 + 3·T_game`) backs radiation,
  render, and (via the named exception `eos_t_amb_k = 290`) the EOS pressure
  calibration; EOS byte-identical through the arc. Erik played and blessed
  the fire 2026-08-14; during that session the recorder captured a live
  multi-room storming blowup (`debug_blowup_20260814_015714.npz`) — evidence
  for the storm session, explicitly out of the arc's scope. Full record:
  `docs/temperature_scale_unification_design_2026-08-13.md` §10.
  **Storm-session preconditions updated:** `phi_exp` exists as a named
  (still frozen) dial; two of the four parked decisions are taken
  (Kelvin-map unification; `phi_exp` naming); **Erik's ruling 2026-08-14:
  the session runs the momentum/energy LEDGER AUDIT before choosing any
  damping dial** (reproduce → two-room bench → budget audit incl. the
  blowup dump → explain the wave_absorb 0.002–0.01 instability window and
  the density-division amplifier → then dissipation). Still open from the
  arc: un-xfail the ignition handoff test (Erik's call).

## Next gameplay arc — momentum / earned sprint / unit collision (2026-07-30)

**Design sketch: `docs/inertia_and_sprint_design_2026-07-30.md`.** Captured
from Erik at the close of the OnePhaseWEGO human-test; needs its own design
session before any build.

**Erik's sequencing ruling: do NOT start with the full inertial model.** Start
with the simple rule — sprint as an EARNED STATE (speed up, navigation down)
after ~4 s of continuous movement, so normal single-round play is unchanged and
only multi-round runs cash in. It is a state machine, not a physics model, so
it can be felt and thrown away cheaply. Real momentum (velocity vector,
acceleration from mass/strength) stays behind it.

Unit collision is needed by both and is the part Erik wants "intricate":
bodies are already solid (`timeline.occupied_by_unit`, 2026-07-30), and
`stability` already exists as the knockdown threshold for shockwave push —
a body collision is the same shape of event and should reuse that rule. A
stationary unit braces; a moving one has already committed its balance.

Two of Erik's framings worth not re-deriving: *certainty decreasing with
distance is a feature*, and *a predicted collision is self-cancelling* (if the
planner sees it, the player routes around it) — so "exact except collisions" is
a far stronger guarantee than it sounds. Momentum also feeds the ML-animation
track: `velocity` + collision impulses are exactly the signal physics-driven
animation wants, one-way, render-only.


## Open threads — cross-arc index (2026-07-22, "don't forget")

One place to see every loose end left by the recent burst of simultaneous
patches (Arc B, W6, Fire & Heat, S8c, animation). Each points to where the real
detail lives; this list is the map, not the territory.

**The big fork that gates other things:**
- **Control-scheme decision** — WEGO vs direct gamepad control (one marine).
  Gates: unit-initiated interactions (`button`/terminal are inert until then),
  the **manual airlock buttons** below, and any AP/phase assumptions. Nothing
  new should bake in AP/phase until this is decided. (Entity design §3d.)
  **Update 2026-07-23:** resolved into the *modularity* split (both schemes
  coexist) — design `docs/archive/control_modularity_design_2026-07-22.md`. **The whole
  control-modularity line (P1–P3 + free-aim shooting F1–F3) is MERGED to main**,
  human-tested and blessed by Erik ("the controller scheme basically works... i
  felt joy shooting zombies"). WEGO byte-identical throughout. Follow-up polish
  below.

**Modular control / action variant — MERGED 2026-07-23, follow-up polish (LEAST-URGENT):**
The direct-control + free-aim slice is in main and works. Remaining items are
polish/feel + parked engine work; none block anything (WEGO untouched). Resume on
Erik's signal — several want their own dedicated session.
- **Flamer / spray "held stream" feel — OWN SESSION (Erik's lean).** Today the
  spray archetype fires a fixed ~1–2 s burst (`burst_ticks`) with the aim latched
  for the whole burst — under direct control you can't re-aim mid-burst. Intended
  feel: a *continuous stream you hold as long as you want / have ammo for* and can
  sweep. Directional spray currently kicks a fixed burst (`start_spray_burst_directional`,
  aim latched via `spray_aim_angle`); the fix is hold-to-fire + per-tick re-aim
  while TRIGGER is down. Delicate weapon → its own design/build pass.
- **Grenade button remap.** Grenade came out on the wrong physical button. A
  button-index logger now prints `[gamepad] raylib button index N pressed`
  (`control_gamepad.py`, human-test debug) — press each pad button, read the
  console, then remap `GamepadDirect` (throw/use/weapon-cycle) to the right
  indices. Weapon-cycle is currently on **keyboard N** (works under gamepad).
- **Action-variant crash** — seen once by Erik in the first human-test; prime
  suspect `_aim_fire_order` is now DELETED and a stress test raises nothing, so
  likely resolved. Confirm if it ever recurs (grab the terminal traceback).
- **Door ↔ unit occupancy — ENGINE-level fix, rule A+B (decided, unbuilt).** A
  unit under direct control can park its footprint in a door and get stuck (WEGO's
  A* never stops on a door, so latent there). Layer verdict: engine —
  `is_passable_block` is shared by A* and the direct-move branch, so one fix serves
  both schemes. **Rule A:** a door may not close onto an occupying unit
  (stays/re-opens/close blocked). **Rule B:** collision never permanently traps an
  already-placed unit (a unit whose current footprint is blocked may always move
  toward a less-blocked cell). Do NOT band-aid in `_step_move_dir` only.
  Feel-adjacent → HUMAN-TEST gate.
- **P4 keyboard+mouse direct variant** — deferred; feel-adjacent (never
  auto-merge) and untestable while Erik has no mouse.
- **Canon fold (arc close):** fold the as-built (Ruleset split, ContinuousRealtime,
  ControlSource/`--control`, per-tick intents, free-aim directional fire) into the
  canon chapters (esp. `mechanics/04_turn_and_control.md` — also fix its stale
  12 Hz table → 24 Hz) and archive the two design brainstorms to `docs/archive/`.
- Cleanup pending: delete the merged `control-modularity` branch + worktree
  (local + `origin`) once canon fold is done. *(The old "origin/main may be
  behind" note is resolved — verified in sync 2026-07-30.)*

**Arc B (entity logic layer — DONE + merged 2026-07-22), its two parked riders:**
- **Resident sensor-gather kernel** — the §5a `(n_sites × n_channels)` int32 GPU
  gather buffer. Arc B stubbed the accessor to the CPU mirror; interface is
  FROZEN, only the GPU impl is missing. Belongs to the **S8 / CUDA-residency
  line** (throughput for batched training — nothing broken without it). **Now
  unblocked** (S8c landed). ↓ "Arc B follow-ups".
- **Manual airlock buttons ("airlock v2")** — gated on the control-scheme fork
  above. ↓ "Arc B follow-ups".

**Weapons (W6 — merged):**
- **Armory tuning session** — deferred until weapons→units→enemies→mission 1
  exist, then tune against real encounters. ↑ "Waiting on Erik".

**Fire & Heat (active arc, NOT a loose end — living plan
`docs/plan-for-tuning-and-graphics.md`):**
- Fire B1 blackbody lights shipped; **Fire B2 smoke-honesty** in flight
  (`fire_b2_smoke_honesty_design_2026-07-21.md`). Fire *tuning* — "burns too
  easily" + the fire→pressure link — is on that plan. (Tracked there; listed
  here only so it isn't mistaken for forgotten.)

**S8c (fire-heat batch — item 1 merged):**
- **Items 2 & 3 DEFERRED** (accepted gaps): render CUDA-GL interop + recorder
  kernels. Part of the S8-optimize line. (`s8c_items_2_3_deferred_2026-07-21.md`.)

**Physics riders on the books (ledger stack #1):**
- **Blast-pressure-threshold material column** — direction decided 2026-07-19;
  impl + tuning is a chat-sized HUMAN-TEST rider (`physics.py:104` blast-tuple
  wart retires here). · **Dust-stirring shockwaves** — dusty-ground flag +
  wave_p threshold → smoke injection. · **Post-EOS doc consolidation.**
- **Grenade energy-budget retune (Erik, 2026-07-30)** — grenades currently dump
  too much static pressure into the room (blowup-adjacent; cf. the fire→pressure
  link under the Fire & Heat tuning session). Re-split the deposit: HEAT as the
  primary payload, less raw over-pressure, and evaluate seeding an initial
  radial WIND (velocity initial condition) so the shockwave is carried by
  momentum — a directional/impulse dial instead of a pressure spike. Check how
  a u-field injection couples with the (separate, post-EOS-confirmed) wave_p
  blast system before tuning. Feel-adjacent → HUMAN-TEST; natural companion to
  the fire-tuning session.

**Animation track:** the weapon/grenade + shockwave-push question (just folded
from TODO2.md) + the appearance/skin-pipeline items — all ↓ "Animation".

**Housekeeping:**
- The `.claude/worktrees/arc-b-logic` directory lingers on disk (Windows
  file-lock after the arc merge); git is clean — `Remove-Item -Recurse -Force`
  it after a reboot.


## Animation / character-render track (3D marines shipped 2026-07-21)

**Shipped (arc `anim-phase0-3d-marines`, merged 2026-07-21):** render-only 3D
marines/zombies over the 2D world (toggle **M**, default off), lit by the
raycast light field so they match the ship, 2× scale, blob shadows; a
tangent-free normal-map *capability* is present but default-off (the Quaternius
model is untextured, so nothing to reveal). Docs: `marine_shader_foundation_design_2026-07-20.md`,
`research/ml_animation_litsearch_2026-07-20.md`, `procedural_animation_brainstorm.md`.
All render-only — no sim/determinism surface, auto-skipped in headless training.

**Wanted next (Erik, 2026-07-21 — capture, later work):**
- **Marine appearance system** — a clean per-unit *visual profile*: one model +
  animation set + skin per unit type, with **variation** (later). Today it's a
  single shared model + a flat group tint (green marines / red zombies via
  `colDiffuse`). Generalize to `unit-type → {model, skin/material, clip set,
  gait params}` so new looks are data, not code.
- **Zombies look like their victim** — when a unit turns into a zombie
  *mid-match*, it **keeps its own skin/model** (appearance unchanged) but
  **swaps to the zombie animation** (shambling gait), optionally with a
  **"bloodied" overlay**. **Pre-placed** zombies (spawned as zombies) get a
  **dedicated zombie skin**. So: turning = animation swap (+ optional blood),
  NOT a skin swap; pre-placed = special skin. (The current green/red tint is
  fine for now.)
- **Weapon & grenade animations + shockwave-push behaviour** (Erik, folded from
  TODO2.md 2026-07-22) — marines should **carry at least one visible weapon**
  and have a **rifle carry + shoot/fire** animation, plus a **throw-grenade**
  animation. Open question: what should a marine do when **pushed by a
  shockwave** (they already get pushed today) — maybe
  *hunker down* a little on a small push; on a big push, unclear. **Meta-
  principle (Erik, load-bearing for this whole track):** the **ML path is the
  priority** — he has no ambition to be an old-school animator, and much
  hand-authored prep may be *thrown out* once ML drives motion. So weigh every
  animation task by "will ML redo this?" — use the right tool for the job, do
  the minimum that reads well now, and stay open to letting the ML track own
  reactive motion (push/stagger/limp) rather than scripting it.

- **Retire the M toggle + sprite path** (Erik, 2026-07-21) — make the 3D marines
  the **default and only** unit render; drop the old 2D sprite fallback (the
  `use_3d_units` toggle / `M` key / `UnitSprites` unit path) once confident. The
  old sprites won't be needed anymore.
- **Skin / appearance-asset pipeline** (Erik, 2026-07-21) — we need real
  **textured skins** (marine, zombie, variations) to unlock the visual-profile
  system above *and* the already-built normal-map capability (P2). The current
  Quaternius model is untextured. Evaluate **AI generation** — mesh-texturing
  tools that paint albedo + normal/PBR onto the existing rig's UVs (Meshy-style
  AI texturing), or text-to-3D for whole rigged+textured models — vs
  hand-authored. A focused tool eval (like the shader lit-search) is the right
  first step when we pursue it.
- **Body-part damage → animation/behaviour hook** — Erik's `01_units.md` note
  (commit `350179c`): body parts carry hp/damaged states that drive a different
  animation, speed, even behaviour (limping) via the ML animation system. Ties
  render ↔ mechanics; needs refining/planning.

**Deferred fixes/items from this arc:**
- **Move-order animation bug (OURS, not the command system)** — when one marine
  gets a move order, ALL marines' models play the WALK clip while staying put.
  `UnitModelRenderer`'s motion inference mis-selects "walk" for stationary units
  (likely `move_path` non-empty on all during planning, or the position-delta
  test). Fix in the clip/motion-inference; its own session.
- **P2 real asset drop-in** — the normal-map capability is inert until a
  textured/normal-mapped marine asset exists: drop it over
  `assets/models/marine/marine_normal_PLACEHOLDER.png`, flip
  `MARINE_USE_NORMAL_DEFAULT` (or `marine_shader.set_use_normal(True)`), tune
  `MARINE_NORMAL_STRENGTH` — no code change.
- **`fire`/`dead` clips dormant** — wired in `CLIP_MAP` but unused (firing not
  inferred from sim; dead units skipped for sprite parity). One-line extensions.
- **GPU skinning (perf lever, deferred)** — CPU-skinning soft ceiling ~20 units;
  when counts grow, rebuild the raylib binding with `-DSUPPORT_GPU_SKINNING=ON`
  and flip the `_draw_one` seam (the fragment/lighting half is unchanged). Not
  needed yet.
- **Ceiling-lamp z** — lights carry a *constant* vertical component
  (`u_light_z`); a true overhead shaft wants per-lamp/per-tile z. Lighting
  nicety, low prio.

## Pending — small (background, queue up next session)

- **Bug list (started 2026-08-20 — Erik wants known bugs tracked in one
  place; graduate items into arcs when picked up):**
  - **Fires burn in hard vacuum** (Erik, 2026-08-21 HUMAN-TEST of the T_abs
    arc; measured in `debug_manual_20260821_143234/143409.npz`: crates at
    fire intensity ~0.37 with N_total = 0.0000 for 2335+ snaps; 78–83% of
    all burning snap-cells sit below a quarter of ambient air). ROOT CAUSE
    (code-verified): the continuous-O2 law (2026-07-24) gates sustain on
    the O2 MOLE FRACTION X = n_O2/n_total — deliberately, to fix the
    "density trap" (thermal expansion suffocating fires) — but venting
    removes O2 and N2 TOGETHER, so X stays ≈0.21 down to zero molecules
    and o2f never crosses X_ext = 0.13. The fraction law needs an absolute-
    availability companion (composition above the flammability limit AND
    enough molar O2 to feed the flame) designed so the density-trap fix
    survives. NOT the T_abs arc's doing (law untouched; the cold overlay
    just made evacuated rooms visible enough to notice). Owner: the fire
    retune / O2-suffocation session already queued under item 4 above —
    graduate this entry into that session's design list.
  - **Pressure-burst walls keep their graphics** (Erik, 2026-08-20 feel
    probe, post-velocity-clamp build). A wall destroyed by pressure
    (`find_burst_walls` → `destroy_wall`) stays visually intact — the only
    tell is smoke flowing through the gap. Sim is correct (flow passes);
    the render/bake layer isn't invalidated on destruction. Suspect: baked
    art/tile layer not refreshed when `solid` flips outside the editor
    path. Render-only.

- **Skills backlog (Erik, 2026-08-20).** Procedures live in skills, not docs
  (master CLAUDE.md rule) — several are overdue. Erik's meta-ruling: *take
  time to get each workflow genuinely right — we'll use them over and over.*
  - **run-game skill** — launch the game per machine (the `<env-py> main.py
    --cuda` line, worktree-aware, common flags like `--res`, where
    stdout/stderr land, how to launch detached for a HUMAN-TEST).
  - **level-editor skill** — launching + driving the Arc-C editor.
  - **level-generator skill** — once levelgen exists (design v0.1 is
    design-only).
  - **bug-report skill** — how to record a bug (repro, build/branch, dump
    if physics, where it goes in this list) so feel-session findings don't
    evaporate.
  - **arc-close skill** — the close ritual (golden re-baseline steps +
    archive + tag + merge) AND the workspace handling: `main` lives in ONE
    worktree at a time, so the close must put it where Erik's VSCode sits
    (solved manually 2026-08-20, must be encoded). The ritual also runs
    **egregore-collect-transcripts** as a close step: verified 2026-08-20
    that transcripts live in `~/.claude/projects/` OUTSIDE the repos and
    survive worktree/branch deletion — the only loss mode is the
    collector's 60-day window, which collecting at every close defeats
    (details in that skill's 2026-08-20 note).
  - **Skill-audit session with Claude** — walk the repo's recurring
    procedures and decide what else deserves a skill. Seed candidates:
    CUDA build per machine, recording + analyzing an F8/blowup dump,
    HUMAN-TEST record-keeping, and Erik's question: documentation
    navigation (canon vs capture vs archive map).

- **Fire & Heat tuning session (Erik, 2026-07-21, after B1 merged) — DEDICATED
  SESSION.** B1 (black-body overlay + brightest-K fire lights) merged and the
  look is blessed ("much better"), but needs a tuning pass:
  - *Render mapping:* fires read too white at the default `k_temp_to_kelvin=2.0`
    (saturates to white by ~T_game 3000). Dial `k_temp_to_kelvin` DOWN / raise
    `kelvin_ref` so white is reserved for extremes (config `[render.blackbody]`).
  - *Sim — EVERYTHING BURNS TOO EASILY (Erik 2026-07-21, headline):* the whole
    room lit and went white. Primary suspect `k_fire_heat = 1600` (the radiation
    heat deposit — "ignites too fast"); then `ignition_temp = 300` (raise),
    `range_per_intensity = 3` (shorten reach), `o2_threshold = 0.01` (raise the
    ignition O2 gate). Sim-side, NOT render — B1 only reveals it. Verify with a
    headless probe (ticks-to-full-involvement).
  - *Fire↔pressure link:* Erik hit a `[recorder] BLOWUP DETECTED` during the B1
    session. NOTE blowups are PRE-EXISTING (dumps back to 2026-07-12; several
    on 07-21 from W6 explosion testing) — not B1. But over-ignition plausibly
    FEEDS it: a fully-involved room = many `fire_pressure_gain = 0.15` plume
    over-pressures summing → blowup trip. Fixing ignition upstream should ease
    the pressure symptom; confirm during the session.
  - *Diagnostics wanted (Erik's preference over more shortcut keys):* make ONE
    dedicated fire-tuning level and hardcode a per-tile value readout into the
    launch script — OR a small "all values of the hovered tile as a table"
    (T in game-units + pseudo-Kelvin, fire intensity, material, ignition_temp /
    ignited flag). Render-only (reads gmap fields). Deferred on purpose; build it
    as the session opener, not now.

- **Blast-tuple wart (Arc A rider, A6, 2026-07-19) — direction DECIDED
  2026-07-19 at physics close-out:** `apply_explosion`'s structural wall
  damage gates on the hardcoded tuple at `physics.py:104`
  (`MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_DOOR_CLOSED`) instead of the material
  table. Fix: NOT tuple-widening — a per-material **blast-pressure-threshold
  column** in the material table (damage only when local blast amplitude ≥
  threshold; Erik's intent: steel shrugs off many small waves, one big one
  can bite; also enables brittle vs space-rated glass as two rows). Defaults
  reproduce current behavior (excluded materials ≈ ∞ threshold —
  digest-safe). Implement + tune as a chat-sized HUMAN-TEST rider AFTER the
  residency patch (priority ledger stack #1).

- **Baker writeback onto level_lib (Arc A rider, A2 accepted gap,
  2026-07-19)** — `bake_level_art.write_bake_blocks` is still its own
  non-atomic `[art]`/`[bake]` writer; entity design §3c says level_lib is
  THE data layer, all clients. Fold it in at Arc C (editor arc). Ctrl+S
  re-records mtime+hash after baking, so staleness tracking stays honest
  meanwhile.

- **Lights → entities convergence (Arc C candidate; captured 2026-07-22
  during fire-B2 design)** — lamps/beacons still live in the pre-entity
  levels-w1 `[[light]]` schema (`src/level_lights.py`), parallel to the
  Arc A–B entity system (doors/sensors/nodes are entities; lights are
  not). Fold at Arc C: lamp/beacon as entity → on/off state, SignalBus
  wiring (lamp toggled by a sensor!), editor placement — replacing the
  `[[light]]` loader. Fire-B2's `renderer/frame_lights.py` assembly
  helper is input-agnostic and survives this migration (B2 design §2).
  Original vision: tuning-plan §1a "entity/prop system + the LAMP".

- **Legacy-level entity migration — RESOLVED by retirement (Erik,
  2026-07-22, Arc C kickoff)** — `unhcr_vessel`/`unhcr_vessel_2`/
  `playground` will NOT be migrated: Erik is retiring that art
  direction ("don't like the graphics anymore, hard to design that
  way"); the replacement is a NEW level authored in the Arc C editor
  (its acceptance drive). They stay on disk in legacy form, untouched.
  `bake_demo` still waits for its art rebake (unchanged). If a legacy
  level is ever migrated after all, it's a new digest event with its
  own rationale (`docs/archive/a7_rebaseline_rationale_2026-07-19.md`).

- **Fire never destroys furniture (audit rider, weapons W2, 2026-07-05)** —
  the C++ fire's burn-through list is `is_wall`-gated, so a burning crate
  depletes its fuel (`wall_hp`) but the tile itself survives as a husk;
  meanwhile bullet chew (W2 widened `destroy_wall` to `material != MAT_AIR`)
  CAN break crates. Inconsistent on purpose for now — fix belongs to the
  fire system (engine/06): let burn-out destroy/convert furniture-class
  tiles too (burn-to-wreck material conversion is the nicer answer).

- **Scorch marks** — grenades and fire should leave permanent visual
  marks on the floor/walls where they hit. Persistent darkening, soot,
  burn patterns. **Design now in `graphics_lighting_design.md` §7
  (Destruction Painting Layer)** — single edit-texture approach, with
  normal-map dot product giving directional grenade burns. Ready to
  implement.

- **Blood splats** — reuse the destruction-painting tech from scorch
  marks for blood. Triggered by ranged/melee damage and unit death;
  brush is a dark-red feathered blob, no normal-map relief. Design
  added to `graphics_lighting_design.md` §7.5.

- **1-bounce raycaster with surface tint** — light rays bounce once
  off walls, tinted by the surface colour. Cheap caustics-lite for
  metal corridors and coloured rooms. Note in `architecture.md` §7
  flags this as "secondary priority".

---

## Gameplay / graphics — small standing items (Erik, 2026-05-23)

1. **Ambient lighting + two-kinds-of-lights discussion** — picks up the
   prior thread (see [[project-lighting-vision]] in memory and
   `graphics_lighting_design.md`). Goal: an ambient floor of light that
   reveals room geometry plus the existing directional/raycast lights
   for flashlights, fires, emergency. Two roles, distinct rendering paths.

2. **Line-of-sight for AI + players** — don't draw zombies (or any
   enemy unit) that the player has no LOS to (e.g. behind closed doors,
   around corners). Representation of "areas we don't see" is undecided
   — fog-of-war shroud, dimmed render, simple "don't draw"; start with
   the simplest "don't draw units there" and iterate. LOS check exists
   in `gamemap.has_los` (Bresenham); the question is how to integrate
   it into both the renderer (visibility filter) and the AI (already
   uses it for trigger detection).

3. **Wall collision** — units can currently walk through walls during
   execution (no per-tick collision check; only `is_passable_block`
   at order placement). Need real collision in the movement loop.
   Also: **grenades should bounce** off walls (currently they just
   stop / detonate). Grenade *explosions* still destroy walls as today.

---

## Resolution audit & consolidation

Tile size / pixel resolution decisions are sprinkled across multiple docs and
sometimes contradict each other (e.g. `graphics_lighting_design.md` still says
"32px tile — exact size TBD"). The actual decisions are presumably resolved in
code now.

Task:
1. Find every doc and code location that touches on resolution / tile size /
   sprite size / normal-map dimensions / physics-vs-render resolution.
2. Consolidate the canonical decisions into a single doc
   (e.g. `docs/resolution.md`).
3. Audit: for each claim, is it (a) still the design intent and
   (b) actually what the implementation does?
4. Update or remove stale resolution mentions in the other docs; have them
   point to the canonical doc instead.

---

## Physics — Open Items

4. **Breach decompression / lingering-smoke venting fix** — sponge + vacuum
   relaxation work but aren't physical, and leave a stubborn haze in a
   vented room. Face-flux *as a pressure sink* was attempted and reverted:
   with `d_atm = 200` it cannot clear the room (interior gradient flattens
   → wind → 0). The real fix needs a *sustained continuity wind toward the
   breach* — an open design decision. (Architecture: engine/04 §4; smoke
   ch.05.) See `atmosphere_solver_analysis_and_patch_plan_20260319.md`.
   *(2026-07-30: pre-EOS item — re-verify against the EOS/BC/sky-exchange
   engine before designing; the substrate changed under it.)*
4. **Shallow water / fluid simulation** — prototype exists (`prototypes/fluid_test.py`: pipe model + shallow water equations, ship tilting). Needs integration into game engine. Use cases: water flooding, coolant leaks, blood pooling.
## Code Cleanup

10. **Remove the vestigial C++ `is_wall` parameter** — Python retired `is_wall`
    for `GameMap.solid` (3c99b1c), but the C++ solvers still take `is_wall`
    arguments (fed `gmap.solid`; ~125 references across `cpp/src/` as of
    2026-07-30); remove the parameter + rebuild.

## Gameplay

12. **Mission 1 implementation** — "Silent Cargo" is fully designed in
    `missions/missions.md`. Needs a level authored for it first (the Arc-C
    editor is the tool; the old art-asset blocker retired with that art
    direction, 2026-07-22).
13. **Creature AI** — genetic soldiers and hybrids not yet designed. Zombies
    work. (Roster expansion — critters/cages/aquariums — is roadmap Track 4.)

## Future (not blocking anything)

15. **Faction campaign system** — see `missions/campaign_meta_design.md`. Depends on tactical layer being solid first.
16. **Narrative systems** — news cycle, phone notifications, Chase Hughes dialogue. See `narrative_media_systems_update_2026-03-08.md`.

---

## Swept from consolidated docs (2026-06-06)

_Still-open items pulled out of landed review/patch docs + architecture §16
before those docs are archived. Source doc noted in parens._

### Rendering — renderer correctness

- **Drop `tobytes()` in `update_rgba_texture`** — pass the numpy array
  directly to `ffi.from_buffer` (zero-copy) and drop the explicit `cast`;
  removes a per-frame 36 KB×3 copy and a fragile FFI-lifetime pattern.
  (code_review_renderer_v1.md C1)
- **sRGB/gamma handling in `lighting.fs`** — diffuse PNGs are sRGB but the
  shader does lighting math in gamma space; decode on sample and re-encode
  before write (or load diffuse as sRGB texture). (code_review_renderer_v1.md
  C2; patch_level_pipeline_v1.md expert review)
- **Audit normal-map orientation/encoding** — confirm Laigter exports OpenGL
  (Y-up) convention and linear; add a sign-flip toggle so inverted lighting
  is a config change, not an afternoon of debugging. (code_review_renderer_v1.md
  C3; patch_level_pipeline_v1.md expert review)
- **Light-field texture is RGBA8, not float** — quantizes light direction to
  256 angles → visible banding; switch to `R32G32B32A32` float format.
  (code_review_renderer_v1.md M4; patch_level_pipeline_v1.md expert review C++)
- **`load_shader_with_fallback` can't detect compile failures** — use
  `IsShaderReady`/raylib 4.x compile check; warn or fail loud instead of a
  silent black screen. (code_review_renderer_v1.md M5)
- **Validate shader uniform locations** — log a warning when any
  `get_shader_location` returns -1 (silent no-op today). (code_review_renderer_v1.md M3)
- **Move `set_shader_value_texture` calls to *after* `begin_shader_mode`** —
  current bind-before order risks stale sampler bindings once a second shader
  (smoke god-rays, post-FX) is added. (code_review_renderer_v1.md H5;
  architecture_review_camera_rt_patch.md H2; code_review_camera_rt_patch.md C3)
- **Centralize renderer resource cleanup** — give each subsystem
  (`LightingPass`, overlays, dynamic textures, `WorldComposite`) its own
  `unload()`; have `GameRenderer.shutdown()` iterate instead of reaching into
  private GPU handles. Also wrap `__init__` so partial construction cleans up.
  (code_review_renderer_v1.md H3; architecture_review_camera_rt_patch.md M7)
- **`TextureSet`/`WorldComposite` leak on reload** — `unload_all` doesn't reset
  slot attributes; `WorldComposite.unload` leaves `self.rt` dangling. Reset to
  None so a second `load_level_textures`/level switch doesn't leak or crash.
  (code_review_renderer_v1.md H3; code_review_camera_rt_patch.md N1)
- **`PhysicsRunner.step` returns `destroyed` walls but `main` ignores them** —
  consume so the renderer can spawn debris/smoke or invalidate light bakes.
  (code_review_renderer_v1.md M7)

### Rendering — camera / world RT

- **`coords.py` is dead code** — zero imports outside itself; either route
  `Camera2D`/`overlays.draw_unit/draw_waypoint_line/draw_grid` through it
  (rename the `ft` param to `world_px_per_tile`) or delete it. Consider a lint
  guard against naked `* ft` arithmetic. (architecture_review_camera_rt_patch.md
  H1; code_review_camera_rt_patch.md M5)
- **`compose_world` signature will balloon** — split into
  begin/draw_terrain/draw_entities/draw_overlays/draw_grid/end (or a `Scene`
  object) before projectiles, particles, decals, debug HUDs land.
  (architecture_review_camera_rt_patch.md H3)
- **Camera/RT smoke test doesn't test** — `tests/test_renderer_smoke.py` is an
  interactive demo: no assertions, no camera pan, no headless CI run. Add a
  pan-camera + place-light + read-back-pixel assertion to catch Y-flip /
  clamp / inverse-transform regressions. Drop unused `os`/`math` imports.
  (architecture_review_camera_rt_patch.md H4, C2 `__all__`/docstring;
  code_review_camera_rt_patch.md N4)
- **`mouse_to_tile` assumes viewport anchored at screen (0,0)** — breaks for
  the planned security-cam inset / offset blit; store viewport screen origin on
  `Camera2D` or route through the dst-rect. Use `math.floor` not `int()`.
  (code_review_camera_rt_patch.md C1; architecture_review_camera_rt_patch.md M2)
- **Camera viewport not updated on resize / zoom** — `FLAG_WINDOW_RESIZABLE`
  is set but `viewport_px_w/h` are baked at construction; wire `IsWindowResized`
  → update camera + RenderConfig + scissor, or drop the resize flag.
  (code_review_camera_rt_patch.md C2; code_review_renderer_v1.md H4;
  architecture_review_camera_rt_patch.md future-proofing)
- **Zoom-out past world bounds smears edges** — `clamp_to_world` lets the
  visible rect exceed world size at low zoom; clamp rect size to world (letterbox
  or stretch) and raise the `set_zoom` floor so `viewport/zoom <= world_size`.
  (code_review_camera_rt_patch.md H2, H3)
- **`clamp_to_world` not called on construction / manual pos set** — clamp in
  `__post_init__` so a level smaller than the initial camera pos doesn't start
  off-world. (code_review_camera_rt_patch.md M4)
- **Pixel-art filter decision is accidental** — bilinear is set on RT and on
  smoke/fire textures (double-blur); decide POINT-vs-BILINEAR per texture
  deliberately and document it. Add a `pixel_perfect` snap option.
  (code_review_camera_rt_patch.md H4; architecture_review_camera_rt_patch.md
  future-proofing)
- **`world_px_per_tile = 24` is unvalidated magic** — compute from diffuse
  dimensions or assert it matches the asset; add a guard that RT size won't blow
  GPU memory on large ships. (architecture_review_camera_rt_patch.md M6;
  code_review_camera_rt_patch.md M6)
- **Add camera helpers before they're written inline** — `zoom_at(focal)`
  (zoom-around-cursor), `add_shake`, `follow(target, lerp)`.
  (architecture_review_camera_rt_patch.md M3, M4)
- **Generalize `blit_world_to_screen` → `blit_view(camera, dst_rect)`** for the
  planned multi-camera / security-cam inset; drop the redundant scissor on the
  full-viewport blit. (architecture_review_camera_rt_patch.md future-proofing;
  code_review_camera_rt_patch.md H1)
- **Multiple opposing lights cancel direction to (0,0)** → "dead spots" where
  cones meet; switch to per-tile dominant-direction (running max) before
  shipping multiple emergency lights. (code_review_renderer_v1.md future-proofing)
- **Decide smoke-vs-unit draw order / occlusion** — units currently draw in
  front of smoke even when inside a cloud; open question whether smoke should
  occlude units. (architecture_review_camera_rt_patch.md M5)
- **Future-proofing render hooks** — `FieldOverlay` is the wrong abstraction for
  particles (need a `ParticleSystem` owning a RenderTexture/batched quads);
  `draw_unit` draws a circle and needs a sprite-atlas batcher for 30+ units;
  add HDR/multi-RT format param to `WorldComposite` before it has many callers.
  (code_review_renderer_v1.md future-proofing; architecture_review_camera_rt_patch.md
  future-proofing)
- **Fix `renderer/__init__.py` docstring + `__all__`** — documents a removed API
  (`draw_world`/`draw_units`/`draw_overlays`); export `RenderConfig`.
  (architecture_review_camera_rt_patch.md C2; code_review_renderer_v1.md N1)
- **Remove stale `math` imports** in `lighting.py`, `overlays.py`,
  `game_renderer.py`. (code_review_renderer_v1.md N3)
- **Cross-check level asset dimensions** — `level_loader.py` validates files
  exist but not that diffuse/normal/etc. share dimensions; a mismatched normal
  map samples wrong silently. (code_review_renderer_v1.md N6)

### Physics / simulation

- **Smoke-at-vacuum-tiles bug** — zero `gmap.smoke` at vacuum tiles before
  upload (~30s fix). (patch_game_logic_migration.md after-migration follow-ups)
- **Liquids system not implemented** — per-tile liquid type (none/blood/water/
  fuel) + depth as state fields, interacting with fire/creatures/electricity;
  needs its own design pass. (architecture.md §16 #11)
- **Double-buffered propagation not implemented** — introduce during C++ port;
  consider replacing the per-frame physics state copy with a buffer swap.
  (architecture.md §16 #12; patch_game_logic_migration.md follow-ups)
- **Fire wall-destruction double-loop** — replace `for fy/for fx if burned_out`
  with `np.argwhere(burned_out)`. (architecture.md §16 #5)
- **Smoke/fire have no substeps** — run once per tick at 83ms dt; evaluate
  stability under large parameters. (architecture.md §16 #6)
- **Config inconsistency** — wave/fire/explosion parameters hardcoded instead
  of in config.toml; unify all tunables. (architecture.md §16 #1)
- **Plumb the seeded RNG through all nondeterminism** — `_add_explosion_smoke`
  (smoke noise), `Raycaster.cast_source` (fire flicker jitter), `_fire_burst`
  (bullet cone) must pull from `Simulation.rng` or AI rollouts/replays diverge.
  (review_game_logic_migration.md C3; patch_game_logic_migration.md Step 8)

### Units / gameplay

- **Phase-transition detection is fragile** — `new_phase = exec_tick // tpp`
  works but consider explicit phase-boundary tracking. (architecture.md §16 #7)
- **Explosion visuals are weak** — no flame burst/flash; port or redesign the
  legacy pressure-to-color drama. (patch_game_logic_migration.md playtest #1)
- **Explosions should emit a transient light source** at the blast center for a
  few ticks (renderer-side transient `LightSource` or sim-side temp entity).
  (patch_game_logic_migration.md playtest #2; cross-ref memory
  project_explosion_as_light_idea.md)
- **Strong/weak zombie variants** — runner (low HP, fast) vs brute (high HP,
  slow); `unit.py` carries `speed_ticks_per_tile`/vitality, wire up variant
  spawning. See `unit_variants_design_brainstorm.md` and the `level.toml` TODO
  for the ogryn-zombie spawn. (patch_game_logic_migration.md playtest #5)
- **Per-creature AI sample rate** — humans slow, robots fast.
  (patch_game_logic_migration.md follow-ups)
- **Promote inventory booleans into `Inventory`** — `has_grenade`/`has_explosive`
  still live on Unit alongside the stub `Inventory`; migrate them in.
  (patch_unit_class_foundation.md decision 8)
- **Non-symmetric footprint rotation** — `occupied_tiles()` applies no rotation;
  add rigid-body rotation for non-symmetric footprints (spec §15 item 3).
  (patch_unit_class_foundation.md §5)
- **Unit-system deferrals (data exists, behaviour missing)** — modifier system
  (`compute_effective_stats` returns base), environment damage
  (`EnvironmentProfile` is data-only), faction relationship table (combat still
  uses `team != team`), fear / Gray hook / awakening trigger. Per spec §13.
  (patch_unit_class_foundation.md decision 9)
- **`apply_action` result/legality convention** — commit to a `Result` enum
  (OK/NO_AP/NO_INVENTORY/BLOCKED) and/or `get_legal_actions` so the UI can show
  failure toasts instead of silent returns. (review_game_logic_migration.md
  #3, #12)
- **Order subclassing** — split single `Order` discriminator into
  `MoveOrder`/`FireOrder`/etc. (its own patch, deliberately deferred).
  (patch_game_logic_migration.md anti-goals; review_game_logic_migration.md #9)

### Pathfinding

- **Temporal A* unused** — `ReservationTable` + `temporal_astar` exist but are
  never called; player units can overlap during execution. Enable or remove.
  (architecture.md §16 #4)
- **Pathfinding constants hardcoded** — `FINE_W=120`, `FINE_H=75`,
  `UNIT_SIZE=3` in pathfinding.py don't reference config. (architecture.md §16 #8)

### Cleanup / resolution

- **Remove the coarse-tile concept** — ~66 references in legacy code, some dead,
  some needed for unit footprints; dedicated cleanup.
  (patch_level_pipeline_v1.md "What this patch does NOT cover")
- **Aspect-ratio / diffuse alignment** — 972×1619 diffuse vs 50×120 tilemap is
  stretched for v1; align the art to tilemap bounds (manual or automate).
  (patch_level_pipeline_v1.md open question #2)
- **`_draw_ui_panel()` is 207 lines** — consider a lightweight UI layout system.
  (architecture.md §16 #9)

### AI training infrastructure

*(2026-07-30: these are now Track 2 of `roadmap_2026-07-30_rl_push.md` — the
substrate milestones M0–M4 subsume them.)*

- **AI training scaffolding (`train.py`)** — Gymnasium loop over `Simulation`;
  add `get_reward`/`is_terminal` hooks (facade stubs exist). Separate patch.
  (patch_game_logic_migration.md follow-ups; review_game_logic_migration.md #3)
- **Save/load + `serialize()/deserialize()`** — nearly free given `get_state()`
  returns flat arrays; needed for replay buffers. (review_game_logic_migration.md
  #13; patch_game_logic_migration.md testing strategy)

### Arc B follow-ups (entity logic layer)

- **Manual airlock / "airlock v2"** (Erik, 2026-07-21 HUMAN-TEST) — manual OPEN
  buttons on each side of both airlock doors, alongside the automatic
  `airlock_controller`. Deferred: `button`/terminal are format-reserved but
  INERT in v1 (entity design §3d) — they were gated on the control-scheme
  decision, which RESOLVED 2026-07-23 (modularity split; direct gamepad control
  is merged and blessed). **Now unblocked** — needs unit-initiated `use`
  interaction under direct control. Slot when wanted.
- **Resident sensor-gather kernel** — Arc B stubbed the §5a accessor to the host
  mirror (no GPU gather kernel, per the S8c-concurrency constraint). Build the
  `(n_sites × n_channels)` int32 gather kernel on the resident path once S8c has
  landed; the accessor interface is already frozen (cuda_s8a spec §5a).
