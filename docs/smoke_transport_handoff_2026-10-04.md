# Handoff: smoke transport does not conserve smoke (2026-10-04)

**For:** the next Claude session in the fire + smoke session (#12, step 3 of the
seven-step plan on #31). **From:** the 2026-10-03/04 session, which landed the
smoke × light work merged with this doc. The decisions recorded here are Erik's.
The session log is #12.

## 1. What to do

Erik's words: *"I'd like to know what needs fixing — then let's try to fix
it."* Two steps, in order:

1. **Understand it fully** (§3 lists the experiments; run them first). Find out
   which mechanisms create and destroy smoke, and how much each contributes,
   measured rather than assumed.
2. **Fix it**: a design note → critique → patches (CPU + CUDA) → one golden
   re-baseline with written reasons. This is arc-sized. Load the
   `autonomous-patch-workflow` and `git-workflow` skills first.

Why it matters now: handles 1–3 below made the smoke AMOUNT physical, so the
light it absorbs and how black it looks follow from it. If transport invents
smoke tenfold and then destroys it, none of that can be judged. Erik thought
smoke transport conserved mass and remembers it doing so once. Check the git
history for that (was there a conservative smoke scheme before S2b?).

## 2. What was measured (smoke-light studio, sealed hull, no venting)

`levels/smoke_light_studio` (0.333 m tiles): three Composition B charges fire
once at 1.0 / 1.5 / 2.0 s. They should leave 1.27 + 2.54 + 5.08 = **8.9 smoke
units** (99 g of soot). Total smoke over the run, seed 42:

| t (s) | total units | expected |
|---|---|---|
| 1.25 | 14.3 | 1.27 |
| 1.75 | 91 (peak) | 3.8 |
| 2.0 | 35 | 8.9 |
| 2.75 | 3.4 | 8.9 |
| 10 | 0.64 | 8.9 |
| 12 | 0.13 (then a slow rise: the crates' fire smoke) | 8.9 |

The probe script was `probe_smoke_total.py` in that session's scratchpad. It is
trivial to rebuild: load the level, step, print the sum of
`gmap.gas[smoke] / 65536`.

## 3. Hypotheses, NOT yet verified: the experiments to run first

The trace gases (smoke, steam, poison, teargas, fuel_gas) ride an integer
semi-Lagrangian (SL) scheme: `cpp/src/smoke_dynamics.cpp` (S2b), its CUDA twin
`cuda_smoke`, called from `PhysicsEngine::run_substeps` (and from
`trace_smoke_resident` on the resident path). Its header already says
*"gently non-conservative ... NO flux form, NO limiter"*. The bulk pair
(o2 / inert_n2) instead rides an exact, conservative donor-cell flux
(`cpp/src/bulk_transport.*`). Candidate mechanisms, each to be confirmed by
measurement:

- **H1, advective form on a divergent flow.** SL solves ∂c/∂t + u·∇c = 0, which
  is right for a MIXING RATIO. Smoke is stored per tile, as a DENSITY, which
  obeys ∂ρ/∂t + ∇·(ρu) = 0. Where a blast makes the air expand, SL keeps the
  per-tile value and creates mass; where air compresses, it destroys it. That
  fits the 10× inflation right after each blast.
- **H2, the bilinear renormalisation near walls** (`1/wsum` when solid corners
  drop out) amplifies the samples next to walls.
- **H3, the [0, 1] clamp** destroys whatever passes 1 (the inflated peaks).
- **H4, the ">>16 truncation"** decays a little every tick (documented,
  accepted Q-S2-1).
- **H5, the wind it advects on.** CLAUDE.md warns that the raw `wind_x/wind_y`
  planes are "fire-spiked, unusable as a velocity" for the renderer. Check what
  the trace loop is handed (the EOS's final corrected wind?) and whether that
  is right.
- **H6, wind-coupled diffusion** (`d_eff = d·(1 + wind_diffusion_scale·|wind|²)`):
  conservative in itself, but clamped to the stability limit in a blast. Check
  that it conserves.
- `decay` (0.008 × sim_time per tick: an e-fold of about 125 s) and the
  vacuum/ambient sinks are real removals. Account for them, don't fix them.

Experiments planned (Erik paused them for the new session):
1. **Quiet sealed room:** a smoke blob (density 0.05, r = 4) and no charges.
   Is the total conserved over 10 s? This isolates H2/H4/H6 from blast flow.
2. **Same blob plus one heat-only frag (`frag_heat`)** executed a few tiles
   away: how much does the total jump, and where (expansion cells?). This tests
   H1/H3.
3. Per-tick attribution: switch each mechanism off in turn (a test-only
   build flag, or the Python reference if one exists).

## 4. The fix direction (to design, not decided)

The canonical system is the bulk pair's conservative flux
(`bulk_transport`). Reuse it rather than building a parallel scheme
(CLAUDE.md, the canonical-systems rule). A natural shape: trace gases ride the
BULK mass fluxes, so each face's trace flux is that face's bulk flux × the
upwind cell's trace / bulk ratio. That is computed once per face in canonical
orientation and applied ± (the arc #54 face-flux idiom), so it is exact in
int64. Smoke then moves exactly with the air and can neither be created nor
lost.

Known snags:
- `GasTable.conservative` today means BOTH "flux transport" AND "counts toward
  bulk N / pressure" (`step_tail` sums the conservative planes). Setting it on
  smoke would make soot push on the air. Split the flag (transport vs bulk
  membership) and do NOT flip it.
- The energy books: trace shares carry mass, not energy
  (`physics_engine.cpp`: "Only BULK shares carry energy"). Keep it that way.
- CUDA lockstep (`cuda_trace_smoke_check.py`) and the resident path.
- The renderer's tamed wind and the smoke detail pass read the smoke field; a
  sharper, conserved cloud will look different. That is a HUMAN-TEST gate.
- Golden: every scenario with smoke in it moves; re-baseline once.

## 5. State at handoff

Merged into fire-12 with this doc (branch `12-smoke-light-studio`):
- **Smoke-light studio** (`tools/gen_smoke_light_studio.py`): the explosion
  studio's rooms, a grey floor plus a real groove normal map, nine lights
  including spots and the probe beam at (50,13) due west.
  Launch: `main.py --level smoke_light_studio --control gamepad --warp 3.0`.
- **`main.py --warp SECONDS`** (a frozen frame by determinism),
  **`e2e_drive --shot PATH@FRAME`** (lets Claude see a frame), and
  **`kind = "spot"`** lights.
- **Fix:** the lit floor sampled an unbound normal map (one-sided lamp light).
- **Handle 1:** `[gases.smoke] light_absorb = [120.66, 134.29, 158.89]`,
  derived from soot's visible extinction (Mulholland & Croarkin), test
  `test_gas_light_absorb.py`.
- **Handle 2:** the light smoke term is Beer-Lambert (`exp_neg_q16`, reference
  first, an FP_HD twin on CPU and CUDA). The renderer's opacity IS the light
  column; `plume_k_scale` is retired. Heat keeps min(1, τ).
- **Handle 3:** blast smoke = the charge's soot (`[explosives.*]`, payload
  `explosive` / `explosive_kg`, `payloads.blast_smoke_peak`). C4 charges make
  no blast smoke.
- **Both studios at 0.333 m tiles.** `[display] sim_speed` (slow-motion
  pacing, render only).
- **Fix (#4):** two CUDA EOS gates own `k_drag2 = 0` (they had gone red with
  0.125; the k_drag2 merge ran with them skipped).

Erik's rulings this session:
- Tune everything at 0.333 m tiles.
- Rendered blackness = one tile's crossing (not the ceiling-height column).
- Keep `k_drag = 0.5`, `k_drag2 = 0.125` (per second / per metre: the same
  physics at any tile size), and `sim_speed = 1`. Drag is revisited after
  smoke conserves mass.
- Blast smoke goes the physical route.

## 6. Parked for after transport (#12 inbox)

- The smoke look: `scatter_albedo` (0.04 today; real soot is ~0.2).
- Lit-side shading of smoke puffs (the light flux direction × the density
  gradient). Erik's idea of explosions bulging in 3D.
- O2/pressure in the temperature → glow mapping (the near-vacuum fireball).
- Hot clean air as a light source.
- Dynamic smoke emission (low O2 + high T); soot burning up in a fireball (the
  TNT yield is a no-afterburn upper bound).
- A proper drag-tuning level.
- The explosion studio's dark floor.
- Explosion studio ignition / the frag grenade re-judged at 0.333 m.

Findings recorded elsewhere:
- The airlock probe's period-2 pressure sawtooth (#48).
- The CUDA EOS check scenarios write `temperature` directly (the CLAUDE.md
  "mirror" rule says that goes vacuous). Seen while fixing them; not touched.
