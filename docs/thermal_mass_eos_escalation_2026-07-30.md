# ESCALATION → Fable design pass: the thermal medium in the EOS pass (2026-07-30)

**Status: OPEN. Blocks P3 of the thermal-mass-axis arc. Erik's ruling 2026-07-30:
"Fable design pass first" — no code on this question until the design lands.**

Raised by: Opus build session, 2026-07-30, during P1/P2 of
`docs/thermal_mass_axis_design_2026-07-25.md`.
Escalation trigger fired: **§4 trigger 3** — "Any consumer found deriving thermal
behavior from `solid` OUTSIDE the temperature solver (grep first; the agent found
none, but verify)." The design anticipated this exact possibility. It exists.

Reading order for whoever picks this up: this doc → the design doc → the build
addendum (`thermal_mass_axis_build_addendum_2026-07-30.md`) → `git show f5e9aa3`
(P1, CPU) → `git show 312e984` (P2, CUDA).

---

## 1. What P1 + P2 already delivered (and it is sound)

The blessed re-route is **built, gated, and committed** on branch
`thermal-mass-axis`:

- `thermal_solid = (thermal_mass > 0)`, derived on the one structural-rebuild seam,
  replaces `solid` at the six **medium**-test sites in `temperature_solver.cpp`
  (P1, `f5e9aa3`) and their six CUDA twins in `cuda_temperature.cu` + the resident
  path (P2, `312e984`). Each site is marked `MEDIUM-TEST SITE n/6` in both files.
- Gate (a) furniture-free byte-identity: **PASS**, zero tolerance, CPU and GPU.
- Gate (b): **no golden moved, none rebased.** Failure set identical to baseline.
- Gate (d) CPU↔CUDA lockstep **tol 0**, step AND resident, on a furniture-burn
  scenario, with non-vacuousness controls (176/240 configs where
  `thermal_solid != solid` agree; the same configs diverge with the mask omitted).
- Gate (e): conservation/sealed-room + sky-exchange green.
- `permeability`, `solid`, `dyn_permeability`, mobility, LoS, unit stamping:
  **untouched**, per §2.4 / trigger 5.

**Within `TemperatureSolver`, the crate now behaves exactly as §2.2 designed**:
measured `+11.3` game/tick = `deposit>>3 − T>>5`, and the §2.5 analytic is
confirmed *to the LSB* (predicted 290.6, measured 290.7 on the isolated pass).
The design's arithmetic is right. Nothing below invalidates any of it.

## 2. The finding: the live T-advection is not in the temperature solver

Two facts, both verified by reading the code (not inferred):

**(a) Three of the six blessed sites are dead code in the live engine.**
`step_tail` passes `wind_x = wind_y = nullptr` to the temperature solver. This is
deliberate and documented at `cpp/src/physics_engine.cpp:211-229`: *"Pass 0 is
REDUNDANT — eos_solver already advected T"*, and *"step_tail passes wind_x =
wind_y = nullptr above"*. So the `:187` advection skip, `:42` `gas_wall_at`, and
`:113` sealed corner are live **only** for the direct Python binding and tests.
The design's §2.2 bullet "Pass-0 advection + compression-work: SKIP thermal_solid
tiles" therefore does not reach the engine at all.

**(b) The live semi-Lagrangian T-advection is `EOSSolver::step` step-1b, and its
compression-work term is step-4c — both keyed on a mask that treats furniture as
gas.**
- `eos_solver.cpp:406-422` — backtrace sample, then `temperature[i] = ... fs.t`,
  once **per EOS substep**.
- `eos_solver.cpp:670-688` — compression work writes `temperature[i]` again.
- `eos_solver.cpp:366` — the mask: `if (solid[i] || dyn_permeability[i] <= 0.0f)
  cmask_[i] = 0;` … `else cmask_[i] = 2`. Furniture is `solid = false`,
  `dyn_permeability = 0.5` → **cmask 2 = interior gas cell**.

So the crate's object temperature is overwritten by an upwind gas sample every
substep. The axis the design added is respected in the temperature solver and
ignored in the pass that actually moves T.

### Measured consequence

Stage-probe (warm seed T=280 on the crate, `run_substeps` vs `step_tail` deltas,
game units/tick; P1 build agent, reproducible via the bench in §5):

| tick | `run_substeps` (EOS) | `step_tail` (thermal) |
|---|---|---|
| 1 | **−21.2** | +11.3 |
| 2 | **−34.7** | +10.4 |
| 3 | **−32.7** | +9.6 |

COOL_SHIFT's own loss at T=280 is `T>>5` = **8.75**/tick. **The EOS pass removes
2–4× that.** Therefore §2.2's promise — "its ONLY loss is COOL_SHIFT — one clean
channel for Erik's tuning" — **does not hold in the live engine**, and §2.5's
equilibrium `T* ≈ 4·k_fire_heat·I` is unreachable: observed/predicted ratio 4.6–12.4
in the old gas regime, and in the new regime T collapses to ~3.6 and the fire dies
by t≈1 s (T never reaches `fire_T_ext`).

**The corroborating diagnostic:** with the crate ALSO hidden from the EOS
advection (scratchpad-only, never committed) plus §9.3-step-1 dial structure
(`fire_T_ext` 250, span 100), the design's intended gate-(c) shape appears
immediately — monotone rise from the 280 seed over the first 2 s = **True**, peak
I 0.316 @ 3.7 s, T at peak 1514. That is the evidence the missing piece is the
**routing**, not the dials.

## 3. Why this is a design question and not a build detail

The EOS `cmask` is **shared**: it drives pressure, velocity, and gas-flow marching,
not just temperature. So there is no mask swap available here, of the kind P1/P2
were. Any fix must decide *where the thermal medium diverges from the gas medium*
inside a solver whose whole design is that they are one coupled system:

- Flipping `cmask` for furniture would **seal the crate to gas flow** — it would
  destroy `permeability = 0.5` / shield-but-not-seal and trip trigger 5. Not an option.
- Excluding thermal_solid tiles only from the T-writes (step-1b and step-4c) leaves
  pressure/velocity untouched, and is the literal reading of design §2.3 ("the
  gas-medium pass no longer advects T ACROSS the crate tile — heat crosses via the
  surrounding air cells"). It is the *shape* Erik and I sketched as the likely answer,
  but it needs the design pass to be sure it is coherent with the EOS energy
  accounting rather than merely making the bench look right.
- §2.3's own **pressure tripwire** is now genuinely relevant: the crate holds gas
  (N > 0) and EOS reads `P = C·N·T[i]`. If T becomes object temperature, the pore
  gas of a 1300 K object drives overpressure. §2.3 accepted that as desirable fire
  behaviour *and* named the fallback (`P = C·N·t_amb` on thermal_solid tiles). That
  choice was made assuming the temperature solver owned T; it should be re-affirmed
  now that the EOS pass is in scope.

## 4. The question for the design pass

Primary: **does the thermal medium get its own mask inside the EOS pass, or does T
stop living on the gas plane for thermal_solid tiles altogether?** Concretely:

1. Which EOS writes to `temperature[]` must skip thermal_solid tiles — step-1b
   advection only, or step-4c compression work too? (Both write T; they are
   physically different claims.)
2. Should a thermal_solid tile still act as a **source** for its neighbours'
   backtrace samples, or be treated as an occluder there (the `gas_wall_at`
   question, one level up)? §2.3's "heat crosses via the surrounding air cells"
   implies occluder, but that is an inference, not a ruling.
3. Does the §2.3 pressure decision stand — hot pore gas — now that the crate's T is
   genuinely an object temperature in the pass that computes P?
4. Energy accounting: the sealed-room conservation gate is load-bearing in this
   engine. If T stops advecting off crate tiles, where does that energy go, and does
   the conservation gate still mean what it meant?
5. Conduction is currently the crate's *only* possible physical exit
   (furniture `conductivity = 0.0` → NO_FACE both sides, so today it exchanges
   nothing). §2.2 called raising furniture κ "a realism dial, not this patch" — is
   that still right once COOL_SHIFT is genuinely the only channel?

## 5. Reproduction

- Branch `thermal-mass-axis` @ `312e984` (off `fire-o2-integration` @ `423cd38`),
  worktree `breach.worktrees/thermal-mass-axis`. Both builds current
  (`cpp/build` CPU, `cpp/build_cuda` Lenovo/Ada sm_89 — **rebuild required** after
  P1's `step_tail` signature change).
- Bench: `tools/fire_timing_harness.build_level` (84×40 planetside, deep crate,
  tile 0.333 m, sky_tau 60, sponge 8) + the warm seed. The in-build A/B lever is the
  nullable `thermal_solid` arg: passing `thermal_solid := solid` reproduces pre-patch
  behaviour exactly, in one binary.
- Suite baseline on this line: **42 failed / 1705 passed / 5 skipped**. The 42 are
  pre-existing by-design reds inherited from the o2-continuous-law line (enumerated
  in that arc's handoff doc); the failure *set* is unchanged by P1 and P2.
- All dial variations quoted above were runtime CFG overrides in scratchpad scripts.
  **`config.toml` carries exactly one edit from this arc: air `thermal_mass` 1 → 0.**

## 6. Standing constraints for whatever the design chooses

- Determinism is non-negotiable: Q16.16 integer only in the sim path, no floats, no
  libm transcendentals (`cpp/src/fixed_point.h`). Gate everything CPU↔CUDA at tol 0,
  step **and** resident.
- Gate (a) furniture-free byte-identity, zero tolerance, must survive any EOS change.
- **No golden rebase in this arc.** It rides the joint re-tune's ONE deliberate
  rebase, with written rationale, with Erik.
- **HUMAN-TEST gate**: fire behaviour is feel-adjacent, so nothing here auto-merges.
  P3 hands back to Erik's manual tuning loop (§9.3), which expects
  `k_fire_heat ≈ 225` vs today's 12 and treats COOL_SHIFT as a live dial (Erik has
  flagged he may prefer 6–7 over 5).
- Out of scope, full stop: `permeability`, `solid`, `dyn_permeability`, mobility,
  LoS, unit stamping. Units-as-ignitable remains queued to the unit environment
  system (design §2.4), not here.

## 7. One process note worth keeping

The build addendum's **D5 sweep concluded "no out-of-solver thermal consumer of
`solid`" and was wrong.** The sweep was a line-oriented grep for `solid` near
thermal keywords, which structurally cannot see `eos_solver.cpp`: the mask
definition (`:366`) and the `temperature[]` writes (`:422`, `:688`) are hundreds of
lines apart, so no single line contains both. The design doc's own instruction —
"grep first; the agent found none, but **verify**" — was the right instinct, and a
grep is not a verification. For a *routing* question, the check that works is to
start from the written field (`temperature[]`) and enumerate every writer, not to
start from the mask and look for thermal words nearby.

---

**Appended 2026-08-14 (supersession note).** Any `T + 290`-only EOS ambient
description above is superseded by the unified canonical map in
`[physics.temperature_scale]`: the sim-wide Kelvin map is now `K = 293 +
3·T_game`, with the EOS pressure calibration keeping a named, deliberate
exception at `eos_t_amb_k = 290` (unchanged value, now a documented exception
rather than the only convention). See
`docs/temperature_scale_unification_design_2026-08-13.md`.
