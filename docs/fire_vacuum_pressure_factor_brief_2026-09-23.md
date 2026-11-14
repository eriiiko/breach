# #7 brief — vacuum kills fire: a pressure factor on the O2 law (2026-09-23)

> **Status**: brief, written by the orchestrator from Erik's ruling of
> 2026-09-23 (issue #7). Not yet built. **HUMAN-TEST** patch: built, gated,
> pushed, NOT merged until Erik breaches a burning room.
>
> **Depends on**: issue #7 (root cause, 2026-08-21) and its ruling comment;
> `docs/continuous_o2_law_design_2026-07-24.md` §2.1 (the density trap, and why
> the law reads the mole FRACTION); `docs/fire_mechanics_inventory_2026-08-31.md`
> §1.2 (G5 and Erik's absolute-density amendment of 2026-08-31);
> `docs/ray_engine_v2_scheme_study_2026-09-13/report_m3.md` finding 4.

## 1. The bug
Fire sustain reads the O2 **mole fraction** `X = Σn_O2 / Σn_total`. Venting
removes O2 and N2 together, so X stays ~0.21 down to zero molecules and a fire
in a room vented to 0.013 atm holds full intensity. On `fire-12` two tests hid
it only because the old radiation (3110× the derived scale) cooled the vented
fire first; the flip removed that mask, and both now carry strict xfails
naming #7.

## 2. The ruling (Erik, 2026-09-23) — do not re-derive
- Today's fraction law stays exactly as it is (`o2_frac_ext = 0.13`).
- It is MULTIPLIED by a pressure factor
  `g = clamp01((p_rel − p_ext) / (p_full − p_ext))`, with `p_rel` = local
  pressure / ambient, **`p_ext` = 0.1 atm** (a fire cannot live below it) and
  **`p_full` = 0.5 atm** (no effect above it).
- It reads **PRESSURE, not density**. Hot air is thin, not low-pressure, so
  the July density-trap fix survives; a vented room drops in pressure.
- The two edges are **config keys** in `[physics.fire]` beside `o2_frac_ext`,
  authored in atm, converted once at load in `PhysicsRunner` (the Q16 number
  doors; never a hardcoded 65536). Each key carries its literature citation
  in the config comment. Global, not a material column.
- It applies at BOTH reads of the law: the intensity ODE's `o2f`
  (`cpp/src/fire_simulation.cpp` + `cuda_fire.cu`) and combustion's claim
  gate `o2f_j` (`cpp/src/combustion.cpp` + `cuda_combustion.cu`).

## 3. What the implementer settles (and reports)
1. **The pressure source.** Use the engine's canonical per-cell pressure; do
   not build a parallel one. Candidate: `gas_energy` (= N·T_abs, exact int64)
   is proportional to pressure in the engine's units, so `p_rel` = E / E_amb
   per cell; an existing pressure/atmosphere plane may already be it. Pick one,
   say why, and prove it is invariant under thermal expansion at constant
   pressure (that IS the density-trap property).
2. **The gather.** Read `p` over the same neighbourhood the X law gathers over,
   so both factors see the same air.
3. **Integer arithmetic.** Q16.16, the span's reciprocal hoisted host-side
   once per tick like X's; CPU and CUDA bit-identical.
4. **The citation.** Find published low-pressure flammability / extinction
   limits for cellulosic solids and cite them beside the keys. If the
   literature disagrees materially with 0.1 / 0.5 atm, REPORT it — Erik's
   values stand until he rules.

## 4. Properties to pin (write them first, run them on the tip)
1. **Ambient is untouched**: at p ≥ 0.5 atm the law is bit-identical to today's
   (g = 1 exactly in Q16).
2. **Hot air keeps its fire** (the density-trap guard): a hot room at ambient
   pressure — thermal expansion, low N, normal p — sustains exactly as today.
3. **A vented room loses its fire**: below 0.1 atm the fire goes out.
4. **CPU == CUDA** at tolerance 0 on both law sites.
5. Remove the two strict-xfail markers (`test_e2e_2_breach_vents_o2_and_kills_fire`,
   `test_payoff_orderings_perturbation_robust`); both must pass for their
   stated reason. `tests/_scenario_all_systems_bench.py` P4b should flip to
   PASS — run it and report.
Validate each new property by breaking the code once (restore afterwards).

**Goldens.** `GOLDEN_AGGREGATE` should NOT move unless the canonical scenario
has a burning cell below 0.5 atm. If it moves, first prove that is the reason
(a move at ambient pressure is a bug); then it is an approved behavioural
change (Erik ruled the law), so re-baseline ONCE with written rationale naming
this ruling.

## 5. Out of scope — one system at a time
Nothing else in the fire law: `hot`, `hotf`, `k_die`, the timer, ignition,
`H_bed`/`H_fuel`, materials. Anything you notice elsewhere is one line in the
report's findings, never a fix.

## 6. Execution
- Branch `7-vacuum-kills-fire` off `fire-12`, own worktree `breach-7`, one
  writer. Model: **Opus**.
- Builds `cpp\build_cpu_home.bat` + `cpp\build_cuda.bat`, both, first and after
  every C++ change. Python `C:/Users/steen/anaconda3/python.exe`. Suite
  `pytest tests -q -p no:cacheprovider` from the worktree root.
- Commits `fix(#7): ...` / `test(#7): ...` / `docs(#7): ...`, explicit paths.
  Do not merge, do not push.
- Report `docs/report_7_vacuum_kills_fire.md`, skeleton first, filled as you
  go. It is the orchestrator's technical record: as long as the evidence
  needs and no longer, anything outside this patch one line, no question
  lists. It must say exactly how to run the HUMAN-TEST (which level, which
  wall to breach, what a working fix looks like).
