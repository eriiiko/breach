# Archive index

What each archived doc *was for* and where its content lives now. This index
is not a substitute for the docs it lists — read the doc itself for detail;
read this to decide whether you need to. Entries are append-only capture,
moved unchanged (`git mv`) at arc close per `CLAUDE.md`'s arc-close rule.
New entries go at the top.

---

## 2026-08-14 — fire arc + temperature-scale unification arc close

Both arcs are complete and merged to main (temperature-scale: commit chain
`9016cd7..ee97f61`, Erik-blessed P-K5, `docs/TODO.md` "Waiting on Erik"). The
as-built record for the temperature-scale arc, including what superseded
what, is `docs/temperature_scale_unification_design_2026-08-13.md` §10. Fire
tuning's living plan (`docs/plan-for-tuning-and-graphics.md`) and the still-
open storm-damping thread are unaffected — see `docs/TODO.md`'s head block.

**Fire realism / sizing / recalibration / tuning family** (superseded by the
canonical `[physics.temperature_scale]` map and the promoted TUNE dial set;
each carries its own append-only supersession note pointing at the design
doc):
- `fire_tuning_plan_2026-07-22.md` — original fire tuning plan.
- `fire_model_design_seed_2026-07-30.md`, `fire_tuning_session_seed_2026-07-30.md`,
  `fire_constants_audit_2026-07-30.md` — pre-realism-pass seeds/audit.
- `fire_realism_design_2026-08-01.md`, `fire_realism_design_plain_2026-08-02.md`,
  `fire_realism_critiques_round1_2026-08-01.md`,
  `fire_realism_critiques_round2_2026-08-01.md`,
  `fire_realism_critiques_round3_2026-08-02.md` — the realism-pass design +
  its adversarial critique rounds.
- `fire_recalibration_2026-08-02.md`, `fire_sizing_package_2026-08-02.md`,
  `fire_sizing_plain_2026-08-02.md` — recalibration/sizing that fed the
  promoted dial set (P-K0).
- `fire_b1_blackbody_fire_lights_design_2026-07-21.md`,
  `fire_b1_build_log_2026-07-21.md` — Fire B1 (blackbody overlay) design +
  build log; shipped and blessed.

**Thermal-mass-axis family** (design → build → EOS escalation → ruling; the
axis itself is not invalidated, but its Kelvin-map question is answered by
the canonical map):
- `thermal_mass_axis_design_2026-07-25.md`,
  `thermal_mass_axis_build_addendum_2026-07-30.md`,
  `thermal_mass_axis_bench_report_2026-07-30.md`
- `thermal_mass_eos_escalation_2026-07-30.md` (the five open questions),
  `thermal_mass_eos_ruling_2026-07-30.md` (Fable's ruling answering them —
  the thermal_solid ownership rule survives, folded into architecture/engine
  chapter 06).

**Radiation / raycaster family** (the ×2 Kelvin map they used is superseded
by the canonical map; extinction-model rulings themselves stand):
- `radiation_and_raycaster_design_seed_2026-07-31.md`
- `radiation_raycaster_extinction_ruling_2026-07-31.md`

**EOS research, old ambient-map description:**
- `eos_research_report.md` — carries its own 2026-08-14 supersession note;
  the `T + 290`-only EOS ambient description is superseded by
  `[physics.temperature_scale]` (`eos_t_amb_k = 290` is now a named,
  deliberate exception, not the whole story).

**OnePhaseWEGO arc close-out** (design doc stays live — see note below):
- `onephase_wego_kickoff_2026-07-28.md` — kickoff/architecture handoff.
- `onephase_wego_asbuilt_2026-07-28.md` — as-built; folded into
  `architecture/mechanics/04_turn_and_control.md` and siblings.
  `onephase_wego_design_2026-07-28.md` (the LOCKED design doc) stays at
  `docs/` top level — `inertia_and_sprint_design_2026-07-30.md` (open,
  un-started) still cites its §4/§6 as background.

**2026-08-03/04 audit family** (Patch A — 9 bounded items — landed; commits
A1–A9 are ancestors of `main`):
- `codebase_audit_2026-08-03.md` — the six-area code quality audit.
- `audit_lessons_and_rules_2026-08-04.md` — the 8-pattern taxonomy + rules
  worth adopting (scored against what each would/wouldn't have caught).
- `audit_handover_patch_a_2026-08-04.md` — the decision-free work package;
  all 9 items landed (see commits A1–A9, e.g. `1e14669`..`af50a3a`).
- `breach_todo_2026-08-03.md` — Erik's original brain-dump (Swedish) naming
  the `T0 + k·T_game` idea that became the canonical Kelvin map.
