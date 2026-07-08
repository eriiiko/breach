# EOS Phase-1.2 visual prototype — plan (agreed Erik + Claude, 2026-07-08)

**Nature:** throwaway spike (like spike0), float numpy, judged by eye. NOT canon, touches no live
engine path → simplest-honest-design, no adversarial design-gate. Branch `eos-prototype`,
dir `prototypes/eos/`. Input: `docs/eos_research_brief.md` + `docs/eos_research_report.md`.

**The question this must answer:** in Breach's **top-down** setting, how much better is **rung B**
(Kwatra semi-implicit compressible — real momentum, baroclinic curl, rolling vortices) than **rung A**
(Feldman–O'Brien prescribed-divergence incompressible — thermal expansion only)? And — the actual
hold-back — **is rung B cheap enough to run in realtime?** So the deliverable is GIFs **plus a cost
table** (ms/tick, substeps/tick at representative grids). Erik's eyes + the timing numbers are the gate.

## Solvers (3 columns)
- **Rung A** — Feldman–O'Brien prescribed-divergence incompressible; `P=C·N·T` drives the divergence
  source; smoke advected on the velocity; separate `wave_p`-style blast-impulse channel.
- **Rung B** — Kwatra semi-implicit compressible; explicit advection at `|u|`-CFL + implicit acoustic
  Poisson/Helmholtz (Gauss-Seidel); derive `P=C·N·T`.
- **Control** — semi-Lagrangian Stable-Fluids baseline (the thing both rungs must beat visually).

## Scenarios (S1–S5, `docs/eos_research_brief.md` §8)
S1 corridor blast · S2 room+door jet · S3 breach-to-vacuum · S4 fireball-over-smoke ·
**S5 tilted ship + breaking water container + non-uniform smoke** (water displacement pushes smoke;
uses a faithful ~40-line numpy shallow-water port of engine/07 §2 — *not* the real `.pyd`).

## Render (dead simple)
Walls = one flat color, open space = another (dark); smoke = brighter veil ∝ density (light-on-dark);
temperature = fire-color where hot; optional sparse velocity arrows. Colors are cosmetic, tune later.

## Patch plan (orchestrate — agents implement; each = fresh subagent)
- **P0 · Scaffold** — grid, fields (density/smoke, T, u, P; masks solid/vacuum/door; floor_height,
  tilt, water_depth), the 5 scenario builders + events, the shallow-water port (S5), a **pluggable
  solver interface** `step(state, dt)->state`, a **placeholder solver** so it runs end-to-end, the
  GIF render harness, **timing instrumentation** (ms/tick + substeps/tick + summary table), CLI
  `run.py --scheme --scenario --grid`. Self-verifies: emits GIFs + prints timing with the placeholder.
- **P1 · Rung A** · **P2 · Rung B** · **P-ctrl · Control** — three solvers dropped into the P0
  interface. Run in **parallel** (all depend only on P0). Each bakes its GIFs + logs timing.
- **Bake/index** — assemble the S×scheme GIF contact-sheet + the cost table. → **HUMAN-TEST: Erik.**

Checkpoint to memory at each patch boundary. Human-in-loop only at the final by-eye + cost gate.
