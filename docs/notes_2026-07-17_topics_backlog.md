# Notes dump — 2026-07-17 (near/mid-term wishlist)

Raw capture of Erik's jotted goals, same format as `notes_2026-07-05_topics_backlog.md`:
each topic tagged with its most-likely existing home so a future session can fold it
in rather than starting cold. Not canon — capture only.

---

## Topic 1 — Shockwave stirs up dust → smoke (map–physics interaction)

> If a shockwave (air pressure) passes dusty ground, it should create smoke as it
> passes by. Scenario: desert map (outdoors), detonate a bomb — you *see* the
> shockwave as a ring of stirred-up dust racing across the ground. Should cost
> ~nothing extra: we already solve the pressure equation for every tile every tick.

Sketch: a per-tile ground material flag ("dusty") + a threshold on the local
pressure gradient / wave_p amplitude; when exceeded, inject a small amount of
smoke/dust density (and maybe deplete a per-tile dust reservoir so the second
blast kicks up less). All local, no new solves — one conditional in a pass that
already touches the tile.

**Home:** `physics_field_interaction_map.md` (new field interaction:
wave_p × ground-material → smoke source) and the material/tile format in the
level docs (`level_editor_and_format_v2_proposal.md`) for the "dusty" flag.
Pairs naturally with Topic 4 (outdoor/planetside boundary conditions) — the
desert demo needs both.

---

## Topic 2 — Overpressurized rooms / wall-breaking overhaul

> Probably needs an overhaul — if we do it, do it properly.
> - Compute the pressure **gradient across the wall** (both sides), not just one
>   side's absolute pressure. If pressure rises on both sides equally, the wall
>   should NOT break.
> - Maybe take **wall thickness** into account — cost-dependent, but it can be
>   precomputed and stored in the wall tile.

Sketch: break condition becomes |P_sideA − P_sideB| > strength(material,
thickness) instead of P > threshold. Thickness ≈ distance to the nearest
opposite-side open tile along the wall normal — precomputable at level load /
edit time and baked into the wall tile (ties into the per-tile wall-normal
"bent tiles" idea from notes_2026-07-05 Topic 5, which would give the
across-the-wall direction for free). Interior walls of a uniformly
overpressurized compartment stay intact; only walls with a real differential
(hull walls, the door you're hiding behind) fail.

**Home:** wall-breaking rules live with the pressure/structure interaction —
`physics_field_interaction_map.md` + `map_and_physics_design.md`.

*Erik's correction (2026-07-17):* rung-B EOS is essentially landed (one
optimization patch left — GPU residency, stop streaming fields to CPU each
tick), and he doesn't see this as deeply interlinked with it. His mental
model of the whole feature: **each tick, READ P at the two sites flanking a
wall, compute the differential, and if it exceeds the break point, delete
one set of wall tiles.** If it gets more complicated than that, defer
indefinitely or find a simpler solution — this is a nice-to-have, not an
arc.

*Code audit (2026-07-17, session survey):* the overhaul is nearly free.
`GameMap.find_burst_walls` (`src/simulation/gamemap.py:909-981`) already
computes a spread = max−min over the wall's 4 neighbours, reading the EOS P
via the `atmosphere` alias (CPU numpy; the EOS design §6 explicitly
preserved this consumer). **The bug matching Erik's complaint:** solid
neighbours contribute p=0 instead of being skipped, so a straight wall
segment (solid on both sides along the wall) gets spread = room's ABSOLUTE
pressure − 0 — i.e. a room pressurised on both sides of a wall still bursts
it. Fix: skip solid neighbours (keep exposed-vacuum = 0, that one is a real
side). Bonus: with that fix, **thickness emerges for free** — inner tiles
of a 2-thick wall have no open neighbour, so the outer layer must fail
before the inner layer sees any differential. No baked thickness field
needed for v1. Remaining scope: the one-line-ish fix + re-tune
`burst_threshold` values (config.toml — hull ships at 0 = never bursts) +
`tests/test_wall_failure.py` update. Chat-sized, not an arc.

---

## Topic 3 — Procedural skeletal animation (BIG — Erik's priority item today)

> Investigate procedurally generated animations. I think "skeletons" that can
> move — and I think this fits perfectly for us: one humanoid skeleton reused
> with many animation sets. We'd need some exotic skeletons as well — a spider,
> if possible an octopus. A bigger project I'd like to put a lot of effort into;
> should perhaps be slotted in somewhere after physics engine v1 is done.

**Home:** own doc — see `procedural_animation_brainstorm.md` (investigation
brief, same session as this capture). Slotting: after physics v1 per Erik;
does not block the ML/training arc (units already render without it).

---

## Topic 4 — Map editor: free boundary conditions (space vs planetside)

> Allow maps that are on planets — I want to set the boundary conditions freely
> per map. Space: atmosphere = 0 at boundary. Planetside: ~1 atm at the
> boundary — or perhaps not *force* it to 1 but just absorb everything
> perfectly; something realistic that models air continuing "indefinitely".

Sketch: a per-map (maybe per-edge) boundary-condition setting in the level
format + editor UI. Three candidate modes:
1. **Vacuum** (current space behavior): boundary sinks to 0.
2. **Fixed ambient** (Dirichlet): boundary held at ambient N/T/P — simple, but
   hard-clamping can reflect artifacts back in.
3. **Non-reflecting / absorbing** (Erik's instinct): outgoing waves and flow
   pass through as if the field continued forever — the "right" feel for
   planetside; standard technique is a sponge/damping layer or characteristic
   (radiation) boundary conditions.
Erik leans 3 over a hard 2. Wind-in-from-boundary (the 2026-06 windy-level
idea, `mission_ideas.md`) would be a fourth mode or a parameter on 2/3.

**Home:** boundary handling lives in the atmosphere/EOS solves — belongs in
`eos_refactor_design.md` as a requirement + a level-format field
(`level_editor_and_format_v2_proposal.md`).

*Erik's update (2026-07-17):* rung-B EOS is almost complete — only the final
optimization step remains (GPU residency: stop streaming all fields to the
CPU each tick). So this is a **candidate NEXT physics project**, spec'd
against the landed EOS rather than into its design. Priority: boundary
conditions are "pretty much a must" — UNLESS it turns into weeks of EOS
rewriting, in which case planetary missions wait ("Breach 2" 🙂) or we use a
cheat (sponge/damping layer — which is in fact the standard non-reflecting
technique anyway, so the cheat and the proper solution may be the same
thing). Sequencing note: spec it BEFORE/ALONGSIDE the residency patch —
boundary handling touches the same kernels the residency patch is about to
freeze into GPU-resident form, so landing BC hooks first (or in the same
pass) avoids reopening it.

*Code survey (2026-07-17):* NOT a big refactor. Key facts:
- The literal grid edge is closed/reflective everywhere (`mirror_idx`,
  `eos_solver.cpp:47-53` + CUDA mirror); "space" boundaries are made in
  LEVEL DATA — a border ring of SPACE tiles (code 9 → `is_vacuum`), which
  the MG solve already treats as Dirichlet P=0 (`eos_solver.cpp:717`) and
  bulk transport treats as a mass sink.
- So planetside = **an AMBIENT border-ring tile type**, symmetric to SPACE:
  MG pins P=P_amb instead of 0 (generalize the existing exclusion-pin to
  carry a value), bulk transport resets its N to ambient each tick
  (infinite reservoir both directions), smoke absorbs to 0 there. All
  local per-tile edits in existing kernels; no new solve structure.
- Reflection control: if the hard ambient ring rings/echoes, add a 4-8 tile
  sponge band (damp u, relax N/T toward ambient) inside it — the standard
  non-reflecting trick, all Q16 muls, determinism-clean.
- **Existing goldens untouched**: current levels contain no AMBIENT tiles,
  so the new branch never executes on them — zero re-baseline risk for
  space maps.
- S8a heads-up found in passing: `cuda_s8a_residency_spec.md` is PRE-EOS
  (enumerates the retired field set incl. wave_p) — it needs a rewrite
  against the rung-B fields before that patch runs. Natural moment to fold
  the BC hooks into the same kernel-touching pass. Current per-tick D2H
  sync: `cuda_eos_step.cu:328-333` (wind, temperature, gas planes; P at
  :469).

---

## Also captured this session

- Smoke-divers mission scenario → appended to `docs/missions/mission_ideas.md`
  (game/mission idea, not an engine item).
