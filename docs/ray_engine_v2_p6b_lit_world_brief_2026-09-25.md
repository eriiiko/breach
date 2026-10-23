# P6b brief — the renderer lights the world from the sweep (2026-09-25)

Ray-engine-v2 arc, issue #12. This is the second of P6's three steps. P6a
(merged) made the sweep compute light, requested and default off. **P6b makes
the game USE it**:
- the accessor;
- `LightingPass` as a consumer;
- cone emitters;
- the directional sky boundary.

**HUMAN-TEST: Erik plays it and judges the look before it merges.**

Design:
- `docs/ray_engine_v2_design_v3_2026-09-15.md`: the P6b row of §11.5, and
  **§4.1–4.4, §5, §7, §8.5, §11.3**;
- the P6a brief, `docs/ray_engine_v2_p6a_light_channels_brief_2026-09-25.md`;
- the P6a merge commit's message, for what exists: `light_requested`, `light_q`,
  `light_flux_q`, `light_glow`, `L_FINE_BITS = 37`, `optics_fixed.dequantize_light`.

## 1. What P6b builds

1. **`src/simulation/light_field.py`**, the ONE accessor (§4.3).
   - `read_light(gmap)` gives the renderer dequantized float views and the
     normalized flux vector.
   - `light_at(gmap, y, x)` gives rules and RL the integer value. P7 consumes it.
   - Nothing outside it may know how the field was produced.
2. **`LightingPass` becomes a consumer.**
   - It uploads the accessor's field into `light_tex_a`/`light_tex_b` once per
     sim tick. The shader samples them per frame as today.
   - It stops calling a producer.
   - Keep the texture layout if you can, because 3D units sample the same
     textures (`renderer/lit3d.py`, the marine byte-identity gate).
   - `gmap.light_map` becomes a scalar derived from the accessor (§4.3). It is
     deleted in P6c.
3. **Cone emitters (§4.1)** are rows of `(cell, rgb_q, angle_center,
   angle_spread)`.
   - Each emits only into the ordinates inside its beam: no range, ray count,
     jitter or heat.
   - They land in the reference first, then the CPU sweep, then the CUDA twin,
     at tol 0.
   - Every light the game shows today becomes one:
     - level lamps and beacons (`level_lights.py`);
     - unit flashlights (`ui.flashlight_cones`);
     - the cursor lamp;
     - the W6 transient emitters (spray jets, plasma bolts).
   - The existing assembly seam (`renderer/frame_lights.py` + `level_lights.py`)
     survives with its output row changed (§4.4), and is assembled once per sim
     tick for the engine. Never build a second assembly.
4. **The directional sky boundary (§2.7, §4.1).** The light ring carries
   per-ordinate RGB instead of the flat shader constant
   `u_ambient = (0.18, 0.18, 0.22)`:
   - an overcast sky is uniform;
   - a sun is a few ordinates;
   - space is dark.

   A small **flat floor stays a separate dial**, never a substitute (design §11
   systems). Pick the smallest honest source for the sky values (per level or
   per boundary type, from data or config) and say why. A sealed compartment
   must go properly black apart from its own lights and the floor.
5. **Light is requested** by the game whenever it renders, on both backends.
   Headless runs, benches and RL never request it (P6a decision 2).
6. **Keep the old path, behind a dev toggle, for this patch only.** The
   raycaster render march, `fire_lights`, and the per-frame sources stay
   switchable, so Erik can flip old/new light in the same scene during the play
   test. P6c deletes them. The default is the NEW light.

## 2. Decisions this brief makes

1. **Determinism of the emitter inputs is P7's problem, not P6b's, but it must
   be visible.**
   - Today several lights come from render or UI state, not sim state:
     - the cursor lamp follows the mouse;
     - flashlight mode and the selected unit are control-source state;
     - the W6 emitters come from the renderer's effect queue.
   - Feeding them to the sweep in P6b is allowed, because `light_q` is not
     digested and no rule reads it yet.
   - **List every emitter's input source in the report, marked sim or render.**
     Add one line to CLAUDE.md's light rule: `light_q` is not sim-pure until P7
     separates those inputs.
   - Do not redesign flashlights or controls here (one system at a time).
2. **Calibrate brightness against the old path.**
   - A lamp, a flashlight and a burning crate should read about as bright as
     under the old raycaster at the default exposure. Tune the render-side gain
     or exposure only: render dials, dequantized at the render read, never
     written back.
   - The integer field and L° are not tuned.
   - Report the before/after numbers you matched.
3. **The GPU per-call path is fine at playground size** (P6a: 3.05 ms against
   the CPU's 3.27). Residency is S8's; do not start it here.

## 3. Properties to pin

Write these from this section, never from the implementation. Each test's
docstring names its property and the change that breaks it. Validate each one
by breaking the code once.

1. **§8.5, the eye is monotone in the rules field.** On a real scene, the
   render's field is a monotone function of `light_q` per channel. *Breaks on
   any render-side post-process that reorders brightness.*
2. **Cone emitters, bit for bit.** They are gate 0 against the reference and
   CPU == CUDA at tol 0. A cone lights only inside its beam's ordinates, and a
   zero-spread cone lights one ordinate.
3. **Sky boundary.**
   - A uniform sky equals the old flat ambient at the boundary. Within 0.1 %
     is enough; the flat floor is excluded.
   - A sealed room with no lights has `light_q` exactly 0, so the eye sees only
     the floor dial.
   - An opening lets sky light in, and it decays with the step transport.
4. **Every old light has a new form.** The report maps each light type to its
   emitter. A scene with each type lights up. *Breaks if a light type is
   dropped.*
5. **Heat is untouched.** With light requested the heat planes are
   bit-identical (the P6a gate, re-run under the game's own conductor), and the
   golden does not move.
6. **The marine byte-identity gate** (`test_lit3d_extraction.py`) stays green,
   or is restated with one line of reason if the texture layout had to change.

A failing pre-existing test is a finding, never a target to bend to. Render
tests that pinned the old march move to P6c's disposition list (§11.2), unless
this patch must touch them.

## 4. The HUMAN-TEST: what Erik plays

The report must say exactly how to run it:
- level, flags, keys, and the old/new toggle key;
- what working looks like;
- the known look trade-offs from design §7.2, so Erik is not surprised:
  - step transport's faint four-point cross around an isolated small light in
    smoke;
  - banding inside a flashlight beam (16 directions);
  - fire light is orange with no blue until 1612 game (`blackbody.py`'s
    physics);
  - sealed rooms go dark.

## 5. Systems

**Uses, never duplicates:**
- the Radiation sweep and its CUDA twin;
- LightingPass ("never a second raycast");
- Frame lights, the one assembly seam, and `level_lights.py`;
- the Lit-3D seam (`lit3d.py`);
- Blackbody;
- the Dequantize convention;
- the Tile inspector: add light rows if cheap;
- the Integer reference, spec first for cone emitters and the sky;
- the RL-batch habits.

**New system: the light accessor (`light_field.py`).**
- Draft rule: *"THE light read: the renderer, rules and RL read light only through
  `read_light`/`light_at`, never a sweep plane or a producer."*
- It goes into CLAUDE.md, and the LightingPass row changes its input.

## 6. Out of scope

- Deleting the old path, `fire_lights`, `light_cull`/`heat_cull`, the smoke
  dials on the Raycaster, and the float `dyn_light_atten`. All of that is P6c.
- `light_q` for rules, the digest, and making the emitters sim-pure. That is P7.
- Controls and flashlight design.
- The heat channel.
- The EOS (#72).

Anything else you notice gets one line in the report.

## 7. Execution

- **Where:** branch `12-p6b-lit-world` off `fire-12`, in worktree `breach-p6b`,
  one writer. **Model: Opus.**
- **Builds:** run `cpp\build_cpu_home.bat` and `cpp\build_cuda.bat` from the
  worktree, first and after every C++ change.
  - Python: `C:/Users/steen/anaconda3/python.exe`.
  - Suite: `pytest tests -q -p no:cacheprovider`, run from the worktree root.
- **Commits:** `feat(#12): P6b -- …` / `test(#12): …` / `docs(#12): …`.
  - Stage explicit paths only, never `git add -A`.
  - End each commit with the Co-Authored-By line.
  - Do not merge, do not push.
- **Trap:** running `tests/_xarch_perfield_digest.py` as a script overwrites
  the tracked `tests/_xarch_perfield_DESKTOP-0E98HUV.txt`. Restore it from HEAD,
  and never commit it.
- **Report:** your FINAL MESSAGE is the report. It is the orchestrator's
  technical record and must include:
  - the emitter-input map (sim vs render);
  - the sky-source choice;
  - the brightness calibration;
  - every test added or rewritten, one line each (its property, and what
    breaks it);
  - suite totals on both builds;
  - the commit list;
  - findings outside the patch, one line each;
  - the exact HUMAN-TEST instructions.
