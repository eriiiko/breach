> **PARKED 2026-09-30 → issue #81 (Light as Game State).** This was the ray-engine v2
> arc's P7 brief, drafted 2026-09-29 and never run. It is INPUT to #81's design
> session with Erik, not a plan: his 2026-09-29 inputs (on #81) override it where
> they disagree, e.g. the flashlight cone is independent of the vision cone.
> Its critique: `light_as_game_state_p7_critique_2026-09-29.md`.

# Ray engine v2 — P7 brief: the light is game state (2026-09-29, v2)

> **Issue** #12, the arc's LAST patch. **Depends on:** P6c and P6d merged.
> **Design:** `docs/ray_engine_v2_design_v3_2026-09-15.md`:
> - §4.2–4.3 — the outputs and the one accessor;
> - §7.1 — light requested;
> - §8.3 — the digest joins "the day a rule reads `light_q`";
> - §11.5 — the P7 row.
>
> Also the P6b brief §2.1, which deferred the determinism of the emitter inputs
> to P7.
>
> **v2** folds in one Opus critique (2 blockers, 8 majors, 6 minors — all
> accepted, one as the design's own option). The disposition table is in the
> orchestrator's record.
>
> **Agent:** Opus. **HUMAN-TEST:** yes — Erik plays before merge.

## 1. What P7 does, and Erik's rulings behind it (2026-09-29)

**Erik's rulings:**
- *"All lights should in general be treated as in game and in sim"* — lights
  from fires and explosions included. The **cursor lamp is the one exception**:
  display-only.
- **Flashlights become each marine's game state:** on/off, and they point where
  the marine faces. They are *"pretty non mature … lay the foundation"*: the
  state and its wiring only, with no balancing.
- **Muzzle flashes do not exist yet.** Weapon lights will be designed later, and
  they are sim.

**What is wrong today.** `gmap.light_q` is fed through `sim.set_light` by
render/UI/wall-clock state:
- the cursor lamp;
- the flashlight's team/selected mode and its planning cursor-aim;
- the W6 weapon-effect queue;
- the sky, built by the renderer;
- F4, which turns the sim's light off.

Light is requested only while a window is open. `relight()` writes the planes a
rule would read. At load and reset the light is zero.

**After P7:**
- `light_q` / `light_flux_q` / `light_glow` are a pure function of synced state
  plus config.
- They are written only by the sim — each tick, and once at load/reset.
- They are computed in every run and read through the accessor.
- The display adds the cursor lamp on top, without ever touching the sim's
  planes.

**The digest is NOT touched (design §8.3, followed as written).**
- `light_q` and `flashlight_on` join the digest the day the first consumer
  reads them — the stealth rule, or the RL wrap-up #76, whichever comes first.
- Until then, flashlight, lamp, sky, transport and L° tweaks never force a
  golden rebuild.
- P7 **must leave the golden `2739f743…` unmoved** (light never feeds heat —
  P6a/P6b proved it bit-identical).
- The future bump's inventory is recorded in the appendix, for whoever does it.

## 2. Decisions

1. **One sim-side emitter assembly: `src/simulation/light_emitters.py`**
   (under `SIM_DIR`, so the ingress lint covers it).
   - **Level lamps.** Level `[[light]]` rows are quantized ONCE at load/reset
     through the one float→row door. Move `frame_lights.spec_cone_row` /
     `cone_rows` and the omni rule here; the renderer imports them. Move
     `LightSpec` and the beacon math out of `src/level_lights.py` (outside the
     lint's scope) into this module, or add the file to `EXTRA_SIM_FILES` —
     pick one and say which.
   - **Rewrite the "render-only" contracts:** `entities/light.py:9-12`,
     `level_loader.py:110-129` (`LightEntry`), and engine/15 §2.2 (level lights
     now feed sim state through this door). Keep the entity fields'
     `KIND_FLOAT_RENDER`: the door is here, not in the entity quantizer.
   - **Beacons, exact integer clock:**
     `center_q = (phase_q + (T·TURN·2^16) // period_ticks_q16) mod TURN`.
     - `period_ticks_q16 = round(period_s·tps·2^16)` is computed once.
     - `T` comes from the ruleset's own geometry,
       `round_index·ticks_per_round + round_tick`, never main.py's
       `monotonic_total_tick` with the two-phase 240 (it jumps a round per round
       under OnePhaseWEGO).
   - **The sky** (critique BLOCKER 1): `sky_for_level` moves here. It is built
     once at load/reset from `level.boundary` + `[light.sky.*]`, and the sim
     sets it on the runner.
   - **Flashlights:** decision 2.
   - **The slot:** the assembly is called from ONE conductor slot, pinned
     **between slot 6b (stamp flush) and slot 7 (physics)**, with the ordering
     comment. It must follow the unit moves (3–5) and `stamp_units` (6), or a
     walking marine's lens lands in its own freshly stamped body and the light
     flickers.
   - **The budget:** `LIGHT_CONE_BUDGET` is checked where a light is created
     (load; a unit gaining a flashlight at spawn), with a clear error — never
     raised mid-tick. Report the headroom in marines per channel.
2. **Flashlights are marine state.**
   - **The field:** a synced `Unit.flashlight_on: bool`, initialised from
     `[light.flashlight] default_on` (quantized once at construction, together
     with `[light.flashlight] rgb` — a Ctrl+R mid-run must not change a sim
     input).
   - **Who emits:** emission is gated at ASSEMBLY time on `alive and not
     is_zombie and flashlight_on`, because a marine can turn zombie later.
   - **Aim:** aim = `facing`. It passes through ONE float→`center_q` door, which
     needs only `/` and `round` — every facing writer produces exact n/65536 or
     π/2. The lens sits one tile outside the footprint (P6b's rule), placed
     with `unit_fixed.sin_rad`/`cos_rad`, never `math.*`.
   - **Spread:** it KEEPS reading `2 × [onephase] vision_cone_half_deg`. Erik's
     ruling (main.py:277-282) is that the lighting IS the facing indicator, so
     no separate spread dial.
   - **The team filter goes**, because all lights are sim: a rival team's
     flashlight shows, even under fog of war (one line to Erik).
   - **Deleted:**
     - the team/selected `flashlight_mode` and its L key;
     - the planning cursor-aim.
     Also update the stale docstrings: `ui/model.py:586-599`, `ui/draw.py:284`,
     `control_onephase.py:28`/`:72`, `main.py:402`.
   - **No toggle verb** in P7.
3. **The display adds the cursor; the sim's planes are the sim's**
   (critique BLOCKER 2).
   - Add `light_field.display_light(sim, extra_rows) -> LightView`, reached
     through a read-only `Simulation` method. It returns **the tick's planes +
     a CURSOR-ONLY field**.
   - The cursor-only field is a relight with an ambient-temperature plane
     (L°[0] = 0, so no thermal light), no sky, and only the display rows. It
     writes into renderer-owned scratch; `PhysicsEngine::relight` already takes
     caller planes and keeps its heat outputs in engine scratch.
   - Without display rows, `display_light` is exactly `read_light`.
   - The renderer never calls a producer directly — the accessor rule.
   - Measure and report the cost of one extra relight per tick while the WEGO
     cursor is on the map. `relight` also runs the heat sweep, a P6c finding
     that is not fixed here.
4. **The W6 weapon-effect lights stop emitting** (flame jet, miasma jet, plasma
   bolt). Their visuals stay. *(Stated to Erik as vetoable.)*
5. **Light is always computed:** `[light] enabled = true` (a sim setting,
   bound once in PhysicsRunner).
   - This **reverses** design §7.1 / P6a decision 2 ("headless training simply
     does not request"). Name the reversal in the report with the cost:
     per-tick on the playground, CPU and `--cuda`.
   - `light_at` and `observation` RAISE while light is off. The tile inspector
     (`hover_readout.py:272-276`) is guarded.
   - **STOP if the suite's wall time grows more than 25 %.**
6. **Light at load/reset:** `_reset_internal` computes the construction
   state's light into gmap's planes (a pure function of state) — critique
   finding 6.
7. **The old API goes:**
   - `Simulation.set_light` / `_apply_light` / `_light`;
   - the public `PhysicsRunner.set_light_cones` / `set_light_sky` (they become
     internal to the emitter slot);
   - the renderer-facing `relight` (it becomes internal: reset light and
     `display_light`).
   - main.py's per-tick cone assembly and its sky build.
   - F4 becomes render-only: it hides the light pass and never touches the sim.
   - **`tools/lighting_demo.py`** (ported by P6c onto `set_light`) is re-ported:
     its slider lights become display extras through `display_light`.
     Erik's parked `breach-props` uses it.
8. **The rules/RL surface — only what has a consumer:**
   - `light_field.light_at` (exists);
   - ONE RL observation hook, `light_field.observation(gmap)`: a read-only int64
     view (RL-batch habits).
   - No stealth rule and no AI change.
9. **`light_q` joins `SIM_FIELDS`** (`field_ab_harness.py:83-111`), so the A/B
   lockstep harness compares it per cell. It does NOT join `DIGEST_FIELDS`
   (decision in §1).
10. **Tests that break BY DESIGN** — dispose of each one in the report, one
    line each; this list is expected, not a finding:
    - `test_light_field.py:188-219` — restate: every synced field identical
      with display rows and display relights interleaved;
    - `test_light_field.py:322-343` — off-by-default / request survives reset;
    - `test_radiation_sweep_light.py:715-740` — keep its heat-isolation half as
      `[light] enabled` false vs true;
    - `test_onephase_ui_model.py:446-471` — modes, cursor aim, "never touch the
      sim";
    - `test_frame_lights_cones.py` — `EMITTER_INPUTS`, the flashlight/cursor
      specs, `set_light`;
    - `test_light_table.py:217` — survives only if `PhysicsEngine`'s raw C++
      default stays False (keep it False; the config turns it on).

    **Silent vacuity to fix:** `tests/cuda_radiation_sweep_cones_check.py:281-291`
    and `cuda_radiation_sweep_light_check.py:409` inject cones through
    `set_light_cones`, and the emitter slot now overwrites them. Rewrite both
    through the engine's direct binding or a lamp-bearing level — never a
    production back door.
11. **Docs follow the code.**
    - **CLAUDE.md rows:**
      - new "Light emitters (sim)" (draft: *"THE per-tick sim emitter assembly —
        lamps, beacons, flashlights, the sky — from synced state + config,
        slotted between 6b and 7; every game light is a row here; display-only
        extras (the cursor lamp) go through `light_field.display_light` and
        never reach gmap's light planes"*);
      - Radiation sweep: drop "REQUESTED, never always on … default OFF" and
        "not sim-pure until P7";
      - Light accessor: rules read the tick's planes; the display is tick +
        extras;
      - Frame lights: display extras only;
      - Blackbody: say an L° retune will move goldens once light is digested.
    - **Design v3:** dated pointers at §7.1 (always on) and §8.3 (digest still
      deferred — to the first consumer).

## 3. Gates

- **G0** — Golden `2739f743…` UNMOVED. The full suite is green on both builds.
  Report wall time before and after.
- **G1a — sim purity, committed, no GL.** Run a scripted scene headless. Then
  run the identical scene with the renderer's calls interleaved every tick:
  `display_light` with moving cursor rows, the F4 hide/show, a paused stretch
  with display calls.
  - `light_q` and every synced field are identical on every tick.
  - Validate by making `display_light` write gmap's planes, which must fail
    the gate.
  - Run it on the **playground** (lamps, marines facing different ways, one
    moving marine, one tile ignited BY HEAT, one smoke puff) and on
    **planetside_demo** (the sky).
- **G1b — sim purity through the real game, recorded once in the report.**
  - Extend `tools/e2e_drive.py` (never a parallel harness) with:
    - `--mouse X,Y@FRAME` (patching `get_mouse_x/y`);
    - `--digest-out PATH` (wrapping `Simulation.step`: per tick, the digest +
      a `light_q` hash);
    - `--scenario module:fn` (applied at tick 0: facings, heat, smoke — never
      through keys);
    - SPACE presses to run the rounds.
  - Extract main.py's sim construction into ONE function that both main.py
    and the headless replay call.
  - Compare over the common tick prefix, with a minimum tick count so the gate
    cannot pass empty.
  - Run it on CPU and `--cuda`.
- **G2** — Two headless runs in SEPARATE processes with different
  `PYTHONHASHSEED` give identical `light_q` on every tick.
- **G3** — CPU == CUDA (per-call and resident) `light_q` on G1a's scenes,
  tol 0.
- **G4 — load/reset light.** After construction and after `reset()`,
  `light_q` equals a relight of that state and is nonzero on a lit level.
- **G5 — flashlight properties:**
  - a facing change moves the lit region;
  - `flashlight_on = False` removes it;
  - a dead marine and a marine converted to a zombie stop emitting;
  - the lens sits outside the body for a MOVING marine, checked against the
    same tick's `dyn_light_atten_q`.
- **G6** — The ingress lint and float ratchet are green, and their counts only
  go down.
- **G7 — the display is exactly tick + cursor:**
  `display − tick == the cursor-only field` (int64); that field is ≥ 0; and
  `display == tick` with no rows.
- **Testing rules:**
  - Tests assert properties, and each docstring names the property and what
    breaks it.
  - Validate each by breaking the code once.
  - A red pre-existing test outside decision 10's list is a finding.

## 4. HUMAN-TEST (Erik)

- **WEGO (playground):** the marines now carry flashlights (NEW in WEGO), which
  point where each marine faces (North at spawn). The cursor lamp still lights,
  both paused and running.
- **Onephase:** every marine's flashlight follows its facing, with no
  selected-only mode and no cursor aim.
- **Weapons:** flame jets no longer carry their own glow. This is expected.
- **The first frame after load is lit.**
- **Otherwise:** the look is as before.

## 5. Out of scope

- The digest entry.
- The stealth rule and AI.
- A flashlight toggle verb.
- Flashlight balance.
- Weapon lights and muzzle flashes.
- Recording `light_q`.
- Light transport (P6d).
- A light-only relight (skipping the heat sweep) — record the cost only.

## 6. Systems

- **Uses:**
  - Simulation facade, Tick conductor, Unit, Config
  - Light accessor, Radiation sweep
  - A/B lockstep harness, CUDA harness
  - E2E driver (extended)
  - `unit_fixed` trig
  - Level data layer (read)
  - Ingress lint
- **Creates:** `src/simulation/light_emitters.py` (row in 2.11).

## 7. Execution

- **Where:**
  - Worktree `C:\Users\steen\projects\breach-p7`, on branch
    `12-p7-light-is-game-state`.
  - Python: `C:/Users/steen/anaconda3/python.exe`.
- **Builds:** `cmd /c cpp\build_cpu_home.bat` and `cmd /c cpp\build_cuda.bat`,
  first and after every C++ change.
- **Suite:** `-m pytest tests -q -p no:cacheprovider`. CUDA tests go through
  `tests/cuda_harness.py` subprocesses.
- **The xarch trap:** running `tests/_xarch_perfield_digest.py` as a script
  overwrites the tracked `tests/_xarch_perfield_DESKTOP-0E98HUV.txt`. Restore it
  from HEAD and never commit it.
- **Commits:**
  - Commit as you go, with prefixes `feat|test|docs(#12): P7 -- …`.
  - Stage explicit paths only; never `git add -A`.
  - End each message with the Co-Authored-By line.
- **No merge or push.** Leave both builds compiled from HEAD.
- **The report is your final message** (the harness blocks report files):
  - decisions as built;
  - G0–G7 results (G1b's comparison summary, the cost numbers);
  - tests added, and decision 10's dispositions;
  - suite totals and wall time;
  - commits;
  - findings;
  - the exact HUMAN-TEST commands.

## Appendix — for the future digest bump (NOT P7)

When the first consumer lands:
- **Fields:**
  - `light_q` joins `DIGEST_FIELDS` (it is already in `SIM_FIELDS` after P7);
  - `flashlight_on` joins `SYNCED_UNIT_FIELDS` AND `_unit_record()`
    (`field_ab_harness.py:69`, `:174-211`), with a named sub-hash in
    `_xarch_perfield_digest.UNIT_SUBFIELD_LABELS`.
- **Version:** bump `DIGEST_SPEC_VERSION`.
- **Regenerate:**
  - GOLDEN_AGGREGATE (`_xarch_perfield_digest.py:430`, ~20 importers);
  - LOOP_GOLDEN_TRAJ_DIGEST (`test_b6_logic_golden.py:118`);
  - DOORTEST_NOPHYS_TRAJ_DIGEST (`test_b1_signal_bus.py:128`,
    `test_b2_nodes.py:546`).
- **Do NOT regenerate:** SCALAR_ERA_DIGEST, the bake16 PNGs, the lit3d shader
  golden, and the per-host attestation `.txt` files.
- **Owed:** record the Ada re-attestation in `XARCH_PENDING.md`.
- **Non-vacuity:** assert on `field_digest` / the per-field hash, never the
  folded tick digest (the unit hash would change anyway).
