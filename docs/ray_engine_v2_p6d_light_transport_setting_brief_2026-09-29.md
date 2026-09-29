# Ray engine v2 — P6d brief: light keeps both transports, step and shear (2026-09-29)

> **Issue** #12. **Depends on:** P6c merged. **Design:**
> `docs/ray_engine_v2_design_v3_2026-09-15.md` §2.4 (the transport
> parameter: "shear for heat, step for light — a parameter, never a fork"),
> §7.2. **Agent:** Sonnet — the arithmetic already exists and is gated, so this
> is plumbing, one config setting, and gates on the live path. **No HUMAN-TEST gate:** the
> default stays `step`, so nothing changes unless someone flips the setting.
> Erik plays the switch after the merge and picks the default himself, as a
> one-line config change.

## 1. What P6d does

Erik ruled on 2026-09-29 that he wants to try shear light, and asked
*"can't we support both?"* Yes. The sweep already carries the light group's
transport as a parameter:
- the reference: `LightGroup.transport`;
- the CPU sweep: `LightChannels.transport`;
- the CUDA twin: its light walk (the step anti-diagonals) is topological for
  both transports.

Gate 18 and `tests/test_radiation_sweep_light.py` already cover light step and
shear, with GPU == CPU. The ONE place the choice is fixed is
`cpp/src/physics_engine.cpp:553`, which hardcodes
`lc.transport = RadiationSweep::TRANSPORT_STEP`. P6d turns that constant into a
setting that lives permanently.

## 2. Decisions (locked)

1. **`[light] transport = "step"`** in `config.toml`, validated at the config
   door: exactly `"step"` or `"shear"`, and anything else raises. It is bound
   ONCE, in `PhysicsRunner` (solver params are bound there only), onto a new
   `PhysicsEngine.light_transport` attribute (an int, `TRANSPORT_STEP` /
   `TRANSPORT_SHEAR`, bound to Python).
   - `step_tail` and `relight()` read it at the one sweep invocation
     (`run_sweep_`). Both backends, per-call and resident, must honour it.
   - It is a SIM setting, not a render dial: at P7 the rules read `light_q`.
   - Heat's transport does not change and is not configurable.
2. **NO KEY.** Erik ruled on 2026-09-29 that the setting lives in the toml
   only: *"I try to get away from all these shortcut keys."* To try shear,
   edit `config.toml` and restart. Do not add a key, a HUD toggle, or a
   runtime setter beyond what config binding needs. The HUD may SHOW the
   active transport as a read-only line if one already lists light state.
3. **Nothing else moves.** Cone projection, the sky, the books, the currency
   and L° stay as they are. If shear light needs any arithmetic change, STOP
   and report: the reference would have to change first.

## 3. Gates

- **G1** — Golden `2739f743…` unmoved. The full suite is green on both builds.
- **G2** — On the LIVE conductor (a real `Simulation` with light requested, the
  playground with its lamps, one ignited tile and one smoke puff, N ticks):
  - with `transport = "shear"`, GPU == CPU at tol 0 on `light_q`,
    `light_flux_q`, `light_glow` and the light books, per-call and resident;
  - the same holds for `relight()`;
  - this goes through the CUDA harness subprocess idiom.
- **G3** — Heat stays bit-identical whichever light transport is chosen: every
  synced field is compared after N ticks, step vs shear.
- **G4** — The switch is REAL (non-vacuous). On the same scene, step and shear
  `light_q` differ, and a room lit only through a corridor receives MORE light
  under shear than under step. This is the scheme study's corridor-throughput
  property (report.md, "What the heat schemes look like": step 0.52 vs shear
  0.90 of exact). Assert the ordering, never a number.
- **G5** — The config door refuses an unknown transport. Keep this test
  pure-Python.
- **Testing rules:** tests assert properties, and each docstring names the
  property and what breaks it. Validate each test by breaking the code once.

## 4. Optional, if everything above is green

**A glowing strip in the playground.** Erik said "lights on the ground may be
good for now … not prio". Add a short row of dim level lamps along one wall
face via `level_lib` (the ONE writer, never a hand edit), for comparing step
and shear on an extended source.
- If any test pins the playground's lamp count, that is a FINDING (a snapshot
  test). Report it and do not bend the test.

## 5. Systems

- **Uses:** the radiation sweep (the transport parameter), Config (the bind in
  PhysicsRunner only), the Simulation facade, the CUDA harness, the E2E driver
  (`tools/e2e_drive.py`, P6c) for one run of the playground with `transport = "shear"`, and the level data
  layer.
- **Creates:** none. Update the CLAUDE.md "Radiation sweep" row, where
  "step for light at P6" becomes "light's transport is the `[light] transport`
  setting, default step".

## 6. Execution

- **Worktree:** `C:\Users\steen\projects\breach-p6d`, on branch
  `12-p6d-light-transport-setting`.
- **Builds:** `cmd /c cpp\build_cpu_home.bat` and `cmd /c cpp\build_cuda.bat`.
- **Python:** `C:/Users/steen/anaconda3/python.exe`.
- **Suite:** `-m pytest tests -q -p no:cacheprovider`.
- **The xarch trap:** restore `tests/_xarch_perfield_DESKTOP-0E98HUV.txt` from
  HEAD if a script run overwrites it.
- **Commits:** prefixes `feat|test|docs(#12): P6d -- …`. Stage explicit paths.
  End each message with the Co-Authored-By line.
- **No merge or push.** Leave both builds compiled from HEAD.
- **The report is your final message:** gates, the test list, suite totals,
  commits, findings, and the exact config line for Erik to try it.
