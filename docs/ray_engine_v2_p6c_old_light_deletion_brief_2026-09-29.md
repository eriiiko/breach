# Ray engine v2 — P6c brief: the old light path goes (2026-09-29)

> **Issue** #12 (ray-engine-v2 arc) + #80 (3D marines crash). **Depends on:** P6b
> merged (`fire-12` `f33e475`). **Design:** `docs/ray_engine_v2_design_v3_2026-09-15.md`
> §4.3, §4.4 (THE table of what the old raycaster exposed and where each member
> goes), §7.3, §11.5 (P6c row). **Agent:** Opus (the design said Sonnet; the
> Raycaster still reaches into `physics_runner.py`, `exchange.py` and
> `field_edit.py` and owns a second emissive table, so this patch involves
> judgment calls rather than only mechanical deletion). **No HUMAN-TEST** —
> gate G2 proves the look unchanged. Merge on green by the orchestrator.

## 1. What P6c does

Since P6b the game is lit by the sweep's light channels. The OLD path — the C++
`Raycaster` render march, `fire_lights`, the per-frame light sources — survives
only behind F11 for Erik's A/B. He played P6b on 2026-09-29 and blessed it. P6c
deletes the old path so exactly one light producer exists. It also fixes #80,
because the fix is one line and lets Erik see the 3D marines lit by the new
field, which P6b claimed but nobody has seen.

## 2. Decisions (locked — do not re-derive)

1. **The Raycaster dies.** Execute design §4.4 for every row whose fate is P6:
   - delete `cpp/src/raycaster.{h,cpp}`, its bindings, CMake entries and the
     `/fp:strict` list line;
   - move whatever is still live to the home §4.4 names. That covers the
     emissive dials assigned to the engine (not to a `Raycaster`), `kelvin_ambient`,
     and `HEAT_SCALE` / `heat_quantize` / `heat_saturating_add` → `fixed_point.h`
     if P3 has not already moved them.
   Where a row's move already happened (P3, T6), verify it and move on.
   `PhysicsEngine::emissive` becomes the emissive table's ONLY owner:
   `tests/test_emissive_table.py` keeps its equality with the reference bake and
   drops the two-owner comparison.
2. **`renderer/fire_lights.py` dies** with its tests, `[render.fire_lights]` and
   the HUD's "K / n" count (§7.3).
3. **The A/B dies.** This covers:
   - F11 and its HUD row, `[render.lighting] new_light`, and the old flat ambient;
   - `LightingPass`'s march branch and `main.py`'s old-path branch;
   - `frame_lights`' old `light_sources` row and the old bits of `level_lights.py`.
   `tests/test_frame_lights.py` (the old-row oracle) is not simply deleted:
   every property it holds that still means something for the cone row moves
   onto `test_frame_lights_cones.py`; the rest go, each named in the report.
4. **`gmap.light_map` goes** once its consumers (`game_renderer.py`'s `lmap`,
   `lighting.py`) read the accessor directly (§4.3). Grep proves no consumer is
   left.
5. **`gmap.smoke_glow` goes** (only the old cast wrote it). Every reader (see
   `grep -rn smoke_glow`) reads the sweep's glow — P6b's new-mode branch becomes
   the only branch.
6. **`smoke_absorb_scale`:** the gas medium reads `[smoke]` from config directly,
   with the same value and one owner (§4.4 row "smoke_absorption…"). The look
   must not change (G2).
7. **The float `dyn_light_atten` plane goes**, and with it its entry in
   `tests/field_digest_spec.toml`'s `excluded_float_fields` (CLAUDE.md
   "Extinction planes": "the float dyn_light_atten dies at P6c").
   **The golden must not move.** If the Field digest rule says this edit needs a
   spec version bump, do NOT bump: STOP and report, because P7 owns the spec
   bump.
8. **`tools/lighting_demo.py` is PORTED, never deleted.** Its sliders feed the
   new path through the one assembly (`LightSpec` → cone rows →
   `sim.set_light`/`relight`). Arc #60's garden (Erik's parked worktree
   `breach-props`) uses it. If the port grows past roughly 300 lines, STOP and
   report.
9. **Level `[[light]] range` stays in the data.** The loader still parses it;
   the light ignores it (a sweep has no range). Say so in the loader/schema
   docstring. Migrate no level file.
10. **#80 — the 3D marines crash.** `unit_model_renderer.py:385` reads
    `anim.keyframeCount`, which is raylib 6.x's name; this machine's binding is
    raylib 5.5.0.4 (`frameCount`).
    - Read the frame count through ONE small helper that accepts either name.
    - Then drive the real game with M pressed (below) on the playground, CPU and
      `--cuda`.
    - Fix any further 5.5-vs-6.x binding mismatch the drive exposes in the 3D
      path. Anything else it exposes is a finding.
11. **The E2E driver becomes a tool.** The orchestrator's reproduction is
    `C:\Users\steen\AppData\Local\Temp\claude\c--Users-steen-projects-breach\6884c865-fdeb-47a9-96cc-ebc4885bc098\scratchpad\repro_3d_crash.py`.
    It runs the real `main.py` via `runpy` in a hidden raylib window, patches
    `pyray.window_should_close` (a frame counter) and `pyray.is_key_pressed`
    (scripted keys), and reported the #80 traceback on the first try.
    - Generalize it into `tools/e2e_drive.py`: `--level`, `--frames N`,
      repeatable `--press KEY@FRAME`, and pass-through `main.py` args
      (`--cuda`, `--control`, …). Exit non-zero on any exception.
    - Draft CLAUDE.md Tools row: *"E2E driver | `tools/e2e_drive.py` | THE
      headless end-to-end reproduction: the real `main.py` in a hidden window
      with scripted keys — bug fixes start here; never a parallel harness."*
12. **Archive = a banner, not a copy.** `docs/architecture/engine/08*` (the
    raycaster chapter) gets a dated banner: historical, the raycaster was deleted
    at P6c, pointing to design v3 and this brief. Git history keeps the code; no
    code goes into `docs/archive/`.
13. **CLAUDE.md follows the code.** Update the rows that name the old path or
    its fields:
    - Emissive table: one owner.
    - Extinction planes: the float plane is gone.
    - LightingPass, Frame lights, Light accessor: no A/B.
    - Radiation sweep: its "P6c deletes" notes.
    - Tile inspector, if it changes.
    Add the E2E driver row. Regenerate `stubs/breach_physics.pyi` from the CUDA
    build.

## 3. Properties to pin (gates)

- **G1** — Golden `2739f743…` unmoved. Full suite green on both builds, with the
  CUDA checks going through `tests/cuda_harness.py` subprocesses as always.
- **G2 — the new look is untouched.**
  - **Before deleting anything**, capture at the branch base (`f33e475`) a
    scripted playground scene: its level lamps, one onephase-style flashlight
    cone, one tile ignited through `debug_ignite` and run N ticks, and one smoke
    puff.
  - What to capture: `LightingPass`'s packed textures (`packed_a`, `packed_b`)
    and the gas medium's per-frame light/glow/absorption inputs, as hashes.
  - At HEAD they must be byte-identical. Record both hash sets in the report.
  - A mismatch is a STOP, not a re-baseline.
- **G3** — `tools/e2e_drive.py --level playground --press M@40 --frames 120`
  exits 0, both on CPU and with `--cuda`.
- **G4** — The float ratchet and ingress lint are green, and their counts only
  go down (`raycaster.cpp` leaves the lists).
- **G5** — No live `Raycaster`/`raycaster` identifier is left in `src/`,
  `renderer/`, `tools/`, `ui/`, `main.py` or `tests/`. Comments that pointed at
  `cpp/src/raycaster.h` point at the new homes.
- **G6** — Every deleted or rewritten test is listed in the report, one line
  each: what it tested and why that thing is gone. A test protecting a property
  of the NEW path is never deleted.
  - New tests assert properties. Each docstring names the property and the
    change that must break it.
  - Validate each new test by breaking the code once and restoring it.
  - A failing pre-existing test is a FINDING, never a target to bend the design
    to.

## 4. Out of scope (record as findings, do not fix)

- F10 double-bound (#59).
- `relight()` also running the heat sweep.
- Any sim arithmetic (the sweep, the fold, Fleck).
- The sky model — Erik ruled 2026-09-29: the horizontal sky stays, no roofs.
- Step vs shear for light (P6d, next).
- `light_q` in the digest and the rules/RL hook (P7).
- Other 3D-render bugs not caused by the raylib binding.

## 5. Systems

- **Uses:**
  - Light accessor (`light_field.py`)
  - Frame lights (the one assembly)
  - LightingPass
  - Emissive table (now single owner)
  - Fixed-point kits
  - Field digest + GOLDEN_AGGREGATE
  - CUDA harness
  - Level data layer (read-only here)
- **Creates:** `tools/e2e_drive.py` (row drafted in decision 11).

## 6. Execution

- **Worktree:** `C:\Users\steen\projects\breach-p6c`, on branch
  `12-p6c-old-light-deletion`, cut from `fire-12` at this brief's commit.
- **Builds:** `cmd /c cpp\build_cpu_home.bat` and `cmd /c cpp\build_cuda.bat`,
  first and after every C++ change.
- **Python:** `C:/Users/steen/anaconda3/python.exe`.
- **Suite:** `… -m pytest tests -q -p no:cacheprovider` from the worktree root.
- **Trap:** running `tests/_xarch_perfield_digest.py` as a script overwrites the
  tracked `tests/_xarch_perfield_DESKTOP-0E98HUV.txt`. Restore it from HEAD and
  never commit it.
- **Commits:** as you go, with prefixes `feat|test|docs|fix(#12): P6c -- …`
  (`fix(#80)` for the 3D fix). Stage explicit paths, never `git add -A`.
- **No merge, no push.** Leave both builds compiled from HEAD.
- **The report is your final message** (the harness blocks report files). It
  contains:
  - the §4.4 row-by-row disposition;
  - the G2 hashes;
  - the deleted/rewritten test list;
  - suite totals;
  - the commit list;
  - findings, one line each.
