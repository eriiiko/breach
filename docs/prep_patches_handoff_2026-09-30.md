# Prep patches for the "explosion + air" session — handoff (2026-09-30)

**Status:** ready to execute. Authored by Claude (Fable) at the end of the
pre-weapons triage walkthrough; every decision below is Erik's ruling of
2026-09-30 (issue comments are the authoritative record: #31 session log, #4,
#59, #62, #79, #9; digest in `docs/issue_triage_pre_weapons_2026-09-25.md` §7).
**Execute per the `autonomous-patch-workflow` skill's "resuming from a design
doc authored elsewhere" mode.** Do not re-derive the rulings; a missing field
goes back to Erik (post on #31), never inferred.

**Goal.** Step 1 of the accepted seven-step ordering: the mechanical patches
that need no screen time, so that Erik can sit down to the "explosion + air"
session (step 2) with a working test level, an honest heat-delivered grenade,
and no confounders on the keys and readouts he will use. All five are
**pre-authorized as mechanical: auto-merge on green.** None is feel-gated.

## 0. Standing rules for this run

- **Preflight:** the orchestrating session runs the newest Opus (or Fable).
  Implementers: Sonnet for oracle-gated patches, Opus where judgement is
  needed (tiers below). **One implementation agent at a time.** Checkpoint
  after every patch: one comment on **#31** (the session's home) and the
  memory note `project_pre_weapons_triage_review.md`.
- **Branch and worktree discipline — explicit, not the Agent tool's
  `isolation: "worktree"`.** *Incident 2026-09-30: that isolation cut the
  worktree from `main`, not from the checked-out `fire-12`; the agent then
  reported the current build as "stale" because its code was 349 files old.*
  Each patch: from the main checkout run
  `git worktree add ../breach-<slug> -b <issue#>-<slug> fire-12`, and the
  agent VERIFIES `git log -1` in its worktree equals `fire-12`'s tip before
  touching anything. One writer per worktree. The main checkout is for
  merging only.
- **Build:** the C++ sources do not change in this run. Copy the main
  checkout's `cpp/build/Release/` (the CPU `.pyd`) into the worktree at the
  same path; never rebuild. Python on the Home Desktop is
  `C:/Users/steen/anaconda3/python.exe` (base, 3.11); on other machines see
  `docs/dev_setup.md`. Tests: `<python> -m pytest tests -q`, never bare
  `pytest`.
- **Merge rule:** gate green ⇒ the agent that ran it merges `--no-ff` into
  `fire-12`, deletes the branch, pushes `fire-12`. Green means: the patch's
  own oracle passes; `pytest tests -q` has **0 failed** (baseline on
  `fire-12` @ `5580081`: 3589 passed, 31 skipped without a GPU build,
  5 xfailed); **`GOLDEN_AGGREGATE` unchanged** (`tests/_xarch_perfield_digest.py`;
  none of these patches may move a digest). Never `git add -A`; stage paths.
- **Stop-and-report conditions** (post on #31, leave the branch, move on or
  stop): a pre-existing test goes red (a finding, never a target); a golden
  moves; a question of design surfaces (a new control scheme, how a heat
  deposit should split between gas and solids, any explosion-design matter,
  any `k_drag2` value). Tests assert **properties**, never snapshots, and
  each names the change that breaks it.
- **Out of scope for this run:** tuning any dial or `heat_amount` beyond the
  provisional values below; #73; #72; #82's sweep; anything CUDA; #22.

## 1. The patches (in this order)

### P1 — #59: the B key stops toggling bilinear filtering (Sonnet)

- **What:** `renderer/game_renderer.py:1471` toggles
  `self.lighting.toggle_bilinear()` on `KEY_B`; B also arms explosives in
  the input handler. Erik will arm explosives all session, and the toggle
  would confound every look judgement. Erik's standing rule (2026-09-29):
  fewer shortcut keys; a switch is a `config.toml` setting.
- **Change:** delete the B branch in the renderer's debug key block; make
  the filter mode a setting `[render.lighting] bilinear = true|false` read
  once where `LightingPass` is constructed (default = today's shipped
  state); drop or re-word the HUD line at `:1352` (`"B  bilinear"`); keep
  `toggle_bilinear` only if a tool still calls it (grep; otherwise delete).
  Update the comment at `:1478` that records the collision.
- **Oracle:** a property test that `LightingPass` binds the config value
  (both values), and one that the renderer's key dispatch no longer reaches
  `toggle_bilinear` on B (monkeypatch the key query; the G→F10 fix of
  2026-09-01 is the precedent for how this block is tested).
  `tools/e2e_drive.py --level playground --press B@40 --frames 80` exits 0
  on CPU. Suite green.
- **After merge:** close **#59** with the commit hash (Erik's ruling).

### P2 — #62: the pressure overlay reads `atmosphere` alone (Sonnet)

- **What:** `renderer/pressure_overlay.py:70` renders
  `atmosphere + wave_p`; since EOS P3 `wave_p` is the previous tick's
  pressure buffer (~1 atm DC by design), so a 1 atm room shows ~2 atm.
  Display-only (verified in the triage).
- **Change:** render the dequantized `atmosphere` alone; fix the module
  docstring (`:4`) and any legend / colour scale that assumed the sum;
  `renderer/hover_readout.py` does not read `wave_p` (checked) — leave it.
- **Oracle:** a property test on the overlay's pure packing function: a quiet
  1 atm room packs to 1 atm, not 2 (build a small `GameMap`, no stepping
  needed). Suite green. No digest.

### P3 — #79: ignition by delivered heat, and the proxy rows (Sonnet)

- **What:** `src/simulation/physics.py::apply_explosion` (l.138–149) queues a
  heatless `fire = max(...)` on every flammable tile inside 0.7·radius. Since
  R3 that is a decorative fire (no O2 draw, no fuel burn, ~125 s to die).
  Erik ruled fix direction 2: **delete that ignite block** (and its docstring
  line); ignition comes from delivered heat through
  `combat.py::apply_temperature_ignition`, which already lights any tile at or
  above its `ignition_temp`.
- **Config (the proxy rows, provisional values, Erik tunes at the session):**
  keep `[payloads.frag_standard]` byte-identical (the mass-only CONTROL).
  Add beside it:
  - `[payloads.frag_heat]` — identical to `frag_standard` except
    `pressure = 0.0`, `heat_amount = 3200.0`, `heat_radius = 5.0` (heat only).
  - `[payloads.frag_hot_mass]` — identical to `frag_standard` plus the same
    `heat_amount` / `heat_radius` (mass AND heat).
  Provisional `heat_amount` follows the plasma-splash derivation in
  `config.toml` (`heat / thermal_mass` = the centre jump; 3200 ignites wood
  instantly at the centre). **Note for the session, do not act on it here:**
  a gas tile's capacity is `N·c_v ≈ 0.0077` in the same unit, so the same
  deposit rails the air at the centre (`T_MAX_PHYS`, the excess counted);
  air holds ~130× less heat per tile than wood. Whether the deposit should
  split between gas and solids is the design session's question (#31 inbox).
  Add a comment block over the three rows saying exactly that, and that the
  hand grenade's ammo row (`[ammo.grenade_frag]`, its `payload` column) is
  the one line Erik edits to switch variants.
- **Oracle (the property #79 names):** `frag_heat` executed beside a
  furniture crate on a small level, stepped ~3 s: the crate's `fire > 0` AND
  its O2 falls AND its fuel falls (an honest fire, `hotf`-driven). Also: the
  shipped `frag_standard` no longer writes `fire` anywhere (assert no
  `fire` edit is queued by `apply_explosion`). `tests/_blast_bench.py` still
  runs. Suite green; goldens untouched (the golden scenario has no blast).

### P5 — the explosion studio: a timed-charge entity + the level (Opus)

*(P5 before P4: the level is what P4 is verified on.)*

- **Entity `timed_charge`** in `src/simulation/entities/` (registry row, the
  editor sees it for free, one serializer): fields `x`, `y` (tile ints),
  `payload` (the `[payloads.*]` row name, validated at load against the
  `PayloadTable` — an unknown name is a loud load error), `first_at_s`,
  `period_s` (0 = fire once), `enabled`. Runtime sim-side (beside the other
  entity runtimes), ONE call line in `Simulation.step()`'s slot 9e with an
  ordering comment: at ticks `first + k·period` (via
  `config.ticks_from_seconds`) it calls `execute_payload(...)` exactly as the
  grenade fuse-out / door-charge sites do (find the canonical call; same
  argument order, the sim `rng` passed but never drawn). Per-tick state
  (`next_fire_tick`) is a synced runtime row in the serializer and in the
  ENTITY digest section, **presence-gated**: a level without charges hashes
  nothing new, so every golden stays. Deterministic: two runs of the level
  produce identical digests (property test).
- **Generator `tools/gen_explosion_studio.py` → `levels/explosion_studio/`,
  through `level_lib` (the ONE writer; CLAUDE.md "Level data layer").**
  `tools/gen_fire_studio.py` is the layout precedent but it bypasses the
  writer with `np.savetxt` (#67 item 3) — do not copy that: if `level_lib`
  has no tilemap-CSV writer, add one THERE (LF, the bytes
  `write_tilemap_csv` was meant to write) and use it. Layout: a hull box
  (~48×48 at 1 m tiles) with the vacuum band outside (boundary "space"),
  two or three interior rooms with doorways, furniture crates near the
  charges (fuel), a few lamps; **four charges**: three of increasing yield
  inside (`payload` = three new provisional rows
  `[payloads.studio_small|medium|large]` = `frag_heat` at 1×, 2×, 4×
  `heat_amount`, radius 4/5/6) at `first_at_s` = 2, 4, 6 and the fourth at
  the hull (`demolition_c4`-scale `wall_damage`, plus heat) at 8 s so the
  breach vents smoke to vacuum; **all with `period_s = 5`** so the blasts
  repeat in the emptied room. Erik edits the rows and the level afterwards.
- **Continuous time:** Erik wants no two-phase pause. Check whether an
  existing control scheme runs continuously with the normal keyboard/mouse
  UI (`--control gamepad` → `ContinuousRealtime`, `src/control_gamepad.py`;
  `--control onephase` has ~4 s rounds). If one does, document the launch
  line in the level's README. **If none does, STOP on that item and report
  on #31: a new control scheme is a design matter, not this run's.**
  ACCEPTED GAP until then: `--control wego` with the sim unpaused runs
  between round seams; the charges fire on the sim clock regardless.
- **Oracle:** property tests — a charge fires at its tick (the payload's
  `ExplosionEvent` appears; a `fire`-free level stays fire-free until then);
  it repeats at its period; `enabled = false` never fires; serializer
  round-trip; digest presence-gating (playground golden unchanged);
  determinism (two runs, equal digests). `tools/e2e_drive.py --level
  explosion_studio --frames 400` exits 0 on CPU (and `--cuda` where a GPU
  build exists; else say so). Suite green.

### P4 — #9: broken walls look broken, one path (Opus, E2E first)

- **Erik's ruling (2026-08-29):** when solved, ONE robust wall-destruction →
  render-invalidation path whatever destroyed the wall (pressure burst,
  explosion, door charge), reproduced end-to-end first.
- **Do:** (1) reproduce with `tools/e2e_drive.py` on `explosion_studio` (P5)
  or the playground: a wall destroyed by the fourth charge / a door charge
  keeps its baked art. Find where wall art comes from (`WorldComposite`, the
  baked diffuse from `tools/bake_level_art.py`, the tileset) and how a tile
  change reaches the renderer today (`GameMap.on_tile_changed` patches the
  sim planes in place; `destroy_wall` / `seal_tiles` are the only topology
  writers). (2) Design the minimal single seam: topology change → a
  render-side dirty set → `compose_world()` re-blits those tiles from the
  tileset's floor/rubble art once per frame; no per-frame scans, no
  per-cause code. (3) Oracle: a unit test on the seam (a destroyed wall marks
  its tile; the composite consumes and clears it; every topology writer
  reaches it), the e2e run exits 0, and where feasible a pixel probe of the
  world RT before/after. Render-only: no digest. Erik's eye check happens at
  the session.

## 2. Settled questions (with the reason) and accepted gaps

- **Why heat rows and not a new mechanism:** the W6 `heat_amount` /
  `heat_radius` columns already run through `execute_payload`, land on gas
  and solids, and are booked (Erik, 2026-09-30: "low-hanging fruit"; the
  explosion design proper is a later session).
- **Why `frag_standard` stays byte-identical:** it is the control Erik
  compares against; changing it would also move `tests/_blast_bench.py`'s
  baseline for no reason.
- **Why no hot-reload work:** the drag dials bind at `PhysicsRunner`
  construction and payload rows at `Simulation` construction; edit + restart
  is Erik's stated preference (config over keys). ACCEPTED GAP.
- **Why the charge is an entity and not a script in `main.py` or the e2e
  driver:** registry = editor, one serializer, digested, deterministic,
  reusable for benches and RL scenarios. Draft rule below.
- **`k_drag2` stays 0.0 in this run.** Its value is the session's (#4 P3).
- **ACCEPTED GAP:** the provisional `heat_amount` values are placeholders;
  the honest yield → deposit mapping and the gas/solid split are #31's
  design session.

## 3. Systems

**Existing canonical systems this run must use (CLAUDE.md inventory):**
Payload executor (`payloads.py::execute_payload`, the one entry for gameplay
events perturbing fields) · FieldEdit (`field_edit.py`; the `heat` policy) ·
Starting a fire = delivering heat (`apply_temperature_ignition`) · Entity
system (`entities/` schema + registry, one serializer, sim-side runtimes,
slot 9e) · Level data layer (`level_lib.py` write / `level_loader.py` read —
one writer ever) · Tick conductor (`Simulation.step()`: one call line + an
ordering comment) · Config (`config.toml` via `CFG`) · E2E driver
(`tools/e2e_drive.py`) · Field digest / GOLDEN_AGGREGATE (presence-gated
sections) · GameRenderer + WorldComposite (world drawing inside
`compose_world()`) · LightingPass (`renderer/lighting.py`) · main.py
`_parse_*` flags.

**New systems this run creates, with draft rules (written into CLAUDE.md at
implementation, pointing at the real code):**
- *Timed charge* — "A scripted or periodic detonation is a `timed_charge`
  entity row executing a `[payloads.*]` row through `execute_payload` on the
  sim clock (slot 9e); never a scheduler in `main.py`, a test-only hook, or
  a second payload path."
- *Level generators* — "A generated level is written through `level_lib`
  (the one writer); `tools/gen_explosion_studio.py` is the precedent that
  does it right; `gen_fire_studio.py`'s `np.savetxt` is the anti-pattern
  (#67 item 3)."
- *Wall-destruction render seam* (P4) — "A topology change reaches the
  renderer through ONE dirty-tile seam consumed by `compose_world()`, for
  every cause; never per-cause redraw code."
