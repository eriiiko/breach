# Residency + batching design — critique, lens 4: game-development continuity (2026-11-07)

Critic: independent, one lens. Read: design §0–§13 (not §14, not the other
critiques), the project CLAUDE.md, `unit.py`, `weapons.py` + `config.toml`'s
weapon / loadout rows, `combat.py`, `status.py`, `damage.py`, `exchange.py`,
`door_system.py`, `entities/` (schema, serialize, charge, door), `match_replay.py`,
`tests/field_digest.py` + spec, `tests/field_ab_harness.py`, `tests/cuda_harness.py`
and the `test_cuda_*` wrappers, `tests/test_match_replay.py`, the master CLAUDE.md
Git bullet and the `git-workflow` skill, `docs/rl_ready_scope_2026-10-29.md` §2–§3,
`docs/residency_where_we_stand_2026-11-07.md` §4, main's git log since 2026-09-07,
and the build folders actually present in the 39 worktree folders.

**VERDICT: SOUND WITH FIXES** — the architecture survives this lens, but as
written the twin-debt rule cannot fire in the place gameplay code is written, and
the design never says which device code the PLAYED game depends on; both blockers
must be fixed in the design before P1, or a training at a new snapshot can run a
device game that silently differs from the played game while main's GPU play modes
rot.

Measured context used below: on main since 2026-09-07 there are 76 non-merge commits
touching `src/simulation`, 50 touching `cpp/src/*.cpp|*.h` and 32 touching
`cpp/src/*.cu|*.cuh` (`git log main --since=2026-09-07 --no-merges`), i.e. today the
CUDA twins are kept level per patch (e.g. `fd41b9ed` "#31 Patch B: the blast's source
on the GPU, per-call and resident"). Of the 39 folders under `.claude/worktrees/`, 6
hold a CPU build and exactly 1 (`140-wind-q-threshold`) holds a `cpp/build_cuda`.

---

## BLOCKERS

### B1. §6 / §10 — twin debt cannot be detected where gameplay is written, so the "xfail in the same patch" rule is fiction and drift is silent

**Evidence.**
- Every CUDA gate skips without a CUDA build in the checkout running pytest:
  `tests/cuda_harness.py:27` (`CUDA_BUILD_DIR = ROOT / "cpp" / "build_cuda"`, ROOT =
  the checkout) and `:74` `cuda_available()` = "is the .pyd present"; 29 of the 30
  `tests/test_cuda_*.py` carry `pytestmark = pytest.mark.skipif(not
  cuda_harness.cuda_available(...))`. The master rule puts all code in a fresh
  worktree; 1 worktree folder in 39 has a CUDA build. So in an ordinary gameplay
  patch, `pytest tests -q` is green with every device gate SKIPPED. Nobody is told
  to mark anything.
- The replay corpus (§5.3: "ship_demo seeds 1–8, the freighters") does not exercise
  the features that will change next. The only loadout is `[loadouts.rifleman]`
  (`config.toml:4382`) with `k5_carbine`, whose `mag_size = 0` (`config.toml:3077`) —
  "untracked", and `combat.mag_gate` returns before touching any state
  (`combat.py:695`). Magazines, reload stalls, sprays, pellets (`jackhammer_8`,
  `shots_per_trigger = 8`, `config.toml:3302–3311`), hitscan, melee rows: none is
  fired by the corpus. A device twin of any of them can be wrong, or absent, and the
  replay-lockstep gate stays GREEN. `rl_ready_scope` §2 already plans "one
  flamethrower per team as a roster variant" — the first training variant would hit
  exactly an unexercised twin.
- Mag state is deliberately outside the unit digest (`unit.py:258–270` comment;
  `field_ab_harness.py:69–73` `SYNCED_UNIT_FIELDS`), so even an exercised mag
  divergence shows only when it changes a hit.
- If the corpus is a STORED artefact (§9 P3: "replay corpus re-pinned"), every
  behavioural reference change stales its checkpoints on the CPU side too, so the
  gameplay author re-pins it each week and the device side's failure is
  indistinguishable from routine re-pinning.
- The precedent is already in the repo: #71, the CUDA combustion twin left behind by
  arc #54, discovered late and "handled" by switching the backend off —
  `docs/issue_triage_pre_weapons_2026-09-25.md:63–64` ("does not touch the shipped
  game, because `--cuda` never enables that backend"), still open in
  `docs/residency_where_we_stand_2026-11-07.md:105`.

**Fix (all four; each is a CPU-side or door-side mechanism, no discipline needed).**
1. **A computed debt ledger instead of hand-placed xfails.** Each device twin
   declares, in one manifest (`device/twin_manifest.toml`, provisional), the
   reference paths it mirrors (files, or `module:function`), and the commit at which
   it was last brought level (`level_at = <sha>`). `tools/twin_debt.py` lists every
   twin whose reference paths changed in `git diff <level_at>..HEAD`. That list IS
   the catch-up issue's body; nothing has to be remembered by the gameplay author.
   Reuse the precedent the repo already deleted once: `EOS_P6_PENDING_KERNELS`
   (`tests/cuda_harness.py:60–71`, a pending SET with one key per surface, emptied as
   kernels were re-proved) — a registry, not marks scattered over test files.
2. **Refusing doors on the training side.** `build_level_bank` refuses a template
   containing an entity class, and `[rl]` / the bank refuse a loadout naming a
   weapon row, a status kind, a payload row or a coupling row, that has no device
   twin at a level commit (a `DEVICE_SUPPORTED` table generated from the manifest),
   naming the item and the catch-up issue. `require_snapshot(tag,
   backend="device")` refuses while the ledger is non-empty for anything the run's
   tables or bank use; `backend="cpu_farm"` never needs the ledger, so RL is never
   blocked by debt (state this in §6: the farm trains at any tag).
3. **Live lockstep on the training's own distribution, not a stored corpus.** The
   twin oracle records on the reference at HEAD and replays on the device in the
   same run (today's pattern: `tests/test_match_replay.py:85–92` records ship_demo
   live in the test). A pre-flight in `require_snapshot(backend="device")` runs K
   episodes of the run's actual bank, loadouts and `[rl]` tables with the random-
   action driver on both backends and compares per-tick digests. Nothing stored, so
   nothing goes stale.
4. **Coverage counters.** The lockstep reports per-feature exercise counts (shots
   per weapon row, reloads, statuses applied per kind, entity classes ticked,
   coupling rows fired, payload rows executed) and refuses when a feature present in
   the run's config has zero exercise. This is what turns "GREEN because never
   exercised" into a failure.

### B2. §4.2 / §6 / §10 — the design never partitions device code into "the played game runs it" vs "only training runs it", so the catch-up rule either breaks `--cuda` / `--resident` or contradicts "the played game never waits for a kernel"

**Evidence.**
- §10 bullet 1: "A gameplay change is made on the REFERENCE path first (Python /
  C++) … the played game never waits for a kernel." §6: "a reference-path change
  that breaks a device twin's gate is marked `xfail`". Neither distinguishes the
  physics twins — which the played game's `--cuda` and `--resident` modes run today
  and which §3 calls "still the played game's fastest single-instance path" — from
  the gameplay kernels (P6–P8) only training runs.
- §4.2 makes every resident core `(N, h, w)` (the §A habit): after P5 the played
  `--resident` game at N = 1 runs the batched kernels; after P4 it runs the device
  edit queue and the stamp_units kernel. So an EOS change, a FieldEdit policy change
  or a stamp change on main that follows §10 literally (reference first, twin at
  catch-up) breaks main's GPU play modes until catch-up — the #71 outcome
  generalised. Today's practice (32 CUDA commits beside 50 C++ commits in two months;
  CLAUDE.md's radiation-sweep row: "an arithmetic change lands in the reference, then
  `radiation_sweep.cpp`, then `cuda_radiation_sweep.cu`") is the opposite rule, and
  the design silently replaces it.
- The design does not say whether the played `--resident` game ever uses the device
  unit store (P6). If it does, every gameplay change breaks `--resident`.

**Fix.** Add to §6 / §10 an explicit two-class rule and put it in §13 (b):
- **Class P — played-game device code:** the physics cores, the trace tail, the
  sweep, combustion / temperature / fire, the device edit queue, stamp_units, and
  anything `main.py --cuda/--resident` runs. Kept level PER PATCH on main, exactly
  as today (the existing CLAUDE.md twin rows stand); never xfail; a physics patch
  without its twin does not merge. State its cost honestly: today's ratio of CUDA to
  C++ commits continues for every physics change.
- **Class T — training-only device code:** the unit store and its slots, entities on
  the device, combat, the horde field kernel, reset, obs / reward / done. These
  carry debt under B1's ledger.
- State that the played `--resident` game keeps Python gameplay over device physics
  (no device unit store), so Class T never touches play. If Erik wants otherwise,
  that is a ruling for §12, not an implementation detail.

---

## MAJORS

### M1. §6 / §9 — P3's played-game changes and the snapshot ride the arc branch while main keeps moving

**Evidence.** §6: "a git tag `train/<name>` on main (on the arc branch while the arc
runs)"; §9: P3 (Philox at the draw sites, Q16 `Unit.x/y`, zombies on the field, the
unit schema / digest) is merged into the arc's line. The files P3 rewrites are the
most edited on main: `simulation.py` 19 commits, `payloads.py` 11, `combat.py` 8,
`exchange.py` 8, `ai_zombie.py` 4 since 2026-09-07. Precedent: fire-12 ran for months
and merged at arc close. A snapshot on the arc branch is a game Erik has never had on
main, and the arc's final merge starts with every gameplay change main took in the
meantime as twin debt.

**Fix.** P1, P2 and each P3 commit merge to MAIN as they pass (P3's two HUMAN-TESTs
on main's terms); the arc branch takes main in (`git merge main`) at every patch
start; the first `train/*` tag is on main. Only Class-T device work (P6–P9) may ride
the arc branch, because nothing on main depends on it.

### M2. §6 — `require_snapshot` has provenance holes that will be hit in practice

**Evidence and fixes, one per hole.**
- **Lightweight tags.** The repo mixes both kinds (`octopus-v1`, `archive/*` are
  lightweight; `bedrock`, `beatrice-b1g` annotated). `git describe` without `--tags`
  ignores lightweight tags — this checkout describes as
  `beatrice-b1g-551-geb30c9e3`. Fix: `train/*` are annotated (`git tag -a`), placed
  by Erik (the skill: "release — a tag on the integration line, placed by Erik"),
  pushed, never moved; compare commit SHAs (`git rev-parse train/<n>^{commit}` vs
  HEAD), not describe strings.
- **Untracked files change the game invisibly.** CLAUDE.md's level-data row: the
  per-tile seeds `air_init.npy` / `temp_init.npy` are "found by presence, dormant
  when absent". `git describe --dirty` ignores untracked files, and the bank records
  only "the build id of the generator" (§4.6). Fix: "clean" = `git status --porcelain
  --untracked-files=all` empty under `src/`, `levels/`, `config.toml`, `cpp/`; the
  bank records sha256 of every input file it read plus the tag's commit SHA, and
  `require_snapshot` verifies them.
- **Which checkout's module is imported.** Worktrees routinely import main's CUDA
  build (`tools/bench_resident_scaling.py:5` and `tools/bench_bot_matches.py:77`
  pass `--build C:/Users/steen/projects/breach/cpp/build_cuda`); main's
  `cpp/build_cuda` right now holds a cp311 .pyd built 06:28 and a cp312 .pyd built
  09:10, while main's HEAD is from 13:07 — one folder, mixed builds, neither at HEAD.
  Fix: `require_snapshot` also asserts `bp.__file__` lies under the snapshot
  checkout's own `cpp/build*` and `simulation.__file__` under its `src/`.
- **Build id "at configure".** Stale after an incremental `cmake --build` (agents run
  it without the .bat). Fix: generate the id at BUILD time into ONE generated
  translation unit (written only when it changes), so a commit does not force a full
  CUDA rebuild and an incremental build cannot carry the previous configure's id.
- **The environment is not in the snapshot.** CuPy / torch / numpy / CUDA runtime
  are the `data` env's, moved by the `ml-stack` rollout (master CLAUDE.md, "Temporary:
  CuPy + ML stack rollout"); the farm's legacy draws are numpy `Generator` streams
  (`simulation.py:282`), whose distribution methods numpy does not promise stable
  across versions. Fix: the run manifest records those versions and the GPU driver;
  a resumed run refuses on a change unless forced. On Berzelius, the container image
  is built from the TAG's `containers/breach.def` and named after the tag.
- **Outputs.** Checkpoints, logs and recordings written inside the snapshot tree make
  it dirty (or, if untracked files are ignored, hide real changes). Fix: §6 says all
  run outputs go outside the checkout.

### M3. §4.5 / §5.3 / §11 — "a training episode is replayable" is claimed but the device path has no tap and format v1 cannot name a bank template

**Evidence.** Format v1's header names the level by folder name only
(`match_replay.py:164` `level_id` = `Path(level.path).name`) and `build` is
informational (`:171–181`, "a replay never refuses on it"). A device episode's level
is a bank template index (§4.6); there is no folder. The tap lives in the
`Simulation` entry points (`match_replay.py` docstring, "The tap"); on the device
backend actions go straight into the `(N, U, A)` buffer, so nothing records them. The
action buffer is int32 but `INTENT_KINDS["throw_grenade_intent"]` carries
`fuse_seconds` as a float (`match_replay.py:86`, `"fhex"`), so "the intents are
INTENT_KINDS minus drive_zombies" (§4.5) is not a lossless mapping.

**Fix.** A training-episode record (format v2, or a sibling format): snapshot tag +
commit SHA, bank sha256 + template index + the template's source (folder name +
content hash, or its `[levelgen]` LevelSpec record), episode seed / Philox key,
`decimation`, and the env's action rows per step; written for selected envs (every
K-th episode, every anomaly), converted to a replayable `Recording` on the reference
in the snapshot checkout. Decide the fuse in the action schema (an integer tick count
or a fixed fuse) and amend `INTENT_KINDS` to match.

### M4. §4.5 / §10 / §4.8 — the cost statements are not honest about behaviour, and the device has no story for gameplay TABLES

**Evidence.**
- §4.5: "A new verb is a column in the intent schema and a row in `INTENT_KINDS`,
  never a kernel rewrite"; §10: "A new per-unit field is a schema row; a new verb is
  an intent column … None is a kernel edit." True for plumbing (digest, obs,
  serialization), false for meaning: a reload verb does nothing until the shooting
  kernel consumes it.
- §10: "A rare event (a new payload …) is a host handler writing through the edit
  queue and needs no twin." But §4.8's window is "the open mask and the two air
  planes over its reach"; a payload that reads another plane, writes a field whose
  policy the device queue has not ported, or touches UNITS (blast damage
  `exchange.apply_blast_damage`, a stun grenade applying a status) needs device-side
  work — the host must read and write the device unit store, which §4.8 does not
  describe.
- §4.2 names only "material and gas tables" as batch config. The kernels will also
  read weapons, ammo, loadouts, env profiles, species, the status registry
  (`status.py:131`, a Python tuple), payload rows, `[exchange]`, `[zombie]`,
  `[movement]`. With no generic upload, a new weapon COLUMN (a new pellet pattern
  dial) is a hand edit to a device struct.

**Fix.** Replace the two sentences with a cost table per change type ("plumbing:
free; behaviour: a Class-T kernel twin at catch-up; a new table column: free if the
table is schema-uploaded"); add a **device table upload** system to §13 (b): every
gameplay table the kernels read is uploaded from the same validated Python table
object through a column schema, so a new column is a schema row; specify the rare
event's unit read/write path.

### M5. §10 — no rule keeps future reference gameplay twinnable, so every catch-up re-derives IEEE float chains and shared-stream draws

**Evidence.** The match path's per-tick gameplay is float64 today: the bullet march
(`combat.py:1058–1095`: `math.radians(spread_deg)`, `base_angle +
float(rng.uniform(-cone, cone))`, "the rx/ry accumulation is plain float"), the
damage chain before the Q2 snap (`damage.py:11–22, 82–90`), the cover / crit rolls
(`attack_resolver.py:80`, `:138`: `float(rng.uniform(0.0, 1.0)) < p`), `current_hp`
(`unit.py:216`). After P3, nothing stops Erik's next combat change from adding a new
`sim.rng` draw (the habit every draw site shows) or a new float chain. Variable-length
state has no rule either: `POISONED` is `STACK_STACK` (`status.py:141`) and a stacking
kind "always appends" (`status.py:244`), but §4.4 stores statuses as "a fixed-width
block"; the first poison coupling row overflows it.

**Fix.** §10 gains three rules, each with a CPU-side gate in the ordinary suite:
(1) a draw on the match path is a Philox draw with a registered salt — `sim.rng` on a
match path raises at runtime (a refusing stand-in installed by `spawn_match`); (2) new
per-tick match-path arithmetic is integer, with a float ratchet over the match-path
modules (the `test_no_float_in_sim_tu.py` pattern, counts only go down); (3) every
per-unit collection has a schema capacity and an overflow rule (refused at the
status / effect door, counted), and `STACK_STACK` gets a cap row. P3 also audits the
existing float chains above: each is lifted to integer (a further played-game
re-baseline, listed in §7) or declared "device replicates IEEE double, op order
pinned, `--fmad=false`", with its gate.

### M6. §4.4 / §5.7 / §13 — the unit schema is a parallel mechanism beside the entity registry, nothing makes a new `Unit` attribute land in it, and every new column churns every golden

**Evidence.**
- Today a per-unit field enters the digest only by hand: `SYNCED_UNIT_FIELDS` + the
  matching key in `_unit_record` (`tests/field_ab_harness.py:69–73, 179–217`),
  statuses through `serialize_statuses`. Entities have their own vocabulary (field
  kinds `int/q16/bool/enum/entity_ref`, `runtime_digest_rows`, ENTITY_SECT_V1,
  `field_digest_spec.toml` `[entity_section]`), and the schema module itself says
  units "are not entities until the stack-2 convergence (design §3e)"
  (`entities/schema.py:30`). The design adds `unit_schema.py` as a third list without
  saying whether it reuses the entity kinds and serializer.
- `Unit` is a plain class with ~50 attributes set in `__init__` plus ones added
  elsewhere (`use_intent` sets `u.pending_use`, `simulation.py:989`). Erik's next
  attribute (`self.ammo_reserve = 90`) lands in `Unit.__init__` only; no test fails;
  the device has no column, the digest does not hash it, obs cannot see it.
- §5.7: "The unit store's synced columns enter the entity section (a version
  bump …)". The unit hash is not in the entity section: it is `UNIT_DIGEST_KEY`,
  folded separately by `tick_digest`; the entity section carries its own preamble,
  not `DIGEST_SPEC_VERSION` (`tests/field_digest.py`, `tick_digest`).
- Adding a hashed column changes the record of every unit in every snapshot, so
  every golden and every recorded checkpoint moves per new field. Today's authors
  avoid that by keeping new state OUT of the digest (`combat.py:688–693`, the mag
  state) — the opposite of what training needs.

**Fix.** Say the unit schema USES the entity field-kind vocabulary and the one
serializer's encoding (`name|int64`) under a new section preamble (`UNIT_SECT_V2`),
and make columns **absence-transparent**, the `DIGEST_SECTIONS` / A4 precedent
(`tests/field_digest.py:148`): a column at its declared default hashes nothing, so a
new column moves only the goldens that exercise it. Enforce membership on the CPU
side: `Unit` gets `__slots__` = schema columns + a declared `REFERENCE_ONLY` list
(WEGO, render, AI scratch), so an undeclared attribute raises at the line that sets
it. Correct §5.7.

### M7. §7 — the Q16-position change is labelled but its blast radius on the played game is not, and it is avoidable

**Evidence.** §7: "`Unit.x/y` integers". Readers that treat `u.x` as float tile units
outside the sim: `renderer/game_renderer.py:1417, 1512` (`int(u.x) + fp // 2`),
`renderer/unit_model_renderer.py:738, 791, 803, 869`, `renderer/follow_camera.py:89`,
`ui/model.py:300, 391, 409`, `ui/draw.py:235`, `src/control_bots.py:365`
(`quantize_scalar(u.x)`), `tools/gen_ship.py`, `tools/wind_load_probe.py`, plus every
open branch (`75-g4-explosions`, the swarm and octopus branches) that reads it.
Changing the type turns each `int(u.x)` into a raw Q16 count — a silent wrong tile,
not an error.

**Fix.** Store `x_q` / `y_q` (int, synced, the schema columns) and keep `x` / `y` as
read-only float properties (`dequantize`), exactly dyadic, so no reader changes and
the HUMAN-TEST measures only the movement law. State the cost of NOT doing each §7
change in one line (positions: the device would replicate float64 slide math op for
op; Philox: no per-env stream; field: no device pathing).

### M8. §6 — the training checkout has no place in the git workflow, and a hot-fix to a running snapshot has no defined route

**Evidence.** The `git-workflow` skill knows three roles (integration line, release
tag "placed by Erik", implementation branch off the integration line) and a worktree
home `.claude/worktrees/<branch>` (§3); its §6 says "Merged means gone — worktree,
local branch, remote branch" and §7's audit removes merged worktrees. A training
worktree is a detached HEAD at a tag whose commit is an ancestor of main: by the
skill's own test it is "merged" and an audit agent removes it mid-run. A bug found
during a device-backend training cannot always be fixed by moving to main's HEAD,
because main may carry Class-T debt; the alternative, a branch off the tag, is not an
implementation branch in the skill's sense.

**Fix.** Add a row to the skill: **training checkout** — `git worktree add --detach
../breach-train-<name> train/<name>` (outside `.claude/worktrees`), its own builds,
outputs outside it, never removed by the audit while a run uses it (a
`TRAINING-IN-USE` marker file the audit honours). Hot-fix route: fix on main as a
normal patch; if the run cannot move to HEAD, branch `<issue>-train-<name>-fix` from
the tag, cherry-pick the main fix, tag `train/<name>.1` (annotated, by Erik), and the
fix is already on main, so nothing has to flow back.

---

## MINORS

1. §10 bullet 3 says the "reference test is `xfail`"; §6 says the device twin's gate. A reference xfail would hide a played-game regression — say "the device gate".
2. §4.7's draw-site list names charges, heat sources and timed charges, which never draw (`timed_charge_system.py:12`, "executor never draws from `rng`"), and omits the cover-exposure and crit rolls (`attack_resolver.py:80`, `:138`).
3. `test_twin_debt` "asserts every such mark names an open issue": checking "open" needs the GitHub API inside `pytest tests -q` (offline machines, Berzelius); check the issue number's format in pytest and its state in `tools/twin_debt.py`.
4. §4.9 / §7: the field on the reference contradicts two CLAUDE.md rows that must be amended in the same patch — A* "The only pathfinder" and `zombies` "the shipped `ai_zombie` law WRAPPED, never rewritten … pinned bit for bit in `tests/test_zombie_control.py`".
5. The reference field's CPU cost is unstated; a Python BFS over 17 600 cells every K ticks against today's 0.15 ms zombie slot (§2.1) could slow the played game — say C++ or vectorised.
6. Dropping `ap` / `n_orders` (in `SYNCED_UNIT_FIELDS`) from a schema-driven digest removes WEGO's order-consumption coverage — one-line finding, WEGO is legacy.
7. `current_mag = None` means "full, bound lazily" (`unit.py:258–269`); the schema needs its integer sentinel, written down.
8. Door open / close is `unseal_tiles` / `seal_tiles` (`door_system.py:253–259`), a topology change §4.3 puts on the host, while doors are device rows (P6b); an airlock cycling makes the "rare" path frequent. Say which it is.
9. Philox in Python at the FieldEdit noise site (one draw per noisy tile at flush, `field_edit.py:673`) — measure the played game's blast-tick cost after P3.
10. Every snapshot costs a full CPU + CUDA build (and a Linux container build on Berzelius); say so, and say "never copy a .pyd between checkouts".
11. §13 (a) lists "entity system + serializer" as used but (b) 3 introduces the unit schema without saying it extends them (see M6).

---

## Walk-throughs

### (a) Magazines next month: a manual reload verb + a "reloading" status (mag state already exists, W3)

| step | where it lands | what breaks on the device until catch-up | what the rule says / should say |
|---|---|---|---|
| `current_mag`, `reload_done_tick` | already on `Unit` (`unit.py:269–270`), outside the digest | nothing yet: §4.4 lists them as columns; the `None` sentinel needs an encoding (minor 7) | schema rows, absence-transparent (M6) |
| `ammo_reserve` (new) | `Unit.__init__` | the device has no column; no test fails | §10 says "schema row", nothing enforces it → `__slots__` (M6) |
| `reload_intent(unit)` | new `Simulation` entry point, the tap, `INTENT_KINDS` row | the intent column exists but no kernel consumes it: on the device "reload" does nothing, in the played game it reloads | §4.5 says "never a kernel rewrite" — false (M4); a Class-T twin, ledgered (B1) |
| RELOADING status | a `STATUS_REGISTRY` row (`can_act = False`) | free IF the device status kernel reads flags from an uploaded registry; a new flag column is a kernel edit | needs the table-upload system (M4) |
| bots / corpus | `control_bots` reloads only if taught; the corpus loadout is K5 (`mag_size 0`) | the replay-lockstep gate stays GREEN | coverage counters + pre-flight on the run's loadouts (B1) |
| played game | reference path | works | — |

### (b) A shotgun with a new behaviour

| step | where it lands | what breaks | what the rule says / should say |
|---|---|---|---|
| a row only (`jackhammer_8` exists: 8 pellets, mag 8) | `config.toml` | nothing, if the device `fire_burst` twin loops `shots_per_trigger` with keyed draws and the weapon table is uploaded | a table row (M4's upload) |
| a new column (choke pattern) | weapon row + `weapons.py` door | the device weapon struct, by hand, unless schema-uploaded | M4 |
| `combat.py` grows a branch (pellet falloff) | Python, float64 like the march (`combat.py:1058–1095`) | the device twin must replicate the float chain op for op, or the branch is integer | M5's integer rule + ratchet |
| its pellet draws | `rng.uniform` habit | a shared-stream draw the device cannot key | M5's refusing `sim.rng` on the match path |
| training with it | a loadout naming it | silent if the twin is missing | the bank / `[rl]` refusing door (B1.2) |

### (c) A pressure door with a button (or a new entity class)

| step | where it lands | what breaks | what the rule says / should say |
|---|---|---|---|
| wiring existing classes (`door`, `button`, `pressure` sensor, a decider) | level data | nothing, if the device signal bus is generic (nodes as opcode rows, wires as edge lists per template) | P6b must say "generic bus"; a new level is data |
| a new CLASS (auto-closing on Δp) | an `entities/` class + runtime system + one slot line (today's cost) | on the device: a SoA layout, a kernel, the bank's entity rows, its `runtime_digest_rows` twin (variable length: door `hp_i` per tile, `door.py:80–88`) — 2–4x today's cost, unstated | B1.2: the bank refuses a template using a class without a twin; state the per-class cost (M4) |
| its open / close | `seal_tiles` / `unseal_tiles` (`door_system.py:253–259`) | lands on the host rare-event path by §4.3; frequency unbounded | minor 8 |
| played game | reference | works | — |

### (d) A physics change in the shared C++ (inside the EOS), or a new coupling row

| step | where it lands | what breaks | what the rule says / should say |
|---|---|---|---|
| EOS change | `eos_solver.cpp` + its CUDA twins (`cuda_eos_step.cu`, the resident launch — after P5 the batched launch IS the resident one) | main's `--cuda` / `--resident` if the twin waits for catch-up (the #71 outcome) | Class P: level per patch, as today (B2) |
| goldens | behavioural change | GOLDEN_AGGREGATE + every stored recording | live lockstep, no stored corpus (B1.3) |
| running training | its own checkout + build at the tag | nothing — the tag protects it, IF it imports its own .pyd (M2) | `require_snapshot` path check (M2) |
| a new coupling row (poison → POISONED) | `exchange.py`: a Python response function, not data | Class-T kernel twin; `STACK_STACK` overflows the fixed status block | ledger (B1); capacity rule (M5) |

---

## QUESTIONS (not grounded enough to be findings)

- Does Erik intend the played `--resident` game ever to run the device unit store? B2 assumes no; if yes, every gameplay change breaks `--resident`.
- Is the §5.3 corpus a stored artefact? §9 P3's "re-pinned" says yes; §5.3 does not say.
- Can Berzelius clone `eriiiko/breach` and fetch tags (credentials, network policy)? The tag-and-build route depends on it.
- How often do interior airlocks cycle in a training ship? That decides whether door topology fits the host rare-event path.
