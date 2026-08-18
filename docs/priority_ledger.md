# Priority ledger

**What we want to work on and complete, in order.** Coarser than a roadmap:
roadmaps (e.g. `roadmap_2026-07.md`) plan a window; this ledger holds the
standing stack so any session can orient in ten seconds. Update it whenever a
priority is decided, finished, or dropped — a stale ledger is worse than none.
Created 2026-07-17 from Erik's stated stack; Erik owns the ordering.

**Planning window 2026-07-30 →** `roadmap_2026-07-30_rl_push.md` (Erik-blessed):
the RL push — real-time-primary + player-insertable-at-any-seat vision; tracks
F (fire re-tune, ACTIVE, Erik) · 1 combat completeness (momentum/sprint next) ·
2 RL substrate (recorder-first; milestones M0–M4 can start in parallel
worktrees) · 3 training ladder (full-AI R1–R4 first, human-insertion H1–H2
after) · 4 content · 5 rendering (motivation-driven, never blocking). The stack
below remains the standing order.

---

## Live thread — the physics lid (storm → energy books → pressure)

**Energy-books arc — CLOSED 2026-08-17, Erik-blessed at the P-E5 HUMAN-TEST.**
The EOS no longer copies temperature, it moves energy: the semi-Lagrangian
T-copy that minted **+7,805 eth per 200 s bench run** is gone, transport is
one-way non-positive on every tick (**+3.72e16 → −5.33e14** on the hot-rail
scenario), the hot rail is closed (`t_max_phys_hits` **2130 → 0**, peak T
**15984 → 3702**), conduction's free cold-rail leg flipped sign (`t_min_gas`
**−0.1908 → 0.0000**), and traces left the physics books entirely. Shipped
dials `k_drag = 0.5` / `k_drag_heat_frac = 0.0014`. Full record:
`docs/energy_books_arc_close_2026-08-17.md`; canon folded into engine
chapters 04/05/06; the arc's design, critique and seven as-builts are in
`docs/archive/`. Origin: `docs/storm_audit_2026-08-14.md`.

**Sequencing note (design §9, load-bearing for track 2 of the RL push): this
arc landed BEFORE any recorder milestone that snapshots physics for
training.** That was the point — the recorder must capture a substrate whose
books close, or every trajectory in the replay buffer carries a mint. M0–M4
are unblocked on this axis now.

**Pressure arc — CLOSED 2026-08-18, Erik HUMAN-TESTED.** The "storming
atmosphere" was **not physics**: the pressure solve was running under-converged
at `mg_cycles = 2` and re-injecting its residual every tick. Shipped
**`mg_cycles = 8`** (now config-visible in `[physics.eos]`). On `playground`:
**P_max 103.239 → 1.405 atm**, negative `P_min` gone, `u_clamp_hits` 69,672 → 0,
`work_clamp_hits` 386,835 → 0, `n_sub` 8 → 1 — and **~18% faster per tick**,
because a converged solve collapses the substep count. Grid size was the
dominant variable (25× worse at fixed tile size from 14×27 → 70×99), which is
why every bench we owned was blind to it. Erik's verdict: *"fires dont blow up
anymore."* Full record `docs/pressure_arc_root_cause_2026-08-17.md`; canon
folded into engine chapter 04; suite **48 → 37 reds, zero newly red**.

Bonus from the close: the sanctioned golden had been stale for **three**
approved changes (P-T0, P-E5's `k_drag`, this) and was duplicated as a hardcoded
literal in 11 CUDA check scripts, so one re-baseline fixed only one test. Now
single-sourced from `tests/_xarch_perfield_digest.py`; the mis-scoped W6 canary
was split so its durable half (RNG dormancy, physics-independent) keeps its
"never a re-baseline" contract. **CUDA lockstep is now genuinely green** rather
than masked by a stale constant.

Queued next, in this order:

1. **★ MASS-BOOKS ARC — AUDIT FIRST** (opened 2026-08-18 by the pressure arc's
   HUMAN-TEST). Erik: *"grenades still can [blow up], especially after i broke
   a wall with a high pressure room."* **Mass is being created.** Total map N
   grows **2.15×** on `playground`, which is `boundary = space, ambient = None`
   — no reservoir, no legitimate source, and the only sink (venting to vacuum)
   can only *remove* mass. Locally one cell reaches ~710× ambient, doubling
   every tick for 12 ticks, while the *solved* pressure sits at 1.371 — the mass
   and pressure fields have decoupled. Grenade bulk-N deposits are real and by
   design but are worth a few cells, not thousands.
   **The mint is UNATTRIBUTED**: the donor-cell transport has a per-cell outflow
   limiter and is mass-exact by construction, so three plausible mechanisms have
   already been proposed and falsified by measurement (density-division
   amplifier; SL duplication; O₂ suffocation). Do NOT choose a fix before the
   instrument exists. **First patch = a per-pass MASS LEDGER** — who creates N,
   who removes it, asserted every tick — the exact shape that worked for energy
   (`test_no_transport_mint`). Carries a known pre-existing lockstep divergence
   on its own target scenario: `test_cuda_p64_kick_compression` PART 2
   (blast + venting) diverges CPU↔GPU at both C=2 and C=8, despite P-E4's
   as-built claiming it repaired. Seed doc:
   `docs/human_test_2026-08-18_mass_books.md`.
2. **T_abs compression work** (design §2.9, RULING R1) — a short designed
   patch with its own critique round and its own HUMAN-TEST: run the
   reversible work on absolute temperature, `T_new = (T + 290)·(1±w) − 290`,
   so compression stops *freezing* sub-ambient gas and ambient air finally
   heats under compression at all. Feel-adjacent (breach rarefaction becomes
   genuinely cold).
3. **Post-pressure retune pass** — **DO NOT START BEFORE THE MASS ARC LANDS.**
   Retuning against a substrate that mints mass would bake the mint into the
   dials — the same argument the energy-books arc made for landing before any
   recorder milestone. Note the fires are now *legitimately* weaker: total fire
   intensity roughly halved on playground (79 → 33) once the spurious wind
   stopped delivering oxygen, so the anchors moved for a real reason.
   One sweep over everything: the fire anchors (`peak time` fell out of its band
   at P-E1; `peak I`, plateau T and `fire death` were already MISSing), `k_drag`
   (0.5 is a *starting* value, not a tuned one — Erik's explicit ruling, and it
   may now be unnecessary since it was damping a storm that was largely solver
   artifact), and the arc's one declared red,
   `tests/test_p3_direct_e2e.py::test_directional_spray_cone_follows_facing`
   (damped air shortens a spray cone's throw — expected, left honestly red).

---

## The stack

### 1. Physics engine v1 — CLOSED 2026-07-21 (S8a complete: Path B + Path A)
- ~~**Boundary conditions** (space vs planetside, per-map)~~ **DONE
  2026-07-20** — merged to main (`110e142`), Erik feel-blessed ("atmosphere
  behaves, feel is nice"). Planetside AMBIENT ring symmetric to SPACE:
  shift-pin P=P_amb, per-substep reservoir clamp, u-damping band absorber
  (the σ pressure-sponge reflects — off), trace absorption, `boundary_flux`
  rail; N-primary dials (effective pin 65540 at Earth defaults). CPU==CUDA
  tol 0; space goldens untouched. Canon: engine/04 as-built section. Design
  archived: `archive/boundary_conditions_spec_2026-07-19.md` (v2.4) +
  `archive/bc_step_a_audit_2026-07-19.md`. `levels/planetside_demo` is the
  fixture. Open follow-up: sky render tint is a placeholder (final art pass
  later); u-damping width/k are per-level dials.
- **EOS residency patch (S8a)** ← **NEXT** — the last rung-B step: stop
  streaming all fields GPU→CPU each tick. Finishes the EOS arc. Spec ready:
  `cuda_s8a_residency_spec_2026-07-19.md` (post-EOS; carries the
  sensor-gather contract §5a — Arc B gated on it — and the structural
  dirty-set rider §5b; two-rung H2D plan). NOTE: BC landed first as planned,
  so the kernels S8a freezes now include the ambient branches (shift,
  reservoir clamp, u-damping) — the spec's launch-core extraction covers
  them. **PATH B DONE + MERGED 2026-07-20** (`1ae6f86`, --no-ff, auto-merge on
  green): leaf solvers (water substep loop + smoke 5-plane loop + decay) GPU-
  resident via `step_resident` + a GameMap CuPy mode; EOS kept on its host
  island and BRACKETED. tol-0 bit-identical (40-tick full-engine A/B, all synced
  fields incl heat/ripple); suite 997 passed. Payoff (resident vs per-call on the
  multiplied tax): **1.75× @128² · 3.20× @256² · 4.77× @384²** (scales with area —
  the big-map win); full-tick @256² 45.8→34.2 ms even with EOS bracketed. Flag
  default OFF (`set_residency`, `--resident`); CuPy never imported on the CPU path.
  §5b unit-stamp always-upload masks in `GameMap._RESIDENT_MASKS` (body-shielding
  preserved). **PATH A DONE + MERGED 2026-07-21** (`93a014c`, --no-ff, Fable,
  auto-merge on green per the kickoff): the EOS stage is FULLY device-resident —
  on-device `mg_build_levels` port (single-writer gather kernels + per-cell
  integer divides; host pre-stage keeps ALL the global reductions on the mirror,
  they consume tick-entry state), device mid-stage (div_u/Dalton/p*), shared
  vcycle + kick launch cores, zero mid-tick plane transfers (the "S8 endpoint").
  Design: `cuda_s8a_path_a_impl_2026-07-21.md` (v2, 3-lens adversarial critique
  survived). Gate: space + AMBIENT 40-tick A/Bs tol 0 (fields + telemetry +
  boundary_flux rail) + a device-vs-host BUILD-PARITY probe (poisoned hierarchy,
  3 scenarios incl. odd dims); suite 997 passed; NO re-baseline. Payoff: EOS
  stage 1.8×→2.2× vs the per-call bracket (grows with area); **full tick @256²:
  CPU 47.8 | per-call 53.6 | RESIDENT 27.1 ms** — the big-map win. ★ Found+fixed
  en route: `is_ambient` is NOT static (destroy_wall's joins-ambient twin) — now
  rides the per-tick EOS upload; this was also a latent Path-B bug (stale device
  trace sink after a ring breach — the space-only gate couldn't see it). Resident
  path skips the six digest_* checkpoints + dbg probes (documented gap; per-call
  path unchanged and still carries them). **S8a COMPLETE → physics engine v1 is
  CLOSED.** Next in the S8 line: S8b (CUDA graphs), then S8c (render CUDA-GL
  interop + recorder kernels + the `cast_fire_heat` device port — the fire-FPS
  fix). Earlier en-route fix (Path B): main's `BREACH_CUDA=ON` build broken since
  `472871d` (smoke_clamp arity) — fixed on main (`ae85906`).
  **S8c ITEM 1 (fire-FPS fix) DONE 2026-07-21** (`s8c-fire-heat-batch`,
  auto-merge on green per session pre-auth): new `bp.cuda_raycaster_cast_batch`
  concatenates `build_ray_list` over all burning-tile sources and marches them in
  ONE `raycaster_cast_directional` (one H2D/march/D2H) instead of one whole-plane
  round-trip PER source; `cast_fire_heat` issues a single batched cast on the CUDA
  path (CPU path unchanged). `heat` byte-identical (order-free saturating atomic
  adds; no re-baseline). Design + 3-lens critique
  (`docs/s8c_item1_fire_heat_batch_impl_2026-07-21.md`). Gate: cuda_s2 batch
  witness + cuda_s2b live heat A/B (65 src, tol 0) + s8a full-engine A/B + 997
  suite + payoff bench (600-fire firestorm 424 ms/~2.4 fps -> 1.5 ms, 277x, heat
  identical). ★ Items 2 (render CUDA-GL interop) + 3 (recorder kernels) reassessed
  as LOW-PAYOFF as literally framed — recon: renderer is raylib/pyray (cffi, no
  exposed GL-texture hook for cupy interop, no repo precedent) and the genuine
  render-only fields (`smoke_glow`/`ripple`) are HOST-computed (no device copy to
  skip); the recorder reads already-mirrored host data (the Q4 D2H is mandatory).
  **Items 2 & 3 DEFERRED + documented as accepted gaps (Erik, 2026-07-21) —
  `docs/s8c_items_2_3_deferred_2026-07-21.md`** (revisit triggers inside). S8c's
  one load-bearing win (item 1) is done; next real S8 item stays S8b (CUDA
  graphs, parked).
- Riders (chat-sized, slot when convenient):
  - ~~Wall-burst differential fix~~ **DONE 2026-07-18** — merged to main
    (true differential; only 1-deep membranes burst; Erik blessed).
  - **Dust-stirring shockwaves** — dusty-ground flag + wave_p threshold →
    smoke injection (notes 2026-07-17 Topic 1).
  - Post-EOS doc consolidation (roadmap §1.3 rider).
  - **`physics.py:104` blast-tuple wart — DECIDED 2026-07-19 (direction):**
    do NOT widen the tuple; replace it with a per-material
    **blast-pressure-threshold column in the material table** — damage only
    when local blast amplitude ≥ threshold (many small waves harmless, one
    big one bites; Erik's steel-resilience intent). Defaults reproduce
    today's behavior (excluded materials ≈ ∞ threshold → digest-safe);
    enables two glass types (brittle vs space-rated) as table rows.
    Implementation + tuning = chat-sized HUMAN-TEST rider AFTER residency.

### 2. Weapons, units, classes, a small enemy roster
- Weapons wave finale: W6 armory tuning session (Erik, human-gated — see
  TODO.md) → merge → wave close.
- Unit classes per `breach_unit_class_design.md`; enemy marines; a few
  critters (see `beastiary/beastiary.md`); zombies already work.

### 3. The vertical slice — "Counter-Strike, but the map fights back"
Two opposing teams with objectives. The twist is the physics sandbox:
- destructible map, pressure/fire/water fully in play — rooms can be
  flooded, atmosphere drained, walls blown through;
- an **animal pen** that can open (by plan or by damage) and release
  critters for further chaos;
- **zombies as a third faction**: hostile to both teams, and infection
  creates more zombies mid-match.
This is the setting the ML end-goal trains in: chaotic initial conditions,
genuinely different rounds, agents that learn to *handle* fire/flood/poison
rather than memorize an optimal line (`missions/mission_ideas.md` ML note).

### 4. The end goal (standing, shapes everything above)
Self-play NN training on the finished physics — train once, on final
physics. Big training runs wait for the S8 optimize-hard pass.

## Side tracks (not blocking the stack)
- **Procedural skeletal animation** — marines first (render-only), then the
  menagerie; post physics-v1. `procedural_animation_brainstorm.md`.
- **Entity system + editor v3** — design LOCKED 2026-07-18
  (`entity_system_design_2026-07-18.md` canon model +
  `level_editor_v3_design_2026-07-18.md` view; both erratad as-built).
  **Arc A (entity foundation) DONE 2026-07-19** — A1–A9 merged to main,
  Erik blessed (doors v0 human-tested, A7 re-baseline blessed — and found
  EMPTY of committed artifacts). **Arc B (logic layer) DONE 2026-07-22** —
  B1–B7 merged to main, Erik blessed at the airlock HUMAN-TEST: SignalBus +
  `[[wire]]` dotted format + split-gated slot-9e + node set
  (decider/gates/filter) + v1 sensor catalog (§5a accessor stubbed to the
  host mirror) + N-feed pump + the **bidirectional** automatic
  airlock_controller (chamber pressure sensor + deciders — Erik's Option 2) +
  cross-machine logic golden. Dormancy held (zero goldens re-baselined).
  Canon: `architecture/engine/16_entity_system.md` §8; arc design records in
  `docs/archive/` (incl. `arc_b_impl_2026-07-21.md`).
  Build order (Erik): ~~physics close-out~~ ✅ → ~~Arc B~~ ✅ →
  ~~Arc C~~ ✅ **DONE** (editor UX panes, tools, transaction-log undo,
  wiring, play-from-editor, icons — C0–C9 merged 2026-07-22, merge
  `54cd6cd`; autonomous-patch-workflow, ONE end-of-arc HUMAN-TEST). Build
  docs archived under `docs/archive/` (`arc_c_kickoff_2026-07-22.md`,
  `arc_c_impl_2026-07-22.md`, `arc_c_c2_undo_design_2026-07-22.md`); canon
  fold in `architecture/engine/16_entity_system.md`. AI tilesets (levels-w1
  P6) still parked. Deferred Arc B rider: the resident sensor-gather kernel
  (§5a interface frozen; was gated on S8c, now buildable). Arc-C riders —
  both ✅:
  - baker `[art]`/`[bake]` writeback → `level_lib` atomic client (A2 gap
    closed, C9) + the MAT_DOOR_CLOSED-outside-span validator.
  - **Level-folder cleanup (Erik 2026-07-24 — revised the 07-22 "retire
    from migration" ruling):** `playground` + `planetside_demo` migrated
    +baked into the entity system (playground: painted doors → `door`
    entities, greybox bake; now the default level); `unhcr_vessel` KEPT in
    `levels/` as the legacy physics-test fixture (migrating it would break
    its ~27 fixture tests — not worth it); `unhcr_vessel_2` retired →
    `prototypes/`; `bake_demo` stays legacy until its art rebakes. A NEW
    flagship level authored in the editor ("describe one level fully, build
    everything it needs") is Erik's next direction (assets/prefab-library +
    the now-unblocked interactive entities — buttons/terminals/turrets, the
    control-scheme decision having landed).
  - `physics.py:104` blast-tuple wart → decide at physics close-out
    (listed under stack #1).
- **Sound-ML** — parked (`sound_ml_research_brief.md`), junior to the EOS arc.
- Beauty tracks: black-body emitter, smoke visuals, scorch/blood painting.

## Next chat-sized sessions (each its own chat)
1. Boundary-conditions spec (against landed rung-B; decide mode set + level
   format field + kernel touch points).
2. ~~Wall-burst differential fix~~ (DONE 2026-07-18) — the
   `burst_threshold` re-tune dial stays open.
3. EOS residency patch (if not already in flight).
4. W6 armory tuning (Erik) → weapons wave close ritual.
5. Dust-stirring shockwaves spec/impl.
6. Animation P0 — marine prototype, 3D-model-vs-part-sprites by eye.
