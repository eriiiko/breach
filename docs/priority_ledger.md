# Priority ledger

**What we want to work on and complete, in order.** Coarser than a roadmap:
roadmaps (e.g. `roadmap_2026-07.md`) plan a window; this ledger holds the
standing stack so any session can orient in ten seconds. Update it whenever a
priority is decided, finished, or dropped — a stale ledger is worse than none.
Created 2026-07-17 from Erik's stated stack; Erik owns the ordering.

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
  Build order (Erik): ~~physics close-out~~ ✅ → ~~Arc B~~ ✅ → **next =
  Arc C** (editor UX panes, tools, undo, wiring, play-from-editor, icons) —
  **KICKED OFF 2026-07-22**: `docs/arc_c_kickoff_2026-07-22.md` (Opus runs
  it, autonomous-patch-workflow; Erik's rulings inside: ONE end-of-arc
  HUMAN-TEST, legacy migrations dropped, Lenovo). AI tilesets (levels-w1
  P6) still parked behind it. Deferred Arc B rider: the resident
  sensor-gather kernel (§5a interface frozen; was gated on S8c, now
  buildable). Arc riders on the books:
  - baker `[art]`/`[bake]` writeback → `level_lib` client (A2 accepted
    gap; folds in at Arc C, in the kickoff).
  - `bake_demo` stays legacy-form until its committed baked art rebakes
    (migrating now would desync tilemap ↔ baked PNGs). Other legacy
    showcase levels (`unhcr_vessel`, `unhcr_vessel_2`, `playground`):
    **retired from migration (Erik 2026-07-22)** — a NEW level authored
    in the Arc C editor replaces them (the acceptance drive); they stay
    on disk untouched.
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
