# Swarm units — patch plan (arc #63, branch `63-swarm-units`)

> Executes `docs/architecture/engine/17_swarm_units.md` (BLESSED 2026-09-10)
> under the autonomous-patch-workflow. Plan agreed with Erik before execution;
> re-planned at patch boundaries. Oracle-gated patches auto-merge on green
> (standing grant); HUMAN-TEST patches wait for Erik. Worktree
> `C:/Users/steen/projects/breach-swarm`, one writer at a time: each patch is
> its own branch `63-pN-<slug>` cut from `63-swarm-units` in this worktree and
> merged back `--no-ff`. Critics one at a time, verdicts on disk. Tests assert
> properties, never snapshots. Model tiers name the family (newest of each).

**Re-plan 2026-09-24 (Erik + Claude, before P1).** Sequencing principles:
(1) one system per patch, each with its own oracle; (2) sim before looks —
the checkable patches land first, so the look-check judges only the look;
(3) implementation docs are written just-in-time, against code that exists,
immediately before their patch (chapter 17 already settles the
architecture). Changes vs the 2026-09-10 plan: the `swarm_brood` entity moves
up from the playground patch so larvae can be placed and play-tested early,
with a diagnostic dot view; real bodies are no longer "minimal" (Erik lifted
the graphics hold for the larvae only — the decal layer and other graphics
stay parked); the render patch's premise is corrected — **breach has no
instanced draw path today** (every prop and marine is one `draw_model_ex`),
so batch drawing is a new, reusable system and gets a research + design pass.

| # | Patch | Mode | Tier / effort | Gate (oracle) | HUMAN-TEST |
|---|---|---|---|---|---|
| P1 | **Adopt philox32** — vendor `cpp/src/philox32.h` + `src/simulation/philox32.py` (pinned to philox32 `6c33a3d`, hash in header); KAT + cross-language test in `tests/`; chapter-14 door-4 amendment + CLAUDE.md canonical row "Deterministic RNG"; ingress lint accepts it | subagent | Sonnet / medium | KAT + Python-vs-C++ identity; full suite no new reds; goldens untouched (no sim call yet) | no |
| P2 | **Swarm store scaffold** — `gmap.swarm_*` SoA `(N, max_units)` arrays (§2 roster, `max_units` config default 8192), residency allocation, `SWARM_SECT_V1` presence-gated digest section + recorder + save hooks, spawn/reclaim (`valid`, `next_unit_id`, `high_water`), `src/simulation/swarm.py` facade Simulation drives, species table `[swarm.larva]` schema-in-code/numbers-in-TOML, CFG hot-reload; NO behavior kernel | subagent, **design-gated** (impl doc `docs/swarm_P2_impl.md` + one critic first — wire/digest-touching) | Opus / high for the impl doc; Sonnet for the build | every existing golden byte-identical (presence gate); A/B harness extended; property tests: spawn fills sequentially, reclaim, high_water monotone, section absent ⇒ digest identical | no |
| P3 | **Larva behaviour kernel + CPU twin** — `cpp/src/swarm_larva.cu` + `.cpp` (§4 state machine, head-sweep spatial sensing on temperature, §4.1 integer movement law, threshold heat kill + cold coma), `SwarmSolver` in PhysicsEngine at the end of the slot-7 chain, bindings; 3-part CUDA gate (`tests/cuda_swarm_check.py` + wrapper). Kill/coma thresholds are `[swarm.larva]` config rows authored in **Kelvin** through `temperature_scale` (hot-reloadable, retunable after the system lands; immune to fire-12's scale change) | subagent, **design-gated** (impl doc written AFTER P2 lands + critic: determinism lens) | Opus / high for impl doc; Sonnet build | CPU==GPU byte-identity on all swarm arrays over forced-branch fuzz + 130-tick lockstep that drives every transition; golden reproduction; property tests on the movement law (no tunnelling at max speed, out-of-grid = solid, heading canonical); a matplotlib trajectory plot (room with a hot and a cold corner) as Erik's first look at the behaviour | no — lands on the arc branch on its oracle; **its feel test happens at P4**, the first patch that makes larvae visible in a level. Nothing reaches `main` unplayed (the arc merges to `main` at close) |
| P4 | **Brood entity + diagnostic view** — `swarm_brood` zone entity in the registry (editor gets it for free), spawn at level load through the P2 spawn path; render-only debug view (dots + heading tick, coloured by `ai_state`) via the existing overlay system | subagent | Sonnet / medium | registry/serializer round-trip; level lint; spawn count/placement properties; overlay render-only (goldens untouched) | **yes** — Erik's first play-test: **behaviour only** (crawl, flee fire, freeze) |
| P5 | **Real bodies** — research + design pass first (batch/instanced drawing in raylib, per-instance crawl phase from `dist_walked`, LODs, fit with the `lit3d` seam), a look-check shot of the low-poly larva (spike is 4,332 tris → target 50–300) beside the blessed spike, then promote `larvagen` into `renderer/` (reusing `propgen` primitives, no copy), batch renderer, crawl wave, husk fade on death event | research + design (Opus), build subagent (Sonnet) | Opus / high for design; Sonnet build | render-only (goldens unchanged); screenshot smoke test | **yes** — Erik's second play-test: **look only** |
| P6 | **Food field** — `food` synced int32 per-tile field (0–255), painted sidecar `food.npy` via `level_lib`, editor paint tool (pure core + shell), FieldEdit writer for events, kernel depletion + clamp pass, EAT state + dwell law; **digest spec v6 (all goldens regenerate)** | subagent | Sonnet / medium | field-digest v6 regen in the same commit with rationale; A/B; property tests (depletion conserves up to clamp, larvae dwell longer on food) | no — **SEQUENCED after fire-12 merges to main; merge main in first**. May swap with P5 |
| P7 | **Playground + arc close** — brood + painted food on the playground level; CLAUDE.md rows, issue #63/#46 updates, archive working docs | inline + Haiku for mechanical parts | — | suite green; level lint | **yes** — Erik plays it |

**Research that may run in any natural wait** (it depends only on the
renderer and chapter 17 §2's array layout, not on P2/P3): the P5 research
pass. One agent at a time still applies.

**Deliberate omissions (ACCEPTED GAP, per the chapter):** larva–larva collision,
blast/wave coupling, shootability, stamping, vision presence, gas emission,
per-segment damage, satiety, temporal-sensing A/B switch, world decal layer
(separate arc), follow-the-path spine at swarm scale.

**Order:** P1 → P2 → P3 → P4 → P5 → P6 → P7 (P6 may swap with P5 depending
on when fire-12 merges).

**Future hook (not this arc):** the larva-infested zombie
(`docs/beastiary/beastiary.md`, Erik 2026-09-24) — a zombie's death event
spawns a brood through the P4 spawn path.
