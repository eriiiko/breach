# Swarm units — patch plan (arc #63, branch `63-swarm-units`)

> Executes `docs/architecture/engine/17_swarm_units.md` (BLESSED 2026-09-10)
> under the autonomous-patch-workflow. Plan agreed with Erik before execution;
> re-planned at patch boundaries. Oracle-gated patches auto-merge on green
> (standing grant); HUMAN-TEST patches wait for Erik. Worktree
> `C:/Users/steen/projects/breach-swarm`, one writer at a time. Critics one at a
> time, verdicts on disk. Tests assert properties, never snapshots.

| # | Patch | Mode | Tier / effort | Gate (oracle) | HUMAN-TEST |
|---|---|---|---|---|---|
| P1 | **Adopt philox32** — vendor `cpp/src/philox32.h` + `src/simulation/philox32.py` (pinned to philox32 `6c33a3d`, hash in header); `/fp:strict` list; KAT + cross-language test in `tests/`; chapter-14 door-4 amendment + CLAUDE.md canonical row "Deterministic RNG"; ingress lint accepts it | subagent | Sonnet 5 / medium | KAT + Python-vs-C++ identity; full suite green; goldens untouched (no sim call yet) | no |
| P2 | **Swarm store scaffold** — `gmap.swarm_*` SoA `(N, max_units)` arrays (§2 roster, `max_units` config default 8192), residency allocation, `SWARM_SECT_V1` presence-gated digest section + recorder + save hooks, spawn/reclaim (`valid`, `next_unit_id`, `high_water`), `src/simulation/swarm.py` facade Simulation drives, species table `[swarm.larva]` schema-in-code/numbers-in-TOML, CFG hot-reload; NO behavior kernel | subagent, **design-gated** (impl doc `P2_impl.md` + one critic first — wire/digest-touching) | Opus / high for the impl doc; Sonnet 5 for the build | every existing golden byte-identical (presence gate); A/B harness extended; property tests: spawn fills sequentially, reclaim, high_water monotone, section absent ⇒ digest identical | no |
| P3 | **Larva kernel + CPU twin** — `cpp/src/swarm_larva.cu` + `.cpp` (§4 state machine, head-sweep spatial sensing on temperature, §4.1 integer movement law, threshold heat kill), `SwarmSolver` in PhysicsEngine at the end of the slot-7 chain, bindings; 3-part CUDA gate (`tests/cuda_swarm_check.py` + wrapper) | subagent, **design-gated** (`P3_impl.md` + critic: determinism lens) | Opus / high for impl doc; Sonnet 5 build | CPU==GPU byte-identity on all swarm arrays over forced-branch fuzz + 130-tick lockstep that drives every transition; golden reproduction; property tests on the movement law (no tunnelling at max speed, out-of-grid = solid, heading canonical) | **yes** — Erik watches larvae crawl/flee/freeze in the playground (feel) before merge |
| P4 | **Food field** — `food` synced int32 per-tile field (0–255), painted sidecar `food.npy` via `level_lib`, editor paint tool (pure core + shell), FieldEdit writer for events, kernel depletion + clamp pass, EAT state + dwell law; **digest spec v6 (all goldens regenerate)** | subagent | Sonnet 5 / medium | field-digest v6 regen in the same commit with rationale; A/B; property tests (depletion conserves up to clamp, larvae dwell longer on food) | no — **but SEQUENCED after fire-12's re-baselines merge; rebase first** |
| P5 | **Render bridge (minimal — graphics on hold)** — instanced low-poly larvagen bodies (≤300 tris, 2 LODs) via the existing unit-model path, crawl wave from `dist_walked`, husk fade on death event, hover-readout count | subagent | Sonnet 5 / medium | render-only (no digest impact — proven by golden unchanged); screenshot smoke test | **yes** — Erik feel-check |
| P6 | **Playground** — `swarm_brood` zone entity in the registry, brood + painted food on the playground level; arc close: CLAUDE.md rows, issue #63/#46 updates, archive working docs | inline + Haiku for mechanical parts | — | suite green; level lint | yes — Erik plays it |

**Deliberate omissions (ACCEPTED GAP, per the chapter):** larva–larva collision,
blast/wave coupling, shootability, stamping, vision presence, gas emission,
per-segment damage, satiety, temporal-sensing A/B switch, world decal layer
(separate arc), follow-the-path spine at swarm scale.

**Order:** P1 → P2 → P3 → (P5 may run before P4 if fire-12 has not merged) → P4 → P6.
