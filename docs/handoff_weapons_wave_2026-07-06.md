# HANDOFF — the weapons wave (W-wave), 2026-07-06

**For:** a fresh Claude Code session on the **Lenovo laptop** (or any box), after Erik's
weekly token reset (night to Tuesday 2026-07-07). Erik is traveling to Norway — this doc is
the complete resume point. Work top-to-bottom.

**Erik's opening line can simply be:** *"Read docs/handoff_weapons_wave_2026-07-06.md and
resume the weapons wave."*

---

## 0. Memory transfer (do this first)

This wave was orchestrated from the Work Desktop, whose Claude auto-memory does NOT sync.
A full snapshot was copied to the ClaudeSync Google Drive folder:

    <ClaudeSync>/breach_memory_snapshot_2026-07-06/
    (G:\My Drive\ClaudeSync\... — "Min enhet" on Swedish-locale boxes)

Read **`project_game_design_kickoff.md`** (its tail = the wave log: every design decision
Erik blessed, every patch's findings) and **`MEMORY.md`** (the index). Merge what you need
into your own memory. The chapter below is canon regardless — memory is color, the repo is
truth.

## 1. State of the wave (as frozen)

| | |
|---|---|
| main | `1d765b7` (pushed) — W1 `2abf7dc` + W2 `bbfb26a` + W3 `1d765b7` all merged |
| suite | **587 passed** via the exact command in §4 (twice-verified on main at freeze) |
| golden | `07c3f37043c62cb47ec1abfef1a59d47c5f7a9c313490b38ecd2ddc543d1833d` — **UNCHANGED through all three patches** (the lazy-roll rule: dormant features draw no RNG) |
| branch `weapons-w4-spray` | pushed, **one WIP commit `f6d9bf8`** (weapons.py schema columns only — the W4 agent died at a session limit right after starting) |
| canon spec | `docs/architecture/mechanics/03_combat_and_weapons.md` — the framework chapter; §8 = the wave plan + per-patch findings of record |

Shipped so far: weapon/ammo/payload tables + re-home (W1) · unified march, aim/snap
spread, exposure-vs-cover + crit-vs-facing live, Lance-3 laser with integer gas attenuation
(W2) · payload executor, smoke/tear/poison grenades, TEARGAS→BLINDED + POISON→DoT coupling
rows (zombies poison-immune), GL-6 sharing grenade payload rows, C4, mag/reload machinery
(W3).

## 2. What remains

- **W4 — SPRAY (Dragon-7 flamethrower + Miasma Vent)**: barely started; branch exists.
  **HUMAN-TEST GATED**: build + gate + push the branch, **do NOT merge** — Erik feel-checks
  first (edit `[marine] weapon = "dragon_7"` in config, run the playground, hose a room),
  and only after his blessing does the orchestrator merge. Full verbatim agent prompt in §5
  — relaunch a fresh subagent with it (note: the branch already exists with `f6d9bf8`, so
  the agent should `git checkout weapons-w4-spray && git pull` and continue, not recreate).
- **W5 — MELEE**: combat knife + arc baton through the W2 resolver (to-hit trivially 1.0
  without cover; crit arcs do the work — knife crit_chance 0.15, behind-arc assassin
  fantasy); baton applies STUNNED 1.5 s at the hit site (statuses at delivery sites,
  packets stay damage-only); zombie melee stays on its ai_zombie path. Hands-off,
  auto-merge on green, golden expected unchanged (armory rows §6 of the chapter).
- **W6 — the armory + Erik's session**: all remaining armory rows as pure data — P12
  Whisper, MP-11, LR-50, Jackhammer-8 (pellets = shots_per_trigger), Lance-5, **Sunspot +
  Helios plasma** (= the GL-6 payload-on-projectile machinery + small heat-splash payload +
  glow visual), incendiary grenade + 40mm variants if missing; a **weapon-cycle debug key**
  in the playground; full standard-values audit pass (comment style per config precedent);
  playground guide update. **HUMAN-TEST**: ends in Erik's grand tuning session.

After W6: the wave-close ritual — chapter §8 rows flipped, README, memory checkpoint,
suite + golden statement, and Erik compacts. Then agenda item 6 (rules/missions design
session — Claude leads, decisions together).

## 3. The working model (blessed, standing)

Plan is agreed → each patch runs in a **fresh subagent on a branch** → gate = full suite +
golden procedure → **auto-merge on green + push autonomously** — EXCEPT human-test patches
(W4, W6), which push the branch and stop for Erik. Commit at every stage boundary
immediately (this is what made two session-limit deaths in this wave costless). One branch
agent at a time, never concurrent. Checkpoint memory at every patch boundary.

## 4. Standing rules (verbatim-critical)

- Suite command, ALWAYS (never bare `pytest` — it hangs):
  `<python> -m pytest tests/ --ignore=tests/test_main_smoke.py --ignore=tests/test_renderer_smoke.py -q`
  (Work Desktop python: `C:/Users/steen/anaconda3/python.exe`; **Lenovo: `C:/Users/steen/miniconda3/python.exe`**.)
- Golden procedure: run `tests/_xarch_perfield_digest.py` BEFORE touching any golden;
  re-baseline ONLY with a per-field proof of expected movement (P3/P4 precedent: update
  GOLDEN_AGGREGATE + append-only lineage blocks in `tests/test_cuda_s*.py` + engine/14 note).
  W4/W5/W6 add no RNG consumers → golden should stay `07c3f370…`; movement = a bug.
- `git add` **explicit file paths only** — never a directory (Erik's untracked files:
  lore docs, spikes, pkl baselines must never enter a commit). Never `git reset --hard`,
  never force-push, never `rm -rf`.
- Don't touch `docs/architecture/engine/07_fluid_and_water.md` or anything water — that is
  a separate Claude's project (water interface = per-tile `water_depth` heat-sink only).
- Design docs land as canon chapters in `docs/architecture/`; status sections stay honest.
- Determinism: the four doors (engine/14). Kit trig (`unit_fixed`) only on synced paths;
  `rng.uniform/.integers/.random` legal, distribution methods banned; quantize constants
  once at load; ingress lint (`tests/test_ingress_lint.py`) is part of the suite.
- Lenovo gotchas: miniconda interpreter (`.pyd` ABI cp3xx must match); CUDA `.pyd` needs
  `os.add_dll_directory(<CUDA>/bin)` (cuda_harness handles it); the CUDA gates run there —
  it is the attested Ada box (`cuda-breached` at `86aad36`).

## 5. The W4 agent prompt (verbatim — relaunch with this)

> You are implementing patch W4 of the breach weapons wave: the SPRAY archetype — the
> Dragon-7 flamethrower and the Miasma Vent poison projector. THIS PATCH IS HUMAN-TEST
> GATED: you build, gate, and PUSH THE BRANCH — you DO NOT merge to main. Erik feel-checks
> first. Repo: the breach repo root.
>
> **Read first (your spec):** (1) docs/architecture/mechanics/03_combat_and_weapons.md:
> §1 (SPRAY = sustained cone of field writes, no projectile; two-terminals invariant —
> SPRAY touches NO unit directly), §5 SPRAY bullet, §6 armory rows (Dragon-7: 30° cone,
> range 8, 1.5 s burst, loudness 0.6; Miasma Vent: 25°, range 7, 1.5 s), §8 wave plan.
> (2) docs/architecture/engine/06_temperature_and_fire.md — how heat becomes temperature
> becomes ignition; find IN CODE where a heat deposit must land so the C++
> TemperatureSolver converts it (grep heat handling in src/simulation/physics_runner.py,
> gamemap.py, the fire heat-ray path) — the flamethrower deposits into THAT field through
> the established write path. (3) src/simulation/payloads.py (W3's deterministic gas
> deposit — REUSE for the cone's gas part), field_edit.py (gas edits: field="gas" +
> channel), exchange.py (the heat coupling row that damages units standing in flames — you
> write NO unit damage), combat.py (archetype dispatch — add the "spray" branch),
> status.py (composed can_act), weapons.py, simulation.py (conductor — call lines only).
> (4) engine/14 + tests/test_ingress_lint.py constraints (kit trig only on synced paths;
> NO RNG anywhere in W4).
>
> **Branch:** git fetch origin && git checkout weapons-w4-spray && git pull (the branch
> exists with WIP commit f6d9bf8 — continue it; if you must re-derive its weapons.py edit,
> diff it against main first). Commit at every stage boundary. NEVER git reset --hard /
> force-push / rm -rf. Nothing water-related. git add EXPLICIT FILE PATHS ONLY — never a
> directory.
>
> **Build — A. SPRAY mechanics:** a fire order with a spray-archetype weapon starts a
> sustained burst (v1: spray only on explicit stationary fire orders; auto-fire skips
> spray weapons; the sprayer stands still — document): for burst_seconds × tps ticks the
> unit sprays, each tick depositing into a CONE toward the order target. Cone membership:
> tiles within range_tiles whose bearing from the shooter is within cone_half_angle of the
> aim bearing — INTEGER-SAFE geometry: dot(tile_dir, aim_dir) >= |tile_dir| ·
> cos(half_angle), cos via the deterministic kit or a load-time quantized constant (door
> 2); avoid per-tile atan2; fixed row-major traversal. Occlusion: a cone tile receives
> deposits only if gmap.has_los(shooter, tile) — flames do not pour through walls.
> Falloff: simple deterministic integer 1/distance-style form (document). Per-tick
> deposits (established write paths, quantized once, NO RNG): Dragon-7 (ammo
> fuel_standard): heat_deposit per tile into the heat/temperature ingress field found
> above (falloff-scaled) + fuel_gas emission (small gas_amount, falloff-scaled) via W3 gas
> edits — units in flames take damage via the EXISTING heat coupling row (assert in a test
> that no W4 code touches unit HP). Miasma Vent (ammo toxin_standard): poison gas emission
> only (sustained, bigger than the grenade's instantaneous cloud); damage via W3's poison
> row; no blindness (poison ≠ teargas). Spray interruption: composed can_act False stops
> the burst that tick, order consumed, no resume. Economy: mag_size counts BURSTS
> (Dragon-7 mag 4, reload 4.0 s; Miasma 4, 4.0 s) via W3 machinery. Tick placement: sprays
> deposit in the shooting slot; call line only in simulation.py.
>
> **B. Config rows:** [weapons.dragon_7]: archetype="spray", ammo_family="fuel_tank",
> cone_half_angle_degrees=15.0 (armory's "30° cone" is the FULL angle — implement
> half-angle and document the convention in the chapter), range_tiles=8,
> burst_seconds=1.5, mag_size=4, reload_seconds=4.0, ap_cost=1, crit 0, mass_kg=8.0,
> loudness=0.6, spread fields 0. [ammo.fuel_standard]: family="fuel_tank",
> heat_deposit=<derive a standard value that clearly ignites wood-class flammables within
> ~1 s of spraying given the temperature convert path — show the derivation from
> ignition_temp and the convert/cool rates in a comment>, gas_species="fuel_gas",
> gas_amount=0.15/tick at source falloff. [weapons.miasma_vent]: spray, "toxin_tank",
> 12.5° half, range 7, burst 1.5 s, mag 4, reload 4.0, mass 6.0, loudness 0.4.
> [ammo.toxin_standard]: family="toxin_tank", gas_species="poison", gas_amount=0.35/tick
> at source. STANDARD VALUE comments in the established style (Erik's dials).
>
> **C. The feel-check hookup (required):** make the marine's default weapon data-driven:
> [marine] weapon = "k5_carbine" in config.toml, consumed where unit generation assigns
> weapon_id (replace W1's literal). Erik's loop: edit config to "dragon_7", relaunch the
> playground (main.py --level playground), issue fire orders, watch rooms burn. Note in
> your report whether restart or Ctrl+R applies (construction-bind).
>
> **D. Docs:** mechanics/03: §5 SPRAY fleshed to match implementation (cone convention,
> occlusion, stationary rule, deposits), §8 W4 row → "⏳ BUILT on branch weapons-w4-spray,
> awaiting Erik's feel-check" + gate facts + findings; README row 03 likewise. Explicit
> paths.
>
> **Determinism (hard):** NO RNG anywhere in W4. Kit/quantized trig only. Integer/Q16.16
> deposits through established write paths. Fixed traversal. Ingress lint green.
>
> **Gate (on the branch — hard):** suite command EXACTLY as §4 of the handoff (adapt the
> interpreter to this machine). New tests (tests/test_spray_weapons.py): cone membership
> exactness (hand-computed boundary tiles); occlusion; falloff pinned; Dragon-7 ignites a
> wood tile within the expected tick count (end-to-end through temperature convert +
> ignition, CPU backend); no-direct-unit-damage assertion + end-to-end marine-in-flames
> loses HP via the existing row; Miasma poison accumulation + zombie takes 0 (immunity);
> can_act interruption; stationary-only rule; burst/mag/reload cadence; dormancy replica
> (no spray weapon equipped → bit-identical trajectory + RNG end-state vs pre-W4). Golden:
> tests/_xarch_perfield_digest.py must stay 07c3f370… — movement is a bug, fix, never
> re-baseline. Full suite green on the branch.
>
> **On green: PUSH THE BRANCH, DO NOT MERGE.** git push -u origin weapons-w4-spray, then
> STOP (Erik feel-checks; the orchestrator merges after his blessing).
>
> **Report back:** branch head; test count; golden verdict; WHERE the heat deposit lands
> (field + why) + heat_deposit derivation; cone-angle convention; exact feel-check
> instructions for Erik; files; findings.

## 6. W5/W6 launch notes (for the orchestrator, after W4 is blessed + merged)

Write the W5/W6 prompts in the W1–W4 style (read the chapter §8 rows + §6 armory + the W3
report facts in the memory snapshot). W5 is small and hands-off. W6's extra inputs: plasma
= GL-6 detonate-at-stop machinery + [payloads.plasma_splash_*] (small radius heat/ignite,
no wall damage beyond modest, glow event like LaserFiredEvent — check events.py) + the
armory table; the playground weapon-cycle key goes through the renderer/input layer as a
debug action (renderer never mutates sim state — send it as an action/order through the
facade, mirroring how existing debug keys work); update docs/playground_guide.md.

---

*Frozen 2026-07-06 by the Work Desktop orchestrator session (weapons wave, agenda item 5).
Wave log lives in the memory snapshot; canon in mechanics/03. Good flight, Erik.*
