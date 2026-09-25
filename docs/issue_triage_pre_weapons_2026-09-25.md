# Issue triage: what to fix before weapons and explosions (2026-09-25)

This is the working list for the session at #12's close. That session decides
what to fix before the **RL ready** milestone starts. The milestone's order is:

1. #31 + #8 explosion
2. #6 smoke graphics
3. #74 black-body graphics
4. #75 explosion VFX
5. #20 flamethrower
6. #77 S8 residency
7. #76 RL wrap-up

Every recommendation here is a **proposal**; nothing is ruled yet.

**Method.** An Opus agent read all 68 issues open on 2026-09-25: every body,
the last two comments, and #46 in full. It checked eight claims against the
code on `fire-12` @ `011ef17`. Anything it reasoned rather than read is marked
*(inference)*. The orchestrator then re-checked three claims:

- #11's test passes on the current CUDA build.
- The grenade's heatless ignition is real in code; it is now filed as **#79**.
- #62's overlay is display-only.

## 1. The worry: is there a fundamental heat↔atmosphere mistake that could go unstable?

**No open issue reports a live runaway or storm at the shipped settings.**

**The one fundamental defect is #72, and it is bounded.** The EOS breaks the
second law under expansion.
- In a sealed box, the lowest specific entropy falls 0.10 c_v below its start
  within 0.3 s, and is still 0.12 c_v below at 600 s.
- A corner cools to −54 K where the isentrope allows −28 K.
- A pen sealed at load grows a ±12 K checkerboard within about 1 s, then drifts
  +1.1 K over 600 s.
- Energy is conserved; it is entropy that is wrong. It saturates rather than
  growing, so it is inaccuracy, not instability.
- The issue lists one candidate cause among several: energy leaves a cell one
  tick before its mass does. That would explain the 12 K lost at tick 1
  *(inference)*.

**Instability appears only at settings we do not ship:**
- `mg_cycles = 6` destabilises the map (we ship 8).
- `flat_gs` (no multigrid) storms (closed #70).
- `k_drag = 10` is "venting death" (we ship 0.5).
- `n_floor_heat` must be ≥ 0.01, and we ship exactly 0.01, so there is no margin
  below it.

**One signal under sustained forcing has not been re-measured.** In #62's
probes (2026-09-06), the playground's vent loops ran for 3000 ticks with:
- P_std climbing linearly;
- return apertures 267–400 game units hot;
- room mean T falling;
- gas N exactly conserved.

The issue blames missing makeup/Kp regulation. The growth is linear, not
exponential. Nobody has measured it since T5b/P5. It is worth one bench run
*(the orchestrator's suggestion)*.

**Books that do not close** (energy moves and no counter sees it):
- #73 items 1–3: the solid low-T rail, `destroy_wall` (2^21 raw in one
  measurement), and combustion below `n_floor_heat`.
- #71: the CUDA combustion twin has no gas-energy ledger. This does not touch the
  shipped game, because `--cuda` never enables that backend.

**Books that close but hide drains** (inaccuracy):
- #73 item 4: the solid floor-division sink, about 100 kW map-wide by the
  agent's arithmetic. Walls drift down by up to ~1.3 K/h.
- #72 item 3: gas moving through permeable furniture loses 292 kJ per 60 s.
- A Helmholtz resonance in the playground's SW doorway (#70).

**One old oddity:** an O2 fraction of 0.2226 (above ambient) about 18 minutes
after a fire died, at the ~1 % level, unexplained (#12, comment of 2026-09-06).

## 2. Fix before the milestone starts (proposed, ranked)

1. **#72: the EOS second-law defect.** Large; its own arc, CUDA twins included.
   - A grenade adds up to +10 N_amb of gas at ambient temperature with no clamp.
     That is the same expansion, only larger.
   - #8 wants a blast to "offset its own expansion cooling", and #31 calibrates
     overpressure, impulse and duration against Kingery–Bulmash. Both would be
     fitted to the wrong cooling and a lagging pressure solve.
   - #31 reduces every weapon to one yield number, so the error would reach
     every weapon row.
   - #31's design doc can be written in parallel.
2. **#73: the thermal books gaps.** Small to medium.
   - `destroy_wall` is exactly the call a blast uses to break a wall. Every
     explosion bench would carry a known hole where explosions act.
   - #5 and #29 already demand closed books before tuning and replay.
3. **#4: your ruling on `k_drag2`.** A decision, not a build.
   - The measured band at the blast scenario is k2 ≈ 0.10–0.15; we ship 0.
   - Either freeze it at 0 for the milestone, or turn it before #8 is tuned, so
     the explosion is not tuned under a moving dial.
4. **#11: close it.** Its bit-identity test passes on the current build
   (verified 2026-09-25), and the kernel it indicted was deleted in arc #54.

## 3. Gates inside the milestone (not before it)

- **#31's first patch** (each item small):
  - #79, the heatless grenade ignition (new);
  - #62, the pressure overlay reading `atmosphere + wave_p`;
  - #59's B key, which both toggles bilinear filtering and arms explosives;
  - #22, the hardcoded blast-material tuple (steel, glass and furniture are
    blast-immune);
  - #9, walls broken by a blast keep their graphics.
- **Before #6's retune:** #51, the wind units, which blast smoke reads.
- **Before #20's experiments:** #68 (two-node solids; lumped-model ignition is
  20–45× too slow), plus #61 (merge into #68) and #23.
- **Before any all-GPU run that claims determinism:** #71.
- **At #12's close:**
  - #67's arc-close items;
  - the cross-arch attestation file, stale since July;
  - a refresh of #46.

## 4. Close or merge (proposed)

**Close:**
- **#11:** verified green.
- **#10:** its own comments show water conserved to the bit through fire, blasts
  and vacuum. Keep the water check among #31's gates.
- **#5:** superseded. `wall_damage` has been re-decided, the fire anchors were
  redone in #12, and what is left is #4's P3.
- **#60:** done. Your glass-terrarium feel test is still owed.
- **#59:** after fixing the B key.
- **#66:** its ruling is in use; write it into the skill.
- **#12:** at P7.

**Merge:**
- #61 → #68: the same defect.
- #8 is already inside #31.
- #63's grid-scale T pattern is #72's checkerboard.
- #62's vent-loop note duplicates #48's owed re-test.
- #37's explosion visuals duplicate #75.
- #38's hardcoded explosion params overlap #31.
- Riders: #24 and #25 belong to #77; #27–#29 feed #76; #40 belongs to #75; #36's
  light banding goes to #12 P6b/#75.

**Re-scope:**
- #53: mostly landed; keep the controls review.
- #15: split its deferred items into #74, #31 and #6.

## 5. Corrections to what we thought

- **#62's "~1 atm DC offset" is by design.** `wave_p` is the previous tick's
  pressure buffer since EOS P3. Only the overlay is wrong.
- **#8's "no heat term" is stale.** Explosions have deposited temperature since
  EOS P3, but in legacy units and skipping solids, so #8's practical point
  stands.
- **#4 is built.** P1 and P2 are merged; only turning the dial (P3) is left.
- **#51 is still live.** CLAUDE.md's "Tamed wind" row repeats the wording #51
  says is wrong.
- **#46 is stale in places.** It still lists #7 as open (closed 2026-09-23) and
  #59 as un-root-caused.
- **Tracked nowhere, from #63's 2026-09-24 comment:**
  - the float ratchet does not scan headers (a determinism-gate gap);
  - fire-12's fan phase keys on the round-local tick;
  - `GameMap.residency_on()` is never cleared;
  - `monotonic_total_tick` double-counts under OnePhaseWEGO.

## 6. Every open issue

| # | Short title | Area | Kind | Affects weapons & explosions? | Size | Proposal |
|---|---|---|---|---|---|---|
| 4 | Drag law v2 (only P3 dial-turn left) | physics | design debt | maybe — blast venting band depends on k_drag2 | S ruling / M | BEFORE (ruling) |
| 5 | Post-pressure retune pass | physics | superseded | maybe — only k_drag sizing (= #4 P3) | M | CLOSE |
| 6 | Smoke saturates to black | render | live bug | yes — grenade clouds | M | milestone 2 |
| 8 | Grenade energy budget | physics | live bug | yes — the first explosion instance | M | milestone 1 (with #31) |
| 9 | Pressure-burst walls keep their graphics | render | live bug | yes — broken walls must look broken | M | WITH #31 |
| 10 | Water escapes sealed aquarium (unverified) | physics | superseded | maybe — keep the water gate | S | CLOSE |
| 11 | CUDA kick-compression PART 2 diverges | determinism | stale | no longer — test green 2026-09-25 | S | CLOSE |
| 12 | Fire & heat / ray-engine arc | physics | in flight | yes — P6 light feeds #75 | M | CLOSE at P7 |
| 13 | Golden-suite co-design | determinism | design debt | yes — the golden never gated real fire or blasts | M | WITH #31 |
| 14 | Skills backlog | tooling | feature | no | M | AFTER |
| 15 | Canonical-systems cleanup (deferred half) | housekeeping | design debt | maybe — blast-path tombstones | M | split: #31 / #74 / #6, rest AFTER |
| 16 | Momentum / sprint / unit collision | gameplay | feature | no | M | AFTER |
| 17 | Armory tuning (parked) | feel | feature | yes — weapon balance, after mission 1 | M | AFTER |
| 18 | Airlock v2 buttons | gameplay | feature | no | S | AFTER |
| 19 | Door↔unit occupancy | gameplay | live bug | no | M | AFTER (flag for #76) |
| 20 | Flamethrower session | gameplay | feature | yes | L | milestone 5 |
| 21 | Grenade gamepad remap | feel | live bug | no | S | AFTER |
| 22 | Blast-pressure-threshold material column | physics | design debt | yes — steel/glass/furniture blast-immune | S–M | WITH #31 |
| 23 | Fire never destroys furniture | physics | live bug | yes — flamed crates stay as husks | M | WITH #20 |
| 24 | CUDA graphs (parked) | tooling | feature | no | M | AFTER #77 |
| 25 | Resident sensor-gather kernel | tooling | feature | no | M | WITH #77 |
| 26 | Remove vestigial C++ is_wall | housekeeping | design debt | no | S–M | AFTER |
| 27 | Levelgen v0.1 | tooling | feature | no | M | WITH #76 |
| 28 | TM communication contract | research | feature | no | M | WITH #76 |
| 29 | RL substrate umbrella | tooling | feature | no | L | WITH #76/#77 |
| 30 | Mission 1 | gameplay | feature | no | L | AFTER |
| 31 | Generic explosion archetype | physics | feature | yes — it is the explosion | L | milestone 1 |
| 32 | Marines walk in place | render | live bug | no | S | AFTER (before #77's watch gate) |
| 33 | Marine appearance / skins | render | feature | no | L | AFTER |
| 34 | Weapon/grenade animations | render | feature | maybe | M | AFTER |
| 35 | Retire the M toggle | housekeeping | design debt | no | S | AFTER |
| 36 | Renderer review sweep | render | design debt | maybe — light banding in long rays | M | banding WITH #75; rest AFTER |
| 37 | Units/gameplay small items | gameplay | design debt | maybe — duplicates #75 | M | AFTER |
| 38 | Pathfinding + legacy sweep | housekeeping | design debt | maybe — explosion params hardcoded | M | AFTER |
| 39 | Lights → entities | render | design debt | no | M | AFTER |
| 40 | Scorch marks + blood | render | feature | maybe — grenade decals | M | rider of #75 |
| 41 | LOS filter + wall collision | gameplay | live bug | maybe — grenades don't bounce | M | WITH #76 |
| 42 | Resolution audit | housekeeping | design debt | no | S–M | AFTER |
| 43 | Breach decompression / lingering haze | physics | design debt | maybe — breaching blasts vent | M | AFTER (re-check in #6) |
| 44 | Creature AI + roster | gameplay | feature | no | L | AFTER |
| 45 | Campaign / narrative (parked) | gameplay | idea | no | L | AFTER |
| 46 | ROADMAP | housekeeping | — | no | S | refresh at #12 close |
| 48 | Vent entity (feel patch) | physics | design debt | maybe — vents skew blast baselines | M | AFTER (re-test after #72) |
| 49 | Vents v2 state machine | gameplay | feature | no | M | AFTER |
| 50 | Duct auto-placement | tooling | feature | no | M | AFTER |
| 51 | gas_detail wind units | render | live bug | yes — blast smoke reads the tamed wind | S | WITH #6 |
| 52 | Flame-structure VFX (parked) | render | feature | maybe | L | AFTER |
| 53 | Pre-tuning instrument pass | tooling | mostly landed | maybe — key collisions | M | re-scope |
| 55 | Research parking (WASP) | research | idea | no | — | AFTER |
| 56 | Behaviour trees | gameplay | idea | no | L | AFTER |
| 57 | Watchable scenario mode | tooling | feature | maybe | M | AFTER |
| 59 | Brightness shifts on grenade | render | live bug (G fixed, B left) | yes — B arms explosives | S | WITH #31, then CLOSE |
| 60 | Props & vegetation | render | done | no | S | CLOSE |
| 61 | Wood can't sustain fire | physics | live bug | yes — flamer/blasts can't keep wood lit | (in #68) | merge into #68 |
| 62 | Pressure readout double-counts | render | live bug (display) | yes — F7 reads +1 atm | S | WITH #31 |
| 63 | Fauna arc (larvae) | gameplay | feature | no | L | parallel track |
| 64 | Two-state lit/dark look | render | idea | maybe — how blast light reads | S | WITH #75 (or at P6b) |
| 66 | Token budget | housekeeping | superseded | no | S | CLOSE |
| 67 | Ray-engine v2 housekeeping | housekeeping | design debt | no | S | arc-close items BEFORE |
| 68 | Two-node thermal solids | physics | design debt | yes — ignition 20–45× too slow | L | WITH #20 (before experiments) |
| 69 | pyright baseline | tooling | design debt | no | S | AFTER |
| 71 | CUDA combustion not bit-identical | determinism | live bug (GPU only) | maybe | M | WITH #77 |
| 72 | EOS second-law defect + checkerboard | physics | live bug | yes — blasts are expansion | L | **BEFORE** |
| 73 | Thermal books gaps | energy books | live bug | yes — blast breaches unbooked | S–M | **BEFORE** |
| 74 | Black-body graphics | render | live bug | yes | M | milestone 3 |
| 75 | Explosion VFX | render | feature | yes | L | milestone 4 |
| 76 | RL-ready wrap-up | tooling | feature | no — the exit gate | L | milestone 7 |
| 77 | S8 GPU residency | tooling | feature | maybe | L | milestone 6 |
| 79 | Heatless explosion ignition (new) | physics | live bug | yes — decorative grenade fires | S | WITH #31 (first patch) |
