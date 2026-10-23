# Fire session #12 — R4, the tuning lab, and the radiation finding (2026-09-06, evening)

> **Status**: NOTHING HERE IS BLESSED FOR MERGE. The branch `fire-12` carries
> it all so it survives the night and the machine switch, and the canonical
> golden is deliberately LEFT RED (see §5). Continuation of
> `fire_3c_r3_report_2026-09-06.md`, which this supersedes on every point
> where the two disagree.
>
> **Pick up at §6** — the radiation question is the whole of tomorrow.

---

## 1. What actually happened tonight

Erik drove `tools/fire_tuning_lab.py` (new, §2) for a long tuning session.
Three of his rulings landed as code/config, a fourth (the burn-duration
target) landed and was then thrown into doubt by the finding in §4.

Sequence, honestly: we tuned a **single tile** all evening, got it doing
exactly what Erik wanted, he blessed it — and then the first spread
measurement of the entire session showed the blessed dials set the whole
room on fire in seconds. The blessing was given without that information.
It is not a bug in the tuning; it is a genuinely new question (§6).

## 2. The lab (`tools/fire_tuning_lab.py`) — landed and useful

Erik's ask: params at the top of one script, edit → run → plot → repeat, on
a real level, extensible to H_bed later.

- Panel at the top: `LEVEL`, `SIM_SECONDS`, `IGNITE_TILES` (station
  catalogue in comments), `IGNITE_T_MARGIN`, and `DIALS` grouped by role.
  Every dial is a runtime CFG override through `fire_timing_harness`'s own
  `apply_overrides` seam — never a parallel bench, nothing written to disk.
- Loads a **real shipped level** via `level_loader.load` (default
  `fire_tuning`, which is `boundary="space"` — sealed hull, no sky refill,
  one fixed O2 inventory: honest ship physics, and what Erik wants to tune in).
- Records per tick: I, T (game/K/°C), hot, **hotf**, o2f, F, avail, I_cap,
  X_local, X_room → 4-panel plot + CSV under `tests/_fire_lab/`.
- Cross-validated against the m1 bench: peak I 0.850 @ 13.1 s vs the bench's
  0.849 @ 13.1 s.
- The summary line **names the death cause** (FUEL / HEAT-COLLAPSE / O2).
  Added after a heat-collapse at 162 s with 85% fuel left was misread as a
  fuel burnout from the plot alone — that misread cost us an hour.

**`IGNITE_T_MARGIN` is a measuring stick, not a game dial**: it seeds a tile
that many kelvin above its own ignition point, standing in for "how
decisively was this lit?" (a graze ≈ 0; a flamethrower held on ≈ +50…+150).
The property being measured is the *survival threshold* — the smallest
margin that lives.

**Methodological lesson, paid for twice**: 120 s / 300 s / 700 s windows all
produced false "SURVIVES" verdicts. Anything you intend to keep gets a
1800–2400 s confirmation run.

## 3. Rulings and measurements

### 3.1 R3 re-anchor — BLESSED, landed

`burn_rate 0.00965 → 0.018`. The R3 landing anchored "neutral" at the
measured *plateau* (f_ref 2.0717); fires are born at *ignition*, where
`hotf = Δ/span = 200/180 = 1.1111` for **every** material (the per-material Δ
cancels exactly). Anchoring there makes the bootstrap pre-R3-neutral while
everything hotter burns faster — which is also exactly Erik's independent
wish for the effective factor to run "[1, 9] from ignition upward".

**Co-move rule**: `burn_rate = 0.02·span/Δ`, `wall_damage = 0.03·span/Δ`.
Moving Δ without moving both reproduces the R3 anchor bug in miniature — it
bit us once tonight (Δ=120 with Δ200-rates killed a fire that should have
lived).

### 3.2 R4 non-flammable gate — BLESSED, landed (both twins)

`fire_simulation.cpp` + `cuda_fire.cu`: the `wall_hp` **depletion** is now
`flammable`-gated, as the destroy decision above it always was. Fire no
longer drains hp from tiles that can never burn.

### 3.3 The survival-edge tuning — measured, then SUPERSEDED

Erik wanted grazing ignitions to die and decisive ones to take hold.
Measured chain:

- **Δ (`ignition_to_ext_delta`) is the only dial that places the edge.**
  `k_grow`/`k_die`/the rates cannot: `k_die` up to 0.30 saved nothing and
  killed nothing.
- Δ=120 (rates co-moved) put the edge at ≈ +25 K. The boundary is a
  deterministic **fringe**, not a line: at Δ=110, margins 45 LIVE / 49 die /
  51 die / 55 LIVE — repeatable, caused by tick-discretisation near the
  separatrix.
- **But Δ=120 makes the plateau unstable**: heat-collapse at ~162 s for any
  ignition strength (+55 and +300 both), 85% of the fuel wasted.
- **A deposit strong enough for stable burns erases the edge**: at H_bed ×2
  a margin-0 graze bottoms out at 273.5 vs ignition 280 — even Δ=90 saves it.

**Conclusion: Δ cannot serve both the entry edge and the exit mode.**
Hence the reframe Erik agreed to: **the entry edge belongs to G8
exposure-integral ignition** — a graze never ignites *at all*, so there is
no young fire to kill afterwards — and Δ stays 200. Tonight's data is G8's
justification; promote it.

### 3.4 Burn duration — Erik's target, landed but now in doubt

Erik's ruling: the smallest fuel unit (a 30-hp furniture crate) burns out in
**~3 minutes**, and **fuel is the killer**.

Landed `wall_damage 0.36` + `H_BED_SHIFT 4 → 7` (H_bed ×8) → FUEL burnout at
**180–182 s**, robust to ignition strength (margins 10–100) and to k_grow
(0.382–0.5). Verified end-to-end through pure config.toml on the rebuilt
engine.

**Why ×8 was needed** — and this is the load-bearing structural fact:
`I_cap ∝ F`, so as fuel wanes the fire shrinks and its own drain fades with
it. F = 0 is asymptotically unreachable at weak deposits; measured deaths
were heat-collapses leaving 59% char (×1) and 19% char (×2). Only at ×8 does
hp win the race. **The char fraction is the real tunable, not "burns to
zero".**

H_bed ladder (plateau, slow-burn dials): ×1 296 K (dead) / ×2 683 K / ×4
769 K / ×8 917 K — the R2 fourth-root law visible end to end. Peak I falls
0.66 → 0.50 across the same ladder (toward G4).

## 4. THE FINDING: ×8 sets the whole room on fire

First spread measurement of the session (everything before it was one tile).
On `fire_tuning`, igniting station 2 at (22,8):

| H_bed | station 1 at (8,8) — **14 tiles / 4.7 m away** | its T after 2 min |
|---|---|---|
| ×1 (pre-session) | never ignites | 1 game (ambient) |
| ×2 | never ignites | 138 game |
| **×8 (tonight's landing)** | **ignites at 7.5 s** | 467 game |

Mechanism is unambiguous — at t = 7 s along the path:

```
x=22 SOLID  T = 1262.7 game (1556 K)   <- the burning crate
x=20..12 gas  T = -6 .. +3 game (287-296 K)   <- the air is COLD
x= 8 SOLID  T =  261.4 game ( 554 K)   <- ignites moments later
```

The air carries nothing; the two solids are hot. **This is radiation.**
T⁴ means ~600 K → ~1550 K multiplies radiant power ~44×, and the raycaster's
documented ~2-tile reach becomes 14+.

Consequences: it breaks the `fire_tuning` level's premise (stations spaced
≥6 tiles to be independent — so our own future measurements there would be
contaminated), and in a furnished deck one crate would light everything in
line of sight within seconds. Note also 1556 K overshoots G1's 1300 K target
in the transient.

## 5. Branch / gate state — deliberately red

- **Golden NOT re-baselined** (Erik's call: "way too early to produce new
  goldens", and he is right — the iron rule budgets ONE deliberate rebase per
  approved change-set, and this change-set is not finished). So
  `test_canonical_scenario_matches_sanctioned_golden` is **RED on the branch
  by design**, and that redness is the marker that this is not landed.
- **When it is time**, the move is fully diagnosed already: the new value is
  `167b96bddfe37c0d256afed4d3b9271371fcaf3edd7557e02cf685a17208953f`, and
  **R4 is its sole cause** — re-running with the R4 code but the PRE-tuning
  config.toml yields the identical hash, i.e. the fire dials contribute
  nothing (the canonical scenario's ghost fire is non-flammable, so it never
  reaches a burn site). Pleasant consequence: **the golden is now independent
  of the fire dials**, so the rest of Phase 4 cannot move it.
- **Suite: 2329 passed, 4 failed.** Three are pre-existing/parked (2×
  `test_cool_shift_axis`, `test_bench_two_room`). One is new and honest:
  `test_eos_p4_combustion::test_e2e_1_sealed_room_fire_self_starves` asserts
  a starving fire leaves fuel "barely touched", but `wall_damage 0.36` (25×
  the old value) takes 60 hp → 25.8. If §6 keeps the duration ruling, that
  test's expectation needs rewriting; if §6 changes the dials, it may heal
  itself. **Second signal that 0.36 is aggressive.**
- CUDA: R4's twin is written but **not built or gated tonight**.

## 6. NEXT SESSION — the radiation question (Erik's hypothesis, confirmed in source)

Erik: *"I think radiation is not decreasing with distance as it is now —
radiation only decreases with ray density."* **The source says exactly that**:

- `cpp/src/raycaster.h:557` — "there is NO per-ray distance falloff; the 1/r
  intensity law emerges from ray density (N cancels)."
- `raycaster.h:514` — survival is reduced "ONLY by occlusion (never by
  distance)".

So heat falls as **1/r** (2D ray density) and clear air absorbs none of it.
Real thermal radiation from a compact source falls as **1/r²**. At 14 tiles
that is a ~14× overestimate — the right order of magnitude to explain §4
entirely.

The design history matters here: the `dist_atten` term was **deliberately
dropped** in the raycaster redesign because it double-counted with ray
density. For *light* in a faithful-2D model that is correct. The open
question is whether it is correct for **heat from a fire**, which is a
compact 3D emitter seen in a 2D slice. Candidate framings for the session:

1. **Is 2D-faithful the right model for heat?** Light and heat currently
   share one falloff law by construction. They need not.
2. **Air absorption for heat.** Currently zero in clear air. Real air +
   soot/CO₂/H₂O does attenuate. A physical extinction term would shorten
   reach without touching the geometric law.
3. **What sets the reach we WANT?** Design target first (how far should a
   burning crate ignite fuel, and after how long?), then fit — rather than
   discovering the reach from whatever the dials produce.
4. Erik also noted **conduction may be ~0** on several materials
   (furniture κ = 0), so radiation is doing essentially all the transport.
   Worth checking the balance between the two channels.

Only after that: revisit H_bed (§4 may dissolve on its own once reach is
right), the duration ruling (§3.4), and the sealed-room test (§5).

## 7. Other open items from tonight

- **#61 wood can't sustain fire** (filed today): conductivity 0.15 drains
  ignition heat with a ~3 s e-fold; furniture (κ = 0) is why every bench
  before tonight missed it. Interacts with §6 item 4.
- **The relight**: at the landed dials the crate fades ~150 s, **re-ignites
  off its own residual heat ~165 s**, then dies at 182 s. Needs a ruling —
  this is the G6/G7 ember / auto-reignite / hysteresis territory, still
  undesigned. Erik has not yet said whether it reads as a natural last flare
  or as a glitch.
- **Dev key `I` fixed** (`src/debug_keys.py`): it set `fire = 1.0` but left
  the tile at ambient, and since R3 tied O2 demand to `hotf`, a cold seed
  draws no oxygen, deposits no heat and fades in 60 s **without ever burning
  its fuel** — i.e. the key was dead on arrival for anyone testing fire
  in-game. It now heats its patch to each material's own ignition point +
  `DEBUG_IGNITE_MARGIN` (55) like the igniter that "pressed" it, and seeds at
  `ignition_seed` so the in-game curve is the one the lab plots. Temperature
  write is `thermal_solid`-only (the gas-mirror rule). Note this is the
  general truth now: **a fire cannot bootstrap from cold — heat must come
  from outside.** Physically right, and the in-engine ignition path already
  respected it.
- **O2 > ambient anomaly**: local flame-zone X climbs to 0.2226 (> 0.21)
  starting ~18 min AFTER the fire dies, still rising at 40 min. Room
  *inventory* X is conserved exactly (the #54 books are clean), and it is not
  differential diffusion (both bulk gases are conservative-transport, D = 0).
  ~1% level, unexplained. Investigation offered, not started.
- **Smoke from a lone crate is ~invisible by design**: soot = 0.5 × consumed
  O2 ≈ 0.5 units over the crate's whole life, spread over ~2800 tiles. Flag
  for the fire-looks work, not a bug.
- **Spread/wind is the big untested area**: `k_wind_fan = 0.5` still carries
  its "NEEDS TUNING vs the live wind scale" flag, `k_wind_strip = 0` is
  parked, and before tonight spread had never been measured in this session.
