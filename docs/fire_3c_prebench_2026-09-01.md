# Fire session #12, Phase 3c pre-benches — infinite-fuel O2-wall probe + cluster coupling

> **Status: MEASUREMENT MEMO.** No sim code or `config.toml` values were
> touched. Both benches are runtime CFG/instrument-side only: Bench 1 pins
> `wall_hp` back to full every tick via a debug field write (an allowed
> measurement instrument, not a mechanic change); Bench 2 uses stock dials
> throughout. Raw per-tick CSVs and JSON summaries live under
> `tests/_phase3a_artifacts/` (untracked, reproducible from the appendix
> commands). Runs §3's first two rows of
> `docs/fire_3c_design_brief_2026-09-01.md`, informing item 1 (die-term
> review), item 5 (o2f vacuum amendment / the 0.13 gate), item 6 (the M4
> config-algebra rewrite), and the die-mechanic §1.3/G11 cluster question.

## Findings first

**Bench 1 — the O2 wall does not exist under current dials, at any fuel
level, sealed or open.** Even with `wall_hp` pinned to full every tick
(infinite fuel) in a sealed 12×12-interior hull chamber, the fire still
self-extinguishes — at **t=1772.1 s (29.5 min)**, ~56% longer than the same
scenario with real fuel depletion (3a's M7 furniture-sealed run died at
1131.2 s) — and the cause is unambiguously the **T-gate**, not O2: `hot`
(the `clamp01((T−fire_T_ext)/fire_T_span)` gate) sits pinned at 1.0 through
t=1200 s, then collapses to exactly **0.0 by t=1740 s** and stays there,
while `X_local` (flame-neighbour O2 mole fraction) never falls below
**~0.165** (its minimum, reached around t=900–1200 s) and has actually
**risen back to 0.1756 at the moment of death** — a full **35% above** the
`o2_frac_ext=0.13` extinction gate. **X plainly asymptotes well above 0.13:
infinite fuel never starves this fire of oxygen, in a sealed box, at any
dial of this scenario.** The open sky-fed M1 control (same infinite-fuel
pin, open ambient boundary) never dies at all inside its 60-minute cap and
settles into a **genuine flat plateau** — I≈0.75, X≈0.189, T≈460.5 game
(753.5 K) — the first true fixed point measured in this arc. Re-testing the
M4 equilibrium algebra (`config.toml` ~:855) against this genuine plateau
(where the constant-I assumption unambiguously DOES hold, unlike 3a's
transient windows) still overshoots by **~110×** — the SAME order of
overshoot 3a found in a transient window. **This is the headline surprise:
3a's explanation ("the constant-I assumption never holds") is wrong, or at
least incomplete** — the algebra overshoots just as badly when I really is
constant for a full hour, so item 6's fix needs to be a structural
re-derivation of the formula, not merely a transient-regime disclaimer.

**Bench 2 — mutual radiation measurably raises cluster temperature and
extends life, at the cost of lower peak per-tile intensity.** At t=1645 s
(the single crate's death instant), the isolated crate is extinct
(I≈0.02, T=76.6 game) while the 1×2 pair is still burning at I=0.085
(T=227.4 game, **+150.7 game / +2.9× hotter**) and the 2×2 block is
thriving at I=0.155 (T=323.1 game, **+246.5 game / +4.2× hotter**). Both
clusters outlive the single crate's 1645 s death and the 1800 s bench cap
without dying (censored — see limitations). Peak instantaneous hottest-tile
T rises modestly and monotonically with cluster size (single 453.7 → pair
468.3 → 2×2 481.6 game, +3.2%/+6.2%), but the SUSTAINED/plateau gap is much
larger (331.1 → 349.0 → 398.9 game, +5.4%/+20.5%) — mutual radiation is a
sustain effect more than a peak effect. Peak per-tile burning intensity
*falls* with cluster size (single 0.696 → pair 0.614 → 2×2 0.504,
−11.8%/−27.6%), consistent with clustered tiles sharing a locally-depleted
O2 pool even as they keep each other hot.

---

## Bench 1 — infinite fuel, sealed vs open

### Mechanism recap (for readers of this memo alone)

`grow = k_grow·avail·hot·I(1−I)·(1+k_wind_fan·W)`,
`die = k_die·(1−avail·hot)·I + k_wind_strip·W(1−I)·I`, `avail = F·o2f`,
`F = wall_hp/hp_mat`, `o2f = clamp01((X−o2_frac_ext)/(X_amb−o2_frac_ext))`,
`hot = clamp01((T−fire_T_ext)/fire_T_span)`. Pinning `wall_hp` every tick
forces `F ≡ 1` for the life of the run, so `avail ≡ o2f` — this isolates
whether death comes from `o2f→0` (X approaching the 0.13 gate) or from
`hot→0` (T falling below the material's extinction floor, 80 game /
373 K for furniture) with fuel and (for the open leg) O2 both freely
available.

### Sealed leg (M7-run-2 shape: 20×20 hull box, ~12×12 interior, furniture crate, infinite fuel)

Cap 4200 s (70 min, generous); **run did NOT hit the cap** — died at
t=1772.125 s.

| t (s) | I | T (game / K) | X_local | hot |
|---|---|---|---|---|
| 0.04 | 0.120 | 275.0 / 568.0 | 0.2100 | 1.00 |
| 60 | 0.499 | 395.7 / 688.7 | 0.1924 | 1.00 |
| 90 | 0.597 (near-peak) | 403.7 / 696.7 | 0.1828 | 1.00 |
| 300 | 0.452 | 320.8 / 613.8 | 0.1701 | 1.00 |
| 900 | 0.359 | 267.3 / 560.3 | **0.1650 (X minimum)** | 1.00 |
| 1200 | 0.342 | 260.1 / 553.1 | 0.1656 | 1.00 |
| 1500 | 0.169 | 179.9 / 472.9 | 0.1659 | 0.555 |
| 1600 | 0.117 | 152.6 / 445.6 | 0.1700 | 0.403 |
| 1700 | 0.057 | 101.6 / 394.6 | 0.1719 | 0.120 |
| **1740** | 0.032 | 70.7 / 363.7 | 0.1746 | **0.000** |
| **1772.1 (death)** | 0 (snapped, `I_min`) | 51.3 / 344.3 | **0.1756** | 0.000 |

Peak I=0.601 (t≈90–95 s, near-peak plateau in the 0.34–0.45 range through
t=300–1200 s). X_local drops from ambient 0.21 to a minimum of ~0.165 by
t≈900–1200 s (the fastest-burning window), then **rises back** to 0.176 as
I collapses and local O2 demand falls faster than the sealed room can
re-deplete it — the room's finite O2 reservoir is never remotely close to
exhausted; local O2 recovers as the fire dies rather than causing the
death. `hot` stays pinned at 1.0 through t=1200 s, then falls smoothly to
exactly 0 by t=1740 s and stays there — the fire's `I` decays through
`I_min=0.02` on pure `k_die·(1−0)·I` (the O2 term contributes nothing to
the death, since `hot=0` already zeroes the whole `avail·hot` product) and
snaps out 32 s later. **Cause of death: T-gate, unambiguously — the O2 term
never gets a chance to matter because hot reaches 0 first, with 35% margin
still left in X before the O2 gate.**

Comparison to 3a's real-fuel furniture-sealed run (M7 supplementary,
`m7_sealed_furniture.csv`): infinite fuel extends life from 1131.2 s to
1772.1 s (**+56.6%**) — pinning `F=1` measurably delays the collapse (fuel
erosion was accelerating the T-gate death, consistent with `avail=F·o2f`
feeding `grow`), but does **not prevent** it: the SAME T-gate mechanism
still kills the fire, just later. `X_at_death` is 0.1756 here vs 0.184 for
the real-fuel run — both far above the 0.13 gate; the small difference is
noise-level (different death time, different local O2 history), not a
systematic effect worth reading into.

### Open M1-control leg (infinite fuel, sky-fed, natural wind)

Cap 3600 s (60 min); **run hit the cap — never died.** This is the closest
thing to a true fixed point measured in this arc: I, T, and X are all flat
across the full hour (checkpoints every 300 s, full table in
`b1_open_inf_fuel_summary.json`):

| metric | value (game / K where applicable) | range across the hour |
|---|---|---|
| I (steady) | 0.7514 (median), peak 0.7655 | 0.740 – 0.761 |
| T (steady) | 460.5 / 753.5 | 455.9 – 465.0 / 748.9 – 758.0 |
| X_local (steady) | 0.1892 | 0.1883 – 0.1906 |

`hot=1.0` throughout (T never remotely threatens the extinction floor with
sky-fed O2 + infinite fuel); the fire simply burns at a genuine steady
state, exactly the "burns forever at some steady I" outcome the brief
predicted.

### M4 equilibrium algebra re-test (the genuine-plateau regime)

Stock dials (unchanged, confirmed live): `H_bed = H_BED_M·2^H_BED_SHIFT =
18125·16 = 290000`, `cool_shift=13` (furniture), `heat_inv_shift =
log2(thermal_mass=8) = 3`, `burn_rate=0.02`, `dt=1/24`, `n_faces=4`
(measured `nbrs`).

Using THIS run's own measured plateau (I=0.75140, X=0.18923):

```
o2f    = (0.18923 − 0.13) / 0.87                    = 0.06808
B      = burn_rate · dt · I · o2f · n_faces
       = 0.02 · (1/24) · 0.75140 · 0.06808 · 4       = 1.7053e-4
T*_pred = H_bed · B · 2^(cool_shift − heat_inv_shift)
       = 290000 · 1.7053e-4 · 2^10 (=1024)          ≈ 50,639 game
         (clamps at T_MAX_PHYS=16000 game / ~16,293 K if the deposit
          path is reached that far)

T*_measured (this run's genuine hour-long plateau)  = 460.5 game (753.5 K)
```

**Ratio: predicted / measured ≈ 110×** — statistically the SAME overshoot
3a found (≈105×) in a transient window where I was still decaying. **This
is the pre-bench's headline surprise for item 6**: 3a's read ("the algebra
is a correct fixed point of the ODE in the abstract, but the dial regime
[constant I long enough to converge] doesn't occur") does not survive this
test — here I unambiguously IS constant, for a full simulated hour, and the
naive algebra still overshoots by two orders of magnitude. The formula
itself (not just the "when does it apply" caveat) needs re-derivation —
most likely it predates the net-T⁴ radiation model (config.toml's own
comments flag the derivation as tied to an older `cool_shift=9` era) and is
missing a loss term the current model actually pays, or the
`2^(cool_shift−heat_inv_shift)` factor no longer means what the comment
says it means. Phase 3c/4 should NOT ship a "regime note" fix alone for
item 6 — the arithmetic itself is off by ~2 orders of magnitude even at
its own best-case regime.

---

## Bench 2 — cluster coupling (1 vs 1×2 vs 2×2, M1 open scenario, stock dials)

Same scenario/dials as 3a's M1 (interior 84×40, tile 0.333 m, natural wind,
sky_tau_s=60, sponge_width=8); origin at M1's crate_xy=(12,21), cluster
tiles adjacent to it. Cap 1800 s (30 min, matches 3a's M1 cap). "Burning
tiles" mean = per-tile mean I over tiles with I>0.05 (0 if none burning).

| variant | peak I (burning-tile mean) | ramp (s) | peak T hottest (game/K) | plateau T hottest (game/K) | plateau T cluster-mean (game/K) | death (s) | fuel unburned at death/cap |
|---|---|---|---|---|---|---|---|
| single | 0.696 | 80.3 | 453.7 / 746.7 | 331.1 / 624.1 | 331.1 / 624.1 | **1645.3** | 26.7% |
| 1×2 pair | 0.614 | 75.5 | 468.3 / 761.3 | 349.0 / 642.0 | 348.7 / 641.7 | **>1800 (censored)** | 27.8% (at cap) |
| 2×2 block | 0.504 | 68.2 | 481.6 / 774.6 | 398.9 / 691.9 | 389.0 / 682.0 | **>1800 (censored)** | 31.9% (at cap) |

(Single-crate numbers reproduce 3a's M1 exactly — peak I=0.696 at
t=122.0 s, ramp 80.3 s, peak T=453.7 game, death at 1645.3 s, 26.7% fuel
unburned — confirming this new cluster instrument is a faithful
generalisation of the existing bench, not a parallel measurement.)

**At matched wall-clock time (t=1645 s, the single crate's death instant)**
the coupling is starkest — this is the cleanest single number for "how much
does mutual radiation help":

| variant | I at t=1645s | T hottest at t=1645s (game/K) | fuel remaining |
|---|---|---|---|
| single | 0.020 (extinguishing) | 76.6 / 369.6 | 26.7% |
| 1×2 pair | 0.085 (**4.2× single's I**) | 227.4 / 520.4 (**+150.7 game**) | 28.9% |
| 2×2 block | 0.155 (**7.7× single's I**) | 323.1 / 616.1 (**+246.5 game**) | 34.8% |

**Reading.** Mutual net-T⁴ radiation between burning tiles is real and
substantial: at the moment the lone crate has already gone extinct, the
paired and blocked crates are still burning at meaningfully higher
intensity and are hundreds of game-degrees hotter, both in the hottest
tile and (nearly identically, since the cluster is thermally near-uniform)
the cluster mean. The effect is a **sustain** effect more than a **peak**
one: peak instantaneous T only rises 3–6% with cluster size, but the
*plateau* T (the decay-window median, which is where the radiative
feedback has time to accumulate) rises 5–21%, and life is extended past
the single crate's 1645 s death for both clusters (neither died inside the
30-minute cap). The flip side: peak per-tile burning intensity *falls*
monotonically with cluster size (0.696 → 0.614 → 0.504) — clustered tiles
are drawing on a shared, locally-depleted O2 pool even while radiating heat
into each other, so the "hot" gate improves while the "avail" gate
degrades slightly; net effect is still a longer, hotter burn, but growth is
not simply additive.

---

## Instrument surprises / notes

- **The sealed chamber runs far faster than the open arena.** The
  open-arena legs (M1's 86×42 full grid) run at the previously-measured
  ~9× realtime, but the small 20×20-tile sealed box (400 tiles vs ~3,612)
  finished its full 1772 simulated seconds in well under a minute of wall
  time — an order of magnitude faster per sim-second. Grid size, not
  scenario complexity, is the dominant cost driver — worth remembering when
  budgeting future sealed-vs-open bench pairs.
- **X_local is non-monotonic in the sealed leg**: it falls to a minimum
  around the peak-burning window (t≈900–1200 s) then *recovers* as I
  decays — the room's O2 reservoir is never remotely stressed. A reader
  expecting "O2 falls, then the fire dies from that fall" will be
  surprised; the actual shape is fall-then-recover, uncorrelated with the
  death event 500+ seconds later.
- **1×2 and 2×2 death times are right-censored at the 1800 s cap** — this
  bench answers "is there coupling, and roughly how big" cleanly, but does
  not give exact cluster death/burnout times. A follow-up with a longer cap
  (e.g. 3600–5400 s) would be needed for exact burnout comparisons; not run
  here to keep this pre-bench pass bounded (the coupling direction and
  rough magnitude are already unambiguous from the matched-time and
  fuel-remaining numbers).
- No run in this pass hit an unintended tick cap; the only capped runs are
  the open M1-control (intentional, by design — proves the plateau) and
  the two larger cluster variants (intentional censoring, noted above).
- The M4 re-test's ~110× overshoot at a genuine plateau is a stronger
  negative result than 3a's — worth carrying directly into item 6 of the
  3c design session as "the formula needs re-derivation, not a caveat."

## Instrument extension (what changed in the driver)

`tests/_phase3a_driver.py` gained (no `tools/fire_timing_harness.py`
changes — the harness's `run_one` has no per-tick hook, so both new benches
run their own tick loop, reusing the harness's `build_level`, gate-math, and
X_local definitions verbatim):

- `_sealed_box_level()` — factored out of the existing `run_m7_sealed_generic`
  (behavior-preserving refactor; same 20×20 tilemap) so Bench 1's sealed leg
  reuses the exact M7-run-2 scenario shape.
- `_run_infinite_fuel()` — the Bench 1 core loop: pins `wall_hp` to the
  crate's full value both before and after `sim.step()` every tick (the
  debug field write), records I/T/X/hot every tick, and derives death cause
  (T-gate vs O2-gate), a late-window "plateau" reading, and an X-tail slope
  for trajectory-shape diagnosis. `run_b1()` calls it once for the sealed
  box, once for the open M1 arena.
- `_build_cluster_level()` / `run_cluster()` / `run_b2()` — the Bench 2
  cluster scenario builder + custom loop tracking per-tile I/T/hp
  aggregates (burning-tile-mean I, hottest-tile T, cluster-mean T,
  cluster-mean fuel fraction) across an arbitrary tile offset list.
- New CLI flags `--b1` / `--b2` (+ `--b1-sealed-max-seconds`,
  `--b1-open-max-seconds`, `--b2-max-seconds`), not added to `--all` (kept
  opt-in, like the existing `--m7-furniture`, so 3a's own `--all`
  reproduction command is unaffected).

---

## Appendix — exact commands + artifacts

All runs use `C:/Users/steen/anaconda3/python.exe`, repo root
`C:\Users\steen\projects\breach`, stock `config.toml` on disk (no dial
edits; Bench 1's only non-stock behaviour is the `wall_hp` pin, an
instrument write). Raw CSVs/JSON: `tests/_phase3a_artifacts/` (untracked).

- **Bench 1** (sealed + open-control infinite-fuel legs):
  `python tests/_phase3a_driver.py --b1 --b1-sealed-max-seconds 4200 --b1-open-max-seconds 3600`
  → `b1_sealed_inf_fuel.csv` / `b1_sealed_inf_fuel_summary.json`,
    `b1_open_inf_fuel.csv` / `b1_open_inf_fuel_summary.json`.
  Sealed leg did not hit its 4200 s cap (died at 1772.1 s); open leg hit its
  3600 s cap by design (proves the plateau).
- **Bench 2** (cluster coupling, three variants):
  `python tests/_phase3a_driver.py --b2 --b2-max-seconds 1800`
  → `b2_single.csv`, `b2_pair_1x2.csv`, `b2_block_2x2.csv` + matching
    `*_summary.json` + `b2_summary.json` (all three rows). 1×2/2×2 hit their
    1800 s cap by design (censored, see limitations above); single crate did
    not (died at 1645.3 s, matching 3a's M1 exactly).
- Both benches reuse `COMMON` (interior 84×40, tile 0.333 m, crate_xy
  (12,21)) and the harness's own `build_level`/gate-math for the open legs;
  the sealed leg reuses `run_m7_sealed_generic`'s exact tilemap via the
  newly-factored `_sealed_box_level()`.
