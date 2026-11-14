# Fire session #12, Phase 3a — July re-measurement on the healed engine

> **Status: MEASUREMENT MEMO.** No sim code or config.toml values were
> touched for this pass; every dial variation below is a runtime CFG patch
> through `tools/fire_timing_harness.py`'s own `apply_overrides`/`run_one`
> seam. Raw per-tick CSVs and JSON summaries live under
> `tests/_phase3a_artifacts/` (untracked, reproducible from the commands in
> the appendix — not committed). Companion reads:
> `docs/fire_mechanics_inventory_2026-08-31.md` (§2 mechanism, §4 the July
> claims table this re-measures, §1.1 goals G1–G4) and
> `docs/fire_g12_one_map_patch_2026-08-31.md` (the honest Kelvin map,
> `K = 293 + T_game`, already live on this branch — confirmed in
> `config.toml` before any run: `k_temp_to_kelvin=1.0`, `kelvin_ambient=293`,
> `rad_scale=5.1427e-5`).

## (a) Engine state

- Branch `fire-12`, commit `f0b6a7e0c474c36138002f4ea681afc040f795ad`
  ("feat(#12,#53): Phase 2 part B -- author the fire-tuning level via
  level_lib"). G12 (the one-map patch) is already on this commit.
- Map: `K = 293 + T_game` (slope 1, `k_temp_to_kelvin=1.0`,
  `eos_t_amb_k=293.0`) — the ONLY frame; every Kelvin figure below is
  `293 + T_game` via `src/temperature_scale.py`'s `load().to_kelvin()`.
- Build: CPU `cpp/build/Release` (current).
- Machine: `DESKTOP-0E98HUV`. Python: `C:/Users/steen/anaconda3/python.exe`
  3.11.7 (no conda `data` env on this box, per the harness docstring's own
  caveat).
- Instruments: `tools/fire_timing_harness.py` (single-crate, M1/M5/M7-support
  supplementary), `tests/_fire_bench.py` (far-field, M2, run AS-IS
  unmodified), and a thin driver `tests/_phase3a_driver.py` that only calls
  into the harness's `run_one`/`apply_overrides` and adds one small
  self-contained sealed-box scenario builder for M7 (documented in §(f) and
  the instrument-limitation note below) — no parallel physics, no config
  edits.

## (b) The §4 July-claims table, re-measured

| July claim | Today's measurement | Verdict |
|---|---|---|
| Blessed hp=25 shape (ramp → I 0.40 @ ~4 min → out ~14 min) at `k_fire_heat=1600` → 14764 K plateau | `k_fire_heat` is deleted (P-R4); mechanism is now H_bed + net-T⁴ radiation. Stock crate (furniture, hp=30): peak I=0.696 at t=122 s, near-peak plateau I≈0.67 / T≈445 game (738 K), self-extinguishes (I→0, NOT hp→0) at t=1645 s (27.4 min) with **26.7% fuel unburned**. Shape is qualitatively similar (ramp → high I → long tail → out) but every number changed | **VOID-CONFIRMED-GONE** (mechanism verdict) / new baseline established |
| Pass-2 floor: below ~1834 K the fire goes marginal (`k_fire_heat` dials) | Mechanism deleted; not applicable | **VOID** (as predicted in the inventory) |
| Far-field room-T rise ~200 game vs target ≤20 | `tests/_fire_bench.py` as-is, 10 s crate-stack burn: sealed box far-field ΔT=**+3.73** game, bunker R6 ΔT=+0.01, pen R8 ΔT=+0.01, arena mirror mean ΔT=+1.26. All ≤ target (≤20) by a wide margin | **IMPROVED** (>50× reduction from July; #54's gas-energy fix is the cause, matches the inventory's prediction) |
| "Cool AND vigorous flame needs a MODEL change (decouple sustain heat from displayed T)" | See M6 below: I and T* are coupled but WEAKLY and sub-linearly in the actually-reached transient regime (peak T only rose 275→521 game, ~1.9×, while peak I rose 0.12→0.95, ~8×, across the k_grow sweep) — a genuinely cool-but-vigorous point is not obviously reachable by k_grow alone under current H_bed/cool_shift, though the "coupling forces hot-means-intense" framing does NOT hold in the strong form July implied | **STILL-A-WALL, but the shape of the wall changed** — decouple question re-opens with new numbers, not July's |
| k_grow/k_die knife-edge (only 1600 sustains; 800 stalls) | 5-point sweep (M5, k_die held stock): k_grow=0.125 STALLS (peak I=0.120, snaps at 321 s with 96.6% fuel unburned); k_grow ∈ {0.25, 0.5, 1.0, 2.0} all SUSTAIN, with peak I and burn fraction increasing monotonically (0.499→0.951 peak I; 46.9%→22.4% fuel unburned by the 1200 s cap) | **VOID** — no knife-edge; graceful monotonic response across a 16× k_grow range, only the lowest point (4× below stock) fails |
| Fire NOT O2-limited locally (X stays 0.184–0.210) | Confirmed directly: over the full 1645 s stock run, `X_local` (flame-neighbour O2 mole fraction) stayed in **0.183–0.210** the entire time, including through the self-extinction event — the `hot` gate (T-vs-fire_T_ext), not the O2 gate, drives the death | **HOLDS** |

## (c) G1–G4 scorecard

| Goal | Target | Measured (stock dials, furniture crate) | Gap |
|---|---|---|---|
| G1 — plateau ~1300 K | order-of-magnitude anchor | Near-peak plateau T≈445 game = **738 K**; peak instantaneous T=454 game = **747 K** (k_grow sweep's hottest point, k_grow=2.0, still only 521 game = 814 K) | **~1.6–1.8× too COLD** (opposite direction from July's "too hot") |
| G2 — ramp 30–120 s | ignition → 90% of peak | Stock (k_grow=0.5): **80.3 s** — inside target | **HOLDS at stock dial** |
| G3 — burnout 5–10 min, fuel-governed | | No true burnout inside 1800 s (30 min) cap; fire self-extinguishes (I→0, not hp→0) at **27.4 min**, driven by the T-gate collapsing, not fuel exhaustion — **26.7% fuel unburned** at snap-out | **STILL-A-WALL**: ~3–5× too long AND wrong governing mechanism (heat, not fuel) — though a real improvement over July's 76% unburned |
| G4 — nominal I≈0.2 with headroom | | Near-peak I≈0.67 (stock); full-window median I≈0.40 (biased down by the long decay tail) — both well above 0.2 | **STILL-A-WALL**: current dials run ~2–3× hotter in intensity than the nominal target |

## (d) M4 — equilibrium algebra check

Formula (config.toml ~:855): `T* = H_bed · B · 2^(cool_shift − heat_inv_shift)`,
`B = burn_rate·dt·I·o2f·n_faces`, `o2f = clamp01((X − 0.13)/0.87)`.

Shipped dials for the furniture crate: `H_bed = H_BED_M·2^H_BED_SHIFT =
18125·16 = 290000`; `cool_shift = 13`; `heat_inv_shift = log2(thermal_mass) =
log2(8) = 3`; `burn_rate = 0.02`; `dt = 1/24`; `n_faces = 4` (measured,
matches `nbrs` in every run).

Measured plateau (stock run, t=90–250 s window, before fuel-availability
erosion dominates): I=0.670, X_local=0.1912 → o2f=(0.1912−0.13)/0.87=0.0703.

```
B      = 0.02 * (1/24) * 0.670 * 0.0703 * 4        = 1.571e-4
T*_pred = 290000 * 1.571e-4 * 2^(13-3)             = 290000 * 1.571e-4 * 1024
        ≈ 46,650 game  (≈ 46,940 K nominal — would clamp at T_MAX_PHYS=16000
          game / 16,293 K if the deposit path is reached that far)

T*_measured (same window)                          = 445 game = 738 K
```

**Ratio: predicted / measured ≈ 105×.** The naive fixed-point algebra
massively overshoots what is actually observed. Reading: the formula assumes
a *constant* B sustained long enough for `cool_shift`'s 341 s e-fold to
converge to its fixed point. The real system never does this — I peaks at
~120 s and then *decays* (the k_grow sweep's peak-T column confirms this:
raising k_grow 16× only moved peak T ~1.9×, nowhere near the ~8× swing in
peak I), driven by the fuel-availability feedback (`avail = F·o2f`,
`F = wall_hp/hp_mat` falling as `wall_damage` erodes hp). T is therefore
governed by the *coupled I–F–T transient dynamics*, not by the simple
H_bed/cool_shift balance the config comment derives — the algebra is a
correct fixed point of the ODE in the abstract, but the dial regime that
would need to hold to reach it (a long, high, roughly-constant I on
undepleted fuel) doesn't occur under current dials.

## (e) M6 — the decouple question's input

Is `(I≈0.2, T≈1300 K)` reachable by dial choice, or does the mechanism force
hot-means-intense? **Reachable in direction, not in magnitude, by `k_grow`
alone.** The M5 sweep shows I and peak-T are positively but *sub-linearly*
coupled: k_grow 0.125→2.0 (16×) moves peak I 0.12→0.95 (~8×) but peak T only
275→521 game (~1.9×, 568→814 K). Even the most aggressive sweep point
(k_grow=2.0, I saturating near 1.0) reaches only 814 K — the gap to G1's
1300 K anchor (≈1.6×) is *larger* than the T gain bought by doubling k_grow
again would plausibly deliver, since the T response is clearly saturating,
not linear, in this transient regime. Conversely G4's I≈0.2 point sits
*below* every sustaining k_grow value measured (the closest, k_grow=0.125,
stalls) — so hitting a low nominal I on today's k_grow/k_die pair alone
looks like it undershoots the sustain floor before it undershoots the
G1 target. Net: the July "decouple sustain heat from displayed T" question
survives, but its shape changed — the fix is not "I and T are the same knob
in disguise" (M4/M5 refute a strong linear coupling), it's that **k_grow is
the wrong knob for reaching G1's magnitude at all** — H_bed and/or
cool_shift (which directly set the *asymptote* B convergence targets, not
just the ramp) are the levers Phase 4 needs to move to close the ~1.6–1.8×
T gap, while a slower/lower-I sustain regime (G4) is a separate,
probably-compatible retune of k_grow/k_die/I_cap_per_avail once the
fuel-availability collapse (G3/G11) is addressed so I has time to actually
sit near a target value instead of decaying from a transient peak.

## (f) M7 — sealed-chamber O2 death

Two runs, both sealed (fully hull-walled, no opening — vacuum outside via
`boundary != "ambient"`, confirmed no sky/ambient refill):

1. **Station 4 of `levels/fire_tuning`** (as-authored, `MAT_WOOD` crate at
   `(9,31)`): fire never really ramps past the ignition seed (peak I=0.124,
   STALL by the harness's own 1.3×-seed threshold) and self-extinguishes at
   **t=214.2 s**, with `X_local`=0.203 at death (essentially unmoved from
   ambient 0.21) and **98.9% fuel unburned**. Death is a fast T-gate
   collapse, not O2 starvation.
   - **Instrument-limitation / confound found**: station 4's crate material
     is `wood` (`conductivity=0.15`), unlike M1's reference `furniture`
     crate (`conductivity=0.0`) — wood has an *extra* heat-loss channel
     (conduction into the surrounding sealed air) that furniture lacks, on
     top of the shared `cool_shift` relaxation. This alone can explain the
     much faster, much weaker burn, independent of the sealed-vs-open
     question M7 is trying to isolate.
2. **Supplementary furniture-in-sealed-box** (a thin harness-variant scene
   in the driver, same material as M1, 14×14-tile sealed hull chamber):
   sustains much better (peak I=0.581) and self-extinguishes at
   **t=1131.2 s (18.9 min)**, `X_at_death`=**0.184**, **60.4% fuel
   unburned**. `X_at_death` is still far above the O2-extinction gate
   (`o2_frac_ext=0.13`).

**Conclusion**: neither run's proximate death cause is O2 starvation under
current dials — both die via the same heat/fuel-availability collapse seen
in the open-room M1 bench (T falls below `fire_T_ext` as `avail` erodes with
`wall_hp`), well before local O2 mole fraction approaches the 0.13
extinction gate. The sealed room DOES shorten fire life versus the open,
sky-fed M1 bench (furniture-sealed snaps at 1131 s/60% fuel left vs. M1's
1645 s/26.7% fuel left) — a real, measurable effect of the finite O2
reservoir feeding into `o2f` — but it is a *quantitative* dampening of the
same collapse mechanism, not the qualitatively distinct "hit the O2 wall"
death G5 describes. **G5's sealed-room ordering is not confirmed as the
actual death cause on this engine build** — worth carrying into Phase 3c's
full die-mechanic review (§1.3 G11).

## Instrument limitations hit this pass

- `fire_timing_harness.run_one` has no direct "plateau I" output — only
  `peak_I` and a `steady_T` computed over the whole
  `[time_to_peak, fuel-out-or-snap]` window, which for these long
  slowly-decaying runs is dominated by the multi-hundred-second decay tail
  and reads well below the true near-peak quasi-steady value. The driver
  recomputes a windowed `I_plateau`/`X_plateau` (`tests/_phase3a_driver.py`
  `_plateau_window`/`_metrics_summary`) mirroring the harness's own window
  logic rather than adding a second metric to the shared harness.
- No fire in this pass reached a genuinely flat plateau at the stock dials —
  every sustaining run is a continuous ramp-then-decay curve (fuel
  availability erodes `avail` over minutes), so "plateau I"/"plateau T" in
  this memo are near-peak quasi-steady windows (t=90–250 s for the stock
  run), not a true fixed point — flagged wherever used (M4, scorecard).
- `levels/fire_tuning` station 4's crate material (wood) is not the same
  material `fire_timing_harness` uses as the G1–G4 reference fire
  (furniture) — see M7's confound note. The supplementary sealed-box
  furniture scenario (a small in-driver scene builder, not a new committed
  level) closes this gap for this pass; a Phase-3c/4 ruling should decide
  whether station 4 should ship a furniture crate instead of wood, or
  whether both are intentionally different reference points.
- No k_fire_heat-map or ×3-Kelvin assumption was found hardcoded anywhere in
  the harness or `_fire_bench.py` — both read dials/materials through
  `CFG`/`config.toml` live, so the G12 map is picked up automatically; no
  workaround via `temperature_scale` was needed beyond using it to convert
  reported game-units to honest Kelvin for this memo.

## (g) Appendix — exact commands + dial sets

All runs use `C:/Users/steen/anaconda3/python.exe`, repo root
`C:\Users\steen\projects\breach`, stock `config.toml` on disk (dials patched
at runtime only). Raw CSVs/JSON: `tests/_phase3a_artifacts/`.

- **M1** (reference single crate, stock dials):
  `python tests/_phase3a_driver.py --m1 --m1-max-seconds 1800`
  → `m1_reference.csv`, `m1_summary.json`. Scenario: furniture crate, interior
  84×40, crate at (12,21), tile 0.333 m, natural wind, sky_tau_s=60,
  sponge_width=8 (all `fire_timing_harness` defaults).
- **M2** (far-field, unmodified): `python tests/_fire_bench.py` →
  `m2_fire_bench.log`. `playground` level, crate stack ignited at (26,41),
  10 s run, vents/ducts stripped (script's own setup).
- **M3**: read from M1's `X_local` column directly (no separate run).
- **M4**: algebra computed from M1's t=90–250 s window (see §(d)); no
  additional run.
- **M5** (k_grow sweep, k_die held stock 0.008):
  `python tests/_phase3a_driver.py --m5 --m5-max-seconds 1200`
  → `m5_kgrow_{0.125,0.25,0.5,1.0,2.0}.csv`, `m5_summary.json`. Same scenario
  as M1, only `physics.fire.k_grow` overridden per point.
- **M6**: derived from M1 + M5 data (see §(e)); no additional run.
- **M7**: `python tests/_phase3a_driver.py --m7 --m7-max-seconds 5400`
  (station 4 of `levels/fire_tuning`, wood crate at (9,31), ignition seed +
  T=300 written directly to `gmap.fire`/`gmap.temperature`, mirroring the
  harness's own seeding convention) → `m7_sealed.csv`, `m7_summary.json`;
  and `python tests/_phase3a_driver.py --m7-furniture --m7-max-seconds 5400`
  (supplementary sealed 14×14-tile hull box, furniture crate, same seeding)
  → `m7_sealed_furniture.csv`, `m7_summary_furniture.json`.
- Every run cap noted above was sufficient — **no run hit its tick cap**
  except the four sustaining M5 sweep points (0.25/0.5/1.0/2.0), which were
  deliberately capped at 1200 s to bound sweep cost; their reported
  `fuel_unburned_frac` is read AT the cap, not at eventual snap-out (M1's own
  longer 1800 s run shows the stock point, k_grow=0.5, continuing to decay
  smoothly past the 1200 s mark to snap out at 1645 s).
