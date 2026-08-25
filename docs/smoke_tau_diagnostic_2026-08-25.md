# Smoke tau diagnostic — #6's histogram half (2026-08-25)

Instrument: `tools/smoke_tau_histogram.py` (imports the shipped
`renderer.gas_medium.gas_optical_depth`; dials read live from config).
Data: three real recorder dumps — the two 2026-08-21 HUMAN-TEST sessions
(2400 snapshots each, playground 70x100, fires burning) and the 2026-08-18
velocity-clamp blast seed (775 snapshots). 19.2 M visible-smoke tile-samples
pooled (alpha > 0.05).

## Dials as shipped

`tau = smoke_absorb_scale(1.4) x plume_k_scale(300) x k_soot(0.9033) x rho
     = 379.4 x rho`, curve a=1, b=1 (honest), alpha = 1 - exp(-tau).

The entire visible ramp (alpha 0.05 -> 0.98) therefore spans
**rho in [1.35e-4, 1.03e-2] — 1.02% of the 0..1 density scale.**

## Results

| Metric | Value |
|---|---|
| Visible smoke that is flat black (alpha > 0.98) | **60.3%** pooled |
| alpha > 0.90 / > 0.50 | 70.4% / 89.6% |
| Worst snapshots (both real fire sessions, late) | **99.9–100% flat black** |
| tau p50 / p90 / p99 | 7.5 / 23.7 / 106 |
| rho p50 / p90 / p99 (visible tiles) | 0.021 / 0.060 / 0.27 |
| rho >= 0.5 / >= 0.99 (the 0..1 cap) | 0.024% / **0.000%** |
| max rho seen anywhere | 0.95 |

Median visible tile: tau 7.5 → alpha 0.9994. The soot mass distribution has
~3 decades of genuine structure (1e-3 … 0.3) and **never touches the 0..1
cap** — the structure Erik misses exists in the density field; the render
mapping crushes all of it to alpha ≈ 1.

## Verdict on #6's split question

**The render constants own the saturation, not the sim.** Smoke mass is
healthy (no cap slamming, wide dynamic range); `plume_k_scale = 300` — a
hack calibrated on the molasses-era tiny densities (peak ~0.002) — places
the Beer-Lambert knee at rho ≈ 0.002 while real fire sessions live at
0.02–0.3. No sim change is needed; digests never move.

## Retune menu (render-only, all four dials are LIVE sliders in lighting_demo)

Anchor values derived from the measured distribution:

- `plume_k_scale ~ 52` puts alpha=0.98 at rho p90 (0.060).
- `plume_k_scale ~ 12` puts alpha=0.98 at rho p99 (0.27).
- e.g. k=30: median tile alpha 0.55 (translucent, structured) instead of 0.9994.
- Optional: `tau_curve_b ~ 0.4–0.5` (the designed-for tau-space remap,
  b<1 = compress) makes alpha discriminate across the full 3-decade mass
  range — tau 1 -> 0.63, tau 100 -> ~0.96 at b=0.45 — interior structure
  stays readable everywhere; retune `tau_curve_a` so wisps stay thin.
- Starting point for the taste pass: `plume_k_scale ~ 30, b ~ 0.45, a ~ 1`,
  then Erik's eye decides. (Taste half runs AFTER #48's ambient drift per
  the standing order.)

## Noted, parked

- Beams vs body: the ray march absorbs at the base scale only
  (~1.26 x rho) while the plume body renders at 379 x rho — a 300x
  inconsistency the plume dial papers over. Lowering plume_k_scale shrinks
  it. The honest fix is defining what rho = 1 physically MEANS (the P-S2
  density-scale question) — out of scope for the retune.
- Dumps carry the SMOKE plane only; steam (k_s 0.10) omitted — negligible
  for the flat-black question.

Regenerate: `conda run -n data python tools/smoke_tau_histogram.py <dumps...>`.
