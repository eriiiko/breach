# Velocity-clamp audit — 2026-08-19

**The arc-opening question (Erik): does the clamp fire at all on the spike
path?** That decides bug-versus-dial before anything is designed.

**Answer: it fires constantly — and leaks anyway. Two distinct defects, both
confirmed by measurement against the seed dump. This is a BUG, twice over, not
a tuning item.**

Evidence: `debug_manual_20260818_194038_velocity_clamp_seed.npz` (775 snapshots,
70×100, recorded on the fixed mass-books build — no mint confounding it).
Analysis scripts ran the solver's own cap formula
(`c_local = c_amb·√((T_max_open+290)/290)`, `c_amb = 300`, `s_eos = 1`)
against every snapshot's recorded `wind_x/wind_y`.

## Defect 1 — CONFIRMED: the "local" cap is a global scalar (the TODO's lead)

`eos_solver.cpp:401-427` (CUDA twin `cuda_eos_step.cu:170-181`): `t_max_abs_raw`
is an unweighted MAX over every open cell on the map, folded once per tick into
a single `c_local_q`, which the step-4 kick then applies as **every** cell's
ceiling (`eos_solver.cpp:770`, `cuda_kick_compression.cu:192`). The "local" in
the name is a lie, exactly as the lead suspected.

Measured consequence: the session's fires held T_max ≈ 720–740, so the global
ceiling sat at ~555–565 m/s all through the blast window — while an
ambient-temperature cell's true sound speed is 310. Result:

- **52,865 cell-snapshots supersonic vs their own cell's local c**, present in
  255/775 snapshots, up to 506 cells in a single snapshot.
- Cool gas (T ≈ 3 at the spike cells) legally carries Mach 1.8 flow.

The clamp is genuinely *firing* on this path: ~2,045 cell-snapshots sit pinned
within 1% of the global cap (mean ~8 cells/snap through the active window,
254 snapshots) — scale-to-cap is engaging, against the wrong number.

## Defect 2 — NEW: the Chebyshev pre-test leaks diagonal flow past the cap

The magnitude clamp is guarded by a component pre-test —

```c
if ((ax > u_cap_q) || (ay > u_cap_q)) { ... sqrt ... scale-to-cap ... }
```

(`eos_solver.cpp:793`, again at `eos_solver.cpp:1669`, CUDA twin
`cuda_kick_compression.cu:201`). A cell whose components are each ≤ cap skips
the magnitude check entirely — but its magnitude can reach √2·cap. The comment
calls it "the cheap Chebyshev pre-test avoids a sqrt per cell"; as written it is
not a conservative filter, it is a hole.

Measured, against the reconstructed per-snapshot global cap:

- **4,545 cell-snapshot violations of even the GLOBAL cap** across 254/775
  snapshots.
- **90% (4,074) have both components ≤ cap** — the leak, exactly.
- Median violating flow angle is **45.0°** (pure diagonal); the angle histogram
  peaks hard in the 45–60° bin.
- Max |u|/cap over the whole run = **1.3754 < √2 = 1.4142**. The session peak
  |u| = 773 m/s is 562 (that snap's cap) × 1.375 — the leak's envelope, not a
  third mechanism.
- The residual 10% (471 cells) exceed the cap by ≤ 7.3% on a component —
  consistent with tick-entry-vs-tick-end T_max drift in the reconstruction
  (the solver folds the cap from tick-entry T; the dump records tick-end).
  Nothing exceeds 1.1×cap on a component: **no unexplained mechanism remains.**

The two defects compose: global cap 565 (1.82× an ambient cell's true c) ×
diagonal leak (up to 1.414×) = up to Mach 2.57 possible in ambient-T gas.
Measured max own-cell Mach: **2.47**.

## The spike mechanism, seen at the events

The 100–280× pile-up cells are **near-vacuum pockets adjacent to Mach-1.8
flow**. Snap 615→616: target cell N = 0.11 → 280.9 in one tick, surrounded by
|u| ≈ 546–598 against a 559 global cap and own-cell Mach 1.8–1.9. Snap 616→617:
N = 0.00 → 278.4, same picture. Transport is slamming mass into empty cells at
speeds advection cannot resolve at the substep count `n_sub` was derived for —
the TODO's "three symptoms, one cause" framing holds.

## The two related threads

- **Map-edge clustering (x≈95): NOT a boundary artefact.** Supersonic
  violations span x ∈ [3, 96], median x = 72, with large clusters at x 0–9,
  50–89 — wherever the fighting was, not the ring. The worst *pressure* cells
  clustering at x≈95 reflects where walls got broken, not a BC defect. A
  bigger map is not needed to separate this.
- **The aquarium report (water leaving a sealed box): untested here.** Water
  rides the same u field, so shock-driven supersonic wind is a *plausible*
  carrier, but nothing in this dump exercises water. The TODO's ordered checks
  (seal verify, conservation) stand unchanged.

## Verdict for scoping

Both defects are mechanical, small-surface fixes in the same few lines
(CPU kick ×2 + CUDA twin — the kick/compression kernel, which is also where
`test_cuda_p64_kick_compression` PART 2 diverges):

1. Pre-test fix: compare `rad = ux²+uy²` against `u_cap²` directly — no sqrt
   needed until a clamp actually engages, so it stays exactly as cheap and the
   filter becomes exact.
2. Cap fix (Erik's ruling 2026-08-19: squares against squares, NO per-cell
   sqrt): `cap²_cell = min(c_amb²·t_abs_cell/t_amb, U_MAX²)` — the √(T/T_amb)
   in the sound-speed formula squares away, so the per-cell cap is one multiply
   and one divide, int64-safe (cap² raw ≈ 1.4e15 ≪ 2⁶³). Only cells that
   actually clamp pay a sqrt (the rescale factor `cap/|u|`), exactly as the
   engage branch does today. Net cost vs today: REMOVES the per-tick global
   sqrt from the cap path; the global `c_local` (one sqrt/tick) survives solely
   as `u_est`'s ceiling for the `n_sub` derivation, where a global max is the
   correct shape.

**Scope ruling (Erik, 2026-08-19): quadratic drag (TODO item 3) is NOT part of
this patch.** With the sqrt-free cap there is no shared machinery; the drag law
stays linear at `k_drag = 0.5`, and item 3 keeps its own design session and its
try-with-and-without evaluation.

Gates already named by the TODO: supersonic |u| (now measurable per-cell),
negative P_min, transient ≫100× cells — plus the golden re-baseline debt this
arc inherits (re-baseline once, at close, with rationale).

Note for the design session: the same global reduction also steers `n_sub`
(`eos_solver.cpp:485-492`). For the *substep count* a global max is the right
shape (CFL is set by the worst cell) — fixing the velocity ceiling must NOT
naively localize `n_sub`'s input. The two consumers of `t_max_abs_raw` want
different things and should be separated deliberately.
