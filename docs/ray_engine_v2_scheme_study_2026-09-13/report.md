# Spatial-scheme bake-off — results (2026-09-13)

> Erik asked: *"is it possible to test the alternatives in a simple python
> render, only rendering one frame, to see how the different methods compare to
> each other?"* This is the answer. `scheme_study.py`, one frame, numpy only,
> ~80 s. Figures: `scheme_fields.png`, `scheme_profiles.png`.
>
> It was run to check a claim I made in
> `docs/ray_engine_v2_design_2026-09-12.md` §2.4 and did not believe: that step
> differencing's numerical diffusion is "comfortably below the tile
> quantisation we already accept". **It is not.** The claim is withdrawn.

## The setup

All schemes share the S16 angular quadrature, a 49×49 grid and the 0.333 m
tile. The comparison is built so the two error sources separate:

| | angular error | spatial error | conservative |
|---|---|---|---|
| **exact** — analytic `E/(2πr)` with Bresenham visibility | none | none | n/a (yardstick) |
| **long characteristics** — back-integrate each ordinate to the grid edge | yes | ~none | no |
| **step** — first-order upwind finite volume (what §2.4 specifies) | yes | yes | yes |
| **shear** — advance one cell on the major axis, interpolate the transverse fraction | yes | yes | **not as implemented** |

A normalisation check runs first, because the first version of this study was
wrong and the check is what caught it: every scheme must land on the analytic
`E/(2π) = 0.1592` for `mean ring fluence × r`, and must be flat in `r`. All
four now do, to three digits.

## Result 1 — step differencing does not merely soften shadows, it removes them

90 % → 10 % edge width of a shadow cast by a wall with a 7-tile gap, measured
12 tiles downstream:

| scheme | edge width (tiles) | (metres) |
|---|---|---|
| exact | 0.00 | 0.00 |
| long characteristics S16 | 0.00 | 0.00 |
| **step S16** | **11.00** | **3.66** |
| **shear S16** | **0.00** | **0.00** |

Eleven tiles of penumbra from a seven-tile aperture is not a soft shadow, it is
the absence of one. My own `sqrt(distance)` estimate had predicted ~1.7 tiles
and was optimistic by more than 6×. A doorway would cast essentially no heat
shadow.

This is the study's headline, and it is a *different* finding from critique 1's
§6, which measured shadows through **solid walls** (2.8 % leak — genuinely
fine) and characterised apertures as "soft but not wrong". Through an aperture,
at gameplay distances, it is wrong.

## Result 2 — the near field is anisotropic, and rotation makes it worse

Ring ripple (peak-to-peak as a percentage of the mean; 0 % is isotropic):

| scheme | r = 3 | r = 8 |
|---|---|---|
| exact | 2.4 % | 0.5 % |
| long char S16 | 129.2 % | 281.7 % |
| **long char S16 + rotation** | **8.3 %** | **13.2 %** |
| step S16 | 85.8 % | 45.0 % |
| step S16 + rotation | 101.3 % | 107.5 % |
| **shear S16** | **50.7 %** | **33.7 %** |
| shear S16 + rotation | 69.9 % | 90.8 % |

Two things here, and the second contradicts the design.

**Rotation helps long characteristics enormously (129 % → 8.3 %) and hurts both
grid schemes.** The reason is visible in the numbers: with the half-offset
quadrature no ordinate sits exactly on an axis, and an axis-aligned ordinate in
a grid scheme is a pencil beam that never spreads. Rotation sweeps the
quadrature *through* the axis-aligned case and samples the pathology. Critique 1
found the same thing from the other side — its best near-field row is "S16,
half-offset (no ordinate on an axis or a 45°)".

**So §2.5's per-tick quadrature rotation is wrong for the scheme §2.4 picks.**
Rotation is a remedy for the ray effect, which is an *angular* artifact; the
near-field error here is a *stencil* artifact and rotation cannot touch it.
Critique 1 reached the same conclusion independently by measuring S64 (1.96×
axis/diagonal at r = 1, versus S16's 2.13× — essentially unmoved).

## Result 3 — conservation, and the honest problem with the candidate

| scheme | emitted | absorbed + escaped | residual |
|---|---|---|---|
| step S16 | 1.00000 | 1.00000 | **+0.0000 %** |
| shear S16 | 1.00000 | 0.73796 | **+26.2 %** |

Step differencing is exactly conservative, as designed and as critique 1 also
measured. **My shear implementation is not**, and 26 % is far too large to
dismiss as bookkeeping noise.

The transport operator itself is fine — the interpolation weights are
non-negative and sum to one, so a column's total stream is preserved. The leak
is in the *energy* accounting: absorption is charged as `κ·I·w` using the
cell-centre intensity, while the intensity is actually advanced along a path of
length `1/|major|` that varies with direction, so what is charged and what is
removed from the stream disagree. Critique 1 reports an integer shear variant
with an exactly-zero residual, using a remainder-based split; reconciling the
two is the open engineering question, not a settled win.

**Stated plainly: the sharp scheme is not yet known to be conservative in our
hands, and exact conservation is a hard requirement. That is the gap.**

## Result 4 — more ordinates make the sharp scheme worse

| quadrature | ripple r = 3 | ripple r = 8 | shadow edge (tiles) |
|---|---|---|---|
| shear S16 | 50.7 % | 33.7 % | **0.00** |
| shear S32 | 65.4 % | 70.5 % | 11.00 |
| shear S64 | 68.7 % | 84.7 % | 10.00 |
| step S16 (reference) | 85.8 % | 45.0 % | 11.00 |

Counter-intuitive and worth understanding before anyone "improves" the design by
raising the ordinate count. A shear step is *exact* when the transverse fraction
is 0 (axis) or 1 (45°) — no interpolation happens at all. Angles in between
interpolate, and interpolation is what diffuses. S16's sixteen directions happen
to sit close to the exact cases; S32 and S64 add intermediate angles whose
fractions sit near 0.5, where transverse diffusion is maximal.

So for this scheme the ordinate count is **not** a quality dial, and §12 Q2
("ordinate count", pencilled for measurement in P2) partly dissolves: S16 is not
a budget compromise, it is near a sweet spot.

## What this changes in the design

1. **§2.4's diffusivity claim is withdrawn.** Step differencing costs the
   shadows, and heat shadows through doorways are gameplay.
2. **§2.5's per-tick quadrature rotation should not ship with a grid scheme.**
   It is a ray-effect remedy applied to a stencil artifact, and measured, it
   makes both grid schemes worse. Keep the half-offset quadrature instead, so no
   ordinate is ever axis-aligned.
3. **The spatial scheme is now the arc's central open problem**, not a detail
   inside P1. We need one that is simultaneously sharp and exactly conservative.
   Candidates, in order of promise: the integer shear variant critique 1
   measured at zero residual (reconcile against this study's 26 %); the lumped
   Linear Characteristic family, which satisfies corner balance and so is
   conservative and low-diffusion by construction, at the cost of carrying
   moments per cell.
4. **S16 stands**, for a better reason than budget.

## Caveats

- Everything here is float, not the integer scheme. It measures the *scheme's*
  behaviour, not the quantisation. Critique 1 covers the integer side.
- A single-cell emitter is the harshest possible isotropy test. A real fire is a
  cluster, which averages several sources and will look better than these
  numbers.
- The `exact` shadow is hard-edged because the source is one cell. A physical
  extended source has a real penumbra, so "0.00 tiles" is the right answer *for
  this scene*, not a universal target.
- Conservation is measured on the shadow scene only.
