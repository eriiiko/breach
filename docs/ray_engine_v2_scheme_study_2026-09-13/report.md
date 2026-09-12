# Spatial-scheme bake-off — results (2026-09-13)

> Erik asked: *"is it possible to test the alternatives in a simple python
> render, only rendering one frame, to see how the different methods compare to
> each other?"* This is the answer. `scheme_study.py` (isotropy, conservation)
> and `aperture_study.py` (shadow fidelity), one frame each, numpy only.
> Figures: `scheme_fields.png`, `scheme_profiles.png`, `aperture_fidelity.png`.

## ⚠ Correction to the first pass — read this first

The first version of this report led with **"step differencing does not merely
soften shadows, it removes them — 11 tiles of penumbra"**. **That conclusion is
WITHDRAWN. It was my measuring stick failing, not the scheme.**

I had measured a 90 %→10 % *edge width*. The characteristic schemes at S16 do
not produce soft-edged plateaus through an aperture — they produce **pencil
beams**, two narrow spikes where an ordinate happens to line up with the gap and
nothing in between. An edge-width metric crosses both thresholds inside one tile
of a spike and reports `0.00`, which I read as "perfect shadow". It was the
opposite: the worst failure in the study.

Looking at the plotted profile is what caught it. The corrected measurement asks
the question that actually matters — *does the scheme reproduce the lit region* —
and the ranking changes substantially. Lesson worth keeping: a scalar summary of
a shape needs the shape looked at once before it is trusted.

## The setup

All schemes share the S16 quadrature, a 49×49 grid, the 0.333 m tile. Built so
the two error sources separate:

| | angular error | spatial error | conservative |
|---|---|---|---|
| **exact** — analytic `E/(2πr)` with Bresenham visibility | none | none | n/a (yardstick) |
| **long characteristics** — back-integrate each ordinate to the grid edge | yes | ~none | no |
| **step** — first-order upwind finite volume (what §2.4 specifies) | yes | yes | yes |
| **shear** — advance one cell on the major axis, interpolate transversally | yes | yes | **not as implemented** |

A normalisation check runs first, because the *first* version of this study was
wrong and the check is what caught that one: every scheme must land on the
analytic `E/(2π) = 0.1592` for `mean ring fluence × r` and be flat in `r`. All
four now do, to three digits.

## Result 1 — aperture fidelity: no grid scheme is good, and rotation is what matters

A 7-tile gap in a wall, cut 12 tiles downstream. `fill` is the mean fluence
inside the true lit band relative to exact (1.0 ideal), `spill` the mean outside
it (0.0 ideal), `RMS err` the normalised error against the analytic profile.

| scheme | fill | spill | RMS err |
|---|---|---|---|
| *ideal* | 1.00 | 0.00 | 0.000 |
| **long char S16 + rotation** | **1.01** | **0.00** | **0.102** |
| shear S16 + rotation | 0.92 | 0.06 | 0.201 |
| step S16 + rotation | 0.92 | 0.10 | 0.228 |
| step S16 | 0.97 | 0.09 | 0.244 |
| shear S16 | 1.03 | 0.02 | 0.313 |
| **long char S16 (no rotation)** | 1.17 | 0.00 | **1.627** |

Read `aperture_fidelity.png` alongside it. Unrotated long characteristics spikes
to **8.6× the correct peak** at two angles and sits at zero everywhere else —
the fill of 1.17 is an average over a profile that is nowhere near right, which
is why RMS is the honest column. Rotation collapses that to a near-perfect
match.

**Rotation is the single most important variable here**, worth more than the
choice of spatial scheme: it takes long characteristics from the worst result in
the study to the best by a factor of two.

Both grid schemes land around RMS 0.2, under-filling the aperture and bleeding
~10 % outside it. Not catastrophic, not good.

## Result 2 — point-source isotropy, where rotation does the opposite

Ring ripple, peak-to-peak as a percentage of the mean (0 % is isotropic):

| scheme | r = 3 | r = 8 |
|---|---|---|
| exact | 2.4 % | 0.5 % |
| long char S16 | 129.2 % | 281.7 % |
| **long char S16 + rotation** | **8.3 %** | **13.2 %** |
| step S16 | 85.8 % | 45.0 % |
| step S16 + rotation | 101.3 % | 107.5 % |
| shear S16 | 50.7 % | 33.7 % |
| shear S16 + rotation | 69.9 % | 90.8 % |

**Rotation helps long characteristics enormously and hurts both grid schemes.**
The mechanism is visible in the numbers: with the half-offset quadrature no
ordinate sits exactly on an axis, and an axis-aligned ordinate in a grid scheme
is a pencil that never spreads. Rotation sweeps the quadrature *through* the
axis-aligned case and samples the pathology. Critique 1 found the same thing
from the other direction — its best near-field row is "S16, half-offset (no
ordinate on an axis or a 45°)".

So rotation is **good for apertures and bad for point sources** in a grid
scheme, and the design cannot have both from one lever. Critique 1's independent
S64 measurement (near-field 1.96× versus S16's 2.13×, essentially unmoved)
says the near-field error is a *stencil* artifact that no angular remedy
touches.

## Result 3 — conservation, and the honest problem with the sharp schemes

| scheme | emitted | absorbed + escaped | residual |
|---|---|---|---|
| step S16 | 1.00000 | 1.00000 | **+0.0000 %** |
| shear S16 | 1.00000 | 0.73796 | **+26.2 %** |

Step differencing is exactly conservative, as designed and as critique 1 also
measured. **My shear implementation is not.**

The transport operator is fine — interpolation weights are non-negative and sum
to one, so a column's total stream is preserved. The leak is in the *energy*
accounting: absorption is charged as `κ·I·w` from the cell-centre intensity,
while the intensity is advanced along a path of length `1/|major|` that varies
with direction, so what is charged and what is removed disagree. Critique 1
reports an integer shear variant at exactly zero residual using a
remainder-based split; reconciling the two is open work, not a settled win.

Long characteristics is not conservative either, and its unrotated drift is
visible in the normalisation check (`0.1704 / 0.1495 / 0.1293` against a flat
`0.1592`).

**Stated plainly: the two schemes that win on accuracy are the two that are not
yet known to conserve in our hands, and exact conservation is a hard
requirement.**

## Result 4 — more ordinates make the shear scheme worse

| quadrature | ripple r = 3 | ripple r = 8 |
|---|---|---|
| shear S16 | 50.7 % | 33.7 % |
| shear S32 | 65.4 % | 70.5 % |
| shear S64 | 68.7 % | 84.7 % |

Counter-intuitive, and worth knowing before anyone "improves" the design by
raising the ordinate count. A shear step is *exact* when the transverse fraction
is 0 (axis-aligned) or 1 (45°) — no interpolation at all. Angles in between
interpolate, and interpolation is what diffuses. S16's directions sit close to
the exact cases; S32 and S64 add intermediate angles whose fractions cluster
near 0.5, where transverse diffusion peaks.

So for this scheme the ordinate count is **not** a quality dial. S16 is near a
sweet spot rather than a budget compromise.

## What this changes in the design

1. **§2.4's step-differencing choice is not refuted.** My first pass claimed it
   was; that claim is withdrawn. Step is the only candidate measured to conserve
   exactly, and its aperture error (RMS 0.244) is comparable to shear's.
2. **§2.5's per-tick rotation is now a genuine trade, not a free win.** It
   improves apertures for every scheme and degrades point-source isotropy for
   grid schemes. The design asserts it as an unambiguous good; it is not.
3. **The spatial scheme remains the arc's central open problem.** We want sharp,
   isotropic *and* exactly conservative, and nothing measured here is all three.
   Candidates: reconcile critique 1's zero-residual integer shear against this
   study's 26 %; or the lumped Linear Characteristic family, which satisfies
   corner balance and so is conservative and low-diffusion by construction, at
   the cost of moments per cell.
4. **S16 stands**, for a better reason than budget (Result 4).

## Caveats

- Everything here is float, not the integer scheme. It measures the *scheme*,
  not the quantisation. Critique 1 covers the integer side.
- A single-cell emitter is the harshest possible isotropy test; a real fire is a
  cluster of sources and will average better than these numbers.
- Conservation is measured on the shadow scene only.
- `fill` is a poor statistic for a spiky profile (it flatters unrotated long
  characteristics at 1.17). RMS and the plotted profile are the honest columns —
  which is the same lesson as the correction at the top.

---

# Renders (2026-09-13) — "a picture of the actual light rendering"

Erik asked to see it rather than read percentages. `room_render.py` and
`cascade_render.py` put one fire in a 64×64 ship deck — two doorways, a
corridor, a partition gap, an inside corner — and tone-map the field the way a
frame would be shown. Figures: `room_render.png`, `room_doorway.png`,
`cascade_render.png`.

## What the heat schemes look like

`room_render.png`. The artifacts that the ripple percentages describe are
immediately visible, and they are worse to the eye than the numbers suggest:

- **step S16** paints a hard four-pointed **cross** on every source. Those are
  the axis-aligned ordinates, which in a grid scheme are pencils that never
  spread. Rotation does not remove it and slightly sharpens it.
- **shear S16** paints a sixteen-pointed **star** with dark gaps between the
  spokes. Rotation folds it into a cleaner eight-pointed star. Still obvious.
- **exact** is a smooth round glow with a crisp wedge through the partition gap.

Through the corridor into the far room, which only one opening feeds:

| scheme | far-room mean vs exact |
|---|---|
| exact | 1.00 |
| shear S16 | 0.90 |
| shear S16 + rotation | 0.89 |
| step S16 | 0.52 |
| step S16 + rotation | 0.49 |

**Step loses half the energy that should reach the next room.** Its diffusion
spreads the beam into the walls on the way down the corridor, where it is
absorbed. For a game whose fire is supposed to travel through a ship, that is a
gameplay-relevant error, not a cosmetic one — and it is the strongest argument
found so far against step differencing.

## What the light solver looks like

`cascade_render.png`. This is a working 2D radiance-cascades prototype — five
cascades, branching factor 4, the real interval-merge identity — so the design's
claim that cascades do not carry these artifacts can be looked at rather than
trusted.

| scheme | ripple r = 3 | ripple r = 8 |
|---|---|---|
| exact | 2.4 % | 0.5 % |
| **radiance cascades (16 base dirs)** | **15.3 %** | **19.8 %** |
| cascades, 8 base dirs | 19.1 % | 24.4 % |
| cascades, 4 base dirs | 76.0 % | 15.1 % |
| step S16 sweep | 85.8 % | 45.0 % |

**No spokes, at any setting.** The failure mode is completely different: where
the grid sweep produces directional striping, cascades produce a smooth glow
whose error shows up as blotchiness and as **light leaking through walls** into
rooms that should be dark. That leak is the known "bilinear leak" of radiance
cascades, named in the literature with published fixes, and it is the thing to
watch when the GLSL version is built — a game about darkness cannot have light
in sealed rooms.

Base direction count matters a lot at the low end (4 → 8 is the big jump) and
saturates after that.

## What the renders change

1. **Step differencing's real cost is corridor throughput, not shadow softness.**
   Losing 48 % of the energy reaching the next room is a bigger problem than
   anything in the earlier profile measurements, and it is the clearest
   argument yet for a sharper transport step.
2. **The cascade half of the design is de-risked.** It renders cleanly, the
   merge identity works as written, and the artifact to plan for is leakage
   rather than striping.
3. **Two solvers with two different failure modes is a feature**, not a
   redundancy: the sweep's striping is invisible in a heat field that is
   integrated over time and thermal mass, and the cascade's leak is harmless to
   heat because heat does not use cascades.
