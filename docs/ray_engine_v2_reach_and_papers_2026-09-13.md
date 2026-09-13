# Ray engine v2 — the reach lever, the papers read in full, and two study cases (2026-09-13)

> **Status:** measurement and reading session. Nothing built, nothing ruled. This
> answers the blocking question in `docs/ray_engine_v2_NEXT_SESSION_2026-09-13.md`
> §3 with an option that was not on its menu, reads the three priority papers in
> full, and delivers Tasks 1 and 2 from its §5.
>
> **Depends on:** `ray_engine_v2_survey_2026-09-12.md` (rulings R-A..R-S) ·
> `ray_engine_v2_design_2026-09-12.md` (the design under review) ·
> `ray_engine_v2_critique_1_physics_2026-09-12.md` (12 required changes) ·
> `ray_engine_v2_scheme_study_2026-09-13/report.md` (the bake-off) ·
> `fire_radiation_assumptions_2026-09-11.md` (the baseline being replaced).
>
> **Instruments**, all in `docs/ray_engine_v2_scheme_study_2026-09-13/`:
> `sweep_ref.py` (a faithful float reference of design §2.3 with critique 1's
> fixes 3/4/5 applied, plus the leak channel and both transport steps) ·
> `reach_and_leak_study.py` (reproduces every table in §§1-5 in one run) ·
> `flashover_study.py` (Task 2) · `real_scene_heat_sweeps.py` + `real_scene_light_cascades.py`
> (Task 1) · `heatmap_<level>.npz` (real fields dumped from the real Simulation).
>
> Every number here was measured, and the instrument is calibrated against
> critique 1: it independently reproduces that document's ignition radius
> (axis 19.4 tiles against their ≈17) and its §7c stability table to three
> figures (287.5 / 941 / 2985 / 3.46e6 game per tick).

---

## BOTTOM LINE

**The blocking question has a fourth answer, and it is the only one a sweep can
actually execute.** Critique 1's option (a), "adopt 1/r² as structural", is not
implementable: a discrete-ordinates sweep has no notion of distance from a
source, because it has no source. The only two laws available to it are in-plane
geometric spreading (1/r, not negotiable) and extinction (exp(−r/L)). The 1/r²
in survey §7.3 is a memory of the *old* per-emitter ray engine, where `r` is
known along the ray; it does not survive the move to a field solve.

The lever that does work is an **out-of-plane leak**: the fraction of in-plane
flux lost per tile through the deck's floor and ceiling, exported to a named
channel rather than deposited in the air. It violates no ruling — air never
warms (R-L), the plateau is untouched (R-E), smoke absorption stays a separate
additive coefficient (R-M), and R-R governs light rather than heat.

**And it is derivable rather than fitted.** Osborne & Sannikov §3.1 gave the
route: standard 2D radiative transfer augments the flatland quadrature with
inclined rays under an assumption of homogeneity along the third axis. Our slab
is bounded, so an inclined ray leaves it after a finite horizontal distance.
Working that through gives `k = ln(1 + r²/a²)/(2r)` with `a = h/2`, which for a
2.5 m deck is **k ≈ 0.10 and nearly constant** — one value tracks the exact slab
law within 6% from r = 1 to r = 12. Measured with that unfitted value, the
ignition radius lands at **4.8 tiles against a target of 4.4**, and the design's
*other* reach point reproduces too (a 3×3 fire reaches 10.3 tiles from its own
edge against a target of ~11). Both ends of §9.3's reach curve, nothing tuned.

Three other things changed, all from reading the papers properly:

- **Per-tick quadrature rotation should be dropped from the heat design.**
  Camminady's rSN rotates *and interpolates the carried angular flux*, which
  adds an angular-diffusion term; we carry no angular flux between ticks, so
  what §2.5 calls rotation is phase jitter plus time-averaging, which is a
  different and weaker thing. The paper's own §7 says rSN "is not directly
  compatible with sweeping". Measured, our jitter makes ring ripple **worse** at
  every radius, with or without the leak.
- **Conservation no longer discriminates between the candidate schemes**, and the
  scheme study's open 26%-versus-0% disagreement is explained and closed.
- **Erik's §4b worry is real, quantified, and the amplification has a mechanism.**
  A 1% unaccounted transport loss costs 8% of compartment surface temperature; 2%
  breaks flashover; 26% is fatal. The multiplier is the number of surface-to-
  surface bounces in the compartment, which is why a compartment is the sensitive
  test and a point source is not.

---

## 1. Why 1/r² is not on the menu

A sweep update is local: a cell takes what arrives on its upwind faces, absorbs
`a·I`, emits `a·E°[T]`, and passes the rest downwind. Nowhere in that is there a
distance from anything. Distance-dependence can therefore only enter as a *rate*
— a per-tile multiplier — and a constant per-tile multiplier is an exponential,
not a power law.

This is not a small print. Survey §9.5 and critique 1's required change 1(a)
both propose 1/r², and critique 1's supporting number (`r_ign = 3.68`) is an
analytic rescaling of a measured near field, not something produced by a sweep.
The confusion is traceable: the old engine cast rays *from emitters*, so `r` was
available and `dist_atten` was a legitimate per-ray factor
(`fire_radiation_assumptions_2026-09-11.md` §4). Field solves gave that up in
exchange for being flat in emitter count, which is the whole point of the arc.

So the menu is really:

| lever | shape | available to a sweep? |
|---|---|---|
| in-plane geometric spreading | 1/r | yes — and unavoidable |
| extinction / leak | 1/r · exp(−k r) | **yes** |
| inverse square | 1/r² | **no** |
| a genuinely 3D slab solve (several z-layers) | 1/r² near, steeper far | yes, at ×n_z cells and a 3D quadrature |
| lower the source plateau | scales the whole curve | yes, but contradicts R-E |

The last row but one is honest and worth recording: a few z-layers would give
the real geometry, and it connects to the already-parked 2.5D multi-layer smoke
idea. It is not affordable on the sim tick today.

## 2. Deriving the leak instead of fitting it

Osborne & Sannikov §3.1: *"we take the model to be infinite and homogeneous
along the axis perpendicular to the two varying axes. To achieve this we augment
the flatland quadrature with a number of inclined rays"* — eight Gauss-Radau
inclinations, each flatland ray replaced by eight with the same in-plane
projection.

Our slab is not homogeneous in z. It has a floor and a ceiling that absorb. An
inclined ray at elevation θ leaves the deck after horizontal distance
`(h/2)/tan θ`. So the fraction of an isotropic emitter's power still inside the
slice at horizontal distance `r` is the fraction of the sphere within
`|θ| < arctan(a/r)`, which is

```
    S(r) = a / sqrt(a² + r²)          a = h/2, in tiles
```

The sweep tracks height-integrated flux per unit horizontal length, which in
pure 2D is `P/(2πr)`; the slab correction is exactly `S(r)`. A constant per-tile
leak gives `exp(−k r)` instead, so the equivalent coefficient is

```
    k(r) = ln(1 + r²/a²) / (2r)
```

**Measured** (`reach_and_leak_study.py` §2):

| deck height | a (tiles) | k at r=1 | r=3 | r=5 | r=8 | r=12 | r=16 |
|---|---|---|---|---|---|---|---|
| 2.2 m | 3.30 | 0.044 | 0.100 | 0.119 | 0.120 | 0.111 | 0.100 |
| **2.5 m** | 3.75 | 0.034 | 0.082 | **0.102** | **0.107** | **0.101** | 0.092 |
| 3.0 m | 4.50 | 0.024 | 0.061 | 0.080 | 0.089 | 0.087 | 0.082 |
| 4.0 m | 6.01 | 0.014 | 0.037 | 0.053 | 0.064 | 0.067 | 0.065 |

`k` is nearly flat across the whole gameplay-relevant range, which is what makes
a single constant legitimate. Against the exact slab law, `k = 0.10` on a 2.5 m
deck is within **6%** from r = 1 to r = 12, and goes conservatively steeper
beyond (0.73× at r = 20) — erring toward a harder horizon, which is the side to
err on.

**Consequence for level authoring:** the leak becomes a per-cell coefficient
that a level's deck height sets. Survey §7.3 anticipated exactly this — *"one
interpretable per-cell coefficient"* — but got the sense backwards: it says
"zero under a sealed ceiling, positive under open sky". A ceiling does not stop
the loss, it *causes* it (the escaping rays are absorbed by ceiling and floor).
The correct statement is **lower ceiling ⇒ larger k**.

## 3. What the leak does to reach

Source held at the measured 1556 K crate plateau, receiver at radiative
equilibrium, S16 half-offset, 81×81, single phase. `r_ign` is where the fluence
crosses the level a receiver equilibrates at 573 K.

| deck | k_leak | step: ring / axis / diag | ax÷di | shear: ring / axis / diag | ax÷di |
|---|---|---|---|---|---|---|
| 2.2 m | 0.110 | 5.17 / 6.79 / 3.23 | 2.11 | 4.63 / 5.69 / 3.71 | 1.53 |
| **2.5 m** | **0.100** | 5.38 / 7.03 / 3.39 | 2.08 | **4.77** / 5.88 / 3.79 | **1.55** |
| 3.0 m | 0.085 | 5.74 / 7.42 / 3.65 | 2.03 | 5.01 / 6.18 / 3.91 | 1.58 |
| 4.0 m | 0.062 | 6.51 / 8.09 / 4.16 | 1.94 | 5.60 / 6.72 / 4.12 | 1.63 |
| *none — the design as written* | 0 | 11.05 / 10.73 / 7.81 | 1.37 | 7.77 / 8.64 / 4.76 | 1.82 |

(Single phase, i.e. rotation dropped per §4b. Earlier drafts of this table were
averaged over eight rotated phases, which changes the axis and diagonal columns
by up to a tile and inflates the axis÷diagonal spread — another way of seeing
that the jitter is not free.)

### 3a. The target is 4.4 tiles, not 3

Critique 1 §3d flagged that §9.3 derives reach from a **flux** threshold
(10–12 kW/m², piloted ignition of wood) while breach ignites on a **temperature**
threshold. Working it through: `ignition_temp` 280 game = 573 K is radiative
balance at **6.11 kW/m²**, against 11 kW/m² for the flux criterion — 1.8× less
flux, so a larger radius. §9.3's 1.1 m / 3.3 tiles becomes **≈4.4 tiles** under
the engine's own criterion. The §6 bench must state which criterion it uses;
this document uses the engine's, because that is what actually lights a tile.

### 3b. Both ends of the reach curve reproduce

§9.3 gives two points: a ~150 kW crate at ~3 tiles and a ~1 MW room fire at
~8 tiles, both under the flux criterion, so ×1.34 under the temperature
criterion: **4.4 and ~11 tiles**. Measured with the derived k = 0.10 and per-tile
power held constant (a bigger fire is genuinely more powerful):

| source | shear, r_ign from centre | from the fire's own edge | target |
|---|---|---|---|
| 1 tile (≈ a crate) | 4.77 | 4.77 | 4.4 |
| 3×3 (≈ a room fire) | 11.25 | 10.25 | ~11 |

Nothing was tuned to land those.

### 3c. Fragility to the plateau

`r_ign ∝ K_s⁴` under pure 1/r is the property `fire_radiation_assumptions` §5
names as the thing to avoid. Ring-mean `r_ign` across plausible plateaus:

| k_leak (shear) | 1100 K | 1300 K | 1556 K | 1800 K | 2200 K |
|---|---|---|---|---|---|
| 0 (as designed) | 1.77 | 3.79 | 7.77 | 14.43 | **∞** |
| 0.10 (derived) | 1.54 | 3.05 | 4.77 | 7.14 | 10.94 |
| *1/r², for comparison* | 1.30 | 1.82 | 2.61 | 3.49 | 5.21 |

The leak is **more** robust than 1/r² would have been, and unlike 1/r² it is
something the solver can do. This matters because critique 1's open question 4
is right that the plateau will move once R-K gives the source a real radiative
loss channel.

### 3d. The leak keeps both hard gates

Measured in `sweep_ref.py`, float (exact in int64 by the same one-integer-applied-
twice idiom):

- **Conservation**: `Σ ΔE_mat + sky_out − sky_in + ceiling = 0` to relative 1e-16
  on randomised grids, with and without the leak, cold sky and ambient sky.
- **Second law**: a uniform field **at ambient** is a per-cell exact fixed point,
  including non-dyadic absorptivities — and a uniform field above ambient loses
  to the ceiling, which is correct.
- **Positivity**: no negative fluence, given `a ≤ 1` (critique 1's required
  change 9 still applies).

**Two ordering rules are load-bearing and belong in the design text**, because
both broke a gate in my hands before I got them right:

1. **The ceiling must radiate back at ambient** (`E_amb·w·k`), or ambient stops
   being a fixed point. This is the same shape as the Pass-3 ambient thermostat
   already ruled on in arc #54 — a deliberate modelling boundary, counted by name.
2. **The leak acts on the stream BEFORE the material interaction.** Applied
   after, the ceiling's return over-credits by exactly `a·k` and gate 2 fails.

## 4. The papers, read in full

### 4a. Osborne & Sannikov 2024 — confirms the split, and supplies two fixes

The conservation statement the README quotes is verbatim and load-bearing:
ringing is *"due to parallax between the different probes of cascade i+1 and
cascade i, leading to non-conservation of energy at locations where cascades
overlap a light source."* Heat cannot use cascades. That is on record now.

**The bilinear fix, stated concretely enough to implement** (§2.5): each radiance
interval in cascade *i* is traced from its usual start position **to the start
position of the associated child cone on each of the four cascade-(i+1) probes**
that are normally bilinearly interpolated; those four distinct intervals are each
merged with their own cascade-(i+1) sample, and only then averaged with the usual
bilinear weights. Cost: **4× the rays on every cascade except the top one.**

**The finding that matters most for us is the diagnosis of leakage**, and it is
not the one the previous session recorded. The authors say light leaking is
*"typically an indication that the penumbra criterion is violated for some
property of the model (e.g. the spacing of cascade 0 probes is too large)"*, and
that it was not an issue for them because they place *"a cascade 0 probe in every
cell"* and their models resolve their own structure, so *"such small features
(one grid cell or smaller) should not occur"*.

**Breach violates that premise by construction: our occluders are one-tile
walls.** So we should expect leakage where DexRT does not, and the two
mitigations are the bilinear fix and sub-tile light resolution — which turns
survey Q9 from a look question into a correctness one: at 2× sub-tile a 1-tile
wall becomes a 2-cell occluder.

Their parameters, for calibration: branching factor 2, six cascades, radiance
interval length **1.5 voxels** and **4 angular samples** on cascade 0, eight
Gauss-Radau inclinations. They did **not** need the bilinear fix. Their timing
table is also a caution for our §4c finding: **cascade 5 alone is 63.6% of
runtime and cascade 4 another 21.4%** — the top cascades are both the expensive
ones and the leaky ones, so `cascade_diagnose.py`'s "at our map sizes the coarse
cascades add leak without adding reach" is a cost argument as well as an accuracy
one.

### 4b. Camminady 2019 — retires per-tick rotation for heat

Design §2.5 claims we "already independently invented" quadrature rotation and
cites the paper's "under a third of the quadrature points". **Both halves of that
are wrong for our solver.**

rSN is a *time-marching* scheme that carries the angular flux across steps and,
after each step, rotates the ordinates and **interpolates the carried solution
onto them**. The interpolation is the whole mechanism: their modified-equation
analysis (their eq. 13) shows the pair of interpolations adds a discrete
`∂²ψ/∂φ²` — angular diffusion, which they also describe as *"artificial
scattering"*. The rotation angle must be proportional to the timestep.

Our sweep is a steady-state solve, re-done from scratch every tick, carrying no
angular flux. There is nothing to interpolate, so there is no angular diffusion
and none of the paper's result transfers. And the authors close the door
themselves (§7):

> *"In an implicit or steady state transport calculation, the ability to perform
> transport sweeps is of key importance. As the method has been presented, it is
> not directly compatible with sweeping because the rotation might actually change
> the face of the boundary from which the solution is propagated. However, one
> could solve a modified equation directly (without rotations)."*

Measured, ring MAX/MIN, single phase versus averaged over 16 rotated phases —
the average is what a receiver actually experiences through its thermal mass:

| scheme | r=1 | r=2 | r=3 | r=5 | r=8 |
|---|---|---|---|---|---|
| step, leak 0.10, 1 phase | 2.94 | 2.43 | 2.14 | 2.12 | 2.09 |
| step, leak 0.10, 16 phases | **3.06** | **2.60** | **2.38** | **2.55** | **2.87** |
| shear, leak 0.10, 1 phase | 1.31 | 1.97 | 1.82 | 1.82 | 1.54 |
| shear, leak 0.10, 16 phases | 1.27 | 2.33 | 2.25 | 2.32 | 2.17 |

**Jitter makes it worse nearly everywhere.** Dropping it also removes a
`tick mod N` dependence from a synced law, which is a determinism simplification.

The constructive half of the paper's §7 remains open: *solve the modified
equation directly*, i.e. add the angular diffusion a rotation would have produced
— a three-term stencil mixing each ordinate with its angular neighbours, exactly
conservative with the remainder trick. It couples the ordinates, so it costs a
second sweep (a source iteration). Worth measuring if the residual ripple proves
objectionable; not needed to decide P1.

### 4c. Davis 2012 — names our trade as fundamental, and validates the shear ordering

Three things, all directly usable:

1. **Short characteristics is explicitly unsuited to our regime.** *"For problems
   where a few gridzones (or point sources) dominate the total emissivity, a short
   characteristics solver may require very high angular resolution... anomalous
   structure (e.g. spokes) in the heating and cooling rates will emanate from the
   dominant sources."* With Osborne's independent statement that ray effects are
   worst in non-scattering media, two production codes name our exact situation.
2. **The sharp-versus-diffuse trade is named, and it cuts both ways.** *"A less
   diffusive scheme allows one to model... shadowing by optically thick material,
   with greater fidelity"*, but at modest angular resolution *"a greater degree of
   diffusion in the intensity can mitigate unphysical effects"*. That is precisely
   step-versus-shear, and it means neither is simply better — which the
   measurements below confirm.
3. **Shallow rays are handled by switching the sweep order** — rows for steep
   rays, columns for shallow ones. Our shear step already orders by the major
   axis, so it is doing the literature's answer; worth citing in the header when
   it is written.

Davis also suggests a hybrid — short characteristics for diffuse emission, long
characteristics for bright point sources. Recording it as considered and
rejected: it reintroduces per-emitter casting, which is the cost scaling
requirement 2 exists to kill.

## 5. The spatial scheme, with three things settled

### 5a. Conservation is no longer a discriminator, and the disagreement is explained

The scheme study measured shear leaking **26%**; critique 1 measured an integer
shear at **exactly zero**. Both were right about their own implementation.

The study's shear advanced the stream with `exp(−κ·path)` over a
direction-dependent path while charging absorption as `κ·I·w` — **two different
absorption laws in one step**. Charged the paired way the design already
specifies (`absorbed = stream·a`, `emitted = src·a`, each computed once and
consumed twice), the residual is **zero to roundoff for both schemes**, and both
pass the uniform-ambient gate exactly.

The direction-dependent path length does not vanish — it becomes a *physics*
bias (diagonals under-attenuated by √2 through semi-transparent media), which is
critique 1's open question 6 and is not a regression, since today's DDA has it too.

**One implementation gotcha worth writing into the design**, because it first
read to me as a scheme defect: a shear step gathers from **two** upwind cells, so
the transverse inlet edge needs the ambient boundary seeded as well as the
major-axis edge. Seed only one and a hole propagates inward and breaks the
uniform-field fixed point.

### 5b. Judge the heat scheme by the ignition footprint, not by how the field looks

This deserves stating plainly because the renders invite the opposite mistake.
**The heat fluence field is never drawn.** What the player sees is which tiles
catch fire, plus the *light* field, which is a different solver. So the heat
scheme's figure of merit is the isotropy of the ignition footprint.

Footprint measured along 64 directions, per-tile power held constant:

| scheme | leak | source | mean r | spread (max r ÷ min r) |
|---|---|---|---|---|
| step | 0.10 | 1 tile | 5.22 | 1.64 |
| step | 0.10 | 3×3 | 9.87 | 1.32 |
| step | 0.10 | 5×5 | 12.82 | 1.25 |
| **shear** | 0.10 | 1 tile | 4.58 | **1.37** |
| **shear** | 0.10 | 3×3 | 11.20 | **1.22** |
| **shear** | 0.10 | 5×5 | 15.05 | **1.17** |

Shear is rounder at every source size, and **the study's untested caveat is
confirmed**: a clustered fire averages the striping down. A single-cell emitter
really is the harshest case.

But on the *rendered* field over a real map, shear is visibly worse: it paints a
16-spoke star while step is smooth (see `real_scene_heat_sweeps.png`). That is Davis's
trade, and it is why the two schemes keep swapping places depending on which
statistic is quoted. Since heat is not rendered, the footprint column is the one
that decides — but the margin is modest, and **the leak does more for both than
the scheme choice does.**

### 5c. Corridor throughput is not a reliable discriminator

The scheme study called step's corridor loss *"the strongest argument found so
far against step differencing"* (step 0.52 of exact, shear 0.90). On a longer,
axis-aligned corridor I measure the ordering **reversed** and both schemes wildly
wrong:

| | far-room energy ÷ analytic |
|---|---|
| step, single phase | 0.12 |
| shear, single phase | 0.04 |
| step, 8 phases | 1.60 |
| shear, 8 phases | 1.50 |

At S16 with a half-offset quadrature the most axial ordinate is 11.25° off-axis,
which over a 28-tile corridor drifts 5.6 tiles — **no ordinate can shoot straight
down an axis-aligned corridor**, and whether one nearly aligns depends on the
phase. The transverse profile is wrong in shape too (a dip in the middle of the
corridor where the analytic answer is flat).

So: long-throw corridor transport is dominated by ordinate alignment, not by the
spatial scheme, and it swings by an order of magnitude. **It should not be used to
choose between schemes, and the earlier conclusion drawn from it should be
withdrawn.** With the leak in, a 28-tile corridor carries `exp(−2.8) ≈ 6%` of
even the correct flux, so the regime where the sweep is least trustworthy is also
the regime that no longer matters for heat. For **light** it matters a great
deal — and cascades do not have this failure mode.

## 6. Task 2 — the flashover study case

`flashover_study.py`. A 2×2 fire held at the measured plateau in a sealed 24×12
compartment lined with furniture, every surface radiating by its own temperature
(R-K), real constants (`rad_scale`, `heat_inv_shift` 3, `cool_shift` 13,
ignition 280 game), 15 minutes of sim time. The sweep is exactly linear in the
emission field, so the transfer matrix is built once per geometry and the run is
a matrix-vector product per tick — exact, not an approximation.

### 6a. The leak does not kill flashover; it un-collapses two phenomena

| k_leak | mean surface T | min | max | surfaces at ignition |
|---|---|---|---|---|
| 0 (as designed) | 507 | 332 | 775 | **100% — FLASHOVER in 30 s** |
| **0.10 (derived)** | 269 | 63 | 684 | **47%** |
| 0.20 | 168 | 5 | 616 | 29% |
| 0.30 | 114 | 0 | 550 | 11% |

Read against survey §9.4, which separates *direct radiant ignition* (seconds,
view-factor limited) from *flashover* (minutes, gas-mediated, R-O): at k = 0
radiation alone lights the entire compartment in half a minute, leaving nothing
for the gas mechanism to do — the two phenomena are collapsed into one, which is
the behaviour Erik called wrong. At the derived k = 0.10 radiation lights the
near half and leaves the far half **pre-heated but not lit** (63–270 game), which
is exactly the state a gas-borne hot layer would then push over.

**So the leak removes *radiative* flashover, which is the bug, and leaves room
for *compartment* flashover, which is the feature.** The honest caveat: R-O is
unbuilt, so this says the radiation channel behaves as designed, not that
flashover will arrive on schedule once the gas channel exists.

Not scheme-dependent: at k = 0.10, step lights 34%, blend 42%, shear 47% — all
partial.

### 6b. §4b made quantitative

An unaccounted transport loss of fraction *f*, no ceiling return:

| f | mean surface T | vs f=0 | surfaces at ignition |
|---|---|---|---|
| 0.000 | 507 | 1.00× | 100% |
| 0.005 | 484 | 0.95× | 100% |
| 0.010 | 466 | **0.92×** | 100% |
| 0.020 | 433 | 0.85× | **87%** |
| 0.030 | 403 | 0.80× | 71% |
| 0.050 | 352 | 0.69× | 55% |

**Erik's guess was right, and the amplification is about 8×**: a 1% transport
loss costs 8% of surface temperature. The mechanism is that a compartment is a
multiply-reflecting cavity — energy makes roughly thirty surface-to-surface
crossings, each losing *f*, and `T ∝ Φ^¼` turns a ~26% flux loss into a ~7%
temperature loss. That is why a compartment is the sensitive test and a point
source is not.

**The tolerable unaccounted leak is ≈1%. Flashover starts breaking at 2%. The
scheme study's 26% would have been fatal.** Erik's reframe — that the requirement
is *accounting*, not exactness — survives, with an order of magnitude of margin
rather than three.

One structural note that falls out: **an accounted leak and an unaccounted leak
of the same size are physically identical** (f = 0.10 gives 250 game against
k = 0.10's 269, the difference being only the ceiling's return radiation). The
distinction is entirely about whether the ledger closes. Which means the two
cannot be chosen separately: the reach lever *is* a transport loss, deliberately
sized, and the only question is whether it is booked.

### 6c. A stability confirmation, end to end

My first run of this case blew up on the first tick, reaching 6.7e5 game units —
because the design as written has no rail on the material update. That is
critique 1's required change 2, reproduced end-to-end in a real compartment
rather than argued from a table. (The blow-up in my own first pass was partly a
missing Q16.16 conversion; with the conversion right, the stability table
reproduces critique 1 exactly, and the *design's* instability at
T ≳ 1800 game stands as they measured it.)

## 7. Task 1 — the solvers on a real heatmap

`real_scene_heat_sweeps.py`, `real_scene_light_cascades.py`, inputs dumped by driving the real
`Simulation` on `levels/playground`: four ignition sites, 60 s of real fire,
**117 burning tiles, 521 tiles above 100 game, peak 977 game (1270 K)**. No
synthetic point source anywhere.

**`real_scene_heat_sweeps.png` — the heat solvers.** Columns: analytic yardstick, step
S16, shear S16, shear + leak 0.10. The yardstick had to be rebuilt once: under
R-K *every* cell with `a > 0` radiates, so a yardstick containing only the fires
is not solving the same problem — with all 783 emitters plus the ambient sky
floor, agreement improves from 0.386 to **0.218** (step) and 0.236 (shear) in
mean |log₁₀ ratio|, i.e. within a factor 1.65 on average across the whole map.

What the picture shows: shear's 16 spokes are obvious on a real multi-source
field; step is smooth; **and the leak column is the one that looks like fire** —
discrete local glows instead of a whole-deck wash. The leak changes the picture
more than the scheme does.

**`real_scene_light_cascades.png` — the light solver, with flashlights.** Same scene,
radiance cascades, flashlights as cone emitters. Three panels: fires only,
fires + flashlights, and the same field at a steeper transfer curve.

- **No spokes at any setting**, on a real map with 326 emitters. The design's
  claim for cascades holds where it matters.
- **Light in rooms that should be dark: 0.4–0.6%** of the brightest region
  (0.0038–0.0059 across the three sealed regions), and adding flashlights barely
  moves it (0.0058 → 0.0059). That is the bilinear leak quantified on a real map.
  Small, but a game about darkness should apply the published fix rather than
  hope tone mapping hides it.
- **The third panel differs from the second only in gamma.** This is the direct
  answer to Erik's *"the dials didn't seem to do anything"*: the transfer curve is
  the lever for the #64 two-state look, not field resolution.

Two of my own configuration errors are worth recording, since both produced
convincing-looking artifacts that were mine and not the solver's: flashlights set
65× brighter than the hottest fire erased the fires from the picture entirely,
and a top-cascade march coarser than a tile stepped straight over one-tile walls
and read exactly like the bilinear leak.

## 8. Corrections to existing documents

| document | claim | correction |
|---|---|---|
| survey §7.3, §9.5 | 1/r² available as a per-cell coefficient; "zero under a sealed ceiling" | Not available to a sweep at all (§1). And a lower ceiling means a *larger* leak, not zero. |
| design §2.5 | "we already independently invented quadrature rotation"; rotation is an unambiguous good | Ours is phase jitter, not rSN; the paper's §7 rules rotation out for sweeps; measured, jitter makes ripple worse (§4b). |
| design §6 | the calibration constant sets reach | It sets the *rate*. Confirmed independently (critique 1 was right). |
| critique 1, required change 1(a) | "adopt 1/r² as structural — measured 3.68 tiles" | The number is an analytic rescaling; a sweep cannot produce it. The leak reaches the same place by a shape the sweep can execute. |
| critique 1 §3d / survey §9.3 | reach target 3 tiles | ≈4.4 tiles under the engine's own temperature ignition criterion (§3a). |
| scheme study, Result 3 | shear leaks 26% | An artifact of mixed absorption bookkeeping; zero in the paired form (§5a). |
| scheme study, renders §1 | corridor throughput is the strongest argument against step | Not a reliable discriminator; ordering reverses with geometry and swings 10× with phase (§5c). |
| scheme study, caveat 2 | "a real fire is a cluster and will average better" — untested | Confirmed and measured (§5b). |
| papers README | the leak is "the thing to watch" | It is *diagnostic of a violated penumbra criterion*, and our one-tile walls violate it by construction (§4a). |

## 9. What is still Erik's to rule

1. **The reach lever.** Does heat get a real, exponential horizon at the price of
   a named out-of-plane export channel charged to the ceiling — a modelling
   boundary of the same kind as the arc-#54 ambient thermostat? Everything in §2
   and §3 is the case for yes, derived rather than fitted. Nothing downstream can
   be calibrated until this is settled.
2. **Deck height as a level property.** If the leak is adopted, `k` follows from
   ceiling height. Is that a per-level constant to start with, or a per-cell
   plane from the outset?
3. Unchanged from critique 1's list, and not addressed here: adjacent-solid
   radiative exchange (its open question 2), `dyn_heat_atten` as a gameplay
   ruling, foliage `heat_atten`, and the 24 Hz tick budget.

Not needing a ruling, but needing writing into the design: drop per-tick rotation
from the heat solver (§4b); the two leak-ordering rules (§3d); the shear boundary
seeding (§5a); and the reach target of 4.4 rather than 3 (§3a).

## Systems

**Existing canonical systems this work must use**, unchanged from the survey and
design: the ray engine (`cpp/src/raycaster.*` + `cuda_raycaster.cu`) as the one
radiation path · the temperature solver's Pass-1 fold · the gas energy seam · the
energy closure identity (a new channel needs a counter and a term in one of the
four groups) · the face-flux energy step as the conservative pattern · the
material table · the Q16 boundary modules · `pack_hover_readout` ·
`tools/fire_tuning_lab.py` as the instrument · GOLDEN_AGGREGATE re-baseline
discipline · **"Starting a fire"** (a fire is started by delivering heat).

**New systems this work proposes**, with draft rules for CLAUDE.md once code exists:

- *The out-of-plane leak channel.* Draft rule: the reach of radiant heat is set
  by ONE per-cell coefficient derived from the deck's ceiling height, applied to
  the stream before the material interaction, with the ceiling radiating back at
  ambient; every count it removes is booked to a named export channel. Never a
  second distance term, never an absorption that warms the air.
- *The float reference sweep* (`docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref.py`).
  Draft rule: the integer solver is checked against this reference, and the
  reference is checked against critique 1's measured numbers; a new transport
  step lands here first and must pass conservation, uniform-ambient and
  positivity before it is written in C++.
- *The compartment flashover bench* (`flashover_study.py`). Draft rule: any
  change to the radiation law reports its flashover-case numbers, because a
  compartment amplifies a transport loss by the number of surface crossings and
  a point-source bench cannot see it.
