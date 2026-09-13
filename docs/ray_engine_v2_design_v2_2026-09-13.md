# Ray engine v2 — design v2 (2026-09-13)

> **Status:** design, ready for the engine-integration critique. Nothing built.
> This supersedes `ray_engine_v2_design_2026-09-12.md` wherever they disagree;
> §0 is the list of disagreements and why.
>
> **Depends on:** `ray_engine_v2_survey_2026-09-12.md` (rulings R-A..R-S) ·
> `ray_engine_v2_critique_1_physics_2026-09-12.md` (12 required changes, all
> resolved here) · `ray_engine_v2_scheme_study_2026-09-13/report.md` (the
> bake-off) · `ray_engine_v2_reach_and_papers_2026-09-13.md` (the reach lever,
> the papers, Tasks 1 & 2) · `fire_radiation_assumptions_2026-09-11.md` (what
> the engine does today) · `docs/papers/README_ray_engine_v2_2026-09-13.md`.
>
> **Instruments** (all measurements below are reproducible from these):
> `docs/ray_engine_v2_scheme_study_2026-09-13/` — `sweep_ref.py`,
> `reach_and_leak_study.py`, `flashover_study.py`, `contact_faces_study.py`,
> `stability_study.py`, `cascade_fix_study.py`, `reflections_study.py`,
> `smoke_study.py`, and the real-scene renders.

---

## 0. What changed since v1, and why

Every row is a measurement or a ruling, not a preference.

| # | v1 said | v2 says | why |
|---|---|---|---|
| 1 | Two solvers: a sweep for heat, radiance cascades for light | **One sweep, two transport steps.** Shear for heat, step for light; one parameter apart | Erik preferred step's look for light on the real-scene renders. Step *is* the sweep, so light collapses into machinery we build anyway, and the render field and the rules field stop being two different algorithms |
| 2 | Reach set by a calibration constant | The constant sets the **rate**, not the reach. Reach is geometry + source temperature | Measured: identical equilibrium at a 100× smaller `rad_scale` (critique 1 §3d, reproduced independently) |
| 3 | 1/r² available as a feel dial | **1/r² is not implementable in a sweep at all** | A sweep has no distance from a source because it has no source. Only 1/r spreading and extinction exist |
| 4 | Per-tick quadrature rotation, "we already invented it" | **Rotation dropped** | Camminady's rSN interpolates a *carried* angular flux; we carry none, and the paper's §7 rules it out for sweeps. Measured, our jitter makes ring ripple worse at every radius |
| 5 | Step differencing, on positivity + diffusion grounds | **Shear for heat** (rounder ignition footprint), **step for light** (smoother field, Erik's call) | Both are exactly conservative in the paired form; the 26% shear leak was a bookkeeping mismatch. So conservation no longer discriminates |
| 6 | "Rule 3 is subsumed — adjacent opaque cells exchange nothing" | **False.** They exchange strongly, and we **keep it**, summed with conduction | Measured 7×–42× the conduction face for wood. Combined conduction–radiation is the standard model and the sum is the textbook form |
| 7 | The flux limiter is deleted; "a sweep cannot diverge" | **Category error.** The limiter guarded the *material* update, which is unstable above ~1800 game | Reproduced end to end: the design as written blows up on the first tick of a real compartment |
| 8 | (no stability scheme) | **Fleck factor (α = 0.5) + a maximum-principle clamp** | Fleck & Cummings 1971 via Wollaber's review. Emission-only damping alone fails catastrophically at plasma; the clamp fixes it exactly |
| 9 | `rad_amb` "unchanged in meaning" | It loses per-emitter attribution; becomes a global/per-exit-cell term | A sweep's escaping stream has no emitter |
| 10 | (no inflow boundary) | **Ambient blackbody inflow**, and for light a **directional sky BC** | Without it the map edge is a 0 K sky and drains every exposed surface at 4.96 game/s |
| 11 | Bodies block heat — open question | **Yes**, absorbing into `rad_flux`; no unit emission yet | Erik's ruling. Blocking without absorbing is not available in a conservative scheme |
| 12 | Smoke absorbs heat (R-M), unscheduled | **Its own patch**, because it makes the sweep a booked writer on the gas energy seam | Gas heat is a ledgered quantity since arc #54 |

---

## 1. What we are building, in one paragraph

Radiation stops being a per-emitter ray cast and becomes a **field solve**, like
pressure and conduction already are. **One** directional sweep visits every cell
once per ordinate and carries several payloads: an integer, exactly conservative
heat channel inside the sim; an integer, coarse light channel for the rules and
for RL; and a float light channel for the renderer. Heat and light differ in
their transport step, their number type and their per-cell coefficients — not in
their machinery. Cost is set by the grid, so reach is free and the number of hot
or bright things does not matter, which is the entire point.

---

## 2. The sweep

### 2.1 The law

The radiative transfer equation, non-scattering (R-L: clear air is transparent;
smoke absorbs but does not scatter in the heat channel):

```
    ŝ · ∇I(x, ŝ)  =  −κ(x) I(x, ŝ)  +  κ(x) B(T(x))
```

Discretise `ŝ` into `N` ordinates and the grid into cells; each ordinate becomes
an independent transport problem.

### 2.2 Why ONE sweep is exact — and exactly what would break it

In a **non-scattering** medium the ordinates do not couple: nothing scatters from
direction `m` into `m'`, so there is no source iteration, no convergence
criterion, no residual. One upwind sweep per ordinate is the exact discrete
solution **of the frozen-source problem** — and the source *is* frozen, because
temperature is fixed for the whole cast (the temperature solver consumes
`rad_net` afterwards) and because light crosses a 40 m compartment in 0.13 µs
against a 41.67 ms tick.

Two things would break it, and both are deliberately excluded from P1:

- **Diffuse reflection** is an isotropic re-emission of what a surface received,
  i.e. a scattering term, and scattering couples the ordinates. Measured: it costs
  one extra full solve per bounce (`reflections_study.py`).
- **Angular diffusion** (Camminady's "solve the modified equation directly")
  couples neighbouring ordinates and costs the same way.

Neither is needed for heat — R-K already gives walls a re-emission channel through
their own temperature, which is the physically dominant path for thermal
wavelengths on dull surfaces. For light, bounce is available later and can be
amortised across frames (§6.4), because the light field is render-only.

**Upwind order** is a topological order of the per-ordinate dependency DAG. For an
ordinate in the (+x, +y) octant a cell depends on its upwind neighbours, so
row-major from the upwind corner works. This matters for the CUDA twin: **shear's
dependency is a whole column** (both of its outputs land in the next column), so a
column is fully parallel; **step's is an anti-diagonal**, the classic KBA skew.
Shear is the friendlier shape.

### 2.3 The integer scheme, and why it conserves exactly

Per ordinate `m`, in upwind order, for cell `i`:

```
    i_in       =  Σ upwind face fluxes
    leaked     =  (i_in * k_leak_i) >> 16            // out-of-plane, if enabled (§5)
    amb_m      =  (E°[0] * w_m) >> 16                // ONE Q16 factor per shift —
    ret        =  (amb_m * k_leak_i) >> 16           //   the same fix as `emitted`
    stream     =  i_in − leaked + ret
    absorbed   =  (stream * a_i) >> 16
    src_i,m    =  (E°[T_i] * w_m) >> 16              // ONE Q16 factor, ONE shift
    emitted    =  (src_i,m * f_i * a_i) >> 32        // f_i = the Fleck factor, §2.8
    i_out      =  stream − absorbed + emitted
    ΔE_mat[i] +=  absorbed − emitted                 // the SAME two integers
    fx         =  (i_out * WX) >> 16
    fy         =  i_out − fx                         // THE REMAINDER, not a second shift
```

**Conservation is structural.** Every integer that leaves the stream is added to
the material as *the same integer*; every integer the material emits is removed
from the stream as the same integer. So

```
    Σ_cells ΔE_mat  +  Σ sky_out  −  Σ sky_in  +  Σ ceiling  ≡  0
```

exactly in int64, by the `eos_solver.cpp::face_flux` idiom: one integer, applied
twice with opposite signs. Verified in `sweep_ref.py` to float roundoff (relative
1e-16) on randomised grids for both transport steps, with and without the leak,
under cold and ambient skies.

Four things in that listing are corrections to v1, each of which leaks or
overflows if omitted (critique 1, required changes 3–5):

- **The remainder face split.** v1 said "summed from its upwind faces" and never
  said how `i_out` is divided between the two *downwind* faces. Two independent
  shifts leak: measured **−5 626 counts per tick** on a 24×24 grid, always a loss,
  always unbookable. `fy = i_out − fx` is exact. The residue lands on whichever
  face is written second, so **put it on the larger component** and say so.
- **The re-associated emission.** v1 wrote `(E°·a·w) >> 16`, which has two Q16
  factors and one shift — dimensionally wrong by 2¹⁶ — and overflows int64 by
  ×1687 at high temperature. Computing `(E°·w)` first makes `absorbed` and
  `emitted` the *same arithmetic*, which is what makes §2.9's second-law gate
  exact for any `a` and any `w` rather than by the accident that 1/16 is dyadic.
- **`a_i ≤ ONE` is an invariant**, not an assumption. Positivity depends on it and
  nothing validates `heat_atten` today. It becomes an ingress check with a test,
  and every dynamic stamp (§4) is a **MAX**, never a sum.
- **`E°` and the stream widen to int64.** Measured magnitudes already reach 1.9e7
  on a small scene. **Correction (critique 2):** this is *not* a Recorder contract
  extension — `rad_net` and `rad_flux` are not in `DEFAULT_FIELDS` and the Recorder
  never records them (`src/simulation/recorder.py:81-94`). What it actually touches
  is the field-digest spec and the C++/Python plane dtypes, which is a smaller
  change but a different one, and the digest membership/dtype rule applies:
  version bump plus regenerated goldens in the same commit.

### 2.4 The transport step: one parameter, two settings

```
    step  (θ = 0):  push to the side neighbour and the down neighbour, split by
                    the direction cosines.  Rebuilt from two moves at every cell,
                    so the beam smears like √distance.  Axis-aligned ordinates
                    become pencils that never spread: the four-point cross.
    shear (θ = 1):  advance a FULL cell along the dominant axis, split only
                    transversally by f = |minor/major|.  The beam keeps its
                    heading, so it barely smears.  With 16 ordinates you see 16
                    of them: the star.
```

Davis et al. 2012 names this trade as fundamental: a sharper step buys shadow
fidelity and pays in *"fan-shaped spokes"* when the angular resolution is modest,
and *"a greater degree of diffusion in the intensity can mitigate unphysical
effects"*.

**Heat takes shear. Light takes step.**

Heat is judged by the **ignition footprint**, not by how the field looks — the
heat field is never drawn. Measured spread (max ÷ min radius over 64 directions,
leak on): shear 1.37 / 1.22 / 1.17 for a 1-tile, 3×3 and 5×5 fire, against step's
1.64 / 1.32 / 1.25. Shear is rounder at every source size, and a clustered fire
averages the striping down — the scheme study's untested caveat, now confirmed.

Light is judged by eye, and Erik chose step on the real-scene renders.

Both are exactly conservative, both hold the uniform-ambient fixed point, both are
positive. **Conservation does not discriminate**, which is the finding that made
this a free choice: the scheme study's 26% shear leak came from charging
absorption as `κ·I·w` while advancing the stream with `exp(−κ·path)` — two
absorption laws in one step. In the paired form the residual is zero.

**Implementation note that first read to me as a scheme defect:** a shear step
gathers from **two** upwind cells, so the transverse inlet edge needs the ambient
boundary seeded as well as the major-axis edge. Seed only one and a hole
propagates inward and breaks the uniform-field fixed point.

### 2.5 Angular: S16, half-offset, no rotation

- **N = 16** ordinates, evenly spaced, **half-offset** so no ordinate lies on an
  axis or a 45°. `w_m = Δφ_m/2π = 1/16`, and **`Σ_m w_m = 1`** is the
  normalisation that lets the existing baked `E°` table drop in unchanged: `E°[T]`
  is already the cell's *total* per-tick emissive power, so `a_i·E°[T_i]` is its
  total emission. That is also the quantity the stability criterion is written
  against. `w_m` is Q16.16.
- **No per-tick rotation.** v1's §2.5 claimed we had independently invented
  quadrature rotation and cited Camminady's "under a third of the quadrature
  points". Both halves are wrong for a steady-state sweep: rSN carries the angular
  flux across steps and **interpolates it** onto rotated ordinates, and *that
  interpolation* is the entire mechanism — it adds a discrete `∂²ψ/∂φ²`. We carry
  no angular flux, so what we had was phase jitter plus time-averaging. The paper's
  own §7 says rSN *"is not directly compatible with sweeping"*. Measured, ring
  ripple with 16 rotated phases against a single phase: step 2.94→3.06 at r=1 and
  2.09→2.87 at r=8; shear 1.31→1.27 and 1.54→2.17. Worse nearly everywhere.
- Dropping it also removes a `tick mod N` dependence from a synced law.
- **S16 is a sweet spot, not a budget compromise.** More ordinates make shear
  *worse* (the transverse fractions cluster near the diffusive middle), and do
  nothing for step's near field, which is a stencil artifact.

### 2.6 Emission: the temperature map IS the emission map (R-J)

No emitter list, no `T_emit_gate`. Every cell contributes `f_i·a_i·E°[T_i]`. The
existing baked `E°` table survives unchanged — int64, 4000 buckets of 4 game
units, `K⁴` by repeated integer multiplication, never `pow`.

This is also what makes glowing smoke free (§5.3): a gas cell already has a
temperature, so warming it makes it emit with no extra machinery.

### 2.7 Boundaries

- **Outflow**: a stream leaving the grid is booked to the sky channel. `RADIATION_RANGE`
  disappears — a sweep has no range, which is what that constant was faking.
- **Inflow, heat**: every boundary face admits `(E°[0]·w_m) >> 16`, an ambient
  blackbody. v1 had no inflow at all, which is a 0 K sky: measured **−4.96 game/s
  on the worst exposed cell** and a permanently-firing low-rail counter, which the
  temperature solver documents as a RED.
- **Inflow, light — this is new and it replaces a flat constant.** Today ambient
  light is `u_ambient = (0.18, 0.18, 0.22)`, a shader floor added to every lit
  surface, which does not respect walls: a sealed room receives as much as an open
  deck. Under v2 it becomes a per-boundary-face, per-ordinate inflow, so the
  sky is *transported and occluded like any other light*. A sealed compartment
  goes properly black; a hull breach throws a real shaft.
  - uniform overcast = equal inflow into every inward ordinate
  - a sun = inflow into only the ordinates near one direction
  - a turning planet = that direction rotates with time
  - it costs nothing: it is a boundary term on a solve already running
  - **keep a small flat floor as a separate dial** from the BC, because today's
    constant is what stops interiors being unplayably black.

### 2.8 Stability: the Fleck factor, and the clamp it needs

**The problem.** The material update is explicit and the loss goes as T⁴, so it is
monotone only while `4 × (per-tick radiative loss) < T_absolute`. Measured at the
shipped `rad_scale` that holds to ≈1800 game and fails hard above: at
`T_MAX_PHYS` = 16000 the cell overshoots by ×864 in one tick, and at 5000 game it
goes negative. No calibration constant fixes it — the constant that would be
stable at `T_MAX` is 432× smaller, making every radiative rate 432× slower.
Plasma weapons live exactly here.

**The remedy, from Fleck & Cummings 1971** (Wollaber's review, eqs. 17–21).
Time-average the emission across the step as a blend of its start- and
end-of-step values and solve; emission is simply scaled by

```
    f_i = 1 / (1 + α · 4 · loss_per_tick_i / T_absolute_i)       α = 0.5
```

**α follows Fleck's own condition, and it is not a constant.** Wollaber's review
gives only the lower bound (`α ≥ 0.5` for unconditional stability). The original
paper gives the other half, which matters to us: *"for large values of βcΔtσ, α
must be set equal to 1 or else the coefficient of `u_r^n` … will be negative,
tending to cause oscillations in the solution from cycle to cycle."* That
coefficient is `[1 − (1−α)g]/[1 + αg]` with `g` the dimensionless group, so it
turns negative once `g > 1/(1−α)` — i.e. **α = 0.5 admits an oscillatory mode
above `g = 2`**.

Our `g = 4·loss/T_absolute`, by temperature: 0.04 at ignition, 0.33 at 900, 0.74
at the measured crate plateau, 1.80 at 1800 game — and then 4.27 at 2500, 29 at
5000, 848 at `T_MAX_PHYS`. **So a fixed α = 0.5 is safe through the entire fire
range and admits oscillation in exactly the plasma range this scheme exists to
survive.** Hence:

```
    α_i = max(0.5, 1 − 1/g_i)
```

which is *identically* 0.5 wherever `g ≤ 2` — measured, the fire-range results are
unchanged to five figures — and rises smoothly toward 1 precisely where Fleck says
it must. Free insurance: a fixed α = 1 would cost +6.0% accuracy at 1263 game,
and the adaptive rule costs +0.2%, the same as α = 0.5.

(Measured honestly: with the clamp in place I could not actually *provoke* an
oscillation at any temperature or starting point tried, including starting a cell
at 20000 game under a 16000-game source. The clamp is suppressing a mode the
scheme admits. Relying on that is fragile when removing the mode is free.)

**The denominator is the ABSOLUTE temperature**
(`T_game + 293`), because `β = 4aT³/c_v` is defined on Kelvin. Ours is a delta
above ambient, and using it directly makes the group diverge as `T → 0` and drives
`f` toward zero *at ambient*, damping the whole map.

Applied **before the sweep**, so conservation is untouched: the stream simply
carries a smaller source. Nothing leaks, and the arc-#54 books need no new counter
for it.

Measured: stable, monotone and positive at every temperature up to 60000 game
(a 60000-game cell cools to 310 in 2 s, 24 in 10 s, where the explicit scheme
blows up). And **it is more accurate than what ships today**: cooling 1263 game
for half a second, explicit lands 5.8% *below* the analytic solution, α = 0.5
lands 2.2% above.

**But emission-only damping is not enough, and reading Wollaber §4.2.2 in full is
what caught it.** Damping the loss but not the gain moves equilibrium from
`Φ = E°(T)` to `Φ = f·E°(T)`; since `f` falls as `T⁻³`, the equilibrium stops
going as the fourth root of the flux and goes **linearly** in it. Measured: a cell
one tile from a 16000-game source settles at **1.73 million game units** instead
of 11228. That is the maximum-principle violation the review calls *"arguably the
most serious deficiency of the IMC equations"*.

Real IMC avoids it by scaling absorption by `f` too and making the remainder
**effective scattering**. We cannot copy that: scattering couples the ordinates
and costs a second sweep per tick.

**And the obvious alternative is worse — Fleck measured it in 1971.** Solving the
local nonlinear material equation directly instead (set `f = 1`, evaluate the
opacity at the end-of-step temperature, iterate) is the scheme he calls
"semi-implicit", and of it he says: *"This method … **does not conserve energy, and
energy checks in typical problems may run as high as 20%**. By employing effective
scattering, one has the double advantage of both exact energy conservation and
what appears to be unconditional stability."* That closes off the local-Newton
route I had considered, on the one criterion this project will not trade.

**So pair it with a maximum-principle clamp.** A body cannot be hotter than a
black body in equilibrium with the radiation it is absorbing:

```
    after the material update:   if E°[T_i] > Φ_i:   T_i ← E°⁻¹(Φ_i)
```

| source | true equilibrium | Fleck alone | Fleck + clamp |
|---|---|---|---|
| 1263 game | 807 | 845 | **807** |
| 5000 game | 3450 | 18982 | **3450** |
| 16000 game | 11228 | 1727794 | **11228** |
| 60000 game | 42341 | 324062265 | **42341** |

It reproduces the exact equilibrium at every temperature tried, **binds only where
the bias appears** (never at 1263 game or below), is one inverse lookup on a table
we already bake, and the energy it removes books to a **named rail counter** in
the `eos_solver` Pass-A donor-rail idiom.

This is also the literature's architecture, not an improvisation. He, Wibking &
Krumholz 2024: *"the transport terms are handled explicitly, while the
matter-radiation interacting part is treated implicitly and **locally**,
eliminating the need for non-local implicit terms in iteration."* The Fleck factor
is the local implicit treatment of the emission; the clamp is the local
enforcement of the property IMC is known to violate.

**Both must cover gas cells, and gas is the harder half** (§5.3).

### 2.9 Gates the scheme must pass

1. **Conservation**, exactly in int64, on randomised grids, every tick.
2. **Second law**: a uniform field **at ambient** is a per-cell exact fixed point —
   including non-dyadic `a` *and* non-dyadic `w`, per-cell zero rather than a sum,
   over a transparent-interior sealed room and a mixed-material scene. Every
   configuration that caught the v1 emission-form defect used a non-power-of-two
   `w`; a gate built the obvious way would have missed it.
3. **Positivity**: no negative stream anywhere, with `a ≤ 1` enforced.
4. **Stability**: assert the rail counter is zero in the normal gate scenarios and
   **non-zero in a deliberately over-driven one**, so the gate cannot pass vacuously.
5. **Maximum principle**: no cell exceeds `E°⁻¹` of the fluence it receives.
6. **Isotropy**: ring MAX/MIN and the 64-direction ignition footprint, in P1 — not
   P2, because P1 is where the transport step is chosen.
7. **CPU↔GPU bit-identity**, tol 0.
8. **Non-vacuousness on every one of the above.**

---

## 3. What rides the sweep

| payload | number type | transport | digested? | consumer |
|---|---|---|---|---|
| Heat | int64, exactly conservative | shear | yes | temperature solver Pass 1, as today |
| Light, for the rules | int, coarse (tile res) | step | **not yet** (§7.2) | stealth query, RL observation |
| Light, for the eye | float | step | never | the renderer |
| `rad_flux` | int, positive-only | either | as today | `exchange.py`, unit heat damage |

One traversal, several channels. A new payload is a channel on the sweep, never a
second traversal.

---

## 4. How every current ray interaction maps

| Interaction today | Under v2 |
|---|---|
| **Units block light** (`dyn_light_atten`, a per-tick MAX stamp) | **Unchanged, and it is the model for everything else.** A unit is a cell whose extinction went up this tick; the sweep never learns units exist |
| **Units block heat** — they do not; there is no `dyn_heat_atten` | **New.** `dyn_heat_atten` as the heat twin, a **MAX** stamp. Absorbed energy lands in `rad_flux` (§5.2) |
| Per-material `light_atten` RGB, `heat_atten` | Unchanged; they become the per-cell κ for their channel |
| Glass light-clear / heat-opaque | Unchanged, and now *structural*: two coefficients read by two channels rather than one march carrying two survivals |
| Smoke attenuating light | Unchanged. The `exp` stays render-side where it is legal |
| **Smoke attenuating heat** — does not exist | **New (R-M), its own patch** (§5.3) |
| Kirchhoff (ε == a) | Unchanged and now exactly right: κ appears identically on absorption and emission |
| **Rule 3, contact faces radiation-inert** | **Deleted deliberately, not "subsumed"** (§5.1) |
| Rule 2, mutual pairs at half weight | Genuinely subsumed — there is no pair term to halve |
| The flux limiter | **Replaced, not deleted** (§2.8) |
| `rad_flux` → unit heat damage | Kept, and now physical rather than range-gated. Consumer unchanged |
| Weapon beams | **Unchanged and out of scope.** A beam is a mutating serial pre-pass, not a field solve |
| Flashlights, lamps | Cone emitters: they emit only into the ordinates inside the cone, which is native to a directional solve. Cost stops scaling with light count |
| Fire's own light | The 16-light cap and its NMS are **deleted** — every burning tile can light the room |
| Ambient light | A boundary condition instead of a shader constant (§2.7) |
| Line of sight / infravision | §7.2 |

---

## 5. Contact, units, and gas

### 5.1 Touching solids exchange radiation, and we keep it

v1 said two adjacent opaque cells "exchange nothing through a sweep anyway". That
is false, and measurably so: an opaque cell **emits**. A's emission lands entirely
in B, and B is hotter next tick, so it chains. Measured with radiation only and no
conduction, a held 1263-game face drives heat **ten cells into a wall in 17
seconds**.

Radiation and conduction are **parallel channels and they are summed.** That is
the standard combined conduction–radiation model; in an optically thick medium
radiation becomes a diffusion with an effective conductivity
`k_rad = 16σT³/(3β)`, added to the molecular one. Erik's ruling — radiation as the
base neighbour transport, conductivity as a material-specific term on top — is
that formulation.

Measured, radiation against the conduction face at the same gap:

| material | 20-unit gap | fire gap (1263→0) | 3000→300 |
|---|---|---|---|
| hull / steel (κ 45–50) | 0.0× | 0.2× | 1.5× |
| glass (κ 1.0) | 0.3× | 0.9× | 12× |
| wood (κ 0.15) | 7.1× | 41.7× | 392× |

**Two things must be written down honestly, because they are modelling choices
rather than physics:**

1. **This is teleportation error, and it has a name.** Wollaber §5.2: absorption is
   scored uniformly across a zone whose absorption depth is far smaller than the
   zone, and re-emitted next step from the far side, *"artificially increasing the
   wave speed"*. A histogram temperature field — one value per tile, which is
   exactly ours — is documented as inadequate against it. We are choosing to use it
   as the contact-spread channel. That is legitimate; presenting it as physics is
   not.
2. **The magnitude is a choice.** Our tile-scale exchange implies a mean free path
   of one tile (≈33 cm) inside a solid. Real metal is opaque at the micron scale,
   so physically `k_rad` inside steel is negligible. The consequence is the wood row
   above: at 40:1 the radiative term largely erases wood's low conductivity.

> **`config.toml` note, and the reason it matters:** the `furniture` row is a
> placeholder test crate whose `conductivity = 0.0` exists so a lone crate burns in
> isolation during tuning. Do not generalise from it. The v1 argument "furniture has
> no conduction, so radiation must be its only loss channel" is an argument about a
> fixture. Real objects and each tree species get their own rows. (Commented in the
> table as of this session.)

### 5.2 Units block heat, and absorb into `rad_flux`

**Blocking without absorbing is not available** in a conservative transport scheme:
block means absorb, scatter, or reflect; scattering needs a scattering term and a
second sweep, and reflection is excluded top-down by R-A. So a body that blocks
must absorb, and the absorbed energy needs a home.

`rad_flux` is that home. A body absorbs radiant heat, the absorbed integer lands in
`rad_flux` at that cell, and `exchange.py` turns it into burn damage — an existing
consumer and an already-accepted, named leak. No unit temperature field, no new
plane.

Ordering is already correct and needs no change: units move at slot 3, obstacles
re-stamp at slot 6, physics runs at slot 7. **The stamp is frozen before the sweep
touches it.**

**Known limitation, deliberately accepted:** flux→damage has no thermal inertia, so
a cold unit entering a warm room takes damage immediately in proportion to flux
rather than heating up first. The later shape is a scalar temperature per unit,
heated by absorbed flux and cooled toward ambient, with damage a function of that
temperature — and once units have a temperature they **emit for free**, which is
infravision. **So the interface must be per-unit absorbed flux, not a hardcoded
flux→damage**, or that upgrade stops being additive.

### 5.3 Gas, glowing smoke, and where the stability problem actually lives

Under R-M smoke absorbs heat. The emission half is **free** — a gas cell already
has a temperature and the temperature map is the emission map, so warmed smoke
glows with no extra machinery. This is the honest version of the black-body smoke
idea, which was previously specced as a render-only advected `glow_temperature`
field *because the old engine could not afford many emitters*. A field solve
removes that constraint entirely. And it self-limits correctly: soot in the flame
glows, smoke a metre away is cool and dark.

**What it costs is bookkeeping, not emission.** Since arc #54, `gas_energy` (int64)
is the conserved truth and a gas cell's `temperature` is only its mirror — nothing
writes gas temperature directly. Today `rad_net` is folded into **solids only**
(*"Gas never receives ray radiation"*).

**Correction (critique 2): the sweep must NOT become a seam writer, and the path
already exists.** The sweep keeps writing `rad_net`; the only thing blocking gas
today is the `ts[i]` thermal-solid mask on the Pass-1 fold
(`cpp/src/temperature_solver.cpp:247`). Its gas branch already deposits through
`gas_energy::deposit_railed` and books `e_gas_deposit_sum`
(`:373-388`). So this patch relaxes a mask and reuses a booked channel, rather
than inventing a writer.

Note also that "the four existing closure groups", inherited from the survey and
from CLAUDE.md, is **stale**: the shipped identity has **six** terms — EOS,
thermal-solver gas, combustion, the seam net, the water-evac export, and the P-G5
solid group (`tests/test_thermostat_books.py:70-92`). That rule wants amending in
CLAUDE.md itself.

**Hence its own patch.**

**And this is where the stability problem is worst.** A solid's heat capacity is its
`thermal_mass`; a gas cell's is its particle count. The clamp is the protection that
does not care about heat capacity at all — it bounds temperature from the radiation
field — so it is *more* important on the gas side, not less. The Fleck factor and
the clamp both apply to gas cells.

Second-order effect worth predicting before it surprises us: radiation heats gas →
pressure rises → wind. That is a new EOS coupling and it is exactly what R-O wanted.
The feedback is stabilising (hot gas expands, density drops, it absorbs less).

---

## 6. Light

### 6.1 For the eye — the same sweep, step transport, float

Erik's choice on the real-scene renders. Reads the same `dyn_light_atten` and gas
tables the heat channel reads, so the two cannot disagree about geometry.

**What we accept by choosing step:** its four-point cross is visible as a halo
around an *isolated* small source seen **through smoke**, and more ordinates will
not remove it — it is a stencil artifact. It is averaged away by clusters, which
is why a 117-tile fire rendered cleanly. Measured ring ripple around an isotropic
lamp in a plume: step 1.75×, shear 1.75×, cascades 1.24×.

### 6.2 Cone emitters

A flashlight emits only into the ordinates inside its beam — native to a
directional solve, and it gives the **hard-edged cone** Erik prefers. Its tell at
S16 is visible banding *inside* the beam, since a cone is being quantised into
ordinates; more ordinates smooth it, at linear cost.

### 6.3 The swap seam for cascades

Cascades are not built. They are kept as a **measured, de-risked upgrade** for one
specific thing: volumetric smoke, where they are 1.24× against the sweeps' 1.75×.
The seam that keeps that option open is not "make the solver pluggable" but:

> **Nothing outside may know how the light field was produced.** The renderer and
> the stealth query read the field through one accessor, never by calling a solver.
> The per-cell optical inputs — extinction, albedo, emission, cone emitters — are a
> shared input both producers consume. The float field is never digested, so
> swapping its producer cannot move a golden.

The prototype and the bilinear fix (§6.5) are the reference a future cascade
implementation would be checked against.

### 6.4 Diffuse reflection

Not in P1. When it lands it is source iteration: one extra full solve per bounce,
converging like the albedo, for **either** method — this is not a property that
distinguishes them. For render-only light there is an escape: feed the previous
frame's bounce back in and run one pass per frame, so the bounce converges across
frames. Not available to heat, which must be deterministic per tick.

Note for whoever builds it: shear's low diffusion makes bounced light **spoky**,
because bounce is diffuse by nature and shear preserves the 16 ordinates. Step's
smoothness is the right property here — another reason light takes step.

### 6.5 If cascades are ever built, they need the bilinear fix

Our prototype's artifacts were **ours**, not the algorithm's. Osborne & Sannikov
§2.5: trace each interval from its own start to the start of the child cone on
**each** of the four coarse probes, merge each with that probe's own sample, and
only then apply the bilinear weights. Costs 4× the rays on every cascade but the
top. Measured on a real map: light in a room that should be dark drops from 1.05%
of the lit room to **0.00%**.

Also on record: a light leak is *diagnostic of a violated penumbra criterion*, and
**breach's one-tile walls violate it by construction** — which turns sub-tile light
resolution from a look question into a correctness one.

---

## 7. Determinism and the books

### 7.1 Number ingress

Doors 1 and 2 only. Integer sweep, `E°` baked at load, per-cell coefficients
quantised at the material-table boundary. **No `exp`, no `pow`, no RNG** — neither
the sweep nor anything in this design needs a random number, so ingress door 4 is
untouched and Philox stays a swarm-units concern. The float ratchet is untouched:
the sweep adds no float to a sim TU.

One divide per cell per tick enters, for the Fleck factor. It is exact if the
operands are pinned, which is the existing house rule.

The CUDA twin parallelises across ordinates with integer `atomicAdd` (order-free,
so ordinates may be summed in any order and stay bit-identical), and within an
ordinate by wavefront — **a full column for shear**, KBA skew for step.

### 7.2 What enters the digest

- **Heat**: yes, as today.
- **`rad_amb`**: it **loses per-emitter attribution** — a sweep's escaping stream
  has no emitter — so it becomes a global scalar or a per-exit-cell plane, and
  `tests/test_pf1a_radiation_books.py`'s sealed-room assertion needs rewriting.
  v1's "unchanged in meaning" was false.
- **The rules-side light field**: **computed, not digested, until a rule reads it.**
  A field nothing consumes cannot change behaviour, so digesting it early buys no
  protection and costs a golden re-baseline every time anyone moves a lamp or
  tweaks a material. Entering the digest is a one-way door. One deliberate
  re-baseline on the day the stealth rules land.
- **The render light field**: never.

### 7.3 The gate that keeps the two light fields honest

What looks dark to the player must be dark to the rules. Choosing step shrinks this
from "two different algorithms disagree" to "same algorithm, two precisions" — a
calibration problem, not a design one. The rule: **the integer field is
authoritative; the render field must be a monotone function of it**, with one test
asserting it on a real scene.

For the binary can-A-see-B query, symmetric shadowcasting is the thing to copy — it
guarantees that if A sees B then B sees A, which is what a fair hide rule needs.
That is a different question from brightness and should stay a separate primitive.

---

## 8. Cost

| | work per tick | scales with |
|---|---|---|
| Heat sweep, 128×256, S16 | 524 k cell-updates | grid × ordinates |
| Heat sweep, 256×512, S16 | 2.1 M cell-updates | grid × ordinates |
| Light sweep, same grid | the same again, per channel | grid × ordinates |
| Fleck factor + clamp | 1 divide + 1 conditional lookup per cell | grid |
| *(cascades, if ever built)* | *~55× a sweep pass in the same prototype* | *grid* |

Nothing on this table scales with the number of burning tiles or light sources.

**The budget, stated correctly** — an earlier draft of this section, and the survey
it inherited from, both got this wrong. The sim runs at **24 Hz, so the tick is
41.67 ms**; that was ruled (R-H) and is not in question. What is stale is only the
*per-system gate*, which older perf documents still express as "25% of 83 ms",
i.e. against a 12 Hz tick we no longer use.

And the 18.97 ms figure was mischaracterised twice over. It is not the atmosphere
group — it is `tests/_eos_p3_bench.py` measuring the **whole `Simulation.step()`**,
every system running, end to end from Python with the physics in C++, under a
deliberately hostile load (five explosions, a hull breach to vacuum, a flood). And
160×160 is not "a comparable grid": it is 25 600 cells against `unhcr_vessel`'s
6 000, more than four times the shipped ship scale. At shipped scale the same
bench reports **p50 1.6 ms, p99 9.78 ms** — roughly a quarter of the 41.67 ms tick,
not half of it.

So the headroom is far better than the survey implied: about **32 ms spare per tick
at shipped scale today**. The honest open item is therefore small — restate the
per-system p99 gate against 41.67 ms instead of 83 ms, which is documentation work
rather than a decision (survey Q14).

---

## 9. Migration and patch plan

| # | patch | gate | risk |
|---|---|---|---|
| P1 | The sweep, CPU, heat only, shear, shadow plane. Fleck + clamp included | conservation, second law, positivity, stability rail, isotropy — all exact, all non-vacuous | **the whole design** |
| P2 | Calibration + the reach-curve bench | the §9.3 curve, stated against the engine's own temperature ignition criterion | medium |
| P3 | Flip heat over; delete `T_emit_gate`, `RADIATION_RANGE`, `fire_ray_count`, the spatial-hash fan, rule 2's branch, `RAD_LIM_SHIFT`, the pair budget, `range_base`/`range_per_intensity`, the 16-light cap and NMS, `heat_cull`/`light_cull` | full suite + one deliberate golden re-baseline | HUMAN-TEST |
| P4 | CUDA twin | tol-0 lockstep | mechanical, pattern known |
| P5 | `dyn_heat_atten` + `rad_flux` absorption | shielding bench | feel |
| P6 | Light on the sweep (step), cone emitters, the directional sky BC | timing on a real map; the look | HUMAN-TEST |
| P7 | Smoke heat extinction — the gas energy seam writer | the #54 closure identity still closes, with the new counter named | **books** |
| P8 | `light_q` payload + the stealth query + the eye-vs-rules gate | agreement on a real scene | design-complete |

P1 is deliberately first and deliberately narrow: if exact integer conservation
does not hold, everything downstream changes, and we want to know that in one
patch. The isotropy gate moved **into** P1 (critique 1, required change 12) because
P1 is where the transport step is chosen.

The old raycaster's other customers keep their interfaces throughout: `rad_flux`
consumers unchanged, weapon beams never migrated.

---

## 10. Still open

1. **The per-system perf gate** (survey Q14) — still written against 83 ms /
   12 Hz in the older perf docs. Documentation, not a ruling: 24 Hz is settled and
   the measured headroom is comfortable (§8).
2. **The out-of-plane leak** — derived, measured, and *not ruled*. Erik set the
   reach question aside; the coefficient defaults to 0 and the sweep is correct
   either way. `reach_and_leak_study.py` has the numbers when it is wanted.
3. **Sub-tile light resolution** — now a correctness question if cascades are ever
   built (§6.5), still only a look question for the sweep.
4. **Foliage `heat_atten = 0.0`** — trees are thermally isolated. Unruled.
5. **#61, wood cannot sustain fire** — Erik's reading is that this is a heat-capacity
   problem, not a conductivity one. Fix the property; do not design around it.

---

## Systems

**Existing canonical systems this design must use.** The ray engine
(`cpp/src/raycaster.*` + `cuda_raycaster.cu`) — the replacement lands *here*, never
beside it · the temperature solver's Pass-1 fold, the only place radiation becomes
temperature · the gas energy seam, mandatory for P7 · the energy closure identity —
a new channel needs a counter and a term in one of the EXISTING groups · the face-flux energy step, the conservative pattern copied in §2.3 · the
material table, where every new optical column is a row · the Q16 boundary modules ·
the Recorder's DTYPE-class contract for the int64 widening · `pack_hover_readout`,
the one tile-probe seam · `tools/fire_tuning_lab.py`, the instrument ·
GOLDEN_AGGREGATE re-baseline discipline · **"Starting a fire"** — a fire is started
by delivering heat, never by writing `fire` alone.

**New systems this design creates**, with draft rules for CLAUDE.md once they exist:

- *The directional sweep* (`cpp/src/radiation_sweep.*`). Draft rule: THE radiative
  transport path. ONE traversal carries every payload; a new payload is a channel on
  the sweep, never a second traversal. The transport step is a parameter (shear for
  heat, step for light), never a fork of the solver.
- *The exact-integer conservation idiom for transport.* Draft rule: an integer
  leaving the radiation stream is added to the material as the same integer; the
  downwind face split carries the remainder, never a second shift; the only escapes
  are booked channels.
- *Per-cell extinction* (`heat_atten`/`dyn_heat_atten`, `light_atten`/`dyn_light_atten`).
  Draft rule: every optical interaction — material, unit body, smoke — is a per-cell
  coefficient on these planes and nothing else; dynamic stamps are MAX, never sums;
  `a ≤ 1` is an ingress invariant with a test.
- *The emission stability pair* (Fleck factor + maximum-principle clamp). Draft
  rule: emission is damped before the sweep so the stream carries only what a cell
  can afford, and no cell may end a tick hotter than a black body in equilibrium
  with the flux it absorbed; the clamp's removals are booked to a named rail. Cite
  Fleck & Cummings 1971 in the header.
- *The light-field accessor.* Draft rule: gameplay, RL and the renderer read light
  through the accessor, never by calling a solver; the float field is never
  digested; the integer field is authoritative wherever the two must agree.
- *The directional sky boundary condition.* Draft rule: ambient light enters at the
  boundary and is transported like any other light; the flat shader floor is a
  separate, smaller dial and never a substitute for it.
