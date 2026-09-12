# Ray engine v2 — adversarial critique 1: physics and integer mathematics (2026-09-12)

> **Status:** critique pass, remit = the physics and the integer arithmetic.
> **Reviews:** `docs/ray_engine_v2_design_2026-09-12.md`.
> **Against:** `docs/ray_engine_v2_survey_2026-09-12.md` (rulings R-A..R-S, not
> under critique — only the design's fidelity to them) ·
> `docs/fire_radiation_assumptions_2026-09-11.md` (what the engine does today).
>
> Every numeric claim below was **measured**, by transcribing §2.3's scheme into
> ~80 lines of Python integer arithmetic against the real baked `E°` table
> (`rad_scale = 5.1427e-5`, 4000 buckets, `K = 293 + T_game`) and the real
> material coefficients. The C++ engine was not used. Probe scripts were
> throwaway; the numbers are reproducible from the recipes given inline.

---

## BOTTOM LINE

**Yes-with-fixes for P1, no for the design as a whole.** The integer conservation
idiom in §2.3 is sound in shape and I could not break it — but only after
supplying the one step the document omits, and that omission is not cosmetic:
the downwind *face split*, which §2.3 never mentions, is the only unpaired
rounding in the scheme, and with the obvious naive implementation it leaks
(measured: 5 626 counts per tick on a 24×24 random grid, always a loss). With
the remainder trick the residual is **exactly 0** on randomised grids, and the
second-law gate is **exactly 0 per cell** — but only if the emission term is
re-associated, because as written it (a) overflows int64 by a factor of 1 687,
(b) is dimensionally wrong by 2¹⁶, and (c) passes gate 2 only by the accident
that `w_m = 1/16` is a power of two. Those are all one-line fixes and P1 is
worth cutting once they are in the document. What is *not* a one-line fix is
three larger things. **First**, §6's calibration cannot do the job it is given:
the counts-per-watt constant **cancels exactly** from the equilibrium condition
that sets reach (measured: identical equilibrium at a 100× smaller `rad_scale`),
so reach is fixed by geometry and source temperature alone. Measured under the
design's own 1/r geometry with the measured 1556 K crate, the ignition radius is
**≈17 tiles along the grid axes and ≈10 on the diagonals**, against §9.3's target
of 3 — i.e. v2 replaces an unbounded reach with a finite one (a real win) that is
still worse than the 14-tile flashover the arc exists to cure. **Second**, §3
deletes the flux limiter on an argument about the radiation field while the
limiter guards the *material* update, which is measurably unstable above
T ≈ 1800 game units and catastrophically so at `T_MAX_PHYS`. **Third**, §2.5
diagnoses the near-field error as the ray effect and it is not: the measured
near field is anisotropic by **≈2×** between axis and diagonal neighbours, this
is a property of the spatial stencil rather than of angular resolution, and
neither S64 nor per-tick rotation moves it (both measured). A different transport
step — same cost, same exact conservation — takes that from 1.95× to 1.12×.

---

## 1. The conservation claim (§2.3)

**VERDICT: correct in shape, wrong as literally written, and underspecified in
the single place that decides whether it holds.**

### 1a. Are both terms really rounded "once, before the split"? — YES

Granted in full, and I tried hard to break it. `absorbed_i` is computed once and
consumed twice (`−` into `I_out`, `+` into `ΔE_mat`). `emitted_i` is likewise
computed once and consumed twice (`+` into `I_out`, `−` into `ΔE_mat`). Neither
truncation is unpaired. The emission term's truncation does have a *physical*
consequence (a cell whose `E°·a·w` is under one count emits nothing at all), but
it is not a conservation consequence.

### 1b. The rounding the design does not mention — THE FACE SPLIT

§2.3 says `I_in` is "summed from its upwind faces" — plural. A cell therefore has
up to **two downwind faces**, and `I_out` must be divided between them by the
ordinate's direction cosines. **That division is a third truncating operation,
it is not paired with anything, and the document does not contain it.**

Measured, 24×24 random grid (mixed air / glass 0.3 / furniture 0.5 / wall 1.0,
temperatures in {0, 50, 300, 900, 1263}, S16, one tick):

| downwind split | `Σ ΔE_mat + Σ sky − Σ inflow` |
|---|---|
| `fx = (I*WX)>>16; fy = (I*WY)>>16` (naive) | **−5 626** |
| `fx = (I*WX)>>16; fy = I − fx` (remainder) | **0** |

The leak is always a *loss* (both shifts truncate toward zero), so it is a
silent, unbookable cooling of the whole world, scaling with cells × ordinates ×
ticks. **Fix:** state the remainder rule in §2.3 as part of the law, exactly as
`face_flux`'s canonical orientation is part of the law. Note the residue lands
entirely on whichever face you write second, which introduces a half-count
transverse bias; alternate the residue face by ordinate, or put it on the larger
component, and say which.

### 1c. `absorbed > I_in`, and negative `I_out`

Cannot happen **provided `a_i ≤ ONE`**. `absorbed = (I_in·a)>>16 ≤ I_in` when
`a ≤ 2¹⁶`, and `I_in` is a sum of non-negative face fluxes, so
`I_out = I_in − absorbed + emitted ≥ emitted ≥ 0`. §2.4's positivity claim is
therefore correct but *conditional*, and the condition is not enforced anywhere:
`heat_atten` is read as a raw float column in `src/simulation/materials.py`
(line 91) with no range validation. §3 then proposes `dyn_heat_atten` as the heat
twin of `dyn_light_atten` — which is a **MAX** stamp today; if the heat twin is
ever written as a sum of material + unit opacity it can exceed 1.0 and the
positivity argument dies silently. Make `a_i ≤ ONE` an ingress invariant with a
test, and say in §3 that the stamp is a MAX.

### 1d. Overflow — REAL, as literally written

`E°` tops out at `5.1427e-5 · 16291⁴ = 3.622e12` (measured off the transcribed
bake; the 1.71e13 figure in the `raycaster.cpp` bake comment is stale, pre-G12).
Then

```
    E°_max · a_i · w_m  =  3.622e12 × 65536 × 65536  =  1.556e22
    INT64_MAX                                        =  9.223e18
```

**Overflows by ×1 687.** Must be staged (`mul128_shr` or an explicit
re-association). Not a footnote: it overflows for any cell above roughly
T_game ≈ 2 000 with `a = 1.0`, which is inside normal fire range.

### 1e. Dimensions — WRONG by 2¹⁶

`absorbed = (I_in · a_i) >> 16` is right (count × Q16 → count). `emitted =
(E° · a_i · w_m) >> 16` has **two** Q16 factors and **one** shift, so it is 65 536×
too large. Either the shift is `>>32` or `w_m` is not Q16 — and §4 below shows
the document never says which, which is exactly the problem.

### 1f. Is the *radiation field* conserved, with nothing lost at the corners? — YES

With the remainder split, yes, and I measured it rather than argued it: the
identity that actually holds is

```
    Σ_cells ΔE_mat  +  Σ sky  −  Σ boundary-inflow  ==  0
```

exactly, on randomised grids, for every combination of the emission forms and
opacities I tried. The proof is one line: per cell `I_in − I_out == absorbed −
emitted`, and summing over cells telescopes the interior faces exactly because
the split is exact. Nothing is lost at DDA corners because there is no DDA — the
stencil only ever moves flux to two face neighbours.

Note the design's stated identity `Σ ΔE_mat + Σ sky ≡ 0` is missing the inflow
term, which is fine only if the inflow is zero — and §5 below shows it must not be.

### 1g. Is the `face_flux` analogy real, or superficial? — HALF REAL, and the half that is missing is exactly where the leak lives

I read `cpp/src/eos_solver.cpp:1377-1620`. What genuinely carries over: *one
integer, computed once, applied twice with opposite signs*, plus the discipline
of booking every escape into a named channel. That is real and it is the right
idiom.

What does **not** carry over, and is the reason the analogy misled the author:

- `face_flux` is evaluated **on a face, in canonical orientation, from operands
  both cells share** (`pcur_`, `n_total_`, `wind_*`). Its ± pair is between two
  *cells* in the same book. The sweep's ± pair is between two *different books* —
  the transient radiation stream and the persistent material field — and the
  stream is then **subdivided**, which `face_flux` never does. Subdivision is the
  new degree of freedom, and §1b is what it costs.
- `face_flux` returns a *magnitude* and re-applies the sign after truncation
  (the house `scale_mag` idiom, with an explicit comment about why scaling a
  signed value would round the magnitude up). The sweep needs the same discipline
  the moment any rail scales `I_out`.
- `face_flux` classifies every face (`cls` 0/1/2/3) and books the wall and export
  classes by name. §2.7 gives the sweep exactly one named channel (`rad_amb`) and
  no class taxonomy, which is thin for a scheme that will grow a rail (§7c).

### 1h. Interface consequences the design does not state

- `rad_amb` is today a **per-tile int32 plane indexed by the emitter**
  (`raycaster.cpp:566`, `gamemap.py:499`) and
  `tests/test_pf1a_radiation_books.py:243` asserts `room.rad_amb.sum() == 0`
  ("a ray escaped a SEALED room"). Under a sweep the escaping stream **has no
  emitter** — attribution is structurally impossible. §2.7's "unchanged in
  meaning" is false; `rad_amb` becomes a global scalar or a per-exit-cell plane,
  and that test needs rewriting. Say so in §10.
- `rad_net` and `rad_flux` are **int32**. Measured stream magnitudes already
  reach 1.9e7 on a small scene and scale with `E°`; at `T_MAX_PHYS` the summed
  per-cell flux reaches ~3.6e12. Both planes must widen to int64, which is a
  Recorder DTYPE-class contract extension (CLAUDE.md's Recorder rule), not a cast.

---

## 2. The single-sweep claim (§2.2)

**VERDICT: true, but only as a statement about a frozen-source problem, and the
design never says "frozen".**

Temperature **is** frozen for the whole cast: `physics_runner.py:819` and `:1207`
call `cast_fire_heat` and the temperature solver consumes `rad_net` afterwards
(`:977`, `:1372`), and `temperature_solver.cpp:246-262` is where `rad_net` becomes
`ΔT`. So the ordinates genuinely do not couple, one upwind sweep per ordinate is
the exact discrete solution of the linear system, and §2.2 is correct. This is
also the right physical approximation: light crosses a 40 m compartment in 0.13 µs
against a 41.67 ms tick.

Two things the paragraph should nevertheless say, because both are load-bearing
downstream:

1. **"Exact" means exact for the frozen source.** The composite scheme is
   *implicit in space within the tick, explicit in time*: the entire grid's
   radiation field responds instantly to the entire grid's temperature, and the
   temperature is then updated explicitly by the accumulated `ΔE_mat`. That is
   precisely the configuration in which an explicit T⁴ loss term can overshoot,
   and it is the premise the flux-limiter deletion in §3 needs and does not have
   (see §7c).
2. **"Upwind order" needs a definition.** For the sweep to be well-posed, the cell
   order must be a topological order of the per-ordinate dependency DAG: for a
   direction in the (+x, +y) octant, cell (x, y) depends on (x−1, y) and (x, y−1),
   so row-major from the upwind corner works. One sentence, but §7's KBA note and
   the CUDA twin both rest on it.

---

## 3. What falloff does the scheme actually produce?

**VERDICT: the document diagnoses the wrong error, and the real one is ≈2× and
does not respond to either of the two remedies §2.5 proposes.**

### 3a. Measured near-field anisotropy

One hot opaque cell (T = 1263 game) in an otherwise transparent field (R-L), one
furniture probe (a = 0.5) moved around it, absorbed counts per tick, averaged
over 16 rotated ticks. For an isotropic 2D 1/r field the diagonal neighbour at
distance k√2 should receive `1/√2 = 0.707` of the axis neighbour at distance k.

| k | axis (k,0) | diagonal (k,k) | measured ratio | ideal |
|---|---|---|---|---|
| 1 | 3.739e7 | 1.346e7 | **0.360** | 0.707 |
| 2 | 2.374e7 | 8.328e6 | **0.351** | 0.707 |
| 3 | 1.691e7 | 6.034e6 | **0.357** | 0.707 |
| 5 | 1.036e7 | 3.873e6 | **0.374** | 0.707 |

The diagonal is starved by a factor of **1.96×, at every radius, permanently**.
Ring statistics (MAX/MIN of `flux · r` around each ring) put it at 1.95–2.09×
from r = 1 out to r = 13.

### 3b. Why, and why neither remedy in §2.5 helps

An ordinate that is exactly axis-aligned has `WY = 0`, so **all** of its flux
travels down a single row and never spreads: a pencil beam that does not decay at
all. Measured: `flux(8,0)·8 / flux(1,0)·1 = 2.27` where 1/r demands 1.00. Every
emitter therefore sits at the centre of a faint cross.

That is a property of the **spatial stencil**, not of angular resolution, and the
measurements say so unambiguously (ring MAX/MIN, tick 0, no rotation):

| quadrature | r≈1.2 | r≈3.2 | r≈8.2 | r≈14.2 |
|---|---|---|---|---|
| S8, on-axis | 2.83× | 4.74× | 56.9× | 2179× |
| S16, on-axis | 2.13× | 2.59× | 6.15× | 29.1× |
| S32, on-axis | 1.99× | 2.17× | 2.72× | 4.97× |
| **S64, on-axis** | **1.96×** | **2.07×** | 2.10× | 2.53× |
| S16, half-offset (no ordinate on an axis or a 45°) | 1.87× | 1.82× | 1.32× | 1.79× |
| **S16, rotation-averaged over 16 ticks** | **1.95×** | **2.04×** | 1.92× | 1.95× |

Adding ordinates cures the far field beautifully (2179× → 2.5×) and **does
nothing** to the near field: S64 is 1.96× at r = 1, S16 is 2.13×. Rotation
likewise cures the far field and leaves the near field at 2.04×. The phase step
§2.5 specifies, `2π/N² = 1.4°`, gives a near-axis ordinate `WX = 0.976` — still a
pencil. So §2.5's sentence *"at r = 1–2 tiles the angular gap is under one tile,
so the near field … is fully covered"* is **true about the angular gap and
irrelevant**: the near-field error is the stencil, and the paragraph's conclusion
("the artifact degrades a quantity that no longer matters") does not follow.

Gameplay consequence, stated plainly: since flux ∝ 1/r, a 2× flux ratio is a 2×
*ignition-radius* ratio. Calibrate the crate to ignite at 3 tiles and it ignites
at ~4 along the axes and ~2 on the diagonals — a permanent, non-averaging,
cross-shaped ignition footprint.

### 3c. There is a cheap fix, and I measured it

Replace the two-face cosine split with a **shear / characteristic transport step**:
advance a full cell along the dominant axis and interpolate the transverse
fraction between the two cells there (`f = |dy/dx|` for |dx| ≥ |dy|). Every cell
is still visited exactly once per ordinate — same cost — and conservation is
still exact (measured residual **0** on a random grid, same remainder trick).

| | diag/axis at k=1 | at k=3 | ring MAX/MIN r≈1 | r≈3 | r≈13 |
|---|---|---|---|---|---|
| §2.4 step differencing | 0.360 | 0.357 | 1.95× | 2.04× | 1.94× |
| shear sweep | **0.788** | 0.645 | **1.12×** | 2.11× | 2.61× |
| ideal | 0.707 | 0.707 | 1.00× | 1.00× | 1.00× |

It trades near field for far field, which is the trade §2.5's own argument says we
want ("striping can only appear in the far field, which … carries flux far too
weak to ignite anything"). Note that for this scheme the far field does **not**
improve with ordinate count (S16/S32/S48 all give 2.5× at r = 18) — its residual
is interpolation asymmetry, not the ray effect — so if it is chosen, §12 Q2
("ordinate count") largely evaporates and S16 stands.

### 3d. The reach itself — the biggest finding in this document

The equilibrium of a receiver under the sweep is `Φ(r) = E°[T_r]`: the receiver's
own absorptivity `a_r` cancels (it multiplies both absorption and emission) and
so does the source's `a_s`, once folded into a pure geometric factor
`G(r) = Φ(r)/E°[T_s]`. So

```
    K_r  =  G(r)^(1/4) · K_s
```

and **the counts-per-watt calibration constant cancels completely**. Measured, to
be sure: re-baking `E°` at a 100× smaller `rad_scale` gives `G(3, axis) = 0.11249`
against `0.11248` — identical equilibrium.

Measured `G(r)` and the implied equilibrium, source held at the measured crate
plateau 1556 K, rotation-averaged, S16, ignition at 280 game (573 K):

| tiles k | G axis | G diag | T_eq axis (game) | T_eq diag |
|---|---|---|---|---|
| 1 | 0.2487 | 0.0895 | 806 | 558 |
| 3 | 0.1125 | 0.0401 | 608 | 404 |
| 6 | 0.0572 | 0.0218 | 468 | 305 |
| 8 | 0.0422 | 0.0166 | 412 | 265 |
| 12 | 0.0273 | 0.0110 | 339 | 211 |
| 16 | 0.0200 | 0.0081 | **292** | 174 |
| 20 | 0.0157 | 0.0063 | 258 | 145 |

**Ignition radius: ≈17 tiles along an axis, ≈10 tiles (Euclidean) on a diagonal.**
§9.3's design target is 3.

This is not a tuning miss, it is structural, and three consequences follow:

1. **§6 is doing a job it cannot do.** *"The single free constant is the
   counts-per-watt scale … fixed by requiring the flux at 3 tiles … to equal the
   ignition threshold. Everything else follows."* The constant fixes the **rate**
   (how fast a receiver climbs, how fast an emitter cools) and has **no effect on
   reach**. Reach is `G(r)` and `K_s`, full stop.
2. **Under 1/r, the reach target pins the plateau — and R-E has already pinned the
   plateau somewhere else.** Because `G ∝ 1/r` and `G_ign ∝ (K_r/K_s)⁴`, reach
   scales as **the fourth power of the source temperature**: `r_ign ∝ K_s⁴`. To
   land r_ign = 3 on the axis the crate would have to plateau at **989 K**, not
   1556 K. R-E says do not revert `H_BED_SHIFT`. The two requirements are in
   direct conflict and one of them has to give.
3. **The 1/r² out-of-plane leakage is not a feel dial — it is the fix.** Survey
   §9.5 demotes it to "retained as a feel dial, not a necessity" and §9.2 argues
   1/r suffices. Measured: with the same near field and a 1/r² falloff,
   `r_ign = √(0.2487/0.01839) = 3.68 tiles` — dead on §9.3's target. And the
   sensitivity halves (`r_ign ∝ K_s²` instead of `K_s⁴`), so the law stops
   handing us a new flashover every time the plateau moves — which
   `fire_radiation_assumptions_2026-09-11.md` §5 names as the thing to avoid.

To be fair to the design: v2 *does* introduce a real horizon. Today's law has an
ignition radius of ~670 tiles (assumptions doc §2) because receivers are
loss-free; v2 makes it finite. It is a genuine improvement. It is just still
worse than the 14-tile flashover that opened the arc.

One smaller consistency note: §9.3 derives its reach from a **flux** threshold
(10–12 kW/m²) while breach ignites on a **temperature** threshold
(`ignition_temp` 280–300 game). A grey surface in radiative balance at 11 kW/m²
sits at 680 K, while 573 K corresponds to 6.1 kW/m² — the two criteria differ by
~1.8× in flux, ~1.8× in radius under 1/r. The §6 bench must pick one and say so.

---

## 4. Ordinate weights and normalisation

**VERDICT: UNDERSPECIFIED, and §5's exactness silently depends on the gap.**

The design writes `w_m` and never defines the quadrature, the normalisation, or
the number type. What is needed:

- **`Σ_m w_m = 1`** is the correct condition, and it is the one that lets the
  existing `E°` table drop in unchanged: the current 8-ray law already uses
  `w = 1/8` with 8 rays, so `E°[T]` is already the cell's *total* per-tick
  emissive power and `a_i·E°[T_i]` is its total emission. Say it.
- In a 2D azimuthal discretisation that means `w_m = Δφ_m/2π`, uniform for an
  evenly spaced set — which is why nothing else in the design fixes the absolute
  scale, and why §6's constant is the only scale knob (and §3d shows what it does
  and does not control).
- **Dimensional bookkeeping is wrong** (§1e): two Q16 factors, one shift.
- **Overflow** (§1d): 1.556e22 against 9.223e18.
- Path length is direction-dependent and the scheme ignores it: a 45° ordinate's
  energy passes through 1.41 cells per unit distance against 1.00 for an axis
  ordinate, so diagonal directions are over-attenuated by √2 through
  semi-transparent media (glass 0.3, furniture 0.5). The current DDA has exactly
  the same bias, so this is **not a regression** — but §3's "Same numbers, better-
  defined meaning" overstates it, and a finite-volume form with the
  `κV/(μΔy + ηΔx)` denominator would get it right for free.

---

## 5. Equilibrium / second law (gate 2)

**VERDICT: the gate CAN be made bit-exact, and it is a genuinely strong property —
but the emission form in §2.3 passes it by luck, and the missing boundary
condition fails it outright.**

### 5a. It works, and better than the current bucket trick

The uniform field is an **exact fixed point** of the discrete scheme, because the
remainder split makes a cell's two upwind inflows sum to exactly the upwind cell's
`I_out`. Measured across uniform-temperature scenes (T ∈ {0, 300, 900, 1263};
all-furniture, all-opaque, and randomly mixed non-power-of-two materials
{0, 0.15, 0.3, 0.5, 0.87, 1.0}; sealed rooms with transparent interiors; four
rotated phases): **every interior cell's `ΔE_mat` is exactly 0**, not approximately.

This is strictly stronger than the current `diff == 0` bucket trick, which needs
two cells to land in the same 4-unit `E°` bucket. Report it as a win.

### 5b. …but only for the *paired* emission form

The exactness needs `absorbed` and `emitted` to be the **same arithmetic**.
Rewrite §2.3's emission as

```
    src_i,m    =  (E°[T_i] * w_m) >> 16        // the ordinate's source intensity
    emitted_i  =  (src_i,m  * a_i) >> 16       // SAME multiply, SAME shift as absorbed
```

Then `absorbed == emitted` identically whenever `I_in == src`, for **any** `a` and
**any** `w`. As written, `emitted = (E°·a·w)>>16` truncates in a different order
and the two differ. Measured on a uniform-T mixed-material grid:

| `w_q` | design form `(E°·a·w)` | paired form `((E°·w)·a)` |
|---|---|---|
| 4096 = ONE/16 | 0 nonzero cells / 91 | 0 / 91 |
| **4095** | **47 / 91**, dE ∈ [−16, +10], Σ = −67 | **0 / 91** |
| **4097** | **41 / 91**, dE ∈ [−16, +13], Σ = −86 | **0 / 91** |

**The design passes gate 2 only because N = 16 makes `w_m = 1/16` a power of
two.** Any quadrature with non-dyadic weights — a level-symmetric S_N set, a
Gauss set, an N that is not a power of two, or a normalisation correction — breaks
it. Since §4 shows the quadrature is never actually specified, the design is
currently relying on an assumption it does not make. The re-association is free
and also fixes the overflow and the dimension, so there is no reason not to adopt
it.

The magnitudes are small (−67 counts/tick ≈ 7e-4 game/s at furniture `his = 3`),
so this is a gate failure and a second-law wart, not a gameplay bug. But §9 gate 2
says *"bit-exactly, not approximately"*, and that is the right standard.

### 5c. The inflow boundary condition — MISSING, and it fails gate 2 hard

§2.7 covers outflow only. Today rule 4 charges the emitter a **net** against an
ambient blackbody, `a_s·τ·w·(E°[T_s] − E°[0])` (`raycaster.cpp:556-560`): today's
sky is *at ambient*. A sweep with zero inflow at the grid edge is a **0 K sky**,
which is not the same term and not "unchanged in meaning".

Measured, open field of furniture (a = 0.5) entirely **at ambient** (T = 0):

| boundary inflow | Σ ΔE_mat | worst cell | mean |
|---|---|---|---|
| 0 (cold sky) | −9 218 000 /tick | **−4.96 game/s** | −0.73 game/s |
| `(E°[0]·w_m)>>16` (ambient sky) | **0** | **0** | **0** |

`E°[0] = rad_scale·293⁴ = 389 475` counts — not negligible; `E°[300]/E°[0] = 16.5`.
With a cold sky every exposed surface drains permanently, the whole map is pinned
on the Pass-1 low rail, and `t_low_rail_hits` fires continuously — which
`temperature_solver.cpp:279-291` documents as *"a hit inside a gate run is a RED"*.

**Fix:** §2.7 must state the inflow boundary condition as an ambient blackbody,
`I_in = (E°[0] · w_m) >> 16` on every boundary face, booked into the same channel
with the opposite sign. That also restores the exact meaning of today's net sky
term and keeps `Σ ΔE_mat + Σ sky − Σ inflow == 0` closing.

---

## 6. Step differencing (§2.4)

**VERDICT: positivity CORRECT (conditionally); the diffusivity assessment is right
about walls and wrong about what the diffusivity actually costs.**

- **Positivity:** correct, given `a_i ≤ ONE` (§1c). The argument against diamond
  differencing ("negative intensity ⇒ clamp ⇒ a conservation leak that has to be
  booked") is sound and well reasoned. Keep it.
- **Wall far face heating as fast as the near face: NOT a problem.** An opaque
  cell absorbs everything arriving and re-emits only at its **own** temperature,
  so a cold wall's far face sees only ambient-level emission. Measured, hot source
  behind a solid wall, flux 4 tiles past it: **2.5e6 against 8.8e7 unblocked
  (2.8%)**, and the entire residual is the wall's own ambient re-emission, not
  leakage. For a semi-transparent wall the far face gets half the near face's
  flux, which is correct Beer–Lambert. Shadow quality is a genuine strength of
  this scheme and is worth saying in §2.4.
- **Apertures are soft but not wrong.** A 1-tile gap in a wall gives a penumbra
  decaying ~2× per tile over 3–4 tiles at 4 tiles downstream. Over-soft against
  the true geometry, defensible for heat.
- **What the diffusivity actually costs is the near-field anisotropy of §3**, and
  §2.4's third bullet ("the cost is smearing across cells, which at our reach of
  3–8 tiles is comfortably below the tile quantisation we already accept")
  measures the wrong thing. The smearing is not the cost; the *lack* of smearing
  along the axes is.
- §2.4 claims first-order step differencing "smooths the ray effect rather than
  sharpening it — the error works in our favour here". Measured, that is true in
  the far field and false in the near field, where it *creates* a 2× artefact that
  the ray effect did not have.
- One consequence worth recording even though it is not a defect: an opaque wall
  facing a fire now heats and then re-emits into the **next compartment** at its
  own temperature. That is physically right and new. It is also the mechanism by
  which the `fire_tuning` level's "stations ≥6 tiles apart are independent"
  premise will keep being false through walls, not just through air.

---

## 7. The deletions (§5 and the §3 table)

### 7a. Rule 2, half-weight mutual pairs — GENUINELY SUBSUMED ✓

Granted without reservation. The half-weight branch exists because the current
law is a **pair** term `a_s·a_r·τ·w·(E°[T_s] − E°[T_r])` cast from both ends, so
a mutual pair would otherwise count twice (`raycaster.cpp:601`, `:612`). The sweep
has no pair term at all — each cell independently emits `a·E°[T]` and absorbs
`a·I_in` — so there is nothing to halve. The design's phrasing ("symmetry is
automatic") is slightly off (there is no symmetry to be automatic about; the
construct is gone), but the conclusion is right.

### 7b. Rule 3, contact faces radiation-inert — THE CLAIM IS FALSE, measured

§3 says: *"Two adjacent opaque cells exchange nothing through a sweep anyway: the
first is opaque, so nothing reaches the second."* That reasons about **external**
radiation and forgets that an opaque cell **emits**. Measured, two face-adjacent
opaque cells, A at 1263 game and B at 0:

```
    ΔE[A] = −300 588 314 /tick      ΔE[B] = +74 781 945 /tick
    B gains  3 420 game units/s (wood, his = 3)  /  856 game units/s (steel, his = 5)
```

So rule 3 is **deleted, not subsumed**, and what comes back is precisely what it
prevented: radiative transfer across a contact face, in parallel with conduction's
own face flux (`temperature_solver.cpp:405-462`). For **furniture**
(`conductivity = 0` ⇒ `NO_FACE`) that is arguably the fix for assumption 5 — it
gives furniture a loss channel for free — and may well be wanted. For **steel and
hull** (`conductivity = 45/50`) it is a straight double count of a path conduction
already owns. Either way it is a **behaviour change needing a ruling and a
measurement against the conduction face**, not a one-line table entry that says
"subsumed". Note also that the design's `heat_atten`-as-`κ` reading makes **glass**
(a = 0.3) a partial absorber, so rule 3's documented "stacked absorbers / flush
glass: the interior of an assembly does not radiate" semantics
(`raycaster.cpp:578-583`) change too.

### 7c. The flux limiter — THE ARGUMENT IS A CATEGORY ERROR, and the thing it guards is measurably unstable

§3: *"It existed to stop a divergent pairwise exchange. A sweep cannot diverge:
`I_out ≤ I_in + emitted`."* That is a bound on the **radiation field**. The
limiter guards the **material update**. `raycaster.h:222-226` says so: *"It is a
STABILITY constant, not a feel dial … a rail against the T³ steepening at
T_MAX_PHYS-scale gaps."* And `temperature_solver.cpp:272-291` makes the low rail's
whole justification the limiter's budget argument — delete the limiter and that
comment is void.

The update is `ΔT = −(a·E°[T]) >> his` on a Q16.16 temperature, and because
`E° ∝ T⁴` it is monotone only while `4 × (per-tick loss) < T`. Measured at the
shipped `rad_scale`, furniture (a = 0.5, `his = 3`):

| T (game) | K | loss/tick (game) | loss/T | 4·loss/T | |
|---|---|---|---|---|---|
| 280 (ignition) | 573 | 5.36 | 0.019 | 0.08 | ok |
| 900 | 1193 | 100 | 0.111 | 0.44 | ok |
| 1263 (measured crate) | 1556 | 287 | 0.227 | **0.91** | marginal |
| **1800** | 2093 | 945 | 0.525 | **2.10** | **overshoots negative** |
| 2500 | 2793 | 2 993 | 1.20 | 4.79 | unstable |
| **16000 = `T_MAX_PHYS`** | 16293 | **3.45e6** | 216 | 864 | catastrophic |

The measured crate plateau already sits at 0.91 — inside the margin, shedding 23%
of its own temperature per tick. Above ≈1800 game the cell goes negative in one
tick and is caught by the low rail, destroying energy that the closure identity
then has to book as a rail hit. That regime is reachable: payload explosions, the
fire transient, and `T_MAX_PHYS = 16000` itself.

**And calibration cannot save it.** The constant that would be stable at
`T_MAX_PHYS` is `c = 1.19e-7`, **432× smaller** than the shipped `5.1427e-5` —
which makes every radiative rate 432× slower, i.e. a crate would take minutes to
ignite its neighbour. There is no single constant that is both fast enough for
gameplay and stable to `T_MAX_PHYS` under an explicit T⁴ loss. Note also that
today the engine is **doubly** protected — `T_emit_gate = 930` means no cell below
1223 K emits **at all** — and R-J deletes that too, in the same patch.

**Fix**, and it reuses a canonical pattern rather than restoring the old one: cap a
cell's **total** emission for the tick as a fraction of its own heat content,
computed once per cell before the ordinate loop, and scale every `src_i,m` by that
factor — which is exactly `eos_solver.cpp`'s Pass A donor rail (`s_plane_`,
`head`, `apply_scale` with the magnitude-first sign discipline,
`e_energy_floor_sum`), one clamp per cell per tick instead of the old one per ray.
Book it as a named channel like `e_rail_sum`. Alternatively make the emission
semi-implicit (linearise `E°[T]` about `T_n` and solve the scalar update), which
removes the constraint entirely at the cost of one divide per cell per tick.

Either way §9 needs a **stability gate**: assert `4 × Σ_m emitted_i < T_i` for
every cell, or assert the rail's counter is zero in the gate scenarios and
non-zero in a deliberately over-driven one.

---

## REQUIRED CHANGES, in priority order

1. **Settle what sets the reach, before P1.** The calibration constant cancels
   (measured); geometry and source temperature are the only levers. As designed,
   `r_ign ≈ 17` tiles (axis) / `≈10` (diagonal) at the 1556 K plateau against a
   target of 3, and `r_ign ∝ K_s⁴`. Rewrite §6 to say what it actually fixes (the
   *rate*), and pick one of: (a) adopt the out-of-plane leakage / 1/r² as
   **structural** — measured `r_ign = 3.68`, on target, and `r_ign ∝ K_s²` so the
   law stops being fragile to the plateau; (b) make R-M medium absorption
   load-bearing, with a stated e-fold length; (c) lower the plateau to ~989 K,
   which contradicts R-E. This changes what P3 calibrates and possibly what P1
   builds, so it cannot wait for P3.
2. **Do not delete the flux limiter; replace it.** State the criterion
   `4 × (per-tick radiative loss) < T`, show it holds at the chosen calibration up
   to `T_MAX_PHYS`, and if it does not (it does not, at any usable constant), add
   a per-cell per-tick emission rail in the `eos_solver` Pass-A donor-rail idiom
   with a named counter, or go semi-implicit. Add a stability gate to §9 and to
   P1's gate list.
3. **Specify the downwind face split in §2.3 and make it exact.** `fx =
   (I_out·WX)>>16; fy = I_out − fx`. Measured: the difference between a 0
   residual and a −5 626 counts/tick unbookable loss. Say which face carries the
   residue and why.
4. **Re-associate the emission term** to `emitted = ((E°[T_i]·w_m)>>16 · a_i)>>16`.
   One change fixing three defects: the ×1 687 int64 overflow, the 2¹⁶ dimensional
   error, and gate 2's accidental dependence on `w_m` being a power of two
   (measured: 47/91 uniform-T cells break at `w_q = 4095`).
5. **State the inflow boundary condition** as an ambient blackbody
   `(E°[0]·w_m)>>16` per ordinate on every boundary face, and correct §2.7's "the
   same sky term as rule 4, unchanged in meaning". Measured: 4.96 game/s spurious
   drain at the map edge and a permanently-hot low-rail counter otherwise. Fix the
   stated identity to `Σ ΔE_mat + Σ sky − Σ inflow ≡ 0`.
6. **Fix or budget for the near-field anisotropy.** Either adopt the shear /
   characteristic transport step (measured: diag/axis 0.360 → 0.788 against an
   ideal 0.707; near-ring MAX/MIN 1.95× → 1.12×; conservation still exactly 0;
   identical cost), or state the 2× axis/diagonal bias as accepted and calibrate
   the reach curve against the **worst** direction. Correct §2.5's diagnosis:
   measured, neither S64 nor per-tick rotation moves the near field at all.
7. **Retract "rule 3 is subsumed."** Measured 7.48e7 counts/tick between
   face-adjacent opaque cells (3 420 game/s for wood). Decide, with a measurement
   against the conduction face flux, whether adjacent-solid radiative exchange is
   wanted (probably yes for `conductivity = 0` furniture, probably a double count
   for steel/hull) and write it as a ruling, not a deletion.
8. **Define the quadrature** in §2.5/§2.3: the direction set, `w_m`'s number type,
   and `Σ_m w_m = 1` as the normalisation that makes the existing `E°` table drop
   in unchanged. Note that `Σ w_m = 1` is what makes `a_i·E°[T_i]` the cell's
   total per-tick emission, which is the quantity the stability criterion in (2)
   is written against.
9. **Enforce `a_i ≤ ONE` at the material-table boundary**, with a test, and state
   in §3 that `dyn_heat_atten` is a **MAX** stamp like `dyn_light_atten`. §2.4's
   positivity claim is conditional on it and nothing currently validates
   `heat_atten`.
10. **Widen the plumbing and say what happens to `rad_amb`.** `rad_net` and
    `rad_flux` are int32 and the sweep's magnitudes need int64 (a Recorder
    DTYPE-class contract extension, not a cast). `rad_amb` loses per-emitter
    attribution entirely — it becomes a global scalar or a per-exit-cell plane —
    and `tests/test_pf1a_radiation_books.py:243`'s sealed-room assertion needs
    rewriting. Put both in §10.
11. **Make gate 2 non-vacuous** per §9 item 5: it must use non-dyadic `a` **and**
    non-dyadic `w`, assert **per-cell** zero (not just the sum), and cover a
    transparent-interior sealed room, a mixed-material scene, and several rotated
    phases. Every configuration I ran that caught the emission-form defect used a
    non-power-of-two `w`; a gate built the obvious way would have missed it.
12. **Move the isotropy measurement from P2 into P1.** §11 puts "ray-effect
    measurement" in P2, but P1 is where the spatial scheme is chosen and §3 shows
    the spatial scheme, not the angular one, decides the near field. Add a ring
    MAX/MIN isotropy gate and a positivity gate to P1's list, since P1 is
    explicitly the patch whose job is to fail early.

---

## OPEN QUESTIONS

1. **Which reach lever?** 1/r² leakage, medium absorption, or a lower plateau.
   This is a feel ruling that collides with R-E (plateau) and R-L (clear air), and
   it is Erik's, not mine. My measurement only says it cannot be the calibration
   constant.
2. **Is adjacent-solid radiative exchange wanted?** It would give furniture the
   loss channel assumption 5 says it lacks, for free — but it also double-counts
   with conduction wherever conductivity is non-zero, and it changes glass
   assembly semantics. Needs a ruling and a number for how it compares with the
   conduction face flux.
3. **What is the physical counts-per-joule mapping?** §6 calls the calibration
   "derived, not fitted", but nothing in the tree pins `thermal_mass` (8 for
   furniture, 32 for steel) to a physical heat capacity, so the derivation cannot
   be completed from the repo. Without it, "the flux at 3 tiles equals the
   ignition threshold" cannot be evaluated in SI at all. Somebody has to write
   down the tile's assumed mass, specific heat and out-of-plane depth.
4. **What is the plateau under v2?** My reach numbers hold the source at the
   *measured* 1556 K. Under R-K the source gains a real radiative loss channel,
   so its plateau will drop to wherever `H_bed` balances it — which is a fire-model
   question the arc defers. The scaling `r_ign ∝ K_s⁴` is the robust statement;
   the absolute 17 tiles is conditional.
5. **Does the per-cell emission rail belong in the arc-#54 books as a named
   channel, or does a semi-implicit update remove the need for one?** The first is
   cheaper to build and adds a counter; the second is cleaner physics and adds a
   divide per cell per tick. I do not have the tick budget (survey Q14) to choose.
6. **Should absorption carry the direction-dependent path length?** Both the
   current DDA and the proposal over-attenuate diagonals by √2 through
   semi-transparent media. Not a regression, and a finite-volume form fixes it for
   free — but whether it is worth the reformulation depends on how much glass and
   furniture end up in the optical path of real levels, which no authored level
   yet answers.
