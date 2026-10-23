# Ray engine v2 — design (2026-09-12)

> **Status:** design, pre-critique. Nothing built. This is the document the
> adversarial critique pass reviews, and the one the patch plan is cut from.
>
> **Depends on:** `docs/ray_engine_v2_survey_2026-09-12.md` (the blessed shape,
> rulings R-A..R-S, the cost audit, the gates) · `docs/fire_radiation_assumptions_2026-09-11.md`
> (what the current model does, with citations) ·
> `docs/architecture/engine/08_ray_engine.md` (the chapter this replaces) ·
> `docs/architecture/engine/14_determinism_and_number_ingress.md` (the four doors).
>
> **Mandate (Erik, 2026-09-12):** design as far as possible without relaxing any
> goal; use the literature; re-enter the loop only for what neither critique nor
> research can settle. §12 is that list.

---

## 1. What we are building, in one paragraph

Radiation stops being a per-emitter ray cast and becomes a **field solve**, like
pressure and conduction already are. Two solvers share one geometry and one set
of per-cell extinction coefficients: a **conservative integer discrete-ordinates
sweep** for heat, inside the sim and bit-identical across machines, and
**radiance cascades in GLSL** for light, render-only. A third, cheap consumer —
a scalar light payload riding the heat sweep — gives the rules and the RL agent
an exact, integer answer to "how lit is this tile", without making the beautiful
float field deterministic.

---

## 2. The heat solver — conservative integer discrete ordinates

### 2.1 The law

The radiative transfer equation, non-scattering (R-L: clear air is transparent;
smoke absorbs but we do not scatter in the heat channel):

```
    ŝ · ∇I(x, ŝ)  =  −κ(x) I(x, ŝ)  +  κ(x) B(T(x))
```

`κ` is the per-cell extinction, `B(T)` the blackbody source. Discretise `ŝ` into
`N` ordinates and the grid into cells, and each ordinate becomes an independent
transport problem.

### 2.2 Why ONE sweep is exact — no iteration

**In a non-scattering medium the ordinates do not couple.** Nothing scatters
from direction `m` into direction `m'`, so there is no source iteration, no
convergence criterion, no residual. One upwind sweep per ordinate is the exact
discrete solution.

That is a large simplification over the general DOM literature, which is
dominated by the scattering case, and it is what makes this affordable and
deterministic: a fixed, finite, non-iterative amount of work per tick.

### 2.3 The integer scheme, and why it conserves exactly

This is the piece §10.2 of the survey named as most likely to force a redesign.
It reduces to an idiom the engine already trusts.

Per ordinate `m`, sweep cells in upwind order. For cell `i` with incoming
integer intensity `I_in` (summed from its upwind faces):

```
    absorbed_i  =  (I_in * a_i) >> 16                  // a_i = extinction, Q16.16
    emitted_i   =  (E°[T_i] * a_i * w_m) >> 16         // w_m = ordinate weight
    I_out       =  I_in − absorbed_i + emitted_i
    ΔE_mat[i]  +=  absorbed_i − emitted_i              // the SAME two integers
```

**Conservation is structural, not tuned.** Every integer that leaves the
radiation stream is added to the material, and every integer the material emits
is removed from it. The only escapes are the grid boundary (booked to the sky
counter, exactly as rule 4 does today) and nothing else. So

```
    Σ_cells ΔE_mat  +  Σ sky  ≡  0        exactly, in int64
```

by the same ± idiom as `eos_solver.cpp`'s `face_flux`: one integer, applied
twice with opposite signs. No rounding can break it because the rounding happens
*once*, before the split.

**Why the ordering is safe.** Within one ordinate the sweep is sequential
(upwind), which is a fixed deterministic order. Across ordinates the
accumulations into `ΔE_mat` are integer adds, which are associative, so ordinates
may be summed in any order — including a GPU `atomicAdd` — and still be
bit-identical. This is the same argument that already makes `rad_net` twin-safe.

### 2.4 Spatial scheme: step (upwind) differencing

First-order step differencing, not diamond difference. Three reasons, in order
of weight:

1. **Positivity.** Step differencing cannot produce a negative intensity.
   Diamond difference can, and a negative intensity in an integer scheme means a
   clamp, and a clamp means a conservation leak that has to be booked. Not worth
   it.
2. Its numerical diffusion **smooths the ray effect** (§2.5) rather than
   sharpening it — the error works in our favour here.
3. It is the cheapest per cell.

The cost is smearing across cells, which at our reach of 3–8 tiles is
comfortably below the tile quantisation we already accept.

### 2.5 Angular: S16 with per-tick quadrature rotation

**The ray effect is the literature's name for our 8-ray artifact.** It is worst
exactly where we live: isolated sources in a weakly-absorbing medium. The
standard remedies are more ordinates (weakly effective, linear cost), a
first-collision source, or **quadrature rotation**.

The first-collision source is **not available to us** and this is worth stating
plainly so nobody re-proposes it: in a non-scattering medium there is no
collided flux, so a first-collision source degenerates into exactly the
per-emitter ray tracing this design exists to escape.

**Quadrature rotation is available, and we already independently invented it.**
The current fan rotates its phase per tick by `2π/N²`
(`raycaster.cpp:943-950`). The rotated-S_N literature reports qualitatively
equivalent solutions at **under a third** of the quadrature points. So:

- `N = 16` ordinates, evenly spaced.
- Phase rotated per tick by `(tick mod N) · 2π/N²`, deterministic, no RNG.
- Effective angular resolution over a thermal time constant ≈ S48+.

Heat integrates over time through thermal mass, so per-tick angular jitter
averages out rather than flickering, which is precisely why rotation is free
for heat where it would be objectionable for light.

**Where the residual ray effect lands.** At `r = 1–2` tiles the angular gap is
under one tile, so the near field — where ignition happens — is fully covered.
Striping can only appear in the far field, which after R-K (every cell radiates)
carries flux far too weak to ignite anything. The artifact degrades a quantity
that no longer matters.

### 2.6 Emission: the temperature map *is* the emission map (R-J)

No emitter list, no `T_emit_gate`. Every cell contributes `κ_i · E°[T_i]` to
every ordinate it sits on. The existing baked `E°` table survives unchanged: int64,
4000 buckets of 4 game units, `K⁴` by repeated integer multiplication, never
`pow`. It is already the right object, it was simply only reachable by emitters.

`rad_scale` is replaced by the calibration in §6.

### 2.7 Boundary and the books

A ray leaving the grid charges the emitter, booked into `rad_amb` — the same
sky term as rule 4, unchanged in meaning. `RADIATION_RANGE` disappears entirely:
a sweep has no range, it runs to the grid edge by construction, which is what
that constant was faking (survey §5.2).

The closure identity gains one named channel and no new group:
`Δ Σ_accountable == EOS + thermal-solver + combustion + seams`, with the sweep
reporting into the thermal-solver group and `rad_amb` unchanged.

---

## 3. How every current ray interaction maps

Erik's question, and the answer is that almost everything survives because the
interactions were always **per-cell coefficients**, never properties of the ray
machinery.

| Interaction today | Where it lives now | Under v2 |
|---|---|---|
| **Units block light** | `dyn_light_atten` = static material MAX per-unit opacity, restamped each tick by `stamp_units` | **Unchanged, and this is the model for everything else.** A unit is an opacity stamp on the grid, so it is just a cell whose extinction went up this tick. The sweep and the cascades both read `dyn_*`, neither knows a unit exists. Erik: *"units work the same way as walls as far as rays go, perhaps that should continue"* — yes, and it is the cleanest part of the whole migration. |
| **Units block heat** | **They do not.** There is no `dyn_heat_atten` (`gamemap.py:404`) | **New, and nearly free**: add `dyn_heat_atten` as the heat twin of the light stamp. A body between you and a fire should shade you. §12 Q1 — it is a gameplay change, not just plumbing. |
| Per-material `light_atten` RGB triple | material table → `light_atten` plane | Unchanged. Becomes the cascade's per-cell per-channel extinction. |
| Per-material `heat_atten` | material table → `heat_atten` plane | Becomes `κ` in §2.1 directly. Same numbers, better-defined meaning. |
| Glass light-clear / heat-opaque | two independent columns | Unchanged and now *structural*: the two solvers read two different coefficients by construction, instead of one march carrying two survivals. |
| Smoke and gas attenuation of light | per-gas absorption + scatter triples, `smoke_absorb_scale`, `exp()` per step | Cascades keep it (they support participating media natively, R-3). The `exp` stays render-side where it is legal. |
| Smoke absorption of heat | **does not exist** — gases never attenuate heat (`raycaster.cpp:453-455`) | **New per R-M**: smoke gets a heat extinction coefficient, one per-gas table column, folded into `κ`. Integer Beer-Lambert, no `exp` — the recipe already ships in `combat.py` for the hitscan laser. |
| Kirchhoff (ε == a) | `heat_atten` doubles as emissivity | Unchanged, and now exactly right: `κ` appears identically on the absorption and emission terms in §2.3. |
| Rule 3, contact faces radiation-inert | explicit early-out in the march | **Deleted, and subsumed.** Two adjacent opaque cells exchange nothing through a sweep anyway: the first is opaque, so nothing reaches the second. Conduction still owns contact. One special case gone. |
| Rule 2, mutual pairs at half weight | explicit half-weight branch | **Deleted, and subsumed.** Symmetry is automatic when both cells emit into the same ordinate set. |
| The flux limiter (`RAD_LIM_SHIFT`) | caps each pair term | **Deleted.** It existed to stop a divergent pairwise exchange. A sweep cannot diverge: `I_out ≤ I_in + emitted`. |
| `rad_flux` → unit heat damage | positive-only plane at air cells within `damage_range`, read by `exchange.py` | **Kept as an output of the sweep**: the total intensity passing through a cell, summed over ordinates, is exactly what a unit standing there is exposed to — and it is now physical rather than a range-gated approximation. Consumer unchanged. |
| Weapon beams (laser) | CPU pre-phase sharing the DDA primitive | **Unchanged.** Explicitly out of scope: a beam is a mutating serial pre-pass, not a field solve, and the chapter's split already says so. |
| Flashlights, lamps | per-source `cast_source_directional`, float, per frame | Become emitters in the cascade, and in the rules-side light payload (§4). Cost stops scaling with light count. |
| Fire's own light | render-side blackbody selector, 16-light cap, NMS | Cap and NMS **deleted** — cascades are flat in emitter count, so every burning tile can light the room. |
| Line of sight / infravision | `has_los` binary Bresenham | §4.2. |

---

## 4. Light, in three pieces

### 4.1 For the eye — radiance cascades, GLSL, render-only

Per R-P and R-S: multi-pass fragment shaders on ping-ponged render textures at
GL 3.3, measured before any decision to build a custom raylib at 4.3. Float,
never digested, never read by a rule. Reads the same `dyn_light_atten` and gas
tables the sweep reads, so the two can never disagree about geometry.

### 4.2 For the rules — a scalar integer payload on the heat sweep

The sweep already visits every cell in every ordinate. Carrying one more
payload is arithmetic, not a second traversal:

- Same ordinates, same sweep, same upwind order.
- Different coefficient: light extinction, not heat.
- Scalar, not RGB — the rules need "how lit", not a colour.
- Integer Q16.16, accumulated into a new `light_q` plane, digested.
- Lamps and flashlights contribute an emission source term; a cone emitter
  emits only into the ordinates inside its cone, which is natural in DOM.

This is what the ML chapter already reserves by name: *"a scalar
light-intensity field for stealth and line-of-sight"*. It is not a parallel
system — one traversal, two payloads, shared inputs — but it **is** a second
number for light, and the risk is that what looks dark to the player is not dark
to the rules. §9 carries the gate that keeps them honest.

### 4.3 The stealth query

`vision.can_see` gains a light term, which is the one chokepoint every consumer
already routes through, and `has_los` can migrate from binary Bresenham to the
attenuation-aware form the chapter already specifies. Both were designed and
never built.

---

## 5. What this deletes

Worth listing, because it is the argument that v2 is simpler and not just
different: `T_emit_gate`, `RADIATION_RANGE` and its floor, `fire_ray_count`,
the spatial-hash fan, rule 2's half-weight branch, rule 3's contact early-out,
`RAD_LIM_SHIFT` and the pair budget, `range_base`/`range_per_intensity`,
the 16-light cap and its NMS, `heat_cull`/`light_cull`, and `rad_scale` (replaced
by a derived calibration). Eleven dials and two special-case branches.

---

## 6. Calibration — derived, not fitted

From survey §9.3, the reach curve follows from view factor and the real ignition
threshold for wood (10–12 kW/m²): a crate-sized 150 kW source ignites at ~1.1 m
(≈3 tiles), a 1 MW room fire at ~2.8 m (≈8 tiles). The single free constant is
the counts-per-watt scale that replaces `rad_scale`, fixed by requiring the
flux at 3 tiles from a crate at its measured plateau to equal the ignition
threshold. Everything else follows.

The **reach-curve bench** (survey §9) is the instrument and the regression gate:
probes at r = 1, 2, 3, 5, 8, 14, 30, logging temperature and ignition time,
one CSV per configuration.

---

## 7. Determinism

Doors 1 and 2 only. Integer sweep, `E°` baked once at load, per-cell
coefficients quantised at the material-table boundary. **No `exp`, no `pow`, no
RNG** — neither candidate algorithm needs a random number, so ingress door 4 is
untouched and Philox stays a swarm-units concern.

The CUDA twin parallelises across ordinates, accumulating with integer
`atomicAdd` (order-free, §2.3). Within an ordinate the sweep is a wavefront; the
KBA decomposition is the known-good pattern if a single-ordinate-per-block
launch proves too coarse. The float ratchet is untouched because the sweep adds
no float to a sim TU.

---

## 8. Cost

| | Work per tick | Scales with |
|---|---|---|
| Heat sweep, 128×256, S16 | 524 k cell-updates | grid × ordinates |
| Heat sweep, 256×512, S16 | 2.1 M cell-updates | grid × ordinates |
| Light cascades, tile res | ~0.25–0.5 ms | grid only |
| Light cascades, full map 2× sub-tile | ~3.8–7.7 ms | grid only |

Nothing on this table scales with the number of burning tiles or light sources,
which was the point. Against a 41.67 ms tick at 24 Hz, with the caveat that the
atmosphere group already measured 18.97 ms at a comparable grid (survey §5.4,
Q14 still unanswered).

---

## 9. Gates

1. **Conservation**, the headline: `Σ ΔE_mat + Σ sky == 0` exactly in int64, on
   randomised grids, every tick. This is the gate that proves §2.3.
2. **Equilibrium**: a uniform-temperature region exchanges exactly zero. Second-law
   safety, and it must hold *bit-exactly*, not approximately.
3. **Reach curve**: the §6 bench, as a regression gate with the derived numbers.
4. **CPU↔GPU bit-identity**, tol 0, per the existing CUDA harness convention.
5. **Non-vacuousness on every one of the above** — after this session's sweep,
   each gate asserts that the thing it measures actually happened. A conservation
   test over a scene where nothing radiates passes trivially.
6. **Eye-versus-rules agreement**: the cascade field and the `light_q` payload
   must agree within tolerance on a real scene, so what looks dark is dark.

---

## 10. Migration

The old raycaster has customers beyond fire (survey §10.2). Order:

1. Build the sweep beside the existing path, gated, writing to a shadow plane.
2. Prove gates 1, 2, 4 on the shadow.
3. Calibrate (§6), prove gate 3.
4. Flip heat to the sweep; delete the emission march, the pair rules, the
   limiter, the dials in §5. `rad_flux` consumers keep their interface.
5. Cascades in GLSL, fragment-first, measured (R-S).
6. `light_q` payload, then the stealth query.

Weapon beams keep the DDA primitive throughout and are never migrated.

---

## 11. Patch plan (draft, for the critique to attack)

| # | Patch | Gate | Risk |
|---|---|---|---|
| P1 | The sweep, CPU, heat only, shadow plane, S16 no rotation | conservation + equilibrium, exact | **the whole design** |
| P2 | Quadrature rotation + ray-effect measurement | reach curve stability across ticks | medium |
| P3 | Calibration + reach-curve bench | gate 3 with derived numbers | medium |
| P4 | Flip heat over; delete the old emission path and its dials | full suite + golden re-baseline, one deliberate move | HUMAN-TEST |
| P5 | CUDA twin | tol-0 lockstep | mechanical, pattern known |
| P6 | Smoke heat extinction (R-M) + `dyn_heat_atten` (Q1) | shielding bench | feel |
| P7 | Cascades in GLSL, fragment-first | timing on a real map; look | HUMAN-TEST |
| P8 | `light_q` payload + stealth query | eye-vs-rules agreement | design-complete |

P1 is deliberately first and deliberately narrow: if exact integer conservation
does not hold, everything downstream changes, and we want to know in one patch.

---

## 12. What I could not settle — for Erik

1. **Should a unit's body block radiant heat?** (`dyn_heat_atten`, §3). Trivial
   to build, but it is a gameplay ruling: it makes bodies into cover against
   fire, and makes a crowd shade the tiles behind it. Not implied by anything
   already decided.
2. **Ordinate count.** S16 with rotation is my recommendation and the literature
   supports it, but the honest answer is a measurement on a real map, which P2
   produces. If P2 shows striping at ignition range, S32 doubles the cost of the
   cheap half of the system and is affordable.
3. **The tick budget** (survey Q14) is still unrestated for 24 Hz, and the
   atmosphere group already eats 18.97 ms at a comparable grid. This does not
   block P1–P3 but it does decide how much of §8 we can afford.
4. **Does the rules-side light field enter the digest immediately, or ride as a
   render-only shadow until the stealth rules actually use it?** Entering the
   digest is a one-way door — it forces a golden re-baseline and pulls the
   sweep's light payload into the cross-machine contract.

---

## Systems

**Existing canonical systems this design must use.** The ray engine
(`cpp/src/raycaster.*` + `cuda_raycaster.cu`) — the replacement lands *here*,
never beside it · the temperature solver's Pass-1 fold, the only place radiation
becomes temperature · the gas energy seam, mandatory if smoke absorption warms
gas · the energy closure identity — a new channel needs a counter and a term in
one of the four groups, there is no fifth · the face-flux energy step, the
conservative pattern copied in §2.3 · the material table, where every new
optical column is a row · the Q16 boundary modules, the required shape of
`light_q`'s conversion layer · `pack_hover_readout`, the one tile-probe seam ·
`tools/fire_tuning_lab.py`, the instrument · GOLDEN_AGGREGATE re-baseline
discipline · **"Starting a fire"** (new this session): a fire is started by
delivering heat, never by writing `fire` alone.

**New systems this design creates**, with draft rules for CLAUDE.md once they
exist:

- *The directional sweep* (`cpp/src/radiation_sweep.*`). Draft rule: THE
  radiative transport path; one traversal carries every payload (heat, and the
  rules-side light scalar); a new payload is a channel on the sweep, never a
  second traversal.
- *Per-cell extinction* (`heat_atten`/`dyn_heat_atten`, `light_atten`/`dyn_light_atten`).
  Draft rule: every optical interaction — material, unit body, smoke — is a
  per-cell coefficient on these planes and nothing else; the solvers never
  special-case what a cell contains.
- *The exact-integer conservation idiom for transport.* Draft rule: an integer
  leaving the radiation stream is added to the material as the same integer;
  the only escape is the booked sky term.
- *The rules-side light field* (`light_q`). Draft rule: gameplay and RL read
  light through this plane or the per-unit query, never by sampling the render
  cascades; the cascade field is never digested.
