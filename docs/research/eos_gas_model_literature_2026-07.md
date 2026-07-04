# EOS / Ideal-Gas Reframe — Literature & Prior-Art Research

> **RESEARCH REPORT — NOT CANON.** Roadmap Phase 1.1 (`docs/roadmap_2026-07.md`), 2026-07-05.
> Feeds the rung-A/rung-B prototype (1.2) and Erik's decision (1.3). Context:
> `docs/blackbody_smoke_and_rendering_brainstorm.md` §0 (the reframe),
> `docs/architecture/engine/04_atmosphere_and_pressure.md` (current model),
> `docs/architecture/engine/14_determinism_and_number_ingress.md` (the law).

---

## TL;DR

1. The exact model Erik wants — per-tile moles `N` + temperature `T`, `P = nRT/V` derived — has shipped for 20+ years in Space Station 13 (LINDA) on literally a spaceship grid. It is proven gameplay, and its artifact list is public.
2. SS13 moves gas by a *relaxation* rule (a fraction of the neighbour delta), not a rate × dt — that is why it never blows up and never goes negative, at the cost of tick-coupled speed. Breach's fixed-tick lockstep can legally use the same trick as a flux *clamp*.
3. Rung A is mathematically a porous-medium-type degenerate diffusion; with two-point face fluxes + donor-cell (upwind-mobility) weighting it is provably monotone, positivity-preserving, non-oscillating — reservoir-simulation bread and butter.
4. Advect energy `E = c_v·N·T`, never `T` itself; recover `T = E/(c_v·N)`. SS13's heat-capacity-weighted mixing is exactly this. Advecting `T` directly creates/destroys energy when cells of unequal mass mix.
5. Everything needed is `+ − × ÷ compare/min/max/shift` on Q16.16 with int64 intermediates (`mul128_shr` pattern); zero transcendentals, including minmod/superbee limiters if ever wanted. One division per donor cell (composition split + T recovery), guarded by an `N_floor`.
6. The killer risk is the substep cliff: effective diffusivity scales with `T`, so a 16× fireball spike would demand 16× substeps — solved by the LINDA-style donor clamp (`min(rate·dt, frac·ΔN, frac·N_donor)`), unconditionally stable at pinned substeps.
7. The wave→atmosphere mass-transfer coupling must become wave→**energy** transfer, or the reframe re-imports the very mass-buildup artifact it exists to kill.
8. The *look* of an explosion is the expansion-driven advection of soot, not the acoustic wave — Feldman–O'Brien built a SIGGRAPH explosion paper on exactly that claim. Top-down loses buoyancy but keeps expansion push, suction/inrush, breach outrush, and fire-starves-when-N-drains.
9. Recommendation: **rung A**, with B-shaped bones (fields chosen so momentum can be added later). Rung B forces the whole gas system onto the acoustic CFL or an implicit Poisson solve per substep — a rewrite that competes with the ML/weapons roadmap for months, for gains (true inertia, mushroom curl) that are mostly invisible top-down at 64–512².
10. Literature is thin exactly where we operate (coarse grids, huge dt, game feel); SS13/ONI practice + reservoir-simulation monotonicity theory are the two solid legs. Flagged honestly below where evidence is community-reverse-engineered.

---

## 1. Prior art in games

### 1.1 Space Station 13 — LINDA (the direct ancestor of rung A)

The closest shipped system to the proposed reframe, running since the 2000s on tile grids of a
spaceship. Primary sources: the maintainer-written technical doc
[Atmospherics.md](https://github.com/tgstation/tgstation/blob/master/code/modules/atmospherics/Atmospherics.md)
in the tgstation repo, the code itself
([gas_mixture.dm](https://github.com/tgstation/tgstation/blob/master/code/modules/atmospherics/gasmixtures/gas_mixture.dm)),
and the [player-facing guide](https://wiki.tgstation13.org/Guide_to_Atmospherics).

**State per tile** — exactly Erik's proposal: a gas mixture holding per-species **moles** (a list
of `[moles, archived_moles, …]` per gas type), one shared **temperature** in Kelvin, and a fixed
volume. Pressure is *derived*, never stored: `P = moles·R·T / V` (kPa). Heat capacity is derived
too: `Σ moles_i · specific_heat_i`.

**Transport** — no momentum, no wind field. Each *active* tile `share()`s with its neighbours:
the moved amount is the **mole difference × a coefficient ≈ 1/(neighbour_count+1)**, computed on
*archived* (start-of-tick) values so update order cannot double-move gas:

```
delta = QUANTIZE(gas[ARCHIVE] − sharer_gas[ARCHIVE])
delta *= our_coeff or sharer_coeff        // ~1/(adjacent_turfs+1)
```

Three properties to steal: (a) it is a **relaxation toward the neighbour mean, not a physical
rate** — moving a fraction of the *difference* can never overshoot or go negative, at any tick
rate (unconditional stability by construction); (b) **archived/double-buffered reads** kill
order dependence; (c) moles are **QUANTIZE**d and any species at ≤ 0 after quantization is
garbage-collected — SS13 independently discovered that float dust must be snapped to a grid,
which is Q16.16 vindication from an unexpected direction.

**Temperature on mixing** — heat-capacity-weighted (see §3 for why this is the correct rule):

```
temperature = (old_self_heat_capacity·T
               − heat_capacity_to_sharer·T_archived
               + heat_capacity_from_sharer·sharer.T_archived) / new_self_heat_capacity
```

**Performance model** — the interesting part is what they *had* to build: tiles **sleep** unless
a `compare()` against thresholds (`MINIMUM_MOLES_DELTA_TO_MOVE`, a relative ratio test, a
temperature delta) says they differ from neighbours; active tiles cluster into **excited
groups** which, when activity decays, run `self_breakdown()` — *instant equalization of the
whole group* — because pure neighbour-relaxation converges exponentially slowly and would
otherwise leave rooms asymptotically almost-equal forever. Their documented artifact list
(same doc): 900+-tile groups lag the server when they breakdown-equalize at once; tiles
flicker asleep/awake at threshold edges; sealed-room heat keeps groups awake with no net flow.
The design creed, verbatim: *"performance and gameplay are much more important than realism."*
Breach note: sleeping/excited groups are SS13's answer to a single-threaded interpreter; a CUDA
dense sweep at 512² does not need them, but the *self_breakdown lesson* — relaxation stalls on
the last 1% — still applies (Gauss–Seidel diffusion already covers that role in breach).

**Monstermos / auxmos (the decompression upgrade)** — a later family of equalizers
([auxmos](https://github.com/Putnam3145/auxmos), Rust, powers modern tgstation) that adds
flood-fill-based fast pressure equalization and **space wind**: pressure differentials exert
force on objects/players (explosive decompression throws you out the breach). Detailed algorithm
docs are thin (acknowledged honestly: it is under-documented outside the source); one artifact
is on record from its own maintainer — **directional anisotropy**, "it prefers to go left and
right rather than up or down," because the adjacency bitfield orders neighbours
([auxmos README/known issues](https://github.com/Putnam3145/auxmos)). Space Station 14 rebuilt
the same idea ("zumos") and reports it as the moment "gases move like gases instead of thick
slime" — decompression became the game's signature moment
([SS14 progress report #24](https://spacestation14.com/post/20-10-16-progress-report-24/)).
SS14's forward roadmap doubles down on the same physics principles rung A proposes:
conservation of energy and matter, and *"scale transfer based on pressure differences"* instead
of hard rate limits ([SS14 atmos roadmap](https://docs.spacestation14.com/en/space-station-14/departments/atmos.html),
[rework proposal](https://docs.spacestation14.com/en/space-station-14/departments/atmos/proposals/atmos-rework.html)).

### 1.2 ZAS — the room-graph shortcut (what we are *not* doing, and why it exists)

Baystation-family SS13 servers use ZAS: flood-fill connected air into **zones** that equalize
as single volumes, with edges only at doors/breaches
([Aurora guide](https://wiki.aurorastation.org/index.php/Guide_to_Atmospherics)). Cheap, and
pressure changes are "fast" for free — but the wiki itself names the artifact: *a canister
opened in a room floods the entire room instantly* — no gradients, no travel time, no wind
texture inside a room. RimWorld's temperature is the same idea (one temperature per room,
equalization through walls/vents proportional to perimeter —
[RimWorld wiki](https://rimworldwiki.com/wiki/Temperature)); Barotrauma likewise keeps one air
state per hull compartment (per-room oxygen/pressure; water level is the sub-tile star —
[FakeFishGames discussion](https://github.com/FakeFishGames/Barotrauma/discussions/7235)).
Breach's whole identity (per-tile wind driving smoke through corridors) rules this class out;
it is listed because it marks the *coarse* end of the fidelity spectrum and because ZAS-style
instant-room-flood is the artifact rung A must beat to justify itself.

### 1.3 Oxygen Not Included — mass + temperature per cell, no wind

ONI simulates per-cell **element + mass (grams) + temperature**, with a hard
[one-element-per-cell rule](https://oxygennotincluded.fandom.com/wiki/One_element_per_cell_rule)
(a foundational simplification: mixtures never exist; gases displace or swap whole cells).
Movement is local displacement driven by mass/pressure differences and per-element
molar-mass-as-density ordering (light gases rise over heavy ones); heated gas rises, cold
sinks, similar temperatures spread semi-randomly sideways
([Gas](https://oxygennotincluded.wiki.gg/wiki/Gas),
[Fluid Mechanics](https://oxygennotincluded.wiki.gg/wiki/Fluid_Mechanics)). The wiki is blunt
about the consequence: *"the game cannot model gas pressure [gradients as flow], wind, or
turbulent liquid flow"* — pressure equalizes slowly, "one square at a time," which players
experience as gas being bottlenecked. There is no official sim writeup (the internals are
community-reverse-engineered — flagged); design-level dev material confirms the ambition was
"the whole game running on a broad simulation" of temperature/pressure/chemistry
([Game Developer interview](https://www.gamedeveloper.com/design/behind-the-design-of-hit-sim-game-i-oxygen-not-included-i-)).

**ONI's documented artifact list is a warning catalogue for us**
([Hidden Mechanics](https://oxygennotincluded.wiki.gg/wiki/Hidden_Mechanics),
[Gas](https://oxygennotincluded.wiki.gg/wiki/Gas)): sub-microgram gas is **deleted**; sub-1 g
packets adjacent to ≥1 kg packets of another element are absorbed-deleted (mass conservation
violated at the small end — our `N_floor`/quantum rule must be designed, not accidental);
buildings that emit at fixed output temperature **delete/create heat** wholesale; the wiki's own
words: the simulation *"only loosely respects conservation of energy."* ONI gets away with it
because nothing downstream audits totals; breach's determinism law (int64 exact sums, golden
digests) means we cannot and should not.

### 1.4 The rest of the spectrum, one line each

- **Dwarf Fortress** — per-tile temperature (16-bit integer °Urist!) updated every tick; the
  wiki's performance page reports **~2× FPS from turning temperature off**, and that disabling
  it breaks fire/melting semantics ([Maximizing framerate](https://dwarffortresswiki.org/index.php/Maximizing_framerate),
  [v0.34:Temperature](https://dwarffortresswiki.org/index.php/v0.34:Temperature)). Lesson:
  integer per-tile T is old wisdom; naive always-on cost is real (breach's answer is CUDA, not sleep lists).
- **Noita** — falling-sand CA: fire/smoke/steam are per-pixel materials with local rules, the
  spectacle is emergence not fields ([GDC: Exploring the Tech and Design of Noita](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design)).
  Orthogonal machinery; relevant only to §5 (what reads as fire).
- **Factorio** — pipes moved fluid by pressure-difference rates and the devs documented the
  failure modes we must design against: **endless sloshing back and forth**, update-order-
  dependent flow, junctions starving consumers ([FFF-260](https://factorio.com/blog/post/fff-260));
  Fluids 2.0 eventually **abandoned** distributed flow for segment-pooling, explicitly trading
  realism away ([FFF-416](https://factorio.com/blog/post/fff-416)). Their oscillation came from
  a *momentum-ish* speed variable + sequential updates — rung A has neither (potential-driven,
  double-buffered), which is precisely why it should not slosh (§2).

---

## 2. Numerical scheme for rung A

### 2.1 What the equations actually are

Write the flux two ways — this choice matters more than anything else in rung A:

- **(i) Fick-in-P:** `F = −k∇P`, so `∂N/∂t = k∇²(C·N·T)`. With uniform `T` this is *linear*
  diffusion of `N` with `D_eff = k·C·T`. Simple, but the flux does **not** vanish as the donor
  empties — it will happily drive `N` negative and must be clamped everywhere.
- **(ii) Darcy mass flux (recommended):** velocity `u = −k∇P`, mass flux `F = N·u = −k·N·∇P`,
  so `∂N/∂t = ∇·(k N ∇(C·N·T))`. With uniform `T` this is `∂N/∂t = (k·C·T/2)·∇²(N²)` — the
  **porous medium equation** (PME, exponent m=2): a *degenerate* parabolic equation whose flux
  vanishes as `N→0`. Consequences, all desirable: natural positivity (empty cells cannot give),
  **finite propagation speed** — a vacuum front (breach!) moves at finite speed instead of
  instantaneously smearing, and well-posedness is classical. Explicit finite-difference schemes
  for generalized PMEs with interface tracking are established literature
  ([M2AN 2016](https://www.esaim-m2an.org/articles/m2an/ref/2016/04/m2an150067/m2an150067.html)).

### 2.2 Does it oscillate or checkerboard? No — if the flux is face-based and upwinded

- The spatial operator is a **two-point flux approximation (TPFA)**: per face,
  `F_face = k_face·(P_i − P_j)`, with the transported `N` (the "mobility") taken from the
  **donor** (upwind, higher-P) cell. TPFA is formally **monotone** on orthogonal grids, and
  upwind-mobility weighting is the standard positivity-preserving choice in reservoir
  simulation — schemes of exactly this shape are proven to keep saturations/densities in
  physical bounds ([positivity-preserving FV for compressible two-phase Darcy flow](https://www.sciencedirect.com/science/article/abs/pii/S0021999120300073),
  [DDFV variant](https://www.sciencedirect.com/science/article/abs/pii/S0898122125002524),
  [implicit hybrid upwinding, monotone w/ buoyancy](https://www.sciencedirect.com/science/article/abs/pii/S0045782516308970)).
  A diffusion-type stencil with positive face coefficients admits a discrete maximum principle:
  no new extrema, **no checkerboard**.
- The checkerboard danger enters only through the **back door**: computing a cell-centred wind
  `u = −∇P` by central differences and then advecting `N` with *that* — the classic collocated
  pressure–velocity decoupling. Breach has already paid this tuition once: the smoke
  central-difference checkerboard documented in `04_atmosphere_and_pressure.md` §4. Rule:
  **`N` (and `E`) move by the face fluxes themselves; the cell-centred wind is derived only for
  downstream consumers** (smoke SL advection, units, render), exactly as today.
- Factorio's sloshing (§1.4) required a momentum/speed state variable that could overshoot the
  equilibrium and ring; a pure potential-driven flux with donor clamps is gradient descent on a
  Lyapunov function — it can creep, it cannot slosh. (Their order-dependence artifact is killed
  by breach's existing double-buffer discipline, same as LINDA's archived values.)

### 2.3 Explicit stability, the substep cliff, and the LINDA clamp

Linearizing (ii): `D_eff ≈ k·C·(N·∂P/∂N)/N·…` — order `k·C·T` (and up to `2kCT` through the
nonlinearity). The explicit monotonicity bound is the one already derived in ch.04 §2.2:
`μ = D_eff·dt/h² ≤ 1/8` for no-sign-flip on the checkerboard mode. **The trap: `D_eff` scales
with `T`.** A fireball at `T = 16×` ambient demands ~16× the substeps of quiescent air — a
data-dependent substep count, which is exactly the "substep-count cliff" the fixed-point arc
banned (pinned integer substep counts, `fixed_point_migration_lessons`).

The resolution comes straight from LINDA (§1.1): make the per-face move the **minimum of the
physical rate and a relaxation cap**:

```
M_face = min( k·(P_don − P_rec)·dt,          # physical Darcy rate
              λ_max·(N_don − N_rec),          # LINDA-style: fraction of the difference
              frac_max·N_don )                # donor-deplete cap (positivity)
```

with `λ_max ≤ 1/5` (4 faces + self, the 1/(n+1) share factor) and `frac_max ≤ 1/4` so four
faces cannot overdraw a donor even in the worst corner. In the gentle regime the physical rate
is active (tunable, physical feel); at a blast front the caps take over and the scheme degrades
gracefully into SS13-style relaxation — **unconditionally stable and positive at any pinned
substep count**. First-order donor-cell advection under its CFL is a convex combination
`u_i ← (1−c)u_i + c·u_{i−1}` — positivity and no overshoot are immediate
([upwind scheme](https://en.wikipedia.org/wiki/Upwind_scheme)); the caps extend the same convex-
combination guarantee to the capped regime. Cost of the caps: blast-front equalization speed
becomes tick-rate-coupled rather than physical — acceptable in a fixed-tick lockstep engine
(30–60 tps is a constant of the world, like `h`), and only active in the extreme regime.
Numerical diffusion of donor-cell is real but is the *tolerable* artifact here — the fields it
smears (bulk `N`, `E`) are slow movers; the visually sharp movers (smoke slices) keep their
existing integer semi-Lagrangian transport ridden on the derived wind.

- **Upwind vs central for N:** donor-cell (upwind). Central is the checkerboard/oscillation
  path (§2.2 and breach's own smoke history) and offers nothing at these Péclet numbers.
- **Substeps:** keep the pinned-integer discipline: choose the substep count offline from the
  *worst-case* `k·C·N_max·T_max` bound with the caps as the safety net; never derive it from
  live field values.

## 3. Energy accounting

**Advect `E = c_v·N·T` (an extensive density), never `T` (intensive).** Recover
`T = E/(c_v·N)` pointwise. Three arguments:

1. **Conservation-correctness.** `T` has no conservation law; naive SL/blend advection of `T`
   creates or destroys energy whenever mixing parcels have unequal mass (a tiny hot wisp
   entering a dense cold cell must barely warm it; mass-blind interpolation warms it a lot).
   `E` moved with the mass flux is conservative to the last bit: per face,
   `E_moved = M_face · (E/N)_donor` — donor specific energy times moved mass. Integer sums of
   `E` and `N` are then exactly conserved (int64 accumulate), auditable in tests.
2. **The mixing rule is automatic.** Two parcels combining yield
   `T_mix = (c₁N₁T₁ + c₂N₂T₂)/(c₁N₁ + c₂N₂)` — the heat-capacity-weighted mean. This is
   *precisely* SS13's shipped `share()` temperature update (§1.1 formula: heat capacity moved
   in/out, divided by new total capacity) — they store `T` but do the bookkeeping in energy
   terms, which is the same thing with more divisions. ONI, by contrast, plays loose with
   energy at its edges (fixed-output-temperature buildings, §1.3) and wears the resulting
   exploit catalogue; its cell *mixing* when packets combine is likewise mass/SHC-weighted
   (community-documented). Store `E`, and the mixing rule stops being code — it is just
   addition.
3. **Dalton composes cleanly.** With per-species `N_i` sharing one `T` (brainstorm §0.2), heat
   capacity is `c_v,mix·N_tot = Σ c_v,i·N_i` — v1 can set all `c_v,i = 1` and add the species
   table later without changing the scheme. Combustion/explosions **deposit energy into `E`**
   (Erik's "add energy, not phantom mass"); adiabatic-expansion cooling appears in rung B
   naturally (the `−P∇·u` work term needs `u`); in rung A the honest statement is: expansion
   cooling must be *approximated* (an optional `−α·P·(net outflux/N)` sink) or deferred — the
   fireball cooling arc in rung A otherwise comes from radiation/decay, not thermodynamic work.
   Flagged as a rung-A limitation, not hidden.

## 4. Determinism / integer feasibility

**Operation inventory of the full §6 scheme — nothing beyond the sanctioned set:**

| Step | Ops |
|---|---|
| `P = C·(Σ N_i)·T` | int64 mul, shift (`mul128_shr`/`mul_q16` ×2) |
| Face flux + caps | sub, mul, `min` (compares) |
| Donor composition split `M_i = M·N_i/N_tot` | one `recip`/div per **donor cell** per substep, then muls |
| Energy move `E_moved = M·(E/N)_donor` | same reciprocal, one mul |
| `T = E/(c_v·N_tot)` with `N_floor` guard | one div (or `recip_mul`), one compare |
| Wind `u = −∇P` (downstream only) | sub, shift |
| Optional TVD upgrade (minmod/superbee) | pure `max/min/compare/mul` — **no division** in minmod/superbee/MC themselves ([flux limiter](https://en.wikipedia.org/wiki/Flux_limiter)); implement on slope *differences* to avoid even the ratio `r` |
| Offline CFL/substep derivation | `sqrt` at most — config-time, not runtime |

Zero transcendentals anywhere — confirming the roadmap's expectation. The kit's missing
`exp/log` (`14_…` §7) stays missing: nothing in rung A wants them. (Black-body *rendering*
reads `T` through the existing LUT — render-side, out of scope here.)

**Dynamic-range sketch (Q16.16, int32 fields):**

- Scales: `N = 1.0` (65536) = 1 atm-equivalent density; `T = 1.0` = 300 K ambient. Ranges:
  `N ∈ [0, 8]` (≤ 2¹⁹ raw), `T ∈ [0, 16]` (4800 K, ≤ 2²⁰ raw).
- `N·T ≤ 2³⁹` raw-product → **must** go through the int64 `mul128_shr`/`mul_q16` pattern
  (already the house style); result ≤ 2²³ ≪ 2³¹ — `P` fits int32 Q16.16 with 8 bits headroom.
- `E = c_v·N·T` same envelope as `P` — fits. Face flux ≤ `frac_max·N_don` — fits trivially.
- Per-tile 4-face net: accumulate in int64, write back int32 (house pattern; matches the S5/S6
  order-free `atomicAdd` discipline on CUDA — int64 atomics for any global sums).
- Divisions: `N_tot` reciprocal only where `N_tot ≥ N_floor` (say 2⁻⁸ ≈ 0.004 atm); below the
  floor the cell is *defined* vacuum: fluxes read it as `P = 0` receiver, `T := T_ambient`
  display value, composition split skipped. This is our designed version of SS13's
  QUANTIZE+garbage-collect and ONI's (accidental, exploit-ridden) sub-gram deletion — with one
  difference: **the residue below `N_floor` is kept, not deleted** (moved-mass rounding always
  rounds toward the donor), so int64 mass conservation stays *exact*, not approximate.
- Rounding: per-face moved mass rounds toward zero (donor keeps the dust) — deterministic,
  conservative, and the same convention on CPU and CUDA (`shr_round0_dev` exists).

**CUDA:** the kernel shape is a 2D 5-point stencil + per-face scatter — the S4/S6 patterns
(gather-once ±, integer atomicAdd scatter, source-only deposit) cover it; nothing new is
required for bit-identity. LINDA's archived-values trick *is* the double-buffer discipline the
CUDA port already enforces.

## 5. The look — what actually sells hot-gas expansion top-down

- **The expansion push is the explosion.** The graphics literature says it outright:
  Feldman–O'Brien–Arikan animate explosions by *not* simulating the blast wave — "rather than
  modeling the numerically troublesome, and largely invisible blast wave," they run a stable
  fluid model whose **divergence field is adjusted directly** to model detonation products
  expanding, and let **particles tracking fuel and soot** be advected by the result
  ([ACM TOG 2003](https://dl.acm.org/doi/10.1145/882262.882336),
  [PDF](https://escholarship.org/content/qt4jk7m50m/qt4jk7m50m.pdf)). Translation to breach:
  the *visible* explosion = soot/smoke/debris riding the radial outflow `−∇P` from a `T`-spiked
  region — which is exactly what rung A produces, while `wave_p` keeps carrying the (invisible,
  gameplay-load-bearing) impulse. The two-field split survives the reframe with cleaner roles.
- **Decompression is a rush, not a fade.** SS14/Monstermos made "gases move like *gases*
  instead of thick slime" the headline of their atmos rework, and space wind (pressure
  differential shoving entities) the marquee moment
  ([SS14 PR#24](https://spacestation14.com/post/20-10-16-progress-report-24/)). Rung A's PME
  structure gives the front finite speed (§2.1) — a breach *front* that arrives, not a global
  exponential fade; the existing decompression-suction-on-units plan (ch.04 §5) then reads it.
- **The inrush/suction phase** (Friedlander negative lobe — brainstorm §4 item 5) is the most
  underused realism cue in games — after the push, air flows *back*. Rung A gets it free:
  the over-expanded hot region cools/thins, `P` undershoots ambient, `−∇P` reverses.
- **Buoyancy is out (no gravity axis top-down) — what remains:** expansion push, inrush,
  breach outrush, channeling/diffraction through doors (already the Neumann stencil's gift),
  fire self-starvation (`N` drains from a sealed burning room → `P` high but `N` low → O₂ proxy
  correctly says *die*, brainstorm §0.3), and per-species molar-mass drift (poison pools,
  fuel-gas spreads) as a later garnish. ONI demonstrates temperature/density *readability* —
  players literally read convection — but its vertical convection loops are the one export that
  doesn't survive the top-down projection; the 2.5D z-layer plan (brainstorm §0.5) is where
  buoyancy re-enters later, driven by the same EOS density.
- **Noita's lesson** is about material-ness, not fields: fire spreads *to things* and everything
  reacts ([GDC talk](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design)).
  Breach's equivalent leverage: `T` is real and load-bearing, so the black-body soot/fire
  render plan (`blackbody_smoke_…` §0.3, §6.2 LUT) reads the *same* field the physics uses —
  hot core whitens, expanding shell reddens, drifting soot dims — the whole §0 promise.

## 6. Recommendation

**Adopt rung A ("Darcy-EOS refit"), build it EOS-first with B-shaped bones. Do not build rung B now.**

Rung B (momentum field, unified acoustics, baroclinic curl) is *feasible* under the determinism
law — Kwatra-style semi-implicit compressible flow even shows how to escape the acoustic CFL
with a Poisson solve ([Kwatra et al. 2009](https://purl.stanford.edu/hn238xx1131)), and breach
already runs implicit GS solves in integers — but it is the wrong trade *now*: it forces every
gas field onto either acoustic substeps (`c = 66` ⇒ the whole multi-slice transport at wave
cadence — a large constant-factor cost at 512²) or per-substep Poisson solves; it re-tunes
every downstream consumer at once (wind, smoke, fire, units, goldens); its signature payoffs
(inertial sloshing, mushroom-cap baroclinic curl) are mostly *vertical-plane* phenomena that
top-down 2D cannot show; and it competes for the same months as weapons/units/ML. Rung A keeps
`wave_p`, the wind interface, the smoke/fire consumers, and the CUDA patterns intact — and the
fields it adds (`N` per species, `E`) are exactly the state rung B would need anyway. If the
prototype's rung-A explosions still feel flat, the escape hatch is incremental: add a face
momentum/inertia term to the same flux structure, not a rewrite.

### 6.1 Concrete rung-A discretization (prototype-ready)

**Fields** (all int32 Q16.16 in the eventual port; the Python prototype runs float):
`N_i` per species (reuse `gmap.gas` slices + fold `atmosphere` into `N_air`), `E` (new),
derived per tick: `N_tot`, `P`, `T` (cached for render/fire), `wind_x/y` (unchanged interface).
`wave_p/wave_v/wave_source` untouched.

**Config constants:** `C` (EOS constant; 1.0 with the §4 normalizations), `k` (face
conductance; tune to match today's `d_atm` equalization timescale), `c_v` (1.0 v1),
`N_floor` (2⁻⁸), `λ_max` (1/5), `frac_max` (1/4), `n_substeps` (pinned integer, from the
offline worst-case bound), `T_ambient` (1.0), `wave_to_E_rate` (replaces `wave_transfer`).

**Per substep, in order** (double-buffered reads, LINDA-style archives):

```
1. N_tot = Σ_i N_i ;  T = E / (c_v·N_tot)   [N_floor-guarded]
2. P     = C · N_tot · T                     (per tile; int64 mul chain)
3. per face (x/y sweeps): dP = P_don − P_rec  (donor = higher-P side)
       M  = min(k·dP·dt, λ_max·(N_don−N_rec)⁺, frac_max·N_don) · perm_face
       M_i = M · N_i,don / N_tot,don ;   E_mv = M · E_don / N_tot,don
       scatter ±M_i, ±E_mv                   (int64 accumulate)
4. wave step unchanged; anomaly transfer deposits into E (energy), NOT into N   ← the fix
5. boundaries: breach tiles = P:=0 receivers (fluxes drain N_i and E through the face;
   sponge/relaxation only on wave_p);  sealed hull: perm 0 as today
6. wind = −∇(P + wave_p·β)  → smoke SL advection, fire, units (interface unchanged)
7. sources: fire/explosions add E (and combustion converts N_fuel → N_soot/N_steam)
```

**Prototype (Phase 1.2) probes — one scenario per risk:**

| # | Risk | Probe |
|---|---|---|
| 1 | **Substep cliff / hot-tile stiffness** — `D_eff ∝ T`, 16× spike wants 16× substeps | Grenade `T`-spike ×16 at pinned `n_substeps` ∈ {2,4,8}; assert no overshoot/ring at the front (the caps must engage); plot cap-engagement mask |
| 2 | **Oscillation/checkerboard** | Seed ±ε alternating `N` pattern → must decay monotonically; vacuum-step response → front monotone, no undershoot below 0 |
| 3 | **Mass/energy books** | int64 `ΣN_i`, `ΣE` exactly constant in sealed map over 10⁴ ticks; breach map: totals decrease monotonically, never increase (the repeated-grenade *mass-buildup regression test* — the artifact this arc exists to kill) |
| 4 | **Feel regression of wind** | Side-by-side wind/smoke on the standard scenarios (corridor, breach, grenade) old vs new; tune `k`, `C` to match today's equalization timescale before judging looks |
| 5 | **Wave coupling double-count** | Grenade with `wave_to_E` on/off: `P` must return to baseline (no permanent bump) yet the lasting-wind feel must survive via the `E` bump — if it doesn't, revisit whether a small direct N-redistribution (not injection) is needed |
| 6 | **Q16.16 dynamic range** (port-time) | Fuzz `N,T` at range corners through the int64 chains vs float64 reference; assert `P/E` never exceed 2²⁴ raw×65536 envelope |

**Honesty about thin literature:** nobody publishes "coarse-grid gas EOS at 30 tps for
lockstep determinism." §2's stability/positivity claims stand on reservoir-simulation theory
(solid, but implicit-scheme-flavored and continuous-time) plus SS13's two decades of empirical
tile-atmos practice (solid, but undocumented in the academic sense — key sources are a repo
markdown and the code). The bridge between them — the `min()` cap hybrid at pinned substeps —
is *our* synthesis; it inherits a convexity argument, not a citation. That is what probe #1/#2
exist to earn. ONI internals are community-reverse-engineered; Monstermos's equalizer is
under-documented beyond its source and its maintainer's own anisotropy note. Where this report
says "proven," it means proven-shipped or proven-in-paper, and says which.

---

## Addendum — Erik's review corrections (2026-07-05, design-canon)

1. **Motive, corrected.** The reframe's motive is (a) more interesting/beautiful
   explosions and fire ("let the air expand where it's hot") and (b) opening the
   door to chemistry later. The wave→mass phantom-injection issue (§TL;DR-7,
   probe #3) is REAL but is an *engineering consequence of adopting conservation*,
   not the arc's purpose — this report over-weighted it. The wave→E re-plumb and
   the sealed-room ΣN regression test stay in the spec on engineering grounds.
2. **In-plane expansion payoff, re-ranked.** "Buoyancy is out" is true only of
   the vertical convection cell. Thermal-expansion outflow — smoke drifting away
   from fires, fires self-starving as they push O2 away, backdraft inrush on
   cooling, temperature-enriched drafts through doors/breaches — is rung A's
   CENTRAL payoff in the x-y plane and should headline any pitch of this design.
   The genuine rung-B-only residue in-plane: persistent rotational flow (eddies);
   the wispy look layers render-side (curl-noise, per the black-body plan).
3. **Process:** the pressure update gets its OWN design session after the
   Phase-1.2 prototype; this report + prototype probes are decision inputs only.
