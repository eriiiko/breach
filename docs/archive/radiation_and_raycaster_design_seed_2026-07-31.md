# SEED — radiation law + raycaster design pass (2026-07-31)

**For: a Fable design session.** Written by the Opus session that built the thermal-mass
arc and got the first crate burning. Self-contained.

**Deliverable:** a ruling doc in the style of `docs/thermal_mass_eos_ruling_2026-07-30.md`
— answers with reasons, a patch spec, gates, escalation triggers. Opus executes from it.

**This supersedes Q2 of `docs/fire_model_design_seed_2026-07-30.md`** (the plume-shim
question), which Erik has now ruled on — see §3. Q1 of that doc is built; Q3 is folded into
§4 here with hard numbers it did not have.

---

## 0. Where the engine is (all on branch `thermal-mass-axis`, pushed, unmerged)

**The fire burns.** After a week of removing limiters, measured 2026-07-30 at
`fire_T_ext 180 / span 40 / k_fire_heat 33 / ignition_seed 0.12 / k_grow 3.5 / k_die 0.28 /
cool_shift 9` (runtime overrides — `config.toml` is untouched, one hunk for the whole arc):

| | measured |
|---|---|
| peak I | 0.166 @ 15.1 s |
| crate T | max 336 game = **966 K** |
| fire death | 334 s (**5.57 min**) |
| `wall_hp` | 26.7 / 30 — charred remains |
| `hot` | 1.000 for essentially the whole burn |

Erik: *"this looks pretty good I have to say… I am happy with max intensity 20% off target."*

Six limiters were removed to get here (thermal advection stealing the crate's temperature;
one global cooling e-fold; the O₂ law saturating at ambient; the fuel fraction normalising
against wood's hp; the seed/`I_sustain` relation; the Q16.16 growth quantum). The governing
algebra is in `docs/fire_tuning_session_seed_2026-07-30.md` and the arc's bench report.

**Read for background:** `docs/fire_constants_audit_2026-07-30.md` (exhaustive constants
audit), `docs/thermal_mass_eos_ruling_2026-07-30.md` (the ownership rule + the method
lesson), `docs/fire_tuning_plan_2026-07-22.md` §9 (the fire chain).

---

## 1. Q1 — Should heat transport become a temperature-driven radiation law?

Today: 8 rays per burning tile deposit `k_fire_heat · I`, range
`range_base + range_per_intensity · I`. The magnitude is a **free parameter**, unanchored to
anything physical.

Erik: *"I wonder if there is some way that we can make the radiation stay realistic. We only
tune the temperature, and radiation then falls out automatically?"*

This is now attractive precisely **because the arc made object temperature real and owned**
(`temperature[]` on a `thermal_solid` tile belongs to the TemperatureSolver; everything else
reads it). Radiation could be derived from it rather than dialled.

### 1.1 The physics, with your geometry

Stefan–Boltzmann: `P = ε σ A T⁴`, σ = 5.67×10⁻⁸ W m⁻² K⁻⁴. Net between two surfaces:

```
P_net = ε σ A F (T_hot⁴ − T_cold⁴)
```

Tile pitch 0.333 m (per-level), `ceiling_h = 2.5 m` ⇒ vertical face **0.833 m²**, four of
them. Temperature mapping is `K = 293 + 2·T_game`.

**Sanity check (done, it passes):** at T_game 400 = 1093 K, `εσT⁴ ≈ 73 kW/m²` with ε = 0.9.
Real flame radiant fluxes are 20–100 kW/m². Critical heat flux for piloted ignition of wood
is ~12 kW/m²; at a neighbour view factor ~0.3 you deliver ~22 kW/m². **Adjacent crates
igniting each other falls out of the physics rather than needing a spread dial.**

### 1.2 ★ The antisymmetry requirement — Erik caught this, and it is load-bearing

Erik: *"if a fire sends a ray at a fire on the tile next to it, both are burning hot — they
enter a divergent feedback loop unless we really honor this formula. Just replacing T_cold
with ambient temp seems dangerous if we have two strong fires next to each other?"*

**Correct, and it is worse than a tuning risk — it is energy created from nothing.** A
one-way `σT⁴` deposit means two hot tiles each gain and neither loses; they climb until they
hit `T_MAX_PHYS`. The net form is **antisymmetric**: what A loses, B gains exactly, so the
pair is driven toward equality and divergence is impossible by construction. **The net form
is required, not preferred.**

### 1.3 Why it should be cheap — the engine already has this shape

**Conduction is already a conservative antisymmetric pairwise exchange** (`dT ∝ T_i − T_j`
across a face, baked `face_shift` tables, pure signed-add + arithmetic shift, deterministic).
Radiation with net T⁴ is structurally the same pass with `T⁴` as the potential instead of `T`.

Suggested shape (decide it, do not treat as spec):
1. **Bake `E(T) = εσT⁴` as an integer lookup table at load** — the `face_shift_table` /
   `heat_inv_shift` pattern. Necessary anyway: T⁴ at 3000 K is ~8×10¹³ and cannot live in
   Q16.16, so a runtime `T⁴` is not an option.
2. Runtime `net = F · (E(T_hot) − E(T_cold))` — two lookups, one subtract, one multiply.
   Integer, deterministic, no libm.
3. **Deposit into `heat[]`, not `temperature[]`** — negative on the emitter, positive on the
   receiver. This preserves the ownership rule AND gives the emission-physical /
   absorption-lumped split for free, since each tile converts heat→T through its own
   `heat_inv_shift`. Two materials exchanging equal *energy* then get different ΔT, which is
   physically right.

### 1.4 ★ The absorption side must stay lumped — do not "physicalise" both ends

`heat_inv_shift` **stays** under every design. It is the absorption side, and it is a
deliberate lumped stand-in for the fact that fire heats a thin surface layer, not the bulk:

- Tile volume 0.333 × 0.333 × 2.5 = 0.277 m³; wood at 500 kg/m³ = **139 kg**; c ≈ 1700
  J/(kg·K) ⇒ C ≈ **236 kJ/K**. Absorbing ~17 kW gives **0.07 K/s** — nothing would ever ignite.
- Thermal penetration `δ ≈ √(αt)`, wood α ≈ 1.1×10⁻⁷ m²/s ⇒ after a full minute δ ≈ **2.6 mm**,
  ~1 kg, C ≈ 1.8 kJ/K ⇒ ~9 K/s. Plausible.

That ~130× gap is exactly why `thermal_mass = 8` is a **tuned lumped parameter, not a
physical J/K**, and must remain one.

### 1.5 The three difficulties to design for

1. **Antisymmetry across ray pairs.** Rays currently deposit one-way from the source; net
   exchange must apply to both ends, which on CUDA is a scatter (atomics, or reformulate as
   a gather). The batched raycaster already enumerates the pairs.
2. **The stability bound becomes temperature-dependent.** `d(T⁴)/dT = 4T³`, so the effective
   exchange coefficient grows as T³. Conduction is stable via a *constant* convex bound
   (`SHIFT_AT_REF = 2`; "4 faces × 1/4 = 1"). Radiation needs a **flux limiter** — clamp the
   per-tick net transfer to a fraction of what would equalise the pair. Conservative and
   stable, but it must be designed in.
3. **It subsumes existing paths.** `k_fire_heat` stops being a free parameter. Decide what
   survives, or there will be three heat currencies instead of two.

**Feasibility is an explicit deliverable.** Erik: *"especially if it's doable on a similar
computation budget."* Estimate the per-tick cost against today's `k_fire_heat·I` deposit and
say plainly whether it fits. Do not assume.

**Credit the source** (project rule): any file implementing this carries an author + paper
citation; archive the paper under `docs/papers/`.

## 2. Q2 — The raycaster: is the algorithm right, and where is the low-hanging fruit?

Erik: *"we'll touch the raycaster, question is how much. Basically I want to know if the
current algorithm is good — and if we're doing it the best we can, and if we need to
optimize it more or not. I know the spawning of additional ray casting sources is not really
working great — but I don't know why, perhaps Python is involved in it."*

**He is right, and the codebase already knows.** From the docstring at
`src/simulation/physics_runner.py:1034`:

> *"The cluster dial is still bound on the raycaster for when the source build moves into C++."*

**Measured facts:**
- `physics_runner.py:1097-1130` builds **one `bp.LightSource()` per burning tile, per tick,
  in a Python loop**, setting ~10 attributes each (`x`, `y`, `max_range`, `ray_count`,
  `angle_spread`, `angle_center`, `intensity`, `heat`, `jitter`, `color`). At 600 fires that
  is ~6000 pybind attribute writes per tick, plus `np.nonzero(...).tolist()` materialising
  Python lists.
- **S8c (`9eb47c0`) fixed the *transfer*, not the *construction*.**
  `cuda_raycaster_cast_batch` collapsed per-source device round-trips (600 fires
  424 ms → 1.5 ms, 277×, heat byte-identical). The Python source-build was left in place.
- **Every per-source parameter is a pure function of `(x, y, intensity)`** — including
  `angle_center = ((x·7 + y·13) mod ray_count)·(2π/ray_count)`, deliberately deterministic
  and non-random. So the whole list is computable C++-side from the fire plane.
- **`Raycaster::update_from_fire` exists and has no production caller** (bound at
  `bindings.cpp:1660`, defined `raycaster.cpp:463`) — a whole-plane C++ cast, but the
  **legacy intensity-only API**: no heat channel, no multi-gas march, no `heat_atten`. So it
  cannot be used as-is; the modern signature is what would need the equivalent.
- **`coarse_cluster = 3` is dead** for the same reason — it only ever applied inside that
  orphaned path. (Both flagged in `docs/fire_constants_audit_2026-07-30.md`.)

**Decide:** does the source build move into C++ (a `cast_from_fire_plane` taking the fire
plane + dials, replacing the Python loop)? Does `coarse_cluster` come back to life with it,
and is clustering even wanted — the current cost discipline is "many sources × few short
rays", per `fire_design_notes`? And **if Q1 lands, does the source concept survive at all**,
since a net T⁴ exchange is pairwise over tile pairs rather than one-way from sources?

**Constraint:** whatever changes, `heat` must stay **bit-identical or deliberately
re-baselined**. Source order is fixed row-major and the deposit is an order-free saturating
integer add — those two properties are what make the batched and CUDA paths safe. Do not
break them.

**Also worth a look while in there** (Erik: *"if we are touching the raycaster, we might as
well see if there are any low hanging fruits"*): the 8-ray fixed fan, the per-source
`max_range` cost model, and whether the render and sim casts can share work.

## 3. ★ RULED — the plume→T shim is REMOVED, not fixed

Erik, 2026-07-31: *"I do not think our radiation will heat its own tile to be honest."*

**This is correct physics and it closes the question.** A burning tile is hot because
**combustion releases energy in it** (the `H_fuel` deposit through `heat[]`), not because it
radiates to itself. Self-radiation is not a real transport term.

So `fire_simulation.cpp:265-293` (+ `cuda_fire.cu:239-259`) — which writes `temperature[]`
**directly, bypassing `heat_inv_shift`** — should be **deleted**, with its heat accounted by
combustion and (under Q1) by radiation from neighbours. It was the 7th `temperature[]`
writer found by P-EOS's enumeration and the one violation of the ownership rule.

**Measured, so the design knows what removing it costs:**
- It is a flat **+6.3%** on `T*` — and it closes to three decimals: predicted 1.0652 at
  T = 280 and 1.0612 at T = 383.5 vs measured 1.066 and 1.060 across 233 quasi-equilibrium
  samples. `T* = 1.063 · gain · I` today.
- **5.6–6.1%** of the crate's deposit at the current dials (was 0.121% at `k_fire_heat` 1600
  — it rose ~48× as `k_fire_heat` came down, exactly the risk flagged earlier).
- On **steel** (`thermal_mass` 32) it would be **19.0%**, because it bypasses the per-tile shift.
- `temp_gain_scale = 50` and `T_FLAME_MAX = 2000` exist **only as C++ defaults** — absent
  from `config.toml`, never bound. `T_FLAME_MAX` = 4293 K, far above the 400–500 game
  operating band, so its taper **never engages**: the shim is an untunable near-linear deposit.

**Removing it drops `T*` by ~6%**, which Erik re-tunes with `k_fire_heat`. Confirm the
number before and after.

## 4. Q3 — The extinction shape: two hard walls, both now measured

Erik: *"extinction is nice and we want it — we want it to be realistic — we should tune
`I_crit` such that wood is extinguished at a realistic temp."*

Extinction is real physics and should stay. The question is **where the walls sit**, and the
burning run put numbers on two of them that no previous pass had.

### 4.0 ★★ THE ROOT FINDING — one margin governs everything (Erik, 2026-07-31)

Erik asked two questions that turn out to have a single answer:
*"O₂ starvation — is it because of expansion? 0.19 O₂ is not starving, we said it would die
at 0.13. If that's lost it's a bug."* and *"`wall_damage`, how does it affect the duration?
The crate was barely damaged after burning for 6 min — I would have expected its hp to be
close to 0."*

Sustain requires `a > r/(1+r)` with `a = F · o2f · hot`. At ambient, pristine, hot:

```
a(ambient)     = 1 × 0.0920 × 1 = 0.0920
threshold      = 0.080/1.080    = 0.07407
headroom ratio = 1.242×
```

**The PRODUCT `F·o2f·hot` may only fall to 80.5% of its ambient value before the fire dies
at any temperature and any `k_fire_heat`.** That single margin sets all three floors:

| factor falling alone | floor | consequence |
|---|---|---|
| `hot` | 0.805 | `h_min` = 0.806 — the temperature floor (§4.2) |
| `o2f` | 0.0741 | **local X floor 0.1945** — only a 7.4% relative O₂ drop (§4.1) |
| `F` | 0.805 | **hp floor 24.2/30 — only 19.5% of the crate can EVER burn** |

**Consequence A — `o2_frac_ext = 0.13` is now dead code.** There are two extinction
thresholds: the physical one (`o2f = 0` at X = 0.13, Peatross-Beyler anchored) and the
logistic one (`a = r/(1+r)`). The logistic one bites at X = 0.1945, far above it, so the
literature-anchored limit **can never be reached**.

**This is a side effect of the `o2_frac_full = 1.0` change (`b340bba`) that nobody
predicted.** With the old ambient-normalised `o2f`, ambient gave `o2f = 1.0` and the same
threshold sat at `X_thr = 0.13 + (X_full − 0.13)·r/(1+r) = 0.13 + 0.08×0.0741 = 0.1359` —
essentially AT the physical limit; the model was coherent. Normalising against pure O₂
compressed ambient to 0.092 and pushed the threshold to 0.1945. **The headroom Erik wanted
above ambient was bought by giving up the margin below it.**

**Consequence B — the burn can never be fuel-governed.** `wall_damage·I` at 0.083 and
I ≈ 0.12 over 334 s predicts ~3.3 hp, matching the observed 30 → 26.688 exactly. Reaching
hp ≈ 0 in 6 min would need `wall_damage ≈ 0.55` (6.7×) — but raising it *shortens* the burn,
because consuming fuel lowers `F`, lowers `a`, and walks the fire toward the threshold. And
`F` cannot go below 0.805 regardless. **"Charred remains" at 26.7/30 is a barely-singed
crate, and no dial in `fire_tune_loop.py` can change that.**

**The structural cause:** `r = k_die/k_grow` sets **both** the equilibrium intensity **and**
the extinction threshold. One parameter, two jobs — the same defect shape as `fuel_ref`
(fuel fraction vs wood's hp), `COOL_SHIFT` (one e-fold for all materials) and `o2_frac_amb`
(ambient vs full-response reference).

**Decide:** should the death term be split so that `I_eq` and the extinction threshold are
independent — e.g. a constant mortality `k_die·I` setting `I_eq`, plus a separate
O₂/temperature-dependent extinction term that only bites near the physical limits? That
restores `o2_frac_ext = 0.13` to meaning something, lets a crate burn most of its mass, and
keeps Erik's `I ≈ 0.21 with headroom` anchor. **Verify the algebra independently before
building on it** — it is derived here, not measured, and two derived floors in this arc have
already been wrong (see §6's method note).

### 4.1 ★ C6 — the oxygen floor (measured; §4.0 explains WHY it sits where it does)

The sustain condition is **symmetric in `hot` and `o2f`**. Alongside the temperature floor
there is an oxygen floor:

```
o2f_min = r/(1+r) = 0.0741     (r = k_die/k_grow = 0.080)
  ⇒ local X floor = 0.19444    against ambient 0.21
```

**The flame ring may lose only 7.4% of its oxygen, relatively, before the fire dies at any
temperature and any `k_fire_heat`.** Observed minimum was 0.19803 — it survived by **1.8%**.

Confirmed causally by a supply probe (sky-τ 60 → 10, no fire dial moved): X min 0.198 →
0.202, peak I 0.166 → 0.178, max T 336 → 384 K. **The plateau is O₂-limited, not
heat-limited**, and death is O₂-and-heat governed, not fuel governed — which is why
`wall_hp` fell only 11%. **No dial in the tuning loop moves that floor**; only `r` or the
oxygen supply/draw does.

Real compartment fires *are* ventilation-limited, so this may be exactly right. But a 1.8%
margin is fragile: any change to `r`, `o2f` or the supply moves it.

### 4.2 The bootstrap window

`I_sustain / I_eq = (fire_T_ext + fire_T_span·h_min) / T_plateau`, with
`h_min = [r/(1+r)]/o2f = 0.806`. It was **0.746** in the passes that failed — the fire had to
be born at 75% of its final size — and is **0.478** at the burning dials. The lever that
bought it was **`span`, not `fire_T_ext`**: `h_min·span` dominates the numerator.

### 4.3 The far-field wall

Measured far-field temperature rise **88.6 game = 177 K**, against a ≤20 target. The tuning
plan already names far-field rise as scaling with `k_fire_heat` alongside the plateau, i.e.
**it — not the 400–500 flame-T band — is the binding constraint on how hot the fire can be.**

### 4.4 Also open from the audit

`fire_T_ext`/`fire_T_span` are **global** while `ignition_temp` is per-material, so a tile
can ignite below its own sustain floor (shipped 350 exceeds both ignition temps). Proposed
shape: `fire_T_ext[mat] = ignition_temp[mat] − Δ` — one new global, zero new per-material
dials (the cool-shift vacuum-offset precedent). And `ignition_seed` must clear a now
per-material `I_sustain`.

**Decide:** are these walls in the right places for the feel Erik wants? Is the 1.8% O₂
margin acceptable or should `r` / supply move? Does `fire_T_ext` go per-material now?

## 5. Sequencing (Erik's, and it matters)

Erik: *"we should first decide what we do with radiation — either we implement
`heat_inv_shift` or we change to a temperature-dependent radiation model. The latter seems
more interesting."* **Q1 is decided first; §3 follows from it; the raycaster scope (Q2)
depends on both.** Do not run them as independent questions — that is what made the last
session feel circular.

## 6. Standing constraints

- **Determinism is a hard requirement.** Q16.16 integers only in the sim path; no floats, no
  libm transcendentals (`cpp/src/fixed_point.h`). `test_no_float_in_sim_tu` guards it. This is
  why T⁴ must be a baked table.
- **Every change gates CPU↔CUDA at tolerance 0**, step *and* resident.
- **Byte-identity gates** where behaviour should not move. **No golden rebase** — the arc
  carries exactly ONE deliberate rebase, unspent, for the blessed tuning.
- **★ Before that rebase, the golden scenario needs fuel**:
  `tests/field_ab_harness.default_scenario_sim` has `flammable.sum() == 0` and seeds fire on
  AIR, and `fire_simulation.cpp:143` early-outs on non-flammable tiles — **no golden in the
  suite can move when a fire or O₂ law changes.** (Bench report §8 item 26.)
- **Feel-adjacent ⇒ HUMAN-TEST.** Nothing here auto-merges.
- Suite baseline on this branch: **39 failed / 1817 passed / 5 skipped** — the 39 are
  inherited by-design reds from the o2-continuous-law line. Match the failure *set*.
- Method lesson, standing practice: **enumerate writers of a field; do not grep the mask name
  near topic keywords.** It found the EOS regression, the combustion deposit, the plume shim,
  and the dead raycaster path.

## 7. What Erik still wants from the bench

*"we should probably tune a little bit anyway."* He is happy with peak I 20% under target. The
open feel items are the ones §4 constrains: fire is small, fast and cool because it is
**starving**, and the two walls (O₂ floor, far-field rise) are more interesting than dial
work. Any ruling here should say what it does to those two numbers.

---

**Appended 2026-08-14 (supersession note).** Any ×2 game-T→Kelvin map referenced
above is superseded by the unified canonical map in
`[physics.temperature_scale]` (`K = 293 + 3·T_game`; EOS pressure calibration
keeps a named, deliberate exception at `eos_t_amb_k = 290`). See
`docs/temperature_scale_unification_design_2026-08-13.md`.
