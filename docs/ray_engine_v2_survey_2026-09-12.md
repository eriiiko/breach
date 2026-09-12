# Ray engine v2 — survey, requirements and open questions (2026-09-12)

> **Status:** design-session input. Nothing built, nothing decided beyond the
> rulings in §1. This is the opener Erik asked for after reading
> `docs/fire_radiation_assumptions_2026-09-11.md` and the follow-up research.
>
> **Depends on:** `docs/fire_radiation_assumptions_2026-09-11.md` (what the
> current radiation model actually does, with citations) ·
> `docs/architecture/engine/08_ray_engine.md` (the canon chapter this will
> eventually replace or amend) · `docs/fire_3c_r4_tuning_and_radiation_2026-09-06.md`
> (the flashover that started this).
>
> **Trigger.** Erik, 2026-09-11, on reading the assumptions inventory: *"this is
> not the system I designed anymore … I think I'd like to rethink radiation from
> first principles — apply it and rewrite the whole thing."*

---

## 1. Rulings already locked

| # | Ruling | Date | Note |
|---|---|---|---|
| R-A | **Diffuse bounce light only. No mirror/specular reflections.** | 2026-09-12 | Erik: *"we can't see mirror reflections from our perspective anyway (top down)."* This matters: sharp specular is the one thing radiance cascades are explicitly bad at, and dropping it removes the only serious objection to them. |
| R-B | **Not holy, may all be replaced:** the 8-ray fan, the emitter temperature gate, pairwise antisymmetric bookkeeping, ray density as the falloff law. | 2026-09-12 | Erik agreed in full. |
| R-C | **Requirement ranking agreed** (§2). | 2026-09-12 | |
| R-D | **No merge of `fire-12` for now.** The ray work may land on the same branch. | 2026-09-12 | Erik: *"we don't have to merge at all … what we've done so far in fire12, I'm pretty happy with. The only thing that wasn't good was the radiation."* |
| R-E | **Do not revert `H_BED_SHIFT` 7 or `wall_damage` 0.36.** | 2026-09-12 | Erik is happy with them. His read is correct: ×8 did not break anything, it made fires hot enough to expose a transport law that is wrong at *every* H_bed. ×1 and ×2 hide the bug rather than fix it. |
| R-F | **Golden may be re-baselined to the current value** `167b96bd…` (R4 the sole cause). | 2026-09-12 | See §6.3 for what the golden does and does not prove. |
| R-G | **`test_foliage_tile_burns` to be rewritten to seed heat**, not bare `fire`. | 2026-09-12 | Its premise predates R3. |
| R-H | **24 Hz is the sim rate.** 12 Hz was too little; 24 Hz reads well for smoke. Intermediate rates untested. | 2026-09-12 | Budget consequence in §5.4. |

---

## 2. Requirements, ranked (Erik, 2026-09-12)

1. **Determinism of anything gameplay-visible; conservation of heat; CPU↔GPU
   bit-identity.** Non-negotiable — these are the iron rules.
2. **Correct reach and falloff.** Spread that is not a flashover, and cost that
   does not grow with the size of the fire.
3. **Shared ray geometry for light and heat.** Smoke and glass attenuation
   preserved; every light and weapon able to emit; sub-tile light resolution.
4. **Light as gameplay state** for stealth and reinforcement learning.
5. **Diffuse reflections** (per R-A).
6. **Fire model rework**, after transport lands.

---

## 3. What we have today, in one page

Full detail with citations: `docs/fire_radiation_assumptions_2026-09-11.md`.
The six load-bearing assumptions and the chain they form:

1. Heat travels along **8 rays per emitter**. A ray loses nothing to distance
   and nothing to air; only material occlusion reduces it. The 1/r falloff
   exists solely as ray *density*.
2. **Each hit deposits a distance-independent quantum.** A tile 14 away gets the
   same energy per hit as a tile 1 away, just ~14× less often.
3. **Emission goes as T⁴** with no ceiling that binds.
4. **A receiver below the emitter gate has no radiative loss channel at all.**
   Only burning tiles, or thermal solids above 930 game units, cast. Casting is
   the only way to lose heat radiatively.
5. **Furniture-class solids have no other loss either** (`conductivity = 0`), so
   the sole loss is a 341 s ambient decay.
6. **Spread *is* radiation.** No cellular spread term, no flame-contact channel.
   A neighbour ignites only when its own temperature crosses `ignition_temp`.

**The chain:** (1)+(2) give a far field that falls only as 1/r; (3) makes it 27×
stronger at the ×8 transient than at ×2; (4)+(5) mean receivers keep nearly
everything they get; (6) forces radiation to be strong at one tile because it is
doing flame contact's job. Hence the measured flashover: a crate at 1556 K
ignited fuel 14 tiles away, through cold air, in 7.5 s.

**Arithmetic check from the constants alone** (independent of the measurement):
each hit lands ≈18 game units; a tile at r=14 is hit about once per 11 ticks;
that is ≈39 game/s of heating against ≈0.8 game/s of loss, reaching ignition in
≈7 s. Measured: 7.5 s. The model is understood, not mysterious.

**Consequence worth stating plainly:** under the current law, *every* furniture
tile in line of sight of a hot crate ignites eventually, at roughly half a
second per tile of distance. There is no horizon. The `fire_tuning` level's
premise that stations ≥6 tiles apart are independent is false, so every
multi-station measurement there is contaminated until reach is fixed.

---

## 4. What the literature says (research, 2026-09-11)

### 4.1 Light — Radiance Cascades

The modern 2D global-illumination answer. Alexander Sannikov, 2023, shipped in
Path of Exile 2; the 2025 **Holographic Radiance Cascades** paper
(arXiv 2505.02041) is the current refinement.

- **Fully deterministic.** No stochastic sampling, no temporal accumulation, no
  denoiser. Single-shot, noiseless.
- **Cost is O(X² log X) and constant for a given grid size, independent of scene
  complexity and of the number of lights.**
- **Measured: 1.85 ms at 512×512, 7.67 ms at 1024×1024** on an RTX 3080 Laptop.
- **Supports participating media and per-channel coloured absorption**, so the
  smoke and glass attenuation Erik wants kept survives natively.
- Core idea, the **penumbra hypothesis**: resolving light from an object needs
  high *spatial* resolution nearby and high *angular* resolution far away. Each
  cascade halves probe density and doubles ray count. Merge rule:
  `L(p←r) = L(p←q) + T(p←q)·L(q←r)` with `T(p←r) = T(p←q)·T(q←r)`.
- **Known weaknesses:** sharp specular reflections (the output carries only ~4
  directions) — **retired by R-A**; checkerboard artifacts for lights smaller
  than ~8× the base probe spacing; memory blow-up in 3D — irrelevant, 2D is its
  home turf.

### 4.2 Heat — the Discrete Ordinates Method

The standard engineering solver for **radiative heat transfer in participating
media on a structured grid**. Conservative finite-volume formulations exist.
Cost is cells × ordinates, again **independent of emitter count**.

Its documented failure mode is named **"ray effect"**: too few angular ordinates
produce unphysical oscillations and direction-dependent striping. That is
precisely our 8-ray artifact. We rediscovered a known problem; the literature
also supplies the cure (more ordinates, or a different angular discretisation).

### 4.3 Gameplay visibility — symmetric shadowcasting

The roguelike family has solved deterministic integer visibility for decades.
*Symmetric* shadowcasting guarantees that if A sees B then B sees A, which is
what a fair hide-in-shadow rule needs.

---

## 5. The cost audit (measured and simulated, 2026-09-11)

### 5.1 The expensive cast is the render light one, not heat

Two casts on two clocks. Radiation runs at 24 Hz on the sim tick. The renderer's
light field runs **per frame, up to 60 Hz**.

| Scene | Radiation, per tick | Light, per frame |
|---|---|---|
| One burning crate | 317 DDA steps | 14,562 DDA steps |
| Room fire, 50 emitters | 3,960 steps | 34,758 steps |
| Firestorm, 600 emitters | 47,520 steps | 34,758 steps (capped at 16 fire lights) |

The steps are not equal either. A radiation step is one memory read plus ~6
flops. A light step is ~9 memory touches, a 7-gas strided gather, ~50 flops and
**three calls to `exp`**. Ten to twenty times dearer.

### 5.2 Rays die far sooner than the nominal range suggests

Simulated against the shipped tilemaps, replicating the exact DDA:

| Level | Grid | Mean DDA steps per ray |
|---|---|---|
| fire_tuning | 72×46 | 17.4 |
| unhcr_vessel | 50×120 | 13.1 |
| playground | 100×70 | 9.9 |
| test_level | 128×256 | 6.8 |
| planetside_demo | 56×38 | 5.8 |

Interior walls are `heat_atten = 1.0`, so survival hits zero in one tile.
**`RADIATION_RANGE = 320` exists for the conservation ledger**, so that escaping
energy can be charged to the sky, **not for reach.**

### 5.3 Three structural facts about the GPU path

- **There is no GPU path for the render light cast at all.** No CUDA branch
  exists in the lighting pass. It runs on the CPU every frame.
- The fire cast is host-side **even on the resident path**.
- The CUDA raycaster we do have `cudaMalloc`s and frees every plane on every
  call, uploads and downloads everything, and synchronises after each launch.
  At our grid sizes the launch tail alone is estimated at **1.5–4 ms per tick**.
  Overhead, not arithmetic, is the dominant GPU cost. A persistent-buffer
  cascade pipeline removes exactly this failure mode.

Measured GPU reference points: the batched device fire cast runs **~1 ms for a
600-fire firestorm** at 128², a 277× win over the per-source loop; the second
device round-trip cost ~2.1 ms and is why the fire visible-light cast is skipped
entirely.

### 5.4 Budget, and the size caveat

**Erik's note (2026-09-12): do not size this from today's test levels.** No full
level has been authored yet. He expects similar dimensions, *possibly* 4× bigger,
untested.

| Grid | Cells | Relative to the 512² cascade benchmark | Implied cascade cost |
|---|---|---|---|
| 128×256 (largest today) | 32,768 | 1/8 | ≈0.25 ms |
| 256×512 (4× the area) | 131,072 | 1/2 | ≈0.9 ms |
| 512×1024 (16× the area) | 524,288 | 2× | ≈3.7 ms |

**The conclusion survives 4× area comfortably and 16× with care.** Open question
Q10 asks which "4× bigger" means.

At 24 Hz the tick budget is **41.67 ms**. Every performance document still
budgets against 12 Hz and 83 ms and has not been restated. That matters, because
the atmosphere group alone measured **18.97 ms p99 at 160²**, a grid comparable
to ours. One system is already consuming ~45% of a 24 Hz tick. Any new ray
engine has to fit in what is left.

---

## 6. Determinism: where the line is, and why

### 6.1 The project already has a name for this

From egregore, the principle locked on the 2026-06-05 ray-engine day (18
architecture decisions): **threshold-driven determinism typing — fixed-point
where a value crosses a gameplay threshold, float elsewhere.** The record says
explicitly that heat and temperature went fixed-point for cross-machine
lockstep, while **light stayed float**, and that the stealth-scalar question was
already part of that discussion.

So the float atomics in the light path are a *consequence* of that scope
decision, not an independent choice, and **not** related to destructibility.

Two corollaries worth keeping:
- **Integer atomics are order-free *and* parallel.** Going integer costs no
  parallelism; the engine already relies on this for heat. Float atomics are the
  only ones that force a choice.
- **Python scalar float arithmetic is already bit-exact.** The real
  nondeterminism risk is C++ reassociation and GPU float atomics, not Python
  geometry such as the cover system's slab test.

### 6.2 Field versus query — and it is already canon

`docs/architecture/engine/08_ray_engine.md` §"The three ray passes" already
defines line of sight as a **pairwise query**, `has_los(a, b)`, on the principle
*"you have line of sight if any light gets through"*, and notes that infravision
is the identical query on the heat channel. It is listed as designed but not
built; the current backing is a binary Bresenham walk.

That gives the clean split:

- **The field is for the eye.** Float, GPU, render-only, never digested.
- **The query is for the rules.** Exact integer, per unit, cheap because a query
  is not a field. Ten units against twenty lights is ~200 integer walks per
  tick, against the ~35,000 float steps the renderer already spends per frame.

**On rounding the float field instead** (Erik's proposal, 2026-09-12): it
reduces the divergence rate but never reaches zero, because rounding *relocates*
the knife edge rather than removing it. Divergence needs the true value to land
within the float error of a rounding boundary, so the odds run at roughly the
float error divided by the rounding step, about 1 in 100,000. At ~144,000
queries per ten-minute match that is about one divergence per match. Acceptable
for single-machine RL; fatal for the cross-machine attestation the determinism
arc earned.

The integer Beer-Lambert needed for the exact version **already ships**, in
`combat.py`, for the hitscan laser: per tile, `energy *= max(0, ONE − Σ absorb·density >> 16) >> 16`.
No `exp`, no transcendentals.

### 6.3 What the golden proves

`GOLDEN_AGGREGATE` is the **cross-machine behavioural tripwire**: it digests the
whole canonical trajectory, every field and cell and tick plus synced unit
state, so it catches any behaviour change on any machine. CPU↔GPU parity is
proven **separately**, by each CUDA test's own bit-identity legs (29 green on
2026-09-11). The twelve CUDA reds shared the golden hash only as one leg.

Re-baselining therefore records "R4 is the approved behaviour" and does not
weaken the parity proof.

### 6.4 Three line-walkers exist

Vision uses an integer Bresenham on the solid mask. Cover uses a float slab
test. The raycaster uses a float DDA. Two of the three feed the same function,
`vision.ray_clear`. Consolidation onto one primitive with several consumers is a
tidiness and correctness win, and canon already gestures at it: *"the DDA march
is a shared primitive with two distinct consumers."* It is not an emergency
(§6.1, Python floats are exact).

---

## 7. Architectural findings to design against

1. **Light is a gather; heat is a conservative transport.** Every cell asks how
   much light arrives (errors are visual). Heat must leave the emitter exactly
   when it arrives at the receiver, or the arc-#54 ledger stops closing. They
   can share geometry; they should not share solvers or number types.
2. **The conservative pattern already exists in-repo and is trusted.**
   `eos_solver.cpp`'s `face_flux`: a per-face flux evaluated once in canonical
   orientation and applied with opposite signs, exact in int64. A finite-volume
   radiation step has the same shape, so heat radiation can be conservative *by
   construction*, exactly like the gas energy step.
3. **The distance law falls out of dimensionality, not a fudge factor.** Two-
   dimensional grid transport gives 1/r for free. You get 1/r² by adding an
   **out-of-plane leakage term** — physically, energy escaping the simulated
   slice. Erik's corridor-versus-planetside instinct is literally the physics,
   and the "swappable distance function" becomes one interpretable per-cell
   coefficient: zero under a sealed ceiling, positive under open sky.
   `levels/planetside_demo` already exists to want it.
4. **Both candidate algorithms abandon "the emitter casts rays."** Cascades
   trace short intervals from *probes*; discrete ordinates sweeps *directions*
   across the grid. This is what makes both flat in emitter count, and it
   changes what "one ray carries light and heat" means — see Q1.
5. **Neither candidate needs a random number.** Both are fully deterministic, so
   the ray engine would not touch ingress door 4 at all, whatever the Philox
   package ends up doing for swarm units.

---

## 8. Open questions for the design session

Sequenced by what gates what. Q1–Q4 gate everything below them.

| # | Question | Why it gates |
|---|---|---|
| Q1 | **One structure or two?** Shared geometry and transmittance, separate payloads and number types (light float/gather, heat integer/conservative) — or something else? | Decides the whole architecture. Also redefines what requirement 3 ("shared ray geometry") means, since both candidates drop emitter-rays. |
| Q2 | **What reach do we want?** Stated as a design target before any fitting: a burning crate lights an adjacent crate in ~X s, one 3 tiles away in ~Y s, one 10 tiles away never (or when?). | The reach curve is the object every law gets fitted to. Without it we are tuning blind again. |
| Q3 | **Does clear air absorb heat?** If yes, does the absorbed energy warm the gas through the energy seam, or is it a counted export? Should smoke and soot absorb more? | Absorption is the only lever that creates a true *horizon* rather than just scaling reach. It also decides whether the gas ledger gains a writer. |
| Q4 | **Do receivers radiate?** Keep the emitter gate (cheap, unphysical, loss-free receivers), lower it toward the original 653 K feel value, or make every warm solid a grey body (physical, but spread may stall). | Decides whether reach is self-limiting. The gate was *raised* historically precisely to stop spread stalling. |
| Q5 | **Confirm the distance law**: 2D (1/r) as default, with out-of-plane leakage as the planetside dial? | Follows from Q3/Q4 but needs an explicit yes. |
| Q6 | **Does light become synced gameplay state, or stay render-only with an exact per-unit query?** (§6.2 recommends the latter.) | Determines whether the raycaster enters the float ratchet and the digest. Large scope difference. |
| Q7 | **Spread channel**: does radiation keep doing flame contact's job, or do we add a short-range adjacency channel and let radiation be physical? | The near field and far field are currently coupled through one channel; this is the only way to let both be right. |
| Q8 | **T⁴: keep?** Recommendation is yes — one table lookup, physically interpretable, and not the cause of the flashover. | Cheap to keep, but it demands a law with a real horizon, since twice the temperature radiates sixteen times harder. |
| Q9 | **Light resolution**: stay at one texel per tile, or go sub-tile? At what factor? | Costs 4× the marching at 2×, 16× at 4×. Memory is free. |
| Q10 | **How big can a full level get?** Does "perhaps 4× bigger" mean 4× the area or 4× per axis? | §5.4: 4× area is comfortable, 16× area needs care. |
| Q11 | **CPU or GPU for the new light field?** Cascades are inherently a GPU algorithm; today's light cast is CPU-only with no GPU path at all. | The single largest engineering commitment in the whole proposal. |
| Q12 | **Scope and sequencing**: does this land on `fire-12`, or its own branch off it? Does the #64 two-state light look ride along? | Erik has said the ray work *may* go in the same patch. |
| Q13 | **Foliage `heat_atten`** is 0.0, so a tree cannot be lit by fire, cannot radiate when burning, and cannot conduct. Recommend 0.5, matching furniture. | Small, but it makes props inert in exactly the system being rebuilt. |
| Q14 | **Tick budget**: confirm 41.67 ms at 24 Hz, and restate the per-system p99 gate. | Every perf doc still says 83 ms. Atmosphere alone already takes 18.97 ms at a comparable grid. |

---

## Systems

**Existing canonical systems this design must use.**

- The ray engine (`cpp/src/raycaster.*` + `cuda_raycaster.cu`) — the only
  radiation path; a replacement lands *here*, never beside it.
- The temperature solver's Pass-1 fold — the only place `rad_net` becomes
  temperature.
- The gas energy seam (`gamemap.py::gas_energy_*`) — mandatory if air ever
  absorbs heat (Q3).
- The energy closure identity — a new channel needs a counter **and** a term in
  one of the four existing groups. There is no fifth group.
- The face-flux energy step (`eos_solver.cpp`) — the conservative pattern to
  copy, not to reinvent.
- The material table — any new per-material optical column is a table row.
- The Q16 boundary modules (`src/simulation/*_fixed.py`) — the required shape of
  any new synced field's conversion layer.
- `pack_hover_readout` — the one tile-probe seam for any new diagnostic.
- `tools/fire_tuning_lab.py` — the instrument; extend it rather than writing a
  second lab.
- GOLDEN_AGGREGATE re-baseline discipline — one deliberate re-baseline per
  approved change, with written rationale.

**New systems this design may create** (draft rules, to be written into
`CLAUDE.md` only once the code exists).

- *A shared directional-transport primitive.* Draft rule: light and heat share
  one geometry and one transmittance computation, and may carry different
  per-ray laws and number types; each law is named and cited at its site.
- *A per-cell out-of-plane leakage coefficient.* Draft rule: the 2D-versus-3D
  falloff choice is this coefficient and nothing else; never a second distance
  term in the march.
- *An exact per-unit light query.* Draft rule: gameplay reads light through the
  query, never by sampling the render field; the field is never digested.
- *A short-range flame-contact spread channel* (only if Q7 says yes). Draft
  rule: short-range spread lives in exactly one place and is never implemented
  as a second radiation.
