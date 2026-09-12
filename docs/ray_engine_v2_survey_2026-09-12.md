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
| R-I | **Radiation becomes a field solve.** Emitters no longer cast rays; every cell emits and absorbs. | 2026-09-12 | Erik: *"yes I am happy with this."* Makes radiation consistent with pressure and conduction, which are already field solves. |
| R-J | **The temperature map IS the emission map.** No emitter list, no gate. `T_emit_gate` is deleted. | 2026-09-12 | Erik's own formulation. Free under R-I, catastrophic without it. |
| R-K | **Every cell radiates by its own temperature** (Q4 answered: yes, because it is free). | 2026-09-12 | This is the actual fix for unbounded reach — see §9.2. |
| R-L | **Clear air does NOT absorb heat.** | 2026-09-12 | Erik, and the physics agrees: clear air is nearly transparent to thermal radiation, and an absorbing atmosphere would heat rooms unrealistically fast. |
| R-M | **Smoke absorbs heat strongly.** | 2026-09-12 | Erik: *"100% in the spirit of what breach is and wants to be."* One per-gas coefficient; gives a radiant-shield mechanic nearly free. |
| R-N | **No short-range flame-contact channel.** Radiation remains the only spread mechanism. | 2026-09-12 | Unnecessary once geometry is honest: flux at one tile exceeds the ignition threshold by an order of magnitude. |
| R-O | **Solids heat adjacent gas.** Direction set; mechanism deferred to its own session. | 2026-09-12 | Erik: *"burning solid will then heat the air around it … it would also cause smoke to travel more realistically."* Resolution for the ignition-drain tension is one-way coupling and/or honest per-medium heat capacity (Erik's proposal). |
| R-P | **Split implementation: heat in C++ with a CUDA twin; light in GLSL, render-only.** One shared directional structure and one shared transmittance definition. | 2026-09-12 | Erik: *"I bless ur recommendations whole heartly."* |
| R-Q | **A coarse integer light payload rides the sim sweep**, for stealth and RL observation. | 2026-09-12 | Recovers the RL shadow-exploitation Erik thought he was giving up; already specified in `ml/01_ml_and_training.md`. |
| R-R | **Light will not fall off faster than 1/r** (Erik's preference; revisit at feel-tuning time). | 2026-09-12 | |

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

## 9. The model as blessed (design session, 2026-09-12)

### 9.1 Shape

Radiation is a **field solve**. Every cell carries an emission source term set by
its own temperature (Stefan–Boltzmann, so T⁴ survives — not because it is holy
but because "the temperature map is the emission map" requires emission to be a
function of temperature). Every cell absorbs what arrives. There is no emitter
list, no gate, and no per-emitter ray budget. Cost is set by the grid, so
**reach is free** and the number of hot things does not matter.

Of the not-holy list (R-B), all four are gone: the 8-ray fan, the emitter gate,
pairwise antisymmetric bookkeeping, and density-as-falloff.

### 9.2 Why reach stops being a problem — the actual mechanism

Not the exponent. **Radiating receivers.**

The flashover happened because a receiver below the gate had no radiative loss
at all, so any trickle of flux accumulated without limit. Once every cell
radiates, a receiver settles where absorbed equals emitted, and because emission
goes as T⁴ the equilibrium temperature is extremely insensitive to flux: cut the
arriving flux tenfold and equilibrium falls only to 56%.

Calibrated so a crate ignites wood at ~3 tiles:

| Falloff | Flux at 20 tiles | Equilibrium there |
|---|---|---|
| 1/r | 15% of ignition flux | ≈373 K |
| 1/r² | 2% of ignition flux | ≈232 K |

Both far below ignition. So the natural 2D 1/r is sufficient, and the swappable
distance function (§7.3) demotes from structural requirement to feel dial.

### 9.3 The reach curve, derived rather than fitted

From view factor and the real ignition threshold for wood (10–12 kW/m²), using
the measured 1556 K and 0.333 m tiles:

| Fire | Radiated power | Ignition radius | In tiles |
|---|---|---|---|
| One burning crate | ~150 kW | 1.1 m | ~3 |
| Fully involved room | ~1 MW | 2.8 m | ~8 |

At one tile the flux exceeds the threshold by roughly tenfold, which is why R-N
needs no separate contact channel. This matches Erik's approved straw man almost
exactly — the design instinct and the physics agree, so the curve can be derived
instead of tuned.

### 9.4 Two phenomena, deliberately separated

| Phenomenon | Mechanism | Timescale |
|---|---|---|
| Direct radiant ignition | Radiation, view-factor limited, ~3 tiles from a crate | Seconds |
| **Flashover** | Solids heat the gas (R-O); gas circulates and accumulates; the hot layer brings every surface to ignition together | **Minutes** |

Erik's correction, which drives this: *"flashovers doesn't happen fast in real
life, it happens after a room has been burning quite much for quite some time,
the mean temp gradually increases until it reaches ignition level."* Flashover is
compartment-scale thermal accumulation, **not** radiation reaching further.
Trying to get flashover out of the radiation law is precisely what the ×8 H_bed
tuning was doing wrong.

### 9.5 The 1/r² near, 1/r far geometry (retained as a feel dial, not a necessity)

Our sim is a horizontal slice of a slab with a floor and a ceiling. Within about
a ceiling height, radiation still spreads in three dimensions and falls as 1/r².
Beyond it, the up/down directions are exhausted and only in-plane directions
remain, giving 1/r. Crossover ≈ ceiling height ≈ 7 tiles at 2.5 m decks. Open
sky has no ceiling, so energy keeps leaving and the falloff stays steep. One
per-cell coefficient expresses all of it, and per R-R light stays at 1/r.

### 9.6 Implementation split

| Half | Where | Numbers | Clock | Why |
|---|---|---|---|---|
| **Heat** | C++ `PhysicsEngine` + CUDA twin, like the other seven solvers | integer, conservative | 24 Hz sim | Must be bit-identical cross-machine and must close the arc-#54 books. A 16-direction sweep over 32,768 cells is ~0.5 M cell-updates: ordinary CPU work. |
| **Light, for the eye** | **GLSL**, render-only, never digested | float | render rate | If you are drawing at all you already have a GPU and a shader pipeline. No CUDA dependency, no NVIDIA lock-in, does not touch the `--cuda` opt-in. Headless training already skips RGB lighting by design. |
| **Light, for the rules** | A payload on the **same sim sweep as heat** | integer, coarse (tile res) | 24 Hz sim | Stealth + RL observation. Already specified by name in `ml/01_ml_and_training.md` §2.2: *"a scalar light-intensity field for stealth and line-of-sight."* |

**This is not a parallel system.** The rules-side light is one extra payload on a
sweep that is already visiting every cell in every direction, not a second
traversal. The eye-side and rules-side computations read the **same** material
and gas coefficients and the **same** geometry, so they cannot disagree
structurally — only in resolution and precision.

**The risk to manage:** what looks dark to the player must be dark to the rules.
Shared inputs make that a calibration problem rather than a correctness one, but
it needs a gate (a test asserting the two agree within tolerance on a scenario).

### 9.7 Light keeps direction, and gains range

Cascades store radiance **per direction** at every probe, which is strictly more
than today's single dominant-direction vector that the normal-mapping pass
consumes. A flashlight is an emitter with a restricted angular range — natural
in a directional solve.

The hard `max_range` cutoff (18–25 tiles) disappears; a beam falls as 1/r and
runs until something blocks it. Per R-R Erik wants to keep that.

### 9.8 OpenGL target (measured on Home Desktop, 2026-09-12)

The raylib Python binding exposes `rlLoadComputeShaderProgram`,
`rlComputeShaderDispatch` and `RL_COMPUTE_SHADER`, but the **runtime context is
OpenGL 3.3 / GLSL 3.30** (`rl_get_version()` → `RL_OPENGL_33`) on an RTX 3070
whose driver offers 4.6. The limit is the shipped raylib build, not the hardware.

**Fact checked 2026-09-12:** the installed raylib is a **prebuilt PyPI wheel**
(`raylib 5.5.0.4`, `cp311-cp311-win_amd64`). We have never built it ourselves,
so a source build would be a new step, not a repeat of one already done.

**RULING R-S (Erik, 2026-09-12): build our own raylib at GL 4.3 — but buy the
decision with a measurement, not a guess.** Write the cascade as multi-pass
fragment shaders on GL 3.3 first (the reference implementation we want anyway,
and what a compute version would be checked against), measure it on a real map
at the resolution whose look Erik approves, and then the build decision is
arithmetic rather than judgement.

Erik's reasoning, which corrected mine: a broken build announces itself the
moment you run the game, unlike a stale doc that lies quietly for months; the
C++ toolchain already exists on every machine, so this is the same class of
chore, not a new one; and 2× is not a micro-optimisation when it buys a
resolution tier.

**It is really a resolution decision.** Extrapolated from the published
benchmark:

| Map / light resolution | Cells | Compute | Fragment |
|---|---|---|---|
| Largest today, tile res | 33 k | 0.2 ms | 0.5 ms |
| Largest today, 2× sub-tile | 131 k | 1.0 ms | 1.9 ms |
| Full-size map, tile res | 131 k | 1.0 ms | 1.9 ms |
| **Full-size map, 2× sub-tile** | **524 k** | **3.8 ms** | **7.7 ms** |
| Full-size map, 4× sub-tile | 2.1 M | 15 ms | 30 ms |

Row four is where we will want to live. At 60 fps (16.7 ms) the fragment version
takes 46% of the frame; compute takes 23%. That tier is what the custom build
buys.

**Erik's structural point, recorded because it shapes the whole budget:** RL
training is headless and renders nothing, so the render budget competes with
nothing. When we *do* draw, the optimisation work already done leaves room to
spend generously on the look.

**Two things to know before buying resolution.**
1. **Cascades give SOFT penumbrae by design** — sharp shadows are their known
   weakness. The crisp prototype look (#64) comes from a steep **transfer curve
   at display time**, not from field resolution. Resolution buys accuracy of
   *where* the edge sits (a thresholded coarse field is blobby and swims under
   motion), not crispness itself.
2. The checkerboard artifact for lights smaller than ~8× base probe spacing gets
   **less** likely at higher resolution, so a small source like a flashlight is
   another argument for resolution.

**The one real risk and its kill:** a `pip install --upgrade`, or a freshly
created env, silently restores the stock wheel and the compute path vanishes.
Mitigations, all cheap: assert the context is ≥ 4.3 at startup when the cascade
path is enabled so a reverted wheel **fails loudly** instead of degrading
silently; pin the package; add the build to the `new-machine-setup` skill.
Machine drift matters little here — this is render-only, so a stock-wheel
machine looks different and runs slower but **desyncs nothing**.

## 10. What gates implementation (2026-09-12)

### 10.1 Hard gates, in order

1. **`fire-12` must be green.** A new system cannot be gated against a red
   suite — new breakage is indistinguishable from old. Four reds beyond the
   three pre-existing parked ones, three of them already blessed for fixing:
   re-baseline `GOLDEN_AGGREGATE` (which also clears the 12 CUDA tests, since
   they share that hash); rewrite `test_foliage_tile_burns` to seed heat
   (R-G); and rewrite `test_e2e_1_sealed_room_fire_self_starves`, whose
   "fuel barely touched" expectation is stale now that `wall_damage = 0.36` is
   blessed (R-E). **This is the immediate next work.**
2. **A real design document plus the adversarial critique pass.** This survey
   carries a blessed *shape*; implementation needs the mathematics — sweep
   formulation, angular discretisation, the exact-integer conservation scheme,
   cascade pass structure, calibration procedure.
3. **Two genuinely unsolved technical designs** (see §10.2). Attack early.

### 10.2 The named risks

- **The exact-integer conservative sweep.** Discrete ordinates conserves in real
  arithmetic; conserving *exactly in int64*, the way `face_flux` does for gas
  energy, is a different problem and is not solved by citing the literature.
  **This is the piece most likely to force a redesign** — do it first, not last.
- **The calibration.** Something replaces `rad_scale` such that ignition lands
  at ~3 tiles for a crate and ~8 for a room fire. §9.3 means this can be
  *derived* rather than fitted, which is new — but the derivation is not done.
- **The old raycaster's other customers.** Unit heat damage reads `rad_flux`,
  weapons share the DDA primitive, the renderer's light path hangs off it. A
  replacement either keeps them working or migrates them, and that must be
  decided before the first patch, not during the last.

### 10.3 Explicitly NOT gating

- **Philox / ingress door 4.** Verified 2026-09-12: `63-swarm-units` is
  **docs-only** (14 files, zero code); the chapter's `cpp/src/philox32.h` +
  `src/simulation/philox32.py` do not exist yet. And the ray engine will never
  want it — both candidate algorithms are fully deterministic with no
  stochastic sampling, so they touch door 4 not at all. Philox stays a
  swarm-units prerequisite only.
- **The raylib / OpenGL build** (R-S: measure first).
- **Light resolution** (the same measurement answers it).
- **Solids heating gas** (R-O, its own session).
- **The fire model rework** (after transport; better for waiting).

### 10.4 Open parameter never yet discussed

**How many directions the sweep carries.** Eight was too few, and that is the
entire content of the ray-effect artifact. 16 / 32 / 64 are all plausible; cost
is linear in it and artifact level inverse. It wants a measurement, and it
belongs in the design document as an explicit parameter with a stated basis.

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
