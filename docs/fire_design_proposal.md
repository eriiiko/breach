Proposal from the fire-design research pass (2026-06-07), addressing Erik's inline comments in engine/06. **Status: proposal for discussion — nothing here is canon or implemented until agreed.** Once agreed, the accepted parts fold into engine/06 (and a new engine/13 for the FieldEdit primitive).

# Fire Design Proposal — Temperature/Fire Chapter (engine/06)

Status: PROPOSAL for discussion, not a spec to implement. Per the discuss-before-implement rule, nothing below should be coded until we agree. Each section quotes Erik's inline comment, lays out the researched options, gives a concrete recommendation, and shows the fit to the existing engine (deposit-only DDA ray engine + Q16.16 heat buffer; two-field IMEX atmosphere with `wind = -grad(p)`; smoke v2 semi-Lagrangian; Level-2 determinism via fixed-point).

---

## 1. Ray-based fire spread + the caching question

> **Erik (§5.1):** "could we do fire spread with rays too? reuse the light + heat rays — a few rays per source is probably enough. everything is set up for that already." And (§5, caching): "if fire spreads a lot, can we cache the rays per source in sparse arrays, only recompute the sources affected by a map-destruction, extinguish one fire without recomputing all? Or is recompute-all cheap enough — is the complexity worth it?"

### The issue

Today fire spreads by a 12-connected cellular stencil (`fire_simulation.cpp` lines 36-86) that fakes gap-leaping with a hardcoded 2-tile reach and never checks occlusion — it can ignite a tile on the far side of a 1-tile wall. Erik's instinct is to delete that and make each burning tile a light/heat source in the ray pass, which already deposits orange RGB light and Q16.16 heat along DDA rays that the per-channel attenuation field already occludes.

### Why the instinct is physically right (not just convenient)

Fire spread in an enclosed structure **is** radiant heat flux crossing an ignition threshold. Radiation dominates preheating in compartments and becomes overwhelmingly dominant approaching flashover; piloted ignition of wood is a critical-radiant-flux threshold (~10 kW/m² minimum, fast above ~40 kW/m²); flashover is defined as floor flux reaching 15-20 kW/m²; crown/gap spread is literally "a series of piloted ignitions" by radiant flux jumping gaps. Every one of those maps onto a buffer Breach already has. The cellular stencil and the radiant model are not two competing options — the radiant model is what the physics says spread *is*, and the deposit side is already shipped. Only the consumer (heat -> temperature -> ignition) is missing.

### Recommended ray profile per fire source

A burning tile is an **omni** source. Override the auto ray count — fire does not need a smooth glow disc, only enough rays that an adjacent flammable tile reliably gets hit, and overlapping deposits from neighbouring burning tiles fill the gaps:

- **`ray_count = 8`** (fixed, not auto). At short fire range, 8 rays put endpoints ~1 tile apart at 2-3 tiles out. A flamethrower at ~30 tiles x 8 rays is ~240 rays — cheaper than one omni lamp.
- **`max_range = 2 + 3·fire_intensity`** tiles (1 tile = 1/3 m). Guttering flame (`fire=0.1`) ~2.3 tiles; full blaze (`fire=1.0`) ~5. Range growing with intensity *is* the firestorm cascade: hotter -> radiates further -> ignites more -> feeds back (§2).
- **`intensity = 0.3 + 0.7·fire_intensity`**, **`heat = k_fire_heat · fire_intensity`** (a config dial — fire's light is dim, its heat is the gameplay payload). `color = warm orange [1.0, 0.45, 0.12]`.
- **Deterministic angular dither, NOT RNG jitter.** The shipped jitter seeds `mt19937` from `src.x*1000+src.y` (`raycaster.cpp` line 77). That is deterministic per-position but the `heat` deposit is now sim-affecting, so any RNG must come from `sim.rng` — or, cleaner, drop RNG entirely: offset ray `i` by `frac(i · golden_ratio) · (2π / ray_count)`. Near-uniform coverage, zero per-frame state, bit-identical across machines. Keep optional temporal flicker on the render-only `light_rgb`, never on `heat`.

This fits the existing `LightSource` struct with **no new fields** — fire is just a source profile, the same discipline as lamps and flashlights (ch.08 source profiles). Fire adds zero ray-kernel code.

### What each ray deposits (all already implemented in `march_ray_directional`)

1. `light_rgb +=` per-channel survivor x falloff `1/(1+d²·0.01)` — warm glow (render-only float).
2. `light_dir +=` weighted vector toward source — normal-map relief (render-only).
3. `heat +=` `heat_quantize(dep_aggregate · src.heat)` saturating-add into Q16.16 — **the only sim-affecting output**. Integer add is order-independent -> deterministic firestorm with many overlapping sources.
4. `smoke_glow +=` scattered energy where the ray crosses smoke — god-rays through the fire's own plume.

Occlusion is free and correct: the per-channel attenuation drives all channels to 0 at an opaque wall and the ray dies that step — fire heat cannot cross a wall (the hard stop Erik requires), but glass transmits dimmed (a fire behind glass radiates a little through — physically right). This is occlusion *without* `is_wall` — it uses the attenuation field, so it stays consistent the instant a wall breaches.

### Why ray-based leaps gaps better than the stencil

- **Leaps gaps believably:** a ray travels several open tiles depositing heat the whole way, so a blaze heats flammable tiles across an open room — and a non-flammable air gap between two crates is transparent, so the far crate ignites across the gap at the physically correct, intensity-scaled distance, not a hardcoded 2.
- **Respects occlusion for free:** the same ray that leaps the gap dies at a wall. Fire spreads *around* a doorway, not *through* the wall beside it — emergent, no special case.
- **Conduction is a second channel:** the ray dumps heat onto a hull tile; conduction races it along connected metal (ch.06 §2) to an interior wood wall and ignites *that* — radiation across air + conduction along solids, each handled by the system built for it.

### The caching question — RECOMMENDATION: do NOT cache. Recompute every tick.

Erik's instinct is the right renderer-engineer instinct, and the literature backs the *shape* (per-light additive cached buffers, dependency lists of which lights an edit touches, regional recompute on geometry change). But it pays off in a regime Breach is **not** in. Those systems cache because one light is milliseconds-expensive and geometry is near-static. Breach inverts both:

- **Fire sources are deliberately cheap.** With `range = 2 + 3·intensity` and 8 rays, a source is ~100-130 tile-steps. A bad firestorm: 300-800 burning tiles -> after the shipped `coarse_cluster=3` clustering, ~100-400 effective sources -> **~10k-50k tile-steps/tick total**. That is sub-millisecond in C++, comparable to one IMEX atmosphere substep over the whole grid. The ray pass is the bottleneck it is not.
- **Geometry mutates constantly in exactly the target scenario.** A firestorm *is* walls burning through and tiles igniting every few ticks — the cache invalidates precisely when it would be hottest.
- **Smoke defeats the cache every single tick.** The march attenuates on *live* smoke density and deposits god-rays; smoke advects every tick on the wind. So a source's footprint changes every tick even with no wall destroyed and no fire moved. You would either exclude smoke and re-march anyway (cache buys nothing) or invalidate every source every tick (cache buys nothing). **This alone is disqualifying** — and smoke interaction is the entire visual value proposition of ray-based fire.
- **The feedback model (§2) re-ranges every fire tile every tick** with its temperature/O2, so the cache invalidates constantly regardless; it would only help static fires, which are already the cheap case.
- **It adds a second determinism surface.** A cached `heat` field that drifts from a recomputed one breaks the bit-exact lockstep that is the entire reason `heat` is fixed-point. You would be hand-maintaining a cache of the one buffer that must be deterministic — and on the future GPU, one-thread-per-ray brute force over thousands of cheap sources is the happy path, while a sparse per-source cache with scattered atomic subtracts is the nightmare. The CUDA contract actively protects you from this cache.

**Verdict: it costs more than it tastes.** Ship ray-based fire on the existing deposit-only summed-buffer contract. Get the win from two cheap, determinism-neutral levers instead:

1. **Clustering — already shipped** (`coarse_cluster=3`). Biggest lever, free. Tune cluster size and the `block_max > 0.1` cull threshold.
2. **Global frame-coherent dirty-skip** (the cheap version of Erik's idea): cache the *decision to recast at all*, not per-source contributions. If no fire tile changed intensity by more than epsilon, no wall was destroyed, and no smoke crossed a fire footprint, reuse last tick's summed buffers wholesale. One global dirty flag, O(1) bookkeeping, no second determinism surface, degrades gracefully.

Revisit the full sparse cache only if **all** become true: a profiler shows fire rays in the top-3 tick cost; grids grow well past 300/side; and smoke attenuation is decoupled into a separate live multiply pass. Documented-but-deferred.

---

## 2. Fire growth as a feedback system

> **Erik (§5.4):** "it's a feedback system... too little intensity/temp they die out — more temp they spread more — constrained by O2." (Replacing the fixed `fire[i] += 0.5·dt`.)

### The issue

The current growth term (`fire_simulation.cpp` line 101) is an unconditional `+0.5·dt` — fire only ever grows, never self-limits, ignores heat and O2. Erik wants a *signed* term whose sign flips at a threshold: grow when hot + supplied, decay when cold or starved.

### The physics, with numbers

The fire triangle is a **gain loop with two brakes**, not a checklist:

```
flame heat -> surface heats -> PYROLYSIS (solid -> fuel gas) -> fuel gas + O2 burns -> more flame heat
     ^                                                                                       |
     +--------------------------------- positive feedback <-----------------------------------+
                              limited by:  O2 supply  &  remaining fuel
```

- **Self-amplification is super-linear.** Free-burning fires grow as `Q = α·t²` (HRR proportional to time squared). NFPA growth classes reach 1 MW in slow 600 s / medium 300 s / fast 150 s / ultra-fast 75 s. The t² shape *is* the feedback: bigger flame -> more radiant feedback to the surface -> faster pyrolysis -> bigger flame. The fixed 0.5/s cannot capture this; real growth accelerates with current intensity.
- **Self-limit 1 — O2 depletion.** Flaming decays below ~15% O2. Clean experimental law: `X_O2,extinction = 0.0076·(1427 − T)` (T in °C) — the hotter the fire, the leaner the air it can tolerate before dying. This couples the two brakes.
- **Self-limit 2 — critical flame temperature.** Independent of O2, a flame dies below a critical temperature (~1600-1700 K for hydrocarbons): radiative + conductive losses exceed chemical heat release (low Damköhler -> extinction). This is *why fires die below a critical temperature* — the loop gain drops below 1, not because fuel ran out.
- **Self-limit 3 — fuel burnout.** Pyrolysis consumes the solid; mass flux falls below the critical value; flame can no longer sustain.
- **Flashover is the loop going runaway** — high HRR drives all surfaces to near-simultaneous auto-ignition (the cascade ch.06 §6 already wants).

**One-sentence model:** intensity grows when chemical heat release exceeds losses AND O2 + fuel are available; decays otherwise. That is a logistic gain term with a hard floor (extinction) and a hard ceiling (O2/fuel cap).

### RECOMMENDATION: one signed logistic update per burning tile

Replace stages §5.4 (growth), §5.6 (O2 check) and §5.7 (O2 consumption) with a single signed update. Inputs per tile per tick: `I` current intensity `[0,1]`; `T` local temperature from the heat->temperature pass; `P` pressure at the source tile (the O2 proxy — read, never subtracted; see §3); `F` remaining fuel `[0,1]` (normalized `wall_hp` or a dedicated `fuel` field).

```
# availability gates (smoothstep, no hard pop)
o2   = clamp01( (P - P_min) / (P_full - P_min) )
fuel = clamp01(  F / F_ref )
avail = o2 * fuel                         # both brakes, multiplicative

# drive: chemical gain vs losses, with O2-dependent extinction temperature
T_ext = T_ext0 - k_o2 * o2                # encodes X_O2 = 0.0076·(1427-T): hot fire tolerates leaner air
hot   = clamp01( (T - T_ext) / T_span )   # 0 below extinction temp, ->1 well above
drive = avail * hot                       # loop-gain proxy in [0,1]

# signed logistic update — GROWS above threshold, DECAYS below
gain = k_grow * drive       * I * (1 - I) # logistic self-amplification (t²-like, O2/fuel/T capped)
loss = k_die  * (1 - drive) * I           # decay when starved/cold
I_next = clamp01( I + dt * (gain - loss) )
if I_next < I_min: I_next = 0             # hard extinguish (snap to unlit)
```

Compact form replacing §5.4:

```
I_next = clamp01( I + dt·( k_grow·avail·hot·I·(1−I) − k_die·(1−avail·hot)·I ) )
```

Couple fuel consumption: `F -= k_burn·I·dt`. As `F->0`, `avail->0`, fire dies — fuel-limited death distinct from O2-limited death.

**Why each piece is right:** `I·(1−I)` gives t²-style accelerating growth at low I and saturation near I=1, replacing the unconditional `+0.5`. `hot = (T−T_ext)/T_span` is the critical-flame-temperature brake — below `T_ext`, loss dominates, fire dies even with O2 and fuel. `T_ext = T_ext0 − k_o2·o2` is the X_O2-T coupling — one multiply, no division. `o2` from pressure (not depleting atmosphere) honors §3. `fuel` is the burnout brake.

**Ignition / sustain / flashover map onto the same function** — no separate flashover code:
- **Ignition** = the existing `temperature >= ignition_temp AND O2` event (§4) sets `I = I_seed ≈ 0.1`. The feedback then decides if it takes.
- **Sustain** = `gain >= loss` at low I (drive above the crossover). Below it, the seed dies — fires die below critical T/O2, automatically.
- **Flashover** = high-I tiles radiate hard (they are ray heat sources, §1) -> neighbours' T crosses `ignition_temp` -> cascade. Emergent from the radiation + ignition loop once growth lets intensity climb.

### Starting constants (tune in the physics demo)

| Constant | Value | Meaning / source |
|---|---:|---|
| `P_min`   | 0.60 | pressure below which O2 proxy = 0 (matches current `o2_threshold` 0.60) |
| `P_full`  | 1.00 | interior pressure where O2 is "full" (config interior ≈ 1.0) |
| `T_ext0`  | ignition_temp + ~50 | extinction temp at zero O2 head-room (just above `ignition_temp`: wood 300, door 280) |
| `k_o2`    | ~80-120 | T_ext drop per unit O2 (from `0.0076·(1427−T)` slope rescaled to your T units) |
| `T_span`  | ~150 | width of the `hot` ramp above `T_ext` |
| `k_grow`  | ~4.0 /s | growth gain (logistic; replaces 0.5/s, now gated and accelerating) |
| `k_die`   | ~2.0 /s | decay rate when starved/cold |
| `I_min`   | 0.02 | snap-to-zero extinguish floor (matches existing dead-tile epsilon) |

### Determinism fit

The update is an O(1) per-tile gather (reads I, T, P, F; writes I, F, an additive smoke/pressure deposit) — order-independent like the heat deposit. Carry `I` in Q16.16 once `temperature` is Q16.16, and make `k_grow`, `k_die`, the smoothstep slopes, and `dt` power-of-two / exact-rational so the update is shifts + adds + one multiply — no `exp`, no `sqrt`, no division in the threshold path. `clamp01` and the `I_min` snap are exact integer compares. Until `temperature` ships, run it in scalar `[0,1]` (single-machine deterministic), then port to Q16.16 with a cross-machine bit-exactness test, exactly as ch.06 §3 prescribes for temperature.

---

## 3. The O2 coupling fix — fire deposits pressure, reads pressure

> **Erik (§5.7):** do NOT remove atmosphere at the fire — it makes the breach-sink suck smoke *toward* the fire. Fire should deposit a little overpressure (push plume/smoke OUT) and read pressure + temperature back as the O2/intensity proxy.

### The issue, made precise

Wind in the solver is `wind = −grad(atmosphere + wave_p)` (`atmosphere_solver.cpp:282`), and smoke advects on that wind (`smoke_dynamics.cpp:138`, `bx = −wind_x·dt_adv`). The current fire removes atmosphere next to itself (`fire_simulation.cpp:135`, `atmosphere[n] -= o2_consumption·dt·fire`). That makes the fire tile a **local pressure minimum** -> `−grad p` points **into** the fire -> smoke is sucked into the flame and pools there. Exactly backwards.

### The physics: a flame is a pressure source, not a sink

At roughly constant pressure, combustion turns cool dense air into hot light gas occupying ~7-8x the volume (expansion ratio ≈ 8:1). A flame is a **positive volumetric source** and an updraft. The buoyant column rises (out of our 2D plane); what survives the top-down projection is: the burning footprint is slightly **over-pressured** relative to the cold room, the plume is pushed radially outward, and fresh air is drawn back to the base along the same gradient. The plume entrains ~10-12x the stoichiometric air requirement; the inflow of fresh air is the *same gradient the O2 proxy reads*, not a separate hand-coded suction.

### RECOMMENDATION

**(a) Fire WRITES a small self-limiting overpressure** into `atmosphere` (the bulk slow field — the same field `apply_explosion` already deposits into safely, since IMEX implicit diffusion absorbs the spike). Per burning tile `i`, on its **own** index:

```
Δp_plume(i) = max( k_buoy · f_i · (1 − atmosphere[i] / p_expand_ref) · dt , 0 )
atmosphere[i] += Δp_plume(i)
```

- `k_buoy ≈ 0.4 /s` — tune so a steady flame holds its source ~5-15% above the room, enough that `−grad p` points outward and smoke is visibly pushed away.
- `p_expand_ref ≈ 1.3` — saturation ceiling (standard atm + bounded thermal-expansion headroom). The `(1 − atm/p_expand_ref)` factor self-limits: strong push into cold dense air, tapering as the cell pressurizes, so it cannot run away into the burst-wall relief valve on its own. Pairs naturally with the existing `find_burst_walls` over-pressure relief.

Deposit into `atmosphere`, **not** `wave_p`: the plume is a *sustained* bulk overpressure that should leave a lasting outward wind for the whole burn (the slow-field role). `wave_p` is zero-mean and decays — wrong timescale. (An optional one-shot `wave_source` kick at *ignition* could model the little "whoomp" — a flourish, not the coupling.)

**(b) Fire READS two local quantities** and lets the §2 feedback decide. Consumption is implicit: a fire in a sealed pocket stops being *replenished* because the bulk field drains/equalizes around it (vented through a breach, or flattened by diffusion), so the pressure it reads falls — no explicit subtraction.

```
# fresh-air pressure proxy — exclude the fire's OWN bump so it measures INCOMING air
P_src(i) = mean over open neighbours n of ( atmosphere[n] − Δp_plume(n) )
```
This reuses the existing neighbour-average loop (`fire_simulation.cpp:110-126`) and the existing permeability / `is_wall` masking. High `P_src` = fresh dense air (high O2); low `P_src` = vitiated/vented/vacuum (low O2). In a sealed ship pressure ≈ density ≈ O2 availability; once a breach drains the room, `P_src -> 0` and the fire dies of suffocation and decompression at once — the decompression-extinguishes-fire loop, driven by a read rather than a kill-threshold. The second read is local temperature `T_src` (from the `temperature` field once the conduction consumer exists; use `f_i` as the stand-in until then — a one-line swap).

### The (pressure, temperature) -> growth function

This is the §2 feedback restated in pressure/temperature terms — they are the same single update; choose the formulation you prefer when we agree the constants. Equivalent product-of-sigmoids form, anchored to real thresholds:

```
df_i/dt = R · f_i · ( s_O2 · s_T − L )
s_O2 = smoothstep( P_low , P_high , P_src )                            # fresh-air supply  ∈ [0,1]
s_T  = smoothstep( T_ign , T_opt , T_src ) · (1 − smoothstep( T_hot , T_max , T_src ))  # hot enough, not flashed-over
```

- `R ≈ 1.0 /s` overall reaction rate (replaces the `0.5` constant). `L ≈ 0.15` baseline radiative/burn-down loss: with perfect supply net is `R·f·(1−L) > 0` (grows); with no supply `−R·f·L < 0` (decays smoothly to 0, no hard snap).
- The **product** `s_O2·s_T` (not a min/threshold) reproduces Erik's statement exactly: growth needs both fresh air and heat; either going to zero starves the fire smoothly. This is the fire triangle as one continuous rate (fuel = the `flammable` mask, heat = `s_T`, oxygen = `s_O2`).
- It **subsumes** the old behaviours: the boolean O2-kill becomes `s_O2 -> 0` (smooth death, no pop), growth becomes the positive branch, and "blown out by wind" stays the separate, orthogonal wind-modulation stage (advective cooling/stripping).

| Symbol | Suggested | Anchor |
|---|---:|---|
| `P_low`  | 0.60 | limiting-oxygen-concentration analog (~15% O2 vs 21% -> ~0.7x density); reuses today's `o2_threshold` as a soft knee |
| `P_high` | 1.0  | standard atmosphere — full support |
| `T_ign`  | `ignition_temp` (300 wood) | below ignition no sustain (with `T_src=f_i` proxy use f ≈ 0.05) |
| `T_opt`  | hot/steady | flame fully established (f ≈ 0.4 proxy) |
| `T_hot`,`T_max` | high | optional roll-off for runaway/flashover ("too much temp" arm); disable initially |

### Determinism fit

The **write** is `atmosphere[i] +=` on the fire's **own** tile (not scattered onto shared neighbours like the old subtraction, which summed contributions from multiple fires into one cell — an order-sensitive scatter). Own-index write = pure per-cell map, order-independent, GPU-friendly (no `atomicAdd` contention). The **read** (`P_src` neighbour mean) is a gather — order-independent. `atmosphere` stays float (it no longer has a hard sim threshold — the boolean O2-kill is gone); the **intensity decision** `f` crosses gameplay thresholds, so `f` is the fixed-point channel (Q16.16), mirroring `temperature`. Solver stability is untouched — the deposit is absorbed by the unconditionally-stable implicit diffusion; the self-limiting factor plus `find_burst_walls` bound the pressure with no new CFL constraint. **This change also fixes a latent determinism risk** by removing the scatter-into-shared-cells subtraction.

---

## 4. A general add/remove-field-at-coordinates primitive

> **Erik (§4):** we need a general way to add or remove a field at coordinates — for lasers, grenades, fire burn-off, gas emitters. Should smoke burn-off be inherent in the smoke system, or a method usable by any system?

### The issue

Three call sites mutate physics fields ad hoc today, each reinventing the same disc/line loop with a different sign and clamp:
- `physics.apply_explosion` — `smoke[...] = 0` in the inner 40%, plus `atmosphere +=`, `wave_source +=`, `fire = max(...)`.
- `physics.add_explosion_smoke` — `smoke = min(1, smoke + base·mult)` over a disc, RNG-noised.
- The planned ray burn-off (ch.06 §4) — clear `smoke` along a beam.

All three are the same operation: **take a field, a region, an amount; combine into the field with a mode and a falloff.** Three near-identical disc loops with divergent signs/clamps is exactly the "an `if` for one scenario" smell the architecture README warns against.

### RECOMMENDATION: `FieldEdit` + `EditQueue` — the canonical *write* primitive

Two composing parts:

**(a) `FieldEdit`** — a pure, stateless description of one edit:

```python
class EditMode(Enum):  ADD; REMOVE; SET (lerp-to-value); MAX; MIN
class Region(Enum):    TILE; DISC; BEAM; RECT
class Falloff(Enum):   FLAT; LINEAR (= today's explosion falloff); SHARP; GAUSS

@dataclass(frozen=True)
class FieldEdit:
    field: str            # "smoke", "atmosphere", "wave_source", "fire", "heat", "poison", ...
    region: Region
    coords: tuple         # TILE:(r,c) · DISC:(r,c,radius) · BEAM:(r0,c0,r1,c1,width) · RECT:(r0,c0,r1,c1)
    amount: float
    mode: EditMode = ADD
    falloff: Falloff = FLAT
    channel: int|None = None   # None=scalar; 0/1/2 = R/G/B of an (h,w,3) field
    clamp: tuple|None = None    # post-combine clamp, e.g. (0,1)
    noise: float = 0.0          # >0 = per-tile RNG multiplier in [1-noise,1], drawn from sim.rng
    source_id: int = 0          # bookkeeping/debug
```

One applier `apply_field_edit(gmap, edit, rng)` is the only code that touches a field through this path. `_iter_region` yields `(row, col, weight)` — the disc loop written once. `heat` gets a fixed-point branch inside `_combine` (Q16.16 -> `heat_quantize` + `heat_saturating_add`, never a float `+=`); the mode enum is identical, only the combine arithmetic differs by dtype. That is the payoff: the fixed-point discipline is implemented once, not re-derived at every future heat-deposit site.

**(b) `EditQueue`** — consumers call `sim.edit(FieldEdit(...))` during their phase; the Simulation flushes the whole queue at **one fixed tick point, before the solvers run**, sorted by a stable key `(field, source_id, region, seq)`.

### Queued, not immediate — the determinism-critical decision

For Level-2 lockstep (hard requirement), the answer is a deterministically-ordered queue, not immediate mutation:

- **Order independence is the whole determinism story.** Today explosions apply inline while iterating `self.projectiles`. Two grenades overlapping a tile give `clamp(clamp(s+a)+b)` — order-dependent the moment a clamp / `SET` / `MAX` is involved. A stable sort makes the applied order identical on every machine regardless of projectile/AI/container order — the same principle ch.03 relies on for the heat buffer, lifted to all field edits and made explicit rather than accidental.
- **One flush = one RNG consumer.** `add_explosion_smoke`'s noise must come from `sim.rng`. With one flush site, the flush is the single RNG consumer, drawing in sorted order — the seeded-rollout guarantee is structural, not a per-caller convention.
- **Solvers see a settled pre-state.** A laser burn-off and a grenade cloud issued the same tick both land before smoke advection runs, so the solver advects the net result once.

**Flush slot** in the tick order: after `stamp_units`, weapon/fire/explosion/gas phases enqueue; then `EditQueue.flush(gmap, rng)`; then physics (atmosphere/wind -> smoke advection -> temperature -> fire); field reactions may enqueue for *next* tick; cleanup clears the heat deposit and the queue. The queue is a per-tick deposit list, exactly like the heat buffer.

**The one honest carve-out:** edits that change **topology** — `destroy_wall` (retriggers `on_tile_changed`, the conductivity/occlusion cache patch, `_sink_dirty`) — are **not** FieldEdits and stay immediate/structural. FieldEdit is strictly continuous scalar/vector values on a fixed grid (smoke, atmosphere, wave_source, fire, heat, gases). `wall_hp -= dmg` *is* a clean REMOVE FieldEdit, but the destruction it triggers must run as a separate post-flush structural sweep (collect tiles at <=0, destroy in sorted order — the pattern fire burn-through already uses: solver returns coords, runner destroys them).

### How this answers Erik's smoke-burn-off question directly

Smoke burn-off is **not** inherent in the smoke system — it is a `REMOVE` FieldEdit any system can issue. Laser, grenade, and fire all clear smoke through the identical call, differing only in `region` and `amount`:

```python
# laser — Erik's exact case
sim.edit(FieldEdit("smoke", BEAM, (r0,c0,r1,c1,1), 1.0, REMOVE, FLAT, clamp=(0,1)))   # tunnel through smoke
sim.edit(FieldEdit("heat",  BEAM, (r0,c0,r1,c1,1), weapon_heat, ADD))                   # melt-through via temperature
# fire smoke emission — was an inline smoke[n] += in fire_simulation.cpp
sim.edit(FieldEdit("smoke", TILE, (ny,nx), smoke_emission*dt*I, ADD, clamp=(0,1)))
# grenade
sim.edit(FieldEdit("smoke", DISC, (fy,fx,r), 0.8, ADD, LINEAR, clamp=(0,1), noise=0.85))
# gas emitter
sim.edit(FieldEdit("poison", DISC, (fy,fx,r), 1.0, ADD, GAUSS, clamp=(0,1)))
```

The smoke system *transports*; everything that *injects or removes* smoke does so through FieldEdit. Same for gas, pressure, fire, heat.

### Why it generalises (multi-gas) and fits canon

- **`field` is a string key.** When smoke goes from one scalar to the planned N gas fields (`white_smoke / black_smoke / poison / teargas / fuel_gas`, ch.05 §6.2), *zero* consumer code changes — a poison grenade is `FieldEdit("poison", ...)`. New gas = new field name, not new edit code. The write-side mirror of ch.05's data-driven gas table.
- **Per-field policy table** (on GameMap's field registry, ch.02 ownership) declares per field: dtype (float vs Q16.16 -> which `_combine` branch), default clamp, skip-mask (smoke/gases skip `solid`; `wave_source`/`atmosphere` skip `solid`+`is_vacuum`; `fire` skips non-`flammable`). Consumers stop knowing these rules.
- **It is the third leg of a pattern canon already names twice:** wind = `−grad(p)` is the canonical *read* primitive ("one field, many readers", ch.04); the DDA march is "one primitive, two consumers" (ch.08); **FieldEdit is the missing canonical *write* primitive** — "many systems write many fields through one operator." It also subsumes ch.04 §2.4's "source injection" (the explosion `wave_source`/`atmosphere` deposits become queued ADD edits) and is complementary to the forward face-flux idea (flux *between* cells vs source *at* cells) — build FieldEdit first (three live call sites today), let face-flux reuse the queue + flush + stable-sort when it lands.
- **CUDA-ready:** a flat pre-sorted array of `FieldEdit` records is a kernel-launch list (one thread per edit, atomic combine), the same shape as the ray-list and gather stencils.

Suggested home: `src/simulation/field_edit.py`; lands as a new canon chapter **engine/13_field_edit.md** ("Depends on: 01 grid, 02 state, 03 materials"), not a loose doc.

---

## Open questions for Erik

1. **Temperature consumer now or first?** The §2/§3 feedback is cleanest with the real `temperature` field (heat -> conduction/relaxation -> temperature), which is *designed but unbuilt* in ch.06 §2. Ship the feedback now with `T_src = fire_intensity` as the stand-in (one-line swap later), or build the temperature consumer first so the law is "real" from day one? My lean: build the temperature consumer first — it is the single shipped-but-unconsumed gap that unlocks ignition, flashover, and conduction-spread all at once.
2. **Plume deposit shape:** own-tile-only (cleanest determinism) vs a small 3x3 outward kernel (visibly stronger plume push, but needs the Q16.16 saturating-add for order-independence). Lean own-tile to start; escalate only if the push reads too weak.
3. **Fuel source:** reuse normalized `wall_hp` as fuel `F`, or add a dedicated per-tile `fuel` field (lets floors/contents burn independently of wall structure)? Affects whether burnout and wall-failure are one mechanic or two.
4. **`I_seed` / sustain crossover:** confirm seed intensity ≈ 0.1 and that we *want* marginal ignitions to flicker and die (realistic) rather than always catch (gamier). This is a feel dial set by `k_grow`/`k_die` vs `drive_crit`.
5. **Constants live in `[physics.fire]` in `config.toml`** — fire params are currently hardcoded in `physics_runner.py` (the gap ch.06 flags). Agreed to move them as part of this work?
6. **FieldEdit scope:** is `wall_hp` a FieldEdit field (with the post-flush <=0 destruction sweep), or kept fully structural? And do we want `SET`/`MAX`/`MIN` modes from day one or just `ADD`/`REMOVE` until a consumer needs them?

---

## Proposed build order

Each step is independently testable and leaves `main` shippable. Determinism is validated at each step (single-machine first; cross-machine bit-exactness when the value enters Q16.16).

1. **Temperature consumer** (unblocks everything). Add `gmap.temperature` + the power-of-two relaxation pass that reads the `heat` buffer non-destructively (ch.06 §2, currently design-only), and quantize `ignition_temp` / wall thermal limits into Q16.16 once at load. No gameplay change yet — just materialize the field the rest depends on.
2. **Ignition reaction** (wire heat -> temperature -> fire). The ch.06 §4 rule: `temperature >= ignition_temp AND O2 -> fire = max(fire, I_seed)`. Still using the old growth/spread for now, so this only *adds* a second ignition path to validate the pipeline.
3. **Fire as ray sources** (§1). Build fire tiles into the ray-pass source list (LightSource profile, 8 rays, `range = 2 + 3·I`, deterministic angular dither). Delete the 12-connected cellular spread (`fire_simulation.cpp` 36-86) and retire the scalar `update_from_fire` path. Spread now *is* radiation -> heat -> temperature -> ignition. Keep clustering; add the global frame-coherent dirty-skip. **No** sparse cache.
4. **Feedback growth** (§2). Replace the fixed `+0.5·dt` (line 101) with the signed logistic update; add fuel read + `F -= k_burn·I·dt`. Validate self-limiting (fires die when cold/starved, accelerate when hot/supplied) in scalar `[0,1]` first.
5. **O2/pressure coupling** (§3). Delete the atmosphere-subtraction (lines 128-139); add the self-limiting own-tile plume deposit into `atmosphere`; switch the O2 model to the `P_src` read. This is also the latent-determinism-risk fix (removes the shared-cell scatter). Verify smoke now pushes *outward*.
6. **Port intensity to Q16.16** + cross-machine bit-exactness test, once temperature is fixed-point (ch.06 §3). Move all fire constants into `[physics.fire]` in `config.toml`.
7. **FieldEdit primitive** (§4) — can land in parallel from step 3 onward; it is independent infrastructure. Build `FieldEdit` + `EditQueue` + the one flush point; migrate the three existing call sites (`apply_explosion`, `add_explosion_smoke`, the planned ray burn-off) and re-express fire's smoke emission and the §3 plume deposit as FieldEdits. Write canon chapter engine/13. Do this before adding lasers/grenades/gas emitters so they are built on the primitive from day one.

Steps 1-2 unlock the pipeline; 3-5 deliver the visible fire redesign; 6 hardens determinism; 7 is the reusable foundation for all future emitters. Steps 4 and 5 are the two that directly resolve Erik's §5.4 and §5.7 comments; step 1 is the precondition both quietly depend on.

---

## Addendum — Erik's comments 5 & 6 (outside the research pass)

### Comment 5 — fixed-point: a class, or does it already exist?
> Erik: "exact fixed point — should we make a class out of it? or does it basically already exist?"

It already exists as **free functions + a constant**, not a class: in `cpp/src/raycaster.h`, `HEAT_SCALE = 65536` (Q16.16), a saturating float→Q16.16 quantize, and a saturating accumulator-add; the `heat` buffer is a raw `int32_t*`. **Recommendation:** don't wrap it in a class just for heat now — but Level-2 lockstep requires an **engine-wide fixed-point migration** (atmosphere, smoke, gas, and fire-intensity all go fixed-point — see this proposal's §2/§3 determinism notes and build-order step 6). *That* is when to promote it to one shared, header-only `Fixed16` type (a struct wrapping `int32` with saturating `+ − ×`, from/to-float, shifts) reused by every field. Build the class as part of that migration, not before — a heat-only class is premature.

### Comment 6 — water ↔ fire coupling, and chapter organisation
> Erik: water→vapour; 3 phase states (gas/liquid/solid; in vacuum only gas+solid); evaporation cools the tile; should temperature / water / fire be separate chapters?

Forward design (water/fluid is not built yet), but the coupling is clean and worth recording:
- **Water → vapour IS `white_smoke`** (ch.05 §6.2) — boiling/evaporation emits the white_smoke gas, so water plugs straight into the multi-gas system (a `FieldEdit` ADD on `white_smoke`, §4).
- **Latent heat couples to `temperature`**: evaporation is endothermic (cools the source tile); condensation releases heat. A (temperature, pressure) → phase map drives it — the same (T, P) the fire feedback (§2/§3) reads.
- **Phase states**: solid / liquid / gas, with correct vacuum behaviour (no liquid in vacuum — it flashes to vapour or freezes), matching the atmosphere's vacuum handling.
- **Chapter organisation**: keep as-is for now (temperature+fire in 06, water+fluid in 07, cross-referenced). Splitting *temperature* into its own chapter is a clean option later if it grows; log the water↔fire coupling as forward design in both 06 and 07 when water is built. No action now.
