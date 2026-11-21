# Blasts that make waves again: measurement, cause, options (#222)

2026-11-21. **Status: PROPOSED, v2.** No simulation code has changed. Erik
gives the go for any change after reading this.

**v2** folds the two critiques of section 10. Every change that answers a
finding carries its id (L1-n = lens 1, L2-n = lens 2), and section 10 ends
with a disposition table.

The headline change since v1 is a new option, (e): a splash set off by the
blast itself. It is now the recommendation, because the measured size and
sign of option (a)'s wave depend on the gas solver and the tick rate, not on
the blast (L1-1).

**How it was measured.**

- Measured on the head of arc #217 (`origin/217-eos-checkerboard`,
  `6405e471`), CPU build, in a detached worktree.
- **"Main's EOS"** means the same build with #217's three switches off
  (`subtick_loop`, `subtick_ceiling`, `cap_heat`), which is main's gas solver
  bit for bit (#217 §8.6.6).
- **"/15"** divides `[blast] expansion_m3_per_strength` and
  `heat_mj_per_strength` by 15 inside the process, near #217 §8.4's physical
  2.6 m³ and 0.9 MJ. No file was edited.
- Scratch tools, untracked: `C:\tmp\222\measure_pool.py` steps the level and
  records fields; `C:\tmp\222\twins.py` steps numpy twins of the ripple on
  those recorded fields.
- **PROVED** means measured, or read in the code. **INFERENCE** says so.

## 1. The problem

Erik, 2026-11-20: "after the arc closes - i'd actually like to fix
explosions again, so they interact with water ... perhaps we shall code
something with the explosions, that they will displace water - or turn water
into steam, a certain amount scaled with explosion strength ... it used to be
so much nicer". In #222: "we already have everything needed to compute it, we
only need to set off the splash."

**The scene.** `levels/water_studio`, the pool hall: 30 x 20 tiles
(10 x 6.7 m) at 0.333 m a tile, water 0.30 m deep. A `timed_charge` at
(18, 14) fires `[payloads.engine_death]` (strength 3, radius 3) at 8 s, which
is tick 192. Today that row means 120 m³ of expansion and 30 MJ of heat;
after the retune it would be about 7.8 m³ and 2.7 MJ (L1-9).

**Water exists in two layers:**

- **The real water.** `water_depth` and `flow_*`: a pipe model in Q16,
  synced, with a device twin. The air reaches it only as a pressure head,
  `surface += k_p·P`, with `k_p` = 0.5 m/atm.
- **The ripple.** `ripple` and `ripple_v`: a float wave, visual only.
  - It is never digested (`tests/field_digest.py:159`).
  - On the resident path it is stepped on the host (`physics_runner.py:1613`).
    `cuda_water.cu` does not port it.
  - Its only source is the splash, `ripple_v += k_splash·|P − P_prev|` once a
    tick on wet tiles, clamped to `±k_amp·depth`.
  - The water shader takes its normal from the ripple's slope. The roughness,
    the foam and W1's detail strength read `ripple_v` through
    `relative_agitation` (L1-5).

## 2. Measurements

### 2.1 The blast over the pool (#217 head, today's `[blast]` data)

Each row is the tick after the step. Δ values are against tick 191.

| tick | max \|P−P_prev\| over water (atm) | wind over the pool (m/s) | live ripple: peak slope | live ripple: share at the cap | real water: \|Δdepth\| (m) | real water: max flow (m/s) |
|---|---|---|---|---|---|---|
| 191 (rest) | 0.0003 | 0.6 | 0.0029 | 0 | — | 0.009 |
| 193 | 1.06 | 59.5 | 0.045 | 0.05 | 0.0005 | 0.008 |
| 194 | 1.26 | 27.0 | 0.028 | 0.24 | 0.002 | 0.145 |
| 195 | 0.08 | 11.5 | 0.005 | 0.87 | 0.006 | 0.139 |
| 196 | 0.02 | 11.8 | 0.006 | 1.00 | 0.010 | 0.129 |
| 204 | 0.0006 | 4.6 | 0.014 | 1.00 | 0.030 | 0.044 |

PROVED:

- **The transient lasts two ticks and is room-wide.**
  - Tick 193: P falls up to 0.89 atm within 1 m of the charge (ring mean), by
    0.37 at 1-2 m and 0.06 beyond 3 m.
  - Tick 194: P rises everywhere, +1.09 near the charge and +0.30 to +0.34 out
    to the far wall.
  - After that the room holds about +0.22 atm (the blast's heat).
- **The ripple goes to the cap and turns flat.**
  - The kicks are 2.0, then 4.1 m/s, with the same sign on every wet tile.
  - The whole pool is at the cap (0.147 m) by the third tick.
  - The slope is above twice rest for 19 ticks (0.8 s) in all. After tick 195
    that slope comes only from the cap edge `k_amp·depth`.
- **The real water moves 3.2 cm and 0.145 m/s**, for 2.1 s of agitation. A
  no-charge run settles by 2.4 mm and 0.016 m³ over the same window, so the
  blast moves about 0.087 m³ (L1-7). The renderer draws no normal from the
  real water.
- **No water law reads the wind.** The air reaches the water only through P.

### 2.2 The drift to the cap, with no blast (PROVED)

| time | ripple pool mean (m) | share at the cap | mean `ripple_v` (m/s) |
|---|---|---|---|
| 1 s | 0.010 | 0 | +0.013 |
| 8 s (when the charge would fire) | 0.098 (67 % of the cap) | 0 | +0.008 |
| 16.7 s | 0.146 | 0.39 | +0.004 |
| 20 s | 0.148 | 1.00 | +0.003 |

- With `k_splash = 0` the ripple stays exactly 0. The drift is the splash,
  fed by the solver's rest noise (|ΔP| of 1e-4 to 2e-3 atm a tick) taken as
  an absolute value.
- No value of `k_splash` fixes it.

### 2.3 Why it "used to be so much nicer" (PROVED, read in the code)

| period | splash source | blast pressure | what the ripple traced |
|---|---|---|---|
| 2026-06-10 (W6a) to 2026-07-10 | `k_splash·wave_p`, signed | into an explicit acoustic field at 66 tiles/s (22 m/s at 0.333 m), damping 3/s, wall reflection 0.9 | a fast ring that rang for about 1 s and reflected |
| EOS P3 (`8a439c34`, 2026-07-10) to 2026-11-06 | `k_splash·\|P − P_prev\|`, rectified | `pressure` column into the atmosphere | a 1-2 tick jump of the implicit solve |
| #31 (2026-11-06) to now | unchanged | divergence source + heat | as above, 1.06-1.26 atm |

- No dial has changed since 2026-06-10.
- INFERENCE: Erik remembers the W6a era. #31 is not the cause.
- That fast ring was the **air's** acoustic front drawn on the water. A real
  water wave in 0.3 m of water runs at √(g·h) ≈ 1.7 m/s, and the ripple runs
  at that speed. No option below brings back a 22 m/s ring without making the
  ripple faster than water (L1-2).

### 2.4 How #217 and the source retune move the real water (PROVED)

Each blast row has the no-charge settling (2.3-2.4 mm, 0.016 m³ on both
solvers) beside it (L1-7).

| run | peak \|ΔP\| (atm) | first-tick wind (m/s) | flow peak (m/s) | Δdepth peak (m) | volume moved (m³) | agitation |
|---|---|---|---|---|---|---|
| main's EOS, today's data | 1.40 | 132 | 0.052 | 0.014 | 0.022 | 0.7 s |
| #217 head, today's data | 1.26 | 59.5 | 0.145 | 0.032 | 0.103 | 2.1 s |
| main's EOS, /15 | 0.054 | 30 | 0.009 | 0.002 | 0.016 (= settling) | none |
| #217 head, /15 | 0.24 | 59.5 | 0.033 | 0.010 | 0.022 | 0.17 s |

- **#217 triples the water's response at the same source.** The retune cuts
  it about 4×.
- **The wind does not scale with strength.** The first-tick wind is 59.5 m/s
  at both strengths on #217: the 64 m/s ceiling binds.
- **Sub-ticks give the water no ring.** `wave_p` is written once a tick, and
  the water steps once a tick (INFERENCE).
- **`k_p` scales the water linearly**: at 2.0 it gives 0.57 m/s and 13 cm.
  Only the depth was checked (it stayed above 0.27 m); stability in general
  was not.

### 2.5 The two candidate splashes, measured on numpy twins

Both twins remove the rectified kick. Both run the ripple's own law
(`gamma_r` 2/s, `h_cap` 0.25 m, `k_amp` 0.5) on the recorded depth, solid and
P of each run.

- **(a)** forces through the pressure head with the boundary law of L1-4,
  ∇²(η + k_h·P), k_h = 0.5 m/atm.
- **(e)** puts a crater of A·strength at the blast's tile at its first ripple
  step, with A = 1 cm per strength (a dial) over the row's radius (1 m).

"Visible" means a slope above 0.003, today's live rest slope.

| splash | run | at the charge | peak height | peak slope | visible for | 1-mm front at +1 / +2 / +6 / +24 ticks (m) | ring speed | pool e-fold | back at the centre 3-5 s later |
|---|---|---|---|---|---|---|---|---|---|
| (a) | #217, today | MOUND, +4 → +13 mm | 3.0 cm | 0.023 | 1.8 s | 1.05 / 3.59 / 3.88 / 4.80 | 1.55 m/s | 0.46 s | 1.6 % |
| (a) | #217, /15 | mound, +1 → +4 mm | 0.9 cm | 0.012 | 0.75 s | 0.33 / 0.47 / 0.67 / 1.41 | 1.34 m/s | 0.38 s | 0.9 % |
| (a) | main's EOS, today | CRATER, −1 → −3 mm | 2.0 cm | 0.030 | 1.0 s | 0 / 0 / 0.33 / 1.41 (the largest structure away from the charge) | none from the charge | 0.21 s | 0.6 % |
| (a) | main's EOS, /15 | −0.1 mm | 0.13 cm | 0.002 | **0 s** (below rest) | none | — | 0.17 s | — |
| (e) | every run | crater, −2.5 cm | 2.95 cm | 0.038 | 1.25 s | 0.94 / 1.05 / 1.33 / 2.35 | 1.43 m/s | 0.38 s | 3.8 % |
| either | no blast, 20 s | — | (a) ~2e-5 m (L1-6); (e) exactly 0 | 0.0000 | 0 | — | — | — | — |

PROVED, measured:

- **(a)'s size and sign come from the gas solver, not the blast (L1-1).**
  - On #217 it is a 3 cm mound with a room-wide part (the 1 mm front is at
    3.6 m after two ticks, forced in place, L1-3), and a ring after that.
  - On main's EOS it is a weaker crater whose largest structure is not at the
    charge.
  - After the retune on main's EOS it is invisible.
  - Nearly all of its forcing lands in one end-of-tick sample, so its size
    scales with 1/tps (INFERENCE, L1-1).
- **(e) is the same in every run.** It reads only the blast's position,
  radius and strength. It is a local pulse that runs out as a ring at
  1.43 m/s.
- **Neither ripple wave lives long.** The pool e-fold is about 0.4 s because
  `gamma_r` = 2/s, and wall reflections come back at a few percent. A longer
  ring and visible reflections are a `gamma_r` look question.
- **Neither reaches the cap** (largest share at the cap: 0.000).

## 3. The cause

- **The splash's shape is wrong** (PROVED). It is a rectified, room-wide kick
  once a tick:
  - pressure noise lifts the whole pool to the flat cap in about 20 s;
  - a blast's two-tick, room-wide transient flattens it in three ticks.
- **It started with EOS P3 (2026-07-10), which removed the signed fast
  acoustic ring the ripple used to trace.** #31 did not cause it.
- **Merely re-shaping the pressure source does not give a dependable wave**
  (L1-1). At 24 tps the gas solver's pressure at the water is a one-tick
  sample whose size and sign change with the solver version and the tick
  rate.

## 4. Options

**Common to every option: remove the rectified |ΔP| kick.** It is the drift.
Removing it alone leaves still water flat and blasts invisible on the ripple.

### (a) The pressure-head splash

- **Law** (L1-4): `ripple_v += dt·(c²·∇²(η + k_h·P) − gamma_r·ripple_v)`.
  - P is read on every non-solid tile.
  - η is 0 on dry tiles.
  - The sum is mirrored at solid tiles.
  - No global mean is needed. v1's "zero-mean in space" is withdrawn.
- **What Erik sees.** On today's #217: a 3 cm bulge over the charge with a
  room-wide stir, then a ring at about 1.5 m/s, visible for about 1.8 s. It
  becomes a crater and much weaker on main's solver, invisible after the
  retune there, and it changes with the tick rate.
- **Side effect.** A standing pressure difference leaves a standing tilt, for
  example a breach to vacuum leaves η at the cap (L1-10).

### (e) The blast-triggered splash (new)

- **Law.** On the first ripple step after a detonation, add a crater
  `Δη = −A·strength·(1 − r²/R²)₊` over the row's radius R. From there the
  ripple's own wave law carries it.
  - A (m per strength) is a look dial. Linear in strength by default; the
    curve is Erik's.
- **Trigger.** It reads the blast's `ExplosionEvent` (position, radius,
  strength). That is the render cue the shock front, sparks and W3's spray
  already read.
  - `GameMap.blast_log` is THE seam for slot-9e entities, but it is not fit
    here. Its tuples carry no strength (`payloads.py:488`). It is cleared in
    slot 9e of the same tick, before the next slot-7 ripple step
    (`simulation.py:2454`). Its device form is a Class-T table.
  - The tick's explosion events reach `step_ripple` as a short list, so the
    step stays the ripple's one writer. The one-tick latency is 42 ms.
- **What Erik sees** (at A = 1 cm per strength, strength 3).
  - A 3 cm crater at the charge throws a ring at about 1.4 m/s: 1.3 m out at
    0.25 s, 1.7 m at 0.5 s, 2.35 m at 1 s.
  - It is visible for about 1.25 s, plus whatever foam it raises.
  - The first wall echo comes back at 3-5 s at about 4 % (unseen at
    `gamma_r` = 2).
  - The same on either solver, before and after the retune, at any tick
    rate. The size is set by A and the strength.

### (a2) Draw the real water's surface (a later render step)

- **The change.** The normal also reads the simulated surface's slope.
- **What it shows.** The real 1-3 cm wave.
- **Collision.** It runs into `render-effects.md:92`, "never a second wave
  source". That needs a rule amendment for Erik later (L2-10). Not proposed
  now.

### (b) Displacement: a blast moves water directly, scaled by strength

As v1, with the ruling restated as a physics-model question (L2-8): may a
blast move water beyond what its air does?

- **Cost.**
  - Synced Q16 water fields.
  - The payload function's device twin, in the same patch.
  - `water_depth` is in `DEVICE_REFUSED_FIELDS`, so ticks with such a blast
    fall back to a host flush.
  - The momentum form needs new FieldEdit policies for `flow_*`.
  - A `[blast]` per-strength column.
- **When.** Decide after the retune, with the numbers then in hand.

### (c) Steam

A statement, not a ruling: blast steam belongs to #219.

- Water has no heat. W5's flash-boil is pressure-keyed and moves depth into
  the `gas[steam]` trace plane with no energy.
- A retuned frag vaporises at most about 0.35 kg of water: a white puff, no
  waves.
- Closing the energy books for it needs water with heat.

### (d) Wind stress on the water: rejected

- At physical values Δv ≈ 0.0015 m/s.
- The wind is capped at 64 m/s, so it does not scale with strength.
- It would move `GOLDEN_AGGREGATE`.

## 5. Recommendation

**(e), with the rectified kick removed.** Reasons:

- **It is dependable.** It is the only option whose wave is the same on both
  gas solvers, before and after the retune, and at any tick rate. Erik's dial
  A sets its size per strength. (a)'s wave flips sign between solvers and
  vanishes after the retune on main's.
- **It is centred where the charge is**, a local pulse that travels as a
  ring. (a) on #217 stirs the whole room in two ticks.
- **It is exactly "set off the splash"**, and it is render-only:
  - no synced field, no device code, no golden;
  - the ripple's wave law, speed and clamp are unchanged.
- **(a) is kept as the alternative** if Erik wants the water to answer air
  pressure itself (doors, vents), knowing the size then moves with the
  solver.
- **The real water's pressure coupling `k_p` stays as it is.** It and A are
  dials for the blast-and-vent retune session, on post-#217 main (L2-9).

## 6. What the patch touches (L1-5, L1-8, L1-10, L2-2 to L2-7, L2-9)

**Consumers whose look changes.** The room-wide heave they were built around
disappears.

- `relative_agitation` (`renderer/water_surface.py:591`) feeds:
  - the roughness (`water.fs:517-519`, `u_roughness_agitation`);
  - the foam's steep-surface source (`renderer/water_foam.py:52-56`);
  - W1's detail strength (`water_surface.py:621`).
- Two rule lines justify the relative agitation by that heave:
  `render-effects.md:96` (Water flow detail) and `:107` (Water foam). Both
  are amended in the patch.
- The #97 still-pool look is re-looked (ruling 2).

**Tests that encode the old law**, each rewritten as a property in the same
patch:

- `test_water_ripple.py`:
  - `_splash_source` / `_splash_rollout` (:89-97, uniform P; used at :186,
    :270);
  - `test_amplitude_clamp_holds_and_engages` (:255; it goes vacuous: drive it
    with an impulse, L1-8);
  - keep `test_point_splash_front_propagates` (:203) and
    `test_visual_only_ab_rollout_transport_bit_identical` (:348) on the new
    source.
- `test_headless_ripple.py` (:49, :67).
- `test_a5_seal_evacuation.py:128`.
- `tests/cuda_s8a_check.py:40`, **the resident-path gate**: the host ripple on
  the CPU path and on the resident path, byte for byte. The splash list must
  reach both the same way.

**The C++ side.**

- `water_solver.{h,cpp}` (`step_ripple`: the source and the event list),
  `physics_engine.{h,cpp}` (`step_ripple`, `step_tail`), `bindings.cpp`, and
  the callers in `physics_runner.py` (:1156, :1615).
- `p_prev` leaves the signature under (e).
- `stubs/breach_physics.pyi` regenerated.
- `config.toml:702`: `k_splash` → `splash_m_per_strength`.

**The float ratchet** (L2-3). `tests/test_no_float_in_sim_tu.py:391` pins
`water_solver.cpp` at 32 float lines.
- (e) removes the |ΔP| lines and adds the crater. Hold the count, or
  re-baseline with a written exemption (render-only float).
- v1's alternative of a separate TU is dropped: it costs a `/fp:strict` row
  and a ratchet row and saves nothing.

**The twin ledger** (L2-5). `[twins.water]`, `[twins.water_tail]` and about a
dozen twins list `water_solver.h`; `bindings.cpp` and `physics_runner.py` are
water-twin files.
- Mark every twin `tools/twin_debt.py` reports.
- One `tools/cuda_green.py` run.
- The device result is unchanged by construction.

**The order** (L2-9). #217 touches `bindings.cpp`, `physics_runner.py` and
`config.toml` too.
- Cut the patch from main after #217 merges.
- Whichever merges second needs a fresh `.cuda_green`.
- Tune A on post-#217 main.

## 7. Verification plan (properties, never snapshots)

1. **No drift.** `water_studio` without the wave charge: over 30 s the
   ripple stays within 1 mm of 0 everywhere. Breaks if any per-tick |ΔP| or
   other rectified source returns (today's law reaches the cap at 16.7 s).
2. **Pressure alone makes no ripple** (under e). A sealed flat pool with a
   uniform heat, and a pool beside a door that opens: the ripple stays 0.
   Breaks if the source reads P.
3. **A blast makes a ring that travels.** On a local pulse in a unit scene
   (reusing `test_point_splash_front_propagates`, L1-3), the 1-mm front's
   radius grows at √(g·min(h, h_cap)) within a band, and the slope returns
   under twice rest in bounded time. Breaks if the impulse is missing,
   saturates or does not propagate.
4. **Independent of the solver** (under e). The same detonation with #217's
   switches on and off gives the same ripple to float tolerance. Breaks if
   the splash reads the gas.
5. **Monotone in strength.** Strength 1 gives a smaller peak than strength 3.
   Breaks if the cap or saturation binds.
6. **Render-only.** Reuse `test_water_ripple.py:348` and
   `test_headless_ripple.py:67` on a local bump; `cuda_s8a_check` PART 1a for
   the resident path (L2-6). Breaks if the ripple feeds back or the paths
   differ.
7. **Dormancy.** A dry map never touches the ripple.
8. **Look** (L2-7). The whole shot set is rendered before and after. A new
   `pool_blast_wave` row is added for the ring about 1 s after the blast;
   `pool_blast_spray` (+0.58 s) and `wade_by_firelight` are watched for the
   agitation change.

## 8. Rulings for Erik (each with the default the build uses)

1. **Which wave.**
   - (e) a splash set off by the blast: the same wave every time, sized by
     strength.
   - (a) a wave driven by the air pressure: it also answers doors and vents,
     but its size and sign follow the gas solver and the tick rate.
   - Either way the ring runs at water's own speed, about 1.5 m/s, not the
     old 22 m/s air front.
   - **Default: (e).** Its size per strength (and `gamma_r`, for how long it
     rings) are Erik's dials, tuned by eye later.
2. **The #97 still-pool look will change.** The room-wide heave goes, so the
   roughness, foam and detail that read the agitation get a re-look in the
   same session. **Default: yes.**
3. **May a blast move the water itself beyond what its air does** (option b)?
   **Default: not now.** Decide in the blast retune, as volume moved per
   strength in `[blast]` if yes.

## 9. Systems

**(a) Existing canonical systems this design must use.**

- The `ExplosionEvent` render cue (the trigger).
- Water flow detail and Water foam (consumers, rule lines amended).
- Look shots.
- The A/B lockstep harness and `GOLDEN_AGGREGATE` (render-only proof), and
  `cuda_s8a_check` (the resident gate).
- Class P / Class T + the twin ledger.
- Suite timing and test conventions.
- The E2E driver.
- For (b) later: the Payload executor and its twin, Blast reach, the
  Fixed-total tile deposit, FieldEdit, the device edit door.

**(b) New: the rule needs a home** (L2-1). No CLAUDE.md row and no rule
file's `paths:` covers the water solver today.

- **One CLAUDE.md index line, Sim core:**
  `| Water solver + ripple | cpp/src/water_solver.* + cpp/src/cuda_water.cu + GameMap water fields | The real water is the Q16 pipe model, reached by the air only through the k_p head; the ripple is render-only float, never synced or fed back, set off only by the blast splash |`.
- **Constraints in `fields-gas.md`**, whose `paths:` gains
  `cpp/src/water_solver.cpp`, `cpp/src/water_solver.h` and
  `cpp/src/cuda_water.cu`, and whose row in the region table gains "water
  solver + ripple". The section says:
  - the ripple is host-only, never digested and never read by a simulation
    law;
  - `step_ripple` is its one writer;
  - its source is the splash of the tick's explosion events;
  - never a per-tick kick from |ΔP| or any rectified source (incident #222:
    the pool drifted to the flat cap in 20 s, and a blast flattened it).

## 10. Critique

| lens | model | verdict | findings |
|---|---|---|---|
| 1. Physics, numerics, measurement soundness (2026-11-21, against f263b718) | Opus | SOUND WITH FIXES | Diagnosis and the render-only claim confirmed (ripple not digested: `tests/field_digest.py:159`, `field_digest_spec.toml:86`; host-only on the resident path, `physics_runner.py:1613-1617`, `applier.py:571`; no obs or sim law reads it); CFL stays `ripple_max_dt` 0.106 s. (1) MAJOR PROVED: the wave's size and SIGN come from the EOS, not the blast: on the detonation tick P minus pool mean within 1 m of the charge is -0.73 atm on #217 (a mound) and +0.10 atm on main's EOS (a crater, 7x weaker); after /15 main gives 1.3 mm / slope 0.002, below the rest slope; nearly all forcing lands in one end-of-tick sample, so it scales with 1/tps (INFERENCE). Re-measure on merged main and after the retune before choosing k_h; weigh a render-only ripple_v impulse read from `GameMap.blast_log` (strength, position), independent of the EOS and the tick rate. (2) MAJOR: (a) does not bring back the remembered look: W6a's ring rode a 22 m/s acoustic front; the new one moves at ~1.5 m/s and decays 1/e per s, its first wall reflection back after ~3.4 s at ~3 %; describe it honestly, which look is Erik's call. (3) MAJOR PROVED: most of the 'travelling' extent is forced in place (|d eta| > 1 mm radius 1.05 m at t=193, 3.59 m at t=194, then ~1 m/s); verification test 3 would fail on this scene; measure speed on a local pulse in a unit scene. (4) MAJOR PROVED: boundaries unspecified and the naive build wrong: P on solid tiles is 0, the twin's Dirichlet shore deforms any room whose P differs from the global mean; specify lap(eta + k_h P) with P read on every non-solid tile, eta = 0 dry, the sum mirrored at solid; then no global mean is needed (reword 'zero-mean in space'). (5) MAJOR PROVED: the still-water look changes too: roughness (`water.fs:517-519`, `roughness_agitation`), foam and W1 detail read ripple_v via `relative_agitation`, built around the uniform heave; list the consumers, expect a re-look of #97 dials, add `pool_blast_spray` and `wade_by_firelight` shots. (6) MINOR: 'exactly 0' with no blast is ~2e-5 m. (7) MINOR: subtract the no-charge baseline in 2.4 (0.016 m3 / 2.4 mm settling). (8) MINOR: `tests/test_water_ripple.py::test_amplitude_clamp_holds_and_engages` uses a uniform splash and will go vacuous; reuse `tests/test_headless_ripple.py`. (9) MINOR: rejections sound; state today's engine_death mass (~11.5 kg, 30 MJ) beside the retuned. (10) MINOR: list `physics_engine.{h,cpp}`, `bindings.cpp` and callers if `p_prev` leaves the signature; a breach to vacuum leaves a standing eta at the cap. |
| 2. Systems reuse, project rules, gates, scope (2026-11-21, against f461e8b4; independent of lens 1) | Opus | SOUND WITH FIXES | No blockers; (a) outside the determinism gates confirmed (`applier.py:571`, `physics_engine.h:172-177`, `physics_runner.py:1608-1611`); (c) to #219 and (d) rejected right; #218 untouched. (1) MAJOR PROVED: the draft rule has no home: no CLAUDE.md row for the water solver / ripple, no rule file's `paths:` covers `water_solver.*` / `cuda_water.cu`; add ONE index line ('Water solver + ripple') and the constraints in an existing region with the paths added; the rule says render-only, never fed back, forced only by the pressure head, never a rectified per-tick kick (incident #222). (2) MAJOR PROVED: two live rules go stale: `render-effects.md:96` / `:107` (and `water_foam.py:52-56`, `water_surface.py:621`) justify relative agitation by the ripple's room-wide heave, which (a) removes; list Water foam and `relative_agitation` as affected, amend both, and give Erik the look ruling (the #97 still-pool look changes; re-look its dials). (3) MAJOR PROVED: the float ratchet is a hard gate: `tests/test_no_float_in_sim_tu.py:62,:391` pins `water_solver.cpp` at 32 float lines; hold the count or re-baseline with a written exemption; drop or honestly cost the own-TU alternative (/fp:strict list, ratchet row). (4) MAJOR PROVED: pre-existing tests encode the old law: `test_water_ripple.py:89-97` (`_splash_source`, uniform P, used at :186, :270; `_splash_rollout` None = no splash), `test_headless_ripple.py:49`, `test_a5_seal_evacuation.py:128`, `tests/cuda_s8a_check.py:40` (host ripple CPU vs resident byte for byte, the real resident-path gate); rewrite each as a property of the new law in the same patch. (5) MINOR: Class P duty understated: `[twins.water]`, `[twins.water_tail]`, `water_solver.h` in ~a dozen twins, `bindings.cpp` a build path, `physics_runner.py:888` a water-twin file; mark every twin `twin_debt.py` reports, regenerate `stubs/breach_physics.pyi`, the `config.toml:702` rename. (6) MINOR: property 5 rebuilds an existing test the wrong way (`field_ab_harness.py:87` has the ripple in SIM_FIELDS); reuse `test_water_ripple.py:348` and `test_headless_ripple.py:67` on a local bump. (7) MINOR: render the whole shot set (`render-core.md:35`); add a blast-wave shot row rather than repurpose the still-pool shots (frame 240 would hold a pool-wide wave). (8) MINOR: ruling 2 misquotes the rule: `explosions-fire.md:32` is about UNITS; restate as a physics-model question (may a blast move water beyond what its air does?). (9) MINOR: the #217 order: #217 touches `bindings.cpp`, `physics_runner.py`, `config.toml` too; cut (a) from main after #217 merges, the second to merge needs a fresh `.cuda_green`; k_h tuned only on post-#217 main. (10) MINOR: (a2) collides with `render-effects.md:92` 'never a second wave source'; name it as a future rule amendment for Erik. |

### Disposition (v2)

| id | severity | where folded / why not |
|---|---|---|
| L1-1 | MAJOR | §2.5 (both EOS × full and /15 on the twins: a mound on #217, a crater on main's, invisible at /15 there), §3, §5; new option (e) in §4, measured, now the recommendation. The 1/tps scaling is kept as INFERENCE, not measured. |
| L1-2 | MAJOR | §2.3 (22 m/s air front vs ≈1.7 m/s water), §2.5 (measured 1.34-1.55 m/s), §4 "what Erik sees", ruling 1 |
| L1-3 | MAJOR | §2.5 (the 1-mm front at 3.6 m after two ticks: forced in place); test 3 moved to a local pulse in a unit scene |
| L1-4 | MAJOR | §4 (a) boundary law; the twin was re-run with it (`twins.py`); "zero-mean in space" withdrawn |
| L1-5 | MAJOR | §1, §6 consumer list, ruling 2, test 8 (`pool_blast_spray`, `wade_by_firelight`) |
| L1-6 | MINOR | §2.5: (a) ~2e-5 m, (e) exactly 0 |
| L1-7 | MINOR | §2.1 and §2.4: the no-charge settling (2.3-2.4 mm, 0.016 m³) beside each row |
| L1-8 | MINOR | §6: the clamp test driven by an impulse; reuse of `test_headless_ripple.py` |
| L1-9 | MINOR | §1: today's 120 m³ / 30 MJ against about 7.8 m³ / 2.7 MJ after the retune. The kg figure is not restated: the mass depends on which energy is counted. |
| L1-10 | MINOR | §6 (`physics_engine`, `bindings`, callers, `p_prev`); §4 (a): a breach to vacuum leaves a standing η at the cap |
| L2-1 | MAJOR | §9 (b): one index line, the constraints in `fields-gas.md` with `paths:` added |
| L2-2 | MAJOR | §6 (Water foam, `relative_agitation`, `render-effects.md:96` / `:107` amended), ruling 2 |
| L2-3 | MAJOR | §6: hold 32 or re-baseline with an exemption; the separate-TU alternative dropped |
| L2-4 | MAJOR | §6 test list, with `cuda_s8a_check` as the resident-path gate |
| L2-5 | MINOR | §6: mark every twin `twin_debt.py` reports, stubs, the config rename |
| L2-6 | MINOR | §7 test 6 reuses `test_water_ripple.py:348`, `test_headless_ripple.py:67` and `cuda_s8a_check` |
| L2-7 | MINOR | §7 test 8: the whole shot set plus a new `pool_blast_wave` row |
| L2-8 | MINOR | §4 (b) and ruling 3 restated as a physics-model question |
| L2-9 | MINOR | §6 "The order"; A tuned on post-#217 main |
| L2-10 | MINOR | §4 (a2): a future rule amendment of `render-effects.md:92`, not proposed now |
