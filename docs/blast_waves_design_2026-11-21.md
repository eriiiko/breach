# Blasts that make waves again: measurement, cause, options (#222)

2026-11-21. **Status: PROPOSED.** No simulation code has changed. Erik gives
the go for any change after reading this.

Measured on the head of arc #217 (`origin/217-eos-checkerboard`, `6405e471`),
because #217 is about to merge and changes blasts a lot. CPU build in a
detached measurement worktree. The scratch probe is `C:\tmp\222\measure_pool.py`
(untracked, read-only: it steps the level and reads fields, and changes no
code or config file). **PROVED** means measured, or read in the code.
**INFERENCE** says so.

## 1. The problem

Erik, 2026-11-20: "after the arc closes - i'd actually like to fix
explosions again, so they interact with water ... perhaps we shall code
something with the explosions, that they will displace water - or turn water
into steam, a certain amount scaled with explosion strength ... it used to be
so much nicer". In #222 he remembers blasts making beautiful waves: "we
already have everything needed to compute it, we only need to set off the
splash."

The scene is `levels/water_studio`, the pool hall: 30 x 20 tiles (10 x 6.7 m)
at 0.333 m a tile, water 0.30 m deep. A `timed_charge` standing in the water at
(18, 14) fires `[payloads.engine_death]` (strength 3, radius 3) at 8 s. At
24 tps the detonation is tick 192.

Water exists in two layers today:

- **The real water.** `water_depth`, `flow_vx` and `flow_vy`: a pipe model in
  Q16, synced, with a device twin (`cpp/src/water_solver.cpp::step`,
  `cpp/src/cuda_water.cu`). It reads the air only through a pressure head,
  `surface += k_p * P`, with `k_p` = 0.5 m/atm.
- **The ripple.** `ripple` and `ripple_v`: a float wave, visual only, never
  synced, stepped on the host (`WaterSolver::step_ripple`; `cuda_water.cu`
  says it is not ported). Its only source is the "splash":
  `ripple_v += k_splash * |P - P_prev|` once per tick on wet tiles. The result
  is clamped to `±k_amp * depth`. The water shader takes its surface normal
  from the ripple's slope (`shaders/water.fs` section A, `surfaceHeight` =
  the ripple texture) plus the procedural sines.

## 2. Measurements

### 2.1 The blast over the pool, #217 head, today's `[blast]` data

Each row is the tick after the step. The detonation is at 192. Δ values are
taken against tick 191.

| tick | max \|P−P_prev\| over water (atm) | wind over the pool (m/s) | live ripple: peak slope | live ripple: share of pool at the cap | real water: max \|Δdepth\| (m) | real water: max flow (m/s) |
|---|---|---|---|---|---|---|
| 191 (rest) | 0.0003 | 0.6 | 0.0029 | 0 | — | 0.009 |
| 193 | 1.06 | 59.5 | 0.045 | 0.05 | 0.0005 | 0.008 |
| 194 | 1.26 | 27.0 | 0.028 | 0.24 | 0.002 | 0.145 |
| 195 | 0.08 | 11.5 | 0.005 | 0.87 | 0.006 | 0.139 |
| 196 | 0.02 | 11.8 | 0.006 | 1.00 | 0.010 | 0.129 |
| 204 | 0.0006 | 4.6 | 0.014 | 1.00 | 0.030 | 0.044 |
| 240 | 0.0008 | 1.4 | 0.004 | 1.00 | 0.006 | 0.030 |

PROVED, all measured:

- **The splash source is large, not small.** The transient over the water is
  1.06 atm, then 1.26 atm, summed over the pool to 93 and then 235 atm·tiles.
  It lasts **two ticks**: from tick 195 on it is below 0.03 atm everywhere.
- **The transient has no travelling front.** On tick 193, P *drops* by up to
  0.89 atm within 1 m of the charge (ring mean), by 0.37 at 1-2 m and 0.06
  beyond 3 m. On tick 194 it rises everywhere, by +1.09 near the charge and
  by +0.30 to +0.34 from 3 m to the far wall. After that the room holds about
  +0.21 to +0.25 atm (the blast's heat). Across the 10 m hall the pressure
  settles inside one tick.
- **The ripple is kicked to the cap and goes flat.**
  - `k_splash` × the transient gives `ripple_v` 2.0 m/s on tick 193 and
    4.1 m/s on tick 194, with the same sign on every wet tile.
  - The share of the pool at the cap goes 5 % → 24 % → 87 % → 100 % in three
    ticks. The cap is 0.147 m at 0.294 m of depth.
  - The visible slope peaks at 0.045 on the detonation tick and falls to
    0.005 two ticks later.
  - The surface is above twice its rest slope for 19 ticks in all (0.8 s).
    After tick 195 that slope comes only from the cap edge, `k_amp × depth`,
    which follows the real water's depth wave at half its slope.
- **The real water moves, but little.**
  - Peak flow 0.145 m/s, peak depth change 3.2 cm (tick 207), volume moved
    0.10 m³ (of about 20 m³ in the pool).
  - Flow stays above twice its rest level for 2.1 s.
  - Its surface slope peaks at 0.028, against 0.004 at rest. The renderer
    draws no normal from it.
- **Nothing in the water reads the wind.**
  - `WaterSolver::step` reads the air only as `k_p * atmosphere` in the
    surface potential, and `step_ripple` reads only P and P_prev
    (`cpp/src/water_solver.cpp`; read in the code).
  - The divergence source's wind (59.5 m/s over the pool on the detonation
    tick) reaches the water only through the pressure it leaves behind.

### 2.2 The drift to the cap, with no blast

PROVED, measured. The run is the same scene with the wave charge removed.

| time | ripple pool mean (m) | share at the cap | mean `ripple_v` (m/s) | mean \|P−P_prev\| (atm) |
|---|---|---|---|---|
| 1 s | 0.010 | 0 | +0.013 | 0.0004 |
| 4 s | 0.058 | 0 | +0.013 | 0.0007 |
| 8 s (when the charge would fire) | 0.098 (67 % of the cap) | 0 | +0.008 | 0.0002 |
| 16.7 s | 0.146 | 0.39 | +0.004 | 0.0002 |
| 20 s | 0.148 | 1.00 | +0.003 | 0.0001 |

- With `k_splash = 0` the ripple stays exactly 0 for the whole run. **The
  drift is all splash.**
- The solver's rest noise in a quiet sealed room is |ΔP| ≈ 1e-4 to 2e-3 atm a
  tick. The source takes its absolute value, so every tick pushes `ripple_v`
  up. Damping (`gamma_r` 2/s) cannot remove a mean that is fed every tick, so
  the pool climbs to the flat cap in about 16-20 s.
- The cap does not hide the splash; the splash fills the cap. **No value of
  `k_splash` fixes this**: any k > 0 drifts, and k = 0 has no splash at all.

### 2.3 Why it "used to be so much nicer" (history, read in the code)

| period | splash source | blast pressure | what the ripple saw |
|---|---|---|---|
| 2026-06-10 (W6a, `ae982841`) to 2026-07-10 | `k_splash * wave_p`, SIGNED | a pressure column into `wave_source`: an explicit acoustic field at `wave_c` = 66 tiles/s, damping 3/s, walls reflecting 0.9; the frag deposited 10 atm over radius 5 | a ring that travels, rings for about 1 s, reflects, and averages to zero |
| 2026-07-10 (EOS P3, `8a439c34`) to 2026-11-06 | `k_splash * \|P − P_prev\|`, RECTIFIED | `pressure` column, an atmosphere ADD (engine_death 5 atm, radius 3; frag 10 atm until the 2026-10-03 retune, then 0) | a 1-2 tick jump of P from the implicit solve, the same sign everywhere |
| 2026-11-06 (#31) to now | unchanged | divergence source + heat; the door refuses `pressure` | as above; measured 1.06-1.26 atm |

- PROVED (code): W6a's source was the acoustic anomaly itself, signed, with a
  finite speed. EOS P3 replaced it by |P − P_prev| of an implicitly solved P.
  That swap took away the ring (no travelling front at 24 tps) and the zero
  mean (the drift).
- `k_splash` (2.0), `k_p` (0.5), `k_amp`, `gamma_r` and `h_cap` have not
  changed since 2026-06-10 (`git log -G` on `config.toml`).
- INFERENCE: the waves Erik remembers are from the W6a era, June-July 2026.
  **#31 is not the cause.** Its transient over the water is large enough to
  saturate the ripple, and the drift alone flattens the pool in 20 s with no
  blast at all.
- A pre-#31 build was not made. The code settles the cause, and the studio
  did not exist at those commits.

### 2.4 How #217 and the coming source retune move it

PROVED, measured on the same build. The rows marked "seams off" turn off
#217's three switches (`subtick_loop`, `subtick_ceiling`, `cap_heat`), which
is main's EOS bit for bit (§8.6.6 of the #217 design). "/15" divides
`[blast] expansion_m3_per_strength` and `heat_mj_per_strength` by 15 in the
process (2.67 m³ and 0.67 MJ, near the 2.6 m³ / 0.9 MJ of #217 §8.4). No file
was edited.

| run | peak \|ΔP\| over water (atm) | first-tick wind (m/s) | flow peak (m/s) | Δdepth peak (m) | volume moved (m³) | flow agitation lasts | ripple at the cap 3 ticks later |
|---|---|---|---|---|---|---|---|
| main's EOS (seams off), today's data | 1.40 | 132 | 0.052 | 0.014 | 0.022 | 0.7 s | 100 % |
| #217 head, today's data | 1.26 | 59.5 | 0.145 | 0.032 | 0.103 | 2.1 s | 100 % |
| main's EOS, /15 | 0.054 | 30 | 0.009 | 0.002 | 0.016 | none | 0 % (drift only) |
| #217 head, /15 | 0.24 | 59.5 | 0.033 | 0.010 | 0.022 | 0.17 s | 4 %, all of it by 11 s |

- The main row reproduces the issue's numbers: about 0.05 m/s and 1-2 cm.
- **#217 makes the real water's response about three times larger** at the
  same source. INFERENCE: the sub-ticked solve leaves more pressure gradient
  over the water.
- **The retune makes it about 4× smaller again**: 0.033 m/s and 1 cm. P falls
  by 5×, not 15×.
- **The ripple still saturates after the retune.** The drift has it at 67 % of
  the cap before any blast, and the rectified kick (0.4-0.8 m/s) lifts the
  whole pool.
- **Sub-ticks give the water no ring.** `wave_p` is written once a tick, at
  entry (#217 §8.6.1). The water and the ripple step once a tick, after the
  EOS, so they see only the tick's net change. At dt_k = 5 ms a front would
  exist *inside* the tick, but nothing samples it (INFERENCE from §8.6.1 and
  the call order).
- **The blast's wind no longer scales with strength over the pool.** The
  first-tick peak is 59.5 m/s at both strengths: #217's 64 m/s ceiling binds
  for anything at or above about 1/15 of today's source. INFERENCE for the
  /15 row: its one-tile source asks for about 58 m/s, just under the ceiling.
- **A dial check.** `k_p` = 2.0 (4×) scales the real water's response
  linearly: 0.57 m/s, 13 cm and 4.9 s at today's data; 0.093 m/s and 2 cm at
  /15. Pool depth stayed above 0.27 m in 20 s (only that was checked, not
  stability in general).

### 2.5 A probe of option (a), on the measured fields

The probe steps a numpy twin of `step_ripple` beside the live one, fed the same
P and depth each tick (§4a).

The twin uses the surface-pressure forcing of linear water waves:
∂²η/∂t² = c²∇²(η + k_h·P), with k_h = `k_p` = 0.5 m/atm. It is the law the
real water already obeys, written in the ripple's float.

| run | peak height | peak slope | slope above twice rest for | extent of \|Δη\| > 1 mm | no blast, 20 s |
|---|---|---|---|---|---|
| today's data | 3.0 cm | 0.023 | 2.9 s | 6.3 m (the whole pool) | exactly 0 |
| /15 | 0.9 cm | 0.012 | 1.4 s | 1.5 m | exactly 0 |

Two other twins were tried:

- A **signed** |ΔP| (the old shape with the sign kept) still drifts. Its sum
  telescopes to P(t) − P(0), so the room's slow pressure rise becomes a level
  offset that sits at the cap.
- A **rectified source with no cap** climbs to 1.36 m.

So **the source must be zero-mean in space**, not only signed.

## 3. The cause

- **The splash has the wrong shape, not the wrong size.** It is a rectified,
  spatially uniform, once-a-tick kick: every tick of pressure noise lifts the
  whole surface. So:
  - The ripple drifts to its flat cap in about 20 s with no blast.
  - A blast's two-tick, room-wide transient (1 atm today, about 0.2 after the
    retune) slams the whole pool to the cap in three ticks. A flat surface
    shows nothing.
- **The change came with EOS P3 (2026-07-10), not with #31.** Before P3 the
  ripple traced a signed acoustic ring that travelled and reflected. The
  implicit solve has no such ring at 24 tps, and the rectified source has no
  zero mean.
- **The real water does make a small true wave** (3 cm on #217, 1 cm after the
  retune), through the pressure head. The renderer does not draw its surface,
  and the clamped ripple shows it only as a cap edge.

## 4. Options

### (a) Restore the pressure splash: a zero-mean source (recommended)

**The change.** Replace the source in `step_ripple`: no |P − P_prev| kick.
Instead the ripple's restoring term includes the pressure head,
`ripple_v += dt * (c² ∇²(ripple + k_h·P) − gamma_r·ripple_v)`. It is the same
surface-pressure law the pipe model uses (`surface += k_p·P`), so the ripple
is the fine, lightly damped visual twin of physics already there. `k_splash`
is retired and replaced by `k_h` (m/atm), a feel dial whose default is `k_p`.

**What Erik would see.**
- Still water stays still: no drift, no flat cap.
- A blast pushes a dip or a bulge around the charge that runs outward as a
  ring at the ripple's speed, c = √(g·min(depth, h_cap)) ≈ 1.6 m/s, and
  reflects off the walls.
- Its size is set by the blast's pressure, so it scales with strength. On
  today's data: 3 cm and 2.9 s across the whole hall. After the retune: about
  1 cm, 1.4 s and 1.5 m. `k_h` sets the look, with no effect on gameplay.

**Systems used.** The Water solver's ripple (render-only), the light curve and
the water shader unchanged, Look shots (`still_pool_flashlight` and
`still_pool_closeup` at frame 240, 2 s after the 8 s blast).

**Cost.**
- Determinism: none. The ripple is float, unsynced and host-only, with no
  device twin.
- The digest, `GOLDEN_AGGREGATE` and the match pins do not move.
- The ledger cost is procedural only. `water_solver.cpp` (and `.h`, if a
  member is added) are `files` of `[twins.water]` and of other twins, so the
  patch needs one `tools/cuda_green.py` run and `--mark-level`. The device
  result is unchanged by construction.
- Alternative: move `step_ripple` into its own TU outside the twin files
  (Claude's call at build time).

**The retune.** It scales the wave with P (about 3× smaller); `k_h` absorbs
that in one look session.

**Collisions.** None. The ripple is not a continuous synced field, so
FieldEdit does not apply, and no blast rule is touched.

### (a2) Draw the real water's own surface (a render follow-up, optional)

**The change.** The shader's normal also takes the slope of the simulated
surface (depth + floor), uploaded through the dequantize convention.

**What Erik would see.** Every real wave: the blast's 3 cm, floods, and later
the bodies of #236.

**Cost.** Render-only, no sim cost.

**Risk.** Two wave layers drawn on top of each other: the ripple and the pipe
model are the same physics at the same resolution. A later design may retire
the ripple in favour of (a2) plus the procedural sines. Not proposed now.

### (b) Displacement: the blast pushes water outward, scaled by strength

**The change.** At detonation, a new `[blast]` column (for example
`water_m3_per_strength`) moves that volume of water out of the blast's walked
reach. It goes either as volume, taken from the reach tiles and landed on its
rim through the Fixed-total tile deposit (`field_edit.Region.TILES`), or as
momentum, a radial Δv on `flow_vx` / `flow_vy`.

**What Erik would see.** A crater that refills, and a wave as large as the
dial asks, independent of the air.

**Cost.**
- The water fields are synced Q16, so the digest moves wherever a charge sits
  in water. GOLDEN_AGGREGATE's scenario seeds water but detonates no strength
  row (INFERENCE: it is unchanged).
- The rule "`execute_payload` has a device twin for EVERY row ... lands there
  in the SAME PATCH" brings a water edit into `cuda_unit_payload.cuh`.
- `water_depth` is in `DEVICE_REFUSED_FIELDS`, so the resident tick falls back
  to a host flush on such ticks, and the device conductor (class T) needs a
  water edit path.
- Momentum needs new FieldEdit policies for `flow_vx` / `flow_vy`.
- It is a Class-P change with one cuda_green run.

**The retune.** No effect: the column is its own number, so it is robust to
the retune, but it is a second calibration.

**Collisions.**
- "A blast pushes units only through its own wind (... never a second blast
  impulse, so nothing is counted twice)". The water already gets the blast
  through the pressure head, so displacement is the water's version of a
  second push. **This needs a ruling.**
- "A new explosive is a row" holds if the column is per strength in `[blast]`.
- "FieldEdit is the only way to write a continuous field" holds if both forms
  go through FieldEdit.

### (c) Steam: the blast turns water into steam, scaled by strength

**What exists.**
- The W5 flash-boil only (`PhysicsEngine` water host tail). It is
  pressure-keyed (`atmosphere < boil_p_thresh`) and moves depth into the
  `gas[steam]` trace plane at `steam_yield`, with no energy.
- Water has no temperature. The fire side's "heat-boil" was never built.
- `[gases.steam]` is a near-white, vision-blocking trace gas.

**What Erik would see.** A white puff over the charge. No waves: the energy
of a retuned frag (about 0.9 MJ) vaporises at most about 0.35 kg of water,
3 mm over one tile. Strength 3 is about 1 kg.

**Cost.** The volume moved is negligible. The books are not: the latent heat
must leave the blast's heat (a new seam term in the energy closure), or energy
is counted twice. That needs water with heat and phase change with mass
conserved, which is **milestone "Water, fire, ice v2" (#219)**.

**Collisions.** The energy closure identity. Steam via the trace door
(`emit_gas`) is fine.

### (d) The blast's own wind as a force on the water surface

**The change.** Wind stress τ = ρ_air·C_d·|u|·u on the flow.

**What Erik would see.** Nothing, at physical values.
- With C_d ≈ 2e-3 and the measured winds over the pool (59.5, 27, 11.5 m/s
  over three ticks), Δv ≈ 0.0015 m/s, against 0.145 from pressure.
- The wind is held at 64 m/s by #217, so it does not scale with strength
  (measured: equal at full and /15).
- A feel gain large enough to show would also drag water under every vent and
  fire plume.

**Cost.** A new read in `WaterSolver::step` on both twins. `GOLDEN_AGGREGATE`
moves (its scenario has water and wind), so it needs a re-baseline.
**Rejected.**

## 5. Recommendation

**Build (a) now; decide (b) after the blast retune; send (c) to #219; reject
(d).**

Reasons:

- (a) fixes the measured cause, the source shape, at its root:
  - The drift is gone by construction (the twin holds exactly 0 for 20 s).
  - The blast's wave grows with the blast's own pressure and stays inside the
    cap, so it keeps scaling.
  - It costs no synced state, no device code and no golden.
  - It is exactly Erik's "we only need to set off the splash".
- (b) answers a different wish: water moving more than the air explains. The
  retune will make the air weaker. Whether (a)'s wave is then enough is a look
  question for the same session, with numbers in hand. If it is not, (b) as
  volume moved through the fixed-total tile deposit is the cleaner form.
- (c) needs water with heat to keep the energy books closed. It is #219's.
- The real water's coupling `k_p` is Erik's dial. Its natural place is the
  blast-and-vent retune session (it scales linearly, §2.4), not this patch.

## 6. Verification plan (properties, never snapshots)

Each test names its property and the change that must break it.

1. **No drift.** On `water_studio` with the wave charge removed, over 30 s the
   pool's mean ripple stays within 1 mm of 0 and no wet tile sits at the cap.
   Breaks if the source is rectified or not zero-mean in space (today's law
   fails at 16.7 s).
2. **A uniform pressure change makes no wave.** Raise P uniformly over a
   sealed flat pool (a seeded uniform heat). The ripple slope stays at rest.
   Breaks if the source reads P − P_prev, or P itself, instead of its spatial
   variation.
3. **A blast makes a wave that travels.** After the pool charge:
   - the peak slope exceeds a few times the rest slope;
   - the radius where |Δη| > 1 mm grows with time at about
     √(g·min(h, h_cap));
   - the slope is back under twice rest within a bounded time.
   Breaks if the source saturates at the cap, is missing, or does not
   propagate.
4. **Monotone in strength.** Strength 1 gives a smaller peak slope than
   strength 3 at the same spot. Breaks if the cap or saturation binds.
5. **Render-only.** The synced fields and the field digest are bit-identical
   with the ripple stepped or not (A/B lockstep harness); `GOLDEN_AGGREGATE`
   does not move. Breaks if the ripple feeds back.
6. **Dormancy.** A dry map never touches the ripple arrays.
7. **E2E.** `tools/e2e_drive.py` on `water_studio` before and after, plus the
   look shots `still_pool_flashlight` and `still_pool_closeup` (frame 240 =
   2 s after the blast) rendered before and after and looked at.

## 7. Rulings for Erik (each with the default the build uses)

1. **Which way to the waves.** Default: (a), the pressure splash with a
   zero-mean source, render-only, with `k_h` defaulting to `k_p` and tuned by
   eye in the blast-and-vent session.
2. **May a blast move water directly** (option b, an exception to "a blast
   pushes only through its own wind / pressure")? Default: not now. Revisit
   after the retune; if yes, as volume moved per strength in `[blast]`.
3. **Steam from blasts.** Default: part of #219 (water with heat, phase change
   with mass and energy conserved), not this patch.

## 8. Systems

**(a) Existing canonical systems this design must use.**

- Water flow detail / the water shader, and the Look shots
  (`tools/look_shots.toml`): the visual check.
- Dequantize convention: any new upload, for (a2).
- The A/B lockstep harness and `GOLDEN_AGGREGATE`: the render-only proof.
- Class P / Class T + the twin ledger (`tools/twin_debt.py`,
  `tools/cuda_green.py`): `water_solver.*` are twin files.
- The E2E driver (`tools/e2e_drive.py`).
- For (b) only: the Payload executor and its device twin, the Blast reach, the
  Fixed-total tile deposit, FieldEdit and the device edit door.
- For (c) only: the energy closure identity and the gas energy seam.

**(b) New reusable systems with a draft one-line rule.**

- **Ripple surface-pressure forcing** (`WaterSolver::step_ripple`): "The
  ripple is forced only through the pressure head in its restoring term
  (zero-mean in space); never a per-tick kick from \|ΔP\| or any rectified
  source." This amends the ripple's entry; no new index line.
- None other for (a). For (b), if ruled in later: "A blast moves water only
  through the `[blast]` per-strength column via the fixed-total tile deposit
  over its walked reach; never a per-row water column."

## 9. Critique

| lens | model | verdict | findings |
|---|---|---|---|
| 1. Physics, numerics, measurement soundness (2026-11-21, against f263b718) | Opus | SOUND WITH FIXES | Diagnosis and the render-only claim confirmed (ripple not digested: `tests/field_digest.py:159`, `field_digest_spec.toml:86`; host-only on the resident path, `physics_runner.py:1613-1617`, `applier.py:571`; no obs or sim law reads it); CFL stays `ripple_max_dt` 0.106 s. (1) MAJOR PROVED: the wave's size and SIGN come from the EOS, not the blast: on the detonation tick P minus pool mean within 1 m of the charge is -0.73 atm on #217 (a mound) and +0.10 atm on main's EOS (a crater, 7x weaker); after /15 main gives 1.3 mm / slope 0.002, below the rest slope; nearly all forcing lands in one end-of-tick sample, so it scales with 1/tps (INFERENCE). Re-measure on merged main and after the retune before choosing k_h; weigh a render-only ripple_v impulse read from `GameMap.blast_log` (strength, position), independent of the EOS and the tick rate. (2) MAJOR: (a) does not bring back the remembered look: W6a's ring rode a 22 m/s acoustic front; the new one moves at ~1.5 m/s and decays 1/e per s, its first wall reflection back after ~3.4 s at ~3 %; describe it honestly, which look is Erik's call. (3) MAJOR PROVED: most of the 'travelling' extent is forced in place (|d eta| > 1 mm radius 1.05 m at t=193, 3.59 m at t=194, then ~1 m/s); verification test 3 would fail on this scene; measure speed on a local pulse in a unit scene. (4) MAJOR PROVED: boundaries unspecified and the naive build wrong: P on solid tiles is 0, the twin's Dirichlet shore deforms any room whose P differs from the global mean; specify lap(eta + k_h P) with P read on every non-solid tile, eta = 0 dry, the sum mirrored at solid; then no global mean is needed (reword 'zero-mean in space'). (5) MAJOR PROVED: the still-water look changes too: roughness (`water.fs:517-519`, `roughness_agitation`), foam and W1 detail read ripple_v via `relative_agitation`, built around the uniform heave; list the consumers, expect a re-look of #97 dials, add `pool_blast_spray` and `wade_by_firelight` shots. (6) MINOR: 'exactly 0' with no blast is ~2e-5 m. (7) MINOR: subtract the no-charge baseline in 2.4 (0.016 m3 / 2.4 mm settling). (8) MINOR: `tests/test_water_ripple.py::test_amplitude_clamp_holds_and_engages` uses a uniform splash and will go vacuous; reuse `tests/test_headless_ripple.py`. (9) MINOR: rejections sound; state today's engine_death mass (~11.5 kg, 30 MJ) beside the retuned. (10) MINOR: list `physics_engine.{h,cpp}`, `bindings.cpp` and callers if `p_prev` leaves the signature; a breach to vacuum leaves a standing eta at the cap. |
| 2. Systems reuse, project rules, gates, scope (2026-11-21, against f461e8b4; independent of lens 1) | Opus | SOUND WITH FIXES | No blockers; (a) outside the determinism gates confirmed (`applier.py:571`, `physics_engine.h:172-177`, `physics_runner.py:1608-1611`); (c) to #219 and (d) rejected right; #218 untouched. (1) MAJOR PROVED: the draft rule has no home: no CLAUDE.md row for the water solver / ripple, no rule file's `paths:` covers `water_solver.*` / `cuda_water.cu`; add ONE index line ('Water solver + ripple') and the constraints in an existing region with the paths added; the rule says render-only, never fed back, forced only by the pressure head, never a rectified per-tick kick (incident #222). (2) MAJOR PROVED: two live rules go stale: `render-effects.md:96` / `:107` (and `water_foam.py:52-56`, `water_surface.py:621`) justify relative agitation by the ripple's room-wide heave, which (a) removes; list Water foam and `relative_agitation` as affected, amend both, and give Erik the look ruling (the #97 still-pool look changes; re-look its dials). (3) MAJOR PROVED: the float ratchet is a hard gate: `tests/test_no_float_in_sim_tu.py:62,:391` pins `water_solver.cpp` at 32 float lines; hold the count or re-baseline with a written exemption; drop or honestly cost the own-TU alternative (/fp:strict list, ratchet row). (4) MAJOR PROVED: pre-existing tests encode the old law: `test_water_ripple.py:89-97` (`_splash_source`, uniform P, used at :186, :270; `_splash_rollout` None = no splash), `test_headless_ripple.py:49`, `test_a5_seal_evacuation.py:128`, `tests/cuda_s8a_check.py:40` (host ripple CPU vs resident byte for byte, the real resident-path gate); rewrite each as a property of the new law in the same patch. (5) MINOR: Class P duty understated: `[twins.water]`, `[twins.water_tail]`, `water_solver.h` in ~a dozen twins, `bindings.cpp` a build path, `physics_runner.py:888` a water-twin file; mark every twin `twin_debt.py` reports, regenerate `stubs/breach_physics.pyi`, the `config.toml:702` rename. (6) MINOR: property 5 rebuilds an existing test the wrong way (`field_ab_harness.py:87` has the ripple in SIM_FIELDS); reuse `test_water_ripple.py:348` and `test_headless_ripple.py:67` on a local bump. (7) MINOR: render the whole shot set (`render-core.md:35`); add a blast-wave shot row rather than repurpose the still-pool shots (frame 240 would hold a pool-wide wave). (8) MINOR: ruling 2 misquotes the rule: `explosions-fire.md:32` is about UNITS; restate as a physics-model question (may a blast move water beyond what its air does?). (9) MINOR: the #217 order: #217 touches `bindings.cpp`, `physics_runner.py`, `config.toml` too; cut (a) from main after #217 merges, the second to merge needs a fresh `.cuda_green`; k_h tuned only on post-#217 main. (10) MINOR: (a2) collides with `render-effects.md:92` 'never a second wave source'; name it as a future rule amendment for Erik. |
