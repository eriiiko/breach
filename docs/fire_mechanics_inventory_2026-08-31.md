# Fire mechanics inventory — 2026-08-31 (fire session #12, Phase 0)

> **Status: DRAFT — §2/§3 pending the code survey.** The step back Erik asked
> for before any retuning: what do we want fire to do, what does the engine
> already simulate, and where are the gaps. This doc is capture (append-only,
> dated); rulings it produces go to the phase design notes and, at
> implementation, to CLAUDE.md rows.
>
> Context: the July fire-tuning campaign (branch `fire-tuning`, plus the
> fire-o2-integration stack) was fought on an engine with two since-fixed
> defects — the atmosphere storming (undamped Helmholtz mode, fixed by the
> interior-drag arc) and the spurious room heating (#54 gas-energy
> conservation, closed 2026-08-30, ledger now exact in int64). Every July
> conclusion is therefore a *hypothesis* today, not a result (§4).

## 1. Design goals (Erik, consolidated 2026-08-31)

Sources: this session's sync, issue #12 + comments, July tuning memories.

### 1.1 The reference fire ("bonfire on a wood crate")

- **G1 — Realistic plateau temperature.** Flame ~1300 K (order-of-magnitude
  anchor, not a precision target) in honest Kelvin under the ONE map (§1.4).
  July's blessed shape peaked at 14764 K — "good, except too hot."
- **G2 — Slow ramp.** Ignition at low intensity → full intensity in
  ~30–120 s. (Q16.16 growth-quantum caveat: a genuinely slow ramp sits near
  1 LSB/tick — may need a residual accumulator, see fire-bootstrap-relations.)
- **G3 — Finite life, fuel-governed.** Self-limiting burnout in ~5–10 min
  for a crate-scale fire (July's 14 min was accepted once; 5–10 is the
  target). Erik (first read, 2026-08-31): fuel exhaustion should be what
  puts a fire out; intensity ∝ remaining-health is his instinct — which the
  F = wall_hp/hp_mat availability factor already implements in shape. The
  open problem is BALANCE: July measured self-extinction with ~76% fuel
  unburned (heat/O2-governed, not fuel-governed). Whether the die-term
  design itself needs rework → 3c (see G11 note); dials → 3a/4.
- **G4 — Low nominal intensity with headroom.** Nominal I ≈ 0.2 preferred
  (Erik 2026-08-31; anchors, not exact targets) so oxygen enrichment has
  room to flare. I and T* are linearly coupled through gain (T* = gain·I),
  so a low nominal I does NOT preclude G1 — it prices gain.

### 1.2 Fire as a citizen of the atmosphere

- **G5 — O2 makes fire live and die.** Sealed room → self-starves; breach →
  vented O2 kills it faster; inert flood smothers; O2 tank release → every
  fire in the room flares (the payoff mechanic; keep as a hard requirement).
  **AMENDMENT (Erik 2026-08-31): mole fraction alone is not enough** — the
  o2f law must also account for ABSOLUTE density (vacuum / very low
  pressure): today a near-vacuum cell with 3 O2 of 4 total particles reads
  X = 0.75 and burns. This is the known "vacuum fires" item handed forward
  by the T_abs arc; fix direction = augment o2f with a total-N (or O2-N)
  factor, suitably scaled. Erik also flags this MAY have fed past
  instability (unprovable now — the engine had unrelated errors then).
  → the O2-law redesign feeds 3c/4.
- **G6 — Wind interacts honestly.** Wind fans a robust fire (O2 supply,
  already emergent) but can strip a *marginal* one below its sustain floor
  (the k_wind_strip term, currently dormant at 0.0 — parked in July
  session-1 as a tuning confounder, promoted verbatim by P-K0, never
  consciously retired). Ruling wanted: revive as a mechanism.
- **G7 — Blow-out is not cool-down.** A stripped flame on still-hot fuel
  auto-reignites when O2 returns (T ≥ ignition already holds). Emergent from
  G6 + the ignition law; wanted explicitly.
- **G8 — Blast waves discriminate.** A shockwave's transient compression
  heating must NOT flash-ignite a wall in one tick (Erik 2026-08-25), but
  sustained heat genuinely ignites — and the blast's *wind* may kill a
  marginal flame or feed a strong one (G6's algebra applied at high speed).
  → the sustained-exposure (dwell-integral) ignition law serves all of G8.

### 1.3 Fire touching solids

- **G9 — Radiation with a purpose.** Fires radiate to line-of-sight targets
  and can ignite air-separated fuel. CURRENT FACT (§2B): emitters are fires
  ∪ hot solids (≥180 game); air/gas is structurally transparent
  (heat_atten[air]=0) — it neither emits nor absorbs, by construction not
  by dial. OPEN DECISION (Phase 3d): should hot gas and smoke radiate?
  Blackbody radiation from smoke (and hot air) has been considered before;
  Erik 2026-08-31: unsure it's worth it — investigate properly and make a
  FINAL decision. If gas does radiate, long range is fine (Erik). Note
  RADIATION_RANGE=320 is a stability constant; the feel question is the
  falloff/ray-count law and T_emit_gate, not the range cap.
- **G10 — Materials react in Kelvin.** Ignition temps and material responses
  are chosen by physical reasoning in Kelvin — which is why the map (§1.4) is
  load-bearing, not cosmetic.
- **G11 — Ember question, WIDENED to the whole die mechanic.** Does the
  ember state survive as a mechanic? A normal fire's own burnout can't
  reach it today (fuel bed settles ~15.5 game << ignition 300) — Erik's
  correction 2026-08-31: not *structurally* unreachable, just unreachable
  from those initial conditions; an externally-kept-hot tile still
  qualifies, which is fine. **And re-justify smolder from first principles
  before keeping it** (Erik doesn't fully buy the why): the RECORDED
  rationale (test file's design story, test_eos_p5_1_stoich.py:22-47) is
  (a) suffocation ≠ extinguishment — O2-starving a fire is reversible
  suppression, O2 return REIGNITES (this is G7 seen from the O2 side);
  (b) the sealed-smolder regime — a starved ember keeps drawing trickle O2
  and draining fuel over thousands of ticks (`fuel_per_o2` = THE
  ember-lifetime dial); (c) char-out completion — fuel is genuinely
  consumed to the floor without visible flame. NOT about temperature
  persistence (that's cool_shift's job). Erik 2026-08-31: "perhaps
  we need to look again at what mechanic drops fire intensity — I am not
  sure the mechanic is well thought out." So Phase 3c reviews the FULL
  death-side design (`die = k_die·(1−avail·hot)·I + strip term`, the
  snap-out floor I_min, the claim-gate exclusion) — not just bolt on a
  sustain floor. Related ruling to revisit there: combustion's
  never-destroys invariant (1-LSB char floor) — Erik is open to combustion
  destroying tiles too if the invariant isn't earning its simplicity; also
  the fuel tile's own coldness (H_bed ~15.5 game while alight) is part of
  why "fuel in a very hot fire can't burn to the ground" post-flame.

### 1.4 One temperature map (Phase 1 — first patch of this arc)

- **G12 — ONE affine map, everywhere.** `Kelvin = T_game + 293` (slope 1,
  0 game = ambient = 293 K = 20 °C). Today THREE frames coexist (§2B): the
  canonical ×3 map (`293 + 3·T_game` — blackbody, radiation T⁴ bake,
  tools), the EOS absolute frame (`290 + T_game` — the actual
  thermodynamics: compression work, gas_energy ledger), and hover's
  sub-ambient K_eos patch papering over the ×3 map's unphysical region.
  G12 = collapse the canonical map ONTO the EOS frame: `k_temp_to_kelvin`
  3→1 (φ_exp ⅓→1, frozen identity φ·k≡1 preserved), `kelvin_ambient` stays
  293, `eos_t_amb_k` 290→293, `T_MIN` −289→−292 (T_abs ≥ 1 K). The hover
  special case then dissolves — slope 1 is valid down to 1 K, which
  retro-explains the "−574 K" display bug as a symptom of the ×3 map.
  Rationale of record: (a) the frame the conserved physics already lives in
  wins — energy books are denominated in it; (b) slope 1 reads as "game
  degrees ARE Kelvin degrees" and Q16.16 still spans ~32,700 K above
  absolute zero — range is no argument for a steeper slope; (c) ambient-zero
  keeps the rest state at integer 0 (zero-filled fields = world at rest),
  keeps relax-to-ambient the cheapest fixed-point op, and matches every
  golden's semantics. MIGRATION COST (Phase 1 scope): every Kelvin-anchored
  dial re-derives — `rad_scale` re-anchors (precedent: P-K2 did exactly
  this for ×2→×3), `T_emit_gate` (180 game = 653 K under ×3 → 360 game
  under ×1 for the same Kelvin), and ignition temps get *reviewed* in
  honest Kelvin (wood 300 game reads 1193 K under ×3 but 593 K under ×1 —
  the latter is realistic for wood, suggesting the dials are already
  closer to slope-1 sense). One deliberate golden re-baseline + HUMAN-TEST
  (radiation strengths shift = feel-adjacent). The flame plateau is a
  TUNING TARGET expressed in this map's units — never a map anchor (the
  two-map phone proposal of 2026-08-30 is superseded by Erik's ruling
  today: Kelvin informs ignition limits and material design, so any map
  feeding intuition is load-bearing).

### 1.5 Explicitly out of scope for this arc

- **dragon_7 / flamethrower** — own design session AFTER the fire pass:
  to be rebuilt on the wind + fumes systems (actually pushes atmosphere
  wind, preferably deposits fumes), not as a bare heat-cone. Its cone test
  is skipped with a pointer (commit on fire-12, 2026-08-31).
- **Fire VFX / rendering beauty** — parked as #52; only the blackbody's
  *map* changes here (G12), not its look.

## 2. What the engine simulates today

### 2A. Combustion / fire-intensity side (code survey 2026-08-31)

Primary: `cpp/src/combustion.cpp|.h`, `cpp/src/fire_simulation.cpp|.h`,
`src/simulation/combat.py::apply_temperature_ignition`, config wiring in
`physics_runner.py`.

**Ignition — purely instantaneous, no dwell anywhere.** Two paths:

1. *Temperature ignition* (`combat.py:463-640`, once per tick after
   physics): edge-triggered via per-tile `ignition_armed` (SYNCED) — re-arms
   below threshold, disarms while hot+burning; seeds
   `fire = ignition_seed (0.12)` on `armed ∧ T ≥ ignition_temp[mat] ∧
   X_4nb > o2_frac_ext ∧ wall_hp > 0 ∧ fire == 0`. Plain Q16 `>=` compare —
   no smoothing, no dwell. (The edge trigger replaced the "eternal 0.1
   smolder" level-trigger, ruling 2026-07-24.) NOTE: fuel gate is
   `wall_hp > 0`, so a charred 1-LSB tile can re-ignite via this path.
2. *Combustion claim gate, not-alight branch* (`combustion.cpp:507-511`):
   a non-burning flammable tile may draw O2 only if `Tsnap ≥ ignition_temp`
   (pass-entry snapshot — a source can't heat AND ignite a neighbour the
   same tick). An alight tile skips this check entirely (P-R4 hysteresis:
   sustain is governed by `fire_T_ext`, not `ignition_temp`).

`fire_T_ext[mat] = ignition_temp[mat] − ignition_to_ext_delta (200)`, baked
to a per-tile plane (`fire_simulation.h:99-111`); the `[physics.fire]`
scalars `fire_T_ext=350` and `fuel_ref=60` are documented-inert fallbacks.
Flammable materials in the live engine: wood (ign 300, hp 60), furniture
(280, 30), kindling (280, 8). `door` carries ignition_temp 280 but is
flammable=false by legacy hardcode (config.toml:1301-1307).

**Intensity ODE** (`fire_simulation.cpp:161-286`, Q16 pinned-order):
`hot = clamp01((T − T_ext)/fire_T_span)`; `o2f = clamp01((X − 0.13)/(1.0 −
0.13))` (linear in mole fraction, Peatross & Beyler 1997 — MOLE FRACTION
ONLY, no absolute-density factor: the G5 vacuum-fires amendment applies
here and at the ignition/claim-gate reads of the same law); `avail = F·o2f`,
`F = clamp01(wall_hp/hp_mat)`; capacity law (P-R3)
`gap = avail·hot − I/I_cap_per_avail`;
`grow = k_grow·I·gap·(1 + k_wind_fan·W)`;
`die = k_die·(1 − avail·hot)·I + k_wind_strip·W·(1−I)·I`;
`I += dt(grow − die)`, clamp01, snap to 0 below `I_min`.
Dials (shipped): k_grow 0.5, k_die 0.008, I_cap_per_avail 14.0,
fire_T_span 180, o2_frac_ext 0.13, o2_frac_full 1.0, I_min 0.02,
k_wind_fan 0.5, **k_wind_strip 0.0**, ignition_seed 0.12, wall_damage 0.03,
ignition_to_ext_delta 200.

**Quantization state — split finding:** combustion's O2-demand truncation
(a 0.12-seed fire drew zero O2 and died at 21 s) is FIXED by the D1
error-feedback accumulator (`dem_acc`, `combustion.cpp:528-556`, per
(air-cell, claimant-slot) sub-count remainder). The fire-I ODE itself has
NO residual accumulator — G2's growth-quantum caveat (Trap 3,
fire-bootstrap-relations) still stands for slow ramps.

**Fuel drain — two channels on `wall_hp`:** (a) ember/sustain-scale:
`fuel_cost = fuel_per_o2 (0.7) · O2_drawn`, floored at `FUEL_FLOOR = 1` raw
LSB, NEVER destroys (`combustion.cpp:719-753`); `fuel_per_o2` is THE
ember-lifetime dial. (b) flame-scale: `wall_damage (0.03)·dt·I` — the only
destruction path (`fire_simulation.cpp:305-322`; hp ≤ 0 → destroy_wall).
"Burned out" is therefore either charred-at-1-LSB (stands, inert to
combustion) or destroyed (flame-scale only).

**Ember is emergent, not a state machine**: ember ≡ `fire==0 ∧ T ≥
ignition_temp ∧ wall_hp > FUEL_FLOOR`. Why a normal burnout can't reach it
(Erik's correction: unreachable from these initial conditions, not
structurally) — confirmed at `combustion.cpp:511`: with the painter retired, the H_bed
fuel-bed deposit (`H_BED_M 18125, H_BED_SHIFT 4` → H_bed ≈ 2.9e5) holds the
burning tile's own T at ~15.5 game — so at flame snap-out the not-alight
gate (`Tsnap < 300`) instantly excludes it. G11's sustain-floor hysteresis
is the confirmed missing piece.

**O2 & products:** demand `∝ burn_rate (0.02)·fire[i]·o2f_j·W_hop·w_path`
over ≤ draw_r=2 hops; contested cells split exactly (proportional int split,
≤3-LSB documented tiebreak bias). Products: exact Dalton split of drawn O2
into soot (`soot_yield 0.5` → `smoke` gas) + `inert_n2`. **There is no
"fumes" gas** — gas table: steam, smoke, poison, teargas, fuel_gas, o2,
inert_n2. `fuel_gas` is a separate flammable trace-gas species (relevant to
the future dragon_7 session, NOT a combustion product). Erik's design
intent for it (2026-08-31, on record): a flammable gas that is NOT always
burning — it can fill a room inert and ignite later, or burn as it flows.
Since #54, every combustion transaction also moves the int64 `gas_energy`
ledger.

**Wind coupling (combustion side):** `combustion.cpp` never reads wind.
Coupling is exactly three things: (a) `k_wind_fan 0.5` — LIVE growth term;
(b) `k_wind_strip 0.0` — live code, config-zeroed (struct default is 0.5;
"plume self-blow-out off, 2026-07-23"); (c) emergent O2 advection.
**⚠ G6 revival tripwire:** the FORBIDDEN BAND (storm audit 2026-08-14 §5):
any material `wave_absorb ∈ (0, 0.02)` while `k_wind_strip > 0` is the
historical rectifier window (KE burst, T-floor spiral);
`physics_runner.py:509-521` hard-errors at load. Reviving k_wind_strip must
re-check current wave_absorb values against this band.

### 2B. Thermal / radiation / wind-field side (code survey 2026-08-31)

Primary: `cpp/src/temperature_solver.cpp|.h`, `cpp/src/raycaster.h|.cpp`,
`cpp/src/eos_solver.cpp`, `src/temperature_scale.py`,
`renderer/blackbody.py`, `renderer/hover_readout.py`.

**★ `k_fire_heat` IS DEAD** — tombstoned at P-R4 (2026-08-01,
config.toml:383-390). The July tuning campaign's central dial no longer
exists. A burning tile heats the world via exactly two live channels:
(a) *radiation* — antisymmetric net-T⁴ pair exchange (P-F1a), landing in
`rad_net[]` → TemperatureSolver Pass 1, thermal solids only; (b) *H_bed* —
the fuel-bed deposit on the burning tile itself, ∝ O2 actually claimed
(`combustion.cpp:686-690`, H_bed = 18125·2⁴ = 2.9e5), plus `H_fuel = 4.0`
heating the donor *air* cell. The equilibrium algebra survives with the
substitution `gain = (H_bed·B)·2^(cool_shift − heat_inv_shift)`
(config.toml:828 derives it).

**Cooling**: Pass 3 per-material `cool_shift` (`T >>= shift`), solids only.
Wood/furniture/kindling promoted 5→13 at P-K0 ("THE SPREAD DIAL": e-fold
1.3 s → 341 s); global `COOL_SHIFT = 5` remains fallback + vacuum-offset
anchor (`cool_shift_vacuum = 3` is an OFFSET, floor SHIFT_MIN=2). The
two-way ambient thermostat (relax toward game 0 from both sides) is
deliberate and counted (`e_thermostat_sum`, Erik's ruling 2026-08-30).

**Radiation chain** (P-F1a, `raycaster.h:79-417`): emitters = burning tiles
∪ any thermal solid with `T ≥ T_emit_gate (180 game)`; **air/gas never
emits or absorbs** (`heat_atten[air] = 0.0` — Kirchhoff, structurally
transparent). 8 rays per emitter through the shared integer march (CPU +
CUDA batch twin); per-pair signed exchange `x = a_s·a_r·τ·w·(E°[T_s] −
E°[T_r])`, `E°(T) = rad_scale·K(T)⁴` with K from the canonical map;
contact faces radiation-inert (conduction owns contact); rays leaving the
grid charge the emitter against a T=0 sky (`rad_amb` ledger,
`Σ rad_net + Σ rad_amb == 0` exact). Flux-limited (RAD_LIM_SHIFT=4).
`radiation_range = 320` is a STABILITY constant (must exceed the grid
diagonal, floor 287 — "ray left the world" is the only escape condition),
not a feel dial. Deposit: thermal solids only, `rad_net >> heat_inv_shift`,
clamped [0, T_MAX_PHYS=16000]. `rad_flux[]` at air cells is a separate
positive-only unit-damage sensor, explicitly not ledger.

**★ THREE temperature frames coexist today** (the inconsistency G12 kills):

| frame | formula | who uses it |
|---|---|---|
| Canonical map (`temperature_scale.py`, `[physics.temperature_scale]`) | `K = 293 + 3·T_game` (k_temp_to_kelvin=3, phi_exp=⅓ frozen so φ·k≡1) | blackbody render, radiation E° bake, hover above ambient, all tuning tools |
| EOS absolute frame | `K_eos = 290 + T_game` (slope 1, `eos_t_amb_k=290`) | THE thermodynamics: compression work, gas_energy = N·T_abs, #54 ledger |
| Hover sub-ambient patch (`hover_readout.py:82-97`) | canonical above 0, K_eos below (labeled "K_eos") | added because the ×3 map goes unphysical below T_game ≈ −97.7 (Erik read "−574 K" off a 1.1 K cell) |

The old ×2 map (`293 + 2·T_game`) is fully retired (P-K2 re-anchored
`rad_scale` to the ×3 map; stale blackbody keys are a load-time hard
error). NOTE the physical incompatibility: gas thermodynamics says
T_game=100 is 390 K; the radiation T⁴ law says it is 593 K.

**Wind/plume**: no explicit plume model — the shim is deleted (P-R2). Fire
wind is implicit ideal-gas: heat → `p* = C·N·T_abs` → pressure solve →
kick. Heat advects conservatively via the EOS flux-form energy transport
(the semi-Lagrangian T-copier was deleted at P-E1 as an energy-minting
channel). **No direct wind-cools-fuel mechanism exists**: only the dormant
`k_wind_strip` intensity term, plus the indirect path (wind advects hot gas
away → smaller gas↔solid conduction gap next tick).

**Gas↔solid heat exchange**: ONE unified pass — TemperatureSolver Pass 2
energy-form conduction (P-E2a), every cell carrying a capacity (solids
`2^heat_inv_shift`, gas `N·c_v` floored at `n_floor_heat=0.01`); air has
tiny nonzero conductivity precisely so solid↔air exchange rides the same
face law. Furniture/kindling have conductivity 0 — no faces; their only
loss is cool_shift, only gains H_bed + radiation. (`exchange.py` is the
physics→UNIT coupling table, not this.)

**Shockwave heating (G8 mechanics)**: compression work heats gas via the
#54 flux-form energy step → recovery writes gas `temperature` once per
tick → Pass 2 conducts into adjacent walls THE SAME TICK — but doubly
damped (per-face limiter LIM_SHIFT=1, and endpoint capacity: a steel wall
warms 32× less than the gas cooled). One-tick flash-ignition of a wall is
already structurally unlikely; whether a sustained event accumulates to
ignition is a magnitude question for the Phase 3a bench, and the dwell
dimension itself remains absent (§2A).

## 3. Gap table — goal × current mechanism × gap

| Goal | What exists today (§2) | Gap → phase |
|---|---|---|
| G1 plateau ~1300 K | Heat via H_bed (2.9e5) + T⁴ radiation; equilibrium `T* = H_bed·B·2^(cool_shift−heat_inv_shift)`; never measured on this mechanism + healed engine + honest map | Measure first → 3a |
| G2 ramp 30–120 s | `k_grow 0.5` tempo dial (P-R3 capacity law); I-ODE has NO residual accumulator | Slow ramps sit at ~1 LSB/tick — accumulator likely needed → 3a measures, 4 decides |
| G3 burnout 5–10 min | Two drain channels; `wall_damage 0.03` tuned for "8 min kindling / 24 min furniture"; `k_die 0.008` e-fold ~3000 ticks (restated e2e tests document no full extinction in 400-tick windows) | Pure dial question once 3a establishes the energy balance → 4 |
| G4 nominal I + O2 headroom | Capacity law `I_cap_per_avail 14`; `o2f` linear up to pure O2 (o2_frac_full=1.0) — flare headroom structurally present | None structural; pick nominal in 4 |
| G5 O2 makes fire live/die | Continuous-O2 law shipped (draw_r 2, D1 accumulator, exact contested splits); sealed/vent/flood orderings verified by restated tests; o2f is MOLE FRACTION ONLY | Vacuum-fires amendment: add absolute-density factor to o2f (3c/4); decay timescales → 4 |
| G6 wind strips marginal fires | `k_wind_strip 0.0` (live code, config-zeroed); `k_wind_fan 0.5` LIVE; no other direct wind→fuel term | Revive + retune; ⚠ forbidden band `wave_absorb ∈ (0,0.02)` while strip > 0 → 3c |
| G7 blown-out hot fuel auto-reignites | Edge-triggered arm REQUIRES cooling below threshold before re-arming — a still-hot stripped tile CANNOT reignite today | Real gap, same hysteresis design as G11 → 3c |
| G8 blast discrimination | Compression work heats gas (#54 flux-form); same-tick conduction into walls exists but doubly damped (LIM_SHIFT + 32× capacity); no dwell law anywhere | Exposure-integral ignition; measure transient magnitudes → 3c (3a bench) |
| G9 radiation with purpose | Full P-F1a chain, books close exactly; emitters = fires ∪ solids ≥ 180 game; air structurally transparent (heat_atten 0); range 320 = stability constant, not feel | Look-over: should air/gas emit? falloff/ray-count law? T_emit_gate value? → 3d |
| G10 materials react in Kelvin | Blocked by the three-frame map; e.g. wood ignition 300 game is 1193 K or 593 K depending on frame | Unblocked by G12 → 1, reviewed in 4 |
| G11 ember + die mechanic | Emergent condition, structurally unreachable (H_bed holds fuel at ~15.5 game vs ignition 300; claim gate `combustion.cpp:511`) | FULL death-side review (die term, I_min snap, claim gate, never-destroys invariant, cold fuel bed), not just a sustain floor → 3c |
| G12 one map | Three frames (§2B) | THE Phase 1 patch |

Minor inconsistencies (fix opportunistically): charred 1-LSB tile can
re-ignite via the temperature path (`wall_hp > 0` gate) though combustion
refuses to feed it — a vacuous flicker; `door` carries ignition_temp 280
but is hardcoded non-flammable; inert fallback scalars (`fire_T_ext 350`,
`fuel_ref 60`) and retired keys (`P_min/P_full`, `o2_threshold`,
`p_expand_ref`) still in config.

## 4. July results: what must be re-measured before belief

Measured on the pre-fix engine (storming + #54 false heating both active),
so: usable as starting dial positions, NOT as conclusions.

**★ Stronger than "bugs fixed": the heat mechanism itself was REBUILT after
July.** The entire July plateau story was fought over `k_fire_heat` — a
dial P-R4 deleted (2026-08-01) and replaced with the net-T⁴ radiation
exchange + H_bed fuel-bed deposit (§2B). The July numbers are measurements
of physics that no longer exists; only the *shape verdicts* (what Erik
liked) carry forward.

| July claim | Status today |
|---|---|
| Blessed hp=25 shape (ramp → I 0.40 @ ~4 min → out ~14 min) at k_fire_heat=1600 → 14764 K plateau | Shape verdict stands as the taste target; the mechanism is GONE (P-R4) — RE-RUN on today's engine (Phase 3a), plateau now set by H_bed·B and radiation |
| Pass-2 floor: below ~1834 K the fire goes marginal (k_fire_heat 25, K_GROW 0.13, K_DIE 0.03, FIRE_T_EXT 250) | VOID — measured on the deleted k_fire_heat painter AND the pre-fix engine; the "model change needed" conclusion re-opens only if Phase 3a reproduces a wall |
| Far-field room-T rise ~200 game vs target ≤20 | Almost certainly #54's false heating; RE-MEASURE (expect large improvement) |
| "Cool AND vigorous flame needs a MODEL change (decouple sustain heat from displayed T)" | DEFER until Phase 3a re-measurement says the wall still exists |
| k_grow/k_die knife-edge (only 1600 sustains; 800 stalls) | Suspect; RE-MEASURE |
| Fire NOT O2-limited locally (X stays 0.184–0.210) | Plausibly still true; cheap to re-confirm |

Instruments for the re-runs: `tools/fire_timing_harness.py` (P-F4a),
`tests/_fire_bench.py`, `_fire_tuning_artifacts/` (plot_burn.py, tune_fire.py
— Erik's hand-tuning script; from_fire-tuning/ burn curves for comparison).

## 5. Session plan (agreed 2026-08-31)

Phase 0 = this doc → Phase 1 = G12 one-map patch → Phase 2 = per-tile hover
diagnostic (Kelvin via THE map) + dedicated fire-tuning level → Phase 3 =
(a) re-measure July on healed engine, (b) decouple-or-not ruling, (c)
exposure-integral ignition + ember/sustain hysteresis + k_wind_strip revival
(one design), (d) radiation look-over → Phase 4 = joint tuning to §1.1
targets (cool_shift red tests settled here).

## Systems (rules-lifecycle section)

**(a) Existing canonical systems this arc must use:** temperature scale
(src/temperature_scale.py — G12 makes it truly single), config/CFG, material
& gas tables, field digest + GOLDEN_AGGREGATE (one deliberate re-baseline at
G12), gas energy seam + closure identity (any new fire heat path books
itself — there is no fifth group), FieldEdit, tick conductor, benches
(reuse fire_timing_harness/_fire_bench, never a new instrument).

**(b) New systems this arc may create** (draft rules, enter CLAUDE.md at
implementation): per-tile hover diagnostic (THE debug readout — tools read
it, never roll their own field probes); fire-tuning level (THE fire test
scene, level_lib-authored); exposure-integral ignition law (if adopted: THE
ignition path — the instantaneous threshold becomes its limiting case).
