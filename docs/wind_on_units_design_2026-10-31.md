# Wind on units: the momentum-flux threshold law (design, 2026-10-31)

Issue **#140** (a 1.44 atm room pushed a squad out of its door for 35 s). Research
behind the numbers: `docs/research/wind_on_a_person_2026-10-31.md`. Erik's rulings
(#140, 2026-10-31): no wind below a threshold moves a marine's position at all; a band
above it only reduces max walking speed (hunched); above a higher threshold the marine
is knocked over; the push is thresholded on **wind speed x density**, never on the
pressure gradient. This doc is the implementation design. **v3** (lens A: one samples builder
for every cadence caller, plan-time sampling stated as a gap, the int64 cast; lens B:
the air density read from the materials table, the builder ruled and moved to
`movement.py`, onephase grouped with the plan-time callers, the prop clause dropped,
the #140 E2E reproduction, the seam split off as its own digest-neutral patch); the
lens table at the end records the critique.

## 1. What changes, in one paragraph

The `atmosphere | grad` coupling row (`exchange.apply_wave_push`, the P4 linear push
with its `k_push` gain, `push_max_tile_per_tick` cap, `push_grad_floor_atm` noise floor
and `knockdown_dv_threshold`) is **retired** and replaced by ONE row, `wind | max`,
reading the **dynamic pressure** `q = 1/2 rho v^2` at the unit's footprint. Below
`q_slow` the air does nothing to a unit. Between `q_slow` and `q_fall` the unit's
walking cadence is scaled down, linearly in q, to `wind_slow_min_scale` (0.5) at
`q_fall`; its position is never written. At or above `q_fall` the unit is laid
KNOCKED_DOWN (the existing status, refresh-stacked while the wind holds). Nothing in
the law displaces a unit. The gradient path is gone, not kept beside it: the flow
behind a shock front already carries a large q (25-65 m/s induced wind at 0.1-0.3 atm
overpressure, research §3), so a blast still knocks down through the same law.

## 2. The law

Per living unit, per tick, over the GAS tiles of its footprint (a solid tile, including
a walk-through door tile that stores atmosphere 0, is excluded exactly as #121 did):

```
W_i  = N_raw[i] * ((wind_x[i]^2 + wind_y[i]^2) >> 16)      # int, per tile; int64 / Python int
W    = max_i W_i                                           # the footprint's wind load
```

The planes are numpy int32: `wind_x`/`wind_y` are read as Python `int` (or cast
`.astype(np.int64)`) BEFORE the square -- an int32 square wraps silently above
0.707 m/s (46341 raw), as `gas_bulk_n_at` already casts its N. In int64 W stays
under ~7e15 at 10 atm and 400 m/s. `N_raw` is the cell's bulk N (`gmap.gas_bulk_n_at`, O2 + N2, Q16 against the
ambient tile's 65536 = one ambient density; measured on ship_demo: a 1 atm 293 K tile
carries exactly 65536). `wind_x/y` are the EOS velocity in m/s, Q16. So
`W = (N / N_amb) * v^2 * 2^32`, and with `rho = rho_ref * N / N_amb`:

```
q [Pa] = 1/2 * rho_ref * W / 2^32          <=>      W_thr = q_thr * 2^33 / rho_ref
```

`rho_ref` is NOT a new dial: it is the materials table's air row, `[materials.air]
density = 1.2047` ("dry air at the engine's own ambient"), read once per pass as
`gmap.materials.density[MAT_AIR]` (the implementer confirms the accessor's name) --
one owner, no second copy. The thresholds are config in Pa (research picks:
`q_slow = 39`, `q_fall = 245`; the hunch point 120 Pa is in the band, see §5) and are
converted ONCE per pass to int `W_thr` (door 2, like today's `grad_floor_q`). In the
kits' names: `N_amb` raw is `gas_fixed.BULK_ONE`, the wind's one is
`atmosphere_fixed.FP_ONE`, so `2^33 = 2 * BULK_ONE * FP_ONE`; the cadence compose's
ONE is the Q16 one, never `movement.MOBILITY_ONE` (1000).

Hot air at 1 atm has a lower N, so the same wind carries less q; a vented room's thin
fast air is judged by its momentum flux, not its speed. That is the point of q.

### 2.1 Bands

| band | condition | effect |
|---|---|---|
| calm | `W < W_slow` | nothing: scale ONE, bit-identical to today's quiet path |
| slowed ("hunched") | `W_slow <= W < W_fall(s)` | cadence scale `scale_q16 = ONE - ((ONE - s_min_q16) * (min(W, W_fall) - W_slow)) // (W_fall - W_slow)`, then `max(s_min_q16, .)` (W clamped BEFORE the product so the slope term stays in int64 for any dtype) |
| down | `W * ONE^2 >= W_fall * stab_q16^2` (the ONE pinned integer form; `stab_q16` = the unit's stability quantized through `unit_fixed`, as `ZOMBIE_STABILITY` already is) | `apply_status(KNOCKED_DOWN, duration=knockdown_getup_ticks)`; cadence scale is moot (prone units do not move) |

`stability` keeps its meaning as a per-unit multiplier on the knockdown **wind
speed**: the threshold compares `W` against `W_fall * stability^2` (q goes as v^2),
so a zombie at 0.9 topples at 0.9x the wind speed a marine does. The slowed band's
edges are NOT scaled by stability (the lean is a human-scale response, the topple is
the balance the dial is about); `_stability_for` is unchanged.

The scale is linear in q because q is proportional to the drag force the walker
leans against (research §2, §4). `s_min = 0.5` at `q_fall` matches Hunt et al.'s
walking-speed table (about half speed at 14 m/s) within the secondhand scan's error
(research §1, flagged); it is a feel dial.

### 2.2 Where the scale lands: the cadence seam

`movement.default_speed(samples, speed_class)` is THE footprint -> speed reduction
(mobility design §4.1). It has FOUR callers today: the WEGO order path
(`simulation.py` ~1172, `_compute_player_paths`), MOVE_DIR (~1969), the zombie walk
(`ai_zombie.py` ~166) and the onephase cadence (`timeline.py` ~155, `tile_cadence`).
`FootprintSamples` gains a field `wind_scale_q16: int = FP_ONE`; `default_speed`
composes it after the mobility cost: `cost' = half_up(cost * FP_ONE, wind_scale_q16)`
(cost' >= cost; the FP_ONE path is bit-identical to today). The struct's contract
is amended in its docstring: it carries per-tile FIELD values the speed function
reduces (mobility) AND a coupling row's already-reduced Q16 SCALE (wind now; the
planned `water_depth` speed row next) -- `default_speed` composes both, and stays
pure (samples + speed_class, never the unit, never config). **ONE builder fills
it**: `movement.footprint_samples(gmap, fy, fx, fp) -> FootprintSamples` (mobility
from `gmap.footprint_mobility`, the wind scale from `gmap.footprint_wind_load` -- the
pure FIELD read, which is all that lands on GameMap -- through the band law
`exchange.wind_scale_q16(W, settings)`). The builder lives in `movement.py`, not on
GameMap: GameMap is the field store and must not import the coupling layer
(`exchange` imports damage, status, events, species). EVERY `default_speed` caller
in `src/` takes its samples from it -- so zombies are slowed by the same band that
slows marines (they topple sooner, at 0.81 W_fall; they are not immune to the lean).
No state lands on the unit; nothing new is digested or serialized.

Timing. MOVE_DIR and the zombie walk sample the field at execution, at their slot,
i.e. the end-of-previous-tick wind that 9c2 read -- so a unit is either prone or
slowed against the same field, never both. The two PLAN-TIME callers sample when
their path is compiled, over each future DESTINATION footprint: the WEGO order path
(`_compute_player_paths`, at order time, locked decision D) and the onephase cadence
(`timeline.tile_cadence`, called only from `_compile_move` under `compile_plan`,
which runs when orders change). A door that opens mid-round does not slow a
compiled walk (it still knocks the walker down at 9c2, which pauses the path).
ACCEPTED GAP, §4.

### 2.3 The knockdown stays at 9c2

`apply_wind_knockdown(units, gmap)` replaces `apply_wave_push` at slot 9c2, same
position in the tick (after the heat row; a unit the heat row killed is a corpse).
Per living unit: `W = gmap.footprint_wind_load(...)`; `W >= W_fall(stability)` ->
`apply_status(KNOCKED_DOWN, ...)`. It writes NO position. The `wind | max` coupling
row replaces the `atmosphere | grad` row in `COUPLING_TABLE` (reduction `"max"`,
response `apply_wind_knockdown`).

### 2.4 Determinism and dormancy

Every quantity is int: `N_raw`, `wind_x/y` are door-1 integers; `W_thr` and
`s_min_q16` are door-2 constants quantized once per pass; the band arithmetic is int
`* // max min`, the stability compare the pinned integer form of §2.1. Dormancy does
NOT rest on `W == 0`: a real quiet map's EOS jitters a few counts (#116), so W is
small but nonzero; it rests on the calm band returning exactly ONE below `W_slow`
(~2.8e11 counts, eleven orders above the jitter) and 9c2 applying nothing below
`W_fall`, so every quiet trajectory keeps its digest. Positions are never
written by the law, so the float position discipline is untouched. There is no CUDA
twin: this is Python-side unit coupling, like every other exchange row.

## 3. What is retired

- `exchange.apply_wave_push` and its docstring's P4 design notes (the history stays in
  git and in mechanics/05 §1, which gets a dated "superseded by the wind law" line).
- `[exchange]` dials `k_push`, `push_max_tile_per_tick`, `push_grad_floor_atm`,
  `knockdown_dv_threshold`; their comment block is replaced by the wind block (§6).
  `knockdown_getup_seconds` and `stability` stay. `reduce_grad` stays in the
  reduction vocabulary (tested in `test_exchange_reductions.py`); no row uses it.
- `tests/test_wave_push.py` is rewritten to the §7 properties (the file keeps its
  name's subject: the row at 9c2; the row's table position is never pinned).
  `tests/test_exchange_reductions.py`'s two table tests that name the push row's
  reduction and response are re-pointed to the wind row -- and their 5-tuple unpack of
  `COUPLING_TABLE` (a count pin on a table designed to grow) is replaced by a lookup
  by field name: a pre-existing finding, fixed in passing because the test is being
  touched anyway. `tests/test_wave_push_door.py` and
  `tests/test_standing_unit_stays_put.py` pass unchanged but their "breaks if" prose
  names the retired floor; both docstrings are restated against the wind law (the
  door test's property becomes: a footprint half on a door tile walks at the calm
  cadence on a quiet map).
- **Debt, recorded**: `unit.mass` loses its only reader (the push divided by it);
  mechanics/05's "big light units fly" is no longer true. Mass stays on the unit
  (drawn from the species, serialized) for the next row that needs it.
- The row being replaced is the table's `field="wave_p", reduction="grad"` row (its
  docstring says atmosphere since the EOS refactor); the new row is
  `field="wind", reduction="max"` with the note that the max is over the DERIVED
  per-tile W, not `REDUCTIONS["max"]` over a plane (as the heat row annotates).
- Every stale reader of the retired dials is reworded in the same patch: the
  `[zombie] stability` comment, `status.py`'s KNOCKED_DOWN note, `environment.py`
  (two sites), `species.py`, the 9c2 block in `simulation.py` and its import,
  `gamemap.py`'s two comments naming `apply_wave_push`, `exchange.py`'s module
  docstring and `__all__`, `docs/playground_guide.md`'s how-to (it tells the reader to
  edit `k_push`), and the canon chapters mechanics/05 §1, mechanics/06 §4 (the ring
  signature, now "measured, not preserved") and engine/04 -- one dated line each,
  never a rewrite. Dated history docs are left as they are.

## 4. ACCEPTED GAPS (decisions, not findings)

- **No displacement anywhere.** A knocked-down unit lies where it fell; no ragdoll
  slide, no blast buffet. The P4 buffet was measured as a +-0.2 tile wobble (config
  comment); losing it is Erik's ruling (the linear push is not realistic). A slide
  for prone units in sustained wind is a later row.
- **No "hunched" state or sprite.** The band's visible effect is the slower walk.
  A hunched posture (no running, a snap cone, a sprite) is a status for a later
  patch; 120 Pa is recorded in the config comment as where it would start.
- **A prone unit keeps the standing footprint and thresholds.** It stays down while
  `W >= W_fall` (refresh stacking) and gets up when the wind drops; a smaller prone
  drag area is not modelled.
- **The grenade knockdown ring is re-measured, not preserved by construction.**
  mechanics/06 §4's signature ("knockdown radius ~2x damage radius") was tuned on
  the gradient law. Under q the ring is whatever the blast's EOS wind gives at
  245 Pa. The patch measures it (§7) and reports; if the ring collapses, that is a
  finding about the blast's pressure tuning (another system, #4/#16's neighbourhood)
  and Erik rules at the screen; the design is not bent to the old ring.
- **One footprint size.** `max` over a 3x3; a larger unit's area scaling is not
  modelled (as before). `max` over-reads a one-tile jet across a three-wide body by
  up to 3x against the drag integral (a mean); the research picked max and it matches
  the gas rows. The probe prints mean W beside max so Erik sees the gap; switching
  the reduction is one word if he rules so.
- **The two plan-time paths (WEGO orders, onephase) sample the wind when the path
  is compiled** (§2.2 timing): locked decisions of those designs, not this one. Bots
  and zombies (MOVE_DIR, the zombie walk), which is where #140 was seen, sample at
  execution. **Erik's ruling
  2026-10-31: WEGO is legacy, not the main way to play** -- the RL path makes
  decisions continuously (ContinuousRealtime, intents / MOVE_DIR, matches), and a
  WEGO regression is reported, never designed around. This gap is therefore not a
  feel concern; the patch keeps WEGO working as it is and spends nothing on it.
- **Cadence quantization blunts the low band**: an 8 ticks/tile walker does not
  slow until the scale is under 0.941 (q ~63 Pa, ~10 m/s); a 1 tick/tile sprinter
  jumps 1 -> 2 at 0.667 and nothing before. Written into the config comment.

## 5. Config (the `[exchange]` wind block)

```toml
# --- Wind on units (#140, docs/wind_on_units_design_2026-10-31.md) -----------
# q = 1/2 rho v^2 at the footprint (max over its gas tiles), rho from the tile's
# bulk N against ambient ([materials.air] density, the one owner). Research picks:
# docs/research/wind_on_a_person_2026-10-31.md (39 Pa = 8 m/s, 245 Pa = 20 m/s
# at sea-level density; the hunch point, 120 Pa = 14 m/s, sits in the band).
wind_q_slow_pa = 39.0           # below this nothing; above, cadence scales down
wind_q_fall_pa = 245.0          # at stability 1.0: KNOCKED_DOWN (stability scales the SPEED: W_fall * stability^2)
wind_slow_min_scale = 0.5       # cadence scale at q_fall (linear in q from 1.0 at q_slow)
knockdown_getup_seconds = 1.5   # unchanged
```

Door: `config.py` validates nothing and Ctrl+R swaps sections unguarded, so the
project's door pattern is a consumer-side settings function read every pass
(`control_bots.bot_settings`, `match_replay.replay_settings`, `relief_settings`):
`exchange.wind_settings(cfg) -> WindSettings` refuses a missing or mistyped key by
name and refuses `q_slow >= q_fall`, and -- on the quantized ints -- `W_slow >=
W_fall` (two Pa values close enough to quantize equal would divide by zero in the
band), and `wind_slow_min_scale` outside (0, 1]. The row and the builder call it
every pass (hot-reload as today). `wind_rho_ref_kg_m3` does NOT exist (§2: the
materials air row is the one owner).

## 6. Patches

| patch | scope | mode | tier | gate |
|---|---|---|---|---|
| P1a | THE SEAM, digest-neutral: `FootprintSamples.wind_scale_q16` (default FP_ONE) + the compose in `default_speed`; `movement.footprint_samples` builder; the four `src/` callers moved onto it (at scale FP_ONE, nothing else); `GameMap.footprint_wind_load` (the pure int64 field read, unused by the law yet); the §7 grep test (6) and int64 test (6b); the E2E reproduction of #140 written and left FAILING-BY-DESIGN (marked xfail with the issue, flips in P1b) | same agent, first commit on worktree `140-wind-q-threshold` off main | (Opus agent; the gate is the oracle) | `pytest tests -q` green and **no golden moved** -- committed before P1b starts, so golden attribution stays clean |
| P1b | THE LAW: `exchange.wind_settings` / `wind_scale_q16` / `apply_wind_knockdown` / the `wind | max` row; the builder wired to the law; 9c2; the config block; retire the four dials and every stale reader (§3); rewrite `test_wave_push.py`, re-point `test_exchange_reductions.py`, restate the two prose-stale tests; the #140 E2E test flipped to pass; the tile inspector's q / W row (`hover_readout`, so Erik sees the number at the screen); the probe + the three measurements; CLAUDE.md rows (§8) and the one-line canon touches | same agent, second commit(s) | **Opus** (feel-gated; no oracle beyond the property tests) | `pytest tests -q` green with GOLDEN_AGGREGATE re-baselined (§6.2), then **HUMAN-TEST**: Erik plays ship_demo's 1.44 atm room and a grenade on fire_smoke_studio / explosion studio before merge |
| P2 | tracker: #140 comment with the measurements, CLAUDE.md row final wording | inline | — | — |

One implementation agent; no parallel work in the tree.

### 6.2 The golden re-baseline (certain, not possible)

GOLDEN_AGGREGATE moves by construction: the canonical harness marine spawns at x = 7
and the current golden records it pushed to (6.98748, 7.00063) after 30 ticks, where
`int(x)` is 6 -- so without the push its tile, its occupancy stamp and the gas fields
under it may move too, not only `__unit_pos__`, and `__unit_status__` if the pulse's q
crosses 245 Pa. The rationale follows the established form: MOVED / UNMOVED
sub-hashes named, SOLE CAUSE shown by reproducing the new digest on main with
`k_push = 0` and `knockdown_dv_threshold` set huge (if that does not reproduce it,
the wind law fires in the canonical scenario and the report explains exactly where),
and the ten `cuda_*_check.py` readers of the golden re-run on a fresh CUDA build
(`cpp/build_cuda.bat` on this machine) as every past re-baseline note records.

### 6.1 The measurements P1 must report (the probe)

`tools/wind_load_probe.py` (headless, a `Simulation` built directly for each scene --
never a copy of `run_match`'s loop): prints per tick the max `W`, `q` in Pa and `|v|`
at a named unit's footprint, for
1. the #140 scene: ship_demo's 1.44 atm room, a marine standing in its doorway,
   from the moment the door opens -- is q at the body above 39 / 245 Pa, and for how
   long? (This is the number the research could not give: whether 245 Pa fixes the
   bug.)
2. the calibration grenade (radius 5, pressure 10.0; the config comment's
   scenario): peak q at a marine by distance d = 2..12 tiles -> the knockdown ring
   under the new law, beside the old 9-10 tiles.
3. a marine standing beside the steady fire on fire_smoke_studio: the raw wind
   planes spike at fire cells (the tamed-wind rule's warning) and nobody has measured
   what q that is at a body -- does walking past a fire slow or floor a marine?
Each prints max AND mean W / q over the footprint. All three land as a table in P1's
report and in the #140 comment.

## 7. Verification plan (properties; each test names what breaks it)

1. **Dormancy**: a REAL quiet level (ship_demo, doors shut; its EOS jitter is the
   case) runs bit-identically against the same run with `q_slow` / `q_fall` set
   astronomically high (the comparator): positions, statuses, cadence. Breaks if the
   law touches a quiet unit.
2. **Position is never written**: a unit in a synthetic wind above `q_fall` has the
   same `x, y` after the row. Breaks if a displacement term returns.
3. **q, not v**: at the same wind, halving the tile's N halves W (int-exact), and a
   wind just under `v_slow` at double N is in the band. Breaks if density leaves
   the law.
4. **Band edges**: scale FP_ONE at `W_slow - 1`, FP_ONE at `W_slow`, `s_min_q16`
   at `W_fall`, the midpoint checked against the FLOAT law within half an LSB (never
   against the implementation's own integer expression), monotone non-increasing
   across a sweep. Breaks if the law is not linear in q or the clamps go.
5. **Knockdown edge and stability**: KNOCKED_DOWN at the pinned integer edge, not
   one count below; a 0.9 overlay zombie goes down at the QUANTIZED edge
   (`58982^2 / 2^32`, about 0.80999 x W_fall, not 0.81). Breaks if stability stops
   scaling the wind speed or the compare form moves.
6. **Every cadence caller takes the scale**: a MOVE_DIR walk, a zombie walk and the
   onephase cadence across a tile in a `s_min` wind each take
   `half_up(cost * ONE, s_min_q16)` ticks against the field at execution; the WEGO
   order path and onephase take it against the field at compile time. The test
   finds the callers of `default_speed` by grep OVER `src/` ONLY (tests build samples
   on purpose) and asserts each goes through `footprint_samples` -- never a pinned
   list, never a count. Breaks if a `src/` caller builds its own samples.
6b. **int64**: at v = 50 m/s and N = 2 x ambient, `footprint_wind_load` equals the
   Python-int formula. Breaks if an int32 square returns.
7. **The #140 reproduction, end to end** (master rule: a bug fix starts with the
   bug as the user sees it): ship_demo's ~1.44 atm room (the #139/#140 scene,
   rebuilt through `Simulation` + FieldEdit if the level no longer ships it): the
   door opens; a squad standing with no orders keeps its exact x, y for the whole
   vent; a MOVE_DIR walker leaving through the door is never moved by the row and
   takes at least the calm cadence. Written in P1a as xfail (it FAILS on main: the
   squad is pushed), flipped in P1b. Breaks if any displacement returns. (The old
   test 7 -- door tiles excluded -- was a tautology: a solid tile's N is 0 so its W
   is 0 under a max; dropped. The door test's docstring is restated, §3.)
8. **Config door**: `q_fall <= q_slow`, `s_min` outside (0, 1], `rho_ref <= 0` are
   refused by name.
9. **Determinism**: the grenade scenario twice -> identical positions and statuses.
10. **Re-knock refreshes**: sustained `W >= W_fall` keeps the unit down past one
    get-up time (the existing P3 stacking, through the new row).

The old `test_knockdown_ring_wider_than_damage_ring` is NOT carried as an assertion
(§4): its measurement moves to the probe and the report.

## 8. Systems

**Existing canonical systems this design uses** (CLAUDE.md inventory):
- Coupling table (`exchange.py`): the law is one row, `wind | max`, replacing the
  `atmosphere | grad` row -- not plumbing beside the table.
- Movement cadence seam (`movement.default_speed` + `FootprintSamples`): the slowed
  band is a composed factor on the one footprint -> speed reduction, never a second
  speed path or a per-unit speed field (its §4.1 lockstep contract: the function
  reads samples, never the unit).
- Status system (`status.py` KNOCKED_DOWN, refresh stacking): the knockdown is the
  existing status; no new kind.
- GameMap field store (`gas_bulk_n_at`, `wind_x/y`): read, never written; the wind
  is the EOS velocity in m/s Q16 (the tamed-wind rule's conversion, minus the render
  taming).
- Config (`CFG.exchange`, Ctrl+R): the four dials; `config.py` holds the door.
- Q16 boundary modules (`atmosphere_fixed`): no inline 65536; `>> 16` on the
  velocity square is written through the kit's shift constant.
- Tick conductor: slot 9c2 keeps its one call line.

**Rule to write into CLAUDE.md's Control seams row** (Erik, 2026-10-31, from this
design's lens A): *WEGO (`TwoPhaseWEGO`, `WEGOPlanningInput`, `--control wego`) is
LEGACY: the main line is continuous control -- `ContinuousRealtime`, intents /
MOVE_DIR, the match controllers and the RL path, which decide every tick. New
designs sample, gate and tune on the continuous path; WEGO's locked decisions
(order-time path precompute, the per-round rewind) constrain nothing new, and a WEGO
regression is a one-line finding, never a design driver. Incident: #140's wind law,
where WEGO's plan-time sampling was raised as a blocker.*

**New reusable system** (draft rule, written into CLAUDE.md at implementation):
- *Wind load on a body* -- `GameMap.footprint_wind_load` + `exchange.wind_scale_q16`
  / `apply_wind_knockdown` (`[exchange] wind_*`): THE read of what the air does to a
  unit -- dynamic pressure `q = 1/2 rho v^2` as the int `W = N_raw * (v^2 >> 16)`,
  max over the footprint's gas tiles, thresholded in Pa; below `q_slow` nothing, a
  linear cadence scale to `q_fall`, KNOCKED_DOWN at `q_fall * stability^2`; rho is
  the materials air row, the thresholds `exchange.wind_settings`. Never a
  pressure-gradient push, never a displacement; a new wind effect on a unit (a prone
  slide, a hunched posture) reads W through this seam, never the planes. (Render-side sway stays the tamed wind's, per the Static
  props row -- this seam is for the SIM's effect on a unit, nothing render.)
- *Footprint samples builder* -- `movement.footprint_samples(gmap, fy, fx, fp)`: THE
  `FootprintSamples` builder; a coupling row's speed output is a Q16 scale member on
  `FootprintSamples`, composed in `default_speed` (pure: samples + speed_class, never
  the unit, never config); never an inline `FootprintSamples(...)` in `src/`
  (tests may build their own). Gated by the grep test. The next member is the
  planned `water_depth` speed row.

## 9. Critique lens table

| lens | model | verdict | blockers resolved in |
|---|---|---|---|
| A. physics and feel semantics, determinism, the band law | Opus | needs v2 for: (B1) the WEGO order path samples wind at order time over the destination footprint, not at execution -- §2.2 claimed otherwise; (B2) `default_speed` has four callers (zombie walk, timeline cadence too), so zombies would never be slowed; (B3) int32 square of the wind planes wraps above 0.707 m/s. Concerns: clamp W before the slope product; door on the quantized ints; max over-reads a jet (print mean); fire-cell spikes unmeasured (probe 3); cadence quantization blunts the low band (comment); dormancy is the calm band, not W == 0; pin the stability compare form; stale prose on the retired dial. Confirmed: W = (N/N_amb) v^2 2^32 and W_thr = q 2^33 / rho_ref; the band is monotone and integer-exact; half_up monotone; MOVE_DIR timing consistent; static overpressure without flow is correctly nothing; hot thin air handled the right way round; prone refresh coherent | v2: §2.1 (clamp, pinned compare), §2.2 (one `footprint_samples` builder, all four callers, the timing paragraph), §2.4, §4 (three gaps), §5 (door, prose), §6.1 (probe 3, mean), §7 (tests 1, 5, 6, 6b) |
| B. systems reuse, scope and regression (tests, goldens, both move sites, retired dials, docs) | Opus | needs v3 for: (B1) `wind_rho_ref_kg_m3` duplicated the `[materials.air] density` row, wrongly (1.204 vs 1.2047); (B2) the samples builder was a new canonical system with no rule, placed on GameMap (the field store importing the coupling layer) and carrying a pre-reduced scale against the struct's contract; (B3) the onephase cadence is compiled at plan time (`_compile_move` under `compile_plan`), so §2.2 and test 6 were false for it; (B4) the draft rule's prop-sway clause contradicted the Static props / Tamed wind rows; (B5) no E2E reproduction of #140 (master rule). Concerns folded: consumer-side `wind_settings` door (config.py validates nothing, Ctrl+R is unguarded); the golden moves by CONSTRUCTION (the harness marine sits at x 6.987, tile 6, under the push) -- MOVED/UNMOVED + SOLE CAUSE + the ten CUDA checks; P1 split into a digest-neutral seam patch and the law; every stale reader listed; `unit.mass` loses its reader (debt); the row's real name is `wave_p | grad`; kit names (`BULK_ONE`, `FP_ONE` vs `MOBILITY_ONE`); grep scoped to `src/`; test 7 was a tautology; test 1's comparator, test 4 against the float law, test 5's quantized 0.80999; the two prose-stale tests; the tile inspector's q row; the probe builds `Simulation` directly. Breaks by construction: all of `test_wave_push.py` (module-load imports), two `test_exchange_reductions.py` table tests, every GOLDEN_AGGREGATE importer until re-baselined. Confirmed: the row is the right home, one 9c2 line; KNOCKED_DOWN reused; no new synced state, spec untouched; defaulted member breaks no constructor; stability already quantized; `occupied_tiles` == the mobility block; match/replay digests survive; no tool reads the push | v3: §2 (rho, kits), §2.2 (builder in movement, contract, plan-time group), §3 (tests, debt, naming, stale readers), §4, §5 (door), §6 (P1a/P1b, §6.2 golden), §6.1, §7 (1, 4, 5, 6, 7), §8 (two rules, prop clause gone) |
