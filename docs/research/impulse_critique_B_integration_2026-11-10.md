# Critique lens B: determinism, integer arithmetic, integration. Unit impulse law, draft 2

VERDICT: BUILD AFTER THE LISTED CHANGES. The law can be made exact and order-free, but four things in draft 2 cannot be built as written:
- B1. The momentum identity cannot be EXACT. The draft divides J/m with rounding, and mass is a float kg.
- B2. The movement slot never reaches the units the law most needs to move: zombies, KNOCKED_DOWN units, and units holding no MOVE_DIR.
- B3. The turn is placed after the move in slot 3. That breaks the Speed table's "this tick's facing" rule and the bots' exact trigger predictor. It also lacks a persistent target, which every non-aim writer needs.
- B4. The dormancy gate ("every input off, unlimited acceleration, all pins unchanged") is false as written. Recoil has no switch. Zombie, travel and snap facings become capped and lagged. The facing currency alone moves `__unit_facing__`.

Twelve majors follow. Most of them concern undefined currencies and rounding. Python's `//` floors while C and CUDA truncate. There is no velocity bound, so the substep loop and int32 pybind arguments are unbounded. The draft leaves undefined where wind, effort and suppressed walking enter the zombie and WEGO paths, the digest and recording rows, and the shove's missing cooldown and selection rule.

Smallest fix set, in one sentence each:
- B1. Keep mass as integer grams, and keep the remainder of each division in the accumulator instead of clearing it. The identity then closes exactly in int64.
- B2. Add one pass over EVERY living unit, and one Q16 move function shared by all paths. Invalidate a pushed zombie's path.
- B3. Add a new conductor slot 1a, "integrate + turn", before 1b. Give each unit a persistent `facing_target_q`. Every facing writer sets the target; only `turning.step` writes `facing_q`.
- B4. Make the representation change its own commit with its own re-pin. Add a `recoil_mult` dial. Rewrite the dormancy gate as three non-vacuous properties (§8 below).

Port-readiness: good once the above is done. All state is fixed-width scalars, which fits P3a. The integration is one thread per unit with bounded loops. The accumulators are integer sums, so phase A records and the applier or an int64 atomicAdd applies them, with no order dependence. Velocity in Q16 tiles/tick is exactly what x_q needs at P3c.

Severity counts: 4 BLOCKER, 12 MAJOR, 11 MINOR. The implementer's sheet (representation, location, order) is at the end.

---

Evidence paths are relative to the worktree
`C:\Users\steen\projects\breach\.claude\worktrees\183-gameplay-pack-v1`.
Numbers: tile 0.333 m, 24 tps, so 1 m/s = 0.125 tiles/tick = 8192 Q16 counts
of tiles/tick.

## 1. Arithmetic left undefined

**B1 (BLOCKER) -- the momentum identity is not exact as specified.** §5 step 1
says "dv = imp / m; clear the accumulators". That division rounds, and
`Unit.mass` is a float: `src/simulation/generation.py:39-69` returns it
Q16-snapped as a float, stored at `src/simulation/unit.py:171`. `kit_kg` will
be another float. `m·Δv` then differs from `ΣJ` by the rounding of every
division, so "exactly, in integers, per unit per tick" (§5, §8) cannot hold.
The in-phase / out-of-phase gate also differs by one rounding per tick of the
split.
*Resolution:*
- Mass becomes an INTEGER per unit, quantized once at spawn: `m_g` in grams
  (`Unit.mass` + `kit_kg`; the door refuses <= 0).
- The impulse currency `J` is Q16 gram·tile/tick, int64.
- Per axis, `dv = div_trunc(imp, m_g)`, and the accumulator KEEPS
  `imp - dv·m_g`; it is never cleared.
- Then `Σ J_in = m_g·Δ(pv+sw) + m_g·resisted + m_g·wall + m_g·clamped +
  Δimp_residual` holds exactly in int64, every tick. An in-phase train and
  its out-of-phase split give the same summed dv to the count.
- The spin works the same way. `I` is an integer in g·tile² (3.0 kg·m² =
  27 000 g·tile² at 0.333 m), `imp_l` is in Q16 g·tile²/tick, and
  `dw = div_trunc(imp_l, I)` is in Q16 rad/tick, with the remainder kept.

**M1 (MAJOR) -- no currency is named for any state or parameter, and the
quantization door is wrong if it is the weapons table.** The parameters are
in m/s, N·s, kg·m² and m. The tile size is PER MAP (`config.toml:170`; the
Speed table row says "quantized ONCE per map (its `tile_size_m`, `[clock]
ticks_per_second`)").
*Resolution:* build an `ImpulseTable` once per `Simulation`, beside
`Simulation.speed_table`. It goes through a refusing door `[impulse]`
(`impulse_settings(cfg)` + `impulse_table(settings, tile_size_m, tps)`) and
holds everything in GAME units:
- `cap_q`, `cap_prone_q`, `v_inplace_q[3]`, `v_fall_q[3]` in Q16 tiles/tick;
- `t_react` in ticks, and `n_rec`;
- `r_body_q` in Q16 tiles;
- `I_g` in g·tile²;
- `a_q`, `w_q` in Q16 rad/tick² and rad/tick;
- per ammo row, `J_q = quantize(bullet_g/1000 · muzzle_mps · mult /
  (tile_m·tps)) · 1000`, in Q16 g·tile/tick (carbine: 3.6 N·s -> 450
  g·tile/tick -> 2.95e7);
- `shove_J_q`;
- `k_slow / k_hold` as a Q16 effort edge.

Every compare in §5 is then between integers of one currency. The door also
refuses `cap_q >= min(v_inplace_q)` on the QUANTIZED values (753 vs 1229
counts at 24 Hz), not on the reals.

**M2 (MAJOR) -- rounding directions the kits do not provide, and a Python /
C++ trap.** Python `//` floors; C and CUDA `/` truncate toward zero
(`cpp/src/fixed_point.h:781-799` exists precisely because of this). The draft
asks for components "rounded toward zero" and "rounded away from zero" but
names no helper.
*Resolution:*
- Add two helpers to `unit_fixed` (Python), each with a `fixed_point.h` twin
  and a twin test: `div_trunc(n, d)` = sign·(|n| // |d|), the C `/`; and the
  effort step `sign(d)·((|d| + 7) >> 3)`.
- The magnitude is `math.isqrt` (precedents: `src/simulation/flash_blind.py:54`,
  `src/simulation/combat.py:1493`). Its C++/CUDA twin is `sqrt_q16`
  (`fixed_point.h:886`: fixed 32 trips, exact floor for radicands < 2^62).
  State it and gate the equality.

Every vector operation then has one integer form:
- `|v|^2 = vx^2 + vy^2`, compared squared wherever possible (steps 3 and 6),
  so no compare needs a sqrt.
- Sway overflow (step 3): `n = isqrt_ceil(|sw|^2)` (ceil, so that
  `|kept| <= v_in`); `kept = div_trunc(sw·v_in, n)` per axis;
  `pv += sw - kept`; `sw = kept`. The transfer is exact: momentum moves
  between `sw` and `pv` and nothing is lost.
- Shrink (step 4): `n = isqrt(|x|^2)`. If `n <= b`, x = 0 and used = n.
  Otherwise `x' = div_trunc(x·(n-b), n)`, and used is the booked `x - x'`
  per axis (the identity books components, not the magnitude).
- Direction limit: `c = movement.facing_cos_q16(vx, vy, facing_q)`, the one
  existing law, never a second atan2-wrap-cos copy. Then apply the Speed
  table's own blend: for `c >= 0`, `side + (fwd - side)·c`; for `c < 0`,
  `side + (side - back)·c`; then `>> 16`, floor. Say which vector it is in
  each step: sw in step 3, pv+sw in step 6, pv for the prone slide.
- Demand: `min(div_trunc(isqrt(|dv|^2)·ONE, cap_q), 2·ONE)`.
- Effort to walking scale: `e0 = k_slow/k_hold` in Q16. The scale is `ONE`
  for `e <= e0`, `ONE - (ONE/2)·(e - e0) // (ONE - e0)` up to `e = ONE`, and
  `ONE/2` above that; floor.

**M3 (MAJOR) -- there is no velocity bound, so int32 arguments, int64
products and the substep loop are all unbounded.** A blast wind of
q ~ 1e5 Pa over 0.9 m² is about 3750 N·s per tick. That is dv ≈ 37 m/s
≈ 3.1e5 Q16 counts per tick, and nothing stops pv accumulating over a
multi-tick front. The consequences:
(a) `facing_cos_q16` hands pv / sw to the pybind `atan2_q16`, which is int32
and raises above 2^31 (`unit_fixed.py:58-67`);
(b) the step-6 compare `|pv+sw| >= v_fall·stability`, written as
`|v|^2·ONE^2 >= (vf_q·st_q)^2`, overflows int64 (1.4e21) in the C++ twin;
(c) the substep count `ceil(|d| / step)` has no bound, so it has no device
form.
*Resolution:*
- Add a door constant `V_MAX_Q` (for example 4 tiles/tick = 32 m/s). Apply
  it to pv after step 3, count each clamp, and book it as `clamped` in the
  identity.
- Form `vf_eff = (vf_q · st_q) >> 16` once per unit per tick and compare
  `|v|^2 >= vf_eff^2`.
- Compare `effort·ONE >= st_q^2`, as the wind law does
  (`src/simulation/exchange.py:579-585`).
- Make the accumulators int64. A shove is 12 500 g·tile/tick, which is
  8.2e8 in Q16, so two shoves overflow int32. The torque
  `J_q · lever_q >> 16` is also int64.

With V_MAX the largest values are: |pv| <= 2.6e5 Q16, |omega| < PI_Q16 =
205 887, effort <= 131 072, |v|^2 <= 1.4e11. Every square fits int64, and
every pybind argument fits int32.

**M4 (MAJOR) -- the torque sign and the lever arm have no integer
definition.** The world is y-down, while `facing` is math-style y-up
(`unit.py:441-459`, `combat.py:1042-1047`). `omega` adds to the facing, so
the cross product must be taken in the facing's frame. Otherwise every
off-centre push turns the unit the wrong way, and that fault would be
deterministic and invisible to every gate in §8.

The draft's "perpendicular distance ... from the march's own geometry" is
float today: `rx/ry` are floats, the bearing is `base_angle + rng.uniform`
(`combat.py:1193-1202`), and the unit centre is the float `x + fp/2`.
*Resolution:*
- lever `= ((cx_q - ox_q)·sy_q - (cy_q - oy_q)·sx_q) >> 16`, in Q16 tiles,
  where:
  - `o` is the round's integer origin `<< 16`;
  - `s` is `quantize(step_x, step_y)`, exact n/65536 by construction
    (`combat.py:1199-1200`), with `|s| = ONE` within the kit's 2 counts,
    stated;
  - `c` is `quantize_scalar(u.x) + (fp << 15)`.
- Clamp the lever to `±r_body_q` per input. The clamp must be on each input,
  never on the running sum, or order-freedom is lost.
- `imp_l += J_q·lever_q >> 16`, with the sign chosen so that positive means
  CCW in the facing's frame. Add a gate for it (§8 G6).
- The shove's lever uses the same formula, on the centre deltas and the
  shover's facing unit vector `(cos_q16(f), -sin_q16(f))` in world axes.

**M5 (MAJOR) -- the turning law is exact but under-specified for an
implementer.**
(i) The `s(e)` search does not say what it searches. The closed form, for
each n:
- On `x ∈ (n·a, (n+1)·a]`, `B(x) = (n+1)x - a·n(n+1)/2`.
- So `x_n = min((n+1)·a, (e + a·n(n+1)/2) // (n+1))` (floor).
- Take the largest feasible `x_n`, capped at `w`, over `n = 0 .. ceil(w/a)`.
  At the shipped dials that is 6 trips (w = 17 157, a = 3 413 counts).
- The door bounds `ceil(w/a)`, so the device loop has a fixed length.

(ii) "Unlimited acceleration" must be a stated condition: `a >= 2w`. Only
then does the clamp `omega ± a` never bind on a reversal, and only then does
the law equal `turn_toward_q16` (`turning.py:56-64`) bit for bit. Make that a
property test.

(iii) `facing += omega` goes through `turning.normalize_q16`, with
`|omega| <= PI_Q16 - 1`.

(iv) What `e` is when nothing names a target is answered by B3: the target
persists.

## 2. Positions are float64 until P3c

**M6 (MAJOR) -- "walk + pv as one displacement in half-tile substeps"
changes today's walk and adds a float site.** Today `_step_move_dir` makes
ONE test plus a slide (`simulation.py:2242-2256`): the full move, else x
alone, else y alone. The marine's full speed is 0.563 tiles/tick
(`config.toml:109`, 1.776 ticks per tile), so a half-tile substep rule splits
a sprint into two tests even at pv = 0, and diagonal slides then resolve
differently. `u.x + pv/65536` is also a new float add on the match path,
against pack §2 rule 1.

*Resolution, and what the implementer should do:*
- Write ONE function, `unit_impulse.move(gmap, u, walk_dx_q, walk_dy_q)` in
  Q16 tiles, shared by every path (§3).
- Convert the walk to Q16 now. `move_speed_tiles_per_tick` gains a
  Q16-returning twin: the same exact rational, rounded half-up to Q16 once,
  instead of a float division (`movement.py:229`). The step is
  `(dx_q·v_q) >> 16` per axis, toward zero.
- On entry, `x_q = quantize_scalar(u.x)`; on exit, `u.x = x_q / 65536`.
  Every other position writer (the WEGO path, the zombie step, the
  `_end_round` rounding, onephase) already writes integers or dyadics, so
  after this patch `u.x` is ALWAYS on the Q16 grid.
  - P3c then becomes a pure representation change: `x_q` is stored and `x`
    becomes a property, with no further trajectory change.
  - Gate it: `u.x·65536` is integral after every tick of every pinned match.
- Substeps: `k = max(1, ceil(max(|dx_q|, |dy_q|) / S))` with `S = ONE - 1`.
  - That is just under one tile, the bound the Speed table's door already
    enforces for a walk step. A step under one tile per axis cannot skip a
    tile column with the floor-anchored footprint.
  - At pv = 0 this gives k = 1, which is today's test exactly.
  - Split `d` into k integer parts that sum to `d` (`div_trunc`, with the
    remainder on the last part).
- Each substep runs today's slide rule:
  - The axis not taken is "into the wall". Zero pv's component on that axis
    (`wall += m_g·pv_axis`) and drop the rest of the walk on that axis.
  - A fully blocked substep ends the move.
  - Spell out the corner case: both single axes are free but the diagonal is
    blocked. Today x wins, so pvy is the one zeroed.
- `is_passable_block(int(ny), int(nx))` becomes `(y_q >> 16, x_q >> 16)`,
  which equals `int()` for non-negative positions.

Under this scheme the walk is unaffected while pv = 0, apart from the Q16
rounding of the walk step itself. That rounding is a one-time re-pin, and it
belongs in the same commit as the facing representation (B4).

## 3. Slot placement and order

**B2 (BLOCKER) -- the movement slot does not reach the units the law most
needs to move.** In `_update_player_movement` (`simulation.py:2155-2194`):
- zombies `continue` at line 2156;
- MOVE_DIR units reach `_step_move_dir` only `if composed_flags(u).can_move`
  (lines 2166-2169), so a KNOCKED_DOWN unit (`status.py:132`, can_move
  False) never slides;
- a unit after `clear_move_dir` (the bots' stand-and-shoot,
  `control_bots.py:217`) falls into the WEGO branch and does nothing;
- WEGO units are TELEPORTED to absolute plan positions (`u.x = px`, line
  2193), which overwrites any push on the next tick;
- under OnePhaseWEGO, `drives_units` replaces slot 3 entirely (lines
  1777-1778).

Zombies move in slot 5 by snapping to the next A* node
(`ai_zombie.py:210-217`), and only the DESTINATION is checked with
`is_passable_block`. A zombie pushed two tiles off its stale path therefore
jumps back across whatever lies between, walls included. The questions "does
a pushed zombie move?" and "where is its integration?" have no answer in the
draft.

*Resolution:*
- The INTEGRATION (steps 1-6 and the turn) runs in its own slot 1a, over
  every living unit (B3).
- The DISPLACEMENT runs in slot 3, as ONE loop over every living unit in id
  order:
  - a MOVE_DIR walker with can_move calls `move(walk + nothing)`: its walk
    and its pv go in as one displacement, as the draft wants;
  - every other unit with `pv != 0` (standing, prone, zombie, a WEGO or
    onephase unit) calls `move(0, 0)`.
  - Each call reads only the map and the unit's own state, so the loop is
    order-free (units pass through each other, #142).
- A zombie displaced this tick gets `zombie_path = []`, so its next slot-5
  step repaths from its floored tile. That step still snaps it onto the tile
  grid, a sub-tile jump: an accepted gap until P3d's distance field.
- WEGO and onephase receive the push, but their plans overwrite it. State
  that as an accepted legacy gap (the Control seams row: a WEGO regression
  is a one-line finding).
- For onephase, call the displacement loop outside the `drives_units`
  branch, so the rule is uniform.

**B3 (BLOCKER) -- the turn's place in the tick contradicts two existing
rows, and the targets have nowhere to live.** The draft puts the turn at step
8 of the slot-3 integration, after the move. Today:
- the aim turn runs at 1b (`simulation.py:1124-1127`);
- the Speed table's direction rule reads "THIS tick's facing (after the turn
  cap)" (`simulation.py:2210-2237`); with the turn after the move it would
  read last tick's facing;
- the bots decide the trigger in `handle_frame`, BEFORE `step`, from
  `facing_after_turn_q16` (`control_bots.py:263-270`,
  `simulation.py:980-989`; the Turn-rate cap row says "never a copy of the
  law").

With the turn in slot 3, the turn consumes `dw` from slot-2 rounds that have
not landed when the predictor runs, so a bot can no longer know where its
shot will go.

Separately, five writers produce a target outside the turn's slot:
- the 4c snap (`simultaneous.py:101-106` and `130-132`, fed by
  `combat.py:1186-1187` and the melee strike at `combat.py:1769`);
- the slot-5 zombie walk (`ai_zombie.py:214`);
- the slot-3 travel facing (`simulation.py:2263`);
- WEGO (`simulation.py:2192`);
- onephase (`timeline.py:542` and `845`).

None of them can be "handed to the law" without a stored target, and the
state table has no target column. "After an outside spin it returns and
settles exactly" also has nothing to return to when no aim is held.

*Resolution:*
1. Add the column `facing_target_q` (Q16 rad).
2. EVERY writer listed above becomes `turning.set_target(u, angle_q)`. The
   4c snap becomes a recorded `set_target`.
3. The target PERSISTS until something replaces it. The policy's zero row
   ("facing untouched", `control_policy.py:50-52`) then means "hold the last
   target".
4. Add a new conductor slot **1a**, `unit_impulse.integrate_all(sim)` (one
   call line, per the god-file policy). It runs for every living unit in id
   order:
   - steps 1-6;
   - then the turn, with target = the live aim's `atan2_q16` if a non-zero
     `live_aim` exists, else `facing_target_q`.
5. Slot 1b then only consumes throw, use, reload and shove.
6. `facing_after_turn_q16(u, dx, dy)` becomes the pure evaluation of exactly
   the slot-1a turn, on `(facing_q, omega, imp_l, I_g, a, w)`. It is exact,
   because every producer of `imp_l` (slot 2, 4c, recoil, 9c2, the previous
   tick's shoves) has already run when `handle_frame` runs.
7. The direction rule keeps reading the post-turn facing.

The costs, stated:
- Every impulse is integrated exactly one tick after it lands, so the
  draft's "next tick" sentence becomes uniformly true. As drafted, slot-2
  hits would integrate in the SAME tick and slot-4c hits in the next.
- Travel, zombie and snap facings lag one tick.
- A unit floored at 1a cannot shoot in that tick's slot 4.

Also delete the `live_aim is None -> cos_q = ONE` shortcut
(`simulation.py:2233-2234`): under inertia, a walker holding no aim does not
face its move.

**M7 (MAJOR) -- recording an impulse at the resolve is not one of the four
effect kinds.** `SimultaneousEffects` documents that "the four effect kinds
are the whole interface" (`simultaneous.py:60-72`).
*Resolution:*
- Add a fifth kind, `push(unit, jx_q, jy_q, l_q)`, applied in phase B in
  record order. Amend the Simultaneous attacks row; the residency applier
  gets a matching device function.
- The hit's `push` is recorded right after its `hit`. It lands on a corpse
  too, which is harmless because dead units are not integrated.
- Recoil is an own-row write in phase A (the residency table allows
  "own-row writes immediate"). Record it through `push` anyway, so there is
  one shape.
- The shove at 1b adds immediately: nothing in 1b reads an accumulator, and
  integer sums commute.
- Do not put a clamp or saturation inside `add` on the running sum (M4).

**M8 (MAJOR) -- the effort scale and the reaction-delay walk ban have no
home on the zombie, WEGO and onephase paths, and the effort scale conflicts
with the Footprint samples contract.**
- `footprint_samples(gmap, fy, fx, fp)` takes no unit (`movement.py:74-96`).
- `default_speed` never sees the unit (`movement.py:112-160`).
- The row says a speed output is a Q16 member of `FootprintSamples`.
- The walk ban cannot be a scale: `default_speed` raises on a scale <= 0
  (`movement.py:156-158`).

*Resolution:*
- Add `FootprintSamples.effort_scale_q16`, default `SCALE_ONE`.
  - The builder gains `effort_scale_q16=SCALE_ONE`, passed by every caller in
    `src/`: `_step_move_dir`, `ai_zombie.py:182`, the WEGO order path and the
    onephase cadence.
  - `default_speed` composes it with the SAME skip-at-ONE guard as wind, so
    zero input stays bit-identical.
  - For patch a, where both exist, name the order: wind first, then effort,
    each one `half_up`.
- The walk ban is a GATE beside `can_move`, read through one predicate,
  `unit_impulse.walk_allowed(u)`, at the three walk sites:
  - `_step_move_dir`: the walk part is 0, and pv is still applied;
  - the zombie's `flags.can_move` check (`ai_zombie.py:176`): the
    accumulator pauses;
  - the WEGO path offset.
- The alternative is a STAGGERED status row (can_move False, can_act True).
  It would get the digest, the serializer and the P3a status block for free,
  but `react_ticks` would then have to BE its remaining duration, read
  through a status accessor, never the list.

## 4. Every facing writer (`src/`)

| site | writer | through a target | pinned behaviour that moves |
|---|---|---|---|
| `simulation.py:1126` | 1b aim turn | it IS the law (moves to 1a) | G1p pins, only through `a` |
| `simulation.py:2263` | travel facing with no aim (`face_towards`) | `set_target(atan2(move))` | scripts and WEGO-era tests (bots always aim, `control_bots.py:232-237`) |
| `simulation.py:2192` | WEGO path step | `set_target` | WEGO pins (legacy: a one-line finding) |
| `ai_zombie.py:214` | zombie step | `set_target` | EVERY match with walking zombies: `__unit_facing__` is hashed every tick (`tests/_xarch_perfield_digest.py:624`) |
| `simultaneous.py:106`, `:132` (from `combat.py:1186` and the melee strike `:1769`) | attack snap | recorded `set_target` | WEGO, onephase, Move & Attack; crit arcs and `facing_defense` read the lagged facing |
| `timeline.py:542` | onephase step | `set_target` | onephase pins |
| `timeline.py:845` | onephase overwatch | `set_target` (keep `overwatch_facing` as it is) | onephase pins |
| `unit.py:216` | spawn `_NORTH`, a float | initial `facing_q = target = quantize(pi/2) = 102944` | the hashed facing of every unit that never turned (pi/2 is not dyadic) |
| `unit.py:459` | `face_towards` | becomes `set_target` (keep the name) | all of the above |

No tool, `ui/` or `renderer/` file writes facing. 21 test sites in 11 files
do, so expect fixture churn.

**Representation:** the synced value is `facing_q`, an int in Q16 rad on
`[-PI_Q16, PI_Q16]`. `Unit.facing` becomes a read-only float property,
`facing_q / 65536`, with NO setter, so any missed writer raises at its own
line. Every reader keeps working:
- the pose layer and the renderer;
- `rl_env.py:143`;
- `combat.py:1047`;
- the throw (`simulation.py:1200`);
- flash_blind;
- vision;
- `facing_compass`.

The pose layer reads `facing` and derives travel from the position delta. A
push is then travel without a walk, so the leg layer will step. If the
animation must tell a push from a walk, expose `pv` read-only (render only,
never written). MINOR.

## 5. Digest and recording

**M9 (MAJOR) -- the new state is cross-tick and must be hashed,
absence-transparently.** `imp_*` is NOT cleared within the tick in which it
is received: hits from 4c, recoil, 9c2 wind and shoves all carry into the
next tick. `sw`, `pv`, `omega`, `effort`, `react_ticks` and the target
persist too. `_unit_record` (`tests/field_ab_harness.py:179-217`) is a fixed
dict, and the golden splits it into sub-hashes
(`_xarch_perfield_digest.py:604-624`).
*Resolution:*
- Add keys ONLY when non-zero: `imp` (3 ints), `sw`, `pv`, `omega`,
  `effort`, `react`.
- The target itself is never zero, so it cannot be absence-transparent.
  Hash `turn = wrap(target - facing_q)` instead, only when non-zero.
- Hash `facing` as the integer `facing_q`.
- Add a `__unit_push__` sub-hash label, so a divergence names the impulse
  state.

The magazine precedent ("caught one tick late") does not transfer here.
Effort and react change the walking speed through a `half_up`, which can
hide a divergence for many ticks.

**M10 (MAJOR) -- the shove verb rows.**
- `INTENT_KINDS["shove_intent"] = ("int",)`, recorded by the tap before
  acting (`match_replay.py:92-101`).
- `VERB_RULES["shove_intent"] = ("latch", "flag")`
  (`control_policy.py:78-86`).
- `FORMAT_VERSION` goes from 4 to 5 (`match_replay.py:86`), and v4
  recordings are refused.
- A `pending_shove` latch on the unit.
- `shove` leaves `INERT_ACTIONS` (`control_direct.py:59`) and gains a
  `DeviceFrame` field.
- Zombies cannot shove: 1b skips them (`simulation.py:1109`).

The draft omits two things pack §7 lists for G5p: `last_shove_tick` (a
cooldown, reset at the WEGO round boundary like `last_fire_tick`,
`simulation.py:2281`) and `pending_shove`.

The selection rule needs an integer definition:
- "Adjacent" means a squared centre distance <= `(fp+1)^2` in tiles. Do not
  use the float `math.sqrt` the zombie melee uses (`ai_zombie.py:139`);
  P3e lists those sites as debt.
- "Nearest that facing" means the smallest
  `|wrap(atan2_q16(-dy, dx) - facing_q)|` over the candidates, with ties
  going to the lower id (the slot-5 precedent).
- The shover's facing is read AFTER its own 1a turn.
- With no candidate, nothing happens and the cooldown is not spent.

**B4 (BLOCKER) -- the dormancy gate is false as written.** The gate says
"with every input switched off and unlimited turning acceleration, all pins
unchanged". It fails four ways:
(a) There is no switch for recoil, which is RULED on, and
`bullet_push_mult = 0` does not obviously cover it.
(b) The facing writers that are instant today (zombie, snap, WEGO, onephase,
travel) become capped at `w` and lagged one tick (B3), even at `a = ∞`, and
zombie facing is hashed every tick.
(c) The facing currency alone moves `__unit_facing__` for every unit that
never turned (the spawn pi/2).
(d) The Q16 walk step moves positions (M6).

An implementer who tried to make the gate pass would have to exempt writers
from the law, which bends the design to fit a test.
*Resolution:* re-baseline in TWO deliberate commits, each with its rationale
(iron rule).
- Commit 1, representation only: `facing_q`, the Q16 walk step, and
  `facing_target_q` written instantly (the law at `a >= 2w` and
  `w >= PI_Q16`, with targets applied in their own slot as today). This
  re-pins once.
- Commit 2: the law.
- Replace the gate with G1-G3 in §8, and add `[impulse] recoil_mult`
  (shipped 1.0, the RULED real momentum) beside the two existing
  multipliers.

## 6. Reuse and conflicts with canonical rows

- **Wind load on a body** ("never a displacement"). There is no conflict in
  patch a; patch b amends the row. Patch b's wind DIRECTION must come
  through the seam: extend `GameMap.footprint_wind_load` to return the
  argmax tile's velocity, because the row forbids reading the planes. MINOR,
  patch b.
- **Footprint samples builder.** Covered by M8: one composition site, as a
  member, never a second multiply in `_step_move_dir`.
- **Turn-rate cap.** The row's "Movement-follows-travel stays instant" is
  overruled by §5.1. `facing_after_turn_q16` stays THE predictor and becomes
  the law's pure form (B3). `turn_toward_q16` stays as the `a >= 2w` special
  case, with its property test.
- **Speed table.** "theta = ... THIS tick's facing (after the turn cap)"
  holds only with the 1a placement (B3).
- **Class P twin ledger (M11, MAJOR).**
  - `src/simulation/unit.py` is in the Class-P stamp twin's `files`
    (`tests/twin_manifest.toml:467`; commit 099df043 moved its `level_at`
    for a single default). The new Unit columns therefore require
    `.cuda_green` == HEAD, written by `tools/cuda_green.py` on this
    checkout's own CUDA build, plus `twin_debt.py --mark-level` in the last
    commit.
  - The new module `src/simulation/unit_impulse.py` is run by the ship_demo
    corpus, so it needs a `[not_twinned]` row ("Class T: slot 1a / 3, built
    from P6"), or `test_twin_debt` goes red.
  - The reason text in the `turning.py` row changes. Regenerate the
    manifest.
- **Status rows.** The law's KNOCKED_DOWN at 1a goes through `apply_status`
  with `knockdown_getup_ticks` (as at `exchange.py:616`), refresh-stacked.
  The draft does not say what entering and leaving prone does to `sw`,
  `effort`, `react`, `omega` and the turn. Today a KNOCKED_DOWN unit still
  turns to its aim, because 1b checks no flag. MINOR, unspecified.
- **Death and conversion (M12, MAJOR).** `convert_marines_to_zombies`
  (`ai_zombie.py:227-246`) revives a corpse. At death and at conversion,
  zero all impulse state, keep `facing_q`, and set target = facing.
  Otherwise a converted zombie inherits a stale push. The same applies at
  `_end_round`.

## 7. Port-readiness

Per unit (P3a's input; this replaces pack §7's "pvx, pvy (+ whatever)" row):

| column | type | default |
|---|---|---|
| `imp_x`, `imp_y` | int64, Q16 g·tile/tick, the remainder kept | 0 |
| `imp_l` | int64, Q16 g·tile²/tick | 0 |
| `swx`, `swy`, `pvx`, `pvy` | int32, Q16 tiles/tick, `|·| <= V_MAX` | 0 |
| `omega` | int32, Q16 rad/tick | 0 |
| `facing_q`, `facing_target_q` | int32, Q16 rad | `quantize(pi/2)` |
| `effort` | int32, Q16, `[0, 2·ONE]` | 0 |
| `react_ticks` | int16 | 0 |
| `last_shove_tick` | int32 | -1 |
| `pending_shove` | latch bit | clear |
| `m_g` (mass + kit) | int32 grams | from species + loadout |

`I_g`, `r_body`, the limits, `a` and `w` are table values (per species or
profile row), not columns.

Device forms, all one thread per unit:
- The integration: `isqrt` is a fixed 32 trips, `s(e)` is at most
  `ceil(w/a)` trips (bounded at the door), and the effort step is a shift.
- The substep move: at most `ceil(V_MAX + v_full)` trips, with
  `is_passable_block` reading the device tile planes.
- The shove selection: wide, reading the start-of-slot positions.

The accumulation follows "phase A wide, the applier applies". Because it is
an integer sum, an int64 `atomicAdd` gives the identical result in any order,
which is cheaper than the serial applier and still exact. There is no draw.
Once V_MAX and the `ceil(w/a)` bound exist, nothing in the law lacks a
device form.

## 8. Gates to add or reword

- **G1 (replaces the dormancy gate).** With `bullet_push_mult = recoil_mult
  = 0` and no shove issued, `imp`, `sw`, `pv`, `effort`, `react` and the
  impulse part of `omega` are EXACTLY 0 on EVERY tick of every pinned match,
  asserted per tick, which makes the gate non-vacuous. It breaks if any
  input bypasses the dials.
- **G2.** At `a >= 2w`, the turn equals `turn_toward_q16` bit for bit over a
  property sweep of (facing, target, w). It breaks if the braking law leaks
  into the instant limit.
- **G3.** At `a >= 2w`, `w >= PI_Q16` and every input off, the pinned
  matches equal commit 1's re-pinned digests. It breaks if the law changes
  anything its dials do not control.
- **G4 (momentum).** The B1 / M3 identity in int64, per unit per tick,
  including `wall`, `clamped` and `Δresidual`, on a scene with walls, a
  prone slide and a V_MAX clamp. It breaks if any term is unbooked.
- **G5 (in phase / out of phase).** The final position is EXACTLY the same
  when no overflow occurs. For the band, compare the mean effort over a
  whole period, within one count per tick of period; the per-tick effort
  sawtooth legitimately differs. Do not write "the same band" as an exact
  per-tick equality: it is not achievable.
- **G6 (torque sign).** A round passing right of the target's centre, as the
  shooter sees it, turns the target clockwise in the facing frame. A
  mirrored scene gives the mirrored facing, to the count.
- **G7 (permutation).** Re-run a match with the unit list order (the ids)
  permuted. Every unit's impulse state and position, keyed by identity, is
  equal. That is the testable form of "changes nothing".
- **G8 (walls).** At V_MAX, in each of the 8 directions, into a wall, a
  corner and a one-tile gap: the unit never ends inside a wall and never
  crosses one. At pv = 0 the move equals today's single-test slide.
- **G9 (predictor).** Under hits and shoves, `facing_after_turn_q16` read at
  `handle_frame` equals the facing slot 4 fires along, every tick. It breaks
  if the turn moves after a producer.
- **G10 (writers).** `facing` has no setter, and a scan of `src/` finds no
  write to `_facing_q` outside `turning.py` (the footprint-samples scan
  shape).
- **G11 (grid).** `u.x·65536` and `u.y·65536` are integral after every tick
  (M6).
- **G12 (zombie).** A zombie pushed off its path never crosses a wall on its
  next steps.
- **G13 (conversion).** A unit killed with pending impulses converts with
  zero impulse state.
- **G14 (replay).** A match with shoves records at format v5 and replays
  through every checkpoint; a v4 file is refused.
- **G15 (twins).** `unit_fixed.isqrt` equals `sqrt_q16`, and `div_trunc`
  equals C `/`, on a sample that includes negatives.

MINOR items not covered above:
- m1. §4's "a hit is integrated in the NEXT tick" is false for slot-2
  rounds as drafted. B3 fixes it.
- m2. Which archetypes push is unspecified. The 40 mm (damage 0, a payload)
  and the beams also stop on units.
- m3. `t_react` rounds 7.2 to 7: name the rounding.
- m4. "Exactly today's 245 Pa" can only mean "within one count" after
  quantization.
- m5. Step 6c, `|pv| > n_rec·margin`, floors any `pv > 0` when `margin` is 0.
  That edge is decided by the floor in `isqrt`; state it.
- m6. The bounded `s(e)` trip count belongs in the door.
- m7. Patch b's wind `q·cd_area·dt` takes `cd_area` in tiles², through the
  same per-map table.
- m8. A render-only `ShoveEvent` or stagger event for #184 (the
  `ReloadEvent` precedent), if the animation is to react.

## Implementer's sheet

**Representation:** see the §7 table. The currencies are Q16 tiles/tick,
Q16 g·tile/tick and Q16 rad. Every real-unit dial is quantized ONCE, in
`ImpulseTable`, built at Simulation construction with the map's tile size,
beside `speed_table`.

**Where things live:**
- `src/simulation/unit_impulse.py`: `add`, `integrate_all` (slot 1a),
  `move` (slot 3), `walk_allowed`, and the door and table.
- `src/simulation/turning.py`: `set_target`; `step` (the §5.1 law, called
  only from `integrate_all`); `predict` (pure, behind
  `facing_after_turn_q16`); `turn_toward_q16` kept.
- `simultaneous.py`: the `push` and `set_target` kinds.
- `movement.py`: the `effort_scale_q16` member and the Q16 walk-speed twin.
- `unit_fixed.py`: `isqrt` and `div_trunc`.
- `tests/field_ab_harness.py`: the absence-transparent keys.
- `match_replay`, `control_policy`, `control_direct`: the shove rows.

**Order of a tick:**
- 1: clear the events.
- **1a, NEW:** for each living unit, in id order:
  1. dv and dw from the accumulators, with the remainder kept;
  2. demand, then effort;
  3. sway, with its overflow into pv (V_MAX clamp);
  4. resist;
  5. the knockdown checks (`apply_status`);
  6. omega += dw;
  7. turn toward the live aim, or else `facing_target_q`.
- 1b: throw, use, reload; the shove, which adds to its target's
  accumulator.
- 2: grenades, then rounds. Phase A records `hit` + `push`; then resolve.
- 2b: statuses.
- **3:** one loop over the living units, in id order. Each calls
  `move(walk if MOVE_DIR and walk_allowed and can_move, else 0)`, which adds
  pv, runs the Q16 substeps and books `wall`. Travel sets the target. A
  zombie that was displaced drops its path. WEGO and onephase follow, and
  they overwrite (an accepted gap).
- 4 / 4b: phase A records the hits, the pushes, the recoil, and the snap as
  a `set_target`. 4c resolves.
- 5: the zombie walk (the effort member; `walk_allowed`). Its step sets the
  target.
- 6: stamp. 7: physics. 9c2: wind (patch a: the old row; patch b: `add`).
- 10: tick end.
