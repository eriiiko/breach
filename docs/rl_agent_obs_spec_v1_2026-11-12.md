# Observation spec v1 and the full view (rung 1, patch 1g)

2026-11-12. The contract for the three implementation patches of 1g (#195),
written from the plan (`docs/rl_agent_rung1_plan_2026-11-11.md` section 5,
"1g") and the design (`docs/rl_agent_design_2026-11-11.md`, "the design",
sections 3, 4, 4a, 5.2, 7, 9, 10), against the code at main `ee25fe98`
(1a to 1f merged). The design is agreed and critiqued and is not re-derived
here; the critique's information lens
(`docs/rl_agent_design_2026-11-11_critique/lens_2_information.md`) is taken as
done. **CHOICE:** marks where the design is silent and the simplest honest
thing was chosen. **STOP:** would mark a design / code contradiction that
cannot be resolved without changing one of them (there is none; section 9).

What is fixed here: every feature as a row (position, name, the seam it is
read from, the table value that divides it, the squash, the stored type); the
list's caps, order, padding and mask; the planes of both windows; the layout
in the env's buffers and the two row flags; the value network's full view;
the cost budget; the tests; the cut into three patches. Everything a trained
network reads is a ratio of a table value, an angle's cosine and sine, a flag,
or a kind (a small integer the network embeds); never a game number.

## 0. Conventions

### 0.1 Frame, clock, reference values

- **Frame:** north up, centred on the soldier, the map's own axes (design 1).
  An offset is `(dx, dy)` in tiles from the soldier's centre tile
  (`Unit.center_tile_x/y`, `src/simulation/unit.py:601-607`), x east, y
  south. A facing is `(cos, sin)` of the sim's math-style angle
  (`Unit.facing_q`, Q16.16 radians, 0 = east, counter-clockwise; `unit.py:521`),
  taken through the kit (`unit_fixed.cos_q16 / sin_q16`, divided by 65536):
  the y-down conversion is the reader's, as in `team_picture.fire_lanes`
  (`src/simulation/team_picture.py:251-252`).
- **Decision:** the farm's step (`[run] decimation` ticks; the design's 4). The
  observation is built once per step per soldier at the step's end, from the
  sim as the next tick's controllers will see it (`EnvRunner.observe`,
  `src/rl_env.py:629`, after `MatchSession.tick`).
- **The reference length `L_ref`** (CHOICE): the `[marine] loadout` row's
  weapon `range_tiles` -- the default kit's reach on this map
  (`[marine] loadout` `config.toml:4824` -> `[loadouts.rifleman] weapon`
  `:4909` -> `[weapons.k5_carbine] range_m` `:3297`, derived once per map by
  `WeaponTable` as `range_tiles`, `src/simulation/weapons.py:122-140`; 270
  tiles on the 0.333 m tile). Every offset, distance, route length and blast
  radius is divided by it. One value per run, read once at `Layout` build.
- **The reference speed `v_ref`** (CHOICE): the marine's full speed on open
  floor in Q16 tiles per tick, `(ONE * ONE) // SpeedTable.full_cost_q`
  (`src/simulation/speeds.py:82`, from `[speeds] marine_full_mps`
  `config.toml:109`). Every velocity is divided by it.
- **The reference kit**: the `[marine] loadout` row's weapon: its
  `rof_interval_ticks` (`weapons.py:143`) and `spread_deg` (`:120`) divide
  every unit's own.
- **The compressed distance** `d_c` (design 3.1 "given again on a compressed
  scale"; CHOICE): `ln(1 + d) / ln(1 + L_ref)`, `d` in tiles: 0.12 at one tile,
  0.43 at ten, 0.70 at fifty, 1.0 at `L_ref`; capped by the squash above.
- **Units of the match**: the roster size `n_t` of a marine team is its
  match row's `n` (`MatchTeam.n`, `src/simulation/match.py:92`); the horde's
  `spec.zombies` (`:103`); the time limit `limit_ticks =
  MatchRules.limit_ticks(sim)` (`src/simulation/ruleset.py:331`).

### 0.2 The squash (design 3.1, the critique's one fixed curve)

Every ratio passes `squash` before it is stored; nothing else is applied, no
running statistic exists.

```
squash(x) = x                              for -1 <= x <= 1
          = sign(x) * (1 + ln|x|)          for |x| > 1, capped at +-4
```

Continuous and monotone; the identity on `[-1, 1]`; 2 at `e` times the
table value, 4 at about twenty times, and capped there (three burning tiles
read at most 4). Angles, flags and kinds are not squashed. CHOICE: the
natural log and the cap 4.

### 0.3 Stored types

| what | type | why |
|---|---|---|
| a flag, a plane of flags, a count plane, a code plane | `uint8` | design 3.4 "yes / no planes as single bytes" |
| a ratio plane | `float16` | design 3.4; the squash bounds it in `[-4, 4]`, where float16 keeps three decimals |
| the soldier's vector, a list entry | `float32` | small; a kind column holds its integer exactly |
| an entry's game id, a row flag | `int32`, `uint8` | the decoder's (1h) and the mask's; never a network input |

A kind (class, weapon, grenade kind, entry type, cause, side, order kind)
is stored as its row index in the named table, `-1` = none; the network
embeds it (design 3.1). The tables are named per row below.

### 0.4 Where the observation is built

- `src/simulation/obs_spec_v1.py` (NEW, patch (a)): the spec as code -- the
  feature tables of sections 1 to 4 as data (name, position, reader,
  denominator, squash, type), the builders, and `SPEC_NAME = "v1"`. Pure host
  Python over the sim, the team picture and the trackers; imports no
  controller and nothing from `renderer/`.
- Served through the sensor accessor, the one observation door (plan 5):
  `EntityFieldAccessor.observe(spec, rows, out)` (NEW method beside `crop`,
  `src/simulation/sensor_accessor.py:271`) dispatches by spec name; `crop`
  stays as the placeholder spec `"crop"`. The env calls only the accessor.
- The run file selects it: `[obs] spec = "v1"` (NEW key of `_OBS_KEYS`,
  `src/rl_env.py:235`; default `"crop"`, today's rows unchanged). Under
  `"v1"` the three `"crop"` keys (`channels`, `crop_radius`, `vector`) are
  refused by name, and `load_levels` (`rl_env.py:454`) refuses a level whose
  `LevelData.rooms` is `None` or not at the map's shape
  (`unit_orders.graph_of`, `src/simulation/unit_orders.py:57`): "a run whose
  observation spec reads the room graph refuses a level without one".
- **The trackers** (design 9: memory updated every tick, not unit state,
  not synced, rebuilt by re-simulation) are members of
  `src/simulation/team_picture.py` on the same driver, slot 11
  (`Simulation._step_post_physics`, `src/simulation/simulation.py:2426`; the
  1d note's "Not in 1d"): see 1.6.

<!-- continued -->
