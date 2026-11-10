# Gameplay pack v1 — design (v1 draft, 2026-11-10)

Issue **#183**. Held-back features: **#182**. Sources: Erik's
`game_feature_ideas_20261109.md`, `prototypes/ideas_patches_20261109_controls.md`,
issue #181, and the session of 2026-11-10.

Status: DRAFT for Erik. Sections marked **RULED** record what he has decided;
**PROPOSED** is Claude's proposal and needs his yes; **OPEN** needs a design
session. §5 (the unit impulse) is OPEN and waits for its literature search.

## 0. Summary

Eight patches that make a marine a richer thing to play and to train, landed
before the residency port's gameplay half (#29 P3a onward):

| # | patch | sim change | new verb | moves the bots' pinned matches |
|---|---|---|---|---|
| G1p | keyboard + mouse, continuous speed, turn-rate cap | the speed argument, the turn cap | (`set_move_dir`'s speed becomes a throttle) | yes (the cap) |
| G2p | reload and magazines | yes | `reload_intent` | yes (the carbine gets a magazine) |
| G3p | grenade kinds; the flashbang blinds | yes | (`throw_grenade_intent` gains a kind; fuse in ticks) | no |
| G4p | focus-fire armour debuff (#181) | yes | none | yes |
| G5p | unit impulse: shove and hit pushback | yes | `shove_intent` | yes (hits push) |
| G6p | melee | yes | `melee_intent` | no |
| G7p | flashlight on/off | one bit | `set_flashlight` | no |
| G8p | bot pass | no (a controller) | none | yes |

(The `p` suffix keeps these apart from the residency arc's gates G1 to G3.)

## 1. Why now

- **One port instead of two.** After #29's P6 / P7, a new verb or a new branch
  in a law needs a device twin, and the device backend refuses a training run
  that uses it until the twin exists (residency design §10). Before P3a each
  of these is ported once, with the rest.
- **The action space is settled before training.** Reload, grenade kind,
  shove, melee and the flashlight are action columns; adding one later
  invalidates trained policies.
- **Independent of G1.** G1's scene has no units. If G1 stops the device
  route, this pack still stands on the CPU farm.

## 2. Rules every patch of the pack follows

These keep the pack from adding work to the port (residency design §7, §10).

1. **Integer law.** New per-tick arithmetic on the match path is Q16 through
   the kits (`fixed_point.h` twins, `unit_fixed`, `atan2_q16` / `sin_q16` /
   `cos_q16`). No new float site is added to P3e's table.
2. **No draws.** None of the pack's laws draws a random number. A law that
   turns out to need one says so in its patch and names its Philox site for
   P3b.
3. **Fixed capacity.** Every new per-unit state is a scalar or a fixed-width
   block. The pack's whole list is §7, which is P3a's input.
4. **One tap.** A new or changed verb is a row of `match_replay.INTENT_KINDS`,
   records itself through the tap, and has a `control_policy.VERB_RULES` row
   in the same patch. A changed row bumps `FORMAT_VERSION`; old recordings
   are refused, not migrated.
5. **Start-of-slot decisions.** A law that reads state other attackers write
   in the same slot (the debuff, the impulse) reads the START state and lands
   through `SimultaneousEffects` (#169), so no unit's outcome depends on its
   id.
6. **Dormant when unused.** A patch whose verbs no shipped controller uses
   leaves every pinned match digest unchanged. That is its gate. The patches
   that do move the pins (table above) re-baseline once each, with the
   rationale written.
7. **HUMAN-TEST.** G1p to G6p and G8p are feel changes: built, gated, pushed,
   Erik plays before merge.

## 3. The patches

### G1p. Keyboard + mouse, two cameras, continuous speed

**Exists.** Move and aim are separate intents, so strafing is already in the
sim; `GamepadDirect` (`--control gamepad`) drives one marine with them. There
is no keyboard + mouse scheme for direct control.

**PROPOSED.**
- One direct-control scheme, `--control direct`, that reads the pad AND the
  keyboard + mouse every frame and takes whichever was touched last (Erik:
  "the player plays with whatever they have connected"). `GamepadDirect`
  becomes its pad half; `--control gamepad` stays as an alias.
- Keys are DATA: a `[controls.keys]` table (action → key), so WASD and RDFG
  are two tables swapped without code. Shipped default **RULED (Erik,
  2026-11-10)**: R / F forward / back, D / G strafe left / right, mouse aim,
  left button trigger, right button melee, Space shove, Z reload, W throws
  the selected grenade, 4 cycles the grenade kind (Counter-Strike's grenade
  slot: each press steps to the next kind carried), Q flashlight, E use, A
  full speed. Melee, shove, reload, the kind and the flashlight are bound in
  the table now and do nothing until their patches land.
- Two camera modes, `[display] camera_mode`, switched live:
  - `"centred"`: north-up, the marine in the centre, the marine turns
    (Diablo / Dota style);
  - `"player_up"`: the view rotates so the marine's facing is up, the marine
    low on the screen (FPS style). This needs the world drawn rotated and the
    mouse mapped back through the rotation: a **one-day spike, slotted AFTER
    the gameplay patches (RULED)**, as G1p-b; G1p ships `"centred"` only. In this mode the mouse turns the marine by its
    horizontal motion; there is no cursor to aim at.
- **Continuous speed (RULED in principle: "continuous, not two states").**
  `set_move_dir`'s fourth argument becomes `throttle_q`, Q16 in [0, ONE]: the
  speed is `v_walk + throttle · (v_sprint − v_walk)` in tiles per second,
  from the two existing cadence dials. The stick's deflection gives it on the
  pad; on the keyboard Shift is ONE and unshifted is 0. Bots send 0 (today's
  walk), so their matches do not move.

- **Turn-rate cap (RULED 2026-11-10: aim is not immediate).** `set_aim`
  names the facing a unit WANTS; each tick its facing turns toward it by at
  most `turn_rate` (a Q16 angle per tick, the shorter way round, integer
  through `atan2_q16`). Shipped PLACEHOLDER: **360° per second** (15° per
  tick; an about-face takes 0.5 s, a quarter turn 0.25 s), which is a fast
  trained pivot with a rifle and slow enough to feel. It is read per unit
  (`[marine] turn_rate_deg_per_s` now; later a stat scaled by dexterity and
  armour weight, Erik's rules to come). A shot still leaves along the
  CURRENT facing, so a unit that fires mid-turn sprays the arc; the bots
  hold the trigger until their facing is within the weapon's spread of the
  aim. G5p replaces the bare cap with angular inertia (§5); the cap is that
  law's top speed.

**Sim change:** the throttle and the turn-rate cap. **Gate:** throttle 0 and
ONE reproduce today's walk and sprint bit for bit; a turn never exceeds the
cap in a tick, always takes the shorter way, and reaches the aim exactly;
with the cap set to a half turn per tick every pin is unchanged (the cap is
the only thing that moves them); the quantization test extended to the mouse
vector. The pins re-baseline at the shipped cap. **HUMAN-TEST:** both
devices.

**OPEN (small):** is sprint unlimited (today) or does it cost something? Not
in this pack unless Erik says so.

### G2p. Reload and magazines

**Exists.** `combat.mag_gate` / `mag_spend`: a tracked magazine, an
auto-reload stall when it empties, infinite spares. The carbine has it
switched off (`mag_size = 0`).

**RULED.** The carbine: 30 rounds, 3 magazines at spawn (one in the gun, two
spare). A reload discards the magazine in the gun with every round in it.
Hand-written bots only auto-reload; the verb is in the action space so an RL
agent reloads when it chooses.

**PROPOSED.**
- `[weapons.*]` rows keep `mag_size` / `reload_seconds`; `[loadouts.*]` gains
  `magazines` (total, including the one loaded). `k5_carbine`: `mag_size =
  30`, `reload_seconds` 2.0 (RULED as a placeholder).
- **The carbine's rate: 12 rounds per second (RULED 2026-11-10)**: one round
  every 2 ticks (`shots_per_trigger = 1`, `rof_interval_seconds = 1/12`),
  from today's 5 rounds every 4 ticks (30 per second). A magazine is 2.5 s
  of fire.
- Unit state: `current_mag`, `spare_mags`, `reload_done_tick`.
- `reload_intent(unit)` (a latch): if a spare exists, the unit is not already
  reloading and the magazine is not full, the current magazine is discarded,
  `spare_mags −= 1`, and the stall runs; the new magazine is in the gun when
  the stall ends. Otherwise nothing happens.
- Emptying the magazine starts the same reload automatically if a spare
  exists. With no spare the trigger does nothing.
- A reload is not interrupted and does not stop movement (v1).
- Weapons other than the carbine keep infinite spares until they get a
  `magazines` value (`magazines = 0` = untracked spares), so nothing else
  moves.

**Gate (properties):** rounds fired never exceed `mag_size × magazines`; a
reload at k rounds left loses exactly k rounds; no shot inside the stall; a
unit with `mag_size = 0` touches none of the state. The pins re-baseline
(bots now stall and can run dry). **HUMAN-TEST.**

**Note.** 4 marines × 90 rounds against Erik's "about one magazine per
zombie" makes ammunition the match's clock. That is intended (#181) and is
tuned in G4p, not here.

**As built (2026-11-10).** `combat.mag_bind` / `mag_settle` / `mag_gate` /
`reload_begin` / `mag_spend` / `mag_status` are the whole law;
`Simulation.reload_intent` latches `pending_reload`, `_reload` consumes it in
`_consume_direct_intents`. State: `current_mag` (None = unbound, bound lazily
at the first trigger), `spare_mags` (`SPARES_UNTRACKED` = -1 = infinite),
`reload_done_tick` (-1 = none pending; a reload completes lazily at the next
gate or settle). `Unit.magazines` is the loadout's count, applied only to the
loadout's own weapon (`magazines_carried`): a swapped-in weapon keeps
untracked spares. Match format v3. Direct control: Z, pad Y. The HUD panel is
`ui.model.panel_state_lines`. The pinned ship_demo match moved by the rate
alone (win 1, 722 -> 700 ticks): no marine in it fires more than 12 rounds, so
no gun empties; the magazines are first felt in a longer match.

### G3p. Grenade kinds; the flashbang blinds

**Exists.** `[ammo.grenade_frag]`, `grenade_smoke`, `grenade_tear`,
`grenade_poison`, `grenade_incendiary` and their payload rows; a
`[payloads.flashbang]` row that is light only; the BLINDED status (teargas
applies it; a blinded unit cannot aim, its fire falls to the snap cone). The
throw verb takes no kind and the loadout has one grenade count.

**RULED.** Frag, smoke and flashbang are in v1. The flashbang's effect
depends on facing.

**PROPOSED.**
- `[loadouts.*] grenades` becomes a table by `[ammo.*]` name
  (`{ grenade_frag = 2, grenade_smoke = 1, grenade_flash = 1 }`; a bare int
  stays legal and means frags). Unit state: one count per hand-grenade row
  (a fixed block; the rows are the capacity).
- `throw_grenade_intent(unit, dx, dy, fuse_ticks, reach_q, kind)`: `kind` is
  the hand-grenade row's index; the fuse becomes INTEGER ticks at the door
  (the residency design's P3e planned exactly this change; it lands here
  because the row changes anyway).
- The flashbang: `[ammo.grenade_flash]` → `[payloads.flashbang]`, which gains
  `blind_radius` (tiles), `blind_seconds`, `blind_away_frac`. At detonation,
  every living unit of ANY team within the radius whose `has_los` to the
  detonation tile holds is BLINDED for `blind_seconds × f × g`, where `g` falls linearly with
  distance from 1 at the flash to 0 at the radius (RULED: direction,
  distance and line of sight; any team), and `f` falls
  from 1 (facing the flash, within ±60°) linearly to `blind_away_frac`
  (PLACEHOLDER 0.25) when facing straight away. The angle is the integer
  difference of `atan2_q16(to the flash)` and the unit's facing. A wall
  protects fully (no line of sight). Looking away is therefore a thing a
  team does on a called throw.
- Smoke: selection only. Whether smoke hides a unit from the sight oracle is
  left as it is (Erik: it can wait).

**Gate:** a unit behind a wall is never blinded; duration is monotone in the
facing angle; frag-only loadouts and the pins unchanged. **HUMAN-TEST.**

**ACCEPTED GAP (proposed):** the blinded PLAYER's screen is not whited out
and a blinded agent's observation is not masked in v1; the status only
removes aim. Both are later work with their own design (the obs door is
P8b's).

### G4p. Focus-fire armour debuff (#181)

**RULED (Erik, #181).** Hits lower a target's defences; the lowering decays;
several shooters lower it further, with no record of who shot; a lone marine
needs about 30 to 40 rounds for a zombie, two together about 20 in all; the
mechanic is for every unit, not only zombies; armour types may resist it and
ammunition may drive it differently.

**Damage types exist already** (`damage.py`): kinetic (bullets, melee),
blast, heat (fire and radiant), energy (beams), poison, asphyxiation, heal,
and two reserved (electric, psy); every unit profile has an `armor` and a
`resist_mult` per type, and the zombie's weakness to fire IS a resistance
(`resist_mult[HEAT] = 4.0`, i.e. four times the damage). The set is a table
and can grow (a `piercing` split from kinetic, say) while this matures.

**PROPOSED law.** A per-unit EXPOSURE `x[T]` (Q16, ≥ 0), one level per
damage type T (a fixed block; RULED: each ammunition has its own debuff):

```
on each hit of type T that reaches the unit:   x[T] += ammo.expose · (ONE − profile.expose_resist[T])   (capped at x_max)
each tick:                                     x[T] −= max(1, x[T] · decay)                             (to 0)
damage of type T taken:                        amount' = mitigate(amount) · (ONE + k_expose · x[T])
```

A round with `expose = 0` (fire, PROPOSED) debuffs nothing. A round that
should soften a target for ANOTHER type (an armour-stripping round) is a
later column, `expose_type`.

- `ammo.expose` is a `[ammo.*]` column (0 = this round does not expose);
  `expose_resist` is an `[env_profiles.*]` / species column (1 = immune: the
  heavy tank against light rounds). `decay`, `k_expose`, `x_max` are
  `[damage.exposure]` dials.
- Within one tick every hit reads the START-of-slot `x` and the additions
  land together at the resolve (rule 5), so two shooters in one tick are
  symmetric.
- Exposure is not a status row: a status carries a duration and a fixed
  magnitude, this is a decaying level. It is one column and one term in
  `damage.mitigate`'s caller.
- In v1 only the bullet rows carry an `expose` (RULED); the zombie's weakness
  to fire stays its low resistance to heat.

**To settle with the tuning: the carbine's rate of fire.** Today it fires 5
rounds per trigger every 1/6 s, i.e. 30 rounds per second, so a 30-round
magazine is ONE SECOND of fire and three magazines are three. A real carbine
is about 13 rounds per second. The rate, the magazine and the zombie's
vitality are tuned together on the bench below.

**Tuning is Erik's, with an instrument.** A headless bench
(`tools/bench_focus_fire.py`): n shooters at the carbine's rate on one
zombie → rounds and seconds to kill, for n = 1..4 and for single shots. The
zombie's vitality and the three dials are set from it to hit 30–40 / ~20.

**Gate:** with `k_expose = 0` every pin is unchanged; rounds-to-kill is
strictly decreasing in the number of simultaneous shooters; `x` returns to 0
without hits. The pins re-baseline at the shipped dials. **HUMAN-TEST.**

### G5p. Unit impulse: shove and hit pushback — see §5 (OPEN)

### G6p. Melee

**Exists.** The `melee` weapon archetype (`combat.melee_strike`, the
adjacency predicate `melee_adjacent`, the `combat_knife` and `arc_baton`
rows), reachable only through the old order path. The zombie bite is its own
law.

**PROPOSED.**
- `[loadouts.*] melee` names a melee `[weapons.*]` row; the rifleman gets a
  new `rifle_butt` row (PLACEHOLDER damage, a short STUNNED, and through G5p
  a small impulse).
- `melee_intent(unit)` (a latch): strike the adjacent enemy nearest the
  facing direction, within a facing arc, at the row's own cadence; decided
  in slot 4's phase A and applied at 4c with the shots (rule 5). It hits
  enemies only.
- A melee kill is credited through the existing `melee` cause.

**Gate:** no strike without adjacency or outside the arc; the cadence holds;
the pins unchanged (no controller uses it until G8p). **HUMAN-TEST.**

### G7p. Flashlight on/off

**Exists.** Flashlights are render state (the selected unit's, a mode key).
Light as something the sim and agents read is #81.

**PROPOSED.** `Unit.flashlight_on` (one synced bit, default ON) and
`set_flashlight(unit, on)` (a hold verb). In direct control and in watched
matches the frame-light assembly draws a unit's cone from that bit. Nothing
in the sim reads it until #81. **Gate:** the pins unchanged; the cone
follows the bit.

Erik is unsure this is worth it before #81; it is the cheapest patch of the
pack and puts the column in the action space now, so the recommendation is
to keep it. It can be dropped without touching anything else.

### G8p. Bot pass

**Exists.** `control_bots.BotTeam`: four rules per unit per tick (shoot the
nearest seen enemy, frag a group, fall back when hurt and alone, advance by
A*), a shared last-seen position, no other coordination. Known: stacking and
no door-opening (#142), judder (#138).

**RULED.** The goal is the best old-school AI we can give the RL agents as
an opponent and a baseline; nothing specific bothers Erik today.

**PROPOSED scope (its own short design when the pack's verbs exist):**
- the new verbs: flashbang before entering a room with a known enemy, smoke
  to cross an open line, shove or melee when rushed, aimed spare-magazine
  use stays automatic (RULED);
- **a team blackboard**: target assignment so a team focuses one zombie
  (G4p and G5p pay for it), spacing so units do not stack, a shared plan to
  enter a room from two doors;
- the rules ranked by a utility score instead of a fixed order, so a new
  behaviour is a scored rule;
- #142's door rule and #138's judder.

Bots stay Python on the host: residency ruling §12.3 leaves how bots run
beside device envs to G2, so a richer bot adds no port work now.

## 4. The action row after the pack

Generated from `INTENT_KINDS`, as today (residency design §4.5):

| column(s) | kind | class |
|---|---|---|
| move dx, dy, throttle; clear flag | `set_move_dir` / `clear_move_dir` | hold |
| aim dx, dy | `set_aim` | hold |
| trigger | `set_trigger` | hold |
| throw dx, dy, fuse ticks, reach, kind | `throw_grenade_intent` | latch |
| use | `use_intent` | latch |
| reload | `reload_intent` | latch |
| shove | `shove_intent` | latch |
| melee | `melee_intent` | latch |
| flashlight | `set_flashlight` | hold |

## 5. The unit impulse (OPEN: design session with Erik + literature search)

**What Erik asked for.** A shove based on inertia: the pushed unit's mass and
the push's strength, and it should ACCELERATE the pushed unit. Hits should
push a target back, or at least slow it, more so under focused fire. The
same mechanic should later serve wind and shock waves. Forces should act
realistically; presented area against mass matters; every unit has its own
mass (drawn from a distribution later).

**What exists.**
- `Unit.mass` (kg) and `strength`, from the species distribution (mean 80 kg,
  sd 10; today every unit gets the mean, deterministically). The individual
  mass Erik wants is already a field.
- The wind law (#140, `docs/wind_on_units_design_2026-10-31.md`): dynamic
  pressure `q = ½ρv²` over the footprint; below `q_slow` nothing, a band
  that only slows walking, `q_fall` knocks down; **nothing displaces a
  unit**. That last clause is Erik's ruling of 2026-10-31, made because the
  earlier push law slid a squad out of a door for 35 s on a weak, lasting
  force. Its research note already carries a person's drag area
  (`C_d·A ≈ 0.75 m²`, 0.9 with kit) and the toppling and sliding limits.
- No unit has a velocity: a unit advances along its move direction at a
  cadence, collision-checked per tick.

**Erik's additions and corrections (2026-11-10).**
- **Rotational momentum.** Units also carry ANGULAR momentum: a push or hit
  off the body's centre turns the unit (torque over moment of inertia), so a
  hit on the shoulder twists an aim. Open for the session: whether a unit's
  OWN turning also has inertia (a turn rate from mass and what it carries),
  or only external pushes spin it while aiming stays immediate.
- **The wind wording corrected.** The ruling is not "wind never displaces a
  unit". It is: SOFT wind does not displace a unit (the old law pushed on the
  smallest wind and made units vibrate); below a threshold it only slows.
  Above the threshold a force may move a unit.
- **One law.** The present wind law is acceptable, but the goal is ONE law
  that also gives the knockdown from explosions: wind, blast, shove and hits
  are four inputs of it. So wind and blast are IN this design, not later;
  whether they switch over in G5p or in a follow-up patch is a sizing
  question for the session. The CLAUDE.md wind row is amended when they do.
- **Bullets are enough as they are.** Erik is right that the real momentum
  adds up. Per round it is small (3.6 N·s; 0.04 m/s on 90 kg), but sustained
  fire is a FORCE: rounds per second × 3.6 N·s. Against the wind note's own
  thresholds for a person (felt and slowed at about 29 N, hard to walk at 88
  N, blown over at 183 N):

  | fire on one target | mean force | band on the wind law's scale |
  |---|---|---|
  | one shooter, 13 rounds/s (a real carbine) | 47 N | slowed |
  | two shooters | 94 N | hard to walk |
  | four shooters | 187 N | falls |
  | one shooter at today's 30 rounds/s | 108 N | hard to walk |

  So the physical impulse, through the SAME footing law as wind, already
  gives "one shooter slows it, a team stops it or floors it" with no
  inflated dial. Claude's earlier "a game rule, not physics" was too strong:
  it holds for a single round, not for automatic fire. Two honest caveats:
  a round that passes through delivers only part of its momentum, and by the
  same law the SHOOTER takes the equal recoil (which the law can carry, as a
  braced force below the footing limit).

**The frame Claude proposes to design inside** (to be confirmed against the
literature search, `docs/research/impulse_on_a_person_2026-11-10.md`, in
progress):

1. Each unit gets a PUSHED VELOCITY `(pvx, pvy)`, Q16 tiles per tick, zero
   nearly always. An impulse `J` adds `J / mass` to it.
2. A unit RESISTS. Below a footing / balance limit a force moves nobody: this
   is #140's ruling kept as a law (static friction and active balance), so a
   weak lasting force can never again slide a squad. Between that limit and
   a fall the unit is STAGGERED: displaced by the pushed velocity, which
   decays quickly, its own walking slowed while it lasts. Past a higher
   limit it is KNOCKED_DOWN (the existing status) and slides to rest.
3. The pushed velocity moves the unit through the SAME collision predicate
   walking uses; a wall stops it.
4. Inputs: the shove (`shove_intent`: an impulse from the shover's strength
   along its facing, on an adjacent unit of any team, so a marine can push a
   teammate out of a line of fire), hits (each round's real momentum along
   its flight), wind (`q · C_d·A`, the #140 read) and blast (the same read
   behind the front, plus whatever the literature says the front itself
   adds). Each also carries a lever arm for the angular part.
5. Hits read the START-of-slot state and land together (rule 5), so two
   shooters push twice as hard with no record of who shot: the same shape as
   G4p's exposure. Whether exposure and balance are ONE decaying accumulator
   or two is a question for the session.

**RULED (Erik, 2026-11-10), after the literature search
(`docs/research/impulse_on_a_person_2026-11-10.md`):**
- Bullets push with their REAL momentum; a declared multiplier
  (`[impulse] bullet_push_mult`, shipped 1.0) exists for experiments.
- The wind bands are restated as force over body weight (Claude's call,
  Erik leans yes): slowing from about 0.036, falling at about 0.22, which
  keeps the #140 numbers for the reference marine and makes mass and
  presented area count.
- A sustained force above the footing limit DOES displace a unit, by its
  excess over the limit; below it, it only slows.
- A unit's OWN turning has inertia as well as being turned by outside
  forces (shots, shoves).

Still to design on these rulings: the integer law itself, its state, the
moment of inertia and lever arms (a short follow-up search), and the
remaining questions below.

**Questions for the session.**
1. Per-round push: the round's real momentum (PROPOSED, see the table), with
   what fraction for a round that passes through?
1b. Rotation: does a unit's own turning have inertia, or only pushes?
1c. Do wind and blast switch to the one law inside G5p, or in a follow-up?
2. Does a marine get the same pushback from hits as a zombie (by mass and
   profile), or do marines only stagger from shoves and blasts?
3. One accumulator with exposure, or two?
4. Does the shove cost anything (a cooldown; later stamina)?
5. Can a shove push a unit into fire, out of an airlock, off its aim? (The
   mechanic allows all three unless ruled out.)
6. Order against P3c: the pushed velocity is a second term in the movement
   law that P3c rewrites in integers. PROPOSED: G5p writes its law in Q16
   from the start and P3c absorbs it.

## 6. Order, size, re-baselines

Order (Erik agreed): G1p → G2p → G3p → G4p → G5p → G6p → G7p → G8p → G1p-b (the `player_up`
camera spike), then the
G1 reading (#29), then P3a onward. G5p's literature search and design
session run while G1p to G4p are built.

| patch | tier | size | HUMAN-TEST | pins |
|---|---|---|---|---|
| G1p | Opus | built 2026-11-10, awaiting the play test | yes | re-baselined (the turn-rate cap) |
| G2p | Sonnet | ½ day | yes | re-baseline |
| G3p | Sonnet | ½ day | yes | unchanged |
| G4p | Opus (the bench, the tuning with Erik) | ½ day + tuning | yes | re-baseline |
| G5p | Opus | 1–2 days after its design | yes | re-baseline |
| G6p | Sonnet | ½ day | yes | unchanged |
| G7p | Sonnet | 2 h | no | unchanged |
| G8p | Opus | 1–2 days after its design | yes | re-baseline |

One implementation agent at a time, each patch its own commits on this
branch or its own branch off main, merged `--no-ff` on green after Erik's
play test.

## 7. New per-unit state (P3a's input)

| column | patch | type | default (hashes nothing) |
|---|---|---|---|
| `current_mag`, `spare_mags`, `reload_done_tick` | G2p | int | untracked sentinel |
| grenade counts, one per hand-grenade row | G3p | fixed block of int | 0 |
| `exposure` | G4p | Q16 | 0 |
| `pvx`, `pvy` (+ whatever balance state §5 settles) | G5p | Q16 | 0 |
| `last_melee_tick` (exists for zombies), `last_shove_tick` | G5p, G6p | int | −1 |
| `flashlight_on` | G7p | bool | on |
| `pending_reload`, `pending_shove`, `pending_melee` | G2p, G5p, G6p | latch bits | clear |

Magazine state is NOT in the unit digest today (`mag_gate`'s docstring says
why). P3a's schema hashes it; until then the hashed hp / event stream
catches a divergence one tick late, as now.

## 8. Open questions for Erik (the short list)

1. G1p: is the proposed default key table acceptable as a start (it is data)?
2. G1p: sprint unlimited for now?
3. G2p: `reload_seconds` for the carbine (PLACEHOLDER 2.0 s)?
4. G3p: flashbang numbers (radius, seconds, the facing-away fraction) are
   placeholders for the play test; is "any team, including the thrower's"
   right?
5. G4p: exposure on kinetic hits only?
6. G7p: keep or drop?
7. §5's six questions, at the session.

## 9. Systems

**(a) Existing canonical systems this design must use.**
Control seams (`ControlSource`, the intent facade, `INTENT_KINDS`, the tap) ·
Match recording / replay (format bump) · MatchSession + the `policy`
controller (`VERB_RULES`) · Weapons tables + `combat.py` archetypes
(`mag_gate` / `mag_spend`, `melee_strike`, `melee_adjacent`) · Simultaneous
attacks (`SimultaneousEffects`) · the DamagePacket pipeline and Credited
cause · status rows (BLINDED, STUNNED, KNOCKED_DOWN) · Loadouts / suits (the
refusing door) · Payload executor and the Generic explosion's flash columns ·
Vision (`has_los`) · Fixed-point kits and `unit_fixed` · Wind load on a body
and the Footprint samples builder (the cadence scale member is where a
slowed walk composes) · Frame lights (`EMITTER_INPUTS`) · the follow camera
and `Camera2D` · Match controller registry (`bots`) · A* · Config · the E2E
driver and the headless match runner · Field digest and the pinned matches
(`tools/match_pin_check.py`).

**(b) New systems, each with a draft one-line rule.**

| system | draft rule |
|---|---|
| Direct control (`control_direct.py`, `[controls.keys]`) | THE one direct-control scheme: pad and keyboard + mouse are device halves of it, keys are a config table; a new device is a half, never a second scheme |
| Camera mode (`[display] camera_mode`) | How the view follows a possessed unit is a mode value; a new framing is a value here, never camera code in a control source |
| Throttle (`set_move_dir`'s `throttle_q`) | A unit's commanded speed is a Q16 throttle between the walk and sprint dials; never a new speed mode id on the continuous path |
| Magazines (`combat.mag_gate` family + `[loadouts.*] magazines`) | Ammunition is magazines: a reload discards the magazine; rounds are never moved between magazines |
| Grenade kinds (`[loadouts.*] grenades` table, the throw's `kind`) | A throwable is a hand-grenade `[ammo.*]` row with a count in the loadout; a new grenade is a row, never a verb |
| Flash blinding (`[payloads.*] blind_*`) | A payload blinds by line of sight and facing through these columns; never a per-grenade check |
| Exposure (`[damage.exposure]`, `ammo.expose`, `expose_resist`) | THE focus-fire law: one decaying per-unit level raised by hits and read at mitigation; start-of-slot read; a new debuff-under-fire is a column here |
| Unit impulse (§5; name and rule after the session) | THE way anything pushes a unit: an impulse over mass into the pushed velocity, resisted by footing; never a direct position write |
| Quick melee / shove verbs | A close attack is a melee weapon row named by the loadout and struck through `melee_intent` / `shove_intent` in slot 4 |

## 10. Not in this pack

#182: grab, mark target, weapon switching, crouch, medpack, action bars, the
knockback stumble animation, aim mode / zoom, night vision, classes, loot.
Animation needs (reload, throw by kind, shove, melee, stagger, blinded) are
listed there as they appear, for the pose layer (#153).
