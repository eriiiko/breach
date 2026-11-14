# RL soldier agent: the design (rung 0)

Started 2026-11-11. **AGREED, version 2; rung 0 is closed** (sections 6, 7 and 8
added on branch `192-rl-agent-design`, delegated by Erik to the lead; the C4
verb of #193 folded into sections 2, 3, 4 and 9). Erik agreed to all of it
on 2026-11-11 ("yes i agree with it all"), before the critique. The critique
(four lenses, section 12) then found no blocker and 23 major points; **this
is version 2, with every fix made on paper**. Where a fix changes something
Erik had agreed, the line says "(critique)" and section 12.3 lists it. He
agreed to all eight the same evening ("yes i agree"). **Rung 0 is closed.**

The plan around it: `docs/rl_agent_roadmap_2026-11-11.md`. The reasons:
`docs/rl_agent_survey_2026-11-11.md`, cited as "survey A.x / B.x".

## 0. The skeleton (AGREED, Erik 2026-11-11)

1. One network for all marines, the same weights for each, deciding about six times a second.
2. Each soldier sees four things: itself; a list of things (team-mates, the enemies anyone in its team sees, last-seen records with their age, grenades and charges, doors); a small map window for walls and the physics fields; an order.
3. A second network, used in training only, sees the whole match.
4. Four choices at once per decision: feet, aim, trigger, one hand action.
5. The soldier aims by pointing at an entry of its list or at a direction. The game turns the body. **The soldier pulls its own trigger**, and is shown the bots' own prediction of whether its next shot will be on target.
6. The order input exists from the first trained network. It is small and concrete. The bots' team layer is the first commander.
7. A unit's class is a type on every list entry, and a mask removes the verbs a class lacks.
8. Hit points, armour and every other game value reach the soldier as ratios of table values, never as raw numbers (they will be re-tuned).
9. C4 on the continuous path is Erik's separate patch; this design assumes its verb exists when needed. (It landed the same evening as #193: `plant_charge_intent`. Sections 2, 3, 4 and 9 are written against the real verb.)

## 1. Frame and rate (AGREED with the action space, Erik 2026-11-11; the window's frame is settled with section 3)

- **Frame: north up, centred on the soldier.** Positions in the list are offsets from the soldier along the map's own axes, and the map window does not turn with it. The soldier's facing is one of its own features. Reason: the ships' walls run along the map's axes, so a window that never rotates keeps every wall straight for the network, and a rotated window would have to be resampled and blurred. It is also the frame of the human's north-up camera.
- **Rate: one decision every 4 ticks** (six a second, 167 ms; OpenAI Five's reaction time was 167 to 267 ms, Honor of Kings acts every 133 ms). Between decisions the choices are held, and the decoder (2.2) keeps issuing verbs on every tick. The rate is one of rung 3's experiments (4, 6, 8 ticks). **Confirmed after rung 1's measurement (Erik, 2026-11-14): the rate is not raised; section 13.1.**

## 2. Action spec v1 (AGREED, Erik 2026-11-11: "your action space is blessed")

### 2.1 The choices

| choice | options |
|---|---|
| **Feet: direction** | stand, or one of 16 compass directions |
| **Feet: speed** | walk, half, full |
| **Aim** | an entry of the list that can be aimed at (an enemy the team sees now, or a last-seen record); or one of 16 compass directions; or "where I walk" |
| **Trigger** | hold, or release |
| **Hand** | nothing, or one verb the unit has and may use now: reload, melee, shove, use (a door), throw one hand-grenade kind (one entry per kind in the grenade table: frag, smoke, flash for today's rifleman), plant the charge it carries (#193) |
| **Hand argument** | the one number the chosen hand verb takes. For a throw, its reach: "onto my aim target" (when the aim is a list entry), or a quarter, half, three quarters or all of the grenade's range. For a plant, its timer: the shortest the plant's table row allows, the longest, or one of two steps between (today 5, 11, 26 and 60 s). The other verbs take none. |

Notes:

- **Strafing needs no choice of its own.** It is what happens when the feet and the aim point different ways. The game's speed law already makes sideways and backward steps slower, and gives full speed only when running the way the body faces.
- **A throw goes where the soldier aims.** The game already releases a grenade along the body's facing at the end of the wind-up, so the throw has no separate direction: the soldier aims, chooses the kind and the reach, and the grenade leaves after the wind-up (0.3 s today). **The aim is held through the wind-up (critique).** The wind-up is 7 ticks and a decision is 4, so without this the next decision's aim would steer a grenade the previous one threw. While a throw is being wound up the decoder keeps the aim of the decision that started it, as the bots do, and the aim choice of the decision in between is not used.
- **"Where I walk"** is the game's own rule for a unit that holds no aim: its travel direction becomes its facing target.
- **The three speeds, in Erik's words:** walk is the quiet pace with the steadier aim, as walking in Counter-Strike; half is the normal pace; full is the sprint. In the game today the three differ in speed only: a footstep makes no noise and the weapon's spread does not depend on how fast the soldier moves. Until the game gives the slow pace a benefit, a learning soldier has no reason to choose it. Footsteps heard by speed and spread by speed are a gameplay patch of Erik's to order (section 11).
- **Why three levels and not a continuous speed.** The game already takes any speed: the throttle is a whole number from 0 to 65 536, which is how the pad's stick feeds it, and no float enters the game. The limit of three is the soldier's, not the game's: networks learn a choice among a few levels far more easily than a continuous number (survey B.7). A finer set of levels is a later spec.
- **Reach is a fraction of the table's range**, never a number of tiles (skeleton line 8).
- **The plant, as #193 built it** (added 2026-11-11; AGREED, Erik the same day). The charge goes down at the soldier's own feet, on the tile it stands on when the plant ends. Planting takes a second (a table value). During it the soldier cannot start a throw or a melee blow, but it may walk and shoot; dying or being stunned plants nothing and keeps the charge. The timer is the verb's one argument, held by the game inside the range of the plant's own table row, `[weapons.c4_satchel]` (5 to 60 s today). The blast hurts everyone in reach, the planter included. The human steps the timer in twelve 5 s steps; the soldier gets four, spaced by a constant factor between the row's shortest and longest, so the spec holds no number of seconds. A finer set is a later spec, as for the speeds.
- **Size.** 49 foot choices, about 17 + the list's aimable entries for the aim, 2 for the trigger, about 8 hand verbs, 5 argument values.

### 2.2 How a choice becomes verbs, tick by tick (the decoder)

A small integer layer **inside the `policy` controller** (critique: a policy
drives units only as that one controller, so the decoder is a per-tick step
of it, never a second controller). It remembers a pointed entry by the game's
own unit id, never by its place in the list, because the list is re-sorted at
every decision. On every tick of the decision it issues the game's existing
verbs:

- **Feet:** `set_move_dir` with the direction's unit vector and the speed's throttle, or `clear_move_dir` for "stand".
- **Aim:** `set_aim` with the direction from the soldier to the chosen entry **as it is on this tick**: to the unit's current position while the team sees it, else to its last-seen position. So the body keeps turning toward a moving enemy between decisions. A compass direction is its fixed vector; "where I walk" is the zero aim.
- **Trigger:** `set_trigger`.
- **Hand:** the chosen verb, once, on the decision's first tick. A throw carries the kind, the table's one default fuse, the reach as a fraction of the table's one throw range, and the body's facing as its direction (the verb needs a direction that is not zero). A plant is `plant_charge_intent` with the chosen timer as a whole number of ticks, converted once from the plant row's range.

Nothing new enters the game: the same verbs, through the same tap, so every
match replays as before. No lead is computed: the carbine's round crosses its
whole range inside one tick.

### 2.3 What depends on what

The feet are chosen on their own. The aim is chosen first among the rest; the
trigger, the hand verb and the hand argument are then chosen knowing which aim
was picked (the network hands the chosen entry's embedding to those heads).
The hand argument is chosen last and also knows the hand verb, since a reach
belongs to a throw and a timer to a plant (added with section 8; both MOBA systems kept this same one dependency,
the "how" on the "what"). Reason: whether to fire, and whether and how far to
throw, depend on the target; nothing else in the survey's systems needed more
dependency than that (survey B.8.3).

### 2.4 The masks

Every option carries a legal / not legal flag for this decision, computed by
the game's own rules, never by a copy of them:

- **an aim entry:** the unit is in the team picture. Not "is alive": a mask that switched off when an unseen enemy died would tell the soldier of a death nobody saw (critique);
- **speed "full":** always legal, but its effect depends on facing, which the soldier sees;
- **"nothing" for the hand and the compass directions for the aim are never masked,** so every choice always has a legal option;
- **the hand verbs,** one row each, as the game's code gates them today (critique: the first draft paraphrased them, and wrongly):

| verb | legal when | not in the mask |
|---|---|---|
| reload | the gun has a magazine, is not full, is not already reloading, and a counted spare exists | the game lets a stunned unit reload (a game finding, section 11) |
| use | a door is within one tile of the body | the game has no other gate: a stunned unit can open a door (section 11) |
| shove | the unit can act; its shove cool-down has run | whether anyone is in reach: a miss costs nothing |
| melee | the unit can act; no throw being wound up, no plant under way; its blow cool-down has run | whether anyone is in the arc: a miss costs nothing |
| throw, per kind | the unit can act; it carries the kind; no throw being wound up, no plant under way | |
| plant | the unit can act; it carries a charge; no plant under way, no throw being wound up | |

Building the masks means lifting each verb's gate out of the verb into a
function that the verb and the mask both call. Each such function only reads
(the reload's uses the game's read-only magazine status, since the verb's own
gate also settles the magazine).

**A mask is made one step early** (critique). The game checks a verb when it
consumes it, a moment after the decision, and something may have happened in
between (a knock-down). So the promise is one-sided: a masked choice, if
issued by force, does nothing in the game; a legal choice usually acts and
may occasionally do nothing. That is an accepted gap.

### 2.5 Verbs are pointed at, not numbered

The hand head does not have a fixed slot per verb. It points into the list of
verbs the unit has, each verb a learned vector, the way OpenAI Five points
into its "available actions". A new verb (C4), or a later unit class with other
verbs, then adds a row to a table and changes no shape.

### 2.6 Left out of v1, and why

- **Flashlight.** Nothing in the game reads the bit until light becomes game state (#81), and the observation refuses light. An action with no effect the soldier can sense is noise for learning. Its entry is reserved and masked; it opens at rung 8.
- **Choosing a fuse.** The table's default fuse per kind. A short / normal / long choice is a later spec.
- **Aiming at team-mates, doors or corners as entries.** No verb acts on them yet, and the compass directions cover looking their way. They become entries when the verbs of 2.8 arrive.
- **A route autopilot.** The soldier walks itself. What it gets instead is a compass in the observation: the direction and length of the shortest walkable route to the order's place (section 3). The same principle as the trigger: the game shows the soldier what the bots know, and the soldier acts.

### 2.7 The tests that can fail (rung 1)

The bots through the door (roadmap rung 1): the layered bots, their every
decision translated to the nearest choice of this spec, against the same bots
unconstrained. Run in these variants, so each number above is a measurement:

- 8 against 16 directions, for the feet and for the aim;
- a decision every 4, 6 and 8 ticks;
- the trigger decided per decision against per tick (how much the 4-tick trigger costs in hits);
- throws onto list entries against throws by direction and reach.

### 2.8 Places reserved for verbs the game will get

Erik, 2026-11-11, wants these in time; none is built: reviving a downed
team-mate, dragging a downed unit, picking up and placing a bag (a mission
objective), a general "use" (buttons and the like), crouching, going prone.

- **Stance becomes a fifth choice:** stand, crouch, prone. It is a state that is held, like the feet, not an event. Reserved, and fixed at "stand" until the game has stances.
- **Revive, drag (take hold / let go), pick up, put down and use are hand verbs.** Each acts on the thing the soldier is aimed at, within reach, the way melee and shove already act on what the body faces. So the aim list gains entry types as these verbs arrive: a downed team-mate, a bag, a button, a door.
- **Reserving costs nothing:** verbs are pointed at (2.5), and an entry type is a row in a table. **Building each is a gameplay patch** with its own design, play test and unit state: what crouching does to cover, sight, speed and aim; a downed state between alive and dead, and how it meets the zombie conversion; what carrying or dragging does to speed and to the hands.

## 3. Observation spec v1 (AGREED, Erik 2026-11-11: "I bless all of it")

### 3.1 Rules

- **Honest.** A soldier is shown only what its team may know (3.6). The training-only value network is shown everything (section 7).
- **Everything the bots have** (AGREED, Erik 2026-11-11). Whatever the scripted bots read to decide, a learned soldier may see: an enemy's health and focus-fire level, the team picture, the shared target, the fire lanes, and what the bots know about the tiles around them (3.4). Two things the bots have and a soldier does not, both accepted gaps: a team-mate's intended standing place (the bots' own memory, which 4.1 keeps out of the display), and the bots' check for an adjacent enemy, which ignores walls (a bot finding, section 11).
- **Every field that acts on a unit, in that unit's own terms** (3.4).
- **No raw numbers.** A quantity is a ratio of a table value (health over full health). An angle is its cosine and sine. A kind (class, weapon, grenade kind, entry type) is a learned vector per kind, never an id number. A yes / no is a flag.
- **Positions are offsets from the soldier, north up,** as a fraction of a reference length from the tables, with the distance given again on a compressed scale so that near things are resolved finely.
- **Every "time left" is a fraction of its own table duration** (the reload time, the wind-up, the plant time, a status's length, the memory time), and a count is a fraction of the loadout's count (critique).
- **Every ratio is squashed by one fixed curve before it is stored** (critique). A ratio is bounded only where its table value bounds it: temperature over the bearable maximum is 0.1 in a room and tens in a fire. Above 1 the value is compressed and capped, by a fixed function written in the spec, so that three burning tiles cannot swamp a network. No running statistics are involved.
- **Sight hides things, not places** (critique; an accepted gap until rung 8). Units, grenades and charges are shown only when the team may know of them. The map, the doors and the physics fields are shown as they are, through walls, in both windows; that is how the bots and the human's screen have them today. So a soldier can see a door open or a fire start in a room no team-mate looks into.
- **Lists are capped, nearest first,** and an empty slot is masked out, never filled with zeros that could be mistaken for a thing at the soldier's feet.

### 3.2 Itself (one vector)

- **Body:** health; facing; how fast it is turning and how far it still has to turn to its aim; its velocity; whether it is being pushed, is catching itself, or is knocked down; stunned, blinded, burning, poisoned, suffocating, each with the time left.
- **Weapon and kit:** rounds left in the magazine, magazines left, reloading and the time left; grenades by kind, a throw being wound up and the time left; charges carried, a plant under way and the time left (#193); the melee and shove cool-downs; what it is (class) and what it carries (the kit's table values as ratios: range, rate of fire, spread).
- **The air where it stands:** pressure, oxygen, temperature, smoke, fire.
- **What just happened to it:** the direction its last hit came from (the round's own flight direction, or the blast's centre; never the attacker's position, which it may not know), how long ago, and by what kind of cause.
- **Its own previous choices,** and the time left in the match. The previous aim, when it was a list entry, is a mark on that entry, never a position in the list, which is re-sorted at every decision (critique).

### 3.3 The list

| entry type | cap | each entry shows |
|---|---|---|
| a team-mate | all | where it is, facing, velocity, health, class, reloading / winding up a throw / stunned / blinded / down. What it intends is section 4. |
| an enemy the team sees now (marine or zombie) | 16 | where it is, facing, velocity, class, its health and its focus-fire level (as the bots read them), what it is doing (firing, reloading, winding up a throw, stunned, down). **For me:** do I see it myself or does a team-mate; is my line of fire to it clear; is a team-mate in that line; how long until my body could be on it; is it in weapon range, in throw range; is its facing on me. |
| a last-seen record | 8 | where it was last seen, how long ago, which way it was moving, its class |
| a grenade in the air or on the floor; a planted charge | 6 | where it is, its velocity, its kind, whose it is, the time left; for anything that blasts, how far its blast reaches and whether I stand inside that reach (the table row's blast radius; that it ignores walls is an accepted gap). A grenade or a charge is shown when a team-mate has a line of sight to its tile, an enemy's with its time left (AGREED, Erik 2026-11-11, with #193; the critic of lens 2 would have hidden an enemy charge's timer, since it is a private choice). |
| a door | 4 | where it is, open or closed, within reach or not |
| reserved | | a downed team-mate, a bag, a button (2.8); a noise heard (rung 8) |

- **"How long until my body could be on it"** is the trigger's aid, given for every candidate target before the choice: the turning law's own time to bring the weapon within its spread of that target. For the current aim target the soldier also sees the bots' exact prediction for the next tick. The time is a new read-only function of the turning law itself, never a copy of it (critique).
- **The line-of-fire flags:** something solid in the way, a team-mate in the way. "In the way" is judged as a round flies (it stops on anything solid), which is not the same as sight: a crate stops a round and a grille stops sight (critique).
- **A last-seen record holds what was seen when it was seen:** the place, the class, the motion. It never follows the unit afterwards (3.6).
- **"The damage it took in the last second"** (4.3) counts only hits my team dealt or saw, in bars (critique).
- About 40 entries at most. Attention over that many costs almost nothing.

### 3.4 The map window

North up, centred on the soldier, at two scales:

- **near:** 25 x 25 tiles at full resolution (about 8 m across), for walls, doors and cover;
- **far:** 25 x 25 blocks of 4 x 4 tiles (about 33 m across: the room, its neighbours and the corridors between), for the layout and where the fire is. Units beyond the window are still in the list.

**The physics planes: every field that acts on a unit, or that the bots
avoid, and no other.** Each is a ratio of a table value, in the unit's own
terms where the game has such a term.

The first draft said "1.0 means this starts to hurt me" for every plane. The
critique checked each against the code, and that holds for only some of them:
the game does not yet hurt anyone by vacuum, low oxygen or water (the
environment profile's limits for them are data that nothing reads), and smoke
and fuel vapour act on no unit. So, plane by plane (critique; changes an
agreed table, section 12.3):

| plane | scaled by | what it does to a unit today |
|---|---|---|
| temperature | the profile's highest bearable temperature | burns come from the heat a unit *feels* (radiation and hot air on its body), which the soldier's own vector carries (3.2); the plane shows where the hot tiles are, and the bots avoid them |
| fire | the field's own full scale | heats; the bots avoid it |
| poison gas | the density that starts to harm, times the unit's resistance | hurts (a zombie not at all) |
| tear gas | the density that blinds | blinds |
| smoke | the density the bots treat as dense | nothing yet (it does not block sight); the bots avoid it |
| pressure | the profile's lowest and highest bearable pressure | nothing yet; the bots avoid a vacuum |
| wind, two planes (its push east and south) | the push that knocks this unit down, read through the game's one wind read | slows, pushes, knocks down |
| *reserved:* oxygen, water depth, fuel vapour | | nothing, and no bot reads them: each plane arrives with the game rule that makes it matter |

When the game makes vacuum or low oxygen hurt, the plane's meaning becomes
"hurts at 1.0" with no change to the spec, and a suited marine then sees a
vacuum as harmless. A later unit class sees the same ship in its own terms,
with no change to the network's shape. The soldier's own tile gets the same
ratios in its vector (3.2). Left out is what acts on nothing a unit can sense
or use: the nitrogen, the energy books, the blast source, and light until it
is game state (#81). Erik, 2026-11-11:
temperature, smoke and poison should be seen "if they don't get the whole
physics tensor"; this rule is the lead's answer to "what do you think".

**The structure planes:** wall, door.

**The sight planes (PROPOSED, near window): what the bots know about the tiles
around them.** A bot judges a tile by whether it sees the target and whether
it is hidden from the other known enemies. To give a soldier the same:

- *watched:* the tiles that at least one known enemy has a line of sight to;
- *sees my target:* the tiles from which the soldier's aim target can be seen.

Both come from one new small routine in the engine: from a point, which tiles
are in line of sight. It runs once per known enemy per decision and is shared
by the whole team. It uses only enemies the team knows of, so it stays honest.
The routine is the existing sight test run for many tiles at once in C++, the
same line walk with the same rules for what blocks sight, proven equal to it
tile for tile; in Python it would cost about a quarter of a second per
decision (critique). When cones return at rung 8 it takes the enemy's facing.
Reason: cover and concealment are most of what a soldier does with its feet,
and line of sight is the one thing a small network infers badly from a map of
walls.

**In the far window also:** where the team-mates are, where the seen enemies
are, and the route map of 3.5. These unit planes are drawn from the list, so
they obey sight; the environment's present unit planes, which show every unit
on the ship, are not part of this spec (critique).

**How the windows are stored** (critique): yes / no planes as single bytes,
ratio planes as 16-bit numbers after the squash of 3.1. Otherwise one decision
is about 90 KB and a training batch does not fit in memory.

### 3.5 The compass

For the order's place the soldier sees the direction of the first step of the
shortest walkable route and the route's length, and in the far window the same
route as a plane (steps to the place from every block). This is the distance
field the bots' team layer already computes for its objective. The feet stay
the soldier's own.

### 3.6 Sight: who may know what

- **One sight rule per match, the same for every seat in it** (bots, learned soldiers, a human). It is a setting of the match. Otherwise a win over the bots would be a win by better eyes.
- **Training matches start with all-round sight, walls block, shared across the team.** The cone the game has today (110 degrees in front, a 4-tile circle all round) returns as its own experiment at rung 8. Erik, 2026-11-11: cones are wanted, but "just to have agents playing good is more important"; reducing what they know comes after, and will cost much more computation, since the soldiers must then keep their own memory of the world; "it's fine to cheat like this a little bit now". He does not need any match played before today preserved. **It is nevertheless built as a setting of the match** (critique): a key in the match file, cone or all-round, with today's cone as the default. The reason is one frozen test: the first-generation bots are kept unchanged as a fixed opponent, their reference match reads the cone, and changing sight outright would break it. Training runs choose all-round. The zombies' own waking rule (a radius and a line of sight) is not sight in this sense and is untouched.
- **Last-seen records:** where, when, what class and which way it was moving, as seen at that moment; forgotten after the memory time (6 s today, a setting). **A record does not vanish when its unit dies unseen** (critique). The bots' own picture drops a record the instant its unit dies, seen or not, which tells them of deaths nobody saw. The shared team picture keeps the record its full time unless the team saw the death, and the bots are moved onto that same picture, losing the small cheat; the bot tournament measures what that costs them.
- **What may be known about a seen enemy** beyond what the screen shows: its health and its focus-fire level. AGREED, Erik 2026-11-11: "let's give them everything that the bots have".

### 3.7 Cost and tests (rung 1, patch 1g)

- The observation is built once per decision per soldier. "Do I see it" comes from the game's per-tick sight cache, which the environment must ask for on every tick (it is built only on demand); the route map from the flood the bots already run. The farm's speed with the new observation is measured against today's.
- **Three things must be built for the cost to hold** (critique; estimates, to be measured in patch 1g): the many-tiles sight routine of 3.4 in C++; the per-tick sight test no longer run for zombies as observers, whose sight nobody reads (all-round sight otherwise triples its cost); the far window made by reducing the map to blocks once per tick, then cropping.
- **Features that need a small new read-only function or tracker,** named so that none is forgotten: the time to turn onto a target (in the turning law's module); how many team-mates a frag would catch (from the frag row's blast radius); a record's motion at the sighting; the wind's push per tile; the point a team-mate is turning to and where a throw will land (from the facing target and the reach).
- Tests: the observation equals an independent reconstruction from the game state; no entry, and no unit plane of the far window, ever shows a unit the team cannot know of; a record outlives its unit's unseen death.

## 4. The team display (AGREED, Erik 2026-11-11: "I think we got it now. I have nothing to add.")

### 4.1 Principles

- **It is read from the game's state, never from a controller's memory.** Where a unit is turning to, whether its trigger is held, where its feet are taking it, a throw being wound up, a reload: all of these are already the unit's state in the game, and its order becomes unit state with patch 1f. So a bot, a learned soldier and a human look the same to their squad-mates, and the display needs no new channel. "Is it firing" reads the held trigger, not the shot events, which exist for the picture only.
- **Fixed and readable.** No learned messages (roadmap rule 4; survey B.8.4).
- **The radio is perfect, for now.** Every living team-mate is shown, wherever it is on the ship. This goes with all-round sight: cheat a little now, reduce later (rung 8).
- **"Whom it aims at" is decided by one rule for everyone,** judged by what the *watching* team knows (critique; the first wording could have revealed a unit the watcher's team has never seen, and used an enemy's intention, which nobody can see). For any unit: the first unit known to my team that lies along the way its body is facing, within its weapon's spread, before anything blocks sight. For a team-mate, the point it is turning to may be used in place of its facing. For a zombie, which has no weapon: toward me, toward a team-mate, or nobody. The answers are: me, a team-mate, a known unit of another side, nobody. A unit my team does not know of always reads as "nobody".

### 4.2 On each team-mate's entry

Added to its state (3.3):

| shown | meaning |
|---|---|
| its aim | the point it is turning to, whether that is a unit, whether it is firing |
| its feet | the direction and speed it has chosen |
| a throw being wound up | the kind, where it will land, the time to release |
| a plant under way (added with #193, same principle: it is the unit's state) | the time until the charge lies down, and the timer it was given |
| a reload | the time left |
| its order | what it has been told, and where |
| the lanes between us | am I in its line of fire; is it in mine |

### 4.3 On each enemy's entry

- how many team-mates aim at it, and how many are firing at it;
- the damage it took in the last second;
- whether it is the commander's shared target (section 5);
- how many team-mates a frag thrown at it would catch.

### 4.4 The team as a whole

The bots' own counts, as one small vector: team-mates alive, the team's
ammunition, enemies known and seen, zombies near. (The first draft added "the
play". It is dropped: the play is the bots' controller memory, against 4.1,
and the order's "what" already carries it to the soldier; critique.)

### 4.5 In the near window

One more plane: **the fire lanes**, the tiles that lie in a team-mate's line
of fire. A lane is read from state (critique): a team-mate whose trigger is
held, from its body along its facing to the first solid tile or its weapon's
range. The bots' own lane grid could not be handed over as it is: it is
filled by the bots' controller, so for a learned team it would be empty. The
grid's drawing routine is shared; the bots are moved onto the state-read lanes
together with the team picture (3.6).

### 4.6 What each part is for

| behaviour wanted | what serves it |
|---|---|
| focus fire | the counts on each enemy; its health and focus-fire level (3.3); the shared target |
| not shooting a team-mate | "is it in my line of fire" on each team-mate and each enemy |
| not standing in a team-mate's fire | "am I in its line of fire"; the lanes plane |
| not walking into a team-mate's grenade | the throw being wound up and where it lands; grenades in the list with whose and time left |
| spacing; covering different directions | each team-mate's position, aim and feet |
| not all reloading at once | each team-mate's reload and time left |

### 4.7 Experiments (rung 3)

- **The display on and off:** sections 4.2 to 4.5 removed, sight still shared. Does the display earn its place, measured in wins and in friendly-fire hits per match?
- **Staggered decisions:** soldiers of a team decide on different ticks, as the bots do, so that each sees what the others have just chosen. It may cure two soldiers picking the same thing in the same instant; it costs more calls to the environment.
- **Later, a "mark" verb:** a soldier points out a target or a place to its team, a fixed readable signal that a human could give too.

### 4.8 Asked of Erik

"If you were the soldier, what would you want to know about your squad that
is not in this picture?" Nothing to add (2026-11-11).

## 4a. Sides: zombies and more than two teams (AGREED, Erik 2026-11-11: "yes i agree with it all")

Erik: three sides are more interesting than two. A symmetric match is hard to
win consistently; the zombies bring chaos, and it can pay to ambush the team
that is busy killing zombies, or to get the zombies to work for you.

- **Zombies are in everything above already:** entries in the list with their class, in the far window, in the team's counts, and in every training match from rung 3.
- **Every enemy entry says which side it is on:** the horde, or which of the other marine teams. The env already takes any number of teams.
- **Every enemy entry says whom it is aiming at:** me, a team-mate, a known unit of another side, nobody. The same rule as for team-mates (4.1, as the critique sharpened it: judged from its facing, among units my team knows). This is what makes "they are busy with the zombies" visible.
- **A zombie's entry says whether it is dormant, hunting, or walking toward a noise.** Zombies come to noise, and a soldier who can see that can use it. (The critic of lens 2 notes that this tells a soldier of a noise it cannot itself hear until rung 8, and that the bots read no such state. Kept: it is what a watcher of the zombie would see, and Erik agreed it.)
- **Matches with three sides** (two marine teams and the horde, or three marine teams) are a setting of the training runs, not a new design. One consequence for the reward section: with three sides one team's gain is no longer exactly another's loss.

## 5. The order (AGREED, Erik 2026-11-11: "Agree with all your suggestions"; the grain of a room or a door is "very good"; "this part of the ship" may be a later refinement)

Erik, 2026-11-11: not a copy of StarCraft but distributed intelligence. "The
agents are their own AIs, the commander is its own." An order to a tile
"defeats the purpose of having actual agents play the soldiers"; a room to
secure, with the soldiers finding their own cover, is the level he leans to.
He wants to try several approaches. His picture of the commander: a schematic
graph of the map (rooms, choke points, objectives, threats) read by a graph
network.

### 5.1 An order names a place on the ship's graph, never a tile

The generator already builds every ship as a graph: rooms are the nodes, and
the doors and other openings between them are the edges. A door is a choke
point by construction. That graph is the commander's map and the order's
vocabulary.

An order is three things:

| part | values in v1 |
|---|---|
| what | none; go to; hold; clear; fall back to |
| where | a room, or an opening between two rooms (a door to watch or to hold) |
| until | a number of seconds, or until told otherwise |

Beside it, optionally, a **priority target**: one enemy the commander wants
shot first. This is the bots' shared target.

How to get there, where to stand, what cover to take, when to shoot and at
whom when no priority is set: all the soldier's.

### 5.2 What the soldier sees of its order

The kind; the time left; whether it is inside the place; the compass (3.5) to
the place; the place drawn as a plane in the far window; a flag on the list
entry that is the priority target. Each team-mate's order is on its entry
(4.2), so a soldier knows who was sent where.

### 5.3 The first commander

The bots' team layer, translated: its objective tile becomes the room that
tile is in; fall back, hold and bound become "fall back to", "hold" and "go
to"; its shared target becomes the priority target. Orders in training also
come from a generator of random legal orders and from labels of what a soldier
in fact did (roadmap rung 4). A caution for that third source (critique): an
order written afterwards from where the soldier went carries knowledge of the
future into the soldier's own input. Rung 4 may use such orders only as a
lesson in obeying, looking no further ahead than the order's own length, and
never in matches that measure strength.

### 5.4 Orders start as information and become binding at rung 4

From the first trained network the order is in the observation, written by
the bots' team layer. Until rung 4 nothing rewards obeying it; it is simply
useful (it says where the enemy was last known to be). At rung 4 a reward for
following comes in, with a weight that is swept (the reward section).

### 5.5 The order format is a named spec

Like the observation and the action space. This one is "rooms". A looser one
("be in this general area") or a tighter one can be tried beside it, and
teams under different formats can fight each other.

### 5.6 The later commander (rung 6), in outline

It reads the same graph, each room carrying what the team knows: who is in
it, threats last seen there, fire, vacuum, the objective. A graph network is
the natural reader. It is attention restricted to neighbours, the same family
as the soldier's list network, and it runs on whatever graph it is handed at
each decision, so rooms whose contents change, a wall blown open (a new edge)
or a door sealed (an edge gone) are all ordinary input. Its design is rung 6's.

### 5.7 What this needs from the game

- **The room graph saved with the level** (made exact by the critique). The generator has it and today writes only the spawn rooms. A patch of its own in rung 1:
  - the fixed part is level data: the rooms, the openings between them, and a map from every tile to its room, written by the generator through the level writer and read through the loader. An opening's id is derived from the two rooms it joins and its first tile, never from a counter, so a recorded order means the same thing in a replay;
  - the changing part is worked out, not stored: which openings can be walked now follows from the fixed map, the walls as they stand and the doors' states, recomputed when the game's own "the walls changed" counter moves. It is neither part of the synced game nor unit state;
  - a level without a graph (hand-made, or generated before this patch) is refused for training;
  - the bots' own room goals read the same map: one map from tile to room, never a second.
- **The order as unit state,** set through one verb by whoever commands: what, where (a room or an opening id), until, the priority target's id. Four consequences the critique made explicit:
  - the verb is a commander's, so it is kept out of the soldier's action columns (as the horde's verb is);
  - it is recorded like every verb, so the recording format gets a new version;
  - it takes effect inside the game's tick, as the other verbs do;
  - an order is hashed only while a unit has one, and the bots' team layer writes orders only for a team that is driven through this door. A match of bots alone therefore plays and hashes exactly as today.

## 6. Reward profile v1 (the lead's, delegated; AGREED by Erik 2026-11-11, then corrected by the critique)

The reward is the number a soldier is handed after each decision. Learning
changes the network so that the sum of rewards over a match grows. So the
reward is the only place where we say what we want; everything a soldier
learns, it learns because this number moved.

### 6.1 Rules

- **Winning is the objective.** Every other term exists to help a new network find its way to winning, is small beside the win, and is turned down once the team wins without it.
- **A helper term is the change of a score that can be read off the game's state.** Then no loop can farm it: whatever a loop gains on the way out, it gives back on the way home. (This is the idea of Ng, Harada and Russell 1999, and what we chose for Civulator's city-value reward in October. Their stronger promise, that such a term changes only how fast the best behaviour is found and never which behaviour is best, does not apply here in full, as the critique pointed out: the score a team ends the match with stays paid. So a win that cost three soldiers is worth less than a clean one, and a draw with the better balance beats a draw with the worse. That is wanted, and it means the balance is a small part of the objective, not only a help.)
- **Few terms.** Every term is something a learner will try to cheat. What is deliberately not rewarded is listed in 6.4.
- **No game number.** Health is counted in bars. **1 bar is the health a unit had when it came into being on its side:** at spawn, or, for a soldier that turns zombie, at the turning. Health above 1 bar counts as 1. (Critique. The first wording, "the unit's full health", has no single meaning for a zombie: one spawned by the level starts with 100 hit points and one that was a marine with 400, while the game's only "full health" figure says 100 for both. See section 11.)
- **A named spec.** A reward profile is a table of weights in the run file, like an observation spec. The environment hands back each group of terms as its own number, and the trainer does the weighting, so a weight can be changed or swept without touching the game.

### 6.2 The terms

All sizes are placeholders, to be measured (6.9).

| group | term | size | who gets it | in plain words |
|---|---|---|---|---|
| **outcome** | the team wins | +10 | every soldier of the team, alive or dead | it is the last marine team with a living soldier |
| | the team is eliminated | −10 | every soldier of the team | its last soldier died or turned zombie, whoever did it: an enemy, the horde, a fire, a vacuum |
| | the clock runs out | −3 | every soldier, alive or dead, of each team that still has a living soldier | nobody won. Worse than nothing, so that hiding until the end is not a plan; far better than losing, so that a hopeless charge is not one either |
| **balance** | an enemy marine loses health | +1 per bar | blended, see 6.3 | |
| | an enemy marine dies | +0.5 | | so a kill is worth 1.5 from full health |
| | a zombie loses health | +0.25 per bar | | the horde is in the way and is not the opponent |
| | a soldier of my team loses health | −1 per bar | | the mirror of the enemy's |
| | a soldier of my team dies or turns | −0.5 | | |
| **friendly fire** | I hurt a team-mate | −2 per bar of the team-mate | the soldier the game credits with the hit, and nobody else | |
| | I kill a team-mate | −3 more | | |
| **order** | (rung 4) | weight 0 in v1 | | 6.7 |

The four balance terms are one thing seen four ways. Write a team's
**balance** as: the health of its living soldiers plus half a point for each;
minus the same for every enemy marine; minus a quarter of every zombie's
health. The balance terms are simply how that number changed during the
decision. Three things follow from it being one number of the game's state:

- between two marine teams the marine terms mirror each other, so a one-for-one trade is worth nothing and only the outcome settles it. (The zombie term does not mirror: a zombie's lost health helps every marine team alike);
- a healed enemy takes back, from the team part, exactly what hurting it paid. (Nothing in the game heals yet.);
- a soldier bitten to death costs its team its health and its half point, and then a quarter more, because the horde has grown by one zombie at one bar.

**Why the clock's −3 goes to the dead as well** (critique): if only the
living paid it, a soldier on a team heading for a draw would gain about 2 by
walking into a fire in the last seconds.

**Friendly fire in plain numbers.** Killing an enemy from full health earns
1.5. Killing a team-mate from full health costs the shooter 5, on top of the
team's own loss of 1.5. So a burst through a team-mate is worth firing only
when it is about three times as likely to hit the enemy as the friend. The
blame is the shooter's alone and is never shared out (6.3). A soldier's own
grenade or charge hurting itself is ordinary lost health, not friendly fire.

### 6.3 Own against team

Who is paid for a change of the balance? Two honest answers:

- **own part:** the pieces the game credits to me. The health I took off an enemy or a zombie, my kills, and the health I lost myself. The game already credits every hit and kill to a unit. **Only health that was really there counts** (critique): the game's hit events report the full force of a hit, also on a target with a sliver of health left or already dead, so a charge on a wounded marine would have paid its whole blast. The environment therefore follows each unit's health through the tick and credits a hit with no more than the health it removed, and with nothing on a corpse. The same holds for friendly-fire blame.
- **team part:** an equal share of the whole change of my team's balance, whoever or whatever caused it.

A soldier's balance reward is a mix of the two, set by one number, the **team
share**: 0 means each for itself, 1 means everything is shared.

- With a low team share a soldier learns quickly, because its reward follows its own actions closely. It also learns to steal kills and to let a team-mate die.
- With a high team share it learns to cover, to draw fire and to set up a team-mate's kill. It learns more slowly, because three quarters of what it is paid is other soldiers' doing.
- **v1: the team share starts at 0.3 and moves to 0.8 during a run.** This is OpenAI Five's schedule and their reason. The rival, to be run beside it: fully shared from the first step, which is what the squad benchmarks do at our team size.

Three things are never blended:

- **the outcome:** every soldier of the team gets all of it;
- **friendly fire:** the shooter gets all of it;
- **what nobody is credited with:** an enemy burnt by a fire, vented through a breach, or bitten by a zombie exists in the team part only. The game credits no unit with those (an accepted gap of the credit system), but the balance sees the enemy's loss whoever caused it.

That last line is where Erik's "make the horde work for you" lives: a team
that lures zombies onto its enemy is paid through the team part and through
the outcome, and through nothing else.

**Friendly fire that the game credits to nobody is not punished** (an
accepted gap, named by the critique): a team-mate lost to a fire I lit a
while ago, a room I vented, a shove into either, a horde I led onto it. The
shooter pays only its share of the team's loss. The friendly-fire experiment
(6.9) therefore also reads off team losses with no attacker while a team-mate
was near.

**A dead soldier stays on the books.** Its row keeps receiving its team's
share and the outcome until the match ends; it makes no more choices. So
dying for a win is worth more than dying for a loss, and the value network
(section 7) learns what a soldier's death was worth. A soldier that turned
zombie is a dead soldier in this sense, and what the zombie it became does to
its old team is the horde's doing, not friendly fire.

For this the environment marks every row with two flags (critique; today's
single flag cannot tell a dead soldier from an empty roster place):

| the row is | in the match | acts | the trainer uses it for |
|---|---|---|---|
| a living soldier | yes | yes | everything |
| dead or turned | yes | no | the reward and the value estimate; never a choice |
| an empty place on the roster | no | no | nothing |

### 6.4 What is deliberately not rewarded

Hits and accuracy. Ammunition saved or spent. Moving, exploring, closing on
the enemy. Taking cover. Staying near team-mates or away from them. Using
grenades or the charge. A zombie's death as such (only its health). Each
second survived. Obeying an order, until rung 4.

Each of these is either already inside the outcome and the balance, or it is
a habit we would be guessing at, and a paid habit gets farmed. (The standing
example: a boat-race agent that circled three bonus pick-ups for ever.) If a
behaviour does not appear, the first suspect is what the soldier can see and
do, and only then the reward.

### 6.5 Sizes, checked

- **The win outweighs the rest.** On `ship_demo` (4 against 4, 4 zombies) everything a whole team can earn from the balance is 4 x 1.5 + 4 x 0.25 = 7, plus a quarter for each soldier that turns and is then killed, shared by four soldiers. The win is 10 to each.
- **Luring pays.** An enemy marine bitten to death is worth 1.5 − 0.25 = 1.25 to my team, against 1.5 for one shot dead: nearly as much, at no risk and no ammunition.
- **The horde is worth less than the enemy.** On a ship with many zombies the zombie weight is lowered in that run's profile so that this stays true.
- **Time.** A reward that lies in the future counts a little less than one now; how fast it fades is a trainer number (the discount). At first a soldier looks about half a minute ahead. That is why the balance terms are needed at all: early in a five-minute match the outcome is too far away to steer by. The look-ahead is lengthened as training settles (rung 2's to set and sweep).
- **The clock is part of the game.** A soldier sees the time left (3.2), so the clock running out is a real ending with its −3, not a cut the trainer has to guess across. The environment today reports the clock as a cut ("truncated"), which a standard trainer would guess across; under these profiles it is reported, and treated, as an ending (critique). Two last soldiers dying in the same tick is an elimination of both teams, not a draw.

### 6.6 Three sides (the reward's half of section 4a)

- **The horde** is in the balance at a low weight and in nothing else. A team wiped out by zombies is eliminated like any other.
- **Three marine teams:** every other team's marines are enemies in the balance, at the same weight. The last team standing wins. A team that is eliminated gets its −10 at that moment and its soldiers' records end there. Two of three teams lose, so the average outcome is below zero; nothing has to add up to zero.
- **When two enemies fight each other,** my team's balance rises without my doing anything. That is correct, since my position did improve, and it is what makes waiting for them to wear each other down worth learning. Whether it also teaches hiding is measured by the share of matches that reach the clock (6.9).
- Ganging up on the strongest team, or sparing one enemy to keep another busy, is not designed for or against. It is watched for.

### 6.7 The later obedience term (rung 4; weight 0 until then)

Reserved now so that the value network has its output from the start:

- **progress:** the route to the order's place got shorter (the compass's own number, 3.5). Walking back pays it back, so pacing earns nothing. It is measured against one order only: when the order changes, the jump between the two places pays nothing;
- **completion:** the order's condition was met in time (arrived; held until the time ran out), paid once per order given, so stepping out and in again earns nothing;
- **personal,** never shared: my order, my obedience;
- **a weight that is swept.** Obedience is a dial. Tencent's soldiers kept their strength while completing about four orders in five, and lost it when pushed toward five in five, because an order is sometimes wrong. That is rung 4's gate;
- **orders are blanked in a share of the matches,** so a soldier also learns to fight with none (AlphaStar blanked its strategy input one time in ten).

### 6.8 What it needs (patch 1i) and how it is tested

Today's counters (survey A.5) count hits and raw damage, do not tell a marine
from a zombie, and hold neither the outcome nor anything about the team.
Patch 1i adds, and keeps every old counter working:

- **per soldier per decision:** health in bars really taken off enemy marines, off zombies, off team-mates (6.3); kills of each; its own health lost; its own death or turning;
- **per team per decision:** the change of its balance, computed from the state before and after; the outcome when it falls;
- **from the environment:** five numbers (outcome, own balance, team balance, friendly fire, order) for every row that is in the match, dead rows included, zero for empty roster places; the two row flags of 6.3; the clock reported as an ending.

The reward is computed outside the synced game, from state and credited
events only, so it changes no match and no replay.

Tests, each a property:

- the books close: a team's balance changes over a whole match add up to its balance at the end minus its balance at the start;
- own parts plus the uncredited rest (a turning included) equal the balance change, on every decision, also when a hit is far stronger than the health left;
- a hit on a corpse pays nothing and blames nobody;
- healing an enemy takes back, from the team part, exactly what hurting it paid;
- a win, an elimination or the clock is paid to every row of the team exactly once, the dead rows included; two last soldiers dying in one tick is paid as two eliminations;
- friendly-fire blame lands on the credited unit and on no other; a turned soldier's bites blame nobody;
- the same match replayed gives the same rewards.

### 6.9 Experiments, each able to fail

| question | runs | read off |
|---|---|---|
| Do the balance terms help at all? | outcome only, against outcome + balance | does anything learn within a day |
| Own against team | team share 0, 0.3 to 0.8, 1 | wins; and enemy losses that the horde caused (does luring appear) |
| How hard to punish friendly fire | the weights at 0, as written, tripled | friendly hits per match **and** rounds fired per match: a soldier that stops shooting has not solved friendly fire |
| Can the helper be removed? | balance weights cut to a quarter, and to zero, after rung 3's gate | is strength kept, and does the play change: by 6.1 the balance also pays for winning cleanly, so this is a change of what is asked for, not only of speed |
| What the clock is worth | −3 against 0 and −10 | the share of matches decided; and Erik's eye on how the matches look |

Rung 2 uses cut-down profiles of the same table: **"range"** (a target loses
health; the clock) and **"duel"** (outcome and balance, one soldier, so no
team share and no friendly fire). The table above is **"fireteam"**.

## 7. What the training-only value network sees (the lead's, delegated; AGREED by Erik 2026-11-11, then corrected by the critique)

The value network answers one question during training: from this situation,
how much reward is still to come for this soldier? The trainer compares what
actually came with that estimate, and so learns whether a choice did better
or worse than expected. It is never run in a played match and never tells a
soldier what to do, so it may see what the soldier may not. The survey's two
largest measured effects are both about this (AlphaStar 82 % against 22 %;
hide-and-seek stalling without it).

### 7.1 Its input

One estimate per soldier, so the view is centred on that soldier, like its
observation.

| part | the soldier sees | the value network sees |
|---|---|---|
| the soldier's own vector and its two map windows | all of it | the same stored arrays, read a second time, not a copy |
| units | team-mates; enemies its team sees; last-seen records | **in place of** the soldier's list (critique: MAPPO did best with duplicates removed), every living unit of every side where it truly is, dormant zombies included; each marked "my team sees it", "my team has a record of it" or "my team does not know of it" |
| per unit | what the bots may read, and the "for me" flags (3.3) | the same, and also its magazine and magazines left, grenades by kind, charges, cool-downs, every status with its time left, its order, and the air where it stands |
| grenades and charges | those its team has a line of sight to | all of them, with owner and time left |
| the match | time left; its team's counts (4.4) | also every team's living count and balance; zombies awake and dormant |
| the map | the two windows; the sight planes drawn from known enemies | the same windows; one extra plane, "watched" drawn from every enemy, stored beside them |

**What it does not see:** the future; the match's random seed; the memory
inside any soldier's network.

### 7.2 Choices made, and why

- **The soldier's own view is part of the input.** MAPPO measured that the best input is the whole state plus the soldier's own view with the duplicates removed.
- **A list, read the same way as the soldier's.** Up to 64 entries: every marine, then the nearest zombies.
- **No memory in v1.** While sight is shared and the game keeps the last-seen records, a memory would add little, and leaving it out keeps the trainer's most error-prone part (carrying memory across a match) on the soldier's side only. (The first draft's reason, "it sees the whole state, so it has nothing to remember", was wrong, as the critique showed: what a soldier will do also depends on its own memory, which the state does not hold. The error this causes is known in the literature and is small today. It grows with cones; see 7.4.)
- **A separate network.** It shares no weights with the soldier's network, so nothing it sees can leak into a choice. This is tested (8.5).
- **A dead soldier's view** keeps the whole match, with its own part replaced by "dead, since when". MAPPO blanked the view of a dead soldier; our dead soldiers still receive a team share (6.3), so their estimate has to follow the match. Both are run (7.4).
- **One output for each of the five numbers the environment returns** (outcome, own balance, team balance, friendly fire, order). The trainer adds them with the profile's weights, the team share among them. A weight can then be moved during a run, the team share can follow its schedule, the order term can be switched on at rung 4, and the estimate needs no rebuilding. (The first draft had four, with one output for the blended balance; two critics found that the blend moves with the team share, so that output would have chased a moving target.) Honor of Kings ran five such outputs and reports that it raised strength.

### 7.3 How much is hidden today

With all-round shared sight (3.6), what the soldier cannot see is: units no
team-mate has a line of sight to (most dormant zombies among them), what the
enemy carries and has been ordered, and charges out of sight. So the full view buys less now
than it will with cones at rung 8, where it becomes essential. It is built
now because adding it later would change the trainer and the environment's
buffers in the middle of the ladder.

### 7.4 Built where; tests; experiments

- Built by the environment beside the observation (patch 1g), into the farm's own buffers. The farm's speed with it is measured.
- Test: the full view equals an independent reconstruction from the game state; every unit alive in the game is in it.
- Experiments (rung 3): the full view against the soldier's own view only; the dead soldier's view kept against blanked; the soldier's memory vector added to the value network's input (as plain numbers, with no learning signal flowing back into the soldier's network). That last one becomes a condition for rung 8: cones are not trained with a value network that lacks it.

## 8. Network v1 (the lead's, delegated; AGREED by Erik 2026-11-11, then corrected by the critique)

About 1.5 million weights for the soldier, inside the 1 to 2 million asked
for, and about 1.1 million for the value network, which is never shipped.
Sizes are the starting point of a sweep.

### 8.1 The soldier's network, part by part

| part | what it does | size | weights |
|---|---|---|---|
| **vector reader** | reads the soldier's own numbers, its order and the team's counts (3.2, 4.4, 5.2) | about 130 numbers in; layers of 256 and 128 | 66 000 |
| **entry readers** | one single layer per entry type (team-mate, enemy, last-seen record, grenade or charge, door, one spare), so every entry becomes a vector of the same length | up to 48 numbers in, 128 out | 38 000 |
| **list attention** | two layers in which every entry looks at every other and at the soldier itself | width 128, 4 heads, inner layer 256 | 265 000 |
| **list summary** | the soldier's own vector asks the list one question and gets one vector back | width 128 | 66 000 |
| **near-map reader** | four 3 x 3 convolution layers over the 25 x 25 near window (about 17 planes); the last three each halve it (25, 13, 7, 4), then one layer to a vector | 32, 64, 64, 64 channels; 128 out | 229 000 |
| **far-map reader** | the same over the far window (about 18 planes) | the same | 229 000 |
| **join** | puts the four results together | 512 in, 256 out | 131 000 |
| **memory** | a GRU: a vector carried from one decision to the next | 256 | 395 000 |
| **heads** | the choices of section 2 | see 8.2; hidden layers of 128 | 135 000 |
| per-kind vectors, normalising layers | 8.3 | | under 5 000 |
| | | **total** | **about 1 550 000** |

The fourth critic recounted the table from these shapes and landed between
1.51 and 1.56 million. The counts of numbers and planes going in are
estimates until patch 1g fixes them.

### 8.2 The heads

- **Feet:** one choice among 49 (stand, or 16 directions at 3 speeds).
- **Aim:** by pointing. The network forms a question vector; every entry that can be aimed at answers with its own vector from the list attention; the 16 directions and "where I walk" answer with learned vectors of their own. The best match wins. Entries that are not legal are masked.
- **Trigger, hand verb, hand argument:** chosen after the aim, each told which aim was picked (2.3). The hand verb is pointed at as well (2.5): each verb the unit has is a learned vector. The argument is chosen last and is told the verb.
- **What is left out of the learning signal:** a choice that had no effect. The argument when the verb takes none; the aim while a throw is being wound up (2.1); every head on a soldier that could not act. (OpenAI Five did the same for unused arguments.) Three rules the critique added so that choosing and learning always agree: "no effect" is decided from the masks at the moment of the decision and stored, never from what happened afterwards; the argument's legal values depend on the aim and the verb just picked, and are stored as they were; a head that was left out also gets no push toward variety (the "entropy bonus"), or it would drift into noise.

### 8.3 Why these parts and these sizes

- **Attention for the list** is the survey's firmest result (rule 1). Two layers at width 128 for about 40 entries: hide-and-seek used one layer, AlphaStar three for 512 entities.
- **Two small convolution readers for the map.** Where a wall stands relative to the soldier matters, so the readers keep position until their last layer. They are deliberately small: the game hands over what a small network infers badly from a map (the line-of-fire flags, the sight planes, the compass).
- **A memory, from the start.** With shared all-round sight and the game's own last-seen records, a soldier needs little memory today. It is in v1 because retrofitting memory into a trainer is the expensive direction, and rung 8's cones cannot work without it. A GRU because the one broad comparison of memories found it the best general choice. Rung 2 runs the duel with the memory on and off.
- **The size.** It comes to about 1.5 million by itself, hide-and-seek's size. The farm produces a few thousand decisions a second, and choosing with a network of this size is many times faster than that on the desktop's GPU. Learning is another matter (critique): several passes over each batch through both networks may take about as long as collecting it. Rung 2 measures this before any size is fixed. It is not smaller, because our Civulator ladder showed a too-small network hiding the effect of four changes in a row. Half and double the width are among the first sweeps.
- **Nothing learned about the scale of numbers.** Every input is already a ratio, a flag, a cosine or a learned vector per kind, and every ratio is squashed by a fixed curve at the door (3.1). Inside, a normalising layer follows each reader and precedes each attention layer. So the network keeps no running statistics of its inputs, one thing fewer that can go wrong. (The first draft relied on the ratios alone; the critique pointed at fire, blast wind and breach pressure, which reach tens of times their table value.)

### 8.4 The value network

The same readers with its own weights, fed the full view of section 7; no
memory; after the join one layer of 256, then five outputs (7.2). About 1.1
million weights.

### 8.5 Left out of v1, and when it returns

- **Writing each unit into the map at its position** (AlphaStar's "scatter connection", worth 71 % to 87 % there). Left out because the observation already carries the unit-to-wall relations it would have to discover. An experiment at rung 3, together with its reverse (each entry reads the map under its own feet).
- **A transformer as memory.** After a task shows the GRU short.
- **One network per class.** Rung 7.
- **Sharing a part of each soldier's inner vector across the team** (OpenAI Five pooled a quarter of it). The team display is our fixed, readable version of that; a learned one comes only if the display's experiment fails.
- **Bigger.** Rung 5.

### 8.6 Where it lives, and its tests (rung 2)

With the trainer, outside the game's source tree. A network is described by a
named spec, like the interfaces: "v1" is this section.

Tests, each a property:

- a masked choice has probability exactly zero, when choosing and when learning;
- the probability of a recorded choice, recomputed for learning before any update, equals the one it was made with to within rounding;
- shuffling the list's entries changes no output, except that the pointer's answer moves with its entry;
- an empty list slot changes nothing;
- scrambling the value network's extra inputs changes no choice;
- the number of weights is printed and matches the spec;

and, added by the critique, the places where a self-written trainer most
often goes wrong:

- the trainer's estimate of how much better a choice did than expected equals a case worked by hand that contains a death in mid-match (the dead row carries reward and value, no choice), a match ending and the next one starting, and the clock;
- the memory vector is zero at the first decision of every match, and recomputing it from where a stored stretch began reproduces it;
- a head that was left out receives no learning signal at all;
- an empty roster place counts in nothing;
- one small known task is solved before Breach is tried.

**What a stored decision holds,** so that these can be tested: the
observation as served; the five choices; the masks as they were; which heads
counted; the probability of each choice; the memory vector where the stored
stretch began; the five reward numbers; the two row flags.

**Order of work at rung 2** (the fourth critic's advice, taken): the shooting
range is trained without the memory, so the trainer is first proven on the
simplest form; the memory is switched on for the duel.

## 9. Unit state this design adds (for the residency patch P3a; checked against the code at `d8d56607`)

The first draft listed only what this design *adds*. The third critic showed
that this is too little for P3a: the observation, the team display and the
masks also *read* unit state that exists today but is in no digest, and if
P3a left any of it on the host side, the device backend could not build the
observation. So the table is both lists.

**State this design adds**

| state | where | P3a |
|---|---|---|
| the order: what, where (a room or an opening id), the tick it runs to, the priority target's id | new, patch 1f | four columns, hashed only while set |
| the sight rule of the match (3.6) | new, the match file | one number per match, not per unit |

**State that exists and that this design reads** (so it must be on the device)

| state | where today | in the digest today | P3a |
|---|---|---|---|
| charges carried, and which charge (`has_explosive`, `charge_ammo`) | #193 | no | a column; the charge as its table row's index |
| a plant under way (`plant_done_tick`, `plant_fuse_ticks`) | #193 | only while planting | columns |
| the planted charges (`Simulation.planted_charges`: tile, owner, detonation tick, charge row, the planter's tile) | #193; a list on the simulation, **not unit state** | only while one lives | a small table of its own, with whole-number codes (a row index for the charge, a set value for "no timer" and "no planter's tile") and a fixed capacity |
| the magazine (`current_mag`, `spare_mags`, `reload_done_tick`) | unit | no | columns |
| grenades by kind (`grenade_counts`) | unit | no | a fixed block |
| a throw being wound up (`throw_release_tick`, its kind, fuse and reach) | unit | no | columns |
| cool-downs (the last blow, shove and shot ticks) | unit | no | columns |
| the held verbs (the move direction and throttle, the trigger, the aim) | unit, intent state | no | columns: the team display reads them |
| a zombie's waking (`zombie_activated`) and its noise state | unit | the noise state only | columns (4a) |
| the kit (weapon, melee row, loadout, class) | unit, fixed per unit | no | table indices |

**Not unit state, and not synced**

| what | where it lives |
|---|---|
| last-seen records; the direction and age of a unit's last hit; damage over the last second | the Team picture (patch 1d): memory that is updated on every tick, not only at a decision. It cannot be rebuilt from a snapshot, only by playing from the start. On the device backend it is that system's own arrays. |
| which openings of the room graph can be walked now | worked out from the walls and doors (5.7) |
| the decoder's held aim | the `policy` controller |
| reward, the full view, the networks | outside the game |
| the stance; an object carried; dragging; a downed state | arrive with Erik's verbs (2.8); the schema extends |

So P3a needs from this design: every row of the second table as a device
column, the planted-charge table, and the four order columns.

## 10. Systems

**Existing systems this design must use** (rows of the project's
canonical-systems table), beyond the roadmap's section 7:

- **Planted charge** (`Simulation.plant_charge_intent`, `charges.py`): the plant choice is this verb and nothing else; its mask calls the verb's own gate.
- **Credited cause** and the env's counters (`rl_env.COUNTER_NAMES`): the own part of the reward is built from the credited hit and kill events; a new term is a new counter generated beside the old ones.
- **Unit health:** a bar is the health a unit had when it came into being on its side (6.1), noted once by the environment; `stats.effective_vitality` agrees with it for a marine and not for a turned zombie (section 11).
- **Match config and `MatchRules.result`:** the outcome group reads the match's own result, never a second rule for who won. The sight rule of 3.6 is a key of the match file, written and read through the match config's one writer and one door.
- **Level data layer and the ship generator** (`level_lib`, `level_loader`, `gen_ship`'s `ShipGraph`): the room graph is written and read through them (5.7).
- **Sight** (`vision.ray_clear`, `GameMap.has_los`): the many-tiles routine of 3.4 is the same test in bulk, never a second sight model.
- **The horde's flood** (`horde.flood`) for the compass and the route map; **Wind load on a body** for the wind planes; the **Turning law** for the time to turn onto a target; the **Speed table and Throttle** for the three speeds; `weapons.fuse_ticks_from_seconds` for every timer.
- **MatchSession + the `policy` controller:** the decoder is a step of that controller; the order verb is excluded from its action columns.
- **Batched environment:** every new per-step output (the reward groups, the full view) joins the env's own buffers.
- **Exposure, Magazines, Grenade kinds, Turning law, Unit impulse, Horde field:** the observation reads each through its own read-only seam (`mag_status`, `facing_after_turn_q16`, the horde's per-zombie state for dormant / hunting / drawn by noise), never the fields.

**New systems**, each with its draft rule (written into the project
instructions only when the code exists). The roadmap's section 7 lists Team
picture, Observation spec, Action spec and decoder, Order, Reward profile,
Trainer and Rating. This document sharpens one and adds five (the last three
named by the critique):

| system | patch | draft rule |
|---|---|---|
| Room graph | 1b | One map from tile to room, and one list of openings, written by the generator through the level writer. Orders, the commander and the bots' room goals read it; which openings are passable now is derived from walls and doors, never stored. |
| Sight for many tiles | 1c | The one sight test, run in bulk in C++ and held equal to it tile for tile. A new reader of "what can be seen from here" calls it; nothing walks its own lines. |
| Verb gates | 1e | Whether a verb may act now is one read-only function per verb, called by the verb and by the mask. A new verb brings its gate with it. |
| Reward profile | 1i | A reward is a named table of weights over counters, in the run file. The env returns one number per group for every row (dead rows included) and never the weighted sum; a new term is a new counter in a group. A term that is not credited to a unit is a change of a team's balance, read from state. |
| Full view | 1g | What the training-only value network is shown is one named spec served beside the observation by the same door; a policy never reads it. A new privileged input is a new version of that spec. |
| Network spec | rung 2 | A network is a named spec outside the game's source tree; it meets the game only through an observation spec and an action spec. |

## 11. Gameplay the design leans on, Erik's to order

Not part of this design; listed so they are not lost.

- **C4 on the continuous path:** done, #193 (2026-11-11). Left open by that patch and of interest here: the bots do not plant; the planter may walk while planting.
- **Found in the game by the critique** (2026-11-11; none is fixed here, each is Erik's to rule on):
  - a zombie spawned by the level starts with 100 hit points, a marine that turns gets 400 (`[zombie] hp` is read only at the turning);
  - a stunned soldier can still reload and can still open a door (those two verbs do not check whether the unit can act);
  - vacuum, low oxygen and water hurt nobody yet (the environment profile's limits for them are read by nothing); smoke does not block sight;
  - the bots' check for an adjacent enemy ignores walls, so a bot can shove or strike at, and react to, an enemy on the other side of a thin wall.
- **A reason to walk:** footsteps heard by speed, spread by speed (2.1).
- **The verbs of 2.8:** stance, revive, drag, pick up / put down, a general use.
- **Sight per match** as a match setting (3.6) is part of rung 1, not a gameplay patch: it adds a rule no existing match uses.

## 12. Critique log

### 12.1 The round

One lens per critic, one critic at a time, each on Opus; a verdict is on disk
(`docs/rl_agent_design_2026-11-11_critique/`) and summarised here before the
next critic starts. The critics did not see each other's findings. One
synthesis after the round, against one version of this document (`eeae95b3`).

| lens | asks | verdict file | blockers | status |
|---|---|---|---|---|
| 1. Reward | How will a learner cheat sections 6 and 4a? Do the books close? Three sides. | `lens_1_reward.md` | 0 (5 major, 8 minor) | in: survives with fixes |
| 2. Honesty and information | Does anything in sections 3, 4, 4a, 5 show a soldier what its team may not know, or withhold what the bots have? Does every feature exist in the game's state? | `lens_2_information.md` | 0 (6 major, 11 minor) | in: survives with fixes |
| 3. Systems and buildability | Is every verb, gate and field the design names real in the code? Does it build a second copy of an existing system? Is section 9 complete for P3a? Does anything from a network reach the synced game? | `lens_3_systems.md` | 0 (7 major, 11 minor) | in: survives with fixes |
| 4. Network and trainer | Do sections 2.3, 7 and 8 fit a PPO trainer: the joint probability of the choices, the masks, the memory, the dead rows, the value outputs? Do the weights add up? | `lens_4_network.md` | 0 (5 major, 10 minor) | in: survives with fixes |

Four verdicts, all "survives with fixes": no blocker, 23 major points, 40
minor. The lead re-read five load-bearing claims at the source before using
them (the two zombie health values; the hit event's unclamped damage and the
hit on a corpse; the bots' picture dropping a record at death; the
environment profile's unread limits; the cone as global config). All five
held.

### 12.2 The synthesis: every major point and what was done

"L2 M3" is lens 2's third major point. Every fix is in the section named.

| point | what the critic found | decision | where |
|---|---|---|---|
| L1 M1 | "a bar" has no single meaning for a zombie (100 or 400 hit points) | a bar is the health a unit had when it came into being on its side; capped at 1 | 6.1, 6.5, 11 |
| L1 M2 | the own part paid the full force of a hit, also on a corpse | credit only health really removed | 6.3, 6.8 |
| L1 M3 | the clock's −3 for the living only makes a late suicide pay | paid to dead rows too | 6.2 |
| L1 M4 | the environment reports the clock as a cut | reported and treated as an ending | 6.5, 6.8 |
| L1 M5, L4 M1 | one value output for the blended balance chases the moving team share | five outputs, one per number the environment returns | 7.2, 8.4 |
| L2 M1 | a record that vanishes tells of an unseen death; so does the aim mask | records live their full time unless the death was seen; the mask reads the picture; the bots move to the same picture | 2.4, 3.3, 3.6 |
| L2 M2 | the fire lanes and "the play" are the bots' controller memory | lanes read from state; "the play" dropped | 4.4, 4.5 |
| L2 M3 | "1.0 = starts to hurt me" is false for several planes | each plane with its true scale and effect; three planes reserved | 3.4 |
| L2 M4 | nothing said the map and fields are shown through walls | stated as an accepted gap; the all-seeing unit planes excluded | 3.1, 3.4 |
| L2 M5 | "whom it aims at" was undefined for an enemy and could reveal an unknown unit | one rule, judged by the watching team's knowledge, from the facing | 4.1, 4a |
| L2 M6 | orders written in hindsight carry the future into the soldier's input | a rung 4 lesson only, bounded, never in measuring matches | 5.3 |
| L3 M1 | the masks paraphrased the verbs' gates, some wrongly; a mask is one step early | a table per verb from the code; read-only gate functions; the one-sided promise | 2.4 |
| L3 M2 | a throw's wind-up spans two decisions | the aim is held through the wind-up | 2.1, 8.2 |
| L3 M3 | changing sight outright breaks the first bots' frozen match | a match setting, default the cone | 3.6 |
| L3 M4 | the order verb would become a soldier's action column, changes the recording format and the digests | excluded from the columns; new format version; hashed only while set; bots-only matches unchanged | 5.7 |
| L3 M5 | the room graph is saved nowhere and "kept current" had no home | fixed part saved with the level, changing part derived; its own patch | 5.7, 10 |
| L3 M6 | section 9 listed too little for P3a | rewritten: what is added, what is read, what is not unit state | 9 |
| L3 M7 | the cost claims need two pieces in C++ | named as requirements, measured in patch 1g | 3.4, 3.7 |
| L4 M2 | no flag tells a dead soldier from an empty roster place | two row flags | 6.3, 6.8 |
| L4 M3 | storing the map windows as they were would not fit a training batch | bytes and 16-bit numbers; the value network re-reads the soldier's windows | 3.4, 7.1 |
| L4 M4 | ratios are unbounded in a fire or a blast | one fixed squash at the door; normalising layers | 3.1, 8.3 |
| L4 M5 | the reason given for a memoryless value network was wrong | reason corrected; an experiment at rung 3; a condition for rung 8 | 7.2, 7.4 |

The minor points were taken where they are a sentence (the decoder inside the
`policy` controller, ids not list places, the wording of fuses and ranges, the
line-of-fire test, the tests added to 6.8 and 8.6, the order term's two
rules, the systems lists). Recorded as accepted gaps: friendly fire through
the world (6.3), the blast reach ignoring walls (3.3), a legal choice that
does nothing (2.4), the map shown through walls (3.1).

**Kept against a critic,** because Erik had ruled: an enemy charge shows its
time left when seen (3.3); a zombie's entry shows that it walks toward a
noise (4a).

**Left to the rung that owns it:** how values are normalised across five
outputs, the batch size and its memory budget, and whether learning keeps
pace with collecting (rung 2, the trainer's design).

### 12.3 What changed in sections Erik had already agreed (AGREED, Erik 2026-11-11: "yes i agree")

1. **Sight is a setting of the match,** with today's cone as the default, not a change of the game's one rule (3.6).
2. **A last-seen record no longer vanishes when its unit dies unseen.** The bots lose that small cheat too (3.6).
3. **The physics planes:** temperature, fire, poison, tear gas, smoke, pressure and wind stay; oxygen, water and fuel vapour wait until the game makes them matter; "1.0 = it hurts" is claimed only where it is true (3.4).
4. **"Whom it aims at"** is read from the way a body faces, among units my team knows (4.1, 4a).
5. **A throw's aim is held through its wind-up** (2.1).
6. **"The play" is dropped** from the team's counts; the order carries it (4.4).
7. **The masks follow the game's real gates,** including two that look like oversights in the game (section 11).
8. **The map and the physics fields are shown through walls,** as the bots and the screen have them; only units, grenades and charges obey sight (3.1). This was true of the design before and is now said.

## 13. Amendments after rung 1

### 13.1 The decision rate stays at 4 ticks; rung 1 is closed (AGREED, Erik 2026-11-14)

**What rung 1 measured** (patch 1j, `docs/rl_agent_rung1_note_1j_2026-11-13.md`,
"As built"). No network was involved. The layered bots played their free
twins; one team's commands had to pass through action spec v1 (five choices,
one decision every 4 ticks, held in between). Equal play is 0.50 of decided
matches; the lead's pass mark was 0.40.

| | result |
|---|---|
| held-out seeds, four cells | 0.368, 0.316, 0.373, 0.271 |
| a decision every 4 / 6 / 8 ticks (200 duels each) | 0.358 / 0.320 / 0.275, each +- 0.033 |
| 8 directions instead of 16 | 0.381 (within noise) |
| the trigger re-issued every tick | 0.389 (within one standard error) |
| against the horde alone | as good as the free bots (the freighter: better) |
| anything masked sent | 0 in 1600 matches |

**A correction.** The note says a straight line through the three rates puts
one decision per tick "near 0.48". The least-squares line is
`-0.021 +- 0.012` per tick, which gives **0.40 +- 0.05 at 2 ticks and
0.42 +- 0.06 at 1 tick** (the note took the drop per 2 ticks as the drop per
tick). Neither rate was run. So the bots' loss through the door is about 14
points at 4 ticks, of which a faster rate would buy back 4 to 6 by
extrapolation.

**What the finding is.** The choices themselves cost nothing measurable. What
costs is reaction: a soldier behind the door answers anything new 0 to 3
ticks (up to 125 ms) later than a bot that acts on every tick. Which wait
costs most (a new enemy, the trigger, a turn in the path) was not separated.
Whether a LEARNED soldier loses what the bots lose is unknown: the bots were
written to act every tick, and a soldier that learns at this rate can
anticipate. Only training answers it.

**Erik's ruling, 2026-11-14.** "I think making a decision 6 times per second
is quite a lot. I don't think we shall increase it." "If we focus too much on
acting every tick, we miss what we're trying to build here: intelligent
agents that can learn how to do smart moves." On the result: "bots vs
handicapped bots: I'd expect every game to be determined by that reaction
speed. I'm surprised that we reach 36 %, and I'd say we've already passed
this test actually."

So:

1. **Section 1's rate stands: one decision every 4 ticks.** It is not raised.
   The rate remains one of rung 3's experiments (4, 6, 8 ticks), as written.
2. **Rung 1's gate is closed.** The missed 0.40 mark is a recorded finding,
   not a blocker. The mark was the lead's guess at "not helpless", set before
   any measurement; the measurement says the door is not helpless.
3. **The free bots are not slowed.** The bots through the door (`bots_door`)
   are the equal-reaction opponent, so both exist.
4. **Two yardsticks from rung 3 on.** The gate stays "beat the free layered
   bots in more than half of the decided matches"; the result against
   `bots_door` is printed beside it. A team that beats the door bots and not
   the free ones has a reflex gap, and this section is then re-opened.
5. **Not taken, and why:** a faster rate (the farm about 15 % slower at 2
   ticks and 34 % at 1 by the measured step times, an estimate; two to four
   times the learning work per game second; no surveyed system decides 24
   times a second); an early decision on an event, and more game-side
   reflexes between decisions (each a new spec version; the second touches
   the ruling that the soldier pulls its own trigger). They stay on the list
   for the day item 4 re-opens the section.

### 13.2 Sixteen directions stay (the lead's; told to Erik 2026-11-14, his to veto)

Erik asked whether 8 directions would make the network's job easier. A head
of 25 options against 49 is no real saving for a network, and nothing
measured says 8 learns faster; 16 is what the spec, its tests and the 600
records are built on, and Erik's eye prefers it ("16 is very smooth, 8 you'll
see that it looks a little not so nice"). Eight is the same table with every
second direction switched off (`stride`), so it stays a cheap experiment if
walking turns out slow to learn.

### 13.3 Rung 2 starts by copying the bots (AGREED, Erik 2026-11-14: "I lean on copying first too")

The first network is trained to predict the door bots' recorded choice from
the recorded observation (imitation), before any learning from reward. The
reasons: it is the simplest first test of network v1, its masks and its
pointer heads (a network that cannot copy has a fault that reward learning
would hide); no game runs while it trains; and it gives a network-driven
fireteam to watch early. At best it plays like `bots_door`. Reward training
then starts from the copied network AND from scratch, side by side, so that
"does copying help or hurt" stays the experiment the roadmap asked for. The
known risk: the bots' choices depend on their inner state (a planned path, a
committed position), which the observation does not show, so the copy will
be imperfect, and habits may have to be unlearned.

The records (`C:/tmp/breach_1j/records`, 600 matches at 4 ticks, 16
directions, observation spec v1) are cheap to make again (about ten minutes
of tournament) and are not precious: a new observation spec version, or a
change of how the bots play, means recording again.

**Training does not start yet** (Erik, 2026-11-14): the unit code is being
changed by the residency arc's P6 (#29, `docs/unit_design_2026-11-14.md`).
That design asks two things of the next observation spec version (its
sections 3 and 9): whether a soldier is shown its own and its team-mates'
hooks, and fixed slot blocks for statuses and items so that a table row no
longer changes a layout. Both are open, the lead's to propose.
