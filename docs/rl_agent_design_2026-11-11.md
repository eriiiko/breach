# RL soldier agent: the design (rung 0)

Started 2026-11-11. **DRAFT, growing section by section** as the discussion
settles each part. Every section is marked AGREED (Erik said yes), PROPOSED
(the lead's proposal, open to veto) or OPEN (not yet written).

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
9. C4 on the continuous path is Erik's separate patch; this design assumes its verb exists when needed.

## 1. Frame and rate (AGREED with the action space, Erik 2026-11-11; the window's frame is settled with section 3)

- **Frame: north up, centred on the soldier.** Positions in the list are offsets from the soldier along the map's own axes, and the map window does not turn with it. The soldier's facing is one of its own features. Reason: the ships' walls run along the map's axes, so a window that never rotates keeps every wall straight for the network, and a rotated window would have to be resampled and blurred. It is also the frame of the human's north-up camera.
- **Rate: one decision every 4 ticks** (six a second, 167 ms; OpenAI Five's reaction time was 167 to 267 ms, Honor of Kings acts every 133 ms). Between decisions the choices are held, and the decoder (2.2) keeps issuing verbs on every tick. The rate is one of rung 3's experiments (4, 6, 8 ticks).

## 2. Action spec v1 (AGREED, Erik 2026-11-11: "your action space is blessed")

### 2.1 The choices

| choice | options |
|---|---|
| **Feet: direction** | stand, or one of 16 compass directions |
| **Feet: speed** | walk, half, full |
| **Aim** | an entry of the list that can be aimed at (an enemy the team sees now, or a last-seen record); or one of 16 compass directions; or "where I walk" |
| **Trigger** | hold, or release |
| **Hand** | nothing, or one verb the unit has and may use now: reload, melee, shove, use (a door), throw one hand-grenade kind (one entry per kind in the grenade table: frag, smoke, flash for today's rifleman), plant a charge (when its verb exists) |
| **Throw reach** | "onto my aim target" (when the aim is a list entry), or a quarter, half, three quarters or all of the grenade's range |

Notes:

- **Strafing needs no choice of its own.** It is what happens when the feet and the aim point different ways. The game's speed law already makes sideways and backward steps slower, and gives full speed only when running the way the body faces.
- **A throw goes where the soldier aims.** The game already releases a grenade along the body's facing at the end of the wind-up, so the throw has no separate direction: the soldier aims, chooses the kind and the reach, and the grenade leaves after the wind-up (0.3 s today).
- **"Where I walk"** is the game's own rule for a unit that holds no aim: its travel direction becomes its facing target.
- **The three speeds, in Erik's words:** walk is the quiet pace with the steadier aim, as walking in Counter-Strike; half is the normal pace; full is the sprint. In the game today the three differ in speed only: a footstep makes no noise and the weapon's spread does not depend on how fast the soldier moves. Until the game gives the slow pace a benefit, a learning soldier has no reason to choose it. Footsteps heard by speed and spread by speed are a gameplay patch of Erik's to order (section 11).
- **Why three levels and not a continuous speed.** The game already takes any speed: the throttle is a whole number from 0 to 65 536, which is how the pad's stick feeds it, and no float enters the game. The limit of three is the soldier's, not the game's: networks learn a choice among a few levels far more easily than a continuous number (survey B.7). A finer set of levels is a later spec.
- **Reach is a fraction of the table's range**, never a number of tiles (skeleton line 8).
- **Size.** 49 foot choices, about 17 + the list's aimable entries for the aim, 2 for the trigger, about 8 hand verbs, 5 reaches.

### 2.2 How a choice becomes verbs, tick by tick (the decoder)

A small integer layer beside the `policy` controller. On every tick of the
decision it issues the game's existing verbs:

- **Feet:** `set_move_dir` with the direction's unit vector and the speed's throttle, or `clear_move_dir` for "stand".
- **Aim:** `set_aim` with the direction from the soldier to the chosen entry **as it is on this tick**: to the unit's current position while the team sees it, else to its last-seen position. So the body keeps turning toward a moving enemy between decisions. A compass direction is its fixed vector; "where I walk" is the zero aim.
- **Trigger:** `set_trigger`.
- **Hand:** the chosen verb, once, on the decision's first tick. A throw carries the kind, the kind's default fuse and the reach.

Nothing new enters the game: the same verbs, through the same tap, so every
match replays as before. No lead is computed: the carbine's round crosses its
whole range inside one tick.

### 2.3 What depends on what

The feet are chosen on their own. The aim is chosen first among the rest; the
trigger, the hand verb and the reach are then chosen knowing which aim was
picked (the network hands the chosen entry's embedding to those heads). Reason:
whether to fire, and whether and how far to throw, depend on the target; nothing
else in the survey's systems needed more dependency than that (survey B.8.3).

### 2.4 The masks

Every option carries a legal / not legal flag for this decision, computed by
the game's own rules, never by a copy of them:

- an aim entry: the unit is alive and in the team picture;
- a hand verb: the unit can act (not stunned, not knocked down, not winding up a throw); it carries the grenade kind; a spare magazine exists and the gun is not full; the melee and shove cool-downs have run; a door is in reach;
- speed "full": always legal, but its effect depends on facing, which the soldier sees.

Building the masks means lifting each verb's gate out of the verb into a
function that the verb and the mask both call.

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
- **Everything the bots have** (AGREED, Erik 2026-11-11). Whatever the scripted bots read to decide, a learned soldier may see: an enemy's health and focus-fire level, the team picture, the shared target, the play, the fire lanes, and what the bots know about the tiles around them (3.4).
- **Every field that acts on a unit, in that unit's own terms** (3.4).
- **No raw numbers.** A quantity is a ratio of a table value (health over full health). An angle is its cosine and sine. A kind (class, weapon, grenade kind, entry type) is a learned vector per kind, never an id number. A yes / no is a flag.
- **Positions are offsets from the soldier, north up,** as a fraction of a reference length from the tables, with the distance given again on a compressed scale so that near things are resolved finely.
- **Lists are capped, nearest first,** and an empty slot is masked out, never filled with zeros that could be mistaken for a thing at the soldier's feet.

### 3.2 Itself (one vector)

- **Body:** health; facing; how fast it is turning and how far it still has to turn to its aim; its velocity; whether it is being pushed, is catching itself, or is knocked down; stunned, blinded, burning, poisoned, suffocating, each with the time left.
- **Weapon and kit:** rounds left in the magazine, magazines left, reloading and the time left; grenades by kind, a throw being wound up and the time left; charges carried (when C4 exists); the melee and shove cool-downs; what it is (class) and what it carries (the kit's table values as ratios: range, rate of fire, spread).
- **The air where it stands:** pressure, oxygen, temperature, smoke, fire.
- **What just happened to it:** the direction its last hit came from, how long ago, and by what kind of cause.
- **Its own previous choices,** and the time left in the match.

### 3.3 The list

| entry type | cap | each entry shows |
|---|---|---|
| a team-mate | all | where it is, facing, velocity, health, class, reloading / winding up a throw / stunned / blinded / down. What it intends is section 4. |
| an enemy the team sees now (marine or zombie) | 16 | where it is, facing, velocity, class, its health and its focus-fire level (as the bots read them), what it is doing (firing, reloading, winding up a throw, stunned, down). **For me:** do I see it myself or does a team-mate; is my line of fire to it clear; is a team-mate in that line; how long until my body could be on it; is it in weapon range, in throw range; is its facing on me. |
| a last-seen record | 8 | where it was last seen, how long ago, which way it was moving, its class |
| a grenade in the air or on the floor; a planted charge | 6 | where it is, its velocity, its kind, whose it is, the time left |
| a door | 4 | where it is, open or closed, within reach or not |
| reserved | | a downed team-mate, a bag, a button (2.8); a noise heard (rung 8) |

- **"How long until my body could be on it"** is the trigger's aid, given for every candidate target before the choice: the turning law's own time to bring the weapon within its spread of that target. For the current aim target the soldier also sees the bots' exact prediction for the next tick.
- **The line-of-fire flags** are the bots' two rules handed over as information: a wall in the way, a team-mate in the way.
- About 40 entries at most. Attention over that many costs almost nothing.

### 3.4 The map window

North up, centred on the soldier, at two scales:

- **near:** 25 x 25 tiles at full resolution (about 8 m across), for walls, doors and cover;
- **far:** 25 x 25 blocks of 4 x 4 tiles (about 33 m across: the room, its neighbours and the corridors between), for the layout and where the fire is. Units beyond the window are still in the list.

**The physics planes: every field that can hurt, blind, push or drown a unit,
and no other.** Each is given in the unit's own terms, as a ratio of what its
environment profile tolerates, so that 1.0 means "this starts to hurt me":

| plane | scaled by |
|---|---|
| temperature | the profile's highest bearable temperature |
| pressure | the profile's lowest and highest bearable pressure |
| oxygen | the profile's lowest breathable level |
| poison gas | the dose that starts to harm |
| tear gas | the density that blinds |
| smoke | the density the bots treat as dense |
| fire | its intensity |
| wind, two planes (its push east and south) | the push that knocks this unit down |
| water depth | whether this unit drowns or swims |
| fuel vapour | the density that can ignite |

A suited marine then sees a vacuum as harmless, and a later unit class sees
the same ship in its own terms, with no change to the network's shape. The
soldier's own tile gets the same ratios in its vector (3.2). Left out is what
acts on nothing a unit can sense or use: the nitrogen, the energy books, the
blast source, and light until it is game state (#81). Erik, 2026-11-11:
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
Reason: cover and concealment are most of what a soldier does with its feet,
and line of sight is the one thing a small network infers badly from a map of
walls.

**In the far window also:** where the team-mates are, where the seen enemies
are, and the route map of 3.5.

### 3.5 The compass

For the order's place the soldier sees the direction of the first step of the
shortest walkable route and the route's length, and in the far window the same
route as a plane (steps to the place from every block). This is the distance
field the bots' team layer already computes for its objective. The feet stay
the soldier's own.

### 3.6 Sight: who may know what

- **One sight rule per match, the same for every seat in it** (bots, learned soldiers, a human). It is a setting of the match. Otherwise a win over the bots would be a win by better eyes.
- **Training matches start with all-round sight, walls block, shared across the team.** The cone the game has today (110 degrees in front, a 4-tile circle all round) returns as its own experiment at rung 8. Erik, 2026-11-11: cones are wanted, but "just to have agents playing good is more important"; reducing what they know comes after, and will cost much more computation, since the soldiers must then keep their own memory of the world; "it's fine to cheat like this a little bit now". He does not need any match played before today preserved, so rung 1 is free to change the game's sight setting outright if that is simpler than a setting per match.
- **Last-seen records** are kept as the bots keep them: where and when, forgotten after the memory time (6 s today, a setting).
- **What may be known about a seen enemy** beyond what the screen shows: its health and its focus-fire level. AGREED, Erik 2026-11-11: "let's give them everything that the bots have".

### 3.7 Cost and tests (rung 1b)

- The observation is built once per decision per soldier. "Do I see it" comes from the per-tick sight cache the game already keeps; the route map from the flood the bots already run. The farm's speed with the new observation is measured against today's.
- Test: the observation equals an independent reconstruction from the game state; no entry ever shows a unit the team cannot know of.

## 4. The team display (PROPOSED)

### 4.1 Principles

- **It is read from the game's state, never from a controller's memory.** Where a unit is turning to, whether its trigger is held, where its feet are taking it, a throw being wound up, a reload, its order: all of these are already the unit's state in the game. So a bot, a learned soldier and a human look the same to their squad-mates, and the display needs no new channel.
- **Fixed and readable.** No learned messages (roadmap rule 4; survey B.8.4).
- **The radio is perfect, for now.** Every living team-mate is shown, wherever it is on the ship. This goes with all-round sight: cheat a little now, reduce later (rung 8).
- **"Whom it aims at" is decided by one rule for everyone:** the enemy the team knows of that lies on the unit's aim line within its weapon's spread, with no wall between. Otherwise it aims at a direction.

### 4.2 On each team-mate's entry

Added to its state (3.3):

| shown | meaning |
|---|---|
| its aim | the point it is turning to, whether that is a unit, whether it is firing |
| its feet | the direction and speed it has chosen |
| a throw being wound up | the kind, where it will land, the time to release |
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
ammunition, enemies known and seen, zombies near, and the play (section 5).

### 4.5 In the near window

One more plane: **the fire lanes**, the tiles that lie in a team-mate's line
of fire. This is the bots' lane grid, handed over.

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

### 4.8 Open with Erik

What a soldier would want to know about its squad that is not in this picture.

## 5. The order (OPEN)

## 6. Reward profile v1 (OPEN)

## 7. What the value network sees (OPEN)

## 8. Network v1 (OPEN)

## 9. Unit state this design adds (running list, for the residency patch P3a)

| state | from | status |
|---|---|---|
| the charges carried and a planting in progress | the C4 patch | Erik's patch |
| the order: its kind, its argument, the tick it runs to | section 5 | to be designed |
| (none for the aim) | the decoder is controller-side; "whom a unit is turning toward" is already unit state, the facing target | — |
| the stance; an object carried; dragging / being dragged; a downed state and its timer | 2.8 | when Erik builds each verb; none is needed at P3a, whose schema extends |

## 10. Systems

As in the roadmap's section 7; this document adds the detail of each as its
section is agreed.

## 11. Gameplay the design leans on, Erik's to order

Not part of this design; listed so they are not lost.

- **C4 on the continuous path** (in work, Erik, 2026-11-11).
- **A reason to walk:** footsteps heard by speed, spread by speed (2.1).
- **The verbs of 2.8:** stance, revive, drag, pick up / put down, a general use.
- **Sight per match** as a match setting (3.6) is part of rung 1, not a gameplay patch: it adds a rule no existing match uses.
