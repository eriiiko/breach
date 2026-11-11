# RL soldier agent: the design (rung 0)

Started 2026-11-11. **DRAFT, every section now written** (sections 6, 7 and 8
added on branch `192-rl-agent-design`, delegated by Erik to the lead; the C4
verb of #193 folded into sections 2, 3, 4 and 9). Every section is marked
AGREED (Erik said yes) or PROPOSED (the lead's proposal, open to veto). The
critique passes are logged in section 12.

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
- **Rate: one decision every 4 ticks** (six a second, 167 ms; OpenAI Five's reaction time was 167 to 267 ms, Honor of Kings acts every 133 ms). Between decisions the choices are held, and the decoder (2.2) keeps issuing verbs on every tick. The rate is one of rung 3's experiments (4, 6, 8 ticks).

## 2. Action spec v1 (AGREED, Erik 2026-11-11: "your action space is blessed")

### 2.1 The choices

| choice | options |
|---|---|
| **Feet: direction** | stand, or one of 16 compass directions |
| **Feet: speed** | walk, half, full |
| **Aim** | an entry of the list that can be aimed at (an enemy the team sees now, or a last-seen record); or one of 16 compass directions; or "where I walk" |
| **Trigger** | hold, or release |
| **Hand** | nothing, or one verb the unit has and may use now: reload, melee, shove, use (a door), throw one hand-grenade kind (one entry per kind in the grenade table: frag, smoke, flash for today's rifleman), plant the charge it carries (#193) |
| **Hand argument** | the one number the chosen hand verb takes. For a throw, its reach: "onto my aim target" (when the aim is a list entry), or a quarter, half, three quarters or all of the grenade's range. For a plant, its timer: the shortest the charge's table row allows, the longest, or one of two steps between (today 5, 11, 26 and 60 s). The other verbs take none. |

Notes:

- **Strafing needs no choice of its own.** It is what happens when the feet and the aim point different ways. The game's speed law already makes sideways and backward steps slower, and gives full speed only when running the way the body faces.
- **A throw goes where the soldier aims.** The game already releases a grenade along the body's facing at the end of the wind-up, so the throw has no separate direction: the soldier aims, chooses the kind and the reach, and the grenade leaves after the wind-up (0.3 s today).
- **"Where I walk"** is the game's own rule for a unit that holds no aim: its travel direction becomes its facing target.
- **The three speeds, in Erik's words:** walk is the quiet pace with the steadier aim, as walking in Counter-Strike; half is the normal pace; full is the sprint. In the game today the three differ in speed only: a footstep makes no noise and the weapon's spread does not depend on how fast the soldier moves. Until the game gives the slow pace a benefit, a learning soldier has no reason to choose it. Footsteps heard by speed and spread by speed are a gameplay patch of Erik's to order (section 11).
- **Why three levels and not a continuous speed.** The game already takes any speed: the throttle is a whole number from 0 to 65 536, which is how the pad's stick feeds it, and no float enters the game. The limit of three is the soldier's, not the game's: networks learn a choice among a few levels far more easily than a continuous number (survey B.7). A finer set of levels is a later spec.
- **Reach is a fraction of the table's range**, never a number of tiles (skeleton line 8).
- **The plant, as #193 built it** (added 2026-11-11, PROPOSED until Erik has read it). The charge goes down at the soldier's own feet, on the tile it stands on when the plant ends. Planting takes a second (a table value). During it the soldier cannot start a throw or a melee blow, but it may walk and shoot; dying or being stunned plants nothing and keeps the charge. The timer is the verb's one argument, held by the game inside the charge row's range (5 to 60 s today). The blast hurts everyone in reach, the planter included. The human steps the timer in twelve 5 s steps; the soldier gets four, spaced by a constant factor between the row's shortest and longest, so the spec holds no number of seconds. A finer set is a later spec, as for the speeds.
- **Size.** 49 foot choices, about 17 + the list's aimable entries for the aim, 2 for the trigger, about 8 hand verbs, 5 argument values.

### 2.2 How a choice becomes verbs, tick by tick (the decoder)

A small integer layer beside the `policy` controller. On every tick of the
decision it issues the game's existing verbs:

- **Feet:** `set_move_dir` with the direction's unit vector and the speed's throttle, or `clear_move_dir` for "stand".
- **Aim:** `set_aim` with the direction from the soldier to the chosen entry **as it is on this tick**: to the unit's current position while the team sees it, else to its last-seen position. So the body keeps turning toward a moving enemy between decisions. A compass direction is its fixed vector; "where I walk" is the zero aim.
- **Trigger:** `set_trigger`.
- **Hand:** the chosen verb, once, on the decision's first tick. A throw carries the kind, the kind's default fuse and the reach. A plant is `plant_charge_intent` with the chosen timer as a whole number of ticks, converted once from the charge row's range.

Nothing new enters the game: the same verbs, through the same tap, so every
match replays as before. No lead is computed: the carbine's round crosses its
whole range inside one tick.

### 2.3 What depends on what

The feet are chosen on their own. The aim is chosen first among the rest; the
trigger, the hand verb and the hand argument are then chosen knowing which aim
was picked (the network hands the chosen entry's embedding to those heads).
The hand argument is chosen last and also knows the hand verb, since a reach
means something only for the grenade kind being thrown and a timer only for a
plant (added with section 8; both MOBA systems kept this same one dependency,
the "how" on the "what"). Reason: whether to fire, and whether and how far to
throw, depend on the target; nothing else in the survey's systems needed more
dependency than that (survey B.8.3).

### 2.4 The masks

Every option carries a legal / not legal flag for this decision, computed by
the game's own rules, never by a copy of them:

- an aim entry: the unit is alive and in the team picture;
- a hand verb: the unit can act (not stunned, not knocked down, not winding up a throw); it carries the grenade kind; a spare magazine exists and the gun is not full; the melee and shove cool-downs have run; a door is in reach;
- the plant: the unit carries a charge, can act, and is neither planting nor winding up a throw. While a plant is under way, the throws and the melee blow are masked too, because the game refuses them then;
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
- **Weapon and kit:** rounds left in the magazine, magazines left, reloading and the time left; grenades by kind, a throw being wound up and the time left; charges carried, a plant under way and the time left (#193); the melee and shove cool-downs; what it is (class) and what it carries (the kit's table values as ratios: range, rate of fire, spread).
- **The air where it stands:** pressure, oxygen, temperature, smoke, fire.
- **What just happened to it:** the direction its last hit came from, how long ago, and by what kind of cause.
- **Its own previous choices,** and the time left in the match.

### 3.3 The list

| entry type | cap | each entry shows |
|---|---|---|
| a team-mate | all | where it is, facing, velocity, health, class, reloading / winding up a throw / stunned / blinded / down. What it intends is section 4. |
| an enemy the team sees now (marine or zombie) | 16 | where it is, facing, velocity, class, its health and its focus-fire level (as the bots read them), what it is doing (firing, reloading, winding up a throw, stunned, down). **For me:** do I see it myself or does a team-mate; is my line of fire to it clear; is a team-mate in that line; how long until my body could be on it; is it in weapon range, in throw range; is its facing on me. |
| a last-seen record | 8 | where it was last seen, how long ago, which way it was moving, its class |
| a grenade in the air or on the floor; a planted charge | 6 | where it is, its velocity, its kind, whose it is, the time left; for anything that blasts, how far its blast reaches and whether I stand inside that reach. A charge is shown when a team-mate has a line of sight to its tile, an enemy's with its time left like an enemy's grenade (PROPOSED with #193; what a soldier may know of an enemy's timer is Erik's to veto). |
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

## 4. The team display (AGREED, Erik 2026-11-11: "I think we got it now. I have nothing to add.")

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

### 4.8 Asked of Erik

"If you were the soldier, what would you want to know about your squad that
is not in this picture?" Nothing to add (2026-11-11).

## 4a. Sides: zombies and more than two teams (PROPOSED, the lead's reading of Erik 2026-11-11)

Erik: three sides are more interesting than two. A symmetric match is hard to
win consistently; the zombies bring chaos, and it can pay to ambush the team
that is busy killing zombies, or to get the zombies to work for you.

- **Zombies are in everything above already:** entries in the list with their class, in the far window, in the team's counts, and in every training match from rung 3.
- **Every enemy entry says which side it is on:** the horde, or which of the other marine teams. The env already takes any number of teams.
- **Every enemy entry says whom it is aiming at:** me, a team-mate, another side, nobody. The same rule as for team-mates (4.1). This is what makes "they are busy with the zombies" visible.
- **A zombie's entry says whether it is dormant, hunting, or walking toward a noise.** Zombies come to noise, and a soldier who can see that can use it.
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
in fact did (roadmap rung 4).

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

- **The room graph saved with the level.** The generator has it and today writes only the spawn rooms. Rung 1: write rooms and openings through the level writer, and keep the graph current when a wall is destroyed or a door sealed.
- **The order as unit state,** set through one verb by whoever commands: what, where (a room or an opening id), until, the priority target's id.

## 6. Reward profile v1 (PROPOSED, the lead's, delegated by Erik 2026-11-11; open to veto)

The reward is the number a soldier is handed after each decision. Learning
changes the network so that the sum of rewards over a match grows. So the
reward is the only place where we say what we want; everything a soldier
learns, it learns because this number moved.

### 6.1 Rules

- **Winning is the objective.** Every other term exists to help a new network find its way to winning, is small beside the win, and is turned down once the team wins without it.
- **A helper term is the change of a score that can be read off the game's state.** Then no loop can farm it: whatever a loop gains on the way out, it gives back on the way home. (The known result behind this is Ng, Harada and Russell 1999: a reward built as the change of such a score does not change which behaviour is best, only how fast it is found. It is also what we chose for Civulator's city-value reward in October.)
- **Few terms.** Every term is something a learner will try to cheat. What is deliberately not rewarded is listed in 6.4.
- **No game number.** Health is counted in bars: 1 bar is the unit's full health, whatever the tables say today.
- **A named spec.** A reward profile is a table of weights in the run file, like an observation spec. The environment hands back each group of terms as its own number, and the trainer does the weighting, so a weight can be changed or swept without touching the game.

### 6.2 The terms

All sizes are placeholders, to be measured (6.9).

| group | term | size | who gets it | in plain words |
|---|---|---|---|---|
| **outcome** | the team wins | +10 | every soldier of the team, alive or dead | it is the last marine team with a living soldier |
| | the team is eliminated | −10 | every soldier of the team | its last soldier died or turned zombie, whoever did it: an enemy, the horde, a fire, a vacuum |
| | the clock runs out | −3 | every soldier of each team still alive | nobody won. Worse than nothing, so that hiding until the end is not a plan; far better than losing, so that a hopeless charge is not one either |
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

- two marine teams' balances mirror each other, so a one-for-one trade is worth nothing and only the outcome settles it;
- a healed enemy takes back exactly what hurting it paid;
- a soldier bitten to death costs its team its health and its half point, and then a little more, because the horde has grown by one.

**Friendly fire in plain numbers.** Killing an enemy from full health earns
1.5. Killing a team-mate from full health costs the shooter 5, on top of the
team's own loss of 1.5. So a burst through a team-mate is worth firing only
when it is about three times as likely to hit the enemy as the friend. The
blame is the shooter's alone and is never shared out (6.3). A soldier's own
grenade or charge hurting itself is ordinary lost health, not friendly fire.

### 6.3 Own against team

Who is paid for a change of the balance? Two honest answers:

- **own part:** the pieces the game credits to me. The health I took off an enemy or a zombie, my kills, and the health I lost myself. The game already credits every hit and kill to a unit.
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

**A dead soldier stays on the books.** Its row keeps receiving its team's
share and the outcome until the match ends; it makes no more choices. So
dying for a win is worth more than dying for a loss, and the value network
(section 7) learns what a soldier's death was worth.

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

- **The win outweighs the rest.** On `ship_demo` (4 against 4, 4 zombies) everything a whole team can earn from the balance is 4 x 1.5 + 4 x 0.25 = 7, shared by four soldiers. The win is 10 to each.
- **The horde is worth less than the enemy.** On a ship with many zombies the zombie weight is lowered in that run's profile so that this stays true.
- **Time.** A reward that lies in the future counts a little less than one now; how fast it fades is a trainer number (the discount). At first a soldier looks about half a minute ahead. That is why the balance terms are needed at all: early in a five-minute match the outcome is too far away to steer by. The look-ahead is lengthened as training settles (rung 2's to set and sweep).
- **The clock is part of the game.** A soldier sees the time left (3.2), so the clock running out is a real ending with its −3, not a cut the trainer has to guess across.

### 6.6 Three sides (the reward's half of section 4a)

- **The horde** is in the balance at a low weight and in nothing else. A team wiped out by zombies is eliminated like any other.
- **Three marine teams:** every other team's marines are enemies in the balance, at the same weight. The last team standing wins. A team that is eliminated gets its −10 at that moment and its soldiers' records end there. Two of three teams lose, so the average outcome is below zero; nothing has to add up to zero.
- **When two enemies fight each other,** my team's balance rises without my doing anything. That is correct, since my position did improve, and it is what makes waiting for them to wear each other down worth learning. Whether it also teaches hiding is measured by the share of matches that reach the clock (6.9).
- Ganging up on the strongest team, or sparing one enemy to keep another busy, is not designed for or against. It is watched for.

### 6.7 The later obedience term (rung 4; weight 0 until then)

Reserved now so that the value network has its output from the start:

- **progress:** the route to the order's place got shorter (the compass's own number, 3.5). Walking back pays it back, so pacing earns nothing;
- **completion:** the order's condition was met in time (arrived; held until the time ran out);
- **personal,** never shared: my order, my obedience;
- **a weight that is swept.** Obedience is a dial. Tencent's soldiers kept their strength while completing about four orders in five, and lost it when pushed toward five in five, because an order is sometimes wrong. That is rung 4's gate;
- **orders are blanked in a share of the matches,** so a soldier also learns to fight with none (AlphaStar blanked its strategy input one time in ten).

### 6.8 What it needs (patch 1e) and how it is tested

Today's counters (survey A.5) count hits and raw damage, do not tell a marine
from a zombie, and hold neither the outcome nor anything about the team.
Patch 1e adds, and keeps every old counter working:

- **per soldier per decision:** health in bars taken off enemy marines, off zombies, off team-mates; kills of each; its own health lost; its own death or turning;
- **per team per decision:** the change of its balance, computed from the state before and after; the outcome when it falls;
- **from the environment:** one number per group (outcome, own balance, team balance, friendly fire, order) for every row, dead rows included.

The reward is computed outside the synced game, from state and credited
events only, so it changes no match and no replay.

Tests, each a property:

- the books close: a team's balance changes over a whole match add up to its balance at the end minus its balance at the start;
- own parts plus the uncredited rest equal the balance change, on every decision;
- healing an enemy takes back exactly what hurting it paid;
- a win or an elimination is paid to every row of the team exactly once, the dead rows included;
- friendly-fire blame lands on the credited unit and on no other;
- the same match replayed gives the same rewards.

### 6.9 Experiments, each able to fail

| question | runs | read off |
|---|---|---|
| Do the balance terms help at all? | outcome only, against outcome + balance | does anything learn within a day |
| Own against team | team share 0, 0.3 to 0.8, 1 | wins; and enemy losses that the horde caused (does luring appear) |
| How hard to punish friendly fire | the weights at 0, as written, tripled | friendly hits per match **and** rounds fired per match: a soldier that stops shooting has not solved friendly fire |
| Can the helper be removed? | balance weights cut to a quarter, and to zero, after rung 3's gate | is strength kept |
| What the clock is worth | −3 against 0 and −10 | the share of matches decided; and Erik's eye on how the matches look |

Rung 2 uses cut-down profiles of the same table: **"range"** (a target loses
health; the clock) and **"duel"** (outcome and balance, one soldier, so no
team share and no friendly fire). The table above is **"fireteam"**.

## 7. What the training-only value network sees (PROPOSED, the lead's; open to veto)

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
| the soldier's own observation | all of it | the same arrays, unchanged |
| units | team-mates; enemies its team sees; last-seen records | every living unit of every side where it truly is, dormant zombies included; each marked "my team sees it", "my team has a record of it" or "my team does not know of it" |
| per unit | what the bots may read (3.3) | also its magazine and magazines left, grenades by kind, charges, cool-downs, every status with its time left, its order, and the air where it stands |
| grenades and charges | those its team has a line of sight to | all of them, with owner and time left |
| the match | time left; its team's counts (4.4) | also every team's living count and balance; zombies awake and dormant |
| the map | the two windows; the sight planes drawn from known enemies | the same windows; the "watched" plane drawn from every enemy |

**What it does not see:** the future; the match's random seed; the memory
inside any soldier's network.

### 7.2 Choices made, and why

- **The soldier's own observation is part of the input.** MAPPO measured that the best input is the whole state plus the soldier's own view with the duplicates removed.
- **A list, read the same way as the soldier's.** Up to 64 entries: every marine, then the nearest zombies.
- **No memory.** It sees the whole state, so it has nothing to remember. That leaves the trainer's most error-prone part (carrying memory across a match) on the soldier's side only.
- **A separate network.** It shares no weights with the soldier's network, so nothing it sees can leak into a choice. This is tested (8.5).
- **A dead soldier's view** keeps the whole match, with its own part replaced by "dead, since when". MAPPO blanked the view of a dead soldier; our dead soldiers still receive a team share (6.3), so their estimate has to follow the match. Both are run (7.4).
- **One output per reward group** (outcome, balance, friendly fire, order). The trainer adds them with the profile's weights. A weight can then be moved during a run, the order term can be switched on at rung 4, and the estimate needs no rebuilding. Honor of Kings ran five such outputs and reports that it raised strength.

### 7.3 How much is hidden today

With all-round shared sight (3.6), what the soldier cannot see is: enemies no
team-mate has a line of sight to, dormant zombies, what the enemy carries and
has been ordered, and charges out of sight. So the full view buys less now
than it will with cones at rung 8, where it becomes essential. It is built
now because adding it later would change the trainer and the environment's
buffers in the middle of the ladder.

### 7.4 Built where; tests; experiments

- Built by the environment beside the observation (patch 1b), into the farm's own buffers. The farm's speed with it is measured.
- Test: the full view equals an independent reconstruction from the game state; every unit alive in the game is in it.
- Experiments (rung 3): the full view against the soldier's own view only; the dead soldier's view kept against blanked.

## 8. Network v1 (PROPOSED, the lead's; open to veto)

About 1.5 million weights for the soldier, inside the 1 to 2 million asked
for, and about 1.1 million for the value network, which is never shipped.
Sizes are the starting point of a sweep.

### 8.1 The soldier's network, part by part

| part | what it does | size | weights |
|---|---|---|---|
| **vector reader** | reads the soldier's own numbers, its order and the team's counts (3.2, 4.4, 5.2) | about 130 numbers in, two layers, 128 out | 66 000 |
| **entry readers** | one small layer per entry type (team-mate, enemy, last-seen record, grenade or charge, door, one spare), so every entry becomes a vector of the same length | up to 48 numbers in, 128 out | 38 000 |
| **list attention** | two layers in which every entry looks at every other and at the soldier itself | width 128, 4 heads | 265 000 |
| **list summary** | the soldier's own vector asks the list one question and gets one vector back | width 128 | 66 000 |
| **near-map reader** | four convolution layers over the 25 x 25 near window (about 17 planes) | 32, 64, 64, 64 channels; 128 out | 229 000 |
| **far-map reader** | the same over the far window (about 18 planes) | the same | 229 000 |
| **join** | puts the four results together | 512 in, 256 out | 131 000 |
| **memory** | a GRU: a vector carried from one decision to the next | 256 | 395 000 |
| **heads** | the choices of section 2 | see 8.2 | 135 000 |
| | | **total** | **about 1 550 000** |

### 8.2 The heads

- **Feet:** one choice among 49 (stand, or 16 directions at 3 speeds).
- **Aim:** by pointing. The network forms a question vector; every entry that can be aimed at answers with its own vector from the list attention; the 16 directions and "where I walk" answer with learned vectors of their own. The best match wins. Entries that are not legal are masked.
- **Trigger, hand verb, hand argument:** chosen after the aim, each told which aim was picked (2.3). The hand verb is pointed at as well (2.5): each verb the unit has is a learned vector. The argument is chosen last and is told the verb.
- **What is left out of the learning signal:** a choice that had no effect. The argument when the verb takes none; every head on a soldier that could not act. (OpenAI Five did the same for unused arguments.)

### 8.3 Why these parts and these sizes

- **Attention for the list** is the survey's firmest result (rule 1). Two layers at width 128 for about 40 entries: hide-and-seek used one layer, AlphaStar three for 512 entities.
- **Two small convolution readers for the map.** Where a wall stands relative to the soldier matters, so the readers keep position until their last layer. They are deliberately small: the game hands over what a small network infers badly from a map (the line-of-fire flags, the sight planes, the compass).
- **A memory, from the start.** With shared all-round sight and the game's own last-seen records, a soldier needs little memory today. It is in v1 because retrofitting memory into a trainer is the expensive direction, and rung 8's cones cannot work without it. A GRU because the one broad comparison of memories found it the best general choice. Rung 2 runs the duel with the memory on and off.
- **The size.** It comes to about 1.5 million by itself, hide-and-seek's size. The farm produces a few thousand decisions a second, and a network of this size is many times faster than that on the desktop's GPU, so it never waits on the network. It is not smaller, because our Civulator ladder showed a too-small network hiding the effect of four changes in a row. Half and double the width are among the first sweeps.
- **Nothing learned about the scale of numbers.** Every input is already a ratio, a flag, a cosine or a learned vector per kind (3.1), so the network needs no running statistics of its inputs, one more thing that cannot go wrong.

### 8.4 The value network

The same readers with its own weights, fed the full view of section 7; no
memory; a final layer with one output per reward group. About 1.1 million
weights.

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
- the probability of a recorded choice is the same when it is recomputed for learning as when it was made;
- shuffling the list's entries changes no output, except that the pointer's answer moves with its entry;
- an empty list slot changes nothing;
- scrambling the value network's extra inputs changes no choice;
- the number of weights is printed and matches the spec.

## 9. Unit state this design adds (for the residency patch P3a; checked against the code at `d8d56607`)

| state | from | status |
|---|---|---|
| the charges carried (`Unit.has_explosive`, a count) and which charge they are (`charge_ammo`, a loadout constant) | #193 | **built** |
| a plant under way: the tick it completes and the timer it was given (`plant_done_tick`, `plant_fuse_ticks`; in the unit digest only while a plant runs). The latched verb (`pending_plant`) is intent state and is not hashed. | #193 | **built** |
| the planted charges themselves. **Not unit state:** a list on the simulation (`Simulation.planted_charges`: tile, owner, detonation tick, charge row, the planter's tile, live), hashed beside the units only while one exists. P3a must carry it as a small table of its own next to the unit schema. | #193 | **built** |
| the order: what, where (a room or an opening id), the tick it runs to, the priority target's id | section 5 | proposed; patch 1d |
| (none for the aim) | the decoder is controller-side; "whom a unit is turning toward" is already unit state, the facing target | — |
| (none for the team picture) the last-seen records, the direction and age of a unit's last hit, the damage it took in the last second | 3.2, 3.3, 4.3 | not unit state: they live in the Team picture (patch 1a), read from the tick's events; on the device backend they are that system's own arrays |
| (none for sections 6, 7, 8) | reward, the full view and the networks are computed outside the synced game | — |
| the stance; an object carried; dragging / being dragged; a downed state and its timer | 2.8 | when Erik builds each verb; none is needed at P3a, whose schema extends |

So P3a needs from this design: the #193 fields as they stand in the code, the
planted-charge table, and four order fields.

## 10. Systems

**Existing systems this design must use** (rows of the project's
canonical-systems table), beyond the roadmap's section 7:

- **Planted charge** (`Simulation.plant_charge_intent`, `charges.py`): the plant choice is this verb and nothing else; its mask calls the verb's own gate.
- **Credited cause** and the env's counters (`rl_env.COUNTER_NAMES`): the own part of the reward is built from the credited hit and kill events; a new term is a new counter generated beside the old ones.
- **Unit health** through `stats.effective_vitality`: the one source of "a bar".
- **Match config and `MatchRules.result`:** the outcome group reads the match's own result, never a second rule for who won.
- **Batched environment:** every new per-step output (the reward groups, the full view) joins the env's own buffers.
- **Exposure, Magazines, Grenade kinds, Turning law, Unit impulse, Horde field:** the observation reads each through its own read-only seam (`mag_status`, `facing_after_turn_q16`, the horde's per-zombie state for dormant / hunting / drawn by noise), never the fields.

**New systems**, each with its draft rule (written into the project
instructions only when the code exists). The roadmap's section 7 lists Team
picture, Observation spec, Action spec and decoder, Order, Reward profile,
Trainer and Rating. This document sharpens one and adds two:

| system | patch | draft rule |
|---|---|---|
| Reward profile | 1e | A reward is a named table of weights over counters, in the run file. The env returns one number per group for every row (dead rows included) and never the weighted sum; a new term is a new counter in a group. A term that is not credited to a unit is a change of a team's balance, read from state. |
| Full view | 1b | What the training-only value network is shown is one named spec served beside the observation by the same door; a policy never reads it. A new privileged input is a new version of that spec. |
| Network spec | rung 2 | A network is a named spec outside the game's source tree; it meets the game only through an observation spec and an action spec. |

## 11. Gameplay the design leans on, Erik's to order

Not part of this design; listed so they are not lost.

- **C4 on the continuous path:** done, #193 (2026-11-11). Left open by that patch and of interest here: the bots do not plant; the planter may walk while planting.
- **A reason to walk:** footsteps heard by speed, spread by speed (2.1).
- **The verbs of 2.8:** stance, revive, drag, pick up / put down, a general use.
- **Sight per match** as a match setting (3.6) is part of rung 1, not a gameplay patch: it adds a rule no existing match uses.

## 12. Critique log

One lens per critic, one critic at a time, each on Opus; a verdict is on disk
(`docs/rl_agent_design_2026-11-11_critique/`) and summarised here before the
next critic starts. The critics do not see each other's findings. One
synthesis after the round, against one version of this document.

| lens | asks | verdict file | blockers | status |
|---|---|---|---|---|
| 1. Reward | How will a learner cheat sections 6 and 4a? Do the books close? Three sides. | `lens_1_reward.md` | 0 (5 major, 8 minor) | in: survives with fixes |
| 2. Honesty and information | Does anything in sections 3, 4, 4a, 5 show a soldier what its team may not know, or withhold what the bots have? Does every feature exist in the game's state? | `lens_2_information.md` | 0 (6 major, 11 minor) | in: survives with fixes |
| 3. Systems and buildability | Is every verb, gate and field the design names real in the code? Does it build a second copy of an existing system? Is section 9 complete for P3a? Does anything from a network reach the synced game? | `lens_3_systems.md` | | pending |
| 4. Network and trainer | Do sections 2.3, 7 and 8 fit a PPO trainer: the joint probability of the choices, the masks, the memory, the dead rows, the value outputs? Do the weights add up? | `lens_4_network.md` | | pending |
