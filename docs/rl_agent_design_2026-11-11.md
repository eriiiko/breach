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

## 1. Frame and rate (PROPOSED)

- **Frame: north up, centred on the soldier.** Positions in the list are offsets from the soldier along the map's own axes, and the map window does not turn with it. The soldier's facing is one of its own features. Reason: the ships' walls run along the map's axes, so a window that never rotates keeps every wall straight for the network, and a rotated window would have to be resampled and blurred. It is also the frame of the human's north-up camera.
- **Rate: one decision every 4 ticks** (six a second, 167 ms; OpenAI Five's reaction time was 167 to 267 ms, Honor of Kings acts every 133 ms). Between decisions the choices are held, and the decoder (2.2) keeps issuing verbs on every tick. The rate is one of rung 3's experiments (4, 6, 8 ticks).

## 2. Action spec v1 (PROPOSED)

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
- **Aiming at team-mates,** and pointing at doors or corners as entries: the compass directions cover them.
- **A route autopilot.** The soldier walks itself. What it gets instead is a compass in the observation: the direction and length of the shortest walkable route to the order's place (section 3, when written). The same principle as the trigger: the game shows the soldier what the bots know, and the soldier acts.
- **Crouch, prone, weapon swap, pick up:** no such verbs exist in the game.

### 2.7 The tests that can fail (rung 1)

The bots through the door (roadmap rung 1): the layered bots, their every
decision translated to the nearest choice of this spec, against the same bots
unconstrained. Run in these variants, so each number above is a measurement:

- 8 against 16 directions, for the feet and for the aim;
- a decision every 4, 6 and 8 ticks;
- the trigger decided per decision against per tick (how much the 4-tick trigger costs in hits);
- throws onto list entries against throws by direction and reach.

## 3. Observation spec v1 (OPEN: next)

## 4. The team display (OPEN)

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

## 10. Systems

As in the roadmap's section 7; this document adds the detail of each as its
section is agreed.
