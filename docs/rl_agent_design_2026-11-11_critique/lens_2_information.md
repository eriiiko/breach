# Lens 2: honesty and information (sections 3, 4, 4a, 5.2; the wall of section 7)

Critic: Opus, 2026-11-11, independent of the other lenses. Read-only on code; nothing run.

## Verdict

**Survives with fixes.** No blockers. The skeleton of the information design holds: shared sight from one
oracle (`vision.state_for`), the per-tick "do I see it myself" from `observers_of`, the trigger aid from
`facing_after_turn_q16`, the team display mostly from real unit attributes (`live_aim`, `live_move_dir`,
`live_trigger`, `throw_release_tick`, `plant_done_tick`, `reload_done_tick`), and a value network kept
apart by separate weights. But four things in the text are not true as written: (1) last-seen records "kept
as the bots keep them" inherit a leak, because the bots delete a record the moment its unit dies, seen or
not, and the aim mask repeats it; (2) the fire lanes and "the play" are controller memory, and for a learned
team the lane grid would be empty; (3) "1.0 means this starts to hurt me" is false for four of the planes
(pressure, oxygen and water act on no unit today; the temperature plane is not the quantity the heat row
burns on); (4) the honesty rule covers units only, while the map, doors and fields of both windows are
unfogged, and the env's existing unit planes are omniscient. Each fix is a sentence or a re-definition on
paper; none needs new machinery beyond what section 9 already names.

## Findings

### MAJOR 1. A last-seen record leaks its unit's death (and its conversion); so does the aim mask
- Claim (3.6): "Last-seen records are kept as the bots keep them." (2.4): an aim entry is legal when "the unit is alive and in the team picture".
- Evidence: `src/control_bots.py:332-337` deletes a picture entry when `not u.alive`, whatever the team saw; the survey says so too (A.7: "An enemy that dies is removed at once, seen or not"). The record holds the live `Unit` object (`Sighting.unit`, `control_bots.py:249-254`), so its class read later (`known(zombies=...)`, `:339-346`) flips marine -> zombie at an unseen conversion. The 4.4 counts ("enemies known") are built from the same picture (`:353-360`).
- Effect: a record vanishing before its 6 s, or the aim mask switching off, tells the soldier an unseen enemy died. With three sides (4a) that is exactly the "who is busy dying to the zombies" signal. 3.7's test ("no entry ever shows a unit the team cannot know of") cannot catch an entry that disappears.
- Fix on paper: the Team picture (patch 1a) keeps a record for the memory time whatever happens to its unit; it stores the class and motion AT the sighting, never a live object; the aim mask reads "in the team picture", not "alive". The bots read the same Team picture (so parity holds both ways). A record is dropped early only when the team SAW the death (a kill event on a seen unit).

### MAJOR 2. The fire lanes and "the play" are controller memory; the lanes are empty for a learned team
- Claim (4.1): read from the game's state, never from a controller's memory. (4.5): "the fire lanes ... the bots' lane grid, handed over." (4.4): "the play". (3.1): the soldier sees "the shared target, the play, the fire lanes".
- Evidence: `TeamBrain.lanes` is built from `TeamBrain.firing` (`control_bots.py:445-472`), which only `BotTeam.handle_frame` fills from each bot's executor (`:936-944`). A team of learned soldiers has no `BotTeam` executor, so with the bots' team layer as its commander (5.3) the grid is all -1. `TeamBrain.play` / `halves` / `bound_moving` (`:310-318`) live in the controller. Also the bound swap reads `bots.arrived(m)` (`:401`), which is bot controller memory (`_BotState.goal`, `:1371-1375`) and returns True for a unit with no bot state, so for a learned team the halves would swap every team period (outside this lens, noted for lens 3).
- Fix on paper: define the lane from state, for everyone: a team-mate whose `live_trigger` is held, from its centre along its FACING to the first solid tile or its weapon's range (or to its 4.1 aimed-at unit), widened by the bots' half width. The bots switch to that same definition (one source). Drop "the play" from 4.4 and 3.1: what the soldier needs of it is the order's "what" (5.1), which becomes unit state at patch 1d. Until 1d, say that the shared-target flag and the order are controller-fed (an accepted gap).

### MAJOR 3. "1.0 means this starts to hurt me" is false for pressure, oxygen, water and temperature
- Claim (3.4): every plane is "a ratio of what its environment profile tolerates, so that 1.0 means this starts to hurt me"; "A suited marine then sees a vacuum as harmless".
- Evidence: `pressure_min/max`, `o2_level_min` and `submersion` are read nowhere in `src/` (only defined, `src/simulation/environment.py:31-36,76`; its docstring `:3-6`: "Behaviour (O2 reserve drain, pressure damage, temperature damage, submersion) is deferred"); `events.py:71` lists vacuum, pressure, O2 and water as causes with no producer. Nothing hurts any unit by vacuum, low O2 or water today: every unit already "sees a vacuum as harmless", and the suffocating status of 3.2 has no producer found (UNVERIFIED beyond the grep). Heat: the heat row burns on `T_felt` from the radiation sensor `rad_flux_sweep` and the `heat` buffer (`src/simulation/exchange.py:358-397`), not on the tile `temperature` plane, so temperature / `temperature_max` = 1.0 does not mark where damage starts. Smoke and fuel vapour act on no unit (smoke does not block sight: `vision.py:90-105`, survey A.7), which contradicts the section's own "and no other" rule. Poison and tear gas thresholds are global coupling values (`poison_min_density`, `teargas_blind_density`, `exchange.py:581,626`), not profile values, and a zombie takes no poison damage (`exchange.py:618`) yet would read 1.0. Fire has no denominator ("its intensity").
- Fix on paper: one column per plane naming its real denominator and its real coupling: heat = the felt temperature from the radiation sensor over `temperature_max` (or keep the tile temperature, scaled by the bots' `hot_temp`, and stop calling it "hurts at 1.0"); poison = density over `poison_min_density` times the unit's poison resist; tear gas over `teargas_blind_density`; smoke over the bots' `smoke_dense`, kept for parity, labelled "acts on nothing, the bots avoid it"; pressure and oxygen over the profile's limits, labelled "no coupling yet, accepted gap"; water dropped until it acts or until the speed row lands; fuel vapour kept only if an ignition threshold exists (UNVERIFIED: `[gases.fuel_gas]` has `flammable = true` and no threshold key at `config.toml:1298-1308`); fire over a named row value.

### MAJOR 4. The map, doors and fields are unfogged, and the env's unit planes are omniscient; the honesty rule does not say so
- Claim (3.1): "A soldier is shown only what its team may know (3.6)". 3.6 defines sight for units only.
- Evidence: today's crop marks every living unit of every side (`src/simulation/sensor_accessor.py:295-336`; survey A.2 "It ignores line of sight"); the env requests those planes (`src/rl_env.py:29-30,623-625`). The design's far window (33 m) carries fire and the physics planes through walls; the near window and the door list carry door state; a blown wall shows in the wall plane. An unseen enemy opening a door, throwing smoke, or setting off a charge in the next room is visible through these.
- Parity note: the bots read fields at every candidate tile without sight (`_f_safe_ground`, `control_bots.py:610-621`) and the human's view fogs units only (`vision.py:15-19`), so unfogged fields are parity-consistent. This is a statement to make, not a leak to plug.
- Fix on paper: in 3.1/3.6 add: "Sight gates units, grenades and charges only; the map, doors and fields are shown unfogged in both windows, as to the bots and the human (ACCEPTED GAP until rung 8)". In 3.4 add: "the crop's `units_own/enemy/zombie` planes are not in this spec; units reach the far window only from the list (team-mates, seen enemies)". Extend 3.7's test to the far window's unit planes.

### MAJOR 5. "Whom it aims at" is not well defined for an enemy, and as written can leak
- Claim (4.1): "the enemy the team knows of that lies on the unit's aim line within its weapon's spread, with no wall between". (4a): every enemy entry says whom it aims at: "me, a team-mate, another side, nobody".
- Problems: (a) "the team" is ambiguous for an enemy E. Read as E's team, it imports E's team picture (another team's private knowledge). (b) "Aim line": for a team-mate the intended aim (`live_aim`, `facing_target_q`) is fair, but for an enemy it is controller intent that a watcher cannot see before the body turns. (c) "another side": answering it with a third-side unit my team does not know of reveals that unit's presence. (d) A zombie has no weapon spread, and its facing target is the horde field's step (`ai_zombie.py:14-24`), so the rule's "spread" is undefined for it. (e) "No wall between" should name the oracle (`vision.ray_clear`, the bots' `los`, `control_bots.py:238-242`).
- Fix on paper: one rule, evaluated by the OBSERVER's knowledge: for a unit U, the first unit in the observer team's knowledge (own team plus team picture) on U's FACING ray within U's weapon spread (a zombie: within a fixed table cone, or "toward me / a team-mate / nobody" only) up to the first `ray_clear` failure. For team-mates only, the facing target may replace the facing. Values: me, a team-mate, a known unit of another side, nobody. An unknown third-side unit reads "nobody".

### MAJOR 6. Hindsight orders put the future into the soldier's input
- Claim (5.3): orders in training also come "from labels of what a soldier in fact did".
- Effect: an order labelled from where the soldier went later carries information from after the decision (it went to room X because an enemy showed up there 5 s later). That is the same kind of leak section 7 keeps out of the value network ("What it does not see: the future"), but here it reaches the policy's own input.
- Fix on paper: name it: hindsight orders are used only as an obedience-learning curriculum at rung 4, never in matches used to measure strength, and their horizon is at most the order's own duration; or drop them from v1 (the bots' layer and the random generator suffice). Rung 4's design decides; rung 0 records the leak.

### MINOR
1. 4.1 says the order is "already the unit's state in the game"; section 9 lists it as proposed (patch 1d). Reword 4.1 to "will be, from patch 1d".
2. 7.3 lists "dormant zombies" among what the soldier cannot see, but 4a shows a seen zombie's dormant / hunting / noise state. Say "unseen units, dormant zombies included".
3. A zombie "walking toward a noise" (`zombie_noise_ticks`, `ai_zombie.py:98-110`) tells the soldier a noise happened that it cannot hear (hearing is reserved for rung 8; noise is heard through walls). Show only its awake / not awake flag, or record this as an accepted gap. The bots read neither state, so this is more than the bots have (allowed, but name it).
4. The enemy charge's time left (3.3) is a private choice (5..60 s), not something a watcher sees; hide it or show the table range only. (Flagged as Erik's to veto; the lens recommends hiding.) Also say in 3.3 that grenades in the list obey the same line-of-sight rule as charges (it is said in 7.1 only).
5. "Damage it took in the last second" (4.3) includes damage the team did not see (a third side, an out-of-sight fire before it came into view). Count only hits credited to my team, or hits while the team saw it. Give it in bars.
6. "The direction its last hit came from" (3.2) must be the round's own flight direction or the blast centre, never the attacker's current position (an unseen shooter, an indirect source).
7. "Line of fire clear" (3.3): the bots have no wall rule; they shoot only what the shooter itself sees (`fire_target`, `control_bots.py:1166-1194`). Sight (`ray_clear`: walls and `blocks_los` cover) and bullet fate (the march on `solid`, crates) differ: a crate blocks the round, a grille blocks sight. Name the bullet march's geometry as the oracle.
8. Parity gaps where the bots have more: team-mates' committed GOAL TILES for spacing (`position_context`, `control_bots.py:1043-1051`) are controller memory, so by 4.1 they cannot be shown; and `adjacent_enemy` (`:1339-1353`) scans all units with no line of sight, so a bot shoves / strikes / is woken by an enemy behind a thin wall that the soldier cannot see. Record both as accepted gaps, or put the line-of-sight check into `adjacent_enemy`.
9. No raw numbers: name the denominator of every "time left" (reload_seconds, throw wind-up, plant_seconds, the status row's duration, the memory time for "how long ago") and of "grenades by kind" (the loadout count; a kind the loadout lacks is masked, not 0/0).
10. Features with no code behind them that the document does not name as to-build: the time for the body to come on target (only a one-tick `turning.predict` exists, `simulation.py:1024-1040`); "how many team-mates a frag would catch" (the bots use their own `danger_radius`, `:1290`, not the frag's blast reach); the motion of a last-seen record ("which way it was moving": `Sighting` has none, `:249-254`; part of patch 1a, say so); the per-tile wind push plane (`GameMap.footprint_wind_load` is per footprint); a team-mate's "point it is turning to" (`live_aim` is a direction; the point comes from the 4.1 rule); a throw's landing point (the facing at release is in the future; give the facing target plus reach).
11. Sight rule consistency: today's cone is a global `[onephase]` setting (`config.toml:55-56`, read in `vision.can_see`, `vision.py:156-161`), used by both bot generations (`control_bots_v1.py:212`) and the human's fog (`simulation.py:1721`). 3.6 and 11 agree on changing it at rung 1. The "watched" plane's routine (3.4) is LOS-only and will need the enemy's facing when cones return at rung 8: one line in 3.4. Zombie activation has its own radius + LOS (`ai_zombie.py:225-238`) and is not a seat, so it is untouched by the change: say so.

## Checked and found sound
- Shared, all-round, walls-block sight is served by one oracle with a per-unit observer list, so "do I see it myself" and "the team sees it" come free (`vision.py:131-133,165-190`).
- Enemy health and focus-fire level are what the bots read (`focus_key`, `control_bots.py:495-500`); they become ratios of effective vitality and `x_max`.
- The trigger aid is the bots' own predictor (`facing_after_turn_q16`, `simulation.py:1024-1040`; `_on_target`, `control_bots.py:1196-1202`).
- The team display's aim, feet, trigger, throw, plant and reload are unit attributes (`simulation.py:981,1007,1051`; `unit.py:290-304,344-346`).
- The decoder aims at the current position only while the team sees the target, else at the last-seen position (2.2): honest.
- The value network's wall: separate weights (7.2), no shared memory, a test that scrambling its extra inputs changes no choice (8.6). In the design as written, no path from the full view to a choice was found, apart from Major 6, which is a separate leak (through the order) and does not pass through the value network.
- The compass and the route plane use the walkable map, which is public, and an objective built from the team picture.

## Not checked
- Whether a grenade in flight and a planted charge expose an owner and a fuse a list entry could read (`charges.py`, the projectile list): UNVERIFIED.
- Whether the suffocating / O2 reserve status has a producer outside the grep (`SUFFOCATING` found only in `status.py` and `events.py`).
- A fuel-vapour ignition threshold in code.
- Renderer-side fog for the human under `--control direct` beyond `simulation.py:1721`.
