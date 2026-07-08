# The Playground — play guide

The standard-values sandbox (mechanics/06 §8): every system the combat wave
shipped (P1 coupling table, P2 damage packets + mitigation, P3 statuses,
P4 wave-push + knockdown), each with a room built to feel it. Balance is
explicitly later — these are placeholder values wired so you can *play*.

## Launch

```
C:/Users/steen/anaconda3/python.exe main.py --level playground
```

(`--level` overrides one launch; the standing selection stays
`[display] level` in config.toml. `--windowed` and `--cuda` combine fine.)

## The keys that matter

| Key | Does |
|---|---|
| Left-click | select a marine / place an order of the current mode |
| **1 / 2 / 3** | Move&Attack / Move w/Cover / **Sprint** |
| **F** | fire mode (click the target) |
| **G** | grenade mode (click target; **mouse wheel = fuse seconds**, default 1.0) |
| **B** | door-explosive mode (wheel = detonation slot) |
| **Tab / Backspace / Esc** | switch planning phase / undo order / clear selection |
| **Space** | run the round ↔ pause |
| **Ctrl+R** | **hot-reload config.toml** — the tuning loop |
| **I** | DEBUG ignite under cursor (needs fuel: wood/furniture, not bare floor) |
| **J / K** | DEBUG spawn gas / cycle gas (white→black→poison→teargas→fuel) |
| **U / O** | DEBUG pour 0.2 m water / toggle water overlay |
| **P / Shift+P** | DEBUG tilt the ship ±2° |
| **T** | temperature overlay (black-body ramp) |
| **N** | **DEBUG cycle the selected marine's weapon** through the whole armory (W6 — YOUR tuning key). The panel shows the equipped row; grenades/charges stay on G/B (they are order modes, not trigger weapons). Swapping arrives with a fresh magazine and kills any burst in progress. |

## The map

**ARENA** (west) — open floor; the NW corner is the empty *grenade range*.
Furniture crates mid-map are cover (and fuel). The squad spawns SW.
**WOOD room** (NE) — flammable walls + pillars: the fire room.
**GLASS gallery** — a fragile box (hp 15, pops at 1.0 atm differential).
**SEALED room** — hull box, walk-through but gas-sealed door, one glass
window pane (east) + one wood wall segment (south): the pressure lab.
**STEEL bunker** — the tough room (blast damage ignores steel entirely).
**POOL basin** (south) — steel tub for the U key; the door holds water.
**ZOMBIE PEN** (SE) — five shamblers in a glass box with **no door**. Glass
blocks their line-of-sight trigger: dormant until you open it.
**BREACH BAY** (SW) — the hazard-striped south wall is ship hull facing
space. The decompression demo.

## Experiments

**1. Bowl the squad over** (P4 wave-push + KNOCKED_DOWN). Select Alpha →
`G` → click ~6–8 tiles from the other marines (not on them) → `Space`.
The blast buffets everyone ~0.3–0.5 tile and the knockdown ring (~9–10
tiles) is ~2× the damage ring (~4.6): the outer ring goes down sprawling
(sideways sprites, 1.5 s) without real damage — the mechanics/06 §4
signature, straight from the physics.

**2. Crank the cannon** (config hot-reload — the core tuning loop). Edit
`config.toml` `[exchange]`: `k_push = 400.0` → `1200.0`, and/or
`knockdown_dv_threshold = 6.0` → `3.0`. Save → **Ctrl+R in-game** → re-throw.
**All four `[exchange]` keys apply LIVE** (read every tick;
`knockdown_getup_seconds` re-derives its tick count on reload). The
`[weapons.grenade]` stats (`pressure`, `unit_damage`, `blast_radius`) are
also read at detonation → live too. Feel comments + interesting ranges sit
next to every key in config.toml.

**3. Burn the wood room** (heat coupling row). Walk a marine to the wood
room; hover a wood wall / pillar / crate and press `I`. Fire feeds on
structure (fuel = wall HP — bare floor won't hold a flame), radiates heat,
and a marine standing close takes heat DPS through the DamagePacket
pipeline (back off and it stops — the felt-temperature band). *Honest
note:* the lingering **BURNING** status (afterburn DoT) is designed and its
machinery is live, but the fire→BURNING trigger lands with the fire
coupling row — next wave.

**4. Zombie flambé, 4×** (mitigation tables). Ignite the crate barricade
(mid-arena), park a marine two tiles away and note its HP drain — then
(after experiment 5) let a zombie shamble across the same fire: it melts
~4× faster. The knob: `[zombie] fire_damage_multiplier` (= the zombie's
`resist_mult[HEAT]`; restart-bound), beside `[zombie] stability` for how
easily they topple.

**5. Open the pen** (glass + the horde). Rifles do **not** break walls yet
(bullet cover-chew rides the exposure-roll pass — designed, not wired), so:
move a marine east a round or two, then lob a grenade **next to the pen's
glass wall**. The +10 atm disc bursts the panes (glass pops at 1.0 atm
differential), the same wave knocks the nearest zombies down, LOS opens,
and the horde activates and pours out. Fall back to the barricade.

**6. Overpressure the sealed room** (the emergent relief valve). Lob a
grenade through the doorway into the sealed room: the sealed volume holds
the +10 atm spike and the **glass window pane blows out first**
(threshold 1.0 < door/wood 2.0; hull never bursts), venting with a
pressure jet — watch it on the pressure overlay. *Honest note:* a fire
alone won't do it — the plume self-limits at ~1.3 atm (`p_expand_ref`),
a 0.3 differential across an interior pane. For the emergent
fire→window-blowout chain, drop `[materials.glass] burst_threshold`
toward `0.25` (materials bind at map construction → restart).

**7. Breach the hull** (decompression). Optionally `J` a smoke blob in the
breach bay first. Walk a marine in, `B`, click the hazard-striped south
wall, `Space`. The door explosive (500 wall damage) opens the 300-HP hull
in one; grenades need two on the same tile. Atmosphere howls out, the
smoke streams into space, and the blast + decompression wave transients
buffet anyone standing in the throat (units ride `wave_p` gradients; a
sustained blast-*wind* throw is a possible later coupling row — Erik's
call, noted in `[exchange]`). Flood the bay first (U) and the water
flash-boils to steam as pressure drops below 0.3 atm.

**8. The steel lie** (material feel). Door-explosive the STEEL bunker
wall: nothing — blast structural damage skips steel, and its 10.0 burst
threshold shrugs off the 5.0-pressure charge. Now detonate a grenade
*touching* the wall: the +10 epicentre can just exceed the 10.0
differential and pop the tile. Small arms chip harmlessly; overpressure
finds the seam.

**Water postscript.** The pool basin is for water feel: `U` to pour
(repeat for depth), `O` overlay, `P` to tilt the ship and watch it slide;
drain a flooded room to vacuum and it boils. *Honest note:* the
water→unit rows (speed multiplier, drowning) and gas→POISONED /
O2→SUFFOCATING are table rows with designed responses — triggers land
with the next coupling wave. Water↔fire interaction belongs to the water
arc (its own session).

## 9. The W6 armory session (the grand tuning pass)

The whole mechanics/03 §6 table is data now, ranges are PHYSICAL METERS
(playground tiles are 0.333 m — a 10 m flamethrower reaches 30 tiles), and
**N** walks a selected marine through every triggerable row. The loop:

1. Select a marine → tap **N** until the panel shows the row you want →
   `F` + click → `Space`. Repeat down the list; every weapon fires through
   the same order flow.
2. **Dragon-7** (the rescue of the W4 feel-check): hose the wood room —
   the jet is VISIBLE now (a flickering flame fan + its own light), 10 m of
   reach, and wood catches through most of the cone inside a burst.
   **Dragon-9 heavy** doubles the reach (20 m) and the deposit. Dials:
   `[weapons.dragon_7] range_m`, `[ammo.fuel_standard] heat_deposit`
   (derivation comment shows the reach arithmetic — deposit and range are
   ONE dial pair).
3. **Miasma Vent**: the same hose in sickly green (fainter on purpose);
   poison drains marines, zombies shrug (they don't breathe).
4. **Sunspot / Helios plasma**: fire across the arena — a slow glowing
   bolt you can WATCH fly (1.5 / 1.25 tiles per tick), then a splash that
   scorches walls, ignites wood, and cooks bystanders through the heat
   row. The direct hit is HEAT: try it on the pen's zombies (×4 heat
   vulnerability) vs a crate wall.
5. **The gun rack**: P12 Whisper (one quiet precise shot — `loudness` is
   its identity, consumer pending), MP-11 (4-round sprays, coin-flip past
   ~10 m), K5, LR-50 (one shot, 90 damage, chews cover fast), Jackhammer-8
   (8 pellets — watch the spread eat damage with distance: geometry, no
   falloff table), Lance-3 / Lance-5 (instant beams; smoke is their
   counter — `J` a cloud into the line of fire), knife/baton up close.
6. **Chain-stun check** (the W5 finding, YOUR call): baton a penned zombie
   — the 19-tick cadence re-stuns inside every 36-tick stun window, so one
   marine holds one zombie forever. The dials sit side by side:
   `[weapons.arc_baton] rof_interval_seconds` vs `status_seconds`.
   Numbers deliberately untouched at W6.
7. **Incendiaries**: `G` still throws frag by default; the incendiary
   rounds (`grenade_incendiary` / `40mm_incendiary`) are authored and
   test-covered — the per-type grenade loadout UI is still owed, so feel
   them via the GL-6 in a test or wait for the loadout pass.

**The weapon dials live in `[weapons.*]` / `[ammo.*]` / `[payloads.*]` —
ALL RESTART-BOUND** (tables rebuild at Simulation construction; Ctrl+R
re-reads config but re-arms nothing — engine/12 §5, the W1 finding; the
guide's old "read at fire time" note predates W1). Edit → relaunch →
`N` back to the row → fire. Every row carries a STANDARD-VALUE comment
with its interesting range.

## Where the knobs live

- **Live on Ctrl+R**: `[exchange]` (push + knockdown), `[combat]` (heat
  band + blast threshold).
- **Restart-bound**: `[weapons.*]` / `[ammo.*]` / `[payloads.*]` (tables
  rebuild at Simulation construction — the W1 finding; the pre-W1 "read at
  fire/detonation time" behavior is gone), `[marine] weapon` (bound at
  unit construction), `[zombie]` (species tables bind at import),
  `[physics.*]` (solvers bind at construction — engine/12 §5),
  `[materials.*]` (table binds at map build).
- Regenerate the map after layout tweaks:
  `C:/Users/steen/anaconda3/python.exe tools/gen_playground_level.py`.
