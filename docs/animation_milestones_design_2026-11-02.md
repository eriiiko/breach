# Animation milestones 13–15 — design (draft for Erik's veto pass)

**Status: DRAFT v0.1, 2026-11-02.** Tracker: #153. Milestones: **13 – Pose layer**,
**14 – Zombie motion**, **15 – Bodies in motion**. Written after a design chat with
Erik; nothing here is built. Every choice is **provisional** unless it says LOCKED
(Erik, 2026-11-02: "I don't know anything about animation, I expect to make lots of bad
decisions and I don't want to lock into the first ones"). The doc is therefore built
around *finding the limits cheaply* (§7) and *keeping decisions reversible* (§2).

Supersedes in part (does not replace; those stay as dated capture):
`procedural_animation_brainstorm.md` (2026-07-17), `anim_phase0_impl_2026-07-20.md`,
`research/ml_animation_litsearch_2026-07-20.md`. §1.3 says which parts of them stand.

---

## 0. What Erik wants (his words, condensed)

- **Zombies** that are not traditional zombies: Genestealer (WH40K) and Bloodborne's
  **Scourge Beast** — spine pitched forward, arms raised forward as if to grab, beast
  weight. **Get the reference right before producing anything.** Gradual human →
  zombie *by continuously blended posture* ("it fits the design philosophy of breach, a
  simple system that can produce lots of things"); levels 1 and 2 of "zombiefied" are an
  acceptable discrete version. Door bashing and rising from the floor are needed.
- **Marines:** a rich set; **a melee animation for every weapon**; legs and upper body
  **layered**; pistol / gun / heavy-gun firing sets; walking, hunching, etc.; carrying a
  **shoulder bag**, planting it on the ground and picking it up.
- **Bodies:** shockwave pushes, a little ragdoll, bounce against walls, recover. Position
  always deterministic sim-side; legs and arms probably not. Eventually **bodies knocking
  each other over**, with momentum and angular momentum — "exactly where I'd want to end
  up". The sim-side consequences of that are **their own research session** (not here).
- **Tools:** try Kimodo (NVIDIA text-to-motion) if it can help.

## 1. Where we start from (verified 2026-11-02)

### 1.1 What exists

- **One clip source:** Quaternius *Universal Animation Library*, free "Standard" edition,
  CC0, in `assets/models/marine/` (46 clips, 53-bone Rigify-style `DEF-…` skeleton, 30 of
  the bones are fingers). Space marine, both workers and Kessler are skins on this one
  skeleton; the game-ready step (`charkit/gameready.py` + `rig.py`) converts clips to
  each character's rest pose.
- **The game uses 4 clips:** `CLIP_MAP` = idle, walk, fire (`Pistol_Shoot`, dormant),
  dead (`Death01`, dormant) — `renderer/unit_model_renderer.py:78`. Zombies are
  `worker_male` / `worker_female` (`[render.unit_looks]`) playing the human walk.
- **The renderer plays exactly one clip per unit at an integer frame.** No crossfade, no
  blend, no bone mask (`_draw_one`, `unit_model_renderer.py:623-656`). **CPU skinning**,
  soft ceiling ~20 animated units; our pyray builds (6.1-dev, 5.5.0.4 on the home
  desktop) expose no `UpdateModelAnimationBoneMatrices`.
- **Sim signals the renderer can read today:** `unit.facing` (one synced angle, radians);
  position (movement is *inferred* from deltas — no velocity field); composed status
  flags (`is_prone`, `can_move`, `can_act`, `can_aim`); status kinds knocked_down,
  immobilized, stunned, paralyzed, burning, poisoned, suffocating, regen, blinded
  (`status.py`); events ShotFired, LaserFired, Explosion, UnitHit, UnitKilled, SprayJet,
  DoorDestroyed, WallDestroyed (`events.py`).

### 1.2 What the clip pack has and lacks

| Group | Have | Missing |
|---|---|---|
| Locomotion | Walk, Walk_Formal, Jog, Sprint, Crouch idle/fwd, Roll (+RM), Jump ×3, Swim ×2 | strafe, backpedal, prone/crawl, limp / wounded |
| Pistol | Idle, Aim up/neutral/down, Shoot, Reload | — |
| Rifle / SMG / shotgun | none | hold, aim, shoot, reload, walk-aiming |
| Heavy / tool | none | braced hold, slow walk, sustained fire |
| Throw / place | `Fixing_Kneeling` (nearest) | grenade throw, plant a charge |
| Melee | Punch ×3, Sword ×3 | a melee per weapon family |
| Hit / death | Hit_Chest, Hit_Head, Death01 | directional hits, stagger, knockdown, get-up, blast death, burning, choking |
| Interact | Interact, PickUp_Table, Push, Sit ×4, Driving, Spell ×4, Dance | carry, set down, door/panel work |

### 1.3 What the older docs say, and what still stands

| Doc | Still stands | Outdated / reversed |
|---|---|---|
| Brainstorm 07-17 | Procedural stack (FK, 2-bone IK, FABRIK, Verlet); **authored poses blended over a base**; "animation set = parameter set; a zombie is the humanoid skeleton with a shuffling gait profile and a slumped posture" (= Erik's continuous infection blend); determinism split by creature class; spider/octopus procedural path (M8) | Part-sprite art pipeline (dead); "non-needs: no clip format, no GPU skinning" (now **needed**) |
| Phase-0 impl 07-20 | Render-only / no anim state on `Unit`; table-driven `select_clip`; `_draw_one` as THE swap seam | GPU skinning "deferred": **now due** (a horde breaks the ~20 ceiling) |
| Lit-search 07-20 | Match technique to force scale; cap concurrent ragdolls; the get-up blend is the hard part; planar-2D ragdoll spike; "network is free, substrate is the cost"; top-down hides footwork, shows facing / torso / arms / silhouette | Predates Kimodo and 4 months of field movement → **refresh scan before committing (E7)** |

Existing issues folded in: #34 (weapon + grenade anims; its shockwave-push half → M15),
#149 (knockdown animation), #105 (octopus animation study — shares the pose layer),
#33 (appearance / skins).

## 2. Keeping decisions reversible

**Cheap to change:** which clips exist and where they came from; clip names; family
membership; posture parameter values; which tool made a clip.
**Expensive to change:** (a) the **interchange skeleton** — every character is baked onto
it; (b) the **pose-layer interface** — everything plugs into it; (c) any animation state
that reaches the sim (digest).

Rules that follow (all provisional):
1. **SOMA (Kimodo's skeleton) is the interchange skeleton — LOCKED by Erik 2026-11-03,
   replacing Quaternius.** Kimodo outputs SOMA natively, so its clips need no retarget;
   NVIDIA's motion family (ARDY, MotionBricks) speaks it; its hierarchy is the common
   mocap layout, so other sources convert into it easily; and the switch is cheapest
   before the pose layer exists (§12). The 46 Quaternius clips are **not ported**: new
   clips are made, only the ones wanted. Every clip source (Kimodo, Mixamo, hand-keyed,
   procedural) is *retargeted into SOMA*; none is adopted as a second skeleton. A
   genuinely new body plan (octopus, spider) is a new skeleton by the existing topology
   test, as today. Consequence for the build: **the pose layer is skeleton-agnostic**
   (bone names, masks and layer tables are data, nothing hard-codes a bone list). The
   cost of the switch is **re-binding the characters** (space marine, workers, Kessler …)
   onto SOMA, not the clips.
2. **The pose layer is designed as the one interface** (§4.1): sources in, bone matrices
   out. Replacing the skinner (CPU → GPU) or a clip source touches nothing else.
3. **Animation never writes sim state.** Anything that changes where a unit is, who it
   hits, what it is (status) or which way it faces when it gets up is integer sim state
   and belongs to the sim research session, not here.
4. **A decision ledger** (below) is kept current in this doc: each row is LOCKED or
   PROVISIONAL, what reversing costs, and what evidence would make us revisit.

### Decision ledger

| # | Decision | Status | Reversal cost | Revisit when |
|---|---|---|---|---|
| D1 | **Kimodo's SOMA skeleton is the humanoid interchange skeleton** (77 joints; the 30-joint SOMA variant for runtime skinning). Quaternius retired; its clips not ported | **LOCKED** (Erik, 2026-11-03) | High (every character re-bound) | a needed pose is impossible on it (twist, long-arm, tail) — E5 |
| D2 | Animation is render-only; no anim state on `Unit` | LOCKED (iron rule, digest) | — | never, without a sim ruling |
| D3 | Pose = base + masked layers + additive + physics override, composed by us | PROVISIONAL | Medium | E1 shows the engine path can't hold it |
| D4 | GPU skinning with our own bone-matrix upload | PROVISIONAL | Low (swap seam) | E1 perf at 100 units |
| D5 | Per-weapon-family clip sets, not per-weapon | PROVISIONAL | Low | A weapon needs unique handling |
| D6 | Zombie = same skeleton, posture parameters + own clip set | PROVISIONAL | Medium | E3/E4 say the humanoid rig can't carry the four-point run |
| D7 | Kimodo is an authoring tool (offline), never in the render loop — and, if good enough, THE clip source | PROVISIONAL | Zero | E3 |
| D8 | Infection stage shown continuously from a sim-provided amount | PROVISIONAL, **needs the sim research session** | — | the sim session |
| D9 | One generic fall / prone / rise set serves marines and zombies (and later conversion) | PROVISIONAL | Low | the sim session fixes how conversion lies down |
| D10 | Generated characters (TRELLIS meshes, e.g. Beatrice #154) are bound to SOMA by the same step as scripted ones: rig scaled to the mesh, arms set to its A-pose, automatic weights | PROVISIONAL | Low (per character) | weights fail on a character (props near the hands already stretch, §12) |
| D11 | Hair is a separate mesh; it sways on a few extra head-child bones driven by a render-side spring | PROVISIONAL, **touches D1** — Erik decides whether extra bones may sit on the interchange skeleton | Low | the pose layer (W13.1) exists |

Companion docs (2026-11-02): `zombie_motion_brief_2026-11-02.md` (the brief, for Erik's
veto) and `kimodo_spike_workflow_2026-11-02.md` (handoff for the work-desktop Claude,
experiments E2/E3).

## 3. Milestone map

| M | Name | Contents | Depends on |
|---|---|---|---|
| 13 | **Pose layer** | pose compositor, GPU skinning, legs / upper-body / head layering, door bash (all humans), state→clip table, weapon families + a melee each, knockdown set (#149), grenade + plant, shoulder bag (carry / set down / pick up), retarget path for a foreign clip | nothing new in the sim |
| 14 | **Zombie motion** | reference brief, posture parameter set, infection levels, Scourge/Genestealer clips, door bash, rise from floor, turning | 13's pose layer; **sim infection signal** (research session) |
| 15 | **Bodies in motion** | render-side shockwave response, visual-body offset spring, additive flinch, ragdoll-lite, wall bounce, get-up blend; **end goal: bodies knocking each other over, with momentum / angular momentum** | 13; the end goal needs the **sim research session** |

Naming note: the lit-search already uses "rung" for two ladders (determinism; reaction
technique). To avoid confusion the knock-over goal is called **M15 phase 3**, not a rung.

## 4. Milestone 13 — Pose layer

### 4.1 Architecture (provisional)

```
 clip sources ──┐
 (glTF clips,   │      ┌──────────── pose layer (renderer/, render-only) ───────────┐
  retargeted)   ├────► │ per-unit state → layers:                                   │
 procedural     │      │   L0 base (legs / locomotion)      mask: hips + legs        │ ──► bone matrices ──► skinner
 offsets        │      │   L1 upper body (aim/fire/carry)   mask: spine + arms      │       (CPU now,
 physics (M15)  ┘      │   L1h head (look / aim target)     mask: neck + head        │        GPU later)
                       │   L2 additive (flinch, breathing, posture offsets)          │
                       │   L3 override (ragdoll-lite blend weight w ∈ [0,1])         │
                       └────────────────────────────────────────────────────────────┘
        inputs read from the sim: facing, inferred velocity, status flags, events
```

- **Sampling is ours:** read the glTF keyframes (cached per look), interpolate (sub-frame,
  not integer frames), compose per-bone local TRS under masks, output skinning matrices.
  Crossfade = a weight ramp on a layer, not a special case.
- **Bone masks** are data: named groups (`legs`, `torso`, `arms`, `head`, `fingers`) over
  the skeleton's bones. **Legs, upper body and head are three independent layers**
  (Erik, 2026-11-02: head "preferably" layered) — the head can look at a target or snap
  toward a sound while the torso aims and the legs walk.
- **Strafe / aim-vs-travel:** aim is `unit.facing`; travel direction is the inferred
  position delta. The lower body picks a locomotion clip by *travel angle relative to
  facing* (fwd / back / left / right, blended), the upper body follows facing. **Open
  check (W13.0): confirm `facing` really is independent of `MOVE_DIR` travel; if not, the
  renderer has no aim signal and this needs a sim field.**
- **State→clip stays a table** (Phase-0 rule): `(role, weapon_family, stance, status
  flags) → (lower clip, upper clip, additives)`. A new weapon = a family choice in its
  `[weapons.*]` row (the weapons table is data); no per-weapon branch in renderer code.
- **Skinner:** keep CPU skinning as the fallback; add a GPU path (bone-matrix uniform +
  skinning vertex shader, inside `_draw_one`'s seam). The lit shader (`marine_shader.py`)
  and projected shadows read the *skinned pose*, so both must work on whichever skinner.

### 4.2 Animation families (draft)

| Family | Weapons (today / expected) | Stance clips | Melee |
|---|---|---|---|
| `one_hand` | pistol | idle, aim, shoot, reload, walk-aim | pistol whip |
| `long_gun` | K5 carbine, rifle, SMG, shotgun | hold, aim, shoot, reload, walk-aim, crouch-aim | butt strike |
| `heavy` | LMG, flamethrower | braced hold, slow walk, sustained fire | shove / swing |
| `thrown` | grenades | wind-up, throw (over / under) | — |
| `unarmed` | civilians, fallback | idle, walk | punch (have) |

Melee is **one per family**, so five clips, not one per weapon (D5).

### 4.3 Marine inventory and its sim signals

"Signal exists?" = the renderer can already read the trigger. **U = unverified, check in
W13.0.** A clip with no existing trigger is not built until its trigger is decided.

| Group | Animation | Source today | Trigger | Signal exists? |
|---|---|---|---|---|
| Locomotion | idle, walk, jog, sprint, crouch | have | inferred speed / stance | speed inferred; crouch U |
| Locomotion | strafe ×2, backpedal | **new** | travel vs `facing` | derivable |
| Locomotion | prone crawl, limp | **new** | `is_prone` (crawl) / hp | prone yes; limp: no hp read in renderer U |
| Weapon | fire, reload per family | pistol have; rest **new** | ShotFiredEvent; reload U | fire yes; reload U |
| Weapon | melee per family | punch/sword have; rest **new** | melee hit event (bites already carry source "melee") | U for marine melee |
| Weapon | grenade throw | **new** | `throw_grenade_intent` | intent exists; render hook U |
| Weapon | plant charge / place | `Fixing_Kneeling` base | placement action | U |
| Reaction | hit flinch ×4 directions | Hit_Chest/Head have | UnitHitEvent (direction?) | U |
| Reaction | knockdown fall / prone loop / get-up — **one generic set for marines AND zombies** (zombie variant = beast-like fall and lurching rise; later also used by conversion) | **new** (#149) | KNOCKED_DOWN status | yes |
| Reaction | burning panic, choking/coughing, suffocating, stunned | **new** | burning / poisoned / suffocating / stunned status | yes |
| Reaction | death: shot, blast, burned | Death01 have; rest **new** | UnitKilledEvent cause | cause exists (#123) |
| Environment | brace against wind | **new** | wind load W (`exchange.wind_scale_q16` seam) | seam exists |
| Interact | open door / panel, hack | Interact have | use_intent | intent exists |
| Carry | bag: idle, walk, run, set down, pick up, drop-on-hit; variants by weight / contents | **new** | carried-item state — **the bag EXISTS as a game item** (Erik, 2026-11-02: heavier tools, mission-critical items, bombs, money, weapons) | **no** — the sim's carried-item state is a design / research item; animation waits on it |

The bag is its own asset attached to a bone (Erik's rule: character models carry no
props); the *carrying* animation is a pose-layer layer plus an attach point. Whether a
bag exists as sim state (an item, a payload, a placeable) is a **game-design decision,
not an animation one**, and is parked for Erik.

### 4.4 Work items (to become issues after the veto pass)

- **W13.0** Verify the open checks above (aim vs travel, crouch/reload/melee/hit-direction
  signals). Read-only; writes the answers into this doc.
- **W13.1** Pose layer core: glTF sampling, masks, layers, crossfade, state table; behind
  a setting, default off. Look-shot / E2E (`tools/e2e_drive.py`) proof.
- **W13.2** GPU skinning inside `_draw_one`'s seam; perf bench at 20 / 100 units.
- **W13.3** Locomotion layering: strafe / backpedal, aim-follow torso.
- **W13.4** Weapon families table (`[weapons.*]` `anim_family`), long-gun + heavy + one-hand
  sets, melee per family.
- **W13.5** Knockdown set (#149), burning / choking / stunned reactions, death variants.
- **W13.6** Grenade throw + plant.
- **W13.7** Shoulder bag layer + attach (after the game-design call on what a bag *is*).
- **W13.8** Foreign-clip retarget path (E2) — a scripted Blender-headless step; reuses
  `charkit/rig.py`'s rest-pose conversion.

**Gate:** feel-adjacent → **HUMAN-TEST** (Erik plays it before merge), per the iron rule.
Mechanical sampling / skinner swaps that are pixel-identical to the previous path may
auto-merge on green when pre-authorized.

## 5. Milestone 14 — Zombie motion

### 5.1 Reference first (before any clip is made)

Erik's recollection (from memory, to be replaced by real reference): Genestealer and
Bloodborne **Scourge Beast** share a feel — body hunched forward from the first stage,
arms raised forward and open as if to grab, weight carried high on the chest, irregular
rhythm, sudden lunge.

**Process (language first):**
1. Erik gathers 3–5 reference clips/stills with timestamps (Genestealer sprint, pounce
   and claw strike; Scourge Beast idle, walk, lunge, roar). I cannot watch video: he can
   export stills or I work from written descriptions he corrects.
2. I write a **motion brief** in words — per motion: spine pitch, shoulder protraction,
   arm carry, head position, tempo, what is irregular, what the telegraph is — and Erik
   vetoes line by line. *No clip is produced before the brief is approved.*
3. The brief is turned into **posture parameters** (below), tested on the existing walk
   as additive offsets *before* any new clip exists.

Top-down view note: what reads is spine lean (torso shifted forward over the feet), arm
reach, head, silhouette and tempo; footwork is hidden. So the cheapest high-value part of
the zombie look is the posture layer, not new gait clips.

### 5.2 The posture parameter set (draft)

Continuous values, applied as additive offsets (L2) over the base:
`spine_pitch`, `shoulder_protraction`, `arm_forward`, `arm_open` (hand spread), `head_drop`,
`head_tilt`, `knee_bend` (lowers the hips), `tempo_jitter` (irregularity in playback
rate), `twitch` (a small noisy additive). A **zombie profile** is one TOML row of these;
the human is all zeros. This is the 2026-07-17 brainstorm's "animation set = parameter
set", now concrete.

### 5.3 Infection — continuous posture, discrete clips

- A unit has an **infection amount a ∈ [0, 1]** (source: the sim; see below). The renderer
  scales the posture parameters by `a` continuously (the "absolutely love it" part).
- **Gait is not blended** between human and beast walks: that only looks right if the two
  cycles are phase-aligned, and generated clips won't be. Instead there are **discrete
  clip levels** (human → *infected* (stagger, cough, limp) → *turning* → *zombie*) with
  crossfades between levels, while posture slides continuously between them. This is
  Erik's "zombiefied level 1 and 2" and the continuous idea combined.
- Skin: a wound / vein texture overlay driven by the same `a` is a shader parameter
  (cheap; belongs with the looks layer #33).
- **Sim dependency (parked, own research session):** the renderer can only show what the
  sim says. If `a` is a game mechanic (bitten timer, spread) it is synced state. Until
  that session decides, M14 can ship the posture layer and clip levels keyed on the
  existing boolean `is_zombie` (a = 0 or 1).

### 5.4 Clip list (about 25; all on the interchange skeleton)

Idle (sway, twitch, head-cock) · stalk walk · charge run · four-point sprint (Scourge /
Genestealer — a bent-over run on the humanoid rig, long-arm proportions via the skin) ·
claw swipe L/R · lunge / pounce · bite-and-grab (the sim already has bites) · roar /
alert telegraph · hit stagger · burning flail · deaths (collapse, crumple, burst) ·
legless crawl · **turning** (the spasm when a marine converts) · **door bash** · **rise
from the floor**.

**Door bashing (Erik: yes, for zombies AND ordinary humans).** One shared bash clip set
on the interchange skeleton (a marine shouldering a door and a zombie ramming it are two
variants of the same action, so it is built once in the pose layer's clip tables, with a
zombie posture on top). It needs a sim rule that does not exist today: a unit attacking a
door entity (door hit points, the attack, an event the renderer can read).
`DoorDestroyedEvent` exists; a unit-attacks-door behaviour does not. The bash clip waits
on that rule; the rule is a game-design / sim item. `[bots]` / zombie control
and the door entity (A6) are the places to look.

**Rising from the floor = the fall / prone / rise set (Erik, 2026-11-02).** Zombies can
also be knocked down and must rise, so the set is generic and shared (see §4.3); the
conversion is one more consumer of it and **how and when conversion uses it is decided
later**. Verified: conversion is **instant today** — only melee kills convert
(`ai_zombie.convert_marines_to_zombies`, run every tick from `ruleset.on_tick_end`: hp
reset, `alive`, team `ZOMBIE_TEAM`) and there is no lying interval in the sim; dead
units are not drawn. Erik: the lying period **should be real** (a sim window, the
converting body inactive for a time) but it barely affects the animation work. Idea for
the sim session (not decided): reuse the status machinery with a `KNOCKED_DOWN`-like
"converting" status. A corpse that lies visibly also needs dead units to be drawn and, in
M15, a ragdolled corpse pose that the rise blends out of.

## 6. Milestone 15 — Bodies in motion

### 6.1 The one boundary rule (restating the iron rule for bodies)

> **The sim owns the integer facts** — position, status (knocked down / prone / dead),
> facing at get-up, the knockdown's start and end tick. **The render side owns the
> float flail** — limb motion, a visual offset from the sim position, spin. The visual
> body is always `sim position + offset`, and the offset is pulled back to zero.
> No float in this milestone may feed the sim.

### 6.2 Phases (each a stopping point)

| Phase | What | Needs sim change? |
|---|---|---|
| 1 | **Visual-body offset spring** + impulse-scaled slide: a shockwave moves the *drawn* body, which overshoots at a wall (the wall is where the sim stopped it) and settles | No — consumes the sim's blast / wind events |
| 2 | **Additive flinch** for bullets, directional (4–8 directions), over any base clip | No (needs hit direction on the event — check W13.0) |
| 3 | **Ragdoll-lite**: light Verlet ragdoll, knocked-down units only, capped (about 12), collides with floor plane and wall tiles read-only from the map, blends out into a get-up pose; the get-up blend must be invisible | No |
| 4 | **Bodies knocking over bodies; momentum and angular momentum that matter** — a tumbling body hits others, what facing a unit gets up with | **Yes — a new sim mechanic.** Erik's separate research session decides what is integer state; this phase waits for it |

Phase 3's concurrency worst case is a grenade into a squad (N simultaneous ragdolls):
budget the peak, cap concurrent ragdolls, overflow to phase 1.

### 6.3 What the sim research session has to answer (handed over, not decided here)

Where momentum lives (a velocity on `Unit`? an impulse event only?); whether angular
momentum is a synced scalar or a render-side decoration of a sim spin; which collisions
between bodies are sim events; what a body pile-up does to the footprint / pathing /
`apply_wind_knockdown`'s stability model; determinism cost (fixed-point contact is the
expensive rung in the lit-search's ladder). Starting point: the brainstorm §0.2 and
lit-search Thread E.

## 7. Learning experiments (each exposes a limit; each can fail)

| # | Experiment | Learns | Cost | Kill / pass criterion |
|---|---|---|---|---|
| E1 | Pose layer prototype: two clips masked (legs walk + torso pistol-aim) with crossfade, GPU skinning, 100 units | Can we own sampling in Python at this scale? What does the GPU path cost to write? | 1–2 days | pass: ≥ 60 fps at 100 units, no visible pops; fail → rethink D3/D4 (e.g. C++ sampler) |
| E2 | Retarget one foreign clip into the 53-bone skeleton (Kimodo output, or a Mixamo clip as the control) | How lossy is retargeting; what breaks (fingers, proportions)? | ½–1 day | pass: plays on the space marine without foot/arm artefacts a reviewer notices |
| E3 | **Kimodo spike** (§8): "hunched, arms-forward stalk walk" with a full-body keyframe constraint | Does it make usable beast motion at all? | ½ day + retarget (E2) | pass: one clip Erik would keep; fail: drop Kimodo for beast motion, keep as a locomotion variant tool or drop |
| E4 | Posture-only zombie: additive offsets on the existing walk/idle/run, no new clips | How far does posture alone get toward "Scourge Beast" from above? | 1 day (needs E1) | pass: Erik says "that reads as a beast" in the top-down view |
| E5 | Humanoid rig with long arms + four-point run pose | Can the one skeleton carry the beast run? | ½ day (needs E1) | fail → D6 revisit (a creature skeleton) |
| E6 | Ragdoll-lite on one unit (Verlet, 15 bodies, floor + wall tiles) + blend back to a pose | Cost, stability, the visibility of the get-up blend | 2 days | the blend-out seam is invisible at play zoom |
| E7 | Tool-landscape refresh: what exists in 2026 for text-to-motion, retargeting, physics-driven animation (a bounded scan, not a rewrite of the lit-search) | What did four months change | ½ day, agent | a dated addendum to the lit-search |

Order: **E7 and E1 first** (they set what the rest is built on), then E4 (cheap, shows
the zombie look), then E2/E3 together, then E5, E6.

## 8. Kimodo (NVIDIA, github.com/nv-tlabs/kimodo) — what is known

- Kinematic motion diffusion, trained on ~700 h of optical mocap; text prompts (with
  multi-prompt sequences) plus constraints (full-body keyframes, end-effector position
  / rotation, 2D root path, waypoints). Interactive timeline demo + CLI.
- **Output:** custom NPZ (positions, rotation matrices, foot contacts, root) on the SOMA
  **77-joint** skeleton; also G1 (robot) and SMPL-X variants. **BVH export exists**
  (`--bvh`, optionally with a standard T-pose rest) — verified 2026-11-02, see §12.
- **Hardware:** ~17 GB VRAM all on GPU; **under 3 GB with `TEXT_ENCODER_DEVICE=cpu`**
  (slower, text encoding only). Tested on RTX 3090 / 4090 / A100. Developed on Linux;
  Windows "should work, especially with Docker". Home PC: 8 GB — expected to run with
  the CPU text encoder (system-RAM need not checked). Own conda env or WSL2; not the
  breach env. Renting a 24 GB cloud GPU is the fallback; animations will come in
  batches, so occasional rental fits.
- **License:** code Apache-2.0; SOMA / G1 checkpoints under the **NVIDIA Open Model
  License**; SMPL-X under an NVIDIA R&D licence (non-commercial: not for Breach).
  **Read 2026-11-03:** the Open Model License allows commercial use, NVIDIA claims no
  ownership of outputs and does not restrict their commercial use, worldwide; a notice
  file is only required when the *model* is redistributed (Breach does not). Generated
  clips may ship. (A summary, not legal advice.)
- **Unknowns that decide its value:** whether the training data contains crawling /
  bent-over / creature-like locomotion; how well keyframe constraints hold a held-object
  pose; SOMA proportions vs ours. The spike answers the first two.
- Role if it works: **offline authoring** of locomotion variants and the beast gait
  (D7). It is not a runtime component, and never touches determinism.

## 9. Settled, and parked for the sim / game-design session

This doc is about animation. Game-mechanic questions are **not asked here**; they are
listed so the sim / game-design session can pick them up.

**Settled (2026-11-02):** milestone names and the 13 / 14 / 15 split · the shoulder bag
exists as a game item (heavier tools, mission-critical items, bombs, money, weapons) ·
door bashing is wanted · rising from the floor is the generic fall / prone / rise set
(§5.4) · the zombie reference is drafted from words and corrected line by line
(`zombie_motion_brief_2026-11-02.md`).

**Parked for the sim / game-design session** (animation work proceeds without them; each
only decides *when* a finished animation is triggered):
- Carried-item state: what a unit carrying a bag is, in the sim.
- Infection as a gradual sim state (the amount `a` in the brief §2).
- A zombie-attacks-door rule (door hit points, the event the renderer reads).
- The conversion's lying period (§5.4).
- Zombie and marine speeds; when a zombie uses its fast gait.
- Bodies knocking each other over (M15 phase 4, §6.3).

## 10. Systems

**(a) Existing canonical systems this design must use**

- **Unit** (`unit.py`): no model / animation state ever on `Unit` (digest). Per-unit
  animation state lives in the renderer, keyed by unit id.
- **3D units** (`unit_model_renderer.py`, `unit_looks.py`, `marine_shader.py`): the pose
  layer lives *inside* this module family; `_draw_one` is the swap seam; looks are data.
  A new look or role clip set is a table entry.
- **Weapons table** (`[weapons.*]`): the animation family is a column on a weapon row;
  never a per-weapon renderer branch.
- **Status flags** (`status.py::composed_flags`, status kinds): reactions are driven from
  composed flags and status kinds, never from new unit fields.
- **Game-ready step + character kit** (`charkit/gameready.py`, `rig.py`): clips reach a
  character only through it; no hand export, no second skeleton. It binds to Quaternius
  today and must move to SOMA (D1); the generated-character binding in local-models
  (`blender/retarget_soma.py`, §12) is the working prototype for that move.
- **Wind load seam** (`exchange.wind_scale_q16`, `GameMap.footprint_wind_load`) and the
  **tamed wind** (render side): wind-response animation reads these, never the planes.
- **Lit-3D seam / one tone-map / projected unit shadow**: the skinned pose feeds all
  three; a new skinner must keep the marine byte-identity gate (`test_lit3d_extraction`).
- **E2E driver** (`tools/e2e_drive.py`) and **look shots**: the proof harnesses; no new
  screenshot harness.

**(b) New systems this design creates (draft one-line rules, to land in CLAUDE.md only
when built)**

- **Pose layer** (`renderer/pose_layer.py`, M13): *the ONLY place a unit's bone matrices
  are composed; clips, masks, additives and physics blend through it; the skinner is
  swappable behind it.*
- **Bone-mask table** (data): *a body region is a named mask over the interchange
  skeleton, never a hard-coded bone list in code.*
- **Animation family table** (`anim_family` on `[weapons.*]` + a family → clip-set
  table): *a weapon animates by choosing a family; a new weapon is a row, not clips.*
- **Posture profile** (TOML): *a creature's look of motion is a parameter row of the
  posture set scaled by an amount in [0, 1]; the human is all zeros.*
- **Visual-body offset** (render-side, M15): *the drawn body is `sim position + offset`;
  the offset is float, render-only, and returns to zero; it never feeds the sim.*

## 11. Provenance and next steps

Drafted from a design chat 2026-11-01/02 plus a read of the three older animation docs,
the clip pack, `unit_model_renderer.py`, `status.py`, `events.py` and issues #34 / #149.
Not verified: items marked **U** in §4.3; Kimodo claims are from its README only (not
run); the Scourge Beast / Genestealer descriptions are Erik's and my recollection, not
checked against footage.

Next: Erik's veto pass on this doc → W13.0 (verify the U's) and E7 / E1 → cut the
work-item issues under milestones 13–15 → HUMAN-TEST gates as marked.

## 12. Kimodo on the work desktop — results 2026-11-02/03 (local-models session)

Run by the work-desktop session that also built the generated-asset pipeline
(local-models repo, skill `game-asset-pipeline`; Beatrice #154, carbine #160). It covers
most of `kimodo_spike_workflow_2026-11-02.md` Phases 0, 1 and 3; the beast prompts
(B1–B8) and the zombie questions are **not yet run**.

**Setup (Windows, no Docker/WSL):** repo `C:\Users\steen\motion\kimodo`, conda env
`kimodo` (Python 3.10, torch cu128). The text encoder is LLM2Vec on the **gated**
Meta-Llama-3-8B-Instruct (Erik's Hugging Face access granted 2026-11-02). The 12 GB card
cannot hold it: it runs on the CPU in bf16 (`TEXT_ENCODER_DEVICE=cpu`, ~16 GB through RAM
+ swap). 8-bit on the GPU fails (LLM2Vec merges a LoRA into the base on load, which
bitsandbytes refuses). Loaded once per batch, generation then takes **~50 s per clip**
(2 samples, 100 steps). Model: Kimodo-SOMA-RP-v1.2.

**Clips made (SOMA, 2 samples each):** walk, run, rifle carry walk, rifle aim walk, rifle
shoot, wind-hunched walk, hit reactions (head front, chest front, back, left, right).
Stick-figure review: hit reactions read clearly (double over, sideways recoil, lurch);
the wind walk is hunched with an arm shielding the face; rifle clips have plausible arm
poses. The rifle-shoot clip lowers and raises the gun mid-clip.

**On a character:** Beatrice (a TRELLIS mesh in A-pose) was bound to SOMA and played the
clips, with the carbine parented to `RightHand` (grip in the palm, barrel toward the
support hand). Lessons: SOMA's `Root` and `*End` bones must not deform (bone heat gave
`Root` 1 228 vertices, which stayed behind as streaks); vertices left unweighted take
their nearest weighted neighbour's; a holster near the A-pose hand gets hand weights and
stretches — weights need a body-part rule for such props.

**Answers to the spike questions so far:** 6 (cost) — above; 7 (retarget onto 53 bones)
— moot, D1 is SOMA; 8 (licence) — §8, generated clips may ship. 1–5 wait for the beast
prompts.
