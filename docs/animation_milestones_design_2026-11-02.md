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

### 4.5 Review of §4.1 (Opus, 2026-11-02, lens: architecture + feasibility)

One independent reviewer, read-only, with probes against the real binding (raylib / pyray
**5.5.0.4** on the home desktop, the three shipped looks). Probe scripts (not in the
repo): `…\scratchpad\pose_review\p1–p8`. **Verdict: GO, with changes.** The changes are
folded into this section and into #162–#164; this section is the spec #162 is built to.

**How our composed pose reaches the skinned mesh (the key unknown — answered).**

| Option | Result (measured) | Use |
|---|---|---|
| **(a)** write our pose into a scratch `ModelAnimation` (`frameCount=1`, `bones` = the look's anim `bones`, `framePoses` = our `Transform[J]`), then `update_model_animation(model, scratch, 0)` | exact (injecting raylib's own frame pose gave max \|dv\| = 0.0), 0.107 ms/unit (= today), ~20 lines. `framePoses` must hold **model-space (global) TRS**, not local. The field is `framePoses` on 5.5 and `keyframePoses` on the 6.x bindings (#80) | **slice 1 (CPU)** behind a binding adapter |
| (b) own numpy skinning + `update_mesh_buffer` | works, 1.07 ms/unit (10× slower) | no |
| (c) GPU skinning: write `mesh.boneMatrices`, skinning vertex shader | works on 5.5 (the module docstring saying no GPU helper exists is **wrong** for 5.5); 0.8 µs/unit to write the matrices, 20 units incl. draws 0.20 ms vs 2.59 ms CPU. Needs `MARINE_VS` and `SHADOW_VS` to skin; a look is all-CPU or all-GPU; **the marine byte-identity test breaks by construction** | **W13.2**, with a deliberate re-baseline |

**Seam:** the pose module outputs, per unit, global TRS `(J,10)` and skin matrices
`(J,4,4)`. A `Skinner` has `prepare(look)` and `apply(look, pose_row)`, called just before
`draw_model_ex`. `CpuInjectSkinner` = option (a); a later `GpuSkinner` writes
`mesh.boneMatrices` (the shadow draw then follows automatically).

**Corrections to §4.1:**
1. **Time base — the biggest feel risk.** raylib 5.5 bakes clips on a 17 ms grid;
   `_draw_one` advances `_ANIM_FPS = 30` grid frames/s, so **every clip plays at ≈0.51×
   speed** (Walk_Loop's 1.33 s cycle takes 2.6 s). A correct sub-frame sampler at 1×
   doubles every speed. **Playback `rate` is data; its default reproduces today's 0.51.**
   *Amended by the #162 implementer (verified against raylib's loader):* the reviewer's
   "snaps to the floor key, no interpolation" holds only for some bones. raylib 5.5 stores
   ONE interpolation type per bone (overwritten by that bone's last channel) and applies it
   to all three channels, so bones whose last channel is STEP hold the floor key while the
   others are slerped. Of the 44 non-stripped clips, 19 are all-step (incl. Walk_Loop) and 25
   mixed (incl. Idle_Loop). The pose layer samples each channel by its own interpolant, so
   mixed clips will look smoother than today's player even at rate 0.51 — a HUMAN-TEST item.
2. **Batching is mandatory.** Per-unit numpy costs 0.5–1.9 ms/unit (fails at 20 units);
   batched over all units with matrix FK by depth level: 1.7 ms at 20 units, 7.4 ms at 100.
   One `compose` per `draw_units` call, grouped by look (`draw_units` runs twice per frame:
   marines, then zombies). Clips are baked once per look at load into dense `(K,J,3/4)`
   arrays with hemisphere-continuous quaternions. FK as 4×4 matmuls per depth level (12
   levels here), never a per-bone Python loop. 100 units at 60 fps needs GPU skinning
   **and** the batched pose, and probably numba / C++ later (numba is not a declared
   dependency).
3. **Additive reference.** `delta = inv(ref) · clip` in local space, **ref = the clip's own
   reference frame (frame 0), never the bind rest** (rest would add the clip's whole
   stance); translation `T_base + (T_clip − T_ref)`. Fixed compose order: base → overrides
   in order → additives → procedural modifiers (post-FK) → skin.
4. **Own glb reader** (json + numpy, ≈80 lines); do not reverse-engineer raylib's stepped
   `framePoses`. Today's data is clean (T/R/S tracks, scale all 1, STEP channels constant,
   only `DEF-hips` and `root` have animated translation, inverse binds agree with the node
   rest, root has a −90° X rest rotation) but the reader **refuses CUBICSPLINE, matrix
   nodes and multiple skins by name**, so a SOMA skeleton later fails loudly.
5. **Masks and parent chains.** Local-space blending inherits correctly. But hips sit in
   `legs`, so the torso inherits the walk's hip yaw/roll and the aim wobbles → an optional
   per-mask "model-space blend" for the first upper bone. A head **look-at** needs the
   parent's global pose: a post-FK modifier that re-propagates its subtree (interface
   defined now, built later).
6. **Root motion.** `Roll_RM` moves the root 5.1 m, `Sword_Attack_RM` 1.5 m: a per-clip
   `root_motion = "strip"` (zero the root's horizontal translation at bake), or the body
   leaves its sim position.
7. **Aim vs travel (#163).** No strafe/backpedal clips exist; masks alone make the marine
   skate sideways. Needs a procedural modifier: the lower body yaws toward travel within
   ±θ, the spine counter-twists to `facing`, beyond θ the clip runs at a negative rate.
8. **Per-look data.** The three looks share joint names/order but have different rests and
   per-look converted clips: clip banks and inverse binds are per look, batches group by
   look.
9. **Off path byte-identical** = a property, not a pixel diff: with the setting off the
   pose module is never constructed and `update_model_animation` receives the look's own
   anim and an int frame.

**Finding in another system (recorded, not fixed here):** raylib 5.5's CPU skinner
transforms *normals* by the full bone matrix including translation; measured normals are
0.25–1.96 long and up to 75° off, 6,268 of 7,029 vertices off by > 5°. Today's marine
lighting (and its tuned gloss / wrap / light_height dials) is therefore shaded with wrong
normals on 5.5. Option (a) keeps today's shading; the GPU skinner (W13.2) fixes it, so
the units will *look different* then — a HUMAN-TEST item. Filed as its own issue.

**Minimal design for #162 (the spec):**
- `renderer/pose_clips.py` — pure, pyray-free glb reader: `Skeleton` (names, parents,
  depth levels, rest locals, bind inverse) and a per-look `ClipBank` (baked T/R arrays,
  rate, loop flag, root_motion, additive reference frame).
- `renderer/pose_layer.py` — pure numpy: `MaskTable`, per-unit `PoseState` (per layer: clip,
  time, weight, ramp target), batched `compose(states) -> (global_trs (N,J,10), skin
  (N,J,4,4))`.
- `renderer/pose_skinner.py` — pyray side: `CpuInjectSkinner`, one scratch `ModelAnimation`
  per look with its cffi owners kept alive, behind a 5.x / 6.x field adapter beside
  `anim_frame_count`.
- Data in config.toml: `[render.pose]` `enabled = false`, `rate` (default 0.51),
  `crossfade_s`; masks per skeleton as subtree/minus rules over bone names (`legs`, `upper`,
  `head`), validated at load (unknown names refused, partition checked); a state table
  replicating `CLIP_MAP` plus per-clip rows (rate, loop, root_motion).
- Per frame: `consume_events` queues each unit's events → `draw_units` updates state
  (motion inference as today, table lookup, ramps by `dt`) → one batched `compose` per look
  group → per unit `skinner.apply`, then projected shadow, blob shadow, `draw_model_ex`
  unchanged. Per-unit state lives in `_UnitAnimState`, pruned with the existing `_prune`.

**Tests, as properties** (each docstring names the property and the change that breaks
it): (1) sampler fidelity — our global pose at a key time equals raylib's `framePoses` at a
frame that snaps to that key, ≤ 1e-5, for every shipped look and clip (breaks on bone
order, parent chain, quaternion convention, time base); (2) continuity across key
boundaries and loop seams (breaks on an off-by-one, a missing hemisphere fix, integer-frame
sampling); (3) masks resolve, partition the bones, refuse unknown names (breaks on a
hard-coded or typo'd list); (4) layer independence — changing the upper clip leaves legs
and head locals and legs globals unchanged; (5) inheritance — the legs clip's hips move an
upper bone's global (breaks on global-space compose); (6) crossfade — weight 0 = source,
1 = target, angle to target monotone non-increasing; (7) additive identity — at its
reference frame or weight 0 the pose is unchanged; (8) root-motion strip — a stripped
clip's root horizontal global translation is constant; (9) skeleton-agnostic — a synthetic
5-bone glb with its own masks composes, and the two pure modules contain no bone-name
literal; (10) render-only — compose takes read-only attributes, a frozen unit stub raises
on any write; (11) off path as in correction 9; (12) `gl_pose_inject_check.py` + a `test_*`
wrapper (hidden window, subprocess): injecting raylib's frame-k global pose reproduces
`update_model_animation(anim, k)`'s `animVertices` bit for bit, on each machine's binding.
Proof beyond tests: `tools/e2e_drive.py --shot` and a look shot, then HUMAN-TEST.

**Ranked risks:** the 5.x / 6.x binding split breaks injection on the other PCs (this
split already crashed the game once, #80 — mitigation: the adapter, our own reader, test
12 on every machine); the speed/smoothness change is certain (mitigation: `rate` data,
default 0.51, HUMAN-TEST); unbatched numpy blows the frame budget; #163 skating without
the aim-vs-travel modifier; normals change at W13.2; a wrong additive reference; 100 units
at 60 fps needs more than the CPU path; #164 posture offsets bend knees without foot IK
(mostly hidden top-down) and the melee clip plays late, on the hit event.

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
  three; the CPU-inject skinner keeps the marine byte-identity gate (`test_lit3d_extraction`),
  the GPU skinner (W13.2) re-baselines it deliberately, once, with written rationale (§4.5).
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
— moot, D1 is SOMA; 8 (licence) — §8, generated clips may ship.

### 12.1 The beast prompts (2026-11-03, 3 samples each, Kimodo-SOMA-RP-v1.2)

Prompts as in the spike workflow §4. Numbers from `clip_metrics.py` (local-models
`motion/`): spine lean = hips→chest angle from vertical, averaged over the first and the
last third of the clip (does a posture hold?). Clips and GIFs on the work desktop in
`C:\Users\steen\motion\clips\beast_v1\`; the best ones also played on the male worker
(SOMA-bound, as the zombie would be).

| id | travel (m) | lean first / last (°) | hips min–max (m) | verdict |
|---|---|---|---|---|
| C0 walk (control) | 4.3–6.7 | 4–7 / 3–11 | 0.90–1.00 | works |
| B1 hunched claw walk, text only | 3.8–6.4 | 24–25 / 23 (2 of 3); 4 / 6 (1 of 3) | 0.69–0.96 | **partly**: 2 of 3 hunch and hold it, hands ~0.3 m forward; 1 of 3 walks upright |
| B2 = B1 + full-body keyframe ×3 | 5.0 (the path) | 30–35 / 30–36 (3 of 3) | 0.70–0.86 | **yes**: every sample holds the hunch for the whole 5 s |
| B3 sprint like an animal | 3.9–7.8 | 43–48 / 43–49 | 0.42–0.56 | **yes, as a crawl**: fast on hands and knees (head ~0.6 m), not a four-legged gallop |
| B4 coil, lunge, swipe | 0.04–1.15 | 3–9 / 5–19 | 0.70–1.18 | **partly**: one sample lunges with a swipe; two barely move |
| B5 sway, twitch | ~0 | 3–14 | 0.96–1.00 | **weak**: standing with small head motion; the twitch is barely there |
| B6 knocked down | 1.2–1.4 | 10–20 / 88–101 | 0.11–0.95 | **yes**: falls and lies flat in all 3 |
| B7 get up | 0.2–0.5 | 58–69 / 7–14 | 0.06–1.01 | **yes**: from lying to standing in all 3 |
| B8 heavy bag on one shoulder | 4.0–6.1 | 5–12 | 0.94–1.01 | **yes, plausibly**: one hand held at the shoulder while walking (no bag object exists, the hand pose carries it) |

**The spike's questions:**
1. Text alone gives the hunched, arms-forward posture — **partly** (2 of 3).
2. A full-body keyframe holds it through a walk — **yes** (3 of 3, 30–36° the whole clip).
   This is the reliable route for the zombie gait: author the pose once, keyframe it.
3. Animal-like locomotion — **partly**: a fast crawl on hands and knees exists; a true
   four-legged run did not appear.
4. Attack / idle / fall / rise — fall and rise **usable**; the lunge **sometimes**; the
   twitchy idle **weak** (a candidate for the posture layer's `twitch` additive, §5.2,
   rather than a clip).
5. Heavy carried object — **usable as a gait** (hand at the shoulder); the bag itself is a
   prop on a bone.

## 13. As built — the pose layer plays SOMA (#179, 2026-11-07)

Branch `179-soma-pose`. The pose layer now plays `assets/models/soma_worker_male`
(30-joint SOMA, 12 Kimodo clips) beside the Quaternius looks, in one run, as data. The
Quaternius looks compose bit for bit as before (a scripted 240-frame run, 4 units, every
shipped look: max |diff| 0.0 against the pre-#179 code; held by
`tests/test_pose_soma.py::test_the_quaternius_looks_ignore_every_other_skeletons_table`).

**What was built.** A `[render.pose.skeletons.<s>]` table may carry its own playback
(absent = the global tables): `clip_names` (the canonical names the shared tables use —
the Quaternius pack's — to the skeleton's own; `""` = no such clip: a state falls back, the
reaction needing it does nothing), `rate` (replaces the global 0.51), `clips` (per-clip
rows keyed by the skeleton's own names, replacing `[render.pose.clips]`) and `roles`
(per-role `posture` + `states` rows). The look's skeleton table is now picked from the
file's skeleton BEFORE the bake, so its own clip rows shape the bake. `[render.pose.skeletons.soma]`:
masks (legs = Hips − Spine1, upper = Spine1 − Neck1, head = Neck1), twist (Hips; Spine1,
Spine2, Chest at 0.4/0.3/0.3), the posture maps, the clip names, the rates, a zombie role.
Tool `tools/pose_lineup.py` (top + side stills of named scenes through the real skinner).

**Deviations from the brief.**
- The per-skeleton keys are OPTIONAL (absent = the global key), not required: requiring
  them would have meant moving the global `rate` and `[render.pose.clips]` into the
  Quaternius table and rewriting the tests that pin them. Unknown keys are still refused,
  every value is type-checked, a `clip_names` key no table uses is refused, a mapped name
  the file lacks is refused at load.
- The per-role override lives in the SKELETON table (`roles.zombie`), not in the global
  `[render.pose.roles]`: only SOMA has the hunch clip and its posture amount must differ
  per skeleton (0.0 on SOMA, 1.0 on Quaternius), so both are per skeleton.
- `arm_open` has no SOMA bones (`arm_open = []`): the hands are rigid; mapping it to a
  wrist bend read as nothing a claw could be. The neck thrust (50°) is shared 25/25 by
  Neck1 and Neck2.
- The posture signs carried over unchanged: model-space rows mean the same motion on any
  rig facing +z, +y up, +x its left, and SOMA's A-pose rest is within a few degrees of the
  Quaternius workers' (both rigs have +y along the bone). Checked by FK (each parameter
  moves the head / hands / hips the same way on both rigs, > 2 cm, tested) and by the
  lineup renders.
- The SOMA melee reaction is OFF (`Punch_Jab = ""`, `Punch_Cross = ""`): no strike clip.

**Measured.**
- Clip stride speeds (stance-foot backward speed, from the clip): Walk 1.344 (left) /
  1.303 (right) m/s, Rifle_Aim_Walk 1.27, Zombie_Hunch_Walk 1.014, Run 3.2–3.6 — the
  work PC's figures confirmed. Every looping clip ends on its first key (< 1.2e-7).
- Unit speed on screen: marine (`marine_attack`, 18 ticks/tile at 24 tps) 1.333 tiles/s,
  zombie (14 ticks/tile) 1.714 tiles/s. The renderer draws the space marine (1.885 model m)
  6 tiles tall, so one drawn tile = 0.3142 model m: 0.419 m/s marine, 0.539 m/s zombie
  (in SIM metres, at 0.333 m tiles: 0.444 and 0.571 — the drawn figure is the one feet
  are seen against). Rates (skeleton rate 1.0 × clip row): Walk 0.312, Rifle_Aim_Walk
  0.330 (the overlay's arm swing at the legs' tempo), Zombie_Hunch_Walk 0.531. Held
  within 5 % by `test_a_soma_walk_strides_at_the_units_drawn_speed`. Consequence: the
  shipped marine walks slowly (0.42 m/s), so the walk loop plays in 7.0 s of wall time —
  slow, long steps; a faster `move_speed`/cadence would bring the steps to life.
- Rifle_Shoot is a HELD AIM, not a recoil: over its 3.97 s no bone moves more than 3.5°
  from its first key (the "lowers and raises the gun mid-clip" seen on the work PC is not
  in the joint rotations). As the shot one-shot it restarts on its own first key, so a
  shot is invisible on SOMA beyond the aim pose; the 4 s length needs no trim (the aim
  hold, 1 s, ends it). A recoil clip is a generation job.
- Hit_Chest as the flinch additive (delta against its own first key — itself the guarded
  stance — so its guarded end cancels): peak 49° at clip 1.03 s; its last 0.2 s still
  holds ~15° (the left hand), dropped over `ramp_out_s` 0.12 s at ≤ 2.45°/frame (60 fps),
  against the Quaternius flinch's own 6.5°/frame ramp-out — no pop worse than today's.
  Clip rate 1.5 (PLACEHOLDER): the flinch lasts 2.0 s wall, peak at 0.7 s.
- raylib 5.5 INTERPOLATES the Kimodo clips when it bakes its 17 ms frames (the composed
  pose at frame f's own time skins to 1.7e-6 of raylib's), while it shows the floor key of
  the Quaternius pack; the GL check accepts either (`tests/gl_pose_inject_check.py`,
  `EXTRA_LOOKS`).
- Cost (pose layer drive + compose, this desktop): SOMA 0.55 ms at 1 unit, 1.75 ms at 20;
  Quaternius worker 0.69 / 3.44 ms (53 bones vs 30; same 6818 vertices).

**What the renders show** (`tools/pose_lineup.py`): idle, walk, the aim walk (empty hands
held at chest height forward — reads as a guard without a rifle prop), mid-flinch (doubled
forward, head down), fall (lying flat, moved by its travel), get-up, the hunch walk at
posture 0 (crouched, one arm reaching) and at posture 1 (the stalk profile on top: much
deeper, the head thrust low — too much on a clip that already hunches; 0.0 stays the
default, Erik's call). The SOMA idle stands ~21° shoulder-turned, as the Quaternius idle
does (18°).

**Registered, unused:** Run (no run state), Hit_Back / Hit_Left / Hit_Right (the hit event
has no direction), Get_Up (knockdown is #149). `dead` = Fall, dormant (dead units are not
drawn). A SOMA zombie at posture 0 has no desync / tempo jitter (both ride the posture
noise, drawn only above 0), so a SOMA horde walks in step.

## 14. As built — the new marine in play (#184, 2026-11-10)

Branch `184-marine-in-play`. The player is drawn with `soma_space_marine` (SOMA, 15 Kimodo
clips) holding the CQB carbine; render side only (no sim state, nothing digested moves). The
Quaternius looks compose bit for bit as before (`tests/test_pose_soma.py::
test_the_quaternius_looks_ignore_every_other_skeletons_table`, `tests/test_pose_locomotion.py::
test_a_look_without_locomotion_steers_exactly_as_before`).

**What was built.**
- `renderer/pose_locomotion.py` + `[render.pose.skeletons.soma.locomotion]`: per role, gaits
  (player walk / run / sprint; zombie one walk) and the walk's four direction sectors. Every
  looping clip's stride speed, travel direction and gait phase are measured at load
  (`measure_gait`: the planted feet's median speed along their mean direction; the phase from
  the fundamental of the feet's height difference) -- the one measurement, which
  `tests/test_pose_soma.py` checks against its own simpler one. A gait clip plays at the drawn
  speed / its stride; the gait steps on the CLIPS' rates (`max_rate` x (1 +- hysteresis)); the
  legs twist by the residual against the playing clip's own measured travel; one gait phase per
  unit locks every gait clip on every layer (the crossfade source too), so a switch carries the
  step over and the run's rifle-carry arms swing in step with its legs.
- Drawn speed: the distance since the last position change over the wall time since it,
  distance and time low-passed apart (one weight per sample), held across the frames between
  sim ticks; stopped after `stop_after_s` without a change (the player 0.15 s; the zombie 0.9
  s: a zombie hops one tile every ~14 ticks).
- `[render.pose.skeletons.soma] states` (the skeleton's own rows), `hit_directions`, `sockets`;
  `[render.pose.reload]` / `[render.pose.throw]` (clips blank); `twist.pivot`.
- `renderer/held_props.py` + `[render.held_props]`: the carbine (`k5_carbine` -> `carbine`) on
  the `right_hand` socket, through the units' lit shader and in the projected shadow.
- Directional hits in the renderer's hand-off: the attacker's last drawn centre (a per-uid map
  across the two `draw_units` calls), for an explosion the nearest blast centre of the frame's
  `ExplosionEvent`s; the pose layer picks the row nearest.
- Tools: `tools/pose_lineup.py` (`loco_*` / `hit_*` scenes, `--prop`, `--focus-socket`,
  `--side-from`), `tools/e2e_drive.py --wheel`.

**Decisions (Claude's, inside the brief).**
- The forward walk is the AIMING walk (`Rifle_Aim_Walk` on every layer): the strafes and the
  back walk only exist aiming, so the marine reads the same in every direction. The strafes and
  the back walk take only their LEGS from their own clips; torso, arms and head play
  `Rifle_Aim_Walk` (phase-locked), because `Rifle_Aim_Left`, `Rifle_Aim_Back` and `Rifle_Shoot`
  hold the rifle LEFT-handed (right hand forward on the handguard: in the clips' middle keys
  the right hand is 0.69-0.82 m forward, the left 0.41-0.50) and the carbine in the right palm
  cannot follow them. Run and sprint play `Run` on the legs and the rifle-carry `Walk` on upper
  and head (Run's own arms pump empty-handed).
- The shot one-shot is OFF on SOMA (`Pistol_Shoot = ""`): `Rifle_Shoot` is left-handed and a
  held aim with no recoil (§13); a shot holds the aim overlay. (A deliberate change: the SOMA
  test that pinned a resolvable shot clip now accepts a declared-absent one.)
- Sector choice by the config's centres (0 / 90 / 180 / -90, 10 deg hysteresis); the twist
  aims at the clip's MEASURED travel, so a strafe that walks at 77 deg is turned 13 deg.
- A unit running sideways (faster than the walk's `max_rate`) runs FORWARD with its legs
  twisted up to `theta_deg` and backpedals past it (the two-sector rule), as the brief allowed.
- The twist pivot is the unit's own vertical line on SOMA (`pivot = "origin"`): about the
  swaying hips the planted foot slid 25 % of the walk speed at a 58 deg residual; about the
  origin 5 % in every direction. The Quaternius table keeps `"bone"` (bit-identity).
- Grip: the grip point is the top of the pistol grip (`attach_weapon.py`'s rule: (-0.1006,
  -0.0749, 0) in the file's units); it sits at the palm (half way from the wrist to the middle
  finger's end), the barrel toward the left palm, its top up, averaged over `Rifle_Aim_Walk`:
  `offset_m = [0.0017, 0.096, -0.0327]`, `rotation_deg = [123.36, 49.24, 98.12]`. Gloss 0.12
  (the roughness map's mean 0.45 through the game-ready mapping).
- Reload / throw: the clip-time scaling is `TimedShot`; a wind-up at least as long as the
  clip's own release time plays the clip slowed from its start, a shorter one starts it part
  way at authored speed; 0 ticks (today's instant throw) plays the follow-through from the
  release.

**Measured (soma_space_marine; drawn m/s = sim m/s / 0.333 x 0.3142).**
- Strides (model m/s of clip time): Walk 1.397, Run 2.601, Rifle_Aim_Walk 1.449,
  Rifle_Aim_Left 1.066 (travel +77.0 deg), Rifle_Aim_Right 1.066 (-76.8), Rifle_Aim_Back 1.411
  (-163.0), Zombie_Hunch_Walk 1.070. (soma_worker_male: Walk 1.338, Run 3.327,
  Rifle_Aim_Walk 1.271, Zombie_Hunch_Walk 1.014.)
- Rates today: walk 0.42 m/s drawn -> Rifle_Aim_Walk 0.29; full speed (the sprint cadence,
  0.94) -> 0.65 forward, Rifle_Aim_Left 0.88 sideways (no run: the shipped sim never moves fast
  enough). A SOMA zombie 0.54 -> 0.53 (the old literal was 0.531).
- Rates at #183's speeds: walk 1.4 m/s (1.32 drawn) -> 0.91; full 4.5 (4.25) -> Run 1.63, the
  sprint state; sideways 1.12 (1.06) -> 0.99; backward 0.84 (0.79) -> 0.56. Walk -> run at
  2.23 m/s drawn (Rifle_Aim_Walk at 1.54), run -> sprint at 3.58 (Run at 1.375); back down at
  2.93 and 1.83. The run's upper `Walk` then plays at 1.455 x Run's rate (2.4 x its authored
  speed at full speed): one arm cycle per stride.
- Skating (the planted foot against the ground, share of the unit's speed): walk gaits 3.5-8 %,
  run 14 % (its flight phase blurs the planted test), the same in every direction.
- The speed estimate: mean within 1.2 % of the true speed at 60 fps (3 % at 30 fps).

**Findings in the assets (Erik's).**
- `Hit_Right` lurches the head 0.55 m to the body's RIGHT at 0.77 s -- the same way as
  `Hit_Left` (0.53 m) -- so a hit from the right leans the marine toward the shooter. It wants
  regenerating, or a mirror of `Hit_Left` (as `Rifle_Aim_Left` is of `Rifle_Aim_Right`).
- `Rifle_Shoot`, `Rifle_Aim_Left` (the mirror of a right-handed strafe), `Rifle_Aim_Back` are
  left-handed (above).
- With `[render.pose] enabled = false` the old clip player reads `CLIP_MAP`'s Quaternius names
  as they are, so the SOMA marine plays its first clip in every state: the pose-off revert wants
  `player = ["space_marine"]` back as well.

**Waits on #183.** `ReloadEvent(unit_id, ticks)` and `ThrowEvent(unit_id, kind, ticks)` (looked
up by name; inert until then); the reload and throw clip names and the throw's `release_s`.

## 15. As built — the clip mirror and left-handed units (#187, 2026-11-11)

Branch `187-clip-mirror`. Render side only: nothing on `Unit`, no sim state, no digest.

**The mirror.** `renderer/pose_clips.py` `MirrorRig`: a skeleton table's `mirror_pairs`
(SOMA `[["Left", "Right"]]`, Quaternius `[[".L", ".R"]]`) pairs each bone with its partner;
the mirror image of a pose is `G'_j = M G_{m(j)} A_j` with `A_j = bind_{m(j)}^-1 M bind_j`
(M: model x -> -x). Equivalently `skin'_j = M skin_{m(j)} M`, so a symmetric mesh deforms as
the mirror image whatever the bones' rest frames. `A_{m(j)} = A_j^-1`, so mirroring twice is
the identity and the bind pose is a fixed point.

**Rest symmetry, measured.** Quaternius (`worker_male`, `space_marine`): symmetric to 1e-6 m,
`A_j` = diag(-1, 1, 1) to 0.12 deg. SOMA (`soma_space_marine`): NOT exactly -- paired joints
up to 7.6 mm off each other's reflection (`LeftHandMiddleEnd`), bone frames up to 0.76 deg,
`A_j` translations up to 10 mm. Hence the conjugation by the rests (exact for any rest)
rather than a bare reflection of local rotations. Parents are partner-consistent on all four
looks (required by the local form; checked at load).

**Two uses.**
- A clip row `mirror_of` is a new clip baked once at load in local space,
  `L'_j = A_p^-1 L_{m(j)} A_j` (a root's left factor M), from the source's raw keys, then its
  own row's root-motion strip. The local bake equals the global mirror to 7.5e-7.
- A left-handed unit (`PoseState.mirrored`) composes in the mirrored world (its travel and a
  hit's source reflected on the way in: locomotion sectors, the twist, the flinch row all
  follow) and its globals are mirrored after the modifiers, one einsum over the left-handed
  rows. A right-handed row is untouched bit for bit. The held carbine moves to the partner
  bone at the derived grip `A_R @ grip @ Z` (Z: the prop's `mirror_axis`, z for the carbine):
  the right-hander's carbine reflected and made proper -- no configured left-hand grip.

**Handedness deal.** `[render.unit_looks] left_handed = { player = 0.10, zombie = 0.10 }`
(PLACEHOLDERS), `unit_looks.HandAssigner`: per role, by ascending unit id at first sight, the
k-th unit left-handed when round((k+1)s) > round(ks) -- the 5th, 15th, ... at 10 %. Kept for
life (a marine turning zombie keeps it).

**The #184 findings, measured (hands' forward z in the clips' middle half, model m; the
head's sideways swing in the hit clips; +x = the body's left).**

| clip | left hand z | right hand z | hold | its mirror |
|---|---|---|---|---|
| Rifle_Aim_Walk | 0.65 | 0.36 | right-handed | left-handed |
| Rifle_Aim_Right | 0.68 | 0.40 | right-handed | = a left-handed LEFT strafe |
| Rifle_Aim_Left | 0.40 | 0.68 | left-handed | = a right-handed RIGHT strafe |
| Rifle_Aim_Back | 0.49 | 0.81 | left-handed | right-handed back walk (travel +163 deg) |
| Rifle_Shoot | 0.41 | 0.68 | left-handed | right-handed |

Hit_Left swings the head 0.45 m to the body's right, the file's Hit_Right 0.50 m ALSO to the
right; Hit_Left mirrored 0.46 m to the left.

**What changed in the config, and the #184 workarounds.**
- REMOVED: `Pistol_Shoot = ""` on SOMA -- the shot one-shot is back, as `Rifle_Shoot_Mirror`.
- REMOVED: the back walk's legs-only row -- `walk_back` plays `Rifle_Aim_Back_Mirror` whole.
- REMOVED: the right strafe's legs-only row -- `walk_right` plays `Rifle_Aim_Right` whole (it
  was always right-handed; #184 took only its legs so every direction read alike).
- KEPT: the LEFT strafe takes only its legs from `Rifle_Aim_Left` (upper and head from
  `Rifle_Aim_Walk`): a mirror swaps direction AND hand, so no right-handed left strafe can be
  made from these clips. It wants a generated clip. (A left-hander gets the opposite: its
  left strafe whole, its right strafe legs-only.)
- `hit_directions` -90 deg plays `Hit_Left_Mirror`; the file's `Hit_Right` is unused.

**Pictures** (`tools/pose_lineup.py`, new: `clip=NAME@F` scenes, `--left-handed`): the
left-handed lineup is the right-handed one reflected, the carbine in the left hand; the
mirrored rifle clips aim the carbine forward from the right hand where the file's point it
down off the right palm.

**For Erik's eye.** The share (10 %), whether zombies should be dealt handedness at all, the
right strafe's own upper body against the aiming walk's, and the shot one-shot's look.
