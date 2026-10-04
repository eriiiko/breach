---
name: scripted-character
description: Breach's procedure for making a 3D character from a reference sheet and getting it into the game — the stages (sheet, high-resolution build, review by eye and one correction round, game-ready step, Erik looks, merge), what goes in each agent brief, what each stage costs, and the lessons from the space marine and the two workers. Load before building a new character, adding a member to a character family (another marine, a female version of a male model, a colour variant), or taking a character game-ready.
---

# Making a scripted character

*Status: PROVISIONAL. Written 2026-10-04 (#33) after three characters: the space marine and the
two workers. It records how those runs went, and it is expected to change — see "Keeping this
skill honest" at the end. For meshes that come from outside (Meshy, Mixamo, a marketplace) read
`docs/reference/adding_character_models.md` instead.*

A character here is a **script**, not a hand-made mesh: Python on the kit in
`art/characters/charkit/`, fitted to a reference sheet, rebuilt by one command. The same script
then goes through one more command that makes the game file. Nothing is modelled by hand, so
every correction is a change to a number or a function and a rebuild.

Steps marked **MUST** carry the incident that made them so. Everything else is the default that
worked; a run may do it differently if its report says what it changed and why.

## What you need before starting

1. **A reference sheet** in `art/characters/`: one image with the character from the front, one
   side and the back, at the same scale, standing with the arms a little out, on a plain
   background. Erik supplies one per character.
2. **What must be changeable.** Say it up front (skin colour, clothes colour, dirt, a team
   colour): it decides what becomes a table entry instead of a fixed value.
3. **What sets this character apart for the camera.** Today the camera looks straight down: it
   sees hair or helmet, shoulders, the back and the overall colour, and a face is a few pixels.
   Team members who must be told apart need their difference there. If the camera becomes
   tilted, chests, faces and whatever is carried come into view; ask which camera the character
   is for.
4. **No props.** A bag, a tool or a weapon is its own asset, never part of the character
   (Erik's rule, in the project CLAUDE.md).

## New family or new member?

- **A new member of an existing family** is a set of tables on that family's builder. The female
  worker is tables only (`art/characters/worker_female/scripts/worker.py`: `PALETTE`, `DIRT`,
  `VARIANTS`, `DIMS`, `HEAD`, `HAIR`); her `build.py` and `game.py` are a few lines each. Workers
  are built by `charkit/workwear.py` + `workerbuild.py` + `workergame.py`.
- **A male or female version of an existing character is a new member, not a new design.** The
  body is the dimensions table: her rows were fitted to her own sheet (hips wider than shoulders,
  a drawn-in waist, 1.68 m against his 1.80 m). What she needed beyond numbers was her hair and
  her own pocket layout. Never stretch a finished mesh instead: details stretch with it. Broad
  or padded shoulders undo a woman's build, so armour needs her own shoulder rows too.
- **A new family** (a different outfit construction or body) starts as its own script on the
  kit. The space marine is one (`art/characters/space_marine_claude/scripts/marine.py`). A
  heavy-weapons or demolitions marine whose armour is built differently is a new family that
  shares parts (helmet, gloves, boots) through `charkit/parts.py`.
- **The second member pays for the builder.** When a second character of a family is wanted, the
  first one's script is turned into a builder that reads tables, and both become tables. Do this
  before building the second, never by copying the first script.

## The stages

Each stage is one implementation agent in the character branch's worktree, one at a time. The
orchestrator writes the brief, reviews the pictures when the agent reports, and decides the next
stage. Measured on the workers, each stage took one Opus agent 45–50 minutes and about 400,000
tokens.

### 0. Branch and sheet
Cut `<issue#>-<name>` off the integration line with its worktree under
`.claude/worktrees/<branch>`. **MUST: commit the reference sheet there before spawning an
agent** — an agent cannot see a file that is only in someone's working folder.

### 1. High-resolution build
The agent builds the character from the sheet and renders it.

- Commands (from `art/characters/`): `bash charkit/run.sh <name> --draft --sheet` (about 15 s,
  the working loop) and `bash charkit/run.sh <name> --sheet --turn --beauty all --save` (about
  2.5 min, every picture).
- Acceptance:
  - the outline matches the sheet in all three views (overlap score ≥ 0.90; the workers reached
    0.93–0.96);
  - a feature checklist made from the sheet (every pocket, seam, strap) is complete, and nothing
    is added that the sheet does not show;
  - every colour lives in the palette table, each region has its own named material, and one
    variant picture is produced by changing table values only;
  - every part is a closed shape facing outward (`source/mesh_stats.json`);
  - every other character still builds exactly as before.
- Pictures land in `art/characters/<name>/previews/` with an `index.html`.

### 2. Review by eye, then one correction round
**MUST: someone who did not build it looks at the pictures, with the reference sheet beside
them, before the report is believed.** By default that is the orchestrator. *Incident: the
first worker's report had every number right and named the weak face, but not the oversized
collar, the shoulder pads or the stain on the seat; all three were plain in the pictures.* The
overlap score only measures the outline; it says nothing about a face, how cloth sits, or where
the dirt is.

Look at, in this order: `sheet_vs_reference.png` (outline and features), `hero.jpg` (overall
feel), `head.jpg` and `head_side.jpg` (face, collar, hairline), `torso_back.jpg` (seams, dirt),
`top.jpg` and `elevated.jpg` (what the game camera sees), and the variant picture.

Then one correction round, written as a list of named defects: the picture that shows each one
and what the sheet shows instead. Corrections go into the **shared builder**, shown on the first
character, before the next member of the family is built, so the next one inherits them.

### 3. Game-ready
One command per character makes the game file:
`blender -b --factory-startup -P art/characters/<name>/scripts/game.py` (about 3.5 min). It fuses
the parts into one skin of 10,000 triangles, bakes one 1024-pixel texture, binds the skin to the
game's one skeleton and converts all 46 animation clips. Output:
`assets/models/<name>/<name>.glb` and a `LICENSE.txt`.

- `-- --rig-only` redoes only the skeleton step; `-- --variant <name>` bakes another palette onto
  the same mesh (about 45 s); `-- --evidence-only` redoes the pictures.
- Acceptance: the marine's budget (10,000 triangles, 53 bones, 46 clips); pictures of the rigged
  model in idle, walk, pistol, death and crouch poses plus a few extreme ones, from the front
  quarter and from straight above; close-ups of face and hands; the high-resolution and game
  versions side by side; screenshots from the real game. They land in `previews/game/`.
- See it in the real game without changing the game:
  `python art/characters/charkit/preview_in_game.py --root <checkout> --model <file.glb> --untinted -- --level playground`.

### 4. Review, record, Erik looks, merge
The orchestrator looks at the game pictures, runs the preview command once, and checks that the
models already in the game were not changed by kit edits. It then updates memory and posts a
checkpoint on the issue. Erik looks at the pictures or plays the preview; the branch merges
after that.

Putting the model on units in the game is a separate, play-tested change to the game, never part
of the character's branch.

## Writing the brief

What made the briefs work:

- State the worktree, the branch, that the agent is its only writer, to commit at each step with
  explicit paths, and not to push.
- A **read-first list** of the precedent: the previous character's scripts and its
  `PROVENANCE.md`.
- The rules: colours only in the palette, the material names, no props, closed parts, kit changes
  only by adding (a new argument whose default reproduces today's output).
- Acceptance as numbers and as named pictures, and the sentence "look at every picture yourself
  and fix what plainly reads wrong before you report".
- Open-ended art gets a **bound**: name the defects (three, for the faces) and allow two honest
  attempts each. Without a bound an agent can spend its whole budget on a face.
- The report is a technical record for the orchestrator, with a list of what still looks weak.
  The summary for Erik is the orchestrator's job.

## Lessons

1. **A fresh pair of eyes on the pictures.** The agents' numbers were accurate every time; their
   reports were soft on looks (stage 2's incident). Whether a separate Opus reviewer given only
   the pictures and the sheet catches as much as the orchestrator did is untested: try it on the
   next character and compare the two lists.
2. **Put the first of a family in the real game before building the rest.** Two things only
   showed there: the navy coverall read light grey from above (the unit shader added an edge
   glow on top of the colour), and every model was drawn at the same height whatever its size.
3. **Fix the builder, not the character.** The four corrections made on the male worker cost the
   female nothing.
4. **A table-driven builder makes the next character cheap.** The female worker needed her own
   tables and a bun; everything else was reuse.
5. **Clothes need care in the game-ready step.** Thin cloth disappears when the parts are fused,
   bare fingers merge into mittens, and a stiff toe cap digs into the floor. The worker step now
   handles all three (`charkit/workergame.py`); a new kind of outfit should expect its own
   surprises here.
6. **Check a "nothing changed" claim on the data, not the file.** A rebuilt game file is never
   byte-identical to the shipped one, because its textures are re-encoded (at most one colour
   step in 255). Compare the mesh, skeleton and animation blocks inside the file.
7. **Spend detail where the camera looks.** From above that is hair, helmet, shoulders and
   overall colour; the faces came out doll-like and Erik accepted them. With a tilted camera the
   front of the body counts for more.
8. **Give absolute-path commands** to whoever will run them from another folder.

## What this skill does not cover yet

- Animations beyond the 46 clips of the shared skeleton (rifle holds, zombie gait, operating a
  terminal).
- Equipment as separate assets attached to a bone (weapons, bags).
- Colours changed at run time instead of by re-baking a texture.

## Keeping this skill honest

This procedure is young. Modelling and animation are still being explored, and a skill that is
followed to the letter while it is wrong costs more than no skill.

- **After every character, the orchestrator amends this file in the character's own branch**,
  before the merge: a step that was skipped or done differently and worked, a step that failed,
  a new lesson, a cost that moved. One dated line is enough.
- **A step becomes MUST only with an incident** written next to it. A MUST whose incident no
  longer applies is demoted or deleted the same way.
- **Trying another route is allowed.** Say in the brief or the report which step is being done
  differently and what would count as better; if it was better, it replaces the step here.
- When three characters in a row needed no amendment, drop "PROVISIONAL" from the status line.
