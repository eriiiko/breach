# Making a scripted character — the workflow

*Written 2026-10-04 (#33), after the space marine and the two workers went from a reference
sheet to animated models in the game. It is the procedure we followed, with what each stage
cost and what went wrong. For meshes that come from outside (Meshy, Mixamo, a marketplace) read
`adding_character_models.md` instead.*

A character here is a **script**, not a hand-made mesh: Python on the kit in
`art/characters/charkit/`, fitted to a reference sheet, rebuilt by one command. The same script
then goes through one more command that makes the game file. Nothing is modelled by hand, so
every correction is a change to a number or a function and a rebuild.

## What you need before starting

1. **A reference sheet** in `art/characters/`: one image with the character from the front, one
   side and the back, at the same scale, standing with the arms a little out, on a plain
   background. All three characters so far were built from such a sheet.
2. **What must be changeable.** Say it up front (skin colour, clothes colour, dirt, a team
   colour): it decides what becomes a table entry instead of a fixed value.
3. **What sets this character apart from above.** The game camera looks down. It sees hair or
   helmet, shoulders, the back, and the overall colour. A face is a few pixels. Team members
   who must be told apart need their difference there: helmet shape or colour, shoulder plates,
   markings on the back, body build.
4. **No props.** A bag, a tool or a weapon is its own asset, never part of the character.

## New family or new member?

- **A new member of an existing family** is a set of tables on that family's builder. The female
  worker is tables only (`art/characters/worker_female/scripts/worker.py`: `PALETTE`, `DIRT`,
  `VARIANTS`, `DIMS`, `HEAD`, `HAIR`); her `build.py` and `game.py` are a few lines each. Workers
  are built by `charkit/workwear.py` + `workerbuild.py` + `workergame.py`.
- **A new family** (a new kind of outfit or body) starts as its own script on the kit. The space
  marine is one (`art/characters/space_marine_claude/scripts/marine.py`).
- **The second member pays for the builder.** When a second character of a family is wanted, the
  first one's script is turned into a builder that reads tables, and both characters become
  tables. Do this before building the second, never by copying the first script. More marines
  therefore start with one step: the marine's script becomes an armour builder.

## The stages

Each stage is one implementation agent in the character branch's worktree, one at a time. The
orchestrator writes the brief, looks at the pictures when the agent reports, and decides the
next stage. Measured on the workers, each stage took one agent 45–50 minutes and about 400,000
tokens (Opus).

### 0. Branch and sheet
Cut `<issue#>-<name>` off the integration line with its worktree under
`.claude/worktrees/<branch>`, and **commit the reference sheet there first**: an agent cannot
see a file that is only in someone's working folder.

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
The orchestrator opens the pictures. The overlap score only measures the outline; it says
nothing about a face, how cloth sits, or where the dirt is.

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

1. **Look at the pictures.** The agents' numbers were accurate every time. Their reports were
   soft on looks: the first worker report named the weak face but not the oversized collar, the
   shoulder pads or the stain on the seat.
2. **Put the first of a family in the real game before building the rest.** Two things only
   showed there: the navy coverall reads light grey from above (the unit shader adds an edge
   glow on top of the colour), and every model is drawn at the same height whatever its size.
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
7. **Spend detail where the camera looks.** Hair, helmet, shoulders and overall colour carry the
   character in the game. Faces came out doll-like and that is fine at game distance.
8. **Give absolute-path commands** to whoever will run them from another folder.

## What this workflow does not cover yet

- Animations beyond the 46 clips of the shared skeleton (rifle holds, zombie gait, operating a
  terminal).
- Equipment as separate assets attached to a bone (weapons, bags).
- Colours changed at run time instead of by re-baking a texture.
