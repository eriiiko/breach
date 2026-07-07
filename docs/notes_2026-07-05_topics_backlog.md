# Notes dump — 2026-07-05 (last Fable day, planning session)

Raw capture of Erik's prepared notes for today's per-topic planning chats.
Topic 1 (smoke/black-body feedback loop) is handled directly in
`blackbody_smoke_and_rendering_brainstorm.md` (§6, decision #2) — it turned out
to already be answered there. This doc holds the rest, lightly organized, each
tagged with its most-likely existing home so a future session can fold it in
rather than starting cold. Not canon — capture only.

---

## Topic 2 — Laser/plasma "bullet holes" glow + light up environment

> Laser / plasma "bullet holes" glow while hot — and cast rays and light up
> environment (flammable items may perhaps start to burn — or the glow + wind
> cause fire).

**Home:** `blackbody_smoke_and_rendering_brainstorm.md` — this is the same
black-body emitter mechanism (Tier A items 1–2) applied to a weapon-heat
source instead of combustion, not a separate system. Once the emitter ships,
this is mostly "add weapon impacts as another `heat`-bearing source feeding
the same LUT." The "may ignite flammables" half ties into the existing
ignition gate (`temperature ≥ ignition_temp`) already in the combustion
lifecycle (§3).

---

## Topic 3 — Smoke visuals upgrade (swirls etc.)

> Add some swirls or something to smoke graphics — see how far we've come with
> the current technique. Want the end result to be really good; unsure how
> much resource to put on sim vs graphics — plan a little upgrade of the smoke
> visuals.

**Home:** also `blackbody_smoke_and_rendering_brainstorm.md` — Tier A item 3
(curl-noise, divergence-free detail velocity) and Tier C item 8 (per-pixel
smoke normal + wisp-normal + flow-map advection) are exactly this ask,
already spec'd. Less a new topic, more a prioritization question: how high do
items 3 and 8 rank against the rest of Tier A/C.

---

## Topic 4 — Rung A/B refinement: realistic scenarios, not empty space

> Plan and execute better simulations / non-deterministic for the rung A/B
> pressure refinement (particle density). Want to see the difference in a
> corridor or room on the spaceship — a more realistic scenario. The first
> gifs were just an explosion in empty space, both looked not that great —
> need interaction with geometry, and smoke: if hot air expands, suspect it
> will look nice on the smoke too.

**Home:** `roadmap_2026-07.md` Phase 1.2 already frames this — "on actual
ship scenarios (corridors, breach, explosion)" — this note is a direct
reinforcement + a concrete ask: **prioritize a corridor/room scenario with
real geometry over an empty-space test**, and check whether hot-air expansion
visibly drives the smoke field (rung A/B + `blackbody` doc §0's adiabatic
expansion cooling would both matter here). Fold into Phase 1.2 when that
arc's spec gets written.

---

## Topic 5 — Map editor: paint tiles, auto-texture, "bent tiles"

> Need a map editor — tiles that can be stitched together (tools exist for
> this, forgot the name). Need to "paint" maps for the coming weeks — doors,
> materials — not autogenerate from scratch. Maps needed for ML training,
> don't need final-quality graphics. Ideal: paint, and the system
> auto-places wall/floor textures where there's wall/floor, from a materials
> set. There's already a level editor to build on, perhaps.
>
> "Bent tiles": for each wall tile, also store its normal direction. A 45°
> wall today is a staircase pattern; if we store per-tile normal direction,
> bullet holes / effects on that wall could render as if it were a true 45°
> surface instead of following the staircase. Less important than having
> levels generated at all, but could be nice.

**Home:** `docs/level_editor_and_format_v2_proposal.md` already exists and
almost certainly overlaps — **not yet read/reconciled this session**. When
this topic gets its own chat, start there before designing anything new
(auto-texture-by-adjacency and the "bent tile" normal-storage idea both need
checking against whatever that doc already decided on the tile format).

---

## Topic 6 — Animal AI + multi-agent bot training architectures

> Animals: leaning toward simple traditional rule-based AI (predators attack
> closest seen; vegetarians graze, ignore players until fired upon, then flee
> or fight) rather than NN-trained — walked back an earlier plan to train
> predator/vegetarian NNs.
>
> **The bigger note — bot training, Counter-Strike-like:** 2+ teams, some
> number of players each. Candidate architectures:
> 1. Port Civulator's "select and move" network to Breach.
> 2. Single-unit-agent: each unit gets the state space (allies + enemies) and
>    acts as its own agent.
> 3. Hybrid: a Civ-like commander network sets targets (what to kill, who
>    attacks what) — "select and move" at the squad level — while individual
>    units use their own AI to execute the assigned goal.
> 4. New design to flesh out: one selector network ranks the top-N targets,
>    then assigns all its units to whichever target it wants. Input = state
>    space; output design "pretty given," needs real design work per
>    architecture.

**Home:** relates to `breach_unit_class_design.md` and
`unit_variants_design_brainstorm.md` (not yet checked this session) and
directly to `cross_project_overlap.md` §6 (RL/NN training) — Civulator's
`StateEncoder`/select-move network is exactly what option 1 proposes porting.
This is a big, foundational topic (ties straight into the project's core ML
goal) — good candidate for a full dedicated session, probably wants its own
new doc (e.g. `docs/multiagent_training_architectures_brainstorm.md`) rather
than folding into an existing one.
>
> **Captured 2026-07-07** → `docs/multiagent_training_architectures_brainstorm.md`
> (exploratory capture, nothing decided; three models + open questions recorded).

---

## Topic 7 — Sound simulation + ML (pressure-field-driven procedural audio)

> Use the simulated pressure waves: track all sounds emitted (perhaps only
> explosions that actually perturb the pressure field). At the listener,
> compute arrival delay = distance / c (speed of sound). Sample the 3×3 or
> 9×9 pressure tiles centered on the listener, train a neural network to
> adjust cutoff/EQ based on the pressure wave. Training data generated via
> our own high-frequency-domain sound simulation. Already partly fleshed out
> from a prior ChatGPT conversation (not yet brought into this repo).

**Home:** no existing doc found this session — standalone topic. **Erik has
additional context from an external ChatGPT conversation that needs to be
brought in** before this topic's real session, or the plan will be working
from a partial picture.

---

## Suggested order for the rest of today (proposed, not decided)

1. **Topic 1 close-out** — walk `blackbody_smoke_and_rendering_brainstorm.md`
   §8's remaining 9 decisions, flip the doc from BRAINSTORM to canon-ready.
   (In progress — see main chat.)
2. **Topic 6** (bot training architectures) — biggest, most foundational,
   most directly serves the project's core ML goal ([[breach-project-goal]]).
3. **Topic 5** (map editor) — TODO.md's own stated priority #1 ("everything
   else blocked on having a real testbed"); needs the existing level-editor
   doc read first.
4. Topics 2–4 are largely *covered* by existing docs/roadmap already (see
   above) — may not need their own chat, just a prioritization pass.
5. **Topic 7** (sound+ML) — needs Erik to first surface the ChatGPT context;
   maybe queue for a later day rather than today.
