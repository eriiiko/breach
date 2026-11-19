# Fauna design inputs — larvae first (captured 2026-09-07)

> Append-only capture from the props-arc session (#60), the night the
> procedural tree pipeline landed. This is the INPUT to a future fauna design
> session, not the design itself — that session produces the real
> `docs/architecture/` chapter. Erik's rulings here are settled; everything
> else is Claude's sketch, to be critiqued properly when the arc starts.
> Inspiration mandate from Erik: biology textbooks, botanical gardens,
> entomology, academic literature — explicitly NOT movies / pop culture.

## Erik's rulings (2026-09-07, settled)

1. **Larvae are UNITS** (not props), and **YES: larvae are the FIRST CUDAUNIT**
   — the pilot for the long-parked GPU-resident simplified-unit swarm idea.
2. Queen larva / larger specimens = regular CPU units with more advanced AI —
   later, not v1.
3. Erik wants to know how big a larva can be and still move naturally —
   explicitly: can we make a **10 m** one? (Answer sketch below: yes.)
4. Erik also wants **large killer wasps or bees** modeled the same
   generated way, and suggested **butterflies**.

## Why this is cheap: the #60 toolkit transfers

Arc #60 (props & vegetation, docs/architecture/graphics/props_and_vegetation.md)
built and shipped the machinery a creature generator reuses wholesale:
- `renderer/propgen.py` — seeded numpy → triangle arrays; blob-chains
  (bodies), tube-sweeps (limbs/antennae), ribbons (wings/membranes/fronds),
  tuft quads (bristles/hair), palettes, decor clustering.
- `renderer/lit3d.py` + `renderer/static_props.py` — lit-3D-in-world-RT with
  the baked light field; owned-memory mesh upload; model cache.
- The P4 sway shader — vertex displacement with per-vertex weight baked in
  vertex-color ALPHA, driven by the sim clock (replay-identical) and the
  TAMED wind seam (`gas_detail.tame_wind`), density-scaled (momentum flux).
  A larva crawl is the SAME traveling-wave machinery with phase along the
  spine instead of height.

## The 10 m larva — feasibility sketch (three biomechanics requirements)

Geometry is free (more spine segments). Natural MOTION needs:
1. **Ground-locked peristalsis** — wave phase locked to distance traveled
   (`phase = k·(s − distance_walked)`), so ground-contact segments PLANT
   while the wave passes. Sliding contact points = instant rubber-toy look.
2. **Scale-slowed frequency** — big animals move at lower frequency
   (~1/√length; R. McNeill Alexander, *Principles of Animal Locomotion*).
3. **Follow-the-path spine** — segments follow the head's recorded trajectory
   at arc-length offsets (head-path ring buffer) → natural cornering through
   corridors, tail never sweeps through walls.
Top-down ortho flatters large bodies (full silhouette reads). Larvae are low,
so no canopy-over-marine style occlusion issue.

## Roster of well-suited animals (with the academic sources to crib)

Canonical umbrella source: D'Arcy Thompson, *On Growth and Form*.

- **Butterflies / moths** — wings patterned from the *nymphalid groundplan*
  (H.F. Nijhout, *The Development and Evolution of Butterfly Wing Patterns*,
  1991): seeded eyespots/bands/margins, no textures. Moths are PHOTOTACTIC →
  behavior driven by our lighting (flock to lamps, flashlight beams).
  Butterflies pollinate the tree decor flowers in the gardens.
- **Wasps / bees (Erik's killers)** — three tagmata blobs + petiole waist,
  banded vertex colors (Snodgrass, *Principles of Insect Morphology*, 1935);
  flight rendered as wing-BLUR ellipses + hover bob (wingbeats are unresolvable
  in reality — cheaper AND more correct than animated wings). Procedural
  hexagonal PAPER NEST as a flammable prop → burning a nest angers the swarm.
- **Millipedes / centipedes** — blob-chain + many tube legs in a metachronal
  wave (the traveling-wave shader applied to legs). Giant precedent:
  *Arthropleura*, a real 2.5 m Carboniferous millipede.
- **Giant isopod** (*Bathynomus giganteus*, real) — overlapping dorsal shell
  plates; conglobation (rolls into a ball) as a defense mechanic.
- **Snail shells** — Raup's logarithmic-spiral shell-coiling model (1966):
  a small, seeded, citable generator.
- **Botanical / fungal** — *Physarum* slime-mold glowing networks
  (algorithmically adjacent to space-colonization tree growth), mushroom
  clusters (propgen nearly does them already), glowworms (*Arachnocampa*) —
  luminous hanging threads, made for the light engine.
- **Sessile reactives** — tube worms / feather-duster worms that RETRACT on
  disturbance (nearby movement or blast): one-behavior props that feel alive.

## The O2-gigantism lore hook (ties fauna to the atmosphere sim)

Earth's giant-insect era (*Meganeura*, ~70 cm dragonfly) is academically
explained by hyperoxia: tracheal O2 diffusion limits insect body size, so
high O2 ⇒ giant arthropods. Breach SIMULATES per-tile O2. Giant fauna in the
exotic gardens can be canonically justified — and even mechanically driven —
by O2-rich sections. Physiologically interpretable, which is the project's
research philosophy; it also couples the fauna arc to the fire/O2 work (#12).

## Design forks the fauna session must settle

- Prop vs unit boundary: nests/egg clusters/dormant broods = #60 props
  (flammable stamp — burning a nest is a gameplay verb; could ride the P5
  dressed garden). Crawling/flying creatures = units (digest rules: generated
  body stays render-only, like the marine model — nothing lands on `Unit`).
- Cudaunit architecture: what a GPU-resident simplified unit IS (state layout,
  behavior kernel, collision/fire coupling per segment tile) — larvae are the
  pilot; consult `docs/rl_env_arc_proposal_2026-08-27.md` §A resident-path
  habits and the GPU-residency canon (engine/02) before designing.
- Segmented-body sim representation: a chain of positions (head + arc-length
  followers) vs single-tile units; per-segment damage/fire interaction.
- Wing-pattern generator scope (Nijhout groundplan) — render-only, propgen row.

## Where the rest of this lives

- Arc #60 (trees/props, the toolkit): issue eriiiko/breach#60 + the
  architecture chapter `docs/architecture/graphics/props_and_vegetation.md`.
- Session memory (auto-loads in breach chats on this machine):
  `project_larvae_idea.md`, `project_fauna_ideas.md`.
- Per the credit rule: archive cited papers under `docs/papers/` when the
  implementing arc starts.
