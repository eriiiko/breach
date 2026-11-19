# Fauna zoo — candidate roster (captured 2026-09-07)

> Append-only capture from the fauna design session (issue #63). Companion to
> `docs/fauna_design_inputs_2026-09-07.md` (Erik's settled rulings + the #60
> toolkit transfer notes). This doc is the CANDIDATE POOL: real animals only,
> academic sources only (Erik's mandate — no movies / pop culture), each judged
> against the propgen toolkit (blob-chains, tube-sweeps, ribbons, tufts,
> traveling-wave shader, wing-blur ellipses). Untracked capture; commits with
> the fauna arc. Per the credit rule, cited papers get archived under
> `docs/papers/` when a candidate's implementing patch lands.

## Working mode (Erik, 2026-09-07)

- Store everything as candidates; aim for **~one new animal per week** (or
  similar cadence), and let the repetition **distill a reusable creature
  workflow**: pick candidate → pull the citation(s) → seeded generator →
  spike viewer shot → Erik feel-check → promote (prop row or unit/cudaunit).
- Rescaling is canonically free: the O2-gigantism hook (see inputs doc §5)
  justifies giant arthropods in O2-rich sections; the collector's-menagerie
  frame justifies everything else (insects made big, mammals made small).

## Status

- **Larva — BUILT (spike v1, 2026-09-07)**: `prototypes/fauna_spike/`
  (`larvagen.py` + `view_fauna.py`). Three scales (0.6 m / 2 m / 10 m)
  crawling a cornered circuit with ground-locked peristalsis, follow-the-path
  spine, and frequency ∝ 1/√length (Alexander 2003). Erik verdict:
  "beautiful". Palettes: grub / garden / brood (glowing spiracles).
- **Wasp — BUILT (spike v1, same session)**: `prototypes/fauna_spike/waspgen.py`
  — three tagmata + petiole waist, seeded dark/amber banding (Snodgrass 1935),
  wing-BLUR ellipse discs + hover bob, no animated wings; three hover in the
  viewer. Still owed from the inputs-doc spec: the flammable paper-nest PROP
  and any flight behavior beyond hover.
- Spike-noted look items to judge live (from the build log): contact planting
  is a compromise (compression amplitude A=0.55 → contact points still drift
  at ~0.45·v; harder planting stretches segments), and the brood palette's
  crease rings barely read. Erik should eyeball the interactive run.

## A. Sim-coupled candidates — each has a real, citable mechanism that maps onto a simulated field

| # | Animal | Real mechanism + source | Build | Couples to |
|---|---|---|---|---|
| A1 | **Bombardier beetle** (*Brachinus*) | ~100 °C exothermic benzoquinone spray from a two-chamber gland (Eisner & Aneshansley, PNAS 1999; Arndt et al., Science 2015) | Two dome elytra blobs + tube legs | Deposits real HEAT into the fire sim (#12) |
| A2 | **Pistol shrimp** (*Alpheus*) | Claw-snap cavitation bubble → ~218 dB shockwave + sonoluminescent flash (Versluis et al., Science 2000; Lohse et al., Nature 2001) | Blob + one oversized claw | A `wave_p` impulse + a frame-light, verbatim |
| A3 | **Tardigrade** (scaled to dog size) | Only animal proven to survive open-space vacuum — FOTON-M3 orbital exposure (Jönsson et al., Current Biology 2008); cryptobiotic "tun" state | Wrinkled 5-blob chain + 8 stub tube legs + claw tufts | Decompression: everything dies, tuns revive when pressure returns |
| A4 | **Naked mole-rat** | Survives 18 min at 0 % O2 via fructose metabolism (Park et al., Science 2017); eusocial queen (Jarvis, Science 1981); near-poikilothermic; hairless | Wrinkly pale blob-chain + incisors (no fur problem) | The INVERSE O2 hook: thrives in low-O2 sections where fire can't burn |
| A5 | **Army / leafcutter ants** | Stigmergy — trail pheromone gradients (Wilson & Hölldobler, *The Ants*, 1990) | Tiny 3-blob bodies; swarm | Pheromone = a TRACE GAS in the gas sim; natural SECOND cudaunit species; leafcutters harvest the #60 trees |
| A6 | **Moths** (inputs doc) | Phototaxis | Butterfly build + drab palette | Flock to lamps and flashlight beams — behavior driven by the light field |

## B. Near-verbatim toolkit builds

| # | Animal | Why it's exotic + source | Build |
|---|---|---|---|
| B1 | **Velvet worm** (Onychophora) | ~500 My "walking fossil"; fires twin oscillating adhesive slime jets (Concha et al., Nature Comms 2015) | larvagen body + stub tube legs → a ranged-attack larva nearly for free |
| B2 | **Basket star** (*Gorgonocephalus*) | Recursively branching arms that unfurl into a fractal net, knot up when disturbed | The TREE generator's branching recursion applied to an animal |
| B3 | **Olm** (*Proteus anguinus*) | Blind pale cave salamander; a decade without food; electro-sense (Bulog et al.) | Elongated blob + tiny limbs + external gill TUFTS on the sway shader; ignores light — flashlights useless. (Axolotl = the cute garden variant of the same build) |
| B4 | **Bobbit worm** (*Eunice aphroditois*) | Real 3 m iridescent ambush polychaete | larvagen + bristle tufts; lurks under floor grating, strikes upward |
| B5 | **Amblypygi** (tailless whip spider) | Whip antenniform legs, sideways scuttle, crevice-dweller (harmless in reality — zoo irony) | Flattened blob + two extreme tube-sweep whips |
| B6 | **Coconut crab** (*Birgus latro*) | Real 1 m land arthropod; strongest measured pinch, ~3,300 N (Oka et al., PLOS ONE 2016); climbs trees | Big blob + armored tube legs; interacts with #60 trees |
| B7 | **Pangolin** (small-mammal track) | Only scaled mammal; ball defense | Overlapping plate decor (same trick as the isopod shell); simple silhouette, no fur |
| B8 | **Giant millipede / Arthropleura** (inputs doc) | Real 2.5 m Carboniferous precedent | Blob-chain + metachronal tube-leg wave |
| B9 | **Giant isopod** (*Bathynomus*, inputs doc) | Real deep-sea giant; conglobation defense | Overlapping dorsal plates |
| B10 | **Snails** (inputs doc) | Raup's logarithmic-spiral coiling model (J. Paleontology 1966) | Small seeded citable shell generator |

## C. Garden ambience & charm

| # | Animal | Hook | Build |
|---|---|---|---|
| C1 | **Giant siphonophore** (*Praya dubia*, 40+ m — among Earth's longest animals) | The zoo's showpiece; drifts on the WIND FIELD in vented / zero-g sections | Glowing larvagen chain + glow dots |
| C2 | **Crinoids / sea pens** | Sea pens bioluminesce WHEN TOUCHED → touch-triggered frame-light | Feathery ribbon + tuft arms |
| C3 | **Honeypot ants** | Living amber storage vessels hanging from the ceiling, glow when backlit | Amber blobs + light engine |
| C4 | **Peacock spider** (*Maratus*, scaled to cat size) | Courtship fan dance | One colorful ribbon flap + a dance loop |
| C5 | **Butterflies** (inputs doc) | Nymphalid groundplan wings (Nijhout 1991); pollinate the tree flowers | Seeded eyespot/band ribbon wings |
| C6 | **Physarum / glowworms / tube worms** (inputs doc) | Glowing networks, luminous threads, retract-on-disturbance | Adjacent to space-colonization growth; sessile reactives |

## Suggested first-five greenlight (Claude's pick, to be ruled on)

Tardigrade (A3), bombardier beetle (A1), naked mole-rat (A4), velvet worm
(B1), basket star (B2) — each exotic, cheap on our primitives, and four of
five couple to a field we already simulate.

## Systems

- **Uses (existing canon)**: propgen/lit3d/static_props + the P4 sway shader
  (arc #60); units + digest rules for anything that moves; the cudaunit pilot
  (larvae) per issue #63; gas tables if pheromone-as-trace-gas is pursued.
- **Creates (future rules, draft)**: a creature generator family
  (`larvagen`-style, one module per body plan) — the fauna arc's design doc
  must decide whether these fold into one `faunagen` registry or stay
  per-plan modules; the weekly-cadence workflow itself, once distilled, may
  become a skill.
