# Octopus reference footage sheet + clip scoring sheet (P0)

**Status: P0 of the animation study, 2026-10-29** (brief
`docs/octopus_animation_study_2026-10-29.md` §5). Author: Claude (lead).
Living document: timestamps are filled in as the footage is watched; a row
without timestamps has not been watched yet. Nothing here was watched by
Claude in this session — the sources come from the motor-biology digest §6
and the full-text papers; the EurekAlert pair was verified to exist today,
the Nature and JEB supplement pages could not be fetched (login / wrong
landing page) and are listed as the papers describe them.

Two uses: (1) **the judging target** — every prototype clip (P2–P8) is put
next to a real clip of the same behaviour; (2) **the scoring sheet** — the
14-item craft checklist and the ten "what a game gets wrong" items, filled
in per clip. Erik's eye is the final gate; this sheet makes the verdict
repeatable.

---

## 1. Watch-list

Priority order: paper supplements first (known species, known behaviour,
often with tracking overlays), then documentaries for the long naturalistic
sequences, then the channels. "Primitive" = which §2.1 motor primitive the
clip is the reference for.

### 1.1 Paper supplements

| # | source | where | what it shows (per the paper) | primitive | watched / timestamps |
|---|---|---|---|---|---|
| S1 | Levy, Flash & Hochner 2015, Curr Biol 25:1195 — "Crawling Octopus (1 of 2)" | eurekalert.org/multimedia/650114 (verified 2026-10-29: Cell Press item, caption "Despite its bilaterally symmetrical body, the octopus can crawl in any direction relative to its body orientation") | crawl direction independent of body heading | PUSH_STEP, TURN | — |
| S2 | Levy 2015 — "Crawling Octopus (2 of 2)" | eurekalert.org/multimedia/650113 | the push cycle: mid-arm anchor, proximal elongation | PUSH_STEP | — |
| S3 | Levy 2015 SI movies (the paper's own, with arm labels) | Current Biology supplement, DOI 10.1016/j.cub.2015.02.064 — Erik has the SI PDF (`docs/papers/octopus/`); movies via Lund | which arms push, the ±45° crawl mode | PUSH_STEP, hub lottery | — |
| S4 | Katija et al. 2025, Nature — Supplementary Videos 1–2 | nature.com/articles/s41586-025-09379-z (open access; page needs a cookie hop — open in a browser) | deep-sea *Muusoctopus* crawling over rock with 3D arm tracking: 3 ± 1 arms in contact, pair 1 probing, ventral→dorsal contact order, release at proximal-strain saturation | PUSH_STEP, EXPLORE, release trigger | — |
| S5 | Huffard 2006, J Exp Biol 209:3697 — Movies 1–3 | jeb.biologists.org supplement (DOI 10.1242/jeb.02435) | medusoid jetting, bipedal walking of *Abdopus* | UMBRELLA, JET, walk | — |
| S6 | Huffard, Boneka & Full 2005, Science 307:1927 — SOM movies | science.org supplement (DOI 10.1126/science.1109616) | coconut octopus and *Abdopus* bipedal walking | walk (T2) | — |
| S7 | Sumbre et al. 2005, Nature 433:595 — supplementary video | nature.com supplement (DOI 10.1038/433595a) | fetching with pseudo-joints: base, elbow at s/2, wrist | FETCH | — |
| S8 | Kennedy et al. 2020, Sci Rep — Supplementary Videos 1–2 | nature.com/srep (DOI 10.1038/s41598-020-77873-7) | bends, torsion, elongation; reach, fetch, pull-in | REACH, FETCH, stretch | — |
| S9 | Bidel et al. 2022, Curr Biol — videos + Dryad dataset | DOI 10.1016/j.cub.2022.06.003; Dryad | 628 prey strikes: reach, pounce, parachute, snaptrap; the 2nd arm leads | REACH (strike), T2 pounce | — |
| S10 | Gendreau et al. 2025, Biol Open | DOI per the motor-biology digest §5 | arm recruitment into a hidden-item dome: nearest arm + neighbours | hub recruitment | — |
| S11 | Sumbre et al. 2001, Science 293:1845 — SOM movie | science.org supplement (DOI 10.1126/science.1060976) | the evoked reach in a denervated arm: the bend travelling, straight arm behind | REACH (the P1 target) | — |

### 1.2 Documentaries (long naturalistic sequences)

| # | source | what to look for | primitive | watched / timestamps |
|---|---|---|---|---|
| D1 | *My Octopus Teacher* (Netflix 2020, Craig Foster; wild *O. vulgaris*) | crawling over kelp, jetting from the shark, hunting (parachute over prey), hiding under shells, the arm exploring the diver's hand, idle on the den | all T1 primitives; idle "alive"; the decision hold before a strike | — |
| D2 | *Secrets of the Octopus* (NatGeo / Disney+ 2024) | hunting alongside fish, coconut octopus walking, crawling | PUSH_STEP, walk, EXPLORE | — |
| D3 | PBS Nature, *Octopus: Making Contact* (2019) | tank behaviour: reaching to a person, fetch to mouth, jet | REACH, FETCH, JET | — |
| D4 | BBC *Blue Planet II*, "Green Seas" (2017) | the shell-armour sequence vs the pyjama shark: many arms working, then a jet | WRAP/HOLD, JET, several arms | — |

### 1.3 Channels (search terms in quotes)

| # | channel | search | primitive |
|---|---|---|---|
| C1 | MBARI (YouTube) | "Octopus Garden", "EyeRIS", "Muusoctopus" | deep-sea crawl, brooding idle |
| C2 | Nautilus Live / Ocean Exploration Trust | "octopus" ROV dives | deep-sea crawl and swim, startle |
| C3 | NOAA Ocean Exploration | "octopus" | as C2 |
| C4 | Monterey Bay Aquarium | "giant Pacific octopus" | large-animal crawl on glass, sucker patches visible from below (the top-down cue!) |
| C5 | Marine Biological Laboratory (Hanlon lab) | "mimic octopus", "Abdopus walking" | walk, mimicry postures |

A glass-wall or glass-floor clip (C4) is the most valuable for the game:
the sucker patches seen through the glass are exactly the contact pattern a
top-down camera must sell (checklist row 9, top-down rule).

---

## 2. Scoring sheet — one copy per prototype clip

Clip: ______  Reference clip beside it: ______  Date: ______  Judge: ______

### 2.1 Craft checklist (games/film digest, synthesis §2) — a clip passes only if every row holds

| # | item | measure where one exists | pass / note |
|---|---|---|---|
| 1 | Few actors: ≤ 2–3 arms do deliberate work at once | per frame, arms whose joint speed > idle threshold | |
| 2 | The eyes lead: mantle/eyes turn before any arm commits | gaze onset precedes arm onset | |
| 3 | Willed motion starts at the base, the tip arrives last | time of peak curvature increases with arc length | |
| 4 | Passive motion is the reverse: the tip lags in a whip | — | |
| 5 | Anticipation: the arm gathers before it releases; the body compresses before a jet | — | |
| 6 | Follow-through and settle; nothing stops dead | — | |
| 7 | Holds and rests: a visible "decide" beat; idle arms are not in perpetual sine motion | — | |
| 8 | De-synchrony and roles: no two arms share phase, rate or amplitude | — | |
| 9 | Contact is causal: planted segments do not slide; speed rises with gripping arms; the body sags toward a loaded arm | drift of planted points ≈ 0 | |
| 10 | Attention habituates; every AI state has a visible body tell | — | |
| 11 | One mood, everywhere: agitation changes all arms coherently | — | |
| 12 | The body answers the arms: arm loads move the mantle; arms root in the crown with continuous curvature | — | |
| 13 | Stable over time: the same command gives the same shape whatever the history | — | |
| 14 | Decisive silhouettes from the game camera | — | |

### 2.2 The ten things a game gets wrong (motor-biology digest §4) — none may appear

| # | wrong | right | seen? |
|---|---|---|---|
| 1 | continuous sine "tentacle wiggle" | arms mostly still; discrete stereotyped events | |
| 2 | IK-style uniform curl toward the target | ONE travelling bend, straight stiff arm behind, limp ahead, suckers forward, short twist wind-up | |
| 3 | fixed arm length | proximal stretch ×1.5–1.8 and thinning; sucker spacing opens | |
| 4 | spider gait tables, tips pulling | 1–3 arms down, planted MID-arm, pushing by proximal elongation; front arms probe | |
| 5 | head bobbing with each step | head held level | |
| 6 | head-first paddling swim | JET mantle-first, arms bundled; medusoid = slow open, fast close | |
| 7 | grabbing by coiling the tip | flat sucker patches mid-arm; fetch = three stiff segments with an elbow | |
| 8 | all suckers stuck, arms sticking to each other | ~1 in 5 sealed; arms slide over each other | |
| 9 | all eight arms equally busy | front pairs explore (2nd arm leads), back pairs support; nearest arm + neighbours recruited | |
| 10 | constant-speed motions | accelerate-late, bend-dissolves-at-tip profile | |

### 2.3 Numbers to read off the clip (P2–P4 gates)

| quantity | real (brief §2.1) | clip |
|---|---|---|
| reach: bend speed / wave travel time | 7–60 cm/s; 0.6–1.2 s on a 24–34 cm arm | |
| crawl: arms pushing at once / step / active push | 1–3 lab, 3 ± 1 wild / 3.6 s / 0.7 s | |
| crawl: body heading vs travel direction | independent (r ≈ 0.12) | |
| umbrella: open : close | ≈ 2.5 : 1 | |
| jet: speed, distance | 45–100 cm/s, a few metres | |
| fetch: elbow sweep, duration | ~100°, ~1.6 s | |
| suckers sealed at a time | ~1 in 5 | |

---

## 3. Filling it in

- One row per clip, timestamps `mm:ss–mm:ss`, species if known, and which
  primitive the segment is the reference for.
- A documentary clip's behaviour label is a guess until matched to a paper
  description; mark it `?`.
- When a prototype clip is judged, the filled scoring sheet goes under
  `docs/research/octopus_clip_scores/<clip>_<date>.md` (create on first use)
  and the verdict line on the issue.
