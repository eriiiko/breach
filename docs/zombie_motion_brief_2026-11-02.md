# Zombie motion brief — v0.1 (for Erik's line-by-line veto)

**Status: DRAFT, 2026-11-02.** Milestone 14 (Zombie motion), design doc
`animation_milestones_design_2026-11-02.md` §5. Written from Erik's recollection of the
Warhammer 40K **Genestealer** and Bloodborne's **Scourge Beast** plus my own memory of
both; **neither has been checked against footage**, so every line is a claim to correct.
Erik's standing description: *"starting to hunch forward, arms raised forward like they
want to grab, the back leaning forward a little like a beast"*.

**How to veto:** answer with the line number and "no" / "more" / "less" / a rewrite.
Nothing is produced (no clip, no Kimodo prompt) until this brief is approved. Numbers in
§3 are first guesses for live tuning, not rulings.

---

## 1. The motion in words

1. **The read:** a predator wearing a human body. Not the dead and slow: hungry, alert,
   wrong. Even idle, it is "about to".
2. **Not this:** not the Romero shamble (stiff legs, straight arms out, a groan); not
   parkour-fast; not an insect. Human proportions stay; the *carriage* is animal.
3. **Silhouette from above (the camera we have):** mass shifted forward of the feet;
   shoulders rolled forward so the shoulder line leads; arms out in front; head thrust
   ahead of the shoulders. Footwork is hidden, so this is where the beast reads.
4. **Spine:** pitched forward from the first stage and more with intent — a stalk is a
   clear hunch, a charge is near-horizontal, a four-point sprint is horizontal.
5. **Hips and knees:** lowered, knees bent, weight low and loaded, like a cat about to
   spring — never upright and stiff-kneed.
6. **Shoulders:** protracted (rolled forward and slightly up); one shoulder leads at a
   time, so the torso twists as it advances.
7. **Arms:** carried forward and slightly up, elbows part-bent, forearms hanging; hands
   **open with fingers splayed and curled like claws**. They swing little when stalking
   and reach or clutch toward the target. In a charge they swing long and low; in the
   four-point sprint the knuckles or hands touch down and carry weight.
8. **Head:** dropped and thrust forward, chin low, eyes on the target; sharp small snaps
   toward a sound; a head-cock when idle.
9. **Tempo:** irregular — stop, hesitation, half-step, then a burst. Never a metronome.
   (Parameter `tempo_jitter`.)
10. **Idle:** weight shifting side to side, a twitch in the shoulders and fingers, the
    head scanning in jerks, an occasional full-body shudder. Never still, never calm.
11. **Telegraph before a lunge:** a coil — spine rounds, arms draw in, one held beat —
    then the explosion. A short roar / head-back-arms-wide shake is the alert telegraph.
12. **Claw swipe:** a long committed arm arc that carries the torso with it; the recovery
    is slower than the strike (a window for the player).
13. **Lunge / pounce:** launch from the legs, arms leading, land low and already coiling.
14. **Bite and grab (the sim's bite):** arms clamp, head drives forward into the target.
15. **Hit stagger:** head and torso whip; one arm flails; it keeps coming (low reaction
    to small hits — it does not fall unless the push is big).
16. **Death:** a crumple or collapse, limbs twitching a beat after the rest has stopped.
    A few variants (forward, sideways, backward).
17. **Fall / prone / rise (also the conversion):** it loses balance without protecting
    itself; on the ground it twitches and its limbs keep moving; it rises by getting the
    arms under and **snapping up in a lurch**, not a human get-up — fast and ugly. Used by
    knockdown for any zombie, and later by conversion.
18. **Door bash:** head down, shoulder or both forearms driving like a ram, a rhythm that
    speeds up; claws scraping the door between hits.
19. **Variation across a horde:** each zombie gets small offsets in tempo, lean and arm
    carry so a horde does not march in unison; a mix of stalkers and chargers.

## 2. Human → zombie (the continuous blend)

20. One amount **a ∈ [0, 1]** scales the posture parameters in §3 continuously. The
    human is all zeros; the zombie is the §3 row.
21. **Discrete clip levels** with crossfades: **0 healthy** → **1 infected** (slight
    hunch, head drops, a cough or stagger, a limp, a tremor) → **2 turning** (heavier
    hunch, arms drawing forward, jerks and spasms, hands starting to curl) → **3 zombie**
    (the full brief). Posture slides continuously between the levels.
22. **Gait is not blended** human→beast (the cycles are not phase-aligned); the clip set
    changes by level and the posture does the smooth part.
23. Skin: a wound / vein overlay driven by the same `a` (a shader parameter).
24. Where `a` comes from is a **sim question** (research session). Until then: `a` is 0
    or 1 from `is_zombie`.

## 3. Posture parameters (first guesses; tune live with Erik judging)

| Parameter | Human | Stalk | Charge | Four-point |
|---|---|---|---|---|
| `spine_pitch` (deg forward) | 0 | 25–35 | 45–60 | 75–90 |
| `knee_bend` (hip drop, fraction of leg) | 0 | 0.15 | 0.25 | 0.35 |
| `shoulder_protraction` | 0 | 0.6 | 0.8 | 1.0 |
| `arm_forward` (deg, from hanging) | 0 | 50–60 | 30 (swing) | weight-bearing |
| `arm_open` (finger spread / curl) | 0 | 0.7 | 0.8 | 0.5 |
| `head_drop` / thrust | 0 | 0.5 | 0.7 | 0.6 |
| `tempo_jitter` | 0 | 0.4 | 0.2 | 0.15 |
| `twitch` | 0 | 0.3 | 0.1 | 0.1 |

Four-point is a pose on the humanoid skeleton with long-arm proportions on the skin
(experiment E5 decides whether the one skeleton can carry it).

## 4. After approval

1. Posture-only test on the existing walk / idle / run as additive offsets — **before any
   new clip** (experiment E4): does the top-down view already read as a beast?
2. The Kimodo spike (`kimodo_spike_workflow_2026-11-02.md`) turns lines 4–11, 17 into
   prompts and keyframe constraints.
3. Clips are named by the brief's line (`z_stalk`, `z_charge`, `z_pounce`, …) so a review
   can point at "line 12".
