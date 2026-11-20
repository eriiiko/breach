# Impulse on a person: push recovery, shoves, bullets, slides, blasts, and a minimal model (2026-11-10)

Research only; nothing in the repo was changed. This builds on the wind work, which is not redone here:
`docs/research/wind_on_a_person_2026-10-31.md` (the drag area, the 8 / 14 / 20 m/s bands) and
`docs/wind_on_units_design_2026-10-31.md` (the ruling: below a threshold the air moves nobody, a band only
slows walking, above a higher threshold the unit is knocked down, and nothing displaces a unit).

Conventions. g = 9.81 m/s². The reference unit is a marine of 80 kg body + 20 kg kit = 100 kg, about 1.80 m
tall, centre of mass (CoM) about 1.0 m up. The leg-length constant of the inverted pendulum is
ω0 = sqrt(g/l) ≈ 3.13 s⁻¹ at l = 1.0 m, so 1/ω0 ≈ 0.32 s. Impulse J is in N·s (= kg·m/s), and Δv = J/m.

The labels mean:
- **[P]** is read in a primary source, either the abstract or the full text, as retrieved here.
- **[S]** is a secondary source (wiki, review, reprint or forum) or a figure not checked against the primary.
- **[U]** is unverified (from memory, or the source could not be opened).
- **[M]** is my own calculation or modelling proposal, not a literature fact.

---

## 0. Summary table

| Quantity | Literature range | Pick for 100 kg marine | Source |
|---|---|---|---|
| CoM velocity change absorbed **in place** (ankle/hip, no step), forward | 0.2–0.5 m/s. Full-BoS XcoM gives about 0.45 m/s. The effective BoS after a sudden push is only about 30 % of the static BoS area | **0.25 m/s** (J ≈ 25 N·s) | [2][3][4] [M] |
| same, backward | smaller: CoM sits nearer the heel than the toe | **0.15 m/s** (15 N·s) | [2][4] [M] |
| same, sideways | between the two (stance half-width ≈ 0.1–0.15 m) | **0.20 m/s** (20 N·s) | [2][3] [M] |
| Stepping triggered, waist pull (standing) | young: the position threshold is 72–107 % of BoS length, and the velocity transition is at 121 % BoS/s | consistent with the picks above | [5] [P] |
| Largest perturbation recovered with **one step**, forward (lean release, prepared) | young 30.7–32.5° lean, older men 23.9°, older women 16.2° | XcoM-equivalent ≈ **1.7 m/s** young; pick **v_fall,fwd = 1.5 m/s** | [6][7] [P] + [M] |
| same, backward / sideways | no direct impulse data found | **0.8 m/s / 1.1 m/s** | [M] |
| Walking: lateral pushes recovered in ≤ 2 steps | 2.7–12.4 N·s over 100 ms (≈ 0.04–0.17 m/s) | (inside the in-place / one-step band) | [8] [P] |
| Walking: AP and ML pelvis pulses, no falls reported | 4–16 % BW for 150 ms (≈ 4–16 N·s at 67 kg) | — | [9] [P] |
| Expected vs unexpected push | expected: peak CoM excursion ≈ 1.6× smaller (28 → 17 mm). Two crowd/stepping studies found no significant effect of awareness | optional "braced" ×1.3–1.5 on thresholds | [10] [P], [11][12] [P] |
| Impulse that makes a whole 5-person row step forward | ≥ 75 N·s (elbow spacing), ≥ 90 N·s (arm spacing) | strong shove ≈ **100 N·s** | [13] [P] |
| Two-handed **isometric** push, standing male | mean ≈ 240–255 N (5th pct ≈ 150 N) | — | [14] [S] |
| Push that caused a fall (forensic video case) | ≈ 315 N over 11/30 s (≈ 115 N·s) | — | [15] [S] |
| Rugby shoulder tackle peak force (into a 45 kg bag) | 1660 N lab, 1997 N field | tackle / charge ≈ **300 N·s** | [16] [P] + [M] |
| Front kick impact force (solid target) | 1570 N (novice) – 3696 N (elite); foot speed ≈ 8.6 m/s into target | push kick ≈ **60 N·s** | [17] [P] + [M] |
| Riot-shield push | no measurement found | treat as a shove at 100–150 N·s | — |
| Bullet momentum (muzzle) | 9 mm 2.9, 5.56 3.8, 7.62×39 5.7, 12 ga slug ≈ 13.5, .50 BMG ≈ 38 N·s | Δv on 100 kg: 0.03–0.38 m/s if **fully stopped** | [18][19] [S] + [M] |
| "Knock-down power" | "a myth"; the impact is "no more than the recoil of the weapon" | physics gives no single-hit knockdown | [20] [P] |
| Time a torso-shot person can act | ≥ 10–15 s of brain oxygen even with the heart destroyed; falls are "most likely psychological" | — | [20] [P] |
| Required friction for level walking | peak RCOF ≈ 0.18–0.20 | 0.2 | [21][22] [S] |
| Navy non-skid deck (MIL-PRF-24667C minimum static COF, new) | dry 0.90–0.95, wet 0.85–0.90, oily 0.75–0.80 (spec test, not a boot) | dry deck μ ≈ 0.8 | [23] [S] |
| Safety boot on steel + glycerol (EN ISO 20345 SRB minimum) | heel ≥ 0.13, flat ≥ 0.18 (≈ 0.22 "safe") | oily smooth steel μ ≈ **0.15** | [24][25] [S] |
| Body sliding on the ground after a fall | 0.58–0.7 on roads (reconstruction literature) | clothing on steel deck μ_slide ≈ **0.4** | [26] [S] + [M] |
| Self-selected walking speed vs resisting force | speed falls linearly; slope −0.37 (normalized to each person's max force); about 55 % of speed left at the highest force | keep the wind design's linear-to-0.5 law, expressed in F/W | [27] [P] |
| Walking at normal speed against a horizontal force (forced pace) | sustained to 15 % BW (treadmill), and against an 18.5 m/s wind (≈ 0.2 BW) | — | [28][29] [P] |
| Tertiary blast: total-body impact criteria | mostly safe 3.05 m/s (10 ft/s), threshold 6.4, 50 % lethal 16.5, ≈ 100 % 42 m/s | (later blast patch) | [30][31] [S] |
| Elden Ring enemy poise | meter 15–65 (bosses 80–120); regen starts after poise/13 s, then 13/s | linear-decay accumulator form | [32] [S] |
| Helldivers 2 | per-hit thresholds: ≥ 10 stagger, ≥ 25 ragdoll (on players); each pellet carries full stagger | per-hit threshold form | [33] [S] |

---

## 1. Balance perturbation and push recovery

### 1.1 The governing quantity: the extrapolated centre of mass (facts)

Hof, Gazendam & Sinke [2] [P] extend the static rule ("CoM over the base of support") to dynamics:
the **extrapolated centre of mass** is XcoM = x + v/ω0, with ω0 = sqrt(g/l). Standing balance without a
step needs XcoM inside the base of support (BoS). The **margin of stability** b is the distance from XcoM to
the BoS edge. The humanoid-robotics **capture point** (Pratt, Carff, Drakunov & Goswami [1] [P]) is the
same quantity: the ground point where the robot must step to come to a full stop.

The consequence for this design [M]: an impulse is judged by the **CoM velocity change it causes**,
Δv = J/m. It crosses the "must step" line when Δv/ω0 exceeds the distance from the CoM to the BoS edge in
the push direction. A heavier unit needs proportionally more impulse, and mass enters exactly as the
designer wants. A taller unit has a larger 1/ω0, which lowers the threshold, but also a longer foot and stance.
With all lengths scaling with height h, v_threshold ∝ sqrt(h), and J_threshold ∝ m·sqrt(h).

**Feasible stability region.** Pai & Patton [3] [P] compute the set of (CoM position, CoM velocity)
states from which a person can stop **without stepping**. Velocities are normalised to body height (s⁻¹).
- The upper (forward) boundary runs from 1.1 s⁻¹ at 2.4 foot lengths behind the heel, to **0.45 s⁻¹ over the heel**, to 0 over the toe.
- The backward boundary runs from 0.9 s⁻¹ at 2.7 foot lengths behind the heel to 0 over the heel.
- BoS size, not strength, sets the region under normal conditions. But **friction below 0.82**, a dorsiflexion loss > 51 %, or a plantar-flexion loss > 35 % shrink it.

For h = 1.8 m, 0.45 s⁻¹ is 0.81 m/s with the CoM over the heel. Quiet stance has the CoM ahead of the heel,
which leaves less [M]. **The friction result matters for an oily deck:** footing friction caps balance,
not only sliding.

**The effective BoS is much smaller than the foot.** Hof & Curtze [4] [P] add the finite reaction and
displacement time of the centre of pressure (CoP). After sudden perturbations of people standing on two
feet, "the area of the effective BoS is only a fraction, some 30 %, of the area of the static BoS". That is
about 55 % in linear size if the shape is kept [M].

**Ankle vs step timing in walking.** Hof, Vermerris & Gjaltema [8] [P] pushed walking subjects
(71.6 ± 9.7 kg) sideways for 100 ms, with impulses 2.7–12.4 N·s (the paper: 8 N·s on 80 kg = 0.1 m/s).
- The **ankle strategy** acts after about 200 ms but moves the CoP "no more than 2 cm".
- The **stepping strategy** needs ≥ 300 ms but has a range of about 20 cm.
- Balance was always recovered within two steps (one stride), with foot placement a constant distance outside the XcoM.

### 1.2 Ankle → hip → step → fall: thresholds in m/s and N·s

**Fixed-support (ankle/hip) limit** [M from [2][4][8]]. Assume foot length 0.26 m, ankle 0.05 m in front of
the heel and CoM 0.03 m in front of the ankle. Then the CoM is about 0.15 m from the toe and 0.08 m from the
heel. With the full static BoS, Δv_max = ω0·distance:
- forward 0.47 m/s,
- backward 0.25 m/s,
- sideways, with a stance half-width of about 0.12 m: 0.38 m/s.

With the ~55 % linear effective BoS of [4]:
- **forward ≈ 0.26 m/s,**
- **backward ≈ 0.14 m/s,**
- **sideways ≈ 0.21 m/s.**

For 100 kg that is 26 / 14 / 21 N·s. The hip strategy adds some range but, per Maki & McIlroy [11]
[P], people prefer to step instead. Change-in-support reactions "are often initiated well before the
centre of mass is near the stability limits", and subjects choose them in preference to the fixed-support
hip strategy. **For a game, "ankle/hip" and "a step" are therefore one boundary.** Below it nothing visible
happens; above it the unit steps (staggers).

**Measured stepping thresholds (waist pull).** Mille et al. [5] [P] pulled standing subjects forward
(the CoM goes forward, i.e. "pushed from behind" in this design's terms). There is a position limit T_L
beyond which subjects always step, however slowly pulled: 107 % of the BoS length in the young, 91 % in the
old. The position threshold falls with pull velocity to T_H = 72 % (young) / 59 % (old), and the transition
happens around 121 % BoS/s (young) / 96 % BoS/s (old). With a BoS of about 0.26 m, 121 % BoS/s ≈ 0.31 m/s
[M]. That matches the forward fixed-support estimate above. Older subjects stepped in 69 % of trials, the
young in 40 %. The authors read stepping as **triggered by the perturbation event**, not a late reaction.
I could not open the full text for absolute pull magnitudes [U].

**One-step recovery limit (forward).** Release from a forward lean, instructed to recover with one step:
- young men 32.5°, old men 23.9° (Thelen et al. [6] [P]);
- young women 30.7°, old women 16.2° (Wojcik et al. [7] [P]).

A lean of θ puts the CoM l·sinθ ahead of the ankle with zero velocity. In XcoM terms that equals a
standing person given Δv = ω0·l·sinθ [M]: **32.5° ≈ 1.7 m/s, 23.9° ≈ 1.3 m/s, 16.2° ≈ 0.87 m/s.** The
lean-release subjects were prepared and instructed, so this is an upper bound for an unprepared, loaded
marine. Hsiao & Robinovitch [34] [P] show the feasibility couples step length, step execution time and leg
strength: larger perturbations bring larger, quicker steps. Their observed steps were shorter and slower
than optimal, by choice, to save effort.

**One-step capture check by the linear inverted pendulum** [M]. The CoP sits at the forward toe while
the step is in flight, so the XcoM grows as ξ(t) = p + (ξ0 − p)·e^{ω0 t}. With the step landing at
t_s ≈ 0.3–0.35 s (≥ 300 ms per [8]), e^{ω0 t_s} ≈ 2.6–3.0. A step of about 0.8–0.9 m captures
ξ0 ≈ 0.4–0.54 m, i.e. **Δv ≈ 1.25–1.7 m/s**, consistent with the lean data. Backward steps are shorter
and slower (I assumed L ≈ 0.5 m, t_s ≈ 0.4 s, CoP at the heel 0.08 m behind the CoM), which gives
**≈ 0.7–0.8 m/s**. A sideways step (L ≈ 0.55 m, half-width 0.12 m) gives **≈ 1.0 m/s**. These backward
and sideways step parameters are my assumptions [U]; I found no human impulse threshold for them.

**Direction.**
- Backward is weakest. The CoM sits closer to the heel than the toe ([3]'s boundary is asymmetric), and a backward step is shorter.
- A humanoid-walking study also found forward pushes resisted more than backward ones [S] (the search summary of the arXiv time-projection paper, not re-read).
- Sideways recovery needs a loaded side step or a crossover step. Chatagnon et al. [12] [P] characterise these for 21 young adults over 5 perturbation angles. Maki & McIlroy [11] note that lateral destabilisation complicates stepping, especially in older adults.

**Expectation.** Santos, Kanekar & Aruin [10] [P] (shoulder-level pendulum impacts): a predictable push gave
a peak backward CoM excursion of 17 ± 5.5 mm against 28 ± 9.6 mm unpredictable, which is 1.6× smaller. The CoP
excursion was 28 against 60 mm. In contrast, for **stepping** outcomes, Chatagnon et al. [12] [P] found
"no significant effect of the level of awareness", and Feldmann et al. [35] [P] found preparedness did not
significantly change how a push propagated through a crowd. Reading [M]: anticipation helps the
fixed-support response (bracing), but not much the decision to step. A "braced" multiplier of 1.3–1.5 on
the **in-place** threshold is defensible, and none on the fall threshold.

**Walking vs standing.** A walker steps every ~0.5 s anyway, so foot placement absorbs perturbations as
part of gait.
- Lateral 100 ms pushes up to 12.4 N·s were handled within a stride [8].
- AP and ML pelvis pulses of 4–16 % BW over 150 ms, i.e. ≈ 16 N·s on 67 kg, produced no reported falls (Vlutters et al. [9] [P]; AP foot placement was not adjusted at the first heel strike; the CoP at the following toe-off took the correction).

I found no walking fall threshold. [M]: for a game, use the standing thresholds for both. A forward push
on a walker is slightly easier (already stepping), and a sideways push during single stance slightly
harder. The difference is below the precision of everything else here.

**Mass and kit.** With the same thresholds in m/s, J scales with m: a 100 kg marine needs 25 % more
impulse than an 80 kg zombie for the same effect. A pack raises the CoM a little (larger 1/ω0, slightly
lower v thresholds) and loads the legs, but I found no quantitative source [U]. Keep thresholds in m/s
and let mass carry the effect.

---

## 2. How hard can a person shove another

- **Crowd push experiments (the best data found).** Feldmann & Adrian [13] [P] (Safety Science 164:106173) pushed the rearmost of a row of five people with a hand-swung punching bag and measured the impulse with a pressure sensor. At elbow distance all participants moved forward once the impulse exceeded **75 N·s**, and at arm distance the push travelled through the whole row from **90 N·s**. A follow-up with up to 36 people [35] [P] found that the propagation distance and speed are mainly functions of push intensity. The propagation speed is linear in J (for example 0.006·J + 1.37 m/s for a row of 5 at elbow distance), and preparedness had no significant effect. That speed is the speed of the *wave* through the row, not of a body. I could not retrieve the full impulse range used [U]; the bag pushes were "weak / medium / strong".
- **Isometric push strength** (feet planted, not a shove): standing two-handed horizontal push means of about 240–255 N in men, 5th percentile about 150 N [14] [S]. These are limited by posture and floor friction, and a dynamic shove adds the pusher's momentum.
- **A real push that caused a fall.** A 2025 video reconstruction estimated about 315 N over 11/30 s, i.e. about 115 N·s [M from the figures], from behind [15] [S]. I could only read the search summary; the publisher refused access.
- **Shoulder tackle.** Usman, McIntosh & Fréchède [16] [P]: the maximum shoulder force in an active rugby tackle into a 45 kg instrumented bag averaged **1660 N in the lab and 1997 N in the field**, with or without pads. No duration was given. [M] A tackler of 90–100 kg at 3–5 m/s carries 270–500 N·s. A perfectly inelastic collision with a 90–100 kg target gives the target Δv ≈ 1.5–2.5 m/s, past the one-step limit, so a fall, which matches experience. Pick **300 N·s**.
- **American-football impacts** are recorded as head kinematics: concussed players had head impacts at 9.3 ± 1.9 m/s and head Δv of 7.2 ± 1.8 m/s (Pellman et al. [36] [P]). These are head values, not whole-body values. They only bound the closing speeds involved.
- **Kick.** A systematic review (Vagner et al. [17] [P]) gives front-kick impact forces on a solid target of 1570 ± 362 N (novice), 2846 ± 805 N (sub-elite) and 3696 ± 1621 N (elite), with a foot speed into the target of about 8.6 m/s. There was no impulse or contact duration. [M] Taking an effective leg mass of 5–10 kg at 8.6 m/s gives about 40–85 N·s. A push kick (teep) extends contact with a drive through the hip, so pick **60 N·s**.
- **Riot shield push:** no measurement found. Treat it as a braced two-handed shove with the pusher's body behind it (100–150 N·s) [M].

Resulting motion [M] for the 100 kg marine (80 kg zombie in brackets):

| Shove | Δv |
|---|---|
| 60 N·s kick | 0.6 (0.75) m/s, one step |
| 100 N·s shove | 1.0 (1.25) m/s, one step (zombie near falling) |
| 300 N·s charge | 3.0 (3.75) m/s, knocked down |

Displacement before recovery is given by the recovery law in §4.3. A 1 m/s stagger moves the CoM roughly
0.4–0.5 m over one to two steps.

---

## 3. Bullets

### 3.1 Momentum and Δv (numbers)

Nominal muzzle values; secondary sources. The M855 spec is 61.7 gr at 3000 ± 40 fps measured 78 ft from a
20" barrel [18] [S]. M33 .50 ball at 661 gr / 2910 fps is the commonly quoted value [19] [S/U]. The others
are typical published loads [U].

| Round | Mass | Speed | p = mv | Δv, 100 kg, fully stopped | Δv, 80 kg |
|---|---|---|---|---|---|
| 9×19 mm, 124 gr | 8.0 g | 360 m/s | 2.9 N·s | 0.029 m/s | 0.036 m/s |
| 5.56×45 M855 | 4.0 g | 915–930 m/s | 3.7–3.8 | 0.038 | 0.047 |
| 7.62×39 M43 | 7.9 g | 715 m/s | 5.6 | 0.056 | 0.071 |
| 12 ga 1 oz slug | 28.4 g | ≈ 475 m/s | 13.5 | 0.135 | 0.17 |
| 12 ga 00 buck, 9 pellets | ≈ 31.5 g | ≈ 400 m/s | 12.6 total | 0.126 | 0.16 |
| .50 BMG M33 | 42.8 g | 887 m/s | 38 | 0.38 | 0.47 |

These are upper bounds. A round that exits keeps part of its momentum, and only Δp is transferred. The
shooter receives at least the same momentum as recoil, plus the propellant gas (for rifles typically a
substantial extra fraction [U]), and is not knocked over.

### 3.2 What the literature says about knockback

Patrick, *Handgun Wounding Factors and Effectiveness*, FBI Academy, 14 July 1989 [20] [P, full text as
reprinted]. Short quotes:
- "A bullet simply cannot knock a man down."
- if it could, "equal energy would be applied against the shooter and he too would be knocked down"
- "The impact of the bullet upon the body is no more than the recoil of the weapon."
- the energy deposited "is approximately equivalent to being hit with a baseball"
- "The much discussed 'shock' of bullet impact is a fable and 'knock down' power is a myth."

MythBusters' season-3 myth "a person will be propelled violently backwards if hit by a bullet" was listed
as busted for lack of momentum [37] [S].

On §1's scale [M], every handgun and rifle round except the .50 gives Δv < 0.08 m/s on an 80–100 kg body.
That is 3–6× below the in-place (ankle) limit even in the weak backward direction. A slug or a full buckshot
charge is at the backward in-place limit for a light target. Only a .50 BMG that stops in the body reaches
a recovery-step level (0.38–0.47 m/s), and it rarely stops.

**Sustained fire, physics only** [M]. A 5.56 rifle at 750 rpm is 12.5 rounds/s × 3.8 N·s ≈ **47 N mean
force** per shooter if every round stops: 0.05 BW on 100 kg, 0.06 BW on 80 kg. That is the force of an
8–9 m/s wind on the same body, i.e. the *onset* of the wind design's slowed band. Three shooters on one
target give about 140 N ≈ 0.14–0.18 BW, mid-band, so a clearly slower walk. Five or more give about 0.24–0.30
BW, the braced-topple level. So **"focus fire visibly slows a target" is physically real, but only as a
sustained force, only from many automatic weapons, and only if the rounds stop in the body.** A single hit
staggering a target is not momentum physics.

### 3.3 What actually makes a shot person stagger, slow or fall

- **Psychological.** Patrick [20] [P]: "Psychological factors are probably the most important relative to achieving rapid incapacitation"; "People largely fall down when shot"; "the causative factors are most likely psychological in origin", needing only "a shot and cognition of being shot".
- **Physiological, slow.** "There is sufficient oxygen in the brain to support full and complete voluntary action for 10–15 seconds" after the heart is destroyed, so incapacitation "becomes an eventual event, not necessarily an immediate one" [20] [P]. Reliable immediate incapacitation from torso hits "is a myth"; only a shot that disrupts the brain (or upper spinal cord) stops a determined adversary at once [20] [P].
- **Structural** [U, general medical reasoning, not sourced here]: a fractured femur, pelvis or knee removes the leg's support at once, and the person goes down because the structure fails, not because they were pushed. A hit to the spine below the brain causes paralysis distal to it.
- **Neuromuscular / pain / startle** [U]: a flinch or startle and guarding from pain plausibly disturb gait for a fraction of a second. I found no quantitative timing data in the time available.

**For zombies** [M]: the psychological and pain channels are absent by premise, so the realistic slowing
mechanisms are (a) structural damage to the legs and pelvis (location-based), and (b) the sustained-force
physics of §3.2 under massed fire.

### 3.4 What a game rule could tie pushback to [M]

To keep the rule scaling sensibly with projectile momentum and target mass, feed every hit into the same
impulse law as a shove, with a declared **stagger gain**:

J_game = G_stagger · Δp_transferred, with Δp_transferred = p·f_stop (f_stop = 1 for a round that stops,
less for a through-and-through).

- G_stagger = 1 is physics: nothing visible from small arms, and slowing under massed fire through §4's effort channel.
- G_stagger ≈ 3–5 is an honest "game exaggeration" knob. It keeps the ordering (a slug pushes harder than a 9 mm, a zombie is pushed further than a kitted marine, and a heavy creature barely moves) because it still multiplies momentum and divides by mass.

The note to the designer should say plainly that G > 1 represents flinch, pain and structural effects
lumped into a push. For zombies, a leg-damage-driven slow is the more truthful alternative.

---

## 4. Friction, sliding and recovery distance

### 4.1 Footing friction (facts)

- **Required friction in gait.** The peak required (utilised) coefficient of friction (RCOF) for level walking by healthy adults is about **0.18–0.20** [21][22] [S]. It is higher at heel strike, fast walking, turning and slopes. Redfern et al.'s review [21] [P, abstract] is the standard reference for RCOF versus available COF. Hanson, Redfern & Mazumdar [38] [P] model the slip/fall probability as a logistic function of (required − available).
- **Available friction, navy non-skid deck.** The minimum static COF in MIL-PRF-24667C for new coatings is 0.95 / 0.90 / 0.80 (dry / wet / oily) for Types I, V–IX, and 0.90 / 0.85 / 0.75 for Types II–IV, X, XI. After wear it is 0.90 / 0.85 / 0.75 and 0.85 / 0.75 / 0.65 [23] [S, emilspec reprint]. These are measured with the spec's own test, not a boot sole.
- **Available friction, smooth steel with oil.** The EN ISO 20345 / ISO 13287 "SRB" safety-boot class needs only COF ≥ 0.13 (heel) and ≥ 0.18 (flat) on steel with glycerol [24] [S]. A study on smooth stainless steel with glycerol put the "safe" kinetic friction at about **0.22** [25] [S, via the abstract summary]. A smooth, oily steel deck is therefore at or below the walking RCOF: people slip while walking normally.
- **Pai & Patton** [3] [P]: friction below 0.82 begins to shrink the no-step balance region. Footing friction caps the force a person can resist even before they slide.

### 4.2 A body sliding after a fall

- Pedestrian-throw reconstruction uses body–road coefficients of about 0.58 (Wood & Simms) to 0.7 (Searle's universal parameter) [26] [S].
- I found no value for clothing or kit on steel. **Pick μ_slide = 0.4 on a dry deck and 0.15 on an oily one** [M].
- Slide distance d = v²/(2 μ g): at 2 m/s, 0.51 m (μ 0.4) or 1.36 m (μ 0.15). At 5 m/s, 3.2 m or 8.5 m [M].

### 4.3 Deceleration while recovering with steps [M]

The data give the *limits* (§1). I found no direct table of "steps and distance against Δv". From the
inverted-pendulum relations:
- A step placed a distance L ahead of the CoM gives a horizontal deceleration of about g·L/l: 3.9 m/s² for L = 0.4 m. Friction caps it at μ·g (7.8 m/s² on a dry deck, about 1.5 on an oily one).
- The CoM keeps drifting during the reaction-plus-step delay of about 0.25–0.35 s [8].

So the displacement is about v·t_d + v²/(2a). With t_d = 0.3 s and an effective a ≈ 2.5 m/s², 1 m/s
gives 0.5 m in about 0.7 s (one to two steps), and 1.5 m/s gives 0.9 m in about 0.9 s. Hof et al. [8] saw
recovery "within two steps (one stride)" for their pushes.

---

## 5. Blast (only to check that one impulse law can serve)

- **Mechanism.** Tertiary blast injury is whole-body translation by the blast wind, followed by impact. The translation literature models the body as a drag body accelerated by the dynamic-pressure impulse plus the diffraction loading (Fletcher & Bowen, "Blast-induced translational effects", 1968 [30] [P, bibliographic record only; full text not retrieved]). The parameter these sources use is an **acceleration coefficient α = A·C_D/m** (m²/kg). I could not retrieve its tabulated value for a man [U]. From the wind research's measured C_dA [§ wind note] [M]: α ≈ 0.75/80 = 0.0094 m²/kg (80 kg, facing), 0.9/100 = 0.009 m²/kg (kitted marine), and 0.44/80 = 0.0055 m²/kg side-on. Δv ≈ α·∫q dt (+ diffraction), which is exactly the wind's drag law integrated over the pulse.
- **Criteria** (the impact velocity of the translated body against a hard surface).
  - *Skull fracture.* UFC 3-340-02 gives a probable safe impact of 10 ft/s (3.05 m/s), a 50 % skull-fracture probability at 18 ft/s (5.5 m/s) and nearly 100 % at 23 ft/s (7.0 m/s). A DASA report gives a skull-fracture minimum of about 13 ft/s (4.0 m/s) [31] [S].
  - *Total body impact.* The DOE/TIC-11268 blast manual (Baker et al.), Table A.12, gives mostly safe 10 ft/s (3.05 m/s), lethality threshold 21 (6.4), 50 % lethality 54 (16.5) and ≈ 100 % lethality 138 ft/s (42 m/s) [31] [S: OCR of the manual seen only in a search summary; Baker et al., *Explosion Hazards and Evaluation* (1983), carries the same tables, not opened [U]].
- **Conclusion** [M]: a blast is one more **impulse** (N·s), J = C_dA·∫q dt (+ A·∫Δp dt for the front), entering the same Δv = J/m law with the same bands:
  - Δv > v_fall knocks the unit down,
  - the slide follows §4.2,
  - impact damage keys off the velocity at a wall (≥ 3 m/s is the "mostly safe" edge).
  
  A design call is owed: blast wind already reaches units through the wind q law (knockdown only), so a later blast impulse must not count the same q twice.

---

## 6. Walking against a sustained force

- **Self-selected speed.** Hurt, Wang, Capo-Lugo & Brown [27] [P] (a self-propelled treadmill with a horizontal resistive force at the pelvis; 9 nonimpaired adults analysed, 14 post-stroke): comfortable speed fell **linearly** with resisting force. The mean slope of normalised speed against normalised force was −0.37 ± 0.04 (nonimpaired), normalised to each person's own maximum force (the force at which the belt no longer moved). At the highest force, speed was about 55 % of the low-resistance speed. Newtons and % BW are not reported in the text.
- **Capacity at a forced pace.**
  - Gottschall & Kram [28] [P] had subjects walk at 1.25 m/s with forces from 15 % BW aiding to **15 % BW impeding**.
  - Pugh [29] [P] had a 75 kg subject walk at 1.25 m/s into winds up to 18.5 m/s: V̇O2 rose from 0.8 to 3.1 l/min. [M] The drag at 18.5 m/s is about 0.5 · 1.2 · 18.5² · 0.75 ≈ 155 N ≈ 0.21 BW.
  - So a person *can* keep normal speed against 0.15–0.2 BW at great cost, but *chooses* to slow.
  - Chang & Kram [39] [P] give the running analogue (−6 % BW impeding raised V̇O2 30 %).
- **Speed-scale curve [M].** The wind design's law (scale 1 at q_slow falling linearly to 0.5 at q_fall) restated as force over weight, using C_dA = 0.9 m² and m = 100 kg:
  - q_slow 39 Pa → F = 35 N = **0.036 W**
  - q_fall 245 Pa → F = 220 N = **0.22 W**
  - s(F/W) = 1 − 0.5·(F/W − 0.036)/(0.22 − 0.036), floored at 0.5. Its slope in "fraction of the topple force" is −0.5 over the band, close to Hurt's −0.37 normalised slope.
  - **Restating the wind law in F/W, with per-unit C_dA and m, gives the designer's mass and area dependence for free and keeps the ruled band edges for the reference marine.**

---

## 7. Other games and simulations

All [S] (community wikis), and the form only matters here.

- **Elden Ring** [32]. Every enemy has a hidden poise (stance) meter of 15–65 (bosses 80–120; outliers to 600). Each attack does poise damage separately from HP. At zero the enemy is stance-broken (a long stun with a riposte opening), and poise resets. Regeneration starts after base poise ÷ 13 seconds without a hit and then runs at 13 per second. For players, poise regenerates only after 30 s without being hit, and in practice resets on a stagger. **Form: an accumulator with a delay, then LINEAR decay, and a break threshold.**
- **Sekiro** [40]. Posture recovers more slowly as vitality (HP) drops. **Form: an accumulator whose decay rate is coupled to health.**
- **Helldivers 2** [33]. Each hit carries a Stagger Force compared against per-enemy thresholds. On players a value ≥ 10 staggers and ≥ 25 ragdolls. Every pellet of a shotgun carries the full stagger force, and "Push Force" is listed as a separate weapon value. **Form: a per-hit threshold with no accumulation (per the wiki; one commercial page claims a hidden build-up, unverified).**
- **Left 4 Dead 2** [41]. The shove staggers and is rate-limited on the *attacker*: melee fatigue after the 7th consecutive melee in campaign, the 5th in Versus. **Form: an accumulator on the attacker side.**
- **Dota 2** [42]. "Knockback" is a category of forced movement by a *set distance* (for example Ballista 50 units per attack; Gust up to 450, scaled by proximity) over a set duration. **Form: scripted displacement, not physics.**
- Not researched for lack of reliable sources in the time: Dark Souls 1 poise, Arma/DayZ (ragdoll on death), Noita.

**Reading [M].** The literature's push-recovery law (§4.3) *is* a poise meter. A unit's unabsorbed momentum
m·v_push rises with each impulse and is worked off at a constant rate m·a_rec (a linear, Coulomb-like
decay). It "breaks" (a fall) at m·v_fall. Elden Ring's delay-then-linear regeneration is the same shape,
with the delay as the reaction time. The designer's armour-debuff accumulator under focus fire can take
the identical form, with its own rate. With linear decay, a sustained input below the decay rate leaves
it at exactly zero, and an input above it breaks it in finite time. "How many shooters break it" is then
simply decay rate ÷ per-shooter input rate, a clean designer knob.

---

## 8. Proposed minimal model [M — everything in this section is my proposal]

### 8.1 State per unit

| State | Type | Meaning |
|---|---|---|
| `v_push` | 2D, Q16 m/s | unabsorbed CoM velocity from impulses |
| `react_ticks` | int | reaction-delay countdown before recovery starts |
| `effort_q` | Q16 N, low-passed | the force the unit has been resisting (for the slowed band) |
| `KNOCKED_DOWN` | existing status | — |

The armour-debuff accumulator, if built, is a separate scalar of the same linear-decay form.

### 8.2 Parameters (per species or env-profile row; reference values for the marine)

| Parameter | Unit | Pick | From |
|---|---|---|---|
| `m` (body + kit) | kg | 100 (zombie ≈ 70–80, drawn later) | — |
| `cd_area` | m² | 0.9 (zombie 0.75) | wind note §1 (Penwarden, Grigg & Rayment 1978, via Jordan 2008) |
| `k_hold` (braced hold, F/W) | — | 0.22 (= the wind q_fall edge) | wind note §2, §6 above |
| `a_rec` | m/s² | min(k_hold, μ_foot)·g = 2.2 dry; 1.5 oily (μ 0.15) | §4.1, §4.3 |
| `t_react` | s | 0.30 (7 ticks) | [8] |
| `v_inplace` fwd / back / side | m/s | 0.25 / 0.15 / 0.20 | §1.2 |
| `v_fall` fwd / back / side | m/s | 1.5 / 0.8 / 1.1 | §1.2 |
| `brace` multiplier on `v_inplace` (expected push) | — | 1.3 (optional) | [10] |
| `μ_foot` | — | dry deck 0.8, oily steel 0.15 | [23][24] |
| `μ_slide` (prone body) | — | dry 0.4, oily 0.15 | [26] + assumption |
| `stability` (existing) | — | scales `v_fall` (and `k_hold`) | wind design |
| `G_stagger` (bullets only) | — | 1 = physics; 3–5 = declared exaggeration | §3.4 |

"Forward" means the push direction along the unit's facing (pushed from behind). The threshold for an
arbitrary direction is an interpolation over the angle to facing; an octagonal blend avoids trig.

### 8.3 The two input channels (this is what keeps #140 from coming back)

1. **Sustained forces (wind, later currents) NEVER enter `v_push`.** They stay in the wind law, restated as
   F = cd_area·q and judged by F/(m·g):
   - < 0.036: nothing;
   - 0.036–0.22: slowed, cadence scale linear to 0.5;
   - ≥ 0.22·stability²: KNOCKED_DOWN.
   
   No sustained force displaces a unit, exactly as ruled. If integrated into `v_push`, a force just above
   `k_hold` would drift a unit for seconds before it fell; that is the #140 slide in another form.
2. **Impulses (shove, kick, charge, bullets, later the blast pulse) enter `v_push`:**
   Δv = J/m (bullets: J = G_stagger·p·f_stop).
   - If |Δv| (and the current |v_push|) is below `v_inplace(dir)`, nothing moves. The impulse is added to `effort_q` as J/dt, so focus fire still feeds the slowed band through the same F/W law as wind.
   - If it is above, `v_push += Δv`, `react_ticks = t_react`, and the unit is STAGGERED: it cannot act or walk while `v_push ≠ 0`.
   - If the component of `v_push` along the push direction exceeds `v_fall(dir)·stability`, the unit gets KNOCKED_DOWN, and its `v_push` then decays at μ_slide·g (the slide).

### 8.4 Per tick (24 Hz, integer)

```
if react_ticks > 0: react_ticks -= 1                      # drift at v_push (the delay)
else:               v_push = shrink(v_push, a_dec * dt)   # a_dec = a_rec standing, mu_slide*g prone
position += v_push * dt                                   # stop at the last open tile (the debris rule)
effort_q += (J_absorbed/dt - effort_q) * dt / tau         # tau ≈ 0.5 s low-pass, integer shift
band law on (cd_area*q + effort_q) / (m*g)                # one slowed/knockdown law for wind and fire
```

`shrink(v, d)` reduces |v| by d and clamps at zero: exact integer subtraction, monotone, deterministic.
Per-axis shrink is simplest. A magnitude shrink with an integer octagonal norm avoids the diagonal bias.

**Zero motion for sub-threshold input is a proof, not a tuning.** `v_push` changes only through impulses
above `v_inplace`, and the decay only removes. With no impulse above threshold, `v_push` stays at its
initial 0 by induction. Sustained forces cannot reach it at all (channel 1).

### 8.5 What each input gives (marine 100 kg, zombie 80 kg, `stability` 0.9 on the zombie, dry deck)

| Input | Marine | Zombie |
|---|---|---|
| 5.56 hit | 0.04 m/s → nothing | 0.05 → nothing |
| 5.56 × 3 shooters, sustained | effort ≈ 140 N = 0.14 W → slowed to ≈ 0.71 | 0.18 W → slowed to ≈ 0.61 |
| 12 ga slug from the front (backward push) | 0.135 → nothing | 0.17 → STAGGER, ≈ 0.06 m |
| .50 BMG (stopped) | 0.38 → STAGGER, ≈ 0.15 m | 0.47 → STAGGER |
| 100 N·s shove from behind | 1.0 → STAGGER, ≈ 0.5 m, ≈ 0.75 s | 1.25 → v_fall 1.35: STAGGER, ≈ 0.7 m |
| 100 N·s shove from the front (backward) | 1.0 > 0.8 → KNOCKED_DOWN, slide ≈ 0.13 m | down |
| 300 N·s charge | down, slide ≈ 1.1 m | down, slide ≈ 1.8 m |
| Blast pulse with Δv 5 m/s | down, slide ≈ 3.2 m (≥ 3 m/s into a wall: impact damage) | — |

With `G_stagger` = 4, a 5.56 hit on a zombie is 0.19 m/s: just a stagger from the front, not from the
side. A burst of them keeps it stumbling, and it scales with momentum and with mass.

### 8.6 Open calls for Erik (modelling decisions, not mine to make)

- Whether bullets get `G_stagger` > 1, or whether zombie slowing under fire is instead tied to leg damage, which is truer.
- Whether the wind law is restated in F/W, so mass and area matter. For the reference marine the bands are unchanged.
- How a later blast avoids counting its wind twice: impulse channel, q channel, or a split by pulse length.
- The backward and sideways thresholds and the walking-vs-standing difference are model-derived, not measured. They are worth a feel check at the screen.

---

## References

[1] Pratt, J., Carff, J., Drakunov, S., Goswami, A. (2006) "Capture Point: A Step toward Humanoid Push Recovery", *Proc. 6th IEEE-RAS Int. Conf. on Humanoid Robots*, Genoa, pp. 200–207. http://www.ambarish.com/paper/Pratt_Goswami_Humanoids2006.pdf (DOI not confirmed)
[2] Hof, A.L., Gazendam, M.G.J., Sinke, W.E. (2005) "The condition for dynamic stability", *J Biomech* 38(1):1–8. doi:10.1016/j.jbiomech.2004.03.025
[3] Pai, Y.C., Patton, J. (1997) "Center of mass velocity-position predictions for balance control", *J Biomech* 30(4):347–354. doi:10.1016/s0021-9290(96)00165-0
[4] Hof, A.L., Curtze, C. (2016) "A stricter condition for standing balance after unexpected perturbations", *J Biomech* 49(4):580–585. doi:10.1016/j.jbiomech.2016.01.021
[5] Mille, M.-L., Rogers, M.W., Martinez, K., Hedman, L.D., Johnson, M.E., Lord, S.R., Fitzpatrick, R.C. (2003) "Thresholds for inducing protective stepping responses to external perturbations of human standing", *J Neurophysiol* 90(2):666–674. doi:10.1152/jn.00974.2002 (abstract only; full text refused)
[6] Thelen, D.G., Wojcik, L.A., Schultz, A.B., Ashton-Miller, J.A., Alexander, N.B. (1997) "Age differences in using a rapid step to regain balance during a forward fall", *J Gerontol A* 52(1):M8–13. doi:10.1093/gerona/52a.1.m8
[7] Wojcik, L.A., Thelen, D.G., Schultz, A.B., Ashton-Miller, J.A., Alexander, N.B. (1999) "Age and gender differences in single-step recovery from a forward fall", *J Gerontol A* 54(1):M44–50. doi:10.1093/gerona/54.1.m44
[8] Hof, A.L., Vermerris, S.M., Gjaltema, W.A. (2010) "Balance responses to lateral perturbations in human treadmill walking", *J Exp Biol* 213(15):2655–2664. doi:10.1242/jeb.042572 (full text read)
[9] Vlutters, M., van Asseldonk, E.H.F., van der Kooij, H. (2016) "Center of mass velocity-based predictions in balance recovery following pelvis perturbations during human walking", *J Exp Biol* 219(10):1514–1523. doi:10.1242/jeb.129338 (methods read)
[10] Santos, M.J., Kanekar, N., Aruin, A.S. (2010) "The role of anticipatory postural adjustments in compensatory control of posture: 2. Biomechanical analysis", *J Electromyogr Kinesiol* 20(3):398–405. doi:10.1016/j.jelekin.2010.01.002
[11] Maki, B.E., McIlroy, W.E. (1997) "The role of limb movements in maintaining upright stance: the 'change-in-support' strategy", *Phys Ther* 77(5):488–507. doi:10.1093/ptj/77.5.488
[12] Chatagnon, T., Olivier, A.-H., Hoyet, L., Pettré, J., Pontonnier, C. (2023) "Stepping strategies of young adults undergoing sudden external perturbation from different directions", *J Biomech* 157:111703. doi:10.1016/j.jbiomech.2023.111703
[13] Feldmann, S., Adrian, J. (2023) "Forward propagation of a push through a row of people", *Safety Science* 164:106173. doi:10.1016/j.ssci.2023.106173 ; arXiv:2503.19104 (75 / 90 N·s figures from the search summary of the paper; PDF not text-extractable here)
[14] Agrawal et al. (2010), CIGR Journal, isometric push/pull strength of agricultural workers of Central India (https://cigrjournal.org/index.php/Ejounral/article/view/1342); Das & Wang (2004) via search summary; Kroemer AMRL report AD0720252 (https://apps.dtic.mil/sti/pdfs/AD0720252.pdf). [S] — values from search summaries only.
[15] "Investigation of Fall from Height Events According to the Laws of Physics" (2025), *Forensic Sciences* 6(3):72. doi:10.3390/forensicsci6030072 [S: search summary; publisher page refused]
[16] Usman, J., McIntosh, A.S., Fréchède, B. (2011) "An investigation of shoulder forces in active shoulder tackles in rugby union football", *J Sci Med Sport* 14(6):547–552. doi:10.1016/j.jsams.2011.05.006
[17] Vagner, M., Cleather, D.J., Olah, V., Vacek, J., Stastny, P. (2023) "A Systematic Review of Dynamic Forces and Kinematic Indicators of Front and Roundhouse Kicks…", *Sports* 11(8):141. doi:10.3390/sports11080141
[18] 5.56 mm M855 specification summary (MIL-C-63989: 61.7 ± 1.5 gr, 3000 ± 40 fps at 78 ft, 20" barrel), https://inetres.com/gp/military/infantry/rifle/556mm_ammo.html [S]; TM 43-0001-27 *Army Ammunition Data Sheets, Small Caliber* (not opened) [U]
[19] .50 M33 ball 661 gr / 2910 fps — commonly quoted, not verified against TM 43-0001-27 [U]. Other rounds' nominal loads in §3.1 [U].
[20] Patrick, U.W. (1989) *Handgun Wounding Factors and Effectiveness*, FBI Academy Firearms Training Unit, Quantico, 14 July 1989. Reprint read at https://randywakeman.com/ballltd73.htm ; archive copy https://ia800200.us.archive.org/16/items/fbi-handgun-wounding-factors-and-effectiveness/fbi-handgun-wounding-factors-and-effectiveness.pdf ; NCJ 116350.
[21] Redfern, M.S., Cham, R., Gielo-Perczak, K., et al. (2001) "Biomechanics of slips", *Ergonomics* 44(13):1138–1166. doi:10.1080/00140130110085547 (abstract)
[22] RCOF ≈ 0.18 for healthy adults and a 0.20 baseline: search summaries of a Pittsburgh paper and a Virginia Tech thesis (https://vtechworks.lib.vt.edu/bitstreams/770a9cad-b9f4-4117-8d0a-723f23fad6c2/download) [S]; Burnfield, J.M., Powers, C.M. (2006) "Prediction of slips…", *Ergonomics* 49(10):982–995, doi:10.1080/00140130600665687 (abstract, no number).
[23] MIL-PRF-24667C, Table V, reprint at https://www.paintsdopesvarnishesrelatedprodu.emilspec.com/MIL-PRF-24667/page11.html [S]
[24] EN ISO 20345:2011 / EN ISO 13287 SRA/SRB minimums, e.g. https://www.hazchemsafety.com/en-iso-20345-2011/ and https://hse.gov.uk/slips/manufactfoot.htm [S]
[25] Industrial Health 52(5):414 (2014), shoes on glycerol-contaminated stainless steel, https://www.jniosh.johas.go.jp/en/indu_hel/doc/IH_52_5_414.pdf [S: search summary]
[26] Body–road friction in pedestrian-throw reconstruction (Searle 0.7, Wood & Simms 0.58), via https://arxiv.org/pdf/1512.00790 and https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-b275f568-8db4-4629-86b9-ebf32a673c6f/c/AM74_08_wicher_EN.pdf [S]
[27] Hurt, C.P., Wang, J., Capo-Lugo, C.E., Brown, D.A. (2015) "Effect of progressive horizontal resistive force on the comfortable walking speed of individuals post-stroke", *J NeuroEng Rehabil* 12:12. doi:10.1186/s12984-015-0007-7 (full text read)
[28] Gottschall, J.S., Kram, R. (2003) "Energy cost and muscular activity required for propulsion during walking", *J Appl Physiol* 94(5):1766–1772. doi:10.1152/japplphysiol.00670.2002
[29] Pugh, L.G.C.E. (1971) "The influence of wind resistance in running and walking and the mechanical efficiency of work against horizontal or vertical forces", *J Physiol* 213(2):255–276. doi:10.1113/jphysiol.1971.sp009381
[30] Fletcher, E.R., Bowen, I.G. (1968) "Blast-induced translational effects", *Ann N Y Acad Sci* 152(1):378–403. doi:10.1111/j.1749-6632.1968.tb11989.x (record only). Also Bowen, Fletcher & Richmond (1968) *Estimate of man's tolerance to the direct effects of air blast*, DASA-2113 — not opened [U]. Baker, W.E. et al. (1983) *Explosion Hazards and Evaluation*, Elsevier — not opened [U].
[31] UFC 3-340-02 (2008) *Structures to Resist the Effects of Accidental Explosions* (skull-fracture velocities), https://pdfcoffee.com/ufc-3-340-02-structures-to-resist-the-effects-of-accidental-explosions-usa-2008-pdf-free.html ; DOE/TIC-11268 *A Manual for the Prediction of Blast and Fragment Loadings on Structures*, Table A.12, https://apps.dtic.mil/sti/tr/pdf/ADA395543.pdf (download refused) [S]
[32] Eldenpedia, "Poise" and "Stance", https://eldenring.wiki.gg/wiki/Poise , https://eldenring.wiki.gg/wiki/Stance [S]
[33] Helldivers Wiki, "Status Effects", https://helldivers.wiki.gg/wiki/Status_Effects [S]
[34] Hsiao, E.T., Robinovitch, S.N. (1999) "Biomechanical influences on balance recovery by stepping", *J Biomech* 32(10):1099–1106. doi:10.1016/s0021-9290(99)00104-9
[35] Feldmann, S., Adrian, J., Boltes, M. (2024/2025) "Propagation of controlled frontward impulses through standing crowds", *Collective Dynamics* (https://collective-dynamics.eu/index.php/cod/article/view/A148); arXiv:2503.19110 (HTML read)
[36] Pellman, E.J., Viano, D.C., Tucker, A.M., Casson, I.R., Waeckerle, J.F. (2003) "Concussion in professional football: reconstruction of game impacts and injuries", *Neurosurgery* 53(4):799–812. doi:10.1093/neurosurgery/53.3.799
[37] MythBusters (2005 season) episode list, https://en.wikipedia.org/wiki/MythBusters_(2005_season) [S]
[38] Hanson, J.P., Redfern, M.S., Mazumdar, M. (1999) "Predicting slips and falls considering required and available friction", *Ergonomics* 42(12):1619–1633. doi:10.1080/001401399184712
[39] Chang, Y.H., Kram, R. (1999) "Metabolic cost of generating horizontal forces during human running", *J Appl Physiol* 86(5):1657–1662. doi:10.1152/jappl.1999.86.5.1657
[40] Sekiro wiki (Fextralife), "Stats", https://sekiroshadowsdietwice.wiki.fextralife.com/Stats [S]
[41] Left 4 Dead Wiki, "Melee", https://left4dead.fandom.com/wiki/Melee [S]
[42] esports.gg, "Dota 2 patch 7.36b biggest buffs and nerfs" (knockback category), https://esports.gg/news/dota-2/dota-2-patch-7-36b-biggest-buffs-and-nerfs ; DotaFire "Gust" https://www.dotafire.com/dota-2/skill/gust-234 [S]

PubMed was the index for references [2]–[13], [16], [17], [27]–[30], [34], [36], [38] and [39]; their DOIs are given above.

**Gaps (searches that returned nothing usable):**
- a measured human impulse threshold for a backward or sideways push (N·s);
- displacement and number of steps against Δv;
- riot-shield push forces;
- clothing-on-steel sliding friction;
- the tabulated blast acceleration coefficient for a man;
- the full text of Mille 2003 and of Hunt et al. 1976.
