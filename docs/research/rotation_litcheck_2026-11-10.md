# Rotation and speed literature check for the unit impulse design (2026-11-10)

Research only; nothing in any repository was changed. This checks the PLACEHOLDER rows of
`docs/unit_impulse_design_2026-11-10.md` §3 (yaw inertia, own-turning top speed and acceleration, the
backward and sideways walking fractions, the walk and run speeds, the prone drag area). It follows the
conventions of `docs/research/impulse_on_a_person_2026-11-10.md`: g = 9.81 m/s², the reference marine is
80 kg body + 20 kg kit = 100 kg, about 1.80 m tall, holding a 3.4 kg carbine at the shoulder.

Labels: **[P]** primary source read (abstract or full text, as retrieved here); **[S]** secondary (a
search summary, a citation in another paper, a popular article); **[U]** unverified or from memory;
**[M]** my own calculation or estimate.

This was a short check (about 20 tool calls). DTIC (where the Santschi et al. 1963 and DuBois et al. 1964
reports are) did not resolve from here, and the MDPI and PMC web pages refused direct fetches, so some
sources were read only through PubMed's full-text service or search summaries. Each number says which.

---

## 1. Yaw (longitudinal-axis) moment of inertia

### 1.1 The body alone

- **NASA OCHMO-HB-004 Rev A (Dec 2023), Appendix A, Table 14, "Whole Body Moment of Inertia of the
  Crewmember, Unsuited"** [P, the table text was extracted from the PDF]: principal axis **Zp (the
  longitudinal axis) min 0.73, max 2.05 kg·m²**. For comparison, Xp is 6.59–17.69 and Yp is 6.12–16.43.
  The handbook's notes say: the data are for the projected 2015 astronaut population, all measured in 1 g,
  "only applicable if the subject is exactly in the anatomical neutral position" (standing, arms at the
  sides, palms forward), and other postures must be built up from segment properties. It gives a min and a
  max only, without the masses behind them. A plausible reading is the smallest female to the largest male
  of that population [U].
- The classic sources (Santschi, DuBois & Omoto 1963, AMRL-TDR-63-36, 66 men in several standing and seated
  postures; DuBois et al. 1964, AMRL-TR-64-110, the pressure-suit follow-up) are what RP-1024 Vol. I and
  later NASA handbooks tabulate. I confirmed that RP-1024 Vol. I lists a "whole body moments of inertia —
  Santschi et al." table [S, the archive.org table of contents], but I could not open the numbers. The
  follow-up gives its compound-pendulum accuracy as 2–8 % depending on posture and axis [S].
- Chandler et al. 1975, Zatsiorsky / de Leva 1996 and Erdmann were not opened in this check [U]. They give
  SEGMENT parameters, from which the whole-body value would be summed. That is the route for the
  rifle-holding posture, but it was not done here.
- **Scaling to an 80 kg man [M].** If I ∝ m·L² and L ∝ m^(1/3), then I ∝ m^(5/3). Taking 2.05 kg·m² as a
  large man of about 100–110 kg gives about **1.2–1.4 kg·m²** at 80 kg. Interpolating linearly between the
  table's ends gives the same band. Pick **1.3 kg·m²** for the body, arms at the sides.

### 1.2 What the posture and the kit add [all M: mass × radius² estimates, not measured]

| Item | Mass | Radius from the yaw axis / shape | Added I |
|---|---|---|---|
| Arms moved from the sides to a shooting grip (each arm about 5 % of the body, 4 kg; CoM at about 0.2 m lateral when hanging, about 0.3–0.35 m forward and out when aiming) | 2 × 4 kg | from about 0.2 m to about 0.32 m | +0.3 to +0.5 |
| Carbine, 3.4 kg, about 0.85 m long, pointing forward, its CoM about 0.35 m in front of the axis | 3.4 kg | m·r² = 0.42, plus its own m·L²/12 about a vertical axis = 0.20 | +0.6 |
| Body armour and plates (about 10 kg close around the torso, r ≈ 0.15–0.2 m) | 10 kg | thin shell | +0.25 to +0.4 |
| Pack, webbing and pouches (about 10 kg, CoM about 0.2–0.25 m behind the axis) | 10 kg | m·r² plus its own about 0.05 | +0.45 to +0.65 |

- Total with the rifle up and full kit: **≈ 2.9–3.4 kg·m²**.
- With the kit but the rifle slung close: ≈ 2.2–2.6 kg·m².
- An unarmed 80 kg zombie with its arms out: ≈ 1.5–1.8 kg·m².

The design's placeholder of 2.5 kg·m² is inside the band but low for "rifle held forward plus pack".
**Pick 3.0 kg·m²** for the reference marine [M], confidence medium (the body term is sourced; the kit and
rifle terms are estimates).

**Effect of a hit [M].** A 5.56 round (3.8 N·s, fully stopped) 0.2 m off the axis gives an angular
impulse of 0.76 N·m·s, so Δω = 0.25 rad/s ≈ 15 °/s on 3.0 kg·m². That is small next to the unit's own
turning authority (below), so a single hit turns a marine a degree or two before it corrects.

---

## 2. How fast a person turns in place

### 2.1 Measured turns (all paced or self-selected; no MAXIMAL standing-turn study was found)

- **Sensors 21(8):2827 (2021), "The Effect of Different Turn Speeds on Whole-Body Coordination in
  Younger and Older Healthy Adults"** [P, full text via PubMed Central; doi:10.3390/s21082827]. 21 young
  adults (22.5 y, 61.6 kg, 1.63 m) turned 180° on the spot, CUED to finish in **1.5 s (fast), 2 s, or
  3 s**. The young group's peak HEAD yaw velocity was **272 ± 48 °/s fast, 181 moderate, 137 slow** [S:
  the table values came through a search summary; the PMC text has no table]. Above a head speed of about
  100 °/s the head leads the pelvis by 40–50° in young adults [P]. The fast condition is a paced turn, not
  a maximal one.
- **Applied Sciences 12(11):5474 (2022)**, standing turns by amplitude: at 180°, young adults' peak head
  yaw velocity was 172.8 ± 24.9 °/s at their own pace [S, search summary].
- Inertial-sensor 180° turn studies (Parkinson's disease with controls) report peak trunk yaw velocities of
  about 160–175 °/s and durations of about 1.8 s at self-selected pace [S, it is not clear which group].
- Hase & Stein 1999, Akram, Frank & Fraser, practical-shooting transition times and military "about turn"
  timings: **not found in this check** [U]. My recollection is that a trained, deliberate 180° turn and
  engage takes about 0.6–1.0 s and a 90° turn about 0.4–0.6 s, but no source was read for it.

### 2.2 What the numbers imply [M]

For a bang-bang turn (full acceleration, then full deceleration, no cruise), θ = α·t²/4 and the peak
ω = α·t/2.
- **The paced "fast" 180° in 1.5 s** needs α = 4π/1.5² ≈ 5.6 rad/s² and a peak of 4.2 rad/s = **240 °/s**.
  That matches the measured head peak of 272 °/s (the head leads and overshoots the pelvis), which is a
  useful check on the method.
- **A plausible maximal 180° in about 0.7 s** [U time] needs α ≈ 26 rad/s² and a peak of about 9 rad/s
  ≈ 515 °/s.
- **A plausible 90° in about 0.4 s** [U time] needs α ≈ 39 rad/s² and a peak of about 450 °/s.

### 2.3 The torque available [M, with one S]

- **Trunk axial rotation strength.** A study of 17 male students gives maximal isometric trunk rotation of
  about **72–76 ± 23 N·m** [S, a search-summary snippet whose posture was unclear]. Isokinetic axial
  rotation torque falls with speed from 30 to 120 °/s (24 men, 66.7 kg, seated with the pelvis fixed) [P,
  doi:10.1155/2014/623191; the values are only in its figures, not read]. McGill 1991 and Kumar 1995 are
  the classic sources and were not opened [U].
- **The foot-friction limit on a whole-body yaw torque.** Turning the whole body needs a free moment from
  the floor. With the weight split over two feet about 0.25–0.3 m apart, the largest couple is about
  μ·W·(half the stance width): 0.8 × 981 N × 0.12–0.15 m ≈ **95–120 N·m** on a dry deck, and about 20 N·m on
  oily steel (μ 0.15).
- **What the design's acceleration asks for.** With I = 3.0 kg·m², 50 rad/s² needs 150 N·m, above both
  limits. 25–35 rad/s² needs **75–105 N·m**, which agrees with the trunk strength and the dry-deck friction
  couple and with the maximal-turn times above. People also turn by STEPPING (each step puts a new
  tangential foot force down), so the sustained limit can exceed the standing couple. Even so, an order of
  100 N·m stands.
- **Pick α_max = 30 rad/s² for the marine** [M], which is about 90 N·m.
  - Lighter, unloaded zombies (I ≈ 1.6) with the same torque would reach about 55 rad/s².
  - On an oily deck the friction couple caps α at about 7 rad/s² (μ 0.15). That is a natural consequence if
    α is derived as min(τ_muscle, μ·W·r)/I. **Erik's call** whether to model it.
- **Top speed.** The design's 360 °/s cap (G1p) sits between the paced-fast 240 °/s and the plausible
  maximal 450–515 °/s. With 360 °/s and 30 rad/s², a 90° turn takes 0.46 s (a triangle profile) and a 180°
  turn 0.73 s [M]. Both feel right for a kitted soldier with a raised rifle. **Keep 360 °/s.**

---

## 3. Backward and sideways walking relative to forward

- **Backward walking.** A study of directional difference (Gait & Posture 2022, "Backward relative to
  forward walking speed and falls in older adults with dementia") estimates the forward-minus-backward
  speed difference at **about 30 % in young and middle-aged adults (backward ≈ 0.7 × forward)**, rising to
  about 50 % in older adults [S, search summary]. Laufer 2005 reports backward slower than forward at both
  the preferred and the fast speed [S, cited]. A 21.8 % drop in controls appears in another study [S].
- **Sideways walking.** No speed ratio was found in this check [U].
- **Maximal backward running.** No source was read [U]. My recollection is that it is roughly 70–80 % of a
  forward sprint over short distances; unverified.
- **Verdict.** The literature does **not contradict** the designer's 0.6 (backward) and 0.8 (sideways).
  Unloaded young adults walk backward at about 0.7–0.8 of their forward speed. 0.6 is slightly
  conservative, which is defensible for a loaded soldier keeping a rifle on target and unable to see behind.
  The sideways 0.8 has no number to contradict it.

---

## 4. Walk and run speeds under about 20 kg

- **Swedish combat engineers in simulated combat** [P, doi:10.1093/milmed/usac163, full text]: 42 soldiers
  (77.3 kg body, 94.6 kg with combat equipment, fighting-order load 17.3 ± 2.6 kg excluding the weapon), with
  10 Hz GPS over 8-hour exercises.
  - Peak velocity per task: **16.6 ± 2.2 km/h (4.6 m/s)** in the urban assault, 11.6–15.3 km/h in the
    delaying actions, individual range 6.7–21.8 km/h (1.9–6.1 m/s).
  - Maximal acceleration was 1.4–2.9 m/s² (task means).
  - The lab protocol used marching at **5.4 km/h (1.5 m/s)** in combat gear as its first, easy stage.
- **US Army load-carriage literature.** The common reference pace is about 1.34 m/s (4.8 km/h,
  "3 mph") [U, recollection of the Knapik reviews; not re-read]. A 2024 Ergonomics study (weapon handling
  under a 23.2 kg load) used 3.5–6.5 km/h plus a self-selected speed; the self-selected value was not in the
  summary [S]. A popular article cites 1.39 m/s (5 km/h) as the soldier's ideal march speed [S].
- **The rush.** The US Army Infantry Magazine (Barringer & Rooney 2016) and Army Times give a goal of about
  **6.7 m/s** over a 3–5 s rush [S]. That is a goal, not a measured mean, and it is said to apply to light
  loads (≤ 16 kg).
- **Verdict.** **1.4 m/s walk: confirmed** (1.34–1.5 m/s). **4.5 m/s run: confirmed** as the typical peak
  under fighting order in the field (4.6 m/s mean peak). A short all-out rush could reach 5.5–6.7 m/s if the
  game wants a sprint above the run.

---

## 5. Prone drag area

- **No source found** for C_d·A of a person lying on the ground [U]: not in the searches of wind-tunnel,
  skeleton-sled or blast-translation literature. Bowen, Fletcher & Richmond (DASA-2113, 1968) and Baker et
  al. (1983) treat the prone orientation as reducing translation, but their tables were not opened [U].
- **Context** [S]: a CFD study gives a standing 1.72 m person's frontal area as 0.555 m², and side-on drag as
  58 % of facing. An alpine-skiing wind-tunnel study found frontal area to drive most of the C_d·A change
  across postures (−41 % area against −23 % C_d from upright to tuck).
- **Estimate [M].**
  - **Wind along the body** (head- or feet-first): the frontal area of a prone marine is about 0.5 m wide ×
    0.3–0.4 m high (pack and helmet included) ≈ 0.15–0.2 m². C_d is near 1 for a bluff body on a wall, so
    **C_d·A ≈ 0.15–0.2 m²**.
  - **Wind across the body**: about 1.8 m × 0.3–0.35 m ≈ 0.55–0.6 m² of area, C_d about 0.8–1.0, so
    **C_d·A ≈ 0.45–0.6 m²**.
  - Both sit low in the floor's boundary layer, where real air is slower than in the free stream. A game
    reading the tile's q overstates this; I did not quantify it.
  - So **0.15 m² is fair for wind along the body, and low (by about 3×) for wind across it**. Against the
    standing 0.9 m², prone is roughly 1/6 to 2/3 depending on the angle.
- **Pick 0.15 head-on, and say it is the favourable orientation** [M]. If the game knows the prone unit's
  facing, an angle blend from 0.15 (along) to 0.5 (across) is the more truthful form. **Erik's call.**

---

## 6. Summary table

| Parameter | Literature range | Pick (reference marine) | Source | Confidence |
|---|---|---|---|---|
| Body yaw inertia, standing, arms at the sides | 0.73–2.05 kg·m² (a crew population, min–max) | 1.3 kg·m² (80 kg) | NASA OCHMO-HB-004 Table 14 [P] + scaling [M] | medium-high |
| Added by the rifle held at the shoulder | — | +0.6 kg·m² | [M] m·r² + m·L²/12 | medium |
| Added by the arms in a shooting grip | — | +0.3–0.5 kg·m² | [M] | low-medium |
| Added by the armour and pack (20 kg) | — | +0.7–1.0 kg·m² | [M] | low-medium |
| **Total yaw inertia I** | 2.9–3.4 kg·m² [M] | **3.0 kg·m²** (placeholder 2.5) | [P]+[M] | medium |
| Paced-fast 180° standing turn | 1.5 s cued; head peak 272 ± 48 °/s | — | Sensors 2021 [P] (peak [S]) | high (as paced) |
| Self-selected 180° turn | about 1.8 s; head / trunk peak 160–175 °/s | — | [S] | medium |
| Maximal 180° / 90° turn time | about 0.6–1.0 s / 0.4–0.6 s | 0.73 s / 0.46 s (implied by the picks) | [U] + [M] | low |
| Own-turning top speed | 240 °/s (paced) to about 500 °/s (maximal, implied) | **360 °/s** (keep G1p's cap) | [P]/[S] + [M] | medium |
| Trunk axial rotation torque | about 72–76 ± 23 N·m isometric | — | [S] | low-medium |
| Floor friction couple (two feet, dry deck μ 0.8) | 95–120 N·m [M] | — | [M] | medium |
| **Own-turning acceleration α** | 25–40 rad/s² (from torque / I and turn times) | **30 rad/s²** (placeholder 50) | [M] | medium-low |
| Backward walk / forward walk | about 0.7 (young), 0.5 (old) | 0.6 (designer's; not contradicted) | [S] | medium |
| Sideways walk / forward walk | no data found | 0.8 (designer's) | — | low |
| Backward run / forward sprint | about 0.7–0.8 | — | [U] | low |
| Walk under about 20 kg | 1.34–1.5 m/s | **1.4 m/s** (confirmed) | [P] (5.4 km/h protocol), [U] Knapik | high |
| Run (peak) under about 17 kg + weapon | 4.6 ± 0.6 m/s field mean peak; range 1.9–6.1 | **4.5 m/s** (confirmed) | Mil Med 2022 [P] | high |
| Rush / sprint goal (light load) | 6.7 m/s | (optional sprint) | Infantry Magazine 2016 [S] | low-medium |
| Prone C_d·A, wind along the body | 0.15–0.2 m² | **0.15 m²** | [M] | low |
| Prone C_d·A, wind across the body | 0.45–0.6 m² | (0.5 if the angle is modelled) | [M] | low |

## References

1. NASA OCHMO-HB-004 Rev A (Dec 2023), *Anthropometry and Biomechanics* handbook, Appendix A, Table 14.
   https://www.nasa.gov/wp-content/uploads/2023/12/ochmo-hb-004-rev-a-dec2023.pdf [P, table extracted]
2. Santschi, W.R., DuBois, J., Omoto, C. (1963) *Moments of Inertia and Centers of Gravity of the Living
   Human Body*, AMRL-TDR-63-36, Wright-Patterson AFB [cited, not opened]. DuBois, J. et al. (1964)
   AMRL-TR-64-110, https://apps.dtic.mil/sti/pdfs/AD0609863.pdf [could not be reached from here].
   NASA RP-1024 Vol. I (1978), https://archive.org/stream/nasa_techdoc_19790003563/19790003563_djvu.txt
   [table of contents only].
3. "The Effect of Different Turn Speeds on Whole-Body Coordination in Younger and Older Healthy Adults",
   *Sensors* 21(8):2827 (2021), doi:10.3390/s21082827 (PMC8074235; PubMed) [P text; the peak values S].
4. "A Comparison of Turning Kinematics at Different Amplitudes during Standing Turns between Older and
   Younger Adults", *Applied Sciences* 12(11):5474 (2022), doi:10.3390/app12115474 [S].
5. "Angular Velocity Affects Trunk Muscle Strength and EMG Activation during Isokinetic Axial Rotation",
   *BioMed Research International* (2014), doi:10.1155/2014/623191 (PMC3996988; PubMed) [P, no numbers in
   the text].
6. Trunk rotation torque in healthy male students (Ishida et al.), via the search snippet of
   https://brieflands.com/articles/mejrh-91039 [S].
7. "Backward relative to forward walking speed and falls in older adults with dementia", *Gait & Posture*
   (2022), https://www.sciencedirect.com/science/article/pii/S0966636222001369 [S]; Laufer, Y. (2005), as
   cited there [S].
8. "Physiological Demands and Characteristics of Movement During Simulated Combat", *Military Medicine*
   (2022), doi:10.1093/milmed/usac163 (PMC10629984; PubMed) [P].
9. Barringer, N., Rooney, M. (2016) "The Rush: How Speed Can Save Lives", *Infantry Magazine*,
   https://www.benning.army.mil/infantry/magazine/issues/2016/APR-JUL/pdf/4)%20Barringer_Rush.pdf [S];
   Army Times (2018) "Move or die" [S].
10. "The effect of weapon handling during load carriage across a range of military-relevant walking
    speeds", *Ergonomics* (2024), doi:10.1080/00140139.2024.2400125 [S].
11. "The Wind Gusts Effects on Human Body based on CFD Simulations" (ResearchGate 346967209) [S];
    "An Empirical Model of Aerodynamic Drag in Alpine Skiing", *Proceedings* 2(6):310 (2018),
    doi:10.3390/proceedings2060310 [S].

**Gaps:**
- the Santschi standing-posture table (DTIC was unreachable);
- any MAXIMAL standing-turn timing or practical-shooting turn data;
- Hase & Stein 1999;
- a sideways walking-speed ratio;
- maximal backward running speed;
- any measured prone drag area;
- McGill / Kumar trunk torque values.
