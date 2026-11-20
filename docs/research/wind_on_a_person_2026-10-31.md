# Wind on a person: slow, hunch and fall thresholds (2026-10-31)

Issue **#140** (a room at 1.44 atm pushed a squad out of its door for 35 s). This is research only and changes nothing in the repo. Every speed below is at ρ = 1.225 kg/m³ (ISA sea level); the game's ambient air is 1.204 kg/m³ (1 atm, 293 K). Dynamic pressure is q = ½ρv².

## 1. Thresholds for a standing adult (~80 kg with kit)

| Band | Literature | Pick (v) | q (Pa) |
|---|---|---|---|
| **v_slow**: the wind's force is felt and walking slows | Penwarden (1973): 5 m/s is the onset of discomfort and >10 m/s is "definitely unpleasant" [1]. Penwarden (1975) Beaufort table: B5 (8.0–10.7 m/s) "force of wind felt on body" [2]. Jordan (2008): the first person was displaced by a sudden gust at 10 m/s (front-on), and the first step came at a normalised force F/W ≈ 0.04 [3] | **8 m/s** (range 5–10) | **39** |
| **v_hunch**: walking is hard, people brace or crouch | B6 (10.8–13.8 m/s) "difficult to walk steadily". B7 (13.9–17.1 m/s) "inconvenience felt when walking" [2]. Jordan: every subject was displaced by a 15 m/s gust in the front and back orientations [3]. Lawson's safety limit for a sensitive pedestrian is 15 m/s [4] | **14 m/s** (range 11–17) | **120** |
| **v_fall**: a person is blown off their feet | B8 (17.2–20.7 m/s) "great difficulty with balance". B9 (20.8–24.4 m/s) "people blown over by gusts" [2]. Hunt, Poulton & Mumford (1976) [5], as summarised by Jordan [3]: a steady uniform wind under 20–30 m/s is safe for walking, and a gusty wind is safe while its equivalent steady speed ū(1+3I) stays under 20 m/s, i.e. ū < 14 m/s at I = 0.13. Lawson's safety limit for a typical pedestrian is 20 m/s [4]. Melbourne (1978): an annual peak gust of 23 m/s [6] (secondary source only). Melbourne and Joubert (1971) saw two girls fall when a 12 m/s wind gusted to 23 m/s [3] | **20 m/s** (range 20–25 for gusts, up to 30 for steady wind) | **245** |

These numbers are **gust or equivalent-steady** speeds. The sim's wind is instantaneous at 0.333 m, so the right comparison is the gust and step-change values. The hourly-mean criteria are not comparable: NEN 8100 puts its danger threshold at a 15 m/s hourly mean exceeded more than 0.05–0.3 % of the time [7], and a mean of that size already carries gusts of 20 m/s or more.

**Drag area.** Penwarden, Grigg & Rayment (1978) measured people standing in a wind tunnel [8], with values as tabulated by Jordan [3]:

- Facing the wind, in trousers and a jacket: C_d = 1.17 and A/A_DuBois = 0.32.
- Side-on: C_d = 1.01 and A/A_DuBois = 0.22.
- In a flapping coat: C_d = 1.33 and A/A_DuBois = 0.35.

For a person of 80 kg and 1.80 m, A_DuBois = 2.0 m². That gives C_dA = **0.75 m²** facing (0.44 m² side-on, 0.93 m² in a coat). A marine in kit sits nearer the coat value. I use **0.75 m²** in the rest of this note, and 0.9 m² if the kit is counted.

## 2. Model: drag against friction and toppling

F = q·C_dA. At 8, 14, 20 and 25 m/s, F = 29, 88, 183 and 286 N, which is F/W = 0.04, 0.11, 0.23 and 0.37 (W = 785 N).

- **Sliding** happens when F ≥ μmg. The friction coefficient for a dry boot on a deck is an assumption I could not source in time: μ ≈ 0.4–0.6. That gives v_slide = 26–29 m/s. The friction a normal walking step needs (the RCOF of Redfern et al. 2001 [9]; Beschorner et al. 2016 [10]) is far lower than what is available on a dry floor. **Sliding binds only on a slick deck** (μ ≲ 0.25 → about 20 m/s).
- **Toppling** happens when F·h ≥ mg·b. Take h ≈ 1.0 m for the drag centre (about 0.55 of body height, a standard anthropometric assumption) and b as half the base of support. With the feet together and the body rigid (b/h ≈ 0.12), the person topples at **14 m/s**. Braced, with a wide stance and leaning into the wind (b/h ≈ 0.25–0.30), the limit is **21–23 m/s**.
- **Agreement with measurement.** Jordan's first step at F/W ≈ 0.04 is 8.3 m/s, which is v_slow. The rigid-stance topple at 14 m/s is v_hunch: past it a person must brace or step to stay up. The braced topple at about 20–23 m/s is v_fall. **Toppling binds, not sliding.** The measured falls (B9, Hunt, Lawson) sit exactly where a braced stance runs out of moment.
- **Is a 2.5× knockdown margin reasonable?** Yes, if it is a margin in **speed**: 20/8 = 2.5, and the onset of B5 against the onset of B9 is 20.8/8.0 = 2.6. In **q** the same margin is 6.25×. A 2.5× margin in q would put the fall at about 12.6 m/s, which is too low.
- **Why ~v² and not v⁴.** A body about 0.4 m wide in a wind of a few m/s has Re = vL/ν ≈ 10⁵–10⁶. Its drag is form drag, the momentum flux it stops, so C_d is constant and **F ∝ ρv²**. The power to walk against it is F·(v_wind + v_walk), which goes as v³. No mechanism makes the force go as v⁴; the exponent 4 belongs to radiation (T⁴). Strictly, the v in the drag law is the *relative* speed: walking into the wind adds about 1.4 m/s, which is small next to the thresholds.

## 3. Threshold on momentum flux, not the pressure gradient

**How the game pushes today.** `apply_wave_push` (`src/simulation/exchange.py`) reads the footprint's edge-line pressure difference Δp (atm, across the 3×3 footprint of about 1 m). It applies dv = `k_push`·Δp/m in tiles/s. The current `[exchange]` values are:

- `k_push = 400.0`
- `knockdown_dv_threshold = 6.0` tiles/s, times `stability`: 1.0 on the default env profile, 0.9 for zombies
- `push_grad_floor_atm = 0.001`
- `push_max_tile_per_tick = 0.5`

At m = 80, the push therefore moves on any Δp > 0.001 atm (101 Pa across 1 m), and knockdown needs Δp = 1.2 atm.

**Why the gradient is the wrong input for wind.** In flowing air the gradient is what accelerates the gas: ρ Dv/Dt = −∇p. The force it puts on a body directly is the buoyancy-like V·∇p. With V ≈ 0.08 m³ and a flow whose dynamic pressure q builds up over a length L, V·∇p ≈ V·q/L. Drag is C_dA·q. The ratio, drag over gradient force, is C_dA·L/V ≈ 9·L[m]. For any flow that accelerates over more than about 0.1 m, drag dominates. A steady gradient therefore does not say how hard the air pushes. The 0.001 atm floor is a noise floor, not a force threshold, and above it dv acts as a drift velocity for as long as the gradient lasts. That is why a weak but long vent shoves a squad for 35 s.

**What is different about a shock front.** At a shock, L is a few cm and the Δp across the body loads it directly (diffraction loading). Even so, the flow behind a shock has a large q: for overpressures of 0.1–0.3 atm the induced wind is about 25–65 m/s (Rankine–Hugoniot). A q reading would therefore also floor people in a blast.

**Recommendation.** Threshold on **q = ½·ρ·|v|²**, read at the footprint (the footprint maximum, matching the gas rows' reduction). Build it from the sim's own fields:

- **Density:** ρ = ρ_amb·(N/N_amb), with N_amb = 1.0 meaning ambient air at 1 atm and 293 K, so ρ_amb = 1.204 kg/m³. A single-ρ_amb conversion ignores molar-mass differences between gases; that is fine for air and steam-poor mixtures.
- **Wind:** `gmap.wind_x`/`wind_y`, in m/s.

The mapping to the existing dials:

- `knockdown_dv_threshold` becomes **q_fall = 245 Pa**, times **stability²**, so that `stability` keeps its current meaning as a multiplier on the knock-down wind speed.
- `k_push` no longer sets the wind's effect, and `push_grad_floor_atm` is moot for wind.

Whether to keep the gradient path as a separate knockdown for fast transients (blasts), with a much higher floor, is a modelling call for Erik. My lean is q alone, with a measurement owed: the blast-ring feel measured under the q law before the gradient path is retired.

## 4. Proposal per band (q at sea level; speeds at game ambient differ by about 1 %)

- **q < 39 Pa (v < 8 m/s):** no effect at all. There is no displacement and no slowdown.
- **39–120 Pa (8–14 m/s), slowed:** the marine stays upright. Max walk speed is scaled by s = 1 − 0.5·(q − 39)/(245 − 39), linear in q because it is proportional to the drag force. That gives s = 1 at 8 m/s and s = 0.80 at 14 m/s.
- **120–245 Pa (14–20 m/s), hunched:** the same law continues down to s = 0.5 at q_fall. The marine is in a braced crouch, so it cannot run and fires with the snap cone. Hunt's walking-speed table, which I could not verify, falls faster: about 0.5 already at 14 m/s, which is linear in v. Flagged for Erik.
- **q ≥ 245 Pa (v ≥ 20 m/s), knockdown:** KNOCKED_DOWN, with `knockdown_getup_seconds` re-applied while it lasts. An optional slide is allowed on decks with μ < 0.25.

## Sources

[1] Penwarden, A.D. (1973) "Acceptable wind speeds in towns", *Building Science* 8, 259–267; abstract at https://swov.nl/en/publicatie/acceptable-wind-speeds-towns
[2] Penwarden (1975), Beaufort table of effects on people, reproduced in Windtech report WA780-01F02 (2009), Table 2: https://apps.planningportal.nsw.gov.au/prweb/PRRestService/DocMgmt/v1/PublicDocuments/DATA-WORKATTACH-FILE%20PEC-DPE-EP-WORK%20PPS-2009SYE010!20190621T104252.017%20GMT
[3] Jordan, S.C. (2008) PhD thesis, Univ. Birmingham, §2.5–2.6, §4.5, Tables 5/6.7: https://etheses.bham.ac.uk/8029/2/Jordan_2008_PhD_01660012.pdf ; and Jordan, Johnson, Sterling & Baker (2008) *Building and Environment* 43(9) 1521–1534, doi:10.1016/j.buildenv.2007.08.004 (https://research.birmingham.ac.uk/en/publications/evaluating-and-modelling-the-response-of-an-individual-to-a-sudde)
[4] Lawson safety criteria (15 / 20 m/s, 0.022 %/yr), as applied in an EIAR wind chapter: https://www.pleanala.ie/publicaccess/EIAR-NIS/309026/3.%20EIAR/Chapter%2011-%20Wind%20and%20Microclimate.pdf
[5] Hunt, J.C.R., Poulton, E.C. & Mumford, J.C. (1976) "The effects of wind on people; new criteria based on wind tunnel experiments", *Building and Environment* 11, 15–28, doi:10.1016/0360-1323(76)90015-9 (full text not retrieved; numbers via [3])
[6] Melbourne, W.H. (1978) "Criteria for environmental wind conditions", *J. Wind Eng. Ind. Aerodyn.* 3, 241–249: https://www.scilit.net/publications/801f4ab07550bb73977cb1b1c5b04c22 (23 m/s from secondary planning reports)
[7] NEN 8100 danger criterion summary: https://www.simscale.com/blog/nen-8100-wind-comfort/
[8] Penwarden, Grigg & Rayment (1978) "Measurements of wind drag on people standing in a wind tunnel", *Building and Environment* 13(2) 75–84: https://structurae.net/fr/litterature/article-de-revue/measurements-of-wind-drag-on-people-standing-in-a-wind-tunnel (values via [3])
[9] Redfern et al. (2001) "Biomechanics of slips", *Ergonomics* 44(13) 1138–1166
[10] Beschorner, Albert & Redfern (2016) *Gait & Posture* 48, 256–260, doi:10.1016/j.gaitpost.2016.06.003 (https://doi.org/10.1016/j.gaitpost.2016.06.003)
