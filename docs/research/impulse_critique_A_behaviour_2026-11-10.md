# Critique A (behaviour and failure modes): docs/unit_impulse_design_2026-11-10.md draft 1

VERDICT: BUILD AFTER THE LISTED CHANGES. The §5 law has four BLOCKERs and must not be built as written. The fixes are small, keep every RULED item, and are stated in full at the end.
- B1: the reaction delay (step 2) switches off ALL resistance, including against the steady force. One impulse at or above v_inplace, plus a wind at 0.99 of the hold, slides a 100 kg marine about 23 m over about 46 s. That is #140 again.
- B2: `effort += (x - effort) >> 3` can never reach ONE, so at stability 1.0 the saturated-effort knockdown can never fire. A marine in a wind 5 % over the hold is not floored within 1/3 s. It slides 3 m (facing the wind) or 10 m (back to it) before v_fall floors it.
- B3: the same shooters at the same mean force give opposite results depending only on whether their shots land on the same tick. Five synchronised carbines on a 100 kg marine re-trigger step 2 every 2 ticks, so footing is off for good and the target is driven at 2.16 m/s^2. Spread over the ticks, the same five rounds move nothing.
- B4: nothing excludes a prone unit from step 2. Four synchronised carbines on a prone 80 kg body then push it with no resistance and no end (6.5 m/s after 3 s).
MAJORs:
- M1: a fully resisted shove knocks a stability-0.9 unit down through condition 2.
- M2: sub-hold impulsive fire makes a target creep (0.08 tile/s).
- M3: step 9 as worded chatters about the aim (±1.3°) forever. A discrete law that provably lands exactly is given below.
- M4: the "0.7° nudge" is wrong. The law gives 0°; physics gives 0.09°.
- M5: the lever arms are unclamped, and an outside spin above the top speed is left unspecified.
- M6: a pushed unit can still walk at full speed. Stagger leaves effort undefined.
- M7: a prone unit is posture-dependent (a smaller area), so in a steady over-hold wind it cycles between getting up and falling.
- M8: moves of more than 1 tile per tick are checked at the destination only. Fast pushes can clip corners and, at 32 m/s and up, pass through a wall.
- M9: three other facing writers (travel facing, the zombie walk, the attack facing snap) erase or bypass the inertia.
- M10: several §6 numbers do not follow.
MINORs: ten, listed at the end.

Numbers throughout: dt = 1/24 s, g = 9.81. The hold per tick is a velocity, cap = k_hold·g·dt = 0.22·9.81/24 = 0.0899 m/s (the same for every mass). The prone cap is 0.4·9.81/24 = 0.1635 m/s. The hold force is 215.8 N at 100 kg and 172.7 N at 80 kg. A carbine round is 3.6 N·s (the doc's), at 12 rounds/s, which is one round every 2 ticks. A tile is 0.333 m, so 1 m/s = 0.125 tile/tick. Own turning in tick units: a = 30/576 = 0.0521 rad/tick² and w_max = 0.2618 rad/tick (15°).

---

## BLOCKERs

### B1. The reaction delay disables the footing against everything (the #140 slide is back)
Step 4 is skipped entirely while `react_ticks > 0`. During those 7 ticks, every concurrent input lands unresisted, including a soft or hard steady wind and recoil. Once the delay ends, the cap is shared between cancelling the steady force and slowing `pv`. The net deceleration is then only (hold − F)/m, which goes to zero as F nears the hold.
- **Calculation.** A marine (100 kg) stands with its back to a wind of 0.99 hold, which is "hard to walk" and permitted by the ruling. One .50 round hits it from behind: 0.38 m/s, at or above v_inplace,fwd of 0.25, so react = 7.
  - The 7 unresisted wind ticks add 7·0.99·0.0899 = 0.62 m/s. So pv = 1.00 m/s, below v_fall,fwd of 1.5: no fall.
  - It then slows at 0.01·2.158 = 0.0216 m/s², which takes 46 s and covers 1.00²/(2·0.0216) = 23 m, about 70 tiles.
  - Throughout, used = cap, so effort pins at saturation. For a marine that never floors it (B2).
  - At 0.95 hold the slide is 4.6 m (14 tiles) over 9 s.
  - In patch b, any teammate shove (ruled legal) does the same in the 1.44 atm doorway.
- The doc's argument "a force below the hold adds less than cap, step 4 removes all of it" is false whenever react > 0. The §8 gate "any sustained force below the hold: position unchanged for any duration" fails as soon as any impulse crosses v_inplace.
- **Second face of the same bug.** A steady force whose per-tick dv reaches v_inplace re-triggers step 2 every tick. That needs F ≥ v_inplace·m/dt: 360 N facing it, 480 N side-on, 600 N with its back to it, i.e. 1.67, 2.22 and 2.78 holds. Footing then switches off for good. At 599 N (back to the force) the unit accelerates at the excess, (599 − 216)/100 = 3.8 m/s². At 601 N it accelerates at the whole force, 6.0 m/s². That breaks "displaces by its excess" with a jump in the law and makes it depend on the tick rate (at 48 Hz the edges double). In patch b, blast wind of 30–40 m/s (486–864 N) sits in this regime.
- **Smallest fix.** The reaction delay suspends only the RECOVERY of the stagger velocity, never the resistance to this tick's new input. Each tick the hold cancels new input first and works off pv only with what is left. See the in-place sway in the corrected law; it also fixes B3, B4 and M2.

### B2. The saturated-effort knockdown cannot fire at stability 1.0
With d = ONE − effort and input ONE, the update is d ← d − ⌊d/8⌋. Since ⌊d/8⌋ < d for every d ≥ 1, d never reaches 0. Effort stalls at ONE − 7 or above and never reaches ONE. The marine's stability is 1.0, so stab² = ONE, and condition 2 is unreachable for it, the reference unit.
- **Calculation (marine 100 kg, F = 1.05 hold, excess 10.8 N, 0.108 m/s²).** Facing the wind: v_fall,back of 0.8 is reached after 7.4 s and 3.0 m (9 tiles). Back to the wind: v_fall,fwd of 1.5 after 13.9 s and 10.4 m (31 tiles). The wind law the doc replaces floors the marine at once at q_fall.
- So these claims do not follow for the reference unit: "effort saturates within about a third of a second and the unit goes down", "identical bands for the reference marine", and the §7 patch-b gate.
- Rounding up alone is not enough. With demand exactly 1.0, a round-away filter reaches ONE only after about ln(2^16)/ln(8/7) ≈ 83 ticks (3.5 s).
- **Fix.**
  - (a) Low-pass the INPUT demand |dv|/cap, clipped at 2·ONE (not the clipped `used`), and round away from zero. Effort then tracks F/hold exactly, crosses ONE in finite time whenever F > hold, and reaches the band edges exactly.
  - (b) Add the unrecoverable-stagger fall: down when react = 0 and |pv| > N_rec·margin, where margin = what is left of the cap after the new input. At F > hold the margin is 0, so the first non-zero residual floors the unit.

### B3. The outcome depends on whether the shots land on the same tick
Rule 5 lands the rounds of every shooter in the same tick, so n carbines firing in phase deliver n·3.6 N·s per firing tick.
- **On a 100 kg marine, shot from the front** (backward push, v_inplace = 0.15 m/s):
  - n = 5 gives 0.18 m/s per firing tick, at or above 0.15. Step 2 fires every 2 ticks, react never drops below 5, step 4 never runs, and pv grows by 0.18 every 2 ticks (2.16 m/s²). The marine reaches v_fall,back after 9 ticks and keeps accelerating; then B4.
  - The same five shooters out of phase (3 + 2 rounds a tick): 0.108 and 0.072 m/s, both resisted almost entirely, with the unit pinned at hold force. It stands still and is never floored (B2).
- One mean force, opposite results, decided by a non-physical detail that players and bots can control. This is an exploit.
- **Fix:** the same as B1 (new input always meets the hold first; only the overflow beyond the in-place capacity staggers).

### B4. Prone units are not excluded from the sudden-push branch, so a prone body can be flung without end
Step 2 is not restricted to standing units. A prone 80 kg body under four in-phase carbines (0.18 m/s per firing tick, at or above 0.15) re-triggers react every 2 ticks, so `mu_slide` never acts. It accelerates at 2.16 m/s², reaching 6.5 m/s after 3 s, until it hits a wall. "Does it stop?" Not while the fire lasts.
- **Fix:** prone units have no sudden-push branch and no reaction delay (a body on the deck is held by friction at once). The prone hold acts every tick.

---

## MAJORs

### M1. Condition 2 floors units that are only working off a resisted impulse
`used = cap` while a shove is being worked off, the same signature as an over-hold force. Take a 100 N·s shove from behind on an 80 kg zombie at stability 0.9:
- dv = 1.25 m/s, under v_fall·0.9 = 1.35, so it should stagger (as the research says).
- After the 7 react ticks there are 13.9 saturated ticks. Effort after 13 of them is 1 − 0.875^13 = 0.824, at or above 0.81, so it is knocked down.
- Every impulse with dv ≥ 13·cap = 1.17 m/s floors a 0.9 unit, whatever its direction.
- At stability 0.8 (0.64), it takes 8 ticks, so 0.72 m/s.
- The fall limit is then set by the filter, not by v_fall.

Fix: B2(a). Effort reads the input demand, so a one-tick shove contributes at most 2·ONE/8 = 0.25.

### M2. Sub-hold impulsive fire makes the target creep
- **Carbines.** When an impulse arrives every other tick with cap < dv < 2·cap, the remainder dv − cap moves the unit for one tick of each pair. The mean force is under the hold.
  - On 100 kg: 3 in-phase carbines (0.6 hold) creep 0.009 m/s; 4 (0.8 hold) creep 0.027 m/s, which is 0.08 tile/s or 4.9 tiles a minute.
  - On 80 kg: 3 in-phase carbines (0.75 hold) creep 0.0225 m/s.
- **Bursts.** Burst weapons put several rounds in one tick, beyond what the §6 table covers. Taking 9 mm at 2.9 N·s, the MP-11's 4 rounds are 11.6 N·s in one tick. Two MP-11s in phase on 80 kg give 0.29 m/s per burst, which crosses v_inplace. Each burst then drifts about 0.09 m, so the target is pushed back about 0.7 tile/s against a mean force of 0.34 hold. Two shotguns in phase (about 0.28 m/s) are the same.
- The §8 gate is stated for a steady force only and would pass on all of this. The §6 rows (43 / 86 / 130 N) assume a steady force; they are right for effort and wrong for motion.
- **Fix:** the in-place sway (see the corrected law): new input is absorbed in place up to v_inplace and drained at the hold. Bursts below the in-place capacity then never move a unit. Gate: "any input train with every per-tick |dv| ≤ min v_inplace and mean ≤ hold: position unchanged".

### M3. Step 9 has no exact discrete form. The natural reading chatters forever.
The obvious "accelerate toward the speed that stops at the aim" is ω_des = sqrt(2·a·|e|), slewed by at most a. In discrete time that underestimates the braking distance: the true sum Σ(ω − k·a) = ω(ω + a)/(2a) is larger than ω²/2a.
- **Calculation from rest at e = 0.01 rad.** ω' = 0.0323, which overshoots to e = −0.022. Then ω' = −0.0198 (e = −0.0025), then −0.0161 (e = +0.0136), then +0.036 (e = −0.022), and so on: a cycle of about ±0.02 rad (±1.3°) that never ends. The doc's §8 gate "reaches the aim exactly, without overshoot" fails, and the aim visibly vibrates.
- **A law that provably lands exactly** (integer, per tick, in the facing's own currency):
  - Definitions:
    - a = the own acceleration per tick² and w = the top speed per tick.
    - B(ω) = Σ_{k≥0} max(ω − k·a, 0) is the distance covered braking from ω, this tick included. In closed form it is (n+1)ω − a·n(n+1)/2 with n = ⌈ω/a⌉ − 1, n ≤ ⌈w/a⌉ ≤ 6.
    - s(e) = the largest ω in [0, w] with B(ω) ≤ e. It is computed exactly by trying n = 0..6, with ω = ⌊(e + a·n(n+1)/2)/(n+1)⌋ valid if n·a < ω ≤ (n+1)·a.
  - Each tick: e = wrap(aim − facing); ω_t = sign(e)·s(|e|); ω' = clamp(ω_t, ω − a, ω + a); facing += ω'.
  - **Proof sketch.** Invariant: |ω| ≤ s(|e|) with the sign of e. B(ω') ≤ |e| implies ω' ≤ |e| (no overshoot) and B(ω' − a) ≤ |e| − ω'. So the next tick's lower clamp never forces a speed above s(e'), and the invariant holds by induction.
    - For e ≤ a, s(e) = e. Once |e| ≤ a, the next ω' equals e exactly (reachable, since |e − ω| ≤ a), which lands on the aim; the tick after, it equals 0.
    - From rest the invariant holds trivially. After an outside spin breaks it, the unit overshoots, reverses at a per tick, and the invariant is re-established the tick ω crosses to the side of e. So it converges exactly with no limit cycle (for a fixed aim; a moving aim gives a bounded lag).
- The gate should say "a fixed aim".

### M4. The "0.7° nudge" from a hit does not follow
- The carbine round at the body radius: 3.6·0.25/3.0 = 0.30 rad/s, which is 0.0125 rad/tick, below a = 0.052. Steps 3 and 9 cancel it in the same tick before `facing += omega`, so the nudge is exactly 0°.
- Physically it is ω²/2α = 0.3²/60 = 0.0015 rad, i.e. 0.09°. The 0.7° is one tick of 0.30 rad/s with the unit's own torque ignored.
- Steady carbine fire off-centre is 12 × 0.9 = 10.8 N·m against 90 N·m of own torque: also invisible.
- Only a .50 (about 6°) or a shove twists the aim. So Erik's "a hit on the shoulder twists an aim" is not delivered by small arms under real momentum.
- **Fix:** state the true outcome. If a visible twist from small arms is wanted, that is a modelling call for Erik: for example a short angular reaction delay (the own controller ignores outside spin for t_react_rot), or a declared multiplier on the bullet's angular impulse beside `bullet_push_mult`.

### M5. Lever arms and super-top-speed spin are unspecified
- The shove's lever is "the offset of the shover's centre line from the target's centre", with no clamp. For two adjacent 3×3 bodies that offset can reach about 0.5–1 m. At 1 m: 100 N·m·s / 3.0 = 33 rad/s (1.39 rad/tick, 5.3× the top speed), which spins the target about three full turns.
- Even clamped to 0.25 m, Δω = 8.3 rad/s (477°/s) is above the 360°/s top speed. The doc does not say whether "never above its top speed" clamps the total ω (an unbooked loss: the spin becomes 30° instead of 57°) or only what the unit's own torque can add.
- **Fix:**
  - Clamp every lever to the body radius (as for rounds).
  - The top speed limits the own torque only; an outside spin above it decays at a (0.25 m: 57° discrete).
  - Clamp |ω| below half a turn per tick so facing cannot alias. That otherwise happens for I < J·r/(π·24), e.g. 0.33 kg·m² for a shove, i.e. a small creature.
  - A door refuses m ≤ 0 and I ≤ 0.

### M6. Walking, and effort, during a stagger are unspecified
- While react > 0, step 4 is skipped, so `used` is undefined. Read as 0, effort decays during the stagger and the walking scale returns to ONE. A shoved unit can then run at 4.5 m/s while being pushed. A shove from behind on a runner adds 1 m/s for 7 ticks and costs it nothing.
- The research had a STAGGERED unit unable to walk. Erik ruled only that it can shoot.
- **Fix (modelling, Erik's call):** propose walking is suppressed while react > 0, and effort counts as saturated while |pv| > 0. Either way the law must say it.

### M7. In patch b a prone unit cycles between getting up and falling in a steady over-hold wind
- Prone, cd_area drops by 6×, so the wind's demand falls below stab². Effort decays, KNOCKED_DOWN expires after 36 ticks and the unit stands. Standing demand is over the hold again, effort needs about 13+ ticks to re-cross, and it falls again.
- That is a 36-down / about-13-up cycle. In every up-phase the unit is slowed and, if over the hold, pushed. The wind law it replaces refresh-stacks while W ≥ W_fall and stays down.
- **Fix:** while prone, the knockdown refresh judges the input as it would be standing (the wind's q·A_standing·dt/m against the standing cap). The prone area is used only for motion. And no get-up while |pv| > 0.

### M8. Collision checks the destination only, so fast pushes clip corners and can pass through walls
`is_passable_block` checks only where the footprint lands (simulation.py `_step_move_dir`).
- Above 1 tile per tick per axis (8 m/s), a moving footprint can clip a corner cell it never stands on: door jambs, pillar corners.
- At 2 tiles per axis (16 m/s per axis) it can skip a 1×1 pillar diagonally.
- At 4 tiles (32 m/s) it passes through a wall 1 tile thick.
- The walk (≤ 0.56 tile/tick) adds to the push.
- Patch-b blasts and a later blast impulse (the research's tertiary range runs to 16–42 m/s) reach these speeds, and §8's "a wall stops a pushed unit" is then false.
- "A blocked axis is zeroed" also loses momentum outside the §5 identity, and the existing X-before-Y slide makes mirrored maps behave differently.
- **Fix:** move walk + pv as ONE displacement, in substeps of at most ½ tile per axis, stopping at the first blocked substep. Zero, and book as a `wall` term in the identity, only the component of pv pointing into the blocked axis. Try X-only and Y-only in a symmetric order (for example the larger component first).

### M9. Other facing writers bypass the angular inertia
- `_step_move_dir`: `face_towards`, for a unit with no live aim.
- `ai_zombie.py:214`: `face_towards`.
- `timeline.py`.
- The attack resolve's facing snap (`SimultaneousEffects.face`).

All of these set the facing instantly. Any outside spin on a zombie or a non-aiming walker is overwritten the next tick, so zombies never feel the ruled spin, and the attack snap turns a marine in zero time. The doc must state that every facing target (aim, travel, attack) goes through step 9.

### M10. §6 numbers that do not follow from the law as written
- **"Five shooters, 216 N: at the hold; pushed back and goes down."** 216 N is the hold of a 100 kg marine. There, a steady 216 N does nothing and never floors it (B2). In phase, B3 flings it. Mixed phasing gives nothing.
  - For an 80 kg zombie at stability 0.9, FOUR shooters (173 N = 1.0 hold) already floor it, and so would anything over 0.81 hold (140 N, 3.3 shooters).
  - With the corrected law, five shooters floor the 100 kg marine and four floor the zombie. The table should say which target each row means.
- **"Firing the carbine: about 1 % slower".** Effort 43/215.8 = 0.199, so scale = 1 − 0.5·(0.199 − 0.164)/0.836 = 0.979: 2.1 % slower.
- **"One shooter: zombie about 4 % slower".** 0.249 gives 5.1 %. At the 13-tick shamble cadence, half_up turns that into 13 → 14 ticks, i.e. 7 %.
- **"Wind bands identical for the reference marine".** The hold 0.22·981 = 215.8 N corresponds to q = 239.8 Pa, not 245 (0.9·245 = 220.5 N), so the fall edge is 2 % low in q. Use k_hold = 0.2248 or say "within 2 %". The slow edge is 39.2 Pa, which is fine.
- **Shove from behind on 100 kg.** 0.29 m drift + 0.23 m slowing = 0.52 m over 18 ticks. That matches the claim. It no longer holds if the unit walks (M6).
- **Shove from the front.** Down: matches.

---

## MINORs
1. A single impulse with cap < dv < v_inplace still moves the unit: dv = 0.24 m/s moves 0.0066 m, then 0.003 m. The research's "below in-place nothing moves" is lost. The sway in the corrected law fixes it.
2. The fall test runs every tick against the current facing. A unit that turns toward its shover during the 7-tick drift moves its limit from 1.5 toward 1.1 and 0.8 and can fall late. Acceptable, but say so, or judge the direction at the impulse.
3. The door must refuse cap ≥ min(v_inplace). At 24 Hz, 0.0899 < 0.15 holds. At 12 Hz the cap would be 0.18, and steady sub-hold forces would trip the stagger.
4. "pv shrinks by used toward zero": a per-axis shrink resists a diagonal push √2× harder (a diagonal stagger is 0.71× as far). State the magnitude form: an integer isqrt, components rounded toward zero, and pv = 0 exactly when used = |pv|.
5. Mixed denominators for effort while prone (a prone cap, a standing cap): name one.
6. Passive motion by pv must ignore `can_move` (KNOCKED_DOWN, IMMOBILIZED, PARALYZED): say so.
7. Recoil of a .50 (0.38 m/s backward on the shooter) staggers the shooter 0.11–0.14 m every shot. Physically plausible standing, but the shooter expects it. A braced ×1.3–1.5 on v_inplace for own recoil (the research's [10]) still staggers, so state it as intended.
8. A hit integrated next tick (resolve at 4c, integration in the movement slot) is a 1-tick delay. Fine; state it.
9. **The patch split.** In patch a the wind's scale (field) and the effort scale (unit) MULTIPLY, and the two knockdowns do not add.
   - Example: wind at 0.5 hold with three shooters (0.6 hold). Patch a gives 0.80 × 0.74 = 0.59 and stays standing. Patch b gives effort 1.1, a 0.5 floor and down.
   - The pins of patch a measure the product. Patch b must delete the wind's scale member in the same commit that wind enters `add`, or 0.5 hold double-counts (0.64 instead of 0.80). Make that a patch-b gate.
10. A wind that alternates direction above the cap (±1.5·cap) ratchets one way under the §5 law. The remainder of the first phase eats part of the second, giving +0.5·cap per 2 ticks of drift from a zero-mean force, and nothing floors a marine (B2). The sway and input-demand fixes remove it: the sway does not move the unit, and a demand of 1.5 floors it.

---

## The corrected per-tick law (keeps every ruling; adds one 2-vector, `sw`)

**Standing.**
1. Δv = J/m and Δω = Σ(J·min(r, R_body))/I, with m > 0 and I > 0 refused at the door. Clear the accumulators.
2. demand = min(|Δv|/cap, 2·ONE). Then effort += (demand − effort)/8, rounded away from zero, so it reaches its input exactly.
3. The in-place sway: sw += Δv.
   - If |sw| > v_inplace(direction of sw), the overflow sw − v_inplace·ŝw moves into pv, sw keeps v_inplace·ŝw, and react = t_react.
   - The door refuses cap ≥ min v_inplace.
4. The budget is cap.
   - sw shrinks toward 0 by min(|sw|, budget) EVERY tick (react or not); margin = what is left of the budget.
   - If react = 0, pv shrinks by min(|pv|, margin). Otherwise react −= 1.
   - Book everything resisted.
5. Walking scale from effort, as in the doc (ONE below k_slow/k_hold, linear to 0.5 at ONE, clamped). Walking is suppressed while react > 0 (Erik's call, M6).
6. KNOCKED_DOWN if any of:
   - |pv + sw| ≥ v_fall(direction)·stability;
   - effort ≥ stability² (the wind law's pinned integer compare);
   - react = 0 and |pv| > N_rec·margin, with N_rec of about 24 ticks (unrecoverable at the margin left: this is what floors a stability-1.0 unit at F > hold, and a shoved unit in a near-hold wind).
7. Move by pv only (the sway never moves the unit), together with the walk, in substeps of at most ½ tile per axis. Stop at the first blocked substep, and zero and book only pv's component into the wall.
8. Turn with the s(e) law of M3 (every facing writer goes through it). The top speed bounds the own torque only, and |ω| is clamped below half a turn per tick.

**Prone.** No sway capacity beyond one prone cap, no react. All input meets cap_prone = mu_slide·g·dt first. The knockdown is refreshed while the standing-equivalent demand is at least stability², or while |pv| > 0, so no get-up while sliding.

**What this gives.**
- Any input train whose per-tick |Δv| stays within v_inplace with mean below the hold moves nobody, by construction.
- Effort equals F/hold for a steady force, so the wind bands are kept exactly.
- An over-hold force floors a unit of any stability within ticks, and displaces it by its excess only while the excess beats the hold it then has.
- Shove outcomes are as in §6 (100 N·s from behind on 100 kg: 0.44 m over 18 ticks).
- In-phase and out-of-phase fire agree.
