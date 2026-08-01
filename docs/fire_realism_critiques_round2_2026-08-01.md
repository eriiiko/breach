# Fire realism design — adversarial critique, ROUND 2 verdicts (2026-08-01)

Panel of four Opus-class critics on `fire_realism_design_2026-08-01.md` **v2**.
Round-2 outcome: **v2 not blessable — the corrected keystone algebra CONVERGED
(two lenses derived the identical fix independently), and the panel discovered
the design's true missing keystone: THE AIR-SUPPLY / ENTRAINMENT CHANNEL.**
Dispositions marked for v3.

## The two round-2 theorems (cross-lens convergence)

**T1 — the corrected emitter books (L2-B1 ≡ L1-R2-1, independently derived,
algebraically identical):** per absorbing cell r on a ray from s,
```
rad_net[r] += pair            pair   = a_s·a_r·τ·w·(E°[T_s] − E°[T_r])
rad_net[s] += credit          credit = a_s·a_r·τ·w·(E°[T_r] − E°[0])   // REPLACES the emitter's pair debit
```
(equivalently: keep `s −= pair` and credit `a_s·a_r·τ·w·(E°[T_s] − E°[0])` — same
total). Limits: ambient scenery ⇒ exactly the open-air sink (Erik's equivalence);
equal-T full view ⇒ zero net loss; partial view ⇒ loss through uncovered
directions only. v2's form (pair debit AND E_r-credit) double-charged one pair
per deposit — up to 2× gross emission beside ambient walls — and both of v2's
proof-cases sat in the error's blind spots. MANDATORY EXCLUSION (L1 R2-2): skip
the credit at the source's own cell (distance 0), else a_s² of the sink refunds
to self (100% for ε=1 materials). Contract note: per-pair antisymmetry is
retired; conservation = the ledger identity, accumulated at separate sites.

**T2 — the air-supply keystone (L1 R2-7/8/9, corroborated by L3-1):** measured
from `tune_r5_lone_wd020.csv`: diffusive O₂ supply to one tile ≈ 0.105
units·s⁻¹ per unit X-depression, depression capped at 0.08 ⇒ **sustainable
delivery ≤ ~27 kW (40 kW only at extinction)**. Honest loss books at the
blessed plateau (sink 45–68 kW + h-anchored convection 15–29 kW) demand
**60–97 kW** with χ_bed ≤ 1. ⇒ NEITHER package closes: GAME cannot spread fire
by any mechanism (radiation 24× under the 12 kW/m² threshold at 1.5 kW books —
physical, not artifact; conduction exactly 0 in-engine for furniture κ=0, ~122 W
for wood vs ~10 kW needed); REAL cannot breathe (delivers ~5× not 26×, halving
I toward the knee). The missing channel is **entrainment** (2D no-gravity: no
buoyant inflow; the plume pushes air OUT). Q-0 is therefore contingent on a
supply-channel design, and the cheap pre-measurements are: (i) the supply bench
(max sustainable O₂ to one burning tile at X_ring ≥ 0.17, still air, F4 room
sweep), (ii) the χ_bed arithmetic check `H_bed·(J/count)/4.83 MJ ∈ [0.3, 1.0]`.
Counter-checks that keep REAL alive: at 127 kW the honest plateau lands 1140–
1310 K (the flame band, no new dial); fuel-energy of a 6–8 min burn ≈ 50 MJ ≈
3 kg wood (real crate 10–30 kg) vs GAME's 120 g. REAL is right if it can breathe.

## L1 (realism) — verdict abridged
Keystone idea right; v2 algebra fails in the limit it was written for (R2-1),
self-credit (R2-2), T_emit_gate cliff MOVED to the credit and bistable (R2-3 —
fix: per-tile covered-weight refund g_i or ungated credit-cast; add the
just-below/just-above gate), sink must share the emission entity φ·E°[T+lift]
with pair/credit (R2-4), the reused pair budget binds INSIDE the operating band
flattening T⁴→T at ~470–540 game (R2-5 — sink gets its own limiter + printed
first-bind temperature per material), touching solids double-channel
conduction+radiation (R2-6b — name the lump or suppress pair where face_shift
exists), mutual feeding CANNOT emerge (hot saturated + O₂ competition ⇒ net
negative; P3 → accepted gap until the ṁ″∝flux/L_v channel exists — R2-10),
air cell = 0.34 m implied ceiling (state where smother timescales are sold —
R2-11), F-BO scaling lacks fire size (W_crit ∝ knee margin not Heskestad L∝Q^0.4
— demote anchor 2 to a falsification test with the ratio prediction written;
no |W|↔m/s conversion exists yet — R2-12), rad_scale is DERIVED (≡ σ·A·dt /
J-per-count with the J anchor from the chemistry side; step 2 authors nothing;
12 kW/m² becomes a step-4 prediction — R2-13), step 4→2 back-arrow: declare
T_flame_ref a Q-0 output + two-pass fixed point (R2-14), ε-retune is a MARCH
change (heat_atten is also occlusion; steel ε 0.3 = 70% heat-transparent —
digest-moving patch §5 must list — R2-15c), **F3's e_abs bypass re-arms the
decisions-#16 runaway and convects into vacuum — fix: density factor
`q_f *= min(1, N_j)`, which also retires cool_shift_vacuum cleanly (R2-16)**,
c_v-as-convention untenable once F3 owns room heat (gas lumped 37× vs solid
140× ⇒ rooms warm in ~12 s not ~7 min; acceptance must gate the TIME CONSTANT,
not just the steady value — R2-17).

## L2 (determinism) — verdict abridged
B1 = T1. B2: the sink cannot live in the temperature solver as spec'd (rad_net
is const; CUDA never copies it back) — apply as direct dT in Pass 1 (int64,
budget, shr_round0, sat_add + rails) with `rad_amb_flux` returned as ull like
t_max_phys_hits; CPU ledger through uint64 (signed overflow UB). B3: **the real
ceiling is E°'s int32 saturation at ~1768 game** (9× below T_MAX_PHYS; zero
exchange between two tiles both above it; uncounted) — widen table to int64
(preferred under REAL) or hard load-check + counted rail. B4: F-BO exact
no-divide form `inv_c_eff = mul_q16(INV_C, FP_ONE + mul_q16(k_strain_q, W))` +
STRAIN_MAX_Q clamp; drop the linearised fallback; byte-identical at k=0 = free
regression oracle. B5: `range_base`/`range_per_intensity` became ENERGY-BOOKS
dials (max_range 2.46 tiles ⇒ a 4-tile room refunds nothing — F4's cavity
prediction impossible; range∝I is an undesigned positive feedback) — add to F8
step 2 as companions + the shrinking-room rad_amb_flux gate. M6: per-term
budgets don't compose convex — clamp the CONVERTED aggregate once at the fold
(counter + site). M7: the ledger gate is near-tautological — state what it
detects (int32 wrap; clamp/booking mismatch), accumulate the two sides at
separate sites. M8: gate (iii) needs a nullable `rad_pair_dbg` plane (rad_flux
idiom); gate (i)'s ε must add the geometric defect (1−Π(1−a) shortfall ≈ 0.1%
of sink at ε=0.9 ≈ 4% of cool_shift-9) or pin to an a=1.0 geometry. M9:
opaque-termination refund must credit the FULL direction share (a_s·τ·w) to
close the cavity — else sealed low-ε rooms shred energy; needs a solid-mask
plane in both marches (heat-touched-set change — gate it). M10: bake
`sink_coef_q[i] = quantize(a_i·Ω)` per material (keep the solver integer; no
float H2D). M11: knee load-check = WARN at P-M1, ERROR at P-R5. M12: state the
plane constraint (every new plane per-tick-wiped & digest-excluded; persistent
state forfeits the freeze) + an ARC-LOCAL golden re-baselined per patch
alongside the frozen canonical set. m13 ull wrapping ledger; m14 Pass-1
order pinning (sink/fold/deposit/LOW/HIGH) + LOW rail value 0 + counter, carried
by the budget argument (sink+8 pairs ≤ 9T/16); m15 inflow needs NO dot/sqrt
(axis component, negate+max — drop the sqrt citation, it would resurrect the
plume artifact); m16 φ/lift into the bake cache key or per-source multiply;
m17 CONV budget engages at n_floor (counter + gate — a feel dial in disguise
otherwise); m18 dem_acc stale-debt reset on the two skip paths (SHIPPED BUG,
one line); m19 mt19937 = dormant cross-machine desync landmine (delete the
distribution); m20 credit costs ZERO extra atomics (same address as the retired
debit) but the global ledger atomic would serialise — per-tile plane + host
int64 reduce; m21 REAL's n_floor re-derivation must target the H_bed int64→int32
clamp path (ALREADY FIRES today on full-drain cells — SHIPPED BUG-ADJACENT);
m22 substep caps (n_smoke, N_SUB) must be ASSERTED not recorded (caps = silent
CFL violations) and the rebase taken only after dials settle; m23 F5′ takes the
max on the INTEGER side.

## L3 (intent) — verdict abridged
Q-0 not a fork (GAME infeasible vs the doc's own 12 kW/m² anchor — 10 kW needed
at one face from 1.5 kW total; REAL reproduces the anchor exactly at 13 kW/m² —
the unmade argument); GAME also breaks the cluster acceptance (contact spread
measured zero); REAL must be framed as COMPLETING D1 (×10 was an uncited hack
Erik declined; ×25-with-Babrauskas is the "literature act" §4.1 sanctions) and
REAL RETIRES the D1 accumulator (~23 counts/tick demand) while GAME keeps it —
the cost asymmetry v2 told backwards. (e-v) must not legislate [40,60] — the
blessed incumbent measures 61.7%; hard-gate the SHAPE (death-by-knee), give the
fraction to Erik with the incumbent stated. (e-vi/vii) scenarios undefined
("pre-heated" is not the ruling's still-air pair — define layouts in F4; state
the separatrix form: same dials, both outcomes). Wind-crossing acceptance still
orphaned (F-BO pushes AGAINST it; third anchor + P-F3 gate + the ruling's
escape clause verbatim). cool_shift knee lever silently spent (one sentence
retiring the LEVER RANGE amendment + escalation). M9 ML/zombie line: promote to
an owned acceptance with a criterion + §6 Q. REAL's sealed-room determinism
needs the O₂-axis variance matrix (volume × leak × fuel; both smother AND
fuel-exhaustion outcomes). Calendar: the golden rebase is SPENT (D2) — declare
a NEW ARC with its own allowance; Erik touchpoints ≥3 working sessions + ~5
play tests (table them); ordering: wind anchors needed at P-F3 not P-R5;
T_emit_gate's A1.8 rationale invalidated — restate what 180 now buys for
re-ruling; watt table must cite run+tick (steady_T 385 vs the table's 440 —
headline ratio 9×→~4.5×); "5–13 s spread" relabeled bench-measured-never-
human-tested, and "is seconds-scale radiative spread the feel you want?"
becomes an explicit Q-0 sub-question; package B ≡ REAL mapping line; filename
xref fix; acknowledge dem_acc + D1b hysteresis as the shipped sustain
semantics the knee gates rest on.

## L4 (simplicity) — verdict abridged
§9's "net ≤ 0" false on its own lists (honest: +4..+5 on the recommended path;
Retunes row missing ~12–15 re-authored values) — claim the true "+2..+3 with
P1–P5+P9 fixed". Credit ledger line `rad_amb_flux −= credit` (superseded by
T1's form but the ledger-balance requirement stands). MQH deleted (no buoyancy
in 2D; depends on the deferred A√H law; unstated constants) → variance matrix +
ΔT bands + (per R2-17) the time constant. Knee check: monotone table-search
formula (3-term balance under F1), β fixed number, flammables-only scope,
warn→error schedule. GAME's φ/lift sizing over-determined (delete the second
clause — moot if Q-0 resolves to REAL-or-gas-spread). w_gain unanchored/unowned
→ F8 step-7 bench + one stated anchor, or delete and let the clamp shape it.
WIND_MULT_MAX_Q needs its stability derivation or reclassification. Ω ≡ Σw = 1
pinned in words (the "sphere" wording invites a 4× error). Under REAL delete
F1(c) entirely (φ≡1, lift≡0 — F1 returns to zero-dial). Steel opacity change =
a law bullet with its own gate, not an F8 parenthetical. P-F1a acceptance in
the P-R2 sentence form with the predicted magnitude (bench fires extinguish;
named red set; restoration point). Units: replace the 2× warning with the
identity `ignition_temp_game ≈ °C_literature − 10`. Validity bound in shipped
columns (hp within 2× of 60; thermal_mass within one step of 8); cellulosic
band ≡ the lone-crate band; enumerate the rails. Moving-parts honesty: sink/
credit = 7 parts vs residual's 4 — sell as "+3 parts, 4 blockers closed".
[Q-0] fork: one marker, losing package → one rejected-alternative paragraph,
"v3 deletes the loser" clause. Docs untracked while §5 plans worktree agents —
commit at blessing. Arc-allowance line (rebase spent). Critique-file must carry
the inventories it indexes (this file does).
