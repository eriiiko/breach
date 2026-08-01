# Fire realism design — adversarial critique, ROUND 1 verdicts (2026-08-01)

Panel of four independent Opus-class critics on `fire_realism_design_2026-08-01.md` v1.
Full per-finding reports live in this session's transcript; this file preserves each
lens's verbatim VERDICT plus the finding index with v2 dispositions. Round-1 outcome:
**v1 not blessable — 14 distinct blockers across lenses; three structural problems
found independently by 3+ lenses each.** v2 dispositions are marked [ADOPTED],
[ADOPTED-MOD] (adopted with modification), [FORKED→Q-0] (depends on Erik's package
ruling), [DEFERRED] (named gap), [REJECTED+reason].

## Triple-confirmed structural findings (the load-bearing three)

1. **The energy books do not close** (L1 A-2/A-3 · L2 #18 · L3 B1/B2 · L4 #12/#14):
   F1's ambient debit at the blessed plateau is 44.5 kW vs 4.85 kW of chemical power
   (17× the H_bed deposit). cool_shift's whole budget is ~1 game/tick vs 38–69 of new
   freight — calibration step 2 has NO solution; H_bed would need 17–25× (outside
   E1's band); with H_bed raised, T* ∝ P^(1/3) flattens the thermal knee below I_min
   and the blessed part-burn (63%) becomes ~89% burn-out — Erik's campfire destroyed
   either way. → v2: the Q-0 anchoring fork + the sink/credit reformulation + F1+F2
   merged as one keystone. [ADOPTED-MOD]
2. **The ember is not a state of the law** (L1 E-1/E-2 · L2 #13 · L3 M5 · L4 #3/#10):
   not a fixed point (relights or needs a shim clamp), never consumes fuel (absorbing
   state, eternal embers, ML-degenerate), and real smolder is a SECOND COMBUSTION
   MODE (char oxidation, consumes mass, often hotter than flaming extinction).
   → v2: F6 deferred to its own mini-design; I_min snap stays. [DEFERRED]
3. **F3's wind is the self-plume again** (L1 C-1 · L4 #7; mechanism critique L1 C-2):
   |W| at the burning cell IS the plume — the k_wind_strip artifact reborn; and
   blow-out is strain/Damköhler physics, not bed cooling (wrong mechanism, wrong
   timescale, razor-thin window under T⁴). → v2: per-face signed INFLOW wind;
   blow-out re-sited onto a capacity-strain term with velocity anchors; q_conv keeps
   only honest jobs (object↔gas exchange, pre-heat, room warming). [ADOPTED-MOD]

## L1 — fire-science realism — VERDICT (verbatim)

> The doc is well-organized and its central instinct — that radiation must have a
> counterparty — is correct; the F1 telescoping algebra genuinely closes […] But the
> doc treats F1 as a bookkeeping repair when it is an order-of-magnitude re-anchoring
> of the entire thermal model. At the shipped constants an ambient-counterparty sink
> charges a burning crate **44.5 kW against a 4.85 kW chemical fire** — 17× the
> `H_bed` deposit — which extinguishes the blessed burn outright (T* falls to ~142
> game, below `fire_T_ext`), makes F8 step 2 arithmetically infeasible, fires E1 by
> 8×, changes the plateau law from T ∝ P to T ∝ P^(1/3) and thereby deletes the very
> thermal knee Erik blessed as the criticality separatrix, and forces a decision the
> doc never names: whether a burning tile's temperature is the flame or the fuel
> surface. That decision makes F2 non-deferrable and puts `burn_rate` and an
> emitting-area term into a calibration DAG that lists neither. On top of that sit
> three concrete code-level breaks (the residual has no flux limiter and voids the
> "no LOW rail needed" invariant and the `rad_net` overflow bound; F3 reuses the same
> plume-sampled W that made `k_wind_strip` a self-blow-out artifact; F5's `H_fuel`
> raise re-opens the decisions-#16 N-collapse runaway on a path that never got the
> `e_abs` guard), one law-level break (F6's ember is not a fixed point of the law and
> can only exist as a clamp), and a taxonomy that cannot express the two properties —
> heat of gasification and surface-area-to-volume — that actually separate its own
> classes. None of these are reasons to abandon the design; they are reasons v2 must
> do the energy arithmetic in watts before it does anything else, and must decide
> what a tile's temperature *is*.

Key numbers now canonical: 1 heat count ≡ 1.968e-4 J; furniture effective C = 51.6
J/K (the ~130–143× δ-layer lump, confirmed); chemical power at the blessed operating
point 4.85 kW; H_bed deposit 2.61 kW (54% of Huggett — defensible bed+radiant share);
flame-read radiative output 44.5 kW (χ_rad 0.35 of a 127 kW real crate fire).

Finding index: A-1 equivalence theorem VERIFIED · A-2 17× imbalance [ADOPTED→Q-0]
· A-3 flame-vs-surface trilemma [ADOPTED→Q-0, F1+F2 merged] · A-4 knee flattens
[ADOPTED: knee re-derivation + gate] · A-5 residual unlimited/low-rail/overflow
[ADOPTED: budget_s + low rail + bound re-derivation] · A-6 gate untestable as
written [ADOPTED: pinned single-tick, ε=(n+1)/2] · A-7 T_emit_gate cliff [ADOPTED:
sink/credit reformulation makes the sink ungated + continuous] · A-8 mutual feeding
inert at hot=1 [ADOPTED: named; intensity feedback = radiative-feedback channel,
Q-0 package B] · B-1 vacuum handling [ADOPTED: ambient counterparty everywhere;
cool_shift_vacuum re-justified] · C-1 plume W [ADOPTED: signed inflow] · C-2 blow-out
mechanism [ADOPTED: strain term on capacity] · C-3 conv/cool degeneracy [ADOPTED:
h-anchored split] · C-4 e_abs guard on conv [ADOPTED: routed around, named] · C-5
linearization label [ADOPTED] · D-1 χ ledger ≤1 + 1600× not 4500× [ADOPTED via F5
deletion] · D-2 N-collapse runaway [ADOPTED as prerequisite gate] · D-3 damage-path
claim false [ADOPTED: honest new consumer] · D-4 gas radiatively inert [NAMED gap;
package-B follow-up] · D-5 far-field target wrong [ADOPTED: Q-D re-scoped] · E-1/E-2
ember [DEFERRED] · F-1..F-6 missing channels [NAMED in §2; F-1 feeds Q-0] · G-1
classes directionally right ✓ · G-2 fuel_per_o2 → class 3 [ADOPTED] · G-3 L_v +
geometry columns [ADOPTED as named lumps] · G-4 dual fuel-drain authoring [NAMED]
· G-5 glass ε [ADOPTED: heat_atten→~0.9, light_atten carries visible] · G-6
cool_shift vestigial [ADOPTED: re-scoped comment] · H-1 payload accelerant-deposit
[NAMED future form] · H-2 DAG incomplete [ADOPTED: rebuilt] · H-3 cavity effect ✓
[kept as F4 prediction] · H-4 OOB termination [MOOT under sink/credit form] · H-5
543-peak promoted [ADOPTED].

## L2 — determinism / Q16.16 / CUDA / cost — VERDICT (verbatim)

> The keystone (F1) is the right physics and its cost story is genuinely cheap —
> cheaper than the doc claims — but it is not yet buildable: as specified, "at ray
> termination" re-couples a synced integer field to the render-side `expf` path and
> breaks tol 0 (finding 1), it has no limiter (2), and it silently retires the two
> conservation gates that made P-R4 trustworthy without replacing them (3). All
> three fixes are small and local; none require re-architecting. F3 is the weakest
> section by a wide margin: it is specified against the wrong plane (`heat[]` is
> positive-saturating, so the pre-heat direction — the entire blow-out-and-rekindle
> story — is dropped while the donor is debited, finding 7), and its stability claim
> ("like conduction's") does not survive contact with two ends that convert
> differently and a gas end whose inverse heat capacity is unbounded at
> `n_floor_heat` (8). F6's ember is one word away from a perpetual-motion machine
> and one sentence away from being free (13/14). F5's headline number is
> dimensionally inconsistent with its own anchor by an order of magnitude, its
> energy-parity target is unrepresentable in Q16.16, and its ΔT-parity target rails
> `T_MAX_PHYS` in a single tick in precisely the sealed-room scenario it exists to
> create, while violating a documented `n_floor_heat` invariant by ~5× (15/16). Most
> consequential of all: F8's step 2 cannot be satisfied — `cool_shift`'s entire loss
> budget is ~1 game/tick against a new radiative freight of 38-69, so the DAG's
> claimed independence of steps 2 and 3 is arithmetically false (18), and once F3
> and F5 exist the DAG has a cycle (19). The §5 plan additionally moves digest
> fields on four separate patches with no re-baseline point and no rationale budget,
> against the iron rule (21).

Finding index: #1 expf coupling [MOOT under sink/credit form — no termination term]
· #2 limiter [ADOPTED] · #3 ledger gate Σrad_net+rad_amb_flux==0 [ADOPTED] · #4
gate well-posedness [ADOPTED] · #5 e_table[0] convention + host-fold [ADOPTED] · #6
cost clean ✓ · #7 signed plane for q_conv [ADOPTED] · #8 both-ends budget, floored
N, wind clamp, own pass [ADOPTED] · #9 fold-order/symmetric-narrow constraints
[ADOPTED] · #10 wind_fan idiom not buckets [ADOPTED] · #11 order-free wording
[ADOPTED] · #12 zero-wind gate unachievable [ADOPTED via DAG rebuild] · #13 ember
pin illegal [DEFERRED with F6] · #14 ember cost [DEFERRED] · #15 H_fuel
unrepresentable [MOOT: F5 deleted] · #16 T_MAX_PHYS/n_floor_heat [carried to Q-0
package B prerequisites] · #17 EOS/smoke CFL cascade [carried: package-B budget
gate] · #18 DAG unsatisfiable [ADOPTED: rebuilt watts-first] · #19 DAG cycle/omits
[ADOPTED] · #20 P-F4 first [ADOPTED] · #21 golden strategy [ADOPTED: freeze + ONE
close-rebase] · #22 gate re-run budget [ADOPTED] · #23 CUDA twins named [ADOPTED]
· #24 vacuum mask cost [superseded by B-1 disposition] · #25 cool_shift
triple-booking [ADOPTED: sky_exchange comment reconciled] · #26 mt19937-per-source
waste [NOTED as free orthogonal win].

## L3 — design-intent coherence — VERDICT (verbatim, abridged to the ruling)

> The doc's diagnosis is excellent and its keystone is real […] But **as written,
> v1 un-does the one behavior Erik explicitly blessed.** F1 swaps a linear loss for
> a quartic one ~20–30× larger at the operating point, which (a) flattens `T(I)`
> until the thermal knee falls below `I_min` — turning "dies part-burnt at 63%"
> into "burns to ~89%", the opposite side of Erik's edge; (b) makes step 2 of the
> calibration DAG unsolvable and forces `H_bed` through escalation trigger E1
> without saying so. The fixes that should *widen* the interesting region (F3 wind,
> F4 scenarios) are the doc's best work; the ones that *narrow* it (F5 guaranteed
> flashover, F6 absorbing-state embers) are under-specified in exactly the
> direction that makes rounds identical — against the ledger's ML premise. […]
> **One consolidation would repair B1, B2, B3, M1 and M6 at once**: make the ray
> residual two-sided into the local gas rather than a 293 K void […] the honest
> counterparty is the **soot-laden** gas, whose emissivity the smoke plane already
> carries. That is his call, not the doc's. **Recommendation: do not bless v1.**

Finding index: B1 blessed-burn destroyed [ADOPTED→Q-0 + knee gate] · B2 E1 tripped
unflagged [ADOPTED: E1 budgeted explicitly] · B3 H_fuel ruling violation [ADOPTED:
F5 deleted; c_v ruled a unit convention] · M1 DAG cycle [ADOPTED] · M2 knee levers
named (Δ, span; not wall_damage/c) [ADOPTED] · M3 cluster burn-out gate missing
[ADOPTED] · M4 wind supply-channel half + k_wind_fan retirement [ADOPTED] · M5
ember absorbing state [DEFERRED with F6] · M6 flashover matrix + damage-path claim
[ADOPTED] · M7 per-material double-count, ε retune precedes H_bed [ADOPTED: DAG
step] · M8 payloads inert — retest post-keystone [ADOPTED] · M9 ML/zombie fire-
dominance acceptance line [ADOPTED] · m1 amendment-6 restoration [ADOPTED] · m2
space rays subtract nothing [superseded by sink/credit form] · m3 far-field target
re-scoped [ADOPTED] · m4 doors both rows + airlock bench [ADOPTED] · m5 extinction-
temp citation [ADOPTED] · m6 isolation probe + rebase budget [ADOPTED] · m7 63%
outside ballpark + 543 peak [ADOPTED: targets table] · m8 exploding containers gap
[ADOPTED]. The soot-gas counterparty proposal: [FORKED→Q-0 package B follow-up —
requires gas radiative participation, D-4].

## L4 — simplicity / scope / extensibility — VERDICT (verbatim, abridged)

> The keystone is sound and the diagnosis is excellent […] But **the document does
> not obey its own §2 rule.** Three of its eight fixes (F3, F5, F6) introduce terms
> with no stated reference, dials with more than one job, and magnitudes with no
> oracle — the precise trio the P-R1..P-R4 arc spent a week deleting, arriving
> under new names. The tell is arithmetic: that arc's ledger was 7 constants dead,
> ~5 born, each with a written job. This design's ledger is **28 authored numbers
> born, 1 dead.** […] Fold `q_conv` and `χ_gas` into one owned channel and F5
> disappears into the P-R5 session where it always belonged. Read wind as
> **per-face signed inflow** […] Derive `I_smolder ≡ I_crit` […] Make `conv_shift`
> global […] Add the thermal-knee load check and §4 becomes the strongest thing in
> the document. **That is a v2 with more capability and fewer constants than v1.**
> **Honest count: of the 28 new numbers this doc would have someone author, I would
> ship 2** […] a **net-zero dial ledger for a design that fixes P1, P2, P3, P4, P5
> and P9.**

Finding index: #1 dial ledger [ADOPTED: v2 carries a born/dies table, net ≤ 0] ·
#2 conv_shift global [ADOPTED] · #3 I_smolder ≡ I_crit derived [carried into the F6
mini-design] · #4 c_v unit convention [ADOPTED] · #5 F1 zero-dial ✓ · #6 q_conv
owns the channel, F5(a) deleted [ADOPTED] · #7 signed inflow [ADOPTED] · #8 w_gain
= lumped flame-strain, velocity-anchored [ADOPTED-MOD: strain moves to capacity
term; velocity anchors kept] · #9 k_wind_fan deleted [ADOPTED] · #10 F6 defer
[ADOPTED] · #11 Q-A zero-dial close [ADOPTED via sink/credit form] · #12 F5
unimplementable [ADOPTED: deleted; MQH observable kept] · #13 damage consumer
honest [ADOPTED] · #14 step-2 pin protocol + ownership split [ADOPTED] · #15
conservation-gate restatement [ADOPTED] · #16 signed plane [ADOPTED] · #17
hand-wave inventory [ADOPTED: v2 fills or marks each] · #18 F5 = tuning [ADOPTED]
· #19 protocol validity bounds [ADOPTED] · #20 patch/HUMAN-TEST/golden resize
[ADOPTED] · #21 classes: cellulosic+none only; anchors stay anchored; sorting rule
[ADOPTED] · #22 foam/steel/jerrycan breakdowns incl. units ~2× literature +
steel opacity-vs-ε fix [ADOPTED] · #23 I_crit load check [ADOPTED] · #24 H_fuel
"anchored-shape-not-value" honesty [ADOPTED] · #25 I_min/max_fire_thresh coupling
[carried to F6 mini-design] · #26 citations [ADOPTED].
