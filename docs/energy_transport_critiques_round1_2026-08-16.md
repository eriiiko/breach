# Energy-books design — adversarial critique, ROUND 1 verdicts (2026-08-16)

Panel of four independent lenses on `energy_transport_design_2026-08-16.md` v1
(committed `bda08eb`). Full per-finding reports live in this session's
transcript; this file preserves each lens's verbatim VERDICT plus a finding
index carrying the load-bearing numbers. Round-1 outcome: **v1 not blessable —
4 distinct blockers; two structural holes found independently by 3+ lenses
each.** Dispositions marked here are PROPOSED by the orchestrator; items
tagged [→ERIK] await Erik's ruling before v2 is written. Panel scope honored
§8 ACCEPTED GAPS (none re-litigated; two bounds attacked, both legally).

## Triple-confirmed structural findings (the load-bearing two)

1. **The ts-face energy law is unspecified, and the natural reading re-mints
   bigger than the leak being killed** (L1-1 · L2-1 · L3-1): permeable
   `thermal_solid` tiles (crates, perm 0.5) hold real bulk gas and act as flux
   donors, but `temperature[]` there is the OBJECT's T (ruling A1) — pricing
   `φ_e = φ·T[donor]` at a burning crate exports ≈ **90 eth/s per face vs
   39 eth/s for the entire +7,805 leak**, debited from nothing; the :423
   free-energy channel reborn on the mass path. → v2 must write the face law
   explicitly. Candidates: (d) no energy flux through ts faces — arriving gas
   priced at receiver's own T, counted signed residual (simplest honest);
   (c) honest object↔gas convective exchange via the heat plane / Pass-1 fold
   (physically richest, reuses deposit machinery, no new synced state);
   (a) persistent gas-T slot at ts cells (REJECT: new synced plane, R4).
   [→ERIK: choose (d) now with (c) as named future upgrade — recommended — or
   (c) now]
2. **Energy-form conduction loses the maximum principle without a per-face
   limiter** (L4-2 · L1-2, independent arithmetic agreeing): at a floored thin
   endpoint the endpoint conversion closes up to 4× the gap per tick — past
   the f=2 divergence line; growing-amplitude ping-pong against the reservoir.
   → v2 adds the fourth §2.3 constraint: per-face |ΔE| ≤ a fraction of the
   gap through the SMALLER endpoint capacity (the P-R4 `LIM_SHIFT`/A1.6
   idiom), counted. [ADOPT — no ruling needed, house-canon fix]

## The other two blockers

3. **"Only combustion + radiation create eth" is false as enumerated**
   (L1-3 · L3-3 · L3-4): under e = N·T_abs every bulk-N writer mints energy —
   the trace-decay→N₂ credit (structurally open; pre-P-S1 scale it was
   ~+30,400 eth/200 s, ≈4× the transport mint), FieldEdit atmosphere deposits,
   water W3 evacuation, seal/unseal redistribution — and the explosion
   `wave_source` FieldEdit writes ΔT directly with no denominator (R2
   violation on any detonation), weapons `deposit_heat` is a third unnamed
   creator. → v2 adds the N-WRITER INVENTORY mirroring §5: each writer ruled
   chemistry-legal / boundary-legal / fix. Proposed class rule for
   non-chemistry mass creation: **born diluting** — T_new = n·T/(n+ΔN),
   energy unchanged (mass arrives cold-relative, no mint). Explosion
   wave_source: named-legal this arc, energy-form rework queued. [ADOPT
   inventory; →ERIK: ratify born-diluting rule + wave_source deferral]
4. **The headline gate is unmeasurable at the named instrument's seams**
   (L2-4 · L3-2): transport, 4c work, and the decay credit all live inside
   the ONE `run_substeps` seam; hashes exist, sums don't. → v2 specifies
   int64 per-tick exported counters on EOSSolver (`eth_transport_delta`,
   `eth_compression_delta`) + energy-valued sums for every one-way site
   (`e_wipe_sum` SIGNED, `e_floor_sum` for T_MIN lifts, ts residual, Pass-1
   attenuation drop — which today has NO counter at all, L3-7), ledger gates
   on counters. `energy_floor_hits = 0` (§7) is unsatisfiable by the
   design's own floor-div bias (L2-3) and is replaced by bounded counter-sum
   gates. [ADOPT]

## L1 — thermodynamics / fire-science realism — VERDICT (verbatim)

> **VERDICT: not blessable (v1).** The central move is right — denominate
> every exchange in energy, let heat ride the conservative mass fluxes — and
> the relative-T energy accounting is exact given mass conservation (the
> 290·ΣN part rides the mass books). But three blockers stand: the furniture
> (ts) pass-through face has no defined energy price, and its most natural
> reading rebuilds the :423 free-energy channel at roughly *twice the rate of
> the leak this arc exists to kill*; the energy-form conduction constraint set
> omits any stability bound, so a spec-compliant P-E2 can be built that
> diverges where today's T-form provably cannot; and the §7 acceptance
> criterion "only combustion + radiation create eth" is unsatisfiable as
> written because every bulk-N writer mints e = ΔN·T_abs on the next rebuild —
> the audit's own smoke→N₂ pump is a ~4× larger eth channel than the +7,805
> transport mint, and it is in neither the creation list nor the accepted
> gaps. Beyond the blockers, every behavioral shift I can find points the
> *same* direction (cooler, weaker, dies-earlier fire), so P-E5 must be
> budgeted as a knee re-placement, not a verification. All three blockers are
> fixable on paper; none invalidates the architecture.

## L2 — Q16.16 / determinism / CPU↔CUDA — VERDICT (verbatim)

> VERDICT: The core move — energy riding the existing donor-cell fluxes — is
> sound, and the CUDA no-atomics claim checks out against the actual gather
> structure of `bulk_flux_transport_cached` and its CUDA port. But §2.1 as
> written is not implementable: it has no answer for gas temperature on
> permeable `thermal_solid` tiles (the transient-e rebuild has no T to rebuild
> from there, and the naive answer re-opens the exact free-energy crate
> channel the A2 occluder was built to kill), it is ambiguous about pre- vs
> post-limiter fluxes (the wrong reading mints), its ≤0 ledger property is
> false in absolute-energy terms on the cold side, its `energy_floor_hits = 0`
> acceptance gate is unsatisfiable against its own floor-division bias, and
> the headline gate is not measurable by the named tool at its current seams.
> The range proof holds only under an undocumented, unenforced N invariant,
> and the digest claim is wrong as stated.

## L3 — energy-ledger forensics — VERDICT (verbatim)

> **VERDICT: NOT BLESSABLE as v1 — with-conditions path exists.** The §2.1
> flux-form transport is the right structural kill for the measured mint, and
> the int64/rounding plan is sound. But the §1 R3 / §7 claim — "after this
> arc NO channel can create thermal energy except combustion and radiation
> deposits, and none silently" — is false as enumerated: I find one
> unspecified face rule that re-opens the exact free-energy channel the arc
> exists to close (ts through-flow), one direct Python ΔT-mint the doc never
> mentions (explosion `wave_source` → `gmap.temperature`), a whole class of
> bulk-mass-at-temperature writers (trace-decay→N₂ credit at shipped dials,
> FieldEdit atmosphere deposits, water W3 evacuation, seal/unseal) that mint
> or destroy `N·T_abs` uncounted, a second dormant SL T-copier with a live
> CUDA twin, and a headline gate (§7 "transport ≤ 0") that the named
> instrument (`storm_ledger.py`) structurally cannot measure. All are fixable
> on paper in v2: specify the ts face law, name the creator/sink channels
> honestly (including Pass-3 as a signed source), and add C++-side per-stage
> energy counters.

## L4 — gameplay feel / regression scope — VERDICT (verbatim)

> VERDICT — The destination is right and the ladder's oracle shapes
> (conservation + parity, recal once at the end) are mostly correct, but the
> plan as written under-scopes its own blast radius in three load-bearing
> ways: it leaves the tree CPU↔CUDA parity-broken for two whole patches
> (P-E1→P-E3) with no pre-declared expected-red manifest, which makes every
> intermediate "set-diff explained" gate a ~20-file wall of true parity
> failures; it never names the test families whose *premises* (not values)
> the arc kills — the fused-3-field SL gate, the plain-ΣT conservation
> metric, and the conduction maximum-principle suite — and its own §3 claim
> that `digest_advect` "keeps its position, same tick point" is contradicted
> by the code it cites; and its P-E0 red gate conflates two observables
> (energy mint vs rail hits) that are fixed by different rungs, so the
> per-patch red/green story as tabled is not executable. The P-E5 HUMAN-TEST
> list also misses the two scenarios where the 0.25 floor raise bites
> hardest: explosion/plasma heat splashes into blast-evacuated cells, and
> water-quench steam. None of this argues against the arc; all of it argues
> the v2 must pre-declare the carnage instead of discovering it patch by
> patch.

## Finding index with proposed v2 dispositions

Blockers 1–4 above. Remaining findings:

- L1-4 / L3-7-tail **floor 0.25 vs the 0.2-trial evidence** — config's own
  record: a 0.2 trial "measurably perturbed marginal ignition timings"; the
  v1 ruling's "untouched" reason is partially false (plateau margin covers
  steady flames, not transient thin-N ignition states; post-arc the floor
  DESTROYS the shortfall). [→ERIK with the evidence: keep 0.25 + explicit
  ignition-chain timing gates (5.0 s touching / 11.2 s gap, P-R4 record), or
  lower the deposit floor and decouple it from the trust N_ref]
- L1-5 / L4-4 **every anchor shift same-signed (cooler/weaker/dies-earlier);
  margin measured at an operating point P-E1 destroys** → scorecard +
  flame-cell N histogram run at the P-E1 BOUNDARY; MacCormack fallback
  decision before CUDA twins are built; P-E5 budgeted as knee re-placement.
  [ADOPT]
- L1-6 **compression work multiplies game-T, not T_abs** — ambient air does
  zero compression work in-sim; the missing acoustic thermalization
  (~N·290·k²/2 per cell-tick) is the one physical damping channel of the
  Helmholtz mode. Fixing honestly needs the KE debit (accepted gap). → new
  ACCEPTED GAP with bound |err| ≤ (γ−1)·|div·dt|·290·N. [ADOPT]
- L1-7 **linear trust fade still compounds ×1.3/tick in the N≈0.15 band**
  (bounded ≈14 atm phantom transients, below the recorder's trip) + one-way
  heating rectification at breathing pockets. [→ERIK: hard-zero below
  0.5·N_ref with fade above (recommended) vs accept-bounded; either way
  P-E0 gains an N≈0.15 pinned-pocket variant and the §2.4 rail claim
  downgrades to "bounded, counted, rare"]
- L1-8 **starved-fire deposit destruction ≈100% of gas-side heat while
  stoichiometry runs** — starvation self-deepens; backdraft-shaped drama
  deleted. [→ERIK: destroy (simplest honest, recommended this arc) vs
  redistribute floored remainder across the extended-draw ring (machinery
  exists, `combustion.cpp:200-209`); either way P-E5 adds a
  smother-then-vent scenario]
- L1-9 / L2-3 **T_MIN rail is a counted CREATOR** → `e_floor_sum` energy
  counter; R3 text amended (floors destroy, rails may create, both counted).
  [ADOPT]
- L1-10 / L2-2 **φ_e pinned to POST-limiter, post-`scale_mag` applied dq**;
  energy rulings for step-4 clamps (N<0→0 benign dilution — stated;
  vacuum zeroing drops e both signs — counted; ambient ring reset is a
  bidirectional channel — ring cells excluded from the transport gate).
  [ADOPT]
- L2-5 / L3-9c **int64 range proof restated from map inventory** (per-cell e
  ≤ N_map·T_max; bench 2e16, 10⁴-ambient map 6.8e17 < 2⁶²) + N ≤ ~2^30 raw
  stated as invariant with debug assert; n_bulk summed as int64. [ADOPT]
- L2-6 **floor-div idiom pinned** (`q = e/n; if (e%n && e<0) --q`; n≥1;
  −n<e<0 → T=−1 stable; quiescent cells rebuild exactly — the fact that
  makes the drift bound acceptable). [ADOPT]
- L2-7 **trust-gate implementation corrections**: `recip_mul_dev` on device;
  magnitude-first multiply (sponge idiom, `eos_solver.cpp:1458-1462`) so
  negative k fades toward zero not past it; fade input = the EXISTING
  trace-inclusive `n_total_`/`d_ntot`/`K_ntot` (zero new reductions; 2%
  trace weight is noise at 0.25); N_ref = 0.25 is a power of two → fade is
  exactly min(4·n, 1). [ADOPT]
- L2-9 / L4-6 / L3-8 **digest_advect necessarily moves across the flux call**
  — declared re-baseline-class; `eos_sl_advect_reference` (:1335) and the
  resident `sl_advect3_device` twin named in P-E1 scope; `cuda_p62` 3-field
  contract + PART-2 replay premise and `test_eos_p2_sealed_room_energy`'s
  plain-ΣT metric are authorized-rewrite items, pre-declared. [ADOPT]
- L2-10 **worst-case drift 5× the mint it replaces if fully active** →
  mitigations stated (quiescent cells lose zero; one-way sign); P-E1 gate
  measures the active-flux fraction instead of assuming it. [ADOPT]
- L2-11 **CUDA no-atomics argument recorded** (gather form verified; face
  values single-written, read by two cells; int64 sums order-immaterial;
  per-cell expression order transcribed). [ADOPT — P-E3 does not re-derive]
- L3-5 **second dormant SL T-copier** (TemperatureSolver Pass-0 gas-T
  advection, live CUDA twin `cuda_temperature.cu:224`, dormant only because
  wind=nullptr) → retire or hard-assert-dead BOTH twins in P-E1. [ADOPT]
- L3-6 **Pass-3 is a SIGNED channel; cold-rail loop survives above the trust
  threshold** (4c at full authority + honest conduction refill + Pass-3
  re-pin = reservoir pump without T_MIN) → Pass-3/sky named signed channels;
  window pocket's N recovered from `ledger_window.npz` BEFORE freezing §7
  rail expectations; cold-rail scenario added to P-E0. [ADOPT — and if the
  measured pocket N sits above the gate band, the residual loop is named an
  ACCEPTED GAP with its measured magnitude, →ERIK at that point]
- L3-7 **Pass-1 misdescribed**: shipped law is the v2.4 absorption form
  `e_abs = deposit·min(N,1)` — silently drops (1−N)·deposit below ambient
  density with NO counter; §2.2 corrected; drop counter added; note the
  0.05→0.25 raise also reduces Pass-1 deposits in the (0.05, 0.25) band.
  [ADOPT]
- L3-9d/e **seal_tiles close-T seed mints object T ex nihilo; low-rail
  creator counted** → named in the writer inventory; `t_low_rail_hits`
  joins the rail list. [ADOPT]
- L4-1 **ladder restructure: pair each CPU rework with its CUDA twin**
  (P-S1/P-R house pattern) — P-E3 dissolves into P-E1/P-E2; expected-red
  manifest per rung committed in v2 (L4's list adopted verbatim as the
  starting manifest). [ADOPT]
- L4-3 **P-E0 splits its two observables**: (a) mint/ΔP-spike gate — green
  at P-E1; (b) rail-hit counters — zero at the trust-gate patch; ladder
  table says which patch owns which. Answers the trust-gate-vacuity worry.
  [ADOPT]
- L4-5 **P-E2 split** into P-E2a (conduction + limiter + Kirchhoff re-gate)
  and P-E2b (floor raise + threshold inventory + margin re-measure). [ADOPT]
- L4-7 **P-E5 HUMAN-TEST additions**: explosion/plasma heat splash into
  blast-evacuated cells; water-quench steam burst; smother-then-vent
  (L1-8). [ADOPT]
- L4-8 **recorder/RL sequencing**: arc lands before any recorder milestone
  that snapshots physics for training; priority-ledger one-liner; P-E5
  validation sessions record the bulk pair (or `n_bulk`) in `fields`.
  [ADOPT]
- L4-9 **the overlay is `HeatFieldOverlay` / `temperature_overlay`**
  (`renderer/game_renderer.py:232-240`), not "HeatMapOverlay"; named in the
  P-E5 brief (phantom max-white pixels disappear = correct; edge softening =
  the §9 fallback trigger). Audit-comparability note for `storm_probe`
  absolute T columns. [ADOPT]
- L4-10 **eth estimator frozen across the arc** (ledger harness-level
  definition already matches §2.1; `fire_tune_loop` plateau reads solid T —
  untouched). [ADOPT — stated in §7]

## What was independently verified sound (one line each)

Relative-T energy accounting exact given mass conservation (L1); "T as
diffusive as the mass it rides" is the standard Le=1 closure and MORE
physical than today's mismatch (L1); the deposit-was-never-the-mint
recognition correct (L1); plateau Charles arithmetic N 0.37–0.43 checks out
(L1); upwind donor identity well-defined across planes (L2); CUDA gather/no-
atomics claim verified against the real code structure (L2); floor-div
portable and bit-identical (L2); `work_clamp_hits` has no absolute asserts —
placement-before-clamp safe; `test_air_boundary.py:820`'s absolute
`t_max_phys_hits==0` moves the SAFE direction under the fade (L2); ledger eth
definition already matches the new law at harness seams (L4); fire itself
writes no T (P-R2/P-S1 confirmed clean, L3).

## Next

v2 of `energy_transport_design_2026-08-16.md` after Erik rules on the five
[→ERIK] items: (1) ts-face law (d) vs (c); (2) floor 0.25 vs the 0.2-trial
evidence + coupling to N_ref; (3) born-diluting rule + wave_source deferral;
(4) trust-gate hard-zero band vs accept-bounded; (5) starved-heat destroy vs
redistribute. Everything tagged [ADOPT] goes into v2 without further debate.
