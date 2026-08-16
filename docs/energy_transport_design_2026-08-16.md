# Energy-books arc — energy-conservative thermal transport (design v2, 2026-08-17)

**Status: v2 — all round-1 blockers resolved on paper; submitted to a round-2
verification pass before build.** v1 (`bda08eb`) went to a 4-lens adversarial
panel; verdicts + finding index: `energy_transport_critiques_round1_2026-08-16.md`
(`6141cfc`). v2 folds every [ADOPT] disposition and Erik's post-panel rulings
(2026-08-16→17 session, each marked `RULING`). Branch: `storm-damping`.
Supersedes the never-built "E1 options 2+3" rails patch (§10 history).

**v1→v2 changelog:** ts-face law written (rule (d)); conduction gains the
per-face limiter; creator/N-writer inventory added; per-stage energy counters
specified (headline gate now measurable); **traces leave the physics books
(0% ruling)** — decay→N₂ credit deleted, born-at-ambient mooted; both floors
became dials; ladder restructured (CUDA twins ride their CPU patches; P-E2
split; P-E0 observables split); digest moves declared; second SL T-copier
retired; game-T compression-work gap named with bound.

---

## 0. Problem and evidence (unchanged from v1)

T is transported by semi-Lagrangian *copy* (`eos_solver.cpp:457-508`; `:423`
self-documents the "FREE-ENERGY channel"). Terms with collapsing denominators
write enormous T values into near-empty cells; transport copies those values
onto real mass; T×mass is energy — the copy is a mint. Measured: the in-game
hot-rail blowup (audit §4.4: ×1.4957/tick to T_MAX_PHYS, 47–65 atm spikes at
shipped dials); the cold-rail window (§4.3); **+7,805 eth injected by EOS
transport in a normal 200 s bench run even post-P-S1** (§4 pump-off row).

**RULING:** bounded-rails rejected as destination ("we still mint energy,
just less — a very unstable system"); close the books structurally, now,
while goldens are deferred and recalibration is already scheduled.

## 1. The principle (one sentence + four rules)

**Every thermal exchange is denominated in energy; temperature is what energy
looks like through a cell's actual mass.**

- **R1 — Transport conserves.** Thermal energy rides the conservative
  donor-cell face fluxes that bulk mass rides. Mixing is mass-weighted by
  construction; phantom T carries ~no energy and dilutes on contact.
- **R2 — Conversions are local and honest.** Energy→T at any endpoint divides
  by that endpoint's actual capacity (gas: N·c_v; object: `thermal_mass` via
  `heat_inv_shift`).
- **R3 — One-way guards, all counted, in ENERGY units.** Floors/wipes may
  only destroy; the T_MIN/low rails may create; every such site carries an
  int64 energy-sum counter, not just a hit counter. After this arc the only
  eth *creators* are the NAMED channels: combustion, radiation, explosions
  (§5 inventory) — and none is silent.
- **R4 — Determinism unchanged.** Q16.16/int64 integer arithmetic,
  order-pinned loops, CPU↔CUDA bit-identical, no new synced planes (the
  energy accumulator is transient within a substep).

## 2. Mechanism, pass by pass

### 2.1 EOS transport (the core change) — replaces the SL T-sample

Inside the existing substep loop (`eos_solver.cpp:453-534`):

1. **Retire the `.t` slot** of the fused SL sample (T-WRITE SITE 1/2) — in
   THREE places: the live step (`:505`), the CPU reference twin
   (`sl_advect_reference`, `:1335`), and the resident device twin
   (`sl_advect3_device`, `cuda_eos_resident.cu:745`). Also retire (or
   hard-assert-dead, both backends) the SECOND dormant SL T-copier:
   TemperatureSolver Pass-0 gas-T advection (`temperature_solver.cpp:199-224`,
   CUDA `cuda_temperature.cu:224`) — live only because `step_tail` passes
   null winds; one plumbing change must not silently re-open the mint.
   The A2 `t_occlude` mask's transport role retires with the sample; the
   `ts`/vacuum/ambient skip semantics keep their exact current meaning.
2. **Build the transient energy accumulator** (int64 scratch, not synced):
   `e[i] = (int64)n_bulk_raw[i] · (int64)T_raw[i]`, `n_bulk = O2 + inert_N2`
   (the `gas_conservative` pair — post-0%-ruling this equals the Dalton
   total, §2.6). Exact, unshifted. Range: per-cell e ≤ N_map·T_max; bench
   ≈2e16, a 10⁴-ambient-count map 6.8e17 < 2⁶² — stated invariant
   N_cell < 2^30 raw with a debug assert (the divergence apply narrows to
   int32 unchecked, `bulk_transport.cpp:181`).
3. **Energy rides the APPLIED mass flux**: for each face, φ_e =
   (Σ_conservative dq_gi) · T_raw[donor], where dq_gi is the **post-limiter,
   post-`scale_mag`** per-plane flux the mass books actually move
   (`bulk_transport.cpp:136-167`) and the donor is the upwind cell the mass
   flux already chose (v_face is wind-only, sign-preserved by the rescale —
   donor identity is shared across planes). Implementation is gather-form
   like the mass pass itself: face values single-written, each read by two
   cells; int64 sums are order-immaterial, per-cell expression order
   transcribed from CPU (L2-11 — P-E3-class work does not re-derive this).
4. **ts faces (RULING — rule (d)):** NO energy flux through any face whose
   donor or receiver is a `thermal_solid` tile. Mass still moves; gas
   arriving on an air-side receiver is priced at the receiver's own current
   T (no eth change from that inflow). The forgone transfer accumulates in a
   SIGNED counter `e_ts_residual`. The honest gas↔object convective exchange
   (via the heat plane / Pass-1 fold) is the NAMED FUTURE UPGRADE — not
   built this arc.
5. **Recovery:** `T[i] = floordiv(e[i], n_bulk_new[i])`, floor division
   toward −∞ pinned by idiom `q = e/n; if (e % n != 0 && e < 0) --q`
   (identical C++/CUDA; n ≥ 1 so INT64_MIN/−1 unreachable; −n < e < 0
   recovers T = −1 and rebuilds exactly — stable). Quiescent cells (no net
   face traffic) rebuild EXACTLY — the LSB loss applies only to
   active-flux cells. Guards, each with an energy counter:
   - `n_bulk_new < N_EPS` (1 raw count): T := 0, residual → signed
     `e_wipe_sum` (bound per cell ≈ N_EPS·T_max ≈ 0.24 dequant — donor
     convexity argument: recovered T is a mass-weighted mix of rail-clamped
     donor T's).
   - T_MIN clamp on recovery: counted in `e_floor_sum` (a CREATOR — R3).
   - Ring/vacuum wipe unchanged; the per-substep ambient ring N-reset is a
     bidirectional energy channel — ring cells are EXCLUDED from the
     transport gate and named a boundary channel (§5).
   - `N < 0 → 0` clamp (`bulk_transport.cpp:208`): creates mass against
     fixed e — benign dilution, stated.
6. **Digest declaration:** `digest_advect` today hashes (wx, wy, T) at `:508`
   BEFORE the flux call (`:516`); post-arc T-after-recovery only exists
   after it. The digest MOVES across the flux call — a re-baseline-class
   digest-stream reorder, declared here, spent under the arc's standing
   golden deferral. `eos_sl_advect_reference` (3-field contract) and the
   `cuda_p62` gate pair are authorized rewrites (Appendix A).

Ledger property (restated per L2-3/L3-9a): the transport pass's Σ eth
contribution is **≤ 0 per tick modulo the counted signed terms**
(`e_wipe_sum`, `e_floor_sum`, `e_ts_residual`, ring channel), with the
truncation loss bounded by Σ_active n_bulk·2⁻¹⁶ per substep. Worst-case
fully-active drift is ~5× the mint it replaces (L2-10) — acceptable because
quiescent cells lose exactly zero and the sign is one-way; **P-E1's gate
measures the active-flux fraction rather than assuming it.**

### 2.2 Deposits — already energy-form; the floor becomes a LOW dial

`ΔT = ΔE/(N·c_v)` at the cell's true N deposits exactly ΔE — the deposit was
never the mint. With R1 in place, deposit spikes at thin cells are honest and
dilute on contact. Therefore combustion (`combustion.cpp:799-803`) keeps its
form, and:

- **RULING: `n_floor_heat` becomes a low, tunable dial — default 0.01,
  swept DOWNWARD during tuning** ("we can see how low we can go"). The v1
  0.25 requirement is STALE: closing the books removed the floor's stability
  job; T_MAX_PHYS is the real value backstop. Decoupled from the trust gate
  (the v1 "one shared constant" ruling is retired with it). The reciprocal
  path gets int64 intermediates so even 0.001 is reachable. Consequence:
  the L1-4 ignition-timing objection INVERTS — lowering the floor heats thin
  cells MORE than today's 0.05, so marginal ignition gets slightly faster,
  not slower; the config's 0.2-trial evidence is moot in this direction.
  Anchors gate still re-measures.
- Floor engagement destruction (fraction 1−N/floor of ΔE) is ~zero at 0.01
  and gets an energy counter anyway (`e_deposit_drop_sum`). The v1 starved-
  fire destruction concern (L1-8) is DISSOLVED by this ruling; the
  redistribute-to-draw-ring alternative is not needed. Smother-then-vent
  still joins the P-E5 HUMAN-TEST list.
- **Pass-1 correction (L3-7):** the shipped radiative gas deposit is the
  v2.4 absorption-proportional form `e_abs = deposit·min(N,1)`,
  `ΔT = e_abs/(max(N, floor)·c_v)` (`temperature_solver.cpp:349-374`) — the
  (1−N)·deposit attenuation drop below ambient density is PHYSICAL
  (absorptivity ∝ density), stays, and gains the energy counter it never
  had. `heat_floor_hits` exists only at the combustion site today; both
  sites get energy-sum twins.
- Object branch (`deposit >> heat_inv_shift`) untouched — always honest.
- **Kirchhoff acceptance re-run (hard gate at P-E2a):** equal-T pairs net
  `rad_net` EXACTLY 0, both backends. P-E2a's spec first inventories which
  paths deposit ray heat into gas before touching them.

### 2.3 Conduction (Pass 2 air↔air, solid↔air) — energy form, LIMITED

Constraints (the fourth added per L4-2/L1-2):
1. Face-antisymmetric energy quantum ΔE_face (what leaves i enters j, exactly).
2. Endpoint-local conversion (R2), floors counted in energy.
3. One-way counted guards only.
4. **Per-face |ΔE| ≤ a fixed fraction of the gap closed through the SMALLER
   endpoint capacity** (the P-R4 `LIM_SHIFT`/A1.6 idiom) — restores the
   discrete maximum principle the ΔT form had for free; without it a floored
   thin endpoint closes 4× the gap per tick (past the f=2 divergence line).
The exact current Pass-2 law is transcribed into P-E2a's spec before rewrite.
Pass-3 cooling / sky / ambient pinning stay open by design but are named
**SIGNED channels** in the ledger (L3-6: Pass-3 relaxes toward 0 from BOTH
sides — it can create). Cold-rail residual: the reservoir loop can survive
above the trust band (4c at full authority + honest refill + free re-pin);
**before freezing §7's rail expectations, P-E0 recovers the window pocket's
N from `ledger_window.npz`**; if it sits above the gate band, the residual
loop returns to Erik as a measured accepted-gap decision.

### 2.4 Compression work — T-form + the trust-gate DIAL

The per-parcel law stays multiplicative on T (`eos_solver.cpp:726-784`).
Additions:

- **RULING: trust gate as a dial** `n_work_ref` (config-plumbed; default
  0.25): fade factor = 0 for n < n_work_ref/2, linear from 0 at n_work_ref/2
  to 1 at n_work_ref (hard-zero below half — kills the mid-band compounding
  L1-7 found; N_ref power-of-two makes the fade exact integer arithmetic).
  Input n = the existing `n_total_`/`d_ntot`/`K_ntot` plane — post-0% this
  IS the bulk sum; zero new reductions in any twin. Implementation:
  magnitude-first multiply (the sponge idiom, `eos_solver.cpp:1458-1462`) so
  negative k fades toward zero, never past it; `recip_mul` host /
  `recip_mul_dev` device; applied BEFORE the ±T_WORK_CLAMP compare
  (placement changes `work_clamp_hits` — only parity-compared, and
  `test_air_boundary.py:820`'s absolute `t_max_phys_hits == 0` moves the
  safe direction).
- Rail claim (downgraded per L1-7): bounded, counted, expected rare — NOT
  zero by fiat. The residual mid-band transient (static worst case ~8–14 atm
  if nothing drains the pocket; expected far lower since honest transport
  and limited conduction now fight the climb every tick) is MEASURED by
  P-E0's pinned N≈0.15 pocket variant; `n_work_ref` is the lever if the
  measurement displeases.
- **ACCEPTED GAP (named per L1-6): the work term multiplies game-T, not
  T_abs.** Ambient air does zero compression work in-sim; the missing
  acoustic thermalization (~N·290·k²/2 per cell-tick, bound
  |err| ≤ (γ−1)·|div·dt|·290·N) is the one physical damping channel of the
  Helmholtz mode. Honest fix needs the KE↔eth coupling (already an accepted
  gap); the ledger names where the acoustic energy should have gone.

### 2.5 What retires / is born / is unchanged

- **Retires:** both SL T-copiers (2.1.1); the `:423` free-energy comment
  (rewritten to state the conservation property); `trace_mass_scale` and the
  decay→N₂ credit (§2.6); the v1 0.25-shared-constant ruling.
- **Born:** transient int64 `e_` scratch; dials `n_floor_heat` (repurposed,
  default 0.01) and `n_work_ref` (default 0.25); energy counters
  `eth_transport_delta`, `eth_compression_delta`, `e_wipe_sum` (signed),
  `e_floor_sum`, `e_ts_residual` (signed), `e_deposit_drop_sum` (both
  deposit sites), `t_low_rail_hits` joins the rail list; the §5 inventory.
- **Unchanged:** mass transport arithmetic, EOS solve/kick, cool laws,
  Huggett anchors, the radiation law, digests infra (one declared move).

### 2.6 Trace gases — the 0% ruling

**RULING (Erik, 2026-08-17): traces leave the physics books entirely.**
Rationale: the half-citizenship (2% pressure weight, zero thermal weight,
full-weight conversion on decay) is the root of both the audit's pressure
pump and the round-1 energy-mint class; "why not approximate 2% with 0% and
lose all this complexity."

- `trace_mass_scale` retires: traces leave the Dalton sum (both sites,
  `eos_solver.cpp:345-358`, `:561-574`, + CUDA twins). n_total ≡ n_bulk
  everywhere afterward.
- **The P4 decay→N₂ credit is DELETED** (`physics_engine.cpp:498-525` + the
  binding note at `bindings.cpp:2812`): decayed trace counts simply vanish.
  The P4 doctrine ("decay is oxidation, not deletion") is deliberately
  retired — with zero pressure weight there is no mass to conserve; written
  rationale = this section. The born-at-ambient question is MOOTED.
- **What stays (nothing else is thrown away):** all five trace planes, their
  once-per-tick SL advection + diffusion (`smoke_dynamics`), breach venting
  drag, gas damage, visibility, render — traces remain fully alive as
  visual/gameplay fields; they just stop whispering into pressure.
- **Bonus:** the P-S1 §5 ex-nihilo queue (grenade puffs, explosion smoke,
  steam deposits) CLOSES for physics purposes — render-only sources cannot
  pump anything.
- **The full-citizenship recipe (preserved upgrade path, per plane):** flip
  the plane's `gas_conservative` flag; weight 1.0 in Dalton; it joins the
  per-substep conservative flux AND the energy build/recovery; every source
  of that plane gets a real mass+energy debit; recalibrate. Cost scaling:
  flux/energy passes are linear in plane count (2→3 planes ≈ 1.5× those
  passes) and the promoted plane moves per-substep (≤8×/tick) instead of
  per-tick. Steam is the expected first candidate, owned by the water arc.
- **Explosive-redesign brief note (queued, not this arc):** explosions do
  not need trace citizenship — as a named creator channel they inject
  real bulk N₂ (detonation products) + ΔT directly; the visible smoke stays
  render-only. Better physics than promoted smoke at none of the cost.

## 3. Q16.16 arithmetic and determinism

Ranges restated from map inventory (§2.1.2); floor-div idiom pinned
(§2.1.5); e built exact/unshifted, flux products exact int64; n_bulk summed
as int64 before divide; order-pinning per §2.1.3; no new synced state;
`digest_advect` move declared (§2.1.6). CUDA: gather-form energy faces (two
int64 face scratch planes, ~1 MB @256², or recompute dq·T[donor] in the
gather — T is frozen until the recovery kernel); `KickScalarFolds` gains
`n_work_ref` folds; resident path plumbs the fade input it already owns.

## 4. Edge semantics

As v1 §4 (thermal_solid exclusion per ruling A1; vacuum/ring wipes; ambient
interior skips) plus: ts-face rule (d) with `e_ts_residual`; ring cells
excluded from the transport gate and named a boundary channel; `seal_tiles`'
close-T seed (`gamemap.py:1995-1997`) named in the inventory (one-shot
object-books mint, bounded, counted at P-E2b if nonzero in practice).

## 5. The creator / N-writer / T-writer inventory (round-1 L3 table, ruled)

**Legal eth creators (named, counted):** combustion deposit (chemistry);
radiative deposit (chemistry); **explosions** — RULING: `wave_source`
FieldEdit ΔT writes (`field_edit.py:539-540, 446-466`) and payload
`deposit_heat` (`payloads.py:112-129`) are legal creators, energy = N·ΔT
counted at the write site; semantics unchanged until the queued explosive
redesign (which moves to bulk-mass+ΔE injection, §2.6).

**Mass writers (class rules):**
- Trace-decay→N₂ credit: DELETED (§2.6).
- FieldEdit atmosphere deposits (`field_edit.py:602-609`): arrive AT AMBIENT
  (T_rel = 0 ⇒ e_rel unchanged — the energy-neutral reading; no machinery).
- Water W3 displacement (`physics_engine.cpp:785-812`) and seal/unseal
  redistribution (`gamemap.py:1966-1979`): mass MOVERS — move energy with
  mass (receiver-priced this arc, counted approximation; dormant on the
  bench; upgraded to donor-priced if the water arc's gates ever demand it).
- Ambient ring clamp: boundary channel, named; energy column added only when
  an ambient-map battery row exists.
- Combustion O2-sink/N₂-credit (`combustion.cpp:762-772`): inside the named
  chemistry channel.

**T-writer table:** round-1 L3 WRITER TABLE adopted verbatim as the
completeness oracle; every row is covered by a § of this doc (rows 3, 6, 13,
14 gained coverage in v2). P-E2b's threshold inventory (§6 of v1, unchanged:
ignition stays temperature-based; no threshold may act on a temperature not
backed by energy) verifies the CONSUMER side the same way.

## 6. Patch contract (per the autonomous-patch-workflow skill)

Merge semantics: green gates commit to the arc branch; auto-continue applies
within the branch; ONE merge to main after P-E5 HUMAN-TEST. Memory
checkpoint at every boundary. Expected-red manifest: Appendix A, maintained
per rung — "a parity red in the manifest is the rung's declared debt; any
OTHER red is a stop."

| # | patch | contents | mode | tier | oracle / gate | HUMAN-TEST |
|---|---|---|---|---|---|---|
| P-E0 | repro + instruments | hot-rail repro (dump anatomy; RED on HEAD: ΔP-spike/mint observable AND rail-counter observable, asserted SEPARATELY) + cold-rail window scenario + pinned N≈0.15 pocket variant + window-pocket N recovered from `ledger_window.npz` + the per-stage energy counters (`eth_transport_delta`, `eth_compression_delta`) landed AHEAD of the law change so P-E1's gate is measurable | subagent | Opus-class (defines the oracles) | all scenarios deterministic + committed; counters exported + ledger reads them; reds on HEAD documented | no |
| P-T0 | trace 0% | `trace_mass_scale` retired from both Dalton sites + CUDA twins; decay→N₂ credit deleted; stale-key guards (P-S1 idiom); bench re-anchor (pressure shifts ≤2% where smoke was dense) | subagent | Sonnet 5 (subtractive, oracle: ledger + suite) | bulk-N creation stays 0 LSB; suite set-diff vs manifest; CPU↔CUDA tol 0 | no |
| P-E1 | energy transport, CPU+CUDA together | §2.1 complete: e-plane on applied fluxes, ts rule (d), recovery+guards, BOTH SL T-copier retirements, reference twin, resident twin, digest move, `cuda_p62`/`eos_sl_advect_reference` authorized rewrites | subagent | Opus-class | `eth_transport_delta` ≤ 0 bounded (counters, not seams) + active-fraction measured; O2 conservation unchanged; P-E0 mint observable GREEN (rail observable stays red by design); CPU↔CUDA tol 0 SAME PATCH; **anchor scorecard + flame-cell N histogram at this boundary** (MacCormack fallback decision point) | no |
| P-E2a | conduction energy form | §2.3 with the 4-constraint set incl. the per-face limiter; Kirchhoff exact-0 re-gate; Pass-3/sky named signed channels; max-principle test rewrite (authorized) | subagent | Opus-class | face-antisymmetry exact; Kirchhoff 0 both backends; ledger; CPU↔CUDA tol 0 same patch | no |
| P-E2b | deposit dial + inventories | `n_floor_heat` → dial default 0.01 (int64 recip path to 0.001); `n_work_ref` dial plumbing; Pass-1 drop counter; §5 T-threshold consumer inventory executed; margin/N-histogram re-measure | subagent | Sonnet 5 (counters+parity oracle; inventory is mechanical against L3's table) | lockstep tol 0; counters bounded; inventory table committed | no |
| P-E4 | trust gate | §2.4 in the three step-4c twins + folds + resident; hard-zero-below-half fade | subagent | Sonnet 5 (parity oracle) | lockstep tol 0; P-E0 rail observable GREEN; N≈0.15 variant measured + recorded | no |
| P-E5 | recal + bless | `fire_tune_loop` scorecard vs pre-arc; storm-ledger battery (all rows; rails bounded, counters within derived bounds); canon fold + archive | inline (Erik + orchestrator) | Opus-class | **HUMAN-TEST: Erik plays** — blowup level, two-room in-game, space/venting map, smoke/fire look, `temperature_overlay` (T-key; phantom max-white gone = correct, edge softening = §9 fallback trigger), explosion/plasma splash into blast-evacuated cells, water-quench steam burst, smother-then-vent | **YES — merge gate** |

(P-E3 dissolved into P-E1/P-E2a per L4-1. Patch IDs otherwise kept.)

## 7. Ledger acceptance (measured by counters + `tools/storm_ledger.py`)

On the two-room bench battery (baseline / damped / window-restored /
N≈0.15-pocket rows), after P-E4:
- `eth_transport_delta` ≤ 0 every tick, |Σ| within the Σ n_bulk·2⁻¹⁶ bound
  scaled by the MEASURED active-flux fraction (was +7,805 unbounded);
- conduction: face-antisymmetric exact; global drift = counted floor terms;
- creators: only the §5 named channels, each counter-attributed;
- rails: `t_max_phys_hits = 0` on baseline/damped rows; window + pocket rows
  per the P-E0 measurements (bounded + counted; if the measured window
  pocket sits above the trust band, the residual loop goes to Erik as a
  measured accepted-gap ruling before P-E5);
- eth estimator frozen across the arc (harness-level Σ N_bulk·T_abs — the
  ledger's existing definition, which the arc makes exactly right).

## 8. ACCEPTED GAPS (decisions, not findings)

- KE↔eth uncoupled (kick creates KE; 4c changes eth) — rails + trust dial
  bound abuse; ledger names it.
- **Compression work multiplies game-T, not T_abs** — bound
  |err| ≤ (γ−1)·|div·dt|·290·N per cell-tick; the missing acoustic
  thermalization is named (L1-6); honest fix rides the KE↔eth reform.
- Traces carry no thermal energy — now trivially true (0% ruling).
- Object pore gas not separately modeled (ruling A3); ts rule (d) residual
  counted; convective-exchange upgrade named, not built.
- Floor/wipe/rail one-way terms — counted in energy, bounded, preferred over
  remainder-redistribution machinery.
- Water/steam thermal coupling absent (verified: `water_solver.cpp` has no
  thermal contact) — steam citizenship is the water arc's decision via the
  §2.6 recipe.

## 9. Out of scope

Damping (audit A/B/C — parked; this arc removes the explosion tail, not the
ring); golden re-bless (deferral stands; new fuel-bearing golden at blessed
dials); the explosive redesign (brief note §2.6 queued to its own arc);
MacCormack/BFECC (named fallback; decision at the P-E1 boundary, before any
further CUDA work); recorder/RL sequencing — this arc lands BEFORE any
recorder milestone that snapshots physics for training (priority-ledger
one-liner at canon fold); P-E5 validation recordings add the bulk pair to
`fields`.

## 10. History

Supersedes "E1 options 2+3" (2026-08-16): its repro became P-E0; its trust
gate became P-E4 (dial, hygiene); its floor raise inverted into the LOW
`n_floor_heat` dial; the audit's option-E value guard was refuted by the
dump itself (T_MAX_PHYS fired; blowup anyway). v1's 0.25-shared ruling
retired by the 0% + low-floor rulings. Round-1 panel record:
`energy_transport_critiques_round1_2026-08-16.md`.

## Appendix A — expected-move manifest (starting point; maintained per rung)

- **P-T0:** pressure-sensitive digests where smoke was dense (bench
  re-anchor documents deltas); no parity reds expected.
- **P-E1 (authorized rewrites):** `test_cuda_p62_sl_advection.py` +
  `cuda_p62_check.py` (3-field contract + PART-2 replay premise),
  `eos_sl_advect_reference` consumers, `digest_advect`/`digest_bulk_flux`
  stream order, `test_cuda_bulk_flux.py`.
- **P-E1 (value-churn on already-red):** the shared canonical golden (11
  CUDA gate files + `test_w6_armory`) moves again.
- **P-E2a (authorized rewrites):** `test_temperature_conduction.py`
  (max-principle → limiter-bounded property + plain-ΣT metric),
  `test_eos_p2_sealed_room_energy.py` (metric premise),
  `test_pf1a_radiation_books.py` (floor counters).
- **Trajectory-coupled, must stay green or be re-derived:**
  `test_eos_p4_combustion.py`, `test_temperature_ignition.py`,
  `test_water_boil.py`, `test_continuous_o2_law.py`,
  `test_pr3_capacity_law.py`, `test_thermal_mass_axis.py`,
  `test_fuel_fraction_axis.py`.
- **Expected stable:** `test_bench_two_room.py`, `test_ps1_smoke_roundtrip.py`
  (bounded-above idiom — note P-T0 makes it strictly easier),
  `test_sky_exchange.py`, `test_temperature_cooling.py`,
  `test_unit_heat_damage.py`, `test_recorder_dump.py`,
  `test_blackbody_ramp.py`, `test_air_boundary.py` (absolute rail assert
  moves the safe direction).
