# Energy-books arc — energy-conservative thermal transport (design v2.2, 2026-08-17)

**Status: v2.2 — round-2 verified (v2.1) and extended by two ruled additions
(§2.7/§2.8) plus the cold-rail engine identification (§2.9). Both additions
passed a focused two-lens critique (thermo + Q16/CUDA determinism); every
finding is folded. Critique phase CLOSED; build in progress (P-E0, P-T0
landed).**

**v2.1→v2.2 changelog:** §2.7 reversible compression work — **hygiene, and the
earlier "standing-wave refrigerator" attribution is RETRACTED** (the window's
engine is §2.9); compression branch kept VERBATIM so the hot rail stays
bit-identical and the sat_add wrap protection survives; expansion divides via
a new shared `floordiv_q` FP_HD helper (plain `/` truncates toward zero and
would MINT on sub-ambient cells — invisible to parity gates, ledger-only).
§2.8 interior drag with heat counterparty as NEW patch P-E3 — per-TICK in
step 4 post-cap (dissolving the substep-count trap), n-weighted int64 oracle
counters, phantom-T guard, ts skip, rail-clip counted, dial default 0.0 +
`k_drag_heat_frac` default 1.0 (RULING R2). §2.9 cold-rail engine identified
(compression on negative game-T) and deferred to its own post-P-E5 patch
(RULING R1). §6 gains P-E3 and two P-E4 gate rows; §7/§8/§9 amended.

**v2→v2.1 changelog (round-2 folds):** §2.1.4 ts-face law given explicit
per-face formulas (the v2 "receiver-priced" phrasing retired as a mint);
ts exclusion from e build/recovery stated; L3 writer table committed
(round-1 doc addendum); P-T0 scope gains the kick-reference Dalton family +
bindings/test signature surface; `test_eos_p4_combustion.py` reclassified to
P-T0; counter definitions pinned (law-independent brackets); conduction
limiter fraction pinned ≤ 1/2; rail-observable xfail-with-owning-patch idiom
stated; §2.4 mid-band claim softened to match the measurement plan. v1 (`bda08eb`) went to a 4-lens adversarial
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
   total, §2.6). **ts cells are EXCLUDED from the e build** (their
   `temperature[]` is the OBJECT's T, ruling A1 — e[ts] would be bogus) and
   **from the recovery write** (the T-WRITE SITE 1/2 guard, `:491`, upheld:
   the EOS never writes a ts tile's temperature). Exact, unshifted. Range: per-cell e ≤ N_map·T_max; bench
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
4. **ts faces (RULING — rule (d), per-face formulas):** relative energy
   never crosses a face touching a `thermal_solid` tile; mass still moves.
   - **air→ts** (air-side donor): the donor is debited at its OWN
     temperature — `e[donor] -= dq·T_rel[donor]` — which leaves the donor's
     recovered T exactly invariant (no concentration mint as mass leaves);
     the debited amount accumulates in the SIGNED counter `e_ts_residual`
     (counted destruction; signed because cold gas carries negative
     relative energy).
   - **ts→air** (air-side receiver): mass arrives carrying ZERO relative
     energy (`e[receiver]` unchanged) — the receiver dilutes toward ambient
     on recovery. This is the §5 born-at-ambient class rule applied to
     emergence from an object; the ts side holds no gas energy, so there is
     nothing to debit and no residual term.
   - **ts→ts:** no energy term (neither side holds gas energy).
   The v2 phrasing "priced at the receiver's own current T" is RETIRED — it
   would have held T constant while minting `dq·T_recv` from nothing.
   Physical story: gas transiting an object sheds its excess relative heat
   (counted in the ledger, not delivered to the object). The honest
   gas↔object convective exchange (via the heat plane / Pass-1 fold) is the
   NAMED FUTURE UPGRADE — not built this arc.
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
4. **Per-face |ΔE| ≤ a fixed fraction — pinned ≤ 1/2, the safe side of the
   f=2 line — of the gap closed through the SMALLER endpoint capacity**
   (the P-R4 `LIM_SHIFT`/A1.6 shift idiom, `cuda_raycaster.cu:263-264`
   precedent) — restores the discrete maximum principle the ΔT form had for
   free; without it a floored thin endpoint closes 4× the gap per tick.
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
  to 1 at n_work_ref (hard-zero below half — REDUCES the L1-7 mid-band fade,
  0.6 → 0.2 at N≈0.15; it does not eliminate it — the residual is exactly
  what the pinned-pocket variant measures; at the 0.25 default the
  power-of-two reciprocal is exact).
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
  deposit sites), `n_active_flux`/`n_bulk_active_sum` (the active-flux
  fraction §7's bound is scaled by), `e_expl_sum` (harness-level, at the
  FieldEdit apply site), `t_low_rail_hits` joins the rail list; the §5
  inventory. **Counter definitions (law-independent, pinned):**
  `eth_transport_delta` = Σ_cells n_bulk·T over the 4c skip-set complement
  (gas cells: !solid, !ts, !vacuum, !ring), sampled at the pinned bracket
  [substep transport-block entry → after step-d flux (HEAD) / after
  recovery (post-P-E1)], accumulated over substeps per tick;
  `eth_compression_delta` = the same sum bracketed around the 4c loop.
  Pure instrumentation, digest-inert; CPU lands at P-E0 (its gate: suite
  failure set + bench digests byte-identical), CUDA twins ride P-E1.
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

### 2.7 Reversible compression work (rides P-E4, same file/twins as the trust gate)

**Motivation — hygiene, NOT the window fix.** The multiplicative update is not
self-inverse: a full oscillation gives `T·(1+k)(1−k) = T·(1−k²)`, so an
oscillating cell loses a proportional slice of T per cycle with no
counterparty. That is real and worth removing. It is **not** the cold-rail
window's amplifier — the earlier claim (v2.2 draft, and the orchestrator's
verbal account to Erik) was WRONG and is retracted here. Measured refutation
(audit §4.3 + P-E0 as-built): at damp 0 the alternating work is harmless
(T_min −0.02); the window descent is monotonic and mostly sub-clamp; and
`T·(1−k)` with k>0 moves a NEGATIVE game-T *toward* zero — it cannot drive
−91 → −288.65. The window's true engine is §2.9.

**Mechanism (asymmetric by design — the rounding is the point).** Let k be the
work factor after the trust-gate fade and the ±T_WORK_CLAMP rail, w = |k|.

- **Compression (k < 0): KEPT VERBATIM** — `dT = mul_q16(k, T);
  T = sat_add_q16(T, −dT)`, exactly today's code. Two consequences, both
  wanted: the hot rail stays **bit-identical to HEAD** (P-E0's measured
  ×1.4972/tick expectations survive untouched, no re-measure), and the
  `sat_add_q16` wrap protection that `eos-p3fix-thermal-ceiling` exists for is
  retained rather than silently dropped. Do NOT reimplement this branch as
  `mul_q16(T, FP_ONE+w)` — that is a 1-LSB-per-tick rounding change on the
  branch this section promises is unchanged.
- **Expansion (k > 0): the inverse** — `T = floordiv_q((int64)T << 16,
  (int64)(FP_ONE + w))`, replacing `T·(1−k)`.
- **k == 0 takes the expansion branch** (pinned; both are exact identity at
  w=0 — `mul_q16(T, FP_ONE) = T`, `floordiv(T<<16, FP_ONE) = T` with zero
  remainder — but an unpinned measure-zero branch is how twins drift on the
  next edit).

**`floordiv_q` is a new shared FP_HD helper in `fixed_point.h`, and its absence
is a silent minter.** C++ and CUDA integer `/` truncate toward ZERO; on
sub-ambient cells (T_rel to −289·65536) that rounds T *upward*, i.e.
eth = ΣN·T_abs *increases* — a mint of exactly the class this arc kills. Both
backends' `/` agree with each other, so **CPU↔CUDA parity gates cannot catch
this; only the ledger can.** Helper (one transcription, shared with §2.1.5's
recovery — not re-derived): `q = n / d; if ((n % d != 0) && ((n ^ d) < 0)) --q;`

**Honest property claims (the draft over-claimed).**

- The pair is exactly self-inverse in reals; in integers the compression branch
  rounds its increment UP (mul_q16 floors a negative dT) while expansion
  floors, so the biases partially cancel. Expected per-cycle residual: 0 in
  compress-then-expand order, ≤1 LSB in expand-then-compress, one-way (never
  above exact). **This is a MEASUREMENT, not an assertion** — P-E4's unit
  oracle sweeps both orders, both signs of T, and w below/at the clamp, and
  the doc records what was measured.
- **"Absolute, not proportional" holds only for symmetric-w cycles.** Under
  asymmetric oscillation a proportional term survives, ≈ ½(Σ_c k² − Σ_e k²)·T
  (worked example: one compression at w=0.4 against two expansions at w=0.2
  deletes 2.8%/cycle — versus 10.4% under the current law). Large gain, not
  total; stated as such.
- **No claim of better adiabatic fidelity.** Exact is `T·r^(γ−1)`; both
  `(1−k)` and `1/(1+k)` match `exp(∓k)` only to first order, in opposite
  directions. The virtue is reversibility, full stop.

**Consequence needing its own gate: expansion cooling weakens ~33% at the
clamp** (0.667 vs 0.5). Nothing in the suite pins the expansion rate
numerically, so this is invisible unless we look. Required follow-ons: (a)
`tests/test_air_boundary.py:820` asserts `t_max_phys_hits == 0` absolutely,
and weaker expansion cooling moves that rail the **unsafe** direction — P-E4
must re-verify it explicitly (contrast §2.4's trust gate, which moved it the
safe way); (b) P-E4 carries no HUMAN-TEST, so **P-E5's play list gains "breach
venting cold / rarefaction look"** — P-S1's blessed look was tuned under the
old rate.

**P-E4 gains a direct oracle** (no battery row can see this section
otherwise): a unit test driving one cell with alternating ±div at fixed |k|,
below and at the clamp, asserting the measured residual bound; plus the
asymmetric-cycle figure recorded as a measurement.

**Cost:** one emulated s64 divide per expansion-branch gas cell per tick (4c
is per-TICK) — tens of µs at 256² on GPU; the CPU reference twin pays
~0.2–0.5 ms. Warp divergence bounded (both arms short).

**Counters:** semantics unchanged (one `work_clamp_hits` per railed cell); pin
the single-compare form `if (w > work_clamp_q)` in all three twins (identical
counts to today's exclusive pair; the verbatim-transcription contract is the
safety story). Values still shift per §2.4's fade declaration.

### 2.8 Interior momentum drag with heat counterparty (NEW patch P-E3, after P-E2b)

**Motivation.** No interior momentum sink exists at shipped dials (audit
headline 5); the undamped Helmholtz mode IS the storming. Damping *without* an
energy destination is what created the 0.002–0.01 rectifier window — real
viscosity deposits wave KE as heat, which makes a damping dial safe at every
value instead of only above a threshold.

**Placement — PER TICK, in the step-4 kick loop, after the |u| cap, before the
store.** The draft said "kick loop per substep"; that is self-contradictory —
the kick runs once per tick (`eos_solver.cpp:618-724`), the substep loop
(`:473`) is advection-only. Correcting placement **dissolves** the kd_sub
question: no per-substep factor, no n_sub dependence (n_sub varies 1–8 per
tick with CFL state, so any per-substep form would make total damping depend
on flow speed and create cliffs at substep-count boundaries), and critically
**no `pow`** — which would have been the codebase's first scalar fold not
built from IEEE-exact ops, a cross-machine desync hazard. Follow the absorb
precedent: `kd_q = quantize(k_drag · dt)` folded once per tick beside the
other scalars. Post-cap placement bounds the deposit (pre-cap, u can
transiently sit at the RAD_SAFE guard ≈16384 m/s and one tick could reach
T_MAX_PHYS). Downstream, 4c then sees dragged velocities — deterministic, and
the direction that cooperates with the storm fix.

**Velocity update.** Component-wise magnitude-first shrink (`u ← u·(1−kd_q)`,
the sponge idiom `eos_solver.cpp:1458-1462`). Load-bearing beyond style:
component-wise magnitude-first makes `|u_old|² − |u_new|² ≥ 0`
**structurally**, so the deposit can never go negative from rounding and needs
no clamp and no signed term in the oracle. An implementer who "improves" this
to a magnitude-based scale silently reintroduces that term — the reason is
recorded here, and a debug assert is cheap.

**Heat deposit.** `ΔE_cell = (|u_old|² − |u_new|²) / 2` in ledger units;
`ΔT = k_drag_heat_frac · ΔE_cell / c_v`, into the same cell's T. Same-cell is
right for a collocated grid (shear heating at a door neck is physically
placed).

- **`k_drag_heat_frac` is a new dial, default 1.0 (RULING R2, Erik
  2026-08-17).** Rationale: Q16 game units put air's heat capacity ~700× below
  physical (c_v = 1 by convention), so a fully honest deposit runs ~700×
  physical. **MEASURED CORRECTION (P-E3, 2026-08-17):** the pre-build estimate
  of "~96 game-deg/s into neck cells at a sustained 20 m/s" was wrong by ~12× —
  the measured rate is **≈8.05 game-deg/s** at k_drag 0.02, verified by hand
  against the locked formula. The ignition-risk concern that motivated this
  dial is therefore far milder than feared, but the dial stays (it is the
  honest knob for the KE→heat exchange rate, and the risk scales with whatever
  k_drag Erik ultimately ships). Default 1.0 keeps the conservation oracle
  EXACT through every gate; Erik sweeps the fraction at P-E5 (physical-air
  anchor ≈0.0014).
  Any non-deposited remainder is a **counted, named** destruction channel
  `e_drag_drop_sum` (R3-legal).
- **Arithmetic:** |u|² is int64-safe post-cap (≤ 2·(6.55e7)² ≈ 8.6e15), but ΔT
  at c_v=1 can exceed int32 (a full stop from 1000 m/s is ~6.55e10 raw) — keep
  int64 until a **saturating** narrow, then apply the rails.
- **Phantom-T guard:** skip the T-write when `n_bulk < 1 raw count` (foregone
  energy ≈0, counted anyway). Without it the deposit writes full ΔT into
  evacuated cells where the books balance trivially (0 = 0) while ignition
  thresholds read the number — precisely the "temperature not backed by
  energy" pattern §5 forbids.
- **ts cells:** the kick's skip set lacks `ts`, but 4c skips it (ruling A1 —
  the EOS may not write a thermal_solid's temperature). Pinned: **skip both
  the drag and the deposit at ts cells**, so the oracle stays exact with no
  new residual term.
- **Order:** deposit lands BEFORE 4c, identically in all twins.

**Oracle (exact, at the drag site).** Two int64 per-tick counters accumulated
inside the drag loop, both **n-weighted** (raw ΔT is not comparable to KE):
`ke_drag_removed += n·(u_old² − u_new²)` and `e_drag_deposit += n·ΔT_applied`,
with the product shifted inside (`n·(Δu² >> 16) >> 16`) — the naive raw
product reaches ~2^70 and overflows int64 before summation. Per-tick reset
(`PER_TICK_COUNTERS` idiom). Identity asserted per tick:

`ke_removed = 2·c_v·(e_deposit + e_drag_drop + e_drag_rail_clipped) ± Σ_active n·LSB_T·2c_v`

The rail-clip term is NOT optional bookkeeping: with c_v=1 a capped jet
reaches T_MAX_PHYS in ~14 ticks, so clipping is an expected regime, and every
clipped LSB is KE destroyed without counterparty unless counted.

**Dormancy.** Dial default 0.0 ⇒ mechanism ships silent and digests must be
byte-identical. Branch on the **quantized** fold (`if (kd_q > 0)`,
dormancy-by-branch precedent) — not the float, since k_drag = 1e-6 quantizes
to 0 and a float-keyed branch would disagree about which code runs. P-E3
carries a P-E0-style inertness gate: suite failure set + bench digests
pre/post at default 0.0, run twice.

**Twins:** all four kick sites — live `step()`, `eos_kick_compression_reference`
(the P6.4 verbatim-replay contract), `cuda_kick_compression.cu` K1, and the
resident path. K1 today takes neither `temperature` nor `ts`; its signature
grows (own-cell T write from K1 is race-free — K2 reads only neighbour u).
`KickScalarFolds` gains kd_q; the `d_cnt` D2H block grows 5 → ~8 slots (an ABI
edit touching the reference signature, the .cu slot comments, bindings, and
`cuda_kick_check.py` — named so it isn't discovered late). int64 `atomicAdd`
on two's-complement is exact and order-free — the right tool for value sums.

**Gates.** Window battery (damp 0.005, strip 0.5 restored, k_drag 0.02
override) expected CLEAN — **stated WITHOUT any dependence on §2.7, which
lands one rung later at P-E4.** Justification is measured, not hoped: the
audit's damped-safe row (damp 0.02, strip 0.5) was clean under the current
law, and k_drag 0.02 is a strictly stronger sink than wave_absorb 0.02
(A@0.02 ≡ k_drag 0.0067, audit §5B). The forbidden-band config load-warn
tripwire rides along. CPU+CUDA same patch.

**Feel (for P-E5's brief) — MEASURED, superseding the pre-build estimate.**
P-E3 measured the free-momentum e-fold at k_drag 0.02 as **≈49.6 s**, not the
≈2.1 s this section originally predicted (~24× off; the corrected figure was
verified by hand against the locked formula). Consequence for the sweep:
**0.02 is far too weak to be the "storm dies in seconds" dial — k_drag ≈ 0.5
gives a ~2 s e-fold**, so Erik's P-E5 sweep should start well above the audit's
suggested 0.02–0.05 band if he wants the originally-described feel. The
qualitative picture stands at whatever value produces a seconds-scale e-fold:
room-scale sloshing and quiet-room drift die, smoke hangs sooner, blast
pushback at range softens, while pressure-DRIVEN flows (breach suction, fire
convection) survive at a terminal velocity where kick balances drag, with
softened transients plus the neck heating above. **P-E5 play list gains "door-neck temperature under sustained venting"
as its own row.**

### 2.9 The cold-rail engine — identified, and deliberately deferred (RULING R1)

**What it actually is.** Step 4c multiplies **ambient-relative** T. Below
ambient (T_rel < 0) the compression branch `×(1+|k|)` makes a cold cell
COLDER: compression freezes sub-ambient gas. The window's pocket is dense
(P-E0 measured n_bulk 1.7–9.3 — sustained convergence, k<0), so this branch is
exactly what drives −91 → −288.65 → the T_MIN floor, after which the pinned
hull conducts in for free and the kick converts the standing pressure deficit
to wind. This is the §8 game-T-vs-T_abs gap not merely omitting physics but
**inverting** it below ambient.

**The honest fix, specified for later:** run the (reversible) work on absolute
temperature — `T_new = (T + 290)·(1±w) − 290` — so compression warms
sub-ambient cells and ambient air finally heats under compression at all,
restoring the missing acoustic thermalization the §8 bound names.

**RULING R1 (Erik, 2026-08-17): NOT in this arc.** It re-opens a ruled
accepted gap mid-arc and is feel-adjacent everywhere (breach rarefaction
becomes genuinely cold — ~97 game-deg at the clamp vs 0 today — and venting
acoustics change). It lands as **its own short designed patch immediately
after P-E5**, with its own critique round and its own HUMAN-TEST, back-to-back
with the recalibration Erik is doing anyway. Within THIS arc the window dies
operationally via k_drag (§2.8, measured basis above) and the engine is
documented here as the sharpened form of the §8 gap.

**Consequence for the cold-rail accepted-gap question:** it is no longer
"accept a mystery." P-E4's window battery re-run measures whether the loop is
gone in practice; §7's window-row expectations are set from that measurement,
and §2.9 names the residual mechanism precisely with its scheduled fix.

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
redesign (which moves to bulk-mass+ΔE injection, §2.6). **Drag deposit
(§2.8, from P-E3)** — a CONVERTER, not a true creator: its source is the
kick's unpaid pressure work (the half-coupled gap, §8), and it is counted at
the drag site by `e_drag_deposit` / `e_drag_drop_sum` /
`e_drag_rail_clipped`.

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

**T-writer table:** the L3 WRITER TABLE is COMMITTED as an addendum to
`energy_transport_critiques_round1_2026-08-16.md` (round-2 fold — it must be
readable by worktree subagents, not live in session transcript) and is the
completeness oracle; every row is covered by a § of this doc (rows 3, 6, 13,
14 gained coverage in v2). Explosions' counter is `e_expl_sum` (§2.5). P-E2b's threshold inventory (§6 of v1, unchanged:
ignition stays temperature-based; no threshold may act on a temperature not
backed by energy) verifies the CONSUMER side the same way.

## 6. Patch contract (per the autonomous-patch-workflow skill)

Merge semantics: green gates commit to the arc branch; auto-continue applies
within the branch; ONE merge to main after P-E5 HUMAN-TEST. Memory
checkpoint at every boundary. Expected-red manifest: Appendix A, maintained
per rung — "a parity red in the manifest is the rung's declared debt; any
OTHER red is a stop." **Cross-rung red idiom:** P-E0's rail observable is
xfail-with-owning-patch (owner P-E4): it stays a DECLARED red carried in
EVERY intermediate rung's manifest (P-T0 through P-E2b) and flips strict at
P-E4. `ledger_window.npz` is a regenerable audit artifact, NOT committed —
P-E0 regenerates it via the audit §7 command.

| # | patch | contents | mode | tier | oracle / gate | HUMAN-TEST |
|---|---|---|---|---|---|---|
| P-E0 | repro + instruments | hot-rail repro (dump anatomy; RED on HEAD: ΔP-spike/mint observable AND rail-counter observable, asserted SEPARATELY) + cold-rail window scenario + pinned N≈0.15 pocket variant + window-pocket N recovered from `ledger_window.npz` + the per-stage energy counters (`eth_transport_delta`, `eth_compression_delta`) landed AHEAD of the law change so P-E1's gate is measurable | subagent | Opus-class (defines the oracles) | all scenarios deterministic + committed; counters exported + ledger reads them; reds on HEAD documented | no |
| P-T0 | trace 0% | `trace_mass_scale` retired from ALL THREE Dalton families: (1) both live-step sites (`eos_solver.cpp:345-358`, `:561-574`) + CUDA twins (`cuda_eos_step.cu:190,507,583`, `cuda_eos_resident.cu:767`); (2) **the P6.4 kick-reference family** — `eos_kick_compression_reference` (`eos_solver.cpp:1402-1414`, sig `:1371`) + device twin (`cuda_kick_compression.cu:327,348`, `.h:66`) so the VERBATIM-replay contract holds; (3) the bindings/API surface (`bindings.cpp:1107-1147, 2129, 2242-2300`; `eos_solver.h:207-215, :484, :508`) + explicit call-sites (`tests/test_thermal_mass_axis.py:566,613`, `tests/cuda_thermal_mass_eos_check.py:160`, `tests/cuda_kick_check.py:71,318`). Decay→N₂ credit deleted (`physics_engine.cpp:498-525` + `bindings.cpp:2812` note); stale-key guards (P-S1 idiom); bench re-anchor — pre-registered expectation = the audit's pump-off row, regenerated (`storm_audit_2026-08-14.md:360` command) | subagent | Sonnet 5 (subtractive, oracle: ledger + suite) | bulk-N creation stays 0 LSB; suite set-diff vs manifest; CPU↔CUDA tol 0 incl. the kick-check pair | no |
| P-E1 | energy transport, CPU+CUDA together | §2.1 complete: e-plane on applied fluxes, ts rule (d), recovery+guards, BOTH SL T-copier retirements, reference twin, resident twin, digest move, `cuda_p62`/`eos_sl_advect_reference` authorized rewrites | subagent | Opus-class | `eth_transport_delta` ≤ 0 bounded (counters, not seams) + active-fraction measured; O2 conservation unchanged; P-E0 mint observable GREEN (rail observable stays red by design); CPU↔CUDA tol 0 SAME PATCH; **anchor scorecard + flame-cell N histogram at this boundary** (MacCormack fallback decision point) | no |
| P-E2a | conduction energy form | §2.3 with the 4-constraint set incl. the per-face limiter; Kirchhoff exact-0 re-gate; Pass-3/sky named signed channels; max-principle test rewrite (authorized) | subagent | Opus-class | face-antisymmetry exact; Kirchhoff 0 both backends; ledger; CPU↔CUDA tol 0 same patch | no |
| P-E2b | deposit dial + inventories | `n_floor_heat` → dial default 0.01 (int64 recip path to 0.001); `n_work_ref` dial plumbing; Pass-1 drop counter; §5 T-threshold consumer inventory executed; margin/N-histogram re-measure | subagent | Sonnet 5 (counters+parity oracle; inventory is mechanical against L3's table) | lockstep tol 0; counters bounded; inventory table committed | no |
| P-E3 | interior drag + heat counterparty | §2.8 complete: per-tick drag in step 4 post-cap (all four kick twins), component-wise magnitude-first shrink, same-cell deposit with `k_drag_heat_frac`, phantom-T guard, ts skip, n-weighted int64 oracle counters, `k_drag`/`k_drag_heat_frac` dials + forbidden-band load-warn tripwire | subagent | Sonnet 5 (hard conservation oracle) | drag identity exact per tick (§2.8) incl. rail-clip + drop terms; **dormancy inertness at default 0.0** (suite set + bench digests byte-identical, twice); window battery clean at k_drag 0.02 **WITHOUT depending on §2.7**; CPU↔CUDA tol 0 same patch | no |
| P-E4 | trust gate **+ §2.7 reversible work** | §2.4 in the three step-4c twins + folds + resident (hard-zero-below-half fade); §2.7 expansion-branch inverse via the shared `floordiv_q` helper, compression branch verbatim | subagent | Sonnet 5 (parity oracle) | lockstep tol 0; P-E0 rail observable GREEN; N≈0.15 variant measured + recorded; **§2.7 unit oracle** (alternating ±div, both cycle orders, both signs of T, below/at clamp — measured residual bound recorded); **explicit re-verification of `test_air_boundary.py:820`'s absolute `t_max_phys_hits == 0`** (weaker expansion cooling moves that rail the UNSAFE direction); window battery re-run → sets §7's window-row expectations | no |
| P-E5 | recal + bless | `fire_tune_loop` scorecard vs pre-arc; storm-ledger battery (all rows; rails bounded, counters within derived bounds); canon fold + archive | inline (Erik + orchestrator) | Opus-class | **HUMAN-TEST: Erik plays** — blowup level, two-room in-game, space/venting map, smoke/fire look, `temperature_overlay` (T-key; phantom max-white gone = correct, edge softening = §9 fallback trigger), explosion/plasma splash into blast-evacuated cells, water-quench steam burst, smother-then-vent, **breach venting cold / rarefaction look** (§2.7 weakens expansion cooling ~33% at the clamp), **door-neck temperature under sustained venting** (§2.8), **the `k_drag` / `k_drag_heat_frac` sweep** (Erik picks the shipped values) | **YES — merge gate** |

(P-E3 was dissolved at v2.1 per L4-1; v2.2 REBORN it as the drag patch —
different contents, same slot position after P-E2b. The T_abs work form
(§2.9) is a separate patch queued immediately AFTER P-E5, with its own
design + critique round + HUMAN-TEST.)

## 7. Ledger acceptance (measured by counters + `tools/storm_ledger.py`)

On the two-room bench battery (baseline / damped / window-restored /
N≈0.15-pocket rows), after P-E4:
- `eth_transport_delta` ≤ 0 every tick, |Σ| within the Σ n_bulk·2⁻¹⁶ bound
  scaled by the MEASURED active-flux fraction (was +7,805 unbounded);
- conduction: face-antisymmetric exact; global drift = counted floor terms;
- creators: only the §5 named channels, each counter-attributed;
- rails: `t_max_phys_hits = 0` on baseline/damped rows; **window-row
  expectations are SET FROM P-E4's measured battery re-run, not asserted in
  advance** (§2.9 — P-E0 measured the window pocket at n_bulk 1.7–9.3, far
  above the trust band, so `n_work_ref` cannot bound that loop; k_drag is
  what closes it operationally and the measurement says by how much);
- **drag identity** (§2.8) holds every tick on every row carrying
  `k_drag > 0`: `ke_removed = 2·c_v·(e_deposit + e_drag_drop +
  e_drag_rail_clipped)` within the stated per-cell LSB slack;
- eth estimator frozen across the arc (harness-level Σ N_bulk·T_abs — the
  ledger's existing definition, which the arc makes exactly right).

## 8. ACCEPTED GAPS (decisions, not findings)

- KE↔eth **HALF-coupled** (amended v2.2): the kick still mints KE with no eth
  debit, and §2.8's drag now launders that mint into eth. Bounded by
  k_drag·KE and small at bench scale (EOS KE injection 146–286 per 200 s vs
  +38k eth elsewhere, audit §4 table), but it is a new positive-feedback path
  at blast scale and is named here rather than discovered later. The kick-side
  debit remains the open half.
- **Compression work multiplies game-T, not T_abs** — bound
  |err| ≤ (γ−1)·|div·dt|·290·N per cell-tick. **SHARPENED + SCHEDULED at
  v2.2:** below ambient this does not merely omit physics, it INVERTS the sign
  (compression freezes cold gas) and is the cold-rail window's engine — see
  §2.9. No longer open-ended: the T_abs form is a queued patch immediately
  after P-E5 (RULING R1).
- §2.7's per-cycle one-way residual (≤1 LSB expected, measured at P-E4) — a
  bounded, named integer ratchet replacing an unbounded proportional leak.
- Traces carry no thermal energy — now trivially true (0% ruling).
- Object pore gas not separately modeled (ruling A3); ts rule (d) residual
  counted; convective-exchange upgrade named, not built.
- Floor/wipe/rail one-way terms — counted in energy, bounded, preferred over
  remainder-redistribution machinery.
- Water/steam thermal coupling absent (verified: `water_solver.cpp` has no
  thermal contact) — steam citizenship is the water arc's decision via the
  §2.6 recipe.

## 9. Out of scope

Damping as a bare feel dial (audit A/B/C) — **superseded within this arc by
§2.8**, which builds the same sink WITH an energy counterparty (a bare
`wave_absorb` raise remains available but is no longer the plan); the T_abs
compression-work form (§2.9 — queued as its own designed patch immediately
after P-E5, not out of scope so much as out of THIS arc); golden re-bless (deferral stands; new fuel-bearing golden at blessed
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

- **P-T0 (authorized rewrites):** `test_eos_p4_combustion.py` —
  `test_trace_decay_credits_inert_n2_exactly` (`:236-269`) asserts the
  deleted credit; the `:159-184` conservation helper's premise ("smoke
  decays back in") dies with it. Signature-surface reds at the explicit
  `trace_mass_scale` call-sites (`test_thermal_mass_axis.py:566,613`,
  `cuda_thermal_mass_eos_check.py:160`, `cuda_kick_check.py:71,318`).
  Pressure-sensitive digests where smoke was dense (bench re-anchor
  documents deltas vs the regenerated pump-off row). Cosmetic:
  `tools/eos_p5_bake.py:679` doc row, `test_ps1_smoke_roundtrip.py:8`
  docstring. No parity reds expected once the kick-reference family moves
  in the same patch.
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
  `test_temperature_ignition.py`, `test_water_boil.py` (`:200-206` comment
  goes stale, assert stays green; `:229-246` decay itself survives),
  `test_continuous_o2_law.py`, `test_pr3_capacity_law.py`,
  `test_thermal_mass_axis.py` (post its P-T0 signature fix),
  `test_fuel_fraction_axis.py`.
- **Expected stable:** `test_bench_two_room.py`, `test_ps1_smoke_roundtrip.py`
  (bounded-above idiom — note P-T0 makes it strictly easier),
  `test_sky_exchange.py`, `test_temperature_cooling.py`,
  `test_unit_heat_damage.py`, `test_recorder_dump.py`,
  `test_blackbody_ramp.py`, `test_air_boundary.py` (absolute rail assert
  moves the safe direction).
