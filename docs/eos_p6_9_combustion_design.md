# EOS P6.9 — Combustion: isotropic proportional oxygen, GPU-resident

*Design section for the last P6 sub-patch. Builds on `docs/eos_p6_gpu_alignment_review.md`
§2.3/§3.1 (which planned a faithful fixed-order port) and the 2026-07-11 decision session
with Erik, which chose to REMOVE the directional bias entirely rather than reproduce it.
**v2 — resolved against the adversarial critique of 2026-07-11** (fixes folded in; the
critique's central catch — FOUR behavioral deltas, not two — is now §5, front and centre,
and is the one item awaiting Erik's bless).*

## 1. Decisions locked (Erik, 2026-07-11)

- **Combustion moves to the GPU.** Motivation is not compute (the pass is cheap) but the
  S8 full-residency cost: a CPU-only combustion pass forces a per-tick host round-trip of
  gas / temperature / wall_hp. Moving it on-card removes that transfer.
- **Full concurrency via two order-free halves:** (1) **gate-snapshot** — ignition gates read
  a frozen pass-entry copy of temperature; (2) **proportional oxygen sharing** — contested
  oxygen is split proportionally to demand, commutatively. Removes both directional biases,
  fully parallel. The performance goal and the anti-"cheese" fairness goal converge here.
- **Validation by principle + objective proof, not feel-check:** golden re-baseline ONCE; a
  new isotropy test; P5.1 lifecycle E2E re-run; O2-trio (265/48/39) re-measured; perturbation
  trio stays green.
- **Sub-LSB remainder:** fixed deterministic tiebreak, NOT rotated (Erik: not a purist).
- **Two-step build:** CPU reformulation first (Erik blesses the behavioral delta), then the
  GPU port (pure bit-identity). The GPU step empties `EOS_P6_PENDING_KERNELS` → **P6 closes.**

## 2. The current pass (baseline being replaced)

`combustion.cpp` is a row-major **scatter**: for each flammable source `i` with fuel and
`temperature[i] ≥ ign`, loop its N/S/W/E neighbours `j`; if `O2[j] > thresh`, burn
`min(burn_cap, O2[j])`, writing `O2[j] -= burn`, `SOOT[j] += …`, `N2[j] += …`,
`temperature[j] += dT` (**per source, against a running `N_total` that falls with each
sub-burn**, `combustion.cpp:114-129`), and `wall_hp[i] -= fuel_per_o2·burn` (floored, P5.1).
Two order dependencies make it non-parallel and biased: the ignition cascade (down-right)
and the oxygen competition (up-left).

## 3. The new structure — two independent gather passes

Scatter → **gather** (cell reads neighbours), single writer per cell, both passes fully
parallel with no cross-cell ordering. Runs identically on CPU and GPU (the CPU is
reformulated to this same algorithm — it is NOT the old scatter).

**Snapshot & the three freezes.** One explicit combustion-local buffer:
`Tsnap = copy(temperature)` (~100 KB at 160²), because temperature is read by a gate AND
written (deposited to `[j]`) — a genuine cross-cell read-after-write that only an explicit
copy breaks. Two OTHER gate-read-and-written fields are frozen **implicitly** by the gather
structure, and this must be stated because one of them IS a behavioral delta:
- **`O2[j]`** — read (threshold + split denominator) and written (`-= burn_j`) by the *same*
  cell, read-before-write ⇒ every claimant sees **pass-entry** O2. This replaces the old
  live/sequentially-decremented O2 and is the mechanism of deltas §5.β/§5.γ.
- **`wall_hp[i]`** — gate-read in Pass A, written only in Pass B (after all Pass-A reads) ⇒
  Pass A sees pre-payment wall_hp, matching the old pass (which checks the fuel gate once,
  before `i` pays). Consistent, no explicit copy needed.
`fire[i]`, `ign[i]`, and the masks are static (never written in-pass) — no freeze needed.

**Pass A — air cells (single writer of `O2[j], SOOT[j], N2[j], temperature[j]`):**
for each open-air cell `j`:
1. Gather its ≤4 flammable neighbour-sources `i`. Source `i` *claims* iff `flammable[i]`,
   `wall_hp[i] > FUEL_FLOOR`, `ign[i] > 0`, `Tsnap[i] ≥ ign[i]`, and (pass-entry) `O2[j] >
   o2_thresh`. `demand_i = burn_cap` (uniform across claimants).
2. `D = Σ demand_i`. If `D ≤ O2[j]`: `alloc_i = demand_i` (no contention). Else, **EXACT
   INTEGER split** — the plain int64 `/` operator, NOT float division and NOT `reciprocal_q16`.
   *Determinism rationale (Erik, 2026-07-11):* integer `/` has a single correct answer (floor
   of the quotient), bit-identical on every CPU and CUDA device — it is FLOAT division we
   forbid (non-portable), never integer. This is not the soot/N2 idiom because that divides by
   a *constant* (`soot_yield`) ⇒ a multiply; here `D` is a per-cell *variable* ⇒ a real divide
   is unavoidable. `reciprocal_q16` is rejected precisely because it is inexact (~1 ULP) and
   would make `Σ alloc ≠ O2` — conservation is non-negotiable here. The Q16.16 scale factors
   cancel (the split is a dimensionless ratio), so the raw-integer form is direct:
   ```
   alloc_i = (int64)O2[j] * demand_i / D          // floor, exact integer divide
   key_i   = (int64)O2[j] * demand_i % D           // integer remainder = tiebreak key
   R       = O2[j] - Σ alloc_i                      // provably an integer in [0, #claimants) ⊂ [0,4)
   ```
   Distribute the `R` leftover LSBs to the `R` claimants with largest `key_i`, ties → lowest
   source index. `Σ alloc_i = O2[j]` **exactly**. Overflow-safe: `O2·demand < 2^43` in the
   int64 stage even against the range-checked O2-tank spike (20 bits slack). *(Because
   `demand_i` is uniform, the keys all tie ⇒ the ≤3 leftover LSBs always land on the
   low-index faces — the fixed sub-LSB bias §1 accepts; the isotropy test §6 tolerates it.)*
3. `burn_j = Σ alloc_i`. Write `O2[j] -= burn_j`; `SOOT[j] += soot(burn_j)`;
   `N2[j] += burn_j − soot(burn_j)` (LSB-exact Dalton split, unchanged); **one aggregate**
   `temperature[j] += dT(burn_j)` against the post-burn `N_total`, T_MAX_PHYS clamp + counter.
   This single aggregate deposit **intentionally replaces** the old per-source sequential
   deposits (delta §5.δ); a per-source replay would reintroduce an order-dependent denominator
   and defeat isotropy. Rail counters (`heat_floor_hits`, `t_max_phys_hits`) are now
   **per-cell**, not per-source — no test may assert their absolute value.

**Pass B — source cells (single writer of `wall_hp[i]`):** each flammable source `i` reads
its ≤4 air-neighbours' allocations to it from the **face buffer** (below), sums
`burn_i = Σ_j alloc_{i→j}`, pays `wall_hp[i] -= round(fuel_per_o2 · burn_i)`, floored ONCE at
FUEL_FLOOR. **Total-then-floor-once** (critique B): proven never to break the "smolder never
destroys" 1-LSB invariant (both rules engage the floor iff the total does; never 0-vs-1) and
differs from P5.1's per-neighbour-floor only by ≤3 LSB away from the floor — inside the
golden re-baseline.

**alloc plumbing = (a) a per-face buffer** (critique D). Pass A writes 4 direction-keyed face
buffers (the `cuda_water` `dq_e/dq_s` precedent) recording each `alloc_{i→j}`; Pass B gathers
the ≤4 faces pointing at it. ~400 KB at 160² (trivial). Rejected (b) recompute-in-B: it would
force a SECOND copy of the split logic that must stay bit-identical forever or O2-drained-at-j
and fuel-paid-by-i silently desynchronize — a conservation hazard no single-cell test catches.

## 4. Why this is bit-identical CPU↔GPU by construction

Both passes are per-cell functions of frozen inputs (`Tsnap`, pass-entry `O2`, pre-payment
`wall_hp`, `fire`, `ign`, masks) and the gas/temperature planes each cell alone writes. No
Pass-A gate reads live temperature (they read `Tsnap`); no cell reads another cell's
within-pass gas write. The only order is the fixed in-cell remainder tiebreak (bounded ≤4,
in-thread). All Q16.16 integer; the divide is exact and identical on CPU and CUDA; division
only in the contended branch. **GPU barrier chain:** `snapshot(T) → Pass A → face buffers →
Pass B` is a real 3-stage dependency — on CPU the loop order gives it free; on GPU it is the
explicit launch-barrier chain, structurally identical to `cuda_fire`'s barriered pass chain
(`tests/cuda_fire_check.py:14-23`). The digest gate is then a formality the design guarantees.

## 5. THE FOUR BEHAVIORAL DELTAS — the item awaiting Erik's bless

My v1 claimed only two changes. The critique proved that false: making combustion **order-free
is a package** — resolving a whole cell's burn as one commutative operation necessarily changes
*how much* burns and *how hot*, not just *who ignites when*. All four are **systematic (new ≥
old)**, **bounded**, and arguably **more physical** than the scan-order scatter. None is a bug;
the point is they must be **named in the golden-rebaseline rationale**, never absorbed silently.

- **α — Ignition defers one tick.** A source can no longer heat a furniture neighbour and
  ignite it the same tick. *Removes the down-right ignition-cascade bias.* (The intended one.)
- **β — Contested oxygen splits proportionally/uniformly** instead of first-come-first-served,
  with the fuel-payment redistributing to match. *Removes the up-left oxygen-competition bias.*
  (The intended one.)
- **γ — Contested cells now fully drain their oxygen (NEW).** Old skipped the last
  sub-threshold sliver per source (`O2 mod burn_cap ≤ o2_thresh` left unburnt); the gather
  drains `O2[j]` to 0. So a contested cell burns up to `o2_thresh` (~0.03 real) **more** O2
  (and matching soot/N2/heat) per tick. Forced by isotropy — the alternative is order-dependent.
- **δ — Multi-source cells deposit aggregate heat (NEW, the largest).** Old deposited per
  source against a running `N_total` that *fell* with each sub-burn; the gather deposits once
  against the post-total-burn N. Net `≈ b²·H·soot_yield/(c_v·n₁·n₂)` **more** heat per
  2-source cell per tick — **~125 LSB ≈ 0.0019 T-units** at typical values. This fires on
  **any air cell with ≥2 burning neighbours** — the norm in a real fire, not an edge case.
  In absolute terms tiny (~0.002 K/tick, fires are ~1500 K) so behaviourally negligible, but
  it moves the golden digest measurably and may nudge self-starve timing by a tick or two.

**All four are consequences of the isotropy Erik chose** — you cannot have a
direction-free, parallel combustion pass without resolving each cell's burn as one order-free
operation, and that operation drains fully (γ) and deposits aggregately (δ). The bless
question is simply: **accept the full package** (fires burn marginally more oxygen and run
marginally hotter in multi-source cells, in exchange for zero directional bias and full GPU
concurrency), which I recommend and expect — or reconsider.

## 6. Validation (updated by the critique)

- **Golden re-baseline ONCE**, rationale enumerating **all four deltas** α–δ so the digest
  move and the O2-trio shift are *explained*, not hidden.
- **O2-differentiation trio** (265/48/39) re-measured; ordering + perturbation gate stay green.
- **P5.1 lifecycle E2E** re-run (the 1-LSB char-out invariant is preserved — critique B).
- **Isotropy test** — center-ignite a symmetric (plus-shaped) furniture cluster, assert the
  four arms burn identically **within a ≤3-LSB tolerance** (the uniform-demand remainder lands
  on low-index faces — §3 step 2; a bit-exact assertion would fail on the sub-LSB bias §1
  already accepted). Alternatively construct the scenario so contested cells have zero
  remainder (`O2` divisible by claimant count) for a bit-exact assert.

## 7. Build steps (after Erik blesses §5)

1. **P6.9a (CPU):** reformulate `combustion.{h,cpp}` to §3; add the isotropy test; re-run
   P5.1 E2E + trio + perturbation; golden re-baseline ONCE (rationale = §5 α–δ).
   **Erik blesses the behavioral delta** (no auto-merge — feel-adjacent).
2. **P6.9b (GPU):** `cuda_combustion.cu` mirroring the two gathers + face buffers + barrier
   chain; digest gate bit-identical vs the P6.9a CPU reference; wire dispatch; remove
   `"combustion"` from `EOS_P6_PENDING_KERNELS` → set empty → delete the migration machinery.
   **P6 closed.** (Cross-machine Ada-vs-Ampere leg batched at the Sweden desktop.)
