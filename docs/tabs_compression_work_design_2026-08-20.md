# T_abs compression work — design (v1, 2026-08-20)

**Arc:** `tabs-compression-work` (TODO ★ item 2). **Origin ruling:** RULING R1
(Erik, 2026-08-17), `docs/archive/energy_transport_design_2026-08-16.md` §2.9 —
this patch is that ruling executed, with its own critique round and its own
HUMAN-TEST. **Feel-adjacent: nothing merges before Erik plays.**

## 1. The defect (verified against HEAD 3b13cf9)

Step 4c runs the reversible work on **ambient-relative** T
(`eos_solver.cpp:984-1022`). Below ambient (T_rel < 0) the compression branch
multiplies a negative number by (1+w): compression makes cold gas COLDER — the
physics is not merely omitted, it is **inverted**. This is the cold-rail
window's engine (§2.9: the measured −91 → −288.65 → T_MIN spiral). Two more
consequences of the relative form:

- **Ambient air (T_rel = 0) never heats under compression or cools under
  expansion at all** — k·0 = 0. Acoustics deposit no temperature signal
  anywhere near ambient; the §8 accepted-gap bound
  |err| ≤ (γ−1)·|div·dt|·290·N names exactly this missing term.
- **Breach rarefaction is thermally invisible**: venting a room to space
  today leaves T_rel = 0 in its wake. Honest expansion at the work clamp
  (w = 0.5) is 290 K → 193 K, i.e. **~97 game-deg of cold** — the figure R1
  blessed as the intended feel.

## 2. The law (the whole change)

The stored convention of `temperature[]` — **ambient-relative, 0 = ambient** —
is LOAD-BEARING and does not change: `eos_energy_books_sum` accumulates
`n·T_rel` (`eos_solver.cpp:234-257`), the vacuum/ring wipe counts a `T<0` pin
as creation (`temperature_solver.cpp:117-137`), render and ignition read
relative T. **Only the interior of the 4c arithmetic goes absolute**, then
shifts back out:

```
t_abs   = temperature[i] + t_amb_q                    // s_eos ≡ 1, see D-3
// compression (k < 0): same structure as HEAD, operand t_abs
dT      = mul_q16(k_signed, t_abs)                    // k_signed = −w
t_new   = sat_add_q16(temperature[i], −dT)            // = T + w·(T+290)
// expansion (k ≥ 0, k==0 pinned here — exact identity):
t_new   = (q16)floordiv_q((t_abs) << 16, FP_ONE + w) − t_amb_q
```

Everything around it is UNCHANGED: the div(u_new) stencil, γ−1 and dt folds,
the P-E4 trust-gate fade (thin-N cells still fade k to 0 before the clamp),
the single-compare ±T_WORK_CLAMP rail with `work_clamp_hits`, the T_MIN floor
(`energy_floor_hits`) and T_MAX_PHYS ceiling (`t_max_phys_hits`), the
`eth_pre_4c`/`eth_compression_delta` bracket, skip set (solid/ts/vacuum/
ambient-ring), and the P-E0 counter semantics.

Q16 notes:
- `t_amb_q = quantize(T_AMB_K)` — the same fold the prestage already computes
  (`cuda_eos_step.cu:153`); value 290·65536.
- Overflow: t_abs ≤ (16000+290)·65536 ≈ 1.07e9; `<<16` ≈ 7.0e13 — int64-safe
  with an order of magnitude to spare.
- **The §2.7 sub-ambient mint hazard dissolves structurally**: t_abs is
  non-negative for all T ≥ −290, so the expansion numerator can no longer be
  negative. `floordiv_q` is KEPT anyway (shared idiom, zero cost on a
  non-negative numerator, robust if T_MIN ever moves).
- Compression keeps `sat_add_q16` (wrap protection retained,
  eos-p3fix-thermal-ceiling lineage). The §2.7 "compression branch verbatim"
  promise was scoped to the energy-books arc and is **deliberately revoked
  here** — changing that operand is this patch's entire point.

## 3. What changes physically (name it before measuring it)

- **Sub-ambient inversion fixed**: compression WARMS cold gas. The cold-rail
  engine dies structurally, not just operationally-via-k_drag (§2.8's fix
  killed the storm that fed it; this kills the mechanism).
- **Ambient air participates**: sound waves now deposit/withdraw T everywhere.
  Under the reversible pair a symmetric cycle cancels to ≤1 LSB one-way
  residual — but that ratchet now acts on EVERY gas cell under acoustics,
  not only on cells already off-ambient. Direction: never above exact (a slow
  bounded cooling drift). **RISK-2, measured at P-W2** (quiet-room long-run
  eth drift per 1000 ticks); expected ~LSB-scale = 1/65536 game-deg per cycle
  per cell, invisible at feel scale, but it belongs in the books.
- **Hot-rail compounding gets FASTER**: at the clamp the compression ratio
  moves from ×1.5 to ×(1.5 + 145/T) per tick (T=1000: ×1.645; T=3000:
  ×1.548). `tests/test_air_boundary.py:820` asserts `t_max_phys_hits == 0`
  absolutely and weaker/stronger expansion/compression both move rails —
  **RISK-1: explicit re-verification is a P-W1 gate row. A red there is a
  STOP** (back to the design table with the measurement), not a re-pin.
- **Venting drives geometric T_abs decay toward the T_MIN floor** (t_abs → 1 K)
  under sustained positive divergence — `energy_floor_hits` becomes live on
  the cold side for the first time. Counted, never silent.
- **Conduction relaxes cold pockets back**: the ambient-pinned hull now
  conducts INTO sub-ambient gas — physical, and it bounds how long cold
  pockets persist. (Energy-books priced this leg; `t_min_gas` gate unaffected
  in sign.)
- **eth_compression_delta changes regime**: ambient cells contribute for the
  first time; sustained venting makes it strongly negative (cooling), sustained
  compression positive. Recon confirmed NO sign/bound gate pins it anywhere —
  only recorded storm-ledger baselines, which regenerate at P-W2.

## 4. Design decisions

**D-1 — The cap²-plane ambient floor STAYS this arc (the inherited D1
question, answered conservatively).** `eos_solver.cpp:434` and its identical
host twin `cuda_eos_step.cu:198` floor a cell's cap-T at ambient; measured
no-op today, LIVE after this patch. We do NOT let the velocity cap follow
T_abs below ambient, for three reasons: (a) scope — R1 ruled the work form;
the cap law was ruled in the velocity-clamp arc WITH this floor explicit (its
D1 note anticipated this arc and handed us the number, not a mandate);
(b) venting survival — a c(T_abs) cap in a rarefaction pocket throttles
exactly the breach-venting flow, and the k_drag=10 probe (2026-08-20) showed
killing venting is a catastrophic feel failure mode; (c) the consequence is
bounded and countable — flow in a cold pocket may run supersonic w.r.t. its
OWN c (up to ~Mach 17 at the T_MIN floor, 300/17.6) but never above ambient c,
i.e. never worse than every cell was before the clamp arc. **P-W2 measures the
real exposure** (sub-ambient census; |u| vs own-c(T_abs) distribution on the
venting bench) and the numbers go to Erik at HUMAN-TEST — lowering the floor
is a future Erik decision made on data, not a default taken here.

**D-2 — No dial. The law is replaced outright.** R1 ruled the honest form IS
the fix; a `work_on_abs_t` toggle would keep the inverted law alive as a
maintenance surface and invite papering. A/B for the HUMAN-TEST is
build-vs-main, not a flag.

**D-3 — The simple form is tied to the frozen scale by a named guard.**
`t_abs = T + 290` is only honest while `S_EOS ≡ 1.0` (value-frozen at P-K3;
`s_eos_q == FP_ONE`, `eos_solver.cpp:373-381`). The general form is
`t_abs = s_eos·T + t_amb` with the inverse scale on the way out — machinery we
refuse to build for a frozen dial (simplest honest design). Guard: one
`assert(s_eos_q == FP_ONE)` beside the 4c fold with a comment naming this doc;
if phi_exp/S_EOS ever unfreezes, the assert fires instead of the law silently
bending. ACCEPTED GAP: the general-scale form is deliberately unbuilt.

**D-4 — k==0 stays pinned to the expansion branch** (exact identity in the
new form too: `floordiv((T+290)<<16, FP_ONE) − 290 = T`, zero remainder).
Measure-zero branch pinning is how twins stay twins (§2.7 precedent).

**D-5 — Counter semantics unchanged.** Same three rails, same counters, same
single-compare clamp form. Values will move (that is the behavioral change);
meanings must not.

## 5. Twin sites and plumbing (recon-verified, 2026-08-20)

Exactly **three** transcriptions of the 4c arithmetic:

1. `cpp/src/eos_solver.cpp:984-1022` — live `step()`.
2. `cpp/src/eos_solver.cpp:~1786-1829` — `eos_kick_compression_reference`
   (P6.4 verbatim-replay contract: it mirrors the live loop line for line).
3. `cpp/src/cuda_kick_compression.cu:268-338` — `compression_kernel` (K2),
   the ONLY device copy; both the per-call entry (`eos_kick_compression`,
   :421-539) and the resident path (`cuda_eos_resident.cu:822-829`) reach it
   through the one shared launch core `kick_compression_launch_resident`
   (`cuda_resident.h:139-143`). No fourth transcription exists.

**ABI edit, named now so it isn't discovered late (P-E3 precedent):**
`KickScalarFolds` (`cuda_resident.h:114-138`) gains a `t_amb_q` field;
`compression_kernel`'s signature grows by one scalar; both CUDA call sites
pass it; the CPU reference twin's signature gains the same parameter, which
touches `bindings.cpp` and the direct callers in `tests/cuda_kick_check.py` /
`tests/test_cuda_p64_kick_compression.py`. The D2H counter ABI
(`counters_out[9]`) is UNTOUCHED — no new counters.

## 6. Patch plan

Merge semantics: green gates commit to the arc branch; ONE merge to main after
the HUMAN-TEST. Memory checkpoint at every boundary. Expected-red manifest
maintained per rung — a declared red is the rung's debt; any OTHER red is a
stop. Golden re-baseline happens ONCE, at close, after Erik's bless, with a
written rationale doc (standing ruling re-confirmed 2026-08-20).

| # | patch | contents | mode | tier | gate | HUMAN-TEST |
|---|---|---|---|---|---|---|
| P-W0 | baselines | On the branch BEFORE the law change: run + record the storm-ledger battery rows, the cold-rail window scenario (`test_e1_cold_rail` trajectory numbers), hot-rail bench (`t_max_phys_hits`, peak T), and a quiet-room eth-drift run — the BEFORE half of every P-W2 comparison, committed as a dated capture doc | subagent | Sonnet, low | commands + outputs committed; no code touched | no |
| P-W1 | the law, all twins together | §2's arithmetic in all three sites + §5's `t_amb_q` plumbing/ABI + the D-3 guard; re-derive `tests/test_p_e4_reversible_work.py` bounds for the absolute law (measure both cycle orders, both T signs, w below/at clamp — record the measured residuals in-file, §2.7-style); pin the ~97-game-deg clamp figure as a unit oracle (new: one cell, w at clamp, expansion from T_rel=0 → assert T_new = −quantized 96.67 exactly as measured); declared-red manifest for the digest/golden set (GOLDEN_AGGREGATE, Arc-B digests, `test_e1_cold_rail` re-pin deferred to close) | subagent | Sonnet 5 | **CPU↔CUDA lockstep tol 0 same patch** (per-call + resident + `cuda_kick_check` + `cuda_eos_step_check` + `cuda_s8a_check`); new unit oracles green; **explicit `test_air_boundary.py:820` re-verify — RED = STOP (RISK-1)**; suite set-diff == manifest exactly | no |
| P-W2 | measurements + HUMAN-TEST brief | AFTER half of P-W0's comparisons; sub-ambient census + \|u\| vs own-c(T_abs) distribution on the venting bench (the D-1 data for Erik); acoustic-ratchet drift number (RISK-2); cold-rail window re-run (the −288.65 spiral should now warm instead); verify the T-overlay (T-key) actually SHOWS sub-ambient — if it clips at 0, extend the render mapping to display cold (render-only, exempt) so the HUMAN-TEST can see what it's testing; write the HUMAN-TEST brief | subagent | Sonnet 5 | all measurements committed as a dated capture doc; overlay shows cold; brief lists exactly what Erik plays | no |
| P-W3 | HUMAN-TEST + close | Build + push; **Erik plays**: breach venting cold look (rarefaction ~97 game-deg at the clamp), grenade compression warmth, fire sanity (hot rail faster — does feel survive?), cold-pocket persistence vs conduction relax; then D-1 presented with P-W2's numbers. On bless: ONE golden re-baseline with rationale doc, archive design+critique to `docs/archive/`, TODO/ledger update, merge --no-ff, branch/worktree cleanup, workspace flip to main | inline (Erik + orchestrator) | — | **HUMAN-TEST: Erik's eyes are the gate. NOTHING auto-merges.** | **YES — merge gate** |

## 7. Test inventory (recon-verified touch list)

- **Re-derive:** `test_p_e4_reversible_work.py` (bounds written for the
  relative law).
- **Explicit re-verify, red = stop:** `test_air_boundary.py:791-822`.
- **Digest re-pins, deferred to close (declared red through the arc):**
  `test_e1_cold_rail.py` (its scenario is the very thing we fix),
  `test_e1_hot_rail.py` if its digests move, GOLDEN_AGGREGATE + the six Arc-B
  digest reds (freshly re-baselined 2026-08-20, trustworthy: any flip is
  REAL behavioral change from this patch alone).
- **Parity sweeps at tol 0:** `cuda_eos_step_check.py`, `cuda_kick_check.py`,
  `cuda_s8a_check.py`, `test_cuda_p64_kick_compression.py` (NOTE: its PART 2
  carries a known PRE-EXISTING CPU↔GPU divergence, documented at the
  velocity-clamp arc and before — it stays a declared pre-existing red, not
  this arc's debt), `cuda_thermal_mass_eos_check.py`.
- **Behavior-adjacent, run + triage:** `test_thermal_mass_axis.py`,
  `test_p_e3_drag.py`, `test_velocity_clamp_property.py`,
  `test_destroy_wall_conserves_mass.py` (gate 6 reads
  `eth_compression_delta` — type-only, should hold).

## 8. Accepted gaps (decisions, not findings)

- **KE↔eth stays HALF-coupled** (energy-books §8, unchanged): the kick still
  mints KE with no eth debit. This patch adds the honest T-side of compression
  work; the kick-side debit remains the open half. Out of scope by the same
  reasoning that deferred it there.
- **Cap²-plane ambient floor kept** (D-1) — Erik decides on P-W2's data
  whether a future arc lets the cap follow T_abs down.
- **General-scale (s_eos ≠ 1) form unbuilt** (D-3) — guarded by assert.
- **The one-way acoustic ratchet** (RISK-2) — bounded ≤1 LSB per cell-cycle,
  one direction, measured at P-W2 and recorded; machinery to null it is
  refused until a measurement says it matters.

## 9. What Erik must know before playing (P-W3 brief seeds)

The ~97-game-deg figure is at the WORK CLAMP — typical rarefaction will be
milder; the T overlay is the instrument. The hot rail compounds slightly
faster (RISK-1 gate protects the tested scenario; feel is Erik's call). Cold
pockets now persist until conduction relaxes them — a breach's wake should
read cold for a while, then fade. If the cold feel is too strong or too weak,
the honest knobs are T_WORK_CLAMP (rate rail) and the P-E4 trust-gate dials —
NOT a return to the relative law.
