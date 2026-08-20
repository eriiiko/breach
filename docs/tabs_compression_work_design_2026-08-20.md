# T_abs compression work — design (v2, 2026-08-20)

**Arc:** `tabs-compression-work` (TODO ★ item 2). **Origin ruling:** RULING R1
(Erik, 2026-08-17), `docs/archive/energy_transport_design_2026-08-16.md` §2.9 —
this patch is that ruling executed, with its own critique round and its own
HUMAN-TEST. **Feel-adjacent: nothing merges before Erik plays.**

**v2 (this version): survived a 3-lens adversarial critique + a full test-blast-
radius sweep.** Changes vs v1: int64 `t_abs` + the floored `t_amb_q` fold pinned
verbatim (A1/A2); complete ABI list (A3/C4/C5); D-3 guard rebuilt Release-live
with the `T_MIN > −T_AMB_K` companion (A4/A5/B-F5); `test_e1_hot_rail:220`
moved to the STOP set and `test_e1_cold_rail` out of the manifest (B-F1/F2/C1);
RISK-1 re-founded on the ambient-runaway entry point (B-F3); RISK-2 split into
two residuals with a bound gate (B-F4); the T_MIN pressure-collapse consequence
named (B-F7); vac/ring wipe channels named (B-F8); §8 KE↔eth bullet rebuilt
(B-F9); dial-derived clamp oracle (B-F10); cap-ceiling side named (B-F11); the
trust-gate cold-ring correction (B-F12); overlay plan replaced (B-F13/C12);
P-W1 split into ABI-plumb + law-flip (C14); P-W0 becomes a P-E0-style
instruments patch with a baseline red-set artifact (C2/C11); close ritual
completed (C13); conduction timescale numbers (B-F6); the ~97 figure framed as
the scheme's number (B-F16).

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
  (w = 0.5) is 290 K → 193.3 K, i.e. **−96.67 game-deg** — the figure R1
  blessed as the intended feel. (This is the scheme's number: exact adiabatic
  at w = 0.5 would be ~−114; the ~15% gap is the first-order form,
  deliberately — §2.7's "reversibility, not adiabatic fidelity" framing
  carries over unchanged.)
- **Honest expectation for the feel (critique B-F12):** the P-E4 trust gate
  fades k to zero below n = n_work_ref/2 = 0.125·N_amb and to full trust only
  at 0.25·N_amb. A hard-vented breach core is thinner than that — so the cold
  appears **as a ring where the wake still holds ≥12.5% of ambient density,
  not in the vent core itself**. That is correct bookkeeping (no temperature
  unbacked by energy), and it is what Erik should expect to see at P-W3.
  `n_work_ref` is the knob if he wants the cold to reach deeper.

## 2. The law (the whole change)

The stored convention of `temperature[]` — **ambient-relative, 0 = ambient** —
is LOAD-BEARING and does not change: `eos_energy_books_sum` accumulates
`n·T_rel` with no offset (`eos_solver.cpp:234-257`; gate 6 of
`test_destroy_wall_conserves_mass.py` restates the rule), the vacuum/ring wipe
prices `T<0` pins as creation (`temperature_solver.cpp:117-137`), render and
ignition read relative T. **Only the interior of the 4c arithmetic goes
absolute**, then shifts back out.

Pinned arithmetic — identical transcription in all three twins:

```c
// fold (host, once per tick) — the A7-floored form, VERBATIM; the CPU live
// path REUSES the existing local at eos_solver.cpp:372, no second fold:
const q16 t_amb_q = std::max<q16>(1, quantize((double)T_AMB_K));

// per cell, inside the existing branch structure:
const int64_t t_abs = (int64_t)temperature[i] + (int64_t)t_amb_q;   // int64: NOT q16
// compression (k < 0): same structure as HEAD, operand t_abs.
//   ((k_signed * t_abs) >> 16) is bit-identical to mul_q16's floor-toward
//   -inf convention; |dT| <= w*t_abs <= 5.4e8 — int32-safe narrow.
const q16 dT = (q16)(((int64_t)k_signed * t_abs) >> 16);
t_new = sat_add_q16(temperature[i], (q16)(-(int64_t)dT));           // = T + w*(T+290), heating rounds UP
// expansion (k >= 0, k==0 pinned here — exact identity, D-4):
//   subtract BEFORE the narrow.
t_new = (q16)(floordiv_q(t_abs << 16, (int64_t)FP_ONE + (int64_t)w)
              - (int64_t)t_amb_q);
```

Everything around it is UNCHANGED: the div(u_new) stencil, γ−1 and dt folds,
the trust-gate fade, the single-compare ±T_WORK_CLAMP rail with
`work_clamp_hits`, the T_MIN floor (`energy_floor_hits`) and T_MAX_PHYS
ceiling (`t_max_phys_hits`), the `eth_compression_delta` bracket, the skip set,
and the counter semantics (D-5).

Q16 facts (critique-verified):
- `t_abs` MUST be int64 (A1): `temperature[i] + t_amb_q` in int32 overflows at
  T > 32477 real — reachable because `T_MAX_PHYS` is a Python-writable dial
  with no entry-side rail. Overflow is UB on host vs wrap on device = a
  determinism break, not just a wrong value.
- Headroom: t_abs ≤ 16290·65536 ≈ 1.07e9; `<<16` ≈ 7.0e13 ≪ int64. The
  quotient ≤ 1.07e9 < INT32_MAX; the subtract-before-narrow keeps the
  dial-change class safe.
- **The §2.7 sub-ambient mint hazard dissolves structurally**: t_abs ≥ 1 raw
  for all T ≥ T_MIN = −289, so the expansion numerator is non-negative.
  `floordiv_q` is KEPT (shared idiom, zero cost, robust if T_MIN moves — and
  the guard below makes T_MIN moving past −T_AMB_K loud).
- **Reversibility, proven exactly (lens A):** with t_abs > 0, compression is
  `C(a) = ceil(a(1+w))` (mul_q16 floors a negative dT ⇒ heating rounds UP) and
  expansion is `E(a) = floor(a/(1+w))`. Then `E(C(a)) = a` **exactly**, and
  `C(E(a)) ∈ {a, a−1}` — compress-then-expand residual 0, expand-then-compress
  ≤1 raw count one-way. The +t_amb shift cancels across any cycle. This is the
  SAME structure `test_p_e4_reversible_work.py:159-177` already asserts at the
  clamp, so **those two at-clamp assertions must stay green through P-W1b — a
  red there is a real transcription bug and a STOP**, not re-derivation noise.
  Per-branch note for P-W2's readers: the biases are NOT one-directional per
  branch (compression rounds up ≤1 count/cell/tick) — only the composed cycle
  is one-way; a compression-dominated bench legitimately reads a
  rounding-positive `eth_compression_delta` without being a mint.
- The §2.7 "compression branch verbatim" promise was scoped to the
  energy-books arc and is **deliberately revoked here** — changing that
  operand is this patch's entire point.

## 3. What changes physically (named before measured)

- **Sub-ambient inversion fixed**: compression WARMS cold gas. The cold-rail
  engine dies structurally, not just operationally-via-k_drag.
- **Ambient air participates**: T=0 stops being a fixed point of 4c. The two
  residual channels this opens (RISK-2, split per critique B-F4):
  1. the **integer LSB ratchet** — ≤1 raw count per expand-compress cycle,
     one-way down, bounded;
  2. the **O(k²) shape-asymmetry drift** — for asymmetric acoustic cycles a
     proportional term ≈ ½(Σ_c k² − Σ_e k²)·(T+290) survives (the archived
     §2.7 names it; it multiplied T_rel≈0 before and 290 now). Steepened
     acoustics make Σk²_c > Σk²_e generically; a mild standing mode estimate
     gives **order game-degrees per 1000 ticks**, i.e. 10⁴× the ratchet.
     Sign set by cycle shape. **This gets a BOUND GATE at P-W2** (quiet-room
     scenario: max |T_rel| over open cells ≤ 10 game-deg over 2000 ticks,
     provisional bound, tightened from the P-W0 baseline + P-W2 measurement),
     not a record-only number. Exceeding the bound is a STOP.
- **Hot side, honestly re-founded (B-F3):** the old ×1.5-at-the-rail
  compounding becomes ×(1.5 + 145/T) for hot cells — but the NEW entry point
  is from ambient itself: sustained rail-rate compression gives
  T_k = 290·(1.5^k − 1), reaching T_MAX_PHYS in **~10 consecutive rail
  ticks** where before a hot seed was required. The quasi-static estimate for
  the rush-in scenario (`test_air_boundary` gate 2) is benign — adiabatic fill
  settles ≈ +272 game-deg interior, three orders under the ceiling; the risk
  is the converging-front transient. Note gate 2 ran at T ≡ 0 on HEAD, so its
  `t_max_phys_hits == 0` was **vacuous until now** — this patch is the first
  time that assertion tests anything. Pre-agreed decision rule: expected
  outcome is interior warming O(+200..400 game-deg) with hits == 0; a red on
  `t_max_phys_hits` means the clamp ratchet, and the levers are
  `T_WORK_CLAMP` / `n_work_ref` — **RED = STOP, never a re-pin.**
- **T_MIN pressure collapse — the genuinely new hazard (B-F7).** p* is
  `C·N·(T + T_AMB_K)` (`eos_solver.h:60`): a cell driven to the floor
  (t_abs = 1 K) loses ×290 of pressure beyond its N loss, in
  ln(290)/ln(1.5) ≈ **14 rail ticks ≈ 0.6 s** of sustained expansion — well
  within a venting jet's lifetime. A near-zero-p* cell beside normal cells is
  a giant ∇P kick → velocity spike → pile-up: a **pressure** route back into
  the flash class the velocity-clamp arc fought, NOT covered by D-1's Mach
  argument. Named, and instrumented: P-W2 measures min P / max |∇P| /
  `u_clamp_hits` before/after on the venting bench; "breach-mouth flashes" is
  on Erik's play list; and D-6 keeps the T_MIN value itself on the table.
- **Cap plane goes live on BOTH sides (B-F11 + D-1):** the floor side is D-1;
  the ceiling side is that compression-warmed air raises its own cap
  (T = 700 ⇒ cap ≈ 554 m/s), and `u_max_hits` — structurally zero today —
  becomes reachable at T ≳ 2930. `n_sub` stays insensitive below T ≈ 2930
  (it derives from max(c_LOCAL, u_max) with u_max = 1000) — stated so nobody
  re-derives it. `u_max_hits`/`u_clamp_hits` join the P-W0/P-W2 rows.
- **Vac/ring wipe channels flip two-way (B-F8):** `e_vac_wipe_sum` /
  `e_ring_pin_sum` are signed by construction and have only ever destroyed
  (0 sub-ambient open cells in 4.86M cell-snapshots). Sub-ambient gas
  advected into a wiped/pinned cell now CREATES energy in the books —
  bounded by ≤ 290·N_vented per tick, an accepted gap once priced at P-W2,
  but named here rather than inherited silently.
- **Cold pockets relax on two clocks (B-F6):** advective mixing (fast — the
  refill flow itself) and conduction (slow floor: harmonic-mean κ gives
  τ ≈ 43 s against a hull face, ≈ 21 s laterally through air). A breach wake
  reads cold, fades quickly where air flows back, lingers tens of seconds in
  stagnant corners. §9 phrases the feel promise this way.
- **eth_compression_delta changes regime**: ambient cells contribute for the
  first time; venting reads strongly negative, compression positive. No gate
  pins it (verified); recorded storm-ledger baselines regenerate at P-W2.

## 4. Design decisions

**D-1 — The cap²-plane ambient floor STAYS this arc** (`eos_solver.cpp:434` +
its byte-identical host twin `cuda_eos_step.cu:198`; both untouched).
Reasons, strengthened by critique: (a) scope — R1 ruled the work form; the
cap law was ruled in the velocity-clamp arc WITH this floor explicit;
(b) venting survival — c(T_abs = 1 K) ≈ 17.6 m/s; a c(T_abs) cap would
throttle breach venting to a crawl, and the k_drag = 10 probe showed killing
venting is a catastrophic feel failure; (c) numerically the floor re-opens
nothing: at shipped scale even the ambient cap already runs ~4.7 tiles/substep
(the audit's "~14× over resolvable Courant during blasts" is owned by
N_SUB_MAX, ruled to stay 8) — a cold pocket at the ambient cap is
indistinguishable from every other cell. **The real re-opened class is the
T_MIN pressure collapse (§3), which is measured and presented instead.**
P-W2's census (sub-ambient count, |u|/c_own percentiles, min P) goes to Erik
at P-W3; lowering the floor is a future Erik decision made on data.

**D-2 — No dial. The law is replaced outright.** R1 ruled the honest form IS
the fix; a toggle would keep the inverted law alive as a maintenance surface.
A/B for the HUMAN-TEST is build-vs-main.

**D-3 — Simple form + a RELEASE-LIVE guard (rebuilt per A4/A5/B-F5).**
`t_abs = T + t_amb_q` is honest only while `S_EOS ≡ 1.0` (value-frozen at
P-K3) AND `T_MIN > −T_AMB_K` (else the compression branch silently
re-inverts — the exact defect this arc kills). Both are Python-writable
dials, and `assert()` is dead in the Release builds every gate and every
play session uses. The guard is therefore: **one always-compiled host-side
check per tick at the two live scalar-fold sites** — `EOSSolver::step()`
beside the `s_eos_q` fold (`eos_solver.cpp:381`) and the CUDA prestage
(`cuda_eos_step.cu`, same spot its own folds live) — throwing a named error
(`"T_abs compression work requires S_EOS==1 and T_MIN>-T_AMB_K; see
tabs_compression_work_design_2026-08-20.md"`) when violated. Two integer
compares per tick; loud in every build. The test-only reference twin is
exempt by contract (it replays step() under the dials it is handed).
ACCEPTED GAP: the general-scale (s_eos ≠ 1) form is deliberately unbuilt —
the forward half exists elsewhere (`eos_solver.cpp:424` idiom) but threading
s_eos through three twins for a frozen dial is machinery refused; the guard
makes the freeze loud instead.

**D-4 — k==0 stays pinned to the expansion branch** (exact identity in the
new form for ALL int32 T incl. negatives: positive divisor, zero remainder —
lens A verified). The quiescent-map byte-exactness tests
(`test_air_boundary` gate 1, `test_level_water_physics:128`) are direct
consumers of this identity: **a gate-1 red means the k==0 identity is broken
— STOP.**

**D-5 — Counter semantics unchanged.** Same three rails, same counters, same
single-compare clamp form. Values move; meanings don't.

**D-6 — T_MIN = −289 is kept this arc, and put on Erik's table at P-W3.**
Once the floor is reachable (§3's pressure collapse), its VALUE is a live
dial: a floor at t_abs = 30 K would cap the p* collapse at ×10 instead of
×290 — a cheaper lever than touching the cap plane. Changing it is
feel+physics and belongs to Erik with P-W2's measurements in hand, not to
this design as a default.

**D-7 — Cold must be visible and the instruments are pinned (B-F13 + C12).**
The heat overlay is additive-emissive and structurally cannot show cold
(additive blending only adds light). P-W2 therefore ships two render-only
instruments: (1) **wire the existing, tested, unwired
`renderer/hover_readout.py`** (per-tile T in game-deg + pseudo-Kelvin —
exactly the readout Erik asked for in TODO's fire-tuning notes; taste-free);
(2) a **minimal cold tier**: a signed/diverging blue ramp for T_rel < 0
reusing the pressure overlay's colormap machinery, alpha-blended UNDER the
additive heat pass, on the same toggle — placeholder constants, explicitly
provisional; Erik judges the look at P-W3 along with the physics. Kelvin-frame
note for the brief: the canonical render map (K = 293 + 3·T_game) and the
EOS's 290-frame diverge wildly below ambient (−96.67 game-deg is 193 K in the
EOS frame but "3 K" through the render map) — the readout shows game-deg;
**game-deg is the honest number sub-ambient** and the brief says so.

## 5. Twin sites and the COMPLETE ABI edit (critique-completed)

Three transcriptions of the 4c arithmetic (verified, no fourth):

1. `cpp/src/eos_solver.cpp:984-1022` — live `step()`. Reuses the in-scope
   `t_amb_q` local (`:372`); no new fold.
2. `cpp/src/eos_solver.cpp:~1786-1829` — `eos_kick_compression_reference`.
   Gains a **`float t_amb_k`** parameter (it folds its scalars from floats,
   `:1786` precedent) folded with the A7-floored expression verbatim.
3. `cpp/src/cuda_kick_compression.cu:268-338` — `compression_kernel`. Gains a
   `q16 t_amb_q` parameter from `KickScalarFolds`.

Full plumbing list (every site, so P-W1a is mechanical):
- `KickScalarFolds` (`cuda_resident.h:114-138`): new `t_amb_q` field.
- `kick_scalar_folds()` factory (`cuda_resident.h:139-143`, def
  `cuda_kick_compression.cu:345-378`): new `float t_amb_k` param; folds
  `max<q16>(1, quantize(t_amb_k))` — ONE transcription of the fold on the
  CUDA side, matching how it already folds t_min/work_clamp.
- Both factory callers: `cuda_kick_compression.cu:445` (per-call entry) and
  `cuda_eos_resident.cu:699-706` (resident) — pass the solver's `T_AMB_K`.
- The public per-call C API `eos_kick_compression`
  (`cuda_kick_compression.h:60-92`, def `cuda_kick_compression.cu:421-436`):
  new `float t_amb_k`, **inserted BEFORE the trailing defaulted pointer params**
  (`is_ambient`, `sponge_udamp`, `thermal_solid`) — position pinned.
- Its non-resident caller `cuda_eos_step.cu:634-650`: passes `solver.T_AMB_K`.
- `kick_compression_launch_resident` takes `const KickScalarFolds&` — **no
  signature change** (stated so nobody hunts for one).
- Reference twin decl `eos_solver.h:674`; BOTH pybind entries in
  `bindings.cpp` (`eos_kick_compression_ref` AND `cuda_eos_kick_compression`,
  ~:1117-1189): expose `t_amb_k` as a KEYWORD arg, default 290.0.
- Python callers (the real list — `test_cuda_p64_kick_compression.py` is a
  subprocess wrapper, NOT a direct caller): `tests/cuda_kick_check.py:144,148`
  — **`:481`'s dial-threading block MUST pass `t_amb_k=float(eos.T_AMB_K)`**
  (a silent 290 default there would compare a defaulted reference against a
  solver-valued device path — the exact quiet hole the lockstep gate exists
  to close); `tests/cuda_thermal_mass_eos_check.py:191-197`;
  `tests/test_p_e3_drag.py:81,188,198,246`;
  `tests/test_p_e4_reversible_work.py:116`;
  `tests/test_velocity_clamp_property.py:111`;
  `tests/test_thermal_mass_axis.py:645,647,694` — **positional call sites:
  convert to keywords or audit the insertion point explicitly.**
- D2H counter ABI (`counters_out[9]`): UNTOUCHED — no new counters.

## 6. Patch plan

Merge semantics: green gates commit to the arc branch; ONE merge to main after
the HUMAN-TEST. Memory checkpoint at every boundary. Expected-red manifest
maintained per rung against P-W0's baseline artifact — a declared red is the
rung's debt; any OTHER red is a stop. Golden re-baseline happens ONCE, at
close, after Erik's bless, with a written rationale doc (standing ruling
re-confirmed 2026-08-20); scenario-expectation re-pins (e.g. b6 latency ticks)
are documented individually in that same rationale doc — they are re-pins with
measured rationale, not rides on the digest event.

| # | patch | contents | mode | tier | gate | HUMAN-TEST |
|---|---|---|---|---|---|---|
| P-W0 | instruments + baselines (P-E0 pattern: instruments land AHEAD of the law) | (1) **Baseline red-set artifact**: full `pytest tests -q` on the untouched branch — red names + failed/passed/**skipped** counts + CUDA build stamp (rebuild first per `cpp/build_cuda*.bat`; record skip count so a vacuous green is detectable) — committed as the dated baseline doc every later set-diff cites; (2) **quiet-room drift instrument**: new deterministic capture script (`tools/quiet_room_drift.py`): sealed 28×28 interior box (the `_ambient_gmap` recipe), +0.1 atm Gaussian pressure seed at centre, 2000 ticks, per-tick `eos_energy_books_sum`, `eth_compression_delta`, max\|T_rel\|, `t_min_gas`, rail counters; (3) **`--mach-census` mode** on `tools/analyze_blowup_dump.py`: sub-ambient open-cell count, \|u\|/c_own percentiles (c_own = 300·√(t_abs/290)), min P, max \|∇P\| proxy; (4) **BEFORE captures**, commands pinned: storm-ledger battery (`conda run -n data python tools/storm_ledger.py --ticks 4800 --damp 0.005 --pf1b ...`, the P-E0 flag set), cold-rail window via `tools/bench_two_room.py::run_bench` (WINDOW dials: PF1B + k_wind_strip=0.5, damp 0.005 — NOT via the cold-rail test, which pins nothing), hot-rail `test_e1_hot_rail.run_scenario(**HOT)` measured FRESH (docstring numbers are stale pre-energy-books), ambient gate-2 counters (all trivially 0 today — recorded so the AFTER is interpretable), quiet-room baseline, `u_max_hits`/`u_clamp_hits`/`e_vac_wipe_sum`/`e_ring_pin_sum` rows | subagent | Sonnet 5, medium | instruments deterministic (run-twice identical) + committed; captures committed as a dated capture doc; NO sim-code changes (tools/tests only) | no |
| P-W1a | ABI plumb, no-op | The complete §5 list: `t_amb_k`/`t_amb_q` threaded through struct, factory, kernel signature, C API, both pybind entries, reference twin, all Python callers (keyword conversion at the positional sites) — **used NOWHERE in arithmetic** | subagent | Sonnet 5, low | **full-suite set-diff vs P-W0 baseline == EMPTY; golden aggregate unchanged; CUDA build green** — provably digest-neutral | no |
| P-W1b | the law | §2's arithmetic in all three twins + D-3's Release-live guard (both fold sites) + **re-derive `cuda_kick_check.py`'s T_MIN rail forcer** — under the new law the floor is crossed by EXPANSION at the boundary cell (seed T = −289.0, positive divergence: t_abs 65536 → floordiv → 43690 → T_rel −289.33, BELOW t_min_q → floor fires; the old compression forcer warms instead and the coverage-hole gate would red) — mechanism pinned here, transcription is the agent's; + **rewrite `test_p_e4_reversible_work.py`**: at-clamp exactness assertions UNCHANGED-and-must-stay-green (§2's proof — a red is a STOP), below-clamp bound re-keyed on \|T0_raw + t_amb_q\|, probe temps re-justified (the −100 rationale is void: −200·1.5 crossed T_MIN under the old law, (−200+290)·1.5−290 = −155 does not), asymmetric-cycle analytic figures re-derived on the (T+290) base; + **the clamp oracle, dial-derived (B-F10)**: `expected = floordiv(t_amb_q<<16, FP_ONE + quantize(T_WORK_CLAMP)) − t_amb_q; assert t_new == expected` with the shipped-default sanity comment (−6,335,147 raw = −96.6666) — tuning T_WORK_CLAMP later must not red the oracle; + finalize the red manifest per the classification rules below | subagent | Sonnet 5, high | **CPU↔CUDA lockstep tol 0 same patch**: `cuda_kick_check`, `cuda_eos_step_check`, `cuda_s8a_check` (resident, live A/B), `cuda_ambient_check`, `cuda_bulk_flux_check` (its PART 3 seeds a −120 sub-ambient pocket — the exact inverted regime), `cuda_thermal_mass_eos_check`, `cuda_thermal_mass_check` (`test_cuda_p64_kick_compression` PART 2 stays a declared PRE-EXISTING red — not this arc's debt); **STOP set green** (below); suite set-diff vs P-W0 baseline == finalized manifest EXACTLY; a one-line mechanism note per manifest entry | no |
| P-W2 | measurements + bound gates + HUMAN-TEST instruments | AFTER halves of every P-W0 row; **mach-census on a fresh venting-bench recorder capture** (D-1's data: sub-ambient count, \|u\|/c_own, min P, max \|∇P\|, `u_clamp_hits` — B-F7's flash-route check); **quiet-room BOUND GATE**: max\|T_rel\| ≤ 10 game-deg over 2000 ticks (provisional; tightened against the measurement — exceeding it is a STOP, this channel must be able to fail); cold-rail window re-run (the −288.65 spiral should now warm); vac/ring creation-channel pricing; **D-7's two render instruments** (hover readout wired; cold tier under the heat overlay); write the HUMAN-TEST brief (§9 content + the numbers) | subagent | Sonnet 5, medium | all measurements committed as a dated capture doc; bound gate green; overlay shows cold on the venting bench; brief committed | no |
| P-W3 | HUMAN-TEST + close | Build + push; **Erik plays**: breach venting cold (RING around the vent, not the core — §1's honest expectation), how long cold lingers (advective fade fast, conduction floor 21–43 s), **breach-mouth flashes** (B-F7's route), grenade compression warmth, fire sanity (hot-rail entry from ambient — does feel survive?), the cold overlay + hover readout look; then **two decisions on P-W2's data: D-1 (cap floor) and D-6 (T_MIN value)**. On bless: ONE golden re-baseline + rationale doc (incl. individually-documented scenario re-pins: b6 latency ticks, gate-3 reflection ratio if moved, b5 airlock bands if moved), archive design+critiques to `docs/archive/`, TODO/ledger update, **tag** (`tabs-compression-work-close` — precedent `velocity-clamp-close`), **egregore-collect-transcripts**, merge --no-ff, branch/worktree cleanup, workspace flip to main | inline (Erik + orchestrator) | — | **HUMAN-TEST: Erik's eyes are the gate. NOTHING auto-merges.** | **YES — merge gate** |

**Red-classification rules for P-W1b's manifest** (pre-agreed so triage is
mechanical, not judgment):
- **STOP set (must stay green; a red halts the arc):**
  `test_air_boundary.py:820` (gate 2 rails) and gate 1 `:767-787` (the k==0
  identity, D-4); `test_e1_hot_rail.py:220` (`t_max_phys_hits == 0` — a
  strict property gate on the compounding scenario, NOT a digest;
  its docstring's protection — the trust gate zeroing w as the pocket's N
  collapses — is unchanged by this patch, which is why green is the honest
  expectation); `test_e1_hot_rail.py:192-206` + `test_thermal_mass_axis.py:577`
  (strict transport-books gates); `test_p_e4_reversible_work.py`'s two
  at-clamp exactness tests (§2's proof); every tol-0 parity script above;
  `test_destroy_wall_conserves_mass.py` gate 6 (the books-convention coupling);
  the Arc-B dormancy trio `test_b1_signal_bus`/`test_b2_nodes` inline digests
  (physics=None — a flip there means a non-physics leak).
- **EXPECTED set (declared red until the close re-baseline):** the golden
  aggregate's 12 importers + `test_w6_armory:583`; `test_b6_logic_golden`
  (inline golden `:83` + latency pins `:92-94` — physics-live).
- **GRAY set (measure, then classify with a written mechanism note each):**
  air gate 3 reflection ratio (ambient cells' new T-evolution enters the
  metric); `test_level_water_physics.py:128` (should hold: 4c feeds NEXT
  tick's p*, and tick-1 atmosphere predates 4c's first write — if it reds,
  find out why before declaring); `test_wall_failure` pop tests; b5 airlock
  band pins; `test_velocity_clamp_property.py` gate 3 (its 1.5× within-tick
  slack vs stronger T evolution); `test_fire_heat_source.py:422` strict xfail
  (XPASS risk if warmer air ignites the plank — an XPASS is a finding for
  Erik, not silently un-xfailed); `test_p_e3_drag.py` near-T_MAX cases
  (earlier clipping shifts the drag identity's terms);
  `test_thermal_mass_axis.py:651`'s non-vacuous control. An unexplained red
  anywhere is a STOP.
- `test_e1_cold_rail.py` is in NO set: it is self-A/B and cannot red from a
  law change. The window scenario's new behavior is P-W2's capture, not a
  manifest row.

## 7. Accepted gaps (decisions, not findings)

- **KE↔eth stays HALF-coupled — argument rebuilt (B-F9), same decision.** The
  kick still mints KE with no eth debit, and 4c's work term now runs in every
  gas cell with no momentum counterparty — the two unbooked halves are
  same-signed during a blast. What makes the deferral still legitimate: the
  per-cell-tick magnitude is exactly the §2.4 bound (γ−1)·|div·dt|·290·N —
  previously foregone, now realized — it **cancels over closed acoustic
  cycles by the reversibility proof**, and accumulates only under sustained
  one-sided divergence, where it is instrumented (`eth_compression_delta` +
  P-W2's bound gate). The kick-side debit remains the open half, unchanged.
- **Cap²-plane ambient floor kept** (D-1) — Erik decides on data at P-W3.
- **T_MIN = −289 kept** (D-6) — Erik decides on data at P-W3.
- **General-scale (s_eos ≠ 1) form unbuilt** (D-3) — guarded loud, Release-live.
- **Two-way vac/ring channels** (§3) — priced at P-W2, bounded by vent traffic.
- **The acoustic residuals** (§3 RISK-2) — the ratchet is bounded ≤1 LSB per
  cell-cycle; the shape-asymmetry drift gets a bound gate, not machinery.

## 8. What Erik must know before playing (P-W3 brief seeds)

The ~97-game-deg figure is at the WORK CLAMP and is the scheme's number
(exact adiabatic would be ~114; the ~15% first-order gap is deliberate).
Typical rarefaction is milder, and the cold reads as a **ring** where the
wake keeps ≥12.5% ambient density — the hardest-vented core is faded out by
the trust gate (correct bookkeeping; `n_work_ref` is the reach-deeper knob).
Cold fades fast where air flows back and lingers 21–43 s in stagnant corners
(conduction's floor). Watch for breach-mouth flashes — the T_MIN pressure
collapse is this patch's one genuinely new hazard route, measured at P-W2,
with D-6 (raise the T_MIN floor) as the cheap lever if it bites. The hot rail
now has an entry from ambient air (10 rail-ticks to ceiling) — the gates held
green through P-W1b, but fire feel is the eyes' call. The honest knobs if the
cold feel is wrong: `T_WORK_CLAMP` (rate rail), `n_work_ref` (reach), D-6's
`T_MIN` (depth) — NOT a return to the relative law. Sub-ambient, read
game-deg on the hover readout; the render Kelvin map is misleading below
ambient (frame note in D-7).
