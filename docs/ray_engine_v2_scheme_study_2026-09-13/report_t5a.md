# T5a — the flip's two provably-neutral widenings

> **Written incrementally** (the arc's standing rule: a kill must cost no
> resume). Branch `12-t5a-neutral-widenings`, cut from `fire-12` at `c8e9a17`.
>
> **This patch changes no behaviour.** It lands two *structural* widenings so
> that T5b — the flip proper, which Erik plays — can move the two values as
> one-line edits with the arithmetic already correct underneath them.
>
> **Depends on**: `report_t2.md` §4.5, §5.3, §6, §10.2 (D1);
> `report_t3.md` §6.3, §7.5, §8 q8 (D2);
> `thermal_model_v2_design_2026-09-19.md` §3.2, §4 ledger item 4, R13.

---

## 0. Status — and the gate

- [x] D1 — the gas conduction structural fix (CPU + CUDA)
- [x] D2 — the `H_fuel` schema widening (CPU + CUDA + config)
- [x] the bit-identity proofs, reproduced not inherited
- [x] the negative controls (break it, watch it go red)
- [x] the gate: suite, goldens, `git status`

**Nothing in this patch changes a number the engine produces.** `c_v` is still
`1.0`; `H_fuel` is still `4.0`. What changed is that the arithmetic underneath
both is now correct, so T5b can move them as one-line edits.

**The gate, at `52d5772`:**

| | |
|---|---|
| suite | **2557 passed, 4 skipped, 4 xfailed, 0 failed** — identical to the pre-patch baseline, CPU build **and** CUDA build present (this box has an RTX 3070, so the 24 CUDA gates RUN rather than skip) |
| `GOLDEN_AGGREGATE` | **unmoved** — `167b96bddfe37c0d256afed4d3b9271371fcaf3edd7557e02cf685a17208953f` |
| per-field digest | all **780** `(field, tick)` hashes **byte-identical** to the pre-patch dump |
| `DIGEST_SPEC_VERSION` | **5**, untouched |
| CUDA conduction lockstep | **tol-0** — 121 configs + a 120-tick trajectory + the 30-tick engine A/B, `P66_RESULT: PASS` |
| `git status` | clean; every path staged explicitly, never `git add -A` |

Two commits, each independently gated and revertible:

| commit | what |
|---|---|
| `2ac0f83` | D1 — gas conduction converts heat counts to books |
| `52d5772` | D2 — `H_fuel` gains its shift companion |

---

## 1. D1 — the gas conduction structural fix

### 1.1 The defect, and why nothing could see it

`temperature_solver.cpp` Pass 2 booked an accountable gas cell's four-face sum
`de` straight into the gas books, on the claim (its own comment) that *"`de` is
already in the books' currency — a face quantum is `|ΔT|·C` with `C = N·c_v`,
and the books' capacity IS N"*. That sentence names the two capacities in one
breath and then equates them. A face quantum is priced at `C = N·c_v`; a book
count is `N·T`. They differ by exactly `c_v`, so the gas received `c_v ×` what
the faces delivered.

At the shipped `c_v = 1` the two coincide **by accident**, which is why the
defect survived arc #54, every golden and the whole closure identity. T2 §3.1
measured the ratio on a live sim as `c_v` to eight figures: at the derived
`c_v = 0.0076849`, **99.23 % of every joule conducted into air was destroyed at
the face**.

### 1.2 The replacement, and the one place it departs from T2

T2 §4.5 specifies `de_books = floordiv_q(de · N, cap_used)` and `de_full`
against `cap_real`. The principle is T2 §10.2's: convert by inverting **the very
capacity the face quantum was built from**, never by a second representation of
`c_v` — because `c_v` has two integer representations in this TU (`c_v_q`,
Q16.16, which built the capacity; `recip_cv`, Q.32, which Pass 1 divides by) and
they agree only at `c_v == 1`.

Landed as specified, with **one substitution forced by a bound T2 got wrong**
(§6.1) — the product is taken through a new shared helper,
`conduction::muldiv_floor_q(a, b, d)`, which divides first:

    a = q·d + r  with 0 <= r < d      ->      floor(a·b/d) == q·b + floor(r·b/d)

an exact identity for either sign of `a`, with no intermediate wider than the
result. Sites:

| # | site | what |
|---|---|---|
| 1 | `cpp/src/temperature_solver.h` (`conduction::`) | **NEW** `muldiv_floor_q` — the shared FP_HD helper, with its exactness identity and its bound |
| 2 | `cpp/src/temperature_solver.cpp` Pass 2 gas branch | the conversion itself (`de_books` at `cap_used`, `de_full` at `cap_real`, `e_cond_cap_sum` their difference) |
| 3 | `cpp/src/temperature_solver.cpp` Pass 2 comment block | the false currency claim, replaced by what it actually is + why the flip cannot be config-only |
| 4 | `cpp/src/cuda_temperature.cu` `temp_conduct` body | the identical twin |
| 5 | `cpp/src/cuda_temperature.cu` `temp_conduct` signature + launch | **two** new arguments, not T2's three (§6.2) |
| 6 | `cpp/src/temperature_solver.h` gas-side + counter doc blocks | the same false claim in prose, twice |
| 7 | `docs/…/currency_audit_t2.py` `report()` | T2's own instrument, which T2's own fix turns into a tautology (§6.3) |

`c_v` itself is untouched: `config.toml` still ships `1.0`.

### 1.3 The `N == 0` corner

`cap_real == 0` means the cell holds no gas: the real books delta is 0 and the
whole of `de` is the capacity floor's fiction. T2's pseudocode converts it
`<de / c_v via recip_cv>` — which contradicts its own §10.2 ruling (never a
second representation of `c_v`) and could not have compiled as written anyway,
since `recip_mul` narrows its first argument to `int32` and `de` is `int64`.
Landed instead as `muldiv_floor_q(de, FP_ONE, c_v_q)`: the same Q16.16 `c_v` the
capacity planes were built from, exact at `c_v == 1` for **every** `n_floor_q`
including 0.

### 1.4 It is not a no-op — the semantic proof

Bit-identity at `c_v = 1` proves the patch is safe. It cannot prove the patch
does anything, and this arc has twice been burned by exactly that. So the fix
was also measured at the value it exists for, on T2's own instrument:

| | `c_v = 1` (shipped) | `c_v = 0.0076849` — T2, BEFORE | `c_v = 0.0076849` — AFTER |
|---|---|---|---|
| solids lost | 6 374 688 counts | 59 392 | 83 200 |
| …destroyed by the endpoint divide | 35 931 (0.56 %) | 9 246 (15.6 %) | 34 474 (41.4 %) |
| delivered to the gas by the faces | 6 338 757 | 50 146 | 48 726 |
| **gas RECEIVED** | 6 338 757 | **385** | **48 724** |
| **received / delivered** | **1.00000000** | **0.00768487** | **0.99996993** |

and per cell, on the watched gas tile — the cleanest single statement of it:

| `c_v` | real heat capacity | gas ΔT over 40 ticks |
|---|---|---|
| 1.0 | 241.31 J/K | +8.944855 game-K |
| 0.0076849 | 241.31 J/K | **+8.946045 game-K** |

The same joules now land in the same real heat capacity and produce the same
temperature rise, whichever `c_v` the engine is dialled to — which is what it
means for `c_v` to be a unit conversion rather than a physical dial. Before the
fix that second row was 130× smaller.

The `c_v = 1` column of both tables is unchanged to the last digit, which is the
bit-identity requirement restated as a measurement.

## 2. D2 — the `H_fuel` schema widening

### 2.1 What was actually blocked

T3 §6.3 derives the plume's share of Huggett 1980's **4.831 MJ per unit N_O2**
at a 25/75 flame-to-surface feedback split (Drysdale ch. 5) and gets
**7.614e+06** counts. `H_fuel` entered the solver as `quantize((double)H_fuel)`
— a plain Q16.16 `int32`, ceiling **32 767.99998**. Measured: `quantize(7.614e6)`
is `498 991 104 000`, which **overflows int32 by 232×**.

So the derived value was never blocked by a tuning decision. It was
unrepresentable, and no amount of `config.toml` editing could have reached it.

### 2.2 The schema, carrying today's number

Split exactly as `H_BED_M`/`H_BED_SHIFT` already is —
`H_fuel = H_FUEL_M · 2^H_FUEL_SHIFT` — and shipped as `M = 4.0`, `SHIFT = 0`.
Fifteen files, all mechanical:

`config.toml` · `combustion.h` (the two fields + the derivation) · `combustion.cpp`
(mantissa quantize + the deposit) · `cuda_combustion.h`/`.cu` (host entry, kernel
signature, launch, body) · `bindings.cpp` (free-function arg, `py::arg`,
`def_readwrite`) · `physics_runner.py` (config bind + the CUDA call) ·
seven test modules · `tools/eos_p5_bake.py`.

`o2_potency` lands on the **mantissa**, exactly as it already does for `H_bed`
(the shift is a pure power of two and stays put), so its documented
"1.0 is byte-neutral" property survives unchanged.

The surface is **renamed**, not extended: leaving a key called `H_fuel` that
actually means "the mantissa" is the same one-constant-two-truths defect this
arc keeps finding (T2 §10.2 found it for `c_v` three weeks ago).

**Bit-identity**: 331 072 cases — 200 000 randomized `burn_dep` across eight
magnitudes and both signs, plus the entire `0 … 131 071` operating band
exhaustively — **0 mismatches** against the shipped expression. At `SHIFT = 0`
the shift and the clamp are both no-ops and the widen takes the
already-narrowed `int32`, so even a hypothetical `mul_q16` wrap is reproduced
rather than papered over.

### 2.3 A number T5b needs — the clamp is not decorative

The deposit now takes the shift in `int64` and clamps before the narrow. At
today's value that clamp can never bind. **At T3's derived value it binds early**:

    clamp binds at  burn_dep > INT32_MAX / (M · 2^SHIFT)

| config | `H_fuel` | clamp binds at `burn_dep` | in N_O2 per cell-tick |
|---|---|---|---|
| shipped `M = 4.0, SHIFT = 0` | 4 | 536 870 912 raw | 8192 |
| T3 derived `M = 14870, SHIFT = 9` | 7 613 440 | **282 raw** | **0.0043** |

Against the shipped `burn_rate = 0.018` N_O2/s at `dt = 1/60`, one claimant
demands ~20 raw counts per tick, so:

| claimants into one air cell | `burn_dep` | headroom |
|---|---|---|
| 1 | 20 raw | 14.1× |
| 4 | 80 raw | 3.5× |
| 9 | 180 raw | 1.6× |
| 21 | 420 raw | **CLAMPS** |

With `draw_r = 2` (the shipped P-O2b extended draw) a single air cell can carry
many claimants, so this is reachable rather than theoretical. Two consequences
for T5b, neither of them T5a's to decide:

1. Without T5a's clamp the narrow would **wrap** there — `mul_q16` casts to
   `int32` — turning the hottest cell in a fire into a negative deposit. That
   could not happen while the value was capped at 32 768; it can once the schema
   is wide. The clamp is load-bearing for the flip.
2. **The clamp is not counted.** It saturates silently (as `H_bed`'s twin has
   always done). The downstream `T_MAX_PHYS` rail *is* counted, but it sees an
   already-saturated deposit. If T5b lands the derived value, this wants either a
   counter or a headroom argument written down — flagged, not fixed here, because
   a new counter is neither D1 nor D2.

## 3. The bit-identity proofs

Reproduced, not inherited. Both are pure-integer models of the two expressions,
run against each other; Python's `//` is floor division, which is exactly
`fixedpoint::floordiv_q`.

### 3.1 D1 — 283 068 randomized triples, stratified

`(de, N, n_floor_q)` at `c_v_q = 65536`, stratified so no branch can hide, with
`de` spanning six magnitudes up to 2^40 in both signs:

| stratum | cases | `de_books` mismatches | `e_cond_cap_sum` mismatches |
|---|---|---|---|
| `N == 0` (`cap_real == 0`) | 70 879 | **0** | **0** |
| `0 < N < floor` (the floor binds) | 70 733 | **0** | **0** |
| `N >= floor` (the common case) | 70 727 | **0** | **0** |
| **`N` at/over the CAPACITY CEILING** | 70 729 | **70 083** | 0 |

The first three strata are the reachable engine, and they are **bit-identical in
both counters**. The fourth is §6.1's finding.

`n_floor_q` was drawn from `{0, 1, 655, 3276, 65536}`, so the `n_floor_q == 0`
case T2's `recip_cv` form needed is covered too.

Separately, `muldiv_floor_q` itself was checked against exact big-integer
`(a*b)//d` over **200 000** randomized triples: **0 mismatches**.

### 3.2 D2 — 331 072 cases

200 000 randomized `burn_dep` over eight magnitudes, both signs, plus the whole
`0 … 131 071` band exhaustively. **0 mismatches.**

## 4. The negative controls

Bit-identity at `c_v = 1` and a green suite prove the patch is *safe*. Neither
can prove it does anything, and this arc has twice found a structural invariant
blind to a semantic defect. So every gate was validated by breaking it.

| # | what was broken | result | caught by |
|---|---|---|---|
| A1 | D1's `de_books` divides by `cap_real` instead of `cap_used` (a plausible transcription slip; only observable where the **floor binds**) | RED | `test_b6_logic_golden` (2 tests). **NOT** the xarch golden — its scenario never binds the capacity floor |
| A2 | D1's `de_books` off by one raw count (the **common** path) | RED | xarch `GOLDEN_AGGREGATE` moves to `5b4e92aa…` |
| B | D1's **CUDA twin only**, off by one | RED, CPU golden intact | `test_cuda_p66_conduction` — **PART 3 only**. PARTS 1 and 2 passed: they run the direct binding with `gas_energy == nullptr`, so they never reach the accountable-gas branch at all |
| C | the semantic control — run the fix at the `c_v` it exists for | see §1.4 | `received/delivered` moves 0.00768 -> **0.99997** |
| D2a | `H_FUEL_SHIFT = 1` in `config.toml` (a **2× change in the gas-side heat yield**), verified live to reach the solver | **GREEN — nothing caught it** | §6.4 |
| D2b | D2's shift path off by one bit, engine only | RED | `test_thermal_mass_axis::test_combustion_deposit_converts_via_heat_inv_shift_on_a_thermal_solid` — one test, and it is the one that asserts the deposit against the solver's own declared constant |

A control that did NOT fire, and why it is not a gate gap: perturbing D2's
mantissa by **+1 LSB** is invisible, because `mul_q16` truncates and a per-tick
`burn_dep` of ~20 raw counts cannot carry that LSB across a 2^16 boundary. The
perturbation was below the arithmetic's own granularity — a badly chosen
control, not a missing gate. D2b is its properly-sized replacement.

## 5. The gate

Run on this branch with both backends built
(`cpp\build_cpu_home.bat`, then `cpp\build_cuda.bat`):

```
C:/Users/steen/anaconda3/python.exe -m pytest tests -q
  -> 2557 passed, 4 skipped, 4 xfailed, 3 warnings      (0 failed)

C:/Users/steen/anaconda3/python.exe tests/_xarch_perfield_digest.py
  -> aggregate digest = 167b96bddfe37c0d256afed4d3b9271371fcaf3edd7557e02cf685a17208953f
     matches golden   = True
     780/780 per-(field, tick) hashes byte-identical to the pre-patch dump

pytest tests/test_cuda_p66_conduction.py
  -> P66_RESULT: PASS   (121 configs + 120 ticks + engine A/B, tol 0)
```

The pre-patch baseline on this box was the same 2557/4/4/0, so the suite count
is a comparison rather than an assertion.

**A note on `tests/_xarch_perfield_DESKTOP-0E98HUV.txt`.** It is TRACKED, and the
committed copy is a stale 2026-07 artifact (`spec_v=1`, golden `98d3dd7e…`, 693
lines). Running the digest harness overwrites it, so every run above was followed
by `git checkout --` on that one path to keep `git status` clean. Re-baselining it
is not T5a's to do, but it is a tripwire for the next implementer: the file in
the tree does not describe the engine in the tree.


## 6. Findings — what T2's and T3's blast radius missed

### 6.1 T2's `de · N` product leaves int64 at N = 2 atm  *(the one that mattered)*

T2 §4.5 wrote the replacement as `floordiv_q(de * nb, cap_used)` and argued the
bound: *"with `|de| <= 4·2^31·cmin` and `cmin = cap_gas`, at the physical `c_v`
`cap_gas ≈ 504·N`, so `|de| ≲ 2^42` and `|de·nb| ≲ 2^60` — inside int64, but by
argument, and the argument has to be in the file."*

**The argument is evaluated at the wrong `c_v`.** T5a ships at `c_v = 1`, where
`cap_gas = N_raw`, not `504·N` — 130× larger, so the product is 130² ≈ 17 000×
looser than T2 assumed. Measured, with `g` bounded by the real `T_MAX_PHYS` span
(~2^30) rather than the full int32 range:

| N (atm) | `cap = N_raw·c_v` | bound on `de` | `de·nb` (T2's form) | int64? |
|---|---|---|---|---|
| 0.01 | 655 | 6.99e11 | 4.58e14 | yes |
| **1.00** | 65 536 | 6.99e13 | **4.59e18** | yes — 2.0× headroom |
| **2.00** | 131 072 | 1.40e14 | **1.83e19** | **NO** |
| 4.00 | 262 144 | 2.80e14 | 7.34e19 | NO |
| 4096 | 268 435 456 | 2.87e17 | 7.69e25 | NO |

(int64 max = 9.22e18.) Signed overflow is UB, and on this engine it is a
determinism break that **both backends would agree on**, so no CPU/CUDA parity
gate could see it — only a ledger could, and `e_gas_cond_sum` books whatever was
computed. Over the randomized sweep, 14 375 of 283 068 cases overflowed.

`conduction::muldiv_floor_q` takes the division first and is exact by the
identity `floor(a·b/d) = q·b + floor(r·b/d)` for `a = q·d + r`, `0 <= r < d`. Its
widest intermediate over the same sweep was **2^58**.

### 6.2 The CUDA kernel needs TWO new arguments, not three

T2 §5.3 says `temp_conduct` must gain `recip_cv`, `n_floor_q` and the N source.
With the conversion written as an inverse of `cap` (which is T2's own §10.2
ruling), **`n_floor_q` is never referenced** — the floor is already baked into
`cap_used`, which the kernel has. And `recip_cv` is the wrong constant *and* the
wrong type: `recip_mul` narrows its first argument to `int32`, so it could not
have carried `de` at all. The kernel takes `n_src` and `c_v_q`.

### 6.3 T2's own instrument goes stale the moment T2's fix lands

`currency_audit_t2.py`'s M1 — the measurement behind the report's headline —
computed **both** `delivered` and `received` from `e_gas_cond_sum`. That was
correct while the counter held the unconverted face sum, and the ratio it printed
was a real cross-ledger measurement. After D1 the counter holds the *converted*
books delta, so `received / delivered` is identically `c_v` **by construction**:
the instrument would have gone on printing `0.0076849` forever and read exactly
as though the fix had never landed. Anyone re-running it at T5b would have
concluded D1 failed.

Amended here (`delivered` now comes from the solid ledger, `lost − trunc`). At
`c_v = 1` both definitions agree to the last digit, so no T2 number moves.

**This is the third time on this arc that a check has been blind to the thing it
was named after** (design v2 §6 item 3; T2 §5.1's closure identity; this). The
pattern is the same each time: the check reads its two sides from one source.

### 6.4 There is no gate at all on the gas-side combustion heat yield

Doubling `H_fuel` through `config.toml` (`H_FUEL_SHIFT = 1`) — verified live to
reach the solver as `effective H_fuel = 8.0` — leaves **2557 passed, 0 failed**.
The xarch golden cannot catch it by design (`_xarch_perfield_digest.py` records
that the canonical golden is *"INDEPENDENT of the fire dials"*), and nothing else
covers it either.

The **expression** is gated (control D2b, one test). The **value** is not gated
by anything. T3 §8 q8 already called the flip a HUMAN-TEST change; this says
something sharper — there is no automated backstop underneath the human test, so
T5b's `H_fuel` move rests entirely on Erik playing it.

### 6.5 The capacity-CEILING corner: a behaviour change, measured unreachable

The one place the new expression is **not** bit-identical. When
`cap_used == cap_real == 2^(CAP_SHIFT_MAX+16)` (both clamped at the ceiling), the
shipped code took the `cap_real == cap_i` fast path and booked `de` unconverted;
the new form books `de·N/cap_used`, which differs whenever the clamp actually
bound. 70 083 of 70 729 synthetic cases in that stratum diverge.

Three reasons it is reported rather than preserved:

1. **The new value is the correct one.** `temperature_solver.h`'s own ceiling
   comment promises that when the clamp binds *"the energy it implies is counted
   … the difference lands in `e_cond_cap_sum` exactly like the n_floor_heat
   floor's does"*. On the gas branch that was **false** — the fast path booked
   nothing. The new form makes the header's documented contract true there for
   the first time.
2. **It needs N > 4096 atm** (`cap = N_raw` at `c_v = 1`, ceiling `2^28`), on an
   *accountable* gas cell, reached only through `PhysicsEngine` — the direct
   binding passes no `gas_energy`, so the branch is unreachable from there.
3. **It disappears at T5b.** At the physical `c_v` the gas ceiling would need
   `N_raw > 3.5e10`, past `int32` entirely. This is a `c_v = 1`-only artifact.

Empirically: goldens unmoved, all 780 per-`(field, tick)` hashes byte-identical,
full suite unchanged. Neither T2's blast radius nor its 300 000-case check names
this corner.

### 6.6 Smaller

- **T2 site #3b is already satisfied, unchanged.** T2 asks for
  `make_recip((double)c_v_q / 65536.0)` so the deposit's Q.32 `c_v` and the
  capacity's Q16.16 `c_v` become exact inverses. D1's conversion removes the only
  site where the *disagreement* could bite, and at `c_v = 1` both are exact
  anyway. Landing the line now would be a no-op the gates cannot distinguish from
  a mistake — **it belongs with the dial, at T5b**, where it is worth 0.0724 %.
- **T2 sites #1, #7–#10 are dial-dependent and deliberately not landed**: the
  `c_v` value (#1), the `test_thermostat_books` mixed-unit sum (#7, a no-op at
  `c_v = 1`), the two CUDA checks' hardcoded `DIALS["c_v"] = 1.0` (#8, still the
  shipped value), `test_fixed_point_i64_twins`'s `c_v >= 1` sweep (#9), and the
  re-baseline (#10, forbidden here).
- **T2's §6.2 "verify only" rows 11–22 were opened and confirmed unchanged.**
  The only one that needed an edit is #21's sibling, the instrument (§6.3).

## 7. What did not hold

1. **"T2 measured the replacement bit-identical over 300 000 triples, so it is
   safe to transcribe."** The brief's framing and my own first read. The
   expression is bit-identical where the engine can reach — but its int64 bound
   was argued at the wrong `c_v` and fails at 2 atm (§6.1), its `N == 0` corner
   was specified with a primitive that cannot take an `int64` (§1.3), and its
   ceiling corner was never enumerated (§6.5). Reproducing the proof rather than
   trusting it is what surfaced all three.
2. **"A green suite plus unmoved goldens means the patch is neutral."** True for
   D1. **False for D2** — nothing in the suite constrains that value at all
   (§6.4), so for D2 the evidence is the expression-level proof and the unchanged
   default, not the suite.
3. **"The CUDA conduction lockstep covers the conduction change."** Only its
   PART 3 does. PARTS 1 and 2 run the direct binding, where `gas_energy` is null
   and the whole accountable-gas branch is dead code. Established by breaking the
   twin and watching which part complained.
4. **"`e_cond_cap_sum` is in one currency."** It is not, and was not before this
   patch either: the T-form branch books heat counts, the gas branch now books
   `N·T`. At `c_v = 1` they are the same number. It is a diagnostic, not a term
   in any closure identity, so nothing breaks — but the flip makes the mixture
   real, and `temperature_solver.h` now says so.
