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
- [ ] D2 — the `H_fuel` schema widening (CPU + CUDA + config)
- [x] D1's bit-identity proof, reproduced not inherited
- [ ] D2's bit-identity proof
- [ ] the negative controls (break it, watch it go red)
- [ ] the gate: suite, goldens, `git status`

**D1 gate, at the D1 commit:**

- suite **2557 passed, 4 skipped, 4 xfailed, 0 failed** — identical to the
  pre-patch baseline on this box, CPU build **and** CUDA build present
- `GOLDEN_AGGREGATE` **unmoved** (`167b96bd…`), `DIGEST_SPEC_VERSION` still 5
- all **780** per-`(field, tick)` hashes byte-identical to the pre-patch dump
- CUDA conduction lockstep **tol-0**: 121 configs + a 120-tick trajectory + the
  30-tick engine A/B, `P66_RESULT: PASS`

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

*(pending)*

## 3. The bit-identity proofs

*(pending)*

## 4. The negative controls

*(pending)*

## 5. The gate

*(pending)*

## 6. Findings — what T2's and T3's blast radius missed

*(pending)*

## 7. What did not hold

*(pending)*
