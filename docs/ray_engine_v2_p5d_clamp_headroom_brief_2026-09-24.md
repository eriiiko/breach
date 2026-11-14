# P5d brief — the clamp's ceiling gets one bucket of headroom (2026-09-24)

Ray-engine-v2 arc, issue #12. Patch after P5c (merged `0d4e07b`), before P6.
Erik ruled the form on 2026-09-24. **Mechanical, oracle-gated: no HUMAN-TEST.
It merges on a green gate.**

## 1. The finding

The Pass-1 maximum-principle clamp (design v3 §2.8, row 21) bounds the
radiative sub-step by

    T_new = min(T_after, max(T_before, E°⁻¹(Φ)))

`e_inv_q` returns the **low edge** of the largest bucket `b` with `E°[b] ≤ Φ`
(design §2.6, chosen so `E°[E°⁻¹(Φ)] ≤ Φ`). The forward emission is a staircase:
4 K buckets, no interpolation, on purpose (`emissive_table.h`). A staircase has
no temperature at which emission equals absorption. An undamped cell with no
clamp settles by flickering across the edge where its bucket's `E°` crosses Φ,
the low edge of bucket `b+1`, and on average it balances exactly. The clamp
stops it a bucket short, at `4b`. There it still emits `E°[b] < Φ`, so every
tick it net-absorbs, and the clamp withholds the whole surplus. That is P5c's
counted shave (`e_rad_clamp_drop_sum`; `tools/bench_clamp_shave.py`: solids
~0.2 % of their gross absorption in the playground, 1.3–1.9 kW, all of it this
bucket-edge pinning at 30 s).

**The orchestrator's 0-D check.** One cell under a held Φ, built on
`sweep_ref_q.cell_rad_net_q` and the solid fold's arithmetic with the live table.
It was a scratch script, not committed. The thin wood row was used (`a` 0.9,
`heat_inv_shift` −3), with Φ sampled uniformly across the buckets:

| ceiling | withheld / gross absorbed (~800 game) | mean T − continuous T_eq |
|---|---|---|
| today: `4b` (low edge of b) | 0.67 % | −3.9 K |
| exact inverse (linear between midpoints) | 0.16 % | −0.5 K |
| low edge of `b+1` | 0.17 % | −0.1 K |
| **top of `b+1`** | **0** | −0.1 K, identical to no clamp at all |

The same ranking holds at 400 and 1500 game. At 5 000 and 11 000 game, Fleck
damps this cell and the clamp is essential: without it the cell runs away by
+434 K and +21 760 K. There every ceiling form withholds the same large share
(7.6 %, 73 %), which is the clamp's real job and the known Fleck trade (design
§2.8), and it is not touched here. The ceiling form only moves where the cell is
held: +4.5 K above the continuous equilibrium instead of −3.5 K below.

## 2. The ruling (Erik, 2026-09-24) — do not re-derive

The clamp's ceiling becomes **the top of the first bucket whose `E°` exceeds
Φ**. In words: radiation may carry a cell into the first bucket in which it
out-emits what it absorbs, and no further. The design's promise "a clamped cell
never emits more than it absorbs" becomes "… by more than one bucket". That is
the resolution the emission itself works at. `e_inv_q` stays exactly as it is:
it is still THE inverse of the table, and the tile inspector's and the heat-law
tests' "radiation temperature" is still `e_inv_q`.

## 3. The change

A new `FP_HD` lookup beside `e_inv_q` in `cpp/src/emissive_table.h` (working name
`e_ceiling_q`; rename it if you have a better one, and say why). It lands in
`sweep_ref_q.py` FIRST, its gates are run, and only then is the C++ written
against it (the Integer-reference rule). With `b` = the largest bucket with
`E°[b] ≤ Φ`:

- `Φ < E°[0]` → **0**, unchanged from today (design row 31: no radiative warming
  above the ambient floor). This is the one deliberate exception to "the top of
  the first out-emitting bucket". Such a cell's `rad_net` is ≤ 0 except through
  rounding, and a shadowed cell must not creep.
- otherwise → `(min(4·(b+2), 4·E_TABLE_SIZE) << 16) − 1`, the last Q16 value in
  bucket `b+1`. It saturates at `(16000 << 16) − 1`, still below `T_MAX_PHYS`, so
  the rail stays unreachable on the radiative sub-step with the clamp on (gate 4
  per counter, unchanged).

It is used at every site that clamps, and nowhere else:
- reference `fold_pass1_solid` and `fold_pass1_gas`
- `temperature_solver.cpp` (solid branch ~L314, gas branch ~L451)
- `cuda_temperature.cu` (~L252, ~L312)
- the EmissiveTable binding and its stub, and the tile inspector's "the clamp's
  ceiling" field (`renderer/hover_readout.py`). The inspector's clamp field must
  show the clamp's actual ceiling. Whether it also keeps `E°⁻¹(Φ)` is your call.
- `tools/bench_clamp_shave.py`: keep its exact-replay assertion. Its "held"
  classification is now relative to the new ceiling.
- `tools/fire_tuning_lab.py` only where it means the clamp's ceiling.

Comments that state the old ceiling are rewritten: `temperature_solver.{h,cpp}`,
`cuda_temperature.{h,cu}`, and the reference's docstrings. CLAUDE.md gets two
edits: the "Temperature solver" row's clamp formula, and the "Emissive table"
row, which lists the new lookup beside `e_inv_q`. `docs/ray_engine_v2_design_v3_2026-09-15.md`
is append-only. Add one dated line at §2.6's `E°⁻¹` paragraph and one at §2.8's
clamp formula pointing here; do not rewrite them.

## 4. The one risk — measure it, and stop if it bites

Near ambient the live table's buckets differ by only ~6.6 counts (`E°[0]` =
125). The sweep's per-ordinate floor on emission (`(x·w_m) >> 16`) can under-read
a slightly warm cell's emission by more than that. P5c's bench found such cells:
undamped net absorbers above their cap's bucket, 31 to 344 cell-ticks per scene,
up to 222 game. Today the strict ceiling withholds their spurious gain. The
headroom lets it land, which warms such a cell up to the top of the next bucket.
Measure on the bench's three scenes, with the playground at both 30 s and
180 s:

- (a) the shave per medium under the new ceiling. Expected: solids ≈ 0 apart
  from their held share; gas keeps its held share.
- (b) the per-cell temperature difference, headroom minus strict, at run end.
  Report the max and the mean over cells the clamp ever touched. The method is
  yours: paired runs diverge, and a replay on shared inputs is an option.
- (c) the artifact's signature: cell-ticks where an UNDAMPED cell (f == 2²⁴) is
  clamped at the headroom ceiling itself. Its bucket out-emits Φ there, so only
  an under-read emission can pin it. Report the count, the hottest such cell,
  and where they are.

**STOP after the reference + CPU, before the CUDA port and the golden, and
report,** if (c) pins any cell whose radiation temperature `e_inv_q(Φ)` stays
below 20 game for more than 24 consecutive ticks: the artifact would then be
warming the unlit environment. Otherwise report the numbers and carry on. The
per-ordinate floor itself is out of scope (§7).

## 5. Properties to pin

Write these from this section, never from the implementation. Each test's
docstring names its property and the change that breaks it. Validate each new
property by breaking the code once, then restore.

1. **The ceiling** (reference == binding, over every bucket): its bucket is the
   first whose `E°` exceeds Φ (`E°[bucket(c)−1] ≤ Φ < E°[bucket(c)]` for
   Φ ∈ [E°[0], E°[3999])). It is ≥ `e_inv_q(Φ)`, it is monotone in Φ, and both
   edge cases hold. *Breaks if the clamp is reverted to `e_inv_q`.*
2. **The fold** (solid and gas, reference and CPU): the radiative sub-step never
   carries a cell above `max(T_before, ceiling)`, which is gate 5 restated. It
   never clips a cooling step, and a burning cell above its ceiling is never
   clamped (the existing gates stay true).
3. **Undamped balance** (0-D, reference, live table): an undamped cell under a
   held Φ ends with zero withheld energy over its last N ticks and a mean within
   one bucket of the continuous equilibrium. Φ must be sampled in BOTH halves of
   its bucket, or the test is vacuous. *Breaks on the low edge AND on the exact
   inverse.*
4. **The clamp's real job** (0-D): a Fleck-damped cell (well above the fire
   range) is held within [T_cont − 1 bucket, T_cont + 2 buckets], where Fleck
   alone runs away. The over-driven scene still engages the clamp (G4). *Breaks
   if the clamp is removed.*
5. **Books**: `e_rad_clamp_drop_sum` still equals the withheld energy exactly
   (the bench replay asserts it), and the #54 closure and P-G5 solid ledger gates
   stay green.
6. **CPU == CUDA at tol 0**, on scenes where the clamp binds AT the headroom
   ceiling on both the solid and the gas branch (a non-vacuity check).

The design's 0-D equilibrium table (804 / 3448 / 11 224) and any test pinning
it are restated as properties: measured values, not snapshots. A failing
pre-existing test is a finding: report it, never bend the design to it.

**Golden.** It should move: radiatively equilibrated cells land up to one
bucket warmer. First prove that is the only cause. For example, show the old
golden reproduces with the clamp routed back to `e_inv_q`, and name the fields
that moved. Then re-baseline ONCE with a written rationale that names this
ruling.

## 6. Systems

Uses, never duplicates:
- the Emissive table (the new lookup lives there, FP_HD, one definition for host
  and device)
- the Integer reference (spec first)
- the Temperature solver's Pass-1 clamp and its gas energy form
- the sweep's CPU↔CUDA discipline and harness
- the Energy closure identity
- `bench_clamp_shave.py` (the existing instrument; extend it, don't fork it)
- the Tile inspector

**New system: none.** The draft rule is an edit to the Emissive table row:
*"…its inverse `e_inv_q` (…), and the clamp's ceiling `e_ceiling_q` (the top of
the first bucket out-emitting Φ; 0 below `E°[0]`; saturating below `T_MAX_PHYS`)
— the clamp reads the ceiling, never `e_inv_q`."*

## 7. Out of scope — one system at a time

- The per-ordinate emission floor, Fleck, the sweep, #73's thermal books, and
  combustion. Anything else you notice gets one line in the report, never a fix.

## 8. Execution

- **Where:** branch `12-p5d-clamp-headroom` off `fire-12` `0d4e07b`, worktree
  `breach-p5d`, one writer. **Model: Opus.**
- **Builds:** run `cpp\build_cpu_home.bat` and `cpp\build_cuda.bat` from the
  worktree, first and after every C++ change.
  - Python is `C:/Users/steen/anaconda3/python.exe`.
  - Run the suite as `pytest tests -q -p no:cacheprovider` from the worktree root.
- **Commits:** `feat(#12): P5d -- …` / `test(#12): …` / `docs(#12): …`, staging
  explicit paths only (never `git add -A`). Do not merge, do not push.
- **Report:** your FINAL MESSAGE is the report (the harness blocks report
  files). It is the orchestrator's technical record:
  - §4's numbers
  - every test added or rewritten, one line each: its property and what breaks it
  - the golden evidence
  - suite totals on both builds
  - findings outside the patch, one line each

  Keep it as long as the evidence needs and no longer, and include no question
  lists.
