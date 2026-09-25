# #78 brief — the sweep's heat channel gets fractional bits (2026-09-25)

Issue #78. Erik asked for a fix before #12's P7 (2026-09-24: "better to really
try to fix this before P7"). It is its own issue, not an arc patch, and it goes
after P5d (merged), before P6a. **Oracle-gated, no HUMAN-TEST: nothing visible
moves. It merges on a green gate.**

## 1. The defect (read #78 and its literature comment first)

At the live scale (`rad_scale_derived`), a room-temperature black body is
**E°[0] = 125 heat counts per tick**. The sweep splits it over 16 ordinates with
`w_m = ONE/16`, so every per-ordinate term is a single-digit integer:
- `amb_m = E°[0] >> 4 = 7`, so a uniform ambient field has Φ = 112, which is
  below E°[0] = 125.
- The excess emission `ex_m = (E°[T] − E°[0]) >> 4` stays 0 until bucket 3,
  about 12 K above ambient.
- Each floor (on emission, on absorption, and in the gather's `fa`) loses up to
  one count.

Near ambient, one count per ordinate (16 counts in total) is worth about 10 K
of emission.
The consequences:
- Cells within a few kelvin of room temperature read as net radiative absorbers
  (+11–13 counts per tick on a thin panel whose Φ − E°[0] is 0–2).
- They cannot shed a small excess radiatively.
- The Pass-1 clamp deletes the spurious gain, mostly under design row 31's zero
  ceiling for Φ < E°[0]. That is about 1 kW, counted, in
  `e_rad_clamp_drop_sum`, on the playground.

Since P5d's headroom, some of it also lands: two thin-wood cells were pinned
1–2 K warm. The literature calls this **stagnation**: Croci & Giles 2023, and
Klöwer et al. 2020, whose remedy is rescaling so that increments sit well above
the quantum.

## 2. The ruling — do not re-derive

**Scaling**, option 1 on #78. The sweep's heat channel works in a currency
2^k finer than one heat count. Stochastic rounding (an RNG inside the sweep) and
remainder carrying are NOT in scope.

## 3. The change

- **ONE constant** for the fractional bits, `k`, in the Emissive table's header
  beside the table it scales. The proposal is **k = 11**. It makes the fine live
  table ≈ 0.66× the integer reference's resolving table (5.1427e-5 ≈ 2^11.6 × the
  live scale), which is the magnitude every arithmetic gate already exercises.
  The per-ordinate rounding then drops from ~10 K of emission-equivalent to
  ~0.005 K, and the table's own 4 K buckets become the only resolution limit, as
  designed. A different k is your call if the headroom analysis in §4 says so;
  state why.
- **The table is baked fine.** The engine's ONE `emissive` table is baked at
  `rad_scale_derived · 2^k` (a power of two, exact in double, so the bake still
  rounds once). Φ, the table, the ambient plane (`derive_ambient`, same chain)
  and the four sweep planes are then all in the fine currency. The clamp's
  `e_inv_q` / `e_ceiling_q` need no change: they compare Φ and E° in one unit.
  Never keep a second, coarse copy of the table (one implementation). The
  vestigial `Raycaster` copy follows the same bake or is excluded with its
  reason; P6c deletes it.
- **Every reader of a sweep plane or a table value that means heat counts
  converts ONCE, through ONE kit helper** in `fixed_point.h` (FP_HD), with
  symmetric rounding (the `shr_round0` idiom, so +x and −x lose the same). Never
  scatter `>> 11`. The known readers:
  - the fold's solid branch (the shift becomes `his + k`)
  - the fold's gas branch (the staged `deposit_dT_wide_i64` chain converts at
    ONE defined point)
  - the Fleck pre-pass, both arms (`fleck_L_solid_q`; `fleck_L_gas_q` in the
    fold's own gas currency, so the damping and the landing stay one arithmetic)
  - the P5c boundary counters (`e_rad_boundary_export_sum`,
    `e_rad_floor_drop_sum`) and `e_rad_clamp_drop_sum`'s pricing
  - `exchange.py`'s `rad_flux` → unit heat damage (`max(heat, rad_flux)`: a
    marine must burn exactly as before, in real units)
  - the tile inspector (`pack_hover_readout`: display in heat counts or label
    the unit)
  - `tools/fire_tuning_lab.py`, `tools/bench_clamp_shave.py`,
    `tests/_sealedbox_bisect_bench.py`, `tools/bench_radiation_sweep.py`
  - the reference's `E_LIVE` / `RAD_SCALE_LIVE` / `config_dials_match()`

  **Inventory the rest yourself** (grep the four planes `rad_net_sweep`,
  `rad_flux_sweep`, `rad_amb_sweep`, `rad_fluence`, plus `emissive.table()`,
  `rad_scale_derived` and `E_LIVE`) and list every site in the report.
- **Order.** The reference `sweep_ref_q.py` comes FIRST, its gates re-run, then
  `radiation_sweep.cpp` and `temperature_solver.cpp`, then the CUDA twins
  (`cuda_radiation_sweep.cu`, `cuda_temperature.cu`), held to the CPU at tol 0.
- **Row 31's zero ceiling (Φ < E°[0] → 0) STAYS.** One change at a time. Report
  what it still catches after the fix. The ambient field's Φ = 16·(E°_k[0] >> 4)
  sits up to 15 fine counts below E°_k[0], so near-ambient cells may still take
  that branch. Say whether that matters, measured, and fix it only if it does,
  and only in the smallest honest way.

## 4. Headroom — prove it before building

Take the int64 bounds of design v3 §2.3 / §3 (critique 3: < 2^46 per cell, no
product above 2^58, at the resolving scale). Restate them at the fine live scale
for your k: the largest stream, the `stream·a` and `ex_m·f` products, the per-cell
sums over 16 ordinates, and `rad_fluence`, at `T_MAX_PHYS` on the table top.
Add a test that asserts the bound on an over-driven scene. **STOP and report**
if k = 11 does not fit with at least 2^3 of margin; then propose a smaller k.

## 5. Properties to pin

Write these from this section, never from the implementation. Each test's
docstring names its property and the change that breaks it. Validate each new
property by breaking the code once.

1. **The near-ambient artifact is gone.** This is the acceptance test, on the P5d
   bench (`tools/bench_clamp_shave.py`, all three scenes, playground at 30 s and
   180 s): the Φ < E°[0] "shadow class" drop and the pinned near-ambient panels
   fall to ≤ 1 % of their P5d values, and the solids' remaining shave is
   reported. *Breaks with k = 0.*
2. **Radiative cooling near ambient** (reference and engine): a cell in bucket 1
   or 2 (4–12 game) in a uniform ambient field has `rad_net < 0` for every
   shipped absorbing row. Today its excess floors to zero in every ordinate. A
   cell AT ambient still has exactly 0. *Breaks with k = 0.*
3. **Uniform ambient is still an exact per-cell fixed point** (gate 2a),
   bodies and leak included, and the sweep's identity
   `Σ rad_net + Σ rad_flux + Σ rad_amb ≡ 0` is still exact in int64, now in the
   fine currency.
4. **The fire range is unchanged in substance.** k buys precision, not physics.
   - G10's equilibrium band, P5d's G16 properties, the cooling curve against the
     analytic reference and the P2b reach crossing all hold, each within one
     bucket or its own stated tolerance.
   - A marine beside a fire loses the same HP per second within 1 %.
   - *Breaks if a conversion is off by 2^k.*
5. **Books.**
   - The #54 closure identity closes per tick, and so does the P-G5 solid
     ledger.
   - The §8.4 boundary is restated: `Σ rad_net − Σ landed − the counted drops`
     is bounded by the fold's one conversion per touched cell, now at the
     temperature field's own resolution (≤ one Q16 T-LSB × capacity).
   - Gate on both.
6. **Gate 0** (C++ == reference, bit for bit) and **CPU == CUDA at tol 0** on
   every plane and counter, on the live table AND the resolving table.
7. **Headroom** (§4) on an over-driven scene.

A failing pre-existing test is a finding: report it, never bend the design to
it. Tests that pin coarse-currency numbers are restated as properties or
converted, each with one line of reason.

**Golden.** It moves: near-ambient radiative exchange is now resolved. Prove
first that the scaling is the only cause. For example, show that k = 0
reproduces `e9cb8ca8…`, and name the fields that moved. Then re-baseline ONCE,
with a rationale naming #78 and Erik's 2026-09-24 request. **Trap:** running
`tests/_xarch_perfield_digest.py` as a script overwrites the tracked
`tests/_xarch_perfield_DESKTOP-0E98HUV.txt` (the July attestation). Restore it
from HEAD and never commit it.

## 6. Systems

Uses, never duplicates:
- the Emissive table (one table, now fine)
- the Integer reference (spec first)
- the Radiation sweep and its CUDA twin
- the Temperature solver's fold (both branches) and the Pass-1 clamp
- the Fixed-point kits (the ONE conversion helper lives there)
- the Energy closure identity
- the Coupling table (`exchange.py`)
- the Tile inspector
- `bench_clamp_shave.py`

**New rule (draft, for CLAUDE.md):** extend the Radiation sweep row: *"the heat
channel's currency is 2^k finer than a heat count (`<constant>`); every reader of
a sweep plane or an E° value converts once through `<helper>`, never an inline
shift."*

Update the Emissive table row's #78 note to match what you measured about row 31.

## 7. Out of scope

Light (P6a picks its own scale), row 31's rule, stochastic rounding, remainder
carrying, #73's fold truncation items, and Fleck's design. Anything else you
notice gets one line in the report.

## 8. Execution

- **Where:** branch `78-sweep-fine-heat-currency` off `fire-12`, worktree
  `breach-78`, one writer. **Model: Opus.**
- **Builds:** run `cpp\build_cpu_home.bat` and `cpp\build_cuda.bat` from the
  worktree, first and after every C++ change.
  - Python is `C:/Users/steen/anaconda3/python.exe`.
  - The suite is `pytest tests -q -p no:cacheprovider` from the worktree root.
- **Commits:** `fix(#78): …` / `test(#78): …` / `docs(#78): …`, staging explicit
  paths only (never `git add -A`). End each commit with the Co-Authored-By
  line. Do not merge, do not push.
- **Report:** your FINAL MESSAGE is the report (the harness blocks report
  files). It is the orchestrator's technical record:
  - the §4 headroom table
  - the full reader inventory
  - the §5.1 numbers before and after
  - every test added or rewritten, one line each: its property and what breaks
    it
  - the golden evidence
  - suite totals on both builds
  - findings outside the patch, one line each

  Keep it as long as the evidence needs and no longer, and include no question
  lists.
