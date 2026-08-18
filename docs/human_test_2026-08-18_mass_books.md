# HUMAN-TEST 2026-08-18 — mg_cycles=8: fires PASS, grenade+breach FAILS
## …and the failure is a second, distinct bug: **the mass books don't close**

Erik played `playground` with `mg_cycles = 8` live (verified in-process:
`mg_cycles=8, nu1=2, nu2=2, coarsest=32, use_multigrid=True`).

**Erik's verdict:** *"fires dont blow up anymore, but grenades still can,
especially after i broke a wall with a high pressure room, it caused a
blowup"* — and, on the mass finding, *"i did throw grenades, i think they
create a little mass, but not that much."* Both correct. The numbers follow.

The recorder tripped: `BLOWUP DETECTED: max |P - P_prev| = 75.7` →
`debug_blowup_20260818_040647.npz` (216 MB). **This is the first dump carrying
`wind_x`/`wind_y`/`inert_n2`** — the diagnosable one the last arc instrumented
for. Keep it.

---

## 1. The storm fix holds; this is not the storm

| | fire storm (fixed) | this blowup |
|---|---|---|
| driver | pressure-solve residual | mass runaway |
| T | normal | normal (T_max 1194, ceiling 16000) |
| onset | slow, volume-filling | localized, at a breach, after a grenade |
| fixed by `mg_cycles=8` | **yes** | **no** |

## 2. Mass is CREATED, and not by a little

Total N summed over all gas cells, per tick:

```
start 2.89632e+08  ->  peak 6.25819e+08  ->  final 6.23429e+08     = 2.15x
deciles: 2.98 3.05 3.14 3.19 3.19 3.28 3.65 4.59 5.45 5.87  (e8)
```

Monotonic. **`playground` is `boundary = space, ambient = None`** — there is no
sky/ambient reservoir, so there is no legitimate external mass source at all,
and the one legitimate *sink* (venting through the breach) can only reduce it.

Scale, in units that mean something: ambient N = 1.0 = 65536 raw, so the
increase of 3.36e8 is **≈ 5,124 cell-equivalents of ambient air created** on a
map with **6,257 gas cells**. The engine minted ~82% of a second atmosphere.

Erik's read is exactly right: grenades *do* legitimately deposit bulk N — that
is by design and documented (`eos_p3_gate_measurements.md` §E: "the sealed room
ends over-pressured (~2.1 atm max) from the deposited bulk N — physical"). But
a handful of grenades is a handful of cells' worth, not five thousand.

**The local figure removes all doubt:** one cell reaches **4.15e7 × ambient**,
growing ~×2 per tick for 12 consecutive ticks:

```
snap        N         |u|        P
2227      6,470      17.7      0.022
2231    5.53e+05     55.7      0.156
2233    7.31e+06     90.7      0.210
2235    1.63e+07    180.4      0.383
2239    4.15e+07    108.7      1.371
```

Note the third column: **P does not track N.** p* = C·N·T_abs should be
astronomical at N=4e7; the solved pressure sits at 1.371. The mass field and
the pressure field have come apart.

## 3. Hypothesis falsified: it is NOT the density-division amplifier

The obvious suspect was the kick's `1/N̂` (`u -= dt·K·∇P/N̂`, floored at
`N_FLOOR_SOLVER = 0.001`, so up to 1000× amplification at a vacuum interface) —
the "density-division amplifier" named but never resolved in `docs/TODO.md`.

**Measured false.** Speed bucketed by N: the fastest cells are the *dense*
ones, median N ≈ 10,954 × ambient in the top-1000 by |u|. Low-N cells
(N < 0.001) have mean |u| = 1.77. The amplifier is not driving this.

## 4. Mechanism — semi-Lagrangian mass duplication at 27× CFL

```
peak |u|                          862.2 m/s   (1.35x c_local ~ 640)
displacement per TICK             107.9 tiles
per SUBSTEP at the n_sub=8 cap     13.5 tiles
CFL_ADV target                      0.5 tiles
=> overshoot                          27x
cell-snaps over 1 tile/substep  1,172,386
```

`N_SUB_MAX = 8` caps the substep count, so at blast/breach velocities the
semi-Lagrangian backtrace runs 27× outside its CFL target. **This repo has
already met this failure mode**: `eos_p3_gate_measurements.md` §C2 records that
a bad advection scale "gave 326-tile/tick displacements and **×5 mass
duplication**". Same signature, reached from the other direction — not a bad
constant this time, but a real supersonic flow the cap cannot resolve.

Secondary observation worth its own look: **|u| = 862 m/s exceeds `c_local`
≈ 640** (976 cell-snaps supersonic), yet the step-4 kick is supposed to clamp
|u| to `c_local`. Either the clamp is not binding on this path or something
downstream re-raises u. `U_MAX = 1000` was never hit, so the outer rail did not
catch it either.

## 5. What this is, in one line

**The energy-books arc closed the ENERGY books. The MASS books are still
open.** Same class of defect, one conserved quantity over — and the same
remedy shape: instrument first, find which pass mints, then fix the law.

## 6. Recommendation

1. **Keep `mg_cycles = 8`.** It fixed what it claimed to fix, confirmed by
   human test, and it is 18% faster. This second bug is independent of it and
   predates it (the pre-fix dumps show the same 98–104 atm density events).
2. **Open a mass-books patch as its own arc, audit-first** — a per-pass mass
   ledger exactly like the energy ledger: who creates N, who removes it,
   asserted every tick. Do not choose between the candidate fixes
   (`N_SUB_MAX` re-pin / making the velocity clamp bind / a conservative
   flux-limited mass advection) before that measurement exists.
3. **Do not retune fire on top of this yet.** A retune against a substrate that
   mints mass would bake the mint into the dials — the same argument the
   energy-books arc made for landing before any recorder milestone.

## 7. Merge status

`mg_cycles = 8` is HUMAN-TESTED and passes its own claim. Whether it merges now
or waits for the mass arc is Erik's call; the two are independent, and shipping
the storm fix first makes the mass bug *easier* to see, not harder, because it
removes the noise that was masking it.
