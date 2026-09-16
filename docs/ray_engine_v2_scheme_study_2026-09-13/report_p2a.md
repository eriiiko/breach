# P2a - the sweep zeroes its own outputs, and the Fleck alpha floor goes to 0

> **Written incrementally** while the patch was implemented (design v3 section
> 11.5, row 40: a kill must cost no resume). Branch `12-p2a-self-zero-floor0`,
> cut from `fire-12` at `78c504f`.
>
> **Depends on:** `ray_engine_v2_design_v3_2026-09-15.md` rows 38-40, section 2.8
> (the RULED paragraph), section 11's P2a row; `report_p0.md` sections 0.4 and
> 0.8 - P0b's measurement of both floors, whose numbers are reproduced here as
> the new default.

## 0. What P2a is

Two mechanical changes, neither of which moves anything the live game computes -
the old ray cast still feeds the temperature fold, so every golden is untouched:

1. **Floor 0** - `alpha = max(0, 1 - 1/g)`, i.e. `D = max(T_abs_q, 4*L_q)` - in
   the integer reference FIRST (its gates re-run), then in the engine's Fleck
   pre-pass, with gate 0 (bit for bit) re-run. Erik's ruling of 2026-09-16
   (row 39).
2. **The sweep zeroes its own four output planes** at the start of `run()`, and
   the four `fill(0)` lines P1 added to `Simulation.step` go - so the tile
   inspector shows Phi, `f` and `E_inv(Phi)` at render time instead of a wiped
   zero (row 38).

---

## 1. T1 - floor 0 in the reference first

`sweep_ref_q.py` already carried `alpha_floor` as a parameter (P0b added it for
section 0.8's measurement), so the change is the **default**, named once:

    ALPHA_FLOOR_HALF    = "half"   # alpha = max(1/2, 1 - 1/g) -- Fleck's IMC bound
    ALPHA_FLOOR_ZERO    = "zero"   # alpha = max(0,   1 - 1/g) -- RULED (row 39)
    ALPHA_FLOOR_DEFAULT = ALPHA_FLOOR_ZERO

and nine `alpha_floor: str = ALPHA_FLOOR_HALF` defaults become
`= ALPHA_FLOOR_DEFAULT`. The half form stays fully testable, and gates 8 and 10
now **measure it beside the default**, the way gates 8 and 12 already measured
the rejected Q16 form - so "f == 2^24 below g = 1" and "the cooling march equals
explicit" cannot pass on a probe that could not tell two laws apart.

### 1.1 G8 - the Fleck integer form at floor 0

The assertion changed from `f_q24 == 2^24 iff L_q == 0` to
**`f_q24 == 2^24 iff 4*L_q <= T_abs_q`**, which is the whole content of row 39.

```
worst |f_q24/2^24 - f_float| over 4000 buckets x 5 (a, his) pairs = 5.960e-08
  (one Q24 count = 5.960e-08) at T=3048 a=65536 his=5                      OK
f_q24 == 2^24 iff 4L <= T_abs (RULED floor 0, row 39): True
  -- 1762 undamped and 18238 damped probes, both sides present;
     the superseded floor of one half damps every L > 0 probe: True
  undamped (f == 2^24) up to   1872 game: hull / steel
  undamped (f == 2^24) up to   1068 game: wood / door / door_closed
  undamped (f == 2^24) up to   2272 game: glass
  undamped (f == 2^24) up to   1424 game: furniture / kindling
the rejected Q16 form on the same probe: worst = 1.525e-05 -- 256x
  T=     0 a=0.5 his=3: g=  0.000  f_q24=16777216 (1.0000000) float=1.0000000
  T=   280 a=0.5 his=3: g=  0.035  f_q24=16777216 (1.0000000) float=1.0000000
  T=  1263 a=0.5 his=3: g=  0.736  f_q24=16777216 (1.0000000) float=1.0000000
  T=  1800 a=0.5 his=3: g=  1.805  f_q24= 9295495 (0.5540547) float=0.5540547
  T= 15996 a=0.5 his=3: g=848.297  f_q24=   19777 (0.0011788) float=0.0011788
```

The four undamped ceilings **reproduce P0b section 0.8a exactly** (1068 / 1424 /
1872 / 2272). The accuracy bound is unchanged at one Q24 count and is now
attained only above the branch - below it integer and float are both exactly 1.

### 1.2 G10 - stability and equilibrium at floor 0

The old assertion (`|err_Fleck| < |err_explicit|`) is **false at floor 0 by
construction** and was replaced by the property the ruling actually claims:

* below `g = 1` the damping is OFF, so the march **equals the explicit one, count
  for count** (asserted `f_new == e_new`), and the superseded floor is measured
  on the same march and must differ (asserted) - that difference is exactly what
  floor 0 gives up on free cooling;
* above `g = 1` the damping still does its whole job (explicit rails, damped does
  not) - unchanged;
* the clamp still pins every equilibrium exactly - unchanged.

```
cooling 1263 game for 0.5 s, NEW law (ambient bath, excess form):
  explicit 470.3 (-5.70%)  Fleck@floor0 470.3 (-5.70%)  analytic 498.8
  OK (g = 0.74 < 1: floor 0 IS forward Euler, count for count)
  the superseded floor of one half on the SAME march: 499.9 (+0.23%) -- differs
  from  1800 game, 0.5 s: explicit  455.3 (low-rail 0)  Fleck  488.6 (0)  analytic 520.0
  from  2500 game, 0.5 s: explicit    0.0 (low-rail 1)  Fleck  508.7 (0)  analytic 529.2
  from  5000 game, 0.5 s: explicit    0.0 (low-rail 1)  Fleck  557.5 (0)  analytic 535.2
  from 16000 game, 0.5 s: explicit    0.0 (low-rail 1)  Fleck  710.0 (0)  analytic 536.2
monotone + positive + finite from every start in 0..15996 game, 48 ticks: clean
at the table top the damped source is 4,270,337,965 vs the undamped
  3,622,281,227,645 counts (x848 less)

equilibrium under a held fluence Phi = 0.25 x E[T_src], 9600 ticks:
   T_src  true (E_inv)  true (float)  Fleck only, 0  Fleck only, 1/2  Fleck+clamp  hits
    1263           804         806.6          807.8            843.9        804.0  9575
    5000          3448        3451.1       38,258.7         38,258.7       3448.0  9600
   16000         11224       11226.5    3,454,182.4      3,454,182.4      11224.0  9600
un-clamped bias at the 1263-game source: floor 0 +0.16 % vs floor 1/2 +4.63 %
```

Every number **reproduces P0b section 0.8b/c**: 470.3 / -5.70 % free cooling,
807.8 / 38 258.7 / 3 454 182.4 un-clamped, 804 / 3448 / 11 224 clamped exactly,
and the two floors bit-identical above the fire range.

### 1.3 G12 - re-checked at floor 0, unchanged

```
row                          Q16 steps  Q16 worst  Q24 steps  Q24 worst  f_q24 @ top
hull / steel                       540     0.547%          0     0.000%        39555
wood / door / door_closed          453     2.468%          0     0.000%         9888
glass                              533     0.290%          0     0.000%        65924
furniture / kindling               507     1.185%          0     0.000%        19777
non-vacuity: the rejected Q16 form has 2033 backward steps over the same rows
pathological (a = 1, thermal_mass = 1): 100 Q24 backward steps, worst 0.047 %
```

**0 backward steps on every shipped row**, bit-identical to P0b's table - below
`g = 1` the source is the undamped `E[T]`, monotone by the bake; above it both
floors take the same `4L` arm. Measured, not assumed.

### 1.4 FINDING - gate 4's fast mode went vacuous at floor 0

The over-driven scene (a held 16 000-game source one tile from a **cold** crate)
starts the crate cold and lets it **climb** to its cap. At floor 0 the crate's
own emission is undamped below `g = 1`, so it climbs more slowly:

| floor | first clamp hit | hits in 24 ticks | crate, clamped | crate, un-clamped (96 ticks) |
|---|---|---|---|---|
| **0 (ruled)** | tick **11** | 14 | 1108.0 game | **1113.6 game** |
| 1/2 (superseded) | tick 8 | 17 | 1108.0 game | 1219.8 game |

The gate's fast mode ran 8 ticks, so it reported **clamp = 0, crate 1074.6 game
and PASS** - a silent clamp read as a pass. Fixed by giving that sub-check its
own tick count (`12 if fast else 24`) with the measurement written into the
comment; the property is untouched, the probe is no longer one tick short.

**A second finding rides on the same table**: design row 34 / P0 section 0.6
quote the 2-D over-driven scene as *1108 clamped vs 1220 un-clamped - "10 %, not
x300"*. At floor 0 that gap collapses to **1108 vs 1113.6, i.e. 0.5 %**. Floor 0
removes almost all of the maximum-principle violation the clamp exists to catch
**in the fire range** - exactly the 0-D result of P0b section 0.8c (+0.16 % vs
+4.63 %), now confirmed in the 2-D sweep. The clamp is still necessary and still
non-vacuous (14 hits in 24 ticks); above the fire range nothing changes.

### 1.5 Also corrected

`p0b_alpha_floor.py` labelled floor 1/2 "(RULED)" and floor 0 "(section 12 item
4)". Both labels are now stale; corrected in the docstring and in `FLOORS` (so
the figure's legend is right). The script names both floors explicitly on every
call, so the new default does not change what it measures.

**T1 result: all 13 reference gates PASS in full mode; the pytest wrapper
`tests/test_ray_engine_v2_integer_reference.py` is 14 passed in 1.6 s.**

---

## 2. T2 - floor 0 in the engine, gate 0 re-run

One line of arithmetic, in `cpp/src/radiation_sweep.h`'s `fleck_f_q24` (the
FP_HD pre-pass primitive the sweep, the binding and the tile inspector all
share - there is only ever one of it):

    const int64_t d1 = T_abs_q + (L_q << 1);           // GONE
    const int64_t d2 = L_q << 2;                       // 4*L_q
    const int64_t D  = (T_abs_q > d2) ? T_abs_q : d2;  // floor 0 (design row 39)

plus the credits block (which quotes Fleck's alpha and now says why our floor
is 0 and not his 1/2), the `.cpp` scheme comment, and the CLAUDE.md sweep row -
which did not carry the formula at all before and now does.

### 2.1 Gate 0 - still bit for bit

`tests/test_radiation_sweep_reference.py`: **34 passed** (0.4 s). All 32
randomised configurations - 4 scenes x {S16, S12} x {k = 0, k = 0.10} x {shear,
step} - plus the per-bucket pre-pass comparison on every shipped `(a, his)` pair
and the door-rejection test. **Every plane and the Fleck plane EQUAL, integer
for integer**, after both sides moved to floor 0. This is the whole point of the
reference-first rule: the two implementations changed in lockstep and the oracle
proved it.

### 2.2 Gates 1-6 on the engine - 24 passed

Numbers that MOVED at floor 0 (all previously floor-1/2 measurements):

| measurement | floor 1/2 (P1) | floor 0 (P2a) |
|---|---|---|
| over-driven crate, clamp hits in 24 ticks | 17 | **14** |
| over-driven crate, clamped | 1108.0 game | **1108.0 game** |
| over-driven crate, un-clamped (96 ticks) | 1219.8 game | **1113.6 game** |
| burning crate (G5b), 1263 game after 24 ticks | (not quoted) | 976.6 game |

The clamp's own gates are unaffected in kind: it still fires, the rails stay
silent on the 2-D scene, and the separately-proved `INT32_MAX`-deposit scene
still lights `t_max_phys_hits` 15 times with the clamp off.

### 2.3 Isotropy with the engine's own Fleck factor ON - REPORTED, not asserted

81x81, a 1263-game source, `k = 0.10`, ambient ring on. Spread = max/min radius
at the `E[280]` ignition crossing (lower is rounder).

| source | shear, Fleck OFF | step, Fleck OFF | shear, Fleck ON | step, Fleck ON |
|---|---|---|---|---|
| 1 tile | 1.459 | 1.627 | **1.520** | **1.762** |
| 3x3 | 1.277 | 1.262 | **1.215** | **1.289** |
| 5x5 | 1.123 | 1.248 | **1.261** | **1.249** |

Mean ignition radius, Fleck ON: shear 3.76 / 9.50 / 13.10 tiles, step 4.39 /
8.47 / 11.33 (Fleck OFF: shear 4.93 / 11.55 / 15.33, step 5.27 / 10.18 / 12.93).

The Fleck-OFF half is the reference's configuration and is **unchanged**, as it
must be (f = 2^24 either way), and still equals the reference's own planes cell
for cell on the full 81x81 grid.

**FINDING - design row 37 needs amending.** P1 reported that with damping on the
**1-tile** ordering flips (1.716 shear vs 1.688 step). At floor 0 **that flip is
gone** (1.520 vs 1.762, shear clearly rounder) and the **5x5** case flips
instead, marginally (1.261 vs 1.249, about 1 %). So the damped ordering is not a
stable fact about the transports at all - it moves with the alpha floor. The
"heat takes shear" ruling rests on the undamped configuration, which is exactly
where the gate asserts it, and on the mean radius, where shear leads in every
row. The stale numbers in the gate's docstring are corrected.
