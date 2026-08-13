#!/usr/bin/env python
"""Erik's manual fire-tuning loop (Fable, 2026-07-25).

Edit the TUNE block, run, read the scorecard, repeat until satisfied.
Reference: docs/fire_tuning_plan_2026-07-22.md §9 (the fire chain + the
tuning order + the targets). This wrapper drives tools/fire_timing_harness.py
via its CLI (--set overrides) — config.toml is NEVER touched; when a combo
is blessed, its values go to config in the close-out commit, not here.

Run (from a worktree root):
    conda run -n data python tools/fire_tune_loop.py

Tuning order (§9.3 — one group at a time, everything else frozen):
    1. STRUCTURE (once)   fire_T_ext=250 < ignition_temp, span 100
    2. THERMAL            k_fire_heat vs cool_shift -> flame T 400-500
    3. RAMP               k_grow / k_die -> peak ~0.5 @ ~3 min
    4. LIFETIME           wall_damage -> fire death 6-8 min
    5. VERIFY anchors     (burn_rate / X_ext / fuel_per_o2 / tau -- no tuning)

--------------------------------------------------------------------------
DEFAULTS REBASED 2026-07-30 by the thermal-mass-axis arc (P3 close).
Full measurement report: docs/thermal_mass_axis_bench_report_2026-07-30.md
--------------------------------------------------------------------------
The crate is now a THERMAL SOLID. The old TUNE block described the gas
regime (crate temperature = hot gas the plume advected away) — that was the
§9.5 regression, and it is FIXED (f5e9aa3 + 312e984 + 6f57762). What the
crate looks like now:

  * medium is chosen by `thermal_mass > 0`, not by `permeability <= 0`;
  * `temperature[]` on the crate is OWNED by the TemperatureSolver — the EOS
    pass no longer touches it (measured: 0.000 game/tick, every run);
  * the ambient decay is therefore the crate's ONE loss channel, and a REAL
    dial.

The equilibrium is exact (measured to +/-1% at equilibrium, 3 operating
points), so you can predict a move before you make it:

    T*(I) = k_fire_heat * I * 2^(cool_shift - heat_inv_shift)

    heat_inv_shift = log2(thermal_mass); furniture thermal_mass = 8 -> 3.
    So at cool_shift=5:  T* = 4 * k_fire_heat * I   (design §2.5's form).

--------------------------------------------------------------------------
DIAL MOVED 2026-07-30 by the COOL-SHIFT AXIS (the loss-side twin of
thermal_mass). READ THIS IF YOU HAVE THE OLD KEY IN MUSCLE MEMORY.
--------------------------------------------------------------------------
The decay shift is now a PER-MATERIAL column:

    --set materials.furniture.cool_shift=12      <-- the crate dial NOW
    --set physics.thermal.COOL_SHIFT=12          <-- NO LONGER MOVES THE CRATE

`[physics.thermal] COOL_SHIFT` is kept and still has two jobs — it is the
DEFAULT for a material row that omits the column, and (with COOL_SHIFT_VACUUM)
it defines the vacuum-exposure OFFSET — but every row in config.toml now
authors `cool_shift` explicitly, so the column WINS and overriding the global
alone changes nothing on the crate. Warning 1 below used to say this dial was
global; that WAS the problem, and it is what the axis fixed.

Two warnings before you turn a dial:

  1. cool_shift is PER MATERIAL now — that is the point. Move the crate
     alone with `materials.furniture.cool_shift` and hull/steel/glass/wood/
     doors keep their 1.3 s e-fold, so the big feel change (and the goldens
     it would move) stays confined to the thing you are tuning. Legal range
     [SHIFT_MIN=2, 20]; e-fold = 2^shift / 24 s (5 -> 1.3 s, 12 -> 171 s).
     A vacuum-exposed tile takes `max(2, cool_shift - 2)` — the 4x space
     discount stays one global rule, so each material keeps ONE dial.

  2. THE I_crit CLIFF (bench report §4). Deposit is linear in I and loss is
     linear in T, so T* is linear in I — and the hot gate opens at
     fire_T_ext. Below

         I_crit = I_peak * fire_T_ext / T_flame

     the gate closes and the fire self-collapses. At §9.3's own targets
     (T_flame 450 @ I_peak 0.5, fire_T_ext 250) that is I_crit = 0.278,
     nearly 3x ignition_seed = 0.1 — so the fire can neither ignite from
     the seed nor burn down gracefully. This is why design §2.5's
     k_fire_heat = 225 @ cool_shift = 5 is arithmetically right but
     DYNAMICALLY DEAD: measured, it snaps out at tick 1.

     The four levers are measured in bench report §4.3. Three of them
     (cool_shift ~12; ignition_seed into the band; fire_T_ext below
     T*(I_seed)) relocate the cliff; only the fourth — making the deposit
     non-linear in I — removes it, and that is a MODEL change, not a dial.
     Erik's call, at the joint re-tune.

THE TUNE BLOCK BELOW WAS RE-DERIVED 2026-07-30 for Erik's tuning session
(session seed `docs/fire_tuning_session_seed_2026-07-30.md`). Its values are
DERIVED FROM THE ANALYTICS, not measured — each carries its provenance inline.
The previous, measured cool_shift-12 set is preserved as ALT_MEASURED_CS12
below; ALT_* are all swap-in-wholesale (`TUNE.update(ALT_...)`).

--------------------------------------------------------------------------
DEPENDENCIES (settled — audit Patch A / A1, 2026-08-04)
--------------------------------------------------------------------------
An earlier REQUIRES banner here warned that the committed harness lacked the
warm seed and the `hot` / `Tfar_game` / `X_local` CSV columns, and that the
scorecard would therefore read n/a or I = 0. **All four are present** and were
verified on this branch: the warm seed at fire_timing_harness.py:349
(`gmap.temperature[cy, cx] = quantize_scalar(280.0)`) and the three columns in
the CSV header at :591-592. tools/fire_tune_plot.py is committed too. The
banner described a state that has not existed since those files were committed,
so it was deleted rather than left to mislead the next reader.
"""
from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from pathlib import Path

# ===========================================================================
# THE TUNE BLOCK — edit me, run me.           (bare key -> [physics.fire];
#                                              dotted  -> that config section)
# ===========================================================================
TUNE = {
    # =====================================================================
    # STRUCTURE FACTS (re-measured 2026-07-30, live path, post-arc):
    #  * The crate is a THERMAL SOLID (thermal_mass 8 -> heat_inv_shift 3).
    #    The ambient decay applies to it; furniture kappa=0, so
    #    `materials.furniture.cool_shift` is its ONLY loss channel
    #    (ruling A5 — deliberate, one clean dial, now per material).
    #  * The EOS pass does NOT advect its temperature away any more.
    #    Measured EOS delta on the crate: 0.000 game/tick, all 70 runs.
    #  * fire_T_ext can go back to PHYSICAL values (§9.2's below-ignition
    #    rule still applies: ignition_temp furniture = 280).
    # =====================================================================

    # =====================================================================
    # ###  OLD LAW (pre-P-R3), KEPT FOR ARCHAEOLOGY — DO NOT TUNE FROM IT  ###
    # The growth term's capacity was the hardwired `(1-I)`, so the block below
    # solves `I_eq = 1 - r(1-a)/a`. That law was REPLACED on 2026-07-31 by the
    # CAPACITY LAW (see the block immediately after this one, which supersedes
    # everything here). Every relation below that mentions `r` doing two jobs is
    # a description of the DEFECT, not of the engine. Kept because it is the
    # derivation trail for h_min, I_sustain and the Q16 quantum — all three
    # survive the change unaltered — and because the measured runs quoted in it
    # are the evidence the ruling stands on.
    # =====================================================================
    # THE VIABLE REGION — DERIVED 2026-07-30 (this pass). Read this BEFORE
    # you move a dial; three previous passes each died by moving one dial
    # against an assumed-away coupling.
    # =====================================================================
    # Fixed by Erik: r = k_die/k_grow = 0.080, k_grow 3.5 / k_die 0.28,
    # cool_shift 9 on BOTH wood and furniture, thermal_mass 8 -> shift 3.
    # Free: fire_T_ext, fire_T_span, k_fire_heat, ignition_seed.
    #
    # The five relations that actually govern the bench (all re-verified
    # against fire_simulation.cpp:150-245 and temperature_solver.cpp:264/458
    # this pass — the gain is `deposit >> heat_inv_shift` vs `T -= T >> cool_shift`,
    # so equilibrium is exactly deposit * 2^(cool_shift - heat_inv_shift)):
    #
    #   gain      = k_fire_heat * 2^(cool_shift - heat_inv_shift) = 64*k
    #   T*(I)     = gain * I
    #   hot(T)    = clamp01((T - fire_T_ext)/fire_T_span)
    #   a         = F * o2f * hot        F = 1 pristine, o2f(X=0.21) = 0.091954
    #   I_eq      = 1 - r(1-a)/a                     = 0.2100  at hot = 1
    #   h_min     = [r/(1+r)]/o2f                    = 0.80556
    #   T_sustain = fire_T_ext + fire_T_span*h_min
    #   I_sustain = T_sustain / gain
    #
    # THE FIVE CONSTRAINTS, solved:
    #   C1 plateau 400-500 game   ->  k_fire_heat in [29.76, 37.20]  (at r=0.080,
    #                                 I_eq = 0.21, cool_shift 9). ONE interval —
    #                                 k is fully pinned by C1 alone.
    #   C2 seed margin >= 15%     ->  ignition_seed >= 1.15 * I_sustain,
    #                                 and ignition_seed < I_eq = 0.21.
    #   C3 warm-seed gate (T=280) ->  T_sustain <= 280, i.e.
    #                                 fire_T_ext + 0.8056*fire_T_span <= 280.
    #   C4 §9.2                   ->  fire_T_ext < 280 (furniture) and < 300 (wood).
    #   C5 physical               ->  fire_T_ext in [140, 215] game
    #                                 = [573, 723] K = [300, 450] C (wood
    #                                 pyrolysis-sustain). C5 is STRICTLY TIGHTER
    #                                 than C4, so C4 is never the binding one.
    #
    # The region is therefore the box  fire_T_ext in [140, 215],
    # fire_T_span in (0, (280 - fire_T_ext)/0.8056],  k in [29.8, 37.2],
    # seed in [1.15*I_sustain, 0.21).  How much room that leaves, as the
    # bootstrap ratio  I_sustain/I_eq = T_sustain/T_plateau  (SMALLER IS SAFER —
    # the last pass died at 0.746, i.e. the fire had to be BORN at 75% of its
    # final size).  At k = 33 (plateau 443.5):
    #
    #   fire_T_ext ->   140     160     180     200     215     (game)
    #   (Kelvin)       573     613     653     693     723
    #   span  20      0.352   0.397   0.442   0.487   0.521     <- ratio
    #   span  40      0.388   0.433   0.478   0.524   0.557
    #   span  60      0.425   0.470   0.515   0.560   0.594
    #   span  80      0.461   0.506   0.551   0.596   0.630
    #   span 100      0.497   0.542   0.587    --      --       (C3 fails)
    #
    # Every cell above is viable; the corner (140, 20) is the safest and
    # (215, 80) the tightest. Erik has ~2x of room in the bootstrap ratio.
    #
    # ** C6 — THE COUPLING NOBODY HAS WRITTEN DOWN YET, and the one most
    # likely to bite next. The sustain condition a/(1-a) > r is symmetric in
    # `hot` and `o2f`: exactly as there is an h_min, there is an o2f floor
    #     o2f_min = r/(1+r) = 0.074074  (at hot = 1, F = 1)
    #  -> local X floor = X_ext + o2f_min*(X_full - X_ext) = 0.19444.
    # Ambient is 0.21. The flame ring may therefore lose only 7.4% of its
    # oxygen, RELATIVE, before the fire dies at ANY temperature and ANY
    # k_fire_heat. Worse, I_eq is hypersensitive there: X 0.210 -> I_eq 0.210,
    # X 0.205 -> 0.152, X 0.200 -> 0.086, X 0.1944 -> 0. No dial in this file
    # moves that floor (it is r and the o2 span); only the O2 SUPPLY (sky tau,
    # ventilation) or burn_rate move the draw. MEASURE X_local, do not assume it.
    #
    # ---------------------------------------------------------------------
    # MEASURED 2026-07-30 AT EXACTLY THE DIALS BELOW — **THE CRATE BURNS.**
    # (bench: 900 s window, sky tau 60, sponge 8, warm seed T = 280)
    #   peak I 0.1656 @ 15.1 s | max T 336.3 game (966 K) @ 38.8 s
    #   death 334.2 s (5.57 min) | wall_hp 26.688/30 -> charred remains
    #   flame-ring X min 0.19803 | far-field X min 0.2005 | N_total min 0.957
    #   far-field T rise 88.6 game (177 K)
    #
    # THE MODEL HELD, with ONE named correction:
    #  * I_sustain = 0.1005 predicted the collapse EXACTLY. I coasted at
    #    0.098-0.11 for ~200 s, then crossed ~0.10, `hot` fell off 1.0, and the
    #    death spiral ran. h_min = 0.806 likewise: hot was 0.850 at death-10 s,
    #    0.549 at death-5 s. The corrected sustain relations are CONFIRMED.
    #  * T*(I) = gain*I under-predicts by a FLAT +6.3% (measured 1.060-1.068
    #    over 233 quasi-equilibrium samples, both bench runs). That is NOT
    #    model error — it is the plume->T shim, which writes temperature[]
    #    DIRECTLY and so BYPASSES the thermal_mass shift:
    #       dT/tick = (k_fire_heat*I >> heat_inv_shift)
    #                 + fire_pressure_gain*temp_gain_scale*dt*sat * I
    #               = 4.1250*I + 0.2600*I        (sat = 1 - T/T_FLAME_MAX)
    #    -> shim share 5.6-6.1% of the deposit; predicted T_meas/T*_pred
    #       1.0652 @ T=280 and 1.0612 @ T=383.5 vs MEASURED 1.066 and 1.060.
    #    The residual is closed. Use  T*(I) = 1.063 * gain * I  when you need
    #    2-figure accuracy. (The shim is PARKED pending a radiation design
    #    decision — do not "fix" it here.)
    #
    # C6 CONFIRMED BY PROBE (sky tau 60 -> 10, the O2-SUPPLY dial, no fire dial
    # moved): flame-ring X min 0.19803 -> 0.20174, and with it peak I
    # 0.1656 -> 0.1778 and max T 336.3 -> 383.5 (1060 K). The plateau is
    # O2-LIMITED, not heat-limited: `hot` sat at 1.0 the whole burn in both
    # runs. I_eq recomputed from the MEASURED flame-ring X tracks the decline
    # (X 0.2099 -> I_eq 0.210; X 0.2037 -> 0.130; X 0.2015 -> 0.103), i.e. a
    # 1-3% RELATIVE O2 dip costs half the fire. Death is O2+heat-governed, NOT
    # fuel-governed: 89% of the crate's hp was still there when it went out.
    # ---------------------------------------------------------------------
    # =====================================================================

    # =====================================================================
    # ##  CAPACITY LAW (P-R3, ruling 2026-07-31 A3) — SUPERSEDES THE BLOCK  ##
    # ##  ABOVE. This is the algebra the engine runs as of 2026-07-31.      ##
    # =====================================================================
    # WHAT CHANGED. The growth term's carrying capacity is no longer the
    # hardwired constant 1 (the `(1-I)` factor); it is RESOURCE-PROPORTIONAL,
    # `I_cap = c*a`. In the solver (fire_simulation.cpp, PINNED order):
    #
    #     gap  = avail*hot - I/c                    <- SIGNED; negative = shrink
    #     grow = k_grow * I * gap * (1 + k_wind_fan*W)
    #     die  = k_die*(1 - avail*hot)*I + k_wind_strip*W*(1-I)*I     [UNCHANGED]
    #
    # i.e. the logistic `k_grow*a*I*(1 - I/(c*a))` with `a` cancelled out of the
    # bracket (which is why no division survives in the sim path).
    #
    # WHY. Under the old law `r = k_die/k_grow` set BOTH the equilibrium
    # intensity AND the extinction wall (`I_eq = 1 - r(1-a)/a`), so asking for a
    # small fire FORCED `r` up against the operating point. That is the "one
    # margin governs everything" finding: the product `F*o2f*hot` could fall only
    # to 80.5% of ambient before the fire died at ANY temperature — which is why
    # a crate could never lose more than 19.5% of its hp (measured: 25.53/30 left
    # at wall_damage 0.55) and why `o2_frac_ext` = 0.13 was dead code. Each dial
    # now has EXACTLY ONE JOB:
    #
    #     I_cap_per_avail (c)  SIZE    I_eq ~= c*a
    #     k_grow               TEMPO   ramp e-fold ~= 1/(k_grow*a)
    #     k_die (r)            DEATH   where the wall sits, and nothing else
    #
    # THE GOVERNING RELATIONS (the five, restated for this law):
    #
    #     gain[mat] = k_fire_heat * 2^(cool_shift[mat] - log2(thermal_mass[mat]))
    #     T*(I)     = gain * I                                        [UNCHANGED]
    #     hot(T)    = clamp01((T - fire_T_ext[mat]) / fire_T_span)
    #                 <- fire_T_ext is PER MATERIAL now: ignition_temp[mat]
    #                    - ignition_to_ext_delta. furniture 280-100 = 180,
    #                    wood 300-100 = 200. fire_T_span is still GLOBAL.
    #     a         = F * o2f * hot        F = 1 pristine, o2f(X=0.21) = 0.091954
    #     I_eq      = c * (a - r*(1-a))                       <- THE NEW ONE
    #     sustain  <=>  a > r/(1+r)                           [SAME SHAPE]
    #     h_min     = [r/(1+r)] / o2f                         [SAME SHAPE]
    #     T_sustain = fire_T_ext[mat] + fire_T_span*h_min
    #     I_sustain = T_sustain / gain[mat]
    #
    # THE Q16.16 QUANTUM CONDITION (the 95bdec0 trap, restated for this law —
    # growth that truncates to zero cannot grow, however right the algebra is):
    #
    #     dt * k_grow * seed * (a - seed/c) * 65536  >=  2      counts/tick
    #
    # AT THE P-R3 DIALS (c 2.53, k_grow 3.5, k_die 0.035 -> r = 0.010):
    #   I_eq(X):  0.21 -> 0.210 | 0.25 -> 0.328 | 0.30 -> 0.474 | pure O2 -> 1.0
    #             LINEAR in X (Erik ruling R-a), not the old law's hyperbola.
    #   headroom on F*o2f*hot     9.3x   (was 1.242x)
    #   local X floor             0.1386 (was 0.1944 — o2_frac_ext is LIVE again)
    #   h_min                     0.1077 (was 0.806)
    #   isolated hp floor         3.2/30 (was 24.2/30)
    #   seed quantum @ 0.12       40 counts/tick net  (MEASURED: 40)
    #
    # ** MEASURED 2026-07-31 (the P-R3 gate benches) — READ BEFORE TUNING. **
    #   THE LAW IS CONFIRMED: over a 900 s burn at wall_damage 0.083 the measured
    #   I tracks I_eq = c*(F*o2f(X_local)*hot - r*(1 - F*o2f*hot)) to mean 0.66%
    #   / worst 2.39% across 20,880 plateau ticks.
    #
    #   BUT THE HP FLOOR IS NOT THE BINDING LIMIT AT THESE DIALS. At
    #   wall_damage 0.55 the crate burns to hp 14.87/30 (50% consumed — up from
    #   the old law's 25.53/30, i.e. 15%), then dies THERMALLY, not for lack of
    #   fuel or oxygen: X_local at death is 0.2098, i.e. AMBIENT. The chain is
    #       F falls -> I_eq falls -> T* = gain*I falls -> hot falls -> a falls
    #   and `hot` stays pinned at 1 only while
    #       gain * c * (F*o2f*(1+r) - r)  >=  fire_T_ext + fire_T_span
    #   which at gain 2112, c 2.53, o2f 0.0897 needs F >= 0.565 (hp >= 17.0).
    #   MEASURED knee: `hot` left 1.0 at hp 15.88 (F = 0.529); death 30 s later.
    #   This is the I_crit cliff of warning 2 above in new clothes — a THERMAL
    #   limit (k_fire_heat / cool_shift / fire_T_ext / span). The capacity law
    #   does not and cannot move it. The levers that do: raise `gain`
    #   (k_fire_heat or cool_shift) or lower fire_T_ext + fire_T_span. NOTE the
    #   ruling retires k_fire_heat at P-R4 (replaced by the combustion-side
    #   `H_bed`, which makes the plateau O2-proportional) and re-tunes the rest
    #   at P-R5 — so do NOT chase this with `c`: `c` only moves SIZE, which is
    #   the entire point of the patch. **
    # =====================================================================

    # -- 1. STRUCTURE (§9.3 step 1). CHOSEN POINT: fire_T_ext 180, span 40. --
    #    Erik's candidate, verified above against C1-C5 — all five PASS:
    #      gain 2112, T_plateau 443.5 (1180 K), T_sustain 212.2,
    #      I_sustain 0.1005, bootstrap ratio 0.478 (vs 0.746 that failed),
    #      warm-seed hot clamps to 1.0 at tick 1 (needs 0.806),
    #      fire_T_ext 653 K = 380 C — mid-band for wood pyrolysis sustain.
    #    span 40 (not 100) is what buys the ratio: it is the h_min*span term,
    #    not fire_T_ext, that dominated T_sustain in the failed passes.
    # =====================================================================
    # ## P-F1b RECALIBRATION (2026-08-02, commit 4133512) — THE EXECUTED SET ##
    # The values below are P-F1b's "the fires live again" package-A dial set,
    # loaded 2026-08-13 for Erik's hand-tuning session (its HUMAN-TEST).
    # Derivations + raw numbers: docs/fire_recalibration_2026-08-02.md (in the
    # pf1b-recalibration commit). Measured there: ignites from seed, ramps
    # ~20 s, holds 836-856 K, spreads a 1-gap in ~30 s, burns 8 min (kindling)
    # / 24 min (furniture), dies BY THE TEMPERATURE GATE at 54-61% fuel gone —
    # or suffocates ~2 min in a sealed room.
    # =====================================================================
    # fire_T_ext DROPPED from the executed dict 2026-08-13: it is the INERT
    # config fallback since ruling A3 (extinction is per-material:
    # ignition_temp[mat] - ignition_to_ext_delta). The old executed value was
    # 180.0; the LIVE foot dial is ignition_to_ext_delta below.
    "ignition_to_ext_delta": 200.0,  # P-F1b: 100 -> 200, knee geometry (the FOOT)
    "fire_T_span": 180.0,   # P-F1b: 150 -> 180, knee geometry (the WIDTH; was 40 here)

    # -- 2. THERMAL operating point (§9.3 step 2). THE dial pair.
    #       T* = k_fire_heat * I * 2^(cool_shift - 3)   [+/-1% at equilibrium]
    #       Raising k_fire_heat raises the plateau AND the far-field rise (they
    #       scale together — far rise is the binding constraint, not the flame
    #       temperature). Since the COOL-SHIFT AXIS the shift is the FURNITURE
    #       row only: every other material keeps its 1.3 s e-fold, so a big
    #       number here does not follow you onto the hull.
    #
    #   DERIVED START, 2026-07-30 (session-seed handoff). NOT measured — this
    #   run is the first measurement. Provenance of each number:
    #
    #   k_fire_heat 33: from the verified analytic
    #       T* = k_fire_heat * I * 2^(cool_shift - heat_inv_shift),
    #   heat_inv_shift = log2(furniture thermal_mass 8) = 3, so at cool_shift 9
    #       T* = k * I * 2^6 = 64 k I;  at Erik's I anchor 0.21 -> T* = 13.4 k.
    #   The §9.3 flame band is 400-500 game, so k ~= 33.5 lands T* ~= 450
    #       (= 1193 K). 33 is that, rounded.
    # *** k_fire_heat TOMBSTONE (P-R4, 2026-08-01) ***  THE PAINTER IS GONE
    # (docs/radiation_raycaster_extinction_ruling_2026-07-31.md A1). Nothing
    # reads this key any more; it is left here ONLY so the derivations above
    # stay readable. The plateau's gain is now the combustion FUEL-BED deposit:
    #
    #   T* = H_bed * (burn_rate*dt*I*o2f*claim_faces) * 2^(cool_shift - his)
    #
    # with H_bed = [physics.combustion] H_BED_M * 2^H_BED_SHIFT and claim_faces
    # = the open air faces filing a demand share (4 for a crate in open air).
    # MEASURED CAVEAT, P-R4 gate (f): at the anchored burn_rate 0.02 the
    # per-claimant demand is ~1 Q16.16 COUNT at the operating point, so this
    # gain is a STAIRCASE in I with a dead zone below I ~ 0.199 (ambient o2f) —
    # see the P-R4 report / E1 escalation before trusting the smooth form.
    #
    # THE KEY IS GONE FROM THIS DICT (audit Patch A / A1, 2026-08-04). It was
    # still being SET here long after P-R4 retired it from [physics.fire], so
    # `_resolve_key` raised KeyError and this tool could not execute at all.
    # The derived value (k = 33, from T* = 13.4 k against the 400-500 band
    # above) is preserved in the prose; only the executed entry is deleted.
    #   cool_shift 9: e-fold 2^9/24 = 21.3 s — a physically plausible wood-
    #   surface time constant (12 = 171 s was the cliff-clearing crutch, not a
    #   wood number). Integer shift. NOTE THE KEY: the old
    #   `physics.thermal.COOL_SHIFT` no longer moves the crate (see the header).
    "materials.furniture.cool_shift": 13,  # P-F1b: 9 -> 13, THE SPREAD DIAL
    #   WOOD MOVES WITH FURNITURE (2026-07-30). Wood and furniture are both
    #   cellulosic and both ship thermal_mass = 8; the seeded 5 was a
    #   byte-identity placeholder, not a physical choice. Raising ONLY furniture
    #   would leave wood at
    #       I_crit = fire_T_ext / (k_fire_heat * 2^(cool_shift-3))
    #              = 250 / (33 * 2^2) = 1.89 > 1,
    #   i.e. wooden WALLS become permanently non-flammable. Both rows at 9 give
    #   both materials I_crit = 250/(33*2^6) = 0.118.
    "materials.wood.cool_shift": 13,       # P-F1b: moves with furniture
    "materials.kindling.cool_shift": 13,   # P-F1b: 9 -> 13 (P-F4a's reference object)

    # -- 3. RAMP (§9.3 step 3 — YOURS). Ratio sets peak, magnitude sets
    #       speed.
    #   k_die/k_grow = 0.080 is ERIK'S EXPLICIT CHOICE (2026-07-30). The
    #   logistic (fire_simulation.cpp:209-224) is
    #       dI/dt = I * [ k_grow*a*(1-I) - k_die*(1-a) ],  a = F * o2f * hot
    #   so the fixed point is  I_eq = 1 - (k_die/k_grow) * (1-a)/a  and the
    #   fire only sustains at all while  k_die/k_grow < a/(1-a).
    #   Since the full-response split (b340bba) o2f = (X-0.13)/(1.0-0.13), so
    #   ambient air gives o2f = 0.092. At F = 1, hot = 1 that is a = 0.092,
    #   a/(1-a) = 0.1013, and 0.080 lands I_eq = 0.210 — exactly Erik's anchor
    #   (0.49 @ X=0.25, 0.67 @ X=0.30, 1.00 at pure O2: normal air is not a
    #   maximum, which is the point).
    #   ** CAVEAT RESOLVED 2026-07-30 by the FUEL-FRACTION AXIS. It used to read:
    #   "F = clamp01(wall_hp/fuel_ref) and fuel_ref below is 40 while furniture
    #   hp is 30, so a FULL-HEALTH crate has F = 0.75, not 1" — and with the
    #   SHIPPED fuel_ref = 60 it was worse still (F = 0.5, ceiling 0.048, i.e.
    #   no sustain at any intensity or temperature). F now normalises against
    #   the tile's OWN material hp, so a pristine crate reads F = 1 and the
    #   arithmetic above is the arithmetic that runs: ceiling 0.1013 > 0.080,
    #   I_eq = 0.210. VERIFIED at the solver with `hot` pinned to 1: the fire
    #   settles at I = 0.2043 (the 3% shortfall is Q16.16 truncation near the
    #   fixed point), where the pre-patch F = 0.5 crate collapses to 0. **
    #
    #   ** STILL NOT ENOUGH ON THE BENCH — two limiters sit BEHIND the fuel one,
    #   both measured 2026-07-30 at exactly these dials, both Erik's call:
    #     (1) THE I_crit CLIFF (bench report §4, warning 2 above). T*(I_seed) =
    #         k_fire_heat * 0.1 * 2^(9-3) = 211 game, and fire_T_ext is 250, so
    #         `hot` collapses from the warm seed's 0.30 toward 0 and takes `a`
    #         with it. Measured: peak I 0.0999 (never above the seed), max
    #         T 279.9, max hot 0.2989, death 0.87 min.
    #     (2) THE Q16.16 GROWTH QUANTUM. Even with hot pinned at 1, the LARGEST
    #         dI/dt anywhere on this logistic is 3.547e-4 /s == 0.969 Q16.16
    #         counts per 1/24 s tick — it TRUNCATES TO ZERO. The ratio 0.080 is
    #         right; the MAGNITUDE (k_grow 0.35 / k_die 0.028) is below the
    #         fixed-point tick quantum. The same ratio at 5x/10x magnitude
    #         converges to I = 0.198 / 0.204 as the analytic says. **
    #   MAGNITUDE RAISED 10x 2026-07-30 — SAME RATIO 0.080 (Erik's anchor, which
    #   is what sets I_eq; only the magnitude, i.e. the ramp speed, changes).
    #   At the old 0.35/0.028 the net growth peaked at 3.54e-4 /s == 0.969
    #   Q16.16 counts per 1/24 s tick, which TRUNCATES TO ZERO: the fire could
    #   not grow because its growth rounded away. 3.5/0.28 puts the same
    #   logistic ~9.7 counts/tick at the seed — above the quantum.
    #   CAPACITY LAW (P-R3, 2026-07-31): k_grow now moves ONLY the tempo — the
    #   ramp e-fold is ~1/(k_grow*a) ~ 3 s at ambient. Erik may slow it later
    #   without touching the fire's size or its death wall, which is new.
    "k_grow": 0.5,           # P-F1b: 4.0 -> 2.0, TEMPO; 2026-08-13 Erik: 2.0 -> 0.5, slower ramp
    #   THE SIZE DIAL (P-R3). I_eq ~= c*a, so c alone answers "how big is a fire
    #   in ordinary air": 2.53 * 0.09195 = 0.2100, exactly Erik's anchor. Raise
    #   it for bigger fires everywhere; it does NOT move the death wall.
    "I_cap_per_avail": 14.0,  # P-F1b: 2.53 -> 14.0, SIZE
    #   k_die RETUNED 0.28 -> 0.035 (r 0.080 -> 0.010) BY THE CAPACITY LAW. `r`
    #   no longer has to sit near the operating point to hold the fire small
    #   (that is c's job now), so it is free to put the death wall at the
    #   PHYSICAL limits: X floor 0.1944 -> 0.1386 (o2_frac_ext 0.13 becomes
    #   reachable again), headroom on F*o2f*hot 1.242x -> 9.3x.
    "k_die": 0.008,  # P-F1b: 2.0 -> 0.008 (r 0.5 -> 0.004) — the death wall; puts the
                     # logistic's extinction at X = 0.1335, just ABOVE o2_frac_ext=0.13,
                     # so THE OXYGEN LIMIT IS THE BINDING ONE (sealed rooms suffocate)

    # -- 3b. IGNITION SEED. Raised 0.1 -> 0.15 (2026-07-30). The sustain floor
    #    is I_crit = I_peak * fire_T_ext / T_flame = 0.21 * 250/450 = 0.117, so a
    #    0.1 seed is BORN BELOW THE FLOOR and dies at any k_fire_heat. Check that
    #    the two constraints now overlap: a plateau in the 400-500 band needs
    #    k_fire_heat in [29.8, 37.2]; seed survival needs
    #    k_fire_heat > fire_T_ext / (seed * 2^6) = 250/(0.15*64) = 26.0.
    #    k_fire_heat = 33 is inside BOTH (at seed 0.1 the floor was 39.1 > 37.2 —
    #    empty intersection, which is why nothing lit).
    #
    #    ** MEASURED AT THESE FOUR DIALS, 2026-07-30 — STILL DOES NOT BURN, and
    #    ** SUPERSEDED 2026-07-30 by the derived region at the top of this block:
    #    seed is now 0.12 against I_sustain 0.1005 (margin 19.4%), and the
    #    STRUCTURE dials moved (fire_T_ext 250 -> 180, span 100 -> 40) which is
    #    what made a sub-0.21 seed viable at all. The post-mortem below is kept
    #    because it is the derivation of h_min. **
    #    the reason is that `I_crit` ABOVE IS THE WRONG THRESHOLD. It is derived
    #    from `hot > 0` (T > fire_T_ext), but the logistic does not sustain at
    #    hot > 0 — it sustains only where  a/(1-a) > k_die/k_grow,  and with
    #    a = F * o2f * hot at F = 1, o2f = 0.092 (ambient air) that needs
    #        hot > (r/(1+r))/o2f = 0.0741/0.0920 = 0.806,   r = k_die/k_grow,
    #    i.e. T > fire_T_ext + fire_T_span*0.806 = 330.6 game, NOT 250. So the
    #    real floor carries fire_T_span:
    #        I_sustain = (fire_T_ext + fire_T_span*h_min)
    #                    / (k_fire_heat * 2^(cool_shift - heat_inv_shift))
    #                  = 330.56 / 2112 = 0.1565.
    #    The seed 0.15 is 4.2% BELOW it, so T tops out at T*(0.15) = 316.8
    #    (hot ceiling 0.668 < 0.806 needed) and the fire cannot bootstrap at ANY
    #    speed. Measured: peak I 0.1488 @ tick 1 (never above the seed), peak T
    #    281.3 @ 1.0 s, hot never above 0.313, I == 0 at 8.9 s, wall_hp 29.948
    #    (0.05 of 30 consumed). Full numbers in the run report. **
    #    CHOSEN 2026-07-30: 0.12 = 1.194 * I_sustain (0.1005) — 19.4% margin,
    #    above C2's 15% floor, and 57% of I_eq so the ramp is still visible.
    #    Q16.16 check at this seed with hot = 1: grow 2226 counts, die 1999,
    #    net delta = +9 counts/tick — above the growth quantum (the 95bdec0 trap).
    "ignition_seed": 0.12,

    # -- 4. LIFETIME (§9.3 step 4 — YOURS). NOT re-derived for the 2026-07-30
    #       start: carried over from the cool_shift-12 point (where it measured
    #       death at 332.7 s with wall_hp 4.5 left). Lifetime is stage 4 —
    #       settle steps 2-3 first, then move this. --
    "wall_damage": 0.03,     # P-F1b: 0.4 -> 0.03, BURN DURATION (8 min kindling / 24 min furniture)

    # -- P-F1b's combustion-side plateau gain (replaces the retired k_fire_heat;
    #    H_bed = H_BED_M * 2^H_BED_SHIFT: 2.023e5 -> 2.900e5) and the gate wall. --
    "physics.combustion.H_BED_M": 18125.0,   # P-F1b: 25290 -> 18125 (mantissa)
    "physics.combustion.H_BED_SHIFT": 4,     # P-F1b: 3 -> 4
    "T_emit_gate": 310.0,    # P-F1b: 180 -> 310 — who CASTS radiation (the gate wall;
                             # at 180 a receiver became an emitter too early and its
                             # ceiling collapsed to E_s/15 — spread stalled)

    # -- ANCHORED — verify, don't tune (see §9.3 for the paper trail) --
    "physics.combustion.burn_rate": 0.02,     # Huggett 1980 — THE O2-draw dial
    "physics.combustion.fuel_per_o2": 0.7,    # wood stoich (0.045 was stale)
    "o2_frac_ext": 0.13,                      # Peatross-Beyler 1997 extinction
    "o2_frac_full": 1.0,                      # full-response reference = PURE O2
                                              # (2026-07-30 split; NOT ambient —
                                              # ambient air now gives o2f 0.092,
                                              # so k_die/k_grow moves ~10x with it)
    # fuel_ref REMOVED from the TUNE block 2026-07-30 (FUEL-FRACTION AXIS). It
    # is now INERT in the live engine: F normalises against the tile's OWN
    # material hp via GameMap.fuel_recip, and the scalar survives only as the
    # solver's fallback for callers that pass no plane — which the engine never
    # does. MEASURED, so nobody has to take it on faith: this bench at
    # fuel_ref = 40 and at fuel_ref = 1000 is byte-identical over 2691 ticks
    # (only the CSV's recorded override string differs).
    #   THE CRATE DIAL IS NOW `materials.furniture.hp` — and it is not a fire
    #   dial, it is the crate's HEALTH. Changing it changes how much punishment
    #   a crate takes as well as how long it burns. That coupling is the point
    #   (fuel IS mass); it is not something to reach for casually.
    "k_wind_strip": 0.0,                      # plume self-blow-out off (2026-07-23)
}

# --- Other measured branches (bench report §4.3). Swap one in wholesale by
# --- doing  TUNE.update(ALT_...)  right here, then re-run.
#
# EACH BLOCK'S `k_fire_heat` WAS STRIPPED (audit Patch A / A1, 2026-08-04) for
# the same reason as the main TUNE dict: the key is retired, so any `.update()`
# from one of these would have re-broken the tool. The retired value is recorded
# in each block's comment so the archaeology survives; only the executed entry
# is gone.

# The P3 close-out's best MEASURED set (the one the 2026-07-30 re-derivation
# replaced above): 6 of 9 §9.3 targets passed — peak I 0.331 @ 143.9 s, plateau
# 414 game (1120 K), death 332.7 s with wall_hp 4.5 left. cool_shift 12 is a
# 171 s e-fold: it clears the I_crit cliff, but it is a crutch, not a wood
# number. Kept as the fallback if the derived start does not sustain.
ALT_MEASURED_CS12 = {                         # retired k_fire_heat was 2.2
    "materials.furniture.cool_shift": 12,
    "k_grow": 0.35, "k_die": 0.06,
    "fire_T_ext": 250.0, "fire_T_span": 100.0,
}

# Keeps the decay shift at Erik's stated preference (6-7) and the blessed
# fire_T_ext — but the flame runs at 1007 game (2307 K), ~2x too hot, and
# lowering k_fire_heat from here kills the fire outright (floor is 175).
ALT_PREFERRED_COOL_SHIFT = {                  # retired k_fire_heat was 175.0
    "materials.furniture.cool_shift": 7,
    "fire_T_ext": 250.0, "fire_T_span": 100.0,
}

# Lands the flame at 450 game (1193 K) and lives 208 s with the fuel actually
# consumed — but fire_T_ext = 90 is 473 K / 200 C, not a defensible
# flame-extinction temperature, and it fails gate (c) monotonicity (dip -46).
ALT_LOW_T_EXT = {                             # retired k_fire_heat was 56.25
    "materials.furniture.cool_shift": 7,
    "fire_T_ext": 90.0, "fire_T_span": 100.0,
}

# Design §2.5's arithmetic point, for the record. Plateau 449 (dead-centre)
# ONLY if ignition_seed is raised to 0.4 — and then it dies at 48.5 s with
# 24.6 wall_hp left (the cliff moves to the decay end). With the stock
# ignition_seed = 0.1 this set snaps out at tick 1.
ALT_DESIGN_2_5 = {                            # retired k_fire_heat was 225.0
    "materials.furniture.cool_shift": 5,
    "fire_T_ext": 250.0, "fire_T_span": 100.0,
    "ignition_seed": 0.4,
}

BENCH = {
    "--max-seconds": 900,      # 15 min window (death target is 6-8 min)
    "--tail-seconds": 60,
    "--sky-tau-s": 60,
    "--sponge-width": 8,
    # bench geometry: harness defaults are already sponge-safe (84x40, crate
    # deep at x=12); override here if needed: "--interior-w": 84, ...
}

# Targets for the scorecard (Erik 2026-07-24/25; §9.3).
T = {
    "peak_lo": 0.40, "peak_aim": 0.50, "peak_hi": 0.60,
    "peak_t_lo_s": 120.0, "peak_t_aim_s": 180.0, "peak_t_hi_s": 300.0,
    "death_lo_s": 360.0, "death_hi_s": 480.0,
    "flameT_lo": 400.0, "flameT_hi": 500.0,
    "far_rise_max": 20.0,
    "ntot_min": 0.90,
    "farX_min": 0.19,
    "I_min_snap": 0.02,
}

# ===========================================================================
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "_fire_tuning_artifacts" / "tune_loop_last.csv"


def run_harness():
    cmd = [sys.executable, str(ROOT / "tools" / "fire_timing_harness.py"),
           "--csv", str(CSV_PATH)]
    for k, v in BENCH.items():
        cmd += [k, str(v)]
    for k, v in TUNE.items():
        # shift counts are INTEGERS (the loader rejects a fractional value)
        int_key = ("COOL" in k) or k.endswith("cool_shift")
        vs = str(int(v)) if float(v).is_integer() and int_key else str(v)
        cmd += ["--set", f"{k}={vs}"]
    CSV_PATH.parent.mkdir(exist_ok=True)
    print("[cmd]", " ".join(cmd), "\n")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.exit(f"harness failed (exit {r.returncode})")
    return r.stdout


def load_series():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "t_s":
                hdr = row
                continue
            rows.append({h: float(x) for h, x in zip(hdr, row)})
    return rows


def _col(s, name, reduce_fn, default=float("nan")):
    """reduce_fn over column `name`, or `default` if the harness didn't write it."""
    if not s or name not in s[0]:
        return default
    return reduce_fn([r[name] for r in s])


def metrics(s):
    m = {}
    peak = max(s, key=lambda r: r["I"])
    m["peak_I"], m["peak_t"] = peak["I"], peak["t_s"]
    i_pk = s.index(peak)
    # fire death: first snap-to-zero AFTER the peak (edge-trigger world:
    # it must STAY dead — take the last transition into I<=snap).
    death = None
    for r in s[i_pk:]:
        if r["I"] <= T["I_min_snap"]:
            death = r["t_s"]
            break
    m["death_t"] = death
    # flame plateau T: median T_game while I >= 60% of peak (the burning core)
    core = [r["T_game"] for r in s if r["I"] >= 0.6 * m["peak_I"]]
    m["flame_T"] = statistics.median(core) if core else float("nan")
    core_rows = [r for r in s if r["I"] >= 0.6 * m["peak_I"]]
    m["hot_at_plateau"] = _col(core_rows, "hot", statistics.median, 0.0) if core else 0.0
    m["far_rise"] = _col(s, "Tfar_game", max)
    m["ntot_min"] = _col(s, "Ntot_room", min)
    m["farX_min"] = _col(s, "O2far_X", min)
    m["Xlocal_min"] = _col(s, "X_local", min)
    m["hp_end"] = s[-1]["wall_hp"]
    m["end_t"] = s[-1]["t_s"]
    return m


def verdict(ok, value=None):
    """PASS/MISS, or 'n/a ' when the harness didn't write the column."""
    import math
    if value is not None and isinstance(value, float) and math.isnan(value):
        return "n/a "
    return "PASS" if ok else "MISS"


# THE TWO PREDICTED ROWS ARE GONE (audit Patch A / A1, 2026-08-04).
# `predicted_T_star` computed T* = k_fire_heat * I * 2^(cool_shift - his) and
# the `I_crit` row divided by it — both keyed on k_fire_heat, which P-R4 retired
# (see its tombstone in the TUNE block). They printed confident numbers derived
# from a dial nothing reads. Re-deriving them from the live H_bed chain is a
# physics judgement, not cleanup, so the rows were deleted rather than guessed.
# The measured rows below are unchanged.


def scorecard(m):
    print("=" * 74)
    print(f"{'metric':<26}{'value':>16}   target                    verdict")
    print("-" * 74)
    print(f"{'peak I':<26}{m['peak_I']:>16.3f}   "
          f"{T['peak_lo']}-{T['peak_hi']} (aim {T['peak_aim']})        "
          f"{verdict(T['peak_lo'] <= m['peak_I'] <= T['peak_hi'])}")
    print(f"{'peak time':<26}{m['peak_t']/60:>13.2f} min   "
          f"~{T['peak_t_aim_s']/60:.0f} min ({T['peak_t_lo_s']/60:.0f}-"
          f"{T['peak_t_hi_s']/60:.0f} ok)      "
          f"{verdict(T['peak_t_lo_s'] <= m['peak_t'] <= T['peak_t_hi_s'])}")
    d = m["death_t"]
    print(f"{'fire death':<26}"
          f"{(d/60 if d else float('nan')):>13.2f} min   "
          f"{T['death_lo_s']/60:.0f}-{T['death_hi_s']/60:.0f} min             "
          f"     {verdict(d is not None and T['death_lo_s'] <= d <= T['death_hi_s'])}")
    try:                       # Kelvin labels follow config's k_temp_to_kelvin
        from fire_tune_plot import kelvin_map   # (tools/ is sys.path[0])
        amb, slope = kelvin_map()
    except Exception:
        amb, slope = 293.0, 2.0
    print(f"{'flame plateau T (game)':<26}{m['flame_T']:>16.0f}   "
          f"{T['flameT_lo']:.0f}-{T['flameT_hi']:.0f}  "
          f"(= {amb+slope*T['flameT_lo']:.0f}-{amb+slope*T['flameT_hi']:.0f} K)  "
          f"{verdict(T['flameT_lo'] <= m['flame_T'] <= T['flameT_hi'], m['flame_T'])}")
    print(f"{'  (hot gate at plateau)':<26}{m['hot_at_plateau']:>16.2f}   "
          f"should be ~1.0 while ablaze")
    print(f"{'far-field T rise (game)':<26}{m['far_rise']:>16.1f}   "
          f"<= {T['far_rise_max']:.0f}                     "
          f"{verdict(m['far_rise'] <= T['far_rise_max'], m['far_rise'])}")
    print(f"{'room N_total min':<26}{m['ntot_min']:>16.3f}   "
          f">= {T['ntot_min']}                   "
          f"{verdict(m['ntot_min'] >= T['ntot_min'], m['ntot_min'])}")
    print(f"{'far-field X min':<26}{m['farX_min']:>16.4f}   "
          f">= {T['farX_min']} (O2 fix alive)    "
          f"{verdict(m['farX_min'] >= T['farX_min'], m['farX_min'])}")
    print(f"{'local X min (vitiation)':<26}{m['Xlocal_min']:>16.4f}   "
          f"info: should DIP near flame")
    print(f"{'wall_hp at end':<26}{m['hp_end']:>16.2f}   "
          f"info: >0 = charred remains (ok)")
    print("=" * 74)
    print(f"[csv] {CSV_PATH}")


if __name__ == "__main__":
    import os
    out = run_harness()
    tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
    print("\n".join("[harness] " + ln for ln in tail), "\n")
    scorecard(metrics(load_series()))
    # Auto-plot (I / T / room-X vs time). The window blocks until closed —
    # that's the loop rhythm: look, close, edit TUNE, re-run.
    # Disable the window (PNG still written) with:  set FIRE_TUNE_SHOW=0
    show = os.environ.get("FIRE_TUNE_SHOW", "1") != "0"
    manual = (f'    conda run -n data python '
              f'"{ROOT / "tools" / "fire_tune_plot.py"}" "{CSV_PATH}" --show')
    try:
        # tools/ is sys.path[0] when this file is run as a script.
        from fire_tune_plot import make_plot
        make_plot(CSV_PATH, show=show, targets=T)
    except Exception as e:      # matplotlib missing, no display, CSV drift, ...
        # NEVER take the run down with the plot: the scorecard above is the
        # deliverable. Print the exact command to reproduce the plot by hand.
        print(f"[plot] skipped ({type(e).__name__}: {e}). Plot it manually:")
        print(manual)
