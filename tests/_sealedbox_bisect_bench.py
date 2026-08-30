"""Issue #54 bisection bench — the sealed-box probe, one solver term per run.

The all-systems scenario (2026-08-27) found the cleanest #54 repro yet: a
glass box born sealed at t=0 via ``seal_tiles`` (no doors, no history, no
interior heat source) heats +124 game-deg and self-pressurizes 1.0->1.52 atm
in 18 s from a crate fire ~20 tiles away, while the arena around it COOLS.
P/T ratio ~ constant-N heating: energy, not mass, crosses the sealed wall.

This bench reduces that to the minimal deterministic probe (fire + sealed
dry box, no water/blasts/breach) and re-runs it with ONE energy-chain term
disabled per pass — the bisection the #54 session plan prescribes:

    baseline        as configured
    drag_heat       k_drag_heat_frac = 0
    drag            k_drag = 0 (whole staged momentum drag off)
    comp_work       adiabatic_index = 1.0 (compression work off)
    flat_gs         use_multigrid = False (flat RB-GS — MG wall suspect)
    no_vrail        U_MAX = 1e9 (v2.4 store-clamp rail effectively off)

All fields are live ``def_readwrite`` members of the C++ EOSSolver
(bindings.cpp), set on ``sim.physics_runner.eos`` post-construction, fresh
Simulation per variant. FIXED behavior = box dT ~ 0 while only the crate's
neighbourhood warms. The toggle that kills the box heating names the
mechanism (or flat_gs indicts the MG wall handling specifically).

HARNESS, not a pytest gate (``_`` prefix): prints the table, exits 0.

Run:
    conda run -n data python tests/_sealedbox_bisect_bench.py
    conda run -n data python tests/_sealedbox_bisect_bench.py --cuda [variant...]

P-G2b: ``--cuda`` (anywhere in argv, stripped before variant-name parsing)
selects the CUDA build + the resident-backend GPU path through the SAME
plumbing ``tools/run_on_cuda.py`` (== ``python main.py --cuda``) uses — the
project's one GPU-launch path (CLAUDE.md) — never a second launch path here.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# P-G2b: --cuda must be resolved BEFORE `import breach_physics` — whichever
# build lands in sys.modules first wins for the rest of this process, exactly
# the constraint tools/run_on_cuda.py's own docstring states.
_USE_CUDA = "--cuda" in sys.argv
if _USE_CUDA:
    sys.argv.remove("--cuda")
    sys.path.insert(0, str(ROOT / "tools"))
    from run_on_cuda import enable_all_backends, setup_cuda_import  # noqa: E402
    setup_cuda_import()
    if str(ROOT / "tests") not in sys.path:
        sys.path.insert(0, str(ROOT / "tests"))
else:
    for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import materials  # noqa: E402
from simulation.payloads import ignite_ring  # noqa: E402

if _USE_CUDA:
    enable_all_backends(bp)

TPS = 24
END_TICK = 18 * TPS
IGNITE_TICK = 2 * TPS
CRATE = (26, 41)                 # mid-arena crate stack — the only heat source
AQ_BOX = (50, 58, 24, 32)        # sealed box, built on open arena floor
AQ_IN = np.s_[51:58, 25:32]
BUNKER = np.s_[27:42, 83:96]     # #54 bench R6 (steel, doored)
PEN = np.s_[49:66, 83:96]        # #54 bench R8 (glass, sealed)
ARENA = np.s_[3:67, 3:58]

VARIANTS = [
    ("baseline",  {}),
    ("drag",      {"k_drag": 0.0}),
    ("comp_work", {"adiabatic_index": 1.0}),
    ("flat_gs",   {"use_multigrid": False}),
    ("no_vrail",  {"U_MAX": 1e9}),
    # MG thin-wall probe (2026-08-29): same box, 2- and 3-tile glass walls.
    ("wall2",       {}, 2),
    ("wall3",       {}, 3),
    # 2026-08-29: does the sealed box drift with NO forcing at all? If P
    # still climbs, the driver is the box's own equation, not the fire.
    ("nofire",        {}, 1, False),
    # arc #54 P-G1a CONTROL: the same run with the TemperatureSolver's
    # conduction pass switched off (every face -> NO_FACE). P-G1a moves the
    # EOS's energy chain onto the conservative flux form but leaves the
    # thermal solver on the T side (design §2.7 row 3 = P-G1b), so this
    # variant isolates THIS PATCH's contribution to the sealed box. It is the
    # gate's honest denominator, not a setting anyone plays with.
    ("nofire_nocond", {}, 1, False, True),
]
# RETIRED VARIANTS (gas-energy conservation arc #54, P-G1a). Every variant
# below keyed a dial this arc deleted, so they are gone rather than silently
# no-oping (a bisection bench whose knob does nothing is worse than no bench):
#   drag_heat        k_drag_heat_frac = 0   -> D5: the deposit constant is
#                                             DERIVED (k_ke) now, not dialled
#   comp_clamp0      T_WORK_CLAMP = 0       -> D11: there is no step-4c work
#   stiff_K          + adiabatic_index      -> term left to clamp; the whole
#   flat_S128_clamp0                           point of `_clamp0` was "run
#   flat_S512_clamp0                           with #54's driver off", which
#   wall2_clamp0                               is now the ONLY way it runs
#   nofire_clamp0
# `comp_work` (adiabatic_index = 1.0) SURVIVES but has changed meaning: it no
# longer zeroes a compression term, it zeroes k_work = (gamma-1)*T_AMB_K, i.e.
# the flux constant -- while ALSO stiffening the kick by 1.4x (K = c_max^2/
# gamma). Still confounded, still useful as a control.


def run_variant(name, overrides, wall_thick=1, ignite=True, no_conduction=False):
    """One fresh Simulation; ``wall_thick`` = glass ring thickness in tiles
    (2026-08-29: the MG thin-wall probe — if a coarse cell straddling a
    1-tile wall is the leak, thicker walls should shrink it)."""
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    lvl = replace(lvl, entities=[e for e in lvl.entities
                                 if e.class_name not in ("vent", "duct")])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    for field, value in overrides.items():
        setattr(sim.physics_runner.eos, field, value)

    r0, r1, c0, c1 = AQ_BOX
    box_in = np.s_[r0 + wall_thick:r1 + 1 - wall_thick,
                   c0 + wall_thick:c1 + 1 - wall_thick]
    # Seal one layer per call, INNERMOST first: seal_tiles evacuates each
    # tile's gas to an OPEN non-span neighbour and refuses a tile with none
    # (its sealed-pocket guard) — an inner layer's corners only have open
    # neighbours while the layer outside them is still open.
    for k in reversed(range(wall_thick)):
        layer = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if min(r - r0, r1 - r, c - c0, c1 - c) == k]
        g.seal_tiles(layer, materials.MAT_GLASS)
    open0 = ~g.solid.copy()
    T0 = g.temperature.astype(np.int64)
    # P-G5: the box's OWN sealed glass ring (footprint minus its accountable
    # interior), for a BOX-SCOPED reading of the thermostat's contribution —
    # the GLOBAL `e_thermostat_sum` (below) is dominated by the crate fire's
    # own immediate walls, nothing to do with this box 20+ tiles away, so it
    # cannot be the (ii) subtraction term; this ring mask is.
    box_footprint = np.zeros_like(g.solid)
    box_footprint[r0:r1 + 1, c0:c1 + 1] = True
    box_in_mask = np.zeros_like(g.solid)
    box_in_mask[box_in] = True
    box_wall_mask = box_footprint & (~box_in_mask) & g.thermal_solid

    # ======================================================================
    # arc #54 SB gate instrumentation (design §6 "SB").
    #
    # (i) THE CLOSURE IDENTITY, exact in int64, measured ACROSS WHOLE TICKS.
    #     P-G1a could only bracket `run_substeps`, because the writers outside
    #     the EOS still wrote `temperature` and the solver's entry re-sync
    #     absorbed them. P-G1b lands every one of those writers on the seam and
    #     DELETES the re-sync (D1 live), so the honest bracket is now the whole
    #     `Simulation.step`: the field's per-tick drift has to equal the sum of
    #     the four counter groups that are allowed to move it —
    #         EOS       (design §2.8's seven terms, reset per step)
    #         tail      the thermal solver's gas side (accumulating)
    #         combustion the two-hop energy ledger (accumulating)
    #         seam      every Python writer's net (GameMap.gas_energy_books)
    #     Restricting the identity to the box would need per-region counters;
    #     the GLOBAL identity being exact is strictly stronger for the flux
    #     term, since the flux contributes exactly 0 to it only by per-face
    #     cancellation -- and the box's own sealed guarantee then follows from
    #     telescoping. The box's Delta E is reported alongside.
    #
    # (ii) DeltaT_box = Delta(Sum E / Sum N) -- N-WEIGHTED (an unweighted mirror
    #     mean is not conserved by mixing). THIS is the arc's headline.
    #
    # (iii) the D4 wall probe, the ts-wall probe, and the Sum N|u|^2 drift (D7).
    # ======================================================================
    eos = sim.physics_runner.eos
    engine = sim.physics_runner.engine
    tsolver = engine.temperature
    comb = sim.physics_runner.combustion
    t_amb_raw = g._gas_energy_t_amb_raw()
    ident = {"worst": 0, "ticks": 0, "bad": 0}

    def _n_plane():
        n = np.zeros(g.temperature.shape, dtype=np.int64)
        for gi in np.flatnonzero(g.gases.conservative):
            n += g.gas[gi].astype(np.int64)
        return n

    def _e_sum(mask):
        """Sum gas_energy over `mask` as a PYTHON int -- design §2.2 forbids
        absolute int64 sums (a 160^2 map at ambient sits near 2^55, but the
        bench must not be the thing that wraps at a blast core)."""
        return int(g.gas_energy[mask].astype(object).sum())

    def _ident_terms():
        """The four counter groups §2.8's identity is allowed to move the field
        with, PLUS (P-G5) the solid side's own three. The EOS and water-evac
        groups RESET every step (so they are read absolutely); the rest
        ACCUMULATE (so they are differenced tick to tick)."""
        return (
            # EOS (design §2.8). `e_entry_resync_sum` is RETIRED at P-G1b and
            # structurally 0; it is kept in the sum so this transcription still
            # names all seven terms.
            int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
            - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
            + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
            + int(eos.e_rail_sum),
            # thermal solver, gas side (temperature_solver.h)
            int(tsolver.e_gas_deposit_sum) + int(tsolver.e_gas_cond_sum)
            + int(tsolver.e_gas_rail_sum),
            # combustion's two-hop ledger, identity (A) (combustion.h)
            -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
            + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
            # every Python seam (GameMap.gas_energy_books, diagnostics excluded)
            int(g.gas_energy_seam_net()),
            # the water-displacement evacuation's export (design 2.7 row 2,
            # R3-#10): `step_water_tail` runs on the host BEFORE the EOS and
            # moves gas_energy with the bulk shares it pushes out of a flooding
            # cell. The move is conservative INSIDE the accountable set, so the
            # only term is what left it. Reset per call, like the EOS group.
            -int(engine.e_water_evac_export_sum),
            # P-G5 (design gas_energy_thermostat_ledger_2026-08-30.md): the
            # SOLID side's own channels — Pass 1's landing on thermal solids,
            # Pass 2's landing on thermal solids, the thermostat (Pass 3
            # relax-to-ambient), AND combustion's own `e_comb_solid_heat_sum`
            # (the object-site fuel deposit that bypasses TemperatureSolver's
            # Pass 1 entirely — combustion.cpp writes `temperature[s]`
            # directly). Accumulating, like the tail/combustion/seam groups
            # above.
            int(tsolver.e_solid_deposit_sum) + int(tsolver.e_solid_cond_sum)
            + int(tsolver.e_thermostat_sum) + int(comb.e_comb_solid_heat_sum),
        )

    def _solid_books():
        """(P-G5) Σ thermal_mass_raw·T_raw over thermal_solid cells — the
        SOLID side's own books, a SNAPSHOT refreshed by every step()."""
        return int(tsolver.solid_energy_books_sum)

    def _box_wall_solid_sum():
        """(P-G5) Σ thermal_mass_raw·T_raw over the box's OWN sealed glass
        ring only — the same currency `solid_energy_books_sum` uses
        (cap_real = thermal_mass << FP_SHIFT for a thermal solid, T already
        ambient-relative), computed directly here since there is no per-
        region C++ counter (and none is needed for a diagnostic print)."""
        shift = np.minimum(
            g.heat_inv_shift[box_wall_mask].astype(np.int64), 30)
        cap_real = np.int64(1) << (shift + 16)
        t = g.temperature[box_wall_mask].astype(np.int64)
        return int((cap_real * t).astype(object).sum())

    def _box_ET():
        """(Sum_box E, Sum_box N) over the box's ACCOUNTABLE cells."""
        acct = g._gas_energy_accountable()
        m = np.zeros_like(acct)
        m[box_in] = acct[box_in]
        n = _n_plane()
        return _e_sum(m), int(n[m].astype(object).sum())

    def _ke_sum():
        n = _n_plane()
        u2 = (g.wind_x.astype(np.int64) ** 2 + g.wind_y.astype(np.int64) ** 2)
        return int((n * (u2 >> 32)).astype(object).sum())

    # P-G1b: NO re-derive here any more. `seal_tiles` is an energy writer now
    # (design §2.7: the evacuated mass is MOVED, carrying the sealed tile's own
    # T_abs to each receiver, and the sub-count remainder retires), so the
    # field is already correct -- and re-deriving it from the mirror would
    # DESTROY exactly the remainders the seam just booked.
    if no_conduction:
        # Every conduction face becomes the NO_FACE sentinel, so the thermal
        # solver's §2 conduction pass is a structural no-op. The dial is read
        # off the live solver so this can never drift from config.
        g.face_shift[:] = sim.physics_runner.engine.temperature.no_face
    E0_box, N0_box = _box_ET()
    KE0 = _ke_sum()
    box_wall_sum0 = _box_wall_solid_sum()   # P-G5: box-scoped solid books, t=0
    P0_aq = float(g.atmosphere[box_in][open0[box_in]].mean()) / 65536.0
    # MASS vs PRESSURE-FIELD (2026-08-29, Erik's question): N inside the box
    # from the two conservative bulk planes — if P rises while N holds, the
    # pressure SOLVE is contaminated (no mass moved); if N rises, mass
    # actually crossed the sealed faces.
    o2 = int(g.gases.name_to_id["o2"])
    n2 = int(g.gases.name_to_id["inert_n2"])

    def n_box():
        return int(g.gas[o2][box_in].sum(dtype=np.int64) +
                   g.gas[n2][box_in].sum(dtype=np.int64))
    N0 = n_box()

    def dT(sl):
        d = (g.temperature.astype(np.int64) - T0)[sl]
        return float(d[open0[sl]].mean()) / 65536.0

    # (i) the ACROSS-TICK closure identity: bracket the WHOLE tick. P-G5 adds
    # a SECOND identity alongside it, over gas books + solid books together
    # (`ident_total`) — the same bracket, the same per-tick loop, extended
    # with the solid side's own three channels (design gas_energy_thermostat_
    # ledger_2026-08-30.md).
    acct0 = g._gas_energy_accountable()
    prev_e = _e_sum(acct0)
    prev_solid = _solid_books()
    prev_terms = _ident_terms()
    ident_total = {"worst": 0, "ticks": 0, "bad": 0}
    for t in range(1, END_TICK + 1):
        if t == IGNITE_TICK and ignite:
            ignite_ring(g, sim.edit_queue, *CRATE, 2.5, 1.0)
        sim.set_paused(False)
        sim.step()
        acct = g._gas_energy_accountable()
        e_now = _e_sum(acct)
        solid_now = _solid_books()
        terms = _ident_terms()
        expected = (terms[0]                      # EOS: absolute (reset/step)
                    + (terms[1] - prev_terms[1])  # tail: accumulating
                    + (terms[2] - prev_terms[2])  # combustion: accumulating
                    + (terms[3] - prev_terms[3])  # seams: accumulating
                    + terms[4])                   # water tail: reset per call
        resid = (e_now - prev_e) - expected
        ident["ticks"] += 1
        if resid:
            ident["bad"] += 1
            ident["worst"] = max(ident["worst"], abs(resid))
        # P-G5: the TOTAL ledger — gas books + solid books — against the same
        # `expected` PLUS the solid side's own accumulating group (terms[5]).
        expected_total = expected + (terms[5] - prev_terms[5])
        resid_total = ((e_now + solid_now) - (prev_e + prev_solid)) - expected_total
        ident_total["ticks"] += 1
        if resid_total:
            ident_total["bad"] += 1
            ident_total["worst"] = max(ident_total["worst"], abs(resid_total))
        prev_e, prev_solid, prev_terms = e_now, solid_now, terms

    P_aq = float(g.atmosphere[box_in][open0[box_in]].mean()) / 65536.0
    # 2026-08-29 LESSON: seal_tiles pushes the ring's gas INTO the box, so the
    # box starts at N ~ 1.29 atm-equivalent while P still reads 1.000 (the
    # solve catches up over ~2 s). The honest sealed-pocket invariant is
    # P == N x T_abs/T_amb, so report P/N — 1.000 means the solve is right.
    n_mean = float((g.gas[o2] + g.gas[n2])[box_in].mean()) / 65536.0
    u = np.sqrt((g.wind_x / 65536.0) ** 2 + (g.wind_y / 65536.0) ** 2)
    print(f"{name:>11}: box dT={dT(box_in):+7.1f}  box P {P0_aq:.3f}->{P_aq:.3f}"
          f"  box N x{n_box()/N0:5.3f} P/N={P_aq/n_mean:5.3f}"
          f"  bunker dT={dT(BUNKER):+7.1f}  pen dT={dT(PEN):+7.1f}"
          f"  arena dT={dT(ARENA):+6.1f}  u_max={float(u.max()):5.1f}"
          f"  wall={wall_thick}"
          + (f"  [{overrides}]" if overrides else ""))

    # ---- arc #54 SB gate report (design §6 "SB") --------------------------
    E1_box, N1_box = _box_ET()
    # (ii) the HEADLINE: N-weighted mean absolute T over the box, in game-deg.
    q = 65536.0
    t0 = (E0_box / N0_box / q - t_amb_raw / q) if N0_box else 0.0
    t1 = (E1_box / N1_box / q - t_amb_raw / q) if N1_box else 0.0
    ok_i = (ident["bad"] == 0)
    ok_i_total = (ident_total["bad"] == 0)
    # P-G5 (Erik's ruling 2026-08-30): the box's OWN sealed glass ring warming
    # over the run, in box-deg equivalent — the sealed box's remaining game-deg
    # above the old +/-2 tolerance is the seal event's cold boundary shell
    # being warmed back up by ambient-held walls, i.e. the thermostat doing
    # its job (directly, and via the arena's own walls staying pinned near
    # ambient and feeding this box's ring through conduction), to be BOOKED
    # here rather than chased as a leak. BOX-SCOPED, not the GLOBAL
    # `e_thermostat_sum` printed below — that global sum is dominated by the
    # crate fire's own immediate walls (a different, much bigger effect with
    # nothing to do with this box 20+ tiles away), so it is not the right
    # subtraction term for a box-local headline.
    box_wall_sum1 = _box_wall_solid_sum()
    thermostat_box_deg = ((box_wall_sum1 - box_wall_sum0) / N1_box / q
                          if N1_box else 0.0)
    dT_box_raw = t1 - t0
    dT_box_adj = dT_box_raw - thermostat_box_deg
    ok_ii = abs(dT_box_adj) <= 2.0
    ok_iii = float(u.max()) < 3.0
    detail = "" if ok_i else (f" ({ident['bad']} bad, worst |resid|="
                              f"{ident['worst']})")
    detail_total = "" if ok_i_total else (
        f" ({ident_total['bad']} bad, worst |resid|={ident_total['worst']})")
    print(f"{'':>11}  arc#54  (i) closure identity: "
          f"{'EXACT' if ok_i else 'BROKEN'} across {ident['ticks']} TICKS"
          f"{detail}"
          f"   (ii) dT_box-thermostat={dT_box_adj:+7.2f} "
          f"{'PASS' if ok_ii else 'FAIL'} (+/-2)"
          f"   (iii) u_max={float(u.max()):5.1f} "
          f"{'PASS' if ok_iii else 'FAIL'} (<3)")
    # `e_thermostat_sum` printed in box-deg equivalent too (as asked) — but
    # flagged GLOBAL: it is the whole-map total (dominated by the crate
    # fire's own nearby walls), not the box-scoped figure used above.
    global_thermostat_box_deg = (int(tsolver.e_thermostat_sum) / N1_box / q
                                 if N1_box else 0.0)
    print(f"{'':>11}  P-G5    (ii) raw dT_box=D(SumE/SumN)={dT_box_raw:+7.2f}  "
          f"box-wall thermostat contribution={thermostat_box_deg:+7.2f} box-deg"
          f"   e_thermostat_sum(GLOBAL)={int(tsolver.e_thermostat_sum)} "
          f"({global_thermostat_box_deg:+7.2f} box-deg eq., whole map)"
          f"   TOTAL ledger (gas+solid): "
          f"{'EXACT' if ok_i_total else 'BROKEN'} across "
          f"{ident_total['ticks']} TICKS{detail_total}")
    print(f"{'':>11}  probes  D4 wall Sum|p.u|={int(eos.e_wall_work_probe_sum)}"
          f"  ts-wall={int(eos.e_ts_work_sum)}"
          f"  Sum N|u|^2 drift={_ke_sum() - KE0}"
          f"  rail_shortfall={int(eos.e_energy_floor_sum)}"
          f"  hits rad_clip={int(eos.rad_clip_hits)}"
          f" p_floor={int(eos.p_face_floor_hits)}"
          f" flux_sat={int(eos.flux_sat_hits)}"
          f" t_max={int(eos.t_max_phys_hits)}")
    # arc #54 P-G1b: THE THERMAL SOLVER'S GAS-SIDE CHANNEL, named and counted.
    # `e_gas_cond_sum` is the whole of gate (ii)'s residual and the bench's
    # `nofire_nocond` control is its zero: conduction across gas<->THERMAL_SOLID
    # faces carries energy INTO the gas books, because the solids are held at
    # ambient by Pass 3's two-way relaxation while conduction diffuses the
    # UNWEIGHTED T and the books are N-WEIGHTED. In an acoustically ringing
    # cell T and N are positively correlated, so the unweighted mean sits below
    # the N-weighted one and the walls top the gas up forever. P-G1b BOOKS this
    # channel (that is what makes (i) exact across ticks); REMOVING it is a
    # physics decision outside this patch's scope -- see the P-G1b report.
    print(f"{'':>11}  gas-side  cond={int(tsolver.e_gas_cond_sum)}"
          f"  deposit={int(tsolver.e_gas_deposit_sum)}"
          f"  rail={int(tsolver.e_gas_rail_sum)}"
          f"  (cond in box-deg = "
          f"{int(tsolver.e_gas_cond_sum) / N1_box / q if N1_box else 0.0:+.2f})")


def main() -> None:
    print(f"sealed-box bisection — crate fire only, {END_TICK/TPS:.0f} s, "
          f"FIXED = box dT ~ 0")
    wanted = set(sys.argv[1:])          # optional: run only the named variants
    for spec in VARIANTS:
        name, overrides = spec[0], spec[1]
        if wanted and name not in wanted:
            continue
        run_variant(name, overrides, *spec[2:])


if __name__ == "__main__":
    main()
