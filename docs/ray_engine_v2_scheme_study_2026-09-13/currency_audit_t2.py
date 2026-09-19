"""T2 — the currency audit, MEASURED against the engine (issue #12, design v2 §6.5).

Answers: *is one gas heat count the same joule as one solid heat count?*

Not a test and not engine code — a measurement instrument, the sibling of this
folder's other `*_study.py` files. It CHANGES NOTHING: it runs the shipped
`Simulation` with the shipped dials, and the only thing it varies is the
`[physics.thermal] c_v` member on the live solver, which it restores.

What it measures, at `c_v = 1` (shipped) and `c_v = 0.0076849` (physical):

  M1  the two conduction ledgers across solid<->gas faces, converted to joules
      through report §1.2's bridge (`gas_energy` raw = (2^16 / c_v) * E_raw).
      If one gas joule is one solid joule, the gas's gain must equal the
      solid's loss. It also runs BOTH closure identities every tick (arc #54's
      gas-only one and P-G5's gas+solid total), so the report can say whether
      they NOTICE the gap. They do not.
  M2  the same thing per cell, through the DIRECT binding — where `gas_energy`
      is absent and the pre-#54 T-form law runs, which is the law whose
      prediction design v2 §6 item 5 asks for.
  M3  the `n_floor_heat` x `c_v` interaction, on Pass 1's deposit.
  M4  design v2 §6 item 5 in its most literal form on the LIVE engine: a known
      energy into `gmap.heat`, one `Simulation.step()`, read the gas books.
      The CONTROL for M1 — the deposit path is the one that is already right.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/currency_audit_t2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                      # noqa: E402
from level_loader import LevelData               # noqa: E402
from simulation import Simulation                # noqa: E402

FP_ONE = 1 << 16

AIR, HULL, STEEL = 0, 1, 4

# --- the two currencies, from the report ---------------------------------
RHO_C_UNIT = 0.9e6 / 8.0          # R13: one thermal_mass unit, J/(m^3 K)
RHO_C_AIR = 101325.0 / (0.4 * 293.0)   # p/((gamma-1)T), the engine's own constants
C_V_PHYS = RHO_C_AIR / RHO_C_UNIT      # 0.00768487
TILE_M = 0.333
V_TILE = TILE_M * TILE_M * 2.5
J_PER_COUNT = 0.9e6 * V_TILE / (8 * FP_ONE)


def build_room(interior=10, block=4, tile=TILE_M):
    """Sealed room, 1-tile HULL ring, a centred STEEL block (conductivity 45,
    thermal_mass 32) surrounded by air — so every block face is a live
    solid<->gas conduction face (air's conductivity is 0.024, NOT zero)."""
    h = w = interior + 2
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = HULL
    tm[:, 0] = tm[:, -1] = HULL
    b0 = (h - block) // 2
    tm[b0:b0 + block, b0:b0 + block] = STEEL
    return LevelData(name="t2_currency", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=float(tile),
                     diffuse_path=Path("."), boundary="space"), b0


def run(c_v, ticks=40, interior=10, block=4, t_hot_game=800.0):
    level, b0 = build_room(interior, block)
    sim = Simulation(level, seed=12345, breach_physics=bp, enable_recorder=False)
    runner = sim.physics_runner
    gmap = sim.gmap

    # THE ONE THING VARIED. `EOSSolver::c_v` is dead surface since arc #54 D5
    # (the drag deposit is the derived k_ke) but is bound from the same key,
    # so it is set identically to keep the two in step.
    runner.temperature.c_v = float(c_v)
    runner.eos.c_v = float(c_v)

    tsol = runner.engine.temperature
    eos = runner.eos
    comb = runner.combustion

    # Heat the block. `seed_gas_temperature` is the gas seam; a THERMAL SOLID
    # has no gas books, so writing its `temperature` directly is correct here
    # (the "gas temperature is a mirror" rule is about GAS cells).
    sl = (slice(b0, b0 + block), slice(b0, b0 + block))
    gmap.temperature[sl] = int(round(t_hot_game * FP_ONE))

    acct = gmap._gas_energy_accountable()

    def e_acct():
        return int(gmap.gas_energy[acct].astype(object).sum())

    def terms():
        return (
            int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
            - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
            + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
            + int(eos.e_rail_sum),
            int(tsol.e_gas_deposit_sum) + int(tsol.e_gas_cond_sum)
            + int(tsol.e_gas_rail_sum),
            -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
            + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
            int(gmap.gas_energy_seam_net()),
            -int(runner.engine.e_water_evac_export_sum),
        )

    def solid_terms():
        return (int(tsol.e_solid_deposit_sum) + int(tsol.e_solid_cond_sum)
                + int(tsol.e_thermostat_sum))

    prev_e, prev_t = e_acct(), terms()
    prev_total = prev_e + int(tsol.solid_energy_books_sum)
    prev_st = solid_terms()
    prev = dict(gas_cond=int(tsol.e_gas_cond_sum),
                solid_cond=int(tsol.e_solid_cond_sum),
                solid_books=int(tsol.solid_energy_books_sum),
                cond_cap=int(tsol.e_cond_cap_sum),
                trunc=int(tsol.e_cond_trunc_sum))

    out = dict(c_v=c_v, bad=0, worst=0, ticks=0,
               gas_cond=0, solid_cond=0, cond_cap=0, trunc=0,
               d_solid_books=0, d_gas_books=0,
               tot_bad=0, tot_worst=0,
               rail_hits0=int(tsol.t_max_phys_hits))
    # a gas cell on the block's face, watched for its own temperature history
    probe = (b0 - 1, b0 + block // 2)
    out["probe_T"] = [int(gmap.temperature[probe])]
    out["probe_E"] = [int(gmap.gas_energy[probe])]
    out["probe_N"] = [int(gmap._gas_bulk_n_raw()[probe])]

    for _ in range(ticks):
        sim.set_paused(False)
        sim.step()
        e_now, t_now = e_acct(), terms()
        expected = (t_now[0]                                   # EOS: reset
                    + sum(t_now[k] - prev_t[k] for k in (1, 2, 3))  # accumulating
                    + t_now[4])
        resid = (e_now - prev_e) - expected
        out["ticks"] += 1
        if resid:
            out["bad"] += 1
            out["worst"] = max(out["worst"], abs(resid))
        # P-G5's TOTAL ledger: gas books + solid books, in ONE number
        # (tests/test_thermostat_books.py:103 — the one place the engine adds
        # the two currencies together).
        st_now = solid_terms()
        total_now = e_now + int(tsol.solid_energy_books_sum)
        tresid = (total_now - prev_total) - (expected + (st_now - prev_st))
        # TICK 1 IS EXCLUDED, and not for a c_v reason: this scenario seeds the
        # block by writing `temperature` directly, outside every counter, and
        # `solid_energy_books_sum` is a SNAPSHOT refreshed only at the end of a
        # step — so the seed lands in the first snapshot with no term to match
        # it (measured: exactly 16 cells x 32<<16 x 800<<16 = 1.759e15, the
        # same at BOTH c_v values, which is how it was identified).
        if tresid and out["ticks"] > 1:
            out["tot_bad"] += 1
            out["tot_worst"] = max(out["tot_worst"], abs(tresid))
        prev_total, prev_st = total_now, st_now
        prev_e, prev_t = e_now, t_now

        out["gas_cond"] += int(tsol.e_gas_cond_sum) - prev["gas_cond"]
        out["solid_cond"] += int(tsol.e_solid_cond_sum) - prev["solid_cond"]
        out["cond_cap"] += int(tsol.e_cond_cap_sum) - prev["cond_cap"]
        out["trunc"] += int(tsol.e_cond_trunc_sum) - prev["trunc"]
        out["d_solid_books"] += int(tsol.solid_energy_books_sum) - prev["solid_books"]
        prev = dict(gas_cond=int(tsol.e_gas_cond_sum),
                    solid_cond=int(tsol.e_solid_cond_sum),
                    solid_books=int(tsol.solid_energy_books_sum),
                    cond_cap=int(tsol.e_cond_cap_sum),
                    trunc=int(tsol.e_cond_trunc_sum))
        out["probe_T"].append(int(gmap.temperature[probe]))
        out["probe_E"].append(int(gmap.gas_energy[probe]))
        out["probe_N"].append(int(gmap._gas_bulk_n_raw()[probe]))

    out["d_gas_books"] = e_acct() - out.get("_e0", e_acct())
    return out


def report(r):
    """M1 — the energy balance across every solid<->gas conduction face.

    Units (report §1.1/§1.2):
      e_solid_cond_sum / e_cond_trunc_sum are E_raw * 2^16  -> counts = /2^16
      e_gas_cond_sum   is a gas-books raw                   -> counts = *c_v/2^16

    AMENDED AT T5a — this measurement had a trap in it, and the trap only
    springs once T2's own recommendation lands.

    `delivered` used to read `e_gas_cond_sum / 2^16`, i.e. the SAME counter
    `received` reads. That was correct while the engine booked the face sum
    unconverted (pre-T5a, `e_gas_cond_sum == de`, a heat count), and the ratio
    it printed was a real cross-ledger measurement of the defect. After T5a the
    counter holds the CONVERTED books delta, so both lines come from one number
    and `received / delivered` is identically `c_v` BY CONSTRUCTION — a
    tautology that would go on printing "0.0076849" forever and read as though
    the fix had never landed.

    `delivered` is now taken from the SOLID side instead: the solids' own net
    energy change minus what their endpoint divide destroyed. Every solid<->
    solid face is internal to `e_solid_cond_sum` and cancels there, so on this
    scenario `lost - trunc` IS what crossed into gas. At c_v = 1 the two
    definitions agree to the last digit (6338756.886 either way), so no T2
    number moves; at the physical c_v they are a genuine comparison of two
    independently-counted ledgers, which is what M1 always claimed to be.
    """
    c_v = r["c_v"]
    lost = -r["solid_cond"] / FP_ONE               # what the solids gave up
    trunc = -r["trunc"] / FP_ONE                   # counted, destroyed by floordiv
    delivered = lost - trunc                       # what the faces handed the gas
    received = r["gas_cond"] * c_v / FP_ONE        # what the gas books recorded
    print(f"\n=== c_v = {c_v:.8f}  ({r['ticks']} ticks) ===")
    print(f"  e_solid_cond_sum {r['solid_cond']:>22d}   e_gas_cond_sum {r['gas_cond']:>22d}")
    print(f"  solids LOST              {lost:>16.3f} counts = {lost*J_PER_COUNT:>13.3f} J")
    print(f"    of which counted trunc {trunc:>16.3f} counts   ({trunc/lost*100 if lost else 0:.2f} % of the loss)")
    print(f"    delivered to the gas   {delivered:>16.3f} counts = {delivered*J_PER_COUNT:>13.3f} J")
    print(f"  gas RECEIVED             {received:>16.3f} counts = {received*J_PER_COUNT:>13.3f} J")
    if delivered:
        print(f"  >>> received / delivered = {received/delivered:.8f}"
              f"   (want 1.0; == c_v = {c_v:.8f})")
    print(f"  arc #54 closure identity: {r['bad']}/{r['ticks']} ticks bad, "
          f"worst |residual| {r['worst']}")
    print(f"  P-G5 TOTAL ledger (gas+solid): {r['tot_bad']}/{r['ticks']} bad, "
          f"worst |residual| {r['tot_worst']}")
    dT = (r["probe_T"][-1] - r["probe_T"][0]) / FP_ONE
    n = r["probe_N"][-1] / FP_ONE
    print(f"  probe gas cell: dT = {dT:+.6f} game-K over the run; N = {n:.4f}; "
          f"real capacity {n*c_v*FP_ONE:.2f} counts/K = {n*RHO_C_AIR*V_TILE:.2f} J/K")


# ---------------------------------------------------------------------------
# M2 — the per-cell form of design v2 §6 item 5, through the DIRECT binding.
#
# `TemperatureSolver.step`'s pybind signature takes no `gas_energy`, so the
# direct path is ALWAYS the pre-#54 T-form law: dT_i = floordiv(de_i, cap_i).
# That law is CORRECT at any c_v, and this leg measures it, so the engine's
# live (gas_energy-supplied) answer has something honest to be compared with.
# ---------------------------------------------------------------------------
_FACE_DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))


def m2_tform(c_v, t_hot_game=800.0, ticks=1):
    from simulation.materials import MaterialTable
    tbl = MaterialTable.from_config()
    no_face = int(tbl.no_face)
    # a 1x3 strip: STEEL | AIR | AIR  (the middle air cell has ONE live face)
    m = np.array([[STEEL, AIR, AIR]], dtype=np.int8)
    h, w = m.shape
    shift = np.ascontiguousarray(tbl.heat_inv_shift[m].astype(np.int32))
    solid = np.ascontiguousarray(tbl.permeability[m] <= 0.0)
    tsol = np.ascontiguousarray(tbl.thermal_solid[m].astype(bool))
    face = np.full((h, w, 4), no_face, dtype=np.int32)
    ft = tbl.face_shift_table
    for d, (dy, dx) in enumerate(_FACE_DIRS):
        for y in range(h):
            for x in range(w):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    face[y, x, d] = ft[m[y, x], m[ny, nx]]
    face = np.ascontiguousarray(face)

    s = bp.TemperatureSolver()
    s.no_face = no_face
    s.cool_shift = s.cool_shift_vacuum = 31      # cooling off: conduction alone
    s.c_v = float(c_v)
    s.n_floor_heat = 0.01

    temp = np.zeros((h, w), dtype=np.int32)
    temp[0, 0] = int(round(t_hot_game * FP_ONE))
    heat = np.zeros((h, w), dtype=np.int32)
    is_vac = np.zeros((h, w), dtype=bool)
    atm = np.full((h, w), FP_ONE, dtype=np.int32)
    nb = np.full((h, w), FP_ONE, dtype=np.int32)
    before = temp.copy()
    for _ in range(ticks):
        s.step(temp, heat, shift, face, solid, is_vac, atm,
               None, None, 0.0, nb, tsol, None, None, None, None, False)
    dT_gas = (int(temp[0, 1]) - int(before[0, 1])) / FP_ONE
    dT_sol = (int(temp[0, 0]) - int(before[0, 0])) / FP_ONE
    cap_gas = (FP_ONE * int(round(c_v * FP_ONE))) >> 16
    return dT_gas, dT_sol, cap_gas


def m4_live_deposit(c_v, deposit_raw=1 << 16, interior=8):
    """M4 — design v2 §6 item 5 in its most literal form, on the LIVE engine.

    Put a known energy into `gmap.heat` on an accountable gas cell right before
    `Simulation.step()` (`heat` is a per-tick buffer wiped at END of tick —
    `simulation.py:1596` — so a pre-step write lands in Pass 1), then read what
    the gas books actually recorded.

    Prediction, from the cell's REAL heat capacity `N*c_v`:
        dT       = deposit / (N * c_v)                    [raw]
        dE_books = N_raw * dT = 2^16 * deposit / c_v      [raw]
    """
    level, _ = build_room(interior, block=0)
    sim = Simulation(level, seed=7, breach_physics=bp, enable_recorder=False)
    runner = sim.physics_runner
    runner.temperature.c_v = float(c_v)
    runner.eos.c_v = float(c_v)
    g = sim.gmap
    tsol = runner.engine.temperature
    cell = (interior // 2, interior // 2)
    assert g.gas_is_accountable(*cell)
    n_raw = int(g._gas_bulk_n_raw()[cell])
    before = int(tsol.e_gas_deposit_sum)
    g.heat[cell] = int(deposit_raw)
    sim.set_paused(False)
    sim.step()
    got = int(tsol.e_gas_deposit_sum) - before
    predicted = (FP_ONE * deposit_raw) / c_v          # exact, pre-quantization
    return n_raw, got, predicted


def m3_floor(c_v, n_real, deposit_counts, t_max_phys=16000.0):
    """M3 — the `n_floor_heat` interaction, through the direct binding.

    Pass 1's gas deposit is `dT = E_abs / (max(N, n_floor_heat) * c_v)`. The
    floor is on **N**, not on the capacity, so rescaling `c_v` does not change
    what the floor MEANS — but it does multiply the dT a floored (near-vacuum)
    cell receives by `1/c_v`. Returns (dT, rail_hits).
    """
    from simulation.materials import MaterialTable
    tbl = MaterialTable.from_config()
    no_face = int(tbl.no_face)
    m = np.array([[AIR, AIR, AIR]], dtype=np.int8)
    h, w = m.shape
    shift = np.ascontiguousarray(tbl.heat_inv_shift[m].astype(np.int32))
    solid = np.ascontiguousarray(tbl.permeability[m] <= 0.0)
    tsol = np.ascontiguousarray(tbl.thermal_solid[m].astype(bool))
    face = np.ascontiguousarray(np.full((h, w, 4), no_face, dtype=np.int32))

    s = bp.TemperatureSolver()
    s.no_face = no_face
    s.cool_shift = s.cool_shift_vacuum = 31
    s.c_v = float(c_v)
    s.n_floor_heat = 0.01
    s.T_MAX_PHYS = float(t_max_phys)

    temp = np.zeros((h, w), dtype=np.int32)
    heat = np.zeros((h, w), dtype=np.int32)
    heat[0, 1] = int(deposit_counts)
    is_vac = np.zeros((h, w), dtype=bool)
    atm = np.full((h, w), int(round(n_real * FP_ONE)), dtype=np.int32)
    nb = np.ascontiguousarray(atm.copy())
    before = int(s.t_max_phys_hits)
    s.step(temp, heat, shift, face, solid, is_vac, atm,
           None, None, 0.0, nb, tsol, None, None, None, None, False)
    return int(temp[0, 1]) / FP_ONE, int(s.t_max_phys_hits) - before


if __name__ == "__main__":
    print(f"J_per_count (R13 0.9 pin) = {J_PER_COUNT:.8f} J")
    print(f"rho*c_v air = {RHO_C_AIR:.4f} J/(m3 K);  c_v_phys = {C_V_PHYS:.8f};"
          f"  1/c_v = {1/C_V_PHYS:.4f}")
    for r in (run(1.0), run(C_V_PHYS)):
        report(r)

    print("\n=== M2: the T-form law (direct binding, gas_energy absent) ===")
    print("    STEEL@800 | AIR | AIR, one tick, cooling off")
    base = None
    for c_v in (1.0, C_V_PHYS):
        dg, ds, cap = m2_tform(c_v)
        if base is None:
            base = dg
        print(f"  c_v = {c_v:.8f}: cap_gas = {cap:>8d} raw ;"
              f"  gas dT = {dg:+.8f} K ; solid dT = {ds:+.8f} K"
              f" ; gas dT / (c_v=1 case) = {dg/base if base else float('nan'):.6f}")
    print("  (the CORRECT law's gas dT is c_v-INDEPENDENT: 130x less energy")
    print("   crosses the face, into a 130x lighter cell. The engine's live")
    print("   path scales it BY c_v instead - see M1.)")

    print("\n=== M3: n_floor_heat x c_v (direct binding, Pass 1 deposit) ===")
    print("    one tick, deposit = 1.0 count (65536 raw), n_floor_heat = 0.01")
    for n_real in (1.0, 0.05, 0.01, 0.002):
        row = []
        for c_v in (1.0, C_V_PHYS):
            dT, hits = m3_floor(c_v, n_real, FP_ONE)
            row.append((dT, hits))
        ratio = row[1][0] / row[0][0] if row[0][0] else float("nan")
        print(f"  N = {n_real:<6}: dT(c_v=1) = {row[0][0]:>12.6f} K "
              f"(rail {row[0][1]}) ;  dT(phys) = {row[1][0]:>12.6f} K "
              f"(rail {row[1][1]}) ;  x{ratio:.2f}")
    print("  (the floor is on N, not on the capacity, so n_floor_heat itself")
    print("   does NOT rescale - but a floored cell's dT does, by 1/c_v.)")

    print("\n=== M4: a known energy into a gas cell, on the LIVE engine ===")
    print("    deposit = 65536 raw into gmap.heat, one Simulation.step()")
    for c_v in (1.0, C_V_PHYS):
        n_raw, got, pred = m4_live_deposit(c_v)
        print(f"  c_v = {c_v:.8f}: N_raw = {n_raw} ; d(e_gas_deposit_sum) = "
              f"{got:>18d} ; predicted from N*c_v = {pred:>20.1f} ; "
              f"got/pred = {got/pred:.8f}")
    print("  (Pass 1 already carries 1/c_v correctly - this leg is the")
    print("   CONTROL for M1, the leg that does not.)")
