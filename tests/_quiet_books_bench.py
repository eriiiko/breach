"""arc #54 P-G1b — THE QUIET GATE (design §6, the P-G1b row).

*A 60 s no-forcing playground run: books drift == Σ counted channels, exactly.*

This is the gate that proves D1 is really live. Every other gate in the arc
brackets ONE subsystem — the SB bench measures a sealed box, VENT measures a
breach, FIRE measures a burn. This one just lets the whole engine run, on the
real level, with every system awake and nothing driving it, and asks the only
question that matters once `gas_energy` is the cross-tick truth:

    did anything change the field without saying so?

The identity, exact in int64, EVERY tick:

    Δ Σ_accountable gas_energy  ==
          [ e_entry_resync_sum + e_transport_net_sum − e_wipe_sum
            − e_kick_ke_sum + e_drag_heat_sum − e_work_export_sum
            + e_rail_sum ]                                        (EOS §2.8)
        + [ e_gas_deposit_sum + e_gas_cond_sum + e_gas_rail_sum ]
                                                    (thermal solver, gas side)
        + [ −e_comb_draw_sum + e_comb_deliver_sum + e_comb_heat_sum
            + e_comb_rail_sum ]                            (combustion §2.7)
        + gas_energy_seam_net()                (every Python seam's net)

The EOS group RESETS every step, so it is read absolutely; the other three
accumulate, so they are differenced tick to tick.

WHY THE PLAYGROUND, AND WHY "QUIET". The playground carries doors, vents and
ducts, so the pump primitives, the vent plenum's relative↔absolute conversion
and the door seal/unseal seams all fire on their own during an idle run — the
seams that a hand-built synthetic would never touch. "Quiet" means no fire, no
blast, no breach: any drift this bench sees is the engine's own resting
behaviour, which is precisely the class #54 was opened for (a sealed box that
heated +121 with nobody in it).

HARNESS, not a pytest gate (`_` prefix): a 60 s run is minutes of wall clock.
The same identity is gated per-tick, on a shorter scenario, by
tests/test_e1_hot_rail.py::test_no_transport_mint.

Run:
    conda run -n data python tests/_quiet_books_bench.py [seconds]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402

TPS = 24
Q = 65536.0


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    ticks = int(seconds * TPS)

    sim = Simulation(load_level("playground", levels_dir=str(ROOT / "levels")),
                     seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    eos = sim.physics_runner.eos
    tsolver = sim.physics_runner.engine.temperature
    comb = sim.physics_runner.combustion

    def e_acct():
        """Σ_accountable gas_energy as a PYTHON int — design §2.2 forbids an
        absolute int64 sum, and the gate must not be the thing that wraps."""
        return int(g.gas_energy[g._gas_energy_accountable()]
                   .astype(object).sum())

    def terms():
        return (
            int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
            - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
            + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
            + int(eos.e_rail_sum),
            int(tsolver.e_gas_deposit_sum) + int(tsolver.e_gas_cond_sum)
            + int(tsolver.e_gas_rail_sum),
            -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
            + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
            int(g.gas_energy_seam_net()),
        )

    e0 = e_acct()
    prev_e, prev_t = e0, terms()
    bad = worst = worst_tick = 0
    # Per-group running totals, so a failure names the group as well as the
    # tick — the four groups are the four places a new writer can appear.
    total = [0, 0, 0, 0]
    for t in range(1, ticks + 1):
        sim.set_paused(False)
        sim.step()
        e_now, cur = e_acct(), terms()
        d = (cur[0],
             cur[1] - prev_t[1], cur[2] - prev_t[2], cur[3] - prev_t[3])
        for i in range(4):
            total[i] += d[i]
        resid = (e_now - prev_e) - sum(d)
        if resid:
            bad += 1
            if abs(resid) > abs(worst):
                worst, worst_tick = resid, t
        prev_e, prev_t = e_now, cur

    e1 = e_acct()
    n_acct = int(g._gas_bulk_n_raw()[g._gas_energy_accountable()]
                 .astype(object).sum())
    print(f"QUIET gate — playground, {seconds:.0f} s ({ticks} ticks), "
          f"no fire / no blast / no breach")
    print(f"  identity: {'EXACT' if bad == 0 else 'BROKEN'} over {ticks} ticks"
          + ("" if bad == 0 else
             f" ({bad} bad, worst {worst} @ tick {worst_tick})"))
    print(f"  books drift  d(Sum_accountable gas_energy) = {e1 - e0}")
    print(f"  == counted:  EOS={total[0]}  tail={total[1]}  "
          f"combustion={total[2]}  seams={total[3]}")
    print(f"     sum      = {sum(total)}   "
          f"residual = {(e1 - e0) - sum(total)}")
    if n_acct:
        print(f"  in game-deg over the accountable set: "
              f"{(e1 - e0) / n_acct / Q:+.4f}")
    print(f"  seam channels: {g.gas_energy_books}")


if __name__ == "__main__":
    main()
