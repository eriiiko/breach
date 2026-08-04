"""batch_rails.py — does interior air damping stay numerically clean?
Reports the EOS rail counters alongside the storming metric. ANALYSIS ONLY.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import storm_probe as sp  # noqa: E402


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fire"
    geom = sys.argv[2] if len(sys.argv) > 2 else "tworoom"
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
    damps = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)

    print(f"\n=== rails vs air damping | mode={mode} geom={geom} {secs}s ===")
    hdr = (f"  {'damp':>6s} {'ke_tail':>10s} {'ke_final':>10s} {'umax_pk':>8s} "
           f"{'umax_fin':>9s} {'n_sub':>6s} {'uclamp':>7s} {'work':>7s} "
           f"{'Tfloor':>8s} {'Tmax':>6s}")
    print(hdr)
    for d in damps:
        kw = dict(mode=mode, geom=geom, interior=12, tile=0.5, seconds=secs,
                  damp=d)
        if mode == "fire":
            kw["dials"] = dict(sp.PF1B)
        if mode == "pulse":
            kw.update(dT=285.0, amp=45.0, period=10.0)
        res = sp.run(**kw)
        s = sp.analyse(res)
        c = res["counters"]
        print(f"  {d:6.3f} {s['ke_last_decile_mean']:10.1f} "
              f"{s['ke_final']:10.1f} {s['umax_peak']:8.2f} "
              f"{s['umax_final']:9.2f} {c.get('dbg_last_n_sub',-1):6d} "
              f"{c.get('u_clamp_hits',-1):7d} {c.get('work_clamp_hits',-1):7d} "
              f"{c.get('energy_floor_hits',-1):8d} "
              f"{c.get('t_max_phys_hits',-1):6d}")
    print("\ndone.")


if __name__ == "__main__":
    main()
