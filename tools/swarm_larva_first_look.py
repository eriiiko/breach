"""Erik's first look at the larva law (arc #63 P3, docs/swarm_P3_impl.md §12)
-> docs/swarm_P3_first_look.png.

The REAL sim path (CPU build, ``Simulation.step``): a 24 x 36 room at 0.333 m
tiles held LAWFULLY (FieldEdits through ``sim.edit``, never a bare
temperature write) at a warm strip along the west wall (40 C), a lethal
corner (70 C) and a cold patch (5 C); 200 larvae spawned after a 10 s
warm-up, run for 120 s, against the ablated law (turn_bias 0.5, alarm rate =
base rate, set through CFG.swarm before construction -- the species table is
the only door).

Exits non-zero, printing the numbers, never plotting silently, when:
1. the bulk's warm-up mean is outside [T_coma_exit + 3 K, T_prefer];
2. 20 % or more of the bulk's cells read below T_ctmin over the warm-up
   (§12's stop-and-ask rule: never retune around it);
3. any kernel decision disagrees with the numpy RAW reader every tick
   (a death on a tile not above T_lethal_hot, RUN -> COMA not below T_ctmin,
   COMA -> RUN not above T_coma_exit, COMA held above T_coma_exit).

    C:/Users/steen/anaconda3/envs/data/python.exe tools/swarm_larva_first_look.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

SEED = 20260924
H, W, TILE = 24, 36, 0.333
WARM, RUN_S = 240, 120
N_LARVAE = 200


def _level():
    from level_loader import LevelData
    tm = np.ones((H, W), dtype=np.int32)
    tm[1:H - 1, 1:W - 1] = 4
    return LevelData(name="larva_first_look", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=TILE, diffuse_path=Path("."))


def _edits(ts):
    from simulation.field_edit import EditMode, FieldEdit, Region, T_MAX_PHYS
    return [
        FieldEdit(field="wave_source", region=Region.RECT, coords=(5, 1, 22, 2),
                  amount=ts.from_kelvin(313.15), mode=EditMode.MAX),
        FieldEdit(field="wave_source", region=Region.RECT, coords=(1, 1, 4, 4),
                  amount=ts.from_kelvin(343.15), mode=EditMode.MAX),
        FieldEdit(field="wave_source", region=Region.RECT, coords=(21, 33, 22, 34),
                  amount=1000.0, mode=EditMode.REMOVE,
                  clamp=(ts.from_kelvin(278.15), T_MAX_PHYS)),
    ]


def run(ablated: bool):
    import breach_physics as bp
    import temperature_scale
    from config import CFG
    from simulation import swarm
    from simulation.events import SwarmUnitDiedEvent
    from simulation.simulation import Simulation
    saved = CFG.swarm
    if ablated:
        from config import Namespace
        row = dict(vars(CFG.swarm.larva))
        row["turn_bias"] = 0.5
        row["tumble_rate_alarm"] = row["tumble_rate_base"]
        CFG.swarm = Namespace({"larva": row})
    try:
        sim = Simulation(_level(), seed=SEED, breach_physics=bp, enable_recorder=False)
    finally:
        CFG.swarm = saved
    g = sim.gmap
    ts = temperature_scale.load()
    row = g.swarm_species.row("larva")
    edits = _edits(ts)
    bulk = (slice(1, H - 1), slice(8, 28))
    warm_means, warm_cold = [], []

    def step():
        for e in edits:
            sim.edit(e)
        sim.set_paused(False)
        sim.step()

    for _ in range(WARM):
        step()
        B = g.temperature[bulk].astype(np.int64)
        warm_means.append(float(np.mean(ts.to_kelvin(B / 65536.0))) - 273.15)
        warm_cold.append(float(np.mean(B < row.T_ctmin_q)))

    rng = np.random.default_rng(SEED)
    open_t = [(y, x) for y in range(1, H - 1) for x in range(5, W - 1)
              if not g.solid[y, x] and not (20 <= y <= 22 and 32 <= x <= 34)]
    pick = rng.choice(len(open_t), size=N_LARVAE, replace=True)
    xs = [(open_t[i][1] << 16) + int(rng.integers(0, 65536)) for i in pick]
    ys = [(open_t[i][0] << 16) + int(rng.integers(0, 65536)) for i in pick]
    hs = [int(rng.integers(-205886, 205888)) for _ in pick]
    sim.swarm.spawn("larva", xs, ys, hs)

    n_ticks = int(RUN_S * sim._tps)
    pos = np.zeros((n_ticks + 1, N_LARVAE, 2), dtype=np.int64)
    state = np.zeros((n_ticks + 1, N_LARVAE), dtype=np.int32)
    felt_acc = np.zeros((H, W))
    felt_c = np.full((n_ticks + 1, N_LARVAE), np.nan)
    deaths, comas = [], []
    mismatches = []
    col = lambda c: swarm.swarm_column(g, "larva", c)  # noqa: E731
    pos[0, :, 0], pos[0, :, 1] = col("pos_x")[:N_LARVAE], col("pos_y")[:N_LARVAE]
    state[0] = col("ai_state")[:N_LARVAE]
    for t in range(1, n_ticks + 1):
        prev_state = col("ai_state")[:N_LARVAE].copy()
        prev_valid = col("valid")[:N_LARVAE].copy()
        step()
        T = g.temperature
        felt_acc += ts.to_kelvin(T / 65536.0) - 273.15
        px, py = col("pos_x")[:N_LARVAE].copy(), col("pos_y")[:N_LARVAE].copy()
        st = col("ai_state")[:N_LARVAE].copy()
        v = col("valid")[:N_LARVAE] == 1
        pos[t, :, 0], pos[t, :, 1] = px, py
        state[t] = np.where(v, st, swarm.AI_DEAD)
        for ev in sim.tick_events:
            if isinstance(ev, SwarmUnitDiedEvent):
                deaths.append((t, ev.pos_x_q, ev.pos_y_q))
                if int(T[ev.pos_y_q >> 16, ev.pos_x_q >> 16]) <= row.T_lethal_hot_q:
                    mismatches.append(f"tick {t}: death on a tile not above T_lethal_hot")
        felt = T[py >> 16, px >> 16]
        felt_c[t] = ts.to_kelvin(felt / 65536.0) - 273.15
        live = v & (prev_valid == 1)
        rc = live & (prev_state == swarm.AI_RUN) & (st == swarm.AI_COMA)
        cr = live & (prev_state == swarm.AI_COMA) & (st == swarm.AI_RUN)
        cc = live & (prev_state == swarm.AI_COMA) & (st == swarm.AI_COMA)
        if (felt[rc] >= row.T_ctmin_q).any():
            mismatches.append(f"tick {t}: RUN -> COMA not below T_ctmin")
        if (felt[cr] <= row.T_coma_exit_q).any():
            mismatches.append(f"tick {t}: COMA -> RUN not above T_coma_exit")
        if (felt[cc] > row.T_coma_exit_q).any():
            mismatches.append(f"tick {t}: COMA held above T_coma_exit")
        for i in np.flatnonzero(rc):
            comas.append((t, int(px[i]), int(py[i])))
    return dict(warm_mean=float(np.mean(warm_means)), warm_cold=float(np.mean(warm_cold)),
                pos=pos, state=state, felt_mean=felt_acc / n_ticks, felt_c=felt_c, deaths=deaths,
                comas=comas, mismatches=mismatches, row=row, ts=ts, tps=sim._tps)


def main() -> int:
    law = run(False)
    abl = run(True)
    row, ts = law["row"], law["ts"]
    q = 65536.0
    c = lambda q16: ts.to_kelvin(q16 / q) - 273.15   # noqa: E731
    lo, hi = c(row.T_coma_exit_q) + 3.0, c(row.T_prefer_q)
    print(f"assertion 1: bulk warm-up mean {law['warm_mean']:.2f} C in [{lo:.2f}, {hi:.2f}] C")
    print(f"assertion 2: bulk cells below T_ctmin over the warm-up {100 * law['warm_cold']:.1f} % "
          f"(< 20 %)")
    print(f"cross-check mismatches: law {len(law['mismatches'])}, ablated {len(abl['mismatches'])}")
    fail = []
    if not lo <= law["warm_mean"] <= hi:
        fail.append("assertion 1")
    if law["warm_cold"] >= 0.20:
        fail.append("assertion 2 (stop and take it to Erik: §12)")
    if law["mismatches"] or abl["mismatches"]:
        fail.append("the kernel / RAW reader cross-check: " +
                    "; ".join((law["mismatches"] + abl["mismatches"])[:5]))
    if fail:
        print("FIRST LOOK FAILED: " + " | ".join(fail))
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig = plt.figure(figsize=(14, 9.5), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, :])
    fm = law["felt_mean"]
    ext = (0, W * TILE, H * TILE, 0)
    im = ax.imshow(fm, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=30.0, vmin=min(float(fm.min()), 0.0),
                                                       vmax=max(float(fm.max()), 60.0)),
                   extent=ext, interpolation="nearest")
    yy, xx = (np.mgrid[0:H, 0:W] + 0.5) * TILE
    cs = ax.contour(xx, yy, fm, levels=[10, 12, 30, 50], colors="k", linewidths=0.6)
    ax.clabel(cs, fmt="%d C", fontsize=6)
    pos = law["pos"] / q * TILE
    alive = law["state"] != 0
    for i in range(N_LARVAE):
        m = alive[:, i]
        ax.plot(pos[m, i, 0], pos[m, i, 1], lw=0.3, alpha=0.35, color="0.15")
    ax.scatter(pos[0, :, 0], pos[0, :, 1], s=3, color="tab:green", label="start", zorder=3)
    if law["deaths"]:
        d = np.array(law["deaths"], dtype=np.float64)
        ax.scatter(d[:, 1] / q * TILE, d[:, 2] / q * TILE, marker="x", s=14, color="k",
                   label="death", zorder=4)
    if law["comas"]:
        cm = np.array(law["comas"], dtype=np.float64)
        ax.scatter(cm[:, 1] / q * TILE, cm[:, 2] / q * TILE, s=5, color="tab:blue",
                   label="coma onset", zorder=4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("(a) time-mean felt temperature (RAW own tile) and larva trajectories")
    ax.set_anchor("C")
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax, label="felt temperature (C)", fraction=0.025)

    tsec = np.arange(law["state"].shape[0]) / law["tps"]
    ax2 = fig.add_subplot(gs[1, 0])
    st = law["state"]
    fr = [np.mean(st == s, axis=1) for s in (1, 3, 0)]
    ax2.stackplot(tsec, fr[0], fr[1], fr[2], labels=["RUN", "COMA", "DEAD"],
                  colors=["tab:orange", "tab:blue", "0.4"])
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("fraction of larvae")
    ax2.set_title("(b) state fractions (law)")
    ax2.legend(loc="lower left", fontsize=7)
    ax2.set_xlim(0, tsec[-1])
    ax2.set_ylim(0, 1)

    ax3 = fig.add_subplot(gs[1, 1])
    for res, lab in ((law, "law (bias 0.8, alarm 10x)"), (abl, "ablated (bias 0.5, alarm = base)")):
        run_m = res["state"] == 1
        dev = np.where(run_m, np.abs(res["felt_c"] - 30.0), np.nan)
        with np.errstate(all="ignore"):
            curve = np.array([np.nanmean(r) if np.isfinite(r).any() else np.nan for r in dev])
        ax3.plot(tsec[1:], curve[1:], lw=1.0, label=lab)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("mean |T_felt - 30 C| of RUN larvae (K)")
    ax3.set_title("(c) distance from the preferred temperature")
    ax3.legend(fontsize=7)
    ax3.set_xlim(0, tsec[-1])
    cap = (f"seed {SEED}; dials: speed {row.speed_mps_q / q:.2f} m/s, turn_bias "
           f"{row.turn_bias_q / q:.2f}, tumble {row.tumble_rate_base_q / q:.2f}/{row.tumble_rate_alarm_q / q:.2f} /s, "
           f"turns {row.turn_min_q / q:.2f}-{row.turn_max_q / q:.2f} rad, sweep {row.sweep_angle_q / q:.2f} rad; "
           f"reader RAW own tile; holds: strip 40 C, corner 70 C, cold 5 C (wave_source FieldEdits); "
           f"bulk warm-up mean {law['warm_mean']:.1f} C, cells below 10 C {100 * law['warm_cold']:.1f} %; "
           f"deaths law {len(law['deaths'])} / ablated {len(abl['deaths'])}")
    fig.text(0.01, 0.005, cap, fontsize=6.5, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "docs" / "swarm_P3_first_look.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
