"""P5z gates: the P1-P5 gates re-run at GAME SCALE (the mid octopus, row from
`game_scale.mid_row()`) in 3-D, plus the two no-slide gates of the P6 design
(section 6.3). Pure numpy + matplotlib.

    <breach-py> prototypes/octopus_motion/gates3d.py      (~4 min on 16 cores, writes out/)

Every scenario (`scenarios3d.JOBS`) runs in its own process; the gates read
their data. Prints PASS/FAIL per gate with its measurement, exits 1 on any
FAIL, writes out/report_p5z.md, out/row_mid.toml, out/fig11_*.png, the GIFs
out/{reach,crawl,swim,grab,idle}3d.gif and out/traces_p5z.npz.

Thresholds with dimensions are the 2-D gates' constants x S (lengths and
speeds; `game_scale.GateScale`); every other threshold is the 2-D gate's.

Author: Claude (octopus lead), 2026-10-30.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from arm_chain import N_SEG, Trace, bend_kinematics, mean_local_angle_deg, segment_angles  # noqa: E402
from crawl_sim import ARM_ANGLES, ARM_NAMES  # noqa: E402
from game_scale import OUT, mid_row  # noqa: E402

os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120})
ROW = mid_row(write=False)
S = ROW.S
GS = ROW.gates
DT = ROW.chain.dt_tick
L = ROW.geom.length_m
results: list[tuple[str, str, bool, str]] = []


def gate(group: str, name: str, ok, note: str) -> None:
    results.append((group, name, bool(ok), note))
    print(f"[{'PASS' if ok else 'FAIL'}] {group} {name}: {note}", flush=True)


def ang_deg(v) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def ang_err(a, b) -> float:
    return float(abs((ang_deg(a) - ang_deg(b) + 180) % 360 - 180))


# ====================================================================== P1

def _trace2d(P3: np.ndarray, T: np.ndarray, n: int) -> Trace:
    """The horizontal projection of a 3-D run (the reach plane), first n+1 ticks."""
    tr = Trace()
    for k in range(n + 1):
        tr.t.append(float(T[k]))
        tr.p.append(P3[k, :, :2].copy())
        tr.activation.append(np.zeros(N_SEG - 1))
    return tr


def profile_shape(t: np.ndarray, vel: np.ndarray) -> dict:
    ok = np.isfinite(vel)
    tt, vv = t[ok], vel[ok]
    if len(vv) < 5:
        return {"n": len(vv), "single_peak": False, "rise": 0.0, "end_frac": 1.0, "t_peak_frac": 0.0, "t_end": 0.0}
    peak = vv.max()
    i_pk = int(vv.argmax())
    ext, direction, ref = 0, 0, vv[0]
    for x in vv[1:]:
        if direction >= 0 and x < ref - 0.08 * peak:
            ext += 1
            direction = -1
        elif direction <= 0 and x > ref + 0.08 * peak:
            ext += (direction == -1)
            direction = 1
        ref = x if (direction > 0 and x > ref) or (direction < 0 and x < ref) else ref
    return {"n": len(vv), "single_peak": ext <= 1, "rise": float(peak / max(vv[: max(3, len(vv) // 6)].mean(), 1e-9)),
            "t_peak_frac": float((tt[i_pk] - tt[0]) / max(tt[-1] - tt[0], 1e-9)), "peak": float(peak), "t_end": float(tt[-1])}


def p1_gates(r: dict, nodrag: dict | None, tag: str) -> dict:
    T, P, s_node = r["T"], r["P"], r["s_node"]
    grp = r["groups"]
    G = f"P1[{tag}]"
    n16 = int(round(1.6 / DT))
    i_reach = grp["reach"][0]
    tr0 = _trace2d(P[:, i_reach], T, n16)
    t, s_b, pos_b, vel_b, amp_b = bend_kinematics(tr0, s_node)
    shape = profile_shape(t, vel_b)
    sag = float(P[: n16 + 1, i_reach, :, 2].min())
    out = dict(t=t, s_b=s_b, pos_b=pos_b, vel_b=vel_b, P=P[: n16 + 1, i_reach], sag=sag)
    gate(G, "a1 bell-shaped bend velocity",
         shape["single_peak"] and shape["rise"] >= 1.4 and shape["t_peak_frac"] >= 0.3,
         f"single peak {shape['single_peak']}, peak/early {shape['rise']:.2f}, peak at {shape['t_peak_frac']:.0%} of the "
         f"tracked time, the bend dissolves at t = {shape['t_end']:.2f} s (s = {np.nanmax(s_b) / L:.2f} L); peak "
         f"{shape.get('peak', 0):.2f} m/s; the base 1.5 m up, the arm sags to z = {sag:.2f} m under g_eff (gates read the horizontal plane)")

    def straight_behind(Pk, sb):
        k = int(np.searchsorted(s_node, sb)) - 2
        return mean_local_angle_deg(Pk, upto_node=k) if k >= 2 else np.nan

    ok_idx = [i for i in range(len(t)) if np.isfinite(s_b[i]) and s_b[i] > 0.4 * L]
    behind = np.array([straight_behind(tr0.p[i], s_b[i]) for i in ok_idx])
    gate(G, "a2 straight arm behind the bend", np.nanmax(behind) <= 3.0,
         f"mean local angle >= 2 nodes behind the bend, once past 0.4 L: max {np.nanmax(behind):.2f} deg, final whole-arm "
         f"{mean_local_angle_deg(tr0.p[-1]):.2f} deg")

    def distal_turn(Pk, n_last=5):
        return float(np.degrees(segment_angles(Pk)[-n_last:].sum()))

    end_idx = [i for i in range(len(t)) if (np.isfinite(s_b[i]) and s_b[i] > 0.8 * L) or (not np.isfinite(s_b[i]) and t[i] > 0.3)]
    mid_idx = [i for i in range(len(t)) if np.isfinite(s_b[i]) and s_b[i] <= 0.8 * L]
    tip_y_min = min(tr0.p[i][-1][1] for i in end_idx)
    dturn = min(distal_turn(tr0.p[i]) for i in end_idx)
    wiggle = min(distal_turn(tr0.p[i]) for i in mid_idx) if mid_idx else np.nan
    note_nd = ""
    if nodrag is not None:
        nd = nodrag["P"][:, :, :2]
        note_nd = (f" Control WITHOUT drag: distal turn {min(distal_turn(p) for p in nd):.0f} deg, tip y "
                   f"{min(p[-1][1] for p in nd) * 1e3:.0f} mm = Yekutieli's overshoot.")
    gate(G, "a3 tip never overshoots (end of the reach)", tip_y_min >= -0.05 * L and dturn >= -10.0,
         f"once the bend has passed 0.8 L: min tip y {tip_y_min * 1e3:.0f} mm (limit {-0.05 * L * 1e3:.0f}), min distal turn "
         f"{dturn:.1f} deg (limit -10).{note_nd} Mid-reach S-flex (reported): {wiggle:.1f} deg")

    t_acts = r["t_acts"]
    peaks = []
    for k, ta in enumerate(t_acts):
        n = int(round((ta + 0.8) / DT))
        tr = _trace2d(P[:, grp["tact"][0] + k], T, n)
        peaks.append(np.nanmax(bend_kinematics(tr, s_node)[3]))
    ratios = np.array(peaks) / (L / t_acts)
    natural = (t_acts >= 0.6) & (t_acts <= 1.2)
    gate(G, "a4 peak bend speed ~ wave speed (natural range t_act 0.6-1.2 s)",
         bool(np.all((ratios[natural] >= 0.7) & (ratios[natural] <= 1.3))),
         "peak / wave = " + ", ".join(f"{q:.2f} @ {ta} s" for q, ta in zip(ratios, t_acts)) + f" (wave {L / 0.9:.2f} m/s at 0.9 s)")

    a_cas = r["a_cas"]
    n_b = int(round(1.05 / DT))
    profiles, norm_curves = {}, {}
    for k, a in enumerate(a_cas):
        tr = _trace2d(P[:, grp["amp"][0] + k], T, n_b)
        tt, ss, pp, vv, aa = bend_kinematics(tr, s_node)
        profiles[a] = (tt, vv, mean_local_angle_deg(tr.p[-1]), tr.p[-1])
        ok = np.isfinite(vv)
        if ok.sum() >= 5:
            tn = (tt[ok] - tt[ok][0]) / (tt[ok][-1] - tt[ok][0])
            norm_curves[a] = np.interp(np.linspace(0, 1, 50), tn, vv[ok] / vv[ok].max())
    above = [a for a in a_cas if profiles[a][2] <= 2.0]
    if len(above) >= 3:
        ref = norm_curves[above[-1]]
        rms = {a: float(np.sqrt(np.mean((norm_curves[a] - ref) ** 2))) for a in above if a in norm_curves}
        pk = {a: float(np.nanmax(profiles[a][1])) for a in above}
        spread = (max(pk.values()) - min(pk.values())) / max(pk.values())
        gate(G, "b kinematics invariant to a_ca above the minimum force", max(rms.values()) <= 0.15 and spread <= 0.2,
             f"straight (<= 2 deg at t_act + 0.15 s) for a_ca {list(above)}; RMS vs {above[-1]}: "
             + ", ".join(f"{a}: {q:.3f}" for a, q in rms.items()) + f"; peak spread {spread:.0%}; below: "
             + ", ".join(f"{a} ends {profiles[a][2]:.1f} deg" for a in a_cas if a not in above))
    else:
        gate(G, "b kinematics invariant to a_ca above the minimum force", False,
             "fewer than three amplitudes straightened: " + ", ".join(f"{a}: {profiles[a][2]:.1f}" for a in a_cas))
    ta_grid, a_grid = r["ta_grid"], r["a_grid"]
    final = np.zeros((len(a_grid), len(ta_grid)))
    trav = np.zeros_like(final)
    for ia in range(len(a_grid)):
        for it, ta in enumerate(ta_grid):
            n = int(round((ta + 0.15) / DT))
            tr = _trace2d(P[:, grp["map"][0] + ia * len(ta_grid) + it], T, n)
            final[ia, it] = mean_local_angle_deg(tr.p[-1])
            ss = bend_kinematics(tr, s_node)[1]
            trav[ia, it] = np.nanmax(ss) / L if np.isfinite(ss).any() else 0.0
    out.update(final=final, trav=trav, profiles=profiles, norm_curves=norm_curves)
    print(f"  {G} min-force map: {int((final <= 2.0).sum())}/{final.size} straight; travel at t_act 0.1 s "
          f"{trav[:, 0].mean():.2f} L vs 0.9 s {trav[:, 5].mean():.2f} L", flush=True)
    if tag == "tanh":
        nd = int(round(2.0 / DT))
        trd = _trace2d(P[:, grp["dead"][0]], T, nd)
        td, sd, pd, vd, ad = bend_kinematics(trd, s_node)
        okd = np.isfinite(vd)
        vv = vd[okd]
        i_pk = int(vv.argmax()) if len(vv) else 0
        after = vv[i_pk:]
        mono = bool(np.all(np.diff(after) <= 0.05 * vv.max())) if len(after) > 2 else False
        gate(G, "c dead arm: bend decelerates monotonically, ends curved", mono and mean_local_angle_deg(trd.p[-1]) > 2.0,
             f"peak {vv.max() if len(vv) else 0:.2f} m/s at {td[okd][i_pk] if len(vv) else 0:.2f} s, then "
             f"{'monotone' if mono else 'NOT monotone'} over {len(after)} ticks; final mean angle {mean_local_angle_deg(trd.p[-1]):.1f} deg")
        out.update(dead=(td, vd, P[: nd + 1, grp["dead"][0]]))
    return out


def op_count_gate() -> None:
    n_nodes, n_dist, n_bend = 18, N_SEG, N_SEG - 1
    # tallied from arm_chain3d.Chain3D.substep for one arm, one substep; +, -, * count 1 each, a reciprocal
    # (division) and a sqrt weigh 8, an atan2 16 (the kit's costs, the P1 gate's weights)
    drag = n_nodes * 50
    drag_sp = n_nodes * (2 * 8 + 3 * 8)
    other = n_nodes * (1 + 6 + 9 + 5)                         # gravity, predict, velocity, floor
    dist = n_dist * 20
    dist_sp = n_dist * (8 + 8)
    frames = n_dist * 12 + (n_dist - 1) * 45 + n_dist * 9
    frames_sp = n_dist * (8 + 8) + (n_dist - 1) * (2 * 8 + 8)
    bend = 2 * n_bend * 140
    bend_sp = 2 * n_bend * (3 * 8 + 16)
    per_pass = dist + frames + bend
    per_pass_sp = dist_sp + frames_sp + bend_sp
    plain = drag + other + ROW.chain.n_pass * per_pass
    weighted = plain + drag_sp + ROW.chain.n_pass * per_pass_sp
    per_oct_tick = weighted * ROW.chain.n_sub * 8
    gate("P1", "d operation count per arm-substep (3-D)", weighted <= 25_000,
         f"~{plain} mul/add + specials -> ~{weighted} weighted mul/add-equivalents per arm-substep "
         f"({2 * n_bend} bend constraints, {n_dist} frame transports, {ROW.chain.n_pass} passes); x {ROW.chain.n_sub} x 8 arms "
         f"= ~{per_oct_tick / 1e6:.2f} M per octopus-tick (~{per_oct_tick / 1e6:.2f} ms at 1 ns/op; the 2-D chain was ~0.58 M). "
         "Cost is not a design input (Erik's ruling 1): reported, P6e measures")


# ====================================================================== P2

def analyse(d: dict) -> dict:
    T, X, Sn, Pr = d["T"], d["X"], d["S"], d["pred"]
    V = np.gradient(X, T, axis=0)
    sp = np.linalg.norm(V, axis=1)
    ok = (Sn.sum(1) > 0) & (sp > GS.min_speed) & (np.linalg.norm(Pr, axis=1) > 0)
    cosines = np.array([(V[i] @ Pr[i]) / (sp[i] * np.linalg.norm(Pr[i])) for i in np.where(ok)[0]])
    pa, aa = [], []
    for arm, t0, t1 in d["steps"]:
        if not np.isfinite(t1):
            continue
        i0, i1 = int(np.searchsorted(T, t0)), int(np.searchsorted(T, t1))
        if i1 - i0 < 2 or i1 >= len(T):
            continue
        dd = X[i1] - X[i0]
        pm = Pr[(i0 + i1) // 2]
        if np.linalg.norm(dd) < GS.push_min_disp or np.linalg.norm(pm) == 0:
            continue
        aa.append(np.arctan2(dd[1], dd[0]))
        pa.append(np.arctan2(pm[1], pm[0]))
    pa, aa = np.array(pa), np.array(aa)
    if len(pa) > 3:
        ma, mb = np.angle(np.exp(1j * pa).mean()), np.angle(np.exp(1j * aa).mean())
        num = np.sum(np.sin(pa - ma) * np.sin(aa - mb))
        den = np.sqrt(np.sum(np.sin(pa - ma) ** 2) * np.sum(np.sin(aa - mb) ** 2))
        r_circ = float(num / den) if den > 0 else np.nan
        perr = float(np.degrees(np.abs((pa - aa + np.pi) % (2 * np.pi) - np.pi)).mean())
    else:
        r_circ, perr = np.nan, np.nan
    disp = X[-1] - X[0]
    return dict(V=V, sp=sp, r=r_circ, cos_tick=float(cosines.mean()) if len(cosines) else np.nan, n_push=len(pa),
                push_err=perr, disp=disp, err=ang_err(disp, d["travel"]),
                heading_drift=float(np.degrees(d["theta"][-1] - d["theta"][0])),
                stance_mean=float(Sn.sum(1).mean()), stance_max=int(Sn.sum(1).max()), idle_frac=float((Sn.sum(1) == 0).mean()))


def spectrum(d: dict):
    a = analyse(d)
    sp = a["sp"] - a["sp"].mean()
    freqs = np.fft.rfftfreq(len(sp), d=DT)
    power = np.abs(np.fft.rfft(sp * np.hanning(len(sp)))) ** 2
    k = 15
    env = np.array([np.median(power[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(power))])
    white = power / np.maximum(env, 1e-30)
    band = (freqs >= 0.1) & (freqs <= 2.0)
    return a, freqs, white, band


def p2_gates(R: dict) -> dict:
    G = "P2"
    dirs = {0.0: "p2_d0", 70.0: "p2_d70", 135.0: "p2_d135", -90.0: "p2_dm90"}
    A = {d: analyse(R[k]) for d, k in dirs.items()}
    for d, a in A.items():
        print(f"  dir {d:+.0f}: disp {np.linalg.norm(a['disp']):.2f} m, err {a['err']:.1f} deg, r {a['r']:.3f} "
              f"({a['n_push']} pushes), speed {a['sp'].mean():.3f} m/s", flush=True)
    rs = [A[d]["r"] for d in dirs]
    errs = [A[d]["err"] for d in dirs]
    gate(G, "1 travels where the pushing arms say", min(rs) >= 0.9 and max(errs) <= 20.0,
         "circular r per push = " + ", ".join(f"{q:.2f}" for q in rs) + "; per-push |err| "
         + ", ".join(f"{A[d]['push_err']:.0f}" for d in dirs) + " deg; per-tick cos "
         + ", ".join(f"{A[d]['cos_tick']:.2f}" for d in dirs) + "; direction error " + ", ".join(f"{e:.0f}" for e in errs)
         + f" deg for wanted {list(dirs)} deg; speed mean " + ", ".join(f"{A[d]['sp'].mean():.3f}" for d in dirs)
         + f" m/s (2-D x S would be {0.05 * S:.2f}). KNOWN near miss in 2-D (0.88-0.94): not bent")
    gate(G, "2 one to three arms push", all(A[d]["stance_max"] <= 3 for d in dirs) and all(A[d]["stance_mean"] >= 0.7 for d in dirs),
         "stance mean " + ", ".join(f"{A[d]['stance_mean']:.2f}" for d in dirs) + ", max "
         + ", ".join(str(A[d]["stance_max"]) for d in dirs) + ", no arm pushing " + ", ".join(f"{A[d]['idle_frac']:.0%}" for d in dirs))
    al, f, w, b = spectrum(R["p2_long"])
    ac, fc, wc, bc = spectrum(R["p2_clock"])
    lr, lrc = float(w[b].max()), float(wc[bc].max())
    steps = R["p2_long"]["steps"]
    durs = np.array([t1 - t0 for _, t0, t1 in steps if np.isfinite(t1)])
    gaps = np.diff(np.sort(steps[:, 1]))
    gate(G, "3 no rhythm: no line in the speed spectrum", lr <= 12.0 and lrc > 12.0,
         f"whitened peak {lr:.1f}x (at {f[b][w[b].argmax()]:.2f} Hz); the CLOCKED control {lrc:.1f}x (at {fc[bc][wc[bc].argmax()]:.2f} Hz); "
         f"{len(steps)} steps in 40 s, push {durs.mean():.2f} +- {durs.std():.2f} s, gaps CV {gaps.std() / gaps.mean():.2f}")
    base = np.linalg.norm(A[0.0]["disp"])
    loss = {"7 arms (L4 dead)": analyse(R["p2_dead1"]), "6 arms (L4, R3 dead)": analyse(R["p2_dead2"])}
    gate(G, "4 still travels with 7 and 6 arms",
         all(np.linalg.norm(a["disp"]) >= 0.5 * base and a["err"] <= 25.0 for a in loss.values()),
         "; ".join(f"{k}: {np.linalg.norm(a['disp']) / base:.0%} of 8-arm displacement, {a['err']:.0f} deg off" for k, a in loss.items()))
    drifts = [A[d]["heading_drift"] for d in dirs]
    gate(G, "5 heading independent of travel direction", max(abs(x) for x in drifts) < 30.0,
         "heading drift over 25 s = " + ", ".join(f"{x:+.1f}" for x in drifts) + " deg")
    z_all = max(float(np.abs(R[k]["P"][..., 2]).max()) for k in dirs.values())
    gate(G, "0z the crawl stays on the deck", z_all <= 0.01 * S,
         f"max |z| of any node over the four 25 s runs {z_all * 1e3:.2f} mm (a crawl is planar; the floor holds it)")
    return dict(A=A, spec=(f, w, fc, wc, lr, lrc), loss=loss, crawl_speed=float(A[0.0]["sp"].mean()))


# ====================================================================== P3

def kin(d):
    T, X = d["T"], d["X"]
    V = np.gradient(X, T, axis=0)
    return T, X, V, np.linalg.norm(V, axis=1)


def p3_gates(R: dict, crawl_speed: float) -> dict:
    G = "P3"
    Tq, Xq, Vq, spq = kin(R["p3_umbrella"])
    Tl, Xl, Vl, spl = kin(R["p3_linear"])
    st = Tq > 5.0
    v_umb = float(Vq[st, 0].mean())
    v_ctl = float(Vl[Tl > 5.0, 0].mean())
    per = ROW.swim.open_s + ROW.swim.close_s
    ML = ROW.body.mantle_long
    sink = float(R["p3_umbrella"]["Z"][0] - R["p3_umbrella"]["Z"][-1])
    gate(G, "1 the umbrella travels from drag alone", v_umb > 0.0 and abs(v_ctl) < 0.15 * v_umb,
         f"steady {v_umb:.3f} m/s ({v_umb / ML:.2f} mantle lengths/s, {v_umb * per:.2f} m per stroke) with the real drag; the "
         f"linear isotropic control {v_ctl:.3f} m/s ({abs(v_ctl) / max(v_umb, 1e-9):.0%}); 8 real arms in a cone; the body "
         f"sank {sink:.2f} m in 10 s under g_eff (reported)")
    Tj, Xj, Vj, spj = kin(R["p3_jet"])
    i_pk = int(spj.argmax())
    after = np.where((Tj > Tj[i_pk]) & (spj < 0.1 * spj[i_pk]))[0]
    t_decay = float(Tj[after[0]] - Tj[i_pk]) if len(after) else np.inf
    dist_jet = float(Xj[-1, 0] - Xj[0, 0])
    gate(G, "3 a jet carries a few mantle lengths, then decays", dist_jet >= 3 * ML and t_decay <= 5.0,
         f"peak {spj[i_pk]:.2f} m/s, {dist_jet:.2f} m = {dist_jet / ML:.1f} mantle lengths ({ML} m), < 10 % of peak after "
         f"{t_decay:.1f} s (jet dv {ROW.swim.jet_dv:.2f} m/s over {ROW.swim.jet_s} s)")
    Te, Xe, Ve, spe = kin(R["p3_escape"])
    Tt, Xt, Vt, spt = kin(R["p3_train"])
    v_esc = float(Ve[Te > 5.0, 0].mean())
    v_tr = float(Vt[Tt > 5.0, 0].mean())
    gate(G, "2 swimming >= 5x the crawl", v_tr >= 5 * crawl_speed,
         f"jet train {v_tr:.2f} m/s = {v_tr / crawl_speed:.1f}x the P5z crawl ({crawl_speed:.3f} m/s; "
         f"{v_tr / GS.crawl_speed_ref:.1f}x the 2-D crawl x S); umbrella alone {v_umb / crawl_speed:.1f}x; umbrella + half-jets "
         f"({R['p3_escape']['jets']} in 10 s) {v_esc:.2f} m/s = {v_esc / crawl_speed:.1f}x; a single jet peaks {spj[i_pk] / crawl_speed:.0f}x")
    return dict(Tq=Tq, Xq=Xq, Vq=Vq, Tl=Tl, Xl=Xl, Vl=Vl, Tj=Tj, spj=spj, Te=Te, spe=spe, Tt=Tt, spt=spt,
                v_umb=v_umb, v_ctl=v_ctl, v_tr=v_tr, dist_jet=dist_jet, t_decay=t_decay)


# ====================================================================== P4

def p4_gates(R: dict) -> dict:
    G = "P4"
    d = R["p4_grab"]
    T, X, PR3, NG = d["T"], d["X"], d["PR3"], d["NG"]
    a, t_contact, t_carry, t_jets = d["arm"], d["t_contact"], d["t_carry"], d["t_jets"]
    sealed_s = d["sealed_s"]
    gate(G, "1 the grab closes mid-arm",
         np.isfinite(t_contact) and len(sealed_s) >= 3 and all(0.2 * L <= s <= 0.65 * L for s in sealed_s),
         f"contact {t_contact - 0.5:.2f} s after the reach began at s = {', '.join(f'{s / L:.2f}' for s in d['grab_s'])} L; after the "
         f"wrap {len(sealed_s)} sealed at s = {', '.join(f'{s / L:.2f}' for s in sealed_s)} L")
    i_c = int(np.searchsorted(T, t_carry))
    i_j = int(np.searchsorted(T, t_jets[0]))
    dp = np.linalg.norm(PR3[:, :2] - X, axis=1)
    d_ref = dp[i_c + 24: i_c + 48].mean()
    ng_min = int(NG[i_c:].min())
    gate(G, "2 the prop stays attached through the carry and three jets", ng_min >= 3 and d["popped"] == 0,
         f"sealed during the carry min {ng_min}, popped {d['popped']} (grip {ROW.grab.grip_limit:.0f} N); held at {d_ref:.2f} m "
         f"from the crown, swinging {dp[i_c:i_j].min() / d_ref - 1:+.0%}..{dp[i_c:i_j].max() / d_ref - 1:+.0%} in the crawl and "
         f"{dp[i_j:].min() / d_ref - 1:+.0%}..{dp[i_j:].max() / d_ref - 1:+.0%} through the jets (reported)")
    steps = d["steps"]
    pushed_after = int(sum(1 for (arm, t0, _) in steps if arm == a and t0 >= t_contact))
    disp = X[i_j] - X[i_c]
    v_loaded = float(np.linalg.norm(disp) / (T[i_j] - T[i_c]))
    err = float(abs((np.degrees(np.arctan2(disp[1], disp[0])) + 180) % 360 - 180))
    Xf = R["p4_free"]["X"]
    v_free = float(np.linalg.norm(Xf[-1] - Xf[0]) / 15.0)
    who = sorted(set(ARM_NAMES[int(arm)] for (arm, t0, _) in steps if t0 >= t_contact))
    gate(G, "3 the other arms re-plan around the loaded arm", pushed_after == 0 and err <= 15.0 and v_loaded >= 0.5 * v_free,
         f"{ARM_NAMES[a]} pushed {pushed_after} times after the grab; loaded crawl {v_loaded:.3f} m/s ({v_loaded / v_free:.0%} of "
         f"the unloaded {v_free:.3f}), {err:.0f} deg off; arms that pushed: {', '.join(who)}")
    z = PR3[:, 2]
    i_lift = int(np.searchsorted(T, d["t_hold"] + ROW.grab.lift_s))
    z_carry = z[i_lift:]
    gate(G, "4 LIFT: the prop leaves the deck and stays up through the crawl and the jets",
         z.max() >= 0.1 and z_carry.min() > 1e-3 and ng_min >= 3,
         f"prop z max {z.max():.2f} m (limit 0.10 m); from the end of the lift ({T[i_lift]:.1f} s) to the end, z "
         f"{z_carry.min():.2f}..{z_carry.max():.2f} m, mean {z_carry.mean():.2f} m; root pitched {ROW.grab.lift_pitch_deg} deg "
         f"over {ROW.grab.lift_s} s; prop {ROW.grab.prop_mass:.0f} kg")
    return dict(i_c=i_c, i_j=i_j, i_lift=i_lift)


# ====================================================================== P5

def p5_gates(R: dict) -> dict:
    G = "P5"
    d = R["p5_idle"]
    T, P, X = d["T"], d["P"], d["X"]
    V = np.gradient(P, T, axis=0)
    ns = np.linalg.norm(V, axis=3)
    moving = ns[:, :, 9:].max(axis=2) > GS.move_speed
    moving[T < 1.0] = False
    nm = moving.sum(axis=1)
    gate(G, "1 few actors", nm.max() <= 3 and np.percentile(nm, 95) <= 2,
         f"arms whose distal half moves (> {GS.move_speed:.2f} m/s) at once: max {nm.max()}, 95th pct {np.percentile(nm, 95):.0f}, mean {nm.mean():.2f}")
    from scipy.stats import spearmanr
    rhos = []
    for arm, t0, t1 in d["gestures"]:
        if not np.isfinite(t1):
            continue
        i0 = int(np.searchsorted(T, t0))
        i1 = min(int(np.searchsorted(T, t0 + 1.3)), len(T) - 1)
        if i1 - i0 < 6:
            continue
        seg = ns[i0:i1, int(arm), 2:]
        rhos.append(float(spearmanr(np.arange(seg.shape[1]), seg.argmax(axis=0))[0]))
    rhos = np.array(rhos)
    frac = float((rhos >= 0.8).mean()) if len(rhos) else 0.0
    gate(G, "3 willed motion starts at the base", len(rhos) >= 5 and frac >= 0.8,
         f"{len(rhos)} gestures; Spearman(node, time of peak speed) median {np.median(rhos):.2f}, {frac:.0%} >= 0.8")
    still = nm == 0
    runs, cur = [], 0
    for s_ in still:
        if s_:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    longest = max(runs) * DT if runs else 0.0
    on = np.sort(d["gestures"][:, 1])
    gaps = np.diff(on)
    cv = float(gaps.std() / gaps.mean()) if len(gaps) > 2 else np.nan
    gate(G, "7 holds and rests", still.mean() >= 0.25 and longest >= 1.5 and cv >= 0.5,
         f"still {still.mean():.0%}, longest still {longest:.1f} s; gaps between gesture starts {gaps.mean():.2f} +- {gaps.std():.2f} s (CV {cv:.2f})")
    starts = sorted((t0, int(arm)) for arm, t0, _ in d["gestures"])
    close = sum(1 for i in range(1, len(starts)) if starts[i][0] - starts[i - 1][0] < 0.3 and starts[i][1] != starts[i - 1][1])
    counts = np.bincount([int(arm) for arm, _, _ in d["gestures"]], minlength=8)
    front, back = counts[[0, 7]].mean(), counts[[3, 4]].mean()
    gate(G, "8 de-synchrony and roles", close / max(len(starts), 1) <= 0.10 and counts.min() >= 1 and front >= back,
         f"starts within 0.3 s of another arm's: {close}/{len(starts)}; per arm " + ", ".join(f"{ARM_NAMES[a]} {counts[a]}" for a in range(8))
         + f"; front {front:.1f} vs back {back:.1f}")
    pe = P[-1]
    seg = np.linalg.norm(pe[:, 1:] - pe[:, :-1], axis=2)
    still_arms = [a for a in range(8) if d["ex_final"][a] == "still"]
    seg_err = float(np.abs(seg[still_arms] / d["rest_final"][still_arms] - 1).max()) if still_arms else np.nan
    reach = np.linalg.norm(pe - pe[:, :1], axis=2).max() / L
    drift = float(np.linalg.norm(X[-1] - X[0]))
    gate(G, "13 stable over time", seg_err <= 0.03 and reach <= 1.5 and drift <= GS.drift,
         f"still arms' segments within {seg_err:.1%} of rest, farthest node {reach:.2f} L from its root, body drift "
         f"{drift * 1e3:.1f} mm in 60 s (limit {GS.drift * 1e3:.0f} mm)")
    return dict(nm=nm)


# ====================================================================== the two new gates

def new_gates(R: dict) -> dict:
    d = R["n1_dead"]
    T, P, V = d["T"], d["P"], d["V"]
    last = T > T[-1] - 5.0
    zmax = float(P[last, :, 2].max())
    vmax = float(np.linalg.norm(V[last], axis=2).max())
    t_touch = float(T[np.argmax(P[:, :, 2].min(axis=1) <= 1e-9)])
    t_all = float(T[np.argmax(P[:, :, 2].max(axis=1) <= 5e-4)]) if (P[:, :, 2].max(axis=1) <= 5e-4).any() else np.nan
    drift = float(np.linalg.norm(P[-1, :, :2].mean(0) - d["p0"][:, :2].mean(0)))
    gate("NEW", "N1 a dead arm released 0.2 m above the deck sinks to it and rests",
         zmax <= 5e-4 and vmax <= 5e-3,
         f"first node on the deck at {t_touch:.1f} s, every node within 0.5 mm of it from {t_all:.1f} s; over the last 5 s of "
         f"{T[-1]:.0f} s: max z {zmax * 1e3:.2f} mm, max node speed {vmax * 1e3:.2f} mm/s (limits 0.5 mm, 5 mm/s); the curled arm "
         f"also relaxes on its passive stiffness, its centroid moving {drift:.2f} m over the deck (reported)")
    f = R["n2_free"]
    C = f["COM"]
    com = float(np.linalg.norm(C[-1, :2] - C[0, :2]))
    com_max = float(np.linalg.norm(C[:, :2] - C[0, :2], axis=1).max())
    body = float(np.linalg.norm(f["X"] - f["X"][0], axis=1).max())
    fq = R["n2_free_quad"]
    Cq = fq["COM"]
    comq = float(np.linalg.norm(Cq[-1, :2] - Cq[0, :2]))
    gate("NEW", "N2 a free animal does not slide: centre of mass < 1 cm over 60 s",
         com_max < 0.01,
         f"symmetric drag (uniform rate, tau {f['tau']} s), no anchors, mu 0, idle waves on ({len(f['gestures'])} gestures): "
         f"horizontal COM moved max {com_max:.2e} m (end {com:.2e} m) while the body itself swung up to {body:.2f} m -- the float "
         f"baseline the integer port must match. With the REAL quadratic anisotropic drag the same animal's COM moves "
         f"{comq:.3f} m in 60 s: drag-nonlinearity thrust (P3's physics), not slide (reported)")
    return dict(com=C, comq=Cq)


# ====================================================================== figures and GIFs

def fig_reach(p1t, p1b, r_t, r_b) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    T = r_t["T"]
    cmap = plt.get_cmap("viridis")
    P = p1t["P"]
    for k in range(0, len(P), 3):
        axs[0, 0].plot(P[k, :, 0], P[k, :, 1], "-", color=cmap(k / len(P)), lw=2)
        axs[0, 1].plot(P[k, :, 0], P[k, :, 2], "-", color=cmap(k / len(P)), lw=2)
    axs[0, 0].set_aspect("equal")
    axs[0, 0].set_title(f"Reach at game scale, top-down (L = {L:.2f} m, a_ca 0.6, t_act 0.9 s)")
    axs[0, 1].set_aspect("equal")
    axs[0, 1].set_title(f"Side view: the base at z = 1.5 m, the arm sags to {p1t['sag']:.2f} m under g_eff")
    axs[0, 1].set_xlabel("x (m)")
    axs[0, 1].set_ylabel("z (m)")
    ax = axs[1, 0]
    ax.plot(p1t["t"], p1t["vel_b"], color="crimson", lw=2, label="bend speed, tanh")
    ax.plot(p1b["t"], p1b["vel_b"], color="k", ls="--", lw=1.2, label="bend speed, tabulated S (1025 samples)")
    ax.axhline(L / 0.9, color="gray", ls=":", label=f"wave L / t_act = {L / 0.9:.2f} m/s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=7)
    ax.set_title("Bend kinematics: the table reproduces the sigmoid")
    ax = axs[1, 1]
    im = ax.pcolormesh(r_t["ta_grid"], r_t["a_grid"], p1t["final"], shading="nearest", cmap="magma_r", vmin=0, vmax=12)
    ax.contour(r_t["ta_grid"], r_t["a_grid"], p1t["final"], levels=[2.0], colors="cyan", linewidths=2)
    fig.colorbar(im, ax=ax, label="final mean local angle (deg)")
    ax.set_xscale("log")
    ax.set_xlabel("t_act (s)")
    ax.set_ylabel("a_ca")
    ax.set_title("Minimum-force map at game scale (cyan = 2 deg)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_reach.png"))
    plt.close(fig)


def fig_crawl(R, p2) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    ax = axs[0, 0]
    for dd, k in [(0.0, "p2_d0"), (70.0, "p2_d70"), (135.0, "p2_d135"), (-90.0, "p2_dm90")]:
        X = R[k]["X"]
        ax.plot(X[:, 0], X[:, 1], lw=1.8, label=f"wanted {dd:+.0f} deg (err {p2['A'][dd]['err']:.0f})")
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.set_title("Body paths at game scale, 25 s each (heading +x)")
    d = R["p2_d0"]
    ax = axs[0, 1]
    for arm in range(8):
        ax.fill_between(d["T"], arm - 0.4, arm + 0.4, where=d["S"][:, arm], color=f"C{arm}", step="mid")
    ax.set_yticks(range(8))
    ax.set_yticklabels(ARM_NAMES)
    ax.set_title("Which arm pushes: no order, no clock")
    ax = axs[1, 0]
    a = p2["A"][0.0]
    ax.plot(d["T"], a["sp"], color="k", lw=1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m/s")
    ax.set_title(f"Body speed (mean {a['sp'].mean():.2f} m/s)")
    ax = axs[1, 1]
    f, w, fc, wc, lr, lrc = p2["spec"]
    ax.semilogy(f[1:], w[1:], color="k", lw=0.8, label=f"lottery {lr:.1f}x")
    ax.semilogy(fc[1:], wc[1:], color="C3", lw=0.8, alpha=0.7, label=f"clocked {lrc:.1f}x")
    ax.axhline(12, color="gray", ls="--")
    ax.set_xlim(0, 3)
    ax.legend(fontsize=7)
    ax.set_title("Whitened speed spectrum, 40 s")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_crawl.png"))
    plt.close(fig)


def fig_swim(R, p3) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axs[0, 0]
    ax.plot(p3["Tq"], p3["Xq"][:, 0], label=f"real drag {p3['v_umb']:.2f} m/s")
    ax.plot(p3["Tl"], p3["Xl"][:, 0], label=f"linear isotropic control {p3['v_ctl']:.3f} m/s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("x (m)")
    ax.legend(fontsize=7)
    ax.set_title("Umbrella, 8 arms in 3-D")
    ax = axs[0, 1]
    ax.plot(p3["Tj"], p3["spj"], label="jet")
    ax.plot(p3["Te"], p3["spe"], label="umbrella + half jets")
    ax.plot(p3["Tt"], p3["spt"], label=f"jet train {p3['v_tr']:.2f} m/s")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=7)
    ax.set_title("Speeds")
    ax = axs[1, 0]
    d = R["p3_umbrella"]
    i0 = int(np.argmin(np.abs(d["T"] - 7.0)))
    n_per = int(round((ROW.swim.open_s + ROW.swim.close_s) / DT))
    cmap = plt.get_cmap("viridis")
    for k in range(0, n_per, max(1, n_per // 7)):
        i = i0 + k
        for a in (1, 6):
            ax.plot(d["P"][i, a, :, 0] - d["X"][i, 0], d["P"][i, a, :, 2] - d["Z"][i], color=cmap(k / n_per), lw=1.5)
    ax.set_aspect("equal")
    ax.set_title("One stroke, side view in the body frame (arms L2 and R2)")
    ax = axs[1, 1]
    for k in ("p3_umbrella", "p3_jet", "p3_train"):
        ax.plot(R[k]["T"], R[k]["Z"], label=k[3:])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("body z (m)")
    ax.legend(fontsize=7)
    ax.set_title("Height: the animal sinks slowly under g_eff (reported)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_swim.png"))
    plt.close(fig)


def fig_grab(R, p4) -> None:
    d = R["p4_grab"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    T, P = d["T"], d["P"]
    cmap = plt.get_cmap("viridis")
    a = d["arm"]
    for ts in [0.5, 0.8, d["t_contact"], d["t_carry"], d["t_carry"] + 5, d["t_jets"][0] + 1]:
        i = min(int(np.searchsorted(T, ts)), len(T) - 1)
        col = cmap(ts / T[-1])
        for arm in range(8):
            axs[0].plot(P[i, arm, :, 0], P[i, arm, :, 1], color=col, lw=2.0 if arm == a else 0.8)
        axs[0].add_patch(Circle(d["PR3"][i, :2], d["prop_radius"], color=col, fill=False))
    axs[0].set_aspect("equal")
    axs[0].set_title("Reach, grab, carry (top-down; thick = the holding arm)")
    axs[1].plot(T, d["PR3"][:, 2], color="C3", lw=1.8, label="prop z")
    for tj in d["t_jets"]:
        axs[1].axvline(tj, color="gray", ls=":")
    axs[1].set_xlabel("t (s)")
    axs[1].set_ylabel("m")
    axs[1].legend(fontsize=7)
    axs[1].set_title("LIFT: the prop's height (dotted = jets)")
    i = p4["i_j"] - 12
    for arm in range(8):
        axs[2].plot(P[i, arm, :, 0], P[i, arm, :, 2], color="C0" if arm != a else "C3", lw=2.0 if arm == a else 0.8)
    axs[2].add_patch(Circle((d["PR3"][i, 0], d["PR3"][i, 2]), d["prop_radius"], color="C3", fill=False))
    axs[2].set_aspect("equal")
    axs[2].set_title(f"Side view while carrying (t = {T[i]:.1f} s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_grab.png"))
    plt.close(fig)


def fig_idle(R, p5) -> None:
    d = R["p5_idle"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    axs[0].plot(d["T"], p5["nm"], color="k", lw=0.8)
    axs[0].set_xlabel("t (s)")
    axs[0].set_title("Arms moving at once")
    for arm, t0, t1 in d["gestures"]:
        axs[1].plot([t0, t1 if np.isfinite(t1) else d["T"][-1]], [arm, arm], lw=6, color=f"C{int(arm)}")
    axs[1].set_yticks(range(8))
    axs[1].set_yticklabels(ARM_NAMES)
    axs[1].set_title("Gestures (reach -> touch -> limp)")
    P = d["P"]
    for k in np.linspace(0, len(P) - 1, 6).astype(int):
        for arm in range(8):
            axs[2].plot(P[k, arm, :, 0], P[k, arm, :, 1], color=plt.get_cmap("viridis")(k / len(P)), lw=0.8)
    axs[2].set_aspect("equal")
    axs[2].set_title("Every 12 s, top-down")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_idle.png"))
    plt.close(fig)


def fig_new(R, nw) -> None:
    d = R["n1_dead"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    axs[0].plot(d["T"], d["P"][:, :, 2].max(axis=1), label="highest node")
    axs[0].plot(d["T"], d["P"][:, :, 2].min(axis=1), label="lowest node")
    axs[0].set_xlabel("t (s)")
    axs[0].set_ylabel("z (m)")
    axs[0].legend(fontsize=7)
    axs[0].set_title("N1: a dead arm released 0.2 m above the deck")
    for k in np.linspace(0, len(d["P"]) - 1, 8).astype(int):
        axs[1].plot(d["P"][k, :, 0], d["P"][k, :, 1], color=plt.get_cmap("viridis")(k / len(d["P"])), lw=1.5)
    axs[1].set_aspect("equal")
    axs[1].set_title("Top-down: it relaxes on its passive stiffness, then rests")
    C, Cq = nw["com"], nw["comq"]
    tt = np.arange(len(C)) * DT
    axs[2].semilogy(tt, np.maximum(np.linalg.norm(C[:, :2] - C[0, :2], axis=1), 1e-18), label="symmetric drag (gated)")
    axs[2].semilogy(np.arange(len(Cq)) * DT, np.maximum(np.linalg.norm(Cq[:, :2] - Cq[0, :2], axis=1), 1e-18), label="real quadratic drag")
    axs[2].axhline(0.01, color="gray", ls="--")
    axs[2].set_xlabel("t (s)")
    axs[2].set_ylabel("|COM displacement| (m)")
    axs[2].legend(fontsize=7)
    axs[2].set_title("N2: a free animal's centre of mass")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_new.png"))
    plt.close(fig)


def _gif(name: str, P: np.ndarray, T: np.ndarray, views, title: str, extra=None, body=None, every: int = 1,
         follow: float | None = None, lim=None) -> str:
    """P (frames, arms, N, 3); views: list of (i, j) axis pairs, e.g. [(0, 1)] top-down, (0, 2) side."""
    frames = list(range(0, len(P), every))
    fig, axs = plt.subplots(1, len(views), figsize=(5.2 * len(views), 4.6), squeeze=False)
    axs = axs[0]
    lines = []
    for ax, (i, j) in zip(axs, views):
        ax.set_aspect("equal")
        ax.set_xlabel("xyz"[i] + " (m)")
        ax.set_ylabel("xyz"[j] + " (m)")
        lines.append([ax.plot([], [], "-", color="#5a3b8a", lw=2.2, solid_capstyle="round")[0] for _ in range(P.shape[1])])
    bodies = []
    if body is not None:
        for ax in axs:
            c = Circle((0, 0), body[1], color="#5a3b8a", alpha=0.5)
            ax.add_patch(c)
            bodies.append(c)
    props = []
    if extra is not None:
        for ax in axs:
            c = Circle((0, 0), extra[1], color="C3", alpha=0.5)
            ax.add_patch(c)
            props.append(c)
    txt = axs[0].text(0.02, 0.97, "", transform=axs[0].transAxes, fontsize=8, va="top")
    fig.suptitle(title, fontsize=9)
    allP = P.reshape(-1, 3)

    def frame(k):
        f = frames[k]
        for ax, (i, j), ls in zip(axs, views, lines):
            for arm, ln in enumerate(ls):
                ln.set_data(P[f, arm, :, i], P[f, arm, :, j])
            if follow is not None:
                cx, cy = (body[0][f, i], body[0][f, j]) if body is not None else (P[f, :, 0, i].mean(), P[f, :, 0, j].mean())
                ax.set_xlim(cx - follow, cx + follow)
                ax.set_ylim(cy - follow * 0.8, cy + follow * 0.8)
        for ax, (i, j), c in zip(axs, views, bodies):
            c.center = (body[0][f, i], body[0][f, j])
        for ax, (i, j), c in zip(axs, views, props):
            c.center = (extra[0][f, i], extra[0][f, j])
        txt.set_text(f"t = {T[f]:.2f} s")
        return []

    if follow is None:
        for ax, (i, j) in zip(axs, views):
            lo, hi = allP.min(0), allP.max(0)
            pad = 0.1 * L
            ax.set_xlim(lo[i] - pad, hi[i] + pad)
            ax.set_ylim(lo[j] - pad, hi[j] + pad)
    anim = FuncAnimation(fig, frame, frames=len(frames), interval=1000 * DT * every)
    path = os.path.join(OUT, name)
    anim.save(path, writer=PillowWriter(fps=max(1, int(round(1 / (DT * every))))))
    plt.close(fig)
    return path


def gif_job(kind: str, d: dict) -> str:
    if kind == "reach":
        P = d["P"][:, None]
        return _gif("reach3d.gif", P, d["T"], [(0, 1), (0, 2)], "P5z reach at game scale: top-down | side (base 1.5 m up)")
    if kind == "crawl":
        n = int(12.0 / DT)
        X3 = np.concatenate([d["X"], d["Z"][:, None]], axis=1)
        return _gif("crawl3d.gif", d["P"][:n], d["T"][:n], [(0, 1)], "P5z crawl, wanted 70 deg off the heading (top-down)",
                    body=(X3[:n], d["radius"]), follow=4.5)
    if kind == "swim":
        X3 = np.concatenate([d["X"], d["Z"][:, None]], axis=1)
        return _gif("swim3d.gif", d["P"], d["T"], [(0, 1), (0, 2)], "P5z swim: umbrella strokes then a jet (top-down | side)",
                    body=(X3, d["radius"]), follow=4.0)
    if kind == "grab":
        X3 = np.concatenate([d["X"], d["Z"][:, None]], axis=1)
        return _gif("grab3d.gif", d["P"], d["T"], [(0, 1), (0, 2)], "P5z reach, grab, LIFT, carry, jets (top-down | side)",
                    body=(X3, d["radius"]), extra=(d["PR3"], d["prop_radius"]), every=2, follow=5.0)
    if kind == "idle":
        n = int(24.0 / DT)
        X3 = np.concatenate([d["X"], d["Z"][:, None]], axis=1)
        return _gif("idle3d.gif", d["P"][:n], d["T"][:n], [(0, 1)], "P5z idle, the seated animal (top-down)",
                    body=(X3[:n], d["radius"]), every=1, follow=4.0)
    raise ValueError(kind)


# ====================================================================== traces

def write_traces(R: dict) -> str:
    keys = {"p1_trace": "reach", "n1_dead": "dead_arm", "n2_free": "free_animal", "p2_d0": "crawl",
            "p3_jet": "swim_jet", "p4_grab": "grab", "p5_idle": "idle"}
    out = {"README": np.array(
        "P5z float reference traces (float64). Per scenario '<name>/<field>'. p: node positions per tick AFTER the step "
        "(ticks, arms, 18, 3) [reach / dead_arm: (ticks, 18, 3)], p0 the initial positions; in_*: the solver inputs per "
        "tick as the hubs wrote them BEFORE the step (activation (A,16), rest_scale (A,17), bend_target (A,16,2) "
        "[plane A = toward nA, plane B = toward nB = nA x t], anchored / anchor_pos, root_local (A,2,3) in the body frame, "
        "grip (N), grab_body (A,18: -1 or the index into body_*), grab_local (A,18,3) in that body's frame, body_v / "
        "body_omega: the bodies' velocity entering the step, i.e. including the hub's jet impulse); body_x / body_theta "
        "after the step; the solver params and geometry as scalars. reach: the activation is the wave "
        "(wave_a, wave_tact, wave_t0, wave_w, wave_hold), evaluated per SUBSTEP from the chain clock. Everything is "
        "deterministic and seeded; dt_tick = 1/24 s, n_sub x n_pass substeps.")}
    for job, name in keys.items():
        tr = R[job].get("trace")
        if tr is None:
            continue
        for k, v in tr.items():
            out[f"{name}/{k}"] = np.asarray(v)
    path = os.path.join(OUT, "traces_p5z.npz")
    np.savez_compressed(path, **out)
    return path


# ====================================================================== main

def main() -> int:
    t_all = time.perf_counter()
    import scenarios3d
    mid_row(write=True)
    names = sorted(scenarios3d.JOBS, key=lambda k: -{"p5_idle": 60, "n2_free": 60, "n2_free_quad": 60, "p2_long": 40,
                                                       "p2_clock": 40, "p4_grab": 32}.get(k, 20))
    R = {}
    workers = max(2, min(len(names), (os.cpu_count() or 4) - 2))
    print(f"running {len(names)} scenarios on {workers} processes ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, res in ex.map(scenarios3d.run_job, names):
            R[name] = res
            print(f"  {name}: {res['wall_s']:.0f} s", flush=True)
    t_sim = time.perf_counter() - t_all
    # ---- gates
    e = R["p1_equal"]
    gate("P1", "0 the 3-D chain == the 2-D P1 chain in a plane (g = 0, mid scale)", e["dmax"] <= 1e-9,
         f"max |dp| {e['dmax']:.1e} m over 1 s of the reach (in-plane and out-of-plane)")
    gate("P1", "0b the reach is the same in a tilted plane (Bishop frame)", e["dtilt"] <= 1e-9,
         f"pose + root reference rotated 50 deg about (1, 0.3, 0) and 20 deg about z, run, rotated back: max |dp| {e['dtilt']:.1e} m")
    gate("P1", "0c the tabulated activation S(x) matches tanh", e["tab_err"] <= 2e-5,
         f"max |S_table - S_tanh| {e['tab_err']:.1e} over x in [-10, 10] (1025 samples, step 1/64, linear)")
    p1t = p1_gates(R["p1_tanh"], R["p1_nodrag"], "tanh")
    p1b = p1_gates(R["p1_table"], None, "table")
    op_count_gate()
    p2 = p2_gates(R)
    p3 = p3_gates(R, p2["crawl_speed"])
    p4 = p4_gates(R)
    p5 = p5_gates(R)
    nw = new_gates(R)
    # ---- figures, traces, GIFs
    fig_reach(p1t, p1b, R["p1_tanh"], R["p1_table"])
    fig_crawl(R, p2)
    fig_swim(R, p3)
    fig_grab(R, p4)
    fig_idle(R, p5)
    fig_new(R, nw)
    tr_path = write_traces(R)
    gif_data = {"reach": dict(P=p1t["P"], T=R["p1_tanh"]["T"][: len(p1t["P"])]), "crawl": R["p2_d70"],
                "swim": R["p3_gif"], "grab": R["p4_grab"], "idle": R["p5_idle"]}
    t_g = time.perf_counter()
    with ProcessPoolExecutor(max_workers=5) as ex:
        paths = list(ex.map(gif_job, list(gif_data), list(gif_data.values())))
    t_gif = time.perf_counter() - t_g
    wall = time.perf_counter() - t_all
    n_fail = sum(1 for r in results if not r[2])
    # ---- report
    lines = ["# P5z gate report -- the chain at GAME scale in 3-D (the mid octopus)\n",
             f"Generated by `gates3d.py`, {time.strftime('%Y-%m-%d %H:%M')}. Row: `out/row_mid.toml` (`game_scale.mid_row()`), "
             f"S = {S:.2f}: arm {L:.2f} m, radii {ROW.geom.r_base_m:.4f} -> {ROW.geom.r_tip_m:.5f} m, body {ROW.body.mass:.0f} kg / "
             f"crown {ROW.body.radius:.3f} m, k_bend {ROW.chain.k_bend:.0f} N m/rad, k_dist {ROW.chain.k_dist:.3g} N/m, g_eff "
             f"{ROW.chain.g_eff:.3f} m/s^2, {ROW.chain.n_sub} x {ROW.chain.n_pass} at {1 / DT:.0f} Hz. 3-D nodes, floor z = 0, "
             "Bishop frame + two signed bends per node.\n",
             f"Wall: {wall:.0f} s total ({t_sim:.0f} s for {len(names)} scenarios on {workers} processes, {t_gif:.0f} s for the GIFs). "
             "Per scenario: " + ", ".join(f"{k} {R[k]['wall_s']:.0f} s" for k in names) + ".\n",
             f"**{len(results) - n_fail}/{len(results)} gates pass.**\n",
             "| group | gate | result | measurement |", "|---|---|---|---|"]
    for g_, name, ok, note in results:
        lines.append(f"| {g_} | {name} | {'PASS' if ok else 'FAIL'} | {note} |")
    lines += ["", "Figures: `fig11_reach.png`, `fig11_crawl.png`, `fig11_swim.png`, `fig11_grab.png`, `fig11_idle.png`, "
              "`fig11_new.png`; GIFs: " + ", ".join(f"`{os.path.basename(p)}`" for p in paths)
              + f"; traces: `{os.path.basename(tr_path)}` ({os.path.getsize(tr_path) / 1e6:.1f} MB).", ""]
    with open(os.path.join(OUT, "report_p5z.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n{len(results) - n_fail}/{len(results)} gates pass in {wall:.0f} s. Report: {os.path.join(OUT, 'report_p5z.md')}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
