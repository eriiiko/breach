"""All-systems scenario harness — the playground put through fire, blast,
water, and decompression in one deterministic run (Erik's spec, 2026-08-27).

Two water tanks, both expected to keep every drop:
  * AQUARIUM — a NEW glass box built in-memory via ``seal_tiles`` (the
    canonical topology primitive) on open arena floor, then filled 0.3 m.
    Glass transmits 60% of wave energy BY DESIGN (config [materials.glass]:
    transmit = 1 - reflect - absorb), so waves inside it are expected —
    water leaving it is not. Erik plans glass aquariums as a real level
    element; this is their first physics test.
  * PRESSURE LAB — the sealed hull room (R5 of the #54 harness), filled
    0.3 m. Hull transmits 0%: it doubles as a #54 false-heating probe.

Script (24 tps): t=2 s ignite the mid-arena crate stack (the ONLY declared
heat source) · t=8 s + t=9 s two frag_standard blasts mid-arena · t=20 s
breach_focus on the NORTH hull — the main hall vents to space and the fire
should suffocate · end t=45 s.

Properties reported PASS / FAIL / KNOWN-FAIL(#issue) — this harness is the
wide-net instrument for #54 (undeclared heat), #10 (water conservation),
#7 (fires must die in vacuum), and the wave-transmission split (steel/hull
enclosures must be wave-silent; wood 10% / glass 60% pass-through is
design, config.toml [materials.*] wave_reflect/wave_absorb). Vents are
stripped in-memory like the hot-plate bench so #48 stays out of frame.

HARNESS, not a pytest gate (``_`` prefix): prints the report, exits 0.
Assertions get promoted to real gates one by one as the issues close.

Run:
    conda run -n data python tests/_scenario_all_systems_bench.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import materials, water_fixed  # noqa: E402
from simulation.payloads import execute_payload, ignite_ring  # noqa: E402

TPS = 24
END_TICK = 45 * TPS

# --- Playground geometry (70x100; ASCII survey 2026-08-27). Region slices
# follow the #54 hot-plate bench convention (row, col).
LAB_IN = np.s_[27:42, 61:77]      # pressure-lab interior — #54 bench R5 verbatim
LAB_BOX = (26, 42, 60, 77)        # inclusive wall box (r0, r1, c0, c1)
AQ_BOX = (50, 58, 24, 32)         # aquarium wall box, built on open floor
AQ_IN = np.s_[51:58, 25:32]
BUNKER = np.s_[27:42, 83:96]      # steel bunker — #54 bench R6 verbatim
PEN = np.s_[49:66, 83:96]         # zombie-pen — #54 bench R8 verbatim
GALLERY = np.s_[5:20, 84:96]      # glass gallery interior (probes are
                                  # open-tile masked, so edge slack is safe)
BASIN = np.s_[49:66, 61:77]       # pool-basin interior (steel)
ARENA = np.s_[3:67, 3:58]         # the main hall
CRATE = (26, 41)                  # fff crate stack — the declared fire site
CRATE_ZONE = np.s_[22:35, 36:47]  # heat declared legitimate around the fire
BLAST1, BLAST1_TICK = (35, 30), 8 * TPS
BLAST2, BLAST2_TICK = (35, 34), 9 * TPS
BREACH, BREACH_TICK = (2, 30), 20 * TPS   # north hull, main hall to space
IGNITE_TICK = 2 * TPS
WATER_M = 0.3
WAVE_WATCH = range(BLAST1_TICK - 12, BLAST2_TICK + 3 * TPS)  # blast window
EPS_WAVE_ATM = 0.01               # steel/hull interior peak above this = leak
EPS_DT_DEG = 5.0                  # undeclared-heat tolerance (game-deg)
FIRE_EPS_Q = int(0.02 * 65536)    # "any fire" threshold on the Q16 field


def box_wall_mask(shape, box):
    r0, r1, c0, c1 = box
    m = np.zeros(shape, dtype=bool)
    m[r0:r1 + 1, c0:c1 + 1] = True
    m[r0 + 1:r1, c0 + 1:c1] = False
    return m


def main() -> None:
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    lvl = replace(lvl, entities=[e for e in lvl.entities
                                 if e.class_name not in ("vent", "duct")])
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    shape = g.material.shape

    # --- Setup: build the aquarium, fill both tanks -----------------------
    r0, r1, c0, c1 = AQ_BOX
    ring = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)]
    g.seal_tiles(ring, materials.MAT_GLASS)
    for sl in (AQ_IN, LAB_IN):
        if g.solid[sl].any():
            raise RuntimeError(f"fill region {sl} touches solid tiles — "
                               f"geometry drifted; re-survey the map")
    depth_q = water_fixed.quantize_scalar(WATER_M)
    g.water_depth[AQ_IN] = depth_q
    g.water_depth[LAB_IN] = depth_q
    open0 = ~g.solid.copy()   # probes read open-at-t0 tiles only — wall
                              # cells never pollute T/P/wave region stats

    lab_walls = box_wall_mask(shape, LAB_BOX)
    aq_walls = box_wall_mask(shape, AQ_BOX)
    wall_hp0 = g.wall_hp.copy()
    mat0 = g.material.copy()
    T0 = g.temperature.astype(np.int64)
    water_total0 = int(g.water_depth.sum(dtype=np.int64))
    aq0 = int(g.water_depth[AQ_IN].sum(dtype=np.int64))
    lab0 = int(g.water_depth[LAB_IN].sum(dtype=np.int64))

    tank_mask = np.zeros(shape, dtype=bool)
    tank_mask[AQ_IN] = True
    tank_mask[LAB_IN] = True

    payloads = sim.weapons_tables.payloads.by_name
    events = getattr(sim, "tick_events", None)

    wave_peak = {"bunker": 0.0, "lab": 0.0, "aq": 0.0, "pen": 0.0}
    fire_last_tick = -1
    fire_at_breach = False
    first_leak = None      # (tick, tank, stray bounding box) — leak forensics
    pre_breach = None      # water books snapshot the tick before the breach

    def dT(sl):
        d = (g.temperature.astype(np.int64) - T0)[sl]
        return float(d[open0[sl]].mean()) / 65536.0

    def pmean(sl):
        return float(g.atmosphere[sl][open0[sl]].mean()) / 65536.0

    def water_row(t):
        wt = int(g.water_depth.sum(dtype=np.int64))
        aq = int(g.water_depth[AQ_IN].sum(dtype=np.int64))
        lab = int(g.water_depth[LAB_IN].sum(dtype=np.int64))
        stray = wt - int(g.water_depth[tank_mask].sum(dtype=np.int64))
        print(f"{t/TPS:5.0f}s water aq={aq/aq0*100:6.2f}% lab={lab/lab0*100:6.2f}% "
              f"stray_q={stray} | dT bunker={dT(BUNKER):+6.1f} pen={dT(PEN):+6.1f} "
              f"gallery={dT(GALLERY):+6.1f} basin={dT(BASIN):+6.1f} "
              f"arena={dT(ARENA):+6.1f} aq={dT(AQ_IN):+6.1f} | "
              f"P arena={pmean(ARENA):5.3f} lab={pmean(LAB_IN):5.3f} "
              f"aq={pmean(AQ_IN):5.3f} "
              f"| fire={'Y' if fire_last_tick == t else 'n'}")

    print(f"seeded: aquarium {aq0} q-units, lab {lab0} q-units, "
          f"grid total {water_total0}")
    water_row(0)

    for t in range(1, END_TICK + 1):
        if t == IGNITE_TICK:
            ignite_ring(g, sim.edit_queue, *CRATE, 2.5, 1.0)
        if t == BLAST1_TICK:
            execute_payload(g, sim.edit_queue, sim.units, *BLAST1,
                            payloads["frag_standard"], sim.rng, events=events)
        if t == BLAST2_TICK:
            execute_payload(g, sim.edit_queue, sim.units, *BLAST2,
                            payloads["frag_standard"], sim.rng, events=events)
        if t == BREACH_TICK:
            fire_at_breach = fire_last_tick >= t - TPS
            execute_payload(g, sim.edit_queue, sim.units, *BREACH,
                            payloads["breach_focus"], sim.rng, events=events,
                            kind="door_explosive")
        sim.set_paused(False)
        sim.step()

        if t in WAVE_WATCH:
            w = np.abs(g.wave_p) / 65536.0
            for key, sl in (("bunker", BUNKER), ("lab", LAB_IN),
                            ("aq", AQ_IN), ("pen", PEN)):
                wave_peak[key] = max(wave_peak[key],
                                     float(w[sl][open0[sl]].max()))
        if int(g.fire.max()) > FIRE_EPS_Q:
            fire_last_tick = t
        if first_leak is None:
            aq_t = int(g.water_depth[AQ_IN].sum(dtype=np.int64))
            lab_t = int(g.water_depth[LAB_IN].sum(dtype=np.int64))
            if aq_t != aq0 or lab_t != lab0:
                stray_yx = np.argwhere((g.water_depth > 0) & ~tank_mask)
                bbox = (stray_yx.min(0).tolist(), stray_yx.max(0).tolist()) \
                    if len(stray_yx) else None
                first_leak = (t, "aq" if aq_t != aq0 else "lab", bbox)
                print(f"  !! first water anomaly at tick {t} ({t/TPS:.1f}s): "
                      f"{first_leak[1]} changed; stray water bbox "
                      f"(rmin,cmin)-(rmax,cmax) = {bbox}")
        if t == BREACH_TICK - 1:
            pre_breach = {
                "aq": int(g.water_depth[AQ_IN].sum(dtype=np.int64)),
                "lab": int(g.water_depth[LAB_IN].sum(dtype=np.int64)),
                "stray": int(g.water_depth.sum(dtype=np.int64)) -
                         int(g.water_depth[tank_mask].sum(dtype=np.int64)),
            }
        if t % (6 * TPS) == 0:
            water_row(t)

    # --- Property report --------------------------------------------------
    wt = int(g.water_depth.sum(dtype=np.int64))
    aq = int(g.water_depth[AQ_IN].sum(dtype=np.int64))
    lab = int(g.water_depth[LAB_IN].sum(dtype=np.int64))
    stray = wt - int(g.water_depth[tank_mask].sum(dtype=np.int64))
    quiet = {"bunker": dT(BUNKER), "pen": dT(PEN),
             "gallery": dT(GALLERY), "basin": dT(BASIN)}

    def verdict(ok, known_fail=None):
        if ok:
            return "PASS"
        return f"KNOWN-FAIL {known_fail}" if known_fail else "FAIL"

    # Wall forensics: every solid tile whose material changed, grid-wide,
    # classified — the breach hole is declared; glass->air is a design
    # burst (config burst_threshold 1.0 atm differential); anything else
    # is undeclared structural damage and fails P6.
    changed = np.argwhere((g.material != mat0) &
                          ((mat0 > 0) | (g.material > 0)))
    names = materials.MATERIAL_NAMES
    undeclared = []
    n_glass_burst = 0
    print("\n--- wall/material changes (grid-wide) ---")
    for y, x in changed:
        old, new = int(mat0[y, x]), int(g.material[y, x])
        if (old == materials.MAT_HULL and new == materials.MAT_AIR and
                abs(int(y) - BREACH[0]) <= 3 and abs(int(x) - BREACH[1]) <= 3):
            tag = "DECLARED breach hole"
        elif old == materials.MAT_GLASS and new == materials.MAT_AIR:
            tag = "design burst (glass)"
            n_glass_burst += 1
        else:
            tag = "UNDECLARED"
            undeclared.append((int(y), int(x)))
        print(f"  ({y:2d},{x:2d}) {names[old]:>8} -> {names[new]:<6} {tag}")

    print("\n=== PROPERTY REPORT " + "=" * 47)
    p1a = (pre_breach["aq"] == aq0 and pre_breach["lab"] == lab0 and
           pre_breach["stray"] == 0)
    print(f"P1a water conservation through fire+blasts (#10): {verdict(p1a)} "
          f"— pre-breach aq {pre_breach['aq']}/{aq0}, "
          f"lab {pre_breach['lab']}/{lab0}, stray {pre_breach['stray']}; "
          f"first anomaly: {first_leak}")
    p1 = (aq == aq0 and lab == lab0 and stray == 0)
    print(f"P1b water intact end-to-end: {verdict(p1)} — "
          f"aq {aq}/{aq0}, lab {lab}/{lab0}, stray {stray} q-units "
          f"(post-breach boil is design if the tank lost its air by burst)")
    p2 = wave_peak["bunker"] < EPS_WAVE_ATM
    print(f"P2 steel silence (transmit=0 by config): {verdict(p2)} — peak "
          f"|wave_p| atm: bunker {wave_peak['bunker']:.4f} (budget "
          f"{EPS_WAVE_ATM}) | design-transmitting for scale: lab(hull+apertures) "
          f"{wave_peak['lab']:.4f}, aquarium(glass) {wave_peak['aq']:.4f}, "
          f"pen(glass) {wave_peak['pen']:.4f}")
    p3 = all(abs(v) < EPS_DT_DEG for v in quiet.values())
    print(f"P3 no undeclared heat: {verdict(p3, '#54')} — dT " +
          ", ".join(f"{k} {v:+.1f}" for k, v in quiet.items()))
    vented = pmean(ARENA) < 0.1
    fire_died = fire_last_tick < BREACH_TICK + 10 * TPS
    print(f"P4 breach vents the hall: {verdict(vented)} — arena P "
          f"{pmean(ARENA):.3f} atm")
    print(f"P4b vacuum kills the fire: {verdict(fire_died, '#7')} — last "
          f"fire tick {fire_last_tick} (breach {BREACH_TICK}); burning at "
          f"breach: {fire_at_breach} (Erik: crate should hold >45 s — "
          f"tuning sanity)")
    lab_burst = any(mat0[y, x] == materials.MAT_GLASS
                    for y, x in np.argwhere(lab_walls & (g.material != mat0)))
    aq_burst = any(mat0[y, x] == materials.MAT_GLASS
                   for y, x in np.argwhere(aq_walls & (g.material != mat0)))
    for name, sl, burst in (("lab", LAB_IN, lab_burst),
                            ("aq", AQ_IN, aq_burst)):
        if burst:
            print(f"P5 {name} holds air post-breach: BURST(design) — its "
                  f"glass gave way at the 1.0 atm decompression "
                  f"differential (P now {pmean(sl):.3f} atm); water then "
                  f"flash-boiled. Aquariums that must survive a hull "
                  f"breach need a tougher pane row ([materials] decision)")
        else:
            ok = abs(pmean(sl) - 1.0) < 0.2
            print(f"P5 {name} holds air post-breach: {verdict(ok)} — "
                  f"P {pmean(sl):.3f} atm")
    p6 = not undeclared
    chipped = int(((g.wall_hp != wall_hp0) & (g.material == mat0) &
                   (lab_walls | aq_walls)).sum())
    print(f"P6 no undeclared structural damage: {verdict(p6)} — "
          f"{len(changed)} changed tiles: breach hole + {n_glass_burst} "
          f"glass design-bursts + {len(undeclared)} undeclared "
          f"{undeclared if undeclared else ''}; tank-wall tiles chipped "
          f"(hp down, still standing): {chipped}")


if __name__ == "__main__":
    main()
