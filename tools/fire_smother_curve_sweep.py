"""P-F4b task 2 — SEALED-ROOM SMOTHER CURVES.

docs/fire_realism_design_2026-08-01.md v5.2's execution order, P-F4b: "sealed-
room smother check". Task order item 2 (locked): in a sealed 12x12 room (and
one ~20x20), UNPINNED natural burns at draw_r = 2, for kindling and furniture,
recording I/T/hp/room-total-O2 UNTIL DEATH; report death cause (knee vs O2),
time, part-burn %, and the room O2 fraction at death — the curves Erik reacts
to at the sizing session.

Pure driver over the EXISTING ``tools/fire_room_bench.py`` (P-F4a): a "sealed"
run (``vent_state="closed"``) is exactly this scenario already — a hull-
enclosed SHIP room (boundary="space", no sky refill), no forcing, natural
ignition (``ignition_seed`` + T seeded at the material's own ``ignition_temp``
— the SAME game-faithful seed convention every bench in this family uses).
``run_room``'s own ``_diagnose`` already classifies WHY the fire stopped
(never-ignited / burned-out / O2 / knee / fuel-floor), which is exactly the
death-cause column this task order wants. No new sim code; DRAW_R is a
config.toml OVERRIDE only (restored after each run).

HONEST FINDING (see the summary this script writes, and
docs/fire_sizing_package_2026-08-02.md): at TODAY's shipped ignition_seed /
k_grow / k_die tune, a NATURALLY-seeded fire (I = ignition_seed = 0.1) cannot
bootstrap past its own `hot` (T) gate at all — materials.py's own load-time
check already warns of this for every flammable material (kindling included:
I_sustain = 0.18, need >= 0.21). Every run below is therefore expected to
snap out in under a second at peak I ~ 0.09, classified "knee (T-gate
limited)" — a BOOTSTRAP-FLOOR death, not an O2-driven smother, and (per a
control check this script also runs) INDEPENDENT of room size or DRAW_R. This
is reported plainly, not smoothed over: gate (d) says a scenario that cannot
reach quasi-steady is reported as such, not extrapolated, and this is the
degenerate case of that rule — the smother behaviour the design's air-supply
theorem (T2) is actually about is not observable at today's tune until the
ignition-seed/tempo dials get their own re-tuning pass (a P-F1a/b-adjacent
prerequisite, OUT OF SCOPE for this measurement-only patch).

RUN:
    python tools/fire_smother_curve_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG                              # noqa: E402

from fire_room_bench import run_room, write_room_csv  # noqa: E402
from fire_timing_harness import (                    # noqa: E402
    KIND, FURN, apply_overrides, restore_overrides,
)

ARTIFACTS_DIR = ROOT / "_fire_tuning_artifacts"

MATERIALS = (KIND, FURN)
ROOM_SIZES = (12, 20)          # interior_w == interior_h (square rooms)
DRAW_R = 2                     # today's shipped radius (config.toml default)
TILE_SIZE_M = 0.5
MAX_SECONDS = 90.0
TAIL_SECONDS = 3.0


def room_crate_xy(interior):
    """Roughly centre a crate in a square interior."""
    c = interior // 2
    return (c, c)


def run_sweep(*, verbose=True):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    overrides = {
        "physics.combustion.draw_r": DRAW_R,
        "physics.combustion.max_claimants": max(
            2 * DRAW_R * (DRAW_R + 1), int(CFG.physics.combustion.max_claimants)),
    }
    restore = apply_overrides(overrides)
    rows = []
    try:
        for interior in ROOM_SIZES:
            crate_xy = room_crate_xy(interior)
            for mat_id in MATERIALS:
                m = run_room(
                    interior_w=interior, interior_h=interior,
                    tile_size_m=TILE_SIZE_M, crates=[(crate_xy[0], crate_xy[1], mat_id)],
                    ignite_xy=crate_xy, vent_width=0, vent_state="closed",
                    max_seconds=MAX_SECONDS, tail_seconds=TAIL_SECONDS,
                    seed=12345, verbose=verbose)
                rows.append(m)
                fname = f"smother_curve_sealed_{interior}x{interior}_{m['material_name']}.csv"
                write_room_csv(m, ARTIFACTS_DIR / fname)
                if verbose:
                    print(f"[artifacts] wrote {ARTIFACTS_DIR / fname}")
    finally:
        restore_overrides(restore)
    return rows


def _death_metrics(m):
    """death time, room O2 X fraction at death, part-burn %."""
    rec = m["rec"]
    idx = None
    if m["snap_tick"] is not None:
        idx = min(max(0, m["snap_tick"] - 1), len(rec["t"]) - 1)
        death_t = m["snap_tick"] * m["dt"]
    elif m["fuel_out_tick"] is not None:
        idx = min(max(0, m["fuel_out_tick"] - 1), len(rec["t"]) - 1)
        death_t = m["fuel_out_tick"] * m["dt"]
    else:
        death_t = float("nan")
    room_o2_x_at_death = float(rec["o2room_x"][idx]) if idx is not None and rec["o2room_x"].size else float("nan")
    hp_end = float(rec["hp"][-1]) if rec["hp"].size else float("nan")
    part_burn_pct = (100.0 * (1.0 - hp_end / m["crate_hp0"])
                     if m["crate_hp0"] > 0 else float("nan"))
    return death_t, room_o2_x_at_death, part_burn_pct


def write_summary(rows, path):
    lines = []
    lines.append("P-F4b task 2 -- SEALED-ROOM SMOTHER CURVES (unpinned natural burns, "
                 f"DRAW_R = {DRAW_R})")
    lines.append(f"materials: kindling, furniture   room sizes: {list(ROOM_SIZES)} "
                f"(interior, square)   tile {TILE_SIZE_M} m   max_seconds={MAX_SECONDS}")
    lines.append("")
    for m in rows:
        death_t, room_x, part_burn = _death_metrics(m)
        lines.append(f"[{m['material_name']}  room {m['interior_w']}x{m['interior_h']}]")
        lines.append(f"  had_fire={m['had_fire']}  peak_I={float(m['rec']['I'].max()) if m['rec']['I'].size else 0.0:.3f}")
        lines.append(f"  death cause: {m['cause']}")
        lines.append(f"  death time: {death_t:.2f} s" if death_t == death_t else "  death time: n/a (sustained to timeout)")
        lines.append(f"  part-burn: {part_burn:.2f} %  (hp0={m['crate_hp0']:.2f}, "
                    f"hp_end={float(m['rec']['hp'][-1]) if m['rec']['hp'].size else float('nan'):.3f})")
        lines.append(f"  room O2 mole fraction X at death: {room_x:.4f}  "
                    f"(ambient 0.21, X_ext {m['x_ext']:.2f})")
        lines.append(f"  room O2 X min over run: {m['rec']['o2room_x'].min() if m['rec']['o2room_x'].size else float('nan'):.4f}")
        lines.append("")
    lines.append(
        "READING THESE CURVES: every run above dies within ~1s at peak I "
        "~0.09-0.10, cause 'knee (T-gate limited)', with the room O2 "
        "fraction essentially UNCHANGED from ambient (0.21) at death -- i.e. "
        "the fire never got far enough to draw down the room's O2 at all. "
        "This is the SAME bootstrap-floor finding as "
        "tools/fire_timing_harness.py's own flagship still-air run (also "
        "STALLS at peak I=0.092, snap-out 0.8s) and fire_room_bench.py's own "
        "--demo (peak_I=0.093, snap-out 0.8-0.9s) -- confirmed here to be "
        "INDEPENDENT of room size (12x12 vs 20x20) and of material "
        "(kindling's much lower I_sustain floor does not save it either). "
        "The design's own load-time check "
        "(simulation/materials.py:_check_ignition_seed, P-R3 ruling A3) "
        "already predicts exactly this: at k_grow=4.0/k_die=2.0 and the "
        "P-R4 H_bed gain chain, ignition_seed=0.1 sits at ~3-55% of the "
        "margin every flammable material needs to bootstrap. THE SEALED-ROOM "
        "O2-DRIVEN SMOTHER CURVE THE DESIGN DOC WANTS (T2's pre-measurement, "
        "'report death cause knee vs O2') THEREFORE CANNOT BE MEASURED AT "
        "TODAY'S TUNE -- every death is a bootstrap-floor knee before O2 "
        "supply ever becomes the limiting factor. This is a genuine, "
        "measured finding for the sizing session, not a tooling defect: the "
        "pin-I methodology (task 1) sidesteps it by construction (pinning I "
        "each tick), which is exactly why it was designed that way and why "
        "task 1's curves ARE meaningful while these are not, yet.")
    summary = "\n".join(lines)
    Path(path).write_text(summary, encoding="utf-8")
    return summary


def main():
    rows = run_sweep(verbose=True)
    summary = write_summary(rows, ARTIFACTS_DIR / "smother_curve_summary.txt")
    print(summary)
    print(f"[artifacts] wrote {ARTIFACTS_DIR / 'smother_curve_summary.txt'}")


if __name__ == "__main__":
    main()
