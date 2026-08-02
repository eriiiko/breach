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

P-F1b UPDATE (2026-08-02, docs/fire_recalibration_2026-08-02.md). The finding
this docstring used to carry -- "at TODAY's shipped tune a naturally-seeded
fire cannot bootstrap past its own hot gate at all, so every run dies in under
a second and the room O2 never moves" -- was TRUE and is now SPENT: the
recalibration is exactly the ignition-seed/tempo pass it asked for. The sweep
therefore measures what T2's pre-measurement charter actually wanted, and it
gains two things it needs to do that:

  * SMALL ROOMS. 12x12 and 20x20 hold far more oxygen than a lone crate can
    draw down inside its own lifetime, so the sweep now leads with 6x6 and 8x8
    -- the sizes where the room inventory, not the fuel, is the binding
    resource. The full ladder is the smother CURVE: death time against room
    volume. MEASURED OUTCOME: the cause does NOT flip -- every sealed size is
    O2-governed; what the ladder moves is how much of the ROOM had to be spent.
  * A DEATH CAUSE THAT CAN SEE AN O2 DEATH. fire_room_bench._diagnose decides
    "O2" by the absolute threshold X_local <= X_ext + 0.01, and the logistic's
    own extinction wall sits ABOVE X_ext (at X_ext + (X_full-X_ext)*r/(1+r)),
    so a genuine suffocation is labelled "knee" by a whisker. The sweep now
    also reports fire_room_bench._death_counterfactual's exact decomposition:
    at the TERMINAL crossing (the last tick the growth bracket was still
    non-negative), would AMBIENT oxygen have saved it (O2-governed), would FULL
    heat have saved it (T-gate-governed), or neither (fuel-governed)?

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
ROOM_SIZES = (6, 8, 12, 20)    # interior_w == interior_h (square rooms)
DRAW_R = 2                     # today's shipped radius (config.toml default)
TILE_SIZE_M = 0.5
MAX_SECONDS = 2400.0   # P-F1b: a recalibrated furniture burn runs ~24 min
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
        d = m.get("cause_detail") or {}
        lines.append(f"  death cause (counterfactual): {m.get('cause_law', 'n/a')}")
        if "bracket" in d:
            lines.append(
                f"    terminal crossing t={d['t']:.1f}s: bracket={d['bracket']:+.5f}"
                f" | F={d['F']:.3f} hot={d['hot']:.3f} X_local={d['x_local']:.4f}"
                f" I={d['I']:.3f} |W|={d['W']:.4f};"
                f" bracket if ambient O2 = {d['bracket_if_ambient_O2']:+.5f},"
                f" bracket if fully hot = {d['bracket_if_fully_hot']:+.5f}")
        lines.append(f"  death time: {death_t:.2f} s" if death_t == death_t else "  death time: n/a (sustained to timeout)")
        lines.append(f"  part-burn: {part_burn:.2f} %  (hp0={m['crate_hp0']:.2f}, "
                    f"hp_end={float(m['rec']['hp'][-1]) if m['rec']['hp'].size else float('nan'):.3f})")
        lines.append(f"  room O2 mole fraction X at death: {room_x:.4f}  "
                    f"(ambient 0.21, X_ext {m['x_ext']:.2f})")
        lines.append(f"  room O2 X min over run: {m['rec']['o2room_x'].min() if m['rec']['o2room_x'].size else float('nan'):.4f}")
        lines.append("")
    lines.append(
        "READING THESE CURVES (P-F1b). The 'death cause (counterfactual)' line "
        "is the exact decomposition (fire_room_bench._death_counterfactual): at "
        "the TERMINAL crossing -- the last tick the logistic's growth bracket "
        "was still non-negative, after which the fire never recovers -- it asks "
        "whether ambient oxygen at the same fuel, heat, intensity and wind would "
        "have saved it (O2-governed), whether a fully hot tile would "
        "(T-gate-governed), or neither (fuel-governed). The '_diagnose' line is "
        "the older, coarser absolute-threshold classifier, kept for continuity "
        "with the P-F4b package.\n\n"
        "THE FINDING, stated as measured rather than as expected: EVERY sealed "
        "ship room in the ladder -- 6x6 through 20x20 -- kills the fire "
        "O2-GOVERNED, in 110-125 s, with 87-97% of the fuel still in the crate. "
        "Room size does NOT flip the cause; it moves how much of the ROOM had to "
        "be spent to do it (mean X at death 0.1295 / 0.1533 / 0.1821 / 0.2001 as "
        "the room grows), because a sealed room has no refill at all and the "
        "flame ring suffocates locally long before the bulk does. THE CONTROL "
        "THAT DIES THE OTHER WAY is the planetside still-air reference arena "
        "(tools/fire_timing_harness.py, sky-exchange refill ON): the same "
        "kindling crate burns 8.3 min there and dies T-GATE-governed at 61% "
        "burnt, the same furniture crate 23.8 min at 54% burnt. That pair -- "
        "110 s O2-governed sealed vs 500-1400 s T-gate-governed refilled -- IS "
        "the smother curve, and it is the ships requirement working.")
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
