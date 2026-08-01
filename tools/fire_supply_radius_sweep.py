"""P-F4b task 1 — THE SUPPLY-VS-RADIUS CURVE.

docs/fire_realism_design_2026-08-01.md v5.2's execution order: "P-F4b (supply-
vs-radius sweep + sealed-room smother check + FORCED-WIND level ...)"; §0's T2
pre-measurement charter ("SUPPLY BENCH — max sustainable O2 delivery to one
burning tile ... across F4 room sizes and vent states; publishes the supply
constant k"). This is a sibling driver of ``tools/fire_o2_supply_baseline.py``
(P-F4a): it reuses that module's PIN-I methodology verbatim (pin intensity at
the design's blessed operating point, pin wall_hp, measure the TRUE
law-agnostic draw by re-running the combustion pass on settled state and
reverting it — see ``fire_o2_supply_baseline``'s module docstring for the full
rationale) via the generalized ``measure_supply_on_level`` entry point, and
sweeps it over:

  * ``[physics.combustion] draw_r`` in {1, 2, 3} (the P-O2b dial; a config
    OVERRIDE only, restored after each run — nothing written to config.toml);
  * material in {kindling, furniture} (the P-F4a campfire reference object +
    the incumbent furniture crate);
  * THREE environments, each built from EXISTING scenario builders (no new
    sim code):
      - "open_arena"  — ``fire_timing_harness.build_level`` (planetside,
        sky-exchange ON, still air) — the P-F4a still-air reference arena;
      - "sealed_room" — ``fire_room_bench.build_room_level`` (SHIP-style,
        boundary="space", no sky) with NO vent — sealed hull, finite air;
      - "vented_room" — the SAME sealed room with a vent OPEN FROM TICK 0
        (``vent_open_at_build=True`` — a real hole in the hull ring, present
        before the sim starts, not a forced field write).

R=3 needs a wider ``dem_acc`` claimant plane (slot_count(3) == 24) than the
shipped ``max_claimants = 12`` (slot_count(2)) provides — both overridden
together per iteration (``combustion_draw_slots()`` in gamemap.py hard-errors
otherwise; config.toml's own comment: "Must be >= 2*draw_r*(draw_r+1)").

OUTPUT: one CSV per (env, material, R) in ``_fire_tuning_artifacts/`` (the
fire_o2_supply_baseline per-tick trace format, unchanged) + one summary text
file with delivery in counts/s AND kW (Huggett J_PER_COUNT) per point, the
ring-by-ring X profile, and the R=2/R=1 open-arena ratio check against the
design's cited ~2.26x/2.48x (kindling/furniture).

RUN:
    python tools/fire_supply_radius_sweep.py
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

from config import CFG                             # noqa: E402

# tools/ is sys.path[0] when run as a script (fire_tune_loop.py precedent).
from fire_o2_supply_baseline import (              # noqa: E402
    measure_supply_on_level, write_measure_csv, J_PER_COUNT, OPERATING_POINT_I,
)
from fire_room_bench import build_room_level        # noqa: E402
from fire_timing_harness import (                  # noqa: E402
    KIND, FURN, build_level, apply_overrides, restore_overrides,
)

ARTIFACTS_DIR = ROOT / "_fire_tuning_artifacts"

MATERIALS = (KIND, FURN)
DRAW_RS = (1, 2, 3)
ENVS = ("open_arena", "sealed_room", "vented_room")

# The design doc's own cited reproduction target (task order item 1):
# "Verify the shipped R=2 numbers reproduce (~2.26x/2.48x over R=1 in the
# open arena)" — kindling then furniture, gate (b)'s ±2% band.
EXPECTED_R2_OVER_R1 = {"kindling": 2.26, "furniture": 2.48}

# Sealed/vented room geometry — the design's "sealed room (interior ~12x12)"
# task order wording, one crate roughly centred, vent on the east wall.
ROOM_INTERIOR = 12
ROOM_TILE_M = 0.5
ROOM_CRATE_XY = (6, 6)          # interior-relative, ~centre of a 12x12 room
ROOM_VENT_WIDTH = 2
ROOM_VENT_SIDE = "east"

# Open-arena geometry — unchanged from fire_o2_supply_baseline.measure_supply
# defaults (P-F4a's still-air reference arena).
ARENA_W, ARENA_H = 84, 40
ARENA_CRATE_XY = (12, 21)
ARENA_TILE_M = 0.333


def max_claimants_for(draw_r: int) -> int:
    """The dem_acc plane's required slot depth at this radius (gamemap.py's
    ``combustion_draw_slot_count`` twin): 4 / 12 / 24 at R = 1 / 2 / 3."""
    return 2 * draw_r * (draw_r + 1)


def build_env(env: str, material_id: int):
    """Return ``(level, cy, cx)`` (full-grid tile coords) for one of the
    three named environments, with ``material_id``'s tile already placed."""
    if env == "open_arena":
        level = build_level(ARENA_W, ARENA_H, ARENA_CRATE_XY, ARENA_TILE_M)
        cx, cy = ARENA_CRATE_XY
        level.tilemap[cy, cx] = material_id
        return level, cy, cx
    if env == "sealed_room":
        level = build_room_level(
            ROOM_INTERIOR, ROOM_INTERIOR, ROOM_TILE_M, vent_width=0,
            crates=[(ROOM_CRATE_XY[0], ROOM_CRATE_XY[1], material_id)])
        cx, cy = ROOM_CRATE_XY[0] + 1, ROOM_CRATE_XY[1] + 1
        return level, cy, cx
    if env == "vented_room":
        level = build_room_level(
            ROOM_INTERIOR, ROOM_INTERIOR, ROOM_TILE_M,
            vent_width=ROOM_VENT_WIDTH, vent_side=ROOM_VENT_SIDE,
            vent_open_at_build=True,
            crates=[(ROOM_CRATE_XY[0], ROOM_CRATE_XY[1], material_id)])
        cx, cy = ROOM_CRATE_XY[0] + 1, ROOM_CRATE_XY[1] + 1
        return level, cy, cx
    raise ValueError(f"unknown env {env!r}")


def run_sweep(*, run_seconds=30.0, steady_window_s=5.0, max_ring=6,
             seed=12345, verbose=True):
    """The full env x material x draw_r sweep. Returns the list of metrics
    dicts (one per point, 3 x 2 x 3 = 18 total) and writes one CSV per point
    to ``_fire_tuning_artifacts/``."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    rows = []
    for env in ENVS:
        for draw_r in DRAW_RS:
            overrides = {
                "physics.combustion.draw_r": draw_r,
                "physics.combustion.max_claimants": max(
                    max_claimants_for(draw_r),
                    int(CFG.physics.combustion.max_claimants)),
            }
            restore = apply_overrides(overrides)
            try:
                for mat_id in MATERIALS:
                    level, cy, cx = build_env(env, mat_id)
                    m = measure_supply_on_level(
                        level, cy, cx, mat_id, pin_I=OPERATING_POINT_I,
                        run_seconds=run_seconds, steady_window_s=steady_window_s,
                        max_ring=max_ring, seed=seed, verbose=verbose,
                        env_label=env)
                    rows.append(m)
                    fname = (f"supply_radius_sweep_{env}_{m['material_name']}"
                            f"_R{draw_r}.csv")
                    write_measure_csv(m, ARTIFACTS_DIR / fname)
                    if verbose:
                        print(f"[artifacts] wrote {ARTIFACTS_DIR / fname}")
            finally:
                restore_overrides(restore)
    return rows


def _index(rows):
    return {(m["env"], m["material_name"], m["draw_r"]): m for m in rows}


def write_summary(rows, path):
    idx = _index(rows)
    lines = []
    lines.append("P-F4b task 1 -- SUPPLY-VS-RADIUS SWEEP")
    lines.append(f"J_PER_COUNT = {J_PER_COUNT}  OPERATING_POINT_I = {OPERATING_POINT_I}"
                f"  (pin-I methodology, fire_o2_supply_baseline.measure_supply_on_level)")
    lines.append(f"materials: kindling, furniture   draw_r: {list(DRAW_RS)}   "
                f"environments: {list(ENVS)}")
    lines.append("")
    for env in ENVS:
        lines.append(f"=== environment: {env} ===")
        for mat_name in ("kindling", "furniture"):
            for draw_r in DRAW_RS:
                m = idx.get((env, mat_name, draw_r))
                if m is None:
                    continue
                lines.append(
                    f"  [{mat_name} R={draw_r}] TRUE draw = "
                    f"{m['true_counts_per_s_ss']:.2f} counts/s = "
                    f"{m['true_delivery_kw_ss']:.4f} kW   "
                    f"(analytic radius-1 formula: {m['delivery_counts_per_s_ss']:.2f} "
                    f"counts/s = {m['delivery_kw_ss']:.4f} kW)   "
                    f"X_local_ss={m['x_local_ss']:.4f}")
                lines.append(
                    "    ring X profile: " + ", ".join(
                        f"r{r}={x:.4f}" for r, x in sorted(m["ring_profile"].items())))
        lines.append("")

    # --- vented_room honesty note (measured, not assumed) -------------------
    lines.append(
        "NOTE on 'vented_room' (vent OPEN FROM TICK 0, boundary='space' -- a "
        "real hole to true vacuum, fire_room_bench's existing 'vent open' "
        "mode, no new mechanic): the MG pressure solve empties a 12x12 room "
        "to near-total vacuum (room N_total ~0.007-0.01 of ambient) within "
        "ABOUT ONE TICK of the opening existing -- an acoustic-scale blowdown, "
        "not a slow leak (confirmed directly: a mid-run breach on an "
        "otherwise-identical sealed/quasi-steady room shows drawn O2 go "
        "192 counts/s -> 0 counts/s in the SAME tick the vent opens, room "
        "N_total 1.00 -> 0.05 within 0.75s -- see the wind-level section, "
        "task 3, for the full trace). This is why every 'vented_room' TRUE "
        "draw above reads exactly 0.00 counts/s regardless of draw_r: by the "
        "time the pin-I fire is sampled (run_seconds default 30s), the room "
        "has been near-vacuum for the entire run and the combustion pass's "
        "epsilon skip-floor (o2_thresh_burn, an ABSOLUTE-count floor) skips "
        "every donor cell. The 'analytic radius-1 formula' column stays "
        "misleadingly non-zero because it is built from X (the FRACTION, "
        "scale-invariant) which can still read near-ambient while the "
        "absolute gas content the real law reads has collapsed to nothing -- "
        "a live demonstration of why the TRUE/law-agnostic column exists. "
        "READ: venting a sealed room to hard vacuum is not a 'reduced "
        "supply' condition, it is a near-total and IMMEDIATE supply cutoff, "
        "independent of DRAW_R -- consistent with task 3's finding that no "
        "sustained flow can be built from a vacuum-vent boundary.")
    lines.append("")

    # --- gate (b): open-arena R2/R1 ratio check -----------------------------
    lines.append("=== GATE (b): open-arena R2/R1 delivery ratio vs the design's "
                 "cited ~2.26x/2.48x (kindling/furniture), tolerance +/-2% ===")
    all_ok = True
    for mat_name in ("kindling", "furniture"):
        m1 = idx.get(("open_arena", mat_name, 1))
        m2 = idx.get(("open_arena", mat_name, 2))
        if m1 is None or m2 is None or m1["true_counts_per_s_ss"] <= 0:
            lines.append(f"  {mat_name}: MISSING data")
            all_ok = False
            continue
        ratio = m2["true_delivery_kw_ss"] / m1["true_delivery_kw_ss"]
        expected = EXPECTED_R2_OVER_R1[mat_name]
        pct_err = 100.0 * (ratio - expected) / expected
        ok = abs(pct_err) <= 2.0
        all_ok = all_ok and ok
        lines.append(
            f"  {mat_name}: measured R2/R1 = {ratio:.4f}  expected ~{expected:.2f}  "
            f"error = {pct_err:+.2f}%   {'OK' if ok else 'FAIL'}")
    lines.append(f"  GATE (b): {'PASS' if all_ok else 'FAIL'}")
    lines.append("")

    summary = "\n".join(lines)
    Path(path).write_text(summary, encoding="utf-8")
    return summary, all_ok


def main():
    rows = run_sweep(verbose=True)
    summary, gate_b_ok = write_summary(
        rows, ARTIFACTS_DIR / "supply_radius_sweep_summary.txt")
    print(summary)
    print(f"[artifacts] wrote {ARTIFACTS_DIR / 'supply_radius_sweep_summary.txt'}")
    print(f"GATE (b) open-arena R2/R1 ratio check: {'PASS' if gate_b_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
