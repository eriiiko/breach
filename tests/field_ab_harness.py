"""Field-level A/B determinism harness — PhysicsEngine unification, Patch 0.

The unification's safety argument is "behavior test-identical": every sim field,
every cell, every tick, the refactored path must match the old path. The existing
determinism check (``tests/test_simulation.py:_state_signature``) compares five
whole-grid MEANS — it cancels per-cell sign-flipped errors and re-runs the same
build, so it CANNOT detect the float-reorder desync a glue->C++ port risks
(panel finding, docs/physics_engine_unification_plan.md §1).

This module is the real gate: snapshot every sim field each tick, run two paths on
the same seed+inputs, and assert per-FIELD per-CELL equality (exact under the
/fp:precise 0-ULP plan, or within a stated tolerance for the fallback path).

Usage (Patch 1, same machine — float bit-identity is NOT cross-machine until the
fixed-point migration):

    from field_ab_harness import capture_trajectory, assert_trajectories_match
    base = capture_trajectory()            # pre-refactor code  (git checkout old)
    # ... land a refactor phase, rebuild ...
    new  = capture_trajectory()            # refactored code
    assert_trajectories_match(base, new)   # tol=0.0 == 0-ULP

Run:
    C:/Users/steen/anaconda3/python.exe tests/field_ab_harness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import breach_physics as bp
from level_loader import LevelData
from simulation import Simulation
from simulation.unit import Unit

SEED = 20260615

# Every field the physics writes — the sim state a structural refactor must
# preserve bit-for-bit. `gas` is (N,h,w) and covers `smoke` (a view into
# gas[BLACK_SMOKE]); `dyn_*` are the per-tick stamp_units outputs; material /
# is_vacuum / wall_hp catch any topology-path change. Render-only buffers
# (light_rgb, light_dir, smoke_glow) are intentionally EXCLUDED — they are not
# sim state and may legitimately differ between machines/builds.
SIM_FIELDS = (
    "atmosphere", "wave_p", "wave_v", "wave_source", "wind_x", "wind_y",
    "gas", "fire", "water_depth", "flow_vx", "flow_vy",
    "heat", "temperature", "ripple", "ripple_v",
    "dyn_permeability", "dyn_wave_absorb", "obstacles", "dyn_light_atten",
    "wall_hp", "material", "is_vacuum",
)


def _scenario_level() -> LevelData:
    """A 16x16 hull-walled room, border on the map edge (so a border breach
    exposes vacuum), interior air carved out — synthetic, no asset files."""
    h = w = 16
    tm = np.ones((h, w), dtype=np.int32)   # all hull
    tm[1:15, 1:15] = 4                       # carve interior air
    return LevelData(name="ab_harness", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def default_scenario_sim() -> Simulation:
    """The canonical A/B scenario — every solver activated, fully deterministic.

    Seeds smoke + fire + water + a wave pulse over the interior, spawns one static
    marine (so stamp_units stamps a footprint each tick), and opens a hull breach
    (so the sink-pull + venting engage). Returns an unpaused sim ready to step.
    """
    sim = Simulation(_scenario_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    interior = (~g.solid) & (~g.is_vacuum)
    g.smoke[interior] = 0.6          # smoke transport + diffusion (gas[BLACK_SMOKE] view)
    g.fire[8, 8] = 0.8               # fire feedback -> heat deposit -> temperature
    g.fire[8, 9] = 0.5
    g.water_depth[10, 10] = 0.3      # water pipe model + W3 displacement + ripple
    g.water_depth[10, 11] = 0.3
    g.wave_source[4, 4] = 8.0        # explicit wave kick + reflection off the hull
    sim.add_unit(Unit("M1", x=7, y=7, team=0))   # stamp_units footprint
    g.destroy_wall(8, 0)             # hull breach on the map edge -> vacuum (venting)
    sim.set_paused(False)
    return sim


def _snapshot(gmap, fields):
    return {name: np.copy(getattr(gmap, name))
            for name in fields if hasattr(gmap, name)}


def capture_trajectory(make_sim=default_scenario_sim, n_steps=30, fields=SIM_FIELDS):
    """Run ``make_sim()`` for ``n_steps``, returning a per-tick list of field-snapshot
    dicts. Forces unpause each step so a phase/round boundary cannot silently halt
    the trajectory (the round reset itself is deterministic)."""
    sim = make_sim()
    traj = []
    for _ in range(n_steps):
        sim.set_paused(False)
        sim.step()
        traj.append(_snapshot(sim.gmap, fields))
    return traj


def diff_trajectories(a, b, tol=0.0):
    """Per-field per-cell mismatches between two trajectories (empty list == match).

    ``tol == 0.0`` -> exact equality (the /fp:precise 0-ULP gate). ``tol > 0`` ->
    absolute tolerance (the fallback path). Each entry is a human-readable line
    locating the worst-offending cell."""
    if len(a) != len(b):
        return [f"trajectory length {len(a)} != {len(b)}"]
    diffs = []
    for t, (sa, sb) in enumerate(zip(a, b)):
        for k in sorted(set(sa) | set(sb)):
            if k not in sa or k not in sb:
                diffs.append(f"tick {t}: field '{k}' present in only one run")
                continue
            fa, fb = sa[k], sb[k]
            if fa.shape != fb.shape:
                diffs.append(f"tick {t}: '{k}' shape {fa.shape} != {fb.shape}")
                continue
            if tol == 0.0:
                if np.array_equal(fa, fb):
                    continue
            elif np.allclose(fa, fb, rtol=0.0, atol=tol, equal_nan=True):
                continue
            d = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
            idx = tuple(int(i) for i in np.unravel_index(int(np.argmax(d)), d.shape))
            diffs.append(
                f"tick {t}: '{k}' differs — {int((d > tol).sum())} cell(s), "
                f"max|delta|={d.max():.3e} at {idx} (a={fa[idx]!r} b={fb[idx]!r})")
    return diffs


def assert_trajectories_match(a, b, tol=0.0, max_report=10):
    """Raise AssertionError with the first ``max_report`` mismatches if a != b."""
    diffs = diff_trajectories(a, b, tol=tol)
    if diffs:
        head = "\n  ".join(diffs[:max_report])
        more = "" if len(diffs) <= max_report else f"\n  ... +{len(diffs) - max_report} more"
        raise AssertionError(
            f"A/B trajectories differ ({len(diffs)} mismatch group(s)):\n  {head}{more}")


def save_trajectory(traj, path):
    """Persist a trajectory so the OLD path's golden survives a C++ rebuild.

    The unification gate is OLD-build vs NEW-build on the SAME machine; since a
    rebuild swaps the .pyd in-process, capture the golden, pickle it here, rebuild,
    then ``load_trajectory`` + ``assert_trajectories_match`` against a fresh capture.
    (Underscore-prefixed golden files are untracked dev artifacts, never committed.)
    """
    import pickle
    with open(path, "wb") as f:
        pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_trajectory(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    a = capture_trajectory()
    b = capture_trajectory()
    assert_trajectories_match(a, b, tol=0.0)
    nfields = len(a[-1])
    print(f"OK: field-level A/B harness — {len(a)} ticks x {nfields} fields, "
          f"per-cell 0-ULP self-match")
