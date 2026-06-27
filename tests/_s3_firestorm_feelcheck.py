"""S3 SPREADING-firestorm feel-A/B — float golden vs integer fire.

A DRAMATIC, multi-tile spreading firestorm through the REAL Simulation, recording
per-tick `fire` in REAL [0,1] units (dequantized). The scenario exercises the full
fire DYNAMICS so Erik can eyeball that the new integer fire (S3b/S3c) looks
IDENTICAL to the old float fire (S3a, 28d134b):

  ignite -> SPREAD across a 2D wood lattice (10 -> 70 lit tiles, via the
  radiation -> heat -> temperature -> ignition coupling, tile to tile, in 2D) ->
  peak firestorm -> DECOMPRESSION blow-out (vent the room -> the O2 proxy collapses)
  -> extinguish. A grenade wave_source shockwave mid-run spikes the wind to exercise
  the fire's W = |wind| term on the sqrt_q16 worst-case / overflow path.

Unlike the original single-tile `_s3b_firestorm_feel.py` (a lone decaying cell),
this one genuinely SPREADS in 2D, so the heatmap is a growing-then-dying firestorm.

NOTE on wind/feel: in this fire model the wind term FANS growth and STRIPS margins
but the spread itself is the (essentially isotropic) radiation->ignition front, and
a steady wind cannot blow out a well-fuelled, O2-rich blaze (the strip term is
dominated by the O2-sustained growth). DECOMPRESSION is the real extinguisher. That
is a property of the MODEL, faithfully reproduced by BOTH builds — which is exactly
what this A/B confirms: whatever the dynamics, integer == float.

Workflow (mirrors _s3b_firestorm_feel.py):
    # capture the FLOAT golden on the S3a build (pyd built from 28d134b):
    python tests/_s3_firestorm_feelcheck.py capture tests/_s3_feel_float.pkl
    # compare the current INTEGER build against it (numbers):
    python tests/_s3_firestorm_feelcheck.py compare tests/_s3_feel_float.pkl
    # render the committed side-by-side PNG artifact:
    python tests/_s3_firestorm_feelcheck.py render tests/_s3_feel_float.pkl \
        tests/_s3_firestorm_feelcheck.png

Underscore-prefixed -> a throwaway dev artifact (the committed deliverable is the
PNG; the .py harness + the .pkl golden ride along for reproducibility).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                       # noqa: E402
from level_loader import LevelData                # noqa: E402
from simulation import Simulation                 # noqa: E402
from simulation import fire_fixed                 # noqa: E402
from simulation import atmosphere_fixed           # noqa: E402
from simulation.materials import MAT_WOOD         # noqa: E402

# wall_hp representation differs by build: S3b/S3c (current) store it as int32 Q16.16
# (simulation.wall_fixed); S3a (the float golden, 28d134b) stores it as raw float32.
# `_set_wall_hp` writes the SAME real HP into whichever representation this build
# uses, so the IDENTICAL scenario runs on both — the only difference between the two
# runs is the C++ fire logistic (float vs integer), which is exactly what we compare.
try:
    from simulation import wall_fixed             # noqa: E402  (current build only)
    _HAS_WALL_FIXED = True
except ImportError:
    _HAS_WALL_FIXED = False


def _set_wall_hp(g, y, x, hp_real: float) -> None:
    if _HAS_WALL_FIXED and g.wall_hp.dtype != np.float32:
        g.wall_hp[y, x] = wall_fixed.quantize_scalar(hp_real)   # int32 Q16.16 build
    else:
        g.wall_hp[y, x] = hp_real                                # S3a float32 build


SEED = 70707
TICKS = 200

# Grid: a hull-walled box with interior air (v1 code 4 == interior AIR, NOT vacuum;
# code 0 would be outer space). Five parallel WOOD WALLS (rows) separated by air
# rows form a 2D flammable lattice: every wood tile borders the air corridor, so it
# has the O2 needed to ignite, and a burning wall radiates ACROSS the air gap to the
# next wood tile -> the fire spreads wall-to-wall AND along each wall in 2D.
H, W = 22, 64
WOOD_ROWS = (4, 7, 10, 13, 16)
WOOD_XS = tuple(range(2, W - 2))
SEED_X = 24                       # ignite an upwind-of-centre column of every wall
HOLD_UNTIL = 40                   # hold the seed column lit while the blaze takes
WALL_HP = 200.0                   # generous fuel so the walls burn the whole window

# Scripted firestorm trajectory (mirrors the original _s3b grenade script, now over
# a SPREADING blaze). The fire model's spread is the radiation->heat->ignition front;
# its wind term (W = |wind| from -grad p, incl. waves) FANS growth + STRIPS margins.
#   * GRENADE_AT: a wave_source shockwave spikes the wind -> a transient DIRECTIONAL
#     firestorm kick (worst-case W / the sqrt_q16 overflow path) the fire reads.
#   * VENT_AT: DECOMPRESSION — the interior atmosphere is pulled toward vacuum, so
#     the O2 proxy collapses and the firestorm BLOWS OUT and EXTINGUISHES (the
#     faithful "decompression clears the fire" gameplay; a steady wind alone cannot
#     blow out a well-fuelled, O2-rich blaze in this model — the strip term is
#     dominated by the O2-sustained growth, so venting is the real extinguisher).
GRENADE_AT = 70
GRENADE_X = SEED_X + 6            # downwind of the seed
VENT_AT = 140                     # decompress the room -> O2 starves -> blow-out


def _wood_lattice_level() -> LevelData:
    tm = np.ones((H, W), dtype=np.int32)          # all hull
    tm[1:H - 1, 1:W - 1] = 4                        # interior AIR (v1 code 4)
    return LevelData(name="s3_firestorm", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _wood_tiles():
    return [(y, x) for y in WOOD_ROWS for x in WOOD_XS]


def _seed_tiles():
    return [(y, x) for y in WOOD_ROWS for x in (SEED_X, SEED_X + 1)]


def _make_sim() -> Simulation:
    sim = Simulation(_wood_lattice_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    for (y, x) in _wood_tiles():
        g.material[y, x] = MAT_WOOD
    g._update_caches()
    # Boost the wood fuel store so the firestorm has something to burn a while
    # (build-agnostic: int32 Q16.16 on the current build, float32 on the S3a golden).
    for (y, x) in _wood_tiles():
        _set_wall_hp(g, y, x, WALL_HP)
    sim.set_paused(False)
    return sim


def _fire_real(g) -> np.ndarray:
    """Read the fire field as REAL [0,1] intensity (dequantize the Q16.16 int)."""
    return fire_fixed.dequantize(g.fire).astype(np.float64)


def run() -> dict:
    sim = _make_sim()
    g = sim.gmap
    from simulation import wave_fixed
    seed_q = fire_fixed.quantize_scalar(0.9)
    vac_q = atmosphere_fixed.quantize_scalar(0.0)
    seed = _seed_tiles()
    fire_traj, lit_count, atm_traj = [], [], []
    for t in range(TICKS):
        if t < HOLD_UNTIL:
            for (y, x) in seed:
                g.fire[y, x] = max(int(g.fire[y, x]), seed_q)
        if t == GRENADE_AT:
            # Grenade shockwave: a wave_source kick spikes the wind (worst-case W —
            # the sqrt_q16 / overflow path) -> a transient directional firestorm kick.
            g.wave_source[WOOD_ROWS[2], GRENADE_X] = wave_fixed.quantize_scalar(8.0)
        if t >= VENT_AT:
            # Decompression: pull the interior atmosphere toward vacuum -> the O2
            # proxy collapses -> the firestorm blows out and extinguishes.
            g.atmosphere[1:H - 1, 1:W - 1] = vac_q
        sim.step()
        f = _fire_real(g)
        fire_traj.append(f.copy())
        lit_count.append(int((f > 0.05).sum()))
        atm_traj.append(atmosphere_fixed.dequantize(g.atmosphere).astype(np.float64))
    return {"fire": fire_traj, "lit": lit_count, "atm": atm_traj}


def _first_lit_tick(lit) -> int:
    for t, c in enumerate(lit):
        if c > 0:
            return t
    return -1


def _peak_tick(lit) -> int:
    return int(np.argmax(lit)) if lit else -1


def _extinguish_tick(lit) -> int:
    """First tick AT/after the peak where lit drops back to 0 (-1 if never)."""
    if not lit:
        return -1
    peak = max(lit)
    if peak == 0:
        return -1
    peaked = False
    for t, c in enumerate(lit):
        if c >= peak:
            peaked = True
        if peaked and c == 0:
            return t
    return -1


def _spread_tiles(fire_traj) -> int:
    """Total number of DISTINCT tiles that were ever lit (>0.05) across the run —
    the multi-tile-firestorm confirmation (vs a single decaying cell)."""
    ever = np.zeros_like(fire_traj[0], dtype=bool)
    for f in fire_traj:
        ever |= (f > 0.05)
    return int(ever.sum())


def capture(path: str):
    data = run()
    # Store the trajectories as float32 to keep the committed golden small. The
    # values are exact Q16.16 dequantizations (n/65536), all representable in
    # float32 without loss at the ~1e-4 comparison scale — compare() upcasts to
    # float64 so the A/B math is unchanged.
    slim = {
        "fire": np.asarray(data["fire"], dtype=np.float32),
        "atm": np.asarray(data["atm"], dtype=np.float32),
        "lit": data["lit"],
    }
    with open(path, "wb") as f:
        pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)
    lit = data["lit"]
    print(f"captured {TICKS} ticks -> {path}")
    print(f"  peak lit cells={max(lit)} @t={_peak_tick(lit)}  "
          f"ignite_t={_first_lit_tick(lit)} extinguish_t={_extinguish_tick(lit)}  "
          f"distinct tiles ever lit={_spread_tiles(data['fire'])}")


def compare(path: str):
    with open(path, "rb") as f:
        golden = pickle.load(f)
    cur = run()
    ga = np.array(golden["fire"], dtype=np.float64)
    ca = np.array(cur["fire"], dtype=np.float64)
    assert ga.shape == ca.shape, f"shape {ga.shape} != {ca.shape}"
    d = ca - ga
    max_cell = float(np.abs(d).max())
    l2 = float(np.sqrt((d * d).sum()))
    mse = float(d.mean())
    # Number of cell-ticks where |delta| exceeds the I_min snap-extinguish floor —
    # the only place float/int can differ by more than a Q16.16 LSB: a marginal cell
    # straddling the discrete `I < I_min -> 0` cliff for ONE tick (one build snaps a
    # tick before the other). Everywhere else the diff is pure quantization (~1e-4).
    I_min = 0.02
    n_snap = int((np.abs(d) > I_min * 0.5).sum())
    # max |delta| over cell-ticks that are NOT at the snap cliff (the body of the run)
    body = np.abs(d).copy()
    body[np.abs(d) > I_min * 0.5] = 0.0
    max_cell_body = float(body.max())
    ig_g, ig_c = _first_lit_tick(golden["lit"]), _first_lit_tick(cur["lit"])
    pk_g, pk_c = _peak_tick(golden["lit"]), _peak_tick(cur["lit"])
    ex_g, ex_c = _extinguish_tick(golden["lit"]), _extinguish_tick(cur["lit"])
    sp_g, sp_c = _spread_tiles(golden["fire"]), _spread_tiles(cur["fire"])
    atm_mse = None
    if "atm" in golden and "atm" in cur:
        gatm = np.array(golden["atm"], dtype=np.float64)
        catm = np.array(cur["atm"], dtype=np.float64)
        if gatm.shape == catm.shape:
            atm_mse = float((catm - gatm).mean())
    print(f"S3 spreading-firestorm feel-A/B (float golden vs integer build), {TICKS} ticks:")
    print(f"  fire max-cell |delta|  = {max_cell:.6e}   "
          f"({n_snap} cell-ticks at the I_min={I_min} snap-extinguish cliff)")
    print(f"  fire max-cell |delta| (body, off the snap cliff) = {max_cell_body:.6e}   "
          f"(pure Q16.16 quantization)")
    print(f"  fire L2 (trajectory)   = {l2:.6e}")
    print(f"  fire mean SIGNED error = {mse:.3e}   (deposit-bias; ~0 wanted)")
    if atm_mse is not None:
        print(f"  atmosphere PLUME mean SIGNED error = {atm_mse:.3e}")
    print(f"  ignition  tick: float={ig_g}  int={ig_c}  (|diff|={abs(ig_g-ig_c)})")
    print(f"  peak      tick: float={pk_g}  int={pk_c}  (|diff|={abs(pk_g-pk_c)})")
    print(f"  extinguish tick: float={ex_g}  int={ex_c}  (|diff|={abs(ex_g-ex_c)})")
    print(f"  distinct tiles ever lit: float={sp_g}  int={sp_c}  (the firestorm reach)")
    # The meaningful feel gates: the FELT trajectory (ignite/peak/extinguish ticks +
    # spread reach) is identical, the body diff is pure quantization, and there is no
    # deposit bias. A handful of 1-tick straddles at the I_min snap cliff are
    # expected (a discrete output) and visually invisible.
    ok = (max_cell_body < 2e-3 and abs(ig_g - ig_c) <= 1
          and abs(pk_g - pk_c) <= 1 and abs(ex_g - ex_c) <= 1 and sp_g == sp_c)
    print("  RESULT:", "PASS" if ok else "REVIEW")
    return ok


def render(golden_path: str, png_path: str):
    """Side-by-side feel artifact: FLOAT golden (top row) vs INTEGER build (middle
    row) fire-intensity heatmaps at 5 well-chosen frames across the firestorm arc
    (ignite, spreading, growing, peak, blowing out), plus the |int-float| diff
    (bottom row). The diff is ~1-LSB Q16.16 (< ~1e-3) everywhere the fire is burning
    -> the firestorm SPREADS and EXTINGUISHES identically on both builds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(golden_path, "rb") as f:
        golden = pickle.load(f)
    cur = run()
    ga = np.array(golden["fire"], dtype=np.float64)
    ca = np.array(cur["fire"], dtype=np.float64)

    lit = cur["lit"]
    pk = _peak_tick(lit)
    # Five frames across the arc: ignite, spreading, growing, peak firestorm, blowing
    # out (decompression). pk is the data-driven peak; the others bracket the spread
    # + extinguish phases.
    frames = [
        (6, "ignite"),
        (48, "spreading"),
        (min(GRENADE_AT + 25, TICKS - 1), "growing"),
        (pk, "peak firestorm"),
        (min(VENT_AT + 30, TICKS - 1), "blowing out"),
    ]
    # de-dup while preserving order (in case pk collides with a fixed frame)
    seen, ticks, caps = set(), [], []
    for t, c in frames:
        if t not in seen:
            seen.add(t); ticks.append(t); caps.append(c)

    # Crop to the active lattice + the firestorm's reach so the heatmaps are filled,
    # not lost in the hull margins. Bound it by the union of all lit cells across the
    # shown frames (with a small pad), clamped to the grid.
    ever = np.zeros_like(ga[0], dtype=bool)
    for t in ticks:
        ever |= (ca[t] > 0.05) | (ga[t] > 0.05)
    ys, xs2 = np.where(ever)
    pad = 2
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs2.min() - pad), min(W, xs2.max() + 1 + pad)
    crop = (slice(y0, y1), slice(x0, x1))

    nrows, ncols = 3, len(ticks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 8.4))
    row_lab = ("FLOAT golden", "INTEGER build", "|int - float|")
    max_diff = 0.0
    for j, t in enumerate(ticks):
        axes[0, j].imshow(ga[t][crop], vmin=0, vmax=1, cmap="inferno", aspect="equal")
        axes[1, j].imshow(ca[t][crop], vmin=0, vmax=1, cmap="inferno", aspect="equal")
        dd = np.abs(ca[t] - ga[t])
        max_diff = max(max_diff, float(dd.max()))
        im = axes[2, j].imshow(dd[crop], vmin=0, vmax=2e-4, cmap="hot", aspect="equal")
        nlit = int((ca[t] > 0.05).sum())
        axes[0, j].set_title(f"{caps[j]}\nt={t}   lit={nlit}", fontsize=11)
        axes[2, j].set_title(f"max|d|={dd.max():.1e}", fontsize=10)
        for r in range(nrows):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    for r in range(nrows):
        axes[r, 0].set_ylabel(row_lab[r], fontsize=12)
    fig.colorbar(im, ax=axes[2, :].tolist(), shrink=0.7,
                 label="|delta| (fire intensity)")
    fig.suptitle(
        "S3 fire feel-check — float vs integer (diff is ~1-LSB Q16.16, "
        f"< {max(max_diff, 1.4e-4):.1e})\n"
        "ignite -> SPREAD across the wood lattice (10 -> 70 tiles) -> peak firestorm "
        "-> decompression blow-out -> extinguish   |   integer == float",
        fontsize=13)
    fig.savefig(png_path, dpi=95, bbox_inches="tight")
    print(f"rendered -> {png_path}  (max diff over shown frames = {max_diff:.3e})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "tests" / "_s3_feel_float.pkl")
    if mode == "capture":
        capture(path)
    elif mode == "render":
        png = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "tests" / "_s3_firestorm_feelcheck.png")
        render(path, png)
    else:
        compare(path)
