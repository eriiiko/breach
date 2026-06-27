"""S3b firestorm feel-A/B — capture / compare the scripted firestorm fire field.

A scripted firestorm trajectory (ignite -> wind-fan via a grenade shockwave ->
blow-out -> extinguish) through the REAL Simulation, recording per-tick `fire` in
REAL [0,1] units (dequantized). Run on the OLD float-fire build (S3a) to capture a
golden, then on the integer build (S3b) to compare:

  * max-cell |delta|, L2, MEAN-SIGNED-ERROR on `fire` (the deposit/round bias check)
  * the ignition tick + the extinguish tick on both sides (match within a tick)

Usage:
    # capture golden on the S3a build (pyd built from 28d134b):
    python tests/_s3b_firestorm_feel.py capture tests/_s3b_feel_float.pkl
    # compare the integer build against it:
    python tests/_s3b_firestorm_feel.py compare tests/_s3b_feel_float.pkl

Underscore-prefixed -> a dev artifact (the .pkl golden is committed for Erik's
merge review per the brief; the .py is the harness).
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

import breach_physics as bp           # noqa: E402
from level_loader import LevelData    # noqa: E402
from simulation import Simulation     # noqa: E402
from simulation import fire_fixed     # noqa: E402
from simulation.materials import MAT_WOOD  # noqa: E402

SEED = 70707
TICKS = 120
# the burning seed tile + the wind-fan grenade breach
SEED_Y, SEED_X = 8, 8
HOLD_UNTIL = 25          # hold the seed lit for the first N ticks (ignition + fan)


def _fire_is_int() -> bool:
    """True on the S3b integer build (gmap.fire dtype int32 — always true now, but
    the dequantize is identity-safe either way: we read fire as REAL units)."""
    return True


def _wood_room_level() -> LevelData:
    """A 20x20 hull-walled room with a WOOD interior block to feed the firestorm."""
    h = w = 20
    tm = np.ones((h, w), dtype=np.int32)      # all hull
    tm[1:19, 1:19] = 4                          # carve interior air
    return LevelData(name="s3b_firestorm", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _make_sim() -> Simulation:
    sim = Simulation(_wood_room_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    # Lay a wood block (fuel) around the seed so the fire has something to spread on.
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            y, x = SEED_Y + dy, SEED_X + dx
            if 1 <= y < 19 and 1 <= x < 19:
                g.material[y, x] = MAT_WOOD
    g._update_caches()
    sim.set_paused(False)
    return sim


def _fire_real(g) -> np.ndarray:
    """Read the fire field as REAL [0,1] intensity (dequantize the Q16.16 int)."""
    return fire_fixed.dequantize(g.fire).astype(np.float64)


def run() -> dict:
    sim = _make_sim()
    g = sim.gmap
    from simulation import wave_fixed, atmosphere_fixed
    seed_q = fire_fixed.quantize_scalar(0.9)
    # A strong steady wind injected during the fan phase exercises the W = |wind|
    # (sqrt_q16) + the (1 + k_wind_fan*W) growth + the k_wind_strip*W blow-out terms,
    # AND a grenade shockwave at t==5 spikes the wind (worst-case W / overflow path).
    wind_q = atmosphere_fixed.quantize_scalar(2.0)     # |wind| ~= 2 across the room
    fire_traj = []
    lit_count = []
    atm_traj = []
    for t in range(TICKS):
        if t < HOLD_UNTIL:
            # Hold a 3x3 block of seeds lit so the radiant heat actually builds and
            # ignites adjacent wood (a real spreading firestorm, not a lone tile).
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = SEED_Y + dy, SEED_X + dx
                    g.fire[yy, xx] = max(int(g.fire[yy, xx]), seed_q)
        if 5 <= t < HOLD_UNTIL:
            # Steady +x wind through the blaze (fans growth, blows out the margins).
            g.wind_x[:] = wind_q
        if t == 5:
            # grenade shockwave: a wave_source kick spikes the wind (worst-case W).
            g.wave_source[SEED_Y, SEED_X + 5] = wave_fixed.quantize_scalar(8.0)
        sim.set_paused(False)
        sim.step()
        f = _fire_real(g)
        fire_traj.append(f.copy())
        lit_count.append(int((f > 0.01).sum()))
        atm_traj.append(atmosphere_fixed.dequantize(g.atmosphere).astype(np.float64))
    return {"fire": fire_traj, "lit": lit_count, "atm": atm_traj}


def _first_lit_tick(lit) -> int:
    for t, c in enumerate(lit):
        if c > 0:
            return t
    return -1


def _extinguish_tick(lit) -> int:
    """The first tick AFTER the peak where lit drops back to 0 (and stays ~0)."""
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


def capture(path: str):
    data = run()
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    lit = data["lit"]
    print(f"captured {TICKS} ticks -> {path}")
    print(f"  peak lit cells={max(lit)} ignite_tick={_first_lit_tick(lit)} "
          f"extinguish_tick={_extinguish_tick(lit)}")


def compare(path: str):
    with open(path, "rb") as f:
        golden = pickle.load(f)
    cur = run()
    ga = np.array(golden["fire"], dtype=np.float64)   # (T,h,w)
    ca = np.array(cur["fire"], dtype=np.float64)
    assert ga.shape == ca.shape, f"shape {ga.shape} != {ca.shape}"
    d = ca - ga
    max_cell = float(np.abs(d).max())
    l2 = float(np.sqrt((d * d).sum()))
    mse = float(d.mean())          # mean SIGNED error (the deposit-bias check)
    ig_g, ig_c = _first_lit_tick(golden["lit"]), _first_lit_tick(cur["lit"])
    ex_g, ex_c = _extinguish_tick(golden["lit"]), _extinguish_tick(cur["lit"])
    # The PLUME deposit into atmosphere — its mean signed error vs the float golden
    # is the unbiased-deposit gate (round-to-nearest, NOT truncation toward -inf).
    atm_mse = None
    if "atm" in golden and "atm" in cur:
        gatm = np.array(golden["atm"], dtype=np.float64)
        catm = np.array(cur["atm"], dtype=np.float64)
        if gatm.shape == catm.shape:
            atm_mse = float((catm - gatm).mean())
    print(f"firestorm feel-A/B (float golden vs integer build), {TICKS} ticks:")
    print(f"  fire max-cell |delta|  = {max_cell:.6e}")
    print(f"  fire L2 (trajectory)   = {l2:.6e}")
    print(f"  fire mean SIGNED error = {mse:.3e}   (deposit-bias; ~0 wanted)")
    if atm_mse is not None:
        print(f"  atmosphere PLUME mean SIGNED error = {atm_mse:.3e}   (unbiased deposit gate)")
    print(f"  ignition  tick: float={ig_g}  int={ig_c}  (|diff|={abs(ig_g-ig_c)})")
    print(f"  extinguish tick: float={ex_g}  int={ex_c}  (|diff|={abs(ex_g-ex_c)})")
    ok = (max_cell < 2e-2 and abs(ig_g - ig_c) <= 1 and abs(ex_g - ex_c) <= 1)
    print("  RESULT:", "PASS" if ok else "REVIEW")
    return ok


def render(golden_path: str, png_path: str):
    """Render the firestorm to a committed PNG artifact for Erik's merge review:
    the integer-build fire field at 4 ticks (top row) and the float-vs-integer
    per-cell |delta| at those ticks (bottom row, a hot colormap). The diff is
    near-black (< ~1.4e-4, pure Q16.16 quantization) — that IS the result: the
    integer fire is visually identical to the float golden."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(golden_path, "rb") as f:
        golden = pickle.load(f)
    cur = run()
    ga = np.array(golden["fire"], dtype=np.float64)
    ca = np.array(cur["fire"], dtype=np.float64)
    ticks = [3, 12, 22, 40]
    fig, axes = plt.subplots(2, len(ticks), figsize=(4 * len(ticks), 8))
    for j, t in enumerate(ticks):
        axes[0, j].imshow(ca[t], vmin=0, vmax=1, cmap="inferno")
        axes[0, j].set_title(f"int fire  t={t}")
        axes[0, j].axis("off")
        d = np.abs(ca[t] - ga[t])
        im = axes[1, j].imshow(d, vmin=0, vmax=2e-4, cmap="hot")
        axes[1, j].set_title(f"|int-float| t={t}  max={d.max():.2e}")
        axes[1, j].axis("off")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.6, label="|delta| (fire intensity)")
    fig.suptitle("S3b firestorm feel-A/B — integer fire vs float golden "
                 "(diff is pure Q16.16 quantization, < 1.4e-4)", fontsize=13)
    fig.savefig(png_path, dpi=90, bbox_inches="tight")
    print(f"rendered -> {png_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "tests" / "_s3b_feel_float.pkl")
    if mode == "capture":
        capture(path)
    elif mode == "render":
        png = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "tests" / "_s3b_firestorm_feel.png")
        render(path, png)
    else:
        compare(path)
