"""THE PER-CELL AMBIENT (thermal model v2 R3 / R2, patch T1) — design v2 §6
items 2, 3 and 4, on the ENGINE.

`amb_m = (E°[0]·w_m) >> 16` — the emissive LEVEL the sweep radiates against —
stops being hoisted once per run and becomes a per-cell read, derived from the
`is_vacuum` plane the conductor already holds (`RadiationSweep::derive_ambient`).
`k_leak` goes live at the same time. Three properties pin that:

  * **item 2** — the uniform case is unchanged, integer for integer;
  * **item 3** — the non-uniform case genuinely varies per cell;
  * **item 4** — conservation survives both changes at once.

WHY ITEM 3 IS WRITTEN THE WAY IT IS. An earlier draft of this patch (v1, caught
by critique L2-B1) named the wrong scalar entirely: it made **`t_amb_q`** — the
absolute-zero offset of the temperature scale, which feeds only the Fleck
denominator — per-tile, and left `amb_m` hoisted. That draft would have passed
gate 0, the conservation identity, the uniform-case pin and every existing sweep
gate while delivering **nothing**: no cell would have radiated against a
different ambient than any other. Item 3 therefore is not "assert something
changed" — it is built so that **no once-per-run scalar ambient of any value can
produce its measured answer** (see the test's own docstring for the argument).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_ambient_plane.py -q
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import (  # noqa: E402  (sets up sys.path: build dir + study)
    R, cpp_sweep, random_scene, ref_sweep, reference_table)
import breach_physics as bp  # noqa: E402

Q = R.quant
K_LEAK = Q(0.10)
E0 = R.E0

# Gate 0's own configuration matrix, in gate 0's own order: 4 scenes x S16/S12 x
# leak off/on x both transports = 32 runs. The scenes come from the harness's
# `random_scene` at gate 0's seeds, so the two gates measure the SAME scenes.
MATRIX = [(1, 7, 9), (2, 9, 11), (3, 12, 8), (4, 5, 5)]

# THE SCALAR ERA, sha256 (item 2). NOT a snapshot of current behaviour: it was
# captured from the PRE-PATCH integer reference at commit b3488fc — the spec the
# pre-patch engine was held to bit for bit by gate 0 — and therefore records what
# the sweep produced BEFORE `amb_m` went per-cell. Re-derivable at any time with
#     git show b3488fc:docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py
# driven through `_scalar_era_digest` below. It moves only when the uniform path
# is DELIBERATELY changed, with written rationale, exactly as a golden does.
SCALAR_ERA_DIGEST = "6a28a3fba102cf63fe51a0aab7880ae8e411ef5995a41c5bb49cd56798655f00"


def _scalar_era_digest(driver):
    """sha256 over gate 0's 32-configuration matrix: for each config, the config
    tag, then the five planes (rad_net, rad_flux, rad_amb, rad_fluence, fleck) as
    contiguous little-endian int64, then (min_stream, max_stream) as int64.

    `driver(a, d, k_q, T, his, ts, transport, n_ord)` returns
    ((five planes), min_stream, max_stream).
    """
    hsh = hashlib.sha256()
    for seed, h, w in MATRIX:
        for n_ord in (16, 12):
            for k_q in (0, K_LEAK):
                for transport in ("shear", "step"):
                    rng = random.Random(20260916 + seed)
                    a, d, T, his, ts = random_scene(rng, h, w)
                    planes, mn, mx = driver(a, d, k_q, T, his, ts, transport, n_ord)
                    hsh.update(f"{seed}|{h}|{w}|{n_ord}|{k_q}|{transport}|".encode())
                    for p in planes:
                        hsh.update(np.ascontiguousarray(
                            np.asarray(p, dtype=np.int64)).tobytes())
                    hsh.update(np.asarray([mn, mx], dtype=np.int64).tobytes())
    return hsh.hexdigest()


# --------------------------------------------------------------------------- #
# ITEM 2 — the uniform case is unchanged
# --------------------------------------------------------------------------- #
def test_item2_the_uniform_ambient_path_reproduces_the_scalar_era_integer_for_integer():
    """ITEM 2. PROPERTY: with the ambient uniform, the per-cell machinery
    produces the SCALAR ERA's output, integer for integer — every plane of every
    one of gate 0's 32 configurations, plus the Fleck plane and the stream
    telemetry. The engine and the reference are both checked against the same
    frozen digest, so the claim is made on the shipped code and not only on the
    spec.

    BREAKS IF: `amb_level` is mis-indexed (a wrong stride, a transposed read, an
    off-by-one, the scratch read past its fill) — a uniform plane hides a
    mis-index only while the read stays inside the plane; BREAKS IF the per-cell
    `amb_m_` loses the `(amb·w_m) >> 16` scaling or applies it twice; BREAKS IF
    the excess `E°[T_i] − amb_level[i]` stops reducing to the scalar era's
    `E°[T_i] − E°[0]`; BREAKS IF the Fleck pre-pass's `L` measures its excess
    over a different ambient than the sweep does.

    The digest is a HISTORICAL fact, not a behaviour snapshot: it was captured
    from the pre-patch reference (b3488fc), so it cannot be satisfied by "what
    the code happens to do today".
    """
    def cpp(a, d, k_q, T, his, ts, transport, n_ord):
        rn, rf, ra, rl, fl, sw = cpp_sweep(a, d, k_q, T, his, ts,
                                           transport=transport, n_ord=n_ord,
                                           amb=None)
        return (rn, rf, ra, rl, fl), sw.min_stream, sw.max_stream

    def ref(a, d, k_q, T, his, ts, transport, n_ord):
        rn, rf, ra, rl, fl, res = ref_sweep(a, d, k_q, T, his, transport=transport,
                                            n_ord=n_ord, amb=None)
        return (rn, rf, ra, rl, fl), res.min_stream, res.max_stream

    assert _scalar_era_digest(cpp) == SCALAR_ERA_DIGEST, (
        "the C++ sweep's UNIFORM-ambient path moved off the scalar era")
    assert _scalar_era_digest(ref) == SCALAR_ERA_DIGEST, (
        "the integer reference's UNIFORM-ambient path moved off the scalar era")


@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("transport", ["shear", "step"])
def test_item2_the_three_uniform_doors_are_the_same_integers(transport, n_ord):
    """ITEM 2 (the doors). PROPERTY: the three ways to ask for a uniform ambient
    — `amb=None` (the binding's broadcast), `amb=E°[0]` (an explicit scalar), and
    `derive_ambient(is_vacuum, vac_level=-1)` (thermal v2 R4, space at room
    temperature) — give the SAME integers on every plane, whatever the
    `is_vacuum` mask says.

    BREAKS IF: the binding's `None` broadcast and `derive_ambient`'s R4 branch
    drift apart, so that the shipped conductor path stops being the one the
    scalar-era digest above pins. Note the mask is NON-TRIVIAL here: under R4 it
    must make no difference, which is what makes today's live path bit-identical
    to the scalar era.
    """
    rng = random.Random(20260919 + n_ord)
    h, w = 8, 10
    a, d, T, his, ts = random_scene(rng, h, w)
    tbl = reference_table()
    sweep = bp.RadiationSweep()
    iv = np.asarray([[rng.random() < 0.4 for _ in range(w)] for _ in range(h)],
                    dtype=bool)
    amb_r4 = sweep.derive_ambient(iv, tbl, -1)
    assert np.all(np.asarray(amb_r4) == E0), "R4: vac_level < 0 must give E°[0] everywhere"

    runs = [cpp_sweep(a, d, K_LEAK, T, his, ts, transport=transport, n_ord=n_ord,
                      table=tbl, amb=amb)
            for amb in (None, E0, amb_r4)]
    for name, i in (("E°[0] scalar", 1), ("derive_ambient(R4)", 2)):
        for pi, plane in enumerate(("rad_net", "rad_flux", "rad_amb", "rad_fluence",
                                    "fleck")):
            assert np.array_equal(np.asarray(runs[0][pi], dtype=np.int64),
                                  np.asarray(runs[i][pi], dtype=np.int64)), (
                f"{name} differs from the None door on {plane}")


# --------------------------------------------------------------------------- #
# ITEM 3 — the non-uniform case genuinely varies, and CANNOT be faked by a hoist
# --------------------------------------------------------------------------- #
#
# THE SCENE: two identical compartments, mirror images of each other about the
# vertical midline —
#
#       x =   0   1   2   3   4   5   6   7   8   9
#           [out out|WALL| in  in  in  in |WALL|out out]
#
# every plane the sweep reads (a, d, T, his, thermal_solid) is EXACTLY
# mirror-symmetric, and every temperature sits at ambient. The ONLY asymmetric
# input in the whole scene is `is_vacuum`: the PORT side is open to space, the
# starboard side is not. Two structurally identical hull walls, one facing
# vacuum and one facing interior air — design v2 §6 item 3's "a vacuum-ring cell
# versus an interior cell", built so the difference has exactly one cause.
H_HULL, W_HULL = 7, 10
WALL_PORT, WALL_STBD = 2, W_HULL - 3
A_WALL = Q(0.91)


def _hull_scene():
    a = [[0] * W_HULL for _ in range(H_HULL)]
    d = [[0] * W_HULL for _ in range(H_HULL)]
    T = [[0] * W_HULL for _ in range(H_HULL)]          # every cell AT ambient
    his = [[4] * W_HULL for _ in range(H_HULL)]
    ts = [[0] * W_HULL for _ in range(H_HULL)]
    for y in range(H_HULL):
        for x in (WALL_PORT, WALL_STBD):
            a[y][x] = d[y][x] = A_WALL
            ts[y][x] = 1
    iv = np.zeros((H_HULL, W_HULL), dtype=bool)
    iv[:, 0:2] = True                                  # PORT side open to space
    return a, d, T, his, ts, iv


def _mirrored(p):
    arr = np.asarray(p, dtype=np.int64)
    return np.array_equal(arr, arr[:, ::-1])


@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("transport", ["shear", "step"])
def test_item3_a_hull_wall_facing_vacuum_radiates_against_a_different_ambient(transport, n_ord):
    """ITEM 3. PROPERTY: on a scene whose every other input is mirror-symmetric,
    a cold sky on ONE side alone makes the two structurally identical hull walls
    radiate differently — the one facing vacuum loses strictly more. This is R3's
    substance: "room temp in ship, 0 K outside", expressible at last.

    THIS TEST CANNOT PASS WITH A HOISTED, ONCE-PER-RUN `amb_m`, and that is why
    it exists. The argument, measured rather than assumed:

      1. the CONTROL leg runs the same scene under `vac_level = -1` (R4, the
         uniform answer) and measures that all four output planes come back
         EXACTLY mirror-symmetric — the sweep commutes with the mirror;
      2. so does every uniform ambient: the HOIST leg re-measures the symmetry at
         each level the derived plane actually contains (0 and E°[0]);
      3. therefore any implementation that collapses the ambient to one scalar —
         whatever value it picks, whichever cell it reads it from — produces a
         mirror-SYMMETRIC output on this scene;
      4. the CLAIM leg asserts the output is NOT mirror-symmetric.

    A v1-style patch that made `t_amb_q` per-tile and left `amb_m` hoisted lands
    in (3) and fails (4). So does a per-cell plane that is read once outside the
    cell loop, or read at a fixed index.

    BREAKS IF: any of the four ambient-derived sites at cell i (its emission
    floor `src`, its ceiling return `ret`, its body's re-emission `emit_body`,
    the virtual ambient ring it reads) goes back to a global — each of them is
    what carries the cold sky into this wall's ledger.
    NON-VACUITY: the control leg is an exact per-cell ZERO on all three ledger
    planes (the G2a fixed point), so every integer the claim leg measures is the
    per-cell ambient's doing and nothing else's.
    """
    a, d, T, his, ts, iv = _hull_scene()
    for p, name in ((a, "a"), (d, "d"), (T, "T"), (his, "his"), (ts, "ts")):
        assert _mirrored(p), f"the scene's {name} plane is not mirror-symmetric"
    assert not _mirrored(iv.astype(np.int64)), "is_vacuum must be the ONE asymmetry"

    tbl = reference_table()
    sweep = bp.RadiationSweep()

    def run(amb):
        return cpp_sweep(a, d, K_LEAK, T, his, ts, transport=transport,
                         n_ord=n_ord, table=tbl, sweep=sweep, amb=amb)

    # 1. CONTROL — R4 (space at room temperature). The sweep commutes with the
    #    mirror, and the scene is an exact fixed point.
    amb_r4 = sweep.derive_ambient(iv, tbl, -1)
    c_net, c_flux, c_amb, c_flu, _cf, _cs = run(amb_r4)
    for p, name in ((c_net, "rad_net"), (c_flux, "rad_flux"), (c_amb, "rad_amb"),
                    (c_flu, "rad_fluence")):
        assert _mirrored(p), f"control: {name} is not mirror-symmetric"
    assert int(np.count_nonzero(c_net) + np.count_nonzero(c_flux)
               + np.count_nonzero(c_amb)) == 0, "control is not the G2a fixed point"

    # 2. CLAIM — the cold sky (vac_level = 0) on the port side only.
    amb_cold = np.asarray(sweep.derive_ambient(iv, tbl, 0))
    assert set(np.unique(amb_cold).tolist()) == {0, E0}, "the derived plane is uniform"
    net, flux, amb_p, flu, _f, _s = run(amb_cold)
    assert not _mirrored(net), (
        "rad_net came back mirror-symmetric under a one-sided cold sky — the "
        "ambient is not being read per cell")
    assert not _mirrored(amb_p), "rad_amb came back mirror-symmetric"

    # the physical direction, at every row: the wall facing 0 K loses strictly
    # more than its structurally identical twin facing room-temperature air, and
    # both lose (the cold side chills the whole compartment).
    port = net[:, WALL_PORT]
    stbd = net[:, WALL_STBD]
    assert np.all(port < stbd), (port.tolist(), stbd.tolist())
    assert np.all(stbd < 0), stbd.tolist()

    # 3. HOIST — no single global ambient reproduces this, and each is symmetric.
    for level in sorted(set(np.unique(amb_cold).tolist())):
        h_net, h_flux, h_amb, h_flu, _hf, _hs = run(int(level))
        for p, name in ((h_net, "rad_net"), (h_flux, "rad_flux"),
                        (h_amb, "rad_amb"), (h_flu, "rad_fluence")):
            assert _mirrored(p), (
                f"a UNIFORM ambient at {level} produced an asymmetric {name}; the "
                "hoist-impossibility argument rests on this being symmetric")
        assert not np.array_equal(np.asarray(h_net, dtype=np.int64),
                                  np.asarray(net, dtype=np.int64)), (
            f"the per-cell run is reproduced by a hoist to {level}")


@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("transport", ["shear", "step"])
def test_item3_the_cold_cell_is_the_cell_the_mask_names(transport, n_ord):
    """ITEM 3 (localisation — the mis-index half). PROPERTY: with a SINGLE
    vacuum cell in an otherwise interior grid, the cell that books the ambient
    return of a cold sky is the one `is_vacuum` names, and no other — its
    `rad_amb` is strictly the largest on the grid.

    BREAKS IF: the per-cell ambient is read at a shifted index (a wrong stride, a
    roll, a transpose): the response then lands on a NEIGHBOUR and the strict
    maximum moves. Also BREAKS IF the ambient is hoisted: a uniform ambient makes
    this all-at-ambient scene an exact fixed point, so `rad_amb` is identically
    zero and has no strict maximum at all.

    Gate 0's non-uniform ambient axis (test_radiation_sweep_reference.py) is the
    decisive mis-index gate — it holds the engine bit for bit to a reference that
    reads the right cell — and a roll-by-one was measured to fail it. This leg is
    the same property asserted where a reader of THIS file will look for it.
    """
    h, w = 9, 11
    a = [[0] * w for _ in range(h)]
    d = [[0] * w for _ in range(h)]
    T = [[0] * w for _ in range(h)]
    his = [[4] * w for _ in range(h)]
    ts = [[0] * w for _ in range(h)]
    for (y, x) in ((1, 1), (2, 7), (5, 3), (6, 8), (7, 2)):
        a[y][x] = d[y][x] = A_WALL
        ts[y][x] = 1
    y0, x0 = 4, 6
    tbl = reference_table()
    sweep = bp.RadiationSweep()
    iv = np.zeros((h, w), dtype=bool)
    iv[y0, x0] = True
    amb = sweep.derive_ambient(iv, tbl, 0)

    _rn, _rf, ra, _rl, _fl, _sw = cpp_sweep(a, d, K_LEAK, T, his, ts,
                                            transport=transport, n_ord=n_ord,
                                            table=tbl, sweep=sweep, amb=amb)
    am = np.asarray(ra, dtype=np.int64)
    peak = am[y0, x0]
    others = np.delete(am.reshape(-1), y0 * w + x0)
    assert peak > 0, "the masked cell books nothing — the ambient is not per-cell"
    assert peak > others.max(), (
        f"rad_amb peaks at {np.unravel_index(int(np.argmax(am)), am.shape)}, not at "
        f"the masked cell ({y0},{x0}) — the ambient plane is read at the wrong index")


def test_item3_derive_ambient_is_the_r3_derivation_and_keys_on_is_vacuum_alone():
    """ITEM 3 (the derivation). PROPERTY: `RadiationSweep::derive_ambient` builds
    the per-cell ambient LEVEL from `is_vacuum` and nothing else — a vacuum cell
    takes `vac_level`, every other cell E°[0] — with `vac_level < 0` meaning
    E°[0] (R4) and a level above E°[0] refused at the door.

    BREAKS IF: the derivation starts reading a material column or level data
    (R3's whole point is that NOTHING is authored per tile); BREAKS IF the R4
    sentinel stops collapsing the plane to uniform, which is what keeps today's
    live path bit-identical to the scalar era; BREAKS IF the invariant
    `0 <= amb <= E°[0]` stops being enforced — a hotter-than-room ambient makes a
    cell's emission excess NEGATIVE, which the excess-form Fleck factor
    (design v3 row 22) and the sweep's positivity both rest on being >= 0.
    """
    tbl = reference_table()
    sweep = bp.RadiationSweep()
    h, w = 5, 6
    iv = np.zeros((h, w), dtype=bool)
    iv[0, :] = True          # the ambient ring's top row, as a breach would be
    iv[2, 3] = True

    cold = np.asarray(sweep.derive_ambient(iv, tbl, 0))
    assert cold.shape == (h, w)
    assert np.array_equal(cold == 0, iv), "the select is not keyed on is_vacuum"
    assert np.all(cold[~iv] == E0), "an interior cell must take E°[0]"

    mid = np.asarray(sweep.derive_ambient(iv, tbl, E0 // 3))
    assert np.array_equal(mid == E0 // 3, iv)

    assert np.all(np.asarray(sweep.derive_ambient(iv, tbl, -1)) == E0), "R4"
    assert np.all(np.asarray(sweep.derive_ambient(iv, tbl, E0)) == E0), (
        "vac_level == E°[0] is the uniform ambient by another name")

    with pytest.raises(ValueError):
        sweep.derive_ambient(iv, tbl, E0 + 1)


# --------------------------------------------------------------------------- #
# ITEM 4 — conservation, with BOTH changes live at once
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ambient", ["cold-patch", "derived", "random"])
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("transport", ["shear", "step"])
def test_item4_conservation_is_exact_with_a_per_cell_ambient_and_the_leak_live(
        transport, n_ord, ambient):
    """ITEM 4. PROPERTY: Σ rad_net + Σ rad_flux + Σ rad_amb == 0 EXACTLY, in
    int64 — no tolerance — on a randomised scene with a NON-UNIFORM ambient and
    `k_leak > 0` at the same time, both transports, S16 and S12. Each of the
    three sums is individually non-zero, so none of the three terms is vacuous.

    The identity is structural: every integer that leaves a cell's stream is
    booked once with each sign, so it must survive an ambient that varies from
    cell to cell. The two changes are exercised TOGETHER on purpose — `k_leak`
    is what makes the ceiling return `ret` load-bearing, and `ret` is one of the
    four sites that took the per-cell ambient.

    BREAKS IF: a writer books one side of a transfer at cell i's ambient and the
    other at a different cell's — the ceiling's `leaked`/`ret` pair, the virtual
    ring's two books, or the body's absorb/re-emit pair losing their
    same-integer association.

    The three ambients: a hull-breach cold patch, the R3 derivation driven off a
    real `is_vacuum` mask (the shipped path), and a per-cell random level over
    the whole legal range [0, E°[0]].
    """
    rng = random.Random(20260919 + n_ord + (transport == "step") * 7)
    h, w = 9, 11
    a, d, T, his, ts = random_scene(rng, h, w)
    tbl = reference_table()
    sweep = bp.RadiationSweep()
    if ambient == "cold-patch":
        amb = [[0 if x < w // 3 else E0 for x in range(w)] for _ in range(h)]
    elif ambient == "derived":
        iv = np.zeros((h, w), dtype=bool)
        iv[:, 0:2] = True
        iv[0, :] = True
        amb = sweep.derive_ambient(iv, tbl, 0)
    else:
        amb = [[rng.choice([0, 1, E0 // 4, E0 // 2, E0 - 1, E0]) for _ in range(w)]
               for _ in range(h)]
    assert len(np.unique(np.asarray(amb))) > 1, "the ambient must be non-uniform"

    rn, rf, ra, _rl, _fl, _sw = cpp_sweep(a, d, K_LEAK, T, his, ts,
                                          transport=transport, n_ord=n_ord,
                                          table=tbl, sweep=sweep, amb=amb)
    sn, sf, sa = int(rn.sum()), int(rf.sum()), int(ra.sum())
    assert sn + sf + sa == 0, (sn, sf, sa)
    assert sn != 0 and sf != 0 and sa != 0, "a term is vacuous"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
