"""arc #54 P-G1d — the pressure solve's divergence is the FACE FORM (D4).

The defect P-G1c indicted: ``mirror_idx`` returns SELF on a solid neighbour,
so the old central stencil ``(u_ir - u_il)/(2dx)`` implied a wall-face
velocity of ``u_i`` (a zero-GRADIENT ghost), while BOTH of its partners see a
wall as ``u = 0`` — the kick (mirror on ``p`` => no pressure gradient across a
solid face) and the S2.4 face-flux energy step (``face_flux``'s WALL branch).
The solve therefore asked for a compression nobody was charged for, and the
measured result was a standing wall/interior thermal dipole that conduction
against ambient-held walls rectified into a global energy source.

THE TWO CLAIMS THIS GATE PINS

  (1) INTERIOR BIT-IDENTITY. Where all four neighbours are open, the face form
      and the old central form are the SAME INTEGER, before any shift:
          2u_E - 2u_W = (u_i + u_ir) - (u_il + u_i) = u_ir - u_il
      so nothing about ordinary open-air flow moved. (The ``u_i`` terms cancel
      exactly; the 1/2 of the face mean rides in the same ``inv_2dx_q`` the
      central form already carried, so there is no second rounding site.)

  (2) AT A SOLID FACE the divergence now reads ``u = 0``, and it CHANGED there
      — the fix is real and is confined to wall-adjacent cells.

HOW IT IS MEASURED WITHOUT A MOCK. ``EOSSolver::dbg_mg_inputs()`` hands back
``div_u`` exactly as the pressure solve consumed it. The wind that fed it is
the POST-SUBSTEP wind, which is not otherwise observable from Python — so the
tick is driven at a dt so small that every semi-Lagrangian displacement
quantizes to zero (``dt_s_q == 0`` => the sampler's zero-displacement fast
path returns the source values outright). The post-substep wind is then
exactly the wind this test seeded, and both stencils can be transcribed in
NumPy against it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

LEVEL = "playground"        # big, ring-free, and densely walled — both halves
                            # of the claim need a real map, not a toy
TINY_DT = 1e-6          # dt_s_q = quantize(dt) == 0 -> SL displacement is 0


def _seeded_wind(shape, seed=20260830):
    """A deterministic, non-degenerate wind field in Q16 raw (about +/-4 m/s)."""
    rng = np.random.default_rng(seed)
    return (rng.integers(-4 * 65536, 4 * 65536, size=shape)
            .astype(np.int32))


def _build():
    from level_loader import load as load_level
    from simulation.gamemap import GameMap
    from simulation.physics_runner import PhysicsRunner

    level = load_level(LEVEL, levels_dir=str(ROOT / "levels"))
    g = GameMap(level)
    g.stamp_units([])
    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    return g, runner


def _stencils(wx, wy, solid, inv_2dx_q):
    """(face form, old central form) as int32 Q16 planes, transcribed from
    eos_solver.cpp. ``mul_q16(a, b) == (int64(a)*b) >> 16`` (arithmetic shift,
    floors toward -inf), narrowed to int32."""
    h, w = solid.shape

    def mul_q16(a):
        return ((a.astype(np.int64) * np.int64(inv_2dx_q)) >> np.int64(16)
                ).astype(np.int32)

    # --- face form: 2*u_hat, with u_hat = 0 at a solid / out-of-bounds face --
    def face2(u, dy, dx):
        out = np.zeros((h, w), dtype=np.int64)
        ys = slice(max(0, -dy), h - max(0, dy))
        yd = slice(max(0, dy), h - max(0, -dy))
        xs = slice(max(0, -dx), w - max(0, dx))
        xd = slice(max(0, dx), w - max(0, -dx))
        nb_u = np.zeros((h, w), dtype=np.int64)
        nb_solid = np.ones((h, w), dtype=bool)      # OOB counts as a wall
        nb_u[yd, xd] = u[ys, xs]
        nb_solid[yd, xd] = solid[ys, xs]
        return np.where(nb_solid, 0, u.astype(np.int64) + nb_u)

    fe = face2(wx, 0, -1)       # neighbour at x+1 lands in cell x  -> east face
    fw = face2(wx, 0, +1)       # neighbour at x-1                  -> west face
    fs = face2(wy, -1, 0)       # neighbour at y+1                  -> south face
    fn = face2(wy, +1, 0)       # neighbour at y-1                  -> north face
    face_form = mul_q16(fe - fw) + mul_q16(fs - fn)

    # --- old central form: mirror_idx returns SELF on a solid / OOB neighbour
    def mirror(u, dy, dx):
        out = u.astype(np.int64).copy()
        ys = slice(max(0, -dy), h - max(0, dy))
        yd = slice(max(0, dy), h - max(0, -dy))
        xs = slice(max(0, -dx), w - max(0, dx))
        xd = slice(max(0, dx), w - max(0, -dx))
        nb_u = u.astype(np.int64).copy()
        nb_solid = np.ones((h, w), dtype=bool)
        nb_u[yd, xd] = u[ys, xs]
        nb_solid[yd, xd] = solid[ys, xs]
        return np.where(nb_solid, out, nb_u)

    central = (mul_q16(mirror(wx, 0, -1) - mirror(wx, 0, +1))
               + mul_q16(mirror(wy, -1, 0) - mirror(wy, +1, 0)))
    return face_form, central


def test_div_u_is_the_face_form_and_interior_is_bit_identical():
    g, runner = _build()
    assert not g.is_ambient.any(), (
        f"{LEVEL} grew an ambient ring — this gate's exclusion mask assumes "
        "none; pick a ring-free level or extend the mask")

    wx = _seeded_wind(g.wind_x.shape)
    wy = _seeded_wind(g.wind_y.shape, seed=987654321)
    g.wind_x[:] = wx
    g.wind_y[:] = wy

    runner.step(g, TINY_DT)
    eos = runner.eos
    assert eos.dbg_last_n_sub >= 1
    _pstar, div_flat, _ntot = eos.dbg_mg_inputs()
    div = np.asarray(div_flat, dtype=np.int32).reshape(g.temperature.shape)

    solid = np.asarray(g.solid, dtype=bool)
    excl = solid | np.asarray(g.is_vacuum, dtype=bool)
    # fixedpoint::quantize is floor(v*2^16 + 0.5) for v >= 0 — NOT np.round
    # (which is half-to-even). Transcribe it, do not approximate it.
    inv_2dx_q = int(65536.0 / (2.0 * float(g.tile_size_m)) + 0.5)

    face_form, central = _stencils(wx, wy, solid, inv_2dx_q)
    face_form = np.where(excl, 0, face_form)
    central = np.where(excl, 0, central)

    # THE IMPLEMENTATION CLAIM: the solve consumed the face form, exactly.
    assert np.array_equal(div, face_form), (
        f"{int(np.count_nonzero(div != face_form))} cells differ from the "
        "face-form transcription")

    # (1) INTERIOR BIT-IDENTITY — no open-air cell moved.
    solid_nb = np.zeros_like(solid)
    solid_nb[1:, :] |= solid[:-1, :]
    solid_nb[:-1, :] |= solid[1:, :]
    solid_nb[:, 1:] |= solid[:, :-1]
    solid_nb[:, :-1] |= solid[:, 1:]
    solid_nb[0, :] = solid_nb[-1, :] = True     # OOB faces count as walls
    solid_nb[:, 0] = solid_nb[:, -1] = True
    interior = ~excl & ~solid_nb
    assert interior.sum() > 1000, "level too small to be an interior gate"
    assert np.array_equal(face_form[interior], central[interior]), (
        "the face form moved an INTERIOR cell — the u_i terms did not cancel")

    # (2) THE FIX IS REAL, and confined to wall-adjacent cells.
    moved = face_form != central
    assert moved.any(), "the face form changed nothing at any wall"
    assert not moved[interior].any()
    assert (moved & ~excl & solid_nb).sum() > 100


def test_wall_face_velocity_is_zero_in_a_one_cell_pocket():
    """A gas cell walled on all four sides has div == 0 whatever its own u —
    the face form's whole content, in one assertion. The old central form gave
    exactly 0 there too (self on every side), so this is a REGRESSION pin on a
    property both forms share, not a new behaviour: what it guards is that the
    OOB / all-solid path still cancels after the rewrite."""
    solid = np.ones((7, 7), dtype=bool)
    solid[3, 3] = False
    wx = np.full((7, 7), 3 * 65536, dtype=np.int32)
    wy = np.full((7, 7), -2 * 65536, dtype=np.int32)
    face_form, central = _stencils(wx, wy, solid, 65536 // 2)
    assert face_form[3, 3] == 0
    assert central[3, 3] == 0


def test_one_open_neighbour_pocket_differs_by_the_ghost_term():
    """The single face that is open carries u_hat = (u_i + u_j)/2; the old form
    read u_j - u_i across the pair (self on the walled side). With u_i == u_j
    the old form reported ZERO divergence at a cell that is in fact blowing
    into its one open face — that is the D4 mismatch, in one cell."""
    solid = np.ones((7, 7), dtype=bool)
    solid[3, 3] = False
    solid[3, 4] = False                      # the one open (east) face
    u = 3 * 65536
    wx = np.full((7, 7), u, dtype=np.int32)
    wy = np.zeros((7, 7), dtype=np.int32)
    inv_2dx_q = 65536 // 2                   # dx = 1 m
    face_form, central = _stencils(wx, wy, solid, inv_2dx_q)
    # face form: (2*u_hat_E - 0) * inv_2dx = (u + u) * 1/(2dx) = u/dx
    assert face_form[3, 3] == ((2 * u * inv_2dx_q) >> 16)
    assert central[3, 3] == 0                # the ghost: u_ir - u_il == u - u


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
