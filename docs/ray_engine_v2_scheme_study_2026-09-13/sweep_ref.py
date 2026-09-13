"""Reference implementation of the ray-engine-v2 heat sweep, float, in numpy.

Faithful to docs/ray_engine_v2_design_2026-09-12.md section 2.3 WITH critique 1's
required fixes 3, 4, 5 applied:
  - the downwind face split carries the remainder (fix 3)
  - emission re-associated as (E * w) * a, the same arithmetic as absorption (fix 4)
  - an ambient-blackbody inflow boundary condition (fix 5)

Plus the out-of-plane leak channel under test:
  leaked_i = (I_in - absorbed_i) * kleak        -> booked to the ceiling counter
  ceiling_return = E_amb * w_m * kleak          -> so a uniform AMBIENT field is a
                                                   fixed point (second-law gate 2)

Float here; the integer version is a transcription (every operation is a multiply
then a shift).  This file is a measuring instrument, not engine code.
"""
from __future__ import annotations
import math
import numpy as np

TWO_PI = 2.0 * math.pi


def ordinates(n=16, phase=0.0, half_offset=True):
    off = 0.5 if half_offset else 0.0
    ang = (np.arange(n) + off) * (TWO_PI / n) + phase
    return np.cos(ang), np.sin(ang), 1.0 / n        # weights sum to 1 (design's w_m)


def sweep(a, E, kleak=0.0, E_amb=0.0, n=16, phase=0.0, payload=False):
    """One tick of the sweep.

    a      : per-cell absorptivity == emissivity (Kirchhoff), 0..1
    E      : per-cell total emissive power E_deg[T] (the baked table's value)
    kleak  : per-cell out-of-plane leak coefficient (scalar or array)
    E_amb  : the ambient blackbody level, used for the sky inflow AND the
             ceiling's return radiation
    Returns (dE_mat, fluence, sky_out, sky_in, ceiling_net)
    """
    h, w = a.shape
    kl = np.broadcast_to(np.asarray(kleak, float), (h, w))
    mus, etas, wt = ordinates(n, phase)
    dE = np.zeros((h, w))
    fluence = np.zeros((h, w))
    sky_out = 0.0
    sky_in = 0.0
    ceiling = 0.0
    for mu, eta in zip(mus, etas):
        am, ae = abs(mu), abs(eta)
        fx_w = am / (am + ae)                      # downwind face split
        sx = 1 if mu > 0 else -1
        sy = 1 if eta > 0 else -1
        xs = range(w) if mu > 0 else range(w - 1, -1, -1)
        ys = range(h) if eta > 0 else range(h - 1, -1, -1)
        inx = np.zeros((h, w))                     # flux waiting on each cell's x face
        iny = np.zeros((h, w))
        # ambient inflow boundary (fix 5): every boundary face admits E_amb*w
        bx = E_amb * wt * fx_w
        by = E_amb * wt * (1.0 - fx_w)
        if mu > 0:
            inx[:, 0] = bx
        else:
            inx[:, -1] = bx
        if eta > 0:
            iny[0, :] = by
        else:
            iny[-1, :] = by
        sky_in += (bx * h + by * w)
        src = E * wt                               # (E * w) -- fix 4, first half
        for y in ys:
            for x in xs:
                i_in = inx[y, x] + iny[y, x]
                # the out-of-plane channel acts on the STREAM, before the material
                # interaction: otherwise the ceiling's return over-credits by a*k
                # and a uniform ambient field is no longer a fixed point.
                leaked = i_in * kl[y, x]
                ret = E_amb * wt * kl[y, x]        # the ceiling radiates back at ambient
                stream = i_in - leaked + ret
                ceiling += leaked - ret
                ai = a[y, x]
                absorbed = stream * ai
                emitted = src[y, x] * ai           # (E*w) * a -- fix 4, second half
                i_out = stream - absorbed + emitted
                dE[y, x] += absorbed - emitted
                fluence[y, x] += stream
                fx = i_out * fx_w
                fy = i_out - fx                    # remainder split -- fix 3
                nx, ny = x + sx, y + sy
                if 0 <= nx < w:
                    inx[y, nx] = fx
                else:
                    sky_out += fx
                if 0 <= ny < h:
                    iny[ny, x] = fy
                else:
                    sky_out += fy
    return dE, fluence, sky_out, sky_in, ceiling


def sweep_shear(a, E, kleak=0.0, E_amb=0.0, n=16, phase=0.0):
    """Same paired/remainder bookkeeping, SHEAR transport (critique 1 fix 6).

    Advance one full cell along the major axis and split the outgoing stream between
    the two transverse neighbours there with f = |minor/major|.  The split carries the
    remainder, so it is exact.  Absorption and emission stay the SAME paired terms as
    in sweep(), which is why conservation survives: the transport operator only moves
    the stream around, it never decides how much is absorbed.

    (This is the reconciliation of the two disagreeing measurements: the scheme study's
    26% leak came from charging absorption as kappa*I*w while advancing the stream with
    exp(-kappa*path) over a direction-dependent path -- two different absorption laws in
    one step.  Charged the paired way, the residual is zero for BOTH schemes, and the
    direction-dependent path length becomes a physics bias, not a conservation leak.)
    """
    h, w = a.shape
    kl = np.broadcast_to(np.asarray(kleak, float), (h, w))
    mus, etas, wt = ordinates(n, phase)
    dE = np.zeros((h, w)); fluence = np.zeros((h, w))
    sky_out = sky_in = ceiling = 0.0
    for mu, eta in zip(mus, etas):
        am, ae = abs(mu), abs(eta)
        x_major = am >= ae
        f = (ae / am) if x_major else (am / ae)
        sx = 1 if mu > 0 else -1
        sy = 1 if eta > 0 else -1
        inflow = np.zeros((h, w))
        bnd = E_amb * wt
        # Ambient inflow boundary.  A shear step gathers from TWO upwind cells, so a
        # cell on the transverse inlet edge is missing its fb source as well: seeding
        # only the major-axis edge leaves a hole that propagates inward and breaks the
        # uniform-field fixed point.  Both contributions have to be seeded.
        col = 0 if mu > 0 else w - 1
        row = 0 if eta > 0 else h - 1
        if x_major:
            inflow[:, col] += (1.0 - f) * bnd          # fa source outside
            inflow[:, col] += f * bnd                  # fb source outside too
            inflow[row, :] += f * bnd
            inflow[row, col] -= f * bnd                # the corner was counted twice
            sky_in += bnd * h + f * bnd * (w - 1)
        else:
            inflow[row, :] += (1.0 - f) * bnd
            inflow[row, :] += f * bnd
            inflow[:, col] += f * bnd
            inflow[row, col] -= f * bnd
            sky_in += bnd * w + f * bnd * (h - 1)
        src = E * wt
        xs = range(w) if mu > 0 else range(w - 1, -1, -1)
        ys = range(h) if eta > 0 else range(h - 1, -1, -1)
        order = ((y, x) for x in xs for y in ys) if x_major else ((y, x) for y in ys for x in xs)
        for y, x in order:
            i_in = inflow[y, x]
            leaked = i_in * kl[y, x]
            ret = E_amb * wt * kl[y, x]
            stream = i_in - leaked + ret
            ceiling += leaked - ret
            ai = a[y, x]
            absorbed = stream * ai
            emitted = src[y, x] * ai
            i_out = stream - absorbed + emitted
            dE[y, x] += absorbed - emitted
            fluence[y, x] += stream
            fa = i_out * (1.0 - f)
            fb = i_out - fa                      # remainder split, exact
            if x_major:
                nx = x + sx
                if 0 <= nx < w:
                    inflow[y, nx] += fa
                    ny = y + sy
                    if 0 <= ny < h:
                        inflow[ny, nx] += fb
                    else:
                        sky_out += fb
                else:
                    sky_out += i_out
            else:
                ny = y + sy
                if 0 <= ny < h:
                    inflow[ny, x] += fa
                    nx = x + sx
                    if 0 <= nx < w:
                        inflow[ny, nx] += fb
                    else:
                        sky_out += fb
                else:
                    sky_out += i_out
    return dE, fluence, sky_out, sky_in, ceiling


def sweep_blend(a, E, kleak=0.0, E_amb=0.0, n=16, phase=0.0, theta=0.5):
    """Convex blend of the STEP split and the SHEAR split, theta = shear fraction.

    Motivated by two independent statements in the literature:
      - Davis+2012 section 5: a less diffusive transport step gives sharper shadows but
        grows "fan-shaped spokes" when the angular resolution is modest, and there "a
        greater degree of diffusion in the intensity can mitigate unphysical effects".
      - Camminady+2019 section 7: for a STEADY-STATE SWEEP their rotation is not
        applicable, and the remedy is instead to solve the modified equation directly,
        i.e. to add the diffusion that rotation would have produced.

    A convex combination of two conservative, positive transport operators is itself
    conservative and positive, so theta costs nothing structurally: it is one dial that
    trades near-field isotropy against spoke suppression, and both ends are already
    measured (theta=0 is step, theta=1 is shear).
    """
    h, w = a.shape
    kl = np.broadcast_to(np.asarray(kleak, float), (h, w))
    mus, etas, wt = ordinates(n, phase)
    dE = np.zeros((h, w)); fluence = np.zeros((h, w))
    sky_out = sky_in = ceiling = 0.0
    for mu, eta in zip(mus, etas):
        am, ae = abs(mu), abs(eta)
        sx = 1 if mu > 0 else -1
        sy = 1 if eta > 0 else -1
        x_major = am >= ae
        f = (ae / am) if x_major else (am / ae)
        fx_w = am / (am + ae)
        inflow = np.zeros((h, w))
        bnd = E_amb * wt
        col = 0 if mu > 0 else w - 1
        row = 0 if eta > 0 else h - 1
        # step part of the boundary + shear part of the boundary
        inflow[:, col] += (1 - theta) * bnd * fx_w
        inflow[row, :] += (1 - theta) * bnd * (1 - fx_w)
        if x_major:
            inflow[:, col] += theta * bnd
            inflow[row, :] += theta * f * bnd
            inflow[row, col] -= theta * f * bnd
            sky_in += (1-theta)*(bnd*fx_w*h + bnd*(1-fx_w)*w) + theta*(bnd*h + f*bnd*(w-1))
        else:
            inflow[row, :] += theta * bnd
            inflow[:, col] += theta * f * bnd
            inflow[row, col] -= theta * f * bnd
            sky_in += (1-theta)*(bnd*fx_w*h + bnd*(1-fx_w)*w) + theta*(bnd*w + f*bnd*(h-1))
        src = E * wt
        xs = range(w) if mu > 0 else range(w - 1, -1, -1)
        ys = range(h) if eta > 0 else range(h - 1, -1, -1)
        order = (((y, x) for x in xs for y in ys) if x_major
                 else ((y, x) for y in ys for x in xs))
        for y, x in order:
            i_in = inflow[y, x]
            leaked = i_in * kl[y, x]
            ret = E_amb * wt * kl[y, x]
            stream = i_in - leaked + ret
            ceiling += leaked - ret
            ai = a[y, x]
            absorbed = stream * ai
            emitted = src[y, x] * ai
            i_out = stream - absorbed + emitted
            dE[y, x] += absorbed - emitted
            fluence[y, x] += stream
            nx, ny = x + sx, y + sy
            def push(tgt_y, tgt_x, amt):
                nonlocal sky_out
                if 0 <= tgt_x < w and 0 <= tgt_y < h:
                    inflow[tgt_y, tgt_x] += amt
                else:
                    sky_out += amt
            i_step = i_out * (1.0 - theta)
            i_shear = i_out - i_step
            sfx = i_step * fx_w
            push(y, nx, sfx)
            push(ny, x, i_step - sfx)
            if x_major:
                ha = i_shear * (1.0 - f)
                push(y, nx, ha)
                push(ny, nx, i_shear - ha)
            else:
                ha = i_shear * (1.0 - f)
                push(ny, x, ha)
                push(ny, nx, i_shear - ha)
    return dE, fluence, sky_out, sky_in, ceiling
