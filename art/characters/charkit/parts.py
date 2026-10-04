"""Body parts shared between characters, built from the kit."""
import math

import numpy as np

from kit import Loft, loft_mesh, mirror, unit

# Fingers at scale 1 (a large gloved hand): name, position across the palm, fan,
# length, radius, curl of each joint in radians (relaxed).
DIGITS = (
    ("Index", 0.033, 0.07, 0.068, 0.0116, (0.20, 0.35, 0.30)),
    ("Middle", 0.011, 0.01, 0.075, 0.0120, (0.26, 0.42, 0.32)),
    ("Ring", -0.011, -0.05, 0.070, 0.0114, (0.30, 0.45, 0.32)),
    ("Pinky", -0.033, -0.11, 0.055, 0.0102, (0.34, 0.48, 0.32)))
# Palm sections, wrist to knuckles: distance down the hand, half-width, half-thickness.
PALM = ((-0.020, .040, .034), (0.012, .042, .028), (0.042, .048, .023), (0.072, .050, .020), (0.092, .048, .017), (0.102, .040, .010))


def hand(prefix, wrist, L, B, mat, coll, scale=1.0, curl=1.0, mirrored=True):
    """A gloved LEFT hand with separate fingers, mirrored to the right by default.

    `L` is the unit vector down the fingers, `B` out of the back of the hand. Returns
    `(palm, hp)`: the palm loft (phi = 0 thumb side, 90 deg the back of the hand, t from
    the wrist down) and `hp(l, w, b)`, a point in the hand's frame, for the caller's plates."""
    L, B = unit(L), unit(B)
    W = np.cross(L, B)  # towards the thumb
    k = scale
    fin = mirror if mirrored else (lambda o: o)

    def hp(l, w=0.0, b=0.0):
        return np.asarray(wrist, float) + k * (l * L + w * W + b * B)

    palm = Loft([dict(p=hp(l), a=a * k, b=b * k, n=2.6) for l, a, b in PALM], front=B)
    fin(loft_mesh(prefix + "_Palm", palm, res=0.003 * k, cap1=True, mat=mat, coll=coll))

    def digit(name, base, d0, length, r, curls):
        """Three phalanges, each bent `curls[i]` further towards the palm; rounded tip."""
        pts, rad = [base - 0.016 * k * d0, base.copy()], [r, 1.05 * r]
        p, ang, d = base.copy(), 0.0, d0
        for seg, c, f in zip(np.array([0.46, 0.30, 0.24]) * length, curls, (1.0, 0.94, 0.82)):
            ang += c * curl
            d = unit(d0 * math.cos(ang) - B * math.sin(ang))
            p = p + seg * d
            pts.append(p.copy())
            rad.append(f * r)
        pts += [p + 0.45 * r * d, p + 0.72 * r * d]
        rad += [0.58 * r, 0.16 * r]
        loft = Loft([dict(p=q, a=a, b=a * 0.96, n=2.0) for q, a in zip(pts, rad)], front=B)
        fin(loft_mesh(name, loft, res=0.0022 * k, cap1=True, mat=mat, coll=coll))

    for name, w, fan, length, r, curls in DIGITS:
        digit("%s_%s" % (prefix, name), hp(0.094, w, -0.002), unit(L + fan * W), length * k, r * k, curls)
    digit(prefix + "_Thumb", hp(0.028, 0.038, -0.010), unit(0.62 * L + 0.70 * W - 0.25 * B), 0.072 * k, 0.0142 * k, (0.0, 0.25, 0.22))
    return palm, hp


# Weld a LEFT half-body loft to its mirror: clamp it at the mid-plane and drop what lies beyond.
CLAMP = dict(drop=lambda P: P[:, 0] <= 1e-6, post=lambda P: np.column_stack([np.maximum(P[:, 0], 0.0), P[:, 1:]]))


def ring(z, cx, cy, a, bf, bb, n=2.0, pin=0):
    """A horizontal body section; `pin` keeps its plane level whatever the centre line does."""
    r = dict(p=(cx, cy, z), a=a, bf=bf, bb=bb, n=n)
    if pin:
        r["t"] = (0.0, 0.0, 1.0)
    return r


def rivets(name, anchor, st, off, r=0.003, mat=None, coll="Armor"):
    from kit import studs
    s, t = zip(*st)
    P, N = anchor.pn(s, t, off)
    return studs(name, P, N, r, mat=mat, coll=coll)


def helmet_dome(center, radii, z0, p=2.0, flare=0.55, n=2.1):
    """Helmet shell as a loft: a dome of profile exponent `p` (2 = ellipsoid, 3 = boxy,
    flat-topped) whose part below the centre stays wide (`flare` < 1). Returns the loft
    and `rows(res)`, row heights evenly spaced along the dome rather than in z."""
    c = np.asarray(center, float)
    A, B, CZ = radii

    def fn(t):
        z = z0 + t
        u = np.clip((z - c[2]) / CZ, -1.0, 1.0)
        k = np.maximum(1.0 - np.abs(np.where(u < 0, flare * u, u)) ** p, 4e-5 ** (p / 2.0)) ** (1.0 / p)
        m = len(t)
        return (np.column_stack([np.full(m, c[0]), np.full(m, c[1]), z]), np.tile([0.0, 0.0, 1.0], (m, 1)),
                np.column_stack([A * k, B * k, B * k, np.full(m, n), np.full(m, n)]))

    def rows(res):
        b0 = math.asin((z0 - c[2]) / CZ)
        beta = np.linspace(b0, math.pi / 2 - 0.012, int(round((math.pi / 2 - b0) * CZ / res)))
        return c[2] + CZ * np.sin(beta) - z0

    return Loft(fn=fn, length=c[2] + CZ - z0), rows


def tread(y_toe, pitch=0.026, lug=0.0035, arch_y=-0.015, arch_w=0.04, arch_h=0.011):
    """Sole displacement for a foot loft running toe -> heel: lugs cut into the side walls,
    and the arch lifted between ball and heel."""
    def f(g):
        y = g.t + y_toe
        sn = np.sin(g.phi)
        lugs = np.clip((0.5 - np.abs(sn)) / 0.15, 0, 1) * (np.clip(np.sin(2.0 * math.pi * g.t / pitch) * 3.0, -1, 1) * 0.5 + 0.5)
        arch = np.clip((-sn - 0.6) / 0.2, 0, 1) * np.clip(1.6 * (1.0 - np.abs(y - arch_y) / arch_w), 0, 1)
        return -lug * lugs - arch_h * arch
    return f
