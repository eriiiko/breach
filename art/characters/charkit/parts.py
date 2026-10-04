"""Body parts shared between characters, built from the kit."""
import math

import numpy as np

from kit import Loft, loft_mesh, mirror, smax, unit

# Fingers at scale 1 (a large gloved hand): name, position across the palm, fan,
# length, radius, curl of each joint in radians (relaxed).
DIGITS = (
    ("Index", 0.033, 0.07, 0.068, 0.0116, (0.20, 0.35, 0.30)),
    ("Middle", 0.011, 0.01, 0.075, 0.0120, (0.26, 0.42, 0.32)),
    ("Ring", -0.011, -0.05, 0.070, 0.0114, (0.30, 0.45, 0.32)),
    ("Pinky", -0.033, -0.11, 0.055, 0.0102, (0.34, 0.48, 0.32)))
# Palm sections, wrist to knuckles: distance down the hand, half-width, half-thickness.
PALM = ((-0.020, .040, .034), (0.012, .042, .028), (0.042, .048, .023), (0.072, .050, .020), (0.092, .048, .017), (0.102, .040, .010))


def hand(prefix, wrist, L, B, mat, coll, scale=1.0, curl=1.0, mirrored=True, girth=1.0, palm_girth=1.0, palm=PALM):
    """A gloved LEFT hand with separate fingers, mirrored to the right by default.

    `L` is the unit vector down the fingers, `B` out of the back of the hand. Returns
    `(palm, hp)`: the palm loft (phi = 0 thumb side, 90 deg the back of the hand, t from
    the wrist down) and `hp(l, w, b)`, a point in the hand's frame, for the caller's plates.
    `girth` scales the finger radii, `palm_girth` the palm's thickness and `palm` replaces the palm
    section table (bare hands: a narrower wrist); a bare hand is
    about 0.8 / 0.85 of the gloved default (1.0 / 1.0 is the original gloved hand)."""
    L, B = unit(L), unit(B)
    W = np.cross(L, B)  # towards the thumb
    k = scale
    fin = mirror if mirrored else (lambda o: o)

    def hp(l, w=0.0, b=0.0):
        return np.asarray(wrist, float) + k * (l * L + w * W + b * B)

    palm = Loft([dict(p=hp(l), a=a * k, b=b * k * palm_girth, n=2.6) for l, a, b in palm], front=B)
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
        digit("%s_%s" % (prefix, name), hp(0.094, w, -0.002), unit(L + fan * W), length * k, r * k * girth, curls)
    digit(prefix + "_Thumb", hp(0.028, 0.038, -0.010), unit(0.62 * L + 0.70 * W - 0.25 * B), 0.072 * k, 0.0142 * k * girth, (0.0, 0.25, 0.22))
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


# ======================================================================== heads
# A bare human head as one closed loft from inside the collar to the crown, with the
# face authored as a displacement field, plus separate ears, eyeballs and hair. The
# spec is a plain dict (see `head()`); a different head is a different spec.

def rows_by_arclength(loft, res, t0=0.0, t1=None, phis=None, n=600):
    """Row parameters spaced evenly in SURFACE arclength (the worst meridian), so a dome's
    flat top or a chin's underside gets as many rows as its steep sides."""
    t1 = loft.L if t1 is None else t1
    ts = np.linspace(t0, t1, n)
    phis = np.radians(np.arange(0.0, 360.0, 22.5)) if phis is None else phis
    P = np.stack([loft.pos(np.full(n, p), ts) for p in phis])
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2).max(axis=0)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    m = max(2, int(round(s[-1] / res)))
    return np.interp(np.linspace(0.0, s[-1], m + 1), s, ts)


def phi_at_x(loft, t, x, front=True):
    """The section angle on the front (or back) half of a loft where the surface reaches world x."""
    lo, hi = (0.0, math.pi) if front else (math.pi, 2.0 * math.pi)
    ph = np.linspace(lo, hi, 721)
    X = loft.pos(ph, np.full(ph.shape, t))[:, 0]
    order = np.argsort(X) if X[0] < X[-1] else np.argsort(X)
    return float(np.interp(x, X[order], ph[order]))


def gauss2(x, z, x0, z0, sx, sz):
    return np.exp(-((x - x0) / sx) ** 2 - ((z - z0) / sz) ** 2)


# The face's features at their default size: (x0, dz from its landmark, sx, sz, height).
# `x0 > 0` features are mirrored (they act on |x|). Heights in metres along the normal.
FACE = dict(
    # a brow RIDGE the upper lid sits in under, and a soft fold between the two
    brow=((0.029, ("eye", 0.0150), 0.022, 0.0065, 0.0070), (0.0, ("eye", 0.012), 0.012, 0.0080, 0.0035)),
    lid_fold=((0.031, ("eye", 0.0072), 0.013, 0.0022, -0.0012),),
    # the socket: wide and shallow, deepest at the corners, so the skin meets the lids near
    # the opening instead of leaving the eyeball standing out of a pit
    socket=((0.032, ("eye", 0.0), 0.0180, 0.0068, -0.0068),),
    lid=((0.032, ("eye", -0.0105), 0.0140, 0.0030, 0.0012),),
    cheekbone=((0.050, ("eye", -0.021), 0.017, 0.012, 0.0070),),
    hollow=((0.050, ("mouth", 0.009), 0.015, 0.015, -0.0030),),
    temple=((0.072, ("eye", 0.020), 0.012, 0.016, -0.0025),),
    lip_upper=((0.0, ("mouth", 0.0045), 0.022, 0.0040, 0.0042),),
    lip_lower=((0.0, ("mouth", -0.0060), 0.019, 0.0046, 0.0048),),
    mouth=((0.0, ("mouth", 0.0), 0.026, 0.0015, -0.0035), (0.027, ("mouth", 0.0), 0.006, 0.006, -0.0020)),
    philtrum=((0.0, ("mouth", 0.012), 0.0045, 0.006, -0.0012),),
    chin=((0.0, ("chin", 0.010), 0.022, 0.011, 0.0090), (0.0, ("mouth", -0.016), 0.018, 0.004, -0.0025)),
    # nostril wings either side of the tip, the crease round them, and the nostrils
    alae=((0.0150, ("nose", 0.0035), 0.0062, 0.0060, 0.0085),),
    alar_crease=((0.0230, ("nose", 0.0060), 0.0030, 0.0070, -0.0022), (0.0175, ("nose", 0.0105), 0.0060, 0.0025, -0.0015)),
    nostril=((0.0080, ("nose", 0.0005), 0.0035, 0.0022, -0.0020),),
)

# Nose defaults (spec key `nose`): tip dome radii across / up-down as fractions of the
# projection, how far down the ridge reaches towards the tip, the bridge's width.
NOSE = dict(tip_rx=0.42, tip_rz=0.45, tip_round=0.35, ridge=0.82, bridge=0.0060, tip_w=0.0070)


def face_field(spec):
    """Displacement (m) and masks over the head loft's grid from a head spec."""
    fz = dict(eye=spec["eye_z"], mouth=spec["mouth_z"], chin=spec["chin_z"], nose=spec["nose_z"])
    feats = dict(FACE, **spec.get("features", {}))
    nose = dict(NOSE, **spec.get("nose", {}))
    proj, z_tip, z_nasion = spec["nose_proj"], spec["nose_z"] + 0.010, spec["eye_z"] + 0.007
    cy = spec["rings"][0][1]

    def basis(g):
        x, z = g.P0[..., 0], g.P0[..., 2]
        frontish = np.clip((-g.N0[..., 1] - 0.15) / 0.45, 0.0, 1.0)
        return x, np.abs(x), z, frontish

    def disp(g):
        x, ax, z, fr = basis(g)
        d = np.zeros(x.shape)
        for items in feats.values():
            for x0, (lm, dz), sx, sz, h in items:
                d += h * gauss2(ax, z, x0, fz[lm] + dz, sx, sz)
        # nose: a bridge from the nasion that widens and rises towards the tip, and a ROUND
        # tip (a dome, not the bridge's sharp end) whose underside curves back to the lip
        s = np.clip((z_nasion - z) / (z_nasion - z_tip), 0.0, 1.0)
        h = proj * nose["ridge"] * (0.25 + 0.75 * s ** 1.5)
        h = np.where(z < z_tip, proj * nose["ridge"] * np.exp(-((z - z_tip) / 0.0055) ** 2), h)
        h = np.where(z > z_nasion, proj * nose["ridge"] * 0.25 * np.exp(-((z - z_nasion) / 0.010) ** 2), h)
        sx = nose["bridge"] + nose["tip_w"] * s
        ridge = h * np.exp(-(x / sx) ** 2)
        rx, rz = nose["tip_rx"] * proj, nose["tip_rz"] * proj
        q = np.clip(1.0 - (x / rx) ** 2 - ((z - z_tip + 0.15 * rz) / rz) ** 2, 0.0, 1.0)
        dome = proj * q ** nose.get("tip_round", 0.35)  # a blunt dome: the tip's radius ~ rz^2 / (0.7 proj)
        d += smax(ridge, dome, 0.004)
        d = d * fr
        if "jaw" in spec:  # the jaw: a rounded edge of the mandible, then the surface turns in to the neck
            yf, zf, ya, za, depth = spec["jaw"]
            y = g.P0[..., 1]
            zl = zf + (za - zf) * np.clip((y - yf) / (ya - yf), 0.0, 1.0)
            u = np.clip((zl - z) / spec.get("jaw_soft", 0.022), 0.0, 1.0)
            under = u * u * (3.0 - 2.0 * u) * np.clip((z - (zf - 0.06)) / 0.02, 0.0, 1.0)
            behind = np.clip((y - (ya + 0.02)) / -0.015, 0.0, 1.0)
            side = np.clip((ax - 0.016) / 0.03, 0.0, 1.0) * behind
            edge = np.exp(-((z - zl - 0.004) / 0.008) ** 2)  # the jaw's own rounded edge, just above the line
            d += depth * (0.30 * edge - under) * side
        return d

    def masks(g):
        x, ax, z, fr = basis(g)
        mz, cz, ez = fz["mouth"], fz["chin"], fz["eye"]
        lips = np.clip(1.6 * (gauss2(x, z, 0.0, mz + 0.0045, 0.020, 0.0042) + gauss2(x, z, 0.0, mz - 0.006, 0.018, 0.0050)), 0, 1) * fr
        side = np.clip((g.P0[..., 1] - cy - 0.035) / -0.03, 0.0, 1.0)  # in front of the ears
        beard = np.clip((ez - 0.026 - z) / 0.012, 0.0, 1.0) * side
        beard = np.maximum(beard, np.clip((z - (cz - 0.040)) / 0.01, 0, 1) * np.clip((cz + 0.012 - z) / 0.01, 0, 1)
                           * np.clip((g.P0[..., 1] - cy + 0.0) / -0.02, 0, 1))  # under the jaw
        beard = beard * (1.0 - lips) * (1.0 - np.clip(gauss2(x, z, 0.0, spec["nose_z"] + 0.012, 0.016, 0.014) * 2.0, 0, 1))
        beard = beard * spec.get("stubble", 1.0)
        # the brows: `brow` = dict(x, dz, sx, sz, arch, gain); `arch` lifts the middle of each brow
        bw = dict(dict(x=0.030, dz=0.0155, sx=0.021, sz=0.0042, arch=0.0, gain=2.2), **spec.get("brow", {}))
        zb = ez + bw["dz"] + bw["arch"] * np.exp(-((ax - bw["x"] + 0.004) / (0.8 * bw["sx"])) ** 2)
        brow = np.clip(bw["gain"] * gauss2(ax, z - zb, bw["x"], 0.0, bw["sx"], bw["sz"]) - 0.35, 0.0, 1.0) * fr
        nostril = np.clip(3.0 * gauss2(ax, z, 0.009, spec["nose_z"] + 0.002, 0.004, 0.0028) - 0.5, 0, 1) * fr
        return dict(lips=lips, stubble=beard, brow=brow, nostril=nostril)

    return disp, masks


def head(prefix, spec, mats, coll="Head", res=None):
    """Head + neck (one closed loft, cap at the crown, closed inside the collar), two ears and
    two eyeballs. Returns `(loft, objects)`; the loft is what `hair()` and the caller's
    collar fit against. `mats` = dict(skin=, eye=).

    spec keys:
      rings     ((z, cy, a, bf, bb, n), ...) neck bottom -> top of the cranium's side; the
                first ring sits INSIDE the collar, so no seam ever shows
      crown     (z_top, n): the dome closing the last ring, top height and squareness
      eye_z, nose_z (base of the nose), mouth_z, chin_z: landmark heights
      nose_proj  how far the nose tip stands off the face
      eye_dx, eye_r, eye_sink: eye spacing (half), eyeball radius, how deep it sits
      ear       dict(z, y, h, w, tilt, flare): lobe-to-top height, width, tilt back, flare out (deg)
      features  optional overrides of `FACE` rows
    """
    import kit
    from kit import Loft, loft_mesh

    res = res or min(kit.RES * 0.6, 0.0022)  # the face's features are a few mm: never coarser, even in drafts
    rings = [dict(p=(0.0, cy, z), a=a, bf=bf, bb=bb, n=n, t=(0.0, 0.0, 1.0)) for z, cy, a, bf, bb, n in spec["rings"]]
    z_top, n_top = spec["crown"]
    z0, cy0, a0, bf0, bb0, nn = spec["rings"][-1]
    for beta in np.radians([14, 28, 42, 56, 68, 78, 85, 89.2]):
        k, z = math.cos(beta) ** (2.0 / n_top), z0 + (z_top - z0) * math.sin(beta)
        rings.append(dict(p=(0.0, cy0, z), a=a0 * k, bf=bf0 * k, bb=bb0 * k, n=nn, t=(0.0, 0.0, 1.0)))
    loft = Loft(rings)
    disp, masks = face_field(spec)

    def base(g):
        g.P0, g.N0 = g.P, g.N
        return g

    def head_disp(g):
        return disp(base(g))

    attrs = {k: (lambda g, k=k: masks(base(g))[k]) for k in ("lips", "stubble", "brow", "nostril")}
    objs = [loft_mesh(prefix + "_Head", loft, rows=rows_by_arclength(loft, res), res=res, disp=head_disp, attrs=attrs,
                      cap0=True, cap1=True, mat=mats["skin"], coll=coll)]

    # eyeballs, set into the sockets on the undisplaced surface
    t_eye = loft.t_at_z(spec["eye_z"])
    for sgn, side in ((1, "L"), (-1, "R")):
        ph = phi_at_x(loft, t_eye, sgn * spec["eye_dx"])
        P, N = loft.pn([ph], [t_eye])
        gaze = unit(np.array([0.0, -1.0, 0.0]) + 0.08 * N[0])
        c = P[0] - N[0] * (spec["eye_r"] - spec.get("eye_sink", 0.0))
        lids = dict(spec.get("lids", dict(open_w=1.20, upper=0.30, lower=0.44)), mat=mats["skin"])
        objs += list(eyeball("%s_Eye_%s" % (prefix, side), c, spec["eye_r"], gaze, mats["eye"], coll, n_lat=40, n_lon=64, lids=lids))
    objs += ear(prefix, loft, spec["ear"], mats["skin"], coll)
    return loft, objs


def eyeball(name, c, r, gaze, mat, coll, n_lat=24, n_lon=40, lids=None):
    """A sphere whose `iris` attribute is the cosine to the gaze direction (1 at the pupil)."""
    from kit import new_mesh
    gaze = unit(gaze)
    ref = np.array([0.0, 0.0, 1.0]) if abs(gaze[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(ref, gaze))
    e2 = np.cross(gaze, e1)
    th = np.linspace(0.0, math.pi, n_lat + 1)[1:-1]
    al = np.linspace(0.0, 2 * math.pi, n_lon, endpoint=False)
    TH, AL = np.meshgrid(th, al, indexing="ij")
    D = (np.cos(TH)[..., None] * gaze + (np.sin(TH) * np.cos(AL))[..., None] * e1 + (np.sin(TH) * np.sin(AL))[..., None] * e2).reshape(-1, 3)
    V = np.vstack([np.asarray(c) + r * D, np.asarray(c) + r * gaze, np.asarray(c) - r * gaze])
    R, C = len(th), n_lon
    idx = np.arange(R * C).reshape(R, C)
    nxt = np.roll(idx, -1, axis=1)
    quads = np.stack([idx[:-1], nxt[:-1], nxt[1:], idx[1:]], axis=-1).reshape(-1, 4)
    front, back = R * C, R * C + 1
    tris = np.vstack([np.stack([np.full(C, front), nxt[0], idx[0]], axis=-1), np.stack([np.full(C, back), idx[-1], nxt[-1]], axis=-1)])
    iris = np.concatenate([D @ gaze, [1.0, -1.0]])
    eye = new_mesh(name, V, quads, tris, mat, coll, attrs=dict(iris=iris))
    if lids is None:
        return eye
    # eyelids: a skin shell a little larger than the eyeball, open in an almond round the
    # gaze, its edge an exact curve (not the sphere's grid). lids = dict(mat=, open_w=,
    # upper=, lower=, gap=, thick=): the opening's half-width and its upper / lower edge
    # above / below the gaze (radians), the shell's clearance (fraction of r) and thickness.
    # An upper edge below the iris's top angle makes the lid cover the top of the iris.
    up = np.asarray(lids.get("up", (0.0, 0.0, 1.0)), float)
    if e2 @ up < 0:
        e1, e2 = -e1, -e2
    w, ku, kl = lids["open_w"], lids["upper"], lids["lower"]
    C = 96
    al = np.linspace(0.0, 2 * math.pi, C, endpoint=False)
    sa = np.sin(al)
    ah = w * np.cos(al)
    av = np.where(sa >= 0, ku * np.abs(sa) ** 1.2, -kl * np.abs(sa) ** 1.5)
    De = (np.cos(av) * np.sin(ah))[:, None] * e1 + np.sin(av)[:, None] * e2 + (np.cos(av) * np.cos(ah))[:, None] * gaze
    th_e = np.arccos(np.clip(De @ gaze, -1.0, 1.0))
    psi = np.arctan2(De @ e2, De @ e1)
    v = np.linspace(0.0, 1.0, 12) ** 1.4
    TH = th_e[None, :] + (math.radians(118.0) - th_e[None, :]) * v[:, None]
    PS = np.broadcast_to(psi[None, :], TH.shape)
    DL = (np.cos(TH)[..., None] * gaze + (np.sin(TH) * np.cos(PS))[..., None] * e1 + (np.sin(TH) * np.sin(PS))[..., None] * e2)
    rl = r * (1.0 + lids.get("gap", 0.10))
    VL = np.asarray(c) + rl * DL.reshape(-1, 3)
    idx = np.arange(len(v) * C).reshape(len(v), C)
    nx = np.roll(idx, -1, axis=1)
    fq = np.stack([idx[:-1], idx[1:], nx[1:], nx[:-1]], axis=-1).reshape(-1, 4)
    f0 = fq[C // 4]
    nrm = np.cross(VL[f0[1]] - VL[f0[0]], VL[f0[3]] - VL[f0[0]])
    if nrm @ (VL[f0[0]] - np.asarray(c)) < 0:
        fq = fq[:, ::-1]
    lid = new_mesh(name + "_Lids", VL, fq, None, lids["mat"], coll)
    kit_finish = __import__("kit")._finish
    kit_finish(lid, lids.get("thick", 0.0014), None, 2)
    return eye, lid


# Ear outline along its height, 0 = bottom of the lobe: (fraction of height, half-width
# front-back, outward thickness, inward thickness), at a 62 mm ear.
EAR_PROFILE = ((0.0, .003, .002, .002), (0.08, .009, .0045, .004), (0.24, .0115, .0055, .0045), (0.42, .0135, .0065, .0055),
               (0.62, .0155, .0075, .0060), (0.80, .0145, .0075, .0055), (0.92, .0100, .0060, .0045), (1.0, .0025, .0025, .002))


def ear(prefix, head_loft, spec, mat, coll):
    """Left ear on the head's side (mirrored to the right): a flat loft tilted back and flared
    out, its outer face dished (concha) inside a raised rim (helix)."""
    from kit import Loft, loft_mesh, mirror
    k = spec["h"] / 0.062
    kw = spec["w"] / 0.031
    tilt, flare = math.radians(spec.get("tilt", 15.0)), math.radians(spec.get("flare", 20.0))
    z0 = spec["z"]
    t_mid = head_loft.t_at_z(z0 + 0.5 * spec["h"])
    P, _ = head_loft.pn([0.0], [t_mid])  # phi 0: the head's left side
    root = np.array([P[0][0] - spec.get("sink", 0.004), spec["y"], z0])
    up = np.array([0.0, math.sin(tilt), math.cos(tilt)])
    out = unit(np.array([math.cos(flare), math.sin(flare), 0.0]))
    out = unit(out - (out @ up) * up)
    rings = [dict(p=root + f * spec["h"] * up + (bb * k) * out, a=a * kw, bf=bf * k, bb=bb * k, n=2.2) for f, a, bf, bb in EAR_PROFILE]
    loft = Loft(rings, front=out)

    def dish(g):
        on_out = np.clip(np.sin(g.phi) * 2.0, 0.0, 1.0)  # the outer face
        f = g.t / loft.L
        mid = np.clip(1.0 - np.abs(np.cos(g.phi)) * 1.6, 0.0, 1.0)  # 0 on the rim, 1 mid-face
        return -on_out * 0.0045 * k * np.exp(-((f - 0.42) / 0.20) ** 2) * mid ** 1.5

    objs = [mirror(loft_mesh(prefix + "_Ear", loft, res=0.0018, disp=dish, cap0=True, cap1=True, mat=mat, coll=coll))]
    return objs


def hair(prefix, head_loft, spec, mat, coll="Head", res=None, seed=7):
    """Short hair as a separate closed shell on the head loft. Below the hairline the shell
    dives under the skin, so the visible hairline is a clean intersection, never an edge.

    spec keys:
      line      ((phi_deg, z), ...) the hairline height around the head (phi 0 = left side,
                90 = front, 270 = back), interpolated periodically
      thick     (at the hairline, on the crown) shell thickness over the skull
      tufts     number of clumps; tuft_h (lo, hi) height; tuft_size (lo, hi)
    """
    import kit
    from kit import loft_mesh, solid, wrap

    res = res or min(kit.RES * 0.6, 0.003)
    cy = head_loft.Pk[0][1]
    lp = np.array([p for p, _ in spec["line"]], float)
    lz = np.array([z for _, z in spec["line"]], float)
    order = np.argsort(lp)
    lp, lz = lp[order], lz[order]
    lp_ext = np.concatenate([lp - 360.0, lp, lp + 360.0])
    lz_ext = np.concatenate([lz, lz, lz])

    def line_z(phi_rad):
        return np.interp(np.degrees(np.mod(phi_rad, 2 * math.pi)), lp_ext, lz_ext)

    z_min = lz.min() - 0.02
    t0 = head_loft.t_at_z(z_min)
    rng = np.random.default_rng(seed)
    n = spec.get("tufts", 260)
    tz_lo = lz.min()
    tp = rng.uniform(0, 2 * math.pi, n)
    tt = np.array([head_loft.t_at_z(z) for z in rng.uniform(tz_lo, spec["crown_z"] - 0.004, n)])
    tc = head_loft.pos(tp, tt)
    flow = unit(head_loft.pos(tp, np.minimum(tt + 0.004, head_loft.L)) - head_loft.pos(tp, np.maximum(tt - 0.004, 0.0)))
    if "flow_to" in spec:  # hair drawn back towards one point (a bun): the flow runs along the scalp towards it
        nrm = head_loft.pn(tp, tt)[1]
        to = np.asarray(spec["flow_to"], float) - tc
        flow = unit(to - np.sum(to * nrm, axis=1)[:, None] * nrm)
    flow = unit(flow + rng.normal(0.0, spec.get("flow_noise", 0.35), flow.shape))
    tufts = dict(c=tc, f=flow, h=rng.uniform(*spec.get("tuft_h", (0.002, 0.006)), n), s=rng.uniform(*spec.get("tuft_size", (0.008, 0.018)), n))
    thin, thick = spec["thick"]

    def zof(g):  # height of the skull surface under the grid (g.P is not set yet when `offset` runs)
        return head_loft.pos(g.phi.ravel(), g.t.ravel())[:, 2].reshape(g.phi.shape)

    def cover(g):
        # distance to the hairline measured across it, not just in z, so a steep stretch
        # (the back of a sideburn) fades as softly as a level one
        h = 0.01
        slope = (line_z(g.phi + h) - line_z(g.phi - h)) / (2.0 * h * np.maximum(g.r, 0.02))
        return np.clip((zof(g) - line_z(g.phi)) / (spec.get("fade", 0.006) * np.sqrt(1.0 + slope ** 2)), 0.0, 1.0)

    def offset(g):
        m = cover(g)
        if "thick_z" in spec:  # thin over the ears and nape, full on the crown
            z_lo, z_hi = spec["thick_z"]
            grow = np.clip((zof(g) - z_lo) / (z_hi - z_lo), 0.0, 1.0)
        else:
            grow = np.clip((zof(g) - line_z(g.phi)) / 0.05, 0.0, 1.0)
        return (thin + (thick - thin) * grow) * m - 0.0025 * (1.0 - m)

    def disp(g):
        m = cover(g)
        z = zof(g)
        d = np.zeros(z.shape)
        P = head_loft.pos(g.phi.ravel(), g.t.ravel())
        for i in range(n):  # clumps in 3-D (no streaking at the crown), long along the hair's flow
            dv = P - tufts["c"][i]
            u = dv @ tufts["f"][i]
            w2 = np.einsum("ij,ij->i", dv, dv) - u * u
            s = tufts["s"][i]
            d += (tufts["h"][i] * np.exp(-(u / s) ** 2 - w2 / (0.40 * s) ** 2)).reshape(z.shape)
        return d * m

    def drop(P):
        phi = np.arctan2(-(P[:, 1] - cy), P[:, 0])
        return P[:, 2] < line_z(phi) - 0.012

    # stop at the last ring but one: at the pole the loft's normal is ill-defined (a spike)
    rows = rows_by_arclength(head_loft, res, t0=t0, t1=head_loft.t_ring(len(head_loft.tk) - 2))
    obj = loft_mesh(prefix + "_Hair", head_loft, rows=rows, res=res, offset=offset, disp=disp, drop=drop, cap1=True, mat=mat, coll=coll,
                    attrs=dict(cover=cover))
    return solid(obj, spec.get("shell", 0.002), bevel=0.0, seg=1)


def _blob(name, c, axes, radius, mat, coll, n_lat=40, n_lon=64, attrs=None):
    """A closed sphere-topology mesh: `radius(D)` (m) along each unit direction D in the
    frame `axes` = (e1, e2, e3), e3 the pole. `attrs` = {name: f(D)}."""
    from kit import new_mesh
    e1, e2, e3 = (np.asarray(a, float) for a in axes)
    th = np.linspace(0.0, math.pi, n_lat + 1)[1:-1]
    al = np.linspace(0.0, 2 * math.pi, n_lon, endpoint=False)
    TH, AL = np.meshgrid(th, al, indexing="ij")
    Dl = np.stack([np.sin(TH) * np.cos(AL), np.sin(TH) * np.sin(AL), np.cos(TH)], axis=-1).reshape(-1, 3)
    Dl = np.vstack([Dl, [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    R = np.asarray(radius(Dl), float)
    V = np.asarray(c, float) + (Dl[:, :1] * e1 + Dl[:, 1:2] * e2 + Dl[:, 2:] * e3) * R[:, None]
    nr, C = len(th), n_lon
    idx = np.arange(nr * C).reshape(nr, C)
    nxt = np.roll(idx, -1, axis=1)
    quads = np.stack([idx[:-1], idx[1:], nxt[1:], nxt[:-1]], axis=-1).reshape(-1, 4)
    top, bot = nr * C, nr * C + 1
    tris = np.vstack([np.stack([np.full(C, top), idx[0], nxt[0]], axis=-1), np.stack([np.full(C, bot), nxt[-1], idx[-1]], axis=-1)])
    va = {k: np.asarray(f(Dl), float) for k, f in (attrs or {}).items()}
    return new_mesh(name, V, quads, tris, mat, coll, attrs=va)


def hair_bun(prefix, spec, mat, coll="Head"):
    """A bun of coiled hair: an ellipsoid (`r` = half-sizes across, along the axis, up)
    centred at `c`, its pole along `axis` (pointing out of the head), with `coils` turns of
    hair spiralling in to the pole, each a rounded ridge `depth` high. One closed object."""
    from kit import unit as _u
    c = np.asarray(spec["c"], float)
    e3 = _u(np.asarray(spec["axis"], float))
    e1 = _u(np.cross(np.array([0.0, 0.0, 1.0]), e3)) if abs(e3[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(e3, e1)
    ra, rc, rb = spec["r"]  # across, up, along the axis
    k, h = spec.get("coils", 6), spec.get("depth", 0.003)

    def radius(Dl):
        x, y, z = Dl[:, 0], Dl[:, 1], Dl[:, 2]
        base = 1.0 / np.sqrt((x / ra) ** 2 + (y / rc) ** 2 + (z / rb) ** 2)
        th = np.arccos(np.clip(z, -1.0, 1.0))
        al = np.arctan2(y, x)
        coil = np.abs(np.sin(0.5 * (al + k * 2.0 * th)))  # a spiral groove between rounded strands
        fine = 0.25 * np.sin(al * 23.0 + th * 41.0)
        return base + h * (coil ** 0.6 - 0.6 + 0.3 * fine) * np.clip(th / 0.3, 0.0, 1.0)

    return _blob(prefix + "_Hair_Bun", c, (e1, e2, e3), radius, mat, coll, n_lat=48, n_lon=96, attrs=dict(cover=lambda Dl: np.ones(len(Dl))))


def hair_strands(prefix, head_loft, rows, mat, coll="Head", seed=11):
    """Loose strands falling from the hairline: each row (phi deg, root z, length, sway out,
    forward) is a small clump of three tapered tubes starting under the hair and hanging
    down beside the face, curving out and in. Closed objects (capped ends)."""
    from kit import new_mesh, unit as _u
    import garment
    rng = np.random.default_rng(seed)
    objs = []
    for i, (phd, z0, length, out, fwd) in enumerate(rows):
        ph = math.radians(phd)
        P0, N0 = head_loft.pn([ph], [head_loft.t_at_z(z0)])
        P0, N0 = P0[0], N0[0]
        side = _u(np.array([N0[0], N0[1], 0.0]))
        for j in range(3):
            u = np.linspace(0.0, 1.0, 22)
            jit = rng.normal(0.0, 0.002, 3)
            sway = np.sin(np.pi * u) * (out + jit[0]) + u * 0.004
            pts = (P0 - 0.002 * N0)[None, :] + side[None, :] * (0.003 + sway)[:, None]                 + np.array([0.0, fwd + jit[1], 0.0])[None, :] * u[:, None] ** 2 + np.array([0.0, 0.0, -1.0])[None, :] * (length * (1.0 + 0.15 * jit[2] / 0.002) * u)[:, None]
            pts[:, 0] += jit[1] * u
            r = 0.0016 * (1.0 - 0.7 * u)
            T = _u(np.gradient(pts, axis=0))
            ref = np.where(np.abs(T[:, 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
            Nn = _u(np.cross(ref, T))
            Bn = np.cross(T, Nn)
            al = np.linspace(0.0, 2 * math.pi, 8, endpoint=False)
            V = pts[:, None, :] + r[:, None, None] * (np.cos(al)[None, :, None] * Nn[:, None, :] + np.sin(al)[None, :, None] * Bn[:, None, :])
            idx = np.arange(len(pts) * 8).reshape(len(pts), 8)
            nx = np.roll(idx, -1, axis=1)
            faces = np.stack([idx[:-1], nx[:-1], nx[1:], idx[1:]], axis=-1).reshape(-1, 4)
            objs.append(new_mesh("%s_Hair_Strand_%d_%d" % (prefix, i, j), V.reshape(-1, 3), faces, None, mat, coll,
                                 attrs=dict(cover=np.ones(V.shape[0] * V.shape[1]))))
    garment.close_holes(objs)
    return objs
