"""A shako (Blender 4.5, numpy + bpy): the tall cylindrical cap of Napoleonic light cavalry and
infantry, from one spec table. Built for the French guard officer; another shako is another
spec (a plain infantry one simply leaves out the cords).

  shako(prefix, spec, mats, coll)   body (a loft of oval rings, flaring to the top, its section
                                    planes tilted back by `tilt`), flat top, top band, peak,
                                    front plate, cockade, cords in festoons front and back, a
                                    hanging cord with its tassel on one side. Returns the loft.
  chin_chain(prefix, head_loft, spec, mat, coll)
                                    the chin scales from the shako's sides round under the chin

spec keys:
  rings    ((z, cy, a, bf, bb), ...) bottom -> top, all centred on x = 0
  tilt     deg: the section planes lean back (the top's front edge stands higher)
  band     (height, lift): the top band, `height` down from the top edge
  peak     dict(length, droop, span deg, thick)
  plate    dict(dz from the top, hs, ht, point, lift)
  cockade  dict(dz from the top, r, rim)
  cords    dict(r, front=(dz top at the sides, dz bottom at the centre), back=(...))
  tassel   dict(side +1 his left / -1 his right, dz from the top, x, y, drop, length, r_head, r_skirt)
  chain    dict(dz at the shako, r, flat, ear_y, chin)  (chin_chain)
mats: body, gold (cords, band), metal (plate, chain), peak, cockade.
"""
import math

import numpy as np

import braid
import kit
from dresswear import grid_faces, orient, smooth_path, swept
from kit import TAU, Loft, OnLoft, new_mesh, unit

D = math.radians
Z = np.array([0.0, 0.0, 1.0])


def shako_loft(spec):
    tl = D(spec.get("tilt", 0.0))
    nrm = (0.0, math.sin(tl), math.cos(tl))
    return Loft([dict(p=(0.0, cy, z), a=a, bf=bf, bb=bb, n=2.0, t=nrm) for z, cy, a, bf, bb in spec["rings"]])


def shako(prefix, spec, mats, coll="Shako"):
    lo = shako_loft(spec)
    L = lo.L
    objs = [kit.loft_mesh(prefix + "_Body", lo, res=min(kit.RES, 0.004), cap0=True, cap1=True, mat=mats["body"], coll=coll)]
    # the top: the cap fan is flat; a gold band round the top edge
    bh, lift = spec["band"]
    objs.append(kit.band_on(prefix + "_Band", lo, L - bh, L, offset=lift, thick=lift + 0.001, mat=mats["gold"], coll=coll, res=0.003))
    # peak: a half-oval visor from the front of the bottom ring, out and down
    pk = spec["peak"]
    span = D(pk["span"])
    cols = np.linspace(D(90) - span, D(90) + span, 49)
    rows = np.linspace(0.0, 1.0, 14)
    P0, N0 = lo.pn(cols, np.full(len(cols), 0.004))
    out = unit(np.column_stack([N0[:, 0], N0[:, 1], np.zeros(len(cols))]))
    reach = pk["length"] * np.sqrt(np.clip(1.0 - ((cols - D(90)) / span) ** 2, 0.0, 1.0))
    V = (P0[None, :, :] + (rows[:, None] * reach[None, :])[..., None] * out[None, :, :]
         - (pk["droop"] * rows[:, None] ** 1.6 * (reach[None, :] / pk["length"]))[..., None] * Z)
    Vf = V.reshape(-1, 3)
    up = np.tile(Z, (len(Vf), 1))
    faces = orient(grid_faces(len(rows), len(cols)), Vf, up)
    objs.append(kit._finish(new_mesh(prefix + "_Peak", Vf, faces, None, mats["peak"], coll), pk["thick"], 0.0015, 2))
    # front plate and cockade
    pl = spec["plate"]
    t_pl = L - pl["dz"]
    objs.append(kit.plate(prefix + "_Plate", OnLoft(lo, D(90), t_pl), pl["hs"], pl["ht"], n=2.6, offset=pl["lift"], thick=0.003, dome=0.004,
                          mat=mats["metal"], coll=coll))
    ck = spec["cockade"]
    an = OnLoft(lo, D(90), L - ck["dz"])
    objs.append(kit.plate(prefix + "_Cockade", an, ck["r"], ck["r"], n=2.0, offset=lift + 0.004, thick=0.003, dome=0.002, mat=mats["cockade"], coll=coll))
    objs.append(kit.plate(prefix + "_Cockade_Rim", an, ck["r"] + ck["rim"], ck["r"] + ck["rim"], n=2.0, offset=lift + 0.003, thick=0.003, hole=0.75,
                          mat=mats["gold"], coll=coll))
    # cords: festoons across the front and the back, on the surface
    cd = spec["cords"]
    tz = lambda dz: L - dz
    for side, (dz_top, dz_low), a0, a1 in (("Front", cd["front"], 0.0, math.pi), ("Back", cd["back"], math.pi, TAU)):
        ph = np.linspace(a0 + 0.05, a1 - 0.05, 40)
        s = np.sin(np.linspace(0.0, math.pi, 40))
        t = tz(dz_top) + (tz(dz_low) - tz(dz_top)) * s ** 0.8
        P, _ = lo.pn(ph, t, cd["r"] + 0.0025)
        objs.append(braid.cord(prefix + "_Cord_" + side, P, r=cd["r"], mat=mats["gold"], coll=coll))
    # the hanging cord and its tassel on one side
    ts = spec["tassel"]
    sg = ts["side"]
    phs = 0.0 if sg > 0 else math.pi
    P, N = lo.pn([phs], [tz(ts["dz"])], cd["r"] + 0.002)
    start = P[0]
    end = np.array([sg * ts["x"], ts["y"], start[2] - ts["drop"]])
    mid = 0.5 * (start + end) + np.array([sg * 0.006, 0.0, 0.0])
    objs.append(braid.cord(prefix + "_Cord_Hang", [start, mid, end], r=cd["r"] * 0.9, mat=mats["gold"], coll=coll))
    objs.append(braid.tassel(prefix + "_Tassel", end - Z * 0.002, ts["length"], ts["r_head"], ts["r_skirt"], mat=mats["gold"], coll=coll))
    return lo, objs


def chin_chain(prefix, shako_lo, head_loft, spec, mat, coll="Shako"):
    """Gilt chin scales: from the shako's bottom ring at each side, down in front of the ear
    and round under the chin, lying `lift` off the skin (head loft) -- one closed piece."""
    ch = spec["chain"]
    t_lo = shako_lo.t_at_z(spec["rings"][0][0] + ch["dz"])
    pts = []
    zs = ch["path_z"]
    ys = ch["path_y"]
    for sgn in (1, -1):
        side = []
        P, _ = shako_lo.pn([0.0 if sgn > 0 else math.pi], [t_lo], 0.002)
        side.append(P[0])
        for z, y in zip(zs, ys):  # on the head's side at that height, nudged to the given y
            t = head_loft.t_at_z(z)
            ph = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 181) if sgn > 0 else np.linspace(0.5 * math.pi, 1.5 * math.pi, 181)
            Q, Nq = head_loft.pn(ph, np.full(181, t), ch["lift"])
            k = np.argmin(np.abs(Q[:, 1] - y))
            side.append(Q[k])
        pts.append(side)
    right = pts[1][::-1]
    chin = np.array(ch["chin"])
    path = np.array(pts[0] + [chin] + list(right))
    c = smooth_path(path, 70)
    radial = unit(c - np.array([0.0, c[:, 1].mean(), c[:, 2].mean()]))
    up = unit(np.cross(radial, np.gradient(c, axis=0)))  # across the strap: `flat` thins it towards the skin
    return swept(prefix + "_Chin_Chain", c, ch["r"], n_u=10, flat=ch["flat"], up=up, mat=mat, coll=coll,
                 disp=lambda u, al: 0.25 * ch["r"] * np.abs(np.sin(u * 60.0 * math.pi)))
