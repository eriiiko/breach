"""Render-only arm prototype: ordinary tube/cups and fixed-length curve poses.

Coordinates are glTF / raylib: X along the rest arm, Y up (oral face), Z across.
No Blender, game, physics, or saved animation is needed to evaluate a pose.
The specimen is displayed oral-side up so the two sucker rows are inspectable.
"""
from __future__ import annotations

import math
import numpy as np

LENGTH = 1.20
SEGMENTS = 32
STEP = LENGTH / SEGMENTS
BASE = np.array([0., .15, 0.])
CONTACT_X = 1.0
PERIOD = 9.0
PALETTE = {
    "dorsal": (.38, .105, .045), "mottle": (.65, .31, .16),
    "oral": (.86, .69, .51), "rim": (.96, .86, .69),
    "bowl": (.60, .36, .20), "floor": (.052, .067, .080),
    "target": (.09, .29, .32),
}


def unit(v):
    return v / np.linalg.norm(v)


def radius(s):
    """Muscular base, sustained taper, rounded final 1 cm (zero at tip)."""
    s = np.asarray(s)
    r = .088 * (1 - s) ** 1.15 + .0035
    return r * np.sqrt(np.clip((1 - s) / .009, 0, 1))


def weights_at(x):
    f = np.clip(np.asarray(x) / STEP, 0, SEGMENTS)
    a = np.minimum(f.astype(int), SEGMENTS - 1)
    t = f - a
    return np.stack([a, a + 1], -1), np.stack([1 - t, t], -1)


def cycle_weights(seconds):
    """C1 continuous rest -> curl -> contact -> rest, with short holds."""
    phase = (seconds % PERIOD) / (PERIOD / 3)
    i = int(phase)
    t = np.clip((phase - i - .14) / .72, 0, 1)
    t = t * t * (3 - 2 * t)
    states = [(0., 0.), (1., 0.), (0., 1.), (0., 0.)]
    return tuple((1 - t) * a + t * b for a, b in zip(states[i], states[i + 1]))


def _tangents(curl, contact, reach_angle):
    s = (np.arange(SEGMENTS) + .5) / SEGMENTS
    heading = curl * 5.35 * s ** 1.45 + contact * reach_angle * np.sin(np.pi * s)
    elevation = curl * .40 * np.sin(np.pi * s) + contact * .42 * np.sin(2 * np.pi * s)
    return np.stack([np.cos(elevation) * np.cos(heading), np.sin(elevation),
                     np.cos(elevation) * np.sin(heading)], -1)


def _contact_angle():
    lo, hi = 0., 1.8
    for _ in range(42):
        mid = (lo + hi) / 2
        x = _tangents(0., 1., mid)[:, 0].sum() * STEP
        if x > CONTACT_X:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


REACH_ANGLE = _contact_angle()


def pose(curl=0., contact=0.):
    """Absolute head positions and deformation rotations, one per joint.

    Integrate fixed-length unit tangent samples. Transport the oral axis by
    the shortest rotation between successive tangents, then apply authored
    axial twist. No Frenet normal (and thus no inflection sign flip).
    Supported control domain is curl >= 0, contact >= 0, curl+contact <= 1.
    """
    if min(curl, contact) < 0 or curl + contact > 1.00000001:
        raise ValueError("controls must lie in the rest/curl/contact triangle")
    tangents = _tangents(curl, contact, REACH_ANGLE)
    points = np.vstack([BASE, BASE + np.cumsum(tangents * STEP, axis=0)])
    # Each head frame aims down its outgoing segment; terminal frame repeats.
    tangents = np.vstack([tangents, tangents[-1]])
    frames = np.empty((SEGMENTS + 1, 3, 3))
    prev, oral = (1., 0., 0.), (0., 1., 0.)

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    for i, tangent in enumerate(tangents):
        tangent = tuple(float(x) for x in tangent)
        axis = cross(prev, tangent)
        cosine = dot(prev, tangent)
        if cosine < -.99:
            raise ValueError("unsupported near-180-degree adjacent bend")
        first = cross(axis, oral)
        second = cross(axis, first)
        oral = tuple(oral[k] + first[k] + second[k] / (1 + cosine) for k in range(3))
        projection = dot(oral, tangent)
        oral = tuple(oral[k] - projection * tangent[k] for k in range(3))
        norm = math.sqrt(dot(oral, oral))
        oral = tuple(x / norm for x in oral)
        across = cross(tangent, oral)
        s = i / SEGMENTS
        twist = -.95 * curl * s - 1.1 * contact * np.sin(np.pi * s)
        c, s = math.cos(twist), math.sin(twist)
        twisted = tuple(c * oral[k] + s * across[k] for k in range(3))
        frames[i] = np.array([tangent, twisted, cross(tangent, twisted)]).T
        prev = tangent
    return points, frames


def skin(vertices, joints, weights, points, frames):
    rest = BASE + np.column_stack([np.arange(SEGMENTS + 1) * STEP,
                                  np.zeros(SEGMENTS + 1), np.zeros(SEGMENTS + 1)])
    local = vertices[:, None, :] - rest[joints]
    deformed = np.einsum("nkij,nkj->nki", frames[joints], local) + points[joints]
    return (deformed * weights[..., None]).sum(axis=1)


def quaternion(matrix):
    """Unit quaternion in raylib xyzw order (handles rotations over 180 deg)."""
    m = matrix
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, .25 * s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = [0., 0., 0., (m[k, j] - m[j, k]) / s]
        q[i], q[j], q[k] = .25 * s, (m[j, i] + m[i, j]) / s, (m[k, i] + m[i, k]) / s
    return np.array(q) / np.linalg.norm(q)


def geometry():
    """Closed skin and closed hollow cups. Cup weights equal their attachment.

    Cups use the local two-joint deformation at their centre, preventing
    detached/fanned lips; they remain separate editable islands in one mesh.
    Returns per-corner UVs in a tiny shared atlas (tube lower 3/4, cup upper).
    """
    verts, faces, uvfaces, attachment, groups = [], [], [], [], []
    sectors, rings = 24, 129
    svals = np.linspace(0, 1, rings)
    # Exclude zero-radius last ring, close with a single vertex.
    for s in svals[:-1]:
        r = float(radius(s))
        for j in range(sectors):
            a = 2 * np.pi * j / sectors
            verts.append([s * LENGTH, BASE[1] + .80 * r * np.cos(a), r * np.sin(a)])
            attachment.append(s * LENGTH)
            groups.append(-1)
    for i in range(rings - 2):
        for j in range(sectors):
            a, b = i * sectors + j, i * sectors + (j + 1) % sectors
            faces.append([a, a + sectors, b + sectors, b])
            uvfaces.append([(i / (rings - 1), .75 * j / sectors),
                            ((i + 1) / (rings - 1), .75 * j / sectors),
                            ((i + 1) / (rings - 1), .75 * (j + 1) / sectors),
                            (i / (rings - 1), .75 * (j + 1) / sectors)])
    # Rounded cap is the last taper ring plus the unique endpoint.
    for endpoint, ring in [(LENGTH, rings - 2), (0., 0)]:
        idx = len(verts)
        verts.append([endpoint, BASE[1], 0.])
        attachment.append(endpoint)
        groups.append(-1)
        for j in range(sectors):
            f = [ring * sectors + j, idx, ring * sectors + (j + 1) % sectors]
            if endpoint == 0:
                f.reverse()
            faces.append(f)
            uvfaces.append([(.05, .375)] * 3)
    tube_count = len(verts)
    cup_info = []
    s = .045
    while s < .96:
        r = float(radius(s))
        cup_r = .29 * r
        for row in [-1, 1]:
            centre = np.array([s * LENGTH, BASE[1] + .80 * r * math.sqrt(1 - .48 ** 2), row * .48 * r])
            normal = unit(np.array([.035, 1., row * .38]))
            cross = unit(np.cross([1., 0., 0.], normal))
            along = np.cross(normal, cross)
            start = len(verts)
            # Turned cup profile: small attached foot, fleshy outer lip,
            # inner rim, depressed bowl, then underside centre to close.
            profile = [(0., -.10), (.62, -.10), (.92, .18), (1., .43),
                       (.78, .55), (.51, .32), (0., .07)]
            ring_ids = []
            for pr, ph in profile:
                ids = []
                for j in range(12 if pr else 1):
                    a = 2 * np.pi * j / 12
                    v = centre + cup_r * (pr * (math.cos(a) * along + math.sin(a) * cross) + ph * normal)
                    ids.append(len(verts))
                    verts.append(v.tolist())
                    attachment.append(s * LENGTH)
                    groups.append(len(cup_info))
                ring_ids.append(ids)
            for k in range(len(profile) - 1):
                ra, rb = ring_ids[k], ring_ids[k + 1]
                for j in range(12):
                    if len(ra) == 1:
                        f = [ra[0], rb[(j + 1) % 12], rb[j]]
                    elif len(rb) == 1:
                        f = [ra[j], ra[(j + 1) % 12], rb[0]]
                    else:
                        f = [ra[j], ra[(j + 1) % 12], rb[(j + 1) % 12], rb[j]]
                    faces.append(f)
                    uvfaces.append([(.15 + .7 * k / 5, .87)] * len(f))
            cup_info.append(dict(start=start, stop=len(verts), centre=centre.tolist(), radius=cup_r, s=s))
        # Cup pitch follows taper: densely spaced small suckers near tip.
        s += max(.013, 2.60 * cup_r / LENGTH)
    joints, weights = weights_at(np.array(attachment))
    return dict(vertices=np.array(verts), faces=faces, uvfaces=uvfaces,
                joints=joints, weights=weights, groups=np.array(groups),
                tube_count=tube_count, cups=cup_info, sectors=sectors, rings=rings)


def atlas(size=512):
    """Simple deterministic mottled skin; alpha is Breach gloss, not coverage."""
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    a = y / .75 * 2 * np.pi
    oral = np.clip((np.cos(a) - .28) / .35, 0, 1)
    # Smooth elementary interference gives broad mottling, not image detail.
    mottle = (np.sin(115 * x + 3 * np.sin(27 * y)) * np.sin(93 * y + 8 * np.cos(21 * x)))
    mottle = np.clip((mottle - .23) * 1.25, 0, 1)
    dorsal = np.array(PALETTE["dorsal"]) + mottle[..., None] * (np.array(PALETTE["mottle"]) - PALETTE["dorsal"])
    rgb = dorsal * (1 - oral[..., None]) + np.array(PALETTE["oral"]) * oral[..., None]
    cup = np.array(PALETTE["rim"]) * (1 - np.clip((x - .72) / .18, 0, 1)[..., None]) + np.array(PALETTE["bowl"]) * np.clip((x - .72) / .18, 0, 1)[..., None]
    rgb = np.where((y > .78)[..., None], cup, rgb)
    return np.concatenate([rgb, np.full((*x.shape, 1), .23)], -1).astype(np.float32)
