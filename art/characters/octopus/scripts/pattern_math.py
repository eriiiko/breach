"""Octopus skin patterns: the PATTERN TABLE and the skin coordinates they are drawn in (issue #113).

Pure Python + NumPy (no Blender). `build_patterns.py` turns a row into a hero
material (Blender shader nodes) and bakes it onto the unchanged game mesh.

Every pattern is taken from a real animal and obeys the same three rules:

- countershading: the oral side and the suckers are paler (the existing `pale`
  attribute, 0 dorsal .. 1 oral, drives it);
- an arm pattern is drawn in the ARM's own coordinates (`pat_arm`), so it
  rings and runs along the arm and shrinks toward the tip with the arm's
  radius;
- every coordinate is a REST-pose quantity stored per vertex, so the pattern
  rides on the skin and stretches with it when the arm bends (never a
  projection from outside).

Skin coordinates (all in metres at OCTOPUS_SCALE 1):

- `pat_body` = the vertex's rest position (body, crown, web, eyes, siphon and
  arms alike: one continuous field over the whole closed surface);
- `pat_arm`  = (r_ref * U(x), r_ref * cos(theta) + 3k, r_ref * sin(theta)) on
  arm k, with x the distance along the rest arm, theta the angle around it
  (0 = oral) and U(x) = integral dx / (r_ref * (r(x)/r_ref)**TAPER_GAMMA): a
  feature of size d metres at the arm base (radius r_ref) is d * r(x)/r_ref
  metres around the arm at x and d * (r(x)/r_ref)**TAPER_GAMMA along it, so it
  shrinks toward the tip and grows a little longer than wide there, as real arm
  spots and bands do. The 3k offset decorrelates the eight arms;
- `pat_w`    = 0 on the body, rising smoothly to 1 over the first BLEND_S of
  the free arm: patterns morph from the body field into the arm field there,
  so there is no seam at the junction ring;
- `pat_s`    = arm length coordinate s in [0, 1] (0 on the body).

Colours are written in sRGB 0..255 (legible) and converted to linear here.
"""
from __future__ import annotations

import numpy as np

import arm_math as arm
import octopus_math as om

R_REF = float(arm.radius(om.S_J))      # arm radius at the junction ring (~0.076 m)
R_MIN = 0.010                          # tip floor for U(x): stops the last centimetre aliasing
TAPER_GAMMA = 0.7                      # along-arm shrink exponent (1 = isotropic, ~40 bands crowd the tip)
BLEND_S = 0.09                         # body -> arm field morph over s in [S_J, S_J + BLEND_S]


def linear(c):
    """sRGB 0..255 -> linear 0..1 (Blender colour sockets are linear)."""
    c = np.asarray(c, float) / 255.0
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist())


def arm_length_coordinate(x):
    """U(x) = integral_0^x dx / (R_REF (r/R_REF)^TAPER_GAMMA), r floored at R_MIN; in arm radii (monotone)."""
    xs = np.linspace(0.0, arm.LENGTH, 4001)
    r = np.maximum(arm.radius(np.clip(xs / arm.LENGTH, 0, 1)), R_MIN)
    inv = 1.0 / (R_REF * (r / R_REF) ** TAPER_GAMMA)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(xs))])
    return np.interp(np.clip(x, 0, arm.LENGTH), xs, cum)


def skin_coords(data):
    """Per-vertex pattern coordinates for an `octopus_math.assemble()` mesh (scale 1).

    Returns dict(pat_body (n,3), pat_arm (n,3), pat_w (n,), pat_s (n,)).
    Adds nothing to the mesh itself: geometry, rig and skin are untouched.
    """
    V = np.asarray(data["vertices"], float)
    n = len(V)
    pat_arm = np.zeros((n, 3))
    pat_w = np.zeros(n)
    pat_s = np.zeros(n)
    for k, (a, b) in enumerate(data["arm_ranges"]):
        local = (V[a:b] - om.MOUNT_POINTS[k]) @ om.MOUNT_FRAMES[k]
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        theta = np.arctan2(z, y / .8)
        pat_arm[a:b] = np.column_stack([R_REF * arm_length_coordinate(x),
                                        R_REF * np.cos(theta) + 3.0 * k,
                                        R_REF * np.sin(theta)])
        s = np.clip(x / arm.LENGTH, 0, 1)
        t = np.clip((s - om.S_J) / BLEND_S, 0, 1)
        pat_w[a:b] = t * t * (3 - 2 * t)
        pat_s[a:b] = s
    return dict(pat_body=V.copy(), pat_arm=pat_arm, pat_w=pat_w, pat_s=pat_s)


# --------------------------------------------------------------------------
# THE PATTERN TABLE. One row per skin; `builder` names the pattern function in
# build_patterns.py. Sizes are metres on the 1.2 m-arm animal (OCTOPUS_SCALE 1);
# on the arms they are the size at the arm base and shrink with the radius.
PATTERNS = {
    "default": dict(
        animal="the build's own skin (common octopus, Octopus vulgaris)",
        builder="default",
        note="build_octopus.skin_material / cup_material unchanged",
    ),
    "zebra": dict(
        animal="mimic octopus (Thaumoctopus mimicus), wunderpus (Wunderpus photogenicus)",
        builder="zebra",
        base=(232, 222, 200), dark=(52, 30, 22), oral=(240, 230, 214),
        rim=(244, 236, 222), bowl=(206, 186, 166),
        band_m=0.095,           # band period at the arm base (dark + light)
        band_duty=0.46,         # dark share of a period
        stripe_m=0.13,          # mantle stripe period
        note="dark bands ring each arm and tighten toward the tip; warped bold bands across the mantle",
    ),
    "reticulated": dict(
        animal="honeycomb moray (Gymnothorax favagineus), reticulated giraffe",
        builder="reticulated",
        base=(70, 40, 22), dark=(42, 23, 13), net=(226, 204, 150), oral=(232, 214, 176),
        rim=(240, 228, 200), bowl=(196, 168, 128),
        cell_m=0.085,           # polygon size on the mantle / at the arm base
        net_w=0.075,            # net half-width, fraction of a cell
        note="dark polygonal patches in a thin pale net, shrinking along the arms",
    ),
    "rosette": dict(
        animal="leopard (Panthera pardus), jaguar (Panthera onca)",
        builder="rosette",
        base=(200, 146, 72), centre=(166, 104, 46), dark=(34, 22, 14), oral=(240, 222, 186),
        rim=(244, 232, 210), bowl=(204, 172, 130),
        cell_m=0.10,
        note="broken dark rings round a slightly darker centre; solid spots between rosettes and down the arms",
    ),
    # Poison-frog colourways (Erik, 2026-10-27: palm green with dark rosettes and bright centres,
    # from his own reference picture; yellow against red centres to be judged side by side).
    "jungle_yellow": dict(
        animal="Erik's reference; green-and-black poison frog (Dendrobates auratus) palette",
        builder="rosette",
        base=(74, 150, 44), centre=(246, 208, 36), dark=(14, 38, 18), oral=(206, 214, 150),
        rim=(226, 230, 186), bowl=(178, 186, 120),
        cell_m=0.075, arm_solid=False, ring_r=0.62, ring_whole=True,
        note="palm green, packed near-black rosettes with yellow centres, rosettes down the arms too",
    ),
    "jungle_red": dict(
        animal="red-eyed tree frog (Agalychnis callidryas), strawberry poison frog (Oophaga pumilio) red",
        builder="rosette",
        base=(74, 150, 44), centre=(222, 40, 28), dark=(14, 38, 18), oral=(206, 214, 150),
        rim=(226, 230, 186), bowl=(178, 186, 120),
        cell_m=0.075, arm_solid=False, ring_r=0.62, ring_whole=True,
        note="as jungle_yellow with red centres",
    ),
    "azureus": dict(
        animal="blue poison frog (Dendrobates tinctorius 'azureus')",
        builder="rosette",
        base=(36, 86, 196), centre=(96, 156, 238), dark=(10, 14, 30), oral=(150, 176, 226),
        rim=(196, 210, 238), bowl=(120, 146, 200),
        cell_m=0.075, arm_solid=False, ring_r=0.62, ring_whole=True,
        note="cobalt blue, near-black rosettes with pale-blue centres",
    ),
    "bumblebee": dict(
        animal="yellow-banded poison frog (Dendrobates leucomelas)",
        builder="reticulated",
        base=(30, 26, 20), dark=(14, 12, 10), net=(250, 204, 30), oral=(232, 214, 150),
        rim=(240, 228, 190), bowl=(196, 176, 110),
        cell_m=0.085, net_w=0.11,
        note="black patches in a bold yellow net",
    ),
    "red_whitespot": dict(
        animal="white-spotted octopus (Callistoctopus macropus)",
        builder="red_whitespot",
        base=(156, 38, 22), deep=(122, 25, 14), spot=(242, 230, 220), oral=(226, 150, 124),
        rim=(240, 206, 188), bowl=(196, 128, 104),
        cell_m=0.075,
        note="brick red, scattered oval white spots of uneven size, smaller toward the tips",
    ),
    "blue_ringed": dict(
        animal="greater blue-ringed octopus (Hapalochlaena lunulata)",
        builder="blue_ringed",
        base=(204, 164, 82), blotch=(146, 102, 44), edge=(22, 16, 12), blue=(24, 112, 255),
        oral=(232, 210, 150), rim=(240, 228, 190), bowl=(196, 170, 120),
        cell_m=0.12,
        note="ochre-yellow with brown blotches, each carrying an electric-blue ring with a black edge",
    ),
    "ghost": dict(
        animal="Casper octopus (an undescribed incirrate octopod, Hawaii 2016), dumbo octopuses (Grimpoteuthis)",
        builder="ghost",
        base=(230, 218, 216), blush=(226, 140, 150), speck=(178, 96, 104), oral=(244, 238, 236),
        rim=(248, 240, 238), bowl=(232, 200, 198),
        subsurface=0.30,       # hero only; the game bake is albedo only
        note="pale, near-translucent white with a pink blush gathering toward the arm tips and web",
    ),
}

ORDER = list(PATTERNS)
