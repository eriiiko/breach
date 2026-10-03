"""The space marine's materials, and its re-shoot of the reference sheet.

The turnaround sheet is re-shot with orthographic cameras that reproduce the
reference sheet's own framing (`refkit`), so a render and the artwork can be
compared silhouette for silhouette. Studio, cameras and the material builders
are the shared `charkit/studio.py`.
"""
import os

import numpy as np

import refkit
import studio
from studio import BACKDROP, BLACK, mat_armor, mat_cloth, mat_plain, mat_visor, rgb, setup  # noqa: F401

IVORY_A, IVORY_B = rgb(0.54, 0.45, 0.32), rgb(0.64, 0.56, 0.42)


def make_materials():
    return dict(
        suit=mat_cloth("suit_fabric", IVORY_A, IVORY_B),  # padded ivory fabric; `flex` = black ribbed stretch panels
        armor=mat_armor("armor_ivory", IVORY_A, IVORY_B),
        blue=mat_armor("panel_blue", rgb(0.065, 0.105, 0.17), rgb(0.085, 0.13, 0.21), (0.42, 0.6), 1.0),
        rubber=mat_plain("glove_rubber", BLACK, 0.55, bump_scale=420.0, bump=0.25, sheen=0.3),
        strap=mat_plain("strap_webbing", rgb(0.035, 0.035, 0.04), 0.7, bump_scale=900.0, bump=0.3),
        sole=mat_plain("boot_sole", rgb(0.03, 0.03, 0.032), 0.75, bump_scale=300.0, bump=0.2),
        dark=mat_plain("dark_fitting", rgb(0.02, 0.02, 0.022), 0.45),
        metal=mat_plain("fastener_metal", rgb(0.30, 0.29, 0.27), 0.4, metallic=1.0),
        visor=mat_visor(),
    )


# The three sheet views: (name, rig azimuth). The rig carries camera AND lights, so
# every view is lit like a front view, as on the reference sheet.
SHEET_VIEWS = (("front", 0.0), ("side", -90.0), ("back", 180.0))


def render_sheet(rig, cam, floor, out_dir, tag="sheet"):
    """Orthographic front/side/back in the reference sheet's framing; returns RGBA panels."""
    views = [(name, az, w) for (name, az), (_, _, w) in zip(SHEET_VIEWS, refkit.PANELS)]
    return studio.ortho_panels(rig, cam, floor, out_dir, views, refkit.M_PER_PX, refkit.SHEET_H,
                               (refkit.FOOT_ROW - refkit.SHEET_H / 2) * refkit.M_PER_PX, tag)


def compare_sheet(panels, out_dir, tag="sheet", verbose=True):
    """Write the model's sheet + a silhouette overlay against the artwork; print mismatches."""
    ref = refkit.load_rgba(refkit.REF_PATH)
    rmask, _ = refkit.ref_mask(ref)
    over = np.ones((refkit.SHEET_H, sum(w for _, _, w in refkit.PANELS), 4), np.float32)
    x0 = 0
    z_levels = [round(z, 3) for z in np.arange(1.86, 0.05, -0.04)]
    for name, _, w in refkit.PANELS:
        mine = panels[name][..., 3] > 0.5
        sl = refkit.panel_slice(name)[1]
        rm = rmask[:, sl]
        o = np.zeros((refkit.SHEET_H, w, 3), np.float32) + 0.93
        o[rm & mine] = (0.62, 0.62, 0.62)
        o[rm & ~mine] = (0.90, 0.25, 0.20)  # artwork only
        o[~rm & mine] = (0.20, 0.40, 0.90)  # model only
        over[:, x0:x0 + w, :3] = o
        iou = (rm & mine).sum() / max(1, (rm | mine).sum())
        print("\n== %s  IoU %.3f  (red = artwork only, blue = model only) ==" % (name, iou))
        if verbose:
            full = np.zeros_like(rmask)
            full[:, sl] = mine
            ra, rb = refkit.runs_table(rmask, name, z_levels), refkit.runs_table(full, name, z_levels)
            for z in z_levels:
                r, m = ra.get(z, []), rb.get(z, [])
                bad = len(r) != len(m) or any(abs(p[0] - q[0]) > 0.012 or abs(p[1] - q[1]) > 0.012 for p, q in zip(r, m))
                if bad:
                    print("z=%.2f ref %s\n       got %s" % (z, refkit.fmt_runs(r), refkit.fmt_runs(m)))
        x0 += w
    studio.save_rgba(studio.compose(panels, [n for n, _, _ in refkit.PANELS]), os.path.join(out_dir, tag + ".png"))
    studio.save_rgba(over, os.path.join(out_dir, tag + "_overlay.png"))


# Beauty views: name -> (rig azimuth, camera azimuth within the rig, elevation, distance,
# target height, lens mm, resolution). The rig turns the lights with the camera.
BEAUTY = {
    "hero": (0.0, 32.0, 8.0, 5.4, 0.93, 85.0, (1200, 1600)),
    "hero_back": (180.0, 32.0, 8.0, 5.4, 0.93, 85.0, (1200, 1600)),
    "closeup": (0.0, 24.0, 4.0, 2.7, 1.50, 85.0, (1400, 1400)),
    "closeup_back": (180.0, 25.0, 10.0, 2.7, 1.48, 85.0, (1400, 1400)),
    "elevated": (0.0, 35.0, 52.0, 5.2, 0.95, 85.0, (1200, 1400)),
    "legs": (0.0, 28.0, 6.0, 2.9, 0.48, 85.0, (1400, 1400)),
    "top": (0.0, 0.0, 89.0, 4.6, 0.9, 85.0, (1200, 1200)),
}


def render_beauty(rig, cam, name, out_dir, scale=1.0):
    studio.render_beauty(rig, cam, BEAUTY[name], name, out_dir, scale)
