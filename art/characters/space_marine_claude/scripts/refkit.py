"""Reference-sheet measurement kit (runs inside Blender's Python: numpy + bpy only).

The turnaround sheet is 1536 x 1024 with three figures (front, side, back).
Everything here works in the sheet's own pixel frame and converts to metres
through one scale, so a render made with the matching orthographic cameras
(`scene.py::SHEET_VIEWS`) can be compared against the artwork row for row.
"""
import os

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "Space Marine Turnaround Sheet.png"))

SHEET_W, SHEET_H = 1536, 1024
# Pixel rows of the figure's feet and helmet top on the sheet, and its height.
FOOT_ROW = 935.0
TOP_ROW = 48.0
HEIGHT_M = 1.88
M_PER_PX = HEIGHT_M / (FOOT_ROW - TOP_ROW)

# Per-view panel: (name, centre column on the sheet, panel width in px).
PANELS = (("front", 315, 512), ("side", 776, 400), ("back", 1218, 512))


def load_rgba(path):
    """Image as float32 (h, w, 4), row 0 at the TOP, display-referred values."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1].copy()


def save_rgba(arr, path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new("_out", w, h, alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).astype(np.float32).ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def ref_mask(rgba, thr=0.075):
    """Figure mask of the reference sheet: colour distance from the backdrop."""
    rgb = rgba[..., :3]
    bg = np.median(rgb[40:200, 20:60].reshape(-1, 3), axis=0)
    # the backdrop has a soft vignette: compare against a per-row backdrop sample
    row_bg = np.median(rgb[:, 8:40], axis=1)[:, None, :]
    d = np.abs(rgb - row_bg).max(axis=2)
    mask = d > thr
    mask[960:, :] = False  # view labels
    return mask, bg


def panel_slice(name):
    for n, cx, w in PANELS:
        if n == name:
            return cx, slice(cx - w // 2, cx + w // 2)
    raise KeyError(name)


def row_runs(row, min_len=3, max_gap=2):
    """Runs of True in a 1-D bool row as [(start, end_exclusive), ...]."""
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    runs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > max_gap + 1:
            runs.append((s, p + 1))
            s = i
        p = i
    runs.append((s, p + 1))
    return [(a, b) for a, b in runs if b - a >= min_len]


def runs_table(mask, name, z_levels):
    """{z: [(x0_m, x1_m), ...]} for one panel; x in metres from the panel centre."""
    cx, sl = panel_slice(name)
    out = {}
    for z in z_levels:
        y = int(round(FOOT_ROW - z / M_PER_PX))
        if not 0 <= y < mask.shape[0]:
            continue
        runs = row_runs(mask[y, sl])
        out[z] = [((a + sl.start - cx) * M_PER_PX, (b + sl.start - cx) * M_PER_PX) for a, b in runs]
    return out


def fmt_runs(runs):
    return "  ".join("[%+.3f %+.3f]" % r for r in runs) if runs else "-"
