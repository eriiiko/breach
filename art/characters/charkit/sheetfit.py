"""Fit a character to a three-view reference sheet (numpy + bpy, inside Blender).

The generic form of the space marine's `refkit.py` + `scene.py` sheet tools, for any
sheet laid out as front / side / back panels on a flat backdrop. A `Sheet` holds the
sheet's own pixel frame (feet row, head-top row, the figure's height in metres, the
panel columns); everything converts to metres through that one scale, so the model can
be re-shot with orthographic cameras in exactly the sheet's framing and compared
row for row:

    sheet = Sheet(path, foot_row, top_row, height_m, panels)
    sheet.measure()                         # the sheet's per-height silhouette table
    panels = sheet.render(rig, cam, floor, out_dir)
    sheet.compare(panels, out_dir)          # overlay + IoU per view + mismatch table
"""
import os

import numpy as np

import studio

# The three sheet views and the studio rig azimuth that shows each. The rig carries the
# camera AND the lights, so every view is lit like a front view, as on a sheet.
VIEW_AZIMUTH = {"front": 0.0, "side": -90.0, "back": 180.0}


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


def fmt_runs(runs):
    return "  ".join("[%+.3f %+.3f]" % r for r in runs) if runs else "-"


class Sheet:
    """A reference sheet. `panels` = ((name, centre column, width px), ...) with names
    from VIEW_AZIMUTH; `foot_row` / `top_row` are the pixel rows of the soles and the
    top of the head; `cut_row` blanks everything below it (labels, floor shadows)."""

    def __init__(self, path, foot_row, top_row, height_m, panels, thr=0.075, cut_row=None, bg_cols=(8, 40), azimuth=None):
        self.path, self.foot_row, self.top_row, self.height_m = path, float(foot_row), float(top_row), height_m
        self.panels, self.thr, self.cut_row, self.bg_cols = panels, thr, cut_row, bg_cols
        # the side view's azimuth decides which way the profile faces: -90 shows the
        # character facing right, +90 facing left
        self.azimuth = dict(VIEW_AZIMUTH, **(azimuth or {}))
        self.m_per_px = height_m / (self.foot_row - self.top_row)
        self._ref = None

    # ----------------------------------------------------------------- the artwork
    def image(self):
        if self._ref is None:
            self._ref = studio.load_rgba(self.path)
        return self._ref

    @property
    def size(self):
        h, w = self.image().shape[:2]
        return w, h

    def mask(self):
        """Figure mask: colour distance from a per-row backdrop sample (sheets vignette)."""
        rgb = self.image()[..., :3]
        a, b = self.bg_cols
        row_bg = np.median(rgb[:, a:b], axis=1)[:, None, :]
        m = np.abs(rgb - row_bg).max(axis=2) > self.thr
        if self.cut_row is not None:
            m[int(self.cut_row):, :] = False
        return m

    def panel_slice(self, name):
        for n, cx, w in self.panels:
            if n == name:
                return cx, slice(cx - w // 2, cx + w // 2)
        raise KeyError(name)

    def runs_table(self, mask, name, z_levels):
        """{z: [(x0_m, x1_m), ...]} for one panel; x in metres from the panel centre."""
        cx, sl = self.panel_slice(name)
        out = {}
        for z in z_levels:
            y = int(round(self.foot_row - z / self.m_per_px))
            if not 0 <= y < mask.shape[0]:
                continue
            runs = row_runs(mask[y, sl])
            out[z] = [((a + sl.start - cx) * self.m_per_px, (b + sl.start - cx) * self.m_per_px) for a, b in runs]
        return out

    def measure(self, step=0.04):
        """Print the artwork's bounding boxes and its runs per height, per view."""
        mask = self.mask()
        print("sheet", self.size, "m/px %.5f" % self.m_per_px)
        for name, cx, w in self.panels:
            sl = self.panel_slice(name)[1]
            ys, xs = np.nonzero(mask[:, sl])
            print("%s bbox: rows %d..%d  cols %d..%d (sheet px)" % (name, ys.min(), ys.max(), xs.min() + sl.start, xs.max() + sl.start))
        z_levels = [round(z, 3) for z in np.arange(self.height_m, -0.01, -step)]
        for name, _, _ in self.panels:
            print("\n== %s ==" % name)
            for z, runs in self.runs_table(mask, name, z_levels).items():
                print("z=%.2f  %s" % (z, fmt_runs(runs)))

    # --------------------------------------------------------------------- the model
    def render(self, rig, cam, floor, out_dir, tag="sheet"):
        """Orthographic front/side/back in the sheet's framing; returns RGBA panels."""
        w_img, h_img = self.size
        views = [(name, self.azimuth[name], w) for name, _, w in self.panels]
        return studio.ortho_panels(rig, cam, floor, out_dir, views, self.m_per_px, h_img,
                                   (self.foot_row - h_img / 2) * self.m_per_px, tag)

    def compare(self, panels, out_dir, tag="sheet", verbose=True, tol=0.012, step=0.04):
        """Write the model's sheet, the artwork-beside-model sheet and a silhouette overlay;
        print IoU per view and the heights where a run differs by more than `tol` metres.
        Returns {view: IoU}."""
        rmask = self.mask()
        h_img = self.size[1]
        over = np.ones((h_img, sum(w for _, _, w in self.panels), 4), np.float32)
        x0, scores = 0, {}
        z_levels = [round(z, 3) for z in np.arange(self.height_m - 0.02, 0.05, -step)]
        for name, _, w in self.panels:
            mine = panels[name][..., 3] > 0.5
            sl = self.panel_slice(name)[1]
            rm = rmask[:, sl]
            o = np.zeros((h_img, w, 3), np.float32) + 0.93
            o[rm & mine] = (0.62, 0.62, 0.62)
            o[rm & ~mine] = (0.90, 0.25, 0.20)  # artwork only
            o[~rm & mine] = (0.20, 0.40, 0.90)  # model only
            over[:, x0:x0 + w, :3] = o
            iou = (rm & mine).sum() / max(1, (rm | mine).sum())
            scores[name] = float(iou)
            print("\n== %s  IoU %.3f  (red = artwork only, blue = model only) ==" % (name, iou))
            if verbose:
                full = np.zeros_like(rmask)
                full[:, sl] = mine
                ra, rb = self.runs_table(rmask, name, z_levels), self.runs_table(full, name, z_levels)
                for z in z_levels:
                    r, m = ra.get(z, []), rb.get(z, [])
                    bad = len(r) != len(m) or any(abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol for p, q in zip(r, m))
                    if bad:
                        print("z=%.2f ref %s\n       got %s" % (z, fmt_runs(r), fmt_runs(m)))
            x0 += w
        order = [n for n, _, _ in self.panels]
        model = studio.compose(panels, order)
        studio.save_rgba(model, os.path.join(out_dir, tag + ".png"))
        studio.save_rgba(over, os.path.join(out_dir, tag + "_overlay.png"))
        # artwork (cropped to the same panels) above the model's re-shoot
        ref = self.image()
        top = np.concatenate([ref[:, self.panel_slice(n)[1]] for n in order], axis=1)
        top[..., 3] = 1.0
        studio.save_rgba(np.concatenate([top, model], axis=0), os.path.join(out_dir, tag + "_vs_reference.png"))
        return scores
