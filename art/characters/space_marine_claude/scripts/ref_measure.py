"""Print the reference sheet's silhouette extents per height, per view.

    blender -b --factory-startup -P ref_measure.py

One line per height: the runs of figure pixels across that row, in metres from
the panel centre. This is the numeric target the model's proportions are fitted to.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refkit  # noqa: E402

rgba = refkit.load_rgba(refkit.REF_PATH)
mask, bg = refkit.ref_mask(rgba)
print("sheet", rgba.shape, "backdrop rgb", np.round(bg, 3), "m/px %.5f" % refkit.M_PER_PX)
for name, cx, w in refkit.PANELS:
    sl = refkit.panel_slice(name)[1]
    ys, xs = np.nonzero(mask[:, sl])
    print("%s bbox: rows %d..%d  cols %d..%d (sheet px)" % (name, ys.min(), ys.max(), xs.min() + sl.start, xs.max() + sl.start))

z_levels = [round(z, 3) for z in np.arange(1.88, -0.01, -0.04)]
for name, _, _ in refkit.PANELS:
    print("\n== %s ==" % name)
    for z, runs in refkit.runs_table(mask, name, z_levels).items():
        print("z=%.2f  %s" % (z, refkit.fmt_runs(runs)))
