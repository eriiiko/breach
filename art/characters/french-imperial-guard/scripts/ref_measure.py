"""Print the reference sheet's silhouette extents per height, per view (the pre-made mask, `make_mask.py`).

    blender -b --factory-startup -P scripts/ref_measure.py [-- STEP_M]

One line per height: the runs of figure pixels across that row, in metres from the
panel centre. This is the numeric target the dimensions table is fitted to.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import sheet_ref  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sheet_ref.SHEET.measure(step=float(argv[0]) if argv else 0.04)
