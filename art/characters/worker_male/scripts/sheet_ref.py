"""The male worker's reference sheet, in its own pixel frame (see `charkit/sheetfit.py`).

`Three-view spaceship maintenance mechanic-1.png`: 1536 x 1024, front / side / back on
a white backdrop. Rows and columns measured with `ref_measure.py`: the soles sit on row
950, the top of the hair on row 17; the figure is taken as 1.80 m to the top of the hair.
The side view faces LEFT, so it is re-shot from the character's left (+90 deg).
"""
import os

from sheetfit import Sheet

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "Three-view spaceship maintenance mechanic-1.png"))

# thr 0.16: the backdrop is white with soft contact shadows under the boots, which a
# lower colour-distance threshold counts as figure.
SHEET = Sheet(REF_PATH, foot_row=950, top_row=17, height_m=1.80,
              panels=(("front", 316, 512), ("side", 766, 400), ("back", 1218, 512)),
              thr=0.16, azimuth={"side": 90.0})
