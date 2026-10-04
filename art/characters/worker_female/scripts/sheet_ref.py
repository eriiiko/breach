"""The female worker's reference sheet, in its own pixel frame (see `charkit/sheetfit.py`).

`Spaceship maintenance mechanic turnaround sheet-2.png`: 1536 x 1024, front / her LEFT
side / back on a white backdrop. Rows and columns measured with `ref_measure.py`: the soles
sit on row 950, the top of the hair (front view) on row 15; she is taken as 1.68 m from the
soles to the top of the hair as seen from the front. The side view faces LEFT, so it is
re-shot from the character's left (+90 deg).
"""
import os

from sheetfit import Sheet

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "Spaceship maintenance mechanic turnaround sheet-2.png"))

# thr 0.16 as for the male sheet: a white backdrop with soft contact shadows under the boots
SHEET = Sheet(REF_PATH, foot_row=950, top_row=15, height_m=1.68,
              panels=(("front", 321, 520), ("side", 760, 400), ("back", 1215, 520)),
              thr=0.16, azimuth={"side": 90.0})
