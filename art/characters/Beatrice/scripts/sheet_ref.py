"""Beatrice's reference, in its own pixel frame (see `charkit/sheetfit.py`).

`without_hair.png`: 1024 x 1536, ONE front view in the modelling pose on a light grey
gradient (no side or back view exists). Measured with `charkit/sheetfit.Sheet.measure`: the
soles sit on row 1505, the top of the head on row 53.5; she is taken as 1.68 m from the soles
to the top of the head. The figure's centre line is column 511.5. `concept.jfif` (the original
artwork, three-quarter) is not a sheet: its proportions are measured by hand (beatrice.CONCEPT).
"""
import os

from sheetfit import Sheet

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
REF_PATH = os.path.join(ROOT, "without_hair.png")
CONCEPT_PATH = os.path.join(ROOT, "concept.jfif")

# thr 0.075 (the default): a smooth light backdrop; the pale bare hands sit barely off it and
# are only partly in the mask (their outline), which costs the hip-knee band a little. The glossy
# suit's white highlights read as backdrop: `fill_holes` counts what the figure encloses as figure.
# Below row 1330 (z 0.20 m) the soft floor shadows beside the boots differ from the backdrop by up
# to 0.2; the boots by 0.4 or more: `floor` raises the threshold there.
SHEET = Sheet(REF_PATH, foot_row=1505, top_row=53.5, height_m=1.68, panels=(("front", 512, 1024),), thr=0.075, fill_holes=True,
              floor=(1330, 0.25))

# Placing the concept beside the front view at the same scale (shape_vs_concept.jpg): concept
# pixels per front-view pixel (the far-side shoulder half-width, 305 concept px against the front
# view's 162), and the concept's waist point (column, row) on the midline.
CONCEPT_FRAME = dict(scale=305.0 / 162.0, waist=(585.0, 1040.0), sheet_waist_row=520.0)
