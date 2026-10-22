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

# The side and back views (`without_hair_side.png`, `without_hair_back.png`, 1024 x 1536 each, supplied
# after stage 1), each its own sheet in its own pixel frame, scaled to the same 1.68 m. Rows measured
# the same way as the front (the figure mask's lowest and highest rows, +1 / -0.5). The side view faces
# image-left, which the studio shows at rig azimuth +90. Its centre column is where the model's y = 0
# plane lands: fitted once by sliding the model's side silhouette along the drawing (best overlap of
# the head, neck and boots, the parts that are the drawing's own and not our shape decision). The back
# view's centre column is the figure's midline (head, collar, between the heels).
SIDE_PATH = os.path.join(ROOT, "without_hair_side.png")
BACK_PATH = os.path.join(ROOT, "without_hair_back.png")
SIDE = Sheet(SIDE_PATH, foot_row=1467, top_row=36.5, height_m=1.68, panels=(("side", 518, 1000),), thr=0.075, fill_holes=True,
             floor=(1330, 0.25), azimuth=dict(side=90.0))
BACK = Sheet(BACK_PATH, foot_row=1453, top_row=46.5, height_m=1.68, panels=(("back", 510, 1000),), thr=0.075, fill_holes=True,
             floor=(1330, 0.25))
# the extra views `--sheet` scores and pictures (tag, sheet)
VIEWS = (("side", SIDE), ("back", BACK))

# The owner's frontal BODY reference (good-body-silhouette.jfif, 683 x 1024, supplied 2026-10-05; head to the upper
# thighs, arms a little out, the earlier high-cut suit): not a sheet, a shape witness (hipsview.silhouette_picture).
# Measured by hand between the arms and the body where the backdrop shows: centre column 341, outer deltoids 213.5 px
# (row 480), narrowest waist 128 px (row 760), widest 198.5 px just above the crotch (rows 980-1000) -- half-widths.
# B1f: light first shows between its thighs at row 1013 (columns 323-346), 23-26 px under the suit's crotch (rows 987-990);
# its frame ends at row 1023 (hipsview.lens_picture aligns the model there).
SILHOUETTE = dict(path=os.path.join(ROOT, "good-body-silhouette.jfif"), centre=341.0, shoulder_hw=213.5, waist_row=760.0, waist_z=1.14,
                  light_row=1013.0)
