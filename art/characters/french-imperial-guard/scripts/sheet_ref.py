"""The guard officer's reference sheet, in its own pixel frame (see `charkit/sheetfit.py`).

`Gemini_Generated_Image_k4m4suk4m4suk4m4.jfif`: 832 x 1277, front / side / back on a light
grid inside a frame, with a header, view labels, a callout box and a saber section. The
figures are cut out beforehand (`make_mask.py` -> `sheet_mask.png`, `charkit/sheetmask.py`).

Scale. He is taken as 1.80 m from the soles to the crown of his skull. The crown is hidden in
the shako, so it is placed from the face: the soles sit on row 816 (front view, where the
sheet's floor line runs under them), the eye line on row 260 (the front view's pupils), and
the eye line is taken at 0.935 of stature (1.683 m). That gives 0.0030270 m per pixel and the
crown on row 221.4 -- 32 px above the shako's peak (row 253), so the shako sits about 10 cm
down over the skull.

The side view faces image-LEFT, so it shows his left side and is re-shot from his left (+90).
Panel centres: the front and back views' feet (columns 180 and 675; the front's legs are 2 px off symmetric); the side view's at the
column that puts the model's mid-plane where the sheet's body is (fitted, `ref_measure.py`).
"""
import os

from sheetmask import MaskedSheet

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
REF_PATH = os.path.join(ROOT, "Gemini_Generated_Image_k4m4suk4m4suk4m4.jfif")
MASK_PATH = os.path.join(ROOT, "sheet_mask.png")

FOOT_ROW, EYE_ROW, EYE_FRACTION, HEIGHT_M = 816.0, 260.0, 0.935, 1.80
M_PER_PX = EYE_FRACTION * HEIGHT_M / (FOOT_ROW - EYE_ROW)  # 0.0030270
TOP_ROW = FOOT_ROW - HEIGHT_M / M_PER_PX                   # 221.4: the crown of the skull

SHEET = MaskedSheet(REF_PATH, MASK_PATH, foot_row=FOOT_ROW, top_row=TOP_ROW, height_m=HEIGHT_M,
                    panels=(("front", 180, 336), ("side", 440, 170), ("back", 675, 300)),
                    azimuth={"side": 90.0})
