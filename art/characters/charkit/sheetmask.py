"""A reference sheet whose figure mask is made beforehand (Blender 4.5, numpy).

`sheetfit.Sheet` finds the figures by colour distance from a per-row backdrop sample,
which is right for a plain backdrop. A sheet drawn on a grid, with a frame, labels or a
callout box, needs its figures cut out with more care; that cut is made once, outside
Blender (the French guard's `scripts/make_mask.py`), and saved as a black-and-white PNG
the size of the sheet. `MaskedSheet` reads that PNG as its mask and is otherwise a
`Sheet`: the same frame, render, compare and measure.
"""
import numpy as np

import studio
from sheetfit import Sheet


class MaskedSheet(Sheet):
    """`Sheet` with a pre-made figure mask: `mask_path` is a PNG of the sheet's size, white
    on the figures. `cut_row`, if given, still blanks everything below it."""

    def __init__(self, path, mask_path, foot_row, top_row, height_m, panels, **kw):
        super().__init__(path, foot_row, top_row, height_m, panels, **kw)
        self.mask_path = mask_path
        self._mask = None

    def mask(self):
        if self._mask is None:
            m = studio.load_rgba(self.mask_path)[..., 0] > 0.5
            if m.shape != self.image().shape[:2]:
                raise ValueError("mask %s is %s, the sheet %s" % (self.mask_path, m.shape, self.image().shape[:2]))
            if self.cut_row is not None:
                m[int(self.cut_row):, :] = False
            self._mask = m
        return np.array(self._mask)
