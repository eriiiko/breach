"""smoke^gamma render-contrast knob (smoke v2 — render contrast).

Canon: ch.05 §6.1 step 5 "smoke^gamma contrast". A power curve is applied to
the *rendered* smoke opacity (``FieldOverlay.update``), NOT to the simulation
field: ``value = pow(saturate(smoke_density), gamma)``. gamma > 1 crushes thin
smoke toward transparent and sharpens wispy edges (filmic). 1.0 = identity.

The curve lives inside ``FieldOverlay.update``, which uploads to a GPU texture
and so needs a live GL context — it is exercised visually by the renderer smoke
test and the lighting demo, not here. This headless test verifies the two
things that ARE checkable without a window:

1. The config value is present and plumbed: ``CFG.smoke.smoke_render_gamma``
   loads as the shipped default (1.5).
2. The power curve behaves as specified — identity at gamma=1, crushes thin
   smoke and preserves full density at gamma>1, monotonic — mirroring exactly
   the transform ``FieldOverlay.update`` applies (``v ** gamma`` after a
   ``clip(field, 0, 1)``).

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_smoke_render_gamma.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from config import CFG


def _render_curve(field, gamma):
    """Exactly the transform FieldOverlay.update applies to the rendered value
    (renderer/overlays.py): clip to [0,1], then ** gamma when gamma != 1.0.
    """
    v = np.clip(field, 0.0, 1.0)
    if gamma != 1.0:
        v = v ** gamma
    return v


# --------------------------------------------------------------------------
# 1. Config value is present and plumbed
# --------------------------------------------------------------------------
def test_config_default_gamma():
    g = getattr(CFG.smoke, "smoke_render_gamma", None)
    assert g is not None, "[smoke] smoke_render_gamma missing from config.toml"
    assert abs(float(g) - 1.5) < 1e-9, (
        f"shipped default changed unexpectedly: {g} (expected 1.5)")
    print(f"OK: config_default_gamma (smoke_render_gamma = {g})")


# --------------------------------------------------------------------------
# 2. Curve behaviour matches the spec
# --------------------------------------------------------------------------
def test_gamma_curve_behaviour():
    d = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    # gamma = 1.0 is identity.
    assert np.allclose(_render_curve(d, 1.0), d), "gamma=1.0 not identity"

    # gamma > 1 crushes thin smoke toward transparent (value drops) but leaves
    # the endpoints 0 and 1 fixed — exactly the filmic "thin -> clear, dense
    # stays dense" remap.
    out = _render_curve(d, 1.5)
    assert out[0] == 0.0 and abs(out[-1] - 1.0) < 1e-6, "endpoints not fixed"
    # Every interior value is pushed DOWN (more transparent) by gamma>1.
    assert np.all(out[1:-1] < d[1:-1]), "gamma>1 did not crush thin smoke"
    # Thin smoke is crushed proportionally harder than dense smoke (the
    # contrast: ratio out/d shrinks as density shrinks).
    ratio = out[1:-1] / d[1:-1]
    assert np.all(np.diff(ratio) > 0), (
        "thin smoke not crushed harder than dense smoke")

    # Monotonic and within range for the documented range bounds.
    for gamma in (1.2, 1.5, 2.5):
        o = _render_curve(d, gamma)
        assert np.all(np.diff(o) >= 0), f"non-monotonic at gamma={gamma}"
        assert o.min() >= 0.0 and o.max() <= 1.0, f"out of range at gamma={gamma}"

    print("OK: gamma_curve_behaviour (identity@1, crushes thin@>1, monotonic)")


if __name__ == "__main__":
    test_config_default_gamma()
    test_gamma_curve_behaviour()
    print("\nAll smoke render-gamma tests passed.")
