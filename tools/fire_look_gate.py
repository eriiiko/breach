"""tools/fire_look_gate.py -- the fire LOOK GATE: colour and brightness of the
fire region of look shots, before vs after (graphics arc G3 "Fire", patch F3;
docs/g3_fire_design_2026-10-23.md §3.6, §13.7, §13.8).

G0's P1 lesson: moving the black-body heat layer into the linear scene washed
its orange toward beige, and nobody had a number for it. This tool is the
number. It reads two shot folders (``tools/lighting_demo.py --shots ...
--shots-out DIR``: ``<name>_<tag>.png``) and, per shot, measures the FIRE
REGION -- the burning tiles (``fire > i_min`` and at or above the glow floor;
the glowing tiles when nothing burns, e.g. a heat-only blast), dilated by one
tile and projected through the shot's own camera -- in CIELAB:

  hue      atan2 of the CHROMA-WEIGHTED mean (a*, b*): the colour the eye
           reads, not dragged by grey pixels;
  chroma   the mean C* = |(a*, b*)|;
  lum      the mean relative luminance Y (linear, Rec. 709 / sRGB D65);
  halo     how far the fire's own light reaches: the distance from the
           burning tiles (tiles) at which the glow -- the shot's luminance
           minus its ``<name>_noheat`` twin's (the same shot with the heat
           layer and the flame hidden) -- falls to 10 % of its value on them.

The mask comes from the shot's META (``<png>.meta.npz``, written by the
look-shot child since F3: camera + per-tile Kelvin and fire). The sim is
deterministic per shot, so the META of the ``--after`` run masks both runs
(the ``--before`` folder may come from a commit that wrote none).

Gates (``--gate``):
  hue    |hue shift| <= 5 deg and chroma within +-4 % of before, on every
         shot with a fire region (design §13.8: G0's tolerance);
  lum    masked luminance within +-4 % (§13.7's compensation gate: run it on
         ``fire_thin_smoke`` / ``fire_thick_smoke`` with the flame off).

Look shots are not pixel-reproducible (the units' animation and the effect
rings run on wall time): ``--noise DIR`` measures the same metrics between
two runs of the SAME commit (before vs DIR) and prints that floor beside each
row, so a gate that fails by less than its own noise is visible as such.

    python tools/fire_look_gate.py --before A --after B [--shots a,b]
        [--noise A2] [--gate hue|lum] [--csv out.csv]

Pure numpy + Pillow; no window. Credit: CIELAB / sRGB conversion per CIE
15:2004 and IEC 61966-2-1 (the standard D65 formulas).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.look_shots import meta_path  # noqa: E402

# The fire shots the gates run on (design §3.6 / §13.8).
GATED = ("fss_overview", "fss_close", "fss_lone", "fss_windy", "fire_plume",
         "fire_thin_smoke", "fire_thick_smoke", "explosion_t04",
         "explosion_t12", "explosion_t40")
# Each shot's halo twin: the same shot with the fire's own light hidden.
TWIN = {n: f"{n}_noheat" for n in GATED}
TWIN["fire_plume"] = "smoke_plume_noheat"

HUE_TOL_DEG = 5.0
REL_TOL = 0.04
I_MIN = 0.01                    # [render.fire_vfx] i_min: "burning"
HALO_FRAC = 0.10


# ---------------------------------------------------------------------------
# Colour (sRGB D65 -> linear -> XYZ -> CIELAB)
# ---------------------------------------------------------------------------

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB -> (..., 3) CIELAB (L*, a*, b*), D65."""
    lin = srgb_to_linear(np.asarray(rgb_u8, dtype=np.float64) / 255.0)
    xyz = lin @ _M.T / _WHITE
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def luminance(rgb_u8: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB -> relative luminance Y in [0, 1]."""
    lin = srgb_to_linear(np.asarray(rgb_u8, dtype=np.float64) / 255.0)
    return lin @ _M[1]


# ---------------------------------------------------------------------------
# The fire region
# ---------------------------------------------------------------------------

@dataclass
class Meta:
    kelvin: np.ndarray
    fire: np.ndarray
    glow_min_k: float
    cam: np.ndarray             # (pos_x, pos_y, zoom, view_w, view_h)

    @classmethod
    def load(cls, png: Path) -> "Meta":
        z = np.load(meta_path(png))
        return cls(z["kelvin"], z["fire"], float(z["glow_min_k"]), z["cam"])


def fire_tiles(meta: Meta):
    """(mask, kind): the burning tiles, or the glowing ones when none burns."""
    hot = meta.kelvin >= meta.glow_min_k
    burning = hot & (meta.fire > I_MIN)
    if burning.any():
        return burning, "burning"
    return hot, "glowing"


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = np.zeros_like(mask)
            sh[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
                mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            out |= sh
    return out


def tile_coords(meta: Meta, shape) -> np.ndarray:
    """(H, W, 2) float tile coordinates (x, y) of each pixel's centre."""
    px, py, zoom = meta.cam[0], meta.cam[1], meta.cam[2]
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.stack([px + (xs + 0.5) / zoom, py + (ys + 0.5) / zoom], axis=-1)


def project(tile_mask: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """A per-tile mask sampled at pixel tile coordinates (nearest tile)."""
    th, tw = tile_mask.shape
    tx = np.floor(coords[..., 0]).astype(np.int64)
    ty = np.floor(coords[..., 1]).astype(np.int64)
    inside = (tx >= 0) & (ty >= 0) & (tx < tw) & (ty < th)
    out = np.zeros(coords.shape[:2], dtype=bool)
    out[inside] = tile_mask[ty[inside], tx[inside]]
    return out


def distance_tiles(tile_mask: np.ndarray, coords: np.ndarray,
                   sub: int = 8) -> np.ndarray:
    """Per pixel, the distance (tiles) to the nearest masked TILE (0 on it):
    a Euclidean distance transform of the mask on a ``sub`` x finer grid,
    sampled at the pixels' tile coordinates (exact to 1 / sub tile)."""
    from scipy.ndimage import distance_transform_edt
    fine = np.repeat(np.repeat(~tile_mask, sub, axis=0), sub, axis=1)
    d = distance_transform_edt(fine) / float(sub)
    fh, fw = fine.shape
    fx = np.clip(np.floor(coords[..., 0] * sub).astype(np.int64), 0, fw - 1)
    fy = np.clip(np.floor(coords[..., 1] * sub).astype(np.int64), 0, fh - 1)
    out = d[fy, fx]
    off = ((coords[..., 0] < 0) | (coords[..., 1] < 0)
           | (coords[..., 0] >= tile_mask.shape[1])
           | (coords[..., 1] >= tile_mask.shape[0]))
    out[off] = np.inf
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def load_rgb(png: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(png).convert("RGB"))


def region_metrics(rgb: np.ndarray, pix: np.ndarray) -> Dict[str, float]:
    lab = srgb_to_lab(rgb[pix])
    a, b = lab[:, 1], lab[:, 2]
    C = np.hypot(a, b)
    wsum = max(C.sum(), 1e-9)
    hue = math.degrees(math.atan2((C * b).sum() / wsum, (C * a).sum() / wsum))
    return {"hue": hue, "chroma": float(C.mean()),
            "lum": float(luminance(rgb[pix]).mean()),
            "L": float(lab[:, 0].mean())}


def halo_tiles(rgb: np.ndarray, twin: np.ndarray, dist: np.ndarray,
               step: float = 0.125, max_t: float = 12.0) -> float:
    """The distance from the fire tiles (tiles) at which its glow -- the
    luminance above the twin's -- falls to HALO_FRAC of its value on them."""
    glow = np.maximum(luminance(rgb) - luminance(twin), 0.0)
    base = glow[dist <= 0.0]
    if base.size == 0 or base.mean() <= 1e-6:
        return float("nan")
    ref = base.mean()
    edges = np.arange(step, max_t + step, step)
    prev = 0.0
    for e in edges:
        ring = (dist > prev) & (dist <= e)
        prev = e
        if ring.any() and glow[ring].mean() < HALO_FRAC * ref:
            return float(e)
    return float("inf")


def find_png(folder: Path, name: str) -> Optional[Path]:
    hits = sorted(folder.glob(f"{name}_*.png"))
    hits = [h for h in hits if h.stem.rsplit("_", 1)[0] == name]
    return hits[-1] if hits else None


@dataclass
class Row:
    name: str
    kind: str
    tiles: int
    before: Dict[str, float]
    after: Dict[str, float]
    halo_before: float
    halo_after: float
    noise: Optional[Dict[str, float]] = None

    @property
    def d_hue(self) -> float:
        d = self.after["hue"] - self.before["hue"]
        return (d + 180.0) % 360.0 - 180.0

    def rel(self, key: str) -> float:
        return self.after[key] / max(self.before[key], 1e-9) - 1.0


def measure(before: Path, after: Path, names: Sequence[str],
            noise: Optional[Path] = None) -> List[Row]:
    rows = []
    for n in names:
        pa, pb = find_png(after, n), find_png(before, n)
        if pa is None or pb is None:
            print(f"[gate] {n}: missing PNG (before {pb}, after {pa}) -- skipped")
            continue
        meta = Meta.load(pa)
        tiles, kind = fire_tiles(meta)
        if not tiles.any():
            print(f"[gate] {n}: no fire region -- skipped")
            continue
        ra, rb = load_rgb(pa), load_rgb(pb)
        coords = tile_coords(meta, ra.shape[:2])
        pix = project(dilate(tiles, 1), coords)
        if not pix.any():
            print(f"[gate] {n}: fire region off screen -- skipped")
            continue
        dist = distance_tiles(tiles, coords)
        ha = hb = float("nan")
        tw = TWIN.get(n)
        if tw:
            ta, tb = find_png(after, tw), find_png(before, tw)
            if ta is not None:
                ha = halo_tiles(ra, load_rgb(ta), dist)
            if tb is not None:
                hb = halo_tiles(rb, load_rgb(tb), dist)
        row = Row(n, kind, int(tiles.sum()), region_metrics(rb, pix),
                  region_metrics(ra, pix), hb, ha)
        if noise is not None:
            pn = find_png(noise, n)
            if pn is not None:
                row.noise = region_metrics(load_rgb(pn), pix)
        rows.append(row)
    return rows


def report(rows: Sequence[Row], gate: Optional[str]) -> bool:
    ok = True
    hdr = (f"{'shot':22s} {'region':>12s} {'hue b->a':>15s} {'dHue':>6s} "
           f"{'chroma b->a':>15s} {'dC%':>6s} {'lum dY%':>8s} "
           f"{'halo b->a (tiles)':>18s} {'noise dHue/dC%/dY%':>20s}  verdict")
    print(hdr)
    for r in rows:
        nz = ""
        if r.noise is not None:
            nh = (r.noise["hue"] - r.before["hue"] + 180.0) % 360.0 - 180.0
            nc = r.noise["chroma"] / max(r.before["chroma"], 1e-9) - 1.0
            nl = r.noise["lum"] / max(r.before["lum"], 1e-9) - 1.0
            nz = f"{nh:+.1f}/{100 * nc:+.1f}/{100 * nl:+.1f}"
        verdict = ""
        if gate == "hue":
            good = abs(r.d_hue) <= HUE_TOL_DEG and abs(r.rel("chroma")) <= REL_TOL
            verdict = "PASS" if good else "FAIL"
            ok &= good
        elif gate == "lum":
            good = abs(r.rel("lum")) <= REL_TOL
            verdict = "PASS" if good else "FAIL"
            ok &= good
        print(f"{r.name:22s} {r.kind[:7]:>7s} {r.tiles:4d}t "
              f"{r.before['hue']:6.1f}->{r.after['hue']:6.1f} {r.d_hue:+6.1f} "
              f"{r.before['chroma']:6.1f}->{r.after['chroma']:6.1f} "
              f"{100 * r.rel('chroma'):+6.1f} {100 * r.rel('lum'):+8.1f} "
              f"{r.halo_before:8.2f}->{r.halo_after:6.2f} {nz:>20s}  {verdict}")
    if gate:
        print(f"[gate] {gate}: {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--before", required=True, type=Path)
    ap.add_argument("--after", required=True, type=Path)
    ap.add_argument("--shots", default=",".join(GATED))
    ap.add_argument("--noise", type=Path, default=None)
    ap.add_argument("--gate", choices=("hue", "lum"), default=None)
    ap.add_argument("--csv", type=Path, default=None)
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.shots.split(",") if n.strip()]
    rows = measure(a.before, a.after, names, a.noise)
    ok = report(rows, a.gate)
    if a.csv is not None:
        keys = ("hue", "chroma", "lum", "L")
        with open(a.csv, "w", encoding="utf-8") as f:
            f.write("shot,region,tiles," + ",".join(f"before_{k}" for k in keys)
                    + "," + ",".join(f"after_{k}" for k in keys)
                    + ",halo_before,halo_after\n")
            for r in rows:
                f.write(f"{r.name},{r.kind},{r.tiles},"
                        + ",".join(f"{r.before[k]:.4f}" for k in keys) + ","
                        + ",".join(f"{r.after[k]:.4f}" for k in keys)
                        + f",{r.halo_before:.3f},{r.halo_after:.3f}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
