r"""Greybox tileset generator — the tiled-path dev tileset (engine/15 §3, P1).

Writes ``art/tilesets/greybox/``: one diffuse + one normal PNG strip per
canon material, plus the ``tileset.toml`` manifest the baker (P2,
``tools/bake_level_art.py``) consumes. Pure numpy + PIL — no Stable
Diffusion, no network, no wall-clock randomness: the same CLI args produce
byte-identical files.

The material set is DERIVED from ``src/simulation/materials.py``
(``MATERIAL_NAMES``) at runtime — this tool carries no material vocabulary
of its own (engine/15 §1 rule). A brand-new material (one config row) gets a
readable fallback wall automatically. The one non-material tilemap code,
SPACE (``level_loader.SPACE_CODE`` = 9), bakes fully transparent and is
declared in the manifest's ``[special.space]`` table.

Piece-index convention — "edge16" autotiling
============================================
Wall-family materials get a horizontal strip of 16 pieces. A wall tile's
piece index is its 4-bit edge bitmask: for each of N/E/S/W the bit is SET
when the neighbouring tile's material belongs to the SAME connectivity
group (the "wall family", declared in ``tileset.toml [groups]``, not code):

    bit 0 (+1) = N   neighbour at row-1  (image up)
    bit 1 (+2) = E   neighbour at col+1  (image right)
    bit 2 (+4) = S   neighbour at row+1  (image down)
    bit 3 (+8) = W   neighbour at col-1  (image left)

    piece index = N*1 + E*2 + S*4 + W*8   (0..15 == strip column)

A SET bit means the wall continues that way, so the art runs flush to that
tile edge; a CLEAR bit is an exposed wall face, chamfered (beveled) toward
that edge:

    index 0 (isolated)     index 5 (N|S run)      index 15 (interior)
      +----------+            |##########|           ############
      | /######\ |            |##########|           ############
      | ######## |            |##########|           ############
      | \######/ |            |##########|           ############
      +----------+          bevel E and W only     flat, no bevel at all

Normal maps are derived from the bevel height profile (slope-1 chamfer,
``BEVEL_FRAC`` of a tile wide, so bevel normals are honest 45-degree
diagonals). The channel-sign convention is pinned to the repo's existing
``tools/depth_to_normal.py`` (``n = normalize(-dh/dcol, -dh/drow, 1)``) —
the convention ``shaders/lighting.fs`` consumes at its default
``u_normal_y_sign = +1`` with the sim's row-down light directions. Flat
ground encodes exactly (128, 128, 255); a north bevel has G < 128, an east
bevel R > 128.

Run:
    python tools/make_tileset.py [--out art/tilesets/greybox] [--px 128] [--seed N]
"""
from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path

# Make project modules importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from simulation.materials import MATERIAL_NAMES  # noqa: E402


# ---------------------------------------------------------------------------
# edge16 bit assignments (see module docstring)
# ---------------------------------------------------------------------------
BIT_N, BIT_E, BIT_S, BIT_W = 1, 2, 4, 8
WALL_PIECES = 16                 # one piece per 4-bit edge mask

# ---------------------------------------------------------------------------
# Tunables (engine/15 §3: parameters, not magic constants sprinkled around)
# ---------------------------------------------------------------------------
DEFAULT_PX = 128                 # source-authoring resolution, px per tile edge
DEFAULT_SEED = 0                 # seeds the deck-plating noise only
BEVEL_FRAC = 1 / 8               # chamfer width as a fraction of the tile edge
FLOOR_VARIANTS = 4               # deck-plating variants for MAT_AIR
AUTOTILE_SCHEME = "edge16"       # manifest-declared; 47-case blob is a later drop-in

# Fixed top-left "workbench lamp" baked faintly into the diffuse so bevels
# read even before the engine's normal-map lighting touches them. Expressed
# in the same (-dh/dcol, -dh/drow, 1) frame as the stored normals.
_SHADE_LIGHT = np.array([-0.40, -0.50, 1.10], dtype=np.float32)
_SHADE_LIGHT /= np.sqrt((_SHADE_LIGHT ** 2).sum())
SHADE_AMBIENT = 0.60             # flat-light floor of the baked shading
SHADE_DIFFUSE = 0.40             # directional share (ambient + diffuse <= 1)

# ---------------------------------------------------------------------------
# Appearance recipes, keyed by canon material NAME. Only the LOOK lives here;
# the material SET comes from MATERIAL_NAMES at runtime (§1 rule). A material
# with no recipe falls back to a wall in its OWN connectivity group with a
# deterministic hue from its id, so a new config.toml row appears in the
# tileset without touching this file.
# ---------------------------------------------------------------------------
WALL_GROUP = "wall"              # the one v1 connectivity group (§3)
_WALL_RGBA = {
    "hull":  (128, 131, 138, 255),   # neutral grey — the default bulkhead
    "wood":  (146, 102, 57, 255),    # warm brown
    "door":  (208, 168, 62, 255),    # amber — a door must read at a glance
    "steel": (152, 168, 189, 255),   # cool blue-grey
    "glass": (130, 186, 224, 150),   # translucent blue (alpha < 255)
}
_DECK_RGB = (68, 70, 76)             # MAT_AIR: dark deck plating
_CRATE_RGBA = (168, 128, 78, 255)    # MAT_FURNITURE: crate


def fallback_wall_rgba(mat_id: int) -> tuple:
    """Deterministic, readable colour for a material with no recipe yet:
    golden-angle hue walk over the id keeps successive new materials apart."""
    hue = (mat_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.35, 0.62)
    return (round(r * 255), round(g * 255), round(b * 255), 255)


# ---------------------------------------------------------------------------
# Height field -> normals -> shaded diffuse (the whole greybox look)
# ---------------------------------------------------------------------------

def edge16_height(px: int, bevel_px: int, mask: int) -> np.ndarray:
    """Height field (float32, PIXEL units) of one edge16 wall piece.

    The wall top sits at ``bevel_px``; every edge whose mask bit is CLEAR is
    chamfered toward it at slope 1 (rise 1 px per px), so bevel normals are
    honest 45-degree diagonals at any resolution. Connected edges run flush
    to the tile border so adjacent pieces continue seamlessly; concave
    corners mitre naturally via the running minimum.
    """
    dist_lo = np.arange(px, dtype=np.float32) + 0.5   # px from the low edge
    h = np.full((px, px), float(bevel_px), dtype=np.float32)
    if not mask & BIT_N:
        h = np.minimum(h, dist_lo[:, None])           # ramp from the top edge
    if not mask & BIT_S:
        h = np.minimum(h, dist_lo[::-1, None])        # ... the bottom edge
    if not mask & BIT_W:
        h = np.minimum(h, dist_lo[None, :])           # ... the left edge
    if not mask & BIT_E:
        h = np.minimum(h, dist_lo[None, ::-1])        # ... the right edge
    return h


def normal_from_height(height_px: np.ndarray) -> np.ndarray:
    """Unit normals (H, W, 3 float32) from a pixel-unit height field.

    Sign convention pinned to ``tools/depth_to_normal.py`` (what
    ``shaders/lighting.fs`` expects at its default ``u_normal_y_sign = +1``):
    ``n = normalize(-dh/dcol, -dh/drow, 1)``. A north (image-top) bevel
    therefore encodes G < 128 and an east (image-right) bevel R > 128; flat
    ground encodes exactly (128, 128, 255).
    """
    gy, gx = np.gradient(height_px)                   # gy = d/d(row)
    n = np.stack([-gx, -gy, np.ones_like(height_px)], axis=-1)
    n /= np.sqrt((n * n).sum(axis=-1, keepdims=True))
    return n


def encode_normal(n: np.ndarray) -> np.ndarray:
    """[-1, 1] normals -> uint8 RGB (same rounding as depth_to_normal.py)."""
    return ((n * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8)


def flat_normal_strip(px: int, pieces: int) -> np.ndarray:
    """An all-(0,0,1) normal strip — floors are flat (engine/15 §3)."""
    strip = np.empty((px, px * pieces, 3), dtype=np.uint8)
    strip[...] = (128, 128, 255)
    return strip


def shade_rgba(rgba: tuple, normals: np.ndarray) -> np.ndarray:
    """Flat material-identity colour, lit once by the fixed workbench lamp so
    bevels read in unlit contexts; the real lighting is the engine's."""
    ndotl = np.clip((normals * _SHADE_LIGHT).sum(axis=-1), 0.0, None)
    bright = SHADE_AMBIENT + SHADE_DIFFUSE * ndotl
    out = np.empty(normals.shape[:2] + (4,), dtype=np.uint8)
    rgb = np.asarray(rgba[:3], dtype=np.float32) * bright[..., None]
    out[..., :3] = (np.clip(rgb, 0.0, 255.0) + 0.5).astype(np.uint8)
    out[..., 3] = rgba[3]
    return out


# ---------------------------------------------------------------------------
# Per-material strip builders
# ---------------------------------------------------------------------------

def wall_strips(px: int, bevel_px: int, rgba: tuple) -> tuple:
    """(diffuse RGBA strip, normal RGB strip) — 16 edge16 pieces, index=mask."""
    diffuse, normal = [], []
    for mask in range(WALL_PIECES):
        h = edge16_height(px, bevel_px, mask)
        n = normal_from_height(h)
        diffuse.append(shade_rgba(rgba, n))
        normal.append(encode_normal(n))
    return np.concatenate(diffuse, axis=1), np.concatenate(normal, axis=1)


def deck_tile(px: int, base_rgb: tuple, rng: np.random.Generator) -> np.ndarray:
    """One deck-plating floor variant (RGBA): 2x2 plates behind jittered
    seams, corner rivets, faint per-plate tone shifts, and a whisper of
    grain — subtle enough to read as flooring, varied enough that four
    variants break the tiling."""
    img = np.full((px, px, 3), base_rgb, dtype=np.float32)
    line = max(1, px // 32)                    # seam / rivet linework width
    jitter = max(1, px // 8)
    seam_c = px // 2 + int(rng.integers(-jitter, jitter + 1))
    seam_r = px // 2 + int(rng.integers(-jitter, jitter + 1))

    # Per-plate tone: each of the four plates sits at its own faint value.
    for r0, r1, c0, c1 in ((0, seam_r, 0, seam_c), (0, seam_r, seam_c, px),
                           (seam_r, px, 0, seam_c), (seam_r, px, seam_c, px)):
        img[r0:r1, c0:c1] += float(rng.integers(-5, 6))

    # Interior seams (darker), plus half-seams along the tile border so two
    # adjacent floor tiles compose a full seam between them.
    img[seam_r:seam_r + line, :] -= 16.0
    img[:, seam_c:seam_c + line] -= 16.0
    img[:line, :] -= 8.0
    img[-line:, :] -= 8.0
    img[:, :line] -= 8.0
    img[:, -line:] -= 8.0

    # Rivets: one near each tile corner, inset from the border.
    inset = max(2, px // 10)
    rivet = max(1, px // 32)
    for rr in (inset, px - inset - rivet):
        for cc in (inset, px - inset - rivet):
            img[rr:rr + rivet, cc:cc + rivet] -= 22.0

    img += rng.integers(-2, 3, size=(px, px, 1)).astype(np.float32)  # grain

    out = np.empty((px, px, 4), dtype=np.uint8)
    out[..., :3] = (np.clip(img, 0.0, 255.0) + 0.5).astype(np.uint8)
    out[..., 3] = 255
    return out


def crate_tile(px: int, bevel_px: int, rgba: tuple) -> tuple:
    """(diffuse RGBA, normal RGB) for MAT_FURNITURE: a crate — raised plank
    frame + X cross-brace over a recessed panel, chamfered on all four sides
    (a crate never autotiles), normals derived from that height field."""
    frame = max(2, px // 8)                    # plank frame width
    brace = max(2, px // 10)                   # diagonal brace width

    h = np.full((px, px), 0.75 * bevel_px, dtype=np.float32)   # recessed panel
    planks = np.zeros((px, px), dtype=bool)
    planks[:frame, :] = planks[-frame:, :] = True
    planks[:, :frame] = planks[:, -frame:] = True
    rr, cc = np.mgrid[0:px, 0:px]
    planks |= np.abs(rr - cc) < brace / 2.0                    # the X brace
    planks |= np.abs(rr + cc - (px - 1)) < brace / 2.0
    h[planks] = float(bevel_px)
    h = np.minimum(h, edge16_height(px, bevel_px, 0))          # isolated chamfer

    n = normal_from_height(h)
    diffuse = shade_rgba(rgba, n)
    dark = diffuse[..., :3].astype(np.float32)
    dark[planks] *= 0.88                       # planks read darker than panel
    diffuse[..., :3] = (dark + 0.5).astype(np.uint8)
    return diffuse, encode_normal(n)


# ---------------------------------------------------------------------------
# Manifest (tileset.toml) — the schema the P2 baker consumes
# ---------------------------------------------------------------------------

def render_manifest(px: int, seed: int, groups: dict, entries: list) -> str:
    """The tileset.toml text: explicit, boring, and generated in one place.

    ``groups`` is {group name: [member material names]} in declaration order;
    ``entries`` is [(material name, {key: value})] in material-id order.
    """
    lines = [
        "# greybox tileset — GENERATED by tools/make_tileset.py; edit the",
        "# generator (not this file), then regenerate:",
        f"#   python tools/make_tileset.py --px {px} --seed {seed} [--out ...]",
        "#",
        "# Schema (engine/15 §3, consumed by tools/bake_level_art.py):",
        "#   [tileset]           px_per_tile (source resolution), the autotile",
        "#                       scheme, and generation provenance.",
        "#   [groups]            connectivity groups — a wall tile's edge16 bit",
        "#                       is SET when the neighbour's material shares",
        "#                       its group (the wall family, not per-material",
        "#                       islands).",
        "#   [materials.<name>]  one table per canon material; names are the",
        "#                       canon set from src/simulation/materials.py",
        "#                       (ids stay in code, never in this file).",
        '#       mode    "wall"  = 16-piece edge16 strip (piece index =',
        "#                         N*1 + E*2 + S*4 + W*8); requires `group`.",
        '#               "floor" = non-autotiled variant strip; the baker picks',
        "#                         a variant per tile deterministically from",
        "#                         [bake].seed.",
        "#       pieces  piece count (strip width = pieces * px_per_tile).",
        "#       diffuse/normal  strip filenames relative to this manifest.",
        "#   [special.space]     the one non-material tilemap code, SPACE",
        "#                       (level_loader.SPACE_CODE) — bakes fully",
        "#                       transparent; the background starfield shows.",
        "",
        "[tileset]",
        'name = "greybox"',
        f"px_per_tile = {px}",
        f'autotile = "{AUTOTILE_SCHEME}"',
        'generator = "tools/make_tileset.py"',
        f"seed = {seed}",
        "",
        "[groups]",
    ]
    for group, members in groups.items():
        member_list = ", ".join(f'"{m}"' for m in members)
        lines.append(f"{group} = [{member_list}]")
    for name, entry in entries:
        lines.append("")
        lines.append(f"[materials.{name}]")
        for key, value in entry.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            else:
                lines.append(f"{key} = {value}")
    lines += [
        "",
        "[special.space]",
        'mode = "transparent"',
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_tileset(out_dir, px: int = DEFAULT_PX, seed: int = DEFAULT_SEED) -> Path:
    """Generate every strip + tileset.toml into ``out_dir``; returns the
    manifest path. Deterministic: same (px, seed) => byte-identical files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bevel_px = max(2, round(px * BEVEL_FRAC))

    groups: dict = {}
    entries: list = []
    for mat_id in sorted(MATERIAL_NAMES):
        name = MATERIAL_NAMES[mat_id]
        if name == "air":                      # walkable interior: deck plating
            tiles = [deck_tile(px, _DECK_RGB,
                               np.random.default_rng([seed, mat_id, variant]))
                     for variant in range(FLOOR_VARIANTS)]
            diffuse = np.concatenate(tiles, axis=1)
            normal = flat_normal_strip(px, FLOOR_VARIANTS)
            stem, entry = "air_floor", {"mode": "floor", "pieces": FLOOR_VARIANTS}
        elif name == "furniture":              # crates
            diffuse, normal = crate_tile(px, bevel_px, _CRATE_RGBA)
            stem, entry = "furniture_crate", {"mode": "floor", "pieces": 1}
        else:                                  # wall family (+ future fallbacks)
            rgba = _WALL_RGBA.get(name)
            group = WALL_GROUP if rgba is not None else name
            if rgba is None:                   # new material, no recipe yet
                rgba = fallback_wall_rgba(mat_id)
            diffuse, normal = wall_strips(px, bevel_px, rgba)
            groups.setdefault(group, []).append(name)
            stem = f"{name}_wall"
            entry = {"mode": "wall", "group": group, "pieces": WALL_PIECES}

        entry["diffuse"] = f"{stem}.png"
        entry["normal"] = f"{stem}_n.png"
        Image.fromarray(diffuse).save(out_dir / entry["diffuse"])
        Image.fromarray(normal).save(out_dir / entry["normal"])
        entries.append((name, entry))

    manifest_path = out_dir / "tileset.toml"
    manifest_path.write_text(render_manifest(px, seed, groups, entries),
                             encoding="utf-8", newline="\n")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate the procedural greybox dev tileset "
                    "(engine/15 §3 — PNG strips + tileset.toml).")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "art" / "tilesets" / "greybox",
                    help="output tileset directory "
                         "(default: <repo>/art/tilesets/greybox)")
    ap.add_argument("--px", type=int, default=DEFAULT_PX,
                    help="source resolution, px per tile edge "
                         "(default: %(default)s)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="deck-plating noise seed (default: %(default)s)")
    return ap


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = build_tileset(args.out, px=args.px, seed=args.seed)
    n_png = len(list(Path(args.out).glob("*.png")))
    print(f"greybox tileset -> {args.out}")
    print(f"  {n_png} PNG strips at {args.px} px/tile (seed {args.seed}) "
          f"+ {manifest.name}")


if __name__ == "__main__":
    main()
