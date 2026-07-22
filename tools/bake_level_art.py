r"""Autotile baker — tilemap.csv + tileset -> baked level art (engine/15 §4, P2).

``tilemap.csv + tileset (+ seed) -> baked_diffuse.png + baked_normal.png`` at a
target resolution (default 64 px/tile, engine/15 §3), composed from the P1
tileset strips (``tools/make_tileset.py`` — piece index = edge16 bitmask). The
output is an ordinary format-v2 level: loader and renderer never know the art
was baked.

Pure function of its inputs: same tilemap + tileset files + px + seed produce
byte-identical PNGs — no wall clock, no unseeded randomness (golden-image
tested). Importable API for the P3 editor's live preview, CLI for save-time /
procgen bakes.

Compose rule (per tile)
=======================
  - SPACE (``level_loader.SPACE_CODE``) -> fully transparent, alpha 0 in BOTH
    outputs (the renderer's tilemap-derived vacuum mask punches the hole; the
    baked alpha keeps the PNG honest and editor previews correct).
  - every non-SPACE tile first gets a FLOOR UNDER-LAYER: the AIR deck-plating
    variant for its position (see the hash below), then its own material piece
    is alpha-composited on top:
      * AIR            -> floor only.
      * wall-mode mats (hull/wood/door/steel/glass) -> the edge16 piece
        selected by the 4-bit N*1+E*2+S*4+W*8 neighbour mask; a bit is SET
        iff the neighbour's material shares the tile's ``[groups]`` entry
        (off-map, SPACE and non-group neighbours = bit clear).
      * floor-mode mats (furniture) -> their position-hash variant.
    Glass (alpha 150) therefore shows the deck plating through the pane.
  - baked normal = the material's OWN normal piece (floor normal for AIR),
    alpha 255; normals are not alpha-blended — the top surface owns the tile.

Floor-variant hash (pinned by golden test)
==========================================
  variant = (tx*73856093 + ty*19349663 + seed) % pieces
POSITION-based (fixed primes, exact integer arithmetic), so a region re-bake
of any rectangle reproduces the full bake exactly — no sequential RNG state.

Downscale (128 px source -> 64 px bake, deterministic)
======================================================
Strips are downscaled per-piece BEFORE compositing, by an integer BOX filter:
the source px must be an integer multiple of the target px; each k x k block
averages with round-half-up in uint32 arithmetic (bit-exact everywhere).
Normal strips are decoded to vectors, box-averaged, RENORMALIZED to unit
length, and re-encoded with make_tileset's rounding — a naively resized
normal map is denormalized on every bevel diagonal.

Region re-bake API (the P3 editor's live preview)
=================================================
    bake_region(tilemap, tileset, tile_rect, *, px_per_tile=64, seed=0)
        -> BakedPatch(diffuse, normal, rect)
``tilemap`` is the FULL (H, W) grid of v2 codes (edge masks read neighbours
outside the rect); ``tile_rect`` = (tx0, ty0, tw, th) in tiles, clipped to the
grid (ValueError when the intersection is empty); ``rect`` on the result is
the clipped rect actually baked. The patch equals the corresponding crop of a
full bake — tested property.

CLI
===
    python tools/bake_level_art.py <level_name_or_path> \
        [--tileset art/tilesets/greybox] [--px-per-tile 64] [--seed N]

Bakes ``levels/<name>/``: writes ``baked_diffuse.png`` + ``baked_normal.png``
into the level folder and rewrites the ``[art.bare]`` (diffuse/normal),
``[art.align]`` (offset [0,0], px_per_tile = the bake value) and ``[bake]``
(tileset/px_per_tile/seed) blocks of ``level.toml``. Omitted flags fall back
to the level's existing ``[bake]`` values, then to the chapter defaults — so
``python tools/bake_level_art.py <name>`` re-bakes a tiled level exactly as
recorded. Emissive output exists only when the manifest declares emissive
pieces (greybox: none — the file is skipped entirely).

level.toml writeback (Arc C9 rider — A2 accepted gap, closed)
==============================================================
The three blocks above are ``level_lib``-managed families
(``"art.bare"``/``"art.align"``/``"bake"``, all single-table): the write
goes through :func:`level_lib.write_managed_blocks` — a same-directory
temp file + ``os.replace``, so a crash mid-write can never leave a torn
``level.toml`` (the original hand-rolled regex line-upsert, replaced by
this port, wrote directly with no such guarantee). Every byte outside
these three tables is preserved exactly, matching every other managed
family; a stray hand-comment INSIDE one of them does not survive a
rewrite (whole-block replace, not the old key-level upsert) — the same
trade-off ``[[spawn]]``/``[[entity]]``/etc. already made, and no shipped
level carries one. :func:`bake_level` computes the bake (PNGs) AND does
this write itself (a self-contained, standalone save — CLI/procgen);
:func:`compute_bake_replacements` does only the PNG-writing half and
returns the replacements dict UNWRITTEN, for a caller (the map editor's
Ctrl+S) that wants to fold them into its OWN ``write_managed_blocks``/
``LevelHandle.save()`` call so the whole save (spawn/light/entity/water/
wire/art/bake) lands as ONE atomic write.
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Make project modules importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

import level_lib  # noqa: E402
from level_loader import SPACE_CODE  # noqa: E402
from simulation.materials import MATERIAL_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Conventions pinned by P1 (make_tileset.py module docstring)
# ---------------------------------------------------------------------------
BIT_N, BIT_E, BIT_S, BIT_W = 1, 2, 4, 8
WALL_PIECES = 16                     # edge16: one piece per 4-bit mask

# ---------------------------------------------------------------------------
# Tunables / pinned constants (engine/15 §3-§4)
# ---------------------------------------------------------------------------
DEFAULT_PX_PER_TILE = 64             # bake output resolution (chapter §3)
DEFAULT_TILESET = "art/tilesets/greybox"
DIFFUSE_FILENAME = "baked_diffuse.png"
NORMAL_FILENAME = "baked_normal.png"
# Floor-variant position hash: fixed spatial-hash primes, pinned by golden
# test — changing either re-bakes every tiled level differently.
FLOOR_HASH_P1 = 73856093
FLOOR_HASH_P2 = 19349663

NAME_TO_CODE = {name: code for code, name in MATERIAL_NAMES.items()}


# ---------------------------------------------------------------------------
# Pure per-tile selectors
# ---------------------------------------------------------------------------

def floor_variant(tx: int, ty: int, seed: int, pieces: int) -> int:
    """Deterministic floor/furniture variant for tile (tx, ty).

    POSITION-based (not sequential): exact Python-int arithmetic, so a region
    re-bake picks the same variant as the full bake for every tile. Python's
    ``%`` is non-negative for positive ``pieces``, seed may be any int.
    """
    return (int(tx) * FLOOR_HASH_P1 + int(ty) * FLOOR_HASH_P2 + int(seed)) \
        % int(pieces)


def edge16_mask(tilemap: np.ndarray, tx: int, ty: int, same_codes) -> int:
    """4-bit edge mask of tile (tx, ty): N*1 + E*2 + S*4 + W*8, a bit SET iff
    the neighbour's CSV code is in ``same_codes`` (the tile's connectivity
    group as codes). Off-map neighbours are clear; SPACE is never in a group.
    N = row-1 (image up), matching the P1 strip convention.
    """
    h, w = tilemap.shape
    mask = 0
    if ty > 0 and int(tilemap[ty - 1, tx]) in same_codes:
        mask |= BIT_N
    if tx < w - 1 and int(tilemap[ty, tx + 1]) in same_codes:
        mask |= BIT_E
    if ty < h - 1 and int(tilemap[ty + 1, tx]) in same_codes:
        mask |= BIT_S
    if tx > 0 and int(tilemap[ty, tx - 1]) in same_codes:
        mask |= BIT_W
    return mask


# ---------------------------------------------------------------------------
# Deterministic pixel kit — integer compositing + box downscale
# ---------------------------------------------------------------------------

def alpha_composite_over(src_rgba: np.ndarray, dst_rgba: np.ndarray) -> np.ndarray:
    """Straight-alpha 'src over dst' in exact integer arithmetic (uint8 in/out).

    out = (src*a + dst*(255-a) + 127) // 255 per RGB channel (round-half-up of
    x/255); alpha composes the same way, so over the baker's opaque floor
    under-layer the result alpha is exactly 255. No float anywhere —
    bit-identical on every machine.
    """
    src = src_rgba.astype(np.uint32)
    dst = dst_rgba.astype(np.uint32)
    a = src[..., 3:4]
    out = np.empty(src.shape, dtype=np.uint32)
    out[..., :3] = (src[..., :3] * a + dst[..., :3] * (255 - a) + 127) // 255
    out[..., 3:4] = a + (dst[..., 3:4] * (255 - a) + 127) // 255
    return out.astype(np.uint8)


def box_downscale_u8(img: np.ndarray, k: int) -> np.ndarray:
    """Integer k x k box filter (area average, round-half-up) over a (H, W, C)
    uint8 image whose dimensions are multiples of k. uint32 block sums are
    exact, so the filter is deterministic bit-for-bit.
    """
    h, w, c = img.shape
    s = img.reshape(h // k, k, w // k, k, c).astype(np.uint32).sum(axis=(1, 3))
    return ((s + (k * k) // 2) // (k * k)).astype(np.uint8)


def downscale_normal_strip(strip_rgb: np.ndarray, k: int) -> np.ndarray:
    """Box-downscale a normal strip and RENORMALIZE to unit length.

    Naive resizing denormalizes a normal map (the average of two unit vectors
    is shorter than 1 — visibly dimming every bevel diagonal). The block sum
    is exact integer; decode is affine, so ``mean(decode(u)) ==
    decode(mean(u))`` and only the final normalize touches float (IEEE
    mul/add/div/sqrt — correctly rounded, cross-machine identical). Re-encoded
    with make_tileset's rounding, so a flat region stays exactly
    (128, 128, 255).
    """
    h, w, _ = strip_rgb.shape
    s = strip_rgb.reshape(h // k, k, w // k, k, 3).astype(np.int64).sum(axis=(1, 3))
    v = s.astype(np.float64) / (k * k * 127.5) - 1.0
    length = np.sqrt((v * v).sum(axis=-1, keepdims=True))
    v = v / np.maximum(length, 1e-12)     # z > 0 everywhere; guard regardless
    return ((v * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tileset — manifest + strips (schema: art/tilesets/<name>/tileset.toml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TilesetMaterial:
    name: str
    mode: str                 # "wall" | "floor"
    group: str | None         # connectivity group (wall mode only)
    pieces: int
    diffuse: np.ndarray       # (px, pieces*px, 4) uint8 RGBA
    normal: np.ndarray        # (px, pieces*px, 3) uint8 RGB


@dataclass(frozen=True)
class Tileset:
    path: Path
    px: int                                  # source resolution, px per tile
    materials: dict                          # {canon name: TilesetMaterial}
    groups: dict                             # {group: tuple(member names)}
    group_codes: dict                        # {group: frozenset(member CODES)}
    has_emissive: bool                       # any [materials.*] emissive strip


def load_tileset(tileset_dir) -> Tileset:
    """Load + validate a tileset directory (manifest and every strip).

    The manifest is the truth (engine/15 §3): groups, modes and piece counts
    come from ``tileset.toml``, material NAMES must be canon
    (``MATERIAL_NAMES``) — the baker carries no vocabulary of its own.
    """
    tileset_dir = Path(tileset_dir)
    manifest_path = tileset_dir / "tileset.toml"
    if not manifest_path.is_file():
        raise ValueError(f"tileset.toml not found in {tileset_dir}")
    with open(manifest_path, "rb") as f:
        raw = tomllib.load(f)

    ts_tbl = raw.get("tileset", {})
    px = int(ts_tbl.get("px_per_tile", 0))
    if px <= 0:
        raise ValueError(f"{manifest_path}: [tileset] px_per_tile must be > 0")
    autotile = ts_tbl.get("autotile", "edge16")
    if autotile != "edge16":
        raise ValueError(
            f"{manifest_path}: autotile scheme {autotile!r} not supported "
            f"(this baker speaks edge16; the 47-case blob set is a later "
            f"manifest upgrade)")

    groups = {g: tuple(members) for g, members in raw.get("groups", {}).items()}
    for g, members in groups.items():
        for m in members:
            if m not in NAME_TO_CODE:
                raise ValueError(
                    f"{manifest_path}: [groups] {g} names unknown material "
                    f"{m!r} (canon set: {sorted(NAME_TO_CODE)})")
    group_codes = {g: frozenset(NAME_TO_CODE[m] for m in members)
                   for g, members in groups.items()}

    materials: dict = {}
    has_emissive = False
    for name, entry in raw.get("materials", {}).items():
        if name not in NAME_TO_CODE:
            raise ValueError(
                f"{manifest_path}: [materials.{name}] is not a canon material "
                f"(canon set: {sorted(NAME_TO_CODE)})")
        mode = entry.get("mode")
        if mode not in ("wall", "floor"):
            raise ValueError(
                f"{manifest_path}: materials.{name}.mode must be 'wall' or "
                f"'floor', got {mode!r}")
        pieces = int(entry.get("pieces", 0))
        group = entry.get("group")
        if mode == "wall":
            if pieces != WALL_PIECES:
                raise ValueError(
                    f"{manifest_path}: materials.{name}: edge16 wall strips "
                    f"have {WALL_PIECES} pieces, got {pieces}")
            if group not in group_codes:
                raise ValueError(
                    f"{manifest_path}: materials.{name}.group {group!r} is "
                    f"not declared in [groups]")
        elif pieces < 1:
            raise ValueError(
                f"{manifest_path}: materials.{name}: pieces must be >= 1")

        diffuse = np.asarray(Image.open(tileset_dir / entry["diffuse"])
                             .convert("RGBA"))
        normal = np.asarray(Image.open(tileset_dir / entry["normal"])
                            .convert("RGB"))
        for label, arr, ch in (("diffuse", diffuse, 4), ("normal", normal, 3)):
            want = (px, pieces * px, ch)
            if arr.shape != want:
                raise ValueError(
                    f"{manifest_path}: materials.{name}.{label}: strip shape "
                    f"{arr.shape} != expected {want} (pieces * px_per_tile)")
        if "emissive" in entry:
            has_emissive = True
        materials[name] = TilesetMaterial(
            name=name, mode=mode, group=group, pieces=pieces,
            diffuse=diffuse, normal=normal)

    air = materials.get("air")
    if air is None or air.mode != "floor":
        raise ValueError(
            f"{manifest_path}: tileset must declare a floor-mode 'air' strip "
            f"(the deck-plating under-layer)")
    space_mode = raw.get("special", {}).get("space", {}).get("mode")
    if space_mode != "transparent":
        raise ValueError(
            f"{manifest_path}: [special.space] mode must be 'transparent' "
            f"(SPACE is the one non-material code), got {space_mode!r}")

    return Tileset(path=tileset_dir, px=px, materials=materials,
                   groups=groups, group_codes=group_codes,
                   has_emissive=has_emissive)


def _scaled_pieces(ts: Tileset, px_out: int) -> dict:
    """Per-material piece lists at the bake resolution:
    {name: ([diffuse RGBA pieces], [normal RGB pieces])}.

    The source px must be an integer multiple of the target (128 -> 64 is the
    chapter default 2x supersample); px_out == px is a byte-exact no-op.
    Downscaling the whole strip == per-piece: block boundaries align with
    piece boundaries because px is a multiple of px_out.
    """
    if px_out <= 0:
        raise ValueError(f"px_per_tile must be > 0, got {px_out}")
    if ts.px % px_out != 0:
        raise ValueError(
            f"px_per_tile {px_out} must divide the tileset's source "
            f"resolution {ts.px} (integer box filter keeps the bake "
            f"deterministic)")
    k = ts.px // px_out
    out = {}
    for name, m in ts.materials.items():
        if k == 1:
            d, n = m.diffuse, m.normal
        else:
            d = box_downscale_u8(m.diffuse, k)
            n = downscale_normal_strip(m.normal, k)
        out[name] = (
            [d[:, i * px_out:(i + 1) * px_out] for i in range(m.pieces)],
            [n[:, i * px_out:(i + 1) * px_out] for i in range(m.pieces)],
        )
    return out


# ---------------------------------------------------------------------------
# The bake — region core (a full bake is the whole-grid region)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BakedPatch:
    diffuse: np.ndarray       # (th*px, tw*px, 4) uint8 RGBA
    normal: np.ndarray        # (th*px, tw*px, 4) uint8 RGBA (alpha 0 = SPACE)
    rect: tuple               # (tx0, ty0, tw, th) — the CLIPPED rect baked


def bake_region(tilemap, tileset: Tileset, tile_rect, *,
                px_per_tile: int = DEFAULT_PX_PER_TILE,
                seed: int = 0) -> BakedPatch:
    """Bake one tile rectangle of ``tilemap`` (full grid of v2 CSV codes).

    ``tile_rect`` = (tx0, ty0, tw, th) in tiles; it is clipped to the grid
    (ValueError if the intersection is empty) and the clipped rect is returned
    on the patch. Edge masks read neighbours from the FULL tilemap, so the
    patch equals the corresponding crop of a full bake — the P3 editor blits
    ``patch.diffuse``/``patch.normal`` at ``patch.rect`` over its preview.
    """
    tilemap = np.asarray(tilemap)
    if tilemap.ndim != 2:
        raise ValueError(f"tilemap must be 2D, got shape {tilemap.shape}")
    h, w = tilemap.shape
    rx0, ry0, rw, rh = (int(v) for v in tile_rect)
    x0, y0 = max(0, rx0), max(0, ry0)
    x1, y1 = min(w, rx0 + rw), min(h, ry0 + rh)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"tile_rect {tuple(tile_rect)} does not intersect the "
            f"{w}x{h}-tile grid")
    px = int(px_per_tile)
    pieces = _scaled_pieces(tileset, px)
    air_d, air_n = pieces["air"]
    n_air = tileset.materials["air"].pieces

    diffuse = np.zeros(((y1 - y0) * px, (x1 - x0) * px, 4), dtype=np.uint8)
    normal = np.zeros_like(diffuse)
    for ty in range(y0, y1):
        for tx in range(x0, x1):
            code = int(tilemap[ty, tx])
            if code == SPACE_CODE:
                continue                      # transparent in BOTH outputs
            name = MATERIAL_NAMES.get(code)
            if name is None:
                raise ValueError(
                    f"tilemap ({tx}, {ty}) carries unknown code {code}; "
                    f"valid: material ids {sorted(MATERIAL_NAMES)} + "
                    f"{SPACE_CODE} (SPACE)")
            entry = tileset.materials.get(name)
            if entry is None:
                raise ValueError(
                    f"tileset {tileset.path} declares no strip for material "
                    f"'{name}' (tile ({tx}, {ty}))")
            # Floor under-layer: the AIR deck variant for this position.
            fv = floor_variant(tx, ty, seed, n_air)
            d, n = air_d[fv], air_n[fv]
            if name != "air":
                if entry.mode == "wall":
                    piece = edge16_mask(tilemap, tx, ty,
                                        tileset.group_codes[entry.group])
                else:
                    piece = floor_variant(tx, ty, seed, entry.pieces)
                mat_d, mat_n = pieces[name]
                d = alpha_composite_over(mat_d[piece], d)
                n = mat_n[piece]              # the top surface owns the tile
            r0, c0 = (ty - y0) * px, (tx - x0) * px
            diffuse[r0:r0 + px, c0:c0 + px] = d
            normal[r0:r0 + px, c0:c0 + px, :3] = n
            normal[r0:r0 + px, c0:c0 + px, 3] = 255
    return BakedPatch(diffuse=diffuse, normal=normal,
                      rect=(x0, y0, x1 - x0, y1 - y0))


def bake_full(tilemap, tileset: Tileset, *,
              px_per_tile: int = DEFAULT_PX_PER_TILE,
              seed: int = 0) -> BakedPatch:
    """Bake the whole grid (== bake_region over (0, 0, W, H))."""
    tilemap = np.asarray(tilemap)
    return bake_region(tilemap, tileset,
                       (0, 0, tilemap.shape[1], tilemap.shape[0]),
                       px_per_tile=px_per_tile, seed=seed)


# ---------------------------------------------------------------------------
# level.toml writeback — level_lib's atomic managed-block writer (Arc C9
# rider: ported off the original hand-rolled regex line-upsert)
# ---------------------------------------------------------------------------

def write_bake_blocks(toml_path, *, tileset_rel: str, px_per_tile: int,
                      seed: int, write_bak: bool = True):
    """Rewrite the [art.bare] + [art.align] + [bake] blocks of ``toml_path``
    — ONE atomic write through :func:`level_lib.write_managed_blocks` (a
    same-directory temp file + ``os.replace``; the original hand-rolled
    regex upsert wrote directly, with no such guarantee). Every byte outside
    these three tables is preserved exactly, matching every other
    level_lib-managed family; a stray hand-comment INSIDE one of them does
    NOT survive a rewrite (whole-block replace, not the old key-level
    upsert — no shipped level carries one). Returns the .bak path, or None
    when ``write_bak`` is False.
    """
    toml_path = Path(toml_path)
    replacements = {
        "art.bare": lambda nl: level_lib.format_art_bare_lines(
            DIFFUSE_FILENAME, NORMAL_FILENAME, nl),
        "art.align": lambda nl: level_lib.format_art_align_lines(
            px_per_tile, nl),
        "bake": lambda nl: level_lib.format_bake_lines(
            tileset_rel, px_per_tile, seed, nl),
    }
    return level_lib.write_managed_blocks(toml_path, replacements,
                                          write_bak=write_bak)


# ---------------------------------------------------------------------------
# Level bake — the CLI / procgen entry point
# ---------------------------------------------------------------------------

def compute_bake_replacements(level_dir, tileset=None, px_per_tile=None,
                              seed=None):
    """Bake one level folder's art (writes the PNGs to disk) and return the
    level_lib ``[art.bare]``/``[art.align]``/``[bake]`` REPLACEMENTS —
    NOT written to ``level.toml`` here (Arc C9 rider). The caller folds
    these into its OWN ``write_managed_blocks``/``LevelHandle.save()``
    replacements dict so the art/bake TOML write composes atomically (one
    temp+rename) with whatever other families (spawn/light/entity/water/
    wire) that save touches — see ``tools/map_editor.py``'s Ctrl+S handler
    and ``tools/play_scratch.py``'s dirty-bake branch. A caller with
    nothing else to save alongside should just call :func:`bake_level`,
    which does this AND the write together.

    ``None`` parameters fall back to the level's existing ``[bake]``
    values, then to the chapter defaults (greybox, 64 px/tile, seed 0).
    Returns ``(replacements, summary)`` — ``summary`` matches
    :func:`bake_level`'s own dict minus ``toml_bak`` (the caller's own
    write governs that).
    """
    level_dir = Path(level_dir)
    toml_path = level_dir / "level.toml"
    if not toml_path.is_file():
        raise ValueError(f"level.toml not found in {level_dir}")
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)
    version = str(raw.get("version", "")).strip()
    if version != "2":
        raise ValueError(
            f"{toml_path}: the baker speaks level format v2 only (codes ARE "
            f"canon material ids); this level is version {version!r}")
    tilemap_rel = raw.get("tilemap")
    if not tilemap_rel:
        raise ValueError(f"{toml_path}: missing required 'tilemap' field")
    tilemap = np.loadtxt(level_dir / tilemap_rel, delimiter=",",
                         dtype=np.int32)
    if tilemap.ndim != 2:
        raise ValueError(f"tilemap must be 2D, got shape {tilemap.shape}")

    bake_tbl = raw.get("bake", {})
    if not isinstance(bake_tbl, dict):
        raise ValueError(f"{toml_path}: [bake] must be a table")
    tileset_arg = tileset if tileset is not None \
        else bake_tbl.get("tileset", DEFAULT_TILESET)
    px = int(px_per_tile) if px_per_tile is not None \
        else int(bake_tbl.get("px_per_tile", DEFAULT_PX_PER_TILE))
    seed = int(seed) if seed is not None else int(bake_tbl.get("seed", 0))

    tileset_dir = Path(tileset_arg)
    if not tileset_dir.is_absolute():
        tileset_dir = ROOT / tileset_dir
    ts = load_tileset(tileset_dir)
    if ts.has_emissive:
        raise NotImplementedError(
            f"tileset {ts.path} declares emissive strips; the emissive "
            f"compose lands with the first emissive tileset (engine/15 §7 "
            f"P6) — P2 bakes diffuse + normal only")

    patch = bake_full(tilemap, ts, px_per_tile=px, seed=seed)
    diffuse_path = level_dir / DIFFUSE_FILENAME
    normal_path = level_dir / NORMAL_FILENAME
    Image.fromarray(patch.diffuse).save(diffuse_path)
    Image.fromarray(patch.normal).save(normal_path)

    # Record the tileset repo-relative (posix) when it lives under the repo.
    try:
        tileset_rel = tileset_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        tileset_rel = Path(tileset_arg).as_posix()

    replacements = {
        "art.bare": lambda nl: level_lib.format_art_bare_lines(
            DIFFUSE_FILENAME, NORMAL_FILENAME, nl),
        "art.align": lambda nl: level_lib.format_art_align_lines(px, nl),
        "bake": lambda nl: level_lib.format_bake_lines(
            tileset_rel, px, seed, nl),
    }
    summary = {
        "diffuse": diffuse_path,
        "normal": normal_path,
        "tileset": tileset_rel,
        "px_per_tile": px,
        "seed": seed,
        "size_tiles": (int(tilemap.shape[1]), int(tilemap.shape[0])),
    }
    return replacements, summary


def bake_level(level_dir, tileset=None, px_per_tile=None, seed=None, *,
               write_bak: bool = True) -> dict:
    """Full bake of one level folder + level.toml writeback — a
    SELF-CONTAINED, standalone save (CLI / procgen / any caller with no
    other families to compose atomically with).

    ``None`` parameters fall back to the level's existing ``[bake]`` values,
    then to the chapter defaults (greybox, 64 px/tile, seed 0) — a bare
    ``bake_level(dir)`` re-bakes a tiled level exactly as recorded. The
    level.toml write goes through :func:`level_lib.write_managed_blocks`
    (Arc C9 rider — one atomic temp+rename, closing the A2 gap); ``write_bak``
    still means "one .bak of the pre-write bytes", same contract as before.
    A caller that DOES have other families to save in the SAME atomic write
    (the map editor's Ctrl+S, play_scratch's dirty-bake branch) should call
    :func:`compute_bake_replacements` directly instead — see those callers.
    Returns a summary dict (paths + the resolved bake parameters + the .bak
    path).
    """
    level_dir = Path(level_dir)
    replacements, summary = compute_bake_replacements(
        level_dir, tileset=tileset, px_per_tile=px_per_tile, seed=seed)
    toml_bak = level_lib.write_managed_blocks(
        level_dir / "level.toml", replacements, write_bak=write_bak)
    summary["toml_bak"] = toml_bak
    return summary


def _resolve_level_dir(level: str) -> Path:
    """<name> under levels/, or a direct path to a level folder."""
    direct = Path(level)
    if (direct / "level.toml").is_file():
        return direct
    named = ROOT / "levels" / level
    if (named / "level.toml").is_file():
        return named
    raise ValueError(
        f"no level.toml under {direct} or {named} — pass a level name from "
        f"levels/ or a path to a level folder")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Bake tiled-level art: tilemap.csv + tileset -> "
                    "baked_diffuse.png + baked_normal.png + level.toml "
                    "[art]/[bake] blocks (engine/15 §4).")
    ap.add_argument("level",
                    help="level folder name under levels/ (or a path to a "
                         "level folder)")
    ap.add_argument("--tileset", default=None,
                    help="tileset directory (default: the level's "
                         f"[bake].tileset, else {DEFAULT_TILESET})")
    ap.add_argument("--px-per-tile", type=int, default=None,
                    help="bake output resolution, px per tile (default: the "
                         f"level's [bake].px_per_tile, else "
                         f"{DEFAULT_PX_PER_TILE})")
    ap.add_argument("--seed", type=int, default=None,
                    help="floor-variant seed (default: the level's "
                         "[bake].seed, else 0)")
    return ap


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    level_dir = _resolve_level_dir(args.level)
    summary = bake_level(level_dir, tileset=args.tileset,
                         px_per_tile=args.px_per_tile, seed=args.seed)
    w, h = summary["size_tiles"]
    print(f"baked {level_dir.name}: {w}x{h} tiles @ "
          f"{summary['px_per_tile']} px/tile (seed {summary['seed']}, "
          f"tileset {summary['tileset']})")
    print(f"  {summary['diffuse'].name} + {summary['normal'].name}; "
          f"level.toml blocks updated (backup: {summary['toml_bak'].name})")


if __name__ == "__main__":
    main()
