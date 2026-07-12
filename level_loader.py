"""Level loader: reads a level folder, validates assets, returns a LevelData object.

A "level" lives at levels/<name>/ and contains:
  - level.toml      : metadata (required)
  - tilemap.csv     : physics tile grid (required)
  - diffuse.png     : base color art (required)
  - normal.png      : Laigter normal map (optional)
  - emissive_mask.png  : pixels that glow (optional)
  - emissive_bloom.png : blurred halos (optional)
  - wall_mask.png   : overrides CSV-derived walls (optional)

The CSV is the source of truth for physics. Art assets are for rendering only.

Level format v2 adds an ``[art]`` block (level_editor_and_format_v2_proposal
§1.2/§1.3): layered art (``[art.bare]`` / ``[art.furniture]`` /
``[art.destroyed]``, each with diffuse/normal/specular) plus a non-destructive
grid alignment (``[art.align]`` with ``offset_px`` + ``px_per_tile``; the
latter is a scalar or a per-axis ``[x, y]`` pair — the v2 art's proportions
differ from the tilemap per axis).
``[art.bare]`` is the new spelling of the old flat ``diffuse``/``normal`` keys;
the flat keys keep working unchanged (v1 levels parse bit-identically, and v2
levels may still use them). Furniture/destroyed layers are stored as paths now
and consumed by the layer compose in F3.
"""
from __future__ import annotations

import math
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# The unified material ids live in ``src/simulation/materials.py``. Ensure
# ``src/`` is importable even when a caller (e.g. a focused unit test) only put
# the repo root on sys.path — every other entry point already adds ``src/``.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


SUPPORTED_VERSIONS = {"1", "2"}

# v2 CSV vocabulary (level format v2 §1.1): codes ARE canon material ids from
# ``simulation.materials``, plus exactly ONE reserved non-material code:
SPACE_CODE = 9   # SPACE — MAT_AIR + is_vacuum (the one thing that isn't a material)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_size(path: Path) -> tuple:
    """Read (width, height) from a PNG header without decoding the image.

    Pure-Python IHDR parse (the IHDR chunk is mandatory and always first), so
    the loader can compute the default ``px_per_tile`` (= art_width /
    grid_width) without an imaging dependency. Raises ValueError on anything
    that is not a well-formed PNG.
    """
    with open(path, "rb") as f:
        head = f.read(33)   # 8 signature + 8 chunk header + 13 IHDR payload
    if len(head) < 24 or head[:8] != _PNG_SIGNATURE or head[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG file (cannot read dimensions): {path}")
    width = int.from_bytes(head[16:20], "big")
    height = int.from_bytes(head[20:24], "big")
    return width, height


def tile_to_art_px(tile_x: float, tile_y: float,
                   offset_px, px_per_tile) -> tuple:
    """Align transform (level format v2 §1.3): grid tile -> art pixel.

    Art pixel ``offset_px`` lands on grid (0, 0); ``px_per_tile`` art pixels
    span one tile — a scalar (same scale both axes) or an ``(x, y)`` pair
    (the v2 art's proportions differ from the tilemap per axis). Single
    source of truth for the renderer's art src-rect and the ALIGN tooling
    (``tools/align_level_art.py``). Accepts fractional tiles.
    """
    if isinstance(px_per_tile, (list, tuple)):
        ppt_x, ppt_y = float(px_per_tile[0]), float(px_per_tile[1])
    else:
        ppt_x = ppt_y = float(px_per_tile)
    return (float(offset_px[0]) + float(tile_x) * ppt_x,
            float(offset_px[1]) + float(tile_y) * ppt_y)


@dataclass
class SpawnEntry:
    """One unit spawn declared in level.toml. Fed to Simulation.add_unit."""
    name: str
    team: int           # 0 = marine, 1 = zombie
    x: float            # physics-tile x (top-left of footprint)
    y: float            # physics-tile y
    footprint: int = 3  # side length of unit's square footprint in tiles


LIGHT_KINDS = ("static", "beacon")

# [[light]] keys the loader REJECTS outright (P4 design §2.2, critique M2):
# level lights are RENDER-ONLY — `heat` is the one synced ray output (a leak
# would silently diverge interactive sessions from their headless replays,
# since goldens never run the render light pass) and `jitter` pulls C++ RNG.
# src/level_lights.py hard-pins both to 0.0; the schema never carries them.
_LIGHT_FORBIDDEN_KEYS = ("heat", "jitter")


@dataclass
class LightEntry:
    """One ``[[light]]`` entity declared in level.toml (engine/15 §2.2, P4).

    Render-only in P4 (Erik's locked call 2026-07-07): consumed by main.py
    as raycaster ``LightSource`` parameters via :mod:`level_lights`; never
    enters synced sim state. Values are render-local floats — no Q16.16
    snap (same class as ``light_rgb``; the sim-side migration note lives in
    engine/15 §2.2). Beacons freeze with the sim: their facing angle is a
    pure function of the sim tick (:func:`level_lights.beacon_angle`).
    """
    x: float                # tile coords (tile centers at .5)
    y: float
    color: tuple            # (r, g, b) 0-1 floats (toml carries 0-255 ints)
    intensity: float = 1.0
    range: float = 12.0     # tiles
    kind: str = "static"    # "static" | "beacon"
    period_s: float = 2.0   # beacon: seconds per full rotation
    beam_deg: float = 30.0  # beacon: cone width in degrees
    phase: float = 0.0      # beacon: fraction of a turn (0-1); a red/blue
                            # cop-car pair = two beacons, phase 0.0 / 0.5


def _parse_light_entry(entry, index: int, toml_path) -> LightEntry:
    """Validate one raw ``[[light]]`` table -> LightEntry (P4 design §2.1).

    Every error names the entry index and carries the required-fields hint;
    ``heat``/``jitter`` keys are rejected with the render-only rationale.
    """
    hint = ("Required fields: pos = [x, y] (tile floats), color = [r, g, b] "
            "(0-255 ints). Optional: intensity (> 0), range (tiles, > 0), "
            "kind ('static' | 'beacon'), period_s (> 0), beam_deg (0-360], "
            "phase (fraction of a turn).")

    def err(msg: str) -> ValueError:
        return ValueError(
            f"Invalid [[light]] entry #{index} in {toml_path}: {msg} {hint}")

    if not isinstance(entry, dict):
        raise err(f"expected a table, got {type(entry).__name__}.")
    for key in _LIGHT_FORBIDDEN_KEYS:
        if key in entry:
            raise err(
                f"'{key}' is not authorable: level lights are render-only "
                f"and never write the synced heat channel or enable C++ RNG "
                f"jitter (engine/14 synced-vs-local; P4 design). Remove the "
                f"'{key}' key.")

    pos = entry.get("pos")
    if (not isinstance(pos, (list, tuple)) or len(pos) != 2
            or not all(isinstance(v, (int, float))
                       and not isinstance(v, bool) for v in pos)):
        raise err(f"'pos' must be an [x, y] number pair, got {pos!r}.")

    col = entry.get("color")
    if (not isinstance(col, (list, tuple)) or len(col) != 3
            or not all(isinstance(v, int)
                       and not isinstance(v, bool) for v in col)):
        raise err(
            f"'color' must be an [r, g, b] triple of 0-255 ints, got {col!r}.")
    if not all(0 <= v <= 255 for v in col):
        raise err(f"'color' components must be within 0-255, got {col!r}.")

    kind = entry.get("kind", "static")
    if kind not in LIGHT_KINDS:
        raise err(f"'kind' must be one of {LIGHT_KINDS}, got {kind!r}.")

    def num(key: str, default: float, positive: bool = False) -> float:
        v = entry.get(key, default)
        if (isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(float(v))):
            raise err(f"'{key}' must be a finite number, got {v!r}.")
        v = float(v)
        if positive and v <= 0.0:
            raise err(f"'{key}' must be > 0, got {v!r}.")
        return v

    x, y = num_pos = (float(pos[0]), float(pos[1]))
    if not all(math.isfinite(v) for v in num_pos):
        raise err(f"'pos' must be finite, got {pos!r}.")
    beam_deg = num("beam_deg", 30.0, positive=True)
    if beam_deg > 360.0:
        raise err(f"'beam_deg' must be within (0, 360], got {beam_deg!r}.")

    return LightEntry(
        x=x, y=y,
        color=tuple(v / 255.0 for v in col),
        intensity=num("intensity", 1.0, positive=True),
        range=num("range", 12.0, positive=True),
        kind=str(kind),
        period_s=num("period_s", 2.0, positive=True),
        beam_deg=beam_deg,
        phase=num("phase", 0.0),
    )


def _parse_water_table(raw: dict, base: Path, toml_path,
                       tilemap: np.ndarray):
    """Parse the optional ``[water]`` table (engine/15 §2.3, P5).

    Returns the initial-depth grid as int32 Q16.16 metres — the file IS the
    field (P5 design §2.1): a ``.npy`` loaded via ``np.load`` (zero new
    deps, exact round-trip by identity, no runtime imaging dependency —
    the 8-bit PNG + max_depth_m carrier was dropped on record: auto-scaled
    re-quantization made edits non-local). Returns None when the level has
    no ``[water]`` key — water dormancy. Validation is hard (ValueError,
    path-bearing messages): shape must equal the tilemap's, dtype must be
    int32, depths must be non-negative.
    """
    if "water" not in raw:
        return None                   # no [water] key at all — dormancy
    water_tbl = raw["water"]
    if not isinstance(water_tbl, dict):
        raise ValueError(
            f"[water] in {toml_path} must be a table "
            f"(got {type(water_tbl).__name__}) — spell it [water], "
            f"not [[water]]")
    # A DECLARED [water] table is a statement of intent: missing/empty
    # depth_map is a hard error, never silently dry.
    depth_rel = water_tbl.get("depth_map")
    if not depth_rel:
        raise ValueError(
            f"[water] in {toml_path} missing required 'depth_map' field "
            f"(an .npy path, int32 Q16.16 metres, shape == tilemap)")
    depth_path = base / depth_rel
    if not depth_path.is_file():
        raise ValueError(
            f"[water] depth_map declared but file missing: {depth_path}")
    try:
        depth_q = np.load(depth_path, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise ValueError(
            f"[water] depth_map is not a readable .npy array: "
            f"{depth_path}: {e}")
    if depth_q.dtype != np.int32:
        raise ValueError(
            f"[water] depth_map must be dtype int32 (Q16.16 metres), "
            f"got {depth_q.dtype}: {depth_path}")
    if depth_q.shape != tilemap.shape:
        raise ValueError(
            f"[water] depth_map shape {depth_q.shape} != tilemap shape "
            f"{tilemap.shape}: {depth_path}")
    if int(depth_q.min()) < 0:
        raise ValueError(
            f"[water] depth_map contains negative depths "
            f"(min = {int(depth_q.min())} raw Q16.16): {depth_path}")
    return depth_q


@dataclass
class LevelData:
    name: str
    version: str
    path: Path                      # levels/<name>/
    tilemap: np.ndarray             # (rows, cols), int
    tile_size_m: float
    diffuse_path: Path              # required, exists
    normal_path: Optional[Path] = None
    emissive_mask_path: Optional[Path] = None
    emissive_bloom_path: Optional[Path] = None
    wall_mask_path: Optional[Path] = None
    background_path: Optional[Path] = None     # screen-fixed backdrop
    floor_id: int = 0
    spawns: list = field(default_factory=list)  # list[SpawnEntry]
    lights: list = field(default_factory=list)  # list[LightEntry] (P4 §2.2)
    raw_toml: dict = field(default_factory=dict)
    # ---- level format v2 [art] block (F2) --------------------------------
    # [art.bare] is the new spelling of diffuse_path/normal_path above (the
    # loader maps it onto them); specular is stored now, consumed when the
    # lighting learns specular (proposal §2.3).
    specular_path: Optional[Path] = None
    # Optional per-pixel floor HEIGHTMAP (greyscale relief, 0..1) for the
    # DISPLAYED layer — parsed exactly like normal/specular (same opt_art /
    # flat-fallback pattern). Consumed ONLY by the water pass to ATTENUATE the
    # water alpha so raised features (crates, consoles, debris) poke above the
    # surface and the water laps around them (graphics/water_rendering.md §2 §8).
    # It does NOT feed depth/volume/tint/refraction (those stay per-tile). Most
    # levels carry no height; this stays None and the water pass behaves exactly
    # as before (no attenuation).
    height_path: Optional[Path] = None
    # [art.furniture] / [art.destroyed] overlay layers — stored as paths in
    # F2, consumed by the per-tile layer compose in F3.
    furniture_diffuse_path: Optional[Path] = None
    furniture_normal_path: Optional[Path] = None
    furniture_specular_path: Optional[Path] = None
    destroyed_diffuse_path: Optional[Path] = None
    destroyed_normal_path: Optional[Path] = None
    destroyed_specular_path: Optional[Path] = None
    # [art.align] — non-destructive grid alignment (proposal §1.3): art pixel
    # `art_offset_px` lands on grid (0, 0); `art_px_per_tile` art pixels span
    # one tile, normalized to an (x, y) PAIR (the toml may say a scalar =
    # same scale both axes, or [x, y] — the v2 art's proportions differ from
    # the tilemap per axis). When level.toml carries no [art.align], the
    # defaults are offset (0, 0) + px_per_tile = (art_w / grid_w,
    # art_h / grid_h) (None if the art dimensions could not be read) and
    # `art_align_explicit` stays False — the renderer then keeps the legacy
    # stretch-art-to-grid-rect draw, which is exactly the same per-axis
    # transform and bit-identical to the pre-F2 output.
    art_offset_px: tuple = (0.0, 0.0)
    art_px_per_tile: Optional[tuple] = None
    art_align_explicit: bool = False
    # [water] initial state (engine/15 §2.3, P5): the starting water field,
    # int32 Q16.16 metres, shape == tilemap — or None when the level carries
    # no [water] key (water dormancy: a dry level is bit-identical to before
    # the key existed). GameMap.__init__ seeds gmap.water_depth from it,
    # masked to (~solid) & (~is_vacuum) (the solver zeroes depth on solid —
    # a mass sink — so seeding those tiles would silently destroy water).
    # Lives in the defaulted tail: synthetic LevelData(...) in tests keeps
    # constructing unchanged.
    water_depth_q: Optional[np.ndarray] = None

    @property
    def height(self) -> int:
        return int(self.tilemap.shape[0])

    @property
    def width(self) -> int:
        return int(self.tilemap.shape[1])


def load(level_name: str, levels_dir: str = "levels") -> LevelData:
    """Load a level by folder name. Raises ValueError on validation failure."""
    here = Path(__file__).resolve().parent
    base = (here / levels_dir / level_name).resolve()
    if not base.is_dir():
        raise ValueError(f"Level folder does not exist: {base}")

    toml_path = base / "level.toml"
    if not toml_path.is_file():
        raise ValueError(f"level.toml not found in {base}")

    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    version = str(raw.get("version", "")).strip()
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported level.toml version '{version}'. "
            f"Supported: {sorted(SUPPORTED_VERSIONS)}"
        )

    name = raw.get("name", level_name)

    tilemap_rel = raw.get("tilemap")
    if not tilemap_rel:
        raise ValueError("level.toml missing required 'tilemap' field")
    tilemap_path = base / tilemap_rel
    if not tilemap_path.is_file():
        raise ValueError(f"Tilemap CSV not found: {tilemap_path}")
    tilemap = np.loadtxt(tilemap_path, delimiter=",", dtype=np.int32)
    if tilemap.ndim != 2:
        raise ValueError(f"Tilemap must be 2D, got shape {tilemap.shape}")

    tile_size_m = float(raw.get("tile_size_m", 0.333))

    # ---- [art] block (level format v2 §1.2) ------------------------------
    # Parsed format-driven, not version-gated: the `version` gate governs the
    # CSV *vocabulary* (see materials_from_tilemap); art keys are recognised
    # by spelling. v1 levels carry no [art] table and take the flat-key paths
    # below completely unchanged.
    art_tbl = raw.get("art", {})
    if not isinstance(art_tbl, dict):
        raise ValueError("level.toml [art] must be a table")

    def art_sub(name: str) -> dict:
        sub = art_tbl.get(name, {})
        if not isinstance(sub, dict):
            raise ValueError(f"level.toml [art.{name}] must be a table")
        return sub

    bare_tbl = art_sub("bare")
    furniture_tbl = art_sub("furniture")
    destroyed_tbl = art_sub("destroyed")
    align_tbl = art_sub("align")

    def resolve(rel, label: str) -> Path:
        p = base / rel
        if not p.is_file():
            raise ValueError(f"{label} declared but file missing: {p}")
        return p

    def opt(key: str) -> Optional[Path]:
        """Optional flat (v1-spelling) key — behaviour unchanged from v1."""
        rel = raw.get(key)
        if not rel:
            return None
        return resolve(rel, key)

    def opt_art(table: dict, label_prefix: str, key: str) -> Optional[Path]:
        """Optional [art...] key; errors name the full toml location."""
        rel = table.get(key)
        if not rel:
            return None
        return resolve(rel, f"{label_prefix} {key}")

    # Bare diffuse — required. [art.bare] diffuse is the new spelling; the
    # old flat `diffuse` keeps working and acts as the fallback when both
    # spellings appear.
    diffuse_rel = bare_tbl.get("diffuse") or raw.get("diffuse")
    if not diffuse_rel:
        if art_tbl:
            raise ValueError(
                "level.toml missing required [art.bare] 'diffuse' field "
                "(or the legacy flat 'diffuse' key)")
        raise ValueError("level.toml missing required 'diffuse' field")
    diffuse_path = base / diffuse_rel
    if not diffuse_path.is_file():
        raise ValueError(f"Diffuse texture not found: {diffuse_path}")

    normal_path = opt_art(bare_tbl, "[art.bare]", "normal") or opt("normal")
    specular_path = opt_art(bare_tbl, "[art.bare]", "specular")
    # Optional displayed-layer heightmap (greyscale relief). Mirrors the
    # normal/specular spelling: [art.bare] height (or the flat `height` key).
    # Optional everywhere — a level with no height loads fine (field None).
    height_path = opt_art(bare_tbl, "[art.bare]", "height") or opt("height")
    emissive_mask_path = (opt_art(art_tbl, "[art]", "emissive_mask")
                          or opt("emissive_mask"))
    background_path = (opt_art(art_tbl, "[art]", "background")
                       or opt("background"))

    # Overlay layers — stored in F2, consumed by the layer compose in F3.
    furniture_diffuse_path = opt_art(furniture_tbl, "[art.furniture]", "diffuse")
    furniture_normal_path = opt_art(furniture_tbl, "[art.furniture]", "normal")
    furniture_specular_path = opt_art(furniture_tbl, "[art.furniture]", "specular")
    destroyed_diffuse_path = opt_art(destroyed_tbl, "[art.destroyed]", "diffuse")
    destroyed_normal_path = opt_art(destroyed_tbl, "[art.destroyed]", "normal")
    destroyed_specular_path = opt_art(destroyed_tbl, "[art.destroyed]", "specular")

    # [art.align] — non-destructive grid alignment (§1.3).
    art_align_explicit = "align" in art_tbl
    offset_raw = align_tbl.get("offset_px", (0, 0))
    if (not isinstance(offset_raw, (list, tuple)) or len(offset_raw) != 2
            or not all(isinstance(v, (int, float)) for v in offset_raw)):
        raise ValueError(
            f"[art.align] offset_px must be an [x, y] number pair, "
            f"got {offset_raw!r}")
    art_offset_px = (float(offset_raw[0]), float(offset_raw[1]))
    ppt_raw = align_tbl.get("px_per_tile")
    if ppt_raw is not None:
        # Scalar = same scale both axes; [x, y] = per-axis. Either spelling
        # normalizes to a pair on LevelData (the renderer always consumes the
        # pair).
        if isinstance(ppt_raw, (int, float)):
            ppt_pair = (float(ppt_raw), float(ppt_raw))
        elif (isinstance(ppt_raw, (list, tuple)) and len(ppt_raw) == 2
                and all(isinstance(v, (int, float)) for v in ppt_raw)):
            ppt_pair = (float(ppt_raw[0]), float(ppt_raw[1]))
        else:
            raise ValueError(
                f"[art.align] px_per_tile must be a positive number or an "
                f"[x, y] pair of positive numbers, got {ppt_raw!r}")
        if ppt_pair[0] <= 0.0 or ppt_pair[1] <= 0.0:
            raise ValueError(
                f"[art.align] px_per_tile must be positive, got {ppt_raw!r}")
        art_px_per_tile: Optional[tuple] = ppt_pair
    else:
        # Default px_per_tile = (art_width / grid_width, art_height /
        # grid_height) (§1.2) — exactly the implicit per-axis transform of
        # the legacy stretch-art-to-grid draw.
        try:
            art_w, art_h = _png_size(diffuse_path)
            art_px_per_tile = (art_w / float(tilemap.shape[1]),
                               art_h / float(tilemap.shape[0]))
        except (OSError, ValueError):
            if art_align_explicit:
                raise ValueError(
                    f"[art.align] declared without px_per_tile and the "
                    f"diffuse dimensions could not be read: {diffuse_path}")
            art_px_per_tile = None

    spawns = []
    for i, entry in enumerate(raw.get("spawn", [])):
        try:
            spawns.append(SpawnEntry(
                name=str(entry["name"]),
                team=int(entry["team"]),
                x=float(entry["x"]),
                y=float(entry["y"]),
                footprint=int(entry.get("footprint", 3)),
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid [[spawn]] entry #{i} in {toml_path}: {e}. "
                f"Required fields: name (str), team (int), x (float), y (float)."
            )

    # ---- [[light]] entities (engine/15 §2.2, P4) -------------------------
    lights_raw = raw.get("light", [])
    if not isinstance(lights_raw, list):
        raise ValueError(
            f"[[light]] in {toml_path} must be an array of tables "
            f"(got {type(lights_raw).__name__}) — spell it [[light]], "
            f"not [light]")
    lights = [_parse_light_entry(entry, i, toml_path)
              for i, entry in enumerate(lights_raw)]

    # ---- [water] initial state (engine/15 §2.3, P5) ----------------------
    water_depth_q = _parse_water_table(raw, base, toml_path, tilemap)

    return LevelData(
        name=name,
        version=version,
        path=base,
        tilemap=tilemap,
        tile_size_m=tile_size_m,
        diffuse_path=diffuse_path,
        normal_path=normal_path,
        emissive_mask_path=emissive_mask_path,
        emissive_bloom_path=opt("emissive_bloom"),
        wall_mask_path=opt("wall_mask"),
        background_path=background_path,
        floor_id=int(raw.get("floor_id", 0)),
        spawns=spawns,
        lights=lights,
        raw_toml=raw,
        specular_path=specular_path,
        height_path=height_path,
        furniture_diffuse_path=furniture_diffuse_path,
        furniture_normal_path=furniture_normal_path,
        furniture_specular_path=furniture_specular_path,
        destroyed_diffuse_path=destroyed_diffuse_path,
        destroyed_normal_path=destroyed_normal_path,
        destroyed_specular_path=destroyed_specular_path,
        art_offset_px=art_offset_px,
        art_px_per_tile=art_px_per_tile,
        art_align_explicit=art_align_explicit,
        water_depth_q=water_depth_q,
    )


def materials_from_tilemap(tilemap: np.ndarray, version: str):
    """Translate CSV tile values into material grid + vacuum mask.

    Returns (material, is_vacuum) — both (H, W) numpy arrays. ``version`` is
    the level.toml format version (``LevelData.version``); the CSV *vocabulary*
    depends on it, so callers must thread it through.

    **version "2" (canon, level format v2 §1.1): codes ARE material ids** from
    :mod:`simulation.materials` — read literally, no translation — plus exactly
    one reserved non-material code, :data:`SPACE_CODE` (9) = outer space
    (MAT_AIR + ``is_vacuum``). Any other value is a hard ValueError: a v2 CSV
    never carries silent garbage.

    **version "1" (legacy generator vocabulary)** — kept bit-exact for old
    levels; codes are distinct from material ids:

      0     -> outer space (vacuum, MAT_AIR)
      1     -> hull wall    (MAT_HULL)
      2     -> wood wall    (MAT_WOOD) — flammable
      3     -> door         (MAT_DOOR) — currently behaves as a wall
               for occlusion; movement still allowed through it;
               full door system deferred
      4..8  -> interior air (MAT_AIR), no vacuum

    ``tools/migrate_tilemap_v2.py`` converts a v1 level in place (0->9, 1->1,
    3->3, 2/4..8->0 — code 2 was the generator's *floor*, retired as the
    "generator floor vs MAT_WOOD wall" landmine).
    """
    from simulation.materials import (
        MAT_AIR, MAT_HULL, MAT_WOOD, MAT_DOOR, MATERIAL_NAMES,
    )
    version = str(version)
    if version == "1":
        material = np.full(tilemap.shape, MAT_AIR, dtype=np.int8)
        material[tilemap == 1] = MAT_HULL
        material[tilemap == 2] = MAT_WOOD
        material[tilemap == 3] = MAT_DOOR
        is_vacuum = (tilemap == 0)
        return material, is_vacuum
    if version == "2":
        allowed = set(MATERIAL_NAMES) | {SPACE_CODE}
        bad = sorted(int(c) for c in np.unique(tilemap) if int(c) not in allowed)
        if bad:
            raise ValueError(
                f"v2 tilemap contains unknown codes {bad}; valid codes are "
                f"material ids {sorted(MATERIAL_NAMES)} + {SPACE_CODE} (SPACE)"
            )
        is_vacuum = (tilemap == SPACE_CODE)
        material = tilemap.astype(np.int8, copy=True)
        material[is_vacuum] = MAT_AIR
        return material, is_vacuum
    raise ValueError(
        f"materials_from_tilemap: unsupported tilemap version {version!r}; "
        f"supported: {sorted(SUPPORTED_VERSIONS)}"
    )


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "unhcr_vessel"
    lvl = load(name)
    print(f"Loaded: {lvl.name} (v{lvl.version})")
    print(f"  Grid: {lvl.width} x {lvl.height} tiles @ {lvl.tile_size_m} m each")
    print(f"  Diffuse: {lvl.diffuse_path.name}")
    print(f"  Normal:  {lvl.normal_path.name if lvl.normal_path else '(none)'}")
    print(f"  Height:  {lvl.height_path.name if lvl.height_path else '(none)'}")
    print(f"  Emissive: {lvl.emissive_mask_path.name if lvl.emissive_mask_path else '(none)'}")
    print(f"  Bloom:   {lvl.emissive_bloom_path.name if lvl.emissive_bloom_path else '(none)'}")
    print(f"  Wall mask: {lvl.wall_mask_path.name if lvl.wall_mask_path else '(derived from CSV)'}")
    print(f"  Align:   offset={lvl.art_offset_px} px_per_tile={lvl.art_px_per_tile}"
          f" ({'explicit' if lvl.art_align_explicit else 'default'})")
    print(f"  Layers:  furniture={'yes' if lvl.furniture_diffuse_path else 'no'}"
          f" destroyed={'yes' if lvl.destroyed_diffuse_path else 'no'}")
    print(f"  Floor:   {lvl.floor_id}")
    print(f"  Lights:  {sum(1 for l in lvl.lights if l.kind == 'static')} static"
          f" + {sum(1 for l in lvl.lights if l.kind == 'beacon')} beacon")
    print(f"  Water:   "
          f"{int((lvl.water_depth_q > 0).sum()) if lvl.water_depth_q is not None else 0}"
          f" wet tiles{'' if lvl.water_depth_q is not None else ' (no [water] key)'}")
    print(f"  Tile values: {sorted(np.unique(lvl.tilemap).tolist())}")
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    print(f"  Materials: hull={int((mat==1).sum())} door={int((mat==3).sum())} air={int((mat==0).sum())} vacuum={int(vac.sum())}")
