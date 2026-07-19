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
import re
import sys
import tomllib
import warnings
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

# [[entity]] validation (A3): the registry IS the validator (entity design
# §3b). The entities package is import-light (stdlib-only, CI-tested), so
# this pulls in no compiled physics and never simulation.simulation.
from simulation.entities import (  # noqa: E402
    KIND_ENTITY_REF, REGISTRY as ENTITY_REGISTRY, effective_defaults,
    field_value_error,
)


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


_ENTITY_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-]*\Z")

# Instance-level keys every [[entity]] table carries besides its schema
# fields: the id/tags instance facts (schema.INSTANCE_FIELDS) + the class
# binding itself.
_ENTITY_META_KEYS = ("id", "class", "tags")


@dataclass
class EntityInstance:
    """One ``[[entity]]`` instance declared in level.toml (design §3a, A3).

    Parsed data ONLY in Arc A — nothing steps entities yet (the dormancy
    guarantee; the runtime slot is Arc B's, digests are A4's). ``fields``
    holds the EFFECTIVE values: authored keys over the registry defaults
    (entities.toml overlay applied), stored as authored — length_m fields
    are NOT quantized here; the canonical tile rule is the consumer's load
    step (editor design §4) and no consumer needs tiles yet.
    ``authored_keys`` records which schema fields the file actually spelled
    out (file order), so level_lib's writer round-trips byte-stably without
    ever materializing defaults into the file.
    """
    id: str
    class_name: str     # the toml `class` key ("class" is a Python keyword)
    ordinal: int        # runtime ordinal id — FILE ORDER at load (§3a)
    tags: tuple = ()
    fields: dict = field(default_factory=dict)
    authored_keys: tuple = ()


def _light_entry_from_entity(inst: EntityInstance) -> LightEntry:
    """``[[entity]]`` light instance -> the SAME LightEntry a ``[[light]]``
    block yields (the legacy-alias equivalence contract, editor design §6):
    one downstream render path either way, asserted field-for-field in
    tests/test_entity_format.py."""
    f = inst.fields
    return LightEntry(
        x=float(f["x"]), y=float(f["y"]),
        color=tuple(v / 255.0 for v in f["color"]),
        intensity=float(f["intensity"]),
        range=float(f["range"]),
        kind=str(f["kind"]),
        period_s=float(f["period_s"]),
        beam_deg=float(f["beam_deg"]),
        phase=float(f["phase"]),
    )


def _parse_entities(raw: dict, toml_path, spawns: list) -> list:
    """Parse + validate the ``[[entity]]`` array (design §3a/§3b/§3e, A3).

    The registry IS the validator (§3b): unknown class, unknown field, kind
    or bounds mismatch, missing required field, duplicate or malformed id —
    all hard ValueErrors naming the entry. Ids are assigned runtime ordinals
    in FILE ORDER (§3a). Refs (KIND_ENTITY_REF fields) are checked across
    the whole file AFTER parsing: a ref naming a missing id WARNS (an
    authoring error, not fatal — a destroyed entity at runtime is not an
    error either); a ref naming a ``[[spawn]]`` unit HARD-ERRORS (units are
    not entities until the stack-2 convergence, §3e).
    """
    entities_raw = raw.get("entity", [])
    if not isinstance(entities_raw, list):
        raise ValueError(
            f"[[entity]] in {toml_path} must be an array of tables "
            f"(got {type(entities_raw).__name__}) — spell it [[entity]], "
            f"not [entity]")

    entities: list = []
    seen: dict = {}                    # id -> entry index (duplicate check)
    for i, entry in enumerate(entities_raw):

        def err(msg: str) -> ValueError:
            return ValueError(
                f"Invalid [[entity]] entry #{i} in {toml_path}: {msg}")

        if not isinstance(entry, dict):
            raise err(f"expected a table, got {type(entry).__name__}.")

        eid = entry.get("id")
        if not (isinstance(eid, str) and _ENTITY_ID_RE.fullmatch(eid)):
            raise err(
                f"'id' must be a slug (letters/digits/_/-, e.g. 'door_3'), "
                f"got {eid!r}. Every instance carries a mandatory unique "
                f"id (entity design §3a).")
        if eid in seen:
            raise err(
                f"duplicate id '{eid}' (first declared by entry "
                f"#{seen[eid]}) — ids are mandatory and UNIQUE; all "
                f"references address ids (entity design §3a).")

        cls_name = entry.get("class")
        if not isinstance(cls_name, str) or cls_name not in ENTITY_REGISTRY:
            raise err(
                f"unknown entity class {cls_name!r} — the registry is the "
                f"validator (entity design §3b); registered classes: "
                f"{sorted(ENTITY_REGISTRY)}.")

        tags = entry.get("tags", [])
        if not (isinstance(tags, list)
                and all(isinstance(t, str) for t in tags)):
            raise err(f"'tags' must be an array of strings (entity design "
                      f"§3c), got {tags!r}.")

        schema_fields = {f.name: f for f in ENTITY_REGISTRY[cls_name].FIELDS}
        values = effective_defaults(cls_name)
        authored = tuple(k for k in entry if k not in _ENTITY_META_KEYS)
        for key in authored:
            if key not in schema_fields:
                raise err(
                    f"unknown field '{key}' for class '{cls_name}' — the "
                    f"registry schema is the validator (entity design "
                    f"§3b); declared fields: {sorted(schema_fields)}.")
            v = entry[key]
            verr = field_value_error(schema_fields[key], v)
            if verr:
                raise err(f"'{key}' = {v!r} {verr}.")
            values[key] = v     # stored as authored (length_m NOT quantized)
        missing = sorted(k for k, v in values.items() if v is None)
        if missing:
            raise err(f"missing required field(s) {missing} for class "
                      f"'{cls_name}' (no default exists).")

        seen[eid] = i
        entities.append(EntityInstance(
            id=eid, class_name=cls_name, ordinal=i, tags=tuple(tags),
            fields=values, authored_keys=authored))

    # Refs are checked across the WHOLE file, once every id is known.
    ids = set(seen)
    unit_names = {str(s.name) for s in spawns}
    for inst in entities:
        for f in ENTITY_REGISTRY[inst.class_name].FIELDS:
            if f.kind != KIND_ENTITY_REF:
                continue
            target = inst.fields.get(f.name)
            if not target:
                continue               # empty string = unwired ref
            if target in unit_names:
                raise ValueError(
                    f"[[entity]] '{inst.id}' field '{f.name}' in "
                    f"{toml_path} references '{target}', a [[spawn]] unit "
                    f"— units are NOT entities (entity design §3e): no "
                    f"wire, tag or ref may address a unit until the "
                    f"stack-2 convergence.")
            if target not in ids:
                warnings.warn(
                    f"[[entity]] '{inst.id}' field '{f.name}' in "
                    f"{toml_path} is a dangling ref: no instance has id "
                    f"'{target}' (authoring error, not fatal — entity "
                    f"design §3a).")
    return entities


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


# ---- zones (editor design §5, A8) ----------------------------------------
# The painted zone mask: uint8 paint ids, 0 = unpainted, discovered by
# PRESENCE next to level.toml (no toml table exists for zones — the file IS
# the mask; contrast [water], whose depth_map key predates the pattern).
ZONES_FILENAME = "zones.npy"

# The two zone classes, ever (editor design §5). One paint-id namespace
# across both: a breach_site and an extraction_zone may not share a zone_id.
ZONE_CLASSES = ("breach_site", "extraction_zone")


def _parse_zones_grid(base: Path, tilemap: np.ndarray):
    """Load the optional ``zones.npy`` paint grid (editor design §5, A8).

    Returns the uint8 paint-id grid, or None when the file is absent — zone
    dormancy (an un-zoned level is bit-identical to before zones existed).
    Validation is hard (ValueError, path-bearing messages): dtype must be
    uint8 (paint ids are 1..255, 0 = unpainted — the id namespace is pinned
    by the schema's zone_id bounds), shape must equal the tilemap's.
    """
    zones_path = base / ZONES_FILENAME
    if not zones_path.is_file():
        return None                   # no zones.npy at all — dormancy
    try:
        zone_grid = np.load(zones_path, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise ValueError(
            f"zones.npy is not a readable .npy array: {zones_path}: {e}")
    if zone_grid.dtype != np.uint8:
        raise ValueError(
            f"zones.npy must be dtype uint8 (paint ids 1..255, "
            f"0 = unpainted), got {zone_grid.dtype}: {zones_path}")
    if zone_grid.shape != tilemap.shape:
        raise ValueError(
            f"zones.npy shape {zone_grid.shape} != tilemap shape "
            f"{tilemap.shape}: {zones_path}")
    return zone_grid


# ---- air_init.npy + boundary (entity design §10, A9) ----------------------
# The optional atmosphere-override grid: int32 Q16.16 atm (1.0 atm == 65536
# counts, the S2c scale), discovered by PRESENCE next to level.toml exactly
# like zones.npy (the file IS the field; no toml key — [water]'s depth_map
# key predates the pattern). Editor wand PAINTING of this file is Arc C's
# (editor design §7); A9 is the format + load side only.
AIR_INIT_FILENAME = "air_init.npy"

# The `boundary` top-level level.toml field — v1 value set. "space" is
# today's behavior (the vacuum ring) and the default when the key is absent;
# "ambient" is parsed, validated and stored on LevelData but changes NO
# behavior in Arc A: the AMBIENT border-ring mode (MG pins P=P_amb, species
# reservoir) belongs to the boundary-conditions physics project (priority
# ledger #1). This field is that project's format hook.
BOUNDARY_SPACE = "space"
BOUNDARY_AMBIENT = "ambient"
BOUNDARY_MODES = (BOUNDARY_SPACE, BOUNDARY_AMBIENT)


def _parse_air_init_grid(base: Path, tilemap: np.ndarray):
    """Load the optional ``air_init.npy`` atmosphere override (A9).

    Returns the int32 Q16.16 atmosphere grid (1.0 atm == 65536 counts,
    shape == tilemap), or None when the file is absent — air dormancy: the
    engine then derives today's ambient seeding (FP_ONE in open air, 0 on
    solid/SPACE) exactly as before the feature existed. NOTE the dormancy
    boundary is file ABSENCE, not grid content: an all-zero grid is a real
    authored state (a fully depressurized start) and an all-ambient grid is
    a real pinned state — neither means "derive as today".

    Validation is hard (ValueError, path-bearing messages): dtype must be
    int32 (Q16.16 atm), shape must equal the tilemap's, values must be
    non-negative. The solid/SPACE-tile rule (values there are IGNORED) is
    the seed consumer's — see GameMap.__init__.
    """
    air_path = base / AIR_INIT_FILENAME
    if not air_path.is_file():
        return None                   # no air_init.npy at all — dormancy
    try:
        air_q = np.load(air_path, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise ValueError(
            f"air_init.npy is not a readable .npy array: {air_path}: {e}")
    if air_q.dtype != np.int32:
        raise ValueError(
            f"air_init.npy must be dtype int32 (Q16.16 atm, 1.0 atm == "
            f"65536 counts), got {air_q.dtype}: {air_path}")
    if air_q.shape != tilemap.shape:
        raise ValueError(
            f"air_init.npy shape {air_q.shape} != tilemap shape "
            f"{tilemap.shape}: {air_path}")
    if int(air_q.min(initial=0)) < 0:
        raise ValueError(
            f"air_init.npy contains negative pressures "
            f"(min = {int(air_q.min())} raw Q16.16): {air_path}")
    return air_q


def _parse_boundary(raw: dict, toml_path) -> str:
    """Parse the top-level ``boundary`` field (A9 format hook).

    Absent key == "space" (today's behavior). Any value outside
    :data:`BOUNDARY_MODES` is a hard error naming the two options. NO
    behavior differs between the two values in Arc A (see the
    BOUNDARY_MODES comment)."""
    boundary = raw.get("boundary", BOUNDARY_SPACE)
    if boundary not in BOUNDARY_MODES:
        raise ValueError(
            f"Unknown 'boundary' value {boundary!r} in {toml_path}: the "
            f"two options are \"{BOUNDARY_SPACE}\" (today's vacuum ring, "
            f"the default when absent) and \"{BOUNDARY_AMBIENT}\" "
            f"(planetside — parsed and stored now; ring behavior lands "
            f"with the boundary-conditions physics project, ledger #1).")
    return str(boundary)


def _validate_zone_binding(zone_grid, entities, toml_path) -> None:
    """The zone binding validators (editor design §5, A8).

    The npy grid holds paint ids; the ``[[entity]]`` instance holds
    everything else. Every painted id must have exactly ONE zone instance:
    a duplicate ``zone_id`` claim is a HARD load error; an orphaned paint id
    (painted, no instance) and a zero-tile instance (claimed, never painted
    — including when zones.npy is absent entirely) are ``warnings.warn``
    authoring warnings, not crashes (the A3 dangling-ref precedent).
    """
    claims: dict = {}                 # zone_id -> instance (file order)
    for inst in entities:
        if inst.class_name not in ZONE_CLASSES:
            continue
        zid = int(inst.fields["zone_id"])
        if zid in claims:
            raise ValueError(
                f"[[entity]] '{inst.id}' in {toml_path} claims zone_id "
                f"{zid}, already claimed by '{claims[zid].id}' — every "
                f"painted id has exactly ONE zone instance (level editor "
                f"v3 design §5); duplicate zone_id is a load error. The "
                f"paint-id namespace is one space across both zone classes.")
        claims[zid] = inst
    painted = (set(int(v) for v in np.unique(zone_grid)) - {0}
               if zone_grid is not None else set())
    for zid in sorted(painted - set(claims)):
        warnings.warn(
            f"zones.npy beside {toml_path} paints zone id {zid} but no "
            f"zone instance claims it — orphaned paint (a validator "
            f"warning, not a crash: level editor v3 design §5).")
    for zid, inst in claims.items():
        if zid not in painted:
            warnings.warn(
                f"[[entity]] '{inst.id}' in {toml_path} claims zone_id "
                f"{zid} but zones.npy paints no tile with it (0 painted "
                f"tiles — authoring warning, level editor v3 design §5).")


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
    # ---- [[entity]] instances (entity design §3a, A3) --------------------
    # Parsed data ONLY in Arc A: nothing steps them (the dormancy
    # guarantee; digests are A4's). File order IS runtime ordinal order —
    # ids are assigned in file order at load and every runtime sweep
    # iterates in id order (§3a). Entity light instances ALSO land in
    # `lights` above as equivalent LightEntry values (the [[light]]
    # legacy-alias contract; mixed forms hard-error at load).
    entities: list = field(default_factory=list)  # list[EntityInstance]
    # ---- zones.npy paint grid (editor design §5, A8) ---------------------
    # uint8 paint ids, shape == tilemap, 0 = unpainted — or None when the
    # level folder carries no zones.npy (zone dormancy: an un-zoned level
    # is bit-identical to before zones existed). Discovered by file
    # PRESENCE, never a toml key. Binding: each nonzero painted id belongs
    # to exactly one zone [[entity]] instance via its zone_id field
    # (validators above). Parsed data only in Arc A — spawn realization
    # from breach-site rosters is stack-2's.
    zone_grid: Optional[np.ndarray] = None
    # ---- air_init.npy atmosphere override (entity design §10, A9) --------
    # int32 Q16.16 atm (1.0 atm == 65536 counts), shape == tilemap — or
    # None when the level folder carries no air_init.npy (air dormancy:
    # the engine derives today's ambient seeding exactly). Discovered by
    # file PRESENCE, never a toml key (the zones.npy pattern).
    # GameMap.__init__ seeds atmosphere + the O2/N2 species from it on
    # open-air tiles; values on solid or SPACE tiles are IGNORED (the
    # pinned solid-tile rule — see the seed block's docstring/comment).
    air_init_q: Optional[np.ndarray] = None
    # ---- boundary mode (A9 format hook; ledger #1 owns semantics) --------
    # "space" (default — today's vacuum ring) | "ambient" (planetside).
    # Parsed + validated + stored ONLY in Arc A: the AMBIENT border ring
    # is the boundary-conditions physics project's. A top-level scalar
    # key, so level_lib's managed-block writer round-trips it untouched.
    boundary: str = BOUNDARY_SPACE
    # ---- --res base-resolution recovery (A6, S1 — a6 doors design §3) ----
    # `_upscale_level` (main.py) divides tile_size_m by the factor BEFORE
    # GameMap ever sees the level, which would leave meters-first entity
    # consumers (door span quantization) a non-integral tiles-per-meter.
    # It therefore records the accumulated integer replication factor and
    # the PRE-scale tile size here, BEFORE mutating. Runtime fields only —
    # never authored in level.toml, never written back. tile_size_m_base
    # None == "unscaled: use tile_size_m". Entity consumers quantize at
    # base resolution and replicate their tile sets by res_factor.
    res_factor: int = 1
    tile_size_m_base: Optional[float] = None

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

    # ---- [[entity]] instances (entity design §3a/§3b/§3e, A3) ------------
    entities = _parse_entities(raw, toml_path, spawns)
    entity_lights = [e for e in entities if e.class_name == "light"]
    if lights_raw and entity_lights:
        raise ValueError(
            f"{toml_path} mixes legacy [[light]] blocks with [[entity]] "
            f"light instances — a level carries ONE form, never both "
            f"(level editor v3 design §6). Legacy [[light]] stays valid "
            f"as-is; migration is explicit, never a save side effect: "
            f"convert the whole level at once with the one-shot migration "
            f"tool (tools/migrate_level_entities.py, Arc A patch A7).")
    # The alias contract: entity lights feed the SAME downstream render
    # path as [[light]] blocks — one LightEntry list either way.
    lights += [_light_entry_from_entity(e) for e in entity_lights]

    # ---- [water] initial state (engine/15 §2.3, P5) ----------------------
    water_depth_q = _parse_water_table(raw, base, toml_path, tilemap)

    # ---- zones.npy paint grid + binding (editor design §5, A8) -----------
    zone_grid = _parse_zones_grid(base, tilemap)
    _validate_zone_binding(zone_grid, entities, toml_path)

    # ---- air_init.npy + boundary (entity design §10, A9) -----------------
    air_init_q = _parse_air_init_grid(base, tilemap)
    boundary = _parse_boundary(raw, toml_path)

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
        entities=entities,
        zone_grid=zone_grid,
        air_init_q=air_init_q,
        boundary=boundary,
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
    print(f"  Air:     "
          f"{'override grid' if lvl.air_init_q is not None else 'ambient (no air_init.npy)'}"
          f"  boundary={lvl.boundary}")
    print(f"  Tile values: {sorted(np.unique(lvl.tilemap).tolist())}")
    mat, vac = materials_from_tilemap(lvl.tilemap, lvl.version)
    print(f"  Materials: hull={int((mat==1).sum())} door={int((mat==3).sum())} air={int((mat==0).sum())} vacuum={int(vac.sum())}")
