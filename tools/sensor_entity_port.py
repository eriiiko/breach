r"""tools/sensor_entity_port.py — sensor placement bridge (Arc C3).

Sensor placement (editor doc §6, canon engine/16 §8) needs the ACTUAL
sample-family split (AIR vs BODY, D7/D8) and `resolve_sample_tile` behavior
declared on :mod:`simulation.entities.sensors` — the registry's exported
JSON payload (`entity_registry.json` / `registry_payload()`) carries schema
SHAPE only (fields/kinds/defaults), not the `SENSOR`/`SAMPLE_FAMILY` class
flags or the `resolve_sample_tile` classmethod, and extending that export
shape means editing `src/simulation/entities/registry.py` — out of Arc C3's
allowed surface (read-only there). So this module imports
`simulation.entities.sensors` directly (read-only import, entity design §3b
import-light — stdlib only) and delegates to its OWN family/offset logic
rather than reimplementing it from the JSON shape.

Degradation note: if the live registry import is broken (C1's fallback
path), `simulation.entities.sensors` may still import fine on its own (it is
a leaf, import-light module) UNLESS the break is inside sensors.py itself —
in that case sensor placement simply is not offered; the class still shows
in the palette (from the JSON fallback) and falls through to the generic
place-one path (no arrow, no solid refusal) until the import is fixed. That
mirrors C1's "the palette still renders, minus nothing it can prove" stance.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from simulation.entities import sensors as sensor_entities  # noqa: E402
from level_loader import EntityInstance  # noqa: E402

# The six field-sensor classes — `resolve_sample_tile` exists on ALL of
# them (inherited from `_FieldSensor`); `clock`/`sensor_motion` do NOT (no
# sampled field tile, so no arrow / solid-refusal concept applies to them —
# they place via the generic place-one path instead).
FIELD_SENSOR_CLASSES = ("pressure", "smoke", "water_depth", "o2",
                        "temperature", "fire")
SAMPLE_AIR = sensor_entities.SAMPLE_AIR
SAMPLE_BODY = sensor_entities.SAMPLE_BODY


def is_field_sensor(cls_name: str) -> bool:
    return cls_name in FIELD_SENSOR_CLASSES


def sample_family(cls_name: str) -> Optional[str]:
    """`SAMPLE_AIR` / `SAMPLE_BODY`, or `None` when `cls_name` is not a
    field sensor class this module's live import can see."""
    cls = getattr(sensor_entities, cls_name, None)
    if cls is None:
        return None
    return getattr(cls, "SAMPLE_FAMILY", None)


def resolve_sample_tile(cls_name: str, fields: dict):
    """`(fy, fx)` the sensor samples — delegates to the class's OWN
    `resolve_sample_tile` classmethod (never re-derive the AIR-vs-BODY
    split here). `None` when the class can't be resolved (import broken /
    not a field sensor)."""
    cls = getattr(sensor_entities, cls_name, None)
    if cls is None or not hasattr(cls, "resolve_sample_tile"):
        return None
    return cls.resolve_sample_tile(fields)


def tile_is_open(grid, tile, solid_codes, space_code) -> bool:
    """True iff `tile = (fy, fx)` is in-bounds AND neither solid-for-flow
    (permeability <= 0, the frozen D6 "solid" channel — the SAME predicate
    `map_editor.water_solid_codes()` already computes, sim-exact) nor SPACE
    (a sensor cannot sample vacuum any more than a solid — no atmosphere
    field exists there). `tile=None` (an unresolvable class) reads as not
    open, so a placement always refuses rather than guessing."""
    if tile is None:
        return False
    fy, fx = tile
    h, w = grid.shape
    if not (0 <= fx < w and 0 <= fy < h):
        return False
    v = int(grid[fy, fx])
    return v != space_code and v not in solid_codes


def build_sensor_instance(cls_name: str, x: int, y: int, dx: int, dy: int,
                          id_: str) -> EntityInstance:
    """The `[[entity]]` instance a placed field sensor writes: `x`/`y`
    always authored (REQUIRED, no default); `sample_dx`/`sample_dy` authored
    only when non-zero (their schema default is 0 — "defaults are never
    materialized into the file", `level_lib.format_entity_lines`'s own
    contract) — a BODY-family sensor (temperature/fire, which ignores them)
    or an AIR-family sensor placed with no drag gets a clean `x`/`y`-only
    block."""
    fields = {"x": int(x), "y": int(y), "sample_dx": int(dx),
             "sample_dy": int(dy)}
    authored = ["x", "y"]
    if dx:
        authored.append("sample_dx")
    if dy:
        authored.append("sample_dy")
    return EntityInstance(id=id_, class_name=cls_name, ordinal=0, tags=(),
                          fields=fields, authored_keys=tuple(authored))


def commit_sensor_placement(log, entities, instance: EntityInstance):
    """The ONE `CollectionOp("entities")` transaction a sensor placement
    commits (editor doc §6: sensors do not stamp grid material — they
    occupy their mount tile logically). `log` is an
    `undo_log.TransactionLog` whose ctx already registers `"entities"`;
    `entities` is the SAME live list the ctx holds. Returns the committed
    `Transaction` (never `None` — appending always changes the list)."""
    log.begin(f"place {instance.class_name}")
    log.snapshot_coll("entities")
    entities.append(instance)
    return log.commit()
