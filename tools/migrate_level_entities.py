"""One-shot legacy -> [[entity]] level migration — Arc A patch A7.

Level editor v3 design §6 (LOCKED): "Legacy migration is explicit, never a
save side effect." This tool converts a level folder IN PLACE, whole-level
at once, with a ``.bak`` of every touched file:

  (a) painted ``MAT_DOOR`` runs in tilemap.csv  ->  ``door`` [[entity]]
      instances (ids ``door_1..n`` in scan order), and the painted tiles
      are rewritten ``MAT_DOOR`` (3) -> ``MAT_DOOR_CLOSED`` (7).
      THE SANCTIONED BEHAVIOR CHANGE (a6 doors design §1, arc plan ruling
      2): a legacy painted door is the walkable-but-flow-solid HYBRID
      (``MAT_DOOR.mobility = 1000``); a migrated door is an entity whose
      CLOSED state is FULLY solid — flow AND movement. "A door standing
      there" honestly reads as a closed door, so migrated doors are
      ``initial_state = "closed"`` and their tiles become truly solid.
      This is exactly the change A7's single deliberate re-baseline
      sanctions (docs/a7_rebaseline_rationale_2026-07-19.md).
  (b) legacy ``[[light]]`` blocks  ->  ``light`` [[entity]] instances
      (ids ``light_1..n`` in file order, same fields — the alias
      equivalence contract of the A3 loader). Digest consequence per the
      A4 impl note: entity lights trip digest presence, so lights-only
      levels flip digests too — A7-scoped by design.
  (c) ``[[spawn]]`` is untouched — permanent (units are NOT entities,
      entity design §3e).

Grouping rule for painted runs: maximal HORIZONTAL runs (length >= 2)
claim first in row-major scan order, then maximal VERTICAL runs (length
>= 2) among the remaining tiles, then leftover singletons become 1-tile
"h" doors. L-shaped adjacencies therefore stay separate h/v runs (spans
disjoint — the GameMap §4.2 overlap rule). Door ids number door_1..n over
all runs sorted by anchor (row, col).

Safety rails:

  - idempotent: a level with nothing to migrate (already entity-form, or
    never had legacy doors/lights) is a report-and-exit no-op.
  - a HALF-migrated level (entity doors/lights present ALONGSIDE painted
    MAT_DOOR or legacy [[light]]) is refused as corrupt — restore from
    the ``.bak`` files and re-run. (The loader itself already hard-errors
    the [[light]]-plus-entity-light mix.)
  - v2 levels only: the painted-door rewrite is v2 CSV vocabulary; run
    tools/migrate_tilemap_v2.py first on a v1 level.
  - all validation happens BEFORE any write; level.toml is written first
    (via level_lib, THE writer — one atomic managed-block replace), then
    tilemap.csv (atomic too): a crash between the two leaves a level that
    still loads (door spans may sit on MAT_DOOR) and that this tool then
    refuses as half-migrated, with the .baks there to recover.
  - post-write verification: the migrated level is reloaded; lights must
    be field-for-field equivalent, spawns unchanged, door spans must
    cover exactly the previously painted tiles, and a GameMap must
    construct (span bounds / vacuum / overlap validation + field seeding).
  - ``--dry-run`` prints the full plan and writes nothing.

Usage:
    conda run -n data python tools/migrate_level_entities.py <level> [...]
        [--dry-run]

``<level>`` is a levels/ folder name (e.g. ``test_level``) or a path.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field as dc_field
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import level_lib
import level_loader
from level_loader import EntityInstance
from simulation.entities import (REGISTRY as ENTITY_REGISTRY,
                                 effective_defaults, field_value_error)
from simulation.entities import door as door_schema
from simulation.materials import MAT_DOOR, MAT_DOOR_CLOSED

# The [[light]] keys this tool knows how to carry over 1:1. The loader has
# already validated the block (open_level parses it), so anything else here
# is a logic drift between loader and tool — refuse loudly rather than drop.
_LIGHT_SCALAR_KEYS = ("intensity", "range", "kind",
                      "period_s", "beam_deg", "phase")
_LIGHT_KNOWN_KEYS = frozenset(("pos", "color") + _LIGHT_SCALAR_KEYS)


@dataclass
class MigrationPlan:
    """What one migrate_level() call found / did (returned for callers +
    tests; the CLI prints it)."""
    level_dir: Path
    noop: bool = False
    dry_run: bool = False
    doors: list = dc_field(default_factory=list)    # EntityInstance
    lights: list = dc_field(default_factory=list)   # EntityInstance
    door_tiles: list = dc_field(default_factory=list)  # painted (row, col)
    toml_bak: Path | None = None
    csv_bak: Path | None = None


# ---------------------------------------------------------------------------
# Painted-door run grouping
# ---------------------------------------------------------------------------

def door_runs(tilemap: np.ndarray) -> list:
    """Group painted MAT_DOOR tiles into maximal straight runs.

    Returns ``[(orientation, [(row, col), ...]), ...]`` sorted by anchor
    (row, col). Horizontal maximal runs (length >= 2) claim first in
    row-major order; vertical maximal runs (length >= 2) among the
    REMAINING tiles second; leftover singletons become 1-tile "h" runs.
    Every painted tile lands in exactly one run.
    """
    painted = {(int(r), int(c)) for r, c in np.argwhere(tilemap == MAT_DOOR)}
    runs: list = []
    claimed: set = set()

    for (r, c) in sorted(painted):                 # horizontal pass
        if (r, c) in claimed or (r, c - 1) in painted:
            continue                               # claimed or not a run start
        run = [(r, c)]
        cc = c + 1
        while (r, cc) in painted:
            run.append((r, cc))
            cc += 1
        if len(run) >= 2:
            runs.append(("h", run))
            claimed.update(run)

    remaining = painted - claimed
    for (r, c) in sorted(remaining):               # vertical pass
        if (r, c) in claimed or (r - 1, c) in remaining:
            continue
        run = [(r, c)]
        rr = r + 1
        while (rr, c) in remaining:
            run.append((rr, c))
            rr += 1
        if len(run) >= 2:
            runs.append(("v", run))
            claimed.update(run)

    for t in sorted(painted - claimed):            # singletons -> 1-tile "h"
        runs.append(("h", [t]))

    runs.sort(key=lambda oc: (oc[1][0][0], oc[1][0][1], oc[0]))
    return runs


def length_m_for_tiles(n: int, tile_size_m: float, *,
                       context: str = "door") -> float:
    """The shortest decimal ``length_m`` whose canonical quantization
    (simulation.entities.door.quantize_span_tiles — round-half-up in exact
    Fraction arithmetic) yields exactly ``n`` tiles at this level's base
    ``tile_size_m``. Verified through THE canonical rule before returning
    — a mismatch is a hard error, never a silently resized door."""
    tpm = door_schema.tiles_per_m(tile_size_m)
    exact = Fraction(n, tpm)
    if exact.denominator == 1:
        s = f"{int(exact)}.0"
    else:
        s = f"{n / tpm:.3f}".rstrip("0")
        if s.endswith("."):
            s += "0"
    val = float(s)
    got = door_schema.quantize_span_tiles(val, tile_size_m, context=context)
    if got != n:
        raise ValueError(
            f"{context}: no stable length_m found for a {n}-tile run at "
            f"tile_size_m={tile_size_m!r} (candidate {s} quantizes to "
            f"{got} tiles) — the canonical rule (a6 doors design §3) "
            f"cannot express this run")
    return val


def _door_instances(runs, tile_size_m, start_ordinal, taken_ids) -> list:
    """Painted runs -> closed ``door`` EntityInstances, ids door_1..n."""
    out = []
    for i, (orient, tiles) in enumerate(runs, 1):
        eid = f"door_{i}"
        if eid in taken_ids:
            raise ValueError(
                f"generated id '{eid}' collides with an existing [[entity]] "
                f"id — rename that instance and re-run")
        r0, c0 = tiles[0]
        ctx = f"migrated door '{eid}'"
        fields = effective_defaults("door")
        fields.update(x=int(c0), y=int(r0), orientation=orient,
                      length_m=length_m_for_tiles(len(tiles), tile_size_m,
                                                  context=ctx),
                      initial_state="closed")
        span = door_schema.base_span(fields, tile_size_m, context=ctx)
        if span != tiles:
            raise ValueError(
                f"{ctx}: derived span {span} != painted run {tiles} — "
                f"quantization drift (a6 doors design §3)")
        out.append(EntityInstance(
            id=eid, class_name="door", ordinal=start_ordinal + i - 1,
            fields=fields,
            authored_keys=("x", "y", "orientation", "length_m",
                           "initial_state")))
    return out


# ---------------------------------------------------------------------------
# [[light]] -> entity light conversion
# ---------------------------------------------------------------------------

def _light_instances(raw_lights, start_ordinal, taken_ids) -> list:
    """Legacy ``[[light]]`` tables (RAW toml dicts, already loader-validated)
    -> ``light`` EntityInstances, ids light_1..n in file order. Authored
    values carry over VERBATIM (pos -> x/y, the rest 1:1); keys the file
    did not spell stay unauthored so defaults never materialize into the
    file (the A3 round-trip contract)."""
    out = []
    for i, entry in enumerate(raw_lights, 1):
        eid = f"light_{i}"
        if eid in taken_ids:
            raise ValueError(
                f"generated id '{eid}' collides with an existing [[entity]] "
                f"id — rename that instance and re-run")
        unknown = sorted(set(entry) - _LIGHT_KNOWN_KEYS)
        if unknown:
            raise ValueError(
                f"[[light]] entry #{i - 1} carries key(s) {unknown} this "
                f"tool does not know how to migrate — loader/tool drift, "
                f"refuse rather than drop")
        fields = effective_defaults("light")
        fields["x"], fields["y"] = entry["pos"][0], entry["pos"][1]
        fields["color"] = [int(v) for v in entry["color"]]
        authored = ["x", "y", "color"]
        for key in _LIGHT_SCALAR_KEYS:
            if key in entry:
                fields[key] = entry[key]
                authored.append(key)
        # Validate through THE registry schema BEFORE any write: legacy
        # [[light]] bounds are looser in places (e.g. phase), and a value
        # the entity schema rejects must refuse the migration up front,
        # never fail verification after the file moved.
        schema_fields = {f.name: f for f in ENTITY_REGISTRY["light"].FIELDS}
        for key in authored:
            verr = field_value_error(schema_fields[key], fields[key])
            if verr:
                raise ValueError(
                    f"[[light]] entry #{i - 1} -> '{eid}': '{key}' = "
                    f"{fields[key]!r} {verr} — fix the level, then re-run")
        out.append(EntityInstance(
            id=eid, class_name="light", ordinal=start_ordinal + i - 1,
            fields=fields, authored_keys=tuple(authored)))
    return out


# ---------------------------------------------------------------------------
# The one-shot migration
# ---------------------------------------------------------------------------

def _canonical_csv_bytes(csv_bytes: bytes, grid: np.ndarray) -> bytes:
    newline = "\r\n" if b"\r\n" in csv_bytes else "\n"
    return (newline.join(
        ",".join(str(int(v)) for v in row) for row in grid.tolist())
        + newline).encode("ascii")


def migrate_level(level_dir, *, dry_run: bool = False,
                  verbose: bool = True) -> MigrationPlan:
    """Migrate one level folder in place (or plan it, with ``dry_run``).

    Raises ValueError on refusal (v1 level, half-migrated level, id
    collision, non-canonical CSV, verification failure). Returns the
    :class:`MigrationPlan` — ``noop=True`` when there was nothing to do.
    """
    level_dir = Path(level_dir)
    say = print if verbose else (lambda *a, **k: None)
    plan = MigrationPlan(level_dir=level_dir, dry_run=dry_run)

    # -- read (level_lib's read side == level_loader; the loader already
    # hard-errors the [[light]]+entity-light mix, naming this tool) --------
    handle = level_lib.open_level(str(level_dir))
    lvl = handle.data
    if str(lvl.version) != "2":
        raise ValueError(
            f"{level_dir}: level format version {lvl.version!r} — the "
            f"painted-door rewrite is v2 CSV vocabulary; run "
            f"tools/migrate_tilemap_v2.py first")

    raw_lights = lvl.raw_toml.get("light", [])
    runs = door_runs(lvl.tilemap)
    plan.door_tiles = sorted(t for _, tiles in runs for t in tiles)
    legacy_present = bool(runs or raw_lights)

    migrated_classes = sorted({e.class_name for e in lvl.entities
                               if e.class_name in ("door", "light")})
    if legacy_present and migrated_classes:
        raise ValueError(
            f"{level_dir}: HALF-MIGRATED level — [[entity]] "
            f"{migrated_classes} instances coexist with legacy forms "
            f"({len(plan.door_tiles)} painted MAT_DOOR tile(s), "
            f"{len(raw_lights)} [[light]] block(s)). A level carries ONE "
            f"form (level editor v3 design §6); this state is corrupt. "
            f"Restore level.toml.bak / tilemap.csv.bak and re-run.")

    if not legacy_present:
        say(f"{level_dir.name}: nothing to migrate — no painted MAT_DOOR, "
            f"no legacy [[light]] (already migrated or never legacy). No-op.")
        plan.noop = True
        return plan

    # -- build the new instances (all validation before any write) --------
    taken = {e.id for e in lvl.entities}
    base_ordinal = len(lvl.entities)
    plan.doors = _door_instances(runs, float(lvl.tile_size_m),
                                 base_ordinal, taken)
    taken |= {e.id for e in plan.doors}
    plan.lights = _light_instances(raw_lights,
                                   base_ordinal + len(plan.doors), taken)
    all_entities = list(lvl.entities) + plan.doors + plan.lights

    csv_path = None
    new_grid = None
    if runs:
        tilemap_rel = lvl.raw_toml.get("tilemap")
        csv_path = lvl.path / tilemap_rel
        csv_bytes = csv_path.read_bytes()
        if csv_bytes != _canonical_csv_bytes(csv_bytes, lvl.tilemap):
            raise ValueError(
                f"{csv_path}: not in canonical CSV form (plain ints, "
                f"comma-separated, trailing newline) — refusing to rewrite; "
                f"the migration guarantees cell-only diffs")
        new_grid = lvl.tilemap.copy()
        new_grid[new_grid == MAT_DOOR] = MAT_DOOR_CLOSED

    # -- report the plan --------------------------------------------------
    say(f"{level_dir.name}: migration plan")
    for d in plan.doors:
        f = d.fields
        say(f"  door  {d.id}: anchor (x={f['x']}, y={f['y']}) "
            f"orientation={f['orientation']} length_m={f['length_m']} "
            f"({len(door_schema.base_span(f, float(lvl.tile_size_m)))} "
            f"tile(s)) initial_state=closed  [MAT_DOOR -> MAT_DOOR_CLOSED]")
    for l in plan.lights:
        f = l.fields
        say(f"  light {l.id}: pos ({f['x']}, {f['y']}) kind={f['kind']}")
    if runs:
        say(f"  tilemap: {len(plan.door_tiles)} tile(s) rewritten "
            f"{MAT_DOOR} -> {MAT_DOOR_CLOSED}")
    say(f"  [[spawn]] untouched ({len(lvl.spawns)} entries — permanent)")
    if dry_run:
        say("  DRY RUN — nothing written.")
        return plan

    # -- write: level.toml first (THE writer, one atomic replace), then the
    # CSV — a crash in between leaves a loadable, tool-refusable state ----
    plan.toml_bak = handle.save({
        "light": lambda nl: [],
        "entity": lambda nl: level_lib.format_entity_lines(all_entities, nl),
    }, write_bak=True)
    if runs:
        plan.csv_bak = level_lib.write_tilemap_csv(
            lvl.path, new_grid, tilemap_rel=lvl.raw_toml.get("tilemap"),
            csv_bak=True)

    _verify_migration(level_dir, lvl, plan)
    say(f"  migrated. .bak written: {plan.toml_bak}"
        + (f", {plan.csv_bak}" if plan.csv_bak else ""))
    return plan


def _verify_migration(level_dir, old_lvl, plan) -> None:
    """Reload the migrated level and prove equivalence (hard errors)."""
    new_lvl = level_loader.load(str(level_dir))
    if int((new_lvl.tilemap == MAT_DOOR).sum()) != 0:
        raise ValueError("verification: painted MAT_DOOR tiles remain")
    # Door spans must cover exactly the previously painted tiles.
    span_tiles: list = []
    for e in new_lvl.entities:
        if e.class_name == "door":
            span_tiles += door_schema.base_span(
                e.fields, float(new_lvl.tile_size_m),
                context=f"door entity '{e.id}'")
    if sorted(span_tiles) != plan.door_tiles:
        raise ValueError(
            f"verification: door spans {sorted(span_tiles)} != painted "
            f"tiles {plan.door_tiles}")
    if int((new_lvl.tilemap == MAT_DOOR_CLOSED).sum()) < len(plan.door_tiles):
        raise ValueError("verification: MAT_DOOR_CLOSED stamp count short")
    # Lights: field-for-field equivalent LightEntry lists (the A3 alias
    # contract makes entity lights land in .lights the same way).
    if new_lvl.lights != old_lvl.lights:
        raise ValueError(
            f"verification: lights changed across migration:\n"
            f"  before: {old_lvl.lights}\n  after:  {new_lvl.lights}")
    if new_lvl.spawns != old_lvl.spawns:
        raise ValueError("verification: [[spawn]] changed across migration")
    # GameMap constructs: span bounds/vacuum/overlap validation + seeding.
    from simulation.gamemap import GameMap
    GameMap(new_lvl)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="One-shot legacy -> [[entity]] level migration (A7). "
                    "Converts painted MAT_DOOR runs to closed door entities "
                    "(tiles -> MAT_DOOR_CLOSED) and [[light]] blocks to "
                    "entity lights, in place, with .bak files.")
    ap.add_argument("levels", nargs="+",
                    help="levels/ folder name(s) or path(s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, write nothing")
    args = ap.parse_args(argv)

    rc = 0
    for name in args.levels:
        level_dir = Path(name)
        if not level_dir.is_dir():
            level_dir = ROOT / "levels" / name
        if not level_dir.is_dir():
            print(f"Level folder does not exist: {name}")
            rc = 2
            continue
        try:
            migrate_level(level_dir, dry_run=args.dry_run)
        except ValueError as e:
            print(f"REFUSED: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
