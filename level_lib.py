"""level_lib: THE single read+write data layer for level folders.

Entity system design 2026-07-18 §3c (LOCKED): the map editor, the baker's
TOML writeback, the migration tool, and ML variant generation are all
CLIENTS of this module — one writer implementation, ever. The read side
delegates to :mod:`level_loader` (whose public ``load()``/``LevelData``
keep working for every existing caller); the write side owns managed-block
writeback for ``level.toml``.

Managed families (``MANAGED_FAMILIES``): ``[[spawn]]``, ``[[light]]``,
``[water]``, ``[[entity]]`` (A3), ``[[wire]]`` (B1) and — since Arc C9 —
``[art.bare]``/``[art.align]``/``[bake]`` (the baker's writeback, ported off
its original bespoke line-targeted regex upsert); the writer is
family-generic. On save every existing table of a replaced
family is removed and the new block is written at the position of the first
one (or appended at EOF); every byte OUTSIDE the managed tables — comments,
[art]/[bake] blocks, hand formatting, newline style — is preserved exactly.
A multi-family replace is applied in memory and lands as ONE atomic write
(temp file + ``os.replace``), so a crash mid-save can never leave a torn
level.toml (level editor v3 design §6).

Two-writers session conflicts: :func:`open_level` returns a
:class:`LevelHandle` that records level.toml's mtime+hash at load and after
every save; ``check_stale()`` compares the disk state against the record so
the Arc C editor can prompt reload-or-overwrite on mismatch.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Make the repo-root siblings importable regardless of cwd (level_loader's
# own bootstrap then covers src/).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import level_loader

WATER_FILENAME = "water_init.npy"
ZONES_FILENAME = level_loader.ZONES_FILENAME    # "zones.npy" — one source
AIR_INIT_FILENAME = level_loader.AIR_INIT_FILENAME  # "air_init.npy" (A9)

_TABLE_HEADER_RE = re.compile(r"^\s*\[")          # any table / array-of-tables


# ---------------------------------------------------------------------------
# Managed-family registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManagedFamily:
    """One managed block family of level.toml — a name plus its table
    spelling (``[[name]]`` array-of-tables vs ``[name]`` single table).
    ``header_re`` matches a header line, optionally followed by a comment;
    a table's span runs from its header through the line before the next
    table header of ANY kind."""
    name: str
    array: bool

    @property
    def header_re(self) -> re.Pattern:
        # re.escape: "art.bare"/"art.align" carry a literal dot, which must
        # match itself, not the regex any-char wildcard.
        name = re.escape(self.name)
        if self.array:
            return re.compile(r"^\s*\[\[\s*" + name + r"\s*\]\]\s*(#.*)?$")
        return re.compile(r"^\s*\[\s*" + name + r"\s*\]\s*(#.*)?$")


MANAGED_FAMILIES = {
    "spawn": ManagedFamily("spawn", array=True),
    "light": ManagedFamily("light", array=True),
    "water": ManagedFamily("water", array=False),
    # A3 ([[entity]] format): one registry entry, zero writer changes —
    # exactly the promise the family-generic writer was built on.
    "entity": ManagedFamily("entity", array=True),
    # B1 ([[wire]] logic bindings): another array-of-tables family, same
    # family-generic writer, byte-stable round-trip via format_wire_lines.
    "wire": ManagedFamily("wire", array=True),
    # Arc C9 rider (A2 accepted gap, closed): the baker's [art.bare]/
    # [art.align]/[bake] TOML writeback, formerly a bespoke hand-rolled
    # regex line-upsert in tools/bake_level_art.py, ported onto THE single
    # writer so it composes atomically (one temp+rename) with whatever
    # other families a save touches. Dotted single-table headers, same as
    # [water] — see format_art_bare_lines/format_art_align_lines/
    # format_bake_lines below.
    "art.bare": ManagedFamily("art.bare", array=False),
    "art.align": ManagedFamily("art.align", array=False),
    "bake": ManagedFamily("bake", array=False),
}


# ---------------------------------------------------------------------------
# Formatting — schema objects -> managed-block lines
# ---------------------------------------------------------------------------

def _fmt_coord(v) -> str:
    """Float formatting for coordinates: repr(float) is the shortest exact
    round-trip form (3.0 -> '3.0', 10.666667 stays 10.666667)."""
    return repr(float(v))


def color_255(color) -> tuple:
    """Normalized 0-1 color -> the 0-255 int triple the toml schema wants
    (level_loader divides by 255 at parse; round-trips int-sourced values
    exactly)."""
    return tuple(min(255, max(0, int(round(float(c) * 255.0))))
                 for c in color)


def format_spawn_lines(spawns, nl: str = "\n") -> list:
    """The managed [[spawn]] block as a list of ``nl``-terminated lines —
    one table per entry (name/team/x/y/footprint), blank line between
    entries. Schema per level_loader.SpawnEntry."""
    lines = []
    for i, s in enumerate(spawns):
        if i:
            lines.append(nl)
        name = str(s.name).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f"[[spawn]]{nl}")
        lines.append(f'name = "{name}"{nl}')
        lines.append(f"team = {int(s.team)}{nl}")
        lines.append(f"x = {_fmt_coord(s.x)}{nl}")
        lines.append(f"y = {_fmt_coord(s.y)}{nl}")
        lines.append(f"footprint = {int(s.footprint)}{nl}")
    return lines


def format_light_lines(lights, nl: str = "\n") -> list:
    """The managed [[light]] block as ``nl``-terminated lines — schema per
    level_loader.LightEntry / engine/15 §2.2 (color back to 0-255 ints;
    period_s/beam_deg/phase written for beacons only — static lights take
    the loader defaults)."""
    lines = []
    for i, l in enumerate(lights):
        if i:
            lines.append(nl)
        r, g, b = color_255(l.color)
        lines.append(f"[[light]]{nl}")
        lines.append(f"pos = [{_fmt_coord(l.x)}, {_fmt_coord(l.y)}]{nl}")
        lines.append(f"color = [{r}, {g}, {b}]{nl}")
        lines.append(f"intensity = {_fmt_coord(l.intensity)}{nl}")
        lines.append(f"range = {_fmt_coord(l.range)}{nl}")
        lines.append(f'kind = "{l.kind}"{nl}')
        if l.kind == "beacon":
            lines.append(f"period_s = {_fmt_coord(l.period_s)}{nl}")
            lines.append(f"beam_deg = {_fmt_coord(l.beam_deg)}{nl}")
            lines.append(f"phase = {_fmt_coord(l.phase)}{nl}")
    return lines


def format_water_lines(depth_map_rel: str = WATER_FILENAME,
                       nl: str = "\n") -> list:
    """The managed [water] block as ``nl``-terminated lines — schema per
    engine/15 §2.3 (P5): one table, one key, the .npy carrier."""
    return [f"[water]{nl}", f'depth_map = "{depth_map_rel}"{nl}']


def _fmt_value(v) -> str:
    """Generic TOML value formatting for entity fields: bools lowercase,
    strings escaped + quoted, ints plain, floats via repr (the shortest
    exact round-trip form), lists/tuples recursively."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        esc = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{esc}"'
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    raise ValueError(f"unsupported [[entity]] field value {v!r}")


def format_entity_lines(entities, nl: str = "\n") -> list:
    """The managed [[entity]] block as ``nl``-terminated lines — schema per
    level_loader.EntityInstance (A3, entity design §3a).

    Canonical form: ``id``, ``class``, ``tags`` (omitted when empty), then
    the AUTHORED schema fields in authored order — defaults are never
    materialized into the file, so load -> format round-trips byte-stably.
    The writer emits the list as given: file order is id-assignment order
    (design §3a), preserved end to end."""
    lines = []
    for i, e in enumerate(entities):
        if i:
            lines.append(nl)
        lines.append(f"[[entity]]{nl}")
        lines.append(f"id = {_fmt_value(e.id)}{nl}")
        lines.append(f"class = {_fmt_value(e.class_name)}{nl}")
        if e.tags:
            lines.append(f"tags = {_fmt_value(list(e.tags))}{nl}")
        for key in e.authored_keys:
            lines.append(f"{key} = {_fmt_value(e.fields[key])}{nl}")
    return lines


def format_wire_lines(wire_specs, nl: str = "\n") -> list:
    """The managed [[wire]] block as ``nl``-terminated lines — schema per
    level_loader.WireSpec (Arc B impl doc §1).

    Canonical form: ``from`` then ``to``, the authored dotted strings verbatim
    (a ``tag:name.input`` target is written back as the author's single line,
    never its pre-expanded members), blank line between entries. Load -> format
    round-trips byte-stably. Accepts WireSpec objects (``.from_``/``.to``) or
    ``(from, to)`` pairs."""
    lines = []
    for i, w in enumerate(wire_specs):
        if i:
            lines.append(nl)
        from_ = w.from_ if hasattr(w, "from_") else w[0]
        to = w.to if hasattr(w, "to") else w[1]
        lines.append(f"[[wire]]{nl}")
        lines.append(f"from = {_fmt_value(str(from_))}{nl}")
        lines.append(f"to = {_fmt_value(str(to))}{nl}")
    return lines


def format_art_bare_lines(diffuse: str, normal: str, nl: str = "\n") -> list:
    """The managed ``[art.bare]`` block as ``nl``-terminated lines — the
    baker's diffuse/normal PNG filenames (engine/15 §4 P2). Arc C9 rider:
    ported off ``tools/bake_level_art.py``'s original bespoke line-targeted
    regex upsert onto this family-generic writer — format/meaning
    unchanged, only WHERE the write happens."""
    return [
        f"[art.bare]{nl}",
        f'diffuse = "{diffuse}"{nl}',
        f'normal = "{normal}"{nl}',
    ]


def format_art_align_lines(px_per_tile, nl: str = "\n") -> list:
    """The managed ``[art.align]`` block. ``offset_px`` is always
    ``[0.0, 0.0]`` for a bake (a nonzero offset is ``align_level_art.py``'s
    own job, untouched by the baker); ``px_per_tile`` is the bake's per-axis
    pair, 2-decimal formatting (the pre-existing save_align house style,
    unchanged by the Arc C9 port)."""
    ppt = float(px_per_tile)
    return [
        f"[art.align]{nl}",
        f"offset_px = [0.0, 0.0]{nl}",
        f"px_per_tile = [{ppt:.2f}, {ppt:.2f}]{nl}",
    ]


def format_bake_lines(tileset_rel: str, px_per_tile: int, seed: int,
                      nl: str = "\n") -> list:
    """The managed ``[bake]`` block — the recorded parameters a bare
    re-bake (``bake_level_art.bake_level(level_dir)``, all-``None``
    arguments) replicates exactly (engine/15 §4)."""
    return [
        f"[bake]{nl}",
        f'tileset = "{tileset_rel}"{nl}',
        f"px_per_tile = {int(px_per_tile)}{nl}",
        f"seed = {int(seed)}{nl}",
    ]


# ---------------------------------------------------------------------------
# THE writer — multi-family managed-block replace, one atomic write
# ---------------------------------------------------------------------------

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: temp file in the same
    directory, flush + fsync, then ``os.replace`` — readers see either the
    old bytes or the new bytes, never a torn file. The temp file is removed
    on any failure."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _replace_family_lines(out: list, header_re: re.Pattern,
                          block: list, nl: str) -> None:
    """Replace ONE managed family in the keepends line list ``out`` IN
    PLACE (the reported P3 design call, generalized for every family).

    Every existing table matching ``header_re`` (header line through the
    line before the next table header) is removed and ``block`` is inserted
    at the position of the FIRST one — or appended at EOF when the file had
    none. An empty ``block`` removes the family. Every line OUTSIDE the
    managed tables is untouched; comments INSIDE individual managed tables
    are managed away."""
    spans = []                                # existing managed tables
    i = 0
    while i < len(out):
        if header_re.match(out[i]):
            j = next((k for k in range(i + 1, len(out))
                      if _TABLE_HEADER_RE.match(out[k])), len(out))
            spans.append((i, j))
            i = j
        else:
            i += 1
    insert_at = spans[0][0] if spans else None
    for a, b in reversed(spans):
        del out[a:b]

    if insert_at is None:
        if block:
            if out and not out[-1].endswith(("\n", "\r")):
                out[-1] += nl
            out += [nl] + block
    else:
        if block and insert_at < len(out) and out[insert_at].strip():
            block = block + [nl]     # keep a blank line before the next table
        out[insert_at:insert_at] = block


def write_managed_blocks(toml_path, replacements: dict,
                         *, write_bak: bool = False):
    """Rewrite managed families of ``toml_path`` in ONE atomic write.

    ``replacements`` maps a MANAGED_FAMILIES name to a ``format_lines``
    callable (``nl -> list of nl-terminated lines``; an empty list removes
    the family). Families NOT in the dict are preserved byte-for-byte, as
    is everything outside managed tables (newline style included). When
    ``write_bak`` is True the original bytes go to ``<name>.bak`` first
    (interactive tools pass True once per session — the .bak carries the
    pre-session state). Returns the .bak path, or None when not written."""
    toml_path = Path(toml_path)
    unknown = sorted(set(replacements) - set(MANAGED_FAMILIES))
    if unknown:
        raise ValueError(
            f"unknown managed famil{'y' if len(unknown) == 1 else 'ies'} "
            f"{unknown}; known: {sorted(MANAGED_FAMILIES)}")
    original = toml_path.read_bytes()
    text = original.decode("utf-8")
    out = text.splitlines(keepends=True)
    nl = "\r\n" if "\r\n" in text else "\n"   # match the file's newline style

    for name, format_lines in replacements.items():
        _replace_family_lines(out, MANAGED_FAMILIES[name].header_re,
                              format_lines(nl), nl)

    bak = None
    if write_bak:
        bak = Path(str(toml_path) + ".bak")
        bak.write_bytes(original)
    _atomic_write_bytes(toml_path, "".join(out).encode("utf-8"))
    return bak


# ---------------------------------------------------------------------------
# Single-family conveniences (the pre-A2 editor writeback surface)
# ---------------------------------------------------------------------------

def write_spawns(toml_path, spawns, write_bak: bool = True):
    """Rewrite the ``[[spawn]]`` array-of-tables as ONE managed block —
    see :func:`write_managed_blocks` for the byte-preservation and .bak
    contract. Returns the .bak path, or None when not written."""
    return write_managed_blocks(
        toml_path, {"spawn": lambda nl: format_spawn_lines(spawns, nl)},
        write_bak=write_bak)


def write_lights(toml_path, lights, write_bak: bool = True):
    """Rewrite the ``[[light]]`` array-of-tables as ONE managed block —
    see :func:`write_managed_blocks` for the byte-preservation and .bak
    contract. Returns the .bak path, or None when not written."""
    return write_managed_blocks(
        toml_path, {"light": lambda nl: format_light_lines(lights, nl)},
        write_bak=write_bak)


def write_water_npy(level_dir, depth_q: np.ndarray, *,
                    npy_bak: bool = True):
    """Write (or, all-dry, delete) ``water_init.npy`` — int32 Q16.16, the
    file IS the field (P5 §2.1). The .npy carries its OWN once-per-session
    pre-session .bak (``npy_bak`` True on the session's first save, and
    only if the file predates the session). Returns
    ``(npy_bak_path | None, has_water)``."""
    level_dir = Path(level_dir)
    npy_path = level_dir / WATER_FILENAME
    d = np.ascontiguousarray(np.asarray(depth_q, dtype=np.int32))
    has_water = bool(d.any())
    nbak = None
    if npy_bak and npy_path.is_file():
        nbak = Path(str(npy_path) + ".bak")
        nbak.write_bytes(npy_path.read_bytes())
    if has_water:
        np.save(npy_path, d)
    elif npy_path.is_file():
        npy_path.unlink()
    return nbak, has_water


def write_zones_npy(level_dir, zone_grid, *, npy_bak: bool = True):
    """Write (or, unpainted-everywhere, delete) ``zones.npy`` — uint8 paint
    ids, the file IS the mask (editor design §5, A8: water's .npy-carrier
    pattern MINUS the toml key — the loader discovers zones.npy by presence,
    so no managed [zones] family exists and none is written). All-zero grid
    = no zones: the file is deleted, matching the loader's absent-file
    dormancy. The .npy carries its OWN once-per-session pre-session .bak
    (``npy_bak`` True on the session's first save), mirroring
    :func:`write_water_npy`. Returns ``(npy_bak_path | None, has_zones)``.

    The caller may pass any integer-dtype grid; values are range-checked to
    0..255 (paint ids are 1..255, 0 = unpainted — the schema's zone_id
    bounds) and stored as uint8, the loader's pinned on-disk dtype.
    """
    level_dir = Path(level_dir)
    npy_path = level_dir / ZONES_FILENAME
    g = np.asarray(zone_grid)
    if not np.issubdtype(g.dtype, np.integer):
        raise ValueError(
            f"zone grid must be integer paint ids (stored uint8), got "
            f"dtype {g.dtype}")
    if g.size and (int(g.min()) < 0 or int(g.max()) > 255):
        raise ValueError(
            f"zone paint ids must fit uint8 (0..255, 0 = unpainted), got "
            f"range [{int(g.min())}, {int(g.max())}]")
    g = np.ascontiguousarray(g.astype(np.uint8))
    has_zones = bool(g.any())
    nbak = None
    if npy_bak and npy_path.is_file():
        nbak = Path(str(npy_path) + ".bak")
        nbak.write_bytes(npy_path.read_bytes())
    if has_zones:
        np.save(npy_path, g)
    elif npy_path.is_file():
        npy_path.unlink()
    return nbak, has_zones


def write_air_init_npy(level_dir, air_q, *, npy_bak: bool = True):
    """Write (or, on ``None``/empty, delete) ``air_init.npy`` — int32
    Q16.16 atm, the file IS the field (A9; the zones.npy presence pattern,
    no toml key).

    THE DELETE RULE differs deliberately from water/zones: their grids have
    a content value that MEANS "nothing here" (all-dry / all-unpainted ==
    absent file), so an all-default grid deletes. For air there is NO such
    value — absence means "derive ambient as today", while 0 is a real
    authored state (a depressurized start) and FP_ONE is a real pinned
    ambient — so an explicit grid ALWAYS stays a file (even all-zero, even
    all-ambient) and only ``air_q=None`` (or an empty array) removes the
    override. The .npy carries its OWN once-per-session pre-session .bak
    (``npy_bak`` True on the session's first save), mirroring
    :func:`write_water_npy`. Returns ``(npy_bak_path | None, has_air)``.

    The caller may pass any integer-dtype grid; values are range-checked to
    non-negative int32 (the loader's pinned on-disk dtype + its negative-
    pressure hard error) and stored as int32.
    """
    level_dir = Path(level_dir)
    npy_path = level_dir / AIR_INIT_FILENAME
    if air_q is None or np.asarray(air_q).size == 0:
        nbak = None
        if npy_bak and npy_path.is_file():
            nbak = Path(str(npy_path) + ".bak")
            nbak.write_bytes(npy_path.read_bytes())
        if npy_path.is_file():
            npy_path.unlink()
        return nbak, False
    a = np.asarray(air_q)
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError(
            f"air grid must be integer Q16.16 atm counts (stored int32), "
            f"got dtype {a.dtype}")
    if int(a.min()) < 0:
        raise ValueError(
            f"air grid contains negative pressures (min = {int(a.min())} "
            f"raw Q16.16) — the loader hard-errors on them")
    if int(a.max()) > np.iinfo(np.int32).max:
        raise ValueError(
            f"air grid values must fit int32 Q16.16, got max {int(a.max())}")
    a = np.ascontiguousarray(a.astype(np.int32))
    nbak = None
    if npy_bak and npy_path.is_file():
        nbak = Path(str(npy_path) + ".bak")
        nbak.write_bytes(npy_path.read_bytes())
    np.save(npy_path, a)
    return nbak, True


_BOUNDARY_LINE_RE = re.compile(
    r'^\s*boundary\s*=\s*"(space|ambient)"\s*(#.*)?$')


def write_boundary_field(toml_path, boundary: str, *,
                         write_bak: bool = False):
    """Set (or insert) the top-level ``boundary`` scalar (editor design §7 /
    canon engine/16 §7 — format + client-side writeback ONLY, never the
    physics it will eventually drive).

    ``boundary`` is validated against :data:`level_loader.BOUNDARY_MODES` so
    the editor can never write a value its own loader would then reject.
    Unlike every other managed field, ``boundary`` is a BARE top-level key —
    not a ``[[table]]``/``[table]`` header — so it does not fit
    :func:`write_managed_blocks`' table-span replace; this is its own tiny
    single-line find-or-insert, same atomic temp+rename write. An existing
    ``boundary = "..."`` line is replaced in place; an absent one is
    inserted right before the FIRST table header (keeping it grouped with
    the file's other top-level scalars, never inside a managed block); a
    file with no table at all (never true for a map-editor level, which
    requires ``[bake]``) appends at EOF. Every other byte is untouched.
    Returns the ``.bak`` path, or ``None``."""
    boundary = str(boundary)
    if boundary not in level_loader.BOUNDARY_MODES:
        raise ValueError(
            f"boundary must be one of {level_loader.BOUNDARY_MODES}, "
            f"got {boundary!r}")
    toml_path = Path(toml_path)
    original = toml_path.read_bytes()
    text = original.decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    out = text.splitlines(keepends=True)
    new_line = f'boundary = "{boundary}"{nl}'
    for i, line in enumerate(out):
        if _BOUNDARY_LINE_RE.match(line):
            out[i] = new_line
            break
        if _TABLE_HEADER_RE.match(line):
            out.insert(i, new_line)
            break
    else:
        if out and not out[-1].endswith(("\n", "\r")):
            out[-1] += nl
        out.append(new_line)
    bak = None
    if write_bak:
        bak = Path(str(toml_path) + ".bak")
        bak.write_bytes(original)
    _atomic_write_bytes(toml_path, "".join(out).encode("utf-8"))
    return bak


def write_tilemap_csv(level_dir, grid, *, tilemap_rel: str = "tilemap.csv",
                      csv_bak: bool = True):
    """Write a level's tilemap CSV atomically — the migration tool's grid
    writer (A7; entity doc §3c: one writer implementation, ever).

    Canonical form: plain ints, comma-separated, one row per line, trailing
    newline; the existing file's newline style (LF/CRLF) is preserved.
    Every committed level is already byte-identical to this canonical form
    (callers that need that guarantee — e.g. the migration tool's
    only-cell-diffs contract — verify it BEFORE calling). The write goes
    through the same-directory temp + ``os.replace`` path as level.toml.
    When ``csv_bak`` is True and the file exists, the original bytes go to
    ``<name>.bak`` first. Returns the .bak path, or None.
    """
    level_dir = Path(level_dir)
    csv_path = level_dir / tilemap_rel
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError(f"tilemap grid must be 2D, got shape {g.shape}")
    if not np.issubdtype(g.dtype, np.integer):
        raise ValueError(f"tilemap grid must be integer codes, got {g.dtype}")
    newline = "\n"
    bak = None
    if csv_path.is_file():
        original = csv_path.read_bytes()
        if b"\r\n" in original:
            newline = "\r\n"
        if csv_bak:
            bak = Path(str(csv_path) + ".bak")
            bak.write_bytes(original)
    text = newline.join(
        ",".join(str(int(v)) for v in row) for row in g.tolist()) + newline
    _atomic_write_bytes(csv_path, text.encode("ascii"))
    return bak


def water_block_format(has_water: bool):
    """The [water] family's ``format_lines`` callable for
    :func:`write_managed_blocks`: the one-key .npy carrier table when the
    level has water, the empty block (= remove the family) when all-dry —
    the dormancy pin (a dry level carries no [water] key at all)."""
    return (lambda nl: format_water_lines(WATER_FILENAME, nl)) if has_water \
        else (lambda nl: [])


def write_water(level_dir, depth_q: np.ndarray, *,
                toml_bak: bool = False, npy_bak: bool = True):
    """SAVE-time [water] writeback (P5 §2.4): :func:`write_water_npy` +
    the managed ``[water]`` table via :func:`write_managed_blocks`. An
    all-dry grid REMOVES the block and deletes the stale .npy. The caller
    passes ``depth_q`` ALREADY masked (map_editor.mask_water_to_open).
    Returns ``(npy_bak_path | None, toml_bak_path | None, has_water)``."""
    level_dir = Path(level_dir)
    nbak, has_water = write_water_npy(level_dir, depth_q, npy_bak=npy_bak)
    tbak = write_managed_blocks(
        level_dir / "level.toml", {"water": water_block_format(has_water)},
        write_bak=toml_bak)
    return nbak, tbak, has_water


# ---------------------------------------------------------------------------
# The read side — level_loader delegation + the session handle
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class LevelHandle:
    """One opened level folder: the parsed :class:`level_loader.LevelData`
    plus level.toml's recorded disk state (mtime+hash at load and after
    every save) — the two-writers seam of entity doc §3c. The Arc C editor
    prompts reload-or-overwrite when :meth:`check_stale` is True."""
    data: level_loader.LevelData
    toml_path: Path
    toml_mtime_ns: int = 0
    toml_sha256: str = ""

    @property
    def level_dir(self) -> Path:
        return self.data.path

    def record_disk_state(self) -> None:
        """Re-snapshot level.toml's mtime+hash (called at open, after every
        save, and by clients that rewrote the file through another
        level_lib entry point)."""
        content = self.toml_path.read_bytes()
        self.toml_mtime_ns = int(self.toml_path.stat().st_mtime_ns)
        self.toml_sha256 = _sha256(content)

    def check_stale(self) -> bool:
        """True when level.toml on disk no longer matches the recorded
        state (another writer touched it — or it vanished). Unchanged mtime
        is trusted as the fast path; a changed mtime falls through to the
        hash, so a touch-without-change never reads as stale."""
        try:
            st = self.toml_path.stat()
        except OSError:
            return True
        if int(st.st_mtime_ns) == self.toml_mtime_ns:
            return False
        return _sha256(self.toml_path.read_bytes()) != self.toml_sha256

    def save(self, replacements: Optional[dict] = None,
             *, write_bak: bool = False):
        """Managed-block save through :func:`write_managed_blocks` (one
        atomic write; no replacements = rewrite the file byte-identically),
        then re-record mtime+hash. Returns the .bak path, or None."""
        bak = write_managed_blocks(self.toml_path, replacements or {},
                                   write_bak=write_bak)
        self.record_disk_state()
        return bak


def open_level(level_name: str, levels_dir: str = "levels") -> LevelHandle:
    """Open a level folder: parse via :func:`level_loader.load` (the read
    side stays level_loader's — never forked) and record level.toml's
    mtime+hash for staleness detection. Raises ValueError exactly like
    ``level_loader.load``."""
    data = level_loader.load(str(level_name), levels_dir)
    handle = LevelHandle(data=data, toml_path=data.path / "level.toml")
    handle.record_disk_state()
    return handle
