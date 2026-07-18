"""level_lib: THE single read+write data layer for level folders.

Entity system design 2026-07-18 §3c (LOCKED): the map editor, the baker's
TOML writeback, the migration tool, and ML variant generation are all
CLIENTS of this module — one writer implementation, ever. The read side
delegates to :mod:`level_loader` (whose public ``load()``/``LevelData``
keep working for every existing caller); the write side owns managed-block
writeback for ``level.toml``.

Managed families (``MANAGED_FAMILIES``): ``[[spawn]]``, ``[[light]]`` and
``[water]`` today; Arc A3 adds ``[[entity]]`` as one more registry entry —
the writer is family-generic. On save every existing table of a replaced
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
        if self.array:
            return re.compile(r"^\s*\[\[\s*" + self.name + r"\s*\]\]\s*(#.*)?$")
        return re.compile(r"^\s*\[\s*" + self.name + r"\s*\]\s*(#.*)?$")


# A3 adds "entity" here — one registry entry, no writer change.
MANAGED_FAMILIES = {
    "spawn": ManagedFamily("spawn", array=True),
    "light": ManagedFamily("light", array=True),
    "water": ManagedFamily("water", array=False),
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
