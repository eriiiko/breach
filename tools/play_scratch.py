r"""tools/play_scratch.py — play-from-editor (F5), the scratch-level bridge
(editor doc §8, C7).

F5 saves the LIVE editor state — unsaved edits included — to a fresh,
complete, loadable level folder at ``levels/_editor_scratch/<name>/``
(gitignored), then launches the real game against it via
``[sys.executable, "main.py", "--level", "_editor_scratch/<name>"]`` — NEVER
bare ``python`` (main.py's own documented machine footgun). The scratch
folder is deleted both when that subprocess exits and when the editor quits
(:func:`cleanup_scratch_dir`, best-effort, never raises).

:func:`write_scratch_level` REUSES the exact functions the map editor's
Ctrl+S already calls (``level_lib.write_tilemap_csv``/``write_water_npy``/
``write_zones_npy``/``write_air_init_npy``/``write_boundary_field``/
``write_managed_blocks``, ``light_entity_port.light_and_entity_replacements``,
``bake_level_art.compute_bake_replacements``) — never a parallel serializer.
The one new idea is copying the REAL level's already-baked PNGs into the
scratch dir when the material grid hasn't changed since they were last
written to disk (``bake_clean`` — the caller's own ``not log.dirty``, C2's
saved-marker) so F5 does not force a full re-bake every press; a dirty grid
always re-bakes from the freshly-written scratch tilemap, so a stale bake can
never ship.

Arc C9 rider: a DIRTY bake's [art.bare]/[art.align]/[bake] TOML replacements
now fold into the SAME ``write_managed_blocks`` call as spawn/water/wire/
light|entity, matching Ctrl+S's own one-shot atomic save exactly (method
equivalence, not just output equivalence) — see ``tools/map_editor.py``'s
Ctrl+S handler for the same pattern.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_lib  # noqa: E402
import light_entity_port  # noqa: E402
from bake_level_art import (DIFFUSE_FILENAME, NORMAL_FILENAME,  # noqa: E402
                            compute_bake_replacements)

SCRATCH_SUBDIR = "_editor_scratch"
SCRATCH_ROOT = ROOT / "levels" / SCRATCH_SUBDIR


def scratch_dir_for(name: str) -> Path:
    """The ``levels/_editor_scratch/<name>/`` folder for a level named
    ``name`` — the one place both the writer and the launcher compute this
    path, so they can never disagree."""
    return SCRATCH_ROOT / name


def scratch_level_arg(name: str) -> str:
    """The ``--level`` value the loader/main.py accept for this scratch
    level (editor doc §8: ``"_editor_scratch/<name>"``)."""
    return f"{SCRATCH_SUBDIR}/{name}"


def build_launch_argv(name: str, *, python: str | None = None) -> list:
    """``[sys.executable, "main.py", "--level", "_editor_scratch/<name>"]``.

    ``python=`` is a test-only override seam (asserting the DEFAULT is
    ``sys.executable`` is the gate's own explicit requirement — never bare
    ``python``, main.py's documented machine footgun); real callers never
    pass it.
    """
    exe = python if python is not None else sys.executable
    return [exe, str(ROOT / "main.py"), "--level", scratch_level_arg(name)]


def cleanup_scratch_dir(path) -> None:
    """Best-effort recursive delete of one scratch level dir.

    Called on BOTH the launched subprocess's exit and editor quit (C7's own
    dual-path requirement) — a file the OS still has open (a just-exited
    game process on Windows can hold a lock a beat longer) or an
    already-gone directory must never raise into the editor's main loop or
    its shutdown path."""
    shutil.rmtree(Path(path), ignore_errors=True)


def write_scratch_level(name, *, level_dir, grid, water_masked, zones, air,
                        level_boundary, spawns, lights, entities, wires,
                        light_form, bake_clean: bool, tileset_arg,
                        bake_ppt: int, bake_seed: int) -> Path:
    """Write a fresh, complete, loadable level to
    ``levels/_editor_scratch/<name>/`` from the LIVE editor state (unsaved
    edits included) and return the scratch dir.

    ``level_dir`` is the currently-open REAL level folder — its
    ``level.toml`` is copied as the base structure (name, tile_size_m,
    version, whatever [art]/[bake] blocks it already carries), then every
    managed family below is overwritten with the CURRENT session state, the
    same way Ctrl+S overwrites them in place: the result is byte-equivalent
    to what Ctrl+S would write to ``level_dir`` itself, just aimed at the
    scratch dir instead. ``water_masked`` is the caller's OWN
    ``mask_water_to_open(...)`` result — computed, never mutated back into
    the live grid, so F5 can apply the save-time wall-over-pool guard
    without perturbing the real undo timeline. ``zones``/``air`` are the
    live grids or ``None`` (dormant — never allocated this session).
    """
    dest = scratch_dir_for(name)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    # tilemap.csv — level_lib's own canonical grid writer (not
    # map_editor.save_tilemap_csv, which requires a PRE-EXISTING file to
    # read the prior newline style + .bak from; a fresh scratch dir has
    # neither, and level_lib.write_tilemap_csv already tolerates that).
    level_lib.write_tilemap_csv(dest, grid, csv_bak=False)

    # level.toml base structure: copy the REAL level's file, then every
    # managed-block write below targets THIS copy.
    scratch_toml = dest / "level.toml"
    shutil.copy2(Path(level_dir) / "level.toml", scratch_toml)

    # zones.npy / air_init.npy carry no managed toml family — the loader
    # discovers them by FILE PRESENCE alone (A8/A9), so nothing else is
    # needed once written.
    _, has_water = level_lib.write_water_npy(dest, water_masked, npy_bak=False)
    if zones is not None:
        level_lib.write_zones_npy(dest, zones, npy_bak=False)
    if air is not None:
        level_lib.write_air_init_npy(dest, air, npy_bak=False)

    # `boundary` is a bare top-level scalar (level_lib.write_boundary_field,
    # C5) — always written to the CURRENT live value; unlike Ctrl+S (which
    # skips a no-op rewrite to avoid a needless .bak), the scratch copy has
    # no .bak concern, so there is no reason to special-case "unchanged".
    level_lib.write_boundary_field(scratch_toml, level_boundary,
                                   write_bak=False)

    # spawn / water / wire / (light|entity) — the SAME replacements dict
    # shape Ctrl+S builds, reusing the SAME formatting functions.
    replacements = {
        "spawn": lambda nl: level_lib.format_spawn_lines(spawns, nl),
        "water": level_lib.water_block_format(has_water),
        "wire": lambda nl: level_lib.format_wire_lines(wires, nl),
    }
    replacements.update(
        light_entity_port.light_and_entity_replacements(
            light_form, lights, entities))

    # Bake: reuse the real level's on-disk PNGs when the material grid
    # hasn't changed since they were last written (bake_clean, the caller's
    # `not log.dirty`) — the copied level.toml already carries the correct
    # [art.bare]/[art.align]/[bake] blocks for those PNGs, so nothing more
    # to fold in. Otherwise bake fresh from the tilemap.csv just written
    # above (a dirty grid must never ship a stale bake): compute_bake_
    # replacements writes the PNGs and hands back its TOML replacements,
    # which fold into the SAME write_managed_blocks call below — Arc C9
    # rider, matching Ctrl+S's own one-shot atomic save method-for-method.
    src_diffuse = Path(level_dir) / DIFFUSE_FILENAME
    src_normal = Path(level_dir) / NORMAL_FILENAME
    if bake_clean and src_diffuse.is_file() and src_normal.is_file():
        shutil.copy2(src_diffuse, dest / DIFFUSE_FILENAME)
        shutil.copy2(src_normal, dest / NORMAL_FILENAME)
    else:
        art_replacements, _ = compute_bake_replacements(
            dest, tileset=tileset_arg, px_per_tile=bake_ppt,
            seed=bake_seed)
        replacements.update(art_replacements)

    level_lib.write_managed_blocks(scratch_toml, replacements, write_bak=False)

    return dest
