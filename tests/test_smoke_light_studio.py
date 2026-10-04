"""The smoke-light studio (#12 fire + smoke session, step 3; 2026-10-04): the
still-picture level for tuning smoke against light, its ``--warp`` launch flag,
and the e2e driver's ``--shot`` frame capture.

Only the level's PURPOSE is asserted -- the explosion studio's geometry, and
charges that never repeat. Light placement, colours, intensities and charge
timing are Erik's to retune and no test reads them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402

LEVELS = str(ROOT / "levels")


def test_studio_has_the_explosion_studios_geometry_and_no_recurring_charge():
    """PROPERTY: the smoke-light studio is the explosion studio's exact room
    geometry (Erik's ask: the same rooms, without the recurring explosions),
    and every timed charge in it fires ONCE (`period_s == 0`), so the smoke
    a freeze shows is never a later blast's.

    BREAKS IF: the two levels' tilemaps drift apart (a copied geometry edited
    on one side) or a charge is given a repeat period.
    """
    still = level_loader.load("smoke_light_studio", levels_dir=LEVELS)
    studio = level_loader.load("explosion_studio", levels_dir=LEVELS)
    assert np.array_equal(np.asarray(still.tilemap), np.asarray(studio.tilemap))
    charges = [e for e in still.entities if e.class_name == "timed_charge"]
    assert charges, "the studio has no charge to make smoke"
    for e in charges:
        assert float(e.fields.get("period_s", 0.0)) == 0.0, (
            f"{e.id} repeats every {e.fields['period_s']} s")


def test_warp_flag_parses_seconds_and_refuses_nonsense(monkeypatch):
    """PROPERTY: `--warp SECONDS` reads a finite, non-negative sim time and is
    0 when absent; anything else stops the launch with a message.

    BREAKS IF: the flag is misread, or a negative / missing / non-numeric
    value launches anyway.
    """
    from main import _parse_warp_flag  # heavy import: test-local
    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert _parse_warp_flag() == 0.0
    monkeypatch.setattr(sys, "argv", ["main.py", "--warp", "3.5"])
    assert _parse_warp_flag() == 3.5
    for bad in (["--warp"], ["--warp", "-1"], ["--warp", "soon"],
                ["--warp", "inf"], ["--warp", "nan"]):
        monkeypatch.setattr(sys, "argv", ["main.py"] + bad)
        with pytest.raises(SystemExit):
            _parse_warp_flag()


def test_e2e_shot_flag_is_consumed_not_passed_to_main():
    """PROPERTY: `tools/e2e_drive.py --shot PATH@FRAME` is the driver's own
    flag (repeatable, a Windows drive colon in PATH allowed) and never reaches
    main.py's argv.

    BREAKS IF: --shot leaks through to main.py, loses a path, or splits a
    `C:/...` path at its drive colon.
    """
    import e2e_drive
    frames, presses, shots, passthrough = e2e_drive.parse_args(
        ["--level", "x", "--shot", "C:/tmp/a.png@5", "--shot", "b.png@5",
         "--frames", "9"])
    assert frames == 9 and presses == {}
    assert shots == {5: ["C:/tmp/a.png", "b.png"]}
    assert "--shot" not in passthrough
    assert passthrough[:2] == ["--level", "x"]
