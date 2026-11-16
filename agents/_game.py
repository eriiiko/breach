"""agents/_game.py -- the one place ``agents`` puts the game's Python on the
path (the repo root and ``src/``, the way ``tools/`` find it). Every module
of the package that reads the game imports this first."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
