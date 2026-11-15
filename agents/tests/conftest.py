"""The agents' own tests: ``pytest agents/tests -q`` from the repo root.

NOT part of the game's suite (``pyproject.toml`` pins ``testpaths =
["tests"]``), so the game's suite never imports PyTorch. They run on the
processor at the reduced width, in seconds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch.set_num_threads(4)


@pytest.fixture
def rng():
    return np.random.default_rng(20261115)


@pytest.fixture(scope="session")
def sizes6():
    from agents.sizes import sizes_for
    return sizes_for(6)
