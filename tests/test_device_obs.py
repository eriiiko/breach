"""#29 P8.0 -- the device observation's host side (``simulation.device_obs``):
its generated header, its served shapes and the column-to-patch table
(``tests/obs_column_owners.toml``). CPU only; the card's gates are
``tests/test_cuda_obs.py``.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import device_obs  # noqa: E402

OWNERS = ROOT / "tests" / "obs_column_owners.toml"
#: The reward-side keys a step serves (``rl_env.Layout.shapes``), owned by the table.
REWARD_KEYS = ("reward", "counters", "team_balance", "team_uncredited",
               "team_outcome", "rows_ended", "terminated", "truncated")


def _table():
    return tomllib.loads(OWNERS.read_text(encoding="utf-8"))


def _v1_layout():
    import rl_env
    return rl_env.Layout(n_envs=2, n_teams=2, n_slots=3, n_actions=5, channels=(),
                         crop_radius=0, vector=(), spec="v1", full_view=True,
                         action_spec="v1", n_kinds=6, reward_profile="fireteam")


def _served_columns() -> dict:
    """Every served key -> its column / plane names from the spec's own
    tables (None: a key without named columns)."""
    from simulation import obs_spec_v1 as o
    lay = _v1_layout()
    keys = list(lay.obs_shapes()) + list(lay.full_shapes()) + list(REWARD_KEYS)
    named = {"vector": tuple(f.name for f in o.VECTOR_COLUMNS),
             "list": tuple(f.name for f in o.ENTRY_COLUMNS),
             "near_f": o.NEAR_F_PLANES, "near_u": o.NEAR_U_PLANES,
             "far_f": o.FAR_F_PLANES, "far_u": o.FAR_U_PLANES}
    return {k: named.get(k) for k in keys}


def test_the_generated_header_is_current():
    """Property: ``cpp/src/obs_layout.h`` is exactly what
    ``device_obs.header_text`` generates from the spec's tables. Breaks when
    a spec constant the kernels read (the window, a plane's index, a fill, a
    door code) changes without ``tools/gen_obs_layout.py --write``."""
    text = (ROOT / device_obs.HEADER_REL).read_text(encoding="utf-8")
    assert text == device_obs.header_text()
    assert device_obs.fingerprint() in text


def test_the_wall_share_bit_patterns_are_numpys_float16():
    """Property: the generated float16 table is numpy's direct conversion of
    k / BLOCK^2 (what ``build_windows`` writes into a float16 buffer).
    Breaks when the table is built by another rounding."""
    import numpy as np
    from simulation import obs_spec_v1 as o
    n = o.BLOCK * o.BLOCK
    got = device_obs._share_halves()
    want = [int(np.asarray(np.float64(k) / n).astype(np.float16).view(np.uint16))
            for k in range(n + 1)]
    assert got == want


def test_the_device_buffers_are_the_served_keys_shapes_and_dtypes():
    """Property: the device observation's buffers have exactly the keys,
    shapes and dtypes the env serves for a v1 layout (the observation, the
    row flags, the masks, the full view). Breaks when the env serves a key
    the device has no buffer for, or a shape / dtype drifts."""
    lay = _v1_layout()
    want = dict(lay.obs_shapes())
    want.update(lay.full_shapes())
    got = device_obs.served_shapes(lay.n_envs, lay.n_teams, lay.n_slots, lay.n_kinds)
    assert set(got) == set(want)
    for k in want:
        assert tuple(got[k][0]) == tuple(want[k][0]), k
        assert got[k][1] == want[k][1], k


def test_every_served_column_has_exactly_one_owner():
    """Property: every column / plane of the observation, the full view,
    the masks and the reward is owned by exactly one patch of the version-2
    table, named either per column or as a whole key (never both), and the
    table names nothing the env does not serve. Breaks when a column is
    added to the spec without an owner, owned twice, or owned by a patch
    that is not in the table."""
    t = _table()
    patches = t["patches"]
    assert len(set(patches)) == len(patches)
    served = _served_columns()
    whole, cols = t["keys"], t["columns"]
    for k, names in served.items():
        assert (k in whole) != (k in cols), f"{k}: owned per column XOR as a whole key"
        if k in whole:
            assert whole[k] in patches, k
            continue
        assert names is not None, f"{k} has no named columns in the spec"
        assert sorted(cols[k]) == sorted(names), k
        assert all(v in patches for v in cols[k].values()), k
    assert set(whole) | set(cols) == set(served)
    assert all(v in patches for v in t["final"].values())


def test_built_is_what_the_landed_patches_own():
    """Property: ``device_obs.BUILT`` (what the device blocks build) is
    exactly what the table gives the patches up to the shared check's
    ``PATCH``. Breaks when a block lands without its owner rows, or the
    table moves an owner without its block."""
    src = (ROOT / "tests" / "cuda_obs_check.py").read_text(encoding="utf-8")
    patch = re.search(r'^PATCH = "([^"]+)"', src, re.M).group(1)
    t = _table()
    upto = set(t["patches"][:t["patches"].index(patch) + 1])
    want = {k: None for k, v in t["keys"].items() if v in upto}
    for k, cols in t["columns"].items():
        names = tuple(n for n, v in cols.items() if v in upto)
        if names:
            want[k] = names
    got = {k: (None if v is None else tuple(v)) for k, v in device_obs.BUILT.items()}
    assert got.keys() == want.keys()
    for k in got:
        assert (got[k] is None) == (want[k] is None), k
        if got[k] is not None:
            assert sorted(got[k]) == sorted(want[k]), k
