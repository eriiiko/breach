"""#29 P8.10a -- the doors of the batched environment the device backend
brought, on the CPU (no card needed): the ``[run] commander`` setting through
the one controller builder, the ``[run] checkpoint_every_ticks`` cadence,
the device backend's refusals that come before any card, the coverage
counters and the names of the observation parts the device has not built.
The device backend itself is gated in ``tests/cuda_env_check.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import gen_training_rooms as TR  # noqa: E402
import rl_env  # noqa: E402

ROOM = "range_door"


@pytest.fixture(scope="module")
def rooms(tmp_path_factory):
    out = tmp_path_factory.mktemp("env_doors_rooms")
    for d in TR.drafts():
        if d.name == ROOM:
            TR.write(d, out)
    return out


def _run(rooms, **run):
    raw = {"run": {"seed": 3, "n_envs": 1, "decimation": 4, "levels": [ROOM],
                   "policy": "policy", "levels_dir": str(rooms)},
           "obs": {"spec": "v1", "full_view": True},
           "reward": {"profile": "fireteam"}, "action": {"spec": "v1"}}
    raw["run"].update(run)
    return rl_env.parse_run(raw, "doors run")


def _runner(rooms, **run):
    cfg = _run(rooms, **run)
    levels = rl_env.load_levels(cfg)
    r = rl_env.EnvRunner(0, cfg, rl_env.make_layout(cfg, levels), levels, breach_physics=bp)
    return r, r.reset()


def test_the_commander_setting_goes_through_the_one_builder(rooms):
    """Property: ``[run] commander`` names who writes a policy team's
    orders, honoured by the one controller builder: the default ("bots")
    builds the bots' commander beside every policy team on a level with a
    room graph (today's behaviour, unchanged), "none" builds none, and an
    unknown value is refused by name at the run-file door. Breaks if the
    setting is ignored, changes the default, or is a per-backend special
    case outside ``create_match_controllers``."""
    from control_bots import BotCommander
    r, _ = _runner(rooms)
    assert sum(isinstance(c, BotCommander) for c in r.session.controllers) == 2
    r, _ = _runner(rooms, commander="none")
    assert not any(isinstance(c, BotCommander) for c in r.session.controllers)
    with pytest.raises(ValueError, match="commander"):
        _run(rooms, commander="humans")


def test_the_checkpoint_cadence_is_a_run_setting(rooms):
    """Property: ``[run] checkpoint_every_ticks`` sets the cadence of the
    env's checkpoints (default the replay cadence, as before); 0 takes none,
    not even tick 0's; a negative value is refused by name. Breaks if a run
    could not turn off a cost the device backend pays per checkpoint, or the
    default moved."""
    from config import CFG
    import match_replay
    r, cps = _runner(rooms)
    assert r.every == match_replay.replay_settings(CFG) and len(cps) == 1
    r, cps = _runner(rooms, checkpoint_every_ticks=0)
    assert cps == []
    acts = np.zeros((r.layout.n_rows, 5), dtype=np.int32)
    sh = dict(r.layout.obs_shapes())
    sh.update(r.layout.full_shapes())
    r.observe({k: np.zeros(s[1:], dtype=t) for k, (s, t) in sh.items()})
    assert r.step(acts)["checkpoints"] == []
    with pytest.raises(ValueError, match="checkpoint_every_ticks"):
        _run(rooms, checkpoint_every_ticks=-1)


@pytest.mark.parametrize("over,needle", [
    ({"run": {"commander": "bots"}}, "commander has no device form"),
    ({"run": {"policy": ["policy", "bots"]}}, "every marine team is 'policy'"),
    ({"reward": {"profile": "counters"}}, "'counters' reward profile"),
])
def test_the_device_backend_refuses_before_any_card(rooms, over, needle):
    """Property: ``BatchedEnv(run, "device")`` refuses by name, before
    importing CuPy or touching a card, a run it does not serve: the
    commander on (no device form yet), a team not on 'policy' (design
    §12.3), the "counters" profile. Breaks if such a run reached the device
    and played something other than the farm."""
    raw = {"run": {"seed": 3, "n_envs": 1, "decimation": 4, "levels": [ROOM],
                   "policy": "policy", "levels_dir": str(rooms), "commander": "none"},
           "obs": {"spec": "v1", "full_view": True},
           "reward": {"profile": "fireteam"}, "action": {"spec": "v1"}}
    for t, kv in over.items():
        raw[t].update(kv)
    cfg = rl_env.parse_run(raw, "refused run")
    import rl_env_device
    with pytest.raises(rl_env_device.DeviceBackendRefused, match=needle):
        rl_env.BatchedEnv(cfg, "device")


def test_the_coverage_counts_choices_legal_options_and_moved_counters():
    """Property: the coverage counters count, per head, the options CHOSEN
    on valid rows only (each head's counts sum to the valid rows), per mask
    the options LEGAL on valid rows, per counter the cells that moved, and
    the endings -- the same function on numpy and CuPy. Breaks if a dead
    row's decision is counted, or a counter is attributed to another."""
    from simulation.action_spec_v1 import MASK_KEYS
    lay = rl_env.Layout(n_envs=2, n_teams=2, n_slots=3, n_actions=5, channels=(),
                        crop_radius=0, vector=(), spec="v1", full_view=True,
                        action_spec="v1", n_kinds=6, reward_profile="fireteam")
    cov = rl_env.coverage_init(np, lay)
    rng = np.random.default_rng(0)
    obs = {k: (rng.random(s) < 0.5).astype(np.uint8) for k, (s, _t) in
           lay.obs_shapes().items() if k in MASK_KEYS}
    for k in MASK_KEYS:
        obs[k][..., 0] = 1
    obs["valid"] = np.array([[1, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 1]], dtype=np.uint8)
    acts = rl_env.random_legal_actions(rng, obs)
    rl_env.coverage_actions(np, cov, lay, acts, obs)
    for h in ("feet", "aim", "trigger", "hand", "arg"):
        assert cov[f"chosen_{h}"].sum() == obs["valid"].sum()
    v = obs["valid"].astype(bool)
    assert np.array_equal(cov["legal_mask_hand"], obs["mask_hand"][v].sum(axis=0))
    counters = np.zeros((2, 6, len(rl_env.COUNTER_NAMES)), dtype=np.float32)
    counters[1, 5, 3] = 2.0
    rl_env.coverage_step(np, cov, counters, np.array([1, 0], np.uint8),
                         np.array([0, 0], np.uint8))
    assert cov["counters_moved"].sum() == 1 and cov["counters_moved"][3] == 1
    assert int(cov["ended_terminated"]) == 1 and int(cov["steps"]) == 1


def test_the_unbuilt_parts_are_named_from_the_built_table():
    """Property: the observation parts the backend-equality gate names as
    NOT YET BUILT are exactly the spec's planes / columns absent from
    ``device_obs.BUILT`` (today the order, route and place columns of #29
    P8.6b part 1), each a real name of the spec. Breaks if the gate could
    silently skip a built column or compare an unbuilt one."""
    import rl_env_device as rd
    from simulation import device_obs as d, obs_spec_v1 as o
    un = rd.unbuilt_parts()
    every = {"near_f": o.NEAR_F_PLANES, "near_u": o.NEAR_U_PLANES, "far_f": o.FAR_F_PLANES,
             "far_u": o.FAR_U_PLANES, "vector": [f.name for f in o.VECTOR_COLUMNS],
             "list": [f.name for f in o.ENTRY_COLUMNS]}
    for k, names in every.items():
        want = tuple(n for n in names if n not in d.BUILT[k])
        assert un.get(k, ()) == want
