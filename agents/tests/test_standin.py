"""The stand-in environment (agents/standin.py) behaves as rl_env.BatchedEnv
does: note 2b, 8; design 8; critique lens 2 major 6. Each test names the
property and the change that must break it."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import rl_env
from agents import network
from agents.standin import ATTRIBUTES, INFO_KEYS, StandIn
from agents.tasks import networks
from agents.tests._scripted import ScriptTask, mark
from simulation import action_spec_v1 as A

N, MATCH = 2, 3


def _env():
    task = ScriptTask(MATCH)
    env = StandIn(task, N, seed=3)
    task.bind(env.sizes)
    return env, task


def _default(env, obs):
    from agents.collector import default_actions
    return default_actions({k: obs[k] for k in A.MASK_KEYS}, len(A.HEADS))


def test_every_name_the_trainer_reads_is_the_games():
    """Protects: the stand-in serves the trainer exactly the names the
    game's BatchedEnv serves -- every ``info`` key is written by
    ``BatchedEnv.step``, ``reset`` or ``_info_arrays``, every attribute
    and method exists on BatchedEnv -- and its ``run`` and ``layout`` are
    the game's own records.
    Breaks when: the game renames an info key or a method (the stand-in
    would then test a trainer that cannot meet the farm), or the stand-in
    invents a name."""
    src = (inspect.getsource(rl_env.BatchedEnv.step)
           + inspect.getsource(rl_env.BatchedEnv.reset)
           + inspect.getsource(rl_env.BatchedEnv._info_arrays))
    for k in INFO_KEYS:
        assert f'info["{k}"]' in src or f"\"{k}\": " in src, k
    init = inspect.getsource(rl_env.BatchedEnv.__init__)
    for a in ATTRIBUTES:
        assert hasattr(rl_env.BatchedEnv, a) or f"self.{a} =" in init, a
    env, _ = _env()
    assert isinstance(env.run, rl_env.RunConfig)
    assert isinstance(env.layout, rl_env.Layout)
    for a in ATTRIBUTES:
        assert hasattr(env, a), a


def test_the_arrays_have_the_games_names_formats_and_shapes(sizes6):
    """Protects: every served array (observation, masks, flags, the full
    view, their ``final_`` twins, the reward, ``rows_ended``,
    ``masked_choices``) has the name, number format and shape of the game's
    own shape tables (``rl_env.Layout.shapes``), and the values lie in the
    game's ranges: the real networks read them without a kind past its
    table.
    Breaks when: a key is missing or renamed, a format differs (float32
    windows), or a kind column leaves its table."""
    env, _ = _env()
    want = env.layout.shapes()
    obs, info = env.reset()
    for k, v in obs.items():
        assert (v.shape, v.dtype) == (want[k][0], np.dtype(want[k][1])), k
    for k, v in info["full_view"].items():
        assert (v.shape, v.dtype) == (want[k][0], np.dtype(want[k][1])), k
    obs, reward, term, trunc, info = env.step(_default(env, obs))
    for k, v in info["final_obs"].items():
        assert v.shape == want["final_" + k][0], k
    assert reward.shape == want["reward"][0] and reward.dtype == np.float32
    assert info["rows_ended"].shape == want["rows_ended"][0]
    assert info["masked_choices"].shape == want["rows_ended"][0]
    soldier, value = networks(env, 0)
    flat = {k: v.reshape((-1,) + v.shape[2:]) for k, v in obs.items()}
    fv = {k: v.reshape((-1,) + v.shape[2:])
          for k, v in info["full_view"].items()}
    with torch.no_grad():
        soldier(network.as_tensors({k: flat[k]
                                    for k in network.SOLDIER_KEYS}))
        value(network.as_tensors({k: flat[k]
                                  for k in network.VALUE_OBS_KEYS}),
              network.as_tensors({k: fv[k]
                                  for k in network.VALUE_FULL_KEYS}))


def test_the_arrays_are_the_same_on_every_call_overwritten_in_place():
    """Protects: the observation, the full view and the ``final`` arrays
    are the SAME arrays on every call, overwritten in place -- what makes a
    kept reference a fault the trainer must not have.
    Breaks when: the stand-in returns fresh arrays (a trainer keeping
    references would then pass every test and fail on the farm)."""
    env, task = _env()
    obs0, info0 = env.reset()
    held = obs0["vector"]
    m0 = float(held[0, 0, task.col])
    obs1, _, _, _, info1 = env.step(_default(env, obs0))
    obs2, _, _, _, info2 = env.step(_default(env, obs1))
    for k in obs0:
        assert obs0[k] is obs1[k] is obs2[k], k
    for k in info0["full_view"]:
        assert info0["full_view"][k] is info2["full_view"][k], k
    for k in info1["final_obs"]:
        assert info1["final_obs"][k] is info2["final_obs"][k], k
    assert float(held[0, 0, task.col]) != m0          # overwritten


def test_a_match_that_ends_restarts_inside_the_step():
    """Protects: a match that ends inside a step is restarted in the same
    call: ``ended`` says so, the returned view is the new match's first,
    the ended match's last view is under ``final_obs`` / ``final_full_view``
    (zeros on every other env), wiped the step after, and the step's
    rewards are the ended match's (``rows_ended`` on every row with open
    records); a match ends on its clock as an ending (``terminated``).
    Breaks when: the restart waits for a reset, the final view is the new
    match's, the final arrays are not wiped, or the clock is a truncation."""
    env, task = _env()
    obs, _ = env.reset()
    for k in range(MATCH):
        obs, reward, term, trunc, info = env.step(_default(env, obs))
    assert info["ended"].all() and term.all() and not trunc.any()
    assert obs["vector"][0, 0, task.col] == mark(1, 0, 0)
    assert info["final_obs"]["vector"][0, 0, task.col] == mark(0, MATCH, 0)
    assert info["final_full_view"]["full_match"][0, 0, 0] == mark(0, MATCH, 0)
    assert reward[0, 0, 1] == MATCH                   # own: decision + 1
    assert info["rows_ended"][0, [0, 1, 2, 3, 4]].all()
    assert not info["rows_ended"][0, 5]               # the empty place
    obs, _, _, _, info = env.step(_default(env, obs))
    assert not info["ended"].any()
    for v in list(info["final_obs"].values()) + list(
            info["final_full_view"].values()):
        assert not v.any()


def test_rewards_flags_and_masks_of_dead_empty_and_closed_rows():
    """Protects: the reward is ``(N, U, 5)`` in the game's group order,
    zero for an empty roster place and for a row whose records ended in an
    earlier step; every team's rows are observed and paid (the bots' too);
    a row with ``valid`` 0 is all zeros with the default masks.
    Breaks when: a closed or empty row is paid, a non-learning team is left
    unobserved, or a dead row keeps its observation."""
    env, task = _env()
    obs, _ = env.reset()
    obs, r0, _, _, info = env.step(_default(env, obs))
    assert r0.shape == (N, 6, 5)
    assert (r0[:, 5] == 0).all()                      # empty place
    assert (r0[:, 2:4, 1] != 0).all()                 # the bots' rows paid
    # env 0, row 1 died in step 0: valid 0, in the match, all zeros
    assert obs["valid"][0, 1] == 0 and obs["in_match"][0, 1] == 1
    assert not obs["vector"][0, 1].any() and not obs["near_f"][0, 1].any()
    for k in A.MASK_KEYS:
        assert obs[k][0, 1].sum() == 1, k
    obs, r1, _, _, info = env.step(_default(env, obs))
    assert info["rows_ended"][1, 4] and r1[1, 4, 1] == 2
    assert obs["in_match"][1, 4] == 1                 # still a roster place
    if MATCH > 2:
        obs, r2, _, _, _ = env.step(_default(env, obs))
        assert (r2[1, 4] == 0).all()                  # records closed


def test_actions_are_checked_and_masked_choices_counted_as_the_game_does():
    """Protects: an action outside its head's range is refused (the game's
    ``check_actions``), a wrong shape is refused, and a masked in-range
    choice is counted per row against the masks served with the
    observation it answers (the game's ``masked_choices``), on every row.
    Breaks when: the stand-in accepts what the game refuses, or counts
    against other masks."""
    env, task = _env()
    obs, _ = env.reset()
    a = _default(env, obs)
    bad = a.copy()
    bad[0, 0, 0] = 999
    with pytest.raises(ValueError, match="out of range"):
        env.step(bad)
    with pytest.raises(ValueError, match="integer"):
        env.step(a[:, :3])
    m = a.copy()
    m[1, 2, 3] = A.HAND_FLASHLIGHT
    _, _, _, _, info = env.step(m)
    want = np.zeros((N, 6), dtype=np.uint8)
    want[1, 2] = 1
    np.testing.assert_array_equal(info["masked_choices"], want)


def test_team_rows_are_views_of_one_teams_rows():
    """Protects: ``team_rows`` gives VIEWS of team t's rows of every key
    (as the game's), refusing a team outside the layout.
    Breaks when: it copies (a policy's writes would vanish) or slices the
    wrong block."""
    env, task = _env()
    obs, _ = env.reset()
    part = env.team_rows(obs, 2)
    assert part["vector"].shape[1] == 2
    assert np.shares_memory(part["vector"], obs["vector"])
    assert part["vector"][0, 0, task.col] == mark(0, 0, 4)
    with pytest.raises(ValueError):
        env.team_rows(obs, 3)


def test_the_clock_is_shown_as_the_game_shows_it():
    """Protects: the time left of a match is in the soldier's vector
    (``time_left``) and in the full view's match block, as the game serves
    it: 1 at a match's first decision, falling every decision, back to 1
    after the in-step restart -- so the stand-in's clock ending is one a
    value network can see coming (design 6.5).
    Breaks when: the stand-in ends matches on a clock it does not show (no
    value network can then be right at a match's end, and every grade near
    the end carries that error)."""
    from simulation import obs_spec_v1 as O
    env, task = _env()
    col = next(f.pos for f in O.VECTOR_COLUMNS if f.name == "time_left")
    fcol = next(f.pos for f in O.full_match_columns(env.layout.n_teams)
                if f.name == "time_left")
    obs, info = env.reset()
    seen = [float(obs["vector"][0, 0, col])]
    assert float(info["full_view"]["full_match"][0, 0, fcol]) == seen[0] == 1.0
    for _ in range(MATCH):
        obs, _, _, _, info = env.step(_default(env, obs))
        seen.append(float(obs["vector"][0, 0, col]))
    assert seen[:MATCH] == sorted(seen[:MATCH], reverse=True)
    assert len(set(seen[:MATCH])) == MATCH and seen[MATCH] == 1.0
