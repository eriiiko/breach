"""agents/batch.py -- the stored decision and the batch (rung 2 note 2b, 2;
design 4.2).

WHOSE DECISIONS ARE STORED. Only the rows of the LEARNING TEAMS: the teams
the run file puts on the ``policy`` controller. Their roster places, every
one at a fixed place: ``R = N x (learning teams) x S`` rows, row ``r = (env
* L + l) * S + s`` for env ``env``, the l-th learning team, roster slot
``s`` (:class:`LearningRows`). Every array of the batch is time first,
``(T, R, ...)``: that order is what patch 2d's stretches cut.

WHAT IS KEPT PER STORED DECISION (the note's table):

- the observation the decision answered, as served: ``vector``, ``list``,
  ``list_mask``, ``near_f``, ``near_u``, ``far_f``, ``far_u`` (``list_uid``
  is not kept: it is no input);
- the four masks as served (bytes): ``mask_feet``, ``mask_aim``,
  ``mask_trigger``, ``mask_hand``;
- ``arg_mask`` (bytes, 5): the argument's mask the draw worked out;
- the value network's view at the decision: ``full_list``, ``full_mask``,
  ``full_match``, ``full_watched``, as served;
- ``actions`` (int16, 5), ``used`` (bytes, 5: which heads counted),
  ``logp_heads`` (float32, 5) and ``logp`` (float32): the probability of
  each counted choice and of the whole decision;
- ``reward`` (float32, 5): the group numbers paid by the step that carried
  the decision out;
- ``valid`` and ``in_match`` (bytes): the two row flags served with the
  observation;
- ``done`` (bytes): the row's story ended in that step (its match ended, or
  its records ended: ``rows_ended``);
- ``first`` (bytes): the decision is the first of a match for this row;
- ``value_row`` (bytes): the step is a VALUE ROW -- ``in_match`` 1 and the
  row's story not ended since its match began (the collector's bookkeeping,
  carried across rounds; grading reads it);
- ``memory`` (float32, ``(T + 1, R, core)``, only with the soldier's memory
  on; patch 2d): the soldier's memory BEFORE its decision at each step --
  the state carried in, before the network zeroes it where ``first`` says a
  match starts -- and at index ``T`` the memory after the last step (what
  the next round starts from). Learning replays a stretch from
  ``memory[t0]`` with the stretch's ``first`` flags, and the live check
  compares the memory at the stretch's end with ``memory[t0 + stretch]``.

Once per batch, for the value after the last stored decision: the
observation keys the value network reads and the full view as they stand
after the last step, ``last_*`` ``(R, ...)``.

The POLICY ROWS of a step are those with ``valid`` 1 (a living soldier: the
soldier's network and the value network learn). The VALUE ROWS are those
with ``value_row`` 1 (a dead soldier is one: the value network learns what
the death was worth). An empty roster place and a row after its records
ended are in nothing.

Everything is COPIED in from the environment's arrays (they are overwritten
by its next step). Shapes and number formats come from the environment's
layout (the game's own shape tables), never a literal.
"""
from __future__ import annotations

import numpy as np

from agents import _game  # noqa: F401
from agents.network import SOLDIER_KEYS, VALUE_FULL_KEYS, VALUE_OBS_KEYS
from simulation import action_spec_v1 as A
from simulation.reward_v1 import REWARD_GROUPS

#: The served observation keys a stored decision keeps (the soldier's input
#: and the masks).
OBS_STORED = SOLDIER_KEYS + A.MASK_KEYS
#: The served full-view keys a stored decision keeps (the value network's).
FULL_STORED = VALUE_FULL_KEYS
#: The keys kept once per batch for the view after the last step.
LAST_OBS = VALUE_OBS_KEYS
LAST_FULL = VALUE_FULL_KEYS


class LearningRows:
    """The learning teams' rows of an environment, in the batch's order:
    ``R = N x L x S``, row ``(env * L + l) * S + s``. Every read of the
    environment's ``(N, U, ...)`` arrays goes through ``env.team_rows``."""

    def __init__(self, env, teams):
        self.env = env
        self.teams = tuple(int(t) for t in teams)
        if not self.teams:
            raise ValueError("no learning team: the run file puts no team "
                             "on the 'policy' controller")
        self.n_envs = int(env.n_envs)
        self.n_slots = int(env.layout.n_slots)
        self.n_rows = self.n_envs * len(self.teams) * self.n_slots

    def gather(self, src: dict, dest: dict, keys) -> None:
        """Copy the learning rows of ``src[k]`` (``(N, U, ...)``) into
        ``dest[k]`` (``(R, ...)``, the caller's own array) for ``k`` in
        ``keys``."""
        S = self.n_slots
        for li, t in enumerate(self.teams):
            part = self.env.team_rows({k: src[k] for k in keys}, t)
            for k in keys:
                if not dest[k].flags.c_contiguous:
                    raise ValueError(f"{k}: the destination must be one "
                                     f"contiguous (R, ...) array")
                d = dest[k].reshape((self.n_envs, len(self.teams), S)
                                    + dest[k].shape[1:])
                np.copyto(d[:, li], part[k])

    def scatter(self, rows, full):
        """Write ``rows`` ``(R, ...)`` into the learning rows of ``full``
        ``(N, U, ...)`` (the actions sent)."""
        S = self.n_slots
        r = np.asarray(rows).reshape((self.n_envs, len(self.teams), S)
                                     + np.shape(rows)[1:])
        for li, t in enumerate(self.teams):
            full[:, t * S:(t + 1) * S] = r[:, li]

    def env_of_row(self):
        """``(R,)``: the env of every row."""
        return np.repeat(np.arange(self.n_envs), len(self.teams) * self.n_slots)

    def name(self, r: int) -> str:
        """A row's place, for an error: env, team, slot."""
        S, L = self.n_slots, len(self.teams)
        return (f"env {r // (L * S)}, team {self.teams[(r // S) % L]}, "
                f"slot {r % S}")


class Batch:
    """One round's stored decisions, pre-allocated once and reused every
    round (see the module text). ``obs`` and ``full`` hold the served
    arrays by key; the per-decision arrays are attributes; ``last_obs`` /
    ``last_full`` the view after the last step."""

    def __init__(self, layout, rows: LearningRows, decisions: int,
                 core: int | None = None):
        T, R = int(decisions), rows.n_rows
        if T < 1:
            raise ValueError(f"a batch of {T} decisions")
        self.T, self.R = T, R
        self.rows = rows
        self.groups = tuple(REWARD_GROUPS)
        obs_shapes = layout.obs_shapes()
        full_shapes = layout.full_shapes()

        def tail(shapes, k):
            shape, dt = shapes[k]
            return tuple(shape[2:]), np.dtype(dt)

        self.obs = {}
        for k in OBS_STORED:
            sh, dt = tail(obs_shapes, k)
            self.obs[k] = np.zeros((T, R) + sh, dtype=dt)
        self.full = {}
        for k in FULL_STORED:
            sh, dt = tail(full_shapes, k)
            self.full[k] = np.zeros((T, R) + sh, dtype=dt)
        self.last_obs = {k: np.zeros(self.obs[k].shape[1:], self.obs[k].dtype)
                         for k in LAST_OBS}
        self.last_full = {k: np.zeros(self.full[k].shape[1:],
                                      self.full[k].dtype) for k in LAST_FULL}
        n_heads, n_groups = len(A.HEADS), len(REWARD_GROUPS)
        self.arg_mask = np.zeros((T, R, A.N_ARG), dtype=np.uint8)
        self.actions = np.zeros((T, R, n_heads), dtype=np.int16)
        self.used = np.zeros((T, R, n_heads), dtype=np.uint8)
        self.logp_heads = np.zeros((T, R, n_heads), dtype=np.float32)
        self.logp = np.zeros((T, R), dtype=np.float32)
        self.reward = np.zeros((T, R, n_groups), dtype=np.float32)
        self.valid = np.zeros((T, R), dtype=np.uint8)
        self.in_match = np.zeros((T, R), dtype=np.uint8)
        self.done = np.zeros((T, R), dtype=np.uint8)
        self.first = np.zeros((T, R), dtype=np.uint8)
        self.value_row = np.zeros((T, R), dtype=np.uint8)
        #: the memory before each decision and after the last (None: the
        #: soldier has no memory, nothing is allocated)
        self.memory = (None if core is None else
                       np.zeros((T + 1, R, int(core)), dtype=np.float32))

    # -- the per-decision arrays by name (for the learner's flat views) ---
    PER_DECISION = ("arg_mask", "actions", "used", "logp_heads", "logp",
                    "reward", "valid", "in_match", "done", "first",
                    "value_row")

    def flat(self, name: str):
        """A per-decision array or a stored key as ``(T * R, ...)`` (a view
        of the batch's own array)."""
        if name in self.obs:
            a = self.obs[name]
        elif name in self.full:
            a = self.full[name]
        else:
            a = getattr(self, name)
        return a.reshape((self.T * self.R,) + a.shape[2:])

    def nbytes(self) -> int:
        arrays = (list(self.obs.values()) + list(self.full.values())
                  + list(self.last_obs.values())
                  + list(self.last_full.values())
                  + [getattr(self, n) for n in self.PER_DECISION]
                  + ([] if self.memory is None else [self.memory]))
        return int(sum(a.nbytes for a in arrays))


__all__ = ["OBS_STORED", "FULL_STORED", "LAST_OBS", "LAST_FULL",
           "LearningRows", "Batch"]
