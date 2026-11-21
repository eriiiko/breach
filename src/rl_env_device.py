"""THE DEVICE BACKEND of the batched environment (#29 P8.10a, residency
design §4.4c "P8 RE-CUT, VERSION 2"): ``rl_env.BatchedEnv(run,
backend="device")``, the same API, keys, shapes and dtypes as ``cpu_farm``
(``rl_env``'s module doc; inventory H), over ONE
:class:`simulation.conductor.DeviceConductor` of N envs with the device
observation, team picture, books and decoder attached. ``cpu_farm`` stays THE
reference: the backend-equality gate (``tests/cuda_env_check.py``) and the
pre-flight (:func:`preflight`) hold this backend equal to it, step by step.

Imported only by ``BatchedEnv`` on ``backend="device"`` (a farm process never
imports CuPy or the CUDA build).

WHAT A STEP RETURNS (and why). Everything the card builds stays on the card
as a TORCH tensor (the trainer's type; DLPack, no copy):

- ``obs`` and ``info["full_view"]``: torch views of the device observation's
  OWN buffers (``DeviceObservation.buffers``), made once at the build, the
  same tensors every step (the farm's env-owned rule: valid until the next
  step / reset; a caller that keeps one copies it);
- ``info["final_obs"]`` / ``info["final_full_view"]``: torch views of this
  backend's own ``final_`` buffers, the envs that ended in the step holding
  their terminal observation (``DeviceObservation.final``, built at the
  ending tick, P8.8b), every other row zeros (the farm's per-step semantics);
- ``reward``, ``terminated``, ``truncated``, ``info["counters"]``,
  ``["team_balance"]``, ``["team_uncredited"]``, ``["team_outcome"]``,
  ``["rows_ended"]``, ``["masked_choices"]``: FRESH device copies (the farm's
  "every other array is a fresh copy"), taken before any reset re-seeds the
  books; ``terminated`` / ``truncated`` / ``rows_ended`` as bool (one cast
  kernel each);
- HOST numpy (the host holds them anyway, no transfer): ``info["tick"]``,
  ``["episode"]``, ``["episode_seed"]``, ``["level"]``, ``["ticks"]``,
  ``["ended"]`` (the conductor's books read already carries each env's
  ended-now flag), ``["checkpoints"]`` and ``["result"]`` (one download of
  the books' scalars, on a step where some env ended only).

A steady step (no env ends, no checkpoint due) makes exactly the conductor's
2 host syncs per tick and no other transfer: the decision is one device write
(``DeviceDecoder.set_decisions``), every copy above a device kernel.

EPISODES (decision 4, synchronous form): env e's episode k is
``rl_env.episode_match(run, levels, e, k)`` (the farm's own draw and spec),
built on the host as ``MatchSession(level, spec, seed, commander="none")``
and handed to ``DeviceConductor.reset_env`` at the step it ended, so the
returned live observation is episode k + 1's tick 0, as on the farm. The
first episode of every env is built with the backend.

WHO DRIVES (design §12.3, P8.6a's refusals): every marine team ``policy`` (the
device decoder), the horde ``zombies`` or ``idle``; ``[run] commander`` must
be ``"none"`` (the commander has no device form yet, #29 P8.6b part 2); a
named reward profile (``"counters"`` refused: numpy's pairwise fold); ``[obs]
spec`` and ``[action] spec`` ``"v1"``. A level set that cannot share one batch
(:func:`batch_signature`) is refused at the build, naming what differs.

FAULTS: a conductor refusal (``ConductorFault``, ``RewardBooksMismatch``, an
out-of-range decision as the host's ``ValueError`` with ``check_actions``'
message, ...) is raised by name with ``.env`` set, from the step that met it;
the batch is then partly written in every env, so this backend refuses every
later call (:class:`DeviceEnvFailed` names the first fault).

CHECKPOINTS: ``(episode, tick, field_digest, tick_digest)`` at the run's
cadence (``rl_env.checkpoint_every``), tick 0 and an episode's last tick, as
the farm: each is a ``DeviceConductor.download`` of the envs due that tick and
the host digest of the landed twin (``match_replay.match_digests``).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import rl_env

ROOT = Path(__file__).resolve().parent.parent
#: The controllers the device serves for the horde (the decoder's forms).
DEVICE_HORDE = ("zombies", "idle")
#: The one commander the device backend takes (P8.6b part 2 builds the rest).
DEVICE_COMMANDER = "none"
#: THE BREAK hooks of the backend-equality gate (``DeviceEnv.brk``), each a
#: deliberate defect the gate must turn red; "" in a real run.
BREAKS = ("reset_late", "final_kept", "episode_env", "decide_late")


class DeviceBackendRefused(ValueError):
    """A run the device backend does not serve (named at the build)."""


class DeviceEnvFailed(RuntimeError):
    """A call after a fault refused the batch (names the first fault)."""


class PreflightFailed(RuntimeError):
    """The pre-flight found the two backends apart (names the step, env,
    key and index), or a run feature it never exercised."""


# ---------------------------------------------------------------------------
# The refusals at the build
# ---------------------------------------------------------------------------
def check_run(run, levels) -> None:
    """Refuse, by name, a run the device backend does not serve."""
    where = "device backend"
    if run.spec != "v1" or run.action_spec != "v1":
        raise DeviceBackendRefused(
            f"{where}: [obs] spec {run.spec!r} / [action] spec {run.action_spec!r}: the "
            f"device serves observation spec v1 and action spec v1 only")
    if run.reward_profile == "counters":
        raise DeviceBackendRefused(
            f"{where}: the 'counters' reward profile is refused on the device backend (its "
            f"float32 fold is numpy's pairwise sum, which a serial kernel cannot equal); name "
            f"a profile of reward_v1.PROFILES")
    if run.commander != DEVICE_COMMANDER:
        raise DeviceBackendRefused(
            f"{where}: [run] commander {run.commander!r}: the commander has no device form "
            f"yet (#29 P8.6b part 2); a device run sets [run] commander = \"none\"")
    for lv in levels:
        ctrls = run.team_controllers(len(lv.match.teams), repr(lv.name))
        for t, c in enumerate(ctrls):
            if c != "policy":
                raise DeviceBackendRefused(
                    f"{where}: level {lv.name!r} team {t} is driven by {c!r}: on the device "
                    f"backend every marine team is 'policy' (design §12.3: self-play or "
                    f"against the horde)")
        z = getattr(lv.match, "zombie_controller", "zombies")
        if z not in DEVICE_HORDE:
            raise DeviceBackendRefused(
                f"{where}: level {lv.name!r}'s horde controller {z!r} has no device form "
                f"({' or '.join(DEVICE_HORDE)})")


def batch_signature(sim) -> dict:
    """What one batch of the device conductor holds equal across its envs
    (the resident batch's grid, gas table, tilt and tick rate; the store's U;
    the observation's per-batch capacities: doors, covers, places; the
    layout's teams), read from a level's tick-0 sim."""
    from simulation import device_obs
    g = sim.gmap
    return {"grid": (int(g._h), int(g._w)),
            "gas table": tuple(g.gases.name_to_id.items()) + (tuple(g.gas.shape),),
            "tilt": (g.tilt_x, g.tilt_y), "tick rate": sim._tps,
            "units (U)": len(sim.units),
            "doors": len(getattr(sim, "_doors", ()) or ()),
            "covers": len(device_obs.cover_values(sim)),
            "places": len(device_obs.place_values(sim))}


def check_homogeneous(names, sims) -> None:
    """Refuse a level set whose tick-0 sims cannot share one batch, listing
    every field that differs (each level's value)."""
    sigs = [batch_signature(s) for s in sims]
    diff = [k for k in sigs[0] if len({repr(s[k]) for s in sigs}) > 1]
    if diff:
        parts = "; ".join(f"{k}: " + ", ".join(f"{n}={s[k]!r}" for n, s in zip(names, sigs))
                          for k in diff)
        raise DeviceBackendRefused(
            f"device backend: the run's levels cannot share one batch (one grid, gas table, "
            f"tick rate, U and table capacities): {parts}")


def _cuda_bp():
    import breach_physics as bp
    if not getattr(bp, "HAS_CUDA", False):
        raise DeviceBackendRefused(
            f"device backend: breach_physics from {bp.__file__} is not the CUDA build "
            f"(cpp/build_cuda): the device backend runs in a process that imported it")
    return bp


def _tests_on_path() -> None:
    p = str(ROOT / "tests")           # match_replay.match_digests reads the gates' snapshot
    if p not in sys.path:
        sys.path.append(p)


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------
class DeviceEnv:
    """The device backend of one ``BatchedEnv`` (see the module doc)."""

    def __init__(self, run, layout, levels, bp=None):
        import cupy as cp
        import torch
        from simulation.conductor import DeviceConductor
        from simulation.device_decoder import DeviceDecoder
        from simulation.device_obs import DeviceObservation
        check_run(run, levels)
        _tests_on_path()
        self.cp, self.torch = cp, torch
        self.bp = _cuda_bp() if bp is None else bp
        self.run, self.layout, self.levels = run, layout, list(levels)
        self.N = N = run.n_envs
        self.every = rl_env.checkpoint_every(run)
        self.brk = ""
        self.timing = {}                # host wall seconds per part of a step
        self.profile = False            # the bench's: each tick's GPU time per call line
        self.prof = {}                  # call line -> GPU ms summed (profile on)
        self._failed = None
        if len(self.levels) > 1:
            probes = [self._session(lv, lv.match.seed, self._policy_spec(lv)) for lv in
                      self.levels]
            check_homogeneous([lv.name for lv in self.levels], [s.sim for s in probes])
            del probes
        # -- every env's episode 0, then the conductor and the observation ----
        self.k = [-1] * N
        self.seed = [0] * N
        self.level_index = [0] * N
        self.sessions = [None] * N
        self._cps0 = [[] for _ in range(N)]
        for e in range(N):
            self._cps0[e] = self._take(e, self._build(e))
        self.c = DeviceConductor([s.sim for s in self.sessions], bp=self.bp)
        self.obs = DeviceObservation(self.c, [s.spec for s in self.sessions], layout.n_slots,
                                     run.decimation, n_teams=layout.n_teams,
                                     reward_profile=run.reward_profile,
                                     reward_overrides=dict(run.reward_terms))
        self.dec = DeviceDecoder(self.obs, [s.spec for s in self.sessions])
        if tuple(self.dec.sizes) != tuple(layout.head_sizes):
            raise DeviceBackendRefused(f"device backend: the decoder's head sizes "
                                       f"{self.dec.sizes} are not the layout's "
                                       f"{layout.head_sizes}")
        self.obs_keys = layout.obs_keys
        self.full_keys = layout.full_keys
        tv = self.obs.torch()
        self._obs_t = {k: tv[k] for k in self.obs_keys}
        self._full_t = {k: tv[k] for k in self.full_keys}
        keys = self.obs_keys + self.full_keys
        self._final = {k: cp.zeros_like(self.obs.final[k]) for k in keys}
        self._final_t = {k: torch.from_dlpack(v) for k, v in self._final.items()}
        self._final_rows = np.zeros(N, dtype=bool)
        self._pending_reset = np.zeros(N, dtype=bool)     # the reset_late break only
        self.started = np.zeros(N, dtype=bool)
        self.cov = rl_env.coverage_init(cp, layout)

    # -- episodes -------------------------------------------------------------
    def _policy_spec(self, lv):
        from dataclasses import replace
        ctrls = self.run.team_controllers(len(lv.match.teams), lv.name)
        return replace(lv.match, teams=tuple(replace(t, controller=c)
                                             for t, c in zip(lv.match.teams, ctrls)))

    def _session(self, level, seed, spec):
        from simulation.match_session import MatchSession
        return MatchSession(level, spec, seed=seed, breach_physics=self.bp,
                            commander=self.run.commander)

    def _build(self, e: int) -> tuple:
        """Env ``e``'s NEXT episode on the host: ``(k, seed, level index,
        session, tick-0 checkpoints)`` (the farm's ``EnvRunner.reset``)."""
        t0 = time.perf_counter()
        k = self.k[e] + 1
        draw_env = (e + 1) % self.N if self.brk == "episode_env" else e   # THE BREAK
        seed, li, level, spec = rl_env.episode_match(self.run, self.levels, draw_env, k)
        s = self._session(level, seed, spec)
        cps = [(k, 0, *s.digests())] if self.every else []
        self._time("reset: host build", t0)
        return k, seed, li, s, cps

    def _take(self, e: int, built) -> list:
        k, seed, li, s, cps = built
        self.k[e], self.seed[e], self.level_index[e], self.sessions[e] = k, seed, li, s
        return cps

    def _reset_one(self, e: int) -> list:
        built = self._build(e)
        t0 = time.perf_counter()
        self.c.reset_env(e, built[3].sim, built[3].spec)
        self._time("reset: restage (reset_env)", t0)
        return self._take(e, built)

    def _time(self, key: str, t0: float) -> None:
        self.timing[key] = self.timing.get(key, 0.0) + (time.perf_counter() - t0)

    def _check_ok(self) -> None:
        if self._failed is not None:
            raise DeviceEnvFailed(f"device backend: an earlier step refused the batch "
                                  f"({self._failed}); build a new BatchedEnv")

    # -- the info arrays ------------------------------------------------------
    def _info_arrays(self) -> dict:
        return {"tick": np.array([int(s.tick) for s in self.c.sims], dtype=np.int64),
                "episode": np.array(self.k, dtype=np.int64),
                "episode_seed": np.array(self.seed, dtype=np.uint64),
                "level": np.array(self.level_index, dtype=np.int64)}

    # -- the API --------------------------------------------------------------
    def reset(self, mask) -> tuple:
        self._check_ok()
        cps = [[] for _ in range(self.N)]
        try:
            for e in np.flatnonzero(mask):
                e = int(e)
                if not self.started[e]:          # episode 0, staged with the backend
                    cps[e] = self._cps0[e]
                    self._cps0[e] = []
                else:
                    cps[e] = self._reset_one(e)
        except BaseException as ex:
            self._failed = f"{type(ex).__name__}: {ex}"
            raise
        self.started |= np.asarray(mask, dtype=bool)
        info = self._info_arrays()
        info["checkpoints"] = cps
        info["full_view"] = dict(self._full_t)
        return dict(self._obs_t), info

    def _as_device(self, actions):
        cp = self.cp
        if isinstance(actions, cp.ndarray):
            return actions
        if isinstance(actions, np.ndarray):
            return cp.asarray(actions)
        if hasattr(actions, "__dlpack__"):
            return cp.from_dlpack(actions)
        return cp.asarray(np.asarray(actions))

    def step(self, actions) -> tuple:
        self._check_ok()
        try:
            return self._step(actions)
        except BaseException as ex:
            self._failed = f"{type(ex).__name__}: {ex}"
            raise

    def _step(self, actions) -> tuple:
        cp, torch, c, dec = self.cp, self.torch, self.c, self.dec
        t_step = time.perf_counter()
        N = self.N
        a = self._as_device(actions)
        want = (N, self.layout.n_rows, self.layout.n_actions)
        if tuple(a.shape) != want or not cp.issubdtype(a.dtype, cp.integer):
            raise ValueError(f"actions must be an integer {want} array, got "
                             f"{tuple(a.shape)} {a.dtype}")
        t0 = time.perf_counter()
        b = self.obs.buffers
        rl_env.coverage_actions(cp, self.cov, self.layout, a, b)
        self._time("coverage", t0)
        if self.brk != "decide_late":
            t0 = time.perf_counter()
            dec.set_decisions(a)                 # THE decision: one device write
            self._time("decisions (hand-over)", t0)
        ended = np.zeros(N, dtype=bool)
        active = np.ones(N, dtype=bool)
        ticks = np.zeros(N, dtype=np.int64)
        cps = [[] for _ in range(N)]
        for i in range(self.run.decimation):
            t0 = time.perf_counter()
            c.tick(profile=self.profile)
            self._time("conductor ticks", t0)
            if self.profile:
                for label, g, _h in c.last_profile:
                    if g is not None:
                        self.prof[label] = self.prof.get(label, 0.0) + float(g)
            if i == 0 and self.brk == "decide_late":   # THE BREAK: after the first tick
                dec.set_decisions(a)
            now = np.asarray(c.ended_now, dtype=bool) & active
            ticks[active] += 1
            if self.every:
                due = [e for e in np.flatnonzero(active)
                       if now[e] or int(c.sims[e].tick) % self.every == 0]
                if due:
                    t0 = time.perf_counter()
                    c.download([int(e) for e in due])
                    for e in due:
                        cps[e].append((self.k[e], int(c.sims[e].tick),
                                       *self.sessions[e].digests()))
                    self._time("checkpoints", t0)
            ended |= now
            active &= ~now
        # -- the served arrays: fresh device copies, before any reset ---------
        t0 = time.perf_counter()
        sv = self.obs.books.served
        reward = torch.from_dlpack(sv["reward"].copy())
        term = torch.from_dlpack(sv["terminated"].astype(cp.bool_))
        trunc = torch.from_dlpack(sv["truncated"].astype(cp.bool_))
        info = {}
        info["counters"] = torch.from_dlpack(sv["counters"].copy())
        for key in ("team_balance", "team_uncredited", "team_outcome"):
            info[key] = torch.from_dlpack(sv[key].copy())
        info["rows_ended"] = torch.from_dlpack(sv["rows_ended"].astype(cp.bool_))
        info["masked_choices"] = torch.from_dlpack(dec.masked.copy())
        rl_env.coverage_step(cp, self.cov, sv["counters"], sv["terminated"],
                             sv["truncated"])
        # -- the terminal observation: this step's ended rows, zeros elsewhere
        if self._final_rows.any() and self.brk != "final_kept":
            idx = cp.asarray(np.flatnonzero(self._final_rows))
            for v in self._final.values():
                v[idx] = 0
        if ended.any():
            idx = cp.asarray(np.flatnonzero(ended))
            for k, v in self._final.items():
                v[idx] = self.obs.final[k][idx]
        self._final_rows = ended.copy()
        self._time("served copies + final_", t0)
        result = [None] * N
        if ended.any():
            from simulation.device_books import OUTCOME_CODES, SCALAR_FIELDS
            t0 = time.perf_counter()
            sc = self.obs.books.scalars()            # one download, an ending step only
            names = {v: k for k, v in OUTCOME_CODES.items()}
            f = {n: i for i, n in enumerate(SCALAR_FIELDS)}
            for e in np.flatnonzero(ended):
                team = int(sc[e, f["RES_TEAM"]])
                result[e] = {"outcome": names[int(sc[e, f["RES_OUTCOME"]])],
                             "team": None if team < 0 else team,
                             "ticks": int(sc[e, f["RES_TICKS"]])}
            self._time("result (books scalars)", t0)
        # -- the auto-reset of the envs that ended --------------------------
        resets = ended
        if self.brk == "reset_late":                 # THE BREAK: one step late
            resets, self._pending_reset = self._pending_reset, ended.copy()
        for e in np.flatnonzero(resets):
            cps[e] = cps[e] + self._reset_one(int(e))
        info.update(self._info_arrays())
        info["ticks"] = ticks
        info["ended"] = ended
        info["final_obs"] = {k: self._final_t[k] for k in self.obs_keys}
        info["final_full_view"] = {k: self._final_t[k] for k in self.full_keys}
        info["full_view"] = dict(self._full_t)
        info["result"] = result
        info["checkpoints"] = cps
        self._time("step (total)", t_step)
        return dict(self._obs_t), reward, term, trunc, info

    def coverage(self) -> dict:
        """The coverage counters as host numpy (one download)."""
        return {k: v.get() for k, v in self.cov.items()}

    def close(self) -> None:
        for name in ("dec", "obs", "c", "sessions", "_final", "_final_t", "_obs_t",
                     "_full_t", "cov"):
            if hasattr(self, name):
                setattr(self, name, None)


# ---------------------------------------------------------------------------
# Backend equality: the comparison (the gate's and the pre-flight's)
# ---------------------------------------------------------------------------
def unbuilt_parts() -> dict:
    """``key -> names`` of the planes / columns no device patch has built
    yet (``device_obs.BUILT``; ``tests/obs_column_owners.toml``): served as
    zeros by the device, compared by NAME only (counted, never equal-gated)."""
    from simulation import device_obs as d, obs_spec_v1 as o
    names = {"near_f": o.NEAR_F_PLANES, "near_u": o.NEAR_U_PLANES, "far_f": o.FAR_F_PLANES,
             "far_u": o.FAR_U_PLANES, "vector": [f.name for f in o.VECTOR_COLUMNS],
             "list": [f.name for f in o.ENTRY_COLUMNS]}
    out = {}
    for k, ns in names.items():
        built = d.BUILT.get(k)
        if built is None:
            continue
        miss = tuple(n for n in ns if n not in built)
        if miss:
            out[k] = miss
    return out


def _index(key: str, name: str) -> int:
    from simulation import obs_spec_v1 as o
    planes = {"near_f": o.NEAR_F_PLANES, "near_u": o.NEAR_U_PLANES, "far_f": o.FAR_F_PLANES,
              "far_u": o.FAR_U_PLANES}
    if key in planes:
        return list(planes[key]).index(name)
    cols = {"vector": o.VECTOR_COLUMNS, "list": o.ENTRY_COLUMNS}[key]
    return next(f.pos for f in cols if f.name == name)


def host(x):
    """A returned array as numpy (a torch / CuPy array downloaded: the
    gate's read, a sync)."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get") and not isinstance(x, np.ndarray):
        return x.get()
    return np.asarray(x)


def _raw(a) -> np.ndarray:
    a = np.ascontiguousarray(a)
    if a.dtype == bool:
        return a.view(np.uint8)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def array_difference(name: str, ref, got, unbuilt=(), key=None, stats=None):
    """The first raw-bit difference of ``got`` against ``ref`` (dtype and
    shape included), or None; the planes / columns ``unbuilt`` (of an
    observation ``key``) are cut out and their differing cells counted into
    ``stats``."""
    ref, got = np.asarray(ref), host(got)
    if ref.dtype != got.dtype or ref.shape != got.shape:
        return f"{name}: {got.dtype} {got.shape}, the farm's {ref.dtype} {ref.shape}"
    if unbuilt:
        axis = -1 if key in ("vector", "list") else 2      # (N, U, plane, ...) or (..., col)
        idx = [_index(key, n) for n in unbuilt]
        if stats is not None:
            for n, i in zip(unbuilt, idx):
                d = int((_raw(np.take(ref, i, axis=axis)) != _raw(np.take(got, i, axis=axis)))
                        .sum())
                stats[(key, n)] = stats.get((key, n), 0) + d
        keep = [i for i in range(ref.shape[axis]) if i not in idx]
        ref, got = np.take(ref, keep, axis=axis), np.take(got, keep, axis=axis)
    a, b = _raw(ref), _raw(got)
    if not np.array_equal(a, b):
        at = tuple(int(x) for x in np.argwhere(a != b)[0])
        return (f"{name}: {int((a != b).sum())} value(s) differ, first at {at} "
                f"(farm {ref[at]!r}, device {got[at]!r})")
    return None


def _dict_difference(name, ref: dict, got: dict, unbuilt, stats):
    if set(ref) != set(got):
        return f"{name}: keys {sorted(got)} vs the farm's {sorted(ref)}"
    for k in ref:
        d = array_difference(f"{name}[{k!r}]", ref[k], got[k], unbuilt.get(k, ()), k, stats)
        if d:
            return d
    return None


#: The info keys compared as arrays (every documented one but the dicts and
#: the lists, compared on their own).
INFO_ARRAYS = ("tick", "episode", "episode_seed", "level", "ticks", "ended", "counters",
               "team_balance", "team_uncredited", "team_outcome", "rows_ended",
               "masked_choices")


def reset_difference(ref: tuple, got: tuple, unbuilt=None, stats=None):
    """``reset``'s ``(obs, info)`` of the device against the farm's."""
    unbuilt = unbuilt_parts() if unbuilt is None else unbuilt
    d = _dict_difference("obs", ref[0], got[0], unbuilt, stats)
    if d:
        return d
    for k in ("tick", "episode", "episode_seed", "level"):
        d = array_difference(f"info[{k!r}]", ref[1][k], got[1][k])
        if d:
            return d
    if ref[1]["checkpoints"] != got[1]["checkpoints"]:
        return "info['checkpoints'] differ"
    return _dict_difference("info['full_view']", ref[1]["full_view"], got[1]["full_view"],
                            unbuilt, stats)


def step_difference(ref: tuple, got: tuple, unbuilt=None, stats=None):
    """``step``'s ``(obs, reward, terminated, truncated, info)`` of the
    device against the farm's, every documented field, raw bits; the first
    difference named, or None."""
    unbuilt = unbuilt_parts() if unbuilt is None else unbuilt
    d = _dict_difference("obs", ref[0], got[0], unbuilt, stats)
    if d:
        return d
    for i, name in ((1, "reward"), (2, "terminated"), (3, "truncated")):
        d = array_difference(name, ref[i], got[i])
        if d:
            return d
    ri, gi = ref[4], got[4]
    for k in INFO_ARRAYS:
        d = array_difference(f"info[{k!r}]", ri[k], gi[k])
        if d:
            return d
    for k in ("final_obs", "final_full_view", "full_view"):
        d = _dict_difference(f"info[{k!r}]", ri[k], gi[k], unbuilt, stats)
        if d:
            return d
    if ri["result"] != gi["result"]:
        return f"info['result']: {gi['result']} vs the farm's {ri['result']}"
    if ri["checkpoints"] != gi["checkpoints"]:
        e = next(e for e in range(len(ri["checkpoints"]))
                 if ri["checkpoints"][e] != gi["checkpoints"][e])
        return (f"info['checkpoints'] env {e}: {gi['checkpoints'][e]} vs the farm's "
                f"{ri['checkpoints'][e]}")
    return None


def compare_backends(farm, dev, *, episodes: int, seed: int = 0, max_steps: int = 100000,
                     on_step=None, out=print) -> dict:
    """THE BACKEND-EQUALITY LOOP (the gate's and the pre-flight's): both
    backends reset, then stepped on the SAME actions -- drawn seeded-random
    under the farm's served masks once the device's masks are shown
    identical to them, raw bits -- until every env ended ``episodes``
    episodes (each auto-reset on both sides). Every reset / step output
    compared (:func:`reset_difference`, :func:`step_difference`); returns
    the run's statistics, ``"difference"`` the first one (None: equal)."""
    from simulation.action_spec_v1 import MASK_KEYS
    rng = np.random.default_rng(seed)
    unbuilt = unbuilt_parts()
    stats = {"steps": 0, "endings": np.zeros(farm.n_envs, dtype=np.int64),
             "end_steps": [], "difference": None, "unbuilt_cells": {}, "ticks": 0}
    ref, got = farm.reset(), dev.reset()
    d = reset_difference(ref, got, unbuilt, stats["unbuilt_cells"])
    obs_f, obs_d = ref[0], got[0]
    while d is None and stats["endings"].min() < episodes and stats["steps"] < max_steps:
        for k in MASK_KEYS:                     # identical masks before drawing from them
            d = array_difference(f"served {k} before step {stats['steps']}", obs_f[k], obs_d[k])
            if d:
                break
        if d:
            break
        acts = rl_env.random_legal_actions(rng, obs_f)
        ref = farm.step(acts)
        got = dev.step(acts)
        d = step_difference(ref, got, unbuilt, stats["unbuilt_cells"])
        if d is None and on_step is not None:
            d = on_step(stats["steps"], ref, got)
        if d is not None:
            d = f"step {stats['steps']}: {d}"
        ended = np.asarray(ref[4]["ended"], dtype=bool)
        stats["endings"] += ended
        for e in np.flatnonzero(ended):
            stats["end_steps"].append((int(e), stats["steps"]))
        stats["ticks"] += int(np.asarray(ref[4]["ticks"]).sum())
        stats["steps"] += 1
        obs_f, obs_d = ref[0], got[0]
    stats["difference"] = d
    return stats


def preflight(run, *, n_envs: int = 2, episodes: int = 1, seed: int = 0,
              max_steps: int = 20000, out=print) -> dict:
    """THE PRE-FLIGHT (design §5.3 / §5.5, critique 19): ``episodes`` random
    episodes per env of the run's own levels, reward profile and tables on
    BOTH backends (``n_envs`` envs each, no output directory), compared at
    every step (:func:`compare_backends`). Refuses (:class:`PreflightFailed`)
    on the first difference, and on a weighted reward term or a hand verb
    the run never exercised (its coverage: a counter of a term with a
    non-zero weight that never moved, a hand option never chosen). Returns
    ``{steps, endings, coverage, unbuilt_cells}``."""
    from dataclasses import replace
    r = replace(run, n_envs=int(n_envs), out_dir="")
    farm = rl_env.BatchedEnv(r, "cpu_farm", n_workers=min(int(n_envs), 4))
    try:
        dev = rl_env.BatchedEnv(r, "device")
        try:
            st = compare_backends(farm, dev, episodes=episodes, seed=seed,
                                  max_steps=max_steps, out=out)
            cov_f, cov_d = farm.coverage(), dev.coverage()
        finally:
            dev.close()
    finally:
        farm.close()
    if st["difference"] is not None:
        raise PreflightFailed(f"pre-flight: the backends differ at {st['difference']}")
    if st["endings"].min() < episodes:
        raise PreflightFailed(f"pre-flight: {st['endings'].tolist()} episodes ended in "
                              f"{st['steps']} steps, fewer than {episodes} per env")
    for k in cov_f:
        if not np.array_equal(cov_f[k], cov_d[k]):
            raise PreflightFailed(f"pre-flight: coverage {k!r} differs between the backends")
    missing = rl_env.unexercised(r, cov_f)
    if missing:
        raise PreflightFailed(f"pre-flight: never exercised in {st['steps']} steps: "
                              + "; ".join(missing))
    return {"steps": st["steps"], "endings": st["endings"].tolist(), "coverage": cov_f,
            "unbuilt_cells": st["unbuilt_cells"]}


__all__ = ["DeviceEnv", "DeviceBackendRefused", "DeviceEnvFailed", "PreflightFailed",
           "check_run", "batch_signature", "check_homogeneous", "unbuilt_parts",
           "array_difference", "reset_difference", "step_difference", "compare_backends",
           "preflight", "BREAKS", "DEVICE_HORDE", "DEVICE_COMMANDER", "host"]
