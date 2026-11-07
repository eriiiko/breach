"""THE batched resident driver (#29 P5, docs/residency_batching_design_2026-11-07.md
§4.2 / §9 P5): N independent space-map simulations whose slot 7 runs as ONE
batched device call while their host slots (gameplay, the 9e entities) still
run per env on the host.

A :class:`ResidentBatch` owns one (N, ...) CuPy buffer per resident field
(gas (N, n_gases, h, w), the light triples (N, h, w, 3)) and points every
env's ``GameMap`` residency at its own slice (``gmap._dev[name]`` is
``buffer[e]``, a contiguous view), so each env's ``from_host`` / ``to_host``
fill and drain its slice unchanged and env 0's addresses are the batch's
base. :meth:`step` is one tick of every env:

1. per env, slots 1 .. 6b (``Simulation._step_pre_physics``) and the physics
   head (``PhysicsRunner._physics_head``: the blast source's bind, the trace
   dt check, the per-tick books' reset, the solid-books sink);
2. slot 7 of the whole batch (``physics_runner.run_resident_batch``: each
   env's entry upload into its slice, the device chain over N, the ONE books
   D2H folded into each env's engine, each env's exit download);
3. per env, slots 9 .. 10 (``Simulation._step_post_physics``) with that
   env's burn-through walls.

Env e after K ticks of the batch equals the same simulation stepped alone
on the resident path (batch invariance, ``tests/cuda_batch_invariance_
check.py``). A batch is HOMOGENEOUS: one grid shape, one gas table, one tilt,
space maps only (the sky exchange is the host's: ACCEPTED GAP), no light
(render state until #81). The templates, seeds and states differ per env.

This is the P5 driver G1 measures ("NO units, the ship's entities ON, device
vs farm") -- not the ``device`` backend of ``BatchedEnv`` (refused until
P8b). It needs the CUDA build and CuPy; ``physics_runner.set_residency``
must be on (the runner's step would otherwise take the host path).
"""
from __future__ import annotations

import numpy as np


class ResidentBatch:
    """Slot 7 of ``len(sims)`` simulations in one batched resident call."""

    def __init__(self, sims):
        sims = list(sims)
        if not sims:
            raise ValueError("ResidentBatch: at least one simulation")
        for e, s in enumerate(sims):
            if s.physics_runner is None:
                raise ValueError(f"ResidentBatch: env {e} has no physics runner")
        import cupy as cp
        self._cp = cp
        self.sims = sims
        self.n_env = len(sims)
        g0 = sims[0].gmap
        self._check_homogeneous(sims)
        self.h, self.w = g0._h, g0._w
        # One (N, ...) buffer per resident field; each env's map points at its
        # slice. The current host values seed the slices (enable_residency's
        # job for a lone map).
        from simulation.gamemap import GameMap
        self.buffers = {}
        for name in GameMap._RESIDENT_FIELD_NAMES:
            a0 = getattr(g0, name)
            buf = cp.empty((self.n_env,) + tuple(a0.shape), dtype=a0.dtype)
            self.buffers[name] = buf
        # The water stage's device `before` plane, (N, h, w), batch-owned.
        self.water_before = cp.zeros((self.n_env, self.h, self.w), dtype=np.int32)
        for e in range(self.n_env):
            self._bind(e, sims[e])

    @staticmethod
    def _check_homogeneous(sims):
        g0 = sims[0].gmap
        for e, s in enumerate(sims):
            g = s.gmap
            if (g._h, g._w) != (g0._h, g0._w):
                raise ValueError(f"ResidentBatch: env {e}'s grid {(g._h, g._w)} differs "
                                 f"from env 0's {(g0._h, g0._w)} (a batch is homogeneous)")
            if g.gas.shape != g0.gas.shape:
                raise ValueError(f"ResidentBatch: env {e}'s gas table differs")
            if (g.tilt_x, g.tilt_y) != (g0.tilt_x, g0.tilt_y):
                raise ValueError(f"ResidentBatch: env {e}'s tilt differs")
            if s._tps != sims[0]._tps:
                raise ValueError(f"ResidentBatch: env {e}'s tick rate differs")
            if g.is_ambient.any() and len(sims) > 1:
                raise ValueError(f"ResidentBatch: env {e} is an ambient map -- a batch "
                                 "is space maps (ACCEPTED GAP, design §4.2)")

    def _bind(self, e, sim):
        """Point env ``e``'s map at slice ``e`` and seed it from the host."""
        g = sim.gmap
        if g.__dict__.get("_residency_on", False) and getattr(g, "_batch", None) is not self:
            raise ValueError("ResidentBatch: a map already resident elsewhere")
        dev = {}
        for name, buf in self.buffers.items():
            view = buf[e]
            view.set(getattr(g, name))
            dev[name] = view
        g._cp = self._cp
        g._dev = dev
        g._residency_on = True
        g._res_tile_serial = g._tile_serial   # every field just copied
        g._batch = self
        r = sim.physics_runner
        r._batch_water_before = self.water_before[e]
        r._d_water_before = None              # re-seeded from the host twin
        r._water_depth_before = None          # the runner's lazy init re-runs

    def reset_env(self, e, sim):
        """Replace env ``e`` by ``sim`` (mid-batch reset: a fresh episode in
        that slot). The other envs are untouched."""
        if not 0 <= e < self.n_env:
            raise IndexError(e)
        self._check_homogeneous([self.sims[0] if e else sim, sim] + [
            s for k, s in enumerate(self.sims) if k not in (0, e)])
        self.sims[e] = sim
        self._bind(e, sim)

    def step(self):
        """One tick of every env (paused envs refuse: a batch steps together)."""
        from simulation import physics_runner as pr
        if not pr._RESIDENCY_ENABLED:
            raise RuntimeError("ResidentBatch.step: physics_runner.set_residency(True) first")
        for e, s in enumerate(self.sims):
            if s.paused:
                raise RuntimeError(f"ResidentBatch.step: env {e} is paused")
        for s in self.sims:
            ok = s._step_pre_physics()
            assert ok
        sim_time = 1.0 / float(self.sims[0]._tps)
        runners = [s.physics_runner for s in self.sims]
        gmaps = [s.gmap for s in self.sims]
        for r, g in zip(runners, gmaps):
            r._physics_head(g, sim_time)
        destroyed = pr.run_resident_batch(
            runners, gmaps, sim_time,
            water_before_base=int(self.water_before.data.ptr))
        for s, d in zip(self.sims, destroyed):
            s._step_post_physics(d)
        return destroyed
