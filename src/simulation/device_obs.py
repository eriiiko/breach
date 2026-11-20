"""THE DEVICE OBSERVATION (#29 P8.0, residency design §4.4c "P8 RE-CUT,
VERSION 2"; inventory ``docs/residency_p8_observation_inventory_2026-11-20.md``
parts A.0, A.1, A.4, H): observation spec v1, its masks and the full view as
CuPy buffers OWNED BY THE ENV, built on the card inside the device
conductor's tick (:class:`simulation.conductor.DeviceConductor`) at the
decision cadence, handed to torch by DLPack with no host copy. The host
builders (``obs_spec_v1``, ``rl_env.EnvRunner.row_flags``) stay THE
reference; this is their twin, block by block, one P8 patch at a time.

CLASS T (training only): nothing here is synced state, no digest reads it,
and its device code writes no synced buffer.

THE SHAPE::

    c = DeviceConductor(sims)
    obs = DeviceObservation(c, specs, n_slots, decimation)   # builds tick 0
    c.tick()                 # at the walk's end, on a decision tick: the blocks
    obs.buffers["near_u"]    # CuPy (N, U, 5, 25, 25) uint8, the env's own
    obs.torch()["near_u"]    # the same memory as a torch tensor (DLPack)
    obs.team_rows(t)         # team t's rows, torch views (THE policy feed)

THE BUFFERS (:func:`served_shapes`): exactly the served keys, shapes and
dtypes -- ``obs_spec_v1.shapes`` (the observation and the two row flags),
``action_spec_v1.mask_shapes`` (the four masks) and
``obs_spec_v1.full_shapes`` (the full view) -- and a second set for every
``final_`` key (the terminal observation; P8.8b builds it). A key or plane
no patch has built yet is allocated, zero, and UNOWNED (:data:`BUILT` names
what is built; ``tests/obs_column_owners.toml`` is the column-to-patch
table).

THE ROW MAP: row ``t * S + s`` of env e is the store row of roster slot s of
marine team t (``Simulation.roster``: the spawn team, dead and turned units
kept; ``-1`` at an empty roster place), read from the env's own roster and
checked against the store's ``slot`` column at the build. ``in_match`` is
"the map holds a row"; ``valid`` is that row alive and still on team t
(``EnvRunner.row_flags``), built on the card from the store each decision.

THE BLOCKS (P8.0: the near window's ``wall`` / ``door`` planes, the far
window's ``wall_share`` / ``door`` planes, ``valid`` / ``in_match``): one
launch for every env (``cpp/src/cuda_obs.cu``), the spec's constants from the
GENERATED ``cpp/src/obs_layout.h`` (:func:`header_text`, ``tools/
gen_obs_layout.py``), never typed into the kernel. A row that is not valid
reads zeros (``observe_rows`` zeroes it).

THE CADENCE: an env's block is built on its decision ticks -- its episode
start (the tick it held when the observation was built) and every
``decimation`` ticks after (the run's ``[run] decimation``, the value the
reference's masks take; never a literal here) -- read by the kernel from the
device tick row and a per-env parameter row, so envs on different phases
(after a per-env reset, P8.9) are each built on their own ticks. The host
launches only when some env is due (the twins' host clocks; no sync).

THE ENDED LATCH: a per-env ``(N,)`` uint8 on the card; a set env is skipped
by every block (its buffers keep the last observation). A stub: P8.7 sets it.

THE HAND-OVER: ``torch.from_dlpack`` on each CuPy buffer (the tensor's data
pointer IS the buffer's). The kernels run on THE tick stream, a BLOCKING
stream (``cudaStreamCreate``), so torch / CuPy work on the legacy default
stream after a tick waits for them with no explicit event
(:func:`stream_ordering`, asserted by the check, never assumed).
"""
from __future__ import annotations

import hashlib

import numpy as np

#: The generated header and its generator (the unit layouts' precedent).
HEADER_REL = "cpp/src/obs_layout.h"
GENERATOR = "tools/gen_obs_layout.py --write"

#: What THIS patch's blocks build, per served key: the planes by name (a
#: windowed key) or None (a whole key). Every later P8 patch adds its rows
#: here with its kernel; ``tests/obs_column_owners.toml`` must agree.
BUILT = {
    "near_u": ("wall", "door"),
    "far_f": ("wall_share",),
    "far_u": ("door",),
    "valid": None,
    "in_match": None,
}

#: The per-env parameter row (int64): the episode's first tick, the decimation.
PARAM_EPISODE_TICK, PARAM_DECIMATION, PARAM_W = 0, 1, 2


class DeviceObsRefused(RuntimeError):
    """A batch or a level the device observation does not serve (by name)."""


# ---------------------------------------------------------------------------
# The generated header (the spec's constants for the kernels)
# ---------------------------------------------------------------------------
def _constants() -> list:
    """``(name, value)`` of every constant the observation's kernels read,
    each from the spec's own tables (``obs_spec_v1``, the door entity, the
    unit schema): the window, the plane indices, the fills, the door codes,
    the float16 bit patterns of the only float values P8.0 writes."""
    from simulation import obs_spec_v1 as o
    from simulation import unit_schema as us
    from simulation.entities.door import DOOR_CLOSED, DOOR_OPEN
    half = lambda x: int(np.float16(x).view(np.uint16))  # noqa: E731
    out = [
        ("OBS_WIN", o.WIN), ("OBS_HALF", o.HALF), ("OBS_BLOCK", o.BLOCK),
        ("OBS_NEAR_U_PLANES", len(o.NEAR_U_PLANES)),
        ("OBS_FAR_F_PLANES", len(o.FAR_F_PLANES)),
        ("OBS_FAR_U_PLANES", len(o.FAR_U_PLANES)),
        ("OBS_NEAR_U_WALL", o.NEAR_U_PLANES.index("wall")),
        ("OBS_NEAR_U_DOOR", o.NEAR_U_PLANES.index("door")),
        ("OBS_FAR_F_WALL_SHARE", o.FAR_F_PLANES.index("wall_share")),
        ("OBS_FAR_U_DOOR", o.FAR_U_PLANES.index("door")),
        ("OBS_WALL_OUTSIDE", int(o.OUTSIDE_FILL["wall"])),
        ("OBS_WALL_SHARE_OUTSIDE_HALF", half(o.OUTSIDE_FILL["wall_share"])),
        ("OBS_DOOR_CODE_NONE", o.DOOR_NONE),
        ("OBS_DOOR_CODE_OPEN", o.DOOR_CODE_OPEN),
        ("OBS_DOOR_CODE_CLOSED", o.DOOR_CODE_CLOSED),
        ("OBS_DOOR_STATE_OPEN", int(DOOR_OPEN)),
        ("OBS_DOOR_STATE_CLOSED", int(DOOR_CLOSED)),
        ("OBS_LIFE_ALIVE", us._LIFE_CHOICES.index("alive")),
        ("OBS_PARAM_EPISODE_TICK", PARAM_EPISODE_TICK),
        ("OBS_PARAM_DECIMATION", PARAM_DECIMATION),
        ("OBS_PARAM_W", PARAM_W),
    ]
    return out


def _share_halves() -> list:
    """float16 bit patterns of ``k / (BLOCK * BLOCK)``, k = 0 .. BLOCK^2:
    the far ``wall_share`` (``blocks(wall, 1).sum / 16.0``, written into a
    float16 buffer by numpy's direct conversion)."""
    from simulation import obs_spec_v1 as o
    n = o.BLOCK * o.BLOCK
    return [int(np.float16(k / float(n)).view(np.uint16)) for k in range(n + 1)]


def _layout_text() -> str:
    lines = [f"{k}={int(v)}\n" for k, v in _constants()]
    lines.append("WALL_SHARE_HALF=" + ",".join(str(x) for x in _share_halves()) + "\n")
    return "".join(lines)


def fingerprint() -> str:
    """SHA-256 over every generated constant (names, values, order)."""
    return hashlib.sha256(_layout_text().encode("ascii")).hexdigest()


def header_text() -> str:
    """The generated C++ header's full text (``cpp/src/obs_layout.h``)."""
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_obs_layout.py from src/simulation/device_obs.py",
        "// (#29 P8.0) -- NEVER EDIT BY HAND. The device observation's constants,",
        "// each read from observation spec v1's own tables (obs_spec_v1), the door",
        "// entity and the unit schema. A new constant an observation kernel reads",
        f"// is a row of device_obs._constants, then `{GENERATOR}`, commit,",
        "// rebuild cpp/build_cuda; tests/test_device_obs.py refuses a stale copy",
        "// and DeviceObservation a module whose fingerprint differs.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace obs_layout {",
        "",
        f'constexpr const char* OBS_LAYOUT_FINGERPRINT = "{fingerprint()}";',
        "",
    ]
    for k, v in _constants():
        lines.append(f"constexpr int {k} = {int(v)};")
    halves = ", ".join(f"0x{x:04x}" for x in _share_halves())
    lines += [
        "",
        "// float16 bit patterns of k / (OBS_BLOCK * OBS_BLOCK), k = 0 .. OBS_BLOCK^2",
        "// (a __constant__ table's initialiser in the kernel's TU).",
        f"#define OBS_WALL_SHARE_HALF_INIT {{{halves}}}",
        "",
        "}  // namespace obs_layout",
        "",
    ]
    return "\n".join(lines)


def check_compiled(bp) -> None:
    """Refuse a compiled module whose observation layout is not this one's."""
    got = getattr(bp, "OBS_LAYOUT_FINGERPRINT", None)
    if got != fingerprint():
        raise DeviceObsRefused(
            f"device observation: the compiled module's OBS_LAYOUT_FINGERPRINT "
            f"{got!r} is not device_obs.fingerprint() {fingerprint()[:12]!r} -- run "
            f"`{GENERATOR}`, commit, and rebuild cpp/build_cuda")


# ---------------------------------------------------------------------------
# The served shapes
# ---------------------------------------------------------------------------
def served_shapes(n_envs: int, n_teams: int, n_slots: int, n_kinds: int) -> dict:
    """Every served key (inventory A.0): the observation and the row flags
    (``obs_spec_v1.shapes``), the four masks (``action_spec_v1.mask_shapes``),
    the full view (``obs_spec_v1.full_shapes``); key -> (shape, dtype)."""
    from simulation import action_spec_v1 as a
    from simulation import obs_spec_v1 as o
    n_rows = int(n_teams) * int(n_slots)
    out = dict(o.shapes(n_envs, n_rows, n_slots))
    out.update(a.mask_shapes(n_envs, n_rows, int(n_slots) + o.ENTRIES_BEYOND_SLOTS,
                             n_kinds))
    out.update(o.full_shapes(n_envs, n_rows, n_teams))
    return out


def stream_ordering(bp) -> dict:
    """What the hand-over's ordering rests on, read from the runtime (the
    check asserts it): the tick stream's flags (0 = ``cudaStreamDefault``,
    i.e. BLOCKING: ordered with the legacy default stream), CuPy's and
    torch's current streams (0 = the legacy default stream)."""
    import cupy as cp
    out = {"tick_stream": int(bp.tick_stream_ptr()),
           "tick_stream_flags": int(cp.cuda.runtime.streamGetFlags(int(bp.tick_stream_ptr()))),
           "cupy_stream": int(cp.cuda.get_current_stream().ptr)}
    try:
        import torch
        out["torch_stream"] = int(torch.cuda.current_stream().cuda_stream)
    except ImportError:                       # pragma: no cover - torch is in the stack
        out["torch_stream"] = None
    return out


# ---------------------------------------------------------------------------
# The device observation
# ---------------------------------------------------------------------------
class DeviceObservation:
    """The served observation of a conductor's N envs, on the card (see the
    module docstring). ``specs``: each env's ``MatchSpec`` (its marine teams,
    as ``EnvRunner`` reads them); ``n_slots``: S (the layout's largest
    ``max_per_team``); ``decimation``: the run's ticks per decision;
    ``n_teams``: T (default: the most teams of any env)."""

    def __init__(self, conductor, specs, n_slots: int, decimation: int,
                 n_teams: int | None = None):
        import cupy as cp
        from simulation import unit_store
        from simulation.weapons import get_tables
        c = conductor
        self.cp, self.bp, self.c = cp, c.bp, c
        check_compiled(self.bp)
        specs = list(specs)
        if len(specs) != c.N:
            raise DeviceObsRefused(f"device observation: {len(specs)} specs for "
                                   f"{c.N} envs")
        self.decimation = int(decimation)
        if self.decimation < 1:
            raise DeviceObsRefused(f"device observation: decimation {decimation} < 1")
        self.env_teams = [len(s.teams) for s in specs]
        self.T = int(n_teams) if n_teams is not None else max(self.env_teams)
        if max(self.env_teams) > self.T:
            raise DeviceObsRefused(f"device observation: an env has {max(self.env_teams)} "
                                   f"marine teams, the layout {self.T}")
        self.S = int(n_slots)
        self.N = N = c.N
        self.n_rows = self.T * self.S
        self.n_kinds = len(get_tables().hand_grenade_rows)
        # -- the buffers: the served keys, and their final_ twins -----------
        self.shapes = served_shapes(N, self.T, self.S, self.n_kinds)
        self.buffers = {k: cp.zeros(sh, dtype=dt) for k, (sh, dt) in self.shapes.items()}
        self.final = {k: cp.zeros(sh, dtype=dt) for k, (sh, dt) in self.shapes.items()}
        self.unowned = tuple(k for k in self.shapes if k not in BUILT)
        # -- the row map: the env's roster, checked against the store ----------
        rowmap = np.full((N, self.n_rows), -1, dtype=np.int32)
        slot_col = unit_store.layout().column("slot").offset
        store_slot = c._rows[:, :, slot_col].get()
        for e, sim in enumerate(c.sims):
            for t in range(self.env_teams[e]):
                roster = sim.roster(t)
                if len(roster) > self.S:
                    raise DeviceObsRefused(f"device observation (env {e}): team {t} holds "
                                           f"{len(roster)} roster slots, the layout {self.S}")
                for u in roster:
                    if u is None:
                        continue
                    r = c.row_of[e].get(int(u.id))
                    if r is None or int(store_slot[e, r]) != int(u.slot):
                        raise DeviceObsRefused(
                            f"device observation (env {e}): roster unit {u.id} (team {t}, "
                            f"slot {u.slot}) is not the store's row {r}")
                    rowmap[e, t * self.S + int(u.slot)] = r
        self.rowmap = cp.asarray(rowmap)
        # -- the cadence and the ended latch -----------------------------------
        self.episode_tick = [int(s.tick) for s in c.sims]
        params = np.zeros((N, PARAM_W), dtype=np.int64)
        params[:, PARAM_EPISODE_TICK] = self.episode_tick
        params[:, PARAM_DECIMATION] = self.decimation
        self.params = cp.asarray(params)
        self.ended = cp.zeros((N,), dtype=cp.uint8)
        self.door_plane = cp.zeros((N, c.h, c.w), dtype=cp.int32)
        self.builds = 0                       # block launches (a decision tick of some env)
        self.skip_build = False               # THE BREAK hook: a stale buffer (the check's)
        self._torch = None
        c.attach_observation(self)
        self.build_if_due()                   # the episode's first decision (tick 0)

    # -- the cadence ---------------------------------------------------------
    def due(self) -> bool:
        """Some env is on a decision tick (the twins' host clocks: no sync)."""
        return any((int(s.tick) - t0) % self.decimation == 0
                   for s, t0 in zip(self.c.sims, self.episode_tick))

    def build_if_due(self) -> None:
        """The conductor's call line (the walk's last stage, and the build):
        the blocks of every env due this tick, one launch each, on the tick
        stream."""
        if not self.due() or self.skip_build:
            return
        self.build()

    def build(self, store_ptr=None) -> None:
        """Every block this patch owns, for every env whose decision tick it
        is (the kernel reads the tick row and the parameter row) and whose
        ended latch is clear. ``store_ptr``: the ``(N, U, W)`` store the
        blocks read (default the conductor's; the check's break hands a
        perturbed copy)."""
        c = self.c
        b = self.buffers
        store = int(c.store.device_ptr) if store_ptr is None else int(store_ptr)
        self.bp.obs_block(
            int(self.N), store, int(c.U), int(c.store.shape[2]),
            int(self.rowmap.data.ptr), int(self.n_rows), int(self.S),
            int(c.fields["solid"].data.ptr), int(c.h), int(c.w),
            c.doors.buffer, c.door_tiles.buffer, int(self.door_plane.data.ptr),
            int(c.trow.ptr), int(self.params.data.ptr), int(self.ended.data.ptr),
            int(b["near_u"].data.ptr), int(b["far_f"].data.ptr), int(b["far_u"].data.ptr),
            int(b["valid"].data.ptr), int(b["in_match"].data.ptr))
        self.builds += 1

    # -- the hand-over -------------------------------------------------------
    def torch(self) -> dict:
        """Every served buffer as a torch tensor sharing its memory (DLPack;
        no copy), built once."""
        if self._torch is None:
            import torch
            self._torch = {k: torch.from_dlpack(v) for k, v in self.buffers.items()}
        return self._torch

    def team_rows(self, team: int) -> dict:
        """Team ``team``'s rows of every served key as torch views
        (``BatchedEnv.team_rows``' slicing: ``[:, t*S:(t+1)*S]``)."""
        t, S = int(team), self.S
        if not 0 <= t < self.T:
            raise IndexError(f"device observation: team {t} of {self.T}")
        return {k: v[:, t * S:(t + 1) * S] for k, v in self.torch().items()}

    def host(self, keys=None) -> dict:
        """THE GATE'S DOWNLOAD (a sync; never inside a tick): the buffers as
        numpy."""
        keys = list(self.buffers) if keys is None else list(keys)
        return {k: self.buffers[k].get() for k in keys}


__all__ = ["HEADER_REL", "GENERATOR", "BUILT", "DeviceObsRefused", "DeviceObservation",
           "served_shapes", "stream_ordering", "header_text", "fingerprint",
           "check_compiled", "PARAM_EPISODE_TICK", "PARAM_DECIMATION", "PARAM_W"]
