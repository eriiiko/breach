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
reads zeros (``observe_rows`` zeroes it). P8.1: the vector columns of the
store and the tables (``BUILT["vector"]``), from the OBS TABLE
(:func:`table_values`: per env every value ``ObsContext.build`` and
``build_vector`` read that is not unit state, floats as their own float64
bit patterns; :func:`team_values`: per team its n and loadout row) through
the observation's float kit (``cpp/src/cuda_obs_float.cuh``: float64 in the
reference's order, never fused, one cast per value into the served dtype:
:data:`FLOAT16_KEYS` through the hand double -> float16, :data:`FLOAT32_KEYS`
through the plain cast).

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
    # P8.1: the vector columns that need only the unit store and the tables
    "vector": (
        "facing_cos", "facing_sin", "omega", "turn_left", "pushed_x", "pushed_y",
        "effort", "react", "prone", "can_move", "can_act", "can_aim",
        *(f"status_left[{k}]" for k in range(9)),
        "rounds", "mags", "ammo_untracked", "reloading", "reload_left",
        *(n for k in range(6) for n in (f"grenade[{k}]", f"grenade_kind_ok[{k}]")),
        "throw_windup", "throw_left", "throw_kind", "charges", "has_charge_kind",
        "planting", "plant_left", "plant_timer", "melee_cool", "has_melee",
        "shove_cool", "class", "weapon", "range", "rof", "spread", "trigger",
        "time_left", "mates_alive", "team_ammo"),
}

#: The served dtypes of the observation's float keys (P8.1, said once): the
#: float16 keys go through the kit's hand conversion, the float32 ones through
#: the plain cast, the float64 ones (``final_`` books) are written as they are.
FLOAT16_KEYS = ("near_f", "far_f")
FLOAT32_KEYS = ("vector", "list", "full_list", "full_match")

#: The per-env parameter row (int64): the episode's first tick, the decimation.
PARAM_EPISODE_TICK, PARAM_DECIMATION, PARAM_W = 0, 1, 2


# ---------------------------------------------------------------------------
# THE OBS TABLE (P8.1; inventory A.6, K4)
# ---------------------------------------------------------------------------
# Per env ONE int64 vector of every value ``ObsContext.build`` and
# ``build_vector`` read that is not unit state, each read from the
# reference's OWN object (the episode's ``ObsContext``, the sim's weapon rows,
# the loadout and env-profile tables, the species registry): the "Device
# tables" rule, the observation's side. A float the host reads as a float is
# carried as ITS OWN float64 bit pattern (``*_BITS``), never a Q16 row. The
# per-team rows (``n_t``, the team's loadout) are a second ``(N, T, TEAM_W)``
# buffer, T being the run's layout size. Offsets are generated into
# ``obs_layout.h`` (``OBS_TAB_*``), in the fingerprint.
#: A loadout row's fields (``[loadouts.*]``; GRENADES: the count per
#: hand-grenade row of the weapon table's family, in its order).
LOADOUT_FIELDS = ("MAGAZINES", "EXPLOSIVES", "CHARGE", "WEAPON", "GRENADES")
#: A weapon row's fields the observation divides by (``[weapons.*]``;
#: CONE_AIM_Q = ``combat.cone_q16_of_deg(spread_deg)``, the cone door).
WEAPON_FIELDS = ("MAG_SIZE", "RANGE_TILES", "ROF_TICKS", "RELOAD_TICKS", "CONE_AIM_Q")
#: An environment profile's floats (``[env_profiles.*]`` and the species'
#: own profiles), as float64 bit patterns.
PROFILE_FIELDS = ("TEMPERATURE_MAX_BITS", "PRESSURE_MIN_BITS", "PRESSURE_MAX_BITS")
#: A team's row: its ``n`` and its loadout's row (-1: none).
TEAM_FIELDS = ("N", "LOADOUT")
#: The scalars, in order: (name, the ObsContext attribute or a reader key).
TABLE_SCALARS = (
    "L_REF", "REF_ROF_TICKS", "REF_CONE_Q", "REF_MELEE_ARC_Q", "V_REF_Q", "V_FALL_Q",
    "REACT_TICKS", "SHOVE_COOLDOWN_TICKS", "LIMIT_TICKS", "MEMORY_TICKS", "TPS",
    "C4_FUSE_MAX_TICKS", "HAND_FUSE_MAX_TICKS", "TURN_RATE_CLASS_Q", "HALF_TURN_TICKS",
    "LINE_CLEAR_TILES", "ZOMBIE_NEAR_TILES", "X_MAX_Q", "ZOMBIE_CLASS", "CAP_Q",
    "FRAG_RADIUS", "MAX_THROW_RANGE", "N_TEAMS", "ZOMBIES", "HAND_WINDUP_TICKS",
    "PLANT_TICKS", "POISON_MIN_DENSITY_BITS", "TEARGAS_BLIND_DENSITY_BITS",
    "SMOKE_DENSE_BITS",
)


def _f64_bits(x) -> int:
    """A float's own float64 bit pattern as a signed int64."""
    return int(np.array(float(x), dtype=np.float64).view(np.int64))


def _bits_f64(b: int) -> float:
    return float(np.array(int(b), dtype=np.int64).view(np.float64))


def _table_counts() -> dict:
    """The widths the table's blocks take from the config tables (the
    unit schema's own row indexing: a column's row IS its position)."""
    from simulation import unit_schema as us
    from simulation.species import SPECIES_REGISTRY
    from simulation.weapons import get_tables as weapon_tables
    return {"loadouts": len(us.TABLES["loadouts"]()),
            "weapons": len(us.TABLES["weapons"]()),
            "profiles": len(us.TABLES["env_profiles"]()),
            "species": len(SPECIES_REGISTRY),
            "grenades": len(weapon_tables().hand_grenade_rows)}


def table_layout() -> list:
    """``[(name, offset, width)]`` of the obs table's rows, and the record
    widths: THE layout ``obs_layout.h`` carries."""
    n = _table_counts()
    loadout_w = len(LOADOUT_FIELDS) - 1 + n["grenades"]
    rows = [(s, 1) for s in TABLE_SCALARS]
    from simulation.status import N_STATUS_KINDS
    rows.append(("STATUS_SCALE", N_STATUS_KINDS))
    rows.append(("LOADOUT", n["loadouts"] * loadout_w))
    rows.append(("WEAPON", n["weapons"] * len(WEAPON_FIELDS)))
    rows.append(("PROFILE", n["profiles"] * len(PROFILE_FIELDS)))
    rows.append(("SPECIES_PROFILE", n["species"] * len(PROFILE_FIELDS)))
    out, off = [], 0
    for name, w in rows:
        out.append((name, off, w))
        off += w
    return out


def table_width() -> int:
    name, off, w = table_layout()[-1]
    return off + w


def _table_constants() -> list:
    n = _table_counts()
    out = [(f"OBS_TAB_{name}", off) for name, off, _ in table_layout()]
    out.append(("OBS_TAB_W", table_width()))
    out += [("OBS_TAB_N_LOADOUTS", n["loadouts"]), ("OBS_TAB_N_WEAPONS", n["weapons"]),
            ("OBS_TAB_N_PROFILES", n["profiles"]), ("OBS_TAB_N_SPECIES", n["species"]),
            ("OBS_TAB_N_GRENADES", n["grenades"])]
    for i, f in enumerate(LOADOUT_FIELDS):
        out.append((f"OBS_LOADOUT_{f}", i))
    out.append(("OBS_LOADOUT_W", len(LOADOUT_FIELDS) - 1 + n["grenades"]))
    for i, f in enumerate(WEAPON_FIELDS):
        out.append((f"OBS_WEAPON_{f}", i))
    out.append(("OBS_WEAPON_W", len(WEAPON_FIELDS)))
    for i, f in enumerate(PROFILE_FIELDS):
        out.append((f"OBS_PROFILE_{f}", i))
    out.append(("OBS_PROFILE_W", len(PROFILE_FIELDS)))
    for i, f in enumerate(TEAM_FIELDS):
        out.append((f"OBS_TEAM_{f}", i))
    out.append(("OBS_TEAM_W", len(TEAM_FIELDS)))
    return out


def table_values(sim, ctx) -> np.ndarray:
    """ONE env's obs table (``(OBS_TAB_W,)`` int64) from the reference's own
    objects: ``ctx`` (the episode's ``ObsContext``), ``sim.weapons_tables``
    (the rows ``build_vector`` reads), the loadout / env-profile tables and
    the species registry. A float is its own bit pattern."""
    from simulation import combat
    from simulation import unit_schema as us
    from simulation.loadouts import get_tables as loadout_tables
    from simulation.species import SPECIES_REGISTRY
    from simulation.weapons import PLANT_WEAPON
    rows = sim.weapons_tables.weapons.by_name
    scal = {
        "L_REF": ctx.l_ref, "REF_ROF_TICKS": ctx.ref_rof_ticks,
        "REF_CONE_Q": ctx.ref_cone_q, "REF_MELEE_ARC_Q": ctx.ref_melee_arc_q,
        "V_REF_Q": ctx.v_ref_q, "V_FALL_Q": ctx.v_fall_q, "REACT_TICKS": ctx.react_ticks,
        "SHOVE_COOLDOWN_TICKS": ctx.shove_cooldown_ticks, "LIMIT_TICKS": ctx.limit_ticks,
        "MEMORY_TICKS": ctx.memory_ticks, "TPS": ctx.tps,
        "C4_FUSE_MAX_TICKS": ctx.c4_fuse_max_ticks,
        "HAND_FUSE_MAX_TICKS": ctx.hand_fuse_max_ticks,
        # what build_vector's omega divides by: Simulation.unit_turn_rate_q16's
        # class value (the TURN hook's base)
        "TURN_RATE_CLASS_Q": sim._marine_turn_rate_q16,
        "HALF_TURN_TICKS": ctx.half_turn_ticks, "LINE_CLEAR_TILES": ctx.line_clear_tiles,
        "ZOMBIE_NEAR_TILES": ctx.zombie_near_tiles, "X_MAX_Q": ctx.x_max_q,
        "ZOMBIE_CLASS": ctx.zombie_class, "CAP_Q": ctx.cap_q,
        "FRAG_RADIUS": ctx.frag_radius, "MAX_THROW_RANGE": ctx.max_throw_range,
        "N_TEAMS": ctx.n_teams, "ZOMBIES": int(ctx.spec.zombies),
        "HAND_WINDUP_TICKS": rows["hand_grenade"].throw_windup_ticks,
        "PLANT_TICKS": rows[PLANT_WEAPON].plant_ticks,
        "POISON_MIN_DENSITY_BITS": _f64_bits(ctx.poison_min_density),
        "TEARGAS_BLIND_DENSITY_BITS": _f64_bits(ctx.teargas_blind_density),
        "SMOKE_DENSE_BITS": _f64_bits(ctx.smoke_dense),
    }
    out = np.zeros(table_width(), dtype=np.int64)
    off = {name: o for name, o, _ in table_layout()}
    for s in TABLE_SCALARS:
        out[off[s]] = int(scal[s])
    out[off["STATUS_SCALE"]:off["STATUS_SCALE"] + len(ctx.status_scale)] = ctx.status_scale
    # loadouts, in the unit schema's row order
    lt = loadout_tables()
    hand = tuple(ctx.hand_rows)
    lw = len(LOADOUT_FIELDS) - 1 + len(hand)
    for i, name in enumerate(us.TABLES["loadouts"]()):
        lo = lt.loadouts.by_name[name]
        b = off["LOADOUT"] + i * lw
        out[b + 0] = int(lo.magazines)
        out[b + 1] = int(lo.explosives)
        out[b + 2] = int(bool(lo.charge))
        out[b + 3] = (us.table_row("weapons", lo.weapon)
                      if lo.weapon in us.TABLES["weapons"]() else us.NO_ROW)
        for k, row in enumerate(hand):
            out[b + 4 + k] = int(lo.grenades.get(row, 0))
    # weapons: the schema's row order, the values from the SIM's rows
    ww = len(WEAPON_FIELDS)
    for i, name in enumerate(us.TABLES["weapons"]()):
        w = rows.get(name)
        if w is None:
            raise DeviceObsRefused(f"device observation: weapon row {name!r} of the "
                                   f"process table is not in the sim's weapon tables")
        b = off["WEAPON"] + i * ww
        out[b + 0] = int(w.mag_size)
        out[b + 1] = int(w.range_tiles)
        out[b + 2] = int(w.rof_interval_ticks)
        out[b + 3] = int(w.reload_ticks)
        out[b + 4] = int(combat.cone_q16_of_deg(w.spread_deg))
    # environment profiles: [env_profiles.*] rows, then every species' own
    pw = len(PROFILE_FIELDS)
    profs = [("PROFILE", list(us.TABLES["env_profiles"]().values())),
             ("SPECIES_PROFILE", [s.environment for s in SPECIES_REGISTRY.values()])]
    for block, objs in profs:
        for i, p in enumerate(objs):
            b = off[block] + i * pw
            out[b + 0] = _f64_bits(p.temperature_max)
            out[b + 1] = _f64_bits(p.pressure_min)
            out[b + 2] = _f64_bits(p.pressure_max)
    return out


def team_values(spec, n_teams: int) -> np.ndarray:
    """ONE env's team rows (``(T, TEAM_W)`` int64): per marine team of the
    run its ``n`` and its loadout's row, ``(0, -1)`` past the match's teams
    (``build_vector``'s ``n_t = teams[team].n if team < len(teams) else 0``)."""
    from simulation import unit_schema as us
    out = np.zeros((int(n_teams), len(TEAM_FIELDS)), dtype=np.int64)
    out[:, 1] = us.NO_ROW
    names = us.TABLES["loadouts"]()
    for t, team in enumerate(spec.teams[:int(n_teams)]):
        out[t, 0] = int(team.n)
        out[t, 1] = us.table_row("loadouts", team.loadout) if team.loadout in names \
            else us.NO_ROW
    return out


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
    # -- P8.1: the vector's layout, the arithmetic's spec constants, the table
    from simulation import status, turning, unit_fixed
    out.append(("OBS_V", o.V))
    out += [(f"OBS_V_{_cname(f.name)}", f.pos) for f in o.VECTOR_COLUMNS]
    out += [("OBS_SQUASH_CAP_BITS", _f64_bits(o.SQUASH_CAP)),
            ("OBS_FP_ONE", unit_fixed.FP_ONE), ("OBS_PI_Q16", turning.PI_Q16),
            ("OBS_N_STATUS_KINDS", status.N_STATUS_KINDS),
            ("OBS_HAND_GRENADE_SLOTS", o.HAND_GRENADE_SLOTS),
            ("OBS_NO_ROW", us.NO_ROW)]
    out += _table_constants()
    return out


def _cname(name: str) -> str:
    """A column name as a C identifier tail (``status_left[3]`` ->
    ``STATUS_LEFT_3``)."""
    return name.upper().replace("[", "_").replace("]", "")


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
        "// (#29 P8.0, P8.1) -- NEVER EDIT BY HAND. The device observation's",
        "// constants, each read from observation spec v1's own tables (obs_spec_v1),",
        "// the door entity and the unit schema, and the obs table's offsets",
        "// (device_obs.table_layout: its block widths are config table sizes). A",
        "// new constant an observation kernel reads",
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
        v = int(v)
        if -(1 << 31) <= v < (1 << 31):
            lines.append(f"constexpr int {k} = {v};")
        else:                                 # a float64 bit pattern
            lines.append(f"constexpr int64_t {k} = {v}LL;")
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
        # -- the obs table and the team rows (P8.1): from each env's own
        # ObsContext, the reference's object for the episode -------------------
        from simulation.obs_spec_v1 import ObsContext
        self.ctx = [ObsContext.build(sim, spec, self.S, n_teams=self.T)
                    for sim, spec in zip(c.sims, specs)]
        self.table_host = np.stack([table_values(sim, ctx)
                                    for sim, ctx in zip(c.sims, self.ctx)])
        self.teams_host = np.stack([team_values(spec, self.T) for spec in specs])
        self.table = cp.asarray(self.table_host)
        self.teams = cp.asarray(self.teams_host)
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
        # P8.1: the vector columns of the store and the tables
        self.bp.obs_vector(
            int(self.N), store, int(c.U), int(c.store.shape[2]), c.store.tables,
            int(self.rowmap.data.ptr), int(self.n_rows), int(self.S), int(self.T),
            int(self.table.data.ptr), int(self.table.shape[1]), int(self.teams.data.ptr),
            int(c.trow.ptr), int(self.params.data.ptr), int(self.ended.data.ptr),
            int(b["vector"].data.ptr))
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
           "check_compiled", "PARAM_EPISODE_TICK", "PARAM_DECIMATION", "PARAM_W",
           "FLOAT16_KEYS", "FLOAT32_KEYS", "table_layout", "table_width", "table_values",
           "team_values", "LOADOUT_FIELDS", "WEAPON_FIELDS", "PROFILE_FIELDS",
           "TEAM_FIELDS", "TABLE_SCALARS"]
