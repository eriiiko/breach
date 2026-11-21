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

P8.2 (no served column; what every later block reads): WHO SEES WHOM every
tick for every env (:meth:`DeviceObservation.vision`: ``vision.compute``'s
``VisionState`` as a ``(N, U, U)`` seen-by table and a ``(N, T + 1, U)``
per-team visible table, both sight rules, the rule and its dials rows of
the obs table), run at the walk's ``11'`` line before the blocks
(:meth:`DeviceObservation.stage`); SIGHT IN BULK (:func:`rays_clear_device`,
:func:`window_clear_device`); THE COVER TABLE (:func:`cover_values`, staged
once: nothing changes a cover on the continuous path) and the line of fire
over it (:meth:`DeviceObservation.lof_pairs`). The device functions are
``cpp/src/cuda_obs_sight.cuh``; a cover that blocks sight is refused.

P8.3: THE DEVICE TEAM PICTURE (:attr:`DeviceObservation.picture`,
``simulation.device_picture``): ``TeamPicture.update`` on the card every
tick, at :meth:`DeviceObservation.stage` between who sees whom and the
blocks; the vector's picture columns read it (``bp.obs_picture_vector``).
``picture_from_host``: the envs whose first state is restaged from their
host picture instead of the device's own first update.

THE CADENCE: an env's block is built on its decision ticks -- its episode
start (the tick it held when the observation was built) and every
``decimation`` ticks after (the run's ``[run] decimation``, the value the
reference's masks take; never a literal here) -- read by the kernel from the
device tick row and a per-env parameter row, so envs on different phases
(after a per-env reset, P8.9) are each built on their own ticks. The host
launches only when some env is due (the twins' host clocks; no sync).

THE ENDED LATCH: a per-env ``(N,)`` uint8 on the card; a set env is skipped
by every block (its buffers keep the last observation). P8.7: THE DEVICE BOOKS
(:attr:`DeviceObservation.books`, ``simulation.device_books``) set it at the
tick the env's match ends (after that tick's blocks) and freeze the env's
reward books; the books run at :meth:`DeviceObservation.stage` after the
picture and the blocks, under ``reward_profile`` (a named profile of
``reward_v1.PROFILES``; ``"counters"`` refused by name).

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
    "near_u": ("wall", "door",
               # P8.4a: the team-mates' fire lanes
               "lanes"),
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
        "time_left", "mates_alive", "team_ammo",
        # P8.3: the columns of the team picture (bars, motion, last hit, records)
        "hp", "vel_x", "vel_y", "hit_recent", "hit_age", "hit_from_cos", "hit_from_sin",
        "hit_cause", "enemies_known", "enemies_seen", "zombies_known", "zombies_near"),
    # P8.4b: the list -- every entry column but the decoder's (on_next_tick,
    # prev_aim: P8.6a) and the order block's (P8.6b) -- its mask and its uids
    "list": (
        "type", "dx", "dy", "dist_c", "cos", "sin", "vel_x", "vel_y", "side", "class", "hp",
        "exposure", "firing", "reloading", "reload_left", "winding", "throw_left",
        "throw_kind", "planting", "plant_left", "plant_timer", "stunned", "blinded", "down",
        "seen_by_me", "lof_solid", "lof_mate", "ticks_to_on", "in_range", "in_throw_range",
        "facing_on_me", "aims_at", "zombie_state", "mates_aiming", "mates_firing",
        "dmg_last_s", "priority", "frag_catch", "aim_cos", "aim_sin", "aim_is_unit",
        "feet_dx", "feet_dy", "throttle", "land_dx", "land_dy", "me_in_its_lane",
        "it_in_my_lane", "age", "obj_kind", "whose", "time_left", "remote", "blast_r",
        "in_blast", "on_floor", "open", "in_reach", "hull", "wired"),
    "list_mask": None,
    "list_uid": None,
    # P8.4c: the four masks (action_spec_v1.build_masks)
    "mask_feet": None,
    "mask_aim": None,
    "mask_trigger": None,
    "mask_hand": None,
}
# P8.5a: THE WINDOWS -- the eight physics planes of both windows, the two
# sight planes, the far unit planes, the vector's air block and felt
# temperature (``DeviceObservation.build_windows``)
_PHYSICS = ("temperature", "fire", "poison", "teargas", "smoke", "pressure", "wind_x", "wind_y")
BUILT["near_f"] = _PHYSICS
BUILT["near_u"] = BUILT["near_u"] + ("watched", "sees_target")
BUILT["far_f"] = BUILT["far_f"] + _PHYSICS
BUILT["far_u"] = BUILT["far_u"] + ("mates", "enemies", "records")
BUILT["vector"] = BUILT["vector"] + tuple(f"air[{k}]" for k in range(8)) + ("felt_temp",)
# P8.7: THE DEVICE BOOKS -- the reward side a step serves (``rl_env.Layout.shapes``;
# not observation buffers: ``DeviceBooks.served``, ``simulation.device_books``)
REWARD_KEYS = ("reward", "counters", "team_balance", "team_uncredited", "team_outcome",
               "rows_ended", "terminated", "truncated")
BUILT.update({k: None for k in REWARD_KEYS})
# P8.6a: THE DEVICE DECODER'S COLUMNS -- the vector's prev block, the list's
# prev_aim / on_next_tick (``device_decoder``, ``DeviceObservation.build_decoder_obs``)
BUILT["vector"] = BUILT["vector"] + tuple(f"prev[{h}]" for h in range(6))
BUILT["list"] = BUILT["list"] + ("on_next_tick", "prev_aim")

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
#: An ammo row's fields the list reads (P8.4b, ``[ammo.*]`` in the unit
#: schema's row order, the grenade / charge tables' ``ammo_name`` index):
#: OBJ_KIND = its place in ``ObsContext.obj_rows`` (-1: not a hand or charge
#: row), RADIUS = its payload row's radius (0: no payload,
#: ``_object_entry``'s KeyError), TRAVEL_SPEED_BITS = the round's
#: ``travel_speed_tiles_per_second`` as its float64 bits (what
#: ``Projectile.travel_speed`` holds).
AMMO_FIELDS = ("OBJ_KIND", "RADIUS", "TRAVEL_SPEED_BITS")
#: The scalars, in order: (name, the ObsContext attribute or a reader key).
TABLE_SCALARS = (
    "L_REF", "REF_ROF_TICKS", "REF_CONE_Q", "REF_MELEE_ARC_Q", "V_REF_Q", "V_FALL_Q",
    "REACT_TICKS", "SHOVE_COOLDOWN_TICKS", "LIMIT_TICKS", "MEMORY_TICKS", "TPS",
    "C4_FUSE_MAX_TICKS", "HAND_FUSE_MAX_TICKS", "TURN_RATE_CLASS_Q", "HALF_TURN_TICKS",
    "LINE_CLEAR_TILES", "ZOMBIE_NEAR_TILES", "X_MAX_Q", "ZOMBIE_CLASS", "CAP_Q",
    "FRAG_RADIUS", "MAX_THROW_RANGE", "N_TEAMS", "ZOMBIES", "HAND_WINDUP_TICKS",
    "PLANT_TICKS", "POISON_MIN_DENSITY_BITS", "TEARGAS_BLIND_DENSITY_BITS",
    "SMOKE_DENSE_BITS",
    # P8.2: who sees whom -- the match's sight rule (match.SIGHT_RULES' index of
    # Simulation.sight), the cone's cosine threshold (vision.cos_threshold of
    # [onephase] vision_cone_half_deg) and the awareness radius
    # (float([onephase] awareness_radius_tiles)), floats as their bits
    "SIGHT_RULE", "CONE_COS_BITS", "AWARENESS_R_BITS",
    # P8.4a: the fire lanes' dials (TeamKnowledge.lane_dials: [bots] lane_cell,
    # lane_halfwidth -- the one source every lane reader sees)
    "LANE_CELL", "LANE_HALFWIDTH",
    # P8.5a: the windows' table values P8.1 left -- the zombie profile's
    # resist_mult[POISON] (damage.mitigation_for on is_zombie) as its float64
    # bits, and the wind's impulse dial (FieldPlanes: the sim's ImpulseTable
    # wind_k_q, the observation's own impulse table)
    "ZOMBIE_RESIST_POISON_BITS", "WIND_K_Q",
)

# ---------------------------------------------------------------------------
# THE COVER TABLE (P8.2; critique finding 7): per env one row per cover
# entity of ``sim.cover`` (ordinal order, the order ``line_of_fire`` walks),
# its rectangle as the CoverRuntime's own float64 bit patterns and its
# ``alive`` / ``blocks_los``. Staged ONCE at the build: on the continuous path
# nothing changes a cover's ``alive`` (the only writer is
# ``CoverRuntime.chew``, reached only through a round marched with ``cover``,
# which only OnePhaseWEGO's ``timeline._fire_at`` passes; the conductor
# refuses that ruleset), so the table is the game's for the whole episode.
COVER_FIELDS = ("X0_BITS", "Y0_BITS", "X1_BITS", "Y1_BITS", "ALIVE", "BLOCKS_LOS")


def _row_of(name: str) -> int:
    """The obs table's offset of the scalar row ``name``."""
    return {n: o for n, o, _ in table_layout()}[name]


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
            "grenades": len(weapon_tables().hand_grenade_rows),
            "ammo": len(us.TABLES["ammo"]())}


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
    rows.append(("AMMO", n["ammo"] * len(AMMO_FIELDS)))          # P8.4b
    # P8.5a: per species its mitigation's resist_mult[POISON] (float64 bits;
    # a non-zombie's damage.mitigation_for is its species' profile)
    rows.append(("SPECIES_RESIST_POISON", n["species"]))
    # P8.7: the reward profile's terms (device_books.TERM_NAMES order) and its
    # group matrix (reward_v1.group_weights over rl_env.COUNTER_NAMES, (K, 5)
    # row-major), float64 bits: what the device books weigh with
    from simulation import device_books
    from simulation.reward_v1 import REWARD_GROUPS
    rows.append(("BOOK_TERMS", len(device_books.TERM_NAMES)))
    rows.append(("BOOK_G", len(device_books.counter_names()) * len(REWARD_GROUPS)))
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
            ("OBS_TAB_N_GRENADES", n["grenades"]), ("OBS_TAB_N_AMMO", n["ammo"])]
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
    for i, f in enumerate(COVER_FIELDS):
        out.append((f"OBS_COVER_{f}", i))
    out.append(("OBS_COVER_W", len(COVER_FIELDS)))
    for i, f in enumerate(AMMO_FIELDS):
        out.append((f"OBS_AMMO_{f}", i))
    out.append(("OBS_AMMO_W", len(AMMO_FIELDS)))
    return out


def table_values(sim, ctx, terms=None) -> np.ndarray:
    """ONE env's obs table (``(OBS_TAB_W,)`` int64) from the reference's own
    objects: ``ctx`` (the episode's ``ObsContext``), ``sim.weapons_tables``
    (the rows ``build_vector`` reads), the loadout / env-profile tables and
    the species registry; ``terms`` the reward profile's term table the books
    weigh with (P8.7; default ``"fireteam"``'s, ``device_books.profile_terms``).
    A float is its own bit pattern."""
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
        "TURN_RATE_CLASS_Q": sim._marine_turn_rate_q16,  # traits-class: the class value; the kernel applies the TURN hook (unit_lib::scaled)
        "HALF_TURN_TICKS": ctx.half_turn_ticks, "LINE_CLEAR_TILES": ctx.line_clear_tiles,
        "ZOMBIE_NEAR_TILES": ctx.zombie_near_tiles, "X_MAX_Q": ctx.x_max_q,
        "ZOMBIE_CLASS": ctx.zombie_class, "CAP_Q": ctx.cap_q,
        "FRAG_RADIUS": ctx.frag_radius, "MAX_THROW_RANGE": ctx.max_throw_range,
        "N_TEAMS": ctx.n_teams, "ZOMBIES": int(ctx.spec.zombies),
        "HAND_WINDUP_TICKS": rows["hand_grenade"].throw_windup_ticks,  # traits-class: the kernel applies HANDLING (unit_lib::handling_ticks)
        "PLANT_TICKS": rows[PLANT_WEAPON].plant_ticks,  # traits-class: the kernel applies HANDLING
        "POISON_MIN_DENSITY_BITS": _f64_bits(ctx.poison_min_density),
        "TEARGAS_BLIND_DENSITY_BITS": _f64_bits(ctx.teargas_blind_density),
        "SMOKE_DENSE_BITS": _f64_bits(ctx.smoke_dense),
        **sight_values(sim),
        **lane_values(),
        **window_values(sim),
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
        out[b + 3] = int(w.reload_ticks)  # traits-class: the kernel applies HANDLING
        out[b + 4] = int(combat.cone_q16_of_deg(w.spread_deg))  # traits-class: the kernel applies SPREAD (unit_lib::scaled)
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
    # ammo rows (P8.4b): the schema's row order, the values from the SIM's
    # tables (the objects _object_entry and Projectile read)
    aw = len(AMMO_FIELDS)
    wt = sim.weapons_tables
    obj_rows = tuple(ctx.obj_rows)
    for i, name in enumerate(us.TABLES["ammo"]()):
        b = off["AMMO"] + i * aw
        out[b + 0] = obj_rows.index(name) if name in obj_rows else -1
        try:
            out[b + 1] = int(wt.payload_for_ammo(name).radius)
        except KeyError:
            out[b + 1] = 0
        row = wt.ammo.by_name.get(name)
        out[b + 2] = _f64_bits(getattr(row, "travel_speed_tiles_per_second", 0.0)
                               if row is not None else 0.0)
    # P8.5a: each species' resist_mult[POISON], as physics_value reads it
    from simulation.damage import POISON
    b = off["SPECIES_RESIST_POISON"]
    for i, sp in enumerate(SPECIES_REGISTRY.values()):
        out[b + i] = _f64_bits(sp.mitigation.resist_mult[POISON])
    # P8.7: the reward profile's terms and group matrix
    from simulation import device_books
    tb, gb = device_books.table_rows(terms if terms is not None
                                     else device_books.profile_terms("fireteam"))
    out[off["BOOK_TERMS"]:off["BOOK_TERMS"] + len(tb)] = tb
    out[off["BOOK_G"]:off["BOOK_G"] + len(gb)] = gb
    return out


def window_values(sim) -> dict:
    """The obs table's window rows (P8.5a), read from what ``physics_value``
    and ``FieldPlanes`` read: the zombie mitigation's ``resist_mult[POISON]``
    (``damage.mitigation_for`` of a zombie) as its float64 bits, the sim's
    impulse table's ``wind_k_q`` (``observe_rows`` hands ``sim.impulse_table``
    to ``FieldPlanes``)."""
    from simulation.damage import POISON
    from simulation.species import ZOMBIE_MITIGATION
    return {"ZOMBIE_RESIST_POISON_BITS": _f64_bits(ZOMBIE_MITIGATION.resist_mult[POISON]),
            "WIND_K_Q": int(sim.impulse_table.wind_k_q)}


def sight_values(sim) -> dict:
    """The obs table's who-sees-whom rows (P8.2), read from what
    ``vision.can_see`` reads: the match's rule (``Simulation.sight``), the
    cone threshold through ``vision.cos_threshold`` and the awareness radius
    as ``float()`` of the live ``[onephase]`` values, floats as their bits."""
    from config import CFG
    from simulation import match, vision
    rule = getattr(sim, "sight", "cone")
    if rule not in match.SIGHT_RULES:
        raise DeviceObsRefused(f"device observation: sight rule {rule!r} is not one of "
                               f"{match.SIGHT_RULES}")
    return {"SIGHT_RULE": match.SIGHT_RULES.index(rule),
            "CONE_COS_BITS": _f64_bits(vision.cos_threshold(CFG.onephase.vision_cone_half_deg)),
            "AWARENESS_R_BITS": _f64_bits(float(CFG.onephase.awareness_radius_tiles))}


def lane_values() -> dict:
    """The obs table's fire-lane rows (P8.4a): ``TeamKnowledge.lane_dials``,
    the reference's own reader of the bots' dials."""
    from simulation.obs_spec_v1 import TeamKnowledge
    cell, hw = TeamKnowledge().lane_dials
    return {"LANE_CELL": int(cell), "LANE_HALFWIDTH": int(hw)}


def lane_grid_shape(h: int, w: int, cell: int) -> tuple:
    """``draw_lanes``' coarse grid shape ``(h // cell + 1, w // cell + 1)``."""
    return (int(h) // int(cell) + 1, int(w) // int(cell) + 1)


def cover_values(sim) -> np.ndarray:
    """ONE env's cover table (``(C, COVER_W)`` int64, C = len(sim.cover)):
    each ``CoverRuntime`` of ``sim.cover`` in its order, the rectangle as its
    own float64 bit patterns, ``alive``, ``blocks_los``. A cover that blocks
    SIGHT is refused by name: the host applies it inside every
    ``vision.ray_clear`` / ``rays_clear`` / ``window_clear`` (so in who sees
    whom and in every window), which the device's lines do not; no level
    places one (``gen_training_rooms``' crates are ``blocks_los = false``),
    and its port lands with the first level that does."""
    covers = list(getattr(sim, "cover", ()) or ())
    out = np.zeros((len(covers), len(COVER_FIELDS)), dtype=np.int64)
    for i, c in enumerate(covers):
        if bool(c.blocks_los):
            raise DeviceObsRefused(
                f"device observation: cover {c.id!r} blocks line of sight (blocks_los = "
                f"true): its device form is not built (the host applies it in "
                f"vision.ray_clear / rays_clear / window_clear)")
        out[i] = (_f64_bits(c.x0), _f64_bits(c.y0), _f64_bits(c.x1), _f64_bits(c.y1),
                  int(bool(c.alive)), int(bool(c.blocks_los)))
    return out


def check_teams(sim, n_teams: int, where: str = "") -> None:
    """Refuse a sim holding a unit whose team is neither a marine team of
    the layout (``0 <= team < n_teams``) nor the zombie team: the visible
    table has a slot per marine team and one for the zombies."""
    from simulation.factions import ZOMBIE_TEAM
    for u in sim.units:
        t = int(u.team)
        if not (0 <= t < int(n_teams) or t == ZOMBIE_TEAM):
            raise DeviceObsRefused(f"device observation{where}: unit {u.id} is on team {t}, "
                                   f"neither a marine team < {n_teams} nor the zombie team")


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


# ---------------------------------------------------------------------------
# P8.4a: THE LANES, AIM_OF, THE LINE OF FIRE, TICKS_TO_WITHIN (the per-unit
# answers the list reads; ``bp.obs_aims``)
# ---------------------------------------------------------------------------
#: The codes of the aim / line-of-fire buffers: ``aim`` holds aim_of's uid,
#: AIM_NONE for nobody, AIM_NOT_ASKED where the list asks nothing (neither a
#: mate of the team nor a unit it sees with a record); ``lof`` holds LOF_ASKED
#: | LOF_SOLID (solid_in_way) | LOF_MATE (mate_in_way). The lane grids' -1 /
#: -2 are draw_lanes' own (nobody / several owners).
AIM_CODES = {"AIM_NONE": -1, "AIM_NOT_ASKED": -2, "LOF_ASKED": 1, "LOF_SOLID": 2,
             "LOF_MATE": 4}
#: Fault bits of ``obs_aims`` (a per-env word the gate reads; no served level
#: reaches one): the lane grid's shape is not the dials', a coordinate past
#: the kit's int32 / the sight's range, the turning law past its trial bound.
AIM_FAULT_BITS = {"GRID": 1, "RANGE": 2, "LAW": 4}
#: The check's breaks (``OBS_BREAK``): ``lane_trunc`` (C's truncating
#: division in place of Python's flooring ``//`` in the lane walk and the
#: rasteriser), ``lane_wide`` (the lane walk one tile wide of its line).
AIM_BREAK_CODES = {"lane_trunc": 1, "lane_wide": 2}


# ---------------------------------------------------------------------------
# P8.4b: THE LIST (``bp.obs_list``)
# ---------------------------------------------------------------------------
#: Fault bits of ``obs_list`` (a per-env word the gate reads; no served level
#: reaches one): a seen object whose serial is on no live table row.
LIST_FAULT_BITS = {"OBJECT": 1}
#: The check's breaks (``OBS_BREAK``): ``list_tie`` (a block's sort ties by the
#: HIGHER id, the tie-break turned round), ``list_cap`` (the door block holds
#: one entry fewer than its capacity).
LIST_BREAK_CODES = {"list_tie": 1, "list_cap": 2}


# ---------------------------------------------------------------------------
# P8.4c: THE FOUR MASKS (``bp.obs_masks``)
# ---------------------------------------------------------------------------
#: Fault bits of ``obs_masks`` (a per-env word the gate reads; no served level
#: reaches one): GATE (a unit-library gate set its fault: an index past its
#: table), DOOR (a live door's span off the door-tile table).
MASK_FAULT_BITS = {"GATE": 1, "DOOR": 2}
#: The check's breaks (``OBS_BREAK``): ``mask_swap`` (the reload and shove
#: bits' gates swapped), ``mask_reach`` (the use mask's door reach one tile
#: wider: the footprint read one larger).
MASK_BREAK_CODES = {"mask_swap": 1, "mask_reach": 2}


# ---------------------------------------------------------------------------
# P8.5a: THE WINDOWS (``bp.obs_planes``)
# ---------------------------------------------------------------------------
#: Fault bits of ``obs_planes`` (a per-env word the gate reads; no served level
#: reaches one): SIGHT (a sight walk's coordinate past the kit's range), FELT
#: (the felt temperature's footprint or exposure past its proved range),
#: PROFILE (a unit's environment row past the obs table's profile blocks).
WIN_FAULT_BITS = {"SIGHT": 1, "FELT": 2, "PROFILE": 4}
#: The check's breaks (``OBS_BREAK``): ``win_profile`` (every plane's profile
#: read one row further in the obs table: the neighbouring row; today's config
#: holds one env profile and one species with the same three floats, so it
#: cannot turn red), ``win_divisor`` (each gas plane divided by the
#: neighbouring density row of the obs table),
#: ``win_anchor`` (the far window's block anchor one block off), ``win_half``
#: (the float16 cast through float32: double rounding), ``win_argmax`` (the far
#: wind's argmax takes the LAST largest load, the reduction in the other order).
WIN_BREAK_CODES = {"win_profile": 1, "win_anchor": 2, "win_half": 3, "win_argmax": 4,
                   "win_divisor": 5}


def wired_values(sim, cap: int) -> np.ndarray:
    """ONE env's wired flags (``(cap,)`` uint8) per door table row (row k =
    ``sim._doors[k]``, ordinal order): ``room_graph.wired_door_ordinals`` of
    the level's wires (``ObsWorld.wires``), static for the episode."""
    from simulation import room_graph
    wired = room_graph.wired_door_ordinals(tuple(getattr(sim.level, "wires", ()) or ()))
    out = np.zeros((int(cap),), dtype=np.uint8)
    for k, d in enumerate(getattr(sim, "_doors", ()) or ()):
        out[k] = 1 if int(d.ordinal) in wired else 0
    return out


class DeviceObsRefused(RuntimeError):
    """A batch or a level the device observation does not serve (by name)."""


#: The resident planes the windows read, each in its own dtype (P8.5a): the
#: dequantizes' sources (``FieldPlanes``), the wind's mass (``gas`` O2 + N2),
#: the felt temperature's exposures (``heat``, ``rad_flux_sweep``) and the
#: sight planes' (``solid``, ``sight_screen``).
WINDOW_PLANES = {"temperature": "int32", "fire": "int32", "gas": "int32",
                 "atmosphere": "int32", "wind_x": "int32", "wind_y": "int32",
                 "heat": "int32", "rad_flux_sweep": "int64", "solid": "bool",
                 "sight_screen": "bool"}


def check_window_sources(conductor, ctxs) -> None:
    """Refuse a batch whose windows' sources are not what the kernel reads:
    every plane of :data:`WINDOW_PLANES` resident (a device buffer of the
    batch) in its dtype, and the felt temperature's Q16 dials -- the unit
    library's law reads the DEVICE TABLES' ``heat_gain_q`` / ``heat_ambient_q``
    -- equal to ``exchange.heat_dials_q`` of each episode's ``ObsContext``
    ``heat_dials`` (the observation's own object)."""
    from simulation import exchange, unit_tables
    f = conductor.fields
    for name, dt in WINDOW_PLANES.items():
        if name not in f:
            raise DeviceObsRefused(f"device observation: the windows read plane {name!r}, "
                                   f"which is not resident")
        if str(f[name].dtype) != dt:
            raise DeviceObsRefused(f"device observation: plane {name!r} is "
                                   f"{f[name].dtype}, the windows read {dt}")
    for e, (sim, ctx) in enumerate(zip(conductor.sims, ctxs)):
        vec = unit_tables.values(sim)
        want = exchange.heat_dials_q(ctx.heat_dials)
        got = (unit_tables.read(vec, "heat_gain_q"), unit_tables.read(vec, "heat_ambient_q"))
        if tuple(int(x) for x in want) != tuple(got):
            raise DeviceObsRefused(f"device observation (env {e}): the device tables' heat "
                                   f"dials {got} are not the episode's {tuple(want)}")


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
    # -- P8.2: the sight rules' codes, the zombie team (the visible table's
    # last slot)
    from simulation import match
    from simulation.factions import ZOMBIE_TEAM
    out += [("OBS_SIGHT_CONE", match.SIGHT_RULES.index("cone")),
            ("OBS_SIGHT_ALL_ROUND", match.SIGHT_RULES.index("all_round")),
            ("OBS_ZOMBIE_TEAM", int(ZOMBIE_TEAM))]
    out += _table_constants()
    # -- P8.3: the device team picture's block, faults and breaks
    from simulation import device_picture
    out += device_picture.constants()
    # -- P8.4a: the lanes plane, the aim / line-of-fire buffers' codes, faults
    # and breaks
    out += [("OBS_NEAR_U_LANES", o.NEAR_U_PLANES.index("lanes"))]
    out += [(f"OBS_{k}", v) for k, v in AIM_CODES.items()]
    out += [(f"OBS_AIM_FAULT_{k}", v) for k, v in AIM_FAULT_BITS.items()]
    out += [(f"OBS_AIM_BREAK_{k.upper()}", v) for k, v in AIM_BREAK_CODES.items()]
    # -- P8.4b: the entry's layout, the blocks' types and capacities, the uid
    # bases, the sentinels the entry reads
    from simulation import damage, movement, team_picture
    out.append(("OBS_F", o.F))
    out += [(f"OBS_L_{_cname(f.name)}", f.pos) for f in o.ENTRY_COLUMNS]
    out += [("OBS_TYPE_MATE", o.TYPE_MATE), ("OBS_TYPE_SEEN", o.TYPE_SEEN),
            ("OBS_TYPE_RECORD", o.TYPE_RECORD), ("OBS_TYPE_OBJECT", o.TYPE_OBJECT),
            ("OBS_TYPE_DOOR", o.TYPE_DOOR),
            ("OBS_CAP_SEEN", o.CAPS_FIXED[0]), ("OBS_CAP_RECORD", o.CAPS_FIXED[1]),
            ("OBS_CAP_OBJECT", o.CAPS_FIXED[2]), ("OBS_CAP_DOOR", o.CAPS_FIXED[3]),
            ("OBS_ENTRIES_BEYOND_SLOTS", o.ENTRIES_BEYOND_SLOTS),
            ("OBS_DOOR_UID_BASE", o.DOOR_UID_BASE),
            ("OBS_LIST_OBJECT_UID_BASE", team_picture.OBJECT_UID_BASE),
            ("OBS_THROW_NO_REACH", us.THROW_NO_REACH), ("OBS_NO_TIMER", us.NO_TIMER),
            ("OBS_THROTTLE_ONE", movement.THROTTLE_ONE), ("OBS_KINETIC", damage.KINETIC)]
    out += [(f"OBS_LIST_FAULT_{k}", v) for k, v in LIST_FAULT_BITS.items()]
    out += [(f"OBS_LIST_BREAK_{k.upper()}", v) for k, v in LIST_BREAK_CODES.items()]
    # -- P8.4c: the four masks' layout (action spec v1's own constants), the
    # aimable entry types as a bit set, faults and breaks
    from simulation import action_spec_v1 as a
    out += [("OBS_MASK_N_FEET", a.N_FEET), ("OBS_MASK_FEET_STAND", a.FEET_STAND),
            ("OBS_MASK_N_TRIGGER", a.N_TRIGGER), ("OBS_MASK_AIM_WALK", a.AIM_WALK),
            ("OBS_MASK_AIM_KEEP", a.AIM_KEEP), ("OBS_MASK_AIM_TOKENS", a.AIM_TOKENS),
            ("OBS_MASK_AIMABLE_BITS", sum(1 << int(t) for t in a.AIMABLE_TYPES))]
    out += [(f"OBS_MASK_HAND_{n.upper()}", i) for i, n in enumerate(a.FIXED_HANDS)]
    out += [("OBS_MASK_HAND_THROW0", a.HAND_THROW0)]
    out += [(f"OBS_MASK_FAULT_{k}", v) for k, v in MASK_FAULT_BITS.items()]
    out += [(f"OBS_MASK_BREAK_{k.upper()}", v) for k, v in MASK_BREAK_CODES.items()]
    # -- P8.5a: the windows -- the planes' indices (the physics planes in the
    # spec's order, which the kernel's law follows by name), the currencies
    # the dequantizes divide by (gas_fixed / fire_fixed / atmosphere_fixed),
    # the gas rows FieldPlanes and wind_load_plane read, faults and breaks
    from simulation import atmosphere_fixed, fire_fixed, gamemap, gas_fixed, gases
    if tuple(o.PHYSICS_PLANES) != _PHYSICS or tuple(o.NEAR_F_PLANES) != _PHYSICS:
        raise DeviceObsRefused(f"device observation: the physics planes {o.PHYSICS_PLANES} "
                               f"are not the windows kernel's law order {_PHYSICS}")
    out += [("OBS_NEAR_F_PLANES", len(o.NEAR_F_PLANES)),
            ("OBS_N_PHYSICS", len(o.PHYSICS_PLANES)),
            ("OBS_NEAR_U_WATCHED", o.NEAR_U_PLANES.index("watched")),
            ("OBS_NEAR_U_SEES_TARGET", o.NEAR_U_PLANES.index("sees_target")),
            ("OBS_FAR_F_PHYSICS0", o.FAR_F_PLANES.index(o.PHYSICS_PLANES[0])),
            ("OBS_FAR_U_MATES", o.FAR_U_PLANES.index("mates")),
            ("OBS_FAR_U_ENEMIES", o.FAR_U_PLANES.index("enemies")),
            ("OBS_FAR_U_RECORDS", o.FAR_U_PLANES.index("records")),
            ("OBS_Q16_ONE", int(gas_fixed._Q16_ONE_F)),
            ("OBS_TRACE_ONE", int(gas_fixed._TRACE_ONE_F)),
            ("OBS_FIRE_ONE", int(fire_fixed.FP_ONE_F)),
            ("OBS_ATM_ONE", int(atmosphere_fixed.FP_ONE_F)),
            ("OBS_WIND_SHIFT", int(atmosphere_fixed.FP_SHIFT)),
            ("OBS_GAS_POISON", int(gases.POISON)), ("OBS_GAS_TEARGAS", int(gases.TEARGAS)),
            ("OBS_GAS_SMOKE", int(gases.SMOKE)), ("OBS_GAS_O2", int(gamemap.O2)),
            ("OBS_GAS_N2", int(gamemap.INERT_N2))]
    out += [(f"OBS_WIN_FAULT_{k}", v) for k, v in WIN_FAULT_BITS.items()]
    out += [(f"OBS_WIN_BREAK_{k.upper()}", v) for k, v in WIN_BREAK_CODES.items()]
    # -- P8.7: the device books (counters' columns, fields, terms, codes)
    from simulation import device_books
    out += device_books.constants()
    # -- P8.6a: the device decoder (the action spec's head sizes, the state
    # record, the table, the horde latch's layout, faults and breaks)
    from simulation import device_decoder
    out += device_decoder.constants()
    return out


def _cname(name: str) -> str:
    """A column name as a C identifier tail (``status_left[3]`` ->
    ``STATUS_LEFT_3``)."""
    return name.upper().replace("[", "_").replace("]", "")


def _compass() -> str:
    from simulation import device_decoder
    return device_decoder.compass_init()


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
    lines.append("DEC_COMPASS=" + _compass() + "\n")
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
        "// (#29 P8.0 .. P8.7) -- NEVER EDIT BY HAND. The device observation's",
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
        "// P8.6a: action_spec_v1.COMPASS_Q16 as (x, y) pairs, direction d at 2d.",
        f"#define OBS_DEC_COMPASS_INIT {{{_compass()}}}",
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
# P8.2: sight in bulk on the card (vision.rays_clear / window_clear's twins)
# ---------------------------------------------------------------------------
def _plane_ptrs(solid, screen):
    """``(N, h, w)`` one-byte CuPy planes (bool or uint8) -> (N, h, w, ptrs)."""
    if solid.shape != screen.shape or solid.ndim != 3:
        raise ValueError(f"sight planes: solid {solid.shape} / screen {screen.shape} are not "
                         f"one (N, h, w) shape")
    if solid.dtype.itemsize != 1 or screen.dtype.itemsize != 1:
        raise ValueError("sight planes: solid / screen must be one byte per tile")
    n, h, w = (int(x) for x in solid.shape)
    return n, h, w, int(solid.data.ptr), int(screen.data.ptr)


def rays_clear_device(bp, solid, screen, pairs, physical=False):
    """``vision.rays_clear`` for every env in ONE launch, on the tick stream:
    ``pairs`` ``(N, P, 4)`` int32 CuPy (y0, x0, y1, x1) over env e's planes
    ``solid[e]`` / ``screen[e]`` -> ``(N, P)`` uint8 (1 = clear) and the
    ``(N,)`` int32 fault bits. Equal to ``rays_clear`` on every level the
    device serves (no cover that blocks sight: :func:`cover_values`)."""
    import cupy as cp
    n, h, w, ps, pc = _plane_ptrs(solid, screen)
    pairs = cp.ascontiguousarray(pairs, dtype=cp.int32)
    if pairs.ndim != 3 or pairs.shape[0] != n or pairs.shape[2] != 4:
        raise ValueError(f"rays_clear_device: pairs {pairs.shape} is not (N={n}, P, 4)")
    out = cp.zeros(pairs.shape[:2], dtype=cp.uint8)
    faults = cp.zeros((n,), dtype=cp.int32)
    bp.obs_rays(n, ps, pc, h, w, bool(physical), int(pairs.data.ptr), int(pairs.shape[1]),
                int(out.data.ptr), int(faults.data.ptr))
    return out, faults


def window_clear_device(bp, solid, screen, queries, wh, ww, to_window=False, physical=False):
    """``vision.window_clear`` for every env in ONE launch, on the tick
    stream: ``queries`` ``(N, Q, 4)`` int32 CuPy (y, x, wy, wx) -> ``(N, Q,
    wh, ww)`` uint8 (FROM (y, x) to each cell of the window whose top-left is
    (wy, wx), or ``to_window`` FROM each cell) and the ``(N,)`` fault bits."""
    import cupy as cp
    n, h, w, ps, pc = _plane_ptrs(solid, screen)
    queries = cp.ascontiguousarray(queries, dtype=cp.int32)
    if queries.ndim != 3 or queries.shape[0] != n or queries.shape[2] != 4:
        raise ValueError(f"window_clear_device: queries {queries.shape} is not (N={n}, Q, 4)")
    out = cp.zeros((n, int(queries.shape[1]), int(wh), int(ww)), dtype=cp.uint8)
    faults = cp.zeros((n,), dtype=cp.int32)
    bp.obs_windows(n, ps, pc, h, w, bool(physical), bool(to_window), int(queries.data.ptr),
                   int(queries.shape[1]), int(wh), int(ww), int(out.data.ptr),
                   int(faults.data.ptr))
    return out, faults


def vision_state(sees, visible, uids, teams_of_slot) -> tuple:
    """One env's device tables as ``VisionState``'s two dicts (the gate's
    reading): ``sees`` ``(U, U)``, ``visible`` ``(T + 1, U)`` numpy, ``uids``
    the store rows' unit ids (row order), ``teams_of_slot`` the team value of
    each visible slot. ``seen_by``: target id -> sorted observer ids;
    ``visible_by_team``: team -> sorted target ids (only non-empty, as the
    reference builds them)."""
    seen_by = {}
    n = len(uids)
    for t in range(n):
        obs = [int(uids[o]) for o in np.flatnonzero(sees[:n, t])]
        if obs:
            seen_by[int(uids[t])] = tuple(sorted(obs))
    by_team = {}
    for s, team in enumerate(teams_of_slot):
        ids = [int(uids[t]) for t in np.flatnonzero(visible[s, :n])]
        if ids:
            by_team[int(team)] = tuple(sorted(ids))
    return seen_by, by_team


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
                 n_teams: int | None = None, picture_from_host=(),
                 reward_profile: str = "fireteam", reward_overrides=None):
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
        from simulation import device_books
        # P8.7: the reward profile the books weigh with ("counters" refused by name)
        self.reward_profile = str(reward_profile)
        self.reward_terms = device_books.profile_terms(self.reward_profile, reward_overrides)
        self.ctx = [ObsContext.build(sim, spec, self.S, n_teams=self.T)
                    for sim, spec in zip(c.sims, specs)]
        self.table_host = np.stack([table_values(sim, ctx, self.reward_terms)
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
        # -- P8.2: who sees whom (every tick) and the cover table --------------
        for e, sim in enumerate(c.sims):
            check_teams(sim, self.T, f" (env {e})")
        covers = [cover_values(sim) for sim in c.sims]
        self.cover_cap = max(1, max(len(x) for x in covers))
        cov = np.zeros((N, self.cover_cap, len(COVER_FIELDS)), dtype=np.int64)
        for e, x in enumerate(covers):
            cov[e, :len(x)] = x
        self.cover_host = cov
        self.cover_count_host = np.array([len(x) for x in covers], dtype=np.int32)
        self.cover = cp.asarray(cov)
        self.cover_count = cp.asarray(self.cover_count_host)
        U = int(c.U)
        self.sees = cp.zeros((N, U, U), dtype=cp.uint8)           # [observer row, target row]
        self.visible = cp.zeros((N, self.T + 1, U), dtype=cp.uint8)  # [team slot, target row]
        self.vision_faults = cp.zeros((N,), dtype=cp.int32)
        self.visions = 0                      # vision launches (one per tick)
        self.early_vision = False             # THE BREAK hook: vision one slot early (conductor)
        self.early_picture = False            # THE BREAK hook: vision + picture before the conversion
        self.builds = 0                       # block launches (a decision tick of some env)
        self.skip_build = False               # THE BREAK hook: a stale buffer (the check's)
        self._torch = None
        # -- P8.3: the device team picture (state across ticks) ----------------
        from simulation.device_picture import DevicePicture
        self.picture = DevicePicture(self)
        # -- P8.4a: the lanes, aim_of, the line of fire, ticks_to_within ------
        cells = {int(self.table_host[e, _row_of("LANE_CELL")]) for e in range(N)}
        if len(cells) != 1 or min(cells) < 1:
            raise DeviceObsRefused(f"device observation: the envs' lane cells {sorted(cells)} "
                                   f"are not one positive value (the lane grid's shape)")
        self.lane_shape = lane_grid_shape(c.h, c.w, cells.pop())
        gh, gw = self.lane_shape
        self.lane_grid = cp.full((N, self.T, gh, gw), -1, dtype=cp.int32)
        self.lane_own = cp.zeros((N, U, gh, gw), dtype=cp.uint8)
        self.aim = cp.full((N, self.T, U), AIM_CODES["AIM_NOT_ASKED"], dtype=cp.int32)
        self.lof = cp.zeros((N, self.n_rows, U), dtype=cp.uint8)
        self.ttw = cp.full((N, self.n_rows, U), -1, dtype=cp.int32)
        self.aim_faults = cp.zeros((N,), dtype=cp.int32)
        self.aims_brk = 0                     # THE BREAK hook (the check's), 0 in a real run
        # -- P8.4b: the list (its wired flags per door table row, static) -----
        from simulation.obs_spec_v1 import ENTRIES_BEYOND_SLOTS
        self.wired_cap = max(1, max(len(getattr(s, "_doors", ()) or ()) for s in c.sims))
        self.wired = cp.asarray(np.stack([wired_values(s, self.wired_cap) for s in c.sims]))
        self.n_entries = self.S + ENTRIES_BEYOND_SLOTS
        self.list_faults = cp.zeros((N,), dtype=cp.int32)
        self.list_brk = 0                     # THE BREAK hook (the check's), 0 in a real run
        # -- P8.4c: the four masks ---------------------------------------------
        self.mask_faults = cp.zeros((N,), dtype=cp.int32)
        self.masks_brk = 0                    # THE BREAK hook (the check's), 0 in a real run
        # -- P8.5a: the windows -------------------------------------------------
        check_window_sources(c, self.ctx)
        from simulation.obs_spec_v1 import BLOCK
        self.block_shape = (-(-c.h // BLOCK), -(-c.w // BLOCK))
        self.far_raw = cp.zeros((N, len(_PHYSICS)) + self.block_shape, dtype=cp.float64)
        # the decoder's aimed uid per row (-1: none): what sees_target is FROM
        # every window tile TO (the aimed entry's recorded centre); P8.6a's
        # decoder writes it, the check feeds it, a run without a decoder keeps -1
        self.aimed = cp.full((N, self.n_rows), -1, dtype=cp.int32)
        self.win_faults = cp.zeros((N,), dtype=cp.int32)
        self.win_brk = 0                      # THE BREAK hook (the check's), 0 in a real run
        self._null_decoder = None             # P8.6a: the decoder's columns with no decoder
        # -- P8.7: the device books (the reward, termination, the ended latch)
        self.books = device_books.DeviceBooks(self, self.reward_terms)
        c.attach_observation(self)
        # tick 0: who sees whom, the picture's first state, the first decision,
        # the books' first state
        self.vision()
        self.picture.start(picture_from_host)
        self.build_if_due()
        self.books.start()

    # -- the walk's call line ------------------------------------------------
    def stage(self) -> None:
        """The conductor's ``11'`` call line (and the build's tick 0): who sees
        whom for every env (EVERY tick: the team picture reads it every
        tick), then the blocks of every env due this tick. It runs where the
        reference's slot 11 asks ``vision.state_for`` first: after the tick
        advance, order expiry and the ruleset's tick end (the conversion), so
        positions, facings, deaths and conversions are the tick's own. P8.3:
        the team picture's update between the two (the reference's slot 11
        itself; every block reads the picture at the tick)."""
        if not (self.early_vision or self.early_picture):
            self.vision()
        if not self.early_picture:
            self.picture.update()
        self.build_if_due()
        # P8.7: the books after the picture (its bars) and the tick's blocks:
        # the ending tick's blocks are built as the reference builds its last
        # observation; the latch the books set skips every later tick
        self.books.update()

    # -- P8.2: who sees whom ---------------------------------------------------
    def vision(self, store_ptr=None) -> None:
        """``vision.compute`` for every env on the card into :attr:`sees`
        ``(N, U, U)`` (``sees[e, o, t]``: store row o is among
        ``seen_by[row t]``) and :attr:`visible` ``(N, T + 1, U)`` (slot s < T:
        marine team s; slot T: the zombie team), under each env's sight rule
        (its obs table row). One launch, on the tick stream, no sync.
        ``store_ptr``: the store it reads (default the conductor's; the WALL
        gate hands a perturbed copy)."""
        c = self.c
        store = int(c.store.device_ptr) if store_ptr is None else int(store_ptr)
        self.bp.obs_vision(int(self.N), store, int(c.U), int(c.store.shape[2]),
                           int(c.fields["solid"].data.ptr),
                           int(c.fields["sight_screen"].data.ptr), int(c.h), int(c.w),
                           int(self.table.data.ptr), int(self.table.shape[1]), int(self.T),
                           int(self.sees.data.ptr), int(self.visible.data.ptr),
                           int(self.vision_faults.data.ptr))
        self.visions += 1

    def lof_pairs(self, out=None, faults=None):
        """``line_of_fire.solid_in_way`` between the centre tiles of every
        (row a, row b) of every env's store, over the cover table: ``(N, U,
        U)`` uint8 CuPy (1 = solid in the way) and the ``(N,)`` fault bits.
        One launch on the tick stream (the gate's; P8.4a's line of fire is
        its own kernel over the same device function)."""
        cp, c = self.cp, self.c
        U = int(c.U)
        out = cp.zeros((self.N, U, U), dtype=cp.uint8) if out is None else out
        faults = cp.zeros((self.N,), dtype=cp.int32) if faults is None else faults
        self.bp.obs_lof_pairs(int(self.N), int(c.store.device_ptr), U, int(c.store.shape[2]),
                              int(c.fields["solid"].data.ptr), int(c.h), int(c.w),
                              int(self.cover.data.ptr), int(self.cover_cap),
                              int(self.cover.shape[2]), int(self.cover_count.data.ptr),
                              int(out.data.ptr), int(faults.data.ptr))
        return out, faults

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
        # P8.3: the vector columns of the team picture
        pic = self.picture
        self.bp.obs_picture_vector(
            int(self.N), store, int(c.U), int(c.store.shape[2]), int(self.rowmap.data.ptr),
            int(self.n_rows), int(self.S), int(self.T), int(self.table.data.ptr),
            int(self.table.shape[1]), int(self.teams.data.ptr), int(c.trow.ptr),
            int(self.params.data.ptr), int(self.ended.data.ptr), int(self.visible.data.ptr),
            int(pic.state.data.ptr), int(pic.W), int(b["vector"].data.ptr))
        # P8.4a: the lanes (and the lanes plane), aim_of, the line of fire
        self.aims(store)
        # P8.4b: the list, its mask and its uids (reads the P8.4a answers)
        self.build_list(store)
        # P8.4c: the four masks (read the list just built)
        self.build_masks(store)
        # P8.5a: the windows (sees_target reads the list just built)
        self.build_windows(store)
        # P8.6a: the decoder's columns (prev, prev_aim, on_next_tick) after the
        # vector, the list and the windows
        self.build_decoder_obs(store)
        self.builds += 1

    def build_decoder_obs(self, store_ptr=None) -> None:
        """P8.6a: the vector's ``prev`` block and the list's ``prev_aim`` /
        ``on_next_tick`` (``device_decoder.DeviceDecoder.build_obs``); with no
        decoder attached, over a state where no slot was ever decoded (prev
        reads -1) and :attr:`aimed` as it is (the check's stand-in feeds it)."""
        dec = self.c.decoder
        if dec is None:
            if self._null_decoder is None:
                from simulation.device_decoder import NullDecoderView
                self._null_decoder = NullDecoderView(self)
            dec = self._null_decoder
        dec.build_obs(store_ptr)

    def aims(self, store_ptr=None) -> None:
        """P8.4a for every env on its decision tick: the fire lanes of every
        marine team (:attr:`lane_grid` ``(N, T, GH, GW)``, draw_lanes' codes)
        and every owner's own lane (:attr:`lane_own` ``(N, U, GH, GW)``), the
        served lanes plane of ``near_u``, aim_of's uid per (team, store row)
        (:attr:`aim`), the line of fire and ticks_to_within per (row, store
        row) of a unit the row's team sees with a record (:attr:`lof`,
        :attr:`ttw`); faults into :attr:`aim_faults`. One call, five
        launches on the tick stream, no sync. ``store_ptr``: the store read
        (default the conductor's; the WALL gate hands a perturbed copy)."""
        c = self.c
        store = int(c.store.device_ptr) if store_ptr is None else int(store_ptr)
        pic = self.picture
        gh, gw = self.lane_shape
        self.bp.obs_aims(
            int(self.N), store, int(c.U), int(c.store.shape[2]), c.store.tables,
            int(self.rowmap.data.ptr), int(self.n_rows), int(self.S), int(self.T),
            int(self.table.data.ptr), int(self.table.shape[1]), int(c.trow.ptr),
            int(self.params.data.ptr), int(self.ended.data.ptr),
            int(c.fields["solid"].data.ptr), int(c.fields["sight_screen"].data.ptr),
            int(c.h), int(c.w), int(self.cover.data.ptr), int(self.cover_cap),
            int(self.cover.shape[2]), int(self.cover_count.data.ptr),
            int(self.visible.data.ptr), int(pic.state.data.ptr), int(pic.W),
            int(self.lane_grid.data.ptr), int(self.lane_own.data.ptr), int(gh), int(gw),
            int(self.aim.data.ptr), int(self.lof.data.ptr), int(self.ttw.data.ptr),
            int(self.buffers["near_u"].data.ptr), int(self.aim_faults.data.ptr),
            int(self.aims_brk))

    def build_list(self, store_ptr=None) -> None:
        """P8.4b for every env on its decision tick: ``list`` / ``list_mask``
        / ``list_uid`` (``obs_spec_v1.build_list``'s five blocks, each sorted
        by (squared centre distance, id) -- integer keys, ids unique within a
        block, so the order is total and the card takes it by successive
        minima -- and cut at its capacity after the sort), from the store, the
        device tables, the obs table, the device vision (``visible``,
        ``sees``), the picture's block, the P8.4a answers (``aim``, ``lof``,
        ``ttw``, ``lane_own``), the grenade / charge / door tables and the
        vacuum / ambient planes. One launch on the tick stream, no sync;
        faults into :attr:`list_faults`. ``store_ptr``: the store read
        (default the conductor's; the WALL gate hands a perturbed copy)."""
        c = self.c
        b = self.buffers
        store = int(c.store.device_ptr) if store_ptr is None else int(store_ptr)
        pic = self.picture
        gh, gw = self.lane_shape
        self.bp.obs_list(
            int(self.N), store, int(c.U), int(c.store.shape[2]), c.store.tables,
            int(self.rowmap.data.ptr), int(self.n_rows), int(self.S), int(self.T),
            int(self.table.data.ptr), int(self.table.shape[1]), int(self.teams.data.ptr),
            int(c.trow.ptr), int(self.params.data.ptr), int(self.ended.data.ptr),
            int(self.visible.data.ptr), int(self.sees.data.ptr), int(pic.state.data.ptr),
            int(pic.W), int(self.aim.data.ptr), int(self.lof.data.ptr), int(self.ttw.data.ptr),
            int(self.lane_own.data.ptr), int(gh), int(gw), c.grenades.buffer,
            c.charges.buffer, c.doors.buffer, c.door_tiles.buffer,
            int(c.fields["is_vacuum"].data.ptr), int(c.fields["is_ambient"].data.ptr),
            int(c.h), int(c.w), int(self.wired.data.ptr), int(self.wired_cap),
            int(self.n_entries), int(b["list"].data.ptr), int(b["list_mask"].data.ptr),
            int(b["list_uid"].data.ptr), int(self.list_faults.data.ptr), int(self.list_brk))

    def build_masks(self, store_ptr=None) -> None:
        """P8.4c for every env on its decision tick: the four masks
        (``action_spec_v1.build_masks`` at the START of the decision tick:
        ``mask_feet`` / ``mask_trigger`` all legal, ``mask_aim`` the wind-up's
        KEEP alone or the tokens, "walk" and the aimable entries of the list
        just built, ``mask_hand`` ONE unit-library gate per bit -- reload,
        melee at the tick's start, shove, use through
        ``unit_lib::reach_key`` over the live doors, plant, every throw kind
        -- the flashlight reserved); a row that is not valid reads
        ``default_masks``. One launch on the tick stream, no sync; faults
        into :attr:`mask_faults`. ``store_ptr``: the store read (default the
        conductor's; the WALL gate hands a perturbed copy)."""
        c = self.c
        b = self.buffers
        store = int(c.store.device_ptr) if store_ptr is None else int(store_ptr)
        self.bp.obs_masks(
            int(self.N), store, int(c.U), int(c.store.shape[2]), c.store.tables,
            int(self.rowmap.data.ptr), int(self.n_rows), int(self.S), int(c.trow.ptr),
            int(self.params.data.ptr), int(self.ended.data.ptr), int(b["list"].data.ptr),
            int(b["list_mask"].data.ptr), int(self.n_entries), c.doors.buffer,
            c.door_tiles.buffer, int(self.n_kinds), int(b["mask_feet"].data.ptr),
            int(b["mask_aim"].data.ptr), int(b["mask_trigger"].data.ptr),
            int(b["mask_hand"].data.ptr), int(self.mask_faults.data.ptr), int(self.masks_brk))

    def build_windows(self, store_ptr=None) -> None:
        """P8.5a for every env on its decision tick: ``build_windows``' eight
        physics planes of the near window (the tile's real-unit value over
        the soldier's own table values, squashed, one cast to float16) and of
        the far window (the block MAX of each plane, the wind of the block's
        first largest load: :attr:`far_raw`, then the same law), the sight
        planes ``watched`` (FROM every record's centre of the team TO each
        window tile, OR-ed) and ``sees_target`` (FROM each tile TO the
        recorded centre of the entry :attr:`aimed` names, when the list holds
        it past the mates), the far unit planes (mates at live centres,
        records at recorded centres split seen / not seen), and the vector's
        ``air`` block and ``felt_temp``. Reads the resident planes in their
        own dtype and currency; three launches on the tick stream, no sync;
        faults into :attr:`win_faults`. ``store_ptr``: the store read (default
        the conductor's; the WALL gate hands a perturbed copy)."""
        c = self.c
        b = self.buffers
        f = c.fields
        store = int(c.store.device_ptr) if store_ptr is None else int(store_ptr)
        pic = self.picture
        hb, wb = self.block_shape
        self.bp.obs_planes(
            int(self.N), store, int(c.U), int(c.store.shape[2]), c.store.tables,
            int(self.rowmap.data.ptr), int(self.n_rows), int(self.S), int(self.T),
            int(self.table.data.ptr), int(self.table.shape[1]), int(c.trow.ptr),
            int(self.params.data.ptr), int(self.ended.data.ptr),
            int(f["temperature"].data.ptr), int(f["fire"].data.ptr), int(f["gas"].data.ptr),
            int(f["gas"].shape[1]), int(f["atmosphere"].data.ptr), int(f["wind_x"].data.ptr),
            int(f["wind_y"].data.ptr), int(f["heat"].data.ptr),
            int(f["rad_flux_sweep"].data.ptr), int(f["solid"].data.ptr),
            int(f["sight_screen"].data.ptr), int(c.h), int(c.w), int(self.far_raw.data.ptr),
            int(hb), int(wb), int(self.visible.data.ptr), int(pic.state.data.ptr), int(pic.W),
            int(self.aimed.data.ptr), int(b["list_uid"].data.ptr),
            int(b["list_mask"].data.ptr), int(self.n_entries), int(b["near_f"].data.ptr),
            int(b["near_u"].data.ptr), int(b["far_f"].data.ptr), int(b["far_u"].data.ptr),
            int(b["vector"].data.ptr), int(self.win_faults.data.ptr), int(self.win_brk))

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
           "TEAM_FIELDS", "TABLE_SCALARS", "COVER_FIELDS", "sight_values", "cover_values",
           "check_teams", "rays_clear_device", "window_clear_device", "vision_state",
           "lane_values", "lane_grid_shape", "AIM_CODES", "AIM_FAULT_BITS", "AIM_BREAK_CODES",
           "AMMO_FIELDS", "LIST_FAULT_BITS", "LIST_BREAK_CODES", "wired_values",
           "MASK_FAULT_BITS", "MASK_BREAK_CODES", "WIN_FAULT_BITS", "WIN_BREAK_CODES",
           "WINDOW_PLANES", "check_window_sources", "window_values"]
