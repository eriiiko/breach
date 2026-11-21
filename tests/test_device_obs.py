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
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
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


# ---------------------------------------------------------------------------
# #29 P8.1: the obs table (device_obs.table_values / team_values)
# ---------------------------------------------------------------------------
def _ship_session():
    from dataclasses import replace

    import breach_physics as bp
    import level_loader
    import rl_env
    from simulation.match_session import MatchSession
    from simulation.obs_spec_v1 import ObsContext
    lv = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = replace(lv.match, teams=tuple(replace(t, controller="bots") for t in lv.match.teams))
    ses = MatchSession(lv, spec, seed=1, breach_physics=bp)
    S = max(rl_env._team_slots(lv))
    T = len(ses.spec.teams)
    return ses, ObsContext.build(ses.sim, ses.spec, S, n_teams=T), T


def _row(name):
    return {n: o for n, o, _ in device_obs.table_layout()}[name]


def test_every_float_the_obs_table_carries_is_the_reference_objects_own_bits():
    """Property: every float the obs table carries (the three plane
    densities, every environment profile's temperature_max / pressure_min /
    pressure_max, the species' own profiles) is, bit for bit, the float the
    host builders read from the reference's OWN object (the episode's
    ``ObsContext``, the ``[env_profiles.*]`` rows, each unit's
    ``environment``), never a Q16 row. Breaks when a float is re-quantized,
    rounded through float32, read from another object, or a row lands at
    another offset."""
    import numpy as np
    from simulation import unit_schema as us
    from simulation.species import SPECIES_REGISTRY
    ses, ctx, _ = _ship_session()
    t = device_obs.table_values(ses.sim, ctx)
    bits = lambda x: int(np.array(float(x)).view(np.int64))  # noqa: E731
    for name, attr in (("POISON_MIN_DENSITY_BITS", "poison_min_density"),
                       ("TEARGAS_BLIND_DENSITY_BITS", "teargas_blind_density"),
                       ("SMOKE_DENSE_BITS", "smoke_dense")):
        assert int(t[_row(name)]) == bits(getattr(ctx, attr)), name
    pw = len(device_obs.PROFILE_FIELDS)
    fields = ("temperature_max", "pressure_min", "pressure_max")
    for block, objs in (("PROFILE", list(us.TABLES["env_profiles"]().values())),
                        ("SPECIES_PROFILE", [s.environment for s in SPECIES_REGISTRY.values()])):
        for i, p in enumerate(objs):
            for k, f in enumerate(fields):
                assert int(t[_row(block) + i * pw + k]) == bits(getattr(p, f)), (block, i, f)
    # each unit's own environment is a row the table carries, bit for bit
    for u in ses.sim.units:
        env = u.environment
        if u.env_profile_name is not None:
            i = us.table_row("env_profiles", u.env_profile_name)
            b = _row("PROFILE") + i * pw
        else:
            i = list(SPECIES_REGISTRY).index(u.species_id)
            b = _row("SPECIES_PROFILE") + i * pw
        for k, f in enumerate(fields):
            assert int(t[b + k]) == bits(getattr(env, f)), (u.id, f)


def test_every_integer_the_obs_table_carries_is_the_reference_value():
    """Property: the obs table's integers are the values ``build_vector``
    reads: the ``ObsContext`` scalars and status scales, the sim's OWN
    weapon rows in the unit schema's row order (mag size, range, rof, reload,
    the cone door of spread_deg), the loadout rows (magazines, explosives,
    charge, weapon row, the per-kind grenade counts in the hand family's
    order), and per team its n and loadout row. Breaks when a value is read
    from another table, a row order drifts from the schema's indexing (the
    store's ``weapon_id`` / ``loadout_name`` would index the wrong row), or
    a column's divisor moves."""
    from simulation import combat
    from simulation import unit_schema as us
    from simulation.loadouts import get_tables as lt
    from simulation.weapons import PLANT_WEAPON
    ses, ctx, T = _ship_session()
    sim = ses.sim
    t = device_obs.table_values(sim, ctx)
    for name, want in (("L_REF", ctx.l_ref), ("REF_ROF_TICKS", ctx.ref_rof_ticks),
                       ("REF_CONE_Q", ctx.ref_cone_q), ("V_FALL_Q", ctx.v_fall_q),
                       ("REACT_TICKS", ctx.react_ticks),
                       ("SHOVE_COOLDOWN_TICKS", ctx.shove_cooldown_ticks),
                       ("LIMIT_TICKS", ctx.limit_ticks),
                       ("C4_FUSE_MAX_TICKS", ctx.c4_fuse_max_ticks),
                       ("TURN_RATE_CLASS_Q", sim._marine_turn_rate_q16),
                       ("ZOMBIES", ctx.spec.zombies), ("N_TEAMS", T)):
        assert int(t[_row(name)]) == int(want), name
    rows = sim.weapons_tables.weapons.by_name
    assert int(t[_row("HAND_WINDUP_TICKS")]) == int(rows["hand_grenade"].throw_windup_ticks)
    assert int(t[_row("PLANT_TICKS")]) == int(rows[PLANT_WEAPON].plant_ticks)
    s0 = _row("STATUS_SCALE")
    assert tuple(int(x) for x in t[s0:s0 + len(ctx.status_scale)]) == tuple(ctx.status_scale)
    ww = len(device_obs.WEAPON_FIELDS)
    for name in us.TABLES["weapons"]():
        w, b = rows[name], _row("WEAPON") + us.table_row("weapons", name) * ww
        assert [int(x) for x in t[b:b + ww]] == [
            int(w.mag_size), int(w.range_tiles), int(w.rof_interval_ticks),
            int(w.reload_ticks), int(combat.cone_q16_of_deg(w.spread_deg))], name
    lw = len(device_obs.LOADOUT_FIELDS) - 1 + len(ctx.hand_rows)
    for name, lo in lt().loadouts.by_name.items():
        b = _row("LOADOUT") + us.table_row("loadouts", name) * lw
        assert int(t[b]) == int(lo.magazines) and int(t[b + 1]) == int(lo.explosives)
        assert int(t[b + 2]) == int(bool(lo.charge))
        assert int(t[b + 3]) == us.table_row("weapons", lo.weapon)
        assert [int(x) for x in t[b + 4:b + lw]] == [int(lo.grenades.get(r, 0))
                                                    for r in ctx.hand_rows]
    teams = device_obs.team_values(ses.spec, T)
    for i, team in enumerate(ses.spec.teams):
        assert int(teams[i, 0]) == int(team.n)
        assert int(teams[i, 1]) == us.table_row("loadouts", team.loadout)


def test_the_ammo_rows_are_what_the_list_reads():
    """Property (#29 P8.4b): the obs table's ammo block holds, per ``[ammo.*]``
    row in the unit schema's row order (the grenade / charge tables'
    ``ammo_name`` index), what ``obs_spec_v1._object_entry`` and
    ``Projectile`` read for that round: its place in ``ObsContext.obj_rows``
    (-1 when it is neither a hand nor a charge row), its payload radius (0
    without a payload: the entry's KeyError) and its travel speed as its own
    float64 bits. Breaks when the block follows another row order, a radius
    is read from another payload, or the speed is re-quantized."""
    import numpy as np
    from simulation import unit_schema as us
    ses, ctx, _ = _ship_session()
    sim = ses.sim
    t = device_obs.table_values(sim, ctx)
    aw = len(device_obs.AMMO_FIELDS)
    wt = sim.weapons_tables
    bits = lambda x: int(np.array(float(x)).view(np.int64))  # noqa: E731
    names = list(us.TABLES["ammo"]())
    assert names, "no ammo rows"
    kinds = set()
    for name in names:
        b = _row("AMMO") + us.table_row("ammo", name) * aw
        want_kind = ctx.obj_rows.index(name) if name in ctx.obj_rows else -1
        try:
            want_r = int(wt.payload_for_ammo(name).radius)
        except KeyError:
            want_r = 0
        assert int(t[b]) == want_kind, name
        assert int(t[b + 1]) == want_r, name
        assert int(t[b + 2]) == bits(wt.ammo.by_name[name].travel_speed_tiles_per_second), name
        kinds.add(want_kind)
    # every object row of the list is a row of the block
    assert set(range(len(ctx.obj_rows))) <= kinds


def test_the_float_dtype_lists_are_the_served_float_keys():
    """Property: ``device_obs.FLOAT16_KEYS`` / ``FLOAT32_KEYS`` (which cast
    of the float kit a key's values take) name exactly the served keys of
    that dtype. Breaks when a served key changes its float dtype, or a new
    float key lands without saying which cast it takes."""
    import numpy as np
    lay = _v1_layout()
    got = device_obs.served_shapes(lay.n_envs, lay.n_teams, lay.n_slots, lay.n_kinds)
    assert {k for k, (_, dt) in got.items() if np.dtype(dt) == np.float16} == \
        set(device_obs.FLOAT16_KEYS)
    assert {k for k, (_, dt) in got.items() if np.dtype(dt) == np.float32} == \
        set(device_obs.FLOAT32_KEYS)


# ---------------------------------------------------------------------------
# #29 P8.2: who sees whom's table rows, the cover table, the team slots
# ---------------------------------------------------------------------------
def _room_session(name):
    import tempfile

    import breach_physics as bp
    import level_loader
    sys.path.insert(0, str(ROOT / "tools"))
    import gen_training_rooms as TR
    from simulation.match_session import MatchSession
    d = tempfile.mkdtemp(prefix="p82_rooms_")
    for dr in TR.drafts():
        if dr.name == name:
            TR.write(dr, d)
    return MatchSession(level_loader.load(name, d), None, seed=1, breach_physics=bp)


def test_the_sight_rows_are_what_can_see_reads():
    """Property: the obs table's sight rows are what ``vision.can_see``
    reads: the match's rule (``Simulation.sight``'s index in
    ``match.SIGHT_RULES``: ship_demo's cone, a training room's all_round),
    ``vision.cos_threshold`` of the live cone half-angle and
    ``float(awareness_radius_tiles)``, as their own float64 bits. Breaks
    when the rule is read from another place than the sim, the threshold is
    re-derived (another cosine, a Q16 row), or a row lands at another
    offset."""
    import numpy as np
    from config import CFG
    from simulation import match, vision
    bits = lambda x: int(np.array(float(x)).view(np.int64))  # noqa: E731
    ses, ctx, _ = _ship_session()
    t = device_obs.table_values(ses.sim, ctx)
    assert ses.sim.sight == "cone"
    assert int(t[_row("SIGHT_RULE")]) == match.SIGHT_RULES.index("cone")
    assert int(t[_row("CONE_COS_BITS")]) == bits(
        vision.cos_threshold(CFG.onephase.vision_cone_half_deg))
    assert int(t[_row("AWARENESS_R_BITS")]) == bits(float(CFG.onephase.awareness_radius_tiles))
    room = _room_session("range_door")
    assert device_obs.sight_values(room.sim)["SIGHT_RULE"] == match.SIGHT_RULES.index("all_round")


def test_the_cover_table_is_the_sims_cover_and_sight_blocking_cover_is_refused():
    """Property: the cover table holds every ``sim.cover`` entity in its
    order, its rectangle as the CoverRuntime's own float64 bits, its alive
    and blocks_los; a cover that blocks SIGHT is refused by name (the
    device's lines do not apply it). Breaks when a cover is dropped,
    re-ordered or re-quantized, or a sight-blocking cover passes silently."""
    import numpy as np
    import pytest
    bits = lambda x: int(np.array(float(x)).view(np.int64))  # noqa: E731
    ses = _room_session("range_hall")
    cov = list(ses.sim.cover)
    assert len(cov) >= 1
    t = device_obs.cover_values(ses.sim)
    assert t.shape == (len(cov), len(device_obs.COVER_FIELDS))
    for i, c in enumerate(cov):
        assert [int(x) for x in t[i]] == [bits(c.x0), bits(c.y0), bits(c.x1), bits(c.y1),
                                          int(c.alive), int(c.blocks_los)]
    cov[0].blocks_los = True
    with pytest.raises(device_obs.DeviceObsRefused, match="blocks line of sight"):
        device_obs.cover_values(ses.sim)


def test_a_team_without_a_visible_slot_is_refused():
    """Property: a unit whose team is neither a marine team of the layout
    nor the zombie team is refused by name (the visible table has a slot per
    marine team and one for the zombies). Breaks when such a unit would be
    folded into another team's slot."""
    import pytest
    from simulation.factions import ZOMBIE_TEAM
    ses, _, T = _ship_session()
    device_obs.check_teams(ses.sim, T)
    assert any(int(u.team) == ZOMBIE_TEAM for u in ses.sim.units)
    ses.sim.units[0].team = T + 3
    with pytest.raises(device_obs.DeviceObsRefused, match="neither a marine team"):
        device_obs.check_teams(ses.sim, T)


# ---------------------------------------------------------------------------
# #29 P8.5a: the windows' table values and sources
# ---------------------------------------------------------------------------
def test_the_window_rows_are_what_physics_value_reads():
    """Property: the obs table's window rows are, bit for bit, what
    ``obs_spec_v1.physics_value`` and ``FieldPlanes`` read for every unit:
    its ``mitigation_for(u).resist_mult[POISON]`` (the zombie profile on
    ``is_zombie``, else the species' row the table carries), the sim's
    impulse table's ``wind_k_q``; and every constant the windows' kernel
    divides by is the host dequantizes' own currency. Breaks when a resist
    is read from another species or the neutral profile, re-quantized, the
    wind dial is read from another table, or a currency drifts from its
    ``*_fixed`` module."""
    import numpy as np
    from simulation import atmosphere_fixed, fire_fixed, gas_fixed
    from simulation import unit_schema as us
    from simulation.damage import POISON, mitigation_for
    bits = lambda x: int(np.array(float(x)).view(np.int64))  # noqa: E731
    ses, ctx, _ = _ship_session()
    t = device_obs.table_values(ses.sim, ctx)
    assert int(t[_row("WIND_K_Q")]) == int(ses.sim.impulse_table.wind_k_q)
    seen_zombie = False
    for u in ses.sim.units:
        want = bits(mitigation_for(u).resist_mult[POISON])
        if u.is_zombie:
            seen_zombie = True
            got = int(t[_row("ZOMBIE_RESIST_POISON_BITS")])
        else:
            got = int(t[_row("SPECIES_RESIST_POISON") + us.table_row("species", u.species_id)])
        assert got == want, u.id
    assert seen_zombie
    k = dict(device_obs._constants())
    assert k["OBS_Q16_ONE"] == gas_fixed._Q16_ONE_F
    assert k["OBS_TRACE_ONE"] == gas_fixed._TRACE_ONE_F
    assert k["OBS_FIRE_ONE"] == fire_fixed.FP_ONE_F
    assert k["OBS_ATM_ONE"] == atmosphere_fixed.FP_ONE_F
    assert k["OBS_WIND_SHIFT"] == atmosphere_fixed.FP_SHIFT


def test_the_felt_dials_the_card_reads_are_the_episodes():
    """Property: the windows' felt temperature runs the unit library's law
    over the DEVICE TABLES' Q16 dials, and the build refuses a batch whose
    dials are not ``exchange.heat_dials_q`` of the episode's ``ObsContext``
    (the observation's own object): equal on a shipped match, refused by
    name when the episode's dials differ. Breaks when the device tables'
    heat rows stop being the observation's dials, or the refusal is
    dropped."""
    from dataclasses import replace
    from types import SimpleNamespace

    import pytest
    ses, ctx, _ = _ship_session()
    fields = {n: SimpleNamespace(dtype=dt) for n, dt in device_obs.WINDOW_PLANES.items()}
    c = SimpleNamespace(fields=fields, sims=[ses.sim])
    device_obs.check_window_sources(c, [ctx])
    hot = replace(ctx, heat_dials=(ctx.heat_dials[0], ctx.heat_dials[1],
                                   ctx.heat_dials[2] * 2.0, ctx.heat_dials[3]))
    with pytest.raises(device_obs.DeviceObsRefused, match="heat dials"):
        device_obs.check_window_sources(c, [hot])
    del fields["rad_flux_sweep"]
    with pytest.raises(device_obs.DeviceObsRefused, match="not resident"):
        device_obs.check_window_sources(c, [ctx])
