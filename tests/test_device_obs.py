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
