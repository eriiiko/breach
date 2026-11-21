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


# ---------------------------------------------------------------------------
# #29 P8.7: the device books' host side (simulation.device_books)
# ---------------------------------------------------------------------------
def test_the_book_rows_are_the_profiles_own_terms_and_group_matrix():
    """Property: the obs table's BOOK_TERMS / BOOK_G rows are, bit for bit,
    every named profile's own terms (``reward_v1.profile_terms``, in
    ``device_books.TERM_NAMES`` order) and ``reward_v1.group_weights`` over
    ``rl_env.COUNTER_NAMES`` (the matrix ``EnvRunner._reward`` multiplies by),
    a run file's override included. Breaks when a weight reaches the kernel
    from anywhere but the reference's own profile objects, or a counter's
    row moves."""
    import numpy as np
    import rl_env
    from simulation import device_books, reward_v1
    for name in reward_v1.PROFILES:
        for over in (None, {"clock": -1.25}):
            terms = device_books.profile_terms(name, over)
            tb, gb = device_books.table_rows(terms)
            want_t = reward_v1.profile_terms(name, over)
            assert tb.view(np.float64).tolist() == [want_t[n] for n in device_books.TERM_NAMES]
            g = reward_v1.group_weights(want_t, rl_env.COUNTER_NAMES)
            assert np.array_equal(gb, g.reshape(-1).view(np.int64))
    ses, ctx, _T = _ship_session()
    terms = device_books.profile_terms("range")
    tab = device_obs.table_values(ses.sim, ctx, terms)
    tb, gb = device_books.table_rows(terms)
    assert np.array_equal(tab[_row("BOOK_TERMS"):_row("BOOK_TERMS") + len(tb)], tb)
    assert np.array_equal(tab[_row("BOOK_G"):_row("BOOK_G") + len(gb)], gb)


def test_the_counters_profile_is_refused_by_name_on_the_device():
    """Property: the device books refuse the ``"counters"`` reward profile
    by name (its float32 fold is numpy's pairwise sum, which a serial kernel
    cannot equal; design "P8 RE-CUT, VERSION 2") and an unknown profile.
    Breaks when the device would serve a fold it cannot hold equal."""
    import pytest
    from simulation import device_books
    with pytest.raises(device_books.BooksRefused, match="counters"):
        device_books.profile_terms("counters")
    with pytest.raises(ValueError):
        device_books.profile_terms("no_such_profile")


def test_the_named_fold_is_numpys_sequential_sum():
    """Property: ``EnvRunner._reward``'s named-profile fold,
    ``(counters[:, :, None] * G[None]).sum(axis=1)``, is a sequential sum of
    the K products from +0.0 (k = 0 first) -- the order the device books
    add in -- on random operands over many magnitudes, and a sum of -0.0
    products is +0.0. Breaks when numpy changes that reduction's order (the
    device fold would then differ in the last bit)."""
    import numpy as np
    rng = np.random.default_rng(87)
    for _ in range(200):
        c = rng.standard_normal((12, 32)) * 10.0 ** rng.integers(-3, 4, (12, 32))
        G = rng.standard_normal((32, 5)) * 10.0 ** rng.integers(-2, 3, (32, 5))
        got = (c[:, :, None] * G[None]).sum(axis=1)
        p = c[:, :, None] * G[None]
        seq = np.zeros((12, 5))
        for k in range(32):
            seq = seq + p[:, k, :]
        assert np.array_equal(got.view(np.uint64), seq.view(np.uint64))
    z = (np.ones((1, 2))[:, :, None] * np.full((2, 1), -0.0)[None]).sum(axis=1)
    assert np.signbit(z).tolist() == [[False]]


def test_the_old_counters_add_in_float32_after_one_cast():
    """Property: the old counters' ``counters[r, c] += dmg`` (a float32
    array, a Python float) is numpy 2's NEP 50 rule -- the damage cast to
    float32 once, then a float32 add -- the arithmetic the device books do.
    Breaks when numpy returns to adding in float64 and casting back."""
    import numpy as np
    rng = np.random.default_rng(7)
    a = np.zeros((1, 1), dtype=np.float32)
    for _ in range(20000):
        acc = np.float32(rng.random() * 1000)
        x = float(rng.integers(1, 1 << 30)) / 65536.0 * 1.37
        a[0, 0] = acc
        a[0, 0] += x
        assert a[0, 0] == np.float32(acc) + np.float32(x)


def test_the_books_layout_holds_every_section_once():
    """Property: ``device_books.layout`` lays its sections end to end (no
    overlap, no gap) and its width grows with the teams, the store rows and
    the roster rows. Breaks when a section is added without its offset
    following the one before (the compiled layout check then refuses)."""
    from simulation import device_books
    lay = device_books.layout(2, 12, 12)
    import numpy as np
    off = 0
    for _n, o, shape in lay:
        assert o == off
        off += int(np.prod(shape))
    assert off == device_books.width(2, 12, 12)
    assert device_books.width(3, 12, 12) > device_books.width(2, 12, 12)
    assert device_books.width(2, 13, 12) > device_books.width(2, 12, 12)
    assert device_books.width(2, 12, 13) > device_books.width(2, 12, 12)


# ---------------------------------------------------------------------------
# #29 P8.6b (part 1): every served column built; the room graph (device)
# ---------------------------------------------------------------------------
def test_every_served_key_and_column_is_built_on_the_card():
    """Property: with P8.6b every served key of the policy observation, the
    masks, the full view and the reward -- and every named column / plane of
    each -- is built by some device block (``device_obs.BUILT``); nothing the
    env serves is left unowned or unbuilt. Breaks when the spec gains a key,
    a column or a plane (a new idea is a new spec) without its device block,
    so the device backend would serve a zero where the farm serves a value."""
    served = _served_columns()
    built = device_obs.BUILT
    assert set(built) == set(served), (sorted(set(served) - set(built)),
                                       sorted(set(built) - set(served)))
    for k, names in served.items():
        if names is None:
            assert built[k] is None, k
        else:
            assert built[k] is not None and sorted(built[k]) == sorted(names), k


def _graph_level():
    import level_loader
    return level_loader.load("ship_demo", str(ROOT / "levels"))


def test_the_room_graphs_device_table_is_the_levels_own_graph():
    """Property: the room graph's device table of a sim is the level's graph
    itself -- the place plane answers ``room_graph.place_at`` at every tile,
    the grid is ``graph.grid``, the openings are the graph's rows in its
    order with their rectangles. Breaks when the table is built from a second
    map of places (another tie between overlapping openings, a room id off by
    one) or drops / reorders an opening."""
    import numpy as np
    from types import SimpleNamespace
    from simulation import room_graph
    lv = _graph_level()
    g = lv.rooms
    h, w = g.grid.shape
    sim = SimpleNamespace(level=lv, gmap=SimpleNamespace(material=np.zeros((h, w), np.int8)))
    has, place, grid, opens = device_obs.room_graph_values(sim, (h, w))
    assert has == 1 and place.dtype == np.int32 and grid.dtype == np.int32
    assert np.array_equal(grid, np.asarray(g.grid))
    want = np.array([[room_graph.place_at(g, y, x) for x in range(w)] for y in range(h)])
    assert np.array_equal(place, want)
    assert [tuple(int(v) for v in r) for r in opens] == [
        (int(o.id), int(o.x), int(o.y), int(o.w), int(o.h)) for o in g.openings]
    from simulation.unit_orders import ORDER_OPENING_BASE
    assert len(opens) > 0 and (place >= ORDER_OPENING_BASE).any()


def test_a_level_without_a_graph_stages_no_place_and_an_opening_off_the_map_is_refused():
    """Property: a sim without a room graph stages "no graph" (every tile
    NO_PLACE, no opening), and an opening whose rectangle leaves the map is
    refused by name (the host's walk would raise or wrap). Breaks when a
    graph-less env reads another env's places, or an opening is clipped
    silently."""
    import dataclasses
    import numpy as np
    import pytest
    from types import SimpleNamespace
    from simulation.unit_orders import NO_PLACE
    lv = _graph_level()
    h, w = lv.rooms.grid.shape
    bare = SimpleNamespace(level=SimpleNamespace(rooms=None),
                           gmap=SimpleNamespace(material=np.zeros((h, w), np.int8)))
    has, place, grid, opens = device_obs.room_graph_values(bare, (h, w))
    assert has == 0 and (place == NO_PLACE).all() and (grid == -1).all() and len(opens) == 0
    o0 = lv.rooms.openings[0]
    off = dataclasses.replace(o0, x=w - 1, w=2)
    g2 = dataclasses.replace(lv.rooms, openings=(off,) + tuple(lv.rooms.openings[1:]))
    bad = SimpleNamespace(level=SimpleNamespace(rooms=g2),
                          gmap=SimpleNamespace(material=np.zeros((h, w), np.int8)))
    with pytest.raises(device_obs.DeviceObsRefused, match="not inside the map"):
        device_obs.room_graph_values(bad, (h, w))
