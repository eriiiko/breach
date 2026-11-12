"""#29 P3a -- THE UNIT SCHEMA (``simulation.unit_schema``): one table of what a
``Unit`` holds; ``Unit.__slots__`` generated from it; the unit digest built
from it.

PROPERTIES PROTECTED (each test names its own and the change that breaks it)
  a. Every attribute a unit holds is in exactly ONE of group S (the synced
     columns), group D (derived constants) and REFERENCE_ONLY; setting an
     undeclared attribute raises.
  b. Every group-D value equals its derivation from group S at every tick of
     a match, conversions included.
  c. Every group-S column round-trips through its integer encoding at every
     tick of a match.
  d. A column at its default adds no key to the record (absence-transparent).
  e. No REFERENCE_ONLY attribute is read on the continuous match path.
  f. The capacities (the status stack cap, the charge table) are never
     reached by the shipped scenarios, and overflow is refused and counted.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402,F401

import level_loader  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import charges, status, unit_schema as us  # noqa: E402
from simulation.entities.serialize import _pack_i64  # noqa: E402
from simulation.intents import AimIntent, MoveDirIntent, ThrowIntent  # noqa: E402
from simulation.unit import Unit  # noqa: E402


def _ship_demo():
    return level_loader.load("ship_demo", str(ROOT / "levels"))


def _bots_v1_spec(level):
    """ship_demo with bots_v1 on both teams: 264 ticks, five conversions."""
    return replace(level.match, teams=tuple(
        replace(t, controller="bots_v1") for t in level.match.teams))


# ---------------------------------------------------------------------------
# (a) the sort
# ---------------------------------------------------------------------------
def test_every_attribute_is_in_exactly_one_group_and_an_undeclared_one_raises():
    """Property (a): ``Unit.__slots__`` is exactly group S (by store) + group
    D + the optics overrides + REFERENCE_ONLY, no name in two groups, a unit
    has no ``__dict__``, and setting an undeclared attribute raises
    AttributeError at that line. Breaks if a name is declared in two groups
    (``slot_names`` refuses), if ``__slots__`` stops being the schema's
    (a ``__dict__`` reappears), or if code sets an attribute the schema does
    not declare (validated: ``self.foo = 1`` in ``Unit.__init__`` fails every
    construction)."""
    slots = set(Unit.__slots__)
    s = {c.store for c in us.COLUMNS}
    d = set(us.DERIVED) | set(us.OPTICS_OVERRIDES)
    r = set(us.REFERENCE_ONLY)
    assert not (s & d) and not (s & r) and not (d & r)
    assert slots == s | d | r and len(Unit.__slots__) == len(slots)
    for name in slots:
        assert us.group_of(name) in "SDR"
    u = Unit("u", 3.0, 4.0)
    assert not hasattr(u, "__dict__")
    with pytest.raises(AttributeError):
        u.not_a_column = 1
    for reason in us.REFERENCE_ONLY.values():
        assert reason.strip()


def test_a_unit_pickles_and_keeps_its_unset_optics_overrides():
    """Property (a), the slots' consequences: a slotted unit pickles (a farm
    worker or a snapshot may), and the four optics overrides stay UNSET so the
    stamp's ``getattr(u, name, map_default)`` reads the map default. Breaks if
    ``apply_defaults`` or ``__init__`` gives the overrides a value, or a slot
    loses its pickle support."""
    u = Unit("u", 3.0, 4.0)
    for name in us.OPTICS_OVERRIDES:
        assert getattr(u, name, "default") == "default"
    v = pickle.loads(pickle.dumps(u))
    assert us.digest_record(v) == us.digest_record(u)


# ---------------------------------------------------------------------------
# (b) + (c) every tick of two matches, conversions included
# ---------------------------------------------------------------------------
def _same_derived(name, got, want):
    if name == "inventory":
        return type(got) is type(want) and got == want
    return got == want


def _check_unit(u):
    for name, derive in us.DERIVED.items():
        assert _same_derived(name, getattr(u, name), derive(u)), \
            f"unit {u.id}: group-D {name} differs from its derivation"
    for c in us.COLUMNS:
        v = getattr(u, c.attr)
        e = c.encode(v)
        flat = e if isinstance(e, tuple) else (e,)
        assert len(flat) == us.column_width(c), c.name
        for x in flat:
            _pack_i64(c.name, x)                  # fits the int64 row
        assert c.encode(c.decode(e)) == e, f"{c.name}: encode∘decode"
        if c.until:
            continue                               # a float until P3e
        back = c.decode(e)
        if c.name == "statuses":
            assert us._enc_statuses(back) == e
        else:
            assert back == v, f"{c.name}: {v!r} -> {e!r} -> {back!r}"


@pytest.mark.parametrize("which", ["bots_v1", "bots"])
def test_derived_constants_and_round_trips_hold_at_every_tick(which):
    """Properties (b) and (c), over every tick of a ship_demo match (``bots``:
    2721 ticks, two conversions; ``bots_v1``: 264 ticks, five conversions):
    every group-D value equals its derivation from group S (the species,
    footprint, env profile, weapon table), and every group-S column encodes
    to integers of its declared width that fit int64 and decode back to the
    host value (a float column: ``encode(decode(e)) == e``). Breaks if a
    group-D value is written after spawn (validated: conversion setting
    ``u.mass = 0.0`` fails at the first converted tick), or an encoding loses
    a field (validated: ``_enc_throw`` dropping ``kind`` fails the
    round-trip)."""
    level = _ship_demo()
    spec = _bots_v1_spec(level) if which == "bots_v1" else None
    seen = {"conv": 0}

    def tick(sim):
        for u in sim.units:
            _check_unit(u)
        seen["conv"] = sum(1 for u in sim.units
                           if u.is_zombie and u.loadout_name is not None)
    run_match(level, spec, on_tick=tick)
    assert seen["conv"] >= 2


def test_every_intent_column_round_trips_when_set():
    """Property (c) for the intent columns a match may leave unset at a
    capture: every held verb and latch at a non-default value encodes and
    decodes to itself. Breaks if an intent encoding drops a field."""
    u = Unit("u", 3.0, 4.0)
    u.live_move_dir = MoveDirIntent(65536, -3, 40000)
    u.live_aim = AimIntent(-65536, 7)
    u.live_trigger = True
    u.pending_throw = ThrowIntent(1, 2, 30, None, 2)
    u.pending_plant = 120
    u.pending_reload = u.pending_shove = u.pending_melee = True
    u.pending_use = True
    u.spray_target = (3.25, 4.5)
    u.spray_aim_angle = 0.75
    _check_unit(u)
    u.pending_throw = ThrowIntent(1, 2, 30, 5 << 16, 1)
    _check_unit(u)


# ---------------------------------------------------------------------------
# (d) absence-transparent
# ---------------------------------------------------------------------------
_SAMPLES = {
    "life_state": "dead", "loadout_name": "rifleman",
    "env_profile_name": "default", "weapon_id": "k5_carbine",
    "melee_id": "rifle_butt", "charge_ammo": "grenade_frag",
    "magazines_weapon": "k5_carbine",
    "current_hp": 12.5,
    "live_move_dir": MoveDirIntent(1, 2, 3), "live_aim": AimIntent(1, 2),
    "pending_throw": ThrowIntent(1, 2, 3, None, 0),
    "pending_plant": 120, "spray_target": (1.0, 2.0),
    "spray_aim_angle": 0.5, "speed_ticks_per_tile": 9,
    "current_mag": 3,
}


def _sample(c):
    if c.name in _SAMPLES:
        return _SAMPLES[c.name]
    d = c.host_default()
    if isinstance(d, bool):
        return not d
    if isinstance(d, int):
        return d + 5
    if isinstance(d, list) and c.name != "statuses":
        return [1] + list(d[1:])
    raise AssertionError(f"no sample for {c.name}")


def test_a_column_at_its_default_adds_no_key():
    """Property (d): a stand-in object with no attributes records exactly
    the *always* keys (each at its default's encoding); every other column
    adds its key only while its value differs from the default. Breaks if a
    column's digest compares to the wrong default or hashes unconditionally
    (validated: marking ``flashlight_on`` *always* puts ``flashlight_on`` in
    the bare record)."""
    # §4.4a's *always* columns, written out (not read from the table, so a
    # column wrongly made *always* fails here).
    always = ["id", "life_state", "x_q", "y_q", "facing", "current_hp"]
    assert [c.key for c in us.COLUMNS if c.digest == us.DIGEST_ALWAYS] == always
    bare = us.digest_record(SimpleNamespace(facing_q=0))
    assert list(bare) == always
    for c in us.COLUMNS:
        if c.digest == us.DIGEST_ALWAYS or c.name == "statuses":
            continue
        u = SimpleNamespace(facing_q=0)
        object.__setattr__(u, c.attr, c.host_default())
        assert c.key not in us.digest_record(u), c.name
        if c.digest_of is None and c.name != "species_id":   # one species
            setattr(u, c.attr, _sample(c))
            assert c.key in us.digest_record(u), c.name
    u = SimpleNamespace(facing_q=0, facing_target_q=5)
    assert us.digest_record(u)["turn"] == 5
    st = status.StatusEffect(kind=status.STUNNED, magnitude_q16=0,
                             remaining_ticks=3)
    assert "statuses" in us.digest_record(SimpleNamespace(facing_q=0,
                                                          statuses=[st]))


# ---------------------------------------------------------------------------
# (e) REFERENCE_ONLY is not read on the continuous match path
# ---------------------------------------------------------------------------
class _Untouchable:
    """Raises on any use: truth, length, iteration, comparison, arithmetic,
    item access, attribute access."""

    def __init__(self, name):
        object.__setattr__(self, "_n", name)

    def _boom(self, *a, **k):
        raise AssertionError(f"REFERENCE_ONLY {self._n} used on the "
                             f"continuous match path")

    def __getattr__(self, item):
        self._boom()

    __bool__ = __len__ = __iter__ = __eq__ = __ne__ = __lt__ = __gt__ = \
        __le__ = __ge__ = __hash__ = __getitem__ = __setitem__ = \
        __contains__ = __add__ = __radd__ = __iadd__ = __sub__ = __isub__ = \
        __int__ = __index__ = __float__ = __call__ = _boom


#: The R attributes the continuous path TOUCHES without reading them into a
#: decision, each perturbed (not poisoned) and the match shown unmoved:
#: ``recent_attackers`` / ``impulse_books`` are written in place on every
#: path (a diagnostic tally and the momentum books); ``path_tick_offset`` is
#: bumped for a downed unit by slot 3's WEGO branch, which reads it only to
#: index ``move_path`` -- EMPTY on the continuous path, a length the branch
#: reads (so ``move_path`` stays ``[]``, its continuous-path invariant,
#: asserted at the end); ``name`` is prefixed at a conversion. And two WEGO
#: lists the continuous path reads only for EMPTINESS, so they keep their
#: empty value (asserted still empty at the end) -- ``move_path`` (slot 3's
#: WEGO branch takes its length) and ``orders`` (slot 4's
#: ``process_shooting`` looks for a fire order in it on every path).
_EMPTY = ("move_path", "orders")
_PERTURBED = {
    "recent_attackers": lambda: {9999: 77},
    "impulse_books": lambda: [123456789] * 8,
    "path_tick_offset": lambda: 10 ** 6,
    "name": lambda: "perturbed",
}


def _play(level, spec, mutate):
    from simulation.match_session import MatchSession
    s = MatchSession(level, spec)
    if mutate:
        for u in s.sim.units:
            for name in us.REFERENCE_ONLY:
                if name in _EMPTY:
                    continue
                if name in _PERTURBED:
                    object.__setattr__(u, name, _PERTURBED[name]())
                else:
                    object.__setattr__(u, name, _Untouchable(name))
    while not s.is_terminal():
        s.tick()
    if mutate:
        assert all(u.move_path == [] and u.orders == [] for u in s.sim.units)
    return s.result(), s.digests()


@pytest.mark.parametrize("which", ["bots_v1", "bots"])
def test_reference_only_is_not_read_on_the_continuous_match_path(which):
    """Property (e): a ship_demo match (bots_v1 on both teams: 264 ticks,
    five conversions; bots: 2721 ticks, two) with every REFERENCE_ONLY attribute of every unit replaced
    by an object that raises on any use -- bar the four the path writes
    without deciding on them (``_PERTURBED``), given garbage values instead,
    and ``move_path`` / ``orders``, which stay empty -- plays to the SAME result and the
    same final field and tick digests. Breaks if a rule on the continuous
    path reads WEGO / onephase state or a diagnostic (validated: a rule
    reading ``u.ap`` in slot 1b raises; an ``impulse_books`` value feeding
    the integration moves the digests)."""
    level = _ship_demo()
    spec = _bots_v1_spec(level) if which == "bots_v1" else None
    assert _play(level, spec, True) == _play(level, spec, False)


# ---------------------------------------------------------------------------
# (f) capacities
# ---------------------------------------------------------------------------
def test_the_status_stack_cap_refuses_and_counts_the_overflow():
    """Property (f): a stacking kind holds at most ``STATUS_STACK_CAP``
    instances; one more is refused (``None``, nothing changes) and counted in
    ``status.OVERFLOW_REFUSED``; the status block's width covers every kind
    at its cap. Breaks if the door is removed (the ninth poison appends) or
    the width stops covering the registry."""
    u = Unit("u", 3.0, 4.0)
    before = dict(status.OVERFLOW_REFUSED)
    try:
        for _ in range(us.STATUS_STACK_CAP):
            assert status.apply_status(u, status.POISONED, 1.0, 5) is not None
        assert status.apply_status(u, status.POISONED, 1.0, 5) is None
        assert len(u.statuses) == us.STATUS_STACK_CAP
        assert status.OVERFLOW_REFUSED[status.POISONED] == \
            before.get(status.POISONED, 0) + 1
        for k in range(status.N_STATUS_KINDS):
            status.apply_status(u, k, 1.0, 5)
        us.COLUMN_BY_NAME["statuses"].encode(u.statuses)   # fits the block
    finally:
        status.OVERFLOW_REFUSED.clear()
        status.OVERFLOW_REFUSED.update(before)


def test_the_charge_table_refuses_and_counts_a_plant_when_full():
    """Property (f): ``charges.place`` admits at most ``CHARGE_CAPACITY``
    live charges; the next is refused (nothing planted) and counted in
    ``sim.charges_refused``. Breaks if the door is bypassed."""
    sim = SimpleNamespace(planted_charges=[], charges_refused=0)
    for i in range(us.CHARGE_CAPACITY):
        assert charges.place(sim, charges.PlantedCharge(i, 0, "c4", 0, 10))
    assert not charges.place(sim, charges.PlantedCharge(0, 0, "c4", 0, 10))
    assert sim.charges_refused == 1
    assert len(sim.planted_charges) == us.CHARGE_CAPACITY


def test_the_shipped_matches_never_reach_a_capacity():
    """Property (f): over the pinned ship_demo match (bots, 2721 ticks) and
    its bots_v1 twin no unit holds more than half the stack cap of any kind
    or more statuses than the block's width, the charge table never fills,
    and nothing was refused. Breaks if a shipped row starts stacking a kind
    past the cap or a match plants past the table."""
    level = _ship_demo()
    peak = {"stack": 0, "charges": 0}
    before = sum(status.OVERFLOW_REFUSED.values())

    def tick(sim):
        for u in sim.units:
            kinds = [st.kind for st in u.statuses]
            for k in set(kinds):
                peak["stack"] = max(peak["stack"], kinds.count(k))
            assert len(kinds) <= us.status_capacity()
        peak["charges"] = max(peak["charges"],
                              sum(1 for c in sim.planted_charges if c.live))
        assert sim.charges_refused == 0
    for spec in (None, _bots_v1_spec(level)):
        run_match(level, spec, on_tick=tick)
    assert peak["stack"] <= us.STATUS_STACK_CAP // 2
    assert peak["charges"] < us.CHARGE_CAPACITY
    assert sum(status.OVERFLOW_REFUSED.values()) == before
