"""agents/sizes.py -- every size the networks need, asked from the game's
own spec tables for a team size ``S`` (rung 2 design 2; plan section 4
item 2). No size is a literal in the network: :mod:`agents.network` reads
only a :class:`Sizes`.

What depends on ``S``: the list's length ``E = S + 33`` and with it the aim
head ``18 + E``. Network v1 does not care: every entry is read by the same
weights and the aim points at entries, so one set of weights runs at any
``S`` (design 2, "Network v1 does not care about S").

KINDS (observation spec 0.3: a kind is its row index in a named table, -1 =
none; the network embeds it). Each kind column gets an embedding of
``rows`` rows: row 0 is "none" (-1), row ``v + 1`` is value ``v``. The rows
come from the table the column names (:func:`kind_rows`). Four columns are
codes the observation's builders write with no named table (``aims_at``,
``zombie_state``, ``whose``, ``place_kind``): they get :data:`CODE_ROWS`
rows, a capacity. ``side`` is a team id, the zombies' ``ZOMBIE_TEAM`` (255)
included: it gets :data:`CODE_ROWS` rows, the zombies' the last one, and a
run of more marine teams than fit is refused. A value past its rows fails
loudly in the embedding (never clamped).

KIND PLANES (a code plane, e.g. ``door``: none / open / closed) are read as
one channel per code beside "none"; every other plane is one channel.
"""
from __future__ import annotations

from dataclasses import dataclass

from agents import _game  # noqa: F401  (the game's Python on the path)
from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O
from simulation.factions import ZOMBIE_TEAM
from simulation.reward_v1 import REWARD_GROUPS

#: The rows of a kind column whose codes have no named table in the game
#: (and of ``side``): room for the codes written today (at most 4) and a
#: few more; a code past it fails loudly.
CODE_ROWS = 8
#: The code columns read with :data:`CODE_ROWS` rows (no named table).
CODE_COLUMNS = ("aims_at", "zombie_state", "whose", "place_kind")

#: The codes of each KIND plane, read as one channel per code (``none``
#: is all channels 0). A KIND plane missing here is refused: a new code
#: plane is a new observation spec, and so a new network spec.
PLANE_CODES = {"door": (O.DOOR_CODE_OPEN, O.DOOR_CODE_CLOSED)}

#: The value network's extra near plane (``info["full_view"]``).
FULL_PLANE = "full_watched"


@dataclass(frozen=True)
class KindColumn:
    """One kind column of a table: its position, name, embedding rows and
    (``side`` only) the row ``ZOMBIE_TEAM`` is read into."""
    pos: int
    name: str
    rows: int
    zombie_row: int = -1


@dataclass(frozen=True)
class Window:
    """One map window as served: its float planes (``float16``) and its
    byte planes (``uint8``), each byte plane with its codes (``()`` = one
    channel as a number)."""
    f_planes: tuple
    u_planes: tuple          # ((name, codes), ...)
    side: int

    @property
    def channels(self) -> int:
        return len(self.f_planes) + sum(max(1, len(c))
                                        for _, c in self.u_planes)


@dataclass(frozen=True)
class Sizes:
    """The sizes of one run's layout, from the game's spec tables."""
    n_slots: int             # S (the largest team of the run)
    n_teams: int             # T (the value network's match block)
    n_kinds: int             # K: the hand-grenade rows
    vector: int              # V
    entry: int               # F
    n_entries: int           # E = S + 33
    heads: tuple             # (49, 18 + E, 2, 7 + K, 5)
    n_aim_tokens: int        # the aim's fixed tokens before the entries
    near: Window
    far: Window
    vector_kinds: tuple      # KindColumn of the soldier's vector
    entry_kinds: tuple       # KindColumn of a list entry
    entry_types: int         # the list's entry types (the per-type readers)
    full_entries: int        # the value network's list length
    full_entry: int          # its entry width
    full_kinds: tuple        # KindColumn of a full-view entry
    n_match: int             # full_match's width, 2 T + 3
    groups: tuple            # the reward groups, in the env's order
    obs_spec: str
    action_spec: str
    full_view: str

    @property
    def n_hands(self) -> int:
        return self.heads[3]

    @property
    def value_near_channels(self) -> int:
        return self.near.channels + 1           # + FULL_PLANE


def _table_counts(n_kinds: int) -> dict:
    """The number of VALUES of each kind column with a named source."""
    from simulation import unit_schema
    from simulation.events import CAUSE_NAMES
    from simulation.unit_orders import ORDER_WHAT
    from simulation.weapons import get_tables
    w = get_tables()
    types = max(O.TYPE_MATE, O.TYPE_SEEN, O.TYPE_RECORD, O.TYPE_OBJECT,
                O.TYPE_DOOR) + 1
    return {
        "type": types,
        # the loadout rows and one code past them: the zombies'
        # (ObsContext.zombie_class)
        "class": len(unit_schema.TABLES["loadouts"]()) + 1,
        "weapon": len(unit_schema.TABLES["weapons"]()),
        "throw_kind": int(n_kinds),
        "hit_cause": len(CAUSE_NAMES),
        "order_what": len(ORDER_WHAT),
        "obj_kind": len(w.hand_grenade_rows) + len(w.charge_rows),
        "knowledge": max(O.KNOW_SEEN, O.KNOW_RECORD, O.KNOW_UNKNOWN) + 1,
        # the decoder's six decoded choices (control_policy: feet direction
        # with N_DIRS = stand, speed, the collapsed aim index, trigger,
        # hand, argument)
        "prev[0]": A.N_DIRS + 1,
        "prev[1]": A.N_SPEEDS,
        "prev[2]": A.AIM_TOKENS + 1,
        "prev[3]": A.N_TRIGGER,
        "prev[4]": A.HAND_THROW0 + int(n_kinds),
        "prev[5]": A.N_ARG,
    }


def kind_rows(columns, n_kinds: int, n_teams: int) -> tuple:
    """The :class:`KindColumn` of every KIND column of ``columns`` (a spec
    table of ``obs_spec_v1.Feature``), refusing a kind with no source."""
    counts = _table_counts(n_kinds)
    out = []
    for f in columns:
        if f.denom != O.KIND:
            continue
        if f.name == "side":
            if int(n_teams) > CODE_ROWS - 2:
                raise ValueError(
                    f"agents.sizes: {n_teams} marine teams do not fit the "
                    f"side column's {CODE_ROWS} rows (none, teams, zombies)")
            out.append(KindColumn(f.pos, f.name, CODE_ROWS, CODE_ROWS - 1))
        elif f.name in CODE_COLUMNS:
            out.append(KindColumn(f.pos, f.name, CODE_ROWS))
        elif f.name in counts:
            out.append(KindColumn(f.pos, f.name, counts[f.name] + 1))
        else:
            raise ValueError(
                f"agents.sizes: kind column {f.name!r} (col {f.pos}) has no "
                f"source table here: a new observation spec needs a new "
                f"network spec")
    return tuple(out)


def _window(f_planes, u_planes) -> Window:
    us = []
    for name in u_planes:
        denom = O.PLANE_COLUMNS[name][0]
        if denom == O.KIND:
            if name not in PLANE_CODES:
                raise ValueError(f"agents.sizes: code plane {name!r} has no "
                                 f"codes here: a new network spec")
            us.append((name, tuple(PLANE_CODES[name])))
        else:
            us.append((name, ()))
    return Window(tuple(f_planes), tuple(us), O.WIN)


def sizes_for(n_slots: int, n_teams: int = 2, n_kinds: int | None = None) -> Sizes:
    """The sizes for a layout of ``n_slots`` roster slots (``S``) and
    ``n_teams`` marine teams (``T``); ``n_kinds`` defaults to the game's
    hand-grenade rows (as ``rl_env.make_layout`` reads them)."""
    if n_kinds is None:
        from simulation.weapons import get_tables
        n_kinds = len(get_tables().hand_grenade_rows)
    S, T, K = int(n_slots), int(n_teams), int(n_kinds)
    if S < 1 or T < 1:
        raise ValueError(f"agents.sizes: S = {S}, T = {T}")
    E = S + O.ENTRIES_BEYOND_SLOTS
    full_columns = O.ENTRY_COLUMNS + O.FULL_EXTRA_COLUMNS
    return Sizes(
        n_slots=S, n_teams=T, n_kinds=K, vector=O.V, entry=O.F,
        n_entries=E, heads=tuple(A.head_sizes(E, K)),
        n_aim_tokens=A.AIM_TOKENS,
        near=_window(O.NEAR_F_PLANES, O.NEAR_U_PLANES),
        far=_window(O.FAR_F_PLANES, O.FAR_U_PLANES),
        vector_kinds=kind_rows(O.VECTOR_COLUMNS, K, T),
        entry_kinds=kind_rows(O.ENTRY_COLUMNS, K, T),
        entry_types=_table_counts(K)["type"],
        full_entries=O.FULL_ENTRIES, full_entry=O.FULL_F,
        full_kinds=kind_rows(full_columns, K, T),
        n_match=len(O.full_match_columns(T)),
        groups=tuple(REWARD_GROUPS),
        obs_spec=O.SPEC_NAME, action_spec=A.ACTION_SPEC_NAME,
        full_view=O.FULL_VIEW_NAME,
    )


__all__ = ["CODE_ROWS", "CODE_COLUMNS", "PLANE_CODES", "FULL_PLANE",
           "KindColumn", "Window", "Sizes", "kind_rows", "sizes_for"]
