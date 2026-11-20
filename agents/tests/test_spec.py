"""Properties of the sizes and of the spec record (agents/sizes.py,
agents/spec.py): every size from the game's tables; a network refused by a
layout it was not made for (plan section 4's test table, last row). Each
test names the property and the change that must break it."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from agents import network
from agents.sizes import CODE_ROWS, kind_rows, sizes_for
from agents.spec import SpecMismatch, SpecRecord, check_fits, reduced, v1
from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O
from simulation.reward_v1 import REWARD_GROUPS


@pytest.mark.parametrize("S", [1, 3, 6])
def test_every_size_is_the_games(S):
    """Protects: the sizes are asked from the game's spec tables (the list
    length, the heads, the vector and entry widths, the planes, the full
    view, the reward groups, the served formats), never literals here.
    Breaks when: a size is hard-coded in agents/ and the game's table moves
    (e.g. a head size typed in, a plane list copied)."""
    s = sizes_for(S)
    from simulation.weapons import get_tables
    K = len(get_tables().hand_grenade_rows)
    assert s.n_entries == S + O.ENTRIES_BEYOND_SLOTS
    assert s.heads == tuple(A.head_sizes(s.n_entries, K))
    assert (s.vector, s.entry, s.full_entry) == (O.V, O.F, O.FULL_F)
    assert s.near.f_planes == O.NEAR_F_PLANES
    assert tuple(n for n, _ in s.far.u_planes) == O.FAR_U_PLANES
    assert s.groups == tuple(REWARD_GROUPS)
    assert s.n_match == 2 * s.n_teams + 3
    shapes = O.shapes(1, 1, S)
    for k, (dt, tail) in network.served_formats(s).items():
        if k in shapes:
            assert np.dtype(shapes[k][1]) == dt
    kinds = {f.name for f in O.VECTOR_COLUMNS if f.denom == O.KIND}
    assert {k.name for k in s.vector_kinds} == kinds


def test_a_kind_column_with_no_source_is_refused():
    """Protects: a kind column the sizes cannot find a table for is
    refused, so a new observation spec cannot slip into network v1 with a
    guessed vocabulary. Breaks when: unknown kinds get a default size."""
    col = (O.Feature(0, "a_kind_nobody_named", O.KIND),)
    with pytest.raises(ValueError, match="no source"):
        kind_rows(col, 6, 2)


def test_too_many_teams_for_the_side_column_are_refused():
    """Protects: a team id that would land on the zombies' row of the
    ``side`` kind is refused at the sizes. Breaks when: the side column's
    rows stop being checked against the run's teams."""
    sizes_for(1, n_teams=CODE_ROWS - 2)
    with pytest.raises(ValueError, match="side"):
        sizes_for(1, n_teams=CODE_ROWS - 1)


def test_the_record_refuses_a_network_made_for_another_spec(sizes6):
    """Protects: a network carries a record of the network, observation
    and action specs, the full view, the grenade kinds, the planes and the
    kinds it fits; the check accepts its own layout at any team size and
    refuses any other, naming the field.
    Breaks when: a saved network could be loaded into the wrong layout (a
    field dropped from the record or the check)."""
    net = network.SoldierNet(reduced(), sizes6)
    check_fits(net.record, SpecRecord.of(v1(), sizes_for(1)))
    want = SpecRecord.of(v1(), sizes6)
    for field, other in (("observation", "v2"), ("action", "v2"),
                         ("network", "v2"), ("full_view", "full_v2"),
                         ("n_kinds", sizes6.n_kinds + 1),
                         ("near_planes", want.near_planes[:-1])):
        with pytest.raises(SpecMismatch, match=field):
            check_fits(net.record, dataclasses.replace(want, **{field: other}))
    val = network.ValueNet(reduced(), sizes6)
    with pytest.raises(SpecMismatch, match="n_teams"):
        check_fits(val.record, SpecRecord.of(v1(), sizes_for(6, n_teams=3),
                                             value=True))
    with pytest.raises(SpecMismatch):
        check_fits(net.record, val.record)
