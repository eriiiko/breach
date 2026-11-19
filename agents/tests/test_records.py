"""The record reader and loader (agents/records.py): rung 2 note 2e,
sections 2 to 4 and 7. On small record files written in the game's own
format (``_records``). Each test names the property it protects and the
change that must break it."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from agents import records as RC
from agents.tests import _records as W


@pytest.fixture(scope="module")
def recs(tmp_path_factory):
    root = W.folder(tmp_path_factory.mktemp("records"))
    return root, RC.census(root, workers=1)


def _marks(records) -> Counter:
    """Every living decision of ``records`` by its mark."""
    out = Counter()
    for f in records:
        arrays, _ = RC.read(f.path)
        live = arrays["valid"] != 0
        out.update(arrays["vector"][..., 0][live].tolist())
    return out


def test_the_census_counts_files_matches_and_living_decisions(recs):
    """Protects: the census sees every file, puts each in its ship, and
    counts its living decisions from the records' own flag. Breaks when a
    file is skipped, a dead soldier's decision is counted, or a match is
    counted by file and team instead of by its name."""
    root, c = recs
    assert c["files"] == c["fit"] == 20 and not c["refused"]
    for lv in ("ship_a", "ship_b"):
        s = c["ships"][lv]
        assert s["files"] == 10 and s["matches"] == 10
        assert s["living"]["train"] + s["living"]["held"] == 10 * 32
        assert s["not_prefix"] == 0
    assert len(c["hash"]) == 64


def test_the_split_is_by_level_and_seed_never_by_file(recs):
    """Protects: every file of one (level, seed) -- both door sides, with and
    without the horde -- falls on one side, and the held side is exactly the
    seeds with ``seed % HELD_EVERY == 0``. Breaks when the split is made by
    file (a file's name or path deciding its side)."""
    _, c = recs
    parts = RC.split(c["records"])
    side_of = {}
    for s, files in parts.items():
        for f in files:
            assert side_of.setdefault((f.level, f.seed), s) == s
    assert {k for k, s in side_of.items() if s == "held"} == {
        k for k in side_of if k[1] % RC.HELD_EVERY == 0}
    assert all(len([f for f in c["records"] if (f.level, f.seed) == k]) == 2
               for k in side_of)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_pass_deals_every_living_training_decision_once(recs, workers):
    """Protects: one pass of the decision dealer deals every living decision
    of the training records exactly once, no dead soldier's and no held-back
    decision, in near-equal batches of at most ``batch``. Breaks when a
    decision is dealt twice or dropped (the buffer's bookkeeping), the
    ``valid`` flag is ignored, or a short last batch is cut."""
    _, c = recs
    train = RC.split(c["records"])["train"]
    held = RC.split(c["records"])["held"]
    d = RC.DecisionDealer(train, seed=3, batch=50, buffer_matches=2,
                          workers=workers)
    got, sizes = Counter(), []
    for b in d.batches(0):
        got.update(b["vector"][:, 0].tolist())
        sizes.append(len(b["vector"]))
    assert got == _marks(train)
    assert not set(got) & set(_marks(held))
    assert max(sizes) <= 50 and max(sizes) - min(sizes) <= 1


def test_the_same_seed_gives_the_same_order(recs):
    """Protects: the order of a pass is a function of the seed and the pass
    alone (a run can be repeated), whatever process unpacks first; another
    seed or pass deals another order. Breaks when the dealer takes files as
    the processes finish them or draws from an unseeded generator."""
    _, c = recs
    train = RC.split(c["records"])["train"]

    def order(seed, p, workers):
        d = RC.DecisionDealer(train, seed=seed, batch=40, buffer_matches=3,
                              workers=workers)
        return [b["vector"][:, 0].tolist() for b in d.batches(p)]
    a = order(7, 0, 2)
    assert a == order(7, 0, 2) == order(7, 0, 1)
    assert a != order(8, 0, 2)
    assert a != order(7, 1, 2)


def test_the_stretch_mode_runs_each_soldier_from_its_first_decision(recs):
    """Protects: in the memory mode every soldier's match is dealt in order
    from its first decision, in stretches cut by the trainer's rule (at
    most ``stretch``; a stretch that does not open the soldier's match
    continues, in the same lane, the stretch before it at the next
    decision), never a stretch started in mid-match from an empty memory;
    every living training decision once. Breaks when a lane's next stretch
    is another soldier's middle, a stretch is longer than the rule allows,
    or a soldier's first stretch is not flagged ``first``."""
    _, c = recs
    train = RC.split(c["records"])["train"]
    d = RC.StretchDealer(train, seed=1, lanes=3, stretch=4,
                         buffer_matches=2, workers=1)
    last = [None] * 3          # per lane: the mark of its last real decision
    got = Counter()
    for b in d.batches(0):
        S, L = b["kept"].shape
        assert S == 4
        for j in range(L):
            n = int(b["length"][j])
            assert b["kept"][:n, j].all() and not b["kept"][n:, j].any()
            if n == 0:
                continue
            marks = b["vector"][:n, j, 0]
            got.update(marks.tolist())
            t = b["t"][:n, j]
            assert (np.diff(t) == 1).all()
            if b["first"][0, j]:
                assert t[0] == 0
            else:
                assert last[j] is not None and marks[0] == last[j] + 10, \
                    "a stretch in mid-match that does not continue its lane"
            assert not b["first"][1:n, j].any()
            last[j] = marks[n - 1]
    assert got == _marks(train)


def test_a_record_that_does_not_fit_is_refused_by_name(tmp_path):
    """Protects: a record whose spec name, grenade rows, array shape or
    number format, or kind values do not fit today's layout is refused,
    naming the field (the guard for the day the observation changes again).
    Breaks when the check is skipped or a field is converted or padded."""
    cases = {
        "obs_spec": dict(over_meta={"obs_spec": "v0"}),
        "hand_names": dict(over_meta={"hand_names": ["nothing"]}),
        "vector": dict(over_arrays={"vector": np.zeros((20, 3, 96),
                                                       np.float32)}),
        "near_f": dict(over_arrays={"near_f": np.zeros(
            (20, 3) + RC.expected(3, 6)["near_f"][0][1:], np.float32)}),
    }
    for i, (name, kw) in enumerate(cases.items()):
        p = W.write(tmp_path / name, "ship_a", 1, 0, i + 1, **kw)
        arrays, meta = RC.read(p)
        with pytest.raises(RC.RecordRefused, match=name):
            RC.check_record(arrays, meta, p.name)
        with pytest.raises(RC.RecordRefused, match=name):
            RC.unpack(str(p))
        assert RC.census(tmp_path / name, workers=1)["fit"] == 0
    # a kind value past the network's rows
    p = W.write(tmp_path / "kind", "ship_a", 1, 0, 9)
    arrays, meta = RC.read(p)
    col = next(f for f in RC.O.VECTOR_COLUMNS if f.name == "prev[3]")
    arrays["vector"][0, 0, col.pos] = 999
    with pytest.raises(RC.RecordRefused, match=r"prev\[3\]"):
        RC.check_record(arrays, meta, p.name)
