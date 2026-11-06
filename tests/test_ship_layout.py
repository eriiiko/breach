"""Properties of the LAYOUT stage (``tools/ship_layout.py``, stage 1.5 of
family B; #136, docs/levelgen_layout_stage_2026-10-31.md with its §13 and its
"As built" section).

Each test names the property it protects and the change that must break it.
Nothing here pins how many rooms, departments or cycles a ship has, nor any
rectangle: the recipes are designed to grow and every draw is seeded. The
placement primitives are exercised on hand-built departments (fixtures, not
layouts the generator owes); the ship-level properties on graphs the grammar
draws over seeds x both sizes x every profile.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import levelgen_recipes as R  # noqa: E402
import ship_grammar as S  # noqa: E402
import ship_layout as L  # noqa: E402

SEEDS = range(10)
NEED = G.DOOR_W + 2 * L.CLEAR


def _rng(*key):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [int(k) for k in key])))


@pytest.fixture(scope="module")
def arch():
    return S.load_archetype("freighter")


@pytest.fixture(scope="module")
def ships(arch):
    """``(quotient graph, Layout)`` per (size, profile, seed): the grammar's
    accepted graph and layout attempt 0 of it, on the recipe path's own
    stream."""
    out = {}
    for size in arch.sizes:
        for prof in arch.profiles:
            for seed in SEEDS:
                q = S.plan_quotient(arch, size, prof, seed,
                                    zombies=(0, 3, 6)[seed % 3])
                out[(size, prof, seed)] = (q, L.layout(q, _rng(seed, 1, q.g,
                                                               0)))
    return out


def _air(rooms, shape) -> np.ndarray:
    grid = np.zeros(shape, dtype=bool)
    for rm in rooms:
        grid[rm.y:rm.y + rm.h, rm.x:rm.x + rm.w] = True
    return grid


def _thick_steel(air: np.ndarray) -> bool:
    """Is there a (VOID_MAX + 2)-square of steel? A wall is one tile, a void
    at most VOID_MAX: two floors are never further apart than that."""
    k = L.VOID_MAX + 2
    return bool(sliding_window_view(~air, (k, k)).all(axis=(2, 3)).any())


# ---------------------------------------------------------------------------
# The placement primitives, on hand-built inputs
# ---------------------------------------------------------------------------

def test_pack_fills_its_run_inside_every_range():
    """PROPERTY (the PACK primitive): the sizes lie inside their ranges and,
    with one wall tile between neighbours, fill the run leaving only what
    the maxima cannot reach, never more than the slack; a run no sizes can
    fill is a ``_Reject``. BREAKS IF pack lets a room leave its range, leaves
    steel it could have filled, or returns a short run silently."""
    ranges = [(6, 9), (6, 12), (7, 8)]
    for length in range(21, 36):
        for seed in range(5):
            if length > 31 + L.VOID_MAX:
                with pytest.raises(G._Reject):
                    L.pack(_rng(seed), length, ranges, L.VOID_MAX)
                continue
            sizes = L.pack(_rng(seed), length, ranges, L.VOID_MAX)
            assert all(lo <= s <= hi for s, (lo, hi) in zip(sizes, ranges))
            gap = length - sum(sizes) - (len(sizes) - 1)
            assert gap == max(0, length - 31)
    with pytest.raises(G._Reject):
        L.pack(_rng(0), 20, ranges, L.VOID_MAX)          # below the minima
    with pytest.raises(G._Reject):
        L.pack(_rng(0), 32, ranges, 0)                   # no slack allowed


SIDE = dict(w=(6, 10), h=(6, 12))
DEPTS = {
    "one_stub": L.Dept("d", (L.Stub("s", (6, 8), (10, 40)),),
                       L.End("e", (9, 30), (9, 16)),
                       tuple(L.Side(f"r{i}", **SIDE) for i in range(4))),
    "exit_and_corner": L.Dept(
        "d", (L.Stub("s", (6, 8), (10, 40)),),
        L.End("e", (9, 30), (9, 16), kid=L.Kid("e.k", (6, 14), (6, 12))),
        (L.Side("x", exit=True, **SIDE), L.Side("r1", **SIDE),
         L.Side("r2", kid=L.Kid("r2.k", (6, 10), (6, 13)), **SIDE),
         L.Side("r3", **SIDE))),
    "two_stub": L.Dept(
        "d", (L.Stub("s", (6, 8), (10, 40)), L.Stub("s2", (6, 8), (10, 40))),
        L.End("e", (9, 40), (9, 16)),
        (L.Side("r0", stub=0, **SIDE), L.Side("r1", stub=1, **SIDE),
         L.Side("r2", stub=0, **SIDE), L.Side("r3", stub=1, **SIDE))),
    "bare_stub": L.Dept("d", (L.Stub("s", (6, 8), (10, 40)),),
                        L.End("e", (6, 30), (9, 16),
                              kid=L.Kid("e.k", (6, 20), (6, 12)))),
    "promoted": L.Dept("d", (L.Stub("s", (6, 8), (10, 40)),),
                       L.End("e", (6, 30), (9, 16)),
                       (L.Side("r0", w=(6, 10), h=(12, 40)),
                        L.Side("r1", **SIDE)), promoted=True),
}


def _check_department(dept, rects, x, depth):
    """The contracts of a packed department (layout doc §3, §4)."""
    rooms = {rid: G.Room(id=i, x=r[0], y=r[1], w=r[2], h=r[3])
             for i, (rid, r) in enumerate(rects.items())}
    stubs = [rooms[s.id] for s in dept.stubs]
    ids = [s.id for s in dept.stubs] + [r.id for r in dept.sides] \
        + [r.kid.id for r in dept.sides if r.kid] \
        + ([dept.end.id] + ([dept.end.kid.id] if dept.end.kid else [])
           if dept.end else [])
    assert sorted(rects) == sorted(ids)                 # no room dropped
    for rm in rooms.values():                           # inside the rectangle
        assert 0 <= rm.x and rm.x + rm.w <= x and 0 <= rm.y \
            and rm.y + rm.h <= depth
    for spec, st in zip(dept.stubs, stubs):             # STRIP
        assert st.y == 0 and spec.across[0] <= st.w <= spec.across[1]
        assert spec.along[0] <= st.h <= spec.along[1]
    assert len({(st.w, st.h) for st in stubs}) == 1     # TWO-STUB: twins
    for r in dept.sides:                                # PACK
        rm, st = rooms[r.id], stubs[r.stub]
        assert r.w[0] <= rm.w <= r.w[1] and r.h[0] <= rm.h <= r.h[1]
        orient, _line, lo, hi = G.shared_wall(rm, st)
        assert orient == "v" and hi - lo + 1 == rm.h    # its whole length
        if r.exit:
            assert rm.x == 0                            # reaches the slab
        if r.kid:                                       # BEHIND (a corner)
            k = rooms[r.kid.id]
            assert r.kid.w[0] <= k.w <= r.kid.w[1]
            assert r.kid.h[0] <= k.h <= r.kid.h[1]
            wall = G.shared_wall(rm, k)
            assert wall and wall[3] - wall[2] + 1 >= r.kid.need
    if dept.end:                                        # END ROOM
        e = rooms[dept.end.id]
        assert dept.end.w[0] <= e.w <= dept.end.w[1]
        assert dept.end.h[0] <= e.h <= dept.end.h[1]
        for st in stubs:
            orient, _line, lo, hi = G.shared_wall(st, e)
            assert orient == "h" and hi - lo + 1 == st.w
        if dept.end.kid:                                # BEHIND (the end's)
            k = rooms[dept.end.kid.id]
            assert dept.end.kid.w[0] <= k.w <= dept.end.kid.w[1]
            assert dept.end.kid.h[0] <= k.h <= dept.end.kid.h[1]
            wall = G.shared_wall(e, k)
            assert wall and wall[3] - wall[2] + 1 >= dept.end.kid.need
        assert max(rm.y + rm.h for rm in rooms.values()) == depth
    # no two rooms share or touch a tile; steel is never VOID_MAX + 2 thick
    grid = np.zeros((depth + 2, x + 2), dtype=np.int32)
    for rm in rooms.values():
        assert not grid[rm.y + 1:rm.y + 1 + rm.h, rm.x + 1:rm.x + 1 + rm.w].any()
        grid[rm.y + 1:rm.y + 1 + rm.h, rm.x + 1:rm.x + 1 + rm.w] = rm.id + 1
    a, b, c, d = grid[:, :-1], grid[:, 1:], grid[:-1, :], grid[1:, :]
    assert not ((a != b) & (a > 0) & (b > 0)).any()
    assert not ((c != d) & (c > 0) & (d > 0)).any()
    assert not _thick_steel(grid[1:-1, 1:-1] > 0)


@pytest.mark.parametrize("name", list(DEPTS))
def test_a_packed_department_keeps_every_contract(name):
    """PROPERTY (the department packer: strip, pack, end room, behind,
    two-stub): for every arrangement of a hand-built department and the
    corners of its feasible x-extents and depths, each stub is a strip from
    the passageway inside its ranges (two stubs alike), each side room lies
    inside its ranges and touches ITS stub along its whole length, a declared
    exit room reaches the slab-side edge, the end room spans every stub's end
    and reaches the hull, each behind room touches its parent by at least its
    opening + 2, nothing overlaps or touches without a wall, nothing is
    dropped, and no steel is thicker than a wall plus VOID_MAX. BREAKS IF a
    primitive places a room off its stub, lets a range slip, leaves a void
    unbounded or loses a room."""
    dept = DEPTS[name]
    cfgs = L.dept_configs(dept)
    assert cfgs, name
    done = 0
    for cfg in cfgs:
        xs = [b for b in range(cfg.xs.bit_length()) if cfg.xs >> b & 1]
        for x in {xs[0], xs[len(xs) // 2], xs[-1]}:
            for depth in {cfg.dp[0], (cfg.dp[0] + cfg.dp[1]) // 2, cfg.dp[1]}:
                for seed in range(2):
                    rects = L.pack_department(dept, x, depth, _rng(seed), cfg)
                    _check_department(dept, rects, x, depth)
                    done += 1
    assert done >= 6


def test_a_promoted_stub_keeps_its_vestibule_clear():
    """PROPERTY: on a promoted department (its entry an airlock whose
    vestibule stands in the stub's first PROMOTED_CLEAR tiles) the first room
    of each column is long enough to open onto the stub beyond the
    vestibule, the stub is at least 7 across and long enough for the
    vestibule and a footprint. BREAKS IF the promoted case packs like a
    plain one (a side door would open into the vestibule's wall)."""
    dept = DEPTS["promoted"]
    for cfg in L.dept_configs(dept):
        for seed in range(4):
            xs = [b for b in range(cfg.xs.bit_length()) if cfg.xs >> b & 1]
            rects = L.pack_department(dept, xs[seed % len(xs)], cfg.dp[0],
                                      _rng(seed), cfg)
            _u, _v, across, along = rects["s"]
            assert across >= L.OWN_STUB_ACROSS_MIN
            assert along >= G.VESTIBULE_D + 1 + G.FORMATION_FOOTPRINT
            for z in cfg.zones:
                if z.rooms:
                    _u0, v0, _w, h = rects[z.rooms[0].id]
                    assert v0 == 0 and h - L.PROMOTED_CLEAR >= NEED


def test_the_feasible_sets_are_exactly_what_packs():
    """PROPERTY: an arrangement's ``xs`` and ``dp`` are EXACT -- the packer
    succeeds on every (x-extent, depth) inside them and rejects every one
    outside (so a row or skeleton computed from them never draws a
    department that cannot be packed, and never refuses one that could).
    BREAKS IF the closed-form sets and the packer's own arithmetic drift
    apart."""
    dept = DEPTS["exit_and_corner"]
    cfgs = L.dept_configs(dept)
    for cfg in cfgs:
        for x in range(10, 40):
            for depth in range(cfg.dp[0] - 2, cfg.dp[1] + 3, 3):
                ok = bool(cfg.xs >> x & 1) and cfg.dp[0] <= depth <= cfg.dp[1]
                if ok:
                    L.pack_department(dept, x, depth, _rng(x, depth), cfg)
                else:
                    with pytest.raises(G._Reject):
                        L.pack_department(dept, x, depth, _rng(x, depth), cfg)
    front = L.dept_front(dept)
    assert front and all(any((f.xs | c.xs) == f.xs and f.dp[0] <= c.dp[0]
                             and f.dp[1] >= c.dp[1] for f in front)
                         for c in cfgs)


# ---------------------------------------------------------------------------
# What every laid-out ship satisfies
# ---------------------------------------------------------------------------

def test_every_declared_edge_is_a_shared_wall_with_its_opening(ships):
    """PROPERTY (the layout's guarantee, layout doc §1): every DECLARED
    quotient edge is an edge of the ship between the rooms it names (room
    ``a`` first: an airlock's vestibule side), their shared wall at least the
    opening + one tile each side, the opening inside that wall; every edge
    not between two axis rooms has its twin, cut at the reflected tiles; and
    the ship has no other edge (the layout adds none, drops none). BREAKS IF
    a primitive leaves a declared contact short or missing, an ``at`` leaves
    its wall, the reflection loses or moves an opening, or the layout
    invents a door."""
    for key, (q, lay) in ships.items():
        g = lay.graph
        W = g.width
        rooms = {r.id: r for r in q.rooms}
        have = {(e.a, e.b, e.kind): e for e in g.edges}
        n = 0
        for e in q.edges:
            a, b = lay.ids[e.a][0], lay.ids[e.b][0]
            assert (a, b, e.kind) in have, (key, e)
            ge = have[(a, b, e.kind)]
            width = G.EDGE_KINDS[e.kind].width
            _o, _line, lo, hi = G.shared_wall(g.rooms[a], g.rooms[b])
            assert hi - lo + 1 >= width + 2 * L.CLEAR, (key, e)
            tiles = G._opening_tiles(g, ge)
            assert len(tiles) == (width if G.EDGE_KINDS[e.kind].stamps
                                  is not None else 0)
            n += 1
            if rooms[e.a].side == "axis" and rooms[e.b].side == "axis":
                continue
            ta, tb = lay.ids[e.a][1], lay.ids[e.b][1]
            assert (ta, tb, e.kind) in have, (key, e)
            twin = G._opening_tiles(g, have[(ta, tb, e.kind)])
            assert sorted(twin) == sorted((y, W - 1 - x) for y, x in tiles)
            n += 1
        assert n == len(g.edges), key


def test_every_room_is_inside_the_hull_its_ranges_and_its_width_class(ships):
    """PROPERTY: every quotient room is ONE rectangle strictly inside the
    hull, inside the size ranges the graph gave it (a corridor's ``across``
    is its short side), at least 6 tiles across unless it is the ``narrow``
    strip (exactly 3), and no two rooms share or touch a tile without a wall
    between them. BREAKS IF the layout stretches a room past its range to
    fill slack, lets a corridor go single-file, or packs two rooms into one
    compartment."""
    for key, (q, lay) in ships.items():
        g = lay.graph
        hx0, hy0, hx1, hy1 = g.hull
        assert len(lay.ids) == len(q.rooms)
        for r in q.rooms:
            for rid in set(lay.ids[r.id]):
                rm = g.rooms[rid]
                x0, y0, x1, y1 = rm.box
                assert hx0 < x0 and x1 < hx1 and hy0 < y0 and y1 < hy1
                short = min(rm.w, rm.h)
                if r.width == "narrow":
                    assert short == L.FLANK_STRIP_ACROSS, (key, r.id)
                else:
                    assert short >= R.MIN_WIDE, (key, r.id, rm.box)
                if r.size is None:
                    continue
                got = ((rm.w, rm.h) if r.size.axes == ("w", "h")
                       else (short, max(rm.w, rm.h)))
                for rng_, v in zip((r.size.a, r.size.b), got):
                    assert rng_ is None or rng_[0] <= v <= rng_[1], \
                        (key, r.id, got, r.size)
        ids = np.zeros((g.height, g.width), dtype=np.int32)
        for rm in g.rooms:
            sub = ids[rm.y:rm.y + rm.h, rm.x:rm.x + rm.w]
            assert not sub.any(), (key, rm.id)
            sub[:] = rm.id + 1
        a, b, c, d = ids[:, :-1], ids[:, 1:], ids[:-1, :], ids[1:, :]
        assert not ((a != b) & (a > 0) & (b > 0)).any(), key
        assert not ((c != d) & (c > 0) & (d > 0)).any(), key


def test_the_right_wing_is_the_mirror_and_the_axis_is_centred(ships):
    """PROPERTY: the wing is laid out once and reflected -- every wing-class
    room's twin is its reflection about the grid's centre and they name each
    other; every axis room is its own twin and symmetric; an opening between
    two axis rooms is centred on the axis and of even width (axis parity).
    BREAKS IF the two wings are drawn separately, an axis room drifts off
    the centre line, or an odd-width opening lands on the axis."""
    for key, (q, lay) in ships.items():
        g = lay.graph
        W = g.width
        assert g.mirror_x
        side = {r.id: r.side for r in q.rooms}
        for r in q.rooms:
            a, b = lay.ids[r.id]
            ra, rb = g.rooms[a], g.rooms[b]
            assert ra.mirror == b and rb.mirror == a
            assert (rb.x, rb.y, rb.w, rb.h) == (W - ra.x - ra.w, ra.y, ra.w,
                                                ra.h), (key, r.id)
            assert (a == b) == (r.side == "axis"), (key, r.id)
        for e in q.edges:
            if side[e.a] == "axis" and side[e.b] == "axis":
                ge = next(x for x in g.edges if (x.a, x.b, x.kind) == (
                    lay.ids[e.a][0], lay.ids[e.b][0], e.kind))
                xs = sorted(x for _y, x in G._opening_tiles(g, ge))
                assert len(xs) % 2 == 0
                assert xs == sorted(W - 1 - x for x in xs), (key, e)


def test_steel_voids_are_bounded(ships):
    """PROPERTY (layout doc §3): the hull's interior is rooms, one-tile
    walls and steel voids at most VOID_MAX thick -- there is no
    (VOID_MAX + 2)-square of steel anywhere inside the hull. BREAKS IF a
    department is padded to its row with a thick wall, voids of two
    neighbours add up, or a row or the band is left unfilled."""
    for key, (_q, lay) in ships.items():
        g = lay.graph
        hx0, hy0, hx1, hy1 = g.hull
        air = _air(g.rooms, (g.height, g.width))
        assert not _thick_steel(air[hy0 + 1:hy1, hx0 + 1:hx1]), key


def test_the_ship_reads_as_its_skeleton(ships):
    """PROPERTY (layout doc §2, §13 A2): the slab stands at the wing's outer
    end and the lobby at its band end, the one passageway runs between them,
    every wing department's rooms lie in ITS row (A above the passageway, B
    below) between the slab and the lobby, in their row order from the slab;
    the band's sections are stacked fore to aft in the graph's order,
    spanning the band, the hub department's block aft of every main section.
    BREAKS IF a department is moved to another row or order (the grammar
    deals rows), the lobby or slab stops spanning the wing, or the block
    leaves the band's aft end."""
    for key, (q, lay) in ships.items():
        g = lay.graph

        def rm(qid):
            return g.rooms[lay.ids[qid][0]]
        slab, pw, lobby = rm(L.SLAB), rm(L.PASSAGEWAY), rm(L.LOBBY)
        band = [rm(x) for x in q.band]
        assert slab.x == g.hull[0] + 1 and slab.x + slab.w + 1 == pw.x
        assert pw.x + pw.w + 1 == lobby.x
        assert lobby.x + lobby.w + 1 == band[0].x
        assert (slab.y, slab.h) == (lobby.y, lobby.h)
        assert slab.y < pw.y and pw.y + pw.h < slab.y + slab.h
        last = {}
        for d in sorted((d for d in q.departments if d.placement == "wing"),
                        key=lambda d: (d.row, d.position)):
            xs = [rm(i).x for i in d.rooms]
            for i in d.rooms:
                r = rm(i)
                assert pw.x <= r.x and r.x + r.w <= pw.x + pw.w, (key, i)
                if d.row == "A":
                    assert slab.y <= r.y and r.y + r.h < pw.y, (key, i)
                else:
                    assert r.y > pw.y + pw.h - 1 \
                        and r.y + r.h <= slab.y + slab.h, (key, i)
            assert min(xs) > last.get(d.row, -1), (key, d.name)
            last[d.row] = max(rm(i).x + rm(i).w for i in d.rooms)
        ys = [r.y for r in band]
        assert ys == sorted(ys) and band[0].y == g.hull[1] + 1
        assert band[-1].y + band[-1].h == g.hull[3]
        for a, b in zip(band, band[1:]):
            assert a.y + a.h + 1 == b.y
        roles = [q.room(x).role == L.SECTION for x in q.band]
        assert roles == sorted(roles, reverse=True) and roles[0]


def test_the_layout_is_a_function_of_its_stream(ships):
    """PROPERTY: a layout attempt is a pure function of (quotient graph,
    stream): the same stream gives the same rooms and openings, and other
    attempts of one graph are not all the same ship. BREAKS IF a draw
    escapes the generator (set order, a global rng) or the stream is
    ignored."""
    def sig(lay):
        return ([(r.x, r.y, r.w, r.h) for r in lay.graph.rooms],
                [(e.a, e.b, e.kind, e.at) for e in lay.graph.edges])
    for (size, prof, seed), (q, lay) in ships.items():
        if seed > 2:
            continue
        assert sig(L.layout(q, _rng(seed, 1, q.g, 0))) == sig(lay)
        others = [sig(L.layout(q, _rng(seed, 1, q.g, a))) for a in (1, 2, 3)]
        assert any(o != sig(lay) for o in others), (size, prof, seed)


def test_each_wing_has_its_own_ducts_and_every_room_its_lamps(ships):
    """PROPERTY: every room carries ``lamp_span = gen_ship.LAMP_SPAN`` (long
    rooms are lit along their length); a duct of the quotient wing exists
    once per wing -- no duct serves rooms of both wings -- with the
    quotient's filter, and an axis duct exists once. BREAKS IF the twin
    rooms keep the left wing's duct names (the two wings would breathe
    through one plenum) or the lamps tag is dropped."""
    seen = False
    for key, (q, lay) in ships.items():
        g = lay.graph
        W = g.width
        filters = dict(q.ducts)
        sides: dict = {}
        for rm in g.rooms:
            assert rm.tags.get("lamp_span") == G.LAMP_SPAN
            if "vent" not in rm.tags:
                continue
            mid = 2 * rm.x + rm.w
            where = "axis" if mid == W else "left" if mid < W else "right"
            for dn, _roles, _sh in G.vent_specs(rm.tags["vent"]):
                sides.setdefault(dn, set()).add(where)
                assert dn in g.ducts, (key, dn)
        for dn, where in sides.items():
            assert len(where) == 1, (key, dn, where)
            seen = seen or where == {"right"}
            base = dn[:-2] if where == {"right"} else dn
            assert g.duct_filter(dn) == filters[base], (key, dn)
    assert seen


# ---------------------------------------------------------------------------
# The cycle types and the budgets, each by its primitive
# ---------------------------------------------------------------------------

def _with(ships, pred):
    hits = [(k, q, lay) for k, (q, lay) in ships.items() if pred(q)]
    assert hits, "the sample draws no such ship: widen SEEDS"
    return hits


def test_a_retreat_loop_is_two_stubs_under_one_end_room(ships):
    """PROPERTY (two-stub, layout doc §4.1): a department with a retreat
    loop has two parallel stubs of one size, each opening onto the
    passageway, and its end room's wall runs across BOTH stubs' ends.
    BREAKS IF the second stub is packed as a side room or the end room
    spans only the first."""
    for key, q, lay in _with(ships, lambda q: any(
            len(d.stubs) == 2 for d in q.departments)):
        g = lay.graph
        for d in q.departments:
            if len(d.stubs) != 2:
                continue
            s1, s2 = (g.rooms[lay.ids[s][0]] for s in d.stubs)
            end = next(g.rooms[lay.ids[i][0]] for i in d.rooms
                       if q.room(i).attach == "end")
            pw = g.rooms[lay.ids[L.PASSAGEWAY][0]]
            assert (s1.y, s1.w, s1.h) == (s2.y, s2.w, s2.h), key
            assert s1.x < s2.x
            for s in (s1, s2):
                assert G.shared_wall(s, pw)[3] - G.shared_wall(s, pw)[2] + 1 \
                    == s.w
                assert G.shared_wall(s, end)[3] - G.shared_wall(s, end)[2] \
                    + 1 == s.w


def test_a_flank_pair_is_a_hull_strip_over_a_row_that_can_take_it(ships):
    """PROPERTY (hull strip, layout doc §4.2 / §13 A3): the flank pair's
    narrow leg is one corridor 3 across lying against the hull over the whole
    wing, entered from its long side by an opening down into the slab and
    one into the lobby; the slab, the lobby and that row give up the strip
    and its wall; and no hull-door room lies under it (it runs over the
    first row, A then B, that can take it). BREAKS IF the strip is entered
    through its end, is laid over the cargo hold's hull door, or the rooms
    under it keep their height."""
    for key, q, lay in _with(ships, lambda q: any(
            r.role == L.FLANK_STRIP for r in q.rooms)):
        g = lay.graph
        hx0, hy0, hx1, hy1 = g.hull
        strip = g.rooms[lay.ids[next(r.id for r in q.rooms
                                     if r.role == L.FLANK_STRIP)][0]]
        slab = g.rooms[lay.ids[L.SLAB][0]]
        lobby = g.rooms[lay.ids[L.LOBBY][0]]
        assert strip.h == L.FLANK_STRIP_ACROSS
        assert strip.x == slab.x and strip.x + strip.w == lobby.x + lobby.w
        top = strip.y == hy0 + 1
        assert top or strip.y + strip.h == hy1
        for other in (slab, lobby):
            orient, _l, lo, hi = G.shared_wall(strip, other)
            assert orient == "h" and hi - lo + 1 == other.w
        row = "A" if top else "B"
        assert lay.params["strip_row"] == row
        for d in q.departments:
            if d.row == row:
                assert not any("hull_door" in q.room(i).tags
                               for i in d.rooms), key


def test_an_airlock_bypass_shares_the_airlocks_wall(ships):
    """PROPERTY (layout doc §4.3): with a bypass the lobby and the hub carry
    two openings on their one shared wall, the airlock (its vestibule in the
    lobby) and the breakable, both placed explicitly and apart by at least a
    mount tile and a clearance tile (``gen_ship``'s own bypass check passes:
    the graph was built). BREAKS IF both openings are centred, overlap, or
    the hub section is not tall enough for both."""
    for key, q, lay in _with(ships, lambda q: any(
            c.type == "airlock_bypass" for c in q.cycles)):
        g = lay.graph
        hub = next(d.entry for d in q.departments if d.placement == "axis")
        a, b = lay.ids[L.LOBBY][0], lay.ids[hub][0]
        es = [e for e in g.edges if {e.a, e.b} == {a, b}]
        assert sorted(e.kind for e in es) == ["airlock", "breakable"], key
        al = next(e for e in es if e.kind == "airlock")
        br = next(e for e in es if e.kind == "breakable")
        assert al.a == a and al.at is not None and br.at is not None
        ya = {y for y, _x in G._opening_tiles(g, al)}
        yb = {y for y, _x in G._opening_tiles(g, br)}
        assert min(abs(p - r) for p in ya for r in yb) >= 3, key
        g.check()


def test_a_maintenance_ring_surrounds_the_hub(ships):
    """PROPERTY (the ring, layout doc §4.4 / §13 A1): the ring's fore and
    aft rooms are axis rooms spanning the band directly fore and aft of the
    hub's row; its side rooms are a mirrored pair of columns RING_DEPTH wide
    beside the hub; and the lobby's airlock lands on the aft-most side room
    (the vestibule still in the lobby). BREAKS IF the ring is laid out off
    the axis, the hub keeps the whole band, or the airlock still targets the
    hub."""
    for key, q, lay in _with(ships, lambda q: any(
            r.role == L.RING_SIDE for r in q.rooms)):
        g = lay.graph

        def rm(qid, twin=0):
            return g.rooms[lay.ids[qid][twin]]
        hub = rm(next(d.entry for d in q.departments
                      if d.placement == "axis"))
        fore = rm(next(r.id for r in q.rooms if r.role == L.RING_FORE))
        aft = rm(next(r.id for r in q.rooms if r.role == L.RING_AFT))
        sides = [r.id for r in q.rooms if r.role == L.RING_SIDE]
        band_w = fore.w
        assert fore.w == aft.w and fore.y + fore.h + 1 == hub.y
        assert hub.y + hub.h + 1 == aft.y
        assert hub.w == band_w - 2 * (L.RING_DEPTH + 1)
        for s in sides:
            left, right = rm(s), rm(s, 1)
            assert left.w == L.RING_DEPTH and left.x == fore.x
            assert left.x + left.w + 1 == hub.x
            assert right.x == hub.x + hub.w + 1
            assert hub.y <= left.y and left.y + left.h <= hub.y + hub.h
        lock = next(e for e in q.edges if e.kind == "airlock"
                    and e.a == L.LOBBY)
        assert lock.b == sides[0], key
        assert rm(sides[0]).y + rm(sides[0]).h == hub.y + hub.h


def test_a_promoted_department_is_entered_through_its_stub(ships):
    """PROPERTY (layout doc §4, the hard profile's armory): an own-cell wing
    department's entry is an airlock whose room ``a`` is its stub; the
    vestibule fits inside the stub, and every opening of a side room onto
    that stub lies beyond the vestibule's inner wall. BREAKS IF the airlock
    is carved on the passageway's side, or a side door opens into the
    vestibule."""
    def promoted(q):
        return any(e.kind == "airlock" and e.b == L.PASSAGEWAY
                   for e in q.edges)
    for key, q, lay in _with(ships, promoted):
        g = lay.graph
        for i, e in enumerate(g.edges):
            if e.kind != "airlock":
                continue
            stub = g.rooms[e.a]
            if stub.tags.get("role") == L.LOBBY:
                continue
            al = G._airlock(g, e)
            x0, y0, x1, y1 = stub.box
            assert all(x0 <= x <= x1 and y0 <= y <= y1 for y, x in al.carve)
            used = {y for y, _x in al.carve}
            for f in g.edges:
                if f is e or stub.id not in (f.a, f.b) or f.kind == "airlock":
                    continue
                wall = G.shared_wall(g.rooms[f.a], g.rooms[f.b])
                if wall[0] == "v":
                    ys = {y for y, _x in G._opening_tiles(g, f)}
                    assert not ys & used, (key, f)


# ---------------------------------------------------------------------------
# Feasibility: the same arithmetic, asked without a draw
# ---------------------------------------------------------------------------

def test_feasibility_is_exact_on_drawn_graphs(arch, monkeypatch):
    """PROPERTY (layout doc §13 A5): on graphs the grammar draws WITHOUT
    asking, ``layout_feasibility`` answers exactly what the layout does --
    a graph it accepts is laid out on every attempt tried (no packing
    reject), a graph it refuses cannot be laid out, and its refusal names
    what does not pack (a row with its departments, a department, the band
    or the depth). BREAKS IF the feasibility arithmetic and the layout's
    draws drift apart, or a refusal stops naming its cause."""
    real = L.layout_feasibility
    monkeypatch.setattr(L, "layout_feasibility", lambda size, **kw: None)
    rhash = {p: S.recipe_hash(arch, p) for p in arch.profiles}
    seen = set()
    for size in arch.sizes:
        for prof in arch.profiles:
            for seed in range(12):
                q = S.grammar_attempt(arch, size, prof, seed, 0, zombies=4,
                                      rhash=rhash[prof])
                why = real(size, graph=q)
                seen.add(why is None)
                if why is None:
                    for a in range(2):
                        L.layout(q, _rng(seed, 1, 0, a))
                else:
                    assert any(w in why for w in (
                        "row A", "row B", "rows A", "the band")), why
                    assert "crew_quarters" in why or "band" in why \
                        or "department" in why, why
                    with pytest.raises(G._Reject):
                        L.layout(q, _rng(seed, 1, 0, 0))
    assert seen == {True, False}


def test_the_grammar_only_returns_graphs_the_layout_packs(ships):
    """PROPERTY: the grammar's attempt loop ends on the layout's feasibility
    (the one call site), so every graph ``plan_quotient`` returns has a
    non-empty set of skeleton draws and each of those draws is realisable.
    BREAKS IF the call site is removed or answers for another size."""
    for (size, prof, seed), (q, _lay) in ships.items():
        assert L.layout_feasibility(size, graph=q) is None
        opts = L.skeleton_options(L.problem_of(q))
        assert opts
        if seed == 0:
            for a in range(4):
                L.layout(q, _rng(seed, 9, a))


def _copy(tmp_path: Path, rel: str, old: str, new: str) -> Path:
    root = tmp_path / "recipes"
    shutil.copytree(R.RECIPES_DIR, root)
    p = root / rel
    text = p.read_bytes().decode()
    assert text.count(old) == 1, (rel, old)
    p.write_bytes(text.replace(old, new).encode())
    return root / "archetypes" / "freighter.toml"


def test_the_archetype_door_refuses_at_minima_and_warns_at_maxima(
        arch, tmp_path):
    """PROPERTY (archetype doc §6 items 1-2): a recipe whose MINIMUM demand
    overflows a size (here: a third wing department on a medium ship, whose
    rows hold one each) is refused at load naming the size, the profile and
    the row; a recipe that only overflows at its MAXIMA loads and carries a
    warning per (size, profile) naming what overflows; and where even the
    maxima fit there is none. BREAKS IF the loader stops asking the layout,
    asks at the maxima and refuses, or drops the warning."""
    bad = _copy(tmp_path, "archetypes/freighter.toml",
                'department = "medbay"           # v1: without its isolation '
                'room (a room-level cell; §7.1)\ncount = { medium = [0, 0], ',
                'department = "medbay"\ncount = { medium = [1, 1], ')
    with pytest.raises(S.ArchetypeError) as err:
        S.load_archetype(bad)
    msg = str(err.value)
    assert "size 'medium'" in msg and "profile" in msg and "minima" in msg
    assert "row A (['" in msg or "row B (['" in msg, msg   # and who is in it
    assert set(arch.warnings) <= {(s, p) for s in arch.sizes
                                  for p in arch.profiles}
    for (size, _prof), why in arch.warnings.items():
        assert "maxima" in why and size in why, why
    for size in arch.sizes:
        for prof in arch.profiles:
            assert (L.maxima_warning(size, arch, prof) is None) \
                == ((size, prof) not in arch.warnings)


def test_a_department_that_can_never_pack_is_refused_by_the_layouts_twin():
    """PROPERTY (layout doc §4 "Loader twin"): the recipe loader's
    feasibility check IS the layout's (one implementation): every shipped
    department passes it, and a department whose minimum rooms cannot be
    dealt along its stub, or whose end room cannot stand across it, is
    refused with the layout's reason. BREAKS IF the loader keeps a second,
    weaker bound, or the twin starts judging a department by its maxima."""
    lib = R.load_recipes()
    for dept in lib.departments.values():
        assert L.department_feasibility(dept) is None, dept.name
        if dept.placement == "wing":
            assert L.dept_configs(L.recipe_dept(dept, "min"))
    from dataclasses import replace
    cq = lib.departments["crew_quarters"]
    many = replace(cq, rooms=tuple(
        replace(r, count=(12, 12)) if r.id == "bunk" else r
        for r in cq.rooms))
    assert "side rooms at minimum count" in L.department_feasibility(many)
    arm = lib.departments["armory"]
    wide = replace(arm, rooms=tuple(
        replace(r, size=R.Size((12, 30), r.size.b)) if r.id == "armory"
        else r for r in arm.rooms))
    assert "as wide as its stub" in L.department_feasibility(wide)


def test_a_request_the_archetype_cannot_seat_is_refused_before_any_draw(arch):
    """PROPERTY (archetype doc §4): the recipe path's ``check_request``
    refuses, by name, a size or profile the archetype does not allow, a team
    the slab cannot seat and a horde -- AFTER the profile's scale -- the
    zombie section cannot; a request it passes is seatable in some skeleton
    draw. BREAKS IF the scaled count is not the one checked, or the refusal
    moves into the attempt loop (32 rejected attempts instead of one
    message)."""
    L.check_request(arch, "medium", "standard", 6, 4)
    for kw, match in [
            (dict(size="small"), "does not allow size"),
            (dict(profile="nightmare"), "no profile"),
            (dict(max_per_team=0), "max_per_team"),
            (dict(max_per_team=400), "cannot seat max_per_team"),
            (dict(zombies=-1), "zombies"),
            (dict(zombies=400), "cannot seat zombies")]:
        args = dict(size="medium", profile="standard", max_per_team=6,
                    zombies=4)
        args.update(kw)
        with pytest.raises(G.ShipGenError, match=match):
            L.check_request(arch, args["size"], args["profile"],
                            args["max_per_team"], args["zombies"])
    # the scale is what is checked: a horde hard cannot seat, easy can
    p = G.SIZES["medium"]
    cols = (p.band_w[1] + G.FORMATION_GAP) // G.FORMATION_PITCH
    for z in range(1, 400):
        try:
            L.check_request(arch, "medium", "hard", 6, z)
        except G.ShipGenError:
            L.check_request(arch, "medium", "easy", 6, z)
            assert z >= cols
            break
    else:
        raise AssertionError("no horde is ever refused")
