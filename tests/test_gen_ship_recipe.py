"""Properties of the RECIPE PATH end to end: ``gen_ship.generate(recipe=)``
runs the grammar (``tools/ship_grammar.py``), the layout stage
(``tools/ship_layout.py``) and then the unchanged stage 2 (``embed``) and
gates (``validate``) (#136; docs/levelgen_archetype_recipe_2026-10-31.md §2,
docs/levelgen_layout_stage_2026-10-31.md).

Each test names the property it protects and the change that must break it.
No ship, room count or rectangle is pinned: only what every recipe ship must
satisfy. The in-suite sample is small; the 100-seed failure-rate table is a
reported measurement (the layout doc's "As built" section).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import level_loader  # noqa: E402
import ship_grammar as S  # noqa: E402
import ship_layout as L  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402

try:
    import breach_physics as _bp  # noqa: E402
except Exception:                  # pragma: no cover - a box without the build
    _bp = None
needs_bp = pytest.mark.skipif(_bp is None,
                              reason="needs the compiled breach_physics")

SEEDS = (0, 1, 2)


@pytest.fixture(scope="module")
def arch():
    return S.load_archetype("freighter")


@pytest.fixture(scope="module")
def plans(arch):
    return {(size, prof, seed): G.generate(seed, size, recipe=arch,
                                           profile=prof, zombies=4)
            for size in arch.sizes for prof in arch.profiles
            for seed in SEEDS}


def test_recipe_ships_pass_the_gates_and_the_validation_profile(arch, plans):
    """PROPERTY: every ship the recipe path returns passes stage 2's own
    gates UNCHANGED (airtight, mirrored, every formation cell floor, A*
    between the spawns and to the horde, exactly the graph's compartments,
    every compartment reachable) and the archetype's gates (the two slabs
    joined by a bot-walkable path with every airlock's outer door closed,
    the spawn distance, the cell count and the room count inside the
    ``[validation]`` ranges); it records which grammar and layout attempt
    passed, inside the two loops' bounds. BREAKS IF the layout emits a ship
    stage 2 must be bent to accept, the validation profile is not run, or
    (g, a) is not recorded."""
    for key, plan in plans.items():
        emb, g = plan.embedding, plan.graph
        assert G.validate(emb.tilemap, emb.spawn_rooms, mirrored=True,
                          graph=g) == [], key
        assert L.acceptance(arch, plan.quotient, g, emb) == [], key
        path = L.bot_path(emb)
        lo, hi = arch.validation["spawn_distance"][plan.size]
        assert path and lo <= len(path) - 1 <= hi
        lo, hi = arch.validation["rooms"][plan.size]
        assert lo <= len(g.rooms) <= hi
        assert 0 <= plan.g < S.G_MAX and 0 <= plan.attempt < G.MAX_ATTEMPTS
        assert plan.quotient.g == plan.g
        assert (plan.recipe, plan.profile) == ("freighter", key[1])
        assert plan.name == f"ship_freighter_{key[1]}_s{key[2]}_{key[0]}"


def test_a_recipe_ship_is_a_function_of_its_two_streams(arch, plans):
    """PROPERTY (archetype doc §2, determinism): the plan is reproduced
    exactly from the two documented streams -- the quotient graph from
    grammar attempt ``g`` (``SeedSequence([seed, 0, g])``), the layout AND
    stage 2 from ``SeedSequence([seed, 1, g, a])`` -- so the same request
    gives the same tilemap, entities and wires, and other seeds give other
    ships. BREAKS IF a draw moves to another stream, the layout and embed
    stop sharing one generator, or a set / dict order leaks into a draw."""
    maps = {}
    for (size, prof, seed), plan in plans.items():
        q = S.grammar_attempt(arch, size, prof, seed, plan.g, zombies=4)
        assert q.digest() == plan.quotient.digest()
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            [seed, 1, plan.g, plan.attempt])))
        lay = L.layout(q, rng)
        emb = G.embed(lay.graph, rng)
        assert np.array_equal(emb.tilemap, plan.tilemap), (size, prof, seed)
        assert [(e.id, e.class_name, e.fields) for e in emb.entities] == \
            [(e.id, e.class_name, e.fields) for e in plan.entities]
        assert emb.wires == plan.embedding.wires
        maps.setdefault((size, prof), []).append(plan.tilemap.tobytes())
    again = G.generate(1, "medium", recipe="freighter", profile="standard",
                       zombies=4)
    assert np.array_equal(again.tilemap,
                          plans[("medium", "standard", 1)].tilemap)
    for key, ms in maps.items():
        assert len(set(ms)) > 1, key


def test_the_recipe_less_path_is_v1_untouched():
    """PROPERTY: without ``recipe`` nothing changed -- ``generate`` is v1's
    loop on v1's stream (``SeedSequence([seed, attempt])``, ``plan_graph``
    then ``embed`` on one generator), its plan carries no recipe and keeps
    v1's name. (``levels/ship_demo`` regenerating identically is
    test_gen_ship's gate.) BREAKS IF the recipe wiring touches the
    recipe-less stream, defaults or naming."""
    for seed, size in ((3, "small"), (5, "medium")):
        plan = G.generate(seed, size, vents=True)
        rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([seed, plan.attempt])))
        graph = G.plan_graph(rng, size, 6, 4, True)
        emb = G.embed(graph, rng)
        assert np.array_equal(emb.tilemap, plan.tilemap)
        assert plan.recipe is None and plan.g == 0
        assert plan.name == f"ship_s{seed}_{size}"


def test_a_recipe_request_is_refused_before_any_draw(arch):
    """PROPERTY: the recipe path refuses, by name and before any attempt, a
    size the archetype does not allow, an unknown profile, v1's ``vents``
    flag (a recipe's ducts are its departments') and a horde its zombie
    section cannot seat. BREAKS IF such a request burns the attempt loops or
    silently falls back to v1's generator."""
    for kw, match in [(dict(size="small"), "does not allow size"),
                      (dict(profile="nightmare"), "no profile"),
                      (dict(vents=True), "vents"),
                      (dict(zombies=500), "cannot seat zombies")]:
        args = dict(size="medium", profile="standard")
        args.update(kw)
        size = args.pop("size")
        with pytest.raises(G.ShipGenError, match=match):
            G.generate(1, size, recipe=arch, **args)


def test_a_written_recipe_ship_loads_with_its_scaled_horde(arch, plans,
                                                           tmp_path):
    """PROPERTY (archetype doc §4): a recipe ship written through
    ``gen_ship.write`` (level_lib, nothing else) loads through the level
    loader's door -- spawn rooms, entities, wires, the hot rooms'
    temp_init -- and the horde it seats and its match.toml name is the
    PROFILE-SCALED count, whose zombie room holds it. BREAKS IF the layout
    emits something the loader refuses (an entity off its wall, a wire to
    nothing), or the base count is written where the scaled one belongs."""
    for prof, scale in (("easy", 0.5), ("hard", 1.5)):
        plan = plans[("medium", prof, 0)]
        assert plan.zombies == arch.scaled_zombies(prof, 4) == round(4 * scale)
        out = G.write(plan, tmp_path / plan.name)
        lv = level_loader.load(plan.name, levels_dir=str(tmp_path))
        assert lv.match.zombies == plan.zombies
        z = next(r for r in lv.spawn_rooms if r.team == ZOMBIE_TEAM)
        assert z.max_per_team == plan.zombies
        assert len(lv.wires) == len(plan.embedding.wires)
        assert (out / "temp_init.npy").is_file()        # hot engineering
        text = (out / "level.toml").read_text(encoding="utf-8")
        assert f"--recipe freighter --profile {prof}" in text


_HEADLESS = r"""
import sys
from pathlib import Path
root = Path.cwd()
sys.path[:0] = [str(root), str(root / 'src'), str(root / 'tools'),
                str(root / 'cpp' / 'build' / 'Release'), str(root / 'tests')]
import level_loader, level_airtight
import breach_physics as bp
from simulation import Simulation
from simulation.entities.actuators import AIRLOCK_OPEN_FAR, AIRLOCK_STATE_NAMES
from simulation.entities.door import DOOR_OPEN
from simulation.logic_nodes import AirlockControllerRuntime
from simulation.unit import Unit
d, name = sys.argv[1], sys.argv[2]
spots = [a.split(':') for a in sys.argv[3:]]
lv = level_loader.load(name, levels_dir=d)
assert level_airtight.check(name, levels_dir=d)
sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False)
ctrls = {e.id: e for e in sim.entities
         if isinstance(e, AirlockControllerRuntime)}
doors = {x.id: x for x in sim._doors}
for _ in range(48):                      # empty: every airlock idles
    sim.set_paused(False); sim.step()
idle = sorted({AIRLOCK_STATE_NAMES[c.state] for c in ctrls.values()})
for k, (p, x, y) in enumerate(spots):
    sim.add_unit(Unit('U%d' % k, x=int(x), y=int(y), team=0, footprint=3))
opened = {}
for t in range(900):                     # a unit in each vestibule: it cycles
    sim.set_paused(False); sim.step()
    for (p, _x, _y) in spots:
        if p not in opened and ctrls[p + '_ctrl'].state == AIRLOCK_OPEN_FAR \
                and doors[p + '_outer'].state == DOOR_OPEN:
            opened[p] = t + 1
    if len(opened) == len(spots):
        break
bad = [m for m in sys.modules
       if m in ('pyray', 'raylib') or m.split('.')[0] == 'renderer']
print('RESULT', idle, sorted(opened), bad)
"""


@needs_bp
def test_the_engineering_airlocks_cycle_for_a_unit(plans, tmp_path):
    """PROPERTY (layout doc §13 A1): the two engineering airlocks the layout
    stands in the lobbies (the vestibule on the LOBBY's side of the band
    wall, mirrored) are working interior airlocks -- the written level
    loads, passes the GameMap airtight check, both idle while empty, and
    each seals, equalises and OPENS ITS OUTER DOOR for a unit standing in
    its vestibule, headless. That is what the reachability gate's "an
    airlock's outer door is passable" rests on for every recipe ship.
    BREAKS IF the layout places the airlock where a mount tile, a probe's
    sample tile or the vestibule's wall is cut by another opening, or
    carves it on the wrong side."""
    plan = plans[("medium", "standard", 0)]
    g = plan.graph
    locks = G._airlocks(g)
    assert len(locks) == 2 and {g.rooms[g.edges[i].a].tags["role"]
                                for i, _al in locks} == {L.LOBBY}
    G.write(plan, tmp_path / plan.name)
    spots = [f"airlock_e{i}:{al.chamber[0] + 1}:{al.chamber[1] + 1}"
             for i, al in locks]
    proc = subprocess.run(
        [sys.executable, "-c", _HEADLESS, str(tmp_path), plan.name] + spots,
        cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, proc.stdout[-2000:]
    assert line[0] == (f"RESULT ['IDLE'] "
                       f"{sorted(s.split(':')[0] for s in spots)} []"), line
