"""#29 P3b -- the keyed match draws (``simulation/draws.py``), the one salt
enum (``simulation/rng_salts.py``) and the refusing ``sim.rng`` stand-in.

Properties, one per test (each names the change that must break it):

* LAYOUT: a draw is the first word of Philox-4x32-10 at key ``(seed lo, seed
  hi)``, counter ``(tick, index, salt, id)`` -- breaks if the packing moves
  a word, drops the seed's high half or uses another output word.
* DISTINCT COUNTERS: two draws differing in any of site, id, tick or index
  never share a counter, and the in-flight round index is injective --
  breaks if two sites share a salt or ``round_index`` overlaps its fields.
* ONE SALT ENUM: every salt is four distinct ASCII letters, the episode and
  wander salts keep their values, and no module in ``src/`` passes a salt
  literal to the block function -- breaks on a second salt definition.
* LOCAL INDEX: ``next`` counts per (salt, id) and restarts each tick --
  breaks if the count becomes global (a stream) or survives the tick.
* SITE INVARIANCE (the property a global stream FAILS): in a played match,
  adding draws at an unrelated site / unit changes no other draw.
* THE STAND-IN: any use of ``sim.rng`` in a match raises, naming the call;
  the legacy WEGO paths still play -- breaks if spawn_match stops installing
  it or a match-path site goes back to ``sim.rng``.
* DETERMINISM: same seed twice = same digests; a different seed moves them.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import draws                                 # noqa: E402
from simulation.philox32 import philox32_4x32_10             # noqa: E402
from simulation.rng_salts import Salt, _ascii                # noqa: E402


# ---------------------------------------------------------------- layout
@pytest.mark.parametrize("seed", [0, 1, 0xDEADBEEF, (7 << 32) | 5,
                                  2 ** 64 - 1])
def test_a_draw_is_philox_on_the_layout(seed):
    """key = (seed lo, seed hi); counter = (tick, index, salt, id); out[0]."""
    md = draws.MatchDraws(seed)
    md.set_tick(1234)
    d = md.at(Salt.SHOT_CONE, 17, 3)
    want = philox32_4x32_10((1234, 3, int(Salt.SHOT_CONE), 17),
                            (seed & 0xFFFFFFFF, seed >> 32))[0]
    assert d.word() == want
    # "no unit" (-1) is the all-ones word, never a real id's
    assert md.at(Salt.SHOT_CONE, -1, 0).counter()[3] == 0xFFFFFFFF


def test_seed_refused_outside_64_bits():
    for bad in (-1, 2 ** 64, True, 1.5):
        with pytest.raises((ValueError, TypeError)):
            draws.MatchDraws(bad)


def test_distinct_draws_have_distinct_counters():
    """Every (site, id, tick, index) combination is its own counter."""
    md = draws.MatchDraws(5)
    seen = {}
    for t in (0, 1, 2 ** 20):
        md.set_tick(t)
        for salt in Salt:
            for ident in (-1, 0, 1, 300):
                for idx in (0, 1, 2, draws.round_index(1, 0),
                            draws.round_index(3, 2)):
                    c = md.at(salt, ident, idx).counter()
                    key = (t, salt, ident, idx)
                    assert c not in seen, (key, seen.get(c))
                    seen[c] = key


def test_round_index_is_injective_and_bounded():
    got = {draws.round_index(a, k) for a in range(40) for k in range(40)}
    assert len(got) == 40 * 40
    with pytest.raises(ValueError):
        draws.round_index(1 << 16, 0)
    with pytest.raises(ValueError):
        draws.round_index(0, 1 << 16)


def test_next_counts_per_site_and_restarts_each_tick():
    md = draws.MatchDraws(9)
    md.set_tick(10)
    assert [md.next(Salt.SHOT_CONE, 4).index for _ in range(3)] == [0, 1, 2]
    assert md.next(Salt.SHOT_CONE, 5).index == 0       # another unit
    assert md.next(Salt.BEAM_CONE, 4).index == 0       # another site
    md.set_tick(11)
    assert md.next(Salt.SHOT_CONE, 4).index == 0       # a new tick


def test_helpers_refuse_a_numpy_stream():
    import numpy as np
    with pytest.raises(TypeError, match="keyed"):
        draws.roll_q16(np.random.default_rng(0), 100)


def test_word_maps_unchanged():
    """P3e's word -> outcome maps over the keyed word (spot values)."""
    md = draws.MatchDraws(3)
    for i in range(200):
        d = md.at(Salt.EXPOSURE, 1, i)
        u = d.word()
        assert draws.roll_q16(d, 30000) == ((u >> 16) < 30000)
        assert draws.cone_offset_q16(d, 100) == ((u * 201) >> 32) - 100
        assert draws.noise_mult_q16(d, 50000) == 50000 + (
            (u * (65536 - 50000 + 1)) >> 32)


# ---------------------------------------------------------------- the enum
def test_salts_are_four_distinct_letters_and_the_old_values_kept():
    vals = [int(s) for s in Salt]
    assert len(set(vals)) == len(vals)
    for s in Salt:
        tag = int(s).to_bytes(4, "big").decode("ascii")
        assert tag.isalpha() and _ascii(tag) == int(s)
    assert int(Salt.EPISODE) == 0x45504953      # rl_env episode draws
    assert int(Salt.WANDER) == 0x57414E44       # the horde's wander


_BLOCK_FUNCS = {"philox32_4x32_10", "philox32_4x32_10_np", "philox32_draw"}


def test_no_salt_literal_outside_the_enum():
    """No ``*SALT* = <int>`` outside rng_salts.py, and no int literal of 32
    bits' width (a salt) inside a Philox block call's arguments in src/."""
    bad = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name in ("rng_salts.py", "philox32.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (isinstance(tgt, ast.Name) and "SALT" in tgt.id
                            and isinstance(node.value, ast.Constant)
                            and isinstance(node.value.value, int)):
                        bad.append(f"{path.name}:{node.lineno} {tgt.id}")
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(
                    f, "id", "")
                if name in _BLOCK_FUNCS:
                    for sub in ast.walk(node):
                        if (isinstance(sub, ast.Constant)
                                and isinstance(sub.value, int)
                                and not isinstance(sub.value, bool)
                                and sub.value >= 1 << 16
                                and sub.value != 0xFFFFFFFF):
                            bad.append(f"{path.name}:{sub.lineno} literal "
                                       f"{sub.value:#x}")
    assert not bad, bad


# ---------------------------------------------------------------- in a match
def _session(seed=1, zombies=None):
    import dataclasses
    import breach_physics as bp
    import level_loader
    from simulation.match_session import MatchSession
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = level.match
    if zombies is not None:
        spec = dataclasses.replace(spec, zombies=zombies)
    return MatchSession(level, spec, seed=seed, breach_physics=bp)


def test_a_match_refuses_sim_rng():
    """The match's sim.rng raises on any use, naming it."""
    sim = _session().sim
    with pytest.raises(RuntimeError, match=r"sim\.rng\.uniform"):
        sim.rng.uniform(0.0, 1.0)
    with pytest.raises(RuntimeError, match=r"sim\.rng\.integers"):
        sim.rng.integers(10)
    assert isinstance(sim.draws, draws.MatchDraws)
    assert sim.draws.seed == 1


def test_draws_read_the_sims_clock():
    sess = _session()
    assert sess.sim.draws.tick == sess.sim.tick
    sess.tick()
    sess.tick()
    assert sess.sim.draws.tick == sess.sim.tick == 2


def _play(seed, ticks, extra, monkeypatch):
    sess = _session(seed=seed, zombies=0)
    log = []
    real_word = draws.Draw.word

    def word(self):
        w = real_word(self)
        log.append((self.tick, self.salt, self.ident, self.index, w))
        if extra and self.salt == int(Salt.SHOT_CONE):
            # an unrelated consumer draws in between, on its own site
            for _ in range(3):
                real_word(self.owner.next(Salt.MELEE_CRIT, 99999))
        return w
    monkeypatch.setattr(draws.Draw, "word", word)
    try:
        for _ in range(ticks):
            if sess.is_terminal():
                break
            sess.tick()
    finally:
        monkeypatch.setattr(draws.Draw, "word", real_word)
    return log, sess.digests()


_PLAIN = {}


def _plain(seed, monkeypatch):
    """The 700-tick firefight of ``seed`` with no extra draws (cached)."""
    if seed not in _PLAIN:
        _PLAIN[seed] = _play(seed, 700, False, monkeypatch)
    return _PLAIN[seed]


def test_site_invariance_in_a_played_match(monkeypatch):
    """Draws at an unrelated site change no other draw: a ship_demo bots
    firefight played twice, the second time with three extra draws of a
    foreign site after every cone draw, makes the SAME draws and ends in the
    same digests. A global stream (P3e's numpy generator) fails this: the
    extra draws shift every later draw of every unit."""
    log_a, dig_a = _plain(3, monkeypatch)
    log_b, dig_b = _play(3, 700, True, monkeypatch)
    assert sum(1 for e in log_a if e[1] == int(Salt.SHOT_CONE)) > 10,         "the scenario fired too few rounds to prove anything"
    assert log_a == log_b
    assert dig_a == dig_b


def test_same_seed_same_match_other_seed_another(monkeypatch):
    """Determinism: seed 3 twice is equal; seed 4 differs in its draws."""
    log_a, dig_a = _plain(3, monkeypatch)
    log_c, dig_c = _play(3, 700, False, monkeypatch)
    log_d, _ = _play(4, 700, False, monkeypatch)
    assert log_a and log_d
    assert (log_a, dig_a) == (log_c, dig_c)
    assert [e[4] for e in log_a[:20]] != [e[4] for e in log_d[:20]]
