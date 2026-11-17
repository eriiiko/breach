"""#29 P7.0a -- the device draws' Python side (``simulation.unit_draws``).

CPU only (the device forms are held equal by tests/cuda_unit_kit_check.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import unit_draws  # noqa: E402
from simulation.draws import MatchDraws  # noqa: E402
from simulation.rng_salts import Salt  # noqa: E402


def test_the_generated_draws_header_is_current():
    """Property: ``cpp/src/unit_draws_layout.h`` is exactly what
    ``unit_draws.header_text`` writes, so every Salt member and every
    recorded .next site reaches C++ generated. Breaks if a Salt member, a
    .next site or the key row changes without ``tools/gen_unit_layout.py
    --write``."""
    path = ROOT / unit_draws.HEADER_REL
    assert path.read_text(encoding="utf-8").replace("\r\n", "\n") == \
        unit_draws.header_text(), f"stale: run `{unit_draws.GENERATOR}`"


def test_every_salt_is_in_the_header():
    """Property: each Salt member appears in the header as a constant with its
    own value (the device names a salt only through it). Breaks if the
    generator drops or mis-writes a member."""
    text = unit_draws.header_text()
    for m in Salt:
        assert f"SALT_{m.name} = 0x{int(m.value):08X}u;" in text


def test_each_next_salt_is_drawn_at_its_one_recorded_site():
    """Property: every ``.next(Salt.X, ...)`` call in ``src/`` is the ONE
    site ``unit_draws.NEXT_SITES`` records for X, and every recorded site
    exists (the device ``.next`` counter is local to one kernel's thread).
    Breaks if ``src/`` gains a second ``.next`` site for a salt, a ``.next``
    on an unrecorded salt, or a recorded site moves or disappears."""
    problems = unit_draws.next_site_problems(unit_draws.next_sites(ROOT))
    assert problems == [], "\n".join(problems)


def test_a_second_next_site_is_named():
    """Property: the scanner names a second ``.next`` site for a recorded salt
    and a ``.next`` on an unrecorded one (the check above cannot pass
    vacuously). Breaks if the AST scan stops seeing ``rng.next(Salt.X, ..)``
    calls or the problem check stops counting sites per salt."""
    sites = unit_draws.next_sites(ROOT)
    extra = unit_draws.sites_in_source(
        "def stray(rng, u):\n"
        "    a = rng.next(Salt.SHOT_CONE, u.id)\n"
        "    b = self.draws.next(_Salt.EXPOSURE, u.id)\n",
        "simulation/stray.py")
    assert [(s, f) for s, _m, f, _l in extra] == [("SHOT_CONE", "stray"),
                                                  ("EXPOSURE", "stray")]
    problems = unit_draws.next_site_problems(sites + extra)
    assert any("SHOT_CONE" in p and "2 sites" in p for p in problems), problems
    assert any("EXPOSURE" in p and "does not record" in p for p in problems), problems


def test_the_draw_key_is_what_a_draw_puts_in_its_key_and_first_word():
    """Property: ``draw_key`` is ``(seed lo, seed hi, the tick word)`` of
    exactly the counter a ``sim.draws`` draw uses this tick, the WEGO
    ``tick_base`` and a tick past 2^32 included. Breaks if the key row
    reads the clock without the base, splits the seed the other way or does
    not wrap the tick to 32 bits."""
    class _S:
        pass
    for seed, tick, base in ((0, 0, 0), ((1 << 64) - 1, (1 << 32) + 9, 0),
                             ((1 << 32) + 3, 77, 9000)):
        d = MatchDraws(seed)
        d.set_tick(tick)
        d.tick_base = base
        s = _S()
        s.draws = d
        k = unit_draws.draw_key(s)
        draw = d.at(Salt.SHOT_CONE, 5, 0)
        assert (int(k[0]), int(k[1])) == d.key
        assert int(k[2]) == draw.counter()[0]
