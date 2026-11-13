"""The ONE enum of Philox stream salts (#29 P3b, design §4.7).

Every Philox-4x32-10 draw the game makes names its PURPOSE in the counter's
third word, ``stream_salt`` (the layout of ``philox32.h``: key = the seed,
counter = ``(tick, draw_index, stream_salt, unit_or_site_id)``). A salt
separates one draw site from every other, so adding a site never shifts the
draws of an existing one. They live HERE and nowhere else: a new draw site
is a new member of :class:`Salt` (four ASCII letters, distinct), never a
literal at the call (``tests/test_match_draws.py`` scans ``src/`` for one).

Members (values are their four letters as one big-endian word):

* ``EPISODE`` ("EPIS") -- the farm's episode draw, ``rl_env.episode_draw``
  (#29 P1a; value unchanged, so every run's episode sequence is as before);
* ``WANDER`` ("WAND") -- the horde's wander direction, a SEEDLESS hash
  (``horde.wander_direction``, #183 G9p; value unchanged);
* ``SHOT_CONE`` ("CONE") -- a projectile round's cone angle (slot 4);
* ``BEAM_CONE`` ("BEAM") -- a hitscan beam's cone angle (slot 4);
* ``EXPOSURE`` ("EXPO") -- a round's exposure roll against cover (slots 2, 4);
* ``SHOT_CRIT`` ("CRIT") -- a round's crit roll on connect (slots 2, 4);
* ``MELEE_CRIT`` ("MELC") -- the order path's melee crit roll (slot 4);
* ``FIELD_NOISE`` ("NOIS") -- a FieldEdit's per-tile noise multiplier, at
  the queue's flush (slot 6b), on EVERY path.
"""
from __future__ import annotations

from enum import IntEnum


def _ascii(tag: str) -> int:
    """Four ASCII letters as one big-endian 32-bit word."""
    b = tag.encode("ascii")
    if len(b) != 4:
        raise ValueError(f"a salt is four ASCII letters, got {tag!r}")
    return int.from_bytes(b, "big")


class Salt(IntEnum):
    """Every Philox stream salt the game uses (see the module docstring)."""
    EPISODE = 0x45504953        # "EPIS"
    WANDER = 0x57414E44         # "WAND"
    SHOT_CONE = 0x434F4E45      # "CONE"
    BEAM_CONE = 0x4245414D      # "BEAM"
    EXPOSURE = 0x4558504F       # "EXPO"
    SHOT_CRIT = 0x43524954      # "CRIT"
    MELEE_CRIT = 0x4D454C43     # "MELC"
    FIELD_NOISE = 0x4E4F4953    # "NOIS"


__all__ = ["Salt"]
