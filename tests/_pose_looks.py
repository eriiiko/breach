"""Which looks the pose tests run over (#184).

Until #184 every shipped look (``[render.unit_looks]``) was on the Quaternius
skeleton, and the pose tests took "the shipped looks" to mean "the Quaternius
looks". Since #184 the player is drawn with ``soma_space_marine`` (SOMA), so
the two lists differ:

* :func:`shipped_looks` -- every look the table names (what the renderer
  loads);
* :func:`quaternius_looks` -- the looks on the Quaternius skeleton the game
  draws or keeps as its A/B: the shipped ones on that skeleton plus
  ``space_marine`` (the scale reference, the player's look before #184),
  in table order with the marine first, so a test that takes one Quaternius
  look takes the marine as before.
"""
from __future__ import annotations

from typing import List

from config import CFG
from renderer.pose_layer import _glb_skeleton, masks_for, pose_settings
from renderer.unit_looks import all_look_names, look_path, looks_table

QUATERNIUS_TABLE = "quaternius_ual"
QUATERNIUS_MARINE = "space_marine"


def shipped_looks() -> List[str]:
    return all_look_names(looks_table(CFG))


def skeleton_table_of(look: str) -> str:
    return masks_for(_glb_skeleton(look_path(look)), pose_settings(CFG))[0]


def quaternius_looks() -> List[str]:
    out = [QUATERNIUS_MARINE]
    for name in shipped_looks():
        if name not in out and skeleton_table_of(name) == QUATERNIUS_TABLE:
            out.append(name)
    return out
