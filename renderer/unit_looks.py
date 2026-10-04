"""Which model each unit is drawn with -- the pure, raylib-free half of the
per-unit look (#33).

The table is ``[render.unit_looks]`` in config.toml: one list of model names
per ROLE (``player``, ``zombie``); a name is ``assets/models/<name>/<name>.glb``.
Units of a role are dealt the role's list round-robin in ascending unit id, so
the k-th unit of a role gets entry ``k mod n``.

A unit keeps the look it was first given: its role is read ONCE, the first
time the renderer sees its id (``is_zombie`` at that moment). A unit that
turns zombie mid-match keeps its model (#33: "mid-match turning = animation
swap, NOT skin swap").

Render-only: the assignment lives here, keyed by unit id. Nothing is read
from a unit but ``id`` and ``is_zombie``, and nothing is ever written to one.
``UnitModelRenderer`` owns one ``LookAssigner`` and resolves the names it
returns to loaded models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

_REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = _REPO_ROOT / "assets" / "models"

ROLE_PLAYER = "player"
ROLE_ZOMBIE = "zombie"
ROLES = (ROLE_PLAYER, ROLE_ZOMBIE)


def look_path(name: str) -> Path:
    """The model file a look name stands for."""
    return MODELS_DIR / name / f"{name}.glb"


def looks_table(cfg) -> Dict[str, List[str]]:
    """The ``[render.unit_looks]`` table from a loaded config (``CFG``), as
    ``{role: [name, ...]}`` for every role. Raises ValueError naming the key
    when a role is missing or its list is empty -- a look table that draws no
    one is a config error, not a silent sprite fallback."""
    section = getattr(getattr(cfg, "render", None), "unit_looks", None)
    if section is None:
        raise ValueError("config.toml has no [render.unit_looks] section")
    table: Dict[str, List[str]] = {}
    for role in ROLES:
        names = getattr(section, role, None)
        if not names or not all(isinstance(n, str) and n for n in names):
            raise ValueError(
                f"[render.unit_looks] {role} must be a non-empty list of "
                f"model names, got {names!r}")
        table[role] = list(names)
    return table


def all_look_names(table: Mapping[str, Sequence[str]]) -> List[str]:
    """Every distinct name in the table, in table order (roles in ``ROLES``
    order, then list order) -- the load order, so the first entry is the
    fallback for a look that fails to load."""
    out: List[str] = []
    for role in ROLES:
        for name in table.get(role, ()):
            if name not in out:
                out.append(name)
    return out


def unit_key(unit) -> int:
    """The id a look is remembered under (the renderer's anim-state key)."""
    return int(getattr(unit, "id", id(unit)))


def role_of(unit) -> str:
    """A unit's role as the look table names it, read from ``is_zombie``."""
    return ROLE_ZOMBIE if getattr(unit, "is_zombie", False) else ROLE_PLAYER


class LookAssigner:
    """Deals look names to units, once per unit id, and remembers them."""

    def __init__(self, table: Mapping[str, Sequence[str]]) -> None:
        self._table = {role: list(names) for role, names in table.items()}
        self._look: Dict[int, str] = {}
        self._dealt: Dict[str, int] = {role: 0 for role in self._table}

    def looks_for(self, units: Iterable) -> List[str]:
        """The look name of each unit, in the order given. Units never seen
        before are dealt first, in ascending id, each taking its role's next
        entry round-robin; a unit seen before keeps the name it was given."""
        units = list(units)
        fresh = sorted({unit_key(u): u for u in units
                        if unit_key(u) not in self._look}.items())
        for uid, u in fresh:
            role = role_of(u)
            names = self._table[role]
            k = self._dealt[role]
            self._look[uid] = names[k % len(names)]
            self._dealt[role] = k + 1
        return [self._look[unit_key(u)] for u in units]


__all__ = ["LookAssigner", "look_path", "looks_table", "all_look_names",
           "role_of", "unit_key", "ROLES", "ROLE_PLAYER", "ROLE_ZOMBIE", "MODELS_DIR"]
