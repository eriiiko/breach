"""Loadout + environment-profile tables (#119 P1) — what a unit carries and
what it survives, as config rows.

    [loadouts.<name>]      weapon (a [weapons.*] id), grenades, explosives
    [env_profiles.<name>]  any subset of :class:`EnvironmentProfile`'s fields

A loadout is a ROW, never code: ``[marine] loadout`` names the row every marine
is armed with when its constructor is given none (the shipped ``rifleman`` row
carries the pre-#119 ``[marine]`` weapon / grenades / explosives, so default
behaviour is byte-identical), and a match config (#119 P2) names a row per
team. A suit is an ``[env_profiles.*]`` row over the existing
:class:`~simulation.environment.EnvironmentProfile` dataclass; an absent field
takes the dataclass default, so ``[env_profiles.default]`` (empty or equal to
the defaults) IS the human baseline. A unit given no profile keeps its
species' baseline (``species.environment``) exactly as before.

The door (mirroring :class:`simulation.weapons.WeaponsTables`): built from the
``CFG`` namespaces (plain dicts accepted for tests), loud ``ValueError`` /
``KeyError`` naming the row on an unknown weapon id, a negative or non-integer
count, an unknown profile field, an unknown submersion rule, or a ``[marine]
loadout`` that names no row. Construction-bound like the weapon tables: rebuilt
at :class:`Simulation` construction/reset (:func:`rebuild_tables`), never per
tick. Pure config data — nothing here is synced state or digested.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from simulation.environment import EnvironmentProfile, SubmersionRule


def _iter_rows(section_cfg):
    if section_cfg is None:
        return ()
    if isinstance(section_cfg, dict):
        return section_cfg.items()
    return vars(section_cfg).items()


def _row_items(row):
    return row.items() if isinstance(row, dict) else vars(row).items()


def _count(value, where: str) -> int:
    # bool is an int subclass; a `grenades = true` row is a typo, not a count.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer "
                         f"(got {value!r})")
    return int(value)


@dataclass(frozen=True)
class LoadoutDef:
    name: str
    weapon: str
    grenades: int
    explosives: int


_LOADOUT_COLUMNS = ("weapon", "grenades", "explosives")


class LoadoutTable:
    """``[loadouts.*]`` rows keyed by name; every ``weapon`` must name a row
    of the weapon table (``weapon_names``)."""

    def __init__(self, loadouts_cfg, weapon_names):
        weapon_names = set(weapon_names)
        self.by_name: dict[str, LoadoutDef] = {}
        for name, row in _iter_rows(loadouts_cfg):
            cols = dict(_row_items(row))
            unknown = sorted(set(cols) - set(_LOADOUT_COLUMNS))
            if unknown:
                raise ValueError(
                    f"loadouts.{name}: unknown column(s) {unknown} "
                    f"(a loadout row carries {list(_LOADOUT_COLUMNS)})")
            for c in _LOADOUT_COLUMNS:
                if c not in cols:
                    raise KeyError(f"loadouts.{name} missing column '{c}'")
            weapon = cols["weapon"]
            if not isinstance(weapon, str) or weapon not in weapon_names:
                raise ValueError(
                    f"loadouts.{name}.weapon {weapon!r} names no [weapons.*] "
                    f"row (known: {sorted(weapon_names)})")
            self.by_name[name] = LoadoutDef(
                name=name, weapon=weapon,
                grenades=_count(cols["grenades"], f"loadouts.{name}.grenades"),
                explosives=_count(cols["explosives"],
                                  f"loadouts.{name}.explosives"))

    def get(self, name: str) -> LoadoutDef:
        try:
            return self.by_name[name]
        except KeyError:
            raise KeyError(f"no [loadouts.{name}] row (known: "
                           f"{sorted(self.by_name)})") from None


_PROFILE_FIELDS = {f.name: f for f in dataclasses.fields(EnvironmentProfile)}


class EnvProfileTable:
    """``[env_profiles.*]`` rows keyed by name, each an
    :class:`EnvironmentProfile` (absent fields = the dataclass defaults)."""

    def __init__(self, profiles_cfg):
        self.by_name: dict[str, EnvironmentProfile] = {}
        for name, row in _iter_rows(profiles_cfg):
            kwargs = {}
            for col, value in _row_items(row):
                if col not in _PROFILE_FIELDS:
                    raise ValueError(
                        f"env_profiles.{name}: unknown field {col!r} "
                        f"(EnvironmentProfile has {sorted(_PROFILE_FIELDS)})")
                if col == "submersion":
                    try:
                        value = SubmersionRule(value)
                    except ValueError:
                        raise ValueError(
                            f"env_profiles.{name}.submersion {value!r} is not "
                            f"one of {[r.value for r in SubmersionRule]}"
                        ) from None
                elif isinstance(_PROFILE_FIELDS[col].default, bool):
                    if not isinstance(value, bool):
                        raise ValueError(f"env_profiles.{name}.{col} must be "
                                         f"a bool (got {value!r})")
                elif isinstance(value, bool) or not isinstance(
                        value, (int, float)):
                    raise ValueError(f"env_profiles.{name}.{col} must be a "
                                     f"number (got {value!r})")
                else:
                    value = float(value)
                kwargs[col] = value
            self.by_name[name] = EnvironmentProfile(**kwargs)

    def get(self, name: str) -> EnvironmentProfile:
        try:
            return self.by_name[name]
        except KeyError:
            raise KeyError(f"no [env_profiles.{name}] row (known: "
                           f"{sorted(self.by_name)})") from None


class LoadoutTables:
    """The two tables plus the marine default, validated together."""

    def __init__(self, loadouts_cfg, profiles_cfg, weapon_names,
                 default_loadout: str):
        self.loadouts = LoadoutTable(loadouts_cfg, weapon_names)
        self.env_profiles = EnvProfileTable(profiles_cfg)
        if default_loadout not in self.loadouts.by_name:
            raise ValueError(
                f"[marine] loadout {default_loadout!r} names no [loadouts.*] "
                f"row (known: {sorted(self.loadouts.by_name)})")
        self.default_loadout = default_loadout

    def loadout(self, name: str | None = None) -> LoadoutDef:
        """The named row, or the marine default for ``None``."""
        return self.loadouts.get(self.default_loadout if name is None
                                 else name)

    def env_profile(self, name: str) -> EnvironmentProfile:
        return self.env_profiles.get(name)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            from config import CFG
            cfg = CFG
        from simulation.weapons import get_tables
        return cls(getattr(cfg, "loadouts", None),
                   getattr(cfg, "env_profiles", None),
                   get_tables().weapons.by_name.keys(),
                   str(cfg.marine.loadout))


_TABLES: LoadoutTables | None = None


def get_tables() -> LoadoutTables:
    """The shared tables, built lazily from CFG on first use."""
    global _TABLES
    if _TABLES is None:
        _TABLES = LoadoutTables.from_config()
    return _TABLES


def rebuild_tables() -> LoadoutTables:
    """Rebuild from the live CFG (Simulation construction/reset)."""
    global _TABLES
    _TABLES = LoadoutTables.from_config()
    return _TABLES


__all__ = ["LoadoutDef", "LoadoutTable", "EnvProfileTable", "LoadoutTables",
           "get_tables", "rebuild_tables"]
