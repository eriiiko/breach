"""The creature table (#106 P6a) -- ``[creatures.*]`` rows in config.toml, read
through ONE door into frozen :class:`CreatureDef` records.

Design: docs/octopus_p6_sim_port_design_2026-10-30.md section 3. A creature's
body plan, tissue, solver resolution, primitive dials and rules are ONE
versioned row; a species joins it by ``SpeciesDef.creature`` (species.py).
Nothing in code names a creature: every creature branch keys on
``species.creature is not None``, which is data.

THE DOOR (``build_creature``) refuses, with a ``ValueError`` naming the key
path (``creatures.<name>.<group>.<key>``):

* a missing group or key, and an UNKNOWN one (so a typo cannot silently
  default);
* a mistyped value (a bool is not a number, a float is not an int, a list of
  the wrong length or element type);
* a value the physics cannot take: a negative stiffness / mass / length /
  duration, a fraction outside [0, 1], a cosine outside [-1, 1],
  ``nodes_per_arm < 3``, ``arms < 1``, ``substeps < 1``, ``passes < 1``, a
  tip radius above the base radius;
* an index outside its range: ``anchor_nodes`` / ``grab_nodes`` outside
  ``[0, nodes_per_arm)``, ``front_pairs`` / ``back_pairs`` / ``grab.arm``
  outside ``[0, arms)``, ``arm_angles_deg`` not ``arms`` long,
  ``max_stance`` / ``max_active`` above ``arms``;
* a size the integer chain cannot hold -- the HEADROOM check
  (``creature_fixed.chain_headroom``: ten bounds, each named in the refusal).

Quantized ONCE here: every duration ``*_s`` becomes ``*_ticks`` with
``config.ticks_from_seconds`` (at least one tick), every seconds-valued jitter
``*_jitter`` becomes ``*_jitter_ticks`` (0 stays 0: no jitter is a meaning),
the two activation-wave travel times also become ``*_substeps`` (the
activation clock runs in substeps, design section 2), and the chain numbers
go through ``creature_fixed`` into ``CreatureDef.chain``. The seconds are kept
beside the ticks for reports only. ``tissue.bend_damp_s`` is a damping TIME
CONSTANT (1 ms), never a tick count: it enters as ``chain.bend_gamma_q16``.

Construction-bound and cached like the weapon / loadout tables
(:func:`creature_table`, :func:`rebuild_creature_table`): an F5 reload never
retunes a live creature. Pure config data -- nothing here is synced state;
the row's ``version`` and ``row_hash`` are what a creature record hashes
(design section 5, P6d1).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields

from config import ticks_from_seconds
from simulation import creature_fixed

# ---------------------------------------------------------------------------
# Value kinds (a field's metadata["kind"])
# ---------------------------------------------------------------------------
INT_GE0 = "int>=0"
INT_GE1 = "int>=1"
INT_GE2 = "int>=2"
INT_GE3 = "int>=3"
POS = "number>0"              # a length, mass, density, ratio
NONNEG = "number>=0"          # a stiffness, force, drag, gain, speed
FRAC = "fraction[0,1]"
COSINE = "cosine[-1,1]"
ANY = "number"                # signed (g_eff)
ANGLE = "angle_deg[-360,360]"
DUR = "duration_s>0"          # -> <stem>_ticks
WAVE_DUR = "wave_duration_s>0"   # -> <stem>_ticks and <stem>_substeps
JITTER = "jitter_s>=0"        # -> <name>_ticks (0 stays 0)
TIME_CONST = "time_constant_s>=0"   # kept as seconds; never ticks
BOOL = "bool"
ARM_INDEX = "arm_index"
ARM_LIST = "arm_index_list"
NODE_LIST = "node_index_list"
ANGLE_LIST = "angle_list"
VEC2 = "number_pair"
BLOCKS = "blocks_enum"

_BLOCKS_VALUES = ("body",)


def _k(kind):
    """A row key of this kind (a field read from config)."""
    return field(metadata={"kind": kind})


def _derived():
    """A field the door derives (never read from config)."""
    return field(metadata={"derived": True})


# ---------------------------------------------------------------------------
# The groups
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BodyDef:
    arms: int = _k(INT_GE1)
    nodes_per_arm: int = _k(INT_GE3)
    arm_length_m: float = _k(POS)
    radius_base_m: float = _k(POS)
    radius_tip_m: float = _k(POS)
    tissue_density: float = _k(POS)
    water_density: float = _k(POS)
    body_mass_kg: float = _k(POS)
    body_radius_m: float = _k(POS)
    body_c_drag: float = _k(NONNEG)
    body_crawl_area_m2: float = _k(NONNEG)
    body_rot_damp: float = _k(NONNEG)
    mantle_long_m: float = _k(POS)
    footprint: int = _k(INT_GE1)
    arm_angles_deg: tuple = _k(ANGLE_LIST)
    front_pairs: tuple = _k(ARM_LIST)
    back_pairs: tuple = _k(ARM_LIST)


@dataclass(frozen=True)
class TissueDef:
    k_bend: float = _k(NONNEG)
    k_passive_frac: float = _k(FRAC)
    k_dist: float = _k(NONNEG)
    bend_damp_s: float = _k(TIME_CONST)
    c_drag_perp: float = _k(NONNEG)
    c_drag_tan: float = _k(NONNEG)
    g_eff: float = _k(ANY)
    mu_floor: float = _k(NONNEG)


@dataclass(frozen=True)
class SolverDef:
    substeps: int = _k(INT_GE1)
    passes: int = _k(INT_GE1)
    activation_table_samples: int = _k(INT_GE2)


@dataclass(frozen=True)
class ReachDef:
    a_ca: float = _k(FRAC)
    t_act_s: float = _k(WAVE_DUR)
    front_width_m: float = _k(POS)
    t_act_ticks: int = _derived()
    t_act_substeps: int = _derived()


@dataclass(frozen=True)
class CrawlDef:
    anchor_nodes: tuple = _k(NODE_LIST)
    e_max: float = _k(POS)
    push_s: float = _k(DUR)
    push_jitter: float = _k(JITTER)
    recover_s: float = _k(DUR)
    recover_jitter: float = _k(JITTER)
    decision_s: float = _k(DUR)
    p_recruit: float = _k(FRAC)
    max_stance: int = _k(INT_GE1)
    s_min: float = _k(COSINE)
    back_bias: float = _k(NONNEG)
    neighbour_bonus: float = _k(NONNEG)
    grip_limit: float = _k(NONNEG)
    stance_activation: float = _k(FRAC)
    steer_gain: float = _k(NONNEG)
    steer_window_s: float = _k(DUR)
    steer_min_dist: float = _k(POS)
    push_ticks: int = _derived()
    push_jitter_ticks: int = _derived()
    recover_ticks: int = _derived()
    recover_jitter_ticks: int = _derived()
    decision_ticks: int = _derived()
    steer_window_ticks: int = _derived()


@dataclass(frozen=True)
class SwimDef:
    open_deg: float = _k(ANGLE)
    closed_deg: float = _k(ANGLE)
    bundled_deg: float = _k(ANGLE)
    open_s: float = _k(DUR)
    close_s: float = _k(DUR)
    activation: float = _k(FRAC)
    web_factor: float = _k(NONNEG)
    jet_dv: float = _k(NONNEG)
    jet_s: float = _k(DUR)
    body_c_drag: float = _k(NONNEG)
    body_area_m2: float = _k(NONNEG)
    open_ticks: int = _derived()
    close_ticks: int = _derived()
    jet_ticks: int = _derived()


@dataclass(frozen=True)
class GrabDef:
    arm: int = _k(ARM_INDEX)
    prop_mass: float = _k(POS)
    prop_radius: float = _k(POS)
    prop_at_m: tuple = _k(VEC2)
    reach_wave_s: float = _k(WAVE_DUR)
    reach_stretch: float = _k(POS)
    reach_activation: float = _k(FRAC)
    grab_nodes: tuple = _k(NODE_LIST)
    sucker_reach: float = _k(POS)
    wrap_activation: float = _k(FRAC)
    hold_activation: float = _k(FRAC)
    pull_in_scale: float = _k(POS)
    pull_in_s: float = _k(DUR)
    grip_limit: float = _k(NONNEG)
    hold_s: float = _k(DUR)
    aim_max_deg: float = _k(ANGLE)
    lift_pitch_deg: float = _k(ANGLE)
    lift_s: float = _k(DUR)
    carry_jet_dv: float = _k(NONNEG)
    reach_wave_ticks: int = _derived()
    reach_wave_substeps: int = _derived()
    pull_in_ticks: int = _derived()
    hold_ticks: int = _derived()
    lift_ticks: int = _derived()


@dataclass(frozen=True)
class IdleDef:
    max_active: int = _k(INT_GE0)
    p_start: float = _k(FRAC)
    hold_s: float = _k(DUR)
    hold_jitter: float = _k(JITTER)
    wave_s: float = _k(DUR)
    wave_jitter: float = _k(JITTER)
    stretch: float = _k(POS)
    stretch_jitter: float = _k(NONNEG)     # a rest-scale factor, NOT seconds
    activation: float = _k(FRAC)
    touch_s: float = _k(DUR)
    touch_jitter: float = _k(JITTER)
    limp_s: float = _k(DUR)
    limp_jitter: float = _k(JITTER)
    aim_deg: float = _k(ANGLE)
    front_bias: float = _k(NONNEG)
    hold_ticks: int = _derived()
    hold_jitter_ticks: int = _derived()
    wave_ticks: int = _derived()
    wave_jitter_ticks: int = _derived()
    touch_ticks: int = _derived()
    touch_jitter_ticks: int = _derived()
    limp_ticks: int = _derived()
    limp_jitter_ticks: int = _derived()


@dataclass(frozen=True)
class RulesDef:
    can_become_zombie: bool = _k(BOOL)
    armed: bool = _k(BOOL)
    blocks: str = _k(BLOCKS)


@dataclass(frozen=True)
class CreatureDef:
    """One ``[creatures.<name>]`` row, validated and quantized once."""
    name: str
    version: int
    body: BodyDef
    tissue: TissueDef
    solver: SolverDef
    reach: ReachDef
    crawl: CrawlDef
    swim: SwimDef
    grab: GrabDef
    idle: IdleDef
    rules: RulesDef
    chain: creature_fixed.ChainHeadroom
    #: sha256 of the row's canonical JSON (keys sorted): what a creature
    #: record hashes beside ``version`` (design section 5, P6d1).
    row_hash: str


_GROUPS = (("body", BodyDef), ("tissue", TissueDef), ("solver", SolverDef),
           ("reach", ReachDef), ("crawl", CrawlDef), ("swim", SwimDef),
           ("grab", GrabDef), ("idle", IdleDef), ("rules", RulesDef))
_ROW_KEYS = ("version",) + tuple(g for g, _ in _GROUPS)


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------
def _as_dict(obj, where: str) -> dict:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return dict(vars(obj))
    raise ValueError(f"{where} must be a table (got {obj!r})")


def _number(v, where: str) -> float:
    # bool is an int subclass: `k_dist = true` is a typo, not a stiffness.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where} must be a number (got {v!r})")
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{where} must be finite (got {v!r})")
    return v


def _integer(v, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{where} must be an integer (got {v!r})")
    return int(v)


def _int_list(v, where: str) -> tuple:
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f"{where} must be a non-empty list of integers "
                         f"(got {v!r})")
    return tuple(_integer(x, f"{where}[{i}]") for i, x in enumerate(v))


def _value(kind: str, v, where: str):
    """Type + range check of one key by its kind (cross-key checks later)."""
    if kind in (INT_GE0, INT_GE1, INT_GE2, INT_GE3):
        x = _integer(v, where)
        lo = int(kind[-1])
        if x < lo:
            raise ValueError(f"{where} must be >= {lo} (got {x})")
        return x
    if kind in (POS, DUR, WAVE_DUR):
        x = _number(v, where)
        if not x > 0.0:
            raise ValueError(f"{where} must be > 0 (got {x})")
        return x
    if kind in (NONNEG, JITTER, TIME_CONST):
        x = _number(v, where)
        if x < 0.0:
            raise ValueError(f"{where} must be >= 0 (got {x})")
        return x
    if kind == FRAC:
        x = _number(v, where)
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"{where} is a fraction, must be in [0, 1] "
                             f"(got {x})")
        return x
    if kind == COSINE:
        x = _number(v, where)
        if not -1.0 <= x <= 1.0:
            raise ValueError(f"{where} is a cosine, must be in [-1, 1] "
                             f"(got {x})")
        return x
    if kind == ANY:
        return _number(v, where)
    if kind == ANGLE:
        x = _number(v, where)
        if not -360.0 <= x <= 360.0:
            raise ValueError(f"{where} is an angle in degrees, must be in "
                             f"[-360, 360] (got {x})")
        return x
    if kind == BOOL:
        if not isinstance(v, bool):
            raise ValueError(f"{where} must be a bool (got {v!r})")
        return v
    if kind == BLOCKS:
        if not isinstance(v, str) or v not in _BLOCKS_VALUES:
            raise ValueError(f"{where} must be one of {list(_BLOCKS_VALUES)} "
                             f"(got {v!r})")
        return v
    if kind in (ARM_INDEX,):
        return _integer(v, where)       # range checked against body.arms
    if kind in (ARM_LIST, NODE_LIST):
        return _int_list(v, where)      # ranges checked against body
    if kind == ANGLE_LIST:
        if not isinstance(v, (list, tuple)) or not v:
            raise ValueError(f"{where} must be a non-empty list of angles "
                             f"(got {v!r})")
        return tuple(_value(ANGLE, x, f"{where}[{i}]")
                     for i, x in enumerate(v))
    if kind == VEC2:
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"{where} must be a list of 2 numbers "
                             f"(got {v!r})")
        return tuple(_number(x, f"{where}[{i}]") for i, x in enumerate(v))
    raise AssertionError(f"unknown kind {kind!r}")   # pragma: no cover


def _derived_name(name: str, kind: str) -> list[tuple[str, str]]:
    """(derived field, clock) pairs for a duration key."""
    if kind == JITTER:
        return [(f"{name}_ticks", "tick")]
    stem = name[:-2] if name.endswith("_s") else name
    out = [(f"{stem}_ticks", "tick")]
    if kind == WAVE_DUR:
        out.append((f"{stem}_substeps", "substep"))
    return out


def _build_group(cls, raw, where: str, tps: int, substeps: int):
    cols = _as_dict(raw, where)
    kinds = {f.name: f.metadata["kind"] for f in fields(cls)
             if "kind" in f.metadata}
    unknown = sorted(set(cols) - set(kinds))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown} (a [{where}] "
                         f"table carries {sorted(kinds)})")
    values = {}
    for name, kind in kinds.items():
        if name not in cols:
            raise ValueError(f"{where}.{name}: missing key")
        values[name] = _value(kind, cols[name], f"{where}.{name}")
        if kind in (DUR, WAVE_DUR, JITTER):
            for dname, clock in _derived_name(name, kind):
                rate = tps if clock == "tick" else tps * substeps
                sec = values[name]
                # A zero jitter is "no jitter", never one tick of it.
                values[dname] = (0 if kind == JITTER and sec == 0.0
                                 else ticks_from_seconds(sec, rate))
    derived = {f.name for f in fields(cls) if f.metadata.get("derived")}
    missing = derived - set(values)
    assert not missing, f"{cls.__name__}: underived {sorted(missing)}"
    return cls(**values)


def _canonical(obj):
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        return {k: _canonical(v) for k, v in vars(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def build_creature(name: str, row, ticks_per_second: int) -> CreatureDef:
    """THE door: one ``[creatures.<name>]`` row -> a :class:`CreatureDef`, or
    a ``ValueError`` naming the key path (see the module docstring)."""
    where = f"creatures.{name}"
    tps = _integer(ticks_per_second, "[clock] ticks_per_second")
    cols = _as_dict(row, where)
    unknown = sorted(set(cols) - set(_ROW_KEYS))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown} (a creature row "
                         f"carries {list(_ROW_KEYS)})")
    for key in _ROW_KEYS:
        if key not in cols:
            raise ValueError(f"{where}.{key}: missing "
                             f"{'key' if key == 'version' else 'table'}")
    version = _value(INT_GE1, cols["version"], f"{where}.version")
    # The solver's substeps clock the wave durations: read it first.
    solver = _build_group(SolverDef, cols["solver"], f"{where}.solver", tps, 1)
    groups = {"solver": solver}
    for gname, cls in _GROUPS:
        if gname == "solver":
            continue
        groups[gname] = _build_group(cls, cols[gname], f"{where}.{gname}",
                                     tps, solver.substeps)
    _cross_checks(where, groups)
    body, tissue = groups["body"], groups["tissue"]
    crawl, grab, idle, swim = (groups["crawl"], groups["grab"],
                               groups["idle"], groups["swim"])
    # The rest-scale range every shipped primitive can drive a segment to
    # (crawl's elongation, the grab's reach and pull-in, idle's stretch); a
    # new primitive that scales rest further adds its dial here.
    stretch_max = max(1.0, crawl.e_max, grab.reach_stretch,
                      idle.stretch + idle.stretch_jitter)
    stretch_min = min(1.0, grab.pull_in_scale)
    chain = creature_fixed.chain_headroom(
        where=where, nodes_per_arm=body.nodes_per_arm,
        arm_length_m=body.arm_length_m, radius_base_m=body.radius_base_m,
        radius_tip_m=body.radius_tip_m, tissue_density=body.tissue_density,
        body_mass_kg=body.body_mass_kg, body_radius_m=body.body_radius_m,
        k_bend=tissue.k_bend, k_passive_frac=tissue.k_passive_frac,
        k_dist=tissue.k_dist, bend_damp_s=tissue.bend_damp_s,
        ticks_per_second=tps, substeps=solver.substeps,
        stretch_min=stretch_min, stretch_max=stretch_max,
        max_speed_m_s=max(swim.jet_dv, grab.carry_jet_dv))
    blob = json.dumps(_canonical(cols), sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return CreatureDef(name=name, version=version, chain=chain,
                       row_hash=hashlib.sha256(blob).hexdigest(), **groups)


def _cross_checks(where: str, g: dict) -> None:
    body = g["body"]
    arms, nodes = body.arms, body.nodes_per_arm
    if body.radius_tip_m > body.radius_base_m:
        raise ValueError(f"{where}.body.radius_tip_m {body.radius_tip_m} is "
                         f"above radius_base_m {body.radius_base_m} (an arm "
                         f"tapers base -> tip)")
    if len(body.arm_angles_deg) != arms:
        raise ValueError(f"{where}.body.arm_angles_deg has "
                         f"{len(body.arm_angles_deg)} angles for {arms} arms")
    for key in ("front_pairs", "back_pairs"):
        for i, a in enumerate(getattr(body, key)):
            if not 0 <= a < arms:
                raise ValueError(f"{where}.body.{key}[{i}] = {a} is not an "
                                 f"arm index (0 .. {arms - 1})")
    if not 0 <= g["grab"].arm < arms:
        raise ValueError(f"{where}.grab.arm = {g['grab'].arm} is not an arm "
                         f"index (0 .. {arms - 1})")
    for path, lst in (("crawl.anchor_nodes", g["crawl"].anchor_nodes),
                      ("grab.grab_nodes", g["grab"].grab_nodes)):
        for i, k in enumerate(lst):
            if not 0 <= k < nodes:
                raise ValueError(f"{where}.{path}[{i}] = {k} is outside the "
                                 f"node range 0 .. {nodes - 1}")
        if any(b <= a for a, b in zip(lst, lst[1:])):
            raise ValueError(f"{where}.{path} must be strictly ascending "
                             f"(got {list(lst)})")
    for path, n in (("crawl.max_stance", g["crawl"].max_stance),
                    ("idle.max_active", g["idle"].max_active)):
        if n > arms:
            raise ValueError(f"{where}.{path} = {n} is above the {arms} arms")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
class CreatureTable:
    """Every ``[creatures.*]`` row, keyed by name."""

    def __init__(self, creatures_cfg, ticks_per_second: int):
        self.by_name: dict[str, CreatureDef] = {}
        if creatures_cfg is None:
            return
        for name, row in _as_dict(creatures_cfg, "creatures").items():
            self.by_name[name] = build_creature(name, row, ticks_per_second)

    def get(self, name: str) -> CreatureDef:
        try:
            return self.by_name[name]
        except KeyError:
            raise KeyError(f"no [creatures.{name}] row (known: "
                           f"{sorted(self.by_name)})") from None

    @classmethod
    def from_config(cls, cfg=None) -> "CreatureTable":
        if cfg is None:
            from config import CFG
            cfg = CFG
        return cls(getattr(cfg, "creatures", None),
                   cfg.clock.ticks_per_second)


_TABLE: CreatureTable | None = None


def creature_table() -> CreatureTable:
    """The shared table, built lazily from CFG on first use."""
    global _TABLE
    if _TABLE is None:
        _TABLE = CreatureTable.from_config()
    return _TABLE


def rebuild_creature_table() -> CreatureTable:
    """Rebuild from the live CFG (construction-bound, like the weapon tables)."""
    global _TABLE
    _TABLE = CreatureTable.from_config()
    return _TABLE


def get_creature(name: str) -> CreatureDef:
    """The named row (``KeyError`` naming the known rows when absent)."""
    return creature_table().get(name)


__all__ = [
    "BodyDef", "TissueDef", "SolverDef", "ReachDef", "CrawlDef", "SwimDef",
    "GrabDef", "IdleDef", "RulesDef", "CreatureDef", "CreatureTable",
    "build_creature", "creature_table", "rebuild_creature_table",
    "get_creature",
]
