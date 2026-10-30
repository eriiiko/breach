"""P5z: the mid octopus's row, built by the scaling law from the validated
0.24 m animal (the P1-P5 defaults) and the SIZES table
(art/characters/octopus/CONTROLS.md, row `mid`).

`mid_row()` returns every parameter set the 3-D sims take and writes
`out/row_mid.toml`, shaped like the design's section 3 groups (version, body,
tissue, solver, reach, crawl, swim, grab, idle, rules) with a comment per value
saying where it came from: `measured` (a literature or P1-P5 number, scale-free),
`scaled L^n` (the law), `SIZES` (the mid row), `tuned` (moved by a P5z gate,
with the reason), `new` (a P5z dial the 2-D prototype did not have).

THE LAW -- same-duration similarity, S = L_mid / L_ref = 2.52 / 0.24 = 10.5:
lengths x S, durations x 1, so speeds x S and accelerations x S;
masses x S^3 (geometry x S, density unchanged); forces (m a) x S^4;
k_dist (N/m) x S^3; k_bend (N m/rad) x S^5; grip limits (N) x S^4; torque
damping (N m s) x S^5; quadratic drag coefficients c_per / c_tan unchanged
(the drag force 1/2 rho A c v^2 goes as S^2 S^2 = S^4, like m a: self-similar --
the reference AREAS scale S^2, recomputed from the scaled radii); g_eff NOT
scaled (it is physics, 0.188 m/s^2): so under the law gravity is 10.5x weaker
relative to every other acceleration than in the 0.24 m animal. Every
dimensionless dial (activation, ratios, probabilities, cosines, mu, angles)
is unchanged.

Author: Claude (octopus lead), 2026-10-30.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace

import numpy as np

from arm_chain import N_SEG, ArmGeometry, ChainParams, ReachCommand
from arm_chain3d import ChainParams3D
from crawl_sim import CrawlParams
from grab_sim import GrabParams
from idle_sim import IdleParams
from swim_sim import SwimParams

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# ---------------------------------------------------------------- SIZES, row `mid`
SIZES_MID = dict(body_length_m=1.40, mantle_wide_m=0.93, mantle_long_m=1.08, arm_rest_m=2.52,
                 arm_stretched_m=3.31, arm_root_thickness_m=0.30, stretch_max=1.40)
L_REF = ArmGeometry().length_m          # 0.24 m, Yekutieli's arm, every P1-P5 number
G_EFF = 9.81 * (1.0 - 1022.0 / 1042.0)  # 0.188 m/s^2, tissue 1042 vs sea water 1022


@dataclass(frozen=True)
class BodyParams:
    """The rigid body (mantle + head), crown centre = the body point."""
    mass: float = 0.3                   # kg
    radius: float = 0.05                # m, crown radius where the arms root
    c_drag: float = 1.0                 # crawling, broadside
    crawl_area: float = 0.1 * 0.05      # m^2 (the 2-D Body2D default)
    swim_c_drag: float = 0.5            # mantle first, streamlined
    swim_area: float = np.pi * 0.05 ** 2
    rot_damp: float = 2e-4              # N m s
    mantle_long: float = 0.1            # m, the mantle length (gate unit: "mantle lengths")


@dataclass(frozen=True)
class GrabParams3D(GrabParams):
    lift_pitch_deg: float = 30.0        # new: the holding arm's root pitched up this far while it carries
    lift_s: float = 1.0                 # new: time to raise it
    carry_jet_dv: float = 0.5           # m/s, the P4 gate's in-plane jets during the carry


@dataclass(frozen=True)
class GateScale:
    """Gate thresholds with dimensions (the 2-D gates' constants, scaled)."""
    S: float = 1.0
    move_speed: float = 0.03            # P5 row 1 "moving" (m/s)
    drift: float = 0.05                 # P5 row 13 body drift (m)
    push_min_disp: float = 0.005        # P2 per-push statistic: minimum displacement (m)
    min_speed: float = 0.01             # P2 per-tick cosine: minimum speed (m/s)
    crawl_speed_ref: float = 0.05       # P3: the crawl speed swimming is compared to (m/s), replaced by P5z's own P2


@dataclass
class Row:
    S: float
    geom: ArmGeometry
    chain: ChainParams3D
    body: BodyParams
    crawl: CrawlParams
    swim: SwimParams
    grab: GrabParams3D
    idle: IdleParams
    reach: ReachCommand
    gates: GateScale
    notes: dict = field(default_factory=dict)   # "group.key" -> provenance comment


def _ellipsoid_mass(wide: float, long: float, rho: float) -> float:
    """The mantle as a prolate ellipsoid, wide x wide x long, at tissue density."""
    return rho * np.pi / 6.0 * wide * wide * long


def ref_row() -> Row:
    """The validated 0.24 m animal, exactly the P1-P5 defaults."""
    return Row(S=1.0, geom=ArmGeometry(), chain=ChainParams3D(g_eff=G_EFF), body=BodyParams(),
               crawl=CrawlParams(), swim=SwimParams(), grab=GrabParams3D(), idle=IdleParams(),
               reach=ReachCommand(a_ca=0.6, t_act=0.9), gates=GateScale())


def mid_row(write: bool = True, **tuned) -> Row:
    """Every parameter set for the mid octopus, from the law + SIZES. `tuned`
    overrides (group__key=value) are applied last and recorded as tuned."""
    ref = ref_row()
    S = SIZES_MID["arm_rest_m"] / L_REF
    notes = {}

    def sc(group, key, value, n, why=""):
        notes[f"{group}.{key}"] = f"scaled L^{n}" + (f" ({why})" if why else "")
        return value * S ** n

    def keep(group, key, why="measured, scale-free"):
        notes[f"{group}.{key}"] = why

    g0 = ref.geom
    geom = ArmGeometry(length_m=sc("body", "arm_length_m", g0.length_m, 1, "= SIZES mid arm at rest 2.52 m"),
                       r_base_m=sc("body", "radius_base_m", g0.r_base_m, 1, "SIZES draws 0.15 m: see README P5z"),
                       r_tip_m=sc("body", "radius_tip_m", g0.r_tip_m, 1),
                       rho_arm=g0.rho_arm, rho_water=g0.rho_water)
    keep("body", "tissue_density", "measured (Yekutieli I), kg/m^3")
    keep("body", "water_density", "sea water, kg/m^3")
    c0 = ChainParams()
    chain = ChainParams3D(dt_tick=c0.dt_tick, n_sub=c0.n_sub, n_pass=c0.n_pass,
                          k_bend=sc("tissue", "k_bend", c0.k_bend, 5, "N m/rad"),
                          k_passive_frac=c0.k_passive_frac,
                          k_dist=sc("tissue", "k_dist", c0.k_dist, 3, "N/m"),
                          bend_damp_s=c0.bend_damp_s, c_per=c0.c_per, c_tan=c0.c_tan, g_eff=G_EFF)
    keep("tissue", "k_passive_frac", "P1 choice, dimensionless")
    keep("tissue", "bend_damp_s", "P1 choice, a time: unchanged")
    keep("tissue", "c_drag_perp", "measured (Yekutieli I); areas scale L^2 from the radii")
    keep("tissue", "c_drag_tan", "measured (Yekutieli I)")
    keep("tissue", "g_eff", "physics, NOT scaled: 9.81 (1 - 1022/1042)")
    keep("solver", "substeps", "P1 finding (16 x 2), unchanged at mid: no gate forced a change")
    keep("solver", "passes", "P1 finding")

    b0 = ref.body
    r_crown = SIZES_MID["mantle_wide_m"] / 2.0
    body = BodyParams(
        mass=_ellipsoid_mass(SIZES_MID["mantle_wide_m"], SIZES_MID["mantle_long_m"], g0.rho_arm),
        radius=r_crown,
        c_drag=b0.c_drag,
        crawl_area=2.0 * r_crown * r_crown,          # the 2-D default's form, (2 r) x r
        swim_c_drag=b0.swim_c_drag,
        swim_area=np.pi * r_crown ** 2,
        rot_damp=sc("body", "rot_damp", b0.rot_damp, 5, "N m s"),
        mantle_long=SIZES_MID["mantle_long_m"])
    notes["body.body_mass_kg"] = ("SIZES: mantle 0.93 x 1.08 m as a prolate ellipsoid 0.93 x 0.93 x 1.08 at 1042 kg/m^3 "
                                  f"(the law would give {b0.mass * S ** 3:.0f} kg)")
    notes["body.body_radius_m"] = f"SIZES: half the mantle width (the law would give {b0.radius * S:.3f} m)"
    notes["body.crawl_area_m2"] = "SIZES radius, the 2-D form (2 r) x r"
    notes["body.swim_area_m2"] = "SIZES radius, pi r^2 (mantle first)"
    keep("body", "c_drag", "2-D choice, dimensionless")
    keep("body", "swim_c_drag", "2-D choice, dimensionless")
    notes["body.mantle_long_m"] = "SIZES (the gate's 'mantle length')"

    k0 = ref.crawl
    crawl = replace(k0, grip_limit=sc("crawl", "grip_limit", k0.grip_limit, 4, "N per anchored node"),
                    steer_min_dist=sc("crawl", "steer_min_dist", k0.steer_min_dist, 1, "m"))
    for k in ("anchor_nodes", "e_max", "push_s", "push_jitter", "recover_s", "recover_jitter", "decision_s",
              "p_recruit", "max_stance", "s_min", "back_bias", "neighbour_bonus", "mu_k", "stance_activation",
              "steer_gain", "steer_window_s"):
        keep("crawl", k, "P2, a duration / ratio / probability: unchanged")
    notes["crawl.mu_k"] = "P2, dimensionless -- but g_eff is not scaled, so friction is 10.5x weaker relative to the crawl's accelerations"

    s0 = ref.swim
    swim = replace(s0, jet_dv=sc("swim", "jet_dv", s0.jet_dv, 1, "a SPEED: same durations => x L"),
                   body_c_drag=body.swim_c_drag, body_area=body.swim_area)
    for k in ("open_deg", "closed_deg", "bundled_deg", "open_s", "close_s", "activation", "web_factor", "jet_s"):
        keep("swim", k, "P3, an angle / duration / ratio: unchanged")

    r0 = ref.grab
    pa = np.array(r0.prop_at)
    grab = GrabParams3D(
        arm=r0.arm,
        prop_mass=sc("grab", "prop_mass", r0.prop_mass, 3, "kg; the game's marine is 90 kg"),
        prop_radius=sc("grab", "prop_radius", r0.prop_radius, 1, "m"),
        prop_at=tuple(pa * S), reach_wave_s=r0.reach_wave_s, reach_stretch=r0.reach_stretch,
        reach_activation=r0.reach_activation, grab_nodes=r0.grab_nodes,
        sucker_reach=sc("grab", "sucker_reach", r0.sucker_reach, 1, "m"),
        wrap_activation=r0.wrap_activation, hold_activation=r0.hold_activation, pull_in_scale=r0.pull_in_scale,
        pull_in_s=r0.pull_in_s, grip_limit=sc("grab", "grip_limit", r0.grip_limit, 4, "N per sealed node"),
        hold_s=r0.hold_s, aim_max_deg=r0.aim_max_deg,
        lift_pitch_deg=r0.lift_pitch_deg, lift_s=r0.lift_s,
        carry_jet_dv=sc("grab", "carry_jet_dv", r0.carry_jet_dv, 1, "a speed"))
    notes["grab.prop_at_m"] = "scaled L^1"
    for k in ("reach_wave_s", "reach_stretch", "reach_activation", "grab_nodes", "wrap_activation",
              "hold_activation", "pull_in_scale", "pull_in_s", "hold_s", "aim_max_deg"):
        keep("grab", k, "P4, a duration / ratio / index: unchanged")
    keep("grab", "lift_pitch_deg", "new (P5z): the lift, the holding arm's root pitched up")
    keep("grab", "lift_s", "new (P5z)")

    idle = ref.idle
    for f_ in fields(IdleParams):
        keep("idle", f_.name, "P5, a duration / ratio / angle: unchanged")

    reach = ReachCommand(a_ca=0.6, t_act=0.9, front_width_m=geom.length_m / N_SEG)
    keep("reach", "a_ca", "P1 (Yekutieli's natural reach)")
    keep("reach", "t_act", "P1, a duration: unchanged")
    notes["reach.front_width_m"] = "one segment, L/17: scaled L^1"

    gs = GateScale(S=S, move_speed=0.03 * S, drift=0.05 * S, push_min_disp=0.005 * S, min_speed=0.01 * S,
                   crawl_speed_ref=0.05 * S)
    row = Row(S=S, geom=geom, chain=chain, body=body, crawl=crawl, swim=swim, grab=grab, idle=idle,
              reach=reach, gates=gs, notes=notes)
    for k, v in tuned.items():
        group, key = k.split("__")
        setattr(row, group, replace(getattr(row, group), **{key: v}))
        notes[f"{group}.{key}"] = f"TUNED to {v}"
    if write:
        write_toml(row, os.path.join(OUT, "row_mid.toml"))
    return row


# ---------------------------------------------------------------- the TOML hand-off

def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(np.float64(f"{float(v):.6g}")))
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (tuple, list, np.ndarray)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    raise TypeError(type(v))


def write_toml(row: Row, path: str) -> None:
    from crawl_sim import ARM_ANGLES
    n = row.notes
    g, c, b, k, s, r, i = row.geom, row.chain, row.body, row.crawl, row.swim, row.grab, row.idle
    groups = {
        "version": None,
        "body": [
            ("arms", 8, "fixed"), ("nodes_per_arm", 18, "one per game bone + the tip"),
            ("arm_length_m", g.length_m, n["body.arm_length_m"]),
            ("radius_base_m", g.r_base_m, n["body.radius_base_m"]),
            ("radius_tip_m", g.r_tip_m, n["body.radius_tip_m"]),
            ("tissue_density", g.rho_arm, n["body.tissue_density"]),
            ("water_density", g.rho_water, n["body.water_density"]),
            ("body_mass_kg", b.mass, n["body.body_mass_kg"]),
            ("body_radius_m", b.radius, n["body.body_radius_m"]),
            ("body_c_drag", b.c_drag, n["body.c_drag"]),
            ("body_crawl_area_m2", b.crawl_area, n["body.crawl_area_m2"]),
            ("body_rot_damp", b.rot_damp, n["body.rot_damp"]),
            ("mantle_long_m", b.mantle_long, n["body.mantle_long_m"]),
            ("arm_angles_deg", [float(np.degrees(a)) for a in ARM_ANGLES], "P2 layout, from the heading, CCW"),
            ("front_pairs", [0, 7], "L1, R1 (P5's front bias)"), ("back_pairs", [3, 4], "L4, R4"),
        ],
        "tissue": [
            ("k_bend", c.k_bend, n["tissue.k_bend"]),
            ("k_passive_frac", c.k_passive_frac, n["tissue.k_passive_frac"]),
            ("k_dist", c.k_dist, n["tissue.k_dist"]),
            ("bend_damp_s", c.bend_damp_s, n["tissue.bend_damp_s"]),
            ("c_drag_perp", c.c_per, n["tissue.c_drag_perp"]),
            ("c_drag_tan", c.c_tan, n["tissue.c_drag_tan"]),
            ("g_eff", c.g_eff, n["tissue.g_eff"]),
            ("mu_floor", k.mu_k, n["crawl.mu_k"]),
        ],
        "solver": [
            ("substeps", c.n_sub, n["solver.substeps"]), ("passes", c.n_pass, n["solver.passes"]),
            ("ticks_per_second", int(round(1.0 / c.dt_tick)), "the game's clock"),
            ("activation_table_samples", 1025, "S(x) on x in [-8, 8], step 1/64, linear interpolation (P5z)"),
        ],
        "reach": [
            ("a_ca", row.reach.a_ca, n["reach.a_ca"]), ("t_act_s", row.reach.t_act, n["reach.t_act"]),
            ("front_width_m", row.reach.front_width_m, n["reach.front_width_m"]),
        ],
        "crawl": [(f.name, getattr(k, f.name), n.get(f"crawl.{f.name}", "")) for f in fields(CrawlParams)
                  if f.name not in ("seed", "mu_k")],
        "swim": [(f.name, getattr(s, f.name), n.get(f"swim.{f.name}", "")) for f in fields(SwimParams)
                 if f.name not in ("arms_per_group", "body_c_drag", "body_area")]
        + [("body_c_drag", s.body_c_drag, n["body.swim_c_drag"]), ("body_area_m2", s.body_area, n["body.swim_area_m2"])],
        "grab": [(f.name if f.name != "prop_at" else "prop_at_m", getattr(r, f.name),
                  n.get(f"grab.{f.name}", n.get("grab.prop_at_m", "") if f.name == "prop_at" else ""))
                 for f in fields(GrabParams3D)],
        "idle": [(f.name, getattr(i, f.name), n.get(f"idle.{f.name}", "")) for f in fields(IdleParams)],
        "rules": [("can_become_zombie", False, "design section 3"), ("armed", False, "design section 3"),
                  ("blocks", "body", "design section 3")],
    }
    lines = ["# [creatures.octopus] for the MID octopus -- the P5z hand-off to P6a.",
             "# Generated by prototypes/octopus_motion/game_scale.py::mid_row(); do not hand-edit.",
             f"# Scaling law: same-duration similarity from the 0.24 m animal, S = {row.S:.4f}",
             "# (lengths S, masses S^3, k_dist S^3, k_bend S^5, grips S^4, durations 1, g_eff unscaled).",
             "# Provenance per value: measured / scaled L^n / SIZES / tuned / new.", ""]
    lines.append("version = 1   # hashed into every creature record (design section 5)")
    for grp, items in groups.items():
        if items is None:
            continue
        lines += ["", f"[{grp}]"]
        for key, val, note in items:
            lines.append(f"{key} = {_fmt(val)}" + (f"   # {note}" if note else ""))
    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    row = mid_row()
    print(open(os.path.join(OUT, "row_mid.toml"), encoding="utf-8").read())
