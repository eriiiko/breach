"""#106 P6c -- the C++ ArmChainSolver is the integer reference, integer for integer.

The reference (docs/octopus_integer_reference/chain_ref_q.py, `World.tick`) is the
executable specification; cpp/src/arm_chain_solver.cpp is its transcription. Each
scenario builds the reference World and the solver's buffers from the SAME state,
feeds both the SAME per-tick inputs (the scenario's hub-like `inputs`, packed from
the reference World before every tick) and asserts, after EVERY tick, that every
state array (positions, velocities, activation, anchors, the bodies' pose and
velocity, the activation clocks) and every output (contacts, pops, anchor and grab
forces) are equal with np.array_equal -- tolerance 0.

Scenarios (chain_ref_q_gates.SCENARIOS): the reach (the wave), the dead arm falling
onto the floor, a straight arm thrown flat at a wall, an oblique throw (gravity,
floor, friction), a tip-first throw at a pillar's corner, crowns pushed into a wall
and into another unit's footprint, and the synthetic animal (8 arms on a crown,
waves, anchors + pops, a grab, walls + a footprint) -- at the 1.80 / 2.52 / 3.39 m
rows (creatures.scale_row_by_law). CI runs short slices; the 10 s runs are behind
``BREACH_OCTOPUS_LONG_GATES=1`` (plus P5z's recorded traces replayed, when present).

Also: the solver's door (src/simulation/arm_chain.py) quantizes the row and the set-up
exactly as the reference's door does; its layout constants are the C++ enums'; the
compiled activation table is the reference's lookup; the binding refuses a malformed
buffer before writing anything.

Last long run (2026-10-31, P6c): every scenario x 1.80 / 2.52 / 3.39 m x 240 ticks
(21 tests) and the four P5z traces (reach, dead_arm, crawl, free_animal; 240 ticks,
open topology) -- all equal at tol 0, every tick; short slices in CI: reach 12 ticks
at the three sizes, dead 48, wall_thrown 24, wall_oblique 36, wall_corner 36,
body_wall 24, animal 16. The harness fails on a one-count change of gamma at tick 0.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_octopus_arm_chain_solver.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_REF = ROOT / "docs" / "octopus_integer_reference"
for _p in (ROOT, ROOT / "src", _REF, ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")
if not hasattr(bp, "arm_chain_tick"):
    pytest.skip("breach_physics predates the arm chain solver (#106 P6c): rebuild cpp/",
                allow_module_level=True)

import chain_ref_q as R  # noqa: E402
import chain_ref_q_gates as G  # noqa: E402
from simulation import arm_chain as AC  # noqa: E402

LONG = os.environ.get("BREACH_OCTOPUS_LONG_GATES") == "1"
TRACES = ROOT / "prototypes" / "octopus_motion" / "out" / "traces_p5z.npz"
SIZES = (1.80, 2.52, 3.39)


# ---------------------------------------------------------------------------
# Packing: the reference World -> the solver's buffers
# ---------------------------------------------------------------------------
def row_from_ref(rr: R.ChainRow) -> tuple[np.ndarray, np.ndarray]:
    """The solver's (row, row_node) from the reference's ChainRow (whatever
    the scenario replaced in it, e.g. g = 0)."""
    row = np.zeros(AC.ROW_N, np.int64)
    row[AC.ROW_N_NODES] = rr.n_nodes
    row[AC.ROW_N_SUB] = rr.n_sub
    row[AC.ROW_N_PASS] = rr.n_pass
    row[AC.ROW_INV_DT] = rr.inv_dt
    row[AC.ROW_INV_DT2] = rr.inv_dt2
    row[AC.ROW_L_Q24] = rr.L_q24
    row[AC.ROW_K_DIST_Q16] = rr.k_dist_q16
    row[AC.ROW_GAMMA_Q16] = rr.gamma_q16
    row[AC.ROW_ONEPG_Q16] = rr.onepg_q16
    row[AC.ROW_ATAN2_SHIFT] = rr.atan2_shift
    row[AC.ROW_G_DV_Q24] = rr.g_dv_q24
    row[AC.ROW_PLANE_MIN_Q32] = rr.plane_min_q32
    n = rr.n_nodes
    rn = np.zeros((n, AC.RN_N), np.int64)
    for i in range(n):
        rn[i, AC.RN_S_NODE_Q24] = rr.s_node_q24[i]
        rn[i, AC.RN_W_Q32] = rr.w_q32[i]
        rn[i, AC.RN_K_PER_Q16] = rr.K_per_q16[i]
        rn[i, AC.RN_K_TAN_Q16] = rr.K_tan_q16[i]
        if i < n - 1:
            rn[i, AC.RN_REST0_Q24] = rr.rest0_q24[i]
        if i < n - 2:
            rn[i, AC.RN_K_ACT_Q16] = rr.k_act_q16[i]
            rn[i, AC.RN_K_PAS_Q16] = rr.k_pas_q16[i]
    return row, rn


def _clock(w: R.World, a: R.Arm) -> int:
    return w.sub_count - (a.wave.t0_sub if a.wave is not None else 0)


def allocate_for(w: R.World) -> AC.ArmChainBuffers:
    t = w.topo
    h, wd = (len(t.grid), len(t.grid[0])) if t is not None else (1, 1)
    U = len(t.unit_rects) if t is not None else 0
    bufs = AC.allocate(1, len(w.arms), w.row.n_nodes, len(w.bodies), len(w.grabs), U, h, wd)
    if t is not None:
        bufs.blocked[0] = np.array(t.grid, dtype=np.uint8)
        bufs.world[0, AC.WD_HAS_TOPO] = 1
        bufs.world[0, AC.WD_TILE_Q24] = t.tile_q24
        bufs.world[0, AC.WD_ORIGIN_X_Q24], bufs.world[0, AC.WD_ORIGIN_Y_Q24] = t.origin
        for u, r in enumerate(t.unit_rects):
            bufs.unit_rects[0, u] = r
    pack_state(w, bufs)
    return bufs


def pack_state(w: R.World, b: AC.ArmChainBuffers) -> None:
    """The synced state, once at the start (the solver owns it from then on)."""
    for ai, a in enumerate(w.arms):
        b.p[0, ai] = a.p
        b.v[0, ai] = a.v
        b.wave_clock[0, ai] = _clock(w, a)
    for bi, bd in enumerate(w.bodies):
        b.body[0, bi] = bd.x + bd.v + [bd.th, bd.om]


def pack_inputs(w: R.World, b: AC.ArmChainBuffers) -> AC.ArmChainBuffers:
    """What the hub writes before a tick, from the reference World. Returns the
    buffers (re-allocated grab slots when the grab count changed)."""
    if b.grabs.shape[1] != len(w.grabs):
        b.grabs = np.full((1, len(w.grabs), AC.GR_N), -1, np.int64)
        b.grab_force = np.zeros((1, len(w.grabs)), np.int64)
    b.world[0, AC.WD_FLOOR_ON] = int(w.floor_on)
    for ai, a in enumerate(w.arms):
        c = b.arm_cfg[0, ai]
        c[:] = 0
        c[AC.AC_DRAG_MODE] = a.drag_mode
        c[AC.AC_VREF_Q24] = a.vref_q24
        c[AC.AC_U_Q30] = a.u_q30
        c[AC.AC_MU_Q16] = a.mu_q16
        c[AC.AC_ROOT_REF_X_Q30:AC.AC_ROOT_REF_Z_Q30 + 1] = a.root_ref
        c[AC.AC_BODY] = a.body
        c[AC.AC_GRIP_Q8] = AC.NO_GRIP if a.grip_q8 is None else a.grip_q8
        if a.wave is not None:
            c[AC.AC_WAVE_ON] = 1
            c[AC.AC_WAVE_A_CA_Q16] = a.wave.a_ca_q16
            c[AC.AC_WAVE_TACT_SUB] = a.wave.tact_sub
            c[AC.AC_WAVE_W_Q24] = a.wave.w_q24
            c[AC.AC_WAVE_HOLD] = int(a.wave.hold)
        b.pinned[0, ai] = a.pinned
        if a.root_local:
            b.root_local[0, ai] = a.root_local
        b.act[0, ai] = a.act
        b.rest_scale[0, ai] = a.rest_scale
        b.bend_target[0, ai] = a.target
        b.anchored[0, ai] = a.anchored
        b.anchor_pos[0, ai] = a.anchor_pos
    for bi, bd in enumerate(w.bodies):
        c = b.body_cfg[0, bi]
        c[AC.BC_W_Q32] = bd.w_q32
        c[AC.BC_WINVI_Q32] = bd.winvI_q32
        c[AC.BC_CB_Q32] = bd.cb_q32
        c[AC.BC_DRAG_MODE] = bd.drag_mode
        c[AC.BC_VREF_Q24] = bd.vref_q24
        c[AC.BC_U_Q30] = bd.u_q30
        c[AC.BC_Y_Q30] = bd.y_q30
        c[AC.BC_MU_Q16] = bd.mu_q16
        c[AC.BC_GRAVITY] = int(bd.gravity)
        c[AC.BC_FLOOR] = int(bd.floor)
        c[AC.BC_COLL_R_Q24] = bd.coll_r_q24
    for g, (ai, k, bi, local) in enumerate(w.grabs):
        b.grabs[0, g] = (ai, k, bi, *local)
    return b


def assert_equal(w: R.World, b: AC.ArmChainBuffers, where: str) -> None:
    """Every state array and every output, tolerance 0."""
    A = len(w.arms)

    def eq(name, ref, got):
        ref = np.asarray(ref, dtype=np.int64).reshape(np.shape(got))
        if not np.array_equal(ref, got.astype(np.int64)):
            idx = np.argwhere(ref != got.astype(np.int64))[0]
            raise AssertionError(f"{where}: {name} differs first at {tuple(idx)}: reference "
                                 f"{ref[tuple(idx)]} vs C++ {got[tuple(idx)]} "
                                 f"({int((ref != got).sum())} entries differ)")

    if A:
        eq("p", [a.p for a in w.arms], b.p[0])
        eq("v", [a.v for a in w.arms], b.v[0])
        eq("act", [a.act for a in w.arms], b.act[0])
        eq("anchored", [[int(x) for x in a.anchored] for a in w.arms], b.anchored[0])
        eq("popped", [[int(x) for x in a.popped] for a in w.arms], b.popped[0])
        eq("contact", [a.contact for a in w.arms], b.contact[0])
        eq("anchor_force", [a.anchor_force for a in w.arms], b.anchor_force[0])
        eq("wave_clock", [_clock(w, a) for a in w.arms], b.wave_clock[0])
    if w.bodies:
        eq("body", [bd.x + bd.v + [bd.th, bd.om] for bd in w.bodies], b.body[0])
        eq("body_contact", [bd.contact for bd in w.bodies], b.body_contact[0])
    if w.grabs:
        eq("grab_force", w.grab_force, b.grab_force[0])


def run_lockstep(sc: G.Scenario, ticks: int) -> dict:
    """Step the reference and the solver tick by tick; assert equality after
    every tick. Returns what happened (for the scenario's own sanity checks)."""
    w = sc.world
    row, rn = row_from_ref(w.row)
    bufs = allocate_for(w)
    seen = dict(floor=0, wall=0, unit=0, pops=0, grab_force=0)
    for t in range(ticks):
        if sc.inputs is not None:
            sc.inputs(w, t)
        bufs = pack_inputs(w, bufs)
        w.tick()
        AC.tick(row, rn, bufs, engine=bp)
        assert_equal(w, bufs, f"{sc.name} tick {t}")
        seen["floor"] += int((bufs.contact & AC.CONTACT_FLOOR).any() or
                             (bufs.body_contact & AC.CONTACT_FLOOR).any())
        seen["wall"] += int((bufs.contact & AC.CONTACT_WALL).any() or
                            (bufs.body_contact & AC.CONTACT_WALL).any())
        seen["unit"] += int((bufs.body_contact & AC.CONTACT_UNIT).any())
        seen["pops"] += int(bufs.popped.sum())
        seen["grab_force"] = max(seen["grab_force"], int(bufs.grab_force.max(initial=0)))
    return seen


def _scenario(name, L, ticks):
    cdef = G.creature(L)
    return G.SCENARIOS[name](cdef, ticks=ticks)


# ---------------------------------------------------------------------------
# Tol 0, short (CI)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", SIZES)
def test_the_reach_is_the_reference_tick_for_tick(L):
    """Property: P1's reach (the stiffening wave through the compiled
    activation table, pinned base, gravity, drag, Bishop frames, two bends per
    node) at each art size equals the reference, every array, every tick.
    Breaks on any arithmetic difference from chain_ref_q -- a rounding, a
    shift, an order of operations, the table lookup or the wave clock."""
    run_lockstep(_scenario("reach", L, 12), 12)


def test_the_dead_arm_falls_onto_the_floor_as_the_reference():
    """Property: the dead arm released 0.2 m above the deck (no pins, the
    row's friction) matches the reference through the fall and the landing
    -- the floor projection, its friction and the clamped velocity rebuild.
    Breaks if the floor contact or friction differs from the reference."""
    seen = run_lockstep(_scenario("dead", 2.52, 48), 48)
    assert seen["floor"] > 0, "the slice must reach the floor"


@pytest.mark.parametrize("name, ticks", [("wall_thrown", 24), ("wall_oblique", 36),
                                         ("wall_corner", 36), ("body_wall", 24)])
def test_the_static_topology_contacts_are_the_reference(name, ticks):
    """Property: the wall contacts (face choice and its tie-break, the push,
    the tangential friction, the velocity clamp at a face, the positional
    contact report) and the crown's disc pushed out of blocked tiles and
    another unit's footprint equal the reference, every tick. Breaks on any
    divergence in the P6c contact arithmetic."""
    seen = run_lockstep(_scenario(name, 2.52, ticks), ticks)
    assert seen["wall"] > 0, f"{name}: the slice must touch a wall"
    if name == "body_wall":
        assert seen["unit"] > 0, "the second crown must touch the footprint"


def test_the_animal_is_the_reference_tick_for_tick():
    """Property: eight arms on a crown with every path live -- root
    attachments, hub inputs each tick, two waves (one released), anchors
    with grip limits, a grabbed prop, the room's walls and a footprint
    touching the crown -- equal the reference, every array, every tick
    (a 16-tick slice; the 10 s run is the long gate). Breaks on any
    divergence in the attachments, anchors, grabs or the bodies."""
    seen = run_lockstep(_scenario("animal", 2.52, 16), 16)
    assert seen["unit"] > 0 and seen["grab_force"] > 0


# ---------------------------------------------------------------------------
# Tol 0, long (BREACH_OCTOPUS_LONG_GATES=1): 10 s at the three sizes
# ---------------------------------------------------------------------------
LONG_SCENARIOS = ["reach", "dead", "wall_thrown", "wall_oblique", "wall_corner", "body_wall", "animal"]


@pytest.mark.skipif(not LONG, reason="long gate: set BREACH_OCTOPUS_LONG_GATES=1")
@pytest.mark.parametrize("L", SIZES)
@pytest.mark.parametrize("name", LONG_SCENARIOS)
def test_long_ten_seconds_at_tol_0(name, L):
    """Property: 10 s (240 ticks) of every scenario at every art size, tol 0
    after every tick. Breaks on any divergence the short slices miss (the
    pops, the release and re-anchor, long contact sequences)."""
    seen = run_lockstep(_scenario(name, L, 240), 240)
    if name == "animal":
        assert seen["pops"] > 0, "the animal's grip limits must pop an anchor in 10 s"


@pytest.mark.skipif(not (LONG and TRACES.exists()), reason="long gate + P5z traces")
@pytest.mark.parametrize("name", ["reach", "dead_arm", "crawl", "free_animal"])
def test_long_p5z_traces_replayed_at_tol_0(name):
    """Property: P5z's recorded hub inputs (the crawl's anchors and grips, the
    free animal's uniform drag) replayed through both, 10 s, tol 0, with an
    open topology. Breaks on a divergence on the recorded scenarios G3 / G4
    measure."""
    cdef = G.creature()
    row = R.chain_row(cdef)
    tr = R.load_traces()[name]
    w = R.build_replay_world(row, tr, topo=G.open_topology())
    bank = "in_activation" in tr
    state = {"prev": None}

    def inputs(world, t):
        if bank:
            state["prev"] = R.apply_trace_inputs(world, tr, t, state["prev"])

    n = min(240, tr["in_activation"].shape[0] if bank else tr["p"].shape[0])
    run_lockstep(G.Scenario(w, inputs, n, name), n)


# ---------------------------------------------------------------------------
# The door, the layout, the table, the binding's refusals
# ---------------------------------------------------------------------------
def test_the_layout_constants_are_the_cpp_enums():
    """Property: arm_chain.py's layout literals equal arm_chain_solver.h's
    enums (bp.ARM_CHAIN_LAYOUT), name for name. Breaks if a column is added or
    moved on one side only."""
    cpp = dict(bp.ARM_CHAIN_LAYOUT)
    assert cpp.pop("ACT_TABLE_SAMPLES") == 1025
    assert cpp == {k: v for k, v in AC.layout().items() if k != "NO_GRIP"}


@pytest.mark.parametrize("L", SIZES)
def test_the_solvers_door_quantizes_the_row_as_the_reference(L):
    """Property: arm_chain.row_arrays(cdef) (the P6d1 path) equals the
    reference's chain_row at every art size. Breaks if the solver's door and
    the reference's door disagree about a constant."""
    cdef = G.creature(L)
    row, rn = AC.row_arrays(cdef, 24)
    r2, rn2 = row_from_ref(R.chain_row(cdef))
    assert np.array_equal(row, r2) and np.array_equal(rn, rn2)


def test_the_solvers_door_quantizes_the_setup_as_the_reference():
    """Property: arm_chain's set-up quantizers (arm_cfg_row, set_wave,
    body_cfg_row, grip_q8, unit_q30) equal World.add_arm / set_wave / add_body
    / set_grip for the same floats. Breaks if P6d1's buffers would hold a
    different animal than the reference runs."""
    cdef = G.creature()
    b = cdef.body
    rr = R.chain_row(cdef)
    w = R.World(rr)
    w.add_body((1.0, 2.0, 0.3), **G._body_kw(cdef))
    w.add_arm(G.hooked3d(cdef, 0.5), mu=0.1, root_ref=(0.2, -0.3, 0.9), body=0, grip=123.4,
              drag_mode=R.DRAG_LINEAR, vref=0.07, tau=0.8)
    w.set_wave(0, 0.55, 0.85, width_m=0.2, hold=False)
    a = w.arms[0]
    cfg = AC.arm_cfg_row(inv_dt=rr.inv_dt, drag_mode=R.DRAG_LINEAR, vref_m_s=0.07, tau_s=0.8,
                         mu=0.1, root_ref=(0.2, -0.3, 0.9), body=0, grip_n=123.4)
    AC.set_wave(cfg, inv_dt=rr.inv_dt, L_q24=rr.L_q24, a_ca=0.55, t_act_s=0.85, width_m=0.2,
                hold=False)
    bufs = pack_inputs(w, allocate_for(w))
    assert np.array_equal(cfg, bufs.arm_cfg[0, 0])
    assert a.grip_q8 == AC.grip_q8(123.4) and AC.grip_q8(None) == AC.NO_GRIP
    bc = AC.body_cfg_row(inv_dt=rr.inv_dt, water_density=b.water_density, mass_kg=b.body_mass_kg,
                         radius_m=b.body_radius_m, c_drag=b.body_c_drag, area_m2=b.body_crawl_area_m2,
                         rot_damp=b.body_rot_damp, mu=cdef.tissue.mu_floor,
                         coll_radius_m=b.body_radius_m)
    assert np.array_equal(bc, bufs.body_cfg[0, 0])


def test_the_compiled_activation_table_is_the_references_lookup():
    """Property: the C++ S(x) through the compiled activation_table.inc equals
    the reference's activation_s_q16 on the parsed .inc, densely across and
    beyond [-8, 8] (the clamp, every interval, the last index). Breaks if the
    compiled table or the lookup law drifts from the reference's."""
    tab = R.load_activation_table()
    xs = list(range(-(9 << 16), 9 << 16, 997)) + [-(8 << 16), (8 << 16), (8 << 16) - 1, 0, 1, -1,
                                                  (1 << 31) - 1, -(1 << 31)]
    for x in xs:
        assert bp.arm_chain_activation_s_q16(x) == R.activation_s_q16(x, tab), x


def test_the_binding_refuses_a_malformed_buffer_and_writes_nothing():
    """Property: a buffer of the wrong dtype or shape, an out-of-range index
    or a bend target beyond +-pi is REFUSED before the solver writes anything
    (noconvert: never a silent copy whose output is discarded). Breaks if the
    binding converts or skips a check."""
    sc = _scenario("reach", 2.52, 1)
    row, rn = row_from_ref(sc.world.row)
    bufs = pack_inputs(sc.world, allocate_for(sc.world))
    before = bufs.copy()

    def call(b):
        AC.tick(row, rn, b, engine=bp)

    bad = bufs.copy()
    bad.p = bad.p.astype(np.int64)
    with pytest.raises(TypeError):
        call(bad)
    bad = bufs.copy()
    bad.act = bad.act[:, :, :-1].copy()
    with pytest.raises(ValueError, match="act"):
        call(bad)
    bad = bufs.copy()
    bad.arm_cfg[0, 0, AC.AC_BODY] = 3
    with pytest.raises(ValueError, match="body index"):
        call(bad)
    bad = bufs.copy()
    bad.bend_target[0, 0, 0, 0] = 205888
    with pytest.raises(ValueError, match="bend target"):
        call(bad)
    for f in ("p", "v", "act", "wave_clock", "body"):
        assert np.array_equal(getattr(bufs, f), getattr(before, f))
