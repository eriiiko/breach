"""#29 P8 -- the device observation on the card (pytest wrapper of
tests/cuda_obs_check.py). One test per property; each runs its part in an
isolated CUDA subprocess (cuda_harness) and SKIPS cleanly without a CUDA
build / device.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def _run(parts, timeout=3600):
    proc = cuda_harness.run_cuda_script(
        "import cuda_obs_check, sys; "
        f"sys.exit(cuda_obs_check.main({list(parts)!r}))",
        timeout=timeout,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "OBS_RESULT: PASS" in out, out[-6000:]


def test_the_device_observation_equals_the_host_one_on_the_corpus():
    """PROPERTY: on the pinned ship_demo matches, a training room and the
    hand match (corpus (d): the cases the others never reach), at every
    decision tick, every column / plane owned so far (the column-to-patch
    table; from P8.1 the float32 vector columns of the store and the
    tables) equals the host observation as raw bits, and no owned column is
    constant over the corpus; from P8.2 who sees whom (seen_by,
    visible_by_team) equals the reference's VisionState at EVERY tick under
    both sight rules (a rule whose pairs were never both seen and unseen
    fails), and the line of fire over the cover table equals
    ``line_of_fire.solid_in_way`` on every (unit, unit) pair at every
    decision tick (a corpus where no cover decided a pair fails). BREAKS IF
    a block reads the wrong anchor (OBS_BREAK=anchor), is not rebuilt on a
    decision tick (OBS_BREAK=stale), the gate compares a column that never
    moves (OBS_BREAK=constant), a column divides by a neighbouring table row
    (OBS_BREAK=divisor), the cone is a degree wider (OBS_BREAK=cone), who
    sees whom runs before the conversion (OBS_BREAK=early), or a cover box
    sits one tile off (OBS_BREAK=cover). From P8.3 the device team picture's
    WHOLE state (records, bars, turned, death ticks, motion, last hits, the
    damage window and its sums, the object tracker by object position, the
    id counters, seen_now) equals the reference picture's at EVERY tick, and
    every coverage case (records created / refreshed / expired, deaths seen
    and unseen, a conversion with a record, a grenade tracked / lost /
    re-seen / carried through a compaction, a charge tracked, the damage
    window adding and dropping) is reached; BREAKS IF the picture updates
    before the conversion (OBS_BREAK=early_picture), a record lives one tick
    longer (OBS_BREAK=lifetime), the damage window takes the tick's hits in
    reverse order (OBS_BREAK=dmg_order) or the object serial is the row
    index (OBS_BREAK=serial_row). From P8.4a the lanes plane is raw-bit
    equal, and the intermediates the list reads (every team's fire lanes,
    every own lane, every aim_of answer, every (soldier, seen enemy)'s
    solid_in_way / mate_in_way / ticks_to_within) equal the reference's own
    values at every decision tick, every coverage case reached (lanes
    crossing a mate, a cell of several owners, a line of fire blocked by a
    wall / a cover crate / a mate, aim_of finding nobody / a mate / another
    side, ticks 0 / between / at the cap); BREAKS IF the lane walk or the
    rasteriser truncates where Python floors (OBS_BREAK=lane_trunc) or the
    lane walk runs one tile wide of its line (OBS_BREAK=lane_wide). From
    P8.4b the list (every entry column but the decoder's and the order's),
    its mask and its uids are raw-bit equal, every coverage case reached (an
    entry of every block, a grenade and a charge, a door in reach and out of
    reach, a block overflowing its capacity); BREAKS IF a block's sort ties
    by the higher id (OBS_BREAK=list_tie) or the door block holds one entry
    fewer than its capacity (OBS_BREAK=list_cap). From P8.4c the four masks
    are raw-bit equal to action_spec_v1.build_masks, every mask bit the spec
    does not fix both set and clear over the corpus, the wind-up's KEEP-only
    aim, an aimable entry, a door in use reach and a default row reached;
    BREAKS IF two hand bits' gates are swapped (OBS_BREAK=mask_swap) or the
    use mask's door reach is one tile wider (OBS_BREAK=mask_reach)."""
    _run(["gate"], timeout=7200)


def test_each_env_of_a_batch_equals_its_own_reference():
    """PROPERTY: in batches of N = 2 and 4 (ship_demo and the training
    room, different seeds per env, envs ending apart) every env's owned
    observation equals its own host reference at each of its decision
    ticks. BREAKS IF a block reads or writes another env's slice, or the
    row map or cadence of one env leaks into another."""
    _run(["batch"], timeout=7200)


def test_torch_gets_the_buffers_themselves_in_stream_order():
    """PROPERTY: the torch tensors of the observation (and team t's row
    views) share the CuPy buffers' memory, and the ordering the hand-over
    rests on holds: the tick stream is blocking and CuPy / torch run on the
    legacy default stream, so torch's read after a tick sees the block.
    BREAKS IF the hand-over copies, or the tick stream becomes non-blocking
    (then a torch read could race the block)."""
    _run(["handover"])


def test_the_observation_adds_no_host_sync_to_a_tick():
    """PROPERTY: with the observation attached a conductor tick still makes
    only its two named syncs (max_n_sub, the books read), at N = 1 and 2,
    while blocks are built. BREAKS IF the observation downloads or syncs
    inside the tick."""
    _run(["syncs"])


def test_an_ended_env_keeps_its_observation():
    """PROPERTY: an env whose ended latch is set is skipped by the blocks
    (its buffers unchanged) while its neighbour is rebuilt and equal to its
    reference. BREAKS IF a block ignores the latch."""
    _run(["ended"])


def test_the_observation_float_kit_rounds_like_the_host():
    """PROPERTY: the device observation's float kit (cpp/src/cuda_obs_float.cuh)
    converts double -> float16 exactly as numpy does on EVERY float16
    rounding case (each half, each midpoint and its float64 neighbours, the
    overflow and underflow edges, signed zeros, NaNs), double -> float32 as
    numpy does, never contracts a*b + c into an FMA, and serves the squash
    (CUDA's log) as the host's float32 on a dense sweep. BREAKS IF the half
    conversion rounds twice through float32 (OBS_BREAK=half_f32), the kit's
    arithmetic is fused (OBS_BREAK=fused), or CUDA's log moves a served
    squashed value (then the shared log module is due)."""
    _run(["floatkit"])


def test_sight_in_bulk_on_the_card_equals_rays_clear_and_window_clear():
    """PROPERTY (#29 P8.2): the device's bulk sight (rays and windows, both
    has_los forms, both window directions) equals ``vision.rays_clear`` /
    ``window_clear`` on dense random pairs and windows over real maps:
    ship_demo with its doors shut, half opened and with grille tiles (closed
    to sight, open to air), a training room with its door flipped, the
    training room with cover crates; ends outside the map and pairs on one
    tile included. BREAKS IF the two forms are swapped (OBS_BREAK=forms), a
    window cell is walked the wrong way, or an env reads another's planes."""
    _run(["sight"])


def test_who_a_team_sees_depends_only_on_the_enemies_in_view():
    """PROPERTY (#29 P8.2, THE WALL's first device gate): perturbing the
    store row of an enemy no member of a team sees leaves that team's
    visible set unchanged, the same perturbation of a seen enemy leaves it
    unchanged, and killing the seen enemy removes it; under both sight rules.
    From P8.4a the same perturbation of the unseen enemy (its trigger flipped
    too) moves none of the team's fire lanes, lanes plane, aim_of answers,
    lines of fire or turns, from P8.4b no entry, mask or uid of its
    list, and from P8.4c none of its four masks. BREAKS IF the visible table is indexed by the
    wrong row or team, a target's own facing / hp decides whether it is
    seen, or an answer of the team reads a unit it does not see."""
    _run(["wall"])


def test_a_restaged_picture_plays_on_equal_to_its_reference():
    """PROPERTY (#29 P8.3, the seam a per-env reset uses): env 1 of a batch
    of two, its device picture scrambled mid-match and restaged from its
    reference's normal form (``DevicePicture.restage``), and env 0 left
    alone, both equal their reference picture at every later tick. BREAKS
    IF the restage drops a field, renumbers the object serials apart from
    the tables' rows, or writes another env's block."""
    _run(["restage"])
