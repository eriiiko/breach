"""The sweep's checked-in LIGHT EMISSION TABLE L°[T] (ray-engine-v2 P6a).

docs/ray_engine_v2_p6a_light_channels_brief_2026-09-25.md decisions 3 and 4;
design v3 §2.6 / §7.3 / §8.1. The table lives in cpp/src/light_emission_table.inc,
written OFFLINE by tools/gen_light_table.py from renderer/blackbody.py (the ONE
temperature -> colour map) and compiled into the engine; the integer reference
reads the same file. These tests hold that file to its generator, to the colour
map, and to the currency's two floors.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_light_table.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_light_table as GEN  # noqa: E402

_STUDY = ROOT / "docs" / "ray_engine_v2_scheme_study_2026-09-13"
if str(_STUDY) not in sys.path:
    sys.path.insert(0, str(_STUDY))
import sweep_ref_q as R  # noqa: E402


@pytest.fixture(scope="module")
def checked_in():
    return GEN.parse_inc()


@pytest.fixture(scope="module")
def ramp():
    from config import CFG
    from renderer.blackbody import BlackbodyRamp
    return BlackbodyRamp.from_config(CFG)


def test_the_checked_in_table_is_what_the_generator_makes(checked_in):
    """PROPERTY (decision 3): cpp/src/light_emission_table.inc is EXACTLY what
    tools/gen_light_table.py computes from renderer/blackbody.py at the shipped
    dials -- every entry within ONE count (a machine's float64 libm may differ
    in its last ULP, which x 2^40 can flip one rounding and no more) -- in the
    currency emissive_table.h declares.

    BREAKS IF: blackbody.py's curves, [render.blackbody], the T -> Kelvin map
    or L_FINE_BITS move without `tools/gen_light_table.py --write` (the table
    would then disagree with the colour map it claims to be), or someone edits
    the .inc by hand. Validated: kelvin_ref 3000 -> 3001 in the generator's
    config reddens it by 2^30.9 counts.
    """
    k_inc, table = checked_in
    k_hdr = GEN.l_fine_bits_from_header()
    assert k_inc == k_hdr, "the .inc was generated at another currency: regenerate it"
    regen = GEN.compute_light_table(fine_bits=k_hdr)
    assert table.shape == regen.shape == (3, GEN.E_TABLE_SIZE)
    diff = np.abs(table - regen)
    assert int(diff.max()) <= 1, (
        f"the checked-in L° drifted from blackbody.py by up to {int(diff.max())} "
        f"counts (first at {np.unravel_index(int(diff.argmax()), diff.shape)}): "
        f"run tools/gen_light_table.py --write")


def test_a_room_temperature_body_emits_no_light(checked_in):
    """PROPERTY (decision 3): L°_c[0] == 0 on every channel, and no entry is
    negative. Bucket 0 is where every cell at or below ambient reads (the
    emissive table floors there), so this is what makes the light ambient DARK:
    a room of ambient cells emits exactly nothing, and a body -- which re-emits
    the ambient level -- never glows (decision 5).

    BREAKS IF: the ramp's glow floor (kelvin_glow_min) is lowered to ambient,
    or the table is baked with an offset.
    """
    _k, table = checked_in
    assert all(int(table[c, 0]) == 0 for c in range(3))
    assert int(table.min()) >= 0


def test_the_table_stays_inside_its_headroom_door(checked_in):
    """PROPERTY (decision 4): every entry is below L_TABLE_TOP_MAX (2^44, the
    door emissive_table.cpp static_asserts), and the top is where the currency
    says: intensity_max x 2^L_FINE_BITS in the peak channel.

    BREAKS IF: L_FINE_BITS grows past 40, or intensity_max past 2^(44 - k).
    """
    k, table = checked_in
    assert int(table.max()) < (1 << 44)
    from config import CFG
    top = float(CFG.render.blackbody.intensity_max) * float(1 << k)
    assert int(table[:, -1].max()) == int(round(top))


def test_the_chroma_is_blackbody_py_s_at_every_bucket(checked_in, ramp):
    """PROPERTY (brief section 3.7, the table half): at every bucket whose peak
    channel carries at least 2^16 counts, the table's chroma L°_c / L°_peak
    equals blackbody.py's Helland/Bartlett chroma at the bucket's midpoint
    temperature (BlackbodyRamp.emission_at_kelvin) within 2^-15 -- the table's
    own rounding -- and the peak channel is the ramp's intensity x 2^k. Against
    the ramp's PUBLIC per-frame map (its 256-entry LUT, light_color) the chroma
    lies within that LUT's own step there.

    BREAKS IF: the generator swaps or drops a channel, multiplies the chroma by
    the wrong intensity, or samples the ramp at another temperature than the
    bucket's (a bucket's low edge instead of its midpoint moves the chroma by
    ~2 K, far above 2^-15 at the red end).
    """
    k, table = checked_in
    kelvin = GEN.bucket_kelvin()
    chroma, inten = ramp.emission_at_kelvin(kelvin)
    peak = table.max(axis=0).astype(np.float64)
    live = peak >= float(1 << 16)
    assert live.sum() > 3000, "vacuous: too few glowing buckets"
    got = table[:, live].astype(np.float64) / peak[live]
    err = np.abs(got - chroma[live].T)
    assert float(err.max()) <= 2.0 ** -15, float(err.max())
    assert np.all(np.abs(peak[live] - inten[live] * float(1 << k)) <= 1.0)
    # the public LUT (what the renderer draws with): within its own step
    lut = ramp._chroma_lut.astype(np.float64)
    step = np.max(np.abs(np.diff(lut, axis=0)), axis=0)          # per channel
    t_game = 4.0 * np.arange(GEN.E_TABLE_SIZE) + 2.0
    worst = 0.0
    for t in np.flatnonzero(live)[::97]:
        (r, g, b), _i = ramp.light_color(float(t_game[t]))
        d = np.abs(np.array([r, g, b]) - got[:, np.flatnonzero(live).tolist().index(t)])
        worst = max(worst, float(np.max(d - 2 * step)))
    assert worst <= 1e-6


def test_hotter_is_brighter_and_colour_warms_to_white(checked_in, ramp):
    """PROPERTY (brief section 3.7): the peak channel -- what one ordinate of an
    opaque emitter carries, (L°_peak * w_m) >> 16 -- NEVER falls as T rises,
    over the whole table, and rises STRICTLY through the ramp's T^4 range (from
    the first bucket that carries a count per ordinate up to where the ramp
    clips at intensity_max). The chroma walks the Planckian order: red leads at
    a burning crate's 1263 game, blue leads at the table top.

    BREAKS IF: the table is sampled out of order, a channel is scaled
    separately, or the intensity curve stops being monotone (a negative
    exponent, a clip below its floor).
    """
    _k, table = checked_in
    per_ord = (table * 4096) >> 16
    peak = per_ord.max(axis=0)
    assert np.all(np.diff(peak) >= 0), "a hotter bucket emits less in its peak channel"
    first = int(np.argmax(peak > 0))
    from config import CFG
    i_max = float(CFG.render.blackbody.intensity_max)
    _chroma, inten = ramp.emission_at_kelvin(GEN.bucket_kelvin())
    clip = int(np.argmax(inten >= i_max))
    assert first < clip
    assert np.all(np.diff(peak[first:clip]) > 0), "flat inside the T^4 range"
    b_burn = (1263 << 16) >> 18
    assert int(np.argmax(table[:, b_burn])) == 0          # red leads in a fire
    assert int(np.argmax(table[:, -1])) == 2              # blue leads at the top
    assert table[2, b_burn] == 0                          # no blue below 1900 K (Helland)


def test_the_faintest_glow_the_ramp_shows_is_resolved(checked_in, ramp):
    """PROPERTY (decision 4, the second floor -- the one that SETS L_FINE_BITS):
    the faintest glow blackbody.py's ramp ever shows -- the smallest non-zero
    intensity its public per-frame map returns (its LUT's first step, 7.2e-8 of
    the reference level) -- lands >= 2^8 counts per ordinate on an opaque cell
    at the S16 weight: I_min * 2^k / 16 >= 256. And every bucket the ramp shows
    at least that bright is resolved to >= 2^8 per ordinate in the table itself.

    PAIR (the criterion binds): two bits coarser (k - 2) the same glow lands
    below 2^8, so the currency is not over-provisioned by accident -- k = 36
    would clear it by 2^0.28, k = 37 by 2^1.28.

    BREAKS IF: L_FINE_BITS drops below 36, the ramp's glow floor or exponent
    moves so its faintest step dims by 2x, or the table's rounding eats the
    first glowing buckets.
    """
    k, table = checked_in
    t_q = ((4 * np.arange(GEN.E_TABLE_SIZE) + 2) << 16).astype(np.int64)
    _chroma, inten_pub = ramp.chroma_intensity(t_q.reshape(1, -1))
    shown = inten_pub[inten_pub > 0].astype(np.float64)
    i_min = float(shown.min())
    per_ord = i_min * float(1 << k) / 16.0
    assert per_ord >= 256.0, (i_min, per_ord)
    assert i_min * float(1 << (k - 2)) / 16.0 < 256.0
    _c, inten = ramp.emission_at_kelvin(GEN.bucket_kelvin())
    bright = inten >= i_min
    assert bright.sum() > 3000
    peak_ord = ((table * 4096) >> 16).max(axis=0)
    assert int(peak_ord[bright].min()) >= 256
    print(f"faintest shown glow: intensity {i_min:.3e} -> {per_ord:.1f} = "
          f"2^{math.log2(per_ord):.2f} counts per ordinate at k = {k}")


def test_the_engine_compiles_the_same_numbers_in_the_same_currency(checked_in):
    """PROPERTY: the engine's L° (LightEmissionTable.table(), compiled from the
    .inc) is the checked-in table value for value; its currency
    (LightEmissionTable.fine_bits, bp.L_FINE_BITS) is the header's; the
    headroom door the .cpp static_asserts is 2^44; and PhysicsEngine owns one.

    BREAKS IF: the build compiles a stale .inc, or the binding reports another
    currency than the table was generated at.
    """
    import breach_physics as bp
    k, table = checked_in
    eng_tbl = np.asarray(bp.LightEmissionTable().table(), dtype=np.int64)
    assert eng_tbl.shape == (3, GEN.E_TABLE_SIZE)
    assert np.array_equal(eng_tbl, table)
    assert int(bp.L_FINE_BITS) == int(bp.LightEmissionTable.fine_bits) == k
    assert int(bp.L_TABLE_TOP_MAX) == 1 << 44 and int(bp.L_CHANNELS) == 3
    eng = bp.PhysicsEngine()
    assert np.array_equal(np.asarray(eng.light_emission.table()), table)
    assert eng.light_requested is False


def test_the_reference_reads_the_same_numbers(checked_in):
    """PROPERTY: the integer reference's L_LIVE (sweep_ref_q.load_l_table) is
    the checked-in table value for value, in the same currency -- ONE copy of
    the numbers in the tree, read by the spec and compiled by the engine.

    BREAKS IF: the reference grows its own table, or the two parsers diverge.
    """
    k, table = checked_in
    assert R.L_FINE_BITS == k
    assert np.array_equal(np.asarray(R.L_LIVE, dtype=np.int64), table)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
