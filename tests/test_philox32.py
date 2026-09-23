"""philox32 (arc #63 P1) -- vendoring gate for the Philox-4x32-10 RNG package.

P1 adopts github.com/eriiiko/philox32 (MIT, commit 6c33a3d0737e1f9f64b7dd84b
dbda02fbbc863d2) as breach's second sanctioned raw-draw source (chapter 14
door 4, amended by docs/architecture/engine/17_swarm_units.md §3). No sim
call site uses it yet -- this file is the proof the vendored copies are
correct and agree with each other, not a behavior test.

Properties protected, one per test class/function below:

  (a) the pure-Python twin (src/simulation/philox32.py) reproduces every
      Random123-published KAT vector for philox4x32-10 -- breaks if the
      Python port of the block function is wrong in any round.
  (b) the vendored C++ header (cpp/src/philox32.h), called through the
      breach_physics binding, reproduces the same KAT vectors -- breaks if
      the C++ port (or the pybind11 glue) diverges from the algorithm.
  (c) Python and C++ agree over a large seeded sweep of counters/keys,
      including the 32-bit edge words -- breaks if the two languages
      disagree anywhere the KAT vectors (only 3 points) don't cover; this is
      the property the whole vendoring exercise exists to buy (cross-machine
      bit-identity, engine/14 §2).
  (d) the Q16 output mappings (uniform_q16, angle_q16, below) agree Python
      vs C++ -- breaks if a mapping is re-derived instead of shared, or if
      one side rounds/truncates differently.
  (e) the vendored file bodies (below their added header comment) still hash
      to the value recorded in that header -- breaks the moment either
      vendored file is hand-edited without re-vendoring (the iron rule:
      "never hand-edit, update by re-vendoring").

Fewer, stronger tests: KAT vectors and the drift hash are oracles (not
mocks); the cross-language sweep is the actual property this patch buys.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import philox32 as pytwin  # noqa: E402  (path setup above)

try:
    import breach_physics as bp
except ImportError as exc:  # pragma: no cover - build missing
    pytest.skip(f"breach_physics extension not built: {exc}",
                allow_module_level=True)

if not hasattr(bp, "philox32_4x32_10"):  # pragma: no cover - stale build
    pytest.skip("breach_physics built without philox32_4x32_10 "
                "(rebuild cpp/ after arc #63 P1)", allow_module_level=True)

KAT_PATH = ROOT / "tests" / "data" / "philox4x32_10_kat.txt"
PHILOX32_H = ROOT / "cpp" / "src" / "philox32.h"
PHILOX32_PY = ROOT / "src" / "simulation" / "philox32.py"
MARKER = "vendored body (byte-for-byte from upstream) begins below ----\n"


# ---------------------------------------------------------------------------
# KAT vector loading (line format documented in the file's own header).
# ---------------------------------------------------------------------------
def _load_kat_vectors():
    """Parse tests/data/philox4x32_10_kat.txt into (ctr, key, out) tuples.

    Format: ``philox4x32 10  ctr0 ctr1 ctr2 ctr3  key0 key1  out0 out1 out2
    out3`` (all hex), comments start with '#'. Random123's published vectors
    for the 10-round variant -- the independent oracle for both ports.
    """
    vectors = []
    for line in KAT_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        assert parts[0] == "philox4x32" and parts[1] == "10", (
            f"unexpected KAT line shape: {line!r}")
        hexwords = parts[2:]
        assert len(hexwords) == 10, f"expected 10 hex words, got {line!r}"
        words = [int(h, 16) for h in hexwords]
        ctr, key, out = words[0:4], words[4:6], words[6:10]
        vectors.append((tuple(ctr), tuple(key), tuple(out)))
    return vectors


KAT_VECTORS = _load_kat_vectors()


def test_kat_file_has_the_three_published_vectors():
    """Sanity on the fixture itself: non-empty, well-formed KAT file.

    Guards against a truncated or mis-vendored copy silently emptying the
    parametrization below (which would make (a)/(b) vacuously pass)."""
    assert len(KAT_VECTORS) >= 3
    for ctr, key, out in KAT_VECTORS:
        assert len(ctr) == 4 and len(key) == 2 and len(out) == 4
        assert all(0 <= w < 2**32 for w in ctr + key + out)


# ---------------------------------------------------------------------------
# (a) Python twin vs KAT.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ctr,key,expected", KAT_VECTORS)
def test_python_twin_matches_kat_vector(ctr, key, expected):
    """The Python twin's block function reproduces Random123's published
    vector. Breaks on any round-count, multiplier, Weyl-increment, or
    permutation-wiring mistake in the Python port."""
    assert pytwin.philox32_4x32_10(ctr, key) == expected


# ---------------------------------------------------------------------------
# (b) C++ (via the binding) vs KAT.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ctr,key,expected", KAT_VECTORS)
def test_cpp_binding_matches_kat_vector(ctr, key, expected):
    """The SHIPPED C++ header, called through breach_physics, reproduces the
    same published vector. Breaks on a C++ port bug or a pybind11 glue bug
    (e.g. an argument transposed or truncated across the language boundary)."""
    got = bp.philox32_4x32_10(ctr[0], ctr[1], ctr[2], ctr[3], key[0], key[1])
    assert got == expected


# ---------------------------------------------------------------------------
# (c) Python == C++ over a large seeded sweep, including edge words.
# ---------------------------------------------------------------------------
def _edge_words():
    return (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF)


def _random_sweep_cases(n: int, seed: int):
    """n deterministic random (ctr, key) pairs plus every all-edge-word
    combination of ctr/key, so the sweep hits both the generic case and the
    32-bit boundary words a random draw would rarely land on exactly."""
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        ctr = tuple(rng.randint(0, 2**32 - 1) for _ in range(4))
        key = tuple(rng.randint(0, 2**32 - 1) for _ in range(2))
        cases.append((ctr, key))
    edges = _edge_words()
    for e in edges:
        cases.append(((e, e, e, e), (e, e)))
        cases.append(((e, 0, e, 0), (0, e)))
    return cases


SWEEP_CASES = _random_sweep_cases(n=4000, seed=0xC0FFEE)


def test_python_matches_cpp_over_random_and_edge_sweep():
    """Python and the shipped C++ agree bit-for-bit over a large seeded sweep
    of (ctr, key) pairs, including every combination of 32-bit edge words (0,
    1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF). Only 3 points are
    covered by the KAT file; this is the property the vendoring exists to
    buy -- breaks the instant either port disagrees anywhere the 3 KAT
    vectors don't reach (e.g. a sign-extension or carry bug at a word
    boundary)."""
    mismatches = []
    for ctr, key in SWEEP_CASES:
        py_out = pytwin.philox32_4x32_10(ctr, key)
        cpp_out = bp.philox32_4x32_10(ctr[0], ctr[1], ctr[2], ctr[3],
                                       key[0], key[1])
        if py_out != cpp_out:
            mismatches.append((ctr, key, py_out, cpp_out))
    assert not mismatches, (
        f"{len(mismatches)}/{len(SWEEP_CASES)} mismatches, first: "
        f"{mismatches[0]}")


def test_python_matches_cpp_vectorised_numpy_path():
    """The vectorised C++ path (philox32_4x32_10_np, a thin loop over the
    same scalar block function) agrees with the vectorised numpy Python twin
    over the same sweep -- breaks if the numpy dtype/shape glue on either
    side silently promotes to float or mis-strides the (n, 4)/(n, 2) arrays."""
    np = pytest.importorskip("numpy")
    ctrs = np.array([c for c, _ in SWEEP_CASES], dtype=np.uint32)
    keys = np.array([k for _, k in SWEEP_CASES], dtype=np.uint32)
    py_out = pytwin.philox32_4x32_10_np(ctrs, keys)
    cpp_out = bp.philox32_4x32_10_np(ctrs, keys)
    assert np.array_equal(py_out, np.asarray(cpp_out))


# ---------------------------------------------------------------------------
# (d) Q16 output mappings agree Python vs C++.
# ---------------------------------------------------------------------------
def test_q16_mappings_agree_python_vs_cpp():
    """uniform_q16, angle_q16 and below() -- the three integer-only output
    mappings -- produce identical results in Python and C++ over the same
    swept words (plus n in {0, 1, 2, 6, 10, 1_000_000} for below()). Breaks
    if a mapping's shift/multiply constant drifts between the two ports, or
    if the checked-in TWO_PI_Q16 constant disagrees."""
    assert bp.PHILOX32_TWO_PI_Q16 == pytwin.TWO_PI_Q16
    words = [w for ctr, key in SWEEP_CASES for w in ctr] + list(_edge_words())
    ns = (0, 1, 2, 6, 10, 1_000_000, 2**32 - 1)
    for w in words:
        assert bp.philox32_uniform_q16(w) == pytwin.philox32_uniform_q16(w)
        assert bp.philox32_angle_q16(w) == pytwin.philox32_angle_q16(w)
        for n in ns:
            assert bp.philox32_below(w, n) == pytwin.philox32_below(w, n), (
                f"below mismatch at w={w:#x} n={n}")


# ---------------------------------------------------------------------------
# (e) drift check: vendored bodies still match their recorded hash.
# ---------------------------------------------------------------------------
def _recorded_hash_and_body(path: Path) -> tuple[str, bytes]:
    """Split a vendored file into (recorded SHA-256 from its header comment,
    raw bytes of everything after the marker line)."""
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    marker_idx = text.index(MARKER)
    header = text[: marker_idx + len(MARKER)]
    body_start = len(header.encode("utf-8"))
    body = raw[body_start:]
    hash_line = next(
        l for l in header.splitlines()
        if "SHA-256 of the vendored file" in l)
    recorded = hash_line.split(":")[-1].strip()
    return recorded, body


@pytest.mark.parametrize("path", [PHILOX32_H, PHILOX32_PY])
def test_vendored_body_matches_its_recorded_hash(path):
    """The vendored file's body (everything below the added header comment's
    marker line) still hashes to the SHA-256 recorded in that header. This is
    the honest drift check: it breaks the moment someone hand-edits the
    vendored body without updating the header, which is exactly the mistake
    the 'never hand-edit, re-vendor instead' rule exists to catch."""
    recorded, body = _recorded_hash_and_body(path)
    actual = hashlib.sha256(body).hexdigest()
    assert actual == recorded, (
        f"{path.name}: vendored body hash {actual} != header-recorded "
        f"{recorded} -- the body was edited without re-vendoring")
