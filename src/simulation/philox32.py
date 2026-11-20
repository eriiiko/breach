# philox32.py -- VENDORED verbatim, do not hand-edit.
#
# Source: github.com/eriiiko/philox32 (MIT, Copyright (c) 2026 Erik Steen),
# file philox32/philox32.py, commit 6c33a3d0737e1f9f64b7dd84bdbda02fbbc863d2.
# SHA-256 of the vendored file at that commit: 725f8475ae80343b63587b9b306ea52ba45d31e622be28f4744e4092caf4e6a2
# To update: re-vendor from that commit (copy the file verbatim, refresh this
# header and its recorded hash) -- never hand-edit the body below the marker.
#
# Algorithm citation:
# John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw, "Parallel
# Random Numbers: As Easy as 1, 2, 3", Proceedings of the 2011 International
# Conference for High Performance Computing, Networking, Storage and
# Analysis (SC'11), DOI 10.1145/2063384.2063405.
#
# ---- vendored body (byte-for-byte from upstream) begins below ----
"""philox32 -- Philox-4x32-10, pure-Python twin of philox32.h (plus a numpy variant).

Algorithm: Salmon, Moraes, Dror, Shaw, "Parallel random numbers: as easy as
1, 2, 3", SC'11, DOI 10.1145/2063384.2063405.  Reference implementation:
Random123 (BSD-3-Clause); this module is gated against its published
known-answer vectors (tests/kat_philox4x32_10.txt).

License: MIT, Copyright (c) 2026 Erik Steen.

Everything here is integer arithmetic on values in [0, 2**32), so the results
are bit-identical to the C++/CUDA header.  The scalar functions are the STRICT
twin: they accept any int-like value (``operator.index``: Python ints, numpy
integers) and reject bools, floats and anything outside [0, 2**32) -- the C++
helpers take uint32_t and truncate silently.  The numpy variants use explicit
uint32/uint64 dtypes throughout (no Python-int fallback, no float) and are the
path for bulk work; the scalar path re-validates on every call.

Recommended layout for agent simulations (see philox32.h for the reasoning):
    key     = (seed_lo, seed_hi)                        -- the seed and nothing else
    counter = (tick, draw_index, stream_salt, agent_id)
"""

import operator

__version__ = "0.1.0"

MASK32 = 0xFFFFFFFF

# Round multipliers and Weyl key increments (Random123 PHILOX_M4x32_* / PHILOX_W32_*).
M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85
ROUNDS = 10

# round(2*pi * 65536): 2*pi in Q16.16.  A checked-in constant, never computed.
TWO_PI_Q16 = 411775


def _u32(name, v):
    """-> v as a plain int, if it is an int-like (not bool, not float) in [0, 2**32)."""
    if isinstance(v, bool):
        raise ValueError(f"{name} must be an integer in [0, 2**32), got bool {v!r}")
    try:
        i = operator.index(v)
    except TypeError:
        raise ValueError(f"{name} must be an integer in [0, 2**32), got {v!r}") from None
    if not 0 <= i <= MASK32:
        raise ValueError(f"{name} must be in [0, 2**32), got {i}")
    return i


def _u32s(name, values, n):
    if len(values) != n:
        raise ValueError(f"{name} must have {n} words, got {len(values)}")
    return tuple(_u32(name, v) for v in values)


def philox32_4x32_10(ctr, key):
    """Philox-4x32-10 block function.  ctr: 4 ints, key: 2 ints, all in [0, 2**32).

    Returns a tuple of four ints in [0, 2**32).  Pure function of its inputs.
    """
    c0, c1, c2, c3 = _u32s("ctr", ctr, 4)
    k0, k1 = _u32s("key", key, 2)
    for r in range(ROUNDS):
        if r > 0:
            k0 = (k0 + W0) & MASK32
            k1 = (k1 + W1) & MASK32
        p0 = M0 * c0
        p1 = M1 * c2
        hi0, lo0 = p0 >> 32, p0 & MASK32
        hi1, lo1 = p1 >> 32, p1 & MASK32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0, c1, c2, c3)


# --------------------------------------------------------------- output mappings
def philox32_uniform_q16(w):
    """Uniform in [0, 1) as Q16.16 (65536 == 1.0): the top 16 bits of w.

    Range 0..65535.  Exactly uniform, but 65536 attainable values is also the
    resolution -- take the raw word for anything finer.
    """
    return _u32("w", w) >> 16


def philox32_angle_q16(w):
    """Uniform angle in [0, 2*pi) as Q16.16: (w * TWO_PI_Q16) >> 32.

    Range 0..TWO_PI_Q16-1.  Bias: each value owns 10430 or 10431 of the 2**32
    words (max relative excess 9.6e-5).
    """
    return (_u32("w", w) * TWO_PI_Q16) >> 32


def philox32_below(w, n):
    """Integer in [0, n): (w * n) >> 32 -- Lemire's multiply-shift, no division.

    Range 0..n-1 (n == 0 gives 0).  Bias: relative excess of the fattest bin
    < n / 2**32.  Deliberately no rejection-sampling variant: rejection makes
    the number of words consumed depend on the draw, which is a determinism
    hazard for draw_index bookkeeping in a lockstep simulation.
    """
    return (_u32("w", w) * _u32("n", n)) >> 32


# ------------------------------------------------------------------ the layout
def philox32_make_key(seed_lo, seed_hi):
    """The recommended key: (seed_lo, seed_hi) -- the seed and nothing else."""
    return (_u32("seed_lo", seed_lo), _u32("seed_hi", seed_hi))


def philox32_make_counter(tick, draw_index, stream_salt, agent_id):
    """The recommended counter: (tick, draw_index, stream_salt, agent_id)."""
    return (_u32("tick", tick), _u32("draw_index", draw_index),
            _u32("stream_salt", stream_salt), _u32("agent_id", agent_id))


def philox32_draw(seed_lo, seed_hi, agent_id, tick, draw_index, stream_salt):
    """The block function on the recommended layout -> four words."""
    return philox32_4x32_10(
        philox32_make_counter(tick, draw_index, stream_salt, agent_id),
        philox32_make_key(seed_lo, seed_hi),
    )


# --------------------------------------------------------------------------- numpy
def _require_uint32(name, a):
    import numpy as np

    a = np.asarray(a)
    if a.dtype != np.uint32:
        raise TypeError(f"{name} must be uint32, got {a.dtype}")
    return a


def philox32_4x32_10_np(ctr, key):
    """Vectorised Philox-4x32-10.  ctr: uint32 (n, 4), key: uint32 (n, 2) -> uint32 (n, 4).

    Exact: products are formed in uint64, everything else in uint32 with the
    natural wrap-around.  Both inputs must already be uint32 -- no implicit
    conversion, so a caller can never feed floats or oversized ints by accident.
    """
    import numpy as np

    ctr = _require_uint32("ctr", ctr)
    key = _require_uint32("key", key)
    if ctr.ndim != 2 or ctr.shape[1] != 4 or key.ndim != 2 or key.shape[1] != 2 or ctr.shape[0] != key.shape[0]:
        raise ValueError(f"expected ctr (n, 4) and key (n, 2), got {ctr.shape} / {key.shape}")

    m0 = np.uint64(M0)
    m1 = np.uint64(M1)
    w0 = np.uint32(W0)
    w1 = np.uint32(W1)
    shift32 = np.uint64(32)

    c0 = ctr[:, 0].copy()
    c1 = ctr[:, 1].copy()
    c2 = ctr[:, 2].copy()
    c3 = ctr[:, 3].copy()
    k0 = key[:, 0].copy()
    k1 = key[:, 1].copy()
    for r in range(ROUNDS):
        if r > 0:
            k0 = k0 + w0  # uint32 + uint32 wraps
            k1 = k1 + w1
        p0 = c0.astype(np.uint64) * m0
        p1 = c2.astype(np.uint64) * m1
        hi0 = (p0 >> shift32).astype(np.uint32)
        lo0 = p0.astype(np.uint32)  # low 32 bits
        hi1 = (p1 >> shift32).astype(np.uint32)
        lo1 = p1.astype(np.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    out = np.empty((ctr.shape[0], 4), dtype=np.uint32)
    out[:, 0] = c0
    out[:, 1] = c1
    out[:, 2] = c2
    out[:, 3] = c3
    return out


def philox32_uniform_q16_np(w):
    """Vectorised philox32_uniform_q16 over a uint32 array."""
    import numpy as np

    return _require_uint32("w", w) >> np.uint32(16)


def philox32_angle_q16_np(w):
    """Vectorised philox32_angle_q16 over a uint32 array (exact, via uint64)."""
    import numpy as np

    w = _require_uint32("w", w)
    return ((w.astype(np.uint64) * np.uint64(TWO_PI_Q16)) >> np.uint64(32)).astype(np.uint32)


def philox32_below_np(w, n):
    """Vectorised philox32_below: w uint32 array, n uint32 array or scalar (broadcast)."""
    import numpy as np

    w = _require_uint32("w", w)
    n = _require_uint32("n", n)
    return ((w.astype(np.uint64) * n.astype(np.uint64)) >> np.uint64(32)).astype(np.uint32)


__all__ = [
    "philox32_4x32_10",
    "philox32_uniform_q16",
    "philox32_angle_q16",
    "philox32_below",
    "philox32_make_key",
    "philox32_make_counter",
    "philox32_draw",
    "philox32_4x32_10_np",
    "philox32_uniform_q16_np",
    "philox32_angle_q16_np",
    "philox32_below_np",
    "TWO_PI_Q16",
    "ROUNDS",
    "MASK32",
]
