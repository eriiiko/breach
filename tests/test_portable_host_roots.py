"""The observation's and the decoder's roots are the IEEE square root (#29
P8.0b): arithmetic a CUDA kernel reproduces bit for bit, so the device
observation and the device decoder can be held EXACT against the host.

The exhaustive proof that the lob reach moved no quantized value when its
root changed from ``** 0.5`` to ``math.sqrt`` is ``tools/reach_sqrt_sweep.py``
(every d2 < 2**31, run once, result in the P8.0b commit and the design's
as-built); this file keeps a dense sample of it in the suite.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O
from simulation.unit_fixed import quantize_scalar

ROOT = Path(__file__).resolve().parent.parent


class _At:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def center_tile_x(self):
        return self._x

    def center_tile_y(self):
        return self._y


def _exact_reach_q(d2: int) -> int:
    """round-half-up(sqrt(d2) * 65536) in exact integers."""
    return (math.isqrt(d2 << 34) + 1) // 2


def test_onto_reach_is_the_exactly_rounded_q16_root_and_unmoved():
    """Property: ``onto_reach_q`` is the Q16 lob reach of the exactly rounded
    root, and equals the old ``pow`` form's quantized value, for every
    offset within 256 tiles and a seeded sample up to the int32 position
    range. Breaks if the root leaves IEEE ``sqrt`` for something that moves a
    quantized reach (a float32 root, a different rounding, a pow that is
    off by more than the quantum's slack), i.e. a change of game state."""
    me = _At(40_000 // 2, 40_000 // 2)
    for dx in range(-256, 257):
        for dy in range(0, 257):
            q = A.onto_reach_q(me, (me._x + dx, me._y + dy))
            d2 = dx * dx + dy * dy
            assert q == _exact_reach_q(d2), (dx, dy)
            assert q == quantize_scalar(float(d2) ** 0.5), (dx, dy)
    rng = np.random.default_rng(29_0802)
    origin = _At(0, 0)
    for dx, dy in rng.integers(0, 32768, (20_000, 2)).tolist():
        q = A.onto_reach_q(origin, (dx, dy))
        d2 = dx * dx + dy * dy
        assert q == _exact_reach_q(d2), (dx, dy)
        assert q == quantize_scalar(float(d2) ** 0.5), (dx, dy)


def test_dense_d2_quantized_root_unmoved_by_sqrt():
    """Property: over EVERY d2 below 2**22 (offsets past 2000 tiles, beyond
    any map's throw), ``quantize(sqrt(d2))`` equals ``quantize(pow(d2,
    0.5))`` although the two roots differ before quantizing (the C pow is
    not correctly rounded on this machine's runtime). Breaks if the
    quantizer's rounding changes so the slack between them is lost."""
    k = np.arange(1 << 22, dtype=np.float64)
    old = np.power(k, np.full(k.shape, 0.5))    # array exponent: the C pow
    new = np.sqrt(k)
    q_old = np.floor(old * 65536.0 + 0.5)
    q_new = np.floor(new * 65536.0 + 0.5)
    assert np.array_equal(q_old, q_new)


def test_length_is_sqrt_of_the_sum_of_squares():
    """Property: the observation's one ``length`` is IEEE ``sqrt(dx*dx +
    dy*dy)`` in float64, the same value the wind plane's array form gives
    element for element. Breaks if ``length`` becomes ``hypot`` or another
    algorithm on inputs where they part (non-integer offsets), or the array
    form stops matching the scalar one."""
    rng = np.random.default_rng(29_0803)
    v = rng.standard_normal((50_000, 2)) * 1e5
    arr = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])
    for (dx, dy), a in zip(v.tolist(), arr.tolist()):
        assert O.length(dx, dy) == math.sqrt(dx * dx + dy * dy) == a


def _roots_called(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "hypot":
            bad.append(f"{path.name}:{node.lineno} hypot")
        elif (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
              and isinstance(node.right, ast.Constant)
              and isinstance(node.right.value, float)):
            bad.append(f"{path.name}:{node.lineno} ** {node.right.value}")
    return bad


def test_observation_and_decoder_take_no_libm_root():
    """Property: the observation builders, the action spec and the policy
    decoder reach a length only through IEEE ``sqrt``: no ``hypot`` and no
    float-exponent ``**`` (the C runtime's ``pow``). Breaks when either is
    written into one of these files, which the device twins could not hold
    EXACT."""
    files = [ROOT / "src" / "simulation" / "obs_spec_v1.py",
             ROOT / "src" / "simulation" / "action_spec_v1.py",
             ROOT / "src" / "simulation" / "team_picture.py",
             ROOT / "src" / "simulation" / "reward_v1.py",
             ROOT / "src" / "simulation" / "sensor_accessor.py",
             ROOT / "src" / "control_policy.py",
             ROOT / "src" / "rl_env.py"]
    bad = [b for f in files for b in _roots_called(f)]
    assert not bad, bad
