"""The value targets' scale (agents/value_scale.py, PopArt): note 2b, 5;
critique lens 2 minor 2. Each test names the property and the change that
must break it."""
from __future__ import annotations

import torch

from agents.value_scale import ValueScale


def _layer(seed=0, n_in=6, n_out=5):
    torch.manual_seed(seed)
    return torch.nn.Linear(n_in, n_out)


def _targets(mean, sd):
    t = torch.randn(256, 5, dtype=torch.float64) * sd + mean
    t[:, 3] = 0.0                                 # always-zero group
    return t


def test_every_move_after_the_first_leaves_every_real_output_where_it_was():
    """Protects: after the scale's first move, every further move toward a
    batch's targets (means and spreads of very different sizes, one group
    always 0) corrects the last layer so that every real-unit output
    (spread x normalised + mean) is what it was before, for any input.
    Breaks when: the last layer's correction is skipped or uses the new
    spread for both terms (the real outputs then jump by the scale's move:
    the MAPPO running normaliser, not PopArt)."""
    layer = _layer()
    scale = ValueScale(layer, share=0.05, floor=0.01)
    h = torch.randn(32, 6)
    scale.update(_targets(10.0, 5.0))             # the first move
    for k, (mean, sd) in enumerate(((3.0, 100.0), (-50.0, 2.0),
                                    (400.0, 30.0))):
        with torch.no_grad():
            before = scale.real(layer(h))
        scale.update(_targets(mean, sd))
        with torch.no_grad():
            after = scale.real(layer(h))
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-4)
        assert scale.rounds == k + 2


def test_the_first_move_leaves_a_fresh_layer_alone():
    """Protects: the scale's FIRST move (from mean 0, spread 1 to the first
    batch's own, here hundreds) does not touch the fresh last layer, so its
    untaught output, near 0 in normalised units, reads as the batch's mean
    and no weight is divided by the spread; the always-zero group's weights
    are not multiplied by 1 / floor.
    Breaks when: the first move is corrected like the later ones: the
    weights shrink by the spread and the bias lands far from 0, which Adam
    (a step size per step whatever the gradient) needs thousands of steps
    to undo -- the "scale" task then stays at a random player's score."""
    layer = _layer()
    w0, b0 = layer.weight.detach().clone(), layer.bias.detach().clone()
    scale = ValueScale(layer, share=0.05, floor=0.01)
    t = _targets(690.0, 540.0)
    scale.update(t)
    assert torch.equal(layer.weight, w0) and torch.equal(layer.bias, b0)
    assert scale.rounds == 1
    h = torch.randn(64, 6)
    with torch.no_grad():
        real = scale.real(layer(h))
    # a fresh output (order 1 in normalised units) now sits around the mean
    assert float((real[:, 0].mean() - t[:, 0].mean()).abs()) \
        < 2 * float(scale.spread[0])
    assert float(real[:, 3].abs().max()) < 1.0    # the zero group stays small


def test_the_first_round_is_unbiased_and_a_zero_group_hits_the_floor():
    """Protects: the moving mean and mean square are corrected for the first
    rounds' bias (after one round the scale IS the batch's mean and spread,
    not 5 % of it), and a group that is always 0 gets the floor 0.01 as its
    spread, so its normalised targets are 0, never 0 / 0.
    Breaks when: the bias correction is dropped (the first rounds' spread
    near sqrt(0.05) of the truth), or the floor is removed (NaN)."""
    layer = _layer()
    scale = ValueScale(layer, share=0.05, floor=0.01)
    targets = torch.zeros(1000, 5, dtype=torch.float64)
    targets[:, 0] = torch.linspace(-100, 100, 1000, dtype=torch.float64)
    targets[:, 1] = 7.0
    scale.update(targets)
    torch.testing.assert_close(scale.mean[0], targets[:, 0].mean())
    torch.testing.assert_close(scale.spread[0],
                               targets[:, 0].std(unbiased=False))
    assert abs(float(scale.mean[1]) - 7.0) < 1e-12
    assert float(scale.spread[1]) == 0.01 and float(scale.spread[3]) == 0.01
    n = scale.normalise(targets)
    assert torch.isfinite(n).all()
    assert float(n[:, 3].abs().max()) == 0.0
    # the second round moves only a share of the way toward the new batch
    scale.update(targets + 10.0)
    m = float(scale.mean[1])
    assert 7.0 < m < 17.0 and abs(m - (7.0 + 10.0 * 0.05 / (1 - 0.95 ** 2))) \
        < 1e-9
