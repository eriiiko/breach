"""agents/value_scale.py -- the value targets' scale (rung 2 note 2b, 5;
design 4.3), in the form that leaves the network's real-unit outputs
unchanged when the scale moves (PopArt: van Hasselt, Guez, Hessel, Mnih,
Silver, "Learning values across many orders of magnitude", NeurIPS 2016).

Per output (one per reward group) a mean and a spread. The value network's
last layer gives the NORMALISED estimate; the real one is ``spread *
normalised + mean``. Each round, in this order:

1. grade with the scale as it was while collecting (:meth:`real`);
2. move the mean and the mean square a share (0.05 a round) toward this
   batch's targets, corrected for the first rounds' bias the way Adam
   corrects its moments; the spread never under its floor (0.01), so a
   group that is always 0 does not divide by nothing (:meth:`update`);
3. change the last layer so that every real-unit output is what it was
   before the move (inside :meth:`update`);
4. learn on the targets normalised with the new scale (:meth:`normalise`).

All five outputs learn, the weight-0 ones too.
"""
from __future__ import annotations

import torch


class ValueScale:
    """The per-output mean and spread of the value targets; ``layer`` is
    the value network's last ``nn.Linear`` (the one :meth:`update`
    corrects)."""

    def __init__(self, layer: torch.nn.Linear, share: float, floor: float):
        n = layer.out_features
        dev = layer.weight.device
        self.layer = layer
        self.share = float(share)
        self.floor = float(floor)
        self.rounds = 0
        # the raw moving averages (zero-started; bias-corrected on reading)
        self._m1 = torch.zeros(n, dtype=torch.float64, device=dev)
        self._m2 = torch.zeros(n, dtype=torch.float64, device=dev)
        self.mean = torch.zeros(n, dtype=torch.float64, device=dev)
        self.spread = torch.ones(n, dtype=torch.float64, device=dev)

    def real(self, normalised):
        """Real-unit estimates from the network's normalised ones
        ``(..., n)``."""
        return (normalised.double() * self.spread + self.mean)

    def normalise(self, real):
        """Real-unit targets ``(..., n)`` in the current scale's units."""
        return (real.double() - self.mean) / self.spread

    @torch.no_grad()
    def update(self, targets) -> None:
        """Move the scale toward ``targets`` ``(M, n)`` (this round's value
        targets, value rows only, real units), then correct the last layer
        so that every real-unit output is unchanged."""
        targets = torch.as_tensor(targets, dtype=torch.float64,
                                  device=self.mean.device)
        if targets.shape[0] == 0:
            return
        old_mean, old_spread = self.mean.clone(), self.spread.clone()
        b = self.share
        self._m1 = (1 - b) * self._m1 + b * targets.mean(dim=0)
        self._m2 = (1 - b) * self._m2 + b * (targets ** 2).mean(dim=0)
        self.rounds += 1
        unbias = 1.0 - (1.0 - b) ** self.rounds
        m1, m2 = self._m1 / unbias, self._m2 / unbias
        self.mean = m1
        self.spread = (m2 - m1 ** 2).clamp(min=0.0).sqrt().clamp(
            min=self.floor)
        # real = spread * (W h + b) + mean stays what it was:
        # W' = W * old_spread / spread, b' = (old_spread * b + old_mean -
        # mean) / spread
        w, bias = self.layer.weight, self.layer.bias
        ratio = (old_spread / self.spread).to(w.dtype)
        w.mul_(ratio.unsqueeze(1))
        bias.copy_(((old_spread * bias.double() + old_mean - self.mean)
                    / self.spread).to(bias.dtype))


__all__ = ["ValueScale"]
