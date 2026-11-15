"""Properties of THE choosing module (agents/choosing.py): plan section 4's
test table, rows 1-4, and design 8.2 / 4.4. Each test names the property
and the change that must break it."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents import choosing as C
from agents import fake, network
from agents.spec import reduced
from simulation import action_spec_v1 as A


def _soldier(sizes, memory=True, seed=0):
    torch.manual_seed(seed)
    return network.SoldierNet(reduced(memory), sizes).eval()


def _inputs(obs):
    return network.as_tensors({k: obs[k] for k in network.SOLDIER_KEYS})


class StubOutputs:
    """Fixed raw outputs per head, the later heads depending on the earlier
    choices through lookup tables, so a test can perturb one head."""

    def __init__(self, n, sizes, n_entries, gen):
        h = A.head_sizes(n_entries, sizes.n_kinds)
        self.aim_keys = torch.zeros(n, h[1], 1)
        self.t = {
            "feet": torch.randn(n, h[0], generator=gen),
            "aim": torch.randn(n, h[1], generator=gen),
            "trigger": torch.randn(h[1], h[2], generator=gen),
            "hand": torch.randn(h[1], h[3], generator=gen),
            "arg": torch.randn(h[3], h[4], generator=gen),
        }
        for v in self.t.values():
            v.requires_grad_(True)

    @property
    def n_rows(self):
        return self.t["feet"].shape[0]

    def feet(self):
        return self.t["feet"]

    def aim(self):
        return self.t["aim"]

    def trigger(self, aim):
        return self.t["trigger"][aim]

    def hand(self, aim):
        return self.t["hand"][aim]

    def arg(self, aim, hand):
        return self.t["arg"][hand]


@pytest.mark.parametrize("explore", [0.0, 0.05])
def test_recomputed_probability_equals_the_drawn_one(sizes6, rng, explore):
    """Protects: the probability of a drawn decision, worked out again from
    the same inputs (a fresh forward pass) and the same choices, equals the
    one returned at drawing -- per head and for the whole decision.
    Breaks when: acting and learning differ anywhere in the module (a mask
    applied in one walk only, the explore mix in one only, a head's
    conditioning on another choice than the drawn one)."""
    net = _soldier(sizes6)
    obs = fake.make_obs(sizes6, 256, rng)
    masks = C.masks_of(obs)
    with torch.no_grad():
        out, _ = net(_inputs(obs))
        d = C.draw(out, masks, torch.Generator().manual_seed(3), explore)
        out2, _ = net(_inputs(obs))
        e = C.evaluate(out2, masks, d.actions, explore)
    assert torch.equal(e.actions, d.actions)
    assert torch.equal(e.used, d.used)
    torch.testing.assert_close(e.logp_heads, d.logp_heads, rtol=0, atol=1e-5)
    torch.testing.assert_close(e.logp, d.logp, rtol=0, atol=1e-5)
    assert torch.isfinite(d.logp).all()


def test_forbidden_options_are_never_drawn(sizes6):
    """Protects: an option the masks forbid is never drawn, over many random
    masks, in every head, with and without the explore mix; the argument
    obeys the mask worked out from the drawn hand and aim.
    Breaks when: a mask is ignored or misapplied (e.g. the logits are not
    masked before the softmax, or the argument uses a stale mask)."""
    net = _soldier(sizes6)
    gen = torch.Generator().manual_seed(11)
    for seed in range(6):
        r = np.random.default_rng(seed)
        obs = fake.make_obs(sizes6, 400, r)
        masks = C.masks_of(obs)
        with torch.no_grad():
            out, _ = net(_inputs(obs))
            for explore in (0.0, 0.3, 1.0):
                a = C.draw(out, masks, gen, explore).actions.numpy()
                assert A.masked_choices(a, obs).max() == 0


def test_no_head_is_ever_without_a_legal_option(sizes6, rng):
    """Protects: every head of every row has a legal option -- the game's
    default (invalid-row) masks give exactly one, and a head offered none is
    refused loudly instead of turning into NaN.
    Breaks when: the module silently draws from an all-forbidden head, or
    the game's default masks lose their one legal option."""
    obs = fake.make_obs(sizes6, 4, rng)
    A.default_masks(obs, 0)
    for k in A.MASK_KEYS:
        assert obs[k][0].sum() == 1
    with pytest.raises(ValueError, match="no legal option"):
        C.chances(torch.zeros(2, 5), torch.zeros(2, 5, dtype=torch.bool))


def test_batched_heads_used_and_arg_mask_equal_the_games(sizes6, rng):
    """Protects: the batched "which heads counted" and argument mask equal
    the game's own functions (action_spec_v1.heads_used / arg_mask) on
    drawn cases covering every hand and aims on both sides of the tokens.
    Breaks when: the batched form drifts from the game's rule (a hand
    added to the throws, "onto" allowed for a token aim, the count
    threshold changed)."""
    E = sizes6.n_entries
    h = A.head_sizes(E, sizes6.n_kinds)
    n = 3000
    hand = rng.integers(0, h[3], size=n)
    aim = rng.integers(0, h[1], size=n)
    obs = fake.make_obs(sizes6, n, rng)
    game_arg = A.arg_mask(hand, aim)
    ours = C.arg_mask(torch.as_tensor(hand), torch.as_tensor(aim))
    np.testing.assert_array_equal(ours.numpy(), game_arg != 0)
    game_used = A.heads_used(obs, game_arg)
    used = C.heads_used(C.masks_of(obs), ours).numpy()
    for i, head in enumerate(A.HEADS):
        np.testing.assert_array_equal(used[:, i], game_used[head])
    # every (hand, aim class) pair at least once
    for hd in range(h[3]):
        for am in (0, A.AIM_WALK, A.AIM_KEEP, A.AIM_TOKENS, h[1] - 1):
            np.testing.assert_array_equal(
                C.arg_mask(torch.tensor([hd]), torch.tensor([am])).numpy(),
                A.arg_mask(np.array([hd]), np.array([am])) != 0)


def test_a_head_that_did_not_count_adds_nothing(sizes6, rng):
    """Protects: a head that did not count (one legal option) adds nothing
    to the decision's probability or to the variety, and receives no
    learning signal at all (zero gradient), whatever its raw outputs.
    Breaks when: an unused head is graded (its log-probability or entropy
    summed in, e.g. by dropping the `used` filter)."""
    n = 512
    obs = fake.make_obs(sizes6, n, rng)
    masks = C.masks_of(obs)
    gen = torch.Generator().manual_seed(5)
    out = StubOutputs(n, sizes6, sizes6.n_entries, gen)
    d = C.draw(out, masks, gen)
    assert (~d.used).any(dim=0).all(), "every head unused somewhere"
    assert torch.equal(d.logp_heads[~d.used],
                       torch.zeros_like(d.logp_heads[~d.used]))
    assert torch.equal(d.variety[~d.used], torch.zeros_like(d.variety[~d.used]))
    torch.testing.assert_close(d.logp, d.logp_heads.sum(-1))
    (d.logp.sum() + d.variety.sum()).backward()
    # the per-row outputs (feet, aim): an unused row's gets no gradient
    for head, key in ((0, "feet"), (1, "aim")):
        g = out.t[key].grad
        unused = ~d.used[:, head]
        assert torch.equal(g[unused], torch.zeros_like(g[unused]))
    # perturbing an unused head's outputs changes nothing
    before = C.evaluate(out, masks, d.actions)
    with torch.no_grad():
        out.t["feet"][~d.used[:, 0]] += 100 * torch.randn(
            int((~d.used[:, 0]).sum()), out.t["feet"].shape[1], generator=gen)
    after = C.evaluate(out, masks, d.actions)
    assert torch.equal(before.logp, after.logp)
    assert torch.equal(before.variety, after.variety)


def test_variety_is_a_share_of_each_heads_most(sizes6):
    """Protects: each counted head's variety is its entropy over the log of
    its legal options: 1 for even chances over the legal options of a head
    of any size, 0 for one option, finite gradients with forbidden options
    (no 0 x -inf). Breaks when: the variety is raw entropy (heads of 2 and
    49 options pushed unequally) or the masked entropy produces NaN."""
    for n_opt in (2, 5, 49):
        logits = torch.zeros(3, n_opt + 3, requires_grad=True)
        mask = torch.zeros(3, n_opt + 3, dtype=torch.bool)
        mask[:, :n_opt] = True
        mask[2] = False
        mask[2, 0] = True
        p, logp = C.chances(logits, mask)
        share = C.variety_share(p, logp, mask)
        torch.testing.assert_close(share[:2], torch.ones(2))
        assert share[2] == 0
        share.sum().backward()
        assert torch.isfinite(logits.grad).all()
        assert (p[~mask] == 0).all()


def test_the_explore_chance_reaches_every_legal_option_and_no_other():
    """Protects: the optional random-legal chance gives every legal option
    at least explore / n_legal, a forbidden one still exactly 0, and is 0 by
    default. Breaks when: the mix is spread over forbidden options too, or
    the default is not the plain chances."""
    logits = torch.tensor([[50.0, 0.0, 0.0, 9.0]])
    mask = torch.tensor([[True, True, False, True]])
    p0, _ = C.chances(logits, mask)
    pd, _ = C.chances(logits, mask, 0.0)
    assert torch.equal(p0, pd)
    p, logp = C.chances(logits, mask, 0.3)
    assert p[0, 2] == 0 and torch.isinf(logp[0, 2])
    assert (p[0, mask[0]] >= 0.1 - 1e-7).all()
    torch.testing.assert_close(p.sum(), torch.tensor(1.0))


def test_a_given_choice_the_masks_forbid_is_refused(sizes6, rng):
    """Protects: the probability of given choices is never computed for a
    forbidden choice (it would be -inf and poison learning); it is refused.
    Breaks when: evaluate stops checking the gathered log-probability."""
    obs = fake.make_obs(sizes6, 8, rng)
    masks = C.masks_of(obs)
    out = StubOutputs(8, sizes6, sizes6.n_entries,
                      torch.Generator().manual_seed(1))
    d = C.draw(out, masks)
    bad = d.actions.clone()
    bad[:, 3] = A.HAND_FLASHLIGHT              # reserved: always masked
    with pytest.raises(ValueError, match="forbid"):
        C.evaluate(out, masks, bad)
