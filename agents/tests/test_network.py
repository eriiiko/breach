"""Properties of network spec v1 (agents/network.py): plan section 4's test
table, rows 5-12, and design 8.6. Reduced width, processor. Each test names
the property and the change that must break it."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents import choosing as C
from agents import fake, network
from agents.print_network import parts, table
from agents.sizes import sizes_for
from agents.spec import reduced, v1
from simulation import action_spec_v1 as A
from simulation.reward_v1 import REWARD_GROUPS

ATOL = 1e-5


def _soldier(sizes, memory=True, seed=0):
    torch.manual_seed(seed)
    return network.SoldierNet(reduced(memory), sizes).eval()


def _value(sizes, seed=1):
    torch.manual_seed(seed)
    return network.ValueNet(reduced(), sizes).eval()


def _t(obs, keys=network.SOLDIER_KEYS):
    return network.as_tensors({k: obs[k] for k in keys})


def _all_chances(out, masks, actions):
    """Every head's chances given the choices ``actions`` (N, 5)."""
    a = actions.long()
    argm = C.arg_mask(a[:, 3], a[:, 1])
    return [C.chances(out.feet(), masks["mask_feet"])[0],
            C.chances(out.aim(), masks["mask_aim"])[0],
            C.chances(out.trigger(a[:, 1]), masks["mask_trigger"])[0],
            C.chances(out.hand(a[:, 1]), masks["mask_hand"])[0],
            C.chances(out.arg(a[:, 1], a[:, 3]), argm)[0]]


def test_the_pointer_follows_the_entry_not_its_place(sizes6, rng):
    """Protects: shuffling the list's entries (their masks, uids and aim
    options moved with them) leaves the probability of pointing at that
    ENTRY unchanged, and every other head's chances unchanged (the later
    heads given the same entry).
    Breaks when: the pointer or any reader follows a place in the list (a
    position embedding, a per-position weight, an entry read by index)."""
    net = _soldier(sizes6)
    n, E = 64, sizes6.n_entries
    obs = fake.make_obs(sizes6, n, rng)
    perm = rng.permutation(E)
    sh = dict(obs)
    for k in ("list", "list_mask", "list_uid"):
        sh[k] = obs[k][:, perm]
    sh["mask_aim"] = np.concatenate(
        [obs["mask_aim"][:, :A.AIM_TOKENS],
         obs["mask_aim"][:, A.AIM_TOKENS:][:, perm]], axis=1)
    with torch.no_grad():
        out, _ = net(_t(obs))
        d = C.draw(out, C.masks_of(obs), torch.Generator().manual_seed(2))
        a = d.actions.clone()
        inv = np.argsort(perm)                 # entry i now stands at inv[i]
        on_entry = a[:, 1] >= A.AIM_TOKENS
        a2 = a.clone()
        a2[on_entry, 1] = torch.as_tensor(
            inv[(a[on_entry, 1] - A.AIM_TOKENS).numpy()]) + A.AIM_TOKENS
        out2, _ = net(_t(sh))
        e1 = C.evaluate(out, C.masks_of(obs), a)
        e2 = C.evaluate(out2, C.masks_of(sh), a2)
        torch.testing.assert_close(e2.logp_heads, e1.logp_heads, rtol=0,
                                   atol=ATOL)
        p1 = _all_chances(out, C.masks_of(obs), a)
        p2 = _all_chances(out2, C.masks_of(sh), a2)
    assert on_entry.any()
    torch.testing.assert_close(p2[0], p1[0], rtol=0, atol=ATOL)
    torch.testing.assert_close(p2[1][:, :A.AIM_TOKENS], p1[1][:, :A.AIM_TOKENS],
                               rtol=0, atol=ATOL)
    torch.testing.assert_close(p2[1][:, A.AIM_TOKENS:][:, inv],
                               p1[1][:, A.AIM_TOKENS:], rtol=0, atol=ATOL)
    for h in (2, 3, 4):
        torch.testing.assert_close(p2[h], p1[h], rtol=0, atol=ATOL)


def test_a_longer_list_of_padding_gives_the_same_chances(sizes6, rng):
    """Protects: the same soldier's input padded to a longer list (the
    padding masked off) gives the same chances in every head, the padded
    aim options chance exactly 0 (an empty list slot changes nothing).
    Breaks when: the network depends on the list's length (a mean over all
    slots, padding read as keys, a size baked from E)."""
    net = _soldier(sizes6)
    obs = fake.make_obs(sizes6, 48, rng)
    padded = fake.pad_list(obs, sizes6.n_entries + 9)
    with torch.no_grad():
        out, _ = net(_t(obs))
        d = C.draw(out, C.masks_of(obs), torch.Generator().manual_seed(4))
        out2, _ = net(_t(padded))
        p1 = _all_chances(out, C.masks_of(obs), d.actions)
        p2 = _all_chances(out2, C.masks_of(padded), d.actions)
    E = sizes6.n_entries
    for h in (0, 2, 3, 4):
        torch.testing.assert_close(p2[h], p1[h], rtol=0, atol=ATOL)
    torch.testing.assert_close(p2[1][:, :A.AIM_TOKENS + E], p1[1], rtol=0,
                               atol=ATOL)
    assert (p2[1][:, A.AIM_TOKENS + E:] == 0).all()


def test_one_set_of_weights_runs_at_every_team_size(rng):
    """Protects: one set of weights (built for S = 6) runs at S = 1, 3 and
    6, its aim head as long as each list (18 + S + 33), and the record it
    carries is the same at every S.
    Breaks when: a size is baked in (a layer whose width is E, a head of a
    fixed length, a record naming the list's length)."""
    net = _soldier(sizes_for(6))
    for S in (1, 3, 6):
        s = sizes_for(S)
        assert network.SoldierNet(reduced(), s).record == net.record
        obs = fake.make_obs(s, 16, rng)
        with torch.no_grad():
            out, _ = net(_t(obs))
            d = C.draw(out, C.masks_of(obs))
        assert out.aim().shape == (16, s.heads[1])
        assert A.masked_choices(d.actions.numpy(), obs).max() == 0


def test_a_rows_output_does_not_depend_on_its_batch(sizes6, rng):
    """Protects: a row's outputs (every head of the soldier, the value
    network's five numbers) do not change with what else is in the batch,
    in learning mode as well as acting mode.
    Breaks when: a batch statistic creeps in (a batch norm, a mean over the
    batch, dropout drawing per batch)."""
    net, val = _soldier(sizes6), _value(sizes6)
    obs = fake.make_obs(sizes6, 32, rng)
    full = fake.make_full(sizes6, 32, rng)
    for mode in ("eval", "train"):
        getattr(net, mode)()
        getattr(val, mode)()
        with torch.no_grad():
            big, _ = net(_t(obs))
            vb = val(_t(obs, network.VALUE_OBS_KEYS), network.as_tensors(full))
            for i in (0, 7, 31):
                one = {k: v[i:i + 1] for k, v in obs.items()}
                o1, _ = net(_t(one))
                torch.testing.assert_close(o1.feet(), big.feet()[i:i + 1],
                                           rtol=0, atol=ATOL)
                torch.testing.assert_close(o1.aim(), big.aim()[i:i + 1],
                                           rtol=0, atol=ATOL)
                a = torch.tensor([A.AIM_TOKENS + 1])
                torch.testing.assert_close(o1.hand(a),
                                           big.hand(a.expand(32))[i:i + 1],
                                           rtol=0, atol=ATOL)
                v1_ = val(_t(one, network.VALUE_OBS_KEYS),
                          network.as_tensors({k: v[i:i + 1]
                                              for k, v in full.items()}))
                torch.testing.assert_close(v1_, vb[i:i + 1], rtol=0, atol=ATOL)


def test_no_layer_depends_on_the_batch_or_the_mode(sizes6):
    """Protects: layer normalisation only, no dropout, nothing with running
    statistics: no module of a batch-norm, instance-norm or dropout type in
    either network. Breaks when: such a layer is added."""
    banned = (torch.nn.modules.batchnorm._BatchNorm,
              torch.nn.modules.instancenorm._InstanceNorm,
              torch.nn.modules.dropout._DropoutNd)
    for m in (network.SoldierNet(reduced(), sizes6),
              network.ValueNet(reduced(), sizes6)):
        bad = [type(x).__name__ for x in m.modules() if isinstance(x, banned)]
        assert not bad


def _seq(obs, L, n):
    return {k: torch.as_tensor(v).reshape((L, n) + v.shape[1:])
            for k, v in obs.items() if k in network.SOLDIER_KEYS}


def test_memory_step_by_step_equals_one_stretch(sizes6, rng):
    """Protects: with memory on, feeding a stretch decision by decision
    equals feeding it as one stretch from the same start (every head, the
    final memory), and a zero start mid-stretch (a match's first decision)
    is honoured: from there on the outputs equal a fresh stretch from zero.
    Breaks when: the memory path differs between acting and replay (another
    cell, a reset applied after the step, a reset ignored in one path)."""
    net = _soldier(sizes6, memory=True)
    L, n, k = 8, 5, 3
    obs = fake.make_obs(sizes6, L * n, rng)
    first = torch.zeros(L, n, dtype=torch.bool)
    first[k, 1] = True
    first[k, 3] = True
    h0 = torch.randn(n, net.memory.hidden_size)
    with torch.no_grad():
        seq_out, h_seq = net.forward_sequence(_seq(obs, L, n), h0, first)
        feet_seq = seq_out.feet().view(L, n, -1)
        h = h0
        for t in range(L):
            step = {k_: v[t] for k_, v in _seq(obs, L, n).items()}
            o, h = net(step, h, first[t])
            torch.testing.assert_close(o.feet(), feet_seq[t], rtol=0,
                                       atol=ATOL)
            torch.testing.assert_close(o.core, seq_out.core.view(L, n, -1)[t],
                                       rtol=0, atol=ATOL)
        torch.testing.assert_close(h, h_seq, rtol=0, atol=ATOL)
        # a fresh stretch from zero at k for the rows that started a match
        tail = {k_: v[k:] for k_, v in _seq(obs, L, n).items()}
        fresh, _ = net.forward_sequence(tail, torch.zeros_like(h0))
        got = seq_out.core.view(L, n, -1)[k:]
        want = fresh.core.view(L - k, n, -1)
        torch.testing.assert_close(got[:, [1, 3]], want[:, [1, 3]], rtol=0,
                                   atol=ATOL)
        assert not torch.allclose(got[:, 0], want[:, 0], atol=ATOL)


def test_with_memory_off_there_is_no_state(sizes6, rng):
    """Protects: with the memory switched off there is no memory module, no
    state in or out, and a state offered is refused.
    Breaks when: the switch leaks (a GRU built anyway, a state returned or
    silently ignored)."""
    net = _soldier(sizes6, memory=False)
    assert net.memory is None and net.initial_state(4) is None
    assert not any("memory" in n for n, _ in net.named_parameters())
    obs = fake.make_obs(sizes6, 4, rng)
    with torch.no_grad():
        _, state = net(_t(obs))
        assert state is None
        _, state = net.forward_sequence(_seq(obs, 2, 2))
        assert state is None
        with pytest.raises(ValueError, match="no memory"):
            net(_t(obs), torch.zeros(4, reduced().core))


def test_the_windows_are_taken_only_in_the_served_number_formats(sizes6, rng):
    """Protects: both networks take the arrays in the served number formats
    (float16 planes, uint8 planes and masks, float32 tables: the game's own
    shape tables), wrapped without conversion, and refuse any other format,
    so acting and learning convert in one place only.
    Breaks when: a silent conversion is added on one path (a float32 window
    accepted, as_tensors casting)."""
    net, val = _soldier(sizes6), _value(sizes6)
    obs = fake.make_obs(sizes6, 4, rng)
    full = fake.make_full(sizes6, 4, rng)
    t = network.as_tensors(obs)
    for k, (dt, _) in network.served_formats(sizes6).items():
        if k in t:
            assert t[k].dtype == network._TORCH[dt], k
    with torch.no_grad():
        net(_t(obs))
        val(_t(obs, network.VALUE_OBS_KEYS), network.as_tensors(full))
        for key in ("near_f", "far_u", "list"):
            bad = _t(obs)
            bad[key] = bad[key].to(torch.float64)
            with pytest.raises(TypeError, match="served as"):
                net(bad)
        badf = network.as_tensors(full)
        badf["full_watched"] = badf["full_watched"].float()
        with pytest.raises(TypeError, match="served as"):
            val(_t(obs, network.VALUE_OBS_KEYS), badf)


def test_a_count_plane_is_squashed_and_still_tells_counts_apart(sizes6):
    """Protects: whatever byte a plane serves, no channel the map readers
    get from it stands far above the flags beside it (a count is read
    through a fixed squash), and a count plane still reads a larger count
    as larger.
    Breaks when: a count plane is fed raw (a horde of 30 in one block is 30
    beside flags of 1), or the squash flattens two counts into one."""
    for window in (sizes6.near, sizes6.far):
        w, nf = window.side, len(window.f_planes)
        f = torch.zeros(1, nf, w, w, dtype=torch.float16)
        seen = {}
        for v in (1, 2, 30, 255):
            u = torch.full((1, len(window.u_planes), w, w), v,
                           dtype=torch.uint8)
            x = network.window_planes(window, f, u)[:, nf:]
            ch = 0
            for name, codes in window.u_planes:
                n_ch = max(1, len(codes))
                if name in window.count_planes:
                    seen.setdefault(name, []).append(float(x[0, ch].max()))
                ch += n_ch
        for name, vals in seen.items():
            assert vals == sorted(set(vals)), name      # strictly growing
            assert vals[-1] < 8.0, name                 # 255 units: still small
    assert sizes6.far.count_planes, "the far window serves count planes"


def test_a_kind_past_its_table_fails_loudly(sizes6, rng):
    """Protects: a kind column's value outside its table's rows is an
    error, never clamped or read as another kind.
    Breaks when: the kinds share one table with offsets and no check, or a
    clamp is added."""
    net = _soldier(sizes6)
    obs = fake.make_obs(sizes6, 2, rng)
    k = sizes6.vector_kinds[0]
    obs["vector"][0, k.pos] = k.rows + 3
    with pytest.raises(IndexError):
        net(_t(obs))


def test_the_value_network_answers_per_reward_group_and_stands_apart(sizes6,
                                                                    rng):
    """Protects: the value network gives one output per reward group in the
    environment's order (reward_v1.REWARD_GROUPS), shares no weight with the
    soldier's network, and the soldier's network refuses the full view (so
    scrambling the full view changes no choice).
    Breaks when: the outputs are reordered or counted from a literal, a
    part is shared between the two, or the soldier starts reading
    info["full_view"]."""
    net, val = _soldier(sizes6), _value(sizes6)
    assert val.groups == tuple(REWARD_GROUPS)
    obs = fake.make_obs(sizes6, 6, rng)
    full = fake.make_full(sizes6, 6, rng)
    with torch.no_grad():
        v = val(_t(obs, network.VALUE_OBS_KEYS), network.as_tensors(full))
    assert v.shape == (6, len(REWARD_GROUPS))
    ids = {p.data_ptr() for p in net.parameters()}
    assert not any(p.data_ptr() in ids for p in val.parameters())
    mixed = dict(_t(obs))
    mixed.update(network.as_tensors(full))
    with pytest.raises(ValueError, match="full view"):
        net(mixed)


def test_the_weight_count_is_printed_and_every_weight_is_in_the_table(sizes6,
                                                                      capsys):
    """Protects: the table of parts accounts for every weight of both
    networks (its rows add up to the module's own count); the full-width
    counts are printed and reported against design 8's 1 to 2 million (not
    forced). Breaks when: a part is left out of the table (a new top-level
    parameter not listed)."""
    for label, net in (("soldier v1", network.SoldierNet(v1(), sizes6)),
                       ("value v1", network.ValueNet(v1(), sizes6)),
                       ("soldier v1 no memory",
                        network.SoldierNet(v1(False), sizes6)),
                       ("soldier reduced", network.SoldierNet(reduced(),
                                                              sizes6))):
        total = sum(p.numel() for p in net.parameters())
        assert sum(n for _, n in parts(net)) == total
        assert table(label, net).splitlines()[-1].strip().endswith(
            f"{total:,}")
        with capsys.disabled():
            print(f"\n[weights] {label}: {total:,} (design 8: the soldier "
                  f"1 to 2 million, the value network about 1.1 million)",
                  end="")


@pytest.mark.parametrize("make", [reduced, v1])
def test_a_fresh_network_gives_every_legal_option_nearly_the_same_chance(
        sizes6, rng, make):
    """Protects: a near-even first policy (note 2b, 6) -- a fresh soldier's
    network, at reduced and at full width, gives every counted head a
    variety above 0.99 of the most it could be on every made-up row, so no
    legal option starts favoured before any learning.
    Breaks when: the heads' last layers start at PyTorch's default scale
    (the 2a network: shares down to 0.97 on some rows), or one head (the
    aim's or the hand's question, which point at vectors) is left out of
    the scaling."""
    torch.manual_seed(7)
    net = network.SoldierNet(make(False), sizes6).eval()
    obs = fake.make_obs(sizes6, 256, rng, p_invalid=0.0, p_windup=0.0)
    with torch.no_grad():
        out, _ = net(_t(obs))
        d = C.draw(out, C.masks_of(obs), torch.Generator().manual_seed(1))
    for h, head in enumerate(A.HEADS):
        counted = d.used[:, h]
        assert counted.any(), head
        assert float(d.variety[counted, h].min()) > 0.99, head
