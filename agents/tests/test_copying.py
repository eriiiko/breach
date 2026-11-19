"""The copy (agents/copying.py), the choosing module's ``mode``, the copy's
training file and export: rung 2 note 2e, sections 5 to 7. Each test names
the property it protects and the change that must break it."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from agents import choosing as CH
from agents import copying as C
from agents import records as RC
from agents import saved as SV
from agents import training_file as TF
from agents.network import SoldierNet
from agents.settings import CopySettings
from agents.sizes import sizes_for
from agents.spec import reduced
from agents.tests import _records as W
from agents.tests import _thousand as K
from simulation import action_spec_v1 as A

E = 3                                   # entries of the stub's list
NK = 6                                  # grenade rows (today's)


class Stub:
    """Fixed raw outputs per head (the later heads read a table row of the
    earlier choice), so a case can be worked by hand."""

    def __init__(self, n):
        h = A.head_sizes(E, NK)
        self.aim_keys = torch.zeros(n, h[1], 1)
        self.t = {"feet": torch.zeros(n, h[0]), "aim": torch.zeros(n, h[1]),
                  "trigger": torch.zeros(h[1], h[2]),
                  "hand": torch.zeros(h[1], h[3]),
                  "arg": torch.zeros(h[3], h[4])}

    def grads(self):
        for v in self.t.values():
            v.requires_grad_(True)
        return self

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


def _two():
    """The hand-worked case: two decisions.

    0: feet legal {0, 3}, raw (0, ln 3) -> the bot's 3 at 3/4; aim only
       "walk" (not counted); trigger {0, 1} even -> the bot's 1 at 1/2; hand
       only "nothing", argument only 0 (not counted).
    1: feet only stand (not counted); aim {walk, entry 0}, raw (0, ln 4) ->
       the bot's entry at 4/5; trigger only 0; hand {nothing, throw 0} even
       -> the bot's throw at 1/2; argument all five even -> the bot's 2 at
       1/5."""
    h = A.head_sizes(E, NK)
    out = Stub(2)
    out.t["feet"][0, 3] = math.log(3)
    out.t["aim"][1, A.AIM_TOKENS] = math.log(4)
    masks = {k: torch.zeros(2, n, dtype=torch.bool)
             for k, n in zip(A.MASK_KEYS, h)}
    masks["mask_feet"][0, [0, 3]] = True
    masks["mask_feet"][1, 0] = True
    masks["mask_aim"][0, A.AIM_WALK] = True
    masks["mask_aim"][1, [A.AIM_WALK, A.AIM_TOKENS]] = True
    masks["mask_trigger"][0, [0, 1]] = True
    masks["mask_trigger"][1, 0] = True
    masks["mask_hand"][0, A.HAND_NOTHING] = True
    masks["mask_hand"][1, [A.HAND_NOTHING, A.HAND_THROW0]] = True
    actions = torch.tensor([[3, A.AIM_WALK, 1, A.HAND_NOTHING, 0],
                            [0, A.AIM_TOKENS, 0, A.HAND_THROW0, 2]])
    argm = CH.arg_mask(actions[:, 3], actions[:, 1])
    want = [-(math.log(3 / 4) + math.log(1 / 2)),
            -(math.log(4 / 5) + math.log(1 / 2) + math.log(1 / 5))]
    return out, masks, actions, argm, want


def test_the_copys_loss_is_the_choosing_modules_probability():
    """Protects: the copy's loss is minus the choosing module's
    log-probability of the bot's choice over the counted heads, checked on
    two decisions worked by hand (feet floor 0). Breaks when the copy
    computes a probability of its own (another softmax, the masks not
    applied, an uncounted head counted)."""
    out, masks, actions, argm, want = _two()
    loss, d = C.copy_loss(out, masks, actions, argm, 0.0)
    torch.testing.assert_close(loss, torch.tensor(want), rtol=0, atol=1e-6)
    torch.testing.assert_close(loss, -CH.evaluate(out, masks, actions).logp)


def test_the_feet_floor_is_spread_over_the_legal_options_only():
    """Protects: with the floor f, the feet term is ``(1 - f) log p(bot) +
    f / (legal - 1) * sum of log p(other legal)``, and an illegal feet
    option has no part in it (no gradient on its raw output); the other
    heads have no floor. Breaks when the floor is spread over every option
    (illegal ones too), or laid on another head."""
    f = 0.02
    out, masks, actions, argm, want = _two()
    out.grads()
    loss, _ = C.copy_loss(out, masks, actions, argm, f)
    lv = loss.detach()
    feet0 = (1 - f) * math.log(3 / 4) + f * math.log(1 / 4)
    hand0 = -math.log(1 / 2)                       # the trigger's term
    assert abs(float(lv[0]) - (-feet0 + hand0)) < 1e-6
    assert abs(float(lv[1]) - want[1]) < 1e-6      # its feet did not count
    loss.sum().backward()
    g = out.t["feet"].grad
    illegal = ~masks["mask_feet"]
    assert float(g[illegal].abs().max()) == 0.0
    assert float(g[0, [0, 3]].abs().min()) > 0.0


def test_a_head_that_did_not_count_adds_nothing():
    """Protects: a head with one legal option adds nothing to the copy's
    loss nor its gradient (decision 1's trigger, decision 0's aim). Breaks
    when the copy grades every head, counted or not."""
    out, masks, actions, argm, want = _two()
    out.t["trigger"][A.AIM_TOKENS] = torch.tensor([5.0, -3.0])
    out.t["aim"][0] = torch.randn(out.t["aim"].shape[1])
    out.grads()
    loss, _ = C.copy_loss(out, masks, actions, argm, 0.02)
    assert abs(float(loss.detach()[1]) - want[1]) < 1e-6
    loss.sum().backward()
    assert float(out.t["trigger"].grad[A.AIM_TOKENS].abs().max()) == 0.0
    assert float(out.t["aim"].grad[0].abs().max()) == 0.0


def test_a_stored_argument_mask_against_the_games_rule_is_refused():
    """Protects: the records' argument mask is held equal to the game's rule
    for the bot's hand and aim; a record where they differ is refused, never
    trained on. Breaks when the copy ignores the stored mask."""
    out, masks, actions, argm, _ = _two()
    bad = argm.clone()
    bad[1, 4] = False
    with pytest.raises(C.CopyRefused, match="argument mask"):
        C.copy_loss(out, masks, actions, bad, 0.0)


def test_mode_is_the_most_likely_legal_option_by_the_same_walk(sizes6, rng):
    """Protects: ``choosing.mode`` gives, per head, the legal option of the
    largest chance, each later head told the network's own earlier choices
    (or the given ones); its free choice is what it gives when told its own
    choices. Breaks when it ignores a mask or conditions a head on other
    choices than the walk's."""
    from agents import fake
    torch.manual_seed(0)
    net = SoldierNet(reduced(False), sizes6).eval()
    obs = fake.make_obs(sizes6, 64, rng)
    masks = CH.masks_of(obs)
    with torch.no_grad():
        out, _ = net({k: torch.as_tensor(obs[k]) for k in
                      ("vector", "list", "list_mask", "near_f", "near_u",
                       "far_f", "far_u")})
        m = CH.mode(out, masks)
        assert torch.equal(CH.mode(out, masks, given=m), m)
        p, _ = CH.chances(out.feet(), masks["mask_feet"])
        assert torch.equal(m[:, 0], p.argmax(dim=-1))
        CH.evaluate(out, masks, m)                  # every choice legal
        d = CH.draw(out, masks, torch.Generator().manual_seed(1))
        g = CH.mode(out, masks, given=d.actions)
        p, _ = CH.chances(out.hand(d.actions[:, 1]), masks["mask_hand"])
        assert torch.equal(g[:, 3], p.argmax(dim=-1))


def test_the_carried_memory_is_the_soldiers_memory_from_its_match_start(
        tmp_path):
    """Protects: in the memory copy, the memory a lane starts a stretch
    with is the soldier's memory from the first decision of its match (the
    network run over every decision before it), and zero at its first:
    never a window started in mid-match from an empty memory. Checked with
    step size 0 (the weights do not move). Breaks when the lanes' memory is
    reset between stretches, carried from another lane, or not taken at a
    short stretch's true end."""
    root = W.folder(tmp_path, seeds=(1, 2), levels=("ship_a",))
    c = RC.census(root, workers=1)
    files = {int(RC.read(f.path)[0]["vector"][0, 0, 0]) // 100000: f.path
             for f in c["records"]}
    torch.manual_seed(0)
    net = SoldierNet(reduced(True), sizes_for(3)).eval()
    cop = C.Copier(net, CopySettings(step_size=0.0, lump=8, batch=8),
                   "cpu", planned=10 ** 9)
    dealer = RC.StretchDealer(c["records"], seed=0, lanes=2, stretch=4,
                              buffer_matches=2, workers=1)
    state = net.initial_state(2)
    checked = 0
    for b in dealer.batches(0):
        for j in range(2):
            if b["length"][j] == 0:
                continue
            m = int(b["vector"][0, j, 0])
            fno, t, slot = m // 100000, (m % 100000) // 10, m % 10
            if t == 0:
                assert b["first"][0, j]
                continue
            d = RC.unpack(files[fno])
            idx = np.flatnonzero(d["slot"] == slot)[:t]
            obs = {k: torch.as_tensor(d[k][idx][:, None]) for k in
                   ("vector", "list", "list_mask", "near_f", "near_u",
                    "far_f", "far_u")}
            with torch.no_grad():
                _, ref = net.forward_sequence(
                    obs, None, torch.as_tensor(d["first"][idx][:, None]))
            torch.testing.assert_close(state[j:j + 1], ref, rtol=0,
                                       atol=1e-5)
            checked += 1
        _, state = cop.step_stretches(b, state)
    assert checked >= 4


def test_the_copy_training_file_sets_every_field_and_refuses_by_name():
    """Protects: a copying run's numbers reach ``CopySettings`` only through
    its file's ``[copy]`` table, every field a key; an unknown key, a wrong
    type or another table is refused by name. Breaks when a field is added
    without a key or a key is ignored."""
    from dataclasses import fields
    vals = {}
    for f in fields(CopySettings):
        v = getattr(CopySettings(), f.name)
        vals[f.name] = (not v) if isinstance(v, bool) else (
            v + 1 if isinstance(v, int) else v * 0.5)
    vals["lump"] = 256
    vals["batch"] = 1024
    vals["feet_floor"] = 0.1
    raw = {"network": {"spec": "v1", "memory": True, "width": "double"},
           "copy": vals}
    s, net = TF.parse_copy(raw)
    assert {f.name: getattr(s, f.name) for f in fields(s)} == vals
    from agents.spec import double, v1
    assert net == double(True) and net.join == 2 * v1().join
    for bad, name in ((dict(raw, train={"planned_rounds": 1}), "train"),
                      ({**raw, "copy": {**vals, "nope": 1}}, "nope"),
                      ({**raw, "copy": {**vals, "passes": 1.5}}, "passes"),
                      ({**raw, "copy": {**vals, "blank_prev": 1}},
                       "blank_prev")):
        with pytest.raises(TF.TrainingFileRefused, match=name):
            TF.parse_copy(bad)


def test_a_copys_export_names_what_it_was_started_from(tmp_path):
    """Protects: the copy is written through ``agents.saved`` only, as a
    soldier's export that names the records' spec names and what it was
    started from, and loads back through ``load_soldier``. Breaks when the
    copy writes its own file or drops ``started_from``."""
    torch.manual_seed(0)
    net = SoldierNet(reduced(False), sizes_for(3))
    started = {"copy": {"records": "x", "census_sha256": "h" * 64,
                        "split": "held iff seed % 5 == 0", "pass": 3}}
    specs = {"observation": "v1", "action": "v1", "reward_profile": None}
    SV.export_soldier(tmp_path / "copy.pt", net, specs=specs,
                      started_from=started)
    d = SV.load(tmp_path / "copy.pt")
    assert d["started_from"] == started and d["specs"] == specs
    torch.manual_seed(1)
    other = SoldierNet(reduced(False), sizes_for(3))
    SV.load_soldier(tmp_path / "copy.pt", other)
    for (k, a), b in zip(net.state_dict().items(),
                         other.state_dict().values()):
        assert torch.equal(a, b), k
    with pytest.raises(ValueError, match="one of them"):
        SV.export_soldier(tmp_path / "x.pt", net)


def test_the_measures_score_a_perfect_copy_and_the_baselines(tmp_path):
    """Protects: the measures' bookkeeping: a network that always gives the
    bot's choice scores 1 on every counted head and on all heads at once;
    "repeat the previous choice" scores 0 on the changes; the decisions and
    files are counted. Breaks when the counted heads, the previous choice
    or the changes are taken from the wrong decisions."""
    root = W.folder(tmp_path, seeds=(1,), levels=("ship_a",))
    c = RC.census(root, workers=1)
    sizes = A.head_sizes(3 + RC.O.ENTRIES_BEYOND_SLOTS, NK)
    counts = C.OptionCounts(sizes)
    t = C.Tally(sizes, counts)
    n = 0
    for f in c["records"]:
        d = RC.unpack(f.path)
        counts.add(d)
        t.add(d, d["actions"].astype(np.int64), d["actions"].astype(np.int64))
        n += len(d["t"])
    r = t.result()
    assert r["decisions"] == n and r["files"] == 2
    assert r["all_heads_teacher_forced"] == r["all_heads_own_choices"] == 1.0
    for h, v in r["heads"].items():
        if v["decisions"]:
            assert v["agreement"] == 1.0, h
            assert v["changes"]["repeat_previous"] in (None, 0.0), h
    assert r["feet_angle_deg"]["mean"] in (None, 0.0)


@pytest.mark.tasks
@pytest.mark.skipif(not K.RECORDS.is_dir(),
                    reason=f"THOUSAND: the real records are not at "
                           f"{K.RECORDS} (set BREACH_RECORDS)")
def test_a_thousand_real_decisions_are_learnt_to_99_percent():
    """Protects (design 3.3, "is the data path sound?"): a network trained on
    1 000 living decisions of the real records, drawn across 50 training
    matches, agrees with the bots on at least 99 % of them on every counted
    head. Breaks when the path from the records to the loss is unsound
    (masks or actions misaligned with their observation, a head's
    conditioning wrong, the argument's mask from other choices) or the
    network cannot hold what it is shown. Reduced width on the processor
    (full width on the card passes in 400 steps; reduced in about 1 750:
    note 2e, as built), with a budget of 4 000 steps."""
    d = K.draw()
    assert len(d["files"]) >= K.MATCHES and len(d["t"]) == K.DECISIONS
    r = K.learn(d, reduced(False), steps=K.STEPS)
    assert all(a is None or a >= K.LINE for a in r["agreement"]), r
