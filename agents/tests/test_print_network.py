"""The table-printing script's processor path (agents/print_network.py).
The card path is the orchestrator's to run at the patch's boundary."""
from __future__ import annotations

import pytest
import torch

from agents import network, print_network
from agents.sizes import sizes_for
from agents.spec import reduced


@pytest.mark.parametrize("memory", [True, False])
@pytest.mark.parametrize("half", [False, True])
def test_one_learning_step_runs_on_the_processor(memory, half):
    """Protects: the measured learning step (both networks forward over
    stretches, the choosing module's probability, one backward) runs end to
    end on the processor, with and without memory, at both precisions, and
    leaves gradients cleared. Breaks when: the step the card measures
    stops matching the networks' or the choosing module's interfaces."""
    s = sizes_for(3)
    torch.manual_seed(0)
    sol, val = network.SoldierNet(reduced(memory), s), network.ValueNet(
        reduced(), s)
    assert print_network.learning_step(sol, val, s, 32, "cpu", half) is None
    assert all(p.grad is None for p in sol.parameters())


def test_the_script_prints_both_networks_at_both_widths(capsys):
    """Protects: the script prints the table of parts and weights for the
    soldier and the value network at full and reduced width. Breaks when:
    a network or a width drops out of the printout."""
    assert print_network.main(["--slots", "3"]) == 0
    text = capsys.readouterr().out
    for title in ("soldier, network spec v1, full width",
                  "value network, network spec v1, full width",
                  "soldier, network spec v1, reduced width",
                  "value network, network spec v1, reduced width"):
        assert title in text
    assert text.count("total") >= 4
