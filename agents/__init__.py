"""agents -- the trainer's side of Breach's learned soldiers (#192, rung 2).

Beside ``src/``, never inside it: the game never imports this package (a
scan in the game's suite, ``tests/test_game_never_imports_agents.py``,
holds it), so the game never depends on PyTorch. This package may import
the game's Python (``simulation.*`` from ``src/``) to read its spec tables.

Holds: the sizes (:mod:`agents.sizes`), the network spec and its record
(:mod:`agents.spec`), the networks (:mod:`agents.network`), the ONE
choosing module (:mod:`agents.choosing`), made-up inputs in the served
formats (:mod:`agents.fake`) and the table of parts and weights
(``python -m agents.print_network``).
"""
