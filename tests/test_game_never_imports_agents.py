"""The game never imports the trainer's side (#192 rung 2, design 1).

`agents/` (the networks, the choosing module, later the trainer) sits beside
`src/` and may import the game; the game must never import it, so the game
never depends on PyTorch and one tag fixes both without coupling them.
"""
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GAME = [REPO / "src", REPO / "renderer", REPO / "ui", REPO / "main.py"]


def _game_files():
    for root in GAME:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.py"))


def _imports_agents(tree: ast.AST) -> list:
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names
                     if a.name == "agents" or a.name.startswith("agents.")]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "agents" or node.module.startswith("agents."):
                hits.append(node.module)
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(
                node.func, "id", "")) in ("import_module", "__import__"):
            for arg in node.args[:1]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.split(".")[0] == "agents":
                    hits.append(arg.value)
    return hits


def test_nothing_in_the_game_imports_agents():
    """Protects: no file under src/, renderer/, ui/ or main.py imports the
    `agents` package (plain, from-, or importlib by a literal name).
    Breaks when: a game file starts importing agents (and so PyTorch)."""
    files = list(_game_files())
    assert any(f.name == "rl_env.py" for f in files), "the scan found the game"
    bad = {}
    for f in files:
        hits = _imports_agents(ast.parse(f.read_text(encoding="utf-8")))
        if hits:
            bad[str(f.relative_to(REPO))] = hits
    assert not bad, f"game files importing agents: {bad}"
