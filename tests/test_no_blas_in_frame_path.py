"""NO BLAS IN THE FRAME PATH -- the per-frame / per-tick Python never hands a
contraction to the BLAS library (graphics G3 bug fix, 2026-10-29).

Born from Erik's first play of G3 on fire_smoke_studio: the HUD's "Upload:"
line went over 1000 ms per frame. Root cause, measured: ``np.tensordot`` in
``gas_relief.relief_group_layers`` (three calls per species group) and ``@``
in ``fire_vfx_ref.curl_noise_at`` (the embers' drift, every frame). numpy
sends those to BLAS (MKL in the ``data`` env), whose worker THREAD POOL is
pathological for tiny products while another process keeps the cores busy:
a (3 x 5) . (5 x 3136) product took 0.026 ms with one BLAS thread and
150-490 ms with the pool, under a 16-process CPU load. The game never gains
from BLAS threads (every render contraction is over a handful of species or
modes), so the rule is simple: per-frame code contracts with plain loops /
elementwise sums -- ``renderer.gas_medium.species_sum`` is THE helper for the
species axis.

This is an AST scan of every module the game loop runs per frame or per tick
(``renderer/``, ``src/simulation/``, ``ui/``, ``main.py``,
``level_lights.py``): a ``@`` / ``@=`` operator, or a call to ``tensordot``,
``dot``, ``matmul``, ``inner``, ``vdot``, ``linalg.*`` (except
``linalg.norm``, an elementwise sqrt-of-sum when given an ``axis``), or
``einsum(..., optimize=...)`` (the optimizer may route to BLAS). Comments
and docstrings never false-positive. To exempt a deliberate LOAD-TIME use,
put ``blas-exempt: <why>`` in a comment on the flagged line.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCAN = [ROOT / "renderer", ROOT / "src" / "simulation", ROOT / "ui",
        ROOT / "main.py", ROOT / "level_lights.py"]
BLAS_CALLS = {"tensordot", "dot", "matmul", "inner", "vdot"}
EXEMPT = "blas-exempt:"


def _files():
    for p in SCAN:
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.py"))


def _attr_chain(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _violations(path: Path, root: Path = ROOT):
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    out = []
    for node in ast.walk(ast.parse(src, filename=str(path))):
        what = None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            what = "@"
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.MatMult):
            what = "@="
        elif isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            name = chain.rsplit(".", 1)[-1]
            if name in BLAS_CALLS:
                what = chain + "()"
            elif ".linalg." in f".{chain}" and name != "norm":
                what = chain + "()"
            elif name == "einsum" and any(k.arg == "optimize"
                                          for k in node.keywords):
                what = chain + "(optimize=...)"
        if what is None:
            continue
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if EXEMPT in line:
            continue
        out.append(f"{path.relative_to(root)}:{node.lineno}: {what}  "
                   f"| {line.strip()}")
    return out


def test_no_blas_call_in_the_frame_path():
    """PROPERTY: no module the game loop runs per frame / per tick contracts
    through BLAS (``@``, tensordot, dot, matmul, inner, vdot, linalg.* bar
    norm, einsum with the optimizer), unless the line says ``blas-exempt:``
    and why. BREAKS IF: someone writes ``np.tensordot`` / ``a @ b`` back into
    e.g. ``gas_relief.relief_group_layers`` or ``fire_vfx_ref.curl_noise_at``
    -- the change that made fire_smoke_studio's upload > 1000 ms per frame
    under CPU contention (2026-10-29)."""
    bad = [v for f in _files() for v in _violations(f)]
    assert not bad, ("BLAS in the frame path (use plain loops / "
                     "gas_medium.species_sum, or mark a load-time use "
                     "'blas-exempt: <why>'):\n" + "\n".join(bad))


def test_the_lint_sees_what_it_bans(tmp_path):
    """PROPERTY: the scanner flags each banned form and honours the
    exemption. BREAKS IF the AST walk stops recognising one of them (the
    lint would then pass vacuously)."""
    f = tmp_path / "m.py"
    f.write_text(
        "import numpy as np\n"
        "a = x @ y\n"
        "a @= y\n"
        "b = np.tensordot(x, y, axes=1)\n"
        "c = x.dot(y)\n"
        "d = np.linalg.solve(x, y)\n"
        "e = np.linalg.norm(x, axis=-1)\n"
        "g = np.einsum('i,i', x, y, optimize=True)\n"
        "h = np.einsum('i,i', x, y)\n"
        "k = np.dot(x, y)  # blas-exempt: load time\n",
        encoding="utf-8")
    got = sorted(int(v.split(":")[1]) for v in _violations(f, tmp_path))
    assert got == [2, 3, 4, 5, 6, 8], got


def test_species_sum_is_the_contraction():
    """PROPERTY: ``species_sum`` is exactly the species-axis contraction it
    replaced (``tensordot`` over axis 0), for (S,) and (S, C) weights.
    BREAKS IF the helper drops a species or a channel, or transposes."""
    from renderer.gas_medium import species_sum
    rng = np.random.default_rng(7)
    rho = rng.random((5, 13, 17))
    w1 = rng.random(5)
    w3 = rng.random((5, 3))
    np.testing.assert_allclose(species_sum(w1, rho),
                               np.einsum("s,shw->hw", w1, rho), rtol=1e-13)
    np.testing.assert_allclose(species_sum(w3, rho),
                               np.einsum("sc,shw->chw", w3, rho), rtol=1e-13)
