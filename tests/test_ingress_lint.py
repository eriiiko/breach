"""THE NUMBER-INGRESS LINT — enforce the ingress rule over src/simulation/.

The rule (Erik, 2026-07-04 — born from the tick-0 __unit_hp__ cross-machine
divergence, docs/lenovo_dev_setup.md §8b): synced simulation state may only be
written through four doors —

  1. integer / Q16.16 arithmetic (incl. the C++ kit: mul/div/sqrt/trig _q16)
  2. config constants quantized ONCE onto the Q16.16 grid at load
  3. audited algebraic float bridges: only + - * / sqrt on deterministic
     inputs, quantized at the write boundary
  4. the seeded integer RNG stream, with transforms that pass doors 1-3

BANNED in src/simulation/ (each is machine/library/version-dependent):

  - libm transcendentals: math.sin/cos/exp/log/atan2/pow/... (math.sqrt is
    ALLOWED — IEEE requires it correctly rounded, door 3)
  - BLAS/LAPACK: anything .linalg, scipy
  - distribution methods on RNG objects: .multivariate_normal / .normal /
    .gamma / ... (they hide libm/LAPACK inside; draw raw integers/uniform
    dyadics instead and transform through doors 1-3)
  - random-module transcendental transforms: random.gauss & friends
  - UNSEEDED np.random.default_rng() (OS entropy in the sim layer)

This is an AST scan, so comments/docstrings never false-positive. To exempt a
deliberate, audited use, put ``ingress-exempt: <reason>`` in a comment on the
flagged line or within the 6 lines above it (see materials.py's config-time
log2 shift tables for the pattern — exemptions must say WHY they are safe).

Known v1 gaps (documented, not enforced): float ``**`` (goes through libm
pow), and C++-side ingress (governed by /fp:strict + the digest gates).
Prototypes/, tools/, renderer code are deliberately OUT of scope — experiment
freely there; the gate applies to the synced sim layer.

The cross-machine digest attestation stays the empirical backstop; this lint
is the cheap tripwire that fires at commit time, not at the next attestation.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "src" / "simulation"

# pathfinding.py lives at repo ROOT (not src/simulation/) but is imported
# lazily by three sim modules (ai_zombie.py, simulation.py, timeline.py) and
# its A* results feed synced state (unit orders/movement) — scan it too, so
# it can't dodge the ingress rule just by sitting outside SIM_DIR (issue #15).
EXTRA_SIM_FILES = [ROOT / "pathfinding.py"]

BANNED_MATH_FUNCS = {
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "expm1", "log", "log2", "log10", "log1p",
    "pow", "hypot", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "erf", "erfc", "gamma", "lgamma",
}
BANNED_RANDOM_FUNCS = {
    "gauss", "normalvariate", "lognormvariate", "expovariate",
    "gammavariate", "betavariate", "paretovariate", "vonmisesvariate",
    "weibullvariate", "triangular",
}
# Distribution methods banned on ANY receiver (rng.normal(...), legacy
# np.random.normal(...), ...). Raw-stream methods (.integers, .random,
# .choice, .shuffle, .permutation, .bytes) stay allowed — they are pure
# integer/dyadic transforms of the bitstream.
BANNED_DIST_METHODS = {
    "multivariate_normal", "standard_normal", "normal", "randn",
    "lognormal", "standard_gamma", "gamma", "beta", "exponential",
    "standard_exponential", "poisson", "chisquare", "noncentral_chisquare",
    "dirichlet", "laplace", "logistic", "standard_cauchy", "standard_t",
    "vonmises", "wald", "weibull", "rayleigh", "pareto", "zipf",
    "negative_binomial", "binomial", "geometric", "hypergeometric",
    "multinomial", "gumbel", "power", "noncentral_f", "logseries",
}
BANNED_MODULE_IMPORTS = {"scipy"}
EXEMPT_TOKEN = "ingress-exempt"
EXEMPT_LOOKBACK = 6   # pragma may sit on the line or up to N lines above


def _is_exempt(lines: list[str], lineno: int) -> bool:
    lo = max(0, lineno - 1 - EXEMPT_LOOKBACK)
    return any(EXEMPT_TOKEN in l for l in lines[lo:lineno])


def _scan_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src, filename=str(path))
    rel = path.relative_to(ROOT)
    hits: list[str] = []

    def hit(node: ast.AST, why: str):
        if not _is_exempt(lines, node.lineno):
            hits.append(f"{rel}:{node.lineno}  {why}")

    for node in ast.walk(tree):
        # import scipy / from scipy import ...
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BANNED_MODULE_IMPORTS:
                    hit(node, f"import {a.name} (BLAS/LAPACK-backed)")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in BANNED_MODULE_IMPORTS:
                hit(node, f"from {node.module} import ... (BLAS/LAPACK-backed)")
            elif mod == "math":
                bad = [a.name for a in node.names if a.name in BANNED_MATH_FUNCS]
                if bad:
                    hit(node, f"from math import {', '.join(bad)} (libm transcendental)")
        elif isinstance(node, ast.Attribute):
            base = node.value
            base_name = base.id if isinstance(base, ast.Name) else None
            if base_name == "math" and node.attr in BANNED_MATH_FUNCS:
                hit(node, f"math.{node.attr} (libm transcendental — use the "
                          f"fixed-point kit / unit_fixed)")
            elif base_name == "random" and node.attr in BANNED_RANDOM_FUNCS:
                hit(node, f"random.{node.attr} (libm-backed transform)")
            elif node.attr == "linalg":
                hit(node, "*.linalg (BLAS/LAPACK — not bit-portable)")
            elif node.attr in BANNED_DIST_METHODS:
                hit(node, f"*.{node.attr}() (distribution method — hides "
                          f"libm/LAPACK; draw raw stream + transform via doors 1-3)")
        elif isinstance(node, ast.Call):
            # unseeded default_rng() — OS entropy in the sim layer
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == "default_rng"
                    and not node.args and not node.keywords):
                hit(node, "np.random.default_rng() with NO seed (OS entropy)")
    return hits


def test_no_banned_number_ingress_in_simulation():
    files = sorted(SIM_DIR.rglob("*.py")) + EXTRA_SIM_FILES
    assert files, f"no python files found under {SIM_DIR}"
    violations: list[str] = []
    for path in files:
        violations.extend(_scan_file(path))
    assert not violations, (
        "NUMBER-INGRESS RULE violations in src/simulation/ (see this test's "
        "docstring + docs/lenovo_dev_setup.md §8b). Every number entering "
        "synced sim state must pass an approved door; fix the call or add an "
        "audited 'ingress-exempt: <why safe>' comment:\n  "
        + "\n  ".join(violations)
    )


if __name__ == "__main__":
    sys.exit(0 if not test_no_banned_number_ingress_in_simulation() else 1)
