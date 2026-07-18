"""THE IMPORT-LIGHT GUARD — enforce entity design §3b's import rule.

The rule: ``import simulation.entities`` must succeed with NO compiled
``breach_physics`` importable and must NOT pull in ``simulation.simulation``
(the sim loop / physics stack). The map editor imports the registry directly
— like ``simulation.materials`` — and its palette must stay alive on a
machine (or in a state) where the C++ build is absent or broken.

Enforced in a SUBPROCESS with a clean sys.path (src/ only, no
cpp/build/Release), because the pytest process itself imports breach_physics
all over — module state here proves nothing. Same guard-test spirit as
tests/test_ingress_lint.py: a cheap tripwire at commit time.

Run:
    conda run -n data python -m pytest tests/test_entities_import_light.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_GUARD_SCRIPT = textwrap.dedent("""
    import importlib.util
    import sys

    sys.path.insert(0, {src!r})

    # Prove the environment really lacks the compiled physics (nothing on
    # sys.path provides it) — otherwise this guard would be vacuous.
    assert importlib.util.find_spec("breach_physics") is None, (
        "guard env broken: breach_physics IS importable in the clean "
        "subprocess; sys.path=" + repr(sys.path))

    import simulation.entities

    assert "simulation.simulation" not in sys.modules, (
        "import simulation.entities pulled in simulation.simulation "
        "(the sim loop) — import-light rule violated (entity design 3b)")
    assert "breach_physics" not in sys.modules
    # The machinery itself came up: exemplar registered, hash computable.
    assert "light" in simulation.entities.REGISTRY
    assert len(simulation.entities.registry_content_hash()) == 64
""")


def test_import_simulation_entities_is_light():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)   # nothing may sneak the build dir back in
    script = _GUARD_SCRIPT.format(src=str(ROOT / "src"))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"import-light guard subprocess failed "
        f"(exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )


if __name__ == "__main__":
    test_import_simulation_entities_is_light()
    print("import-light guard OK")
