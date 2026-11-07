"""tools/cuda_green.py -- run the CUDA gates; on green, stamp HEAD (#29 P1b).

Residency design §6, Class P (critique 5 B1): a Class-P change merges only
with a green CUDA suite run in ITS OWN checkout. This tool is that run:

1. refuses a tracked change not committed (the stamp is about HEAD);
2. refuses without this checkout's CUDA build (``cpp/build_cuda``, the
   harness's ``cuda_available``): build it with ``cpp/build_cuda.bat``;
3. refuses when that module's ``build_id()`` is not HEAD (a build from an
   older commit proves nothing about this one): ``cmake --build
   cpp/build_cuda`` recompiles the one generated TU;
4. runs ``pytest tests -q -k cuda`` in this checkout (the gates run their
   CUDA work in ``data`` subprocesses against this checkout's
   ``cpp/build_cuda``, ``tests/cuda_harness.py``), and requires exit 0 with
   at least one test passed;
5. only then writes the untracked stamp ``.cuda_green``: HEAD's SHA on its
   first line, the pytest summary on its second. The stamp is deleted when
   the run starts, so a failed run leaves none.

``tests/test_class_p_level.py`` reads the stamp::

    C:/Users/steen/anaconda3/envs/data/python.exe tools/cuda_green.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import cuda_harness  # noqa: E402
import twin_debt  # noqa: E402

PYTHON = cuda_harness.CUDA_PYTHON


def refuse(msg: str) -> int:
    print(f"cuda_green: REFUSED -- {msg}")
    return 1


def module_build_id() -> str:
    """``breach_physics.build_id()`` of this checkout's CUDA build."""
    r = cuda_harness.run_cuda_script(
        "import breach_physics as bp\n"
        "print(getattr(bp, 'build_id', lambda: 'none')())\n", timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"importing the CUDA build failed:\n{r.stderr}")
    return r.stdout.strip().splitlines()[-1]


def main(argv=None) -> int:
    stamp = ROOT / twin_debt.STAMP
    stamp.unlink(missing_ok=True)
    head = twin_debt.git(ROOT, "rev-parse", "HEAD")
    dirty = twin_debt.git(ROOT, "status", "--porcelain",
                          "--untracked-files=no")
    if dirty:
        return refuse(f"tracked changes are not committed (the stamp is "
                      f"HEAD's):\n{dirty}")
    if not cuda_harness.cuda_available():
        return refuse(f"no CUDA build in this checkout "
                      f"({cuda_harness.CUDA_BUILD_DIR}): run "
                      f"cpp/build_cuda.bat")
    bid = module_build_id()
    if bid != head:
        return refuse(f"cpp/build_cuda was built from {bid!r}, HEAD is "
                      f"{head}: rebuild (cmake --build cpp/build_cuda)")
    print(f"cuda_green: running the CUDA gates at {head[:12]} ...",
          flush=True)
    proc = subprocess.run(
        [str(PYTHON), "-m", "pytest", "tests", "-q", "-k", "cuda",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    print(out[-4000:])
    summary = ""
    for line in reversed(out.splitlines()):
        if re.search(r"\d+ (passed|failed|error)", line):
            summary = line.strip("= ").strip()
            break
    passed = re.search(r"(\d+) passed", summary)
    if proc.returncode != 0:
        return refuse(f"the CUDA gates are red ({summary or 'exit '}"
                      f"{'' if summary else proc.returncode})")
    if not passed or int(passed.group(1)) == 0:
        return refuse(f"no CUDA gate passed ({summary or 'no summary'})")
    stamp.write_text(f"{head}\n{summary}\n", encoding="utf-8")
    print(f"cuda_green: wrote {stamp.name} = {head} ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
