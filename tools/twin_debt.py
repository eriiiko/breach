"""tools/twin_debt.py -- THE twin debt ledger (#29 P1b, residency design §6).

Debt is COMPUTED, never remembered: a twin of ``tests/twin_manifest.toml`` is
in debt when any of its generated ``files`` changed in ``git diff --name-only
<level_at>..HEAD``, or when it is marked ``level = false`` (then its ``issue``
names the catch-up). Prints one line per twin in debt and exits 1 when there
is any, 0 when the ledger is empty::

    C:/Users/steen/anaconda3/envs/data/python.exe tools/twin_debt.py
    ... tools/twin_debt.py --mark-level eos            # level_at := HEAD
    ... tools/twin_debt.py --mark-level eos --at <rev>

``--mark-level`` rewrites that twin's ``level_at`` line and nothing else: a
patch that moves a twin's reference files and its device side together (every
Class-P patch, with the ``.cuda_green`` stamp as the proof) commits the
change, runs this, and commits the manifest, so ``level_at..HEAD`` then holds
only the manifest edit.

The same module answers the Class-P question for ``tests/test_class_p_level.py``
and ``tools/cuda_green.py``: :func:`class_p_changes` (the branch's changed
paths that are Class P: a Class-P twin's ``files`` or ``device``, or a
``[class_p] build`` pattern) and :func:`stamp_sha` (the untracked
``.cuda_green`` stamp). ``rl_env.require_snapshot(tag, "device")`` refuses
while :func:`debt` is non-empty.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_REL = "tests/twin_manifest.toml"
STAMP = ".cuda_green"
INTEGRATION = ("main", "origin/main")


class LedgerError(RuntimeError):
    """The manifest or the repo cannot answer the ledger's question."""


def git(repo, *args, check=True) -> str:
    """``git -C repo <args>``'s stdout, stripped; LedgerError on failure
    when ``check``."""
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True, timeout=60)
    if check and r.returncode != 0:
        raise LedgerError(f"git {' '.join(args)} failed in {repo}: "
                          f"{r.stderr.strip()}")
    return r.stdout.strip() if r.returncode == 0 else ""


def load(repo=ROOT, manifest=None) -> dict:
    """The parsed manifest of ``repo`` (or the file ``manifest``)."""
    path = Path(manifest) if manifest else Path(repo) / MANIFEST_REL
    if not path.is_file():
        raise LedgerError(f"no twin manifest at {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def changed_since(repo, rev) -> set:
    """Repo-relative paths changed in ``git diff --name-only rev..HEAD``."""
    out = git(repo, "diff", "--name-only", f"{rev}..HEAD")
    return {p for p in out.splitlines() if p}


def debt(repo=ROOT, manifest=None) -> list:
    """``[(twin, reason), ...]`` for every twin in debt, in manifest order.
    A ``level_at`` that does not name a commit is a LedgerError."""
    m = load(repo, manifest)
    out = []
    for name, twin in m.get("twins", {}).items():
        reasons = []
        if not twin.get("level", True):
            reasons.append(f"not level ({twin.get('issue') or 'no issue'})")
        at = twin.get("level_at", "")
        if not git(repo, "rev-parse", "--verify", "--quiet",
                   f"{at}^{{commit}}", check=False):
            raise LedgerError(f"twin {name!r}: level_at {at!r} is not a "
                              f"commit in {repo}")
        moved = sorted(changed_since(repo, at) & set(twin.get("files", ())))
        if moved:
            reasons.append(f"changed since {at[:12]}: {', '.join(moved)}")
        if reasons:
            out.append((name, "; ".join(reasons)))
    return out


# ---------------------------------------------------------------------------
# Class P
# ---------------------------------------------------------------------------
def class_p_paths(m: dict) -> tuple:
    """``(exact paths, glob patterns)`` that make a change Class P."""
    exact = set()
    for twin in m.get("twins", {}).values():
        if twin.get("class") == "P":
            exact |= set(twin.get("files", ()))
            exact |= set(twin.get("device", ()))
    return exact, tuple(m.get("class_p", {}).get("build", ()))


def is_class_p(path: str, m: dict) -> bool:
    exact, globs = class_p_paths(m)
    return path in exact or any(fnmatch.fnmatchcase(path, g) for g in globs)


def merge_base(repo) -> str:
    """``git merge-base HEAD main`` (``origin/main`` when there is no local
    main); LedgerError when neither exists."""
    for ref in INTEGRATION:
        if git(repo, "rev-parse", "--verify", "--quiet", ref, check=False):
            return git(repo, "merge-base", "HEAD", ref)
    raise LedgerError(f"no integration line ({' / '.join(INTEGRATION)}) in "
                      f"{repo}")


def class_p_changes(repo=ROOT, manifest=None) -> dict:
    """The branch's Class-P changes: ``{"committed": [...], "uncommitted":
    [...]}`` -- paths changed from the merge-base with main to HEAD, and
    tracked paths that differ from HEAD in the working tree or the index.
    On main itself the committed list is empty."""
    m = load(repo, manifest)
    base = merge_base(repo)
    committed = changed_since(repo, base)
    uncommitted = {p for p in git(repo, "diff", "--name-only", "HEAD")
                   .splitlines() if p}
    return {"committed": sorted(p for p in committed if is_class_p(p, m)),
            "uncommitted": sorted(p for p in uncommitted if is_class_p(p, m))}


CLASS_P_MESSAGE = ("a Class-P path changed: build cpp/build_cuda and run "
                   "`tools/cuda_green.py` before merging")


def class_p_problems(repo=ROOT, manifest=None) -> list:
    """Why this branch may not merge yet, or ``[]``: when it changed a
    Class-P path (:func:`class_p_changes`), every such change must be
    committed and the ``.cuda_green`` stamp must hold HEAD's SHA."""
    ch = class_p_changes(repo, manifest)
    if not ch["committed"] and not ch["uncommitted"]:
        return []
    out = []
    if ch["uncommitted"]:
        out.append("uncommitted Class-P change(s): "
                   + ", ".join(ch["uncommitted"]))
    head = git(repo, "rev-parse", "HEAD")
    st = stamp_sha(repo)
    if st != head:
        out.append(f"{STAMP} holds {st!r}, HEAD is {head}; Class-P "
                   f"path(s) changed on this branch: "
                   + ", ".join(ch["committed"] or ch["uncommitted"]))
    return out


def stamp_sha(repo=ROOT):
    """The SHA the ``.cuda_green`` stamp holds (its first line), or None."""
    p = Path(repo) / STAMP
    if not p.is_file():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else None


# ---------------------------------------------------------------------------
# --mark-level
# ---------------------------------------------------------------------------
def mark_level(path, twin: str, sha: str) -> None:
    """Rewrite ``level_at`` of ``[twins.<twin>]`` in the manifest at
    ``path`` to ``sha``; every other line is kept."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    sect = re.compile(r"^\[twins\." + re.escape(twin) + r"\]\s*$", re.M)
    hit = sect.search(text)
    if not hit:
        raise LedgerError(f"no [twins.{twin}] in {path}")
    nxt = re.compile(r"^\[", re.M).search(text, hit.end())
    end = nxt.start() if nxt else len(text)
    body = text[hit.end():end]
    new, n = re.subn(r'^level_at\s*=\s*"[^"]*"', f'level_at = "{sha}"',
                     body, count=1, flags=re.M)
    if n != 1:
        raise LedgerError(f"[twins.{twin}] in {path} has no level_at line")
    path.write_text(text[:hit.end()] + new + text[end:], encoding="utf-8",
                    newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--manifest", default=None,
                    help="the manifest file (default <repo>/" + MANIFEST_REL
                         + ")")
    ap.add_argument("--mark-level", metavar="TWIN", action="append",
                    help="set TWIN's level_at to --at (repeatable)")
    ap.add_argument("--at", default="HEAD",
                    help="the revision --mark-level records (default HEAD)")
    args = ap.parse_args(argv)
    repo = Path(args.repo)
    if args.mark_level:
        sha = git(repo, "rev-parse", "--verify", f"{args.at}^{{commit}}")
        path = Path(args.manifest) if args.manifest else repo / MANIFEST_REL
        known = load(repo, args.manifest).get("twins", {})
        for twin in args.mark_level:
            if twin not in known:
                print(f"no twin {twin!r} (known: {sorted(known)})")
                return 2
            mark_level(path, twin, sha)
            print(f"{twin}: level_at = {sha}")
        return 0
    rows = debt(repo, args.manifest)
    for name, reason in rows:
        print(f"DEBT {name}: {reason}")
    if not rows:
        print("twin debt: none")
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main())
