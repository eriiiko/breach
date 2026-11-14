"""Throwaway git repositories for the snapshot / ledger door tests (#29 P1b).

A door that refuses on repo state is tested by CONSTRUCTING the state in a
temp repo (``git init`` under pytest's ``tmp_path``), never by mutating the
real checkout. Harness only (underscore: not collected).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_ID = ["-c", "user.name=breach-test", "-c", "user.email=test@breach.invalid"]


def git(repo, *args) -> str:
    """``git -C repo <args>``'s stdout (stripped); raises on failure."""
    r = subprocess.run(["git", "-C", str(repo), *_ID, *args],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def write(repo, files: dict) -> None:
    """Write ``{relpath: text}`` under ``repo``."""
    for rel, text in files.items():
        p = Path(repo) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def commit(repo, files: dict | None = None, msg: str = "c") -> str:
    """Write ``files``, stage everything, commit; returns the new HEAD."""
    if files:
        write(repo, files)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def make_repo(root, files: dict) -> Path:
    """A fresh repo at ``root`` on branch ``main`` with ``files`` committed."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "core.autocrlf", "false")
    commit(root, files, "base")
    return root
