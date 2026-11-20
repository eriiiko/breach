#!/usr/bin/env python3
"""PreToolUse hook: inject path-scoped rules for files outside the session's project root.

Claude Code loads `.claude/rules/*.md` from the session's project root and injects a
rule that has `paths:` only when a Read / Edit / Write touches a matching file under
that same root. A file in a git worktree, or in any other checkout, never matches, so
the tripwires are blind for every implementer that works in a worktree (perfusio #99,
probes of 2026-09-22).

This hook closes that gap. For the file a Read / Edit / Write / NotebookEdit targets
it walks up to the nearest `.claude/rules/` directory, matches the file against each
rule's `paths:` globs relative to that root, and returns the matching rules as
`additionalContext`. A root that contains the session's own project directory is left
to the native mechanism, so nothing is injected twice. Each rule is injected once per
session and per subagent, keyed on the rule's root and file name.

It never blocks a tool call: on any failure it exits 0 with nothing on stdout, and it
never sets `permissionDecision`, so the permission flow is untouched.
Install and verification: the user-level `path-rules-hook` skill. Set
PATH_RULES_DEBUG_LOG=<file> to append one JSON line per invocation (what was
emitted, for whom), since a hook that fails is silent by design.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

PATH_FIELDS = ("file_path", "path", "notebook_path")
STATE_FILE = "path_rules_seen.json"
CASE_INSENSITIVE = os.name == "nt"


def expand_braces(pattern: str) -> list[str]:
    """`a/{b,c}.d` -> [`a/b.d`, `a/c.d`], recursively."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out += expand_braces(pattern[: m.start()] + alt.strip() + pattern[m.end():])
    return out


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """picomatch-style: `**` spans directories, `*` and `?` stay inside one segment."""
    p = pattern.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    out = ""
    i = 0
    while i < len(p):
        if p.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif p.startswith("**", i):
            out += ".*"
            i += 2
        elif p[i] == "*":
            out += "[^/]*"
            i += 1
        elif p[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(p[i])
            i += 1
    return re.compile("^" + out + "$", re.IGNORECASE if CASE_INSENSITIVE else 0)


def parse_rule(text: str) -> tuple[list[str] | None, str]:
    """Return (paths, body). paths is None when the rule has no `paths:` key: such a
    rule loads natively at every session start and is not this hook's business."""
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", text, re.S)
    if not m:
        return None, text
    front, body = m.group(1), m.group(2)
    paths: list[str] | None = None
    lines = front.splitlines()
    for i, line in enumerate(lines):
        km = re.match(r"^paths:\s*(.*)$", line)
        if not km:
            continue
        rest = km.group(1).strip()
        if rest.startswith("["):
            paths = [s.strip().strip("'\"") for s in rest.strip("[]").split(",") if s.strip()]
        else:
            paths = []
            for follow in lines[i + 1:]:
                im = re.match(r"^\s*-\s*(.+?)\s*$", follow)
                if not im:
                    break
                paths.append(im.group(1).strip().strip("'\""))
        break
    return paths, body


def find_rules_root(file_path: str) -> str | None:
    """The nearest ancestor of the file that has a `.claude/rules/` directory."""
    d = os.path.dirname(os.path.abspath(file_path))
    while True:
        if os.path.isdir(os.path.join(d, ".claude", "rules")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def _norm(p: str) -> str:
    return os.path.normcase(os.path.realpath(p))


def within(child: str | None, ancestor: str | None) -> bool:
    """True when `child` is `ancestor` or lies under it (case-insensitive on Windows,
    so a `c:` / `C:` drive-letter mismatch never splits one tree into two)."""
    if not child or not ancestor:
        return False
    c, a = _norm(child), _norm(ancestor)
    return c == a or c.startswith(a.rstrip(os.sep) + os.sep)


def matching_rules(root: str, file_path: str) -> list[tuple[str, list[str], str]]:
    rel = os.path.relpath(os.path.abspath(file_path), root).replace(os.sep, "/")
    rules_dir = os.path.join(root, ".claude", "rules")
    hits = []
    for name in sorted(os.listdir(rules_dir)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(rules_dir, name), encoding="utf-8") as fh:
            paths, body = parse_rule(fh.read())
        if not paths:
            continue
        for pattern in paths:
            if any(glob_to_regex(x).match(rel) for x in expand_braces(pattern)):
                hits.append((name, paths, body.strip()))
                break
    hits.sort(key=lambda h: len(h[2]))  # short tripwires before long inventories
    return hits


def state_path(data: dict) -> str:
    base = data.get("scratchpad_dir") or os.path.join(
        tempfile.gettempdir(), "claude-path-rules", str(data.get("session_id", "nosession")))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, STATE_FILE)


def not_yet_seen(data: dict, keys: list[str]) -> set[str]:
    """The keys not injected before for this session + agent; records them as seen."""
    agent = str(data.get("agent_id") or "main")
    path = state_path(data)
    try:
        with open(path, encoding="utf-8") as fh:
            seen = json.load(fh)
    except (OSError, ValueError):
        seen = {}
    done = set(seen.get(agent, []))
    fresh = [k for k in keys if k not in done]
    if fresh:
        seen[agent] = sorted(done | set(fresh))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(seen, fh)
    return set(fresh)


def build_context(data: dict) -> tuple[str | None, list[str]]:
    """Return (context to inject or None, names of the rules it carries)."""
    tool_input = data.get("tool_input") or {}
    file_path = next((tool_input[k] for k in PATH_FIELDS if tool_input.get(k)), None)
    if not file_path:
        return None, []
    root = find_rules_root(file_path)
    if root is None:
        return None, []
    project = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd")
    if within(project, root):
        return None, []  # the native mechanism owns this root
    hits = matching_rules(root, file_path)
    if not hits:
        return None, []
    keys = [f"{_norm(root)}|{name}" for name, _, _ in hits]
    fresh = not_yet_seen(data, keys)
    parts = []
    for (name, paths, body), key in zip(hits, keys):
        if key not in fresh:
            continue
        rule_file = os.path.join(root, ".claude", "rules", name)
        parts.append(
            f"[path-rules hook] `{file_path}` lies outside this session's project root, so "
            f"Claude Code did not inject the rule that covers it. This is `{rule_file}` "
            f"(paths: {', '.join(paths)}):\n\n{body}")
    injected = [name for (name, _, _), key in zip(hits, keys) if key in fresh]
    return ("\n\n---\n\n".join(parts) if parts else None), injected


def main() -> int:
    try:
        data = json.load(sys.stdin)
        context, injected = build_context(data)
        if context:
            json.dump({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                              "additionalContext": context}}, sys.stdout)
        log = os.environ.get("PATH_RULES_DEBUG_LOG")
        if log:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"session": data.get("session_id"), "agent": data.get("agent_id"),
                                     "tool": data.get("tool_name"),
                                     "file": (data.get("tool_input") or {}).get("file_path"),
                                     "scratch": data.get("scratchpad_dir"), "injected": injected,
                                     "chars": len(context or "")}) + "\n")
    except Exception as exc:  # never block the tool call
        print(f"path_rules hook: {exc!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
