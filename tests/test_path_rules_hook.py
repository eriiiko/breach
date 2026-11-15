"""Property tests for the path-rules PreToolUse hook, `.claude/hooks/path_rules.py`.

The hook exists because Claude Code injects a path-scoped rule only for files under
the session's own project root, so implementers working in a worktree never see the
tripwires (#99, probes of 2026-09-22). Every test names the property it pins and the
change that must break it. The hook is exercised through its real entry point, as a
subprocess reading JSON on stdin, because that is the contract Claude Code holds it to.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "path_rules.py"
_spec = importlib.util.spec_from_file_location("path_rules", HOOK)
assert _spec is not None and _spec.loader is not None
path_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(path_rules)

WIRE_RULE = (
    '---\npaths:\n  - "cpp/src/ipc_protocol.h"\n  - "python/perfusio/client/**"\n---\n\n'
    "# TRIPWIRE — wire\nbody-wire\n"
)
PLAIN_RULE = "# a rule with no paths key\nbody-plain\n"


def make_checkout(root: Path, rules: dict) -> Path:
    (root / ".claude" / "rules").mkdir(parents=True)
    for name, text in rules.items():
        (root / ".claude" / "rules" / name).write_text(text, encoding="utf-8")
    return root


def payload(target: Path, session_root: Path, scratch: Path, agent_id=None, tool="Read"):
    data = {
        "session_id": "s1",
        "tool_name": tool,
        "tool_input": {"file_path": str(target)},
        "cwd": str(session_root),
        "scratchpad_dir": str(scratch),
    }
    if agent_id:
        data["agent_id"] = agent_id
    return data


def run_hook(data, project_dir, raw=None):
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(project_dir))
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=raw if raw is not None else json.dumps(data),
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout) if proc.stdout.strip() else None


@pytest.mark.parametrize("pattern,rel,expected", [
    ("cpp/src/**", "cpp/src/a/b.h", True),
    ("cpp/src/**", "cpp/src/x.h", True),
    ("cpp/src/*.h", "cpp/src/a/b.h", False),
    ("**/*.{ts,tsx}", "a/b/c.tsx", True),
    ("**/*.{ts,tsx}", "c.ts", True),
    ("python/perfusio/store/*csv_writer*.py", "python/perfusio/store/systemic_csv_writer.py", True),
    ("cpp/config/calibration.toml", "cpp/config/calibration.toml", True),
    ("cpp/config/calibration.toml", "cpp/config/hardware.toml", False),
])
def test_glob_semantics_match_claude_code_path_rules(pattern, rel, expected):
    """`**` spans directories, `*` and `?` stay within one path segment, and braces
    expand, as in Claude Code's own `paths:` matching. Breaks if the converter lets
    `*` cross a slash, or drops brace expansion."""
    hit = any(path_rules.glob_to_regex(p).match(rel) for p in path_rules.expand_braces(pattern))
    assert hit is expected


def test_file_in_another_checkout_gets_that_checkouts_rule(tmp_path):
    """A file outside the session's project root gets the matching rule of the
    checkout that contains it, as PreToolUse additionalContext, and no permission
    decision rides along. Breaks if the hook stops walking up from the file, matches
    against the session root instead, or starts setting permissionDecision."""
    session = make_checkout(tmp_path / "main", {"wire-protocol.md": WIRE_RULE})
    other = make_checkout(tmp_path / "wt", {"wire-protocol.md": WIRE_RULE})
    target = other / "cpp" / "src" / "ipc_protocol.h"
    out = run_hook(payload(target, session, tmp_path / "scratch"), session)
    assert out is not None
    assert out["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "body-wire" in ctx and "wire-protocol.md" in ctx
    assert "permissionDecision" not in out["hookSpecificOutput"]


def test_file_under_the_session_root_is_left_to_the_native_mechanism(tmp_path):
    """Under the session's own project root Claude Code injects the rule itself, so
    the hook stays silent there and nothing is injected twice. Breaks if the
    in-root check is removed or compares paths case-sensitively on Windows."""
    session = make_checkout(tmp_path / "main", {"wire-protocol.md": WIRE_RULE})
    target = session / "cpp" / "src" / "ipc_protocol.h"
    assert run_hook(payload(target, session, tmp_path / "scratch"), session) is None
    # a session whose cwd wandered into a subdirectory is still inside that root
    assert run_hook(payload(target, session / "cpp", tmp_path / "scratch"), session / "cpp") is None


def test_each_rule_is_injected_once_per_session_and_per_agent(tmp_path):
    """The same rule is not repeated on every Read of the same region within one
    session, but a subagent, which has its own context, gets its own copy. Breaks
    if the seen-state stops being keyed on (session scratch dir, agent_id)."""
    session = make_checkout(tmp_path / "main", {})
    other = make_checkout(tmp_path / "wt", {"wire-protocol.md": WIRE_RULE})
    target = other / "cpp" / "src" / "ipc_protocol.h"
    scratch = tmp_path / "scratch"
    assert run_hook(payload(target, session, scratch), session) is not None
    assert run_hook(payload(target, session, scratch), session) is None
    assert run_hook(payload(target, session, scratch, agent_id="sub-1"), session) is not None


def test_rules_without_paths_and_non_matching_files_inject_nothing(tmp_path):
    """A rule without `paths:` loads natively at every start and is never the
    hook's business; a file no pattern covers gets nothing. Breaks if the parser
    treats a missing key as match-everything."""
    session = make_checkout(tmp_path / "main", {})
    other = make_checkout(tmp_path / "wt", {"plain.md": PLAIN_RULE, "wire-protocol.md": WIRE_RULE})
    target = other / "docs" / "notes.md"
    assert run_hook(payload(target, session, tmp_path / "scratch"), session) is None


def test_never_blocks_a_tool_call(tmp_path):
    """Garbage on stdin, or no file path at all, exits 0 with empty stdout: the hook
    may fail to help, never to block. Breaks if an exception escapes main()."""
    session = make_checkout(tmp_path / "main", {})
    assert run_hook(None, session, raw="this is not json") is None
    assert run_hook({"tool_name": "Read", "tool_input": {}}, session) is None


def test_short_rules_are_injected_before_long_ones(tmp_path):
    """A short tripwire comes before a long inventory that matches the same file,
    so the imperative text is read first. The names are chosen so that alphabetical
    order would put the inventory first. Breaks if the size ordering in
    matching_rules is dropped."""
    session = make_checkout(tmp_path / "main", {})
    inventory = '---\npaths:\n  - "cpp/src/**"\n---\n\n# INVENTORY\n' + ("row " * 500) + "\n"
    other = make_checkout(tmp_path / "wt", {"a-inventory.md": inventory, "z-tripwire.md": WIRE_RULE})
    target = other / "cpp" / "src" / "ipc_protocol.h"
    out = run_hook(payload(target, session, tmp_path / "scratch"), session)
    assert out is not None
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert ctx.index("z-tripwire.md") < ctx.index("a-inventory.md")
