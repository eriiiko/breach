"""Property tests for the rules homes: CLAUDE.md's index and `.claude/rules/` (#206).

The master CLAUDE.md ("What a rule may hold", 2026-11-15) says a rule names files by
path and the repo carries a test that every path a rule names exists. A rule pointing
at a file that was renamed or deleted sends the next agent looking for a system that is
not there, which is how parallel copies get built. Every test names the property it
protects and the change that must break it. None pins a count or a set of regions or
systems: the index and the rule files are designed to grow.
"""
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLAUDE_MD = REPO / "CLAUDE.md"
RULES_DIR = REPO / ".claude" / "rules"

# Top-level folders and files a path in a rule may start from.
ROOTS = (
    "src/", "cpp/", "renderer/", "shaders/", "tools/", "tests/", "docs/", "levels/",
    "assets/", "audio/", "ui/", "art/", "containers/", ".claude/",
)
ROOT_FILES = {
    "main.py", "config.toml", "config.py", "level_lib.py", "level_loader.py",
    "pathfinding.py", "CLAUDE.md", "entities.toml",
}
# A bare or partial name is a path only when it ends in a code extension (data names such
# as level.toml or run.toml are formats, not repo files).
CODE_EXT = r"\.(py|cpp|h|cu|cuh|fs|vs|glsl|inc|cmake)$"
# Named on purpose although no tracked file holds them: build outputs.
UNTRACKED_OK = {
    "cpp/build/Release": "the CPU build's output directory (Environment)",
    "cpp/build_cuda/": "the CUDA build's output directory (Environment)",
    "cpp/build_cuda": "the CUDA build's output directory (Environment)",
    "assets/models/props/": "prop packs are untracked art, present on the machines that hold them",
}


@lru_cache(maxsize=1)
def tracked() -> tuple[str, ...]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return tuple(out.splitlines())


def rule_files() -> list[Path]:
    return sorted(RULES_DIR.glob("*.md"))


def expand_braces(pattern: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    out: list[str] = []
    for alt in m.group(1).split(","):
        out += expand_braces(pattern[: m.start()] + alt.strip() + pattern[m.end():])
    return out


def glob_regex(pattern: str) -> re.Pattern[str]:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile("^" + out + "$")


def resolves(path: str) -> bool:
    """A named path exists when a tracked file is it, lies under it, or matches its glob."""
    files = tracked()
    if path in UNTRACKED_OK:
        return True
    for alt in expand_braces(path):
        if any(c in alt for c in "*?"):
            rx = glob_regex(alt)
            if not any(rx.match(f) for f in files):
                return False
        elif alt.endswith("/"):
            if not any(f.startswith(alt) for f in files):
                return False
        elif alt not in files and not any(f.startswith(alt + "/") for f in files):
            return False
    return True


def named_paths(text: str) -> list[str]:
    """Every backticked token that names a file or folder of the repo."""
    found = []
    for span in re.findall(r"`([^`\n]+)`", text):
        words = span.split("::")[0].split()
        token = words[0].rstrip(",;:") if words else ""
        if not token or "<" in token or ">" in token or token.startswith(("/", "[")):
            continue
        if token.startswith(ROOTS) or token in ROOT_FILES:
            found.append(token)
        elif re.search(CODE_EXT, token):
            found.append(token)  # a bare or partial name: checked as a suffix
    return found


def exists_somewhere(token: str) -> bool:
    if token.startswith(ROOTS) or token in ROOT_FILES:
        return resolves(token)
    if any(c in token for c in "*?"):
        rx = glob_regex("**/" + token)
        return any(rx.match(f) for f in tracked())
    return any(f == token or f.endswith("/" + token) for f in tracked())


def index_row_names() -> set[str]:
    names = set()
    in_index = False
    for line in CLAUDE_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_index = line.startswith("## Canonical systems")
        if in_index and line.startswith("| ") and not line.startswith("| System"):
            names.add(line.split(" | ")[0][2:].strip())
    return names


def front_paths(text: str) -> list[str] | None:
    m = re.match(r"^---\r?\n(.*?)\r?\n---", text, re.S)
    if not m or not re.search(r"(?m)^paths:\s*$", m.group(1)):
        return None
    return re.findall(r'(?m)^\s*-\s*"?([^"\n]+?)"?\s*$', m.group(1))


DOCS = [CLAUDE_MD] + rule_files()


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_path_a_rule_names_exists(doc):
    """Protects: a rule never points at a file that is not there.
    Breaks when: a named file is renamed or deleted without amending the rule that names it."""
    missing = [t for t in named_paths(doc.read_text(encoding="utf-8")) if not exists_somewhere(t)]
    assert not missing, f"{doc.name} names paths that do not exist: {missing}"


@pytest.mark.parametrize("rule", rule_files(), ids=lambda p: p.name)
def test_every_rule_file_fires_on_files_that_exist(rule):
    """Protects: each rule file has a `paths:` list and every glob in it matches a tracked
    file, so the rule actually fires on contact with its region.
    Breaks when: a rule file loses its front matter, or its region's files move away."""
    paths = front_paths(rule.read_text(encoding="utf-8"))
    assert paths, f"{rule.name} has no paths: front matter"
    dead = [p for p in paths if not resolves(p)]
    assert not dead, f"{rule.name}: paths that match no tracked file: {dead}"


@pytest.mark.parametrize("rule", rule_files(), ids=lambda p: p.name)
def test_every_rule_section_belongs_to_an_index_row(rule):
    """Protects: a rule file's `## <name>` sections are constraints of systems the index
    lists by that exact name, so a reader of the index can find them.
    Breaks when: a section is added for a system with no index line, or a row is renamed
    in one place only."""
    sections = re.findall(r"(?m)^## (.+?)\s*$", rule.read_text(encoding="utf-8"))
    orphans = [s for s in sections if s not in index_row_names()]
    assert not orphans, f"{rule.name}: sections with no CLAUDE.md index row: {orphans}"


def test_the_region_table_and_the_rule_files_agree():
    """Protects: CLAUDE.md's region table (read by agents whose shell reads fire no rule)
    lists exactly the rule files that exist.
    Breaks when: a rule file is added, renamed or deleted without its table row."""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    table = text.split("## Rule regions", 1)[1].split("\n## ", 1)[0]
    listed = set(re.findall(r"(?m)^\| `([\w.-]+\.md)` \|", table))
    present = {p.name for p in rule_files()}
    assert listed == present, f"table only: {listed - present}; files only: {present - listed}"
