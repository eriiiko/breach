"""Generate the swarm kernels' checked-in table header (arc #63 P3,
docs/swarm_P3_impl.md v4.3).

``cpp/src/swarm_tables_layout.h`` -- the offsets of every table row a swarm
kernel reads, per species, and ``SWARM_TABLES_FINGERPRINT``, from
``simulation.swarm_tables.header_text``. The swarm has its OWN generator:
``tools/gen_unit_layout.py`` is the unit store's.

``tests/test_swarm_tables.py`` refuses a stale copy. A row added, removed or
moved means: run this with ``--write``, commit the header, rebuild both
modules (``swarm.engine_args`` refuses a compiled module whose fingerprint is
not the tables').

    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_swarm_layout.py          # check (exit 1 when stale)
    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_swarm_layout.py --write  # regenerate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--write", action="store_true",
                    help="rewrite the header (default: check only)")
    args = ap.parse_args(argv)
    from simulation import swarm_tables
    path = ROOT / swarm_tables.HEADER_REL
    text = swarm_tables.header_text()
    lay = swarm_tables.layout()
    old = path.read_text(encoding="utf-8") if path.is_file() else None
    if old == text:
        print(f"{swarm_tables.HEADER_REL} is current ({lay.fingerprint[:12]})")
        return 0
    if not args.write:
        print(f"STALE: {swarm_tables.HEADER_REL} differs from swarm_tables.ROWS; "
              f"run `{swarm_tables.GENERATOR}` and commit")
        return 1
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {swarm_tables.HEADER_REL} ({lay.fingerprint[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
