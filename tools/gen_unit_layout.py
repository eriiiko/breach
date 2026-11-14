"""Generate the device unit store's checked-in headers (#29 P6.1, P6.2).

Two generated, checked-in headers (the light table's and the edit ordinals'
precedent), one command:

- ``cpp/src/unit_store_layout.h`` -- the store's ROW layout (residency design
  §4.4b decision 1: a unit is a row of W int64, the unit schema's group-S
  columns in schema order at their encoded widths, then `valid`), from
  ``simulation.unit_store.header_text``;
- ``cpp/src/unit_tables_layout.h`` -- the unit kernels' DEVICE TABLES (§4.11:
  every table value a kernel reads, one row of ``simulation.unit_tables.ROWS``
  each; plus the hooks' and the status kinds' indices), from
  ``simulation.unit_tables.header_text``.

``tests/test_unit_store.py`` / ``tests/test_unit_tables.py`` refuse a stale
copy. A schema column or a table row added, removed or re-widened (a config
row that changes a block's or a table's width included -- a hand-grenade row,
a status kind, a damage type, a hook, an env profile) means: run this with
``--write``, commit the headers, rebuild cpp/build_cuda (the device store
refuses a compiled layout whose fingerprint is not the schema's).

    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_unit_layout.py          # check (exit 1 when stale)
    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_unit_layout.py --write  # regenerate
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
                    help="rewrite the headers (default: check only)")
    args = ap.parse_args(argv)
    from simulation import unit_store, unit_tables
    stale = 0
    for mod in (unit_store, unit_tables):
        path = ROOT / mod.HEADER_REL
        text = mod.header_text()
        lay = mod.layout()
        old = path.read_text(encoding="utf-8") if path.is_file() else None
        if old == text:
            print(f"{mod.HEADER_REL} is current (W {lay.width})")
            continue
        if not args.write:
            print(f"STALE: {mod.HEADER_REL} differs from its schema; run "
                  f"`{mod.GENERATOR}` and commit")
            stale += 1
            continue
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {mod.HEADER_REL} (W {lay.width}, "
              f"{lay.fingerprint[:12]})")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
