"""Generate the device unit store's checked-in row layout header (#29 P6.1).

Residency design §4.4b decision 1: a unit is a row of W int64 -- the unit
schema's group-S columns in schema order at their encoded widths, then
`valid` -- and the column offsets reach C++ as a GENERATED, checked-in header,
``cpp/src/unit_store_layout.h`` (the light table's and the edit ordinals'
precedent). The text comes from ``simulation.unit_store.header_text`` (the one
derivation); ``tests/test_unit_store.py`` refuses a stale copy.

A schema column added, removed or re-widened (a config row that changes a
block's width included -- a hand-grenade row, a status kind, a damage type, a
hook) means: run this with ``--write``, commit the header, rebuild
cpp/build_cuda (the device store refuses a compiled layout whose fingerprint
is not the schema's).

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
                    help="rewrite the header (default: check only)")
    args = ap.parse_args(argv)
    from simulation import unit_store
    path = ROOT / unit_store.HEADER_REL
    text = unit_store.header_text()
    old = path.read_text(encoding="utf-8") if path.is_file() else None
    if old == text:
        print(f"{unit_store.HEADER_REL} is current "
              f"(W {unit_store.layout().width})")
        return 0
    if not args.write:
        print(f"STALE: {unit_store.HEADER_REL} differs from the schema; run "
              f"`{unit_store.GENERATOR}` and commit")
        return 1
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {unit_store.HEADER_REL} (W {unit_store.layout().width}, "
          f"{unit_store.layout().fingerprint[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
