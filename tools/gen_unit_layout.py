"""Generate the device unit store's checked-in headers (#29 P6.1, P6.2, P6.3).

Three generated, checked-in headers (the light table's and the edit ordinals'
precedent), one command:

- ``cpp/src/unit_store_layout.h`` -- the store's ROW layout (residency design
  §4.4b decision 1: a unit is a row of W int64, the unit schema's group-S
  columns in schema order at their encoded widths, then `valid`), from
  ``simulation.unit_store.header_text``;
- ``cpp/src/unit_tables_layout.h`` -- the unit kernels' DEVICE TABLES (§4.11:
  every table value a kernel reads, one row of ``simulation.unit_tables.ROWS``
  each; plus the hooks' and the status kinds' indices), from
  ``simulation.unit_tables.header_text``;
- ``cpp/src/unit_events_layout.h`` -- the EVENT RING's record layout and the
  vocabulary its kernels name (#29 P6.3, §4.5: the event kinds, the causes,
  the damage types, the heat credit channel), from
  ``simulation.unit_events.header_text``;
- ``cpp/src/unit_requests_layout.h`` -- the REQUEST LIST's record layout (#29
  P6.4a: what a kernel does that is not a row write -- a grenade thrown, a
  charge planted, a door flipped), from ``simulation.unit_requests.header_text``.
- ``cpp/src/applier_layout.h`` -- the PER-ENV SERIAL APPLIER's effect record,
  kinds, book channels and the topology seam's / tick-end clears' planes
  (#29 P6.4b, §4.3), from ``simulation.applier.header_text``.
- ``cpp/src/unit_draws_layout.h`` -- the DEVICE DRAWS' vocabulary (#29 P7.0a,
  4.4c decision 5: the salts, the key row, the one site per ``.next`` salt),
  from ``simulation.unit_draws.header_text``.

``tests/test_unit_store.py`` / ``tests/test_unit_tables.py`` /
``tests/test_unit_events.py`` refuse a stale copy. A schema column or a
table row added, removed or re-widened (a config
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
    from simulation import unit_events, unit_requests, unit_store, unit_tables
    from simulation import applier          # #29 P6.4b
    from simulation import unit_draws       # #29 P7.0a: the salts, the .next sites
    stale = 0
    for mod in (unit_store, unit_tables, unit_events, unit_requests, applier,
                unit_draws):
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
