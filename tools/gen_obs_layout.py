"""Generate the device observation's checked-in header (#29 P8.0).

``cpp/src/obs_layout.h`` -- every constant an observation kernel reads (the
window, the plane indices, the fills, the door codes, the float16 bit
patterns P8.0 writes) and ``OBS_LAYOUT_FINGERPRINT``, from
``simulation.device_obs.header_text``, each read from observation spec v1's
own tables. The observation has its OWN generator: ``tools/gen_unit_layout.py``
is the unit store's, ``tools/gen_swarm_layout.py`` the swarm's.

``tests/test_device_obs.py`` refuses a stale copy. A constant added, removed
or changed means: run this with ``--write``, commit the header, rebuild
cpp/build_cuda (``DeviceObservation`` refuses a compiled module whose
fingerprint is not the header's).

    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_obs_layout.py          # check (exit 1 when stale)
    C:/Users/steen/anaconda3/envs/data/python.exe tools/gen_obs_layout.py --write  # regenerate
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
    from simulation import device_obs
    path = ROOT / device_obs.HEADER_REL
    text = device_obs.header_text()
    old = path.read_text(encoding="utf-8") if path.is_file() else None
    if old == text:
        print(f"{device_obs.HEADER_REL} is current ({device_obs.fingerprint()[:12]})")
        return 0
    if not args.write:
        print(f"STALE: {device_obs.HEADER_REL} differs from device_obs; "
              f"run `{device_obs.GENERATOR}` and commit")
        return 1
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {device_obs.HEADER_REL} ({device_obs.fingerprint()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
