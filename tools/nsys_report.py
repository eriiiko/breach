"""tools/nsys_report.py -- the resident tick under Nsight, per captured tick (#29 P2).

Reads the CSV tables ``nsys stats --format csv --output <base>`` writes for an
Nsight Systems capture of ``tools/bench_bot_matches.py ... --nvtx`` (the
capture Erik runs elevated with ``C:\\tmp\\breach_nsys_capture.ps1``) and prints,
PER CAPTURED TICK (the window's totals divided by its tick count):

* kernel launches, and the top kernels by GPU time;
* memory copies: count and bytes by direction (H2D, D2H, D2D, memset);
* ``cudaMalloc`` / ``cudaFree`` calls;
* explicit device syncs (``cudaDeviceSynchronize``, ``cudaStreamSynchronize``,
  ``cudaEventSynchronize``) and the blocking-copy API calls.

THE TICK COUNT is the number of distinct ``tick N`` NVTX ranges in
``nvtx_sum`` (``tools/tick_stage_profile.py`` pushes one per tick inside the
``cudaProfilerStart`` / ``Stop`` window); ``--ticks`` overrides it.

THE INPUT -- the five reports, one file each, named ``<base>_<report>.csv``
(nsys 2023.4's naming for ``--output <base>``). The columns this reads (any
others are ignored; a missing file is reported and its rows skipped):

``cuda_gpu_kern_sum``      ``Time (%)``, ``Total Time (ns)``, ``Instances``,
                           ``Avg (ns)``, ..., ``Name``
``cuda_gpu_mem_time_sum``  ``Total Time (ns)``, ``Count``, ..., ``Operation``
                           (``[CUDA memcpy Host-to-Device]``,
                           ``[CUDA memcpy Device-to-Host]``,
                           ``[CUDA memcpy Device-to-Device]``, ``[CUDA memset]``)
``cuda_gpu_mem_size_sum``  ``Total (MB)``, ``Count``, ..., ``Operation``
``cuda_api_sum``           ``Total Time (ns)``, ``Num Calls``, ..., ``Name``
                           (an API name may carry a ``_vNNNN`` version suffix,
                           e.g. ``cudaLaunchKernel_v7000``; it is stripped)
``nvtx_sum``               ``Total Time (ns)``, ``Instances``, ..., ``Range``
                           (a default-domain range may print as ``:tick 7``)

A table may be preceded by non-CSV lines (a banner); the reader skips to the
first line holding the table's key column.

Usage::

    C:/Users/steen/anaconda3/envs/data/python.exe tools/nsys_report.py \\
        C:/tmp/breach_resident_medium_stats [--ticks 100] [--top 15]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPORTS = ("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum",
           "cuda_api_sum", "nvtx_sum")
# The column that marks each table's header line.
_KEY = {"cuda_gpu_kern_sum": "Name", "cuda_gpu_mem_time_sum": "Operation",
        "cuda_gpu_mem_size_sum": "Operation", "cuda_api_sum": "Name",
        "nvtx_sum": "Range"}
_DIRECTIONS = (("H2D", "Host-to-Device"), ("D2H", "Device-to-Host"),
               ("D2D", "Device-to-Device"), ("memset", "memset"))
_SYNCS = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize")
_COPIES = ("cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpyToSymbol",
           "cudaMemcpy2D", "cudaMemcpy2DAsync", "cudaMemset", "cudaMemsetAsync")
_LAUNCHES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchCooperativeKernel")
_TICK = re.compile(r"^:?tick \d+$")


def read_table(path: Path, report: str) -> list:
    """The rows of one ``nsys stats`` CSV as dicts; [] when the file is
    missing (reported on stderr)."""
    if not path.is_file():
        print(f"nsys_report: missing {path.name} -- its rows are skipped",
              file=sys.stderr)
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    key = _KEY[report]
    for k, line in enumerate(lines):
        if key in next(csv.reader([line]), []):
            return list(csv.DictReader(lines[k:]))
    print(f"nsys_report: {path.name} holds no '{key}' header", file=sys.stderr)
    return []


def _num(row: dict, col: str) -> float:
    v = (row.get(col) or "0").replace(",", "").strip()
    try:
        return float(v)
    except ValueError:
        return 0.0


def api_name(name: str) -> str:
    """``cudaLaunchKernel_v7000`` -> ``cudaLaunchKernel``."""
    return re.sub(r"_v\d+$", "", name.strip())


def summarize(base, ticks=None) -> dict:
    """Totals over the capture window and the tick count, from the five
    tables at ``<base>_<report>.csv``."""
    base = Path(base)
    t = {r: read_table(base.with_name(f"{base.name}_{r}.csv"), r) for r in REPORTS}
    n_ticks = ticks
    if n_ticks is None:
        n_ticks = sum(1 for r in t["nvtx_sum"] if _TICK.match(r.get("Range", "").strip()))
    kernels = sorted(((r["Name"], _num(r, "Total Time (ns)"), _num(r, "Instances"))
                      for r in t["cuda_gpu_kern_sum"]), key=lambda k: -k[1])
    copies = {d: {"count": 0.0, "mb": 0.0, "ns": 0.0} for d, _ in _DIRECTIONS}
    for r in t["cuda_gpu_mem_time_sum"]:
        for d, needle in _DIRECTIONS:
            if needle in r.get("Operation", ""):
                copies[d]["count"] += _num(r, "Count")
                copies[d]["ns"] += _num(r, "Total Time (ns)")
    for r in t["cuda_gpu_mem_size_sum"]:
        for d, needle in _DIRECTIONS:
            if needle in r.get("Operation", ""):
                copies[d]["mb"] += _num(r, "Total (MB)")
    api = {}
    for r in t["cuda_api_sum"]:
        n = api_name(r.get("Name", ""))
        calls, ns = api.get(n, (0.0, 0.0))
        api[n] = (calls + _num(r, "Num Calls"), ns + _num(r, "Total Time (ns)"))
    stages = sorted(((r["Range"].lstrip(":"), _num(r, "Total Time (ns)"),
                      _num(r, "Instances"))
                     for r in t["nvtx_sum"]
                     if not _TICK.match(r.get("Range", "").strip())),
                    key=lambda s: -s[1])
    return {"ticks": n_ticks, "kernels": kernels, "copies": copies, "api": api,
            "stages": stages}


def _api(s: dict, names) -> tuple:
    calls = sum(s["api"].get(n, (0.0, 0.0))[0] for n in names)
    ns = sum(s["api"].get(n, (0.0, 0.0))[1] for n in names)
    return calls, ns


def report(s: dict, top: int = 15) -> str:
    """The per-tick table, as Markdown (pasted into the benchmark doc)."""
    n = s["ticks"]
    if not n:
        return "no captured ticks (no 'tick N' NVTX range and no --ticks)"
    out = [f"Per captured tick (window totals / {n} ticks).", "",
           "| quantity | per tick | window total |", "|---|---|---|"]
    k_launch = sum(k[2] for k in s["kernels"])
    k_ns = sum(k[1] for k in s["kernels"])
    out.append(f"| kernel launches (GPU side) | {k_launch / n:.1f} | {k_launch:.0f} |")
    out.append(f"| kernel time | {k_ns / n / 1e3:.1f} us | {k_ns / 1e6:.2f} ms |")
    calls, _ = _api(s, _LAUNCHES)
    out.append(f"| launch API calls | {calls / n:.1f} | {calls:.0f} |")
    for d, _ in _DIRECTIONS:
        c = s["copies"][d]
        out.append(f"| {d} copies | {c['count'] / n:.1f} x {c['mb'] * 1e3 / n:.1f} KB "
                   f"({c['ns'] / n / 1e3:.1f} us) | {c['count']:.0f} x {c['mb']:.2f} MB |")
    for label, names in (("cudaMalloc", ("cudaMalloc", "cudaMallocAsync")),
                         ("cudaFree", ("cudaFree", "cudaFreeAsync")),
                         ("explicit device syncs", _SYNCS),
                         ("copy / memset API calls", _COPIES)):
        calls, ns = _api(s, names)
        out.append(f"| {label} | {calls / n:.1f} ({ns / n / 1e3:.1f} us) | {calls:.0f} |")
    out += ["", f"Top {top} kernels by GPU time:", "",
            "| kernel | per tick (us) | launches per tick | share |", "|---|---|---|---|"]
    for name, ns, inst in s["kernels"][:top]:
        share = 100.0 * ns / k_ns if k_ns else 0.0
        out.append(f"| `{name[:80]}` | {ns / n / 1e3:.1f} | {inst / n:.1f} | {share:.1f} % |")
    if s["stages"]:
        out += ["", "NVTX stage ranges (host wall time):", "",
                "| range | per tick (us) | instances |", "|---|---|---|"]
        for name, ns, inst in s["stages"][:top]:
            out.append(f"| {name} | {ns / n / 1e3:.1f} | {inst:.0f} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nsys_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="the --output base of `nsys stats` (no _<report>.csv)")
    ap.add_argument("--ticks", type=int, default=None,
                    help="captured ticks (default: the 'tick N' NVTX ranges)")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    print(report(summarize(a.base, a.ticks), a.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
