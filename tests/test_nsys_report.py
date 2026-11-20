"""tools/nsys_report.py on hand-made ``nsys stats`` CSVs (#29 P2).

Properties: (1) the tick count is the number of distinct ``tick N`` NVTX
ranges (a default-domain ``:tick N`` included), and every per-tick figure is
the window total divided by it; (2) copies are split by direction from the
Operation column, bytes from the size table, a ``_vNNNN`` API suffix is
stripped before calls are summed; (3) a banner line before a table and a
missing table do not break the report. Breaks if the reader keys on a column
position instead of its name, double-counts a direction, or counts a stage
range as a tick.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import nsys_report  # noqa: E402

KERN = '''Processing [x.sqlite] with [cuda_gpu_kern_sum.py]...
"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
75.0,3000000,40,75000,75000,70000,80000,1000,"combustion_pass_a<2>(int, ...)"
25.0,1000000,120,8333,8000,7000,9000,100,"eos_kick(...)"
'''
MEM_T = '''"Time (%)","Total Time (ns)","Count","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Operation"
60.0,600000,20,30000,30000,1,1,0,"[CUDA memcpy Host-to-Device]"
30.0,300000,10,30000,30000,1,1,0,"[CUDA memcpy Device-to-Host]"
10.0,100000,40,2500,2500,1,1,0,"[CUDA memcpy Device-to-Device]"
'''
MEM_S = '''"Total (MB)","Count","Avg (MB)","Med (MB)","Min (MB)","Max (MB)","StdDev (MB)","Operation"
2.000,20,0.1,0.1,0.1,0.1,0,"[CUDA memcpy Host-to-Device]"
1.000,10,0.1,0.1,0.1,0.1,0,"[CUDA memcpy Device-to-Host]"
0.400,40,0.01,0.01,0.01,0.01,0,"[CUDA memcpy Device-to-Device]"
'''
API = '''"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
40.0,4000,160,25,25,1,1,0,"cudaLaunchKernel_v7000"
20.0,2000,30,66,66,1,1,0,"cudaMemcpy_v3020"
10.0,1000,8,125,125,1,1,0,"cudaMalloc_v3020"
10.0,1000,8,125,125,1,1,0,"cudaFree_v3020"
20.0,2000,6,333,333,1,1,0,"cudaDeviceSynchronize_v3020"
'''
NVTX = '''"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Style","Range"
30.0,9000000,1,9000000,0,0,0,0,"PushPop",":tick 3"
30.0,9000000,1,9000000,0,0,0,0,"PushPop",":tick 4"
40.0,4000000,2,2000000,0,0,0,0,"PushPop",":7 physics"
'''


def _write(tmp_path, tables):
    base = tmp_path / "cap_stats"
    for name, text in tables.items():
        (tmp_path / f"cap_stats_{name}.csv").write_text(text, encoding="utf-8")
    return base


def test_per_tick_figures_are_window_totals_over_the_tick_ranges(tmp_path):
    base = _write(tmp_path, {"cuda_gpu_kern_sum": KERN, "cuda_gpu_mem_time_sum": MEM_T,
                             "cuda_gpu_mem_size_sum": MEM_S, "cuda_api_sum": API,
                             "nvtx_sum": NVTX})
    s = nsys_report.summarize(base)
    assert s["ticks"] == 2                         # the two tick ranges, not the stage
    assert s["copies"]["H2D"]["count"] == 20 and s["copies"]["D2H"]["count"] == 10
    assert s["copies"]["H2D"]["mb"] == 2.0 and s["copies"]["D2D"]["mb"] == 0.4
    assert s["api"]["cudaLaunchKernel"][0] == 160  # the _v7000 suffix stripped
    assert s["kernels"][0][0].startswith("combustion_pass_a")   # by GPU time
    text = nsys_report.report(s)
    assert "| kernel launches (GPU side) | 80.0 | 160 |" in text
    assert "| H2D copies | 10.0 x 1000.0 KB" in text
    assert "| cudaMalloc | 4.0" in text and "| explicit device syncs | 3.0" in text
    assert "7 physics" in text and ":tick" not in text


def test_a_missing_table_and_an_explicit_tick_count(tmp_path):
    base = _write(tmp_path, {"cuda_gpu_kern_sum": KERN})
    s = nsys_report.summarize(base, ticks=10)
    assert s["ticks"] == 10
    assert s["copies"]["H2D"]["count"] == 0
    assert "| kernel launches (GPU side) | 16.0 | 160 |" in nsys_report.report(s)
