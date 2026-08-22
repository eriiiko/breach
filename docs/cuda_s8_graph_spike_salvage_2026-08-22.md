# CUDA-S8 graph-spike salvage notes (2026-08-22)

Context: a local, uncommitted "THROWAWAY" version of the S8 de-risk spike
(`cpp/src/cuda_spike.{cu,h}`, `tests/_spike_s8.py`) on the home desktop went
further than the version that was later committed upstream as the narrowed S8a
spike (`37753f8`, raw-CuPy-pointer +1 kernel). The extra code was discarded
(2026-08-22, Erik's call: low salvage effort); these are the hard-won facts it
proved on the real RTX 3070, kept so S8b doesn't re-derive them.

## Spike 2 — CUDA graph capture + replay (the S8b per-tick-graph premise)

All of this EXECUTED green on the 3070:

- **Graph replay is bit-identical to direct launches**, and stable over 100
  replays of the same instantiated graph (int32-wrap-correct reference).
- **Capture gotcha:** you must NOT capture on the default/null stream — the
  legacy default stream is non-capturable. Create a dedicated
  `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)` stream for capture.
- **Use `cudaStreamCaptureModeThreadLocal`** so an unrelated library op on
  another thread (e.g. CuPy housekeeping) can't invalidate the capture.
- **Instantiate once, replay cheaply:** the same `cudaGraphExec_t` is
  re-launched every iteration with no re-instantiation — that's the whole
  premise of the per-tick graph, and it held.

## Spike 1 extras (beyond the committed +1 kernel)

- **Non-contiguous CuPy slices:** element stride = `strides[0] // itemsize`;
  `arr.data.ptr` already points at the slice's first element (the view offset
  is folded into the pointer). A strided kernel walking `base[i*stride]`
  mutates exactly the right elements of the parent.
- **3D plane `gas[k]` of a C-contiguous (G,H,W) array is itself contiguous** —
  the plain contiguous kernel works on the real S8 field-plane shape, no
  strided path needed.
- **`.data.ptr` is stable** across reads and across unrelated new-array
  allocations while the array is alive and not reassigned.
