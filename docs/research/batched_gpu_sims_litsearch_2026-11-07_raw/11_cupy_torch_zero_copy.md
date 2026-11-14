# Q11. Zero-copy CuPy -> PyTorch: DLPack / CUDA Array Interface, streams, memory pools

Raw research note, 2026-11-07, for #29. Briefer pass.

## Sources
- CuPy interoperability guide: https://docs.cupy.dev/en/stable/user_guide/interoperability.html
- CUDA Array Interface spec (v3, hosted by Numba):
  https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
- DLPack Python specification: https://dmlc.github.io/dlpack/latest/python_spec.html
- PyTorch CUDA semantics: https://docs.pytorch.org/docs/2.15/notes/cuda.html
- Madrona's export path (nanobind + DLPack, note 01); WarpDrive's in-place store (note 02).

## Findings
- Zero copy: CuPy imports any DLPack producer with `cupy.from_dlpack()`, PyTorch imports with
  `torch.from_dlpack()`; the CuPy guide says exchange via `__cuda_array_interface__` with PyTorch
  is "at no cost" (the tensor must already be on the GPU). Lifetime is the user's job: "it is an
  undefined behavior if the exported object is destroyed while still in use by the consumer
  library".
- DLPack streams: "the consumer must pass the stream it will use to the producer"
  (`__dlpack__(stream=...)`), and the producer synchronises or waits on it when necessary; no
  synchronisation is needed when both use the default stream. Ownership: the consumer renames the
  capsule to `used_dltensor` and calls the deleter when it frees the array. Arrays imported via
  DLPack are "safely accessible on CuPy's current stream" (CuPy guide).
- CUDA Array Interface v3 `stream` field: an integer names the stream on which the producer may
  still have work; the consumer must synchronise on it or enqueue its work on it. `None` = no
  synchronisation needed; `1` = legacy default stream; `2` = per-thread default; `0` is
  disallowed (ambiguous). Consumers may disable their automatic sync (Numba:
  `NUMBA_CUDA_ARRAY_INTERFACE_SYNC=0`; CuPy names `CUPY_CUDA_ARRAY_INTERFACE_SYNC` without
  documenting it on that page).
- Shared streams: the CuPy guide's recommended route is `cupy.cuda.Stream.from_external()`
  (holds a reference to the foreign stream); the older `cupy.cuda.ExternalStream(s.cuda_stream)`
  pattern is marked "deprecated since v14.0". Breach's `data` env runs CuPy 14.2.0 (benchmark),
  so the new API applies.
- Memory pools: CuPy and PyTorch each keep a caching allocator. pytorch-pfn-extras offers
  "memory pool sharing and stream sharing" (`ppe.cuda.use_torch_mempool_in_cupy()`), per the
  CuPy guide. PyTorch's caching allocator can hand freed memory to a new tensor while another
  stream still uses it: "it is still necessary to synchronize, because A may correspond to memory
  reallocated by the CUDA caching allocator"; `record_stream` "ensures that we do not deallocate
  A before sum(A) has completed".

## Pitfalls for Breach's step -> obs -> policy -> action loop
1. Stream order: if the sim's kernels run on a non-default stream (the CUDA-graph spike needed a
   non-blocking stream for capture, where-we-stand §4) and the policy runs on torch's current
   stream, the obs tensor needs an event wait (or one shared stream for both). DLPack's stream
   argument does this when the producer honours it; a raw CAI pointer without the `stream`
   field (or with `None`) gives NO protection.
2. Lifetime: the sim's obs buffer must outlive every torch view of it; a persistent,
   preallocated buffer (WarpDrive's and Madrona's design) removes the question.
3. Reuse races: the action tensor written by torch and read by the sim, or a torch-side
   temporary, can be recycled by torch's allocator while a sim kernel on another stream still
   reads it; use one stream or `record_stream`.
4. Two pools on an 8 GB card: CuPy's pool and torch's both hold freed memory; sharing the pool
   (pytorch-pfn-extras) or capping CuPy's (`MemoryPool.set_limit`; an API I know of but did not
   re-read in this pass: unverified here) avoids starving one side.
