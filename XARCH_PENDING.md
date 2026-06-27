# X-ARCH pending — kernels proven on one GPU arch, awaiting the other

The CUDA port's real proof is **cross-GPU** bit-identity: the per-tick integer
field digest (`tests/field_digest.py`, spec v1) from the **Ampere RTX 3070
(sm_86, desktop)** and the **Ada Lenovo (sm_89)** must be byte-identical to each
other and to the CPU-int golden. Each kernel merges on a **two-beat** gate:

- **Beat A (at merge, blocks merge):** Ampere `digest(CUDA) == digest(CPU-int)`
  in-process, every tick, `tol=0.0` — the P1 gate (`tests/test_cuda_*`).
- **Beat B (batched, when the Lenovo is online):** re-run every merged kernel's
  scenario on Ada, diff `digest_<host>_cuda_gpu.txt` against the Ampere + CPU
  golden. Clears the kernel's row below.

Until a kernel's **Ada** row is green it is labelled
*"single-machine determinism proven; cross-GPU UNVERIFIED."* Turing (sm_75) is an
opportunistic third data point.

Capture a machine's digest with:
`C:/Users/steen/anaconda3/python.exe tests/xarch_digest.py --steps 30 --write`

| Kernel / step | Ampere (sm_86) | Ada (sm_89) | Turing (sm_75) | Notes |
|---|---|---|---|---|
| S0 toolchain + hello-world | ✅ bit-identical (mul_q16, 250k inputs × 9 factors) | ⬜ pending Lenovo | — | No physics kernel yet; proves the toolkit is device-clean + the plumbing. |

_(S1 temperature and later kernels append rows here as they merge Ampere-only.)_
