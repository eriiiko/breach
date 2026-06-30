# Lenovo — what to do today (2026-06-29)

Two things: (A) get the Lenovo set up as the breach dev machine for vacation, and
(B) — optional — a 1-command confirm of the x-arch finding. The actual *localizing*
of the cross-machine divergence I'm doing on the work-desktop (it has MSVC 14.44, the
same compiler as your Lenovo, so I can reproduce + pin it there without you). So this
is mostly the move; the determinism step is a quick optional confirm.

---

## A. Vacation dev-setup — make the Lenovo the breach machine

1. **Pull latest** (`git pull` on `main`). HEAD should be at/after the run-on-CUDA +
   raycaster-live + x-arch commits.

2. **Copy the breach MEMORY** (this is the thing that doesn't live in the repo). On the
   work-desktop it's at:
   `C:\Users\steen\.claude\projects\c--Users-steen-projects-breach\memory\`
   Copy the **whole `memory\` folder wholesale** to the SAME path on the Lenovo
   (the project-hash folder name is identical as long as the repo is at
   `C:\Users\steen\projects\breach` on both — same username → same hash). No distilling
   needed; it's already a curated index (`MEMORY.md`) + one-fact files. (Claude on the
   desktop can zip it to your Google-Drive `ClaudeSync` folder for an easy transfer —
   just ask. A permanent fix — symlinking the memory dir through Google Drive like
   `CLAUDE.md` — is a nice follow-up if you want cross-machine memory to auto-sync.)

3. **Confirm the builds run** (you already proved these during the attestation):
   - CPU: `python main.py` (the default, unchanged).
   - CUDA: build via your `cpp/build_cuda_lenovo.bat`, then
     `<lenovo-python> main.py --cuda` (+ `--res 2/3/4…` to crank resolution). All 7
     solvers run on the Ada GPU; bit-identical to CPU on this box (you verified it).

That's the move done — you can develop breach on the Lenovo from here.

---

## B. (Optional) confirm the x-arch finding in one command

`git pull` brought a small tool + the Ampere baseline. Run:
```
<lenovo-python> tests/_xarch_perfield_digest.py
```
It dumps a per-(field, tick) digest for the default scenario AND — because the Ampere
baseline (`tests/_xarch_perfield_ampere.txt`) is now in the tree — **auto-prints the
FIRST diverging (field, tick)** between your Lenovo and the Ampere golden. Send me that
one line. It confirms my desktop reproduction matches your real Lenovo and names the
exact solver. (If I've already localized it on the desktop by the time you read this,
this just double-confirms — still worth the 1 command.)

> Not needed: a separate clean-MSVC rebuild on the Lenovo. I'm doing the
> compiler-isolation on the desktop (rebuild with MSVC 14.44 = your compiler), which
> pins compiler-vs-Python without tying up your machine.

---

## C. Note on the fix (for later, your call already given)

You OK'd integerizing the raycaster `cos/sin` (no feel-check) — good, but to be clear:
that's **future-proofing hygiene, not this bug** (we proved `cos/sin` doesn't reach the
default-scenario digest). The actual divergence is a non-transcendental float bridge (or
the py3.11↔3.12 seed difference), which the localization above pins down. We fix the
*real* one once it's named; the `cos/sin` integerization is a separate, low-risk,
golden-preserving cleanup we can do whenever.

**None of this blocks ML:** same-box training is float-deterministic regardless — the
cross-machine divergence only matters for cross-machine lockstep / heterogeneous
distributed training, which is deferrable. So you're free to pivot to ML whenever.
