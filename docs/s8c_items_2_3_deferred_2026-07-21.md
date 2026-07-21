# S8c items 2 & 3 — DEFERRED (accepted gap), 2026-07-21

**Decision (Erik, 2026-07-21):** after the S8c item-1 fire-FPS fix landed
(`9eb47c0`, batched `cast_fire_heat` device cast — 277× on a 600-fire
firestorm, `heat` byte-identical), items 2 (render CUDA-GL interop) and 3
(recorder kernels) are **deferred and documented as accepted gaps**, not built.
Two recon passes showed neither, *as literally framed in the kickoff*
(`docs/s8c_kickoff_2026-07-21.md`), fits the current architecture well enough to
justify the effort. This is a "bias to the simplest honest design" call
(autonomous-patch-workflow: *document an accepted gap rather than build
machinery for marginal payoff*). Not a rejection of the ideas — a "not now, not
in this shape."

---

## Item 2 — render CUDA-GL interop: WHY DEFERRED

Recon (renderer map, 2026-07-21):

- **The renderer is raylib via `pyray` (cffi).** Not moderngl / PyOpenGL /
  pygame. Fields reach the screen through per-frame CPU texture uploads
  (`renderer/core.py::update_rgba*_texture` → `rl.update_texture` with
  `ffi.from_buffer`). There is **no** CUDA-GL interop anywhere (no PBO, no
  `cudaGraphicsGLRegisterBuffer`), and **cupy is not imported renderer-side**
  (only in `gamemap.enable_residency`).
- **raylib/pyray hides the GL context and texture ids** behind cffi. There is
  **no exposed hook** to register a raylib texture (`tex.id`) with cupy's GL
  graphics-resource API, and **no repo precedent** for doing so. This plumbing
  is the real unknown, independent of which field is chosen.
- **The genuinely render-only fields — `smoke_glow`, `ripple`, `ripple_v` — are
  computed on the HOST**, not the GPU (`smoke_glow` by the renderer's own CPU
  raycaster; `ripple`/`ripple_v` by `step_tail` on the mirror). So there is **no
  device copy for interop to "skip."**
- **Every GPU-resident field the renderer reads is SYNCED** (`gas`,
  `atmosphere`, `wave_p`, `temperature`, `water_depth`) and therefore must D2H
  to the numpy mirror each tick anyway — the **Q4 locked decision**
  (combat/recorder read the mirror). Interop for these would be an *additional*
  device→texture copy running alongside the mandatory mirror D2H, not a saved
  transfer.

**Net:** to get any interop win you would first have to (a) port the render-only
computations (`smoke_glow`, ripple) onto the GPU — a much larger change than a
buffer-sharing tweak — and (b) solve the raylib↔cupy GL-buffer sharing with no
exposed hook and no precedent. Large, speculative, and feel-adjacent
(Erik-gated). Not justified while raylib owns the GL side and the mirror D2H is
mandatory.

**Revisit if:** the renderer moves off raylib (to moderngl/PyOpenGL with
accessible buffer ids), OR the render-only fields get ported to GPU kernels for
another reason (then a device→GL path for them becomes a real, bounded win), OR
batched-training makes the Q4 mirror droppable (the kickoff's separate
end-state, explicitly *not* S8c).

## Item 3 — recorder kernels: WHY DEFERRED

Recon (recorder map, 2026-07-21):

- The recorder (`src/simulation/recorder.py::PhysicsRecorder`) reads the
  **already-mirrored host data**: the Q4 synced-set D2H
  (`physics_runner._step_resident` step 6) lands the fields on the mirror each
  tick for combat regardless, and the recorder copies from there.
- Its per-tick cost is **6 host plane copies + an fp64 dequant + one blowup
  abs-diff/max reduction** on already-host arrays (recorder.py:117-166) — cheap
  CPU work, and **not the fire bottleneck** item 1 fixed.
- A device-side ring buffer would need the fields to stay resident AND its own
  device buffer, then **D2H on dump anyway**; and the recorded float32 bytes are
  produced by `astype(f64)/65536 → f32` plus the blowup `.max()` that decides
  *whether a dump fires* — a device path must reproduce numpy's rounding
  **bit-exactly** (recordings are replay/desync evidence;
  `tests/test_entity_digest.py::test_recorder_sections_present_iff_entities`
  pins the recorder↔digest shared-serializer bytes). Narrow payoff, real
  identity risk.

**Net:** the recorder is not where the frame budget goes, and moving it
device-side trades a small host cost for a byte-identity hazard. Not justified.

**Revisit if:** a future bench shows the always-on per-tick ring-buffer capture
is a measurable frame-budget cost — in which case the first move is a **cheap
host-side** fix (sample/opt-out, or drop the redundant per-tick blowup
reduction), NOT GPU kernels.

---

## Where S8c stands

- **Item 1 (fire-FPS fix): DONE + merged + pushed** (`9eb47c0`). The one
  load-bearing S8c win.
- **Items 2 & 3: deferred, documented (this file).** The S8 optimize line's next
  real item remains **S8b (CUDA graphs)** — parked, its payoff is many-env
  training (needs a design pass on the n_sub-variable launch topology), per the
  kickoff's "NOT this task."
