# Heightmap refinement tool — design note

**Status:** first-draft design for discussion — NOT canon, NOT built. Erik's idea (2026-06-13),
sketched for us to argue over before building. On approval it becomes a tool + (the format parts)
canon in the graphics/level chapters.

---

## 1. The problem

We generate the per-level **heightmap** (per-pixel floor relief) with a single global Depth-Anything
pass over the whole ship art. It now drives two things — the **ship's normal-mapped lighting** and the
**water's per-pixel depth** (the protrusion/pooling effect) — so its quality is suddenly load-bearing.
But a single global pass has two weaknesses, both of which Erik hit:

1. **Low local detail.** The model sees the whole 3900×6456 ship at once and resolves only coarse
   relief — fine features (a console's buttons, crate edges, debris) come out mushy.
2. **No defined "level 0."** Depth-Anything emits a *relative, per-image-normalized* depth, so the
   floor itself sits at some arbitrary nonzero value (~0.5). That is what caused the "fill forever"
   water bug — there's no known floor baseline. (Patched at render time by the `height_floor` dial;
   the *cure* is baking a real zero into the map.)

## 2. The vision (Erik)

Keep the global heightmap (it does the main job), but **refine the regions that matter, at higher
quality, and merge them back** — iterate on details as much as you like, with any method, for as long
as you like, **one art layer at a time** (base / furniture / destroyed, independently). The principle
is sound and well-founded: a *zoomed crop* gives the depth model far more pixels per feature, so a
per-region regen genuinely beats the global pass on local detail.

## 3. The workflow

```
   in-game: zoom + box-select a region        (reuse align_level_art.py's view + cursor→art-px)
        │
        ▼  save the art-space rect (per layer)
   crop the exact pixels from the chosen art layer's diffuse (bare | furniture | destroyed)
        │
        ▼  regenerate (any method, iterate until satisfied)
   upscale crop (UltraSharp) → depth (Depth-Anything, now seeing fine detail) → [normal]
        │
        ▼  ALIGN the patch to the global heightmap (§4 — the hard part)
   rescale + offset the patch so it agrees with the global map at the seam, against level-0
        │
        ▼  FEATHERED MERGE into the global heightmap (+ regenerate the normal from the merged height)
   blend over a soft border → no seam
        │
        ▼  iterate (re-do the region, try a different method, move on)
```

One layer at a time: regenerating the base layer's heightmap does **not** force regenerating the
furniture/destroyed ones — you select *which layer* and work it alone (they have different geometry).

## 4. The hard parts (where the design earns its keep)

**4.1 Patch-to-global alignment — the core challenge.** A regenerated crop is *relatively normalized*
(Depth-Anything rescales every image to [0,1] independently), so the patch's depth values are **not on
the same scale** as the global map. Dropping it in raw would create a step at the border. The fix is
the classic stitching move: in the *overlap ring* around the patch, fit an affine map `patch' = a·patch
+ b` that matches the patch's statistics (mean + slope) to the global map there, then apply it to the
whole patch. (Equivalent: Poisson/gradient-domain blending if affine isn't enough — heavier, hold as
the upgrade.) This is the make-or-break step; without it, refinement looks worse than the global pass.

**4.2 Level 0 — a shared baseline.** Establish a **known floor value = 0** for the whole heightmap, so
the render needs no `height_floor` fudge and patches share a reference:
- **Detect the floor:** the floor is the most *common* surface, so the dominant mode of the relief
  histogram ≈ floor level. Subtract it → floor at 0, raised features positive, dips negative.
- **Per-patch:** §4.1's affine alignment is *to the already-level-0 global map*, so a merged patch
  inherits the same zero automatically. Level-0 is set once globally, preserved by every merge.

**4.3 Feathered merge — no seams.** Blend the aligned patch into the global over a soft border (a
cosine/smoothstep falloff of width *w*), so the transition is invisible. Re-derive the *normal map*
from the merged *height* (one Sobel pass) rather than merging normals directly — heights merge
cleanly, normals don't.

**4.4 Coordinate bookkeeping.** Each region is an **art-space rect per layer**; the level's
`[art.align]` transform maps it to the grid for the in-game selection. Save the rects (a sidecar
`<layer>_patches.json`?) so refinements are reproducible and re-runnable.

**4.5 Per-layer independence.** Base/furniture/destroyed each have their own global heightmap + patch
set. The tool's "which layer" selector gates the whole pipeline; nothing couples them.

## 5. What we build on (most of it exists)

- **In-game region select + coords:** `tools/align_level_art.py` already has the pyray viewer, zoom,
  cursor→art-px, and the `[art.align]` transform. A box-select mode + "export rect" is a small add.
- **Regen pipeline:** `tools/depth_to_normal.py` (Depth-Anything → height → normal) and the upscalers
  (`tools/upscale_pth.py` UltraSharp, the ESRGAN exes) — already produce exactly the crop→depth→normal
  chain; the tool calls them on a crop instead of the whole image.
- **Level wiring:** the `[art] height` path now exists (the water feature), so the merged heightmap
  drops straight into the level.

So the tool is mostly *orchestration + the align/merge math* — the heavy ML pieces are built.

## 6. Proposed build shape

1. **Level-0 pass (standalone, first):** a script that takes the existing global heightmap, detects
   the floor mode, and re-baselines it to floor=0. Immediately retires the `height_floor` fudge and is
   useful on its own. (`tools/heightmap_level0.py`.)
2. **Region-select + crop export** in the editor: box-select, choose layer, write the art-space rect.
3. **Regen-a-patch** (CLI on a rect): crop → upscale → depth → the §4.1 affine align → §4.3 feathered
   merge into the global height → re-derive the normal. Iterable (re-run until happy).
4. **(later) In-tool preview/iteration loop** — see the merge in place before committing.

## 7. Open questions for Erik

1. **Selection UX:** box-drag in `align_level_art.py` (extend the existing editor), or a separate
   dedicated tool window? (I lean: extend the editor — the coords/view are already there.)
2. **Align method:** affine (mean+slope match in the overlap ring) as v1, Poisson/gradient-domain as
   the upgrade if affine seams show — agree?
3. **Level-0 floor detection:** histogram-mode auto-detect, or a manual "click a floor tile to set
   zero" in-tool (more control)? (I lean: auto-detect with a manual override.)
4. **Does the heightmap stay one map per layer, or do we ever want a single merged "current" heightmap
   the game swaps per art-state?** (Probably per-layer, mirroring the diffuse layers.)
5. **Scope of v1:** is the standalone **level-0 re-baseline** (step 1) enough to start — it fixes the
   most visible problem today — with the patch-refinement as a fast-follow? Or build the whole loop?
