# References: the water look (arc #97, 2026-11)

The water look's code cites these in its headers (iron rule: "any file
implementing a published technique carries an author + paper citation").
Design: `docs/water_look_design_2026-11-19.md` (sections 2, 9 and 10).

## W1: the fine surface that flows

1. **Vlachos, A. (2010).** "Water Flow in Portal 2." SIGGRAPH 2010 course
   *Advances in Real-Time Rendering in 3D Graphics and Games*. Author's copy:
   alex.vlachos.com/graphics/Vlachos-SIGGRAPH10-WaterFlow.pdf.
   **Archived here:** `vlachos2010_water_flow_portal2.pdf` (fetched
   2026-11-20, 6.6 MB, 54 slides).
   - What we use: the two-phase flow map. Two copies of a texture slid along
     a flow field, half a cycle apart, cross-faded so each one's reset sits
     under zero weight; a per-point phase offset from a noise so the reset
     never pulses the whole surface. In the code: the crossfade law
     `renderer/advected_noise.py` (`advection_phase`, `layer_ages_weight`,
     `phase_offset_field`, `ADVECTION_GLSL`), applied to the water by
     `renderer/water_surface.py::detail_slope` and `shaders/water.fs`.
2. **Mikkelsen, M. S. (2022).** "Practical Real-Time Hex-Tiling." *Journal of
   Computer Graphics Techniques (JCGT)* 11(3), 77-94.
   jcgt.org/published/0011/03/05/ (open access; the paper's PDF is
   jcgt.org/published/0011/03/05/paper.pdf).
   **Not archived as a PDF:** the published file is 67 MB (full-resolution
   figures), too large for the repository; fetch it from the address above.
   - What we use: the hex-tiled lookup that hides a repeating texture --
     three taps on a hexagonal (simplex) lattice, each with a per-vertex
     offset and rotation, blended by `(1 - c + c * luminance) * w^7`
     normalised (his reference shader's `hex2colTex` with `g_exp = 7`,
     `g_fallOffContrast = 0.6`). Our changes: the sampled value is a SLOPE
     vector, so each tap's slope is rotated back with its tap (the gradient
     of a rotated field) and its length stands in for luminance; the
     per-vertex hash is an integer one (lowbias32), so the GPU and the numpy
     twin draw the same bits (his `frac(sin(.) * 43758.5)` hash differs
     between float32 and float64). In the code:
     `renderer/water_surface.py::hex_slope` and `shaders/water.fs::hexSlope`.

## Later patches

W4's caustic (Wallace 2011) and W2's foam (Parberry) archive theirs here when
they land, per the design's section 10.
