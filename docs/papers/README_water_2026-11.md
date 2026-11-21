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
     **Since W1b** the lattice tiles a WAVE SET instead of a texture: each
     vertex turns and shifts the procedural sines (`water_surface.hex_waves`,
     `shaders/water.fs::hexWaves`), the same weights and hash.
3. **Finch, M. (2004).** "Effective Water Simulation from Physical Models."
   In R. Fernando (ed.), *GPU Gems*, chapter 1. Addison-Wesley / NVIDIA.
   Open online: developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/
   chapter-1-effective-water-simulation-physical-models.
   **Not archived as a PDF:** NVIDIA publishes the book chapter as a web page
   only; fetch it from the address above.
   - What we use (W1b, 2026-11-20): the surface as a SUM OF TRAVELLING SINES,
     each with its own direction, wavelength and phase, moving at the speed
     the dispersion relation gives its wavelength (his eq. for
     `omega = sqrt(g k)`; we use the capillary-gravity form `sqrt(g k +
     (sigma / rho) k^3)`), and his sharpening exponent `2A((sin + 1)/2)^k`
     (eq. 8b) with its analytic derivatives (we also derive the second, for
     the caustic). In the code: `renderer/water_surface.py::wave_set`,
     `wave_sum` and `shaders/water.fs::waveSum`.

## W2: foam with a life

3. **Yingst, M., Alford, J. R. & Parberry, I. (2011).** "Very Fast Real-Time
   Ocean Wave Foam Rendering Using Halftoning." *Proceedings of the 6th
   International North American Conference on Intelligent Games and
   Simulation (GAMEON-NA)*, EUROSIS, Troy NY, pp. 27-34.
   ianparberry.com/research/seafoam (the plan's "Parberry's sea foam").
   **Archived here:** `yingst2011_ocean_foam_halftoning.pdf` (the conference
   version, fetched 2026-11-20, 2.7 MB).
   - What we use: the foam AMOUNT as a threshold on a fixed pattern (their
     precomputed dither array), so foam breaks up into holes and strings as
     it thins instead of turning transparent. Our pattern is a bubble raft
     (Worley membranes, `noise_bake.bake_foam_rgba`), rank-equalised so the
     threshold that covers a share f is exact (`water_foam.lace_threshold`,
     for the mean of two sizes), and it slides with the flow map. In the code:
     `renderer/water_foam.py::draw_cover`, `shaders/water.fs::foamCover`.
4. **Selle, A., Fedkiw, R., Kim, B., Liu, Y. & Rossignac, J. (2008).** "An
   Unconditionally Stable MacCormack Method." *J. Sci. Comput.* 35. Already
   the fire's reference (`docs/papers/fire_vfx_references.md`, item 6).
   - What we use: MacCormack advection with the min / max revert, for the
     foam amount (`water_foam.advect`, `shaders/water_foam.fs`).
5. **Ojeda and Susín** (river foam: an amount made, carried by the flow and
   lost per frame) -- through the plan's section 4
   (`docs/presentation_graphics_plan_2026-11-15.md`); no paper archived.

## W4: sharp caustics

6. **Wallace, E. (2011).** "WebGL Water." Demo and source,
   madebyevan.com/webgl-water (github.com/evanw/webgl-water, MIT licence);
   the method is written up in his "Rendering Realtime Caustics in WebGL"
   (medium.com/@evanwallace/rendering-realtime-caustics-in-webgl-2a99a29a0b2c,
   which refused an automated fetch on 2026-11-20).
   **Archived here:** `wallace2011_webgl_water_renderer.js` (his
   `renderer.js`, 13 KB, fetched 2026-11-20 from the repository's master;
   the caustics shader is `causticsShader`, the MIT notice is in its header).
   - What we use (since 2026-11-21, Erik's call: "scatter"): his SCATTER
     form itself. A grid of samples of the water surface (24 per tile),
     each moved by the refraction map `x -> x + depth (1 - 1/n) grad h`
     (Snell to first order in the slope; his `IOR_WATER = 1.333`) onto the
     floor, each grid triangle rasterised additively into a light image with
     `oldArea / newArea` (his ratio, per triangle instead of from `dFdx` /
     `dFdy`), so the light image's mean is the incident light; the floor's
     source light x `1 + (E - 1) x caustic_strength`. In the code:
     `renderer/water_surface.py::scatter_light` / `scatter_disp` /
     `caustic_gain` / `caustic_term`; `shaders/water_scatter_slope.glsl`,
     `shaders/water_scatter.vs/.fs`, `shaders/water.fs::causticLight`.
     (W4 as first built, 2026-11-20 to 11-21, used the ratio's analytic
     limit, `1/|det J|`, in gather form from the surface's Hessian; it was
     biased bright and darkened past the first focus, and is deleted.)
7. **Snell's law of refraction** in vector form, `t = eta d + (eta c1 -
   c2) n` (any optics text, e.g. Hecht, E. (2017), *Optics*, 5th ed.,
   Pearson, sec. 4.4). **Not archived:** a textbook.
   - What we use: at small slopes a vertical ray through a surface of slope
     `g` leaves at the horizontal rate `(1 - 1/n) g`, along the slope: the
     caustic's map above, and THE FLOOR BEND (the floor is read where the
     map sends the view ray; it replaced `refract_strength`). In the code:
     `renderer/water_surface.py::floor_offset`, `shaders/water.fs` (`bend_m`).

## The lamps' own glints (2026-11-21)

8. **Karis, B. (2013).** "Real Shading in Unreal Engine 4." SIGGRAPH 2013
   course *Physically Based Shading in Theory and Practice*, course notes.
   cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf.
   **Archived here:** `karis2013_real_shading_unreal_engine4.pdf` (fetched
   2026-11-21, 2.9 MB, 59 pages).
   - What we use: the sphere light's widened GGX lobe, `alpha' =
     saturate(alpha + sourceRadius / (3 distance))` (his Eq. 10) with the
     lobe renormalised by `(alpha / alpha')^2` (Eq. 14), for each mounted
     lamp's mirror glint seen from its true position. In the code:
     `renderer/water_surface.py::lamp_glint`, `shaders/water.fs` (the lamps'
     loop).
