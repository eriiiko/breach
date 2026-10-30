# References: the flame VFX (graphics arc G3 "Fire", 2026-10-23)

The flame layer's spec twin, `renderer/fire_vfx_ref.py` (patch F0b), carries
these in its header (iron rule: "any file implementing a published technique
carries an author + paper citation"). F1/F2's GL pass (`renderer/fire_vfx.py`,
`shaders/fire_vfx_*.fs`) cites the same list. Design:
`docs/g3_fire_design_2026-10-23.md` §11.

**PDFs are NOT archived here.** F0b did not download them, so this index
holds the references and where to get them. **ACTION FOR ERIK:** drop the PDFs
he wants kept beside this note. DOIs and URLs were written from memory and
have not been checked online in F0b. Confirm them when a PDF is fetched.

## The papers

1. **Stam, J. (1999).** "Stable Fluids." *Proc. SIGGRAPH 99*, 121–128.
   DOI 10.1145/311535.311548.
   - What we use: semi-Lagrangian advection and the projection step.
2. **Stam, J. (2003).** "Real-Time Fluid Dynamics for Games." *Game
   Developers Conference 2003*. Author's copy:
   dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf.
   - What we use: the grid-solver recipe that games use.
3. **Fedkiw, R., Stam, J. & Jensen, H. W. (2001).** "Visual Simulation of
   Smoke." *Proc. SIGGRAPH 2001*, 15–22. DOI 10.1145/383259.383260.
   - What we use: vorticity confinement (stage `confine`).
4. **Nguyen, D. Q., Fedkiw, R. & Jensen, H. W. (2002).** "Physically Based
   Modeling and Animation of Fire." *ACM Trans. Graph.* 21(3) (SIGGRAPH
   2002), 721–728.
   - What we use: expansion across the reaction front, entered as a
     prescribed divergence in the pressure solve (`q`, `pressure_rhs`).
5. **Harris, M. J. (2004).** "Fast Fluid Dynamics Simulation on the GPU."
   *GPU Gems*, ch. 38, Addison-Wesley. Free online at
   developer.nvidia.com/gpugems (part VI, chapter 38).
   - What we use: the collocated texture-grid formulation, Jacobi pressure
     and the boundary handling that the GL port follows.
6. **Selle, A., Fedkiw, R., Kim, B., Liu, Y. & Rossignac, J. (2008).** "An
   Unconditionally Stable MacCormack Method." *J. Sci. Comput.* 35(2–3),
   350–371. DOI 10.1007/s10915-007-9166-4.
   - What we use: MacCormack advection of R and R·T, with the min/max
     limiter and the fallback to semi-Lagrangian (`advect_flame`).
7. **Bridson, R., Hourihan, J. & Nordenstam, M. (2007).** "Curl-Noise for
   Procedural Fluid Flow." *ACM Trans. Graph.* 26(3) (SIGGRAPH 2007), art. 46.
   DOI 10.1145/1276377.1276435.
   - What we use: divergence-free noise as `curl(psi)`, scaled by the local
     fire intensity (`curl_noise`).
8. **Cetegen, B. M. & Ahmed, T. A. (1993).** "Experiments on the periodic
   instability of buoyant plumes and pool fires." *Combustion and Flame*
   93(1–2), 157–184. DOI 10.1016/0010-2180(93)90090-P.
   - What we use: the puffing frequency `f ≈ 1.5/√D` Hz, with D in metres
     (`puff_factor`).
9. **Sørb, "Ignitement" fire-VFX technical breakdown** (Unity blog, 2026):
   unity.com/blog/real-time-fluid-simulation-fire-vfx-ignitement-breakdown.
   This is the URL `docs/fire_vfx_plan_2026-08-24.md` records.
   - What we use: the overall approach, a cheap 2D fluid driven by gameplay
     fire. `docs/Breach_Fire_Rendering_Summary.md` summarises it.

## Not a paper: the screened solve

The screened Poisson problem, `(∇² − 1/ℓ²) p = ∇·u* − q` (design §13.1), is
the design's own amendment, not a published technique. The `−p/ℓ²` term
stands for the plume leaving the plane upward. It makes a sealed room's
problem unique and bounded.
