# Water-surface shader research — methods + citations

**Status:** research backing the `architecture/graphics/water_rendering.md` chapter (2026-06-12).
Methods/citations for the design discussion — not canon, not code. Everything here is screen-space /
texture-pass math, matching our raylib + GLSL pipeline. We already compute the two inputs every
method needs: a per-pixel **surface normal** `N` (ripple field + ambient sines) and a per-pixel water
**depth** `d`.

**Framing that changes the ranking:** we look straight *down* — there is no sky. So "reflection" means
*specular response to our actual light sources* (flashlights, fire, emissive screens), not sky/cubemap
reflection. This makes the cheap end of the spectrum the *correct* end, not a compromise.

---

## 2a — Water-surface reflections (cheapest → highest quality)

**Tier 0 — Schlick Fresnel weight (build first, ~free).** A per-pixel `0..1` "how mirror-like is the
water here," driven by view angle; it's the *weight* everything else multiplies through.
`R(θ) = R0 + (1−R0)(1−cosθ)^5`, `cosθ = max(dot(N,V),0)`, `R0 ≈ 0.02` for air→water. **Top-down fix:**
feed Fresnel the *perturbed* ripple normal, not the geometric one — straight down, geometric `dot(N,V)≈1`
gives a dead flat surface; the ripple tilts are what create grazing angles and therefore glints. Our
normal map is the linchpin of the whole effect. (Schlick — Wikipedia; Pete Shirley; lettier *Fresnel Factor*.)

**Tier 1 — specular glints of the light sources via the normal (the visual core).** Per light, a
specular lobe on the perturbed normal = the moving highlight of a flashlight/fire/screen on the ripples.
- *Blinn-Phong* (cheapest): `pow(max(dot(N,H),0), shininess)`, `H=normalize(L+V)`. One `pow`, but the
  small intense highlight **sparkle-aliases ("fireflies")** on an animated ripple field under bloom.
- *GGX / Cook-Torrance* (**recommended**): narrow bright core + soft tail → realistic wet glint *and*
  attenuates the firefly aliasing Blinn-Phong suffers on moving water. `D = α²/(π((dot(N,H))²(α²−1)+1)²)`,
  `α=roughness²`. Cost worry is overstated: the `dot(L,H)` simplification (Filmic Worlds) folds
  visibility+Fresnel into one term (optionally a 2D LUT), making "cheap GGX" barely above Blinn-Phong.
  **Roughness as a lever:** low = sharp mirror glint (still puddle), high = broad shimmer (choppy) —
  drive it from the ripple agitation field. (gfxdevunity; LearnOpenGL PBR; Graphics Compendium ch.69;
  Filmic Worlds *Optimizing GGX with dot(L,H)*.)

**Tier 2 — matcap / spherical-env lookup (optional polish).** Bake a "lit sphere," look it up by the
screen-space normal (`muv = N.xy*0.5+0.5`). Fakes a *static* environment — can't know where the moving
flashlight is, so **not** a substitute for Tier 1. Possible niche use: a faint baked ambient tint so
water isn't pure black where no light hits. One texture fetch. (hughsk/matcap; Clicktorelease SEM.)

**Tier 3 — screen-space reflection (overkill here).** Smear real framebuffer pixels along the
normal-offset and keep only bright/emissive results: `refl = texture(scene, uv + N.xy*k); refl *=
step(thresh, luminance(refl))`. Top-down, the normal-offset mostly grabs the *dark floor next to the
water*, so the bright-gate kills it most of the time; the cases it helps (fire/screen adjacent to a
flooded patch) are already covered, more controllably, by the Tier-1 glint. Expensive insurance for a
narrow win — revisit only if we want literal reflected *shapes*. (lettier *SSR*; devfault; hedgefield.)

| Rank | Method | Cost | Verdict |
|---|---|---|---|
| 1 | Schlick Fresnel (on perturbed N) | ~free | **Build first — foundational** |
| 2 | GGX specular glints (cheap dot(L,H)) | low | **Top recommendation — *the* wet glint** |
| 3 | Blinn-Phong (softened) | lowest | Fallback for #2 (watch fireflies) |
| 4 | Matcap ambient fake-env | 1 fetch | Optional polish (static only) |
| 5 | SSR (bright-gated framebuffer) | medium | Overkill for v1 |

---

## 2b — Refraction ("water breaking the light" on the floor below)

**Tier 0 — normal-driven UV distortion × depth (the workhorse, must-have).** Sample the floor art at
an offset ∝ surface normal XY: `bump = 2*texture(normalMap,uv).xy−1; newUV = baseUV + bump*vScale`
(vScale ≈ 0.05). **Depth-scale it** — `newUV = baseUV + bump*vScale*saturate(d)` — so shallow film
barely distorts and deep water distorts strongly: physically correct (longer path) and it kills the
shimmer-at-the-shoreline artifact. One extra fetch + a few ALU; a perfect match for our `N`+`d`.
(GPU Gems 2 ch.19 *Generic Refraction*; Catlike Coding *Looking Through Water*.)

**Tier 1 — Beer–Lambert depth tint (strong recommend, trivial).** Tint the refracted floor toward a
water colour, stronger with depth — what makes a flooded corridor read as *volume*, not a wet decal.
`fog = exp2(−fogDensity*d); col = mix(waterFogColor, floorCol, fog)`. Per-channel absorption (red dies
first) is why water goes blue-green; on a dark ship a desaturated green-black likely reads better than
ocean-blue. (Beer–Lambert — Wikipedia; Catlike Coding; arXiv 1109.6494 ocean survey.)

**Tier 2 — caustics on the floor (signature mood, phase 2).** Dancing refracted light filigree — very
high-impact in a dark ship. Cheap procedural Voronoi (no precompute): `c1=voronoi(uv*s+t*v1);
c2=voronoi(uv*s−t*v2); caustic=pow(min(c1,c2),k)`. **Crucial:** *multiply caustics by the per-pixel
light intensity* — caustics are light being refocused, so they must ride the flashlight/fire pool
(caustics in darkness make no sense; caustics on the flashlight pool are gorgeous and motivated). First
thing to cut under budget. (Mirza Beig procedural Voronoi; godot-realtimecaustics; GPU Gems 1 ch.2.)

**Tier 3 — chromatic aberration on the refraction (optional flourish).** Per-channel offset of the
refracted sample (`r,g,b` at slightly different `bump` scales) = light dispersion at the boundary.
Turns 1 floor fetch into 3 — the main expense in 2b; keep the amount tiny. (halisavakis; GM Shaders
Mini / Xor; Maxime Heckel.)

| Rank | Method | Cost | Verdict |
|---|---|---|---|
| 1 | Normal-driven UV distortion × depth | 1 fetch + ALU | **Must-have core refraction** |
| 2 | Beer–Lambert depth tint | trivial | **Strong recommend — sells volume** |
| 3 | Procedural Voronoi caustics × light | moderate | Phase 2 — biggest mood payoff |
| 4 | Chromatic aberration | +2 fetches | Optional polish |

---

## How 2a + 2b compose — one unified Fresnel reflect+refract pass

Fresnel is the single dial that unifies both halves. One fragment shader, one normal, one depth:

```glsl
vec3  N = perturbedSurfaceNormal;        // ripple field + ambient sines
float d = waterDepth;

// REFRACTION branch (head-on, dominates top-down) [2b]
vec2  off    = N.xy * refractStrength * saturate(d);   // GPU Gems 2 + depth scale
vec3  floorC = sampleFloor(uv + off);                  // (optionally 3x for CA)
float fog    = exp2(-fogDensity * d);                  // Beer–Lambert
vec3  refr   = mix(waterColor, floorC, fog);           // depth tint
refr        += causticPattern(uv) * lightAtPixel;      // caustics ride the light

// REFLECTION branch (grazing + glints) [2a]
float spec = ggxSpecular(N, L, V, roughness);          // per light source
vec3  refl = spec * lightColor;                         // flashlight/fire glint

// UNIFY with Fresnel (Schlick)
float F = R0 + (1.0 - R0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);   // R0≈0.02
vec3  water = mix(refr, refl, F);        // head-on → see floor; grazing/rippled → glint
```

`mix(refraction, reflection, Fresnel)` is the textbook water equation. Top-down, base `F≈0.02` (we
mostly see the floor — correct), and the **ripple normals locally spike `F`** to create crest glints.
**Order:** refract + tint + caustics → spec/reflect → Fresnel-mix → composite **foam** last (foam is
surface scatter, masked by shallow depth `1−saturate(d)` and/or ripple-crest height). (Cyanilux water
breakdown; ameye.dev stylized water; USPTO 10290142 `Reflection·R + Underwater·(1−R)`.)

---

## What to build first (the recommendation, for the discussion)

**v1 — the unified Fresnel pass, three terms:** (1) Schlick Fresnel on the *perturbed* ripple normal as
the master weight; (2) cheap GGX specular glint per light source (`dot(L,H)` form — barely above
Blinn-Phong, no sparkle-alias on moving ripples); (3) normal-driven UV-distortion refraction × depth +
Beer–Lambert depth tint. ~90% of "the flashlight glints off the wet floor, and the floor warps and
darkens with depth" for a handful of fetches + a couple dozen ALU — no SSR, no caustics, no extra render
targets, reusing the `N` and `d` we already compute.

**phase 2 — mood:** procedural Voronoi caustics × per-pixel light intensity, so flooded corridors come
alive under the flashlight. Highest-drama addition for a dark ship; cleanly optional, shouldn't gate v1.

**hold as polish:** SSR + chromatic aberration — low return for a straight-down dark scene.

---

## Sources
Fresnel/Schlick: Wikipedia; Pete Shirley (psgraphics); lettier *Fresnel Factor*. — GGX vs Blinn-Phong:
gfxdevunity; LearnOpenGL PBR; Graphics Compendium ch.69; Filmic Worlds *Optimizing GGX with dot(L,H)*.
— SSR/2D framebuffer reflection: lettier *SSR*; devfault *Screen-space Water*; hedgefield *2D water
reflections*. — Matcap: hughsk/matcap; Clicktorelease SEM. — Refraction: GPU Gems 2 ch.19; Catlike
Coding *Looking Through Water*. — Beer–Lambert: Wikipedia; arXiv 1109.6494. — Caustics: Mirza Beig
procedural Voronoi; godot-realtimecaustics; GPU Gems 1 ch.2. — Chromatic aberration: halisavakis; GM
Shaders Mini (Xor); Maxime Heckel. — Foam/compositing: Cyanilux; ameye.dev stylized water.
