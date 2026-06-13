# Visual tuning guide

The practical companion to the graphics chapters: **what each look-knob means and how to tune it by
eye**. Per the graphics README, look-and-feel tuning is a first-class graphics concern — this is where
the "what does this slider do" lives, separate from the *why* in the design chapters.

**The tuning workflow.** Run `tools/lighting_demo.py` — it exposes every visual knob as a live raygui
slider (the panel on the right), so you drag and see the result instantly while the sim runs. When a
value looks right: either **Save** a preset (to `tools/lighting_presets.toml`) or copy the number into
the matching `config.toml` section (those are the *shipped* defaults; the demo seeds its sliders from
them at startup). The main game reads config at construction, so config changes need a restart; the
demo sliders are live.

---

## Water surface optics — `[graphics.water]`

The water look is a single shader pass (`shaders/water.fs`, design: `water_rendering.md`). It reuses
the ripple field + depth the sim already computes, and the scene's light buffers. **Glints only ride
real ripples** — a dead-flat puddle barely glints by design, so to evaluate the surface, *pour a puddle
(`U`) and throw a grenade in it* (`T`, then click) to get waves moving.

### Recommended tuning order
Tune in this order — each stage makes the next easier to judge:
1. **Presence** (`alpha_*`) — get the water visible at the opacity you want first.
2. **Volume** (`fog_density`, `water_color`) — the depth/menace feel.
3. **Surface shape** (`ripple_scale`) — how pronounced the ripples read.
4. **Sparkle** (`glint_strength`, then `roughness_base`/`roughness_agitation`) — the highlights.
5. **Floor warp** (`refract_strength`) — the refraction, last (subtle).

### The knobs

| Knob (config key / slider) | Default | Range | What it does — how to tune |
|---|---|---|---|
| **glint_strength** (Glint) | 2.0 | 0–8 | Brightness of the specular highlights — light glinting off the ripple crests. **Up** = brighter, more prominent sparkle; **down** = subtle. This is the master "how much does the surface catch the light" dial. (Glints are gated to lit, forward-facing ripples, so they only appear under a flashlight/fire and where the surface tilts toward it.) |
| **roughness_base** (Roughness) | 0.08 | 0.02–0.5 | How **sharp vs. broad** the glints are. **Low** = tight, mirror-like, bright pinpoint glints (a still puddle); **high** = broad, soft shimmer (choppy water). The art lever for "glassy" vs. "agitated." |
| **roughness_agitation** (Rough agit) | 0.6 | 0–2 | How much *ripple activity* broadens the roughness. **Up** = sloshing/disturbed water shimmers broad while calm water stays sharp (ties the look to what's happening); **0** = roughness is constant regardless of motion. |
| **fog_density** (Fog dens) | 3.0 | 0.2–12 | Beer–Lambert depth tint rate — how fast the floor fades into `water_color` with depth. **Up** = water reads deep/murky, darkens quickly (menace, can't see the bottom); **down** = clear, you see the floor through deep water. The main "how deep does it feel" dial. |
| **water_color** (config only) | 0.03, 0.10, 0.18 | rgb 0–1 | The deep-water volume tint the floor fades toward. On a **dark ship**, a desaturated green-black reads more like grim flooded steel than ocean-blue — push toward green/teal-black, away from bright blue. (No slider yet — edit config + restart, or we can add one.) |
| **ripple_scale** (Ripple scl) | 8.0 | 0–24 | Converts the (millimetric) ripple height into an on-screen normal slope — i.e. how **pronounced/bumpy** the surface reads. **Up** = ripples tilt the normal more → more glint variation + more floor-warp; **down** = flatter, calmer surface. Interacts with glint and refract (it scales the normal both feed on). |
| **refract_strength** (Refract) | 0.02 | 0–0.08 | How much the floor **warps/wobbles** under the surface (the refraction UV offset). **Up** = more distortion, watery wobble; **too high** = swimmy/unreal. Depth-scaled, so shoreline doesn't shimmer. Keep subtle. |
| **r0** (R0 Fresnel) | 0.02 | 0–0.2 | Fresnel reflectance head-on — the base "wet sheen" the surface reflects even looking straight down. Physically ~0.02 for water; **up** = more uniform sheen everywhere (glossier, less see-through). Mostly leave low; nudge up for a wetter, more reflective floor. |
| **alpha_scale** (Alpha scl) | 6.0 | 0–20 | How fast opacity ramps with depth. **Up** = water becomes opaque/present at *shallower* depth; **down** = stays see-through deeper. |
| **alpha_min** (Alpha min) | 0.15 | 0–0.6 | Minimum opacity of *any* water (even a thin film). **Up** = even a shallow puddle has visible presence; **down** = thin water nearly invisible. This is the "see-through vs. always-visible" floor — raise it if water disappears on you. |
| **alpha_max** (Alpha max) | 0.95 | 0.4–1.0 | Maximum opacity (deep water). **Down** = even deep water stays partly see-through (the "see-through, not opaque blue" look Erik wanted); **up** = deep water reads solid. |
| **ambient** (driven by the Ambient lighting sliders) | from lighting | — | How much the water body is lit by the scene's *ambient* light (vs. only the flashlight). Shares the lighting pass's ambient — raise the **Ambient** sliders to make water visible outside the flashlight beam, dim it to make water near-black except under a light. (Glints stay flashlight/fire-driven regardless — ambient lights the *body*, sources *glint*.) |
| **caustic_strength** (Caustic) | 2.5 | 0–8 | Brightness of the **caustics** — focused light dancing on the flooded floor, where the wavy surface acts as a lens. Computed from `+laplacian(ripple)` (concave-up/focusing = bright, clamped ≥ 0) × the per-pixel light, so it **only appears under a flashlight/fire** and **dances with the real ripples** (splash a grenade in a puddle, `T` then click). **Up** = brighter filigree; **0** = off. Rides the floor (× alpha), not the surface. |
| **caustic_scale** (Caust scl) | 6.0 | 0–24 | Spatial frequency of the optional **high-frequency procedural detail** layered onto the surface-curvature caustics for crispness (modulated by ripple energy, so it only crisps up where the water is moving). **Up** = finer sparkle; **0** = pure surface-curvature caustics (softer, the physics-only look). |
| **foam_threshold** (Foam thr) | 0.02 | 0.001–0.1 | The steepness/edge above which **foam** forms. Foam = whitecaps where `|grad ripple|` exceeds this + the wet/dry shoreline (high depth gradient). **Down** = foam appears on gentler fronts (more foam everywhere); **up** = only the steepest crests/edges foam (subtle). |
| **foam_intensity** (Foam int) | 0.6 | 0–2 | How **white/strong** the foam reads once it forms. **Up** = thicker, brighter whitecaps; **0** = no foam. Composited last into the base (surface scatter, alpha-bound — not additive light). Keep subtle: white at crests, not a snowfield. |
| **ca_amount** (Chrom ab) | 0.012 | 0–0.06 | **Chromatic aberration** on the refracted floor — the r/g/b channels are sampled at slightly offset refraction UVs (scaled `1 ± ca_amount`), giving a faint prismatic fringe. **Barely-there by default**; **up** = stronger colour fringing (stylised); **0** = off (single sample). |
| **wave_scale** (Wave scl) | 2.0 | 0.2–6 | **Idle-shimmer wave size.** Multiplies the ambient-sine spatial frequencies. **Up** = smaller/tighter idle waves; **down** = bigger sweeping waves. (Erik's fix for the idle waves reading "very big" — default 2.0 tightens them from the old hardcoded `1.0`.) The temporal speed is unchanged; this only resizes the lattice. |
| **ambient_amp** (Amb amp) | 0.06 | 0–0.3 | **Idle-shimmer strength** on still water — the amplitude base of the ambient sine lattice (the old hardcoded `0.06`). **Up** = more idle motion/shimmer even on a dead-calm puddle; **down/0** = still water reads glassy-flat (only real ripples move it). Energy from actual ripples still adds on top regardless. |
| **height_scale** (Height scl) | 0.4 | 0–2 | **Heightmap floor-relief gain** (metres of floor lift subtracted from depth). Needs a level WITH a floor heightmap ([art.bare] `height`, e.g. `unhcr_vessel_2`) — **inert** with no heightmap. The per-pixel relief × this gain is **subtracted from the per-tile water depth**, so the shader sees a *shallower* water column over raised floor: that ADJUSTED per-pixel depth drives **every** effect — refraction offset, Beer–Lambert tint, and the alpha ramp all vary per-pixel with it (a crate refracts/tints as the shallow water it sits in). Where the relief lifts the floor above the surface (depth ≤ 0) the feature **protrudes** (transparent, pokes through). **NOT alpha-only** — it adjusts the depth the whole pass computes off. **Up** = relief reads taller (more protrudes / shallower over features); **0** = off (depth == per-tile depth, bit-identical). Tuned ~0.4 so furniture emerges from a ~0.3–0.5 m puddle. |
| **height_edge** (Height edge) | 0.1 | 0.01–0.5 | **Protrusion-shoreline softness** (metres) — a `smoothstep` ramp over the first metres of *positive* adjusted depth, fading the alpha to 0 right at the protrusion edge so the waterline laps softly around a feature instead of cutting hard (it overrides the `alpha_min` floor at the edge). **Down** = a harder, crisper waterline against the crate; **up** = a softer, wider wet→dry fade. Only matters when `height_scale > 0` and a heightmap is bound. |

### Interactions worth knowing
- **No glints without ripples.** `glint_strength` does nothing on a dead-flat puddle — the normal must tilt. Splash it.
- **`ripple_scale` amplifies both glint variation and floor-warp** (they share the normal). If both feel too strong/weak together, it's probably `ripple_scale`.
- **`fog_density` and `alpha_*` both affect "presence"** but differently: fog is *color* (darkens toward water_color), alpha is *opacity* (how much floor shows through). A shallow puddle that's too invisible → raise `alpha_min`; a deep pool that doesn't feel deep → raise `fog_density`.
- **`r0` and `alpha` both add "wetness"** — r0 via surface sheen, alpha via opacity. Start with alpha; use r0 for the final gloss.

### Tuning the mood pass (phase 2 — shipped)
Caustics, foam, chromatic aberration, and the wave-size dials are now live (rows above). Tune them after
the core look:
6. **Wave size** (`wave_scale`, `ambient_amp`) — get the idle shimmer reading right (small + present,
   not the old big sweeping waves) before judging caustics/foam, since both react to the surface motion.
7. **Caustics** (`caustic_strength`, then `caustic_scale`) — pour a puddle (`U`) under the flashlight,
   throw a grenade in it (`T` + click), and dial the dancing focused light. Caustics are × the light, so
   they only show under a source — aim the flashlight at the puddle.
8. **Foam** (`foam_threshold`, then `foam_intensity`) — keep it subtle: white at the steepest crests +
   the wet/dry edge, not a snowfield. Lower the threshold for more, raise the intensity for whiter.
9. **Chromatic aberration** (`ca_amount`) — barely-there; nudge up only if you want a stylised fringe.
10. **Heightmap depth-adjust** (`height_scale`, then `height_edge`) — *only on a level with a floor
    heightmap* (e.g. `unhcr_vessel_2`). Pour/flood the floor (`U`), then raise `height_scale` until the
    furniture/debris pokes through the puddle at the depth you want, and set `height_edge` for the
    waterline crispness around it. The relief now ADJUSTS the per-pixel depth, so as you raise it you
    also see the refraction/tint vary across a feature (shallower water over the raised parts) — not just
    the protrusion appearing. `height_edge` softens the waterline where a feature breaks the surface.

### Still deferred
- **matcap (environment reflection)** — the outdoor sky-reflection hook (off indoors; `water_rendering.md`
  §6). Not built — gated on dropping in a matcap texture.
- **SSR** — screen-space reflections of literal shapes; out (polish, low return straight-down).

---

## Other visual knobs (reference)

The demo also tunes these; they live in their own config sections and are documented in their chapters.
Detailed rows can be added here as each gets a tuning pass:

- **Lighting** — `Ambient r/g/b`, `Light z` (overhead-ness of the light direction → affects glint
  geometry too), `Normal strength`, flashlight `max_range`/`intensity`/`angle_spread`. (Ray engine
  ch.08 / rendering ch.09.)
- **Smoke / gas** — `smoke_tint`, `smoke_max_alpha`, `smoke_render_gamma` (contrast), per-throw
  `explosion_smoke_noise` (cloud texture, also `N`/`Shift+N`). (Smoke ch.05 §6.)
- **Pressure overlay** — `pressure_scale` (debug viz). (Atmosphere ch.04.)
- **Grenade (demo)** — `blast_radius`, `blast_pressure`, `wall_damage`, `unit_damage`, `fuse_seconds`,
  `smoke_amount` — gameplay/test tuning, not a visual look.

---

**Status:** water section reflects the shipped v1 core + ambient (`a5d177a` + the ambient fix), the
phase-2 mood pass (caustics × light, foam, chromatic aberration, the `wave_scale`/`ambient_amp` wave-
size dials) **and the heightmap depth-adjust** (`height_scale`/`height_edge`, optional per-level floor
relief that adjusts the per-pixel depth all effects compute off — supersedes the earlier alpha-only
attenuation). Only the matcap hook + SSR remain deferred. Grows as the other systems get their tuning
passes.
