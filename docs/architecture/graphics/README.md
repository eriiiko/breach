# Breach — Graphics (the look layer)

The **visual-design** layer: *how the world looks* and the shader / rendering techniques that get
it there — distinct from `engine/`, which owns the simulation and the **rendering pipeline
architecture** (ch.08 ray engine, ch.09 rendering). Where engine chapters fix *what is true*,
graphics chapters fix *how it is shown*.

---

## Founding principle — the rendering budget is decoupled from the simulation budget

Graphics **never run during headless self-play** — the NN trains on the deterministic engine alone,
with no renderer present. So rendering cost lands *only when a human is watching*, and it does **not**
trade against parallel-sim throughput (the thing that sets gradient signal per GPU-hour).

Therefore the discipline that governs the engine **inverts here.** In sim we hoard every byte (one
rich field, derive the views, fixed-point determinism — see `engine/02` state-economy). In graphics
we **spend freely on the look**: a more expensive shader costs the training farm nothing, because it
isn't in the farm. This is the standing licence to do surfaces — water especially — *properly*, and
it is why visual richness and sim leanness are not in tension. (Erik, 2026-06-12.)

A corollary: this is the home for **look-and-feel tuning**. The dials a human eyeballs — water
see-through vs. opaque, ripple amplitude, god-ray strength, fire glow, tone-map — live in these
chapters and their `config.toml` sections, not scattered through the engine.

---

## The engine ↔ graphics split

- **`engine/08` ray engine** — the deposit-only DDA light/heat/vision march. Sim-coupled and
  deterministic (the fire heat-cast feeds temperature). Stays engine.
- **`engine/09` rendering** — the world-RT compose, the camera, the pipeline plumbing: *how pixels
  are assembled*. Stays engine.
- **`graphics/` (here)** — the *look*: surface shaders (water…), volumetric optics (smoke / gas),
  lighting mood & post (god-rays, tone-map, bloom), surface materials (normal / specular / emissive
  maps), decals & particles. The *how it should appear* and the shader math behind it.

**Rule of thumb:** if it changes simulation state or determinism, it is engine. If it only changes
pixels for a watching human, it is graphics.

---

## Convention — chapters by topic, not numbered

Unlike `engine/` and `mechanics/`, graphics chapters are **not** number-prefixed. The folder grows
by *adding a chapter when a visual system gets its look-pass*, with no fixed sequence — so **this
index defines the structure**, and a new chapter is simply a new file. (The hardcoded ordinals in the
other folders bake reading-order into filenames; here the index carries it instead.)

---

## Chapters

Legend: ✅ built · ⚠️ partial · 🧪 prototype · 📝 design-only

- [Water rendering](water_rendering.md) — 📝 *(in progress)* — the surface look: the ripple field +
  ambient sines + foam (today's `engine/07` §6), then the **unified Fresnel reflect/refract surface
  pass** (Schlick Fresnel on the perturbed ripple normal · GGX specular glints of the light sources ·
  normal-driven refraction × depth · Beer–Lambert depth-tint · phase-2 light-modulated caustics).
  Research + citations: `docs/water_rendering_research.md` (2026-06-12). **First inhabitant.**
- *(planned)* **Smoke & gas optics** — migrate from `engine/05` §6.1: per-channel Beer–Lambert
  absorption + additive scatter, per-gas colour, normal-mapped volumetric look, black-body emission.
- *(planned)* **Lighting mood & post** — god-rays, ACES tone-map, bloom — the visual-tuning side of
  the ray engine.
- *(planned)* **Surface materials** — normal / specular / emissive map usage; the destruction
  art-layer look (bare ↔ furnished ↔ destroyed).
- *(planned)* **Decals & particles** — blood, stains, sparks (the "Graphics chapter" the fluid doc
  already references for blood).

**Status:** folder + index established 2026-06-12. `water_rendering` is the first chapter, pending the
methods decision (the research recommends a 3-term v1 pass + a phase-2 caustics mood pass).
