# Breach --- Realistic Fire Rendering in a Top-Down Game

## Context

The goal is to develop a realistic, dynamically changing fire effect for
**Breach**, a top-down game.

Breach already has an unusually rich atmospheric simulation:

-   Pressure
-   Temperature
-   Density
-   Wind velocity
-   Wind data available every game frame, including substeps
-   An atmosphere based on the ideal gas law, (PV=nRT)

The fire simulation already tracks:

-   Fire intensity
-   Fire temperature
-   A custom quantity representing fire size / extent

The desired fire should:

-   Look realistic
-   Dynamically change size
-   React naturally to wind
-   Be compatible with the existing atmosphere simulation
-   Ideally allow fire, smoke, embers, temperature and airflow to
    interact visually and potentially physically

------------------------------------------------------------------------

# Main conclusion

There is no single universally "best" fire-rendering technique for every
game.

However, **for Breach specifically**, the strongest approach is:

> **A 2D GPU fluid simulation coupled to the existing atmospheric
> simulation, rendered using a pseudo-volumetric / parallax fire shader,
> with GPU particles for smoke and embers.**

This provides most of the visual benefits of expensive volumetric fire
without requiring a full 3D combustion simulation.

The most relevant modern example discussed was **Ignitement**, whose
2026 Unity technical breakdown uses a dynamic 2D fluid simulation,
pseudo-3D fire rendering and GPU particles.

------------------------------------------------------------------------

# 1. Major approaches to game fire

## 1.1 Animated sprites / flipbooks

The simplest traditional method.

A fire is represented by a sequence of pre-rendered images:

``` text
frame 1 → frame 2 → frame 3 → frame 4
```

### Advantages

-   Extremely cheap
-   Easy to implement
-   Predictable
-   Good for many tiny fires

### Problems for Breach

The fire does not actually know about the surrounding atmosphere.

Wind has to be faked by:

-   Rotating the sprite
-   Stretching it
-   Changing animation
-   Distorting the texture

This becomes increasingly artificial when the fire needs to interact
with dynamically changing wind.

### Verdict

Useful for tiny decorative flames, but not ideal as the primary Breach
fire system.

------------------------------------------------------------------------

# 2. Particle-based fire

Fire can instead be represented by many particles.

Each particle can have:

-   Position
-   Velocity
-   Lifetime
-   Size
-   Temperature
-   Opacity
-   Color

Wind can influence particle velocity.

For example:

\[ `\frac{d\mathbf{x}}{dt}`{=tex}=`\mathbf{v}`{=tex} \]

and approximately:

\[ `\frac{d\mathbf{v}}{dt}`{=tex} = `\mathbf{g}`{=tex} +
`\mathbf{F}`{=tex}*{wind} + `\mathbf{F}`{=tex}*{turbulence} \]

### Advantages

-   Flexible
-   Cheap with GPU particles
-   Easy to make particles respond to wind
-   Good for embers
-   Good for secondary effects

### Problems

Particles do not naturally create coherent fluid structures.

Real flames contain:

-   Vortices
-   Rising structures
-   Coherent turbulent regions
-   Continuous deformation

Particles tend to look like many independent objects unless heavily
engineered.

### Verdict

Excellent as a **secondary system**, especially for embers and sparks,
but not ideal as the main fire representation.

------------------------------------------------------------------------

# 3. Procedural fire shaders

Another approach is to create fire entirely or mostly in a shader.

Noise functions can generate animated flame patterns.

A conceptual formulation is:

\[ F(x,y,t)=N_1(x,y,t)N_2(x,y,t)M(x,y) \]

where:

-   (N_1,N_2) are noise fields
-   \(M\) is a flame-shaped mask

The result can then be mapped to a color gradient:

``` text
dark
↓
red
↓
orange
↓
yellow
↓
white
```

### Advantages

-   Extremely cheap
-   Very controllable
-   Easy to dynamically scale
-   Can look surprisingly realistic
-   Excellent renderer for a simulated fire field

### Problems

A shader alone does not naturally know how the fire should flow.

Wind interaction can be faked by:

-   UV distortion
-   Stretching
-   Noise movement
-   Directional deformation

But it is not actually fluid dynamics.

### Verdict

Very useful, especially as the **rendering layer on top of a
simulation**, rather than as the entire fire system.

------------------------------------------------------------------------

# 4. 2D fluid simulation

This is where the approach becomes much more interesting for Breach.

A 2D grid can contain fields such as:

\[ `\mathbf{u}`{=tex}(x,y) \]

Velocity

\[ T(x,y) \]

Temperature

\[ D(x,y) \]

Density / smoke

\[ R(x,y) \]

Reaction / flame intensity

Potentially also:

\[ F(x,y) \]

Fuel

A generic transported scalar can be represented as:

\[ `\frac{\partial q}{\partial t}`{=tex} +
`\mathbf{u}`{=tex}`\cdot`{=tex}`\nabla `{=tex}q =
D`\nabla`{=tex}\^2q+S-`\lambda `{=tex}q \]

where:

-   \(q\) is a quantity such as temperature or smoke
-   (`\mathbf{u}`{=tex}) is velocity
-   \(D\) is diffusion
-   \(S\) is a source
-   (`\lambda`{=tex}) is dissipation

The classic game-oriented foundation for this is Jos Stam's Stable
Fluids work.

### Advantages

-   Natural fluid motion
-   Excellent wind interaction
-   Natural turbulent structures
-   Dynamic fire size
-   Natural smoke transport
-   Can run efficiently on the GPU

### Verdict

Very strong candidate for Breach.

------------------------------------------------------------------------

# 5. Full 3D volumetric fire

The most physically expressive approach would be a 3D volume:

\[ D(x,y,z) \]

with quantities such as:

-   Temperature
-   Density
-   Reaction
-   Velocity
-   Smoke

The volume can then be rendered by ray marching.

Conceptually:

``` text
Camera
  |
  | ray
  v
+------------------+
|                  |
|    🔥🔥          |
|   🔥🔥🔥         |
|     🔥           |
|                  |
+------------------+
```

### Advantages

-   Genuine depth
-   Very realistic volume
-   Strong 3D turbulence
-   Best for cinematic / hero effects

### Problems

-   Expensive
-   Large memory requirements
-   More difficult to simulate
-   Probably unnecessary for a top-down game

Modern engines such as Unreal support both 2D and 3D gas simulations,
but 2D is much more appropriate for many real-time game effects while 3D
is more suitable for expensive hero effects.

### Verdict

Probably overkill for normal Breach fire.

------------------------------------------------------------------------

# 6. The recommended Breach architecture

The recommended design separates the **game simulation** from the
**visual fire simulation**.

## Layer A --- authoritative game simulation

Breach already has:

``` text
Atmosphere
 ├── Pressure
 ├── Temperature
 ├── Density
 └── Velocity

Fire
 ├── Intensity
 ├── Temperature
 └── Size
```

This remains authoritative.

The game simulation determines how much fire actually exists and what
its physical state is.

------------------------------------------------------------------------

## Layer B --- fire VFX simulation

Create a local 2D fire grid around each fire or group of nearby fires.

Potential fields:

``` text
velocity       vec2
temperature    float
reaction       float
smoke          float
fuel           float
flameHeight    float
```

The fire VFX simulation receives information from the atmosphere and
from the authoritative fire simulation.

### Atmosphere → fire VFX

-   Wind velocity
-   Ambient temperature
-   Density
-   Potentially pressure

### Game fire → fire VFX

-   Intensity
-   Fire temperature
-   Fire size
-   Source location

The VFX system then generates:

-   Flame structure
-   Smoke
-   Embers
-   Turbulence
-   Visual glow

------------------------------------------------------------------------

# 7. Wind interaction

One of the biggest advantages of Breach's existing atmosphere simulation
is that the fire does not have to invent wind.

The fire can directly use:

\[ `\mathbf{u}`{=tex}\_{wind} \]

as part of its velocity.

A useful conceptual decomposition is:

\[ `\mathbf{u}`{=tex}*{fire} = `\mathbf{u}`{=tex}*{wind} +
`\mathbf{u}`{=tex}*{buoyancy} + `\mathbf{u}`{=tex}*{turbulence} \]

This allows fire to:

-   Rise when there is little wind
-   Lean strongly in high wind
-   Change direction immediately when the wind changes
-   Transport smoke and embers consistently

Because Breach already calculates wind during substeps, the fire VFX can
potentially be updated using the same temporal information instead of
only receiving a once-per-frame approximation.

------------------------------------------------------------------------

# 8. Buoyancy

Hot gas is less dense than cool surrounding gas.

A simple visual buoyancy force can be based on:

\[ `\mathbf{F}`{=tex}*b = `\beta`{=tex}(T-T*{ambient})`\mathbf{g}`{=tex}
\]

This makes hotter regions rise.

The important idea is that buoyancy should interact with atmospheric
wind rather than replace it.

Conceptually:

``` text
No wind:

     ↑
    🔥
   🔥🔥
  🔥🔥🔥


Strong wind:

>>>>>>>>>>>>>>>>>

             🔥🔥🔥
           🔥🔥🔥🔥
```

------------------------------------------------------------------------

# 9. Vorticity confinement

Low-resolution fluid simulations can become too smooth.

Real fire contains lots of small swirling structures.

Vorticity is:

\[ `\omega`{=tex}=`\nabla`{=tex}`\times`{=tex}`\mathbf{u}`{=tex} \]

Vorticity confinement can be used to reinforce rotational structures
that would otherwise disappear due to numerical diffusion.

The visual effect is approximately:

### Without strong vorticity

``` text
     ↑
    🔥
   🔥🔥
  🔥🔥🔥
```

### With vorticity

``` text
      ↗~~
    ↗🔥~~
   ~~🔥🔥↘
  🔥🔥~~
```

This is particularly useful because it can make a relatively
low-resolution simulation appear much more detailed.

------------------------------------------------------------------------

# 10. Do not try to simulate "fire" in the VFX layer

A key conceptual recommendation:

> The VFX simulation does not need to reproduce combustion physics
> exactly.

The existing Breach fire model should remain the authoritative physical
model.

The VFX system should simulate the **visual consequences** of that fire.

Useful visual fields include:

## Reaction

\[ R(x,y) \]

Represents where combustion is visually occurring.

## Temperature

\[ T(x,y) \]

Controls:

-   Color
-   Brightness
-   Buoyancy
-   Smoke generation

## Smoke

\[ D(x,y) \]

Controls visible smoke density.

## Velocity

\[ `\mathbf{u}`{=tex}(x,y) \]

Controls advection.

## Fuel

\[ F(x,y) \]

Potentially useful if a more detailed fire propagation model is wanted.

------------------------------------------------------------------------

# 11. Pseudo-volumetric rendering

This is probably the most important rendering technique for Breach.

A 2D simulation does not have to look flat.

Instead, the reaction field can be interpreted as a kind of height
field:

\[ H(x,y) \]

The renderer can then construct multiple visual layers:

``` text
Layer 3     ~~~
Layer 2    ~~~~~
Layer 1   ~~~~~~~
Layer 0  ████████
```

Each layer can use slightly different noise, distortion and opacity.

The result gives the appearance of a small 3D volume even though the
simulation itself remains 2D.

A particularly relevant modern example is the 2026 Unity technical
breakdown of **Ignitement**, which uses a 2D fluid simulation and a
pseudo-3D/parallax-style fire renderer.

------------------------------------------------------------------------

# 12. Dynamic fire size

This architecture is especially well suited to the user's requirement
that fire dynamically change size.

The existing fire simulation can control the source strength:

\[ S\_{fire}=f(I,T,size) \]

A small source might produce:

``` text
  🔥
```

A larger source:

``` text
   🔥
  🔥🔥🔥
   🔥🔥
```

A very large source:

``` text
       🔥
     🔥🔥🔥
   🔥🔥🔥🔥🔥
  🔥🔥🔥🔥🔥🔥
 🔥🔥🔥🔥🔥🔥🔥
```

The important difference from scaling a sprite is that increasing the
source causes the simulated flow structure to become larger.

Thus:

> Fire size is a physical source property, not merely a rendering scale.

This allows the fire to grow, shrink, flicker and respond to wind
naturally.

------------------------------------------------------------------------

# 13. Smoke

Smoke can be generated from the fire fields.

A simple conceptual source term is:

\[ S\_{smoke} = k_sR f(T) \]

Smoke can then be transported by the same velocity field as the fire.

This means wind naturally moves both flame and smoke.

Conceptually:

``` text
              smoke
               ↑
             ~~~~~
            ~~~~~~~
             🔥🔥
            🔥🔥🔥
           🔥🔥🔥🔥
```

When wind changes:

``` text
>>>>>>>>>>>>>>>>>

             ~~~~~~~
          ~~~~~~~~~~
       ~~~~~~~~~
      🔥🔥🔥
```

No separate smoke animation is required.

------------------------------------------------------------------------

# 14. Embers

GPU particles are ideal for embers.

Instead of giving each ember an entirely independent velocity, it can
sample the fire velocity field:

\[ `\frac{d\mathbf{x}_{ember}}{dt}`{=tex} = `\mathbf{u}`{=tex}*{fire} +
`\mathbf{v}`{=tex}*{random} + `\mathbf{v}`{=tex}\_{buoyancy} \]

This makes embers naturally follow the same airflow as the fire.

The 2026 Ignitement example uses GPU-driven particles that interact with
the simulated velocity field in a similar manner.

------------------------------------------------------------------------

# 15. Local simulation domains

A major optimization is to avoid simulating the entire spaceship at high
resolution.

Instead, create a local simulation around each fire or cluster of nearby
fires.

For example:

``` text
Whole ship

+--------------------------------------+
|                                      |
|       +------------------+           |
|       |   FIRE DOMAIN    |           |
|       |                  |           |
|       |     🔥🔥🔥       |           |
|       |    🔥🔥🔥🔥      |           |
|       |                  |           |
|       +------------------+           |
|                                      |
+--------------------------------------+
```

Possible resolutions:

-   Small fire: roughly 128²
-   Larger fire: roughly 256²
-   Very large effects: higher resolution

Nearby fires can potentially share one simulation domain.

This provides much better scalability than a single huge fire grid
covering the entire map.

------------------------------------------------------------------------

# 16. Temperature-driven color

Instead of simply assigning "orange" to fire, use temperature as a major
input to color.

Conceptually:

``` text
very hot
   ↓
white
   ↓
yellow
   ↓
orange
   ↓
red
   ↓
dark
```

A physically inspired black-body color mapping can be used as a basis,
followed by artistic adjustment.

This makes hotter regions naturally brighter and more yellow/white,
while cooler regions become darker and redder.

Unreal Niagara Fluids includes temperature-based black-body rendering as
one example of this approach.

------------------------------------------------------------------------

# 17. Recommended hierarchy

Not every flame needs the same expensive simulation.

A scalable system could use:

## Tiny flame

``` text
Sprite / procedural shader
```

## Small fire

``` text
2D fluid simulation
+
pseudo-volume
```

## Large fire

``` text
Higher-resolution 2D fluid simulation
+
pseudo-volume
+
particles
```

## Exceptional cinematic event

``` text
3D volume / expensive simulation
```

This follows the general modern game-VFX principle of reserving
expensive 3D simulation for effects where it is actually necessary.

------------------------------------------------------------------------

# 18. Recommended Breach Fire v1

A practical first implementation would contain:

## Simulation grid

``` text
velocity       vec2
temperature    float
reaction       float
smoke          float
fuel           float
```

Potentially later:

``` text
flameHeight    float
```

## Velocity

Use:

\[ `\mathbf{u}`{=tex} = `\mathbf{u}`{=tex}*{atmosphere} +
`\mathbf{u}`{=tex}*{buoyancy} + `\mathbf{u}`{=tex}\_{vorticity} \]

with atmospheric wind as the dominant external influence.

## Fire source

Use the existing:

-   Intensity
-   Temperature
-   Size

to control the source term.

## Rendering

Use:

``` text
reaction field
      ↓
flame mask
      ↓
pseudo-volume / parallax
      ↓
temperature-based color
      ↓
emissive output
```

Then add:

``` text
smoke
+
embers
+
small amount of glow
```

------------------------------------------------------------------------

# 19. Why this approach fits Breach particularly well

The major advantage is that Breach already has information that ordinary
games normally have to fake.

Breach already knows:

-   How hot the atmosphere is
-   How dense it is
-   What the pressure is
-   Which way the wind is moving
-   How the wind changes over substeps
-   How intense the fire is
-   How hot the fire is
-   Approximately how large the fire is

Therefore the renderer does not have to invent these properties.

The ideal architecture is:

``` text
                  ATMOSPHERE
                       │
              wind / temp / density
                       │
                       ▼
             +-------------------+
             |   FIRE VFX GRID   |
             |                   |
             | velocity          |
             | reaction          |
             | temperature       |
             | smoke             |
             | flame height      |
             +---------+---------+
                       |
          +------------+------------+
          |            |            |
          ▼            ▼            ▼
        FIRE         SMOKE        EMBERS
          |            |            |
          +------------+------------+
                       |
                       ▼
                  FIRE SHADER
                       |
              pseudo-volume/parallax
                       |
                       ▼
                    SCREEN
```

------------------------------------------------------------------------

# 20. Overall ranking for Breach

  ----------------------------------------------------------------------------------
  Technique             Visual          Wind Dynamic size         Cost   Suitability
                       quality   interaction                           
  --------------- ------------ ------------- ------------ ------------ -------------
  Animated sprite           ★★             ★           ★★        ★★★★★           Low

  Particles                ★★★           ★★★         ★★★★         ★★★★        Medium

  Procedural              ★★★★            ★★        ★★★★★        ★★★★★        Medium
  shader                                                               

  2D fluid                ★★★★         ★★★★★        ★★★★★         ★★★★          High

  2D fluid +             ★★★★★         ★★★★★        ★★★★★         ★★★★      **Best**
  pseudo-volume                                                        

  Full 3D fluid +       ★★★★★+         ★★★★★        ★★★★★            ★      Probably
  volume                                                                    overkill
  ----------------------------------------------------------------------------------

------------------------------------------------------------------------

# 21. Useful references mentioned

## Unity --- Ignitement technical breakdown

Highly relevant because it demonstrates a modern top-down fire system
using:

-   2D fluid simulation
-   GPU simulation
-   Pseudo-3D fire rendering
-   GPU particles
-   Velocity-field-driven embers

https://unity.com/cn/blog/real-time-fluid-simulation-fire-vfx-ignitement-breakdown

## Unreal Engine --- Niagara Fluids

Relevant for understanding modern 2D/3D gas simulation, temperature,
velocity, buoyancy, turbulence and rendering.

https://dev.epicgames.com/documentation/en-us/unreal-engine/niagara-fluids-in-unreal-engine

## Unreal Engine --- Niagara Fluids Reference

Useful for the individual simulation and rendering parameters.

https://dev.epicgames.com/documentation/unreal-engine/niagara-fluids-reference-in-unreal-engine

## Jos Stam --- Real-Time Fluid Dynamics for Games

Classic foundation for stable real-time fluid simulation.

https://www.josstam.com/publications

## Keijiro --- Stable Fluids

A modern Unity implementation useful as a practical reference for GPU
stable fluids.

https://github.com/keijiro/StableFluids

## GPU Stable Fluids

Another implementation/reference for GPU-based 2D/3D stable fluids.

https://github.com/matthiasbroske/GPUStableFluids

------------------------------------------------------------------------

# Final recommendation

For Breach, the best target is not "the most physically accurate fire
simulation possible."

Instead:

> **Use the existing Breach atmosphere and fire model as the physical
> authority, then build a localized 2D GPU fluid simulation that
> represents the visual consequences of fire. Drive it with the actual
> atmospheric wind and temperature, add buoyancy and vorticity
> confinement, and render the resulting reaction field using a
> pseudo-volumetric/parallax shader. Use the same velocity field for
> smoke and GPU embers.**

This gives a very strong combination of:

-   Realistic appearance
-   Dynamic size
-   Genuine wind interaction
-   Coherent turbulent motion
-   Natural smoke movement
-   Consistent ember movement
-   Good performance
-   Compatibility with the existing Breach physics
-   A clear path toward more advanced fire-atmosphere coupling later

The especially interesting long-term possibility is a feedback loop:

``` text
fire
 ↓
heat
 ↓
atmosphere
 ↓
density / velocity
 ↓
wind
 ↓
fire
```

Even if that coupling is only partially physical, it could make Breach's
fire effects unusually convincing because the fire is participating in
the same atmospheric world as the rest of the game rather than being a
separate animated effect.
