# Atmosphere solver analysis and implementation plan

Prepared for patching the current 2D atmosphere solver.

## Executive summary

Your current solver is numerically fragile because it evolves the **wave part** and the **diffusion part** as separate operators on the same scalar field `atmosphere`, with diffusion applied afterward as a separate pass. In the current code:

- the wave solver advances `wave_v` and then advances `atmosphere` from `wave_v`
- the diffusion solver later advances `atmosphere` again
- both use the same discrete Laplacian stencil
- the diffusion step is chosen near the explicit 2D stability limit (`0.24 / D`)

That combination is enough to destabilize high-frequency grid modes even when:
- the wave solver alone is stable
- the diffusion solver alone is stable

The strongest analytical result is:

> In the current split scheme, if the diffusion factor for some Fourier mode becomes negative, i.e.
>
> \[
> 1-\mu \sigma < 0,
> \]
>
> then the coupled wave+diffusion update becomes unstable for that mode in the undamped case, and remains fragile even with damping.

Here:
- \(\mu = D \,\Delta t / h^2\)
- \(\sigma \in [0,8]\) is the 2D discrete Laplacian mode magnitude
- the highest-frequency mode has \(\sigma_{\max}=8\)

With your current explicit diffusion cap,
\[
\mu \le 0.24,
\]
the worst mode gets
\[
1-\mu \sigma_{\max} = 1 - 0.24 \cdot 8 = -0.92,
\]
so the sign flips every diffusion substep for those high-frequency components.

That is the main numerical reason for the blow-up.

---

## 1. What the code currently implements

## 1.1 Diffusion

`atmo_diffusion.cpp` computes the five-point Laplacian into a temporary buffer and then applies an explicit diffusion step:

```cpp
lap[i] = p_up + p_down + p_left + p_right - 4.0f * p;
atmosphere[i] += coeff * lap[i];
```

with

```cpp
const float dt = 0.24f / std::max(d_atm, 0.01f);
const float coeff = d_atm * actual_dt;
```

So the scheme is

\[
u^{n+1}_{i,j}
=
u^n_{i,j}
+
\mu
\left(
u^n_{i+1,j}
+u^n_{i-1,j}
+u^n_{i,j+1}
+u^n_{i,j-1}
-4u^n_{i,j}
\right),
\]

where

\[
\mu = \frac{D\,\Delta t}{h^2}.
\]

Assuming one tile is one grid spacing, you are effectively using \(h=1\).

---

## 1.2 Wave solver

`wave_solver.cpp` evolves a pair of fields:

- `atmosphere` = scalar pressure-like field \(u\)
- `wave_v` = velocity-like auxiliary field \(v\)

The code does:

\[
v^{n+1} = v^n + \Delta t \left(c^2 \Delta_h u^n - \gamma v^n\right)
\]

then

\[
u^{n+1} = u^n + \Delta t\, v^{n+1}.
\]

That is a first-order form of the damped wave equation:

\[
u_t = v,
\qquad
v_t = c^2 \Delta u - \gamma v.
\]

This is **not** the same as plain forward Euler on the second-order wave equation. It is a kick-drift style update and is better than naive forward Euler for wave-like motion.

---

## 1.3 Coupling order in the engine

`physics_engine.cpp` currently does:

1. wave source feed into `atmosphere`
2. wave update of `wave_v`
3. wave update of `atmosphere`
4. boundary conditions
5. wind computation
6. smoke advection
7. **after all wave substeps**, diffusion step on `atmosphere`

So diffusion is not interleaved with the wave update. It is applied as a separate operator at the end of the tick.

That split ordering is the central issue.

---

## 2. The 2D discrete Laplacian mode factor

For von Neumann / Fourier analysis, take a discrete mode

\[
u_{i,j}^n = \hat u^n e^{i(i\theta_x + j\theta_y)}.
\]

For the 5-point Laplacian,

\[
\Delta_h e^{i(i\theta_x + j\theta_y)}
=
\lambda(\theta_x,\theta_y)
e^{i(i\theta_x + j\theta_y)},
\]

with

\[
\lambda(\theta_x,\theta_y)
=
2\cos\theta_x + 2\cos\theta_y - 4.
\]

It is convenient to write

\[
\lambda = -\sigma,
\qquad
\sigma = 4\left(\sin^2\frac{\theta_x}{2}+\sin^2\frac{\theta_y}{2}\right),
\qquad
0 \le \sigma \le 8.
\]

Important special case:
- highest-frequency checkerboard mode in 2D gives \(\sigma_{\max}=8\)

---

## 3. Pure diffusion stability factor

For one explicit diffusion step on one Fourier mode:

\[
\hat u^{n+1}
=
(1-\mu\sigma)\hat u^n.
\]

So the scalar amplification factor is

\[
g_{\text{diff}} = 1-\mu\sigma.
\]

### Classical explicit stability
To avoid growth in magnitude, we need

\[
|1-\mu\sigma| \le 1
\quad \text{for all } \sigma \in [0,8],
\]

which gives

\[
0 \le \mu \le \frac14.
\]

That is the standard 2D explicit diffusion CFL condition.

### Stronger monotonicity / no sign flip condition
To avoid alternating-sign high-frequency behavior, require

\[
1-\mu\sigma \ge 0
\quad \text{for all } \sigma \in [0,8],
\]

which gives

\[
0 \le \mu \le \frac18.
\]

This condition is much more relevant here than the looser \(\mu \le 1/4\), because the split wave+diffusion coupling reacts very badly once the diffusion factor becomes negative.

### What your current value implies
Your explicit cap is effectively

\[
\mu \le 0.24.
\]

So for \(\sigma_{\max}=8\),

\[
g_{\text{diff,max}} = 1 - 0.24\cdot 8 = -0.92.
\]

So the highest modes are being strongly damped **but also flipped in sign every diffusion substep**.

Diffusion alone can tolerate that.
The coupled scheme does not.

---

## 4. Pure wave update matrix and stability

Define

\[
\alpha = c^2 \sigma \Delta t^2,
\qquad
\beta = 1 - \gamma \Delta t.
\]

Since \(\lambda=-\sigma\), one wave substep becomes:

\[
v^{n+1} = \beta v^n - \frac{\alpha}{\Delta t} u^n,
\]

\[
u^{n+1} = u^n + \Delta t\, v^{n+1}.
\]

So in matrix form,

\[
\begin{bmatrix}
u^{n+1} \\
v^{n+1}
\end{bmatrix}
=
A_{\text{wave}}
\begin{bmatrix}
u^n \\
v^n
\end{bmatrix},
\qquad
A_{\text{wave}}
=
\begin{bmatrix}
1-\alpha & \beta \Delta t \\
-\alpha/\Delta t & \beta
\end{bmatrix}.
\]

The eigenvalues are the roots of

\[
r^2 - \operatorname{tr}(A_{\text{wave}}) r + \det(A_{\text{wave}})=0,
\]

with

\[
\operatorname{tr}(A_{\text{wave}}) = 1-\alpha+\beta,
\qquad
\det(A_{\text{wave}})=\beta.
\]

In the undamped case \(\beta=1\), the standard stability condition reduces to

\[
0 \le \alpha \le 4.
\]

Since \(\sigma_{\max}=8\),

\[
c^2 (8)\Delta t^2 \le 4
\quad \Longrightarrow \quad
\Delta t \le \frac{1}{\sqrt{2}\,c}.
\]

Your code uses approximately

\[
\Delta t_{\text{wave}} \approx \frac{0.65}{c},
\]

and

\[
\frac{1}{\sqrt{2}} \approx 0.707,
\]

so the wave solver by itself is inside the usual stability region.

That matches your observation that the wave solver alone behaves fine.

---

## 5. Current split coupled scheme: exact mode matrix

Now analyze the actual split structure used by the engine:

1. wave substep on \((u,v)\)
2. diffusion substep on \(u\) only

For one Fourier mode, the diffusion matrix is

\[
A_{\text{diff}}
=
\begin{bmatrix}
1-\mu\sigma & 0 \\
0 & 1
\end{bmatrix}.
\]

So the combined one-step split matrix is

\[
A_{\text{split}}
=
A_{\text{diff}} A_{\text{wave}}
=
\begin{bmatrix}
(1-\mu\sigma)(1-\alpha) & (1-\mu\sigma)\beta \Delta t \\
-\alpha/\Delta t & \beta
\end{bmatrix}.
\]

Its trace and determinant are

\[
\operatorname{tr}(A_{\text{split}})
=
\beta + (1-\mu\sigma)(1-\alpha),
\]

\[
\det(A_{\text{split}})
=
\beta(1-\mu\sigma).
\]

The eigenvalues are therefore

\[
r_{\pm}
=
\frac{
\operatorname{tr}(A_{\text{split}})
\pm
\sqrt{
\operatorname{tr}(A_{\text{split}})^2
-4\det(A_{\text{split}})
}
}{2}.
\]

---

## 6. Why the current split scheme blows up

For a 2x2 real matrix, the eigenvalues lie inside the unit disk if the Jury / Schur conditions hold:

\[
1 + \operatorname{tr}(A) + \det(A) > 0,
\]

\[
1 - \operatorname{tr}(A) + \det(A) > 0,
\]

\[
1 - \det(A) > 0.
\]

For the **current split matrix**, the second condition becomes

\[
1 - \operatorname{tr}(A_{\text{split}}) + \det(A_{\text{split}})
=
\alpha(1-\mu\sigma) + \mu\sigma(1-\beta).
\]

### Undamped case
If \(\beta=1\), this simplifies to

\[
1 - \operatorname{tr}(A_{\text{split}}) + \det(A_{\text{split}})
=
\alpha(1-\mu\sigma).
\]

Since \(\alpha > 0\) for any nonzero wave mode, stability requires

\[
1-\mu\sigma > 0.
\]

That is exactly the **no sign flip** condition.

So for the split scheme:

> the moment the diffusion factor becomes negative for some mode, the coupled wave+diffusion scheme becomes unstable for that mode.

This is the key analytical result.

### With damping
With \(\beta < 1\), the condition becomes

\[
\alpha(1-\mu\sigma) + \mu\sigma(1-\beta) > 0.
\]

Damping helps, but it does **not** fundamentally remove the fragility if \(\mu\sigma > 1\). It only softens it.

### Why your current parameters are bad
With \(\mu=0.24\) and \(\sigma_{\max}=8\),

\[
\mu\sigma_{\max} = 1.92 > 1.
\]

So the highest-frequency modes necessarily enter the unstable region of the split scheme.

That is why the combined solver blows up even though each part separately behaves.

---

## 7. Unified explicit update: much better than the current split scheme

If you want a unified update where diffusion and wave are applied **within the same substep**, the simplest consistent explicit form is:

\[
v^{n+1} = \beta v^n - \frac{\alpha}{\Delta t} u^n,
\]

\[
u^{n+1}
=
u^n + \Delta t\, v^{n+1} - \mu\sigma\,u^n.
\]

Equivalently, in physical space:

\[
v^{n+1} = v^n + \Delta t \left(c^2 \Delta_h u^n - \gamma v^n\right),
\]

\[
u^{n+1} = u^n + \Delta t\, v^{n+1} + D \Delta t\, \Delta_h u^n.
\]

Matrix form:

\[
A_{\text{unified-explicit}}
=
\begin{bmatrix}
1-\alpha-\mu\sigma & \beta \Delta t \\
-\alpha/\Delta t & \beta
\end{bmatrix}.
\]

Trace and determinant:

\[
\operatorname{tr}(A_{\text{unified-explicit}})
=
1+\beta-\alpha-\mu\sigma,
\]

\[
\det(A_{\text{unified-explicit}})
=
\beta(1-\mu\sigma).
\]

### Important difference from the split scheme
For this unified explicit form, the critical Jury condition becomes

\[
1-\operatorname{tr}(A)+\det(A)
=
\alpha + \mu\sigma(1-\beta),
\]

which is always positive for \(\alpha>0\).

So the unified explicit update removes the exact instability mechanism that kills the split scheme.

### Conservative stability bound for unified explicit
The other nontrivial Jury condition is

\[
1+\operatorname{tr}(A)+\det(A) > 0,
\]

which becomes

\[
2 + 2\beta - \alpha - \mu\sigma(1+\beta) > 0.
\]

In the undamped case \(\beta=1\), this is

\[
\alpha + 2\mu\sigma < 4.
\]

For the worst 2D mode \(\sigma=8\),

\[
8c^2\Delta t^2 + 16D\Delta t < 4.
\]

Equivalently,

\[
2c^2\Delta t^2 + 4D\Delta t < 1.
\]

This is the practical explicit combined-CFL condition for the unified explicit scheme.

### Practical explicit recommendation
If you keep everything explicit, use a very conservative step such as

\[
\Delta t
\le
\min\!\left(\frac{0.5}{c}, \frac{0.10}{D}\right)
\]

and also enforce

\[
2c^2\Delta t^2 + 4D\Delta t \le 1.
\]

That should be robust in practice.

---

## 8. Best recommendation: unified IMEX scheme

If you want something that is much more safely stable, the best patch is:

- keep the wave part explicit
- make the diffusion part implicit
- do both inside one common substep

This is often called an IMEX scheme (implicit-explicit).

### Proposed unified IMEX update

Step 1: wave kick
\[
v^{n+1}
=
v^n + \Delta t\left(c^2 \Delta_h u^n - \gamma v^n\right).
\]

Step 2: pressure update target before diffusion
\[
u^\star = u^n + \Delta t\, v^{n+1}.
\]

Step 3: implicit diffusion solve
\[
u^{n+1} - D\Delta t\, \Delta_h u^{n+1} = u^\star.
\]

That means solving

\[
(I - D\Delta t\,\Delta_h)u^{n+1} = u^\star.
\]

### Why this is the best route
For one Fourier mode, the implicit diffusion factor is

\[
g_{\text{imp-diff}} = \frac{1}{1+\mu\sigma}.
\]

This is always positive and always bounded by 1 in magnitude for any \(\mu \ge 0\).

So there is:
- no sign flip
- no diffusion-driven high-frequency oscillation
- no need for a diffusion CFL limit

The only remaining time-step restriction is basically the wave CFL.

### IMEX mode matrix
For one Fourier mode, the IMEX update gives

\[
A_{\text{IMEX}}
=
\begin{bmatrix}
\dfrac{1-\alpha}{1+\mu\sigma}
&
\dfrac{\beta \Delta t}{1+\mu\sigma}
\\[1.2ex]
-\alpha/\Delta t
&
\beta
\end{bmatrix}.
\]

with

\[
\operatorname{tr}(A_{\text{IMEX}})
=
\beta + \frac{1-\alpha}{1+\mu\sigma},
\qquad
\det(A_{\text{IMEX}})
=
\frac{\beta}{1+\mu\sigma}.
\]

This is much better behaved than the current split scheme.

---

## 9. Recommendation on field splitting: split later, not first

Question: should the fields be split?

### My recommendation
For the **first patch**, do **not** split the field into two separate gameplay quantities yet.

Instead:
- keep one main `atmosphere` field
- keep `wave_v` as the auxiliary wave velocity field
- unify the update
- make diffusion implicit if you want the most stable implementation

That gives the smallest code change with the highest chance of fixing the blow-up.

### Why not split immediately
Because your current gameplay effect that you like — rapid pressure emptying through a hull breach under Dirichlet/open-like boundaries — is already tied to the current single pressure field.

If we split too early into:
- a slow mean field
- a zero-mean acoustic field

then we risk accidentally removing the strong decompression effect that currently feels good in the game.

### But keep this in mind
Longer-term, a split can become very useful:

- `p_base`: slow mean/background pressure or air amount
- `p_wave`: zero-mean acoustic shock component
- total pressure seen by smoke/wind: `p_total = p_base + p_wave`

That gives you much better artistic and physical control:
- grenades excite `p_wave`
- decompression / venting acts mainly on `p_base`
- shockwave and long-term leak can be tuned separately

That is probably the best long-term architecture.
It is just not the first patch I would make.

---

## 10. Hull breach boundary condition: yes, smooth it

You suggested ramping the breach boundary from local mean pressure down toward zero instead of enforcing hard zero instantly.

That is a very good idea.

### Why hard Dirichlet can be rough
Setting breach cells instantly to zero every substep is numerically sharp and excites high-frequency modes.
It also makes the wave solver do a lot of work at a single-cell discontinuity.

### Better options

### Option A: ramped Dirichlet / relaxation to vacuum
For each breach cell:

\[
u^{n+1}_{\text{breach}}
=
(1-\eta) u^\star_{\text{breach}} + \eta\,p_{\text{ext}}(t),
\]

where:
- \(p_{\text{ext}}(t)\) ramps from local pressure toward 0
- \(\eta = \Delta t/\tau\), clamped to \([0,1]\)

This is easy to implement and much smoother.

### Option B: sink term near breach
Add a leakage sink in breach cells:

\[
u_t \;\mathrel{+}= -\kappa(x)\,\bigl(u-p_{\text{ext}}\bigr).
\]

This acts like a sponge/open boundary.
It is often better than hard clamping.

### Option C: short sponge layer
Instead of one breach cell, use a small region near the hole where you:
- damp `wave_v`
- relax `u` toward vacuum

That strongly reduces reflections and grid-scale ringing.

### Recommendation
For the first patch, use **Option A** or **Option B**.
Both are much gentler than hard zeroing.

---



## 10A. Additional boundary-condition ideas for hull breach and atmosphere draining

This section expands the breach discussion with concrete boundary-condition choices, because the gameplay question is not only numerical stability but also **what physical effect should be represented by the boundary**.

### Key modeling point
A hull breach is not well described by diffusion alone. It includes at least three distinct effects:

1. **Acoustic response**: pressure waves and shock propagation
2. **Bulk outflow / venting**: gas leaves the ship through the opening
3. **Mixing / smoothing**: smaller-scale equilibration inside the ship

So it is completely reasonable to represent a breach with an additional boundary or sink mechanism. This is **not** a fake special case if it is formulated as a general local flux law.

### BC option 1: hard Dirichlet vacuum

At breach cells:

\[
u = p_{\text{ext}} = 0.
\]

Advantages:
- simplest
- very strong decompression effect
- easy to implement

Disadvantages:
- numerically sharp
- strongly excites high-frequency modes
- tends to cause ringing / checkerboard activity / reflections unless the time step is conservative
- can feel too abrupt or too "digital"

Recommendation: avoid as the default long-term breach BC, though it can still be useful as a debugging reference.

### BC option 2: ramped Dirichlet vacuum

Instead of setting the breach instantly to zero, use a time-dependent external pressure:

\[
p_{\text{ext}}(t) : p_{\text{local-start}} \rightarrow 0
\]

and enforce

\[
u_{\text{breach}}^{n+1} = p_{\text{ext}}(t_{n+1}).
\]

A practical version is to initialize the boundary target from the nearby mean interior pressure when the hull breach opens, then relax that target toward zero over a chosen timescale \(\tau_{\text{open}}\).

Advantages:
- preserves the intuitive Dirichlet interpretation
- smoother than hard zeroing
- gives direct artistic control over how abruptly decompression starts

Disadvantages:
- still a prescribed boundary value rather than a true flux law
- can still create reflections if imposed too hard

Recommendation: a good first experiment if you want to stay close to your current code.

### BC option 3: relaxation-to-vacuum boundary (recommended practical first choice)

At breach cells or breach-adjacent cells, use a Robin / sink-like update:

\[
\frac{\partial p}{\partial t} = -k_{\text{breach}}(p - p_{\text{ext}}),
\qquad p_{\text{ext}} = 0.
\]

Explicit discrete form:

\[
p^{n+1} = p^n - \Delta t\,k_{\text{breach}}(p^n - p_{\text{ext}}).
\]

Or equivalently as a blend after the main pressure update:

\[
p^{n+1} \leftarrow (1-\eta)p^{n+1} + \eta p_{\text{ext}},
\qquad \eta = \operatorname{clamp}(\Delta t\,k_{\text{breach}},0,1).
\]

Advantages:
- much smoother than hard Dirichlet
- easy to tune
- directly controls drain speed
- still generates gradients that drive wind and smoke
- easy to combine with unified explicit or IMEX integration

Disadvantages:
- the drain rate is a model parameter, not derived from a full compressible CFD model

Recommendation: this is the best **first practical boundary model** for the 1-field branch.

### BC option 4: flux boundary at breach faces (best long-term physical abstraction)

Instead of forcing cell values, define an outward flux through any face connecting an interior cell to vacuum.

For example:

\[
F_{\text{breach}} = g_{\text{breach}}\,\max(p_{\text{inside}} - p_{\text{ext}}, 0).
\]

Then subtract this flux from the interior cell during the update.

Advantages:
- more physically interpretable than clamping a cell value
- naturally scales with opening size (more breach faces => more venting)
- preserves the idea that gas leaves through openings, not because arbitrary cells are overwritten
- works well with your design principle that behavior should follow from simple local rules

Disadvantages:
- slightly more implementation work
- needs careful bookkeeping at obstacle / vacuum interfaces

Recommendation: probably the best long-term solution once the unified solver is stable.

### BC option 5: sponge layer / absorbing vent region

Use a small region around the breach where you:
- relax pressure toward vacuum
- damp `wave_v`

For example:

\[
\frac{\partial p}{\partial t} \mathrel{+}= -k_p(x)(p-p_{\text{ext}}),
\qquad
\frac{\partial v}{\partial t} \mathrel{+}= -k_v(x)v.
\]

Advantages:
- reduces reflections strongly
- helps the breach behave like an open boundary instead of a rigid pressure pin
- good for wave quality near the opening

Disadvantages:
- slightly less local / more "engineered"
- requires a chosen sponge width

Recommendation: very useful if the breach still rings too much after moving away from hard Dirichlet.

### Which boundary idea should be tried first?

For the **1-field unified branch**:

1. First try **relaxation-to-vacuum** at breach cells
2. If needed, extend it into a **short sponge layer** near the opening
3. Later, if you want a more physical abstraction, replace the cell-based sink with a **face flux law**

That gives a natural progression:
- easy implementation first
- physically cleaner formulation later

### Important design conclusion

If you add a breach sink / flux law, that should not be viewed as violating the design principle of "simple physics-like rules".

A local law of the form

\[
\text{outflow} \propto \max(p-p_{\text{ext}}, 0)
\]

is exactly the kind of simple local rule that belongs in this kind of simulation game.
It is more honest than forcing diffusion to represent atmosphere blowout when diffusion is not the dominant physics of venting.


## 11. Grenade source: smooth it slightly

A grenade source injected on a single tile is a near-grid-scale impulse. That strongly excites the highest-frequency modes, i.e. exactly the modes that are hardest numerically.

Recommended change:
- distribute grenade pressure over a small symmetric stencil, for example 3x3
- keep the total injected pressure equal to the original amount

For example:

\[
\frac{1}{16}
\begin{bmatrix}
1 & 2 & 1 \\
2 & 4 & 2 \\
1 & 2 & 1
\end{bmatrix}
\]

times the desired source amount.

That makes the solver less sensitive without removing the sharp blast feeling.

---

## 12. Concrete implementation plan

## Plan A — fastest robust patch
Use this if you want the smallest code change with good odds of fixing the blow-up.

1. Remove the separate call to `diffusion.step(...)` from the end of `PhysicsEngine::tick`.
2. Move diffusion into the per-substep loop so there is one common `dt`.
3. Compute one Laplacian per substep.
4. Update `wave_v`.
5. Update `atmosphere` using both `wave_v` and diffusion in the same substep.
6. Use a conservative common explicit `dt`.
7. Replace hard breach zeroing with ramp-to-vacuum or sink cells.
8. Smooth grenade source over 3x3.

### Unified explicit pseudocode

```cpp
// choose common dt from both wave and diffusion limits
float dt_wave = 0.5f / c;
float dt_diff = 0.10f / std::max(D, 1e-6f);
float dt = std::min(dt_wave, dt_diff);

int n_steps = std::max(1, int(std::ceil(sim_time / dt)));
float actual_dt = sim_time / n_steps;

for (int step = 0; step < n_steps; ++step) {
    feed_sources(atmosphere, wave_source, actual_dt);

    compute_laplacian(atmosphere, lap);

    // wave velocity update
    for (i) {
        wave_v[i] += (c*c * lap[i] - damping * wave_v[i]) * actual_dt;
    }

    // unified pressure update
    for (i) {
        atmosphere[i] += wave_v[i] * actual_dt
                       + D * lap[i] * actual_dt;
    }

    apply_breach_relaxation_or_sink(atmosphere, wave_v, actual_dt);
    apply_obstacle_rules(wave_v);
    compute_wind_from_pressure(atmosphere, wind_x, wind_y);
    advect_smoke(smoke_field, wind_x, wind_y, actual_dt);
}
```

### Notes
- This is much better than the current split version.
- But it is still explicit, so it still needs a conservative `dt`.

---

## Plan B — recommended stable patch
Use this if you want the cleanest numerical fix.

1. Keep one common per-substep loop.
2. Keep wave explicit.
3. Make diffusion implicit.
4. Solve
   \[
   (I - D\Delta t\,\Delta_h)u^{n+1} = u^\star
   \]
   each substep.
5. Keep the wave CFL, but remove diffusion CFL sensitivity.
6. Use ramped breach cells or sink cells instead of hard Dirichlet zero.

### Unified IMEX pseudocode

```cpp
float dt = 0.5f / c;   // wave CFL only, or slightly smaller
int n_steps = std::max(1, int(std::ceil(sim_time / dt)));
float actual_dt = sim_time / n_steps;

for (int step = 0; step < n_steps; ++step) {
    feed_sources(atmosphere, wave_source, actual_dt);

    compute_laplacian(atmosphere, lap);

    // explicit wave kick
    for (i) {
        wave_v[i] += (c*c * lap[i] - damping * wave_v[i]) * actual_dt;
    }

    // explicit drift target
    for (i) {
        u_star[i] = atmosphere[i] + wave_v[i] * actual_dt;
    }

    // implicit diffusion solve:
    // (I - D*dt*L) atmosphere_new = u_star
    solve_diffusion_implicit(atmosphere_new, u_star, D, actual_dt, obstacles, is_vacuum);

    swap(atmosphere, atmosphere_new);

    apply_breach_relaxation_or_sink(atmosphere, wave_v, actual_dt);
    apply_obstacle_rules(wave_v);
    compute_wind_from_pressure(atmosphere, wind_x, wind_y);
    advect_smoke(smoke_field, wind_x, wind_y, actual_dt);
}
```

### How to solve the implicit diffusion system
Because this is a 2D grid with local stencils, a Gauss-Seidel or red-black Gauss-Seidel solve is completely reasonable.

For a cell with coefficient

\[
u - \mu (u_\text{up}+u_\text{down}+u_\text{left}+u_\text{right}-4u) = u^\star,
\]

the interior update is

\[
(1+4\mu)u_{i,j}
-
\mu(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1})
=
u^\star_{i,j}.
\]

So a Gauss-Seidel iteration can use

\[
u_{i,j}
=
\frac{
u^\star_{i,j}
+
\mu(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1})
}{
1+4\mu
}.
\]

At obstacles, keep your Neumann reflection logic.
At breach/vacuum cells, use the relaxation/open-boundary logic instead of hard zero if you want smoother behavior.

### Why this is the recommended path
- diffusion becomes much more stable
- one common update loop
- no separate diffusion pass at the end
- still easy to reason about
- keeps the current game feel closer to what you already have

---

## 13. Suggested rollout order

1. **First**: unify the updates into one loop
2. **Second**: reduce grenade source from single-cell impulse to 3x3 kernel
3. **Third**: replace hard breach zero with relaxation-to-vacuum, then optionally add a short sponge layer near the breach
4. **Fourth**: if explicit still needs too small `dt`, switch to IMEX implicit diffusion
5. **Fifth**: only after that, consider splitting into base pressure + acoustic pressure

---

## 14. Bottom line

### Why the current solver blows up
Because the current split coupling becomes unstable when explicit diffusion makes high-frequency modes change sign:

\[
1-\mu\sigma < 0.
\]

Your current diffusion cap allows that in 2D.

### What to do now
The best next patch is:

- **unify wave and diffusion into one common per-substep update**
- **do not keep a separate diffusion pass at the end of the tick**
- **smooth breach handling**
- **smooth grenade source slightly**
- ideally make diffusion **implicit** inside the unified loop

### Should the fields be split?
Not yet for the first patch.
Keep one main pressure field for now so you preserve the decompression behavior you currently like.
Split later only if you want separate artistic control over:
- long-term air loss
- short shockwave/acoustic motion

---

## 15. Code locations to patch

- `physics_engine.cpp`
  - remove the trailing standalone diffusion call
  - replace with unified per-substep atmosphere update

- `atmo_diffusion.cpp`
  - likely becomes obsolete if Plan A is used
  - or becomes an internal implicit solve helper if Plan B is used

- `wave_solver.cpp`
  - logic may be merged into the unified atmosphere solver
  - boundary handling should be softened at breach cells

---

## 16. One-sentence recommendation for Claude Code

Implement a single unified per-substep atmosphere solver where `wave_v` is updated explicitly from the Laplacian, `atmosphere` is updated in the same substep, the old trailing explicit diffusion pass is removed, breach cells are relaxed smoothly toward vacuum instead of hard-clamped, and diffusion is preferably handled implicitly via a Gauss-Seidel solve so stability is controlled by the wave CFL rather than by the diffusion CFL.

---

## 17. Implementation notes (develop branch, 2026-03-19)

### What was implemented: Plan A (unified explicit)

The following changes implement the rollout steps from §13:

**1. Unified substep loop** — new `AtmosphereSolver` class (`cpp/src/atmosphere_solver.cpp/.h`):
- Single `step(dt)` function: one Laplacian computation reused for both wave and diffusion
- Update order per step: feed sources → Laplacian → wave_v kick → unified atmosphere update → BCs → wind
- Combined CFL: `dt = min(0.5/c, 0.10/D, combined_quadratic)` → 1.667ms with c=300, D=50
- ~50 substeps per tick (vs 39 wave + 70 diffusion = 109 before). Fewer total operations.
- Wind (pressure gradient) computed every substep as a byproduct of the Laplacian pass

**2. Grenade source smoothing** — `game.py apply_explosion()`:
- Wave source deposits spread over 3×3 kernel `[1,2,1; 2,4,2; 1,2,1]/16`
- Total energy preserved, high-frequency grid-scale excitation reduced

**3. Relaxation-to-vacuum breach BC** — replaces hard Dirichlet `p=0`:
- `atmosphere[breach] *= (1 - eta)` where `eta = clamp(breach_rate * dt, 0, 1)`
- `breach_rate = 5.0` (configurable in config.toml) — drains a room in ~1-2 seconds
- Much smoother than hard zeroing, reduces high-frequency ringing at breaches

**4. Architecture change: PhysicsEngine removed**
- `physics_engine.cpp/.h` and `wave_solver.cpp/.h` and `atmo_diffusion.cpp/.h` replaced by single `atmosphere_solver.cpp/.h`
- Orchestration moved to Python (`Physics.step()` in `game.py`)
- Each C++ solver is a single-step function: `AtmosphereSolver.step(dt)`, `SmokeDynamics.step(dt)`, `FireSimulation.step(dt)`
- Python calls them in a loop, choosing substep count from `max_dt()`
- This makes solvers composable, testable, and debuggable from Python

### What was NOT implemented yet
- Plan B (IMEX implicit diffusion) — reserved for main branch
- Sponge layer near breaches (§10, option 5) — try if relaxation still rings
- Field splitting into `p_base + p_wave` (§9) — deferred per recommendation

### Parameters (develop branch)
| Parameter | Value | Source |
|---|---|---|
| `c` | 300.0 | `Physics.WAVE_C` |
| `damping` | 3.0 | `Physics.WAVE_DAMPING` |
| `d_atm` | 50.0 | `config.toml` |
| `feed_rate` | 200.0 | `Physics.SOURCE_FEED_RATE` |
| `breach_rate` | 5.0 | `config.toml` |
| unified dt | 1.667ms | computed by `max_dt()` |
| substeps/tick | ~50 | `ceil(83.3ms / 1.667ms)` |

---

## 18. Implementation notes (main branch, 2026-03-19)

### What was implemented: Plan B (IMEX — implicit diffusion)

**Two-field architecture:** `wave_p` (acoustic) + `atmosphere` (bulk pressure), coupled via anomaly transfer.

**1. AtmosphereSolver with IMEX scheme** (`cpp/src/atmosphere_solver.cpp/.h`):
- Single `step(dt)` function with explicit wave + implicit diffusion
- Wave: explicit kick on wave_p (Laplacian → wave_v update → wave_p update → BCs)
- Transfer: wave anomaly `(wave_p - mean)` transferred into atmosphere
- Diffusion: **implicit** via red-black Gauss-Seidel (8 iterations/substep)
  - Solve `(I - μΔ) atm_new = atm_current` where `μ = D * dt`
  - Amplification factor `1/(1+μσ)` always positive, always ≤1 → unconditionally stable
- Wind: gradient of `atmosphere + wave_p` computed every substep
- Only wave CFL matters: `dt = 0.5/c = 1.67ms`, ~50 substeps/tick

**2. Boundary conditions — sealed vs breach distinction:**
- Border vacuum tiles are `is_vacuum + obstacles + is_wall` → Neumann reflection blocks waves/diffusion
- Breach vacuum tiles are `is_vacuum` only (wall was destroyed) → waves/diffusion propagate into them
- Sponge layer seeds ONLY from exposed (non-obstacle) vacuum → sealed ship doesn't leak
- Relaxation BC: `atmosphere *= (1-η)` on vacuum tiles (smooth drain)
- Neighbor-mean fill on wall destruction (no hard atmosphere=0)

**3. Source injection stabilization:**
- 3×3 kernel smoothing on wave_source deposits
- Rate limiter: `max_source_per_step = 0.5` caps per-substep injection
- Direct atmosphere deposit in `apply_explosion` removed (caused spikes)

**4. Architecture: PhysicsEngine removed, Python orchestrates:**
- `AtmosphereSolver.step(dt)` — one substep, called N times from Python
- `SmokeDynamics.step(dt)` — unchanged, called once per tick
- `FireSimulation.step(dt)` — unchanged
- `Raycaster.update_from_fire()` — unchanged

**5. Stress test results:**
- `wave_source = 50` (5× triple grenade): no NaN, no blowup
- Sealed ship: holds pressure indefinitely (sponge blocked by hull)
- Hull breach: drains correctly via relaxation + diffusion

### Parameters (main branch)
| Parameter | Value | Source |
|---|---|---|
| `c` | 300.0 | `Physics.WAVE_C` |
| `damping` | 3.0 | `Physics.WAVE_DAMPING` |
| `transfer` | 0.5 | `Physics.WAVE_TRANSFER` |
| `d_atm` | 50.0 | `config.toml` |
| `feed_rate` | 200.0 | `Physics.SOURCE_FEED_RATE` |
| `breach_rate` | 5.0 | `config.toml` |
| `max_source_per_step` | 0.5 | `AtmosphereSolver` default |
| `gs_iters` | 8 | `AtmosphereSolver` default |
| dt | 1.667ms | wave CFL only (diffusion is implicit) |
| substeps/tick | ~50 | `ceil(83.3ms / 1.667ms)` |
