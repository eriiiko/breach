# Spec Review — Findings & Revision Agenda

_Review run 2026-06-05 · 4 perspectives (CUDA/perf, rendering, sim/determinism, architecture)._
_All four verdicts: **sound-with-revisions.** Core validated; spec has completeness holes._

Tags: **[DECIDE]** = needs Erik's call · **[FIX]** = clear correction, apply on nod.
"×N" = how many independent reviewers raised it (convergence = signal).

---

## ✅ RESOLUTION STATUS — rev.2 applied (2026-06-05)

**All `[DECIDE]` items decided by Erik and all `[FIX]` items applied** across ch.01–05 (rev.2).
Key resolutions: stealth scalar **removed** (deferred + image/field-stack based) · `wall_hp`
**GPU-resident/GPU-written**, `destroy_wall`=delta · LoS = **any light through** (attenuation,
not `is_wall`) · temperature = **faked relaxation** (no substeps, shifts not divides) ·
**fixed-point kept** for heat+temperature (for **lockstep multiplayer** / cross-machine) · render
buffers = **2× RGBA16F** · god-rays **supersede** `light_modulation` · `smoke_glow` = **RGB** ·
tone-map = **ACES** · units occlude via the **dynamic per-channel RGB attenuation field**
(programmable colour, read-only-safe) · `walkable` = **CPU-only** · acoustic `wave_reflect`/
`wave_absorb` **kept** (transmit = remainder), separated from optics · unified **tick order**
written into ch.03 · the **summed render interface** + **light-is-simulation** rationale added.

**Remaining = validation / CUDA-phase only** (not blocking the CPU build): cross-machine
bit-exactness test for fixed-point temperature; exact fixed-point width; verify the wave solver's
current use of reflect/absorb; per-tick ray-list construction + atomic contention (CUDA phase);
serialization/replay section drafted (ch.01). A lighter **re-review** runs next.

## ✅ RE-REVIEW + rev.3 (2026-06-05)

A second review of rev.2 returned **all four verdicts at "minor-revisions"** (up from
sound-with-revisions), each verified against the actual code. It confirmed every rev.2 fix
genuinely landed, and found **2 new majors I'd introduced + minor polish** — all applied as
**rev.3**:

- **f16 accumulator slip → f32 accumulate, 16F store.** "16F" is the storage format only; render
  channels accumulate in f32 (scalar f16 atomics are CC≥7.0-only + mis-fit 3-ch RGB). [ch.03/05]
- **wall_hp seam → weapon pre-phase reads `material` only.** `wall_hp` stays GPU-side; walls
  breach via the GPU thermal-failure reaction, not a CPU `wall_hp` write. [ch.03]
- Tick order marked a **target requiring a `step()` refactor**; "identical trajectories" qualified
  as the post-refactor invariant. [ch.03]
- **`heat` is a non-destructive per-tick deposit buffer**, cleared at cleanup → unit damage is
  independent of conduction order. [ch.03]
- Shader **composite formula** written explicitly; **light_dir** signed-16F (drop the 0.5-bias),
  two-texture reconstruction stated; light_dir **normalize** is a separate full-grid pass. [ch.05]
- God-ray glow **draw-order** pinned (additive, before units). [ch.05]
- Door open/closed framing corrected: **doors always occlude + always walkable until the deferred
  door-state system**. [ch.02]
- `has_los` migration note added (binary→attenuation). [ch.03]

**Remaining = validation / CUDA-phase only** (non-blocking): cross-machine bit-exactness test;
exact fixed-point width; verify the wave solver's reflect/absorb use; per-tick ray-list
construction + atomic contention (CUDA phase); reword cuda-plan §3/§4/§7. A few cosmetic nits
(walkable-not-uploaded clause, ch.01 heat-line grouping, headless "(confirm)"→test) are trivial
final polish.

---

Items below are the original findings, retained for traceability.

---

## What the panel validated (don't second-guess these)

Deposit-only/read-only ray kernel · no in-kernel forking (reflection = entity re-emit) ·
int-atomic **heat deposit** is genuinely order-independent/deterministic · `gmap.<field>`
ownership indirection as a clean migration seam · arrays+table over tile-objects ·
fixed-point-where-a-threshold-lives principle · premultiplied-alpha discipline ·
per-channel attenuation subsuming block/glass · physics-deposits / shader-consumes boundary.

---

## Blockers — convergent (multiple reviewers)

1. **[DECIDE] ×3 — The stealth/LoS light scalar is homeless AND non-deterministic.**
   Stealth ("below threshold = in shadow") is a value that **crosses a gameplay threshold**,
   but it's derived from the **float** `light_rgb` that C8 keeps float "because no downstream
   threshold" — and that premise is *false*. It appears in **no buffer table**, has no owner.
   *Recommended:* add a dedicated **fixed-point int luminance accumulator** (sim channel,
   owner = GameMap) feeding stealth, OR derive stealth from `has_los` + ambient. **Your call:
   is stealth a hard threshold (→ int) or soft/hysteresis (→ float-tolerant)?**

2. **[DECIDE] ×2 — `wall_hp` ownership is contradictory (ch.01 vs ch.04).** ch.01 says
   CPU-authoritative; ch.04 has the temperature stencil deplete it **on-GPU** (thermal
   failure). Two writers, unreconciled; and "tiny deltas up" breaks if a firestorm melts many
   walls/tick GPU-side. *Recommended:* `wall_hp` is **GPU-resident, GPU-written** by thermal
   failure; `destroy_wall` (CPU, from weapons/explosions) becomes an **uploaded delta**.
   Reword ch.01. **Confirm.**

3. **[DECIDE] ×2 — Glass: LoS vs light-attenuation are incoherent.** C12 says rays never read
   `is_wall` (glass transmits light); but `has_los` stops on `is_wall` (glass *is* a wall). So
   you can be **lit through glass but not see/shoot through it.** Infravision inherits it.
   *Recommended:* LoS uses an **accumulated-attenuation threshold** (see dimly through
   glass/smoke), or a separate explicit `opaque-to-vision` mask. **Your call: can you
   see/shoot through the tinted window you can be lit through?**

4. **[DECIDE] ×3 — Temperature ships BOTH the 17-substep model AND the relaxation model
   without choosing.** *Recommended (your own idea resolves it):* **commit to the faked
   unconditionally-stable relaxation** (one pass/tick, power-of-two rates). It dissolves the
   substep loop, the fixed-point division, AND the CFL-on-κ dependence in one move. Delete the
   17-substep model from the locked spec. **Confirm.**

5. **[DECIDE] ×2 — Fixed-point temperature determinism rests on rounding, not atomics.** The
   diffusion stencil is a gather (no atomics), so "int atomicAdd is deterministic" doesn't
   apply to *temperature*. Cross-machine bit-exactness depends on rounding discipline, deferred
   to "confirm-at-implementation." *Recommended:* with the relaxation + power-of-two rates the
   update is shifts/adds (exact) — prototype to confirm bit-exact; fallback = **float
   temperature + int heat-deposit**. **Tied to: is cross-machine determinism actually
   required?**

6. **[DECIDE] ×4 (gap) — Fixed-point format is undefined.** Scale/shift, bit width, and
   **overflow/saturation** (int `atomicAdd` can *wrap* — catastrophic for the ignition
   threshold under a firestorm), plus the quantization rule for float thresholds
   (`ignition_temp`) into the fixed-point domain. *Recommended:* pick a scale (e.g. Q16.16),
   **saturating** add (clamp, not wrap), `ignition_temp` quantized once at load. **OK the
   defaults or specify.**

## Blockers — single-reviewer

7. **[DECIDE] — Render buffers no longer fit one RGBA8 texture.** 6 render channels (`light_rgb`
   3 + `light_dir` 2 + `smoke_glow` 1). ch.05 calls `LightingPass` a "trivial format
   conversion" — it's a multi-texture redesign. 8-bit packing reintroduces the near-dark
   banding C8 cites as the reason for float light. *Recommended:* texture A = `light_rgb`+
   `smoke_glow` (**RGBA16F** to avoid banding), texture B = `light_dir` (RG). **16F or accept
   8-bit banding?**

8. **[DECIDE] — God-rays double-count smoke lighting.** Smoke is lit twice: the existing
   `light_modulation` surface-tint (overlays.py) AND the new `smoke_glow` deposit. *Recommended:*
   `smoke_glow` **supersedes** `light_modulation` (one energy-conserving mechanism). **Confirm
   replace vs layer.**

9. **[DECIDE] — `smoke_glow` is scalar → grey god-rays only.** A red beam's shaft should be
   red. *Recommended:* make `smoke_glow` **RGB** (the marquee visual). Adds to buffer packing.
   **RGB glow or accept grey?**

10. **[FIX] — `destroy_wall` never rebuilds caches.** The new attenuation/conductivity caches
    must update on wall destruction, but `destroy_wall` patches arrays inline. *Fix:* define
    `on_tile_changed(x,y)` that **incrementally patches all derived caches** for that tile
    (no O(grid) rebuild); `destroy_wall` + laser pre-phase funnel through it.

11. **[FIX] — No unified tick-order.** The new systems are never slotted into the existing
    10-step `step()`. *Fix:* write the concrete numbered order — weapon pre-phase **after
    `stamp_units`, before the raycaster**; unify laser wall-destruction with the existing
    burn-through/`destroy_wall` path; state laser mutations are visible to the **same tick's**
    heat raycaster.

12. **[FIX] — `is_wall` is NOT the collision predicate.** Code uses `material in {AIR, DOOR}`;
    `is_wall` *includes* DOOR (doors occlude but are walkable). The spec wrongly calls `is_wall`
    the collision source. *Fix:* name two masks — **`occludes`** (light/smoke/vision, incl.
    closed doors) vs **`walkable`** — and put door duality in the table (`passable` +
    `light_atten` columns), retiring the `np.isin` special-case.

## Majors

13. **[FIX] — Static vs dynamic attenuation conflated.** Material attenuation is per-id static
    (structural-change cache); **smoke/liquid attenuation is per-tile dynamic** (every tick).
    The march multiplies **both**. *Fix:* separate them in ch.02/03; only the static part is a
    structural-change cache.

14. **[DECIDE] — Material table column list is incomplete/inconsistent.** Drops `reflectivity`,
    `absorption`, `blast_resist` from old §4/config without saying removed/renamed/deferred.
    *Recommended:* one authoritative column list — `absorption`→folded into `*_atten`;
    `reflectivity`→deferred to the entity-reflection chapter; `blast_resist`→**keep** (explosions
    use it). **Confirm dispositions.**

15. **[FIX] — Laser pre-phase = potential mid-tick CPU↔GPU stall.** *Fix/clarify:* the resolve
    needs **only structural arrays** (`material`/`wall_hp`, CPU-reachable) — so resolve on CPU
    against current structural state, emit deltas, **no GPU download**. State this in ch.01/03.

16. **[FIX] — Multi-light directional shading limitation.** A single aggregate `light_dir`
    cancels opposing lights and can't carry per-light colored relief. *Fix:* acknowledge as a
    deliberate single-dominant-direction approximation; weight direction by deposited intensity.

17. **[DECIDE] — No tone-mapping for over-bright additive colored light.** Shader will clip +
    hue-shift. *Recommended:* add a tone-map stage (ACES/Reinhard) before sRGB. **Which, or
    hard-clamp?**

18. **[FIX] — One-frame-stale download conflated for render AND sim.** Render tolerates stale;
    **sim must read current-tick post-pass values** (headless has no "frame"). *Fix:* split the
    contract in ch.01.

19. **[FIX] — `block_light` array named in C14 doesn't exist** (only `obstacles` + `is_wall`).
    *Fix:* name the actual dynamic occlusion array units stamp into; relate it to `obstacles`;
    define how a stamped unit injects opacity into the attenuation march.

## Gaps (no home yet)

- **[DECIDE]** GPU **VRAM budget** + grid-resolution target for the full GPU-resident field set.
- **[DECIDE]** **Save/load / serialization / replay** determinism — C2 killed tile-objects'
  "serialization for free"; nothing replaces it (load-bearing for ML/headless).
- **[FIX]** **Sparse side-structures** (named doors, terminals) — no interface/owner; how their
  state reconciles into `is_wall`/attenuation per tick.
- **[FIX]** **Liquid attenuation** — same static-vs-dynamic problem as smoke, unsolved.
- **[FIX]** **Config hot-reload** of the material table (old §14 non-negotiable) + GPU-mirror/
  cache-rebuild interaction.
- **[FIX]** **Units & ray-range units** (tile = 1/3 m; falloff function form) — absent from ch.03.
- **[FIX]** **Determinism regression test** must extend to heat/temperature/ignition.
- **[DECIDE]** **Dynamic emitter discovery** (fire-source clustering) — CPU full-grid scan/tick
  vs GPU stream-compaction (a forking step). Where does the per-ray work-list get built?
- **[FIX]** Add cuda-plan **§3/§4** to the doc-debt (not just §7): scalar→RGB, CFL→relaxation.

---

## Proposed revision sequence

1. **Erik decides the [DECIDE] items** (≈11, but several are quick confirms of the recommended
   resolution — stealth-hardness, glass-see-through, temperature-relaxation, smoke_glow-RGB,
   16F-buffers, tone-map, wall_hp-GPU, fixed-point-format/cross-machine, blast_resist, VRAM
   target, serialization).
2. **Apply all [FIX] items** (mechanical corrections) + the decided ones across ch.01–05.
3. **Re-review** (optional, lighter pass) → then the build.
