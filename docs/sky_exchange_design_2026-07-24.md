# Sky exchange (planetside volumetric O₂) — design (2026-07-24, Fable)

**Context:** fire-tuning §7 **Q2 resolution** (`docs/fire_tuning_plan_2026-07-22.md`,
"Q2 RESOLVED" — read it first; this doc turns that decision into a build).
The 2-D slice has no sky: an open planetside field deplete-able by one crate
is the missing third dimension, not a BC bug. **Erik's decision: Option A —
sky exchange, composition-swap variant.** Every sky-connected air tile slowly
relaxes its gas COMPOSITION toward ambient at fixed local N_total — the
vertical mixing a top-down slice cannot resolve, as a local source term.

**Status: DESIGN — awaiting Erik's read + Opus build. Own branch
(`sky-exchange`). Independent of `o2-continuous-law` (different files/passes;
either merges first). The joint fire re-tune waits for BOTH.**

---

## 1. Spec

### 1.1 The sky mask

`GameMap.sky_mask` (bool, h×w): flood fill from the `is_ambient` ring through
open air (`~solid`), exactly the sponge-BFS reachability (`gamemap.py`
`_build_sponge_grids` pattern — same seeds, same passability, no distance cap).
Rebuilt where the sponge/structural grids are rebuilt (the structural-change
cache pattern; a wall breach EXPANDS the mask on rebuild — a newly opened room
starts breathing, which is correct and desirable).

- Space maps / no ring → mask all-false → the pass is dead → byte-identity.
- Sealed rooms → excluded by the flood fill (correct: no sky).
- **Accepted approximation (Erik):** a ROOFED room with an open door is
  sky-connected and gets refill. Revisit with an authored roof mask only if it
  bites in practice.

### 1.2 The exchange (per tick, per sky tile — NOT per substep)

```
target = mul_q16(o2_frac_q, N_total[i])      # ambient composition at LOCAL mass
dN     = mul_q16(lambda_q, target − N_O2[i]) # signed, sign-symmetric rounding
N_O2[i]    += dN
N_inert[i] −= dN                             # N_total invariant BY CONSTRUCTION
```

- `o2_frac_q` from the authored `[ambient] o2_frac` (one source of truth with
  the ring clamp).
- `lambda_q = quantize(dt_tick / sky_tau_s)` — **config dial `[ambient]
  sky_tau_s`**, the vertical-mixing timescale. Default **60.0** (seconds;
  bench-calibrated in P3). `0`/absent → pass disabled (also the back-compat
  default for existing levels until blessed).
- Per TICK, not per substep: τ ≫ tick, so substep placement buys nothing and
  costs resident-path seams. Ordering: immediately AFTER the combustion pass
  (combustion vitiates, sky replenishes, fire's next-tick read sees the net).
- Defensive clamps: `N_O2 ∈ [0, N_total]`, `N_inert = N_total − N_O2`
  restated exactly after the swap (no LSB leak between the pair).
- **Scope: the conservative pair ONLY.** Smoke's upward-removal λ is DEFERRED
  (a B2-adjacent look decision — §7 Q2 block). Temperature untouched
  (`COOL_SHIFT` is already the vertical heat channel).

### 1.3 Conservation accounting

Per-plane totals now change volumetrically (O₂ up, inert down, N_total
conserved). Extend the open-system rail pattern (`boundary_flux` in
`bulk_transport.cpp`): accumulate `sky_flux[plane] += dN` (int64, per tick) so
the conservation gate closes: `Δtotal(plane) = boundary_flux + sky_flux`.
`tests/test_atmosphere_conservation.py` learns the new term.

## 2. Gates

a. **Space-map byte-identity** — no ring → no mask → untouched paths. Zero
   tolerance.
b. **Sealed-room identity** — a sealed room's planes bit-identical with the
   pass on (mask excludes it).
c. **Conservation rail** — per-plane budget closes to zero with `sky_flux`
   included; N_total per tile invariant across the pass (exact, not approx).
d. **The acceptance the feature EXISTS for (harness):** locked-combo crate
   burn, sponge-safe bench, `sky_tau_s = 60` → far-field O₂ stays ≥ 0.19 for
   the WHOLE burn (vs 0.21→0 in ~5 min today); post-burnout field returns to
   0.21 with time-constant ≈ τ.
e. **CPU↔CUDA lockstep** — bit-identical including the resident path (planes
   are device-resident since S8a; the pass runs where the planes live — a
   small kernel + mask upload on the resident side).
f. **Determinism digest** — moves on planetside maps only; deliberate golden
   handling at the joint re-tune (space goldens must NOT move — that's gate a).

## 3. Patch plan (Opus; branch `sky-exchange`)

- **P1** — CPU: `sky_mask` build + rebuild hook; the exchange pass (C++,
  called from the runner tick after combustion); `sky_flux` rail;
  `[ambient] sky_tau_s` config + `level_loader` validation (mirror the
  `sponge_u_damp` validation pattern); gates a/b/c unit tests.
- **P2** — CUDA resident mirror + lockstep (gate e).
- **P3** — bench calibration of τ (gate d; sweep τ ∈ {30, 60, 120} for the
  re-tune session's menu) + planetside_demo smoke-look sanity (smoke is
  untouched — verify no indirect change).
- **P4** — canon fold: `docs/architecture/engine/04_atmosphere_and_pressure.md`
  gains the sky-exchange section (BC chapter: ring = edge reservoir + sky =
  volumetric composition reservoir); archive this doc; HUMAN-TEST: Erik burns
  a crate on a planetside map and the field does not suffocate.

**Escalation triggers (stop, back to Fable/Erik):**
1. The composition swap measurably perturbs pressure/wind anywhere (it must
   not — N_total invariance is the design's load-bearing property; any
   coupling means an implementation bug or a design miss).
2. The resident-path seam for a per-tick host-side pass is awkward (extra
   D2H/H2D per tick) — needs an on-device pass decision.
3. Mask rebuild cost or trigger granularity surprises (giant maps, frequent
   breaches).
4. Any temptation to touch smoke, temperature, or the ring/sponge — out of
   scope here.

## 4. What this deliberately does NOT do

No smoke λ (deferred), no temperature term, no buoyant-chimney mass venting
(Option B — a possible LATER feel addition), no ring/sponge changes (the EMA
sponge is its own item: `docs/ema_sponge_design_2026-07-23.md`), no roof
authoring. The joint fire re-tune (burn_rate=0.02 world) happens AFTER this
and `o2-continuous-law` both merge — plan in the Q1 doc §5.
