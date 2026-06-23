# stamp_units → C++ (port plan)

**Status:** plan, awaiting Erik's OK (2026-06-23). Closes the one deferred loose end from the
PhysicsEngine unification (Patch 1 scoped it but it shipped Python). **Pure-structure move —
0-ULP-gated**, like Patch 1's S-steps. Maps from the Explore audit of `gamemap.py:485-589`.

---

## 1. Why

`stamp_units` rebuilds the **dynamic** per-tile fields the solvers read each tick — `obstacles`,
`dyn_permeability`, `dyn_wave_absorb`, `dyn_light_atten` — from the static material grids plus every
living unit's footprint. It runs **once per tick in Python, before `physics.step`**. The resolution
research flagged it as a farm cost (a full-field Python rebuild every tick × hundreds of sims) and a
**GPU-residency prerequisite**: those dyn_ fields are read by the (soon GPU-resident) solvers, so the
rebuild must live next to them in C++ (then CUDA), not in Python. Moving it now finishes the
unification chapter before the fixed-point arc.

It is the *cleanest possible* port: **no float arithmetic** — only full-field resets (in-place
copies), a boolean comparison, and per-cell min/max. So it is bit-exact by construction.

---

## 2. The exact contract (must reproduce byte-for-byte)

Per tick, over a deterministic `List[Unit]` (stable append order; no RNG):

**a. Reset to static baseline (full-field, in-place):**
```
obstacles      = (permeability <= 0.0)          # walls only; units NOT stamped into obstacles
dyn_permeability[:]  = permeability             # in-place copy (keeps the C++ view valid)
dyn_wave_absorb[:]   = wave_absorb              # in-place
dyn_light_atten[:]   = light_atten              # in-place (h,w,3)
```

**b. Stamp each living unit's footprint** (`for u in units if u.alive`, then
`for (tx,ty) in u.occupied_tiles()`, bounds-checked):
```
u_perm    = u.permeability  or CFG.physics.unit_permeability  (0.5)
u_wabsorb = u.wave_absorb   or CFG.physics.unit_wave_absorb   (0.5)
u_atten   = u.light_atten   or (1.0, 1.0, 1.0)

dyn_permeability[ty,tx]    = min(u_perm, permeability[ty,tx])          # MIN — never unseal a door
dyn_wave_absorb[ty,tx]     = max(dyn_wave_absorb[ty,tx], u_wabsorb)    # MAX — body only adds damping
dyn_light_atten[ty,tx,c]   = max(dyn_light_atten[ty,tx,c], u_atten[c]) # per-channel MAX — opacity ↑ only
```

**c. Atmosphere refill** (`gamemap.py:586-588`): a small conditional fill where wall→free transitions
freed a cell. *Decision needed (Q1): port it too, or leave this one bit in the Python wrapper.* It is
not unit-driven; it can stay Python without losing the GPU benefit.

**Combine ops are exact** (min/max/copy/compare — no rounding), so float32 Python and float C++ agree
to the bit. No `/fp:precise` translation unit required (unlike the Patch-1 solver glue).

---

## 3. The seam: how unit data reaches C++

Units are CPU/actor-side and stay Python (canon GPU split: *GPU owns fields, CPU owns actors*). So we
pass a **small per-tick delta** down — the stamped footprint tiles — exactly the "deltas-up" seam:

- **Python** (thin builder): from living units, build flat arrays of the stamped tiles —
  `ys[], xs[], perm[], wabsorb[], atten_r[], atten_g[], atten_b[]` (one row per (unit, footprint
  tile)). This keeps unit iteration + `occupied_tiles()` on the CPU.
- **C++** new `PhysicsEngine::stamp_units(...)`: does the full-field reset (in-place) + the boolean
  `obstacles` + the min/max stamp loop over those flat arrays. Writes the dyn_ fields in place
  (never reassigns — keeps the engine's re-fetched pointers valid).
- **Binding:** one new entry in `bindings.cpp` (array-flatten pattern, like gas/fields — not a bound
  Unit class). Pass the static grids + the dyn_ targets + the flat stamp arrays + the two config
  defaults.

*Decision (Q2):* flatten in Python (above, simplest, unit loop stays Python) **(recommended)** vs.
bind the `Unit` class and iterate in C++. Recommend the flatten — it matches the actor/field split
and the existing pass-by-array precedent, and the Python unit loop is negligible.

---

## 4. Call-site wiring + the W3 ordering constraint (load-bearing)

Today: `simulation.py:609` (round-start, conditional) and `:626` (every tick, phase 2), **before**
`physics.step`. The new C++ `stamp_units` is called at the **same two points**, before the engine's
physics calls.

**Critical ordering (W3 flood-seal):** `stamp_units` must FULLY RESET `dyn_permeability` *before*
`PhysicsEngine::step_water`'s W3 pass overwrites flooded cells to `0` (seal). The order
`stamp_units (reset+stamp) → field-edit flush → physics.step (→ step_water seals)` MUST be preserved,
or last tick's flood-seals go stale. The port keeps the exact same call order — we are only moving
*where the work runs*, not *when*. (This is also why we do NOT make stamping incremental/delta —
deferred to the GPU patch with a proper stale-clear story.)

---

## 5. Gate (0-ULP)

The field-level A/B harness, as in Patch 1: capture `obstacles`, `dyn_permeability`, `dyn_wave_absorb`,
`dyn_light_atten` (+ everything downstream) per tick; run Python `stamp_units` vs the C++ path on the
same seed/scenario; assert **bit-identical** per cell over a trajectory.

- Ensure the harness scenario has **living units with footprints that move and die** (so the stamp
  actually changes tick to tick). The default scenario seeds a unit; extend if needed so the stamp is
  non-trivial. *(Q3: confirm the dyn_ fields are in `SIM_FIELDS`; add if missing.)*
- Plus: full suite green; `--auto` smoke runs exit 0.

Because the ops are exact, this should be 0-ULP first try (the risk is a *contract* slip — a wrong
min/max direction or a missed reset — which the harness catches immediately, not codegen drift).

---

## 6. Steps

1. **Harness prep** — confirm/extend the A/B scenario to exercise unit stamping; confirm dyn_ fields
   captured. Snapshot a golden.
2. **C++** — add `PhysicsEngine::stamp_units(...)` (reset + obstacles + min/max stamp loop) + the
   binding. In-place writes only.
3. **Python** — build the flat stamp arrays from living units; swap the two call sites to call the C++
   path; keep the atmosphere-refill bit (Q1) wherever we decide.
4. **Gate** — 0-ULP A/B over the trajectory; full suite; `--auto`.
5. **Commit** (its own commit), then it rides to `main` with the next push.

---

## 7. Open questions for Erik

- **Q1** — atmosphere-refill (gamemap.py:586-588): port to C++ too, or leave in the Python wrapper?
  (Recommend: leave Python for now — not unit-driven, no GPU benefit lost.)
- **Q2** — pass unit data by **flat arrays** (recommended) or bind the `Unit` class?
- **Q3** — (implementation detail) confirm dyn_ fields are in the harness `SIM_FIELDS`.
- **Autonomy** — this is a 0-ULP-gated pure-structure move (Patch-1 style). OK to implement it
  autonomously against the gate while you're away, or hold for your review?
