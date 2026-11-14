# G12 — ONE temperature map (fire session #12, Phase 1 patch spec)

> **Status: SPEC — authored for the implementation agent, 2026-08-31.**
> Ruling G12 (Erik, issue #12 comment 2026-08-31): collapse the canonical
> ×3 map onto the EOS frame — slope 1, ambient 293. Supersedes the
> 2026-08-30 two-map phone proposal AND the P-K1 "ruling 6" eos_t_amb_k
> exception. Source survey: `docs/fire_mechanics_inventory_2026-08-31.md`
> §2B. This doc is the WRITTEN RATIONALE for the golden re-baseline.

## 0. The one sentence

After this patch, `K = 293 + T_game` is the only temperature frame in the
repo — thermodynamics, radiation bake, render blackbody, hover, tools —
and the sub-ambient `K_eos` hover patch (born from the frames disagreeing)
is deleted.

## 1. Dial moves (config.toml)

| Key | Section | Old | New | Why |
|---|---|---|---|---|
| `k_temp_to_kelvin` | `[physics.temperature_scale]` (~:705) | 3.0 | **1.0** | THE collapse |
| `phi_exp` | same (~:706) | 0.333… | **1.0** | invariant `phi_exp·k ≡ 1.0` exactly (quantizes to 65536; 1.0·1.0 is exact, no tie) |
| `eos_t_amb_k` | same (~:711) | 290.0 | **293.0** | one ambient; kills ruling 6's exception. C = 1/293 now (≈1% pressure-calibration shift — golden change) |
| `kelvin_ambient` | same (~:704) | 293.0 | 293.0 (unchanged) | already the target ambient |
| `T_MIN` | `[physics.eos]` (:617) | −289.0 | **−292.0** | floor T_abs = 293 − 292 = 1 K, same convention as before |
| `rad_scale` | `[physics.fire]` (:427) | 3.1394e-6 | **5.1427e-5** | §2 re-anchor |
| `T_emit_gate` | `[physics.fire]` (:437) | 310.0 | **930.0** | §3 physical-gate preservation |

Everything else in game units — `ignition_temp`, `fire_T_ext`,
`fire_T_span`, `T_MAX_PHYS`, `t_light_min`, cool_shift table — is
**unchanged** (§4).

## 2. rad_scale re-anchor (the P-K2 precedent, applied again)

The E° bake is `E°(T) = rad_scale · K(T)⁴` with K from the canonical map.
The map's slope change alters K at every stored T, so rad_scale is
re-anchored to **preserve emitted flux at the P-F1b plateau anchor
T_a = 300 game** — the same anchor P-K2 used for ×2→×3:

```
K_old(300) = 293 + 3·300 = 1193      K_new(300) = 293 + 300 = 593
rad_scale' = 3.1394e-6 × (1193/593)⁴ = 3.1394e-6 × 16.3811 = 5.1427e-5
```

Cross-check against the original ×2-era literal (chain must agree):
`1.0e-5 × (893/593)⁴ = 5.1426e-5` ✓ (same anchor through both re-anchors).

Flux ratio E'/E vs today, by stored game-T (document in the config
comment, replacing the stale ×2→×3 table):

| T_game | 0 | 100 | 200 | 300 | 443 | 600 | 1000 | ∞ |
|---|---|---|---|---|---|---|---|---|
| E'/E | 16.38* | 3.16 | 1.52 | **1.00** | 0.69 | 0.54 | 0.39 | 0.20 |

*gated: sub-`T_emit_gate` solids don't cast, so the ambient-end blowup is
mostly unreachable; net exchange between near-equal tiles still cancels.
Direction of change: the honest map compresses K's dynamic range, so
radiation varies less steeply across game-T — hot flames radiate relatively
less than today, warm solids relatively more. This is intended physics
(Phase 3a measures on this frame); the anchor pins the plateau.

**Two-tile ignition inequality — RE-DERIVE with CURRENT dials.** The
shipped config comment derives it with cool_shift=9 (stale — P-K0 promoted
wood/furniture/kindling to 13). Scaling the old requirement
(1.73e-6 at cool 9, ×3 map: K_s=1622, K_r=1133 → ΔK⁴=5.274e12) to the new
map (K_s=736, K_r=573 → ΔK⁴=1.856e11) and cool_shift 13:

```
rs_req = 1.73e-6 × (5.274e12/1.856e11) × 2^(9−13) = 3.07e-6
margin = 5.1427e-5 / 3.07e-6 ≈ ×16.7
```

⚠ AGENT: read the ACTUAL furniture cool_shift from config.toml before
writing the comment. If it is still 9 anywhere relevant, the margin is
only ×1.05 (knife-edge) — in that case STOP and flag, don't ship silently.

## 3. T_emit_gate: preserve the physical gate

The gate decides which hot solids CAST. Stored T doesn't move in this
patch, but the gate's Kelvin meaning does. Rescale so the emitter set is
**bit-identical before/after** at any given state:

```
old physical gate: 293 + 3·310 = 1223 K  →  new game value: 1223 − 293 = 930
```

Config comment note for Phase 4: Erik's original blessed feel value
(2026-07-31) was 653 K — ≈ 360 game under the honest map — LOWER = warm
objects gently toasting their surroundings; the P-K0 stall
(receiver-becomes-emitter ceiling collapse) is the guard-rail to re-check
when lowering it.

## 4. What deliberately does NOT change

- **Ignition temps** (wood 300 / furniture 280 / kindling 280): these were
  already compared against game-T in the sim — and under the honest frame
  they now READ as sensible Kelvin: wood 593 K, furniture/kindling 573 K
  (real-world wood ignition ≈ 573–673 K). Record this table in the config
  comment; review = done, values kept. (`fire_T_ext` = ign − 200 → wood
  393 K extinction; Phase 4 territory.)
- `T_MAX_PHYS = 16000` (now honestly 16293 K — absurd headroom, harmless).
- `[render.blackbody]` LUT keys (`kelvin_floor/ceil/glow_min/ref`): they
  are Kelvin-NATIVE and frame-independent. Consequence, not a change:
  flames will read ~⅓ the Kelvin they used to → **less white-hot glow
  automatically** (July's "14764 K plateau" symptom partly self-corrects).
  Judged at the HUMAN-TEST.
- Digest spec: field membership/dtype unchanged → **no version bump**;
  golden VALUES change → one deliberate re-baseline (§7).

## 5. Code sites

1. `config.toml` — the seven rows in §1 + comment rewrites at each site
   (the rad_scale derivation block :405-427 gets the §2 story; the
   temperature_scale block :697-711 loses the ruling-6 exception text;
   `T_emit_gate` gets the §3 note).
2. `src/temperature_scale.py` — `DEFAULTS` → {k_temp_to_kelvin: 1.0,
   phi_exp: 1.0, eos_t_amb_k: 293.0}; assert message + module docstring
   updated (the invariant `quantize(eos_slope)==65536` STAYS — it's the
   protection, G12 is the sanctioned value move).
3. `renderer/hover_readout.py` — DELETE the dual-frame branch (:83-97):
   one `kelvin_fn` call, label "K" always; map is valid to the floor
   (T_MIN=−292 → 1 K). Update docstring + `tests/test_hover_readout.py`.
4. `src/simulation/physics_runner.py` — stale comments only (:455 area
   "eos_t_amb_k stays 290 (ruling 6…)" and the S_EOS notes; :350-370
   raycaster wiring comments if they cite ×3).
5. `cpp/src/raycaster.cpp` :44-46 — comment-only (bucket-midpoint example
   becomes `295 + 4t`); the bake reads member vars, no C++ logic change.
   `cpp/src/eos_solver.cpp` :445 comment likewise if it cites 290.
6. Tests pinning old constants (grep hits, fix each to the new map):
   `test_temperature_scale.py`, `test_blackbody_ramp.py`,
   `test_eos_p1_calibration.py`, `test_gas_energy_field.py`,
   `test_hover_readout.py`, `test_pf1a_radiation_books.py`,
   `test_pr1_fire_plane_cast.py`, `test_p_e3_drag.py`,
   `test_velocity_clamp_property.py`, `cuda_kick_check.py`,
   `cuda_kick_drag2_timing_check.py`, `cuda_po2b_check.py`,
   `cuda_pr1_fire_plane_check.py`, `cuda_thermal_mass_eos_check.py`.
   Rule: update the pinned VALUE to the honest frame; never weaken a gate's
   structure. If a test's whole premise was the frame split (K_eos hover
   tests), delete it and say so in the commit message.

## 6. Expected behavior deltas (what the goldens will show)

1. EOS pressure calibration C: 1/290 → 1/293 — every pressure/gas golden
   shifts ~1%.
2. `gas_energy = N·T_abs` ledger values shift (T_abs now +293).
3. Radiation exchange reshaped per §2's ratio table (anchor-preserved).
4. T floor 3 units lower (−292).
5. Render: glow less white (Kelvin-native LUT, honest K reads).
Nothing else should move. A diff showing e.g. combustion-side changes is a
bug in the patch, not a consequence.

## 7. Gates (all before commit; goldens same commit)

- Full suite `pytest tests -q` — green except the 2 known parked
  cool_shift reds (+ this box's scipy/numpy `test_bench_two_room` env red).
- CUDA wrappers (`test_cuda_*`) green — the mirror inherits constants
  through the same config path; consts cross-check legs updated in §5.6.
- GOLDEN_AGGREGATE + per-field goldens: regenerate ONCE, same commit,
  rationale = THIS DOC (link it in the commit message).
- Ingress lint + float ratchet untouched (config/comment/test churn only —
  no new float ingress in sim TUs).
- HUMAN-TEST (after commit, Erik at the screen): a burning scene — glow
  reads less white; hover shows one honest K everywhere including cold
  tiles (no more "−574 K" class artifacts, no K_eos label).
