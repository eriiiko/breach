# Ray engine v2 — adversarial critique 2: engine integration (2026-09-13)

> **Status:** critique pass, remit = **ENGINE INTEGRATION ONLY**. Not physics,
> not integer numerics — critique 1 covered those and its 12 required changes
> are resolved in v2 and are not re-litigated here.
> **Reviews:** `docs/ray_engine_v2_design_v2_2026-09-13.md`.
> **Against:** `docs/ray_engine_v2_survey_2026-09-12.md` (rulings R-A..R-S, not
> under critique — only the design's fidelity to them) · `CLAUDE.md`'s
> canonical-systems table (the heart of this remit) ·
> `docs/canonical_systems_survey_2026-08-22.md` ·
> `docs/fire_radiation_assumptions_2026-09-11.md` ·
> `docs/architecture/engine/14_determinism_and_number_ingress.md`.
>
> Every finding below carries a file:line citation into the tree at `fire-12`
> `5155894`, or a command whose output is quoted. Nothing was modified; the
> working tree is untouched. Line numbers verified 2026-09-13.

---

## BOTTOM LINE

**P1 is not cuttable as scoped. It becomes cuttable after four document-level
fixes, none of which needs new measurement.** The sweep core is sound and P1 is
the right first patch — I could not find an engine-integration reason to
reorder it. What blocks it is that P1 as written cannot be built from the parts
the tree actually has:

1. **There is no integer extinction plane.** §2.3's `(stream * a_i) >> 16`
   needs `a_i` as a Q16 integer. `heat_atten` is `np.float32`
   (`src/simulation/materials.py:91`, `src/simulation/gamemap.py:406`) and the
   C++ march consumes it as a float (`cpp/src/raycaster.cpp:403`). The design
   never declares the integer plane, its `*_fixed.py` boundary module, or its
   load-time quantization — and the "Q16 boundary modules" canonical row is the
   one it names in its own Systems section.
2. **The maximum-principle clamp cannot live where P1 puts it.** §2.8's
   `T_i ← E°⁻¹(Φ_i)` is a direct `temperature[i] =` write. On a gas cell that
   is forbidden outright by the "Gas temperature is a mirror" row — and the row
   records that the failure is *silent*: "A bare `temperature[...] =` now moves
   no books at all, so it does not fail loudly — it goes VACUOUS." On a solid
   it must sit inside the temperature solver's Pass-1 fold or the P-G5 solid
   ledger stops closing (`cpp/src/temperature_solver.cpp:297-299`). Either way
   the clamp belongs to a TU P1 does not otherwise touch.
3. **The clamp needs a carrier plane the design never declares.** Φ (the
   per-cell absorbed fluence) is produced by the sweep at the *start* of the
   physics step and consumed by the clamp *after* the material update, which is
   `PhysicsEngine::step` step 3 (`cpp/src/physics_engine.cpp:217`). That is a
   new per-tick plane with its own lifetime and its own clear line in the
   conductor (`src/simulation/simulation.py:1597-1616`).
4. **P1's own gate list omits the gate P1 introduces.** §2.9 lists eight gates
   including "5. Maximum principle"; §9's P1 row lists "conservation, second
   law, positivity, stability rail, isotropy". The clamp ships in P1 ungated.

Beyond P1: **§5.3 names the wrong writer for the gas-energy books and would
lead an implementer to build the fifth group the arc forbids** (§2 below);
**P3 is both mis-sequenced and far too large to gate** — it deletes `light_cull`
and the 16-light cap three patches before P6 gives light a field solve, and its
"full suite + one golden re-baseline" understates 46 pytest tests across four
files plus three CUDA check scripts and five tools (§3, §7); and the design's
"four existing closure groups" is **stale** — the shipped identity has six
terms (`tests/test_thermostat_books.py:70-92`).

Counts: **5 BLOCKING · 31 REQUIRED · 9 NOTE**, consolidated into the 30
numbered entries below (the first six of which are the cut-blockers — finding
2e is REQUIRED by severity but must land at P1, so it rides the blocking
group). Nine of the design's factual claims about the tree were checked and
hold; three do not, and one is overstated (§8).

---

## 1. Canonical-systems fidelity

The iron rule: *"Check the canonical-systems table below before building
anything new. If a system covers your need, use it; if it almost fits, extend
it — never build a parallel copy."*

### 1a. The integer extinction plane does not exist — **BLOCKING**

`heat_atten` is declared `np.float32` in the material table
(`src/simulation/materials.py:91`) and projected onto the grid as float32
(`src/simulation/gamemap.py:406`, `:1356`). The current march multiplies in
float: `heat_survival *= (1.0f - heat_atten[idx])`
(`cpp/src/raycaster.cpp:403`, twin at `cpp/src/cuda_raycaster.cu:127`).

§2.3's scheme is integer end to end and its positivity argument rests on
`a_i ≤ ONE` *as an integer*. So P1 needs a Q16 `heat_atten` plane that does not
exist. Under the **Q16 boundary modules** row (`src/simulation/*_fixed.py` per
field — "All Python↔field conversion; never hardcode 65536") that is a named
module, not an inline `* 65536`. Seven such modules exist
(`atmosphere_fixed`, `fire_fixed`, `gas_fixed`, `unit_fixed`, `wall_fixed`,
`water_fixed`, `wave_fixed`); the design adds none and mentions none.

**Required:** name the plane, its dtype, its boundary module (new
`optics_fixed.py`, or an extension of an existing one — this is a reuse-or-new
call), and the load-time door-2 quantization site. Say explicitly that the
float `heat_atten` stays only as long as `raycaster.cpp`'s render march needs
it, and dies with P6.

### 1b. `dyn_heat_atten`'s number type and digest membership are unstated — **REQUIRED**

The design's model is `dyn_light_atten`, and it is right about the MAX stamp
(`cpp/src/physics_engine.h:476`, `cpp/src/physics_engine.cpp:981`). But
`dyn_light_atten` is `float32 (h,w,3)` (`src/simulation/gamemap.py:417`) and is
**deliberately excluded** from the cross-GPU digest
(`tests/field_digest.py:19`, `:104` — `EXCLUDED_FLOAT_FIELDS`). The heat twin
cannot inherit that: it feeds a digested channel.

So `dyn_heat_atten` is a *different kind of object* from its model — an integer
synced-input plane. The design must say: its dtype; whether it rides
`field_ab_harness.SIM_FIELDS` (its model does, `tests/field_ab_harness.py:87`);
and whether it enters `DIGEST_FIELDS`. If it enters, that is a
`DIGEST_SPEC_VERSION` bump **plus regeneration of every golden in the same
commit** — `tests/field_digest.py:39-44` and the CLAUDE.md "Field digest" row
both say so.

Also: `stamp_units` currently writes exactly three dynamic planes
(`cpp/src/physics_engine.cpp:940-981`, signature at `physics_engine.h:496`).
Adding a fourth changes that signature, the always-upload set
(`src/simulation/gamemap.py:137`), the Python wrapper
(`gamemap.py:1824`), and `tests/test_stamp_units_cpp_ab.py`.

### 1c. The Fleck divide must come from the fixed-point kit — **REQUIRED**

CLAUDE.md: *"Fixed-point kits | `cpp/src/fixed_point.h` + `cuda_fixedpoint_device.cuh`
| The only sim arithmetic, CPU and device — never re-derive a shift/round/reciprocal."*

The kit already ships three reciprocal forms: `make_recip`
(`cpp/src/fixed_point.h:235`, load-time pin) + `recip_mul` (:244), and the
per-tile Newton `reciprocal_q16` (:458) — which is exactly what the temperature
solver's own per-tile gas divide uses (`cpp/src/temperature_solver.cpp:347`).

§7.1 instead says *"One divide per cell per tick enters, for the Fleck factor.
It is exact if the operands are pinned, which is the existing house rule."*
That is a float divide by any reading, and it is a canonical-systems violation
before it is anything else. See also §5a below for why it also contradicts
§7.1's own headline.

**Required:** specify `reciprocal_q16` (or a `make_recip` pin if the
denominator turns out to be per-material rather than per-cell), and state its
rounding: it is **truncated toward −∞ and not correctly rounded** — the header
says so in terms — "it is NOT correctly-rounded (a Newton fixed-iteration
cannot be), but it is DETERMINISTIC and converges to within ~1 ULP"
(`cpp/src/fixed_point.h:444-450`). That matters, because `f_i` multiplies a
synced source term and the §2.9 gate-2 fixed point is asserted at zero.

### 1d. `radiation_sweep.*` versus "lands here, never beside it" — **REQUIRED**

The design's own Systems section says the ray engine
(`cpp/src/raycaster.*` + `cuda_raycaster.cu`) is a system it must use, and that
*"the replacement lands **here**, never beside it"* — copied verbatim from the
survey. Two lines later the New Systems list creates
`cpp/src/radiation_sweep.*`. Those cannot both be true.

The staged migration is defensible (P1 builds the sweep, P3 flips heat over,
P6 flips light), but for P1→P5 the tree genuinely carries two radiation paths,
and the design must say so out loud and say when the duplication ends. It must
also answer where the `E°` bake goes: it is today a **method on `Raycaster`**
(`cpp/src/raycaster.cpp:62-97`) whose `rad_scale` is a live attribute set from
Python at `src/simulation/physics_runner.py:383` and re-baked at `:410`. §2.6
says the table "survives unchanged" without saying who owns it after P3.

**Required:** either rename the new TU as the ray engine's new home and update
the canonical row at P3, or state the two-path window explicitly with its end
date. Name the `E°` table's post-P3 owner.

### 1e. The new sim TU is not on `/fp:strict` and not on the float ratchet — **REQUIRED**

CLAUDE.md iron rule: *"New C++ sim TUs go on the `/fp:strict` list in
`cpp/CMakeLists.txt`, always."* The list is
`cpp/CMakeLists.txt:177-189` (eleven TUs, `raycaster.cpp` among them since
CUDA-S2). The design never mentions it.

Worse, the float ratchet does not cover the new TU either.
`tests/test_no_float_in_sim_tu.py:49-56` lists six `SIM_TUS` and the comment
above them (`:47-48`) reads *"Render-only TUs like raycaster.cpp and the
pure-glue bindings.cpp are NOT in scope — they are not part of the synced
lockstep state."* That comment is already stale (`raycaster.cpp` has written
the synced `rad_net` ledger since P-R4), and if `radiation_sweep.cpp` is not
added to `SIM_TUS` + `BASELINE` at `{0, 0, 0}`, the ratchet is **vacuous on the
one TU the whole arc's determinism claim rests on**.

**Required:** both list additions, named in P1's deliverables, plus a one-line
correction to that stale comment.

### 1f. Where the sweep is *called from* is never stated — **REQUIRED**

CLAUDE.md: *"PhysicsRunner / PhysicsEngine | The only solver callers; new C++
orchestration lands in PhysicsEngine, not Python glue."* Survey §9.6 says heat
goes in *"C++ `PhysicsEngine` + CUDA twin, like the other seven solvers."*

Today the cast is Python glue, at **two** call sites:
`src/simulation/physics_runner.py:819` (normal path) and `:1207` (the resident
path, explicitly "on the mirror"). The design says nothing about either. If the
sweep lands in `PhysicsEngine::step` as the canonical row demands, it needs a
numbered step and the `cast_fire_heat` Python entry point dies — which is a P3
deletion the design's deletion list does not contain.

### 1g. The resident path is not addressed at all — **REQUIRED**

CLAUDE.md: *"RL-batch habits | `docs/rl_env_arc_proposal_2026-08-27.md` §A |
New resident-path code follows §A (Erik's ruling 2026-08-27): new
`*_launch_resident` born `(N, h, w)`-shaped (N=1 today), no new host-side tick
logic, no new mirror-only fields, per-env masks over host gating."*

The sweep is new resident-path code by construction, and Φ (§4b) is a candidate
"new mirror-only field" — exactly what §A forbids. The design mentions
residency nowhere. At minimum P4's row must carry the §A constraints.

### 1h. The A/B lockstep harness is missing from every gate list — **NOTE**

CLAUDE.md: *"A/B lockstep harness | `tests/field_ab_harness.py` | The refactor
gate — never prove a refactor with whole-grid means."* Neither §2.9 nor §9
mentions it. P3 in particular ("flip heat over") is the archetypal refactor
this gate exists for.

### 1i. Two draft rules are too wide — **NOTE**

- *"Per-cell extinction … every optical interaction — material, unit body,
  smoke — is a per-cell coefficient on these planes and nothing else"* is
  violated on day one by the weapon beam, which carries its own per-gas
  absorption table (`GasTable.beam_absorb_q16`, `src/simulation/gases.py:148`,
  consumed at `src/simulation/combat.py:1084`, `:1127`) — and §4 correctly
  declares beams out of scope. Scope the rule to the sweep's channels.
- *"The light-field accessor … the renderer read[s] light through the
  accessor, never by calling a solver"* collides with two existing rows:
  `LightingPass` ("New lit passes sample `light_tex_a/b` — never a second
  raycast") and `frame_lights.py` ("The only per-frame light-list assembly").
  Today `LightingPass` *is* the solver caller
  (`renderer/lighting.py:293`). Say what the accessor does to that role.

### 1j. `pack_hover_readout` is listed but unused — **NOTE**

It is named in Systems as a canonical system the design must use, but the
design never adds a readout row. `renderer/hover_readout.py:107-160` carries no
radiation quantity today. Φ, `f_i` and "clamp bound here" are exactly the
per-tile debug the calibration in P2 will want, and the row exists precisely to
stop a parallel field probe being rolled.

### 1k. What the design gets right here

- `dyn_light_atten` as a per-tick **MAX** stamp — correct
  (`cpp/src/physics_engine.h:476`).
- The `face_flux` idiom as the conservative pattern to copy — correct, and the
  design cites it in the right place.
- "the temperature solver's Pass-1 fold, the only place radiation becomes
  temperature" — correct, and it is the statement that resolves §2's confusion
  if the design lets it.
- The god-file policy is **not** violated: the sweep sits inside physics slot 7
  and the conductor gains at most one clear line. Worth stating so no reviewer
  re-raises it.

---

## 2. The arc-#54 energy ledger

### 2a. "Four existing closure groups … there is no fifth" is stale — **REQUIRED**

The design repeats this in §5.3 and twice in Systems. The shipped identity has
**six** terms, enumerated in `tests/test_thermostat_books.py:70-92` and summed
at `:107-114`:

| # | group | counters |
|---|---|---|
| 0 | EOS (absolute) | `e_entry_resync_sum`, `e_transport_net_sum`, `e_wipe_sum`, `e_kick_ke_sum`, `e_drag_heat_sum`, `e_work_export_sum`, `e_rail_sum` |
| 1 | thermal solver, **gas** side | `e_gas_deposit_sum`, `e_gas_cond_sum`, `e_gas_rail_sum` |
| 2 | combustion | `e_comb_draw_sum`, `e_comb_deliver_sum`, `e_comb_heat_sum`, `e_comb_rail_sum` |
| 3 | python seams | `GameMap.gas_energy_seam_net()` |
| 4 | water evacuation (absolute) | `e_water_evac_export_sum` |
| 5 | thermal solver, **solid** side (P-G5) | `e_solid_deposit_sum`, `e_solid_cond_sum`, `e_thermostat_sum`, `+ comb.e_comb_solid_heat_sum` |

CLAUDE.md's own "four groups" phrasing is the stale source. The design should
say **six** and cite the test, and the arc should amend the CLAUDE.md row at
close.

### 2b. §5.3 names the wrong writer, and the wrong group — **BLOCKING for P7**

§5.3: *"letting radiation warm gas makes the sweep a **booked writer on the gas
energy seam**, needing its own counter and a term in one of the four existing
closure groups — the thermal-solver gas side, which already routes through the
seam."*

Two things are wrong and they point an implementer in opposite directions.

- **The sweep must not write `gas_energy` at all.** If
  `radiation_sweep.cpp` calls `gas_energy_*`, that is a genuine new group — it
  is not the EOS, not the thermal solver, not combustion, not a Python seam.
  The design's phrasing ("the sweep [is] a booked writer on the gas energy
  seam") describes exactly that.
- **The correct architecture already exists and needs no new seam writer.**
  The sweep writes `rad_net`, as it does today. The temperature solver's Pass-1
  fold gates the radiation branch on `ts[i]`
  (`cpp/src/temperature_solver.cpp:247`) — that mask is the *only* thing
  stopping gas from receiving radiation. Its gas branch, two dozen lines down,
  already deposits through the seam and books itself:
  `gas_energy::deposit_railed(...)` then `e_gas_deposit_sum += nb * dT`
  (`cpp/src/temperature_solver.cpp:373-388`), with `e_gas_rail_sum` for the
  rail. So radiation-into-gas is **group 1, with the counters that already
  exist** — a mask change, not a new channel.

**Required:** rewrite §5.3 to say the sweep produces `rad_net` and the Pass-1
fold's `ts[i]` mask is what changes; "the sweep is a booked writer on the seam"
must go.

### 2c. The gas branch double-counts density — **REQUIRED**

That same gas branch applies the v2.4 absorption-proportional law
`E_abs = deposit · min(N, N_AMB)/N_AMB`
(`cpp/src/temperature_solver.cpp:344-349`), with its own counted destruction
`e_deposit_drop_sum += deposit − e_abs` (:353-358).

Under R-M the sweep will already have applied a per-cell extinction `a_i`
derived from smoke/gas density. Routing radiation through this branch applies a
density factor **twice**. That is a ruling P7 needs, written down in the design
rather than discovered in the lab: either the radiative deposit bypasses the
v2.4 factor (a second entry point into the gas branch) or `a_i` is defined so
the composition is the intended law.

### 2d. The clamp's book is the wrong idiom, and on gas it is forbidden — **BLOCKING**

§2.8: *"the energy it removes books to a **named rail counter** in the
`eos_solver` Pass-A donor-rail idiom."*

On **solids** the right idiom is next door and cheaper. Pass 1 already books
the *actual applied* ΔT × capacity, post every rail:

```
e_solid_deposit_sum += ((int64_t)temperature[i] - t_before_rad) * cap_real_[i];
```
(`cpp/src/temperature_solver.cpp:297-299`, and the header explains at `:251-253`
"why this needs no separate rail counter"). A clamp applied inside that block is
booked automatically; the *engagement count* rides the existing
`t_max_phys_hits` / `t_low_rail_hits` pattern
(`cpp/src/temperature_solver.h:395`, `:404`).
A clamp applied anywhere else — in `radiation_sweep.cpp`, say — writes
`temperature` outside the P-G5 ledger and the solid identity stops closing.

On **gas** the clamp as written is forbidden outright. CLAUDE.md, "Gas
temperature is a mirror": *"NOTHING writes a gas cell's `temperature` directly
— tests, tools and benches included … A bare `temperature[...] =` now moves no
books at all, so it does not fail loudly — it goes VACUOUS (incident: three
test fixtures at P-G1b, one of which had stopped driving any wind)."* §5.3
insists "The Fleck factor and the clamp both apply to gas cells", so this is
not hypothetical. On gas the clamp must be expressed as a `gas_energy` change
through `gas_energy::deposit_railed` (or a sibling), with the mirror refreshed
by the seam — never as `T_i ← E°⁻¹(Φ_i)`.

### 2e. §2.3's identity is missing the unit-absorption exit — **REQUIRED**

§2.3 states `Σ ΔE_mat + Σ sky_out − Σ sky_in + Σ ceiling ≡ 0` and §2.9 gate 1
gates it exactly in int64. §5.2 then makes units absorb from the stream into
`rad_flux`.

`rad_flux` is documented as outside the ledger, and the reason is that nothing
is debited for it: *"**\*** NOT part of the energy ledger. **\*** It moves no
energy, changes no temperature, and nothing is debited to pay for it"*
(`src/simulation/gamemap.py:501-503`). §5.2 changes that — the absorbed integer
*is* debited from the stream. So `rad_flux` becomes a ledger **exit**, and the
identity needs a `− Σ unit_absorbed` term.

Calling it "an already-accepted, named leak" does not discharge this: the leak
that was accepted was an *undebited sensor*, which is a different object.
**Required:** put the term in §2.3 at P1 — not at P5 — or gate 1 is built
without it and fails the day P5 lands.

### 2f. The sweep's exactness does not survive the fold — **REQUIRED**

§1 promises *"an integer, exactly conservative heat channel inside the sim."*
The sweep's books are exact; the *channel* is not. The Pass-1 fold truncates:
`dTr = shr_round0(rn, heat_inv_shift[i])` (`cpp/src/temperature_solver.cpp:255`),
and then the T_MAX_PHYS and low rails clamp (:262-294). The solid ledger stays
closed because it books the *applied* ΔT, not `rad_net` — but that means energy
`rad_net` carried and the fold dropped is simply gone, counted only as the
difference between two ledgers.

This is the pre-existing design of the fold and is not a v2 defect. It *is* a
place where the design overclaims. **Required:** state the boundary — "§2.3's
identity holds on the sweep's own books; the sweep→fold shift is the existing
lossy boundary and the solid ledger absorbs it at the rail."

### 2g. The Fleck factor genuinely needs no counter — **correct, briefly**

§2.8's "Applied **before the sweep**, so conservation is untouched … the
arc-#54 books need no new counter for it" is right. A smaller source is not a
leak; nothing is removed from any ledgered quantity. Granted in full.

---

## 3. Migration of the old raycaster's consumers

Survey §10.2 named this the risk that *"must be decided before the first patch,
not during the last."* The design's total treatment is one sentence in §7.2
(one test) plus one sentence at the end of §9 ("The old raycaster's other
customers keep their interfaces throughout"). Here is what is actually out
there.

### 3a. The test surface is ~46 pytest tests, not one — **REQUIRED**

Four files encode the *deleted* law as property gates:

| file | tests | what it pins |
|---|---|---|
| `tests/test_pf1a_radiation_books.py` | 11 | rule 1 pairs, rule 2 half-weight, rule 3 contact faces, rule 4 sky, the emitter gate, `rad_net + rad_amb == 0` |
| `tests/test_pr1_fire_plane_cast.py` | 11 | the ledger identity, air-inertness, the `rad_flux` reach |
| `tests/test_fire_heat_source.py` | 14 | the 8-ray fan, `range_base + range_per_intensity`, occlusion |
| `tests/test_heat_attenuation.py` | 10 | `heat_atten` as the 4th march channel |

Named assertions that die outright, beyond the one the design cites:
`test_pf1a_radiation_books.py:217` (`rad_net.sum() + rad_amb.sum() == 0`),
`:242-243` (the sealed-room sky assertion — the one §7.2 names),
`:399-443` (per-tile `rad_net == 0` on an isothermal lattice),
`:516-524` (rule 2's mutual-vs-oneway ratio),
`:639-651` (the emitter gate's knife edge at `T_emit_gate ± 1` count),
`:914-921` (per-ray magnitude = total ÷ `RAY_COUNT`);
`test_pr1_fire_plane_cast.py:255-263` (lone emitter books sky to exactly one
tile), `:335-340`, `:393-434` (see 3d), `:558-576`.

Three CUDA check scripts are on the same law:
`tests/cuda_pr1_fire_plane_check.py` (its `DIALS` at `:50` hardcode
`fire_ray_count=8, range_base=2.0, range_per_i=3.0`),
`tests/cuda_s2_check.py`, `tests/cuda_s2b_raycaster_live_check.py`.

Per CLAUDE.md's test rule — *"every test names the property it protects and the
change that must break it … a failing pre-existing test is a finding to report,
never a target to bend the design to"* — each of these needs an explicit
disposition: rewritten to the new property, or deleted with rationale. That is
a patch-sized piece of work in itself and it is currently invisible in §9.

### 3b. Five tools and benches break by name — **REQUIRED**

- `tools/storm_ledger.py:291` wraps `runner.cast_fire_heat` **by name** through
  `getattr(runner, nm)` (:293). Renaming or removing it raises `AttributeError`
  and the instrument dies. CLAUDE.md: *"Benches … Reuse the existing instrument
  before writing a new one."*
- `tests/bench_s8c_fire_heat_check.py:108` reads `runner.fire_ray_count`;
  `:115` reads `runner.fire_range_base + runner.fire_range_per_i * intensity`.
- `tools/fire_tune_loop.py:515` and `tools/storm_probe.py:66` hardcode
  `T_emit_gate = 310.0` as a dial override.
- `tools/fire_timing_harness.py:259` documents the tick order by that function
  name.
- `tools/fire_tuning_lab.py` — the instrument the design's Systems section
  names — contains **no radiation dial at all** (`grep -n
  "rad_scale\|T_emit_gate\|radiation\|fire_ray" tools/fire_tuning_lab.py`
  returns nothing), and its `IGNITE_TILES` default is a single station
  (`:60`). P2's "reach-curve bench" is therefore a real extension with a
  multi-probe design, not a dial addition. Survey §9 said as much; the design
  dropped it.

### 3c. P3 deletes `light_cull` while the renderer still needs it — **REQUIRED**

`light_cull` is the render march's per-channel survival threshold
(`cpp/src/cuda_raycaster.cu:96-97`, bound at `cpp/src/bindings.cpp:2111`,
documented at `:2109` as "ε_rgb (render)"). §9's P3 row deletes
`heat_cull`/`light_cull` together. But light does not move to the sweep until
**P6**. Deleting `light_cull` at P3 breaks the shipped render light path for
three patches.

### 3d. P3 deletes the `rad_flux` reach gate it promises to keep — **REQUIRED**

§9's closing line: *"`rad_flux` consumers unchanged."* §9's P3 row deletes
`range_base`/`range_per_intensity`. Those two dials **are** `rad_flux`'s reach:
assumption A7 of the inventory records it, and there is a test whose entire
premise is that the reach was deliberately preserved —
`tests/test_pr1_fire_plane_cast.py:401`: *"But D3's `rad_flux` sensor
DELIBERATELY KEPT THE OLD REACH (v7.1 item 4): it is written at AIR cells
behind a deterministic `damage_range` guard that is exactly the legacy
`range_base + range_per_intensity * I`."* The two statements cannot both stand.

### 3e. P3 deletes the 16-light cap three patches before it is safe — **REQUIRED**

The cap and its NMS live in `renderer/fire_lights.py:13-15` (the pipeline
comment), `:66` (`max_lights: int = 16`), `:82-91` (NMS then brightest-K) —
a **render-only** module behind the canonical "Frame lights" row. P3's gate is
"full suite + one deliberate golden re-baseline", which cannot see a render
regression at all.

Removing it at P3 means every burning tile becomes a `LightSource` on the
**old per-source CPU march**, which is the expensive one: survey §5.1 measures
34,758 DDA steps per frame *at* the cap, and survey §5.3 records that *"there
is no GPU path for the render light cast at all."* The cap becomes unnecessary
only when P6 makes light a field solve.

**Required:** move the cap/NMS deletion (and `light_cull`) out of P3 and into
P6, where the HUMAN-TEST gate lives.

### 3f. "`rad_flux` → unit heat damage: Consumer unchanged" is false in substance — **REQUIRED**

The consumer is `src/simulation/exchange.py:299-340`. It reads
`max(heat[ty,tx], rad_flux[ty,tx])` over the unit's footprint (`:316-336`),
dequantizes (`phi = peak_raw / HEAT_SCALE`, `:339`) and applies five `[combat]`
dials: `unit_absorption`, `unit_reflectivity`, `heat_flux_to_temp`,
`heat_ambient_ref`, `heat_overtemp_scale` (`:303-310`).

§5.2 changes the *quantity* `rad_flux` carries — from incident
`τ·w·a_s·E°[T_s]` per ray to *absorbed* energy summed over 16 ordinates. The
Python call signature is unchanged; the calibration is not, by an unknown
factor. The `max()` against `heat` also needs a ruling: `heat` is combustion's
deposit at burn sites, and "the larger of an incident flux and a deposit" was
coherent when both were incident-flux-shaped.

This is the "Coupling table" row (`src/simulation/exchange.py` — "a
physics→unit coupling is one row, not plumbing") and P5's "feel" risk label is
right. The doc's "Consumer unchanged" is what is wrong.

Related: §5.2's *"the interface must be per-unit absorbed flux, not a hardcoded
flux→damage"* mis-describes today. The interface is already a per-tile plane
read per unit with per-unit absorption/reflectivity applied in `exchange.py`;
it is not hardcoded. The upgrade §5.2 wants (a scalar unit temperature) is
additive against what exists.

### 3g. Consumers correctly left alone, but never said so — **NOTE**

- `src/simulation/vision.py:88-103` (`ray_clear`) walks
  `gmap.has_los` (integer Bresenham) plus cover's slab test
  (`src/simulation/cover_system.py:84-100`). Neither touches the raycaster's
  DDA. §7.3's "should stay a separate primitive" covers this by implication;
  survey §10.2 asked for it explicitly. One line in §4's table closes it.
- `gmap.light_map` (`src/simulation/gamemap.py:380`, "legacy: fire raycaster
  output + render unit/smoke tinting") is unmentioned anywhere.
- Weapon beams: correctly out of scope (`src/simulation/combat.py:1057-1200`,
  its own integer Beer-Lambert with `beam_absorb_q16`).

### 3h. A latent float in the synced heat path, which §6.1 would repeat — **REQUIRED**

`cast_fire_heat` passes `gmap.dyn_light_atten` — a float32 plane excluded from
the cross-GPU digest — into the heat-only cast today
(`src/simulation/physics_runner.py:1630`, `:1660`). That is a pre-existing
oddity, not the design's.

But §6.1 says the light channel *"Reads the same `dyn_light_atten` and gas
tables the heat channel reads, so the two cannot disagree about geometry."*
Under R-Q the **rules-side light payload is integer and gameplay-visible**
(P8). Deriving synced state from a float plane that is deliberately outside the
cross-GPU contract (`tests/field_digest.py:19`, `:104`) is a determinism hole
the design does not see. Either the rules channel reads an integer extinction
plane (the same one heat reads), or `dyn_light_atten` gains an integer twin.
State which, in §7.2, before P8.

---

## 4. Tick order and seam placement

### 4a. The slots 3/6/7 claim is TRUE — verified

`src/simulation/simulation.py:1392` — `# 6. Re-stamp obstacles.` →
`self.gmap.stamp_units(self.units)`; `:1405` — `# 7. Physics.` →
`self.physics_runner.step(...)`. Units move in the slot-3/4 unit-simulation
block (`:1356`). So the stamp is frozen before the physics step, and the design
is right. Granted.

The caveat is 1b: the *plane the sweep reads* does not exist yet, so what is
frozen at slot 6 today is three planes, not four.

### 4b. Φ, the clamp's carrier plane, is undeclared — **BLOCKING**

The sweep runs at the start of physics (today: `physics_runner.py:819`, before
`_step_water` and the atmosphere loop). The material update is
`PhysicsEngine::step` **step 3**, the temperature pass
(`cpp/src/physics_engine.cpp:217`, `:371`) — after the fire step at step 2
(`:113`). §2.8's clamp fires *"after the material update"* and needs `Φ_i`, the
fluence that cell absorbed during the sweep.

So Φ must be a new per-tick plane carried across the whole physics step. That
means: a `GameMap` allocation next to `rad_net`/`rad_amb`/`rad_flux`
(`src/simulation/gamemap.py:479`, `:499`, `:514`); a dtype (int64, by §2.3's
own magnitude argument); a pybind signature; a CUDA upload; and a clear line in
the conductor beside the other three
(`src/simulation/simulation.py:1597-1616`, "cleared together at the very end of
Simulation.step"). None of that is in the design, and it is P1 scope because
P1 ships the clamp.

Note this also collides with 1g: "no new mirror-only fields" on the resident
path.

### 4c. Gate 5 has no stated evaluation point — **REQUIRED**

*"5. Maximum principle: no cell exceeds `E°⁻¹` of the fluence it receives."*
Evaluated when? If the clamp is inside Pass 1, then Pass 2 conduction
(`cpp/src/temperature_solver.cpp:400`ff), Pass 3's two-way thermostat
(`:551-660`) and combustion's direct `temperature[s]` write (the one
`e_comb_solid_heat_sum` exists for — `cpp/src/combustion.h:511-517`) all run
afterwards and can legitimately lift a cell above the clamp. Measured at
end-of-tick the gate is falsely red; measured immediately after the clamp it is
close to tautological. State the point and what makes it non-vacuous.

### 4d. Gate 5 is missing from P1's gate list — **BLOCKING**

§9's P1 row: *"conservation, second law, positivity, stability rail, isotropy —
all exact, all non-vacuous."* §2.9 lists eight gates. The maximum principle is
the one P1 introduces (the clamp is explicitly "included" in P1) and the one
whose absence §2.8 says produced a **1.73-million-game-unit** equilibrium
error. Shipping the clamp in P1 without its gate is the exact shape of the
mistake critique 1's required change 12 fixed for isotropy.

### 4e. The god-file policy is not violated — **NOTE, in the design's favour**

The sweep is inside physics slot 7; the conductor
(`Simulation.step`) gains at most one `fill(0)` line for Φ. No logic block.
Worth one sentence in §9 so it is not re-litigated.

---

## 5. Determinism and the number doors

### 5a. §7.1 contradicts itself in three lines — **REQUIRED**

> *"Doors 1 and 2 only. … The float ratchet is untouched: the sweep adds no
> float to a sim TU. … One divide per cell per tick enters, for the Fleck
> factor. It is exact if the operands are pinned, which is the existing house
> rule."*

A float divide is **door 3** by name. `docs/architecture/engine/14_...md`
§3, Door 3: *"Audited algebraic float bridges — Float chains restricted to
`+ − × ÷ √` on deterministic inputs, compiled under the pinned floor … Door 3
is a *concession*, not a preference — prefer door 1; every bridge is one audit
away from a leak."* An integer divide is door 1 but then it must be the kit's
(§1c), and the doc must say so.

Either way the sentence "doors 1 and 2 only" and the sentence "one divide per
cell per tick" cannot both survive as written. **Required:** say `reciprocal_q16`,
door 1, truncated rounding — and then "doors 1 and 2 only" becomes true.

### 5b. None of the three guards currently covers the new TU — **REQUIRED**

The design's determinism assurances rest on gates that, today, do not reach
`radiation_sweep.cpp`:

1. **The ingress lint** — `tests/test_ingress_lint.py` is a Python AST scan
   over `src/simulation/` (`:45`), and its own docstring lists the gap:
   *"Known v1 gaps (documented, not enforced): … C++-side ingress (governed by
   /fp:strict + the digest gates)."* It will never see the sweep.
2. **The float ratchet** — see 1e: `SIM_TUS` does not and would not include the
   new TU unless it is added.
3. **`/fp:strict`** — see 1e: the `set_source_files_properties` list at
   `cpp/CMakeLists.txt:177-189`.

So the "CPU↔GPU bit-identity, tol 0" gate (§2.9 item 7, P4) is the *only* live
guard, and it arrives three patches after the arithmetic is written.
**Required:** P1 adds the TU to (2) and (3), and says so in its deliverables.

### 5c. `a_i ≤ ONE` as an ingress invariant — **NOTE, right instinct**

The right site is `src/simulation/materials.py` — inside the lint's scope, and
it already carries the `ingress-exempt:` precedent the lint documents
(`test_ingress_lint.py:24-28`). The validated-range precedent is the filter
table (CLAUDE.md: *"a filter is a table row (per-gas efficiency, validated
[0,1])"*). Say both in §2.3 so the implementer copies rather than invents.

### 5d. "wrapping" at int64 — **NOTE**

`rad_net`'s order-freedom argument is recorded and correct:
*"PLAIN SIGNED adds (wrapping). … under SATURATION that sum would be
order-DEPENDENT and the CPU↔CUDA tol-0 gate would break"*
(`src/simulation/gamemap.py:469-478`). At int32 the wrap is defined by the
numpy/C++ two's-complement convention the comment relies on; at int64, signed
overflow is UB in C++. Since §2.3 widens, say "int64 with a stated headroom
argument" rather than inheriting the word "wrapping".

### 5e. Door 4 is genuinely untouched — **correct**

*"neither the sweep nor anything in this design needs a random number, so
ingress door 4 is untouched and Philox stays a swarm-units concern"* — matches
survey §10.3's verification. Granted.

---

## 6. Contract extensions

### 6a. "A Recorder DTYPE-class contract extension" is FALSE — **REQUIRED**

§2.3: *"`E°` and the stream widen to int64. … This is a Recorder DTYPE-class
contract extension with its own ring branch, not a cast."*

The Recorder does not record these fields at all.
`src/simulation/recorder.py:81-83`:

```
DEFAULT_FIELDS = ('atmosphere', 'temperature', 'gas_o2', 'inert_n2',
                  'wind_x', 'wind_y', 'smoke', 'fire', 'obstacles',
                  'gas_energy')
```

and `_INT64_FIELDS = ('gas_energy',)` (`:94`). Neither `rad_net` nor `rad_flux`
appears, and nothing in the tree passes them in `fields=` (the only
non-default construction is `tools/tabs_pw2_venting_capture.py:99`, which uses
the defaults).

What the widening actually touches: `GameMap`'s dtypes
(`src/simulation/gamemap.py:479`, `:514`), four pybind signatures
(`cpp/src/bindings.cpp:609-611`, `:2276-2278`), the CUDA header
(`cpp/src/cuda_raycaster.h:108`ff), and every test harness that allocates
`np.int32` planes (`tests/test_pf1a_radiation_books.py:100-102`,
`tests/test_pr1_fire_plane_cast.py:146-148`,
`tests/test_fire_heat_source.py:85-89`,
`tests/cuda_pr1_fire_plane_check.py:76`).

The `_INT64_FIELDS` rule *would* apply if a future session recorded them; the
design should say that rather than claim the extension is being made now.

Also `E°` is **already int64** (`cpp/src/raycaster.cpp:62-97`; `e_table_` is a
`std::vector<int64_t>`). Only the stream and the planes widen.

### 6b. "digested? yes" is loose — **REQUIRED**

§3's payload table says the heat payload is digested. It is not, and the reason
is recorded: *"(The two per-TICK radiation planes `rad_net` and `rad_flux` are
deliberately NOT here: both are wiped at end of tick, so they are identically
zero at every snapshot point and would add nothing.)"*
(`tests/field_digest.py:50-52`; `DIGEST_FIELDS` at `:78-99`). The heat channel
reaches the digest only through `temperature` and `heat`. Restate the column so
nobody plans a spec bump that is not needed — or plans one that is (Φ, 4b, is a
per-tick plane and inherits the same argument; `dyn_heat_atten`, 1b, does not).

### 6c. `rad_amb` as "a global scalar" reopens a settled decision — **REQUIRED**

§7.2 offers *"a global scalar or a per-exit-cell plane."* The plane was chosen
for a recorded reason:

> *"WHY A PLANE AND NOT A SCALAR: a single global counter would be a contended
> atomic on the device and — worse — an ORDER-DEPENDENT one if it ever
> saturated. A per-tile int32 with PLAIN adds is order-free by the same
> argument `rad_net` uses, and the host reduces it to a uint64 total once per
> tick."* (`src/simulation/gamemap.py:487-494`)

The design may still choose the scalar, but it has to engage that paragraph.

### 6d. The per-emitter-attribution loss is bigger than one test — **REQUIRED**

§7.2 names `tests/test_pf1a_radiation_books.py`'s sealed-room assertion. The
full list is in 3a. Two specifically depend on *attribution*, not just on the
sum: `test_pr1_fire_plane_cast.py:255-263` asserts the sky is booked to exactly
one tile (the emitter's), and `test_pf1a_radiation_books.py:756` asserts
`rad_net[15,15] == -rad_amb.sum()` — emitter-keyed by construction. Both die,
not merely move.

### 6e. The P8 digest entry is a spec bump, not a re-baseline — **REQUIRED**

§7.2: *"Entering the digest is a one-way door. One deliberate re-baseline on
the day the stealth rules land."* Adding a field to `DIGEST_FIELDS` is a
`DIGEST_SPEC_VERSION` bump **and** regeneration of every committed golden in
the same commit — `tests/field_digest.py:39-44` states the procedure, and the
CLAUDE.md "Field digest" row states it as a rule. Say "spec bump + regenerate
all goldens", which is a larger and more scheduled thing than a re-baseline.

---

## 7. The patch plan (§9)

### 7a. P1 — not cuttable as scoped

Blocked by 1a (no integer extinction plane), 2d (the clamp's home), 4b (Φ), and
4d (gate 5 missing). Add 1e (`/fp:strict` + ratchet) and 2e (the unit-absorption
term in the identity, so gate 1 is built once) as deliverables.

One more thing P1 must state: **is P1 wired?** "shadow plane" implies the
sweep's output is computed and not applied. If so, say it — because unwired
means the stability and maximum-principle gates run against a harness rather
than the engine, and wired means **P1**, not P3, moves the golden. §9's P3 row
is the one that carries "one deliberate golden re-baseline", which reads as the
unwired answer; make it explicit.

### 7b. P3 — mis-sequenced and too large to gate

- Depends on things later patches deliver: `light_cull` (3c) and the 16-light
  cap/NMS (3e) are P6's to delete; `range_base`/`range_per_intensity` (3d)
  belong with P5's `rad_flux` rework.
- Its stated gate ("full suite + one deliberate golden re-baseline") cannot
  see what it actually breaks: ~46 pytest tests whose *property* no longer
  exists (3a), three CUDA check scripts, five tools (3b), and a render
  regression the golden does not cover (3e).
- **Required:** split P3. A "delete the old heat law + rewrite its property
  gates" patch, and a separate render-deletions patch folded into P6 behind the
  HUMAN-TEST gate P6 already carries.

### 7c. P6 — the clock and the language are unstated, and R-P drifts

R-P: *"Split implementation: heat in C++ with a CUDA twin; **light in GLSL,
render-only**."* §6.1 makes light-for-the-eye *"the same sweep, step transport,
float"* and never says where it runs. §8's cost table gives "Light sweep, same
grid | the same again, per channel" — **per tick**. But the render light field
runs **per frame at up to 60 Hz** (survey §5.1), and there is no GPU path for
it at all (survey §5.3). At 256×512 that is 2.1 M cell-updates per frame, ~126 M
per second, on the CPU.

This is a budget gap, not a rounding one, and §8's otherwise-excellent budget
correction does not address it. **Required:** state the language (GLSL per R-P,
or C++), the clock, and a per-frame number.

### 7d. P7 — see 2b and 2c

Not "the sweep becomes a seam writer" but "the Pass-1 fold's `ts[i]` mask
opens", plus a ruling on the density double-count.

### 7e. Small — **NOTE**

P2's gate reads *"the §9.3 curve"*. There is no §9.3 in this document; it means
the **survey's** §9.3. Fix the cross-reference — a reader who follows it lands
in the patch plan.

---

## 8. Claims about the current code

### Checked and TRUE

| § | claim | evidence |
|---|---|---|
| 2.5/2.6 | `E°` is int64, 4000 buckets of 4 game units, `K⁴` by repeated integer multiplication, never `pow` | `cpp/src/raycaster.h:203-205`, `cpp/src/raycaster.cpp:81-92` |
| 2.7 | a 0 K sky leaves "a permanently-firing low-rail counter, which the temperature solver documents as a RED" | `cpp/src/temperature_solver.cpp:283-290`: *"a hit inside a gate run is a RED"* |
| 2.7 | `u_ambient = (0.18, 0.18, 0.22)`, a shader floor added to every lit surface | `renderer/lighting.py:105`, `shaders/lighting.fs:133` |
| 4 | units block light via `dyn_light_atten`, a per-tick **MAX** stamp | `cpp/src/physics_engine.h:476`, `cpp/src/physics_engine.cpp:981` |
| 4 | the 16-light cap and its NMS are render-side | `renderer/fire_lights.py:13-15`, `:66`, `:82-91` |
| 5.2 | units move at slot 3, obstacles re-stamp at slot 6, physics at slot 7 | `src/simulation/simulation.py:1356`, `:1392`, `:1405` |
| 5.3 | today `rad_net` is folded into solids only | `cpp/src/temperature_solver.cpp:247` (`ts[i]` mask) |
| 6.3 | the float light field is never digested, so swapping its producer cannot move a golden | `tests/field_ab_harness.py:81` (render buffers "intentionally EXCLUDED"), `tests/field_digest.py:104` |
| 8 | `tests/_eos_p3_bench.py` times the **whole `Simulation.step()`**, and its gate is written against 83 ms | `tests/_eos_p3_bench.py:213-215` (`t0 = perf_counter(); sim.step()`), `:370` (`p99 <= 0.25 * 83.0`), plus the ~50×120 ship-scale leg at `:373-378` |

On that last row: **the design is right and the survey is the wrong document.**
Survey §5.4's *"the atmosphere group alone measured 18.97 ms p99 at 160²"* is
the mischaracterisation; the fix belongs in the survey, not here. (The specific
1.6 / 9.78 ms figures were not re-run for this critique; the methodology claim
is what I verified.)

### Checked and FALSE

**8a. "(Commented in the table as of this session.)" — §5.1 — REQUIRED**

The furniture note claims a comment was added to `config.toml` this session.
`git status --porcelain config.toml` returns **empty** — the file is clean —
and `config.toml:1560` still reads:

```
conductivity = 0.0   # kappa 0 = no conduction face (like air); fire eats it via flammability instead
```

There is no "placeholder test crate" comment anywhere in the file
(`grep -n "placeholder" config.toml` → five hits, none on a material row). The
parenthetical is false; either make the edit or delete the claim.

**8b. `ret` carries the exact dimensional defect critique 1 fixed for `emitted` — REQUIRED**

§2.3's corrected listing:

```
    leaked     =  (i_in * k_leak_i) >> 16            // 1 Q16 factor, 1 shift  — OK
    ret        =  (E°[0] * w_m * k_leak_i) >> 16     // 2 Q16 factors, 1 shift — WRONG by 2^16
    emitted    =  (src_i,m * f_i * a_i) >> 32        // 2 Q16 factors, 2 shifts — correct
```

§2.5 states `w_m` is Q16.16, and `k_leak_i` is Q16 by the `leaked` line. So
`ret` is 65 536× too large, for precisely the reason critique 1's required
change 4 gave for `emitted` — and v2 fixed `emitted` and not this. It should be
`((E°[0]·w_m) >> 16 · k_leak_i) >> 16`.

It is dormant only because the leak coefficient defaults to 0 (§10 item 2), so
today it is a latent bug that arms the moment anyone tries the reach lever —
and it breaks the §2.9 gate-2 uniform-ambient fixed point when it does, since
`ret` is the term that exists to make that fixed point hold under leak.

The cited reference does not catch it: `sweep_ref.py:77` computes
`ret = E_amb * wt * kl[y, x]` in float.

**8c. "Verified in `sweep_ref.py`" is cited for an exactness claim the reference cannot make — REQUIRED**

§2.3: *"Conservation is structural … exactly in int64 … Verified in
`sweep_ref.py` to float roundoff (relative 1e-16) on randomised grids."*

`docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref.py` is a **float**
reference: `kl = np.broadcast_to(np.asarray(kleak, float), (h, w))` (`:41`,
`:116`, `:206`), and the whole sweep runs in float64. It proves conservation to
1e-16. It does not and cannot prove int64 exactness — critique 1's integer
probes did that, and they were throwaway.

So **P1's gate 1 has no committed integer reference to check against.**
**Required:** either commit an integer `sweep_ref_q.py` alongside, or say
plainly that P1's first deliverable is that reference. This is the "do it
first, not last" piece survey §10.2 named as *"the piece most likely to force a
redesign."*

### Checked and OVERSTATED

**8d. §2.8's 16000 / 60000 rows are outside the table's range — NOTE**

`T_MAX_PHYS = 16000.0` (`config.toml:186`, `cpp/src/temperature_solver.h:394`)
and the `E°` table spans `T_game ∈ [0, 16000)` in 4000 buckets, **saturating on
the last** (`cpp/src/raycaster.h:203-213`, `e_bucket_of`). So `E°⁻¹` cannot
resolve anything above 16000, and the "60000 game" rows describe states the
engine clamps away. The demonstration of headroom is fine; the design must
state `E°⁻¹`'s saturation behaviour, or gate 5 is undefined at the top of the
range — which is exactly where the plasma case the clamp exists for lives.

**8e. §5.2's "already-accepted, named leak" — see 2e**

---

## Suite state (context for gate 1)

Survey §10.1's first hard gate is *"`fire-12` must be green."* It is not, as of
this critique:

```
C:/Users/steen/anaconda3/python.exe -m pytest tests -q
  → 2 failed, 2451 passed, 4 skipped, 4 xfailed in 200.36s

  FAILED tests/test_cool_shift_axis.py::test_every_material_carries_the_column_seeded_at_the_old_global
  FAILED tests/test_cool_shift_axis.py::test_a_crate_grid_from_config_is_uniform_today_but_addressable
  AssertionError: materials.wood.cool_shift is 13, expected the seeded 5.
```

**Good news, and better than the survey's §10.1 stocktake implies**: both reds
are in one file and neither is in the radiation path. Nothing in
`test_pf1a_radiation_books.py`, `test_pr1_fire_plane_cast.py`,
`test_fire_heat_source.py` or `test_heat_attenuation.py` is red — the 46 tests
§3a says P3 must dispose of are all currently *green*, which is exactly the
condition that makes P3's blast radius measurable.

Both failures are **snapshot tests of the kind CLAUDE.md's 2026-09-08 rule
forbids** — they pin a config value against the day they were written, and the
first one's own assertion message concedes the re-tune may be intended ("If
this is an intended re-tune … this test must be updated together with a
HUMAN-TEST play session"). That is a finding for the fire-12 session, not for
this arc; recorded here only because P1's readiness is gated on green.

---

## REQUIRED CHANGES, in priority order

**Blocking — P1 cannot be cut until these are in the document.**

1. **Declare the integer extinction plane** (§1a): dtype, `*_fixed.py` boundary
   module, load-time quantization site, and the fate of the float `heat_atten`.
   §2.3's arithmetic has no inputs without it.
2. **Move the clamp into the temperature solver and split it by medium**
   (§2d): solids inside Pass 1 so `e_solid_deposit_sum` books it automatically;
   gas through `gas_energy::deposit_railed`, never as `temperature[i] =`. Cite
   the "Gas temperature is a mirror" row and its vacuous-failure mode.
3. **Declare Φ** (§4b): a new per-tick int64 plane, its allocation site, its
   clear line in the conductor, and its resident-path treatment under the
   RL-batch-habits rule.
4. **Put the maximum-principle gate in P1's gate list** (§4d), with a stated
   evaluation point (§4c) and a non-vacuousness condition.
5. **Rewrite §5.3's writer** (§2b): the sweep writes `rad_net`; the Pass-1
   fold's `ts[i]` mask is what opens; the counter is the existing
   `e_gas_deposit_sum`. As written it specifies a fifth group.
6. **Add the unit-absorption exit to §2.3's identity** (§2e), at P1, so gate 1
   is built once.

**Required — must be fixed before implementation.**

7. Fix `ret`'s shift to `>>32`-equivalent (§8b) — the defect critique 1 fixed
   for `emitted`, still live on the leak line.
8. Either commit an **integer** reference sweep or make it P1's first
   deliverable (§8c). `sweep_ref.py` is float.
9. Specify `reciprocal_q16` for the Fleck divide and state its truncated
   rounding; then delete or correct "doors 1 and 2 only" (§1c, §5a).
10. Add `radiation_sweep.cpp` to `cpp/CMakeLists.txt`'s `/fp:strict` list and
    to `test_no_float_in_sim_tu.py`'s `SIM_TUS` + `BASELINE` at 0/0/0, and
    correct that file's stale "raycaster.cpp is render-only" comment (§1e, §5b).
11. Say where the sweep is **called from** (`PhysicsEngine::step` per the
    canonical row), and that both `cast_fire_heat` call sites
    (`physics_runner.py:819`, `:1207`) go with it (§1f).
12. Address the resident path under RL-batch habits §A (§1g).
13. Resolve `radiation_sweep.*` versus "lands here, never beside it", and name
    the `E°` table's post-P3 owner (§1d).
14. Give `dyn_heat_atten` a dtype, an A/B-harness entry, and a digest decision
    (§1b); note the `stamp_units` signature change.
15. Correct "four closure groups" to six, citing
    `tests/test_thermostat_books.py:70-92` (§2a).
16. Rule on the gas branch's density double-count (§2c).
17. State the sweep→fold lossy boundary; stop calling the channel "exactly
    conservative inside the sim" without it (§2f).
18. Replace "a Recorder DTYPE-class contract extension" with what the widening
    actually touches (§6a), and correct §3's "digested? yes" (§6b).
19. Engage `gamemap.py:487-494` before offering "a global scalar" for `rad_amb`
    (§6c); extend the `rad_amb` consumer list beyond the one named test (§6d).
20. Say "DIGEST_SPEC_VERSION bump + regenerate all goldens" for the P8 digest
    entry, not "re-baseline" (§6e).
21. **Split P3**: move `light_cull`, the 16-light cap and NMS into P6; move
    `range_base`/`range_per_intensity` to P5; give the ~46 property-gate
    rewrites their own scheduled patch (§3a, §3c, §3d, §3e, §7b).
22. Withdraw "`rad_flux` consumers unchanged" and state the recalibration P5
    owes, including the `max(heat, rad_flux)` ruling (§3f).
23. Say where the render light sweep runs, in what language, and give a
    **per-frame** number (§7c) — R-P said GLSL.
24. State where the rules-side light channel gets its extinction, given that
    `dyn_light_atten` is float and outside the cross-GPU contract (§3h).
25. Delete or make true the "(Commented in the table as of this session.)"
    claim (§8a); correct "`E°` … widen[s] to int64" (it already is, §6a).
26. Add the disposition lines the survey's §10.2 asked for: `vision.ray_clear`,
    `cover_system`, `gmap.light_map`, `tools/storm_ledger.py`,
    `tests/bench_s8c_fire_heat_check.py`, `tools/fire_tune_loop.py`,
    `tools/storm_probe.py` (§3b, §3g).
27. Fix P2's "§9.3 curve" cross-reference (§7e).
28. State `E°⁻¹`'s behaviour at table saturation, and drop or qualify the
    60000-game rows (§8d).
29. Add the A/B lockstep harness to the gate lists (§1h); scope the two
    over-wide draft rules (§1i); add the `pack_hover_readout` row the Systems
    section promises (§1j).
30. State whether P1 is **wired** (§7a).

---

## OPEN QUESTIONS — only Erik can settle these

1. **Where does the sweep live — `raycaster.*` or a new TU?** The canonical row
   says a replacement lands in the ray engine, never beside it; the design
   creates `radiation_sweep.*`. A staged two-path window is reasonable and this
   is a lasting-name decision, which CLAUDE.md says is Erik's.
2. **Does the render light field move to GLSL (R-P, R-S) or become a second
   CPU/C++ sweep?** §6.1 collapses light onto the sweep for *look* reasons and
   silently reopens R-P's implementation split. This decides whether the raylib
   4.3 build question (R-S) still exists at all.
3. **Radiation into gas: does it ride the v2.4 density-proportional absorption
   law, or bypass it?** Both are defensible; one is a double count of density
   and the other is a second entry point into the gas deposit. It changes what
   R-M's smoke coefficient means physically.
4. **`rad_flux` recalibration is a feel ruling.** Changing it from incident flux
   to absorbed energy moves five `[combat]` dials and the burn-a-marine
   threshold. P5 is labelled "feel" already; this needs a HUMAN-TEST session,
   not a bench.
5. **Is the 16-light cap's removal a look decision or a cost decision?** If the
   look of "every burning tile lights the room" is wanted *before* P6, that is
   a different (and much more expensive) patch than the deletion P3 implies.
6. **`tests/test_cool_shift_axis.py`'s two reds** — is `wood.cool_shift = 13`
   the intended re-tune (in which case both tests are snapshots to rewrite per
   the 2026-09-08 rule) or a regression? Gate 1 of the survey is blocked on it
   either way, and it is the whole of what stands between `fire-12` and green.

---

## What I could not break

For the record, so the next pass does not re-tread:

- The **tick-order argument** (slots 3/6/7) is correct, verified against
  `Simulation.step`. The stamp really is frozen.
- The **god-file policy** is not violated.
- **Door 4** is genuinely untouched — no RNG anywhere in the design.
- The **Fleck factor needs no ledger counter**; §2.8's reasoning is right.
- **§8's budget correction is correct and the survey is the document that is
  wrong** — `_eos_p3_bench.py` times the whole `Simulation.step()` and its gate
  literally reads `p99 <= 0.25 * 83.0`.
- The **float render light field is safely undigested**, so §6.3's swap-seam
  argument holds.
- **Weapon beams** are correctly out of scope, and they really are a separate
  integer Beer-Lambert with their own gas table.
