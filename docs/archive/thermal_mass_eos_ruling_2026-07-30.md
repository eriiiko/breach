# RULING — the thermal medium in the EOS pass (Fable design pass, 2026-07-30)

Answers `docs/thermal_mass_eos_escalation_2026-07-30.md` (§4, five questions).
Reading order: escalation → this ruling → the P-EOS patch spec (§4 below).
Design lineage: `thermal_mass_axis_design_2026-07-25.md` + its build addendum —
nothing in P1/P2 is invalidated; this ruling extends the same axis into the
one pass that actually moves T.

**The one rule everything below follows from:**

> **On `thermal_solid` tiles, `temperature[]` is OWNED by the TemperatureSolver
> (deposit-convert / conduct / COOL_SHIFT). Every other system is a READER.
> The EOS reads T (for P = C·N·T) and never writes it there.**

An object's temperature changes because energy is deposited into the object or
lost through its declared sinks — never because a fluid parcel was sampled
across it. That is chapter 06's original contract ("air temperature would …
advect everything — the wrong behavior"), now stated as an ownership rule so
no future pass re-introduces the regression by accident.

---

## 1. The five answers

**A1 — Both EOS writes skip thermal_solid tiles: step-1b AND step-4c.**
Step-1b (semi-Lagrangian sample) is a *fluid-parcel transport* claim — "the
gas now at i came from upstream." The object at i did not come from upstream;
the write is semantically void for an object. Step-4c (−P∇·u compression
work) is work done *on gas by compression* — the object does not compress.
Both are gas-medium claims; neither may touch object temperature. (The pore
gas's own advection/compression state on that tile is subsumed into the
pore-gas approximation — see A3 — exactly the kind of sub-tile detail a
one-field-per-tile model deliberately does not resolve.)

**A2 — Occluder.** A thermal_solid tile is a WALL to the EOS backtrace
sampler (the eos-side analog of P1's `gas_wall_at` site, same semantics).
Two reasons, one physical, one structural:
- *Physical:* gas percolating through a packed object exchanges heat with it
  intensely; the parcel that emerges does not carry the upstream temperature
  identity through the tile. Occluding the sample and letting heat route
  around via the open cells (which combustion H_fuel and the heat rays warm
  directly) is the honest 1-tile approximation.
- *Structural (the decisive one):* sampling the object's T as a source is a
  **free-energy channel** — semi-Lagrangian sampling copies without debiting,
  so a 1300 K crate would heat every parcel the wind drags past it at zero
  cost to itself, forever. Object→gas heat transfer must be a *deliberate,
  two-sided, accounted* exchange. That channel is the planned wind-scaled
  convective term (the Phase-3 `k_wind_strip` replacement — Erik's
  convective-cooling fork; also the path units-ignition will eventually
  read). Until it exists, the object heats gas only via the already-accounted
  routes (rays; combustion H_fuel into adjacent air; pressure — A3).

**A3 — The §2.3 pressure decision STANDS: P = C·N·T[i] with object T.**
Hot pore gas is physical: the steady state is thin hot gas,
`N_eq ≈ N_amb·T_amb_abs/T_abs` — the *same* equilibrium every hot AIR tile
already reaches under the EOS today. Nothing numerically novel; the elliptic
solve sees one more hot spot. It is also the crate's one honest push on the
gas world (its plume). The O₂ law is mole-fraction-based (thinning-invariant),
so pore-gas thinning cannot re-open the density trap. **The tripwire is
retained verbatim** and now instrumented as gate (f): log P and wind at the
crate through the bench burn; sustained P/wind oscillation artifacts pinned
to the crate ⇒ flip to the named fallback (`t_amb` pore gas, a one-line
branch) and bring the choice back. Prediction on record: it will not fire.

**A4 — Energy accounting: the conservation gate is untouched because it is a
MASS gate, and T transport is non-conservative BY DESIGN.** The sealed-room
gate audits the conservative gas planes (N). Temperature rides
semi-Lagrangian transport, documented non-conservative-but-deterministic
(`src/simulation/gas_fixed.py:11`). Skipping two T-writes on object tiles
*narrows* an existing approximation and moves zero mass; the gate's meaning
is unchanged — and this ruling documents that meaning explicitly so it stops
being folklore. Where the energy "goes": the object's ledger is sources =
ray deposit + combustion deposit (A5-adjacent, §2 site 3), sink = COOL_SHIFT
(the engine-wide ambient sink) — the same open-system convention as
everywhere else in the thermal model. A true engine-wide energy ledger would
be its own project; explicitly out of scope.

**A5 — Furniture κ stays 0 this arc — REAFFIRMED with eyes open.** Now that
COOL_SHIFT genuinely is the only exit, that is a *feature* for the tuning
loop: the §2.5 analytic `T* ≈ (k_fire_heat·I >> shift)·2^COOL_SHIFT` holds
exactly (proven to the LSB in the isolated pass), giving Erik one clean loss
channel per dial. Opening κ now would half-build the object↔gas exchange
with the wrong physics (static face conduction, wind-independent) and add a
coupled loss to hand-tuning. The real channel is A2's convective term,
designed once, later, deliberately.

## 2. The writer enumeration (the escalation's §7 lesson, applied)

Every writer of `temperature[]`, classified. The build verifies this list by
enumeration from the field — not by grep near the mask:

1. **TemperatureSolver** convert / conduct / COOL_SHIFT — P1 done; routes on
   `thermal_solid`. ✓
2. **EOS step-1b + step-4c** (`eos_solver.cpp:406-422`, `:670-688` + CUDA
   twins) — THIS ruling: skip writes on thermal_solid; occluder sampling. ✓
3. **★ Combustion aggregate deposit** (`combustion.cpp:~267-270`, verify at
   build): writes `T[j] += burn·H_fuel/(c_v·max(N,n_floor))` directly on burn
   sites. A furniture tile CAN be a burn site (open, holds O₂) for an
   adjacent burning tile — and under A3 its pore gas is THIN (N ≈ 0.3–0.4),
   so the gas-divisor deposit would spike the OBJECT's T by ~2.5–3× per unit
   burn. Wrong conversion, and rail-hunting. **RULE: on thermal_solid sites
   the combustion deposit converts via the tile's `heat_inv_shift` (the
   object path), same as ray deposits.** Same energy in, object-appropriate
   scale; adjacent-crate fire spread keeps working, now honestly.
4. **Destroy/evacuation seed** (`gamemap.py:1521-1543`) — addendum D5 already
   routed. ✓
5. **field_edit T-paint** (dev aid, rails-respecting) — audit only; a dev
   brush may write either medium; acceptable, note in the patch.
6. **Water/steam/boil paths + anything else the enumeration surfaces** —
   build task: list every remaining writer, classify gas/object/either;
   escalate ONLY a writer that resists classification.

## 3. Gates (P-EOS)

a. **Furniture-free byte-identity, zero tolerance** — structurally free:
   where `thermal_solid == solid`, every edited site is already unreachable
   for those tiles (solid ⇒ cmask 0 ⇒ step-1b/4c never touch them; the
   sampler already occludes solids). Assert it, don't assume it.
b. **No golden rebase**; furniture-scenario digests that move are enumerated
   in the build report; rides the joint re-tune's ONE rebase.
c. **The live-engine bench replicates the escalation's diagnostic**: warm
   seed 280 → monotone T rise from the seed while I grows (no dip), fire
   sustains at physical `fire_T_ext` (≈250, span 100 for the check), and the
   §2.5 analytic holds within ~20% — in `run_substeps`+`step_tail`, not the
   isolated pass.
d. **CPU↔CUDA lockstep tol 0**, step AND resident, furniture-burn scenario,
   with the non-vacuousness controls P2 established.
e. **Conservation / sealed-room / sky-exchange gates green** (no gas plane
   touched; cmask untouched).
f. **The A3 tripwire, instrumented** (P/wind trace at the crate; §1-A3).

## 4. P-EOS patch spec (Opus, branch `thermal-mass-axis`, stacked on P2)

1. Plumb nullable `thermal_solid` into `EOSSolver::step` + the CUDA step and
   resident twins (one static mask upload — the sponge-grid precedent;
   `nullptr` ⇒ today's behavior byte-for-byte, the legacy/space-map path).
2. Step-1b: skip the `temperature[i]` write on thermal_solid tiles; treat
   thermal_solid as wall in the backtrace sampler (mirror P1's `gas_wall_at`
   semantics — keep the two implementations' semantics identical).
3. Step-4c: skip the `temperature[i]` write on thermal_solid tiles.
4. `cmask` UNTOUCHED — pressure, velocity, and gas flow identical;
   `permeability`/shield-not-seal preserved (trigger 5 stands).
5. Combustion deposit re-route per §2 site 3.
6. The §2 writer enumeration, in the build report.
7. Gates a–f; then P3 as the addendum wrote it (bench report, tune-loop TUNE
   defaults at the now-live §2.5 operating point, §9.5 rewrite, hand back to
   Erik's manual loop).

**Escalation triggers:** the A3 tripwire (f); a resident-path seam that
resists the static-upload pattern; any `temperature[]` writer that resists
§2 classification; anything tempting a `cmask`/`permeability`/`solid` change
— full stop, per the standing constraints.

## 5. For the record

The escalation's process note is adopted as standing practice for routing
questions: **verify by enumerating writers of the field, not by grepping
near the mask.** It caught in one day what the D5 sweep structurally could
not. Credit where due — the escalation doc itself is the model of what §4
trigger 3 was written hoping to receive.
