# SEED — fire-model design session (three coupled questions, 2026-07-30)

**For: a fresh Fable design session.** Written by the Opus session that just built the
thermal-mass-axis arc, at Erik's request ("prepare seeding docs"). Self-contained — the
new chat does not need the old one's context.

**What this session is for:** three questions the thermal-mass arc surfaced but did not
answer. They are entangled — each one changes the surface the other two are tuned on — so
decide them together, in one pass, the way the EOS escalation was decided.

**Deliverable:** a ruling doc in the style of `docs/thermal_mass_eos_ruling_2026-07-30.md`
— answers with reasons, a patch spec, gates, escalation triggers. Opus executes from it.

---

## 0. Where the engine is right now

The **thermal-mass-axis arc** is built, gated, pushed, and **awaiting Erik's play-test**
on branch `thermal-mass-axis` (7 commits: `f5e9aa3` P1 CPU, `312e984` P2 CUDA, `6f57762`
P-EOS, `d781b5c` P3 close, plus the design/escalation/ruling docs). It is **not merged**.

What it changed: the thermal medium used to be selected by `solid` (= `permeability <= 0`,
a **flow** property), so furniture (`permeability = 0.5`) was treated as gas and its
temperature was advected away by the fire's own plume. Now a derived `thermal_solid` mask
(`thermal_mass > 0`) routes the thermal medium, and — per the EOS ruling — **on
`thermal_solid` tiles `temperature[]` is OWNED by the TemperatureSolver; every other
system is a READER.** A crate's temperature is now genuinely an object's.

Reading order for background: `docs/thermal_mass_axis_design_2026-07-25.md` →
`docs/thermal_mass_eos_escalation_2026-07-30.md` →
`docs/thermal_mass_eos_ruling_2026-07-30.md` →
`docs/thermal_mass_axis_bench_report_2026-07-30.md` (P3's measurements) →
`docs/fire_tuning_plan_2026-07-22.md` §9 (the fire chain, the dial order, the targets).

**Also on the branch (built, do not re-decide):** a **per-material `cool_shift` axis**
(`344f3ed`) — the loss-side twin of `thermal_mass`, because one global `COOL_SHIFT` cannot
serve both a wooden crate (wants a slow e-fold) and thin hull plate (wants a fast one).
Every material seeded at the old global 5, so it landed byte-identical and is a dial the
moment Erik turns it (verified e-fold 1.333 s @5, 170.708 s @12, ratio 1.000 vs predicted).
**It changes the answer to Q3** — the cliff no longer has to be cleared by a global crutch.
⚠ Note `--set physics.thermal.COOL_SHIFT=N` is now **silently inert** for any material
carrying an explicit column.

**Erik has set no fire numbers himself yet** — he said so explicitly. The current
`config.toml` values are not blessed choices. Do not treat any of them as sacred.

---

## 1. Ground facts — settled, do not re-derive

**Ignition is temperature-triggered.** `src/simulation/combat.py:392`
`apply_temperature_ignition` fires when a tile's **temperature** crosses `ignition_temp`
(wood 300 = 620 °C) with O₂ above the extinction limit and fuel remaining; commit
`423cd38` made it an edge-trigger (seeds once, re-arms only after cooling).
`ignition_seed = 0.1` is **not a trigger** — it is the value `I` is set to at that moment,
i.e. the fire's starting size. (Worth stating explicitly because the two are easy to
conflate, and §4's `I_crit` is a *survival* floor, not an ignition threshold.)

**What `I` is:** the tile's normalized burn rate ∈ [0,1] — "how vigorously this cell is
burning". It scales heat out (`k_fire_heat·I` per ray), O₂ draw (`burn_rate·I·o2f`), fuel
loss (`wall_damage·I` hp/s) and flame reach (`range_base + range_per_intensity·I`).
I = 1 is a fully-involved tile. It grows logistically:
`dI/dt = k_grow·avail·hot·I(1−I) − k_die·(1−avail·hot)·I`, with `avail = F·o2f`
(`F = wall_hp/fuel_ref`) and `hot = clamp01((T − fire_T_ext)/fire_T_span)`.

---

## 2. Q1 — The headroom question: what should `I = 1` mean?

**★ Erik's intent, stated directly (2026-07-30) — this is the design target, treat it as
the spec:**

> *"I don't think I want it unbounded — but my idea was that a normal fire in air burns at
> I ≈ 0.5, and the reason for that was to keep the higher values I = 1 to fires which have
> more than normal O₂, or wind that feeds it more O₂, or — I don't know — if it burns next
> to other tiles perhaps their radiation can also feed I to become higher."*

And, correcting a conflation in an earlier draft of this doc:

> *"`o2_frac_amb` should NOT be 1.0. The ambient O₂ should be 21%, but we may have O₂
> reservoirs later on, increasing the O₂ content locally etc — that should affect fire's
> intensity."*

So: **the [0,1] cap stays. A normal fire in normal air sits at ~0.5. The upper half is
reserved as headroom for enhanced conditions** — locally elevated O₂ (reservoirs, leaks,
wind delivery), and possibly neighbour radiation. **`o2_frac_amb` stays 0.21 and keeps
meaning "what the ambient atmosphere is".**

### 2.1 The core defect: the law normalizes by ambient, so ambient IS the maximum

Do not confuse the two names:
- **`o2_frac_amb`** — a *dial* (config `[physics.fire]`, and **per-map**: the level's
  `[ambient] o2_frac` overwrites it at load, `src/simulation/physics_runner.py:956-957`).
  It states what the ambient atmosphere is. Correct value: **0.21**.
- **`o2f`** — a *computed per-cell factor*: how much this cell's local O₂ supports burning.

Today: `o2f = clamp01((X − o2_frac_ext)/(o2_frac_amb − o2_frac_ext))`, i.e. `X_ext = 0.13`,
denominator reference = **ambient**
(`cpp/src/fire_simulation.cpp:194`, mirrored **bit-identically** in
`cpp/src/combustion.cpp:164-177` — the two O₂ laws must stay bit-identical, so any change
lands in **both**, plus the CUDA twins).

**Because the denominator IS ambient, ambient always yields `o2f = 1`, which the `clamp01`
then makes the ceiling.** Every enrichment route Erik wants is therefore invisible *by
construction*: at X = 0.30 the raw ratio is 2.125 → clamped to 1.0. An O₂ reservoir, a
leak, wind delivering more oxygen — none of it can ever register. This is the defect, and
it is a normalization mistake, not a missing feature.

**The fix direction (decide the specifics, but not the diagnosis):** give the law its own
**full-response reference**, separate from ambient. Pure O₂ (1.0) is the natural physical
choice — `o2f` then becomes a true physical fraction, "O₂ above extinction, normalized to
pure oxygen", the `clamp01` effectively never binds, and headroom always exists. Ambient
air lands at `(0.21 − 0.13)/(1 − 0.13)` = **0.092**.

**⚠ Trap: this cannot be done by editing a config value.** Setting `o2_frac_amb = 1.0`
would be **silently overwritten** on any real map by `physics_runner.py:956-957`, and
would also corrupt the dial's real meaning. The two roles must be split — a new reference
dial that is NOT map-overridden, with `o2_frac_amb` left alone.

> **★ STATUS 2026-07-30: §2.1 is BEING BUILT, at Erik's explicit direction** (*"this should
> probably just be `o2f = (X − 0.13)/(1 − 0.13)` — that would fix everything"*, confirmed
> twice). A new non-map-overridden `o2_frac_full` dial (default 1.0) carries the
> denominator in both O₂ laws and the CUDA twins; `o2_frac_amb` is untouched at 0.21.
> Gated on back-compat byte-identity (`o2_frac_full := o2_frac_amb` reproduces today
> exactly) and CPU↔CUDA tol 0; **goldens will move and are deliberately NOT rebased.**
> **This session should therefore REFINE §2.1, not re-derive it** — and must still rule on
> §2.3 (does `hot` uncap too?) and §2.4 (heat output, O₂ draw, composition), which are
> untouched and genuinely open. If the ruling disagrees with the pure-O₂ reference, say so
> plainly; it is one dial and cheap to change.

### 2.2 The `I ≈ 0.5` target then falls out of the EXISTING logistic — no model surgery

**With the reference split of §2.1, the current logistic already produces Erik's semantics
— the death term does NOT need restructuring.** This is worth stating up front because the
obvious-looking move here is to give `die` a constant mortality (`die = k_die·I`) so that a
"perfect conditions" fire equilibrates below 1. That is unnecessary once ambient no longer
saturates `o2f`, and it should not be built on reflex.

`I_eq = 1 − d/a`, with `a = k_grow·avail·hot·fan`, `d = k_die·(1 − avail·hot)`,
`avail = F·o2f`:

| condition | `avail·hot` (full fuel, hot) | `I_eq` at `k_die/k_grow ≈ 0.05` |
|---|---|---|
| normal air, 21 % | 0.092 | **≈ 0.5** |
| enriched, 30 % | 0.195 | ≈ 0.79 |
| pure O₂ | 1.0 | **1.0** |

So a normal fire sits at half, local enrichment pushes it up, and I approaches but never
passes 1 — exactly the spec — at the cost of rescaling `k_die/k_grow` by roughly 10×,
which is a **dial**, not a structural change. **Verify this algebra independently before
building on it** (it was derived, not measured), and check what it does to the burn-down
tail, where `avail` falls as fuel depletes — that interacts with §9.3's "fire death
6-8 min" target and with the `I_crit` cliff in §4.

### 2.3 The other two enhancement routes

- **Wind already works**: `grow` carries `wind_fan = 1 + k_wind_fan·W`, unbounded above 1
  (`fire_simulation.cpp:206`). Wind also has a two-sided effect via `k_wind_strip` in
  `die` — and the EOS ruling A2 earmarked `k_wind_strip` for replacement by a proper
  wind-scaled convective term, so check this design does not contradict that.
- **Neighbour radiation is blocked by the same shape as O₂**:
  `hot = clamp01((T − fire_T_ext)/fire_T_span)` (`fire_simulation.cpp:193`) is **also**
  `clamp01`, so a neighbour's rays can raise T without limit but stop mattering above
  `fire_T_ext + fire_T_span`. Decide whether `hot` gains headroom too, or whether O₂ and
  wind are enough.

### 2.4 Decide

The full-response reference (pure O₂, or a documented "enriched" point?) and where it
lives; whether `hot` also uncaps; whether enrichment should raise **heat output** as well
as `I` (`k_fire_heat·I·o2f`, and/or scaling `T_FLAME_MAX`) — note under the `I ≈ 0.5`
design a higher `I` already yields more heat via `k_fire_heat·I`, so this may be redundant
or may be the more honest home for enrichment, and it moves `I_crit` (§4); what happens to
O₂ **draw** (`burn_rate·I·o2f`), since enriched fires eating oxygen faster is physical and
self-limiting; and how the routes compose (three multiplicative unbounded factors can
stack alarmingly). **Watch the rails:** `fire[]` is Q16.16 with rail counters — no new
saturation path. Note **§9.3's existing target is already "peak ~0.5"**, so the tuning
plan and Erik's intent already agree; only the model disagrees.

### 2.5 Downstream consequence worth knowing

Erik (2026-07-30): *"I almost forgot we were tuning the look of smoke. I think it was
impossible to tune smoke because fire was too intense — I didn't see much of the smoke."*
The `fire-b2-smoke-honesty` render work (in this branch's history, merged at `2875408`,
**never play-tested**) is blocked behind fire intensity being sane. Whatever this session
decides should be sanity-checked for what it does to smoke production, since that is the
next thing Erik will look at.

## 3. Q2 — The plume→T shim: two heat currencies

`cpp/src/fire_simulation.cpp:265-293`. Every burning tile heats **itself**:
`gain = gain_q · I · (1 − T/T_FLAME_MAX) · dt`, then
`dT = gain · temp_gain_scale`, hard-capped by the headroom to `T_FLAME_MAX`. This is the
term that keeps a burning tile above the `hot` gate and thus sustains the fire.

**The tension with the arc's ownership rule:** it writes `temperature[]` **directly, in
temperature units, with no `heat_inv_shift` divide**. It was found by P-EOS's writer
enumeration as the **7th** `temperature[]` writer and deliberately left alone (it
pre-dates the arc and behaved this way on wood and hull walls too). Consequence: a
furniture crate (`thermal_mass = 8`) and a steel wall (`thermal_mass = 32`) receive the
**same ΔT** from an identical flame, though steel should heat 4× less per unit energy. It
bypasses the axis the arc just built, and — strictly — it violates the ownership rule's
letter (a non-TemperatureSolver system writing object T).

**The honest counter-argument, which the session should weigh rather than assume away:**
this may not be an energy deposit at all. Read as a *flame temperature* model — "a flame
drives its substrate toward flame temperature, tapering as it approaches `T_FLAME_MAX`" —
it is defensible physics, because a flame's temperature is set by combustion chemistry,
not by the substrate's heat capacity. If that reading is right, dividing by
`heat_inv_shift` would be **wrong**, and the correct fix is to say so explicitly in canon
and exempt it from the ownership rule by name.

**Why it is more urgent than its current size suggests:** P3 measured it at **<1% of
steady state** — but at its own chosen operating point. P3's recommended defaults drop
`k_fire_heat` from 1600 to **2.2**, shrinking the ray deposit ~700× while `temp_gain_scale`
does not move. Its *relative* weight could grow enormously under the very defaults being
handed to Erik. **The session should have this re-measured across the plausible dial range
before ruling** — a term that ignores thermal mass must not be allowed to quietly become
the dominant heat path.

**Decide:** is the shim (a) an energy deposit that must convert through the tile's
`heat_inv_shift`; (b) a flame-temperature drive that is correctly substrate-independent
and should be named as an explicit exemption in the ownership rule; or (c) something to
fold into the TemperatureSolver so there is exactly one writer? And in every case: what is
its weight at the operating point Erik will actually tune to?

## 4. Q3 — The `I_crit` cliff: extinction is real, placement is the question

P3 (`docs/thermal_mass_axis_bench_report_2026-07-30.md`) established, and re-measured at
equilibrium, that §2.5's analytic is exact to ±1%:

> `T*(I) = k_fire_heat · I · 2^(COOL_SHIFT − heat_inv_shift)`, `heat_inv_shift = 3` for furniture

Deposit is linear in I and loss is linear in T, so `T*` is linear in I, and the `hot` gate
opens only above **`I_crit = I_peak · fire_T_ext / T_flame` = 0.278** at §9.3's own targets
— roughly **3× `ignition_seed = 0.1`**. The fire is fenced into `I > 0.278` at both the
ignition and the burn-down ends.

**This is a race, not a threshold.** A tile ignites at T = 300 on *borrowed* heat, starting
at `hot ≈ 0.5` (§9.2's fix working). It must grow I to where its own output holds its own T
above `fire_T_ext`, before the borrowed heat decays. At `COOL_SHIFT = 5` the e-fold is
2⁵/24 ≈ **1.3 s** — race lost. P3's defaults clear it with `COOL_SHIFT = 12`
(e-fold ≈ **171 s**) and `k_fire_heat` dropped 280 → 2.2 (≈2⁷) to cancel the plateau gain
— so **the steady state is unchanged and the fix buys only time**, by making every thermal
response in the game ~128× slower. It is a crutch, and a global one. That is precisely why
the per-material `cool_shift` axis is being built (§0).

**Frame the question correctly:** bistability here is **physical**. Real fires have
extinction limits; a match does not light a log; requiring sustained external heat to
establish a fire may be exactly the behaviour Erik wants. So the question is **not
"remove the cliff"** — it is *where should it sit, and by which lever*. P3 measured four
levers: three relocate the cliff, only a non-linear deposit removes it.

**Decide:** with per-material `cool_shift` available, does the cliff sit correctly for a
wood crate at a physical cooling time? Should `ignition_seed` be *derived* from `I_crit`
rather than set independently (they are currently unrelated numbers that must agree)? Is a
non-linear deposit wanted at all, or is "fire needs sustained heat to take" the desired
feel? Note Q1 interacts: if enrichment raises heat output, it moves `I_crit` too.

---

## 5. Standing constraints (non-negotiable, apply to any patch this session specs)

- **Determinism is a hard requirement** (multiplayer + distributed training). Synced sim
  state is **Q16.16 integer only**: no floats, no libm transcendentals in the sim path —
  use `cpp/src/fixed_point.h`. `test_no_float_in_sim_tu` guards this.
- **Every change gates CPU↔CUDA at tolerance 0**, step **and** resident.
- **Byte-identity gates** where behaviour should not move; **no golden rebase** — goldens
  are re-baselined once per approved behavioural change, deliberately, with written
  rationale, and this line is already carrying a pending one (the joint re-tune).
- **Feel-adjacent ⇒ HUMAN-TEST.** Nothing here auto-merges; Erik plays it before merge.
- The two O₂ laws (`fire_simulation.cpp`, `combustion.cpp`) are deliberately
  **bit-identical** — any Q1 change lands in both, and in the CUDA twins.
- `config.toml` discipline on the current branch: the arc carries exactly one edit
  (air `thermal_mass` 1 → 0) plus the new `cool_shift` column. Don't re-tune dials in a
  build patch; dial values are Erik's loop.
- Suite baseline: **42 failed / 1714 passed / 5 skipped** — the 42 are pre-existing
  by-design reds inherited from the o2-continuous-law line (`FireSimulation.step` missing
  the `n_total` arg; enumerated in `docs/continuous_o2_law_p3_handoff_2026-07-24.md`).
  Match the failure **set**, not just the count.

## 6. Method note that earned its place

From the escalation, adopted as standing practice in the ruling §5: **for a routing or
ownership question, verify by enumerating every writer of the field — do not grep for the
mask name near topic keywords.** A line-oriented grep cannot see a file where the mask
definition and the field writes sit hundreds of lines apart; that is exactly how the EOS
regression hid, and how the plume shim hid after it. If this session specs anything
touching `temperature[]`, `fire[]` or the O₂ planes, enumerate the writers first.
