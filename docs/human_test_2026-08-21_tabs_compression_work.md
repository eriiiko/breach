# HUMAN-TEST record — T_abs compression-work arc (2026-08-21, PASS → merged)

**Verdict: BLESSED (Erik, 2026-08-21: "ok bless").** Two live play sessions on
`playground` (branch `tabs-compression-work`, CUDA build), two manual recorder
dumps (`debug_manual_20260821_143234/143409.npz`, kept at repo root). Erik
breached to space, threw grenades, burned crates, and probed with the new
instruments. Every observation was run to ground before the bless — three
explained as correct physics, two fixed in-session, one real pre-existing bug
filed. Design: `docs/archive/tabs_compression_work_design_2026-08-20.md`
(v2.2); re-baseline rationale:
`docs/tabs_compression_work_rebaseline_2026-08-21.md`.

## What Erik saw, and what each observation was

1. **Cold overlay flooded the screen blue** ("i first thought u filled
   everything with water") — instrument bug, FIXED in-session (`1149969`):
   the ramp painted every near-vacuum cell; now N-masked to cells holding
   ≥25% ambient gas (the trust-gate threshold), so a breach reads as a cold
   ring, not a flood.
2. **Post-breach temperatures snapping back to room temp** — correct physics,
   three mechanisms explained: trust-gate fade in the near-empty vent core
   (cold ring, not core), fast advective re-fill mixing, slow (21–43 s)
   conduction relaxation in stagnant corners.
3. **Grenades turn the air cold** — correct physics, WRONG PAYLOAD, accepted:
   `payloads.frag_standard` is `pressure = 10.0` with NO heat term — a burst
   of room-temperature compressed gas, which honestly cools ~2× on expansion
   (the CO₂-cartridge effect). Invisible under the old relative-T law; the
   honest law revealed it. Erik: "we'll fix it soon anyway" — owner is the
   queued grenade energy-budget retune (HEAT as primary payload).
4. **Fires burning in a nearly airless room** — REAL pre-existing bug, FILED
   (`7a6f970`, TODO bug list): the continuous-O2 law (2026-07-24) gates
   sustain on mole FRACTION, which venting never changes (O2 and N2 leave
   together) — measured crates burning at intensity ~0.37 with N = 0.0000
   for 2335+ snaps. Not this arc's scope (law untouched); owner = the queued
   fire retune / O2-suffocation session. (The warm-crates half of the
   observation is the thermal-mass system working as designed.)
5. **"Did we record temperatures below −289?"** — floor VERIFIED INTACT,
   display FIXED in-session (`06bb973`): zero snap-cells below −289 in
   either dump (coldest −288.927 = 1.07 K, on the rail); what Erik saw was
   the hover readout's canonical-frame Kelvin (K = 293 + 3·T_game) going
   absurd below ambient ("−574 K" on a 1.1 K cell). Sub-ambient now displays
   the EOS frame, labeled `K_eos`.

## Decisions taken at the gate (on P-W2's data)

- **D-1 — cap²-plane ambient floor: KEPT.** Mach census on the venting bench
  and the live dumps: max |u|/c_own ≈ 3, min P at quantization scale, no
  flash route; k_drag=10 precedent says a c(T_abs) cap would kill venting.
- **D-6 — T_MIN = −289 (t_abs = 1 K): KEPT.** The feared ×290 pressure
  collapse did not materialize (measured min P −0.0003 atm on the bench,
  −0.041 atm in live play — 7× better than the previously blessed −0.310).
- **Grenade cold: accepted as-is** — graduate to the grenade energy-budget
  retune.
- **Quiet-room acoustic drift** (near-linear, +4.6 game-deg/2000 ticks,
  +10.4 @ 10k, non-saturating): presented, accepted as a monitored accepted
  gap (instrumented by `tools/quiet_room_drift.py` + its smoke gate); the
  deeper fix is the KE↔eth kick-side debit, still the named open half.

## Suite state at close

24 failed / 2256 passed / 5 skipped — the 24 are the pre-arc fire/materials
baseline debt, name-for-name (`docs/archive/tabs_compression_work_baseline_2026-08-20.md`
§2a), plus the pre-existing XPASS(strict) on
`test_fire_heat_source::test_full_chain_heat_ignites_air_separated_wood`
(baseline debt, reported not un-xfailed). Lockstep CPU↔CUDA tol 0 throughout.
One sanctioned golden re-baseline (`fe5530d`), rationale in the dated doc.
