# Breach — Design Principles (brainstorm)

> **Status: brainstorm, not decisions.** These are *handles* for ideas we want to keep
> well-formulated — named so they stop living only in the back of the head. Nothing here is
> committed to the design or the code yet. Session: June 2026 (2026-06-13).
>
> Companion reference (other games, web-verified): `breach_research_distilled_2026-06-13.md`.

---

## North star
A playfield where destruction and other dynamic systems **interact**, so the combination of
states blows up → a **huge, emergent state space** → a rich environment for the machine-learning
goal (training AI to play). Every mechanic below is judged against this: does it *feed* the
state-space-and-positioning core, or compete with it?

Lineage note: this puts us in the tradition of the **original X-COM: UFO Defense (1994)** —
orthogonal, always-on systems that multiply into emergent tactics — **not** the streamlined
Firaxis reboot's authored, discrete design. Our coupled physics (pressure / smoke / fire /
water / destructible structure) is a continuous-simulation descendant of that philosophy.

---

## Principles

1. **The simulation IS the delayed-consequence engine.** Every continuous system propagates over
   seconds, so the map at *t+10* is the consequence of your choice at *t*: smoke drifts into your
   own lane, fire spreads onto your route, a breach vents pressure that moves gas unexpectedly,
   cover erodes while you sit behind it. We don't *add* delayed consequences — we **reveal** the
   ones the physics already generates.

2. **Delayed consequences > instant punishment** (the "Demoralizing Shout" principle). Effects
   whose payoff is invisible in the moment and decisive later are what create a real skill
   discriminator. We want this *one way or another* — and it need **not** require big health bars.
   Non-HP channels:
   - **Morale / panic** (à la 1994 X-COM) — losses/suppression drain morale; the cost surfaces
     later as a panic at the worst moment.
   - **Bleeding wounds** — a hit bleeds out over time unless someone spends an action.
   - **Positional debt** — committing a unit to suppress/overwatch is a loan against the future.
   - **Ammo attrition** — the wasted burst only hurts when you're reloading at the wrong instant.

3. **Ammo-as-physics.** Interchangeable ammo types each plug into a *different* simulation, so
   choosing a round = choosing which physical subsystem to provoke:
   - **AP** → penetration / direct damage (and punching through android armour).
   - **HE** → the **pressure** field (overpressure, structural / burst-threshold failure).
   - **Incendiary** → the **fire** sim (ignite fuel gas, spread, area denial over time).
   This is a decision no aura-stat game can offer, because no one else has the coupled sim.

4. **Continuous cover.** Cover is not a binary half/full tag — it's the raycaster's **continuous
   occlusion value** (we get cover *and* flanking almost for free from the existing raycaster).
   Smoke becomes temporary cover; a burning/eroding wall is *degrading* cover. Cover **durability
   already lives in the material table** (per-material HP / burst threshold): concrete outlasts
   drywall outlasts furniture, for free. "Find better cover" = read the material map.

5. **Positional roles, not aura classes.** "Classes" express value through *space and the
   destructible systems*, not proximity buffs (proximity buffs pull toward clumping, which fights
   our LOS/destruction core). Working role list: **breacher, suppressor, overwatch-anchor,
   recon/spotter, demolitions** (+ medic, + maybe mage).

6. **No arbitrary weapon locks.** Any soldier can use any weapon; gate only by *simulation* —
   **weight / strength**, and maybe **Intelligence** for tech/psi gear (balance reasons only,
   never arbitrary). We **will** have weapon *classes* (shotgun / rifle / SMG / LMG / sniper /
   launcher) as distinct tools with real tradeoffs (range, spread, suppression value, penetration).

7. **Overwatch in active-pause.** A reactive *conditional* stance ("fire when anything crosses
   this arc") — underexplored in freely-pausable real-time tactics, and it marries perfectly with
   the pause: freeze → paint arc/condition → unpause → it resolves itself. Keep a reaction-time /
   accuracy penalty on snap reactions so *when to set it* stays a real decision. Bonus: it's a
   conditional policy, which is gold for the ML side.

8. **Suppression.** Fire in the general direction of a target even without a clear shot → pins the
   target + applies an accuracy/positional debt. Synergises with **cover destruction**: suppress →
   cover erodes → eventually they're exposed (compounding).

9. **Force concentration vs area control** ("focused fire"). Concentrating the squad on one target
   (drop it now → less incoming) vs spreading for area control is a genuine decision. Watch the
   tension: focus-fire + low TTK can spike lethality — which is exactly what **high-HP androids**
   are *for*: the deliberate exception that justifies focus-fire / AP / HE and a *selective* longer
   TTK, without making every grunt a sponge.

10. **Fallback = bounding overwatch (emergent).** Don't design retreat as a separate mechanic; it
    falls out of overwatch + movement (one element watches while another moves). Also: seeking
    *longer-lasting* cover is just reading principle 4's material map.

11. **The opportunity-cost comp puzzle** (the thing that made WoW comp feel great): *more good
    options than you can field, so you must leave something good out for something that fits
    better.* Its home in our game:
    > **Limited squad slots × weight-budgeted loadouts × situational missions.**
    - **Slots** — which specialists you bring (can't bring them all) = raid-comp puzzle, one level up.
    - **Weight** — what each one carries (can't take every gun / grenade / ammo / charge).
    - **Situational missions** — *the load-bearing piece.* The puzzle only survives if there's **no
      dominant comp**, which requires missions to differ enough that the optimal subset *changes*
      (sealed pressurised hull → demo/HE; dark hold → recon/lights; android boarding → AP/focus-fire).
      Samey maps ⇒ the loadout solves to one build and the puzzle collapses. **Mission variety is
      therefore load-bearing, not flavour.**
    - **Lever:** build a roster *bigger than the deployable squad* and make missions diverge.
      Leaning toward a **unit of ~8, deploy 5–6** (cutting *people*, not just gear, deepens it).

12. **Magic (maybe).** Opens new verbs. Strong candidate: **Conceal** — hide a charge (or a unit),
    proximity/remote detonate. This is a *convergence* of three threads (proximity-fused charge +
    concealment/ambush + delayed consequence), which is usually the sign of a keeper. Needs
    **counterplay** (detection, or enemy AI learning to suspect chokepoints) — and "the AI learns
    to fear your favourite ambush spots" is itself a beautiful ML outcome.

13. **Guided ordnance → a drone.** The "fly around the corner and detonate" fantasy, made plausible
    *indoors*, is a small piloted/way-pointed drone weaving through a ship. Doubles as the
    **recon/spotter** role (reveals fog / LOS).

14. **Control scheme.** A second game mode: **Dragon-Age / Baldur's-Gate-style free active-pause**
    (pause anytime, issue/queue orders, unpause to watch them resolve), alongside the existing
    semi-turn-based mode. The engine already supports this as a *ruleset extraction*, not a rewrite.
    Bindings: **left-click select, right-click contextual order** (move to ground / attack if on
    enemy). Positioning depth doesn't require many abilities — Baldur's Gate proved you can shape
    engagements through *space alone*.

15. **Time-Units philosophy worth chewing on.** The 1994 game's single continuous budget shared
    across move / shoot / react / inventory may map onto a real-time stamina/AP model more naturally
    than discrete actions — worth exploring alongside active-pause.

---

## Parked / explicitly deferred
- **Full gear system → v2.** Interesting but too much to hold now (revisit, esp. with a partner).
- **Stats** — already fields on the unit model (strength / agility / endurance / vitality /
  intelligence / will_*); cheap to explore now and **reuse for the RPG**.
- **Gunshot → pressure-field coupling** — infra exists; default-off toggle to avoid confusing players.

## Open questions to keep live
- **TTK / health philosophy** — avoid sponges; get compounding via *attrition of position* (cover
  destroyed, smoke dissipated, ammo/tempo spent), not big HP pools. Androids are the selective
  exception.
- **Where exactly the comp puzzle's scarcity should bind** — squad slots vs weight vs ammo vs
  mission gating. No candidate feels *perfect* yet; keep the principle (11) well-formulated and let
  the right home reveal itself.
