This is a doc authored by Erik

Goal: think through the next steps for getting breach into a playable state

> **LIVING DOC (Erik's ruling, 2026-07-21):** this file is the living plan for
> the Fire & Heat Beauty + Tuning arc — we edit the relevant parts in place and
> iterate until everything is done or we're happy. (Deliberate exception to the
> append-only doc rule; at arc close it folds to canon + archive as usual.)
> Current state + the active plan: see the dated section at the BOTTOM.

i need - some basic graphics improvements - we have a strong foundation in the phyiscs engine now - i still htink we need some improvement on the render side to make it look good- And we need it to at least look a little bit like the end result to test weather it feels fun or not- if it is fun to look at, it will be more fun to play - that is my hope. 

Fire - well right now fires are active on the phyiscs grid -but i wonder if we can have their graphics begin a little higher resolution.

based off the temperature of the tile - we spawn flames, probably a mix of dark and bright ones? perhaps in prop to how much smoke is generated vs temp , or some  other fitting rule. spawning smoke is also on the talbe i suppise.
can the amount of smoke spawned by a fire be depending on how much O2 is left? so that when O2 is getting low, more black smoke comes out, because we dont have total combustion. just a thought

thought: particles spawned by the fire - graphical ones i mean, ride the wind - to make them more irregular, they could ride the wind vector + each particle adds some noise to it, or sine wave or whatever - i would be very happy if fire looked realistic - and if we cna do a litterature search on how to get the best looking fire - i think it's worth spending time on

fires glow prop to their temperature
but explosions, should they use the exact same principle? 
And how do we visualize explosions now, is temperaute enough? how well does temp ride the wind? The wind is beautiful, perhaps we can use it for explosion graphisc, but not like we did before with pressure ,but perhaps temp + wind could act together -i honestly dont know, perhaps temp alone looks amazing all ready - all i know is that we have the tech now to produce amazing looking explosioins and i want to use it

explosions should be pretty brief - well we can try to tune it - and perhaps different bombs will behave differntly
i want the explosons to be bright, they light up rooms and cast rays a long distance - thats what makes them really cool, that they cast rays longdistances  ,cast very lpong shadows
Explosions cast rays - and perhaps the smoke from the explosions can interact with those rays creating dynamic lightning, that casts shadows in a chaotic manner


As for the tuning constants k_drag, k_push, k_p and k_fire_heat and fuel_per_o2 - we should perhaps defer tuning until we have gameplay forsome of them, fuel_per_o2 --- hmm yeah, perhaps defer until gameplay too.



----
Lightning -we should try to get lightning right now once and for all - but in order to do that i need at least one "entity" - perhaps i recall the name of entity wrong, and if so please correct me, claude, whenu read thius. What i mean are "objects" in the game world that can do stuff, for example a lamp. 
I want to once and for all set ambient lightnig - which will be very very dark, but i think not pitch black. we go for a faint light from the stars. Perhaps if i do underground levels in the future, they can have pitch black - but for now we use the very very faint light as base- and fillthe space ships with lamps that give the lgith we want to have for "normal gameplay" - i meanso we cansee what we do. These lamps may be able to turn on and off, 


----
The water - i think the water surface may need some texture-because rignt now its HARD to see water at all. 
im not sure how to solve it,but we needsometing. perhaps the untis need to be halv submerged too. Needs a little brainstorm.
Water ripples- i now know how small they mustbe, i took a picture when i walked in a pool with Ellen, the ripples andwaves around our feet have wavelehtngs of a few cm or so - meaning we really want a very small area around units in the water to be quite rippley - with samll ripples - i think we can redner it cheap.


----

smoke andlight interaction
I think gping through the kinds of smoke we need:
ithink we need water vapour, dust,black smoke (soot), + the poison / combustion etc gameplay gasses

but dust and black smoke i htink are the two that will show up the most.
dust i havent got yet - and idont know how expensice it is for me to add more types- perhaps we can drop one.

or perhapswatervapur could act as both dust and water vapour.

anyway, there isall ready a system in place where light is attenuated, and the marhing ray gets weaker and weaker 
i think this really needs rebalancing.
think of the opening scene of alien 2- when they enter ellen ripleys escape pod - there are some lights coming trough smoke in a dark room.
the light beams are not visibly dampened at all - they go stragith trhough the smoke - but the smoke still lgihts up (and even lights up the rooom). Well, i want to mimic this for the dust(or if we are economig, the light smoke /water vaopur)
for black smoke on the other hand, i really want it to obscure light, and absorb it.

mission for tuning smoke and light -get these things right AND decide what species of smoke we will have
perhaps decide on a protoclass for these gasses, where water vapour has certain properties, combustion gas has totally different- poison has different etc


Please if i have forgotten somthing, let me know.



Then - i would like to tune explosives a little bit - how much temp should we add after an explosion- this ties in with above

I just wonder how to tune all this stuff in a really good way.
i think i should perhaps create a level where some fires burn, and explosions occur, all in a controlled way.
perhaps i should add frag grenade (is thatthe name in counter strike)
proper military grenade
C4
and the door explosive, it's cool.

but yes, i have ALOT of tuning - so i'd really loike to be smart about it. i'd like to prepare well, but still use the lion partof my time on quality tuning,not creating tools. or the truth is - i dont know where the sweet spot is -this we will probably find out together.


====================================================================
PROPOSED ORDERING — draft by Claude 2026-07-11, for us to refine together
====================================================================

!!!!! STOP — PRE-EXISTING WORK on branch `levels-w1` — REVIEW BEFORE BUILDING !!!!!
(added 2026-07-12; Erik remembered it, Claude verified against the branch)
(✔ RESOLVED 2026-07-08/…: levels-w1 was reconciled and SHIPPED to main (457ba16) —
 the editor, [[light]] lamps + beacons, and the level pipeline are on main. The
 Arc A entity system is also merged. Phase 0's blocker is gone; Phases 1a and 3a
 are substantially MET. Block kept for history.)

A parallel arc — the LEVEL EDITOR, designed + built by Fable — is FULLY BUILT on
branch `levels-w1` (7 commits e5b20b3..97b3de8, ~6,700 lines, 704 tests green, build
phase CLOSED 2026-07-08) and NOT yet merged. Erik hasn't human-tested it yet — his
BLESSING is owed. It already contains a LARGE chunk of THIS plan's foundations. Do
NOT rebuild these — review + human-test + reconcile-merge levels-w1 FIRST:

  • P4 "[[light]] ENTITIES" (9ce00b3): `src/level_lights.py`, a `[[light]]` loader,
    LAMPS ported into both vessels + playground, rotating BEACONS (kind="beacon"),
    an editor LIGHT mode (F6) to place them. Lights feed the raycaster's existing
    LightSource (cone via angle_center/angle_spread).
    ⇒ this IS most of PHASE 1a (entity/lamp/lighting foundation) — ALREADY DONE.
  • The MAP EDITOR (`tools/map_editor.py`, ~1,900 lines): paint NEW/ROOM/CORRIDOR/
    DOOR/SPAWN + LIGHT + WATER modes, live baked preview, SAVE.
    ⇒ this IS the PHASE 3a test-level tool — ALREADY DONE (author test levels in it).
  • P5 [water] initial-state seeding + aquarium demo ⇒ feeds the (parked) water phase.
  • Autotile baker + greybox tileset generator (P1/P2) ⇒ the level-art pipeline.

  THE CATCH (the real work item): levels-w1 branched off an OLD main (f94944f),
  BEFORE the entire EOS refactor. Main has moved enormously since (gamemap.py,
  level_loader.py, main.py, simulation/* all rewritten under EOS + the GPU port).
  So merging it now is a NON-TRIVIAL RECONCILIATION, not a clean merge — and THAT
  reconciliation is the true first step. Two accepted build-time deviations to carry:
  --res scales light range; the water carrier is .npy, not the chapter's PNG idea.
  Erik's owed items: HUMAN-TEST from `.claude/worktrees/erik-preview` (paint session,
  beacon smoke-test, `main.py --level aquarium_demo` shoot-the-glass drain) → bless →
  reconcile-merge onto post-EOS main.

  NET EFFECT ON THE ORDERING BELOW: the true PHASE 0 is "review + bless + reconcile-
  merge levels-w1." Phases 1a (entity/lamp) and 3a (test level) are then LARGELY
  ALREADY MET — only build-new what levels-w1 lacks. (Full status:
  .claude memory `levels-w1-arc-status.md`.)
--------------------------------------------------------------------

THE GUIDING PRINCIPLE (answers "so we don't jump back and forth"):
Split the work into two kinds —
  (A) FOUNDATIONS we can build WITHOUT your focused tuning slot (design +
      systems; Claude/agents can drive these), and
  (B) TUNING that NEEDS your focused, uninterrupted solo time.
Build ALL of (A) first, so when you finally sit down to tune, you tune — you
don't discover you need a lamp system or a smoke-species decision first. And
within (A), follow the dependency chain so nothing is built twice.

DEPENDENCY CHAIN (what unblocks what — this is why the order is what it is):
  • Smoke-species + gas "protoclass" decision  →  unblocks fire-smoke,
    explosion-smoke, AND smoke+light. Pure design, cheap, no tuning. FIRST.
  • Entity/prop system + lamp + dark ambient    →  you can't SEE (or tune) the
    drama of explosion rays / light-beams-through-smoke without a dark room
    with lamps. Foundation for ALL the ray work.
  • Blackbody glow (temperature → emissive col) →  the ONE shared primitive
    behind both "fires glow ∝ temp" AND "bright explosions." Build once.
  • The visual systems then RIDE those foundations.
  • The TUNING comes last — once the systems + a test level + the weapons all
    exist, so your scarce focused time is 100% tuning.

--------------------------------------------------------------------
PHASE 0 — DECISIONS (quick, together, NO tuning slot needed)
--------------------------------------------------------------------
  0a. Smoke species + per-gas PROTOCLASS. Decide the species list and each
      one's properties (optical: does it absorb/scatter light? physical: does
      it decay, react?). Candidates: black smoke/soot (absorbs light — the
      obscuring one), a "light haze" (water-vapour that DOUBLES as dust —
      Alien-2 beams pass through but it lights up), + the gameplay gases
      (poison, combustion products). LEANING: merge dust into water-vapour to
      keep the species count (and cost) down — decide this here.
  0b. "Playable" bar + ambient-light target. What counts as first-playable for
      the fun-test? And lock the ambient = very-dark faint-starlight base (not
      pitch black; underground levels can go black later).

--------------------------------------------------------------------
PHASE 1 — FOUNDATIONS (build; these UNBLOCK the visuals)
--------------------------------------------------------------------
  1a. Entity/prop system + the LAMP (on/off, emits light) + set the dark
      ambient. This is the lighting foundation "once and for all."
  1b. Blackbody-glow primitive. SETTLED 2026-07-11 (Claude checked the code):
      it is ONE primitive (temperature → blackbody colour, one ramp/function),
      with TWO consumers that ALREADY EXIST — you're not building light
      machinery, you're feeding it a physical colour:
        (a) the hot tile's OWN pixels drawn in that colour (self-emissive) —
            today runs off fire *intensity* + an ad-hoc ramp, not temperature;
        (b) the hot tile as a LIGHT SOURCE whose colour = that same blackbody
            colour, cast through the volumetric ray engine (already built:
            per-channel rays, per-gas absorption+scatter, god-ray/smoke_glow
            buffer) — today fire's light colour is a FIXED orange
            [1.0,0.45,0.12] (physics_runner.py:244), NOT temperature-derived.
      THE WORK = write the one temperature→colour ramp and wire it into BOTH
      (a) and (b), replacing the intensity-ramp and the fixed orange. Then an
      ember glows dim-red AND casts dim-red light; an explosion glows AND casts
      white — consistent by construction (one function). Render-only (float,
      determinism-exempt). Much smaller than it sounds — the hard part (the
      volumetric colour light engine) is DONE.

--------------------------------------------------------------------
PHASE 2 — VISUAL SYSTEMS (build; ride the foundations)
--------------------------------------------------------------------
  2a. Fire graphics: higher-res flames spawned off tile temperature; dark/
      bright mix; graphical particles ride the wind vector + per-particle
      noise/sine for irregularity. NOTE: "more black smoke as O2 runs low
      (incomplete combustion)" is CHEAP — the engine already tracks O2 and
      soot_yield, so it's a real physics-driven rule, not a fake. Optional:
      a lit-search on realistic real-time fire before this (worth it — Erik
      flagged fire as worth spending time on).
  2b. Explosion graphics (= the temp-based blast-viz rebuild): brief, bright,
      light up rooms, cast LONG rays/shadows; smoke interacts with the rays →
      chaotic dynamic shadows. Try temp-alone first; add temp+wind if needed.
      (Replaces the old pressure-brightness hack that whited-out.)
  2c. Smoke + light REBALANCE: the light-haze passes light through but glows
      (Alien-2 escape-pod beams); black smoke absorbs/obscures. NOTE (checked
      2026-07-11): this is ALREADY MODELLED — the raycaster has per-gas scatter
      vs absorption (its comment: scatter "may exceed absorption → barely
      absorbs, glows brightly (steam)"). So 2c is TUNING two per-gas
      coefficients (scatter, absorption) per species, NOT building new tech.
      Needs 0a (species) + 1a (lights) done first.
  2d. Water rendering: surface texture (it's near-invisible now), units half-
      submerged, tiny ripples (few-cm wavelength — Ellen/pool observation) in a
      small area around units, rendered cheap. Fairly INDEPENDENT — can slot in
      anywhere from Phase 1 on.

--------------------------------------------------------------------
PHASE 3 — TUNING HARNESS + CONTENT (the ONLY "tool", kept minimal)
--------------------------------------------------------------------
  3a. Controlled test level (fires + explosions, hand-placed). KEY: author it
      in your EXISTING levels-w1 editor — it's NOT a new tool to build, which
      dissolves your "don't waste time building tools" worry.
  3b. Weapon/explosive types for tuning variety: frag grenade, military
      grenade, C4, the door charge. (Yes, "frag grenade" is the CS name.)

--------------------------------------------------------------------
PHASE 4 — THE TUNING (your focused, solo, continuous slots)
--------------------------------------------------------------------
  Explosion temp/feel → fire feel → then the deferred k_* constants
  (k_drag, k_push, k_p, k_fire_heat, fuel_per_o2) WITH gameplay context, which
  is exactly why you deferred them. This is where your lion's-share time goes.

KEY SEQUENCING CALLS — Erik's answers (2026-07-11):
  1. Foundations-first / tuning-last — CONFIRMED. Start with entity + base
     lighting + lamps; only then do we have the tools to test the light-
     interaction stuff (explosions, smoke, etc).
  2. Merge dust into water-vapour (one fewer species)? — STILL OPEN, settle at
     Phase 0a. [Claude leans yes]
  3. Water rendering — PARK TO THE END. No real water level exists yet (a real
     water level would make it easier anyway); zero rush to get water right.
     Moved to after the fire/explosion/light work.
  4. Fire lit-search — DONE (2026-07-11 night). Report:
     docs/fire_rendering_research.md on branch `fire-lit-search` (unmerged;
     read before Phase 2a). Headline: every high-value pick is a RENDER-LAYER
     read of fields that already exist — NO solver changes, determinism safe.
     Standout novel pick: "oxygen-starvation as a visible smoke story" — a
     sealed fire genuinely chokes + blackens itself and reflares when O2 rushes
     back in, emergent from the real combustion chemistry (the "no one has seen
     this" one, and it's free). Also: blackbody Kelvin→RGB + bloom is the
     cheapest highest-leverage first step; sub-tile flame detail = noise
     advected along the real wind + curl noise (never a finer solve); god-rays
     are ~90% built already. Caveat: exact coefficients (Tanner Helland Kelvin→
     RGB, GPU-Gems defaults) are cited by link, not reproduced verbatim.

====================================================================
2026-07-21 — FIRE & HEAT BEAUTY ARC: verified state + active plan
(Erik + Claude, post-physics-v1-close; supersedes the ordering above
where they differ. S8c + Arc B are in flight — see "sequencing".)
====================================================================

WHAT THE CODE ACTUALLY DOES TODAY (scouted + verified 2026-07-21):
  • The beloved "temp view" = HeatFieldOverlay (T key), renderer/overlays.py
    :317-407. Debug overlay: flat additive 5-stop LUT over gmap.temperature,
    normalized by temp_display_max=300 → everything ≥ wood-ignition already
    renders WHITE-HOT. This is the "saturates to white too quickly" culprit.
  • Fire does NOT light rooms because fire is absent from the render light-
    source list (main.py:375-402 — only static lamps, beacons, flashlight).
    The volumetric colored raycaster + light-field textures + marine shader
    all exist; fire lights = add sources to that list. Render-only.
  • gmap.smoke IS the black_smoke gas slice (gamemap.py:209). Fire emits into
    it (smoke_emission=0.8, fire_simulation.cpp:264-280) and combustion adds
    real O2-gated soot (soot_yield=0.3). It LOOKS light grey only because
    FieldOverlay draws every gas with one flat grey tint (overlays.py:45-90).
    → black smoke is a RENDER fix + coefficients, not new sim.
  • Explosions inject raw PRESSURE/mass (physics.py:84-88) and never touch
    temperature — temp didn't exist when grenades were designed. Confirmed
    plan: remove the pressure add (payload tunable), inject HEAT instead
    (small new sim path: payload → heat deposit → existing heat→T→P chain).
  • Wind stretching flames is real: gas-T advects with wind
    (gas_advection_rate=900) AND fire reads |wind| (k_wind_fan/k_wind_strip,
    both flagged NEEDS TUNING in config.toml).

ANSWERED QUESTIONS (Erik asked 2026-07-21):
  • "Do rays from adjacent sources combine?" — NO. Each source casts its own
    independent ray fan (omni ray count = ceil(2π·range), raycaster.h:65-70);
    the light FIELD accumulates additively, but compute cost is the SUM:
    ~2π·range² tile-steps per source. Per-tile lights on every T>300 tile is
    fine for small fires; a big blaze (100s of tiles) needs a budget. Plan:
    v1 = one light per hot tile with a hard cap (brightest-K), clustering
    (Erik's 3×3 march) held as the optimization step — visually near-lossless
    since adjacent hot tiles throw near-identical light.
  • "Use the fire debug overlay as source data?" — effectively yes: the
    overlay's source data IS gmap.temperature; fire lights read the same
    field (threshold on T, color from the shared blackbody ramp). We read
    the field, not the overlay texture.
  • Bloom: Erik doesn't love it — SKIP bloom entirely. (Plain-words glossary:
    HDR = let brightness values exceed 1.0 instead of clipping at white;
    tone-mapping = the curve that maps those back to screen range gracefully.
    The world shader already ACES-tone-maps; the temp overlay bypasses it.)
  • White-saturation fix: widen the display range (temp_display_max ↑) and
    reshape the ramp so white is RESERVED for truly extreme T; more of the
    ramp lives in red/orange/yellow.
  • Erik's speckle idea (Swedish "spräcklig" — speckled/mottled): interleave
    dark/grey cells in T∈[250,300] and sparsely at higher T for life/texture.
    Two variants to try side by side in the demo tool:
      (a) pure render noise modulating the LUT (cheap, decorative);
      (b) "DIRTY PLANCK": modulate the blackbody color by the tile's REAL
          soot density (black_smoke) — physically interpretable, and the
          O2-starvation choking-fire story then shows up in the flame color
          for free. Try one clean Planck scale + dirtied variants.

THE ARC — four beats, in order (all feel-adjacent → HUMAN-TEST gates):
  B1. Blackbody ramp + fire lights (RENDER-ONLY, no S8c/Arc B collision):
      one T→RGB Planck function wired into (a) the temp overlay color and
      (b) fire light sources placed from the temp field (per-tile, capped;
      3×3 clustering as optimization). Replaces the fixed orange
      [1.0,0.45,0.12]. Rooms + 3D marines light up for free.
  B2. Smoke honesty (RENDER + coefficients): per-gas plume tint in the
      overlay (soot near-black, steam white), density/opacity curve, dirty-
      Planck speckle experiments; tune soot_yield / smoke_emission so dying
      fires hand over to black smoke.
  B3. Explosion rework (SIM change): pressure-add → heat-inject per
      blackbody_smoke_and_rendering_brainstorm.md (Tier A item 5).
      Digest/golden rationale + design-gate + HUMAN-TEST.
      (Was gated on S8c — UNBLOCKED 2026-07-21: S8c merged, 9eb47c0.)
  B4. THE TUNING PASS, LAST (Erik's focused solo slots, live-slider harness
      extended from tools/lighting_demo.py): k_fire_heat, T_MAX_PHYS
      (provisional, Erik review owed), fire_T_ext/fire_T_span, k_wind_fan/
      k_wind_strip, temp_display_max, smoke optics, explosion heat. Tuning
      is last so nothing gets tuned twice.

SEQUENCING / TRAFFIC (updated 2026-07-21 late): S8c is MERGED (9eb47c0) —
only Arc B remains in flight (logic layer; disjoint files). B1+B2 are
render-layer → can start NOW in their own worktree. B3 is sim-side →
unblocked, but still design-gate + golden rationale + HUMAN-TEST. Fire
lit-search report (docs/fire_rendering_research.md, branch fire-lit-search,
unmerged) — B1's patch P0 merges it.
  ► B1 DESIGN IS WRITTEN: docs/fire_b1_blackbody_fire_lights_design_
    2026-07-21.md (spec + patch plan P0–P4 + Opus kickoff prompt).
  ► B1 BUILT + MERGED 2026-07-21 (merge 85cbe14; Erik blessed "much
    better"). Tuning session queued in docs/TODO.md.
  ► B2 DESIGN IS WRITTEN + ADVERSARIALLY CRITIQUED (blockers resolved
    in-doc, 2026-07-21/22): docs/fire_b2_smoke_honesty_design_2026-07-21.md
    (patch plan P0–P5 + Opus kickoff §9; research base
    docs/research/smoke_render_litsearch_2026-07-21.md). Fire-studio level
    carries Erik's rotating beacon (his ruling, 2026-07-21).

DECISIONS — Erik's rulings 2026-07-21:
  ✔ Smoke species (old Phase 0a): DROP DUST (not needed yet; steam covers the
    haze role). Keep TWO visual species + the gameplay gases:
      • "smoke"  = what fire produces (soot) — RENAME black_smoke → smoke.
        "Smoke" now means fire-smoke and NOTHING else. (Code already agrees:
        gmap.smoke aliases the black_smoke slice.)
      • "steam"  = water vapour — RENAME white_smoke → steam (also §8 item 5,
        pre-approved there). Steam is the Alien-2 light-haze medium.
      • gameplay gases (poison, teargas, fuel_gas) unchanged.
    Rename mechanics: config [gases.*] keys + any level refs; mechanical
    patch, check goldens/recorder untouched. Protoclass idea (per-species
    properties) = the existing GasTable columns; extend as needed.
  ✔ Fire-light budget: DON'T cap range hard — long-range lights are the
    point (big rooms lit by explosions; geometry caps range naturally and
    our lights are cheap). CAP THE COUNT instead: brightest-K promoted to
    real ray-casting sources. NOTE: this two-tier design already exists in
    brainstorm §8 item 2 (per-tile in-march glow is ray-free ~3 MADs/tile;
    only brightest-K tiles become LightSources) — build that. If 3×3 stride
    isn't enough, go 5×5; restrict marching to neighbourhoods of actual
    fires. Detonations are separate: spawn ONE discrete light source with a
    predetermined lifetime/decay envelope.
  ✔ temp_display_max + ramp stops: tune by eye (B1/B4).
  ✔ Explosion heat magnitude: anchor on REAL-LIFE data per payload class —
    frag grenade / military field grenade / C4 (building-scale) / breach
    charge (door/wall). Small research task in the B3 design; then tune by
    feel.
  ✔ Explosion pressure add: start at EXACTLY 0 (heat only); if underwhelming,
    add a little back until acceptable.

OPEN DECISIONS (remaining):
  ✔ Speckle (RULED 2026-07-21): build BOTH variants behind one toggle,
    pick by eye in the studio; both must ride the advected noise (static
    speckle reads as a screen overlay). B2 design §5.
  ⭘ Fire-light K + stride numbers (pick during B1 by eye/perf).
  ✔ MacCormack anti-diffusion (RULED 2026-07-21): DEFERRED with a WRITTEN
    TRIGGER (B2 design §7): reopen ONLY if plumes lose peak density /
    silhouette while TRAVELING after B2 (transport dissipation — no render
    trick recovers it); then full ceremony (digest re-baseline + rationale
    + HUMAN-TEST). B2 ships render-side sharpness instead (bicubic +
    τ-space curve + noise erosion).
  ⭘ Curl-noise determinism flavor (§8 item 4): render-only wisps → likely
    exempt; if anything sim-adjacent, use the host-precomputed noise texture.
  ✔ ACES precondition (§8 item 8): CHECKED in B1 P4 — lighting.fs tonemap
    is component-wise on vec3 + HDR-unclamped; no shader change needed.
  ⭘ GAS-CHEMISTRY BEAT (Erik wish list 2026-07-21; sim-side, NOT in B2):
    fuel_gas ignition (the M3 hook — GasTable column exists, consumer never
    built), heat-driven water→steam boil + steam→water condensation (only
    the W5 pressure flash-boil exists today), smoke ATTENUATING radiant
    heat (CORRECTED Erik 2026-07-22: attenuate-and-DELETE only — NO
    re-emission, NO deposit-into-T. This PRESERVES the 2026-07-05 one-way
    ruling, which is about re-emission feedback loops
    (heat→glow→emission→heat; brainstorm §8 item 2, :519-522). Today
    gases NEVER attenuate the heat channel — only material heat_atten
    does (physics_runner ~:884/:957, raycaster.h :195) — so the negative
    feedback does NOT exist yet; it's a new gas-driven heat_atten fill,
    same energy-out shape as K1 wall occlusion and wave_absorb), poison
    thermal breakdown (fire destroys poison). Ice: DEFERRED (Erik
    reaffirmed 2026-07-22; needs a solid water phase that doesn't exist). Registry answer:
    GasTable columns are the right HOME for the coefficients, but every
    PROCESS is a solver/exchange-row change (the exchange.py coupling-row
    idiom: teargas-blind + poison-dose are the live precedents) with full
    digest ceremony. SLOT DECISION owed (Erik): before B4 (tune-once
    principle) vs after with an accepted re-tune.

BRAINSTORM §8 STATUS (checked 2026-07-21 — mostly resolved by history):
  item 1 glow_temperature field → MOOT: the EOS reframe was adopted; real
    gas T carries the warmth and feeds the LUT.
  item 2 soot re-emits light-yes/heat-no → DECIDED 2026-07-05 (one-way heat
    channel) — and it carries the brightest-K light design (see above).
  item 3 expansion→pressure tap → MOOT/ABSORBED: post-EOS, injected heat
    raises P = C·N·T by itself; plus Erik's pressure-add=0 ruling.
  item 4 confinement OUT / curl-noise IN → confirmed; flavor open (above).
  item 5 white_smoke→steam rename → APPROVED (matches species ruling).
  item 6 MacCormack → open (above).  item 7 2.5D smoke layers → out of arc.
  item 8 ACES per-channel check → open, mechanical (above).
  item 9 build order → superseded by beats B1–B4.
  item 10 canon fold at arc close → standing rule, unchanged.




