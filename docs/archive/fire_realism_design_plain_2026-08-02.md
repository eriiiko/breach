# Making fire behave — the plain-language design document (2026-08-02)

**Who this is for.** Erik, and any reader who does not know the engine's
internals or the last two weeks of history. Everything is spelled out; nothing
assumes you read the earlier documents. The heavily technical version of this
design (with formulas, code references, and the full critique record) lives in
`fire_realism_design_2026-08-01.md` — that one is for the implementation agents;
this one is the version to read carefully and make decisions from. Where the
two disagree, tell us — this one is newer and includes your latest input
(space ships, scalable object masses, crate conduction, the requirements list).

**Status.** This is a design on paper. Nothing described here has been built.
It has survived two rounds of deliberate expert attack (eight critical reviews
that found and fixed real errors, including two mistakes in our own math), and
it now waits on: one cheap measurement, your decisions in the "Decisions"
section, and one final review round.

---

## 1. What we want fire to do (the requirements)

This is your list from 2026-08-01, plus a few items you had ruled on earlier
that belong with it. If anything here is wrong or missing, this is the list to
correct — everything else in the document exists to serve it.

1. **The campfire arc.** A burning crate should ignite, grow in intensity,
   peak, and then decay — and a lone crate in still air should quite often (Eriks comment - quite often - i dont know wabout this language- it shall ofc be totally deterministic, but what i meant was, it's totally realistic that a fire dies without burning through all the fuel -perhaps thefire was too small to heat up the surrounding fuel to ignite before burning trhough the fuel hot engouf to burn) -- thefire die's
   *part-burnt*, like an untended campfire, rather than always burning to ash.
   You explicitly ruled this partial burn a feature: "it's totally normal that
   fires die before burning everything." This should happen if a pretty cool (say room temperature) object burns will very low intensity i suppose, under normal or reduced O2 supply - it'll be a tuning mission. 
2. **Oxygen governs fire, in both directions.** How much oxygen the flame can
   get — set by air pressure, wind, and the oxygen fraction of the air —
   should feed or starve it. Rich air makes bigger fires; depleted or sealed
   air smothers them.
3. **Wind cuts both ways.** Fanning wind feeds a fire more oxygen and can grow
   it; but a small or cool fire can be chilled and blown out by the same wind.
   A match dies in a breeze that a bonfire ignores. --- Eriks comment: I am not sure how cooling works in the engine - in real life also im not sure, i know that expanding air cools it, wind may cool i suppose, by blowing away hot air replacing it with cool air, perhaps othermechanisms are also present - I would like to be aware of which mechanics blows out fires,inreality and in the engine specifically. I want an explanation of it - if we know it, if we have seen it work or if wejust theorized aboutit- I remember theinitial mechanism (before the EOS refactor) - we had, if my memory serves me - something like a function that took wind and fire intensity or temperature as arguments, perhaps also pressure, i cant exactly remember ,and outputted wether the fire should increase or decrease (in intensity perhaps)  - anyway, the point is, it wasnt an emergent phenomenon - however it's been talk about after the EOS refactor, this phenomenon (blowing out fires) IS an emergant phenomenon - if this is the case it's beautiful - but i'd like to see proof of it and an explanation because otherwise ican't at all rule on the tuning or even architectureof this.Perhaps it would be nice to have a physical explanation too of how this works inreality.
4. **Fires interact with shockwaves.** When an explosion's pressure wave
   sweeps through a burning room, each fire's fate should depend on its own
   condition at that moment — some snuffed out, some fanned bigger. (This was
   your original motivation for wind-driven extinction.)
5. **Flashover** (your "over-ignition" — this is the firefighting term). A
   localized fire heats a room until, at some point, everything ignitable in
   the room catches almost at once. This is the reason we built heat radiation
   at all, and it is now a named requirement rather than a hope.
6. **Fire spreads believably to nearby fuel.** A strong fire next to a crate
   or a wooden wall should be able to set it alight, on a timescale that feels
   right. (Where the honest physics is too weak at our scale, we are allowed
   to tune ignition sensitivity — see requirement 11.) (i think - ofc this will depend on temperature etc- but say two crates stand next to each other, or a whole row of them - and we set it alight at one end, or in the middle - I'd say that spreading to the next crate could take, well, sinceitdepends on wind, temp and all other stuff -let's assume it's in a windstill place with O2, then i'd like it to spread in something like 15-30 seconds probably. when we have all the systems working properly, we might tune even more to our liking, but i think the initial spread happened within a few seconds, which is in my opinion way too fast. It takes some time to start a fire, and i think depending on how warm and big it is, it varies alot, from room temprerature, i'd like to mimic starting a campfire- if you just light a match, it'll take a good 5 min before it really burns, perhaps even more -when the fuel is hot all ready (if we're talking about wood whihc doesnt burn that easily), then it will be quiucker. We do have the ignition temp as our condition, which should be able to simulate all of this exactly how we want)
7. **Rooms and ships behave differently from open ground.** A sealed room can
   choke a fire; opening a vent or blowing a hole changes the outcome. On
   planet-side maps the sky endlessly refreshes the air at the map edge; on
   space ships there is *no sky* — the only air is what the rooms hold and
   what moves through vents and doors. Fire must make sense in both worlds.
8. **Different materials burn differently**, and adding a new burnable
   material (foam, fuel, paper) must not require re-tuning the whole system
   from scratch. Yes - we'll just add the materials properties - ignition temp, etcetera!(i dont know them by heart, but i suppose heat capacity is one of them too, how rich infuelthey are)
9. **Outcomes vary.** The same room should not play out the same way every
   time — variation with conditions is what makes fire tactically interesting
   and what the future AI training needs. No mechanic should turn fire into a
   fixed script (every sealed room smothering on the same clock would be as
   bad as every crate always burning to ash). --- this is a confusing sentense. because we aim for determinism - the thing that i think wants to be communicated here is that, fireis a dynamic system depending on many params, temp, the atmpshere (o2, rpessure wind), so it's alittle chaotic - small changes in initial conditions make two fires behacve differently aftersometime -- and im not even surei use chaotic in theright sensehere.i gues if u lookat the total state on the last decimal,it's very chaotic, if u look at a rougher lens; Probably setting fire to a bunch of crates placed next to each other, nomatterwhichcrate u start with, it may allendup buring down thehouse -but yeah, each initialconditions willresult in it's fire propagation.
10. **The hard engineering constraints.** Fire must stay perfectly
    deterministic (two machines computing the same battle must agree to the
    last bit — required for multiplayer and for AI training) and cheap enough
    to run every tick on big maps.
11. **Realism within reasonable limits — your scaling ruling.** Our world is
    a 2D grid of 1/3-metre tiles, and nothing says a "crate" fills its tile
    with solid wood or a "wall" is massive timber. We are free to choose each
    object's effective mass, fuel content, and ignition sensitivity within
    sensible bounds, precisely so the mechanics above become achievable. We
    aim for fire that *behaves* real, not for a fire-laboratory simulation.

Two items we discussed that are deliberately **not** requirements right now:
smouldering embers that can rekindle (a nice second act for campfires — parked
as its own small future design, because our first attempt at it had real
problems), and movable physical crates (today crates are fixed tiles like
walls; if we ever build movable "furniture objects", that will be a separate
system layered on top, and it can have simpler fire rules).

Yes - let me comment on this:
Todays crates will stay like they are, immovable
we might replace them with, or extend the engine with movable entities later on, that may or may not bew able to burn, perhaps they will have their own simplified fire system.
For now we use the crates as if they are map obejvcts, which if i understand everything right, does mean wecann freely set condiuctivness on them as well.

## 2. How fire works in the engine today, in one page

The world is a grid of tiles. Some tiles are air; some are solid material
(hull, steel, wood walls, glass, doors, furniture — "furniture" is the crate).
Each material has a table row: hit points, ignition temperature, how easily
heat passes through it, and so on.

A burning tile carries a **fire intensity** — a number between 0 and 1. Each
tick, intensity grows or shrinks by a simple law: growth is proportional to
how much the fire has available to it (fuel remaining × local oxygen × whether
the tile is hot enough), with a ceiling proportional to those resources; decay
kicks in when resources fall short. We rebuilt this law two days ago so that
one dial sets the fire's size, another sets its growth speed, and a third sets
how close to the physical limits it dies — previously one ratio controlled
everything at once, which made tuning nearly impossible.

Burning tiles **consume oxygen** from neighbouring air cells (the amount is
tied to real combustion chemistry — energy released per kilogram of oxygen is
nearly the same for all common fuels, a lab fact we anchor to) and **release
heat**: some into the burning object itself (this is what keeps the fire hot)
and a little into the air. --- Erik question : is the heat prodiuced by the burning fuel split into hgeating the object and radiated, as if the sum is always constant? i suppose it's the same from all obects too -this is probaby fine and even a good cap on complexity.

Every solid tile has a **temperature**. Temperature spreads between touching
solids by conduction, decays slowly toward room temperature (a stand-in for
all the losses we don't model explicitly), and — since this week — moves
between separated tiles by **thermal radiation**: hot surfaces beam heat at
what they can "see", computed with rays, with the crucial property that two
equally hot surfaces exchange exactly nothing (this guarantees two fires can
never boot-strap each other into infinity). --we should probably increase the threshhold temperature they start to radiate drastically to save resources -and wecancompensate by making the radiating process quickeri suppose.--- this is actually also  a question for the tuning session , but still, this is my current view.

A tile **ignites** when its temperature crosses its material's ignition point.
Fires die when starved (oxygen, fuel) or when they can no longer keep their
own tile hot enough — that last one is the "thermal knee" that produces your
part-burnt campfires, and we now understand exactly where it sits and which
dials move it.

Finally, everything above runs in exact integer arithmetic in a fixed order,
so that every machine computes identical results — this is why you will see
"no division at runtime" and "same result on CPU and GPU, bit for bit" appear
as constraints throughout: they are not preferences, they are what keeps
multiplayer and training honest.

## 3. What is wrong today — the problem list, in plain words

**Problem 1 — radiation has no "outdoors".** Your catch. Today a hot tile
only loses radiant heat if another solid happens to be within its short ray
reach. A lone burning crate in open air radiates *nothing* — while the same
crate beside cold crates pays heavily to them. Same surroundings, wildly
different cost, depending on whether the scenery is air or wood. Physically a
fire always radiates its heat away; whether that heat lands on a wall or
escapes to the surroundings is bookkeeping, and ours is missing the
"surroundings" account entirely. This one error made crate stacks act as
fire-extinguishers in our tests.  --My solution to this is -if the ray never hitsanother solid object, let it empty all it's energy,well itmay vannish actually, no problem but let the sending crate lose energy prop to (T^4 - T_ambient^4) -- T_ambient, is thatspace 0Kelvinor room temperature, i strongly lean on room tempereature - i do not know if we have specialcases when we are out in space - it would ofc be a coolfeature if the temp would shrink to close to 0 K if we are actually in open space- (say one wall out to space gets destroyed, so actually ambient in that directionwill now be around 0 K). but that may very well be a later refinement, in which case ambient temp most probably should be room temp (which happens to be game temp = 0).

**Problem 2 — wind cannot blow out fires.** The old blow-out dial had to be
switched off because the fire's own rising plume registered as wind and every
healthy fire tried to blow itself out. And objects deliberately do not lose
heat to moving air in today's model (that channel was removed for good
reasons — it was being abused by an earlier bug — and was never rebuilt
honestly). So requirement 3 and requirement 4 currently cannot happen at all.
I remember seeing the temperature curves (T,vs time) and they were so beautiful ,flickering, which i assumed came from just this thing - and i actually loved it.
I wonder if it's possible to scale this phenomenon?
I think  i was under the impression that we removed this because we saved itform some emergan thing elsewhere -but i now think i may have been mistaken thenb.
I am seriously confused on this topic, is thie the whole plume discussion? 
I d thik i want it back, it connected the intensity of thefire to the whole O2 modelling, which depends on the whole wind andeveryhting, which made the whole thing come so much alive- it's a shame to remove it i think.
If restoring it creates otherproblems, id like to really understand them
Seems to me that if a normal fire raises a wind so heavily that thefire is put out, then we need to retune a liuttle bit - tune so that a normal fire flickers and all that cool stuff- I wonder if perhaps this question, Problem 2-either if u can solve it with only my strange littleinouthere, then that'sgreat, but ifd not we cna have a design session only for it- because that's how beautiful i think this effect was. Imean, i am only assuming,but i am assuming this was the effect making the temperature vs time graph beingso noisy and to me looked really realistic.went up anddown between a pretty stable mean valiue, just like fire flames are very varied in intensity.

**Problem 3 — fire barely spreads.** Crates cannot ignite each other by
radiation at today's honest numbers, and crates have *zero* conduction (their
material row says heat does not pass through them at all — originally chosen
to make tuning simpler), so touching crates don't share heat either. Wooden
walls do conduct, but weakly. Requirement 6 fails except at point-blank range
with a very hot fire.
This is my reasoning about htis: Let them conduct agian, which willbe the main method fires will spread.
We tune the radiation spreading to kick in when sustained fires burn for long and cause flashover - that's the tuning scenario for radiation.


**Problem 4 — rooms never warm up.** The old system heated room air by an
accounting trick (rays painted heat into the air they crossed). We removed
that trick — rightly, it invented energy — but nothing honest replaced it, so
today a raging fire leaves the room air stone cold. Requirement 5 (flashover)
is impossible without warm rooms; even discomfort near a bonfire is missing.

Ok this is interesting - i thought flashover wouldhappen due to the walls gradually heating up.
Now - how is room temperature evendone? Without it, how can we even "plot" the flames, which we did previously so beautifully do? was it form rays only? 
what if thefires heat air eitheron their onwn tile only- or in a vicinity of them? That hot air would move around in theroom i suppose, and warm uop the rest aswell, wouldnt it? That would be a good mechanice i suppose - and realistic too.

**Problem 5 — sealed rooms and ships are untested.** Our test bench is an
open planet-side field with a sky that constantly refills oxygen from the map
edge. That is the *most forgiving possible* environment. Ship interiors — most
of our levels — have no sky at all; their oxygen budget is whatever the rooms
hold. Nothing about choking, venting, or ship-scale fire has ever been
measured. (Your space-ship reminder is folded in here: the sky is the special
case, not the rule.) --- very good that you brought this up!
the current test level wasnt built to tune everything on - it was meant to get hte first few parameters about right! I think this has gotten lost in the past week of pathcing - really this level was meant just to tune the first few params(all of which iforgot the name all ready) . but i've noticed to my irritation that it sort of has taken up the place as THE tuning arena, which it never was meant to.
We wanted some windstill place with unlimited O2 to tune params governing the "camp fire scenaroio"
we actually started in a space level, but all the O2 was used (the initial values ofr O2 consumption was too high). I think , since i forget all of the params -we should probably toegehr restate thisgoal, and make a plan for what to tune on that level, 
i Remembererd that the plan was to have a second level with constant wind (artificially made - force the wind to a number) - and try to tune some parameter to make those fires burn out twice as quiuckly at 10m/s wind- or some made up numbers like this. try to make it feel realistic- ofc it'dbe better to try to find answers online how suchand suchwindwould affect a fire - and afterthis weeks patches, is the wind grow / shrink fire phenomenton all ready emergant or not? This needs answresing and whatever the answer is, we test it in a wind level.

**Problem 6 — every fix so far broke the previous tuning.** We spent this
week discovering couplings one implementation at a time. That is why this
document exists: your instruction was to design *all* the fixes together, on
paper, have experts attack the design until it stops breaking, and only then
build. That process has already caught two math errors of ours and one wrong
"expert" idea before any code was written.
EriK : Let me add one more thing . we could perfecly return to python implementations, if htat makes iteasier- ium not sure it does anymore since weneed to use the atmosphere, but by now we've all ready changed the fire system so much - that IF it's preferable to test some systems in easty to implement python float models, it's totally fine.
Thisis not me pushingfor this, just saying all tools are allowed here.

**Problem 7 — only one fuel is tuned.** Everything is calibrated for the
wooden crate. Wood walls partly share its numbers; nothing else burns. There
is no recipe for adding foam, fuel drums, or paper without a bespoke tuning
campaign each time. --- This is kind of OK i think. I'm actually fine with this, after the systems are in place, one tuning session means perhaps one night- and we can afford spending one night for each new material -the important point isn't to have an easily extendable system (alltho itdoesnt hurt) - but to have interesting phenomenon in thegame engine.

Three small genuine bugs also turned up during the reviews (stale bookkeeping
in the oxygen-demand accumulator, an overflow guard that already triggers in
rare cases, and a random-number object that could someday break cross-machine
agreement). They get fixed regardless of any decision below.

## 4. The fixes we designed

Each of these survived two rounds of expert attack; the wording below is
plain, but every one has an exact specification in the technical document.

**Fix A — give radiation its missing "surroundings" account.** Every solid
tile constantly loses radiant heat to its surroundings, always, no matter
what is nearby — that is just what hot things do. When a wall stands in some
direction, the loss in that direction goes to the wall instead of to the
surroundings (and if the wall is just as hot, the exchange nets to zero). Your
own sentence became the acceptance test: *a crate beside room-temperature
walls must burn exactly like a crate in open air* — because room-temperature
scenery and room-temperature surroundings are the same thing. This fix also
makes warm walls give heat *back*, so enclosed rooms genuinely run hotter —
the seed of flashover. (The first two versions of this fix had real
bookkeeping errors — an emitter could be charged twice, or refund itself
through its own tile. The reviews caught both; the third version's arithmetic
was derived identically by two independent reviewers, which is as close to
proof as we get.)

**Fix B — honest convection: objects exchange heat with the air.** Hot
objects warm the air next to them; moving air carries that heat away faster;
hot wind arriving at a cold object pre-heats it (that is how fire jumps
downwind in reality). The exchange is two-sided and scales with how much air
is actually present — so in a vacuum it does nothing (a breached room cannot
"cool by convection to nothing"), and it is completely immune to the
self-plume problem that poisoned the old wind dial, because it only responds
to air flowing *toward* a face, and a fire's own plume flows *outward*. This
fix is also what finally warms rooms (requirement 5's first half).

**Fix C — blow-out done right.** What extinguishes a match in wind is not
cooling of the wood — it is the flame itself being stripped away faster than
it can burn (this is why blowing out a candle takes a tenth of a second).
We model that as wind shrinking the fire's *effective size*. Small fires near
their survival threshold get pushed under it and die; big fires barely notice.
Together with Fix B and the oxygen supply, this gives your full requirement 3
and — because a shockwave is just a violent moment of wind, pressure and
fresh air — requirement 4 falls out of the same three terms with no special
shockwave code at all. One honest caveat from the physics review: our simple
version makes big fires somewhat easier to blow out than reality says they
should be; the review wrote down the exact measurement that will tell us, and
the fallback if it fails is a known, small extension.

**Fix D — the test bench for rooms and ships.** A new test scenario: a sealed
metal room of adjustable size, with an optional vent that can open, close, or
be blown open mid-test, and crates arranged as we choose. This is pure test
tooling — no game code changes — and it is where choking, venting, cluster
fires, wind calibration, and the big measurement below all run. It also gets
one scenario dedicated to the AI-training concern: a scripted arson round on a
real map layout, to confirm fire does not become the one dominant strategy
(fire is currently the only thing that permanently stops zombies, so this
needs an explicit check). ERIK: thislast sentenceis not true - fire is very effective against zombies -but not the only thing that permanently stops them.

**Fix E — a recipe for materials.** Every fire-related number gets sorted
into one of four boxes: laws of physics (never touched per material); physical
properties of the piece (ignition point, heat capacity, hit points — looked
up or chosen per material); properties of the fuel *chemistry* (how fast it
grows, how much soot — shared by a whole class like "wood-like" or
"plastic-like"); and numbers computed automatically from the others (never
hand-authored). Adding a material then means: pick its class, fill in its
physical row following the row conventions, and pass an automatic sanity
check at load — including a new check that would have caught a bug this very
week where a config combination silently made wooden walls unburnable. Two
classes ship now (wood-like, and non-burnable like steel); plastic-like and
liquid-fuel are named but locked until they get their own short designs —
because writing chemistry numbers for materials that don't exist yet is how
dial-zoos start.Eriks comment: I lovethis, and it can be combined with my tuning nights for certain materials.

**Fix F — a written calibration order.** Every dial gets re-measured in a
stated order with a stated test for each step, so that landing one fix never
silently un-tunes another again. The order matters and is non-obvious (for
instance, the per-material emissivity corrections must land *before* the
fire's heat deposit is re-balanced); it is written down once, in the technical
document, and the tuning session follows it.

**Also decided:** molotov-style payloads get re-tested after the fixes land
(today their big ignition splash burns off in about two seconds — probably
not what we want from a weapon); and true smouldering embers wait for their
own future design, as agreed.

-- Grenades too needs another review.

## 5. The big discovery, and the choice in front of you

While checking Fix A's energy arithmetic in real units, the reviews found
something bigger than a bug: **our fire's look and our fire's substance are
two different sizes.** The burning crate's temperature is that of a real
flame (about 1,200 Kelvin — deliberately, so it glows right), but its
chemistry — how much oxygen it drinks per second — is that of a fire twenty
times smaller (about a wastebasket fire). Nobody ever paid for that mismatch
before, because radiation was free. The moment radiation becomes honest
(Fix A), a flame-hot surface radiates flame-scale power, and the little
chemistry underneath cannot pay the bill: at today's numbers the crate would
radiate itself cold and go out. --- Good job finding this out - i want us to fix this!

There is a second, deeper wall behind that one. We measured how fast oxygen
can actually reach one burning tile in our world (from our own test data):
even planet-side with a sky, at most about a fifth of what a real crate-fire's
chemistry would demand. Real fires solve this by *sucking air in* — the rising
hot column pulls fresh air along the floor toward the fire. Our flat, 2D,
gravity-free world has no rising column: our fires *push* air away. And on
your ships there is no sky at all. **Air supply, not heat, is the true
governor of fire size in our world.**

This is where your scaling ruling changes everything. Since we are free to
choose effective sizes — a crate is slats and air, not a solid wood block; a
wall need not be massive — the honest question is not "how do we feed a real
127-kilowatt crate fire?" but "**what fire size do we scale the world to, so
that everything balances?**" Three coherent answers, now including yours:

- **Option 1 — scale the world to the air it has (my new recommendation,
  enabled by your ruling).** Pick the fire's true size to be what our oxygen
  transport can actually feed — roughly a strong campfire. Keep the flame-hot
  *look*. Choose effective fuel masses to match (a crate holds a few
  kilograms' worth of burn, not thirty). Then make spread work by scaling
  ignition sensitivity: thin, light fuels genuinely ignite at a fraction of
  the heat flux that thick timber needs — your "lower their catch-fire
  trigger" suggestion is real physics for slatted crates, not a cheat. And
  give crates the modest conduction you suggested, so touching crates also
  share heat honestly (the old reason for zero conduction — keeping the test
  bench simple — expires once Fixes A and B exist). Costs: we must verify by
  the same arithmetic that scaled-down radiation plus conduction plus
  convection really does ignite the neighbour on a timescale you like; the
  fireman's flashover then comes mostly from room-heat plus sensitive fuels,
  which needs the room-warming fix to carry more of the load.
- **Option 2 — real-size fires plus a designed air-supply channel.** Keep
  real crate-fire chemistry (a deliberate, literature-cited re-anchoring of
  the burn-rate — note this would *complete*, not reverse, your earlier
  ruling against a bare unjustified increase), and build one new term that
  stands in for the missing air-suction, shaped for both worlds (sky on
  planets; vents and doors on ships). Biggest payoff (seconds-scale radiative
  spread, ferocious room fires, fully "real" books), biggest build, and the
  supply term needs its own small design first.
  **Option 2b** what if we simualte tehe suction by just consuming O2 from tile within a  small distance of the fire, would that help? i like this option 2 best, and perhapsuse option 3 on top of it. I am afraid of option 3 due to the feedback loop it creates, if smoke can emit and absorb rays, we need to think carefully howto handle it, without the number of rays exploding. I'd absolutrely love having smoke light up like true black bodies- itwouldmake explosionslook so much better - but i am simply afraid of what it will do to our real time computations. Perhaps it's possible to calculate those rays on a slower timescale, and kind of add them up, let's say they liveon only 5 ticks or 10 ticks per second, but they presist so they comppund --- iu really do not know ,perhapswecanfind solutions for thisonline too. Otherwise - ilikeoption 2 or 2b (whichismeant tobe asimplified version of 2, my own idea, maybe it'snot good -i'd love to have aseparate discussion about this- so anchor the decision between us.
- **Option 3 — let the smoke carry the heat.** Make the soot-filled hot gas
  layer itself absorb and re-radiate heat (in real building fires this is the
  dominant channel and *the* driver of flashover). Physically the most
  faithful path to requirement 5; also the largest amount of new machinery.
  It layers naturally on top of either option above, later.

One more note here:
I dont like ooption 1 too much, not if it assumes 1-2 kg weight on the crates,
ihad thoiught about thecrates to be 10-30 kg, or even more
I was thinking in my head,( didnt talk to you about it yet) thatn when the crate works, we can rename it from furnitue to crate or even medium crate or crate30kg, then wecould design morekind of furniture, like crate10kg, table, etcetc, all with different stats (hp conductivityt,fuel etc)

one more thing we could do. Cheat a little, and let our game O2 be more potent than real world O2 - this would kind of siumulate the extra sucktion etc- i dont know what consequesnesthat wouldhave, if it'sonly some unrealistic nubmbers on how much O2 was spent, then im absolutely fine with it.

Bbtw- if it turns out that the ships get depleted of O2- we could add vents etc to resupply -but htis is a game map design question,not a fire system desig nquestion.
Perhapssome ships will have big O2 reservoirs, othersnotthat big.
Perhaps big space stations have vents that makesfires behave as if they were planetside... but all of that shouldnt affect how we build the fire system right now too much.

**What settles it: one cheap measurement, then your call.** The new test
bench (Fix D) measures the true maximum oxygen delivery to a burning tile —
open field, sealed rooms of several sizes, vented rooms — before any game
code changes. With that number on the table, the choice between Option 1's
scale and Option 2's supply-channel becomes concrete instead of speculative.
My recommendation is to run that measurement, then take Option 1 unless the
numbers or your spread-feel verdict (next paragraph) push us to Option 2.

**One feel question only you can answer, please think about it before the
session:** in the version you saw work this week, a crate ignited its
neighbour through radiation in five to thirteen seconds. That speed was never
play-tested — it existed only in bench logs, powered by the very bookkeeping
error we are now fixing. **How fast should fire jump from crate to crate,
when you imagine playing?** Seconds (aggressive, roomfuls go up fast),
half-a-minute-ish (deliberate, you can react), or minutes (fire is a slow
siege weapon)? The answer directly sets which option and which numbers we
choose.
Answer: i am thinking much slower than 13 s
i wasthinking more like 30s to 60s (from one crate to another). Especially initalally. i guessreally hot fires may spread faster.

## 6. The decisions list (everything that waits on you)

1. **The fire-scale choice above** (after the bench measurement) — plus your
   spread-speed feel verdict. -see above
2. **Crate conduction:** add a modest conductivity to crates so they share
   heat by touch — your lean, my agreement. Yes/no. YES
3. **Doors:** doors carry an ignition temperature but are flagged
   non-flammable (a leftover). Make both door states burnable when the
   material recipe lands? (Includes one test of a burning airlock door, since
   doors participate in pressure logic.) -- doors should be flammable, but it depends on the door material. we'll add several types of doors with ther iown materials and HP etc- but in principle there should exist burnable doors (and other metal doors that dont burn or do not burn easily)
4. **Room-feel targets:** once rooms warm honestly, how hot should a burning
   room *feel* — and how fast? (We will bring you curves to react to, not
   numbers to invent.) : This idont know what u mean. I think units may take damage if they are in too hot rooms for too long time. 
5. **Hot-air damage to units:** should standing in a superheated room hurt,
   in addition to the existing line-of-sight radiant heat? (Small addition;
   recommend yes, together with Fix B.) Yes- well i wonderif units also couldbe heated by their surroundings, inculding air - and that this will determine wether they take damage. Theri equipment could then determine what temperature ranges are safe for the unit wearing them.
6. **The smoulder mini-design:** commission it now for later, or leave
   "death is final" indefinitely? --- why isthisthe case? does it mean that O2 no longer heats objects  that are just under the ignition temperature? This could be fine for now. it would be cool i thikn if "blowing on smpldering material" could cause it to startr burning again- we'd probably need onre more threshhold then, for when O2 no longer affects tempereture, right? If it's too big, im fine with moving this down the line-but on the other hand, if we dont do it now, when shall we hten do it=? slot it in out planning?
7. **Config shape for material classes** (a small structural choice —
   recommend one shared section per fuel class). Yes weneed this- perhapsto be designed when  wood works perfectly andweknowall params we need?

## 7. What happens after you decide

Build order (each step gated by tests; nothing merges into the main game
without your play-test at the end): the test bench and its measurements
first; then the radiation fix at frozen dials (the bench will briefly read
"all fires die" — expected, listed, and restored by the very next step); then
the recalibration in the written order; then convection, wind and blow-out
together (their calibration anchors come from you at that point: "a marginal
fire dies at this wind, a strong one survives that wind"); then the material
recipe; then your tuning-and-play session, where the campfire fraction, the
knee position, room feel, and the wind anchors all get set by eye; then the
close-out: documentation folded into the engine's canon chapters, one
deliberate refresh of the regression baselines (this design counts as a new
arc with a fresh allowance for that), and the archive.

Rough honesty about your calendar: roughly three working sessions (the
fire-scale decision; the wind/feel anchors; the final tuning session) plus
about five shorter play-tests spread across the steps.

## 8. Direct answers to your latest questions

- **Space ships:** you are right, and the document now treats the sky as the
  special case. All choking/venting design and the supply measurement run on
  sealed-room scenarios first; planet-side is the easy case.
- **"Diffusion and wind were the source of O₂ renewal?"** Yes — plus, on
  planet maps only, the sky-edge refill. In the open-field test, heating
  thinned the air and oxygen wandered back in by diffusion; that diffusion
  rate is exactly the ceiling we measured, and it is the number the bench
  will pin down properly.
- **"Is 3D-to-2D just a scaling problem?"** Mostly yes — that is Option 1,
  and your instinct matches the reviews' conclusion. The one thing scaling
  alone cannot conjure is air *suction* (entrainment); if we ever want truly
  building-scale fires, that needs Option 2's supply term. For crate-and-room
  scale, scaling works. -- let me reiterate-  id like to discuss in the chat with you about option 1 vs option 2.
- **"Are crates movable?"** Not today — they are fixed tiles with a material
  row, exactly as you remembered. Movable "furniture objects" remain a
  possible future system, separate from this design, and they could use the
  simplified fire rules you suggested.
- **"Can we cheat on the ignition trigger?"** Yes, and it is not even a
  cheat: thin fuels ignite at far lower incident heat than massive ones.
  Choosing per-material ignition sensitivity is part of Option 1's recipe.
- **"We do not need to assume tiles are massive."** Agreed, and adopted as
  requirement 11. Every place the old reasoning assumed solid-wood tiles now
  reads "effective mass/fuel, chosen within reasonable limits". This is
  exactly what unlocks Option 1.

## 9. What was reviewed, in one paragraph

The technical version of this design went through two full adversarial review
rounds — four independent expert reviews per round, each attacking from a
different angle: fire physics, exact-arithmetic/GPU determinism and cost,
faithfulness to your recorded decisions, and simplicity. They produced about
180 findings, including: two genuine algebra errors in our proposed radiation
bookkeeping (both fixed, with the correction independently derived twice); the
discovery that air supply, not heat, governs fire size in our world; the
death of one superficially attractive option that could not spread fire at
all; a wrong "expert" test target (a laboratory formula that assumes buoyancy
our 2D world doesn't have); and three latent bugs in code that is already
shipped. The full reports are archived beside the technical document. A final
short review round runs after your decisions, before anything is built.

---

## 10. Resolutions (2026-08-02, after Erik's annotated read + two chat discussions)

Your comments above are preserved as written; this section records what they
decided. **The decision package is BLESSED.**

- **The big decision: Option 2b — your extended oxygen draw — is the supply
  mechanism.** Burning tiles draw oxygen from open cells within a small radius,
  reached only through connected open air from each open face (never through
  solids; walls breathe via their open faces exactly as you suggested). No
  oxygen-potency boost — we try without it; it is only a bounded fallback if
  the measurement says the draw alone falls short, because potency would weaken
  sealed-room smothering, which your ships cannot afford. Fire power: the
  flame-look scale that the measured supply can actually feed.
- **Burn times are per material family.** The 30 kg crate may honestly burn
  ~30 minutes (you like it). The tuning reference becomes a campfire-scale
  fuel — the 1–3 kg fires your intuition is actually calibrated on.
- **Spread is conduction-led** (crate conductivity: yes), 30–60 s to the next
  crate initially; radiation is the flashover channel; first ignition from
  cold takes minutes, like starting a campfire.
- **Air is heated at the fire only** — its own tile and its breathing faces,
  no vicinity radius; the wind spreads it. This also restores the hot-air
  visuals you missed (the black-body renderer reads air temperature — your
  correction was right).
- **Ambient is room temperature**; cold-space directions after a hull breach
  are a named later refinement.
- The wind explanation (what is emergent, what is designed, why the old
  flicker died and how it returns) was delivered and accepted; the flicker is
  a named feel item for the tuning session.
- Everything else from your margins: crates stay immovable tiles; doors become
  a family of materials; hot-air damage yes (gear-based heat protection logged
  for the unit system); grenades join the payload review; embers are slotted
  right after the wind fixes; classes get designed once wood is perfect;
  Python prototypes are allowed; one tuning night per new material is the
  accepted cost; the zombie sentence is corrected.

**What happens now:** one final short review round on the updated technical
document; then both documents are committed; then the test bench is built and
the supply-versus-radius measurement runs; then implementation begins in the
§7 order. Your next appearances: the wind anchors, the room-feel curves, and
the tuning-and-play session.
