# Critique, lens 1: the reward (sections 6, 4a, 5.4, 6.7, 7.2's value heads)

Critic: independent, adversarial, 2026-11-11. Read: the design in full (sections 4 to 7),
the roadmap's reward lines, and the code named below at `eeae95b3`. Nothing run.

## Verdict

**Survives with fixes.** The shape is right and simple: an outcome that dominates, one
state-read balance seen four ways, friendly-fire blame on the credited shooter only, an
explicit own/team blend. But three of section 6's stated properties are false against the
code as it stands, and one of them is a reward leak a learner will find: (1) the "bar" is
undefined for zombies, because a converted zombie has four times a spawned zombie's
health while both report the same maximum, so every bite moves both teams' balances by
four times what 6.2 says and luring the horde pays a third of what shooting pays; (2) the
own part is read from hit events, which carry overkill and hits on corpses, so it is not a
function of state and "own parts plus the uncredited rest equal the balance change" fails;
(3) the clock's −3 is ambiguous about dead rows, and on the reading "living soldiers only"
dying just before the time limit pays about +1.8. Two smaller structural items: the env
marks the clock as *truncated* (a trainer bootstraps across it, contradicting 6.5's "a real
ending"), and 7.2's single "balance" value head cannot follow the own-to-team schedule that
is 7.2's own reason for having per-group heads. All fixes are on paper and small.

## BLOCKER

None.

## MAJOR

**M1. "1 bar is the unit's full health" is undefined for zombies; conversion breaks 6.2's arithmetic and the luring channel.**
Evidence: a spawned zombie gets `current_hp = base_stats.vitality` (`simulation.py:545`,
`unit.py:251`; species vitality mean 100, `species.py:8`); a *converted* zombie gets
`current_hp = CFG.zombie.hp` = 400 (`ai_zombie.py:411`, `config.toml:4695`). The only
maximum the game has is `effective_vitality(unit)` = `base_stats.vitality` (`stats.py:49-56`;
no `max_hp`, `unit.py:38-39`), i.e. 100 for both. Measured against that, a fresh converted
zombie is 4 bars, and its appearance costs *every* marine team 4 x 0.25 = 1.0 (not "a little
more", 6.2 last bullet). Consequences, from full health:
- enemy marine shot dead: my team +1.5; the same marine bitten to death: +1.5 − 1.0 = **+0.5**.
  Erik's "make the horde work for you" (6.3) pays a third of shooting, and a soldier is paid to
  *finish* a weakened enemy before the horde reaches it;
- my soldier bitten: −2.5 for my team, against −1.5 when shot;
- 6.5's "everything a team can earn is 7" grows by 1.0 per conversion (killable new zombie health).
If instead the bar is `CFG.zombie.hp`, a spawned zombie is 0.25 bar and 6.5's ship_demo sum is
4 x 1.5 + 4 x 0.0625 = 6.25. Either way the document's numbers are not the game's.
Smallest fix: define the bar per class in 6.1 (marine: its spawn vitality; zombie: one number,
the zombie bar, = `[zombie] hp`), clamp health at 1 bar in the balance (also closes M-overheal,
m4), and report the spawned-vs-converted zombie health split to Erik as a game finding
(probably unintended; a gameplay ruling, not a reward one). Re-derive 6.2's last bullet and 6.5.

**M2. The own part is event-based, not state-based: overkill and corpse hits are paid, so "own parts plus the uncredited rest equal the balance change" (6.8) is false and the own part can be pumped.**
Evidence: `apply_packet` subtracts and reports the *applied* amount, never clamped to the
remaining health (`damage.py:369-393`); a packet landing on a unit an earlier packet of the
same slot killed still emits its `UnitHitEvent` (`damage.py:346-353`, `simultaneous.py:100-107`);
`EnvRunner._count` counts every positive-damage hit with no alive check (`rl_env.py:576-595`).
Arithmetic: a C4 (`unit_damage` 150, `config.toml:4417`) on a full marine credits 1.5 bars
+ 0.5 kill = 2.0 own, against a 1.5 state change; a frag (60) on a marine at 10 hp credits
0.6 bar for a 0.1 change; with exposure scaling (`damage.py:368-370`) overkill grows with focus
fire. Every round of a slot that lands after the kill is free own reward, and the same flaw puts
friendly-fire blame on hits to a team-mate's corpse. It biases learners toward explosives on
wounded targets and piling on, which is not winning. The 6.8 test, if "uncredited rest" is
computed independently (from `attacker_id == -1` events), fails on every overkill.
Smallest fix: in the env (outside the synced game, so no digest moves), walk each tick's events
in order with a running health per unit starting from its tick-start health, credit
`min(dmg, max(0, running))` and 0 on a corpse. Every hp write goes through `apply_packet` with
the tick's events (checked: `status.py:324-328`, `simultaneous.py:107/167`, the exchange sites;
`ai_zombie.py:411`, the conversion, is the one write outside it and is a state change the balance
already sees), so the reconstruction is exact. Then "own + uncredited = Δbalance" holds except
for the conversion term, which belongs to "uncredited".

**M3. The clock's −3: dead rows unspecified; on one reading suicide before the limit pays.**
Evidence: 6.2 says "every soldier of each team still alive"; 6.8's payment test names only "a
win or an elimination" for dead rows. If only living soldiers pay −3, a soldier of a team headed
for a draw that dies (walks into fire, vacuum, a zombie) late in the match trades −3 for its own
loss: at team share 0.3, 0.7 x 1.5 + 0.3 x 1.5 / 4 = 1.16; at 0.8, 0.3 + 0.3 = 0.6. Net gain
+1.84 to +2.4 per suicide. The trainer will find it in the last half-minute, which is exactly
where the clock is visible (6.5).
Smallest fix: "every row of each marine team that still has a living soldier, dead and turned
rows included", and add the clock to 6.8's "paid exactly once" test.

**M4. The env marks the time limit as truncation; 6.5 needs it to be a termination.**
Evidence: `rl_env.py:561-562` sets `terminated = len(living_teams) < 2`, `truncated = not
terminated` — the clock is `truncated`. A standard PPO/GAE trainer bootstraps V(s_final) across a
truncation, which treats the −3 as a cut rather than an ending and leaks value from beyond the
match. 6.5 says the clock is "a real ending ... not a cut the trainer has to guess across", which
is correct (time left is observed, 3.2) but not what the env says.
Smallest fix: patch 1e reports the time limit as `terminated` (keep a separate `reason` field for
statistics), or 6.5 states that the trainer treats `truncated` as terminal under this profile.

**M5. One "balance" value head cannot follow the scheduled team share.**
Evidence: 7.2 gives heads (outcome, balance, friendly fire, order) and justifies them by "a weight
can then be moved during a run ... the estimate needs no rebuilding"; 6.8 hands back own balance
and team balance as separate numbers; 6.3 moves the team share 0.3 → 0.8 during the run. A single
balance head is regressing a blended target whose blend changes, which is exactly the rebuild 7.2
says it avoids. Smallest fix: five heads matching 6.8's five numbers (outcome, own balance, team
balance, friendly fire, order).

## MINOR

m1. **Section 6.1's Ng-Harada-Russell claim is overstated.** Invariance needs F = γΦ(s') − Φ(s)
with the trainer's γ and Φ(terminal) = 0. Here F = Φ(s') − Φ(s) and the end balance is kept, so the
total return is outcome + Φ_end − Φ_0: draws are ranked by balance, wins by their cost. That is
probably wanted ("win well"), but then the balance is part of the objective, not pure shaping, and
6.9's "can the helper be removed" is a real behaviour change, not a speed one. Fix: say so, or pay
back −Φ_end at the end (true shaping). Recommend saying so.

m2. **"Two marine teams' balances mirror" is true only for marine terms.** B_A + B_B = −Z/2 (Z the
zombies' bars, 6.2's definition): any change of zombie health moves both teams equally. Killing
zombies is a public good shared with the enemy's team part. Harmless; reword.

m3. **"A healed enemy takes back exactly what hurting it paid" holds for the team part only.**
Heals are `UnitHitEvent`s with negative damage (`status.py:322-328`) and the counters skip
`damage <= 0` (`rl_env.py:581-582`), so the shooter's own part keeps 1 − s of it. No heal producer
exists today (REGEN has none; `status.py:130`), so it is latent. Fix: restrict the claim and the
6.8 test to the balance, or debit the own part on heals of a unit it damaged (not worth building).

m4. **Overheal is unbounded** (`status.py:60`, accepted gap of the game): once a heal exists, a
medic overhealing team-mates raises the balance without limit and keeps it at the end. Closed by
M1's clamp at 1 bar.

m5. **Friendly fire can be laundered through the world.** Blame needs a credited hit; a fire that
outlives its tick, a vented room, a shove into either, or a horde led onto a team-mate are all
uncredited (`damage.py:186-208`), so they cost the shooter only s x loss / 4 ≈ 0.11 at s = 0.3
against 5 for a credited kill. Record as an ACCEPTED GAP beside 6.3's third bullet, and read it off
in 6.9's friendly-fire experiment (team losses with no attacker while a team-mate was near).

m6. **A turned soldier's row.** 6.3 keeps dead rows on the books but does not say a *turned*
soldier's row is one, nor that hits the new zombie lands on its old team are not friendly fire. The
current counter rule does this right (`rl_env.py:570-574`: attacker row only while on the row's team
at tick start); 6.8 should name it as the rule.

m7. **Mutual elimination** (both last soldiers die in one tick): `MatchRules.result` is a draw with
no team (`ruleset.py:344-352`), `terminated` true. 6.2 pays −10 to both, which is consistent; add it
to the 6.8 tests so it is not paid as a −3 draw.

m8. **Order progress (6.7) must be read against one order.** If the order changes between two
decisions, Δdistance compares two places and pays the jump; completion must be once per order
instance, or a commander that reissues the same order lets a soldier step out and in for repeated
completion. Weight 0 until rung 4; write the two rules now.

## Checked and found sound

- Kill and death as "+0.5 per living soldier" is a state read; a corpse drops out of the balance
  at once (`damage.py:394-395`), conversion leaves the team (`ai_zombie.py:408`), and
  `living_teams` excludes zombies (`ruleset.py:335-338`), so elimination by bite is seen the same tick.
- Posthumous credit (a dead thrower's grenade or charge): the counter keys on team at tick start,
  so a dead unit's own part still receives it (`rl_env.py:570-574`); a *turned* thrower's later
  credit falls to "uncredited", which 6.3 already routes to the team part. Consistent.
- Self-damage is counted only as taken (`rl_env.py:588`), matching 6.2's last line.
- The outcome dominates the balance: per soldier at most about 1.75 (7 / 4) at full sharing,
  at most 7 at share 0, against 10 (holds after M1 at either bar definition).
- A one-for-one marine trade is worth nothing; the −3 / −10 gap rewards the losing side for
  surviving to the clock and the leading side for hunting, which is a property of the rules, not a
  leak. In three-team matches the doomed team can deny a winner by hiding; inherent, watched by 6.9.
- The match ending inside a decision: `step` breaks early and the state after the last tick is the
  balance's "after" (`rl_env.py:544-552`); no term is lost.
- Never blending the outcome and the friendly fire: no double payment found. Sum over a team of
  (1 − s)·own + s·Δteam/4 equals the team's Δbalance when own parts sum to the credited share (after M2).

## Not checked

- That every exchange-row `apply_packet` call passes the tick's event list (seen for status and the
  attack slots; the three `exchange.py` sites read only in part). UNVERIFIED.
- Whether `predefined_unit_attributes` gives every marine exactly 100 vitality (the species "means"
  comment suggests so). UNVERIFIED.
- Ticks per second and so the exact discount behind "half a minute"; not needed for any finding.
- The value network's handling of dead rows across GAE (section 7's lens).
