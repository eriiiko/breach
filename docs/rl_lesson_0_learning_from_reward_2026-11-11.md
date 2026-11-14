# Lesson 0: how a policy learns from reward

2026-11-11. The lesson document of rung 0 of the RL agent ladder
(`docs/rl_agent_roadmap_2026-11-11.md`, rule 13: every rung opens with one).
Written for Erik: no reinforcement-learning vocabulary is assumed, and a
little mathematics is. It explains the method the ladder has chosen, PPO,
with Breach's own soldier as the example. Rung 2's lesson will go through
the trainer line by line; this one is the idea.

## 1. The loop

Six times a second, for every soldier:

1. the soldier is shown its **observation** (itself, the list of things, the map window, its order);
2. its network answers with **choices** (feet, aim, trigger, hand);
3. the game runs four ticks;
4. the soldier is handed a **reward**: a number that says how that went (design section 6).

One match from spawn to end is an **episode**. The function from observation
to choices is the **policy**. Here it is the network of design section 8, and
"learning" means changing its 1.5 million weights.

What we want is a policy that collects as much reward as possible over a
whole match. Not at the next decision: over everything still to come.

## 2. The return: reward still to come

For a decision at time t, the **return** is the sum of the rewards from
there on, with later ones counted a little less:

    G(t) = r(t) + g·r(t+1) + g²·r(t+2) + ...

The factor g, a little below 1, is the **discount**. It does two things.
It keeps the sum finite, and it sets how far ahead the soldier looks: about
1 / (1 − g) decisions.

| g | looks ahead | a reward 100 decisions (17 s) away counts as |
|---|---|---|
| 0.99 | 100 decisions, 17 s | 37 % |
| 0.995 | 200 decisions, 33 s | 61 % |
| 0.998 | 500 decisions, 83 s | 82 % |

A five-minute match is 1 800 decisions. With a look-ahead of half a minute,
the +10 for winning is invisible from the first minute. That is why the
reward has the smaller "balance" terms that arrive during the fight, and why
the look-ahead is lengthened once learning is stable.

## 3. The policy gives chances, not answers

The network does not output "aim at enemy 3". It outputs a probability for
every option of every choice: 55 % enemy 3, 30 % enemy 1, 15 % spread over
the rest. The game then draws from those chances.

There are two reasons for this, and both are essential.

- **Trying things.** A soldier that always picks its current favourite never finds out that another option is better.
- **Something to adjust.** "Make this choice a little more likely" is a small, smooth change of the weights. "Pick a different option" is not. Learning by small steps needs the smooth version.

As training goes on the chances sharpen by themselves: 55 % becomes 90 %.

## 4. The one idea

> After a choice, look at what followed. If more reward followed than was
> expected, make that choice a little more likely in that situation. If
> less, a little less likely.

That is all of it. Everything else in this document is about doing this
without drowning in noise and without overshooting.

In mathematical terms it is gradient ascent on the expected return, with the
gradient estimated from samples (played matches). The estimate is unbiased
and extremely noisy: one decision is followed by a thousand others, by the
enemy's decisions and by luck, and all of that lands in the return.

## 5. "Than expected": the value network

Take a soldier alone against three enemies. Whatever it does, little reward
follows. If every choice were judged by the raw return, every choice in that
situation would be pushed down, the good ones with the bad.

So a second network, the **value network** (also called the critic), learns
to predict the return from a situation: "from here, about −1.2 is still to
come". A choice is then judged by the difference:

    advantage = what actually followed − what was expected

The soldier that retreated and collected −0.4 where −1.2 was expected did
well: +0.8. That choice becomes more likely.

Subtracting the expectation does not change what is learned on average. It
removes the part of the noise that had nothing to do with the choice. (For a
statistician: a control variate.) It is the most important trick in the
method.

The value network learns in the simplest way there is: after each batch of
matches, it is nudged toward the returns that were actually seen.

**In Breach** the value network is shown the whole match, also what the
soldier cannot see (design section 7). That is fair, because it never plays;
it only grades. The better it predicts, the less noise is left in the grades.

## 6. How far to trust the prediction

There are two ways to measure "what actually followed", and they fail in
opposite directions.

| measure | uses | weakness |
|---|---|---|
| the long way | all the real rewards to the end of the match | true on average, very noisy |
| the short way | the next reward, plus the value network's prediction from the following situation | calm, but wrong wherever the value network is wrong |

The standard answer is a blend of all lengths in between, set by one number
(called lambda, usually about 0.95: mostly the long way, damped). The
method's name is generalised advantage estimation. It is twenty lines of the
trainer and one of the first things rung 2 tests against a case worked by
hand.

## 7. The step-size problem, and what PPO adds

Sections 4 to 6 are the method from the 1990s. It has one nasty property.

The matches used for an update were played by the policy as it was *before*
the update. They say how to improve that policy a little. Take too large a
step and the policy lands somewhere the data says nothing about, and it may
be bad there. In ordinary machine learning a bad step is corrected by the
next batch. Here the next batch is *played by the damaged policy*, so the
data itself gets worse, and training can collapse and not return.

**PPO** (proximal policy optimization; "proximal" means "staying near") is
the fix that became the standard:

1. Play a batch of matches with the current policy. For every decision, store what was observed, what was chosen, and **the probability the choice had at that moment**.
2. Improve the policy on that batch in several small passes.
3. During those passes, compare each stored choice's probability now with what it was. Once it has moved by more than about 20 % in the direction that helps, that decision stops pushing. This is called the clip.
4. Throw the batch away and play a new one with the new policy.

A worked case. A choice had probability 0.20 when it was made, and its
advantage was +1.5, so the update makes it more likely. When it reaches
about 0.24, the clip ends its influence for this batch, however good the
advantage looked. One lucky match cannot turn the soldier into something
else.

The clip is a crude version of a sound idea (a trust region: only trust the
data near where it was collected), and its crudeness is its strength: it is
a few lines, and it is hard to get wrong.

## 8. Three things are optimised together

| term | pushes toward | why |
|---|---|---|
| the policy term (sections 4 to 7) | choices that did better than expected | the point |
| the value term | value predictions equal to the returns seen | so that "expected" means something |
| the entropy bonus | keeping the chances spread out | a small reward for staying undecided, so the soldier keeps trying things and does not lock onto its first idea |

## 9. One round of training, in Breach's numbers

1. **Collect.** The farm plays many matches at once with the current weights: about 3 000 soldier decisions a second on the small ship. A batch of 100 000 decisions takes about half a minute.
2. **Grade.** The value network predicts; advantages are computed for every decision (sections 5 and 6).
3. **Learn.** A handful of passes over the batch on the graphics card, under the clip.
4. **Repeat,** thousands of times. A duel is expected to take about ten million decisions; a fireteam a hundred million to a billion.

The method uses each match once and discards it. That makes it hungry for
matches, which is why the farm's speed, and later the GPU-resident
simulation, decide how much can be learned in a night.

## 10. What is particular to our soldiers

- **Several choices at once.** The probability of a whole decision is the product of its parts (feet, aim, then trigger, hand and argument given the aim). Everything above applies to that product. A part that had no effect, such as a reach when nothing is thrown, is left out.
- **Masks.** An option that is not legal gets probability zero, when playing and when learning. It can never be drawn and never be blamed.
- **One network for all soldiers.** Every soldier's decisions train the same weights, so a match with four soldiers yields four times the experience of a match with one.
- **Own and team reward.** With a shared reward a soldier is graded on things three others did. The value network absorbs much of that; the team share (design 6.3) sets how much is left.
- **The opponent learns too.** Against copies of itself the target moves: what worked yesterday is countered today. Hence the fixed opponents (the two bot generations, saved earlier selves) beside the current self, and a rating against a fixed pool to see real progress.
- **Memory.** The soldier's network carries a vector from decision to decision. Learning must replay each match in order to rebuild it. This is the part of a trainer that most often hides a bug, so the first task is trained without it.

## 11. Reading the curves at rung 2

What you will be shown, and what each means when it goes wrong.

| curve | healthy | when it is not |
|---|---|---|
| reward per match | rises, in steps and plateaus | flat from the start: the soldier cannot see or do what the task needs, the reward is too far away, or a bug. Check in that order |
| win rate against a fixed opponent | rises | reward rises but wins do not: the soldier has found a way to collect reward that is not winning. Watch a replay |
| how much of the return the value network explains (0 to 1) | climbs toward 0.5 to 0.9 | stays near 0: the grades are noise, nothing can be learned |
| entropy (how spread the chances are) | falls slowly | falls off a cliff: locked onto one behaviour too early. Never falls: nothing is being learned |
| how far each update moved the policy | small and steady | spikes: steps too large, the collapse of section 7 is near |

A replay is worth more than any curve. A learner optimises the reward we
wrote, not the game we meant, and the difference shows in what it does.

## 12. What this method is not

- **It does not plan.** At play time the network reacts to what it sees and remembers. All its foresight is habit learned from the return.
- **It does not understand.** It has no model of walls or bullets; it has weights that worked.
- **It is not frugal.** Millions of decisions for a simple skill. Our advantage is a fast, exactly repeatable simulator.
- **It does not transfer by itself.** A network trained on one build of the game belongs to that build (roadmap rule 10).

## Words

| word | meaning |
|---|---|
| policy | the function from observation to chances for each choice; here the soldier's network |
| return | the discounted sum of rewards still to come |
| discount | the factor below 1 by which later rewards count less; sets the look-ahead |
| value network, critic | predicts the return from a situation; used in training only |
| advantage | what followed minus what was expected |
| on-policy | learning only from matches played by the current policy; PPO is this |
| clip | PPO's limit on how far a choice's probability may move in one batch |
| entropy | how spread out the chances are; high means undecided |
| batch | the decisions collected between two updates |

## Sources

Named from memory, not re-read for this lesson, unless marked: Williams 1992
(the idea of section 4); Schulman et al. 2016 (the blend of section 6);
Schulman et al. 2017 (PPO); Yu et al. 2021 (PPO for teams, read by the
survey's digests); Huang et al. 2022, "The 37 implementation details of PPO"
(read by the survey's trainer digest,
`docs/research/rl_agent_survey_2026-11-11_raw/05_trainer_state_of_the_art.md`).
