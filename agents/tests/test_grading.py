"""Grading (agents/grading.py), said exactly: the case worked by hand (note
2b, 4; design 8, critique lens 2 minors 1 and 5). Each test names the
property and the change that must break it."""
from __future__ import annotations

import numpy as np

from agents.grading import choice_grade, common_size, grade

# gamma = lambda = 1/2, so gamma * lambda = 1/4 and every number below is
# exact in binary: a person can follow each line.
GAMMA = LAM = 0.5
# outcome, own, team, friendly fire, order -- two groups at weight 0
WEIGHTS = (1.0, 1.0, 0.5, 0.0, 0.0)
T, R, G = 4, 5, 5
OUT, OWN, TEAM, FF, ORDER = range(5)


def _case():
    """The batch of the case worked by hand: 4 steps, 5 rows.

    row 0  soldier A of a two-soldier team (env 0): its match ENDS in step
           1 (done), the next match starts at step 2, and the batch ends in
           the middle of that match (the last view's estimate stands in);
    row 1  soldier B, A's team-mate: DIES in step 0 (a dead value row in
           step 1, still paid the team's and the outcome's rewards), the
           match ends in step 1, alive again in the next match;
    row 2  another env: its match ends ON THE BATCH'S LAST STEP (the last
           view is then the next match's: its estimate, 100, must not
           count);
    row 3  an env of three teams: the row's records END in step 1 while
           its match goes on (steps 2 and 3 are no value rows: their
           estimates 7 and the last view's 9 must not count);
    row 4  an empty roster place (in nothing; its estimates 5 ignored).
    """
    reward = np.zeros((T, R, G))
    value = np.zeros((T, R, G))
    last = np.zeros((R, G))
    done = np.zeros((T, R), dtype=np.uint8)
    vrow = np.ones((T, R), dtype=np.uint8)
    # row 0
    done[1, 0] = 1
    reward[:, 0, OUT] = [0, 10, 0, 0]
    value[:, 0, OUT] = [4, 6, 2, 2]
    last[0, OUT] = 2
    reward[:, 0, OWN] = [1, -0.5, 0, 2]
    last[0, OWN] = 4
    value[:, 0, TEAM] = [1, 1, 1, 1]
    last[0, TEAM] = 1
    reward[:, 0, FF] = [0, -1, 0, 0]
    # row 1
    done[1, 1] = 1
    reward[:, 1, OUT] = [0, 10, 0, 0]
    value[:, 1, OUT] = [5, 8, 2, 2]
    last[1, OUT] = 2
    reward[:, 1, OWN] = [-1.5, 0, 0, 0]
    value[:, 1, OWN] = [0, -1, 0, 0]
    reward[:, 1, TEAM] = [0, 0.5, 0, 0]
    # row 2
    done[3, 2] = 1
    reward[:, 2, OUT] = [0, 0, 0, -10]
    value[:, 2, OUT] = [-2, -4, -6, -8]
    last[2, :] = 100
    # row 3
    done[1, 3] = 1
    vrow[2:, 3] = 0
    reward[:, 3, OWN] = [0.5, -1.5, 0, 0]
    value[:, 3, OWN] = [1, 2, 7, 7]
    reward[:, 3, OUT] = [0, -10, 0, 0]
    value[2:, 3, OUT] = 7
    last[3, :] = 9
    # row 4
    vrow[:, 4] = 0
    value[:, 4, :] = 5
    last[4, :] = 5
    return reward, value, last, done, vrow


def test_the_case_worked_by_hand():
    """Protects: grading as the note says it, on the case worked by hand
    (every number below computed by hand, one line each): the estimate of
    the next view after a step that is not an ending, nothing still to come
    after an ending (a match's end, a match's end on the batch's last step,
    a row's records ending while its match goes on), the last view's
    estimate only at the batch's end in mid-match, a dead soldier graded on
    the rewards it is still paid, steps that are not value rows neither
    receiving nor passing anything on, and the weighted sum over five
    groups with two at weight 0.
    Breaks when: the ending's cut is dropped or moved by one step (the
    bootstrap across ``done``), the next match's view is counted after an
    ending, the trace ``gamma * lambda`` is replaced by ``gamma`` or
    ``lambda``, a non-value row passes its advantage back, the target is not
    ``adv + v``, or the weights are applied to the wrong group."""
    reward, value, last, done, vrow = _case()
    adv, target = grade(reward, value, last, done, vrow, GAMMA, LAM)
    want_adv = np.zeros((T, R, G))
    # row 0, outcome:  t3: 0 + .5*2 - 2 = -1          -> -1
    #                  t2: 0 + .5*2 - 2 = -1;  -1 + .25*-1 = -1.25
    #                  t1: done: 10 - 6 = 4             -> 4
    #                  t0: 0 + .5*6 - 4 = -1;  -1 + .25*4 = 0
    want_adv[:, 0, OUT] = [0, 4, -1.25, -1]
    # row 0, own:      t3: 2 + .5*4 - 0 = 4             -> 4
    #                  t2: 0 + 0 - 0 = 0;   0 + .25*4 = 1
    #                  t1: done: -0.5                   -> -0.5
    #                  t0: 1 + 0 - 0 = 1;   1 + .25*-0.5 = 0.875
    want_adv[:, 0, OWN] = [0.875, -0.5, 1, 4]
    # row 0, team:     t3: 0 + .5*1 - 1 = -0.5          -> -0.5
    #                  t2: -0.5 + .25*-0.5 = -0.625
    #                  t1: done: 0 - 1 = -1             -> -1
    #                  t0: 0 + .5*1 - 1 = -0.5; -0.5 + .25*-1 = -0.75
    want_adv[:, 0, TEAM] = [-0.75, -1, -0.625, -0.5]
    # row 0, friendly fire (weight 0): t1 done: -1; t0: 0 + .25*-1
    want_adv[:, 0, FF] = [-0.25, -1, 0, 0]
    # row 1 (dead in step 1), outcome: t3 -1; t2 -1.25; t1 done: 10 - 8 = 2;
    #                  t0: 0 + .5*8 - 5 = -1; -1 + .25*2 = -0.5
    want_adv[:, 1, OUT] = [-0.5, 2, -1.25, -1]
    # row 1, own:      t1 done: 0 - (-1) = 1
    #                  t0: -1.5 + .5*-1 - 0 = -2; -2 + .25*1 = -1.75
    want_adv[:, 1, OWN] = [-1.75, 1, 0, 0]
    # row 1, team (paid while dead): t1 done: 0.5; t0: 0 + .25*0.5 = 0.125
    want_adv[:, 1, TEAM] = [0.125, 0.5, 0, 0]
    # row 2, outcome:  t3 done: -10 - (-8) = -2 (the last view's 100 cut)
    #                  t2: 0 + .5*-8 + 6 = 2;  2 + .25*-2 = 1.5
    #                  t1: 0 + .5*-6 + 4 = 1;  1 + .25*1.5 = 1.375
    #                  t0: 0 + .5*-4 + 2 = 0;  0 + .25*1.375 = 0.34375
    want_adv[:, 2, OUT] = [0.34375, 1.375, 1.5, -2]
    # row 3, own:      t2, t3: no value rows -> 0 (their 7s and the last 9
    #                  never reach t1); t1 done: -1.5 - 2 = -3.5
    #                  t0: 0.5 + .5*2 - 1 = 0.5; 0.5 + .25*-3.5 = -0.375
    want_adv[:, 3, OWN] = [-0.375, -3.5, 0, 0]
    # row 3, outcome:  t1 done: -10 - 0 = -10; t0: 0 + 0 - 0 + .25*-10
    want_adv[:, 3, OUT] = [-2.5, -10, 0, 0]
    # row 4: an empty place, in nothing -> 0 everywhere
    np.testing.assert_array_equal(adv, want_adv)
    # target = adv + v on value rows, 0 elsewhere
    want_target = np.where(vrow[..., None] != 0, want_adv + value, 0.0)
    np.testing.assert_array_equal(target, want_target)
    assert target[0, 0, OUT] == 4 and target[2, 0, OUT] == 0.75
    assert target[0, 2, OUT] == -1.65625 and target[3, 3].sum() == 0
    # the choice's grade: outcome + own + 0.5 team (ff and order weigh 0)
    #   row 0: t0: 0 + 0.875 - 0.375 = 0.5;  t1: 4 - 0.5 - 0.5 = 3
    #          t2: -1.25 + 1 - 0.3125 = -0.5625; t3: -1 + 4 - 0.25 = 2.75
    #   row 1: t0: -0.5 - 1.75 + 0.0625 = -2.1875; t1: 2 + 1 + 0.25 = 3.25
    #          t2: -1.25; t3: -1
    #   row 2: the outcome's; row 3: t0: -2.875, t1: -13.5; row 4: 0
    g = choice_grade(adv, WEIGHTS)
    np.testing.assert_array_equal(g[:, 0], [0.5, 3, -0.5625, 2.75])
    np.testing.assert_array_equal(g[:, 1], [-2.1875, 3.25, -1.25, -1])
    np.testing.assert_array_equal(g[:, 2], [0.34375, 1.375, 1.5, -2])
    np.testing.assert_array_equal(g[:, 3], [-2.875, -13.5, 0, 0])
    np.testing.assert_array_equal(g[:, 4], [0, 0, 0, 0])


def test_grades_are_brought_to_a_common_size_after_the_sum():
    """Protects: the grades are brought to a common size AFTER the weighted
    sum (minus the mean, over the spread plus the floor), so a group that is
    always 0 adds nothing and divides by nothing, and the weights keep their
    meaning; equal grades give zeros, not a division by zero.
    Breaks when: each group is normalised before the weighting (the
    always-0 group then divides 0 by its floor and the weights' ratio is
    lost), or the floor is dropped."""
    rng = np.random.default_rng(3)
    adv = np.zeros((64, G))
    adv[:, OUT] = rng.normal(0, 10, 64)
    adv[:, OWN] = rng.normal(0, 0.1, 64)
    g = choice_grade(adv, WEIGHTS)
    n = common_size(g, 1e-8)
    assert abs(n.mean()) < 1e-12 and abs(n.std() - 1) < 1e-6
    # the large-scale group dominates the sum, as its weight says it should
    assert np.corrcoef(n, adv[:, OUT])[0, 1] > 0.999
    np.testing.assert_array_equal(common_size(np.full(9, 2.5), 1e-8),
                                  np.zeros(9))
    assert common_size(np.zeros(0), 1e-8).size == 0
