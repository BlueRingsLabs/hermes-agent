"""The pilot report's arithmetic on rows with a known effect: carry beats fresh by exactly 0.25 on later episodes,
episode 1 is at parity, and an infrastructure failure is listed but never scored."""
import pytest

from evals.learning_loop import report


def _row(arm, episode, rep, score, tokens, exit_code=0):
    return {"model": "m", "arm": arm, "episode": episode, "rep": rep, "exit_code": exit_code, "score": score,
            "passed": score == 1.0, "main_tokens": {"prompt": tokens}, "review_fired": True, "cost_usd": 0.5}


def _rows():
    rows = []
    for rep, noise in enumerate((0.0, 0.05, -0.05)):
        rows += [_row("carry", "e1", rep, 0.5, 1000), _row("fresh", "e1", rep, 0.5, 1000),
                 _row("carry", "e2", rep, 0.75 + noise, 800), _row("fresh", "e2", rep, 0.5 + noise, 1000)]
    rows.append(_row("carry", "e2", 3, 0.0, 0, exit_code=3))
    return rows


def test_carry_minus_fresh_is_paired_by_rep_and_infra_rows_are_not_scored():
    model = report.summarise(_rows())["models"]["m"]
    later = model["later_episodes"]
    assert model["parity_episode_1"]["score_diff"] == [0.0, 0.0, 0.0]
    assert later["score_diff_mean"] == 0.25 and later["score_diff_ci95"] == [0.25, 0.25]
    assert later["prompt_token_diff_mean"] == -200.0
    assert model["cells"]["carry/e2"]["n"] == 3  # the exit-3 row is excluded


def test_infrastructure_failures_are_listed_with_whether_their_rerun_was_scored():
    failures = report.summarise([*_rows(), _row("carry", "e2", 0, 0.0, 0, exit_code=124)])["infra_failures"]
    assert failures == [
        {"model": "m", "arm": "carry", "rep": 3, "episode": "e2", "exit_code": 3, "error": None,
         "rerun_scored": False},
        {"model": "m", "arm": "carry", "rep": 0, "episode": "e2", "exit_code": 124, "error": None,
         "rerun_scored": True}]


def test_legacy_rows_do_not_imply_task_failure_or_benefit():
    model = report.summarise(_rows())["models"]["m"]
    assert model["later_episodes"]["oracle_reps"] == 0
    assert model["later_episodes"]["oracle_pass_diff_mean"] is None
    assert model["cells"]["carry/e2"]["oracle_n"] == 0
    assert model["cells"]["carry/e2"]["oracle_pass_rate"] is None


@pytest.mark.parametrize("unavailable", [
    {"oracle_pass": None},
    {"oracle_pass": "false"}, {"exit_code": 3},
])
def test_oracle_contrast_requires_two_explicit_outcomes_in_the_same_rep(unavailable):
    rows = [_row("carry", "e1", 0, 1.0, 100), _row("fresh", "e1", 0, 1.0, 100)]
    for rep in (0, 1):
        rows += [{**_row("carry", "e2", rep, 0.75, 100), "protocol_valid": True, "oracle_pass": False},
                 {**_row("fresh", "e2", rep, 0.5, 100), "protocol_valid": True, "oracle_pass": True}]
    rows[-1].update(unavailable)
    # Unmatched observations must not be pooled into an arm-level difference.
    rows.append({**_row("carry", "e2", 2, 1.0, 100), "protocol_valid": True, "oracle_pass": True})
    model = report.summarise(rows)["models"]["m"]
    assert model["later_episodes"]["oracle_reps"] == 1
    assert model["later_episodes"]["oracle_pass_diff_mean"] == -1.0
    assert model["cells"]["fresh/e2"]["oracle_n"] == 1
    assert model["cells"]["fresh/e2"]["oracle_pass_rate"] == 1.0


def test_required_reps_grow_with_the_spread_of_paired_differences():
    tight = report._reps_needed([0.1, 0.11, 0.09], 0.10)
    wide = report._reps_needed([0.4, -0.2, 0.1], 0.10)
    assert tight == 2 and wide > 20


def test_attribution_stages_are_reported_per_cell():
    chain = lambda activated, adhered: [{"skill": "s", "eligible": True, "activated": activated, "adhered": adhered,
                                         "effective": False}]
    rows = [{**_row("carry", "e2", 0, 1.0, 900), "review_eligible": True, "review_wrote": [{"action": "create"}],
             "skill_chain": chain(True, True)},
            {**_row("carry", "e2", 1, 0.5, 900), "review_eligible": False, "review_fired": False, "review_wrote": [],
             "skill_chain": chain(False, False)}]
    summary = report.summarise(rows)
    assert summary["models"]["m"]["cells"]["carry/e2"]["attribution"] == {
        "review_eligible": 0.5, "review_fired": 0.5, "review_wrote": 0.5, "carried_skills": 2,
        "skill_eligible": 1.0, "skill_activated": 0.5, "skill_adhered": 0.5, "skill_effective": 0.0}
    assert "| carry/e2 | 0.5 | 0.5 | 0.5 | 2 | 1.0 | 0.5 | 0.5 | 0.0 |" in report.render(summary)


def test_the_repetition_is_the_unit_and_incomplete_repetitions_do_not_count():
    rows = []
    for rep, (e2, e3) in enumerate(((0.5, 0.0), (0.25, 0.25))):  # each repetition averages to +0.25
        rows += [_row("carry", "e1", rep, 0.5, 900), _row("fresh", "e1", rep, 0.5, 900),
                 _row("carry", "e2", rep, 0.5 + e2, 900), _row("fresh", "e2", rep, 0.5, 900),
                 _row("carry", "e3", rep, 0.5 + e3, 900), _row("fresh", "e3", rep, 0.5, 900)]
    rows += [_row("carry", "e2", 2, 1.0, 900), _row("fresh", "e2", 2, 0.0, 900)]  # rep 2 never ran e3
    later = report.summarise(rows)["models"]["m"]["later_episodes"]
    assert later["score_diff_per_rep"] == [0.25, 0.25]
    assert later["score_diff_mean"] == 0.25 and later["score_diff_ci95"] == [0.25, 0.25]


def test_oracle_uses_complete_repetitions_instead_of_pooled_episode_pairs():
    rows = []
    for rep in (0, 1, 2):
        rows += [_row("carry", "e1", rep, 0.5, 900), _row("fresh", "e1", rep, 0.5, 900)]
        # The complete reps each average +0.5, with opposite episode effects.
        for ep in ("e2", "e3", "e4", "e5"):
            if rep == 2 and ep != "e2":
                continue  # this repetition is incomplete and must not count
            outcome = rep == 2 or (ep in ("e2", "e3")) == (rep == 0)
            rows += [{**_row("carry", ep, rep, float(outcome), 900), "oracle_pass": outcome,
                      "protocol_valid": True},
                     {**_row("fresh", ep, rep, 0.0, 900), "oracle_pass": False, "protocol_valid": True}]
    later = report.summarise(rows)["models"]["m"]["later_episodes"]
    assert later["oracle_pass_diff_mean"] == 0.5  # pooling would give 5/9
    assert later["oracle_reps"] == 2
    assert later["oracle_pass_diff_per_rep"] == [0.5, 0.5]
    assert later["oracle_pass_diff_ci95"] == [0.5, 0.5]  # resample reps, not episodes


@pytest.mark.parametrize("protocol_valid", [False, None])
def test_oracle_keeps_explicit_outcomes_when_the_model_breaches_protocol(protocol_valid):
    rows = [_row("carry", "e1", 0, 0.5, 900), _row("fresh", "e1", 0, 0.5, 900)]
    rows += [{**_row("carry", "e2", 0, 0.5, 900), "oracle_pass": False, "protocol_valid": protocol_valid},
             {**_row("fresh", "e2", 0, 1.0, 900), "oracle_pass": True, "protocol_valid": protocol_valid}]
    model = report.summarise(rows)["models"]["m"]
    assert model["later_episodes"]["oracle_pass_diff_mean"] == -1.0
    assert model["cells"]["carry/e2"]["oracle_n"] == model["cells"]["fresh/e2"]["oracle_n"] == 1
    assert model["cells"]["carry/e2"]["oracle_pass_rate"] == 0.0
    assert model["cells"]["fresh/e2"]["oracle_pass_rate"] == 1.0


def test_episode_one_tries_are_reported_and_only_the_one_that_learned_is_scored():
    tries = [{**_row("carry", "e1", 0, 0.25, 900), "learned": False}, {**_row("carry", "e1", 0, 1.0, 900), "learned": True}]
    summary = report.summarise(tries + [_row("carry", "e2", 0, 1.0, 800), _row("fresh", "e2", 0, 0.5, 1000)])
    model = summary["models"]["m"]
    assert model["first_episode"] == {"tries_per_rep": {0: 2}, "learned_reps": [0]}
    assert model["cells"]["carry/e1"]["n"] == 1 and model["cells"]["carry/e1"]["score"] == 1.0
    assert "learned in repetitions [0]" in report.render(summary)
