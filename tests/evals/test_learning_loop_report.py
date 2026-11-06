"""The pilot report's arithmetic on rows with a known effect: carry beats fresh by exactly 0.25 on later episodes,
episode 1 is at parity, and an infrastructure failure is listed but never scored."""
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


def test_infrastructure_failures_are_listed():
    failures = report.summarise(_rows())["infra_failures"]
    assert failures == [{"model": "m", "arm": "carry", "rep": 3, "episode": "e2", "exit_code": 3, "error": None}]


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
