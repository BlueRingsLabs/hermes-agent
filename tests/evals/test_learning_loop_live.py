"""evals/learning_loop/live.py: a run resumes from its results after the machine went away mid-write, a failed
episode is run again before its cell goes on, and a server that stops answering stops the block."""
from __future__ import annotations

import argparse
import json
import sys

import pytest

from evals.learning_loop import live

EPISODES = ["e1", "e2", "e3"]


def test_a_line_cut_off_mid_write_is_not_a_finished_episode(tmp_path):
    results = tmp_path / "results.jsonl"
    done = {"model": "m", "arm": "carry", "rep": 0, "episode": "e1", "exit_code": 0}
    failed = {**done, "episode": "e2", "exit_code": 3}
    results.write_text(json.dumps(done) + "\n" + json.dumps(failed) + "\n" + json.dumps(done)[:25], encoding="utf-8")
    assert live._done(results) == {("m", "carry", 0, "e1")}
    assert live._done(tmp_path / "missing.jsonl") == set()


def test_a_resumed_run_starts_its_next_row_on_a_new_line(tmp_path):
    results = tmp_path / "results.jsonl"
    whole = json.dumps({"model": "m", "arm": "carry", "rep": 0, "episode": "e1", "exit_code": 0})
    results.write_text(whole + "\n" + whole[:30], encoding="utf-8")
    live._repair_tail(results)
    assert results.read_text(encoding="utf-8") == whole + "\n"


def _block(tmp_path, monkeypatch, arms, fails=None, answers=lambda: True, learn_first=0, writes_on=()):
    """A block of one rep over a stand-in worker, and the list of every episode it started. The worker leaves a mark
    in the cell's home: '<episode>.learned' when the episode succeeds, '<episode>.partial' when it fails (exit 124,
    the timeout's code). *fails* says how many times each episode fails before it succeeds. The review writes a
    skill in carry's episode 1 on the tries numbered in *writes_on*."""
    ran, fails = [], dict(fails or {})

    def worker(args, key, route, arm, rep, cell, episode):
        ran.append(f"{arm} {episode}")
        failed = fails.get(f"{arm} {episode}", 0) > 0
        fails[f"{arm} {episode}"] = fails.get(f"{arm} {episode}", 0) - 1
        (cell / "hermes-home" / f"{episode}.{'partial' if failed else 'learned'}").write_text("x", encoding="utf-8")
        wrote = arm == "carry" and episode == "e1" and ran.count("carry e1") in writes_on
        record = {"review_wrote": [{"action": "create", "skill": "s"}]} if wrote else {}
        (cell / f"{episode}.json").write_text(json.dumps(record), encoding="utf-8")
        return 124 if failed else 0

    monkeypatch.setattr(live, "_worker", worker)
    monkeypatch.setattr(live.preflight, "server_answers", lambda base, key: answers())
    template = tmp_path / "template-home"
    template.mkdir()
    (template / "installed").write_text("x", encoding="utf-8")
    args = argparse.Namespace(out=tmp_path, first_rep=0, reps=1, arms=arms, budget_usd=None, provider="custom",
                              base_url="http://127.0.0.1:1/v1", learn_first=learn_first)
    run = lambda: live._run_block(args, "key", [{"model": "m"}], EPISODES, template, tmp_path / "results.jsonl", None)
    return run, ran


def test_a_failed_episode_stops_its_cell_and_runs_again_from_the_home_it_started_with(tmp_path, monkeypatch):
    run, ran = _block(tmp_path, monkeypatch, "carry,fresh", fails={"carry e2": 1, "carry e3": 1})
    run()
    # carry stops at e2 while fresh runs whole, then e2 runs again before e3, and e3 gets its own second run
    assert ran == ["carry e1", "carry e2", "fresh e1", "fresh e2", "fresh e3", "carry e2", "carry e3", "carry e3"]
    home = tmp_path / "cells" / "m" / "carry" / "rep0" / "hermes-home"
    assert sorted(p.name for p in home.iterdir()) == ["e1.learned", "e2.learned", "e3.learned", "installed"]
    assert live._done(tmp_path / "results.jsonl") == {("m", a, 0, e) for a in ("carry", "fresh") for e in EPISODES}


def test_an_episode_that_fails_its_rerun_too_ends_its_cell_for_good(tmp_path, monkeypatch):
    run, ran = _block(tmp_path, monkeypatch, "carry", fails={"carry e2": 9})
    run()
    run()  # resuming the block later does not try it a third time
    assert ran == ["carry e1", "carry e2", "carry e2"]


def test_an_episode_the_server_stopped_under_is_not_scored_and_the_block_stops(tmp_path, monkeypatch):
    replies = iter([True])  # the server answers after e1, then never again
    run, ran = _block(tmp_path, monkeypatch, "carry", answers=lambda: next(replies, False))
    with pytest.raises(live._Stop, match="carry rep0 e2"):
        run()
    assert ran == ["carry e1", "carry e2"]  # nothing more starts against a server that is gone
    rows = live._rows(tmp_path / "results.jsonl")
    assert [(r["episode"], r["exit_code"]) for r in rows] == [("e1", 0), ("e2", live.INFRA_EXIT)]
    assert rows[1]["error"] == "the server stopped answering"
    # e2's starting home is kept, so the resumed block runs it again from there
    assert (tmp_path / "cells" / "m" / "carry" / "rep0" / "home.before-01").is_dir()


@pytest.mark.platforms("posix")
def test_a_second_run_on_the_same_output_is_refused(tmp_path, monkeypatch):
    import fcntl

    key_file, out = tmp_path / "key", tmp_path / "run"
    key_file.write_text("key", encoding="utf-8")
    out.mkdir()
    monkeypatch.setattr(sys, "argv", ["live.py", "--key-file", str(key_file), "--out", str(out), "--provider", "custom",
                                      "--base-url", "http://127.0.0.1:1/v1", "--route", "m==", "--disposable-host"])
    with (out / "run.lock").open("w", encoding="utf-8") as running:
        fcntl.flock(running, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a run already writing there
        with pytest.raises(SystemExit, match="another run is writing"):
            live.main()
    assert not (out / "results.jsonl").exists()


def test_episode_one_runs_in_carry_until_the_review_writes_and_fresh_starts_at_episode_two(tmp_path, monkeypatch):
    run, ran = _block(tmp_path, monkeypatch, "carry,fresh", learn_first=5, writes_on={3})
    run()
    assert ran == ["carry e1", "carry e1", "carry e1", "carry e2", "carry e3", "fresh e2", "fresh e3"]
    rows = live._rows(tmp_path / "results.jsonl")
    assert [r.get("learned") for r in rows if r["episode"] == "e1"] == [False, False, True]
    expected = {("m", "carry", 0, e) for e in EPISODES} | {("m", "fresh", 0, e) for e in EPISODES[1:]}
    assert live._done(tmp_path / "results.jsonl") == expected


def test_a_repetition_whose_episode_one_never_learns_runs_nothing_else(tmp_path, monkeypatch):
    run, ran = _block(tmp_path, monkeypatch, "carry,fresh", learn_first=2)
    run()
    run()  # resuming the block later does not try a third time
    assert ran == ["carry e1", "carry e1"]
