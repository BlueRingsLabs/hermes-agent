"""End-to-end pilot wiring, offline: real AIAgent sessions, real tools and the real review fork against a scripted
OpenAI-compatible provider on 127.0.0.1.

One cell per arm runs two episodes. Episode 1 is solved through Hermes' terminal tool in twelve tool turns, enough
to trigger the skill review, whose fork writes a skill. Episode 2 only looks around. The scripted provider records
what each request carried, so the test can check what the agent actually saw:

  - the provider pin reached the wire on every main and review request;
  - in episode 2 the learned skill is in the system prompt for "carry" and absent for "fresh" (the manipulation);
  - the grader gives the scripted solution full marks; every call is audited on the pinned route;
  - the key never reaches any output file.
"""
import json
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from evals.learning_loop import report
from tests.evals.learning_loop_scripted import EPISODE_1, KEY, MODES, ScriptedProvider as _Provider

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def provider():
    _Provider.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@pytest.mark.parametrize("mode", sorted(MODES))
def test_two_episodes_per_arm_wire_learning_through_the_real_agent(provider, tmp_path, mode):
    flag, route, _Provider.upstream = MODES[mode]
    # A local server's window reaches each episode's config. OpenRouter reports its own, so none is passed there.
    window = ["--context-length", "98304"] if mode == "local" else []
    key_file = tmp_path / "key"
    key_file.write_text(KEY, encoding="utf-8")
    out = tmp_path / "run"
    # --disposable-host: this test's container is throwaway and its "model" is the script above.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "evals/learning_loop/live.py"), "--key-file", str(key_file), "--out", str(out),
         "--base-url", provider, "--provider", flag, "--route", route, "--reps", "1", "--first-rep", "3",
         "--disposable-host", "--episodes", "percent_of,skip_empty_lines", *window],
        cwd=ROOT, capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rows = {(r["arm"], r["episode"]): r for r in map(json.loads, (out / "results.jsonl").read_text().splitlines())}
    assert set(rows) == {(a, e) for a in ("carry", "fresh") for e in ("percent_of", "skip_empty_lines")}
    # The block started at rep 3, and no call came near the point where Hermes compresses.
    assert {r["rep"] for r in rows.values()} == {3}
    for r in rows.values():
        assert 0 < r["audit"]["max_input"]["main"] < r["compression_threshold_tokens"] < r["context_length"]
        assert r["context_length"] == 98304 or mode == "openrouter"
    for arm in ("carry", "fresh"):
        first = rows[(arm, "percent_of")]
        assert first["exit_code"] == 0 and first["passed"], first.get("checks")
        assert first["review_fired"] and "invoicer-workflow" in first["skills_created"], first["audit"]
        assert first["audit"]["route_ok"] and first["audit"]["calls_main"] >= len(EPISODE_1)
        # The fork's create is the review's write, by the ledger's actor, and the solution followed the rules.
        assert first["review_wrote"] == [{"action": "create", "skill": "invoicer-workflow"}], first["review_wrote"]
        assert first["foreground_wrote"] == [] and "invoicer-workflow" in first["learning_after"]["learned_skills"]
        assert first["protocol_valid"] and first["adhered"] and first["oracle_pass"] and first["skill_chain"] == []
    # The replayed trigger agrees with Hermes' own on every row: twelve tool turns are eligible and fire, a
    # two-call episode is neither.
    assert all(r["review_eligible"] == r["review_fired"] for r in rows.values()), [
        (k, r["review_trigger"]) for k, r in rows.items()]
    # Episode 1 starts from the same skills in both arms, episode 2 differs only by what carry kept.
    assert rows[("carry", "percent_of")]["digests"]["skills"] == rows[("fresh", "percent_of")]["digests"]["skills"]
    carried, reset = rows[("carry", "skip_empty_lines")], rows[("fresh", "skip_empty_lines")]
    assert carried["digests"]["skills"] != reset["digests"]["skills"]
    assert "invoicer-workflow" in carried["skills_offered"] and "invoicer-workflow" not in reset["skills_offered"]
    assert carried["skill_chain"] == [{"skill": "invoicer-workflow", "eligible": True, "activated": False,
                                       "adhered": False, "effective": False}] and reset["skill_chain"] == []
    # On OpenRouter every chat request, the review fork's included, carries the pin; a local server gets none.
    sent = [s["provider"] for s in _Provider.seen]
    expected = {"only": ["fake-up"], "require_parameters": True} if mode == "openrouter" else None
    assert sent and all(p == expected for p in sent), sent
    second = [s["skill_in_system"] for s in _Provider.seen if s["kind"] == "episode2"]
    # Cells run carry first, then fresh; each episode 2 makes two requests (one tool turn, one answer).
    assert second[:2] == [True, True] and second[2:] == [False, False], second
    assert json.loads((out / "manifest.json").read_text())["key_found_in"] == []
    # The report reads the real record format: episode 1 at parity, nothing to separate the arms afterwards here.
    summary = report.summarise(list(rows.values()))["models"]["fake/learner"]
    assert summary["parity_episode_1"]["score_diff"] == [0.0]
    assert summary["later_episodes"]["score_diff_mean"] == 0.0 and summary["cost_per_episode_usd"] is None


def test_a_model_never_runs_as_root_unless_the_host_is_declared_disposable(monkeypatch, tmp_path):
    from evals.learning_loop import live
    key_file = tmp_path / "key"
    key_file.write_text(KEY, encoding="utf-8")
    monkeypatch.setattr(live.os, "geteuid", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["live.py", "--key-file", str(key_file), "--out", str(tmp_path / "run"),
                                      "--provider", "custom", "--base-url", "http://127.0.0.1:9/v1",
                                      "--route", "local/qwen=="])
    with pytest.raises(SystemExit, match="as root"):
        live.main()
    assert not (tmp_path / "run").exists()  # refused before anything was created
