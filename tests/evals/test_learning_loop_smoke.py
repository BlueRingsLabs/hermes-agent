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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from evals.learning_loop import report

ROOT = Path(__file__).resolve().parents[2]
KEY = "sk-or-smoke-0123456789abcdef"
# (provider flag, --route, upstream the stand-in reports): OpenRouter names who served the call, llama-server does not.
MODES = {"openrouter": ("openrouter", "fake/learner=fake-up=FakeUp", "FakeUp"),
         "local": ("custom", "fake/learner==", None)}
# Long enough that a worker which skipped the review barrier would exit before the fork writes its skill.
REVIEW_LATENCY_S = 2.0
SKILL = ("---\nname: invoicer-workflow\ndescription: Test, commit and changelog rules for invoicer.\n---\n\n"
         "# invoicer workflow\n\n1. Run tests with ./scripts/check (it sets APP_MODE=test).\n"
         "2. Commit as \"<area>: <summary>\" (money, invoice, docs or tests).\n"
         "3. Add \"- <area>: <summary>\" under \"## Unreleased\" in CHANGES.md.\n")
EPISODE_1 = [
    "cat CONTRIBUTING.md", "ls invoicer tests scripts", "cat invoicer/money.py",
    "cat >> invoicer/money.py <<'EOF'\n\n\ndef percent_of(amount: Decimal, percent: Decimal) -> Decimal:\n"
    "    \"\"\"Percentage of an amount, rounded half-up to cents.\"\"\"\n"
    "    return (amount * percent / Decimal(\"100\")).quantize(Decimal(\"0.01\"), rounding=ROUND_HALF_UP)\nEOF",
    "cat >> tests/test_money.py <<'EOF'\n\n\nclass PercentOfTest(unittest.TestCase):\n"
    "    def test_rounds_half_up_to_cents(self):\n        from invoicer.money import percent_of\n"
    "        self.assertEqual(percent_of(Decimal(\"10.01\"), Decimal(\"50\")), Decimal(\"5.01\"))\nEOF",
    "./scripts/check",
    "python3 - <<'EOF'\nimport pathlib\np = pathlib.Path('CHANGES.md')\n"
    "p.write_text(p.read_text().replace('## Unreleased\\n', '## Unreleased\\n\\n- money: add percent_of helper\\n', 1))\nEOF",
    "cat CHANGES.md", "git add -A", "git commit -q -m 'money: add percent_of helper'", "git log -1 --format=%s",
    "git status --short",
]


def _text(message):
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


class _Provider(BaseHTTPRequestHandler):
    seen: list = []
    upstream: str | None = None
    counter = [0]

    def log_message(self, *args):
        pass

    def do_GET(self):
        self._json(404, {"error": {"message": "not found"}})

    def _json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _plan(self, messages):
        """(kind, tool name, arguments) or (kind, None, final text) for this request."""
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        last_user = users[-1] if users else 0
        steps = sum(1 for m in messages[last_user:] if m.get("role") == "tool")
        prompt = _text(messages[last_user]) if users else ""
        if "Review the conversation above" in prompt:
            script = [("skills_list", {}), ("skill_manage", {"action": "create", "name": "invoicer-workflow",
                                                             "category": "software-development", "content": SKILL})]
            return ("review", *script[steps]) if steps < len(script) else ("review", None, "Saved the workflow.")
        if "percent_of" in prompt:
            return ("episode1", "terminal", {"command": EPISODE_1[steps]}) if steps < len(EPISODE_1) else (
                "episode1", None, "Added percent_of, logged it and committed it.")
        if "Call record_number with n set to 7" in prompt:  # the local pre-flight's probe
            return ("preflight", "record_number", {"n": 7}) if steps < 1 else ("preflight", None, "Recorded.")
        if "quantity is zero" in prompt:
            return ("episode2", "terminal", {"command": "git status --short"}) if steps < 1 else (
                "episode2", None, "Looked at the repository.")
        return ("other", None, "invoicer work")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.path.endswith("/chat/completions"):  # e.g. the Ollama /api/show probe a local base URL invites
            return self._json(404, {"error": {"message": "not found"}})
        messages = body.get("messages") or []
        kind, tool, payload = self._plan(messages)
        if kind == "review":
            # Injected provider latency, not a wait: the fork must outlive grading so a worker without the review
            # barrier loses its skill; with the barrier the test passes at any load.
            time.sleep(REVIEW_LATENCY_S)  # health: allow HX013 -- injected provider latency, not a wait
        system = " ".join(_text(m) for m in messages if m.get("role") == "system")
        _Provider.seen.append({"kind": kind, "provider": body.get("provider"),
                               "skill_in_system": "invoicer-workflow" in system})
        self.counter[0] += 1
        if tool:
            delta = {"role": "assistant", "content": None, "tool_calls": [{
                "index": 0, "id": f"call_{self.counter[0]}", "type": "function",
                "function": {"name": tool, "arguments": json.dumps(payload)}}]}
            finish = "tool_calls"
        else:
            delta, finish = {"role": "assistant", "content": payload}, "stop"
        usage = {"prompt_tokens": len(json.dumps(messages)) // 4, "completion_tokens": 20}
        usage["total_tokens"] = usage["prompt_tokens"] + 20
        head = {"id": f"gen-{self.counter[0]}", "created": 0, "model": body.get("model")}
        if self.upstream:
            head["provider"] = self.upstream
        if not body.get("stream"):
            message = {k: v for k, v in delta.items() if k != "tool_calls"}
            if tool:
                message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in delta["tool_calls"]]
            return self._json(200, {**head, "object": "chat.completion", "usage": usage,
                                    "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        chunks = [{**head, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}]},
                  {**head, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {},
                                                                           "finish_reason": finish}]},
                  {**head, "object": "chat.completion.chunk", "choices": [], "usage": usage}]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


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
    key_file = tmp_path / "key"
    key_file.write_text(KEY, encoding="utf-8")
    out = tmp_path / "run"
    # --disposable-host: this test's container is throwaway and its "model" is the script above.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "evals/learning_loop/live.py"), "--key-file", str(key_file), "--out", str(out),
         "--base-url", provider, "--provider", flag, "--route", route, "--reps", "1", "--disposable-host",
         "--episodes", "percent_of,skip_empty_lines"],
        cwd=ROOT, capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rows = {(r["arm"], r["episode"]): r for r in map(json.loads, (out / "results.jsonl").read_text().splitlines())}
    assert set(rows) == {(a, e) for a in ("carry", "fresh") for e in ("percent_of", "skip_empty_lines")}
    for arm in ("carry", "fresh"):
        first = rows[(arm, "percent_of")]
        assert first["exit_code"] == 0 and first["passed"], first.get("checks")
        assert first["review_fired"] and "invoicer-workflow" in first["skills_created"], first["audit"]
        assert first["audit"]["route_ok"] and first["audit"]["calls_main"] >= len(EPISODE_1)
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
