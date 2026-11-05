"""Behaviour of the live-run pre-flight against a local stand-in for OpenRouter (127.0.0.1, no network).

The pre-flight is the gate that stops a misrouted or uncapped run before it spends money, so each way it must
refuse is exercised: a key without a credit limit, a call served by another provider, a model that answers
without the tool call, and the account-privacy 404. The key must never reach any output.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from evals.learning_loop import preflight

KEY = "sk-or-test-key-0123456789abcdef"


class _OpenRouterStub(BaseHTTPRequestHandler):
    scenario: dict = {}

    def log_message(self, *args):  # keep the test output clean
        pass

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        assert self.headers["Authorization"] == f"Bearer {KEY}"
        url = urlparse(self.path)
        if url.path == "/key":
            return self._send(200, {"data": {"limit": self.scenario.get("limit", 50.0), "limit_remaining": 49.9,
                                             "usage": 0.1}})
        if url.path == "/generation":
            model = parse_qs(url.query)["id"][0].removeprefix("gen-")
            served = self.scenario.get("served_by", {}).get(model)
            route = next(r for r in preflight.ROUTES if r["model"] == model)
            return self._send(200, {"data": {"provider_name": served or route["provider_name"],
                                             "model": route["served_model"]}})
        return self._send(404, {"error": {"message": "unknown path"}})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = payload["model"]
        if "provider" in payload or model in {r["model"] for r in preflight.ROUTES}:  # a local check sends no pin
            assert payload["provider"] == {"only": [next(r["only"] for r in preflight.ROUTES if r["model"] == model)],
                                           "allow_fallbacks": False, "require_parameters": True}
        if model in self.scenario.get("no_endpoints", ()):
            return self._send(404, {"error": {"message": "No endpoints found matching your data policy"}})
        calls = [] if model in self.scenario.get("no_tool_call", ()) else [
            {"id": "call_1", "type": "function", "function": {"name": "record_number", "arguments": '{"n": 7}'}}]
        self._send(200, {"id": f"gen-{model}", "model": model,
                         "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}],
                         "usage": {"prompt_tokens": 120, "completion_tokens": 30}})


@pytest.fixture
def stub(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenRouterStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n", encoding="utf-8")

    def run(capsys, **scenario):
        _OpenRouterStub.scenario = scenario
        out = tmp_path / "preflight.json"
        code = preflight.main(["--key-file", str(key_file), "--out", str(out),
                               "--base-url", f"http://127.0.0.1:{server.server_address[1]}"])
        printed = capsys.readouterr()
        written = out.read_text(encoding="utf-8")
        assert KEY not in printed.out + printed.err + written  # the key never reaches any output
        return code, printed.out, json.loads(written)

    run.base = f"http://127.0.0.1:{server.server_address[1]}"
    yield run
    server.shutdown()


def test_every_pinned_route_passes(stub, capsys):
    code, printed, checks = stub(capsys)
    assert code == 0
    assert [c["ok"] for c in checks] == [True] * (1 + len(preflight.ROUTES))
    assert printed.count("PASS") == 1 + len(preflight.ROUTES)


def test_key_without_credit_limit_is_refused(stub, capsys):
    code, _, checks = stub(capsys, limit=None)
    assert code == 1
    assert checks[0]["ok"] is False and "no credit limit" in checks[0]["detail"]


def test_call_served_by_another_provider_fails_the_route(stub, capsys):
    code, _, checks = stub(capsys, served_by={"z-ai/glm-5.3": "Wafer"})
    glm = next(c for c in checks if c.get("model") == "z-ai/glm-5.3")
    assert code == 1
    assert glm["ok"] is False and "'Wafer'" in glm["detail"]


def test_answer_without_the_tool_call_fails_the_route(stub, capsys):
    code, _, checks = stub(capsys, no_tool_call={"anthropic/claude-sonnet-5.5"})
    sonnet = next(c for c in checks if c.get("model") == "anthropic/claude-sonnet-5.5")
    assert code == 1
    assert sonnet["ok"] is False and "tool call" in sonnet["detail"]


def test_privacy_excluded_provider_names_the_likely_cause(stub, capsys):
    code, _, checks = stub(capsys, no_endpoints={"z-ai/glm-5.3"})
    glm = next(c for c in checks if c.get("model") == "z-ai/glm-5.3")
    assert code == 1
    assert glm["ok"] is False and "privacy settings" in glm["detail"]


def test_local_server_passes_when_it_serves_the_model_and_calls_tools(stub, capsys):
    base = stub.base
    check = preflight.check_local(base, KEY, "local/qwen")
    assert check["ok"] and check["served_model"] == "local/qwen"


def test_local_server_without_the_tool_call_fails(stub, capsys):
    base = stub.base
    _OpenRouterStub.scenario = {"no_tool_call": {"local/qwen"}}
    check = preflight.check_local(base, KEY, "local/qwen")
    assert not check["ok"] and "tool call" in check["detail"]
