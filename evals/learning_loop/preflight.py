#!/usr/bin/env python3
"""Pre-flight for the live learning-loop runs: prove every pinned model route works before an episode spends money.

For each model route the live arms use, one short tool-calling request goes through OpenRouter with the pin the
harness applies (``provider.only``, no fallbacks, ``require_parameters``). A route passes only if:

  - the request succeeds (HTTP 404 "no endpoints" usually means the account's privacy settings exclude the
    pinned provider; see OpenRouter, Settings, Privacy);
  - OpenRouter names the pinned provider as the one that served it;
  - the model answers with the requested tool call;
  - usage is reported.

The key must also carry a credit limit (``GET /api/v1/key``), so no run can spend past it.

The key is read from ``--key-file`` and is never printed, logged or passed on a command line.
Cost: one short completion per route.

  python3 evals/learning_loop/preflight.py --key-file ~/.config/openrouter/hermes-eval.key --out preflight.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://openrouter.ai/api/v1"
# The live arms' routes: Hermes' default model (model-catalog.json, "default": true) on its maker's endpoint, and a
# frontier model from another family on its maker's endpoint. ``only`` is OpenRouter's provider slug,
# ``provider_name`` how OpenRouter reports it, ``served_model`` the dated snapshot that endpoint listed on 2026-11-05.
ROUTES = (
    {"model": "z-ai/glm-5.3", "only": "z-ai", "provider_name": "Z.AI", "served_model": "z-ai/glm-5.3-20260816"},
    {"model": "anthropic/claude-sonnet-5.5", "only": "anthropic", "provider_name": "Anthropic",
     "served_model": "anthropic/claude-sonnet-5.5-20260928"},
)
# One short completion; a reasoning model may think for a while before the call, never for minutes.
REQUEST_TIMEOUT_S = 120
# Room for a reasoning model to think before emitting the call, so a truncated answer is not mistaken for a failure.
MAX_TOKENS = 2048
# Generation records appear within seconds of the response; 6 polls 2 s apart bound the wait at about 12 s.
GENERATION_POLLS = 6
GENERATION_POLL_INTERVAL_S = 2.0
TOOL = {"type": "function", "function": {
    "name": "record_number", "description": "Record one integer.",
    "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}}
PROMPT = "Call record_number with n set to 7. Reply only with the tool call."


def _request(base: str, key: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    """(HTTP status, parsed JSON body). Error bodies are returned, not raised; the key lives only in the header."""
    request = urllib.request.Request(
        base + path, data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", "replace")
        try:
            return error.code, json.loads(text)
        except ValueError:
            return error.code, {"error": {"message": text[:300]}}


def check_key(base: str, key: str) -> dict:
    status, body = _request(base, key, "/key")
    data = body.get("data") or {}
    limit = data.get("limit")
    ok = status == 200 and limit is not None
    detail = ("credit limit set" if ok else
              f"HTTP {status}" if status != 200 else "the key has no credit limit: set one before any run")
    return {"check": "key credit limit", "ok": ok, "detail": detail, "limit": limit,
            "limit_remaining": data.get("limit_remaining"), "usage": data.get("usage")}


def key_usage(base: str, key: str) -> float | None:
    """Spend so far on this key in USD, by OpenRouter's own counter (every charged call, whatever made it)."""
    status, body = _request(base, key, "/key")
    usage = (body.get("data") or {}).get("usage") if status == 200 else None
    return float(usage) if isinstance(usage, (int, float)) else None


def _tool_call_ok(body: dict) -> bool:
    calls = ((body.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
    if not calls:
        return False
    function = calls[0].get("function") or {}
    raw = function.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except ValueError:
        return False
    return function.get("name") == "record_number" and args.get("n") == 7


def _generation(base: str, key: str, generation_id: str | None) -> dict:
    """OpenRouter's record of who served the call, polled until it appears (bounded)."""
    for _ in range(GENERATION_POLLS if generation_id else 0):
        status, body = _request(base, key, f"/generation?id={generation_id}")
        if status == 200 and body.get("data"):
            return body["data"]
        time.sleep(GENERATION_POLL_INTERVAL_S)
    return {}


def check_route(base: str, key: str, route: dict) -> dict:
    payload = {"model": route["model"], "max_tokens": MAX_TOKENS, "tools": [TOOL], "tool_choice": "auto",
               "messages": [{"role": "user", "content": PROMPT}],
               "provider": {"only": [route["only"]], "allow_fallbacks": False, "require_parameters": True}}
    status, body = _request(base, key, "/chat/completions", payload)
    result = {"check": f"route {route['model']} via {route['only']}", "model": route["model"]}
    if status != 200:
        message = str((body.get("error") or {}).get("message", ""))[:200]
        hint = " (usually the account's privacy settings exclude this provider)" if status == 404 else ""
        return {**result, "ok": False, "detail": f"HTTP {status}: {message}{hint}"}
    generation = _generation(base, key, body.get("id"))
    provider = body.get("provider") or generation.get("provider_name")
    served = generation.get("model") or body.get("model")
    usage = body.get("usage") or {}
    problems = [p for p, bad in (
        (f"served by {provider!r}, not {route['provider_name']!r}", provider != route["provider_name"]),
        ("no record_number(n=7) tool call", not _tool_call_ok(body)),
        ("no usage reported", not usage.get("prompt_tokens"))) if bad]
    note = None if served == route["served_model"] else f"served model {served!r}, listed {route['served_model']!r}"
    return {**result, "ok": not problems, "detail": "; ".join(problems) or "pinned route works",
            "provider": provider, "served_model": served, "snapshot_note": note,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}


def check_local(base: str, key: str, model: str) -> dict:
    """A local OpenAI-compatible server (llama-server): it answers for *model* with the requested tool call."""
    payload = {"model": model, "max_tokens": MAX_TOKENS, "tools": [TOOL], "tool_choice": "auto",
               "messages": [{"role": "user", "content": PROMPT}]}
    status, body = _request(base, key, "/chat/completions", payload)
    usage = body.get("usage") or {}
    problems = [p for p, bad in ((f"HTTP {status}", status != 200), ("no record_number(n=7) tool call",
                                                                     status == 200 and not _tool_call_ok(body)),
                                 ("no usage reported", status == 200 and not usage.get("prompt_tokens"))) if bad]
    return {"check": f"local server {base} serving {model}", "ok": not problems, "model": model,
            "detail": "; ".join(problems) or "serves the model and calls tools", "served_model": body.get("model")}


def server_props(base: str, key: str) -> dict:
    """llama-server's own account of what it serves (model file, context, build), for the run's manifest."""
    status, body = _request(base.removesuffix("/v1"), key, "/props")
    return body if status == 200 else {"error": f"HTTP {status}"}


def server_answers(base: str, key: str) -> bool:
    """Whether a local server still answers at all. Any HTTP status counts, as llama-server says 503 while it loads a
    model. Only a refused, reset or silent connection means the process is gone."""
    request = urllib.request.Request(base.removesuffix("/v1") + "/health", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S):
            return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--key-file", type=Path, required=True, help="file holding the OpenRouter key, nothing else")
    parser.add_argument("--out", type=Path, help="write the observations (never the key) as JSON")
    parser.add_argument("--base-url", default=API, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    key = args.key_file.expanduser().read_text(encoding="utf-8").strip()
    if not key:
        parser.error("the key file is empty")
    checks = [check_key(args.base_url, key), *(check_route(args.base_url, key, r) for r in ROUTES)]
    for check in checks:
        print(f"{'PASS' if check['ok'] else 'FAIL'}  {check['check']}: {check['detail']}")
        if check.get("snapshot_note"):
            print(f"      note: {check['snapshot_note']} (recorded; the harness logs the served model per call)")
    if args.out:
        args.out.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    return 0 if all(check["ok"] for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
