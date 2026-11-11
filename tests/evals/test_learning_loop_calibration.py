"""RFC #96704: following a learned workflow is not evidence of improved task outcomes.

Reuse the conventions family's byte-identical repository and independent grader. Real
Hermes sessions learn the workflow through the review fork, then load and follow it.
Only the HTTP model is scripted: we prescribe the outcomes in both arms so the report
can be calibrated without paying for inference or making an efficacy claim.
"""
import json
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from evals.learning_loop import report
from tests.evals.learning_loop_scripted import KEY, ScriptedProvider, message_text

ROOT = Path(__file__).resolve().parents[2]
SKILL_NAME = "invoicer-workflow"
# This task is about ignoring non-positive quantities; workflow compliance cannot fix it.
FIX_TOTAL = (
    "python3 - <<'PY'\nfrom pathlib import Path\np = Path('invoicer/invoice.py')\n"
    "p.write_text(p.read_text().replace('item.quantity for item in self.items)', "
    "'item.quantity for item in self.items if item.quantity > 0)'))\nPY"
)
LOG_CHANGE = (
    "python3 - <<'PY'\nfrom pathlib import Path\np = Path('CHANGES.md')\n"
    "p.write_text(p.read_text().replace('## Unreleased\\n', "
    "'## Unreleased\\n\\n- invoice: check nonpositive quantities\\n', 1))\nPY"
)


class CalibrationProvider(ScriptedProvider):
    review_latency_s = 0
    carry_correct = False
    fresh_correct = False

    def _plan(self, messages):
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        last_user = users[-1] if users else 0
        prompt = message_text(messages[last_user]) if users else ""
        if "quantity is zero" not in prompt:
            return super()._plan(messages)
        steps = sum(m.get("role") == "tool" for m in messages[last_user:])
        system = " ".join(message_text(m) for m in messages if m.get("role") == "system")
        carried = SKILL_NAME in system
        script = [("skill_view", {"name": SKILL_NAME})] if carried else []
        correct = self.carry_correct if carried else self.fresh_correct
        if correct:
            script.append(("terminal", {"command": FIX_TOTAL}))
        # Both arms execute the same real project tests. Carry additionally follows the
        # learned commit/changelog conventions, regardless of whether the fix is correct.
        script.append(("terminal", {"command": "./scripts/check"}))
        if carried:
            script.extend(("terminal", {"command": cmd}) for cmd in (
                LOG_CHANGE, "git add -A", "git commit -q -m 'invoice: check nonpositive quantities'"))
        return ("control", *script[steps]) if steps < len(script) else (
            "control", None, "Completed the scripted calibration episode.")


@pytest.fixture
def calibration_provider():
    # Each case gets its own handler state and ephemeral port.
    handler = type("CaseProvider", (CalibrationProvider,), {"seen": [], "counter": [0]})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", handler
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("fresh_correct,carry_correct,expected_effect", [
    pytest.param(False, False, 0.0, id="same-failure"),
    pytest.param(True, True, 0.0, id="same-success"),
    pytest.param(False, True, 1.0, id="improvement"),
    pytest.param(True, False, -1.0, id="regression"),
])
def test_adherence_is_not_task_benefit(calibration_provider, tmp_path,
                                      fresh_correct, carry_correct, expected_effect):
    url, handler = calibration_provider
    handler.fresh_correct, handler.carry_correct = fresh_correct, carry_correct
    key_file = tmp_path / "key"
    key_file.write_text(KEY, encoding="utf-8")
    out = tmp_path / "run"
    proc = subprocess.run([
        sys.executable, str(ROOT / "evals/learning_loop/live.py"),
        "--key-file", str(key_file), "--out", str(out), "--base-url", url,
        "--provider", "custom", "--route", "fake/calibration==", "--reps", "1",
        "--episodes", "percent_of,skip_empty_lines", "--disposable-host",
    ], cwd=ROOT, capture_output=True, text=True, timeout=900, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rows = list(map(json.loads, (out / "results.jsonl").read_text().splitlines()))
    assert all(r["exit_code"] == 0 and r["protocol_valid"] for r in rows)
    by_case = {(r["arm"], r["episode"]): r for r in rows}
    for arm in ("carry", "fresh"):
        training = by_case[(arm, "percent_of")]
        assert training["oracle_pass"] and training["review_eligible"] and training["review_fired"]
        assert training["review_wrote"] == [{"action": "create", "skill": SKILL_NAME}]
    carry, fresh = (by_case[(a, "skip_empty_lines")] for a in ("carry", "fresh"))
    assert carry["digests"]["gold"] == fresh["digests"]["gold"]
    assert carry["digests"]["suite"] == fresh["digests"]["suite"]
    assert carry["digests"]["skills"] != fresh["digests"]["skills"]
    assert carry["skill_chain"] == [{"skill": SKILL_NAME, "eligible": True, "activated": True,
                                     "adhered": True, "effective": carry_correct}]
    assert fresh["skill_chain"] == [] and not fresh["adhered"]
    assert carry["oracle_pass"] is carry_correct and fresh["oracle_pass"] is fresh_correct
    # An attempted skill_view alone is insufficient: check that its response actually
    # returned the learned instructions and that the next commands followed them.
    cell = out / "cells" / "fake__calibration" / "carry" / "rep0"
    transcript = json.loads((cell / "skip_empty_lines.transcript.json").read_text())
    view_ids = {call["id"] for m in transcript for call in m.get("tool_calls") or []
                if call["function"]["name"] == "skill_view"}
    loaded = [message_text(m) for m in transcript
              if m.get("role") == "tool" and m.get("tool_call_id") in view_ids]
    assert loaded and any("invoicer workflow" in text and "CHANGES.md" in text for text in loaded)
    assert json.loads((out / "manifest.json").read_text())["key_found_in"] == []
    summary = report.summarise(rows)
    later = summary["models"]["fake/calibration"]["later_episodes"]
    # In all four cases the workflow-inclusive score rises. It must never substitute
    # for independent task benefit, which is zero, positive or negative as prescribed.
    assert later["score_diff_mean"] > 0
    assert "oracle_pass_diff_mean" in later, (
        "The report has no independent task-outcome contrast; its composite score rises "
        f"by {later['score_diff_mean']} while the known task effect is {expected_effect}.")
    assert later["oracle_pass_diff_mean"] == expected_effect
    assert f"oracle pass rate {expected_effect}" in report.render(summary)
