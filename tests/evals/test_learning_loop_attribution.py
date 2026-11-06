"""evals/learning_loop/attribution.py: the review trigger replay, the offered-skills parse and the actor split."""
from __future__ import annotations

from evals.learning_loop import attribution as at


def _turn(*tools: str | None) -> list[dict]:
    """One user message, then one assistant message per iteration (None = the final answer, no tool call)."""
    messages = [{"role": "user", "content": "task"}]
    for i, tool in enumerate(tools):
        calls = [{"id": f"c{i}", "function": {"name": tool, "arguments": "{}"}}] if tool else []
        messages.append({"role": "assistant", "content": None if tool else "done", "tool_calls": calls})
        if tool:
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    return messages


def test_the_review_trigger_counts_iterations_since_the_last_skill_write():
    assert not at.review_eligibility(_turn(*["terminal"] * 8, None), 10, True)["eligible"]  # 9 iterations
    assert at.review_eligibility(_turn(*["terminal"] * 9, None), 10, True)["eligible"]  # 10, the final answer counts
    written = at.review_eligibility(_turn("terminal", "skill_manage", *["terminal"] * 8, None), 10, True)
    assert not written["eligible"] and written["iterations_since_skill_write"] == 9
    assert not at.review_eligibility(_turn(*["terminal"] * 20), 10, False)["eligible"]  # no skill_manage tool


def test_offered_skills_come_only_from_the_available_skills_block():
    prompt = ("  - not-a-skill: an indented list elsewhere in the prompt\n"
              "<available_skills>\n  devops:\n    - deploy-flow: Ship it.\n  - plain-skill: No category.\n"
              "</available_skills>\n    - also-not: after the block\n")
    assert at.offered_skills(prompt) == ["deploy-flow", "plain-skill"]
    assert at.offered_skills(None) == [] and at.offered_skills("no index here") == []


def test_ledger_writes_are_split_by_actor_and_only_new_entries_count():
    before = [{"id": "a", "actor": "curator", "action": "create", "skill": "old"}]
    after = before + [{"id": "b", "actor": "curator", "action": "patch", "skill": "learned"},
                      {"id": "c", "actor": "agent", "action": "create", "skill": "asked-for"}]
    assert at.new_writes(before, after, at.REVIEW_ACTOR) == [{"action": "patch", "skill": "learned"}]
    assert at.new_writes(before, after, at.FOREGROUND_ACTOR) == [{"action": "create", "skill": "asked-for"}]
