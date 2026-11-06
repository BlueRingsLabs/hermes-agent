"""Per-episode attribution: when the review could fire, what it wrote, and how far each carried skill got.

Review stages follow Hermes' own trigger. The skill review is eligible when the turn's tool-calling iterations
since the last skill_manage call reach skills.creation_nudge_interval: the counter rises once per iteration
(agent/turn_iteration_prep.py), skill_manage resets it (agent/tool_executor.py) and the turn's end compares it
(agent/turn_finalizer.py). Writes are told apart by the skill ledger's actor, which is "curator" for the review
fork and "agent" for the foreground session (tools/skill_ledger.py, derive_actor).

Skill stages follow the attribution chain proposed in #96704, per skill the episode carried in: eligible (listed
in the index the session was sent), activated (loaded with skill_view), adhered (the family's mechanical
convention checks hold) and effective (the family's independent oracle passes). The last two are properties of
the episode, recorded per skill so a row reads on its own. Whether a skill helps is the carry minus fresh
comparison, never one row.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

_INDEX_ROW = re.compile(r"^\s+- ([^:\s]+):", re.MULTILINE)
REVIEW_ACTOR = "curator"
FOREGROUND_ACTOR = "agent"


def digest(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def offered_skills(system_prompt: str | None) -> list[str]:
    """Skill names in the <available_skills> block of the system prompt the session sent."""
    if not system_prompt or "<available_skills>" not in system_prompt:
        return []
    block = system_prompt.split("<available_skills>", 1)[1].split("</available_skills>", 1)[0]
    return sorted(set(_INDEX_ROW.findall(block)))


def _arguments(function: dict) -> dict:
    raw = function.get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _calls(messages: Iterable[dict]) -> Iterator[tuple[str, dict]]:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            yield function.get("name", ""), _arguments(function)


def review_eligibility(messages: list[dict], interval: int, skill_manage_available: bool) -> dict:
    """Replays the skill-review trigger over one turn's messages."""
    since = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        since += 1
        if any((call.get("function") or {}).get("name") == "skill_manage" for call in message.get("tool_calls") or []):
            since = 0
    return {"eligible": skill_manage_available and interval > 0 and since >= interval,
            "iterations_since_skill_write": since, "interval": interval}


def viewed_skills(messages: list[dict]) -> set[str]:
    return {str(args["name"]) for name, args in _calls(messages) if name == "skill_view" and args.get("name")}


def ran_command(messages: list[dict], needle: str) -> bool:
    """Any terminal call whose command contains *needle*."""
    return any(needle in str(args.get("command", "")) for name, args in _calls(messages) if name == "terminal")


def ledger_rows(home: Path) -> list[dict]:
    path = home / "skills" / ".curator_ledger.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def new_writes(before: list[dict], after: list[dict], actor: str) -> list[dict]:
    """Ledger entries appended during the episode by *actor*, oldest first."""
    seen = {row.get("id") for row in before}
    return [{"action": row.get("action"), "skill": row.get("skill")}
            for row in after if row.get("id") not in seen and row.get("actor") == actor]


def skills_digest(home: Path) -> str:
    """One hash over every SKILL.md the episode started with, by path and content."""
    root = home / "skills"
    h = hashlib.sha256()
    for path in sorted(root.rglob("SKILL.md")) if root.exists() else []:
        h.update(str(path.relative_to(root)).encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def skill_chain(carried: Iterable[str], offered: Iterable[str], viewed: set[str], adhered: bool,
                effective: bool) -> list[dict]:
    listed = set(offered)
    return [{"skill": name, "eligible": name in listed, "activated": name in viewed, "adhered": adhered,
             "effective": effective} for name in sorted(carried)]
