#!/usr/bin/env python3
"""One learning-loop episode in a fresh interpreter, as a new CLI session would run it.

A real ``AIAgent`` with the default CLI tool surface (``hermes-cli``), memory and background review on, works in
the cell's ``HERMES_HOME`` as the orchestrator left it, on the fixture rebuilt for this episode. After the turn the
worker waits for the review fork to finish (its writes are what the next episode may inherit), grades the
workspace, audits every API call from Hermes' own log lines, and prints one JSON record.

Exit codes: 0 the record is valid (pass or fail); 3 infrastructure failure, never scored (no agent, an auth error,
or a call served off the pinned route).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import logging
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# A review fork replays the conversation once; ten minutes is far past any healthy run and bounds a hung one.
REVIEW_WAIT_S = 600
REVIEW_POLL_S = 0.5
_AUTH_MARKERS = ("401", "403", "auth", "Authentication", "No LLM provider", "invalid api key")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    for flag in ("cell", "episode", "model", "only", "upstream", "base-url", "arm", "out"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--provider", default="openrouter", help="openrouter, or custom for a local server")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--model-revision", default="", help="the exact weights and engine, e.g. a GGUF sha256")
    parser.add_argument("--context-length", type=int, help="the local server's context window")
    return parser.parse_args()


def _sandbox(cell: Path) -> tuple[Path, Path]:
    """Hermetic env before any Hermes import: one provider key, this cell's home, HOME inside the sandbox."""
    for var in list(os.environ):
        if var.endswith(("_API_KEY", "_TOKEN")) and var != "OPENROUTER_API_KEY":
            os.environ.pop(var)
    os.environ.pop("HERMES_PROFILE", None)
    home, workspace = cell / "hermes-home", cell / "ws"
    os.environ["HERMES_HOME"] = str(home)
    os.environ["TERMINAL_CWD"] = str(workspace)
    sys.path.insert(0, str(REPO))
    from evals.core_tool_deferral.sandbox import isolate_host
    isolate_host(str(cell), str(home))
    os.chdir(workspace)
    return home, workspace


def _write_config(home: Path, model: str, provider: str, base_url: str, context_length: int | None = None) -> None:
    """The model as a configured user's config.yaml names it (docs: integrations/providers.md); a local server is a
    ``custom`` provider with its base_url. An OpenRouter pin travels as AIAgent arguments, as the CLI derives them
    from provider_routing, and the review fork copies it from the parent."""
    lines = ["model:", f"  default: {model}", f"  provider: {provider}"]
    if provider != "openrouter":
        lines.append(f"  base_url: {base_url}")
    if context_length:  # what the managed runtime knows from its own preset; compression keys off it
        lines.append(f"  context_length: {context_length}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _learning_state(home: Path) -> dict:
    """What a later session can inherit: agent-created skills, the review fork's (curator-managed) ones among
    them, and the two memory files."""
    from tools import skill_usage
    memories = home / "memories"
    size = lambda name: len((memories / name).read_text(encoding="utf-8")) if (memories / name).exists() else 0
    usage = skill_usage.load_usage()
    return {"agent_skills": sorted(n for n in usage if skill_usage.is_agent_created(n)),
            "learned_skills": sorted(n for n in usage if skill_usage.is_curator_managed(n)),
            "memory_chars": size("MEMORY.md"), "user_chars": size("USER.md")}


def _await_reviews() -> dict:
    """Barrier: the review fork runs on a daemon thread; the next episode must see what it wrote."""
    start = time.monotonic()
    while any(t.name == "bg-review" and t.is_alive() for t in threading.enumerate()):
        if time.monotonic() - start > REVIEW_WAIT_S:
            return {"waited_s": round(time.monotonic() - start, 1), "timed_out": True}
        time.sleep(REVIEW_POLL_S)
    return {"waited_s": round(time.monotonic() - start, 1), "timed_out": False}


def _tool_counts(messages: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name", "?")
            counts[name] = counts.get(name, 0) + 1
    return counts


def _run_agent(args: argparse.Namespace, prompt: str) -> tuple[object, dict, str | None]:
    from run_agent import AIAgent
    pins = {"providers_allowed": [args.only], "provider_require_parameters": True} if args.provider == "openrouter" else {}
    agent = AIAgent(base_url=args.base_url, api_key=os.environ["OPENROUTER_API_KEY"], provider=args.provider,
                    model=args.model, quiet_mode=True, enabled_toolsets=["hermes-cli"], platform="cli",
                    max_iterations=args.max_iterations, session_id=f"ll-{args.arm}-{args.rep}-{args.episode}", **pins)
    try:
        return agent, agent.run_conversation(prompt) or {}, None
    except Exception as error:  # the record must say why the run ended; the traceback goes to stderr
        logging.getLogger(__name__).exception("the episode's run ended with an error")
        return agent, {}, f"{type(error).__name__}: {error}"[:500]


def _attribution(agent, convo: dict, messages: list[dict], graded: dict, episode, before: dict,
                 ledgers: tuple[list[dict], list[dict]], skills_at_start: str, error: str | None) -> dict:
    """The review's three stages and each carried skill's four, plus the digests that pin what was compared."""
    from evals.learning_loop import attribution as at, family_conventions as fam
    system_prompt = getattr(agent, "_cached_system_prompt", None) or ""
    offered = at.offered_skills(system_prompt)
    trigger = at.review_eligibility(messages, int(getattr(agent, "_skill_nudge_interval", 0) or 0),
                                    "skill_manage" in (getattr(agent, "valid_tool_names", None) or ()))
    checks = graded["checks"]
    adhered = all(checks[c] for c in fam.CONVENTION_CHECKS) and at.ran_command(messages, fam.CHECK_COMMAND)
    effective = all(checks[c] for c in fam.ORACLE_CHECKS)
    return {
        "review_eligible": trigger["eligible"], "review_trigger": trigger,
        "review_wrote": at.new_writes(*ledgers, at.REVIEW_ACTOR),
        "foreground_wrote": at.new_writes(*ledgers, at.FOREGROUND_ACTOR),
        "skills_offered": offered,
        "skill_chain": at.skill_chain(before["learned_skills"], offered, at.viewed_skills(messages), adhered, effective),
        "adhered": adhered, "oracle_pass": effective,
        "protocol_valid": error is None and bool(convo.get("completed"))
        and not any(convo.get(k) for k in ("failed", "partial", "interrupted")),
        "turn_exit_reason": convo.get("turn_exit_reason"),
        "context_length": getattr(getattr(agent, "context_compressor", None), "context_length", None),
        "compression_threshold_tokens": getattr(getattr(agent, "context_compressor", None), "threshold_tokens", None),
        "digests": {"suite": at.digest(Path(fam.__file__).read_bytes()), "gold": at.digest(episode.hidden_test),
                    "skills": skills_at_start, "prompt_policy": at.digest(system_prompt) if system_prompt else None},
    }


def main() -> int:
    args = _parse()
    cell = Path(args.cell)
    home, workspace = _sandbox(cell)
    _write_config(home, args.model, args.provider, args.base_url, args.context_length)
    from evals.learning_loop import attribution as at, audit, family_conventions as fam
    capture = audit.attach()
    episode = next(e for e in fam.EPISODES if e.id == args.episode)
    base = (cell / "base-commit").read_text(encoding="utf-8").strip()
    before = _learning_state(home)
    ledger_before, skills_at_start = at.ledger_rows(home), at.skills_digest(home)
    prompt = f"{episode.prompt}\n\n(The project is the git repository at {workspace}.)"
    started = time.monotonic()
    agent, convo, error = _run_agent(args, prompt)
    barrier = _await_reviews()
    wall = round(time.monotonic() - started, 1)
    graded = fam.grade(workspace, episode, base)
    calls = audit.summarise(capture.lines, args.upstream)
    after, ledger_after = _learning_state(home), at.ledger_rows(home)
    messages = convo.get("messages") or getattr(agent, "messages", []) or []
    record = {
        "record_version": 2, "model": args.model, "model_revision": args.model_revision or args.model,
        "arm": args.arm, "rep": args.rep, "episode": args.episode, "error": error,
        **graded, "wall_s": wall, "review_barrier": barrier, "audit": calls,
        "review_fired": bool(calls["reviews"]), "review_results": [r["result"] for r in calls["reviews"]],
        "main_tokens": {"prompt": getattr(agent, "session_prompt_tokens", None),
                        "completion": getattr(agent, "session_completion_tokens", None)},
        "tool_counts": _tool_counts(messages), "learning_before": before, "learning_after": after,
        "skills_created": sorted(set(after["agent_skills"]) - set(before["agent_skills"])),
        **_attribution(agent, convo, messages, graded, episode, before, (ledger_before, ledger_after),
                       skills_at_start, error),
    }
    out = Path(args.out)
    out.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    out.with_suffix(".transcript.json").write_text(json.dumps(messages, default=str), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("model", "arm", "rep", "episode", "score", "review_fired", "error")}))
    infra = (error is not None and any(m in error for m in _AUTH_MARKERS)) or not calls["route_ok"]
    return 3 if infra else 0


if __name__ == "__main__":
    sys.exit(main())
