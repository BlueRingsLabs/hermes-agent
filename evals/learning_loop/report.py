#!/usr/bin/env python3
"""Pilot report: what carry-over did per model and episode, what an episode costs, and how many reps the full study
needs. Reads a run directory's results.jsonl, writes report.md and report.json next to it.

  python3 evals/learning_loop/report.py evals/learning_loop/runs/pilot

Rows with a non-zero exit code are infrastructure failures: counted and listed, never scored. Differences are
carry minus fresh, paired by rep (the same rep index ran the same episodes in both arms), with a bootstrap 95%
interval over reps. Required reps use the normal approximation n = ((z_a + z_b) * sd / delta)^2 for a paired
difference, alpha 0.05 two-sided and power 0.8; with three pilot reps the sd itself is rough, so treat it as a
floor to round up, not a promise.
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

Z_ALPHA, Z_POWER = 1.96, 0.8416  # two-sided 5%, 80% power
TARGET_SCORE_DELTA = 0.10  # a 10-point change in mean score
TARGET_TOKEN_RATIO = 0.25  # a 25% change in main-agent prompt tokens
BOOTSTRAP_DRAWS, SEED = 2000, 20261105


def _rows(run: Path) -> list[dict]:
    lines = (run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _ci(diffs: list[float]) -> list[float] | None:
    if len(diffs) < 2:
        return None
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(BOOTSTRAP_DRAWS))
    return [round(means[int(0.025 * BOOTSTRAP_DRAWS)], 4), round(means[int(0.975 * BOOTSTRAP_DRAWS) - 1], 4)]


def _reps_needed(diffs: list[float], delta: float) -> int | None:
    if len(diffs) < 2 or delta <= 0:
        return None
    sd = statistics.stdev(diffs)
    return max(2, math.ceil(((Z_ALPHA + Z_POWER) * sd / delta) ** 2)) if sd else 2


def _prompt_tokens(row: dict) -> float | None:
    value = (row.get("main_tokens") or {}).get("prompt")
    return float(value) if isinstance(value, (int, float)) else None


def _cell_summary(rows: list[dict]) -> dict:
    return {"n": len(rows), "pass_rate": _mean([float(r.get("passed", False)) for r in rows]),
            "score": _mean([r.get("score", 0.0) for r in rows]),
            "prompt_tokens": _mean([t for r in rows if (t := _prompt_tokens(r)) is not None]),
            "review_fired": _mean([float(r.get("review_fired", False)) for r in rows]),
            "cost_usd": _mean([r["cost_usd"] for r in rows if isinstance(r.get("cost_usd"), (int, float))]),
            "attribution": _attribution_summary(rows)}


def _attribution_summary(rows: list[dict]) -> dict:
    """Share of rows at each review stage, and of carried skills at each skill stage (#96704's chain)."""
    chain = [s for r in rows for s in r.get("skill_chain") or []]
    share = lambda items, key: _mean([float(bool(i.get(key))) for i in items])
    return {"review_eligible": share(rows, "review_eligible"), "review_fired": share(rows, "review_fired"),
            "review_wrote": share(rows, "review_wrote"), "carried_skills": len(chain),
            **{f"skill_{stage}": share(chain, stage) for stage in ("eligible", "activated", "adhered", "effective")}}


def _paired(scored: list[dict], model: str, episode: str, metric) -> list[float]:
    by_arm = {arm: {r["rep"]: metric(r) for r in scored if (r["model"], r["arm"], r["episode"]) == (model, arm, episode)}
              for arm in ("carry", "fresh")}
    reps = sorted(set(by_arm["carry"]) & set(by_arm["fresh"]))
    return [by_arm["carry"][k] - by_arm["fresh"][k] for k in reps
            if by_arm["carry"][k] is not None and by_arm["fresh"][k] is not None]


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r.get("exit_code") == 0]
    episodes = list(dict.fromkeys(r["episode"] for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))
    out = {"infra_failures": [{k: r.get(k) for k in ("model", "arm", "rep", "episode", "exit_code", "error")}
                              for r in rows if r.get("exit_code") != 0], "models": {}}
    for model in models:
        cells = {f"{arm}/{ep}": _cell_summary([r for r in scored if (r["model"], r["arm"], r["episode"]) == (model, arm, ep)])
                 for ep in episodes for arm in ("carry", "fresh")}
        later = episodes[1:]
        score_d = [d for ep in later for d in _paired(scored, model, ep, lambda r: r.get("score"))]
        token_d = [d for ep in later for d in _paired(scored, model, ep, _prompt_tokens)]
        fresh_tokens = _mean([t for r in scored if r["model"] == model and r["arm"] == "fresh"
                              and r["episode"] in later and (t := _prompt_tokens(r)) is not None])
        out["models"][model] = {
            "cells": cells,
            "parity_episode_1": {"score_diff": _paired(scored, model, episodes[0], lambda r: r.get("score"))},
            "later_episodes": {"score_diff_mean": _mean(score_d), "score_diff_ci95": _ci(score_d),
                               "prompt_token_diff_mean": _mean(token_d), "prompt_token_diff_ci95": _ci(token_d)},
            "cost_per_episode_usd": _mean([r["cost_usd"] for r in scored if r["model"] == model
                                           and isinstance(r.get("cost_usd"), (int, float))]),
            "reps_needed": {"score_10_points": _reps_needed(score_d, TARGET_SCORE_DELTA),
                            "prompt_tokens_25_percent": _reps_needed(
                                token_d, TARGET_TOKEN_RATIO * fresh_tokens if fresh_tokens else 0)}}
    return out


def render(summary: dict) -> str:
    lines = ["# Learning-loop pilot report", ""]
    for model, data in summary["models"].items():
        lines += [f"## {model}", "", "| arm/episode | n | pass rate | score | prompt tokens | review fired | cost $ |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for cell, s in data["cells"].items():
            lines.append(f"| {cell} | {s['n']} | {s['pass_rate']} | {s['score']} | {s['prompt_tokens']} | "
                         f"{s['review_fired']} | {s['cost_usd']} |")
        lines += ["", "| arm/episode | review eligible | fired | wrote | carried skills | listed | loaded | adhered "
                  "| oracle |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cell, s in data["cells"].items():
            a = s["attribution"]
            lines.append(f"| {cell} | {a['review_eligible']} | {a['review_fired']} | {a['review_wrote']} | "
                         f"{a['carried_skills']} | {a['skill_eligible']} | {a['skill_activated']} | "
                         f"{a['skill_adhered']} | {a['skill_effective']} |")
        later = data["later_episodes"]
        lines += ["", f"- Episode 1 parity (carry minus fresh score, per rep): {data['parity_episode_1']['score_diff']}",
                  f"- Later episodes, carry minus fresh: score {later['score_diff_mean']} (95% CI "
                  f"{later['score_diff_ci95']}), prompt tokens {later['prompt_token_diff_mean']} (95% CI "
                  f"{later['prompt_token_diff_ci95']})",
                  f"- Cost per episode: ${data['cost_per_episode_usd']}",
                  f"- Reps needed per arm for the full study: {data['reps_needed']}", ""]
    lines += ["## Infrastructure failures (not scored)", "", json.dumps(summary["infra_failures"], indent=1), ""]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(__doc__)
        return 2
    run = Path(sys.argv[1])
    summary = summarise(_rows(run))
    (run / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run / "report.md").write_text(render(summary), encoding="utf-8")
    print(render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
