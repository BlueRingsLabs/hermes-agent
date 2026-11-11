#!/usr/bin/env python3
"""Learning-loop pilot: does what Hermes learns in one session make the next ones better or cheaper?

A cell is (model, arm, rep): the family's episodes, in order, each a separate session. Both arms run the default
product, review fork included, so inside an episode they are the same; they differ only in what carries over:

  carry  the cell's HERMES_HOME persists from episode to episode, as for a real user;
  fresh  HERMES_HOME is reset to a just-installed state before every episode.

Episode 1 is therefore the same in both arms (a built-in parity check), and carry minus fresh on later episodes
is the effect of carry-over alone.

  scripts/run-in-hermes-env python evals/learning_loop/live.py \\
      --key-file ~/.config/openrouter/hermes-eval.key --out evals/learning_loop/runs/pilot --budget-usd 40

A live run starts only after the pre-flight passes, reads spend from OpenRouter's per-key counter before every
episode and stops at the budget, and resumes where it stopped. An episode that fails (no valid record) stops its
cell, and the block runs it once more from the home it started with before the cell goes on. A local server that
stops answering stops the block. It refuses to run as root: the agent executes real shell commands, and the
operating system is the only boundary (SECURITY.md section 2.2).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evals.learning_loop import family_conventions as fam
from evals.learning_loop import preflight

ARMS = ("carry", "fresh")
# A 40-iteration episode with a review fork finishes in minutes; half an hour means a hang, and bounds the spend.
EPISODE_TIMEOUT_S = 1800
# A SIGKILLed process group is reaped in milliseconds; this only bounds a kernel that does not deliver it.
KILL_GRACE_S = 30
# OpenRouter's usage counter can trail the last response by a few seconds.
USAGE_SETTLE_S = 3.0
# How many more times an episode that failed (an infrastructure failure: no valid record) is run. A fixed number,
# so retrying cannot pick the outcome.
RERUNS = 1
INFRA_EXIT = 3  # episode.py's own code for a record that must not be scored


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", help="comma list of model ids (default: every pre-flight route)")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--first-rep", type=int, default=0,
                        help="number of the first rep, so separate sessions or machines can each run their own block")
    parser.add_argument("--episodes", help="comma list of episode ids (default: the whole family, in order)")
    parser.add_argument("--budget-usd", type=float, help="stop before an episode once this much is spent")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--episode-timeout", type=int, default=EPISODE_TIMEOUT_S,
                        help="seconds before an episode counts as hung (raise it for CPU or older GPUs)")
    parser.add_argument("--provider", default="openrouter", help="openrouter, or custom for a local server")
    parser.add_argument("--context-length", type=int,
                        help="the server's context window, written to each episode's config as model.context_length")
    parser.add_argument("--model-revision", default="",
                        help="the exact weights and engine served (e.g. the GGUF sha256), recorded on every row")
    parser.add_argument("--learn-first", type=int, default=0, metavar="N",
                        help="run episode 1 in the carry cell only, from a fresh install, up to N times until the "
                             "background review writes something; the fresh cell starts at episode 2")
    parser.add_argument("--disposable-host", action="store_true",
                        help="this machine is a throwaway sandbox (e.g. a Kaggle notebook): allow running as root")
    parser.add_argument("--base-url", default=preflight.API, help=argparse.SUPPRESS)
    parser.add_argument("--route", action="append", default=[], help=argparse.SUPPRESS)  # model=only=upstream
    return parser.parse_args()


def _routes(args: argparse.Namespace) -> list[dict]:
    if args.route:
        return [dict(zip(("model", "only", "provider_name"), r.split("="))) for r in args.route]
    wanted = set(args.models.split(",")) if args.models else None
    return [r for r in preflight.ROUTES if wanted is None or r["model"] in wanted]


def _seed_template(out: Path) -> Path:
    """One just-installed HERMES_HOME (bundled skills synced), copied wherever an arm starts fresh."""
    template = out / "template-home"
    if not (template / "skills").is_dir():
        template.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HERMES_HOME": str(template), "HOME": str(out / "template-user-home"),
               "PYTHONPATH": os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])])}
        seeded = subprocess.run([sys.executable, str(REPO / "evals/learning_loop/prompt_growth.py"), "--worker",
                                 "seed"], cwd=REPO, env=env, capture_output=True, text=True,
                                timeout=EPISODE_TIMEOUT_S, check=False)
        if seeded.returncode != 0:
            raise SystemExit(f"seeding the fresh-install home failed:\n{seeded.stderr[-3000:]}")
    return template


def _replace_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=True)


def _prepare_home(cell: Path, template: Path, arm: str, index: int) -> Path:
    """Restore this episode's starting home after an interruption, else derive it and snapshot it."""
    home, snapshot = cell / "hermes-home", cell / f"home.before-{index:02d}"
    if snapshot.exists():
        _replace_tree(snapshot, home)
    else:
        if arm == "fresh" or index == 0:
            _replace_tree(template, home)
        _replace_tree(home, snapshot)
    return snapshot


def _rows(results: Path) -> list[dict]:
    """Complete rows only: a line cut off when a machine went away mid-write is not a result."""
    if not results.exists():
        return []
    rows = []
    for line in results.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _repair_tail(results: Path) -> None:
    """Cut a line left unfinished when the machine went away, so the next row starts on a line of its own."""
    if results.exists():
        data = results.read_bytes()
        if data and not data.endswith(b"\n"):
            results.write_bytes(data[: data.rfind(b"\n") + 1])


def _attempt(row: dict) -> tuple:
    return row["model"], row["arm"], row["rep"], row["episode"]


def _done(results: Path) -> set[tuple]:
    return {_attempt(r) for r in _rows(results) if r.get("exit_code") == 0 and r.get("learned", True)}


def _failures(results: Path) -> Counter:
    return Counter(_attempt(r) for r in _rows(results) if r.get("exit_code") != 0)


def _worker(args: argparse.Namespace, key: str, route: dict, arm: str, rep: int, cell: Path, episode: str) -> int:
    out = cell / f"{episode}.json"
    cmd = [sys.executable, str(REPO / "evals/learning_loop/episode.py"), "--cell", str(cell), "--episode", episode,
           "--model", route["model"], "--only", route["only"], "--upstream", route["provider_name"],
           "--base-url", args.base_url, "--provider", args.provider, "--arm", arm, "--rep", str(rep), "--out", str(out),
           "--max-iterations", str(args.max_iterations), "--model-revision", args.model_revision]
    if args.context_length:
        cmd += ["--context-length", str(args.context_length)]
    env = {**os.environ, "OPENROUTER_API_KEY": key,
           "PYTHONPATH": os.pathsep.join([str(REPO), *filter(None, [os.environ.get("PYTHONPATH")])])}
    with (cell / f"{episode}.stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=stderr,
                                start_new_session=True)
        try:
            return proc.wait(timeout=args.episode_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # the agent's shell children go with it
            proc.wait(timeout=KILL_GRACE_S)
            return 124


def _spent(args: argparse.Namespace, key: str, start: float | None) -> float | None:
    now = preflight.key_usage(args.base_url, key) if start is not None else None
    return None if now is None or start is None else round(now - start, 6)


def _run_episode(args, key, route, arm, rep, cell, template, index, episode, usage_start) -> dict:
    snapshot = _prepare_home(cell, template, arm, index)
    (cell / "base-commit").write_text(fam.build_fixture(cell / "ws") + "\n", encoding="utf-8")
    before = _spent(args, key, usage_start)
    code = _worker(args, key, route, arm, rep, cell, episode)
    time.sleep(USAGE_SETTLE_S if usage_start is not None else 0)
    after = _spent(args, key, usage_start)
    record_path = cell / f"{episode}.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    if code == 0 and _server_down(args, key):  # it failed under the episode, whatever the record says
        code, record = INFRA_EXIT, {**record, "error": record.get("error") or "the server stopped answering"}
    if code == 0:
        shutil.rmtree(snapshot, ignore_errors=True)
    return {**record, "model": route["model"], "arm": arm, "rep": rep, "episode": episode, "exit_code": code,
            "cost_usd": None if before is None or after is None else round(after - before, 6),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _server_down(args: argparse.Namespace, key: str) -> bool:
    return args.provider != "openrouter" and not preflight.server_answers(args.base_url, key)


class _Stop(Exception):
    """The block cannot go on: the budget is spent, or the local server stopped answering."""


def _first_episode(args, results: Path, route: dict, rep: int, first: str) -> str:
    """Under --learn-first, where the carry cell's episode 1 stands: "learned" once the review wrote something,
    "due" while it may run again, "never" once its tries or reruns are spent (the repetition is then not scored)."""
    rows = [r for r in _rows(results) if _attempt(r) == (route["model"], "carry", rep, first)]
    if any(r.get("exit_code") == 0 and r.get("learned") for r in rows):
        return "learned"
    tries, failed = sum(r.get("exit_code") == 0 for r in rows), sum(r.get("exit_code") != 0 for r in rows)
    return "due" if tries < args.learn_first and failed <= RERUNS else "never"


def _run_cell(args, key, route, arm, rep, episodes, template, results, usage_start) -> bool:
    """Run one cell's episodes in order, and say whether one failed and is due to run again.

    The cell stops at a failed episode: in the carry arm every later episode starts from the home that one leaves.
    The block runs it again from the home it started with before the cell goes on, RERUNS times at most. After
    that the cell ends there for good, and its repetition is not scored.

    With --learn-first, episode 1 runs in the carry cell only, from a fresh install, until the background review
    writes something. The fresh cell waits for that and starts at episode 2."""
    if args.learn_first and arm != "carry":
        state = _first_episode(args, results, route, rep, episodes[0])
        if state != "learned":
            return state == "due"
    done, failures = _done(results), _failures(results)
    cell = args.out / "cells" / route["model"].replace("/", "__") / arm / f"rep{rep}"
    cell.mkdir(parents=True, exist_ok=True)
    for index, episode in enumerate(episodes):
        attempt, first = (route["model"], arm, rep, episode), bool(args.learn_first) and index == 0
        if attempt in done or (first and arm != "carry"):
            continue
        if failures[attempt] > RERUNS or (first and _first_episode(args, results, route, rep, episode) == "never"):
            return False
        where = f"{route['model']} {arm} rep{rep} {episode}"
        spent = _spent(args, key, usage_start)
        if args.budget_usd is not None and spent is not None and spent >= args.budget_usd:
            raise _Stop(f"budget reached before {where}: ${spent:.2f}")
        row = _run_episode(args, key, route, arm, rep, cell, template, index, episode, usage_start)
        if first and row["exit_code"] == 0:
            row["learned"] = bool(row.get("review_wrote"))
        with results.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        print(json.dumps({k: row.get(k) for k in ("model", "arm", "rep", "episode", "exit_code", "score",
                                                  "review_fired", "cost_usd")}))
        if row["exit_code"] != 0:
            if _server_down(args, key):
                raise _Stop(f"the server stopped answering during {where}: start it again, then resume")
            return failures[attempt] < RERUNS
        if row.get("learned") is False:
            return True  # nothing learned: episode 1 runs again from a fresh install in the next pass
    return False


def _run_block(args, key, routes, episodes, template, results, usage_start) -> None:
    """Every cell of the block, then more passes until no failed episode is due to run again."""
    cells = [(route, arm, rep) for rep in range(args.first_rep, args.first_rep + args.reps)
             for route in routes for arm in args.arms.split(",")]
    again = True
    while again:
        again = False
        for route, arm, rep in cells:
            again |= _run_cell(args, key, route, arm, rep, episodes, template, results, usage_start)


def _hold_lock(out: Path):
    """One writer per output folder: a second run would append to the same results and homes."""
    import fcntl  # POSIX only, like the process groups the episodes run in

    lock = (out / "run.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run is writing to {out}: let it finish, or stop it first") from None
    return lock


def _scan_for_key(out: Path, key: str) -> list[str]:
    needle = key.encode("utf-8")
    return [str(p) for p in out.rglob("*") if p.is_file() and needle in p.read_bytes()]


def _local_preflight_or_exit(args: argparse.Namespace, key: str, routes: list[dict]) -> dict:
    checks = [preflight.check_local(args.base_url, key, r["model"]) for r in routes]
    for check in checks:
        print(f"{'PASS' if check['ok'] else 'FAIL'}  {check['check']}: {check['detail']}")
    if not all(c["ok"] for c in checks):
        raise SystemExit("local pre-flight failed: nothing was run")
    return preflight.server_props(args.base_url, key)


def _preflight_or_exit(args: argparse.Namespace, key: str, routes: list[dict]) -> float:
    checks = [preflight.check_key(args.base_url, key), *(preflight.check_route(args.base_url, key, r) for r in routes)]
    for check in checks:
        print(f"{'PASS' if check['ok'] else 'FAIL'}  {check['check']}: {check['detail']}")
    if not all(c["ok"] for c in checks):
        raise SystemExit("pre-flight failed: nothing was run")
    return preflight.key_usage(args.base_url, key) or 0.0


def main() -> int:
    args = _parse()
    live = args.base_url == preflight.API
    if os.geteuid() == 0 and not args.disposable_host:
        raise SystemExit("refusing to run a model's shell commands as root: use an unprivileged account, or pass "
                         "--disposable-host on a throwaway sandbox")
    if live and args.budget_usd is None:
        raise SystemExit("a live run needs --budget-usd")
    key = args.key_file.expanduser().read_text(encoding="utf-8").strip()
    routes = _routes(args)
    episodes = args.episodes.split(",") if args.episodes else [e.id for e in fam.EPISODES]
    if args.learn_first and "carry" not in args.arms.split(","):
        raise SystemExit("--learn-first needs the carry arm")
    args.out.mkdir(parents=True, exist_ok=True)
    _lock = _hold_lock(args.out)  # held until this process exits
    usage_start = _preflight_or_exit(args, key, routes) if live else None
    server = _local_preflight_or_exit(args, key, routes) if args.provider != "openrouter" else None
    template = _seed_template(args.out)
    results, stopped = args.out / "results.jsonl", None
    _repair_tail(results)
    try:
        _run_block(args, key, routes, episodes, template, results, usage_start)
    except _Stop as stop:
        stopped = str(stop)
    done = _done(results)
    ended = [" ".join(map(str, a)) for a, n in _failures(results).items() if n > RERUNS and a not in done]
    never = [f"{route['model']} rep{rep}" for route in routes for rep in range(args.first_rep, args.first_rep + args.reps)
             if args.learn_first and _first_episode(args, results, route, rep, episodes[0]) == "never"]
    leaks = _scan_for_key(args.out, key)
    manifest = {"routes": routes, "provider": args.provider, "base_url": args.base_url, "server": server,
                "arms": args.arms, "reps": args.reps, "episodes": episodes, "live": live,
                "budget_usd": args.budget_usd, "spent_usd": _spent(args, key, usage_start), "stopped": stopped,
                "cells_ended_at": ended, "never_learned_in_episode_1": never, "key_found_in": leaks}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if leaks:
        print(f"ALARM: the key appears in {len(leaks)} output file(s); see manifest.json", file=sys.stderr)
        return 1
    if ended:
        print(f"{len(ended)} cell(s) ended at an episode that failed on its rerun too: {', '.join(ended)}")
    if never:
        print(f"episode 1 never learned anything in: {', '.join(never)} (not scored)")
    print(stopped or "run complete")
    return 2 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
