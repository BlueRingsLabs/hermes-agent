# Learning-loop evals

Hermes leads with its learning loop (`README.md:19`): memory the agent curates, skills it writes after
complex tasks, a background review that forks after turns to save both, and a curator that ages unused
skills out. This directory measures what that loop **costs** (`prompt_growth.py`, offline) and runs a **pilot of
what it buys** (`live.py`, real models). The outcome question is the one
[RFC #96704](https://github.com/NousResearch/hermes-agent/issues/96704) asks, and its first arm is
[#122544](https://github.com/NousResearch/hermes-agent/pull/122544); nothing here duplicates them.

## `prompt_growth.py`: what learning costs every later API call

Offline and deterministic: no model calls, no network, no credentials. It drives Hermes' own code
against a throwaway `HERMES_HOME` and `HOME`, created per run and removed afterwards:

| step | code path |
|---|---|
| fresh install | `tools.skills_sync.sync_skills()` seeds the bundled skills |
| learning | `tools.skill_manager_tool.skill_manage(action="create")` with the write origin bound to the background-review fork, as `run_agent._spawn_background_review` binds it: every write passes the fork's guards, the 60-character description budget and the ledger, and is recorded `created_by: agent` |
| the next session | `agent.prompt_builder.build_skills_system_prompt()` for a default CLI session (`toolsets: [hermes-cli]`, plugins discovered before the tool snapshot as `agent/agent_init.py` does), in a fresh interpreter per checkpoint |
| memory | `tools.memory_tool_store.MemoryStore.add()` until the store refuses, then a fresh store's frozen system-prompt snapshot |

From the repository root:

```bash
scripts/run-in-hermes-env python evals/learning_loop/prompt_growth.py --out evals/learning_loop/results/<label>
uv run --no-project --with "tiktoken>=0.9,<1" python evals/learning_loop/retokenize.py evals/learning_loop/results/<label>
```

The first command makes two full, independent runs (about six minutes here) and writes `results.json`,
`results.md` and every rendered block. The second recounts those blocks with `o200k_base` in a throwaway
environment, so the Hermes environment is never modified.

### Results

[`results/2026-11-05-linux-py3.14-bare/`](results/2026-11-05-linux-py3.14-bare/): Hermes code identical to
`main` at `90c19e31`; the exact commit is in `results.json`.

| learned skills | index rows | `o200k_base` tokens | Hermes estimate | share of 32k | share of 128k |
|---:|---:|---:|---:|---:|---:|
| 0 (fresh install) | 39 | 1,015 | 1,084 | 3.1% | 0.8% |
| 10 | 49 | 1,211 | 1,278 | 3.7% | 0.9% |
| 25 | 64 | 1,474 | 1,566 | 4.5% | 1.1% |
| 50 | 89 | 1,944 | 2,048 | 5.9% | 1.5% |
| 100 | 139 | 2,875 | 3,000 | 8.8% | 2.2% |
| 200 | 239 | 4,914 | 4,980 | 15.0% | 3.7% |
| 400 | 439 | 8,989 | 8,940 | 27.4% | 6.9% |

Each learned skill adds **19.95 tokens** (least squares, `o200k_base`; Hermes' own estimate gives 19.64)
to the system prompt of every later session, on every API call of that session. The two counts agree
within 7% at every checkpoint. At capacity, `MEMORY.md` and `USER.md` together cost **941 tokens** and
stop there: the store refuses the next write (after 18 and 11 entries of this length).

What this shows:

- Memory does not get more expensive as the agent learns. The character limits
  (`hermes_cli/config_defaults.py:1306-1307`) are hard caps on writes.
- The skills index does, linearly. Every learned skill stays listed in every new session
  (`agent/prompt_builder.py:1303`, "NEVER drop entries") until the curator archives it, by default after
  30 days unused (`hermes_cli/config_defaults.py:1490`). In practice the index tracks the skills used in
  the last month; 400 is a stress point, not a typical profile.
- Where the prefix is cached this is mostly cache-read cost. Where it is not (a cold cache, a local server
  without prefix reuse) it is prefill on every call.
- A learned skill pays for itself only if the tokens and turns it saves in the sessions that use it exceed
  about 20 tokens times every API call of every session that does not. That is the break-even line any
  efficacy result ([#96704](https://github.com/NousResearch/hermes-agent/issues/96704) Arms B and C) has
  to clear.

What it does not show: whether a larger library makes the agent choose worse. That needs task outcomes.

### How I know the numbers are right

- Two full runs in fresh sandboxes must agree exactly; the script exits non-zero otherwise.
- Every run checks three invariants: every create succeeded, the index grew by exactly one row per
  learned skill, and the usage sidecar records every learned skill as agent-created.
- Those invariants were shown to fail. With `skills.write_approval: true`, every create "succeeds" while
  only staging the write, and the row invariant fails. With an `environments: [s6]` skill in the corpus,
  on a host without s6, the row invariant fails too; that is why the corpus excludes gated skills.
- The rendered blocks are committed beside the numbers, so any tokenizer can be applied to the same text.
- Nothing is written outside the sandbox: the worker's `HOME` stays empty and the real Hermes home is
  never read or written.
- `python scripts/check` passes on these files.

### Conditions and limits

- Linux x86_64, Python 3.14.9, a bare install: 20 tools in 11 toolsets, no provider keys or service
  credentials. Service-gated tools follow what a host has configured, and conditional skills follow
  them, so the number of listed bundled skills can differ by a few rows elsewhere. Here 39 of the 44 are
  listed; the other five are macOS-only or need the kanban toolset. The per-skill slope does not depend
  on this.
- The default posture only. Focus mode (`agent/coding_context.py`) demotes categories to a names-only
  line, which costs less; it is not measured.
- Learned skills are represented by the 106 shipped optional skills that carry no platform,
  environment, app or toolset gate (real names, real descriptions within the 60-character budget),
  cycled with a `-vN` suffix past 106. Names the agent writes may be longer or shorter; a row costs its
  name plus a description of at most 60 characters.
- Measured on the cold path. On current `main` a build served from the on-disk snapshot can list an
  `environments:`-gated skill that the cold scan hides; three open PRs fix this
  ([#50316](https://github.com/NousResearch/hermes-agent/pull/50316),
  [#50335](https://github.com/NousResearch/hermes-agent/pull/50335),
  [#128038](https://github.com/NousResearch/hermes-agent/pull/128038)). The learned corpus holds no gated
  skill, so the numbers above are unaffected.
- The background review's own tokens (the fork that writes these skills) are not measured here.

## `live.py`: a pilot of what carry-over buys

The pilot answers one question per model: when a later session inherits what earlier sessions learned, is it more
often right, or cheaper? It runs the default product end to end and tests no hypothesis: its numbers (cost per
episode, the spread between reps, whether the review fires at all) size the full study, whose hypotheses and
thresholds are fixed in this README before its first scored run.

### Design

- **Unit.** A cell is (model, arm, rep): the family's five episodes in order, each a new session in a fresh
  interpreter (`episode.py`), a real `AIAgent` with the default CLI tool surface, memory and background review on.
- **Arms.** Both run the identical default product, review fork included, so inside an episode they are the same.
  `carry` keeps the cell's `HERMES_HOME` from episode to episode, as a real user's home does; `fresh` resets it to
  a just-installed home before every episode. Carry minus fresh on episodes 2 to 5 is the effect of carry-over
  alone, and episode 1, identical in both arms, is a built-in parity check.
- **Family.** `family_conventions.py`: a small invoicing project with three rules written only in its
  CONTRIBUTING.md (tests through `./scripts/check`, a commit format its hook enforces, a changelog line nothing
  enforces) and five realistic changes. The project is rebuilt byte-identical before every episode. Grading needs
  no model: hidden tests, the project's suite, the commit and the changelog line.
- **When learning can happen at all.** The skill review fires only after a turn with at least 10 tool iterations
  (`agent/turn_finalizer.py:863-866`) and the memory review every 10 user turns of a session
  (`agent/turn_context.py:765-773`). Every record says whether the review fired and what it wrote, so a null
  result can be told apart from a loop that never ran.
- **Models.** `z-ai/glm-5.3`, the model Hermes lands on when the user picks none (`"default": true` in
  `website/static/api/model-catalog.json`), and `anthropic/claude-sonnet-5.5`, a frontier model from another
  family. Each is pinned to its maker's own OpenRouter endpoint (`preflight.ROUTES`): OpenRouter serves glm-5.3
  from 48 endpoints between fp4 and fp8, and an unpinned run would mix quantisations nobody could reproduce.
- **Audit.** Every API call of the main agent and the review fork is read from Hermes' own log lines
  (`audit.py`), with the upstream provider OpenRouter reports; one call off the pinned route makes the episode an
  infrastructure failure, never scored. Spend comes from OpenRouter's per-key counter, which counts every charged
  call whatever made it.

### Running it

```bash
python3 evals/learning_loop/preflight.py --key-file ~/.config/openrouter/hermes-eval.key
scripts/run-in-hermes-env python evals/learning_loop/live.py --key-file ~/.config/openrouter/hermes-eval.key \
    --out evals/learning_loop/runs/pilot --budget-usd 40
python3 evals/learning_loop/report.py evals/learning_loop/runs/pilot
```

The default pilot is 2 models x 2 arms x 3 reps x 5 episodes, run one episode at a time. Once the budget is
spent it starts no further episode (the key's own credit limit is the hard cap), and it resumes where it stopped:
each episode's starting home is snapshotted and restored if the episode is retried. At the end it searches every output file for the key and
fails loudly if it finds it.

### Safety

The agent runs real shell commands. In a programmatic session there is no approval surface, and commands the gate
flags are approved unless an unattended context applies (`tools/approval.py:1211`), so the boundary is the
operating system, as SECURITY.md section 2.2 says. A live run therefore refuses to start as root; run it as an
unprivileged account that holds nothing you care about, or in a VM. Each session's `HOME` is redirected into the
cell with the `core_tool_deferral` sandbox, so `~` never reaches the operator's real home.

### How I know it works

- `tests/evals/test_learning_loop_smoke.py` runs two cells end to end against a scripted OpenAI-compatible server on
  127.0.0.1: real `AIAgent` sessions, real tools, the real review fork. Episode 1 is solved through Hermes' terminal
  tool in 12 tool turns and graded full marks; the review fires and writes a skill; every chat request, the fork's
  included, carries the provider pin on the wire; in episode 2 the learned skill is in the system prompt for
  `carry` and absent for `fresh`; the key reaches no output file.
- The smoke test was shown to fail when the fresh arm stopped resetting, when the pin was dropped, and when the
  worker stopped waiting for the review fork (the fork's skill was then lost at exit).
- `tests/evals/test_learning_loop_family.py` holds the grader to known answers on all five episodes, both ways.
  `tests/evals/test_learning_loop_preflight.py` and `tests/evals/test_learning_loop_report.py` cover the pre-flight's
  refusals and the report's arithmetic.

### What the pilot must show before the full study

1. Every scored episode stayed on its pinned route.
2. The review fires in episode 1 for both models in most reps; if it does not, the family does not exercise the
   loop and is redesigned before anything scales.
3. Episode 1 is at parity between the arms, within the spread between reps.
4. The measured cost per episode and the reps the report says the full study needs, which together set its cap.
