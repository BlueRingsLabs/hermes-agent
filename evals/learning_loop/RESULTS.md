# Results: does keeping what Hermes learned make later sessions better?

Written on 18 November 2026, after the analysis. The plan is [`PREREGISTRATION.md`](PREREGISTRATION.md), fixed on
6 November before any real model ran an episode and amended twice before any confirmatory data (Deviation 1 and
Amendment 1). The sections marked exploratory go beyond that plan.

## In short

On a small project whose conventions are written only in its CONTRIBUTING.md, Hermes with Qwen3.8 27B scored 0.17
higher in later sessions when it kept the home that a first session had taught it something in (95% interval 0.09
to 0.25, 19 repetitions). By the plan's decision rule, keeping what Hermes learned changed the score.

All of the gain came from following the project's process, the commit format and the changelog line. The task
itself went the same way in both arms in every repetition, but this family can't show a change there: three of its
tasks were solved in every session of both arms and one failed in every session. Keeping the learned home cost
nothing measurable in prompt tokens or wall time.

## What ran

| | |
|---|---|
| Hermes | `f98da72`, the commit Amendment 1 names |
| Model | `unsloth/Qwen3.8-27B-GGUF` at `4ca72078`, `Qwen3.8-27B-UD-Q4_K_M.gguf`, sha256 `322e194f…3482` |
| Engine | llama.cpp b11671 (`ba3c05f7`), built from source with CUDA 13.0 for sm_80, because Colab's driver (580.82, CUDA 13.0) can't run Hermes' CUDA 13.4 build. Its sha256 (`27335e6d…4670`) matches the first calibration's build |
| Server | the pre-registered preset (sha256 `fefc2320…96d0`), giving each session the full 98,304-token window |
| GPU | one NVIDIA A100-SXM4-40GB on Colab. The first calibration had the 80 GB card, with the same preset |
| When | 11 November 2026, 16:48 to 22:37 UTC, in one session |

The block's two manifests come from the same machine and server process. The second, written at 02:11 UTC, is from
running the notebook again after the block had finished. It found every episode done and added no rows.

The run itself had no problems. There were 199 rows and no infrastructure failure, and every scored session was
protocol-valid. Every row ran with the 98,304-token window, and the largest prompt any call sent was 46,934 tokens,
under the compression threshold of 73,728, so compression never ran.

## Episode 1: how often Hermes learned

Episode 1 (`currency_symbol`) ran in the `carry` cell until the background review wrote something, five tries at
most. It learned in 19 of the 20 repetitions: on 19 of its 47 tries (40%), and on the first try in 8 repetitions.
Repetition 18 learned nothing in five tries, so it isn't scored, as the plan says.

In 15 repetitions the review created a skill for the project and then refined it, with names such as
`invoicer-repo`, `repo-conventions` and `repo-contribution-workflow`. In the other 4 it edited a skill Hermes ships
(`test-driven-development` three times, `requesting-code-review` once).

## Primary outcome

The score is the share of the four checks an episode passes: the hidden tests, the project's suite, the commit rule
and the changelog line. Each repetition contributes its mean carry minus fresh over episodes 2 to 5, and the
interval is a bootstrap over repetitions (2,000 resamples, seed 20261105).

| | |
|---|---|
| `carry`, mean score in episodes 2 to 5 | 0.84 |
| `fresh`, mean score in episodes 2 to 5 | 0.67 |
| **carry minus fresh** | **+0.17, 95% interval 0.09 to 0.25** |

The interval excludes zero, so carry-over changed the score. The estimate is above the smallest effect of interest
(0.10), and the lower end of the interval is just under it.

Per repetition, in order: +0.25, -0.125, +0.375, +0.4375, 0, +0.25, +0.375, +0.25, +0.125, 0, 0, +0.375, -0.125,
+0.1875, +0.375, 0, +0.125, +0.375 and 0 (repetitions 0 to 17, then 19).

## Secondary outcomes

The same unit and interval as the primary.

| carry minus fresh | mean | 95% interval |
|---|---:|---|
| Oracle pass (hidden tests and suite both pass) | 0.00 | 0.00 to 0.00 |
| Sessions that pass all four checks | +0.26 | 0.08 to 0.43 |
| Main agent's prompt tokens per session | +1,376 | -13,715 to +17,593 |
| Tool calls per session | +1.5 | +0.2 to +2.9 |
| Wall time per session | -4 s | -22 s to +14 s |

## How it worked (descriptive)

A learned skill was carried into 15 of the 19 sessions in episodes 2 and 3, and into 17 and 18 in episodes 4 and 5,
as reviews in later sessions created more. Whenever one was carried in, it was listed in the session's skills
index. The agent loaded it with `skill_view` in 80% to 100% of those sessions, and followed it (the convention
checks held and the tests ran through `./scripts/check`) in 73% to 88%.

## Exploratory: which checks moved

Not part of the plan. Carry minus fresh per check, with the same unit and interval:

| check | carry minus fresh | 95% interval |
|---|---:|---|
| Commit rule | +0.33 | 0.13 to 0.51 |
| Changelog line | +0.36 | 0.20 to 0.51 |
| Hidden tests | 0.00 | 0.00 to 0.00 |
| Project's suite | 0.00 | 0.00 to 0.00 |

The pass rate of each check per task over the 19 repetitions, `carry` then `fresh`:

| task | hidden tests | suite | commit rule | changelog |
|---|---|---|---|---|
| `percent_of` | 1.00, 1.00 | 1.00, 1.00 | 0.74, 0.42 | 0.79, 0.42 |
| `skip_empty_lines` | 1.00, 1.00 | 1.00, 1.00 | 0.74, 0.37 | 0.79, 0.42 |
| `discounted_total` | 0.00, 0.00 | 1.00, 1.00 | 0.89, 0.63 | 0.95, 0.68 |
| `parse_symbols` | 1.00, 1.00 | 1.00, 1.00 | 0.74, 0.37 | 0.79, 0.37 |

## Limits

- **Correctness couldn't move.** Three tasks passed their hidden tests in every session of both arms, and
  `discounted_total` failed them in every one. The oracle had no room to change, so its zero says nothing about
  whether learning helps an agent get a task right. It does show that the score's gain is not correctness.
- **The question is conditional.** Since Deviation 1 the study asks what keeping a learned home does once a first
  session has taught Hermes something. In its default configuration, Hermes learned from 40% of first-session tries
  on this task.
- **One model, one project, single-turn sessions.** Qwen3.8 27B at Q4_K_M, five small tasks in one repository.
- **The agent and the harness share an account.** The harness code is read-only to the agent, but the results file
  isn't. Every command the agent ran is in its transcript, and the transcripts are published.
- **Hardware.** The confirmatory session got a 40 GB A100 and the first calibration an 80 GB one. Both ran the same
  preset and the same build for the same architecture.

## Calibration

The first calibration failed gate 1, because episode 1, then `percent_of`, never reached the review's trigger of 10
tool iterations. Deviation 1 fixed the cause, the second calibration passed all four gates, and Amendment 1 set 20
repetitions. Both calibrations' rows are published below and enter no analysis.

## Data and how to check it

- Rows (gzipped), reports and the confirmatory manifests are in [`results/study/`](results/study/), one folder per
  block.
  `python3 evals/learning_loop/report.py evals/learning_loop/results/study/confirmatory` rebuilds the report.
- Every transcript, the skills Hermes wrote and each block's run folder are in the
  [release archive](https://github.com/BlueRingsLabs/hermes-agent/releases/tag/learning-loop-study-2026-11).

## Next

The pre-registered replication, 3 repetitions on Kaggle's 2×T4, comes next and will be reported on its own. A family
whose tasks can go either way would let the same design ask whether learning helps an agent get things right, not
only do them the project's way.
