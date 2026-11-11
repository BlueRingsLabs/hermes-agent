# Pre-registration: does what Hermes learns make later sessions better?

I fixed this plan before any real model ran an episode of the study. It is pushed to a public branch before the
first such run, and GitHub records when it was pushed. Anything I change after that point is listed under
**Deviations** with the date and the reason, before any analysis that depends on it. The number of repetitions is
set after a calibration block, by the rule below, and written in as **Amendment 1** before the first counted
episode.

## Question

In Hermes with the local model its catalog recommends (Qwen3.8 27B), once a first session has taught Hermes
something, does keeping what it learned change how the next sessions on the same project go, compared with starting
each of them from a fresh install? (Worded this way since Deviation 1.)

## Design

- **Arms.** Both arms run the same default product, background review included. `carry` keeps the cell's
  `HERMES_HOME` from one episode to the next, as a real user's home does. `fresh` resets it to a just-installed
  home before every episode. Inside an episode the two arms are identical.
- **Unit.** A repetition is one `carry` cell and one `fresh` cell over the family's five episodes in this order:
  `currency_symbol`, `percent_of`, `skip_empty_lines`, `discounted_total`, `parse_symbols`
  (`family_conventions.py`). Every episode is a new single-turn session on a project rebuilt byte-identical.
- **Episode 1.** It runs once per repetition, in the `carry` cell, from a just-installed home, and again from one
  (five tries at most) until the background review writes something. The `fresh` cell starts at episode 2. Both
  arms then face episodes 2 to 5 on the same project and differ only in whether the home that learned is kept. A
  repetition whose episode 1 never learns is reported and not scored. Episodes 2 to 5 carry the effect.
- **What can carry over.** In a single-turn session only the skill review can fire (after 10 tool iterations).
  The memory review needs 10 user turns, so memory changes only when the agent writes it itself.

## Fixed configuration

| part | pinned to |
|---|---|
| Hermes | the commit recorded in each run's manifest. The confirmatory run uses the commit named in Amendment 1 |
| model | `unsloth/Qwen3.8-27B-GGUF` at revision `4ca720788d1e01f1bff70c033e0d0028fd02e502`, file `Qwen3.8-27B-UD-Q4_K_M.gguf`, 16,464,440,224 bytes, sha256 `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| engine | llama.cpp `b11671` (commit `ba3c05f71da85cd0439e84ba42b0478b24d5db16`): Hermes' pinned CUDA build from `pm/lock.json`, or the same commit built from source on a machine whose driver cannot run it. Which one ran is recorded |
| server | started the way Hermes' supervisor starts it (router mode, `--models-preset`), with the preset Hermes writes for this file on a 24 GiB card: `study_preset.ini`, sha256 `fefc23208ed9c2559caf85347d85e8fa1ba77a6b0b3e9fbfa4bdf11fe4b696d0` |
| inference | a 98,304-token window, q8_0 KV cache, flash attention, MTP speculative decoding at depth 2, the file's own sampling defaults (temp 1, top-p 0.95, top-k 20) with min-p 0.0 and reasoning effort medium. The same preset on every GPU |
| agent | `AIAgent` with the default CLI toolset (`hermes-cli`), memory and background review on, at most 40 iterations, a 3,600 s episode timeout, `model.context_length: 98304` |
| isolation | the agent runs as its own unprivileged account. The harness and grader are read-only to it, and the run refuses to start if that account can read the checkpoint storage |

`gpu_study.py` regenerates the preset from the downloaded file with Hermes' own code and refuses to start the
server unless it matches `study_preset.ini` byte for byte.

## Outcomes

- **Primary.** The episode score: the share of the family's four checks that pass (hidden tests, the project's
  suite, the commit rule, the changelog line).
- **Secondary.** The oracle pass: the hidden tests and the project's suite both pass, which a learned convention
  cannot move by itself. Episodes that pass all four checks, the main agent's prompt tokens, tool calls and wall
  time.
- **How it worked (descriptive).** Per row: whether the review was eligible, fired and wrote, and for every skill
  carried in, whether it was listed in the session's index, loaded, followed (the convention checks and tests run
  through `./scripts/check`) and whether the independent oracle passed. Per repetition, how many tries episode 1
  needed.

## Analysis

- For each repetition, the mean of carry minus fresh over episodes 2 to 5. The repetition is the unit, because a
  cell's episodes share one evolving home. A repetition counts only when all four later episodes were scored in
  both arms.
- The estimate is the mean of those per-repetition differences, with a 95% bootstrap interval over repetitions
  (2,000 resamples, seed 20261105). This is `report.py` as committed with this plan.
- **Decision rule.** I say carry-over changed the score only if that interval excludes zero (two-sided). The
  smallest effect of interest is 0.10, one check in ten.
- Episode 1 runs in the `carry` cell only, so it has no difference to report. Secondary outcomes get the same
  treatment as the primary, labelled as secondary. Anything else is labelled exploratory.

## Sample size

- A calibration block of 2 repetitions runs first on the same setup. It is not part of the confirmatory analysis.
- The number of confirmatory repetitions N is `report.py`'s `reps_needed` for a 0.10 difference at 80% power,
  two-sided 5% (normal approximation), from the spread of the calibration's per-repetition differences. N is at
  least 8 and at most 20, which is what the compute budget allows. If the rule asks for more than 20, I run 20 and
  report the smallest effect that 20 can detect.

## Gates after calibration

The confirmatory run starts only if all of these hold. If one fails, I fix the cause, record it under Deviations
and calibrate again.

1. In every calibration repetition, episode 1 learned something within its five tries. Otherwise the design cannot
   deliver its treatment.
2. At most one infrastructure failure per block that its automatic rerun did not recover (`rerun_scored` in the
   report).
3. Every row ran with a 98,304-token window, and no call's prompt reached the compression threshold.
4. The preset, model and engine hashes in every manifest match this document.

## Exclusions

- A row with a non-zero exit code is an infrastructure failure: no valid record came out, because the server or
  the harness failed, or the episode ran past its 3,600 s limit, whether the server hung or the model never stopped.
  It is not scored. Its cell stops there, and at the end of the block the runner runs that episode once more from
  the home it started with, so no later episode builds on a failed one. If it fails again, its cell ends there and
  that repetition does not count. If the server stops answering, the block stops, and it resumes once the server
  is back.
- A row where the model misbehaved (malformed tool calls, the iteration limit) is scored as it is.
- Limitation: the agent and the harness share an OS account. The harness code is read-only to it, but the results
  file is not. Every command the agent ran is in its transcript, and the transcripts are published with the rows.

## Replication

After the confirmatory run, 3 repetitions on Kaggle's free 2×T4 with the same notebook and preset, the model split
across both cards. Reported on their own, then pooled with the main run as a sensitivity analysis.

## What gets published

Everything, whatever the result: every row, every transcript, the manifests with all hashes and the GPU and driver,
the report and this document with its amendments.

## Deviations

**Deviation 1, 11 November 2026, after the first calibration and before any confirmatory run.** The first
calibration (2 repetitions, 6 and 7 November, Hermes at `fcf98f0`) passed gates 2 to 4. There was no
infrastructure failure, every row ran with a 98,304-token window, the largest prompt any call sent was 35,659
tokens against a compression threshold of 73,728, and every hash matched. It failed gate 1. Episode 1
(`percent_of`) took 7 to 9 tool iterations in all four cells, under the review's trigger of 10, so the review never
ran there. It wrote skills in 5 of the 16 later episodes, and the result split on that. Carry minus fresh was
+0.375 in the repetition whose `carry` cell learned in episode 2, and -0.19 in the one whose `carry` cell never
learned. The design was measuring whether learning happened as much as what it did.

Two changes fix the cause without touching the product, the episodes or their grading:

- Episode 1 is now `currency_symbol`, the task that took the most tool iterations on average in calibration (9 to
  12). The other four keep their order.
- Episode 1 runs only in the `carry` cell, from a fresh install, until the background review writes something,
  five tries at most. The `fresh` cell starts at episode 2. The question is therefore what keeping a learned home
  does once Hermes has learned something, and how often a first session teaches it something is reported on its
  own. Gate 1 changes to match.

A second calibration of 2 repetitions runs under these rules before Amendment 1. The first calibration's rows are
kept and published, and they enter no analysis.

## Amendment 1 (after calibration)

Written 11 November 2026, after the second calibration and before any confirmatory episode.

- **Second calibration** (2 repetitions, 11 November, Hermes at `f98da72`, under Deviation 1's rules). All four
  gates passed. Episode 1 learned in both repetitions, after 2 tries each. There was no infrastructure failure.
  Every row ran with a 98,304-token window, and the largest prompt any call sent was 36,776 tokens against a
  compression threshold of 73,728. Every row names the pinned model file and engine, and the run cannot start
  unless the preset matches.
- **Repetitions.** The per-repetition differences were +0.125 and +0.375, so `reps_needed` for a 0.10 difference is
  25. That is over the maximum, so the confirmatory run has **N = 20** repetitions, which detect a difference of
  about 0.11 at 80% power, two-sided 5%.
- **Code.** The confirmatory run uses commit `f98da72bb606547a3ae9847bc099de5c2226db8f`, as `STUDY =
  "confirmatory"`, repetitions 0 to 19. Only its rows enter the confirmatory analysis.
- **For the record.** The second calibration's estimates were a score difference of +0.25, an oracle difference of
  0.0 and 43,864 more prompt tokens per later episode for `carry`. They are not results, they enter no analysis,
  and they are published with the calibration rows.
