## Skills index vs learned skills

| learned skills | index rows | chars | tokens_hermes_estimate | share of a 32k window |
|---|---|---|---|---|
| 0 | 39 | 4,311 | 1,084 | 3.3% |
| 10 | 49 | 5,088 | 1,278 | 3.9% |
| 25 | 64 | 6,223 | 1,566 | 4.8% |
| 50 | 89 | 8,152 | 2,048 | 6.2% |
| 100 | 139 | 11,959 | 3,000 | 9.2% |
| 200 | 239 | 19,860 | 4,980 | 15.2% |
| 400 | 439 | 35,661 | 8,940 | 27.3% |

Marginal cost per learned skill (least squares): tokens_hermes_estimate: 19.639

## Memory blocks at capacity

Accepted entries before the first refusal: {'memory': 18, 'user': 11}; limits {'memory': 2200, 'user': 1375} chars.
MEMORY.md block: {'chars': 2322, 'tokens_hermes_estimate': 632}
USER.md block: {'chars': 1470, 'tokens_hermes_estimate': 417}

## Conditions

```json
{
  "commit": "383f28428cbfc5b0dc29119275430bd40a9d8c75",
  "dirty": false,
  "python": "3.14.9",
  "platform": "Linux-6.18.44-fc-v236-x86_64-with-glibc2.39",
  "toolset": "hermes-cli",
  "posture": "default (no focus-mode category demotion)",
  "tokenizer": "hermes estimate only",
  "learned_corpus": "optional-skills/, ungated only (106 skills, cycled -vN)"
}
```
