# Learning-loop pilot report

## Qwen3.8-27B-UD-Q4_K_M

Pass rate and score include workflow checks. Oracle pass rate measures the independent task checks, including protocol-invalid rows. Missing outcomes are unavailable, not failures.

| arm/episode | n | pass rate | score | oracle n | oracle pass rate | prompt tokens | review fired | cost $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| carry/currency_symbol | 2 | 1.0 | 1.0 | 2 | 1.0 | 202176.0 | 1.0 | None |
| fresh/currency_symbol | 0 | None | None | 0 | None | None | None | None |
| carry/percent_of | 2 | 1.0 | 1.0 | 2 | 1.0 | 189508.0 | 0.5 | None |
| fresh/percent_of | 2 | 0.5 | 0.75 | 2 | 1.0 | 180224.5 | 0.5 | None |
| carry/skip_empty_lines | 2 | 1.0 | 1.0 | 2 | 1.0 | 227956.5 | 0.5 | None |
| fresh/skip_empty_lines | 2 | 0.5 | 0.75 | 2 | 1.0 | 152109.0 | 0.5 | None |
| carry/discounted_total | 2 | 0.0 | 0.75 | 2 | 0.0 | 179724.5 | 0.5 | None |
| fresh/discounted_total | 2 | 0.0 | 0.75 | 2 | 0.0 | 183245.0 | 1.0 | None |
| carry/parse_symbols | 2 | 1.0 | 1.0 | 2 | 1.0 | 190206.0 | 0.5 | None |
| fresh/parse_symbols | 2 | 0.0 | 0.5 | 2 | 1.0 | 96362.0 | 0.0 | None |

| arm/episode | review eligible | fired | wrote | carried skills | listed | loaded | adhered | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| carry/currency_symbol | 1.0 | 1.0 | 1.0 | 0 | None | None | None | None |
| fresh/currency_symbol | None | None | None | 0 | None | None | None | None |
| carry/percent_of | 0.5 | 0.5 | 0.5 | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| fresh/percent_of | 0.5 | 0.5 | 0.0 | 0 | None | None | None | None |
| carry/skip_empty_lines | 0.5 | 0.5 | 0.5 | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| fresh/skip_empty_lines | 0.5 | 0.5 | 0.5 | 0 | None | None | None | None |
| carry/discounted_total | 0.5 | 0.5 | 0.5 | 2 | 1.0 | 1.0 | 1.0 | 0.0 |
| fresh/discounted_total | 1.0 | 1.0 | 1.0 | 0 | None | None | None | None |
| carry/parse_symbols | 0.5 | 0.5 | 0.5 | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| fresh/parse_symbols | 0.0 | 0.0 | 0.0 | 0 | None | None | None | None |

- Episode 1, carry only, run until the review wrote something: tries per repetition {0: 2, 1: 2}, learned in repetitions [0, 1]
- Later episodes, carry minus fresh: oracle pass rate 0.0 (95% CI [0.0, 0.0]; 2 complete repetitions).
- Later episodes, carry minus fresh: workflow-inclusive score 0.25 (95% CI [0.125, 0.375]), prompt tokens 43863.625 (95% CI [42047.75, 45679.5])
- Cost per episode: $None
- Reps needed per arm for the full study: {'score_10_points': 25, 'prompt_tokens_25_percent': 2}

## Infrastructure failures (not scored)

[]
