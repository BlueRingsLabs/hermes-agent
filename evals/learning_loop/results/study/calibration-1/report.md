# Learning-loop pilot report

## Qwen3.8-27B-UD-Q4_K_M

Pass rate and score include workflow checks. Oracle pass rate measures the independent task checks, including protocol-invalid rows. Missing outcomes are unavailable, not failures.

| arm/episode | n | pass rate | score | oracle n | oracle pass rate | prompt tokens | review fired | cost $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| carry/percent_of | 2 | 0.0 | 0.5 | 2 | 1.0 | 127532.5 | 0.0 | None |
| fresh/percent_of | 2 | 0.0 | 0.5 | 2 | 1.0 | 146922.0 | 0.0 | None |
| carry/skip_empty_lines | 2 | 0.5 | 0.75 | 2 | 1.0 | 152513.5 | 0.5 | None |
| fresh/skip_empty_lines | 2 | 0.5 | 0.75 | 2 | 1.0 | 187142.0 | 1.0 | None |
| carry/currency_symbol | 2 | 0.5 | 0.75 | 2 | 1.0 | 187024.5 | 0.5 | None |
| fresh/currency_symbol | 2 | 0.5 | 0.75 | 2 | 1.0 | 209726.0 | 1.0 | None |
| carry/discounted_total | 2 | 0.0 | 0.625 | 2 | 0.0 | 162382.0 | 0.0 | None |
| fresh/discounted_total | 2 | 0.0 | 0.5 | 2 | 0.0 | 180951.0 | 0.5 | None |
| carry/parse_symbols | 2 | 0.5 | 0.75 | 2 | 1.0 | 129254.0 | 0.0 | None |
| fresh/parse_symbols | 2 | 0.0 | 0.5 | 2 | 1.0 | 152006.0 | 0.0 | None |

| arm/episode | review eligible | fired | wrote | carried skills | listed | loaded | adhered | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| carry/percent_of | 0.0 | 0.0 | 0.0 | 0 | None | None | None | None |
| fresh/percent_of | 0.0 | 0.0 | 0.0 | 0 | None | None | None | None |
| carry/skip_empty_lines | 0.5 | 0.5 | 0.5 | 0 | None | None | None | None |
| fresh/skip_empty_lines | 1.0 | 1.0 | 0.5 | 0 | None | None | None | None |
| carry/currency_symbol | 0.5 | 0.5 | 0.5 | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| fresh/currency_symbol | 1.0 | 1.0 | 0.5 | 0 | None | None | None | None |
| carry/discounted_total | 0.0 | 0.0 | 0.0 | 1 | 1.0 | 1.0 | 1.0 | 0.0 |
| fresh/discounted_total | 0.5 | 0.5 | 0.5 | 0 | None | None | None | None |
| carry/parse_symbols | 0.0 | 0.0 | 0.0 | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| fresh/parse_symbols | 0.0 | 0.0 | 0.0 | 0 | None | None | None | None |

- Episode 1 parity (carry minus fresh score, per rep): [0.0, 0.0]
- Later episodes, carry minus fresh: oracle pass rate 0.0 (95% CI [0.0, 0.0]; 2 complete repetitions).
- Later episodes, carry minus fresh: workflow-inclusive score 0.0938 (95% CI [-0.1875, 0.375]), prompt tokens -24662.75 (95% CI [-59580.75, 10255.25])
- Cost per episode: $None
- Reps needed per arm for the full study: {'score_10_points': 125, 'prompt_tokens_25_percent': 10}

## Infrastructure failures (not scored)

[]
