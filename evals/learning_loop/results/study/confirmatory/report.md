# Learning-loop pilot report

## Qwen3.8-27B-UD-Q4_K_M

Pass rate and score include workflow checks. Oracle pass rate measures the independent task checks, including protocol-invalid rows. Missing outcomes are unavailable, not failures.

| arm/episode | n | pass rate | score | oracle n | oracle pass rate | prompt tokens | review fired | cost $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| carry/currency_symbol | 19 | 0.7895 | 0.8947 | 19 | 1.0 | 210458.1053 | 1.0 | None |
| fresh/currency_symbol | 0 | None | None | 0 | None | None | None | None |
| carry/percent_of | 19 | 0.7368 | 0.8816 | 19 | 1.0 | 135133.3158 | 0.0526 | None |
| fresh/percent_of | 19 | 0.4211 | 0.7105 | 19 | 1.0 | 140544.4211 | 0.1579 | None |
| carry/skip_empty_lines | 19 | 0.7368 | 0.8816 | 19 | 1.0 | 153248.8947 | 0.2105 | None |
| fresh/skip_empty_lines | 19 | 0.3684 | 0.6974 | 19 | 1.0 | 150696.2632 | 0.2632 | None |
| carry/discounted_total | 19 | 0.0 | 0.7105 | 19 | 0.0 | 184822.0526 | 0.4737 | None |
| fresh/discounted_total | 19 | 0.0 | 0.5789 | 19 | 0.0 | 181228.4737 | 0.4211 | None |
| carry/parse_symbols | 19 | 0.7368 | 0.8816 | 19 | 1.0 | 162759.1053 | 0.3684 | None |
| fresh/parse_symbols | 19 | 0.3684 | 0.6842 | 19 | 1.0 | 157988.3158 | 0.3158 | None |

| arm/episode | review eligible | fired | wrote | carried skills | listed | loaded | adhered | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| carry/currency_symbol | 1.0 | 1.0 | 1.0 | 0 | None | None | None | None |
| fresh/currency_symbol | None | None | None | 0 | None | None | None | None |
| carry/percent_of | 0.0526 | 0.0526 | 0.0526 | 15 | 1.0 | 0.9333 | 0.7333 | 1.0 |
| fresh/percent_of | 0.1579 | 0.1579 | 0.0526 | 0 | None | None | None | None |
| carry/skip_empty_lines | 0.2105 | 0.2105 | 0.2105 | 15 | 1.0 | 0.8 | 0.7333 | 1.0 |
| fresh/skip_empty_lines | 0.2632 | 0.2632 | 0.2632 | 0 | None | None | None | None |
| carry/discounted_total | 0.4737 | 0.4737 | 0.4211 | 17 | 1.0 | 1.0 | 0.8824 | 0.0 |
| fresh/discounted_total | 0.4211 | 0.4211 | 0.4211 | 0 | None | None | None | None |
| carry/parse_symbols | 0.3684 | 0.3684 | 0.3158 | 18 | 1.0 | 0.9444 | 0.7778 | 1.0 |
| fresh/parse_symbols | 0.3158 | 0.3158 | 0.1579 | 0 | None | None | None | None |

- Episode 1, carry only, run until the review wrote something: tries per repetition {0: 1, 1: 1, 2: 5, 3: 3, 4: 2, 5: 2, 6: 1, 7: 1, 8: 3, 9: 1, 10: 3, 11: 5, 12: 1, 13: 3, 14: 2, 15: 1, 16: 3, 17: 3, 18: 5, 19: 1}, learned in repetitions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]
- Later episodes, carry minus fresh: oracle pass rate 0.0 (95% CI [0.0, 0.0]; 19 complete repetitions).
- Later episodes, carry minus fresh: workflow-inclusive score 0.1711 (95% CI [0.0888, 0.2533]), prompt tokens 1376.4737 (95% CI [-13714.8947, 17593.2237])
- Cost per episode: $None
- Reps needed per arm for the full study: {'score_10_points': 28, 'prompt_tokens_25_percent': 7}

## Infrastructure failures (not scored)

[]
