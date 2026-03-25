# IAA Experiment Report


### Scenario: code

## Pairwise Agreement Matrix (Risk Detection)

Upper triangle: Cohen's κ | Lower triangle: Percent Agreement

| | Human | claude-sonnet-4-5 | deepseek-chat | gemini-2.5-flash | grok-4 | o4-mini | qwen-plus |
|---|---|---|---|---|---|---|---|
| Human | — | 0.243 | 0.042 | 0.000 | 0.124 | 0.083 | 0.000 |
| claude-sonnet-4-5 | 67.6% | — | 0.267 | 0.000 | 0.646 | 0.477 | 0.000 |
| deepseek-chat | 60.3% | 92.6% | — | 0.000 | -0.023 | -0.020 | 0.000 |
| gemini-2.5-flash | 58.8% | 91.2% | 98.5% | — | 0.000 | 0.000 | 1.000 |
| grok-4 | 63.2% | 95.6% | 94.1% | 95.6% | — | 0.793 | 0.000 |
| o4-mini | 61.8% | 94.1% | 95.6% | 97.1% | 98.5% | — | 0.000 |
| qwen-plus | 58.8% | 91.2% | 98.5% | 100.0% | 95.6% | 97.1% | — |

### Scenario: text

## Pairwise Agreement Matrix (Risk Detection)

Upper triangle: Cohen's κ | Lower triangle: Percent Agreement

| | Human | claude-sonnet-4-5 | deepseek-chat | gemini-2.5-flash | grok-4 | o4-mini | qwen-plus |
|---|---|---|---|---|---|---|---|
| Human | — | 0.031 | 0.019 | 0.005 | 0.020 | 0.020 | 0.037 |
| claude-sonnet-4-5 | 36.7% | — | 0.546 | 0.163 | 0.551 | 0.432 | 0.586 |
| deepseek-chat | 35.5% | 97.3% | — | 0.252 | 0.574 | 0.418 | 0.509 |
| gemini-2.5-flash | 33.9% | 96.4% | 97.9% | — | 0.366 | 0.261 | 0.163 |
| grok-4 | 35.5% | 97.4% | 98.2% | 98.5% | — | 0.543 | 0.485 |
| o4-mini | 35.6% | 96.6% | 97.4% | 98.0% | 98.1% | — | 0.386 |
| qwen-plus | 37.5% | 96.5% | 96.5% | 95.5% | 96.5% | 95.7% | — |

### Scenario: overall

## Pairwise Agreement Matrix (Risk Detection)

Upper triangle: Cohen's κ | Lower triangle: Percent Agreement

| | Human | claude-sonnet-4-5 | deepseek-chat | gemini-2.5-flash | grok-4 | o4-mini | qwen-plus |
|---|---|---|---|---|---|---|---|
| Human | — | 0.037 | 0.020 | 0.005 | 0.023 | 0.022 | 0.037 |
| claude-sonnet-4-5 | 38.5% | — | 0.519 | 0.144 | 0.563 | 0.437 | 0.549 |
| deepseek-chat | 36.9% | 97.0% | — | 0.245 | 0.528 | 0.393 | 0.503 |
| gemini-2.5-flash | 35.4% | 96.1% | 97.9% | — | 0.329 | 0.245 | 0.163 |
| grok-4 | 37.1% | 97.3% | 97.9% | 98.3% | — | 0.567 | 0.465 |
| o4-mini | 37.1% | 96.5% | 97.3% | 97.9% | 98.1% | — | 0.376 |
| qwen-plus | 38.7% | 96.2% | 96.7% | 95.8% | 96.5% | 95.8% | — |

## Overall IAA Statistics

| Dimension | Metric | code | text | overall |
|---|---| --- | --- | --- |
| Risk Detection | Fleiss κ | 0.2444 | 0.4380 | 0.4256 |
| Risk Detection | Krippendorff α | 0.2444 | 0.4380 | 0.4256 |
| Risk Detection | Avg Pairwise κ | 0.2093 | 0.4157 | 0.4017 |
| Risk Detection | Avg Pairwise PABAK | 0.9137 | 0.9421 | 0.9405 |
| Risk Detection | Avg Pairwise AC1 | 0.9535 | 0.9694 | 0.9685 |
| Value ID | Macro Fleiss κ | 0.4914 | 0.5223 | 0.5320 |
| Value ID | Avg Pairwise Jaccard | 0.5794 | 0.6021 | 0.6008 |
| Value ID | Avg Pairwise F1 | 0.6965 | 0.7230 | 0.7214 |

## Per-Value Fleiss' Kappa

| Value ID | Name | Fleiss κ | Interpretation |
|---|---|---|---|
| HV1 | Conformity | 0.3756 | Fair |
| HV2 | Pleasure | 0.3121 | Fair |
| HV3 | Dignity | 0.3230 | Fair |
| HV4 | Inclusiveness | 0.6970 | Substantial |
| HV5 | Sense of Belonging | 0.3705 | Fair |
| HV6 | Freedom | 0.5812 | Moderate |
| HV7 | Independence | 0.5487 | Moderate |
| HV8 | Wealth | 0.5872 | Moderate |
| HV9 | Privacy | 0.6788 | Substantial |
| HV10 | Security | 0.6789 | Substantial |
| SV1 | Trust | 0.3817 | Fair |
| SV2 | Correctness | 0.5701 | Moderate |
| SV3 | Compatibility | 0.6151 | Substantial |
| SV4 | Portability | 0.5691 | Moderate |
| SV5 | Reliability | 0.6980 | Substantial |
| SV6 | Efficiency | 0.4444 | Moderate |
| SV7 | Energy Preservation | 0.7267 | Substantial |
| SV8 | Usability | 0.5986 | Moderate |
| SV9 | Accessibility | 0.5650 | Moderate |
| SV10 | Longevity | 0.3190 | Fair |
| **Macro Avg** | | **0.5320** | **Moderate** |

## Human vs. LLM Agreement

| Metric | vs claude-sonnet-4-5 | vs deepseek-chat | vs gemini-2.5-flash | vs grok-4 | vs o4-mini | vs qwen-plus |
|---| --- | --- | --- | --- | --- | --- |
| Risk κ | 0.0370 | 0.0198 | 0.0046 | 0.0233 | 0.0225 | 0.0374 |
| Risk PABAK | -0.2292 | -0.2618 | -0.2927 | -0.2584 | -0.2584 | -0.2258 |
| Risk AC1 | -0.1226 | -0.1405 | -0.1552 | -0.1362 | -0.1374 | -0.1227 |
| Risk %Agree | 38.5% | 36.9% | 35.4% | 37.1% | 37.1% | 38.7% |
| Value Jaccard | 0.1419 | 0.1187 | 0.1036 | 0.1279 | 0.1399 | 0.1424 |
| Value F1 | 0.1901 | 0.1656 | 0.1508 | 0.1752 | 0.1866 | 0.1882 |