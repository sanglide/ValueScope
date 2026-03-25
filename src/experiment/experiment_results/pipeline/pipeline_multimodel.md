# Pipeline Evaluation: Multi-Model Comparison

## Dimension 1: Risk Detection Agreement (Human vs Pipeline)

| Model | PA (Bef.) | κ (Bef.) | PABAK (Bef.) | AC1 (Bef.) | PA (Aft.) | κ (Aft.) | PABAK (Aft.) | AC1 (Aft.) | Δκ |
|---|---|---|---|---|---|---|---|---|---|
| Claude-3.5-Sonnet | 0.758 | 0.000 | 0.516 | 0.693 | 0.758 | 0.000 | 0.516 | 0.693 | +0.000 |
| GPT-5.2 | 0.268 | 0.017 | -0.465 | -0.399 | 0.363 | 0.058 | -0.274 | -0.265 | +0.041 |
| Grok-4 | 0.650 | 0.102 | 0.299 | 0.427 | 0.656 | 0.009 | 0.312 | 0.474 | -0.092 |
| Gemini-2.5-Flash | 0.261 | 0.004 | -0.478 | -0.415 | 0.312 | 0.038 | -0.376 | -0.342 | +0.035 |
| DeepSeek-V3 | 0.236 | -0.013 | -0.529 | -0.448 | 0.484 | 0.150 | -0.032 | -0.029 | +0.163 |
| Qwen-Plus | 0.548 | 0.238 | 0.096 | 0.101 | 0.548 | 0.238 | 0.096 | 0.101 | +0.000 |

*PA = Percent Agreement, κ = Cohen's Kappa, PABAK = Prevalence-Adjusted Bias-Adjusted Kappa,*
*AC1 = Gwet's AC1, Bef. = Before Evidence Filter, Aft. = After Evidence Filter, Δκ = Improvement*

## Dimension 2: Value Identification Agreement (Human vs Pipeline)

| Model | Jaccard (Bef.) | F1 (Bef.) | Jaccard (Aft.) | F1 (Aft.) | ΔJaccard |
|---|---|---|---|---|---|
| Claude-3.5-Sonnet | 0.758 | 0.758 | 0.758 | 0.758 | +0.000 |
| GPT-5.2 | 0.073 | 0.098 | 0.195 | 0.215 | +0.122 |
| Grok-4 | 0.561 | 0.561 | 0.605 | 0.605 | +0.045 |
| Gemini-2.5-Flash | 0.038 | 0.046 | 0.088 | 0.094 | +0.050 |
| DeepSeek-V3 | 0.021 | 0.033 | 0.284 | 0.293 | +0.263 |
| Qwen-Plus | 0.323 | 0.328 | 0.321 | 0.324 | -0.002 |

## Pipeline Statistics per Model

| Model | Samples | Hypotheses | Confirm Rate | Avg Confidence |
|---|---|---|---|---|
| Claude-3.5-Sonnet | 157 | 0 | 0.0% | 0.000 |
| GPT-5.2 | 157 | 388 | 55.8% | 1.000 |
| Grok-4 | 157 | 77 | 64.4% | 1.000 |
| Gemini-2.5-Flash | 157 | 462 | 73.8% | 1.000 |
| DeepSeek-V3 | 157 | 435 | 49.4% | 1.000 |
| Qwen-Plus | 157 | 308 | 71.7% | 1.000 |