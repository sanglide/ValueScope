# Pipeline Evaluation: Multi-Model Comparison

## Dimension 1: Risk Detection Agreement (Human vs Pipeline)

| Model | PA (Bef.) | κ (Bef.) | PABAK (Bef.) | AC1 (Bef.) | PA (Aft.) | κ (Aft.) | PABAK (Aft.) | AC1 (Aft.) | Δκ |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 0.200 | 0.000 | -0.600 | -0.538 | 0.800 | 0.545 | 0.600 | 0.655 | +0.545 |

*PA = Percent Agreement, κ = Cohen's Kappa, PABAK = Prevalence-Adjusted Bias-Adjusted Kappa,*
*AC1 = Gwet's AC1, Bef. = Before Evidence Filter, Aft. = After Evidence Filter, Δκ = Improvement*

## Dimension 2: Value Identification Agreement (Human vs Pipeline)

| Model | Jaccard (Bef.) | F1 (Bef.) | Jaccard (Aft.) | F1 (Aft.) | ΔJaccard |
|---|---|---|---|---|---|
| DeepSeek-V3 | 0.000 | 0.000 | 0.600 | 0.600 | +0.600 |

## Pipeline Statistics per Model

| Model | Samples | Hypotheses | Confirm Rate | Avg Confidence |
|---|---|---|---|---|
| DeepSeek-V3 | 5 | 10 | 15.8% | 1.000 |