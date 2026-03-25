# Pipeline Evaluation: Multi-Model Comparison

## Dimension 1: Risk Detection Agreement (Human vs Pipeline)

| Model | PA (Bef.) | κ (Bef.) | PABAK (Bef.) | AC1 (Bef.) | PA (Aft.) | κ (Aft.) | PABAK (Aft.) | AC1 (Aft.) | Δκ |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-Plus | 0.592 | 0.277 | 0.185 | 0.199 | 0.592 | 0.277 | 0.185 | 0.199 | +0.000 |

*PA = Percent Agreement, κ = Cohen's Kappa, PABAK = Prevalence-Adjusted Bias-Adjusted Kappa,*
*AC1 = Gwet's AC1, Bef. = Before Evidence Filter, Aft. = After Evidence Filter, Δκ = Improvement*

## Dimension 2: Value Identification Agreement (Human vs Pipeline)

| Model | Jaccard (Bef.) | F1 (Bef.) | Jaccard (Aft.) | F1 (Aft.) | ΔJaccard |
|---|---|---|---|---|---|
| Qwen-Plus | 0.371 | 0.376 | 0.373 | 0.378 | +0.002 |

## Pipeline Statistics per Model

| Model | Samples | Hypotheses | Confirm Rate | Avg Confidence |
|---|---|---|---|---|
| Qwen-Plus | 157 | 553 | 73.1% | 0.619 |