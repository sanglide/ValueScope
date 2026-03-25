# Experiment 3: Profile as Bayesian Hypothesis Calibrator

**Model**: qwen-plus  
**Total Samples**: 1165 (68 code + 1097 text)  
**Optimal alpha**: 0 (by Value F1)  
**Threshold**: 0.5  
**Profile Match**: {'project_profile': 832, 'uniform_prior': 333}

## Method

Profile is used as a Bayesian prior to calibrate LLM prediction confidences post-hoc, without participating in hypothesis generation.

Formula: `weighted_conf[v] = raw_conf[v] * (profile_score[v] / mu)^alpha`

## Alpha Sweep Results

- alpha=0: Value F1=0.1978, Risk F1=0.5296, Jaccard=0.1425
- alpha=0.25: Value F1=0.1978, Risk F1=0.5296, Jaccard=0.1425
- alpha=0.5: Value F1=0.1962, Risk F1=0.5296, Jaccard=0.1418
- alpha=0.75: Value F1=0.1941, Risk F1=0.5296, Jaccard=0.1420
- alpha=1.0: Value F1=0.1921, Risk F1=0.5296, Jaccard=0.1410
- alpha=1.5: Value F1=0.1914, Risk F1=0.5296, Jaccard=0.1408
- alpha=2.0: Value F1=0.1911, Risk F1=0.5296, Jaccard=0.1405

## Comparison: Baseline vs Optimal

### Code Scenarios (N=68)

| Metric | alpha=0 | alpha=optimal | Delta |
|--------|---------|--------------|-------|
| Risk P | 0.5882 | 0.5882 | +0.0000 |
| Risk R | 1.0000 | 1.0000 | +0.0000 |
| Risk F1 | 0.7407 | 0.7407 | +0.0000 |
| Value P | 0.3048 | 0.3048 | +0.0000 |
| Value R | 0.7736 | 0.7736 | +0.0000 |
| Value F1 | 0.4373 | 0.4373 | +0.0000 |
| Jaccard | 0.2676 | 0.2676 | +0.0000 |

### Issue Text Scenarios (N=1097)

| Metric | alpha=0 | alpha=optimal | Delta |
|--------|---------|--------------|-------|
| Risk P | 0.3471 | 0.3471 | +0.0000 |
| Risk R | 0.9864 | 0.9864 | +0.0000 |
| Risk F1 | 0.5135 | 0.5135 | +0.0000 |
| Value P | 0.1010 | 0.1010 | +0.0000 |
| Value R | 0.7800 | 0.7800 | +0.0000 |
| Value F1 | 0.1788 | 0.1788 | +0.0000 |
| Jaccard | 0.1347 | 0.1347 | +0.0000 |

### Overall (N=1165)

| Metric | alpha=0 | alpha=optimal | Delta |
|--------|---------|--------------|-------|
| Risk P | 0.3618 | 0.3618 | +0.0000 |
| Risk R | 0.9877 | 0.9877 | +0.0000 |
| Risk F1 | 0.5296 | 0.5296 | +0.0000 |
| Value P | 0.1133 | 0.1133 | +0.0000 |
| Value R | 0.7790 | 0.7790 | +0.0000 |
| Value F1 | 0.1978 | 0.1978 | +0.0000 |
| Jaccard | 0.1425 | 0.1425 | +0.0000 |
