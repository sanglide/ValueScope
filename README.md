# ValueScope: Profile-Guided Value Risk Identification

This repository contains the implementation and evaluation code for **ValueScope**, a profile-guided, evidence-verified approach to identifying human-value risks in software engineering artifacts (code changes and issue discussions).

## Research Question

> To what extent does profile-guided, evidence-verified value risk identification (ValueScope) improve agreement with human annotations compared to zero-shot LLM baselines, and what is the contribution of each pipeline component?

- **RQ1 (Effectiveness)**: Does ValueScope agree with human annotations significantly better than LLM-only zero-shot classification?
- **RQ2 (Component Contribution)**: How much do the project-value profile and the evidence-verification step each contribute?
- **RQ3 (Generalization)**: Are the conclusions stable across different LLM backbones?

## Repository Structure

```
.
├── src/
│   ├── experiment/                  # Experiment scripts and benchmarks
│   │   ├── main_experiment.py       # Main experiment entry point
│   │   ├── config.yaml              # Unified experiment configuration
│   │   ├── benchmark_builder.py
│   │   ├── unified_pipeline.py
│   │   ├── pipeline_evaluator.py
│   │   ├── iaa_experiment.py
│   │   ├── run_experiment.py
│   │   ├── traditional_baselines.py
│   │   ├── report_generator.py
│   │   ├── iaa_report_generator.py
│   │   └── profile_experiment/      # Value-profile characterization experiments
│   └── valueguard/                  # Core ValueScope framework
├── tables/                          # Value model definitions (L2/L3 themes)
├── experiment_outputs/              # All experiment outputs (created at runtime)
├── .env                             # API keys
└── pyproject.toml
```

All runtime artifacts (results, logs, caches) are written under a single root: `experiment_outputs/`.

## Installation

Requires Python `>=3.10,<3.14`. We recommend [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
pip install uv
uv sync
```

Create a `.env` file in the project root with your API keys:

```bash
DASHSCOPE_API_KEY=...
DEEPSEEK_API_KEY=...
OPENAI_API_KEY=...
```

## Main Experiment

The main experiment compares ValueGuard against LLM-only and ablation baselines on a unified benchmark of code and text scenarios.

### Quick Start

List all configured ablation variants:

```bash
uv run python -m experiment.main_experiment --list-variants
```

Run the full main experiment with the primary model (`qwen-plus`). This invokes real LLM APIs, so ensure your API keys are configured in `.env`:

```bash
uv run python -m experiment.main_experiment
```

Limit the number of samples for a faster run:

```bash
uv run python -m experiment.main_experiment --max-samples 50
```

Run cross-model validation with additional models:

```bash
uv run python -m experiment.main_experiment --validation-models deepseek-chat grok-4
```

### Configuration

The main experiment uses `src/experiment/config.yaml` by default. Edit it to customize:

- `main_experiment.primary_model`: primary LLM for the full pipeline.
- `main_experiment.validation_models`: additional models for cross-model validation.
- `main_experiment.ablation.variants`: ablation configurations (e.g., w/o Profile, w/o Evidence, different alpha values).
- `main_experiment.profile.cache_dir`: profile cache directory.
- `main_experiment.llm_only.iaa_cache_dir`: IAA LLM-output cache used by the LLM-only baseline.
- `output.output_dir`: unified output root (default: `experiment_outputs`).

### What the Main Experiment Does

1. **Builds the benchmark** from a single unified code-scenario file and text issue discussions.
2. **Loads project value profiles** from the profile cache.
3. **Runs baselines**:
   - Human (ground truth)
   - LLM-only zero-shot (reuses IAA cache)
   - Traditional baselines (TF-IDF / BM25 / BERT zero-shot)
4. **Runs ValueGuard variants**:
   - Full pipeline (real profile + hypothesis + evidence)
   - w/o Profile (uniform prior)
   - w/o Evidence (real profile, no verification)
   - Additional ablations configured in `config.yaml`
5. **Computes metrics** for risk detection (Dim1) and value identification (Dim2).
6. **Performs statistical tests** (McNemar, Wilcoxon signed-rank, bootstrap CI) and reports component contributions.

### Expected Output

Results are written to `experiment_outputs/results/main_exp/`:

```
experiment_outputs/results/main_exp/
├── benchmark.json                 # Unified benchmark samples
├── main_results.json              # Per-method results
├── metrics_summary.md             # Human-readable metrics summary
├── ablation_summary.md            # Ablation contribution table
├── statistical_tests.json         # Statistical test results
├── variant_{name}.json            # Per-variant detailed results
└── figures/                       # Generated plots
```

Logs are written to `experiment_outputs/logs/main_experiment_*.log`.

### Interpreting the Results

The summary Markdown files report:

- **Dim1 (Risk Detection)**: Precision, Recall, F1, Cohen's κ, PABAK, Gwet's AC1.
- **Dim2 (Value Identification)**: Micro Precision/Recall/F1, Pairwise Jaccard, Symmetric F1.
- **Ablation Contribution**: ΔF1, Δκ, ΔJaccard, ΔSym-F1 when removing Profile or Evidence.
- **Cross-Model Validation**: consistency of ValueGuard improvements across different LLMs.

Higher κ and Jaccard indicate stronger agreement with human annotations. Positive ablation deltas indicate that the removed component was beneficial.

## Other Experiments

### IAA Experiment

Measures inter-annotator agreement between human annotations and LLM annotators.

```bash
uv run python -m experiment.iaa_experiment --config src/experiment/config.yaml
```

Output: `experiment_outputs/results/iaa/`

### Profile Experiment

Characterizes project value profiles and evaluates cross-model consistency and downstream impact.

```bash
uv run python -m experiment.profile_experiment.run_all --experiments all
```

Output: `experiment_outputs/results/profile/`

### Pipeline Experiment

Standalone pipeline evaluation on commit samples. It can read the unified `code_scenarios.json` directly; only samples that contain `diff_hunks_data` are evaluated.

```bash
uv run python -m experiment.pipeline_evaluator \
    --samples-file src/experiment/data/code_scenarios.json \
    --repo-dir src/experiment/data/repos \
    --mock --max-samples 10
```

Output: `experiment_outputs/results/pipeline/`

## Unified Output Layout

All experiments write to `experiment_outputs/`:

```
experiment_outputs/
├── results/
│   ├── main_exp/                  # Main experiment results
│   ├── iaa/                       # IAA experiment results
│   ├── pipeline/                  # Pipeline experiment results
│   └── profile/                   # Profile experiment results
├── logs/                          # Experiment logs
└── cache/
    ├── llm_outputs/               # LLM response caches
    │   ├── pipeline_cache/
    │   ├── text_pipeline_cache/
    │   └── gt_annotation_cache/
    └── profile_cache/             # Value profile caches
```

## Citation

If you use this code in your research, please cite the associated paper.

## Support

For questions or issues, please open an issue in the repository.
