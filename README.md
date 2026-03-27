# ValueScope

**A Multi-Agent System for Value-Oriented Analysis in Software Projects**

ValueScope is a multi-agent framework that analyzes code changes for potential value deviations in software projects. It employs a four-layer cross-layer reasoning approach grounded in Schwartz's theory of basic human values, tracing from normative values (L1) through human value themes (L2) and system value themes (L3) down to concrete code indicators (L4).


## Project Structure

```
ValueScope/
├── pyproject.toml                     # Project configuration
├── tables/                            # Value taxonomy definitions
│   ├── L2_Value_Themes.csv            #   L2 Human Value Themes (HV1-HV10)
│   └── L3_system_value_themes.csv     #   L3 System Value Themes (SV1-SV10)
├── src/
│   ├── valueguard/                    # Core multi-agent system
│   │   ├── core/                      #   Dispatcher, models, config, exceptions
│   │   ├── agents/                    #   Profiler, Hypothesis, Evidence agents
│   │   ├── skills/                    #   Pluggable skills (LLM, AST, vector, etc.)
│   │   ├── memory/                    #   Three-layer memory system
│   │   ├── output/                    #   Report generation (JSON, Markdown, Console)
│   │   └── adapters/                  #   CLI and GitHub Action adapters
│   ├── experiment/                    # Experiment evaluation framework
│   │   ├── config.yaml                #   Experiment configuration
│   │   ├── data/                      #   Input datasets and repository data
│   │   ├── experiment_logs/           #   LLM output caches and intermediate logs
│   │   ├── experiment_results/        #   Final experiment outputs
│   │   ├── profile_experiment/        #   Value Profile experiments (Exp 1-3)
│   │   ├── iaa_experiment.py          #   Inter-Annotator Agreement experiment
│   │   ├── pipeline_evaluator.py      #   End-to-end pipeline evaluation
│   │   └── ...                        #   Supporting modules
```

## Installation

### Note

This repository uses [Git LFS](https://git-lfs.com) to manage large data files. 
The data that support the evalution are openly available in 'src/experiment/data.zip'
Please install Git LFS first, then run:

```bash
git lfs install
git clone <repository-url>
```

### Requirements

- Python >= 3.10, < 3.14

### Setup

```bash
# Install base dependencies
pip install -e .

# Install experiment dependencies (scipy, matplotlib, scikit-learn)
pip install -e ".[experiment]"

# Or install experiment dependencies directly
pip install -r src/experiment/requirements.txt
```

### Environment Variables

Configure API keys for the LLM providers you intend to use:

```bash
export OPENAI_API_KEY="your-openai-api-key"             # For GPT and o-series models
export ANTHROPIC_API_KEY="your-anthropic-api-key"        # For Claude models
export DEEPSEEK_API_KEY="your-deepseek-api-key"          # For DeepSeek models
export GOOGLE_API_KEY="your-google-api-key"              # For Gemini models
export XAI_API_KEY="your-xai-api-key"                    # For Grok models
export DASHSCOPE_API_KEY="your-dashscope-api-key"        # For Qwen models
```

LLM model configurations are defined in `src/experiment/config.yaml` under the `llm_models` section.

## Experiments

All experiments are located under `src/experiment/`. Results are stored in `src/experiment/experiment_results/` and intermediate caches in `src/experiment/experiment_logs/`.

### Experiment 1: Inter-Annotator Agreement (IAA)

Evaluates the consistency of value risk identification across multiple LLM annotators and a human annotator. Computes agreement metrics on two dimensions:
- **Dim1 (Risk Detection)**: Binary classification -- Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha, Percent Agreement
- **Dim2 (Value Identification)**: Multi-label -- Pairwise Jaccard, Symmetric F1, Per-value Fleiss' Kappa

```bash
cd src/experiment
python iaa_experiment.py --config config.yaml
```

**Options:**
- `--max-samples N`: Limit the number of samples processed
- `--models model1 model2`: Run with specific LLM models only
- `--output-dir DIR`: Override output directory

**Output:** `src/experiment/experiment_results/iaa/`

### Experiment 2: Pipeline Evaluation

End-to-end evaluation of the Hypothesis Generator + Evidence Location pipeline. Compares the multi-agent pipeline against zero-shot LLM baselines across multiple models.

```bash
cd src/experiment
python pipeline_evaluator.py \
    --samples-files data/focus_android_samples.json data/signal_android_samples.json \
    --repo-paths data/repo_data/focus-android data/repo_data/Signal-Android \
    --all-models qwen-plus \
    --output-dir experiment_results/pipeline \
    --output-name pipeline_eval \
    --no-cache \
    --parallel-workers 4
```

**Output:** `src/experiment/experiment_results/pipeline/`

### Experiment 3: Value Profile Evaluation

Three sub-experiments evaluating the Value Profile component:

1. **Profile Characterization**: Case study demonstrating that ValueProfile distinguishes value signatures across different projects
2. **Cross-Model Consistency**: Evaluates agreement of profiles generated by 7 different LLMs (Kendall's W, Spearman rho, Cosine Similarity)
3. **Bayesian Profile Calibration**: Ablation study using profile-based Bayesian posterior weighting on hypothesis confidence scores

```bash
cd src
python -m experiment.profile_experiment.run_all --experiments all

# Run individual experiments
python -m experiment.profile_experiment.run_all --experiments 1    # Characterization
python -m experiment.profile_experiment.run_all --experiments 2    # Cross-Model
python -m experiment.profile_experiment.run_all --experiments 3    # Bayesian Calibration
```

**Options:**
- `--force-api`: Force fresh LLM API calls (bypass cache)
- `--output-dir DIR`: Override output directory
- `--tables-dir DIR`: Override value model tables directory

**Output:** `src/experiment/experiment_results/profile/`

## Value Model

ValueScope uses a three-layer value taxonomy:

### L2: Human Value Themes

| ID | Theme | Description |
|----|-------|-------------|
| HV1 | Conformity | Following rules, meeting expectations |
| HV2 | Pleasure | User enjoyment, satisfaction |
| HV3 | Dignity | Respect for users, ethical treatment |
| HV4 | Inclusiveness | Accessibility, supporting diverse users |
| HV5 | Sense of Belonging | Community, connection |
| HV6 | Freedom | User autonomy, choice |
| HV7 | Independence | Self-sufficiency, not locked-in |
| HV8 | Wealth | Economic value, efficiency |
| HV9 | Privacy | Data protection, user control over information |
| HV10 | Security | Safety, protection from harm |

### L3: System Value Themes

| ID | Theme | Description |
|----|-------|-------------|
| SV1 | Trust | Reliability of the system |
| SV2 | Correctness | Accuracy, bug-free operation |
| SV3 | Compatibility | Works with other systems |
| SV4 | Portability | Works across platforms |
| SV5 | Reliability | Consistent operation |
| SV6 | Efficiency | Performance, resource usage |
| SV7 | Energy Preservation | Green computing |
| SV8 | Usability | Ease of use |
| SV9 | Accessibility | Support for users with disabilities |
| SV10 | Longevity | Long-term maintainability |

## Configuration

Experiment configuration is centralized in `src/experiment/config.yaml`:

- `llm_models`: LLM provider configurations (API endpoints, keys, parameters)
- `datasets`: Input data sources for experiments
- `prompts`: LLM prompt templates for value risk identification
- `iaa_experiment`: IAA experiment settings (annotators, metrics, output paths)
- `pipeline_experiment`: Pipeline evaluation settings (agent configs, repos, sample extraction)
- `profile_experiment`: Profile experiment settings (sub-experiment configs, repos, cache)

## Motivation Example
<img width="360" height="382" alt="image" src="https://github.com/user-attachments/assets/d702d52c-7ddb-48e3-b6e0-f966e1b0e9d3" />

For the motivation example shown in above, a privacy-first application named Signal is presented, where anonymous sending in Signal conceals the sender's identity when enabled. However, when secure fallback fails, the system reverts to unsealed (plain-text) transmission, enhancing usability but exposing metadata that conflicts with Signal’s "privacy first" statement. Next, we will demonstrate how ValueScope discovers and localizes this issue.

As shown in the figure below, the Hypothesis Generator produces two possible hypotheses for the input. The Evidence Location Agent accepts Hypothesis 1, rejects Hypothesis 2, and provides a formalized output.

<img width="360" height="505" alt="image" src="https://github.com/user-attachments/assets/9099db3e-f7fc-4840-84db-79c87bf73eca" />

