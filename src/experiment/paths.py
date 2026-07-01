"""Unified output paths for all experiments.

All experiment outputs (results, logs, caches) live under a single root:
    <project_root>/experiment_outputs

This module provides the canonical subdirectories so that scripts do not
hard-code separate ``experiment_results/`` and ``experiment_logs/`` roots.
"""

from pathlib import Path

# Project root is three levels up: src/experiment/paths.py -> src/experiment -> src -> project_root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Single root for all experiment artifacts
OUTPUT_ROOT = PROJECT_ROOT / "experiment_outputs"

# Top-level categories
RESULTS_DIR = OUTPUT_ROOT / "results"
LOGS_DIR = OUTPUT_ROOT / "logs"
CACHE_DIR = OUTPUT_ROOT / "cache"

# Result subdirectories
MAIN_EXP_DIR = RESULTS_DIR / "main_exp"
IAA_DIR = RESULTS_DIR / "iaa"
PIPELINE_DIR = RESULTS_DIR / "pipeline"
PROFILE_DIR = RESULTS_DIR / "profile"

# Cache subdirectories
LLM_OUTPUTS_DIR = CACHE_DIR / "llm_outputs"
PROFILE_CACHE_DIR = CACHE_DIR / "profile_cache"
GT_ANNOTATION_CACHE_DIR = CACHE_DIR / "gt_annotation_cache"
PIPELINE_CACHE_DIR = LLM_OUTPUTS_DIR / "pipeline_cache"
TEXT_PIPELINE_CACHE_DIR = LLM_OUTPUTS_DIR / "text_pipeline_cache"


def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_all_dirs() -> None:
    """Ensure all canonical output directories exist."""
    for p in (
        RESULTS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        MAIN_EXP_DIR,
        IAA_DIR,
        PIPELINE_DIR,
        PROFILE_DIR,
        LLM_OUTPUTS_DIR,
        PROFILE_CACHE_DIR,
        GT_ANNOTATION_CACHE_DIR,
        PIPELINE_CACHE_DIR,
        TEXT_PIPELINE_CACHE_DIR,
    ):
        ensure_dir(p)
