#!/usr/bin/env python
"""
IAA (Inter-Annotator Agreement) Experiment
Treats human annotations and LLM outputs as peer annotators,
computes agreement metrics on Dim1 (risk detection) and Dim2 (value identification).

Usage:
    python iaa_experiment.py --config config.yaml
    python iaa_experiment.py --config config.yaml --max-samples 20
    python iaa_experiment.py --config config.yaml --force-api
    python iaa_experiment.py --config config.yaml --batch-size 5
    python iaa_experiment.py --config config.yaml --models deepseek-chat qwen-plus
"""

import argparse
import csv
import json
import logging
import itertools
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import yaml

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

try:
    from .data_loader import (
        ScenarioDataLoader,
        ValueScenarioSample,
        IssuesDatasetLoader,
        ValueModelLoader,
        create_sample_dataset,
    )
    from .iaa_data_structures import (
        ALL_VALUE_IDS,
        VALUE_NAMES,
        AnnotatorAnnotation,
        AnnotationMatrix,
        PairwiseAgreementResult,
        MultiAnnotatorAgreementResult,
        IAAExperimentResults,
    )
    from . import iaa_metrics
    from .iaa_report_generator import IAAReportGenerator
    from .llm_client import LLMClientFactory, BaseLLMClient
    from . import paths as exp_paths
except ImportError:
    from data_loader import (
        ScenarioDataLoader,
        ValueScenarioSample,
        IssuesDatasetLoader,
        ValueModelLoader,
        create_sample_dataset,
    )
    from iaa_data_structures import (
        ALL_VALUE_IDS,
        VALUE_NAMES,
        AnnotatorAnnotation,
        AnnotationMatrix,
        PairwiseAgreementResult,
        MultiAnnotatorAgreementResult,
        IAAExperimentResults,
    )
    import iaa_metrics
    from iaa_report_generator import IAAReportGenerator
    from llm_client import LLMClientFactory, BaseLLMClient
    import paths as exp_paths


class IAAExperiment:
    """IAA Experiment Main Class"""

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 5  # seconds

    def __init__(
        self,
        config_path: str = "config.yaml",
        max_samples: int = 0,
        force_api: bool = False,
        batch_size: int = 0,
        models: Optional[list[str]] = None,
        max_workers: int = 4,
    ):
        """
        Args:
            config_path: Config file path
            max_samples: Max samples per scenario type, 0 = no limit
            force_api: Force re-call API (ignore cache)
            batch_size: Process N samples per batch then pause, 0 = no batching
            models: Only run these specific models (None = all enabled)
            max_workers: Number of parallel threads for API calls (per model)
        """
        self.config = self._load_config(config_path)
        self.max_samples = max_samples
        self.force_api = force_api
        self.batch_size = batch_size
        self.model_filter = models
        self.max_workers = max_workers
        self.datasets: dict[str, list[ValueScenarioSample]] = {}
        self.sample_scenario_map: dict[str, str] = {}
        self.value_model_loader: Optional[ValueModelLoader] = None
        self.llm_clients: dict[str, BaseLLMClient] = {}

        # Logging
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / self.config.get("output", {}).get(
            "logs_dir",
            str(exp_paths.LOGS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"iaa_experiment_{timestamp}.log"

        # Reset handlers to avoid duplicates from previous runs
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(),
            ]
        )
        self.logger = logging.getLogger(__name__)

        # LLM output cache directory
        iaa_config = self.config.get("iaa_experiment", {})
        self.llm_outputs_dir = project_root / iaa_config.get(
            "llm_outputs_dir",
            str(exp_paths.LLM_OUTPUTS_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        )
        self.llm_outputs_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(__file__).parent / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Load datasets, value model, and LLM clients."""
        print("=" * 60)
        print("IAA Experiment — Initialization")
        print("=" * 60)
        self._load_datasets_from_config()

        if self.max_samples > 0:
            self._apply_sampling()

        for scenario_type, samples in self.datasets.items():
            for s in samples:
                self.sample_scenario_map[s.sample_id] = scenario_type

        # Load value model
        tables_dir = Path(__file__).parent.parent.parent / "tables"
        self.value_model_loader = ValueModelLoader(str(tables_dir))
        self.value_model_loader.load()
        self.logger.info(
            f"Loaded {len(self.value_model_loader.l2_values)} L2 values, "
            f"{len(self.value_model_loader.l3_values)} L3 values"
        )

        # Create LLM clients
        llm_configs = self.config.get("llm_models", {})
        all_clients = LLMClientFactory.create_all_enabled(llm_configs)

        # Filter models if specified
        if self.model_filter:
            self.llm_clients = {
                k: v for k, v in all_clients.items() if k in self.model_filter
            }
            skipped = set(self.model_filter) - set(self.llm_clients.keys())
            if skipped:
                self.logger.warning(f"Models not found/enabled: {skipped}")
        else:
            self.llm_clients = all_clients

        self.logger.info(f"Active LLM clients ({len(self.llm_clients)}): {list(self.llm_clients.keys())}")

        # Print sample counts
        total = sum(len(s) for s in self.datasets.values())
        print(f"\nTotal samples: {total}")
        for st, samples in self.datasets.items():
            print(f"  {st}: {len(samples)} samples")
        print(f"Active models: {list(self.llm_clients.keys())}")
        print()

    def _apply_sampling(self) -> None:
        for scenario_type in list(self.datasets.keys()):
            samples = self.datasets[scenario_type]
            if len(samples) > self.max_samples:
                global_seed = 42
                for ds_conf in self.config.get("datasets", {}).values():
                    if "seed" in ds_conf:
                        global_seed = ds_conf["seed"]
                        break
                rng = random.Random(global_seed)
                sampled = rng.sample(samples, self.max_samples)
                self.datasets[scenario_type] = sampled
                print(f"[{scenario_type}] Sampled {self.max_samples}/{len(samples)}")
            else:
                print(f"[{scenario_type}] Using all {len(samples)} samples")

    def _load_datasets_from_config(self) -> None:
        datasets_config = self.config.get("datasets", {})
        base_dir = Path(__file__).parent

        for ds_key, ds_conf in datasets_config.items():
            if not ds_conf.get("enabled", True):
                continue

            ds_type = ds_conf.get("type", "json")
            scenario_type = ds_conf.get("scenario_type", "code")
            ds_path = base_dir / ds_conf.get("path", "")

            if ds_type == "json":
                loader = ScenarioDataLoader()
                loader.load_from_json(str(ds_path))
                samples = loader.get_samples()
                for s in samples:
                    s.metadata["scenario_type"] = scenario_type
            elif ds_type == "json_dir":
                loader = ScenarioDataLoader()
                loader.load_from_directory(str(ds_path))
                samples = loader.get_samples()
                for s in samples:
                    s.metadata["scenario_type"] = scenario_type
            elif ds_type == "issues_dataset":
                issues_loader = IssuesDatasetLoader(str(ds_path))
                samples = issues_loader.load(
                    sample_per_project=ds_conf.get("sample_per_project"),
                    max_text_length=ds_conf.get("max_text_length", 8000),
                    seed=ds_conf.get("seed", 42),
                )
            else:
                print(f"Warning: Unknown dataset type '{ds_type}', skipping {ds_key}")
                continue

            if scenario_type not in self.datasets:
                self.datasets[scenario_type] = []
            self.datasets[scenario_type].extend(samples)
            print(f"[{ds_key}] Loaded {len(samples)} {scenario_type} samples")

        if not self.datasets:
            print("No dataset config found, using built-in sample data")
            self.datasets["code"] = create_sample_dataset()

    # ------------------------------------------------------------------
    # Collect annotations
    # ------------------------------------------------------------------
    def collect_all_annotations(self) -> dict:
        """Collect all annotator annotations.

        For each LLM annotator: load from cache if available and force_api=False,
        otherwise call real API with retry and write result to cache.
        Within each model, samples are processed in parallel using ThreadPoolExecutor.

        Returns:
            {sample_id: {annotator_id: AnnotatorAnnotation}}
        """
        iaa_config = self.config.get("iaa_experiment", {})
        annotators_config = iaa_config.get("annotators", {})
        llm_annotators = annotators_config.get("llm_annotators", [])
        human_id = annotators_config.get("human", {}).get("annotator_id", "Human")

        raw: dict[str, dict[str, AnnotatorAnnotation]] = {}
        raw_lock = Lock()

        # Collect all samples
        all_samples: dict[str, ValueScenarioSample] = {}
        for samples in self.datasets.values():
            for s in samples:
                all_samples[s.sample_id] = s

        # --- Human annotations ---
        for sid, sample in all_samples.items():
            if sid not in raw:
                raw[sid] = {}
            conf_vec = {v: 1.0 for v in sample.ground_truth_values}
            raw[sid][human_id] = AnnotatorAnnotation(
                annotator_id=human_id,
                sample_id=sid,
                has_risk=sample.ground_truth_has_risk,
                value_set=set(sample.ground_truth_values),
                confidence_vector=conf_vec,
            )

        # --- LLM annotations ---
        # Also load cache for models not in active clients (from previous runs)
        all_model_keys = list(set(llm_annotators))

        def _process_sample(model_key: str, idx: int, sid: str, sample: ValueScenarioSample, total: int):
            """Process a single sample for a given model. Returns (status, annotation_or_none)."""
            cache_file = self.llm_outputs_dir / f"{model_key}_{sid}_output.json"
            use_cache = cache_file.exists() and not self.force_api

            if use_cache:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    has_risk = data.get("predicted_has_risk", False)
                    values = set(data.get("predicted_values", []))
                    confidences = data.get("predicted_confidences", {})
                    annot = AnnotatorAnnotation(
                        annotator_id=model_key,
                        sample_id=sid,
                        has_risk=has_risk,
                        value_set=values,
                        confidence_vector=confidences,
                    )
                    return "cache", annot
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Cache parse failed {cache_file}: {e}")

            # If no client available, skip API call
            if model_key not in self.llm_clients:
                return "skip", None

            # Call real API with retry
            progress = f"[{idx+1}/{total}]"
            self.logger.info(f"[{model_key}] {progress} API call: {sid}")

            has_risk, values, confidences = self._call_llm_with_retry(model_key, sample)

            if has_risk is None:
                self.logger.error(
                    f"[{model_key}] {progress} FAILED after {self.MAX_RETRIES} retries: {sid}"
                )
                return "failed", None

            # Write cache
            output_data = {
                "sample_id": sid,
                "model": model_key,
                "predicted_has_risk": has_risk,
                "predicted_values": list(values),
                "predicted_confidences": confidences,
                "ground_truth_has_risk": sample.ground_truth_has_risk,
                "ground_truth_values": sample.ground_truth_values,
                "timestamp": datetime.now().isoformat(),
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            annot = AnnotatorAnnotation(
                annotator_id=model_key,
                sample_id=sid,
                has_risk=has_risk,
                value_set=values,
                confidence_vector=confidences,
            )
            return "api", annot

        for model_key in all_model_keys:
            sample_items = list(all_samples.items())
            total = len(sample_items)
            loaded = 0
            called = 0
            failed = 0

            self.logger.info(
                f"[{model_key}] Processing {total} samples "
                f"(max_workers={self.max_workers})"
            )

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_sid = {
                    executor.submit(_process_sample, model_key, idx, sid, sample, total): sid
                    for idx, (sid, sample) in enumerate(sample_items)
                }

                done_count = 0
                for future in as_completed(future_to_sid):
                    sid = future_to_sid[future]
                    done_count += 1
                    try:
                        status, annot = future.result()
                    except Exception as exc:
                        self.logger.error(f"[{model_key}] Unexpected error for {sid}: {exc}")
                        failed += 1
                        continue

                    if status == "cache":
                        with raw_lock:
                            if sid not in raw:
                                raw[sid] = {}
                            raw[sid][model_key] = annot
                        loaded += 1
                    elif status == "api":
                        with raw_lock:
                            if sid not in raw:
                                raw[sid] = {}
                            raw[sid][model_key] = annot
                        called += 1
                        # Batch checkpoint
                        if self.batch_size > 0 and called % self.batch_size == 0:
                            self.logger.info(
                                f"[{model_key}] Batch checkpoint: {called} API calls done, "
                                f"{loaded} from cache, {failed} failed. "
                                f"Progress: {done_count}/{total}"
                            )
                    elif status == "failed":
                        failed += 1
                    # "skip" — no action needed

            self.logger.info(
                f"[{model_key}] Summary: cache={loaded}, api={called}, failed={failed}, "
                f"total={loaded + called}/{total}"
            )

        return raw

    def _call_llm_with_retry(
        self, model_key: str, sample: ValueScenarioSample
    ) -> tuple[Optional[bool], set, dict]:
        """Call LLM API with exponential backoff retry.

        Returns:
            (has_risk, value_set, confidences) or (None, set(), {}) if all retries fail
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                has_risk, values, confidences = self._call_llm_for_sample(model_key, sample)
                return has_risk, values, confidences
            except Exception as e:
                delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.warning(
                    f"[{model_key}] {sample.sample_id} attempt {attempt}/{self.MAX_RETRIES} "
                    f"failed: {e}. Retrying in {delay}s..."
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(delay)

        return None, set(), {}

    def _call_llm_for_sample(
        self, model_key: str, sample: ValueScenarioSample
    ) -> tuple[bool, set, dict]:
        """Call LLM for a single sample. Returns (has_risk, value_set, confidence_vector)."""
        client = self.llm_clients[model_key]
        scenario_type = self.sample_scenario_map.get(sample.sample_id, "code")

        prompts_config = self.config.get("prompts", {})
        mapping = {"code": "code_value_risk", "text": "text_value_risk"}
        prompt_key = mapping.get(scenario_type, "value_risk_identification")
        if prompt_key not in prompts_config:
            prompt_key = "value_risk_identification"
        prompt_config = prompts_config[prompt_key]

        system_prompt = prompt_config.get("system_prompt", "")
        user_prompt_template = prompt_config.get("user_prompt_template", "")

        if self.value_model_loader:
            value_model_text = self.value_model_loader.format_value_model_for_prompt()
        else:
            value_model_text = ""

        user_prompt = user_prompt_template.replace("{value_scenario}", sample.scenario_content)
        user_prompt = user_prompt.replace("{value_model}", value_model_text)

        response = client.call(system_prompt, user_prompt)

        if response.error:
            raise RuntimeError(f"API error: {response.error}")

        if not response.parsed_result:
            raise RuntimeError(f"Failed to parse JSON from response: {response.raw_response[:200]}")

        result = response.parsed_result
        has_risk = result.get("has_value_risk", False)
        identified_values = []
        confidences = {}
        for item in result.get("identified_values", []):
            value_id = item.get("value_id", "")
            if value_id:
                identified_values.append(value_id)
                confidences[value_id] = item.get("confidence", 1.0)

        return has_risk, set(identified_values), confidences

    # ------------------------------------------------------------------
    # CSV Export
    # ------------------------------------------------------------------
    def export_annotations_csv(
        self, raw_annotations: dict, output_path: str
    ) -> str:
        """Export all annotations to CSV for manual inspection.

        Columns:
            sample_id, scenario_type, human_has_risk, human_values,
            <model>_has_risk, <model>_values, <model>_risk_match, <model>_value_jaccard
            for each LLM model.

        Returns:
            Path to saved CSV file.
        """
        iaa_config = self.config.get("iaa_experiment", {})
        annotators_config = iaa_config.get("annotators", {})
        human_id = annotators_config.get("human", {}).get("annotator_id", "Human")

        # Discover all LLM annotator IDs present in annotations
        llm_ids = set()
        for sid_annots in raw_annotations.values():
            for aid in sid_annots.keys():
                if aid != human_id:
                    llm_ids.add(aid)
        llm_ids = sorted(llm_ids)

        # Build CSV header
        header = ["sample_id", "scenario_type", "human_has_risk", "human_values"]
        for model in llm_ids:
            header.extend([
                f"{model}_has_risk",
                f"{model}_values",
                f"{model}_risk_match",
                f"{model}_value_jaccard",
            ])

        rows = []
        for sid in sorted(raw_annotations.keys()):
            annots = raw_annotations[sid]
            human = annots.get(human_id)
            if not human:
                continue

            scenario_type = self.sample_scenario_map.get(sid, "unknown")
            human_values_str = "; ".join(sorted(human.value_set)) if human.value_set else ""

            row = [
                sid,
                scenario_type,
                str(human.has_risk),
                human_values_str,
            ]

            for model in llm_ids:
                llm_annot = annots.get(model)
                if llm_annot:
                    llm_values_str = "; ".join(sorted(llm_annot.value_set)) if llm_annot.value_set else ""
                    risk_match = "MATCH" if human.has_risk == llm_annot.has_risk else "MISMATCH"

                    # Jaccard similarity
                    h_set = human.value_set or set()
                    l_set = llm_annot.value_set or set()
                    if not h_set and not l_set:
                        jaccard = 1.0
                    elif not h_set or not l_set:
                        jaccard = 0.0
                    else:
                        jaccard = len(h_set & l_set) / len(h_set | l_set)

                    row.extend([
                        str(llm_annot.has_risk),
                        llm_values_str,
                        risk_match,
                        f"{jaccard:.4f}",
                    ])
                else:
                    row.extend(["N/A", "N/A", "N/A", "N/A"])

            rows.append(row)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        self.logger.info(f"Annotations CSV exported: {output_file} ({len(rows)} rows)")
        return str(output_file)

    # ------------------------------------------------------------------
    # Build matrix
    # ------------------------------------------------------------------
    def build_annotation_matrix(self, raw_annotations: dict) -> AnnotationMatrix:
        sample_ids = sorted(raw_annotations.keys())
        annotator_set = set()
        for sid_annots in raw_annotations.values():
            annotator_set.update(sid_annots.keys())
        annotator_ids = sorted(annotator_set)

        scenario_types = {}
        for sid in sample_ids:
            scenario_types[sid] = self.sample_scenario_map.get(sid, "unknown")

        return AnnotationMatrix(
            sample_ids=sample_ids,
            annotator_ids=annotator_ids,
            annotations=raw_annotations,
            scenario_types=scenario_types,
        )

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    def compute_pairwise_agreement(
        self, matrix: AnnotationMatrix
    ) -> dict[str, PairwiseAgreementResult]:
        """Compute pairwise agreement between all annotators (including Human)."""
        results = {}
        for a, b in itertools.combinations(matrix.annotator_ids, 2):
            key = f"{a}_vs_{b}" if a < b else f"{b}_vs_{a}"

            # Dim1
            risk_a, risk_b = matrix.get_risk_labels_for_pair(a, b)
            kappa = iaa_metrics.cohen_kappa_binary(risk_a, risk_b)
            pab = iaa_metrics.pabak(risk_a, risk_b)
            ac1 = iaa_metrics.gwet_ac1(risk_a, risk_b)
            pct = iaa_metrics.percent_agreement(risk_a, risk_b)

            # Dim2
            sets_a, sets_b = matrix.get_value_sets_for_pair(a, b)
            jaccard = iaa_metrics.pairwise_jaccard(sets_a, sets_b)
            f1 = iaa_metrics.pairwise_symmetric_f1(sets_a, sets_b)

            results[key] = PairwiseAgreementResult(
                annotator_a=a if a < b else b,
                annotator_b=b if a < b else a,
                dim1_cohen_kappa=kappa,
                dim1_pabak=pab,
                dim1_gwet_ac1=ac1,
                dim1_percent_agreement=pct,
                dim1_n_samples=len(risk_a),
                dim2_jaccard=jaccard,
                dim2_symmetric_f1=f1,
                dim2_n_samples=len(sets_a),
            )
        return results
    
    def _get_llm_only_matrix(self, matrix: AnnotationMatrix, human_id: str = "Human") -> AnnotationMatrix:
        """Return a new AnnotationMatrix with only LLM annotators (excluding Human)."""
        llm_ids = [aid for aid in matrix.annotator_ids if aid != human_id]
        if not llm_ids:
            return matrix
        
        # Create filtered annotations
        filtered_annotations = {}
        for sid in matrix.sample_ids:
            sample_annots = matrix.annotations.get(sid, {})
            filtered_annots = {
                aid: sample_annots[aid] 
                for aid in llm_ids 
                if aid in sample_annots
            }
            if filtered_annots:
                filtered_annotations[sid] = filtered_annots
        
        return AnnotationMatrix(
            sample_ids=matrix.sample_ids,
            annotator_ids=llm_ids,
            annotations=filtered_annotations,
            scenario_types=matrix.scenario_types,
        )

    def compute_multi_annotator_agreement(
        self,
        matrix: AnnotationMatrix,
        pairwise_results: dict[str, PairwiseAgreementResult],
    ) -> MultiAnnotatorAgreementResult:
        # Use LLM-only matrix for overall statistics (excluding Human)
        llm_matrix = self._get_llm_only_matrix(matrix)
        
        # Dim1: Fleiss' Kappa (LLM only)
        fleiss_matrix = llm_matrix.build_fleiss_risk_matrix()
        dim1_fleiss = iaa_metrics.fleiss_kappa(fleiss_matrix)

        # Dim1: Krippendorff's Alpha (LLM only)
        kripp_data = llm_matrix.build_krippendorff_risk_data()
        dim1_kripp = iaa_metrics.krippendorff_alpha_nominal(kripp_data)

        # Dim1: Avg pairwise (LLM only)
        llm_pairwise_results = {
            key: res for key, res in pairwise_results.items()
            if res.annotator_a != "Human" and res.annotator_b != "Human"
        }
        kappas = [r.dim1_cohen_kappa for r in llm_pairwise_results.values()]
        dim1_avg_kappa = sum(kappas) / len(kappas) if kappas else 0.0
        pabaks = [r.dim1_pabak for r in llm_pairwise_results.values()]
        ac1s = [r.dim1_gwet_ac1 for r in llm_pairwise_results.values()]
        dim1_avg_pabak = sum(pabaks) / len(pabaks) if pabaks else 0.0
        dim1_avg_ac1 = sum(ac1s) / len(ac1s) if ac1s else 0.0

        # Dim2: Avg pairwise (LLM only)
        jaccards = [r.dim2_jaccard for r in llm_pairwise_results.values()]
        f1s = [r.dim2_symmetric_f1 for r in llm_pairwise_results.values()]
        dim2_avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
        dim2_avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        # Dim2: Per-value Fleiss' Kappa (LLM only)
        per_value_kappa = iaa_metrics.per_value_fleiss_kappa(
            llm_matrix.build_per_value_binary_matrix,
            ALL_VALUE_IDS,
        )
        macro_values = [v for v in per_value_kappa.values() if v != 0.0]
        macro_avg = sum(macro_values) / len(macro_values) if macro_values else 0.0

        return MultiAnnotatorAgreementResult(
            dim1_fleiss_kappa=dim1_fleiss,
            dim1_krippendorff_alpha=dim1_kripp,
            dim1_avg_pairwise_kappa=dim1_avg_kappa,
            dim1_avg_pairwise_pabak=dim1_avg_pabak,
            dim1_avg_pairwise_ac1=dim1_avg_ac1,
            dim2_avg_pairwise_jaccard=dim2_avg_jaccard,
            dim2_avg_pairwise_f1=dim2_avg_f1,
            dim2_per_value_fleiss_kappa=per_value_kappa,
            dim2_macro_avg_value_kappa=macro_avg,
            n_annotators=len(llm_matrix.annotator_ids),
            n_samples=len(llm_matrix.sample_ids),
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _compute_for_matrix(
        self, matrix: AnnotationMatrix, scenario_type: str
    ) -> Optional[IAAExperimentResults]:
        if len(matrix.sample_ids) < 2:
            self.logger.warning(
                f"Scenario '{scenario_type}' has too few samples ({len(matrix.sample_ids)}), skipping"
            )
            return None

        pairwise = self.compute_pairwise_agreement(matrix)
        multi = self.compute_multi_annotator_agreement(matrix, pairwise)
        return IAAExperimentResults(
            pairwise=pairwise,
            multi_annotator=multi,
            scenario_type=scenario_type,
            annotator_ids=matrix.annotator_ids,
            n_samples=len(matrix.sample_ids),
        )

    def run(self) -> dict[str, IAAExperimentResults]:
        """Run full IAA experiment.

        Returns:
            {"code": IAAExperimentResults, "text": ..., "overall": ...}
        """
        self.setup()
        raw = self.collect_all_annotations()

        # Export CSV
        iaa_config = self.config.get("iaa_experiment", {})
        output_dir = project_root / iaa_config.get(
            "output_dir",
            str(exp_paths.IAA_DIR.relative_to(exp_paths.PROJECT_ROOT)),
        )
        csv_path = output_dir / "annotations_comparison.csv"
        self.export_annotations_csv(raw, str(csv_path))

        matrix = self.build_annotation_matrix(raw)

        self.logger.info(
            f"Annotation matrix: {len(matrix.sample_ids)} samples x {len(matrix.annotator_ids)} annotators"
        )

        results = {}

        # Per scenario
        for scenario_type in sorted(set(matrix.scenario_types.values())):
            sliced = matrix.slice_by_scenario(scenario_type)
            res = self._compute_for_matrix(sliced, scenario_type)
            if res is not None:
                results[scenario_type] = res

        # Overall
        res_overall = self._compute_for_matrix(matrix, "overall")
        if res_overall is not None:
            results["overall"] = res_overall

        return results


def main():
    parser = argparse.ArgumentParser(description="IAA Experiment")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        metavar="N",
        help="Max samples per scenario type. 0 (default) = use all samples",
    )
    parser.add_argument(
        "--force-api",
        action="store_true",
        help="Force re-call API (ignore cache, overwrite)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="N",
        help="Process N samples per batch with progress checkpoints. 0 = no batching",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Only run specific models (e.g., --models deepseek-chat qwen-plus)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel threads for API calls per model (default: 4)",
    )
    args = parser.parse_args()

    experiment = IAAExperiment(
        config_path=args.config,
        max_samples=args.max_samples,
        force_api=args.force_api,
        batch_size=args.batch_size,
        models=args.models,
        max_workers=args.max_workers,
    )
    results = experiment.run()

    # Print summary
    for scenario, res in results.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario} ({res.n_samples} samples, {len(res.annotator_ids)} annotators)")
        print(f"{'='*60}")

        if res.multi_annotator:
            m = res.multi_annotator
            print(f"  [Dim1] Fleiss κ = {m.dim1_fleiss_kappa:.4f}")
            print(f"  [Dim1] Krippendorff α = {m.dim1_krippendorff_alpha:.4f}")
            print(f"  [Dim1] Avg Pairwise κ = {m.dim1_avg_pairwise_kappa:.4f}")
            print(f"  [Dim1] Avg Pairwise PABAK = {m.dim1_avg_pairwise_pabak:.4f}")
            print(f"  [Dim1] Avg Pairwise AC1 = {m.dim1_avg_pairwise_ac1:.4f}")
            print(f"  [Dim2] Avg Pairwise Jaccard = {m.dim2_avg_pairwise_jaccard:.4f}")
            print(f"  [Dim2] Avg Pairwise F1 = {m.dim2_avg_pairwise_f1:.4f}")
            print(f"  [Dim2] Macro Avg Value κ = {m.dim2_macro_avg_value_kappa:.4f}")

        print("\n  Pairwise results:")
        for key, pr in res.pairwise.items():
            print(f"    {key}: κ={pr.dim1_cohen_kappa:.4f}, "
                  f"PABAK={pr.dim1_pabak:.4f}, "
                  f"AC1={pr.dim1_gwet_ac1:.4f}, "
                  f"%agree={pr.dim1_percent_agreement:.4f}, "
                  f"Jaccard={pr.dim2_jaccard:.4f}, "
                  f"F1={pr.dim2_symmetric_f1:.4f}")

    # Generate reports
    iaa_config = experiment.config.get("iaa_experiment", {})
    output_dir = args.output_dir or iaa_config.get(
        "output_dir",
        str(exp_paths.IAA_DIR.relative_to(exp_paths.PROJECT_ROOT)),
    )
    output_dir = Path(output_dir)
    reporter = IAAReportGenerator(output_dir=str(output_dir))
    saved = reporter.save_all(results)
    print(f"\nReports saved to: {output_dir}")
    for name, path in saved.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
