"""
IAA (Inter-Annotator Agreement) Data Structure Definitions
All annotators (LLM + human) are treated as equal peers; no ground truth is assumed.
"""

from dataclasses import dataclass, field
from typing import Optional


# All 20 value IDs
ALL_VALUE_IDS = [f"HV{i}" for i in range(1, 11)] + [f"SV{i}" for i in range(1, 11)]

VALUE_NAMES = {
    "HV1": "Conformity", "HV2": "Pleasure", "HV3": "Dignity",
    "HV4": "Inclusiveness", "HV5": "Sense of Belonging", "HV6": "Freedom",
    "HV7": "Independence", "HV8": "Wealth", "HV9": "Privacy", "HV10": "Security",
    "SV1": "Trust", "SV2": "Correctness", "SV3": "Compatibility",
    "SV4": "Portability", "SV5": "Reliability", "SV6": "Efficiency",
    "SV7": "Energy Preservation", "SV8": "Usability", "SV9": "Accessibility",
    "SV10": "Longevity",
}


@dataclass
class AnnotatorAnnotation:
    """A single annotator's annotation for a single sample."""
    annotator_id: str
    sample_id: str
    has_risk: bool
    value_set: set = field(default_factory=set)
    confidence_vector: dict = field(default_factory=dict)


@dataclass
class AnnotationMatrix:
    """Symmetric annotation matrix, the core data container.

    annotations structure: {sample_id: {annotator_id: AnnotatorAnnotation}}
    scenario_types structure: {sample_id: "code"/"text"}
    """
    sample_ids: list
    annotator_ids: list
    annotations: dict  # {sample_id: {annotator_id: AnnotatorAnnotation}}
    scenario_types: dict  # {sample_id: "code"/"text"}

    def slice_by_scenario(self, scenario_type: str) -> "AnnotationMatrix":
        """Filter by scenario type and return a new AnnotationMatrix."""
        filtered_ids = [
            sid for sid in self.sample_ids
            if self.scenario_types.get(sid) == scenario_type
        ]
        filtered_annotations = {
            sid: self.annotations[sid]
            for sid in filtered_ids
            if sid in self.annotations
        }
        filtered_types = {
            sid: self.scenario_types[sid]
            for sid in filtered_ids
            if sid in self.scenario_types
        }
        return AnnotationMatrix(
            sample_ids=filtered_ids,
            annotator_ids=self.annotator_ids,
            annotations=filtered_annotations,
            scenario_types=filtered_types,
        )

    def get_risk_labels_for_pair(
        self, annotator_a: str, annotator_b: str
    ) -> tuple:
        """Get risk label pairs for two annotators, skipping missing samples.

        Returns:
            (labels_a: list[bool], labels_b: list[bool])
        """
        labels_a = []
        labels_b = []
        for sid in self.sample_ids:
            sample_annots = self.annotations.get(sid, {})
            annot_a = sample_annots.get(annotator_a)
            annot_b = sample_annots.get(annotator_b)
            if annot_a is None or annot_b is None:
                continue
            labels_a.append(annot_a.has_risk)
            labels_b.append(annot_b.has_risk)
        return labels_a, labels_b

    def get_value_sets_for_pair(
        self, annotator_a: str, annotator_b: str
    ) -> tuple:
        """Get value set pairs for two annotators, skipping missing samples.

        Returns:
            (sets_a: list[set], sets_b: list[set])
        """
        sets_a = []
        sets_b = []
        for sid in self.sample_ids:
            sample_annots = self.annotations.get(sid, {})
            annot_a = sample_annots.get(annotator_a)
            annot_b = sample_annots.get(annotator_b)
            if annot_a is None or annot_b is None:
                continue
            sets_a.append(annot_a.value_set)
            sets_b.append(annot_b.value_set)
        return sets_a, sets_b

    def build_fleiss_risk_matrix(self) -> list:
        """Build the risk detection matrix required for Fleiss' Kappa.

        Returns:
            N x 2 matrix, each row is [n_no_risk, n_has_risk]
        """
        matrix = []
        for sid in self.sample_ids:
            sample_annots = self.annotations.get(sid, {})
            n_has_risk = 0
            n_no_risk = 0
            for aid in self.annotator_ids:
                annot = sample_annots.get(aid)
                if annot is None:
                    continue
                if annot.has_risk:
                    n_has_risk += 1
                else:
                    n_no_risk += 1
            if n_has_risk + n_no_risk > 0:
                matrix.append([n_no_risk, n_has_risk])
        return matrix

    def build_per_value_binary_matrix(self, value_id: str) -> list:
        """Build a binary Fleiss matrix for a specific value_id.

        Returns:
            N x 2 matrix, each row is [n_not_mentioned, n_mentioned]
        """
        matrix = []
        for sid in self.sample_ids:
            sample_annots = self.annotations.get(sid, {})
            n_mentioned = 0
            n_not_mentioned = 0
            for aid in self.annotator_ids:
                annot = sample_annots.get(aid)
                if annot is None:
                    continue
                if value_id in annot.value_set:
                    n_mentioned += 1
                else:
                    n_not_mentioned += 1
            if n_mentioned + n_not_mentioned > 0:
                matrix.append([n_not_mentioned, n_mentioned])
        return matrix

    def build_krippendorff_risk_data(self) -> list:
        """Build the reliability data matrix required for Krippendorff's Alpha.

        Returns:
            M x N matrix (M = annotators, N = samples), elements are 0/1/None (missing)
        """
        data = []
        for aid in self.annotator_ids:
            row = []
            for sid in self.sample_ids:
                sample_annots = self.annotations.get(sid, {})
                annot = sample_annots.get(aid)
                if annot is None:
                    row.append(None)
                else:
                    row.append(1 if annot.has_risk else 0)
            data.append(row)
        return data


@dataclass
class PairwiseAgreementResult:
    """Pairwise annotator agreement result (Dimension 1 + Dimension 2)."""
    annotator_a: str
    annotator_b: str
    # Dimension 1: Risk Detection
    dim1_cohen_kappa: float = 0.0
    dim1_pabak: float = 0.0
    dim1_gwet_ac1: float = 0.0
    dim1_percent_agreement: float = 0.0
    dim1_n_samples: int = 0
    # Dimension 2: Value ID Identification
    dim2_jaccard: float = 0.0
    dim2_symmetric_f1: float = 0.0
    dim2_n_samples: int = 0


@dataclass
class MultiAnnotatorAgreementResult:
    """Overall multi-annotator agreement result."""
    # Dimension 1
    dim1_fleiss_kappa: float = 0.0
    dim1_krippendorff_alpha: float = 0.0
    dim1_avg_pairwise_kappa: float = 0.0
    dim1_avg_pairwise_pabak: float = 0.0
    dim1_avg_pairwise_ac1: float = 0.0
    # Dimension 2
    dim2_avg_pairwise_jaccard: float = 0.0
    dim2_avg_pairwise_f1: float = 0.0
    dim2_per_value_fleiss_kappa: dict = field(default_factory=dict)
    dim2_macro_avg_value_kappa: float = 0.0
    # Statistics
    n_annotators: int = 0
    n_samples: int = 0


@dataclass
class IAAExperimentResults:
    """Final experiment results organized by scenario."""
    pairwise: dict = field(default_factory=dict)  # {"A_vs_B": PairwiseAgreementResult}
    multi_annotator: Optional[MultiAnnotatorAgreementResult] = None
    scenario_type: str = ""
    annotator_ids: list = field(default_factory=list)
    n_samples: int = 0
