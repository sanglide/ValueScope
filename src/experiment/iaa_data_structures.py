"""
IAA (Inter-Annotator Agreement) 数据结构定义
所有标注者（LLM + 人类）地位平等，无 ground truth
"""

from dataclasses import dataclass, field
from typing import Optional


# 全部 20 个价值 ID
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
    """单个标注者对单个样本的标注"""
    annotator_id: str
    sample_id: str
    has_risk: bool
    value_set: set = field(default_factory=set)
    confidence_vector: dict = field(default_factory=dict)


@dataclass
class AnnotationMatrix:
    """对称标注矩阵，核心数据容器

    annotations 结构: {sample_id: {annotator_id: AnnotatorAnnotation}}
    scenario_types 结构: {sample_id: "code"/"text"}
    """
    sample_ids: list
    annotator_ids: list
    annotations: dict  # {sample_id: {annotator_id: AnnotatorAnnotation}}
    scenario_types: dict  # {sample_id: "code"/"text"}

    def slice_by_scenario(self, scenario_type: str) -> "AnnotationMatrix":
        """按场景类型过滤，返回新的 AnnotationMatrix"""
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
        """获取两个标注者的风险标签对，跳过缺失样本

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
        """获取两个标注者的价值集合对，跳过缺失样本

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
        """构建 Fleiss' Kappa 所需的风险检测矩阵

        Returns:
            N x 2 矩阵，每行 [n_no_risk, n_has_risk]
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
        """针对某个 value_id 构建二元 Fleiss 矩阵

        Returns:
            N x 2 矩阵，每行 [n_not_mentioned, n_mentioned]
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
        """构建 Krippendorff's Alpha 所需的可靠性数据矩阵

        Returns:
            M x N 矩阵 (M=标注者, N=样本), 元素为 0/1/None(缺失)
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
    """成对标注者的一致性结果（维度1 + 维度2）"""
    annotator_a: str
    annotator_b: str
    # 维度1: 风险检测
    dim1_cohen_kappa: float = 0.0
    dim1_pabak: float = 0.0
    dim1_gwet_ac1: float = 0.0
    dim1_percent_agreement: float = 0.0
    dim1_n_samples: int = 0
    # 维度2: 价值ID识别
    dim2_jaccard: float = 0.0
    dim2_symmetric_f1: float = 0.0
    dim2_n_samples: int = 0


@dataclass
class MultiAnnotatorAgreementResult:
    """全体标注者的一致性结果"""
    # 维度1
    dim1_fleiss_kappa: float = 0.0
    dim1_krippendorff_alpha: float = 0.0
    dim1_avg_pairwise_kappa: float = 0.0
    dim1_avg_pairwise_pabak: float = 0.0
    dim1_avg_pairwise_ac1: float = 0.0
    # 维度2
    dim2_avg_pairwise_jaccard: float = 0.0
    dim2_avg_pairwise_f1: float = 0.0
    dim2_per_value_fleiss_kappa: dict = field(default_factory=dict)
    dim2_macro_avg_value_kappa: float = 0.0
    # 统计
    n_annotators: int = 0
    n_samples: int = 0


@dataclass
class IAAExperimentResults:
    """按场景组织的最终实验结果"""
    pairwise: dict = field(default_factory=dict)  # {"A_vs_B": PairwiseAgreementResult}
    multi_annotator: Optional[MultiAnnotatorAgreementResult] = None
    scenario_type: str = ""
    annotator_ids: list = field(default_factory=list)
    n_samples: int = 0
