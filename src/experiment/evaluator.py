"""
评估指标计算模块
计算LLM预测与人工标注之间的一致性指标
"""

from dataclasses import dataclass, field
from typing import Optional, List

# numpy is optional - use pure Python fallback if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _mean(values: List[float]) -> float:
    """Calculate mean, with numpy if available, otherwise pure Python"""
    if not values:
        return 0.0
    if HAS_NUMPY:
        return float(np.mean(values))
    return sum(values) / len(values)


@dataclass
class PredictionResult:
    """单个样本的预测结果"""
    sample_id: str
    predicted_has_risk: bool
    predicted_values: list[str]  # 预测的价值ID列表
    predicted_confidences: dict[str, float] = field(default_factory=dict)  # value_id -> confidence
    ground_truth_has_risk: bool = False
    ground_truth_values: list[str] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    model_name: str
    # 风险检测指标（二分类）
    risk_precision: float = 0.0
    risk_recall: float = 0.0
    risk_f1: float = 0.0
    risk_accuracy: float = 0.0
    # 价值识别指标（多标签）- 严格匹配
    value_precision: float = 0.0
    value_recall: float = 0.0
    value_f1: float = 0.0
    jaccard_index: float = 0.0
    # 价值识别指标（宽松匹配）- 新增
    value_precision_loose: float = 0.0  # 只要包含人工标注的就认为正确
    value_recall_loose: float = 0.0     # 只要包含人工标注的就认为正确
    value_f1_loose: float = 0.0         # 只要包含人工标注的就认为正确
    # 整体指标
    exact_match: float = 0.0
    cohen_kappa: float = 0.0
    # 置信度相关
    avg_confidence: float = 0.0
    # 统计信息
    total_samples: int = 0
    valid_predictions: int = 0
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "risk_precision": round(self.risk_precision, 4),
            "risk_recall": round(self.risk_recall, 4),
            "risk_f1": round(self.risk_f1, 4),
            "risk_accuracy": round(self.risk_accuracy, 4),
            "value_precision": round(self.value_precision, 4),
            "value_recall": round(self.value_recall, 4),
            "value_f1": round(self.value_f1, 4),
            "jaccard_index": round(self.jaccard_index, 4),
            "value_precision_loose": round(self.value_precision_loose, 4),
            "value_recall_loose": round(self.value_recall_loose, 4),
            "value_f1_loose": round(self.value_f1_loose, 4),
            "exact_match": round(self.exact_match, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "total_samples": self.total_samples,
            "valid_predictions": self.valid_predictions,
        }


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def calculate(self, predictions: list[PredictionResult], model_name: str) -> EvaluationMetrics:
        """计算所有评估指标"""
        metrics = EvaluationMetrics(model_name=model_name)
        metrics.total_samples = len(predictions)
        metrics.valid_predictions = len([p for p in predictions if p.predicted_values is not None])
        
        if not predictions:
            return metrics
        
        # 计算风险检测指标
        self._calculate_risk_metrics(predictions, metrics)
        
        # 计算价值识别指标
        self._calculate_value_metrics(predictions, metrics)
        
        # 计算整体指标
        self._calculate_overall_metrics(predictions, metrics)
        
        # 计算置信度相关指标
        self._calculate_confidence_metrics(predictions, metrics)
        
        return metrics
    
    def _calculate_risk_metrics(self, predictions: list[PredictionResult], metrics: EvaluationMetrics) -> None:
        """计算风险检测的二分类指标"""
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for pred in predictions:
            if pred.ground_truth_has_risk and pred.predicted_has_risk:
                tp += 1
            elif not pred.ground_truth_has_risk and pred.predicted_has_risk:
                fp += 1
            elif not pred.ground_truth_has_risk and not pred.predicted_has_risk:
                tn += 1
            else:
                fn += 1
        
        # Precision
        metrics.risk_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall
        metrics.risk_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1
        if metrics.risk_precision + metrics.risk_recall > 0:
            metrics.risk_f1 = 2 * metrics.risk_precision * metrics.risk_recall / (metrics.risk_precision + metrics.risk_recall)
        # Accuracy
        total = tp + fp + tn + fn
        metrics.risk_accuracy = (tp + tn) / total if total > 0 else 0.0
    
    def _calculate_value_metrics(self, predictions: list[PredictionResult], metrics: EvaluationMetrics) -> None:
        """计算价值识别的多标签指标"""
        total_precision = 0.0
        total_recall = 0.0
        total_jaccard = 0.0
        # 宽松匹配指标
        total_precision_loose = 0.0
        total_recall_loose = 0.0
        valid_count = 0
        
        for pred in predictions:
            pred_set = set(self._filter_by_confidence(pred))
            gt_set = set(pred.ground_truth_values)
            
            # 跳过两边都为空的情况
            if not pred_set and not gt_set:
                continue
            
            valid_count += 1
            
            # 计算交集
            intersection = pred_set & gt_set
            
            # 严格匹配指标
            # Precision: 预测中有多少是正确的
            if pred_set:
                total_precision += len(intersection) / len(pred_set)
            
            # Recall: 真实标签中有多少被预测到
            if gt_set:
                total_recall += len(intersection) / len(gt_set)
            
            # Jaccard Index
            union = pred_set | gt_set
            if union:
                total_jaccard += len(intersection) / len(union)
            
            # 宽松匹配指标 - 新增逻辑
            # 只要预测包含人工标注的任意一个就认为正确
            if gt_set:  # 有真实标签时才计算
                if pred_set & gt_set:  # 预测包含至少一个人工标注的价值
                    total_precision_loose += 1.0
                    total_recall_loose += 1.0
                else:  # 预测不包含任何人工标注的价值
                    total_precision_loose += 0.0
                    total_recall_loose += 0.0
            else:  # 无真实标签时
                if pred_set:  # 有预测但无真实标签
                    total_precision_loose += 0.0
                    total_recall_loose += 0.0
                # 都为空的情况已在前面跳过
        
        if valid_count > 0:
            # 严格匹配指标
            metrics.value_precision = total_precision / valid_count
            metrics.value_recall = total_recall / valid_count
            metrics.jaccard_index = total_jaccard / valid_count
            
            # 严格匹配F1
            if metrics.value_precision + metrics.value_recall > 0:
                metrics.value_f1 = 2 * metrics.value_precision * metrics.value_recall / (metrics.value_precision + metrics.value_recall)
            
            # 宽松匹配指标
            metrics.value_precision_loose = total_precision_loose / valid_count
            metrics.value_recall_loose = total_recall_loose / valid_count
            
            # 宽松匹配F1
            if metrics.value_precision_loose + metrics.value_recall_loose > 0:
                metrics.value_f1_loose = 2 * metrics.value_precision_loose * metrics.value_recall_loose / (metrics.value_precision_loose + metrics.value_recall_loose)
    
    def _calculate_overall_metrics(self, predictions: list[PredictionResult], metrics: EvaluationMetrics) -> None:
        """计算整体指标"""
        # Exact Match: 预测与真实完全一致的比例
        exact_matches = 0
        for pred in predictions:
            pred_set = set(self._filter_by_confidence(pred))
            gt_set = set(pred.ground_truth_values)
            if pred_set == gt_set:
                exact_matches += 1
        
        metrics.exact_match = exact_matches / len(predictions) if predictions else 0.0
        
        # Cohen's Kappa（针对风险检测的二分类）
        metrics.cohen_kappa = self._calculate_cohen_kappa(predictions)
    
    def _calculate_cohen_kappa(self, predictions: list[PredictionResult]) -> float:
        """计算Cohen's Kappa系数"""
        if not predictions:
            return 0.0
        
        n = len(predictions)
        
        # 构建混淆矩阵
        tp, fp, tn, fn = 0, 0, 0, 0
        for pred in predictions:
            if pred.ground_truth_has_risk and pred.predicted_has_risk:
                tp += 1
            elif not pred.ground_truth_has_risk and pred.predicted_has_risk:
                fp += 1
            elif not pred.ground_truth_has_risk and not pred.predicted_has_risk:
                tn += 1
            else:
                fn += 1
        
        # 观察一致性
        po = (tp + tn) / n
        
        # 期望一致性
        p_pred_pos = (tp + fp) / n
        p_pred_neg = (tn + fn) / n
        p_gt_pos = (tp + fn) / n
        p_gt_neg = (tn + fp) / n
        
        pe = p_pred_pos * p_gt_pos + p_pred_neg * p_gt_neg
        
        # Kappa
        if pe == 1:
            return 1.0
        return (po - pe) / (1 - pe)
    
    def _calculate_confidence_metrics(self, predictions: list[PredictionResult], metrics: EvaluationMetrics) -> None:
        """计算置信度相关指标"""
        all_confidences = []
        for pred in predictions:
            all_confidences.extend(pred.predicted_confidences.values())
        
        if all_confidences:
            metrics.avg_confidence = _mean(all_confidences)
    
    def _filter_by_confidence(self, pred: PredictionResult) -> list[str]:
        """根据置信度阈值过滤预测结果"""
        if not pred.predicted_confidences:
            return pred.predicted_values
        
        filtered = []
        for value_id in pred.predicted_values:
            confidence = pred.predicted_confidences.get(value_id, 1.0)
            if confidence >= self.confidence_threshold:
                filtered.append(value_id)
        return filtered


def create_ground_truth_metrics(predictions: list[PredictionResult]) -> EvaluationMetrics:
    """创建人工标注的基准指标（与自身比较，所有指标应为1.0）"""
    metrics = EvaluationMetrics(model_name="Human (Ground Truth)")
    metrics.total_samples = len(predictions)
    metrics.valid_predictions = len(predictions)
    metrics.risk_precision = 1.0
    metrics.risk_recall = 1.0
    metrics.risk_f1 = 1.0
    metrics.risk_accuracy = 1.0
    metrics.value_precision = 1.0
    metrics.value_recall = 1.0
    metrics.value_f1 = 1.0
    metrics.jaccard_index = 1.0
    metrics.value_precision_loose = 1.0
    metrics.value_recall_loose = 1.0
    metrics.value_f1_loose = 1.0
    metrics.exact_match = 1.0
    metrics.cohen_kappa = 1.0
    metrics.avg_confidence = 1.0
    return metrics
