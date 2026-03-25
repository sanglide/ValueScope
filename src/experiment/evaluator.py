"""
Evaluation Metrics Module
Compute agreement metrics between LLM predictions and human annotations
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
    """Prediction result for a single sample"""
    sample_id: str
    predicted_has_risk: bool
    predicted_values: list[str]  # List of predicted value IDs
    predicted_confidences: dict[str, float] = field(default_factory=dict)  # value_id -> confidence
    ground_truth_has_risk: bool = False
    ground_truth_values: list[str] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics data class"""
    model_name: str
    # Risk detection metrics (binary classification)
    risk_precision: float = 0.0
    risk_recall: float = 0.0
    risk_f1: float = 0.0
    risk_accuracy: float = 0.0
    # Value identification metrics (multi-label) - strict matching
    value_precision: float = 0.0
    value_recall: float = 0.0
    value_f1: float = 0.0
    jaccard_index: float = 0.0
    # Value identification metrics (loose matching) - additional
    value_precision_loose: float = 0.0  # Correct if contains any human annotation
    value_recall_loose: float = 0.0     # Correct if contains any human annotation
    value_f1_loose: float = 0.0         # Correct if contains any human annotation
    # Overall metrics
    exact_match: float = 0.0
    cohen_kappa: float = 0.0
    # Confidence-related
    avg_confidence: float = 0.0
    # Statistics
    total_samples: int = 0
    valid_predictions: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
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
    """Metrics calculator"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def calculate(self, predictions: list[PredictionResult], model_name: str) -> EvaluationMetrics:
        """Calculate all evaluation metrics"""
        metrics = EvaluationMetrics(model_name=model_name)
        metrics.total_samples = len(predictions)
        metrics.valid_predictions = len([p for p in predictions if p.predicted_values is not None])
        
        if not predictions:
            return metrics
        
        # Calculate risk detection metrics
        self._calculate_risk_metrics(predictions, metrics)
        
        # Calculate value identification metrics
        self._calculate_value_metrics(predictions, metrics)
        
        # Calculate overall metrics
        self._calculate_overall_metrics(predictions, metrics)
        
        # Calculate confidence-related metrics
        self._calculate_confidence_metrics(predictions, metrics)
        
        return metrics
    
    def _calculate_risk_metrics(self, predictions: list[PredictionResult], metrics: EvaluationMetrics) -> None:
        """Calculate binary classification metrics for risk detection"""
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
        """Calculate multi-label metrics for value identification"""
        total_precision = 0.0
        total_recall = 0.0
        total_jaccard = 0.0
        # Loose matching metrics
        total_precision_loose = 0.0
        total_recall_loose = 0.0
        valid_count = 0
        
        for pred in predictions:
            pred_set = set(self._filter_by_confidence(pred))
            gt_set = set(pred.ground_truth_values)
            
            # Skip cases where both sides are empty
            if not pred_set and not gt_set:
                continue
            
            valid_count += 1
            
            # Calculate intersection
            intersection = pred_set & gt_set
            
            # Strict matching metrics
            # Precision: how many predictions are correct
            if pred_set:
                total_precision += len(intersection) / len(pred_set)
            
            # Recall: how many ground truth labels are predicted
            if gt_set:
                total_recall += len(intersection) / len(gt_set)
            
            # Jaccard Index
            union = pred_set | gt_set
            if union:
                total_jaccard += len(intersection) / len(union)
            
            # Loose matching metrics - additional logic
            # Correct if prediction contains any human annotation
            if gt_set:  # Only calculate when ground truth labels exist
                if pred_set & gt_set:  # Prediction contains at least one human-annotated value
                    total_precision_loose += 1.0
                    total_recall_loose += 1.0
                else:  # Prediction does not contain any human-annotated value
                    total_precision_loose += 0.0
                    total_recall_loose += 0.0
            else:  # No ground truth labels
                if pred_set:  # Has predictions but no ground truth labels
                    total_precision_loose += 0.0
                    total_recall_loose += 0.0
                # Both empty case already skipped above
        
        if valid_count > 0:
            # Strict matching metrics
            metrics.value_precision = total_precision / valid_count
            metrics.value_recall = total_recall / valid_count
            metrics.jaccard_index = total_jaccard / valid_count
            
            # Strict matching F1
            if metrics.value_precision + metrics.value_recall > 0:
                metrics.value_f1 = 2 * metrics.value_precision * metrics.value_recall / (metrics.value_precision + metrics.value_recall)
            
            # Loose matching metrics
            metrics.value_precision_loose = total_precision_loose / valid_count
            metrics.value_recall_loose = total_recall_loose / valid_count
            
            # Loose matching F1
            if metrics.value_precision_loose + metrics.value_recall_loose > 0:
                metrics.value_f1_loose = 2 * metrics.value_precision_loose * metrics.value_recall_loose / (metrics.value_precision_loose + metrics.value_recall_loose)
    
    def _calculate_overall_metrics(self, predictions: list[PredictionResult], metrics: EvaluationMetrics) -> None:
        """Calculate overall metrics"""
        # Exact Match: proportion where predictions exactly match ground truth
        exact_matches = 0
        for pred in predictions:
            pred_set = set(self._filter_by_confidence(pred))
            gt_set = set(pred.ground_truth_values)
            if pred_set == gt_set:
                exact_matches += 1
        
        metrics.exact_match = exact_matches / len(predictions) if predictions else 0.0
        
        # Cohen's Kappa (for risk detection binary classification)
        metrics.cohen_kappa = self._calculate_cohen_kappa(predictions)
    
    def _calculate_cohen_kappa(self, predictions: list[PredictionResult]) -> float:
        """Calculate Cohen's Kappa coefficient"""
        if not predictions:
            return 0.0
        
        n = len(predictions)
        
        # Build confusion matrix
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
        
        # Observed agreement
        po = (tp + tn) / n
        
        # Expected agreement
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
        """Calculate confidence-related metrics"""
        all_confidences = []
        for pred in predictions:
            all_confidences.extend(pred.predicted_confidences.values())
        
        if all_confidences:
            metrics.avg_confidence = _mean(all_confidences)
    
    def _filter_by_confidence(self, pred: PredictionResult) -> list[str]:
        """Filter prediction results by confidence threshold"""
        if not pred.predicted_confidences:
            return pred.predicted_values
        
        filtered = []
        for value_id in pred.predicted_values:
            confidence = pred.predicted_confidences.get(value_id, 1.0)
            if confidence >= self.confidence_threshold:
                filtered.append(value_id)
        return filtered


def create_ground_truth_metrics(predictions: list[PredictionResult]) -> EvaluationMetrics:
    """Create baseline metrics for human annotations (compared with itself, all metrics should be 1.0)"""
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
