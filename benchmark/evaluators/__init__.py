"""把 framework 预测事实转换为正式 benchmark 评估结果。"""

from .registry import evaluate_prediction, get_evaluator

__all__ = ["evaluate_prediction", "get_evaluator"]
