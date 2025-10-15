"""
Интерфейс для обучения LLM работе с FreeCAD
"""

from .llm_trainer import LLMTrainer
from .training_data_manager import TrainingDataManager
from .model_evaluator import ModelEvaluator

__all__ = ["LLMTrainer", "TrainingDataManager", "ModelEvaluator"]
