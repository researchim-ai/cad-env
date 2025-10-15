"""
CAD Environment - Окружение для обучения LLM и RL агентов работе с CAD
"""

__version__ = "0.1.0"
__author__ = "CAD Environment Team"

from .core import CADEnvironment
from .api import CADAPI
from .llm_interface import LLMInterface

__all__ = ["CADEnvironment", "CADAPI", "LLMInterface"]

