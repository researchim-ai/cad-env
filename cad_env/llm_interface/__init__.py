"""
Интерфейс для LLM агентов
"""

from .llm_interface import LLMInterface
from .command_parser import CommandParser
from .natural_language_processor import NaturalLanguageProcessor

__all__ = ["LLMInterface", "CommandParser", "NaturalLanguageProcessor"]

