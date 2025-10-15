"""
Интерфейс для LLM агентов
"""

import logging
from typing import Dict, Any, List, Optional
from ..core import CADEnvironment
from .command_parser import CommandParser
from .natural_language_processor import NaturalLanguageProcessor

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Интерфейс для взаимодействия LLM агентов с CAD Environment
    """
    
    def __init__(self, environment: Optional[CADEnvironment] = None):
        """
        Инициализация LLM интерфейса
        
        Args:
            environment: CAD окружение
        """
        self.env = environment or CADEnvironment()
        self.command_parser = CommandParser()
        self.nlp = NaturalLanguageProcessor()
        
    def process_natural_language(self, text: str) -> Dict[str, Any]:
        """
        Обработать естественный язык и выполнить соответствующие действия
        
        Args:
            text: Текст на естественном языке
            
        Returns:
            Результат обработки
        """
        try:
            # Анализ текста
            intent = self.nlp.analyze_intent(text)
            logger.info(f"Определен интент: {intent}")
            
            # Парсинг команды
            command = self.command_parser.parse_natural_language(text, intent)
            logger.info(f"Спарсена команда: {command}")
            
            # Выполнение команды
            if command:
                result = self.env.execute_command(command['action'], **command.get('parameters', {}))
                return {
                    "success": True,
                    "intent": intent,
                    "command": command,
                    "result": result,
                    "message": f"Выполнена команда: {command['action']}"
                }
            else:
                return {
                    "success": False,
                    "intent": intent,
                    "message": "Не удалось определить команду"
                }
                
        except Exception as e:
            logger.error(f"Ошибка обработки естественного языка: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Ошибка обработки запроса"
            }
    
    def get_available_commands(self) -> List[Dict[str, Any]]:
        """
        Получить список доступных команд
        
        Returns:
            Список команд с описаниями
        """
        return [
            {
                "command": "create_box",
                "description": "Создать прямоугольный параллелепипед",
                "parameters": ["length", "width", "height"],
                "example": "Создай коробку размером 10x5x3"
            },
            {
                "command": "create_cylinder",
                "description": "Создать цилиндр",
                "parameters": ["radius", "height"],
                "example": "Создай цилиндр радиусом 5 и высотой 10"
            },
            {
                "command": "create_sphere",
                "description": "Создать сферу",
                "parameters": ["radius"],
                "example": "Создай сферу радиусом 3"
            },
            {
                "command": "extrude",
                "description": "Выдавить объект",
                "parameters": ["distance"],
                "example": "Выдави на 5 единиц"
            },
            {
                "command": "rotate",
                "description": "Повернуть объект",
                "parameters": ["angle", "axis"],
                "example": "Поверни на 90 градусов вокруг оси Z"
            },
            {
                "command": "translate",
                "description": "Переместить объект",
                "parameters": ["x", "y", "z"],
                "example": "Перемести на 10 единиц по X"
            }
        ]
    
    def get_context_info(self) -> Dict[str, Any]:
        """
        Получить контекстную информацию для LLM
        
        Returns:
            Информация о текущем состоянии
        """
        doc_info = self.env.get_document_info()
        history = self.env.get_history()
        
        return {
            "current_document": doc_info,
            "recent_operations": history[-5:] if history else [],
            "available_commands": self.get_available_commands(),
            "environment_status": "ready"
        }
    
    def execute_structured_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнить структурированную команду
        
        Args:
            command: Структурированная команда
            
        Returns:
            Результат выполнения
        """
        try:
            action = command.get('action')
            parameters = command.get('parameters', {})
            
            if not action:
                return {"success": False, "error": "Не указано действие"}
            
            result = self.env.execute_command(action, **parameters)
            
            return {
                "success": True,
                "action": action,
                "parameters": parameters,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Ошибка выполнения структурированной команды: {e}")
            return {
                "success": False,
                "error": str(e)
            }

