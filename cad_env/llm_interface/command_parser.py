"""
Парсер команд для LLM интерфейса
"""

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class CommandParser:
    """
    Парсер команд из естественного языка
    """
    
    def __init__(self):
        """Инициализация парсера"""
        self.command_patterns = {
            "create_box": [
                r"создай?\s+(?:коробку|прямоугольник|куб|параллелепипед)",
                r"сделай?\s+(?:коробку|прямоугольник|куб|параллелепипед)",
                r"добавь?\s+(?:коробку|прямоугольник|куб|параллелепипед)"
            ],
            "create_cylinder": [
                r"создай?\s+цилиндр",
                r"сделай?\s+цилиндр",
                r"добавь?\s+цилиндр"
            ],
            "create_sphere": [
                r"создай?\s+сферу",
                r"сделай?\s+сферу",
                r"добавь?\s+сферу"
            ],
            "extrude": [
                r"выдави?\s+(?:на\s+)?(\d+(?:\.\d+)?)",
                r"вытяни?\s+(?:на\s+)?(\d+(?:\.\d+)?)"
            ],
            "rotate": [
                r"поверни?\s+(?:на\s+)?(\d+(?:\.\d+)?)\s*градусов?\s*(?:вокруг\s+оси\s+)?([xyz])?",
                r"поворот\s+(?:на\s+)?(\d+(?:\.\d+)?)\s*градусов?"
            ],
            "translate": [
                r"перемести?\s+(?:на\s+)?(\d+(?:\.\d+)?)\s*(?:единиц?\s+)?(?:по\s+оси\s+)?([xyz])",
                r"сдвинь?\s+(?:на\s+)?(\d+(?:\.\d+)?)\s*(?:единиц?\s+)?(?:по\s+оси\s+)?([xyz])"
            ]
        }
        
        self.parameter_patterns = {
            "dimensions": r"размером?\s+(\d+(?:\.\d+)?)\s*[xх]\s*(\d+(?:\.\d+)?)(?:\s*[xх]\s*(\d+(?:\.\d+)?))?",
            "radius": r"радиусом?\s+(\d+(?:\.\d+)?)",
            "height": r"высотой?\s+(\d+(?:\.\d+)?)",
            "length": r"длиной?\s+(\d+(?:\.\d+)?)",
            "width": r"шириной?\s+(\d+(?:\.\d+)?)"
        }
    
    def parse_natural_language(self, text: str, intent: str) -> Optional[Dict[str, Any]]:
        """
        Парсить естественный язык в команду
        
        Args:
            text: Текст на естественном языке
            intent: Определенный интент
            
        Returns:
            Структурированная команда или None
        """
        text = text.lower().strip()
        
        # Поиск команды по паттернам
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return self._build_command(command, text, match.groups())
        
        return None
    
    def _build_command(self, command: str, text: str, groups: tuple) -> Dict[str, Any]:
        """
        Построить структурированную команду
        
        Args:
            command: Тип команды
            text: Исходный текст
            groups: Группы из регулярного выражения
            
        Returns:
            Структурированная команда
        """
        parameters = {}
        
        if command == "create_box":
            # Поиск размеров
            dim_match = re.search(self.parameter_patterns["dimensions"], text)
            if dim_match:
                dims = [float(d) for d in dim_match.groups() if d]
                if len(dims) >= 2:
                    parameters["length"] = dims[0]
                    parameters["width"] = dims[1]
                    if len(dims) >= 3:
                        parameters["height"] = dims[2]
                    else:
                        parameters["height"] = dims[1]  # По умолчанию
        
        elif command == "create_cylinder":
            # Поиск радиуса и высоты
            radius_match = re.search(self.parameter_patterns["radius"], text)
            height_match = re.search(self.parameter_patterns["height"], text)
            
            if radius_match:
                parameters["radius"] = float(radius_match.group(1))
            if height_match:
                parameters["height"] = float(height_match.group(1))
        
        elif command == "create_sphere":
            # Поиск радиуса
            radius_match = re.search(self.parameter_patterns["radius"], text)
            if radius_match:
                parameters["radius"] = float(radius_match.group(1))
        
        elif command == "extrude":
            # Извлечение расстояния из групп
            if groups and groups[0]:
                parameters["distance"] = float(groups[0])
        
        elif command == "rotate":
            # Извлечение угла и оси
            if groups:
                if groups[0]:
                    parameters["angle"] = float(groups[0])
                if len(groups) > 1 and groups[1]:
                    parameters["axis"] = groups[1].upper()
                else:
                    parameters["axis"] = "Z"  # По умолчанию
        
        elif command == "translate":
            # Извлечение расстояния и оси
            if groups:
                if groups[0]:
                    distance = float(groups[0])
                    axis = groups[1].upper() if len(groups) > 1 and groups[1] else "X"
                    parameters[axis.lower()] = distance
        
        return {
            "action": command,
            "parameters": parameters,
            "original_text": text
        }
    
    def validate_command(self, command: Dict[str, Any]) -> bool:
        """
        Проверить валидность команды
        
        Args:
            command: Команда для проверки
            
        Returns:
            True если команда валидна
        """
        if not command.get("action"):
            return False
        
        action = command["action"]
        parameters = command.get("parameters", {})
        
        # Проверка обязательных параметров
        required_params = {
            "create_box": ["length", "width", "height"],
            "create_cylinder": ["radius", "height"],
            "create_sphere": ["radius"],
            "extrude": ["distance"],
            "rotate": ["angle"],
            "translate": []  # Должна быть хотя бы одна ось
        }
        
        if action in required_params:
            required = required_params[action]
            if action == "translate":
                # Для translate нужна хотя бы одна ось
                return any(axis in parameters for axis in ["x", "y", "z"])
            else:
                return all(param in parameters for param in required)
        
        return True

