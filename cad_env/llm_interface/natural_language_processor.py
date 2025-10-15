"""
Обработчик естественного языка для LLM интерфейса
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class NaturalLanguageProcessor:
    """
    Обработчик естественного языка для определения намерений
    """
    
    def __init__(self):
        """Инициализация NLP процессора"""
        self.intent_patterns = {
            "create_object": [
                r"создай?\s+",
                r"сделай?\s+",
                r"добавь?\s+",
                r"построй?\s+",
                r"сгенерируй?\s+"
            ],
            "modify_object": [
                r"измени?\s+",
                r"модифицируй?\s+",
                r"отредактируй?\s+",
                r"настрой?\s+"
            ],
            "transform_object": [
                r"поверни?\s+",
                r"перемести?\s+",
                r"сдвинь?\s+",
                r"выдави?\s+",
                r"вытяни?\s+",
                r"масштабируй?\s+"
            ],
            "delete_object": [
                r"удали?\s+",
                r"убери?\s+",
                r"уничтожь?\s+"
            ],
            "query_object": [
                r"покажи?\s+",
                r"отобрази?\s+",
                r"выведи?\s+",
                r"расскажи?\s+",
                r"какой?\s+",
                r"что?\s+",
                r"где?\s+"
            ],
            "file_operation": [
                r"сохрани?\s+",
                r"загрузи?\s+",
                r"открой?\s+",
                r"экспортируй?\s+",
                r"импортируй?\s+"
            ]
        }
        
        self.object_types = [
            "коробка", "прямоугольник", "куб", "параллелепипед",
            "цилиндр", "сфера", "шар", "конус", "пирамида",
            "тор", "кольцо", "труба", "объект", "элемент"
        ]
        
        self.measurement_units = {
            "мм": 1.0,
            "см": 10.0,
            "м": 1000.0,
            "дюйм": 25.4,
            "inch": 25.4,
            "фут": 304.8,
            "foot": 304.8
        }
    
    def analyze_intent(self, text: str) -> str:
        """
        Анализировать текст и определить намерение
        
        Args:
            text: Текст для анализа
            
        Returns:
            Определенное намерение
        """
        text = text.lower().strip()
        
        # Поиск намерения по паттернам
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    logger.info(f"Найдено намерение '{intent}' для текста: {text}")
                    return intent
        
        # Если намерение не определено, возвращаем общее
        return "unknown"
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Извлечь сущности из текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с извлеченными сущностями
        """
        entities = {
            "object_types": [],
            "measurements": [],
            "coordinates": [],
            "angles": [],
            "colors": [],
            "materials": []
        }
        
        text = text.lower()
        
        # Поиск типов объектов
        for obj_type in self.object_types:
            if obj_type in text:
                entities["object_types"].append(obj_type)
        
        # Поиск измерений
        measurement_pattern = r"(\d+(?:\.\d+)?)\s*([а-яa-z]+)?"
        matches = re.findall(measurement_pattern, text)
        for value, unit in matches:
            value = float(value)
            if unit and unit in self.measurement_units:
                # Конвертация в мм
                value *= self.measurement_units[unit]
            entities["measurements"].append(value)
        
        # Поиск координат
        coord_pattern = r"([xyz])\s*[=:]\s*(\d+(?:\.\d+)?)"
        coord_matches = re.findall(coord_pattern, text)
        for axis, value in coord_matches:
            entities["coordinates"].append({
                "axis": axis.upper(),
                "value": float(value)
            })
        
        # Поиск углов
        angle_pattern = r"(\d+(?:\.\d+)?)\s*градусов?"
        angle_matches = re.findall(angle_pattern, text)
        for angle in angle_matches:
            entities["angles"].append(float(angle))
        
        return entities
    
    def get_suggestions(self, text: str) -> List[str]:
        """
        Получить предложения для автодополнения
        
        Args:
            text: Частично введенный текст
            
        Returns:
            Список предложений
        """
        suggestions = []
        text = text.lower().strip()
        
        # Предложения по командам
        if not text or text.startswith("создай") or text.startswith("сделай"):
            suggestions.extend([
                "создай коробку размером 10x5x3",
                "создай цилиндр радиусом 5 и высотой 10",
                "создай сферу радиусом 3"
            ])
        
        elif text.startswith("поверни") or text.startswith("поворот"):
            suggestions.extend([
                "поверни на 90 градусов вокруг оси Z",
                "поверни на 45 градусов вокруг оси X"
            ])
        
        elif text.startswith("перемести") or text.startswith("сдвинь"):
            suggestions.extend([
                "перемести на 10 единиц по оси X",
                "перемести на 5 единиц по оси Y"
            ])
        
        elif text.startswith("выдави") or text.startswith("вытяни"):
            suggestions.extend([
                "выдави на 5 единиц",
                "вытяни на 10 единиц"
            ])
        
        return suggestions[:5]  # Ограничиваем количество предложений
    
    def validate_text(self, text: str) -> Dict[str, Any]:
        """
        Проверить валидность текста
        
        Args:
            text: Текст для проверки
            
        Returns:
            Результат валидации
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not text or not text.strip():
            result["valid"] = False
            result["errors"].append("Пустой текст")
            return result
        
        # Проверка на наличие командных слов
        has_command = any(
            re.search(pattern, text.lower())
            for patterns in self.intent_patterns.values()
            for pattern in patterns
        )
        
        if not has_command:
            result["warnings"].append("Не найдено командных слов")
        
        # Проверка на наличие числовых значений для команд с параметрами
        has_numbers = bool(re.search(r"\d+(?:\.\d+)?", text))
        if not has_numbers and any(word in text.lower() for word in ["размером", "радиусом", "высотой"]):
            result["warnings"].append("Указаны размеры, но не найдены числовые значения")
        
        return result

