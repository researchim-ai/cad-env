"""
Генератор FreeCAD Python кода
"""

import logging
from typing import Dict, Any, List, Optional
from .code_templates import FreeCADTemplates

logger = logging.getLogger(__name__)


class FreeCADCodeGenerator:
    """
    Генератор FreeCAD Python кода для LLM
    """
    
    def __init__(self):
        """Инициализация генератора"""
        self.templates = FreeCADTemplates()
        self.generated_code_history = []
    
    def generate_from_natural_language(self, description: str) -> Dict[str, Any]:
        """
        Генерировать FreeCAD код из естественного языка
        
        Args:
            description: Описание на естественном языке
            
        Returns:
            Словарь с сгенерированным кодом и метаданными
        """
        try:
            # Анализ описания
            intent = self._analyze_intent(description)
            parameters = self._extract_parameters(description)
            
            # Генерация кода
            code = self._generate_code(intent, parameters)
            
            result = {
                "success": True,
                "description": description,
                "intent": intent,
                "parameters": parameters,
                "generated_code": code,
                "executable": True
            }
            
            self.generated_code_history.append(result)
            logger.info(f"Сгенерирован код для: {description}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка генерации кода: {e}")
            return {
                "success": False,
                "description": description,
                "error": str(e),
                "generated_code": None
            }
    
    def generate_from_structured_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерировать код из структурированного запроса
        
        Args:
            request: Структурированный запрос
            
        Returns:
            Сгенерированный код
        """
        try:
            action = request.get("action")
            parameters = request.get("parameters", {})
            
            if not action:
                raise ValueError("Не указано действие")
            
            code = self._generate_code(action, parameters)
            
            return {
                "success": True,
                "action": action,
                "parameters": parameters,
                "generated_code": code,
                "executable": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generated_code": None
            }
    
    def _analyze_intent(self, description: str) -> str:
        """Анализ намерения из описания"""
        description = description.lower()
        
        # Создание объектов
        if any(word in description for word in ["создай", "сделай", "добавь", "построй"]):
            if any(word in description for word in ["коробка", "куб", "прямоугольник"]):
                return "create_box"
            elif any(word in description for word in ["цилиндр", "труба"]):
                return "create_cylinder"
            elif any(word in description for word in ["сфера", "шар"]):
                return "create_sphere"
            elif any(word in description for word in ["конус"]):
                return "create_cone"
            elif any(word in description for word in ["тор", "кольцо"]):
                return "create_torus"
        
        # Трансформации
        elif any(word in description for word in ["поверни", "поворот"]):
            return "rotate"
        elif any(word in description for word in ["перемести", "сдвинь"]):
            return "translate"
        elif any(word in description for word in ["масштабируй", "увеличь", "уменьши"]):
            return "scale"
        elif any(word in description for word in ["выдави", "вытяни"]):
            return "extrude"
        
        # Булевы операции
        elif any(word in description for word in ["объедини", "сложи"]):
            return "union"
        elif any(word in description for word in ["вычти", "удали"]):
            return "cut"
        elif any(word in description for word in ["пересечение"]):
            return "intersection"
        
        return "unknown"
    
    def _extract_parameters(self, description: str) -> Dict[str, Any]:
        """Извлечение параметров из описания"""
        import re
        
        parameters = {}
        description = description.lower()
        
        # Размеры
        size_match = re.search(r'размером?\s+(\d+(?:\.\d+)?)\s*[xх]\s*(\d+(?:\.\d+)?)(?:\s*[xх]\s*(\d+(?:\.\d+)?))?', description)
        if size_match:
            dims = [float(d) for d in size_match.groups() if d]
            if len(dims) >= 2:
                parameters["length"] = dims[0]
                parameters["width"] = dims[1]
                if len(dims) >= 3:
                    parameters["height"] = dims[2]
                else:
                    parameters["height"] = dims[1]
        
        # Радиус
        radius_match = re.search(r'радиусом?\s+(\d+(?:\.\d+)?)', description)
        if radius_match:
            parameters["radius"] = float(radius_match.group(1))
        
        # Высота
        height_match = re.search(r'высотой?\s+(\d+(?:\.\d+)?)', description)
        if height_match:
            parameters["height"] = float(height_match.group(1))
        
        # Угол поворота
        angle_match = re.search(r'(\d+(?:\.\d+)?)\s*градусов?', description)
        if angle_match:
            parameters["angle"] = float(angle_match.group(1))
        
        # Ось поворота
        axis_match = re.search(r'вокруг\s+оси\s+([xyz])', description)
        if axis_match:
            parameters["axis"] = axis_match.group(1).upper()
        
        # Расстояние перемещения
        distance_match = re.search(r'на\s+(\d+(?:\.\d+)?)\s*единиц?', description)
        if distance_match:
            parameters["distance"] = float(distance_match.group(1))
        
        return parameters
    
    def _generate_code(self, intent: str, parameters: Dict[str, Any]) -> str:
        """Генерация FreeCAD Python кода"""
        
        if intent == "create_box":
            return self.templates.create_box(
                length=parameters.get("length", 10),
                width=parameters.get("width", 10),
                height=parameters.get("height", 10)
            )
        
        elif intent == "create_cylinder":
            return self.templates.create_cylinder(
                radius=parameters.get("radius", 5),
                height=parameters.get("height", 10)
            )
        
        elif intent == "create_sphere":
            return self.templates.create_sphere(
                radius=parameters.get("radius", 5)
            )
        
        elif intent == "create_cone":
            return self.templates.create_cone(
                radius1=parameters.get("radius1", 5),
                radius2=parameters.get("radius2", 0),
                height=parameters.get("height", 10)
            )
        
        elif intent == "create_torus":
            return self.templates.create_torus(
                radius1=parameters.get("radius1", 10),
                radius2=parameters.get("radius2", 3)
            )
        
        elif intent == "rotate":
            return self.templates.rotate(
                angle=parameters.get("angle", 90),
                axis=parameters.get("axis", "Z")
            )
        
        elif intent == "translate":
            return self.templates.translate(
                x=parameters.get("x", 0),
                y=parameters.get("y", 0),
                z=parameters.get("z", 0)
            )
        
        elif intent == "scale":
            return self.templates.scale(
                factor=parameters.get("factor", 2)
            )
        
        elif intent == "extrude":
            return self.templates.extrude(
                distance=parameters.get("distance", 5)
            )
        
        elif intent == "union":
            return self.templates.union()
        
        elif intent == "cut":
            return self.templates.cut()
        
        elif intent == "intersection":
            return self.templates.intersection()
        
        else:
            return f"# Неизвестная команда: {intent}\n# Параметры: {parameters}"
    
    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Получить историю генерации кода"""
        return self.generated_code_history.copy()
    
    def generate_complex_script(self, descriptions: List[str]) -> str:
        """
        Генерация сложного скрипта из нескольких описаний
        
        Args:
            descriptions: Список описаний
            
        Returns:
            Полный FreeCAD скрипт
        """
        script_parts = [
            "import FreeCAD",
            "import Part",
            "import Draft",
            "",
            "# Создание документа",
            "doc = FreeCAD.newDocument('GeneratedDocument')",
            ""
        ]
        
        for i, description in enumerate(descriptions):
            result = self.generate_from_natural_language(description)
            if result["success"]:
                script_parts.append(f"# Шаг {i+1}: {description}")
                script_parts.append(result["generated_code"])
                script_parts.append("")
        
        script_parts.extend([
            "# Показ документа",
            "FreeCAD.Gui.SendMsgToActiveView('ViewFit')",
            "",
            "print('Скрипт выполнен успешно!')"
        ])
        
        return "\n".join(script_parts)
