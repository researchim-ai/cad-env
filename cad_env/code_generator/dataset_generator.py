"""
Генератор датасетов для обучения LLM
"""

import json
import random
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
from .freecad_code_generator import FreeCADCodeGenerator

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Генератор датасетов для обучения LLM работе с FreeCAD
    """
    
    def __init__(self):
        """Инициализация генератора"""
        self.code_generator = FreeCADCodeGenerator()
        self.generated_samples = []
    
    def generate_training_dataset(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        Генерация датасета для обучения
        
        Args:
            num_samples: Количество образцов
            
        Returns:
            Список образцов для обучения
        """
        samples = []
        
        # Шаблоны для генерации
        templates = [
            # Создание базовых объектов
            ("Создай коробку размером {length}x{width}x{height}", "create_box"),
            ("Создай цилиндр радиусом {radius} и высотой {height}", "create_cylinder"),
            ("Создай сферу радиусом {radius}", "create_sphere"),
            ("Создай конус с радиусом основания {radius1} и высотой {height}", "create_cone"),
            ("Создай тор с внешним радиусом {radius1} и внутренним {radius2}", "create_torus"),
            
            # Трансформации
            ("Поверни объект на {angle} градусов вокруг оси {axis}", "rotate"),
            ("Перемести объект на {distance} единиц по оси {axis}", "translate"),
            ("Увеличь объект в {factor} раз", "scale"),
            ("Выдави объект на {distance} единиц", "extrude"),
            
            # Булевы операции
            ("Объедини два объекта", "union"),
            ("Вычти один объект из другого", "cut"),
            ("Найди пересечение объектов", "intersection"),
            
            # Сложные операции
            ("Создай скругление радиусом {radius}", "create_fillet"),
            ("Создай фаску размером {distance}", "create_chamfer"),
        ]
        
        for i in range(num_samples):
            try:
                # Выбираем случайный шаблон
                template, intent = random.choice(templates)
                
                # Генерируем параметры
                parameters = self._generate_random_parameters(intent)
                
                # Заполняем шаблон
                description = template.format(**parameters)
                
                # Генерируем код
                result = self.code_generator.generate_from_natural_language(description)
                
                if result["success"]:
                    sample = {
                        "id": f"sample_{i:06d}",
                        "description": description,
                        "intent": intent,
                        "parameters": parameters,
                        "generated_code": result["generated_code"],
                        "complexity": self._calculate_complexity(result["generated_code"]),
                        "category": self._categorize_intent(intent)
                    }
                    samples.append(sample)
                    self.generated_samples.append(sample)
                
            except Exception as e:
                logger.error(f"Ошибка генерации образца {i}: {e}")
                continue
        
        logger.info(f"Сгенерировано {len(samples)} образцов из {num_samples} попыток")
        return samples
    
    def generate_complex_scenarios(self, num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """
        Генерация сложных сценариев
        
        Args:
            num_scenarios: Количество сценариев
            
        Returns:
            Список сложных сценариев
        """
        scenarios = []
        
        scenario_templates = [
            # Создание механических деталей
            "Создай болт с резьбой: создай цилиндр, добавь головку, создай резьбу",
            "Создай шестерню: создай цилиндр, добавь зубья, создай отверстие",
            "Создай корпус: создай коробку, вычти внутреннее пространство, добавь отверстия",
            
            # Архитектурные элементы
            "Создай колонну: создай цилиндр, добавь капитель, создай базу",
            "Создай арку: создай полукруг, выдави, добавь опоры",
            "Создай купол: создай сферу, обрежь нижнюю часть",
            
            # Технические детали
            "Создай подшипник: создай внешнее кольцо, внутреннее кольцо, шарики",
            "Создай пружину: создай спираль, выдави по траектории",
            "Создай трубу: создай цилиндр, вычти внутренний цилиндр",
        ]
        
        for i in range(num_scenarios):
            try:
                # Выбираем случайный шаблон
                scenario_description = random.choice(scenario_templates)
                
                # Разбиваем на шаги
                steps = scenario_description.split(": ")[1].split(", ")
                
                # Генерируем код для каждого шага
                step_codes = []
                for step in steps:
                    result = self.code_generator.generate_from_natural_language(step.strip())
                    if result["success"]:
                        step_codes.append({
                            "description": step.strip(),
                            "code": result["generated_code"]
                        })
                
                if step_codes:
                    scenario = {
                        "id": f"scenario_{i:06d}",
                        "title": scenario_description.split(": ")[0],
                        "description": scenario_description,
                        "steps": step_codes,
                        "full_script": self._generate_full_script(step_codes),
                        "complexity": "high"
                    }
                    scenarios.append(scenario)
                
            except Exception as e:
                logger.error(f"Ошибка генерации сценария {i}: {e}")
                continue
        
        return scenarios
    
    def _generate_random_parameters(self, intent: str) -> Dict[str, Any]:
        """Генерация случайных параметров"""
        parameters = {}
        
        if intent == "create_box":
            parameters = {
                "length": round(random.uniform(5, 50), 1),
                "width": round(random.uniform(5, 50), 1),
                "height": round(random.uniform(5, 50), 1)
            }
        elif intent == "create_cylinder":
            parameters = {
                "radius": round(random.uniform(2, 20), 1),
                "height": round(random.uniform(5, 30), 1)
            }
        elif intent == "create_sphere":
            parameters = {
                "radius": round(random.uniform(2, 15), 1)
            }
        elif intent == "create_cone":
            parameters = {
                "radius1": round(random.uniform(3, 20), 1),
                "height": round(random.uniform(5, 25), 1)
            }
        elif intent == "create_torus":
            parameters = {
                "radius1": round(random.uniform(8, 25), 1),
                "radius2": round(random.uniform(2, 8), 1)
            }
        elif intent == "rotate":
            parameters = {
                "angle": random.choice([45, 90, 135, 180, 270]),
                "axis": random.choice(["X", "Y", "Z"])
            }
        elif intent == "translate":
            parameters = {
                "distance": round(random.uniform(5, 25), 1),
                "axis": random.choice(["X", "Y", "Z"])
            }
        elif intent == "scale":
            parameters = {
                "factor": round(random.uniform(0.5, 3.0), 1)
            }
        elif intent == "extrude":
            parameters = {
                "distance": round(random.uniform(2, 15), 1)
            }
        elif intent == "create_fillet":
            parameters = {
                "radius": round(random.uniform(0.5, 3.0), 1)
            }
        elif intent == "create_chamfer":
            parameters = {
                "distance": round(random.uniform(0.5, 2.0), 1)
            }
        
        return parameters
    
    def _calculate_complexity(self, code: str) -> str:
        """Расчет сложности кода"""
        lines = len(code.split('\n'))
        if lines <= 5:
            return "low"
        elif lines <= 15:
            return "medium"
        else:
            return "high"
    
    def _categorize_intent(self, intent: str) -> str:
        """Категоризация намерения"""
        if intent.startswith("create_"):
            return "creation"
        elif intent in ["rotate", "translate", "scale", "extrude"]:
            return "transformation"
        elif intent in ["union", "cut", "intersection"]:
            return "boolean"
        else:
            return "other"
    
    def _generate_full_script(self, step_codes: List[Dict[str, Any]]) -> str:
        """Генерация полного скрипта"""
        script_parts = [
            "import FreeCAD",
            "import Part",
            "import Draft",
            "",
            "# Создание документа",
            "doc = FreeCAD.newDocument('GeneratedDocument')",
            ""
        ]
        
        for i, step in enumerate(step_codes):
            script_parts.append(f"# Шаг {i+1}: {step['description']}")
            script_parts.append(step['code'])
            script_parts.append("")
        
        script_parts.extend([
            "# Показ документа",
            "FreeCAD.Gui.SendMsgToActiveView('ViewFit')",
            "",
            "print('Скрипт выполнен успешно!')"
        ])
        
        return "\n".join(script_parts)
    
    def save_dataset(self, samples: List[Dict[str, Any]], filepath: str):
        """
        Сохранение датасета в файл
        
        Args:
            samples: Образцы для сохранения
            filepath: Путь к файлу
        """
        dataset = {
            "metadata": {
                "total_samples": len(samples),
                "generated_at": str(Path().cwd()),
                "version": "1.0"
            },
            "samples": samples
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Датасет сохранен: {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Загрузка датасета из файла
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            Загруженные образцы
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        return dataset.get("samples", [])
    
    def generate_validation_dataset(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Генерация валидационного датасета
        
        Args:
            num_samples: Количество образцов
            
        Returns:
            Валидационные образцы
        """
        # Используем более сложные и разнообразные примеры
        complex_templates = [
            "Создай деталь для механизма: коробка 20x15x10, вычти цилиндр радиусом 3, добавь отверстия",
            "Создай архитектурный элемент: колонна высотой 30, диаметром 5, с капителью",
            "Создай техническую деталь: подшипник с внешним диаметром 20, внутренним 10",
            "Создай декоративный элемент: сложная форма из нескольких примитивов",
        ]
        
        validation_samples = []
        
        for i in range(num_samples):
            try:
                description = random.choice(complex_templates)
                result = self.code_generator.generate_from_natural_language(description)
                
                if result["success"]:
                    sample = {
                        "id": f"validation_{i:06d}",
                        "description": description,
                        "generated_code": result["generated_code"],
                        "complexity": "high",
                        "category": "validation"
                    }
                    validation_samples.append(sample)
                
            except Exception as e:
                logger.error(f"Ошибка генерации валидационного образца {i}: {e}")
                continue
        
        return validation_samples
