"""
Пример генерации FreeCAD Python кода
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_env.code_generator import FreeCADCodeGenerator, CodeExecutor


def basic_code_generation():
    """Базовые примеры генерации кода"""
    print("=== Генерация FreeCAD Python кода ===")
    
    generator = FreeCADCodeGenerator()
    
    # Примеры описаний
    descriptions = [
        "Создай коробку размером 20x15x10",
        "Создай цилиндр радиусом 5 и высотой 15",
        "Создай сферу радиусом 8",
        "Поверни объект на 90 градусов вокруг оси Z",
        "Перемести объект на 10 единиц по оси X",
        "Выдави объект на 5 единиц"
    ]
    
    for description in descriptions:
        print(f"\nОписание: {description}")
        result = generator.generate_from_natural_language(description)
        
        if result["success"]:
            print("✅ Код сгенерирован:")
            print(result["generated_code"])
        else:
            print(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")


def complex_script_generation():
    """Генерация сложного скрипта"""
    print("\n=== Генерация сложного скрипта ===")
    
    generator = FreeCADCodeGenerator()
    
    # Сложный сценарий
    scenario = [
        "Создай коробку размером 30x20x15",
        "Создай цилиндр радиусом 5 и высотой 20",
        "Перемести цилиндр на 15 единиц по оси X",
        "Вычти цилиндр из коробки",
        "Создай сферу радиусом 3",
        "Перемести сферу на 10 единиц по оси Y",
        "Объедини сферу с коробкой"
    ]
    
    print("Сценарий:")
    for i, step in enumerate(scenario, 1):
        print(f"{i}. {step}")
    
    # Генерация полного скрипта
    full_script = generator.generate_complex_script(scenario)
    
    print(f"\nПолный FreeCAD скрипт:")
    print("-" * 50)
    print(full_script)
    print("-" * 50)


def code_execution_demo():
    """Демонстрация выполнения кода"""
    print("\n=== Выполнение сгенерированного кода ===")
    
    executor = CodeExecutor()
    generator = FreeCADCodeGenerator()
    
    # Генерируем простой код
    description = "Создай коробку размером 10x5x3"
    result = generator.generate_from_natural_language(description)
    
    if result["success"]:
        code = result["generated_code"]
        print(f"Сгенерированный код:")
        print(code)
        
        # Валидация кода
        validation = executor.validate_code(code)
        print(f"\nВалидация кода:")
        print(f"Валидный: {validation['valid']}")
        print(f"Сложность: {validation['complexity']}")
        
        # Выполнение кода (в режиме симуляции)
        execution_result = executor.execute_code(code)
        print(f"\nРезультат выполнения:")
        print(f"Успешно: {execution_result['success']}")
        if execution_result.get('simulation'):
            print("Режим симуляции")
        print(f"Вывод: {execution_result.get('output', '')}")


def structured_request_demo():
    """Демонстрация структурированных запросов"""
    print("\n=== Структурированные запросы ===")
    
    generator = FreeCADCodeGenerator()
    
    # Структурированные запросы
    requests = [
        {
            "action": "create_box",
            "parameters": {"length": 25, "width": 15, "height": 8}
        },
        {
            "action": "create_cylinder", 
            "parameters": {"radius": 7, "height": 12}
        },
        {
            "action": "rotate",
            "parameters": {"angle": 45, "axis": "Y"}
        }
    ]
    
    for i, request in enumerate(requests, 1):
        print(f"\nЗапрос {i}: {request}")
        result = generator.generate_from_structured_request(request)
        
        if result["success"]:
            print("✅ Код:")
            print(result["generated_code"])
        else:
            print(f"❌ Ошибка: {result.get('error')}")


if __name__ == "__main__":
    try:
        basic_code_generation()
        complex_script_generation()
        code_execution_demo()
        structured_request_demo()
        print("\n=== Все примеры выполнены успешно! ===")
    except Exception as e:
        print(f"Ошибка выполнения примеров: {e}")
        import traceback
        traceback.print_exc()
