"""
Базовые примеры использования CAD Environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_env import CADEnvironment, LLMInterface


def basic_cad_operations():
    """Базовые операции с CAD"""
    print("=== Базовые операции с CAD ===")
    
    # Создание окружения
    env = CADEnvironment()
    
    # Создание документа
    doc_id = env.create_document("TestDocument")
    print(f"Создан документ: {doc_id}")
    
    # Получение информации о документе
    info = env.get_document_info()
    print(f"Информация о документе: {info}")
    
    # Выполнение простой команды
    result = env.execute_command("create_box", length=10, width=5, height=3)
    print(f"Результат создания коробки: {result}")
    
    # Сохранение документа
    success = env.save_document("test_document.fcstd")
    print(f"Документ сохранен: {success}")
    
    # Получение истории операций
    history = env.get_history()
    print(f"История операций: {len(history)} записей")


def llm_interface_example():
    """Пример использования LLM интерфейса"""
    print("\n=== LLM интерфейс ===")
    
    # Создание LLM интерфейса
    llm = LLMInterface()
    
    # Обработка естественного языка
    natural_commands = [
        "Создай коробку размером 10x5x3",
        "Создай цилиндр радиусом 5 и высотой 10",
        "Поверни на 90 градусов вокруг оси Z",
        "Перемести на 10 единиц по оси X"
    ]
    
    for command in natural_commands:
        print(f"\nКоманда: {command}")
        result = llm.process_natural_language(command)
        print(f"Результат: {result}")
    
    # Получение доступных команд
    commands = llm.get_available_commands()
    print(f"\nДоступные команды: {len(commands)}")
    for cmd in commands[:3]:  # Показываем первые 3
        print(f"- {cmd['command']}: {cmd['description']}")
    
    # Получение контекстной информации
    context = llm.get_context_info()
    print(f"\nКонтекстная информация: {context['environment_status']}")


def api_example():
    """Пример использования API"""
    print("\n=== API пример ===")
    
    from cad_env.api import CADAPI
    
    # Создание API
    api = CADAPI()
    
    # Получение FastAPI приложения
    app = api.get_app()
    print(f"API приложение создано: {app.title}")
    
    # Пример структурированной команды
    command = {
        "action": "create_sphere",
        "parameters": {"radius": 5}
    }
    
    result = api.env.execute_command(command["action"], **command["parameters"])
    print(f"Результат выполнения команды через API: {result}")


if __name__ == "__main__":
    try:
        basic_cad_operations()
        llm_interface_example()
        api_example()
        print("\n=== Все примеры выполнены успешно! ===")
    except Exception as e:
        print(f"Ошибка выполнения примеров: {e}")
        import traceback
        traceback.print_exc()

