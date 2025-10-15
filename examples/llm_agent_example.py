"""
Пример LLM агента для работы с CAD
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_env import LLMInterface
import json


class CADAgent:
    """
    Простой LLM агент для работы с CAD
    """
    
    def __init__(self):
        """Инициализация агента"""
        self.llm_interface = LLMInterface()
        self.conversation_history = []
    
    def process_user_input(self, user_input: str) -> str:
        """
        Обработать пользовательский ввод
        
        Args:
            user_input: Ввод пользователя
            
        Returns:
            Ответ агента
        """
        # Добавляем в историю
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Обрабатываем через LLM интерфейс
        result = self.llm_interface.process_natural_language(user_input)
        
        # Формируем ответ
        if result["success"]:
            response = f"✅ Выполнено: {result['message']}"
            if result.get("result"):
                response += f"\nРезультат: {result['result']}"
        else:
            response = f"❌ Ошибка: {result.get('message', 'Неизвестная ошибка')}"
        
        # Добавляем ответ в историю
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_suggestions(self, partial_input: str = "") -> list:
        """
        Получить предложения для автодополнения
        
        Args:
            partial_input: Частичный ввод
            
        Returns:
            Список предложений
        """
        return self.llm_interface.nlp.get_suggestions(partial_input)
    
    def get_context(self) -> dict:
        """
        Получить контекстную информацию
        
        Returns:
            Контекстная информация
        """
        return self.llm_interface.get_context_info()
    
    def reset(self):
        """Сбросить состояние агента"""
        self.conversation_history = []
        self.llm_interface.env.reset()


def interactive_demo():
    """Интерактивная демонстрация агента"""
    print("=== CAD Agent - Интерактивная демонстрация ===")
    print("Введите команды на естественном языке или 'help' для справки")
    print("Введите 'quit' для выхода")
    print("-" * 50)
    
    agent = CADAgent()
    
    while True:
        try:
            user_input = input("\nCAD Agent> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'выход']:
                print("До свидания!")
                break
            
            elif user_input.lower() in ['help', 'справка']:
                print_help()
                continue
            
            elif user_input.lower() in ['context', 'контекст']:
                context = agent.get_context()
                print(f"Контекст: {json.dumps(context, indent=2, ensure_ascii=False)}")
                continue
            
            elif user_input.lower() in ['reset', 'сброс']:
                agent.reset()
                print("Состояние сброшено")
                continue
            
            elif user_input.lower().startswith('suggest'):
                partial = user_input[8:].strip()
                suggestions = agent.get_suggestions(partial)
                print(f"Предложения для '{partial}':")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
                continue
            
            if not user_input:
                continue
            
            # Обрабатываем команду
            response = agent.process_user_input(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nВыход...")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


def print_help():
    """Вывести справку"""
    help_text = """
Доступные команды:
- help/справка - показать эту справку
- context/контекст - показать контекстную информацию
- reset/сброс - сбросить состояние
- suggest <текст> - получить предложения для автодополнения
- quit/exit/выход - выйти из программы

Примеры CAD команд:
- "Создай коробку размером 10x5x3"
- "Создай цилиндр радиусом 5 и высотой 10"
- "Поверни на 90 градусов вокруг оси Z"
- "Перемести на 10 единиц по оси X"
- "Выдави на 5 единиц"
"""
    print(help_text)


def batch_demo():
    """Демонстрация пакетной обработки"""
    print("=== CAD Agent - Пакетная демонстрация ===")
    
    agent = CADAgent()
    
    # Список команд для демонстрации
    demo_commands = [
        "Создай коробку размером 20x10x5",
        "Создай цилиндр радиусом 3 и высотой 8",
        "Создай сферу радиусом 4",
        "Поверни на 45 градусов вокруг оси Z",
        "Перемести на 15 единиц по оси Y"
    ]
    
    print("Выполнение демонстрационных команд:")
    print("-" * 40)
    
    for i, command in enumerate(demo_commands, 1):
        print(f"\n{i}. {command}")
        response = agent.process_user_input(command)
        print(f"   {response}")
    
    # Показываем контекст
    print(f"\n{'='*50}")
    print("Финальный контекст:")
    context = agent.get_context()
    print(json.dumps(context, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CAD Agent Demo")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                       help="Режим работы: interactive или batch")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_demo()
    else:
        batch_demo()

