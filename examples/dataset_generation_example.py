"""
Пример генерации датасетов для обучения
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_env.code_generator import DatasetGenerator
from cad_env.training import TrainingDataManager
import json


def generate_training_dataset():
    """Генерация датасета для обучения"""
    print("=== Генерация датасета для обучения ===")
    
    generator = DatasetGenerator()
    data_manager = TrainingDataManager("./training_data")
    
    # Генерация базового датасета
    print("Генерация базового датасета...")
    basic_dataset = generator.generate_training_dataset(num_samples=100)
    
    print(f"Сгенерировано {len(basic_dataset)} базовых образцов")
    
    # Показываем несколько примеров
    print("\nПримеры сгенерированных образцов:")
    for i, sample in enumerate(basic_dataset[:3]):
        print(f"\nОбразец {i+1}:")
        print(f"Описание: {sample['description']}")
        print(f"Намерение: {sample['intent']}")
        print(f"Категория: {sample['category']}")
        print(f"Сложность: {sample['complexity']}")
        print(f"Код (первые 200 символов): {sample['generated_code'][:200]}...")
    
    return basic_dataset


def generate_complex_scenarios():
    """Генерация сложных сценариев"""
    print("\n=== Генерация сложных сценариев ===")
    
    generator = DatasetGenerator()
    
    # Генерация сложных сценариев
    scenarios = generator.generate_complex_scenarios(num_scenarios=5)
    
    print(f"Сгенерировано {len(scenarios)} сложных сценариев")
    
    # Показываем один сценарий
    if scenarios:
        scenario = scenarios[0]
        print(f"\nПример сценария: {scenario['title']}")
        print(f"Описание: {scenario['description']}")
        print(f"Шаги:")
        for i, step in enumerate(scenario['steps'], 1):
            print(f"  {i}. {step['description']}")
        
        print(f"\nПолный скрипт (первые 300 символов):")
        print(scenario['full_script'][:300] + "...")
    
    return scenarios


def dataset_statistics():
    """Анализ статистики датасета"""
    print("\n=== Статистика датасета ===")
    
    data_manager = TrainingDataManager("./training_data")
    
    # Генерация датасета
    dataset = data_manager.generate_training_dataset(num_samples=200)
    
    # Получение статистики
    stats = data_manager.get_dataset_statistics(dataset)
    
    print(f"Общее количество образцов: {stats['total_samples']}")
    print(f"Категории: {stats['categories']}")
    print(f"Сложность: {stats['complexities']}")
    print(f"Длина описаний: {stats['description_length']}")
    print(f"Длина кода: {stats['code_length']}")


def dataset_augmentation():
    """Аугментация датасета"""
    print("\n=== Аугментация датасета ===")
    
    data_manager = TrainingDataManager("./training_data")
    
    # Создаем небольшой базовый датасет
    base_dataset = [
        {
            "id": "sample_001",
            "description": "Создай коробку размером 10x5x3",
            "intent": "create_box",
            "category": "creation",
            "complexity": "low",
            "generated_code": "# Создание коробки\nbox = doc.addObject(\"Part::Box\", \"Box\")\nbox.Length = 10\nbox.Width = 5\nbox.Height = 3"
        }
    ]
    
    print(f"Исходный датасет: {len(base_dataset)} образцов")
    
    # Аугментация
    augmented_dataset = data_manager.augment_dataset(base_dataset)
    
    print(f"Аугментированный датасет: {len(augmented_dataset)} образцов")
    
    # Показываем примеры аугментации
    print("\nПримеры аугментации:")
    for i, sample in enumerate(augmented_dataset[:3]):
        print(f"{i+1}. {sample['description']}")


def create_balanced_dataset():
    """Создание сбалансированного датасета"""
    print("\n=== Создание сбалансированного датасета ===")
    
    data_manager = TrainingDataManager("./training_data")
    
    # Генерируем базовый датасет
    base_dataset = data_manager.generate_training_dataset(num_samples=100)
    
    # Создаем сбалансированный датасет
    balanced_dataset = data_manager.create_balanced_dataset(base_dataset, target_size=50)
    
    print(f"Исходный датасет: {len(base_dataset)} образцов")
    print(f"Сбалансированный датасет: {len(balanced_dataset)} образцов")
    
    # Статистика сбалансированного датасета
    stats = data_manager.get_dataset_statistics(balanced_dataset)
    print(f"Статистика сбалансированного датасета:")
    print(f"Категории: {stats['categories']}")


def save_and_load_dataset():
    """Сохранение и загрузка датасета"""
    print("\n=== Сохранение и загрузка датасета ===")
    
    generator = DatasetGenerator()
    
    # Генерация датасета
    dataset = generator.generate_training_dataset(num_samples=50)
    
    # Сохранение
    filepath = "example_dataset.json"
    generator.save_dataset(dataset, filepath)
    print(f"Датасет сохранен в {filepath}")
    
    # Загрузка
    loaded_dataset = generator.load_dataset(filepath)
    print(f"Датасет загружен: {len(loaded_dataset)} образцов")
    
    # Проверка целостности
    if len(dataset) == len(loaded_dataset):
        print("✅ Датасет загружен корректно")
    else:
        print("❌ Ошибка загрузки датасета")


if __name__ == "__main__":
    try:
        generate_training_dataset()
        generate_complex_scenarios()
        dataset_statistics()
        dataset_augmentation()
        create_balanced_dataset()
        save_and_load_dataset()
        print("\n=== Все примеры генерации датасетов выполнены успешно! ===")
    except Exception as e:
        print(f"Ошибка выполнения примеров: {e}")
        import traceback
        traceback.print_exc()
