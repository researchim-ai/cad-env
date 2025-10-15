"""
Пример обучения LLM для работы с FreeCAD
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_env.training import LLMTrainer, TrainingDataManager
from cad_env.code_generator import DatasetGenerator
import torch


def prepare_training_data():
    """Подготовка данных для обучения"""
    print("=== Подготовка данных для обучения ===")
    
    data_manager = TrainingDataManager("./training_data")
    
    # Генерация датасета
    print("Генерация датасета...")
    dataset = data_manager.generate_training_dataset(num_samples=200)
    
    # Разделение на train/val/test
    split_data = data_manager.split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    print(f"Обучающих данных: {len(split_data['train'])}")
    print(f"Валидационных данных: {len(split_data['validation'])}")
    print(f"Тестовых данных: {len(split_data['test'])}")
    
    return split_data


def setup_model():
    """Настройка модели"""
    print("\n=== Настройка модели ===")
    
    # Используем меньшую модель для демонстрации
    model_name = "microsoft/DialoGPT-small"  # Меньшая модель для демонстрации
    
    trainer = LLMTrainer(model_name=model_name)
    
    try:
        trainer.setup_model()
        print(f"✅ Модель {model_name} настроена успешно")
        return trainer
    except Exception as e:
        print(f"❌ Ошибка настройки модели: {e}")
        print("Примечание: Для полного обучения требуется установка transformers и torch")
        return None


def prepare_datasets(trainer, split_data):
    """Подготовка датасетов для обучения"""
    print("\n=== Подготовка датасетов ===")
    
    if not trainer:
        print("Модель не настроена, пропускаем подготовку датасетов")
        return None, None
    
    try:
        # Подготовка обучающих данных
        train_dataset = trainer.prepare_training_data(split_data['train'])
        print(f"✅ Обучающий датасет подготовлен: {len(train_dataset)} образцов")
        
        # Подготовка валидационных данных
        val_dataset = trainer.prepare_training_data(split_data['validation'])
        print(f"✅ Валидационный датасет подготовлен: {len(val_dataset)} образцов")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"❌ Ошибка подготовки датасетов: {e}")
        return None, None


def train_model(trainer, train_dataset, val_dataset):
    """Обучение модели"""
    print("\n=== Обучение модели ===")
    
    if not trainer or not train_dataset:
        print("Модель или данные не подготовлены, пропускаем обучение")
        return None
    
    try:
        # Параметры обучения (уменьшенные для демонстрации)
        training_result = trainer.train(
            training_data=train_dataset,
            validation_data=val_dataset,
            output_dir="./freecad_model",
            num_epochs=1,  # Уменьшено для демонстрации
            batch_size=2,  # Уменьшено для демонстрации
            learning_rate=5e-5
        )
        
        print("✅ Обучение завершено!")
        print(f"Финальная потеря: {training_result['train_loss']:.4f}")
        print(f"Время обучения: {training_result['train_runtime']:.2f} секунд")
        print(f"Модель сохранена в: {training_result['output_dir']}")
        
        return training_result
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        print("Примечание: Для полного обучения требуется GPU и больше времени")
        return None


def test_model_generation(trainer):
    """Тестирование генерации кода"""
    print("\n=== Тестирование генерации кода ===")
    
    if not trainer:
        print("Модель не настроена, пропускаем тестирование")
        return
    
    # Тестовые описания
    test_descriptions = [
        "Создай коробку размером 15x10x5",
        "Создай цилиндр радиусом 8 и высотой 12",
        "Поверни объект на 45 градусов вокруг оси Y",
        "Перемести объект на 20 единиц по оси Z"
    ]
    
    print("Тестирование генерации кода:")
    for i, description in enumerate(test_descriptions, 1):
        print(f"\n{i}. Описание: {description}")
        
        try:
            generated_code = trainer.generate_code(description, max_length=200)
            print(f"Сгенерированный код:")
            print(generated_code)
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")


def evaluate_model(trainer, test_data):
    """Оценка качества модели"""
    print("\n=== Оценка качества модели ===")
    
    if not trainer:
        print("Модель не настроена, пропускаем оценку")
        return
    
    try:
        # Оценка на тестовых данных
        evaluation_result = trainer.evaluate_model(test_data)
        
        print(f"Точность: {evaluation_result['accuracy']:.2%}")
        print(f"Правильных предсказаний: {evaluation_result['correct_predictions']}/{evaluation_result['total_samples']}")
        
        # Показываем несколько примеров
        print("\nПримеры результатов:")
        for i, result in enumerate(evaluation_result['results'][:3]):
            print(f"\nПример {i+1}:")
            print(f"Описание: {result['description']}")
            print(f"Схожесть: {result['similarity']:.2%}")
            print(f"Ожидаемый код: {result['expected'][:100]}...")
            print(f"Сгенерированный код: {result['generated'][:100]}...")
            
    except Exception as e:
        print(f"❌ Ошибка оценки: {e}")


def demonstrate_training_pipeline():
    """Демонстрация полного пайплайна обучения"""
    print("=== Полный пайплайн обучения LLM для FreeCAD ===")
    
    # 1. Подготовка данных
    split_data = prepare_training_data()
    
    # 2. Настройка модели
    trainer = setup_model()
    
    # 3. Подготовка датасетов
    train_dataset, val_dataset = prepare_datasets(trainer, split_data)
    
    # 4. Обучение (если возможно)
    if trainer and train_dataset:
        training_result = train_model(trainer, train_dataset, val_dataset)
        
        # 5. Тестирование генерации
        test_model_generation(trainer)
        
        # 6. Оценка качества
        evaluate_model(trainer, split_data['test'])
    
    print("\n=== Пайплайн обучения завершен ===")
    print("\nПримечания:")
    print("- Для полного обучения требуется GPU и больше времени")
    print("- Рекомендуется использовать более мощную модель (GPT-2, GPT-3)")
    print("- Можно использовать предобученные модели и fine-tuning")


def demonstrate_code_generation_without_training():
    """Демонстрация генерации кода без обучения"""
    print("\n=== Демонстрация генерации кода (без обучения) ===")
    
    from cad_env.code_generator import FreeCADCodeGenerator
    
    generator = FreeCADCodeGenerator()
    
    # Примеры генерации кода
    descriptions = [
        "Создай деталь для механизма: коробка 20x15x10, вычти цилиндр радиусом 3",
        "Создай архитектурный элемент: колонна высотой 30, диаметром 5",
        "Создай техническую деталь: подшипник с внешним диаметром 20, внутренним 10"
    ]
    
    for i, description in enumerate(descriptions, 1):
        print(f"\n{i}. Описание: {description}")
        
        result = generator.generate_from_natural_language(description)
        
        if result["success"]:
            print("✅ Код сгенерирован:")
            print(result["generated_code"])
        else:
            print(f"❌ Ошибка: {result.get('error')}")


if __name__ == "__main__":
    try:
        demonstrate_training_pipeline()
        demonstrate_code_generation_without_training()
        print("\n=== Все примеры обучения выполнены успешно! ===")
    except Exception as e:
        print(f"Ошибка выполнения примеров: {e}")
        import traceback
        traceback.print_exc()
