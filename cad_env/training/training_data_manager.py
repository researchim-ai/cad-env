"""
Менеджер данных для обучения
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..code_generator import DatasetGenerator

logger = logging.getLogger(__name__)


class TrainingDataManager:
    """
    Менеджер данных для обучения LLM
    """
    
    def __init__(self, data_dir: str = "./training_data"):
        """
        Инициализация менеджера данных
        
        Args:
            data_dir: Директория для хранения данных
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.dataset_generator = DatasetGenerator()
        
    def generate_training_dataset(self, 
                                 num_samples: int = 1000,
                                 save_to_file: bool = True) -> List[Dict[str, Any]]:
        """
        Генерация датасета для обучения
        
        Args:
            num_samples: Количество образцов
            save_to_file: Сохранить в файл
            
        Returns:
            Сгенерированный датасет
        """
        logger.info(f"Генерация датасета из {num_samples} образцов...")
        
        # Генерация базовых образцов
        basic_samples = self.dataset_generator.generate_training_dataset(num_samples // 2)
        
        # Генерация сложных сценариев
        complex_scenarios = self.dataset_generator.generate_complex_scenarios(num_samples // 4)
        
        # Генерация валидационных образцов
        validation_samples = self.dataset_generator.generate_validation_dataset(num_samples // 4)
        
        # Объединение всех образцов
        all_samples = basic_samples + complex_scenarios + validation_samples
        
        if save_to_file:
            self._save_dataset(all_samples, "training_dataset.json")
        
        logger.info(f"Сгенерировано {len(all_samples)} образцов")
        return all_samples
    
    def generate_validation_dataset(self, num_samples: int = 200) -> List[Dict[str, Any]]:
        """
        Генерация валидационного датасета
        
        Args:
            num_samples: Количество образцов
            
        Returns:
            Валидационный датасет
        """
        validation_samples = self.dataset_generator.generate_validation_dataset(num_samples)
        self._save_dataset(validation_samples, "validation_dataset.json")
        return validation_samples
    
    def load_dataset(self, filename: str) -> List[Dict[str, Any]]:
        """
        Загрузка датасета из файла
        
        Args:
            filename: Имя файла
            
        Returns:
            Загруженный датасет
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл {filepath} не найден")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get("samples", [])
    
    def split_dataset(self, 
                     dataset: List[Dict[str, Any]], 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Разделение датасета на train/val/test
        
        Args:
            dataset: Исходный датасет
            train_ratio: Доля обучающих данных
            val_ratio: Доля валидационных данных
            test_ratio: Доля тестовых данных
            
        Returns:
            Разделенный датасет
        """
        import random
        
        # Перемешивание данных
        shuffled_data = dataset.copy()
        random.shuffle(shuffled_data)
        
        total_samples = len(shuffled_data)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
    
    def augment_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Аугментация датасета
        
        Args:
            dataset: Исходный датасет
            
        Returns:
            Аугментированный датасет
        """
        augmented_samples = []
        
        for sample in dataset:
            # Добавляем оригинальный образец
            augmented_samples.append(sample)
            
            # Создаем вариации описания
            variations = self._create_description_variations(sample["description"])
            
            for variation in variations:
                # Генерируем код для вариации
                result = self.dataset_generator.code_generator.generate_from_natural_language(variation)
                
                if result["success"]:
                    augmented_sample = sample.copy()
                    augmented_sample["description"] = variation
                    augmented_sample["generated_code"] = result["generated_code"]
                    augmented_sample["id"] = f"{sample['id']}_aug_{len(augmented_samples)}"
                    augmented_samples.append(augmented_sample)
        
        logger.info(f"Аугментация: {len(dataset)} -> {len(augmented_samples)} образцов")
        return augmented_samples
    
    def _create_description_variations(self, description: str) -> List[str]:
        """Создание вариаций описания"""
        variations = []
        
        # Синонимы для замены
        synonyms = {
            "создай": ["сделай", "построй", "добавь", "сгенерируй"],
            "коробка": ["прямоугольник", "куб", "параллелепипед"],
            "цилиндр": ["труба", "вал"],
            "сфера": ["шар"],
            "поверни": ["поворот", "вращение"],
            "перемести": ["сдвинь", "передвинь"],
            "размером": ["размерами", "с размерами"]
        }
        
        # Создаем вариации с заменой слов
        for original, replacements in synonyms.items():
            if original in description.lower():
                for replacement in replacements:
                    variation = description.lower().replace(original, replacement)
                    if variation != description.lower():
                        variations.append(variation)
        
        return variations[:3]  # Ограничиваем количество вариаций
    
    def _save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Сохранение датасета в файл"""
        filepath = self.data_dir / filename
        
        data = {
            "metadata": {
                "total_samples": len(dataset),
                "generated_at": str(Path().cwd()),
                "version": "1.0"
            },
            "samples": dataset
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Датасет сохранен: {filepath}")
    
    def get_dataset_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Получение статистики датасета
        
        Args:
            dataset: Датасет для анализа
            
        Returns:
            Статистика датасета
        """
        if not dataset:
            return {"error": "Пустой датасет"}
        
        # Подсчет по категориям
        categories = {}
        complexities = {}
        
        for sample in dataset:
            category = sample.get("category", "unknown")
            complexity = sample.get("complexity", "unknown")
            
            categories[category] = categories.get(category, 0) + 1
            complexities[complexity] = complexities.get(complexity, 0) + 1
        
        # Анализ длины описаний
        description_lengths = [len(sample["description"]) for sample in dataset]
        code_lengths = [len(sample.get("generated_code", "")) for sample in dataset]
        
        return {
            "total_samples": len(dataset),
            "categories": categories,
            "complexities": complexities,
            "description_length": {
                "min": min(description_lengths),
                "max": max(description_lengths),
                "avg": sum(description_lengths) / len(description_lengths)
            },
            "code_length": {
                "min": min(code_lengths),
                "max": max(code_lengths),
                "avg": sum(code_lengths) / len(code_lengths)
            }
        }
    
    def create_balanced_dataset(self, 
                               base_dataset: List[Dict[str, Any]], 
                               target_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Создание сбалансированного датасета
        
        Args:
            base_dataset: Базовый датасет
            target_size: Целевой размер
            
        Returns:
            Сбалансированный датасет
        """
        # Группировка по категориям
        categories = {}
        for sample in base_dataset:
            category = sample.get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(sample)
        
        # Расчет количества образцов на категорию
        samples_per_category = target_size // len(categories)
        
        balanced_dataset = []
        for category, samples in categories.items():
            # Берем нужное количество образцов из каждой категории
            selected_samples = samples[:samples_per_category]
            balanced_dataset.extend(selected_samples)
        
        # Если не хватает образцов, добавляем из самых больших категорий
        while len(balanced_dataset) < target_size:
            for category, samples in categories.items():
                if len(balanced_dataset) >= target_size:
                    break
                if len(samples) > samples_per_category:
                    balanced_dataset.append(samples[samples_per_category])
                    samples_per_category += 1
        
        return balanced_dataset[:target_size]
