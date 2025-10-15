"""
Тренер для LLM работы с FreeCAD
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

logger = logging.getLogger(__name__)


class LLMTrainer:
    """
    Тренер для обучения LLM работе с FreeCAD
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Инициализация тренера
        
        Args:
            model_name: Название базовой модели
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_data = []
        
    def setup_model(self):
        """Настройка модели и токенизатора"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Добавляем специальные токены для FreeCAD
            special_tokens = {
                "pad_token": "<pad>",
                "eos_token": "<eos>",
                "bos_token": "<bos>",
                "sep_token": "<sep>"
            }
            
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"Модель {self.model_name} загружена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def prepare_training_data(self, dataset: List[Dict[str, Any]]) -> Dataset:
        """
        Подготовка данных для обучения
        
        Args:
            dataset: Датасет для обучения
            
        Returns:
            Подготовленный датасет
        """
        if not self.tokenizer:
            raise ValueError("Токенизатор не инициализирован. Вызовите setup_model() сначала.")
        
        training_texts = []
        
        for sample in dataset:
            # Создаем текст для обучения в формате: описание -> код
            text = f"<bos>Описание: {sample['description']}<sep>Код: {sample['generated_code']}<eos>"
            training_texts.append(text)
        
        # Токенизация
        tokenized = self.tokenizer(
            training_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Создание датасета
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })
        
        logger.info(f"Подготовлено {len(dataset)} образцов для обучения")
        return dataset
    
    def train(self, 
              training_data: Dataset,
              validation_data: Optional[Dataset] = None,
              output_dir: str = "./freecad_model",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-5) -> Dict[str, Any]:
        """
        Обучение модели
        
        Args:
            training_data: Данные для обучения
            validation_data: Данные для валидации
            output_dir: Директория для сохранения модели
            num_epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Скорость обучения
            
        Returns:
            Результаты обучения
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Модель не инициализирована. Вызовите setup_model() сначала.")
        
        # Настройка аргументов обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps" if validation_data else "no",
            eval_steps=500 if validation_data else None,
            save_total_limit=2,
            load_best_model_at_end=True if validation_data else False,
        )
        
        # Коллатор данных
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Создание тренера
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_data,
            eval_dataset=validation_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Обучение
        logger.info("Начало обучения...")
        training_result = self.trainer.train()
        
        # Сохранение модели
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Обучение завершено. Модель сохранена в {output_dir}")
        
        return {
            "train_loss": training_result.training_loss,
            "train_runtime": training_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": training_result.metrics.get("train_samples_per_second", 0),
            "output_dir": output_dir
        }
    
    def generate_code(self, description: str, max_length: int = 256) -> str:
        """
        Генерация кода по описанию
        
        Args:
            description: Описание на естественном языке
            max_length: Максимальная длина генерируемого текста
            
        Returns:
            Сгенерированный код
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Модель не инициализирована")
        
        # Подготовка входного текста
        input_text = f"<bos>Описание: {description}<sep>Код:"
        
        # Токенизация
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Декодирование
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлечение только кода
        if "<sep>Код:" in generated_text:
            code = generated_text.split("<sep>Код:")[1].strip()
            # Убираем лишние токены
            code = code.replace("<eos>", "").strip()
            return code
        else:
            return generated_text
    
    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Оценка качества модели
        
        Args:
            test_data: Тестовые данные
            
        Returns:
            Метрики качества
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Модель не инициализирована")
        
        correct_predictions = 0
        total_predictions = len(test_data)
        results = []
        
        for sample in test_data:
            description = sample["description"]
            expected_code = sample["generated_code"]
            
            # Генерация кода
            generated_code = self.generate_code(description)
            
            # Простая оценка (можно улучшить)
            similarity = self._calculate_similarity(generated_code, expected_code)
            
            results.append({
                "description": description,
                "expected": expected_code,
                "generated": generated_code,
                "similarity": similarity
            })
            
            if similarity > 0.7:  # Порог схожести
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            "accuracy": accuracy,
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "results": results
        }
    
    def _calculate_similarity(self, code1: str, code2: str) -> float:
        """Расчет схожести между двумя кодами"""
        # Простая метрика схожести (можно улучшить)
        words1 = set(code1.lower().split())
        words2 = set(code2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def save_model(self, path: str):
        """Сохранение модели"""
        if self.model and self.tokenizer:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Модель сохранена в {path}")
        else:
            raise ValueError("Модель не инициализирована")
    
    def load_model(self, path: str):
        """Загрузка модели"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(path)
            logger.info(f"Модель загружена из {path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
