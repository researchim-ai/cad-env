"""
Оценщик качества модели
"""

import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Оценщик качества LLM модели для генерации FreeCAD кода
    """
    
    def __init__(self):
        """Инициализация оценщика"""
        self.evaluation_history = []
    
    def evaluate_code_quality(self, generated_code: str, expected_code: str) -> Dict[str, Any]:
        """
        Оценка качества сгенерированного кода
        
        Args:
            generated_code: Сгенерированный код
            expected_code: Ожидаемый код
            
        Returns:
            Метрики качества
        """
        metrics = {
            "syntax_valid": self._check_syntax(generated_code),
            "semantic_similarity": self._calculate_semantic_similarity(generated_code, expected_code),
            "freecad_calls_match": self._check_freecad_calls(generated_code, expected_code),
            "parameter_accuracy": self._check_parameters(generated_code, expected_code),
            "overall_score": 0.0
        }
        
        # Расчет общего балла
        metrics["overall_score"] = (
            metrics["syntax_valid"] * 0.3 +
            metrics["semantic_similarity"] * 0.4 +
            metrics["freecad_calls_match"] * 0.2 +
            metrics["parameter_accuracy"] * 0.1
        )
        
        return metrics
    
    def _check_syntax(self, code: str) -> float:
        """Проверка синтаксиса кода"""
        try:
            compile(code, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.5
    
    def _calculate_semantic_similarity(self, code1: str, code2: str) -> float:
        """Расчет семантической схожести"""
        # Простая метрика схожести
        words1 = set(code1.lower().split())
        words2 = set(code2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _check_freecad_calls(self, generated: str, expected: str) -> float:
        """Проверка соответствия вызовов FreeCAD"""
        import re
        
        # Извлечение вызовов FreeCAD
        generated_calls = set(re.findall(r'FreeCAD\.(\w+)', generated))
        expected_calls = set(re.findall(r'FreeCAD\.(\w+)', expected))
        
        if not expected_calls:
            return 1.0 if not generated_calls else 0.5
        
        intersection = len(generated_calls.intersection(expected_calls))
        return intersection / len(expected_calls)
    
    def _check_parameters(self, generated: str, expected: str) -> float:
        """Проверка точности параметров"""
        import re
        
        # Извлечение числовых параметров
        generated_params = set(re.findall(r'(\d+(?:\.\d+)?)', generated))
        expected_params = set(re.findall(r'(\d+(?:\.\d+)?)', expected))
        
        if not expected_params:
            return 1.0
        
        intersection = len(generated_params.intersection(expected_params))
        return intersection / len(expected_params)
    
    def evaluate_model_performance(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Оценка производительности модели
        
        Args:
            test_results: Результаты тестирования
            
        Returns:
            Метрики производительности
        """
        if not test_results:
            return {"error": "Нет данных для оценки"}
        
        total_samples = len(test_results)
        successful_generations = sum(1 for r in test_results if r.get("success", False))
        
        # Расчет метрик
        accuracy = successful_generations / total_samples if total_samples > 0 else 0
        
        # Средние баллы качества
        quality_scores = []
        for result in test_results:
            if "quality_metrics" in result:
                quality_scores.append(result["quality_metrics"].get("overall_score", 0))
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Анализ ошибок
        error_types = {}
        for result in test_results:
            if not result.get("success", False):
                error_type = result.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_samples": total_samples,
            "successful_generations": successful_generations,
            "accuracy": accuracy,
            "average_quality_score": avg_quality,
            "error_distribution": error_types,
            "performance_grade": self._calculate_performance_grade(accuracy, avg_quality)
        }
    
    def _calculate_performance_grade(self, accuracy: float, quality: float) -> str:
        """Расчет общей оценки производительности"""
        overall_score = (accuracy + quality) / 2
        
        if overall_score >= 0.9:
            return "A+"
        elif overall_score >= 0.8:
            return "A"
        elif overall_score >= 0.7:
            return "B"
        elif overall_score >= 0.6:
            return "C"
        else:
            return "D"
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Генерация отчета об оценке
        
        Args:
            evaluation_results: Результаты оценки
            
        Returns:
            Текстовый отчет
        """
        report = f"""
=== ОТЧЕТ ОБ ОЦЕНКЕ МОДЕЛИ ===

Общие метрики:
- Всего образцов: {evaluation_results.get('total_samples', 0)}
- Успешных генераций: {evaluation_results.get('successful_generations', 0)}
- Точность: {evaluation_results.get('accuracy', 0):.2%}
- Средний балл качества: {evaluation_results.get('average_quality_score', 0):.2f}
- Общая оценка: {evaluation_results.get('performance_grade', 'N/A')}

Распределение ошибок:
"""
        
        error_dist = evaluation_results.get("error_distribution", {})
        if error_dist:
            for error_type, count in error_dist.items():
                report += f"- {error_type}: {count}\n"
        else:
            report += "- Ошибок не обнаружено\n"
        
        report += "\n=== РЕКОМЕНДАЦИИ ===\n"
        
        accuracy = evaluation_results.get("accuracy", 0)
        quality = evaluation_results.get("average_quality_score", 0)
        
        if accuracy < 0.7:
            report += "- Увеличить количество обучающих данных\n"
            report += "- Улучшить качество датасета\n"
        
        if quality < 0.7:
            report += "- Улучшить шаблоны генерации кода\n"
            report += "- Добавить больше примеров в обучение\n"
        
        if accuracy >= 0.8 and quality >= 0.8:
            report += "- Модель показывает отличные результаты!\n"
            report += "- Можно переходить к продакшн использованию\n"
        
        return report
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str):
        """Сохранение результатов оценки"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Результаты оценки сохранены: {filepath}")
    
    def load_evaluation_results(self, filepath: str) -> Dict[str, Any]:
        """Загрузка результатов оценки"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
