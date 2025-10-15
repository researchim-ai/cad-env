"""
Исполнитель FreeCAD Python кода
"""

import logging
import tempfile
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeExecutor:
    """
    Исполнитель сгенерированного FreeCAD Python кода
    """
    
    def __init__(self):
        """Инициализация исполнителя"""
        self.execution_history = []
        self.freecad_available = self._check_freecad_availability()
    
    def _check_freecad_availability(self) -> bool:
        """Проверка доступности FreeCAD"""
        try:
            import FreeCAD
            return True
        except ImportError:
            logger.warning("FreeCAD не доступен. Код будет выполняться в режиме симуляции.")
            return False
    
    def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Выполнение FreeCAD Python кода
        
        Args:
            code: Python код для выполнения
            timeout: Таймаут выполнения в секундах
            
        Returns:
            Результат выполнения
        """
        try:
            if self.freecad_available:
                return self._execute_with_freecad(code, timeout)
            else:
                return self._execute_simulation(code)
                
        except Exception as e:
            logger.error(f"Ошибка выполнения кода: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": 0
            }
    
    def _execute_with_freecad(self, code: str, timeout: int) -> Dict[str, Any]:
        """Выполнение с реальным FreeCAD"""
        import time
        import subprocess
        import tempfile
        
        start_time = time.time()
        
        # Создаем временный файл с кодом
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Выполняем код через FreeCAD
            result = subprocess.run(
                ['freecad', '-c', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error_output": result.stderr,
                "execution_time": execution_time,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Таймаут выполнения ({timeout}s)",
                "output": "",
                "execution_time": timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": time.time() - start_time
            }
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _execute_simulation(self, code: str) -> Dict[str, Any]:
        """Симуляция выполнения кода"""
        import time
        import re
        
        start_time = time.time()
        
        # Анализируем код для симуляции
        simulation_result = self._simulate_code_execution(code)
        
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "output": simulation_result["output"],
            "simulation": True,
            "execution_time": execution_time,
            "objects_created": simulation_result["objects_created"],
            "operations_performed": simulation_result["operations"]
        }
    
    def _simulate_code_execution(self, code: str) -> Dict[str, Any]:
        """Симуляция выполнения кода"""
        output_lines = []
        objects_created = []
        operations = []
        
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            # Анализируем создание объектов
            if 'addObject' in line:
                obj_match = re.search(r'addObject\("([^"]+)",\s*"([^"]+)"\)', line)
                if obj_match:
                    obj_type = obj_match.group(1)
                    obj_name = obj_match.group(2)
                    objects_created.append({"type": obj_type, "name": obj_name})
                    output_lines.append(f"Создан объект: {obj_name} ({obj_type})")
            
            # Анализируем операции
            elif 'recompute' in line:
                operations.append("recompute")
                output_lines.append("Пересчет документа")
            
            elif 'Placement' in line:
                operations.append("transformation")
                output_lines.append("Применена трансформация")
            
            elif 'addEdge' in line:
                operations.append("fillet/chamfer")
                output_lines.append("Добавлено скругление/фаска")
            
            elif 'addGeometry' in line:
                operations.append("sketch")
                output_lines.append("Добавлена геометрия в эскиз")
        
        return {
            "output": "\n".join(output_lines),
            "objects_created": objects_created,
            "operations": operations
        }
    
    def execute_script_file(self, filepath: str) -> Dict[str, Any]:
        """
        Выполнение скрипта из файла
        
        Args:
            filepath: Путь к файлу скрипта
            
        Returns:
            Результат выполнения
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return self.execute_code(code)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Ошибка чтения файла: {e}",
                "output": ""
            }
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Валидация кода без выполнения
        
        Args:
            code: Python код для валидации
            
        Returns:
            Результат валидации
        """
        import ast
        
        try:
            # Парсим код
            tree = ast.parse(code)
            
            # Анализируем синтаксис
            syntax_errors = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Проверяем вызовы FreeCAD
                    if hasattr(node.func, 'attr'):
                        if node.func.attr in ['addObject', 'newDocument']:
                            pass  # Валидные вызовы
            
            return {
                "valid": True,
                "syntax_errors": syntax_errors,
                "freecad_calls": self._extract_freecad_calls(code),
                "complexity": self._calculate_code_complexity(code)
            }
            
        except SyntaxError as e:
            return {
                "valid": False,
                "syntax_errors": [str(e)],
                "error": f"Синтаксическая ошибка: {e}"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": f"Ошибка валидации: {e}"
            }
    
    def _extract_freecad_calls(self, code: str) -> List[str]:
        """Извлечение вызовов FreeCAD из кода"""
        import re
        
        freecad_calls = []
        
        # Поиск вызовов FreeCAD
        patterns = [
            r'FreeCAD\.(\w+)',
            r'doc\.(\w+)',
            r'Part\.(\w+)',
            r'Draft\.(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            freecad_calls.extend(matches)
        
        return list(set(freecad_calls))
    
    def _calculate_code_complexity(self, code: str) -> Dict[str, int]:
        """Расчет сложности кода"""
        lines = code.split('\n')
        
        return {
            "total_lines": len(lines),
            "code_lines": len([line for line in lines if line.strip() and not line.startswith('#')]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "freecad_calls": len(self._extract_freecad_calls(code))
        }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Получить историю выполнения"""
        return self.execution_history.copy()
    
    def save_execution_result(self, result: Dict[str, Any]):
        """Сохранить результат выполнения"""
        self.execution_history.append(result)
