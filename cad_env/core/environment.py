"""
Основной класс CAD Environment
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .freecad_wrapper import FreeCADWrapper

logger = logging.getLogger(__name__)


class CADEnvironment:
    """
    Основное окружение для работы с CAD системами.
    Предоставляет интерфейс для LLM и RL агентов.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация CAD окружения
        
        Args:
            config: Конфигурация окружения
        """
        self.config = config or {}
        self.freecad = FreeCADWrapper()
        self.current_document = None
        self.history = []
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        
    def create_document(self, name: str = "NewDocument") -> str:
        """
        Создать новый документ
        
        Args:
            name: Имя документа
            
        Returns:
            ID созданного документа
        """
        try:
            doc_id = self.freecad.create_document(name)
            self.current_document = doc_id
            self.history.append({"action": "create_document", "name": name, "doc_id": doc_id})
            logger.info(f"Создан документ: {name} (ID: {doc_id})")
            return doc_id
        except Exception as e:
            logger.error(f"Ошибка создания документа: {e}")
            raise
    
    def save_document(self, filepath: str) -> bool:
        """
        Сохранить текущий документ
        
        Args:
            filepath: Путь для сохранения
            
        Returns:
            True если успешно сохранено
        """
        try:
            success = self.freecad.save_document(self.current_document, filepath)
            if success:
                self.history.append({"action": "save_document", "filepath": filepath})
                logger.info(f"Документ сохранен: {filepath}")
            return success
        except Exception as e:
            logger.error(f"Ошибка сохранения документа: {e}")
            return False
    
    def load_document(self, filepath: str) -> str:
        """
        Загрузить документ из файла
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            ID загруженного документа
        """
        try:
            doc_id = self.freecad.load_document(filepath)
            self.current_document = doc_id
            self.history.append({"action": "load_document", "filepath": filepath, "doc_id": doc_id})
            logger.info(f"Документ загружен: {filepath} (ID: {doc_id})")
            return doc_id
        except Exception as e:
            logger.error(f"Ошибка загрузки документа: {e}")
            raise
    
    def get_document_info(self) -> Dict[str, Any]:
        """
        Получить информацию о текущем документе
        
        Returns:
            Словарь с информацией о документе
        """
        if not self.current_document:
            return {"error": "Нет активного документа"}
        
        return self.freecad.get_document_info(self.current_document)
    
    def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Выполнить CAD команду
        
        Args:
            command: Команда для выполнения
            **kwargs: Дополнительные параметры
            
        Returns:
            Результат выполнения команды
        """
        try:
            result = self.freecad.execute_command(command, **kwargs)
            self.history.append({
                "action": "execute_command", 
                "command": command, 
                "kwargs": kwargs,
                "result": result
            })
            logger.info(f"Выполнена команда: {command}")
            return result
        except Exception as e:
            logger.error(f"Ошибка выполнения команды {command}: {e}")
            raise
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Получить историю операций
        
        Returns:
            Список операций
        """
        return self.history.copy()
    
    def reset(self):
        """
        Сбросить окружение
        """
        self.freecad.reset()
        self.current_document = None
        self.history = []
        logger.info("Окружение сброшено")

