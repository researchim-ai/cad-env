"""
Обертка для FreeCAD API
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class FreeCADWrapper:
    """
    Обертка для работы с FreeCAD API
    """
    
    def __init__(self):
        """
        Инициализация FreeCAD
        """
        self.freecad = None
        self.documents = {}
        self.current_doc = None
        self._init_freecad()
    
    def _init_freecad(self):
        """
        Инициализация FreeCAD
        """
        try:
            # Попытка импорта FreeCAD
            import FreeCAD
            self.freecad = FreeCAD
            logger.info("FreeCAD успешно инициализирован")
        except ImportError:
            logger.warning("FreeCAD не найден. Используется режим симуляции.")
            self.freecad = None
    
    def create_document(self, name: str) -> str:
        """
        Создать новый документ
        
        Args:
            name: Имя документа
            
        Returns:
            ID документа
        """
        if self.freecad:
            doc = self.freecad.newDocument(name)
            doc_id = str(id(doc))
            self.documents[doc_id] = doc
            self.current_doc = doc_id
            return doc_id
        else:
            # Режим симуляции
            doc_id = f"sim_{len(self.documents)}"
            self.documents[doc_id] = {"name": name, "objects": []}
            self.current_doc = doc_id
            return doc_id
    
    def save_document(self, doc_id: str, filepath: str) -> bool:
        """
        Сохранить документ
        
        Args:
            doc_id: ID документа
            filepath: Путь для сохранения
            
        Returns:
            True если успешно
        """
        if doc_id not in self.documents:
            return False
        
        if self.freecad:
            doc = self.documents[doc_id]
            doc.saveAs(filepath)
            return True
        else:
            # Режим симуляции - создаем пустой файл
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# FreeCAD simulation document: {self.documents[doc_id]['name']}\n")
            return True
    
    def load_document(self, filepath: str) -> str:
        """
        Загрузить документ
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            ID документа
        """
        if self.freecad:
            doc = self.freecad.open(filepath)
            doc_id = str(id(doc))
            self.documents[doc_id] = doc
            self.current_doc = doc_id
            return doc_id
        else:
            # Режим симуляции
            doc_id = f"sim_{len(self.documents)}"
            self.documents[doc_id] = {
                "name": Path(filepath).stem,
                "filepath": filepath,
                "objects": []
            }
            self.current_doc = doc_id
            return doc_id
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """
        Получить информацию о документе
        
        Args:
            doc_id: ID документа
            
        Returns:
            Информация о документе
        """
        if doc_id not in self.documents:
            return {"error": "Документ не найден"}
        
        if self.freecad:
            doc = self.documents[doc_id]
            return {
                "name": doc.Name,
                "objects": [obj.Name for obj in doc.Objects],
                "count": len(doc.Objects)
            }
        else:
            doc = self.documents[doc_id]
            return {
                "name": doc["name"],
                "objects": doc["objects"],
                "count": len(doc["objects"]),
                "simulation": True
            }
    
    def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Выполнить команду FreeCAD
        
        Args:
            command: Команда
            **kwargs: Параметры
            
        Returns:
            Результат выполнения
        """
        if not self.current_doc:
            return {"error": "Нет активного документа"}
        
        if self.freecad:
            try:
                doc = self.documents[self.current_doc]
                primitives = {
                    'create_box': ('Part::Box', {'length': 'Length', 'width': 'Width', 'height': 'Height'}),
                    'create_cylinder': ('Part::Cylinder', {'radius': 'Radius', 'height': 'Height'}),
                    'create_sphere': ('Part::Sphere', {'radius': 'Radius'}),
                }
                if command not in primitives:
                    return {'success': False, 'error': 'Unsupported command; use cad_env.agent for CAD tools'}
                obj_type, parameters = primitives[command]
                import math
                values = {}
                for name, prop in parameters.items():
                    value = kwargs.get(name, 10)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 10000:
                        raise ValueError('Invalid dimension: ' + name)
                    values[prop] = value
                if set(kwargs) - set(parameters):
                    raise ValueError('Unexpected command parameters')
                obj = doc.addObject(obj_type, 'Primitive')
                for prop, value in values.items():
                    setattr(obj, prop, value)
                doc.recompute()
                return {'success': not obj.Shape.isNull() and obj.Shape.isValid(), 'result': obj.Name, 'simulation': False}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            # Режим симуляции
            return {
                "success": False,
                "validated": False,
                "result": f"Симуляция команды: {command}",
                "simulation": True
            }
    
    def reset(self):
        """
        Сбросить состояние
        """
        self.documents = {}
        self.current_doc = None
        if self.freecad:
            # Закрыть все документы
            for doc in self.freecad.listDocuments().values():
                self.freecad.closeDocument(doc.Name)

