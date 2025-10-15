"""
REST API для CAD Environment
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from ..core import CADEnvironment

logger = logging.getLogger(__name__)


class CommandRequest(BaseModel):
    """Запрос на выполнение команды"""
    command: str
    parameters: Optional[Dict[str, Any]] = {}


class DocumentRequest(BaseModel):
    """Запрос для работы с документами"""
    name: Optional[str] = None
    filepath: Optional[str] = None


class CADAPI:
    """
    REST API для взаимодействия с CAD Environment
    """
    
    def __init__(self, environment: Optional[CADEnvironment] = None):
        """
        Инициализация API
        
        Args:
            environment: CAD окружение
        """
        self.env = environment or CADEnvironment()
        self.app = FastAPI(
            title="CAD Environment API",
            description="API для обучения LLM и RL агентов работе с CAD",
            version="0.1.0"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        """Настройка маршрутов API"""
        
        @self.app.get("/")
        async def root():
            """Корневой эндпоинт"""
            return {
                "message": "CAD Environment API",
                "version": "0.1.0",
                "status": "active"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Проверка состояния"""
            return {"status": "healthy", "environment": "ready"}
        
        @self.app.post("/documents/create")
        async def create_document(request: DocumentRequest):
            """Создать новый документ"""
            try:
                name = request.name or "NewDocument"
                doc_id = self.env.create_document(name)
                return {
                    "success": True,
                    "document_id": doc_id,
                    "message": f"Документ '{name}' создан"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/load")
        async def load_document(request: DocumentRequest):
            """Загрузить документ"""
            try:
                if not request.filepath:
                    raise HTTPException(status_code=400, detail="Не указан путь к файлу")
                
                doc_id = self.env.load_document(request.filepath)
                return {
                    "success": True,
                    "document_id": doc_id,
                    "message": f"Документ загружен из {request.filepath}"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/save")
        async def save_document(request: DocumentRequest):
            """Сохранить документ"""
            try:
                if not request.filepath:
                    raise HTTPException(status_code=400, detail="Не указан путь для сохранения")
                
                success = self.env.save_document(request.filepath)
                if success:
                    return {
                        "success": True,
                        "message": f"Документ сохранен в {request.filepath}"
                    }
                else:
                    raise HTTPException(status_code=500, detail="Ошибка сохранения документа")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/documents/current/info")
        async def get_current_document_info():
            """Получить информацию о текущем документе"""
            try:
                info = self.env.get_document_info()
                return {"success": True, "document_info": info}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/commands/execute")
        async def execute_command(request: CommandRequest):
            """Выполнить CAD команду"""
            try:
                result = self.env.execute_command(
                    request.command, 
                    **request.parameters
                )
                return {
                    "success": True,
                    "command": request.command,
                    "result": result
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/history")
        async def get_history():
            """Получить историю операций"""
            try:
                history = self.env.get_history()
                return {"success": True, "history": history}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/reset")
        async def reset_environment():
            """Сбросить окружение"""
            try:
                self.env.reset()
                return {"success": True, "message": "Окружение сброшено"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_app(self) -> FastAPI:
        """
        Получить FastAPI приложение
        
        Returns:
            FastAPI приложение
        """
        return self.app

