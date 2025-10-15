"""
Тесты для API
"""

import pytest
from fastapi.testclient import TestClient
from cad_env.api import CADAPI


class TestCADAPI:
    """Тесты для CADAPI"""
    
    def test_initialization(self):
        """Тест инициализации API"""
        api = CADAPI()
        assert api is not None
        assert api.app is not None
        assert api.env is not None
    
    def test_root_endpoint(self):
        """Тест корневого эндпоинта"""
        api = CADAPI()
        client = TestClient(api.app)
        
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self):
        """Тест проверки состояния"""
        api = CADAPI()
        client = TestClient(api.app)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_create_document(self):
        """Тест создания документа"""
        api = CADAPI()
        client = TestClient(api.app)
        
        response = client.post("/documents/create", json={"name": "TestDoc"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "document_id" in data
    
    def test_execute_command(self):
        """Тест выполнения команды"""
        api = CADAPI()
        client = TestClient(api.app)
        
        # Сначала создаем документ
        client.post("/documents/create", json={"name": "TestDoc"})
        
        # Затем выполняем команду
        response = client.post("/commands/execute", json={
            "command": "create_box",
            "parameters": {"length": 10, "width": 5, "height": 3}
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

