"""
Тесты для основного функционала
"""

import pytest
import tempfile
import os
from cad_env import CADEnvironment


class TestCADEnvironment:
    """Тесты для CADEnvironment"""
    
    def test_initialization(self):
        """Тест инициализации"""
        env = CADEnvironment()
        assert env is not None
        assert env.current_document is None
        assert env.history == []
    
    def test_create_document(self):
        """Тест создания документа"""
        env = CADEnvironment()
        doc_id = env.create_document("TestDoc")
        assert doc_id is not None
        assert env.current_document == doc_id
        assert len(env.history) == 1
    
    def test_save_document(self):
        """Тест сохранения документа"""
        env = CADEnvironment()
        env.create_document("TestDoc")
        
        with tempfile.NamedTemporaryFile(suffix='.fcstd', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            success = env.save_document(filepath)
            assert success
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_execute_command(self):
        """Тест выполнения команды"""
        env = CADEnvironment()
        env.create_document("TestDoc")
        
        result = env.execute_command("create_box", length=10, width=5, height=3)
        assert result is not None
        assert len(env.history) == 2  # create_document + execute_command
    
    def test_get_document_info(self):
        """Тест получения информации о документе"""
        env = CADEnvironment()
        env.create_document("TestDoc")
        
        info = env.get_document_info()
        assert info is not None
        assert "name" in info or "error" in info
    
    def test_reset(self):
        """Тест сброса окружения"""
        env = CADEnvironment()
        env.create_document("TestDoc")
        env.execute_command("create_box", length=1, width=1, height=1)
        
        env.reset()
        assert env.current_document is None
        assert env.history == []

