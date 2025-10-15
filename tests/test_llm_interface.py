"""
Тесты для LLM интерфейса
"""

import pytest
from cad_env import LLMInterface


class TestLLMInterface:
    """Тесты для LLMInterface"""
    
    def test_initialization(self):
        """Тест инициализации"""
        llm = LLMInterface()
        assert llm is not None
        assert llm.env is not None
        assert llm.command_parser is not None
        assert llm.nlp is not None
    
    def test_process_natural_language(self):
        """Тест обработки естественного языка"""
        llm = LLMInterface()
        
        # Тест создания коробки
        result = llm.process_natural_language("Создай коробку размером 10x5x3")
        assert result is not None
        assert "success" in result
    
    def test_get_available_commands(self):
        """Тест получения доступных команд"""
        llm = LLMInterface()
        commands = llm.get_available_commands()
        
        assert isinstance(commands, list)
        assert len(commands) > 0
        
        # Проверяем структуру команд
        for cmd in commands:
            assert "command" in cmd
            assert "description" in cmd
            assert "parameters" in cmd
            assert "example" in cmd
    
    def test_get_context_info(self):
        """Тест получения контекстной информации"""
        llm = LLMInterface()
        context = llm.get_context_info()
        
        assert isinstance(context, dict)
        assert "environment_status" in context
        assert "available_commands" in context
    
    def test_execute_structured_command(self):
        """Тест выполнения структурированной команды"""
        llm = LLMInterface()
        
        command = {
            "action": "create_box",
            "parameters": {"length": 10, "width": 5, "height": 3}
        }
        
        result = llm.execute_structured_command(command)
        assert result is not None
        assert "success" in result

