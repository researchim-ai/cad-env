"""
Веб-сервер для CAD Environment
"""

import uvicorn
import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .cad_api import CADAPI

logger = logging.getLogger(__name__)


class WebServer:
    """
    Веб-сервер для CAD Environment
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Инициализация веб-сервера
        
        Args:
            host: Хост для сервера
            port: Порт для сервера
        """
        self.host = host
        self.port = port
        self.api = CADAPI()
        self.app = self.api.get_app()
        self._setup_web_interface()
    
    def _setup_web_interface(self):
        """Настройка веб-интерфейса"""
        
        @self.app.get("/web", response_class=HTMLResponse)
        async def web_interface():
            """Веб-интерфейс для CAD Environment"""
            html_content = """
            <!DOCTYPE html>
            <html lang="ru">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>CAD Environment</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }
                    .container {
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    h1 {
                        color: #333;
                        text-align: center;
                    }
                    .section {
                        margin: 20px 0;
                        padding: 15px;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }
                    .button {
                        background-color: #007bff;
                        color: white;
                        padding: 10px 20px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        margin: 5px;
                    }
                    .button:hover {
                        background-color: #0056b3;
                    }
                    .input {
                        width: 100%;
                        padding: 8px;
                        margin: 5px 0;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }
                    .result {
                        background-color: #f8f9fa;
                        padding: 10px;
                        border-radius: 4px;
                        margin: 10px 0;
                        white-space: pre-wrap;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>CAD Environment - Веб-интерфейс</h1>
                    
                    <div class="section">
                        <h3>Управление документами</h3>
                        <input type="text" id="docName" class="input" placeholder="Имя документа" value="NewDocument">
                        <button class="button" onclick="createDocument()">Создать документ</button>
                        <button class="button" onclick="getDocumentInfo()">Информация о документе</button>
                    </div>
                    
                    <div class="section">
                        <h3>Выполнение команд</h3>
                        <input type="text" id="command" class="input" placeholder="CAD команда">
                        <button class="button" onclick="executeCommand()">Выполнить команду</button>
                    </div>
                    
                    <div class="section">
                        <h3>Результаты</h3>
                        <div id="result" class="result">Готов к работе...</div>
                    </div>
                    
                    <div class="section">
                        <h3>История операций</h3>
                        <button class="button" onclick="getHistory()">Показать историю</button>
                        <button class="button" onclick="resetEnvironment()">Сбросить окружение</button>
                    </div>
                </div>
                
                <script>
                    async function apiCall(endpoint, method = 'GET', data = null) {
                        const options = {
                            method: method,
                            headers: {
                                'Content-Type': 'application/json',
                            }
                        };
                        
                        if (data) {
                            options.body = JSON.stringify(data);
                        }
                        
                        try {
                            const response = await fetch(endpoint, options);
                            const result = await response.json();
                            return result;
                        } catch (error) {
                            return { error: error.message };
                        }
                    }
                    
                    async function createDocument() {
                        const name = document.getElementById('docName').value;
                        const result = await apiCall('/documents/create', 'POST', { name });
                        document.getElementById('result').textContent = JSON.stringify(result, null, 2);
                    }
                    
                    async function getDocumentInfo() {
                        const result = await apiCall('/documents/current/info');
                        document.getElementById('result').textContent = JSON.stringify(result, null, 2);
                    }
                    
                    async function executeCommand() {
                        const command = document.getElementById('command').value;
                        const result = await apiCall('/commands/execute', 'POST', { 
                            command: command, 
                            parameters: {} 
                        });
                        document.getElementById('result').textContent = JSON.stringify(result, null, 2);
                    }
                    
                    async function getHistory() {
                        const result = await apiCall('/history');
                        document.getElementById('result').textContent = JSON.stringify(result, null, 2);
                    }
                    
                    async function resetEnvironment() {
                        const result = await apiCall('/reset', 'POST');
                        document.getElementById('result').textContent = JSON.stringify(result, null, 2);
                    }
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        """
        Запустить веб-сервер
        
        Args:
            host: Хост (по умолчанию из инициализации)
            port: Порт (по умолчанию из инициализации)
        """
        host = host or self.host
        port = port or self.port
        
        logger.info(f"Запуск веб-сервера на {host}:{port}")
        logger.info(f"Веб-интерфейс доступен по адресу: http://{host}:{port}/web")
        logger.info(f"API документация: http://{host}:{port}/docs")
        
        uvicorn.run(self.app, host=host, port=port)

