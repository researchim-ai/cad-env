"""
Пример запуска веб-сервера для CAD Environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad_env.api import WebServer
import argparse


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="CAD Environment Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Хост для сервера")
    parser.add_argument("--port", type=int, default=8000, help="Порт для сервера")
    parser.add_argument("--debug", action="store_true", help="Режим отладки")
    
    args = parser.parse_args()
    
    print("=== CAD Environment Web Server ===")
    print(f"Запуск сервера на {args.host}:{args.port}")
    print(f"Веб-интерфейс: http://{args.host}:{args.port}/web")
    print(f"API документация: http://{args.host}:{args.port}/docs")
    print("Нажмите Ctrl+C для остановки")
    print("-" * 50)
    
    try:
        # Создание и запуск веб-сервера
        server = WebServer(host=args.host, port=args.port)
        server.run()
    except KeyboardInterrupt:
        print("\nСервер остановлен")
    except Exception as e:
        print(f"Ошибка запуска сервера: {e}")


if __name__ == "__main__":
    main()

