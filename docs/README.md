# CAD Environment - Документация

## Обзор

CAD Environment - это окружение для обучения языковых моделей (LLM) и агентов с подкрепляющим обучением (RL) работе с CAD системами. Проект предоставляет удобный интерфейс для автоматизации CAD операций через естественный язык.

## Основные возможности

- 🔧 **Интеграция с FreeCAD** - Полная поддержка FreeCAD API
- 🤖 **LLM интерфейс** - Обработка естественного языка
- 🌐 **REST API** - Веб-интерфейс и API для интеграции
- 📊 **История операций** - Отслеживание всех действий
- 🔄 **Режим симуляции** - Работа без установленного FreeCAD

## Быстрый старт

### Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd cad-env

# Установка зависимостей
pip install -r requirements.txt

# Установка в режиме разработки
pip install -e .
```

### Базовое использование

```python
from cad_env import CADEnvironment, LLMInterface

# Создание окружения
env = CADEnvironment()

# Создание документа
doc_id = env.create_document("MyDocument")

# Выполнение команды
result = env.execute_command("create_box", length=10, width=5, height=3)

# LLM интерфейс
llm = LLMInterface(env)
result = llm.process_natural_language("Создай коробку размером 10x5x3")
```

### Запуск веб-сервера

```bash
python examples/web_server_example.py --host 0.0.0.0 --port 8000
```

Откройте браузер и перейдите по адресу: `http://localhost:8000/web`

## Архитектура

### Основные компоненты

1. **CADEnvironment** - Основное окружение
2. **FreeCADWrapper** - Обертка для FreeCAD API
3. **LLMInterface** - Интерфейс для LLM агентов
4. **CADAPI** - REST API
5. **WebServer** - Веб-интерфейс

### Структура проекта

```
cad-env/
├── cad_env/                 # Основной пакет
│   ├── core/               # Ядро системы
│   ├── api/                # API и веб-сервер
│   └── llm_interface/      # LLM интерфейс
├── examples/               # Примеры использования
├── docs/                   # Документация
└── tests/                  # Тесты
```

## API Reference

### CADEnvironment

Основной класс для работы с CAD окружением.

#### Методы

- `create_document(name: str) -> str` - Создать документ
- `save_document(filepath: str) -> bool` - Сохранить документ
- `load_document(filepath: str) -> str` - Загрузить документ
- `execute_command(command: str, **kwargs) -> Dict` - Выполнить команду
- `get_document_info() -> Dict` - Получить информацию о документе
- `get_history() -> List[Dict]` - Получить историю операций
- `reset()` - Сбросить окружение

### LLMInterface

Интерфейс для работы с LLM агентами.

#### Методы

- `process_natural_language(text: str) -> Dict` - Обработать естественный язык
- `get_available_commands() -> List[Dict]` - Получить доступные команды
- `get_context_info() -> Dict` - Получить контекстную информацию
- `execute_structured_command(command: Dict) -> Dict` - Выполнить структурированную команду

## Поддерживаемые команды

### Создание объектов

- **Коробка**: "Создай коробку размером 10x5x3"
- **Цилиндр**: "Создай цилиндр радиусом 5 и высотой 10"
- **Сфера**: "Создай сферу радиусом 3"

### Трансформации

- **Поворот**: "Поверни на 90 градусов вокруг оси Z"
- **Перемещение**: "Перемести на 10 единиц по оси X"
- **Выдавливание**: "Выдави на 5 единиц"

## Примеры использования

### Базовые операции

```python
from cad_env import CADEnvironment

env = CADEnvironment()

# Создание документа
doc_id = env.create_document("TestDocument")

# Создание коробки
result = env.execute_command("create_box", length=10, width=5, height=3)

# Сохранение
env.save_document("output.fcstd")
```

### LLM агент

```python
from cad_env import LLMInterface

llm = LLMInterface()

# Обработка естественного языка
result = llm.process_natural_language("Создай коробку размером 10x5x3")
print(result)
```

### Веб-интерфейс

```python
from cad_env.api import WebServer

server = WebServer(host="0.0.0.0", port=8000)
server.run()
```

## Конфигурация

### Переменные окружения

- `FREECAD_PATH` - Путь к FreeCAD (опционально)
- `CAD_ENV_DEBUG` - Режим отладки
- `CAD_ENV_LOG_LEVEL` - Уровень логирования

### Конфигурационный файл

```python
config = {
    "freecad_path": "/path/to/freecad",
    "debug": True,
    "log_level": "INFO"
}

env = CADEnvironment(config)
```

## Разработка

### Установка для разработки

```bash
git clone <repository-url>
cd cad-env
pip install -e ".[dev]"
```

### Запуск тестов

```bash
pytest tests/
```

### Линтинг

```bash
black cad_env/
flake8 cad_env/
mypy cad_env/
```

## Лицензия

Apache 2.0

## Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## Поддержка

- GitHub Issues: [ссылка на issues]
- Документация: [ссылка на docs]
- Примеры: [ссылка на examples]

