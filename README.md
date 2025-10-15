# CAD Environment для обучения LLM и RL агентов

Этот проект предоставляет окружение для обучения языковых моделей и агентов с подкрепляющим обучением работе с CAD системами.

## 🚀 Особенности

- 🔧 **Генерация FreeCAD Python кода** - LLM генерирует прямой Python код для FreeCAD
- 🤖 **Обучение LLM** - Полный пайплайн для обучения языковых моделей
- 📊 **Генерация датасетов** - Автоматическое создание обучающих данных
- 🌐 **REST API** - Веб-интерфейс и API для интеграции
- 🔄 **Режим симуляции** - Работа без установленного FreeCAD
- 🧪 **Готовность к RL** - Архитектура готова для агентов с подкрепляющим обучением
- 📈 **Валидация кода** - Проверка и выполнение сгенерированного кода

## 📦 Установка

### Windows

```bash
# 1. Установка FreeCAD (обязательно!)
# Скачайте и установите FreeCAD с https://www.freecadweb.org/downloads.php

# 2. Клонирование репозитория
git clone <repository-url>
cd cad-env

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Установка в режиме разработки
pip install -e .
```

### Linux/macOS

```bash
# 1. Установка FreeCAD
# Ubuntu/Debian: sudo apt install freecad
# macOS: brew install freecad

# 2. Клонирование и установка
git clone <repository-url>
cd cad-env
pip install -r requirements.txt
pip install -e .
```

## 🎯 Быстрый старт

### Генерация FreeCAD Python кода

```python
from cad_env.code_generator import FreeCADCodeGenerator

# Генератор кода
generator = FreeCADCodeGenerator()

# Генерация кода из естественного языка
result = generator.generate_from_natural_language("Создай коробку размером 10x5x3")
print(result["generated_code"])

# Генерация сложного скрипта
scenario = [
    "Создай коробку размером 20x15x10",
    "Создай цилиндр радиусом 5 и высотой 15", 
    "Вычти цилиндр из коробки"
]
full_script = generator.generate_complex_script(scenario)
```

### Обучение LLM

```python
from cad_env.training import LLMTrainer, TrainingDataManager

# Генерация датасета
data_manager = TrainingDataManager()
dataset = data_manager.generate_training_dataset(num_samples=1000)

# Обучение модели
trainer = LLMTrainer("microsoft/DialoGPT-medium")
trainer.setup_model()
trainer.train(dataset)

# Генерация кода обученной моделью
code = trainer.generate_code("Создай деталь для механизма")
```

### Запуск веб-сервера

```bash
python examples/web_server_example.py --host 0.0.0.0 --port 8000
```

Откройте браузер: `http://localhost:8000/web`

### Интерактивный LLM агент

```bash
python examples/llm_agent_example.py --mode interactive
```

## 📚 Примеры

- [Генерация кода](examples/code_generation_example.py)
- [Генерация датасетов](examples/dataset_generation_example.py)
- [Обучение LLM](examples/llm_training_example.py)
- [Веб-сервер](examples/web_server_example.py)

## 🏗️ Архитектура

```
cad-env/
├── cad_env/                 # Основной пакет
│   ├── core/               # Ядро системы
│   ├── code_generator/     # Генератор FreeCAD кода
│   ├── training/           # Обучение LLM
│   ├── api/                # API и веб-сервер
│   └── llm_interface/      # LLM интерфейс
├── examples/               # Примеры использования
├── docs/                   # Документация
└── tests/                  # Тесты
```

## 🛠️ Разработка

```bash
# Установка для разработки
make install-dev

# Запуск тестов
make test

# Проверка кода
make lint

# Форматирование
make format

# Запуск веб-сервера
make run-web
```

## 📖 Документация

Полная документация доступна в [docs/README.md](docs/README.md)

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

Apache 2.0