# Руководство по установке CAD Environment

## 🚀 Быстрая установка

### 1. Установка FreeCAD (обязательно!)

#### Windows
1. Скачайте FreeCAD с официального сайта: https://www.freecadweb.org/downloads.php
2. Установите FreeCAD в стандартную директорию
3. Добавьте FreeCAD в PATH (опционально)

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install freecad

# CentOS/RHEL
sudo yum install freecad
```

#### macOS
```bash
# Через Homebrew
brew install freecad

# Или скачайте с официального сайта
```

### 2. Установка Python зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/researchim-ai/cad-env.git
cd cad-env

# Установка всех зависимостей одной командой
pip install -r requirements.txt

# Установка в режиме разработки
pip install -e .
```

## ✅ Проверка установки

### Запуск тестов
```bash
python -m pytest tests/ -v
```
**Ожидаемый результат:** 16/16 тестов пройдено ✅

### Запуск примеров
```bash
# Генерация кода
python examples/code_generation_example.py

# Генерация датасетов
python examples/dataset_generation_example.py

# Веб-сервер
python examples/web_server_example.py --host 127.0.0.1 --port 8000
```

### Проверка веб-интерфейса
1. Откройте браузер
2. Перейдите на http://127.0.0.1:8000/web
3. Протестируйте создание документов и выполнение команд

## 🔧 Устранение проблем

### Проблема: "FreeCAD не найден"
**Решение:** Установите FreeCAD согласно инструкции выше

### Проблема: "ModuleNotFoundError"
**Решение:** Убедитесь, что все зависимости установлены:
```bash
pip install -r requirements.txt
```

### Проблема: "Permission denied" на Windows
**Решение:** Запустите командную строку от имени администратора

### Проблема: Медленная работа
**Решение:** 
- Используйте режим симуляции для разработки
- Установите FreeCAD для полной функциональности
- Настройте GPU для ML операций

## 📊 Что установлено

### Основные пакеты
- ✅ **FastAPI** - Веб-фреймворк
- ✅ **Transformers** - ML библиотека
- ✅ **PyTorch** - Глубокое обучение
- ✅ **OpenCV** - Обработка изображений
- ✅ **NumPy/SciPy** - Научные вычисления

### Инструменты разработки
- ✅ **Pytest** - Тестирование
- ✅ **Black** - Форматирование кода
- ✅ **Flake8** - Линтинг
- ✅ **MyPy** - Проверка типов

### ML/RL библиотеки
- ✅ **Gym** - RL окружения
- ✅ **Stable-Baselines3** - RL алгоритмы
- ✅ **Datasets** - Обработка данных
- ✅ **HuggingFace Hub** - Модели

## 🎯 Готовность к использованию

После успешной установки вы можете:

1. **Генерировать FreeCAD Python код** из естественного языка
2. **Создавать датасеты** для обучения LLM
3. **Обучать языковые модели** работе с CAD
4. **Использовать веб-интерфейс** для тестирования
5. **Интегрировать с RL агентами**

## 📞 Поддержка

Если возникли проблемы:
1. Проверьте, что FreeCAD установлен
2. Убедитесь, что все зависимости установлены
3. Запустите тесты для диагностики
4. Проверьте логи в консоли

**Система готова к использованию!** 🎉
# Technical drawing font (external cache)

After installing Python dependencies, install the pinned drawing font:

```bash
pip install -r requirements-drawings.txt
python -m cad_env.drawings.font_cache --install
```

Alternatively run `make install-drawings`. The font, editable source and licenses
are downloaded with SHA-256 verification to the user cache outside this checkout.
`CAD_ENV_CACHE` overrides the cache parent; otherwise `XDG_CACHE_HOME` or
`~/.cache` is used. Installation needs network access; rendering does not.
Re-running the installer reuses valid files and repairs corrupted cache entries.
There is no font substitution or implicit download during an experiment.

The supported CAD runtimes are FreeCAD and optional CadQuery.
