# CAD Environment Makefile

.PHONY: help install install-dev test lint format clean run-web run-examples

help: ## Показать справку
	@echo "Доступные команды:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Установить зависимости
	pip install -r requirements.txt

install-dev: ## Установить в режиме разработки
	pip install -e ".[dev]"

test: ## Запустить тесты
	pytest tests/ -v

test-coverage: ## Запустить тесты с покрытием
	pytest tests/ --cov=cad_env --cov-report=html --cov-report=term

lint: ## Проверить код линтерами
	flake8 cad_env/ examples/ tests/
	mypy cad_env/

format: ## Форматировать код
	black cad_env/ examples/ tests/
	isort cad_env/ examples/ tests/

clean: ## Очистить временные файлы
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

run-web: ## Запустить веб-сервер
	python examples/web_server_example.py

run-examples: ## Запустить примеры
	python examples/basic_usage.py
	python examples/llm_agent_example.py --mode batch

build: ## Собрать пакет
	python setup.py sdist bdist_wheel

check-build: ## Проверить сборку
	twine check dist/*

publish: ## Опубликовать пакет (только для разработчиков)
	twine upload dist/*

docs: ## Создать документацию
	@echo "Документация доступна в docs/README.md"
	@echo "Для веб-документации используйте: sphinx-build docs/ docs/_build/"

all: clean install-dev test lint format ## Выполнить все проверки

