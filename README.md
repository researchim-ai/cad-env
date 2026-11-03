# CAD Environment: verified FreeCAD data for LLM / VLM agents

Окружение FreeCAD и фабрика проверенных обучающих траекторий для **SFT, preference learning и online GRPO**. Новый контур `cad_env.agent` / `cad_env.learning` строит настоящие параметрические детали и проверяет геометрию. Mock не используется для положительных labels.

Практический процесс для **двух GPU по 24 GB**, агента через API/tools и текущего визуального состояния: [TRAINING_PROCESS.md](docs/TRAINING_PROCESS.md). В нём закреплены зависимости, проверки данных и награды, SFT, оценка на отложенных задачах и условия перехода к multi-turn GRPO. Проверенный SFT-контур и экспериментальная RL-интеграция имеют отдельные статусы.

![CAD Environment](src/imgs/cad-env.png)

## Запуск

Нужны Python ≥3.10 и отдельно установленный FreeCAD console. Проверена сборка FreeCAD 1.1.0, revision 43087, OCCT 7.8.1. Закрепляйте точную сборку при сравнении результатов.

```bash
pip install -e '.[data,api,dev]'
export FREECAD_CMD=/usr/bin/freecadcmd-daily  # либо ваш FreeCADCmd/freecadcmd
python -m cad_env.learning.cli doctor
python -m cad_env.learning.cli generate \
  --output artifacts/dataset-v1 --seeds 100 --workers 4 \
  --preferences --holdout-families flange
```

Без рендера фабрика требует только стандартную библиотеку Python и внешний FreeCAD: `pip install -e .`, затем `generate --no-render`. Для изображений нужны NumPy/Pillow из extra `data`.

## Реализовано

- Четыре семейства: пластина, втулка, фланец, ступенчатый вал.
- Создание, изменение размера и исправление неверного размера.
- Ограниченный JSON-язык: Body, прямоугольный/круговой Sketch, Pad, Pocket, изменение constraint, finish.
- Реальное исполнение в отдельном процессе; save/reopen FCStd, таймаут, сохранение диагностики и checkpoints.
- Проверка валидности solid, OCCT symmetric difference, независимых аналитических объёма/габаритов, параметричности и сохранения истории.
- Пошаговый `reset/step`: неудачные действия не портят committed state.
- Четыре вида геометрии с Z-buffer и метаданными камеры для VLM.
- JSONL экспорты: whole-program SFT, next-action SFT, VLM, GRPO prompts и preference pairs.
- Группировка splits по происхождению/base design, дедупликация, optional holdout целого семейства.
- TRL reward callback, SFT/GRPO launchers, sampling модели с rejection-SFT и mining preference пар.
- Multi-turn rollout collector для восстановления после ошибок; локальный HTTP transport.

## Обучение

Тяжёлые ML-зависимости отделены от CAD worker. Используйте отдельное окружение с `requirements-learning.txt` и выбранный вами checkpoint:

```bash
python -m cad_env.learning.train --method sft \
  --model /path/to/code-model --dataset artifacts/dataset-v1 \
  --output artifacts/sft-run --max-steps 1000 --batch-size 2

python -m cad_env.learning.train --method grpo \
  --model /path/to/sft-checkpoint --dataset artifacts/dataset-v1 \
  --output artifacts/grpo-run --num-generations 4 --batch-size 4 \
  --cad-workers 2 --max-steps 100
```

SFT использует completion-only loss. Online GRPO получает свежие completions и оценивает каждую настоящим CAD worker. Предварительно посчитанные normalized advantages служат диагностикой, а не заменяют on-policy optimizer.

## Документация и проверки

- [Полное руководство: данные, reward, SFT/GRPO, sampling, API](docs/LEARNING_PIPELINE.md)
- [Исходное исследование архитектуры и датасетов](docs/FREECAD_AGENT_RESEARCH.md)

```bash
python -m pytest -q
uvicorn cad_env.agent.api:create_app --factory --host 127.0.0.1 --port 8000
```

Integration-тесты требуют настоящий FreeCAD и явно skipped при его отсутствии. Локальный HTTP-сервис выполняет native work последовательно в одном процессе; параллельная фабрика использует отдельные workers.

## Текущие границы

Это ограниченный, проверяемый baseline: четыре фиксированных структурных шаблона с вариациями размеров. VLM-данные содержат рендеры и текст размеров; GUI-действия и инженерные чертежи пока не записываются. GRPO launcher оценивает одну JSON-программу на completion; multi-turn GRPO с учётом log-probabilities ещё не реализован. Editability probe проверяет изменение хотя бы одного размера, а не все инженерные зависимости.

Старые `cad_env.core`, `llm_interface`, `code_generator` и `training` остаются legacy API. Их regex-шаблоны и `training_data/training_dataset.json` не используются новой фабрикой. Симуляция и exit code произвольного Python больше не считаются доказательством корректного CAD.

Apache 2.0.
