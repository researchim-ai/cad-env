# Проверяемые данные FreeCAD для SFT, VLM и GRPO

Реализован отдельный контур `cad_env.agent` / `cad_env.learning`. Он использует настоящий FreeCAD console, создаёт native Sketcher/PartDesign объекты и не подменяет исполнение симуляцией. Старые regex-генераторы оставлены как legacy; новый pipeline их не использует.

## Быстрый запуск

```bash
pip install -e '.[data,api,dev]'
export FREECAD_CMD=/usr/bin/freecadcmd-daily
python -m cad_env.learning.cli doctor
python -m cad_env.learning.cli generate \
  --output artifacts/dataset-v1 --seeds 100 --workers 4 \
  --preferences --holdout-families flange
python -m cad_env.learning.cli audit --dataset artifacts/dataset-v1
```

`--seeds 100` означает до 100 seeds × 4 семейства × 3 режима = 1 200 эпизодов. Дубликаты одинаковых задач удаляются. `--no-render` отключает PNG; `--families plate bushing` и `--modes create edit` ограничивают состав. Выходная директория должна быть новой/пустой. Генерация не скачивает модели и не вызывает платные API.

Автоматически обнаруживаются `FreeCADCmd`, `freecadcmd`, `freecadcmd-daily`; `FREECAD_CMD` или `--freecad` задают конкретный бинарник. `doctor` выполняет полный реальный build/save/reopen/verify, а не проверяет только PATH.

## Что строится

Семейства: пластина с центральным отверстием, втулка, фланец с центральным и четырьмя крепёжными отверстиями, ступенчатый вал. Геометрические параметры выбираются из допустимых диапазонов. Эскизы полностью ограничены; размерные constraints именованы. Плоскость отверстий связана выражением с высотой предыдущего Pad, без ссылок `FaceN`.

Режимы:

- `create`: пустой документ → полная деталь.
- `edit`: исходная деталь с другим размером → изменить только заданное свойство.
- `repair`: исправление намеренно неверного размера в существующей валидной детали. Это пока **dimension repair**, а не корпус CAD crashes/solver failures.

Для более широкого error-repair есть пошаговое окружение и сборщик реальных model rollouts, включая невалидные действия и восстановление после них.

## Артефакты

```text
dataset/
  summary.json                 # runtime, yield, counts, export hashes
  manifest.jsonl               # все принятые и отклонённые задачи
  source_snapshot/             # исходники генератора/worker и их hashes
  episodes/<task_id>/
    task.json                  # PRIVATE: reference/oracle programs
    trajectory.jsonl           # action + observation before/after
    oracle/
      input.FCStd              # у edit/repair
      output.FCStd
      checkpoints/*.FCStd
      result.json              # numeric verdict, runtime, artifact hash
      mesh.json
      initial_mesh.json        # у edit/repair
      stdout.log / stderr.log
    images/view-*.png
    images/view-cameras.json
    negative/                  # при --preferences
  exports/
    train.sft.jsonl            # prompt/completion: целая программа
    train.steps.jsonl          # prompt/completion: одно следующее действие
    train.grpo.jsonl           # prompt/task_id; без решения
    train.vlm.jsonl            # prompt/completion + 4 изображения цели
    train.preference.jsonl     # prompt/chosen/rejected
    validation.*.jsonl
    test.*.jsonl
```

VLM-кадры — ортографические **рендеры геометрии**, не скриншоты приложения. Используется CPU Z-buffer с supersampling; он правильно перекрывает грани и показывает отверстия. Метаданные камер содержат центр, базис, span и размер кадра. В тексте сохраняются размеры, поскольку изображение само по себе не задаёт абсолютный масштаб. Инженерные чертежи с размерными линиями и GUI trajectories пока не генерируются.

`task.json`, `request.json`, worker source и validator — trusted side. Не монтировать их в файловую среду оцениваемого агента. Модель получает только `prompt` из экспорта. Для online GRPO callback разрешает `task_id` в приватный task manifest. При редактировании исходная программа/состояние видимы агенту намеренно; target программа скрыта.

Splits назначаются по hash семейства и базовых неизменяемых размеров. Create/edit/repair, виды, варианты отверстия/высоты ступени и совпадающие геометрии одного base design остаются в одном split. Это проверка параметрического обобщения; `--holdout-families flange` дополнительно исключает всё семейство flange из train. Небольшой pilot может иметь несбалансированные или пустые splits — не копировать test в train для исправления этого.

## Язык действий

Агент выдаёт JSON, без Python `eval`/`exec`. На уровне full-program completion разрешён массив или `{"actions": [...]}`; в конце обязателен `finish`. В sequential env вызывается один action object за шаг.

```json
[
  {"op":"create_body","name":"Body"},
  {"op":"sketch_rectangle","name":"BaseSketch","body":"Body","length":50,"width":30},
  {"op":"pad","name":"BasePad","body":"Body","profile":"BaseSketch","length":6},
  {"op":"sketch_circles","name":"HoleSketch","body":"Body","support":"BasePad","circles":[{"x":25,"y":15,"radius":4}]},
  {"op":"pocket","name":"HolePocket","body":"Body","profile":"HoleSketch","length":6},
  {"op":"finish"}
]
```

Редактирование: `{"op":"set_parameter","target":"HoleSketch","parameter":"diameter_0","value":10}`. Доступны `length`/`width` для rectangle sketch, `diameter_N` для circle sketch и `length` для Pad/Pocket.

Имена/поля/конечность чисел проверяются до запуска; до 64 действий в программе, 16 окружностей в эскизе, размеры до 10 000 мм. FreeCAD запускается в отдельной process group с таймаутом, профилем и cache внутри эпизода. Это ограниченный action interpreter, **не универсальная песочница для произвольного Python**. Raw Python из модели новый pipeline не исполняет.

## Как определяется правильность

Проверки выполняются после сохранения и повторного открытия FCStd:

1. Успешное исполнение, recompute, одно валидное тело с положительным объёмом.
2. Геометрия в исходных координатах/миллиметрах: OCCT boolean differences между candidate и reference. Нет нормализации масштаба или выравнивания, скрывающего неверный размер/placement.
3. Независимые аналитические объём и bounding box по спецификации семейства — защита от одинаковой ошибки генератора и reference.
4. Полностью ограниченные эскизы; изменение размерного constraint реально меняет объём валидного тела. Затем исходный размер восстанавливается.
5. Для edit/repair сохраняются имена и типы всех исходных объектов.

`task_success` требует все обязательные условия и relative symmetric difference ≤ 1e-5. Это threshold для текущего пилота, а не универсальный инженерный допуск.

Reward в диапазоне [0,1]: при нарушении shape/parametric/history gates — 0; иначе `0.8 × geometry_score + 0.2 × exact_spec_match`. `geometry_score = 1 − relative_symmetric_difference`. Дефектное действие/JSON получает 0. Ошибки evaluator и crash инфраструктуры вызывают исключение, чтобы тренировку можно было исправить; timeout кандидата считается неуспехом в рамках бюджета. Все подробности сохраняются в rollout.

Не считать этот scorer доказательством полной инженерной эквивалентности. Editability probe проверяет хотя бы один размер, а не все зависимости; history gate не доказывает неизменность каждой связи. Для добавления семейств нужны predicates, oracle, независимые отрицательные примеры и тесты. Внешний benchmark и сложные многообъектные/сборочные задачи остаются отдельной работой.

## SFT: сначала получить устойчивую политику

В отдельном окружении установить `requirements-learning.txt` (TRL 0.24 API). Не смешивать с несовместимыми глобальными vLLM/scikit-learn; vLLM для этих запусков не нужен.

```bash
python -m venv .venv-train
.venv-train/bin/pip install -r requirements-learning.txt
.venv-train/bin/pip install -e '.[data]'
.venv-train/bin/python -m cad_env.learning.train \
  --method sft --model /path/to/chosen-code-model \
  --dataset artifacts/dataset-v1 --output artifacts/sft-run \
  --max-steps 1000 --batch-size 2 --lora-rank 16
```

Для next-action SFT задать `--export artifacts/dataset-v1/exports/train.steps.jsonl`. Для VLM — `train.vlm.jsonl` и совместимый multimodal checkpoint. Loader превращает относительные изображения в декодируемые HF Image features; VLM использует `max_length=None`, чтобы не обрезать image tokens. Prompt-completion формат включает completion-only loss; проверять длины для выбранной модели.

На раннем этапе смешивать create и dimension edits; затем добавлять реальные ошибки и recovery-траектории. Успешные procedural samples учат пользоваться инструментами, но не доказывают обобщение на неизвестные детали.

## GRPO: prompts + online CAD reward

GRPO не обучается на заготовленном массиве «правильных rewards». Policy генерирует несколько новых решений **одного и того же prompt**, каждое исполняется в FreeCAD, а TRL считает group-relative advantages и policy loss.

```bash
.venv-train/bin/python -m cad_env.learning.train \
  --method grpo --model artifacts/sft-run \
  --dataset artifacts/dataset-v1 --output artifacts/grpo-run \
  --num-generations 4 --batch-size 4 --max-steps 100 \
  --max-completion-length 2048 --cad-workers 2 --lora-rank 16
```

Если SFT сохранён как LoRA adapter, использовать поддерживаемую PEFT/Transformers загрузку adapter или заранее merged checkpoint. Проверять выбранный checkpoint через короткий sampling до дорогого запуска. Для distributed GRPO согласовать effective batch size с num_generations согласно TRL; CLI использует консервативную проверку per-device batch.

В этой версии online GRPO — **single-completion JSON program**. Окружение исполняет программу пошагово, но модель получает terminal reward за всю completion. Multi-turn collection реализован отдельно; специализированный multi-turn GRPO rollout_func/log-prob accounting ещё не реализован. Нельзя подставлять offline normalized advantages вместо on-policy GRPO loss.

Практические параметры подбирать по pilot: начать с G=4, ограничить длину, измерить correct/incorrect/invalid/crash/timeouts и долю групп с нулевой дисперсией. Если все ответы неверны — вернуться к SFT/более простым задачам; если все верны — усложнить задачи. Одинаковый reward всей группы не даёт относительного сигнала. `beta=0` в текущем launcher — явный начальный вариант без KL; для длительных запусков обосновать regularization и отдельно оценить стабильность.

## Сбор данных из настоящей policy

```bash
.venv-train/bin/python -m cad_env.learning.sample \
  --model /path/to/checkpoint --dataset artifacts/dataset-v1 \
  --output artifacts/policy-samples --limit 20 --num-generations 4
```

Сохраняются все кандидаты и CAD verdict, включая неуспехи; из **train** строятся `rejection.sft.jsonl` и `mined.preference.jsonl`. Validation/test используются только для оценки. Искусственные негативы фабрики полезны для проверки verifier и preference baseline; это не on-policy GRPO rollout.

Для готовых внешних completions:

```bash
python -m cad_env.learning.cli score --dataset artifacts/dataset-v1 \
  --completions candidates.jsonl --output artifacts/scored
```

Каждая строка входа: `{"task_id":"plate-000001-create","completion":"[...JSON actions...]"}`. `groups.json` содержит среднее, std, normalized advantages и zero-variance diagnostics. Это диагностика, не trainer.

## Пошаговые агенты и recovery

```python
from cad_env.agent import make_task
from cad_env.agent.environment import CADAgentEnv

env = CADAgentEnv('artifacts/episodes')
task = make_task('plate', seed=3, mode='edit')
observation, info = env.reset(task)
for action in task['oracle_actions']:  # заменить реальным вызовом policy
    observation, reward, terminated, truncated, info = env.step(action)
```

Отклонённое действие записывается вместе с ошибкой, но не меняет committed state. Каждый шаг replay выполняется в новом процессе: надёжно и изолированно, но дороже persistent worker. `max_steps` ограничивает также ошибочные попытки. `finish` завершает эпизод и вызывает проверку. Внешнему policy не передавать `task['oracle_actions']`.

`cad_env.learning.rollouts.collect_group(task, policy, output_dir, num_generations=4)` принимает callable `policy(messages) -> JSON action text`, собирает реальные multi-turn episodes и сохраняет успешные решения как next-action SFT. Неудачные попытки остаются в контексте для обучения recovery. Provider и платные API не выбираются автоматически.

Для HTTP:

```bash
uvicorn cad_env.agent.api:create_app --factory --host 127.0.0.1 --port 8000
```

`POST /sessions` с family/seed/mode → session_id и observation; `POST /sessions/{id}/step` с `{"action": {...}}`; `DELETE /sessions/{id}`. Сессии имеют отдельные состояния и сериализацию шагов. В этой версии native work блокирует HTTP-процесс; для параллельных workloads использовать CLI worker pool; это локальный исследовательский сервис без аутентификации, не публичный multi-tenant endpoint.

## Проверка

```bash
python -m pytest -q
```

Integration-тесты с реальным FreeCAD проверяют 4 семейства × 3 режима, неверные размеры, общую ошибку reference/генератора, восстановление после невалидного действия, лимит шагов, timeout, split lineage, rewards и экспорты. При отсутствии console такие integration-тесты явно skipped, а не проходят в mock.

Дальше расширять структурную вариативность: несколько отверстий/карманов, constraints с зависимостями, revolve/fillet/chamfer, затем чертежи и GUI. До этого использовать этот корпус как воспроизводимый ограниченный baseline, а не как полный датасет владения FreeCAD.

## Проверено в этой рабочей среде, 3 ноября 2026

- FreeCAD 1.1.0 revision 43087, OCCT 7.8.1, native worker Python 3.12.3.
- Полный набор: **51 passed**, 13,61 с; выбранные Ruff correctness/import checks и `git diff --check` прошли.
- `artifacts/pilot`: **96/96** принятых oracle-эпизодов, **32** уникальные target программы, 320 переходов, **640 PNG** (целевые виды и исходные виды edit/repair).
- Train: 57 full-program SFT / GRPO / VLM / preference записей и 190 next-action пар. Validation: 6 эпизодов, test: 33; flange целиком в test.
- Генерация с двумя workers заняла около 58 с. Это один локальный pilot, а не гарантированная скорость на другом железе/семействах.
- Аудит: 1 855 hashes, split leakage и экспортные counts — проверка пройдена. Проверяются FCStd, checkpoints, изображения, task manifests и source snapshot.
- Контрольная группа reward: правильное решение **1,0**, неверная толщина **0,5647**, пустое решение **0**, невалидный JSON **0**. Это специально подготовленные candidates для проверки scorer, не ответы обученной модели.
- В изолированном окружении TRL 0.24 выполнены CPU smoke-запуски SFT и GRPO на случайной локальной модели с 291 296 параметрами. SFT: один optimizer step с ненулевым gradient. GRPO: один step с invalid completions, reward 0 и нулевой дисперсией — **проверка интеграции, без обучающего сигнала и без улучшения CAD-навыков**.
- VLM JSONL успешно загружен в HF Dataset с четырьмя декодированными PIL изображениями на строку; полноценное VLM-обучение не запускалось.

Пилот и диагностические логи сохраняются в `artifacts/` и исключены из git. Запуск полноценного обучения выбранной модели и сравнение CAD success на holdout — следующий эксперимент; его результаты здесь не заявлены.
