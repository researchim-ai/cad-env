# Qwen3.5-4B: FreeCAD benchmark и LoRA SFT

Эксперимент от 4 ноября 2026. Модель — [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Apache-2.0. Это VLM с гибридными linear/full attention слоями. Оба скачанных шарда проверены по upstream SHA-256; origin хранится в `artifacts/models/qwen35-4b/model-origin.json`.

Все GPU-запуски выполнены с `CUDA_VISIBLE_DEVICES=1` на физической RTX 3090. Внутри процесса она отображается как `cuda:0`. Старый стек Qwen2.5 сохранён отдельно; новый стек закреплён в `requirements-qwen35.lock`, полные версии — в environment.json обучающего запуска. Используется Flash Linear Attention 0.6.0; causal convolution использует корректный PyTorch fallback.

## Исправление контракта до основного benchmark

Первоначальная проверка выявила конфликт: ACTION_GUIDE требовал массив действий или `{"actions":[...]}`, а последовательный интерфейс требовал один объект с `op`. Модель выдавала допустимую по первой инструкции обёртку, которую пошаговый executor отвергал.

В `agent_messages` конфликт устранён. Для старых frozen SFT rows применяется та же явная, идемпотентная миграция system prompt; targets, native labels, изображения и splits не изменяются. Миграция записана в training manifest, исходный корпус не перезаписан. Прерванный `qwen35-baseline-validation` сохранён как диагностический и исключён из сравнения. Основной baseline — `qwen35-baseline-v2`.

Исторический Qwen2.5 baseline 0/45 использовал старый контракт; его нельзя напрямую сравнивать с этим baseline и трактовать исключительно как отсутствие CAD-навыка. Новый before/after benchmark использует одинаковый исправленный контракт.

## Данные и оценка

Используется наш процедурный `gpu1-corpus-v2`, а не внешний CAD benchmark: 546 train tasks / 2002 train feedback rows, 69 validation tasks, 297 закрытых test tasks. Validation содержит 22 независимые lineage groups; audit проверяет разделение связанных create/edit/repair. Flange целиком отложен в test. Test в этом эксперименте не использовался.

Модель получает инструкцию, публичное состояние документа, последние переходы и текущее изображение 256 px. Ответ — одно JSON-действие; оно выполняется в настоящем FreeCAD. Валидация использует OCCT, геометрию, размеры, параметричность, историю и protected edits. Одинаковые бюджеты before/after: greedy, до 16 действий, 256 generated tokens/action, context 4096. Thinking выключен через штатный `enable_thinking=False`; сохранённый SFT template даёт точно такой же prompt.

Основной baseline на всех 69 validation tasks:

- Strict success **46/69 = 66,67%**.
- Mean CAD reward **0,69315**.
- Create **0/23**, edit **23/23**, dimension repair **23/23**.
- Invalid-action rate **56,74%**, truncation rate **10,14%**.

Ненулевой partial reward не считается strict success. Ни похожая картинка, ни успешный JSON сами по себе не доказывают правильность детали.

После 200 шагов SFT сохранённая и повторно загруженная модель решила **69/69 = 100%**: create 23/23, edit 23/23, repair 23/23. Mean CAD reward **1,000**, invalid-action rate **0%**, truncation rate **0%**. Прирост strict success **+33,33 п.п.**, reward **+0,30685**. Protocol digest и task IDs совпадают с baseline.

Парный cluster bootstrap учитывает 22 lineage groups, `improvement_supported=true` по pilot gate. Его interval вырожденный [+33,33; +33,33 п.п.], поскольку каждая группа имеет одинаковый профиль прироста. Это не измерение неопределённости общего CAD-домена: структуры программ здесь ограничены тремя семействами.

## Обучение

LoRA rank 16, alpha 32, dropout 0; языковые q/k/v/o projections и projections linear attention. Trainable parameters **14 376 960**, vision trainable **0**. BF16, SDPA, gradient checkpointing без reentrant. Batch 2, accumulation 4, effective batch 8; 200 optimizer steps, LR 1e-4 с линейным уменьшением, seed 42. Это SFT, а не GRPO optimization.

Все 2002 train examples проверяются до оптимизации: точная template boundary, image tokens исключены из loss, targets не обрезаны. После миграции prompt максимум 1565 tokens и 63 target tokens. Проверены native PyTorch и FLA forward/backward smoke runs; batch 2 также проверен отдельно. Loss/gradients контролирует numerical guard.

Обучение завершилось за **1369,0 с = 22,8 минуты** (сам optimization loop, без загрузки/подготовки), 1600 примеров, 0,799 epoch. Все 200 logged gradients конечны и ненулевые. Mean train loss **0,0041931**. Пиковая allocated память **10,11 GiB**, reserved **10,67 GiB**. Это измерения данного запуска; фоновая нагрузка GPU не контролировалась как в отдельном performance benchmark.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv-qwen35/bin/python -m cad_env.learning.workflow evaluate \
  --dataset artifacts/gpu1-corpus-v2 --model artifacts/models/qwen35-4b \
  --output artifacts/qwen35-baseline-v2 --disable-thinking \
  --max-steps 16 --max-new-tokens 256

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  .venv-qwen35/bin/python -m cad_env.learning.train_modern \
  --model artifacts/models/qwen35-4b --dataset artifacts/gpu1-corpus-v2 \
  --output artifacts/qwen35-sft200 --max-steps 200 --batch-size 2 \
  --gradient-accumulation 4 --save-steps 50

OMP_NUM_THREADS=2 .venv-qwen35/bin/python -m cad_env.learning.merge \
  --adapter artifacts/qwen35-sft200 --output artifacts/qwen35-sft200-merged

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv-qwen35/bin/python \
  -m cad_env.learning.workflow evaluate --dataset artifacts/gpu1-corpus-v2 \
  --model artifacts/qwen35-sft200-merged --output artifacts/qwen35-sft200-validation-resumed \
  --disable-thinking --max-steps 16 --max-new-tokens 256
```

Использовать новые output paths для повторных запусков. Первый post-SFT evaluation был остановлен вместе с окружением после пользовательского прерывания, успев пройти 6 задач. Он помечен `interrupted` и исключён; полный прогон выполнен заново в `qwen35-sft200-validation-resumed`, без повторного обучения. Итоговые метрики before/after, память, обучение и cluster-bootstrap comparison сохраняются в `artifacts/qwen35-experiment/run.json` и `comparison.json`; исходные evaluations — в соответствующих каталогах. Данные и веса лежат в gitignored `artifacts/`.

Это проверка внутри трёх процедурных семейств и нашего языка tools. Она не является общим CAD benchmark, проверкой GUI, сложных сборок или доказательством визуального grounding: текстовое состояние содержит много численной информации.

## Контроль без изображений

Та же сохранённая VLM прошла все 69 validation tasks с `--state-only`, сохраняя greedy, thinking off и тот же action budget. Архитектура модели не заменялась текстовой; убраны только изображения из observations. Результат: **69/69, mean reward 1,000, invalid actions 0%, truncations 0%**. Model hashes, task IDs и dataset digest совпадают с основной оценкой; protocol visual flag отличается намеренно.

Следовательно, текущие задачи решаются по текстовому состоянию и не демонстрируют дополнительную пользу изображений. Подтверждённый вывод — адаптация Qwen3.5 к нашему CAD tool-use и знакомым конструкциям. Для проверки визуального обучения нужны отдельные image-conditioned задачи, в которых ответ нельзя восстановить из полного численного состояния.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv-qwen35/bin/python \
  -m cad_env.learning.workflow evaluate --dataset artifacts/gpu1-corpus-v2 \
  --model artifacts/qwen35-sft200-merged --output artifacts/qwen35-sft200-state-only \
  --disable-thinking --state-only --max-steps 16 --max-new-tokens 256
```

Финальный статус `artifacts/qwen35-experiment/run.json` — `complete`. Проверены совпадение исходной модели baseline/SFT, protocol baseline/candidate, task IDs всех трёх прогонов, dataset digest и fingerprints модели в VLM/state-only. Регрессионные проверки проекта: **66 passed**; selected correctness/import lint и `git diff --check` прошли.
