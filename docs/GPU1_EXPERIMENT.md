# Проверка полного SFT-цикла на GPU 1 — 4 ноября 2026

Последующая [проверка Qwen3.5](QWEN35_EXPERIMENT.md) выявила конфликт формата в старом sequential prompt: общий справочник разрешал массив/обёртку действий, а executor требовал один объект. Ниже сохранены исторические результаты этого контракта. Поэтому baseline 0/45 и его прирост нельзя трактовать исключительно как обучение геометрии или напрямую сравнивать с новым исправленным benchmark. Актуальный контракт устранён одновременно для inference и preprocessing SFT; protocol hashes различаются.

Эксперимент выполнен на физической **GPU 1, NVIDIA RTX 3090 24 GB**. Во всех GPU-командах установлен `CUDA_VISIBLE_DEVICES=1`; внутри процесса эта карта закономерно называется `cuda:0`. GPU 0 для этого эксперимента не использовалась.

## Данные

Новый корпус `artifacts/gpu1-corpus-v2` сгенерирован за 572,2 с через настоящий FreeCAD. Получено 912 принятых эпизодов, 0 отклонённых; исключены 48 дубликатов. Разделение: 546 train / 69 validation / 297 test. Сохранены 3040 current-state изображений, 2002 train feedback-примера, включая 364 recovery-примера. Есть реальные ошибки отсутствующего объекта и повторного создания эскиза.

Аудит прошёл; data readiness — true. В новом validation 22 независимые lineage groups. Пересечение новых train-групп с группами старого validation, использованного ниже, — ноль. Test не запускался.

```bash
.venv-learning/bin/python -m cad_env.learning.cli generate \
  --output artifacts/gpu1-corpus-v2 --seeds 80 --workers 4 \
  --preferences --holdout-families flange
```

## Обучение

Qwen2.5-VL-3B-Instruct, закреплённая revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, BF16, SDPA, LoRA rank 16 по q/k/v/o, vision encoder frozen. Microbatch 1, accumulation 8, effective batch 8, 200 optimizer steps, LR 1e-4 с линейным уменьшением, seed 42.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv-learning/bin/python \
  -m cad_env.learning.train --method sft \
  --model artifacts/models/qwen25-vl-3b \
  --dataset artifacts/gpu1-corpus-v2 \
  --export artifacts/gpu1-corpus-v2/exports/train.feedback.jsonl \
  --output artifacts/gpu1-sft200 --require-cuda \
  --freeze-vision --require-learning-signal --max-steps 200 \
  --batch-size 1 --gradient-accumulation 8 --lora-rank 16 \
  --lora-targets q_proj,k_proj,v_proj,o_proj --max-length 4096 \
  --max-images 1 --max-pixels 65536 --learning-rate 0.0001 --save-steps 50
```

Все 2002 train rows проверены до обучения: completion-only masks, отсутствие truncation, максимум 1564 токена. Trainable parameters: 7 372 800; vision trainable: 0. Все 200 шагов имели конечные ненулевые gradients. Обучение завершилось за **913,0 с**, 1,752 примера/с; обработано 1600 примеров. Пиковая allocated память **10,14 GiB**, reserved **18,83 GiB**. Средний train loss 0,01454; первый logged loss 0,3482, на шаге 50 — 0,0008. Checkpoint-50 сохранён отдельно до автоматического удаления старых checkpoints.

## Реальный CAD reward

Оба checkpoint объединены с базой, повторно загружены и исполнялись в FreeCAD **на GPU 1** с текущими изображениями. Использован неизменный `learning-corpus-v1` validation: 45 задач, greedy, максимум 16 действий и 256 generated tokens/action. Baseline взят из ранее завершённой оценки той же закреплённой модели; protocol digest совпадает. Все ошибки входят в метрики.

- Base: mean CAD reward **0,000**, strict success **0/45**.
- Checkpoint-50: mean CAD reward **0,99396**, strict success **44/45**; create 14/15, edit 15/15, dimension repair 15/15; invalid actions 1,32%.
- Checkpoint-200: mean CAD reward **1,000**, strict success **45/45**; create 15/15, edit 15/15, dimension repair 15/15; invalid actions 0%, truncations 0%.

Таким образом, данные генерируются, SFT обучается на одной RTX 3090, и CAD reward растёт при реальном исполнении. Это рост качества **после SFT**; optimizer SFT не использует reward как RL loss.

Оценка ограничена тремя процедурными семействами и 15 независимыми группами старого validation. Поэтому автоматический comparison не подтверждает promotion по правилу минимум 20 групп; 45/45 не означает общего владения FreeCAD. Наличие 22 групп в новом корпусе само по себе не расширяет старый evaluation. Результаты закрытого test здесь не заявляются.

Основные артефакты: `artifacts/gpu1-experiment/run.json`, `artifacts/gpu1-experiment/data-readiness.json`, `artifacts/gpu1-sft200/cad-training-run.json`, `artifacts/gpu1-val50/evaluation.json`, `artifacts/gpu1-val200/evaluation.json`. Checkpoints и изображения находятся в игнорируемом git каталоге `artifacts/`.

## Сигнал для GRPO

После SFT собраны свежие train-only группы: 20 prompts × 4 stochastic tool rollouts, temperature 0,8, на GPU 1. Результат **80/80 успешных rollout, reward 1,0 во всех попытках**, 272 проверенных supervision steps упакованы в immutable bundle. Execution validity 100%; mixed-success groups 0%; zero-variance groups 100%.

GRPO readiness — **false**: текущие задачи стали слишком простыми и не дают относительного reward signal. Следующий шаг — расширить curriculum новыми структурами и сложностью, затем повторить сбор групп. Это не доказательство исправности полного multi-turn GRPO optimizer: он остаётся отдельной экспериментальной интеграцией. Нельзя заменять проверку optimizer успехом SFT или намеренно подмешанными отрицательными teacher-программами.

Статус и relative-reward gate записаны в `artifacts/gpu1-fresh-rollouts/collection.json`, финальный отчёт эксперимента — `artifacts/gpu1-experiment/run.json` со статусом `complete`.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 .venv-learning/bin/python \
  -m cad_env.learning.workflow collect \
  --dataset artifacts/gpu1-corpus-v2 --model artifacts/gpu1-sft200-merged \
  --output artifacts/gpu1-fresh-rollouts --prompts 20 --num-generations 4 \
  --max-steps 16 --max-new-tokens 256
```

Повторные запуски должны использовать новые output paths: существующие артефакты намеренно не перезаписываются. Для evaluation сохранённых моделей используется `workflow evaluate` с `--dataset artifacts/learning-corpus-v1 --max-steps 16 --max-new-tokens 256` и новым `--output`.
