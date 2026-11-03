# Процесс обучения FreeCAD tools/VLM агента на двух RTX 3090

Цель согласована с пользователем: агент управляет **FreeCAD API/tools**, получает **текущий вид модели**, выполняет следующие действия и восстанавливается после ошибок. GUI computer-use в этот эксперимент не входит. Оборудование: RTX 3090 Ti и RTX 3090, примерно по 24 ГБ. Статус и фактические результаты ниже; наличие recipe не заменяет измерение качества модели.

## 1. Что считаем результатом

Первый результат — ограниченный агент для реализованного языка Sketch/Pad/Pocket/constraints. Он должен строить и редактировать native параметрические детали, завершать задачу через `finish`, сохранять редактируемость и выполнять требования. Уменьшение training loss, синтаксически правильный JSON, успешный запуск Python и похожая картинка по отдельности не доказывают успех.

Сейчас библиотека включает четыре структурных семейства. Это bootstrap для инструментов и проверки процесса; широкое владение FreeCAD потребует новых структур, операций, независимых пользовательских задач и внешних benchmarks. Визуальная польза тоже должна быть измерена: состояние содержит много численной информации, поэтому модель может решать текущие задачи без внимательного чтения изображения.

Готовность разделяем на четыре независимых утверждения:

1. **Исполнитель и labels достоверны:** native geometry, constraints и независимые проверки работают.
2. **Обучение технически исправно:** правильные токены входят в loss, gradients не нулевые, checkpoint/resume/merge сохраняют веса, DDP работает.
3. **Выбранная модель обучилась:** вырос CAD success на неизменном validation при одинаковом бюджете.
4. **Обобщение подтверждено:** пройден закрытый test после выбора checkpoint и дополнительные семейства/задачи.

Нельзя автоматически переносить успешность уровня 1–2 на уровни 3–4.

## 2. Выбор опубликованного рецепта

Для основного цикла используем архитектуру, близкую к **IterCAD**: SFT, исполнение, визуальный feedback, исправления, затем multi-turn GRPO. Зафиксирован upstream `KnowledgeXLab/IterCAD` commit `55998c4a4537573f67ca241dedc52ee2840652ca`, интерфейсы перечислены в `recipes/itercad-upstream.json`. Проверены [training script](https://github.com/KnowledgeXLab/IterCAD/blob/55998c4a4537573f67ca241dedc52ee2840652ca/train/IterCAD_RL.sh) и [scheduler](https://github.com/KnowledgeXLab/IterCAD/blob/55998c4a4537573f67ca241dedc52ee2840652ca/train/ms-swift/swift/rollout/multi_turn.py).

Оптимизацию и distributed/token accounting оставляем библиотеке. В проект добавлен экспериментальный `swift_plugin.py`, который связывает этот интерфейс с FreeCAD tools и private terminal reward. Native replay core проверяется отдельно. **Полный ms-swift/vLLM multi-turn optimizer ещё требует отдельного integration-теста на GPU.** Он не объявлен production-ready.

Стабильный TRL launcher обслуживает SFT и отдельный single-completion program-GRPO baseline. Программа и одно действие имеют разные выходные контракты. Планировщик запрещает запускать program-GRPO от next-action SFT checkpoint. Это предотвращает цикл, в котором обученный агент выдаёт один правильный action, а reward считает его неправильной полной программой.

## 3. Репродуцируемая среда

Основной профиль: `recipes/dual-3090-vlm.json`. Начальная модель — **Qwen2.5-VL-3B-Instruct**, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. Модель имеет около 3,75 млрд параметров; название 3B не следует использовать для точного расчёта памяти. Карточка указывает Qwen Research License: [зафиксированная модель](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3).

```bash
python3.12 -m venv .venv-train
.venv-train/bin/pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv-train/bin/pip install -r requirements-learning.lock
.venv-train/bin/pip install -e '.[data,api]'
export FREECAD_CMD=/usr/bin/freecadcmd-daily
.venv-train/bin/python -m cad_env.learning.cli doctor
```

Lock фиксирует прямые ML-зависимости проверенного интерфейса. Каждый training run дополнительно сохраняет `environment.json` с версиями установленных пакетов, Python/CUDA, аргументы, hashes данных/весов, source snapshot и память каждого rank. Не включать в логи ключи API или полный environment.

Одна GPU не использует VRAM другой как дополнительную память. Начальный профиль — DDP с одной копией модели на каждой GPU, BF16, LoRA по decoder projections, замороженный vision encoder, microbatch 1, gradient accumulation 8, effective batch 16, SDPA и gradient checkpointing без reentrant. Для начала — один текущий вид 256 px и общий budget 4096 токенов. Вместимость и скорость проверяются на обеих картах, включая более загруженную graphics GPU.

## 4. Verifier до обучения

```bash
python -m cad_env.learning.calibration --output artifacts/verifier-calibration-v2
```

Калибровка должна принять корректную деталь и другую корректную историю построения. Она должна отвергнуть сдвинутое отверстие при том же объёме/габаритах, слепое отверстие вместо сквозного, неверный радиус, лишнее тело, пустое завершение, невалидный JSON и изменение защищённого размера при той же конечной форме.

Scorer использует исходные координаты и миллиметры, OCCT symmetric difference, независимые аналитические volume/bounds, валидность solid, constrained sketches, изменение размера и edit history. Для edit защищены все именованные размеры и зависимости, кроме явно разрешённых. Worker/schema/runtime не могут незаметно измениться во время одного запуска.

Terminal reward: 0 при нарушении shape/parametric/history/protected-edit gates; иначе `0.8 * geometry_score + 0.2 * exact_spec_match`. Порог 1e-5 принят для текущего диапазона деталей. Этот reward следует перекалибровать при расширении домена; он не заменяет инженерные tolerances произвольного проекта.

## 5. Данные и остановка на дефектах

```bash
python -m cad_env.learning.cli generate --output artifacts/new-corpus \
  --seeds 80 --workers 4 --preferences --holdout-families flange
python -m cad_env.learning.cli audit --dataset artifacts/new-corpus
```

Канонический пример — spec + replay + actual native verdict + observation before/after. Первичное supervision для нашей цели — `exports/train.feedback.jsonl`: текущая модель/эскиз/пустой документ, текст состояния, короткая история и **одно следующее JSON-действие**. Целевой код не попадает в observation. После ошибочного действия используется последний committed state, а не частично испорченная геометрия.

Recovery labels получают после реальной ошибочной команды. Неверное действие сохраняется во входном контексте; target — проверенное исправление. Dimension repair и solver/runtime recovery учитываются раздельно.

После анализа реальных провалов фабрика дополнена `duplicate_name` recovery: повторный Sketch исполняется против committed prefix и действительно отклоняется FreeCAD; target продолжает построение. Эти варианты будут в новом корпусе, а `learning-corpus-v1` остаётся неизменным. Для seed range 0..79 планировщик даёт 22 validation lineage groups до дедупликации; окончательное число проверить через `workflow check`.

Связанные create/edit/repair, изображения и вариации одного base design остаются в одном split. Одинаковые задачи удаляются. Flange целиком в test. Семейства test не использовать для teacher sampling, mining или выбора checkpoint.

Автоматический data gate требует минимум 200 train / 30 validation / 30 test задач, 80 lineage groups, три структуры обучения, current-state изображения и реальные recovery примеры. Это **инженерные условия bootstrap-пилота**, не универсальные научные нормы. Они не разрешают объявить four-template генератор общим CAD-корпусом.

Отдельный gate для подтверждения улучшения требует минимум 20 независимых validation lineage groups. В `learning-corpus-v1` их 15: корпус пригоден для технического пилота, но недостаточен для promotion. `workflow check` выводит этот статус отдельно. Следующий замороженный корпус должен расширить validation до прохождения этого gate; связанные create/edit/repair не считаются независимыми наблюдениями.

Дополнительно до расширения корпуса: визуально просмотреть выборку из каждого семейства/режима; проверить координаты и единицы; включить парафразы, новые graph structures, ошибочные references и более длинные зависимости. Само увеличение числа случайных размеров недостаточно.

## 6. GPU smoke до длинного SFT

```bash
.venv-train/bin/python -m cad_env.learning.workflow check \
  --dataset artifacts/learning-corpus-v1 --recipe recipes/dual-3090-vlm.json \
  --output artifacts/readiness.json
.venv-train/bin/python -m cad_env.learning.workflow plan \
  --dataset artifacts/learning-corpus-v1 --recipe recipes/dual-3090-vlm.json \
  --run-output artifacts/sft-gpu-smoke --smoke --output artifacts/smoke-plan.json
```

`plan` сохраняет точную команду в `shell`/`command`; её запускают в подготовленной среде. Перед запуском переопределить Python-путь при переносе на другой компьютер. Smoke — 10 optimizer steps с save/reload. Условия успеха: обе GPU работают; loss/gradients конечны; меняются нужные LoRA weights; vision encoder frozen; нет OOM; есть checkpoints и память обоих rank; работает resume.

Trainer проверяет **все** train examples: нет потери prompt/completion boundary, image tokens не обрезаны, completion mask оставляет target tokens. Число изображений и полный token budget ограничены до оптимизации. Не лечить OOM молчаливым truncation CAD-ответа.

Обнаружен и исправлен нюанс dense VLM resume в Transformers 4.57.0: стандартный Trainer может обойти преобразование сохранённых имён весов. Strict loader делает архитектурную загрузку и проверяет каждый tensor. Adapter resume тоже проверяется точным round-trip. FSDP/DeepSpeed этим loader не поддерживаются; профиль использует DDP.

## 7. Baseline и первый SFT

До оптимизации записать frozen-model baseline на validation. Для основной метрики — один greedy rollout, одинаковые 16 tool attempts, 256 generated tokens на action, контекст 4096 и один текущий вид. Не скрывать невалидные действия/таймауты/незавершённые эпизоды.

```bash
.venv-train/bin/python -m cad_env.learning.workflow evaluate \
  --dataset artifacts/learning-corpus-v1 --model /path/to/pinned-base-model \
  --output artifacts/baseline-validation --max-steps 16 --max-new-tokens 256
```

Первый SFT cap — 200 steps при effective batch 16: приблизительно 2,8 прохода по текущим 1130 train transitions. Сохранять каждые 50 steps. Выбирать checkpoint по native validation, а не по минимальному training loss. Если качество падает или plateau наступило раньше, большой cap не нужно дорабатывать автоматически. Рецепт задаёт LoRA rank 16 и learning rate 1e-4 как старт пилота; sweep/корректировка опираются на validation.

Смешение шагов текущего teacher: около 60% create и по 20% edit/repair; 113 recovery первых шагов. После первого baseline отдельно оценить mode imbalance и при необходимости изменять sampler. Для визуальной пользы сравнить **ту же VLM** с `--state-only`: архитектура остаётся VLM, убираются только изображения. Эта абляция имеет другой input protocol и не является paired checkpoint promotion.

## 8. LoRA, checkpoint и оценка

```bash
.venv-train/bin/python -m cad_env.learning.merge \
  --adapter artifacts/sft-run --output artifacts/sft-merged
.venv-train/bin/python -m cad_env.learning.workflow evaluate \
  --dataset artifacts/learning-corpus-v1 --model artifacts/sft-merged \
  --output artifacts/sft-validation --max-steps 16 --max-new-tokens 256
.venv-train/bin/python -m cad_env.learning.workflow compare \
  --baseline artifacts/baseline-validation/evaluation.json \
  --candidate artifacts/sft-validation/evaluation.json --output artifacts/comparison.json
```

Для adapter с remote base в merge передать ту же `--revision`, что при SFT. Merge выполняется на CPU в float32 и может требовать значительного объёма RAM. Слитую модель заново загрузить и оценить; это отдельный deployable artifact.

Отчёт содержит success@1, success при фиксированном бюджете попыток, invalid-action rate, steps, latency, truncation и family/mode breakdown. Paired comparison разрешён только при тех же задачах, dataset digest, worker/prompt schema и бюджете. Failed/debug-limited evaluation не может повысить статус checkpoint. Bootstrap ресэмплирует **lineage groups**, учитывая зависимость create/edit/repair; при малом числе групп статистическое улучшение не заявляется.

Перед финальным test дополнительно проверяем отсутствие существенной регрессии по режимам, неприемлемого роста шагов и деградации edit history. Test запускается один раз после выбора checkpoint с явным `--split test --final-test`; повторные попытки и их причины регистрируются. Для широкой CAD-компетенции нужны внешние наборы и новые структуры, не происходящие от нашего генератора.

## 9. Реальные ошибки политики → следующий SFT

Следующий корпус строить по **train rollouts самой SFT policy**. Сохранять успехи и неуспехи. Успешный terminal native verdict разрешает извлечь корректные решения и исправления; не обучать модель повторять ошибочный action как target. Validation/test не превращаются в train supervision.

`collect_group(..., visual=True)` принимает `policy(messages, image_paths)` и исполняет модель пошагово. `bundle_supervision` собирает неизменяемый image bundle с hashes, не меняя исходный dataset. Для обучения по такому экспорту передать `--export <bundle>/exports/train.feedback.jsonl --asset-root <bundle>`; базовый `--dataset` по-прежнему контролирует разрешённые train task IDs.

В rollout budget включаются неудачные попытки, повторения и `finish`. Если агент циклически повторяет одно действие, это не исправляется удалением неудобных эпизодов из метрики. Данные повторяющихся провалов группировать по root cause: JSON/schema, reference selection, geometry, dimension, planning, stop condition.

## 10. GRPO только после наличия сигнала

Сначала собрать минимум 20 train prompts по четыре свежих rollout SFT policy. Проверить:

- ≥80% геометрически допустимых завершений/исполнений для старта пилота;
- ≥10% групп содержат успех и неуспех;
- ≤90% групп с нулевой дисперсией reward.

```bash
python -m cad_env.learning.workflow grpo-readiness \
  --scores actual-policy-scores.jsonl --output artifacts/grpo-readiness.json
```

Пороговые значения — начальные эвристики, а не свойства алгоритма GRPO. Все нули → проще curriculum/больше SFT. Все единицы → сложнее задачи. Artificial negatives/oracle programs пригодны для калибровки verifier, но не доказывают readiness реальной policy.

Для нашего агента основной следующий RL-трек — **multi-turn GRPO** через опубликованный scheduler API и `swift_plugin.py`: реальные действия и ошибки, изображения текущего состояния, private terminal reward. Требования integration gate: точное соответствие token IDs/logprobs/loss masks всех assistant turns (включая finish); environment и image tokens не входят в policy loss; независимое состояние для каждого candidate; корректный checkpoint; измеренная память на двух GPU. Pinned upstream и непроверенные части явно указаны в `recipes/itercad-upstream.json`. Пока этот gate не пройден, большой multi-turn RL запуск не начинать.

Program-GRPO остаётся отдельным baseline для policy, обученной выдавать целый массив действий. Стартовые G=4, LR=5e-6, beta=0, 100 steps — параметры пилота, требующие собственной калибровки. Открытый GRPO optimizer не нужно заменять самодельным loss. Использовать статические offline advantages для обновления новой policy некорректно.

## 11. Что проверено локально

Фактические артефакты находятся в `artifacts/`:

- `learning-corpus-v1`: 564 принятых эпизода, 12 duplicates исключено, 339 train / 45 validation / 180 test, 173 lineage groups, 1130 train feedback rows и 113 recovery rows. Аудит проверяет 18 841 hashes. Это ограниченный teacher corpus.
- `process-verifier-calibration`: девять положительных/отрицательных случаев, включая другую корректную историю и одинаковую геометрию с неверным protected edit.
- `process-vision-smoke`: настоящая случайная tiny LLaVA (72 896 параметров), forward/backward, восемь image tokens в двух примерах, все image tokens masked, loss конечен и веса изменились.
- `process-vlm-lora` / `process-vlm-merged`: LoRA и merge проверены на маленькой VLM.
- `process-resume-equivalence`: непрерывное обучение и resume дали побитовое совпадение всех 39 tensors; max delta = 0.
- `process-ddp-smoke-v2`: реальный DDP/LoRA с frozen vision на RTX 3090 Ti + RTX 3090 завершил три шага. Это проверка distributed plumbing, а не памяти большой модели.
- `qwen-dual3090-smoke`: основная Qwen2.5-VL-3B, 10 optimizer steps / 160 примеров за 52,9 с, 7 372 800 trainable LoRA parameters, vision frozen. Все 1130 feedback rows проверены до обучения, максимум 1563 токена. Градиенты конечны и ненулевые на всех десяти шагах. Пиковая allocated память 10,16 / 10,17 GiB, reserved 21,42 / 21,99 GiB. Оба rank подтвердили завершение. Шарды модели совпали с upstream SHA-256.
- `qwen-dual3090-resume`: все 288 adapter tensors восстановлены точно из checkpoint-5, оба rank завершили продолжение до шага 10. Итоговая максимальная разница с непрерывным GPU запуском — 0,00010747; продолжение не побитово идентично. Не путать точность загрузки checkpoint с детерминизмом последующей BF16 оптимизации.
- `qwen-sft10-validation`: полная оценка merged десятишаговой VLM, greedy, 16 действий и 256 tokens/action. Строгий CAD success 30/45: create 0/15, edit 15/15, dimension repair 15/15. Invalid-action rate 62,4%; этот checkpoint ещё не пригоден как надёжный общий CAD-агент. Средний reward 0,700 включает частичную геометрическую награду и не заменяет strict success. Test не запускался.
- `qwen-baseline-validation` / `qwen-sft10-comparison.json`: исходная закреплённая VLM получила 0/45, SFT — 30/45 на тех же задачах и protocol digest. Прирост +66,7 п.п.; `improvement_supported=false`, поскольку независимых групп только 15 при минимуме 20. Все группы имеют одинаковый профиль create=0/edit=1/repair=1, поэтому вырожденный bootstrap interval не измеряет неопределённость более широкого CAD-домена. Скорость baseline/candidate напрямую не сравнивать: использованы разные GPU с фоновой нагрузкой.
- `process-evaluation-comparison`: random tiny-model benchmark не показал улучшения; promotion правильно отклонён.

GPU smoke основной VLM и парная validation завершены. Результат подтверждает обучение ограниченным dimension tools, но не создание деталей с нуля. Следующий эксперимент: новый frozen corpus с duplicate recovery и ≥20 validation groups, baseline на нём, SFT-пилот 200 steps и сравнение checkpoints при одинаковом rollout budget; закрытый test использовать после выбора модели. Сначала исправить create/repetition failures. Multi-turn GRPO integration gate ещё не пройден. Полный финальный набор проверок: 65 passed; `ruff` выбранных correctness/import checks и `git diff --check` прошли.

Для сбора следующего train-корпуса есть CLI:

```bash
.venv-train/bin/python -m cad_env.learning.workflow collect \
  --dataset artifacts/learning-corpus-v1 --model artifacts/sft-merged \
  --output artifacts/agent-rollouts-v1 --prompts 20 --num-generations 4
```

Prompts выбираются стратифицированно по family/mode из train. `collection.json` содержит readiness на реальных policy samples. Успехи упаковываются в `exports/train.feedback.jsonl` и `supervision.json` с hashes изображений и digest исходного dataset. Следующий SFT принимает этот export с `--asset-root artifacts/agent-rollouts-v1`; несовпадение исходной ревизии dataset останавливает обучение.
