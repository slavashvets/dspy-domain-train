# SRP Debug Status - 2026-05-20

## Исходная проблема

Нужно было показать, что SRP реально улучшает слабую начальную инструкцию из
`prompts/p0.txt`, а не просто возвращает уже сильный docstring из
`DomainClassificationSig`.

Начальное ожидание:

- production default без `initial_prompt_path` использует полный docstring;
- demo profile использует `prompts/p0.txt`;
- SRP должен стартовать с `p0`, предложить улучшенную инструкцию и сохранить ее в
  `runs/.../final_instructions.txt`;
- `srp_candidates.json` должен показывать, какие кандидаты были предложены,
  оценены и приняты.

Проблема до фиксов: `final_instructions.txt` часто оставался равен `p0`, потому
что все refiner rules ухудшали validation score и честно отклонялись.

## Что было в старом коммите

Смотрели коммит:

`3d344374af58395d94905b28e0f0859727786c77`

Там SRP был не как текущий DSPy `Teleprompter`, а как ручной `offline_srp`.
Важные рабочие идеи оттуда:

- refiner генерировал не один append-only набор правил, а полную новую
  инструкцию;
- генерировалось несколько кандидатов (`k`);
- каждый кандидат отдельно оценивался evaluator-ом на devset;
- выбирался лучший кандидат по фактическому score;
- refiner получал не сырой список ошибок, а агрегированный error report.

Что не стоит портировать как есть:

- hardcoded banned words;
- dataset phrase rejection;
- ручные example indicators вроде `kid-friendly museum`;
- length penalty как reward;
- ручные domain seed rules.

Вывод: из старого варианта нужно брать стратегию candidate search + evaluation,
а не доменные подпорки.

## Что было не так с текущим датасетом

После проверки baseline errors на `max_dev = 100` обнаружились фантомные labels,
которые невозможно вывести из текущего turn/context.

Примеры до фикса конвертера:

- `I am looking for a train this Saturday. Can you help?`
  имел gold `['restaurant', 'train']` при пустом context.
- `i need a place to dine in the center thats expensive`
  имел gold `['hotel', 'restaurant']`.
- closing turns вроде `Thank you, that is all that I need.`
  иногда сохраняли активный предыдущий домен.

Причина была в `scripts/convert_multiwoz.py`: конвертер добавлял домен из
`active_intent` и `requested_slots` даже для пустых/фоновых MultiWOZ frames.
Из-за этого refiner учил неправильную стратегию "carry prior/context domains",
которая резко ухудшала validation.

## Что изменено сейчас

### SRP

Файл: `src/dspy_domain_train/srp.py`

Текущая стратегия:

- `SRPRefinerSig` теперь просит `candidate_instructions: list[str]`, то есть
  полные revised instructions, а не append-only rules.
- SRP строит generic `feedback_report`:
  - число ошибок;
  - missing labels;
  - extra labels;
  - representative examples;
  - metric feedback.
- На каждой итерации:
  - собираются ошибки текущего best/current prompt;
  - refiner генерирует несколько instruction candidates;
  - каждый candidate прогоняется через `dspy.Evaluate`;
  - кандидаты сортируются по score;
  - принимается только strict improvement: `candidate_score > best_score`.
- Убран demo hack:
  - нет `seed_rules`;
  - нет tie-accept.

### Settings

Файл: `src/dspy_domain_train/settings.py`

Добавлены SRP настройки:

- `num_candidates`;
- `candidate_retries`;
- `proposal_temperature`.

Файл: `settings.demo.toml`

Сейчас demo profile:

- `max_train = 50`;
- `max_dev = 100`;
- `max_test = 30`;
- `initial_prompt_path = "prompts/p0.txt"`;
- `srp.num_candidates = 4`;
- `srp.candidate_retries = 1`;
- `srp.proposal_temperature = 1.0`.

### Logging/artifacts

Файл: `src/dspy_domain_train/main.py`

`srp_candidates.json` сохраняет:

- iteration;
- rank;
- score;
- accepted;
- instruction.

Важно: текущий прогресс внутри run-dir не виден до конца запуска. Файлы
`program.json`, `metadata.json`, `srp_candidates.json`,
`final_instructions.txt` пишутся только после завершения. Запуск через `grep`
также фильтрует progress bars. Для следующих длинных прогонов нужен `run.log`
через `tee` или явный progress logging внутри SRP.

### Конвертер MultiWOZ

Файл: `scripts/convert_multiwoz.py`

Изменения:

- closing turns сразу становятся `['none']`;
- пустые/фоновые active intents не добавляют домен без evidence;
- evidence берется из slot values, requested slots с упоминанием домена, либо
  явного domain mention в utterance/context;
- slot-derived domains также требуют mention в utterance/context.

После регенерации:

- train: `52588` examples;
- validation: `6834` examples;
- test: `6829` examples.

Sanity checks после фикса:

- `i need a place to dine in the center thats expensive` => `['restaurant']`;
- `I am looking for a train this Saturday. Can you help?` => `['train']`;
- `I am looking for a moderately priced place to stay while visiting north Cambridge.` => `['hotel']`;
- `Thank you, that is all that I need.` => `['none']`.

## Последние результаты

### Baseline после фикса данных

На исправленном dataset, demo profile:

- `p0` примерно: `val 91-92%`, `test 93.33%`;
- полный docstring: `val 94%`, `test 93.33%`.

То есть validation signal появился: сильная инструкция лучше `p0` на val.

### Последний SRP run

Run:

`runs/20260520T121236`

Параметры:

- train: `50`;
- val: `100`;
- test: `30`;
- candidates: `4`;
- retries: `1`;
- initial prompt: `prompts/p0.txt`.

Время:

- run-dir timestamp: `2026-05-20T12:12:36Z`;
- metadata timestamp: `2026-05-20T12:26:32Z`;
- shell observed completion: `2026-05-20T12:26:43Z`;
- duration: примерно `14 минут` (`13:56` до metadata, `14:07` до shell completion).

Результат:

- baseline candidate iteration 0: `92.0` val;
- best candidate iteration 1 rank 1: `93.0` val, accepted;
- final test score: `93.33`;
- stopped reason: `patience`;
- final instruction words: `119`.

Итоговая инструкция уже не равна `p0`; SRP действительно принял улучшенный
candidate.

Файл:

`runs/20260520T121236/final_instructions.txt`

Суть принятого prompt:

- классифицировать latest user turn;
- не carry old domains blindly;
- использовать context только для reference resolution;
- не добавлять domains из locations/endpoints/context references;
- для taxi обычно возвращать только `taxi`;
- simple acknowledgements/refusals/slot-only replies => `none`.

## Текущее состояние рабочей копии

Изменены файлы:

- `scripts/convert_multiwoz.py`;
- `settings.demo.toml`;
- `src/dspy_domain_train/main.py`;
- `src/dspy_domain_train/settings.py`;
- `src/dspy_domain_train/srp.py`;
- `src/dspy_domain_train/training.py`.

Также есть untracked research docs:

- `2026-05-19-dspy-app-review.md`;
- `2026-05-19-multiwoz-domain-labeling.md`.

## Проверки

До изменения конвертера и регенерации данных проходили:

- `ruff check`;
- `mypy src tests`;
- `python -m unittest discover -s tests`.

После изменения конвертера нужно повторить полный набор перед commit.

## Что осталось сделать

1. Повторить `ruff`, `mypy`, `unittest` после последних изменений.
2. Добавить unit tests для:
   - `initial_prompt_path`;
   - SRP multi-candidate selection;
   - converter cases с phantom domains/closing turns.
3. Сделать progress logging:
   - либо `run.log` через `tee`;
   - либо write-through JSONL внутри SRP per iteration/candidate.
4. Решить, надо ли оставить `max_dev = 100` в `settings.demo.toml`.
   Это честнее для validation signal, но медленнее: последний run занял около
   14 минут.
5. Перед демо показать:
   - `prompts/p0.txt`;
   - `runs/20260520T121236/srp_candidates.json`;
   - `runs/20260520T121236/final_instructions.txt`;
   - `metadata.json` с `test_score = 93.33`.
