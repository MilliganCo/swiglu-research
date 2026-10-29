# Phase 2 — доказательство алгоритмического преимущества

Цель Phase 2: отделить реальный структурный эффект SwiGLU-Fold от численного шума, устойчивости residual path и частного случая одного слоя.

Не переходить к Triton, full-model compression или quantization, пока не будет убедительно показано, что proposed ordering/decomposition устойчиво превосходит сильные matched-budget baselines.

## Исходные факты из Phase 1

Base model:
- Qwen/Qwen3-1.7B-Base
- BF16
- слой 14 исследован полностью
- shape: [96, 64, 8, 8, 32]

Phase 1 подтвердил:
- exact joint permutation symmetry gate/up/down;
- 25 tests проходят на synthetic и real Qwen artifacts;
- structure-aware v1 даёт небольшой положительный signal;
- FP32 post-hoc контроль подтвердил, что signal не является следствием BF16 permutation order;
- полезный region находится примерно около 1.37x–2.0x MLP compression;
- экстремальные 9.4x и 62x candidates практически эквивалентны выключенному MLP по PPL;
- zero layer-14 MLP даёт PPL ≈ 11.77131 против dense ≈ 11.39833;
- MiniBatchKMeans v1 формирует крайне несбалансированные clusters при tensor grid 96×64;
- простой balanced 96 groups × 64 neurons probe улучшил MLP relative error относительно v1 на seed 42.

Observed balanced probe:
- ranks [64,48,8,8,32]:
  v1 ≈ 0.926270
  balanced ≈ 0.923181
- ranks [80,56,8,8,32]:
  v1 ≈ 0.768128
  balanced ≈ 0.762546

Это только diagnostic probe, не доказанный результат.

## Главная гипотеза Phase 2

Если intermediate neurons распределять в capacity-constrained группы, строго совпадающие с tensor axes, то decomposition использует локальную структуру лучше, чем:

1. identity ordering;
2. random ordering;
3. unbalanced structure-aware v1;
4. обычный matched-budget low-rank matrix baseline.

## 1. Structure-aware v2-balanced

Создать новый strategy:

`structure-aware-balanced`

Не изменять и не удалять v1.

Для shape [96,64,8,8,32]:

- создать 96 групп;
- в каждой группе должно быть ровно 64 intermediate neurons;
- каждый neuron должен принадлежать ровно одной группе;
- порядок групп deterministic;
- ordering внутри группы deterministic.

Первая реализация может использовать:

1. текущие 96D signatures;
2. обычный KMeans только для получения centroids;
3. capacity-constrained assignment neuron → centroid;
4. capacity каждой группы = 64;
5. deterministic PC1 ordering внутри каждой группы.

Assignment должен быть отдельной функцией и покрыт unit tests.

Нельзя считать naive greedy assignment финальным решением. Он является baseline v2. После него разрешено исследовать улучшенные balanced assignment algorithms.

## 2. Разделить источники случайности

Нельзя использовать один `seed` сразу для всех операций.

Добавить в config:

```yaml
seeds:
  projection: [...]
  clustering: [...]
  permutation: [...]
```

Минимально отделить:

- projection_seed
- clustering_seed
- random_permutation_seed

Для structure-aware methods random permutation seed не использовать.

Все seeds писать в raw results.

## 3. Сильные baselines

Добавить обязательные matched-budget baselines.

### A. Zero-MLP baseline

Для каждого исследуемого слоя измерить quality при полностью выключенном MLP branch.

Сохранять:

- local block/MLP metrics
- PPL subset
- full quality только для финальных сравнений

Это sanity floor.

### B. Dense identity

Исходная модель/слой.

### C. Random permutation + same Tucker ranks

Существующий baseline.

### D. Structure-aware v1

Существующий baseline.

### E. Structure-aware-balanced v2

Новый кандидат.

### F. Matrix low-rank baseline

Для каждой из gate/up/down построить обычное 2D low-rank approximation с matched total parameter budget.

Не сравнивать модели с разным parameter budget.

Использовать truncated SVD / randomized SVD только как baseline, не как часть proposed method.

Budget должен учитывать:
- factors;
- permutation storage;
- любые дополнительные tensors, необходимые inference.

## 4. Уточнить naming metrics

Текущее поле `block_output_mse` фактически измеряет MLP output.

Переименовать в:
- `mlp_output_mse`

Добавить отдельный настоящий:
- `block_output_mse`
- `block_output_relative_error`

Для этого сравнивать полный transformer block output до и после замены MLP.

Старые raw results не переписывать.

## 5. Убрать экстремальные meaningless ranks из main search

Не тратить основной compute budget на точки, которые практически эквивалентны zero-MLP.

Основной search region:
- примерно 1.25x–4x MLP compression.

62x можно оставить только как diagnostic endpoint.

9x также не считать победой без доказательства, что candidate существенно лучше zero-MLP baseline.

## 6. Multi-seed validation

Для layer 14:

- минимум 10 независимых projection/clustering seeds для v1/v2;
- минимум 10 random permutations;
- одинаковые ranks;
- одинаковый calibration dataset;
- одинаковый evaluation dataset.

Для каждой пары v1/v2 использовать paired design там, где это возможно.

Считать:
- mean
- median
- sample SD
- 95% bootstrap CI
- paired win rate
- paired delta

Не объявлять improvement только по среднему.

## 7. Multi-layer validation

После layer 14 проверить:

- layer 0
- layer 7
- layer 14
- layer 21
- layer 27

Сначала только cheap local metrics.

Для каждого слоя:
- dense;
- zero MLP;
- random;
- v1;
- balanced-v2;
- SVD baseline.

Только Pareto candidates переходят к PPL.

Не предполагать одинаковые ranks для всех layers.

## 8. Activation-aware extension

После weight-only balanced-v2 создать отдельный experimental strategy:

`structure-aware-activation`

Не заменять им v2.

Использовать calibration activations.

Для каждого intermediate neuron оценить статистику, связанную с реальным использованием:

- gate preactivation;
- SiLU(gate);
- up activation;
- product SiLU(gate) * up;
- RMS/variance;
- optionally contribution proxy через down column norm.

Не хранить огромный activation tensor без необходимости.

Сформировать compact fingerprint и сравнить с weight-only signatures.

Все normalization decisions должны быть явными и конфигурируемыми.

## 9. Проверка роли 5D

Phase 1 пока не доказал преимущество всех пяти dimensions, поскольку лучшие ranks оставляли hidden axes [8,8,32] full-rank.

Добавить ablation:

A. compress only intermediate axes
B. compress intermediate + one hidden axis
C. compress multiple hidden axes

Matched total parameter budget.

Цель: определить, даёт ли настоящий 5D factorization дополнительный Pareto benefit или практический эффект происходит в основном от structured factorization intermediate dimension.

Отрицательный результат допустим и должен быть честно зафиксирован.

## 10. Quality protocol

Cheap loop:
- calibration activations
- MLP relative error
- MLP MSE
- full-block relative error
- block MSE

Mid-cost:
- fixed WikiText-2 subset

Finalists:
- full WikiText-2
- HellaSwag
- PIQA
- ARC Easy
- WinoGrande

Не прогонять full benchmark для явно dominated candidates.

## 11. Statistical discipline

Raw results append-only.

Каждый experiment должен содержать:
- algorithm version
- layer
- shape
- ranks
- effective parameter count
- effective compression ratio
- all seeds
- model SHA
- dataset SHA
- source SHA
- calibration token SHA
- evaluation token SHA

Нельзя выбирать лучший seed и сообщать только его.

## 12. Acceptance criteria Phase 2

Phase 2 считается положительной, если balanced-v2:

1. устойчиво превосходит random и v1 по local functional metric;
2. преимущество наблюдается более чем на одном layer;
3. 95% CI / paired analysis не указывает на полностью шумовой эффект;
4. candidate лучше zero-MLP baseline содержательно, а не только по parameter count;
5. не проигрывает matched-budget SVD baseline по всем полезным метрикам;
6. имеет хотя бы одну Pareto-optimal configuration по quality vs parameter count.

Предпочтительная цель:
- около 2x compression MLP matrices;
- заметное functional improvement против identity/random;
- PPL degradation существенно меньше zero-MLP;
- сигнал переносится минимум на 3 из 5 исследованных layers.

## 13. Что запрещено в Phase 2

До завершения algorithmic validation не делать:
- Triton;
- custom CUDA;
- INT4/INT8;
- GPTQ/AWQ;
- LoRA;
- distillation;
- full-model replacement всех 28 layers;
- speculative decoding.

Сначала доказать алгоритм.

## Deliverable

Создать:

- `REPORT_PHASE2.md`
- `results/phase2_summary.json`
- append-only raw rows
- таблицы paired comparisons
- zero-MLP comparisons
- SVD matched-budget comparisons
- multi-layer results
- ablation intermediate-only vs true-5D
- explicit verdict

Финальный verdict должен быть одним из:

- supported
- mixed
- not-supported

Без маркетинговых формулировок.
