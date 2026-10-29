# Phase 3A — Joint SwiGLU Subspace

## Статус предыдущей гипотезы

Phase 2 завершена с verdict `not-supported`.

Ветку

`permutation → 5D tensorization → Tucker`

не продолжать как основной метод.

Причины:

- balanced-v2 не показал устойчивого превосходства над v1;
- перенос между слоями слабый;
- ordinary matched-budget SVD доминировал почти все Tucker candidates;
- true-5D hidden-axis compression не показал устойчивого benefit;
- activation-aware permutation v1 ухудшила результат;
- экстремальные compression regimes приближались к zero-MLP floor.

Phase 3A — это новый алгоритмический pivot.

Не переходить к Triton/custom CUDA/quantization/full-model replacement, пока новая гипотеза не пройдёт matched-budget validation против independent SVD.

---

# 1. Новая гипотеза

Для каждого SwiGLU MLP:

- gate_proj: [6144, 2048]
- up_proj: [6144, 2048]
- down_proj.T: [6144, 2048]

Все три матрицы используют одно и то же intermediate space размерности 6144.

Проверяем, можно ли использовать общий intermediate subspace вместо трёх независимых low-rank subspaces.

Обозначим:

`Wg = gate_proj`
`Wu = up_proj`
`Wd = down_proj.T`

## Shared-only model

```
Wg ≈ U @ Ag
Wu ≈ U @ Au
Wd ≈ U @ Ad
```

где:

- U: [6144, r]
- Ag/Au/Ad: [r, 2048]

Parameter count:

`6144*r + 3*2048*r = 12288*r`

Для сравнения, три independent rank-r SVD требуют:

`3 * (6144*r + 2048*r) = 24576*r`

Поэтому при одинаковом storage shared representation может позволить примерно вдвое больший общий rank.

Это только потенциальное преимущество. Его нужно доказать на реальных весах и functional metrics.

---

# 2. Strong baseline

Главный baseline Phase 3A:

`independent SVD`

Использовать ту же реализацию и budget semantics, которые были проверены в Phase 2.

Не ослаблять baseline.

Для каждого budget сравнивать:

1. dense;
2. zero-MLP;
3. independent SVD;
4. shared-only;
5. common+private;
6. functional-tuned common+private, если предыдущий этап перспективен.

Permutation/Tucker Phase-2 методы можно оставить только как historical reference, но не тратить основной compute budget на новый sweep.

---

# 3. Основные budgets

Начать с двух реально полезных budgets из Phase 2.

Исходный MLP:

`37,748,736 parameters`

## High budget

Около 1.368x compression:

`B_high ≈ 27,586,944 BF16 parameter-units`

Independent SVD baseline Phase 2:
примерно ranks [1123, 1122, 1122].

Shared-only maximum:

`floor(B_high / 12288) = 2245`

`12288 * 2245 = 27,586,560`

Остаток budget учитывать явно.

## Medium budget

Около 2.040x compression:

`B_medium ≈ 18,502,368`

Independent SVD baseline:
примерно ranks [753, 753, 752].

Shared-only maximum:

`floor(B_medium / 12288) = 1505`

`12288 * 1505 = 18,493,440`

Остаток budget учитывать явно.

Не начинать с 3x/4x compression. Phase 2 показала, что качество там быстро приближается к zero-MLP floor.

Добавлять более агрессивные budgets только если новая модель уверенно выигрывает около 1.37x и/или 2.04x.

---

# 4. Stage 0 — subspace diagnostics

До decomposition sweep измерить, насколько вообще совпадают intermediate subspaces gate/up/down.

Для layers:

- 0
- 7
- 14
- 21
- 27

Вычислить singular spectra для:

- Wg
- Wu
- Wd = down.T
- concatenated matrix [Wg | Wu | Wd]

Для k:

- 128
- 256
- 512
- 768
- 1024
- 1536
- 2048

измерить pairwise top-k subspace overlap между individual left singular subspaces.

Минимум метрик:

- principal angles;
- mean squared canonical correlation;
- normalized projection overlap:
  `||Ua.T @ Ub||_F^2 / k`;
- cumulative singular energy;
- energy captured by shared stacked-SVD basis in каждой отдельной матрице.

Сохранить diagnostics в JSON/CSV.

Цель: понять, есть ли фактическая общая геометрия intermediate space.

Diagnostics не являются доказательством качества inference.

---

# 5. Shared-only analytic baseline

Построить:

`M = concat([Wg, Wu, Wd], dim=1)`

Shape:

`[6144, 6144]`

Получить top-r left singular vectors M:

`U_shared`

Для каждой матрицы:

`Ai = U_shared.T @ Wi`

и:

`Wi_hat = U_shared @ Ai`

Это является least-squares optimal shared-left rank-r approximation по суммарной Frobenius objective для фиксированного общего subspace rank.

Не использовать обучение на первом шаге.

Хранить BF16 factors.

Quality/local metrics считать после reconstruction из реально сохранённых BF16 factors, как в Phase 2.

---

# 6. Common + private subspace

Главная новая модель:

```
Wi ≈ Uc @ Ac_i + Ui @ Bi
```

где:

- Uc: [6144, rc] — общий basis;
- Ac_i: [rc, 2048];
- Ui: [6144, ri] — private basis конкретной матрицы;
- Bi: [ri, 2048].

Для трёх матриц parameter count:

`P = 12288*rc + 8192*(rg + ru + rd)`

где rg/ru/rd — private ranks.

Специальные случаи:

- rc = 0 → independent SVD;
- rg=ru=rd=0 → shared-only.

Это позволяет построить непрерывный спектр между baseline SVD и полностью shared model.

---

# 7. Analytic initialization common+private

Не начинать с random training.

Для заданного rc:

1. получить Uc из stacked SVD [Wg | Wu | Wd];
2. Ac_i = Uc.T @ Wi;
3. residual Ri = Wi - Uc @ Ac_i;
4. выполнить SVD residual для каждого i;
5. добавить private ranks ri в пределах matched budget.

Для fixed rc распределять оставшийся private rank budget не обязательно поровну.

Реализовать минимум два allocation режима:

### equal
Private rank budget делится максимально равномерно между gate/up/down.

### energy-greedy
На каждом шаге следующая private rank единица отдаётся той residual matrix, у которой следующий singular value даёт наибольшее уменьшение squared reconstruction error.

Так как стоимость одной private rank единицы одинакова: 8192 параметров, это естественный discrete matched-budget allocation.

Записывать выбранные ranks.

---

# 8. Search common/private split

Для каждого budget исследовать ограниченный набор shared fractions, а не огромный Cartesian product.

Например:

- 0.00
- 0.25
- 0.50
- 0.75
- 1.00

где fraction — доля parameter budget, выделенная shared component.

После coarse sweep добавить local refinement около лучших точек.

Ограничить общее число analytic candidates примерно 10–20 на layer/budget.

Нельзя выбирать результат только по weight Frobenius error.

---

# 9. Metrics

Для каждого candidate считать:

## Weight metrics

Для gate/up/down отдельно и суммарно:

- relative Frobenius error;
- spectral norm error при возможности;
- singular-energy retained.

## Functional local metrics

На фиксированных calibration activations:

- mlp_output_relative_error;
- mlp_output_mse;
- block_output_relative_error;
- block_output_mse.

Canonical-order FP32 functional metric оставить primary local metric.

BF16 factors должны быть сериализованы и replayed перед измерением.

## Quality

Selective WikiText-2 subset только для local Pareto candidates.

Full WikiText-2/downstream — только после сильного local result.

---

# 10. Functional calibration

SVD оптимизирует weight reconstruction, а не нелинейную SwiGLU function.

Если analytic common+private хотя бы близок к independent SVD, создать:

`functional-common-private`

Инициализировать factors аналитическим common+private решением.

Заморозить всю исходную модель.

Обучать только factor parameters выбранного MLP.

Основная objective:

`L_mlp = normalized MSE(MLP_compressed(x), MLP_dense(x))`

Дополнительно разрешено:

`L_block = normalized MSE(Block_compressed(h), Block_dense(h))`

Итоговая objective должна быть конфигурируемой:

`L = lambda_mlp * L_mlp + lambda_block * L_block`

Не добавлять logits/KL на первом training sweep.

---

# 11. Train/validation separation

Нельзя оптимизировать и оценивать на одинаковых activations.

Calibration tokens разделить детерминированно:

- train calibration;
- validation calibration.

Например 75/25 или другой заранее фиксированный split.

Local report должен показывать train и held-out validation metrics.

Не выбирать checkpoint по training loss.

Использовать validation metric для early stopping/model selection.

Отдельный WikiText test subset остаётся полностью unseen для factor training.

---

# 12. Functional training constraints

RTX 4060 8 GB.

Тренировать только один MLP candidate одновременно.

Заморозить base model.

Начать с:

- layer 14;
- high и medium budgets;
- максимум нескольких promising common/private splits.

Не запускать training для десятков dominated analytic candidates.

Сохранять:

- optimizer;
- LR;
- steps;
- validation curve;
- best checkpoint step;
- peak VRAM;
- wall time.

Первый optimizer baseline:
- AdamW;
- без weight decay либо с явно зафиксированным малым значением.

Learning rate искать ограниченно, например 2–3 значения максимум.

Не превращать Phase 3A в hyperparameter lottery.

---

# 13. Multi-layer protocol

Последовательность:

## Step A
Layer 14:
- detailed diagnostics;
- analytic shared-only;
- analytic common+private;
- matched independent SVD;
- functional tuning только если analytic candidate конкурентен.

## Step B
Если есть promising method:
cheap local replay на layers 0, 7, 21, 27.

## Step C
Selective PPL только на Pareto candidates.

Не делать full 28-layer replacement в Phase 3A.

---

# 14. Acceptance criteria

Новая гипотеза считается `supported`, только если существует matched-budget candidate, который:

1. превосходит independent SVD по held-out MLP relative error;
2. не проигрывает independent SVD по full-block error;
3. улучшение воспроизводится более чем на одном layer либо очень велико и статистически устойчиво на primary layer;
4. candidate содержательно лучше zero-MLP floor;
5. selective PPL не уничтожает local advantage;
6. improvement нельзя объяснить только training overfit.

Предпочтительная цель medium budget (~2.04x):

- shared/common+private validation MLP error заметно ниже independent SVD;
- block error также ниже;
- subset PPL лучше independent SVD при том же storage.

`mixed`:
- local functional win есть, но перенос/PPL не подтверждён.

`not-supported`:
- independent SVD остаётся равным или лучше после честного matched-budget functional tuning.

---

# 15. Stop conditions

Остановить новую ветку без Triton, если:

- shared-only сильно хуже SVD;
- common+private после analytic search не приближается к SVD;
- functional tuning улучшает train, но не held-out validation;
- после tuning matched-budget SVD остаётся лучше;
- advantage существует только на одном случайном subset/seed.

Отрицательный результат является нормальным результатом.

---

# 16. Что запрещено

До verdict `supported` не делать:

- Triton;
- custom CUDA;
- INT4/INT8;
- GPTQ/AWQ;
- distillation всей модели;
- pruning;
- full 28-layer replacement;
- speculative decoding;
- marketing speedup claims.

Также не возвращаться к exhaustive Tucker shape search без новой доказуемой гипотезы.

---

# 17. Git/reproducibility

Перед изменениями проверить Git.

Если repository ещё не инициализирован:

1. создать локальный Git repository;
2. убедиться, что model weights, large results и caches исключены через .gitignore;
3. сделать baseline commit текущего verified Phase-2 source/config/docs state.

Каждый новый run должен содержать реальный git commit SHA плюс source snapshot.

Raw history остаётся append-only.

Phase-1/Phase-2 artifacts не переписывать.

---

# 18. Tests

Добавить tests минимум для:

- shared factor parameter count;
- exact budget matching;
- stacked shared SVD reconstruction;
- rc=0 equivalence semantics с private-only representation;
- private residual reconstruction;
- energy-greedy allocation;
- BF16 artifact replay;
- common+private serialization;
- train/validation separation;
- frozen base-model parameters;
- checkpoint selection по validation;
- append-only history;
- backward compatibility Phase 1/2.

Запустить real-artifact tests.

---

# 19. Deliverables

Создать:

- `REPORT_PHASE3A.md`
- `results/phase3a_summary.json`
- `results/phase3a_subspace_diagnostics.csv`
- `results/phase3a_candidates.csv`
- append-only raw JSONL
- saved factor artifacts для Pareto candidates
- train/validation curves для tuned candidates
- verification JSON

REPORT_PHASE3A должен ответить:

1. Насколько похожи subspaces gate/up/down?
2. Shared-only лучше или хуже independent SVD?
3. Какой common/private split лучший?
4. Помогает ли energy-aware private rank allocation?
5. Даёт ли functional tuning held-out improvement?
6. Побеждает ли итоговый метод matched-budget SVD?
7. Переносится ли результат между слоями?
8. Что происходит с WikiText PPL?
9. Стоит ли после этого делать direct factorized inference/Triton?

Final verdict только:

- supported
- mixed
- not-supported

Не начинать runtime optimization автоматически.
