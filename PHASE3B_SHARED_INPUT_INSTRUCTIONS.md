# Phase 3B — Shared-Input SwiGLU

## Статус предыдущих веток

Закрыть как основные исследовательские направления:

- Phase 1/2: permutation → 5D tensorization → Tucker;
- balanced permutation v2;
- activation-aware permutation v1;
- Phase 3A shared-left / common-private subspace gate/up/down.

Их результаты сохранить как negative evidence. Не удалять и не переписывать.

Phase 3A завершена с verdict `not-supported`:
после одинакового functional tuning shared-left и common/private уступили independent SVD на held-out validation и confirmation.

Phase 3B — отдельная новая гипотеза.

До отдельного verdict `supported` запрещены Triton/custom CUDA/quantization/full-model replacement.

---

# 1. Мотивация

В SwiGLU:

```
g = Wg x
u = Wu x
m = SiLU(g) ⊙ u
y = Wd m
```

Для Qwen3-1.7B:

- Wg: [6144, 2048]
- Wu: [6144, 2048]
- Wd: [2048, 6144]

Gate и up получают один и тот же input vector `x ∈ R^2048`.

Phase 3A пыталась делить общий левый/intermediate basis размерности 6144.
Pairwise overlap оказался слабым, около 0.17–0.21 при k=1024.

Новая гипотеза:
gate и up могут эффективнее делить общий **правый/input subspace** размерности 2048.

---

# 2. Основная модель Shared-Right Gate/Up

Представить:

```
Wg ≈ Ag @ V.T
Wu ≈ Au @ V.T
```

где:

- V: [2048, r_gu]
- Ag: [6144, r_gu]
- Au: [6144, r_gu]

Тогда inference логически:

```
z = x @ V
gate = z @ Ag.T
up   = z @ Au.T
m = SiLU(gate) * up
```

Shared latent `z` вычисляется один раз.

Down не заставлять делить этот basis.

Использовать independent low-rank factorization:

```
Wd.T ≈ Ld @ Rd
```

где:

- Ld: [6144, r_d]
- Rd: [r_d, 2048]

и:

```
y ≈ (m @ Ld) @ Rd
```

---

# 3. Storage formula

Shared gate/up:

```
V:   2048 * r_gu
Ag:  6144 * r_gu
Au:  6144 * r_gu
------------------
14336 * r_gu
```

Independent down:

```
6144*r_d + 2048*r_d = 8192*r_d
```

Итого:

```
P_shared_right = 14336*r_gu + 8192*r_d + padding
```

Padding считается BF16 storage units и сериализуется как zero tensor, как в Phase 2/3A.

Baseline independent SVD:

```
P_independent = 8192*(r_g + r_u + r_d) + padding
```

Budget должен совпадать точно.

---

# 4. Исходный read-only diagnostic probe

Следующие значения получены вне Phase 3B только как sanity probe.
Они не являются доказательством и должны быть воспроизведены официальным кодом Phase 3B.

## Right gate/up subspace overlap @ k=1024

- layer 0: 0.704691
- layer 7: 0.566463
- layer 14: 0.593530
- layer 21: 0.591302
- layer 27: 0.526979

Это значительно выше shared-left overlap Phase 3A (~0.17–0.21).

Для layer 14 shared right basis rank 1536 сохранял примерно:

- gate energy: 0.916996
- up energy: 0.909770

## Coarse matched-storage functional probe

Против equal-rank independent SVD:

Layer 0:
- high: Δ ≈ -0.1032
- medium: Δ ≈ -0.0310

Layer 7:
- high: Δ ≈ -0.0115
- medium: Δ ≈ +0.0078

Layer 14:
- high: Δ ≈ -0.0244
- medium: Δ ≈ -0.0312

Layer 21:
- high: Δ ≈ +0.0037
- medium: Δ ≈ +0.0036

Layer 27:
- high: Δ ≈ -0.0469
- medium: Δ ≈ -0.0780

Меньше MLP relative error лучше.

После грубого перераспределения independent SVD ranks в пользу down сигнал сохранился на layers 0/14/27, но не на 7/21.

Это только основание для Phase 3B.

---

# 5. Главный strong baseline

Phase 3B не имеет права сравниваться только с equal-rank independent SVD.

Обязательный baseline:

## Optimized Independent SVD

```
Wg ≈ Lg @ Rg
Wu ≈ Lu @ Ru
Wd.T ≈ Ld @ Rd
```

Ranks:

- r_g
- r_u
- r_d

при:

```
8192*(r_g+r_u+r_d) <= exact budget
```

Rank allocation должен быть оптимизирован честно.

Минимум три independent baselines:

1. `independent-equal`
2. `independent-energy`
3. `independent-functional-search`

Shared-right должен победить сильнейший из них, а не только equal.

---

# 6. Independent-energy allocation

Стоимость одной дополнительной rank единицы одинакова для gate/up/down: 8192 parameter units.

Получить singular values каждой матрицы.

Распределять rank единицы greedily туда, где следующая singular value даёт максимальное уменьшение squared Frobenius error.

Это weight-optimal discrete allocation baseline.

Записывать итоговые ranks.

---

# 7. Independent-functional-search

SwiGLU nonlinear, поэтому weight-energy allocation может быть не функционально оптимальным.

Для каждого budget создать заранее ограниченный deterministic grid rank allocations.

Не делать полный O(r²) search.

Пример для total rank R:

- выбрать доли rank для down: 0.20–0.55;
- остаток делить между gate/up несколькими фиксированными ratios;
- включить equal;
- включить energy-optimal;
- включить coarse candidates, похожие на лучшие diagnostic allocations.

Ограничить примерно 12–20 independent allocations на budget.

Все allocations должны быть объявлены до просмотра confirmation results.

Model selection:
- train/search metric разрешён для coarse narrowing;
- validation выбирает finalist;
- confirmation не используется для выбора.

---

# 8. Shared-right analytic basis

Для gate/up общий right basis должен быть оптимальным для joint Frobenius objective.

Использовать:

```
C = Wg.T @ Wg + Wu.T @ Wu
```

Top eigenvectors C:

```
V = top_r_eigenvectors(C)
```

Эквивалентно right singular vectors вертикально stacked matrix:

```
[Wg
 Wu]
```

Далее:

```
Ag = Wg @ V
Au = Wu @ V
```

и reconstruction:

```
Wg_hat = Ag @ V.T
Wu_hat = Au @ V.T
```

Down — ordinary independent SVD.

Не использовать Phase-3A shared-left basis.

---

# 9. Shared-right rank allocation search

Для exact budget B:

```
14336*r_gu + 8192*r_d <= B
```

Добавить zero padding до точного storage equality.

Не выбирать один coarse rank вручную.

Построить deterministic grid по `r_gu`, например 12–20 точек от разумного minimum до maximum.

Для каждого вычислить соответствующий maximum `r_d`.

Включить обязательно diagnostic neighborhoods:

High budget:
- около (r_gu≈1057, r_d≈1517)
- соседние точки

Medium budget:
- около (r_gu≈709, r_d≈1017)
- соседние точки

Но grid должен быть определён до confirmation evaluation.

---

# 10. Основные budgets

Сохранить Phase-3A budgets для прямой сопоставимости.

## High

`B_high = 27,586,944 BF16 units`

MLP compression ≈ 1.368355x.

## Medium

`B_medium = 18,502,368 BF16 units`

MLP compression ≈ 2.040211x.

Не добавлять 3x/4x, пока high/medium не пройдут gate.

---

# 11. Critical controls

Добавить следующие controls.

## A. Independent optimized SVD

Главный baseline.

## B. Shared-right gate/up + independent down

Proposed method.

## C. Separate-right same-rank upper bound

Для выбранного `r_gu, r_d` построить gate/up с двумя независимыми right bases одинакового rank r_gu.

Это использует больше storage и не является matched-budget competitor.

Цель:
понять цену enforced sharing при фиксированных ranks.

Не использовать этот control для заявления о победе по budget.

## D. Shared-right fixed-down control

Для части кандидатов сравнить при одинаковом r_d, чтобы отделить:
- benefit shared gate/up geometry;
- benefit освобождённого бюджета, ушедшего в down.

## E. Optimized-down independent control

Обязателен, чтобы не повторить ошибку сравнения только с equal ranks.

---

# 12. Diagnostics

Для layers:

- 0
- 7
- 14
- 21
- 27

Для k:

- 128
- 256
- 512
- 768
- 1024
- 1280
- 1536
- 1792
- 2048

Посчитать gate/up right-subspace:

- normalized projection overlap;
- principal angles;
- mean squared canonical correlation;
- cumulative singular energy gate/up;
- shared-basis energy capture gate/up.

Использовать numerically reliable path.
Если FP32 Gram и direct FP64 calculation расходятся выше установленного tolerance, сохранять обе диагностики и использовать FP64 reference.

---

# 13. Calibration split

Не использовать весь calibration tensor как одну выборку.

Сохранить или расширить Phase-3A protocol:

- train/search tokens;
- validation tokens;
- independent confirmation tokens.

Предпочтительно reuse exact Phase-3A split semantics для сопоставимости.

Новый WikiText test subset не участвует в rank allocation, factor tuning или checkpoint selection.

---

# 14. Analytic evaluation protocol

Primary sequence:

1. layer 14;
2. high + medium budget;
3. analytic optimized-independent grid;
4. analytic shared-right grid;
5. validation selection;
6. confirmation comparison.

До tuning shared-right должен как минимум быть конкурентен optimized independent SVD.

Если analytic shared-right сильно проигрывает optimized independent SVD на validation и confirmation — остановить Phase 3B без training.

---

# 15. Functional metrics

Использовать те же semantics Phase 3A:

- canonical FP32 MLP replay;
- factors предварительно quantize/save/load как BF16;
- actual full BF16 transformer block replay.

Для train, validation, confirmation отдельно:

- mlp_output_relative_error;
- mlp_output_mse;
- block_output_relative_error;
- block_output_mse.

Также:
- per-matrix Frobenius error;
- aggregate Frobenius error;
- peak VRAM;
- decomposition time;
- evaluation time.

---

# 16. Functional tuning

Если analytic shared-right проходит local gate, разрешено tuning.

Обучать **оба** финалиста с одинаковым allowance:

1. optimized independent SVD;
2. shared-right + independent down.

Все base-model параметры frozen.

## Shared-right trainable parameters

- V
- Ag
- Au
- Ld
- Rd

## Independent trainable parameters

- Lg/Rg
- Lu/Ru
- Ld/Rd

Нельзя давать proposed method больше optimizer steps или больше LR trials.

---

# 17. Tuning objective

Primary:

```
L_mlp = normalized_MSE(MLP_candidate(x), MLP_dense(x))
```

Начать:

```
lambda_mlp = 1
lambda_block = 0
```

Block-loss допускается отдельной заранее записанной secondary configuration только если primary результат близкий.

Не использовать test PPL/logits для training.

---

# 18. Training fairness

Для independent и shared-right:

- одинаковые train tokens;
- одинаковые validation tokens;
- одинаковые confirmation tokens;
- одинаковые optimizer family;
- одинаковое число LR candidates;
- одинаковые maximum steps;
- одинаковая early stopping policy;
- одинаковые training seeds.

Checkpoint selection только validation.

Confirmation полностью unseen для optimizer/model selection.

Минимум два training seeds для selected LR.

Не выбирать лучший seed; report mean + individual results.

---

# 19. Optional orthogonality handling

Shared V аналитически orthonormal при initialization.

Во время functional tuning V может потерять orthogonality.

Сначала разрешить unconstrained tuning.

Отдельный ablation orthogonalized/retracted V разрешён только если:
- unconstrained shared-right близок к independent;
- есть основание считать conditioning проблемой.

Не запускать большой optimizer-method search.

---

# 20. Layer-14 acceptance gate

Для хотя бы одного из high/medium budgets shared-right должен после fair tuning:

1. иметь validation MLP error < tuned optimized-independent;
2. иметь confirmation MLP error < tuned optimized-independent;
3. иметь confirmation block error <= tuned optimized-independent либо быть практически равным в заранее заданном tolerance;
4. выигрывать на обоих training seeds, либо средний выигрыш должен существенно превышать seed variability;
5. не зависеть от использования test data при selection.

Если этого нет — verdict `not-supported` и остановка.

---

# 21. Multi-layer transfer

Только после прохождения layer-14 gate.

Проверить layers:

- 0
- 7
- 21
- 27

Сначала analytic validation/confirmation без tuning.

Если shared-right выигрывает минимум на 2 дополнительных слоях, выполнить tuning только на наиболее информативных transfer layers.

Не считать right-subspace overlap сам по себе transfer success.

---

# 22. Selective PPL

Только после подтверждённого local transfer.

Для каждого выбранного layer:

- dense;
- zero-MLP;
- optimized independent SVD;
- shared-right.

Использовать fixed WikiText-2 subset из предыдущих фаз или новый заранее pinned non-training subset.

Нельзя выбирать rank allocation по PPL.

Success:
shared-right должен быть лучше optimized independent SVD либо как минимум сохранять local advantage без meaningful PPL regression.

---

# 23. Full quality

Full WikiText-2 + downstream запускать только при verdict candidate `supported` до runtime stage.

Не запускать full benchmark для dominated candidate.

---

# 24. Runtime hypothesis

Не реализовывать в Phase 3B, но сохранять формулу будущего direct inference:

Independent gate/up:

```
zg = x @ Vg
zu = x @ Vu
gate = zg @ Ag.T
up   = zu @ Au.T
```

Shared-right:

```
z = x @ V
gate = z @ Ag.T
up   = z @ Au.T
```

Shared-right потенциально экономит:
- один input projection;
- часть factor storage.

Это runtime hypothesis, а не доказанный speedup.

Triton/torch.compile benchmark разрешить только в отдельной Phase 3C после `supported`.

---

# 25. Novelty discipline

Не утверждать academic novelty без отдельного literature review.

Не копировать код ASVD, SVD-LLM, Swift-SVD, AIR, MGLU, Basis Sharing или других compression repositories.

Можно использовать PyTorch/NumPy/SciPy как математическую инфраструктуру.

Если во время работы обнаружена публикация с практически идентичной same-layer gate/up shared-right factorization:
- записать источник;
- продолжить только если есть отличающаяся проверяемая гипотеза;
- не заявлять оригинальность.

---

# 26. Git discipline

Перед новой реализацией:

Current observed HEAD before Phase 3B preparation:
`a76d8405f844040ec7c14ee60ca6fc6b191da182`

Наблюдался untracked:
`task.txt`

Не удалять и не менять его без необходимости.

Перед primary run:
- commit Phase-3B implementation/config/tests;
- tracked working tree должен быть clean;
- записать commit SHA;
- source snapshot обязателен;
- raw append-only history обязателен.

Untracked unrelated file не должен влиять на source hash/run logic.

---

# 27. Tests

Добавить минимум tests:

- shared-right parameter-count formula;
- exact budget + padding;
- joint right basis equivalence to stacked gate/up SVD;
- shared-right reconstruction;
- independent optimized allocation budget;
- energy allocation correctness на small synthetic matrices;
- deterministic functional rank grid;
- fixed-down control;
- BF16 save/load exact replay;
- train/validation/confirmation disjointness;
- base model frozen during tuning;
- shared V actually shared by gate/up;
- independent baseline has separate bases;
- equal training allowance assertions;
- checkpoint chosen only from validation;
- append-only history;
- Phase 1/2/3A protected files unchanged;
- injected OOM restoration;
- real Qwen artifact test.

---

# 28. Outputs

Создать:

- `REPORT_PHASE3B.md`
- `results/phase3b_summary.json`
- `results/phase3b_right_subspace_diagnostics.csv`
- `results/phase3b_candidates.csv`
- `results/phase3b_rank_allocations.csv`
- training curves JSON для tuned finalists
- BF16 factor artifacts
- `VERIFICATION.json`
- append-only global raw rows

---

# 29. REPORT_PHASE3B обязательные ответы

Отчёт должен явно ответить:

1. Насколько совпадают right subspaces gate/up?
2. Воспроизводится ли diagnostic signal Phase-3B официальным pipeline?
3. Shared-right лучше equal independent SVD?
4. Shared-right лучше energy-optimized independent SVD?
5. Shared-right лучше functionally optimized independent rank allocation?
6. Сохраняется ли выигрыш при фиксированном r_d?
7. Что происходит после одинакового functional tuning?
8. Выигрывает ли shared-right на validation и confirmation?
9. На скольких layers переносится эффект?
10. Что показывает selective/full PPL?
11. Есть ли основание переходить к direct factorized runtime/Triton?

Final verdict только:

- supported
- mixed
- not-supported

---

# 30. Stop conditions

Немедленно остановить до Triton, если:

- optimized independent SVD устраняет analytic advantage;
- shared-right проигрывает confirmation после tuning;
- advantage остаётся только на train/validation и исчезает confirmation;
- signal есть только на одном layer и не переносится;
- PPL уничтожает local advantage.

Если verdict `not-supported`, не запускать Phase 3C автоматически.

Если verdict `supported`, также не запускать Phase 3C автоматически — сначала сформировать отчёт и дождаться отдельного решения.
