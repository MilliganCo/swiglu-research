# REPORT_PHASE2

**Verdict: not-supported.**

Run `20261006T074228Z-2c3f30a0`. Реальный Qwen3-1.7B-Base BF16, RTX 4060 8 GB, WSL2. 755 локальных кандидатов; 28 subset quality rows. Ошибок/OOM: 0.

## Протокол

Layer 14: десять заранее заданных независимых пар projection/clustering seeds и десять random permutations; layers 0/7/21/27: три пары для дешёвой проверки переноса. Identity и SVD детерминированы и измерены один раз. Seeds не выбирались по результату. Порядок групп — label centroids, внутри — PC1 с фиксированным знаком и index tie-break.

Balanced assignment — capacitated auction с epsilon=1e-4 и проверкой primal/dual gap ≤ N·epsilon. Это epsilon-optimal assignment к фиксированным KMeans centroids, а не глобальная оптимальность clustering. Каждый raw record сохраняет точные ёмкости, gap и число bids. v1 не изменён.

Первичная MLP metric: FP32, TF32 off, canonical order, веса восстановлены из сохранённых BF16 factors и приведены к BF16. Полный блок отдельно проигрывается в BF16 с исходными hidden states, attention mask и rotary embeddings. Новые mlp_output_mse и block_output_mse относятся к разным выходам; старое поле Phase 1 не переписывалось.

Calibration: первые 1024 train tokens; subset PPL: первые 4096 test tokens WikiText-2 raw, 512/stride 256, 4095 targets ровно один раз. Model SHA, dataset SHA, token hashes, config, source hash/snapshot и реальные BF16 веса записаны в run.

Бюджет — BF16 storage units: три Tucker core/factors + 6144 int64 permutation entries (24 576 units). SVD использует все доступные units на обычные 2D factors, без permutation; остаток <8192 units записывается отдельным zero padding tensor. Поэтому allocated storage равен точно; математических SVD coefficients немного меньше из-за дискретного rank. А и B/C имеют точно одинаковое число Tucker coefficients плюс permutation, без padding. Compression ratio относится к MLP одного слоя, не всей модели. Dense reconstruction не ускоряет inference.

## Восемь ответов

1. **Лучше v1?** Устойчивое преимущество на основных high-A/medium-A не доказано: оба CI пересекают ноль. На floor-A около 4× есть положительный seed signal, но функциональная ошибка остаётся почти как у zero-MLP.
2. **Лучше random?** На layer 14, high-A и medium-A — да по FP32 local metric: оба CI ниже нуля. Это не устойчивое превосходство одновременно над v1 и SVD и не подтверждённый multi-layer result.
3. **Лучше SVD?** SVD доминирует v2 по обеим local metrics в 49/50 matched configurations. Конфигураций с CI ниже нуля против SVD по обеим metrics: 0. Преимущество одного mean без CI не считается победой.
4. **Перенос по слоям:** CI ниже нуля против обоих ordering controls хотя бы на одной общей конфигурации: [14], 1/5. N=3 на transfer layers не доказывает устойчивое обобщение.
5. **Отличие от zero-MLP:** у layer14 v2 ошибка MLP 0.766 около 1.37× и 0.928 около 2.04× против 1.0 у zero. PPL layer14 v2 не измерялась, поскольку эти кандидаты local-dominated SVD. SVD 2.04× имеет subset PPL 11.78184 против zero 11.77131; поэтому full quality выполнена только для SVD 1.37× (subset PPL 11.66785). Единственный v2 Pareto mean на layer7/floor-A имеет N=3 и PPL почти как zero, включая seed хуже zero.
6. **Польза hidden axes:** для v2 layer14 около 2× B−A/C−A CI пересекают ноль; около 3×/4× B/C хуже A с CI выше нуля. Устойчивый benefit true-5D hidden-axis compression этим search не доказан.
7. **Лучший прошедший local Pareto и zero subset-PPL контроль кандидат layer14:** `L14-svd-high-A-R0`. Compression ratio 1.368355×.
Основная цель около 2× не прошла zero quality floor; выбрана ближайшая выжившая конфигурация 1.37×. Pareto строится по group mean MLP/block error и storage; PPL измерена у всех replicas каждой frontier group, не у лучшего seed.
8. **Phase 3/Triton:** не начинались. V2 Pareto-конфигураций с robust local SVD benefit и subset NLL CI ниже zero-MLP: 0. Переход к Phase 3 оправдан только при supported; при текущем недостаточном результате рекомендовано остановить эту гипотезу.

## Layer 14: paired FP32 MLP error

| Config | Control | N | Δ mean | Δ median | SD(Δ) | Win rate | Bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| high-A | random | 10 | -0.012080 | -0.014172 | 0.007727 | 100% | -0.016485 … -0.007463 |
| high-A | structure-aware | 10 | -0.001309 | -0.002963 | 0.011248 | 60% | -0.007983 … 0.005228 |
| medium-A | random | 10 | -0.004590 | -0.003951 | 0.007354 | 70% | -0.008964 … -0.000371 |
| medium-A | structure-aware | 10 | 0.001599 | 0.004223 | 0.010627 | 30% | -0.005170 … 0.007301 |
| medium-B | random | 10 | -0.001357 | -0.000576 | 0.004699 | 60% | -0.004250 … 0.001215 |
| medium-B | structure-aware | 10 | -0.000624 | 0.002678 | 0.006329 | 30% | -0.004609 … 0.002861 |
| medium-C | random | 10 | 0.000611 | 0.000795 | 0.001816 | 40% | -0.000461 … 0.001687 |
| medium-C | structure-aware | 10 | 0.000214 | 0.000461 | 0.001335 | 40% | -0.000615 … 0.000972 |
| low-A | random | 10 | -0.007047 | -0.006208 | 0.004677 | 100% | -0.009971 … -0.004448 |
| low-A | structure-aware | 10 | -0.001933 | -0.002834 | 0.003459 | 70% | -0.003946 … 0.000169 |
| low-B | random | 10 | -0.003331 | -0.004091 | 0.004975 | 70% | -0.006173 … -0.000415 |
| low-B | structure-aware | 10 | 0.000760 | 0.000462 | 0.005229 | 40% | -0.002354 … 0.003819 |
| low-C | random | 10 | 0.000208 | 0.000238 | 0.000779 | 20% | -0.000281 … 0.000622 |
| low-C | structure-aware | 10 | -0.000072 | 0.000108 | 0.000692 | 40% | -0.000493 … 0.000319 |
| floor-A | random | 10 | -0.005318 | -0.004450 | 0.003553 | 90% | -0.007489 … -0.003308 |
| floor-A | structure-aware | 10 | -0.002364 | -0.001617 | 0.002369 | 90% | -0.003803 … -0.001037 |
| floor-B | random | 10 | -0.000341 | -0.000213 | 0.000812 | 60% | -0.000837 … 0.000107 |
| floor-B | structure-aware | 10 | 0.000327 | 0.001065 | 0.001390 | 30% | -0.000542 … 0.001071 |
| floor-C | random | 10 | -0.000532 | -0.000558 | 0.000847 | 70% | -0.001034 … -0.000041 |
| floor-C | structure-aware | 10 | 0.000217 | 0.000888 | 0.001464 | 30% | -0.000679 … 0.001015 |

Знак: v2−control, меньше лучше. 10 000 bootstrap draws, seed 4001; percentile CI. Все конфигурации и обе metrics сохранены, а не только значимые. Множественные comparisons не корректировались; главные заранее заданные A точки high/medium, остальные — exploratory. CI при N=1 не выдумывается.

## Means: все layers, A около 2×

| Layer | Strategy | N | MLP relative mean | Block relative mean |
|---:|---|---:|---:|---:|
| 14 | zero-mlp | 1 | 1.000000 | 0.074383 |
| 14 | identity | 1 | 0.942863 | 0.070133 |
| 14 | random | 10 | 0.932428 | 0.069358 |
| 14 | structure-aware | 10 | 0.926239 | 0.068899 |
| 14 | structure-aware-balanced | 10 | 0.927838 | 0.069017 |
| 14 | svd | 1 | 0.817584 | 0.060815 |
| 0 | zero-mlp | 1 | 1.000000 | 0.630237 |
| 0 | identity | 1 | 0.927752 | 0.584707 |
| 0 | random | 3 | 0.925307 | 0.583168 |
| 0 | structure-aware | 3 | 0.925332 | 0.583185 |
| 0 | structure-aware-balanced | 3 | 0.925066 | 0.583013 |
| 0 | svd | 1 | 0.692878 | 0.436691 |
| 7 | zero-mlp | 1 | 1.000000 | 0.034835 |
| 7 | identity | 1 | 0.901615 | 0.031408 |
| 7 | random | 3 | 0.904839 | 0.031521 |
| 7 | structure-aware | 3 | 0.895822 | 0.031207 |
| 7 | structure-aware-balanced | 3 | 0.893497 | 0.031126 |
| 7 | svd | 1 | 0.859903 | 0.029953 |
| 21 | zero-mlp | 1 | 1.000000 | 0.283108 |
| 21 | identity | 1 | 0.887658 | 0.251320 |
| 21 | random | 3 | 0.876032 | 0.248017 |
| 21 | structure-aware | 3 | 0.884525 | 0.250422 |
| 21 | structure-aware-balanced | 3 | 0.890046 | 0.251983 |
| 21 | svd | 1 | 0.790191 | 0.223714 |
| 27 | zero-mlp | 1 | 1.000000 | 0.998185 |
| 27 | identity | 1 | 0.907522 | 0.905894 |
| 27 | random | 3 | 0.891815 | 0.890206 |
| 27 | structure-aware | 3 | 0.915427 | 0.913774 |
| 27 | structure-aware-balanced | 3 | 0.941939 | 0.940231 |
| 27 | svd | 1 | 0.573709 | 0.572624 |

## Exact-budget ablation: layer 14, balanced-v2

| Candidate−A | Metric | N | Mean Δ | Win rate | 95% CI |
|---|---|---:|---:|---:|---|
| medium-B | mlp_output_relative_error | 10 | -0.000196 | 50% | [-0.0022041523456573487, 0.001831584423780441] |
| medium-B | block_output_relative_error | 10 | -0.000014 | 50% | [-0.0001636397838592529, 0.00013684345409274094] |
| medium-C | mlp_output_relative_error | 10 | -0.003400 | 80% | [-0.006978577375411988, 0.00047898173332214326] |
| medium-C | block_output_relative_error | 10 | -0.000253 | 80% | [-0.0005192875862121582, 3.5503040999174015e-05] |
| low-B | mlp_output_relative_error | 10 | 0.004191 | 20% | [0.0013691164553165437, 0.007246703058481216] |
| low-B | block_output_relative_error | 10 | 0.000311 | 20% | [0.00010242845863103867, 0.0005377592891454695] |
| low-C | mlp_output_relative_error | 10 | 0.009971 | 0% | [0.008073280304670334, 0.011727714836597443] |
| low-C | block_output_relative_error | 10 | 0.000741 | 0% | [0.0005996261164546013, 0.0008722453564405441] |
| floor-B | mlp_output_relative_error | 10 | 0.007408 | 0% | [0.005458324253559113, 0.00903665080666542] |
| floor-B | block_output_relative_error | 10 | 0.000552 | 0% | [0.00040614377707242965, 0.0006725477427244185] |
| floor-C | mlp_output_relative_error | 10 | 0.004896 | 10% | [0.0029403570294380186, 0.006519802957773209] |
| floor-C | block_output_relative_error | 10 | 0.000364 | 10% | [0.00021794630214571955, 0.00048492938280105577] |

## Activation-aware

Отдельные 10 replicas на layer 14, high-A/medium-A. Seven usage features: gate/SiLU/up/product RMS, product × down-norm RMS, product variance, abs mean. Log floor 1e-12, column z-score, coefficient 0.25; weight signatures unchanged, compact fingerprint 6144×7. Полный token×neuron массив не сохранялся. Это одна заранее выбранная normalization, не exhaustive search.

- high-A: activation−weight v2 mean Δ 0.060772, wins 0%, CI [0.05460447758436203, 0.06718446254730223].
- medium-A: activation−weight v2 mean Δ 0.019343, wins 0%, CI [0.015122755616903304, 0.024317059516906738].

## Selective subset PPL

| Layer | Strategy | Config | N | Mean PPL | Min … max |
|---|---|---|---:|---:|---|
| None | dense | — | 1 | 11.398326 | 11.398326 … 11.398326 |
| 14 | zero-mlp | — | 1 | 11.771310 | 11.771310 … 11.771310 |
| 14 | svd | high-A | 1 | 11.667849 | 11.667849 … 11.667849 |
| 14 | svd | medium-A | 1 | 11.781839 | 11.781839 … 11.781839 |
| 14 | svd | low-A | 1 | 11.880568 | 11.880568 … 11.880568 |
| 14 | svd | floor-A | 1 | 11.847511 | 11.847511 … 11.847511 |
| 0 | zero-mlp | — | 1 | 43.444248 | 43.444248 … 43.444248 |
| 0 | svd | high-A | 1 | 11.376091 | 11.376091 … 11.376091 |
| 0 | svd | medium-A | 1 | 11.785290 | 11.785290 … 11.785290 |
| 0 | svd | low-A | 1 | 14.960597 | 14.960597 … 14.960597 |
| 0 | svd | floor-A | 1 | 16.448148 | 16.448148 … 16.448148 |
| 7 | zero-mlp | — | 1 | 12.054600 | 12.054600 … 12.054600 |
| 7 | svd | high-A | 1 | 11.441903 | 11.441903 … 11.441903 |
| 7 | svd | medium-A | 1 | 11.553890 | 11.553890 … 11.553890 |
| 7 | svd | low-A | 1 | 11.710220 | 11.710220 … 11.710220 |
| 7 | structure-aware-balanced | floor-A | 3 | 12.051598 | 12.014821 … 12.099338 |
| 21 | zero-mlp | — | 1 | 13.511595 | 13.511595 … 13.511595 |
| 21 | svd | high-A | 1 | 11.874156 | 11.874156 … 11.874156 |
| 21 | svd | medium-A | 1 | 12.450381 | 12.450381 … 12.450381 |
| 21 | svd | low-A | 1 | 12.860794 | 12.860794 … 12.860794 |
| 21 | svd | floor-A | 1 | 13.153604 | 13.153604 … 13.153604 |
| 27 | zero-mlp | — | 1 | 19.251851 | 19.251851 … 19.251851 |
| 27 | svd | high-A | 1 | 15.898417 | 15.898417 … 15.898417 |
| 27 | svd | medium-A | 1 | 17.060855 | 17.060855 … 17.060855 |
| 27 | svd | low-A | 1 | 18.316887 | 18.316887 … 18.316887 |
| 27 | svd | floor-A | 1 | 18.689209 | 18.689209 … 18.689209 |

Бюджет выбирается отдельно по каждому слою: ближайшая к 2× локальная Pareto точка, прошедшая свой zero-MLP subset NLL floor. Это не рецепт замены всех слоёв.

| Layer | Strategy | Config | SVD ranks | MLP ratio | Subset PPL |
|---:|---|---|---|---:|---:|
| 14 | svd | high-A | [1123, 1122, 1122] | 1.368355 | 11.667849 |
| 0 | svd | medium-A | [753, 753, 752] | 2.040211 | 11.785290 |
| 7 | svd | medium-A | [753, 753, 752] | 2.040211 | 11.553890 |
| 21 | svd | medium-A | [753, 753, 752] | 2.040211 | 12.450381 |
| 27 | svd | medium-A | [753, 753, 752] | 2.040211 | 17.060855 |

Минимальный subset PPL среди всех слоёв: `L0-svd-high-A-R0`, 11.376091. Это выбор по короткому subset; full quality protocol был заранее ограничен layer14, поэтому глобальный выигрыш на полном benchmark не утверждается.

## Full quality

Run `20261006T080009Z-253c6262`. Все 299 078 test tokens WikiText-2; полные benchmark splits, zero-shot, без chat template. Только SVD 1.37×, dense и zero layer14: v2 не прошёл local/zero quality отбор. Это own scorer с prompts/preprocessing lm-evaluation-harness; не утверждается запуск пакета lm-eval.

| Strategy | Full PPL | NLL | Scored targets |
|---|---:|---:|---:|
| dense | 9.987768 | 2.301361 | 299077 |
| zero-mlp | 10.405098 | 2.342296 | 299077 |
| svd | 10.188067 | 2.321217 | 299077 |

| Task | Metric | N | Dense | Zero-MLP | SVD 1.37× |
|---|---|---:|---:|---:|---:|
| hellaswag | accuracy | 10042 | 49.2830% | 47.1818% | 48.5361% |
| hellaswag | accuracy_character_normalized | 10042 | 66.3812% | 63.4834% | 65.4451% |
| piqa | accuracy | 1838 | 75.7345% | 75.0272% | 75.6257% |
| piqa | accuracy_character_normalized | 1838 | 75.4625% | 75.0272% | 75.5713% |
| arc_easy | accuracy | 2376 | 73.4007% | 72.5168% | 73.8215% |
| arc_easy | accuracy_character_normalized | 2376 | 68.3923% | 66.4562% | 68.2239% |
| winogrande | accuracy | 1267 | 64.3252% | 64.7987% | 64.5620% |

Real-model scorer control: FP32 max per-token difference 0.00000127 (strict guard 5e-5). BF16 batch/full-logit max 0.047629; guard 0.1 после отдельной FP32 проверки. Первоначальный guard 0.02 остановил предыдущий run до benchmark metrics. BF16 padding/LM-head shape влияет на near ties; небольшие изменения accuracy не являются сами по себе доказательством улучшения.

Pinned dataset SHA, gzip requests с точными token IDs, prediction JSONL каждого примера, hashes и source snapshot сохранены в full run. Один детерминированный finalist, без выбора лучшего random seed. В raw этого run унаследованное metadata quality_scope относится к preflight 4096 tokens; фактический full scope задают quality_metrics.full_evaluation, tokens/scored_tokens/examples и full_test_token_sha256. Старые rows не переписаны; этот metadata-only дефект исправлен для будущих runs.

## Проверки и границы

Append-only prefix и защищённые Phase-1 files проверены: True/True. Исходный REPORT_PHASE1.md и прежние JSONL rows неизменны. Source snapshot каждого run сохраняет измеренный код; analysis source SHA записан отдельно, если анализ добавлен позже.

CSV statistics/paired_comparisons/svd_comparisons/ablations/quality и summary.json лежат рядом с raw.jsonl. Все N=10 результаты layer14 и N=3 transfer results включены. OOM handler восстанавливает dense веса в finally; проверки записаны в VERIFICATION.json. Полный pytest: 34 passed. Независимый artifact audit: True; full quality measurements: 15. Отдельный read-only verifier Phase 1 подтвердил 28 BF16 artifacts и 120 прежних generation measurements.

Вывод ограничен одной моделью, одним calibration subset и one-layer replacements. Не измерялись ускорение факторизованного inference, VRAM saving модели и качество совместной замены всех 28 MLP. Triton/custom CUDA/quantization и Phase 3 не запускались.

Источники: [capacitated auction / dual certificate](https://www.mit.edu/~dimitrib/Auction_Survey.pdf), [ordinary SVD](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.svd.html). Compression implementation написана здесь; сторонние compression repositories не копировались.
