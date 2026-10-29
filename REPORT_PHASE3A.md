# REPORT_PHASE3A

**Verdict: not-supported.**

Run `20261006T094212Z-fa0b586c`, Git `b2ac2d9172a31b2426baae828a180c22e92d2141`. Реальный Qwen3-1.7B-Base BF16, RTX 4060 8 GB, WSL2.

Diagnostics run: `20261006T100155Z-d0294b98`. Canonical correlations уточнены прямой CPU FP64 SVD cross-basis matrix: прежний FP32 GPU Gram/eigvalsh path показал discrepancy до 2.34e-5 между mean squared correlation и projection overlap. BF16 factors, decomposition, training и functional measurements не изменялись; прежние raw rows сохранены.

## Протокол

Два заранее фиксированных бюджета: 27 586 944 и 18 502 368 BF16 units. Весь remainder хранится как zero padding; BF16 tensors сериализованы и загружены перед каждой оценкой. Shared-only ranks 2245/1505; coefficients 27 586 560/18 493 440; padding 384/8928. Compression 1.368355×/2.040211× относится к MLP одного слоя.

Стартовые shared fractions 0/.25/.5/.75/1; equal и energy-greedy распределение residual ranks. Две соседние fractions ±.125 около лучшего nonzero-common кандидата; выбор по validation MLP, не по weight error. rc=0/equal использует прежнюю обычную SVD и прежние ranks [1123,1122,1122]/[753,753,752]. rc=0/energy дополнительно проверяет более сильное распределение независимых ranks.

Calibration: первые 1024 train tokens разделены детерминированно 768/256 (split seed 4242); следующие 1024 train tokens — независимая confirmation, не checkpoint selection. Canonical FP32 MLP на восстановленных BF16 weights — primary. Полный блок реально проигран BF16 с исходными hidden states, attention mask и rotary embeddings; validation/confirmation subsets берутся из его выхода.

Полные block metrics для всех 2048 calibration tokens находятся в полях all_block_* CSV/JSON. В таблицах показаны выходы блока на validation/confirmation tokens после полного context replay; их relative-error denominators отличаются от whole-calibration Phase-2 metrics.

Вся base model заморожена. Только factors одного MLP обучаются AdamW, weight_decay=0, LR 1e-4/3e-4, до 60 steps, batch 32 tokens, validation каждые 10 steps, patience=3. Objective: normalized MLP MSE (lambda_mlp=1, lambda_block=0). Optional actual block term реализован, но отдельный block-loss sweep не проводился. Один selection seed 5101; выбранный LR повторён seed 5107. SVD получил тот же allowance. Train никогда не выбирает checkpoint; WikiText test не участвует в training.

## Девять ответов

1. **Общая геометрия:** normalized projection overlap при k=1024 по всем слоям/парам 0.1655–0.2114. Полные spectra, canonical correlations, principal angles и energy capture сохранены; это не доказательство inference quality.

| Layer | Gate/up overlap | Gate/down overlap | Up/down overlap | Shared energy gate/up/down, k=1024 |
|---:|---:|---:|---:|---|
| 0 | 0.1663 | 0.1655 | 0.1738 | 0.6688/0.4804/0.4124 |
| 7 | 0.1762 | 0.1762 | 0.2114 | 0.6291/0.4420/0.4504 |
| 14 | 0.1793 | 0.1726 | 0.1865 | 0.5467/0.5272/0.5508 |
| 21 | 0.1799 | 0.1781 | 0.1778 | 0.4904/0.5332/0.5065 |
| 27 | 0.1818 | 0.1802 | 0.1991 | 0.5151/0.5443/0.4722 |

2–4. **Shared-only, common/private и greedy:** результаты для обоих бюджетов ниже. Weight-optimal allocation не гарантирует меньшую SwiGLU functional error.

| Budget | Method | rc | Private ranks | Train MLP | Validation MLP | Confirmation MLP | Val block | Confirm block |
|---|---|---:|---|---:|---:|---:|---:|---:|
| high | independent-svd/equal | 0 | [1123, 1122, 1122] | 0.655696 | 0.654771 | 0.663718 | 0.175473 | 0.047105 |
| high | shared-only/equal | 2245 | [0, 0, 0] | 0.662910 | 0.658722 | 0.662082 | 0.176542 | 0.046992 |
| high | common-private/energy-greedy | 1964 | [130, 163, 128] | 0.668772 | 0.665113 | 0.669262 | 0.178245 | 0.047501 |
| high | private-only-energy/energy-greedy | 0 | [1072, 1206, 1089] | 0.653305 | 0.652648 | 0.659143 | 0.174927 | 0.046784 |
| high | functional-independent-svd/equal | 0 | [1123, 1122, 1122] | 0.431850 | 0.523994 | 0.573345 | 0.140432 | 0.040695 |
| high | functional-shared-only/equal | 2245 | [0, 0, 0] | 0.426354 | 0.529963 | 0.586810 | 0.142043 | 0.041652 |
| medium | independent-svd/equal | 0 | [753, 753, 752] | 0.817472 | 0.817914 | 0.825234 | 0.219206 | 0.058569 |
| medium | shared-only/equal | 1505 | [0, 0, 0] | 0.816963 | 0.813480 | 0.815698 | 0.217997 | 0.057893 |
| medium | common-private/equal | 1317 | [95, 94, 94] | 0.820117 | 0.816801 | 0.819474 | 0.218890 | 0.058159 |
| medium | private-only-energy/energy-greedy | 0 | [711, 820, 727] | 0.814187 | 0.814473 | 0.821411 | 0.218281 | 0.058297 |
| medium | functional-independent-svd/equal | 0 | [753, 753, 752] | 0.541040 | 0.670271 | 0.733907 | 0.179649 | 0.052089 |
| medium | functional-shared-only/equal | 1505 | [0, 0, 0] | 0.565012 | 0.675096 | 0.734726 | 0.180924 | 0.052145 |

high: shared−SVD validation Δ=+0.003951; лучший actual common/private split f=0.875, allocation=energy-greedy; common/private−shared Δ=+0.006391. Greedy меньше equal по validation MLP в 4/4 splits; средняя descriptive Δ=-0.000886. Это сравнение фиксированных splits, не seed CI.
medium: shared−SVD validation Δ=-0.004434; лучший actual common/private split f=0.875, allocation=equal; common/private−shared Δ=+0.003321. Greedy меньше equal по validation MLP в 2/4 splits; средняя descriptive Δ=-0.000079. Это сравнение фиксированных splits, не seed CI.

5. **Held-out improvement от tuning:** все training/validation curves и оба seeds сохранены. Таблица ниже показывает каждый run, включая LR, который не выбран; summary/CSV содержат все analytic candidates.

| Budget | Method | LR | Seed | Best step | Train | Validation | Confirmation | Confirm block |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| high | functional-independent-svd | 0.0001 | 5101 | 60 | 0.430163 | 0.526435 | 0.575197 | 0.040827 |
| high | functional-independent-svd | 0.0003 | 5101 | 60 | 0.450418 | 0.551782 | 0.599751 | 0.042569 |
| high | functional-independent-svd | 0.0001 | 5107 | 60 | 0.431850 | 0.523994 | 0.573345 | 0.040695 |
| high | functional-shared-only | 0.0001 | 5101 | 60 | 0.428295 | 0.533535 | 0.588656 | 0.041783 |
| high | functional-shared-only | 0.0003 | 5101 | 60 | 0.508989 | 0.611661 | 0.663955 | 0.047123 |
| high | functional-shared-only | 0.0001 | 5107 | 60 | 0.426354 | 0.529963 | 0.586810 | 0.041652 |
| medium | functional-independent-svd | 0.0001 | 5101 | 60 | 0.596806 | 0.689566 | 0.736876 | 0.052300 |
| medium | functional-independent-svd | 0.0003 | 5101 | 60 | 0.537875 | 0.674375 | 0.730684 | 0.051860 |
| medium | functional-independent-svd | 0.0003 | 5107 | 60 | 0.541040 | 0.670271 | 0.733907 | 0.052089 |
| medium | functional-shared-only | 0.0001 | 5101 | 60 | 0.565154 | 0.677355 | 0.735976 | 0.052234 |
| medium | functional-shared-only | 0.0003 | 5101 | 60 | 0.557274 | 0.695422 | 0.757686 | 0.053777 |
| medium | functional-shared-only | 0.0001 | 5107 | 60 | 0.565012 | 0.675096 | 0.734726 | 0.052145 |
| high | functional-common-private | 0.0001 | 5101 | 60 | 0.433884 | 0.539416 | 0.594628 | 0.042205 |
| high | functional-common-private | 0.0003 | 5101 | 60 | 0.508951 | 0.614857 | 0.664867 | 0.047190 |
| high | functional-common-private | 0.0001 | 5107 | 60 | 0.432897 | 0.536719 | 0.593186 | 0.042102 |
| medium | functional-common-private | 0.0001 | 5101 | 60 | 0.570125 | 0.681673 | 0.739751 | 0.052502 |
| medium | functional-common-private | 0.0003 | 5101 | 60 | 0.556565 | 0.694352 | 0.758876 | 0.053861 |
| medium | functional-common-private | 0.0001 | 5107 | 60 | 0.569349 | 0.678807 | 0.737969 | 0.052377 |

6. **Победа над matched SVD:** подтверждённых winners по validation и отдельной confirmation, с не худшим block и обоими training seeds: 0. Сравнение включает tuned SVD, а не только исходную weight-optimal SVD.
Ниже усреднены оба заранее заданных seeds при validation-selected LR, без выбора лучшего seed.

| Budget | Method | N | Val MLP mean ± sample SD | Confirmation MLP mean | Confirmation block mean |
|---|---|---:|---:|---:|---:|
| high | functional-independent-svd | 2 | 0.525215 ± 0.001727 | 0.574271 | 0.040761 |
| high | functional-shared-only | 2 | 0.531749 ± 0.002526 | 0.587733 | 0.041718 |
| medium | functional-independent-svd | 2 | 0.672323 ± 0.002902 | 0.732296 | 0.051975 |
| medium | functional-shared-only | 2 | 0.676226 ± 0.001597 | 0.735351 | 0.052190 |
| high | functional-common-private | 2 | 0.538068 ± 0.001907 | 0.593907 | 0.042154 |
| medium | functional-common-private | 2 | 0.680240 ± 0.002027 | 0.738860 | 0.052440 |

Tuning улучшает held-out functional error всех трёх representations; после одинакового allowance SVD лучше shared-only и actual common/private на обоих seeds обоих budgets. Малый analytic medium gain shared-only не сохраняется против tuned SVD. N=2 не даёт сильной статистической оценки произвольных training seeds; вывод ограничен этим bounded protocol.

7. **Перенос:** диагностика выполнена для 0/7/14/21/27. Local transfer не запускался: layer14 не прошёл gate.
8. **WikiText PPL:** selective/full PPL и downstream не запускались при отсутствии подтверждённого local SVD win. Phase-2 quality остаётся историческим результатом и не приписывается Phase 3A.
9. **Direct inference/Triton:** не запускать по текущим данным. Локальное факторное сжатие не измеряет runtime speedup, footprint всей модели или эффект совместной замены 28 MLP.

## Проверки и артефакты

40 успешных local candidates; 105 pair/k diagnostics; 20 spectra. Failed/OOM rows: 0. Старые 352 files и append-only prefix: True/True. BF16 factor SHA, точный storage и global raw correspondence: True/True.

Полный real-artifact pytest: 41 passed, 0 skipped; 32.29 seconds. Проверены actual saved Pareto replay, validation checkpoint selection, frozen base gradients, optional block-loss gradient path и восстановление весов при injected residual-decomposition OOM.

Все decomposition/evaluation/training wall times и peak allocated VRAM находятся в CSV/JSON. decomposition_seconds включает загрузку shared basis и residual SVD; построение shared basis и subspace diagnostics имеют отдельный stage-0 timer. Factor packing/save отдельно не timed. evaluation_seconds включает factor load/reconstruction, local replay и Frobenius metrics. Weight energy explained означает 1−squared relative Frobenius error после BF16 replay; singular-energy capture отдельно сохранён в diagnostics. Spectral-norm error не измерялся. Torch peak относится к процессу; desktop GPU contention не контролировался. Git dirty flag сохранён как наблюдался; измеренный source сохранён в ZIP вместе с реальным commit SHA. Присутствовавший до реализации пустой task.txt не изменялся.

Ранее выполненные primary measurements сохранены в `20261006T093715Z-85642c0c`. Продолжение добавило qualifying actual common/private tuning; config, capture/token hashes, numerical AST и исходные ops проверены до reuse. Каждая reused row сохраняет measurement run/time/source SHA; исторические rows и artifacts не переписаны.

Источники матричных операций: [PyTorch eigh](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.eigh.html), [PyTorch SVD](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.svd.html). Shared left basis получен из sum WiWi.T, эквивалентного MM.T. Factor conventions Phase 2 сохранены.
