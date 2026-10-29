# SwiGLU-Fold

> Public **source-only research snapshot**. The historical conclusion remains **not-supported**;
> no speedup or full-model quality improvement is established by this publication.
> See [PUBLICATION.md](PUBLICATION.md) before interpreting the dated reports or running experiments.


Исследовательский проект из `TZ_SwiGLU-Fold.md`. Протокол Phase 1: реальная
`Qwen/Qwen3-1.7B-Base` BF16, только MLP слоя **14 с нуля**, permutation symmetry,
оригинальная HOSVD/Tucker и сравнение identity/random/structure-aware при одинаковом
бюджете параметров. Quantization, обучение и код сторонних compression repositories
не используются.

## Запуск

Из PowerShell в корне проекта:

```powershell
wsl -d Ubuntu -- bash scripts/setup_wsl.sh
wsl -d Ubuntu -- bash scripts/run_wsl.sh --config configs/phase1.yaml --stage all --smoke
wsl -d Ubuntu -- bash scripts/run_wsl.sh --config configs/phase1.yaml --stage all
```

Внутри WSL те же команды начинаются с `bash scripts/…`.
Окружение хранится в `${SWIGLU_VENV:-$HOME/.venvs/swiglu-fold}`, HF model/dataset —
в стандартном пользовательском cache. `requirements-lock.txt` фиксирует установленный
стек CUDA 12.8. NVIDIA driver в WSL приходит от Windows; отдельный driver не ставится.
Модель скачивается автоматически с Hugging Face; Ollama для доступа к этим весам не нужен.

Режимы `--stage env|baseline|sweep|all`. Smoke уменьшает только объём измерений;
проверяет настоящую модель, все три стратегии и один rank. Исследовательский вывод
делает полный `--stage all`, после успешного smoke.
Повторное формирование отчёта без inference:
`bash scripts/run_wsl.sh --stage report --report-run results/runs/<run-id>`.

## Что сохраняется

- `results/experiments.jsonl`: append-only история; lock и fsync каждого record.
- `results/runs/<run-id>/metadata.json`, `raw.jsonl`, `resolved.yaml`, `summary.json`.
- Реальные `layer14.safetensors`, calibration activations, training/test sample indices,
  permutation and signature metadata и BF16 Tucker core/factors каждой конфигурации.
- `source_snapshot.zip`: исходники соответствующего run, даже без Git commit.
- `REPORT_PHASE1.md`: последний полный отчёт; каждый run сохраняет собственный отчёт.

Metadata содержит model/dataset revision SHA, config/seeds, versions, CUDA/driver/GPU,
memory, Git commit (null с объяснением, если commit отсутствует) и source SHA256.
Репродукция использует `resolved.yaml` соответствующего run и зафиксированные зависимости.
Для безопасного продолжения после отдельного сбоя есть `--baseline-run results/runs/<run-id>`:
совпадение модели, данных, runtime и параметров baseline проверяется; новые records сохраняют
ID/time/source SHA измерений. Исходные raw records остаются без изменений.

Baseline: batch 1, prompts 128/512/2048/4096, ровно 128 новых tokens,
10 warmups и 30 measured iterations. CUDA Events + synchronization; сохраняются все
samples, wall time, p50/p95, tokens/s, peak allocated/reserved VRAM.
Power/energy — опциональные coarse total-board measurements; unavailable записывается
явно. Desktop GPU contention и clocks не контролируются экспериментом.

Signatures: нормализация отдельно gate row/up row/down column, три детерминированных
Gaussian projections 2048→32, MiniBatchKMeans в 96D; cluster ordering по PC1 с фиксированным
знаком и index tie-break. Seeds 42/43/44; identity не повторяется, поскольку seed её не меняет.
HOSVD использует eigendecomposition малых Gram matrices unfoldings; FP32, TF32 выключен.
Все пять форм из ТЗ поддержаны и проверены exact round-trip tests; sweep использует одну.

Weight reconstruction errors оценивают FP32 decomposition. Functional error, MSE,
cosine, KL и PPL относятся к восстановленным **сохранённым BF16 core/factors**.
Compression ratios относятся к MLP одного слоя; permutation storage указано отдельно.
Dense reconstruction используется только для первого эксперимента. Это не direct
factorized inference и не доказательство экономии VRAM или ускорения модели.

Calibration: 1024 train tokens WikiText-2 raw. Evaluation: 4096 test tokens, окна
512/stride 256; каждый target учтён ровно один раз. Эти subsets зафиксированы token
hashes. Это предварительные quality metrics; full WikiText-2/downstream и остальные
фазы ТЗ ещё не выполнены. OOM конкретного rank записывается, sweep продолжает остальные.

## Проверки

```powershell
wsl -d Ubuntu -- bash -c '~/.venvs/swiglu-fold/bin/python -m pytest -q'
```

С реальными артефактами конкретного run (внутри WSL):

```bash
~/.venvs/swiglu-fold/bin/python -m pytest -q \
  --layer-artifact results/runs/<run-id>/layer14.safetensors \
  --calibration-artifact results/runs/<run-id>/calibration.safetensors
```

Без аргументов real-artifact tests явно skipped; synthetic tests остаются активны.
FP32/BF16 tolerances заданы до измерений в YAML; превышение завершает run ошибкой.
Логиты полной модели дополнительно проверяются после каждой exact permutation.

Источники инфраструктуры: [Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base),
[config](https://huggingface.co/Qwen/Qwen3-1.7B-Base/raw/main/config.json),
[PyTorch installation](https://pytorch.org/get-started/locally/),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/).

## Phase 2

В том же WSL environment из корня проекта:

```bash
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase2 --smoke
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase2
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase2 --resume-run results/runs/<interrupted-run>
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase2_finalists --local-run results/runs/<complete-run> --smoke
~/.venvs/swiglu-fold/bin/python scripts/check_scoring_precision.py --requests-run results/runs/<smoke-run>
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase2_finalists --local-run results/runs/<complete-run>
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase2 --stage report --report-run results/runs/<complete-run> --full-run results/runs/<full-quality-run>
```

`configs/phase2.yaml`: capacity-constrained auction, отдельные projection/clustering/
permutation seeds, 10 replicas слоя 14, три replicas остальных четырёх слоёв,
точно равные Tucker budgets для intermediate-only/one-hidden/multiple-hidden абляций.
SVD использует тот же allocated storage, остаток дискретного rank явно сохранён
как alignment padding; permutation у SVD нет. Основной диапазон 1.37–3.97×.
Новые `mlp_output_mse` и настоящий `block_output_mse` разделены; MLP metric FP32
после canonical unpermutation, block metric — фактический BF16 forward полного блока.
Старое поле и raw history Phase 1 не меняются.

Local Pareto selection использует group means, а subset PPL измеряется на всех
replicas выбранных групп. Full quality оставляет группы лучше zero-MLP по subset NLL,
выбирает ближайшую к 2× и median replica
по local MLP metric, сравнивает с dense и zero layer14, полный WikiText-2 и полные
HellaSwag/PIQA/ARC Easy/WinoGrande splits. Zero-shot conditional loglikelihood,
без chat template; определения prompts/preprocessing закреплены commit
`d6de81643928d653435c431bae19945d41d32520` lm-evaluation-harness. Это собственный
scorer, а не запуск пакета lm-eval; его batched scoring проверяется отдельным тестом.
Exact dataset revisions находятся в `configs/downstream_pinned.json`, запросы с
token IDs и все predictions сохраняются. OOM уменьшает batch и записывается.

`REPORT_PHASE2.md`, `results/phase2_summary.json` и CSV в каталоге run содержат
bootstrap CI, paired comparisons, matched SVD, zero controls, multi-layer и абляции.
Scorer проверен против full-logit reference: FP32 допуск 5e-5, BF16 numerical
batch/padding scale сохраняется отдельно; scorer guard 0.1 log-prob на токен.
Phase 3 и factorized inference этим этапом не запускаются.

## Phase 3A: Joint SwiGLU Subspace

```bash
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase3a --smoke
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase3a
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase3a --stage report --report-run results/runs/<run-id>
```

`configs/phase3a.yaml` фиксирует два Phase-2 бюджета, shared fractions и ограниченное
functional tuning. Все factors сохраняются BF16 и загружаются до local evaluation.
Первые 1024 calibration tokens делятся 75/25; следующие 1024 используются только
для подтверждения. Validation выбирает checkpoint/LR, WikiText test не обучает
factors. Замороженная base model и одинаковый training allowance для SVD обязательны.
Диагностика пяти слоёв не является доказательством качества. Transfer/PPL/full quality
выполняются только после соответствующих local gates; runtime optimization отдельно.

Phase 3A завершена с `not-supported`: 40 local candidates, включая 18 tuned,
105 subspace diagnostics и 20 spectra; 41 real-artifact tests passed. Shared-only
и actual common/private не победили equally tuned matched-budget SVD на validation
и независимой confirmation. Local transfer и quality stages не запускались по stop
condition. Подробности: `REPORT_PHASE3A.md`, `results/phase3a_summary.json` и
`VERIFICATION.json`. Phase-1/Phase-2 artifacts и raw prefix сохранены.

Canonical correlations проверяются прямой CPU FP64 SVD. Для воспроизведения
сводного отчёта выполненного эксперимента:

```bash
~/.venvs/swiglu-fold/bin/python -m swiglu_fold.phase3a --stage report \
  --report-run results/runs/20261006T094212Z-fa0b586c \
  --diagnostics-run results/runs/20261006T100155Z-d0294b98
```
