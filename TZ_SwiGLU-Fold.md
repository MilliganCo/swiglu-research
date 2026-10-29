# SwiGLU-Fold

Разработать исследовательский проект структурного сжатия open-weight LLM без копирования реализаций существующих compression-проектов.

## Hardware target

Основная тестовая машина:

- NVIDIA GeForce RTX 4060
- 8188 MiB VRAM
- Compute Capability 8.9
- 32 GB system RAM
- WSL2
- Ubuntu 26.04
- CUDA через Windows NVIDIA driver

Разработка и benchmarks должны выполняться в WSL2.

## Base model

Использовать:

`Qwen/Qwen3-1.7B-Base`

Основные параметры:

- hidden_size = 2048
- intermediate_size = 6144
- num_hidden_layers = 28
- BF16
- SwiGLU FFN
- Apache 2.0

Не применять quantization на первом этапе.

## Research hypothesis

У SwiGLU существует точная permutation symmetry intermediate channels.

Для

`y = W_down [SiLU(W_gate x) ⊙ (W_up x)]`

и permutation matrix `P` преобразования

`W_gate' = P W_gate`

`W_up' = P W_up`

`W_down' = W_down P^T`

не изменяют функцию MLP.

Использовать эту свободу для реорганизации intermediate neurons перед многомерной tensorization.

Гипотеза: если перед 4D/5D tensor decomposition сгруппировать структурно сходные SwiGLU intermediate channels, для заданного compression ratio потребуется меньший tensor rank и будет меньше потеря качества, чем при tensorization исходного порядка channels.

## Constraints

Не копировать код TQCompressor, HEAT, Minima или других tensor-compression repositories.

Допустимые инфраструктурные зависимости:

- PyTorch
- Transformers
- Hugging Face Datasets
- Safetensors
- NumPy/SciPy
- scikit-learn
- Triton
- lm-evaluation-harness

Реализовать decomposition, permutation logic, factorized modules и kernels внутри этого repository.

## Phase 1 — reproducible baseline

Создать benchmark framework.

Каждый run должен сохранять JSON с:

- timestamp
- git commit
- HF model revision SHA
- torch version
- transformers version
- CUDA runtime
- NVIDIA driver
- GPU
- dtype
- prompt length
- batch size
- peak VRAM
- latency
- tokens/sec
- power/energy when available
- quality metrics

Baseline:

- BF16
- batch 1
- prompt lengths 128, 512, 2048, 4096
- generate 128 tokens
- 10 warmups
- минимум 30 measured iterations

GPU timing выполнять через CUDA Events с synchronization.

## Phase 2 — exact SwiGLU symmetry

Начать только с transformer layer 14.

Получить:

- `gate_proj.weight`
- `up_proj.weight`
- `down_proj.weight`

Создать random permutation длины 6144.

Применить одинаковую permutation:

```python
gate_new = gate[perm, :]
up_new = up[perm, :]
down_new = down[:, perm]
```

Создать unit test, который подтверждает эквивалентность исходного и переставленного MLP.

Проверять FP32 и BF16.

Тест должен завершаться ошибкой при превышении tolerance.

## Phase 3 — neuron signatures

Для каждого из 6144 intermediate neurons использовать:

- соответствующую строку gate_proj
- соответствующую строку up_proj
- соответствующий столбец down_proj

Нормализовать три компонента отдельно.

Не кластеризовать непосредственно 6144-dimensional vectors.

Создать deterministic random projection:

- gate → 32 dimensions
- up → 32 dimensions
- down → 32 dimensions

Итого signature размерности 96.

Seed должен быть конфигурируемым и записываться в experiment metadata.

## Phase 4 — permutation strategies

Реализовать минимум три режима:

1. identity
2. random
3. structure-aware

Structure-aware v1:

- MiniBatchKMeans над 96D signatures
- сгруппировать intermediate neurons по cluster
- внутри cluster выполнить deterministic ordering по первой principal component

Само применение permutation не должно изменять logits модели сверх numerical tolerance.

## Phase 5 — 5D tensorization

Для weight `[6144, 2048]` поддержать tensorizations, включая:

- `[96, 64, 8, 8, 32]`
- `[192, 32, 8, 8, 32]`
- `[128, 48, 8, 16, 16]`
- `[256, 24, 4, 16, 32]`
- `[384, 16, 4, 8, 64]`

Произведение первых intermediate axes должно быть 6144.

Произведение input axes должно быть 2048.

Для down_proj использовать соответствующее transposed logical tensorization, сохраняя те же intermediate axes.

Tensorization и обратная reconstruction должны иметь отдельные exact unit tests.

## Phase 6 — Tucker/HOSVD

Реализовать собственную HOSVD/Tucker decomposition средствами PyTorch.

Не использовать TensorLy в основной реализации.

Поддерживать отдельный rank для каждой tensor axis.

Для каждой конфигурации вычислять:

- original parameter count
- factorized parameter count
- compression ratio
- relative Frobenius reconstruction error
- max error
- MLP output relative error

Первые эксперименты не требуют быстрого inference.

## Phase 7 — central experiment

При одинаковом tensor shape и одинаковых ranks сравнить:

A. identity ordering<br>
B. random permutation<br>
C. structure-aware SwiGLU permutation

Главная проверяемая гипотеза:

C должен давать меньшую functional error / perplexity при том же parameter budget.

Результаты писать append-only в `results/experiments.jsonl`.

Никогда не перезаписывать предыдущие результаты.

## Phase 8 — model-quality evaluation

Использовать:

- WikiText-2 raw perplexity
- HellaSwag
- PIQA
- ARC Easy
- WinoGrande

Во время search разрешён ограниченный subset.

Финальный кандидат должен проходить полный evaluation.

Также измерять:

- block output MSE
- final logits cosine similarity
- KL divergence относительно dense model

## Phase 9 — sensitivity

После успешного layer 14 повторить эксперимент на:

- layer 0
- layer 7
- layer 14
- layer 21
- layer 27

Определить чувствительность разных глубин модели к compression.

Не предполагать одинаковый rank для всех layers.

## Phase 10 — full model

После sensitivity analysis подобрать layer-specific ranks под общий parameter budget.

Первая глобальная цель:

уменьшить общий объём BF16 weights модели не менее чем на 25% при увеличении WikiText-2 perplexity не более чем на 5%.

Не компрессировать embedding и attention на этом этапе.

## Phase 11 — direct factorized inference

Создать `FactorizedLinear`, который выполняет tensor contractions непосредственно над factors/core.

Запрещено реконструировать dense weight во время inference.

Сравнить:

- dense `torch.nn.Linear`
- reconstructed Tucker
- direct factorized Tucker

Отдельно измерять peak VRAM и latency.

## Phase 12 — GPU optimization

После подтверждения algorithmic benefit создать Triton implementation наиболее дорогой contraction sequence.

Ориентироваться конкретно на NVIDIA RTX 4060 / SM 8.9.

Автоматически исследовать:

- block sizes
- contraction order
- fusion opportunities

Каждая Triton implementation должна иметь correctness test против PyTorch reference.

Нельзя принимать kernel optimization с неправильным output ради скорости.

## Success conditions

Проект считается давшим положительный результат, если при соблюдении quality guardrails достигается хотя бы один результат:

- ≥25% reduction total weight memory
- ≥10% end-to-end inference speedup
- ≥10% latency reduction
- ≥15% reduction energy/token
- улучшение quality при равном или меньшем model footprint

Основной quality guardrail:

- WikiText-2 perplexity degradation ≤5%
- downstream degradation желательно ≤2 absolute percentage points

## Engineering requirements

Использовать:

- type hints
- pytest
- deterministic seeds
- YAML configuration
- JSON/JSONL experiment outputs
- no hidden global state
- no hard-coded absolute paths

Каждый эксперимент должен быть полностью воспроизводимым одной CLI command.

До начала крупных search jobs сначала выполнять smoke test.

При CUDA OOM автоматически завершать конкретную конфигурацию как `oom`, записывать результат и продолжать search, а не терять experiment run.

## First deliverable

Не начинать с полной модели.

Первый deliverable должен содержать:

1. environment validation
2. dense baseline
3. extraction layer 14 MLP
4. proof of exact SwiGLU permutation equivalence
5. identity/random/structure-aware permutations
6. одну 5D tensorization
7. собственную Tucker/HOSVD
8. rank sweep
9. таблицу compression ratio vs reconstruction error
10. сравнение трёх permutation strategies
11. сохранённые raw results

После этого сформировать `REPORT_PHASE1.md` с выводом, подтверждается ли основная гипотеза данными.

Не переходить к Triton до получения положительного или однозначно отрицательного ответа на этот эксперимент.
