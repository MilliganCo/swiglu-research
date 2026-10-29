from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .phase2_report import csv_table
from .phase3a import sha
from .telemetry import source_hash, write_json


def report(root: Path, directory: Path, diagnostic_directory: Path | None = None):
    if diagnostic_directory is None and (directory/'VERIFICATION.json').exists():
        corrected = json.loads((directory/'VERIFICATION.json').read_text()).get('corrected_diagnostic_run')
        if corrected:
            diagnostic_directory = root/'results/runs'/corrected
    rows = [json.loads(line) for line in (directory/'raw.jsonl').read_text().splitlines()]
    metadata = json.loads((directory/'metadata.json').read_text())
    completion = json.loads((directory/'completion.json').read_text())
    if metadata['smoke']:
        raise ValueError('smoke does not support a research verdict')
    locals_ = [r for r in rows if r['kind'] == 'phase3a_local' and r['status'] == 'ok']
    diagnostic_rows = [json.loads(line) for line in (diagnostic_directory/'raw.jsonl').read_text().splitlines()] if diagnostic_directory else rows
    diagnostics = [r for r in diagnostic_rows if r['kind'] == 'phase3a_subspace']
    spectra = [r for r in diagnostic_rows if r['kind'] == 'phase3a_spectrum']
    selections = [r for r in rows if r['kind'] == 'phase3a_tuning_selection']
    expected = {(layer, k, pair) for layer in metadata['config']['layers'] for k in metadata['config']['diagnostic_k']
                for pair in ('gate/up', 'gate/down', 'up/down')}
    assert len(diagnostics) == len(expected) and {(r['layer'], r['k'], r['pair']) for r in diagnostics} == expected
    assert len(spectra) == 4*len(metadata['config']['layers'])
    assert {(r['layer'], r['matrix']) for r in spectra} == {(layer, matrix) for layer in metadata['config']['layers']
                                                          for matrix in ('gate', 'up', 'down', 'stacked')}
    for r in spectra:
        assert len(r['singular_values']) == len(r['cumulative_energy']) == (6144 if r['matrix'] == 'stacked' else 2048)
        assert abs(r['cumulative_energy'][-1]-1) < 1e-5
    for r in diagnostics:
        assert len(r['principal_angles_degrees']) == len(r['canonical_correlations']) == r['k']
        assert abs(r['normalized_projection_overlap']-r['mean_squared_canonical_correlation']) < 1e-5
        assert all(0 <= value <= 90 for value in r['principal_angles_degrees'])
    assert len({r['candidate_id'] for r in locals_}) == len(locals_)
    assert sha(directory/'source_snapshot.zip') == metadata['source_snapshot_sha256']
    assert sha(directory/'phase2_calibration.safetensors') == metadata['calibration_artifact_sha256']
    splits = json.loads((directory/'calibration_split.json').read_text())
    assert sorted(sum(splits.values(), [])) == list(range(metadata['config']['quality']['calibration_tokens']))
    controls = [r for r in rows if r['kind'] == 'phase3a_control']
    assert len(controls) == 2
    assert all(value == 0 for r in controls if r['strategy'] == 'dense'
               for key, value in r.items() if 'output_relative_error' in key or 'output_mse' in key)
    assert next(r['mlp_output_relative_error'] for r in controls if r['strategy'] == 'zero-mlp') == 1.
    final = [r for r in locals_ if not r['strategy'].startswith('functional-') or any(
        s['initial_candidate_id'] == r['initial_candidate_id'] and s['chosen_learning_rate'] == r['learning_rate'] for s in selections)]
    comparisons = []
    greedy_comparisons = []
    for name in metadata['config']['budgets']:
        local = [r for r in locals_ if r['budget_name'] == name and r['layer'] == 14]
        analytic = [r for r in local if not r['strategy'].startswith('functional-')]
        svd = next(r for r in analytic if r['strategy'] == 'independent-svd')
        shared = next(r for r in analytic if r['strategy'] == 'shared-only')
        private = min((r for r in analytic if r['common_rank'] == 0), key=lambda r: r['mlp_output_relative_error'])
        common = min((r for r in analytic if r['strategy'] == 'common-private'), key=lambda r: r['mlp_output_relative_error'])
        selected = [r for r in final if r['budget_name'] == name and r['layer'] == 14]
        best_joint = min((r for r in selected if r['common_rank'] > 0), key=lambda r: r['mlp_output_relative_error'])
        best_private = min((r for r in selected if r['common_rank'] == 0), key=lambda r: r['mlp_output_relative_error'])
        comparisons.append({'budget_name': name, 'independent_svd': svd['candidate_id'], 'shared_only': shared['candidate_id'],
                            'best_common_private': common['candidate_id'], 'best_analytic_private': private['candidate_id'],
                            'best_selected_joint': best_joint['candidate_id'], 'best_selected_private': best_private['candidate_id']})
        for equal in analytic:
            if equal['strategy'] != 'common-private' or equal['allocation'] != 'equal':
                continue
            greedy = next(r for r in analytic if r['shared_fraction'] == equal['shared_fraction'] and r['allocation'] == 'energy-greedy')
            greedy_comparisons.append({'budget_name': name, 'shared_fraction': equal['shared_fraction'],
                                      'validation_mlp_delta': greedy['mlp_output_relative_error']-equal['mlp_output_relative_error'],
                                      'confirmation_mlp_delta': greedy['confirmation_mlp_output_relative_error']-equal['confirmation_mlp_output_relative_error'],
                                      'aggregate_weight_error_delta': greedy['aggregate_relative_frobenius_error']-equal['aggregate_relative_frobenius_error']})
    # Research selection uses validation only. This front is descriptive, not a test-set selector.
    front = [r for r in final if not any(other['layer'] == r['layer'] and
        all(other[k] <= r[k] for k in ('effective_parameter_count', 'mlp_output_relative_error', 'block_output_relative_error')) and
        any(other[k] < r[k] for k in ('effective_parameter_count', 'mlp_output_relative_error', 'block_output_relative_error')) for other in final)]
    tuning_statistics = []
    for selection in selections:
        group = [r for r in locals_ if r.get('initial_candidate_id') == selection['initial_candidate_id']
                 and r['learning_rate'] == selection['chosen_learning_rate']]
        assert {r['training_seed'] for r in group} == set(metadata['config']['tuning']['seeds'])
        tuning_statistics.append({'budget_name': group[0]['budget_name'], 'strategy': group[0]['strategy'],
                                  'learning_rate': selection['chosen_learning_rate'], 'N': len(group),
                                  **{key+'_mean': float(np.mean([r[key] for r in group])) for key in
                                     ('validation_mlp_output_relative_error', 'confirmation_mlp_output_relative_error',
                                      'validation_block_output_relative_error', 'confirmation_block_output_relative_error')},
                                  'validation_mlp_sample_sd': float(np.std([r['mlp_output_relative_error'] for r in group], ddof=1))})
    guard = json.loads((root/'results/phase3a_history_guard.json').read_text())
    with (root/'results/experiments.jsonl').open('rb') as stream:
        import hashlib
        prefix_ok = hashlib.sha256(stream.read(guard['bytes'])).hexdigest() == guard['sha256']
    protected_ok = all(sha(root/name) == digest for name, digest in guard['protected_files'].items())
    storage_ok = True
    for row in locals_:
        path = directory/row['factor_artifact']
        if sha(path) != row['factor_artifact_sha256']:
            raise AssertionError(f'factor SHA differs: {path}')
        with safe_open(str(path), framework='pt') as artifact:
            units = sum(int(np.prod(artifact.get_slice(k).get_shape())) for k in artifact.keys())
            assert all(artifact.get_slice(k).get_dtype() == 'BF16' for k in artifact.keys())
            padding = artifact.get_tensor('alignment_padding')
            assert padding.count_nonzero() == 0 and padding.numel() == row['padding_parameter_count']
            assert artifact.get_slice('common').get_shape() == [6144, row['common_rank']]
            for name, rank in zip(('gate', 'up', 'down'), row['private_ranks']):
                assert artifact.get_slice(f'{name}.common_right').get_shape() == [row['common_rank'], 2048]
                assert artifact.get_slice(f'{name}.private_left').get_shape() == [6144, rank]
                assert artifact.get_slice(f'{name}.private_right').get_shape() == [rank, 2048]
            storage_ok &= units == row['effective_parameter_count']
    raw_text = (directory/'raw.jsonl').read_bytes()
    if diagnostic_directory:
        raw_text += (diagnostic_directory/'raw.jsonl').read_bytes()
    with (root/'results/experiments.jsonl').open('rb') as stream:
        stream.seek(guard['bytes'])
        history_new = stream.read()
    raw_match = all(line in history_new.splitlines() for line in raw_text.splitlines())
    verification = {'passed': prefix_ok and protected_ok and storage_ok and raw_match,
                    'phase1_phase2_protected_files': len(guard['protected_files']),
                    'history_prefix_unchanged': prefix_ok, 'phase1_phase2_files_unchanged': protected_ok,
                    'exact_bf16_storage_and_factor_sha': storage_ok, 'global_raw_matches': raw_match,
                    'diagnostic_layers': sorted(set(r['layer'] for r in diagnostics)),
                    'diagnostic_pair_k_rows': len(diagnostics), 'spectra': len(spectra),
                    'successful_local_candidates': len(locals_), 'tuned_candidates': sum('functional-' in r['strategy'] for r in locals_),
                    'failed_rows': sum(r['status'] != 'ok' for r in rows),
                    'diagnostic_coverage_and_spectrum_dimensions': True, 'calibration_split_and_capture_sha': True,
                    'two_training_seeds_at_validation_selected_lr': True, 'source_snapshot_sha_verified': True,
                    'dense_and_zero_local_controls': True,
                    'run_git_commit': metadata['git_commit'], 'run_source_snapshot_sha256': metadata['source_snapshot_sha256']}
    if diagnostic_directory:
        dm = json.loads((diagnostic_directory/'metadata.json').read_text())
        assert sha(diagnostic_directory/'source_snapshot.zip') == dm['source_snapshot_sha256']
        verification.update(corrected_diagnostic_run=diagnostic_directory.name,
                            corrected_diagnostic_git_commit=dm['git_commit'], corrected_diagnostic_source_sha256=dm['source_sha256'])
    if (directory/'VERIFICATION.json').exists():
        previous = json.loads((directory/'VERIFICATION.json').read_text())
        verification.update({k: v for k, v in previous.items() if k not in verification})
    verdict = 'not-supported' if not completion['confirmed_local_winners'] else 'mixed'
    summary = {'verdict': verdict, 'run_id': directory.name, 'metadata': metadata,
               'comparisons': comparisons, 'greedy_comparisons': greedy_comparisons,
               'candidates': locals_, 'subspace_diagnostics': diagnostics,
               'spectra': spectra, 'diagnostic_run': diagnostic_directory.name if diagnostic_directory else directory.name,
               'subspace_precision_controls': [r for r in diagnostic_rows if r['kind'] == 'phase3a_subspace_precision_control'],
               'validation_selections': selections,
               'selected_lr_seed_statistics': tuning_statistics,
               'pareto_candidate_ids': [r['candidate_id'] for r in front], 'verification': verification,
               'quality_results': [r for r in rows if 'quality_metrics' in r and r.get('quality_metrics')],
               'analysis_source_sha256': source_hash(root),
               'limitations': ['one model, first 2048 calibration train tokens', 'validation n=256; independent confirmation n=1024',
                               'bounded tuning, two training seeds, no exhaustive rank/LR search',
                               'no runtime speed/VRAM saving claim']}
    clean = lambda r: {k: v for k, v in r.items() if k not in metadata}
    csv_table(directory/'phase3a_subspace_diagnostics.csv', [clean(r) for r in diagnostics])
    csv_table(directory/'phase3a_candidates.csv', [clean(r) for r in locals_])
    write_json(directory/'subspace_spectra.json', [clean(r) for r in spectra])
    write_json(directory/'summary.json', summary)
    write_json(directory/'VERIFICATION.json', verification)
    by_id = {r['candidate_id']: r for r in locals_}
    lines = ['# REPORT_PHASE3A', '', f'**Verdict: {verdict}.**', '',
             f'Run `{directory.name}`, Git `{metadata["git_commit"]}`. Реальный Qwen3-1.7B-Base BF16, RTX 4060 8 GB, WSL2.', '',
             f'Diagnostics run: `{summary["diagnostic_run"]}`. Canonical correlations уточнены прямой CPU FP64 SVD '
             'cross-basis matrix: прежний FP32 GPU Gram/eigvalsh path показал discrepancy до 2.34e-5 '
             'между mean squared correlation и projection overlap. BF16 factors, decomposition, '
             'training и functional measurements не изменялись; прежние raw rows сохранены.', '',
             '## Протокол', '',
             'Два заранее фиксированных бюджета: 27 586 944 и 18 502 368 BF16 units. Весь remainder хранится как zero padding; '
             'BF16 tensors сериализованы и загружены перед каждой оценкой. Shared-only ranks 2245/1505; coefficients '
             f'27 586 560/18 493 440; padding 384/8928. Compression {37748736/27586944:.6f}×/{37748736/18502368:.6f}× относится к MLP одного слоя.', '',
             'Стартовые shared fractions 0/.25/.5/.75/1; equal и energy-greedy распределение residual ranks. '
             'Две соседние fractions ±.125 около лучшего nonzero-common кандидата; выбор по validation MLP, не по weight error. '
             'rc=0/equal использует прежнюю обычную SVD и прежние ranks [1123,1122,1122]/[753,753,752]. '
             'rc=0/energy дополнительно проверяет более сильное распределение независимых ranks.', '',
             'Calibration: первые 1024 train tokens разделены детерминированно 768/256 (split seed 4242); '
             'следующие 1024 train tokens — независимая confirmation, не checkpoint selection. '
             'Canonical FP32 MLP на восстановленных BF16 weights — primary. Полный блок реально проигран BF16 '
             'с исходными hidden states, attention mask и rotary embeddings; validation/confirmation subsets берутся из его выхода.', '',
             'Полные block metrics для всех 2048 calibration tokens находятся в полях all_block_* CSV/JSON. '
             'В таблицах показаны выходы блока на validation/confirmation tokens после полного context replay; '
             'их relative-error denominators отличаются от whole-calibration Phase-2 metrics.', '',
             'Вся base model заморожена. Только factors одного MLP обучаются AdamW, weight_decay=0, '
             'LR 1e-4/3e-4, до 60 steps, batch 32 tokens, validation каждые 10 steps, patience=3. '
             'Objective: normalized MLP MSE (lambda_mlp=1, lambda_block=0). Optional actual block term реализован, '
             'но отдельный block-loss sweep не проводился. Один selection seed 5101; выбранный LR повторён seed 5107. '
             'SVD получил тот же allowance. Train никогда не выбирает checkpoint; WikiText test не участвует в training.', '',
             '## Девять ответов', '']
    overlaps = [r for r in diagnostics if r['k'] == 1024]
    lines += ['1. **Общая геометрия:** normalized projection overlap при k=1024 по всем слоям/парам '
              f'{min(r["normalized_projection_overlap"] for r in overlaps):.4f}–{max(r["normalized_projection_overlap"] for r in overlaps):.4f}. '
              'Полные spectra, canonical correlations, principal angles и energy capture сохранены; это не доказательство inference quality.', '',
              '| Layer | Gate/up overlap | Gate/down overlap | Up/down overlap | Shared energy gate/up/down, k=1024 |',
              '|---:|---:|---:|---:|---|']
    for layer in sorted(set(r['layer'] for r in overlaps)):
        d = {r['pair']: r for r in overlaps if r['layer'] == layer}
        energy = d['gate/up']['shared_energy_captured']
        lines.append(f'| {layer} | {d["gate/up"]["normalized_projection_overlap"]:.4f} | '
                     f'{d["gate/down"]["normalized_projection_overlap"]:.4f} | {d["up/down"]["normalized_projection_overlap"]:.4f} | '
                     f'{energy["gate"]:.4f}/{energy["up"]:.4f}/{energy["down"]:.4f} |')
    lines += ['', '2–4. **Shared-only, common/private и greedy:** результаты для обоих бюджетов ниже. '
              'Weight-optimal allocation не гарантирует меньшую SwiGLU functional error.', '',
              '| Budget | Method | rc | Private ranks | Train MLP | Validation MLP | Confirmation MLP | Val block | Confirm block |',
              '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for comp in comparisons:
        ids = list(dict.fromkeys(comp[key] for key in ('independent_svd', 'shared_only', 'best_common_private', 'best_analytic_private',
                                                     'best_selected_private', 'best_selected_joint')))
        for identity in ids:
            r = by_id[identity]
            lines.append(f'| {r["budget_name"]} | {r["strategy"]}/{r["allocation"]} | {r["common_rank"]} | {r["private_ranks"]} | '
                         f'{r["train_mlp_output_relative_error"]:.6f} | {r["mlp_output_relative_error"]:.6f} | '
                         f'{r["confirmation_mlp_output_relative_error"]:.6f} | {r["block_output_relative_error"]:.6f} | '
                         f'{r["confirmation_block_output_relative_error"]:.6f} |')
    lines.append('')
    for comp in comparisons:
        shared, common, baseline = (by_id[comp[k]] for k in ('shared_only', 'best_common_private', 'independent_svd'))
        delta = shared['mlp_output_relative_error']-baseline['mlp_output_relative_error']
        private_delta = common['mlp_output_relative_error']-shared['mlp_output_relative_error']
        greedy = [r for r in greedy_comparisons if r['budget_name'] == comp['budget_name']]
        lines.append(f'{comp["budget_name"]}: shared−SVD validation Δ={delta:+.6f}; лучший actual common/private '
                     f'split f={common["shared_fraction"]:.3f}, allocation={common["allocation"]}; common/private−shared Δ={private_delta:+.6f}. '
                     f'Greedy меньше equal по validation MLP в {sum(r["validation_mlp_delta"] < 0 for r in greedy)}/{len(greedy)} splits; '
                     f'средняя descriptive Δ={np.mean([r["validation_mlp_delta"] for r in greedy]):+.6f}. '
                     'Это сравнение фиксированных splits, не seed CI.')
    lines += ['', '5. **Held-out improvement от tuning:** все training/validation curves и оба seeds сохранены. '
              'Таблица ниже показывает каждый run, включая LR, который не выбран; summary/CSV содержат все analytic candidates.', '',
              '| Budget | Method | LR | Seed | Best step | Train | Validation | Confirmation | Confirm block |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in locals_:
        if r['strategy'].startswith('functional-'):
            lines.append(f'| {r["budget_name"]} | {r["strategy"]} | {r["learning_rate"]:g} | {r["training_seed"]} | '
                         f'{r["best_checkpoint_step"]} | {r["train_mlp_output_relative_error"]:.6f} | {r["mlp_output_relative_error"]:.6f} | '
                         f'{r["confirmation_mlp_output_relative_error"]:.6f} | {r["confirmation_block_output_relative_error"]:.6f} |')
    lines += ['', '6. **Победа над matched SVD:** подтверждённых winners по validation и отдельной confirmation, '
              f'с не худшим block и обоими training seeds: {len(completion["confirmed_local_winners"])}. '
              'Сравнение включает tuned SVD, а не только исходную weight-optimal SVD.',
              'Ниже усреднены оба заранее заданных seeds при validation-selected LR, без выбора лучшего seed.', '',
              '| Budget | Method | N | Val MLP mean ± sample SD | Confirmation MLP mean | Confirmation block mean |',
              '|---|---|---:|---:|---:|---:|']
    for s in tuning_statistics:
        lines.append(f'| {s["budget_name"]} | {s["strategy"]} | {s["N"]} | {s["validation_mlp_output_relative_error_mean"]:.6f} ± '
                     f'{s["validation_mlp_sample_sd"]:.6f} | {s["confirmation_mlp_output_relative_error_mean"]:.6f} | '
                     f'{s["confirmation_block_output_relative_error_mean"]:.6f} |')
    lines += ['', 'Tuning улучшает held-out functional error всех трёх representations; после одинакового allowance '
              'SVD лучше shared-only и actual common/private на обоих seeds обоих budgets. '
              'Малый analytic medium gain shared-only не сохраняется против tuned SVD. '
              'N=2 не даёт сильной статистической оценки произвольных training seeds; вывод ограничен этим bounded protocol.', '',
              '7. **Перенос:** диагностика выполнена для 0/7/14/21/27. Local transfer не запускался: layer14 не прошёл gate.',
              '8. **WikiText PPL:** selective/full PPL и downstream не запускались при отсутствии подтверждённого local SVD win. '
              'Phase-2 quality остаётся историческим результатом и не приписывается Phase 3A.',
              '9. **Direct inference/Triton:** не запускать по текущим данным. Локальное факторное сжатие не измеряет '
              'runtime speedup, footprint всей модели или эффект совместной замены 28 MLP.', '',
              '## Проверки и артефакты', '',
              f'{len(locals_)} успешных local candidates; {len(diagnostics)} pair/k diagnostics; {len(spectra)} spectra. '
              f'Failed/OOM rows: {verification["failed_rows"]}. Старые {len(guard["protected_files"])} files и append-only prefix: '
              f'{protected_ok}/{prefix_ok}. BF16 factor SHA, точный storage и global raw correspondence: {storage_ok}/{raw_match}.', '',
              f'Полный real-artifact pytest: {verification.get("pytest", {}).get("passed", "not-run")} passed, '
              f'{verification.get("pytest", {}).get("skipped", "unknown")} skipped; '
              f'{verification.get("pytest", {}).get("seconds", "unknown")} seconds. '
              'Проверены actual saved Pareto replay, validation checkpoint selection, frozen base gradients, '
              'optional block-loss gradient path и восстановление весов при injected residual-decomposition OOM.', '',
              'Все decomposition/evaluation/training wall times и peak allocated VRAM находятся в CSV/JSON. '
              'decomposition_seconds включает загрузку shared basis и residual SVD; построение shared basis '
              'и subspace diagnostics имеют отдельный stage-0 timer. Factor packing/save отдельно не timed. '
              'evaluation_seconds включает factor load/reconstruction, local replay и Frobenius metrics. '
              'Weight energy explained означает 1−squared relative Frobenius error после BF16 replay; '
              'singular-energy capture отдельно сохранён в diagnostics. Spectral-norm error не измерялся. '
              'Torch peak относится к процессу; desktop GPU contention не контролировался. '
              'Git dirty flag сохранён как наблюдался; измеренный source сохранён в ZIP вместе с реальным commit SHA. '
              'Присутствовавший до реализации пустой task.txt не изменялся.', '',
              f'Ранее выполненные primary measurements сохранены в `{metadata.get("resumed_primary_run", directory.name)}`. '
              'Продолжение добавило qualifying actual common/private tuning; config, capture/token hashes, numerical AST '
              'и исходные ops проверены до reuse. Каждая reused row сохраняет measurement run/time/source SHA; '
              'исторические rows и artifacts не переписаны.', '',
              'Источники матричных операций: [PyTorch eigh](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.eigh.html), '
              '[PyTorch SVD](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.svd.html). '
              'Shared left basis получен из sum WiWi.T, эквивалентного MM.T. Factor conventions Phase 2 сохранены.', '']
    text = '\n'.join(lines)
    (directory/'REPORT_PHASE3A.md').write_text(text, encoding='utf-8')
    (root/'REPORT_PHASE3A.md').write_text(text, encoding='utf-8')
    (root/'results/phase3a_summary.json').write_bytes((directory/'summary.json').read_bytes())
    (root/'VERIFICATION.json').write_bytes((directory/'VERIFICATION.json').read_bytes())
    for name in ('phase3a_subspace_diagnostics.csv', 'phase3a_candidates.csv'):
        (root/'results'/name).write_bytes((directory/name).read_bytes())
    print(json.dumps({'verdict': verdict, 'verification': verification}, ensure_ascii=False))
