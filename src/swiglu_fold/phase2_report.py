from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .phase2 import bootstrap, grouped, pareto_groups
from .telemetry import source_hash, write_json


def describe(values, config):
    a = np.asarray(values)
    return {'N': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'sample_sd': float(a.std(ddof=1)) if len(a) > 1 else None,
            'bootstrap_95ci': bootstrap(a, config['seed'], config['samples']) if len(a) > 1 else None}


def paired(left, right, metric, config):
    right = {r['replica']: r for r in right}
    differences = [r[metric] - right[r['replica']][metric] for r in left if r['replica'] in right]
    if not differences:
        return None
    return {**describe(differences, config), 'paired_delta': float(np.mean(differences)),
            'paired_win_rate': float(np.mean(np.asarray(differences) < 0)),
            'delta_sign': 'candidate minus control; negative is improvement'}


def csv_table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def report(root: Path, directory: Path, full_directory: Path | None = None):
    rows = [json.loads(line) for line in (directory / 'raw.jsonl').read_text().splitlines()]
    metadata = json.loads((directory / 'metadata.json').read_text())
    verification = json.loads((directory / 'VERIFICATION.json').read_text()) if (directory / 'VERIFICATION.json').exists() else None
    config = metadata['config']
    groups = grouped(rows)
    stats, comparisons, ablations = [], [], []
    metrics = ('mlp_output_relative_error', 'block_output_relative_error')
    for (layer, name, strategy), group in groups.items():
        for metric in metrics:
            stats.append({'layer': layer, 'config_name': name, 'strategy': strategy, 'metric': metric,
                          'effective_parameter_count': group[0]['effective_parameter_count'],
                          'effective_compression_ratio': group[0]['effective_compression_ratio'],
                          **describe([r[metric] for r in group], config['bootstrap'])})
        if strategy == 'structure-aware-balanced':
            for control in ('random', 'structure-aware'):
                for metric in metrics:
                    comparison = paired(group, groups.get((layer, name, control), []), metric, config['bootstrap'])
                    if comparison:
                        comparisons.append({'layer': layer, 'config_name': name, 'strategy': strategy,
                                            'control': control, 'metric': metric, **comparison})
        if strategy == 'structure-aware-activation':
            for metric in metrics:
                comparison = paired(group, groups.get((layer, name, 'structure-aware-balanced'), []), metric, config['bootstrap'])
                if comparison:
                    comparisons.append({'layer': layer, 'config_name': name, 'strategy': strategy,
                                        'control': 'structure-aware-balanced', 'metric': metric, **comparison})
        if name.endswith(('-B', '-C')):
            control_name = name[:-1] + 'A'
            control_rows = groups.get((layer, control_name, strategy), [])
            if control_rows:
                assert group[0]['effective_parameter_count'] == control_rows[0]['effective_parameter_count']
                for metric in metrics:
                    comparison = paired(group, control_rows, metric, config['bootstrap'])
                    ablations.append({'layer': layer, 'config_name': name, 'control_config': control_name,
                                      'strategy': strategy, 'metric': metric, **comparison})
    front, means = pareto_groups(rows)
    quality = [r for r in rows if r['kind'] == 'phase2_quality']
    svd_comparisons = []
    for (layer, name, strategy), group in groups.items():
        if strategy in ('svd', 'zero-mlp'):
            continue
        budget = group[0]['effective_parameter_count']
        svd = next((r for r in rows if r['kind'] == 'phase2_local' and r['status'] == 'ok'
                    and r['strategy'] == 'svd' and r['layer'] == layer and r['effective_parameter_count'] == budget), None)
        if svd:
            deltas = {m: float(np.mean([r[m] for r in group])) - svd[m] for m in metrics}
            uncertainty = {m: {**describe([r[m]-svd[m] for r in group],config['bootstrap']),
                                  'paired_delta':deltas[m], 'paired_win_rate':float(np.mean([r[m]<svd[m] for r in group]))}
                           for m in metrics}
            svd_comparisons.append({'layer': layer, 'config_name': name, 'strategy': strategy,
                                    'effective_parameter_count': budget, 'svd_ranks': svd['svd_ranks'],
                                    'svd_config_name': svd['config_name'], **deltas,
                                    'svd_dominates_both_local_metrics': all(v >= 0 for v in deltas.values()),
                                    'seed_uncertainty_against_fixed_svd':uncertainty})
    robust = [r for r in comparisons if r['strategy'] == 'structure-aware-balanced' and r['metric'] == metrics[0]
              and r['N'] >= 3 and r['bootstrap_95ci'] and r['bootstrap_95ci'][1] < 0]
    transfer = sorted({r['layer'] for r in robust if any(s['layer'] == r['layer'] and s['config_name'] == r['config_name']
                         and s['control'] != r['control'] for s in robust)})
    v2_svd = [r for r in svd_comparisons if r['strategy'] == 'structure-aware-balanced']
    all_svd_dominates = bool(v2_svd) and all(r['svd_dominates_both_local_metrics'] for r in v2_svd)
    v2_front = [k for k in front if k[2] == 'structure-aware-balanced']
    robust_svd_benefit = [r for r in v2_svd if all(r['seed_uncertainty_against_fixed_svd'][m]['bootstrap_95ci']
                              and r['seed_uncertainty_against_fixed_svd'][m]['bootstrap_95ci'][1]<0 for m in metrics)]
    quality_supported = []
    for layer,name,strategy in v2_front:
        q = [r['quality_metrics']['nll'] for r in quality if r['status']=='ok' and r['layer']==layer
             and r.get('config_name')==name and r['strategy']==strategy]
        zero = next(r['quality_metrics']['nll'] for r in quality if r['status']=='ok' and r['layer']==layer and r['strategy']=='zero-mlp')
        ci = bootstrap(q,config['bootstrap']['seed'],config['bootstrap']['samples']) if len(q)>1 else None
        if ci and ci[1]<zero and any(r['layer']==layer and r['config_name']==name for r in robust_svd_benefit):
            quality_supported.append((layer,name,strategy))
    verdict = 'not-supported' if not quality_supported else ('supported' if len(transfer)>=2 else 'mixed')
    guard = json.loads((root / 'results/phase2_history_guard.json').read_text())
    prefix_ok = hashlib.sha256((root / 'results/experiments.jsonl').read_bytes()[:guard['bytes']]).hexdigest() == guard['sha256']
    protected_ok = all(hashlib.sha256((root / p).read_bytes()).hexdigest() == digest for p, digest in guard['protected_files'].items())
    assert prefix_ok and protected_ok
    zero14 = next(r['quality_metrics']['nll'] for r in quality if r['strategy']=='zero-mlp' and r['layer']==14 and r['status']=='ok')
    best = min((r for r in quality if r['status'] == 'ok' and r['strategy'] not in ('dense', 'zero-mlp') and r['layer'] == 14
                and r['quality_metrics']['nll']<zero14),
               key=lambda r: (abs(r['effective_compression_ratio'] - 2), r['quality_metrics']['nll']), default=None)
    full_rows = [json.loads(line) for line in (full_directory / 'raw.jsonl').read_text().splitlines()] if full_directory else []
    per_layer_selection = []
    for layer in config['layers']:
        zero_nll = next(r['quality_metrics']['nll'] for r in quality if r['strategy']=='zero-mlp' and r['layer']==layer)
        candidates = [r for r in quality if r['status']=='ok' and r['layer']==layer and r['strategy'] not in ('dense','zero-mlp')
                      and r['quality_metrics']['nll']<zero_nll]
        if candidates:
            chosen = min(candidates,key=lambda r:(abs(r['effective_compression_ratio']-2),r['quality_metrics']['nll']))
            local_row=next(r for r in rows if r['kind']=='phase2_local' and r.get('candidate_id')==chosen['candidate_id'])
            per_layer_selection.append({k:chosen[k] for k in ('layer','strategy','config_name','ranks','effective_parameter_count','effective_compression_ratio','quality_metrics')}
                                       | {'actual_svd_ranks':local_row.get('svd_ranks')})
    global_best_subset = min((r for r in quality if r['status']=='ok' and r['strategy'] not in ('dense','zero-mlp')),
                             key=lambda r:r['quality_metrics']['nll'])
    summary = {'verdict': verdict, 'run_id': metadata['run_id'], 'metadata': metadata, 'verification':verification,
               'statistics': stats, 'paired_comparisons': comparisons, 'svd_comparisons': svd_comparisons,
               'ablations': ablations, 'pareto_frontier': front, 'pareto_group_means': [{'layer': k[0], 'config_name': k[1], 'strategy': k[2], **v} for k,v in means.items()],
               'robust_v2_vs_both_controls_layers': transfer, 'all_v2_dominated_by_svd': all_svd_dominates,
               'robust_v2_svd_benefit':robust_svd_benefit,'v2_pareto_with_robust_svd_and_zero_ppl_benefit':quality_supported,
               'quality_results': quality, 'best_layer14_near_2x_pareto_candidate': best,
               'per_layer_near_2x_zero_ppl_survivors':per_layer_selection,
               'best_subset_ppl_candidate_across_layers':global_best_subset,
               'full_quality_run': str(full_directory) if full_directory else None, 'full_quality_results': full_rows,
               'history_prefix_unchanged': prefix_ok, 'phase1_artifacts_unchanged': protected_ok,
               'failed_rows': [r for r in rows if r['status'] != 'ok'],
               'analysis_source_sha256': source_hash(root), 'analysis_scope': 'exploratory, unadjusted percentile bootstrap; transfer N=3 is preliminary'}
    summary['full_quality_scope_semantics']='For measured full run, inherited metadata quality_scope is the 4096-token preflight plan; actual scope is quality_metrics.full_evaluation/tokens/scored_tokens or downstream examples, pinned task dataset rows and full_test_token_sha256.'
    write_json(directory / 'summary.json', summary)
    write_json(root / 'results/phase2_summary.json', summary)
    for name, values in [('statistics', stats), ('paired_comparisons', comparisons), ('svd_comparisons', svd_comparisons), ('ablations', ablations)]:
        csv_table(directory / f'{name}.csv', values)
    csv_table(directory / 'quality.csv', [{k:r.get(k) for k in ('layer','strategy','config_name','replica','effective_compression_ratio')}
                                         | r.get('quality_metrics',{}) for r in quality])
    lines = ['# REPORT_PHASE2', '', f'**Verdict: {verdict}.**', '',
             f'Run `{metadata["run_id"]}`. Реальный Qwen3-1.7B-Base BF16, RTX 4060 8 GB, WSL2. '
             f'{sum(r["kind"]=="phase2_local" and r["status"]=="ok" for r in rows)} локальных кандидатов; '
             f'{len(quality)} subset quality rows. Ошибок/OOM: {len(summary["failed_rows"])}.', '',
             '## Протокол', '',
             'Layer 14: десять заранее заданных независимых пар projection/clustering seeds и десять random permutations; '
             'layers 0/7/21/27: три пары для дешёвой проверки переноса. Identity и SVD детерминированы и измерены один раз. '
             'Seeds не выбирались по результату. Порядок групп — label centroids, внутри — PC1 с фиксированным знаком и index tie-break.', '',
             'Balanced assignment — capacitated auction с epsilon=1e-4 и проверкой primal/dual gap ≤ N·epsilon. '
             'Это epsilon-optimal assignment к фиксированным KMeans centroids, а не глобальная оптимальность clustering. '
             'Каждый raw record сохраняет точные ёмкости, gap и число bids. v1 не изменён.', '',
             'Первичная MLP metric: FP32, TF32 off, canonical order, веса восстановлены из сохранённых BF16 factors и приведены к BF16. '
             'Полный блок отдельно проигрывается в BF16 с исходными hidden states, attention mask и rotary embeddings. '
             'Новые mlp_output_mse и block_output_mse относятся к разным выходам; старое поле Phase 1 не переписывалось.', '',
             'Calibration: первые 1024 train tokens; subset PPL: первые 4096 test tokens WikiText-2 raw, 512/stride 256, 4095 targets ровно один раз. '
             'Model SHA, dataset SHA, token hashes, config, source hash/snapshot и реальные BF16 веса записаны в run.', '',
             'Бюджет — BF16 storage units: три Tucker core/factors + 6144 int64 permutation entries (24 576 units). '
             'SVD использует все доступные units на обычные 2D factors, без permutation; остаток <8192 units записывается отдельным zero padding tensor. '
             'Поэтому allocated storage равен точно; математических SVD coefficients немного меньше из-за дискретного rank. '
             'А и B/C имеют точно одинаковое число Tucker coefficients плюс permutation, без padding. '
             'Compression ratio относится к MLP одного слоя, не всей модели. Dense reconstruction не ускоряет inference.', '',
             '## Восемь ответов', '',
             '1. **Лучше v1?** Устойчивое преимущество на основных high-A/medium-A не доказано: оба CI пересекают ноль. '
             'На floor-A около 4× есть положительный seed signal, но функциональная ошибка остаётся почти как у zero-MLP.',
             '2. **Лучше random?** На layer 14, high-A и medium-A — да по FP32 local metric: оба CI ниже нуля. '
             'Это не устойчивое превосходство одновременно над v1 и SVD и не подтверждённый multi-layer result.',
             f'3. **Лучше SVD?** SVD доминирует v2 по обеим local metrics в {sum(r["svd_dominates_both_local_metrics"] for r in v2_svd)}/{len(v2_svd)} matched configurations. '
             f'Конфигураций с CI ниже нуля против SVD по обеим metrics: {len(robust_svd_benefit)}. '
             'Преимущество одного mean без CI не считается победой.',
             f'4. **Перенос по слоям:** CI ниже нуля против обоих ordering controls хотя бы на одной общей конфигурации: {transfer}, '
             f'{len(transfer)}/5. N=3 на transfer layers не доказывает устойчивое обобщение.',
             '5. **Отличие от zero-MLP:** у layer14 v2 ошибка MLP 0.766 около 1.37× и 0.928 около 2.04× против 1.0 у zero. '
             'PPL layer14 v2 не измерялась, поскольку эти кандидаты local-dominated SVD. '
             'SVD 2.04× имеет subset PPL 11.78184 против zero 11.77131; поэтому full quality выполнена только для SVD 1.37× '
             '(subset PPL 11.66785). Единственный v2 Pareto mean на layer7/floor-A имеет N=3 и PPL почти как zero, включая seed хуже zero.',
             '6. **Польза hidden axes:** для v2 layer14 около 2× B−A/C−A CI пересекают ноль; около 3×/4× B/C хуже A с CI выше нуля. '
             'Устойчивый benefit true-5D hidden-axis compression этим search не доказан.',
             f'7. **Лучший прошедший local Pareto и zero subset-PPL контроль кандидат layer14:** `{best["candidate_id"] if best else "none"}`. '
             f'Compression ratio {best["effective_compression_ratio"]:.6f}×. ' if best else '7. Pareto finalist отсутствует. ',
             'Основная цель около 2× не прошла zero quality floor; выбрана ближайшая выжившая конфигурация 1.37×. '
             'Pareto строится по group mean MLP/block error и storage; PPL измерена у всех replicas каждой frontier group, не у лучшего seed.',
             f'8. **Phase 3/Triton:** не начинались. V2 Pareto-конфигураций с robust local SVD benefit и subset NLL CI ниже zero-MLP: {len(quality_supported)}. '
             'Переход к Phase 3 оправдан только при supported; при текущем недостаточном результате рекомендовано остановить эту гипотезу.', '',
             '## Layer 14: paired FP32 MLP error', '',
             '| Config | Control | N | Δ mean | Δ median | SD(Δ) | Win rate | Bootstrap 95% CI |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in comparisons:
        if r['layer']==14 and r['strategy']=='structure-aware-balanced' and r['metric']==metrics[0]:
            lines.append(f'| {r["config_name"]} | {r["control"]} | {r["N"]} | {r["mean"]:.6f} | {r["median"]:.6f} | '
                         f'{r["sample_sd"]:.6f} | {r["paired_win_rate"]:.0%} | {r["bootstrap_95ci"][0]:.6f} … {r["bootstrap_95ci"][1]:.6f} |')
    lines += ['', 'Знак: v2−control, меньше лучше. 10 000 bootstrap draws, seed 4001; percentile CI. '
              'Все конфигурации и обе metrics сохранены, а не только значимые. Множественные comparisons не корректировались; '
              'главные заранее заданные A точки high/medium, остальные — exploratory. CI при N=1 не выдумывается.', '',
              '## Means: все layers, A около 2×', '', '| Layer | Strategy | N | MLP relative mean | Block relative mean |', '|---:|---|---:|---:|---:|']
    for layer in config['layers']:
        for strategy in (*('zero-mlp', 'identity', 'random', 'structure-aware', 'structure-aware-balanced', 'svd'),):
            name = 'zero' if strategy=='zero-mlp' else 'medium-A'
            group = groups.get((layer,name,strategy))
            if group:
                lines.append(f'| {layer} | {strategy} | {len(group)} | {np.mean([r[metrics[0]] for r in group]):.6f} | {np.mean([r[metrics[1]] for r in group]):.6f} |')
    lines += ['', '## Exact-budget ablation: layer 14, balanced-v2', '', '| Candidate−A | Metric | N | Mean Δ | Win rate | 95% CI |', '|---|---|---:|---:|---:|---|']
    for r in ablations:
        if r['layer']==14 and r['strategy']=='structure-aware-balanced':
            lines.append(f'| {r["config_name"]} | {r["metric"]} | {r["N"]} | {r["mean"]:.6f} | {r["paired_win_rate"]:.0%} | {r["bootstrap_95ci"]} |')
    lines += ['', '## Activation-aware', '',
              'Отдельные 10 replicas на layer 14, high-A/medium-A. Seven usage features: gate/SiLU/up/product RMS, '
              'product × down-norm RMS, product variance, abs mean. Log floor 1e-12, column z-score, coefficient 0.25; '
              'weight signatures unchanged, compact fingerprint 6144×7. Полный token×neuron массив не сохранялся. '
              'Это одна заранее выбранная normalization, не exhaustive search.', '']
    for r in comparisons:
        if r['strategy']=='structure-aware-activation' and r['metric']==metrics[0]:
            lines.append(f'- {r["config_name"]}: activation−weight v2 mean Δ {r["mean"]:.6f}, wins {r["paired_win_rate"]:.0%}, CI {r["bootstrap_95ci"]}.')
    lines += ['', '## Selective subset PPL', '', '| Layer | Strategy | Config | N | Mean PPL | Min … max |', '|---|---|---|---:|---:|---|']
    qgroups = {}
    for r in quality:
        if r['status']=='ok':
            qgroups.setdefault((r['layer'],r['strategy'],r.get('config_name','—')),[]).append(r['quality_metrics']['wikitext2_raw_perplexity'])
    for (layer,strategy,name),values in qgroups.items():
        lines.append(f'| {layer} | {strategy} | {name} | {len(values)} | {np.mean(values):.6f} | {min(values):.6f} … {max(values):.6f} |')
    lines += ['', 'Бюджет выбирается отдельно по каждому слою: ближайшая к 2× локальная Pareto точка, '
              'прошедшая свой zero-MLP subset NLL floor. Это не рецепт замены всех слоёв.', '',
              '| Layer | Strategy | Config | SVD ranks | MLP ratio | Subset PPL |', '|---:|---|---|---|---:|---:|']
    for r in per_layer_selection:
        lines.append(f'| {r["layer"]} | {r["strategy"]} | {r["config_name"]} | {r["actual_svd_ranks"]} | {r["effective_compression_ratio"]:.6f} | {r["quality_metrics"]["wikitext2_raw_perplexity"]:.6f} |')
    lines += ['', f'Минимальный subset PPL среди всех слоёв: `{global_best_subset["candidate_id"]}`, '
              f'{global_best_subset["quality_metrics"]["wikitext2_raw_perplexity"]:.6f}. '
              'Это выбор по короткому subset; full quality protocol был заранее ограничен layer14, '
              'поэтому глобальный выигрыш на полном benchmark не утверждается.']
    lines += ['', '## Full quality', '']
    if full_rows:
        lines += [f'Run `{full_directory.name}`. Все 299 078 test tokens WikiText-2; полные benchmark splits, zero-shot, без chat template. '
                  'Только SVD 1.37×, dense и zero layer14: v2 не прошёл local/zero quality отбор. '
                  'Это own scorer с prompts/preprocessing lm-evaluation-harness; не утверждается запуск пакета lm-eval.', '',
                  '| Strategy | Full PPL | NLL | Scored targets |', '|---|---:|---:|---:|']
        for r in (r for r in full_rows if r['kind']=='phase2_full_ppl'):
            q=r['quality_metrics']
            lines.append(f'| {r["strategy"]} | {q["wikitext2_raw_perplexity"]:.6f} | {q["nll"]:.6f} | {q["scored_tokens"]} |')
        downstream={(r['task'],r['strategy']):r['quality_metrics'] for r in full_rows if r['kind']=='phase2_downstream'}
        lines += ['', '| Task | Metric | N | Dense | Zero-MLP | SVD 1.37× |', '|---|---|---:|---:|---:|---:|']
        for task in ('hellaswag','piqa','arc_easy','winogrande'):
            for metric in (('accuracy',) if task=='winogrande' else ('accuracy','accuracy_character_normalized')):
                if all((task,s) in downstream for s in ('dense','zero-mlp','svd')):
                    q=downstream[(task,'dense')]
                    values=' | '.join(f'{downstream[(task,s)][metric]:.4%}' for s in ('dense','zero-mlp','svd'))
                    lines.append(f'| {task} | {metric} | {q["examples"]} | {values} |')
        check=next(r for r in full_rows if r['kind']=='phase2_scoring_check')
        lines += ['', f'Real-model scorer control: FP32 max per-token difference {check["fp32_diagnostic_max_delta"]:.8f} '
                  f'(strict guard 5e-5). BF16 batch/full-logit max {check["mean_logprob_per_token_max_difference"]:.6f}; '
                  'guard 0.1 после отдельной FP32 проверки. Первоначальный guard 0.02 остановил предыдущий run до benchmark metrics. '
                  'BF16 padding/LM-head shape влияет на near ties; небольшие изменения accuracy не являются сами по себе доказательством улучшения.', '',
                  'Pinned dataset SHA, gzip requests с точными token IDs, prediction JSONL каждого примера, hashes и source snapshot '
                  'сохранены в full run. Один детерминированный finalist, без выбора лучшего random seed. '
                  'В raw этого run унаследованное metadata quality_scope относится к preflight 4096 tokens; '
                  'фактический full scope задают quality_metrics.full_evaluation, tokens/scored_tokens/examples и full_test_token_sha256. '
                  'Старые rows не переписаны; этот metadata-only дефект исправлен для будущих runs.']
    else:
        lines.append('Full WikiText-2 и downstream ещё не выполнены; это явно не подтверждённая часть отчёта.')
    lines += ['', '## Проверки и границы', '',
              f'Append-only prefix и защищённые Phase-1 files проверены: {prefix_ok}/{protected_ok}. '
              'Исходный REPORT_PHASE1.md и прежние JSONL rows неизменны. Source snapshot каждого run сохраняет измеренный код; '
              'analysis source SHA записан отдельно, если анализ добавлен позже.', '',
              'CSV statistics/paired_comparisons/svd_comparisons/ablations/quality и summary.json лежат рядом с raw.jsonl. '
              'Все N=10 результаты layer14 и N=3 transfer results включены. OOM handler восстанавливает dense веса в finally; '
              f'проверки записаны в VERIFICATION.json. Полный pytest: {verification["pytest"]["passed"] if verification and "pytest" in verification else "не записан"} passed. '
              f'Независимый artifact audit: {verification["passed"] if verification else "не выполнен"}; '
              f'full quality measurements: {verification["full_quality_measurements"] if verification else 0}. '
              'Отдельный read-only verifier Phase 1 подтвердил 28 BF16 artifacts и 120 прежних generation measurements.', '',
              'Вывод ограничен одной моделью, одним calibration subset и one-layer replacements. '
              'Не измерялись ускорение факторизованного inference, VRAM saving модели и качество совместной замены всех 28 MLP. '
              'Triton/custom CUDA/quantization и Phase 3 не запускались.', '',
              'Источники: [capacitated auction / dual certificate](https://www.mit.edu/~dimitrib/Auction_Survey.pdf), '
              '[ordinary SVD](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.svd.html). '
              'Compression implementation написана здесь; сторонние compression repositories не копировались.']
    text = '\n'.join(lines) + '\n'
    (directory / 'REPORT_PHASE2.md').write_text(text, encoding='utf-8')
    (root / 'REPORT_PHASE2.md').write_text(text, encoding='utf-8')
    print(json.dumps({'verdict':verdict,'frontier':front,'transfer_layers':transfer}), flush=True)
    return summary
