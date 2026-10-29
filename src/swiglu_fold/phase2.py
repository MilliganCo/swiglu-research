from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from math import prod
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
import yaml
from safetensors.torch import save_file

from .ops import (hosvd_bases, mlp, permute_weights, tensorize, tucker,
                  untensorize, validate_shape)
from .phase2_ops import (activation_fingerprint, functional_metrics, seeded_order,
                         svd_basis, svd_budget, svd_factors, tucker_parameters)
from .run import Experiment, cleanup, configure, install_weights, model_weights
from .telemetry import write_json


STRATEGIES = ('identity', 'random', 'structure-aware', 'structure-aware-balanced')
NAMES = ('gate', 'up', 'down')


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(v, device) for v in value)
    if value is not None and not isinstance(value, (bool, int, float, str)):
        raise TypeError(f'capture cannot persist {type(value)}')
    return value


def validate(config):
    validate_shape(tuple(config['shape']), 6144, 2048)
    if config['layers'] != [14, 0, 7, 21, 27] or config['dtype'] != 'bfloat16' or config['device'] != 'cuda':
        raise ValueError('Phase 2 requires the five requested layers, BF16 and CUDA')
    seeds = config['seeds']
    if any(len(seeds[key]) < 10 or len(set(seeds[key])) != len(seeds[key])
           for key in ('projection', 'clustering', 'permutation')):
        raise ValueError('at least ten distinct independent seeds per source required')
    if len({len(v) for v in seeds.values()}) != 1 or not 1 <= config['transfer_replicas'] <= len(seeds['projection']):
        raise ValueError('invalid paired seed arrays or transfer replica count')
    budgets = {}
    for cfg in config['rank_configs']:
        ranks = tuple(cfg['ranks'])
        budget = 3 * tucker_parameters(tuple(config['shape']), ranks) + 6144 * 4
        hidden_compressed = sum(r < d for r, d in zip(ranks[2:], config['shape'][2:]))
        if min(hidden_compressed, 2) != {'A': 0, 'B': 1, 'C': 2}[cfg['ablation']]:
            raise ValueError('ablation family does not match compressed hidden axes')
        if not 1.25 <= (3 * 6144 * 2048) / budget <= 4:
            raise ValueError('rank configuration outside useful compression region')
        budgets[cfg['name']] = budget
    for prefix in ('medium', 'low', 'floor'):
        if len({budgets[f'{prefix}-{f}'] for f in 'ABC'}) != 1:
            raise ValueError('ablation budgets must match exactly')
    quality = config['quality']
    if config['activation']['normalization'] not in ('log-floor-then-column-zscore','column-zscore'):
        raise ValueError('unknown activation normalization')
    if not 1 <= quality['stride'] <= quality['block_size'] or min(quality['calibration_tokens'], quality['evaluation_tokens']) < 2:
        raise ValueError('invalid quality window')


def bootstrap(values, seed, samples):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def grouped(rows):
    groups = {}
    for row in rows:
        if row['kind'] == 'phase2_local' and row['status'] == 'ok':
            key = (row['layer'], row['config_name'], row['strategy'])
            groups.setdefault(key, []).append(row)
    return groups


def pareto_groups(rows):
    groups = grouped(rows)
    means = {key: {metric: float(np.mean([r[metric] for r in group])) for metric in
                  ('effective_parameter_count', 'mlp_output_relative_error', 'block_output_relative_error')}
             for key, group in groups.items() if key[2] != 'zero-mlp'}
    front = []
    for key, values in means.items():
        dominated = any(other[0] == key[0] and other != key
                        and all(m[k] <= values[k] for k in values)
                        and any(m[k] < values[k] for k in values)
                        for other, m in means.items())
        if not dominated:
            front.append(key)
    return front, means


class Phase2(Experiment):
    def __init__(self, config, root, smoke):
        super().__init__(config, root, smoke)
        self.metadata.update({'algorithm_version': 'phase2-auction-1',
                              'metric_schema': 'phase2-mlp-fp32-canonical-block-bf16-1',
                              'mlp_metric_precision': 'FP32, canonical neuron order, saved BF16 factors/weights',
                              'block_metric_precision': 'BF16 full transformer block replay',
                              'budget_unit': 'one BF16 storage element (2 bytes); int64 permutations count as four',
                              'svd_padding_policy': 'saved zero alignment tensor, <8192 BF16 units, no permutation',
                              'inference_scope': 'one replaced MLP layer at a time; dense reconstructed weights'})
        self.captures = {}
        self.layer_weights = {}
        self.completed_candidates = set()

    def reuse_local(self, directory):
        previous = json.loads((directory / 'metadata.json').read_text())
        if previous['smoke'] or previous['config'] != self.config:
            raise ValueError('only an identical full configuration can resume')
        with zipfile.ZipFile(directory / 'source_snapshot.zip') as archive:
            for name in ('ops.py','phase2_ops.py'):
                relative = f'src/swiglu_fold/{name}'
                if archive.read(relative) != (self.root / relative).read_bytes():
                    raise ValueError(f'resume algorithm source differs: {name}')
            def methods(source):
                cls = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Phase2')
                return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
            old = methods(archive.read('src/swiglu_fold/phase2.py'))
            new = methods(Path(__file__).read_bytes())
            for name in ('capture_all','select_layer','local_metrics','seeds','order','budget_values','reconstruct_tucker','reconstruct_svd'):
                if old[name] != new[name]:
                    raise ValueError(f'resume numerical method differs: {name}')
        for key in ('hf_model_revision_sha','dataset_revision_sha','train_token_sha256','test_token_sha256',
                    'calibration_artifact_sha256','torch_version','transformers_version','cuda_runtime',
                    'nvidia_driver','gpu'):
            if previous[key] != self.metadata[key]:
                raise ValueError(f'resume provenance differs: {key}')
        self.metadata['resumed_local_run'] = previous['run_id']
        write_json(self.directory / 'metadata.json',self.metadata)
        for line in (directory / 'raw.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['kind'] != 'phase2_local' or row['status'] != 'ok' or not row.get('candidate_id'):
                continue
            values = {k:v for k,v in row.items() if k not in self.metadata and k not in ('kind','timestamp','run_id')}
            self.record('phase2_local',**values,reused_local_measurement=True,measurement_run_id=row['run_id'],
                        measurement_timestamp=row['timestamp'],measurement_source_sha256=row['source_sha256'])
            self.completed_candidates.add(row['candidate_id'])

    def record(self, kind, **values):
        values.setdefault('shape', self.config['shape'])
        values.setdefault('projection_seed', None)
        values.setdefault('clustering_seed', None)
        values.setdefault('random_permutation_seed', None)
        values.setdefault('ranks', None)
        values.setdefault('layer', None)
        values.setdefault('effective_parameter_count', None)
        values.setdefault('effective_compression_ratio', None)
        return super().record(kind, **values)

    @torch.inference_mode()
    def capture_all(self):
        handles = []
        for layer in self.config['layers']:
            block = self.model.model.layers[layer]
            self.captures[layer] = {'blocks': [], 'mlp_inputs': []}
            self.layer_weights[layer] = model_weights(block.mlp)
            if layer != 14:
                save_file(self.layer_weights[layer], str(self.directory / f'layer{layer}.safetensors'))

            def before(_module, args, kwargs, layer=layer):
                self.captures[layer]['blocks'].append({'args': move(args, 'cpu'), 'kwargs': move(kwargs, 'cpu')})

            def after(_module, _args, output, layer=layer):
                self.captures[layer]['blocks'][-1]['dense_output'] = output.detach().cpu()

            def mlp_before(_module, args, layer=layer):
                self.captures[layer]['mlp_inputs'].append(args[0].detach().cpu())

            handles.extend([block.register_forward_pre_hook(before, with_kwargs=True),
                            block.register_forward_hook(after), block.mlp.register_forward_pre_hook(mlp_before)])
        try:
            for ids in self.train.split(self.config['quality']['block_size'], dim=1):
                self.model(ids, use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        artifact = {}
        for layer, capture in self.captures.items():
            capture['x'] = torch.cat(capture.pop('mlp_inputs'), dim=1)
            artifact[f'layer{layer}_mlp_inputs'] = capture['x']
            for i, block in enumerate(capture['blocks']):
                artifact[f'layer{layer}_block{i}_input'] = block['args'][0]
                artifact[f'layer{layer}_block{i}_output'] = block['dense_output']
                for key, value in block['kwargs'].items():
                    if isinstance(value, torch.Tensor):
                        artifact[f'layer{layer}_block{i}_{key}'] = value
                    elif isinstance(value, tuple):
                        for j, tensor in enumerate(value):
                            artifact[f'layer{layer}_block{i}_{key}{j}'] = tensor
        save_file(artifact, str(self.directory / 'phase2_calibration.safetensors'))
        self.metadata['calibration_artifact_sha256'] = hashlib.sha256((self.directory / 'phase2_calibration.safetensors').read_bytes()).hexdigest()
        write_json(self.directory / 'metadata.json', self.metadata)

    def select_layer(self, layer):
        self.layer = layer
        self.module = self.model.model.layers[layer].mlp
        self.weights = self.layer_weights[layer]
        self.x = self.captures[layer]['x'].to('cuda')
        self.dense_weights = {k: v.to(device='cuda', dtype=torch.float32) for k, v in self.weights.items()}
        self.dense_mlp = mlp(self.x.float(), self.dense_weights)
        self.dense_block = torch.cat([b['dense_output'] for b in self.captures[layer]['blocks']], dim=1).to('cuda')

    @torch.inference_mode()
    def local_metrics(self, canonical_weights):
        try:
            actual_mlp = mlp(self.x.float(), {k: v.float() for k, v in canonical_weights.items()})
            install_weights(self.module, canonical_weights)
            outputs = []
            for b in self.captures[self.layer]['blocks']:
                outputs.append(self.model.model.layers[self.layer](*move(b['args'], 'cuda'), **move(b['kwargs'], 'cuda')))
            actual_block = torch.cat(outputs, dim=1)
            return functional_metrics(actual_mlp, self.dense_mlp, actual_block, self.dense_block)
        finally:
            install_weights(self.module, self.weights)

    def seeds(self, replica, strategy):
        seeds = self.config['seeds']
        return {'replica': replica, 'projection_seed': seeds['projection'][replica] if strategy.startswith('structure-aware') else None,
                'clustering_seed': seeds['clustering'][replica] if strategy.startswith('structure-aware') else None,
                'random_permutation_seed': seeds['permutation'][replica] if strategy == 'random' else None}

    def order(self, strategy, replica, activation=None):
        seeds = self.config['seeds']
        return seeded_order(strategy, self.weights, projection_seed=seeds['projection'][replica],
                            clustering_seed=seeds['clustering'][replica], random_permutation_seed=seeds['permutation'][replica],
                            groups=self.config['shape'][0], epsilon=self.config['auction_epsilon'], activation=activation,
                            activation_weight=self.config['activation']['coefficient'])

    def budget_values(self, cfg):
        factors = 3 * tucker_parameters(tuple(self.config['shape']), tuple(cfg['ranks']))
        total = factors + self.weights['gate'].shape[0] * 4
        return {'config_name': cfg['name'], 'ablation': cfg['ablation'], 'ranks': cfg['ranks'],
                'factor_parameter_count': factors, 'permutation_bytes': self.weights['gate'].shape[0] * 8,
                'padding_parameter_count': 0, 'effective_parameter_count': total,
                'effective_storage_bytes': total * 2, 'effective_compression_ratio': 3 * prod(self.weights['gate'].shape) / total}

    @torch.inference_mode()
    def reconstruct_tucker(self, perm, cfg, bases, save_path=None):
        ordered = permute_weights(self.dense_weights, perm.to('cuda'))
        packed = {'permutation': perm}
        approximate = {}
        for name in NAMES:
            tensor = tensorize(ordered[name], tuple(self.config['shape']), name == 'down')
            result = tucker(tensor, tuple(cfg['ranks']), bases[name])
            result.core = result.core.to(torch.bfloat16)
            result.factors = [f.to(torch.bfloat16) for f in result.factors]
            packed[f'{name}.core'] = result.core.cpu()
            packed.update({f'{name}.factor{i}': f.cpu() for i, f in enumerate(result.factors)})
            result.core = result.core.float()
            result.factors = [f.float() for f in result.factors]
            approximate[name] = untensorize(result.reconstruct(), name == 'down').to(torch.bfloat16)
        if save_path:
            save_file(packed, str(save_path))
        return permute_weights(approximate, perm.argsort().to('cuda'))

    @torch.inference_mode()
    def reconstruct_svd(self, budget, bases, save_path=None):
        ranks, padding = svd_budget(budget, *self.weights['gate'].shape)
        approximate, packed = {}, {'alignment_padding': torch.zeros(padding, dtype=torch.bfloat16)}
        for name, rank in zip(NAMES, ranks):
            logical = self.dense_weights[name].T if name == 'down' else self.dense_weights[name]
            left, right = svd_factors(logical, bases[name], rank)
            left, right = left.to(torch.bfloat16), right.to(torch.bfloat16)
            packed[f'{name}.left'], packed[f'{name}.right'] = left.cpu(), right.cpu()
            reconstructed = (left.float() @ right.float()).to(torch.bfloat16)
            approximate[name] = reconstructed.T.contiguous() if name == 'down' else reconstructed
        if save_path:
            save_file(packed, str(save_path))
        return approximate, ranks, padding

    def candidate(self, strategy, cfg, replica, perm=None, bases=None, diagnostics=None):
        values = {'layer': self.layer, 'strategy': strategy, **self.budget_values(cfg), **self.seeds(replica, strategy)}
        values['candidate_id'] = f'L{self.layer}-{strategy}-{cfg["name"]}-R{replica}'
        if values['candidate_id'] in self.completed_candidates:
            return
        start = time.perf_counter()
        approximate = None
        try:
            torch.cuda.reset_peak_memory_stats()
            if strategy == 'svd':
                approximate, ranks, padding = self.reconstruct_svd(values['effective_parameter_count'], bases)
                values.update(svd_ranks=ranks, factor_parameter_count=sum(ranks) * sum(self.weights['gate'].shape),
                              permutation_bytes=0, padding_parameter_count=padding)
            else:
                approximate = self.reconstruct_tucker(perm, cfg, bases)
                values['permutation_sha256'] = hashlib.sha256(perm.numpy().tobytes()).hexdigest()
                values['group_diagnostics'] = diagnostics
            metrics = self.local_metrics(approximate)
            self.record('phase2_local', **values, **metrics, elapsed_seconds=time.perf_counter() - start,
                        peak_vram_bytes=torch.cuda.max_memory_allocated())
        except torch.cuda.OutOfMemoryError as exc:
            self.record('phase2_local', **values, status='oom', error=str(exc))
        finally:
            install_weights(self.module, self.weights)
            del approximate
            cleanup()

    @torch.inference_mode()
    def local_sweep(self):
        for layer in self.config['layers']:
            self.select_layer(layer)
            dense = self.local_metrics({k: v.to('cuda') for k, v in self.weights.items()})
            self.record('phase2_dense_local', layer=layer, strategy='dense', effective_parameter_count=3*prod(self.weights['gate'].shape),
                        effective_compression_ratio=1., **dense)
            zero = {k: v.to('cuda') for k, v in self.weights.items()}
            zero['down'] = torch.zeros_like(zero['down'])
            metrics = self.local_metrics(zero)
            self.record('phase2_local', layer=layer, strategy='zero-mlp', config_name='zero', replica=0,
                        effective_parameter_count=0, **metrics)
            del zero
            configs = self.config['rank_configs']
            unique_budgets = {}
            for cfg in configs:
                unique_budgets.setdefault(self.budget_values(cfg)['effective_parameter_count'], cfg)
            bases = {name: svd_basis(self.dense_weights[name].T if name == 'down' else self.dense_weights[name]) for name in NAMES}
            for cfg in unique_budgets.values():
                self.candidate('svd', cfg, 0, bases=bases)
            del bases
            replicas = len(self.config['seeds']['projection']) if layer == 14 else self.config['transfer_replicas']
            for strategy in STRATEGIES:
                for replica in range(1 if strategy == 'identity' else replicas):
                    bases = None
                    try:
                        perm, diagnostics = self.order(strategy, replica)
                        save_file({'permutation': perm}, str(self.directory / f'L{layer}-{strategy}-R{replica}-permutation.safetensors'))
                        ordered = permute_weights(self.dense_weights, perm.to('cuda'))
                        bases = {name: hosvd_bases(tensorize(ordered[name], tuple(self.config['shape']), name == 'down')) for name in NAMES}
                        del ordered
                        for cfg in configs:
                            self.candidate(strategy, cfg, replica, perm, bases, diagnostics)
                    except torch.cuda.OutOfMemoryError as exc:
                        self.record('phase2_ordering', layer=layer, strategy=strategy, **self.seeds(replica, strategy), status='oom', error=str(exc))
                    finally:
                        del bases
                        cleanup()
            del self.dense_weights, self.dense_mlp, self.dense_block, self.x
            cleanup()

    @torch.inference_mode()
    def activation_sweep(self):
        if not self.config['activation']['enabled']:
            return
        self.select_layer(self.config['activation']['layer'])
        settings = self.config['activation']
        fingerprint = activation_fingerprint(self.x, self.weights, settings['chunk_tokens'], settings['floor'], settings['normalization'])
        save_file({'fingerprint': fingerprint}, str(self.directory / 'activation_fingerprint.safetensors'))
        self.record('phase2_activation_fingerprint', layer=self.layer, fingerprint_shape=list(fingerprint.shape), normalization=settings)
        configs = [cfg for cfg in self.config['rank_configs'] if cfg['name'] in settings['rank_configs']]
        for replica in range(len(self.config['seeds']['projection'])):
            perm, diagnostics = self.order('structure-aware-activation', replica, fingerprint)
            save_file({'permutation': perm}, str(self.directory / f'L{self.layer}-structure-aware-activation-R{replica}-permutation.safetensors'))
            ordered = permute_weights(self.dense_weights, perm.to('cuda'))
            bases = {name: hosvd_bases(tensorize(ordered[name], tuple(self.config['shape']), name == 'down')) for name in NAMES}
            del ordered
            for cfg in configs:
                self.candidate('structure-aware-activation', cfg, replica, perm, bases, diagnostics)
            del bases
            cleanup()

    def regenerate(self, row):
        cfg = next(c for c in self.config['rank_configs'] if c['name'] == row['config_name'])
        save_path = self.directory / 'factors' / (row['candidate_id'] + '.safetensors')
        if row['strategy'] == 'svd':
            bases = {name: svd_basis(self.dense_weights[name].T if name == 'down' else self.dense_weights[name]) for name in NAMES}
            weights, _, _ = self.reconstruct_svd(row['effective_parameter_count'], bases, save_path)
        else:
            activation = None
            if row['strategy'] == 'structure-aware-activation':
                settings = self.config['activation']
                activation = activation_fingerprint(self.x, self.weights, settings['chunk_tokens'], settings['floor'], settings['normalization'])
            perm, _ = self.order(row['strategy'], row['replica'], activation)
            if hashlib.sha256(perm.numpy().tobytes()).hexdigest() != row['permutation_sha256']:
                raise AssertionError('regenerated permutation differs')
            ordered = permute_weights(self.dense_weights, perm.to('cuda'))
            bases = {name: hosvd_bases(tensorize(ordered[name], tuple(self.config['shape']), name == 'down')) for name in NAMES}
            weights = self.reconstruct_tucker(perm, cfg, bases, save_path)
        metrics = self.local_metrics(weights)
        if any(abs(metrics[k] - row[k]) > 1e-6 for k in metrics):
            raise AssertionError('regenerated candidate metrics differ')
        return weights, save_path

    @torch.inference_mode()
    def quality_sweep(self):
        dense = self.perplexity()
        self.record('phase2_quality', strategy='dense', quality_metrics=dense)
        front, means = pareto_groups(self.rows)
        write_json(self.directory / 'pareto_selection.json', {'frontier': front, 'criteria': list(next(iter(means.values())).keys()),
                    'policy': 'group means; all replicas of each frontier group; no best-seed selection'})
        groups = grouped(self.rows)
        for layer in self.config['layers']:
            self.select_layer(layer)
            try:
                zero = {k: v.to('cuda') for k, v in self.weights.items()}
                zero['down'] = torch.zeros_like(zero['down'])
                install_weights(self.module, zero)
                self.record('phase2_quality', layer=layer, strategy='zero-mlp', quality_metrics=self.perplexity())
            finally:
                install_weights(self.module, self.weights)
            del zero
            for key in (k for k in front if k[0] == layer):
                for row in groups[key]:
                    weights = None
                    try:
                        weights, artifact = self.regenerate(row)
                        install_weights(self.module, weights)
                        values = {k: row[k] for k in ('candidate_id', 'layer', 'strategy', 'config_name', 'ranks', 'replica',
                                  'projection_seed', 'clustering_seed', 'random_permutation_seed', 'effective_parameter_count', 'effective_compression_ratio')}
                        self.record('phase2_quality', **values, factor_artifact=str(artifact.relative_to(self.directory)),
                                    factor_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), quality_metrics=self.perplexity())
                    except torch.cuda.OutOfMemoryError as exc:
                        self.record('phase2_quality', candidate_id=row['candidate_id'], layer=layer, strategy=row['strategy'], status='oom', error=str(exc))
                    finally:
                        install_weights(self.module, self.weights)
                        del weights
                        cleanup()
            del self.dense_weights, self.dense_mlp, self.dense_block, self.x
            cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase2.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=['all', 'local', 'report'], default='all')
    parser.add_argument('--report-run')
    parser.add_argument('--full-run')
    parser.add_argument('--resume-run')
    args = parser.parse_args()
    root = Path.cwd()
    config = yaml.safe_load((root / args.config).read_text())
    validate(config)
    configure(config['seed'])
    if args.stage == 'report':
        from .phase2_report import report
        report(root, root / args.report_run, root / args.full_run if args.full_run else None)
        return
    if args.smoke:
        config = copy.deepcopy(config)
        config['layers'] = [14]
        config['seeds'] = {k: v[:1] for k, v in config['seeds'].items()}
        config['rank_configs'] = config['rank_configs'][1:4]
        config['quality'].update(calibration_tokens=128, evaluation_tokens=256, block_size=128, stride=64)
        config['activation']['rank_configs'] = ['medium-A']
    run = Phase2(config, root, args.smoke)
    (run.directory / 'resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'PHASE2 RUN {run.run_id}', flush=True)
    run.load()
    run.capture_all()
    if args.resume_run:
        run.reuse_local(root / args.resume_run)
    run.local_sweep()
    run.activation_sweep()
    if args.stage == 'all':
        run.quality_sweep()
    write_json(run.directory / 'completion.json', {'local_and_subset_complete': True, 'smoke': args.smoke, 'rows': len(run.rows)})
    print(f'PHASE2 COMPLETE {run.run_id}', flush=True)


if __name__ == '__main__':
    main()
