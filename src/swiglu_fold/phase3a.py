from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import time
import shutil
import zipfile

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file

from .ops import mlp, relative_error
from .phase2 import Phase2, move
from .phase2_ops import functional_metrics
from .phase3a_ops import (NAMES, allocate, analytic_factors, checkpoint_improved, factor_mlp,
                         freeze_base, parameter_count, replay, residual_bases, save_factors,
                         split_indices, stacked_basis, subspace_metrics)
from .run import Experiment, cleanup, configure, install_weights
from .telemetry import write_json


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(config):
    if (config['layers'] != [14, 0, 7, 21, 27] or config['layer'] != 14
            or config['budgets'] != {'high': 27586944, 'medium': 18502368}
            or config['shared_fractions'] != [0., .25, .5, .75, 1.]
            or config['device'] != 'cuda' or config['dtype'] != 'bfloat16'):
        raise ValueError('Phase 3A requires the fixed two-budget, five-layer protocol')
    tuning = config['tuning']
    if (not 1 <= len(tuning['learning_rates']) <= 3 or len(tuning['seeds']) != 2 or len(set(tuning['seeds'])) != 2
            or tuning['optimizer'] != 'AdamW' or tuning['steps'] < 1
            or tuning['batch_tokens'] < 1 or tuning['validation_interval'] < 1
            or tuning['patience'] < 1 or tuning['weight_decay'] < 0
            or tuning['lambda_mlp'] < 0 or tuning['lambda_block'] < 0
            or tuning['lambda_mlp'] + tuning['lambda_block'] == 0
            or any(lr <= 0 for lr in tuning['learning_rates'])):
        raise ValueError('invalid bounded functional tuning configuration')
    split_indices(config['quality']['calibration_tokens'], **config['split'])


class Phase3A(Phase2):
    def __init__(self, config, root, smoke):
        Experiment.__init__(self, config, root, smoke)
        if not self.metadata['git_commit']:
            raise RuntimeError('Phase 3A requires an actual Git commit')
        self.metadata.update(algorithm_version='phase3a-joint-subspace-1',
                             metric_schema='phase3a-heldout-fp32-mlp-bf16-block-1',
                             mlp_metric_precision='canonical FP32; saved BF16 factors reconstructed into BF16 weights',
                             primary_local_metric='validation_mlp_output_relative_error',
                             block_metric_precision='actual full BF16 transformer block replay',
                             budget_unit='BF16 storage element; saved zero remainder padding included',
                             inference_scope='one reconstructed MLP at a time; no runtime optimization')
        self.captures, self.layer_weights = {}, {}

    def record(self, kind, **values):
        return Experiment.record(self, kind, **values)

    @torch.inference_mode()
    def diagnostics(self):
        source = self.root / self.config['diagnostic_source']
        previous = json.loads((source / 'metadata.json').read_text())
        if previous['hf_model_revision_sha'] != self.config['revision']:
            raise ValueError('diagnostic weights belong to another model revision')
        self.metadata['diagnostic_source_run'] = source.name
        self.metadata['hf_model_revision_sha'] = previous['hf_model_revision_sha']
        write_json(self.directory/'metadata.json', self.metadata)
        for layer in self.config['layers']:
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            path = source / f'layer{layer}.safetensors'
            weights = load_file(str(path), device='cuda')
            logical = {n: (weights[n].T if n == 'down' else weights[n]).float() for n in NAMES}
            shared, energy = stacked_basis(logical)
            max_rank = max(self.config['budgets'].values()) // 12288
            save_file({'basis': shared[:, :max_rank].cpu().contiguous()}, str(self.directory / f'L{layer}-shared-basis.safetensors'))
            left, individual = {}, {}
            for name in NAMES:
                eigen, right = torch.linalg.eigh(logical[name].T @ logical[name])
                eigen, right = eigen.flip(0).clamp_min(0), right.flip(1)
                individual[name] = eigen
                vectors = logical[name] @ right / eigen.sqrt().clamp_min(1e-20)
                left[name] = torch.linalg.qr(vectors, mode='reduced').Q
            for name, spectrum in {**individual, 'stacked': energy}.items():
                self.record('phase3a_spectrum', layer=layer, matrix=name,
                            singular_values=spectrum.sqrt().cpu().tolist(),
                            cumulative_energy=(spectrum.cumsum(0) / spectrum.sum()).cpu().tolist())
            coefficients = {n: shared.T @ w for n, w in logical.items()}
            for a, b in (('gate', 'up'), ('gate', 'down'), ('up', 'down')):
                cross = left[a].T @ left[b]
                for k in self.config['diagnostic_k']:
                    c = cross[:k, :k]
                    geometry = subspace_metrics(c)
                    captured = {n: float(coefficients[n][:k].square().sum() / w.square().sum()) for n, w in logical.items()}
                    self.record('phase3a_subspace', layer=layer, k=k, pair=f'{a}/{b}',
                                **geometry,
                                shared_energy_captured=captured,
                                individual_energy_captured={n: float(s[:k].sum()/s.sum()) for n, s in individual.items()},
                                stacked_energy_captured=float(energy[:k].sum()/energy.sum()))
                    if layer == 7 and k == 512 and a == 'up' and b == 'down':
                        self.record('phase3a_subspace_precision_control', layer=layer, k=k, pair=f'{a}/{b}',
                                    previous_fp32_gpu_gram_eigen_mean=float(torch.linalg.eigvalsh(c.T@c).clamp(0, 1).mean()),
                                    fp32_gpu_projection_overlap=float(c.square().sum()/k), **geometry)
            self.record('phase3a_diagnostic_timing', layer=layer, weight_artifact_sha256=sha(path),
                        decomposition_seconds=time.perf_counter()-start,
                        peak_vram_bytes=torch.cuda.max_memory_allocated())
            del weights, logical, shared, left, coefficients, cross, vectors, c
            cleanup()

    def prepare(self):
        self.load()
        self.capture_all()
        freeze_base(self.model)
        source = self.root / self.config['diagnostic_source']
        for layer in self.config['layers']:
            old = load_file(str(source / f'layer{layer}.safetensors'))
            if any(not torch.equal(old[n], self.layer_weights[layer][n]) for n in NAMES):
                raise AssertionError('diagnostic and current model weights differ')
        self.splits = split_indices(self.train.shape[1], **self.config['split'])
        write_json(self.directory / 'calibration_split.json', {k: v.tolist() for k, v in self.splits.items()})
        self.metadata['calibration_split_sha256'] = sha(self.directory / 'calibration_split.json')
        self.metadata['base_frozen'] = all(not p.requires_grad for p in self.model.parameters())
        write_json(self.directory / 'metadata.json', self.metadata)

    @torch.inference_mode()
    def local_metrics(self, weights):
        try:
            actual = mlp(self.x.float(), {n: w.float() for n, w in weights.items()}).flatten(0, 1)
            expected = self.dense_mlp.flatten(0, 1)
            install_weights(self.module, weights)
            outputs = [self.model.model.layers[self.layer](*move(b['args'], 'cuda'), **move(b['kwargs'], 'cuda'))
                       for b in self.captures[self.layer]['blocks']]
            actual_block = torch.cat(outputs, dim=1).flatten(0, 1)
            dense_block = self.dense_block.flatten(0, 1)
            result = {}
            for split, indices in self.splits.items():
                result.update({f'{split}_{k}': v for k, v in functional_metrics(
                    actual[indices], expected[indices], actual_block[indices], dense_block[indices]).items()})
            result.update(functional_metrics(actual[self.splits['validation']], expected[self.splits['validation']],
                                              actual_block[self.splits['validation']], dense_block[self.splits['validation']]))
            result.update({f'all_{k}': v for k, v in functional_metrics(actual, expected, actual_block, dense_block).items()})
            return result
        finally:
            install_weights(self.module, self.weights)

    @torch.inference_mode()
    def evaluate(self, path, values):
        start = time.perf_counter()
        weights = replay(path)
        metrics = self.local_metrics(weights)
        errors, norm = 0., 0.
        for name in NAMES:
            dense = self.dense_weights[name]
            error = (weights[name].float() - dense).square().sum().item()
            energy = dense.square().sum().item()
            metrics[f'{name}_relative_frobenius_error'] = (error / energy)**.5
            metrics[f'{name}_weight_energy_explained'] = 1-error/energy
            errors += error
            norm += energy
        metrics['aggregate_relative_frobenius_error'] = (errors/norm)**.5
        return self.record('phase3a_local', **values, **metrics, evaluation_seconds=time.perf_counter()-start,
                           peak_vram_bytes=torch.cuda.max_memory_allocated(),
                           factor_artifact=str(path.relative_to(self.directory)), factor_artifact_sha256=sha(path))

    def build_candidate(self, budget_name, fraction, allocation, common, residuals, bases, spectra, elapsed):
        budget = self.config['budgets'][budget_name]
        remaining = 0 if fraction == 1 else (budget - common.shape[1]*12288) // 8192
        ranks = allocate(spectra, remaining, allocation)
        factors = analytic_factors({n: (w.T if n == 'down' else w) for n, w in self.dense_weights.items()},
                                  common, residuals, bases, ranks, budget)
        count = parameter_count(common.shape[1], ranks, 6144, 2048)
        method = ('independent-svd' if allocation == 'equal' else 'private-only-energy') if fraction == 0 else (
            'shared-only' if fraction == 1 else 'common-private')
        identity = f'L{self.layer}-{budget_name}-f{fraction:.3f}-{allocation}'
        values = dict(candidate_id=identity, layer=self.layer, budget_name=budget_name, strategy=method,
                      allocation=allocation, shared_fraction=fraction, common_rank=common.shape[1], private_ranks=ranks,
                      factor_parameter_count=count, padding_parameter_count=budget-count,
                      effective_parameter_count=budget, effective_compression_ratio=37748736/budget,
                      effective_storage_bytes=2*budget, decomposition_seconds=elapsed)
        path = self.directory / 'factors' / f'{identity}.safetensors'
        save_factors(path, factors)
        del factors
        return self.evaluate(path, values)

    @torch.inference_mode()
    def sweep_fraction(self, budget_name, fraction, allocations=('equal', 'energy-greedy')):
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            rank = int(fraction * self.config['budgets'][budget_name] // 12288)
            common = load_file(str(self.directory / f'L{self.layer}-shared-basis.safetensors'), device='cuda')['basis'][:, :rank]
            logical = {n: (w.T if n == 'down' else w) for n, w in self.dense_weights.items()}
            residuals, bases, spectra = residual_bases(logical, common)
        except torch.cuda.OutOfMemoryError as exc:
            self.record('phase3a_decomposition', layer=self.layer, budget_name=budget_name, shared_fraction=fraction,
                        status='oom', error=str(exc))
            install_weights(self.module, self.weights)
            cleanup()
            return []
        elapsed = time.perf_counter()-start
        rows = []
        for allocation in allocations[:1] if fraction == 1 else allocations:
            try:
                rows.append(self.build_candidate(budget_name, fraction, allocation, common, residuals, bases, spectra, elapsed))
            except torch.cuda.OutOfMemoryError as exc:
                self.record('phase3a_local', layer=self.layer, budget_name=budget_name, shared_fraction=fraction,
                            allocation=allocation, status='oom', error=str(exc))
            finally:
                install_weights(self.module, self.weights)
                cleanup()
        return rows

    def analytic_sweep(self):
        self.select_layer(14)
        for method in ('dense', 'zero-mlp'):
            weights = {n: w.to('cuda') for n, w in self.weights.items()}
            if method == 'zero-mlp':
                weights['down'] = torch.zeros_like(weights['down'])
            self.record('phase3a_control', layer=14, strategy=method, **self.local_metrics(weights))
        for name in self.config['budgets']:
            rows = []
            for fraction in self.config['shared_fractions']:
                rows.extend(self.sweep_fraction(name, fraction))
            nonzero = [r for r in rows if r['common_rank'] > 0]
            best = min(nonzero, key=lambda r: r['validation_mlp_output_relative_error'])
            # ponytail: two local neighbors only; more splits require a new demonstrated signal.
            for fraction in (best['shared_fraction']-.125, best['shared_fraction']+.125):
                if 0 < fraction < 1:
                    rows.extend(self.sweep_fraction(name, fraction))
        cleanup()

    def block_loss(self, factors, indices):
        block = self.model.model.layers[self.layer]
        original = self.module.forward
        self.module.forward = lambda x: factor_mlp(x.float(), factors).to(torch.bfloat16)
        squared, denominator, offset = 0., 0., 0
        try:
            for capture in self.captures[self.layer]['blocks']:
                size = capture['args'][0].shape[1]
                local = indices[(indices >= offset) & (indices < offset+size)] - offset
                if len(local):
                    output = block(*move(capture['args'], 'cuda'), **move(capture['kwargs'], 'cuda')).flatten(0, 1)[local].float()
                    target = capture['dense_output'].to('cuda').flatten(0, 1)[local].float()
                    squared = squared + (output-target).square().sum()
                    denominator = denominator + target.square().sum()
                offset += size
            return squared / denominator.clamp_min(1e-30)
        finally:
            self.module.forward = original

    def tune(self, initial, lr, seed):
        settings = self.config['tuning']
        identity = f"{initial['candidate_id']}-tuned-lr{lr:g}-s{seed}"
        cached = next((r for r in self.rows if r.get('candidate_id') == identity and r['status'] == 'ok'), None)
        if cached:
            return cached
        path = self.directory / 'factors' / f'{identity}.safetensors'
        last_path = self.directory / 'factors' / f'{identity}-validation.safetensors'
        values = {k: initial[k] for k in ('layer', 'budget_name', 'allocation', 'shared_fraction', 'common_rank',
                  'private_ranks', 'factor_parameter_count', 'padding_parameter_count', 'effective_parameter_count',
                  'effective_compression_ratio', 'effective_storage_bytes')}
        values.update(candidate_id=identity, strategy='functional-'+initial['strategy'], initial_candidate_id=initial['candidate_id'],
                      optimizer='AdamW', learning_rate=lr, training_seed=seed, training_settings=settings)
        start = time.perf_counter()
        factors, optimizer = None, None
        try:
            torch.cuda.reset_peak_memory_stats()
            data = load_file(str(self.directory / initial['factor_artifact']), device='cuda')
            factors = {k: torch.nn.Parameter(v.float()) if v.numel() and k != 'alignment_padding' else v.float()
                       for k, v in data.items()}
            del data
            parameters = [v for v in factors.values() if isinstance(v, torch.nn.Parameter)]
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=settings['weight_decay'], foreach=False)
            if any(p.requires_grad for p in self.model.parameters()) or any(p.grad is not None for p in self.model.parameters()):
                raise AssertionError('base model must remain frozen')
            save_factors(path, factors)
            best = initial['validation_mlp_output_relative_error']
            best_step, stale = 0, 0
            generator = torch.Generator().manual_seed(seed)
            x, targets = self.x.flatten(0, 1).float().clone(), self.dense_mlp.flatten(0, 1).detach().clone()
            train = self.splits['train']
            norm = targets[train].square().mean().clamp_min(1e-30)
            curve = [{'step': 0, 'validation': best, 'train': initial['train_mlp_output_relative_error']}]
            for step in range(1, settings['steps']+1):
                indices = train[torch.randperm(len(train), generator=generator)[:settings['batch_tokens']]]
                optimizer.zero_grad(set_to_none=True)
                output = factor_mlp(x[indices], factors)
                loss = settings['lambda_mlp'] * (output-targets[indices]).square().mean()/norm
                if settings['lambda_block']:
                    loss = loss + settings['lambda_block'] * self.block_loss(factors, indices)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                optimizer.step()
                if step % settings['validation_interval'] == 0 or step == settings['steps']:
                    save_factors(last_path, factors)
                    metrics = self.local_metrics(replay(last_path))
                    validation = metrics['validation_mlp_output_relative_error']
                    improved = checkpoint_improved(validation, best, settings['minimum_improvement'])
                    curve.append({'step': step, 'batch_normalized_loss': float(loss.detach()),
                                  'train': metrics['train_mlp_output_relative_error'], 'validation': validation,
                                  'selected': improved})
                    if improved:
                        best, best_step, stale = validation, step, 0
                        # Persist exactly the already evaluated BF16 checkpoint.
                        path.write_bytes(last_path.read_bytes())
                    else:
                        stale += 1
                    write_json(self.directory / f'{identity}-curve.json', curve)
                    print(json.dumps({'tuning': identity, 'step': step, 'train': curve[-1]['train'], 'validation': validation}), flush=True)
                    if stale >= settings['patience']:
                        break
            if any(p.grad is not None for p in self.model.parameters()):
                raise AssertionError('training changed base model gradient state')
            values.update(best_checkpoint_step=best_step, completed_steps=step, training_seconds=time.perf_counter()-start,
                          early_stopped=step < settings['steps'], decomposition_seconds=initial['decomposition_seconds'],
                          validation_curve=f'{identity}-curve.json', base_frozen=True)
            row = self.evaluate(path, values)
            return row
        except torch.cuda.OutOfMemoryError as exc:
            self.record('phase3a_local', **values, status='oom', error=str(exc))
            return None
        finally:
            install_weights(self.module, self.weights)
            del factors, optimizer
            cleanup()

    def functional_sweep(self):
        for name in self.config['budgets']:
            rows = [r for r in self.rows if r['kind'] == 'phase3a_local' and r.get('status') == 'ok'
                    and r['layer'] == 14 and r['budget_name'] == name and not r['strategy'].startswith('functional-')]
            baseline = next(r for r in rows if r['strategy'] == 'independent-svd')
            joint = min((r for r in rows if r['common_rank'] > 0), key=lambda r: r['validation_mlp_output_relative_error'])
            private_joint = min((r for r in rows if r['strategy'] == 'common-private'), key=lambda r: r['validation_mlp_output_relative_error'])
            promising = [r for r in (joint, private_joint) if r['validation_mlp_output_relative_error'] <=
                         self.config['tuning']['close_ratio']*baseline['validation_mlp_output_relative_error']]
            promising = list({r['candidate_id']: r for r in promising}.values())
            if not promising:
                self.record('phase3a_gate', budget_name=name, functional_tuning=False, reason='best nonzero-common analytic result outside fixed 5% proximity')
                continue
            for initial in (baseline, *promising):
                results = [self.tune(initial, lr, self.config['tuning']['seeds'][0]) for lr in self.config['tuning']['learning_rates']]
                results = [r for r in results if r]
                if results:
                    chosen = min(results, key=lambda r: r['validation_mlp_output_relative_error'])
                    if not any(r['kind'] == 'phase3a_tuning_selection' and r['initial_candidate_id'] == initial['candidate_id'] for r in self.rows):
                        self.record('phase3a_tuning_selection', initial_candidate_id=initial['candidate_id'],
                                    chosen_learning_rate=chosen['learning_rate'], policy='validation only; confirmation and WikiText test not used')
                    self.tune(initial, chosen['learning_rate'], self.config['tuning']['seeds'][1])

    def reuse_primary(self, directory):
        previous = json.loads((directory/'metadata.json').read_text())
        if previous['smoke'] or previous['config'] != self.config:
            raise ValueError('resume requires identical calibration, budgets and tuning configuration')
        with zipfile.ZipFile(directory/'source_snapshot.zip') as archive:
            for name in ('ops.py', 'phase2_ops.py', 'phase3a_ops.py'):
                relative = f'src/swiglu_fold/{name}'
                if archive.read(relative) != (self.root/relative).read_bytes():
                    raise ValueError(f'resume numerical operations differ: {name}')
            def methods(source):
                cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Phase3A')
                return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
            old, new = methods(archive.read('src/swiglu_fold/phase3a.py')), methods(Path(__file__).read_bytes())
            for name in ('prepare', 'local_metrics', 'evaluate', 'build_candidate', 'sweep_fraction'):
                if old[name] != new[name]:
                    raise ValueError(f'resume measurement method differs: {name}')
            # tune changed only by an exact candidate-ID cache; its numerical body is checked separately.
            old_tune = ast.parse(archive.read('src/swiglu_fold/phase3a.py'))
            new_tune = ast.parse(Path(__file__).read_bytes())
            old_function = next(n for c in old_tune.body if isinstance(c, ast.ClassDef) for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'tune')
            new_function = next(n for c in new_tune.body if isinstance(c, ast.ClassDef) for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'tune')
            old_start = next(i for i, n in enumerate(old_function.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'path' for t in n.targets))
            new_start = next(i for i, n in enumerate(new_function.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'path' for t in n.targets))
            if [ast.dump(n) for n in old_function.body[old_start:]] != [ast.dump(n) for n in new_function.body[new_start:]]:
                raise ValueError('resume tuning numerical body differs')
        for key in ('hf_model_revision_sha', 'dataset_revision_sha', 'train_token_sha256', 'test_token_sha256',
                    'calibration_artifact_sha256', 'calibration_split_sha256', 'torch_version', 'cuda_runtime'):
            if previous[key] != self.metadata[key]:
                raise ValueError(f'resume provenance differs: {key}')
        self.metadata['resumed_primary_run'] = previous['run_id']
        write_json(self.directory/'metadata.json', self.metadata)
        copied = set()
        for row in [json.loads(line) for line in (directory/'raw.jsonl').read_text().splitlines()]:
            if row['kind'] not in ('phase3a_spectrum', 'phase3a_subspace', 'phase3a_diagnostic_timing',
                                   'phase3a_local', 'phase3a_control', 'phase3a_tuning_selection'):
                continue
            for key in ('factor_artifact', 'validation_curve'):
                if row.get(key) and row[key] not in copied:
                    shutil.copyfile(directory/row[key], self.directory/row[key])
                    copied.add(row[key])
            values = {k: v for k, v in row.items() if k not in previous and k not in ('kind', 'timestamp', 'run_id')}
            for key in ('reused_measurement', 'measurement_run_id', 'measurement_timestamp', 'measurement_source_sha256'):
                values.pop(key, None)
            self.record(row['kind'], **values, reused_measurement=True, measurement_run_id=row['run_id'],
                        measurement_timestamp=row['timestamp'], measurement_source_sha256=row['source_sha256'])
        for layer in self.config['layers']:
            shutil.copyfile(directory/f'L{layer}-shared-basis.safetensors', self.directory/f'L{layer}-shared-basis.safetensors')

    def confirmed_winners(self):
        winners = []
        for name in self.config['budgets']:
            rows = [r for r in self.rows if r['kind'] == 'phase3a_local' and r['status'] == 'ok'
                    and r['layer'] == 14 and r['budget_name'] == name]
            selections = [r for r in self.rows if r['kind'] == 'phase3a_tuning_selection']
            selected = [r for r in rows if not r['strategy'].startswith('functional-') or any(
                s['initial_candidate_id'] == r['initial_candidate_id'] and s['chosen_learning_rate'] == r['learning_rate'] for s in selections)]
            baseline = [r for r in selected if r['common_rank'] == 0]
            joint = [r for r in selected if r['common_rank'] > 0]
            best = min(joint, key=lambda r: r['validation_mlp_output_relative_error'])
            replicas = [r for r in joint if r['strategy'] == best['strategy'] and r['shared_fraction'] == best['shared_fraction']
                        and r['allocation'] == best['allocation'] and r.get('learning_rate') == best.get('learning_rate')]
            if all(all(r[f'{split}_mlp_output_relative_error'] < min(b[f'{split}_mlp_output_relative_error'] for b in baseline)
                           and r[f'{split}_block_output_relative_error'] <= min(b[f'{split}_block_output_relative_error'] for b in baseline)
                           for split in ('validation', 'confirmation')) for r in replicas):
                winners.append(best)
        return winners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase3a.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=('all', 'diagnostics', 'report'), default='all')
    parser.add_argument('--report-run')
    parser.add_argument('--diagnostics-run')
    parser.add_argument('--resume-run')
    args = parser.parse_args()
    root = Path.cwd()
    if args.stage == 'report':
        from .phase3a_report import report
        report(root, root / args.report_run, root/args.diagnostics_run if args.diagnostics_run else None)
        return
    config = yaml.safe_load((root / args.config).read_text())
    validate(config)
    configure(config['seed'])
    if args.smoke:
        config = copy.deepcopy(config)
        config['layers'] = [14]
        config['quality'].update(calibration_tokens=256, evaluation_tokens=256, block_size=128, stride=64)
        config['split']['selection_tokens'] = 128
        config['diagnostic_k'] = [128]
        config['shared_fractions'] = [0., .5, 1.]
        config['tuning'].update(steps=2, validation_interval=1, learning_rates=[.0001])
    run = Phase3A(config, root, args.smoke)
    (run.directory / 'resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'PHASE3A RUN {run.run_id}', flush=True)
    if args.stage == 'diagnostics':
        run.diagnostics()
        write_json(run.directory/'completion.json', {'diagnostics_complete': True, 'smoke': args.smoke, 'rows': len(run.rows)})
        print(f'PHASE3A DIAGNOSTICS COMPLETE {run.run_id}', flush=True)
        return
    if args.resume_run:
        run.prepare()
        run.reuse_primary(root/args.resume_run)
        run.select_layer(14)
    else:
        run.diagnostics()
        run.prepare()
        run.analytic_sweep()
    run.functional_sweep()
    winners = run.confirmed_winners()
    run.record('phase3a_gate', confirmed_local_winners=[r['candidate_id'] for r in winners],
               reason='require both validation and independent confirmation against strongest matched private-only baseline')
    if winners and not args.smoke:
        raise RuntimeError('Confirmed local win: continue the authorized transfer/quality protocol before reporting completion')
    write_json(run.directory / 'completion.json', {'primary_local_complete': True, 'smoke': args.smoke,
                                                  'confirmed_local_winners': [], 'rows': len(run.rows)})
    print(f'PHASE3A PRIMARY COMPLETE {run.run_id}', flush=True)


if __name__ == '__main__':
    main()
