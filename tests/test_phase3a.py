import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from swiglu_fold.ops import mlp
from swiglu_fold.phase2_ops import svd_basis, svd_budget, svd_factors
from swiglu_fold.phase3a_ops import (NAMES, allocate, analytic_factors, factor_mlp, freeze_base,
                                   parameter_count, reconstruct, replay, residual_bases,
                                   save_factors, split_indices, stacked_basis)


def test_principal_angles_and_projection_overlap_agree():
    from swiglu_fold.phase3a_ops import subspace_metrics
    torch.manual_seed(31)
    a = torch.linalg.qr(torch.randn(96, 32, dtype=torch.float64)).Q
    b = torch.linalg.qr(torch.randn(96, 32, dtype=torch.float64)).Q
    cross = a.T@b
    result = subspace_metrics(cross)
    assert result['mean_squared_canonical_correlation'] == pytest.approx(result['normalized_projection_overlap'], abs=1e-12)
    result = subspace_metrics(torch.eye(32, dtype=torch.float64))
    assert result['principal_angles_degrees'] == [0.]*32 and result['normalized_projection_overlap'] == 1.
    result = subspace_metrics(torch.zeros(32, 32, dtype=torch.float64))
    assert result['principal_angles_degrees'] == [90.]*32 and result['normalized_projection_overlap'] == 0.
    with pytest.raises(ValueError):
        subspace_metrics(torch.ones(2, 3))


def test_shared_and_common_private_math_and_serialized_budget(tmp_path):
    torch.manual_seed(19)
    w = {n: torch.randn(48, 16, dtype=torch.float64) for n in NAMES}
    u, energy = stacked_basis(w)
    actual_u, s, _ = torch.linalg.svd(torch.cat(list(w.values()), dim=1))
    torch.testing.assert_close(energy, s.square(), atol=1e-10, rtol=1e-10)
    common = u[:, :5]
    torch.testing.assert_close(common @ common.T, actual_u[:, :5] @ actual_u[:, :5].T, atol=1e-12, rtol=1e-12)
    residuals, bases, spectra = residual_bases(w, common)
    ranks = [2, 3, 4]
    count = parameter_count(5, ranks, 48, 16)
    assert count == 96*5+64*9
    assert parameter_count(7, [0, 0, 0], 6144, 2048) == 12288*7
    factors = analytic_factors(w, common, residuals, bases, ranks, count+7)
    for name, rank in zip(NAMES, ranks):
        left, right = svd_factors(residuals[name], bases[name], rank)
        expected = common @ (common.T @ w[name]) + left @ right
        actual = reconstruct(factors, bf16_weights=False)[name]
        torch.testing.assert_close(actual.T if name == 'down' else actual, expected.float(), atol=1e-6, rtol=1e-6)
    path = tmp_path / 'factors.safetensors'
    save_factors(path, factors)
    stored = load_file(str(path))
    assert all(t.dtype == torch.bfloat16 for t in stored.values())
    assert sum(t.numel() for t in stored.values()) == count+7
    assert stored['alignment_padding'].count_nonzero() == 0
    assert all(torch.equal(replay(path, 'cpu')[n], reconstruct(stored)[n]) for n in NAMES)
    x = torch.randn(6, 16)
    fp32 = {k: v.float() for k, v in stored.items()}
    torch.testing.assert_close(factor_mlp(x, fp32), mlp(x, reconstruct(fp32, False)), atol=2e-4, rtol=2e-5)


def test_rc_zero_phase2_equivalence_and_greedy():
    torch.manual_seed(27)
    w = {n: torch.randn(48, 16) for n in NAMES}
    common = torch.empty(48, 0)
    residuals, bases, spectra = residual_bases(w, common)
    budget = 64*17+11
    ranks, padding = svd_budget(budget, 48, 16)
    assert ranks == allocate(spectra, 17, 'equal')
    factors = analytic_factors(w, common, residuals, bases, ranks, budget)
    for name, rank in zip(NAMES, ranks):
        left, right = svd_factors(w[name], svd_basis(w[name]), rank)
        assert torch.equal(factors[f'{name}.private_left'], left)
        assert torch.equal(factors[f'{name}.private_right'], right)
    assert factors['alignment_padding'].numel() == padding
    assert allocate([torch.tensor([10., 9., 1.]), torch.tensor([8., 2., 1.]), torch.tensor([7., 6., 1.])], 4, 'energy-greedy') == [2, 1, 1]
    assert allocate([torch.ones(3)]*3, 4, 'energy-greedy') == [3, 1, 0]
    for budget, r, remainder in [(27586944, 2245, 384), (18502368, 1505, 8928)]:
        assert budget // 12288 == r
        assert budget-parameter_count(r, [0]*3, 6144, 2048) == remainder
    with pytest.raises(ValueError):
        allocate([torch.ones(2)]*3, 7, 'equal')
    with pytest.raises(ValueError):
        parameter_count(-1, [1]*3, 48, 16)


def test_deterministic_disjoint_split_and_freeze():
    splits = split_indices(2048, 1024, .75, 4242)
    assert [len(v) for v in splits.values()] == [768, 256, 1024]
    assert torch.equal(torch.cat(list(splits.values())).sort().values, torch.arange(2048))
    assert all(torch.equal(v, split_indices(2048, 1024, .75, 4242)[k]) for k, v in splits.items())
    base = torch.nn.Linear(16, 48)
    freeze_base(base)
    assert not base.training and all(not p.requires_grad for p in base.parameters())
    import copy
    import yaml
    from swiglu_fold.phase3a import validate
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/phase3a.yaml').read_text())
    validate(config)
    for values in ([5101, 5101], [5101, 5101, 5107]):
        invalid = copy.deepcopy(config)
        invalid['tuning']['seeds'] = values
        with pytest.raises(ValueError):
            validate(invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='actual tuning/checkpoint path needs CUDA')
def test_training_uses_validation_checkpoint_and_preserves_base(tmp_path, monkeypatch):
    from swiglu_fold.phase3a import Phase3A
    from swiglu_fold.run import install_weights, model_weights
    run = Phase3A.__new__(Phase3A)
    run.rows = []
    run.directory = tmp_path
    (tmp_path/'factors').mkdir()
    run.config = {'tuning': {'optimizer': 'AdamW', 'weight_decay': 0., 'steps': 2, 'batch_tokens': 8,
                            'lambda_mlp': 1., 'lambda_block': 0., 'validation_interval': 1,
                            'minimum_improvement': .0001, 'patience': 2}}
    run.layer = 14
    w = {n: torch.randn(48, 16, device='cuda')*.1 for n in NAMES}
    common = torch.empty(48, 0, device='cuda')
    residuals, bases, spectra = residual_bases(w, common)
    factors = analytic_factors(w, common, residuals, bases, [5]*3, 960)
    path = tmp_path/'factors/initial.safetensors'
    save_factors(path, factors)
    run.module = torch.nn.Module()
    for name in NAMES:
        shape = (16, 48) if name == 'down' else (48, 16)
        setattr(run.module, name+'_proj', torch.nn.Linear(shape[1], shape[0], bias=False).cuda())
    run.weights = {n: w[n].T.contiguous() if n == 'down' else w[n] for n in NAMES}
    install_weights(run.module, run.weights)
    run.model = run.module
    freeze_base(run.model)
    before = model_weights(run.module)
    run.x = torch.randn(1, 32, 16, device='cuda')
    run.dense_mlp = mlp(run.x, run.weights)
    run.splits = split_indices(32, 24, .75, 9)
    metrics = iter([{'validation_mlp_output_relative_error': .8, 'train_mlp_output_relative_error': .6},
                    {'validation_mlp_output_relative_error': .9, 'train_mlp_output_relative_error': .3}])
    monkeypatch.setattr(run, 'local_metrics', lambda w: next(metrics))
    monkeypatch.setattr(run, 'evaluate', lambda path, values: values)
    monkeypatch.setattr(run, 'record', lambda kind, **values: values)
    initial = dict(candidate_id='initial', factor_artifact='factors/initial.safetensors',
                   layer=14, budget_name='toy', allocation='equal', shared_fraction=0., common_rank=0,
                   private_ranks=[5]*3, factor_parameter_count=960, padding_parameter_count=0,
                   effective_parameter_count=960, effective_compression_ratio=2.4, effective_storage_bytes=1920,
                   strategy='independent-svd', validation_mlp_output_relative_error=1.,
                   train_mlp_output_relative_error=1., decomposition_seconds=0.)
    row = run.tune(initial, .0001, 5101)
    assert row['best_checkpoint_step'] == 1 and row['completed_steps'] == 2
    assert all(torch.equal(model_weights(run.module)[n], before[n]) for n in NAMES)
    assert all(p.grad is None for p in run.model.parameters())
    curve = json.loads((tmp_path/row['validation_curve']).read_text())
    assert [r.get('selected') for r in curve[1:]] == [True, False]
    from types import SimpleNamespace
    run.module.forward = lambda x: mlp(x, run.weights)
    original_forward = run.module.forward
    class Block:
        def __call__(self, x):
            return x + run.module(x)
    run.model.model = SimpleNamespace(layers={14: Block()})
    run.captures = {14: {'blocks': [{'args': (run.x.detach().cpu(),), 'kwargs': {},
                                   'dense_output': (run.x+run.dense_mlp).detach().cpu()}]}}
    trainable = {k: torch.nn.Parameter(v.cuda().float()) if v.numel() and k != 'alignment_padding' else v.cuda().float()
                 for k, v in load_file(str(path)).items()}
    block_loss = run.block_loss(trainable, run.splits['train'])
    block_loss.backward()
    assert torch.isfinite(block_loss) and any(v.grad is not None for v in trainable.values() if isinstance(v, torch.nn.Parameter))
    assert all(p.grad is None for p in run.model.parameters()) and run.module.forward is original_forward
    from safetensors.torch import save_file
    import swiglu_fold.phase3a as experiment
    run.config['budgets'] = {'toy': 12288*8}
    run.dense_weights = run.weights
    save_file({'basis': torch.randn(48, 8)}, str(tmp_path/'L14-shared-basis.safetensors'))
    failures = []
    monkeypatch.setattr(run, 'record', lambda kind, **v: failures.append(v))
    def oom(*args):
        install_weights(run.module, {k: w*0 for k, w in run.weights.items()})
        raise torch.cuda.OutOfMemoryError('injected residual decomposition OOM')
    monkeypatch.setattr(experiment, 'residual_bases', oom)
    assert run.sweep_fraction('toy', .5) == [] and failures[0]['status'] == 'oom'
    assert all(torch.equal(model_weights(run.module)[n], before[n]) for n in NAMES)


def test_phase1_and_phase2_artifacts_and_history_preserved():
    root = Path(__file__).resolve().parents[1]
    guard = json.loads((root/'results/phase3a_history_guard.json').read_text())
    with (root/'results/experiments.jsonl').open('rb') as stream:
        assert hashlib.sha256(stream.read(guard['bytes'])).hexdigest() == guard['sha256']
    for name, digest in guard['protected_files'].items():
        with (root/name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name


def test_real_phase3a_saved_pareto_replay(request):
    from swiglu_fold.ops import relative_error
    weight_path = request.config.getoption('--layer-artifact')
    if not weight_path:
        pytest.skip('supply real Qwen weights after the Phase-3A experiment')
    root = Path(__file__).resolve().parents[1]
    summary_path = root/'results/phase3a_summary.json'
    if not summary_path.exists():
        pytest.skip('Phase-3A research run has not completed yet')
    assert torch.cuda.is_available()
    summary = json.loads(summary_path.read_text())
    directory = root/'results/runs'/summary['run_id']
    calibration = load_file(str(directory/'phase2_calibration.safetensors'))['layer14_mlp_inputs'].cuda().float()
    dense = {n: w.cuda().float() for n, w in load_file(weight_path).items()}
    targets = mlp(calibration, dense).flatten(0, 1)
    splits = json.loads((directory/'calibration_split.json').read_text())
    for row in summary['candidates']:
        if row['candidate_id'] not in summary['pareto_candidate_ids'] or row['layer'] != 14:
            continue
        weights = replay(directory/row['factor_artifact'])
        actual = mlp(calibration, {n: w.float() for n, w in weights.items()}).flatten(0, 1)
        for split in ('train', 'validation', 'confirmation'):
            indices = torch.tensor(splits[split])
            assert relative_error(actual[indices], targets[indices]) == pytest.approx(row[f'{split}_mlp_output_relative_error'], abs=1e-6)
