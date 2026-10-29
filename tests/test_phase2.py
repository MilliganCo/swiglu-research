import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from swiglu_fold.ops import mlp, permutation, permute_weights
from swiglu_fold.phase2_ops import (balanced_assignment, functional_metrics, seeded_order,
                                    svd_basis, svd_budget, svd_factors, tucker_parameters, activation_fingerprint)
from swiglu_fold.telemetry import append_jsonl


def test_balanced_assignment_capacity_determinism_and_optimality():
    for costs in (np.zeros((24, 4)), np.random.default_rng(42).uniform(size=(24, 4))):
        labels, certificate = balanced_assignment(costs, 6)
        assert np.array_equal(np.bincount(labels), [6] * 4)
        assert np.array_equal(labels, balanced_assignment(costs, 6)[0])
        row, col = linear_sum_assignment(np.repeat(costs, 6, axis=1))
        optimum = costs[row, col // 6].sum()
        assert -1e-8 <= certificate['primal_cost'] - optimum <= 24e-4 + 1e-8
        assert certificate['dual_lower_bound'] <= optimum + 1e-8
    with pytest.raises(ValueError):
        balanced_assignment(np.zeros((23, 4)), 6)


def test_independent_seeds_v1_compatibility_and_balanced_symmetry():
    generator = torch.Generator().manual_seed(99)
    w = {k: torch.randn(*shape, generator=generator) for k, shape in
         [('gate', (48, 16)), ('up', (48, 16)), ('down', (16, 48))]}
    kw = dict(projection_seed=42, clustering_seed=42, random_permutation_seed=42, groups=4)
    assert torch.equal(seeded_order('structure-aware', w, **kw)[0], permutation('structure-aware', w, 42, 4))
    perm, diagnostics = seeded_order('structure-aware-balanced', w, **kw)
    assert diagnostics['group_sizes'] == [12] * 4
    assert torch.equal(perm.sort().values, torch.arange(48))
    assert torch.equal(perm, seeded_order('structure-aware-balanced', w, **{**kw, 'random_permutation_seed': 999})[0])
    assert not torch.equal(perm, seeded_order('structure-aware-balanced', w, **{**kw, 'projection_seed': 43})[0])
    assert not torch.equal(perm, seeded_order('structure-aware-balanced', w, **{**kw, 'clustering_seed': 43})[0])
    random = seeded_order('random', w, **kw)[0]
    assert torch.equal(random, seeded_order('random', w, **{**kw, 'projection_seed': 123, 'clustering_seed': 456})[0])
    x = torch.randn(10, 16, generator=generator)
    torch.testing.assert_close(mlp(x, permute_weights(w, perm)), mlp(x, w), atol=1e-4, rtol=1e-4)


def test_svd_budget_and_ordinary_svd_reconstruction():
    shape = (96, 64, 8, 8, 32)
    for group in [((79, 38, 8, 8, 32), (87, 46, 8, 8, 24), (93, 59, 5, 7, 32)),
                  ((40, 51, 8, 8, 32), (75, 30, 8, 8, 29), (90, 56, 6, 6, 23)),
                  ((27, 57, 8, 8, 32), (75, 41, 8, 8, 16), (69, 46, 4, 8, 31))]:
        counts = [3 * tucker_parameters(shape, r) + 6144 * 4 for r in group]
        assert len(set(counts)) == 1
        ranks, padding = svd_budget(counts[0], 6144, 2048)
        assert sum(ranks) * 8192 + padding == counts[0]
        assert 0 <= padding < 8192
    weight = torch.randn(48, 16, dtype=torch.float64)
    left, right = svd_factors(weight, svd_basis(weight), 16)
    torch.testing.assert_close(left @ right, weight, atol=1e-12, rtol=1e-12)
    left, right = svd_factors(weight, svd_basis(weight), 5)
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    torch.testing.assert_close(left @ right, (u[:, :5] * s[:5]) @ vh[:5], atol=1e-12, rtol=1e-12)


def test_batched_continuation_scoring_against_independent_token_loop():
    from types import SimpleNamespace
    from swiglu_fold.phase2_finalists import score_batch, task_choices
    class ToyModel:
        def __call__(self, ids, **kw):
            vocabulary = torch.arange(11)
            logits = -((vocabulary[None,None,:] - (ids[:,:,None]+kw['position_ids'][:,:,None]) % 11).float()).square() / 5
            return SimpleNamespace(logits=logits[:,-kw['logits_to_keep']:])
    requests = [{'tokens':[1,2,3,4,5], 'continuation_tokens':2},
                {'tokens':[8,1,7], 'continuation_tokens':1}]
    scores = score_batch(ToyModel(),requests,0,device='cpu')
    for r, score in zip(requests,scores):
        expected = 0.
        for i in range(len(r['tokens'])-r['continuation_tokens'],len(r['tokens'])):
            distribution = -((torch.arange(11)-(r['tokens'][i-1]+i-1)%11).float()).square()/5
            expected += distribution.log_softmax(0)[r['tokens'][i]].item()
        assert score == pytest.approx(expected,abs=1e-6)
    pairs, gold, lengths = task_choices('winogrande',{'sentence':'The _ is blue.','option1':'sky','option2':'grass','answer':'1'})
    assert pairs == [('The sky',' is blue.'),('The grass',' is blue.')] and gold == 0 and lengths == [1,1]


def test_metric_names_actual_block_and_append_only(tmp_path):
    dense_mlp, dense_block = torch.ones(4, 8), torch.full((4, 8), 100.)
    metrics = functional_metrics(torch.zeros_like(dense_mlp), dense_mlp,
                                 dense_block - 1, dense_block)
    assert metrics == {'mlp_output_mse': 1., 'mlp_output_relative_error': 1.,
                       'block_output_mse': 1., 'block_output_relative_error': pytest.approx(.01)}
    path = tmp_path / 'raw.jsonl'
    append_jsonl(path, {'kind': 'old', 'block_output_mse': 123})
    prefix = path.read_bytes()
    append_jsonl(path, {'kind': 'phase2_local', **metrics})
    assert path.read_bytes().startswith(prefix)
    assert json.loads(path.read_text().splitlines()[0])['block_output_mse'] == 123


def test_actual_phase1_history_unchanged():
    root = Path(__file__).resolve().parents[1]
    guard = json.loads((root / 'results/phase2_history_guard.json').read_text())
    assert hashlib.sha256((root / 'results/experiments.jsonl').read_bytes()[:guard['bytes']]).hexdigest() == guard['sha256']
    for name, digest in guard['protected_files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_activation_normalization_is_consumed_and_finite():
    torch.manual_seed(42)
    w={name:torch.randn(*shape) for name,shape in [('gate',(48,16)),('up',(48,16)),('down',(16,48))]}
    x=torch.randn(1,32,16)
    logged=activation_fingerprint(x,w)
    raw=activation_fingerprint(x,w,normalization='column-zscore')
    assert logged.shape==raw.shape==(48,7) and logged.isfinite().all() and raw.isfinite().all()
    assert not torch.equal(logged,raw)
    with pytest.raises(ValueError):
        activation_fingerprint(x,w,normalization='ignored-mode')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='candidate OOM boundary uses CUDA telemetry')
def test_phase2_oom_restores_and_continues(tmp_path, monkeypatch):
    from swiglu_fold.phase2 import Phase2
    from swiglu_fold.run import install_weights, model_weights
    run = Phase2.__new__(Phase2)
    run.config = {'shape':[6,8,2,2,4], 'seeds':{'projection':[1], 'clustering':[2], 'permutation':[3]}}
    run.layer=14
    run.completed_candidates=set()
    run.weights={name:torch.randn(*shape) for name,shape in [('gate',(48,16)),('up',(48,16)),('down',(16,48))]}
    run.module=torch.nn.Module()
    for name,w in run.weights.items():
        setattr(run.module,name+'_proj',torch.nn.Linear(w.shape[1],w.shape[0],bias=False))
    install_weights(run.module,run.weights)
    rows=[]
    monkeypatch.setattr(run,'record',lambda kind,**v:rows.append({'status':'ok',**v}))
    monkeypatch.setattr(run,'reconstruct_svd',lambda *a:({k:v*0 for k,v in run.weights.items()},[2,2,2],0))
    calls=0
    def metrics(weights):
        nonlocal calls
        calls+=1
        install_weights(run.module,weights)
        if calls==1:
            raise torch.cuda.OutOfMemoryError('injected after installed candidate')
        return {'mlp_output_mse':1.,'mlp_output_relative_error':1.,'block_output_mse':1.,'block_output_relative_error':.1}
    monkeypatch.setattr(run,'local_metrics',metrics)
    cfg={'name':'test','ablation':'A','ranks':[2,2,2,2,4]}
    run.candidate('svd',cfg,0,bases={})
    run.candidate('svd',cfg,0,bases={})
    assert [r['status'] for r in rows]==['oom','ok']
    assert all(torch.equal(model_weights(run.module)[k],w) for k,w in run.weights.items())
