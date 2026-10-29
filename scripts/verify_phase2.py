"""Independent completed-run audit; no inference and no mutation of raw history."""
import argparse
import hashlib
import json
from math import prod
from pathlib import Path

import numpy as np
from safetensors import safe_open

from swiglu_fold.telemetry import write_json


parser = argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('--full-run')
args = parser.parse_args()
root = Path.cwd()
run = root / args.run
metadata = json.loads((run / 'metadata.json').read_text())
assert not metadata['smoke']
rows = [json.loads(line) for line in (run / 'raw.jsonl').read_text().splitlines()]
config = metadata['config']
assert (run / 'completion.json').exists()
assert metadata['hf_model_revision_sha'] == config['revision']
assert metadata['dataset_revision_sha'] == config['quality']['revision']
assert hashlib.sha256((run / 'source_snapshot.zip').read_bytes()).hexdigest() == metadata['source_snapshot_sha256']
for split in ('train','test'):
    assert hashlib.sha256(np.load(run / f'{split}_tokens.npy').tobytes()).hexdigest() == metadata[f'{split}_token_sha256']
guard = json.loads((root / 'results/phase2_history_guard.json').read_text())
assert hashlib.sha256((root / 'results/experiments.jsonl').read_bytes()[:guard['bytes']]).hexdigest() == guard['sha256']
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name,digest in guard['protected_files'].items())
local = [r for r in rows if r['kind']=='phase2_local' and r['status']=='ok']
expected = len(config['rank_configs']) * (1 + 3*len(config['seeds']['projection']) + 4*(1+3*config['transfer_replicas']))
expected += 5*4 + 5 + len(config['activation']['rank_configs'])*len(config['seeds']['projection'])
assert len(local)==expected, (len(local),expected)
ids = [r['candidate_id'] for r in local if r.get('candidate_id')]
assert len(set(ids))==len(ids)
for r in rows:
    if r['kind']=='phase2_dense_local':
        assert all(r[k]==0 for k in ('mlp_output_mse','mlp_output_relative_error','block_output_mse','block_output_relative_error'))
for r in local:
    assert all(k in r for k in ('mlp_output_mse','mlp_output_relative_error','block_output_mse','block_output_relative_error'))
    if r['strategy']=='zero-mlp':
        assert r['mlp_output_relative_error']==1 and r['effective_parameter_count']==0
        continue
    if r['strategy']=='svd':
        assert sum(r['svd_ranks'])*8192 + r['padding_parameter_count']==r['effective_parameter_count']
        assert r['permutation_bytes']==0 and 0<=r['padding_parameter_count']<8192
    else:
        factors = 3*(prod(r['ranks']) + sum(a*b for a,b in zip(r['shape'],r['ranks'])))
        assert factors==r['factor_parameter_count']
        assert factors*2+r['permutation_bytes']==r['effective_storage_bytes']
        if r['strategy'].startswith('structure-aware'):
            assert r['random_permutation_seed'] is None
            assert r['projection_seed']==config['seeds']['projection'][r['replica']]
            assert r['clustering_seed']==config['seeds']['clustering'][r['replica']]
        if r['strategy'] in ('structure-aware-balanced','structure-aware-activation'):
            cert = r['group_diagnostics']
            assert cert['group_sizes']==[64]*96
            assert 0<=cert['primal_dual_gap']<=cert['optimality_gap_bound']+1e-7
    assert 1.25<=r['effective_compression_ratio']<=4
for r in (r for r in rows if r['kind']=='phase2_quality' and r.get('factor_artifact')):
    path = run / r['factor_artifact']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['factor_artifact_sha256']
    units = 0
    with safe_open(path,framework='pt',device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            units += tensor.numel()*tensor.element_size()//2
    assert units==r['effective_parameter_count']
global_rows = {json.dumps(json.loads(line),sort_keys=True) for line in (root / 'results/experiments.jsonl').read_text().splitlines()}
assert all(json.dumps(row,sort_keys=True) in global_rows for row in rows)
full_count=0
if args.full_run:
    full = root / args.full_run
    assert json.loads((full / 'completion.json').read_text())['full_quality_complete']
    full_rows = [json.loads(line) for line in (full / 'raw.jsonl').read_text().splitlines()]
    full_quality=[r for r in full_rows if r['kind'] in ('phase2_full_ppl','phase2_downstream')]
    assert len(full_quality)==15
    for r in full_quality:
        q=r['quality_metrics']
        assert q['full_evaluation']
        if r['kind']=='phase2_downstream':
            assert q['examples']=={'hellaswag':10042,'piqa':1838,'arc_easy':2376,'winogrande':1267}[r['task']]
            predictions=full / f'{r["strategy"]}_{r["task"]}_predictions.jsonl'
            assert hashlib.sha256(predictions.read_bytes()).hexdigest()==q['prediction_sha256']
            items=[json.loads(line) for line in predictions.read_text().splitlines()]
            assert [p['index'] for p in items]==list(range(q['examples']))
            assert sum(p['gold']==p['prediction'] for p in items)/len(items)==q['accuracy']
            if 'accuracy_character_normalized' in q:
                assert sum(p['gold']==p['prediction_normalized'] for p in items)/len(items)==q['accuracy_character_normalized']
        else:
            assert q['scored_tokens']==q['tokens']-1
    full_count=len(full_quality)
    assert all(json.dumps(row,sort_keys=True) in global_rows for row in full_rows)
verification={'passed':True,'local_candidates':len(local),'unique_candidate_ids':len(ids),
              'zero_layers':5,'dense_block_replays':5,'full_quality_measurements':full_count,
              'history_prefix_unchanged':True,'phase1_files_unchanged':True,
              'exact_storage_and_factor_artifacts':True,'global_raw_matches':True}
write_json(run / 'VERIFICATION.json',verification)
print(json.dumps(verification))
