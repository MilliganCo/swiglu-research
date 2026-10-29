from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file

from .ops import Tucker, permute_weights, untensorize
from .phase2 import Phase2, configure
from .run import cleanup, install_weights
from .telemetry import write_json


TASK_SOURCE_REVISION = 'd6de81643928d653435c431bae19945d41d32520'


def task_choices(task, doc):
    if task == 'hellaswag':
        def clean(text):
            return re.sub(r'\[.*?\]', '', text.strip().replace(' [title]', '. ')).replace('  ', ' ')
        context = clean(doc['activity_label'] + ': ' + doc['ctx_a'] + ' ' + doc['ctx_b'].capitalize())
        choices = [clean(s) for s in doc['endings']]
        return [(context, ' ' + choice) for choice in choices], int(doc['label']), [len(s) for s in choices]
    if task in ('piqa', 'arc_easy'):
        question = doc['goal'] if task == 'piqa' else doc['question']
        choices = [doc['sol1'], doc['sol2']] if task == 'piqa' else doc['choices']['text']
        gold = doc['label'] if task == 'piqa' else doc['choices']['label'].index(doc['answerKey'])
        return [(f'Question: {question}\nAnswer:', ' ' + s) for s in choices], int(gold), [len(s) for s in choices]
    if task == 'winogrande':
        before, after = doc['sentence'].split('_', 1)
        pairs = [(before + doc[key], ' ' + after.strip()) for key in ('option1', 'option2')]
        return pairs, int(doc['answer']) - 1, [1, 1]
    raise ValueError(task)


def encode_pair(tokenizer, context, continuation):
    trailing = context[len(context.rstrip()):]
    context, continuation = context.rstrip(), trailing + continuation
    prefix = tokenizer.encode(context, add_special_tokens=False)
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    suffix = whole[len(prefix):]
    if not prefix or not suffix or len(whole) > 2048:
        raise ValueError('benchmark pair must have nonempty context/continuation and fit 2048 tokens; no silent truncation')
    return {'tokens': whole, 'continuation_tokens': len(suffix)}


@torch.inference_mode()
def score_batch(model, requests, pad_token, device='cuda'):
    length = max(len(r['tokens']) - 1 for r in requests)
    keep = max(r['continuation_tokens'] for r in requests)
    ids = torch.full((len(requests), length), pad_token, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, request in enumerate(requests):
        seq = torch.tensor(request['tokens'][:-1], device=device)
        ids[i, -len(seq):] = seq
        mask[i, -len(seq):] = 1
    positions = (mask.cumsum(1) - 1).clamp_min(0)
    logits = model(ids, attention_mask=mask, position_ids=positions, use_cache=False, logits_to_keep=keep).logits
    results = []
    for i, request in enumerate(requests):
        count = request['continuation_tokens']
        targets = torch.tensor(request['tokens'][-count:], device=device)
        # Score each requested continuation token exactly once; no prefix tokens.
        loss = F.cross_entropy(logits[i, -count:].float(), targets, reduction='sum')
        results.append(-loss.item())
    return results


def artifact_weights(path, shape):
    data = load_file(str(path), device='cuda')
    weights = {}
    for name in ('gate', 'up', 'down'):
        if f'{name}.left' in data:
            logical = data[f'{name}.left'].float() @ data[f'{name}.right'].float()
            weights[name] = (logical.T.contiguous() if name == 'down' else logical).to(torch.bfloat16)
        else:
            tensor = Tucker(data[f'{name}.core'].float(), [data[f'{name}.factor{i}'].float() for i in range(len(shape))]).reconstruct()
            weights[name] = untensorize(tensor, name == 'down').to(torch.bfloat16)
    if 'permutation' in data:
        weights = permute_weights(weights, data['permutation'].argsort())
    return weights


def prepare_tasks(run, spec, limit=None):
    prepared = {}
    for task, cfg in spec.items():
        dataset = load_dataset(cfg['path'], cfg['subset'], split=cfg['split'], revision=cfg['revision'])
        if len(dataset) != cfg['rows']:
            raise AssertionError('pinned benchmark cardinality changed')
        rows = []
        for i, doc in enumerate(dataset):
            if limit and i >= limit:
                break
            pairs, gold, lengths = task_choices(task, doc)
            rows.append({'index':i, 'gold':gold, 'normalization_lengths': lengths,
                         'requests':[encode_pair(run.tokenizer, *pair) for pair in pairs]})
        path = run.directory / f'{task}_requests.json.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as handle:
            json.dump(rows, handle)
        prepared[task] = rows
        run.record('phase2_downstream_dataset', task=task, dataset=cfg, examples=len(rows), full_evaluation=limit is None,
                   request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), task_definition_revision=TASK_SOURCE_REVISION)
    return prepared


@torch.inference_mode()
def check_real_scoring(run, prepared):
    requests = [request for docs in prepared.values() for request in docs[0]['requests'][:2]]
    scores = score_batch(run.model,requests,run.tokenizer.pad_token_id or run.tokenizer.eos_token_id)
    differences=[]
    for request,score in zip(requests,scores):
        ids=torch.tensor(request['tokens'],device='cuda')
        logits=run.model(ids[:-1][None],use_cache=False).logits[0].float()
        count=request['continuation_tokens']
        reference=-F.cross_entropy(logits[-count:],ids[-count:],reduction='sum').item()
        differences.append(abs(score-reference)/count)
    diagnostic_path=run.root / 'results/phase2_scoring_precision.json'
    diagnostic=json.loads(diagnostic_path.read_text())
    (run.directory / 'scoring_precision_control.json').write_bytes(diagnostic_path.read_bytes())
    if diagnostic['fp32']['maximum_abs_delta_per_token']>=5e-5:
        raise AssertionError('strict real-model FP32 scorer control failed')
    # BF16 GEMM/SDPA changes with padding and LM-head shape; FP32 control checks indexing.
    if max(differences)>.1:
        raise AssertionError('batched suffix scorer differs from full-logit unbatched reference')
    run.record('phase2_scoring_check',requests=len(requests),mean_logprob_per_token_max_difference=max(differences),
               tolerance=.1,fp32_diagnostic_max_delta=diagnostic['fp32']['maximum_abs_delta_per_token'],
               precision_diagnostic_sha256=hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
               reference='unbatched full logits; BF16 batch-order scale verified separately in real-model FP32')


@torch.inference_mode()
def downstream(run, strategy, prepared, spec, batch_size):
    for task, docs in prepared.items():
        requests = [request for doc in docs for request in doc['requests']]
        results = [None] * len(requests)
        order = sorted(range(len(requests)), key=lambda i: (-len(requests[i]['tokens']), i))
        start = 0
        current_batch = batch_size
        while start < len(order):
            indices = order[start:start+current_batch]
            try:
                scores = score_batch(run.model, [requests[i] for i in indices], run.tokenizer.pad_token_id or run.tokenizer.eos_token_id)
                for i, score in zip(indices, scores):
                    results[i] = score
                start += len(indices)
                if start % 1024 < current_batch or start == len(order):
                    print(f'DOWNSTREAM {strategy} {task} {start}/{len(order)} choices batch={current_batch}', flush=True)
            except torch.cuda.OutOfMemoryError as exc:
                run.record('phase2_downstream_oom', strategy=strategy, task=task, status='oom', error=str(exc), batch_size=current_batch)
                cleanup()
                if current_batch == 1:
                    raise
                current_batch = max(1, current_batch // 2)
        predictions = []
        offset = 0
        for doc in docs:
            count = len(doc['requests'])
            scores = np.array(results[offset:offset+count])
            pred = int(scores.argmax())
            pred_norm = int((scores / np.asarray(doc['normalization_lengths'])).argmax())
            predictions.append({'index':doc['index'], 'gold':doc['gold'], 'prediction':pred,
                                'prediction_normalized':pred_norm, 'loglikelihoods':scores.tolist()})
            offset += count
        path = run.directory / f'{strategy}_{task}_predictions.jsonl'
        # Predictions are a new immutable artifact; research history uses append_jsonl.
        with path.open('x', encoding='utf-8') as handle:
            for row in predictions:
                handle.write(json.dumps(row) + '\n')
        metrics = {'accuracy':float(np.mean([p['gold']==p['prediction'] for p in predictions])),
                   'examples':len(predictions), 'choices_scored':len(requests), 'full_evaluation':not run.smoke,
                   'fewshot':0, 'chat_template':False, 'dataset_revision_sha':spec[task]['revision'],
                   'batch_size_initial':batch_size, 'batch_size_final':current_batch,
                   'task_definition_revision':TASK_SOURCE_REVISION,
                   'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if task != 'winogrande':
            metrics['accuracy_character_normalized'] = float(np.mean([p['gold']==p['prediction_normalized'] for p in predictions]))
        run.record('phase2_downstream', strategy=strategy, task=task, quality_metrics=metrics,
                   quality_scope={'dataset':spec[task]['path'],'subset':spec[task]['subset'],
                                  'split':spec[task]['split'],'examples':len(predictions),'full_evaluation':not run.smoke})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-run', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch_size must be positive')
    root = Path.cwd()
    local = root / args.local_run
    metadata = json.loads((local / 'metadata.json').read_text())
    selection = json.loads((local / 'pareto_selection.json').read_text())
    quality = [json.loads(line) for line in (local / 'raw.jsonl').read_text().splitlines()]
    frontier = {tuple(key) for key in selection['frontier']}
    eligible = [r for r in quality if r['kind']=='phase2_quality' and r['status']=='ok' and r['layer']==14
                and (r['layer'],r.get('config_name'),r['strategy']) in frontier]
    # Predeclared target is near 2x. Replicated frontier groups use the median MLP seed.
    group_keys = {(r['config_name'],r['strategy']) for r in eligible}
    dense_nll = next(r['quality_metrics']['nll'] for r in quality if r['kind']=='phase2_quality' and r['strategy']=='dense')
    zero_nll = next(r['quality_metrics']['nll'] for r in quality if r['kind']=='phase2_quality' and r['layer']==14 and r['strategy']=='zero-mlp')
    mean_nll = {key:float(np.mean([r['quality_metrics']['nll'] for r in eligible if (r['config_name'],r['strategy'])==key])) for key in group_keys}
    # Zero has zero storage; any larger model with worse subset NLL is dominated.
    group_keys = {key for key in group_keys if mean_nll[key] < zero_nll}
    if not group_keys:
        raise RuntimeError('no layer14 Pareto finalist beats zero-MLP on subset NLL')
    def selection_key(key):
        group = [r for r in eligible if (r['config_name'],r['strategy'])==key]
        local_metrics = [s['mlp_output_relative_error'] for s in quality if s['kind']=='phase2_local'
                         and s['layer']==14 and (s['config_name'],s['strategy'])==key]
        return abs(group[0]['effective_compression_ratio']-2), float(np.mean(local_metrics)), key
    selected_key = min(group_keys, key=selection_key)
    group = [r for r in eligible if (r['config_name'],r['strategy'])==selected_key]
    selected = sorted(group,key=lambda r:next(s['mlp_output_relative_error'] for s in quality if s.get('candidate_id')==r['candidate_id'] and s['kind']=='phase2_local'))[len(group)//2]
    config = copy.deepcopy(metadata['config'])
    configure(config['seed'])
    run = Phase2(config, root, args.smoke)
    run.metadata.update({'quality_protocol':'phase2-finalists-zero-shot-1', 'local_source_run':metadata['run_id'],
                         'selected_candidate_id':selected['candidate_id'], 'selected_candidate_source_sha256':selected['source_sha256'],
                         'finalist_policy':'local Pareto group beating zero subset NLL, closest to 2x, median local MLP replica; dense and layer14 zero controls'})
    (run.directory / 'resolved.yaml').write_text(__import__('yaml').safe_dump(config,sort_keys=False))
    print(f'FINALISTS RUN {run.run_id}; selected {selected["candidate_id"]}',flush=True)
    run.load()
    run.capture_all()
    run.select_layer(14)
    source_artifact = local / selected['factor_artifact']
    if hashlib.sha256(source_artifact.read_bytes()).hexdigest() != selected['factor_artifact_sha256']:
        raise AssertionError('finalist factor artifact SHA changed')
    candidate = artifact_weights(source_artifact, config['shape'])
    original_local = next(r for r in quality if r['kind']=='phase2_local' and r.get('candidate_id')==selected['candidate_id'])
    actual = run.local_metrics(candidate)
    if any(abs(actual[k]-original_local[k]) > 1e-6 for k in actual):
        raise AssertionError('saved finalist differs from local measurement')
    run.record('phase2_finalist_replay', candidate_id=selected['candidate_id'], layer=14, strategy=selected['strategy'], **actual)
    spec = json.loads((root / 'configs/downstream_pinned.json').read_text())
    run.metadata['downstream_datasets'] = spec
    prepared = prepare_tasks(run, spec, 4 if args.smoke else None)
    check_real_scoring(run,prepared)
    dataset = load_dataset(config['quality']['dataset'],config['quality']['subset'],revision=config['quality']['revision'],split='test')
    ids = run.tokenizer('\n\n'.join(dataset['text']),return_tensors='pt',add_special_tokens=False,verbose=False)['input_ids']
    if args.smoke:
        ids = ids[:,:512]
    np.save(run.directory / 'full_test_tokens.npy',ids.numpy())
    run.metadata['full_test_token_sha256'] = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    run.metadata['preflight_quality_scope']=run.metadata['quality_scope']
    run.metadata['quality_scope']={'dataset':config['quality']['dataset'],'subset':config['quality']['subset'],
                                   'split':'test','tokens':ids.shape[1],'full_evaluation':not args.smoke}
    write_json(run.directory / 'metadata.json',run.metadata)
    run.test = ids.to('cuda')
    variants = {'dense':{k:v.to('cuda') for k,v in run.weights.items()},
                'zero-mlp':{k:v.to('cuda') for k,v in run.weights.items()}, selected['strategy']:candidate}
    variants['zero-mlp']['down'] = torch.zeros_like(variants['zero-mlp']['down'])
    for strategy,weights in variants.items():
        try:
            install_weights(run.module,weights)
            metrics = run.perplexity()
            metrics.update(full_evaluation=not args.smoke, tokens=ids.shape[1], token_sha256=run.metadata['full_test_token_sha256'])
            run.record('phase2_full_ppl',layer=14,strategy=strategy,candidate_id=selected['candidate_id'] if strategy==selected['strategy'] else None,quality_metrics=metrics)
            downstream(run,strategy,prepared,spec,args.batch_size)
        finally:
            install_weights(run.module,run.weights)
            cleanup()
    write_json(run.directory / 'completion.json', {'full_quality_complete':not args.smoke,'smoke':args.smoke,'source_run':metadata['run_id']})
    print(f'FINALISTS COMPLETE {run.run_id}',flush=True)


if __name__=='__main__':
    main()
