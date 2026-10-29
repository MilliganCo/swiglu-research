"""Real-model indexing control in FP32, then BF16 batch/order numerical scale."""
import gc
import argparse
import gzip
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from swiglu_fold.phase2 import configure
from swiglu_fold.phase2_finalists import score_batch
from swiglu_fold.telemetry import write_json


configure(42)
parser=argparse.ArgumentParser()
parser.add_argument('--requests-run',default='results/runs/20261006T075109Z-87b73fba')
args=parser.parse_args()
directory=Path(args.requests_run)
requests=[]
for task in ('hellaswag','piqa','arc_easy','winogrande'):
    docs=json.load(gzip.open(directory/f'{task}_requests.json.gz','rt'))
    requests+=docs[0]['requests'][:2]
output={'reference_run':directory.name,'model_revision':'ea980cb0a6c2ae4b936e82123acc929f1cec04c1',
        'purpose':'separate scorer correctness from BF16 padding/LM-head matrix-shape variation'}
for label,dtype,device in [('fp32',torch.float32,'cpu'),('bf16',torch.bfloat16,'cuda')]:
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B-Base',revision=output['model_revision'],
        dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().to(device=device)
    if dtype==torch.float32:
        model=model.float()
    differences=[]
    with torch.inference_mode():
        scores=score_batch(model,requests,151643,device)
        for request,score in zip(requests,scores):
            ids=torch.tensor(request['tokens'],device=device)
            count=request['continuation_tokens']
            logits=model(ids[:-1][None],use_cache=False).logits[0]
            reference=-F.cross_entropy(logits[-count:].float(),ids[-count:],reduction='sum').item()
            differences.append({'tokens':len(ids),'continuation_tokens':count,'batch_loglikelihood':score,
                                'unbatched_full_loglikelihood':reference,'abs_delta_per_token':abs(score-reference)/count})
    maximum=max(r['abs_delta_per_token'] for r in differences)
    if label=='fp32':
        assert maximum<5e-5,maximum
    output[label]={'rows':differences,'maximum_abs_delta_per_token':maximum}
    print(label,maximum,flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
write_json(Path('results/phase2_scoring_precision.json'),output)
