"""Separate exact model symmetry from accumulated BF16 rounding on real logits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from swiglu_fold.ops import permutation, permute_weights, relative_error
from swiglu_fold.run import configure, install_weights, model_weights


@torch.inference_mode()
def check(directory: Path) -> None:
    metadata = json.loads((directory / "metadata.json").read_text())
    config = metadata["config"]
    configure(config["seed"])
    ids = torch.from_numpy(np.load(directory / "test_tokens.npy"))[:, :config["quality"]["logits_tokens"]]
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=metadata["hf_model_revision_sha"],
                                               dtype=torch.float32, attn_implementation="sdpa").eval()
    module = model.model.layers[14].mlp
    weights = model_weights(module)
    reference = model(ids, use_cache=False).logits
    rows = []
    for strategy in ("random", "structure-aware"):
        perm = permutation(strategy, weights, 42, config["clusters"])
        try:
            install_weights(module, permute_weights(weights, perm))
            actual = model(ids, use_cache=False).logits
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=3e-4)
            row = {"dtype": "float32", "strategy": strategy,
                   "relative_error": relative_error(actual, reference),
                   "max_error": (actual - reference).abs().max().item()}
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            install_weights(module, weights)
    del weights
    model = model.to(torch.bfloat16).to("cuda")
    bf16 = model(ids.to("cuda"), use_cache=False).logits.float().cpu()
    row = {"comparison": "dense BF16 vs dense FP32", "relative_error": relative_error(bf16, reference),
           "max_error": (bf16 - reference).abs().max().item()}
    rows.append(row)
    print(json.dumps(row), flush=True)
    (directory / "precision_diagnostic.json").write_text(json.dumps({"model_revision": metadata["hf_model_revision_sha"],
                                                                  "logits_tokens": ids.shape[1], "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    check(parser.parse_args().run_directory)
