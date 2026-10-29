from math import prod

import pytest
import torch

from swiglu_fold.ops import (SHAPES, hosvd_bases, mlp, permutation, permute_weights,
                             signatures, tensorize, tucker, untensorize)
from swiglu_fold.telemetry import append_jsonl
from swiglu_fold.run import Experiment, configure, install_weights, model_weights


def weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(9)
    return {"gate": torch.randn(48, 16, generator=generator) / 4,
            "up": torch.randn(48, 16, generator=generator) / 4,
            "down": torch.randn(16, 48, generator=generator) / 7}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strategy", ["identity", "random", "structure-aware"])
def test_symmetry(dtype: torch.dtype, strategy: str) -> None:
    w = weights()
    perm = permutation(strategy, w, 42, clusters=4)
    w = {name: value.to(dtype) for name, value in w.items()}
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(10)).to(dtype)
    actual, expected = mlp(x, permute_weights(w, perm)), mlp(x, w)
    torch.testing.assert_close(actual, expected, rtol=0.03 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.01 if dtype == torch.bfloat16 else 1e-6)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("down", [False, True])
def test_real_tensorization_exact(shape: tuple[int, ...], down: bool) -> None:
    weight = torch.arange(prod(shape), dtype=torch.int32).reshape(6144, 2048)
    if down:
        weight = weight.T.contiguous()
    assert torch.equal(untensorize(tensorize(weight, shape, down), down), weight)


def test_projection_deterministic_and_component_scale_invariant() -> None:
    w = weights()
    signature = signatures(w, 42)
    assert signature.shape == (48, 96)
    assert torch.equal(signature, signatures(w, 42))
    torch.testing.assert_close(signature, signatures({k: v * 3 for k, v in w.items()}, 42))
    assert not torch.equal(signature, signatures(w, 43))


def test_structure_order_deterministic() -> None:
    a = permutation("structure-aware", weights(), 42, 4)
    assert torch.equal(a, permutation("structure-aware", weights(), 42, 4))
    assert torch.equal(a.sort().values, torch.arange(48))


def test_tucker_exact_and_nested_ranks() -> None:
    tensor = torch.randn(6, 4, 2, 2, 4, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    bases = hosvd_bases(tensor)
    full = tucker(tensor, tuple(tensor.shape), bases)
    torch.testing.assert_close(full.reconstruct(), tensor, rtol=1e-12, atol=1e-12)
    low, high = tucker(tensor, (2, 2, 1, 1, 2), bases), tucker(tensor, (4, 3, 2, 2, 3), bases)
    assert low.parameters == 2 * 2 * 1 * 1 * 2 + 6 * 2 + 4 * 2 + 2 + 2 + 4 * 2
    assert torch.linalg.vector_norm(high.reconstruct() - tensor) < torch.linalg.vector_norm(low.reconstruct() - tensor)


def test_truncated_hosvd_exact_for_separable_tensor() -> None:
    factors = [torch.arange(1, n + 1, dtype=torch.float64) for n in (4, 3, 2, 2, 3)]
    tensor = torch.einsum("a,b,c,d,e->abcde", *factors)
    torch.testing.assert_close(tucker(tensor, (1, 1, 1, 1, 1)).reconstruct(), tensor,
                               rtol=1e-12, atol=1e-12)


def test_invalid_shapes_ranks_and_permutations() -> None:
    with pytest.raises(ValueError):
        tensorize(weights()["gate"], (2, 2, 2, 2, 2))
    with pytest.raises(ValueError):
        tucker(torch.ones(2, 2), (3, 1))
    with pytest.raises(ValueError):
        permute_weights(weights(), torch.zeros(48, dtype=torch.long))


def test_append_only(tmp_path) -> None:
    path = tmp_path / "experiments.jsonl"
    append_jsonl(path, {"run": 1})
    first = path.read_bytes()
    append_jsonl(path, {"run": 2})
    assert path.read_bytes().startswith(first)
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA OOM recovery check requires GPU")
def test_oom_records_configuration_continues_and_restores_weights(tmp_path, monkeypatch) -> None:
    configure(42)
    run = Experiment.__new__(Experiment)
    run.config = {"shape": [6, 8, 2, 2, 4], "strategies": ["identity"], "permutation_seeds": [42],
                  "clusters": 4, "ranks": [[2, 2, 1, 1, 2], [4, 4, 2, 2, 3]]}
    run.weights = {name: value.to(torch.bfloat16) for name, value in weights().items()}
    run.directory = tmp_path
    (tmp_path / "factors").mkdir()
    run.module = torch.nn.Module()
    for name, weight in run.weights.items():
        setattr(run.module, name + "_proj", torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False))
    run.module.to(device="cuda", dtype=torch.bfloat16)
    install_weights(run.module, run.weights)
    run.x = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)
    run.original_output = mlp(run.x, {k: v.to("cuda") for k, v in run.weights.items()}).float()
    rows = []
    monkeypatch.setattr(run, "record", lambda kind, **values: rows.append({"kind": kind, "status": "ok", **values}))
    monkeypatch.setattr(run, "symmetry", lambda *args: None)
    monkeypatch.setattr(run, "logits_metrics", lambda: {})
    calls = 0

    def quality():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise torch.cuda.OutOfMemoryError("injected after candidate weights were installed")
        return {"wikitext2_raw_perplexity": 1.0}

    monkeypatch.setattr(run, "perplexity", quality)
    run.sweep({"wikitext2_raw_perplexity": 1.0})
    assert [row["status"] for row in rows] == ["oom", "ok"]
    assert calls == 2 and len(list((tmp_path / "factors").glob("*.safetensors"))) == 1
    restored = model_weights(run.module)
    assert all(torch.equal(restored[name], weight) for name, weight in run.weights.items())
