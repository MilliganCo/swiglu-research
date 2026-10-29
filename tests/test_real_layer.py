import pytest
import torch
from safetensors.torch import load_file

from swiglu_fold.ops import mlp, permute_weights, relative_error
from swiglu_fold.run import configure
from swiglu_fold.phase2_ops import seeded_order


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_layer14_permutation(request, dtype: torch.dtype) -> None:
    weight_path = request.config.getoption("--layer-artifact")
    calibration_path = request.config.getoption("--calibration-artifact")
    if not weight_path or not calibration_path:
        pytest.skip("supply real layer/calibration artifacts to test downloaded Qwen weights")
    assert torch.cuda.is_available(), "real layer checks require CUDA in WSL"
    configure(42)
    w = {name: value.to(device="cuda", dtype=dtype) for name, value in load_file(weight_path).items()}
    assert w["gate"].shape == w["up"].shape == (6144, 2048)
    assert w["down"].shape == (2048, 6144)
    x = load_file(calibration_path)["mlp_inputs"].to(device="cuda", dtype=dtype)
    perm = torch.randperm(6144, generator=torch.Generator().manual_seed(42)).to("cuda")
    expected, actual = mlp(x, w), mlp(x, permute_weights(w, perm))
    torch.testing.assert_close(actual, expected, rtol=0.03 if dtype == torch.bfloat16 else 1e-4,
                               atol=0.03 if dtype == torch.bfloat16 else 2e-5)
    if dtype == torch.bfloat16:
        assert relative_error(actual, expected) <= 0.008


def test_real_balanced_order_and_phase1_v1_compatibility(request):
    from pathlib import Path
    from swiglu_fold.ops import permutation
    weight_path = request.config.getoption('--layer-artifact')
    calibration_path = request.config.getoption('--calibration-artifact')
    if not weight_path or not calibration_path:
        pytest.skip('supply real Phase-1 artifacts')
    configure(42)
    weights = load_file(weight_path)
    seeds = dict(projection_seed=42,clustering_seed=42,random_permutation_seed=999,groups=96)
    old = permutation('structure-aware',weights,42,96)
    assert torch.equal(old,seeded_order('structure-aware',weights,**seeds)[0])
    saved = Path(weight_path).parent / 'permutation-structure-aware-42.safetensors'
    if saved.exists():
        assert torch.equal(old,load_file(str(saved))['permutation'])
    perm,cert = seeded_order('structure-aware-balanced',weights,**seeds)
    assert cert['group_sizes']==[64]*96
    x = load_file(calibration_path)['mlp_inputs'].to('cuda',dtype=torch.float32)
    weights = {k:v.to('cuda',dtype=torch.float32) for k,v in weights.items()}
    dense=mlp(x,weights)
    actual=mlp(x,permute_weights(weights,perm.to('cuda')))
    torch.testing.assert_close(actual,dense,rtol=1e-4,atol=2e-5)
    assert relative_error(actual,dense)<1e-6
