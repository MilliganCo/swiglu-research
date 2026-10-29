from __future__ import annotations

import heapq
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .phase2_ops import svd_basis, svd_factors

NAMES = ('gate', 'up', 'down')


def parameter_count(common_rank: int, private_ranks: list[int], intermediate: int,
                    hidden: int) -> int:
    if len(private_ranks) != 3 or not 0 <= common_rank <= intermediate or any(
            not 0 <= r <= min(intermediate, hidden) for r in private_ranks):
        raise ValueError('invalid common/private ranks')
    return (intermediate + 3 * hidden) * common_rank + (intermediate + hidden) * sum(private_ranks)


def allocate(spectra: list[torch.Tensor], units: int, mode: str) -> list[int]:
    if units < 0 or units > sum(len(s) for s in spectra) or len(spectra) != 3:
        raise ValueError('invalid private rank budget')
    if mode not in ('equal', 'energy-greedy'):
        raise ValueError('unknown allocation mode')
    ranks = [0, 0, 0]
    heap = [(-float(s[0]), i) for i, s in enumerate(spectra) if len(s)]
    heapq.heapify(heap)
    for j in range(units):
        if mode == 'equal':
            i = min((i for i in range(3) if ranks[i] < len(spectra[i])), key=lambda i: (ranks[i], i))
        else:
            _, i = heapq.heappop(heap)
        ranks[i] += 1
        if mode == 'energy-greedy' and ranks[i] < len(spectra[i]):
            heapq.heappush(heap, (-float(spectra[i][ranks[i]]), i))
    return ranks


@torch.inference_mode()
def stacked_basis(weights: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    # MM.T = sum Wi Wi.T: identical stacked left objective without a second 6144² allocation.
    gram = sum(w @ w.T for w in weights.values())
    energy, basis = torch.linalg.eigh(gram)
    basis = basis.flip(1).contiguous()
    pivots = basis.abs().argmax(0)
    basis *= basis[pivots, torch.arange(basis.shape[1], device=basis.device)].sign()
    return basis, energy.flip(0).clamp_min(0)


@torch.inference_mode()
def residual_bases(weights: dict[str, torch.Tensor], common: torch.Tensor):
    residuals, bases, spectra = {}, {}, []
    for name in NAMES:
        residual = weights[name] - common @ (common.T @ weights[name])
        # Keep the unchanged Phase-2 factor convention, including rc=0.
        basis = svd_basis(residual)
        energy = (residual @ basis).square().sum(0).cpu()
        residuals[name], bases[name] = residual, basis
        spectra.append(energy)
    return residuals, bases, spectra


@torch.inference_mode()
def analytic_factors(weights: dict[str, torch.Tensor], common: torch.Tensor,
                     residuals: dict[str, torch.Tensor], bases: dict[str, torch.Tensor],
                     private_ranks: list[int], budget: int) -> dict[str, torch.Tensor]:
    intermediate, hidden = weights['gate'].shape
    count = parameter_count(common.shape[1], private_ranks, intermediate, hidden)
    if count > budget:
        raise ValueError('factor storage exceeds budget')
    factors = {'common': common, 'alignment_padding': torch.zeros(budget - count, device=common.device)}
    for name, rank in zip(NAMES, private_ranks):
        factors[f'{name}.common_right'] = common.T @ weights[name]
        if rank:
            left, right = svd_factors(residuals[name], bases[name], rank)
        else:
            left, right = common.new_empty(intermediate, 0), common.new_empty(0, hidden)
        factors[f'{name}.private_left'], factors[f'{name}.private_right'] = left, right
    return factors


def save_factors(path: Path, factors: dict[str, torch.Tensor]) -> None:
    save_file({k: v.detach().to(device='cpu', dtype=torch.bfloat16).contiguous() for k, v in factors.items()}, str(path))


def reconstruct(factors: dict[str, torch.Tensor], bf16_weights: bool = True) -> dict[str, torch.Tensor]:
    weights = {}
    for name in NAMES:
        logical = (factors['common'].float() @ factors[f'{name}.common_right'].float()
                   + factors[f'{name}.private_left'].float() @ factors[f'{name}.private_right'].float())
        if bf16_weights:
            logical = logical.to(torch.bfloat16)
        weights[name] = logical.T.contiguous() if name == 'down' else logical
    return weights


def replay(path: Path, device: str = 'cuda') -> dict[str, torch.Tensor]:
    return reconstruct(load_file(str(path), device=device))


def split_indices(tokens: int, selection_tokens: int, train_fraction: float, seed: int):
    if not 1 < selection_tokens < tokens or not 0 < train_fraction < 1:
        raise ValueError('selection and independent confirmation tokens required')
    order = torch.randperm(selection_tokens, generator=torch.Generator().manual_seed(seed))
    cut = int(train_fraction * selection_tokens)
    if not 0 < cut < selection_tokens:
        raise ValueError('empty training or validation split')
    return {'train': order[:cut].sort().values, 'validation': order[cut:].sort().values,
            'confirmation': torch.arange(selection_tokens, tokens)}


def freeze_base(model: torch.nn.Module) -> None:
    model.requires_grad_(False)
    model.eval()


def checkpoint_improved(validation: float, best: float, minimum: float) -> bool:
    return validation < best - minimum


def subspace_metrics(cross: torch.Tensor) -> dict:
    if cross.ndim != 2 or cross.shape[0] != cross.shape[1] or not cross.shape[0]:
        raise ValueError('square nonempty top-k cross-basis matrix required')
    # FP64 direct SVD avoids the observed FP32 GPU Gram/eigvalsh trace discrepancy.
    cross = cross.detach().cpu().double()
    correlations = torch.linalg.svdvals(cross).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(correlations))
    return {'normalized_projection_overlap': float(cross.square().sum()/len(cross)),
            'mean_squared_canonical_correlation': float(correlations.square().mean()),
            'canonical_correlation_mean': float(correlations.mean()),
            'canonical_correlation_min': float(correlations.min()),
            'canonical_correlation_max': float(correlations.max()),
            'canonical_correlations': correlations.tolist(), 'principal_angles_degrees': angles.tolist(),
            'principal_angle_mean_degrees': float(angles.mean()),
            'canonical_correlation_precision': 'CPU FP64 direct SVD of FP32 orthonormal-basis cross product'}


def factor_mlp(x: torch.Tensor, factors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Training algebra only; quality replay always installs reconstructed BF16 weights."""
    common = factors['common']
    def project(name):
        return (F.linear(F.linear(x, factors[f'{name}.common_right']), common)
                + F.linear(F.linear(x, factors[f'{name}.private_right']), factors[f'{name}.private_left']))
    product = F.silu(project('gate')) * project('up')
    return (F.linear(F.linear(product, common.T), factors['down.common_right'].T)
            + F.linear(F.linear(product, factors['down.private_left'].T), factors['down.private_right'].T))
