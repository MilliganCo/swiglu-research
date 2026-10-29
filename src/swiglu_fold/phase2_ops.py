from __future__ import annotations

from collections import deque
import heapq
from math import prod

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from .ops import permutation, signatures, validate_permutation


def balanced_assignment(costs: np.ndarray, capacity: int, epsilon: float = 1e-4,
                        max_bids: int = 2_000_000) -> tuple[np.ndarray, dict]:
    """Deterministic capacitated auction; certify the primal/dual epsilon gap."""
    costs = np.asarray(costs, dtype=np.float64)
    if (costs.ndim != 2 or costs.shape[1] < 2 or capacity < 1
            or costs.shape[0] != capacity * costs.shape[1]
            or not np.isfinite(costs).all() or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("finite N x K costs, K >= 2, N = K * capacity, epsilon > 0 required")
    n, groups = costs.shape
    prices = np.zeros(groups)
    heaps: list[list[tuple[float, int]]] = [[] for _ in range(groups)]
    pending = deque(range(n))
    labels = np.full(n, -1, dtype=np.int64)
    bids = 0
    while pending:
        i = pending.popleft()
        adjusted = costs[i] + prices
        group = int(np.argmin(adjusted))
        second = float(np.partition(adjusted, 1)[1])
        bid = prices[group] + second - adjusted[group] + epsilon
        heapq.heappush(heaps[group], (float(bid), i))
        labels[i] = group
        if len(heaps[group]) > capacity:
            _, evicted = heapq.heappop(heaps[group])
            labels[evicted] = -1
            pending.append(evicted)
        if len(heaps[group]) == capacity:
            prices[group] = heaps[group][0][0]
        bids += 1
        if bids > max_bids:
            raise RuntimeError("auction exceeded max_bids; no partial assignment accepted")
    counts = np.bincount(labels, minlength=groups)
    if not np.all(counts == capacity):
        raise AssertionError("auction did not fill every exact capacity")
    primal = float(costs[np.arange(n), labels].sum())
    dual = float((costs + prices).min(axis=1).sum() - capacity * prices.sum())
    gap = primal - dual
    if not -1e-8 <= gap <= n * epsilon + 1e-7:
        raise AssertionError("auction failed its epsilon-optimality certificate")
    return labels, {"assignment": "capacitated-auction-v1", "capacity": capacity,
                    "epsilon": epsilon, "bids": bids, "primal_cost": primal,
                    "dual_lower_bound": dual, "optimality_gap_bound": n * epsilon,
                    "primal_dual_gap": max(0., gap), "group_sizes": counts.tolist()}


def pc_order(values: np.ndarray, labels: np.ndarray, groups: int) -> torch.Tensor:
    ordered = []
    with threadpool_limits(limits=1):
        for group in range(groups):
            indices = np.flatnonzero(labels == group)
            if not len(indices):
                continue
            centered = values[indices].astype(np.float64)
            centered -= centered.mean(axis=0)
            _, vectors = np.linalg.eigh(centered.T @ centered)
            pc = vectors[:, -1]
            if pc[np.argmax(np.abs(pc))] < 0:
                pc = -pc
            ordered.extend(indices[np.lexsort((indices, centered @ pc))].tolist())
    result = torch.tensor(ordered, dtype=torch.long)
    validate_permutation(result, len(values))
    return result


def seeded_order(strategy: str, weights: dict[str, torch.Tensor], *, projection_seed: int,
                 clustering_seed: int, random_permutation_seed: int, groups: int,
                 epsilon: float = 1e-4, activation: torch.Tensor | None = None,
                 activation_weight: float = .25) -> tuple[torch.Tensor, dict]:
    size = weights["gate"].shape[0]
    if strategy in ("identity", "random"):
        return permutation(strategy, weights, random_permutation_seed), {}
    if strategy not in ("structure-aware", "structure-aware-balanced", "structure-aware-activation"):
        raise ValueError(f"unknown strategy {strategy}")
    if not 2 <= groups <= size or size % groups:
        raise ValueError("groups must divide the intermediate dimension")
    values = signatures(weights, projection_seed).numpy()
    if strategy == "structure-aware-activation":
        if activation is None or activation.shape[0] != size or not 0 < activation_weight <= 1:
            raise ValueError("activation fingerprints and positive activation_weight required")
        values = np.concatenate((values, activation_weight * activation.cpu().numpy()), axis=1)
    with threadpool_limits(limits=1):
        fit = MiniBatchKMeans(n_clusters=groups, random_state=clustering_seed, n_init=3,
                              batch_size=min(1024, size), reassignment_ratio=0).fit(values)
    diagnostics = {"unbalanced_group_sizes": np.bincount(fit.labels_, minlength=groups).tolist()}
    if strategy == "structure-aware":
        labels = fit.labels_
    else:
        centers = fit.cluster_centers_
        costs = ((values.astype(np.float64) ** 2).sum(1)[:, None]
                 + (centers.astype(np.float64) ** 2).sum(1)[None, :]
                 - 2 * values.astype(np.float64) @ centers.astype(np.float64).T)
        labels, assignment = balanced_assignment(costs, size // groups, epsilon)
        diagnostics.update(assignment)
    return pc_order(values, labels, groups), diagnostics


@torch.inference_mode()
def activation_fingerprint(x: torch.Tensor, weights: dict[str, torch.Tensor],
                           chunk_tokens: int = 128, floor: float = 1e-12,
                           normalization: str = 'log-floor-then-column-zscore') -> torch.Tensor:
    """Seven compact usage features; no retained token x neuron activation array."""
    w = {k: v.to(device=x.device, dtype=torch.float32) for k, v in weights.items()}
    size = w["gate"].shape[0]
    sums = torch.zeros(5, size, device=x.device, dtype=torch.float64)
    product_sum = torch.zeros(size, device=x.device, dtype=torch.float64)
    count = 0
    for chunk in x.flatten(0, -2).split(chunk_tokens):
        gate, up = F.linear(chunk.float(), w["gate"]), F.linear(chunk.float(), w["up"])
        silu = F.silu(gate)
        product = silu * up
        for i, value in enumerate((gate, silu, up, product, product * w["down"].norm(dim=0))):
            sums[i] += value.double().square().sum(0)
        product_sum += product.double().sum(0)
        count += len(chunk)
    rms = (sums / count).sqrt()
    variance = (sums[3] / count - (product_sum / count).square()).clamp_min(0)
    features = torch.cat((rms, variance[None], (product_sum / count).abs()[None])).T
    if normalization == 'log-floor-then-column-zscore':
        features = features.clamp_min(floor).log()
    elif normalization != 'column-zscore':
        raise ValueError('unknown activation normalization')
    features = features.float()
    features = (features - features.mean(0)) / features.std(0, correction=0).clamp_min(floor)
    return features.cpu().contiguous()


def tucker_parameters(shape: tuple[int, ...], ranks: tuple[int, ...]) -> int:
    if len(shape) != len(ranks) or any(not 1 <= r <= d for r, d in zip(ranks, shape)):
        raise ValueError("invalid ranks")
    return prod(ranks) + sum(d * r for d, r in zip(shape, ranks))


def svd_budget(total_bf16_units: int, intermediate: int, hidden: int) -> tuple[list[int], int]:
    """Maximize ordinary SVD ranks within the exact storage allocation; no permutation."""
    rank_sum, padding = divmod(total_bf16_units, intermediate + hidden)
    ranks = [rank_sum // 3 + (i < rank_sum % 3) for i in range(3)]
    if min(ranks) < 1 or max(ranks) > min(intermediate, hidden):
        raise ValueError("SVD budget gives invalid ranks")
    return ranks, padding


@torch.inference_mode()
def svd_basis(weight: torch.Tensor) -> torch.Tensor:
    """Ordinary matrix truncated SVD through the small right Gram matrix (FP32)."""
    _, vectors = torch.linalg.eigh(weight.T @ weight)
    return vectors.flip(1).contiguous()


def svd_factors(weight: torch.Tensor, basis: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 1 <= rank <= min(weight.shape) or basis.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("invalid SVD rank or basis")
    right = basis[:, :rank].T.contiguous()
    return weight @ right.T, right


def functional_metrics(actual_mlp: torch.Tensor, dense_mlp: torch.Tensor,
                       actual_block: torch.Tensor, dense_block: torch.Tensor) -> dict[str, float]:
    from .ops import relative_error
    return {"mlp_output_mse": (actual_mlp.float() - dense_mlp.float()).square().mean().item(),
            "mlp_output_relative_error": relative_error(actual_mlp, dense_mlp),
            "block_output_mse": (actual_block.float() - dense_block.float()).square().mean().item(),
            "block_output_relative_error": relative_error(actual_block, dense_block)}
