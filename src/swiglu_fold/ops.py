from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


SHAPES = (
    (96, 64, 8, 8, 32), (192, 32, 8, 8, 32), (128, 48, 8, 16, 16),
    (256, 24, 4, 16, 32), (384, 16, 4, 8, 64),
)


def validate_shape(shape: tuple[int, ...], intermediate: int, hidden: int) -> None:
    if len(shape) != 5 or any(d <= 0 for d in shape):
        raise ValueError("tensor shape must contain five positive dimensions")
    if prod(shape[:2]) != intermediate or prod(shape[2:]) != hidden:
        raise ValueError("tensor axes do not match intermediate and hidden sizes")


def tensorize(weight: torch.Tensor, shape: tuple[int, ...], down: bool = False) -> torch.Tensor:
    logical = weight.T if down else weight
    validate_shape(shape, *logical.shape)
    return logical.reshape(shape)


def untensorize(tensor: torch.Tensor, down: bool = False) -> torch.Tensor:
    shape = tuple(tensor.shape)
    validate_shape(shape, prod(shape[:2]), prod(shape[2:]))
    weight = tensor.reshape(prod(shape[:2]), prod(shape[2:]))
    return weight.T.contiguous() if down else weight


def validate_permutation(perm: torch.Tensor, size: int) -> None:
    if perm.dtype != torch.long or perm.ndim != 1 or perm.numel() != size:
        raise ValueError("permutation must be a long vector of intermediate size")
    if not torch.equal(perm.sort().values, torch.arange(size, device=perm.device)):
        raise ValueError("permutation must contain every intermediate index exactly once")


def permute_weights(weights: dict[str, torch.Tensor], perm: torch.Tensor) -> dict[str, torch.Tensor]:
    validate_permutation(perm, weights["gate"].shape[0])
    return {"gate": weights["gate"][perm], "up": weights["up"][perm],
            "down": weights["down"][:, perm]}


def mlp(x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, weights["gate"])) * F.linear(x, weights["up"]),
                    weights["down"])


def signatures(weights: dict[str, torch.Tensor], seed: int, dimensions: int = 32) -> torch.Tensor:
    if dimensions <= 0:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    components = []
    for weight in (weights["gate"], weights["up"], weights["down"].T):
        rows = F.normalize(weight.detach().cpu().float(), dim=1)
        projection = torch.randn(rows.shape[1], dimensions, generator=generator) / dimensions**0.5
        components.append(rows @ projection)
    return torch.cat(components, dim=1)


def permutation(strategy: str, weights: dict[str, torch.Tensor], seed: int,
                clusters: int = 96) -> torch.Tensor:
    size = weights["gate"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    if strategy == "identity":
        return torch.arange(size)
    if strategy == "random":
        return torch.randperm(size, generator=generator)
    if strategy != "structure-aware":
        raise ValueError(f"unknown strategy: {strategy}")
    if not 1 <= clusters <= size:
        raise ValueError("cluster count must be within intermediate size")
    values = signatures(weights, seed).numpy()
    with threadpool_limits(limits=1):
        labels = MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=3,
                                batch_size=min(1024, size), reassignment_ratio=0).fit_predict(values)
        ordered = []
        for cluster in range(clusters):
            indices = np.flatnonzero(labels == cluster)
            if not len(indices):
                continue
            centered = values[indices].astype(np.float64)
            centered -= centered.mean(axis=0)
            # Covariance is only 96x96; signs and equal-score ties have fixed conventions.
            _, vectors = np.linalg.eigh(centered.T @ centered)
            pc = vectors[:, -1]
            if pc[np.argmax(np.abs(pc))] < 0:
                pc = -pc
            scores = centered @ pc
            ordered.extend(indices[np.lexsort((indices, scores))].tolist())
    result = torch.tensor(ordered, dtype=torch.long)
    validate_permutation(result, size)
    return result


def mode_product(tensor: torch.Tensor, matrix: torch.Tensor, axis: int) -> torch.Tensor:
    if not 0 <= axis < tensor.ndim or matrix.ndim != 2 or matrix.shape[1] != tensor.shape[axis]:
        raise ValueError("invalid mode product dimensions")
    return torch.tensordot(matrix, tensor, dims=([1], [axis])).movedim(0, axis)


def hosvd_bases(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Left singular vectors from each unfolding's small Gram matrix; no TensorLy."""
    if tensor.dtype not in (torch.float32, torch.float64):
        raise ValueError("HOSVD requires FP32 or FP64")
    bases = []
    for axis, size in enumerate(tensor.shape):
        unfolding = tensor.movedim(axis, 0).reshape(size, -1)
        _, vectors = torch.linalg.eigh(unfolding @ unfolding.T)
        basis = vectors.flip(1)
        pivots = basis.abs().argmax(dim=0)
        signs = basis[pivots, torch.arange(size, device=basis.device)].sign()
        bases.append(basis * signs.clamp(min=-1))
    return bases


@dataclass
class Tucker:
    core: torch.Tensor
    factors: list[torch.Tensor]

    @property
    def parameters(self) -> int:
        return self.core.numel() + sum(f.numel() for f in self.factors)

    def reconstruct(self) -> torch.Tensor:
        result = self.core
        for axis, factor in enumerate(self.factors):
            result = mode_product(result, factor, axis)
        return result


def tucker(tensor: torch.Tensor, ranks: tuple[int, ...],
           bases: list[torch.Tensor] | None = None) -> Tucker:
    if len(ranks) != tensor.ndim or any(r < 1 or r > d for r, d in zip(ranks, tensor.shape)):
        raise ValueError("one valid rank is required per tensor axis")
    bases = hosvd_bases(tensor) if bases is None else bases
    if len(bases) != tensor.ndim or any(b.shape != (d, d) for b, d in zip(bases, tensor.shape)):
        raise ValueError("bases must match tensor dimensions")
    factors = [b[:, :rank].contiguous() for b, rank in zip(bases, ranks)]
    core = tensor
    # Largest contraction first lowers temporary memory without changing the decomposition.
    for axis in sorted(range(tensor.ndim), key=lambda a: ranks[a] / tensor.shape[a]):
        core = mode_product(core, factors[axis].T, axis)
    return Tucker(core.contiguous(), factors)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)
    return (difference / denominator).item()
