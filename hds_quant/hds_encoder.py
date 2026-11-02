"""Hyperdimensional Sensitivity (HDS) encoder (SPEC §5)."""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, Union

import torch

COORD_BINDINGS = ("raw_index_strict", "salience_bucket")


class HDSEncoder:
    """Encodes a residual matrix ``R_pb`` (``[T, d_out]``) into a bipolar-bundled hypervector ``[D]``.

    Codebooks (all in {-1, +1}, stored as float32 on ``device``):
      * ``L``        ``[num_levels, hd_dim]``  correlated ordinal level codebook.
      * ``C_raw``    ``[max_coords, hd_dim]``  one random hypervector per raw output index.
      * ``C_bucket`` ``[num_buckets, hd_dim]`` one random hypervector per salience bucket.

    Codebooks are generated with a CPU generator so they are identical across devices.
    """

    def __init__(
        self,
        hd_dim: int = 4096,
        num_levels: int = 32,
        max_coords: int = 8192,
        num_buckets: int = 512,
        seed: int = 42,
        device: Union[str, torch.device] = "cuda",
        token_chunk: int = 4096,
    ) -> None:
        if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA is not available; HDSEncoder is falling back to CPU.", stacklevel=2)
            device = "cpu"
        self.device = torch.device(device)
        # Cache a scalar 1.0 tensor for scatter_add and a level offset cache to avoid repeated allocations
        self._one = torch.ones((), dtype=torch.float32, device=self.device)
        self._lv_cache: dict[int, torch.Tensor] = {}

        self.hd_dim = hd_dim
        self.num_levels = num_levels
        self.max_coords = max_coords
        self.num_buckets = num_buckets
        self.seed = seed
        self.token_chunk = token_chunk
        self.flips_per_level = hd_dim // (2 * (num_levels - 1))
        if self.flips_per_level * (num_levels - 1) > hd_dim:
            raise ValueError("hd_dim too small to flip disjoint dimension sets for every level")

        self.L = self._build_level_codebook().to(self.device)
        self.C_raw = self._random_bipolar(max_coords, seed + 1).to(self.device)
        self.C_bucket = self._random_bipolar(num_buckets, seed + 2).to(self.device)
        self._jl_cache: dict = {}

    # ------------------------------------------------------------------ codebooks
    def _random_bipolar(self, n: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        bits = torch.randint(0, 2, (n, self.hd_dim), generator=g, dtype=torch.int8)
        return (bits * 2 - 1).to(torch.float32)

    def _build_level_codebook(self) -> torch.Tensor:
        """L[0] random bipolar; level m flips `flips_per_level` fresh dims (no replacement) of level m-1."""
        g = torch.Generator().manual_seed(self.seed)
        L = torch.empty(self.num_levels, self.hd_dim, dtype=torch.float32)
        L[0] = torch.randint(0, 2, (self.hd_dim,), generator=g, dtype=torch.int8).float() * 2 - 1
        perm = torch.randperm(self.hd_dim, generator=g)  # disjoint flip sets = consecutive slices
        k = self.flips_per_level
        for m in range(1, self.num_levels):
            L[m] = L[m - 1]
            L[m, perm[(m - 1) * k : m * k]] *= -1
        return L

    def jl_matrix(self, d_out: int) -> torch.Tensor:
        """Gaussian JL matrix ``Pi`` ``[d_out, D] ~ N(0, 1/D)`` (cached, deterministic per d_out)."""
        if d_out not in self._jl_cache:
            g = torch.Generator().manual_seed(self.seed + 3 + d_out)
            pi = torch.randn(d_out, self.hd_dim, generator=g) / math.sqrt(self.hd_dim)
            self._jl_cache[d_out] = pi.to(self.device)
        return self._jl_cache[d_out]

    # ------------------------------------------------------------------ fast path
    def level_frequencies(
        self, R_pb: torch.Tensor, return_mean_rank: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        R = R_pb.to(device=self.device, dtype=torch.float32)
        T, d_out = R.shape
        M = self.num_levels
        lv = self._lv_cache.get(d_out)
        if lv is None:
            lv = ((torch.arange(d_out, device=self.device, dtype=torch.int64) * M) // d_out).unsqueeze(0)
            self._lv_cache[d_out] = lv
        F = torch.zeros(d_out * M, dtype=torch.float32, device=self.device)
        rank_sum = torch.zeros(d_out, dtype=torch.float32, device=self.device) if return_mean_rank else None
        ar = torch.arange(d_out, device=self.device)

        for chunk in R.split(self.token_chunk, dim=0):
            order = torch.sort(chunk, dim=1, stable=True).indices
            if return_mean_rank:
                ranks = torch.empty_like(order)
                ranks.scatter_(1, order, ar.expand_as(order))
                rank_sum += ranks.sum(dim=0, dtype=torch.float32)
                del ranks
            torch.add(lv, order, alpha=M, out=order)
            F.scatter_add_(0, order.view(-1), self._one.expand(order.numel()))

        F = F.view(d_out, M) / T
        if return_mean_rank:
            return F, rank_sum / (T * d_out)
        return F

    def bucket_indices(self, fp16_col_norms: torch.Tensor) -> torch.Tensor:
        """``bucket(j) = (salience_rank(j) * num_buckets) // d_out``; rank 0 = largest ||Y[:, j]||_2."""
        norms = fp16_col_norms.to(device=self.device, dtype=torch.float32)
        d_out = norms.numel()
        order = torch.sort(norms, descending=True, stable=True).indices
        salience_rank = torch.empty_like(order)
        salience_rank[order] = torch.arange(d_out, device=self.device)
        return (salience_rank * self.num_buckets) // d_out

    def bind(self, F: torch.Tensor, fp16_col_norms: Optional[torch.Tensor], coord_binding: str) -> torch.Tensor:
        """``v = ((C_mapped * (F @ L)).sum(0)) / sqrt(d_out)`` ``[D]``.

        For ``salience_bucket`` ``C_mapped[j] = C_bucket[bucket(j)]``; since the sum over j is linear,
        rows of ``F`` are first summed per bucket (``index_add_``) so the product is ``[512, D]`` instead
        of ``[d_out, D]``. This is algebraically identical to materializing ``C_mapped``.
        """
        _check_binding(coord_binding)
        d_out = F.shape[0]
        if coord_binding == "raw_index_strict":
            if d_out > self.max_coords:
                raise ValueError(f"d_out={d_out} exceeds max_coords={self.max_coords}")
            return (self.C_raw[:d_out] * (F @ self.L)).sum(dim=0) / math.sqrt(d_out)
        if fp16_col_norms is None or fp16_col_norms.numel() != d_out:
            raise ValueError("salience_bucket binding needs fp16_col_norms of length d_out")
        bucket = self.bucket_indices(fp16_col_norms)
        F_bucket = torch.zeros(self.num_buckets, F.shape[1], dtype=F.dtype, device=F.device)
        F_bucket.index_add_(0, bucket, F)
        return (self.C_bucket * (F_bucket @ self.L)).sum(dim=0) / math.sqrt(d_out)

    def encode(self, R_pb: torch.Tensor, fp16_col_norms: Optional[torch.Tensor], coord_binding: str) -> torch.Tensor:
        """Encode ``R_pb`` ``[T, d_out]`` into the HDS hypervector ``v`` ``[hd_dim]`` (float32, on ``device``)."""
        _check_binding(coord_binding)
        F = self.level_frequencies(R_pb)
        return self.bind(F, fp16_col_norms, coord_binding)

    # ------------------------------------------------------------------ reference path
    @torch.no_grad()
    def encode_naive_reference(
        self, R_pb: torch.Tensor, fp16_col_norms: Optional[torch.Tensor], coord_binding: str
    ) -> torch.Tensor:
        """Slow token-by-token, coordinate-by-coordinate bind-and-bundle. Intended for tiny inputs only.

        For each token t: ``h_t = sum_j C[bucket_or_j] * L[level_idx[t, j]]`` (elementwise bind, bundle by sum);
        then ``v = mean_t(h_t) / sqrt(d_out)``.
        """
        _check_binding(coord_binding)
        R = R_pb.to(device=self.device, dtype=torch.float32)
        T, d_out = R.shape
        M = self.num_levels

        if coord_binding == "raw_index_strict":
            if d_out > self.max_coords:
                raise ValueError(f"d_out={d_out} exceeds max_coords={self.max_coords}")
            coord_hv = [self.C_raw[j] for j in range(d_out)]
        else:
            norms = fp16_col_norms.to(device=self.device, dtype=torch.float32)
            desc = sorted(range(d_out), key=lambda j: (-norms[j].item(), j))  # stable, descending
            salience_rank = {j: r for r, j in enumerate(desc)}
            coord_hv = [self.C_bucket[(salience_rank[j] * self.num_buckets) // d_out] for j in range(d_out)]

        total = torch.zeros(self.hd_dim, dtype=torch.float32, device=self.device)
        for t in range(T):
            row = R[t].tolist()
            asc = sorted(range(d_out), key=lambda j: (row[j], j))  # stable ascending
            ranks = [0] * d_out
            for r, j in enumerate(asc):
                ranks[j] = r
            token_hv = torch.zeros(self.hd_dim, dtype=torch.float32, device=self.device)
            for j in range(d_out):
                level = (ranks[j] * M) // d_out
                token_hv += coord_hv[j] * self.L[level]
            total += token_hv
        return total / T / math.sqrt(d_out)


def _check_binding(coord_binding: str) -> None:
    if coord_binding not in COORD_BINDINGS:
        raise ValueError(f"coord_binding={coord_binding!r} invalid; expected one of {COORD_BINDINGS}")
