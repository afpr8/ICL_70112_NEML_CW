import torch


def exp_map(mu: torch.Tensor, v: torch.Tensor, M_x: torch.Tensor) -> torch.Tensor: ...


def log_map(mu: torch.Tensor, v: torch.Tensor, M_x: torch.Tensor) -> torch.Tensor: ...


def compute_normalization_constant(
    mu: torch.Tensor, sigma: torch.Tensor, metric_fn: callable, n_samples: int = 3000
) -> torch.Tensor: ...


def metric(
    x: torch.Tensor, X: torch.Tensor, sigma: float = 1.0, rho: float = 1e-3
) -> torch.Tensor: ...
