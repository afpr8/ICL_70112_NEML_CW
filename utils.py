import torch


def exp_map(mu: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


def log_map(mu: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


def compute_normalization_constant(
    mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor: ...
