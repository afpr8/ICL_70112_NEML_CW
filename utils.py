# Utilities file for reusable functions for LAND algorithms

from functools import wraps
from typing import Callable, Any

import torch


def tensor_cache(
    func: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    Decorator to cache function outputs for PyTorch tensor inputs
    Converts tensor arguments to bytes for hashing
    
    Params:
        func: A function that returns a torch.Tensor and takes tensor
            (or non-tensor) arguments.
    Returns:
        wrapper: Cached version of the function
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
        # Convert tensors to bytes to use as a cache key
        key = tuple(
            arg.detach().cpu().numpy().tobytes()
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in args
        ) + tuple(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return wrapper


@tensor_cache
def exp_map(x:torch.Tensor, v:torch.Tensor, M_x:torch.Tensor) -> torch.Tensor:
    """
        The exponential map corresponding to the LAND metric implemented below
        Params:
            x (d,): The query point
            v (d,): The tangent vector to map
            M_x (d,d): The local metric tensor
        Returns:
            exp_xv (d,): The point on the manifold obtained by mapping v
                from the tangent plane at x utilizing M_x
    """
    diag_sqrt_inv = torch.sqrt(torch.diag(M_x))**(-1)
    v_scaled = diag_sqrt_inv * v

    return x + v_scaled


@tensor_cache
def log_map(x:torch.Tensor, y:torch.Tensor, M_x:torch.Tensor) -> torch.Tensor:
    """
        The logarithmic map corresponding to the LAND metric implemented below
        Params:
            x (d,): The query point
            y (d,): The manifold point to map to the tangent plane
            M_x (d,d): The local metric tensor
        Returns:
            log_xy (d,): The vector on the tangent plane at x obtained by
                mapping y from the manifold utilizing M_x
    """
    diag_sqrt = torch.sqrt(torch.diag(M_x))
    log_xy = diag_sqrt * (y - x)

    return log_xy


@tensor_cache
def metric(
    x:torch.Tensor,
    X:torch.Tensor,
    sigma:float=1.0,
    rho:float=1e-3
) -> torch.Tensor:
    """
        Compute the local Riemmanian metric tensor at point x
            This metric was chosen by the LAND authors and could be changed
        Params:
            x (d,): The query point
            X (n_samples, d): Data points
            sigma: Gaussian kernel width
            rho: Regularization to ensure the metric matrix is Positive Definite
        Returns:
            M_x (d, d): The local metric tensor
    """
    # Compute squared Euclidean distances & normalize
    diff = X - x.unsqueeze(0)
    dist2 = torch.sum(diff**2, dim=1)
    w = torch.exp(-dist2 / (2 * sigma**2))

    # Compute weighted diagonal covariance matrix & invert
    weighted_sq = torch.sum(w.unsqueeze(1) * (diff**2), dim=0)
    diag_entries = weighted_sq + rho
    M_x = torch.diag(1.0 / diag_entries) # Inversion of a diagonal matrix

    return M_x


def compute_normalization_constant(
    mu:torch.Tensor,
    sigma:torch.Tensor,
    metric_fn,
    n_samples:int=3000 # LAND paper uses 3000 in all experiments
) -> torch.Tensor:
    """
        Compute the multivariate normalization constant
        Params:
            mu (d,): 1D mean tensor
            sigma (d,d): 2D covariance tensor
            metric_fn: The 
            n_samples (optional): The number of Monte Carlo samples to use
        Returns:
            C: The Monte Carlo estimate of the normalizatio constant
    """
    d = mu.shape[0]

    Z = torch.sqrt((2 * torch.pi)**d * torch.det(sigma))
    mvn = torch.distributions.MultivariateNormal(
        loc=torch.zeros(d),
        covariance_matrix=sigma
    )
    v_samples = mvn.sample((n_samples, ))
    M_mu = metric_fn(mu)

    vol_elements = []

    for v in v_samples:
        x = exp_map(mu, v, M_mu)
        M_x = metric_fn(x)

        diag_entries = torch.diag(M_x)
        log_det = torch.sum(torch.log(diag_entries))
        vol = torch.exp(0.5 * log_det)

        vol_elements.append(vol)        

    return Z * torch.stack(vol_elements).mean()
