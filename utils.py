import torch


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


def compute_normalization_constant(
    mu: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    pass

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
