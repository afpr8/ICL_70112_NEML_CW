# Utilities file for reusable functions for LAND algorithms

# Standard library imports
from functools import wraps
from typing import Callable, Any

# Third party imports
import torch
from torchdiffeq import odeint

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
        args_key = tuple(
            arg.detach().cpu().numpy().tobytes()
            if isinstance(arg, torch.Tensor)
            else arg
            for arg in args
        )
        kwargs_key = tuple(
            (k, v.detach().cpu().numpy().tobytes()
            if isinstance(v, torch.Tensor)
            else v)
            for k, v in sorted(kwargs.items())
        )
        key = args_key + kwargs_key
        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return wrapper


@tensor_cache
def exp_map(
    x: torch.Tensor,
    v: torch.Tensor,
    M_x: torch.Tensor
) -> torch.Tensor:
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
def log_map(
    x: torch.Tensor,
    y: torch.Tensor,
    M_x: torch.Tensor
) -> torch.Tensor:
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
    x: torch.Tensor,
    X: torch.Tensor,
    sigma: float = 1.0,
    rho: float = 1e-3
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


# Note that this function assumes our metric is diagonal (the LAND one is)
@tensor_cache
def geodesic_ode(
    t: torch.Tensor, # Even though t is not used the odeint solver passes one
    y: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor], # Requires partial init
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Defines the geodesic ODE for a Riemannian manifold with a diagonal metric
        The ODE system is first-order: dy/dt = [dx/dt, dv/dt], where v = dx/dt
    
    Params:
        t: Current time (scalar tensor)
        y: State tensor concatenating position and velocity [x, v], shape (2*d,)
        metric_fn: Function that takes x (d,) and returns the metric M(x) (d,d)
        eps: Small perturbation for numerical derivatives
    Returns:
        dy/dt: Tensor of shape (2*d,) representing [dx/dt, dv/dt]
    """
    d = y.shape[0] // 2
    x = y[:d]
    v = y[d:]
    
    M_x = metric_fn(x) # (d,d)
    M_inv = torch.diag(1 / torch.diag(M_x))
    
    # Compute derivative of diagonal entries wrt x_1
    # Note this is an approximate derivation for computational efficiency
    dM_dx = torch.zeros(d)
    for i in range(d):
        x_eps = x.clone()
        x_eps[i] += eps
        M_eps = metric_fn(x_eps)
        dM_dx[i] = (M_eps[i,i] - M_x[i,i]) / eps

    # Christoffel symbols for diagonal LAND
    # Gamma^i_ii = 0.5 * M^ii * dM_dx[i]
    gamma = 0.5 * torch.diag(M_inv) * dM_dx

    dx_dt = v
    dv_dt = - gamma * v**2

    return torch.cat([dx_dt, dv_dt])


@tensor_cache
def exp_map_geodesic(
    x: torch.Tensor,
    v: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor], # Requires partial init
    t_span: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes the exp map of a tangent vector v at point x along the manifold
        defined by the given Riemannian metric, by integrating the geodesic ODE
    
    Params:
        x: Base point on the manifold, shape (d,)
        v: Tangent vector at x, shape (d,)
        metric_fn: Function that takes x (d,) and returns metric M(x) (d,d)
        t_span: Optional 1D tensor specifying integration times (default [0,1])
    Returns:
        exp_xv: Point on the manifold corresponding to exp_x(v), shape (d,)
    """
    # Using a default range for t form 0 to 1
    if t_span is None:
        t_span = torch.tensor([0., 1.], dtype=x.dtype, device=x.device)

    y0 = torch.cat([x, v])
    y_t = odeint(lambda t, y: geodesic_ode(t, y, metric_fn), y0, t_span)

    return y_t[-1,:x.shape[0]]  # only x coordinates


# Following the LAND paper, we obtain the log_map_geodesic by solving the IVP
#   for the exp map & then numerically find the inverse rather than analytically
#   solving the BVP because it is substantially more expensive to compute
@tensor_cache
def log_map_geodesic(
    x: torch.Tensor,
    y: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    n_iter: int = 50, # From looking online, 20-100 is a reasonable range
    lr: float = 0.1, # A typical moderate lr, Adam uses "adaptive step sizes"
    tol: float = 1e-6 # Mapping error tolerance threshold
) -> torch.Tensor:
    """
    Computes the logarithmic map log_x(y) using a shooting method
        Finds the tangent vector v at x s.t. exp_x(v) = y along the manifold
    
    Params:
        x: Base point on the manifold, shape (d,)
        y: Target point on the manifold, shape (d,)
        metric_fn: Function that takes x (d,) and returns metric M(x) (d,d)
        n_iter: Number of optimization steps for shooting method
        lr: Learning rate for tangent vector optimization
        tol: Residual tolerance for early stopping
    Returns:
        v: Tangent vector at x such that exp_x(v) ≈ y, shape (d,)
    """
    v = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=lr)

    for _ in range(n_iter):
        x_end = exp_map_geodesic(x, v, metric_fn)
        residual = torch.norm(x_end - y)
        if residual < tol:
            break  # early stopping if close enough

        loss = residual**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return v.detach()
