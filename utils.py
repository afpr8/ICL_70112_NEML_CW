# Utilities file for reusable functions for LAND algorithms

from functools import wraps
from typing import Callable, Any

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx
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
def torch_exp_map(
    x: torch.Tensor, v: torch.Tensor, metric_fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
        The exponential map corresponding to the LAND metric implemented below
        Params:
            x (d,): The query point
            v (d,): The tangent vector to map
            metric_fn (callable): A function returning the metric tensor at a given point
        Returns:
            exp_xv (d,): The point on the manifold obtained by mapping v
                from the tangent plane at x utilizing M_x
    """
    M_x = metric_fn(x)
    diag_sqrt_inv = torch.sqrt(torch.diag(M_x))**(-1)
    v_scaled = diag_sqrt_inv * v

    return x + v_scaled

@tensor_cache
def torch_log_map(
    x: torch.Tensor, y: torch.Tensor, metric_fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """
        The logarithmic map corresponding to the LAND metric implemented below
        Params:
            x (d,): The query point
            y (d,): The manifold point to map to the tangent plane
            metric_fn (callable): A function returning the metric tensor at a given point
        Returns:
            log_xy (d,): The vector on the tangent plane at x obtained by
                mapping y from the manifold utilizing M_x
    """
    M_x = metric_fn(x)
    diag_sqrt = torch.sqrt(torch.diag(M_x))
    log_xy = diag_sqrt * (y - x)

    return log_xy

@tensor_cache
def torch_metric(
    x: torch.Tensor, X: torch.Tensor, sigma: float = 1.0, rho: float = 1e-3
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


@jax.jit
def jax_metric(
    x: jnp.ndarray, X: jnp.ndarray, sigma: float = 1.0, rho: float = 1e-3
) -> jnp.ndarray:
    """
    Compute the local Riemmanian metric tensor at point x using jax
        This metric was chosen by the LAND authors and could be changed
    Params:
        x (d,): The query point
        X (n_samples, d): Data points
        sigma: Gaussian kernel width
        rho: Regularization to ensure the metric matrix is Positive Definite
    Returns:
        M_x (d, d): The local metric tensor
    """
    # Gaussian weights
    diff = X - x[None, :]
    dist2 = jnp.sum(diff**2, axis=-1)
    weights = jnp.exp(-dist2 / (2.0 * sigma**2))

    # Compute weighted diagonal covariance matrix & invert
    weighted_sq = jnp.sum(weights[:, None] * (diff**2), axis=0)
    diag_entries = weighted_sq + rho
    M_x = jnp.diag(1.0 / diag_entries)  # Inversion of a diagonal matrix

    return M_x


@jax.jit(static_argnames=["metric_fn"])
def jax_geodesic_ode(
    x: jnp.ndarray,
    v: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    M_inv = jnp.linalg.inv(metric_fn(x))

    # Flattened metric for jacobian computation (closing over metric_fn)
    def vec_M(pos):
        return metric_fn(pos).flatten()

    # Right Hand Side
    J = jax.jacobian(vec_M)(x)
    v_kron_v = jnp.kron(v, v)
    term = J.T @ v_kron_v

    return -0.5 * M_inv @ term


@jax.jit(static_argnames=["metric_fn"])
def jax_exp_map(
    x: jnp.ndarray,
    v: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    def jax_vector_field(t, y, args):
        d = y.shape[0] // 2
        x_pt, v_pt = y[:d], y[d:]
        a = jax_geodesic_ode(x_pt, v_pt, metric_fn)
        return jnp.concatenate([v_pt, a])

    term = diffrax.ODETerm(jax_vector_field)
    solver = diffrax.Tsit5()
    y0 = jnp.concatenate([x, v])

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,  # Initial time at 0 (the base point)
        t1=1.0,  # Final time at 1 (the end of the geodesic)
        dt0=0.1,  # TODO check other options and why
        y0=y0,
        args=None,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
        adjoint=diffrax.DirectAdjoint(),
    )
    d = x.shape[0]
    return sol.ys[0, :d]


# --- Approach A: JAX Shooting Method ---
@jax.jit(static_argnames=["metric_fn"])
def jax_log_map_shooting(
    x: jnp.ndarray,
    y_target: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    def residual(v, args):
        return jax_exp_map(x, v, metric_fn) - y_target

    solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
    v_guess = y_target - x

    sol = optx.root_find(residual, solver, y0=v_guess, args=None)

    return sol.value


def torch_compute_normalization_constant(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    metric_fn,
    n_samples: int = 3000,  # LAND paper uses 3000 in all experiments
) -> torch.Tensor:
    """
        Compute the multivariate normalization constant
        Params:
            mu (d,): 1D mean tensor
            sigma (d,d): 2D covariance tensor
            metric_fn: The metric function to compute the local metric tensor at any point
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

    vol_elements = []

    for v in v_samples:
        x = torch_exp_map(mu, v, metric_fn)
        M_x = metric_fn(x)

        diag_entries = torch.diag(M_x)
        log_det = torch.sum(torch.log(diag_entries))
        vol = torch.exp(0.5 * log_det)

        vol_elements.append(vol)        

    return Z * torch.stack(vol_elements).mean()

@jax.jit(static_argnames=["metric_fn", "n_samples"])
def compute_normalization_constant(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
    key: jax.Array,
    n_samples: int = 3000,
) -> jnp.ndarray:
    d = mu.shape[0]
    Z = jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(sigma))

    v_samples = jax.random.multivariate_normal(
        key, mean=jnp.zeros(d), cov=sigma, shape=(n_samples,)
    )

    def compute_vol(v):
        x = jax_exp_map(mu, v, metric_fn)
        M_x = metric_fn(x)
        log_det = jnp.sum(jnp.log(jnp.diag(M_x)))
        return jnp.exp(0.5 * log_det)

    vols = jax.vmap(compute_vol)(v_samples)
    return Z * jnp.mean(vols)


# Plotting utilities

def plot_geodesics(ax, X, means, labels, geodesics):
    """
    Visualise the data, cluster means, and the geodesics connecting 
    each data point to its assigned cluster mean.
    """

    cluster_colours = ['#8FBC8F', '#CD5C5C'] # Greenish and Reddish
    ax.scatter(X[:, 0], X[:, 1], c='#5DADE2', s=20, label='Data', zorder=2)
    
    # Plot the geodesics
    # 'geodesics' is expected to be a list of (N_steps, 2) arrays
    plotted_labels = set()
    for i, path in enumerate(geodesics):
        cluster_idx = labels[i]
        col = cluster_colours[cluster_idx % len(cluster_colours)]
        
        lbl = f'Geodesics, cluster {cluster_idx + 1}'
        if lbl in plotted_labels:
            lbl = None # Avoid duplicate legend entries
        else:
            plotted_labels.add(lbl)
            
        ax.plot(path[:, 0], path[:, 1], c=col, alpha=0.4, linewidth=1, label=lbl, zorder=1)

    # Plot the cluster means
    ax.scatter(means[:, 0], means[:, 1], c='orange', marker='D', s=100, 
               edgecolors='black', label='LAND means', zorder=3)
    
    ax.set_title('Geodesics')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower left', fontsize='small')


def plot_mixture_contours(ax, X, means, X_grid, Y_grid, Z, title, mean_label):
    """
    Visualise the data, cluster means, and the density contours of a mixture model.
    """
    # Plot contours (using 'jet' to match the rainbow gradient style)
    ax.contour(X_grid, Y_grid, Z, levels=10, cmap='jet', linewidths=1.5, zorder=1)
    
    # Plot the data points
    ax.scatter(X[:, 0], X[:, 1], c='#5DADE2', s=20, zorder=2)
    
    # Plot the cluster means
    ax.scatter(means[:, 0], means[:, 1], c='orange', marker='D', s=100, 
               edgecolors='black', label=mean_label, zorder=3)
    
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize='small')


def plot_full_comparison(X, land_means, gmm_means, labels, geodesics, 
                         X_grid, Y_grid, Z_land, Z_gmm):
    """
    Creates a full 1x3 figure comparing geodesics, LAND contours, and GMM contours.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Geodesics Plot
    plot_geodesics(axes[0], X, land_means, labels, geodesics)
    
    # LAND Mixture Model Contours
    plot_mixture_contours(axes[1], X, land_means, X_grid, Y_grid, Z_land, 
                          title='LAND mixture model', mean_label='LAND mean')
    
    # Gaussian Mixture Model Contours
    plot_mixture_contours(axes[2], X, gmm_means, X_grid, Y_grid, Z_gmm, 
                          title='Gaussian mixture model', mean_label='GMM mean')
    
    plt.tight_layout()
    return fig
