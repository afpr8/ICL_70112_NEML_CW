# Utilities file for reusable functions for LAND algorithms

from functools import wraps
from typing import Callable, Any

import matplotlib.pyplot as plt
import numpy as np
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

from geomstats.geometry.riemannian_metric import RiemannianMetric

def exp_map(x: torch.Tensor, v: torch.Tensor, metric: RiemannianMetric) -> torch.Tensor:
    """
    Computes the exponential map using the geomstats backend.
    
    Params:
        x (d,): The query point on the manifold.
        v (d,): The tangent vector to map.
        metric (RiemannianMetric): The instantiated geomstats LAND metric.
    Returns:
        y_hat (d,): The point on the manifold obtained by mapping v.
    """
    # geomstats automatically solves the Initial Value Problem (IVP)
    return metric.exp(tangent_vec=v, base_point=x)

def log_map(x: torch.Tensor, y: torch.Tensor, metric: RiemannianMetric) -> torch.Tensor:
    """
    Computes the logarithmic map using the geomstats backend.
    
    Params:
        x (d,): The query point on the manifold.
        y (d,): The target manifold point to map to the tangent plane.
        metric (RiemannianMetric): The instantiated geomstats LAND metric.
    Returns:
        v (d,): The tangent vector at x that shoots exactly to y.
    """
    # geomstats automatically solves the Boundary Value Problem (BVP)
    return metric.log(point=y, base_point=x)

# @tensor_cache
# def exp_map(x:torch.Tensor, v:torch.Tensor, metric_fn: callable) -> torch.Tensor:
#     """
#         The exponential map corresponding to the LAND metric implemented below
#         Params:
#             x (d,): The query point
#             v (d,): The tangent vector to map
#             metric_fn (callable): A function returning the metric tensor at a given point
#         Returns:
#             exp_xv (d,): The point on the manifold obtained by mapping v
#                 from the tangent plane at x utilizing M_x
#     """
#     M_x = metric_fn(x)
#     diag_sqrt_inv = torch.sqrt(torch.diag(M_x))**(-1)
#     v_scaled = diag_sqrt_inv * v

#     return x + v_scaled

# @tensor_cache
# def exp_map(x: torch.Tensor, v: torch.Tensor, metric_fn: callable, steps: int = 10) -> torch.Tensor:
#     """
#     The exponential map corresponding to the LAND metric.
#     Computes the map by solving the geodesic ODE as an Initial Value Problem.

#     Params:
#         x (d,): The query point on the manifold.
#         v (d,): The tangent vector to map.
#         metric_fn (callable): A function returning the (d, d) metric tensor at a point.
#         steps (int): Number of integration steps to discretise the path.
#     Returns:
#         gamma (d,): The resulting point on the manifold.
#     """
#     # Time step for the integration from t=0 to t=1
#     dt = 1.0 / steps

#     # Initialise the curve position (gamma) and its velocity (gamma_dot)
#     gamma = x.clone()
#     gamma_dot = v.clone()

#     # Helper function to get the flattened metric for the Jacobian calculation
#     def metric_vec_fn(pos):
#         return metric_fn(pos).view(-1)

#     for _ in range(steps):
#         # 1. Compute the metric tensor and its inverse at the current position
#         M = metric_fn(gamma)
#         M_inv = torch.linalg.inv(M)

#         # 2. Compute the Jacobian of the vectorised metric tensor wrt position
#         # Shape: (d*d, d)
#         J = torch.autograd.functional.jacobian(metric_vec_fn, gamma, create_graph=True)

#         # 3. Compute the Kronecker product of the velocity vector
#         # Shape: (d*d,)
#         v_kron_v = torch.kron(gamma_dot, gamma_dot)

#         # 4. Calculate the acceleration (gamma_ddot) based on the geodesic ODE
#         # gamma'' = -0.5 * M^{-1} [d(vec(M))/d(gamma)]^T (v \otimes v)
#         force = 0.5 * torch.mv(J.t(), v_kron_v)
#         gamma_ddot = -torch.mv(M_inv, force)

#         # 5. Euler integration step
#         gamma = gamma + gamma_dot * dt
#         gamma_dot = gamma_dot + gamma_ddot * dt

#     return gamma


# # @tensor_cache
# # def log_map(x:torch.Tensor, y:torch.Tensor, metric_fn: callable) -> torch.Tensor:
# #     """
# #         The logarithmic map corresponding to the LAND metric implemented below
# #         Params:
# #             x (d,): The query point
# #             y (d,): The manifold point to map to the tangent plane
# #             metric_fn (callable): A function returning the metric tensor at a given point
# #         Returns:
# #             log_xy (d,): The vector on the tangent plane at x obtained by
# #                 mapping y from the manifold utilizing M_x
# #     """
# #     M_x = metric_fn(x)
# #     diag_sqrt = torch.sqrt(torch.diag(M_x))
# #     log_xy = diag_sqrt * (y - x)

# #     return log_xy

# def log_map(x: torch.Tensor, y: torch.Tensor, metric_fn: callable, max_iters: int = 20, lr: float = 0.1) -> torch.Tensor:
#     """
#     The logarithmic map corresponding to the LAND metric.
#     Computes the map by solving the BVP via a gradient-based shooting method.
#     """
#     # 1. Detach inputs to isolate this inner optimisation from the outer EM graph
#     x_in = x.detach()
#     y_in = y.detach()

#     M_x = metric_fn(x_in)
#     diag_sqrt = torch.sqrt(torch.diag(M_x))
    
#     # 2. Initialise the velocity and flag it for gradients
#     v = (diag_sqrt * (y_in - x_in)).clone().detach().requires_grad_(True)
    
#     optimiser = torch.optim.Adam([v], lr=lr)
    
#     for _ in range(max_iters):
#         optimiser.zero_grad()
        
#         # Shoot the curve from x_in using the current velocity v
#         y_hat = exp_map(x_in, v, metric_fn) 
        
#         loss = torch.sum((y_hat - y_in) ** 2)
        
#         if loss.item() < 1e-5:
#             break
            
#         loss.backward()
#         optimiser.step()
        
#     return v.detach()

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
        x = exp_map(mu, v, metric_fn)
        M_x = metric_fn(x)

        diag_entries = torch.diag(M_x)
        log_det = torch.sum(torch.log(diag_entries))
        vol = torch.exp(0.5 * log_det)

        vol_elements.append(vol)        

    return Z * torch.stack(vol_elements).mean()


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