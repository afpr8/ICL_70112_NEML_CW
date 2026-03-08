# Utilities file for reusable functions for LAND algorithms

from typing import Callable

import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx


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
    # solver = diffrax.Euler()
    solver = diffrax.Dopri5()
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
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.DirectAdjoint(),
    )
    d = x.shape[0]
    return sol.ys[0, :d]

def jax_log_map_shooting(mu: jnp.ndarray, x: jnp.ndarray, metric_fn: Callable, N_waypoints: int = 15, max_iter: int = 200, lr: float = 0.05) -> jnp.ndarray:
    """
    Computes the Riemannian Logarithmic map using discrete curve energy minimisation.
    This is highly stable and compiles perfectly in JAX.
    """
    # 1. Initialise a straight line between mu and x
    t = jnp.linspace(0, 1, N_waypoints)[:, None]
    initial_curve = (1 - t) * mu + t * x
    
    # We only optimise the internal waypoints. The boundaries (mu and x) remain fixed.
    internal_points = initial_curve[1:-1]
    
    def curve_energy(internal_pts):
        """Calculates the discrete Riemannian energy of the curve."""
        # Reconstruct the full curve
        full_curve = jnp.vstack([mu, internal_pts, x])
        
        # Compute discrete velocities (forward finite differences)
        velocities = (full_curve[1:] - full_curve[:-1]) * (N_waypoints - 1)
        
        # Evaluate the metric at the midpoints of each segment
        midpoints = (full_curve[1:] + full_curve[:-1]) / 2.0
        
        def point_energy(m, v):
            M = metric_fn(m)
            return jnp.dot(v, jnp.dot(M, v))
        
        # Sum the energy along the curve
        energies = jax.vmap(point_energy)(midpoints, velocities)
        return jnp.sum(energies) / (N_waypoints - 1)

    # Automatically derive the gradient of the energy with respect to the waypoints
    grad_energy_fn = jax.grad(curve_energy)

    # 2. Optimise the curve to pull it taut (minimise energy)
    def cond_fun(val):
        i, pts, prev_loss = val
        return i < max_iter
        
    def body_fun(val):
        i, pts, prev_loss = val
        g = grad_energy_fn(pts)
        
        # Simple gradient descent update
        pts_new = pts - lr * g
        loss_new = curve_energy(pts_new)
        
        return i + 1, pts_new, loss_new
        
    init_val = (0, internal_points, curve_energy(internal_points))
    
    # Run the optimisation loop entirely on the accelerator
    _, final_internal_pts, _ = jax.lax.while_loop(cond_fun, body_fun, init_val)
    
    # 3. The log map is simply the initial velocity of this optimised geodesic
    final_curve = jnp.vstack([mu, final_internal_pts, x])
    initial_velocity = (final_curve[1] - final_curve[0]) * (N_waypoints - 1)
    
    return initial_velocity

# # --- Approach A: JAX Shooting Method ---
# @jax.jit(static_argnames=["metric_fn"])
# def jax_log_map_shooting(
#     x: jnp.ndarray,
#     y_target: jnp.ndarray,
#     metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
# ) -> jnp.ndarray:
#     def residual(v, args):
#         return jax_exp_map(x, v, metric_fn) - y_target

#     solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
#     v_guess = y_target - x

#     sol = optx.root_find(residual, solver, y0=v_guess, args=None, throw=False)

#     return sol.value


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
