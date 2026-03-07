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

    sol = optx.root_find(residual, solver, y0=v_guess, args=None, throw=False)

    return sol.value


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
