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
    solver = diffrax.Heun()
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

# --- Approach A: JAX Shooting Method ---
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

@jax.jit(static_argnames=["metric_fn"])
def jax_exp_map_rk4(
    x: jnp.ndarray, 
    v: jnp.ndarray, 
    metric_fn: Callable, 
    steps: int = 100
) -> jnp.ndarray:
    """
    4th-order Runge-Kutta (RK4) implementation of the Exponential Map.
    Provides O(h^4) precision for the geodesic path.
    """
    h = 1.0 / steps
    
    @jax.jit(static_argnames=["metric_fn"])
    def compute_geodesic_acceleration(
        pos: jnp.ndarray, 
        vel: jnp.ndarray, 
        metric_fn: Callable
    ) -> jnp.ndarray:
        """
        Computes the geodesic acceleration ddc = -Gamma(c, dc).
        Uses JAX autodiff to derive the metric derivatives.
        """
        # 1. Compute the metric and its inverse at the current position
        M = metric_fn(pos)
        inv_M = jnp.linalg.inv(M + 1e-9 * jnp.eye(pos.shape[0])) # Stability jitter

        # 2. Compute the gradient of the metric tensor w.r.t. position
        # The Jacobian of the metric will have shape (D, D, D)
        dM = jax.jacobian(metric_fn)(pos)

        # 3. Compute the Christoffel symbols (contracted with velocities)
        # The term we need is: 0.5 * inv_M * (2 * dM_vel * vel - grad(vel.T @ M @ vel))
        # Alternatively, using the explicit geodesic equation components:
        
        # term1 = \partial_j M_{ik} * v^i * v^j
        term1 = jnp.einsum('ijk,i,j->k', dM, vel, vel)
        
        # term2 = 0.5 * \partial_k M_{ij} * v^i * v^j
        # This represents the derivative of the kinetic energy w.r.t. position
        def kinetic_energy(p):
            metric = metric_fn(p)
            return 0.5 * jnp.dot(vel, jnp.dot(metric, vel))
        
        term2 = jax.grad(kinetic_energy)(pos)

        # Acceleration ddc = -inv_M @ (term1 - term2)
        accel = -jnp.dot(inv_M, term1 - term2)
        
        return accel

    def geodesic_ode(state):
        # State contains [position, velocity]
        pos, vel = state
        # compute_accel is your geodesic equation: ddc = -Gamma(c, dc)
        accel = compute_geodesic_acceleration(pos, vel, metric_fn)
        return jnp.stack([vel, accel])

    def rk4_step(state, _):
        k1 = geodesic_ode(state)
        k2 = geodesic_ode(state + h/2 * k1)
        k3 = geodesic_ode(state + h/2 * k2)
        k4 = geodesic_ode(state + h * k3)
        return state + (h/6) * (k1 + 2*k2 + 2*k3 + k4), None

    initial_state = jnp.stack([x, v])
    final_state, _ = jax.lax.scan(rk4_step, initial_state, None, length=steps)
    return final_state[0] # Return the position at t=1

@jax.jit(static_argnames=["metric_fn"])
def jax_log_map_shooting(
    x: jnp.ndarray,
    y_target: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Energy-based Log Map using Levenberg-Marquardt.
    Minimises the residual energy between the shot point and target.
    """
    def energy_residual(v, args):
        # We use the RK4 version for higher precision
        y_pred = jax_exp_map_rk4(x, v, metric_fn)
        return y_pred - y_target

    # Levenberg-Marquardt is excellent for least-squares energy minimisation
    solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6)
    
    # Use Euclidean difference as a warm-start guess
    v_guess = y_target - x

    # solve() is often more robust than root_find() for energy residuals
    sol = optx.least_squares(energy_residual, solver, y0=v_guess, throw=False)

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
