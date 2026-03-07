import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import diffrax
import optimistix as optx
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import interp1d
import numpy as np


def compute_knn_initial_path(
    x: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    N_points: int = 20,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Computes a shortest path through the data manifold using a k-NN graph.
    Runs on standard NumPy/SciPy (not JIT compiled).
    """
    nodes = np.vstack([x, y, X])
    graph = kneighbors_graph(nodes, n_neighbors=n_neighbors, mode="distance")
    dist_matrix, predecessors = shortest_path(
        csgraph=graph, directed=False, indices=0, return_predecessors=True
    )

    path_indices = []
    current_node = 1
    while current_node != -9999 and current_node != 0:
        path_indices.append(current_node)
        current_node = predecessors[current_node]
    path_indices.append(0)
    path_indices.reverse()

    raw_path = nodes[path_indices]

    # Remove consecutive duplicate points to avoid division by zero in interpolation
    diffs = np.diff(raw_path, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    keep = np.insert(distances > 1e-8, 0, True)
    raw_path = raw_path[keep]

    diffs = np.diff(raw_path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.insert(np.cumsum(segment_lengths), 0, 0.0)

    total_length = cumulative_length[-1]
    if total_length == 0:
        t = np.linspace(0, 1, N_points)[:, None]
        return x + t * (y - x)

    normalized_length = cumulative_length / total_length

    interpolator = interp1d(normalized_length, raw_path, axis=0, kind="linear")
    t_uniform = np.linspace(0, 1, N_points)
    uniform_path = interpolator(t_uniform)

    return uniform_path


@jtu.register_pytree_node_class
class RiemannianManifold:
    """
    Encapsulates the mathematics of a Riemannian manifold defined by a
    Gaussian kernel density estimate over a set of data points X_data.
    """

    def __init__(
        self,
        X_data: jnp.ndarray,
        sigma: float = 1.0,
        rho: float = 1e-3,
        K_segments: int = 5,
    ):
        self.X_data = X_data
        self.sigma = sigma
        self.rho = rho
        self.K_segments = K_segments

    def tree_flatten(self):
        return (
            (self.X_data,),
            {"sigma": self.sigma, "rho": self.rho, "K_segments": self.K_segments},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def metric(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the local Riemannian metric tensor at point x."""
        diff = self.X_data - x[None, :]
        dist2 = jnp.sum(diff**2, axis=-1)
        weights = jnp.exp(-dist2 / (2.0 * self.sigma**2))

        weighted_sq = jnp.sum(weights[:, None] * (diff**2), axis=0)
        diag_entries = weighted_sq + self.rho
        M_x = jnp.diag(1.0 / diag_entries)

        return M_x

    def _geodesic_ode(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        M_x = self.metric(x)
        M_inv = jnp.linalg.inv(M_x)

        def kinetic_energy(pos):
            return 0.5 * jnp.dot(v, jnp.dot(self.metric(pos), v))

        grad_L = jax.grad(kinetic_energy)(x)

        def Mv_fn(pos):
            return jnp.dot(self.metric(pos), v)

        dot_M_v = jax.jacfwd(Mv_fn)(x) @ v

        return M_inv @ (grad_L - dot_M_v)

    def _vector_field(self, t, y, args):
        d = y.shape[0] // 2
        x_pt, v_pt = y[:d], y[d:]
        a = self._geodesic_ode(x_pt, v_pt)
        return jnp.concatenate([v_pt, a])

    def exp_map(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Exponential map from x with initial velocity v."""
        term = diffrax.ODETerm(self._vector_field)
        solver = diffrax.Tsit5()
        y0 = jnp.concatenate([x, v])

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.1,
            y0=y0,
            args=None,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=1e-2, atol=1e-2),
            adjoint=diffrax.DirectAdjoint(),
        )
        d = x.shape[0]
        return sol.ys[0, :d]

    def log_map_shooting(
        self, x: jnp.ndarray, y: jnp.ndarray, initial_path: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Computes the log map using multiple shooting.
        initial_path: (K_segments + 1, D) array of points.
        """
        D = x.shape[0]
        K = self.K_segments
        dt = 1.0 / K

        # Extract the interior states that we will optimize over.
        v_guess = jnp.diff(initial_path, axis=0) / dt  # (K, D)
        x_guess = initial_path[1:-1]  # (K-1, D)

        y0 = jnp.concatenate([x_guess.flatten(), v_guess.flatten()])

        def residual_fn(vars_flat, args):
            x_opt = vars_flat[: (K - 1) * D].reshape((K - 1, D))
            v_opt = vars_flat[(K - 1) * D :].reshape((K, D))

            # Reconstruct the full trajectory of initial points
            x_k = jnp.vstack([x, x_opt])  # (K, D)

            def integrate_segment(xk, vk):
                term = diffrax.ODETerm(self._vector_field)
                solver = diffrax.Tsit5()
                state0 = jnp.concatenate([xk, vk])

                sol = diffrax.diffeqsolve(
                    term,
                    solver,
                    t0=0.0,
                    t1=dt,
                    dt0=dt / 2,
                    y0=state0,
                    args=None,
                    saveat=diffrax.SaveAt(t1=True),
                    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
                    adjoint=diffrax.DirectAdjoint(),
                )
                return sol.ys[0]  # Return the full state (2*D)

            # Predict end states of each segment
            state_end_pred = jax.vmap(integrate_segment)(x_k, v_opt)  # (K, 2D)
            x_end_pred = state_end_pred[:, :D]
            v_end_pred = state_end_pred[:, D:]

            x_target = jnp.vstack([x_opt, y])  # (K, D)

            # For interior points, position and velocity must match. For the final point, only position matches y.
            x_residuals = x_end_pred - x_target  # (K, D)
            v_residuals = v_end_pred[:-1] - v_opt[1:]  # (K-1, D)

            return jnp.concatenate([x_residuals.flatten(), v_residuals.flatten()])

        solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5)
        sol = optx.root_find(
            residual_fn, solver, y0=y0, args=None, max_steps=1000, throw=False
        )

        opt_vars = sol.value
        v_opt = opt_vars[(K - 1) * D :].reshape((K, D))

        return v_opt[0]

    def log_map_batch(
        self, mu: jnp.ndarray, X_targets: jnp.ndarray, initial_paths: jnp.ndarray
    ) -> jnp.ndarray:
        return jax.vmap(self.log_map_shooting, in_axes=(None, 0, 0))(
            mu, X_targets, initial_paths
        )

    def compute_normalization_constant(
        self,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        key: jax.Array,
        n_samples: int = 3000,
    ) -> jnp.ndarray:
        d = mu.shape[0]
        Z = jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(sigma))

        v_samples = jax.random.multivariate_normal(
            key, mean=jnp.zeros(d), cov=sigma, shape=(n_samples,)
        )

        def compute_vol(v):
            x = self.exp_map(mu, v)
            M_x = self.metric(x)
            log_det = jnp.sum(jnp.log(jnp.diag(M_x)))
            return jnp.exp(0.5 * log_det)

        vols = jax.vmap(compute_vol)(v_samples)
        return Z * jnp.mean(vols)
