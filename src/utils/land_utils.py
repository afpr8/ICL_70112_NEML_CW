import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import diffrax
import optimistix as optx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.interpolate import interp1d
import numpy as np
from typing import Tuple, Any, Dict, Union


def compute_knn_initial_path(
    x: Union[np.ndarray, jax.Array],
    y: Union[np.ndarray, jax.Array],
    X: Union[np.ndarray, jax.Array],
    manifold: "RiemannianManifold",
    N_points: int = 20,
    n_neighbors: int = 5,
) -> np.ndarray:
    # Shortest path through data manifold using k-NN graph
    paths = compute_knn_initial_paths(
        x, np.vstack([y[None, :], X]), manifold, N_points, n_neighbors
    )
    return paths[0]


def compute_knn_initial_paths(
    x: Union[np.ndarray, jax.Array],
    X: Union[np.ndarray, jax.Array],
    manifold: "RiemannianManifold",
    N_points: int = 20,
    n_neighbors: int = 5,
) -> np.ndarray:
    # Shortest paths from x to all points in X using manifold graph
    X_data_np = np.array(manifold.X_data)
    dists, indices = manifold.nn_tree.kneighbors(x[None, :])
    dists, indices = dists[0], indices[0]

    # Weighted edges from x to its manifold neighbors
    M_diag_x = np.array(manifold.metric_diag(jnp.array(x)))
    M_diags_neigh = np.array(jax.vmap(manifold.metric_diag)(manifold.X_data[indices]))
    diffs = X_data_np[indices] - x[None, :]
    w_x = np.sqrt(
        np.clip(
            0.5
            * (
                np.sum(M_diag_x * diffs**2, axis=1)
                + np.sum(M_diags_neigh * diffs**2, axis=1)
            ),
            0,
            None,
        )
    )

    # Construct combined graph with x (index 0)
    x_to_data = csr_matrix(
        (w_x, (np.zeros(n_neighbors, int), indices)), shape=(1, X_data_np.shape[0])
    )
    combined_graph = vstack(
        [
            hstack([csr_matrix((1, 1)), x_to_data]),
            hstack([x_to_data.T, manifold.riemannian_graph]),
        ]
    ).tocsr()

    _, predecessors = shortest_path(
        csgraph=combined_graph, directed=False, indices=0, return_predecessors=True
    )
    nodes, X_in = np.vstack([x, X_data_np]), np.array(X)

    # Reconstruct paths for all targets in X
    if X_in.shape == X_data_np.shape and np.allclose(X_in, X_data_np):
        targets = np.arange(1, X_data_np.shape[0] + 1)
    else:
        targets = (
            manifold.nn_tree.kneighbors(X_in, 1, return_distance=False).flatten() + 1
        )

    all_paths = []
    for idx, i in enumerate(targets):
        path_idx = []
        curr = i
        while curr != -9999 and curr != 0:
            path_idx.append(curr)
            curr = predecessors[curr]
        path_idx.append(0)
        path_idx.reverse()

        raw_path = nodes[path_idx]
        if not np.allclose(raw_path[-1], X_in[idx], atol=1e-8):
            raw_path = np.vstack([raw_path, X_in[idx]])

        if len(raw_path) > 1:
            keep = np.insert(
                np.linalg.norm(np.diff(raw_path, axis=0), axis=1) > 1e-8, 0, True
            )
            raw_path = raw_path[keep]

        if len(raw_path) < 2:
            uniform_path = np.tile(raw_path[0], (N_points, 1))
        else:
            diff_p = np.diff(raw_path, axis=0)
            cum_len = np.insert(np.cumsum(np.linalg.norm(diff_p, axis=1)), 0, 0.0)
            if cum_len[-1] == 0:
                uniform_path = raw_path[0] + np.linspace(0, 1, N_points)[:, None] * (
                    raw_path[-1] - raw_path[0]
                )
            else:
                uniform_path = interp1d(cum_len / cum_len[-1], raw_path, axis=0)(
                    np.linspace(0, 1, N_points)
                )

        all_paths.append(uniform_path)
    return np.array(all_paths)


@jtu.register_pytree_node_class
class RiemannianManifold:
    """
    Encapsulates the mathematics of a Riemannian manifold defined by a
    Gaussian kernel density estimate over a set of data points X_data.
    """

    def __init__(
        self,
        X_data: Union[np.ndarray, jax.Array],
        sigma: float = 1.0,
        rho: float = 1e-3,
        K_segments: int = 5,
        n_neighbors: int = 5,
    ) -> None:
        self.X_data, self.sigma, self.rho, self.K_segments, self.n_neighbors = (
            X_data,
            sigma,
            rho,
            K_segments,
            n_neighbors,
        )
        X_np = np.array(X_data)

        # Cache NearestNeighbors tree and precompute Riemannian graph
        self.nn_tree = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn_tree.fit(X_np)
        self.adj_graph = self.nn_tree.kneighbors_graph(X_np, mode="connectivity")

        M_diags = np.array(jax.vmap(self.metric_diag)(X_data))
        coo = self.adj_graph.tocoo()
        row, col = coo.row, coo.col
        diffs = X_np[col] - X_np[row]
        w = np.sqrt(
            np.clip(
                0.5
                * (
                    np.sum(M_diags[row] * diffs**2, axis=1)
                    + np.sum(M_diags[col] * diffs**2, axis=1)
                ),
                0,
                None,
            )
        )
        self.riemannian_graph = csr_matrix((w, (row, col)), shape=self.adj_graph.shape)

    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Dict[str, Any]]:
        return (
            (self.X_data,),
            {
                "sigma": self.sigma,
                "rho": self.rho,
                "K_segments": self.K_segments,
                "n_neighbors": self.n_neighbors,
            },
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple[jax.Array]
    ) -> "RiemannianManifold":
        return cls(*children, **aux_data)

    def metric_diag(self, x: jax.Array) -> jax.Array:
        diff = self.X_data - x[None, :]
        weights = jnp.exp(-jnp.sum(diff**2, axis=-1) / (2.0 * self.sigma**2))
        return 1.0 / (jnp.sum(weights[:, None] * diff**2, axis=0) + self.rho)

    def metric(self, x: jax.Array) -> jax.Array:
        return jnp.diag(self.metric_diag(x))

    def _local_speed(self, x: jax.Array, v: jax.Array) -> jax.Array:
        return jnp.sqrt(jnp.sum(self.metric_diag(x) * v**2))

    def curve_length(self, x: jax.Array, v: jax.Array, n_steps: int = 50) -> jax.Array:
        # Quadrature-based Riemannian length of a geodesic
        ts = jnp.linspace(0.0, 1.0, n_steps)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._vector_field),
            diffrax.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=0.1,
            y0=jnp.concatenate([x, v]),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jnp.mean(
            jax.vmap(self._local_speed)(
                sol.ys[:, : x.shape[0]], sol.ys[:, x.shape[0] :]
            )
        )

    def _geodesic_ode(self, x: jax.Array, v: jax.Array) -> jax.Array:
        M_inv = jnp.linalg.inv(self.metric(x))
        grad_L = jax.grad(lambda p: 0.5 * jnp.dot(v, jnp.dot(self.metric(p), v)))(x)
        dot_M_v = jax.jacfwd(lambda p: jnp.dot(self.metric(p), v))(x) @ v
        return M_inv @ (grad_L - dot_M_v)

    def _vector_field(self, t: float, y: jax.Array, args: Any) -> jax.Array:
        d = y.shape[0] // 2
        return jnp.concatenate([y[d:], self._geodesic_ode(y[:d], y[d:])])

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self._vector_field),
            diffrax.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=0.1,
            y0=jnp.concatenate([x, v]),
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(1e-2, 1e-2),
            adjoint=diffrax.DirectAdjoint(),
        )
        return sol.ys[0, : x.shape[0]]

    def log_map_shooting(
        self,
        x: jax.Array,
        y: jax.Array,
        initial_path: jax.Array,
        scaled: bool = True,
    ) -> jax.Array:
        D, K = x.shape[0], self.K_segments
        dt = 1.0 / K
        y0 = jnp.concatenate(
            [
                (initial_path[1:-1]).flatten(),
                (jnp.diff(initial_path, axis=0) / dt).flatten(),
            ]
        )

        def residual_fn(vars_flat, args):
            x_opt, v_opt = (
                vars_flat[: (K - 1) * D].reshape((K - 1, D)),
                vars_flat[(K - 1) * D :].reshape((K, D)),
            )
            x_k = jnp.vstack([x, x_opt])

            def integrate(xk, vk):
                return diffrax.diffeqsolve(
                    diffrax.ODETerm(self._vector_field),
                    diffrax.Tsit5(),
                    t0=0.0,
                    t1=dt,
                    dt0=dt / 2,
                    y0=jnp.concatenate([xk, vk]),
                    saveat=diffrax.SaveAt(t1=True),
                    stepsize_controller=diffrax.PIDController(1e-5, 1e-5),
                    adjoint=diffrax.DirectAdjoint(),
                ).ys[0]

            pred = jax.vmap(integrate)(x_k, v_opt)
            return jnp.concatenate(
                [
                    (pred[:, :D] - jnp.vstack([x_opt, y])).flatten(),
                    (pred[:-1, D:] - v_opt[1:]).flatten(),
                ]
            )

        sol = optx.root_find(
            residual_fn,
            optx.LevenbergMarquardt(1e-5, 1e-5),
            y0=y0,
            max_steps=1000,
            throw=False,
        )
        v0 = sol.value[(K - 1) * D :].reshape((K, D))[0]
        if scaled:
            # Scale v0 to have equal length in Euclidean space as in the manifold
            return v0 * (self.curve_length(x, v0) / (jnp.linalg.norm(v0) + 1e-12))
        return v0

    def log_map_batch(
        self,
        mu: jax.Array,
        X_targets: jax.Array,
        initial_paths: jax.Array,
        scaled: bool = True,
    ) -> jax.Array:
        return jax.vmap(self.log_map_shooting, in_axes=(None, 0, 0, None))(
            mu, X_targets, initial_paths, scaled
        )

    def compute_normalization_constant(
        self,
        mu: jax.Array,
        sigma: jax.Array,
        key: jax.Array,
        n_samples: int = 3000,
    ) -> Tuple[jax.Array, jax.Array]:
        d = mu.shape[0]
        v_samples = jax.random.multivariate_normal(
            key, jnp.zeros(d), sigma, (n_samples,)
        )

        def vol(v):
            return jnp.exp(
                0.5 * jnp.sum(jnp.log(jnp.diag(self.metric(self.exp_map(mu, v)))))
            )

        Z = jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(sigma))
        return Z * jnp.mean(jax.vmap(vol)(v_samples)), v_samples
