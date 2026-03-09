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
) -> np.ndarray:
    # Shortest path between x and y through data manifold using k-NN graph
    paths = compute_knn_initial_paths(x, np.vstack([y[None, :], X]), manifold, N_points)
    return paths[0]


def compute_knn_initial_paths(
    x: Union[np.ndarray, jax.Array],
    X: Union[np.ndarray, jax.Array],
    manifold: "RiemannianManifold",
    N_points: int = 20,
) -> np.ndarray:
    """
    Computes the shortest paths from x to all points in X using the manifold graph.

    In order to do this, we
    1) Compute the Riemannian distance from x to its k nearest neighbors in the
    manifold (x_to_data).
    2) Combine the precomputed distance graph for X (manifold.riemannian_graph) with
    the x_to_data graph.
    3) Use the combined graph to compute the shortest paths from x to all points in X.
    4) Reconstruct the paths for all targets in X.
    5) Remove points that are too close to each other, as this causes numerical
    stability issues when computing the geodesics.
    6) Interpolate the paths and sample N_points uniformly along them.

    Args:
        x: The base point from which to compute the paths.
        X: The target points to which to compute the paths.
        manifold: The Riemannian manifold on which to compute the paths.
        N_points: The number of points to sample along each path.

    Returns:
        np.ndarray: The shortest paths from x to all points in X.
    """
    # Compute weighted edges from x to its manifold neighbors
    X_data_np = np.array(manifold.X_data)
    dists, indices = manifold.nn_tree.kneighbors(x[None, :])
    dists, indices = dists[0], indices[0]

    M_diag_x = np.array(manifold.metric_diag(jnp.array(x)))
    M_diags_neigh = manifold.M_diags[indices]
    diffs = X_data_np[indices] - x[None, :]
    w_x = manifold.compute_edge_weights(diffs, M_diag_x, M_diags_neigh)

    # Construct combined graph with x
    x_to_data = csr_matrix(
        (w_x, (np.zeros(manifold.n_neighbors, int), indices)),
        shape=(1, X_data_np.shape[0]),
    )
    combined_graph = vstack(
        [
            hstack([csr_matrix((1, 1)), x_to_data]),
            hstack([x_to_data.T, manifold.riemannian_graph]),
        ]
    ).tocsr()

    # Compute shortest paths from x to all points in X
    _, predecessors = shortest_path(
        csgraph=combined_graph, directed=False, indices=0, return_predecessors=True
    )
    nodes, X_in = np.vstack([x, X_data_np]), np.array(X)

    if X_in.shape == X_data_np.shape and np.allclose(X_in, X_data_np):
        targets = np.arange(1, X_data_np.shape[0] + 1)
    else:
        # Find the closest point in the manifold for each point in X and use that as target
        targets = (
            manifold.nn_tree.kneighbors(X_in, 1, return_distance=False).flatten() + 1
        )

    # Reconstruct paths for all targets in X
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
            # When the target point is not in the manifold, we append it to the path
            raw_path = np.vstack([raw_path, X_in[idx]])

        if len(raw_path) > 1:
            # Remove points that are too close to each other
            keep = np.insert(
                np.linalg.norm(np.diff(raw_path, axis=0), axis=1) > 1e-8, 0, True
            )
            while not keep.all() and len(raw_path) > 1:
                raw_path = raw_path[keep]
                keep = np.insert(
                    np.linalg.norm(np.diff(raw_path, axis=0), axis=1) > 1e-8, 0, True
                )

        if len(raw_path) <= 1:
            uniform_path = np.tile(raw_path[0], (N_points, 1))
        else:
            # Construct a linear interpolation of the path and sample N_points along it
            diff_p = np.diff(raw_path, axis=0)
            cum_len = np.insert(np.cumsum(np.linalg.norm(diff_p, axis=1)), 0, 0.0)
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
        """
        Initializes the Riemannian manifold.

        Args:
            X_data: The data points on which to compute the manifold.
            sigma: The standard deviation of the Gaussian kernel used for the metric.
            rho: The regularization parameter for the metric.
            K_segments: The number of segments to use when computing the log maps.
            n_neighbors: The number of neighbors to use when computing the Riemannian graph.
        """
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

        self.M_diags = np.array(jax.vmap(self.metric_diag)(X_data))
        coo = self.adj_graph.tocoo()
        row, col = coo.row, coo.col
        diffs = X_np[col] - X_np[row]
        w = self.compute_edge_weights(diffs, self.M_diags[row], self.M_diags[col])
        self.riemannian_graph = csr_matrix((w, (row, col)), shape=self.adj_graph.shape)

    def compute_edge_weights(
        self, diffs: np.ndarray, M_i: np.ndarray, M_j: np.ndarray
    ) -> np.ndarray:
        """
        Computes the Riemannian edge weights between two sets of points X_i and X_j.

        Args:
            diffs: The differences between X_i and X_j.
            M_i: The metric diagonal of X_i.
            M_j: The metric diagonal of X_j.

        Returns:
            np.ndarray: The approximated geodesic distances between the point pairs.
        """
        return np.sqrt(
            np.clip(
                0.5 * np.sum((M_i + M_j) * diffs**2, axis=1),
                0,
                None,
            )
        )

    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Dict[str, Any]]:
        """
        JAX tree flattening. Required for JAX pytree registration.
        """
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
        """
        JAX tree unflattening. Required for JAX pytree registration.
        """
        return cls(*children, **aux_data)

    def metric_diag(self, x: jax.Array) -> jax.Array:
        """
        Computes the metric diagonal for a given point x.

        Args:
            x: The point for which to compute the metric diagonal.

        Returns:
            jax.Array: The metric diagonal for the point x.
        """
        diff = self.X_data - x[None, :]
        weights = jnp.exp(-jnp.sum(diff**2, axis=-1) / (2.0 * self.sigma**2))
        return 1.0 / (jnp.sum(weights[:, None] * diff**2, axis=0) + self.rho)

    def metric(self, x: jax.Array) -> jax.Array:
        """
        Computes the metric for a given point x.

        Args:
            x: The point for which to compute the metric.

        Returns:
            jax.Array: The metric for the point x.
        """
        return jnp.diag(self.metric_diag(x))

    def _local_speed(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Computes the local magnitude (speed) of a vector v
        at point x using the Riemannian metric.

        Args:
            x: The point for which to compute the local speed.
            v: The tangent vector at point x.

        Returns:
            jax.Array: The magnitude of vector v.
        """
        return jnp.sqrt(jnp.sum(self.metric_diag(x) * v**2))

    def curve_length(self, x: jax.Array, v: jax.Array, n_steps: int = 50) -> jax.Array:
        """
        Computes the length of a geodesic.

        Args:
            x: The initial point of the geodesic.
            v: The initial velocity of the geodesic.
            n_steps: The number of steps to use for the quadrature.

        Returns:
            jax.Array: The length of the geodesic.
        """
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
        """
        Evaluates the geodesic differential equation to find
        the acceleration at point x with velocity v.

        Args:
            x: The point for which to compute the geodesic acceleration.
            v: The velocity at point x.

        Returns:
            jax.Array: The acceleration at point x with velocity v.
        """
        M_inv = jnp.linalg.inv(self.metric(x))
        grad_L = jax.grad(lambda p: 0.5 * jnp.dot(v, jnp.dot(self.metric(p), v)))(x)
        dot_M_v = jax.jacfwd(lambda p: jnp.dot(self.metric(p), v))(x) @ v
        return M_inv @ (grad_L - dot_M_v)

    def _vector_field(self, t: float, y: jax.Array, args: Any) -> jax.Array:
        """
        Defines the vector field for the ODE solver to integrate geodesics.

        Args:
            t: The time parameter (it is not used but required by diffrax).
            y: The state vector, in this case the position and velocity.
            args: Additional arguments.

        Returns:
            jax.Array: The vector field at time t and state y.
        """
        d = y.shape[0] // 2
        return jnp.concatenate([y[d:], self._geodesic_ode(y[:d], y[d:])])

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Computes the exponential map Exp_x(v).

        Args:
            x: The point for which to compute the exponential map.
            v: The velocity at point x.

        Returns:
            jax.Array: The exponential map at point x with velocity v.
        """
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
        """
        Computes the log map Log_x(y) using a shooting method in which we
        partition the path into segments and solve the ODE for each segment.

        Args:
            x: The initial point of the geodesic.
            y: The final point of the geodesic.
            initial_path: The initial path to use for the shooting.
            scaled: Whether to scale the solution to fit Euclidean space.

        Returns:
            jax.Array: The velocity at point x to reach point y.
        """
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
        """
        Batched computation of the log map.

        Args:
            mu: The initial point of the geodesics.
            X_targets: The final points of the geodesics.
            initial_paths: The initial paths to use for the shooting.
            scaled: Whether to scale the solution to fit Euclidean space.

        Returns:
            jax.Array: The log map for the batch of points.
        """
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
        """
        Estimates the normalization constant for a distribution
        on the manifold using Monte Carlo integration.

        Args:
            mu: The mean of the distribution.
            sigma: The covariance matrix of the distribution.
            key: The random key for the Monte Carlo integration.
            n_samples: The number of samples to use for the Monte Carlo integration.

        Returns:
            jax.Array, jax.Array: The estimated normalization constant and the samples.
        """
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
