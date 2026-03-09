import jax
import jax.numpy as jnp
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass


from src.utils.land_utils import RiemannianManifold, compute_knn_initial_paths

@dataclass
class State:
    """
    A dataclass to hold the state of the model parameters for potential reversion during training.
    """
    mu: list[jnp.ndarray]
    A: list[jnp.ndarray]
    sigma: list[jnp.ndarray]
    pi: jnp.ndarray
    C: list[jnp.ndarray]
    Vs: list[jnp.ndarray]


class LANDMixtureModel:
    def __init__(
        self,
        K: int = 3,
        lr_mu: float = 1e-3,
        lr_A: float = 1e-3,
        S: int = 100,
        lr_scale_down: float = 0.75,  # 0.75 as in the original LAND paper
        lr_scale_up: float = 1.1,  # 1.1 as in the original LAND paper
        epsilon: float = 1e-3,
        patience: int = 5,
        sigma: float = 1.0,
        rho: float = 1e-3,
        K_segments: int = 5,
        n_neighbors: int = 5,
        init_method: str = "GMM",
        seed: int = 42,
    ):
        """
        Initialise the LAND Mixture Model (Algorithm 4)
        Params:
            K (int): The number of mixture components
            lr_mu (float): The learning rate for mu
            lr_A (float): The learning rate for A
            S (int): The number of vectors sampled to estimate the exp_map part of the gradient
            epsilon (float): The tolerance for determining significant improvement
            patience (int): The number of epochs to wait for an improvement before stopping
            sigma (float): Hyperparameter to compute the metric
            rho (float): Hyperparameter to compute the metric
            K_segments (int): The number of segments to use for the Riemannian manifold
            n_neighbors (int): The number of neighbours for KNN path initialisation
            init_method (str): Method to initialise params ("random", "mean", "GMM")
            seed (int): The PRNG seed used for jax RNG initialisation.
        """
        self.K = K
        self.lr_mu = lr_mu
        self.lr_A = lr_A
        self.S = S
        self.epsilon = epsilon
        self.patience = patience
        self.sigma = sigma
        self.rho = rho
        self.K_segments = K_segments
        self.n_neighbors = n_neighbors

        self.lr_scale_up = lr_scale_up
        self.lr_scale_down = lr_scale_down

        self.init_method = init_method
        self.key = jax.random.key(seed)

    def fit(
        self, X: jnp.ndarray
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray], jnp.ndarray]:
        """
        Fit the LAND mixture model to the data using Expectation-Maximisation
        Params:
            X (jnp.ndarray): The data to fit the model to, shape (N, D)
        Returns:
            mu (list[jnp.ndarray]): The means of the K distributions
            sigma (list[jnp.ndarray]): The covariances of the K distributions
            C (list[jnp.ndarray]): The normalisation constants
            pi (jnp.ndarray): The mixing weights
        """
        manifold = RiemannianManifold(
            X, self.sigma, self.rho, self.K_segments, self.n_neighbors
        )
        N = X.shape[0]

        print("Initializing parameters...")
        self.key, subkey = jax.random.split(self.key)
        mu, A, sigma = self._init_params(
            X, key=subkey, method=self.init_method, manifold=manifold
        )
        pi = jnp.ones(self.K) / self.K

        self.key, subkey = jax.random.split(self.key)
        C = []
        Vs = []
        for k in range(self.K):
            self.key, subkey = jax.random.split(self.key)
            c_val, v_s = manifold.compute_normalization_constant(
                mu[k], sigma[k], subkey, n_samples=self.S
            )
            C.append(c_val)
            Vs.append(v_s)

        t = 0
        loss_diff = float("inf")
        prev_loss = float("inf")
        current_loss = float("inf")
        n_wo_improvement = 0

        with tqdm(desc="Mixture Model EM", unit="epoch") as pbar:
            while n_wo_improvement < self.patience:
                r = jnp.zeros((N, self.K))
                log_maps_all = []
                inv_sigmas = []

                prevState = State(mu, A, sigma, pi, C, Vs)

                # E-step: compute responsibilities
                for k in range(self.K):
                    inv_sigma = jnp.linalg.inv(sigma[k])
                    inv_sigmas.append(inv_sigma)

                    log_maps = self._compute_log_maps(mu[k], X, manifold)
                    log_maps_all.append(log_maps)

                    dist_sq = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)

                    # Mask out points that are too far
                    dist_sq = jnp.where(dist_sq > 0.3, 0, dist_sq)

                    # p_M(x_n | mu_k, Sigma_k)
                    p_x = (1.0 / C[k]) * jnp.exp(-0.5 * dist_sq)
                    r = r.at[:, k].set(pi[k] * p_x)

                # Normalise responsibilities across components for each point
                r_sum = r.sum(axis=1, keepdims=True)
                r_sum = jnp.clip(r_sum, a_min=1e-12)
                r = r / r_sum

                # Calculate current negative log-likelihood to monitor convergence
                current_loss = -jnp.sum(jnp.log(r_sum)) / N

                loss_diff = current_loss - prev_loss

                # If the loss increased, revert to previous parameters and reduce learning rate
                if loss_diff > 0:  
                    mu, A, sigma, pi, C, Vs = prevState.mu, prevState.A, prevState.sigma, prevState.pi, prevState.C, prevState.Vs
                    self.lr_A *= self.lr_scale_down
                    loss_diff = 0.0 
                else:
                    self.lr_A *= self.lr_scale_up

                    # If the loss did not decrease significantly (or increased), increment counter
                    if loss_diff <= self.epsilon:
                        n_wo_improvement += 1
                    else:
                        n_wo_improvement = 0

                prev_loss = current_loss

                pbar.set_postfix(
                    loss_diff=float(loss_diff),
                    loss=float(current_loss),
                    no_impr=int(n_wo_improvement),
                )
                pbar.update(1)

                # M-step: update parameters for each component
                for k in range(self.K):
                    N_k = r[:, k].sum()

                    # Compute both gradients sharing MC samples
                    self.key, subkey = jax.random.split(self.key)
                    grad_mu, grad_sigma = self._compute_grads_k(
                        mu[k],
                        A[k],
                        sigma[k],
                        C[k],
                        Vs[k],
                        r[:, k],
                        N_k,
                        subkey,
                        log_maps_all[k],
                        manifold,
                    )

                    # update mu
                    new_mu_k = manifold.exp_map(mu[k], self.lr_mu * grad_mu)
                    mu = mu.at[k].set(new_mu_k)

                    # update A
                    new_A_k = A[k] - (self.lr_A * grad_sigma)
                    A = A.at[k].set(new_A_k)

                    # update Sigma
                    new_sigma_k = jnp.linalg.inv(new_A_k.T @ new_A_k)
                    sigma = sigma.at[k].set(new_sigma_k)

                    # update pi
                    pi = pi.at[k].set(N_k / N)

                    # Update normalization constant
                    self.key, subkey = jax.random.split(self.key)
                    c_val, v_s = manifold.compute_normalization_constant(
                        new_mu_k, new_sigma_k, subkey, n_samples=self.S
                    )
                    C[k] = c_val
                    Vs[k] = v_s

                t += 1

        return mu, sigma, C, pi

    def _compute_log_maps(
        self, mu: jnp.ndarray, X: jnp.ndarray, manifold: RiemannianManifold
    ) -> jnp.ndarray:
        m_np = np.array(mu)
        X_np = np.array(X)
        paths = compute_knn_initial_paths(
            m_np, X_np, manifold, N_points=self.K_segments + 1
        )
        return manifold.log_map_batch(mu, X, jnp.array(paths))

    def _init_params(
        self, X: jnp.ndarray, key: jax.Array, method: str, manifold: RiemannianManifold
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Initialise the parameters of the mixture model.
        Params:
            X (jnp.ndarray): The data to initialise the parameters with
            key (jax.Array): Setup key for random generations
            method (str): The method to use for initialisation.
                - "random": Initialise mu by randomly selecting data points.
                - "mean": Initialise mu near the empirical mean with slight noise.
                - "GMM": Initialise mu with a Euclidean Gaussian Mixture Model.
        Returns:
            mu (jnp.ndarray): The means of the distributions, shape (K, D)
            A (jnp.ndarray): The A matrices of the distributions, shape (K, D, D)
            sigma (jnp.ndarray): The covariances of the distributions, shape (K, D, D)
        """
        N = X.shape[0]

        if method == "random":
            # Randomly select initial means from data points
            indices = jax.random.permutation(key, jnp.arange(N))[: self.K]
            mu = [X[idx].squeeze() for idx in indices]

        elif method == "GMM":
            # Fit standard Euclidean GMM for a warm start
            gmm = GaussianMixture(
                n_components=self.K, covariance_type="full", random_state=42
            )

            gmm.fit(np.array(X))
            mu = []
            for m in gmm.means_:
                m_tensor = jnp.array(m, dtype=jnp.float32)
                # Find the data point closest to this specific GMM mean
                closest_idx = jnp.argmin(jnp.sum((X - m_tensor) ** 2, axis=1))
                mu.append(X[closest_idx])

        elif method == "mean":
            sorted_data = jnp.sort(X, axis=0)
            k_clusters = jnp.array_split(sorted_data, self.K)
            means = [jnp.mean(cluster, axis=0) for cluster in k_clusters]

            # Find the data point closest to the euclidean means to ensure we start on the manifold
            closest_idxs = [
                jnp.argmin(jnp.sum((X - means[k]) ** 2, axis=1)) for k in range(self.K)
            ]
            mu = [X[idx].squeeze() for idx in closest_idxs]

        else:
            raise ValueError(f"Invalid initialisation method: {method}")

        mu = jnp.stack(mu)

        # Precompute log maps and Riemannian distances for all K components
        all_tangent_vectors = []
        all_dists = []
        for k in range(self.K):
            tangent_vectors = self._compute_log_maps(mu[k], X, manifold)
            all_tangent_vectors.append(tangent_vectors)

            dist_sq = jnp.sum(tangent_vectors**2, axis=-1)
            all_dists.append(dist_sq)
        dists = jnp.stack(all_dists, axis=1)

        # Assign each data point to the closest mean based on Riemannian distance
        assignments = jnp.argmin(dists, axis=-1)

        # Compute covariances using only the assigned points
        A = []
        sigma = []
        for k in range(self.K):
            tangent_vectors = all_tangent_vectors[k]

            # 1.0 if datapoint is part of the cluster, 0 otherwise
            weights = (assignments == k).astype(jnp.float32) + 1e-12
            sig = jnp.cov(tangent_vectors.T, aweights=weights, bias=True)

            # Add a tiny ridge to the diagonal to ensure positive definiteness
            sig += jnp.eye(sig.shape[0]) * 1e-6

            sigma.append(sig)
            A.append(self.compute_A(sig))

        return mu, jnp.stack(A), jnp.stack(sigma)

    def _compute_grads_k(
        self,
        mu: jnp.ndarray,
        A: jnp.ndarray,
        sigma: jnp.ndarray,
        normalization_constant: jnp.ndarray,
        v_samples: jnp.ndarray,
        r_k: jnp.ndarray,
        N_k: jnp.ndarray,
        key: jax.Array,
        log_maps: jnp.ndarray,
        manifold: RiemannianManifold,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute both the mu and sigma (A) gradients for a single component,
        sharing MC samples and metric deformation evaluations.
        Params:
            mu (jnp.ndarray): Local component mean
            A (jnp.ndarray): Local component precision factor (A.T @ A = inv(sigma))
            sigma (jnp.ndarray): Local component covariance
            normalization_constant (jnp.ndarray): Normalisation term evaluated at mu, sigma
            v_samples (jnp.ndarray): Samples used for the computation of the normalization constant
            r_k (jnp.ndarray): Responsibility of this component for each point
            N_k (jnp.ndarray): Sum of responsibilities for this component
            key (jax.Array): Random generation key
            log_maps (jnp.ndarray): Precomputed evaluations of log_map_shooting, shape (N, D)
        Returns:
            grad_mu (jnp.ndarray): Gradient for mu
            grad_A (jnp.ndarray): Gradient for A
        """
        # Gradient for mu: responsibility-weighted mean in tangent space
        grad_mu_data = jnp.sum(r_k[:, None] * log_maps, axis=0) / N_k
        # Gradient for sigma: responsibility-weighted outer product in tangent space
        grad_sigma_data = ((log_maps * r_k[:, None]).T @ log_maps) / N_k

        # MC estimate of normalisation integral
        d = mu.shape[0]
        mc_scale = jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(sigma)) / (
            self.S * normalization_constant
        )

        def compute_m(v):
            translated_point = manifold.exp_map(mu, v)
            M_trans = manifold.metric(translated_point)
            return jnp.exp(0.5 * jnp.sum(jnp.log(jnp.diag(M_trans))))

        m_values = jax.vmap(compute_m)(v_samples)

        # Compute gradients
        grad_mu_mc = -mc_scale * jnp.sum(m_values[:, None] * v_samples, axis=0)

        def weighted_outer(m_val, v):
            return m_val * jnp.outer(v, v)

        grad_sigma_mc = -mc_scale * jnp.sum(
            jax.vmap(weighted_outer)(m_values, v_samples), axis=0
        )

        grad_mu = grad_mu_data + grad_mu_mc
        grad_A = A @ (grad_sigma_data + grad_sigma_mc)

        return grad_mu, grad_A

    def compute_A(self, sigma: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the A matrix from the covariance matrix, A.T @ A = inv(sigma)
        """
        return jnp.linalg.cholesky(jnp.linalg.inv(sigma)).T
