import jax
import jax.numpy as jnp
from functools import partial
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from utils import (
    jax_exp_map,
    jax_log_map_shooting,
    compute_normalization_constant,
    jax_metric,
)


class LANDMixtureModel:
    def __init__(
        self,
        K: int = 3,
        lr_mu: float = 1e-3,
        lr_A: float = 1e-3,
        S: int = 100,
        epsilon: float = 1e-3,
        sigma: float = 1.0,
        rho: float = 1e-3,
        init_method: str = "mean",
        seed: int = 42,
    ):
        """
        Initialise the LAND Mixture Model (Algorithm 4)
        Params:
            K (int): The number of mixture components
            lr_mu (float): The learning rate for mu
            lr_A (float): The learning rate for A
            S (int): The number of vectors sampled to estimate the exp_map part of the gradient
            epsilon (float): The tolerance for the end condition
            sigma (float): Hyperparameter to compute the metric
            rho (float): Hyperparameter to compute the metric
            init_method (str): Method to initialize params ("random", "mean", "GMM")
            seed (int): The PRNG seed used for jax RNG initialization.
        """
        self.K = K
        self.lr_mu = lr_mu
        self.lr_A = lr_A
        self.S = S
        self.epsilon = epsilon
        self.sigma = sigma
        self.rho = rho
        self._metric = None

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
        self._metric = partial(jax_metric, X=X, sigma=self.sigma, rho=self.rho)
        N = X.shape[0]
        log_map_vmap = jax.vmap(jax_log_map_shooting, in_axes=(None, 0, None))

        # Initialise the parameters
        self.key, subkey = jax.random.split(self.key)
        mu, A, sigma = self._init_params(X, key=subkey, method=self.init_method)
        pi = jnp.ones(self.K) / self.K

        C = []
        for k in range(self.K):
            self.key, subkey = jax.random.split(self.key)
            C.append(
                compute_normalization_constant(
                    mu[k], sigma[k], self._metric, subkey, n_samples=self.S
                )
            )

        t = 0
        loss_diff = float("inf")
        prev_loss = float("inf")
        current_loss = float("inf")

        with tqdm(desc="Mixture Model EM", unit="epoch") as pbar:
            while loss_diff**2 > self.epsilon:
                r = jnp.zeros((N, self.K))
                log_maps_all = []
                inv_sigmas = []

                # E-step: compute responsibilities
                for k in range(self.K):
                    inv_sigma = jnp.linalg.inv(sigma[k])
                    inv_sigmas.append(inv_sigma)

                    log_maps = log_map_vmap(mu[k], X, self._metric)
                    log_maps_all.append(log_maps)

                    dist_sq = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)

                    # p_M(x_n | mu_k, Sigma_k)
                    p_x = (1.0 / C[k]) * jnp.exp(-0.5 * dist_sq)
                    r = r.at[:, k].set(pi[k] * p_x)

                # Normalise responsibilities across components for each point
                r_sum = r.sum(axis=1, keepdims=True)
                r_sum = jnp.clip(r_sum, a_min=1e-12)
                r = r / r_sum

                # Calculate current negative log-likelihood to monitor convergence
                current_loss = -jnp.sum(jnp.log(r_sum)) / N
                if t > 0:
                    loss_diff = current_loss - prev_loss
                prev_loss = current_loss

                pbar.set_postfix(loss_diff=float(loss_diff), loss=float(current_loss))
                pbar.update(1)

                if loss_diff**2 <= self.epsilon and t > 0:
                    break

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
                        r[:, k],
                        N_k,
                        subkey,
                        log_maps_all[k],
                    )

                    # update mu
                    mu[k] = jax_exp_map(mu[k], self.lr_mu * grad_mu, self._metric)

                    # estimate C_k using eq. 16
                    self.key, subkey = jax.random.split(self.key)
                    C[k] = compute_normalization_constant(
                        mu[k], sigma[k], self._metric, subkey, n_samples=self.S
                    )

                    # update A
                    A[k] -= self.lr_A * grad_sigma

                    # update Sigma
                    sigma[k] = jnp.linalg.inv(A[k].T @ A[k])

                    # update pi
                    pi = pi.at[k].set(N_k / N)

                t += 1

        self._metric = None  # avoid memory leak
        return mu, sigma, C, pi

    def _init_params(
        self, X: jnp.ndarray, key: jax.Array, method: str = "mean"
    ) -> tuple[list, list, list]:
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
            mu (list[jnp.ndarray]): The means of the distributions
            A (list[jnp.ndarray]): The A matrices of the distributions
            sigma (list[jnp.ndarray]): The covariances of the distributions
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
            import numpy as np

            gmm.fit(np.array(X))
            mu = [jnp.array(m, dtype=jnp.float32) for m in gmm.means_]

        elif method == "mean":
            # Calculate the global empirical mean
            global_mean = jnp.mean(X, axis=0)

            # Add small random noise to break symmetry.
            mu = []
            for _ in range(self.K):
                key, subkey = jax.random.split(key)
                mu.append(
                    global_mean + 0.1 * jax.random.normal(subkey, global_mean.shape)
                )

        else:
            raise ValueError(f"Invalid initialisation method: {method}")

        A = []
        sigma = []
        log_map_vmap = jax.vmap(jax_log_map_shooting, in_axes=(None, 0, None))

        for k in range(self.K):
            # Calculate initial covariance from tangent vectors mapped from the new mean
            tangent_vectors = log_map_vmap(mu[k], X, self._metric)
            sig = jnp.cov(tangent_vectors.T)

            # Add a tiny ridge to the diagonal to ensure positive definiteness
            sig += jnp.eye(sig.shape[0]) * 1e-6

            sigma.append(sig)
            A.append(self.compute_A(sig))

        return mu, A, sigma

    def _compute_grads_k(
        self,
        mu: jnp.ndarray,
        A: jnp.ndarray,
        sigma: jnp.ndarray,
        normalization_constant: jnp.ndarray,
        r_k: jnp.ndarray,
        N_k: jnp.ndarray,
        key: jax.Array,
        log_maps: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute both the mu and sigma (A) gradients for a single component,
        sharing MC samples and metric deformation evaluations.
        Params:
            mu (jnp.ndarray): Local component mean
            A (jnp.ndarray): Local component precision factor (A.T @ A = inv(sigma))
            sigma (jnp.ndarray): Local component covariance
            normalization_constant (jnp.ndarray): Normalisation term evaluated at mu, sigma
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

        # MC estimate of normalization integral
        d = mu.shape[0]
        v_samples = jax.random.multivariate_normal(
            key, jnp.zeros(d), sigma, shape=(self.S,)
        )
        mc_scale = jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(sigma)) / (
            self.S * normalization_constant
        )
        m_values = jax.vmap(lambda v: self._m(mu, v))(v_samples)

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

    def _m(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the deformation of the metric at mu in the direction of v.
        """
        translated_point = jax_exp_map(mu, v, self._metric)
        metric_translated_point = self._metric(translated_point)
        return jnp.sqrt(jnp.linalg.det(metric_translated_point))

    def compute_A(self, sigma: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the A matrix from the covariance matrix, A.T @ A = inv(sigma)
        """
        return jnp.linalg.cholesky(jnp.linalg.inv(sigma)).T
