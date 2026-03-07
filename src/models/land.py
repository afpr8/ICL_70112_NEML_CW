import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np

from src.utils.land_utils import RiemannianManifold, compute_knn_initial_path


class LANDMLE:
    """
    Locally Adaptive Normal Distribution (LAND) optimized via Maximum Likelihood Estimation (MLE).

    This class implements the fitting of a Locally Adaptive Normal Distribution to data on a
    Riemannian manifold. It uses Riemannian Gradient Descent to iteratively update the spatial
    mean (mu) and the covariance matrix (sigma) by maximizing the log-likelihood of the data.
    """

    def __init__(
        self,
        initial_lr_mu: float = 1e-3,
        initial_lr_A: float = 1e-3,
        S: int = 3000,  # 3000 as in the original LAND paper
        lr_scale_down: float = 0.75,  # 0.75 as in the original LAND paper
        lr_scale_up: float = 1.1,  # 1.1 as in the original LAND paper
        epsilon: float = 1e-3,
        sigma: float = 1.0,  # Set to 1.0 in the original paper and tested from 0.5 to 1.5
        rho: float = 1e-3,
        K_segments: int = 5,
        init_method: str = "mean",
        seed: int = 42,
    ):
        """
        Initialize the LAND model
        Params:
            initial_lr_mu (float): The initial learning rate for mu
            initial_lr_A (float): The initial learning rate for A
            lr_scale_down (float): The factor by which to scale down the learning rate when the loss increases
            lr_scale_up (float): The factor by which to scale up the learning rate when the loss decreases
            S (int): The number of vectors sampled to estimate the exp_map part of the gradient
            epsilon (float): The tolerance for the end condition
            sigma (float): Hyperparameter to compute the metric
            rho (float): Hyperparameter to compute the metric
            K_segments (int): The number of segments to use for the initial path
            init_method (str): The method to use for initialization.
                - "random": Initialize mu randomly, sigma from empirical cov of tangent vectors.
                - "mean": Initialize mu as the empirical mean, sigma from empirical cov of tangent vectors.
                - "GMM": Initialize mu and sigma with a Gaussian Mixture Model.
            seed (int): The PRNG seed used for jax RNG initialization.
        """
        self.lr_mu = initial_lr_mu
        self.lr_A = initial_lr_A
        self.lr_scale_down = lr_scale_down
        self.lr_scale_up = lr_scale_up
        self.S = S
        self.epsilon = epsilon
        self.sigma = sigma
        self.rho = rho
        self.K_segments = K_segments
        self.init_method = init_method
        self.key = jax.random.key(seed)

    def fit(self, X: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Fit the LAND model to the data
        Params:
            X (jnp.ndarray): The data to fit the model to, shape (N, D)
        Returns:
            mu (jnp.ndarray): The mean of the distribution
            sigma (jnp.ndarray): The covariance of the distribution
            normalization_constant (jnp.ndarray): The normalization constant of the distribution
        """
        manifold = RiemannianManifold(X, self.sigma, self.rho, self.K_segments)

        self.key, subkey = jax.random.split(self.key)
        mu, A, sigma = self._init_params(X, subkey, self.init_method, manifold)

        self.key, subkey = jax.random.split(self.key)
        norm_const = manifold.compute_normalization_constant(mu, sigma, subkey, self.S)
        loss_diff = float("inf")

        with tqdm(desc="LAND MLE Fit", unit="epoch") as pbar:
            while loss_diff**2 > self.epsilon:
                # Store previous values
                prev_sigma = sigma
                prev_norm_const = norm_const

                log_maps = self._compute_log_maps(mu, X, manifold)
                prev_loss = self._loss(sigma, log_maps, norm_const)

                # Update mu
                self.key, subkey = jax.random.split(self.key)
                grad_mu = self._compute_grad_mu(
                    mu, sigma, norm_const, subkey, log_maps, manifold
                )
                mu = manifold.exp_map(mu, self.lr_mu * grad_mu)

                self.key, subkey = jax.random.split(self.key)
                norm_const = manifold.compute_normalization_constant(
                    mu, sigma, subkey, self.S
                )

                # Scale lr_mu
                log_maps = self._compute_log_maps(mu, X, manifold)
                loss_diff = self._loss(sigma, log_maps, norm_const) - prev_loss
                if loss_diff > 0:
                    self.lr_mu *= self.lr_scale_down
                else:
                    self.lr_mu *= self.lr_scale_up

                # Update sigma
                self.key, subkey = jax.random.split(self.key)
                grad_sigma = self._compute_grad_sigma(
                    mu, A, sigma, norm_const, subkey, log_maps, manifold
                )
                A = A - self.lr_A * grad_sigma
                sigma = jnp.linalg.inv(A.T @ A)

                prev_norm_const = norm_const
                self.key, subkey = jax.random.split(self.key)
                norm_const = manifold.compute_normalization_constant(
                    mu, sigma, subkey, self.S
                )

                # Scale lr_A
                new_loss = self._loss(sigma, log_maps, norm_const)
                loss_diff_A = new_loss - self._loss(
                    prev_sigma, log_maps, prev_norm_const
                )
                if loss_diff_A > 0:
                    self.lr_A *= self.lr_scale_down
                else:
                    self.lr_A *= self.lr_scale_up

                loss_diff = new_loss - prev_loss

                pbar.set_postfix(loss_diff=float(loss_diff), loss=float(new_loss))
                pbar.update(1)

        return mu, sigma, norm_const

    def _compute_log_maps(
        self, mu: jnp.ndarray, X: jnp.ndarray, manifold: RiemannianManifold
    ) -> jnp.ndarray:
        m_np = np.array(mu)
        X_np = np.array(X)
        paths = [
            compute_knn_initial_path(m_np, X_np[i], X_np, N_points=self.K_segments + 1)
            for i in range(X_np.shape[0])
        ]
        return manifold.log_map_batch(mu, X, jnp.array(paths))

    def _init_params(
        self, X: jnp.ndarray, key: jax.Array, method: str, manifold: RiemannianManifold
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Initialize the parameters of the model
        Params:
            X (jnp.ndarray): The data to initialize the parameters with
            key (jax.Array): Random key for operations
            method (str): The method to use for initialization.
                - "random": Initialize mu randomly, sigma from empirical cov of tangent vectors.
                - "mean": Initialize mu as the empirical mean, sigma from empirical cov of tangent vectors.
                - "GMM": Initialize mu and sigma with a Gaussian Mixture Model.
            log_map_batch: Pre-built batch compute log_map function for efficiency.
        Returns:
            mu (jnp.ndarray): The mean of the distribution
            A (jnp.ndarray): The A matrix of the distribution
            sigma (jnp.ndarray): The covariance of the distribution
        """
        if method == "random":
            mu = X[jax.random.randint(key, (1,), 0, X.shape[0])].squeeze()
        elif method == "mean":
            # Start at the data point closest to the Euclidean mean to ensure it lies on the manifold
            euclidean_mean = jnp.mean(X, axis=0)
            closest_idx = jnp.argmin(jnp.sum((X - euclidean_mean) ** 2, axis=1))
            mu = X[closest_idx]
        elif method == "GMM":
            raise NotImplementedError("GMM initialization not implemented yet")
        else:
            raise ValueError("Invalid method")

        tangent_vectors = self._compute_log_maps(mu, X, manifold)
        sigma = jnp.cov(tangent_vectors.T)

        A = self.compute_A(sigma)
        return mu, A, sigma

    def _loss(
        self, sigma: jnp.ndarray, log_maps: jnp.ndarray, norm_const: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the negative log-likelihood (empirical risk) of the LAND model given the data.

        This computes the objective function locally using the Mahalanobis-like distance
        in the tangent space (via the log map), as well as the normalization constant.
        Params:
            sigma (jnp.ndarray): The covariance of the distribution
            log_maps (jnp.ndarray): Pre-computed log maps of all data points at mu, shape (N, D)
            normalization_constant (jnp.ndarray): The normalization constant of the distribution
        Returns:
            jnp.ndarray: The objective function value
        """
        inv_sigma = jnp.linalg.inv(sigma)
        distances = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)
        objective = jnp.sum(distances) / (2 * log_maps.shape[0])
        objective += jnp.log(norm_const)
        return objective

    def _compute_grad_mu(
        self,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        norm_const: jnp.ndarray,
        key: jax.Array,
        log_maps: jnp.ndarray,
        manifold: RiemannianManifold,
    ) -> jnp.ndarray:
        """
        Compute the gradient of the log-likelihood with respect to the spatial mean (mu).

        The gradient relies on two terms: an explicitly computable empirical mean in the
        tangent space (via the Riemannian log map on data points X), and an intractable
        integral term representing the gradient of the normalization constant. The intractable
        term is estimated using Monte Carlo sampling.
        Params:
            mu (jnp.ndarray): The mean of the distribution
            sigma (jnp.ndarray): The covariance of the distribution
            normalization_constant (jnp.ndarray): The normalization constant of the distribution
            key (jax.Array): Random key for operations
            log_maps (jnp.ndarray): Pre-computed log maps of all data points at mu, shape (N, D)
        Returns:
            jnp.ndarray: The gradient of the log-likelihood with respect to mu
        """
        # Compute log_map part of the gradient (empirical mean in tangent space)
        grad_mu_log_map = jnp.mean(log_maps, axis=0)

        # Compute exp_map part of the gradient (MC estimate of normalization integral)
        d = mu.shape[0]
        v_samples = jax.random.multivariate_normal(
            key, mean=jnp.zeros(d), cov=sigma, shape=(self.S,)
        )
        mc_scale = jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(sigma)) / (
            self.S * norm_const
        )

        def exp_loss(v):
            translated_point = manifold.exp_map(mu, v)
            M_trans = manifold.metric(translated_point)
            m_val = jnp.sqrt(jnp.linalg.det(M_trans))
            return m_val * v

        grad_mu_exp_map = -mc_scale * jnp.sum(jax.vmap(exp_loss)(v_samples), axis=0)
        return grad_mu_log_map + grad_mu_exp_map

    def _compute_grad_sigma(
        self,
        mu: jnp.ndarray,
        A: jnp.ndarray,
        sigma: jnp.ndarray,
        norm_const: jnp.ndarray,
        key: jax.Array,
        log_maps: jnp.ndarray,
        manifold: RiemannianManifold,
    ) -> jnp.ndarray:
        """
        Compute the gradient of the log-likelihood with respect to the precision factor A.

        The gradient with respect to the covariance matrix sigma is composed of an explicitly
        computable empirical covariance term involving the log-mapped data, and a sampled
        integral term for the normalization constant. The final gradient returned is with
        respect to the matrix A (where A.T @ A = inv(sigma)) through the chain rule.
        Params:
            mu (jnp.ndarray): The mean of the distribution
            A (jnp.ndarray): The A matrix of the distribution
            sigma (jnp.ndarray): The covariance of the distribution
            norm_const (jnp.ndarray): The normalization constant of the distribution
            key (jax.Array): Random key for operations
            log_maps (jnp.ndarray): Pre-computed log maps of all data points at mu, shape (N, D)
            manifold (RiemannianManifold): The Riemannian manifold
        Returns:
            jnp.ndarray: The gradient of the log-likelihood with respect to A matrix
        """
        # Compute log_map part of the gradient (empirical outer product in tangent space)
        grad_sigma_log_map = (log_maps.T @ log_maps) / log_maps.shape[0]

        # Compute exp_map part of the gradient (MC estimate of normalization integral)
        d = mu.shape[0]
        v_samples = jax.random.multivariate_normal(
            key, mean=jnp.zeros(d), cov=sigma, shape=(self.S,)
        )
        mc_scale = jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(sigma)) / (
            self.S * norm_const
        )

        def exp_outer(v):
            translated_point = manifold.exp_map(mu, v)
            M_trans = manifold.metric(translated_point)
            m_val = jnp.sqrt(jnp.linalg.det(M_trans))
            return m_val * jnp.outer(v, v)

        grad_sigma_exp_map = -mc_scale * jnp.sum(jax.vmap(exp_outer)(v_samples), axis=0)
        return A @ (grad_sigma_log_map + grad_sigma_exp_map)

    def compute_A(self, sigma: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the A matrix from the covariance matrix, A.T @ A = inv(sigma)
        Params:
            sigma (jnp.ndarray): The covariance of the distribution
        Returns:
            jnp.ndarray: The A matrix of the distribution
        """
        return jnp.linalg.cholesky(jnp.linalg.inv(sigma)).T
