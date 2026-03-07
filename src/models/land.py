import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.utils.land_utils import (
    compute_normalization_constant,
    jax_exp_map,
    jax_log_map_shooting,
    jax_metric,
)


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
        init_method: str = "random",
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

        self.init_method = init_method
        self.key = jax.random.key(seed)
        self._metric = None

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
        self._metric = partial(jax_metric, X=X, sigma=self.sigma, rho=self.rho)
        self.loss_history = []
        self.mu_history = []

        # Cache the vmapped log_map_shooting for reuse every iteration
        log_map_vmap = jax.vmap(jax_log_map_shooting, in_axes=(None, 0, None))

        self.key, subkey = jax.random.split(self.key)
        mu, A, sigma = self._init_params(X, subkey, self.init_method, log_map_vmap)

        self.mu_history.append(np.array(mu))

        self.key, subkey = jax.random.split(self.key)
        normalization_constant = compute_normalization_constant(
            mu, sigma, self._metric, subkey, self.S
        )
        loss_diff = float("inf")

        t = 0
        with tqdm(desc="LAND MLE Fit", unit=" epoch") as pbar:
            while (loss_diff**2 > self.epsilon) or t < 200:  # Ensure at least 5 epochs for some initial progress
                # Store previous values
                prev_sigma = sigma
                prev_normalization_constant = normalization_constant

                # Compute log_maps once, reused by loss, grad_mu, and grad_sigma
                log_maps = log_map_vmap(mu, X, self._metric)
                prev_loss = self._loss(sigma, log_maps, normalization_constant)
                
                # Capture the initial loss on the very first epoch
                if t == 0:
                    self.loss_history.append(float(prev_loss))

                # Update mu
                self.key, subkey = jax.random.split(self.key)
                grad_mu = self._compute_grad_mu(
                    mu, sigma, normalization_constant, subkey, log_maps
                )
                mu = jax_exp_map(
                    mu, self.lr_mu * grad_mu, self._metric
                )

                # Store mu history for visualization
                self.mu_history.append(np.array(mu))

                # self.key, subkey = jax.random.split(self.key)
                normalization_constant = compute_normalization_constant(
                    mu, sigma, self._metric, subkey, self.S
                )

                # Scale lr_mu: shrink if loss increased, grow if it decreased
                log_maps = log_map_vmap(mu, X, self._metric)
                loss_diff = (
                    self._loss(sigma, log_maps, normalization_constant) - prev_loss
                )
                

                # Update sigma
                # self.key, subkey = jax.random.split(self.key)
                grad_sigma = self._compute_grad_sigma(
                    mu, A, sigma, normalization_constant, subkey, log_maps
                )
                A = A - self.lr_A * grad_sigma
                sigma = jnp.linalg.inv(A.T @ A)

                prev_normalization_constant = normalization_constant
                # self.key, subkey = jax.random.split(self.key)
                normalization_constant = compute_normalization_constant(
                    mu, sigma, self._metric, subkey, self.S
                )

                # Calculate new loss and append to history
                new_loss = self._loss(sigma, log_maps, normalization_constant)
                self.loss_history.append(float(new_loss))
                
                # Scale lr_A: compare new sigma loss against previous sigma
                loss_diff_A = new_loss - self._loss(
                    prev_sigma, log_maps, prev_normalization_constant
                )
                if loss_diff_A > 0:
                    self.lr_A *= self.lr_scale_down
                else:
                    self.lr_A *= self.lr_scale_up

                loss_diff = new_loss - prev_loss

                pbar.set_postfix(loss_diff=float(loss_diff), loss=float(new_loss))
                pbar.update(1)
                
                t += 1
                self.lr_mu *= 0.95
                self.lr_A *= 0.95

        self._metric = None
        return mu, sigma, normalization_constant

    def plot_loss(self):
        """
        Plot the empirical risk (negative log-likelihood) over the training epochs.
        """
        if not hasattr(self, 'loss_history') or not self.loss_history:
            print("No loss history found. Please run fit() first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, marker='o', linestyle='-', color='#CD5C5C', markersize=4)
        
        plt.title('LAND MLE Training Loss Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Negative Log-Likelihood', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ensure the layout is tight so labels aren't cut off
        plt.tight_layout()
        plt.savefig(f"src/plots/land_loss_curve_{self.sigma}.png", dpi=300)
        plt.close()

    def plot_trajectory(self, X: jnp.ndarray, filename: str = "src/plots/land_trajectory.png"):
        """
        Visualise and save the trajectory of the spatial mean (mu) across the manifold.
        """
        if not hasattr(self, 'mu_history') or not self.mu_history:
            print("No trajectory history found. Please run fit() first.")
            return

        
        # Convert the history list to a 2D NumPy array for easy coordinate slicing
        trajectory = np.array(self.mu_history)

        plt.figure(figsize=(10, 8))
        
        # 1. Plot the background dataset
        plt.scatter(X[:, 0], X[:, 1], c='dodgerblue', alpha=0.5, s=30, label='Data')
        
        # 2. Draw the connected path of the mean
        plt.plot(trajectory[:, 0], trajectory[:, 1], c='crimson', linestyle='-', linewidth=2, alpha=0.8, label='Trajectory')
        
        # 3. Mark every individual epoch step along the path
        plt.scatter(trajectory[:, 0], trajectory[:, 1], c='crimson', s=15, zorder=5)
        
        # 4. Highlight the exact start and end coordinates
        plt.scatter(trajectory[0, 0], trajectory[0, 1], c='gold', edgecolors='black', marker='s', s=100, label='Start', zorder=6)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='limegreen', edgecolors='black', marker='*', s=200, label='End', zorder=6)

        plt.title(f'LAND MLE Mean Trajectory (sigma = {self.sigma})', fontsize=14, fontweight='bold')
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def _init_params(
        self, X: jnp.ndarray, key: jax.Array, method: str = "mean", log_map_vmap=None
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
            log_map_vmap: Pre-built vmap of jax_log_map_shooting for efficiency.
        Returns:
            mu (jnp.ndarray): The mean of the distribution
            A (jnp.ndarray): The A matrix of the distribution
            sigma (jnp.ndarray): The covariance of the distribution
        """
        if method == "random":
            mu = X[jax.random.randint(key, (1,), 0, X.shape[0])].squeeze()
        elif method == "mean":
            mu = jnp.mean(X, axis=0)
        elif method == "GMM":
            raise NotImplementedError("GMM initialization not implemented yet")
        else:
            raise ValueError("Invalid method")

        # Compute covariance from tangent vectors at mu
        if log_map_vmap is None:
            log_map_vmap = jax.vmap(jax_log_map_shooting, in_axes=(None, 0, None))
        tangent_vectors = log_map_vmap(mu, X, self._metric)
        sigma = jnp.cov(tangent_vectors.T)

        A = self.compute_A(sigma)
        return mu, A, sigma

    def _loss(
        self,
        sigma: jnp.ndarray,
        log_maps: jnp.ndarray,
        normalization_constant: jnp.ndarray,
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
        # Compute Mahalanobis-like squared distances in the tangent space
        distances = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)

        objective = jnp.sum(distances) / (2 * log_maps.shape[0])
        objective += jnp.log(normalization_constant)
        return objective

    def _compute_grad_mu(
        self,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        normalization_constant: jnp.ndarray,
        key: jax.Array,
        log_maps: jnp.ndarray,
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
            self.S * normalization_constant
        )

        def exp_loss(v):
            return self._m(mu, v) * v

        grad_mu_exp_map = -mc_scale * jnp.sum(jax.vmap(exp_loss)(v_samples), axis=0)

        return grad_mu_log_map + grad_mu_exp_map

    def _m(self, mu: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the deformation of the metric at mu in the direction of v.
        Params:
            mu (jnp.ndarray): The mean of the distribution
            v (jnp.ndarray): The tangent vector v at the point mu
        Returns:
            jnp.ndarray: The square root of the metric determinant at exp_mu(v)
        """
        translated_point = jax_exp_map(mu, v, self._metric)
        metric_translated_point = self._metric(translated_point)
        return jnp.sqrt(jnp.linalg.det(metric_translated_point))

    def _compute_grad_sigma(
        self,
        mu: jnp.ndarray,
        A: jnp.ndarray,
        sigma: jnp.ndarray,
        normalization_constant: jnp.ndarray,
        key: jax.Array,
        log_maps: jnp.ndarray,
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
            normalization_constant (jnp.ndarray): The normalization constant of the distribution
            key (jax.Array): Random key for operations
            log_maps (jnp.ndarray): Pre-computed log maps of all data points at mu, shape (N, D)
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
            self.S * normalization_constant
        )

        def exp_outer(v):
            return self._m(mu, v) * jnp.outer(v, v)

        grad_sigma_exp_map = -mc_scale * jnp.sum(jax.vmap(exp_outer)(v_samples), axis=0)

        # Chain rule: gradient w.r.t. A from gradient w.r.t. sigma
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
