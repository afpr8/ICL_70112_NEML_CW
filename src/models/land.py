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

    def fit(self, X: jnp.ndarray, max_epochs: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the LAND model to the data.
        Params:
            X (jnp.ndarray): The data to fit the model to, shape (N, D)
            max_epochs (int): Safety limit to prevent infinite loops.
        Returns:
            mu (np.ndarray): The mean of the distribution
            sigma (np.ndarray): The covariance of the distribution
            normalization_constant (np.ndarray): The normalization constant of the distribution
        """
        self._metric = partial(jax_metric, X=X, sigma=self.sigma, rho=self.rho)
        log_map_vmap = jax.vmap(jax_log_map_shooting, in_axes=(None, 0, None))

        # Initialise parameters
        self.key, subkey = jax.random.split(self.key)
        mu, A, sigma = self._init_params(X, subkey, self.init_method, log_map_vmap)

        # ---------------------------------------------------------------------
        # Pre-compile the update step to avoid Python loop overhead.
        # This pure function executes entirely on the accelerator.
        # ---------------------------------------------------------------------
        @jax.jit
        def update_step(mu_curr, A_curr, sigma_curr, lr_mu_curr, lr_A_curr, key_curr):
            # 1. Split keys for all stochastic operations in this step
            key_curr, key_mu, key_sigma, key_norm1, key_norm2 = jax.random.split(key_curr, 5)

            # Compute log maps and normalisation for the current state
            log_maps_curr = log_map_vmap(mu_curr, X, self._metric)
            norm_const_curr = compute_normalization_constant(
                mu_curr, sigma_curr, self._metric, key_norm1, self.S
            )
            loss_curr = self._loss(sigma_curr, log_maps_curr, norm_const_curr)

            # --- Update spatial mean (mu) ---
            grad_mu = self._compute_grad_mu(mu_curr, sigma_curr, norm_const_curr, key_mu, log_maps_curr)
            mu_new = jax_exp_map(mu_curr, lr_mu_curr * grad_mu, self._metric)
            
            # --- Evaluate intermediate state (New mu, Old sigma) ---
            # Required to properly isolate and measure the effect of the A update
            log_maps_new_mu = log_map_vmap(mu_new, X, self._metric)
            norm_const_intermediate = compute_normalization_constant(
                mu_new, sigma_curr, self._metric, key_norm2, self.S
            )
            loss_intermediate = self._loss(sigma_curr, log_maps_new_mu, norm_const_intermediate)

            # --- Update precision factor (A) and covariance (sigma) ---
            grad_sigma = self._compute_grad_sigma(
                mu_new, A_curr, sigma_curr, norm_const_intermediate, key_sigma, log_maps_new_mu
            )
            A_new = A_curr - lr_A_curr * grad_sigma
            
            # Stabilised covariance calculation: solve (A^T A) * Sigma = I
            precision_matrix = A_new.T @ A_new + 1e-8 * jnp.eye(A_new.shape[0])
            sigma_new = jnp.linalg.solve(precision_matrix, jnp.eye(A_new.shape[0]))
            
            # --- Evaluate final state (New mu, New sigma) ---
            key_curr, key_norm3 = jax.random.split(key_curr)
            norm_const_new = compute_normalization_constant(
                mu_new, sigma_new, self._metric, key_norm3, self.S
            )
            loss_new = self._loss(sigma_new, log_maps_new_mu, norm_const_new)

            # --- Adjust Learning Rates ---
            loss_diff_A = loss_new - loss_intermediate
            
            # Use jnp.where to handle control flow inside JIT compilation
            lr_A_new = jnp.where(
                loss_diff_A > 0, 
                lr_A_curr * self.lr_scale_down, 
                lr_A_curr * self.lr_scale_up
            )
            lr_mu_new = lr_mu_curr * 0.95
            
            loss_diff_total = loss_curr - loss_new

            return mu_new, A_new, sigma_new, norm_const_new, loss_new, loss_diff_total, lr_mu_new, lr_A_new, key_curr

        # ---------------------------------------------------------------------
        # Training Loop
        # ---------------------------------------------------------------------
        mu_history_jax = [mu]
        loss_history_jax = []
        
        # Calculate the absolute initial loss before starting
        self.key, subkey = jax.random.split(self.key)
        initial_log_maps = log_map_vmap(mu, X, self._metric)
        initial_norm_const = compute_normalization_constant(mu, sigma, self._metric, subkey, self.S)
        current_loss = self._loss(sigma, initial_log_maps, initial_norm_const)
        loss_history_jax.append(current_loss)
        
        loss_diff = jnp.array(float("inf"))
        t = 0
        
        with tqdm(desc="LAND MLE Fit", unit=" epoch", total=max_epochs) as pbar:
            while t < 20 or (loss_diff**2 > self.epsilon and t < max_epochs):
                
                # Execute the compiled JAX step
                mu, A, sigma, norm_const, current_loss, loss_diff, self.lr_mu, self.lr_A, self.key = update_step(
                    mu, A, sigma, self.lr_mu, self.lr_A, self.key
                )
                
                # Append raw JAX tracer arrays (does not trigger device-to-host transfer)
                mu_history_jax.append(mu)
                loss_history_jax.append(current_loss)
                
                # Convert only the strictly necessary scalars for the progress bar
                pbar.set_postfix(loss_diff=float(loss_diff), loss=float(current_loss))
                pbar.update(1)
                t += 1

        # Synchronise arrays back to the host CPU entirely at the end
        self.loss_history = [float(l) for l in loss_history_jax]
        self.mu_history = [np.array(m) for m in mu_history_jax]

        self._metric = None
        return np.array(mu), np.array(sigma), np.array(norm_const)

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

    # def _loss(
    #     self,
    #     sigma: jnp.ndarray,
    #     log_maps: jnp.ndarray,
    #     normalization_constant: jnp.ndarray,
    # ) -> jnp.ndarray:
    #     """
    #     Compute the negative log-likelihood (empirical risk) of the LAND model given the data.

    #     This computes the objective function locally using the Mahalanobis-like distance
    #     in the tangent space (via the log map), as well as the normalization constant.
    #     Params:
    #         sigma (jnp.ndarray): The covariance of the distribution
    #         log_maps (jnp.ndarray): Pre-computed log maps of all data points at mu, shape (N, D)
    #         normalization_constant (jnp.ndarray): The normalization constant of the distribution
    #     Returns:
    #         jnp.ndarray: The objective function value
    #     """
    #     inv_sigma = jnp.linalg.inv(sigma)
    #     # Compute Mahalanobis-like squared distances in the tangent space
    #     distances = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)

    #     objective = jnp.sum(distances) / (2 * log_maps.shape[0])
    #     objective += jnp.log(normalization_constant)
    #     return objective

    def _loss(
        self,
        sigma: jnp.ndarray,
        log_maps: jnp.ndarray,
        normalization_constant: jnp.ndarray,
    ) -> jnp.ndarray:
        # 1. Use a robust linear solver instead of direct inversion.
        # jax.scipy.linalg.solve with assume_a='pos' is highly optimised for covariance matrices.
        inv_sigma_log_maps = jax.scipy.linalg.solve(sigma, log_maps.T, assume_a='pos').T
        
        # Compute Mahalanobis-like squared distances
        distances = jnp.sum(log_maps * inv_sigma_log_maps, axis=-1)
        objective = jnp.sum(distances) / (2 * log_maps.shape[0])
        
        # 2. Prevent NaN from log(0) or log(negative)
        safe_norm_const = jnp.maximum(normalization_constant, 1e-12)
        objective += jnp.log(safe_norm_const)
        
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
