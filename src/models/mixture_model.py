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
        with component-wise loss tracking and reversion.
        """
        manifold = RiemannianManifold(
            X, self.sigma, self.rho, self.K_segments, self.n_neighbors
        )
        N = X.shape[0]

        print("Initialising parameters...")
        self.key, subkey = jax.random.split(self.key)
        mu, A, sigma = self._init_params(
            X, key=subkey, method=self.init_method, manifold=manifold
        )
        pi = jnp.ones(self.K) / self.K

        # Initialise normalisation constants
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
        global_loss_prev = float("inf")
        n_wo_improvement = 0

        with tqdm(desc="Mixture Model EM", unit="epoch") as pbar:
            while n_wo_improvement < self.patience:
                r = jnp.zeros((N, self.K))
                log_maps_all = []
                inv_sigmas = []

                # E-STEP
                for k in range(self.K):
                    inv_sigma = jnp.linalg.inv(sigma[k])
                    inv_sigmas.append(inv_sigma)

                    log_maps = self._compute_log_maps(mu[k], X, manifold)
                    log_maps_all.append(log_maps)

                    dist_sq = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)

                    # p_M(x_n | mu_k, Sigma_k)
                    p_x = (1.0 / C[k]) * jnp.exp(-0.5 * dist_sq)
                    r = r.at[:, k].set(pi[k] * p_x)

                # Normalise responsibilities across components
                r_sum = r.sum(axis=1, keepdims=True)
                r_sum = jnp.clip(r_sum, a_min=1e-12)
                r = r / r_sum

                # Track global convergence
                global_loss_current = -jnp.sum(jnp.log(r_sum)) / N
                if t > 0 and (global_loss_prev - global_loss_current) <= self.epsilon:
                    n_wo_improvement += 1
                else:
                    n_wo_improvement = 0
                global_loss_prev = global_loss_current

                pbar.set_postfix(
                    loss=float(global_loss_current),
                    no_impr=int(n_wo_improvement),
                )
                pbar.update(1)

                # PRE-COMPUTE CLUSTER BASELINE LOSSES
                comp_losses = jnp.zeros(self.K)
                for k in range(self.K):
                    # Q-function loss: -sum( r_nk * log p(x | mu_k, Sigma_k) )
                    dist_sq = jnp.sum((log_maps_all[k] @ inv_sigmas[k]) * log_maps_all[k], axis=-1)
                    log_p_x = -jnp.log(C[k]) - 0.5 * dist_sq
                    comp_losses = comp_losses.at[k].set(-jnp.sum(r[:, k] * log_p_x))

                # M-STEP (Component-wise Propose & Check)
                for k in range(self.K):
                    N_k = r[:, k].sum()

                    # === 1. UPDATE & CHECK MU ===
                    self.key, subkey = jax.random.split(self.key)
                    grad_mu = self._compute_grad_mu(
                        mu[k], sigma[k], C[k], Vs[k], r[:, k], N_k, subkey, log_maps_all[k], manifold
                    )

                    # Propose new mu
                    new_mu_k = manifold.exp_map(mu[k], self.lr_mu * grad_mu)
                    new_C_k_mu, new_Vs_k_mu = manifold.compute_normalization_constant(
                        new_mu_k, sigma[k], subkey, n_samples=self.S
                    )

                    # To check the loss, we compute the new log maps
                    new_log_maps = self._compute_log_maps(new_mu_k, X, manifold)
                    new_dist_sq_mu = jnp.sum((new_log_maps @ inv_sigmas[k]) * new_log_maps, axis=-1)
                    new_log_p_x_mu = -jnp.log(new_C_k_mu) - 0.5 * new_dist_sq_mu
                    new_loss_mu = -jnp.sum(r[:, k] * new_log_p_x_mu)

                    # Accept or Reject mu
                    if new_loss_mu > comp_losses[k]:
                        self.lr_mu *= self.lr_scale_down
                        current_log_maps = log_maps_all[k] # Revert: use old log maps for sigma step
                    else:
                        mu = mu.at[k].set(new_mu_k)
                        C[k] = new_C_k_mu
                        Vs[k] = new_Vs_k_mu
                        current_log_maps = new_log_maps # Accept: pass new log maps to sigma step
                        comp_losses = comp_losses.at[k].set(new_loss_mu)
                        self.lr_mu *= self.lr_scale_up

                    # === 2. UPDATE & CHECK SIGMA ===
                    self.key, subkey = jax.random.split(self.key)
                    grad_sigma = self._compute_grad_sigma(
                        mu[k], A[k], sigma[k], C[k], Vs[k], r[:, k], N_k, subkey, current_log_maps, manifold
                    )

                    # Propose new sigma
                    new_A_k = A[k] - (self.lr_A * grad_sigma)
                    new_sigma_k = jnp.linalg.inv(new_A_k.T @ new_A_k)
                    new_C_k_sig, new_Vs_k_sig = manifold.compute_normalization_constant(
                        mu[k], new_sigma_k, subkey, n_samples=self.S
                    )

                    # Check the loss (we can reuse current_log_maps here)
                    inv_new_sigma_k = jnp.linalg.inv(new_sigma_k)
                    new_dist_sq_sig = jnp.sum((current_log_maps @ inv_new_sigma_k) * current_log_maps, axis=-1)
                    new_log_p_x_sig = -jnp.log(new_C_k_sig) - 0.5 * new_dist_sq_sig
                    new_loss_sigma = -jnp.sum(r[:, k] * new_log_p_x_sig)

                    # Accept or Reject sigma
                    if new_loss_sigma > comp_losses[k]:
                        self.lr_A *= self.lr_scale_down
                    else:
                        A = A.at[k].set(new_A_k)
                        sigma = sigma.at[k].set(new_sigma_k)
                        C[k] = new_C_k_sig
                        Vs[k] = new_Vs_k_sig
                        comp_losses = comp_losses.at[k].set(new_loss_sigma)
                        self.lr_A *= self.lr_scale_up

                    # Update mixing weights
                    pi = pi.at[k].set(N_k / N)

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

    def _compute_grad_mu(
        self,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        norm_const: jnp.ndarray,
        v_samples: jnp.ndarray,
        r_k: jnp.ndarray,
        N_k: jnp.ndarray,
        key: jax.Array,
        log_maps: jnp.ndarray,
        manifold: RiemannianManifold,
    ) -> jnp.ndarray:
        """
        Compute the gradient of the log-likelihood with respect to the spatial mean (mu)
        for a single mixture component.

        The gradient relies on two terms: a responsibility-weighted empirical mean in the
        tangent space (via the Riemannian log map on data points X), and an intractable
        integral term representing the gradient of the normalisation constant.
        Params:
            mu (jnp.ndarray): The mean of the component distribution
            sigma (jnp.ndarray): The covariance of the component distribution
            norm_const (jnp.ndarray): The normalisation constant of the component
            v_samples (jnp.ndarray): The samples used to compute the normalisation constant
            r_k (jnp.ndarray): Responsibility of this component for each point, shape (N,)
            N_k (jnp.ndarray): Sum of responsibilities for this component
            key (jax.Array): Random key for operations
            log_maps (jnp.ndarray): Pre-computed log maps of all data points at mu, shape (N, D)
            manifold (RiemannianManifold): The Riemannian manifold
        Returns:
            jnp.ndarray: The gradient of the log-likelihood with respect to mu
        """
        # Compute log_map part of the gradient (responsibility-weighted mean in tangent space)
        # We multiply each log map by its corresponding responsibility before summing
        grad_mu_log_map = jnp.sum(r_k[:, None] * log_maps, axis=0) / N_k

        # Compute exp_map part of the gradient (MC estimate of normalisation integral)
        d = mu.shape[0]
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
        v_samples: jnp.ndarray,
        r_k: jnp.ndarray,
        N_k: jnp.ndarray,
        key: jax.Array,
        log_maps: jnp.ndarray,
        manifold: RiemannianManifold,
    ) -> jnp.ndarray:
        """
        Compute the gradient of the log-likelihood with respect to the precision factor A
        for a single mixture component.

        The gradient with respect to the covariance matrix sigma is composed of a responsibility-weighted
        empirical covariance term involving the log-mapped data, and a sampled
        integral term for the normalisation constant. The final gradient returned is with
        respect to the matrix A (where A.T @ A = inv(sigma)) through the chain rule.
        Params:
            mu (jnp.ndarray): The mean of the component distribution
            A (jnp.ndarray): The A matrix of the component distribution
            sigma (jnp.ndarray): The covariance of the component distribution
            norm_const (jnp.ndarray): The normalisation constant of the component
            v_samples (jnp.ndarray): The samples used to compute the normalisation constant
            r_k (jnp.ndarray): Responsibility of this component for each point, shape (N,)
            N_k (jnp.ndarray): Sum of responsibilities for this component
            key (jax.Array): Random key for operations
            log_maps (jnp.ndarray): Pre-computed log maps of all data points at mu, shape (N, D)
            manifold (RiemannianManifold): The Riemannian manifold
        Returns:
            jnp.ndarray: The gradient of the log-likelihood with respect to A matrix
        """
        # Compute log_map part of the gradient (responsibility-weighted outer product in tangent space)
        # Multiply each log map by its responsibility before computing the dot product
        grad_sigma_log_map = ((log_maps * r_k[:, None]).T @ log_maps) / N_k

        # Compute exp_map part of the gradient (MC estimate of normalisation integral)
        d = mu.shape[0]
        mc_scale = jnp.sqrt((2 * jnp.pi) ** d * jnp.linalg.det(sigma)) / (
            self.S * norm_const
        )

        def exp_outer(v):
            translated_point = manifold.exp_map(mu, v)
            M_trans = manifold.metric(translated_point)
            # Retaining your diagonal metric tensor assumption here
            m_val = jnp.exp(0.5 * jnp.sum(jnp.log(jnp.diag(M_trans))))
            return m_val * jnp.outer(v, v)

        grad_sigma_exp_map = -mc_scale * jnp.sum(jax.vmap(exp_outer)(v_samples), axis=0)
        
        return A @ (grad_sigma_log_map + grad_sigma_exp_map)
    
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
