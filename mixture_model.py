from functools import partial
from sklearn.mixture import GaussianMixture

import torch

from utils import exp_map, log_map, compute_normalization_constant, metric

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
        """
        self.K = K
        self.lr_mu = lr_mu
        self.lr_A = lr_A
        self.S = S
        self.epsilon = epsilon
        self.sigma = sigma
        self.rho = rho
        self._metric = None

    def fit(self, X: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """
        Fit the LAND mixture model to the data using Expectation-Maximisation
        Params:
            X (torch.Tensor): The data to fit the model to, shape (N, D)
        Returns:
            mu (list[torch.Tensor]): The means of the K distributions
            sigma (list[torch.Tensor]): The covariances of the K distributions
            C (list[torch.Tensor]): The normalisation constants
            pi (torch.Tensor): The mixing weights
        """
        self._metric = partial(metric, X=X, sigma=self.sigma, rho=self.rho)
        N = X.shape[0]

        # Initialise the parameters
        mu, A, sigma = self._init_params(X)
        pi = torch.ones(self.K) / self.K
        C = [compute_normalization_constant(mu[k], sigma[k], self._metric) for k in range(self.K)]

        t = 0
        loss_diff = float("inf")
        prev_loss = float("inf")

        # Repeat (EM Loop)
        while loss_diff**2 > self.epsilon:
            
            # Compute the responsibilities r_nk
            r = torch.zeros((N, self.K))
            for k in range(self.K):
                inv_sigma = torch.linalg.inv(sigma[k])
                for n, x in enumerate(X):
                    lm = log_map(mu[k], x, self._metric)
                    dist_sq = torch.dot(lm, inv_sigma @ lm)
                    
                    # p_M(x_n | mu_k, Sigma_k)
                    p_x = (1.0 / C[k]) * torch.exp(-0.5 * dist_sq)
                    r[n, k] = pi[k] * p_x

            # Normalise responsibilities across components for each point
            r_sum = r.sum(dim=1, keepdim=True)
            
            # Prevent division by zero in case of extreme outliers
            r_sum = torch.clamp(r_sum, min=1e-12) 
            r = r / r_sum

            # Calculate current negative log-likelihood to monitor convergence
            current_loss = -torch.log(r_sum).sum() / N
            if t > 0:
                loss_diff = current_loss - prev_loss
            prev_loss = current_loss

            if loss_diff**2 <= self.epsilon and t > 0:
                break


            # Maximisation step 
            for k in range(self.K):
                N_k = r[:, k].sum()
                
                # compute gradient for mu
                grad_mu = self._compute_grad_mu_k(mu[k], sigma[k], X, C[k], r[:, k], N_k)

                # update mu
                mu[k] = exp_map(mu[k], self.lr_mu * grad_mu, self._metric)

                # estimate C_k using eq. 16
                C[k] = compute_normalization_constant(mu[k], sigma[k], self._metric)

                # compute gradient for A using eq. 48
                grad_sigma = self._compute_grad_sigma_k(mu[k], A[k], sigma[k], X, C[k], r[:, k], N_k)

                # update A
                A[k] -= self.lr_A * grad_sigma

                # update Sigma
                sigma[k] = torch.inverse(A[k].T @ A[k])

                # update pi
                pi[k] = N_k / N

            t += 1

        # self._metric = None  # avoid memory leak
        return mu, sigma, C, pi

    def _init_params(self, X: torch.Tensor, method: str = "mean") -> tuple[list, list, list]:
        """
        Initialise the parameters of the mixture model.
        Params:
            X (torch.Tensor): The data to initialise the parameters with
            method (str): The method to use for initialisation.
                - "random": Initialise mu by randomly selecting data points.
                - "mean": Initialise mu near the empirical mean with slight noise.
                - "GMM": Initialise mu with a Euclidean Gaussian Mixture Model.
        Returns:
            mu (list[torch.Tensor]): The means of the distributions
            A (list[torch.Tensor]): The A matrices of the distributions
            sigma (list[torch.Tensor]): The covariances of the distributions
        """
        N = X.shape[0]
        
        if method == "random":
            # Randomly select initial means from data points
            indices = torch.randperm(N)[:self.K]
            mu = [X[idx].squeeze() for idx in indices]
            
        elif method == "GMM":
            # Fit standard Euclidean GMM for a warm start
            gmm = GaussianMixture(n_components=self.K, covariance_type='full', random_state=42)
            gmm.fit(X.numpy())
            mu = [torch.tensor(m, dtype=torch.float32) for m in gmm.means_]
            
        elif method == "mean":
            # Calculate the global empirical mean
            global_mean = torch.mean(X, dim=0)
            
            # Add small random noise to break symmetry. 
            # If all K components start at the exact same coordinates, 
            # they will compute identical gradients and never separate.
            mu = [global_mean + 0.1 * torch.randn_like(global_mean) for _ in range(self.K)]
            
        else:
            raise ValueError(f"Invalid initialisation method: {method}")

        A = []
        sigma = []
        
        for k in range(self.K):
            # Calculate initial covariance from tangent vectors mapped from the new mean
            tangent_vectors = torch.stack([log_map(mu[k], x, self._metric) for x in X])
            sig = torch.cov(tangent_vectors.T)
            
            # Add a tiny ridge to the diagonal to ensure positive definiteness 
            # and numerical stability during the first inversion
            sig += torch.eye(sig.shape[0]) * 1e-6 
            
            sigma.append(sig)
            A.append(self.compute_A(sig))
            
        return mu, A, sigma


    def _compute_grad_mu_k(
        self, mu: torch.Tensor, sigma: torch.Tensor, X: torch.Tensor, 
        normalization_constant: torch.Tensor, r_k: torch.Tensor, N_k: torch.Tensor
    ) -> torch.Tensor:
        """Weighted gradient computation for mu"""
        grad_mu_log_map = torch.zeros_like(mu)
        grad_mu_exp_map = torch.zeros_like(mu)

        # Compute log_map part weighted by responsibilities
        for n, x in enumerate(X):
            grad_mu_log_map += r_k[n] * log_map(mu, x, self._metric)
        grad_mu_log_map /= N_k

        # Compute exp_map part (independent of empirical data weights)
        dist = torch.distributions.MultivariateNormal(
            torch.zeros_like(mu), covariance_matrix=sigma
        )
        for _ in range(self.S):
            v = dist.sample()
            grad_mu_exp_map -= self._m(mu, v, X) * v

        grad_mu_exp_map *= torch.sqrt(
            (2 * torch.pi) ** mu.shape[0] * torch.linalg.det(sigma)
        ) / (self.S * normalization_constant)

        return grad_mu_log_map + grad_mu_exp_map

    def _compute_grad_sigma_k(
        self, mu: torch.Tensor, A: torch.Tensor, sigma: torch.Tensor, X: torch.Tensor, 
        normalization_constant: torch.Tensor, r_k: torch.Tensor, N_k: torch.Tensor
    ) -> torch.Tensor:
        """Weighted gradient computation for sigma"""
        grad_sigma_log_map = torch.zeros_like(sigma)
        grad_sigma_exp_map = torch.zeros_like(sigma)

        # Compute log_map part weighted by responsibilities
        for n, x in enumerate(X):
            log_map_ = log_map(mu, x, self._metric)
            grad_sigma_log_map += r_k[n] * torch.outer(log_map_, log_map_)
        grad_sigma_log_map /= N_k

        # Compute exp_map part
        dist = torch.distributions.MultivariateNormal(
            torch.zeros_like(mu), covariance_matrix=sigma
        )
        vs = dist.sample((self.S,))
        for v in vs:
            grad_sigma_exp_map -= self._m(mu, v, X) * torch.outer(v, v)

        grad_sigma_exp_map *= torch.sqrt(
            (2 * torch.pi) ** mu.shape[0] * torch.linalg.det(sigma)
        ) / (self.S * normalization_constant)

        return A @ (grad_sigma_log_map + grad_sigma_exp_map)

    def _m(self, mu: torch.Tensor, v: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        translated_point = exp_map(mu, v, self._metric)
        metric_translated_point = self._metric(translated_point)
        return torch.sqrt(torch.linalg.det(metric_translated_point))

    def compute_A(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(torch.linalg.inv(sigma)).T