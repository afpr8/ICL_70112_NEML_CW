from functools import partial
import torch

from utils import compute_normalization_constant, exp_map, log_map, metric


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
        init_method: str = "mean",
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

        self._metric = None

    def fit(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fit the LAND model to the data
        Params:
            X (torch.Tensor): The data to fit the model to, shape (N, D)
        Returns:
            mu (torch.Tensor): The mean of the distribution
            sigma (torch.Tensor): The covariance of the distribution
            normalization_constant (torch.Tensor): The normalization constant of the distribution
        """
        self._metric = partial(metric, X=X, sigma=self.sigma, rho=self.rho)

        mu, A, sigma = self._init_params(X, self.init_method)
        normalization_constant = compute_normalization_constant(
            mu, sigma, self._metric, self.S
        )
        loss_diff = float("inf")

        while loss_diff**2 > self.epsilon:
            # Store previous values
            prev_sigma = sigma
            prev_normalization_constant = normalization_constant
            prev_loss = self._loss(mu, sigma, X, normalization_constant)

            # Update mu
            grad_mu = self._compute_grad_mu(mu, sigma, X, normalization_constant)
            mu = exp_map(
                mu, self.lr_mu * grad_mu, self._metric(mu)
            )  # grad_mu already has the -1 factor
            normalization_constant = compute_normalization_constant(
                mu, sigma, self._metric, self.S
            )

            # Scale lr
            loss_diff = self._loss(mu, sigma, X, normalization_constant) - prev_loss
            if loss_diff > 0:
                self.lr_mu *= self.lr_scale_down
            else:
                self.lr_mu *= self.lr_scale_up

            # Update sigma
            grad_sigma = self._compute_grad_sigma(
                mu, A, sigma, X, normalization_constant
            )
            A = A - self.lr_A * grad_sigma
            sigma = torch.inverse(A.T @ A)

            prev_normalization_constant = normalization_constant
            normalization_constant = compute_normalization_constant(
                mu, sigma, self._metric, self.S
            )

            # Scale lr
            new_loss = self._loss(mu, sigma, X, normalization_constant)
            loss_diff = new_loss - self._loss(
                mu, prev_sigma, X, prev_normalization_constant
            )
            if loss_diff > 0:
                self.lr_A *= self.lr_scale_down
            else:
                self.lr_A *= self.lr_scale_up

            # Compute full loss diff for stopping condition
            loss_diff = new_loss - prev_loss
        self._metric = None  # avoid memory leak

        return mu, sigma, normalization_constant

    def _init_params(
        self, X: torch.Tensor, method: str = "mean"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize the parameters of the model
        Params:
            X (torch.Tensor): The data to initialize the parameters with
            method (str): The method to use for initialization.
                - "random": Initialize mu randomly, sigma from empirical cov of tangent vectors.
                - "mean": Initialize mu as the empirical mean, sigma from empirical cov of tangent vectors.
                - "GMM": Initialize mu and sigma with a Gaussian Mixture Model.
        Returns:
            mu (torch.Tensor): The mean of the distribution
            A (torch.Tensor): The A matrix of the distribution
            sigma (torch.Tensor): The covariance of the distribution
        """
        match method:
            case "random":
                mu = X[torch.randint(0, X.shape[0], (1,))].squeeze()
            case "mean":
                mu = torch.mean(X, dim=0)
            case "GMM":
                raise NotImplementedError("GMM initialization is not implemented yet")
            case _:
                raise ValueError("Invalid method")

        # Compute covariance from tangent vectors
        tangent_vectors = torch.stack([log_map(mu, x, self._metric(mu)) for x in X])
        sigma = torch.cov(tangent_vectors.T)

        A = self.compute_A(sigma)
        return mu, A, sigma

    def _loss(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        X: torch.Tensor,
        normalization_constant: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the negative log-likelihood (empirical risk) of the LAND model given the data.

        This computes the objective function locally using the Mahalanobis-like distance
        in the tangent space (via the log map), as well as the normalization constant.
        Params:
            mu (torch.Tensor): The mean of the distribution
            sigma (torch.Tensor): The covariance of the distribution
            X (torch.Tensor): The data
            normalization_constant (torch.Tensor): The normalization constant of the distribution
        Returns:
            torch.Tensor: The objective function value
        """
        objective = 0
        inv_sigma = torch.linalg.inv(sigma)
        for x in X:
            log_map_ = log_map(mu, x, self._metric(mu))
            objective += torch.dot(log_map_, inv_sigma @ log_map_)

        objective /= 2 * X.shape[0]
        objective += torch.log(normalization_constant)

        return objective

    def _compute_grad_mu(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        X: torch.Tensor,
        normalization_constant: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gradient of the log-likelihood with respect to the spatial mean (mu).

        The gradient relies on two terms: an explicitly computable empirical mean in the
        tangent space (via the Riemannian log map on data points X), and an intractable
        integral term representing the gradient of the normalization constant. The intractable
        term is estimated using Monte Carlo sampling.
        Params:
            mu (torch.Tensor): The mean of the distribution
            sigma (torch.Tensor): The covariance of the distribution
            X (torch.Tensor): The data
            normalization_constant (torch.Tensor): The normalization constant of the distribution
        Returns:
            torch.Tensor: The gradient of the log-likelihood with respect to mu
        """
        grad_mu_log_map = torch.zeros_like(mu)
        grad_mu_exp_map = torch.zeros_like(mu)

        # Compute log_map part of the gradient
        for x in X:
            grad_mu_log_map += log_map(mu, x, self._metric(mu))
        grad_mu_log_map /= X.shape[0]

        # Compute exp_map part of the gradient
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

    def _m(self, mu: torch.Tensor, v: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the deformation of the metric at mu in the direction of v.
        Params:
            mu (torch.Tensor): The mean of the distribution
            v (torch.Tensor): The tangent vector v at the point mu
            X (torch.Tensor): The data
        Returns:
            torch.Tensor: The square root of the metric determinant at exp_mu(v)
        """
        metric_mu = self._metric(mu)
        translated_point = exp_map(mu, v, metric_mu)
        metric_translated_point = self._metric(translated_point)
        return torch.sqrt(torch.linalg.det(metric_translated_point))

    def _compute_grad_sigma(
        self,
        mu: torch.Tensor,
        A: torch.Tensor,
        sigma: torch.Tensor,
        X: torch.Tensor,
        normalization_constant: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gradient of the log-likelihood with respect to the precision factor A.

        The gradient with respect to the covariance matrix sigma is composed of an explicitly
        computable empirical covariance term involving the log-mapped data, and a sampled
        integral term for the normalization constant. The final gradient returned is with
        respect to the matrix A (where A.T @ A = inv(sigma)) through the chain rule.
        Params:
            mu (torch.Tensor): The mean of the distribution
            A (torch.Tensor): The A matrix of the distribution
            sigma (torch.Tensor): The covariance of the distribution
            X (torch.Tensor): The data
            normalization_constant (torch.Tensor): The normalization constant of the distribution
        Returns:
            torch.Tensor: The gradient of the log-likelihood with respect to A matrix
        """
        grad_sigma_log_map = torch.zeros_like(sigma)
        grad_sigma_exp_map = torch.zeros_like(sigma)

        # Compute log_map part of the gradient
        for x in X:
            log_map_ = log_map(mu, x, self._metric(mu))
            grad_sigma_log_map += torch.outer(log_map_, log_map_)
        grad_sigma_log_map /= X.shape[0]

        # Compute exp_map part of the gradient
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

    def compute_A(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute the A matrix from the covariance matrix, A.T @ A = inv(sigma)
        Params:
            sigma (torch.Tensor): The covariance of the distribution
        Returns:
            torch.Tensor: The A matrix of the distribution
        """
        return torch.linalg.cholesky(torch.linalg.inv(sigma)).T
