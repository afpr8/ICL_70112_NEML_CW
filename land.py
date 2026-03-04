from functools import partial
import torch

from utils import exp_map, log_map, compute_normalization_constant, metric


class LANDMLE:
    def __init__(
        self,
        lr_mu: float = 1e-3,
        lr_A: float = 1e-3,
        S: int = 100,
        epsilon: float = 1e-3,
        sigma: float = 1.0,
        rho: float = 1e-3,
    ):
        """
        Initialize the LAND model
        Params:
            lr_mu (float): The learning rate for mu
            lr_A (float): The learning rate for A
            S (int): The number of vectors sampled to estimate the exp_map part of the gradient
            epsilon (float): The tolerance for the end condition
            sigma (float): Hyperparameter to compute the metric
            rho (float): Hyperparameter to compute the metric
        """
        self.lr_mu = lr_mu
        self.lr_A = lr_A
        self.S = S
        self.epsilon = epsilon

        self.sigma = sigma
        self.rho = rho

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

        mu, A, sigma = self._init_params(X)
        t = 0
        normalization_constant = compute_normalization_constant(mu, sigma, self._metric)
        loss_diff = float("inf")

        while loss_diff**2 > self.epsilon:
            # Store previous values
            prev_mu = mu
            prev_sigma = sigma
            prev_normalization_constant = normalization_constant

            # Update mu
            grad_mu = self._compute_grad_mu(mu, sigma, X, normalization_constant)
            mu = exp_map(
                mu, self.lr_mu * grad_mu, self._metric(mu)
            )  # grad_mu already has the -1 factor
            normalization_constant = compute_normalization_constant(
                mu, sigma, self._metric
            )

            # Scale lr
            loss_diff = self._loss(mu, sigma, X, normalization_constant) - self._loss(
                prev_mu, prev_sigma, X, prev_normalization_constant
            )
            if loss_diff > 0:
                self.lr_mu *= 0.75
            else:
                self.lr_mu *= 1.1

            # Update sigma
            grad_sigma = self._compute_grad_sigma(
                mu, A, sigma, X, normalization_constant
            )
            A = A - self.lr_A * grad_sigma
            sigma = torch.inverse(A.T @ A)  # TODO check if we can avoid inverse
            normalization_constant = compute_normalization_constant(
                mu, sigma, self._metric
            )

            # Scale lr
            loss_diff = self._loss(mu, sigma, X, normalization_constant) - self._loss(
                mu, prev_sigma, X, prev_normalization_constant
            )
            if loss_diff > 0:
                self.lr_A *= 0.75
            else:
                self.lr_A *= 1.1

            # Compute full loss diff for stopping condition
            loss_diff = self._loss(mu, sigma, X, normalization_constant) - self._loss(
                prev_mu, prev_sigma, X, prev_normalization_constant
            )

            t += 1
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
                - "random": Initialize mu randomly, sigma from empirical cov of tangent vectors[cite: 519, 520].
                - "mean": Initialize mu empirically, sigma from empirical cov of tangent vectors[cite: 523].
                - "GMM": Initialize mu and sigma with a Gaussian Mixture Model[cite: 525, 526].
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
                raise NotImplementedError(
                    "GMM initialization is not implemented yet [cite: 525]"
                )
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
        Compute the objective function
        Params:
            mu (torch.Tensor): The mean of the distribution
            sigma (torch.Tensor): The covariance of the distribution
            X (torch.Tensor): The data
            normalization_constant (torch.Tensor): The normalization constant of the distribution
        Returns:
            torch.Tensor: The objective function
        """
        objective = 0
        inv_sigma = torch.linalg.inv(sigma)  # TODO cache inverse sigma?
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
        Compute the gradient of the log-likelihood with respect to mu
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
        Compute the deformation of the metric at mu in the direction of v
        Params:
            mu (torch.Tensor): The mean of the distribution
            v (torch.Tensor): The vector to compute the m function with
            X (torch.Tensor): The data
        Returns:
            torch.Tensor: The deformation of the metric at mu in the direction of v
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
        Compute the gradient of the log-likelihood with respect to sigma
        Params:
            mu (torch.Tensor): The mean of the distribution
            A (torch.Tensor): The A matrix of the distribution
            sigma (torch.Tensor): The covariance of the distribution
            X (torch.Tensor): The data
            normalization_constant (torch.Tensor): The normalization constant of the distribution
        Returns:
            torch.Tensor: The gradient of the log-likelihood with respect to sigma
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
        return torch.linalg.cholesky(torch.linalg.inv(sigma)).T  # TODO store inverse?
