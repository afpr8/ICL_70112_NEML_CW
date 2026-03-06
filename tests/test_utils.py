# A testing file for the LAND algorithm utilities functions

import torch

from utils import (
    torch_compute_normalization_constant,
    torch_exp_map,
    torch_log_map,
    torch_metric,
)


def test_metric_shape_and_pd():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = torch.tensor([2.0, 3.0])
    M_x = torch_metric(x, X, sigma=1.0, rho=1e-3)

    # Check shape
    assert M_x.shape == (2, 2)

    # Check diagonal positive entries
    assert torch.all(torch.diag(M_x) > 0)

    # Check that off-diagonal entries are zero
    off_diag = M_x - torch.diag(torch.diag(M_x))

    assert torch.allclose(off_diag, torch.zeros_like(off_diag))


def test_exp_log_inverse():
    # Create a simple metric
    M_x = torch.tensor([[4.0, 0.0], [0.0, 9.0]])
    x = torch.tensor([1.0, 2.0])
    v = torch.tensor([0.5, -0.5])

    # Exponential map
    y = torch_exp_map(x, v, lambda _: M_x)

    # Logarithmic map
    v_rec = torch_log_map(x, y, lambda _: M_x)

    # Check that log(exp(v)) recovers original tangent (approximately)
    assert torch.allclose(v, v_rec, atol=1e-6)


def test_metric_changes_with_point():
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    x1 = torch.tensor([0.5, 0.5])
    x2 = torch.tensor([5.0, 5.0])

    M1 = torch_metric(x1, X, sigma=1.0, rho=1e-3)
    M2 = torch_metric(x2, X, sigma=1.0, rho=1e-3)

    # Check that metrics at different points are not identical
    assert not torch.allclose(M1, M2)


def test_normalization_returns_scalar():
    torch.manual_seed(0)

    d = 2
    mu = torch.zeros(d)
    sigma = torch.eye(d)

    metric_fn = lambda x: torch.eye(d)

    C = torch_compute_normalization_constant(mu, sigma, metric_fn, n_samples=500)

    assert isinstance(C, torch.Tensor)
    assert C.ndim == 0


def test_normalization_matches_euclidean_case(monkeypatch):
    torch.manual_seed(0)

    d = 2
    mu = torch.zeros(d)
    sigma = torch.eye(d)

    metric_fn = lambda x: torch.eye(d)

    C = torch_compute_normalization_constant(mu, sigma, metric_fn, n_samples=3000)

    Z = torch.sqrt((2 * torch.pi)**d * torch.det(sigma))

    assert torch.allclose(C, Z, atol=1e-1)


def test_normalization_is_positive():
    torch.manual_seed(0)

    d = 2
    mu = torch.zeros(d)
    sigma = torch.eye(d)

    metric_fn = lambda x: torch.eye(d)

    C = torch_compute_normalization_constant(mu, sigma, metric_fn, n_samples=500)

    assert C > 0


def test_normalization_increases_with_covariance():
    torch.manual_seed(0)

    d = 2
    mu = torch.zeros(d)

    sigma_small = torch.eye(d)
    sigma_large = 2 * torch.eye(d)

    metric_fn = lambda x: torch.eye(d)

    C_small = torch_compute_normalization_constant(
        mu, sigma_small, metric_fn, n_samples=1500
    )
    C_large = torch_compute_normalization_constant(
        mu, sigma_large, metric_fn, n_samples=1500
    )

    assert C_large > C_small
