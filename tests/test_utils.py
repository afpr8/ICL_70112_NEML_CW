# A testing file for the LAND algorithm utilities functions

import pytest
import torch

from utils import (
    compute_normalization_constant,
    exp_map,
    exp_map_geodesic,
    geodesic_ode,
    log_map,
    log_map_geodesic,
    metric
)


def test_metric_shape_and_pd():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = torch.tensor([2.0, 3.0])
    M_x = metric(x, X, sigma=1.0, rho=1e-3)

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
    y = exp_map(x, v, M_x)

    # Logarithmic map
    v_rec = log_map(x, y, M_x)

    # Check that log(exp(v)) recovers original tangent (approximately)
    assert torch.allclose(v, v_rec, atol=1e-6)


def test_metric_changes_with_point():
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    x1 = torch.tensor([0.5, 0.5])
    x2 = torch.tensor([5.0, 5.0])

    M1 = metric(x1, X, sigma=1.0, rho=1e-3)
    M2 = metric(x2, X, sigma=1.0, rho=1e-3)

    # Check that metrics at different points are not identical
    assert not torch.allclose(M1, M2)


def test_normalization_returns_scalar():
    torch.manual_seed(0)

    d = 2
    mu = torch.zeros(d)
    sigma = torch.eye(d)

    metric_fn = lambda x: torch.eye(d)

    C = compute_normalization_constant(mu, sigma, metric_fn, n_samples=500)

    assert isinstance(C, torch.Tensor)
    assert C.ndim == 0


def test_normalization_matches_euclidean_case(monkeypatch):
    torch.manual_seed(0)

    d = 2
    mu = torch.zeros(d)
    sigma = torch.eye(d)

    metric_fn = lambda x: torch.eye(d)

    C = compute_normalization_constant(mu, sigma, metric_fn, n_samples=3000)

    Z = torch.sqrt((2 * torch.pi)**d * torch.det(sigma))

    assert torch.allclose(C, Z, atol=1e-1)


def test_normalization_is_positive():
    torch.manual_seed(0)

    d = 2
    mu = torch.zeros(d)
    sigma = torch.eye(d)

    metric_fn = lambda x: torch.eye(d)

    C = compute_normalization_constant(mu, sigma, metric_fn, n_samples=500)

    assert C > 0


def test_normalization_increases_with_covariance():
    torch.manual_seed(0)

    d = 2
    mu = torch.zeros(d)

    sigma_small = torch.eye(d)
    sigma_large = 2 * torch.eye(d)

    metric_fn = lambda x: torch.eye(d)

    C_small = compute_normalization_constant(mu, sigma_small, metric_fn, n_samples=1500)
    C_large = compute_normalization_constant(mu, sigma_large, metric_fn, n_samples=1500)

    assert C_large > C_small


# Dummy diagonal metric for testing
def dummy_metric(x: torch.Tensor) -> torch.Tensor:
    # Just return a diagonal matrix with entries depending on x
    return torch.diag(1.0 + 0.1 * x**2)


def test_geodesic_ode_shapes():
    d = 3
    x = torch.rand(d)
    v = torch.rand(d)
    y = torch.cat([x, v])
    
    dy = geodesic_ode(torch.tensor(0.0), y, dummy_metric)
    assert dy.shape == (2*d,), "geodesic_ode should return a tensor of shape 2*d"
    assert isinstance(dy, torch.Tensor)


def test_exp_map_geodesic_returns_correct_shape():
    d = 4
    x = torch.rand(d)
    v = torch.rand(d)
    x_end = exp_map_geodesic(x, v, dummy_metric)
    
    assert x_end.shape == x.shape, "exp_map_geodesic should return tensor of same shape as x"
    assert isinstance(x_end, torch.Tensor)


def test_exp_log_inverse_consistency():
    # Test that log_map_geodesic approximately inverts exp_map_geodesic
    d = 2
    x = torch.rand(d)
    v = torch.rand(d) * 0.1  # small vector to stay in approx linear region
    x_end = exp_map_geodesic(x, v, dummy_metric)
    v_recov = log_map_geodesic(x, x_end, dummy_metric, n_iter=100, lr=0.05, tol=1e-8)
    
    assert torch.allclose(v, v_recov, atol=1e-3), "log_map_geodesic should invert exp_map_geodesic approximately"


def test_log_map_geodesic_shape():
    d = 3
    x = torch.rand(d)
    y = torch.rand(d)
    v = log_map_geodesic(x, y, dummy_metric, n_iter=5, lr=0.01)
    
    assert v.shape == x.shape, "log_map_geodesic should return a tangent vector of same shape as x"
    assert isinstance(v, torch.Tensor)


def test_zero_tangent_vector_identity():
    d = 3
    x = torch.rand(d)
    v = torch.zeros(d)
    x_end = exp_map_geodesic(x, v, dummy_metric)
    
    # Zero vector should map back to same point
    assert torch.allclose(x, x_end), "Exponential map of zero vector should return base point"


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_exp_map_geodesic_random_vectors(dim):
    x = torch.rand(dim)
    v = torch.rand(dim) * 0.1
    x_end = exp_map_geodesic(x, v, dummy_metric)
    # Sanity check: output is finite
    assert torch.isfinite(x_end).all(), "exp_map_geodesic should return finite values"
