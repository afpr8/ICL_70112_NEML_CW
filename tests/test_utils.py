# A testing file for the LAND algorithm utilities functions

import torch

from utils import exp_map, log_map, metric


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
