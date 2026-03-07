# Tests for JAX implementations of the exponential and logarithmic maps

import pytest
import jax.numpy as jnp
import numpy as np
import torch

from src.utils.land_utils import RiemannianManifold
from src.utils.legacy_torch_utils import (
    torch_metric,
    torch_exp_map,
)


@pytest.fixture
def jax_setup():
    """Shared fixture: small 2D synthetic data with fixed parameters."""
    X = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    x = jnp.array([1.0, 1.0])
    sigma = 1.0
    rho = 1e-3
    manifold = RiemannianManifold(X, sigma, rho, K_segments=5)
    return manifold, X, x, sigma, rho


# ──────────────────────────────────────────────
# jax_metric tests
# ──────────────────────────────────────────────

class TestJaxMetric:
    def test_shape_and_positive_diagonal(self, jax_setup):
        """Metric output should be (d, d) with positive diagonal entries."""
        manifold, X, x, sigma, rho = jax_setup
        M = manifold.metric(x)

        assert M.shape == (2, 2)
        assert jnp.all(jnp.diag(M) > 0)

    def test_off_diagonal_is_zero(self, jax_setup):
        """The LAND metric is diagonal, so off-diagonal entries must be zero."""
        manifold, X, x, sigma, rho = jax_setup
        M = manifold.metric(x)

        off_diag = M - jnp.diag(jnp.diag(M))
        assert jnp.allclose(off_diag, jnp.zeros_like(off_diag))

    def test_consistency_with_torch_metric(self, jax_setup):
        """JAX metric should match the PyTorch metric on the same inputs."""
        manifold, X_jax, x_jax, sigma, rho = jax_setup

        X_torch = torch.tensor(np.array(X_jax))
        x_torch = torch.tensor(np.array(x_jax))

        M_jax = np.array(manifold.metric(x_jax))
        M_torch = torch_metric(x_torch, X_torch, sigma, rho).numpy()

        np.testing.assert_allclose(M_jax, M_torch, atol=1e-6)


# ──────────────────────────────────────────────
# jax_exp_map tests
# ──────────────────────────────────────────────

class TestJaxExpMap:
    def test_zero_tangent_returns_base_point(self, jax_setup):
        """Exponential map with zero tangent vector should return the base point."""
        manifold, X, x, sigma, rho = jax_setup
        v_zero = jnp.zeros_like(x)
        result = manifold.exp_map(x, v_zero)

        assert jnp.allclose(result, x, atol=1e-5)

    def test_output_shape(self, jax_setup):
        """Output of exp map should have the same shape as the base point."""
        manifold, X, x, sigma, rho = jax_setup
        v = jnp.array([0.1, -0.2])
        result = manifold.exp_map(x, v)

        assert result.shape == x.shape

    def test_small_tangent_stays_near_base(self, jax_setup):
        """For a very small tangent vector, the result should be close to x."""
        manifold, X, x, sigma, rho = jax_setup
        v_small = jnp.array([1e-6, 1e-6])
        result = manifold.exp_map(x, v_small)

        assert jnp.linalg.norm(result - x) < 1e-4

    def test_consistency_with_torch_exp_map(self, jax_setup):
        """
        For the simplified diagonal metric used in both implementations,
        the JAX ODE-based exp map should yield a result close to the
        closed-form PyTorch exp map.
        """
        manifold, X_jax, x_jax, sigma, rho = jax_setup
        v_jax = jnp.array([0.1, -0.1])

        X_torch = torch.tensor(np.array(X_jax))
        x_torch = torch.tensor(np.array(x_jax))
        v_torch = torch.tensor(np.array(v_jax))

        def metric_fn(pt):
            return torch_metric(pt, X_torch, sigma, rho)

        result_jax = np.array(manifold.exp_map(x_jax, v_jax))
        result_torch = torch_exp_map(x_torch, v_torch, metric_fn).numpy()

        np.testing.assert_allclose(result_jax, result_torch, atol=0.15)


# ──────────────────────────────────────────────
# jax_log_map_shooting tests
# ──────────────────────────────────────────────

class TestLogMapShooting:
    def test_roundtrip_exp_then_log(self, jax_setup):
        """log(x, exp(x, v)) should approximately recover v."""
        manifold, X, x, sigma, rho = jax_setup
        v = jnp.array([0.05, -0.05])

        y = manifold.exp_map(x, v)
        initial_path = jnp.linspace(x, y, manifold.K_segments + 1)
        v_recovered = manifold.log_map_shooting(x, y, initial_path)

        np.testing.assert_allclose(np.array(v_recovered), np.array(v), atol=1e-2)

    def test_log_of_same_point_is_zero(self, jax_setup):
        """log(x, x) should be the zero vector."""
        manifold, X, x, sigma, rho = jax_setup
        initial_path = jnp.linspace(x, x, manifold.K_segments + 1)
        v = manifold.log_map_shooting(x, x, initial_path)

        assert jnp.linalg.norm(v) < 1e-4

    def test_output_shape(self, jax_setup):
        """Output of log map should have the same shape as the base point."""
        manifold, X, x, sigma, rho = jax_setup
        y = jnp.array([1.5, 0.5])
        initial_path = jnp.linspace(x, y, manifold.K_segments + 1)
        v = manifold.log_map_shooting(x, y, initial_path)

        assert v.shape == x.shape
