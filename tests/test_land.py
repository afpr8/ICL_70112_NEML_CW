import pytest
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

from src.models.land import LANDMLE
from src.utils.land_utils import RiemannianManifold

@pytest.fixture
def land_model():
    return LANDMLE(initial_lr_mu=0.1, initial_lr_A=0.1, S=10, epsilon=1e-4, seed=42)

@pytest.fixture
def dummy_data():
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (5, 2))

@pytest.fixture
def mock_manifold():
    manifold = MagicMock()  # Can't use spec easily due to abstract python logic
    manifold.exp_map.return_value = jnp.zeros(2)
    manifold.metric.return_value = jnp.eye(2)
    manifold.log_map_batch.return_value = jnp.ones((5, 2))
    manifold.compute_normalization_constant.return_value = jnp.array(1.0)
    return manifold


def test_init(land_model):
    """Test correctly initialized parameters."""
    assert land_model.lr_mu == 0.1
    assert land_model.lr_A == 0.1
    assert land_model.S == 10
    assert land_model.epsilon == 1e-4

@patch("src.models.land.LANDMLE._compute_log_maps")
def test_init_params_random(
    mock_compute_log_maps, land_model, dummy_data, mock_manifold
):
    """Test random initialization."""
    mock_compute_log_maps.return_value = jnp.ones((dummy_data.shape[0], 2))
    key = jax.random.PRNGKey(42)
    mu, A, sigma = land_model._init_params(
        dummy_data, key=key, method="random", manifold=mock_manifold
    )
    assert mu.shape == (2,)
    assert sigma.shape == (2, 2)
    assert A.shape == (2, 2)

@patch("src.models.land.LANDMLE._compute_log_maps")
def test_init_params_mean(mock_compute_log_maps, land_model, dummy_data, mock_manifold):
    """Test mean initialization."""
    mock_compute_log_maps.return_value = jnp.ones((dummy_data.shape[0], 2))
    key = jax.random.PRNGKey(42)
    mu, A, sigma = land_model._init_params(
        dummy_data, key=key, method="mean", manifold=mock_manifold
    )
    expected_mu = dummy_data[
        jnp.argmin(jnp.sum((dummy_data - jnp.mean(dummy_data, axis=0)) ** 2, axis=1))
    ]
    assert jnp.allclose(mu, expected_mu)
    assert sigma.shape == (2, 2)
    assert A.shape == (2, 2)

def test_compute_A(land_model):
    """Test calculation of A from sigma."""
    sigma = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    A = land_model.compute_A(sigma)
    expected_A = jnp.linalg.cholesky(jnp.linalg.inv(sigma)).T
    assert jnp.allclose(A, expected_A)

def test_loss(land_model, dummy_data):
    """Test loss computation."""
    # Precomputed log_maps: shape (N, d), each row is [1.0, 1.0]
    log_maps = jnp.ones((dummy_data.shape[0], 2))

    sigma = jnp.eye(2)
    normalization_constant = jnp.array(2.0)

    loss_val = land_model._loss(sigma, log_maps, normalization_constant)
    # Expected: N=5. sum of terms = 5 * (1*1 + 1*1) = 10.
    # objective = 10 / (2*5) + log(2) = 1 + log(2)
    expected_obj = 1.0 + jnp.log(jnp.array(2.0))
    assert jnp.allclose(loss_val, expected_obj)

def test_compute_grad_mu(land_model, dummy_data, mock_manifold):
    """Test gradient computation with respect to mu."""
    key = jax.random.PRNGKey(42)
    mu = jnp.zeros(2)
    sigma = jnp.eye(2)
    normalization_constant = jnp.array(1.0)
    log_maps = jnp.ones((dummy_data.shape[0], 2))

    grad = land_model._compute_grad_mu(
        mu, sigma, normalization_constant, key, log_maps, mock_manifold
    )
    assert grad.shape == mu.shape

def test_compute_grad_sigma(land_model, dummy_data, mock_manifold):
    """Test gradient computation with respect to sigma."""
    key = jax.random.PRNGKey(42)
    mu = jnp.zeros(2)
    A = jnp.eye(2)
    sigma = jnp.eye(2)
    normalization_constant = jnp.array(1.0)
    log_maps = jnp.ones((dummy_data.shape[0], 2))

    try:
        grad = land_model._compute_grad_sigma(
            mu, A, sigma, normalization_constant, key, log_maps, mock_manifold
        )
        assert grad.shape == sigma.shape
    except Exception as e:
        pytest.fail(f"grad_sigma raised an unexpected exception: {e}")

@patch("src.models.land.RiemannianManifold")
@patch("src.models.land.LANDMLE._compute_log_maps")
def test_fit_learning_rates_and_convergence(
    mock_compute_log_maps, mock_manifold_class, land_model, dummy_data, mock_manifold
):
    """Test the fit wrapper runs correctly, adapts learning rates and stops when loss diff < epsilon."""
    # Setup mocks
    mock_manifold_class.return_value = mock_manifold

    # We must patch the random output correctly for dummy log_maps across the iteration calls
    mock_compute_log_maps.side_effect = lambda *args, **kwargs: jax.random.normal(
        jax.random.PRNGKey(0), (dummy_data.shape[0], 2)
    )

    land_model._loss = MagicMock(
        side_effect=[
            jnp.array(2.5),
            jnp.array(2.0),
            jnp.array(3.0),
            jnp.array(2.0),
            jnp.array(3.0),
            jnp.array(3.0),
            jnp.array(3.0 + 1e-5),
            jnp.array(3.0),
        ]
    )

    initial_lr_mu = land_model.lr_mu
    initial_lr_A = land_model.lr_A

    mu_final, sigma_final, norm_final = land_model.fit(dummy_data)
    
    assert mu_final.shape == (2,)
    assert sigma_final.shape == (2, 2)
    assert norm_final == jnp.array(1.0)

    assert land_model.lr_mu == pytest.approx(initial_lr_mu * 1.1 * 1.1)
    assert land_model.lr_A == pytest.approx(initial_lr_A * 0.75 * 0.75)

    assert land_model._loss.call_count == 8
