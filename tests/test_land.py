import pytest
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

from land import LANDMLE

@pytest.fixture
def land_model():
    return LANDMLE(initial_lr_mu=0.1, initial_lr_A=0.1, S=10, epsilon=1e-4, seed=42)

@pytest.fixture
def dummy_data():
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (5, 2))

def test_init(land_model):
    """Test correctly initialized parameters."""
    assert land_model.lr_mu == 0.1
    assert land_model.lr_A == 0.1
    assert land_model.S == 10
    assert land_model.epsilon == 1e-4

@patch("land.jax_log_map_shooting")
def test_init_params_random(mock_log_map, land_model, dummy_data):
    """Test random initialization."""
    mock_log_map.side_effect = [
        jnp.array([1.0, 1.0]) for _ in range(dummy_data.shape[0])
    ]
    key = jax.random.PRNGKey(42)
    mu, A, sigma = land_model._init_params(dummy_data, key=key, method="random")
    assert mu.shape == (2,)
    assert sigma.shape == (2, 2)
    assert A.shape == (2, 2)

@patch("land.jax_log_map_shooting")
def test_init_params_mean(mock_log_map, land_model, dummy_data):
    """Test mean initialization."""
    mock_log_map.side_effect = [
        jnp.array([1.0, 1.0]) for _ in range(dummy_data.shape[0])
    ]
    key = jax.random.PRNGKey(42)
    mu, A, sigma = land_model._init_params(dummy_data, key=key, method="mean")
    expected_mu = jnp.mean(dummy_data, axis=0)
    assert jnp.allclose(mu, expected_mu)
    assert sigma.shape == (2, 2)
    assert A.shape == (2, 2)

def test_compute_A(land_model):
    """Test calculation of A from sigma."""
    sigma = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    A = land_model.compute_A(sigma)
    expected_A = jnp.linalg.cholesky(jnp.linalg.inv(sigma)).T
    assert jnp.allclose(A, expected_A)

@patch("land.jax_log_map_shooting")
def test_loss(mock_log_map, land_model, dummy_data):
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

@patch("land.jax_metric")
@patch("land.jax_exp_map")
def test_m(mock_exp_map, mock_metric, land_model, dummy_data):
    """Test the metric deformation calculation _m."""
    mock_exp_map.return_value = jnp.zeros(2)
    mock_metric.return_value = jnp.eye(2) * 4.0
    land_model._metric = mock_metric

    mu = jnp.zeros(2)
    v = jnp.ones(2)

    val = land_model._m(mu, v)
    assert jnp.allclose(val, jnp.array(4.0))

@patch("land.jax_metric")
@patch("land.jax_exp_map")
@patch("land.jax_log_map_shooting")
def test_compute_grad_mu(
    mock_log_map, mock_exp_map, mock_metric, land_model, dummy_data
):
    """Test gradient computation with respect to mu."""
    mock_log_map.return_value = jnp.array([1.0, 0.0])
    mock_exp_map.return_value = jnp.zeros(2)
    mock_metric.return_value = jnp.eye(2)
    land_model._metric = mock_metric
    key = jax.random.PRNGKey(42)

    mu = jnp.zeros(2)
    sigma = jnp.eye(2)
    normalization_constant = jnp.array(1.0)
    log_maps = jnp.ones((dummy_data.shape[0], 2))

    grad = land_model._compute_grad_mu(mu, sigma, normalization_constant, key, log_maps)
    assert grad.shape == mu.shape

@patch("land.jax_metric")
@patch("land.jax_exp_map")
@patch("land.jax_log_map_shooting")
def test_compute_grad_sigma(
    mock_log_map, mock_exp_map, mock_metric, land_model, dummy_data
):
    """Test gradient computation with respect to sigma."""
    mock_log_map.return_value = jnp.array([1.0, 0.0])
    mock_exp_map.return_value = jnp.zeros(2)
    mock_metric.return_value = jnp.eye(2)
    land_model._metric = mock_metric
    key = jax.random.PRNGKey(42)

    mu = jnp.zeros(2)
    A = jnp.eye(2)
    sigma = jnp.eye(2)
    normalization_constant = jnp.array(1.0)
    log_maps = jnp.ones((dummy_data.shape[0], 2))

    try:
        grad = land_model._compute_grad_sigma(
            mu, A, sigma, normalization_constant, key, log_maps
        )
        assert grad.shape == sigma.shape
    except Exception as e:
        pytest.fail(f"grad_sigma raised an unexpected exception: {e}")

@patch("land.compute_normalization_constant")
@patch("land.jax_metric")
@patch("land.jax_exp_map")
@patch("land.jax_log_map_shooting")
def test_fit_learning_rates_and_convergence(
    mock_log_map, mock_exp_map, mock_metric, mock_compute_norm, land_model, dummy_data
):
    """Test the fit wrapper runs correctly, adapts learning rates and stops when loss diff < epsilon."""
    # Setup mocks
    mock_log_map.side_effect = lambda mu, x, m: jax.random.normal(
        jax.random.PRNGKey(0), (2,)
    )
    mock_exp_map.return_value = jnp.zeros(2)
    mock_metric.return_value = jnp.eye(2)
    mock_compute_norm.return_value = jnp.array(1.0)

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
