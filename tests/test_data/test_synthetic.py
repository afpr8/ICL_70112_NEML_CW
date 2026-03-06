# A testing file for the synthetic data generation code

import math

import torch

from src.data.synthetic import sample_non_linear_data

def test_shapes():
    n_samples = 100
    n_components = 10
    samples, means = sample_non_linear_data(n_samples=n_samples, n_components=n_components)
    
    # Check types
    assert isinstance(samples, torch.Tensor), "samples should be a torch.Tensor"
    assert isinstance(means, torch.Tensor), "means should be a torch.Tensor"
    
    # Check shapes
    assert samples.shape == (n_samples, 2), f"samples shape should be ({n_samples},2)"
    assert means.shape == (n_components, 2), f"means shape should be ({n_components},2)"


def test_means_on_half_ellipse():
    n_components = 50
    x_rad, y_rad = 2.0, 3.0
    _, means = sample_non_linear_data(n_components=n_components, x_rad=x_rad, y_rad=y_rad)
    
    # t should go from 0 to pi
    expected_x = x_rad * torch.cos(torch.linspace(0, math.pi, n_components))
    expected_y = y_rad * torch.sin(torch.linspace(0, math.pi, n_components))
    
    # Allow a tiny tolerance for floating point
    assert torch.allclose(means[:,0], expected_x, atol=1e-6), "x-coords of means are incorrect"
    assert torch.allclose(means[:,1], expected_y, atol=1e-6), "y-coords of means are incorrect"


def test_std_effect():
    n_samples = 1000
    std = 0.0
    samples, means = sample_non_linear_data(n_samples=n_samples, std=std)
    
    # If std=0, all samples should exactly match one of the means
    # Check that all sample points exist in the means
    for sample in samples:
        found_match = any(torch.allclose(sample, m, atol=1e-6) for m in means)
        assert found_match, "Sample does not match any mean when std=0"
