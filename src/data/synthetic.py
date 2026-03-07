# Code to generate synthetic data for evaluating LAND

import math
import numpy as np
from sklearn.metrics import pairwise_distances
import torch

def generate_data(data_type: int, N: int, sigma: float = 0.1, r: float = 0.5) -> np.ndarray:
    # The semi-circle data in 2D
    if data_type == 1:
        theta = np.pi * np.random.rand(N, 1)
        data = np.concatenate((np.cos(theta), np.sin(theta)), axis=1) + sigma * np.random.randn(N, 2)
        return data

    # A simple 2-dim surface in 3D with a hole in the middle
    elif data_type == 2:
        Z = np.random.rand(N, 2)
        Z = Z - np.mean(Z, 0).reshape(1, -1)
        Centers = np.zeros((1, 2))
        dists = pairwise_distances(Z, Centers)  # The sqrt(|x|)
        inds_rem = (dists <= r).sum(axis=1)  # N x 1, The points within the ball
        Z_ = Z[inds_rem == 0, :]  # Keep the points OUTSIDE of the ball
        F = (np.sin(2 * np.pi * Z_[:, 0])).reshape(-1, 1)
        F = F + sigma * np.random.randn(F.shape[0], 1)
        data = np.concatenate((Z_, 0.25 * F), axis=1)
        return data

    # Two moons on a surface and with extra noisy dimensions
    elif data_type == 3:
        N_half = N// 2
        theta = np.pi * np.random.rand(N, 1)
        z1 = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
        z2 = np.concatenate((np.cos(theta), -np.sin(theta)), axis=1) + np.array([1.0, 0.25]).reshape(1, -1)
        z = np.concatenate((z1, z2), axis=0) + sigma * np.random.randn(int(N * 2), 2)
        z = z - z.mean(0).reshape(1, -1)
        z3 = (np.sin(np.pi * z[:, 0])).reshape(-1, 1)
        z3 = z3 + sigma * np.random.randn(z3.shape[0], 1)
        data = np.concatenate((z, 0.5 * z3), axis=1)

        labels = np.concatenate((0 * np.ones((z1.shape[0], 1)), np.ones((z2.shape[0], 1))), axis=0)
        return data, labels

    return -1

def sample_non_linear_data(
        n_samples:int=300,
        n_components:int=20,
        x_rad:float=1,
        y_rad:float=1,
        std:float=0.15,
        device="cpu"
    ) -> list[tuple[float, float]]: 
    """
        Generate samples from a mixture of gaussian models centered along
        a half-ellipsoidal curve
        Params:
            n_samples (optional): The number of samples to return
            n_components (optional): The number of Gaussians to sample from
            x_rad (optional): The x-radius of the ellipsoidal curve
            y_rad (optional): The y-radius of the ellipsoidal curve
            std (optional): The std for the Gaussians
            device (optional): The device used for computation
        Returns:
            samples (n_samples, 2): torch.tensor of samples x,y coords
            means (n_components, 2): torch.tensor of Gaussian mean x,y coords
    """
    # Evenly divide the half-ellipsoidal curve for the Gaussian components
    t = torch.linspace(0, math.pi, n_components, device=device)

    # Define the Gaussian means in x,y coords
    means = torch.stack([
        x_rad * torch.cos(t),
        y_rad * torch.sin(t)
    ], dim=1)

    # Create samples
    component_ids = torch.randint(0, n_components, (n_samples, ), device=device)    
    noise = std * torch.randn(n_samples, 2, device=device)
    samples = means[component_ids] + noise

    return samples, means
