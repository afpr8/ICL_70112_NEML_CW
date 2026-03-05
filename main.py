import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from functools import partial


# Import your custom modules here
from mixture_model import LANDMixtureModel
from utils import log_map, exp_map, metric, plot_full_comparison

def get_geodesic_path(mu, x, metric_fn, steps=10):
    """
    Computes a sequence of points along the geodesic from cluster mean (mu) to data point (x).
    """
    v = log_map(mu, x, metric_fn)
    path = []
    # Interpolate from t=0 (at mu) to t=1 (at x)
    for t in torch.linspace(0, 1, steps):
        point = exp_map(mu, t * v, metric_fn)
        path.append(point.detach().numpy())
    return np.array(path)

def evaluate_land_density(X_grid, Y_grid, mu_list, sigma_list, C_list, pi_list, metric_fn):
    """
    Evaluates the LAND mixture model PDF over a 2D grid for contour plotting.
    """
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    Z = torch.zeros(grid_tensor.shape[0])
    
    K = len(mu_list)
    
    # Calculate probability density for each grid point
    for k in range(K):
        inv_sigma = torch.linalg.inv(sigma_list[k])
        for i, x in enumerate(grid_tensor):
            lm = log_map(mu_list[k], x, metric_fn)
            dist_sq = torch.dot(lm, inv_sigma @ lm)
            
            p_x = (1.0 / C_list[k]) * torch.exp(-0.5 * dist_sq)
            Z[i] += pi_list[k] * p_x
            
    return Z.numpy().reshape(X_grid.shape)

def main():
    # 1. Generate Non-Linear Data (Two Moons)
    X_np, true_labels = make_moons(n_samples=400, noise=0.1, random_state=42)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    
    # Define hyperparams matching the LAND setup
    sigma, rho = 1.0, 1e-3
    metric_fn = partial(metric, X=X_tensor, sigma=sigma, rho=rho)

    # 2. Fit standard Gaussian Mixture Model
    print("Fitting GMM...")
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(X_np)
    gmm_means = gmm.means_

    # 3. Fit LAND Mixture Model
    print("Fitting LAND Mixture Model...")
    land = LANDMixtureModel(K=2, lr_mu=1e-2, lr_A=1e-2, S=50, epsilon=1e-3, sigma=sigma, rho=rho)
    land_mu, land_sigma, land_C, land_pi = land.fit(X_tensor)
    
    # Convert LAND means to numpy for plotting
    land_means_np = torch.stack(land_mu).detach().numpy()

    # 4. Assign labels and compute geodesics for LAND
    print("Computing geodesics...")
    labels = []
    geodesics = []
    
    for x in X_tensor:
        # Determine which cluster 'x' belongs to by checking Mahalanobis distance on the manifold
        distances = []
        for k in range(2):
            lm = log_map(land_mu[k], x, metric_fn)
            inv_sigma = torch.linalg.inv(land_sigma[k])
            dist_sq = torch.dot(lm, inv_sigma @ lm).item()
            distances.append(dist_sq)
            
        best_cluster = np.argmin(distances)
        labels.append(best_cluster)
        
        # Generate the visual path
        path = get_geodesic_path(land_mu[best_cluster], x, metric_fn)
        geodesics.append(path)

    # 5. Generate Grid for Density Contours
    print("Evaluating grid densities...")
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), # Lower resolution (40x40) to save compute time
                         np.linspace(y_min, y_max, 40)) 
    
    # GMM Contours
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_gmm = np.exp(gmm.score_samples(grid_points)).reshape(xx.shape)
    
    # LAND Contours
    Z_land = evaluate_land_density(xx, yy, land_mu, land_sigma, land_C, land_pi, metric_fn)

    # 6. Visualise
    print("Plotting results...")
    fig = plot_full_comparison(
        X=X_np, 
        land_means=land_means_np, 
        gmm_means=gmm_means, 
        labels=np.array(labels), 
        geodesics=geodesics, 
        X_grid=xx, 
        Y_grid=yy, 
        Z_land=Z_land, 
        Z_gmm=Z_gmm
    )
    
    plt.show()

if __name__ == "__main__":
    main()