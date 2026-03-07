import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from functools import partial

# Import your custom modules here
from src.models.mixture_model import LANDMixtureModel
from src.utils.land_utils import (
    compute_log_map_batch,
    jax_exp_map,
    jax_metric,
)
from src.utils.plotting_utils import plot_full_comparison


def evaluate_land_density(
    X_grid, Y_grid, mu_list, sigma_list, C_list, pi_list, X_tensor, metric_fn
):
    """
    Evaluates the LAND mixture model PDF over a 2D grid for contour plotting.
    """
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]
    grid_tensor = jnp.array(grid_points, dtype=jnp.float32)
    Z = jnp.zeros(grid_tensor.shape[0])
    
    K = len(mu_list)

    for k in range(K):
        inv_sigma = jnp.linalg.inv(sigma_list[k])

        log_maps = compute_log_map_batch(
            mu_list[k],
            grid_tensor,
            X_tensor,
            metric_fn,
            method="energy_shooting",
            n_points=30,
            n_neighbors=7,
        )

        def compute_density(lm):
            dist_sq = jnp.dot(lm, inv_sigma @ lm)
            return (1.0 / C_list[k]) * jnp.exp(-0.5 * dist_sq)

        p_xs = jax.vmap(compute_density)(log_maps)
        Z += pi_list[k] * p_xs

    return np.array(Z).reshape(X_grid.shape)


def main():
    # 1. Generate Non-Linear Data (Two Moons)
    X_np, true_labels = make_moons(n_samples=400, noise=0.1, random_state=42)
    X_tensor = jnp.array(X_np, dtype=jnp.float32)

    # Define hyperparams matching the LAND setup
    sigma, rho = 0.4, 1e-3
    metric_fn = partial(jax_metric, X=X_tensor, sigma=sigma, rho=rho)

    # 2. Fit standard Gaussian Mixture Model
    print("Fitting GMM...")
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(X_np)
    gmm_means = gmm.means_

    # 3. Fit LAND Mixture Model
    print("Fitting LAND Mixture Model...")
    land = LANDMixtureModel(
        K=2, lr_mu=1e-2, lr_A=1e-2, S=50, epsilon=1e-3, sigma=sigma, rho=rho
    )
    land_mu, land_sigma, land_C, land_pi = land.fit(X_tensor)

    # Convert LAND means to numpy for plotting
    land_means_np = np.array(jnp.stack(land_mu))

    # 4. Assign labels and compute geodesics for LAND
    print("Computing geodesics...")
    labels = []
    geodesics = []

    all_log_maps = []
    for k in range(2):
        all_log_maps.append(
            compute_log_map_batch(
                land_mu[k],
                X_tensor,
                X_tensor,
                metric_fn,
                method="energy_shooting",
                n_points=30,
                n_neighbors=7,
            )
        )

    for i, x in enumerate(X_tensor):
        # Determine which cluster 'x' belongs to by checking Mahalanobis distance on the manifold
        distances = []
        for k in range(2):
            lm = all_log_maps[k][i]
            inv_sigma = jnp.linalg.inv(land_sigma[k])
            dist_sq = jnp.dot(lm, inv_sigma @ lm).item()
            distances.append(dist_sq)
            
        best_cluster = np.argmin(distances)
        labels.append(best_cluster)
        
        # Generate the visual path
        lm_best = all_log_maps[best_cluster][i]
        path = []
        for t in jnp.linspace(0, 1, 10):
            point = jax_exp_map(land_mu[best_cluster], t * lm_best, metric_fn)
            path.append(np.array(point))
        geodesics.append(np.array(path))

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
    Z_land = evaluate_land_density(
        xx, yy, land_mu, land_sigma, land_C, land_pi, X_tensor, metric_fn
    )

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
