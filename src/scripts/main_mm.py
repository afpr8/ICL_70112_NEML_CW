import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture
from typing import List

def allocate_dynamic_jax_memory(safety_buffer_mb: int = 500) -> None:
    """Queries GPU 0 and restricts JAX to the currently available free memory."""
    try:
        # Get total and free memory in MB
        smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            encoding="utf-8",
        )
        total_mem, free_mem = map(float, smi_output.strip().split(","))

        # Calculate fraction based on free memory minus a safety buffer
        usable_mem = max(0, free_mem - safety_buffer_mb)
        fraction = usable_mem / total_mem

        # Set the environment variable
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{fraction:.3f}"
        print(f"JAX Memory Fraction set to {fraction:.3f} (~{usable_mem} MB available)")

    except Exception as e:
        print(f"Could not dynamically allocate memory: {e}")

# Must be called before importing JAX to ensure memory is properly capped
allocate_dynamic_jax_memory()
# Disable command buffer to save additional memory
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="

import jax.numpy as jnp
import jax

# Custom module imports
from src.models.mixture_model import LANDMixtureModel
from src.utils.land_utils import (
    RiemannianManifold,
    compute_knn_initial_paths,
)
from src.utils.plotting_utils import plot_full_comparison


def evaluate_land_density(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    mu_list: List[jax.Array],
    sigma_list: List[jax.Array],
    C_list: List[jax.Array],
    pi_list: List[float],
    manifold: RiemannianManifold,
) -> np.ndarray:
    """
    Evaluates the LAND mixture model PDF over a 2D grid for contour plotting,
    filtering out empty space to drastically reduce compute time.
    """
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

    # Filter out empty space using the manifold's KNN tree
    distances = manifold.nn_tree.kneighbors(grid_points, 1, return_distance=True)[0]
    threshold = 0.5 * manifold.sigma
    valid_mask = distances.flatten() < threshold

    valid_grid_points = grid_points[valid_mask]
    valid_grid_tensor = jnp.array(valid_grid_points, dtype=jnp.float32)

    # Initialise an array to hold the sum of all mixture component densities
    Z_valid = jnp.zeros(valid_grid_tensor.shape[0])
    K = len(mu_list)

    for k in range(K):
        mu_np = np.array(mu_list[k])
        
        # Compute paths and geodesics ONLY for valid points for this component
        paths = compute_knn_initial_paths(
            mu_np, valid_grid_points, manifold, N_points=manifold.K_segments + 1
        )
        log_maps = manifold.log_map_batch(
            mu_list[k], valid_grid_tensor, jnp.array(paths)
        )

        inv_sigma = jnp.linalg.inv(sigma_list[k])

        def compute_density(lm):
            dist_sq = jnp.dot(lm, inv_sigma @ lm)
            return (1.0 / C_list[k]) * jnp.exp(-0.5 * dist_sq)

        p_xs = jax.vmap(compute_density)(log_maps)
        Z_valid += pi_list[k] * p_xs

    # Reconstruct the full grid, leaving empty space as 0.0
    full_densities = np.zeros(len(grid_points))
    full_densities[valid_mask] = np.array(Z_valid)

    return full_densities.reshape(X_grid.shape)


def main() -> None:
    # 1. Generate Non-Linear Data (Two Moons)
    X_np, true_labels = make_moons(n_samples=400, noise=0.1, random_state=42)
    X_tensor = jnp.array(X_np, dtype=jnp.float32)

    # Define hyperparams matching the LAND setup
    sigma, rho = 0.3, 1e-3
    K_segments = 10
    init_method = "random"

    # Instantiate the shared manifold structure
    manifold = RiemannianManifold(X_tensor, sigma, rho, K_segments)

    # 2. Fit standard Gaussian Mixture Model
    print("Fitting GMM...")
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(X_np)
    gmm_means = gmm.means_

    # 3. Fit LAND Mixture Model
    print("Fitting LAND Mixture Model...")
    land = LANDMixtureModel(
        K=2, lr_mu=1e-2, lr_A=1e-2, S=3000, epsilon=1e-3, sigma=sigma, rho=rho, K_segments=K_segments, init_method=init_method
    )
    land_mu, land_sigma, land_C, land_pi = land.fit(X_tensor)

    # Convert LAND means to numpy for plotting
    land_means_np = np.array(jnp.stack(land_mu))

    # 4. Assign labels and compute geodesics for LAND
    print("Computing geodesics...")
    labels = []
    geodesics = []
    all_log_maps = []

    # Pre-compute log maps for all components
    for k in range(2):
        m_np = np.array(land_mu[k])
        paths = compute_knn_initial_paths(
            m_np, X_np, manifold, N_points=K_segments + 1
        )
        log_maps_k = manifold.log_map_batch(
            land_mu[k], X_tensor, jnp.array(paths), scaled=False
        )
        all_log_maps.append(log_maps_k)

    for i, x in enumerate(X_tensor):
        # Determine cluster by checking Mahalanobis distance on the manifold
        distances = []
        for k in range(2):
            lm = all_log_maps[k][i]
            inv_sigma = jnp.linalg.inv(land_sigma[k])
            dist_sq = jnp.dot(lm, inv_sigma @ lm).item()
            distances.append(dist_sq)
            
        best_cluster = np.argmin(distances)
        labels.append(best_cluster)
        
        # Generate the visual path for the assigned cluster
        lm_best = all_log_maps[best_cluster][i]
        path = []
        for t in jnp.linspace(0, 1, 10):
            point = manifold.exp_map(land_mu[best_cluster], t * lm_best)
            path.append(np.array(point))
        geodesics.append(np.array(path))

    # 5. Generate Grid for Density Contours
    print("Evaluating grid densities...")
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 10), 
        np.linspace(y_min, y_max, 10)
    ) 
    
    # GMM Contours
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_gmm = np.exp(gmm.score_samples(grid_points)).reshape(xx.shape)
    
    # LAND Contours
    Z_land = evaluate_land_density(
        xx, yy, land_mu, land_sigma, land_C, land_pi, manifold
    )

    # 6. Visualise
    print("Visualising results...")
    fig = plot_full_comparison(
        X=X_np, 
        land_means=land_means_np, 
        gmm_means=gmm_means, 
        labels=np.array(labels), 
        geodesics=geodesics, 
        X_grid=xx, 
        Y_grid=yy, 
        Z_land=Z_land, 
        Z_gmm=Z_gmm,
        init_method=init_method
    )
    
    # Save systematically just like the LANDMLE script
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    n_plots = len(os.listdir(plots_dir))
    plt.savefig(f"{plots_dir}/land_mixture_result_{n_plots}.svg")
    plt.show()


if __name__ == "__main__":
    main()