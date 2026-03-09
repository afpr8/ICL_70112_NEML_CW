import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
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


# Must be called before importing JAX
allocate_dynamic_jax_memory()
# Disable command buffer to save additional memory
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="


import jax.numpy as jnp
import jax
# Import your custom modules here
from src.models.land import LANDMLE
from src.utils.land_utils import (
    RiemannianManifold,
    compute_knn_initial_paths,
)
from src.utils.plotting_utils import plot_mixture_contours


def evaluate_land_density(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    mu: jax.Array,
    sigma: jax.Array,
    C: jax.Array,
    manifold: RiemannianManifold,
    X_data: jax.Array,
) -> np.ndarray:
    """
    Evaluates the LAND model PDF over a 2D grid for contour plotting.
    """
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

    # Filter out empty space
    distances = manifold.nn_tree.kneighbors(grid_points, 1, return_distance=True)[0]

    threshold = 3.0 * manifold.sigma  # We assume points further than k stds are 0
    valid_mask = distances.flatten() < threshold

    valid_grid_points = grid_points[valid_mask]
    valid_grid_tensor = jnp.array(valid_grid_points, dtype=jnp.float32)
    m_np = np.array(mu)

    # Compute paths and geodesics ONLY for valid points
    paths = compute_knn_initial_paths(
        m_np, valid_grid_points, manifold, N_points=manifold.K_segments + 1
    )
    log_maps = manifold.log_map_batch(mu, valid_grid_tensor, jnp.array(paths))

    inv_sigma = jnp.linalg.inv(sigma)
    def compute_density(lm):
        dist_sq = jnp.dot(lm, inv_sigma @ lm)
        return (1.0 / C) * jnp.exp(-0.5 * dist_sq)

    valid_densities = jax.vmap(compute_density)(log_maps)

    # Reconstruct the full 40x40 grid, leaving empty space as 0.0
    full_densities = np.zeros(len(grid_points))
    full_densities[valid_mask] = np.array(valid_densities)

    return full_densities.reshape(X_grid.shape)


def plot_geodesics(
    ax: plt.Axes, X: np.ndarray, mean: np.ndarray, geodesics: List[np.ndarray]
) -> None:
    """Plots data points and geodesic paths to the mean."""
    ax.scatter(X[:, 0], X[:, 1], s=10, c="lightblue", label="Data")

    # Plot all geodesics
    for path in geodesics:
        ax.plot(path[:, 0], path[:, 1], "g-", alpha=1.0, linewidth=0.2)

    ax.scatter(
        [mean[0]],
        [mean[1]],
        c="orange",
        marker="D",
        s=100,
        linewidth=2,
        label="LAND mean",
    )
    ax.set_title("LAND Data to Mean Geodesics")
    ax.legend(loc="best")
    ax.axis("equal")


def main() -> None:
    # Generate Non-Linear Data (One Moon usually better for a single component, but we'll use a single half-moon)
    X_np, true_labels = make_moons(n_samples=400, noise=0.1, random_state=42)
    # Just take one moon for a single LAND distribution
    X_np = X_np[true_labels == 0]
    X_tensor = jnp.array(X_np, dtype=jnp.float32)

    # Define hyperparams matching the LAND setup
    sigma, rho = 0.15, 1e-3
    K_segments = 10
    manifold = RiemannianManifold(X_tensor, sigma, rho, K_segments)

    # Fit LAND MLE model
    print("Fitting LAND MLE model...")
    land = LANDMLE(
        initial_lr_mu=1e-2,
        initial_lr_A=1e-2,
        S=3000,
        epsilon=1e-3,
        sigma=sigma,
        rho=rho,
        K_segments=K_segments,
    )
    land_mu, land_sigma, land_C = land.fit(X_tensor)

    # Convert LAND means to numpy for plotting
    land_means_np = np.array(land_mu).reshape(1, -1)

    # Compute geodesics for LAND
    print("Computing geodesics...")
    m_np = np.array(land_mu)
    # Use batched path computation
    paths = compute_knn_initial_paths(m_np, X_np, manifold, N_points=K_segments + 1)
    all_log_maps = manifold.log_map_batch(
        land_mu, X_tensor, jnp.array(paths), scaled=False
    )

    geodesics = []
    for i, x in enumerate(X_tensor):
        lm = all_log_maps[i]
        path = []
        for t in jnp.linspace(0, 1, 10):
            point = manifold.exp_map(land_mu, t * lm)
            path.append(np.array(point))
        geodesics.append(np.array(path))

    # 4. Generate Grid for Density Contours
    print("Evaluating grid densities...")
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))

    # LAND Contours
    Z_land = evaluate_land_density(
        xx, yy, land_mu, land_sigma, land_C, manifold, X_tensor
    )

    # 5. Visualise
    print("Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_geodesics(axes[0], X_np, land_means_np[0], geodesics)
    plot_mixture_contours(axes[1], X_np, land_means_np, xx, yy, Z_land, 
                          title='LAND MLE PDF', mean_label='LAND mean')

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    n_plots = len(os.listdir(plots_dir))
    plt.savefig(f"{plots_dir}/land_result_{n_plots}.svg")

if __name__ == "__main__":
    main()
