import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Import your custom modules here
from src.models.land import LANDMLE
from src.utils.land_utils import RiemannianManifold, compute_knn_initial_path
from src.utils.plotting_utils import plot_mixture_contours


def evaluate_land_density(X_grid, Y_grid, mu, sigma, C, manifold, X_data):
    """
    Evaluates the LAND model PDF over a 2D grid for contour plotting.
    """
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]
    grid_tensor = jnp.array(grid_points, dtype=jnp.float32)

    inv_sigma = jnp.linalg.inv(sigma)

    m_np = np.array(mu)
    grid_np = np.array(grid_tensor)
    X_np = np.array(X_data)

    paths = [
        compute_knn_initial_path(
            m_np, grid_np[i], X_np, N_points=manifold.K_segments + 1
        )
        for i in range(grid_np.shape[0])
    ]
    log_maps = manifold.log_map_batch(mu, grid_tensor, jnp.array(paths))

    def compute_density(lm):
        dist_sq = jnp.dot(lm, inv_sigma @ lm)
        return (1.0 / C) * jnp.exp(-0.5 * dist_sq)

    p_xs = jax.vmap(compute_density)(log_maps)
    return np.array(p_xs).reshape(X_grid.shape)


def plot_geodesics(ax, X, mean, geodesics):
    """Plots data points and geodesic paths to the mean."""
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, c="blue", label="Data")

    # Plot all geodesics
    for path in geodesics:
        ax.plot(path[:, 0], path[:, 1], "k-", alpha=0.1)

    ax.scatter(
        [mean[0]], [mean[1]], c="red", marker="x", s=100, linewidth=2, label="LAND mean"
    )
    ax.set_title("LAND Data to Mean Geodesics")
    ax.legend(loc="best")
    ax.axis("equal")


def main():
    # 1. Generate Non-Linear Data (One Moon usually better for a single component, but we'll use a single half-moon)
    X_np, true_labels = make_moons(n_samples=400, noise=0.1, random_state=42)
    # Just take one moon for a single LAND distribution
    X_np = X_np[true_labels == 0]
    X_tensor = jnp.array(X_np, dtype=jnp.float32)

    # Define hyperparams matching the LAND setup
    sigma, rho = 0.15, 1e-3
    K_segments = 10
    manifold = RiemannianManifold(X_tensor, sigma, rho, K_segments)

    # 2. Fit LAND MLE Model
    print("Fitting LAND MLE Model...")
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

    # 3. Compute geodesics for LAND
    print("Computing geodesics...")
    geodesics = []

    m_np = np.array(land_mu)
    paths = [
        compute_knn_initial_path(m_np, X_np[i], X_np, N_points=K_segments + 1)
        for i in range(X_np.shape[0])
    ]
    all_log_maps = manifold.log_map_batch(land_mu, X_tensor, jnp.array(paths))

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
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), # Lower resolution (40x40) to save compute time
                         np.linspace(y_min, y_max, 40)) 
    
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

    plt.savefig("land_result.png")
    print("Saved to land_result.png")

if __name__ == "__main__":
    main()
