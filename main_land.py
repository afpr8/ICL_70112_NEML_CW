import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from functools import partial

# Import your custom modules here
from land import LANDMLE
from utils import jax_log_map_shooting, jax_exp_map, jax_metric, plot_mixture_contours

def get_geodesic_path(mu, x, metric_fn, steps=10):
    """
    Computes a sequence of points along the geodesic from cluster mean (mu) to data point (x).
    """
    v = jax_log_map_shooting(mu, x, metric_fn)
    path = []
    # Interpolate from t=0 (at mu) to t=1 (at x)
    for t in jnp.linspace(0, 1, steps):
        point = jax_exp_map(mu, t * v, metric_fn)
        path.append(np.array(point))
    return np.array(path)

def evaluate_land_density(X_grid, Y_grid, mu, sigma, C, metric_fn):
    """
    Evaluates the LAND model PDF over a 2D grid for contour plotting.
    """
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]
    grid_tensor = jnp.array(grid_points, dtype=jnp.float32)
    
    inv_sigma = jnp.linalg.inv(sigma)

    def compute_px(x):
        lm = jax_log_map_shooting(mu, x, metric_fn)
        dist_sq = jnp.dot(lm, inv_sigma @ lm)
        return (1.0 / C) * jnp.exp(-0.5 * dist_sq)

    p_xs = jax.vmap(compute_px)(grid_tensor)
    return np.array(p_xs).reshape(X_grid.shape)

def plot_geodesics(ax, X, mean, geodesics):
    """Plots data points and geodesic paths to the mean."""
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, c='blue', label='Data')
    
    # Plot all geodesics
    for path in geodesics:
        ax.plot(path[:, 0], path[:, 1], 'k-', alpha=0.1)
        
    ax.scatter([mean[0]], [mean[1]], c='red', marker='x', s=100, linewidth=2, label='LAND mean')
    ax.set_title('LAND Data to Mean Geodesics')
    ax.legend(loc='best')
    ax.axis('equal')

def main():
    # 1. Generate Non-Linear Data (One Moon usually better for a single component, but we'll use a single half-moon)
    X_np, true_labels = make_moons(n_samples=400, noise=0.1, random_state=42)
    # Just take one moon for a single LAND distribution
    X_np = X_np[true_labels == 0]
    X_tensor = jnp.array(X_np, dtype=jnp.float32)
    
    # Define hyperparams matching the LAND setup
    sigma, rho = 1.0, 1e-3
    metric_fn = partial(jax_metric, X=X_tensor, sigma=sigma, rho=rho)

    # 2. Fit LAND MLE Model
    print("Fitting LAND MLE Model...")
    land = LANDMLE(initial_lr_mu=1e-2, initial_lr_A=1e-2, S=50, epsilon=1e-3, sigma=sigma, rho=rho)
    land_mu, land_sigma, land_C = land.fit(X_tensor)
    
    # Convert LAND means to numpy for plotting
    land_means_np = np.array(land_mu).reshape(1, -1)

    # 3. Compute geodesics for LAND
    print("Computing geodesics...")
    geodesics = []
    
    for x in X_tensor:
        # Generate the visual path
        path = get_geodesic_path(land_mu, x, metric_fn)
        geodesics.append(path)

    # 4. Generate Grid for Density Contours
    print("Evaluating grid densities...")
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), # Lower resolution (40x40) to save compute time
                         np.linspace(y_min, y_max, 40)) 
    
    # LAND Contours
    Z_land = evaluate_land_density(xx, yy, land_mu, land_sigma, land_C, metric_fn)

    # 5. Visualise
    print("Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_geodesics(axes[0], X_np, land_means_np[0], geodesics)
    plot_mixture_contours(axes[1], X_np, land_means_np, xx, yy, Z_land, 
                          title='LAND MLE PDF', mean_label='LAND mean')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
