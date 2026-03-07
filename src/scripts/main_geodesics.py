import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import time

from src.data.synthetic import sample_non_linear_data
from src.utils.land_utils import RiemannianManifold, compute_knn_initial_path


def main():
    parser = argparse.ArgumentParser(description="Compute batched geodesics on a manifold.")
    parser.add_argument("--plot-knn", action="store_true", help="Plot the initial KNN paths")
    parser.add_argument("--num-targets", type=int, default=30, help="Number of target points")
    parser.add_argument("--sigma", type=float, default=0.15, help="Sigma parameter")
    parser.add_argument("--rho", type=float, default=1e-3, help="Rho parameter")
    parser.add_argument("--K_segments", type=int, default=10, help="Number of segments")
    args = parser.parse_args()

    # 1. Generate Synthetic Data
    n_samples = 300
    print("Generating synthetic data...")
    samples_torch, means_torch = sample_non_linear_data(n_samples=n_samples, std=0.1)

    # Convert to NumPy and JAX arrays
    X_np = samples_torch.numpy()
    X_tensor = jnp.array(X_np, dtype=jnp.float32)

    # 2. Define hyperparams and create manifold
    sigma, rho = args.sigma, args.rho
    K_segments = args.K_segments
    manifold = RiemannianManifold(X_tensor, sigma, rho, K_segments)

    # 3. Select 1 random base point and n target points
    num_targets = args.num_targets
    # np.random.seed(42)

    # Pick 1 random starting point
    base_idx = np.random.choice(n_samples)
    x_base = X_np[base_idx]

    # Pick n random target points
    target_indices = np.random.choice(n_samples, size=num_targets, replace=False)
    X_targets_np = X_np[target_indices]

    print(
        f"Computing geodesics for {num_targets} random target points via batch log map..."
    )

    # 4. Compute initial paths using KNN graph
    t0 = time.time()
    paths = []
    for i in range(num_targets):
        path = compute_knn_initial_path(
            x_base, X_targets_np[i], X_np, N_points=K_segments + 1
        )
        paths.append(path)

    paths_np = np.stack(paths)
    paths_jnp = jnp.array(paths_np, dtype=jnp.float32)

    x_base_jnp = jnp.array(x_base, dtype=jnp.float32)
    X_targets_jnp = jnp.array(X_targets_np, dtype=jnp.float32)
    t1 = time.time()
    print(f"Time taken for KNN paths: {t1 - t0} seconds")

    # 5. Compute initial velocities in batch using multiple shooting log map
    # log_map_batch signature: (mu, X_targets, initial_paths)
    t0 = time.time()
    v_opts = manifold.log_map_batch(x_base_jnp, X_targets_jnp, paths_jnp)
    t1 = time.time()
    print(f"Time taken for log maps: fucking {t1 - t0} seconds")

    # 6. Generate the continuous geodesics by integrating along the optimized velocities
    geodesics = []
    print("Integrating geodesics... (by calculating 30 exp maps)")
    t0 = time.time()
    for i in range(num_targets):
        v_opt = v_opts[i]
        path = []
        for t in jnp.linspace(0, 1, 30):
            point = manifold.exp_map(x_base_jnp, t * v_opt)
            path.append(np.array(point))
        geodesics.append(np.array(path))
    t1 = time.time()
    print(f"Time taken for exp maps: {t1 - t0} seconds")

    # 7. Plot results
    print("Plotting results...")
    plt.figure(figsize=(8, 6))

    # Plot entire dataset
    plt.scatter(X_np[:, 0], X_np[:, 1], s=10, alpha=0.3, c="blue", label="Data")

    # Plot base point
    plt.scatter(
        [x_base[0]],
        [x_base[1]],
        c="red",
        marker="*",
        s=200,
        edgecolors="k",
        zorder=5,
        label="Base Point",
    )

    # Plot target points
    plt.scatter(
        X_targets_np[:, 0],
        X_targets_np[:, 1],
        c="green",
        marker="X",
        s=80,
        edgecolors="k",
        zorder=4,
        label="Target Points",
    )

    # Plot geodesics
    for i, path in enumerate(geodesics):
        # Only add label for the first line to avoid legend clutter
        label = "Geodesic" if i == 0 else ""
        plt.plot(
            path[:, 0],
            path[:, 1],
            "-",
            linewidth=2,
            alpha=0.8,
            color="orange",
            label=label,
        )

    if args.plot_knn:
        for i, path in enumerate(paths_np):
            label = "KNN Path" if i == 0 else ""
            plt.plot(
                path[:, 0],
                path[:, 1],
                "--",
                linewidth=1.5,
                alpha=0.6,
                color="purple",
                label=label,
            )

    plt.title(f"Batched Geodesics from 1 Base Point to {num_targets} Target Points")
    plt.legend(loc="best")
    plt.axis("equal")

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    n_plots = os.listdir(plots_dir).__len__()
    out_file = f"{plots_dir}/geodesics_batch_result_{n_plots}.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
