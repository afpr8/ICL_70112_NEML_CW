import matplotlib.pyplot as plt


def plot_geodesics(ax, X, means, labels, geodesics):
    """
    Visualise the data, cluster means, and the geodesics connecting
    each data point to its assigned cluster mean.
    """

    cluster_colours = ["#8FBC8F", "#CD5C5C"]  # Greenish and Reddish
    ax.scatter(X[:, 0], X[:, 1], c="#5DADE2", s=20, label="Data", zorder=2)

    # Plot the geodesics
    # 'geodesics' is expected to be a list of (N_steps, 2) arrays
    plotted_labels = set()
    for i, path in enumerate(geodesics):
        cluster_idx = labels[i]
        col = cluster_colours[cluster_idx % len(cluster_colours)]

        lbl = f"Geodesics, cluster {cluster_idx + 1}"
        if lbl in plotted_labels:
            lbl = None  # Avoid duplicate legend entries
        else:
            plotted_labels.add(lbl)

        ax.plot(
            path[:, 0], path[:, 1], c=col, alpha=0.4, linewidth=1, label=lbl, zorder=1
        )

    # Plot the cluster means
    ax.scatter(
        means[:, 0],
        means[:, 1],
        c="orange",
        marker="D",
        s=100,
        edgecolors="black",
        label="LAND means",
        zorder=3,
    )

    ax.set_title("Geodesics")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize="small")


def plot_mixture_contours(ax, X, means, X_grid, Y_grid, Z, title, mean_label):
    """
    Visualise the data, cluster means, and the density contours of a mixture model.
    """
    # Plot contours (using 'jet' to match the rainbow gradient style)
    ax.contour(X_grid, Y_grid, Z, levels=10, cmap="jet", linewidths=1.5, zorder=1)

    # Plot the data points
    ax.scatter(X[:, 0], X[:, 1], c="#5DADE2", s=20, zorder=2)

    # Plot the cluster means
    ax.scatter(
        means[:, 0],
        means[:, 1],
        c="orange",
        marker="D",
        s=100,
        edgecolors="black",
        label=mean_label,
        zorder=3,
    )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize="small")


def plot_full_comparison(
    X, land_means, gmm_means, labels, geodesics, X_grid, Y_grid, Z_land, Z_gmm
):
    """
    Creates a full 1x3 figure comparing geodesics, LAND contours, and GMM contours.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Geodesics Plot
    plot_geodesics(axes[0], X, land_means, labels, geodesics)

    # LAND Mixture Model Contours
    plot_mixture_contours(
        axes[1],
        X,
        land_means,
        X_grid,
        Y_grid,
        Z_land,
        title="LAND mixture model",
        mean_label="LAND mean",
    )

    # Gaussian Mixture Model Contours
    plot_mixture_contours(
        axes[2],
        X,
        gmm_means,
        X_grid,
        Y_grid,
        Z_gmm,
        title="Gaussian mixture model",
        mean_label="GMM mean",
    )

    plt.tight_layout()
    return fig
