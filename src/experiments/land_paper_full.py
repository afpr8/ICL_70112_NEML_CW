from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml, make_moons, make_s_curve, load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

from src.data.physionet_eeg import (
    apply_nmf,
    extract_subject_features,
    select_subjects,
)
from src.data.synthetic import sample_non_linear_data
from src.experiments.synthetic_land_paper import (
    ExperimentConfig,
    LeastSquaresGaussianModel,
    evaluate_land_density_grid,
    fit_land,
    predict_land_labels,
    run_clustering_experiment,
    run_contour_experiment,
    run_nll_experiment,
    true_component_means,
    true_logpdf,
)
from src.models.land import LANDMLE
from src.utils.land_utils import RiemannianManifold, compute_knn_initial_paths
from src.utils.plotting_utils import plot_mixture_contours


@dataclass
class FullPaperConfig:
    seed: int = 42
    output_dir: str = "plots/land_paper_full"
    synthetic_samples: int = 300
    synthetic_datasets: int = 1
    synthetic_eval_samples: int = 10_000
    synthetic_k_min: int = 1
    synthetic_k_max: int = 4
    synthetic_clustering_k: int = 20
    land_mc_samples: int = 3000
    run_eeg: bool = True
    run_mnist: bool = True
    run_scalability: bool = True
    run_normalization: bool = True
    run_model_selection: bool = True


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ls_logpdf(X: np.ndarray, ls_model: LeastSquaresGaussianModel) -> np.ndarray:
    D = X.shape[1]
    terms = []
    for k in range(len(ls_model.weights)):
        mu = ls_model.centroids[k]
        cov = ls_model.covariances[k]
        inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = np.log(np.abs(np.linalg.det(cov)) + 1e-12)
        diff = X - mu[None, :]
        mahal = np.einsum("ni,ij,nj->n", diff, inv, diff)
        log_k = np.log(ls_model.weights[k] + 1e-12) - 0.5 * (
            D * np.log(2.0 * np.pi) + logdet + mahal
        )
        terms.append(log_k)
    return logsumexp(np.stack(terms, axis=1), axis=1)


def _land_logpdf(
    X: np.ndarray,
    mu_list: list[jnp.ndarray],
    sigma_list: list[jnp.ndarray],
    C_list: list[jnp.ndarray],
    pi: jnp.ndarray,
    manifold: RiemannianManifold,
) -> np.ndarray:
    X_jnp = jnp.array(X, dtype=jnp.float32)
    terms = []
    for k in range(len(mu_list)):
        paths = compute_knn_initial_paths(
            np.array(mu_list[k]),
            X,
            manifold,
            N_points=manifold.K_segments + 1,
        )
        log_maps = manifold.log_map_batch(mu_list[k], X_jnp, jnp.array(paths))
        inv_sigma = jnp.linalg.inv(sigma_list[k])
        dist_sq = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)
        log_k = jnp.log(pi[k] + 1e-12) - jnp.log(C_list[k] + 1e-12) - 0.5 * dist_sq
        terms.append(np.array(log_k))
    return logsumexp(np.stack(terms, axis=1), axis=1)


def _mnist_digit_one(seed: int, n_samples: int = 200) -> np.ndarray:
    try:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y = y.astype(str)
        X_ones = X[y == "1"]
    except Exception:
        digits = load_digits()
        X_ones = digits.data[digits.target == 1]

    rng = np.random.default_rng(seed)
    idx = rng.choice(
        X_ones.shape[0], size=min(n_samples, X_ones.shape[0]), replace=False
    )
    return X_ones[idx]


def _plot_fig12_all_k(cfg: ExperimentConfig, output_dir: Path) -> None:
    X_t, _, _ = sample_non_linear_data(
        n_samples=cfg.n_samples_per_dataset,
        n_components=cfg.n_true_components,
        x_rad=cfg.x_rad,
        y_rad=cfg.y_rad,
        std=cfg.std,
        return_labels=True,
        seed=cfg.seed,
    )
    X = X_t.cpu().numpy()
    X_jnp = jnp.array(X, dtype=jnp.float32)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(4, 3, figsize=(14, 16))
    for row, K in enumerate(range(1, 5)):
        manifold = RiemannianManifold(
            X_jnp,
            sigma=cfg.sigma_metric,
            rho=cfg.rho_metric,
            K_segments=cfg.K_segments,
            n_neighbors=cfg.n_neighbors,
        )
        land_mu, land_sigma, land_C, land_pi = fit_land(
            X_jnp, K=K, cfg=cfg, seed=cfg.seed + K
        )
        z_land = evaluate_land_density_grid(
            xx, yy, land_mu, land_sigma, land_C, land_pi, manifold
        )

        ls_model = LeastSquaresGaussianModel.fit(X, K=K, seed=cfg.seed + K)
        z_ls = np.exp(_ls_logpdf(grid, ls_model)).reshape(xx.shape)

        gmm = GaussianMixture(
            n_components=K, covariance_type="full", random_state=cfg.seed + K
        )
        gmm.fit(X)
        z_gmm = np.exp(gmm.score_samples(grid)).reshape(xx.shape)

        plot_mixture_contours(
            axes[row, 0],
            X,
            np.array(jnp.stack(land_mu)),
            xx,
            yy,
            z_land,
            title=f"LAND mixture model (K={K})",
            mean_label="LAND mean",
        )
        plot_mixture_contours(
            axes[row, 1],
            X,
            ls_model.centroids,
            xx,
            yy,
            z_ls,
            title=f"Least Squares mixture model (K={K})",
            mean_label="LS mean",
        )
        plot_mixture_contours(
            axes[row, 2],
            X,
            gmm.means_,
            xx,
            yy,
            z_gmm,
            title=f"Gaussian mixture model (K={K})",
            mean_label="GMM mean",
        )

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig12_appendix_all_k_contours.png"), dpi=180)
    plt.close(fig)


def _build_two_clustering_datasets(seed: int) -> list[np.ndarray]:
    X1, _ = make_moons(n_samples=300, noise=0.08, random_state=seed)
    X2_3d, _ = make_s_curve(n_samples=300, noise=0.06, random_state=seed + 1)
    X2 = X2_3d[:, [0, 2]]
    return [X1.astype(np.float32), X2.astype(np.float32)]


def _land_contour_and_geodesic_figure(
    X: np.ndarray, cfg: ExperimentConfig
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[jnp.ndarray],
    list[jnp.ndarray],
    list[jnp.ndarray],
    jnp.ndarray,
    RiemannianManifold,
]:
    X_jnp = jnp.array(X, dtype=jnp.float32)
    manifold = RiemannianManifold(
        X_jnp,
        sigma=cfg.sigma_metric,
        rho=cfg.rho_metric,
        K_segments=cfg.K_segments,
        n_neighbors=cfg.n_neighbors,
    )
    land_mu, land_sigma, land_C, land_pi = fit_land(X_jnp, K=2, cfg=cfg, seed=cfg.seed)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
    z_land = evaluate_land_density_grid(
        xx, yy, land_mu, land_sigma, land_C, land_pi, manifold
    )

    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=cfg.seed)
    gmm.fit(X)
    z_gmm = np.exp(gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

    return xx, yy, z_land, z_gmm, land_mu, land_sigma, land_C, land_pi, manifold


def _plot_fig6_fig11(cfg: ExperimentConfig, output_dir: Path) -> None:
    datasets = _build_two_clustering_datasets(cfg.seed)

    fig6, axes6 = plt.subplots(2, 3, figsize=(14, 8))
    fig11, axes11 = plt.subplots(2, 3, figsize=(14, 8))

    for row, X in enumerate(datasets):
        xx, yy, z_land, z_gmm, land_mu, land_sigma, land_C, land_pi, manifold = (
            _land_contour_and_geodesic_figure(X, cfg)
        )

        labels_land = predict_land_labels(
            X, land_mu, land_sigma, land_C, land_pi, manifold
        )
        means_land = np.array(jnp.stack(land_mu))

        axes6[row, 0].scatter(X[:, 0], X[:, 1], c="#5DADE2", s=16)
        axes6[row, 0].scatter(
            means_land[:, 0],
            means_land[:, 1],
            marker="D",
            s=90,
            c="orange",
            edgecolors="black",
        )

        subset = np.random.default_rng(cfg.seed + row).choice(
            X.shape[0], size=min(80, X.shape[0]), replace=False
        )
        X_sub = X[subset]
        X_sub_jnp = jnp.array(X_sub, dtype=jnp.float32)
        for i, x in enumerate(X_sub):
            c = int(labels_land[subset[i]])
            path_init = compute_knn_initial_paths(
                np.array(land_mu[c]),
                x[None, :],
                manifold,
                N_points=manifold.K_segments + 1,
            )[0]
            v = manifold.log_map_shooting(
                land_mu[c], X_sub_jnp[i], jnp.array(path_init), scaled=False
            )
            path = np.array(
                [
                    np.array(manifold.exp_map(land_mu[c], t * v))
                    for t in jnp.linspace(0, 1, 20)
                ]
            )
            axes6[row, 0].plot(
                path[:, 0],
                path[:, 1],
                linewidth=0.6,
                alpha=0.5,
                color=("#7CB342" if c == 0 else "#C62828"),
            )

        axes6[row, 0].set_title("Geodesics")
        plot_mixture_contours(
            axes6[row, 1],
            X,
            means_land,
            xx,
            yy,
            z_land,
            title="LAND mixture model",
            mean_label="LAND mean",
        )

        gmm2 = GaussianMixture(
            n_components=2, covariance_type="full", random_state=cfg.seed + row
        )
        gmm2.fit(X)
        plot_mixture_contours(
            axes6[row, 2],
            X,
            gmm2.means_,
            xx,
            yy,
            z_gmm,
            title="Gaussian mixture model",
            mean_label="GMM mean",
        )

        ls = LeastSquaresGaussianModel.fit(X, K=2, seed=cfg.seed + row)
        z_ls = np.exp(_ls_logpdf(np.c_[xx.ravel(), yy.ravel()], ls)).reshape(xx.shape)
        plot_mixture_contours(
            axes11[row, 0],
            X,
            means_land,
            xx,
            yy,
            z_land,
            title="LAND mixture model",
            mean_label="LAND mean",
        )
        plot_mixture_contours(
            axes11[row, 1],
            X,
            ls.centroids,
            xx,
            yy,
            z_ls,
            title="Least Squares mixture model",
            mean_label="LS mean",
        )
        plot_mixture_contours(
            axes11[row, 2],
            X,
            gmm2.means_,
            xx,
            yy,
            z_gmm,
            title="Gaussian mixture model",
            mean_label="GMM mean",
        )

    fig6.tight_layout()
    fig6.savefig(str(output_dir / "fig6_synthetic_clustering_land_vs_gmm.png"), dpi=180)
    plt.close(fig6)

    fig11.tight_layout()
    fig11.savefig(str(output_dir / "fig11_appendix_clustering_all_models.png"), dpi=180)
    plt.close(fig11)


def _run_eeg_experiment(cfg: FullPaperConfig, output_dir: Path) -> dict[str, object]:
    label_map = {"non-REM": 0, "REM": 1, "awake": 2}
    subjects = select_subjects(n_subjects=10, random_state=cfg.seed)

    rows: list[dict[str, float | int]] = []
    sigma_grid = np.round(np.arange(0.5, 1.51, 0.1), 2)
    sigma_curve_ref: dict[str, float] = {}

    for subj in subjects:
        feats, labels, _ = extract_subject_features(subj)
        X = np.array(feats)
        y = np.array([label_map[l] for l in labels])
        X5 = apply_nmf(X, n_components=5, n_starts=10, random_state=cfg.seed)
        X_jnp = jnp.array(X5, dtype=jnp.float32)

        K = 3
        gmm = GaussianMixture(
            n_components=K, covariance_type="full", random_state=cfg.seed
        )
        y_gmm = gmm.fit_predict(X5)
        f_gmm = f1_score(y, y_gmm, average="macro")

        e_cfg = ExperimentConfig(
            seed=cfg.seed,
            sigma_metric=1.0,
            rho_metric=1e-3,
            K_segments=5,
            n_neighbors=7,
            land_S=600,
            land_eps=1e-3,
        )
        land_mu, land_sigma, land_C, land_pi = fit_land(
            X_jnp, K=K, cfg=e_cfg, seed=cfg.seed
        )
        manifold = RiemannianManifold(
            X_jnp, sigma=1.0, rho=1e-3, K_segments=5, n_neighbors=7
        )
        y_land = predict_land_labels(X5, land_mu, land_sigma, land_C, land_pi, manifold)
        f_land = f1_score(y, y_land, average="macro")

        best_sigma, best_f = 1.0, f_land
        sigma_curve = {}
        for sig in sigma_grid:
            t_cfg = ExperimentConfig(
                seed=cfg.seed,
                sigma_metric=float(sig),
                rho_metric=1e-3,
                K_segments=5,
                n_neighbors=7,
                land_S=600,
                land_eps=1e-3,
            )
            t_mu, t_sigma, t_C, t_pi = fit_land(X_jnp, K=K, cfg=t_cfg, seed=cfg.seed)
            t_man = RiemannianManifold(
                X_jnp, sigma=float(sig), rho=1e-3, K_segments=5, n_neighbors=7
            )
            y_t = predict_land_labels(X5, t_mu, t_sigma, t_C, t_pi, t_man)
            f_t = f1_score(y, y_t, average="macro")
            sigma_curve[str(sig)] = float(f_t)
            if f_t > best_f:
                best_f = f_t
                best_sigma = float(sig)

        if int(subj) == 15:
            sigma_curve_ref = sigma_curve

        rows.append(
            {
                "subject": int(subj),
                "fmeasure_land_sigma1": float(f_land),
                "fmeasure_gmm": float(f_gmm),
                "fmeasure_land_tuned": float(best_f),
                "best_sigma": float(best_sigma),
            }
        )

    table_path = output_dir / "table1_eeg_fmeasure.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    focus_subject = 15 if 15 in subjects else int(subjects[0])
    f_feats, f_labels, _ = extract_subject_features(focus_subject)
    Xf = apply_nmf(
        np.array(f_feats), n_components=5, n_starts=10, random_state=cfg.seed
    )
    yf = np.array([label_map[l] for l in f_labels])

    fig7 = plt.figure(figsize=(7, 5))
    ax = fig7.add_subplot(111, projection="3d")
    colors = np.array(["#8BC34A", "#2196F3", "#F57C00"])
    ax.scatter(Xf[:, 0], Xf[:, 1], Xf[:, 2], s=8, c=colors[yf])
    ax.set_title('Figure 7: Leading factors for subject "s151"')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    fig7.tight_layout()
    fig7.savefig(str(output_dir / "fig7_subject_s151_3d.png"), dpi=180)
    plt.close(fig7)

    fig10 = plt.figure(figsize=(12, 7))
    gs = fig10.add_gridspec(2, 3)
    top_subjects = [
        subjects[0],
        subjects[min(1, len(subjects) - 1)],
        subjects[min(2, len(subjects) - 1)],
    ]
    for i, subj in enumerate(top_subjects):
        feats, labels, _ = extract_subject_features(int(subj))
        Xs = apply_nmf(
            np.array(feats), n_components=5, n_starts=10, random_state=cfg.seed
        )
        ys = np.array([label_map[l] for l in labels])
        ax = fig10.add_subplot(gs[0, i], projection="3d")
        ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], s=5, c=colors[ys])
        ax.set_title(f"s{int(subj):03d}")

    ax_b = fig10.add_subplot(gs[1, 1])
    if not sigma_curve_ref:
        sigma_curve_ref = {str(s): np.nan for s in sigma_grid}
    x = np.array([float(s) for s in sigma_curve_ref.keys()])
    y = np.array([float(v) for v in sigma_curve_ref.values()])
    ax_b.plot(x, y, marker="o", label="LAND")
    ax_b.axhline(np.nanmean(y), color="C1", label="GMM")
    ax_b.set_xlabel(r"$\sigma$")
    ax_b.set_ylabel("F-measure")
    ax_b.set_title("F-measure vs sigma")
    ax_b.legend()
    fig10.tight_layout()
    fig10.savefig(
        str(output_dir / "fig10_appendix_eeg_subjects_sigma_curve.png"), dpi=180
    )
    plt.close(fig10)

    return {
        "table_path": str(table_path),
        "subjects": [int(s) for s in subjects],
        "mean_land": float(np.mean([r["fmeasure_land_sigma1"] for r in rows])),
        "mean_gmm": float(np.mean([r["fmeasure_gmm"] for r in rows])),
        "mean_land_tuned": float(np.mean([r["fmeasure_land_tuned"] for r in rows])),
    }


def _run_normalization_constant_experiment(
    cfg: FullPaperConfig, output_dir: Path
) -> dict[str, float]:
    X_t, _ = sample_non_linear_data(n_samples=300, seed=cfg.seed)
    X = X_t.numpy()
    X_jnp = jnp.array(X, dtype=jnp.float32)

    model = LANDMLE(
        initial_lr_mu=1e-2,
        initial_lr_A=1e-2,
        S=cfg.land_mc_samples,
        epsilon=1e-3,
        sigma=0.15,
        rho=1e-3,
        K_segments=6,
    )
    mu, sigma, _ = model.fit(X_jnp)
    manifold = RiemannianManifold(
        X_jnp, sigma=0.15, rho=1e-3, K_segments=6, n_neighbors=7
    )

    inv_sigma = jnp.linalg.inv(sigma)
    bound = 3.5 * float(np.sqrt(np.max(np.diag(np.array(sigma)))))
    vx = np.linspace(-bound, bound, 100)
    vy = np.linspace(-bound, bound, 100)
    Vx, Vy = np.meshgrid(vx, vy)
    points = np.c_[Vx.ravel(), Vy.ravel()]

    values = []
    for v in points:
        vj = jnp.array(v, dtype=jnp.float32)
        x_map = manifold.exp_map(mu, vj)
        vol = jnp.sqrt(jnp.linalg.det(manifold.metric(x_map)))
        gauss = jnp.exp(-0.5 * (vj @ inv_sigma @ vj))
        values.append(float(gauss * vol))
    Z = np.array(values).reshape(Vx.shape)
    numeric = np.trapz(np.trapz(Z, vx, axis=1), vy)

    sample_sizes = list(range(100, 3001, 100))
    curves = []
    for run in range(10):
        run_vals = []
        key = jax.random.key(cfg.seed + run)
        for s in sample_sizes:
            key, sub = jax.random.split(key)
            c_hat, _ = manifold.compute_normalization_constant(
                mu, sigma, sub, n_samples=s
            )
            run_vals.append(float(c_hat))
        curves.append(run_vals)

    curves_arr = np.array(curves)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in curves_arr:
        ax.plot(sample_sizes, row, alpha=0.3, linewidth=1.0)
    ax.plot(sample_sizes, curves_arr.mean(axis=0), "r--", linewidth=2, label="Mean")
    ax.axhline(
        float(numeric), color="black", linewidth=2, label="Numerical integration"
    )
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Normalization constant")
    ax.set_title("Figure 8: Normalization constant estimation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig8_normalization_constant.png"), dpi=180)
    plt.close(fig)

    return {
        "numeric_integral": float(numeric),
        "mc_mean_last": float(curves_arr[:, -1].mean()),
    }


def _run_mnist_experiment(cfg: FullPaperConfig, output_dir: Path) -> dict[str, float]:
    X_mnist = _mnist_digit_one(cfg.seed, n_samples=200)
    X_2d = PCA(n_components=2, random_state=cfg.seed).fit_transform(X_mnist)
    X_jnp = jnp.array(X_2d, dtype=jnp.float32)

    land = LANDMLE(
        initial_lr_mu=1e-2,
        initial_lr_A=1e-2,
        S=cfg.land_mc_samples,
        epsilon=1e-3,
        sigma=1.0,
        rho=1e-3,
        K_segments=6,
        n_neighbors=7,
    )
    mu, sigma, C = land.fit(X_jnp)
    manifold = RiemannianManifold(
        X_jnp, sigma=1.0, rho=1e-3, K_segments=6, n_neighbors=7
    )

    ls = LeastSquaresGaussianModel.fit(X_2d, K=6, seed=cfg.seed)
    gmm_linear = GaussianMixture(
        n_components=1, covariance_type="full", random_state=cfg.seed
    )
    gmm_linear.fit(X_2d)

    x_min, x_max = X_2d[:, 0].min() - 1.0, X_2d[:, 0].max() + 1.0
    y_min, y_max = X_2d[:, 1].min() - 1.0, X_2d[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
    grid = np.c_[xx.ravel(), yy.ravel()]

    distances = manifold.nn_tree.kneighbors(grid, 1, return_distance=True)[0]
    valid_mask = distances.flatten() < (2.0 * manifold.sigma)
    full_densities = np.zeros(grid.shape[0], dtype=np.float64)

    if np.any(valid_mask):
        valid_grid = grid[valid_mask]
        valid_grid_jnp = jnp.array(valid_grid, dtype=jnp.float32)
        paths = compute_knn_initial_paths(
            np.array(mu), valid_grid, manifold, N_points=manifold.K_segments + 1
        )
        log_maps = manifold.log_map_batch(mu, valid_grid_jnp, jnp.array(paths))
        inv_sigma = jnp.linalg.inv(sigma)
        dist = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)
        full_densities[valid_mask] = np.array((1.0 / C) * jnp.exp(-0.5 * dist))

    z_land = full_densities.reshape(xx.shape)

    z_ls = np.exp(_ls_logpdf(grid, ls)).reshape(xx.shape)
    z_linear = np.exp(gmm_linear.score_samples(grid)).reshape(xx.shape)

    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.6))
    axes1[0].scatter(X_2d[:, 0], X_2d[:, 1], s=14, c="#7FB3D5")
    axes1[0].scatter(
        [float(mu[0])], [float(mu[1])], marker="D", c="orange", s=90, edgecolors="black"
    )
    subset = np.random.default_rng(cfg.seed).choice(
        X_2d.shape[0], size=min(60, X_2d.shape[0]), replace=False
    )
    for i in subset:
        x = X_2d[i]
        p = compute_knn_initial_paths(
            np.array(mu), x[None, :], manifold, N_points=manifold.K_segments + 1
        )[0]
        v = manifold.log_map_shooting(
            mu, jnp.array(x, dtype=jnp.float32), jnp.array(p), scaled=False
        )
        geod = np.array(
            [np.array(manifold.exp_map(mu, t * v)) for t in jnp.linspace(0, 1, 15)]
        )
        axes1[0].plot(geod[:, 0], geod[:, 1], color="#7CB342", alpha=0.5, linewidth=0.8)
        axes1[0].plot(
            [float(mu[0]), x[0]],
            [float(mu[1]), x[1]],
            color="#D81B60",
            alpha=0.25,
            linewidth=0.8,
        )
    axes1[0].set_title("Geodesics")

    plot_mixture_contours(
        axes1[1],
        X_2d,
        np.array(mu).reshape(1, -1),
        xx,
        yy,
        z_land,
        "LAND model",
        "LAND mean",
    )
    plot_mixture_contours(
        axes1[2],
        X_2d,
        gmm_linear.means_,
        xx,
        yy,
        z_linear,
        "Linear model",
        "Linear mean",
    )
    fig1.tight_layout()
    fig1.savefig(str(output_dir / "fig1_mnist_geodesics_land_linear.png"), dpi=180)
    plt.close(fig1)

    fig9, axes9 = plt.subplots(1, 3, figsize=(14, 4.6))
    plot_mixture_contours(
        axes9[0],
        X_2d,
        np.array(mu).reshape(1, -1),
        xx,
        yy,
        z_land,
        "LAND model",
        "LAND mean",
    )
    plot_mixture_contours(
        axes9[1], X_2d, ls.centroids, xx, yy, z_ls, "Least Squares model", "LS mean"
    )
    plot_mixture_contours(
        axes9[2],
        X_2d,
        gmm_linear.means_,
        xx,
        yy,
        z_linear,
        "Linear model",
        "Linear mean",
    )
    fig9.tight_layout()
    fig9.savefig(str(output_dir / "fig9_mnist_contours.png"), dpi=180)
    plt.close(fig9)

    return {
        "n_samples": int(X_2d.shape[0]),
        "land_C": float(C),
    }


def _run_scalability_experiment(
    cfg: FullPaperConfig, output_dir: Path
) -> dict[str, list[float]]:
    X_mnist = _mnist_digit_one(cfg.seed, n_samples=1200)
    dims = [2, 5, 10, 20, 30, 40, 50, 60]
    runtimes = []
    usable_dims = []

    rng = np.random.default_rng(cfg.seed)
    for d in dims:
        if d > X_mnist.shape[1]:
            continue
        Xd = PCA(n_components=d, random_state=cfg.seed).fit_transform(X_mnist)
        Xd = Xd.astype(np.float32)
        base = Xd[0]
        tgt_idx = rng.choice(np.arange(1, Xd.shape[0]), size=20, replace=False)
        targets = Xd[tgt_idx]

        X_jnp = jnp.array(Xd, dtype=jnp.float32)
        manifold = RiemannianManifold(
            X_jnp, sigma=1.0, rho=1e-3, K_segments=5, n_neighbors=5
        )

        start = time.perf_counter()
        paths = compute_knn_initial_paths(
            base, targets, manifold, N_points=manifold.K_segments + 1
        )
        _ = manifold.log_map_batch(
            jnp.array(base), jnp.array(targets), jnp.array(paths)
        )
        elapsed = time.perf_counter() - start

        usable_dims.append(d)
        runtimes.append(elapsed)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(usable_dims, runtimes, marker="o")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Seconds (20 log-maps)")
    ax.set_title("Figure 14: Geodesic scalability")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig14_geodesic_scalability.png"), dpi=180)
    plt.close(fig)

    return {"dims": usable_dims, "runtimes": runtimes}


def _run_model_selection_experiment(
    cfg: FullPaperConfig, output_dir: Path
) -> dict[str, dict[str, list[float]]]:
    X_t, _, _ = sample_non_linear_data(n_samples=300, return_labels=True, seed=cfg.seed)
    X = X_t.numpy()
    X_jnp = jnp.array(X, dtype=jnp.float32)

    results = {
        "AIC": {"LAND": [], "LS": [], "GMM": []},
        "BIC": {"LAND": [], "LS": [], "GMM": []},
    }

    N, D = X.shape
    for K in range(1, 5):
        gmm = GaussianMixture(
            n_components=K, covariance_type="full", random_state=cfg.seed
        )
        gmm.fit(X)
        results["AIC"]["GMM"].append(float(gmm.aic(X)))
        results["BIC"]["GMM"].append(float(gmm.bic(X)))

        ls = LeastSquaresGaussianModel.fit(X, K=K, seed=cfg.seed)
        ll_ls = _ls_logpdf(X, ls).sum()
        p = K * D + K * D * (D + 1) / 2 + (K - 1)
        results["AIC"]["LS"].append(float(2 * p - 2 * ll_ls))
        results["BIC"]["LS"].append(float(p * np.log(N) - 2 * ll_ls))

        e_cfg = ExperimentConfig(
            seed=cfg.seed,
            sigma_metric=0.15,
            rho_metric=1e-3,
            K_segments=6,
            n_neighbors=7,
            land_S=600,
        )
        mu, sigma, C, pi = fit_land(X_jnp, K=K, cfg=e_cfg, seed=cfg.seed + K)
        manifold = RiemannianManifold(
            X_jnp, sigma=0.15, rho=1e-3, K_segments=6, n_neighbors=7
        )
        ll_land = _land_logpdf(X, mu, sigma, C, pi, manifold).sum()
        results["AIC"]["LAND"].append(float(2 * p - 2 * ll_land))
        results["BIC"]["LAND"].append(float(p * np.log(N) - 2 * ll_land))

    K_vals = [1, 2, 3, 4]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for model in ["LAND", "LS", "GMM"]:
        axes[0].plot(K_vals, results["AIC"][model], marker="o", label=model)
        axes[1].plot(K_vals, results["BIC"][model], marker="o", label=model)
    axes[0].set_title("AIC")
    axes[1].set_title("BIC")
    for ax in axes:
        ax.set_xlabel("K")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figure 15: Model selection")
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig15_model_selection_aic_bic.png"), dpi=180)
    plt.close(fig)

    return results


def run_all_paper_experiments(cfg: FullPaperConfig) -> dict[str, object]:
    out = Path(cfg.output_dir)
    _ensure_dir(out)

    optional_steps = sum(
        [
            int(cfg.run_eeg),
            int(cfg.run_normalization),
            int(cfg.run_mnist),
            int(cfg.run_scalability),
            int(cfg.run_model_selection),
        ]
    )
    total_steps = 7 + optional_steps
    current_step = 0

    def run_step(name: str, fn: Callable[[], Any]) -> Any:
        nonlocal current_step
        current_step += 1
        print(f"[LAND-PAPER][{current_step}/{total_steps}] START {name}", flush=True)
        t0 = time.perf_counter()
        out_val = fn()
        elapsed = time.perf_counter() - t0
        print(
            f"[LAND-PAPER][{current_step}/{total_steps}] DONE  {name} ({elapsed:.2f}s)",
            flush=True,
        )
        return out_val

    synthetic_cfg = ExperimentConfig(
        n_datasets=cfg.synthetic_datasets,
        n_samples_per_dataset=cfg.synthetic_samples,
        n_eval_samples=cfg.synthetic_eval_samples,
        k_min=cfg.synthetic_k_min,
        k_max=cfg.synthetic_k_max,
        land_S=cfg.land_mc_samples,
        seed=cfg.seed,
        clustering_K=cfg.synthetic_clustering_k,
    )

    result: dict[str, object] = {}

    # Fast/non-mixture stages first
    if cfg.run_normalization:
        result["normalization"] = run_step(
            "Normalization constant estimation (Fig. 8)",
            lambda: _run_normalization_constant_experiment(cfg, out),
        )
    if cfg.run_mnist:
        result["mnist"] = run_step(
            "MNIST digit 1 representation (Fig. 1 & Fig. 9)",
            lambda: _run_mnist_experiment(cfg, out),
        )
    if cfg.run_scalability:
        result["scalability"] = run_step(
            "Geodesic scalability benchmark (Fig. 14)",
            lambda: _run_scalability_experiment(cfg, out),
        )

    # Mixture-heavy stages afterwards
    if cfg.run_model_selection:
        result["model_selection"] = run_step(
            "Model selection AIC/BIC (Fig. 15)",
            lambda: _run_model_selection_experiment(cfg, out),
        )

    nll = run_step(
        "Synthetic NLL benchmark (Fig. 4)",
        lambda: run_nll_experiment(synthetic_cfg, out),
    )
    contour = run_step(
        "Synthetic contours K=2 (Fig. 5)",
        lambda: run_contour_experiment(synthetic_cfg, out),
    )
    clustering_scores = run_step(
        "Synthetic clustering metrics",
        lambda: run_clustering_experiment(synthetic_cfg, out),
    )

    def _rename_synthetic_files() -> None:
        fig4_src = out / "synthetic_nll_vs_k.png"
        if fig4_src.exists():
            fig4_src.replace(out / "fig4_mean_negative_loglikelihood.png")

        fig5_src = out / "synthetic_contours_k2.png"
        if fig5_src.exists():
            fig5_src.replace(out / "fig5_land_gmm_k2_contours.png")

    run_step("Rename synthetic figure artifacts", _rename_synthetic_files)
    run_step(
        "Appendix contours all K (Fig. 12)",
        lambda: _plot_fig12_all_k(synthetic_cfg, out),
    )
    run_step(
        "Synthetic clustering visuals (Fig. 6 & Fig. 11)",
        lambda: _plot_fig6_fig11(synthetic_cfg, out),
    )

    result.update({
        "synthetic_nll": nll,
        "synthetic_contours": contour,
        "synthetic_clustering": clustering_scores,
    })

    if cfg.run_eeg:
        result["eeg"] = run_step(
            "Sleep stages EEG experiment (Table 1, Fig. 7, Fig. 10)",
            lambda: _run_eeg_experiment(cfg, out),
        )

    def _save_summary() -> None:
        with (out / "paper_experiments_summary.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    run_step("Write summary JSON", _save_summary)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all LAND paper experiments")
    parser.add_argument("--output-dir", type=str, default="plots/land_paper_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-datasets", type=int, default=1)
    parser.add_argument("--synthetic-samples", type=int, default=300)
    parser.add_argument("--synthetic-eval-samples", type=int, default=10_000)
    parser.add_argument("--synthetic-k-min", type=int, default=1)
    parser.add_argument("--synthetic-k-max", type=int, default=4)
    parser.add_argument("--synthetic-clustering-k", type=int, default=20)
    parser.add_argument("--land-mc-samples", type=int, default=3000)
    parser.add_argument("--skip-eeg", action="store_true")
    parser.add_argument("--skip-mnist", action="store_true")
    parser.add_argument("--skip-scalability", action="store_true")
    parser.add_argument("--skip-normalization", action="store_true")
    parser.add_argument("--skip-model-selection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FullPaperConfig(
        seed=args.seed,
        output_dir=args.output_dir,
        synthetic_samples=args.synthetic_samples,
        synthetic_datasets=args.synthetic_datasets,
        synthetic_eval_samples=args.synthetic_eval_samples,
        synthetic_k_min=args.synthetic_k_min,
        synthetic_k_max=args.synthetic_k_max,
        synthetic_clustering_k=args.synthetic_clustering_k,
        land_mc_samples=args.land_mc_samples,
        run_eeg=not args.skip_eeg,
        run_mnist=not args.skip_mnist,
        run_scalability=not args.skip_scalability,
        run_normalization=not args.skip_normalization,
        run_model_selection=not args.skip_model_selection,
    )
    print("[LAND-PAPER] Starting full paper experiment suite", flush=True)
    print(
        "[LAND-PAPER] Config: "
        f"seed={cfg.seed}, output_dir={cfg.output_dir}, "
        f"run_eeg={cfg.run_eeg}, run_mnist={cfg.run_mnist}, "
        f"run_scalability={cfg.run_scalability}, run_normalization={cfg.run_normalization}, "
        f"run_model_selection={cfg.run_model_selection}",
        flush=True,
    )
    run_all_paper_experiments(cfg)
    print("[LAND-PAPER] Completed all requested steps", flush=True)


if __name__ == "__main__":
    main()
