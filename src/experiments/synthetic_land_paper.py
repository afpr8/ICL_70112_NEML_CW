from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from src.data.synthetic import sample_non_linear_data
from src.models.mixture_model import LANDMixtureModel
from src.utils.land_utils import RiemannianManifold, compute_knn_initial_paths
from src.utils.plotting_utils import plot_mixture_contours


@dataclass
class ExperimentConfig:
    n_datasets: int = 1
    n_samples_per_dataset: int = 220
    n_true_components: int = 20
    k_min: int = 1
    k_max: int = 4
    n_eval_samples: int = 2000
    x_rad: float = 0.75
    y_rad: float = 1.5
    std: float = 0.15
    sigma_metric: float = 0.15
    rho_metric: float = 1e-3
    K_segments: int = 10
    n_neighbors: int = 5
    land_lr_mu: float = 1e-2
    land_lr_A: float = 1e-2
    land_S: int = 400
    land_eps: float = 1e-3
    contour_cutoff_std: float = 2.0
    contour_grid_size: int = 28
    seed: int = 42
    clustering_K: int = 10
    kmeans_n_init: int = 8


@dataclass
class LeastSquaresGaussianModel:
    centroids: np.ndarray
    covariances: list[np.ndarray]
    weights: np.ndarray
    kmeans: KMeans

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        K: int,
        seed: int,
        reg: float = 1e-6,
        n_init: int = 8,
    ) -> "LeastSquaresGaussianModel":
        kmeans = KMeans(n_clusters=K, random_state=seed, n_init=n_init)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        N, D = X.shape

        covariances: list[np.ndarray] = []
        weights = np.zeros(K, dtype=np.float64)
        global_cov = np.cov(X.T) + reg * np.eye(D)

        for k in range(K):
            X_k = X[labels == k]
            n_k = X_k.shape[0]
            weights[k] = n_k / N
            if n_k <= 1:
                covariances.append(global_cov.copy())
                continue
            centered = X_k - centroids[k]
            cov = (centered.T @ centered) / n_k
            cov = cov + reg * np.eye(D)
            covariances.append(cov)

        if np.isclose(weights.sum(), 0.0):
            weights[:] = 1.0 / K
        else:
            weights /= weights.sum()

        return cls(
            centroids=centroids,
            covariances=covariances,
            weights=weights,
            kmeans=kmeans,
        )

    def sample(self, n_samples: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        comp_ids = rng.choice(len(self.weights), size=n_samples, p=self.weights)
        samples = np.empty((n_samples, self.centroids.shape[1]), dtype=np.float64)

        for k in range(len(self.weights)):
            idx = np.where(comp_ids == k)[0]
            if idx.size == 0:
                continue
            samples[idx] = rng.multivariate_normal(
                mean=self.centroids[k],
                cov=self.covariances[k],
                size=idx.size,
            )

        return samples

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(X)


def true_component_means(cfg: ExperimentConfig) -> np.ndarray:
    t = np.linspace(0.0, np.pi, cfg.n_true_components)
    return np.stack([cfg.x_rad * np.cos(t), cfg.y_rad * np.sin(t)], axis=1)


def true_logpdf(X: np.ndarray, cfg: ExperimentConfig, means: np.ndarray) -> np.ndarray:
    D = X.shape[1]
    var = cfg.std**2
    sq = ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    component_log = -0.5 * sq / var - 0.5 * D * np.log(2.0 * np.pi * var)
    return logsumexp(component_log, axis=1) - np.log(cfg.n_true_components)


def evaluate_land_density_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    mu_list: list[jnp.ndarray],
    sigma_list: list[jnp.ndarray],
    C_list: list[jnp.ndarray],
    pi: jnp.ndarray,
    manifold: RiemannianManifold,
    cutoff_std: float = 2.0,
) -> np.ndarray:
    grid = np.c_[xx.ravel(), yy.ravel()]

    distances = manifold.nn_tree.kneighbors(grid, 1, return_distance=True)[0]
    valid_mask = distances.flatten() < (cutoff_std * manifold.sigma)

    full_densities = np.zeros(grid.shape[0], dtype=np.float64)
    if not np.any(valid_mask):
        return full_densities.reshape(xx.shape)

    valid_grid = grid[valid_mask]
    valid_grid_jnp = jnp.array(valid_grid, dtype=jnp.float32)
    valid_densities = jnp.zeros(valid_grid_jnp.shape[0])

    for k in range(len(mu_list)):
        paths = compute_knn_initial_paths(
            np.array(mu_list[k]),
            valid_grid,
            manifold,
            N_points=manifold.K_segments + 1,
        )
        log_maps = manifold.log_map_batch(mu_list[k], valid_grid_jnp, jnp.array(paths))
        inv_sigma = jnp.linalg.inv(sigma_list[k])
        dist_sq = jnp.sum((log_maps @ inv_sigma) * log_maps, axis=-1)
        valid_densities = valid_densities + pi[k] * (1.0 / C_list[k]) * jnp.exp(
            -0.5 * dist_sq
        )

    full_densities[valid_mask] = np.array(valid_densities)
    return full_densities.reshape(xx.shape)


def sample_land_mixture(
    mu_list: list[jnp.ndarray],
    sigma_list: list[jnp.ndarray],
    pi: jnp.ndarray,
    manifold: RiemannianManifold,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = np.array(pi, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum()

    component_ids = rng.choice(len(mu_list), size=n_samples, p=probs)
    out = np.zeros((n_samples, 2), dtype=np.float64)

    key = jax.random.key(seed)
    for k in range(len(mu_list)):
        idx = np.where(component_ids == k)[0]
        if idx.size == 0:
            continue

        key, subkey = jax.random.split(key)
        v = jax.random.multivariate_normal(
            subkey,
            jnp.zeros(mu_list[k].shape[0]),
            sigma_list[k],
            shape=(idx.size,),
        )
        samples_k = jax.vmap(lambda vv: manifold.exp_map(mu_list[k], vv))(v)
        out[idx] = np.array(samples_k)

    return out


def predict_land_labels(
    X: np.ndarray,
    mu_list: list[jnp.ndarray],
    sigma_list: list[jnp.ndarray],
    C_list: list[jnp.ndarray],
    pi: jnp.ndarray,
    manifold: RiemannianManifold,
) -> np.ndarray:
    X_jnp = jnp.array(X, dtype=jnp.float32)
    log_probs = []
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
        log_p = jnp.log(pi[k] + 1e-12) - jnp.log(C_list[k] + 1e-12) - 0.5 * dist_sq
        log_probs.append(np.array(log_p))

    return np.argmax(np.stack(log_probs, axis=1), axis=1)


def fit_land(
    X_jnp: jnp.ndarray,
    K: int,
    cfg: ExperimentConfig,
    seed: int,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray], list[jnp.ndarray], jnp.ndarray]:
    land = LANDMixtureModel(
        K=K,
        lr_mu=cfg.land_lr_mu,
        lr_A=cfg.land_lr_A,
        S=cfg.land_S,
        epsilon=cfg.land_eps,
        sigma=cfg.sigma_metric,
        rho=cfg.rho_metric,
        K_segments=cfg.K_segments,
        n_neighbors=cfg.n_neighbors,
        init_method="GMM",
        seed=seed,
    )
    return land.fit(X_jnp)


def run_nll_experiment(
    cfg: ExperimentConfig, output_dir: Path
) -> dict[str, dict[str, list[float]]]:
    print(
        f"[SYNTH] NLL experiment: datasets={cfg.n_datasets}, K={cfg.k_min}..{cfg.k_max}, eval_samples={cfg.n_eval_samples}",
        flush=True,
    )
    means = true_component_means(cfg)
    Ks = list(range(cfg.k_min, cfg.k_max + 1))
    models = ["LAND", "GMM", "LeastSquares"]

    nll_scores = {m: {str(k): [] for k in Ks} for m in models}

    for ds_idx in tqdm(
        range(cfg.n_datasets), desc="[SYNTH] Datasets (NLL)", unit="dataset"
    ):
        X_t, _, _ = sample_non_linear_data(
            n_samples=cfg.n_samples_per_dataset,
            n_components=cfg.n_true_components,
            x_rad=cfg.x_rad,
            y_rad=cfg.y_rad,
            std=cfg.std,
            return_labels=True,
            seed=cfg.seed + ds_idx,
        )
        X = X_t.cpu().numpy()
        X_jnp = jnp.array(X, dtype=jnp.float32)

        manifold = RiemannianManifold(
            X_jnp,
            sigma=cfg.sigma_metric,
            rho=cfg.rho_metric,
            K_segments=cfg.K_segments,
            n_neighbors=cfg.n_neighbors,
        )

        for K in tqdm(
            Ks, desc=f"[SYNTH] Dataset {ds_idx + 1} K-sweep", unit="K", leave=False
        ):
            gmm = GaussianMixture(
                n_components=K, covariance_type="full", random_state=cfg.seed + ds_idx
            )
            gmm.fit(X)
            gmm_samples, _ = gmm.sample(cfg.n_eval_samples)
            gmm_nll = -true_logpdf(gmm_samples, cfg, means).mean()
            nll_scores["GMM"][str(K)].append(float(gmm_nll))

            ls_model = LeastSquaresGaussianModel.fit(
                X,
                K=K,
                seed=cfg.seed + ds_idx,
                n_init=cfg.kmeans_n_init,
            )
            ls_samples = ls_model.sample(
                cfg.n_eval_samples, seed=cfg.seed + 10_000 + ds_idx
            )
            ls_nll = -true_logpdf(ls_samples, cfg, means).mean()
            nll_scores["LeastSquares"][str(K)].append(float(ls_nll))

            land_mu, land_sigma, land_C, land_pi = fit_land(
                X_jnp, K=K, cfg=cfg, seed=cfg.seed + ds_idx
            )
            land_samples = sample_land_mixture(
                land_mu,
                land_sigma,
                land_pi,
                manifold,
                n_samples=cfg.n_eval_samples,
                seed=cfg.seed + 20_000 + ds_idx + 100 * K,
            )
            land_nll = -true_logpdf(land_samples, cfg, means).mean()
            nll_scores["LAND"][str(K)].append(float(land_nll))

            print(
                f"Dataset {ds_idx + 1}/{cfg.n_datasets} | K={K} | "
                f"NLL LAND={land_nll:.4f} GMM={gmm_nll:.4f} LS={ls_nll:.4f}"
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in models:
        means_k = [np.mean(nll_scores[model_name][str(k)]) for k in Ks]
        stds_k = [np.std(nll_scores[model_name][str(k)]) for k in Ks]
        ax.errorbar(Ks, means_k, yerr=stds_k, marker="o", capsize=4, label=model_name)

    ax.set_title("Synthetic experiment: mean NLL under true generator")
    ax.set_xlabel("Number of components K")
    ax.set_ylabel("Mean negative log-likelihood")
    ax.set_xticks(Ks)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / "synthetic_nll_vs_k.png"), dpi=180)
    plt.close(fig)

    return nll_scores


def run_contour_experiment(cfg: ExperimentConfig, output_dir: Path) -> dict[str, float]:
    print("[SYNTH] Contour experiment (K=2) starting", flush=True)
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

    manifold = RiemannianManifold(
        X_jnp,
        sigma=cfg.sigma_metric,
        rho=cfg.rho_metric,
        K_segments=cfg.K_segments,
        n_neighbors=cfg.n_neighbors,
    )

    K = 2
    gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=cfg.seed)
    gmm.fit(X)

    land_mu, land_sigma, land_C, land_pi = fit_land(X_jnp, K=K, cfg=cfg, seed=cfg.seed)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, cfg.contour_grid_size),
        np.linspace(y_min, y_max, cfg.contour_grid_size),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    z_gmm = np.exp(gmm.score_samples(grid_points)).reshape(xx.shape)
    z_land = evaluate_land_density_grid(
        xx,
        yy,
        land_mu,
        land_sigma,
        land_C,
        land_pi,
        manifold,
        cutoff_std=cfg.contour_cutoff_std,
    )
    print(
        f"[SYNTH] Contour grid evaluated with cutoff_std={cfg.contour_cutoff_std}",
        flush=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_mixture_contours(
        axes[0],
        X,
        np.array(jnp.stack(land_mu)),
        xx,
        yy,
        z_land,
        title="LAND mixture contours (K=2)",
        mean_label="LAND means",
    )
    plot_mixture_contours(
        axes[1],
        X,
        gmm.means_,
        xx,
        yy,
        z_gmm,
        title="GMM contours (K=2)",
        mean_label="GMM means",
    )
    fig.tight_layout()
    fig.savefig(str(output_dir / "synthetic_contours_k2.png"), dpi=180)
    plt.close(fig)

    return {
        "land_weight_sum": float(np.array(land_pi).sum()),
        "gmm_weight_sum": float(gmm.weights_.sum()),
    }


def run_clustering_experiment(
    cfg: ExperimentConfig, output_dir: Path
) -> dict[str, float]:
    print(
        f"[SYNTH] Clustering experiment: datasets={cfg.n_datasets}, K={cfg.clustering_K}",
        flush=True,
    )
    ari_scores = {"LAND": [], "GMM": [], "LeastSquares": []}
    nmi_scores = {"LAND": [], "GMM": [], "LeastSquares": []}

    for ds_idx in tqdm(
        range(cfg.n_datasets), desc="[SYNTH] Datasets (cluster)", unit="dataset"
    ):
        X_t, _, y_t = sample_non_linear_data(
            n_samples=cfg.n_samples_per_dataset,
            n_components=cfg.n_true_components,
            x_rad=cfg.x_rad,
            y_rad=cfg.y_rad,
            std=cfg.std,
            return_labels=True,
            seed=cfg.seed + ds_idx,
        )
        X = X_t.cpu().numpy()
        y_true = y_t.cpu().numpy()
        X_jnp = jnp.array(X, dtype=jnp.float32)

        K = cfg.clustering_K

        gmm = GaussianMixture(
            n_components=K, covariance_type="full", random_state=cfg.seed + ds_idx
        )
        y_gmm = gmm.fit_predict(X)

        ls_model = LeastSquaresGaussianModel.fit(
            X,
            K=K,
            seed=cfg.seed + ds_idx,
            n_init=cfg.kmeans_n_init,
        )
        y_ls = ls_model.predict(X)

        land_mu, land_sigma, land_C, land_pi = fit_land(
            X_jnp,
            K=K,
            cfg=cfg,
            seed=cfg.seed + ds_idx,
        )
        manifold = RiemannianManifold(
            X_jnp,
            sigma=cfg.sigma_metric,
            rho=cfg.rho_metric,
            K_segments=cfg.K_segments,
            n_neighbors=cfg.n_neighbors,
        )
        y_land = predict_land_labels(
            X,
            land_mu,
            land_sigma,
            land_C,
            land_pi,
            manifold,
        )

        ari_scores["LAND"].append(adjusted_rand_score(y_true, y_land))
        ari_scores["GMM"].append(adjusted_rand_score(y_true, y_gmm))
        ari_scores["LeastSquares"].append(adjusted_rand_score(y_true, y_ls))

        nmi_scores["LAND"].append(normalized_mutual_info_score(y_true, y_land))
        nmi_scores["GMM"].append(normalized_mutual_info_score(y_true, y_gmm))
        nmi_scores["LeastSquares"].append(normalized_mutual_info_score(y_true, y_ls))

        print(
            f"Clustering dataset {ds_idx + 1}/{cfg.n_datasets} | "
            f"ARI LAND={ari_scores['LAND'][-1]:.4f} GMM={ari_scores['GMM'][-1]:.4f} LS={ari_scores['LeastSquares'][-1]:.4f}"
        )

    summary = {}
    for model in ["LAND", "GMM", "LeastSquares"]:
        summary[f"ARI_{model}_mean"] = float(np.mean(ari_scores[model]))
        summary[f"ARI_{model}_std"] = float(np.std(ari_scores[model]))
        summary[f"NMI_{model}_mean"] = float(np.mean(nmi_scores[model]))
        summary[f"NMI_{model}_std"] = float(np.std(nmi_scores[model]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    model_names = ["LAND", "GMM", "LeastSquares"]

    axes[0].bar(
        model_names,
        [summary[f"ARI_{m}_mean"] for m in model_names],
        yerr=[summary[f"ARI_{m}_std"] for m in model_names],
        capsize=4,
    )
    axes[0].set_title("Clustering ARI")
    axes[0].set_ylabel("Adjusted Rand Index")

    axes[1].bar(
        model_names,
        [summary[f"NMI_{m}_mean"] for m in model_names],
        yerr=[summary[f"NMI_{m}_std"] for m in model_names],
        capsize=4,
    )
    axes[1].set_title("Clustering NMI")
    axes[1].set_ylabel("Normalized Mutual Information")

    fig.tight_layout()
    fig.savefig(str(output_dir / "synthetic_clustering_scores.png"), dpi=180)
    plt.close(fig)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAND paper synthetic experiments")
    parser.add_argument("--output-dir", type=str, default="plots/synthetic_land_paper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-datasets", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=220)
    parser.add_argument("--n-eval-samples", type=int, default=2000)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--clustering-k", type=int, default=10)
    parser.add_argument("--sigma-metric", type=float, default=0.15)
    parser.add_argument("--rho-metric", type=float, default=1e-3)
    parser.add_argument("--k-segments", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=7)
    parser.add_argument("--land-S", type=int, default=400)
    parser.add_argument("--land-eps", type=float, default=1e-3)
    parser.add_argument("--contour-cutoff-std", type=float, default=2.0)
    parser.add_argument("--contour-grid-size", type=int, default=28)
    parser.add_argument("--kmeans-n-init", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        n_datasets=args.n_datasets,
        n_samples_per_dataset=args.n_samples,
        k_min=args.k_min,
        k_max=args.k_max,
        n_eval_samples=args.n_eval_samples,
        sigma_metric=args.sigma_metric,
        rho_metric=args.rho_metric,
        K_segments=args.k_segments,
        n_neighbors=args.n_neighbors,
        land_S=args.land_S,
        land_eps=args.land_eps,
        contour_cutoff_std=args.contour_cutoff_std,
        contour_grid_size=args.contour_grid_size,
        seed=args.seed,
        clustering_K=args.clustering_k,
        kmeans_n_init=args.kmeans_n_init,
    )

    print("Running synthetic NLL experiment...")
    nll_results = run_nll_experiment(cfg, output_dir)

    print("Running K=2 contour experiment...")
    contour_stats = run_contour_experiment(cfg, output_dir)

    print("Running clustering experiment...")
    clustering_results = run_clustering_experiment(cfg, output_dir)

    payload = {
        "config": cfg.__dict__,
        "nll": nll_results,
        "contours": contour_stats,
        "clustering": clustering_results,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
