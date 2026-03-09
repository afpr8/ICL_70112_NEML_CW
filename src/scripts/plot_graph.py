

from src.scripts.main_mm import make_moons
from src.utils.plotting_utils import plot_manifold_graph
from src.utils.land_utils import RiemannianManifold

if __name__ == "__main__":
    # 1. Generate the Two Moons dataset
    X, _ = make_moons(n_samples=400, noise=0.1, random_state=42)
    
    # 2. Initialise the minimal manifold
    # Try changing n_neighbors here to see how it bridges the empty space!
    manifold = RiemannianManifold(X, sigma=0.25, rho=1e-3, n_neighbors=5)
    
    # 3. Plot the result
    plot_manifold_graph(manifold)