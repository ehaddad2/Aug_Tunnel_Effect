import numpy as np
from sklearn.metrics import pairwise_distances


# ---------- k‑means++ for (bucketed) augmentation vectors ----------
def kmeans_pp_init(X: np.ndarray,
                   k: int,
                   random_state: int | None = None) -> np.ndarray:
    """
    k‑means++ initialisation.
    X : (n_samples, n_features) float array
    k : number of centroids
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    # 1. first centre
    centres = [X[rng.integers(0, n_samples)]]

    # 2.–k. weighted by squared distance to nearest chosen centre
    for _ in range(1, k):
        d2 = np.min(pairwise_distances(X, np.vstack(centres)), axis=1) ** 2
        probs = d2 / d2.sum()
        centres.append(X[rng.choice(n_samples, p=probs)])

    return np.vstack(centres)


# ---------- helper: sample bucketed augmentation combos ----------
def sample_bucketed_combos(sample_size: int,
                           bounds: list[tuple[float, float]],
                           levels: tuple[float, ...] = (0, 0.1, 0.3, 0.5, 0.7),
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Draw `sample_size` vectors of length len(bounds) where each coordinate
    is chosen from `levels` *within* its (min, max) bound.

    All‑zero rows are removed.
    """
    rng = np.random.default_rng(rng)
    n_bits = len(bounds)
    # Build a (n_bits, |levels|) mask of permissible levels
    allowed = [
        [lvl for lvl in levels if low <= lvl <= high]
        for (low, high) in bounds
    ]
    # Sample per dimension
    data = np.column_stack([
        rng.choice(dim_levels, size=sample_size)
        for dim_levels in allowed
    ])
    # drop all‑zero rows
    return data[data.any(axis=1)]


# ------------------------------ example ------------------------------
if __name__ == "__main__":
    # Per‑augmentation bounds  (min, max)  — blur, noise, invert … capped at 0.1
    aug_bounds = [
        (0, 0.3),   # Horizontal flip
        (0, 0.3),   # Resize crop
        (0, 0.3),   # Random affine
        (0, 0.3),   # Scale jitter
        (0, 0.3),   # Gaussian blur   (semantic‑loss capped)
        (0, 0.3),   # Gaussian noise  (semantic‑loss capped)
        (0, 0.3),   # Color jitter
        (0, 0.3),   # Color distortion
        (0, 0.3),   # Random invert   (semantic‑loss capped)
        (0, 0.3),   # Solarise        (semantic‑loss capped)
        (0, 0.3),   # Autocontrast    (semantic‑loss capped)
        (0, 0.3),   # CutOut
        (0, 0.3),   # Tempered MixUp
        (0, 0.3),   # CutMix
    ]

    X = sample_bucketed_combos(sample_size=200_000, bounds=aug_bounds)
    init_centres = kmeans_pp_init(X, k=10, random_state=42)

    # ensure mandatory zeros on dims 3, 11, 13 if needed
    init_centres[:, [3, 11, 13]] = 0

    print("k‑means++ initial centres:")
    for c in init_centres:
        print(c.tolist())
