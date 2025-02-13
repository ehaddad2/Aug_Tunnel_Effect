import numpy as np
from sklearn.metrics import pairwise_distances

def kmeans_pp_init(X, k, random_state=None):
    """
    Implements k-means++ initialization for binary data.
    X: binary data points (n_samples x n_features).
    k: number of clusters.
    random_state: seed for reproducibility.
    Returns an array of k initial centers.
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    
    # Step 1: Choose the first center uniformly at random
    first_center_idx = rng.integers(0, n_samples)
    centers = [X[first_center_idx]]
    
    # Steps 2-3: Choose each subsequent center using weighted probability
    for _ in range(1, k):
        # Compute the Hamming distance from each point to its nearest chosen center
        dist = np.min(pairwise_distances(X, np.array(centers)), axis=1)
        dist_sq = dist ** 2
        
        # Probability of picking a point is proportional to dist_sq
        probs = dist_sq / np.sum(dist_sq)
        new_center_idx = rng.choice(n_samples, p=probs)
        
        centers.append(X[new_center_idx])
    
    return np.array(centers)

# Generate 4095 12-dimensional binary vectors (all possible combinations except all-zero)
def generate_binary_data(n_bits=12):
    return np.array([list(map(int, bin(i)[2:].zfill(n_bits))) for i in range(1, 2**n_bits)])

def sample_bucketed_combos(n_bits=14, sample_size=10000000):
    levels = [0, 0.1, 0.2, 0.3]
    data = np.random.choice(levels, size=(sample_size, n_bits))
    # Filter out all-zero rows
    mask = np.any(data != 0, axis=1)
    return data[mask]

# Example Usage
if __name__ == "__main__":
    # Generate binary data
    X = sample_bucketed_combos(n_bits=14)
    
    # Initialize 5 centroids using kmeans++ for binary data
    centers = kmeans_pp_init(X, k=14, random_state=42)
    print("Selected centers:\n")
    for center in centers:
        print(f"[{','.join(map(str, center))}]")