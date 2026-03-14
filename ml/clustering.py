import numpy as np
from sklearn.cluster import KMeans

def perform_kmeans(X_pca: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """
    Applies K-Means clustering to identify distinct thermodynamic phases.
    
    Args:
        X_pca (np.ndarray): The dataset projected into the PCA latent space.
        n_clusters (int): The expected number of macroscopic states (e.g., 3).
        
    Returns:
        np.ndarray: Cluster labels for each sample.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    return labels