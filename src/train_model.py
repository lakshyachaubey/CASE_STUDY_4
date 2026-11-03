import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN

from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)


RANDOM_STATE = 42


def safe_silhouette(X, labels):
    """
    Calculate silhouette score safely.
    Uses a sample for large datasets.
    """

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return None

    if len(X) <= len(unique_labels):
        return None

    return silhouette_score(
        X,
        labels,
        sample_size=min(5000, len(X)),
        random_state=RANDOM_STATE
    )


def train_kmeans(X_scaled):
    """
    Find a suitable K and train K-Means.
    """

    results = []

    max_k = min(9, len(X_scaled) - 1)

    for k in range(2, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = safe_silhouette(
            X_scaled,
            labels
        )

        results.append({
            "K": k,
            "Inertia": float(model.inertia_),
            "Silhouette": score
        })

    search_results = pd.DataFrame(results)

    valid = search_results.dropna(
        subset=["Silhouette"]
    )

    if valid.empty:
        raise ValueError(
            "Unable to evaluate K-Means clusters."
        )

    best_k = int(
        valid.loc[
            valid["Silhouette"].idxmax(),
            "K"
        ]
    )

    # Train final K-Means model
    model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    print("\nK-MEANS COMPLETED")
    print("-" * 40)
    print("Selected K:", best_k)

    return model, labels, search_results


def tune_dbscan(X_scaled, min_samples=10):
    """
    Evaluate DBSCAN epsilon candidates using
    nearest-neighbour distances and silhouette score.
    """

    neighbors = NearestNeighbors(
        n_neighbors=min_samples
    )

    neighbors.fit(X_scaled)

    distances, _ = neighbors.kneighbors(X_scaled)

    # Distance to the kth nearest neighbour
    sorted_distances = np.sort(
        distances[:, -1]
    )

    quantiles = [
        0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90
    ]

    eps_candidates = np.unique(
        np.quantile(
            sorted_distances,
            quantiles
        )
    )

    results = []

    best_model = None
    best_labels = None
    best_score = -np.inf
    best_eps = None

    for eps in eps_candidates:

        model = DBSCAN(
            eps=float(eps),
            min_samples=min_samples
        )

        labels = model.fit_predict(X_scaled)

        non_noise = labels != -1

        clusters = np.unique(labels[non_noise])

        noise_rate = float(
            np.mean(~non_noise)
        )

        score = None

        if (
            len(clusters) >= 2
            and non_noise.sum() > len(clusters)
        ):

            score = safe_silhouette(
                X_scaled[non_noise],
                labels[non_noise]
            )

        results.append({
            "Epsilon": float(eps),
            "Clusters": len(clusters),
            "NoisePercentage": noise_rate * 100,
            "Silhouette": score
        })

        # Avoid choosing an extreme solution
        # with almost all observations marked as noise
        if (
            score is not None
            and 0.01 <= noise_rate <= 0.95
            and score > best_score
        ):

            best_score = score
            best_model = model
            best_labels = labels
            best_eps = float(eps)

    search_results = pd.DataFrame(results)

    # Fallback if no candidate has a valid score
    if best_model is None:

        best_eps = float(
            np.quantile(
                sorted_distances,
                0.90
            )
        )

        best_model = DBSCAN(
            eps=best_eps,
            min_samples=min_samples
        )

        best_labels = best_model.fit_predict(
            X_scaled
        )

    print("\nDBSCAN COMPLETED")
    print("-" * 40)
    print("Selected epsilon:", best_eps)
    print("Minimum samples:", min_samples)

    cluster_count = len(
        set(best_labels) - {-1}
    )

    noise_count = int(
        np.sum(best_labels == -1)
    )

    print("Identified clusters:", cluster_count)
    print("Noise observations:", noise_count)

    return (
        best_model,
        best_labels,
        search_results,
        sorted_distances,
        best_eps
    )


def evaluate_models(
    X_scaled,
    kmeans_labels,
    dbscan_labels
):
    """
    Evaluate K-Means and DBSCAN.
    DBSCAN metrics exclude noise observations.
    """

    # K-Means
    km_silhouette = safe_silhouette(
        X_scaled,
        kmeans_labels
    )

    km_db_index = davies_bouldin_score(
        X_scaled,
        kmeans_labels
    )

    # DBSCAN
    mask = dbscan_labels != -1

    non_noise_labels = dbscan_labels[mask]

    db_silhouette = None
    db_db_index = None

    if (
        len(np.unique(non_noise_labels)) >= 2
        and mask.sum() > len(np.unique(non_noise_labels))
    ):

        db_silhouette = safe_silhouette(
            X_scaled[mask],
            non_noise_labels
        )

        db_db_index = davies_bouldin_score(
            X_scaled[mask],
            non_noise_labels
        )

    metrics = {
        "KMeans": {
            "clusters": int(
                len(np.unique(kmeans_labels))
            ),
            "silhouette_score": (
                float(km_silhouette)
                if km_silhouette is not None
                else None
            ),
            "davies_bouldin_index": float(km_db_index)
        },

        "DBSCAN": {
            "clusters": int(
                len(np.unique(non_noise_labels))
            ),
            "noise_count": int(
                np.sum(~mask)
            ),
            "noise_percentage": float(
                np.mean(~mask) * 100
            ),
            "silhouette_score": (
                float(db_silhouette)
                if db_silhouette is not None
                else None
            ),
            "davies_bouldin_index": (
                float(db_db_index)
                if db_db_index is not None
                else None
            )
        }
    }

    return metrics