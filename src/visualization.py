from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_visualizations(
    data,
    rfm,
    X_scaled,
    kmeans_labels,
    dbscan_labels,
    kmeans_search,
    dbscan_search,
    sorted_distances,
    cluster_profile,
    output_dir,
    max_tsne_samples=5000
):
    """
    Generate and save all project visualizations.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sns.set_theme(style="whitegrid")

    # -----------------------------------
    # 1. Monthly revenue
    # -----------------------------------

    monthly = (
        data.set_index("InvoiceDate")
        .resample("ME")["Revenue"]
        .sum()
    )

    plt.figure(figsize=(12, 5))

    monthly.plot()

    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")

    plt.tight_layout()

    plt.savefig(
        output_dir / "monthly_revenue.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------
    # 2. Top countries
    # -----------------------------------

    if "Country" in data.columns:

        country_revenue = (
            data.groupby("Country")["Revenue"]
            .sum()
            .nlargest(10)
            .sort_values()
        )

        plt.figure(figsize=(10, 6))

        country_revenue.plot(
            kind="barh"
        )

        plt.title("Top 10 Countries by Revenue")
        plt.xlabel("Revenue")
        plt.ylabel("Country")

        plt.tight_layout()

        plt.savefig(
            output_dir / "top_countries.png",
            dpi=300
        )

        plt.close()

    # -----------------------------------
    # 3. RFM distributions
    # -----------------------------------

    features = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5)
    )

    for ax, feature in zip(axes, features):

        sns.histplot(
            rfm[feature],
            bins=40,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"{feature} Distribution"
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "rfm_distributions.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------
    # 4. RFM correlation
    # -----------------------------------

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        rfm[features].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("RFM Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        output_dir / "rfm_correlation.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------
    # 5. Elbow Method
    # -----------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        kmeans_search["K"],
        kmeans_search["Inertia"],
        marker="o"
    )

    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.tight_layout()

    plt.savefig(
        output_dir / "elbow_method.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------
    # 6. Silhouette comparison for K
    # -----------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        kmeans_search["K"],
        kmeans_search["Silhouette"],
        marker="o"
    )

    plt.title("K-Means Silhouette Scores")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()

    plt.savefig(
        output_dir / "silhouette_scores.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------
    # 7. DBSCAN parameter search
    # -----------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        sorted_distances,
        marker="."
    )

    plt.title("DBSCAN K-Distance Graph")
    plt.xlabel("Sorted Customers")
    plt.ylabel("Distance to Kth Nearest Neighbor")

    plt.tight_layout()

    plt.savefig(
        output_dir / "dbscan_k_distance.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------
    # 8. PCA
    # -----------------------------------

    pca_full = PCA()

    pca_full.fit(X_scaled)

    cumulative_variance = (
        pca_full.explained_variance_ratio_.cumsum()
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o"
    )

    plt.title("PCA Explained Variance")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance")

    plt.xticks(
        range(1, len(cumulative_variance) + 1)
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "pca_variance.png",
        dpi=300
    )

    plt.close()

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "KMeansCluster": kmeans_labels.astype(str),
        "DBSCANCluster": dbscan_labels.astype(str)
    })

    for cluster_col, filename, title in [
        (
            "KMeansCluster",
            "pca_kmeans.png",
            "K-Means Clusters Using PCA"
        ),
        (
            "DBSCANCluster",
            "pca_dbscan.png",
            "DBSCAN Clusters Using PCA"
        )
    ]:

        plt.figure(figsize=(10, 7))

        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=cluster_col,
            palette="tab10",
            s=25,
            alpha=0.7
        )

        plt.title(title)

        plt.tight_layout()

        plt.savefig(
            output_dir / filename,
            dpi=300
        )

        plt.close()

    # -----------------------------------
    # 9. t-SNE
    # -----------------------------------

    n_samples = min(
        max_tsne_samples,
        len(X_scaled)
    )

    if n_samples >= 4:

        rng = np.random.default_rng(42)

        sample_indices = np.sort(
            rng.choice(
                len(X_scaled),
                size=n_samples,
                replace=False
            )
        )

        X_sample = X_scaled[sample_indices]

        perplexity = min(
            30,
            max(2, (n_samples - 1) // 3)
        )

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
            max_iter=1000
        )

        X_tsne = tsne.fit_transform(
            X_sample
        )

        tsne_df = pd.DataFrame({
            "TSNE1": X_tsne[:, 0],
            "TSNE2": X_tsne[:, 1],
            "KMeansCluster":
                kmeans_labels[sample_indices].astype(str),
            "DBSCANCluster":
                dbscan_labels[sample_indices].astype(str)
        })

        for cluster_col, filename, title in [
            (
                "KMeansCluster",
                "tsne_kmeans.png",
                "K-Means Clusters Using t-SNE"
            ),
            (
                "DBSCANCluster",
                "tsne_dbscan.png",
                "DBSCAN Clusters Using t-SNE"
            )
        ]:

            plt.figure(figsize=(11, 8))

            sns.scatterplot(
                data=tsne_df,
                x="TSNE1",
                y="TSNE2",
                hue=cluster_col,
                palette="tab10",
                s=25,
                alpha=0.7
            )

            plt.title(title)

            plt.tight_layout()

            plt.savefig(
                output_dir / filename,
                dpi=300
            )

            plt.close()

    # -----------------------------------
    # 10. Cluster RFM heatmap
    # -----------------------------------

    heatmap_data = (
        cluster_profile
        .set_index("Cluster")[
            [
                "MedianRecency",
                "MedianFrequency",
                "MedianMonetary"
            ]
        ]
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu"
    )

    plt.title("K-Means Cluster RFM Profiles")

    plt.tight_layout()

    plt.savefig(
        output_dir / "cluster_rfm_heatmap.png",
        dpi=300
    )

    plt.close()

    # -----------------------------------
    # 11. Marketing personas
    # -----------------------------------

    persona_counts = (
        cluster_profile.groupby("Persona")["CustomerCount"]
        .sum()
        .sort_values()
    )

    plt.figure(figsize=(11, 6))

    persona_counts.plot(
        kind="barh"
    )

    plt.title("Customer Distribution by Marketing Persona")
    plt.xlabel("Number of Customers")
    plt.ylabel("Persona")

    plt.tight_layout()

    plt.savefig(
        output_dir / "customer_personas.png",
        dpi=300
    )

    plt.close()

    print("\nVISUALIZATIONS GENERATED")
    print("Saved to:", output_dir)