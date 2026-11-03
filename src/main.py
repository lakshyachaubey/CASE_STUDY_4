import json
import sys

from pathlib import Path

# Allow imports from src/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

sys.path.insert(0, str(CURRENT_DIR))

from data_preprocessing import load_and_clean_data

from rfm_analysis import (
    calculate_rfm,
    scale_rfm_features
)

from train_model import (
    train_kmeans,
    tune_dbscan,
    evaluate_models
)

from customer_personas import (
    build_cluster_profiles,
    attach_personas
)

from visualization import create_visualizations


DATA_PATH = PROJECT_ROOT / "data" / "online_retail_II.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("=" * 60)
    print("E-COMMERCE CUSTOMER SEGMENTATION")
    print("=" * 60)

    # 1. Data preprocessing
    data = load_and_clean_data(
        DATA_PATH
    )

    # 2. RFM analysis
    rfm = calculate_rfm(
        data
    )

    rfm.to_csv(
        RESULTS_DIR / "rfm_dataset.csv",
        index=False
    )

    # 3. Feature scaling
    X_scaled, scaler = scale_rfm_features(
        rfm
    )

    # 4. K-Means clustering
    (
        kmeans_model,
        kmeans_labels,
        kmeans_search
    ) = train_kmeans(
        X_scaled
    )

    # 5. DBSCAN clustering
    (
        dbscan_model,
        dbscan_labels,
        dbscan_search,
        sorted_distances,
        best_eps
    ) = tune_dbscan(
        X_scaled,
        min_samples=10
    )

    # 6. Store customer cluster labels
    rfm["KMeansCluster"] = kmeans_labels

    rfm["DBSCANCluster"] = dbscan_labels

    # 7. Generate customer profiles
    kmeans_profile = build_cluster_profiles(
        rfm,
        "KMeansCluster"
    )

    dbscan_profile = build_cluster_profiles(
        rfm,
        "DBSCANCluster"
    )

    # 8. Assign marketing personas
    customer_segments = attach_personas(
        rfm,
        kmeans_profile,
        dbscan_profile
    )

    # 9. Save customer segmentation results
    customer_segments.to_csv(
        RESULTS_DIR / "customer_segments.csv",
        index=False
    )

    kmeans_profile.to_csv(
        RESULTS_DIR / "cluster_profiles.csv",
        index=False
    )

    dbscan_profile.to_csv(
        RESULTS_DIR / "dbscan_cluster_profiles.csv",
        index=False
    )

    # Save model search results
    kmeans_search.to_csv(
        RESULTS_DIR / "kmeans_search.csv",
        index=False
    )

    dbscan_search.to_csv(
        RESULTS_DIR / "dbscan_search.csv",
        index=False
    )

    # 10. Evaluate both models
    metrics = evaluate_models(
        X_scaled,
        kmeans_labels,
        dbscan_labels
    )

    metrics["dataset"] = {
        "cleaned_transactions": int(len(data)),
        "customers": int(len(rfm)),
        "reference_date": (
            data["InvoiceDate"].max().normalize()
            + __import__("pandas").Timedelta(days=1)
        ).strftime("%Y-%m-%d")
    }

    metrics["model_parameters"] = {
        "kmeans_k": int(kmeans_model.n_clusters),
        "dbscan_eps": float(best_eps),
        "dbscan_min_samples": 10
    }

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # 11. Generate visualizations
    create_visualizations(
        data=data,
        rfm=customer_segments,
        X_scaled=X_scaled,
        kmeans_labels=kmeans_labels,
        dbscan_labels=dbscan_labels,
        kmeans_search=kmeans_search,
        dbscan_search=dbscan_search,
        sorted_distances=sorted_distances,
        cluster_profile=kmeans_profile,
        output_dir=RESULTS_DIR
    )

    # 12. Final report
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nTotal customers:", len(rfm))

    print("\nK-Means:")
    print("Clusters:", kmeans_model.n_clusters)
    print(
        "Silhouette:",
        metrics["KMeans"]["silhouette_score"]
    )

    print("\nDBSCAN:")
    print(
        "Clusters:",
        metrics["DBSCAN"]["clusters"]
    )

    print(
        "Noise percentage:",
        round(
            metrics["DBSCAN"]["noise_percentage"],
            2
        )
    )

    print("\nCustomer personas:")
    print(
        kmeans_profile[
            ["Cluster", "Persona", "CustomerCount"]
        ].to_string(index=False)
    )

    print("\nAll outputs saved inside results/")


if __name__ == "__main__":
    main()