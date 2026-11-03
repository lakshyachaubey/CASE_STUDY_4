import pandas as pd


MARKETING_ACTIONS = {
    "Champions":
        "VIP rewards, exclusive offers and loyalty programs",

    "Loyal Customers":
        "Membership programs, cross-selling and referrals",

    "Recent / New Customers":
        "Welcome discounts and second-purchase incentives",

    "At-Risk High-Value Customers":
        "Personalized win-back campaigns and loyalty incentives",

    "Big Spenders / Occasional Buyers":
        "Premium recommendations and complementary products",

    "Hibernating Customers":
        "Reactivation emails and limited-time discounts",

    "Regular / Developing Customers":
        "Personalized product recommendations and bundles",

    "Noise / Outlier Customers":
        "Investigate unusual purchasing behaviour"
}


def assign_persona(row, thresholds):
    """
    Assign a business persona using cluster-level
    median RFM values and global RFM quartiles.
    """

    recency = row["MedianRecency"]
    frequency = row["MedianFrequency"]
    monetary = row["MedianMonetary"]

    q25 = thresholds.loc[0.25]
    q50 = thresholds.loc[0.50]
    q75 = thresholds.loc[0.75]

    if (
        recency <= q25["Recency"]
        and frequency >= q75["Frequency"]
        and monetary >= q75["Monetary"]
    ):
        return "Champions"

    elif (
        recency >= q75["Recency"]
        and frequency >= q75["Frequency"]
        and monetary >= q50["Monetary"]
    ):
        return "At-Risk High-Value Customers"

    elif (
        frequency >= q75["Frequency"]
        and recency <= q75["Recency"]
    ):
        return "Loyal Customers"

    elif (
        monetary >= q75["Monetary"]
        and frequency <= q50["Frequency"]
    ):
        return "Big Spenders / Occasional Buyers"

    elif (
        recency <= q25["Recency"]
        and frequency <= q50["Frequency"]
    ):
        return "Recent / New Customers"

    elif (
        recency >= q75["Recency"]
        and frequency <= q50["Frequency"]
        and monetary <= q50["Monetary"]
    ):
        return "Hibernating Customers"

    return "Regular / Developing Customers"


def build_cluster_profiles(rfm, cluster_column):
    """
    Create customer statistics for each cluster.
    """

    thresholds = rfm[
        ["Recency", "Frequency", "Monetary"]
    ].quantile([0.25, 0.50, 0.75])

    profiles = (
        rfm.groupby(cluster_column)
        .agg(
            CustomerCount=("CustomerID", "nunique"),

            MedianRecency=("Recency", "median"),

            MedianFrequency=("Frequency", "median"),

            MedianMonetary=("Monetary", "median"),

            AverageMonetary=("Monetary", "mean")
        )
        .reset_index()
        .rename(columns={cluster_column: "Cluster"})
    )

    personas = []

    for _, row in profiles.iterrows():

        if (
            cluster_column == "DBSCANCluster"
            and row["Cluster"] == -1
        ):
            persona = "Noise / Outlier Customers"

        else:
            persona = assign_persona(
                row,
                thresholds
            )

        personas.append(persona)

    profiles["Persona"] = personas

    profiles["MarketingAction"] = (
        profiles["Persona"]
        .map(MARKETING_ACTIONS)
    )

    return profiles


def attach_personas(rfm, kmeans_profile, dbscan_profile):
    """
    Add persona names and marketing actions
    to individual customer records.
    """

    kmeans_map = (
        kmeans_profile
        .set_index("Cluster")["Persona"]
        .to_dict()
    )

    dbscan_map = (
        dbscan_profile
        .set_index("Cluster")["Persona"]
        .to_dict()
    )

    rfm = rfm.copy()

    rfm["KMeansPersona"] = (
        rfm["KMeansCluster"]
        .map(kmeans_map)
    )

    rfm["DBSCANPersona"] = (
        rfm["DBSCANCluster"]
        .map(dbscan_map)
    )

    rfm["MarketingRecommendation"] = (
        rfm["KMeansPersona"]
        .map(MARKETING_ACTIONS)
    )

    return rfm