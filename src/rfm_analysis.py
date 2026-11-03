import pandas as pd


def calculate_rfm(data):
    """
    Calculate customer-level RFM metrics.

    Recency   = Days since last purchase
    Frequency = Number of distinct invoices
    Monetary  = Total customer spending
    """

    # Snapshot date: one day after last transaction
    reference_date = (
        data["InvoiceDate"].max().normalize()
        + pd.Timedelta(days=1)
    )

    rfm = data.groupby("CustomerID").agg(
        LastPurchaseDate=(
            "InvoiceDate",
            "max"
        ),

        Frequency=(
            "InvoiceNo",
            "nunique"
        ),

        Monetary=(
            "Revenue",
            "sum"
        )
    ).reset_index()

    # Calculate recency
    rfm["Recency"] = (
        reference_date -
        rfm["LastPurchaseDate"].dt.normalize()
    ).dt.days

    # Keep valid customers
    rfm = rfm[
        (rfm["Recency"] >= 0) &
        (rfm["Frequency"] > 0) &
        (rfm["Monetary"] > 0)
    ].copy()

    # Keep useful columns
    rfm = rfm[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "LastPurchaseDate"
        ]
    ]

    rfm.reset_index(drop=True, inplace=True)

    print("\nRFM ANALYSIS COMPLETED")
    print("-" * 40)
    print("Total customers:", len(rfm))
    print("\nRFM summary:")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe())

    return rfm


def scale_rfm_features(rfm):
    """
    Log-transform and standardize RFM features.
    """

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    features = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    # Reduce skewness
    X_log = np.log1p(rfm[features])

    # Standardize features
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_log)

    return X_scaled, scaler