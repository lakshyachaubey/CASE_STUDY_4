import pandas as pd
from pathlib import Path


def load_and_clean_data(data_path):
    """
    Load and clean the Online Retail II dataset.
    """

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Place online_retail_II.csv inside the data folder."
        )

    # Load CSV
    try:
        df = pd.read_csv(
            data_path,
            encoding="utf-8-sig",
            low_memory=False
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            data_path,
            encoding="cp1252",
            low_memory=False
        )

    original_rows = len(df)

    # Standardize column names
    aliases = {
        "invoice": "InvoiceNo",
        "invoiceno": "InvoiceNo",
        "stockcode": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoicedate": "InvoiceDate",
        "price": "UnitPrice",
        "unitprice": "UnitPrice",
        "customerid": "CustomerID",
        "country": "Country"
    }

    rename_map = {}

    for column in df.columns:
        normalized = "".join(
            ch for ch in str(column).lower()
            if ch.isalnum()
        )

        if normalized in aliases:
            rename_map[column] = aliases[normalized]

    df.rename(columns=rename_map, inplace=True)

    required = [
        "InvoiceNo",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {missing}"
        )

    # Convert data types
    df["InvoiceNo"] = (
        df["InvoiceNo"]
        .astype("string")
        .str.strip()
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
        dayfirst=True
    )

    for column in ["Quantity", "UnitPrice", "CustomerID"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove missing essential values
    df.dropna(
        subset=required,
        inplace=True
    )

    # Remove cancelled invoices
    df = df[
        ~df["InvoiceNo"]
        .str.upper()
        .str.startswith("C")
    ]

    # Exclude invalid transactions and returns
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0) &
        (df["CustomerID"] > 0)
    ].copy()

    # Convert customer IDs to clean strings
    df["CustomerID"] = (
        df["CustomerID"]
        .astype("int64")
        .astype(str)
    )

    # Calculate transaction revenue
    df["Revenue"] = (
        df["Quantity"] *
        df["UnitPrice"]
    )

    df.reset_index(drop=True, inplace=True)

    print("\nDATA PREPROCESSING COMPLETED")
    print("-" * 40)
    print("Original rows:", original_rows)
    print("Cleaned rows:", len(df))
    print("Removed rows:", original_rows - len(df))
    print("Unique customers:", df["CustomerID"].nunique())
    print("Unique invoices:", df["InvoiceNo"].nunique())
    print("Total revenue:", round(df["Revenue"].sum(), 2))

    return df