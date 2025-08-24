#!/usr/bin/env python3
"""
Re-clean script (fixes Defaulted wiped to 0 issue)

- Auto-detects binary indicator columns (values in {0,1}) and EXCLUDES them from outlier capping.
- Median impute for numeric (non-binary); mode impute for categoricals.
- IQR capping only on continuous numeric columns.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ========= CONFIG (edit these if needed) =========
DATA_PATH = Path(r"C:\Users\sssss\OneDrive\Documents\internship projects bootcamp\raw_dataset_week4.csv")
OUTPUT_PATH = DATA_PATH.with_name("cleaned_dataset_week4_v2.csv")

ID_COLUMNS = ["Customer_ID"]                          # never treat as outliers
PROTECT_ALWAYS = ["Defaulted", "Customer_Churn"]      # binary targets to protect

OUTLIER_METHOD = "cap"  # "cap" or "remove"
IQR_K = 1.5
# =================================================

def summarize_missing(df):
    miss = df.isna().sum()
    pct = miss / len(df) * 100 if len(df) else 0
    return pd.DataFrame({"missing_count": miss, "missing_pct": pct}).sort_values("missing_count", ascending=False)

def iqr_bounds(s: pd.Series, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k*iqr, q3 + k*iqr

def is_binary(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = pd.Series(series.dropna().unique())
    # allow {0,1} only
    return vals.size > 0 and set(vals.astype(float)).issubset({0.0, 1.0})

def treat_outliers(df: pd.DataFrame, cols, method="cap", k=1.5):
    df2 = df.copy()
    mask_remove = pd.Series(False, index=df2.index)
    for c in cols:
        s = df2[c]
        # skip non-numeric or all-NaN columns
        if (not pd.api.types.is_numeric_dtype(s)) or s.dropna().empty:
            continue
        lo, hi = iqr_bounds(s, k)
        if method == "cap":
            df2[c] = s.clip(lower=lo, upper=hi)
        elif method == "remove":
            mask_remove |= (s < lo) | (s > hi)
    if method == "remove":
        df2 = df2.loc[~mask_remove].reset_index(drop=True)
    return df2

def main():
    print("Loading:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Initial shape:", df.shape)

    # Identify column types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Detect binary columns
    binary_cols = [c for c in num_cols if is_binary(df[c])]
    # Ensure protected ones included
    for c in PROTECT_ALWAYS:
        if c in df.columns and c not in binary_cols:
            binary_cols.append(c)

    print("Binary (protected) columns:", binary_cols)

    # --- Missing values BEFORE ---
    print("\nMissing BEFORE:\n", summarize_missing(df))

    # Impute numeric (non-binary) with median
    for c in [c for c in num_cols if c not in binary_cols]:
        df[c] = df[c].fillna(df[c].median())

    # Impute binary with mode (if any NaN)
    for c in binary_cols:
        if df[c].isna().any():
            mode_vals = df[c].mode(dropna=True)
            if len(mode_vals):
                df[c] = df[c].fillna(mode_vals.iloc[0])

    # Impute categoricals with mode
    for c in cat_cols:
        if df[c].isna().any():
            mode_vals = df[c].mode(dropna=True)
            if len(mode_vals):
                df[c] = df[c].fillna(mode_vals.iloc[0])

    # Choose continuous columns for outlier treatment:
    continuous_cols = [
        c for c in num_cols
        if c not in set(binary_cols + ID_COLUMNS)
    ]
    print("Outlier-treated columns:", continuous_cols)

    df = treat_outliers(df, continuous_cols, method=OUTLIER_METHOD, k=IQR_K)

    # --- Sanity check to confirm we didn't break binary columns ---
    for c in binary_cols:
        uniques = sorted(df[c].dropna().unique().tolist())
        print(f"{c} unique values after cleaning:", uniques)

    # --- Missing values AFTER ---
    print("\nMissing AFTER:\n", summarize_missing(df))

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved:", OUTPUT_PATH)
    print("Final shape:", df.shape)

if __name__ == "__main__":
    main()
