# ============================================================
# Week 3 Project — Predictive Modelling & Statistical Analysis
# Dataset columns expected:
# ['Customer_ID','Customer_Name','Region','Total_Spend',
#  'Purchase_Frequency','Marketing_Spend','Seasonality_Index','Churned']
# ============================================================

# ------------ 1) Imports ------------
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             r2_score, mean_squared_error)
from scipy.stats import ttest_ind, f_oneway
from sklearn.cluster import KMeans
np.random.seed(42)

# ------------ 2) Load data ------------
# >>>> UPDATE THIS PATH IF NEEDED <<<<
DATA_PATH = Path(r"C:\Users\sssss\OneDrive\Documents\internship projects bootcamp\raw_sales_data1.xlsx")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find: {DATA_PATH}")

df = pd.read_excel(DATA_PATH)

# ------------ 3) Quick sanity checks & basic cleaning ------------
# Standardize column names (strip spaces)
df.columns = [c.strip() for c in df.columns]

# Ensure numeric columns are numeric
num_cols = ['Total_Spend', 'Purchase_Frequency', 'Marketing_Spend', 'Seasonality_Index']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Handle missing values (median for numeric, mode for categoricals)
for c in num_cols:
    df[c].fillna(df[c].median(), inplace=True)

cat_cols = ['Region', 'Churned']
for c in cat_cols:
    if df[c].isna().any():
        df[c].fillna(df[c].mode()[0], inplace=True)

# Optional: light outlier cap using IQR
for c in num_cols:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    df[c] = df[c].clip(lower=low, upper=high)

# ------------ 4) Encode categoricals ------------
enc_region = LabelEncoder()
df['Region_enc'] = enc_region.fit_transform(df['Region'])

df['Churned_bin'] = df['Churned'].map({'Yes': 1, 'No': 0})

# ------------ 5) EDA: correlation heatmap ------------
plt.figure(figsize=(7,5))
corr = df[['Total_Spend','Purchase_Frequency','Marketing_Spend','Seasonality_Index','Region_enc','Churned_bin']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# PREDICTIVE MODELLING
# ============================================================

# ------------ 6) Classification: Predict Churned (Yes/No) ------------
X_cls = df[['Total_Spend','Purchase_Frequency','Marketing_Spend','Seasonality_Index','Region_enc']]
y_cls = df['Churned_bin']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls)

# Logistic Regression
logit = LogisticRegression(max_iter=1000)
logit.fit(Xc_train, yc_train)
yc_pred_logit = logit.predict(Xc_test)

print("\n=== Logistic Regression (Churn) ===")
print("Accuracy :", accuracy_score(yc_test, yc_pred_logit))
print("Precision:", precision_score(yc_test, yc_pred_logit))
print("Recall   :", recall_score(yc_test, yc_pred_logit))
print("F1       :", f1_score(yc_test, yc_pred_logit))
print("\nConfusion Matrix:\n", confusion_matrix(yc_test, yc_pred_logit))
print("\nClassification Report:\n", classification_report(yc_test, yc_pred_logit))

# Random Forest (often more accurate / feature importance)
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(Xc_train, yc_train)
yc_pred_rf = rf.predict(Xc_test)

print("\n=== Random Forest (Churn) ===")
print("Accuracy :", accuracy_score(yc_test, yc_pred_rf))
print("Precision:", precision_score(yc_test, yc_pred_rf))
print("Recall   :", recall_score(yc_test, yc_pred_rf))
print("F1       :", f1_score(yc_test, yc_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(yc_test, yc_pred_rf))

# Feature importance
fi = pd.Series(rf.feature_importances_, index=X_cls.columns).sort_values(ascending=False)
plt.figure(figsize=(6,4))
fi.plot(kind='bar')
plt.title("Random Forest Feature Importance (Churn)")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# ------------ 7) Regression: Predict Total_Spend ------------
X_reg = df[['Purchase_Frequency','Marketing_Spend','Seasonality_Index','Region_enc']]
y_reg = df['Total_Spend']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

lin = LinearRegression()
lin.fit(Xr_train, yr_train)
yr_pred = lin.predict(Xr_test)

rmse = mean_squared_error(yr_test, yr_pred, squared=False)
print("\n=== Linear Regression (Total_Spend) ===")
print("R^2  :", r2_score(yr_test, yr_pred))
print("RMSE :", rmse)

plt.figure(figsize=(5,5))
plt.scatter(yr_test, yr_pred)
plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--')
plt.xlabel("Actual Total_Spend")
plt.ylabel("Predicted Total_Spend")
plt.title("Actual vs Predicted (Linear Regression)")
plt.tight_layout()
plt.show()

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

# ------------ 8) T-test: Spend for Churned vs Not ------------
grp_yes = df.loc[df['Churned_bin']==1, 'Total_Spend']
grp_no  = df.loc[df['Churned_bin']==0, 'Total_Spend']

if len(grp_yes) > 1 and len(grp_no) > 1:
    tstat, pval = ttest_ind(grp_yes, grp_no, equal_var=False)
    print("\nT-test (Total_Spend: Churned vs Not):")
    print(f"t-stat={tstat:.3f}, p={pval:.4f}")
else:
    print("\nT-test skipped (not enough samples in a group).")

# ------------ 9) ANOVA: Spend across Regions ------------
# Need 2+ samples per region ideally; run anyway on available groups
groups = [df.loc[df['Region']==r, 'Total_Spend'].values for r in df['Region'].unique()]
if all(len(g)>0 for g in groups) and len(groups) >= 2:
    fstat, pval = f_oneway(*groups)
    print("\nANOVA (Total_Spend across Regions):")
    print(f"F={fstat:.3f}, p={pval:.4f}")
else:
    print("\nANOVA skipped (insufficient groups/samples).")

# ============================================================
# CUSTOMER SEGMENTATION (K-Means)
# ============================================================

# ------------ 10) K-Means on spend & behavior ------------
features_cluster = df[['Total_Spend','Purchase_Frequency','Marketing_Spend']]
scaler = StandardScaler()
X_clu = scaler.fit_transform(features_cluster)

kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42)
df['Customer_Segment'] = kmeans.fit_predict(X_clu)

# Visualize clusters (2D using first two features)
plt.figure(figsize=(6,5))
sns.scatterplot(x=df['Total_Spend'], y=df['Purchase_Frequency'],
                hue=df['Customer_Segment'], palette='Set2', s=80)
plt.title("Customer Segments (K-Means)")
plt.tight_layout()
plt.show()

# Segment summary (handy for your report)
seg_summary = df.groupby('Customer_Segment')[['Total_Spend','Purchase_Frequency','Marketing_Spend']].mean().round(1)
print("\nSegment Summary (means):\n", seg_summary)

# ============================================================
# SAVE ENRICHED DATA (optional)
# ============================================================
out_path = DATA_PATH.with_name("raw_sales_data1_enriched.xlsx")
df.to_excel(out_path, index=False)
print(f"\nSaved enriched dataset with encodings & segments to:\n{out_path}")

