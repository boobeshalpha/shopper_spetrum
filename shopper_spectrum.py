import os
import numpy as np # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import joblib # type: ignore

# =========================================================
# Setup
# =========================================================
os.makedirs("outputs", exist_ok=True)

# =========================================================
# Load dataset
# =========================================================
# Keep path as provided
df = pd.read_csv(r"C:\Users\keert\OneDrive\Desktop\GUVI_SHOP\online_retail.csv")

# =========================================================
# Data Access / Exploration
# =========================================================
print("\n📌 First 5 rows:")
print(df.head())

print("\n📌 Dataset info:")
print(df.info())

print("\n📌 Missing values per column:")
print(df.isnull().sum())

print("\n📌 Total duplicates:")
print(df.duplicated().sum())

# =========================================================
# Data Cleaning
# =========================================================
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df = df.drop_duplicates()

# ✅ Convert InvoiceDate
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

# Remove rows with invalid dates if any
df = df.dropna(subset=["InvoiceDate"])

# Add time-based features
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day
df["Hour"] = df["InvoiceDate"].dt.hour
df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()

# Add total price
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("\n✅ Data cleaned and datetime converted. Shape:", df.shape)
print(df.info())

# =========================================================
# EDA
# =========================================================

# 1. Top 10 countries by transactions
country_txn = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
country_txn.plot(kind="bar", color="skyblue")
plt.title("Top 10 Countries by Transactions")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/top_country_transactions.png")
plt.close()
print("✅ Saved: outputs/top_country_transactions.png")

# 2. Top 20 products by sales quantity
top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(12,5))
top_products.plot(kind="bar", color="orange")
plt.title("Top 20 Products by Units Sold")
plt.ylabel("Units")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig("outputs/top_products_unit.png")
plt.close()
print("✅ Saved: outputs/top_products_unit.png")

# 3. Revenue trend over time
daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
plt.figure(figsize=(12,5))
daily_revenue.plot(color="green")
plt.title("Daily Revenue Over Time")
plt.ylabel("Revenue")
plt.xlabel("Date")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig("outputs/daily_revenue_over_time.png")
plt.close()
print("✅ Saved: outputs/daily_revenue_over_time.png")

# 4. Monetary distribution per transaction and per customer
tnx_monetary = df.groupby("InvoiceNo")["TotalPrice"].sum()
cus_monetary = df.groupby("CustomerID")["TotalPrice"].sum()

plt.figure(figsize=(10,5))
tnx_monetary.plot(kind="hist", bins=50)
plt.title("Monetary Distribution per Transaction")
plt.xlabel("Monetary (Transaction Total)")
plt.tight_layout()
plt.savefig("outputs/txn_monetary_hist.png")
plt.close()
print("✅ Saved: outputs/txn_monetary_hist.png")

plt.figure(figsize=(10,5))
cus_monetary.plot(kind="hist", bins=50)
plt.title("Monetary Distribution per Customer")
plt.xlabel("Monetary (Customer Lifetime Value)")
plt.tight_layout()
plt.savefig("outputs/cus_monetary_hist.png")
plt.close()
print("✅ Saved: outputs/cus_monetary_hist.png")

# =========================================================
# RFM Analysis
# =========================================================
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda X: (snapshot_date - X.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
).reset_index()

# Distributions
for col in ["Recency", "Frequency", "Monetary"]:
    plt.figure(figsize=(8,4))
    rfm[col].plot(kind="hist", bins=50)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"outputs/rfm_{col.lower()}_hist.png")
    plt.close()

# =========================================================
# Elbow + Silhouette + Davies-Bouldin for Clustering
# =========================================================
X = rfm[["Recency", "Frequency", "Monetary"]].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
sil_scores = []
db_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = kmeans_tmp.fit_predict(X_scaled)
    inertia.append(kmeans_tmp.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_tmp))
    db_scores.append(davies_bouldin_score(X_scaled, labels_tmp))

# Plots
plt.figure(figsize=(8,5))
plt.plot(list(k_range), inertia, marker="o")
plt.title("Elbow Curve for Customer Clustering (RFM)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster SSE)")
plt.tight_layout()
plt.savefig("outputs/rfm_elbow_curve.png")
plt.close()
print("✅ Saved: outputs/rfm_elbow_curve.png")

plt.figure(figsize=(8,5))
plt.plot(list(k_range), sil_scores, marker="o", color="green")
plt.title("Silhouette Scores vs k (higher is better)")
plt.xlabel("k")
plt.ylabel("Average Silhouette")
plt.tight_layout()
plt.savefig("outputs/rfm_silhouette_curve.png")
plt.close()
print("✅ Saved: outputs/rfm_silhouette_curve.png")

plt.figure(figsize=(8,5))
plt.plot(list(k_range), db_scores, marker="o", color="red")
plt.title("Davies–Bouldin vs k (lower is better)")
plt.xlabel("k")
plt.ylabel("DB Index")
plt.tight_layout()
plt.savefig("outputs/rfm_db_curve.png")
plt.close()
print("✅ Saved: outputs/rfm_db_curve.png")

# Choose k by best silhouette (break ties by lower DB)
best_k = None
best_sil = -np.inf
best_db = np.inf
for k, sil, db in zip(k_range, sil_scores, db_scores):
    if (sil > best_sil) or (np.isclose(sil, best_sil) and db < best_db):
        best_k, best_sil, best_db = k, sil, db

print(f"Selected k={best_k} by silhouette (score={best_sil:.4f}) and DB={best_db:.4f}")

# =========================================================
# Customer Clustering & Profiles (final fit)
# =========================================================
final_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=int(best_k), random_state=42, n_init=10)),
])
final_pipe.fit(X)
rfm["Cluster"] = final_pipe.named_steps["kmeans"].labels_

# Persist pipeline for Streamlit usage
joblib.dump({"pipeline": final_pipe, "features": ["Recency","Frequency","Monetary"]},
            "outputs/rfm_kmeans_pipeline.joblib")
print("✅ Saved: outputs/rfm_kmeans_pipeline.joblib")

# Save clustered RFM
rfm.to_csv("outputs/rfm_clusters.csv", index=False)

# Profiles
cluster_profile = (
    rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
       .agg(["mean", "median", "count"])
       .round(2)
)
cluster_profile.to_csv("outputs/cluster_profiles.csv")

means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
means.plot(kind="bar", figsize=(10,5))
plt.title("Customer Cluster Profiles (Mean R/F/M)")
plt.ylabel("Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/cluster_profiles_means.png")
plt.close()
print("✅ Saved: outputs/cluster_profiles_means.png + cluster_profiles.csv")

# =========================================================
# Heuristic Segment Labels
# =========================================================
# Rank clusters: lower Recency is better; higher F & M are better
r_rank = means["Recency"].rank(ascending=True, method="dense")
f_rank = means["Frequency"].rank(ascending=False, method="dense")
m_rank = means["Monetary"].rank(ascending=False, method="dense")

def label_cluster(c):
    r, f, m = r_rank.loc[c], f_rank.loc[c], m_rank.loc[c]
    if r <= 1 and f <= 1 and m <= 1:
        return "High-Value"
    if r >= r_rank.max() and f >= f_rank.max() and m >= m_rank.max():
        return "At-Risk"
    if f >= f_rank.max() and m >= m_rank.max():
        return "Occasional"
    return "Regular"

labels_map = pd.Series({c: label_cluster(c) for c in means.index}, name="Segment")
rfm = rfm.merge(labels_map, left_on="Cluster", right_index=True, how="left")
rfm.to_csv("outputs/rfm_clusters.csv", index=False)  # overwrite with Segment
print("✅ Saved: outputs/rfm_clusters.csv (with Segment)")

# =========================================================
# Product Recommendation: Item-based Collaborative Filtering
# =========================================================
# Use StockCode as key to avoid duplicate names; map to Description for display
basket_qty = df.pivot_table(index="CustomerID", columns="StockCode",
                            values="Quantity", aggfunc="sum", fill_value=0)
# Binarize for co-purchase signals
basket_bin = (basket_qty > 0).astype(int)

# Compute item-item cosine similarity
item_sim = cosine_similarity(basket_bin.T)
item_sim_df = pd.DataFrame(item_sim, index=basket_bin.columns, columns=basket_bin.columns)

# Build display maps
code_to_name = (df.groupby(["StockCode","Description"]).size()
                  .reset_index(name="cnt")
                  .sort_values(["StockCode","cnt"], ascending=[True,False])
                  .drop_duplicates("StockCode")
                  .set_index("StockCode")["Description"].to_dict())

name_to_codes = (df.groupby(["Description","StockCode"]).size()
                   .reset_index(name="cnt")
                   .sort_values(["Description","cnt"], ascending=[True,False])
                   .groupby("Description")["StockCode"].apply(list).to_dict())

def recommend_by_name(desc, topn=5):
    # Choose the most common code for the given description
    if desc not in name_to_codes:
        return []
    code = name_to_codes[desc][0]
    if code not in item_sim_df.index:
        return []
    sims = item_sim_df.loc[code].sort_values(ascending=False)
    sims = sims[sims.index != code].head(topn)
    return [(c, code_to_name.get(c, c), float(sims[c])) for c in sims.index]

# Persist recommender artifacts
item_sim_df.to_pickle("outputs/item_similarity.pkl")
pd.Series(code_to_name).to_json("outputs/code_to_name.json")
pd.Series(name_to_codes).to_json("outputs/name_to_codes.json")
print("✅ Saved: outputs/item_similarity.pkl, code_to_name.json, name_to_codes.json")

# Optional: Product Recommendation Heatmap (top 50 by frequency)
basket_desc = df.pivot_table(index="CustomerID", columns="Description", values="Quantity", aggfunc="sum", fill_value=0)
top_items = df["Description"].value_counts().head(50).index
product_corr = basket_desc[top_items].corr()

plt.figure(figsize=(12,10))
sns.heatmap(product_corr, cmap="coolwarm", center=0)
plt.title("Product Recommendation Heatmap (Top 50 Products)")
plt.tight_layout()
plt.savefig("outputs/product_recommendation_heatmap.png")
plt.close()
print("✅ Saved: outputs/product_recommendation_heatmap.png")

print("\n🎯 All analysis completed. Artifacts in 'outputs/' folder:")
print("- rfm_elbow_curve.png, rfm_silhouette_curve.png, rfm_db_curve.png")
print("- rfm_clusters.csv (with Segment), cluster_profiles.csv, cluster_profiles_means.png")
print("- rfm_kmeans_pipeline.joblib (for Streamlit)")
print("- item_similarity.pkl, code_to_name.json, name_to_codes.json")
