# analysis_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import shap

# === 1. Caricamento dataset ===
def load_dataset(path):
    df = pd.read_csv(path)
    return df

# === 2. Preprocessing e encoding ===
def preprocess(df, target_col="execution_accuracy"):
    df_clean = df.copy()
    
    # Encode boolean
    for col in ["has_orderby", "has_limit", "has_where"]:
        df_clean[col] = df_clean[col].astype(bool).astype(int)

    # Encode column_types_used as number of unique types
    df_clean["n_col_types_used"] = df_clean["column_types_used"].fillna("").apply(lambda x: len(set(x.split(", "))) if x else 0)

    # Encode test_category and other object columns
    for col in ["test_category"]:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # Drop columns not usable or redundant
    drop_cols = ["db_path", "table_name", "query", "model", "AI_answer", "SQL_query", 
                 "columns_in_query", "column_types_used"]
    df_clean = df_clean.drop(columns=[col for col in drop_cols if col in df_clean])

    df_clean = df_clean.dropna()
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    return X, y, df_clean

# === 3. Feature importance (Random Forest + Permutation) ===
def compute_feature_importance(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)
    
    return importance_df

# === 4. Correlazioni tra feature e target ===
def plot_correlations(df, target_cols):
    corr = df[target_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlazioni tra metriche")
    plt.show()

# === 5. Clusterizzazione ===
def cluster_analysis(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score

# === 6. PCA per visualizzazione ===
def plot_pca(X, labels=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10")
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    plt.title("PCA del Meta-Dataset")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

# === 7. SHAP (opzionale) ===
def shap_analysis(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)

# === USO ===
if __name__ == "__main__":
    input_csv = "enriched_all_tests.csv"  # Sostituisci con il tuo path
    target = "tuple_constraint"  # o "tuple_constraint", ecc.

    df = load_dataset(input_csv)
    X, y, full_df = preprocess(df, target)

    print("\n📊 Feature Importance:")
    importance = compute_feature_importance(X, y)
    print(importance)

    print("\n📈 Correlazioni tra metriche:")
    metric_cols = ["cell_precision", "cell_recall", "execution_accuracy", "tuple_cardinality", "tuple_constraint", "tuple_order"]
    plot_correlations(full_df, metric_cols)

    print("\n🔍 Clusterizzazione e PCA:")
    labels, score = cluster_analysis(X, n_clusters=3)
    print(f"Silhouette score: {score}")
    plot_pca(X, labels)

    # opzionale
    shap_analysis(X, y)