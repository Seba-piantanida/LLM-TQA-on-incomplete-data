# analyze_metrics.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import shap

def main(input_csv):
    df = pd.read_csv(input_csv)

    target_cols = [
        "cell_precision", "cell_recall", "execution_accuracy",
        "tuple_cardinality", "tuple_constraint", "tuple_order"
    ]

    feature_cols = [
        "num_columns", "num_rows", "num_numeric_columns", "num_categorical_columns",
        "numeric_categorical_ratio", "avg_unique_vals", "num_selected_columns",
        "num_order_columns", "num_numeric_in_query", "num_categorical_in_query",
        "query_complexity", "text_length", "num_tokens", "num_nouns", "num_verbs",
        "num_clauses", "keyword_score", "semantic_embedding_complexity"
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]

    # Heatmap
    corr = df[feature_cols + target_cols].corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr[target_cols].T, annot=True, cmap="coolwarm")
    plt.title("Heatmap Correlazioni tra Feature e Metriche")
    plt.tight_layout()
    plt.savefig("heatmap_metrics.png")
    plt.show()

    # Feature Importance e SHAP
    for target in target_cols:
        print(f"\n🔍 Analisi per: {target}")
        X = df[feature_cols].fillna(0)
        y = df[target].fillna(0)

        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X, y)

        # Feature Importance
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

        plt.figure()
        feat_imp.plot(kind="bar", title=f"Feature Importance per {target}")
        plt.tight_layout()
        plt.savefig(f"feature_importance_{target}.png", dpi =300)
        

        # SHAP
        try:
            
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            shap.summary_plot(shap_values, X, show=False, plot_type="bar")
            plt.title(f"SHAP Summary Bar: {target}")
            plt.tight_layout()
            plt.savefig(f"shap_summary_{target}.png", dpi =300)
            plt.close()

            shap.summary_plot(shap_values, X, show=False)
            plt.title(f"SHAP Summary: {target}")
            plt.tight_layout()
            plt.savefig(f"shap_detail_{target}.png", dpi =300)
            plt.close()
        except Exception as e:
            print(f"⚠️ Errore SHAP per {target}: {e}")

if __name__ == "__main__":
    import sys
    csv_path = "v2/all_tests_enriched.csv"
    main(csv_path)