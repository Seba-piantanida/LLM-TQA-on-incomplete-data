# analyze_recommendation_metrics.py - Recommendation System Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
import shap
import warnings
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plot configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_plotting():
    """Global configuration for plots"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def validate_data(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Validates and filters available columns"""
    available_features = [col for col in feature_cols if col in df.columns]
    available_targets = [col for col in target_cols if col in df.columns]
    
    logger.info(f"Available features: {len(available_features)}/{len(feature_cols)}")
    logger.info(f"Available targets: {len(available_targets)}/{len(target_cols)}")
    
    missing_features = [col for col in feature_cols if col not in df.columns]
    missing_targets = [col for col in target_cols if col not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    if missing_targets:
        logger.warning(f"Missing targets: {missing_targets}")
    
    return available_features, available_targets

def create_correlation_analysis(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], output_dir: str = "."):
    """Advanced correlation analysis"""
    logger.info("🔍 Creating correlation analysis...")
    
    # Complete correlation matrix
    corr_data = df[feature_cols + target_cols].corr()
    
    # Main heatmap - feature-target correlations
    plt.figure(figsize=(15, 10))
    target_corr = corr_data.loc[feature_cols, target_cols]
    
    mask = np.abs(target_corr) < 0.1  # Mask for weak correlations
    sns.heatmap(target_corr, annot=True, cmap="RdBu_r", center=0, 
                fmt='.2f', mask=mask, cbar_kws={'label': 'Correlation'})
    plt.title("Feature-Target Correlations (|r| >= 0.1)", fontsize=16, pad=20)
    plt.xlabel("Target Metrics", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Strongest correlations for each target
    n_targets = len(target_cols)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    plt.figure(figsize=(6*n_cols, 5*n_rows))
    for i, target in enumerate(target_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        target_corr_series = corr_data[target].abs().sort_values(ascending=False)
        top_features = target_corr_series[target_corr_series.index.isin(feature_cols)].head(10)
        
        colors = ['red' if corr_data[target][feat] < 0 else 'blue' for feat in top_features.index]
        top_features.plot(kind='barh', color=colors, alpha=0.7)
        plt.title(f'Top Correlations: {target}', fontsize=11)
        plt.xlabel('|Correlation|')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_data

def create_distribution_analysis(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], output_dir: str = "."):
    """Analysis of variable distributions"""
    logger.info("📊 Creating distribution analysis...")
    
    # Feature distributions
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
    n_features = len(numeric_features)
    
    if n_features > 0:
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(numeric_features):
            if i >= len(axes):
                break
            
            ax = axes[i]
            data = df[feature].dropna()
            
            if len(data) > 0:
                # Histogram with density curve
                ax.hist(data, bins=30, alpha=0.7, density=True, color='skyblue')
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                
                ax.set_title(f'{feature}\n(Skew: {data.skew():.2f})')
                ax.legend(fontsize=8)
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        
        # Remove empty axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Target distributions
    if len(target_cols) > 0:
        n_cols = min(3, len(target_cols))
        n_rows = (len(target_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
        if len(target_cols) == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if len(target_cols) > 1 else [axes]
        
        for i, target in enumerate(target_cols):
            ax = axes[i]
            data = df[target].dropna()
            
            if len(data) > 0:
                ax.hist(data, bins=30, alpha=0.7, density=True, color='coral')
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.3f}')
                ax.set_title(f'{target}\n(Skew: {data.skew():.2f})')
                ax.legend(fontsize=8)
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        
        # Remove empty axes
        for j in range(len(target_cols), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

def evaluate_model_performance(X: pd.DataFrame, y: pd.Series, models: Dict, target_name: str) -> Dict:
    """Evaluate model performance"""
    results = {}
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for model_name, model in models.items():
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Fit and prediction
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'r2_test': r2,
                'mse_test': mse,
                'mae_test': mae,
                'model': model
            }
            
        except Exception as e:
            logger.error(f"Error with model {model_name} for {target_name}: {e}")
            continue
    
    return results

def create_feature_importance_analysis(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], output_dir: str = "."):
    """Advanced feature importance analysis"""
    logger.info("🎯 Analyzing feature importance...")
    
    # Multiple models for comparison
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Ridge': Ridge(alpha=1.0)
    }
    
    importance_results = {}
    model_performance = {}
    
    for target in target_cols:
        logger.info(f"📈 Analyzing: {target}")
        
        # Data preparation
        X = df[feature_cols].fillna(0)
        y = df[target].fillna(df[target].median())
        
        if y.nunique() <= 1:
            logger.warning(f"Target {target} has zero variance, skipping...")
            continue
        
        # Model evaluation
        model_results = evaluate_model_performance(X, y, models, target)
        model_performance[target] = model_results
        
        # Feature importance for Random Forest
        if 'Random Forest' in model_results:
            rf_model = model_results['Random Forest']['model']
            importances = rf_model.feature_importances_
            feature_importance = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
            importance_results[target] = feature_importance
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            
            bars = plt.barh(range(len(top_features)), top_features.values, color=colors)
            plt.yticks(range(len(top_features)), top_features.index)
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importance: {target}\n'
                     f'(R² = {model_results["Random Forest"]["r2_test"]:.3f})')
            plt.gca().invert_yaxis()
            
            # Add values on bars
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{top_features.values[i]:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance_{target}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Model performance comparison
    create_model_comparison_plot(model_performance, output_dir)
    
    return importance_results, model_performance

def create_model_comparison_plot(model_performance: Dict, output_dir: str = "."):
    """Model performance comparison"""
    logger.info("📊 Creating model comparison...")
    
    # Prepare data for plot
    comparison_data = []
    for target, models in model_performance.items():
        for model_name, metrics in models.items():
            comparison_data.append({
                'Target': target,
                'Model': model_name,
                'R² Test': metrics['r2_test'],
                'CV Mean': metrics['cv_mean'],
                'MAE': metrics['mae_test']
            })
    
    if not comparison_data:
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # R² comparison plot
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=comparison_df, x='Target', y='R² Test', hue='Model')
    plt.title('R² Test Comparison Between Models')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=comparison_df, x='Target', y='CV Mean', hue='Model')
    plt.title('Cross-Validation R² Comparison Between Models')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_shap_analysis(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], output_dir: str = "."):
    """SHAP analysis for feature interpretability"""
    logger.info("🔍 Running SHAP analysis...")
    
    for target in target_cols:
        try:
            logger.info(f"SHAP analysis for: {target}")
            
            # Data preparation
            X = df[feature_cols].fillna(0)
            y = df[target].fillna(df[target].median())
            
            if y.nunique() <= 1:
                logger.warning(f"Target {target} has zero variance, skipping...")
                continue
            
            # Remove constant features
            constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_cols:
                logger.info(f"Removing constant columns: {constant_cols}")
                X = X.drop(columns=constant_cols)
            
            if X.shape[1] == 0:
                logger.error(f"No variable features for {target}")
                continue
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            # SHAP analysis with sample
            n_samples = min(1000, len(X))
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_indices]
            
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # SHAP bar plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar", max_display=15)
            plt.title(f"SHAP Feature Importance: {target}", fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_bar_{target}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
            plt.title(f"SHAP Summary Plot: {target}", fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap_summary_{target}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"SHAP error for {target}: {e}")
            continue

def create_target_relationships_analysis(df: pd.DataFrame, target_cols: List[str], output_dir: str = "."):
    """Analysis of relationships between target metrics"""
    logger.info("🔗 Analyzing relationships between targets...")
    
    if len(target_cols) < 2:
        return
    
    # Correlation matrix between targets
    target_corr = df[target_cols].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(target_corr, dtype=bool))
    sns.heatmap(target_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Correlations Between Target Metrics', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/target_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Scatter plots between targets
    n_plots = min(6, len(target_cols) * (len(target_cols) - 1) // 2)
    if n_plots > 0:
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        combinations = [(target_cols[i], target_cols[j]) 
                       for i in range(len(target_cols)) 
                       for j in range(i+1, len(target_cols))]
        
        for idx, (target1, target2) in enumerate(combinations[:n_plots]):
            ax = axes[idx]
            
            plot_data = df[[target1, target2]].dropna()
            
            if len(plot_data) > 0:
                ax.scatter(plot_data[target1], plot_data[target2], alpha=0.6, s=30)
                
                # Regression line
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        plot_data[target1], plot_data[target2])
                    line = slope * plot_data[target1] + intercept
                    ax.plot(plot_data[target1], line, 'r--', alpha=0.8)
                    ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except:
                    pass
            
            ax.set_xlabel(target1)
            ax.set_ylabel(target2)
            ax.set_title(f'{target1} vs {target2}')
            ax.grid(True, alpha=0.3)
        
        # Remove empty axes
        for idx in range(len(combinations[:n_plots]), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_relationships.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_report(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], 
                         importance_results: Dict, model_performance: Dict, 
                         output_dir: str = ".", input_file: str = "data.csv"):
    """Creates a summary report of the analysis"""
    logger.info("📋 Creating summary report...")
    
    report = []
    report.append("# RECOMMENDATION METRICS ANALYSIS REPORT\n")
    report.append(f"Input file: {input_file}\n")
    report.append(f"Dataset: {len(df)} rows, {len(feature_cols)} features, {len(target_cols)} targets\n")
    report.append("=" * 70 + "\n\n")
    
    # General statistics
    report.append("## GENERAL STATISTICS\n")
    report.append(f"Total rows: {len(df)}\n")
    report.append(f"Available features: {len(feature_cols)}\n")
    report.append(f"Available targets: {len(target_cols)}\n\n")
    
    # Target statistics
    report.append("## TARGET METRICS STATISTICS\n")
    for target in target_cols:
        if target in df.columns:
            data = df[target].dropna()
            if len(data) > 0:
                report.append(f"\n{target}:\n")
                report.append(f"  Mean: {data.mean():.4f}\n")
                report.append(f"  Median: {data.median():.4f}\n")
                report.append(f"  Std: {data.std():.4f}\n")
                report.append(f"  Min: {data.min():.4f}\n")
                report.append(f"  Max: {data.max():.4f}\n")
    
    # Missing values
    missing_stats = df[feature_cols + target_cols].isnull().sum()
    if missing_stats.sum() > 0:
        report.append(f"\n## MISSING VALUES\n")
        report.append(f"Total missing values: {missing_stats.sum()}\n")
        top_missing = missing_stats[missing_stats > 0].sort_values(ascending=False).head(10)
        if len(top_missing) > 0:
            report.append("Top columns with missing values:\n")
            for col, count in top_missing.items():
                report.append(f"  - {col}: {count} ({count/len(df)*100:.1f}%)\n")
    
    # Model performance
    if model_performance:
        report.append("\n## MODEL PERFORMANCE\n")
        for target, models in model_performance.items():
            report.append(f"\n### {target}:\n")
            for model_name, metrics in models.items():
                report.append(f"  {model_name}:\n")
                report.append(f"    - R² Test: {metrics['r2_test']:.4f}\n")
                report.append(f"    - R² CV: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}\n")
                report.append(f"    - MAE: {metrics['mae_test']:.4f}\n")
                report.append(f"    - MSE: {metrics['mse_test']:.4f}\n")
    
    # Top features per target
    if importance_results:
        report.append("\n## TOP FEATURES PER TARGET (Random Forest)\n")
        for target, importances in importance_results.items():
            report.append(f"\n### {target}:\n")
            top_10 = importances.head(10)
            for i, (feature, importance) in enumerate(top_10.items(), 1):
                report.append(f"  {i}. {feature}: {importance:.4f}\n")
    
    # Feature statistics
    report.append("\n## FEATURE STATISTICS\n")
    for feature in feature_cols:
        if feature in df.columns:
            data = df[feature].dropna()
            if len(data) > 0 and pd.api.types.is_numeric_dtype(data):
                report.append(f"\n{feature}:\n")
                report.append(f"  Mean: {data.mean():.4f}, Std: {data.std():.4f}\n")
                report.append(f"  Range: [{data.min():.4f}, {data.max():.4f}]\n")
    
    # Save report
    with open(f"{output_dir}/analysis_report.txt", "w", encoding="utf-8") as f:
        f.writelines(report)
    
    logger.info(f"📄 Report saved to: {output_dir}/analysis_report.txt")

def main(input_csv: str, output_dir: str = "."):
    """Main function for recommendation metrics analysis"""
    setup_plotting()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        df = pd.read_csv(input_csv)
        logger.info(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Column definitions - TARGET METRICS
        target_cols = [
            "precision",
            "recall", 
            "accuracy",
            "ndcg_10",
            "precision_5"
        ]
        
        # Column definitions - ENRICHED FEATURES
        feature_cols = [
            "dataset_num_rows",
            "dataset_num_columns_original",
            "dataset_num_columns_after_removal",
            "num_removed_columns",
            "query_length",
            "query_word_count",
            "query_avg_word_length",
            "query_unique_words",
            "query_lexical_diversity",
            "token_prompt_count"
        ]
        
        # Validate column availability
        feature_cols, target_cols = validate_data(df, feature_cols, target_cols)
        
        if not feature_cols or not target_cols:
            logger.error("No features or targets available for analysis")
            return
        
        logger.info(f"\n📋 Using features: {feature_cols}")
        logger.info(f"📋 Using targets: {target_cols}\n")
        
        # Main analyses
        logger.info("🚀 Starting comprehensive analysis...")
        
        # 1. Correlation analysis
        corr_matrix = create_correlation_analysis(df, feature_cols, target_cols, output_dir)
        
        # 2. Distribution analysis
        create_distribution_analysis(df, feature_cols, target_cols, output_dir)
        
        # 3. Feature importance and models
        importance_results, model_performance = create_feature_importance_analysis(
            df, feature_cols, target_cols, output_dir
        )
        
        # 4. SHAP analysis
        create_shap_analysis(df, feature_cols, target_cols, output_dir)
        
        # 5. Target relationships
        create_target_relationships_analysis(df, target_cols, output_dir)
        
        # 6. Final report
        create_summary_report(df, feature_cols, target_cols, importance_results, 
                            model_performance, output_dir, input_csv)
        
        logger.info("\n" + "="*70)
        logger.info("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"📁 All files saved to: {output_dir}")
        logger.info(f"📊 Generated files:")
        logger.info(f"   - correlation_heatmap.png")
        logger.info(f"   - top_correlations.png")
        logger.info(f"   - feature_distributions.png")
        logger.info(f"   - target_distributions.png")
        logger.info(f"   - feature_importance_*.png (per target)")
        logger.info(f"   - model_comparison.png")
        logger.info(f"   - shap_bar_*.png (per target)")
        logger.info(f"   - shap_summary_*.png (per target)")
        logger.info(f"   - target_correlations.png")
        logger.info(f"   - target_relationships.png")
        logger.info(f"   - analysis_report.txt")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "enrich_res/enriched_final_results.csv"
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "aresults_analisis/all"
    
    main(csv_path, output_directory)