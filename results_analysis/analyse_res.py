# analyze_metrics.py - Enhanced Version

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
    plt.figure(figsize=(15, 8))
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
    #plt.show()
    
    # Strongest correlations for each target
    plt.figure(figsize=(16, 10))
    for i, target in enumerate(target_cols, 1):
        plt.subplot(2, 3, i)
        target_corr_series = corr_data[target].abs().sort_values(ascending=False)
        top_features = target_corr_series[target_corr_series.index.isin(feature_cols)].head(8)
        
        colors = ['red' if corr_data[target][feat] < 0 else 'blue' for feat in top_features.index]
        top_features.plot(kind='barh', color=colors, alpha=0.7)
        plt.title(f'Top Correlations: {target}', fontsize=11)
        plt.xlabel('|Correlation|')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_correlations.png", dpi=300, bbox_inches='tight')
    #plt.show()
    
    return corr_data

def create_distribution_analysis(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], output_dir: str = "."):
    """Analysis of variable distributions"""
    logger.info("📊 Creating distribution analysis...")
    
    # Feature distributions
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
    n_features = len(numeric_features)
    
    if n_features > 0:
        fig, axes = plt.subplots(nrows=(n_features+2)//3, ncols=3, figsize=(15, 5*((n_features+2)//3)))
        axes = axes.flatten() if n_features > 3 else [axes] if n_features == 1 else axes
        
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
                ax.legend()
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        
        # Remove empty axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
        #plt.show()

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
        
        # Feature importance for Random Forest (best for interpretability)
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
            #plt.show()
    
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
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    sns.barplot(data=comparison_df, x='Target', y='R² Test', hue='Model')
    plt.title('R² Test Comparison Between Models')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    sns.barplot(data=comparison_df, x='Target', y='CV Mean', hue='Model')
    plt.title('Cross-Validation R² Comparison Between Models')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    #plt.show()

def create_shap_analysis(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], output_dir: str = "."):
    """Advanced SHAP analysis with robust data preprocessing and error handling"""
    logger.info("🔍 Running SHAP analysis...")
    
    for target in target_cols:
        try:
            logger.info(f"SHAP analysis for: {target}")
            
            # Enhanced data preprocessing
            X_raw = df[feature_cols].copy()
            y = pd.to_numeric(df[target], errors='coerce')
            
            # Remove rows where target is NaN
            valid_target_mask = ~y.isna()
            X_raw = X_raw[valid_target_mask]
            y = y[valid_target_mask]
            
            if len(y) == 0:
                logger.warning(f"No valid data for target {target}, skipping SHAP...")
                continue
                
            if y.nunique() <= 1:
                logger.warning(f"Target {target} has zero variance, skipping SHAP...")
                continue
            
            # Robust feature preprocessing
            X_processed = pd.DataFrame(index=X_raw.index)
            processed_feature_names = []
            
            for col in feature_cols:
                if col not in X_raw.columns:
                    continue
                    
                series = X_raw[col]
                
                # Check data type and convert accordingly
                if series.dtype == 'object' or series.dtype.name == 'string':
                    # Try to convert to numeric first
                    numeric_series = pd.to_numeric(series, errors='coerce')
                    
                    if numeric_series.notna().sum() > len(numeric_series) * 0.5:  # If >50% can be converted
                        # Use numeric version, fill NaN with median
                        X_processed[col] = numeric_series.fillna(numeric_series.median())
                        processed_feature_names.append(col)
                        logger.info(f"Converted {col} to numeric")
                    else:
                        # Handle as categorical - use label encoding for top categories
                        try:
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            
                            # Fill NaN with 'MISSING'
                            series_filled = series.fillna('MISSING').astype(str)
                            
                            # If too many categories, keep only top N
                            if series_filled.nunique() > 50:
                                top_categories = series_filled.value_counts().head(49).index
                                series_filled = series_filled.apply(
                                    lambda x: x if x in top_categories else 'OTHER'
                                )
                            
                            X_processed[col] = le.fit_transform(series_filled)
                            processed_feature_names.append(col)
                            logger.info(f"Label encoded {col} ({series_filled.nunique()} categories)")
                            
                        except Exception as e:
                            logger.warning(f"Could not process categorical column {col}: {e}")
                            continue
                else:
                    # Already numeric, just handle NaN
                    if series.dtype in ['int64', 'float64', 'int32', 'float32']:
                        X_processed[col] = pd.to_numeric(series, errors='coerce').fillna(0)
                        processed_feature_names.append(col)
                    else:
                        # Try to convert other types to numeric
                        try:
                            numeric_converted = pd.to_numeric(series, errors='coerce')
                            X_processed[col] = numeric_converted.fillna(0)
                            processed_feature_names.append(col)
                            logger.info(f"Converted {col} from {series.dtype} to numeric")
                        except Exception as e:
                            logger.warning(f"Could not convert {col} to numeric: {e}")
                            continue
            
            if len(processed_feature_names) == 0:
                logger.error(f"No features could be processed for target {target}")
                continue
            
            # Final data preparation
            X = X_processed[processed_feature_names].copy()
            
            # Remove any remaining inf or extreme values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            # Remove constant columns
            constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_cols:
                logger.info(f"Removing constant columns for {target}: {constant_cols}")
                X = X.drop(columns=constant_cols)
                processed_feature_names = [col for col in processed_feature_names if col not in constant_cols]
            
            if X.shape[1] == 0:
                logger.error(f"No variable features remain for target {target}")
                continue
            
            # Ensure all data is float64
            X = X.astype(np.float64)
            y = y.astype(np.float64)
            
            logger.info(f"Final data shape for {target}: X={X.shape}, y={len(y)}")
            logger.info(f"Features being used: {list(X.columns)}")
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Handle potential fitting issues
            try:
                model.fit(X, y)
                logger.info(f"Model fitted successfully for {target}")
            except Exception as e:
                logger.error(f"Model fitting failed for {target}: {e}")
                continue
            
            # SHAP Analysis with smaller sample for performance
            n_samples = min(5000, len(X))  # Reduced sample size for stability
            n_samples = len(X) if len(X) < 5000 else n_samples
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_indices].copy()
            
            logger.info(f"Using {n_samples} samples for SHAP analysis")
            
            # Create SHAP explainer with error handling
            try:
                # Use TreeExplainer for Random Forest (more stable)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                logger.info(f"SHAP values computed successfully for {target}")
                
                # SHAP Summary Plot (Bar) - Feature importance
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar", max_display=15)
                plt.title(f"SHAP Feature Importance: {target}", fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/shap_bar_{target}.png", dpi=300, bbox_inches='tight')
                #plt.show()
                
                # SHAP Summary Plot (Detailed) - Feature effects
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
                plt.title(f"SHAP Summary Plot: {target}", fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/shap_summary_{target}.png", dpi=300, bbox_inches='tight')
                #plt.show()
                
                # SHAP Waterfall plot for representative sample
                try:
                    # Find sample with prediction closest to median
                    predictions = model.predict(X_sample)
                    median_pred = np.median(predictions)
                    median_idx = np.argmin(np.abs(predictions - median_pred))
                    
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[median_idx], 
                            base_values=explainer.expected_value, 
                            data=X_sample.iloc[median_idx].values,
                            feature_names=X_sample.columns.tolist()
                        ), 
                        show=False, 
                        max_display=10
                    )
                    plt.title(f"SHAP Waterfall - Representative Sample: {target}")
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/shap_waterfall_{target}.png", dpi=300, bbox_inches='tight')
                    #plt.show()
                    
                except Exception as e:
                    logger.warning(f"Error in waterfall plot for {target}: {e}")
                
                # Feature importance summary
                feature_importance = np.abs(shap_values).mean(0)
                importance_df = pd.DataFrame({
                    'feature': X_sample.columns,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                logger.info(f"Top 5 SHAP features for {target}:")
                for idx, row in importance_df.head().iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                
            except Exception as e:
                logger.error(f"SHAP computation failed for {target}: {e}")
                # Try alternative approach with Explainer
                try:
                    logger.info(f"Trying alternative SHAP approach for {target}")
                    explainer = shap.Explainer(model, X_sample[:50])  # Use smaller background
                    shap_values_alt = explainer(X_sample[:100])  # Explain smaller sample
                    
                    # Simple bar plot
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values_alt, show=False, plot_type="bar", max_display=10)
                    plt.title(f"SHAP Feature Importance (Alternative): {target}")
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/shap_alt_{target}.png", dpi=300, bbox_inches='tight')
                    #plt.show()
                    
                except Exception as e2:
                    logger.error(f"Alternative SHAP approach also failed for {target}: {e2}")
                    continue
            
        except Exception as e:
            logger.error(f"Overall SHAP error for {target}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

# Additional helper function for data inspection
def inspect_data_types(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Inspect data types and provide recommendations"""
    logger.info("🔍 Inspecting data types...")
    
    type_summary = {}
    for col in feature_cols:
        if col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            type_summary[col] = {
                'dtype': dtype,
                'null_count': null_count,
                'null_percentage': (null_count / len(df)) * 100,
                'unique_count': unique_count,
                'sample_values': df[col].dropna().head(3).tolist()
            }
    
    # Print summary
    problematic_cols = []
    for col, info in type_summary.items():
        if info['dtype'] == 'object' and info['unique_count'] > 50:
            problematic_cols.append(col)
            logger.warning(f"{col}: object type with {info['unique_count']} unique values")
        elif info['null_percentage'] > 50:
            logger.warning(f"{col}: {info['null_percentage']:.1f}% missing values")
    
    if problematic_cols:
        logger.info(f"Potentially problematic columns: {problematic_cols}")
    
    return type_summary

def create_target_relationships_analysis(df: pd.DataFrame, target_cols: List[str], output_dir: str = "."):
    """Analysis of relationships between target metrics - focused on execution_accuracy"""
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
    #plt.show()
    
    # Scatter plots: ALL other targets vs execution_accuracy
    if 'tuple_constraint' in target_cols:
        other_targets = [col for col in target_cols if col != 'tuple_constraint']
        
        if len(other_targets) > 0:
            # Calculate number of rows needed (max 3 columns)
            n_plots = len(other_targets)
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            
            # Handle case where we have only one plot
            if n_plots == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            else:
                axes = axes.flatten()
            
            for idx, target in enumerate(other_targets):
                ax = axes[idx]
                
                # Remove NaN for plotting
                plot_data = df[['tuple_constraint', target]].dropna()
                
                if len(plot_data) > 0:
                    ax.scatter(plot_data['tuple_constraint'], plot_data[target], 
                             alpha=0.6, s=30, color=plt.cm.Set1(idx))
                    
                    # Regression line
                    try:
                        from scipy import stats
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            plot_data['tuple_constraint'], plot_data[target])
                        line = slope * plot_data['tuple_constraint'] + intercept
                        ax.plot(plot_data['tuple_constraint'], line, 'r--', alpha=0.8)
                        ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except ImportError:
                        pass
                
                ax.set_xlabel('tuple_constraint')
                ax.set_ylabel(target)
                ax.set_title(f'tuple_constraint vs {target}')
                ax.grid(True, alpha=0.3)
            
            # Remove empty axes if any
            for idx in range(len(other_targets), len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/target_tuple_constraint_relationships.png", dpi=300, bbox_inches='tight')
            #plt.show()
        
        else:
            logger.warning("No other targets found to compare with execution_accuracy")
    
    else:
        logger.warning("execution_accuracy not found in target columns, using original method")
        # Fallback to original method if execution_accuracy is not available
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        combinations = [(target_cols[i], target_cols[j]) 
                       for i in range(len(target_cols)) 
                       for j in range(i+1, len(target_cols))][:4]
        
        for idx, (target1, target2) in enumerate(combinations):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Remove NaN for plotting
            plot_data = df[[target1, target2]].dropna()
            
            if len(plot_data) > 0:
                ax.scatter(plot_data[target1], plot_data[target2], alpha=0.6, s=30)
                
                # Regression line
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[target1], plot_data[target2])
                    line = slope * plot_data[target1] + intercept
                    ax.plot(plot_data[target1], line, 'r--', alpha=0.8)
                    ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except ImportError:
                    pass
            
            ax.set_xlabel(target1)
            ax.set_ylabel(target2)
            ax.set_title(f'{target1} vs {target2}')
            ax.grid(True, alpha=0.3)
        
        # Remove empty axes
        for idx in range(len(combinations), 4):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_relationships.png", dpi=300, bbox_inches='tight')
        #plt.show()


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
    #plt.show()
    
    # Scatter plots between main targets
    if len(target_cols) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        combinations = [(target_cols[i], target_cols[j]) 
                       for i in range(len(target_cols)) 
                       for j in range(i+1, len(target_cols))][:4]
        
        for idx, (target1, target2) in enumerate(combinations):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Remove NaN for plotting
            plot_data = df[[target1, target2]].dropna()
            
            if len(plot_data) > 0:
                ax.scatter(plot_data[target1], plot_data[target2], alpha=0.6, s=30)
                
                # Regression line
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[target1], plot_data[target2])
                    line = slope * plot_data[target1] + intercept
                    ax.plot(plot_data[target1], line, 'r--', alpha=0.8)
                    ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except ImportError:
                    pass
            
            ax.set_xlabel(target1)
            ax.set_ylabel(target2)
            ax.set_title(f'{target1} vs {target2}')
            ax.grid(True, alpha=0.3)
        
        # Remove empty axes
        for idx in range(len(combinations), 4):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_relationships.png", dpi=300, bbox_inches='tight')
        #plt.show()

def create_summary_report(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], 
                         importance_results: Dict, model_performance: Dict, output_dir: str = ".", input_file: str = "data.csv"):
    """Creates a summary report of the analysis"""
    logger.info("📋 Creating summary report...")
    
    report = []
    report.append("# METRICS ANALYSIS REPORT\n")
    report.append(f"input file: {input_file}\n")
    report.append(f"Dataset: {len(df)} rows, {len(feature_cols)} features, {len(target_cols)} targets\n")
    report.append("=" * 50 + "\n\n")
    
    # General statistics
    report.append("## GENERAL STATISTICS\n")
    report.append(f"Total rows: {len(df)}\n")
    report.append(f"Available features: {len(feature_cols)}\n")
    report.append(f"Available targets: {len(target_cols)}\n")
    
    # Missing values
    missing_stats = df[feature_cols + target_cols].isnull().sum()
    if missing_stats.sum() > 0:
        report.append(f"Total missing values: {missing_stats.sum()}\n")
        top_missing = missing_stats.sort_values(ascending=False).head(5)
        report.append("Top 5 columns with missing values:\n")
        for col, count in top_missing.items():
            if count > 0:
                report.append(f"  - {col}: {count} ({count/len(df)*100:.1f}%)\n")
    
    report.append("\n")
    
    # Model performance
    if model_performance:
        report.append("## MODEL PERFORMANCE\n")
        for target, models in model_performance.items():
            report.append(f"\n### {target}:\n")
            for model_name, metrics in models.items():
                report.append(f"  {model_name}:\n")
                report.append(f"    - R² Test: {metrics['r2_test']:.3f}\n")
                report.append(f"    - R² CV: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}\n")
                report.append(f"    - MAE: {metrics['mae_test']:.3f}\n")
    
    # Top features per target
    if importance_results:
        report.append("\n## TOP FEATURES PER TARGET\n")
        for target, importances in importance_results.items():
            report.append(f"\n### {target}:\n")
            top_5 = importances.head(5)
            for i, (feature, importance) in enumerate(top_5.items(), 1):
                report.append(f"  {i}. {feature}: {importance:.3f}\n")
    
    # Save report
    with open(f"{output_dir}/analysis_report.txt", "w", encoding="utf-8") as f:
        f.writelines(report)
    
    logger.info(f"📄 Report saved to: {output_dir}/analysis_report.txt")

def main(input_csv: str, output_dir: str = "."):
    """Enhanced main function"""
    setup_plotting()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # Load data
        df = pd.read_csv(input_csv)
        logger.info(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Column definitions
        target_cols = [
            "cell_precision", "cell_recall", "execution_accuracy",
            "tuple_cardinality", "tuple_constraint", "tuple_order"
        ]
        
        # Extended features (includes new ones from enhanced version)
        feature_cols = [
            "num_columns", "num_rows", "num_numeric_columns", "num_categorical_columns",
            "numeric_categorical_ratio", "avg_unique_vals", "max_unique_vals", "normalized_cardinality", "data_sparsity",
            "num_selected_columns", "num_order_columns", "syntax_complexity",
            "has_where", "has_group_by", "has_having", "has_order_by", "has_limit",
            "has_distinct", "has_join", "has_subquery", "has_union", "has_aggregation",
            "query_complexity", "text_length", "num_tokens", "num_words", "num_nouns",
            "num_verbs", "num_adjectives", "num_entities", "num_clauses",
            "avg_sentence_length", "keyword_score", "semantic_complexity", "token_prompt_count"
        ]
        
        # Validate column availability
        feature_cols, target_cols = validate_data(df, feature_cols, target_cols)
        
        if not feature_cols or not target_cols:
            logger.error("No features or targets available for analysis")
            return
        
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
                            model_performance, output_dir, csv_path)
        
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"📁 All files have been saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "enriched_results/FILTRATI/ALL_DeepSeek_Llama_enriched_with_big.csv"
    output_directory = sys.argv[2] if len(sys.argv) > 2 else f"analysis_results/FILTRATO/DS_LAMA_BIG_DATA"
    
    main(csv_path, output_directory)