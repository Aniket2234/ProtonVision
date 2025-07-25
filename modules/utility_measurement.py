"""
Utility Measurement Module for SafeData Pipeline
Quantifies analytical value preserved after privacy enhancement
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

class UtilityMeasurement:
    """Main class for utility measurement and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_dataset(self, enhanced_data: Dict, config: Dict) -> Dict[str, Any]:
        """
        Analyze utility preservation in enhanced dataset
        
        Args:
            enhanced_data: Dictionary containing original and enhanced datasets
            config: Configuration parameters for utility analysis
            
        Returns:
            Dictionary containing utility analysis results
        """
        self.logger.info("Starting utility analysis")
        
        try:
            original_dataset = enhanced_data.get('original_dataset')
            enhanced_dataset = enhanced_data.get('enhanced_dataset')
            
            if original_dataset is None or enhanced_dataset is None:
                raise ValueError("Original and enhanced datasets required for utility analysis")
            
            selected_metrics = config.get('metrics', [])
            results = {}
            
            # Statistical Utility Metrics
            if any(metric in selected_metrics for metric in 
                   ["Mean Absolute Error", "Root Mean Square Error", "Correlation Preservation", "KL Divergence"]):
                statistical_utility = self._calculate_statistical_utility(original_dataset, enhanced_dataset)
                results.update(statistical_utility)
            
            # Analytical Utility Metrics
            if any(metric in selected_metrics for metric in 
                   ["Query Accuracy", "Model Performance", "Hypothesis Testing"]):
                analytical_utility = self._calculate_analytical_utility(original_dataset, enhanced_dataset)
                results.update(analytical_utility)
            
            # Overall utility score
            overall_utility = self._calculate_overall_utility(results)
            results['overall_utility'] = overall_utility
            
            # Distribution comparison
            distribution_analysis = self._analyze_distributions(original_dataset, enhanced_dataset)
            results['distribution_analysis'] = distribution_analysis
            
            # Query preservation analysis
            query_preservation = self._analyze_query_preservation(original_dataset, enhanced_dataset)
            results['query_preservation'] = query_preservation
            
            # Generate utility recommendations
            recommendations = self._generate_utility_recommendations(results)
            results['recommendations'] = recommendations
            
            self.logger.info("Utility analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Utility analysis failed: {str(e)}")
            raise
    
    def _calculate_statistical_utility(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical utility metrics"""
        
        results = {}
        
        try:
            # Ensure datasets have same structure
            common_columns = list(set(original.columns) & set(enhanced.columns))
            if not common_columns:
                return {'mae': 1.0, 'rmse': 1.0, 'correlation': 0.0}
            
            # Mean Absolute Error for numeric columns
            numeric_cols = original[common_columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                mae_scores = []
                rmse_scores = []
                
                for col in numeric_cols:
                    try:
                        orig_vals = original[col].dropna()
                        enh_vals = enhanced[col].dropna()
                        
                        if len(orig_vals) > 0 and len(enh_vals) > 0:
                            # Align the data (take minimum length)
                            min_len = min(len(orig_vals), len(enh_vals))
                            orig_sample = orig_vals.iloc[:min_len]
                            enh_sample = enh_vals.iloc[:min_len]
                            
                            # Normalize to 0-1 range for comparison
                            orig_norm = (orig_sample - orig_sample.min()) / (orig_sample.max() - orig_sample.min() + 1e-8)
                            enh_norm = (enh_sample - enh_sample.min()) / (enh_sample.max() - enh_sample.min() + 1e-8)
                            
                            mae = mean_absolute_error(orig_norm, enh_norm)
                            rmse = np.sqrt(mean_squared_error(orig_norm, enh_norm))
                            
                            mae_scores.append(mae)
                            rmse_scores.append(rmse)
                    except:
                        continue
                
                results['mae'] = np.mean(mae_scores) if mae_scores else 1.0
                results['rmse'] = np.mean(rmse_scores) if rmse_scores else 1.0
            else:
                results['mae'] = 0.0
                results['rmse'] = 0.0
            
            # Correlation preservation
            correlation_score = self._calculate_correlation_preservation(original, enhanced, common_columns)
            results['correlation'] = correlation_score
            
            # KL Divergence for categorical columns
            kl_divergence = self._calculate_kl_divergence(original, enhanced, common_columns)
            results['kl_divergence'] = kl_divergence
            
        except Exception as e:
            self.logger.warning(f"Statistical utility calculation failed: {str(e)}")
            results = {'mae': 1.0, 'rmse': 1.0, 'correlation': 0.0, 'kl_divergence': 1.0}
        
        return results
    
    def _calculate_analytical_utility(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> Dict[str, float]:
        """Calculate analytical utility metrics"""
        
        results = {}
        
        try:
            # Query accuracy
            query_accuracy = self._test_query_accuracy(original, enhanced)
            results['query_accuracy'] = query_accuracy
            
            # Model performance preservation
            model_performance = self._test_model_performance(original, enhanced)
            results['model_performance'] = model_performance
            
            # Hypothesis testing preservation
            hypothesis_preservation = self._test_hypothesis_preservation(original, enhanced)
            results['hypothesis_preservation'] = hypothesis_preservation
            
        except Exception as e:
            self.logger.warning(f"Analytical utility calculation failed: {str(e)}")
            results = {'query_accuracy': 0.0, 'model_performance': 0.0, 'hypothesis_preservation': 0.0}
        
        return results
    
    def _calculate_correlation_preservation(self, original: pd.DataFrame, enhanced: pd.DataFrame, common_columns: List[str]) -> float:
        """Calculate correlation preservation between datasets"""
        
        try:
            numeric_cols = original[common_columns].select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return 1.0
            
            # Calculate correlation matrices
            orig_corr = original[numeric_cols].corr()
            enh_corr = enhanced[numeric_cols].corr()
            
            # Calculate correlation between correlation matrices
            orig_corr_flat = orig_corr.values.flatten()
            enh_corr_flat = enh_corr.values.flatten()
            
            # Remove NaN values
            mask = ~(np.isnan(orig_corr_flat) | np.isnan(enh_corr_flat))
            orig_corr_clean = orig_corr_flat[mask]
            enh_corr_clean = enh_corr_flat[mask]
            
            if len(orig_corr_clean) > 1:
                correlation = np.corrcoef(orig_corr_clean, enh_corr_clean)[0, 1]
                return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            pass
        
        return 0.0
    
    def _calculate_kl_divergence(self, original: pd.DataFrame, enhanced: pd.DataFrame, common_columns: List[str]) -> float:
        """Calculate KL divergence for categorical columns"""
        
        try:
            categorical_cols = original[common_columns].select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) == 0:
                return 0.0
            
            kl_scores = []
            for col in categorical_cols:
                try:
                    # Get value distributions
                    orig_dist = original[col].value_counts(normalize=True)
                    enh_dist = enhanced[col].value_counts(normalize=True)
                    
                    # Align distributions
                    all_values = set(orig_dist.index) | set(enh_dist.index)
                    orig_aligned = [orig_dist.get(val, 1e-8) for val in all_values]
                    enh_aligned = [enh_dist.get(val, 1e-8) for val in all_values]
                    
                    # Calculate KL divergence
                    kl_div = stats.entropy(orig_aligned, enh_aligned)
                    if not np.isnan(kl_div) and not np.isinf(kl_div):
                        kl_scores.append(kl_div)
                        
                except:
                    continue
            
            return np.mean(kl_scores) if kl_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _test_query_accuracy(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> float:
        """Test accuracy of common statistical queries"""
        
        try:
            query_scores = []
            common_columns = list(set(original.columns) & set(enhanced.columns))
            numeric_cols = original[common_columns].select_dtypes(include=[np.number]).columns
            
            # Test mean preservation
            for col in numeric_cols:
                try:
                    orig_mean = original[col].mean()
                    enh_mean = enhanced[col].mean()
                    if not (np.isnan(orig_mean) or np.isnan(enh_mean)) and orig_mean != 0:
                        relative_error = abs(orig_mean - enh_mean) / abs(orig_mean)
                        accuracy = max(0.0, 1.0 - relative_error)
                        query_scores.append(accuracy)
                except:
                    continue
            
            # Test sum preservation
            for col in numeric_cols:
                try:
                    orig_sum = original[col].sum()
                    enh_sum = enhanced[col].sum()
                    if not (np.isnan(orig_sum) or np.isnan(enh_sum)) and orig_sum != 0:
                        relative_error = abs(orig_sum - enh_sum) / abs(orig_sum)
                        accuracy = max(0.0, 1.0 - relative_error)
                        query_scores.append(accuracy)
                except:
                    continue
            
            return np.mean(query_scores) if query_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _test_model_performance(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> float:
        """Test machine learning model performance preservation"""
        
        try:
            common_columns = list(set(original.columns) & set(enhanced.columns))
            numeric_cols = original[common_columns].select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return 0.0
            
            # Use first numeric column as target, rest as features
            target_col = numeric_cols[0]
            feature_cols = list(numeric_cols[1:])
            
            if len(feature_cols) == 0:
                return 0.0
            
            # Prepare original data
            orig_X = original[feature_cols].dropna()
            orig_y = original.loc[orig_X.index, target_col]
            
            # Prepare enhanced data
            enh_X = enhanced[feature_cols].dropna()
            enh_y = enhanced.loc[enh_X.index, target_col]
            
            if len(orig_X) < 10 or len(enh_X) < 10:
                return 0.0
            
            # Train model on original data
            X_train, X_test, y_train, y_test = train_test_split(
                orig_X, orig_y, test_size=0.3, random_state=42
            )
            
            if np.var(y_train) == 0:  # Regression if continuous target
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)
                orig_score = model.score(X_test, y_test)
            else:
                # Classification for discrete target
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train.astype(str))
                y_test_enc = le.transform(y_test.astype(str))
                
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                model.fit(X_train, y_train_enc)
                orig_score = model.score(X_test, y_test_enc)
            
            # Test model on enhanced data
            min_len = min(len(enh_X), len(X_test))
            enh_X_sample = enh_X.iloc[:min_len]
            enh_y_sample = enh_y.iloc[:min_len]
            
            if np.var(y_train) == 0:
                enh_score = model.score(enh_X_sample, enh_y_sample)
            else:
                enh_y_enc = le.transform(enh_y_sample.astype(str))
                enh_score = model.score(enh_X_sample, enh_y_enc)
            
            # Return relative performance preservation
            performance_ratio = enh_score / max(orig_score, 1e-8)
            return min(1.0, max(0.0, performance_ratio))
            
        except Exception:
            return 0.0
    
    def _test_hypothesis_preservation(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> float:
        """Test preservation of statistical relationships"""
        
        try:
            common_columns = list(set(original.columns) & set(enhanced.columns))
            numeric_cols = original[common_columns].select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return 0.0
            
            preserved_relationships = []
            
            # Test pairwise correlations
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    try:
                        # Original correlation
                        orig_corr, orig_p = stats.pearsonr(
                            original[col1].dropna(),
                            original[col2].dropna()
                        )
                        
                        # Enhanced correlation
                        enh_corr, enh_p = stats.pearsonr(
                            enhanced[col1].dropna(),
                            enhanced[col2].dropna()
                        )
                        
                        # Check if significance is preserved
                        alpha = 0.05
                        orig_significant = orig_p < alpha
                        enh_significant = enh_p < alpha
                        
                        if orig_significant == enh_significant:
                            preserved_relationships.append(1.0)
                        else:
                            preserved_relationships.append(0.0)
                            
                    except:
                        continue
            
            return np.mean(preserved_relationships) if preserved_relationships else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_overall_utility(self, results: Dict[str, float]) -> float:
        """Calculate overall utility score"""
        
        try:
            # Weight different utility aspects
            weights = {
                'mae': -0.2,  # Lower MAE is better
                'rmse': -0.2,  # Lower RMSE is better
                'correlation': 0.3,  # Higher correlation is better
                'query_accuracy': 0.3,
                'model_performance': 0.2,
                'hypothesis_preservation': 0.2
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in results:
                    value = results[metric]
                    if weight > 0:
                        weighted_score += weight * value
                    else:
                        weighted_score += abs(weight) * (1.0 - value)  # Invert for negative metrics
                    total_weight += abs(weight)
            
            if total_weight > 0:
                return max(0.0, min(1.0, weighted_score / total_weight))
            
        except Exception:
            pass
        
        return 0.5  # Default moderate utility
    
    def _analyze_distributions(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution differences between datasets"""
        
        try:
            common_columns = list(set(original.columns) & set(enhanced.columns))
            distribution_metrics = {}
            
            for col in common_columns:
                try:
                    if pd.api.types.is_numeric_dtype(original[col]):
                        # Wasserstein distance for numeric columns
                        orig_vals = original[col].dropna()
                        enh_vals = enhanced[col].dropna()
                        
                        if len(orig_vals) > 0 and len(enh_vals) > 0:
                            distance = wasserstein_distance(orig_vals, enh_vals)
                            distribution_metrics[col] = {
                                'type': 'numeric',
                                'wasserstein_distance': distance,
                                'mean_difference': abs(orig_vals.mean() - enh_vals.mean()),
                                'std_difference': abs(orig_vals.std() - enh_vals.std())
                            }
                    else:
                        # Chi-square test for categorical columns
                        orig_counts = original[col].value_counts()
                        enh_counts = enhanced[col].value_counts()
                        
                        # Align the distributions
                        all_categories = set(orig_counts.index) | set(enh_counts.index)
                        orig_aligned = [orig_counts.get(cat, 0) for cat in all_categories]
                        enh_aligned = [enh_counts.get(cat, 0) for cat in all_categories]
                        
                        if sum(orig_aligned) > 0 and sum(enh_aligned) > 0:
                            try:
                                chi2_stat, p_value = stats.chisquare(enh_aligned, orig_aligned)
                                distribution_metrics[col] = {
                                    'type': 'categorical',
                                    'chi2_statistic': chi2_stat,
                                    'p_value': p_value,
                                    'similar_distribution': p_value > 0.05
                                }
                            except:
                                distribution_metrics[col] = {
                                    'type': 'categorical',
                                    'chi2_statistic': np.inf,
                                    'p_value': 0.0,
                                    'similar_distribution': False
                                }
                                
                except Exception:
                    continue
            
            return distribution_metrics
            
        except Exception:
            return {}
    
    def _analyze_query_preservation(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> Dict[str, float]:
        """Analyze preservation of common queries"""
        
        try:
            common_columns = list(set(original.columns) & set(enhanced.columns))
            numeric_cols = original[common_columns].select_dtypes(include=[np.number]).columns
            categorical_cols = original[common_columns].select_dtypes(include=['object', 'category']).columns
            
            query_results = {}
            
            # Count queries
            query_results['total_count_accuracy'] = 1.0 - abs(len(original) - len(enhanced)) / len(original)
            
            # Aggregation queries for numeric columns
            for col in numeric_cols:
                try:
                    orig_mean = original[col].mean()
                    enh_mean = enhanced[col].mean()
                    if not np.isnan(orig_mean) and orig_mean != 0:
                        query_results[f'{col}_mean_accuracy'] = 1.0 - abs(orig_mean - enh_mean) / abs(orig_mean)
                        
                    orig_sum = original[col].sum()
                    enh_sum = enhanced[col].sum()
                    if not np.isnan(orig_sum) and orig_sum != 0:
                        query_results[f'{col}_sum_accuracy'] = 1.0 - abs(orig_sum - enh_sum) / abs(orig_sum)
                        
                except:
                    continue
            
            # Group-by queries
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                try:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    
                    orig_grouped = original.groupby(cat_col)[num_col].mean()
                    enh_grouped = enhanced.groupby(cat_col)[num_col].mean()
                    
                    common_groups = set(orig_grouped.index) & set(enh_grouped.index)
                    if common_groups:
                        accuracies = []
                        for group in common_groups:
                            orig_val = orig_grouped[group]
                            enh_val = enh_grouped[group]
                            if not np.isnan(orig_val) and orig_val != 0:
                                accuracy = 1.0 - abs(orig_val - enh_val) / abs(orig_val)
                                accuracies.append(accuracy)
                        
                        if accuracies:
                            query_results['groupby_accuracy'] = np.mean(accuracies)
                            
                except:
                    pass
            
            return query_results
            
        except Exception:
            return {}
    
    def _generate_utility_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on utility analysis"""
        
        recommendations = []
        
        try:
            overall_utility = results.get('overall_utility', 0.5)
            
            if overall_utility < 0.3:
                recommendations.append("Low utility detected. Consider reducing privacy enhancement intensity.")
                recommendations.append("Review privacy parameters to balance privacy-utility trade-off.")
                
            if results.get('mae', 0) > 0.3:
                recommendations.append("High mean absolute error. Consider different noise mechanisms.")
                
            if results.get('correlation', 1) < 0.5:
                recommendations.append("Poor correlation preservation. Review generalization strategies.")
                
            if results.get('query_accuracy', 1) < 0.7:
                recommendations.append("Query accuracy degraded significantly. Consider post-processing calibration.")
                
            if results.get('model_performance', 1) < 0.6:
                recommendations.append("Machine learning model performance severely impacted.")
                
            if overall_utility > 0.8:
                recommendations.append("Excellent utility preservation achieved!")
                
            if not recommendations:
                recommendations.append("Utility analysis complete. Consider fine-tuning privacy parameters.")
                
        except Exception:
            recommendations.append("Utility analysis completed with limited metrics.")
        
        return recommendations
