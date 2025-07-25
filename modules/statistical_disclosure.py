"""
Statistical Disclosure Control Module for SafeData Pipeline
Implements suppression, generalization, perturbation, and microaggregation techniques
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import copy

class StatisticalDisclosureControl:
    """Statistical Disclosure Control techniques implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def apply_suppression(self, dataset: pd.DataFrame, threshold: int = 5, 
                         columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply suppression to low-frequency combinations
        
        Args:
            dataset: Input dataset
            threshold: Minimum frequency threshold
            columns: Columns to consider for suppression (default: all)
            
        Returns:
            Dictionary with suppressed dataset and metrics
        """
        self.logger.info(f"Applying suppression with threshold {threshold}")
        
        try:
            suppressed_dataset = dataset.copy()
            total_suppressed = 0
            
            if columns is None:
                columns = list(dataset.columns)
            
            # Find combinations that occur less than threshold times
            for i, column in enumerate(columns):
                value_counts = dataset[column].value_counts()
                low_freq_values = value_counts[value_counts < threshold].index
                
                if len(low_freq_values) > 0:
                    # Suppress low-frequency values
                    suppressed_count = (dataset[column].isin(low_freq_values)).sum()
                    suppressed_dataset.loc[dataset[column].isin(low_freq_values), column] = '*'
                    total_suppressed += suppressed_count
                    
                    self.logger.debug(f"Suppressed {suppressed_count} values in column {column}")
            
            suppression_rate = total_suppressed / (len(dataset) * len(columns))
            
            result = {
                'dataset': suppressed_dataset,
                'suppressed_values': total_suppressed,
                'suppression_rate': suppression_rate,
                'threshold_used': threshold,
                'columns_processed': columns
            }
            
            self.logger.info(f"Suppression completed: {total_suppressed} values suppressed ({suppression_rate:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Suppression failed: {str(e)}")
            raise
    
    def apply_generalization(self, dataset: pd.DataFrame, levels: int = 2,
                           columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply generalization to reduce data granularity
        
        Args:
            dataset: Input dataset
            levels: Number of generalization levels
            columns: Columns to generalize (default: all suitable columns)
            
        Returns:
            Dictionary with generalized dataset and metrics
        """
        self.logger.info(f"Applying generalization with {levels} levels")
        
        try:
            generalized_dataset = dataset.copy()
            generalization_mappings = {}
            total_generalized = 0
            
            if columns is None:
                # Auto-select suitable columns for generalization
                columns = self._select_generalizable_columns(dataset)
            
            for column in columns:
                if column in dataset.columns:
                    result = self._generalize_column(dataset[column], levels)
                    generalized_dataset[column] = result['generalized_values']
                    generalization_mappings[column] = result['mapping']
                    total_generalized += result['values_changed']
            
            generalization_rate = total_generalized / (len(dataset) * len(columns)) if columns else 0
            
            result = {
                'dataset': generalized_dataset,
                'generalization_mappings': generalization_mappings,
                'generalization_rate': generalization_rate,
                'levels_used': levels,
                'columns_processed': columns
            }
            
            self.logger.info(f"Generalization completed: {len(columns)} columns processed")
            return result
            
        except Exception as e:
            self.logger.error(f"Generalization failed: {str(e)}")
            raise
    
    def apply_perturbation(self, dataset: pd.DataFrame, noise_level: float = 0.1,
                          columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply perturbation by adding controlled noise
        
        Args:
            dataset: Input dataset
            noise_level: Level of noise to add (relative to data range)
            columns: Columns to perturb (default: numeric columns)
            
        Returns:
            Dictionary with perturbed dataset and metrics
        """
        self.logger.info(f"Applying perturbation with noise level {noise_level}")
        
        try:
            perturbed_dataset = dataset.copy()
            noise_details = {}
            
            if columns is None:
                # Auto-select numeric columns
                columns = list(dataset.select_dtypes(include=[np.number]).columns)
            
            for column in columns:
                if column in dataset.columns and pd.api.types.is_numeric_dtype(dataset[column]):
                    result = self._perturb_column(dataset[column], noise_level)
                    perturbed_dataset[column] = result['perturbed_values']
                    noise_details[column] = result['noise_stats']
            
            result = {
                'dataset': perturbed_dataset,
                'noise_details': noise_details,
                'noise_variance': np.mean([stats['variance'] for stats in noise_details.values()]) if noise_details else 0,
                'noise_level_used': noise_level,
                'columns_processed': columns
            }
            
            self.logger.info(f"Perturbation completed: {len(columns)} columns processed")
            return result
            
        except Exception as e:
            self.logger.error(f"Perturbation failed: {str(e)}")
            raise
    
    def apply_microaggregation(self, dataset: pd.DataFrame, k: int = 3,
                              columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply microaggregation to replace values with group averages
        
        Args:
            dataset: Input dataset
            k: Minimum group size for aggregation
            columns: Columns to aggregate (default: numeric columns)
            
        Returns:
            Dictionary with microaggregated dataset and metrics
        """
        self.logger.info(f"Applying microaggregation with k={k}")
        
        try:
            aggregated_dataset = dataset.copy()
            aggregation_stats = {}
            
            if columns is None:
                columns = list(dataset.select_dtypes(include=[np.number]).columns)
            
            # Perform clustering for microaggregation
            if len(columns) > 0:
                # Use all specified numeric columns for clustering
                clustering_data = dataset[columns].dropna()
                
                if len(clustering_data) >= k:
                    # Standardize data for clustering
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(clustering_data)
                    
                    # Determine number of clusters
                    n_clusters = max(1, len(clustering_data) // k)
                    
                    # Perform K-means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(scaled_data)
                    
                    # Apply microaggregation within each cluster
                    for column in columns:
                        aggregation_stats[column] = self._microaggregate_column(
                            clustering_data, column, cluster_labels, k
                        )
                        
                        # Update the dataset
                        for idx, orig_idx in enumerate(clustering_data.index):
                            cluster_id = cluster_labels[idx]
                            aggregated_dataset.loc[orig_idx, column] = aggregation_stats[column]['cluster_means'][cluster_id]
            
            result = {
                'dataset': aggregated_dataset,
                'aggregation_stats': aggregation_stats,
                'k_used': k,
                'columns_processed': columns,
                'groups_created': len(set(cluster_labels)) if 'cluster_labels' in locals() else 0
            }
            
            self.logger.info(f"Microaggregation completed: {len(columns)} columns processed")
            return result
            
        except Exception as e:
            self.logger.error(f"Microaggregation failed: {str(e)}")
            raise
    
    def _select_generalizable_columns(self, dataset: pd.DataFrame) -> List[str]:
        """Select columns suitable for generalization"""
        
        generalizable_columns = []
        
        for column in dataset.columns:
            # Check if column is suitable for generalization
            if pd.api.types.is_numeric_dtype(dataset[column]):
                # Numeric columns with reasonable range
                col_range = dataset[column].max() - dataset[column].min()
                if col_range > 10:  # Arbitrary threshold
                    generalizable_columns.append(column)
            elif dataset[column].dtype == 'object' or dataset[column].dtype.name == 'category':
                # Categorical columns with moderate cardinality
                unique_count = dataset[column].nunique()
                if 5 <= unique_count <= 100:
                    generalizable_columns.append(column)
        
        return generalizable_columns
    
    def _generalize_column(self, series: pd.Series, levels: int) -> Dict[str, Any]:
        """Generalize a single column"""
        
        if pd.api.types.is_numeric_dtype(series):
            return self._generalize_numeric_column(series, levels)
        else:
            return self._generalize_categorical_column(series, levels)
    
    def _generalize_numeric_column(self, series: pd.Series, levels: int) -> Dict[str, Any]:
        """Generalize numeric column using binning"""
        
        try:
            # Create bins for generalization
            min_val, max_val = series.min(), series.max()
            
            if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                return {
                    'generalized_values': series,
                    'mapping': {},
                    'values_changed': 0
                }
            
            # Create equal-width bins
            bins = np.linspace(min_val, max_val, levels + 1)
            bin_labels = [f"[{bins[i]:.2f}-{bins[i+1]:.2f})" for i in range(len(bins)-1)]
            bin_labels[-1] = bin_labels[-1][:-1] + "]"  # Make last bin inclusive
            
            # Apply binning
            generalized_values = pd.cut(series, bins=bins, labels=bin_labels, include_lowest=True)
            
            # Count values changed
            values_changed = len(series) - series.isna().sum()
            
            # Create mapping
            mapping = {}
            for i, label in enumerate(bin_labels):
                mapping[label] = f"Range {i+1}: {label}"
            
            return {
                'generalized_values': generalized_values.astype(str),
                'mapping': mapping,
                'values_changed': values_changed
            }
            
        except Exception as e:
            self.logger.warning(f"Numeric generalization failed: {str(e)}")
            return {
                'generalized_values': series,
                'mapping': {},
                'values_changed': 0
            }
    
    def _generalize_categorical_column(self, series: pd.Series, levels: int) -> Dict[str, Any]:
        """Generalize categorical column by grouping similar values"""
        
        try:
            value_counts = series.value_counts()
            
            if len(value_counts) <= levels:
                # No generalization needed
                return {
                    'generalized_values': series,
                    'mapping': {val: val for val in value_counts.index},
                    'values_changed': 0
                }
            
            # Group low-frequency values together
            sorted_values = value_counts.sort_values(ascending=False)
            
            # Keep top (levels-1) values, group the rest
            top_values = sorted_values.head(levels - 1).index.tolist()
            other_values = sorted_values.tail(len(sorted_values) - (levels - 1)).index.tolist()
            
            # Create mapping
            mapping = {val: val for val in top_values}
            for val in other_values:
                mapping[val] = "Other"
            
            # Apply generalization
            generalized_values = series.map(mapping)
            values_changed = series.isin(other_values).sum()
            
            return {
                'generalized_values': generalized_values,
                'mapping': mapping,
                'values_changed': values_changed
            }
            
        except Exception as e:
            self.logger.warning(f"Categorical generalization failed: {str(e)}")
            return {
                'generalized_values': series,
                'mapping': {},
                'values_changed': 0
            }
    
    def _perturb_column(self, series: pd.Series, noise_level: float) -> Dict[str, Any]:
        """Add noise to a numeric column"""
        
        try:
            non_null_data = series.dropna()
            
            if len(non_null_data) == 0:
                return {
                    'perturbed_values': series,
                    'noise_stats': {'variance': 0, 'mean': 0, 'std': 0}
                }
            
            # Calculate noise parameters based on data range
            data_range = non_null_data.max() - non_null_data.min()
            noise_std = data_range * noise_level
            
            # Generate noise
            noise = np.random.normal(0, noise_std, len(series))
            
            # Apply noise only to non-null values
            perturbed_values = series.copy()
            perturbed_values = perturbed_values + noise
            
            # Calculate noise statistics
            noise_stats = {
                'variance': float(np.var(noise)),
                'mean': float(np.mean(noise)),
                'std': float(np.std(noise)),
                'range_ratio': noise_level
            }
            
            return {
                'perturbed_values': perturbed_values,
                'noise_stats': noise_stats
            }
            
        except Exception as e:
            self.logger.warning(f"Perturbation failed: {str(e)}")
            return {
                'perturbed_values': series,
                'noise_stats': {'variance': 0, 'mean': 0, 'std': 0}
            }
    
    def _microaggregate_column(self, data: pd.DataFrame, column: str, 
                              cluster_labels: np.ndarray, k: int) -> Dict[str, Any]:
        """Microaggregate values within clusters"""
        
        try:
            cluster_means = {}
            cluster_stats = {}
            
            for cluster_id in np.unique(cluster_labels):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = data[column].iloc[cluster_mask]
                
                if len(cluster_data) >= k:
                    # Calculate cluster mean
                    cluster_mean = cluster_data.mean()
                    cluster_means[cluster_id] = cluster_mean
                    
                    # Calculate statistics
                    cluster_stats[cluster_id] = {
                        'size': len(cluster_data),
                        'mean': float(cluster_mean),
                        'original_std': float(cluster_data.std()),
                        'min': float(cluster_data.min()),
                        'max': float(cluster_data.max())
                    }
                else:
                    # For small clusters, use original values
                    cluster_means[cluster_id] = cluster_data.iloc[0] if len(cluster_data) > 0 else 0
                    cluster_stats[cluster_id] = {
                        'size': len(cluster_data),
                        'mean': float(cluster_data.mean()) if len(cluster_data) > 0 else 0,
                        'original_std': 0,
                        'min': float(cluster_data.min()) if len(cluster_data) > 0 else 0,
                        'max': float(cluster_data.max()) if len(cluster_data) > 0 else 0
                    }
            
            return {
                'cluster_means': cluster_means,
                'cluster_stats': cluster_stats,
                'total_clusters': len(cluster_means)
            }
            
        except Exception as e:
            self.logger.warning(f"Microaggregation failed for column {column}: {str(e)}")
            return {
                'cluster_means': {},
                'cluster_stats': {},
                'total_clusters': 0
            }
    
    def compute_information_loss(self, original: pd.DataFrame, modified: pd.DataFrame,
                                columns: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute information loss metrics
        
        Args:
            original: Original dataset
            modified: Modified dataset after SDC
            columns: Columns to analyze (default: all numeric)
            
        Returns:
            Dictionary with information loss metrics
        """
        
        if columns is None:
            columns = list(original.select_dtypes(include=[np.number]).columns)
        
        metrics = {}
        
        try:
            for column in columns:
                if column in original.columns and column in modified.columns:
                    # Calculate various information loss metrics
                    orig_data = original[column].dropna()
                    mod_data = modified[column].dropna()
                    
                    if len(orig_data) > 0 and len(mod_data) > 0:
                        # Mean squared error
                        min_len = min(len(orig_data), len(mod_data))
                        mse = np.mean((orig_data.iloc[:min_len] - mod_data.iloc[:min_len]) ** 2)
                        
                        # Variance preservation
                        orig_var = orig_data.var()
                        mod_var = mod_data.var()
                        var_ratio = mod_var / orig_var if orig_var > 0 else 0
                        
                        # Mean preservation
                        mean_diff = abs(orig_data.mean() - mod_data.mean())
                        
                        metrics[column] = {
                            'mse': float(mse),
                            'variance_ratio': float(var_ratio),
                            'mean_difference': float(mean_diff)
                        }
            
            # Overall information loss
            if metrics:
                overall_mse = np.mean([m['mse'] for m in metrics.values()])
                overall_var_ratio = np.mean([m['variance_ratio'] for m in metrics.values()])
                
                metrics['overall'] = {
                    'mse': overall_mse,
                    'variance_preservation': overall_var_ratio,
                    'information_loss_score': 1.0 - overall_var_ratio
                }
            
        except Exception as e:
            self.logger.error(f"Information loss computation failed: {str(e)}")
        
        return metrics
