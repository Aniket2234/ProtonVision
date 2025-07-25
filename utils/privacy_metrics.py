"""
Privacy Metrics Module for SafeData Pipeline
Comprehensive privacy risk measurement and evaluation functions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import Counter
from scipy import stats
import itertools

class PrivacyMetrics:
    """Comprehensive privacy metrics calculation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_k_anonymity(self, dataset: pd.DataFrame, quasi_identifiers: List[str]) -> int:
        """
        Calculate k-anonymity level of the dataset
        
        Args:
            dataset: Input dataset
            quasi_identifiers: List of quasi-identifier columns
            
        Returns:
            k-anonymity level (minimum group size)
        """
        if not quasi_identifiers or not all(col in dataset.columns for col in quasi_identifiers):
            return len(dataset)  # If no QIs, each record is unique
        
        try:
            # Group by quasi-identifiers and find minimum group size
            grouped = dataset.groupby(quasi_identifiers, dropna=False).size()
            k_anonymity = int(grouped.min())
            
            self.logger.debug(f"K-anonymity calculated: {k_anonymity}")
            return k_anonymity
            
        except Exception as e:
            self.logger.error(f"K-anonymity calculation failed: {str(e)}")
            return 1
    
    def calculate_l_diversity(self, dataset: pd.DataFrame, quasi_identifiers: List[str], 
                            sensitive_attribute: str) -> int:
        """
        Calculate l-diversity level of the dataset
        
        Args:
            dataset: Input dataset
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attribute: Sensitive attribute column
            
        Returns:
            l-diversity level (minimum distinct sensitive values per group)
        """
        if (not quasi_identifiers or 
            not all(col in dataset.columns for col in quasi_identifiers) or
            sensitive_attribute not in dataset.columns):
            return 1
        
        try:
            # Group by quasi-identifiers and count distinct sensitive values
            grouped = dataset.groupby(quasi_identifiers, dropna=False)[sensitive_attribute].nunique()
            l_diversity = int(grouped.min())
            
            self.logger.debug(f"L-diversity calculated: {l_diversity}")
            return l_diversity
            
        except Exception as e:
            self.logger.error(f"L-diversity calculation failed: {str(e)}")
            return 1
    
    def calculate_t_closeness(self, dataset: pd.DataFrame, quasi_identifiers: List[str], 
                            sensitive_attribute: str) -> float:
        """
        Calculate t-closeness measure
        
        Args:
            dataset: Input dataset
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attribute: Sensitive attribute column
            
        Returns:
            Maximum t-closeness value across all equivalence classes
        """
        if (not quasi_identifiers or 
            not all(col in dataset.columns for col in quasi_identifiers) or
            sensitive_attribute not in dataset.columns):
            return 0.0
        
        try:
            # Global distribution of sensitive attribute
            global_dist = dataset[sensitive_attribute].value_counts(normalize=True, dropna=False)
            
            # Group by quasi-identifiers
            grouped = dataset.groupby(quasi_identifiers, dropna=False)
            
            max_distance = 0.0
            
            for name, group in grouped:
                if len(group) > 0:
                    # Local distribution
                    local_dist = group[sensitive_attribute].value_counts(normalize=True, dropna=False)
                    
                    # Calculate Earth Mover's Distance (approximated as L1 distance)
                    all_values = set(global_dist.index) | set(local_dist.index)
                    distance = sum(abs(global_dist.get(val, 0) - local_dist.get(val, 0)) 
                                 for val in all_values) / 2
                    
                    max_distance = max(max_distance, distance)
            
            self.logger.debug(f"T-closeness calculated: {max_distance:.4f}")
            return max_distance
            
        except Exception as e:
            self.logger.error(f"T-closeness calculation failed: {str(e)}")
            return 0.0
    
    def calculate_entropy_l_diversity(self, dataset: pd.DataFrame, quasi_identifiers: List[str], 
                                    sensitive_attribute: str) -> float:
        """
        Calculate entropy-based l-diversity
        
        Args:
            dataset: Input dataset
            quasi_identifiers: List of quasi-identifier columns
            sensitive_attribute: Sensitive attribute column
            
        Returns:
            Minimum entropy across all equivalence classes
        """
        if (not quasi_identifiers or 
            not all(col in dataset.columns for col in quasi_identifiers) or
            sensitive_attribute not in dataset.columns):
            return 0.0
        
        try:
            grouped = dataset.groupby(quasi_identifiers, dropna=False)
            min_entropy = float('inf')
            
            for name, group in grouped:
                if len(group) > 0:
                    # Calculate entropy of sensitive attribute in this group
                    value_counts = group[sensitive_attribute].value_counts()
                    probabilities = value_counts / len(group)
                    
                    # Calculate entropy
                    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                    min_entropy = min(min_entropy, entropy)
            
            result = min_entropy if min_entropy != float('inf') else 0.0
            self.logger.debug(f"Entropy l-diversity calculated: {result:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Entropy l-diversity calculation failed: {str(e)}")
            return 0.0
    
    def calculate_individual_risk(self, dataset: pd.DataFrame, quasi_identifiers: List[str]) -> np.ndarray:
        """
        Calculate individual re-identification risk for each record
        
        Args:
            dataset: Input dataset
            quasi_identifiers: List of quasi-identifier columns
            
        Returns:
            Array of individual risk scores
        """
        if not quasi_identifiers or not all(col in dataset.columns for col in quasi_identifiers):
            # If no QIs, assume uniform low risk
            return np.full(len(dataset), 0.001)
        
        try:
            # Create combinations of QI values for each record
            qi_combinations = dataset[quasi_identifiers].apply(
                lambda row: tuple(row.values), axis=1
            )
            
            # Count frequency of each combination
            combination_counts = qi_combinations.value_counts()
            
            # Risk = 1 / frequency for each record
            individual_risks = qi_combinations.map(lambda x: 1.0 / combination_counts[x])
            
            return individual_risks.values
            
        except Exception as e:
            self.logger.error(f"Individual risk calculation failed: {str(e)}")
            return np.full(len(dataset), 0.001)
    
    def calculate_prosecutor_risk(self, individual_risks: np.ndarray) -> float:
        """
        Calculate prosecutor risk (maximum individual risk)
        
        Args:
            individual_risks: Array of individual risk scores
            
        Returns:
            Maximum individual risk
        """
        return float(np.max(individual_risks)) if len(individual_risks) > 0 else 0.0
    
    def calculate_journalist_risk(self, individual_risks: np.ndarray) -> float:
        """
        Calculate journalist risk (average individual risk)
        
        Args:
            individual_risks: Array of individual risk scores
            
        Returns:
            Average individual risk
        """
        return float(np.mean(individual_risks)) if len(individual_risks) > 0 else 0.0
    
    def calculate_marketer_risk(self, individual_risks: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calculate marketer risk (proportion of records above threshold)
        
        Args:
            individual_risks: Array of individual risk scores
            threshold: Risk threshold for classification
            
        Returns:
            Proportion of high-risk records
        """
        if len(individual_risks) == 0:
            return 0.0
        
        high_risk_count = np.sum(individual_risks > threshold)
        return float(high_risk_count / len(individual_risks))
    
    def calculate_global_risk(self, individual_risks: np.ndarray) -> float:
        """
        Calculate global re-identification risk
        
        Args:
            individual_risks: Array of individual risk scores
            
        Returns:
            Global risk score
        """
        # Global risk as weighted average of individual risks
        if len(individual_risks) == 0:
            return 0.0
        
        # Weight by risk level to emphasize high-risk records
        weights = individual_risks / np.sum(individual_risks) if np.sum(individual_risks) > 0 else np.ones(len(individual_risks))
        global_risk = np.average(individual_risks, weights=weights)
        
        return float(min(global_risk, 1.0))  # Cap at 1.0
    
    def calculate_uniqueness_metrics(self, dataset: pd.DataFrame, 
                                   quasi_identifiers: List[str]) -> Dict[str, float]:
        """
        Calculate various uniqueness metrics
        
        Args:
            dataset: Input dataset
            quasi_identifiers: List of quasi-identifier columns
            
        Returns:
            Dictionary containing uniqueness metrics
        """
        if not quasi_identifiers or not all(col in dataset.columns for col in quasi_identifiers):
            return {
                'sample_uniqueness': 0.0,
                'population_uniqueness': 0.0,
                'equivalence_classes': 1
            }
        
        try:
            # Group by quasi-identifiers
            grouped = dataset.groupby(quasi_identifiers, dropna=False).size()
            
            # Sample uniqueness (proportion of singletons)
            singletons = (grouped == 1).sum()
            sample_uniqueness = singletons / len(dataset)
            
            # Estimate population uniqueness using Pitman estimator
            f1 = singletons  # Number of singletons
            f2 = (grouped == 2).sum()  # Number of doubletons
            
            if f2 > 0:
                population_uniqueness = f1**2 / (2 * f2 * len(dataset))
            else:
                population_uniqueness = sample_uniqueness
            
            return {
                'sample_uniqueness': float(sample_uniqueness),
                'population_uniqueness': float(population_uniqueness),
                'equivalence_classes': len(grouped),
                'singleton_count': int(singletons),
                'doubleton_count': int(f2)
            }
            
        except Exception as e:
            self.logger.error(f"Uniqueness metrics calculation failed: {str(e)}")
            return {
                'sample_uniqueness': 0.0,
                'population_uniqueness': 0.0,
                'equivalence_classes': 1
            }
    
    def calculate_information_loss_metrics(self, original: pd.DataFrame, 
                                         anonymized: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate information loss metrics after anonymization
        
        Args:
            original: Original dataset
            anonymized: Anonymized dataset
            
        Returns:
            Dictionary containing information loss metrics
        """
        try:
            metrics = {}
            
            # Ensure same columns
            common_columns = list(set(original.columns) & set(anonymized.columns))
            if not common_columns:
                return {'total_information_loss': 1.0}
            
            # Calculate metrics for numeric columns
            numeric_columns = original[common_columns].select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                # Mean preservation
                mean_preservation = []
                for col in numeric_columns:
                    orig_mean = original[col].mean()
                    anon_mean = anonymized[col].mean()
                    
                    if not np.isnan(orig_mean) and orig_mean != 0:
                        preservation = 1 - abs(orig_mean - anon_mean) / abs(orig_mean)
                        mean_preservation.append(max(0, preservation))
                
                metrics['mean_preservation'] = np.mean(mean_preservation) if mean_preservation else 0.0
                
                # Variance preservation
                var_preservation = []
                for col in numeric_columns:
                    orig_var = original[col].var()
                    anon_var = anonymized[col].var()
                    
                    if not np.isnan(orig_var) and orig_var != 0:
                        preservation = 1 - abs(orig_var - anon_var) / orig_var
                        var_preservation.append(max(0, preservation))
                
                metrics['variance_preservation'] = np.mean(var_preservation) if var_preservation else 0.0
                
                # Correlation preservation
                if len(numeric_columns) > 1:
                    try:
                        orig_corr = original[numeric_columns].corr()
                        anon_corr = anonymized[numeric_columns].corr()
                        
                        # Flatten correlation matrices and calculate correlation
                        orig_flat = orig_corr.values.flatten()
                        anon_flat = anon_corr.values.flatten()
                        
                        # Remove NaN and diagonal elements
                        mask = ~(np.isnan(orig_flat) | np.isnan(anon_flat))
                        if np.sum(mask) > 1:
                            corr_preservation = np.corrcoef(orig_flat[mask], anon_flat[mask])[0, 1]
                            metrics['correlation_preservation'] = max(0, corr_preservation) if not np.isnan(corr_preservation) else 0.0
                        else:
                            metrics['correlation_preservation'] = 0.0
                    except:
                        metrics['correlation_preservation'] = 0.0
                else:
                    metrics['correlation_preservation'] = 1.0
            
            # Calculate metrics for categorical columns
            categorical_columns = original[common_columns].select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_columns) > 0:
                # Distribution preservation
                dist_preservation = []
                for col in categorical_columns:
                    try:
                        orig_dist = original[col].value_counts(normalize=True)
                        anon_dist = anonymized[col].value_counts(normalize=True)
                        
                        # Calculate overlap
                        common_values = set(orig_dist.index) & set(anon_dist.index)
                        if len(common_values) > 0:
                            overlap = sum(min(orig_dist.get(val, 0), anon_dist.get(val, 0)) 
                                        for val in common_values)
                            dist_preservation.append(overlap)
                    except:
                        continue
                
                metrics['distribution_preservation'] = np.mean(dist_preservation) if dist_preservation else 0.0
            
            # Overall information loss
            preservation_scores = [v for v in metrics.values() if isinstance(v, (int, float))]
            overall_preservation = np.mean(preservation_scores) if preservation_scores else 0.0
            metrics['total_information_loss'] = 1.0 - overall_preservation
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Information loss calculation failed: {str(e)}")
            return {'total_information_loss': 0.5}  # Default moderate loss
    
    def calculate_utility_metrics(self, original: pd.DataFrame, 
                                anonymized: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate utility preservation metrics
        
        Args:
            original: Original dataset
            anonymized: Anonymized dataset
            
        Returns:
            Dictionary containing utility metrics
        """
        try:
            metrics = {}
            
            # Query accuracy metrics
            query_metrics = self._calculate_query_accuracy(original, anonymized)
            metrics.update(query_metrics)
            
            # Statistical utility metrics
            stat_metrics = self._calculate_statistical_utility(original, anonymized)
            metrics.update(stat_metrics)
            
            # Machine learning utility metrics
            ml_metrics = self._calculate_ml_utility(original, anonymized)
            metrics.update(ml_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Utility metrics calculation failed: {str(e)}")
            return {}
    
    def _calculate_query_accuracy(self, original: pd.DataFrame, 
                                anonymized: pd.DataFrame) -> Dict[str, float]:
        """Calculate query accuracy metrics"""
        
        metrics = {}
        common_columns = list(set(original.columns) & set(anonymized.columns))
        numeric_columns = original[common_columns].select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return {'query_accuracy': 1.0}
        
        try:
            # Count query accuracy
            count_accuracy = 1.0 - abs(len(original) - len(anonymized)) / len(original)
            metrics['count_query_accuracy'] = max(0.0, count_accuracy)
            
            # Sum query accuracy
            sum_accuracies = []
            for col in numeric_columns:
                orig_sum = original[col].sum()
                anon_sum = anonymized[col].sum()
                
                if not np.isnan(orig_sum) and orig_sum != 0:
                    accuracy = 1.0 - abs(orig_sum - anon_sum) / abs(orig_sum)
                    sum_accuracies.append(max(0.0, accuracy))
            
            metrics['sum_query_accuracy'] = np.mean(sum_accuracies) if sum_accuracies else 1.0
            
            # Mean query accuracy
            mean_accuracies = []
            for col in numeric_columns:
                orig_mean = original[col].mean()
                anon_mean = anonymized[col].mean()
                
                if not np.isnan(orig_mean) and orig_mean != 0:
                    accuracy = 1.0 - abs(orig_mean - anon_mean) / abs(orig_mean)
                    mean_accuracies.append(max(0.0, accuracy))
            
            metrics['mean_query_accuracy'] = np.mean(mean_accuracies) if mean_accuracies else 1.0
            
            # Overall query accuracy
            all_accuracies = [metrics['count_query_accuracy'], 
                            metrics['sum_query_accuracy'], 
                            metrics['mean_query_accuracy']]
            metrics['query_accuracy'] = np.mean(all_accuracies)
            
        except Exception as e:
            self.logger.warning(f"Query accuracy calculation failed: {str(e)}")
            metrics['query_accuracy'] = 0.5
        
        return metrics
    
    def _calculate_statistical_utility(self, original: pd.DataFrame, 
                                     anonymized: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical utility metrics"""
        
        metrics = {}
        common_columns = list(set(original.columns) & set(anonymized.columns))
        numeric_columns = original[common_columns].select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return {'statistical_utility': 1.0}
        
        try:
            # Distribution similarity using KL divergence
            kl_divergences = []
            for col in numeric_columns:
                try:
                    # Discretize continuous variables
                    orig_binned = pd.cut(original[col].dropna(), bins=10, duplicates='drop')
                    anon_binned = pd.cut(anonymized[col].dropna(), bins=10, duplicates='drop')
                    
                    orig_dist = orig_binned.value_counts(normalize=True)
                    anon_dist = anon_binned.value_counts(normalize=True)
                    
                    # Align distributions
                    all_bins = set(orig_dist.index) | set(anon_dist.index)
                    orig_aligned = [orig_dist.get(bin, 1e-8) for bin in all_bins]
                    anon_aligned = [anon_dist.get(bin, 1e-8) for bin in all_bins]
                    
                    # Calculate KL divergence
                    kl_div = stats.entropy(orig_aligned, anon_aligned)
                    if not (np.isnan(kl_div) or np.isinf(kl_div)):
                        kl_divergences.append(kl_div)
                except:
                    continue
            
            if kl_divergences:
                avg_kl = np.mean(kl_divergences)
                metrics['kl_divergence'] = avg_kl
                metrics['distribution_similarity'] = np.exp(-avg_kl)  # Convert to similarity
            else:
                metrics['kl_divergence'] = 0.0
                metrics['distribution_similarity'] = 1.0
            
            # Moment preservation
            moment_preservation = self._calculate_moment_preservation(original, anonymized, numeric_columns)
            metrics.update(moment_preservation)
            
            # Overall statistical utility
            stat_scores = [metrics.get('distribution_similarity', 1.0),
                          metrics.get('moment_preservation', 1.0)]
            metrics['statistical_utility'] = np.mean(stat_scores)
            
        except Exception as e:
            self.logger.warning(f"Statistical utility calculation failed: {str(e)}")
            metrics['statistical_utility'] = 0.5
        
        return metrics
    
    def _calculate_ml_utility(self, original: pd.DataFrame, 
                            anonymized: pd.DataFrame) -> Dict[str, float]:
        """Calculate machine learning utility metrics"""
        
        metrics = {}
        common_columns = list(set(original.columns) & set(anonymized.columns))
        numeric_columns = original[common_columns].select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return {'ml_utility': 1.0}
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            
            # Use first column as target, rest as features
            target_col = numeric_columns[0]
            feature_cols = list(numeric_columns[1:])
            
            # Prepare original data
            orig_X = original[feature_cols].dropna()
            orig_y = original.loc[orig_X.index, target_col]
            
            # Prepare anonymized data
            anon_X = anonymized[feature_cols].dropna()
            anon_y = anonymized.loc[anon_X.index, target_col]
            
            if len(orig_X) < 10 or len(anon_X) < 10:
                metrics['ml_utility'] = 1.0
                return metrics
            
            # Train model on original data
            X_train, X_test, y_train, y_test = train_test_split(
                orig_X, orig_y, test_size=0.3, random_state=42
            )
            
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Test on original data
            orig_score = model.score(X_test, y_test)
            
            # Test on anonymized data
            min_len = min(len(anon_X), len(X_test))
            anon_X_sample = anon_X.iloc[:min_len]
            anon_y_sample = anon_y.iloc[:min_len]
            
            anon_score = model.score(anon_X_sample, anon_y_sample)
            
            # Calculate relative performance
            if orig_score > 0:
                ml_utility = anon_score / orig_score
            else:
                ml_utility = 1.0 if anon_score >= orig_score else 0.0
            
            metrics['ml_utility'] = max(0.0, min(1.0, ml_utility))
            
        except Exception as e:
            self.logger.warning(f"ML utility calculation failed: {str(e)}")
            metrics['ml_utility'] = 0.5
        
        return metrics
    
    def _calculate_moment_preservation(self, original: pd.DataFrame, anonymized: pd.DataFrame, 
                                     numeric_columns: List[str]) -> Dict[str, float]:
        """Calculate moment preservation metrics"""
        
        try:
            preservations = []
            
            for col in numeric_columns:
                col_preservations = []
                
                # First moment (mean)
                orig_mean = original[col].mean()
                anon_mean = anonymized[col].mean()
                if not np.isnan(orig_mean) and orig_mean != 0:
                    mean_pres = 1 - abs(orig_mean - anon_mean) / abs(orig_mean)
                    col_preservations.append(max(0, mean_pres))
                
                # Second moment (variance)
                orig_var = original[col].var()
                anon_var = anonymized[col].var()
                if not np.isnan(orig_var) and orig_var != 0:
                    var_pres = 1 - abs(orig_var - anon_var) / orig_var
                    col_preservations.append(max(0, var_pres))
                
                # Third moment (skewness)
                try:
                    orig_skew = original[col].skew()
                    anon_skew = anonymized[col].skew()
                    if not np.isnan(orig_skew) and orig_skew != 0:
                        skew_pres = 1 - abs(orig_skew - anon_skew) / abs(orig_skew)
                        col_preservations.append(max(0, skew_pres))
                except:
                    pass
                
                if col_preservations:
                    preservations.append(np.mean(col_preservations))
            
            return {
                'moment_preservation': np.mean(preservations) if preservations else 1.0
            }
            
        except Exception:
            return {'moment_preservation': 0.5}
