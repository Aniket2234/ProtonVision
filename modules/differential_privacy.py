"""
Differential Privacy Module for SafeData Pipeline
Implements various DP mechanisms including Laplace, Gaussian, and Exponential
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from scipy import stats
import math

class DifferentialPrivacy:
    """Differential Privacy implementation with multiple mechanisms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.privacy_budget_tracker = {}
        
    def apply_differential_privacy(self, dataset: pd.DataFrame, epsilon: float = 1.0,
                                 delta: float = 1e-5, mechanism: str = 'Laplace',
                                 columns: Optional[List[str]] = None,
                                 sensitivity: Optional[float] = None) -> Dict[str, Any]:
        """
        Apply differential privacy to dataset
        
        Args:
            dataset: Input dataset
            epsilon: Privacy budget parameter
            delta: Delta parameter for (ε,δ)-DP
            mechanism: DP mechanism ('Laplace', 'Gaussian', 'Exponential', 'Random Response')
            columns: Columns to apply DP to (default: numeric columns)
            sensitivity: Global sensitivity (auto-calculated if None)
            
        Returns:
            Dictionary with DP-enhanced dataset and metrics
        """
        self.logger.info(f"Applying {mechanism} mechanism with ε={epsilon}, δ={delta}")
        
        try:
            if epsilon <= 0:
                raise ValueError("Epsilon must be positive")
            
            dp_dataset = dataset.copy()
            
            if columns is None:
                columns = list(dataset.select_dtypes(include=[np.number]).columns)
            
            if not columns:
                self.logger.warning("No numeric columns found for DP application")
                return {
                    'dataset': dataset,
                    'noise_scale': 0,
                    'privacy_spent': epsilon,
                    'mechanism_used': mechanism
                }
            
            # Calculate global sensitivity if not provided
            if sensitivity is None:
                sensitivity = self._calculate_global_sensitivity(dataset, columns)
            
            noise_scale = 0
            mechanism_details = {}
            
            # Apply the specified mechanism
            if mechanism == 'Laplace':
                result = self._apply_laplace_mechanism(dp_dataset, columns, epsilon, sensitivity)
                noise_scale = result['noise_scale']
                mechanism_details = result['details']
                
            elif mechanism == 'Gaussian':
                result = self._apply_gaussian_mechanism(dp_dataset, columns, epsilon, delta, sensitivity)
                noise_scale = result['noise_scale']
                mechanism_details = result['details']
                
            elif mechanism == 'Exponential':
                result = self._apply_exponential_mechanism(dp_dataset, columns, epsilon)
                noise_scale = result['noise_scale']
                mechanism_details = result['details']
                
            elif mechanism == 'Random Response':
                result = self._apply_random_response(dp_dataset, columns, epsilon)
                noise_scale = result['noise_scale']
                mechanism_details = result['details']
                
            else:
                raise ValueError(f"Unsupported mechanism: {mechanism}")
            
            # Track privacy budget
            self._update_privacy_budget(epsilon)
            
            # Calculate privacy metrics
            privacy_metrics = self._calculate_privacy_metrics(
                dataset, dp_dataset, epsilon, delta, mechanism
            )
            
            result = {
                'dataset': dp_dataset,
                'noise_scale': noise_scale,
                'privacy_spent': epsilon,
                'delta_used': delta,
                'mechanism_used': mechanism,
                'sensitivity': sensitivity,
                'columns_processed': columns,
                'mechanism_details': mechanism_details,
                'privacy_metrics': privacy_metrics,
                'total_budget_used': sum(self.privacy_budget_tracker.values())
            }
            
            self.logger.info(f"Differential privacy applied successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Differential privacy application failed: {str(e)}")
            raise
    
    def _calculate_global_sensitivity(self, dataset: pd.DataFrame, columns: List[str]) -> float:
        """Calculate global sensitivity for the dataset"""
        
        try:
            max_sensitivity = 0
            
            for column in columns:
                if pd.api.types.is_numeric_dtype(dataset[column]):
                    # For numeric columns, sensitivity is the maximum possible change
                    col_data = dataset[column].dropna()
                    if len(col_data) > 1:
                        # Use range as a conservative estimate
                        col_range = col_data.max() - col_data.min()
                        max_sensitivity = max(max_sensitivity, col_range)
            
            # Use a reasonable default if no sensitivity calculated
            return max(max_sensitivity, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Sensitivity calculation failed: {str(e)}")
            return 1.0
    
    def _apply_laplace_mechanism(self, dataset: pd.DataFrame, columns: List[str],
                                epsilon: float, sensitivity: float) -> Dict[str, Any]:
        """Apply Laplace mechanism for differential privacy"""
        
        try:
            noise_scale = sensitivity / epsilon
            details = {'noise_parameters': {}}
            
            for column in columns:
                if pd.api.types.is_numeric_dtype(dataset[column]):
                    # Generate Laplace noise
                    noise = np.random.laplace(0, noise_scale, len(dataset))
                    
                    # Add noise to the column
                    dataset[column] = dataset[column] + noise
                    
                    details['noise_parameters'][column] = {
                        'scale': noise_scale,
                        'noise_std': np.std(noise),
                        'noise_mean': np.mean(noise)
                    }
            
            return {
                'noise_scale': noise_scale,
                'details': details
            }
            
        except Exception as e:
            self.logger.error(f"Laplace mechanism failed: {str(e)}")
            raise
    
    def _apply_gaussian_mechanism(self, dataset: pd.DataFrame, columns: List[str],
                                 epsilon: float, delta: float, sensitivity: float) -> Dict[str, Any]:
        """Apply Gaussian mechanism for differential privacy"""
        
        try:
            # Calculate noise scale for Gaussian mechanism
            if delta <= 0 or delta >= 1:
                raise ValueError("Delta must be in (0, 1)")
            
            # For (ε,δ)-DP with Gaussian noise
            c = math.sqrt(2 * math.log(1.25 / delta))
            noise_scale = c * sensitivity / epsilon
            
            details = {'noise_parameters': {}}
            
            for column in columns:
                if pd.api.types.is_numeric_dtype(dataset[column]):
                    # Generate Gaussian noise
                    noise = np.random.normal(0, noise_scale, len(dataset))
                    
                    # Add noise to the column
                    dataset[column] = dataset[column] + noise
                    
                    details['noise_parameters'][column] = {
                        'scale': noise_scale,
                        'noise_std': np.std(noise),
                        'noise_mean': np.mean(noise)
                    }
            
            return {
                'noise_scale': noise_scale,
                'details': details
            }
            
        except Exception as e:
            self.logger.error(f"Gaussian mechanism failed: {str(e)}")
            raise
    
    def _apply_exponential_mechanism(self, dataset: pd.DataFrame, columns: List[str],
                                   epsilon: float) -> Dict[str, Any]:
        """Apply Exponential mechanism for differential privacy"""
        
        try:
            details = {'perturbation_info': {}}
            noise_scale = 1.0 / epsilon  # Simplified noise scale for tracking
            
            for column in columns:
                if pd.api.types.is_numeric_dtype(dataset[column]):
                    # For exponential mechanism, we'll use it for discrete selections
                    # Here we implement a simplified version that adds structured noise
                    
                    col_data = dataset[column].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    # Create utility function based on distance from median
                    median_val = col_data.median()
                    
                    # Calculate utilities (higher for values closer to median)
                    utilities = -np.abs(dataset[column] - median_val)
                    
                    # Apply exponential mechanism probabilities
                    exp_utilities = np.exp(epsilon * utilities / 2)
                    probabilities = exp_utilities / np.sum(exp_utilities)
                    
                    # Sample based on probabilities (simplified implementation)
                    # In practice, this would involve more sophisticated selection
                    perturbation = np.random.exponential(1/epsilon, len(dataset))
                    dataset[column] = dataset[column] + perturbation - np.mean(perturbation)
                    
                    details['perturbation_info'][column] = {
                        'median_anchor': median_val,
                        'perturbation_scale': 1/epsilon,
                        'utility_range': [float(utilities.min()), float(utilities.max())]
                    }
            
            return {
                'noise_scale': noise_scale,
                'details': details
            }
            
        except Exception as e:
            self.logger.error(f"Exponential mechanism failed: {str(e)}")
            raise
    
    def _apply_random_response(self, dataset: pd.DataFrame, columns: List[str],
                              epsilon: float) -> Dict[str, Any]:
        """Apply Random Response mechanism for categorical data"""
        
        try:
            details = {'response_parameters': {}}
            noise_scale = 1.0 / epsilon
            
            # Random response works best with categorical data
            categorical_columns = dataset[columns].select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_columns) == 0:
                # If no categorical columns, apply to numeric by discretizing
                for column in columns:
                    if pd.api.types.is_numeric_dtype(dataset[column]):
                        # Discretize numeric column first
                        col_data = dataset[column].dropna()
                        if len(col_data) > 0:
                            # Create bins
                            n_bins = min(10, len(col_data.unique()))
                            discretized = pd.cut(col_data, bins=n_bins, duplicates='drop')
                            
                            # Apply random response to discretized data
                            p = math.exp(epsilon) / (math.exp(epsilon) + 1)
                            
                            # Flip responses with probability (1-p)
                            flip_mask = np.random.random(len(dataset)) > p
                            
                            # For flipped responses, choose random category
                            if len(discretized.categories) > 1:
                                random_categories = np.random.choice(
                                    discretized.categories, 
                                    size=flip_mask.sum()
                                )
                                # This is a simplified implementation
                                dataset.loc[dataset.index[flip_mask], column] = np.random.normal(
                                    dataset[column].mean(), 
                                    dataset[column].std(), 
                                    flip_mask.sum()
                                )
                            
                            details['response_parameters'][column] = {
                                'truth_probability': p,
                                'flip_probability': 1 - p,
                                'bins_used': n_bins
                            }
            else:
                # Apply to categorical columns
                for column in categorical_columns:
                    col_data = dataset[column].dropna()
                    if len(col_data) > 0:
                        unique_values = col_data.unique()
                        if len(unique_values) > 1:
                            # Calculate truth probability
                            p = math.exp(epsilon) / (math.exp(epsilon) + len(unique_values) - 1)
                            
                            # Apply random response
                            flip_mask = np.random.random(len(dataset)) > p
                            
                            # For flipped responses, choose random value
                            random_values = np.random.choice(
                                unique_values, 
                                size=flip_mask.sum()
                            )
                            
                            flipped_indices = dataset.index[flip_mask]
                            for i, idx in enumerate(flipped_indices):
                                dataset.loc[idx, column] = random_values[i]
                            
                            details['response_parameters'][column] = {
                                'truth_probability': p,
                                'num_categories': len(unique_values),
                                'flipped_responses': flip_mask.sum()
                            }
            
            return {
                'noise_scale': noise_scale,
                'details': details
            }
            
        except Exception as e:
            self.logger.error(f"Random response failed: {str(e)}")
            raise
    
    def _update_privacy_budget(self, epsilon: float):
        """Update privacy budget tracker"""
        
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        self.privacy_budget_tracker[timestamp] = epsilon
    
    def _calculate_privacy_metrics(self, original: pd.DataFrame, private: pd.DataFrame,
                                  epsilon: float, delta: float, mechanism: str) -> Dict[str, Any]:
        """Calculate privacy-related metrics"""
        
        try:
            metrics = {
                'epsilon_used': epsilon,
                'delta_used': delta,
                'mechanism': mechanism,
                'privacy_level': self._classify_privacy_level(epsilon),
                'theoretical_guarantee': f"({epsilon}, {delta})-differential privacy"
            }
            
            # Calculate empirical privacy metrics
            common_columns = list(set(original.columns) & set(private.columns))
            numeric_columns = original[common_columns].select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                # Calculate signal-to-noise ratio
                signal_variance = np.mean([original[col].var() for col in numeric_columns])
                noise_variance = np.mean([
                    np.var(private[col] - original[col]) 
                    for col in numeric_columns
                ])
                
                snr = signal_variance / noise_variance if noise_variance > 0 else float('inf')
                
                metrics.update({
                    'signal_to_noise_ratio': float(snr),
                    'relative_noise_level': float(noise_variance / signal_variance) if signal_variance > 0 else 0,
                    'privacy_cost': float(1.0 / epsilon)  # Higher epsilon = lower cost
                })
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Privacy metrics calculation failed: {str(e)}")
            return {
                'epsilon_used': epsilon,
                'delta_used': delta,
                'mechanism': mechanism
            }
    
    def _classify_privacy_level(self, epsilon: float) -> str:
        """Classify privacy level based on epsilon value"""
        
        if epsilon <= 0.1:
            return "Very High Privacy"
        elif epsilon <= 1.0:
            return "High Privacy"
        elif epsilon <= 5.0:
            return "Medium Privacy"
        elif epsilon <= 10.0:
            return "Low Privacy"
        else:
            return "Very Low Privacy"
    
    def compose_privacy_budgets(self, epsilons: List[float], deltas: List[float],
                               composition_type: str = 'basic') -> Tuple[float, float]:
        """
        Compose privacy budgets for multiple queries
        
        Args:
            epsilons: List of epsilon values
            deltas: List of delta values
            composition_type: Type of composition ('basic', 'advanced')
            
        Returns:
            Tuple of (total_epsilon, total_delta)
        """
        
        if composition_type == 'basic':
            # Basic composition
            total_epsilon = sum(epsilons)
            total_delta = sum(deltas)
            
        elif composition_type == 'advanced':
            # Advanced composition (simplified)
            k = len(epsilons)
            if k > 0:
                max_epsilon = max(epsilons)
                total_epsilon = max_epsilon * math.sqrt(2 * k * math.log(1/min(deltas)))
                total_delta = k * max(deltas)
            else:
                total_epsilon = 0
                total_delta = 0
        else:
            raise ValueError(f"Unknown composition type: {composition_type}")
        
        return total_epsilon, total_delta
    
    def get_privacy_budget_status(self, total_budget: float = 1.0) -> Dict[str, Any]:
        """Get current privacy budget status"""
        
        used_budget = sum(self.privacy_budget_tracker.values())
        remaining_budget = max(0, total_budget - used_budget)
        
        return {
            'total_budget': total_budget,
            'used_budget': used_budget,
            'remaining_budget': remaining_budget,
            'usage_percentage': (used_budget / total_budget) * 100,
            'queries_made': len(self.privacy_budget_tracker),
            'budget_exhausted': remaining_budget <= 0
        }
    
    def reset_privacy_budget(self):
        """Reset privacy budget tracker"""
        
        self.privacy_budget_tracker.clear()
        self.logger.info("Privacy budget tracker reset")
    
    def estimate_noise_for_accuracy(self, target_accuracy: float, sensitivity: float,
                                   confidence: float = 0.95) -> float:
        """
        Estimate required epsilon for target accuracy
        
        Args:
            target_accuracy: Desired accuracy level
            sensitivity: Global sensitivity
            confidence: Confidence level
            
        Returns:
            Recommended epsilon value
        """
        
        try:
            # For Laplace mechanism, the noise scale is sensitivity/epsilon
            # We want the noise to be small enough to maintain accuracy
            
            # Calculate noise tolerance based on target accuracy
            noise_tolerance = target_accuracy * sensitivity
            
            # Calculate required epsilon
            required_epsilon = sensitivity / noise_tolerance
            
            # Adjust for confidence level
            if confidence > 0.5:
                confidence_factor = stats.norm.ppf((1 + confidence) / 2)
                required_epsilon *= confidence_factor
            
            return max(0.1, required_epsilon)  # Minimum epsilon of 0.1
            
        except Exception as e:
            self.logger.warning(f"Epsilon estimation failed: {str(e)}")
            return 1.0  # Default value
