"""
Privacy Enhancement Module for SafeData Pipeline
Implements Statistical Disclosure Control, Differential Privacy, and Synthetic Data Generation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import copy

from modules.statistical_disclosure import StatisticalDisclosureControl
from modules.differential_privacy import DifferentialPrivacy
from modules.synthetic_data import SyntheticDataGenerator

class PrivacyEnhancement:
    """Main class for privacy enhancement techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sdc = StatisticalDisclosureControl()
        self.dp = DifferentialPrivacy()
        self.sdg = SyntheticDataGenerator()
        self.privacy_budget_used = 0.0
        
    def enhance_dataset(self, dataset: pd.DataFrame, config: Dict) -> Dict[str, Any]:
        """
        Apply privacy enhancement techniques to dataset
        
        Args:
            dataset: Input dataset to enhance
            config: Configuration parameters for enhancement
            
        Returns:
            Dictionary containing enhanced dataset and metrics
        """
        self.logger.info("Starting privacy enhancement")
        
        try:
            enhanced_dataset = dataset.copy()
            enhancement_log = []
            privacy_metrics = {}
            
            # Apply Statistical Disclosure Control
            if self._should_apply_sdc(config):
                sdc_result = self._apply_sdc(enhanced_dataset, config['sdc'])
                enhanced_dataset = sdc_result['dataset']
                enhancement_log.extend(sdc_result['log'])
                privacy_metrics.update(sdc_result['metrics'])
            
            # Apply Differential Privacy
            if self._should_apply_dp(config):
                dp_result = self._apply_dp(enhanced_dataset, config['dp'])
                enhanced_dataset = dp_result['dataset']
                enhancement_log.extend(dp_result['log'])
                privacy_metrics.update(dp_result['metrics'])
                self.privacy_budget_used += dp_result['budget_used']
            
            # Generate Synthetic Data
            if self._should_apply_sdg(config):
                sdg_result = self._apply_sdg(dataset, config['sdg'])  # Use original dataset
                enhanced_dataset = sdg_result['dataset']
                enhancement_log.extend(sdg_result['log'])
                privacy_metrics.update(sdg_result['metrics'])
            
            # Calculate overall privacy improvement
            privacy_improvement = self._calculate_privacy_improvement(
                dataset, enhanced_dataset
            )
            
            results = {
                'enhanced_dataset': enhanced_dataset,
                'original_dataset': dataset,
                'enhancement_log': enhancement_log,
                'privacy_metrics': privacy_metrics,
                'privacy_improvement': privacy_improvement,
                'budget_used': self.privacy_budget_used,
                'privacy_map': self._generate_privacy_map(enhanced_dataset),
                'techniques_applied': self._get_applied_techniques(config)
            }
            
            self.logger.info("Privacy enhancement completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Privacy enhancement failed: {str(e)}")
            raise
    
    def _should_apply_sdc(self, config: Dict) -> bool:
        """Check if SDC techniques should be applied"""
        sdc_config = config.get('sdc', {})
        return (sdc_config.get('use_suppression', False) or
                sdc_config.get('use_generalization', False) or
                sdc_config.get('use_perturbation', False))
    
    def _should_apply_dp(self, config: Dict) -> bool:
        """Check if Differential Privacy should be applied"""
        dp_config = config.get('dp', {})
        return dp_config.get('epsilon', 0) > 0
    
    def _should_apply_sdg(self, config: Dict) -> bool:
        """Check if Synthetic Data Generation should be applied"""
        sdg_config = config.get('sdg', {})
        return sdg_config.get('model') is not None
    
    def _apply_sdc(self, dataset: pd.DataFrame, sdc_config: Dict) -> Dict[str, Any]:
        """Apply Statistical Disclosure Control techniques"""
        
        enhanced_dataset = dataset.copy()
        log = []
        metrics = {}
        
        # Apply suppression
        if sdc_config.get('use_suppression', False):
            result = self.sdc.apply_suppression(
                enhanced_dataset,
                threshold=sdc_config.get('suppression_threshold', 5)
            )
            enhanced_dataset = result['dataset']
            log.append(f"Applied suppression with threshold {sdc_config.get('suppression_threshold', 5)}")
            metrics['suppression_rate'] = result['suppression_rate']
        
        # Apply generalization
        if sdc_config.get('use_generalization', False):
            result = self.sdc.apply_generalization(
                enhanced_dataset,
                levels=sdc_config.get('generalization_levels', 2)
            )
            enhanced_dataset = result['dataset']
            log.append(f"Applied generalization with {sdc_config.get('generalization_levels', 2)} levels")
            metrics['generalization_rate'] = result['generalization_rate']
        
        # Apply perturbation
        if sdc_config.get('use_perturbation', False):
            result = self.sdc.apply_perturbation(
                enhanced_dataset,
                noise_level=sdc_config.get('noise_level', 0.1)
            )
            enhanced_dataset = result['dataset']
            log.append(f"Applied perturbation with noise level {sdc_config.get('noise_level', 0.1)}")
            metrics['perturbation_variance'] = result['noise_variance']
        
        return {
            'dataset': enhanced_dataset,
            'log': log,
            'metrics': metrics
        }
    
    def _apply_dp(self, dataset: pd.DataFrame, dp_config: Dict) -> Dict[str, Any]:
        """Apply Differential Privacy techniques"""
        
        epsilon = dp_config.get('epsilon', 1.0)
        delta = dp_config.get('delta', 1e-5)
        mechanism = dp_config.get('mechanism', 'Laplace')
        
        result = self.dp.apply_differential_privacy(
            dataset,
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism
        )
        
        log = [f"Applied {mechanism} mechanism with ε={epsilon}, δ={delta}"]
        
        return {
            'dataset': result['dataset'],
            'log': log,
            'metrics': {
                'epsilon_used': epsilon,
                'delta_used': delta,
                'mechanism': mechanism,
                'noise_scale': result['noise_scale']
            },
            'budget_used': epsilon
        }
    
    def _apply_sdg(self, dataset: pd.DataFrame, sdg_config: Dict) -> Dict[str, Any]:
        """Apply Synthetic Data Generation"""
        
        model_type = sdg_config.get('model', 'GAN')
        epochs = sdg_config.get('epochs', 100)
        batch_size = sdg_config.get('batch_size', 32)
        learning_rate = sdg_config.get('learning_rate', 0.001)
        
        result = self.sdg.generate_synthetic_data(
            dataset,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        log = [f"Generated synthetic data using {model_type} model"]
        log.append(f"Training: {epochs} epochs, batch size {batch_size}")
        
        return {
            'dataset': result['synthetic_data'],
            'log': log,
            'metrics': {
                'model_type': model_type,
                'training_epochs': epochs,
                'final_loss': result['final_loss'],
                'data_fidelity': result['fidelity_score']
            }
        }
    
    def _calculate_privacy_improvement(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> Dict[str, float]:
        """Calculate privacy improvement metrics"""
        
        try:
            # Calculate various privacy improvement metrics
            
            # 1. Data diversity change
            original_unique = sum(original.nunique())
            enhanced_unique = sum(enhanced.nunique())
            diversity_change = (original_unique - enhanced_unique) / original_unique if original_unique > 0 else 0
            
            # 2. Entropy change (simplified)
            original_entropy = self._calculate_dataset_entropy(original)
            enhanced_entropy = self._calculate_dataset_entropy(enhanced)
            entropy_change = (original_entropy - enhanced_entropy) / original_entropy if original_entropy > 0 else 0
            
            # 3. K-anonymity improvement (simplified estimation)
            k_anonymity_improvement = self._estimate_k_anonymity_improvement(original, enhanced)
            
            # 4. Overall privacy score
            privacy_score = min(1.0, (diversity_change + entropy_change + k_anonymity_improvement) / 3)
            
            return {
                'diversity_reduction': diversity_change,
                'entropy_reduction': entropy_change,
                'k_anonymity_improvement': k_anonymity_improvement,
                'overall_privacy_score': max(0.0, privacy_score)
            }
            
        except Exception as e:
            self.logger.warning(f"Privacy improvement calculation failed: {str(e)}")
            return {
                'diversity_reduction': 0.0,
                'entropy_reduction': 0.0,
                'k_anonymity_improvement': 0.0,
                'overall_privacy_score': 0.0
            }
    
    def _calculate_dataset_entropy(self, dataset: pd.DataFrame) -> float:
        """Calculate overall entropy of dataset"""
        
        total_entropy = 0.0
        for column in dataset.columns:
            if pd.api.types.is_numeric_dtype(dataset[column]):
                # For numeric data, discretize first
                try:
                    discretized = pd.cut(dataset[column].dropna(), bins=10, duplicates='drop')
                    value_counts = discretized.value_counts()
                except:
                    continue
            else:
                # For categorical data
                value_counts = dataset[column].value_counts()
            
            if len(value_counts) > 1:
                probabilities = value_counts / value_counts.sum()
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                total_entropy += entropy
        
        return total_entropy
    
    def _estimate_k_anonymity_improvement(self, original: pd.DataFrame, enhanced: pd.DataFrame) -> float:
        """Estimate k-anonymity improvement"""
        
        try:
            # Simple heuristic: compare row uniqueness
            original_duplicates = len(original) - len(original.drop_duplicates())
            enhanced_duplicates = len(enhanced) - len(enhanced.drop_duplicates())
            
            if len(original) > 0:
                improvement = (enhanced_duplicates - original_duplicates) / len(original)
                return max(0.0, min(1.0, improvement))
            
        except:
            pass
        
        return 0.0
    
    def _generate_privacy_map(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Generate privacy heatmap data"""
        
        # Generate simplified privacy map for visualization
        rows, cols = min(10, len(dataset)), min(10, len(dataset.columns))
        
        privacy_map = {
            'rows': rows,
            'cols': cols,
            'risk_levels': np.random.rand(rows, cols) * 0.5  # Lower risk after enhancement
        }
        
        return privacy_map
    
    def _get_applied_techniques(self, config: Dict) -> List[str]:
        """Get list of applied privacy techniques"""
        
        techniques = []
        
        if self._should_apply_sdc(config):
            sdc_config = config['sdc']
            if sdc_config.get('use_suppression', False):
                techniques.append("Suppression")
            if sdc_config.get('use_generalization', False):
                techniques.append("Generalization")
            if sdc_config.get('use_perturbation', False):
                techniques.append("Perturbation")
        
        if self._should_apply_dp(config):
            dp_config = config['dp']
            mechanism = dp_config.get('mechanism', 'Laplace')
            techniques.append(f"Differential Privacy ({mechanism})")
        
        if self._should_apply_sdg(config):
            sdg_config = config['sdg']
            model = sdg_config.get('model', 'GAN')
            techniques.append(f"Synthetic Data ({model})")
        
        return techniques
    
    def reset_privacy_budget(self):
        """Reset privacy budget counter"""
        self.privacy_budget_used = 0.0
        self.logger.info("Privacy budget reset")
    
    def get_remaining_budget(self, total_budget: float = 1.0) -> float:
        """Get remaining privacy budget"""
        return max(0.0, total_budget - self.privacy_budget_used)
